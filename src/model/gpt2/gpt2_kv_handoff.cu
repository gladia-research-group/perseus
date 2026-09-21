#include "model/gpt2.h"
#include "model/gpt2/internal.h"
#include "attention.h"
#include "packing/cachemir/cachemir_attention.h"
#include "packing/cachemir/cachemir_attention_utils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>

PackedCtx extract_token_i_cachemir(Inference& inf, const PackedCtx& filling_ct, int i) {
    Ptx m = inf.encode_at_cached("xtok.pos" + std::to_string(i),
                                 cachemir::real_head_tok0_mask(inf, i), filling_ct);
    PackedCtx masked = inf.fhe->mult(filling_ct, m);
    PackedCtx rot    = inf.fhe->rotate(masked, cachemir::mha_rot(inf, i));
    return inf.pack(rot.ct, PackingKind::Cachemir);
}

void gpt2_kv_handoff_filling_to_cachemir(Inference& inf, int n_blocks, int m) {
    if (!is_cachemir(inf.packing))
        throw std::runtime_error("kv_handoff: inf.packing must be Cachemir on entry");

    const bool saved_suppress = inf.suppress_kv_periodic_bts;
    inf.suppress_kv_periodic_bts = false;
    const bool complex_decode = inf.complex;
    inf.complex = false;
    inf.clear_bootstrap_plan();
    const bool saved_strict = inf.strict_masks;
    inf.strict_masks = false;

    const int t = inf.slots / inf.size.hidDim;
    const std::string saved = inf.block_prefix;
    for (int b = 0; b < n_blocks; ++b) {
        inf.block_prefix = block_scope(b);
        reload_block_kv(inf);

        auto itk = inf.cache.find(inf.scoped("k"));
        auto itv = inf.cache.find(inf.scoped("v"));
        if (itk == inf.cache.end() || itk->second.empty() ||
            itv == inf.cache.end() || itv->second.empty())
            throw std::runtime_error("kv_handoff: missing prefilled K/V for block " +
                                     std::to_string(b));

        for (auto& g : itk->second) {
            g.packing.kind = PackingKind::Cachemir;
            const int klvl = inf.fhe->level_for_ct(g.ct);
            inf.fhe->inplace_im_cleanse(g);
            inf.fhe->inplace_mult(g, 0.25);
            inf.fhe->bootstrap(g.ct);
            inf.fhe->inplace_im_cleanse(g);
            inf.fhe->drop_to_level(g, cachemir::cache_read_level_k());
        }

        std::vector<PackedCtx> fV = itv->second;
        prepare_vcache(inf);
        {
            // Batched V repack: one masked mult per (group, feature) lane instead of a
            // per-token push.
            const int d_head      = inf.size.hidDim / inf.size.numHeads;
            const int d_head_real = inf.size.getRealDHead();
            const int H_real      = inf.size.getRealNumHeads();
            const int tH          = t * inf.size.numHeads;
            auto& vc = inf.cache[inf.scoped("v")];
            for (size_t g = 0; g < fV.size(); ++g) {
                const int Lg = std::min(t, m - static_cast<int>(g) * t);
                for (int f = 0; f < d_head_real; ++f) {
                    WithStep _wl(inf, "v_batch_mask_mult");
                    Ptx m_pt = inf.encode_at_cached(
                        "v.batch." + std::to_string(f) + ".L" + std::to_string(Lg), fV[g],
                        [&] {
                            std::vector<double> mask(inf.slots, 0.0);
                            for (int h = 0; h < H_real; ++h)
                                for (int i = 0; i < Lg; ++i)
                                    mask[f * tH + h * t + i] = 0.5;
                            return mask;
                        });
                    PackedCtx tmp = inf.fhe->mult(fV[g], m_pt);
                    const int lane = (static_cast<int>(g) - f + d_head) % d_head;
                    if (!vc[lane].ct) vc[lane] = std::move(tmp);
                    else              inf.fhe->inplace_add(vc[lane], tmp);
                }
            }
            for (auto& lane_ct : vc) {
                if (!lane_ct.ct) continue;
                inf.fhe->inplace_im_cleanse(lane_ct);
                inf.fhe->drop_to_level(lane_ct, cachemir::cache_read_level_v());
                lane_ct.packing.kind = PackingKind::Cachemir;
            }
            inf.v_count() = m;
        }

        if (complex_decode) {
            auto& ks = inf.cache[inf.scoped("k")];
            std::vector<PackedCtx> kpacked;
            size_t g = 0;
            for (; g + 1 < ks.size(); g += 2) {
                PackedCtx bucket = cachemir::pack_k_groups_complex(inf, ks[g], ks[g + 1]);
                inf.fhe->drop_to_level(bucket, cachemir::cache_read_level_k());
                kpacked.push_back(std::move(bucket));
            }
            auto& pend = inf.cache[inf.scoped("k.pend")];
            pend.clear();
            if (g < ks.size()) pend.push_back(std::move(ks[g]));
            ks = std::move(kpacked);

            auto& vs = inf.cache[inf.scoped("v")];
            const int d_head = inf.size.hidDim / inf.size.numHeads;
            std::vector<PackedCtx> vp(d_head / 2);
            for (int p = 0; p < d_head / 2; ++p) {
                const PackedCtx& re = vs[2 * p];
                const PackedCtx& im = vs[2 * p + 1];
                if (re.ct && im.ct)      vp[p] = inf.fhe->pair_pack(re, im);
                else if (re.ct)          vp[p] = inf.fhe->clone(re);
                else if (im.ct)          vp[p] = inf.fhe->mult_i(im);
            }
            vs = std::move(vp);
            inf.k_count() = m;   // decode pushes continue the complex grouping from token m
        }

        cudaDeviceSynchronize();
        offload_block_kv(inf);
    }
    inf.block_prefix = saved;
    inf.suppress_kv_periodic_bts = saved_suppress;
    inf.complex = complex_decode;
    inf.strict_masks = saved_strict;

    const size_t dropped = inf.clear_enc_cache();
    std::fprintf(stderr, "[handoff] dropped %zu prefill-level plaintext encodes\n", dropped);
    std::fflush(stderr);
}
