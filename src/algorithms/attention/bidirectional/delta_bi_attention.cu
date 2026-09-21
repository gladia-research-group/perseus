#include "packing/bidirectional/bi_attention.h"
#include "packing/cachemir_filling/cachemir_filling_attention.h"
#include "packing/cachemir_filling/cachemir_filling_delta_attention.h"
#include "inference.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bidirectional {

namespace {

PackedCtx fresh_masked(Inference& inf, const PackedCtx& x) {
    const int N  = inf.slots;
    const int t  = N / inf.size.hidDim;
    const int tH = t * inf.size.numHeads;
    const int d_head_real = inf.size.getRealDHead();
    const int H_real      = inf.size.getRealNumHeads();
    const int n_tok       = inf.n_tok;
    Ptx mask_pt = inf.encode_at_cached(
        "cf.kvpush.mask.q.nt" + std::to_string(n_tok), x,
        [&] {
            std::vector<double> mask(N, 0.0);
            for (int c = 0; c < d_head_real; ++c)
                for (int h = 0; h < H_real; ++h)
                    for (int i = 0; i < n_tok; ++i)
                        mask[c * tH + h * t + i] = 0.25;
            return mask;
        });
    PackedCtx masked = inf.fhe->mult(x, mask_pt);
    inf.fhe->inplace_im_cleanse(masked);
    inf.fhe->bootstrap(masked.ct);
    inf.fhe->inplace_im_cleanse(masked);
    return masked;
}

}  // namespace

std::vector<PackedCtx> delta_bi_attention(Inference& inf,
                                          std::vector<PackedCtx> qs,
                                          std::vector<PackedCtx> ks,
                                          std::vector<PackedCtx> vs,
                                          const std::vector<int>& ns) {
    const size_t C = qs.size();
    if (ks.size() != C || vs.size() != C || ns.size() != C)
        throw std::runtime_error("delta_bi_attention: qs/ks/vs/ns length mismatch");
    int K = 0;
    for (int n : ns) K += n;

    // Phase 1: K/V prep for every chunk (the bidirectional schedule reads all
    // groups). Replaces the cache push — the groups are locals, dead on return.
    std::vector<PackedCtx> kg(C), vg(C);
    for (size_t c = 0; c < C; ++c) {
        inf.n_tok = ns[c];
        { WithStep _wk(inf, "bi.k_prep"); kg[c] = fresh_masked(inf, ks[c]); }
        ks[c] = PackedCtx{};
        { WithStep _wv(inf, "bi.v_prep"); vg[c] = fresh_masked(inf, vs[c]); }
        vs[c] = PackedCtx{};
    }

    // Phase 2: per-chunk δ-block attention over the full key set.
    std::vector<PackedCtx> out(C);
    for (size_t c = 0; c < C; ++c) {
        inf.n_tok = ns[c];
        inf.fhe->level_hint(qs[c], inf.fhe->level_limit() - 3);
        std::vector<PackedCtx> S = cachemir_filling::qkt_delta_groups(inf, qs[c], kg, K);
        qs[c] = PackedCtx{};
        std::vector<PackedCtx> P = cachemir_filling::attention_softmax_thor_delta_core(
            inf, std::move(S), "attn", cachemir_filling::bd_delta_view(inf, K), "bi.stg.");
        out[c] = cachemir_filling::softmax_v_groups(inf, std::move(P), vg, K, "bi.stg.");
    }
    return out;
}

}  // namespace bidirectional
