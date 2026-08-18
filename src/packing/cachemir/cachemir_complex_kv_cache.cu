#include "inference.h"
#include "packing/cachemir/cachemir_attention.h"
#include "packing/cachemir/cachemir_attention_utils.h"
#include "ckks_types.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace cachemir {

PackedCtx pack_k_groups_complex(Inference& inf, const PackedCtx& K_even, const PackedCtx& K_odd) {
    Ptx i_pt = inf.encode_complex_const_at(0.0, 1.0, K_odd);
    return inf.fhe->pack_ri(K_even, K_odd, i_pt);
}

namespace {

void bucket_v_complex(Inference& inf, const PackedCtx& v_rot, int right_rot) {
    const int d        = inf.size.hidDim;
    const int t        = inf.slots / d;
    const int d_head   = d / inf.size.numHeads;
    auto& vc = inf.cache[inf.scoped("v")];
    const int d_head_real = inf.size.getRealDHead();
    const int c = inf.v_count() / t;
    for (int p = 0; p < d_head_real / 2; ++p) {
        WithStep _wm(inf, "lane_mask_mult");
        const int i_re = ((c - 2 * p)     % d_head + d_head) % d_head;   // source lane → dest 2p   (Re)
        const int i_im = ((c - 2 * p - 1) % d_head + d_head) % d_head;   // source lane → dest 2p+1 (Im)
        Ptx mask_pt = inf.encode_at_cached_complex(
            vpair_mask_complex_tag(i_re, i_im, right_rot), v_rot,
            [&] { return vpair_mask_complex_vec(inf, i_re, i_im, right_rot); });
        PackedCtx tmp = inf.fhe->mult(v_rot, mask_pt);
        if (vc[p].ct == nullptr) {
            vc[p] = inf.fhe->clone(tmp);
        } else {
            inf.name_graph_ct_if_absent(vc[p], inf.scoped("v.acc." + std::to_string(p)));
            inf.fhe->inplace_add(vc[p], tmp);
        }
    }
    inf.v_count()++;
}

void bucket_k_complex(Inference& inf, const PackedCtx& rotated) {
    const int t = inf.slots / inf.size.hidDim;
    auto& pend = inf.cache[inf.scoped("k.pend")];
    if (inf.k_count() % t == 0) {
        pend.push_back(inf.fhe->clone(rotated));    // new group starts
    } else {
        inf.name_graph_ct_if_absent(pend.back(), inf.scoped("k.acc." + std::to_string(inf.k_count() / t)));
        inf.fhe->inplace_add(pend.back(), rotated); // accumulate into the group
    }
    inf.k_count()++;
    if (pend.size() == 2 && inf.k_count() % t == 0) {                       // even+odd pair complete
        auto _sg = inf.fhe->graph_scope_guard();
        inf.fhe->graph_scope_set("kpack");
        PackedCtx bucket = pack_k_groups_complex(inf, pend[0], pend[1]);
        inf.fhe->drop_to_level(bucket, cache_read_level_k());              // risk-D: pin q*B == q*g level
        inf.cache[inf.scoped("k")].push_back(std::move(bucket));
        pend.clear();                                                       // free the two real groups
    }
}

}  // namespace

void prepare_complex_v_cache(Inference& inf) {
    const int d_head = inf.size.hidDim / inf.size.numHeads;
    inf.cache[inf.scoped("v")] = {};
    inf.cache[inf.scoped("v")].resize(d_head / 2);   // pack two lanes per bucket
    inf.v_count() = 0;
}

void prepare_complex_k_cache(Inference& inf) {
    inf.cache[inf.scoped("k")]      = {};   // completed complex pairs
    inf.cache[inf.scoped("k.pend")] = {};   // pending real groups (0..2)
    inf.k_count() = 0;
}

void cache_kv_push_packed_complex(Inference& inf, const PackedCtx& P_in) {
    WithStep _w(inf, "cache_kv_push");
    const int t = inf.slots / inf.size.hidDim;
    const int right_rot = inf.v_count() % t;

    PackedCtx P = inf.fhe->clone(P_in);
    WithStep _wp(inf, "kv_pack_bts");
    inf.fhe->bootstrap(P.ct);
    PackedCtx conj = inf.fhe->conjugate(P);
    PackedCtx K = inf.fhe->add(P, conj);   // 2·K_raw  (real)
    PackedCtx V = inf.fhe->sub(P, conj);

    {
        WithStep _wm(inf, "tok0_mask_mult");
        Ptx pt = inf.encode_at_cached("kpush.tok0h", K, [&] { return real_head_half_mask(inf); });
        K = inf.fhe->mult(K, pt);
    }
    K = inf.fhe->rotate(K, mha_rot(inf, -(inf.k_count() % t)));
    inf.fhe->drop_to_level(K, cache_read_level_k());
    bucket_k_complex(inf, K);

    PackedCtx v_rot = (right_rot == 0) ? inf.fhe->clone(V)
                                       : inf.fhe->rotate(V, mha_rot(inf, -right_rot));
    Ptx nhi = inf.encode_complex_const_at(0.0, -1.0, v_rot);   // realify 2i·V_raw → 2·V_raw
    v_rot   = inf.fhe->mult(v_rot, nhi);
    inf.fhe->drop_to_level(v_rot, cache_read_level_v());
    bucket_v_complex(inf, v_rot, right_rot);
}

}  // namespace cachemir
