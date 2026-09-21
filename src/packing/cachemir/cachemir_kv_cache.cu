#include "inference.h"
#include "packing/cachemir/cachemir_attention.h"
#include "packing/cachemir/cachemir_attention_utils.h"
#include "ckks_types.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

namespace cachemir {

namespace {

// Tokens packed per ciphertext (slots / hidDim).
inline int tok_stride(const Inference& inf) { return inf.slots / inf.size.hidDim; }

PackedCtx k_mask_and_rotate(Inference& inf, const PackedCtx& key, double extra_scale) {
    WithStep _w(inf, "tok0_mask_mult");
    Ptx pt = inf.encode_at_cached("kpush.tok0h", key, [&] {
        std::vector<double> m = real_head_half_mask(inf);
        if (extra_scale != 1.0)
            for (double& x : m) x *= extra_scale;
        return m;
    });
    PackedCtx masked = inf.fhe->mult(key, pt);
    return inf.fhe->rotate(masked, mha_rot(inf, -(inf.k_count() % tok_stride(inf))));
}

PackedCtx v_rotate(Inference& inf, const PackedCtx& value, int right_rot) {
    return (right_rot == 0) ? inf.fhe->clone(value)
                            : inf.fhe->rotate(value, mha_rot(inf, -right_rot));
}

void k_cache_accumulate(Inference& inf, const PackedCtx& rotated) {
    const int t = tok_stride(inf);
    auto& kc = inf.cache[inf.scoped("k")];
    if (inf.k_count() % t == 0) {
        kc.push_back(inf.fhe->clone(rotated));
    } else {
        inf.name_graph_ct_if_absent(kc.back(),
                                    inf.scoped("k.acc." + std::to_string(inf.k_count() / t)));
        inf.fhe->inplace_add(kc.back(), rotated);
    }
    inf.k_count()++;
}

// Accumulate one masked V contribution into V-cache `lane` (clone on first write).
void v_lane_accumulate(Inference& inf, int lane, const PackedCtx& tmp) {
    auto& vc = inf.cache[inf.scoped("v")];
    if (vc[lane].ct == nullptr) {
        vc[lane] = inf.fhe->clone(tmp);
    } else {
        inf.name_graph_ct_if_absent(vc[lane], inf.scoped("v.acc." + std::to_string(lane)));
        inf.fhe->inplace_add(vc[lane], tmp);
    }
}

template <class MaskFn>
void v_lane_scatter(Inference& inf, const PackedCtx& v_rot, int right_rot, MaskFn&& mask_fn) {
    const int d           = inf.size.hidDim;
    const int d_head      = d / inf.size.numHeads;
    const int t           = tok_stride(inf);
    const int d_head_real = inf.size.getRealDHead();

    std::vector<Ptx> pts;
    pts.reserve(d_head_real);
    for (int i = 0; i < d_head_real; ++i)
        pts.push_back(inf.encode_at_cached(vlane_mask_tag(i, right_rot), v_rot,
                                           [&] { return mask_fn(i); }));

    if (inf.fhe->lane_batch_usable(v_rot.ct, pts)) {
        WithStep _w(inf, "lane_mask_mult");
        auto raw = inf.fhe->mult_batch_exec(v_rot.ct, pts);
        for (int i = 0; i < d_head_real; ++i) {
            PackedCtx tmp = inf.fhe->mult_finish(v_rot, pts[i], std::move(raw[i]));
            const int lane = ((inf.v_count() / t) - i + d_head) % d_head;
            v_lane_accumulate(inf, lane, tmp);
        }
    } else {
        for (int i = 0; i < d_head_real; ++i) {
            WithStep _w(inf, "lane_mask_mult");
            PackedCtx tmp = inf.fhe->mult(v_rot, pts[i]);
            const int lane = ((inf.v_count() / t) - i + d_head) % d_head;
            v_lane_accumulate(inf, lane, tmp);
        }
    }
    inf.v_count()++;
}

}  // namespace

void prepare_mha_masks(Inference& inf) {
    inf.cache[inf.scoped("k")] = {};
    inf.k_count() = 0;
}

PackedCtx prepare_pushed_k(Inference& inf, const PackedCtx& key) {
    PackedCtx rotated = k_mask_and_rotate(inf, key, /*extra_scale=*/1.0);
    inf.fhe->bootstrap(rotated.ct);
    { WithStep _wc(inf, "im_cleanse"); inf.fhe->inplace_im_cleanse(rotated); }
    inf.fhe->drop_to_level(rotated, cache_read_level_k());
    return rotated;
}

void cache_k_push(Inference& inf, const PackedCtx& key) {
    WithStep _w(inf, "cache_k_push");
    PackedCtx rotated = prepare_pushed_k(inf, key);
    k_cache_accumulate(inf, rotated);
}

void prepare_vcache(Inference& inf) {
    int d_head = inf.size.hidDim / inf.size.numHeads;
    inf.cache[inf.scoped("v")] = {};
    inf.cache[inf.scoped("v")].resize(d_head);
    inf.v_count() = 0;
}

void cache_v_push(Inference& inf, const PackedCtx& value) {
    WithStep _w(inf, "cache_v_push");
    const int right_rot = inf.v_count() % tok_stride(inf);
    PackedCtx v_rot = v_rotate(inf, value, right_rot);

    inf.fhe->bootstrap(v_rot.ct);   // K/V push always refreshes (was gated by removed GPT2_KV_PUSH_BTS)
    {
        WithStep _wc(inf, "im_cleanse");
        inf.fhe->inplace_im_cleanse(v_rot);
    }

    inf.fhe->drop_to_level(v_rot, cache_read_level_v());

    const double lane_scale = 0.5;
    v_lane_scatter(inf, v_rot, right_rot,
                   [&](int i) { return vlane_mask_vec(inf, i, right_rot, lane_scale); });
}

void cache_kv_push(Inference& inf, const PackedCtx& key, const PackedCtx& value) {
    if (!inf.fhe->complex_payload) {
        cachemir::cache_k_push(inf, key);
        cachemir::cache_v_push(inf, value);
        return;
    }
    WithStep _w(inf, "cache_kv_push");

    PackedCtx rotated   = k_mask_and_rotate(inf, key, /*extra_scale=*/0.5);
    const int right_rot = inf.v_count() % tok_stride(inf);
    PackedCtx v_rot     = v_rotate(inf, value, right_rot);

    inf.fhe->inplace_im_cleanse(rotated);
    inf.fhe->inplace_im_cleanse(v_rot);

    // --- pack (K + i*V) -> one bootstrap -> unpack ---
    {
        WithStep _wp(inf, "kv_pack_bts");
        Ptx i_pt = inf.encode_complex_const_at(0.0, 1.0, v_rot);
        PackedCtx P = inf.fhe->pack_ri(rotated, v_rot, i_pt);
        inf.fhe->bootstrap(P.ct);
        PackedCtx conj = inf.fhe->conjugate(P);
        rotated = inf.fhe->add(P, conj);   // 2*Re(P) = K  (kpush mask was x0.5)
        v_rot   = inf.fhe->sub(P, conj);   // 2i*Im(P) = V (realified + scaled by the complex v.lane mask)
    }

    inf.fhe->drop_to_level(rotated, cache_read_level_k());
    inf.fhe->drop_to_level(v_rot, cache_read_level_v());

    k_cache_accumulate(inf, rotated);
    v_lane_scatter(inf, v_rot, right_rot,
                   [&](int i) { return complex_vlane_mask_vec(inf, i, right_rot); });
}

void cache_kv_push_packed(Inference& inf, const PackedCtx& P_in) {
    WithStep _w(inf, "cache_kv_push");
    const int right_rot = inf.v_count() % tok_stride(inf);
    PackedCtx P = inf.fhe->clone(P_in);
    {
        WithStep _wp(inf, "kv_pack_bts");
        inf.fhe->bootstrap(P.ct);
        PackedCtx conj = inf.fhe->conjugate(P);
        PackedCtx K = inf.fhe->add(P, conj);   // 2*Re(P) = 2*K_raw   (deg-preserving)
        PackedCtx V = inf.fhe->sub(P, conj);   // 2i*Im(P) = 2i*V_raw (deg-preserving)

        PackedCtx rotated = k_mask_and_rotate(inf, K, /*extra_scale=*/1.0);
        inf.fhe->drop_to_level(rotated, cache_read_level_k());
        k_cache_accumulate(inf, rotated);

        PackedCtx v_rot = v_rotate(inf, V, right_rot);
        inf.fhe->drop_to_level(v_rot, cache_read_level_v());
        v_lane_scatter(inf, v_rot, right_rot,
                       [&](int i) { return complex_vlane_mask_vec(inf, i, right_rot, /*imag_scale=*/-1.0); });
    }
}

}  // namespace cachemir
