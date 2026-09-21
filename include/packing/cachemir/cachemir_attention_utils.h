#pragma once

#include "inference.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace cachemir {

inline constexpr int kBenchRot = 5;
inline constexpr int kCacheBtsPeriod = 8;

// Per-lane cache levels: after the forced K/V push bootstrap (cache fresh at bts level), K is
// dropped to CACHE_READ_LEVEL_K and V to CACHE_READ_LEVEL_V — separate knobs so each lane sits at
// the staler level that makes its downstream consumer (qkt for K, softmax_v for V) cheaper.
inline constexpr int kDefaultCacheReadLevel = 23;
inline int cache_read_level_k() {
    const char* e = std::getenv("CACHE_READ_LEVEL_K");
    return (e && *e) ? std::atoi(e) : kDefaultCacheReadLevel;
}
inline int cache_read_level_v() {
    const char* e = std::getenv("CACHE_READ_LEVEL_V");
    return (e && *e) ? std::atoi(e) : kDefaultCacheReadLevel;
}
inline int mha_rot(const Inference& inf, int real_idx) {
    return inf.bench_mode ? kBenchRot : real_idx;
}

inline std::vector<double> real_head_tok0_mask(const Inference& inf, int tok_offset = 0) {
    const int N  = inf.slots;
    const int d  = inf.size.hidDim;
    const int H  = inf.size.numHeads;
    const int t  = N / d;
    const int tH = t * H;
    const int d_head_real = inf.size.getRealDHead();
    const int H_real      = inf.size.getRealNumHeads();
    std::vector<double> mask(N, 0.0);
    for (int lane = 0; lane < d_head_real; ++lane)
        for (int h = 0; h < H_real; ++h)
            mask[lane * tH + h * t + tok_offset] = 1.0;
    return mask;
}

// real_head_tok0_mask scaled by 0.5 (the conj-doubling factor). Shared by the
// softmax_v "tok0.h" and cache_k_push "kpush.tok0h" sites + generate_decode_masks.
inline std::vector<double> real_head_half_mask(const Inference& inf) {
    std::vector<double> m = real_head_tok0_mask(inf);
    for (double& x : m) x *= 0.5;
    return m;
}

// softmax score-mask additive selector: clip_lo-mean everywhere, -mean on the kc
// active (head,token) slots. VALUE is per-block (clip_lo/mean) ⇒ block-scoped tag.
inline std::vector<double> score_mask_vec(const Inference& inf,
                                          double clip_lo, double mean, int kc) {
    const int N = inf.slots;
    const int d = inf.size.hidDim;
    const int H = inf.size.numHeads;
    const int t = N / d;
    std::vector<double> mask(N, clip_lo - mean);
    for (int h = 0; h < H; ++h)
        for (int tok = 0; tok < kc; ++tok)
            mask[tok / t * t * H + h * t + tok % t] = -mean;
    return mask;
}
// Prefix-taking form is the primitive; the Inference& form forwards to it. Both must produce
// byte-identical tags — the main thread and the staging worker walk the same site table.
inline std::string score_mask_tag(const std::string& prefix, int kc) {
    return Inference::scoped_in(prefix, "sm.score.k" + std::to_string(kc));
}
inline std::string score_mask_tag(const Inference& inf, int kc) {
    return score_mask_tag(inf.block_prefix, kc);
}

// softmax active-mask multiplicative selector: 0.5/kc on the kc active slots.
// VALUE is block-independent ⇒ unscoped tag (shared 12× across blocks).
inline std::vector<double> active_mask_vec(const Inference& inf, int kc) {
    const int N = inf.slots;
    const int d = inf.size.hidDim;
    const int H = inf.size.numHeads;
    const int t = N / d;
    const int tH_active = t * H;
    const double ascale = 0.5 / static_cast<double>(kc);
    std::vector<double> amask(N, 0.0);
    for (int h = 0; h < H; ++h)
        for (int tok = 0; tok < kc; ++tok)
            amask[tok / t * tH_active + h * t + tok % t] = ascale;
    return amask;
}
inline std::string active_mask_tag(int kc) {
    return "sm.active.k" + std::to_string(kc);
}

// qkt group-mask multiplicative selector for group g (num_tok real tokens in g).
// gscale = 0.5 / sqrt(d_head_real). Block-independent ⇒ unscoped tag.
inline std::vector<double> qkt_group_mask_vec(const Inference& inf, int num_tok, int g) {
    const int N  = inf.slots;
    const int d  = inf.size.hidDim;
    const int H  = inf.size.numHeads;
    const int t  = N / d;
    const int tH = t * H;
    const int H_real = inf.size.getRealNumHeads();
    const double gscale = 0.5 * (1.0 / std::sqrt(static_cast<double>(inf.size.getRealDHead())));
    std::vector<double> gmask(N, 0.0);
    for (int i = 0; i < N; ++i) {
        const int h = (i % tH) / t;
        if (i / tH == g && i % t < num_tok && h < H_real)
            gmask[i] = gscale;
    }
    return gmask;
}
inline std::string qkt_group_mask_tag(int kc, int g) {
    return "qkt.gmask.k" + std::to_string(kc) + ".g" + std::to_string(g);
}

inline std::vector<std::complex<double>> qkt_complex_odd_mask_vec(const Inference& inf, int num_tok, int g) {
    const std::vector<double> base = qkt_group_mask_vec(inf, num_tok, g);
    std::vector<std::complex<double>> out(base.size());
    for (size_t i = 0; i < base.size(); ++i) out[i] = std::complex<double>(0.0, -base[i]);
    return out;
}
inline std::string qkt_complex_odd_mask_tag(int kc, int g) {
    return "qkt.gmask.cplx.k" + std::to_string(kc) + ".g" + std::to_string(g);
}

// head_reduce_sum position-0 selector (1.0 on every t-th slot). Block/token-independent.
inline std::vector<double> hrs_pos0_vec(const Inference& inf) {
    const int N = inf.slots;
    const int d = inf.size.hidDim;
    const int t = N / d;
    std::vector<double> m(N, 0.0);
    for (int i = 0; i < N; i += t) m[i] = 1.0;
    return m;
}

// V-push per-lane selector (lane_scale on lane i, real heads, slot right_rot).
// Block-independent value AND level (cache-pin) ⇒ unscoped tag, shared across blocks.
inline std::vector<double> vlane_mask_vec(const Inference& inf, int i, int right_rot,
                                          double lane_scale = 0.5) {
    const int N  = inf.slots;
    const int d  = inf.size.hidDim;
    const int H  = inf.size.numHeads;
    const int t  = N / d;
    const int tH = t * H;
    const int H_real = inf.size.getRealNumHeads();
    std::vector<double> mask(N, 0.0);
    for (int h = 0; h < H_real; ++h)
        mask[i * tH + h * t + right_rot] = lane_scale;
    return mask;
}
inline std::string vlane_mask_tag(int i, int right_rot) {
    return "v.lane." + std::to_string(i) + "." + std::to_string(right_rot);
}

inline std::vector<std::complex<double>> vpair_mask_vec(const Inference& inf, int i, int right_rot,
                                                        bool imag, double lane_scale = 0.5) {
    const std::vector<double> base = vlane_mask_vec(inf, i, right_rot, lane_scale);
    std::vector<std::complex<double>> out(base.size());
    for (size_t j = 0; j < base.size(); ++j)
        out[j] = imag ? std::complex<double>(0.0, base[j]) : std::complex<double>(base[j], 0.0);
    return out;
}
// Combined pair-mask for the complex V push: Re selector on source lane i_re, Im selector on
// source lane i_im (disjoint slots → one complex mask extracts both lanes of a bucket in ONE mult).
// Equivalent by linearity to vpair_mask_vec(i_re,…,false) + vpair_mask_vec(i_im,…,true), so the
// push runs in d_head/2 mults instead of d_head, bit-identically.
inline std::vector<std::complex<double>> vpair_mask_complex_vec(const Inference& inf, int i_re, int i_im,
                                                                int right_rot, double lane_scale = 0.5) {
    const std::vector<double> re = vlane_mask_vec(inf, i_re, right_rot, lane_scale);
    const std::vector<double> im = vlane_mask_vec(inf, i_im, right_rot, lane_scale);
    std::vector<std::complex<double>> out(re.size());
    for (size_t j = 0; j < out.size(); ++j) out[j] = std::complex<double>(re[j], im[j]);
    return out;
}
inline std::string vpair_mask_complex_tag(int i_re, int i_im, int right_rot) {
    return "v.cpc." + std::to_string(i_re) + "." + std::to_string(i_im) + "." + std::to_string(right_rot);
}

inline std::vector<std::complex<double>> complex_vlane_mask_vec(const Inference& inf, int i, int right_rot,
                                                               double imag_scale = -0.5) {
    const std::vector<double> base = vlane_mask_vec(inf, i, right_rot, 0.5);
    std::vector<std::complex<double>> out(base.size());
    for (size_t j = 0; j < base.size(); ++j) out[j] = std::complex<double>(0.0, imag_scale * base[j]);
    return out;
}
// Rearrange weight columns from head-grouped [head0_dims | head1_dims | ...]
// to interleaved [h0_d0, h1_d0, ..., hH_d0, h0_d1, h1_d1, ...].
std::vector<std::vector<double>> rearrange_qkv_weights(
    const std::vector<std::vector<double>>& W, int H);

std::vector<double> rearrange_qkv_biases(
    const std::vector<double>& b, int H);

std::vector<std::vector<double>> rearrange_wo_weights(
    const std::vector<std::vector<double>>& W, int H);

}  // namespace cachemir
