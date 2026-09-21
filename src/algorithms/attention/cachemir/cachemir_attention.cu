#include "ckks_primitives.h"
#include "inference.h"
#include "nonlinear.h"
#include "packing/cachemir/cachemir_attention.h"
#include "packing/cachemir/cachemir_attention_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace cachemir {

PackedCtx qkt(Inference& inf, const PackedCtx& query) {
    WithStep _w(inf, "qkt");
    int N  = inf.slots;
    int d  = inf.size.hidDim;
    int H  = inf.size.numHeads;
    int t  = N / d;
    int tH = t * H;
    
    WithStep _w2(inf, "q_tok0_mask_mult");
    Ptx tok0_pt = inf.encode_at_cached("tok0", real_head_tok0_mask(inf), query);
    PackedCtx q = inf.fhe->mult(query, tok0_pt);

    _w2.next("q_replicate");
    for (int step = 1; step < t; step *= 2) {
        PackedCtx tmp = inf.fhe->rotate(q, mha_rot(inf, -step));
        inf.fhe->inplace_add(q, tmp);
    }

    int keys_per_ct = t;
    int num_groups  = (int)inf.cache[inf.scoped("k")].size();
    PackedCtx attn_ct;

    auto _scope = inf.fhe->graph_scope_guard();

    for (int g = 0; g < num_groups; ++g) {
        WithStep _wg(inf, "qkt_group");
        inf.fhe->graph_scope_set("qkt.group");   // reset body-local names every iteration
        int first_key = g * keys_per_ct;
        int num_tok   = std::min(keys_per_ct, inf.k_count() - first_key);

        PackedCtx tmp = inf.cache[inf.scoped("k")][g];
        inf.name_graph_ct(tmp, inf.scoped("cache.k." + std::to_string(g) + "-lvl=" + std::to_string(inf.fhe->level_for_ct(tmp.ct))));

        WithStep _w2g(inf, "q_dot_k");

        PackedCtx result = inf.fhe->mult(q, tmp);

        _w2g.next("tH_reduce");
        for (int s = tH; s < N; s *= 2) {
            PackedCtx rot = inf.fhe->rotate(result, mha_rot(inf, s));
            inf.fhe->inplace_add(result, rot);
        }

        _w2g.next("gmask_mult");
        Ptx gmask_pt = inf.encode_at_cached(
            qkt_group_mask_tag(inf.k_count(), g), result,
            [&] { return qkt_group_mask_vec(inf, num_tok, g); });
        result = inf.fhe->mult(result, gmask_pt);
        {
            WithStep _wc(inf, "im_cleanse");
            inf.fhe->inplace_im_cleanse(result);   // 0.5*s + conj(0.5*s) = Re(s)
        }

        if (g == 0)
            attn_ct = std::move(result);
        else
            inf.fhe->inplace_add(attn_ct, result);
    }

    return attn_ct;
}

PackedCtx head_reduce_sum(Inference& inf, const PackedCtx& x) {
    WithStep _w(inf, "head_reduce_sum");
    int N  = inf.slots;
    int d  = inf.size.hidDim;
    int H  = inf.size.numHeads;
    int t  = N / d;
    int tH = t * H;

    PackedCtx out = inf.fhe->clone(x);

    for (int step = 1; step < t; step *= 2)
        inf.fhe->inplace_add(out, inf.fhe->rotate(out, mha_rot(inf, step)));

    Ptx pt0 = inf.encode_at_cached("hrs.pos0", out, [&] { return hrs_pos0_vec(inf); });
    inf.fhe->inplace_mult(out, pt0);

    for (int step = 1; step < t; step *= 2)
        inf.fhe->inplace_add(out, inf.fhe->rotate(out, mha_rot(inf, -step)));

    // TODO: Do we need replication here?
    for (int s = tH; s < N; s *= 2) {
        PackedCtx rot = inf.fhe->rotate(out, mha_rot(inf, s));
        inf.fhe->inplace_add(out, rot);
    }

    return out;
}

// Plain linear init: α = 4/(d_min+d_max), β = α²/4
static inline std::pair<double, double>
gs_init_linear(double d_min, double d_max) {
    const double alpha = 4.0 / (d_min + d_max);
    const double beta  = alpha * alpha / 4.0;
    return {alpha, beta};
}

// Chebyshev (sup-norm optimal) linear init.
static inline std::pair<double, double>
gs_init_chebyshev(double d_min, double d_max) {
    const double s    = d_min + d_max;
    const double beta = 8.0 / (s * s + 4.0 * d_min * d_max);
    return {beta * s, beta};
}

static inline std::pair<double, double>
gs_init(GSInitMethod m, double d_min, double d_max) {
    switch (m) {
        case GSInitMethod::LINEAR:    return gs_init_linear(d_min, d_max);
        case GSInitMethod::CHEBYSHEV: return gs_init_chebyshev(d_min, d_max);
    }
    return {0.0, 0.0};
}

static PackedCtx softmax_recip(Inference& inf, const PackedCtx& z, const PackedCtx& s,
                               const PackedCtx& F_init, int iters) {
    return goldschmidt_inv(inf.cc_ctx(), z, s, F_init, iters,
                           /*sparse_df=*/sparse_sm_enabled());
}

static double sm_recip_margin() {
    static const double m = [] {
        const char* v = std::getenv("SM_RECIP_MARGIN");
        return v && *v ? std::atof(v) : 0.0;
    }();
    return m;
}

PackedCtx attention_softmax_thor(Inference& inf, const PackedCtx& scores, const std::string& cfg_name) {
    WithStep _w(inf, "softmax_thor:" + cfg_name);
    const SoftmaxConfig& cfg = inf.sm_cfg.at(cfg_name);

    double mean = (cfg.clip_hi + cfg.clip_lo) / 2.0;

    PackedCtx ct = inf.fhe->clone(scores);

    WithStep _w2(inf, "score_mask_add");

    Ptx mask_pt = inf.encode_at_cached(
        score_mask_tag(inf, inf.k_count()), ct,
        [&] { return score_mask_vec(inf, cfg.clip_lo, mean, inf.k_count()); });
    ct = inf.fhe->add(ct, mask_pt);

    _w2.next("exp_poly");
    const double scale_factor = std::pow(2.0, -cfg.log2delta1 - cfg.log2delta2);

    if (cfg.cheb_coeffs.empty())
        throw std::runtime_error(
            "attention_softmax_thor: cfg.cheb_coeffs is empty -- only the Chebyshev exp path "
            "is supported (poly fallback removed; see docs/speed/chebyshev_fold.md)");
    PackedCtx z = eval_chebyshev_series(inf.cc_ctx(), ct, cfg.cheb_coeffs,
                                        cfg.cheb_a / scale_factor,
                                        cfg.cheb_b / scale_factor);

    _w2.next("exp_squares1");
    for (int i = 0; i < cfg.log2delta1; ++i) {
        inf.fhe->inplace_square(z);
    }
    _w2.next("active_mask_mult");
    const double kc = static_cast<double>(inf.k_count());
    {
        WithStep _wc(inf, "im_cleanse");
        inf.fhe->inplace_im_cleanse(z);   // amask below carries the 0.5
    }

    Ptx active_pt = inf.encode_at_cached(
        active_mask_tag(inf.k_count()), z,
        [&] { return active_mask_vec(inf, inf.k_count()); });
    z = inf.fhe->mult(z, active_pt);

    PackedCtx s = head_reduce_sum(inf, z);

    _w2.next("gs_inv_init");
    PackedCtx F_init = inf.fhe->mult(s, -cfg.init_beta * (1.0 - sm_recip_margin()));
    inf.fhe->inplace_add(F_init, cfg.init_alpha);

    PackedCtx y = softmax_recip(inf, z, s, F_init, cfg.gs_iters_scaled);
    for (int i = 0; i < cfg.log2delta2; ++i) {
        WithStep _w2r(inf, "refine_iter");
        {
            WithStep _wc(inf, "im_cleanse");
            inf.fhe->inplace_im_cleanse(y);
        }
        z = inf.fhe->square(y);
        z = inf.fhe->mult(z, 0.5 * std::sqrt(kc) * 0.25);   // 0.25 absorbs the conj-doubling of y
        s = head_reduce_sum(inf, z);

        double r = 1.0;
        if (!cfg.sm_kc_r.empty() && cfg.log2delta2 > 0) {
            const int C = static_cast<int>(cfg.sm_kc_r.size()) / cfg.log2delta2;
            const int kpos = std::max(0, std::min(static_cast<int>(kc) - 1, C - 1));
            r = cfg.sm_kc_r[i * C + kpos];
        } else if (kc > 0.0) {
            constexpr double kc_ref = 4.0;                       // fallback proxy
            r = std::min(1.0, kc_ref / kc);
        }
        const double sa = std::sqrt(r), sb = r;
        PackedCtx F_init = inf.fhe->mult(s, -cfg.refine_beta[i] * sb * (1.0 - sm_recip_margin()));
        inf.fhe->inplace_add(F_init, cfg.refine_alpha[i] * sa);

        const int it_i = static_cast<int>(cfg.per_step_refine_iters.at(i));
        y = softmax_recip(inf, z, s, F_init, it_i);
    }

    return y;
}

PackedCtx softmax_v(Inference& inf, const PackedCtx& softmax_scores) {
    WithStep _w(inf, "softmax_v");
    int N      = inf.slots;
    int d      = inf.size.hidDim;
    int H      = inf.size.numHeads;
    int t      = N / d;
    int tH     = t * H;
    int d_head = d / H;

    WithStep _w2(inf, "lane0_mult");
    PackedCtx v = inf.cache[inf.scoped("v")][0];
    inf.name_graph_ct(v, inf.scoped("cache.v." + std::to_string(0) + "-lvl=" + std::to_string(inf.fhe->level_for_ct(v.ct))));   // actual level (V dropped to CACHE_READ_LEVEL_V at push)

    PackedCtx res = inf.fhe->mult(v, softmax_scores);

    const int d_head_real = inf.size.getRealDHead();

    std::vector<PackedCtx> lane_scores;
    if (d_head_real > 1) {
        WithStep _w2i(inf, "lane_mult");
        std::vector<int32_t> steps;
        steps.reserve(d_head_real - 1);
        for (int i = 1; i < d_head_real; ++i) steps.push_back(mha_rot(inf, i * tH));
        lane_scores = inf.fhe->rotate_hoisted(softmax_scores, steps);
    }

    for (int i = 1; i < d_head_real; ++i) {
        WithStep _w2i(inf, "lane_mult");
        PackedCtx scores = lane_scores[i - 1];
        PackedCtx v = inf.cache[inf.scoped("v")][i];
        inf.name_graph_ct(v, inf.scoped("cache.v." + std::to_string(i) + "-lvl=" + std::to_string(inf.fhe->level_for_ct(v.ct))));   // actual level (V dropped to CACHE_READ_LEVEL_V at push)

        PackedCtx tmp = inf.fhe->mult(v, scores);
        inf.fhe->inplace_add(res, tmp);
    }

    _w2.next("tok_reduce");
    for (int step = 1; step < t; step *= 2) {
        PackedCtx rot = inf.fhe->rotate(res, mha_rot(inf, step));
        inf.fhe->inplace_add(res, rot);
    }

    _w2.next("tok0_mask_mult");
    {
        inf.fhe->inplace_im_cleanse(res);
        Ptx tok0h_pt = inf.encode_at_cached(
            "tok0.h", res, [&] { return real_head_half_mask(inf); });
        res = inf.fhe->mult(res, tok0h_pt);
    }

    return res;
}

}  // namespace cachemir
