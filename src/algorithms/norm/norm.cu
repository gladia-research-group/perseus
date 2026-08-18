#include "ckks_primitives.h"
#include "inference.h"
#include "nonlinear.h"
#include "packing/cachemir/cachemir_norm_utils.h"
#include "packing/diagonal/diagonal_norm_utils.h"
#include "packing/cachemir_filling/cachemir_filling.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// CachemirFilling shares the diagonal

PackedCtx compute_per_token_sum(Inference& inf, const PackedCtx& x) {
    if (is_cachemir(x.packing)) return cachemir::compute_per_token_sum(inf, x);
    if (is_diagonal(x.packing) || is_cachemir_filling(x.packing))
        return diagonal::compute_per_token_sum(inf, x);
    throw std::runtime_error("compute_per_token_sum: unsupported packing");
}

PackedCtx compute_variance(Inference& inf, const PackedCtx& centered_x) {
    if (is_cachemir(centered_x.packing)) return cachemir::compute_variance_interleaved(inf, centered_x);
    if (is_diagonal(centered_x.packing) || is_cachemir_filling(centered_x.packing))
        return diagonal::compute_per_token_var(inf, centered_x);
    throw std::runtime_error("compute_variance: unsupported packing");
}

PackedCtx floor_inactive_token_lanes(Inference& inf, const PackedCtx& var_scaled, double fill) {
    if (is_cachemir(var_scaled.packing)) return cachemir::floor_inactive_token_lanes(inf, var_scaled, fill);
    if (is_diagonal(var_scaled.packing) || is_cachemir_filling(var_scaled.packing))
        return diagonal::floor_inactive_token_lanes(inf, var_scaled, fill);
    throw std::runtime_error("floor_inactive_token_lanes: unsupported packing");
}

Ptx encode_ln_center_mask(Inference& inf, const PackedCtx& x, const std::string& cfg_name,
                          int d, int t, double scale, int center_pos,
                          const std::vector<double>& per_pos_scale_sq) {
    if (is_cachemir(x.packing))
        return cachemir::encode_ln_center_mask(inf, x, cfg_name, d, t, scale, center_pos);
    if (is_diagonal(x.packing) || is_cachemir_filling(x.packing))
        return diagonal::encode_ln_center_mask(inf, x, d, t, scale, per_pos_scale_sq);
    throw std::runtime_error("encode_ln_center_mask: unsupported packing");
}

void add_layernorm_epsilon(Inference& inf, PackedCtx& var_scaled, double epsilon,
                           double center_scale_sq,
                           const std::vector<double>& per_pos_scale_sq) {
    if (is_cachemir(var_scaled.packing)) {
        cachemir::add_layernorm_epsilon(inf, var_scaled, epsilon,
                                        center_scale_sq, per_pos_scale_sq);
        return;
    }
    if (is_diagonal(var_scaled.packing) || is_cachemir_filling(var_scaled.packing)) {
        diagonal::add_layernorm_epsilon(inf, var_scaled, epsilon,
                                        center_scale_sq, per_pos_scale_sq);
        return;
    }
    throw std::runtime_error("add_layernorm_epsilon: unsupported packing");
}

PackedCtx ln_inv_sqrt_tail(Inference& inf, PackedCtx xs, const NormConfig& cfg,
                           int rD, int t, double c_eff_sq, bool sparse_var_scope,
                           bool tp_probes) {
    WithStep _w2(inf, "mean");
    PackedCtx mean = compute_per_token_sum(inf, xs);

    Ptx scale_mask = inf.encode_stride_mask_at_cached(
        "ln.scalemask.d" + std::to_string(rD) + ".t" + std::to_string(t),
        rD, t, mean, -1.0 / (double)rD);
    mean = inf.fhe->mult(mean, scale_mask);

    _w2.next("center");
    PackedCtx centered_x = inf.fhe->add(xs, mean);   // VECTOR — outside the sparse scope

    _w2.next("var");
    std::optional<CKKSContext::BtsItersScope> precise_scope;
    if (cfg.precise_var_bts) precise_scope.emplace(*inf.fhe, 2);
    PackedCtx var = compute_variance(inf, centered_x);   // slot-periodic from here

    PackedCtx inv_sqrt_var;
    {
        CKKSContext::SparseBtsScope ss(*inf.fhe, sparse_var_scope);
        const double floor_val = std::max(cfg.gs_lo, cfg.taylor_z0);
        PackedCtx var_scaled = floor_inactive_token_lanes(inf, var, floor_val);
        add_layernorm_epsilon(inf, var_scaled, cfg.epsilon, c_eff_sq, cfg.center_scale_sq);
        if (tp_probes) inf.fhe->tp_probe("lnvar", var_scaled.ct);

        _w2.next("inv_sqrt_init");
        PackedCtx inv_sqrt_init_scaled;
        switch (cfg.nr_init_method) {
            case NRInitMethod::TAYLOR: {
                std::vector<double> taylor_c = taylor_inv_sqrt_coeffs(cfg.taylor_z0);
                inv_sqrt_init_scaled = eval_taylor_inv_sqrt(
                    inf.cc_ctx(), var_scaled, taylor_c, cfg.taylor_z0);
                break;
            }
            case NRInitMethod::REMEZ: {
                inv_sqrt_init_scaled = eval_remez_31(
                    inf.cc_ctx(), var_scaled,
                    cfg.Ncoeffs, cfg.Dcoeffs,
                    cfg.lin_alpha, cfg.lin_beta,
                    cfg.gs_iters);
                break;
            }
        }
        precise_scope.reset();   // Newton runs 1-iter: its bts inputs are y-valued (in-window)

        _w2.next("inv_sqrt_newton");
        const double nx_scale =
            (cfg.nr_init_method == NRInitMethod::REMEZ)
                ? 1.0 / (cfg.inv_out_scale * cfg.inv_out_scale) : 1.0;
        inv_sqrt_var = inv_sqrt_newton(
            inf.cc_ctx(), var_scaled, inv_sqrt_init_scaled, cfg.nr_iters, nx_scale, rD, t);
        if (tp_probes) inf.fhe->tp_probe("lninv", inv_sqrt_var.ct);
        inf.fhe->bootstrap_hint(inv_sqrt_var, inf.fhe->level_limit() - 4);
    }  // sparse scope closes -> full-slot restored before the vector mult

    _w2.next("scale");
    return inf.fhe->mult(centered_x, inv_sqrt_var);   // s·LN; weight prep folds 1/s into gamma
}

PackedCtx norm(Inference& inf, const PackedCtx& x, const std::string& cfg_name) {
    if (is_cachemir_filling(x.packing) && inf.token_pair)
        return norm_token_pair(inf, x, cfg_name);
    const bool sparse_var =
        sparse_ln_enabled() && (is_cachemir(x.packing) || is_cachemir_filling(x.packing));
    WithStep _w(inf, "norm:" + cfg_name);
    const NormConfig& cfg = inf.norm_cfg.at(cfg_name);
    const int rD = inf.size.getRealHidDim();
    const int t  = inf.slots / inf.size.hidDim;

    double c_eff_sq = cfg.center_scale * cfg.center_scale;
    int center_pos = -1;   // -1 ⇒ position-invariant (no center_scale_sq); else clamped pos
    if (!cfg.center_scale_sq.empty() && is_cachemir(x.packing)) {
        center_pos = std::max(0, std::min(inf.output.capture_t,
                                          static_cast<int>(cfg.center_scale_sq.size()) - 1));
        c_eff_sq = cfg.center_scale_sq[center_pos];
    }
    const double c_eff = std::sqrt(c_eff_sq);   // linear centering scale for this token

    PackedCtx xs = [&] {
        WithStep _wc(inf, "im_cleanse");
        PackedCtx x2 = inf.fhe->im_cleanse(x);   // 2*Re(x)
        const double ch = 0.5 * c_eff;

        Ptx cmask = encode_ln_center_mask(
            inf, x2, cfg_name, rD, t, ch, center_pos, cfg.center_scale_sq);
        return inf.fhe->mult(x2, cmask);
    }();

    if (center_pos > 0 && cfg_name == "ln_f")
        inf.erase_enc_cache_all(cachemir::ln_center_mask_tag(inf, cfg_name, center_pos - 1));

    return ln_inv_sqrt_tail(inf, std::move(xs), cfg, rD, t, c_eff_sq,
                            sparse_var, /*tp_probes=*/false);
}
