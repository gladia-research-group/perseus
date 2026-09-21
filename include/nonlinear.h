#pragma once

#include "ckks_primitives.h"
#include "packing/packed_ctx.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

struct Inference;

enum class NRInitMethod {
    TAYLOR,
    REMEZ,
};
enum class GSInitMethod { LINEAR, CHEBYSHEV };

enum class GeLUMethod {
    SOFTSIGN_INV_SQRT,
    CHEBYSHEV,        
    THOR_COMPOSITE,   
};

struct NormConfig {
    NRInitMethod nr_init_method = NRInitMethod::TAYLOR;

    int    nr_iters       = 16;

    double epsilon        = 0.0;
    double taylor_z0      = 0.0;
    double center_scale   = 1.0;
    double inv_out_scale  = 1.0;

    std::vector<double> Ncoeffs;
    std::vector<double> Dcoeffs;
    double lin_alpha = 0.0;
    double lin_beta  = 0.0;
    double gs_lo     = 0.0;
    double gs_hi     = 0.0;
    int    gs_iters  = 0;

    std::vector<double> center_scale_sq;
    bool precise_var_bts = false;
};

PackedCtx sign      (Inference& inf, const PackedCtx& x);
// Public dispatcher

PackedCtx compute_per_token_sum(Inference& inf, const PackedCtx& x);
PackedCtx compute_variance     (Inference& inf, const PackedCtx& centered_x);
PackedCtx floor_inactive_token_lanes(Inference& inf, const PackedCtx& var_scaled, double fill);
Ptx  encode_ln_center_mask(Inference& inf, const PackedCtx& x, const std::string& cfg_name,
                           int d, int t, double scale, int center_pos,
                           const std::vector<double>& per_pos_scale_sq);
void add_layernorm_epsilon(Inference& inf, PackedCtx& var_scaled, double epsilon,
                           double center_scale_sq, const std::vector<double>& per_pos_scale_sq);

PackedCtx norm   (Inference& inf, const PackedCtx& x, const std::string& cfg_name);

bool      sparse_ln_enabled();
bool      sparse_sm_enabled();
bool      fused_ln_var_enabled();
bool      fused_sm_den_enabled();

// Prescale applied before the folded-softmax denominator bootstrap so the folded sum
// lands near 0.3 of the EvalMod range; never amplifies (capped at 1.0).
inline double fold_sm_prescale_for(double copies, double s0_expected) {
    constexpr double kTarget = 0.3;
    if (!(s0_expected > 0.0) || !std::isfinite(s0_expected) || copies <= 0.0) return 1.0;
    return std::max(1e-6, std::min(1.0, kTarget * copies / s0_expected));
}

PackedCtx ln_inv_sqrt_tail(Inference& inf, PackedCtx xs, const NormConfig& cfg,
                           int rD, int t, double c_eff_sq, bool sparse_var_scope);

PackedCtx norm_token_pair(Inference& inf, const PackedCtx& x, const std::string& cfg_name);

PackedCtx exp_approx(Inference& inf, const PackedCtx& x, int r);

struct SoftmaxConfig {
    GSInitMethod gs_init_method = GSInitMethod::LINEAR;

    int    log2delta1 = 0;
    int    log2delta2 = 0;
    double clip_lo    = 0.0;
    double clip_hi    = 0.0;

    std::vector<double> poly_coeffs;

    double init_alpha       = 0.0;
    double init_beta        = 0.0;
    int    gs_iters_scaled  = 0;

    std::vector<double> refine_alpha;
    std::vector<double> refine_beta;
    int    gs_iters_refine_scaled = 0;
    std::vector<double> per_step_refine_iters;

    std::vector<double> sm_kc_r;

    std::vector<double> cheb_coeffs;
    double cheb_a = 0.0;                 // = -delta0/2 (poly input lower bound)
    double cheb_b = 0.0;                 // = +delta0/2 (poly input upper bound)
};

struct GeLUConfig {
    GeLUMethod method = GeLUMethod::SOFTSIGN_INV_SQRT;

    bool gate = true;

    int exp_iters    = 12;
    int newton_iters = 2;
    int gs_iters     = 14;

    double a         = 0.0;
    double b         = 0.0;
    double c         = 0.0;
    double xmax      = 0.0;
    double z_min     = 0.0;
    double z_max     = 0.0;
    double gs_lo     = 0.0;
    double gs_hi     = 0.0;
    double lin_alpha = 0.0;
    double lin_beta  = 0.0;
    double inv_out_scale = 1.0;
    std::vector<double> Ncoeffs;
    std::vector<double> Dcoeffs;

    std::vector<double> cheb_coeffs;
    double cheb_a = 0.0;
    double cheb_b = 0.0;

    std::vector<double> thor_p1;
    std::vector<double> thor_p2;

    std::vector<double> thor_p1_cheb;
    std::vector<double> thor_p2_cheb;
    double thor_p1_a = -1.0;
    double thor_p1_b = 1.0;
    double thor_p2_a = 0.0;
    double thor_p2_b = 0.0;

    std::vector<double> gate_cheb_coeffs;
    double gate_cheb_a = 0.0;
    double gate_cheb_b = 0.0;
};

// CutMax argmax schedule
struct CutMaxCalib {
    double entry_scale     = 0.0;
    int    newton_per_pass = 0;
    int    newton_polish   = 0;
    int    gs_sum_iters    = 0;
    double sum_lo          = 0.0;
    double sum_hi          = 0.0;
    std::vector<double> p, c, m, s2_hi, passes, ex2;
    std::vector<double> chord_a, chord_b;
    std::vector<double> cascade_iters;
};

PackedCtx gelu_approx(Inference& inf, PackedCtx& x, const std::string& cfg_name);

PackedCtx gelu_softsign_core(Inference& inf, PackedCtx x2, const GeLUConfig& cfg);
PackedCtx gelu_thor_core   (Inference& inf, PackedCtx t,  const GeLUConfig& cfg);

PackedCtx gelu_token_pair(Inference& inf, const PackedCtx& x, const std::string& cfg_name);