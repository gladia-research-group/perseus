#pragma once

#include "ckks_primitives.h"
#include "packing/packed_ctx.h"
#include <string>
#include <vector>

struct Inference;

enum class NRInitMethod {
    TAYLOR,
    REMEZ,
};
enum class GSInitMethod { LINEAR, CHEBYSHEV };

enum class GeLUMethod {
    SOFTSIGN_INV_SQRT,   // 0.5·x·(1 + g·1/√(1+g²)) with g = x·poly(x²);
    CHEBYSHEV,           // P(x) ≈ GeLU(x), Chebyshev series on [cheb_a,cheb_b] via Paterson-Stockmeyer;
    THOR_COMPOSITE,      // x·(P2(P1(x/S)) + 1/2), THOR tanh-form composite (Moon et al. 2024); P1 deg31, P2 deg27;
};

struct NormConfig {
    NRInitMethod nr_init_method = NRInitMethod::TAYLOR;

    int    nr_iters       = 16;

    double epsilon        = 0.0;
    double taylor_z0      = 0.0;
    double center_scale   = 1.0;   // c: LN input scale (LN(c·x)=LN(x)); 1.0 = no scaling
    double inv_out_scale  = 1.0;   // s: inv_sqrt target is s·z^{-1/2}; gamma absorbs 1/s

    std::vector<double> Ncoeffs;
    std::vector<double> Dcoeffs;
    double lin_alpha = 0.0;
    double lin_beta  = 0.0;
    double gs_lo     = 0.0;
    double gs_hi     = 0.0;
    int    gs_iters  = 0;

    std::vector<double> center_scale_sq;
    // 2-iter bootstraps inside the variance -> GS-init segment. Set by the
    // calibration for sites whose per-position pool variance spread exceeds the
    // absolute rescale window (ViT deep blocks): the cap-clamped tokens'
    // var_scaled lands below the 1-iter bts noise floor, so their inv_sqrt goes
    // 15-30% wrong; the scoped 2-iter bts (~17 bits) absorbs it.
    bool precise_var_bts = false;
};

PackedCtx sign      (Inference& inf, const PackedCtx& x);
PackedCtx lt_function(Inference& inf, const PackedCtx& x, double value, double rescale_factor = 1.0);

// Public dispatcher

// Packing-specific steps inside norm() — each dispatches on packing at its own
// boundary (cachemir vs diagonal/filling); callers use these un-prefixed forms.
PackedCtx compute_per_token_sum(Inference& inf, const PackedCtx& x);
PackedCtx compute_variance     (Inference& inf, const PackedCtx& centered_x);
PackedCtx floor_inactive_token_lanes(Inference& inf, const PackedCtx& var_scaled, double fill);
Ptx  encode_ln_center_mask(Inference& inf, const PackedCtx& x, const std::string& cfg_name,
                           int d, int t, double scale, int center_pos,
                           const std::vector<double>& per_pos_scale_sq);
void add_layernorm_epsilon(Inference& inf, PackedCtx& var_scaled, double epsilon,
                           double center_scale_sq, const std::vector<double>& per_pos_scale_sq);

PackedCtx norm   (Inference& inf, const PackedCtx& x, const std::string& cfg_name);

// Sparse-bts env gates (SPARSE_LN_BTS / SPARSE_SM_BTS; defined in sparse_norm.cu)
bool      sparse_ln_enabled();
bool      sparse_sm_enabled();

// THE shared LN inv_sqrt tail (norm.cu): centered-scale input xs -> s*LN(xs)
// (mean/center/var/init/newton/scale). ONE body for the real, sparse-routed, and
// token-pair arms — op-order-identical to all three, so plans keep binding.
// sparse_var_scope routes the variance-chain bootstraps to the sparse precomp;
// tp_probes fires the token-pair lnvar/lninv probes.
PackedCtx ln_inv_sqrt_tail(Inference& inf, PackedCtx xs, const NormConfig& cfg,
                           int rD, int t, double c_eff_sq, bool sparse_var_scope,
                           bool tp_probes);

// Token-pair LN adapter (token_pair_norm.cu): conj_split -> shared core per half
// (per-half capture_t/n_tok) -> repack. The payload axis, not a packing: the body
// uses only packing-dispatched helpers. Dispatched at norm() entry on inf.token_pair.
PackedCtx norm_token_pair(Inference& inf, const PackedCtx& x, const std::string& cfg_name);

PackedCtx exp_approx(Inference& inf, const PackedCtx& x, int r);

struct SoftmaxConfig {
    GSInitMethod gs_init_method = GSInitMethod::LINEAR;

    int    log2delta1 = 0;
    int    log2delta2 = 0;
    double clip_lo    = 0.0;
    double clip_hi    = 0.0;

    std::vector<double> poly_coeffs;     // exp polynomial (per-layer fit)

    double init_alpha       = 0.0;
    double init_beta        = 0.0;
    int    gs_iters_scaled  = 0;

    std::vector<double> refine_alpha;    // length = log2delta2
    std::vector<double> refine_beta;
    int    gs_iters_refine_scaled = 0;   // max over steps (legacy aggregate)
    std::vector<double> per_step_refine_iters;

    // Per-step per-kc refine GS init scaling
    std::vector<double> sm_kc_r;

    std::vector<double> cheb_coeffs;     // exp(y) Chebyshev coeffs, normalized so series(0)=1
    double cheb_a = 0.0;                 // = -delta0/2 (poly input lower bound)
    double cheb_b = 0.0;                 // = +delta0/2 (poly input upper bound)
};

struct GeLUConfig {
    GeLUMethod method = GeLUMethod::SOFTSIGN_INV_SQRT;

    bool gate = true;          // P3: false = plain softsign (skip the b·exp(-c·x²) gate + its exp)

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
    double inv_out_scale = 1.0;   // s: softsign inv_sqrt target is s·z^{-1/2}; g folds 1/s (keeps output above bts floor)
    std::vector<double> Ncoeffs;          // Remez numerator
    std::vector<double> Dcoeffs;          // Remez denominator

    std::vector<double> cheb_coeffs;        // whole-GELU cheb (CHEBYSHEV method)
    double cheb_a = 0.0;
    double cheb_b = 0.0;

    std::vector<double> thor_p1;            // THOR_COMPOSITE stage-1 poly (deg 31, ascending); S = xmax
    std::vector<double> thor_p2;            // THOR_COMPOSITE stage-2 poly ≈ tanh/2 (deg 27, ascending)

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
    std::vector<double> chord_a, chord_b;   // pass-0 cascade chord init per iter
    std::vector<double> cascade_iters;      // per-iter scalar bts regime (1|2)
};

PackedCtx gelu_approx(Inference& inf, PackedCtx& x, const std::string& cfg_name);

// Shared GELU cores (nonlinear.cu): prescaled input -> gate factor. The output
// x-mult + mask pairing stays with the caller (real: im_cleanse'd x + 0.25/0.5;
// token-pair: raw conj-split half + 0.25/0.5).
PackedCtx gelu_softsign_core(Inference& inf, PackedCtx x2, const GeLUConfig& cfg);
PackedCtx gelu_thor_core   (Inference& inf, PackedCtx t,  const GeLUConfig& cfg);

// Token-pair GELU adapter (token_pair_nonlinear.cu) — see norm_token_pair.
PackedCtx gelu_token_pair(Inference& inf, const PackedCtx& x, const std::string& cfg_name);