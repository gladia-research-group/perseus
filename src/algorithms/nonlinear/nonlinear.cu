#include "ckks_primitives.h"
#include "slot_layout.h"
#include "inference.h"
#include "nonlinear.h"
#include "model/layer_norm.h"   // fold_ln_affine
#include "packing/cachemir_filling/cachemir_filling.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Element-wise packing-agnostic operations.

/// @brief Encrypted sign approximation via composite minimax polynomials.
PackedCtx sign(Inference& inf, const PackedCtx& x) {
    const int S = inf.slots;

    const std::vector<double> F4 = {
        0.0,
         315.0 / 128.0,
         0.0,
        -420.0 / 128.0,
         0.0,
         378.0 / 128.0,
         0.0,
        -180.0 / 128.0,
         0.0,
          35.0 / 128.0
    };

    const std::vector<double> G4 = {
        0.0,
         5850.0 / 1024.0,
         0.0,
        -34974.0 / 1024.0,
         0.0,
         97015.0 / 1024.0,
         0.0,
       -113492.0 / 1024.0,
         0.0,
         46623.0 / 1024.0
    };

    // Composition: F4(F4(G4(G4(x))))
    PackedCtx h = eval_polynomial_ps(inf.cc_ctx(), x, G4, (size_t)S);
    h           = eval_polynomial_ps(inf.cc_ctx(), h, G4, (size_t)S);
    h           = eval_polynomial_ps(inf.cc_ctx(), h, F4, (size_t)S);
    h           = eval_polynomial_ps(inf.cc_ctx(), h, F4, (size_t)S);

    return h;
}
/// @brief exp(x) ≈ (1 + x / 2^r)^{2^r}.
PackedCtx exp_approx(Inference& inf, const PackedCtx& x, int r) {
    double inv_2r = 1.0 / (double)(1 << r);
    PackedCtx y = inf.fhe->mult(x, inv_2r);
    inf.fhe->inplace_add(y, 1.0);

    for (int i = 0; i < r; ++i)
        inf.fhe->inplace_square(y);
    return y;
}

PackedCtx gelu_softsign_core(Inference& inf, PackedCtx x2, const GeLUConfig& cfg) {
    inf.fhe->bootstrap_hint(x2, 16);
    const double h = 0.5;

    PackedCtx g = inf.fhe->mult(x2, h * cfg.xmax * cfg.a);
    PackedCtx axsq = inf.fhe->square(g);
    PackedCtx z = inf.fhe->add(axsq, 1.0);

    WithStep _w2(inf, "remez_init");
    PackedCtx inv = eval_remez_31(inf.cc_ctx(), z, cfg.Ncoeffs, cfg.Dcoeffs,
                                  cfg.lin_alpha, cfg.lin_beta, cfg.gs_iters);
    _w2.next("inv_sqrt_newton");

    inv = inv_sqrt_newton(inf.cc_ctx(), z, inv, cfg.newton_iters,
                          1.0 / (cfg.inv_out_scale * cfg.inv_out_scale));

    _w2.next("inv_sqrt_mult");
    inf.fhe->bootstrap_hint(inv, inf.fhe->level_headroom(4));

    PackedCtx g_over_s = inf.fhe->mult(x2, h * cfg.xmax * cfg.a / cfg.inv_out_scale);   // = g / s
    z = inf.fhe->mult(g_over_s, inv);

    if (cfg.gate) {
        PackedCtx gy = eval_chebyshev_series(inf.cc_ctx(), axsq, cfg.gate_cheb_coeffs,
                                             cfg.gate_cheb_a, cfg.gate_cheb_b);
        inf.fhe->bootstrap_hint(gy, 16);
        for (int i = 0; i < cfg.exp_iters; ++i)
            inf.fhe->inplace_square(gy);
        inf.fhe->inplace_mult(gy, -cfg.b);
        inf.fhe->inplace_add(gy, 1.0);
        z = inf.fhe->mult(z, gy);
    }

    inf.fhe->inplace_add(z, 1.0);
    return z;
}

static PackedCtx gelu_softsign_inv_sqrt(Inference& inf, PackedCtx& x,
                                        const GeLUConfig& cfg) {
    WithStep _w(inf, "gelu_softsign_inv_sqrt");
    PackedCtx x2 = inf.fhe->mult(x, 1 / cfg.xmax);
    PackedCtx z = gelu_softsign_core(inf, std::move(x2), cfg);
    {
        WithStep _wc(inf, "im_cleanse");
        PackedCtx xs = inf.fhe->im_cleanse(x);   // 2*Re(x)
        z = inf.fhe->mult(z, xs);
        Ptx quarter_mask = inf.gelu_half_mask(z, 0.25);          // 0.5 (gelu) * 0.5 (conj)
        PackedCtx out = inf.fhe->mult(z, quarter_mask);
        return out;
    }
}

static PackedCtx gelu_chebyshev(Inference& inf, PackedCtx& x,
                                const GeLUConfig& cfg) {
    WithStep _w(inf, "gelu_chebyshev");
    inf.fhe->bootstrap_hint(x, inf.fhe->level_headroom(4));
    return eval_chebyshev_series(inf.cc_ctx(), x, cfg.cheb_coeffs,
                                 cfg.cheb_a, cfg.cheb_b);
}

PackedCtx gelu_thor_core(Inference& inf, PackedCtx t, const GeLUConfig& cfg) {
    inf.fhe->bootstrap_hint(t, 16);

    const bool cheb = !cfg.thor_p1_cheb.empty();

    WithStep _w2(inf, "thor_p1");
    PackedCtx p1 = cheb
        ? eval_chebyshev_series(inf.cc_ctx(), t, cfg.thor_p1_cheb,
                                cfg.thor_p1_a, cfg.thor_p1_b)
        : eval_polynomial_ps(inf.cc_ctx(), t, cfg.thor_p1, (size_t)inf.slots);
    inf.fhe->bootstrap_hint(p1, inf.fhe->level_headroom(4));  // bounded composite output

    _w2.next("thor_p2");
    PackedCtx g = cheb
        ? eval_chebyshev_series(inf.cc_ctx(), p1, cfg.thor_p2_cheb,
                                cfg.thor_p2_a, cfg.thor_p2_b)
        : eval_polynomial_ps(inf.cc_ctx(), p1, cfg.thor_p2, (size_t)inf.slots);
    inf.fhe->inplace_add(g, 0.5);                            // g = P2(P1(x/S)) + 1/2 ≈ Φ(x)

    if (inf.fhe->composite_degree > 1)
        inf.fhe->bootstrap_hint(g, static_cast<int>(inf.fhe->level_limit()) - 4);
    return g;
}

static PackedCtx gelu_thor_composite(Inference& inf, PackedCtx& x,
                                     const GeLUConfig& cfg) {
    WithStep _w(inf, "gelu_thor_composite");
    PackedCtx t = inf.fhe->mult(x, 1.0 / cfg.xmax);          // t = x/S ∈ [-1,1]
    PackedCtx g = gelu_thor_core(inf, std::move(t), cfg);

    {
        WithStep _wc(inf, "im_cleanse");
        PackedCtx xs = inf.fhe->im_cleanse(x);               // 2·Re(x)
        g = inf.fhe->mult(g, xs);
        Ptx half = inf.gelu_half_mask(g, 0.5);               // 0.5 cancels conj-doubling ⇒ GELU = x·g
        PackedCtx out = inf.fhe->mult(g, half);
        return out;
    }
}

PackedCtx gelu_approx(Inference& inf, PackedCtx& x, const std::string& cfg_name) {
    if (is_cachemir_filling(x.packing) && inf.token_pair)
        return gelu_token_pair(inf, x, cfg_name);
    WithStep _w(inf, "gelu:" + cfg_name);
    const GeLUConfig& cfg = inf.gelu_cfg.at(cfg_name);
    // Elementwise: the slot BASIS survives (slot_layout.h).
    auto keep = [&](PackedCtx y) { slotlayout::propagate(x.ct, y.ct); return y; };
    switch (cfg.method) {
        case GeLUMethod::SOFTSIGN_INV_SQRT:
            return keep(gelu_softsign_inv_sqrt(inf, x, cfg));
        case GeLUMethod::CHEBYSHEV:
            return keep(gelu_chebyshev(inf, x, cfg));
        case GeLUMethod::THOR_COMPOSITE:
            return keep(gelu_thor_composite(inf, x, cfg));
    }
    throw std::runtime_error("gelu_approx: unhandled GeLUMethod");
}
