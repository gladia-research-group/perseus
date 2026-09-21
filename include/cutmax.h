#pragma once

#include "inference.h"
#include "nonlinear.h"   // CutMaxCalib (the configs.json "cutmax" section)

#include <vector>

// CutMax — comparison-free CKKS argmax over the lm_head logit tiles.
// Source: "Efficient Decoding Methods for Language Models on Encrypted Data"
// (arXiv:2509.08383), CKKS-safe reformulation per docs/fhe_argmax_cutmax.md
// and the Phase-0 oracle (scripts/cutmax_oracle.py, T128 head oracle).
//
// Terminal decode op: consumes REAL-packed lm_head tiles (logit column
// j = interleave(m) at slot m, zero pad slots), returns tiles holding a
// ~one-hot Z over the vocab (top slot ≈ 1). Does not touch the verified
// decode path; everything here runs after the last planned op, eager.
//
// CKKS-safe deviations from the paper (all validated in the oracle):
//  - per-iteration constant rescale 1/m folded into the standardize
//    constants (standardization is scale-invariant -> mathematical no-op)
//    pins every intermediate, incl. the power-chain squarings, to <= ~1;
//  - sigma^-1 via cascaded range-reduction Newton (from-below y0=1,
//    newton_per_pass iters/pass, x <- x*u^2 between passes): covers any
//    band ratio with scalar-lane values <= ~7.6; the product telescopes
//    to x^{-1/2} wherever the LAST pass converged;
//  - gentle adaptive powers p (13,3,3,5,13,13) instead of the paper's
//    19/19/13/9: keeps slow tokens above the bootstrap noise floor;
//  - ONE explicit vector bootstrap per iteration, at the shift point
//    (values O(1), slot spread ~1/(c*m)); the powered vector is never
//    bootstrapped (its spread is crushed -> noise there flips argmax).
//    Requires bts-noise <= ~2e-4 at shift points: run BTS_ITERATIONS=2.
struct CutMaxConfig {
    struct Iter {
        int    p;        // odd amplification power (p=19 needs ex2)
        double c;        // standardization divisor
        double m;        // shift rescale (1.15 * (1 + w_hi) from oracle)
        double s2_hi;    // oracle band top: cascade prescale x = s2/s2_hi
        int    passes;   // range-reduction passes (each covers ~25x)
        int    ex2;      // 1: sigma^2 ~= E[y^2] (skip the mu-subtraction;
                         // valid once mu^2 << s2, i.e. concentrated
                         // iterations). Saves 2 vector levels -> p=19 fits
                         // one bootstrap window, and the concentrated state
                         // also tolerates post-power auto-bootstraps.
        double ca = 0.0; // pass-0 calibrated from-below CHORD init of the
        double cb = 0.0; // cascade: y0 = ca - cb*x (0 = legacy const init;
                         // pairs with a budget-clamped s2_hi from the calib)
        int casc_iters = 0;  // scalar-lane bts iters for THIS iteration
                             // (mixed precision; 0 = legacy rule)
    };
    std::vector<Iter> iters;
    int    newton_per_pass = 4;   // from-below iters/pass: y <= 1.5^4 = 5.06
                                  // by hard induction, so 1.5y <= 7.6 and the
                                  // per-pass u bootstrap NEVER sees the bts=1
                                  // EvalMod wall (10), for ANY input
    int    newton_polish   = 0;   // polish let an unconverged last pass reach
                                  // 1.5^6 = 11.4 > wall (live: position 96);
                                  // accuracy comes from the +1 safety pass
                                  // in the schedule instead
    double sum_lo = 0.0;          // final masked-sum band -> reciprocal init
    double sum_hi = 0.0;
    int    gs_sum_iters = 4;
    double entry_scale = 1.0 / 256.0;   // |logits| <= ~340 -> <= 1.33; i0.s2_hi
                                        // must carry the matching entry^2 fold
};

// Oracle-locked schedule (cutmax_oracle.py on all_blocks_lm_head_steps_T128):
// 128/128 argmax recovery at shift-point noise <= 2e-4, top mass >= 0.997.
CutMaxConfig default_gpt2_cutmax_config();

// configs.json "cutmax" section (calibrate.py cutmax_calibrate=true) -> runtime
// config. The calib carries FINAL constants (i0.s2_hi already entry^2-folded).
CutMaxConfig cutmax_config_from_calib(const CutMaxCalib& c);

// tiles: K real-packed lm_head tiles (K = ceil(vocab / slots)), pad slots
// ~0 (|pad| << 1e-6; zero weight columns guarantee this). Returns Z tiles in
// the same layout. Eager-mode only (runs after the plan is cleared).
std::vector<PackedCtx> cutmax_argmax(Inference& inf,
                                     const std::vector<PackedCtx>& tiles,
                                     int vocab, const CutMaxConfig& cfg);

// slot -> logit-column permutation of a full-width cachemir "up" tile
// (mirrors cachemir_linear_utils interleave_idx for d=hidDim, dim=W_tile).
inline int cutmax_tile_col_of_slot(int m, int d, int W_tile) {
    const int a = W_tile / d;
    return (m / a + (m % a) * d) % W_tile;
}
