// PROBE: is the GELU poly-gate exp's "im_cleanse before the squares" actually needed?
//
// The gate is  gate = 1 - b*exp(-c x^2),  with exp built as (cheb_deg8(axsq))^(2^K)  (K = exp_iters
// squarings; axsq = (0.5*a*x)^2, matching nonlinear.cu:86-87). Production (nonlinear.cu, gate arm)
// squares gy either WITHOUT a cleanse (old) or, with the fix, cleanses gy to Re() BEFORE the squares
// (mirrors the softmax exp, cachemir_filling_attention.cu:167-188). Cleansing matters because a full-
// slot CKKS bootstrap PRESERVES the imaginary lane and injects its own ~9-bit imaginary floor; if gy
// is not real going into the K squares, each square folds that Im floor into Re as -Im^2, amplifying
// ~2^K.  This test reproduces the exp core in isolation (no production .cu touched) and runs BOTH:
//     A) NO cleanse  (old):            gy = cheb(axsq);                  bts; K*square
//     B) cleanse-before-square (fix):  gy = im_cleanse(0.5*cheb(axsq));  bts, cleanse; K*square
// against the exact exp(-c x^2), sweeping K.  max|A-B| is the PURE cleanse-induced divergence (A and B
// are identical in exact arithmetic).  B<<A vs exact  => cleanse is load-bearing;  A~=B => not needed.
// It also shows whether B stays finite past K>=9 (the calib gate_fhe_k_ceiling=8 cutoff) — i.e. whether
// the fix could re-enable the gate on the outlier blocks (2/10/11) that are currently gate-OFF.
//
// Disjoint: dense real slot layout, real bootstrap noise, reproduces only the exp core locally.

#include "model/gpt2.h"
#include "ckks_primitives.h"
#include "test_helpers.h"

#include <gtest/gtest.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace test_helpers;

namespace {

// base deg-8 Chebyshev coeffs of exp over [-gate_delta0/2, +gate_delta0/2] (gate_delta0=2.6);
// block-independent (calibrate.py _fit_exp_cheb / every configs.json gate_cheb_coeffs is identical).
const std::vector<double> kExpCheb = {
    1.4692777866520716, 1.5946586177027398, 0.4852346229922357,  0.10162900849586323,
    0.016177660703635088, 0.002074173396569981, 0.00022248073001973262,
    2.050511946498983e-05, 1.656366548692525e-06};
constexpr double kGateDelta0 = 2.6;

// Representative gate-ON block: config block 1, the K=8 ceiling block (biggest Im amplification).
constexpr double kA = 3.0299502800668927;
constexpr double kB = 0.6465146995108697;
constexpr double kC = 0.7823051385788389;

bool all_finite(const std::vector<double>& v) {
    for (double x : v) if (!std::isfinite(x)) return false;
    return true;
}
double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}
std::string fmt(bool finite, double e) {
    if (!finite) return "   NON-FINITE";
    std::ostringstream os;
    os << std::scientific << std::setprecision(3) << std::setw(12) << e;
    return os.str();
}

}  // namespace

TEST(GateExpCleanse, CleanseBeforeSquareAblation) {
    Inference inf = make_gpt2_inference({
        .ckks = {.bts_iterations = default_bts_iterations()},
    });

    // x sweep over the gate-varying region (exp(-c x^2): ~1 at 0, ~0 past |x| ~ 4/sqrt(c)).
    const int NP = 256;
    std::vector<double> x(NP), axsq(NP), exp_exact(NP);
    for (int i = 0; i < NP; ++i) {
        x[i]         = 8.0 * i / (NP - 1);                  // [0, 8]
        axsq[i]      = 0.25 * kA * kA * x[i] * x[i];        // (0.5*a*x)^2, matches nonlinear.cu:86-87
        exp_exact[i] = std::exp(-kC * x[i] * x[i]);
    }

    auto enc_real = [&](const std::vector<double>& v) {
        std::vector<double> slotv(inf.slots, 0.0);
        for (int i = 0; i < NP; ++i) slotv[i] = v[i];
        auto pt = inf.cc()->MakeCKKSPackedPlaintext(slotv, 1, 0);
        return inf.pack(encrypt(inf.cc(), pt, inf.fhe->pk()));
    };

    // One exp-gate variant: reproduce the exp core, return the decrypted exp (gy) over the sweep.
    auto run_exp = [&](int K, bool cleanse) {
        const double cheb_a = 0.5 * kGateDelta0 * std::pow(2.0, K) * (kA * kA / kC);   // calibrate.py:263
        std::vector<double> coeffs = kExpCheb;
        if (cleanse) for (double& c : coeffs) c *= 0.5;                                 // 0.5-fold
        PackedCtx gy = eval_chebyshev_series(inf.cc_ctx(), enc_real(axsq), coeffs, cheb_a, -cheb_a);
        inf.fhe->bootstrap(gy.ct);                       // full-slot bts: injects the imaginary floor
        if (cleanse) inf.fhe->inplace_im_cleanse(gy);    // 2*Re(0.5*root) = Re(root): real pre-square
        for (int i = 0; i < K; ++i) inf.fhe->inplace_square(gy);
        auto raw = decrypt_slots(inf, gy);
        return std::vector<double>(raw.begin(), raw.begin() + NP);
    };

    std::cout << "\n[gate-exp cleanse ablation]  block1 params  a=" << kA
              << "  b=" << kB << "  c=" << kC << "  (sweep x in [0,8], " << NP << " pts)\n"
              << "  K | max|A-exp|  (no cleanse) | max|B-exp|  (cleanse)  | max|A-B| (pure cleanse) | Afin Bfin\n"
              << "  --+-------------------------+-----------------------+-------------------------+---------\n";

    for (int K : {6, 7, 8, 9, 10, 12}) {
        auto gyA = run_exp(K, /*cleanse=*/false);
        auto gyB = run_exp(K, /*cleanse=*/true);
        const bool fA = all_finite(gyA), fB = all_finite(gyB);
        const double eA = fA ? max_abs_diff(gyA, exp_exact) : INFINITY;
        const double eB = fB ? max_abs_diff(gyB, exp_exact) : INFINITY;
        const double dAB = (fA && fB) ? max_abs_diff(gyA, gyB) : INFINITY;
        std::cout << "  " << std::setw(2) << K
                  << " | " << fmt(fA, eA)
                  << "            | " << fmt(fB, eB)
                  << "          | " << fmt(fA && fB, dAB)
                  << "            |  " << (fA ? "Y" : "N") << "    " << (fB ? "Y" : "N") << "\n";
    }
    std::cout << "\n[VERDICT] max|A-B| >> 0 and max|B-exp| < max|A-exp|  => cleanse-before-square is\n"
                 "          load-bearing. A~=B (both small)             => cleanse is not needed.\n"
                 "          B finite where A is NON-FINITE              => fix raises the K>=9 ceiling.\n";
    SUCCEED();
}
