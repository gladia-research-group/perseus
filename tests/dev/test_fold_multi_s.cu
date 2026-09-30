// Multi-s fold_bootstrap gate.
//
// Three algebraic identities the fused reductions rely on, each checked against the
// exact rotation-ladder reference on the SAME ciphertext:
//
//   1. DiagonalFusedVariance — fold at s=t (=32) with n_live=rD reproduces
//      diagonal::compute_per_token_var's ladder+1/rD result per token lane.
//   2. SoftmaxFoldDenominator — fold at s=tH (=512) with n_live=1 reproduces the
//      cross-block replication ladder's raw residue-class sums.
//   3. PreLadderEquivalence — rotate+add at strides s..s'/2 then fold at s' equals
//      the native fold at s (the fold_slots_for bridging rule).
//
// Build the context with SPARSE_BTS_SLOTS=512,32,1 (scripts/local_env.sh n32 default);
// tests skip if a needed precomp is absent.
#include "ckks_fixture.h"

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace test_helpers;

namespace {

using FoldMultiSTest = CkksFixture;

bool has_precomp(CKKSContext& F, uint32_t s) {
    return F.sparse_precomp_slots.count(s) > 0;
}

// Exact ladder reference on cleartext: out[j] = sum over i ≡ j (mod s) of x[i].
std::vector<double> class_sums(const std::vector<double>& x, int s) {
    std::vector<double> out(x.size(), 0.0);
    std::vector<double> per(s, 0.0);
    for (size_t i = 0; i < x.size(); ++i) per[i % s] += x[i];
    for (size_t i = 0; i < x.size(); ++i) out[i] = per[i % s];
    return out;
}

TEST_F(FoldMultiSTest, DiagonalFusedVariance) {
    CKKSContext& F = fhe();
    const int S = slots();
    const int t = 32, rD = 768, d_pad = 1024;
    if (!has_precomp(F, t)) GTEST_SKIP() << "no s=32 precomp (SPARSE_BTS_SLOTS)";
    ASSERT_EQ(S / d_pad, t);

    // diagonal layout: feature k of token lane tok at slot k*t + tok; 4 live lanes.
    const int n_tok = 4;
    std::mt19937 g(0xD1A6);
    std::normal_distribution<double> nd(0.0, 0.5);
    std::vector<double> x(S, 0.0);
    std::vector<double> want(n_tok, 0.0);   // per-token variance
    for (int tok = 0; tok < n_tok; ++tok) {
        double sumsq = 0.0;
        for (int k = 0; k < rD; ++k) {
            const double v = nd(g);
            x[(size_t)k * t + tok] = v;
            sumsq += v * v;
        }
        want[tok] = sumsq / rD;
    }

    Ctx sq = encrypt(F.cc, encode(F.cc, x), F.pk());
    F.inplace_square(sq);
    F.fold_bootstrap(sq, /*s=*/t, /*n_live=*/rD);
    auto got = decrypt_slots(F, sq);

    for (int tok = 0; tok < n_tok; ++tok) {
        // t-periodic result: token lane tok's variance sits at every slot ≡ tok (mod t)
        EXPECT_NEAR(got[tok], want[tok], 5e-2 * std::max(1.0, want[tok]))
            << "tok=" << tok;
        EXPECT_NEAR(got[tok + t], got[tok], 5e-2) << "not t-periodic, tok=" << tok;
    }
}

// The FREE prescale: at s=t the fold barely shrinks (copies≈n_live), so the model arm
// passes prescale=1/16 — EvalMod sees folded/16, recovery multiplies 16 back. Must
// reproduce the same variances, with LOWER error than the unprescaled fold at hot
// amplitudes (m³ law).
TEST_F(FoldMultiSTest, DiagonalFoldPrescale) {
    CKKSContext& F = fhe();
    const int S = slots();
    const int t = 32, rD = 768;
    if (!has_precomp(F, t)) GTEST_SKIP() << "no s=32 precomp (SPARSE_BTS_SLOTS)";

    std::mt19937 g(0x9E5CA1E);
    std::normal_distribution<double> nd(0.0, 2.0);   // hot: var ≈ 4, folded ≈ 3 unprescaled
    std::vector<double> x(S, 0.0);
    double sumsq = 0.0;
    for (int k = 0; k < rD; ++k) {
        const double v = nd(g);
        x[(size_t)k * t] = v;
        sumsq += v * v;
    }
    const double want = sumsq / rD;

    auto run = [&](double p) -> double {
        Ctx sq = encrypt(F.cc, encode(F.cc, x), F.pk());
        F.inplace_square(sq);
        F.fold_bootstrap(sq, t, rD, p);
        try {
            return std::fabs(decrypt_slots(F, sq)[0] - want);
        } catch (const std::exception& e) {
            std::cout << "[fold_prescale] p=" << p << " DECODE THROWS: " << e.what() << "\n";
            return -1.0;
        }
    };
    const double e_hot = run(1.0);
    double best = -1.0, best_p = 1.0;
    for (double p : {0.5, 0.25, 0.125, 1.0 / 16.0, 1.0 / 32.0}) {
        const double e = run(p);
        std::cout << "[fold_prescale] var=" << want << " p=" << p << " err="
                  << e << (e_hot >= 0 ? "" : " (unprescaled threw)") << "\n";
        if (e >= 0 && (best < 0 || e < best)) { best = e; best_p = p; }
    }
    std::cout << "[fold_prescale] err_unprescaled=" << e_hot
              << " best_p=" << best_p << " best_err=" << best << "\n";
    ASSERT_GE(best, 0.0) << "every prescale value failed to decode";
    EXPECT_LT(best, 5e-2 * std::max(1.0, want));
    if (e_hot >= 0) EXPECT_LT(best, e_hot);   // the whole point
}

TEST_F(FoldMultiSTest, SoftmaxFoldDenominator) {
    CKKSContext& F = fhe();
    const int S = slots();
    const int tH = 512;
    if (!has_precomp(F, tH)) GTEST_SKIP() << "no s=512 precomp (SPARSE_BTS_SLOTS)";

    std::mt19937 g(0x50F7);
    std::uniform_real_distribution<double> ud(0.0, 0.02);   // exp-mask-scaled scores
    std::vector<double> x(S, 0.0);
    for (int i = 0; i < S; ++i) x[i] = ud(g);
    const auto want = class_sums(x, tH);

    Ctx ct = encrypt(F.cc, encode(F.cc, x), F.pk());
    F.fold_bootstrap(ct, /*s=*/tH, /*n_live=*/1);   // recovery = S/tH restores raw sums
    auto got = decrypt_slots(F, ct);

    double emax = 0.0;
    for (int j = 0; j < tH; ++j) emax = std::max(emax, std::fabs(got[j] - want[j]));
    std::cout << "[fold512] max_err=" << emax << " sum0=" << want[0] << "\n";
    EXPECT_LT(emax, 5e-2 * std::max(1.0, want[0]));
}

TEST_F(FoldMultiSTest, PreLadderEquivalence) {
    // The fold_slots_for bridge: rotate+add at strides s..s'/2 then fold at s' computes
    // the SAME class sums mod s as a native fold at s (each live value counted once).
    // Checked against the exact cleartext reference. Amplitude matters: the pre-ladder
    // sums 16 members BEFORE the bootstrap, so the EvalMod input grows by that factor —
    // data is sized to keep it ≲0.1 (the copies² error-shrink difference between the
    // two routes is real and documented; this test is about the ALGEBRA, not the shrink).
    CKKSContext& F = fhe();
    const int S = slots();
    const int s_want = 32, s_eff = 512;
    if (!has_precomp(F, s_eff))
        GTEST_SKIP() << "needs the s=512 precomp";

    std::mt19937 g(0x913D);
    std::uniform_real_distribution<double> ud(0.0, 0.004);
    std::vector<double> x(S, 0.0);
    for (int i = 0; i < S; i += 8) x[i] = ud(g);
    const auto want = class_sums(x, s_want);
    const int n_live = 1;

    Ctx b = encrypt(F.cc, encode(F.cc, x), F.pk());
    for (int st = s_want; st < s_eff; st *= 2) {
        Ctx rot = F.rotate(b, st);
        F.inplace_add(b, rot);
    }
    F.fold_bootstrap(b, s_eff, n_live);   // recovery = S/512 restores raw class sums

    auto gb = decrypt_slots(F, b);
    double emax = 0.0;
    for (int j = 0; j < s_want; ++j) emax = std::max(emax, std::fabs(gb[j] - want[j]));
    // and the result must be s_want-periodic (slots j and j+s_want agree)
    double eper = 0.0;
    for (int j = 0; j < s_want; ++j) eper = std::max(eper, std::fabs(gb[j] - gb[j + s_want]));
    std::cout << "[preladder] max_err=" << emax << " periodicity_err=" << eper
              << " sum0=" << want[0] << "\n";
    EXPECT_LT(emax, 5e-2 * std::max(1.0, want[0]));
    EXPECT_LT(eper, 5e-2);
}

}   // namespace
