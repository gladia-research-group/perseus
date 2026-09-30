#include "ckks_fixture.h"
#include "ckks_primitives.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace test_helpers;

namespace {

using CompositeBootstrapTest = CkksFixture;

// Gate for the NATIVEINT=32 COMPOSITESCALING port:
// fresh encrypt -> GPU bootstrap -> decrypt on the composite prod-shaped chain
// (COMPOSITE_DEGREE=2, 2x27-bit primes, 56 q-towers + 8 specials at dnum=7 = exactly the
// 64-prime knife-edge). This deliberately AVOIDS the perseus wrapper's level-prediction
// helpers (encode_at / drain / level_limit) — they still carry ±1-level assumptions that
// are wrong under composite (prime-granular levels). Fresh top-level encrypts are on the
// composite level grid by construction, and the bootstrap's own ModRaise handles the drop
// to the bottom.
//
// PASS bar: the CPU-validated references are 10.8 bits (UNIFORM_TERNARY) and 19.2 bits
// (SPARSE_ENCAPSULATED) at logN=12; the gate accepts > 8.0 bits (same bar as the 64-bit
// sweeps — comfortably above the parameter-limited garbage regime at ~-14 bits, below
// ring-size wiggle). COMPOSITE_BTS_MIN_BITS overrides.

std::vector<double> varied_pattern(int n_slots, int n_active, double amp, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> lg(-4.0, 0.0);  // 1e-4 .. 1 of amp
    std::vector<double> x(n_slots, 0.0);
    for (int i = 0; i < n_active && i < n_slots; ++i)
        x[i] = amp * std::pow(10.0, lg(gen));
    return x;
}

double worst_err(const std::vector<double>& got, const std::vector<double>& want) {
    double e = 0.0;
    int nnan = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) {
            ++nnan;
            continue;
        }
        e = std::max(e, std::fabs(got[i] - want[i]));
    }
    if (nnan > 0) {
        std::cout << "[composite_bts] NON-FINITE slots: " << nnan << " / " << got.size() << "\n";
        return std::numeric_limits<double>::infinity();
    }
    return e;
}

double env_num(const char* k, double dflt) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::atof(v) : dflt;
}

TEST_F(CompositeBootstrapTest, FreshEncryptBootstrap) {
    const int S = slots();
    const int n_active = S / 2;
    const double min_bits = env_num("COMPOSITE_BTS_MIN_BITS", 8.0);
    // Divergence probes: COMPOSITE_BTS_CPU_REF=1 also runs the pure-CPU
    // FHECKKSRNS::EvalBootstrap on the IDENTICAL input ct (same keys/payload) — the honest
    // per-run reference for the GPU-vs-CPU gap. COMPOSITE_BTS_ITERS>1 runs the wrapper's
    // iterative (residual-refined) GPU bootstrap with COMPOSITE_BTS_PREC bits of residual
    // scaling — tells us how much of the floor iterating recovers.
    const int iters = (int)env_num("COMPOSITE_BTS_ITERS", 1);
    const int prec = (int)env_num("COMPOSITE_BTS_PREC", 8);
    const bool cpu_ref = env_num("COMPOSITE_BTS_CPU_REF", 0) > 0;
    // Payload amplitude (default 1.0). This test measures a SINGLE fresh bootstrap —
    // unlike a chained measurement, whose cells chain two bootstraps and are floored
    // by the first pass's 1-iter error. Use THIS test for per-bootstrap operating points.
    const double bts_amp = env_num("COMPOSITE_BTS_AMP", 1.0);

    // Chain shape sanity: print what we actually built (the composite facts live on the
    // OpenFHE side; the [ckks_params] banner from the fixture shows towers/scale).
    const auto want = varied_pattern(S, n_active, bts_amp, 0xC0FFEEu);

    // fresh top-level encrypt: on the composite level grid by construction
    Ctx ct = encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk());
    {
        auto pre = decrypt_slots(fhe(), ct);
        pre.resize(want.size());
        const double e = worst_err(pre, want);
        std::cout << "[composite_bts] pre-bootstrap roundtrip err_max = " << e << " ("
                  << (e > 0 ? -std::log2(e) : 64.0) << " bits)\n";
        ASSERT_LT(e, 1e-2) << "encode/encrypt/decrypt broken before any bootstrap";
    }

    if (cpu_ref) {
        // Pure-CPU OpenFHE bootstrap on the IDENTICAL input ct: the api wrapper stores the
        // lbcrypto context/ct/sk in `.cpu`/`.pimpl` std::any members (same casts as
        // test_fideslib_wrapper.cu MultDiagLevel0).
        auto& occ = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(fhe().cc->cpu);
        auto& octIn = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto& skImpl =
            std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(fhe().sk()->pimpl);
        // FIDESlib's GenBootstrapKeys only generates the GPU's own rotation indexes; the
        // pure-CPU EvalBootstrap needs OpenFHE's full CtS/StC automorphism-key set (else
        // "EvalKey for index [...] not found"). Generate them here (CPU-only, ~1 min).
        occ->EvalBootstrapKeyGen(skImpl, S);
        const auto tc0 = std::chrono::steady_clock::now();
        auto ctCpu = occ->EvalBootstrap(octIn, 1, 0);
        const double cms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc0)
                .count();
        lbcrypto::Plaintext ptCpu;
        occ->Decrypt(skImpl, ctCpu, &ptCpu);
        ptCpu->SetLength(S);
        auto gotCpu = ptCpu->GetRealPackedValue();
        gotCpu.resize(want.size());
        const double ec = worst_err(gotCpu, want);
        std::cout << "[composite_bts] CPU-ref (same ct/keys) err_max = " << ec
                  << "   bits = " << (ec > 0 ? -std::log2(ec) : 64.0) << "   (" << cms
                  << " ms)\n";
    }

    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    Ctx bts = fhe().eval_bootstrap_iter(ct, iters, prec);
    cudaDeviceSynchronize();
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    auto got = decrypt_slots(fhe(), bts);
    got.resize(want.size());
    const double e = worst_err(got, want);
    const double bits = (e > 0 && std::isfinite(e)) ? -std::log2(e) : (e == 0 ? 64.0 : -999.0);

    std::cout << std::setprecision(10) << "[composite_bts] bootstrap wall = " << ms << " ms\n"
              << "[composite_bts] out level (OpenFHE, primes dropped) = " << bts->GetLevel()
              << "  noiseScaleDeg = " << bts->GetNoiseScaleDeg() << "\n"
              << "[composite_bts] err_max = " << e << "   bits = " << bits << "\n"
              << "[composite_bts] got[0..3]  = " << got[0] << " " << got[1] << " " << got[2] << " "
              << got[3] << "\n"
              << "[composite_bts] want[0..3] = " << want[0] << " " << want[1] << " " << want[2]
              << " " << want[3] << "\n";

    EXPECT_GT(bits, min_bits) << "composite bootstrap below the quality bar (CPU refs: "
                                 "10.8 bits uniform / 19.2 bits sparse-encapsulated)";
}

// Second bootstrap in a chain: the first bootstrap's OUTPUT state (deg-2, mid-chain level)
// is what the model actually feeds its bootstraps; this catches level-grid bookkeeping
// mistakes the fresh-encrypt case can't (the raise sees a post-bootstrap ciphertext).
TEST_F(CompositeBootstrapTest, SecondBootstrap) {
    const int S = slots();
    const int n_active = S / 2;
    const double min_bits = env_num("COMPOSITE_BTS_MIN_BITS", 8.0);

    const auto want = varied_pattern(S, n_active, 1.0, 0xBEEF01u);
    Ctx ct = encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk());

    Ctx b1 = fhe().eval_bootstrap_iter(ct, 1, 0);
    Ctx b2 = fhe().eval_bootstrap_iter(b1, 1, 0);

    auto got = decrypt_slots(fhe(), b2);
    got.resize(want.size());
    const double e = worst_err(got, want);
    const double bits = (e > 0 && std::isfinite(e)) ? -std::log2(e) : (e == 0 ? 64.0 : -999.0);
    std::cout << "[composite_bts] second-bootstrap err_max = " << e << "   bits = " << bits
              << "\n";
    EXPECT_GT(bits, min_bits) << "second (chained) composite bootstrap below the quality bar";
}

// Mixed correction factors in ONE execution (CorrectionScope): the correction
// factor is runtime-only, so different bootstraps may use different values — CF picks the
// range<->precision point PER BOOTSTRAP (CF=3 ~ 13.5 bits @ |m|<=1; CF=7 ~ 9 bits @ |m|<=10).
// Build the context with CORRECTION_FACTOR=7 (the precomp default); the middle bootstrap
// overrides to CF=3 via the scope; the last run proves the restore.
TEST_F(CompositeBootstrapTest, MixedCorrectionFactors) {
    const int S = slots();
    const int n_active = S / 2;

    auto one = [&](double amp, uint32_t seed, const char* tag) {
        const auto want = varied_pattern(S, n_active, amp, seed);
        Ctx ct = encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk());
        Ctx b = fhe().eval_bootstrap_iter(ct, 1, 0);
        auto got = decrypt_slots(fhe(), b);
        got.resize(want.size());
        const double e = worst_err(got, want);
        const double bits = (e > 0 && std::isfinite(e)) ? -std::log2(e) : 64.0;
        std::cout << "[mixed_cf] " << tag << ": amp=" << amp << " err_max=" << e
                  << " bits=" << bits << "\n";
        return bits;
    };

    const double a = one(10.0, 0xA11CE01u, "CF=7 (precomp), |m|<=10");
    double b;
    {
        CKKSContext::CorrectionScope cs(fhe(), 3);
        {   // diagnostic: confirm the override armed (via the library-side accessor —
            // direct field reads can hit the wrong offset, see CorrectionScope note)
            auto& gctx = std::any_cast<FIDESlib::CKKS::Context&>(fhe().cc->gpu);
            std::cout << "[mixed_cf] scope armed: override="
                      << gctx->getCorrectionFactorOverride() << " cc=" << (void*)gctx.get()
                      << "\n";
        }
        b = one(1.0, 0xA11CE02u, "CF=3 (scoped),  |m|<=1 ");
    }
    const double c = one(10.0, 0xA11CE03u, "CF=7 (restored),|m|<=10");

    EXPECT_GT(a, 7.5) << "CF=7 baseline bootstrap below its operating point";
    EXPECT_GT(b, 11.5) << "scoped CF=3 bootstrap did not reach the CF=3 operating point";
    EXPECT_GT(c, 7.5) << "post-scope bootstrap regressed: override not restored";
    // and the scoped run must be visibly BETTER than the CF=7 runs at its amplitude:
    EXPECT_GT(b, a + 1.5) << "scope had no effect (still at the CF=7 operating point?)";
}

}  // namespace
