#include "ckks_fixture.h"
#include "ckks_primitives.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

using namespace test_helpers;

namespace {

using BtsPrecisionSweepTest = CkksFixture;

// Realistic single-iteration bootstrap precision + timing grid over the PRODUCTION
// chain (run via scripts/19_mixedlimb_btsprec.sh -> 128-bit mixed-limb params,
// AUTO_BTS_LEVEL=25).  Each cell reproduces the model's bootstrap input STATE:
//
//     encrypt(pattern) -> FIRST bootstrap -> drain (raw mult-by-1) to the target
//     input level -> MEASURED bootstrap (timed).
//
// The first bootstrap establishes the real post-bts noise/degree the model feeds
// its bootstraps (a fresh-encrypt-then-drain input is unrealistically clean), and
// the measured bootstrap is the SECOND in the chain — the one the model actually
// runs.  Rows = measured-bootstrap input level (+deg, +wall ms), columns = input
// |abs|; cell = precision in bits (-log2 worst-slot error).  Override the level
// span with LEVEL_MIN / LEVEL_MAX.

std::vector<double> varied_pattern(int n_slots, int n_active, double amp, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> lg(-4.0, 0.0);   // 1e-4 .. 1 of amp
    std::vector<double> x(n_slots, 0.0);
    for (int i = 0; i < n_active && i < n_slots; ++i)
        x[i] = amp * std::pow(10.0, lg(gen));
    return x;
}

double worst_err(const std::vector<double>& got, const std::vector<double>& want) {
    double e = 0.0;
    const int N = static_cast<int>(got.size());
    for (int i = 0; i < N; ++i) e = std::max(e, std::fabs(got[i] - want[i]));
    return e;
}

double to_bits(double err) { return err <= 0.0 ? 64.0 : -std::log2(err); }

Ctx bts1(CKKSContext& fhe, const Ctx& ct) { return fhe.eval_bootstrap_iter(ct, 1, 0); }

// MEASURED bootstrap: honours the configured BTS_ITERATIONS / BTS_PRECISION so a
// meta-bootstrap (iters>=2, Bae-Cheon residual refinement) chain can be swept.
// iters==1 -> identical to bts1 (eval_bootstrap_iter ignores precision and
// returns after a single EvalBootstrap), so a default run reproduces the 1-iter
// baseline (job 46658870) byte-for-byte.
Ctx bts_measured(CKKSContext& fhe, const Ctx& ct) {
    return fhe.eval_bootstrap_iter(ct, fhe.bts_iterations, fhe.bts_precision);
}

// RAW level drain: cc->EvalMult bypasses the wrapper's auto-bootstrap so we can
// reach any input level, including the ceiling, without an interposed refresh.
Ctx drain_to(CKKSContext& fhe, Ctx ct, int target) {
    int guard = 0;
    while (static_cast<int>(level_of(ct)) < target && guard++ < 96)
        ct = fhe.cc->EvalMult(ct, 1.0);
    return ct;
}

int env_int(const char* k, int dflt) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::atoi(v) : dflt;
}

TEST_F(BtsPrecisionSweepTest, Grid) {
    const int S        = slots();
    const int n_active = S / 2;
    const std::vector<double> amps = {
        1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0,
        1e1, 1e2, 1e3, 1e4, 1e5, 1e6,
    };
    const int bts_out = static_cast<int>(fhe().bootstrap_output_level());
    const int lvl_min = env_int("LEVEL_MIN", bts_out);
    const int lvl_max = env_int("LEVEL_MAX", static_cast<int>(fhe().level_limit()));

    const uint32_t m_iters = fhe().bts_iterations;
    const uint32_t m_prec  = fhe().bts_precision;
    std::cout << "\n[bts_grid] realistic MEASURED bootstrap (iters=" << m_iters
              << " precision=" << m_prec << "): encrypt -> first bts (1-iter) -> drain "
                 "to level -> MEASURED bts\n"
              << "  slots=" << S << " n_active=" << n_active
              << " bts_out_level=" << bts_out
              << " level_limit=" << fhe().level_limit()
              << " levels=" << lvl_min << ".." << lvl_max << "\n"
              << "  rows = measured-bts input level (deg, wall ms); cols = input |abs|; "
                 "cell = bits (-log2 worst err)\n\n";

    std::cout << std::left << std::setw(6) << "lvl" << std::setw(5) << "deg"
              << std::setw(9) << "bts_ms";
    for (double A : amps) {
        std::ostringstream h; h << A;
        std::cout << std::right << std::setw(8) << h.str();
    }
    std::cout << "\n" << std::string(20 + 8 * static_cast<int>(amps.size()), '-') << "\n";

    double corner_bits = -1.0;
    for (int L = lvl_min; L <= lvl_max; ++L) {
        int row_lvl = L, row_deg = 0;
        double row_ms = -1.0;
        std::ostringstream cells;
        for (double A : amps) {
            auto want = varied_pattern(S, n_active, A, 0xC0FFEEu);
            std::ostringstream cell;
            try {
                // realistic input state: first bootstrap, then consume to level L
                Ctx ct = bts1(fhe(), encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk()));
                ct = drain_to(fhe(), ct, L);
                row_lvl = static_cast<int>(level_of(ct));
                row_deg = static_cast<int>(ct->GetNoiseScaleDeg());
                // measured (second) bootstrap, timed — honours configured iters/prec
                cudaDeviceSynchronize();
                auto t0 = std::chrono::steady_clock::now();
                Ctx b = bts_measured(fhe(), ct);
                cudaDeviceSynchronize();
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                if (A == 1.0) row_ms = ms;
                auto out = decrypt_slots(fhe(), b);
                out.resize(want.size());
                const double bits = to_bits(worst_err(out, want));
                if (L == lvl_min && A == 1.0) corner_bits = bits;
                cell << std::fixed << std::setprecision(1) << bits;
            } catch (const std::exception&) {
                cell << "X";
            }
            cells << std::right << std::setw(8) << cell.str();
        }
        std::cout << std::left << std::setw(6) << row_lvl << std::setw(5) << row_deg;
        if (row_ms >= 0) std::cout << std::right << std::setw(7) << std::fixed
                                   << std::setprecision(1) << row_ms << "  ";
        else             std::cout << std::right << std::setw(9) << "-";
        std::cout << cells.str() << "\n";
        std::cout.flush();
    }
    EXPECT_GT(corner_bits, 8.0) << "1-iter bootstrap lost precision at the lowest level / |abs|~1";

    // ---- PROOF of the absolute bts accuracy floor ----------------------------------
    // Bootstrap a UNIFORM input (all slots = A) once, sweeping A down. If the floor is
    // ABSOLUTE, abs_err stays ~constant while A shrinks, so rel_err = abs_err/A blows
    // past 1 once A drops below the floor -> the signal is lost (it IS a floor, not
    // relative precision). The floor value is where abs_err ≈ A (rel_err ≈ 1).
    std::cout << "\n[bts_floor] uniform-A bootstrap (input level " << bts_out
              << ", iters=" << m_iters << " precision=" << m_prec
              << "): abs_err ~constant => absolute floor; rel_err>1 => signal lost\n"
              << std::left << std::setw(12) << "  A"
              << std::setw(14) << "abs_err" << std::setw(14) << "rel_err"
              << "bits(-log2 abs)\n";
    double floor_abs = -1.0;
    for (double A : {1000.0, 100.0, 50.0, 10.0, 1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6}) {
        std::vector<double> u(S, A);
        double abs_err = -1.0;
        try {
            Ctx b = bts_measured(fhe(), encrypt(fhe().cc, encode(fhe().cc, u), fhe().pk()));
            auto out = decrypt_slots(fhe(), b);
            out.resize(S);
            abs_err = 0.0;
            for (int i = 0; i < S; ++i) abs_err = std::max(abs_err, std::fabs(out[i] - A));
        } catch (const std::exception&) {
            std::cout << "  A=" << A << "  THREW\n";
            continue;
        }
        if (A == 1.0) floor_abs = abs_err;   // the floor (best case, |abs|~1)
        std::ostringstream ra; ra << abs_err;
        std::ostringstream rr; rr << abs_err / A;
        std::cout << "  " << std::left << std::setw(10) << A
                  << std::setw(14) << ra.str() << std::setw(14) << rr.str()
                  << std::fixed << std::setprecision(1) << to_bits(abs_err)
                  << std::defaultfloat << "\n";
    }
    std::cout << "[bts_floor] absolute floor ~= " << floor_abs << " ("
              << std::fixed << std::setprecision(1) << to_bits(floor_abs)
              << std::defaultfloat << " bits) — nothing below this survives a bootstrap\n";
    std::cout.flush();
}

}  // namespace
