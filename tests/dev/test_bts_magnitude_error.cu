#include "ckks_fixture.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace test_helpers;

namespace {

using BtsMagnitudeErrorTest = CkksFixture;

struct ErrStats {
    double abs_max, abs_mean, abs_p99;
};

ErrStats err_stats(const std::vector<double>& ref, const std::vector<double>& got) {
    std::vector<double> e(ref.size());
    double sum = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        e[i] = std::fabs(got[i] - ref[i]);
        sum += e[i];
    }
    std::vector<double> s = e;
    std::sort(s.begin(), s.end());
    return {s.back(), sum / e.size(), s[(size_t)(0.99 * (s.size() - 1))]};
}

// BTS_SPARSE_PROBE=1: fold slot data to period sparse_bts_slots (an s-periodic
// slot vector IS a valid sparse ct) and route the bootstrap through
// SparseBtsScope — the dual-slots envelope measurement. Context must be built
// with SPARSE_BTS_SLOTS=s + FIDESLIB_SPARSE_ARCSINE=1.
bool sparse_probe() {
    static const bool on = [] {
        const char* v = std::getenv("BTS_SPARSE_PROBE");
        return v && *v && *v != '0';
    }();
    return on;
}

void fold_sparse(std::vector<double>& v, const CKKSContext& F) {
    const uint32_t s = F.sparse_bts_slots;
    if (!sparse_probe() || !s) return;
    for (size_t i = s; i < v.size(); ++i) v[i] = v[i % s];
}

void bts_probe(CKKSContext& F, Ctx& ct) {
    if (sparse_probe() && F.sparse_bts_slots) {
        CKKSContext::SparseBtsScope ss(F);
        F.bootstrap(ct);
    } else {
        F.bootstrap(ct);
    }
}


TEST_F(BtsMagnitudeErrorTest, ErrorVsAmplitude) {
    const std::vector<double> amps = {1, 5, 8, 10, 12, 15, 18, 20, 25, 30};
    const int reps = 3;

    for (double A : amps) {
        for (int r = 0; r < reps; ++r) {
            std::mt19937 gen(12345 + r);
            std::uniform_real_distribution<double> dist(-A, A);
            std::vector<double> v(slots());
            for (auto& x : v) x = dist(gen);
            fold_sparse(v, fhe());

            Ctx ct = encrypt(fhe().cc, encode(fhe().cc, v), fhe().pk());
            bts_probe(fhe(), ct);
            auto out = decrypt_slots(fhe(), ct);
            out.resize(v.size());

            ErrStats es = err_stats(v, out);
            std::cout << std::scientific << std::setprecision(3)
                      << "[bts_mag] A=" << std::fixed << std::setprecision(0) << A
                      << " rep=" << r << std::scientific << std::setprecision(3)
                      << " abs_max=" << es.abs_max
                      << " abs_p99=" << es.abs_p99
                      << " abs_mean=" << es.abs_mean << std::endl;
        }
    }
    SUCCEED();
}

TEST_F(BtsMagnitudeErrorTest, ErrorVsAmplitudeWide) {
    const std::vector<double> amps = {
        0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 16384};
    const int reps = 2;

    for (double A : amps) {
        double best_max = std::numeric_limits<double>::infinity();
        double best_mean = std::numeric_limits<double>::infinity();
        bool ok = false;
        for (int r = 0; r < reps; ++r) {
            std::mt19937 gen(2024 + r);
            std::uniform_real_distribution<double> dist(-A, A);
            std::vector<double> v(slots());
            for (auto& x : v) x = dist(gen);
            fold_sparse(v, fhe());
            try {
                Ctx ct = encrypt(fhe().cc, encode(fhe().cc, v), fhe().pk());
                bts_probe(fhe(), ct);
                auto out = decrypt_slots(fhe(), ct);
                out.resize(v.size());
                ErrStats es = err_stats(v, out);
                best_max = std::min(best_max, es.abs_max);
                best_mean = std::min(best_mean, es.abs_mean);
                ok = true;
            } catch (const std::exception& e) {
                std::cout << "[bts_wide] A=" << std::fixed << std::setprecision(0) << A
                          << " THREW: " << e.what() << std::endl;
            }
        }
        std::cout << std::fixed << std::setprecision(0)
                  << "[bts_wide] A=" << A << std::scientific << std::setprecision(3)
                  << " abs_max=" << best_max
                  << " abs_mean=" << best_mean
                  << " rel_max=" << (best_max / A)
                  << " ok=" << (ok ? 1 : 0) << std::endl;
    }
    SUCCEED();
}

// Floor side: tiny amplitudes. Decides the REAL absolute bts floor per
// iteration count (the cutmax cascade's laggard scalars live here; the
// folklore figure is 2e-4/2e-3 — measured z_off suggests ~1e-7 in-range).
TEST_F(BtsMagnitudeErrorTest, ErrorVsAmplitudeSmall) {
    const std::vector<double> amps = {
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0};
    const int reps = 3;

    for (double A : amps) {
        for (int r = 0; r < reps; ++r) {
            std::mt19937 gen(4242 + r);
            std::uniform_real_distribution<double> dist(-A, A);
            std::vector<double> v(slots());
            for (auto& x : v) x = dist(gen);
            fold_sparse(v, fhe());

            Ctx ct = encrypt(fhe().cc, encode(fhe().cc, v), fhe().pk());
            bts_probe(fhe(), ct);
            auto out = decrypt_slots(fhe(), ct);
            out.resize(v.size());

            ErrStats es = err_stats(v, out);
            std::cout << std::scientific << std::setprecision(3)
                      << "[bts_small] A=" << A << " rep=" << r
                      << " abs_max=" << es.abs_max
                      << " abs_p99=" << es.abs_p99
                      << " abs_mean=" << es.abs_mean
                      << " rel_max=" << (es.abs_max / A) << std::endl;
        }
    }
    SUCCEED();
}

// Wall-clock per bootstrap under the active routing (BTS_SPARSE_PROBE /
// BTS_ITERATIONS): prices the sparse-vs-full schedule cost (the 55-vs-91 ms
// figure was a projection; this measures it).
TEST_F(BtsMagnitudeErrorTest, BootstrapLatency) {
    // BTS_LAT_N overrides the timed-loop count (default 20). The timed loop is
    // bracketed by cudaProfilerStart/Stop so `nsys profile
    // --capture-range=cudaProfilerApi` attributes CUDA-API time to ONLY the warm
    // steady-state bootstraps (context setup + warm-up excluded).
    const int n = [] {
        const char* v = std::getenv("BTS_LAT_N");
        return v && *v ? std::max(1, std::atoi(v)) : 20;
    }();
    std::mt19937 gen(99);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> v(slots());
    for (auto& x : v) x = dist(gen);
    fold_sparse(v, fhe());
    Ctx ct = encrypt(fhe().cc, encode(fhe().cc, v), fhe().pk());
    for (int i = 0; i < 3; ++i) bts_probe(fhe(), ct);  // warm pool high-water
    cudaDeviceSynchronize();
    cudaProfilerStart();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) bts_probe(fhe(), ct);
    cudaDeviceSynchronize();
    cudaProfilerStop();
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / n;
    std::cout << std::fixed << std::setprecision(1)
              << "[bts_time] sparse=" << (sparse_probe() ? 1 : 0)
              << " iters=" << fhe().bts_iterations
              << " ms/bts=" << ms << std::endl;
    SUCCEED();
}

// K-cache-shaped input: most slots small (|x|<=2), a 5% minority at +-A.
// Measures whether out-of-range slots contaminate the in-range slots
// (cross-slot coupling through the bootstrap DFT), reporting the two
// populations separately.
TEST_F(BtsMagnitudeErrorTest, MixedSlotsCrossContamination) {
    const std::vector<double> peaks = {10, 15, 18, 20, 25};
    const int reps = 3;

    for (double A : peaks) {
        for (int r = 0; r < reps; ++r) {
            std::mt19937 gen(777 + r);
            std::uniform_real_distribution<double> small(-2.0, 2.0);
            std::uniform_real_distribution<double> sign(0.0, 1.0);
            std::vector<double> v(slots());
            std::vector<char> is_peak(slots(), 0);
            for (int i = 0; i < slots(); ++i) {
                if (i % 20 == 7) {  // 5% of slots at the peak magnitude
                    v[i] = (sign(gen) < 0.5 ? -A : A);
                    is_peak[i] = 1;
                } else {
                    v[i] = small(gen);
                }
            }

            Ctx ct = encrypt(fhe().cc, encode(fhe().cc, v), fhe().pk());
            fhe().bootstrap(ct);
            auto out = decrypt_slots(fhe(), ct);
            out.resize(v.size());

            std::vector<double> ref_s, got_s, ref_p, got_p;
            for (int i = 0; i < slots(); ++i) {
                (is_peak[i] ? ref_p : ref_s).push_back(v[i]);
                (is_peak[i] ? got_p : got_s).push_back(out[i]);
            }
            ErrStats ep = err_stats(ref_p, got_p);
            ErrStats es = err_stats(ref_s, got_s);
            std::cout << std::scientific << std::setprecision(3)
                      << "[bts_mix] peak_A=" << std::fixed << std::setprecision(0) << A
                      << " rep=" << r << std::scientific << std::setprecision(3)
                      << " peak_abs_max=" << ep.abs_max
                      << " peak_abs_mean=" << ep.abs_mean
                      << " small_abs_max=" << es.abs_max
                      << " small_abs_mean=" << es.abs_mean << std::endl;
        }
    }
    SUCCEED();
}

// Sparse-cliff diagnosis: does the sparse (real-only, period-512) bts introduce a SYSTEMATIC
// SIGNED bias on the STRUCTURED broadcast values the decode actually bootstraps (LN variance =
// period-1 const; softmax denom = 16 distinct in blocks of 32), vs the random-512 the floor test
// uses? Random abs-error stats can hide a signed bias. Run dense (no env) vs sparse
// (BTS_SPARSE_PROBE=1 SPARSE_BTS_SLOTS=512); a nonzero sparse-only signed_bias that GROWS with a
// small imaginary companion lane (cplx=1, models the observed im~0.1) is the earlier-cliff driver.
TEST_F(BtsMagnitudeErrorTest, SparseStructuredBias) {
    const int S = slots();
    const uint32_t sp = fhe().sparse_bts_slots ? fhe().sparse_bts_slots : 512;
    auto run = [&](const char* name, const std::vector<double>& a, double A, bool cplx) {
        std::vector<std::complex<double>> v(S);
        for (int i = 0; i < S; ++i) v[i] = {a[i], cplx ? 0.1 * a[i] : 0.0};
        Ctx ct = encrypt(fhe().cc, encode(fhe().cc, v), fhe().pk());
        bts_probe(fhe(), ct);
        auto out = decrypt_slots(fhe(), ct);   // real parts
        out.resize(a.size());
        double sbias = 0.0, amax = 0.0;
        for (int i = 0; i < S; ++i) {
            const double d = out[i] - a[i];
            sbias += d;
            if (std::fabs(d) > amax) amax = std::fabs(d);
        }
        sbias /= S;
        std::cout << std::scientific << std::setprecision(3)
                  << "[bts_bias] struct=" << name << " A=" << A << " cplx=" << (int)cplx
                  << " signed_bias=" << sbias << " abs_max=" << amax
                  << " rel_bias=" << sbias / A << std::endl;
    };
    for (double A : {0.5, 1.0, 2.0, 3.0}) {
        for (bool cplx : {false, true}) {
            std::vector<double> cst(S, A);                       // period-1 (LN variance)
            std::vector<double> sm(S);                           // 16 heads x 32 lanes in a 512 block
            for (int i = 0; i < S; ++i) {
                const int head = (i % sp) / (sp / 16);
                sm[i] = A * (0.3 + 0.7 * head / 15.0);
            }
            std::mt19937 gen(99);
            std::uniform_real_distribution<double> d(-A, A);
            std::vector<double> rnd(S);                          // random period-512
            for (int i = 0; i < (int)sp; ++i) rnd[i] = d(gen);
            for (int i = sp; i < S; ++i) rnd[i] = rnd[i % sp];
            run("const", cst, A, cplx);
            run("sm16", sm, A, cplx);
            run("rand", rnd, A, cplx);
        }
    }
    SUCCEED();
}

}  // namespace
