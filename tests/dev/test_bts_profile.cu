#include "ckks_fixture.h"
#include "ckks_primitives.h"

#include <cuda_profiler_api.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace test_helpers;

namespace {

using BtsProfileTest = CkksFixture;

// exists to give nsys/ncu a clean, steady-state capture window containing ONLY bootstraps:
//
//
// Run under `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop`, so the
// report holds no keygen / precomputation / encode kernels. Works on ANY chain (d=1 64-bit
// or d=2 composite 32-bit) — it only calls eval_bootstrap_iter, no level prediction.
//
// Env: BTS_PROF_ITERS (5), BTS_PROF_WARMUP (2), BTS_PROF_AMP (1.0),

int env_i(const char* k, int dflt) {
 const char* v = std::getenv(k);
 return (v && *v) ? std::atoi(v) : dflt;
}
double env_d(const char* k, double dflt) {
 const char* v = std::getenv(k);
 return (v && *v) ? std::atof(v) : dflt;
}

std::vector<double> varied_pattern(int n_slots, int n_active, double amp, uint32_t seed) {
 std::mt19937 gen(seed);
 std::uniform_real_distribution<double> lg(-4.0, 0.0);
 std::vector<double> x(n_slots, 0.0);
 for (int i = 0; i < n_active && i < n_slots; ++i) x[i] = amp * std::pow(10.0, lg(gen));
 return x;
}

TEST_F(BtsProfileTest, SteadyStateBootstrap) {
 const int S = slots();
 const int iters = env_i("BTS_PROF_ITERS", 5);
 const int warmup = env_i("BTS_PROF_WARMUP", 2);
 const double amp = env_d("BTS_PROF_AMP", 1.0);
 const int bts_iters = env_i("COMPOSITE_BTS_ITERS", 1);
 const int bts_prec = env_i("COMPOSITE_BTS_PREC", 0);
 const int sp = env_i("BTS_PROF_SLOTS", 0);
 const bool sparse = sp > 0 && sp < S;
 if (sparse && (int)fhe().sparse_bts_slots != sp) {
 FAIL() << "BTS_PROF_SLOTS=" << sp << " needs SPARSE_BTS_SLOTS=" << sp
 << " (wrapper sparse precomp; got " << fhe().sparse_bts_slots << ")";
 }

 std::vector<double> want;
 if (sparse) {
 const auto base = varied_pattern(sp, sp / 2, amp, 0xC0FFEEu);
 want.resize(S);
 for (int i = 0; i < S; ++i) want[i] = base[i % sp];
 } else {
 want = varied_pattern(S, S / 2, amp, 0xC0FFEEu);
 }
 Ctx ct = encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk());
 if (sparse) ct->SetSlots((uint32_t)sp);

 std::cout << "[bts_prof] slots=" << S << (sparse ? " SPARSE-ROUTED at " + std::to_string(sp) : "")
 << " warmup=" << warmup << " iters=" << iters
 << " amp=" << amp << " bts_iters=" << bts_iters << "\n";

 for (int i = 0; i < warmup; ++i) {
 Ctx b = fhe().eval_bootstrap_iter(ct, bts_iters, bts_prec);
 cudaDeviceSynchronize();
 }

 cudaProfilerStart();
 std::vector<double> ms(iters);
 Ctx last = ct;
 for (int i = 0; i < iters; ++i) {
 cudaDeviceSynchronize();
 const auto t0 = std::chrono::steady_clock::now();
 last = fhe().eval_bootstrap_iter(ct, bts_iters, bts_prec);
 cudaDeviceSynchronize();
 ms[i] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
 .count();
 }
 cudaProfilerStop();

    // Landing evidence for the planner frame (--sparse-bts-out wants MEASURED levels).
    std::cout << "[bts_prof] input_level=" << static_cast<int>(ct->GetLevel())
              << " output_level=" << static_cast<int>(last->GetLevel()) << "\n";

    double sum = 0, best = 1e300;
    for (int i = 0; i < iters; ++i) {
        sum += ms[i];
        best = std::min(best, ms[i]);
        std::cout << "[bts_prof] iter " << i << " = " << std::setprecision(6) << ms[i] << " ms\n";
    }
    std::cout << "[bts_prof] mean = " << (sum / iters) << " ms   min = " << best << " ms\n";

 if (sparse) last->SetSlots((uint32_t)(S));
 auto got = decrypt_slots(fhe(), last);
 got.resize(want.size());
 double e = 0;
 int nnan = 0;
 for (size_t i = 0; i < got.size(); ++i) {
 if (!std::isfinite(got[i])) { ++nnan; continue; }
 e = std::max(e, std::fabs(got[i] - want[i]));
 }
 std::cout << "[bts_prof] err_max = " << e << " bits = " << (e > 0 ? -std::log2(e) : 64.0)
 << " nonfinite = " << nnan << "\n";
 EXPECT_EQ(nnan, 0);
}

// Multi-ciphertext THROUGHPUT probe. The workloads bootstrap cts in
// groups; ksk_dot self-overlaps 1.00x, so a second INDEPENDENT ct is the only thing that
// can overlap it. BTS_PROF_CTS (default 2) cts are bootstrapped per timed iteration, each
// from its own host thread (one host thread enqueues serially and hides nothing).
// Run with FIDESLIB_KS_AUX_POOL=<CTS> so the shared keyswitch workspace singletons don't
// convoy the two pipelines.
// Metric is ms/ct against the 1-ct wall — NOT ms/bts.
TEST_F(BtsProfileTest, MultiCtThroughput) {
 const int S = slots();
 const int iters = env_i("BTS_PROF_ITERS", 5);
 const int warmup = env_i("BTS_PROF_WARMUP", 2);
 const double amp = env_d("BTS_PROF_AMP", 1.0);
 const int bts_iters = env_i("COMPOSITE_BTS_ITERS", 1);
 const int bts_prec = env_i("COMPOSITE_BTS_PREC", 0);
 const int ncts = env_i("BTS_PROF_CTS", 0);
 if (ncts < 2) GTEST_SKIP() << "set BTS_PROF_CTS>=2 to run the throughput probe";

 std::vector<std::vector<double>> want(ncts);
 std::vector<Ctx> cts;
 for (int c = 0; c < ncts; ++c) {
 want[c] = varied_pattern(S, S / 2, amp, 0xC0FFEEu + 17u * c);
 cts.push_back(encrypt(fhe().cc, encode(fhe().cc, want[c]), fhe().pk()));
 }

 std::cout << "[bts_prof] MULTI-CT ncts=" << ncts << " warmup=" << warmup
 << " iters=" << iters << " amp=" << amp << "\n";

 const bool use_threads = env_i("BTS_PROF_CT_THREADS", 0) != 0;
 std::vector<Ctx> out(ncts);
 auto run_all = [&] {
 if (use_threads) {
 std::vector<std::thread> th;
 for (int c = 0; c < ncts; ++c)
 th.emplace_back([&, c] { out[c] = fhe().eval_bootstrap_iter(cts[c], bts_iters, bts_prec); });
 for (auto& t : th) t.join();
 } else {
 for (int c = 0; c < ncts; ++c) out[c] = fhe().eval_bootstrap_iter(cts[c], bts_iters, bts_prec);
 }
 };

 for (int i = 0; i < warmup; ++i) {
 run_all();
 cudaDeviceSynchronize();
 }

 cudaProfilerStart();
 std::vector<double> ms(iters);
 for (int i = 0; i < iters; ++i) {
 cudaDeviceSynchronize();
 const auto t0 = std::chrono::steady_clock::now();
 run_all();
 cudaDeviceSynchronize();
 ms[i] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
 }
 cudaProfilerStop();

 double sum = 0, best = 1e300;
 for (int i = 0; i < iters; ++i) {
 sum += ms[i];
 best = std::min(best, ms[i]);
 std::cout << "[bts_prof] iter " << i << " = " << std::setprecision(6) << ms[i] << " ms ("
 << ms[i] / ncts << " ms/ct)\n";
 }
 std::cout << "[bts_prof] mean = " << (sum / iters) << " ms min = " << best << " ms\n";
 std::cout << "[bts_prof] per-ct mean = " << (sum / iters / ncts)
 << " ms/ct min = " << (best / ncts) << " ms/ct\n";

 int nnan = 0;
 for (int c = 0; c < ncts; ++c) {
 auto got = decrypt_slots(fhe(), out[c]);
 got.resize(want[c].size());
 double e = 0;
 for (size_t i = 0; i < got.size(); ++i) {
 if (!std::isfinite(got[i])) { ++nnan; continue; }
 e = std::max(e, std::fabs(got[i] - want[c][i]));
 }
 std::cout << "[bts_prof] ct" << c << " err_max = " << e
 << " bits = " << (e > 0 ? -std::log2(e) : 64.0) << "\n";
 }
 EXPECT_EQ(nnan, 0);
}

}
