#include <fstream>
#include <NTT.cuh>  // FIDESlib::setDiscardScratch (scratch-discard A/B)
#include <CKKS/Discard.cuh>
#include <CKKS/RNSPoly.cuh>
#include <CKKS/SmallInt.cuh>  // setModupMerge
#include <NTTcluster.cuh>
#include <NTTtc.cuh>    // setNttCluster
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
namespace FIDESlib { namespace CKKS { class Ciphertext;
extern std::vector<std::pair<std::string, std::shared_ptr<Ciphertext>>>* g_btsStageStash;  // Bootstrap.cuh
} }

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

// In-process A/B for FIDESLIB_LT_CHUNK (traffic campaign, lever 1): the chunked
// LinearTransform re-orders the same hoisted-dot / LT-dot kernels over limb ranges, so the
// bootstrap output must be bit-identical to the whole-ciphertext path. BTS_PROF_LT_CHUNK sets
// the chunk under test (default 6 limbs).
TEST_F(BtsProfileTest, LtChunkBitExact) {
 const int S = slots();
 const std::string chA = env_or("BTS_PROF_LT_CHUNK_A", "0");  // control: A=B=0 must be exact
 const std::string chB = env_or("BTS_PROF_LT_CHUNK", "6");
 const auto want = varied_pattern(S, S / 2, 1.0, 0xC0FFEEu);
 Ctx ct = encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk());

 auto run = [&](const std::string& v, FIDESlib::CKKS::RawCipherText& raw) {
 setenv("FIDESLIB_LT_CHUNK", v.c_str(), 1);
 Ctx y = fhe().eval_bootstrap_iter(ct, 1, 0);
 cudaDeviceSynchronize();
 auto gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(y->gpu));
 gpu->store(raw);  // device -> host, no OpenFHE import
 auto got = decrypt_slots(fhe(), y);
 double e = 0;
 for (size_t i = 0; i < want.size() && i < got.size(); ++i) e = std::max(e, std::fabs(got[i] - want[i]));
 std::cout << "[lt_chunk] FIDESLIB_LT_CHUNK=" << v << " bits = " << (e > 0 ? -std::log2(e) : 64.0) << "\n";
 };
 FIDESlib::CKKS::RawCipherText ra, rb;
 run(chA, ra);
 run(chB, rb);
 setenv("FIDESLIB_LT_CHUNK", "0", 1);

 size_t bad = 0, total = 0;
 auto cmp = [&](const char* tag, const std::vector<std::vector<uint64_t>>& x, const std::vector<std::vector<uint64_t>>& y) {
 ASSERT_EQ(x.size(), y.size()) << tag;
 for (size_t l = 0; l < x.size(); ++l) {
 ASSERT_EQ(x[l].size(), y[l].size()) << tag << " limb " << l;
 size_t badl = 0;
 for (size_t i = 0; i < x[l].size(); ++i) badl += (x[l][i] != y[l][i]);
 if (badl) std::cout << "[lt_chunk] " << tag << " limb" << l << " mismatches=" << badl << "\n";
 bad += badl;
 total += x[l].size();
 }
 };
 cmp("c0", ra.sub_0, rb.sub_0);
 cmp("c1", ra.sub_1, rb.sub_1);
 std::cout << "[lt_chunk] A=" << chA << " B=" << chB << " words=" << total << " mismatches=" << bad << "\n";
 EXPECT_EQ(bad, 0u);
}

// In-process A/B for the transient-scratch discard (FIDESlib NTT.cu, g_fides_discard_scratch):
// the permuted NTT scratch layout + discard.global.L2 must not change a single output word.
TEST_F(BtsProfileTest, ScratchDiscardBitExact) {
 const int S = slots();
 const auto want = varied_pattern(S, S / 2, 1.0, 0xC0FFEEu);
 Ctx ct = encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk());
 auto run = [&](int flag, FIDESlib::CKKS::RawCipherText& raw) {
 cudaDeviceSynchronize();
 ::FIDESlib::setDiscardScratch(flag);
 Ctx y = fhe().eval_bootstrap_iter(ct, 1, 0);
 cudaDeviceSynchronize();
 auto gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(y->gpu));
 gpu->store(raw);
 auto got = decrypt_slots(fhe(), y);
 double e = 0;
 for (size_t i = 0; i < want.size() && i < got.size(); ++i) e = std::max(e, std::fabs(got[i] - want[i]));
 std::cout << "[discard] flag=" << flag << " bits = " << (e > 0 ? -std::log2(e) : 64.0) << "\n";
 };
 FIDESlib::CKKS::RawCipherText ra, rb;
 run(0, ra);
 run(1, rb);
 run(0, ra);  // and back: the flag must not leave state behind
 cudaDeviceSynchronize();
 ::FIDESlib::setDiscardScratch(0);
 size_t bad = 0, total = 0;
 auto cmp = [&](const char* tag, const std::vector<std::vector<uint64_t>>& x, const std::vector<std::vector<uint64_t>>& y) {
 ASSERT_EQ(x.size(), y.size()) << tag;
 for (size_t l = 0; l < x.size(); ++l) {
 ASSERT_EQ(x[l].size(), y[l].size()) << tag << " limb " << l;
 size_t badl = 0;
 for (size_t i = 0; i < x[l].size(); ++i) badl += (x[l][i] != y[l][i]);
 if (badl) std::cout << "[discard] " << tag << " limb" << l << " mismatches=" << badl << "\n";
 bad += badl;
 total += x[l].size();
 }
 };
 cmp("c0", ra.sub_0, rb.sub_0);
 cmp("c1", ra.sub_1, rb.sub_1);
 std::cout << "[discard] words=" << total << " mismatches=" << bad << "\n";
 EXPECT_EQ(bad, 0u);
}

// In-process A/B for FIDESLIB_MODUP_MERGE (lever B1: coarser ModUp launches; pure launch geometry).
TEST_F(BtsProfileTest, ModupMergeBitExact) {
 const int S = slots();
 const auto want = varied_pattern(S, S / 2, 1.0, 0xC0FFEEu);
 Ctx ct = encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk());
 auto run = [&](int flag, FIDESlib::CKKS::RawCipherText& raw) {
 cudaDeviceSynchronize();
 ::FIDESlib::CKKS::setModupMerge(flag);
 Ctx y = fhe().eval_bootstrap_iter(ct, 1, 0);
 cudaDeviceSynchronize();
 auto gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(y->gpu));
 gpu->store(raw);
 auto got = decrypt_slots(fhe(), y);
 double e = 0;
 for (size_t i = 0; i < want.size() && i < got.size(); ++i) e = std::max(e, std::fabs(got[i] - want[i]));
 std::cout << "[modup_merge] flag=" << flag << " bits = " << (e > 0 ? -std::log2(e) : 64.0) << "\n";
 };
 const int lvl = env_i("BTS_PROF_MERGE", 1);  // 1 = merged INTT + specials, 2 = + one BConv/NTT over all digits
 FIDESlib::CKKS::RawCipherText ra, rb;
 run(0, ra);
 run(lvl, rb);
 ::FIDESlib::CKKS::setModupMerge(0);
 size_t bad = 0, total = 0;
 auto cmp = [&](const char* tag, const std::vector<std::vector<uint64_t>>& x, const std::vector<std::vector<uint64_t>>& y) {
 ASSERT_EQ(x.size(), y.size()) << tag;
 for (size_t l = 0; l < x.size(); ++l) {
 ASSERT_EQ(x[l].size(), y[l].size()) << tag << " limb " << l;
 size_t badl = 0;
 for (size_t i = 0; i < x[l].size(); ++i) badl += (x[l][i] != y[l][i]);
 if (badl) std::cout << "[modup_merge] " << tag << " limb" << l << " mismatches=" << badl << "\n";
 bad += badl;
 total += x[l].size();
 }
 };
 cmp("c0", ra.sub_0, rb.sub_0);
 cmp("c1", ra.sub_1, rb.sub_1);
 std::cout << "[modup_merge] words=" << total << " mismatches=" << bad << "\n";
 EXPECT_EQ(bad, 0u);
}

// In-process A/B for the single-pass cluster NTT/INTT (FIDESlib NTTcluster.cu): same butterflies, same
// twiddles, one launch per transform — the bootstrap output must be bit-identical.
TEST_F(BtsProfileTest, NttClusterBitExact) {
 const int S = slots();
 const auto want = varied_pattern(S, S / 2, 1.0, 0xC0FFEEu);
 Ctx ct = encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk());
 auto run = [&](int flag, FIDESlib::CKKS::RawCipherText& raw) {
 cudaDeviceSynchronize();
 ::FIDESlib::setNttCluster(flag);
 Ctx y = fhe().eval_bootstrap_iter(ct, 1, 0);
 cudaError_t e = cudaDeviceSynchronize();
 std::cout << "[ntt_cluster] flag=" << flag << " cuda=" << cudaGetErrorString(e) << "\n";
 auto gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(y->gpu));
 gpu->store(raw);
 auto got = decrypt_slots(fhe(), y);
 double err = 0;
 for (size_t i = 0; i < want.size() && i < got.size(); ++i) err = std::max(err, std::fabs(got[i] - want[i]));
 std::cout << "[ntt_cluster] flag=" << flag << " bits = " << (err > 0 ? -std::log2(err) : 64.0) << "\n";
 };
 FIDESlib::CKKS::RawCipherText ra, rb;
 run(0, ra);
 run(1, rb);
 ::FIDESlib::setNttCluster(0);
 size_t bad = 0, total = 0;
 auto cmp = [&](const char* tag, const std::vector<std::vector<uint64_t>>& x, const std::vector<std::vector<uint64_t>>& y) {
 ASSERT_EQ(x.size(), y.size()) << tag;
 for (size_t l = 0; l < x.size(); ++l) {
 ASSERT_EQ(x[l].size(), y[l].size()) << tag << " limb " << l;
 size_t badl = 0;
 for (size_t i = 0; i < x[l].size(); ++i) badl += (x[l][i] != y[l][i]);
 if (badl) std::cout << "[ntt_cluster] " << tag << " limb" << l << " mismatches=" << badl << "\n";
 bad += badl;
 total += x[l].size();
 }
 };
 cmp("c0", ra.sub_0, rb.sub_0);
 cmp("c1", ra.sub_1, rb.sub_1);
 std::cout << "[ntt_cluster] words=" << total << " mismatches=" << bad << "\n";
 EXPECT_EQ(bad, 0u);
}

// Per-limb unit check of the cluster transforms against the two-pass kernels on one ciphertext poly:
// INTT (flag 0 vs 1) on identical input, then NTT (flag 0 vs 1) on the INTT'd data. Reports the first
// mismatching (limb, index) for each direction. Everything through the LimbPartition API, no decrypt.
TEST_F(BtsProfileTest, NttClusterUnit) {
 using FIDESlib::CKKS::Limb;
 const int S = slots();
 const auto want = varied_pattern(S, S / 2, 1.0, 0xC0FFEEu);
 Ctx ct = encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk());
 fhe().cc->LoadCiphertext(ct);
 auto gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(ct->gpu));
 auto& part = gpu->c0.GPU[0];
 const int N = part.cc.N;
 const int nl = part.getLimbSize(*part.level);
 std::vector<std::vector<uint32_t>> orig(nl, std::vector<uint32_t>(N));
 auto dev = [&](int l) { return std::get<Limb<uint32_t>>(part.limb[l]).v.data; };
 auto snap = [&](std::vector<std::vector<uint32_t>>& dst) {
 cudaDeviceSynchronize();
 for (int l = 0; l < nl; ++l) cudaMemcpy(dst[l].data(), dev(l), N * 4, cudaMemcpyDeviceToHost);
 };
 auto restore = [&](const std::vector<std::vector<uint32_t>>& src) {
 cudaDeviceSynchronize();
 for (int l = 0; l < nl; ++l) cudaMemcpy(dev(l), src[l].data(), N * 4, cudaMemcpyHostToDevice);
 cudaDeviceSynchronize();
 };
 auto compare = [&](const char* tag, const std::vector<std::vector<uint32_t>>& a, const std::vector<std::vector<uint32_t>>& b) {
 size_t bad = 0; int shown = 0;
 for (int l = 0; l < nl; ++l)
 for (int i = 0; i < N; ++i)
 if (a[l][i] != b[l][i]) {
 if (shown < 6) { std::cout << "[ntt_unit] " << tag << " limb " << l << " idx " << i << " two-pass=" << a[l][i] << " cluster=" << b[l][i] << "\n"; ++shown; }
 ++bad;
 }
 std::cout << "[ntt_unit] " << tag << " limbs=" << nl << " mismatches=" << bad << " of " << (size_t)nl * N << "\n";
 return bad;
 };
 snap(orig);
 std::vector<std::vector<uint32_t>> a(nl, std::vector<uint32_t>(N)), b = a, c = a, d = a;
 ::FIDESlib::setNttCluster(0); part.INTT<FIDESlib::ALGO_SHOUP, FIDESlib::INTT_NONE>(part.cc.batch, true); snap(a);
 restore(orig);
 ::FIDESlib::setNttCluster(1); part.INTT<FIDESlib::ALGO_SHOUP, FIDESlib::INTT_NONE>(part.cc.batch, true); snap(b);
 const size_t bad_i = compare("INTT", a, b);
 restore(a);  // coefficient domain (two-pass result) as the NTT input
 ::FIDESlib::setNttCluster(0); part.NTT<FIDESlib::ALGO_SHOUP, FIDESlib::NTT_NONE>(part.cc.batch, true); snap(c);
 restore(a);
 ::FIDESlib::setNttCluster(1); part.NTT<FIDESlib::ALGO_SHOUP, FIDESlib::NTT_NONE>(part.cc.batch, true); snap(d);
 const size_t bad_f = compare("NTT", c, d);
 const size_t bad_rt = compare("NTT(INTT) round trip vs input (two-pass)", orig, c);
 ::FIDESlib::setNttCluster(0);
 restore(orig);
 EXPECT_EQ(bad_i, 0u);
 EXPECT_EQ(bad_f, 0u);
 (void)bad_rt;
}

// Lever A (fused ModDown + composite rescale in EvalMod's relins): NOT bit-exact by design (the
// approximate base conversion rounds differently), so the gate is precision: bits within 0.5 of the
// unfused bootstrap on the same ciphertext, both arms decrypting cleanly.
TEST_F(BtsProfileTest, FusedRescaleBits) {
 const int S = slots();
 const auto want = varied_pattern(S, S / 2, 1.0, 0xC0FFEEu);
 Ctx ct = encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk());
 auto run = [&](int flag) {
 cudaDeviceSynchronize();
 ::FIDESlib::CKKS::setFusedRescale(flag);
 Ctx y = fhe().eval_bootstrap_iter(ct, 1, 0);
 cudaError_t e = cudaDeviceSynchronize();
 auto got = decrypt_slots(fhe(), y);
 double err = 0; int nnan = 0;
 for (size_t i = 0; i < want.size() && i < got.size(); ++i) { if (!std::isfinite(got[i])) { ++nnan; continue; } err = std::max(err, std::fabs(got[i] - want[i])); }
 const double bits = err > 0 ? -std::log2(err) : 64.0;
 std::cout << "[fused_rescale] flag=" << flag << " cuda=" << cudaGetErrorString(e) << " out_level=" << (int)y->GetLevel()
 << " bits = " << bits << " nonfinite=" << nnan << "\n";
 return bits;
 };
 const double b0 = run(0), b1 = run(1), b0b = run(0);
 ::FIDESlib::CKKS::setFusedRescale(0);
 std::cout << "[fused_rescale] unfused " << b0 << " / " << b0b << "  fused " << b1 << "\n";
 EXPECT_GT(b1, std::min(b0, b0b) - 0.5);
}

// Lever E (FIDESLIB_PW_FUSE): out-of-place mult/square + fused copy*P are pure reorderings -> bit-exact.
TEST_F(BtsProfileTest, PwFuseBitExact) {
 const int S = slots();
 const auto want = varied_pattern(S, S / 2, 1.0, 0xC0FFEEu);
 Ctx ct = encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk());
 auto run = [&](int flag, FIDESlib::CKKS::RawCipherText& raw) {
 cudaDeviceSynchronize();
 ::FIDESlib::CKKS::setPwFuse(flag);
 Ctx y = fhe().eval_bootstrap_iter(ct, 1, 0);
 cudaError_t e = cudaDeviceSynchronize();
 std::cout << "[pw_fuse] flag=" << flag << " cuda=" << cudaGetErrorString(e) << "\n";
 auto gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(y->gpu));
 gpu->store(raw);
 auto got = decrypt_slots(fhe(), y);
 double err = 0;
 for (size_t i = 0; i < want.size() && i < got.size(); ++i) err = std::max(err, std::fabs(got[i] - want[i]));
 std::cout << "[pw_fuse] flag=" << flag << " bits = " << (err > 0 ? -std::log2(err) : 64.0) << "\n";
 };
 FIDESlib::CKKS::RawCipherText ra, rb;
 run(0, ra);
 run(1, rb);
 ::FIDESlib::CKKS::setPwFuse(0);
 size_t bad = 0, total = 0;
 auto cmp = [&](const char* tag, const std::vector<std::vector<uint64_t>>& x, const std::vector<std::vector<uint64_t>>& y) {
 ASSERT_EQ(x.size(), y.size()) << tag;
 for (size_t l = 0; l < x.size(); ++l) {
 ASSERT_EQ(x[l].size(), y[l].size()) << tag << " limb " << l;
 size_t badl = 0;
 for (size_t i = 0; i < x[l].size(); ++i) badl += (x[l][i] != y[l][i]);
 if (badl) std::cout << "[pw_fuse] " << tag << " limb" << l << " mismatches=" << badl << "\n";
 bad += badl;
 total += x[l].size();
 }
 };
 cmp("c0", ra.sub_0, rb.sub_0);
 cmp("c1", ra.sub_1, rb.sub_1);
 std::cout << "[pw_fuse] words=" << total << " mismatches=" << bad << "\n";
 EXPECT_EQ(bad, 0u);
}

// Tensor-core NTT core (FIDESLIB_TC_NTT): exact integer matmul of the probed core matrix -> bit-exact.
TEST_F(BtsProfileTest, TcNttBitExact) {
 const int S = slots();
 const auto want = varied_pattern(S, S / 2, 1.0, 0xC0FFEEu);
 Ctx ct = encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk());
 auto run = [&](int flag, FIDESlib::CKKS::RawCipherText& raw) {
 cudaDeviceSynchronize();
 ::FIDESlib::setTcNtt(flag);
 Ctx y = fhe().eval_bootstrap_iter(ct, 1, 0);
 cudaError_t e = cudaDeviceSynchronize();
 std::cout << "[tc_ntt] flag=" << flag << " cuda=" << cudaGetErrorString(e) << "\n";
 auto gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(y->gpu));
 gpu->store(raw);
 auto got = decrypt_slots(fhe(), y);
 double err = 0;
 for (size_t i = 0; i < want.size() && i < got.size(); ++i) err = std::max(err, std::fabs(got[i] - want[i]));
 std::cout << "[tc_ntt] flag=" << flag << " bits = " << (err > 0 ? -std::log2(err) : 64.0) << "\n";
 };
 FIDESlib::CKKS::RawCipherText ra, rb;
 run(0, ra);
 run(1, rb);
 ::FIDESlib::setTcNtt(0);
 size_t bad = 0, total = 0;
 auto cmp = [&](const char* tag, const std::vector<std::vector<uint64_t>>& x, const std::vector<std::vector<uint64_t>>& y) {
 ASSERT_EQ(x.size(), y.size()) << tag;
 for (size_t l = 0; l < x.size(); ++l) {
 ASSERT_EQ(x[l].size(), y[l].size()) << tag << " limb " << l;
 size_t badl = 0;
 for (size_t i = 0; i < x[l].size(); ++i) badl += (x[l][i] != y[l][i]);
 if (badl) std::cout << "[tc_ntt] " << tag << " limb" << l << " mismatches=" << badl << "\n";
 bad += badl;
 total += x[l].size();
 }
 };
 cmp("c0", ra.sub_0, rb.sub_0);
 cmp("c1", ra.sub_1, rb.sub_1);
 std::cout << "[tc_ntt] words=" << total << " mismatches=" << bad << "\n";
 EXPECT_EQ(bad, 0u);
}

TEST_F(BtsProfileTest, TcNtt2BitExact) {
 const int S = slots();
 const auto want = varied_pattern(S, S / 2, 1.0, 0xC0FFEEu);
 Ctx ct = encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk());
 auto run = [&](int flag, FIDESlib::CKKS::RawCipherText& raw) {
 cudaDeviceSynchronize();
 ::FIDESlib::setTcNtt(flag ? 2 : 0);
 Ctx y = fhe().eval_bootstrap_iter(ct, 1, 0);
 cudaError_t e = cudaDeviceSynchronize();
 std::cout << "[tc_ntt2] flag=" << flag << " cuda=" << cudaGetErrorString(e) << "\n";
 auto gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(y->gpu));
 gpu->store(raw);
 auto got = decrypt_slots(fhe(), y);
 double err = 0;
 for (size_t i = 0; i < want.size() && i < got.size(); ++i) err = std::max(err, std::fabs(got[i] - want[i]));
 std::cout << "[tc_ntt2] flag=" << flag << " bits = " << (err > 0 ? -std::log2(err) : 64.0) << "\n";
 };
 FIDESlib::CKKS::RawCipherText ra, rb;
 run(0, ra);
 run(1, rb);
 ::FIDESlib::setTcNtt(0);
 size_t bad = 0, total = 0;
 auto cmp = [&](const char* tag, const std::vector<std::vector<uint64_t>>& x, const std::vector<std::vector<uint64_t>>& y) {
 ASSERT_EQ(x.size(), y.size()) << tag;
 for (size_t l = 0; l < x.size(); ++l) {
 ASSERT_EQ(x[l].size(), y[l].size()) << tag << " limb " << l;
 size_t badl = 0;
 for (size_t i = 0; i < x[l].size(); ++i) badl += (x[l][i] != y[l][i]);
 if (badl) std::cout << "[tc_ntt2] " << tag << " limb" << l << " mismatches=" << badl << "\n";
 bad += badl;
 total += x[l].size();
 }
 };
 cmp("c0", ra.sub_0, rb.sub_0);
 cmp("c1", ra.sub_1, rb.sub_1);
 std::cout << "[tc_ntt2] words=" << total << " mismatches=" << bad << "\n";
 EXPECT_EQ(bad, 0u);
}

// Per-stage trace of one bootstrap: every stage the bootstrap stashes (FIDESlib btsStageProbe / CtS-StC stage
// inputs) is copied into the output handle, decrypted and dumped to $BTS_TRACE_DIR/<stage>.bin (doubles) with
// level / NoiseFactor on stdout. Stages under the ephemeral sparse key (MR-atob, MR-raised) are skipped.
// Run it twice under two env settings and diff with logs/bts_traffic/trace_cmp.py.
TEST_F(BtsProfileTest, BtsStageTrace) {
    const int S = slots();
    const char* dir = std::getenv("BTS_TRACE_DIR");
    ASSERT_TRUE(dir && *dir) << "set BTS_TRACE_DIR";
    auto want = varied_pattern(S, S / 2, 1.0, 0xC0FFEEu);
    Ctx ct = encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk());
    std::vector<std::pair<std::string, std::shared_ptr<::FIDESlib::CKKS::Ciphertext>>> stash;
    ::FIDESlib::CKKS::g_btsStageStash = &stash;
    Ctx out = fhe().eval_bootstrap_iter(ct, 1, 0);
    cudaDeviceSynchronize();
    ::FIDESlib::CKKS::g_btsStageStash = nullptr;
    auto gpu = std::static_pointer_cast<::FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(out->gpu));
    {   // FIDESlib coefficient order for the exact decoder
        auto pi = fhe().cc->CoefficientOrderProbe();
        std::ofstream f(std::string(dir) + "/coef_perm.bin", std::ios::binary);
        f.write((const char*)pi.data(), pi.size() * sizeof(uint32_t));
    }
    {   // final error first (the handle still holds the real output); also dump it for a decode self-check
        auto got = decrypt_slots(fhe(), out);
        double e = 0;
        for (size_t i = 0; i < want.size() && i < got.size(); ++i) e = std::max(e, std::fabs(got[i] - want[i]));
        std::cout << "[trace] end err_max=" << e << " bits=" << (e > 0 ? -std::log2(e) : 64.0) << "\n";
        std::ofstream f(std::string(dir) + "/direct-end.bin", std::ios::binary);
        f.write((const char*)got.data(), got.size() * sizeof(double));
        std::ofstream fw(std::string(dir) + "/want.bin", std::ios::binary);
        fw.write((const char*)want.data(), want.size() * sizeof(double));
        {
            auto g0 = std::static_pointer_cast<::FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(out->gpu));
            ::FIDESlib::CKKS::exactDecryptDump(*g0, (std::string(dir) + "/direct-end.ct").c_str());
        }
        // decode self-check: the same ciphertext decrypted twice through the stash path must be identical
        auto gpu0 = std::static_pointer_cast<::FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(out->gpu));
        auto snap = std::make_shared<::FIDESlib::CKKS::Ciphertext>(gpu0->cc_);
        snap->copy(*gpu0);
        cudaDeviceSynchronize();
        stash.emplace_back("same-A", snap);
        stash.emplace_back("same-B", snap);
    }
    int k = 0;
    for (auto& [name, c] : stash) {
        if (name == "MR-raised") {  // under the sparse key: no decrypt, but the small-integer structure is checkable
            ::FIDESlib::CKKS::smallIntConsistencyCheck(c->cc, c->c1, 3, "MR-raised c1");
            ::FIDESlib::CKKS::smallIntConsistencyCheck(c->cc, c->c0, 3, "MR-raised c0");
            ::FIDESlib::CKKS::smallIntScalarCheck(c->cc, *c, 2.31672e-07, "MR-raised");
            continue;
        }
        if (name == "MR-atob") continue;
        gpu->copy(*c);
        cudaDeviceSynchronize();  // the API download does not wait on the copy's streams
        auto got = decrypt_slots(fhe(), out);
        double mx = 0;
        for (double v : got) if (std::isfinite(v)) mx = std::max(mx, std::fabs(v));
        const double sfl = c->cc.sfAtLimb(c->getLevel());
        const double canon = c->NoiseLevel == 2 ? sfl * sfl : sfl;
        std::cout << "[trace] " << std::setw(2) << k++ << " " << name << " level=" << c->getLevel()
                  << " noiseLevel=" << c->NoiseLevel << " NF=2^" << std::log2(c->NoiseFactor) << " max|v|=" << mx
                  << " rho-1=" << std::setprecision(3) << (c->NoiseFactor / canon - 1.0) << std::setprecision(6)
                  << "\n";
        std::ofstream f(std::string(dir) + "/" + name + ".bin", std::ios::binary);
        f.write((const char*)got.data(), got.size() * sizeof(double));
        ::FIDESlib::CKKS::exactDecryptDump(*c, (std::string(dir) + "/" + name + ".ct").c_str());
    }
}

// Unit test of the exact small-integer division (CKKS/SmallInt.cuh) against a host reference: synthetic signed
// integers |x| < q0 (the raised ciphertext's range) on the first 3 limbs, every Q limb := round(x / D).
TEST_F(BtsProfileTest, SmallIntDivideUnit) {
    Ctx keep = fhe().eval_bootstrap_iter(  // a GPU-resident ciphertext: supplies the context to the library-side test
        encrypt(fhe().cc, encode(fhe().cc, std::vector<double>(slots(), 0.0)), fhe().pk()), 1, 0);
    auto gpu = std::static_pointer_cast<::FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(keep->gpu));
    ASSERT_TRUE(gpu);
    const long bad = ::FIDESlib::CKKS::smallIntSelfTest(*gpu, 2.31672e-07);
    EXPECT_EQ(bad, 0);
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
