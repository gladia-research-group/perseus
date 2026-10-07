#include <fstream>
#include <NTT.cuh>  // FIDESlib::setDiscardScratch (scratch-discard A/B)
#include <CKKS/BootstrapPrecomputation.cuh>
#include <CKKS/CoeffsToSlots.cuh>
#include <CKKS/Discard.cuh>
#include <CKKS/RNSPoly.cuh>
#include <CKKS/SmallInt.cuh>
#include <CKKS/LinearKS.cuh>  // setModupMerge
#include <CKKS/Omr.cuh>
#include <CKKS/SparseB.cuh>
#include <CKKS/Spru.cuh>
#include <complex>
#include <NTTcluster.cuh>
#include <NTTtc.cuh>    // setNttCluster
#include "ckks_fixture.h"
#include "ckks_primitives.h"

#include <cuda_profiler_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <map>
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
 const auto base = varied_pattern(sp, std::max(1, sp / 2), amp, 0xC0FFEEu);  // sp=1: one active slot, not an all-zero message
 want.resize(S);
 for (int i = 0; i < S; ++i) want[i] = base[i % sp];
 } else {
 want = varied_pattern(S, S / 2, amp, 0xC0FFEEu);
 }
 // BTS_PROF_COMPLEX=1: complex slots (imaginary parts = a second pattern; only the real parts are checked below, which
 // still exposes a Re/Im mix-up). BTS_PROF_IN_LEVEL=<consumed primes>: encode the input at that level instead of the top.
 // BTS_PROF_IN_DEG2=1: present the input deg-2 (one scalar product, no rescale) as the model often does.
 const int in_level = env_i("BTS_PROF_IN_LEVEL", 0);
 Ctx ct;
 if (env_i("BTS_PROF_COMPLEX", 0)) {
 const auto im = varied_pattern(S, S / 2, amp, 0xBEEFu);
 std::vector<std::complex<double>> cv(S);
 const auto imb = sparse ? varied_pattern(sp, std::max(1, sp / 2), amp, 0xBEEFu) : im;
 for (int i = 0; i < S; ++i) cv[i] = {want[i], sparse ? imb[i % sp] : im[i]};
 ct = encrypt(fhe().cc, encode(fhe().cc, cv, in_level), fhe().pk());
 } else {
 ct = encrypt(fhe().cc, encode(fhe().cc, want, in_level), fhe().pk());
 }
 if (env_i("BTS_PROF_IN_DEG2", 0)) ct = fhe().cc->EvalMult(ct, 1.0);
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
 if (env_i("BTS_PROF_DUMP", 0)) {  // error structure: worst slots, least-squares gain got ~ g * want, rms error
 std::vector<size_t> idx(got.size());
 for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
 std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return std::fabs(got[a] - want[a]) > std::fabs(got[b] - want[b]); });
 double sw = 0, sww = 0, se2 = 0;
 for (size_t i = 0; i < got.size(); ++i) { sw += got[i] * want[i]; sww += want[i] * want[i]; se2 += (got[i] - want[i]) * (got[i] - want[i]); }
 std::cout << "[bts_prof] gain=" << (sww > 0 ? sw / sww : 0) << " rms_err=" << std::sqrt(se2 / got.size()) << "\n";
 for (int i = 0; i < 8 && i < (int)idx.size(); ++i)
 std::cout << "[bts_prof] worst slot " << idx[i] << " want=" << want[idx[i]] << " got=" << got[idx[i]] << "\n";
 for (int i = 0; i < 4; ++i)
 std::cout << "[bts_prof] slot " << i << " want=" << want[i] << " got=" << got[i] << "\n";
 }
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


// Linear key switching (eprint 2024/1629) at our parameters: exact gate vs the single-limb-digit baseline, timings
// next to FIDESlib's hybrid switch. Opt-in: BTS_LINKS_ITERS=n.
TEST_F(BtsProfileTest, LinearKsBench) {
    const char* e = std::getenv("BTS_LINKS_ITERS");
    if (!e) GTEST_SKIP() << "set BTS_LINKS_ITERS";
    const int S = slots();
    auto want = varied_pattern(S, S / 2, 1.0, 0xC0FFEEu);
    Ctx ct = encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk());
    Ctx out = fhe().eval_bootstrap_iter(ct, 1, 0);
    auto gpu = std::static_pointer_cast<::FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(out->gpu));
    const long bad = ::FIDESlib::CKKS::linearKsBench(*gpu, std::atoi(e), std::cout);
    EXPECT_EQ(bad, 0);
}


// Lever D (OverModRaise1) price gate: PtMult-first radix-2 stage 0 vs the shipped hoisted stage 0. Opt-in: BTS_OMR_ITERS=n.
TEST_F(BtsProfileTest, OmrStage0Bench) {
    const char* e = std::getenv("BTS_OMR_ITERS");
    if (!e) GTEST_SKIP() << "set BTS_OMR_ITERS";
    const int S = slots();
    auto want = varied_pattern(S, S / 2, 1.0, 0xC0FFEEu);
    Ctx ct = encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk());
    Ctx out = fhe().eval_bootstrap_iter(ct, 1, 0);
    auto gpu = std::static_pointer_cast<::FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(out->gpu));
    EXPECT_EQ(::FIDESlib::CKKS::omrStage0Bench(*gpu, S, std::atoi(e), std::cout), 0);
}


// Lever B calibration: the sparse route's "post-CtS" / "post-StC" / "end" stage values dumped to $BTS_TRACE_DIR/<tag>-<stage>.bin
// (decrypt_slots doubles). Run with FIDESLIB_BTS_SPARSE_B=1 twice: FIDESLIB_BTS_SPARSE_B_RUN=0 (shipped flow) and =1, then compare
// with logs/bts_traffic/sparseb_calib.py. Needs SPARSE_BTS_SLOTS=512 BTS_PROF_SLOTS=512.
TEST_F(BtsProfileTest, SparseBTrace) {
    const char* dir = std::getenv("BTS_TRACE_DIR");
    if (!dir || !*dir) GTEST_SKIP() << "set BTS_TRACE_DIR";
    const int S = slots();
    const int sp = env_i("BTS_PROF_SLOTS", 512);
    const std::string tag = env_or("BTS_TRACE_TAG", "x");
    const auto base = varied_pattern(sp, std::max(1, sp / 2), 1.0, 0xC0FFEEu);
    const auto baseIm = varied_pattern(sp, std::max(1, sp / 2), 1.0, 0xBEEFu);
    const bool cplx = env_i("BTS_PROF_COMPLEX", 0) != 0;
    std::vector<double> want(S), wantIm(S);
    for (int i = 0; i < S; ++i) { want[i] = base[i % sp]; wantIm[i] = cplx ? baseIm[i % sp] : 0.0; }
    Ctx ct;
    if (cplx) {
        std::vector<std::complex<double>> cv(S);
        for (int i = 0; i < S; ++i) cv[i] = {want[i], wantIm[i]};
        ct = encrypt(fhe().cc, encode(fhe().cc, cv), fhe().pk());
    } else {
        ct = encrypt(fhe().cc, encode(fhe().cc, want), fhe().pk());
    }
    { std::ofstream fw(std::string(dir) + "/want_im.bin", std::ios::binary); fw.write((const char*)wantIm.data(), wantIm.size() * sizeof(double)); }
    if (sp < S) ct->SetSlots((uint32_t)sp);
    std::vector<std::pair<std::string, std::shared_ptr<::FIDESlib::CKKS::Ciphertext>>> stash;
    ::FIDESlib::CKKS::g_btsStageStash = &stash;
    Ctx out = fhe().eval_bootstrap_iter(ct, 1, 0);
    cudaDeviceSynchronize();
    ::FIDESlib::CKKS::g_btsStageStash = nullptr;
    out->SetSlots((uint32_t)S);
    auto gpu = std::static_pointer_cast<::FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(out->gpu));
    {
        auto got = decrypt_slots(fhe(), out);
        double e = 0;
        for (size_t i = 0; i < want.size() && i < got.size(); ++i) e = std::max(e, std::fabs(got[i] - want[i]));
        std::cout << "[sbtrace] " << tag << " end err_max=" << e << " bits=" << (e > 0 ? -std::log2(e) : 64.0) << "\n";
        std::ofstream fw(std::string(dir) + "/want.bin", std::ios::binary);
        fw.write((const char*)want.data(), want.size() * sizeof(double));
    }
    for (auto& [name, c] : stash) {
        if (name != "post-CtS" && name != "post-EvalMod" && name != "post-StC" && name != "end" && name != "post-StC1" &&
            name != "pre-StC1" && name != "post-CtS-std" && name != "post-EvalMod-std") continue;
        gpu->copy(*c);
        gpu->slots = S;
        cudaDeviceSynchronize();
        auto got = decrypt_slots(fhe(), out);
        double mx = 0;
        for (double v : got) if (std::isfinite(v)) mx = std::max(mx, std::fabs(v));
        std::cout << "[sbtrace] " << tag << " " << name << " level=" << c->getLevel() << " deg=" << c->NoiseLevel
                  << " NF=" << c->NoiseFactor << " max|v|=" << mx << " n=" << got.size() << "\n";
        std::ofstream f(std::string(dir) + "/" + tag + "-" + name + ".bin", std::ios::binary);
        f.write((const char*)got.data(), got.size() * sizeof(double));
        if (name == "end" || name == "post-StC" || name == "post-StC1" || name == "pre-StC1" || name == "post-EvalMod" || name == "post-CtS") {  // imaginary part: multiply by -i (X^(3N/2)) and read the real part
            gpu->multMonomial(3 * (int)fhe().cc->GetRingDimension() / 2);
            cudaDeviceSynchronize();
            auto gi = decrypt_slots(fhe(), out);
            std::ofstream fi(std::string(dir) + "/" + tag + "-" + name + "-im.bin", std::ios::binary);
            fi.write((const char*)gi.data(), gi.size() * sizeof(double));
        }
    }
}


// Lever B unit gate: the GPU CtS / StC alone on an encrypted KNOWN layout, against the cleartext formula (no q*I term, so the
// output is directly checkable). A complex round trip through the test's encode/decode path is reported first (the first run
// showed imaginary parts dropped), so the checks use a REAL z: a = (2/n) Re(conj(U)^T z) is the real coefficient vector with
// U a = z. CtS: input 1_r (x) z -> expect c * a_{al s + q} at slot al (r/2) s + q, one constant c (gc * n). StC: input the
// post-EvalMod layout (a at r/4+1 copies per block, junk elsewhere) -> expect c' * 1_r (x) z (z real here).
// Needs SPARSE_BTS_SLOTS=512 BTS_PROF_SLOTS=512 FIDESLIB_BTS_SPARSE_B=1.
TEST_F(BtsProfileTest, SparseBUnit) {
    if (!std::getenv("FIDESLIB_BTS_SPARSE_B")) GTEST_SKIP() << "set FIDESLIB_BTS_SPARSE_B=1";
    using cd = std::complex<double>;
    const int S = slots(), sp = env_i("BTS_PROF_SLOTS", 512);
    const int n = 2 * sp, r = (2 * S) / n, s = 2 * n / r;
    const uint32_t m = 4 * sp, mmask = m - 1;
    std::vector<uint32_t> rot(sp);
    for (uint32_t j = 0, f = 1; j < (uint32_t)sp; ++j, f = (f * 5) & mmask) rot[j] = f;
    auto U = [&](int j, int l) { return std::polar(1.0, 2 * M_PI * (double)(((uint32_t)l * rot[j]) & mmask) / m); };
    std::mt19937 gen(7); std::uniform_real_distribution<double> ud(-1.0, 1.0);
    auto enc_dec = [&](std::vector<cd>& v, int mode, int& nn, int& rr, int& ss) {  // mode 0 = round trip only
        auto pt = fhe().cc->MakeCKKSPackedPlaintext(v, 1, 0, nullptr, (uint32_t)S);
        Ctx ct = fhe().cc->Encrypt(fhe().pk(), pt);
        fhe().cc->LoadCiphertext(ct);
        auto gpu = std::static_pointer_cast<::FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(ct->gpu));
        if (mode) ::FIDESlib::CKKS::sparseBUnit(*gpu, sp, mode == 2, nn, rr, ss);
        gpu->slots = S;
        Ctx view = decrypt_view(fhe().cc, ct);
        Plaintext out;
        fhe().cc->Decrypt(view, fhe().sk(), &out);
        return out->GetCKKSPackedValue();
    };
    int nn = 0, rr = 0, ss = 0;
    {   // complex round trip through this path
        std::vector<cd> v(S); for (auto& x : v) x = cd(ud(gen), ud(gen));
        auto d = enc_dec(v, 0, nn, rr, ss);
        double er = 0, ei = 0;
        for (int i = 0; i < S; ++i) { er = std::max(er, std::fabs(d[i].real() - v[i].real())); ei = std::max(ei, std::fabs(d[i].imag() - v[i].imag())); }
        std::cout << "[sbunit] complex round trip: max |dRe| " << er << " max |dIm| " << ei << (ei > 0.01 ? "  (IMAGINARY PART NOT PRESERVED by this path)" : "") << "\n";
    }
    std::vector<double> z(sp); for (auto& x : z) x = ud(gen);
    std::vector<double> a(n, 0.0);   // a = (2/n) Re(conj(U)^T z)
    for (int l = 0; l < n; ++l) { cd acc = 0; for (int j = 0; j < sp; ++j) acc += std::conj(U(j, l)) * z[j]; a[l] = 2.0 * acc.real() / n; }
    {   // check U a == z (real z)
        double e = 0; for (int j = 0; j < sp; ++j) { cd acc = 0; for (int l = 0; l < n; ++l) acc += U(j, l) * a[l]; e = std::max(e, std::abs(acc - cd(z[j], 0))); }
        std::cout << "[sbunit] cleartext U a - z max " << e << "\n";
    }
    {   // CtS
        std::vector<cd> v(S); for (int i = 0; i < S; ++i) v[i] = z[i % sp];
        auto d = enc_dec(v, 1, nn, rr, ss);
        ASSERT_EQ(nn, n) << "no lever-B precomputation for this route";
        std::vector<double> ratio; double worst = 0;
        for (int al = 0; al < r / 2; ++al) for (int q = 0; q < s; ++q) {
            const cd got = d[al * (r / 2) * s + q]; const double want = a[al * s + q];
            if (std::fabs(want) > 0.05 / std::sqrt((double)n)) ratio.push_back(got.real() / want);
            worst = std::max(worst, std::fabs(got.imag()));
        }
        std::sort(ratio.begin(), ratio.end());
        const double med = ratio[ratio.size() / 2];
        double dev = 0; for (double x : ratio) dev = std::max(dev, std::fabs(x / med - 1));
        std::cout << "[sbunit] CtS: n=" << n << " r=" << r << " s=" << s << " ratio got/a median " << med << " (n*gc would be "
                  << n / 24.0 << ") max rel dev " << dev << " (" << ratio.size() << " slots) max |imag| " << worst << "\n";
        for (int k : {0, 1, 2, 31, 32, 33, 1000}) std::cout << "[sbunit]   k=" << k << " a=" << a[k] << " got=" << d[(k / s) * (r / 2) * s + (k % s)] << "\n";
        EXPECT_LT(dev, 0.02);
    }
    {   // StC
        std::vector<cd> g(S, 0.0);
        for (int al = 0; al < r / 2; ++al) {
            for (int c = 0; c < r / 4 + 1; ++c) for (int q = 0; q < s; ++q) g[al * (r / 2) * s + c * s + q] = a[al * s + q];
            for (int q = 0; q < (r / 4 - 1) * s; ++q) g[al * (r / 2) * s + (r / 4 + 1) * s + q] = 3.0 * ud(gen);  // auxiliary junk
        }
        auto d = enc_dec(g, 2, nn, rr, ss);
        std::vector<double> ratio;
        for (int i = 0; i < S; ++i) if (std::fabs(z[i % sp]) > 0.05) ratio.push_back(d[i].real() / z[i % sp]);
        std::sort(ratio.begin(), ratio.end());
        const double med = ratio[ratio.size() / 2];
        double dev = 0; for (double x : ratio) dev = std::max(dev, std::fabs(x / med - 1));
        std::cout << "[sbunit] StC: ratio got/z median " << med << " (gd = scaleDec ~ 4.1) max rel dev " << dev << "\n";
        for (int i : {0, 1, 2, 512, 1024, 5000}) std::cout << "[sbunit]   i=" << i << " z=" << z[i % sp] << " got=" << d[i] << "\n";
        EXPECT_LT(dev, 0.02);
    }
}


// SPRU (arXiv 2607.27401, n = 2) for the s = 1 route: bootstrap a ciphertext whose slots all hold one complex value,
// report per-phase wall and the error of Re and Im. Opt-in: BTS_SPRU_ITERS=n (needs CKKS_COMPLEX=1). BTS_SPRU_H (64),
// BTS_SPRU_CORR (correction = CF - deg, 4), BTS_SPRU_DEPTH (scalar products applied to the input first, 3), BTS_SPRU_AMP.
TEST_F(BtsProfileTest, SpruS1Proto) {
    const char* e = std::getenv("BTS_SPRU_ITERS");
    if (!e) GTEST_SKIP() << "set BTS_SPRU_ITERS";
    const int iters = std::atoi(e), h = env_i("BTS_SPRU_H", 64), corr = env_i("BTS_SPRU_CORR", 4),
              depth = env_i("BTS_SPRU_DEPTH", 3);
    const double amp = env_d("BTS_SPRU_AMP", 1.0);
    const int S = slots();
    auto& lcc = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(fhe().cc->cpu);
    auto lsk = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(fhe().sk()->pimpl);
    auto lpk = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(fhe().pk()->pimpl);
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> kp(lpk, lsk);
    auto& gctx = std::any_cast<::FIDESlib::CKKS::Context&>(fhe().cc->gpu);
    auto key = ::FIDESlib::CKKS::spruSetup(lcc, kp, gctx, h);

    const std::complex<double> z(0.6180339 * amp, -0.3141592 * amp);
    std::vector<std::complex<double>> v(S, z);
    Ctx base = encrypt(fhe().cc, encode(fhe().cc, v), fhe().pk());
    for (int i = 0; i < depth; ++i) base = fhe().cc->EvalMult(base, 1.0);
    std::vector<double> ms;
    double eRe = 0, eIm = 0;
    for (int it = 0; it < iters + 1; ++it) {
        Ctx c = base->Clone();
        fhe().cc->LoadCiphertext(c);
        auto gpu = std::static_pointer_cast<::FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(c->gpu));
        ::FIDESlib::CKKS::SpruTimes tm;
        ::FIDESlib::CKKS::spruBootstrap(*gpu, *key, (uint32_t)corr, lcc, &tm);
        if (it == 0) {
            std::cout << "[spru] input level " << base->GetLevel() << " -> output limbs " << gpu->getLevel() + 1
                      << " deg " << gpu->NoiseLevel << "\n";
        } else {
            ms.push_back(tm.total_ms);
        }
        std::cout << "[spru] it " << it << " total " << tm.total_ms << " ms: adjust " << tm.adjust_ms << " switch "
                  << tm.switch_ms << " host " << tm.host_ms << " encode " << tm.encode_ms << " extmult "
                  << tm.extmult_ms << " trace " << tm.trace_ms << " product " << tm.product_ms << " finish "
                  << tm.finish_ms << "\n";
        if (it == iters) {
            gpu->slots = S;
            Ctx view = decrypt_view(fhe().cc, c);
            Plaintext out;
            fhe().cc->Decrypt(view, fhe().sk(), &out);
            auto d = out->GetCKKSPackedValue();
            for (int j = 0; j < S; ++j) {
                eRe = std::max(eRe, std::fabs(d[j].real() - z.real()));
                eIm = std::max(eIm, std::fabs(d[j].imag() - z.imag()));
            }
            std::cout << "[spru] slot0 " << d[0] << " slot1 " << d[1] << " slot777 " << d[777] << " want " << z << "\n";
        }
    }
    std::sort(ms.begin(), ms.end());
    std::cout << "[spru] h=" << h << " corr=" << corr << " median total " << ms[ms.size() / 2] << " ms | err Re " << eRe
              << " (" << -std::log2(eRe) << " bits) Im " << eIm << " (" << -std::log2(eIm) << " bits)\n";
    EXPECT_LT(std::max(eRe, eIm), 1e-2);
}

// Upper bound of EvalMod Re/Im key-stream sharing: relinearization wall with the relin key cold (L2 flushed) vs warm (the
// same key just streamed by a relin of OTHER ciphertexts; the timed op's own operands are flushed in both arms).
// Opt-in: BTS_RELIN_ITERS=n, BTS_RELIN_LEVELS=<consumed primes, comma list> (default "20,28,36,42").
TEST_F(BtsProfileTest, RelinL2Bench) {
    const char* e = std::getenv("BTS_RELIN_ITERS");
    if (!e) GTEST_SKIP() << "set BTS_RELIN_ITERS";
    const int iters = std::atoi(e);
    std::vector<int> levels;
    {
        std::string s = std::getenv("BTS_RELIN_LEVELS") ? std::getenv("BTS_RELIN_LEVELS") : "20,28,36,42";
        size_t i = 0;
        while (i < s.size()) { size_t j = s.find(',', i); if (j == std::string::npos) j = s.size(); levels.push_back(std::atoi(s.substr(i, j - i).c_str())); i = j + 1; }
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    const size_t flushBytes = std::max<size_t>((size_t)prop.l2CacheSize * 3, (size_t)256 << 20);
    void* flush = nullptr;
    cudaMalloc(&flush, flushBytes);
    auto doFlush = [&] { cudaMemset(flush, (int)(rand() & 0xff), flushBytes); cudaDeviceSynchronize(); };
    std::cout << "[relin_l2] L2 " << (prop.l2CacheSize >> 20) << " MB, flush " << (flushBytes >> 20) << " MB\n";
    const int S = slots();
    for (int lv : levels) {
        std::vector<Ctx> h(4);
        for (int k = 0; k < 4; ++k)
            h[k] = encrypt(fhe().cc, encode(fhe().cc, varied_pattern(S, S / 2, 0.5, 0xA0u + k), lv), fhe().pk());
        auto dev = [&](int k) {
            Ctx c = h[k]->Clone();
            fhe().cc->LoadCiphertext(c);
            return std::make_pair(c, std::static_pointer_cast<::FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(c->gpu)));
        };
        auto A = dev(0), B = dev(1), C = dev(2), D = dev(3);
        auto& ctx = A.second->cc_;
        std::vector<double> cold, warm, first;
        for (int it = 0; it < iters + 1; ++it) {
            ::FIDESlib::CKKS::Ciphertext x(ctx), y(ctx);
            // cold: only the timed relin
            y.copy(*C.second);
            doFlush();
            auto t0 = std::chrono::steady_clock::now();
            y.mult(*D.second, false);
            cudaDeviceSynchronize();
            const double c_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            // warm key: a relin of A,B first (streams the key), then the timed relin of C,D
            x.copy(*A.second);
            y.copy(*C.second);
            doFlush();
            auto t1 = std::chrono::steady_clock::now();
            x.mult(*B.second, false);
            cudaDeviceSynchronize();
            auto t2 = std::chrono::steady_clock::now();
            y.mult(*D.second, false);
            cudaDeviceSynchronize();
            auto t3 = std::chrono::steady_clock::now();
            if (it == 0) continue;
            cold.push_back(c_ms);
            first.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
            warm.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
        }
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        std::cout << "[relin_l2] consumed " << lv << " limbs " << A.second->getLevel() + 1 << ": cold " << med(cold)
                  << " ms, first-of-pair " << med(first) << " ms, warm-key " << med(warm) << " ms, saving "
                  << med(cold) - med(warm) << " ms (" << 100.0 * (med(cold) - med(warm)) / med(cold) << " %)\n";
    }
    cudaFree(flush);
}

// FIDESLIB_EVALMOD_LOCKSTEP gate: the same dense ciphertext bootstrapped with the lockstep EvalMod off and on must decrypt
// to identical slots (the two halves run the same ops in the same per-half order). Opt-in: BTS_LOCKSTEP_GATE=1.
TEST_F(BtsProfileTest, LockstepBitExact) {
    if (!std::getenv("BTS_LOCKSTEP_GATE")) GTEST_SKIP() << "set BTS_LOCKSTEP_GATE=1";
    const int S = slots();
    Ctx base = encrypt(fhe().cc, encode(fhe().cc, varied_pattern(S, S / 2, 1.0, 0xC0FFEEu)), fhe().pk());
    // raw residues of the output ciphertext (decoding can add noise, so slots are not a deterministic witness)
    std::vector<std::vector<uint64_t>> raw[4];
    int k = 0;
    for (int on : {0, 1, 0, 1}) {
        setenv("FIDESLIB_EVALMOD_LOCKSTEP", on ? "1" : "0", 1);
        Ctx in = base->Clone();
        Ctx c = fhe().eval_bootstrap_iter(in, 1, 0);
        fhe().cc->LoadCiphertext(c);
        auto g = std::static_pointer_cast<::FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(c->gpu));
        cudaDeviceSynchronize();
        std::vector<std::vector<uint64_t>> r0, r1;
        g->c0.store(r0);
        g->c1.store(r1);
        r0.insert(r0.end(), r1.begin(), r1.end());
        raw[k++] = std::move(r0);
    }
    unsetenv("FIDESLIB_EVALMOD_LOCKSTEP");
    auto cmp = [&](int x, int y) {
        if (raw[x].size() != raw[y].size()) return (long)-1;
        long d = 0;
        for (size_t l = 0; l < raw[x].size(); ++l)
            for (size_t i = 0; i < raw[x][l].size(); ++i) d += raw[x][l][i] != raw[y][l][i];
        return d;
    };
    const long dOn = cmp(0, 1), dCtl = cmp(0, 2), dOn2 = cmp(1, 3);
    std::cout << "[lockstep] residues differing off-vs-on " << dOn << ", off-vs-off " << dCtl << ", on-vs-on " << dOn2
              << " (limbs " << raw[0].size() << ")\n";
    EXPECT_EQ(dCtl, 0) << "control not deterministic";
    EXPECT_EQ(dOn, 0);
}

// Price of the lane-wise square (perseus/impl/poly.py lane_square: a^2 + i b^2 from z = a + i b via z conj z, z^2 and
// conj(z^2)) against squaring two separate real halves — the per-product trade a lane-wise EvalMod would make.
// Opt-in: BTS_LANE_ITERS=n, BTS_LANE_LEVELS=<consumed primes> (default "10,18,26").
TEST_F(BtsProfileTest, LaneSquareBench) {
    const char* e = std::getenv("BTS_LANE_ITERS");
    if (!e) GTEST_SKIP() << "set BTS_LANE_ITERS";
    const int iters = std::atoi(e);
    std::vector<int> levels;
    {
        std::string s = std::getenv("BTS_LANE_LEVELS") ? std::getenv("BTS_LANE_LEVELS") : "10,18,26";
        size_t i = 0;
        while (i < s.size()) { size_t j = s.find(',', i); if (j == std::string::npos) j = s.size(); levels.push_back(std::atoi(s.substr(i, j - i).c_str())); i = j + 1; }
    }
    const int S = slots();
    using FC = ::FIDESlib::CKKS::Ciphertext;
    for (int lv : levels) {
        std::vector<Ctx> h(2);
        for (int k = 0; k < 2; ++k)
            h[k] = encrypt(fhe().cc, encode(fhe().cc, varied_pattern(S, S / 2, 0.5, 0xB0u + k), lv), fhe().pk());
        auto dev = [&](int k) {
            Ctx c = h[k]->Clone();
            fhe().cc->LoadCiphertext(c);
            return std::make_pair(c, std::static_pointer_cast<FC>(fhe().cc->GetDeviceCiphertext(c->gpu)));
        };
        auto A = dev(0), B = dev(1);
        auto& ctx = A.second->cc_;
        const int N = A.second->cc.N;
        std::vector<double> sep, lane;
        for (int it = 0; it < iters + 1; ++it) {
            FC x(ctx), y(ctx);
            x.copy(*A.second);
            y.copy(*B.second);
            cudaDeviceSynchronize();
            auto t0 = std::chrono::steady_clock::now();
            x.square(false);
            y.square(false);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::steady_clock::now();
            // lane: 4 (a^2 + i b^2) = 2 (1+i) z conj z + (1-i) (z^2 + conj z^2)
            FC z(ctx), zc(ctx), u(ctx), w(ctx), wc(ctx), t(ctx), ti(ctx);
            z.copy(*A.second);
            cudaDeviceSynchronize();
            auto t2 = std::chrono::steady_clock::now();
            zc.conjugate(z);
            u.copy(z);
            u.mult(zc, false);
            w.copy(z);
            w.square(false);
            if (w.NoiseLevel == 2) w.rescale();
            if (u.NoiseLevel == 2) u.rescale();
            wc.conjugate(w);
            w.add(wc);              // 2 Re(z^2)
            t.copy(u);
            ti.copy(u);
            ti.multMonomial(N / 2); // i u
            t.add(ti);
            t.add(t);               // 2 (1+i) u
            ti.copy(w);
            ti.multMonomial(N / 2);
            w.sub(ti);              // (1-i) * 2Re(z^2)
            t.add(w);
            cudaDeviceSynchronize();
            auto t3 = std::chrono::steady_clock::now();
            if (it == 0) continue;
            sep.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            lane.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
        }
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        std::cout << "[lane_sq] consumed " << lv << " limbs " << A.second->getLevel() + 1 << ": two separate squares "
                  << med(sep) << " ms, lane square " << med(lane) << " ms, ratio " << med(lane) / med(sep) << "\n";
    }
}

// Census for compact CtS/StC diagonals: for every LT plaintext of a route, how many DISTINCT values one limb really has
// in its stored (NTT) layout — as a repeat with period P in natural index order (v[i] == v[i mod P]) and as constant
// aligned blocks of length L (bit-reversed storage of a subring polynomial). A diagonal whose slot vector is periodic is
// a subring polynomial: its limb could be a small table read from L2 instead of 2^16 values from DRAM.
// Opt-in: BTS_DIAG_CENSUS=<slots list> (e.g. "32768,512").
TEST_F(BtsProfileTest, DiagPeriodCensus) {
    const char* e = std::getenv("BTS_DIAG_CENSUS");
    if (!e) GTEST_SKIP() << "set BTS_DIAG_CENSUS";
    auto& gctx = std::any_cast<::FIDESlib::CKKS::Context&>(fhe().cc->gpu);
    std::vector<int> routes;
    { std::string s = e; size_t i = 0; while (i < s.size()) { size_t j = s.find(',', i); if (j == std::string::npos) j = s.size(); routes.push_back(std::atoi(s.substr(i, j - i).c_str())); i = j + 1; } }
    auto minPeriod = [](const std::vector<uint64_t>& v) {
        size_t P = 1;
        while (P < v.size()) {
            bool ok = true;
            for (size_t i = P; i < v.size() && ok; ++i) ok = v[i] == v[i % P];
            if (ok) return P;
            P <<= 1;
        }
        return v.size();
    };
    auto maxBlock = [](const std::vector<uint64_t>& v) {
        size_t L = v.size();
        while (L > 1) {
            bool ok = true;
            for (size_t b = 0; b < v.size() && ok; b += L)
                for (size_t i = b + 1; i < b + L && ok; ++i) ok = v[i] == v[b];
            if (ok) return L;
            L >>= 1;
        }
        return (size_t)1;
    };
    for (int slots : routes) {
        auto& pre = gctx->GetBootPrecomputation(slots);
        for (int which = 0; which < 2; ++which) {
            auto& stages = which == 0 ? pre.CtS : pre.StC;
            for (size_t s = 0; s < stages.size(); ++s) {
                auto& st = stages[s];
                std::map<size_t, int> hist;  // distinct values per limb (min over the two layouts) -> count
                size_t bytesFull = 0, bytesCompact = 0;
                for (auto& pt : st.A) {
                    std::vector<std::vector<uint64_t>> limbs;
                    const_cast<::FIDESlib::CKKS::Plaintext&>(pt).c0.store(limbs);
                    cudaDeviceSynchronize();
                    size_t worst = 0;
                    for (size_t l = 0; l < std::min<size_t>(limbs.size(), 3); ++l) {
                        const size_t P = minPeriod(limbs[l]);
                        const size_t L = maxBlock(limbs[l]);
                        worst = std::max(worst, std::min(P, limbs[l].size() / L));
                    }
                    hist[worst]++;
                    bytesFull += limbs.size() * limbs[0].size() * 4;
                    bytesCompact += limbs.size() * worst * 4;
                }
                std::cout << "[diag] slots " << slots << (which == 0 ? " CtS" : " StC") << " stage " << s << ": "
                          << st.A.size() << " diagonals (bStep " << st.bStep << ", gStep " << st.gStep
                          << "), distinct-values histogram {";
                for (auto& [k, c] : hist) std::cout << k << ":" << c << " ";
                std::cout << "} Q bytes " << (bytesFull >> 20) << " MB -> " << (bytesCompact >> 10) << " KB\n";
            }
        }
    }
}

// FIDESLIB_LT_COMPACT gate: the same ciphertext bootstrapped with the compact diagonal reads bypassed and active must
// give identical raw residues (the masked index reads the same values). Opt-in: BTS_COMPACT_GATE=1 (with
// FIDESLIB_LT_COMPACT=1 so the masks exist).
TEST_F(BtsProfileTest, CompactBitExact) {
    if (!std::getenv("BTS_COMPACT_GATE")) GTEST_SKIP() << "set BTS_COMPACT_GATE=1";
    const int S = slots();
    Ctx base = encrypt(fhe().cc, encode(fhe().cc, varied_pattern(S, S / 2, 1.0, 0xC0FFEEu)), fhe().pk());
    std::vector<std::vector<uint64_t>> raw[3];
    int k = 0;
    for (int bypass : {1, 0, 1}) {
        setenv("FIDESLIB_LT_COMPACT_BYPASS", bypass ? "1" : "0", 1);
        Ctx in = base->Clone();
        Ctx c = fhe().eval_bootstrap_iter(in, 1, 0);
        fhe().cc->LoadCiphertext(c);
        auto g = std::static_pointer_cast<::FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(c->gpu));
        cudaDeviceSynchronize();
        std::vector<std::vector<uint64_t>> r0, r1;
        g->c0.store(r0);
        g->c1.store(r1);
        r0.insert(r0.end(), r1.begin(), r1.end());
        raw[k++] = std::move(r0);
    }
    unsetenv("FIDESLIB_LT_COMPACT_BYPASS");
    auto cmp = [&](int x, int y) {
        long d = 0;
        for (size_t l = 0; l < raw[x].size(); ++l)
            for (size_t i = 0; i < raw[x][l].size(); ++i) d += raw[x][l][i] != raw[y][l][i];
        return d;
    };
    const long dOn = cmp(0, 1), dCtl = cmp(0, 2);
    std::cout << "[compact] residues differing full-vs-compact " << dOn << ", full-vs-full " << dCtl << "\n";
    EXPECT_EQ(dCtl, 0);
    EXPECT_EQ(dOn, 0);
}

// Are the full-period CtS/StC diagonals sparse polynomials (a few monomials X^c, each regenerable from a shared
// root-power table)? Dumps limbs 0..2 of the INTT'd (coefficient-form) plaintext of the first BTS_DIAG_SPARSE diagonals
// of dense CtS stages 0, 1 and StC stage 2 into BTS_DIAG_DIR; logs/bts_traffic/diag_sparsity.py CRTs and counts.
TEST_F(BtsProfileTest, DiagSparsityDump) {
    const char* e = std::getenv("BTS_DIAG_SPARSE");
    const char* dir = std::getenv("BTS_DIAG_DIR");
    if (!e || !dir) GTEST_SKIP() << "set BTS_DIAG_SPARSE and BTS_DIAG_DIR";
    Ctx probe = encrypt(fhe().cc, encode(fhe().cc, std::vector<double>(slots(), 0.0)), fhe().pk());
    fhe().cc->LoadCiphertext(probe);
    auto gct = std::static_pointer_cast<::FIDESlib::CKKS::Ciphertext>(fhe().cc->GetDeviceCiphertext(probe->gpu));
    ::FIDESlib::CKKS::dumpLtDiagCoeffs(gct->cc, dir, std::atoi(e));  // library-side: its own struct layout
}
