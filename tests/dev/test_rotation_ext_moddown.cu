// Isolated microbenchmark: does DEFERRING the ModDown in a rotation actually save time?
//
// Motivation: cachemir_linear.cu BSGS giant-step (line 64) and cascade (line 73) rotations
// each call the wrapper's rotate() == a FULL keyswitch incl. ModDown. When K independent
// partial sums are rotated then summed, the K ModDowns can be deferred to ONE at the end
// (rotate in the extended/ModUp basis, accumulate, single ModDown). FIDESlib's OWN native BSGS
// (third_party/FIDESlib/src/CKKS/LinearTransform.cu:166-248) already does exactly this.
//
// This test is SELF-CONTAINED at the FIDESlib-native level (it does NOT use the cachemir
// wrapper): it builds its own OpenFHE CKKS context + ONE FIDESlib GPU context (mirroring
// third_party/FIDESlib/test/RotationTests.cu), then:
//   (1) proves rotate(idx, /*moddown=*/false) + modDown == rotate(idx, /*moddown=*/true) numerically,
//   (2) isolates the ModDown cost: time [reconstruct], [reconstruct+rotate(true)], [reconstruct+rotate(false)],
//   (3) measures the giant-step accumulate (K rotations + sum) baseline vs deferred-ModDown, with a
//       per-K correctness assert.
//
// Build+run: sbatch scripts/14_rotation_ext_moddown.sh   (1 GPU, compute node)

#include "CKKS/Context.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Parameters.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"

#include <openfhe.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace lbcrypto;

namespace {

// 51/60-bit working + special primes (candidate pool; adaptTo() replaces them with the
// crypto context's actual primes). Copied verbatim from FIDESlib test/ParametrizedTest.cuh.
std::vector<FIDESlib::PrimeRecord> p64{
    {.p = 2305843009218281473}, {.p = 2251799661248513}, {.p = 2251799661641729}, {.p = 2251799665180673},
    {.p = 2251799682088961},    {.p = 2251799678943233}, {.p = 2251799717609473}, {.p = 2251799710138369},
    {.p = 2251799708827649},    {.p = 2251799707385857}, {.p = 2251799713677313}, {.p = 2251799712366593},
    {.p = 2251799716691969},    {.p = 2251799714856961}, {.p = 2251799726522369}, {.p = 2251799726129153},
    {.p = 2251799747493889},    {.p = 2251799741857793}, {.p = 2251799740416001}, {.p = 2251799746707457},
    {.p = 2251799756013569},    {.p = 2251799775805441}, {.p = 2251799763091457}, {.p = 2251799767154689},
    {.p = 2251799765975041},    {.p = 2251799770562561}, {.p = 2251799769776129}, {.p = 2251799772266497},
    {.p = 2251799775281153},    {.p = 2251799774887937}, {.p = 2251799797432321}, {.p = 2251799787995137},
    {.p = 2251799787601921},    {.p = 2251799791403009}, {.p = 2251799789568001}, {.p = 2251799795466241},
    {.p = 2251799807131649},    {.p = 2251799806345217}, {.p = 2251799805165569}, {.p = 2251799813554177},
    {.p = 2251799809884161},    {.p = 2251799810670593}, {.p = 2251799818928129}, {.p = 2251799816568833},
    {.p = 2251799815520257}};

std::vector<FIDESlib::PrimeRecord> sp64{
    {.p = 2305843009218936833}, {.p = 2305843009220116481}, {.p = 2305843009221820417}, {.p = 2305843009224179713},
    {.p = 2305843009225228289}, {.p = 2305843009227980801}, {.p = 2305843009229160449}, {.p = 2305843009229946881},
    {.p = 2305843009231650817}, {.p = 2305843009235189761}, {.p = 2305843009240301569}, {.p = 2305843009242923009},
    {.p = 2305843009244889089}, {.p = 2305843009245413377}, {.p = 2305843009247641601}};

// Device-synced mean wall time per call (us), after `warmup` untimed iters.
template <class F>
double timed_mean_us(F&& f, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) f();
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) f();
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
}

TEST(RotationExtModdown, ProveDeferredModdownSpeedup) {
    // ---- OpenFHE CKKS context (known-good FIDESlib config: logN16, L29, dnum4) ----
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(29);
    parameters.SetFirstModSize(60);
    parameters.SetScalingModSize(59);
    parameters.SetBatchSize(8);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 16);
    parameters.SetNumLargeDigits(4);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    auto keys = cc->KeyGen();

    // Giant-step-like rotation indices: g * 1024, g = 1..16.
    std::vector<int> rot_steps;
    for (int g = 1; g <= 16; ++g) rot_steps.push_back(g * 1024);
    cc->EvalRotateKeyGen(keys.secretKey, std::vector<int32_t>(rot_steps.begin(), rot_steps.end()));

    // ---- FIDESlib GPU context (single context for this process) ----
    std::vector<int> devices{0};
    FIDESlib::CKKS::Parameters fideslibParams{.logN = 16, .L = 29, .dnum = 4, .primes = p64, .Sprimes = sp64};
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
    FIDESlib::CKKS::GenAndAddRotationKeys(cc, keys, GPUcc, rot_steps);

    // ---- encrypt a known vector ----
    std::vector<double> x = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);
    auto c = cc->Encrypt(keys.publicKey, ptxt);
    FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, c);

    auto make_gpu = [&]() { return FIDESlib::CKKS::Ciphertext(GPUcc, raw); };
    auto decrypt8 = [&](FIDESlib::CKKS::Ciphertext& ct) {
        FIDESlib::CKKS::RawCipherText rr;
        ct.store(rr);
        auto out = c->Clone();
        FIDESlib::CKKS::GetOpenFHECipherText(out, rr);
        Plaintext pt;
        cc->Decrypt(keys.secretKey, out, &pt);
        pt->SetLength(8);
        return pt->GetRealPackedValue();
    };

    // ---- (1) correctness: rotate(false)+modDown == rotate(true) ----
    {
        auto a = make_gpu();
        a.rotate(rot_steps[0], true);
        auto b = make_gpu();
        b.rotate(rot_steps[0], false);
        b.modDown(false);
        auto va = decrypt8(a), vb = decrypt8(b);
        double maxd = 0.0;
        for (int i = 0; i < 8; ++i) maxd = std::max(maxd, std::fabs(va[i] - vb[i]));
        printf("[correctness] rotate(false)+modDown vs rotate(true): max|diff| = %.3e\n", maxd);
        EXPECT_LT(maxd, 1e-3);
    }

    // ---- (2) isolate ModDown cost (difference of reconstruct-then-op) ----
    const int W = 10, R = 100;
    double t_make  = timed_mean_us([&]() { auto z = make_gpu(); (void)z; }, W, R);
    double t_full  = timed_mean_us([&]() { auto z = make_gpu(); z.rotate(rot_steps[0], true); }, W, R);
    double t_nomod = timed_mean_us([&]() { auto z = make_gpu(); z.rotate(rot_steps[0], false); }, W, R);
    double rot_full  = t_full - t_make;
    double rot_nomod = t_nomod - t_make;
    double moddown   = rot_full - rot_nomod;
    double frac = (rot_full > 0) ? moddown / rot_full : 0.0;
    printf("[primitive us] reconstruct=%.1f  rotate(full)=%.2f  rotate(noModDown)=%.2f  =>  ModDown=%.2f (%.0f%% of a rotation)\n",
           t_make, rot_full, rot_nomod, moddown, 100.0 * frac);

    // ---- projection from the clean primitive costs (reconstruct already subtracted) ----
    printf("[projected accumulate speedup]  baseline=K*rotate(full)  vs  deferred=K*rotate(noModDown)+1 ModDown\n");
    for (int K : {2, 4, 8, 16, 32}) {
        double base = K * rot_full;
        double def  = K * rot_nomod + moddown;
        printf("   K=%2d : %.2fx   (asymptote %.2fx)\n", K, base / def, rot_full / rot_nomod);
    }

    // ---- (3) MEASURED giant-step accumulate: pre-stage one on-device ct, clone device->device
    // each step (a cheap stand-in for the real partial sum) so the timing is NOT dominated by the
    // 6 ms host->device reload that make_gpu() incurs. ----
    auto base_gpu = make_gpu();
    auto make_fresh = [&]() { FIDESlib::CKKS::Ciphertext z(GPUcc); z.copy(base_gpu); return z; };
    double t_clone = timed_mean_us([&]() { auto z = make_fresh(); (void)z; }, W, R);
    printf("[measured accumulate]  device-clone=%.1f us (vs %.1f us host-reload); K rotations + sum:\n", t_clone, t_make);
    for (int K : {2, 4, 8, 16}) {
        auto run_baseline = [&]() {
            auto acc = make_fresh();
            acc.rotate(rot_steps[0], true);
            for (int k = 1; k < K; ++k) {
                auto t = make_fresh();
                t.rotate(rot_steps[k], true);
                acc.add(t);
            }
            return acc;
        };
        auto run_deferred = [&]() {
            auto acc = make_fresh();
            acc.rotate(rot_steps[0], false);
            for (int k = 1; k < K; ++k) {
                auto t = make_fresh();
                t.rotate(rot_steps[k], false);
                acc.add(t);
            }
            acc.modDown(false);
            return acc;
        };
        // correctness for this K
        { auto rb = run_baseline(); auto rd = run_deferred();
          auto vb = decrypt8(rb), vd = decrypt8(rd);
          double maxd = 0.0; for (int i = 0; i < 8; ++i) maxd = std::max(maxd, std::fabs(vb[i] - vd[i]));
          EXPECT_LT(maxd, 1e-3) << "K=" << K << " deferred-ModDown accumulate diverged"; }
        double tb = timed_mean_us([&]() { auto z = run_baseline(); (void)z; }, 3, 30);
        double td = timed_mean_us([&]() { auto z = run_deferred(); (void)z; }, 3, 30);
        printf("   K=%2d : baseline=%7.1f us   deferred=%7.1f us   speedup=%.2fx\n", K, tb, td, tb / td);
    }
}

}  // namespace
