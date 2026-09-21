// Isolation microbenchmark for the ct-x-pt multiply — the falsifiable test behind
// the ViT per-op verdict ([[vit-per-op-timed-verdict]]).
//
// Measured in production: one weight ct-x-pt mult inside diagonal::linear costs
// 3.14-5.23 ms wall. Estimated arithmetic: 2 polys x L limbs x 65536 coeffs
// (~1.18M modmuls at L=9) ~ tens of microseconds. That gap was an INFERENCE, and
// the attribution (per-plaintext GPUmalloc -> H2D -> mult -> GPUfree round-trip,
// WeightGranularity::Plaintext) was a HYPOTHESIS. This separates the terms:
//
//   A. mult only, plaintext already device-resident   -> the true kernel cost
//   B. load + mult + evict (the production pattern)   -> A + residency round-trip
//   C. load only            / D. evict only           -> the round-trip's halves
//
// If B >> A, the residency policy is the cost and the chunked path is the fix.
// If A alone is already milliseconds, the cost is inside FIDESlib's mult/allocator
// and our call pattern is not the lever.
//
// Timings are wall with a cudaDeviceSynchronize per iteration (same convention as
// StepProfiler's wall mode). LEVEL sweeps the limb count (limbs = depth - level).
//   build: cmake --build build --parallel 16 --target test_ptmult_isolation
//   run:   ITERS=200 LEVEL=19 build/bin/test_ptmult_isolation

#include "fideslib_wrapper.h"
#include "inference.h"
#include "test_helpers.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

int env_int(const char* k, int dflt) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::atoi(v) : dflt;
}

class PtMultIsolation : public ::testing::Test {
 protected:
    static void SetUpTestSuite() {
        CKKSContextOptions o = default_ckks_options();
        // Rotation keys must be requested HERE: make_ckks_context consumes
        // extra_rot_steps to run EvalRotateKeyGen and upload them. Generating them
        // after the fact leaves the device without the key and EvalFastRotation throws.
        // Opt-in (ROT_STEPS=31 for the BSGS pattern) so the other arms pay no keygen.
        rot_steps_ = env_int("ROT_STEPS", 0);
        for (int b = 1; b <= rot_steps_; ++b) o.extra_rot_steps.push_back(b * 32);
        ctx_   = make_ckks_context(o);
        slots_ = static_cast<int>(ctx_->cc->GetRingDimension() / 2);
    }
    static void TearDownTestSuite() { ctx_.reset(); }
    static std::shared_ptr<CKKSContext> ctx_;
    static int slots_;
    static int rot_steps_;
};
std::shared_ptr<CKKSContext> PtMultIsolation::ctx_;
int PtMultIsolation::slots_ = 0;
int PtMultIsolation::rot_steps_ = 0;

// median of a timing vector (robust to a stray first-touch outlier)
double median_us(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

}  // namespace

TEST_F(PtMultIsolation, MultVsResidency) {
    const int iters = env_int("ITERS", 100);
    const int level = env_int("LEVEL", 19);
    auto& fhe = *ctx_;

    std::vector<double> vals(slots_, 0.5);
    Ctx ct = encrypt(fhe.cc, fhe.cc->MakeCKKSPackedPlaintext(vals, 1, level), fhe.pk());

    auto make_pt = [&] {
        return fhe.cc->MakeCKKSPackedPlaintext(vals, /*noiseScaleDeg=*/1,
                                               static_cast<uint32_t>(level));
    };

    const int limbs = static_cast<int>(fhe.total_depth) - level;
    std::printf("\n[ptmult] slots=%d level=%d limbs=%d iters=%d\n",
                slots_, level, limbs, iters);
    std::printf("[ptmult] arithmetic/mult = 2 polys x %d limbs x %d coeffs = %.2f M modmul\n",
                limbs, slots_ * 2, 2.0 * limbs * slots_ * 2 / 1e6);

    // ---- A: mult only, plaintext already resident ----------------------------
    {
        Ptx pt = make_pt();
        fhe.cc->LoadPlaintext(pt, nullptr);   // resident for the whole loop
        Ctx warm = fhe.cc->EvalMult(ct, pt);  // warm caches/allocator
        cudaDeviceSynchronize();
        std::vector<double> t;
        t.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            Ctx out = fhe.cc->EvalMult(ct, pt);
            cudaDeviceSynchronize();
            t.push_back(std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count());
        }
        std::printf("[ptmult] A mult-only (pt resident)      median = %10.1f us\n", median_us(t));
    }

    // ---- B: the production pattern: load + mult + evict -----------------------
    {
        Ptx pt = make_pt();
        std::vector<double> t;
        t.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            fhe.cc->LoadPlaintext(pt, nullptr);
            Ctx out = fhe.cc->EvalMult(ct, pt);
            if (pt->gpu) { fhe.cc->EvictDevicePlaintext(pt->gpu); pt->gpu = 0; pt->loaded = false; }
            cudaDeviceSynchronize();
            t.push_back(std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count());
        }
        std::printf("[ptmult] B load+mult+evict (PRODUCTION) median = %10.1f us\n", median_us(t));
    }

    // ---- C: load only ---------------------------------------------------------
    {
        Ptx pt = make_pt();
        std::vector<double> t;
        t.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            fhe.cc->LoadPlaintext(pt, nullptr);
            cudaDeviceSynchronize();
            t.push_back(std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count());
            if (pt->gpu) { fhe.cc->EvictDevicePlaintext(pt->gpu); pt->gpu = 0; pt->loaded = false; }
        }
        std::printf("[ptmult] C load-only (H2D + alloc)      median = %10.1f us\n", median_us(t));
    }

    // ---- D: evict only --------------------------------------------------------
    {
        Ptx pt = make_pt();
        std::vector<double> t;
        t.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            fhe.cc->LoadPlaintext(pt, nullptr);
            cudaDeviceSynchronize();
            const auto t0 = std::chrono::steady_clock::now();
            if (pt->gpu) { fhe.cc->EvictDevicePlaintext(pt->gpu); pt->gpu = 0; pt->loaded = false; }
            cudaDeviceSynchronize();
            t.push_back(std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count());
        }
        std::printf("[ptmult] D evict-only (free)           median = %10.1f us\n", median_us(t));
    }

    // ---- E: N resident plaintexts, mult each (the CHUNKED pattern) ------------
    {
        const int n = env_int("CHUNK", 64);
        std::vector<Ptx> pts;
        pts.reserve(n);
        for (int i = 0; i < n; ++i) {
            pts.push_back(make_pt());
            fhe.cc->LoadPlaintext(pts.back(), nullptr);
        }
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) { Ctx out = fhe.cc->EvalMult(ct, pts[i]); }
        cudaDeviceSynchronize();
        const double us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("[ptmult] E chunked: %d resident mults    = %10.1f us  (%.1f us/mult)\n",
                    n, us, us / n);
        for (auto& p : pts) if (p->gpu) { fhe.cc->EvictDevicePlaintext(p->gpu); p->gpu = 0; p->loaded = false; }
    }

    // ---- F: in-place mult (no fresh output ciphertext) ------------------------
    // EvalMult builds `make_shared<CiphertextImpl>(*ct1)` (CryptoContext.cpp:1839),
    // whose copy ctor DEEP-COPIES the OpenFHE CPU shadow (Ciphertext.cpp:28-30) —
    // and that shadow is never level-reduced by GPU ops, so it is 12.6-29 MB of
    // host memcpy per call, discarded unused. EvalMultInPlace has no such copy.
    // If F << A, the mult kernel is fine and the cost is that copy — fixable from
    // OUR side by restructuring the accumulate loop, no FIDESlib edit needed.
    {
        Ptx pt = make_pt();
        fhe.cc->LoadPlaintext(pt, nullptr);
        Ctx scratch = fhe.cc->EvalMult(ct, pt);   // one ct to mutate
        cudaDeviceSynchronize();
        std::vector<double> t;
        t.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            fhe.cc->EvalMultInPlace(scratch, pt);
            cudaDeviceSynchronize();
            t.push_back(std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count());
        }
        std::printf("[ptmult] F mult-IN-PLACE (no ct copy)   median = %10.1f us\n", median_us(t));
        if (pt->gpu) { fhe.cc->EvictDevicePlaintext(pt->gpu); pt->gpu = 0; pt->loaded = false; }
    }

    // ---- G: clone cost alone (the same copy ctor, no arithmetic) --------------
    {
        std::vector<double> t;
        t.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            Ctx c = ct->Clone();
            cudaDeviceSynchronize();
            t.push_back(std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count());
        }
        std::printf("[ptmult] G clone-only (CPU shadow copy) median = %10.1f us\n", median_us(t));
    }

    // ---- H: scratch-reuse pattern (device copy + in-place mult) ---------------
    // What a rewritten BSGS linear would do per product: refill ONE scratch
    // ciphertext from the (immutable) rotated input on-device, then multiply in
    // place. Replaces the fresh-ciphertext-per-product pattern whose whole cost is
    // the CPU-shadow deep copy (arm G). If H ~ D2D + 8 us, the linear rewrite is
    // worth it; if H ~ A, the copy is not where the mult's time goes.
    {
        Ptx pt = make_pt();
        fhe.cc->LoadPlaintext(pt, nullptr);
        Ctx scratch = fhe.cc->EvalMult(ct, pt);   // ONE fresh ct, amortised
        cudaDeviceSynchronize();
        std::vector<double> t;
        t.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            fhe.cc->CopyCiphertextDevice(scratch, ct);   // scratch <- x_b[b], on device
            fhe.cc->EvalMultInPlace(scratch, pt);
            cudaDeviceSynchronize();
            t.push_back(std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count());
        }
        std::printf("[ptmult] H copy_device+inplace (REWRITE) median = %10.1f us\n", median_us(t));
        if (pt->gpu) { fhe.cc->EvictDevicePlaintext(pt->gpu); pt->gpu = 0; pt->loaded = false; }
    }

    std::printf("[ptmult] production measured in diagonal::linear = 3140-5230 us/mult\n");
}

// FIDESLIB_LAZY_CPU_SHADOW gate. Arms A/G above show that ~100% of a fresh-output GPU
// op's wall is the copy ctor deep-copying the source's OpenFHE CPU-side DCRTPoly limbs
// (up to ~29 MB at 28 towers) — data the device kernel immediately makes stale and that
// nothing reads before Decrypt overwrites it wholesale. With the env armed,
// CryptoContextImpl::MakeGpuResultLike gives GPU-path results a metadata-only
// (CloneEmpty) shadow instead.
//
// This runs a chain through the ops that mint results — pt-mult, ct-add, scalar-mult,
// negate — and checks the decrypted output against the analytic value. Run the SAME
// binary with FIDESLIB_LAZY_CPU_SHADOW=0 and =1: the printed checksum must be
// BIT-IDENTICAL (the device arithmetic is untouched; only host bookkeeping changed).
// ATTRIBUTING THE LINEAR. The per-op profile charges a whole BSGS linear to one leaf
// (`block.encoder_block.qkv`, 1 call), so dividing it by the plaintext-mult count silently
// bills the BSGS ROTATIONS to the mults. Per 1024x1024 layer diagonal::linear issues
// s*G = 1024 pt-mults AND (s-1)+(G-1) = 62 rotations; the captured graph shows 620
// rotate nodes per ViT block inside the linears (vs 782 in attention). Arms B/C also
// reload ONE plaintext, so they are L3-warm and page-faulted-in, while production loads a
// distinct DRAM-cold plaintext exactly once.
//
// This test separates the two unknowns:
//   COLD  — N distinct plaintexts, each loaded exactly once (the production pattern)
//   ROT   — hoisted rotation, the other occupant of the linear's wall
// Together with arm A (mult, pt resident) they close the linear's budget:
//   LINEAR_per_block ?= 10240*(cold_load + mult) + 620*rot + 10240*add
TEST_F(PtMultIsolation, ColdLoadAndRotation) {
    const int level = env_int("LEVEL", 19);
    const int n     = env_int("COLD_N", 64);
    auto& fhe = *ctx_;
    std::vector<double> vals(slots_, 0.5);

    // ---- COLD: n DISTINCT plaintexts, each loaded once ------------------------
    {
        std::vector<Ptx> pts;
        pts.reserve(n);
        for (int i = 0; i < n; ++i) {           // distinct contents => distinct pages
            std::vector<double> v(slots_, 0.5 + 0.001 * i);
            pts.push_back(fhe.cc->MakeCKKSPackedPlaintext(v, 1, static_cast<uint32_t>(level)));
        }
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) fhe.cc->LoadPlaintext(pts[i], nullptr);
        cudaDeviceSynchronize();
        const double us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("\n[coldrot] COLD load, %d DISTINCT pts = %.1f us/load"
                    "   (arm C, same pt reloaded, was ~947)\n", n, us / n);
        for (auto& p : pts) if (p->gpu) { fhe.cc->EvictDevicePlaintext(p->gpu); p->gpu = 0; p->loaded = false; }
    }

    // ---- ROT: hoisted rotation, the BSGS pattern -------------------------------
    {
        if (rot_steps_ <= 0) {
            std::printf("[coldrot] ROT arm skipped — re-run with ROT_STEPS=31 to seed the "
                        "BSGS rotation keys at context creation\n");
            return;
        }
        std::vector<int32_t> steps;
        for (int b = 1; b <= rot_steps_; ++b) steps.push_back(b * 32);   // t_in-strided, t_in=32
        Ctx ct = encrypt(fhe.cc, fhe.cc->MakeCKKSPackedPlaintext(vals, 1, level), fhe.pk());
        const uint32_t m = 2u * fhe.cc->GetRingDimension();
        auto precomp = fhe.cc->EvalFastRotationPrecompute(ct);   // null on GPU, cheap
        auto warm = fhe.cc->EvalFastRotation(ct, steps, m, precomp);   // prime caches
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        auto rots = fhe.cc->EvalFastRotation(ct, steps, m, precomp);
        cudaDeviceSynchronize();
        const double us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("[coldrot] HOISTED rotate, %zu steps = %.1f us total, %.1f us/rotation\n",
                    steps.size(), us, us / steps.size());
        std::printf("[coldrot] a ViT block issues 620 in-linear rotations => %.2f s/block\n",
                    620.0 * (us / steps.size()) / 1e6);
    }
}

TEST_F(PtMultIsolation, LazyShadowChain) {
    const int level = env_int("LEVEL", 19);
    auto& fhe = *ctx_;
    const char* lazy = std::getenv("FIDESLIB_LAZY_CPU_SHADOW");

    std::vector<double> a(slots_), b(slots_);
    for (int i = 0; i < slots_; ++i) { a[i] = 0.5 + 0.001 * (i % 17); b[i] = 0.25 - 0.002 * (i % 11); }

    Ctx ct = encrypt(fhe.cc, fhe.cc->MakeCKKSPackedPlaintext(a, 1, level), fhe.pk());
    Ptx pt = fhe.cc->MakeCKKSPackedPlaintext(b, 1, static_cast<uint32_t>(level));

    // r = -((a*b + a) * 3)   — one op of each result-minting kind
    Ctx r = fhe.cc->EvalMult(ct, pt);
    r     = fhe.cc->EvalAdd(r, ct);
    r     = fhe.cc->EvalMult(r, 3.0);
    r     = fhe.cc->EvalNegate(r);

    const std::vector<double> got = decrypt(fhe.cc, r, fhe.sk());

    double max_err = 0.0, checksum = 0.0;
    for (int i = 0; i < slots_; ++i) {
        const double want = -((a[i] * b[i] + a[i]) * 3.0);
        max_err = std::max(max_err, std::abs(got[i] - want));
        if (i < 8) checksum += got[i];
    }
    std::printf("\n[lazyshadow] FIDESLIB_LAZY_CPU_SHADOW=%s level=%d\n", lazy ? lazy : "(unset)", level);
    std::printf("[lazyshadow] max abs err vs analytic = %.3e\n", max_err);
    std::printf("[lazyshadow] checksum(first 8 slots) = %.17g   <- must match across arms\n", checksum);
    EXPECT_LT(max_err, 1e-5);
}

// Is copy_device + in-place VALUE-equivalent to a plain mult? The ViT k=1 gate said
// no ("Decrypt: approximation error too high", job 49898400) after the linear was
// rewritten to reuse a scratch. This isolates which step diverges: a fresh scratch
// (no prior mult), a REUSED scratch (already rescaled by an earlier in-place mult —
// RNSPoly::copy then has to dropToLevel+grow), and the accumulate as the linear
// actually does it.
TEST_F(PtMultIsolation, CopyDeviceIsValueEquivalent) {
    const int level = env_int("LEVEL", 19);
    auto& fhe = *ctx_;

    std::vector<double> a(slots_), b(slots_);
    for (int i = 0; i < slots_; ++i) { a[i] = 0.5 + 0.001 * (i % 17); b[i] = 0.25 - 0.002 * (i % 11); }

    Ctx ct = encrypt(fhe.cc, fhe.cc->MakeCKKSPackedPlaintext(a, 1, level), fhe.pk());
    Ptx pt = fhe.cc->MakeCKKSPackedPlaintext(b, 1, static_cast<uint32_t>(level));
    fhe.cc->LoadPlaintext(pt, nullptr);

    auto max_abs_diff = [&](const std::vector<double>& x, const std::vector<double>& y) {
        double m = 0.0;
        for (size_t i = 0; i < std::min(x.size(), y.size()); ++i)
            m = std::max(m, std::abs(x[i] - y[i]));
        return m;
    };

    const std::vector<double> ref = decrypt(fhe.cc, fhe.cc->EvalMult(ct, pt), fhe.sk());

    // (1) fresh scratch: copy a never-multiplied ct, then multiply in place
    Ctx s1 = ct->Clone();
    fhe.cc->CopyCiphertextDevice(s1, ct);
    fhe.cc->EvalMultInPlace(s1, pt);
    const double d_fresh = max_abs_diff(ref, decrypt(fhe.cc, s1, fhe.sk()));

    // (2) REUSED scratch: multiply once (rescales it), then refill from ct and
    //     multiply again — what the rewritten linear does from the 2nd product on.
    Ctx s2 = ct->Clone();
    fhe.cc->EvalMultInPlace(s2, pt);          // s2 now one level down
    fhe.cc->CopyCiphertextDevice(s2, ct);     // refill: needs dropToLevel + grow
    fhe.cc->EvalMultInPlace(s2, pt);
    const double d_reused = max_abs_diff(ref, decrypt(fhe.cc, s2, fhe.sk()));

    std::printf("\n[ptcorrect] level=%d  max|ref| = %.6f\n", level,
                *std::max_element(ref.begin(), ref.end()));
    std::printf("[ptcorrect] (1) fresh  scratch: max abs diff = %.3e\n", d_fresh);
    std::printf("[ptcorrect] (2) REUSED scratch: max abs diff = %.3e   <- the linear's pattern\n",
                d_reused);
    std::printf("[ptcorrect] (bts noise floor is ~1e-8; anything >>1e-6 means divergence)\n");
    EXPECT_LT(d_fresh, 1e-6);
    EXPECT_LT(d_reused, 1e-6);
}
