// Self-contained microbenchmark: does hoisted rotation (EvalFastRotation /
// rotate_hoisted) beat the scalar EvalRotate-in-a-loop that the wrapper's rotate()
// performs?
//
// It builds its own CKKS context with the rotation keys it needs, then for each
// "fan-out" pattern from the decode hot path it (a) checks the hoisted batch is
// numerically identical to the scalar loop, and (b) times three bars:
//
//   1. scalar_loop : K x  cc->EvalRotate(ct, step)                  <- the scalar path
//   2. fastloop    : precomp once, then K x EvalFastRotation(...,1) <- single-index
//                    overload in a loop. On GPU this re-does the ModUp per call, so
//                    it should NOT beat scalar -- this bar proves the win is BATCHING,
//                    not the FastRotation API per se.
//   3. hoist_batch : precomp once, then ONE EvalFastRotation(ct, {steps}, ...)
//                    -> a single rotate_hoisted -> ONE ModUp for all K rotations.
//
// Hot-path sites this models (cachemir, the live decode packing):
//   * softmax P*V lane fan-out  : src/algorithms/attention/cachemir/cachemir_attention.cu
//                                 rotate(softmax_scores, i*tH), i=1..d_head_real-1=63 (frozen source)
//   * cachemir linear x_rotated : src/algorithms/linear/cachemir/cachemir_linear.cu
//                                 rotate(x, j*t*t), j=1..r_i-1 (<=15, frozen source)
// Both rotate ONE frozen ciphertext by many distinct steps -> the textbook hoisting case.
//
// Build: bash scripts/local_build_devtest.sh test_rotation_hoisting

#include "test_helpers.h"   // make_ckks_context, default_ckks_options, CKKSContext,
                            // CKKSContextOptions, Ctx, Ptx, encrypt, decrypt_slots, env_or

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

// Device-synced mean wall-time per call (ms), after `warmup` untimed iterations.
template <class F>
double timed_mean_ms(F&& f, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) f();
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) f();
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;
}

struct Cfg { const char* name; int K; };

}  // namespace

TEST(RotationHoisting, ScalarLoopVsHoistedBatch) {
    // ---- geometry mirrors the decode hot path (logN=16 -> ringDim 65536 -> slots 32768) ----
    const int logN   = std::atoi(env_or("LOGN", "16").c_str());
    const int slots  = (1 << logN) / 2;
    const int stride = std::max(1, slots / 64);   // == tH (=512) at logN16; the P*V lane stride

    // Superset of every step any config rotates by -> the rotation keys to generate.
    std::vector<int32_t> all_steps;
    for (int i = 1; i <= 63; ++i) all_steps.push_back(static_cast<int32_t>(stride) * i);

    CKKSContextOptions o = default_ckks_options();   // honours the decode chain env knobs
    o.extra_rot_steps = all_steps;                   // seed our keys at construction (GPU-resident)
    auto ctx = make_ckks_context(o);

    const int real_slots = static_cast<int>(ctx->cc->GetRingDimension() / 2);
    ASSERT_EQ(real_slots, slots) << "slots mismatch (LOGN env?)";
    const uint32_t m = 2u * static_cast<uint32_t>(ctx->cc->GetRingDimension());  // cyclotomic order

    // ---- random, non-constant input so rotation + equality are meaningful ----
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<double> x(slots);
    for (double& v : x) v = U(rng);

    const std::vector<int> levels = {0, 16};   // fresh (28 limbs) and a decode-ish depth (~11 limbs)
    const std::vector<Cfg> cfgs   = {
        {"softmax_v_PV(K=63)", 63},   // attention P*V lane fan-out
        {"linear_xrot (K=15)", 15},   // cachemir linear input fan-out
        {"small_fanout(K=4)",   4},   // sanity, like FIDESlib's own bench
    };
    const int warmup = 3, repeats = 20;

    std::printf("\n[rotation-hoisting] logN=%d slots=%d stride(tH)=%d  3-bar timing (mean ms/call)\n",
                logN, slots, stride);
    std::printf("%-20s %4s %3s | %10s %11s %10s | %9s %9s | %9s\n",
                "config", "lvl", "K", "scalar_ms", "fastloop_ms", "hoist_ms",
                "x_vs_scal", "x_vs_fl", "max|d|");
    std::printf("%s\n", std::string(98, '-').c_str());

    for (int level : levels) {
        Ptx pt = ctx->cc->MakeCKKSPackedPlaintext(x, /*noiseScaleDeg=*/1, static_cast<uint32_t>(level));
        Ctx ct = encrypt(ctx->cc, pt, ctx->pk());

        for (const Cfg& c : cfgs) {
            const std::vector<int32_t> steps(all_steps.begin(), all_steps.begin() + c.K);

            // ---- correctness: hoisted batch == scalar loop ----
            std::vector<Ctx> sc(c.K);
            for (int k = 0; k < c.K; ++k) sc[k] = ctx->cc->EvalRotate(ct, steps[k]);

            auto pc = ctx->cc->EvalFastRotationPrecompute(ct);
            std::vector<Ctx> ho = ctx->cc->EvalFastRotation(ct, steps, m, pc);
            ASSERT_EQ(static_cast<int>(ho.size()), c.K);

            double max_abs_diff = 0.0;
            for (int k = 0; k < c.K; ++k) {
                const std::vector<double> ds = decrypt_slots(*ctx, sc[k]);
                const std::vector<double> dh = decrypt_slots(*ctx, ho[k]);
                for (int s = 0; s < slots; ++s)
                    max_abs_diff = std::max(max_abs_diff, std::abs(ds[s] - dh[s]));
            }
            EXPECT_LT(max_abs_diff, 1e-3)
                << "hoisted rotation diverges from scalar EvalRotate at " << c.name
                << " (level " << level << ")";

            // anchor (once): scalar EvalRotate is a genuine cyclic shift by steps[0] (either sign).
            if (level == 0) {
                const std::vector<double> r0 = decrypt_slots(*ctx, sc[0]);
                const int s0 = steps[0];
                double dl = 0.0, dr = 0.0;
                for (int s = 0; s < slots; ++s) {
                    dl = std::max(dl, std::abs(r0[s] - x[(s + s0) % slots]));
                    dr = std::max(dr, std::abs(r0[s] - x[((s - s0) % slots + slots) % slots]));
                }
                EXPECT_TRUE(dl < 1e-3 || dr < 1e-3) << "EvalRotate is not a clean cyclic shift";
            }

            // ---- timing: three bars (each allocates K result ciphertexts, so it's fair) ----
            const double t_scalar = timed_mean_ms([&] {
                std::vector<Ctx> out; out.reserve(c.K);
                for (int k = 0; k < c.K; ++k) out.push_back(ctx->cc->EvalRotate(ct, steps[k]));
            }, warmup, repeats);

            const double t_fastloop = timed_mean_ms([&] {
                auto p = ctx->cc->EvalFastRotationPrecompute(ct);
                std::vector<Ctx> out; out.reserve(c.K);
                for (int k = 0; k < c.K; ++k) out.push_back(ctx->cc->EvalFastRotation(ct, steps[k], m, p));
            }, warmup, repeats);

            const double t_hoist = timed_mean_ms([&] {
                auto p = ctx->cc->EvalFastRotationPrecompute(ct);
                auto out = ctx->cc->EvalFastRotation(ct, steps, m, p);  // GPU side effects -> not elided
                (void)out;
            }, warmup, repeats);

            std::printf("%-20s %4d %3d | %10.3f %11.3f %10.3f | %8.2fx %8.2fx | %9.2e\n",
                        c.name, level, c.K, t_scalar, t_fastloop, t_hoist,
                        t_scalar / t_hoist, t_fastloop / t_hoist, max_abs_diff);
        }
    }

    std::printf("\nReading: hoist_ms should be << scalar_ms (one ModUp vs K). fastloop_ms ~= scalar_ms\n"
                "confirms the GPU win is the BATCHED overload, not the FastRotation API alone.\n"
                "Real decode strides: tH=%d (P*V, K=63) and t*t (cachemir linear, K<=15).\n", stride);
}
