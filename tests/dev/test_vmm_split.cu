// Regression guard for the cachemir VMM contraction-BSGS (now the production default).
//
// The contraction is tiled by r_i = d/t blocks; r_o tiles the output expansion (alpha).
// compute_cm_params factors the r_i baby scan into bstep_c * gstep_c (paper Algorithm
// gemv): bstep_c hoisted baby rotations + (gstep_c-1) giant rotations, via the identity
//   rot(x,(g*bstep_c+b)*t^2)*P = rot(rot(x,b*t^2)*rot(P,-g*bstep_c*t^2), g*bstep_c*t^2),
// with the giant weight = the legacy weight at the pre-rotated slot i-g*bstep_c*t^2
// (encode_weight_matrix). This cut the square 1024x1024 from 41 -> 20 rotations (~1.3x,
// pt-mults unchanged); the original flat-vs-balanced A/B + speed proof was job 47490645.
//
// This test now asserts, env-free, that the DEFAULT split is (1) CORRECT (matches x*W to
// CKKS precision) and (2) BALANCED & ACTIVE (factored, fewer rotations than the flat scan)
// — so a revert to the flat scan or a contraction-tiling regression is caught.
//
// Knobs: ENCODE_LEVEL (default 16, production post-bootstrap band), REPS (default 30).

#include "model/gpt2.h"
#include "inference.h"
#include "test_helpers.h"
#include "packing/cachemir/cachemir_linear_utils.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

struct Shape { std::string label; int d_in; int d_out; };

std::vector<std::vector<double>> random_matrix(int d_in, int d_out, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-0.05, 0.05);
    std::vector<std::vector<double>> W(d_in, std::vector<double>(d_out));
    for (int i = 0; i < d_in; ++i)
        for (int j = 0; j < d_out; ++j) W[i][j] = dist(gen);
    return W;
}

std::vector<double> random_vector(int d, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> x(d);
    for (int i = 0; i < d; ++i) x[i] = dist(gen);
    return x;
}

struct TimeStats { double mean_ms, min_ms; };

template <typename F>
TimeStats time_op(int reps, F&& fn) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    double sum = 0.0, mn = 1e30;
    for (int r = 0; r < reps; ++r) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        fn();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        sum += ms;
        mn = std::min(mn, static_cast<double>(ms));
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return {sum / reps, mn};
}

int log2i(int x) { int n = 0; while ((1 << (n + 1)) <= x) ++n; return n; }

// Rotation count for a given (bstep_c, gstep_c) split.
int rot_count(const cachemir::CacheMirParams& p, int bstep_c, int gstep_c) {
    return log2i(p.tp_in) + (bstep_c - 1) + (gstep_c - 1) + (p.r_o - 1) + log2i(p.tp_out);
}

}  // namespace

TEST(VmmSplit, BalancedDefaultCorrectAndActive) {
    const int logN = 16, d = 1024, d_exp = 4096, num_heads = 16;
    std::cout << "[test_vmm_split] creating cachemir CKKS context (logN=" << logN << ")...\n";
    Inference inf = make_gpt2_inference({
        .ckks         = {.bts_iterations = default_bts_iterations()},
        .hidDim       = d,
        .expDim       = d_exp,
        .numHeads     = num_heads,
        .bench_mode   = false,
        .packing_kind = PackingKind::Cachemir,
    });
    std::cout << "[test_vmm_split] context ready. slots=" << inf.slots << "\n";

    const int encode_level = std::stoi(env_or("ENCODE_LEVEL", "16"));
    const int reps         = std::stoi(env_or("REPS", "30"));

    const std::vector<Shape> shapes = {
        {"q",    d,     d},      // square Q/K/V/out
        {"up",   d,     d_exp},  // MLP up
        {"down", d_exp, d},      // MLP down
    };

    std::cout << "\n" << std::left << std::setw(8) << "shape"
              << std::right << std::setw(16) << "split(b x g x ro)"
              << std::setw(10) << "#rot" << std::setw(12) << "flat#rot"
              << std::setw(8) << "#mul" << std::setw(12) << "mean(ms)"
              << std::setw(12) << "mean_rel" << "\n";
    std::cout << std::string(78, '-') << "\n";

    for (const auto& s : shapes) {
        inf.size.hidDim = s.d_in;
        inf.size.dim    = s.d_in;

        auto W  = random_matrix(s.d_in, s.d_out, /*seed=*/1234u + s.d_in + s.d_out);
        auto xv = random_vector(s.d_in, /*seed=*/9876u + s.d_in);

        std::vector<double> y_ref(s.d_out, 0.0);
        for (int j = 0; j < s.d_out; ++j) {
            double a = 0.0;
            for (int i = 0; i < s.d_in; ++i) a += xv[i] * W[i][j];
            y_ref[j] = a;
        }

        const auto p = cachemir::compute_cm_params(inf.slots, s.d_in, s.d_out);

        const std::string wname = s.label;
        inf.w[wname] = encode_weight_matrix(inf, W, s.d_in, s.d_out, encode_level);
        inf.w.erase(wname + "_bias");
        PackedCtx x = encode_linear_input(inf, xv, s.d_in, s.d_out, encode_level);

        for (int w = 0; w < 3; ++w) { PackedCtx y = linear(inf, x, wname, s.d_in, s.d_out); (void)y; }
        auto t = time_op(reps, [&]() { PackedCtx y = linear(inf, x, wname, s.d_in, s.d_out); (void)y; });

        PackedCtx y   = linear(inf, x, wname, s.d_in, s.d_out);
        auto      raw = decrypt_slots(inf, y);
        auto      got = decode_linear_output(inf.packing, raw, inf.slots, s.d_in, s.d_out);
        got.resize(s.d_out);
        auto cmp = compare_vec(got, y_ref);
        inf.w.erase(wname);

        const int n_rot    = rot_count(p, p.bstep_c, p.gstep_c);
        const int flat_rot = rot_count(p, p.r_i, 1);   // the legacy flat scan

        std::cout << std::left << std::setw(8) << s.label
                  << std::right << std::setw(16)
                  << (std::to_string(p.bstep_c) + "x" + std::to_string(p.gstep_c)
                      + "x" + std::to_string(p.r_o))
                  << std::setw(10) << n_rot << std::setw(12) << flat_rot
                  << std::setw(8) << (p.r_i * p.r_o)
                  << std::setw(12) << std::fixed << std::setprecision(3) << t.mean_ms
                  << std::setw(12) << std::scientific << std::setprecision(2) << cmp.mean_rel
                  << "\n";

        // (1) CORRECTNESS: the default (balanced) VMM computes x*W to CKKS precision.
        EXPECT_LT(cmp.mean_rel, 1e-2) << s.label << " default VMM is numerically WRONG";
        // (2) split is internally consistent: bstep_c * gstep_c covers the r_i contraction.
        EXPECT_EQ(p.bstep_c * p.gstep_c, p.r_i) << s.label << " bstep_c*gstep_c != r_i";
        // (3) BALANCED & ACTIVE: for a factorable r_i the default must NOT be the flat scan,
        //     and must use strictly fewer rotations than flat.
        if (p.r_i > 2) {
            EXPECT_LT(p.bstep_c, p.r_i) << s.label << " default reverted to the flat baby scan";
            EXPECT_LT(n_rot, flat_rot)  << s.label << " balanced split is not fewer rotations than flat";
        }
    }
}
