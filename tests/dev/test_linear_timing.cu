// Timing-isolation microbenchmark for the FHE linear layer.
//
// Goal: measure, in isolation, how long a single Q/K/V projection (and the two
// MLP projections) takes under CKKS — no softmax, no bootstrap, no decrypt in
// the timed region. Reference point: the paper reports ~0.15 s for one Q/K/V
// linear; this test tells us where we stand and decomposes the cost into its
// primitive parts (rotations / key-switches vs plaintext-mults).
//
// Run with FHE_PROFILE unset (we time with our own cudaEvents). Knobs:
//   ENCODE_LEVEL  CKKS level to encode weights+input at (default: the
//                 production post-bootstrap level — the level the linear input
//                 actually sits at in a real token).
//   REPS          timed repetitions per shape (default 20).
//   LINEAR_PACKING  cachemir (default) | diagonal

#include "model/gpt2.h"
#include "inference.h"
#include "test_helpers.h"
#include "packing/cachemir/cachemir_linear_utils.h"
#include "packing/cachemir/cachemir_attention_utils.h"
#include "ckks_types.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cctype>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

struct Shape {
    std::string label;
    int d_in;
    int d_out;
};

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

// Parse a comma-separated list of ints (e.g. "4,8,12,16").
std::vector<int> parse_csv_ints(const std::string& s) {
    std::vector<int> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') { if (!cur.empty()) out.push_back(std::stoi(cur)); cur.clear(); }
        else if (!isspace(static_cast<unsigned char>(c))) cur += c;
    }
    if (!cur.empty()) out.push_back(std::stoi(cur));
    return out;
}

// Mean / min / max of a cudaEvent-timed callable over `reps` iterations.
struct TimeStats { double mean_ms, min_ms, max_ms; };

template <typename F>
TimeStats time_op(int reps, F&& fn) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    double sum = 0.0, mn = 1e30, mx = 0.0;
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
        mx = std::max(mx, static_cast<double>(ms));
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return {sum / reps, mn, mx};
}

double bench_shape(Inference& inf, const Shape& s, int encode_level, int reps) {
    auto p = cachemir::compute_cm_params(inf.slots, s.d_in, s.d_out);

    // Predicted primitive counts (cachemir linear, see cachemir_linear.cu):
    //   phase 1 input interleave : log2(tp_in)         rotations
    //   phase 2 baby-step rotates: r_i - 1             rotations
    //   phase 3 mac              : r_i * r_o           plaintext-mults
    //   phase 4 cascade          : r_o - 1             rotations
    //   phase 5 output interleave: log2(tp_out)        rotations
    auto log2i = [](int x) { int n = 0; while ((1 << (n + 1)) <= x) ++n; return n; };
    const int n_rot_interleave_in  = log2i(p.tp_in);
    const int n_rot_baby           = p.r_i - 1;
    const int n_rot_cascade        = p.r_o - 1;
    const int n_rot_interleave_out = log2i(p.tp_out);
    const int n_rot  = n_rot_interleave_in + n_rot_baby + n_rot_cascade + n_rot_interleave_out;
    const int n_mult = p.r_i * p.r_o;

    std::cout << "\n================ [" << s.label << "  " << s.d_in << "->" << s.d_out
              << "] ================\n";
    std::cout << "  cachemir params: d=" << p.d << " alpha=" << p.alpha
              << " t=" << p.t << " tp=" << p.tp
              << " r_i=" << p.r_i << " r_o=" << p.r_o << " n_pt=" << p.n_pt << "\n";
    std::cout << "  predicted prims: rotations=" << n_rot
              << " (interleave_in=" << n_rot_interleave_in
              << " baby=" << n_rot_baby
              << " cascade=" << n_rot_cascade
              << " interleave_out=" << n_rot_interleave_out << ")"
              << "  pt_mults=" << n_mult << "\n";

    const std::string wname = s.label;
    auto W = random_matrix(s.d_in, s.d_out, /*seed=*/1234 + s.d_in + s.d_out);
    inf.w[wname] = encode_weight_matrix(inf, W, s.d_in, s.d_out, encode_level);
    inf.w.erase(wname + "_bias");  // pure matmul: linear() skips bias-add

    inf.size.hidDim = s.d_in;
    inf.size.dim    = s.d_in;

    auto x_vals = random_vector(s.d_in, /*seed=*/9876 + s.d_in);
    PackedCtx x = encode_linear_input(inf, x_vals, s.d_in, s.d_out, encode_level);
    std::cout << "  input ct level=" << level_of(x.ct)
              << "  (encode_level=" << encode_level << ")\n";

    // Warm-up: first calls pay one-time NTT-twiddle / workspace allocation.
    for (int w = 0; w < 3; ++w) {
        PackedCtx y = linear(inf, x, wname, s.d_in, s.d_out);
        (void)y;
    }

    auto full = time_op(reps, [&]() {
        PackedCtx y = linear(inf, x, wname, s.d_in, s.d_out);
        (void)y;
    });

    std::cout << std::fixed << std::setprecision(3)
              << "  >>> linear()     mean=" << full.mean_ms << " ms"
              << "  min=" << full.min_ms << "  max=" << full.max_ms << " ms"
              << "   (" << full.mean_ms / 1000.0 << " s)\n";
    if (n_rot > 0)
        std::cout << "      per-rotation (full/n_rot) ~ " << full.mean_ms / n_rot << " ms\n";

    // Phase breakdown: reset the step profiler, run the linear PHASE_REPS times
    // profiled, and dump the prepare_linear_input/apply_linear sub-phase tree.
    // Localizes the time NOT explained by raw key-switch/mult primitive sum.
    // Needs FHE_PROFILE=wall; dumps divided by PHASE_REPS for per-call ms.
    const int phase_reps = std::stoi(env_or("PHASE_REPS", "20"));
    if (inf.fhe->profile.on()) {
        inf.fhe->profile.reset();
        for (int r = 0; r < phase_reps; ++r) {
            PackedCtx y = linear(inf, x, wname, s.d_in, s.d_out);
            (void)y;
        }
        std::cout << "  [phase profile over " << phase_reps
                  << " calls; divide inc/self by " << phase_reps << " for per-call ms]\n";
        inf.fhe->profile.dump(std::cout);
    }

    inf.w.erase(wname);
    return full.mean_ms;
}

// Isolated single-primitive timing at the same level, to attribute the linear
// cost: one rotation (key-switch) and one plaintext-mult.
void bench_primitives(Inference& inf, int d, int encode_level, int reps) {
    auto p = cachemir::compute_cm_params(inf.slots, d, d);
    inf.size.hidDim = d;
    inf.size.dim    = d;

    auto x_vals = random_vector(d, 42);
    PackedCtx x = encode_linear_input(inf, x_vals, d, d, encode_level);

    // A rotation index the linear itself uses (key guaranteed to exist).
    const int idx = cachemir::mha_rot(inf, p.t * p.t);

    auto W = random_matrix(d, d, 7);
    auto pts = encode_weight_matrix(inf, W, d, d, encode_level);

    // warm-up
    for (int w = 0; w < 3; ++w) { auto r = inf.fhe->rotate(x, idx); (void)r; }

    auto rot = time_op(reps, [&]() {
        PackedCtx r = inf.fhe->rotate(x, idx);
        (void)r;
    });
    auto mul = time_op(reps, [&]() {
        PackedCtx m = inf.fhe->mult(x, pts[0]);
        (void)m;
    });

    std::cout << "\n---------------- isolated primitives @ level "
              << level_of(x.ct) << " ----------------\n"
              << std::fixed << std::setprecision(4)
              << "  rotate (1 key-switch): mean=" << rot.mean_ms << " ms  (min=" << rot.min_ms << ")\n"
              << "  mult   (ct x ptx):     mean=" << mul.mean_ms << " ms  (min=" << mul.min_ms << ")\n";
}

}  // namespace

TEST(LinearTiming, QKV_and_MLP) {
    const int logN      = default_logN();
    const int d         = std::stoi(env_or("HID_DIM", "1024"));  // packing dim (rot keys)
    const int d_exp     = 4 * d;
    const int num_heads = 16;

    const std::string packing_env = env_or("LINEAR_PACKING", "cachemir");
    const PackingKind kind =
        (packing_env == "diagonal") ? PackingKind::Diagonal : PackingKind::Cachemir;

    // Key-switch / chain parameters — the suspected per-primitive gap drivers.
    // total modulus depth = DEPTH + (ENABLE_BTS ? BTP_OVERHEAD : 0); the HYBRID
    // key-switch special-prime count P ≈ ceil((depth+1)/DNUM) is sized for this
    // TOTAL chain, so every rotation pays for it regardless of the ct's level.
    const int  depth      = std::stoi(env_or("DEPTH", "13"));
    const int  dnum       = std::stoi(env_or("DNUM", "3"));
    const int  btp_oh     = std::stoi(env_or("BTP_OVERHEAD", "16"));
    const bool enable_bts = env_or("ENABLE_BTS", "1") != "0";
    const int  total_depth = depth + (enable_bts ? btp_oh : 0);

    std::cout << "[test_linear_timing] packing=" << to_string(kind)
              << " logN=" << logN << " d=" << d
              << " depth=" << depth << " btp_oh=" << (enable_bts ? btp_oh : 0)
              << " total_depth=" << total_depth << " dnum=" << dnum
              << " bts=" << enable_bts << " creating context...\n";
    Inference inf = make_gpt2_inference({
        .ckks         = {.logN = logN,
                         .depth = depth,
                         .enable_bootstrap = enable_bts,
                         .btp_depth_overhead = static_cast<uint32_t>(btp_oh),
                         .num_large_digits = static_cast<uint32_t>(dnum),
                         .bts_iterations = default_bts_iterations()},
        .hidDim       = d,
        .expDim       = d_exp,
        .numHeads     = num_heads,
        .bench_mode   = false,
        .packing_kind = kind,
    });
    std::cout << "[test_linear_timing] context ready. slots=" << inf.slots << "\n";

    const int reps = std::stoi(env_or("REPS", "20"));
    // prod_level via a bootstrap probe only valid when bootstrap is enabled.
    const int prod_level = enable_bts
        ? static_cast<int>(inf.fhe->bootstrap_output_level()) : -1;

    // LEVEL_SWEEP: comma-list of encode levels (must be < total_depth-1). Default
    // sweeps within the available depth.
    const std::string sweep_env =
        env_or("LEVEL_SWEEP", "0,4,8,12,16,20," + std::to_string(total_depth - 3));
    std::vector<int> levels = parse_csv_ints(sweep_env);

    std::cout << "[test_linear_timing] bootstrap_output_level=" << prod_level
              << "  total_depth=" << total_depth << "  reps=" << reps
              << "  LEVEL_SWEEP=" << sweep_env << "\n";
    std::cout << "[test_linear_timing] paper reference: Q/K/V = 0.15 s (GPU), Llama-3-8B d=4096\n";

    const std::vector<Shape> shapes = {
        {"q", d, d},        // Q/K/V projection at the configured dim
    };

    std::vector<std::vector<double>> grid;  // grid[shape][level] = mean_ms
    for (const auto& s : shapes) {
        std::vector<double> row;
        for (int lv : levels) {
            std::cout << "\n######## " << s.label << "  ENCODE_LEVEL=" << lv
                      << (lv == prod_level ? "  (PRODUCTION)" : "") << " ########";
            bench_primitives(inf, s.d_in, lv, reps);
            row.push_back(bench_shape(inf, s, lv, reps));
        }
        grid.push_back(row);
    }

    // Summary: linear time vs level, with the paper's 0.15 s reference line.
    std::cout << "\n================ LEVEL SWEEP SUMMARY (linear mean ms) ================\n";
    std::cout << std::left << std::setw(12) << "level";
    for (int lv : levels) std::cout << std::right << std::setw(10) << lv;
    std::cout << "\n";
    for (size_t si = 0; si < shapes.size(); ++si) {
        std::cout << std::left << std::setw(12) << shapes[si].label;
        for (double ms : grid[si])
            std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(1) << ms;
        std::cout << "\n";
    }
    std::cout << "paper Q/K/V (d=4096) GPU reference = 150.0 ms\n";
    std::cout << "(higher level = fewer RNS limbs = faster; prod_level=" << prod_level << ")\n";
}
