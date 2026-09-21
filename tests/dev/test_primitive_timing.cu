// app:primitives — per-primitive CKKS cost at the PAPER's deployment parameters.
//
// The paper reports every latency as a measured end-to-end run; this is the microbenchmark
// behind its primitive table. Unlike test_linear_timing.cu (which hand-picks DEPTH=13/DNUM=3
// to isolate key-switch drivers), this builds the context from default_ckks_options() — the
// SAME struct the production runs use — so the numbers are the deployment's, not a probe's:
//   logN 16 (N=2^16) | depth 11 usable levels | dnum 7 | first mod 60 bit | scale 58 bit
//   bootstrap on, (4,3) CtS/StC budget, 1 iteration, sparse ternary h=192.
//
// DEFAULT OUTPUT = one row per primitive, timed in isolation, at ONE fixed operating point:
// the level a ciphertext re-enters at after a bootstrap, which is where the circuit runs. A
// ciphertext always sits at some level, so the honest framing is "fixed level", not "no level";
// what the table reports besides time is LEVELS CONSUMED, which is the quantity HEAT optimises.
// (Arithmetic cost does scale with the remaining RNS limbs -- ct x ct runs 0.48 ms with 10
// limbs left and 1.64 ms with 26 -- so set LEVEL_SWEEP=2,6,10,14,16,18 for that curve. A
// bootstrap is ~83 ms at every level: it rebuilds the whole chain regardless.)
//
// Knobs: REPS (default 100), LEVEL_SWEEP (comma list; default spans the usable chain),
//        HID_DIM (default 768 = GPT-2).
// Run:   TASK=... the gtest binary directly; see cluster/primitive_timing.slurm.

#include "inference.h"
#include "test_helpers.h"
#include "ckks_types.h"
#include "packing/cachemir/cachemir_linear_utils.h"
#include "packing/cachemir/cachemir_attention_utils.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace test_helpers;   // env_or, default_ckks_options, default_bts_iterations

namespace {

struct TimeStats {
    double mean_ms, min_ms;
};

template <class F>
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
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        sum += ms;
        mn = std::min(mn, static_cast<double>(ms));
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return {sum / reps, mn};
}

std::vector<double> random_vector(int n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<double> v(n);
    for (auto& x : v) x = d(gen);
    return v;
}

std::vector<std::vector<double>> random_matrix(int r, int c, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> d(-0.05, 0.05);
    std::vector<std::vector<double>> m(r, std::vector<double>(c));
    for (auto& row : m)
        for (auto& x : row) x = d(gen);
    return m;
}

std::vector<int> parse_csv_ints(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) out.push_back(std::stoi(tok));
    return out;
}

}  // namespace

TEST(PrimitiveTiming, DeploymentParameters) {
    const int d    = std::stoi(env_or("HID_DIM", "768"));
    const int reps = std::stoi(env_or("REPS", "100"));

    // THE deployment context: production defaults, untouched.
    CKKSContextOptions ckks = default_ckks_options();
    Inference inf = make_gpt2_inference({
        .ckks         = ckks,
        .hidDim       = d,
        .expDim       = 4 * d,
        .numHeads     = 12,
        .bench_mode   = false,
        .packing_kind = PackingKind::Cachemir,
    });

    const int total_depth = ckks.depth + (ckks.enable_bootstrap
                                          ? static_cast<int>(ckks.btp_depth_overhead) : 0);
    const int prod_level  = ckks.enable_bootstrap
                          ? static_cast<int>(inf.fhe->bootstrap_output_level()) : -1;

    std::cout << "[primitive_timing] logN=" << ckks.logN << " depth=" << ckks.depth
              << " dnum=" << ckks.num_large_digits << " btp_oh=" << ckks.btp_depth_overhead
              << " total_depth=" << total_depth << " h=" << ckks.h_weight
              << " slots=" << inf.slots << " reps=" << reps
              << " bootstrap_output_level=" << prod_level << "\n";

    // ONE operating point by default: the level a ciphertext re-enters at after a bootstrap,
    // which is where the deployed circuit actually runs. A ciphertext always sits at SOME
    // level (its limb count is what the arithmetic costs), so "no level" means "fixed level",
    // not "level-free" -- we fix it here and report cost in ms and in LEVELS CONSUMED instead.
    // LEVEL_SWEEP is opt-in for anyone who wants the limb-scaling curve.
    const int chain_limbs = total_depth + 1;   // 28 at the deployment parameters
    std::vector<int> levels = parse_csv_ints(
        env_or("LEVEL_SWEEP", std::to_string(prod_level)));

    auto p   = cachemir::compute_cm_params(inf.slots, d, d);
    const int rot_idx = cachemir::mha_rot(inf, p.t * p.t);   // a key that provably exists

    // One row per primitive, measured in isolation. "levels" is MEASURED (effective level after
    // the op minus before -- see the eff() lambda: level_of plus a pending-rescale correction),
    // not assumed; bootstrap reports the levels it gives BACK, hence negative.
    auto row = [](const char* name, int levels, TimeStats t, double add_ref) {
        std::cout << std::left << std::setw(20) << name
                  << std::right << std::setw(8)
                  << (levels < 0 ? "-" + std::to_string(-levels) : std::to_string(levels))
                  << std::setw(12) << std::fixed << std::setprecision(3) << t.mean_ms
                  << std::setw(11) << t.min_ms
                  << std::setw(11) << std::setprecision(1) << (t.mean_ms / add_ref) << "\n";
    };

    for (int lv : levels) {
        if (lv < 0 || lv >= total_depth - 1) continue;

        auto x_vals = random_vector(d, 42);
        PackedCtx x = encode_linear_input(inf, x_vals, d, d, lv);
        PackedCtx y = encode_linear_input(inf, random_vector(d, 43), d, d, lv);
        auto W   = random_matrix(d, d, 7);
        auto pts = encode_weight_matrix(inf, W, d, d, lv);

        for (int w = 0; w < 3; ++w) { auto r = inf.fhe->rotate(x, rot_idx); (void)r; }

        auto t_add  = time_op(reps, [&]() { auto r = inf.fhe->add(x, y);       (void)r; });
        auto t_sub  = time_op(reps, [&]() { auto r = inf.fhe->sub(x, y);       (void)r; });
        auto t_mpt  = time_op(reps, [&]() { auto r = inf.fhe->mult(x, pts[0]); (void)r; });
        auto t_mct  = time_op(reps, [&]() { auto r = inf.fhe->mult(x, y);      (void)r; });
        auto t_sq   = time_op(reps, [&]() { auto r = inf.fhe->square(x);       (void)r; });
        auto t_rot  = time_op(reps, [&]() { auto r = inf.fhe->rotate(x, rot_idx); (void)r; });

        // Bootstrap is measured where it is actually CALLED: on a nearly-exhausted ciphertext
        // at the auto-bts trigger level, not at the level it outputs (timing it on an already
        // refreshed ct also makes its level delta 0 by construction). It consumes its input,
        // so each rep needs a fresh one; the encode sits OUTSIDE the timed region.
        const int bts_in_lv = static_cast<int>(ckks.auto_bts_level_override);
        double bts_ms = 0.0, bts_min = 1e30;
        int c_bts = 0;
        if (ckks.enable_bootstrap) {
            double sum = 0.0;
            for (int r = 0; r < reps; ++r) {
                PackedCtx z = encode_linear_input(inf, x_vals, d, d, bts_in_lv);
                const int before = static_cast<int>(level_of(z.ct));
                auto t = time_op(1, [&]() { inf.fhe->bootstrap(z.ct); });
                if (r == 0) c_bts = static_cast<int>(level_of(z.ct)) - before;
                sum += t.mean_ms;
                bts_min = std::min(bts_min, t.mean_ms);
            }
            bts_ms = sum / reps;
        }

        // Levels consumed, MEASURED. Rescale in this wrapper is LAZY: after a multiplication
        // the ciphertext sits at the same level with a pending rescale, flagged by
        // GetNoiseScaleDeg()==2 (bootstrap_hint corrects for exactly this). Reading level_of()
        // alone therefore reports 0 for every op -- the effective level is what counts.
        auto eff = [](const PackedCtx& p) {
            return static_cast<int>(level_of(p.ct)) + (p.ct->GetNoiseScaleDeg() == 2 ? 1 : 0);
        };
        const int lv0 = eff(x);
        auto consumed = [&](auto&& make) { return eff(make()) - lv0; };
        const int c_add = consumed([&] { return inf.fhe->add(x, y); });
        const int c_sub = consumed([&] { return inf.fhe->sub(x, y); });
        const int c_mpt = consumed([&] { return inf.fhe->mult(x, pts[0]); });
        const int c_mct = consumed([&] { return inf.fhe->mult(x, y); });
        const int c_sq  = consumed([&] { return inf.fhe->square(x); });
        const int c_rot = consumed([&] { return inf.fhe->rotate(x, rot_idx); });

        std::cout << "\nprimitives in isolation @ " << (chain_limbs - lv0) << " limbs"
                  << (lv == prod_level ? "  (the level a ciphertext re-enters at after a bootstrap"
                                         " -- where the circuit runs)" : "")
                  << "\n\n"
                  << std::left << std::setw(20) << "primitive"
                  << std::right << std::setw(8) << "levels"
                  << std::setw(12) << "mean ms" << std::setw(11) << "min ms"
                  << std::setw(11) << "x add" << "   (n=" << reps << " each)\n";
        row("add (ct + ct)",      c_add, t_add, t_add.mean_ms);
        row("sub (ct - ct)",      c_sub, t_sub, t_add.mean_ms);
        row("mult (ct x ptx)",    c_mpt, t_mpt, t_add.mean_ms);
        row("mult (ct x ct)",     c_mct, t_mct, t_add.mean_ms);
        row("square (ct^2)",      c_sq,  t_sq,  t_add.mean_ms);
        row("rotate (1 keysw.)",  c_rot, t_rot, t_add.mean_ms);
        row("bootstrap",          c_bts, TimeStats{bts_ms, bts_min}, t_add.mean_ms);
        std::cout << "\n(bootstrap timed on a ct at its auto-bts trigger level "
                  << bts_in_lv << ", i.e. where it is really called; it returns to level "
                  << prod_level << ")\n";
    }

    std::cout << "\n[primitive_timing] a bootstrap buys back the whole chain; the table says what\n"
                 "                   one costs against what one iteration of a solver costs.\n";
}
