// Measures the cost of an explicit drop_to_level (ModReduce limb drop) and the NET gain of a
// pre-linear level lift: drop the linear's INPUT high (cheap, fewer limbs) BEFORE its rotations
// so the keyswitch-heavy linear runs at the cheaper limb-band. Net = linear(@low) − linear(@high)
// − drop_cost. This decides whether the post-hoc planner lift (place a drop_to_level like a bts
// before kv/up) is worth it. Complex regime (CachemirComplex), gpt2_diff configs.
//
// Knobs: START (input level, default 17)  LEVEL_SWEEP (csv targets, default 18..24)  REPS (50)

#include "model/gpt2.h"
#include "inference.h"
#include "config_loader.h"
#include "weight_loader.h"
#include "packing/cachemir/cachemir_linear_utils.h"
#include "test_helpers.h"
#include "ckks_types.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

std::vector<int> parse_csv_ints(const std::string& s) {
    std::vector<int> out; std::string cur;
    for (char c : s) {
        if (c == ',') { if (!cur.empty()) out.push_back(std::stoi(cur)); cur.clear(); }
        else if (!std::isspace((unsigned char)c)) cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(std::stoi(cur));
    return out;
}

template <typename F>
double time_ms(int reps, F&& fn) {
    cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    double sum = 0;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(s); fn(); cudaEventRecord(e); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, s, e); sum += ms;
    }
    cudaEventDestroy(s); cudaEventDestroy(e);
    return sum / reps;
}

TEST(DropTiming, DropToLevelCostAndNetGain) {
    Inference inf = make_gpt2_inference({
        .ckks = {.bts_iterations = default_bts_iterations()},
        .packing_kind = PackingKind::CachemirComplex,
    });
    auto parsed = config_loader::parse_configs_json(
        config_loader::read_file_to_string(default_configs_path()));
    weight_loader::prepare_gpt2_layer_configs(inf, parsed, 0);

    const int d     = inf.size.hidDim;
    const int d_exp = inf.size.expDim;
    const int reps  = std::stoi(env_or("REPS", "50"));
    const int start = std::stoi(env_or("START", "17"));
    auto levels = parse_csv_ints(env_or("LEVEL_SWEEP", "18,19,20,21,22,23,24"));

    std::mt19937 g(7); std::uniform_real_distribution<double> dist(-1, 1);
    auto rnd = [&](int n){ std::vector<double> v(n); for (auto& x : v) x = dist(g); return v; };
    auto lvl = [&](const PackedCtx& p){ return inf.fhe->level_for_ct(p.ct); };

    std::cout << "[drop] d=" << d << " start=" << start << " reps=" << reps << "\n";

    // ---- 1) isolated drop_to_level cost: encode reps cts at `start`, drop each to target ----
    std::cout << "\n[drop] === drop_to_level cost (from level " << start << ") ===\n";
    std::cout << "  target  Δlevels   ms/drop\n";
    for (int L : levels) {
        if (L <= start) continue;
        std::vector<PackedCtx> cts;
        cts.reserve(reps);
        for (int r = 0; r < reps; ++r) cts.push_back(encode_linear_input(inf, rnd(d), d, d, start));
        cudaDeviceSynchronize();
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int r = 0; r < reps; ++r) inf.fhe->drop_to_level(cts[r], L);   // each ct dropped once
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, s, e); cudaEventDestroy(s); cudaEventDestroy(e);
        std::cout << "  " << std::setw(6) << L << std::setw(9) << (L - start)
                  << std::setw(10) << std::fixed << std::setprecision(3) << ms / reps << "\n";
    }

    // ---- 2) NET gain for a real linear: baseline (input@start) vs lift (drop start→L, linear@L) ----
    // up is a standalone linear (d→d_exp) → its input CAN be dropped before its rotations.
    auto net_for = [&](const char* tag, int din, int dout) {
        std::cout << "\n[drop] === net gain: " << tag << " (" << din << "→" << dout << ") ===\n";
        const std::string wn = tag;
        std::cout << "  case            input_lvl   linear_ms\n";
        // baseline: linear with input at `start`
        inf.w[wn] = encode_weight_matrix(inf, std::vector<std::vector<double>>(din, rnd(dout)), din, dout, start);
        inf.w.erase(wn + std::string("_bias"));
        PackedCtx xb = encode_linear_input(inf, rnd(din), din, dout, start);
        double base = time_ms(reps, [&]{ PackedCtx y = linear(inf, xb, wn, din, dout); (void)y; });
        std::cout << "  baseline        " << std::setw(9) << lvl(xb)
                  << std::setw(12) << std::fixed << std::setprecision(2) << base << "\n";
        // lifted: encode weight at target L, input dropped start→L, linear@L
        for (int L : levels) {
            if (L <= start) continue;
            inf.w[wn] = encode_weight_matrix(inf, std::vector<std::vector<double>>(din, rnd(dout)), din, dout, L);
            inf.w.erase(wn + std::string("_bias"));
            // drop cost (one-shot, amortized into the per-call linear comparison)
            std::vector<PackedCtx> xs; xs.reserve(reps);
            for (int r = 0; r < reps; ++r) xs.push_back(encode_linear_input(inf, rnd(din), din, dout, start));
            cudaDeviceSynchronize();
            cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
            cudaEventRecord(s);
            for (int r = 0; r < reps; ++r) inf.fhe->drop_to_level(xs[r], L);
            cudaEventRecord(e); cudaEventSynchronize(e);
            float dms = 0; cudaEventElapsedTime(&dms, s, e); dms /= reps;
            cudaEventDestroy(s); cudaEventDestroy(e);
            PackedCtx xl = encode_linear_input(inf, rnd(din), din, dout, start);
            inf.fhe->drop_to_level(xl, L);
            double lin = time_ms(reps, [&]{ PackedCtx y = linear(inf, xl, wn, din, dout); (void)y; });
            std::cout << "  drop→" << std::setw(2) << L << " +lin   " << std::setw(9) << lvl(xl)
                      << std::setw(12) << lin
                      << "   (drop " << std::setprecision(2) << dms << "ms, total "
                      << lin + dms << "ms, net vs base "
                      << std::showpos << base - (lin + dms) << std::noshowpos << "ms)\n";
        }
    };
    net_for("up", d, d_exp);     // MLP up: standalone, feeds gelu (self-bts) → liftable
    net_for("q",  d, d);         // attention-shape reference

    SUCCEED();
}

}  // namespace
