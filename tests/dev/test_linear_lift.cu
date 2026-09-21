// Phase-0a de-risk gate for the level-placement-lift feature
// (docs/speed/level_placement_lift.md). One primitive (linear), tested in
// ISOLATION, exercising the EXACT mechanism the feature will use.
//
// A linear input sits at its natural post-bootstrap level (L16-18). The feature
// RAISES it to the cliff (L23) with drop_to_level() — a free, value-preserving
// tower drop (no rescale, deg preserved) — so the linear runs on fewer RNS limbs
// and is cheaper (cost is limb-bound, doc §1). This test proves, per the doc's
// core trade-off (§2), the two claims that make the lever real:
//
//   (1) BIT-STABLE: the drop preserves the slot values AND the noise degree, and
//       the lifted linear gives the SAME answer as the natural-level linear (both
//       match the torch oracle). Drops are level-only (doc §7 #3).
//   (2) CHEAPER:    the lifted linear is measurably faster (doc §1 cost curve).
//
// Distinct from test_op_level_sweep.cu (which times each level by encoding
// DIRECTLY at it, random inputs, no correctness check): here we go through the
// real drop_to_level() path with real weights + an oracle, which is what the
// planner-scheduled drop (Phase 1+) actually does at runtime.
//
// Production THOR chain (default make_gpt2_inference: depth 11, total_depth 27,
// post-bts L16, level_limit 24, linear cliff L23). FHE_PROFILE must be UNSET
// (we time with cudaEvents).
//
// Knobs: NAT_LEVELS (csv, default 16,17,18)   LIFT_LEVEL (default 23)
//        REPS (default 20)

#include "all_blocks_test_helpers.h"
#include "model/gpt2.h"
#include "test_helpers.h"
#include "math/matrix_ops.h"
#include "inference.h"
#include "ckks_types.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

struct TimeStats { double mean_ms, min_ms; };

// cudaEvent-timed mean/min of a callable over `reps` iterations (FHE_PROFILE off).
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

std::vector<int> parse_csv_ints(const std::string& s) {
    std::vector<int> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') { if (!cur.empty()) out.push_back(std::stoi(cur)); cur.clear(); }
        else if (!std::isspace(static_cast<unsigned char>(c))) cur += c;
    }
    if (!cur.empty()) out.push_back(std::stoi(cur));
    return out;
}

struct LinearCase {
    std::string label;        // "up", "down"
    std::string wname;        // tensor name in manifest (nn.Linear layout)
    std::string bname;
    int d_in_real, d_out_real, d_in_pad, d_out_pad;
    std::string inp_key;      // all_blocks L=0 tap for the input
    std::string res_key;      // all_blocks L=0 tap for x@W+b (the oracle)
};

void run_lift_case(Inference& inf,
                   const weight_loader::WeightStore& store,
                   const LinearCase& c,
                   const std::string& io_dir,
                   int T_val,
                   const std::vector<int>& nat_levels,
                   int lift_level,
                   int reps) {
    std::cout << "\n############ [" << c.label << "  " << c.d_in_real << "x"
              << c.d_out_real << "] ############\n";

    auto W_linear = store.tensor2d(c.wname, c.d_out_real, c.d_in_real);
    auto W_real   = matrix::transpose(W_linear);
    auto W_pad    = matrix::pad_matrix(W_real, c.d_in_pad, c.d_out_pad);
    auto b_real   = store.tensor1d(c.bname, c.d_out_real);
    auto b_pad    = matrix::pad_vector(b_real, c.d_out_pad);

    const std::string io_path = all_blocks_io_path(io_dir, /*block_idx=*/0, T_val);
    if (!probe_io_file(io_path)) { std::cout << "[skip] no io file: " << io_path << "\n"; return; }

    auto io = read_block0_io(io_path, c.inp_key, c.res_key);
    ASSERT_FALSE(io.inp.empty()) << "empty inp in " << io_path;
    ASSERT_EQ(static_cast<int>(io.inp[0].size()), c.d_in_real);
    ASSERT_EQ(static_cast<int>(io.res[0].size()), c.d_out_real);

    const auto& x_real   = io.inp[0];
    const auto& y_oracle = io.res[0];               // torch x@W+b
    auto x_pad = matrix::pad_vector(x_real, c.d_in_pad);

    // cachemir linear params key off hidDim — set to this op's shape.
    inf.size.hidDim = c.d_in_pad;
    inf.size.dim    = c.d_in_pad;

    auto run_linear_decode = [&](const PackedCtx& x) {
        PackedCtx y = linear(inf, x, c.label, c.d_in_pad, c.d_out_pad);
        auto raw = decrypt_slots(inf, y);
        auto out = decode_linear_output(inf.packing, raw, inf.slots, c.d_in_pad, c.d_out_pad);
        out.resize(c.d_out_real);
        return out;
    };

    for (int L_nat : nat_levels) {
        std::cout << "\n=== [" << c.label << "] L_nat=" << L_nat
                  << "  ->  lift L" << lift_level << " ===\n";

        // ---- NATURAL arm: input + weights both at L_nat (ct.level == pt.level). ----
        inf.w[c.label] = encode_weight_matrix(inf, W_pad, c.d_in_pad, c.d_out_pad, L_nat);
        inf.w[c.label + "_bias"] = { encode_bias_vector(inf, b_pad, c.d_in_pad, c.d_out_pad) };
        PackedCtx x_nat = encode_linear_input(inf, x_pad, c.d_in_pad, c.d_out_pad, L_nat);
        EXPECT_EQ(static_cast<int>(level_of(x_nat.ct)), L_nat)
            << "natural input did not encode at L_nat";

        for (int w = 0; w < 3; ++w) {            // warm-up (NTT twiddles / workspace)
            PackedCtx y = linear(inf, x_nat, c.label, c.d_in_pad, c.d_out_pad); (void)y;
        }
        auto t_nat = time_op(reps, [&] {
            PackedCtx y = linear(inf, x_nat, c.label, c.d_in_pad, c.d_out_pad); (void)y;
        });
        auto y_nat = run_linear_decode(x_nat);

        // ---- LIFTED arm: encode at L_nat, drop_to_level(lift), weights at lift. ----
        PackedCtx x_lift = encode_linear_input(inf, x_pad, c.d_in_pad, c.d_out_pad, L_nat);
        auto      v_before  = decrypt_slots(inf, x_lift);
        const int deg_before = static_cast<int>(x_lift.ct->GetNoiseScaleDeg());

        inf.fhe->drop_to_level(x_lift, lift_level);

        const int deg_after = static_cast<int>(x_lift.ct->GetNoiseScaleDeg());
        auto      v_after   = decrypt_slots(inf, x_lift);

        EXPECT_EQ(static_cast<int>(level_of(x_lift.ct)), lift_level)
            << "drop_to_level did not land at the lift level";
        EXPECT_EQ(deg_after, deg_before)
            << "drop changed the noise degree (deg-2 landmine — drop must NOT rescale)";

        auto drop_stab = compare_vec(v_after, v_before);
        std::cout << std::scientific << std::setprecision(3)
                  << "[drop]        value-preserve  max_abs=" << drop_stab.max_abs
                  << "  max_rel=" << drop_stab.max_rel
                  << "  deg " << deg_before << "->" << deg_after << "\n";
        EXPECT_LT(drop_stab.max_abs, 1e-3) << "drop_to_level changed the slot values";

        // weights re-encoded at the dropped level so the linear runs there.
        inf.w[c.label] = encode_weight_matrix(inf, W_pad, c.d_in_pad, c.d_out_pad, lift_level);
        inf.w[c.label + "_bias"] = { encode_bias_vector(inf, b_pad, c.d_in_pad, c.d_out_pad) };

        for (int w = 0; w < 3; ++w) {
            PackedCtx y = linear(inf, x_lift, c.label, c.d_in_pad, c.d_out_pad); (void)y;
        }
        auto t_lift = time_op(reps, [&] {
            PackedCtx y = linear(inf, x_lift, c.label, c.d_in_pad, c.d_out_pad); (void)y;
        });
        auto y_lift = run_linear_decode(x_lift);

        // ---- correctness: lift transparent vs natural AND vs oracle ----
        auto s_nat   = compare_vec(y_nat,  y_oracle);
        auto s_lift  = compare_vec(y_lift, y_oracle);
        auto s_delta = compare_vec(y_lift, y_nat);
        std::cout << std::scientific << std::setprecision(3)
                  << "[correctness] nat  vs oracle  mean_rel=" << s_nat.mean_rel
                  << "  max_rel=" << s_nat.max_rel << "\n"
                  << "              lift vs oracle  mean_rel=" << s_lift.mean_rel
                  << "  max_rel=" << s_lift.max_rel << "\n"
                  << "              lift vs nat     mean_rel=" << s_delta.mean_rel
                  << "  max_rel=" << s_delta.max_rel << "\n";

        // ---- the lever: lifted op must be cheaper ----
        const double delta = t_lift.mean_ms - t_nat.mean_ms;
        std::cout << std::fixed << std::setprecision(2)
                  << "[lift] " << c.label << "  L" << L_nat << "->L" << lift_level
                  << "   nat="  << t_nat.mean_ms  << "ms (min " << t_nat.min_ms  << ")"
                  << "   lift=" << t_lift.mean_ms << "ms (min " << t_lift.min_ms << ")"
                  << "   delta=" << delta << "ms"
                  << (delta < 0.0 ? "   [CHEAPER]" : "   [NOT CHEAPER]") << "\n";

        // ---- gates ----
        EXPECT_LT(s_nat.mean_rel,   0.01) << "natural-level linear broke vs oracle (L" << L_nat << ")";
        EXPECT_LT(s_lift.mean_rel,  0.01) << "lifted linear broke vs oracle (L" << L_nat << ")";
        EXPECT_LT(s_delta.mean_rel, 0.01) << "lift changed the answer vs natural (L" << L_nat << ")";
        EXPECT_LT(t_lift.mean_ms, t_nat.mean_ms) << "lift not cheaper (L" << L_nat << ")";
    }

    inf.w.erase(c.label);
    inf.w.erase(c.label + "_bias");
}

}  // namespace

TEST(LinearLift, DropToCliffIsCheapAndBitStable) {
    const int d         = 1024;   // GPT-2 hidden, padded to next pow2
    const int d_exp     = 4096;   // GPT-2 expanded, padded to next pow2
    const int num_heads = 16;

    std::cout << "[test_linear_lift] creating production THOR context (logN=16)...\n";
    Inference inf = make_gpt2_inference({
        .ckks         = {.bts_iterations = default_bts_iterations()},
        .hidDim       = d,
        .expDim       = d_exp,
        .numHeads     = num_heads,
        .bench_mode   = false,
        .packing_kind = PackingKind::Cachemir,   // decode path = the feature's scope
    });
    std::cout << "[test_linear_lift] context ready. slots=" << inf.slots
              << "  level_limit=" << inf.fhe->level_limit()
              << "  bootstrap_output_level=" << inf.fhe->bootstrap_output_level() << "\n";

    const std::string weights_path = default_weights_path();
    const std::string io_dir       = default_all_blocks_io_dir();
    {
        std::ifstream probe(weights_path);
        if (!probe) GTEST_SKIP() << "weights not available: " << weights_path;
    }
    auto store = load_store(weights_path);

    const std::vector<int> nat_levels = parse_csv_ints(env_or("NAT_LEVELS", "16,17,18"));
    const int lift_level = std::stoi(env_or("LIFT_LEVEL", "23"));
    const int reps       = std::stoi(env_or("REPS", "20"));
    const int T_val      = default_t_sweep_val();

    std::cout << "[test_linear_lift] NAT_LEVELS=";
    for (int L : nat_levels) std::cout << L << " ";
    std::cout << " LIFT_LEVEL=" << lift_level << " REPS=" << reps << " T=" << T_val << "\n";

    // MLP up/down: real weights with clean block-0 oracle taps (res = x@W+b).
    const std::vector<LinearCase> cases = {
        {"up",   "transformer.h.0.mlp.c_fc.weight",   "transformer.h.0.mlp.c_fc.bias",
         768, 3072, 1024, 4096, "ln_2_out",  "pre_gelu"},
        {"down", "transformer.h.0.mlp.c_proj.weight", "transformer.h.0.mlp.c_proj.bias",
         3072, 768, 4096, 1024, "post_gelu", "mlp_out"},
    };

    for (const auto& c : cases)
        run_lift_case(inf, store, c, io_dir, T_val, nat_levels, lift_level, reps);
}
