#include "all_blocks_test_helpers.h"
#include "cutmax.h"                 // cutmax_tile_col_of_slot (z-decode)
#include "model/gpt2.h"
#include "model/gpt2_model.h"
#include "test_helpers.h"
#include "weight_loader.h"
#include "config_loader.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

std::vector<double> softmax(const std::vector<double>& logits) {
    if (logits.empty()) return {};
    double m = logits[0];
    for (double v : logits) m = std::max(m, v);
    std::vector<double> p(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) { p[i] = std::exp(logits[i] - m); sum += p[i]; }
    for (double& v : p) v /= sum;
    return p;
}

double kl_div(const std::vector<double>& p, const std::vector<double>& q, double eps = 1e-12) {
    double d = 0.0;
    for (size_t i = 0; i < p.size(); ++i)
        if (p[i] > eps) d += p[i] * std::log(p[i] / std::max(q[i], eps));
    return d;
}

std::vector<int> topk_indices(const std::vector<double>& v, int k) {
    std::vector<int> idx(v.size());
    for (size_t i = 0; i < v.size(); ++i) idx[i] = static_cast<int>(i);
    if (k > static_cast<int>(idx.size())) k = static_cast<int>(idx.size());
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(k);
    return idx;
}

}  // namespace

TEST(DecodeBtsPlacement, MatchesOracle) {
    const int T       = std::stoi(env_or("MULTI_T", "8"));
    const int steps_T = std::stoi(env_or("STEPS_T", "16"));
    ASSERT_GE(T, 1) << "MULTI_T must be at least 1";

    const char* placements_dir = std::getenv("FHE_BOOTSTRAP_PLACEMENTS_DIR");
    const bool  planned = (placements_dir != nullptr && *placements_dir);
    std::cout << "[decode_bts] mode=" << (planned ? "PLANNED" : "EAGER")
              << (planned ? std::string(" placements=") + placements_dir : std::string())
              << " T=" << T << " STEPS_T=" << steps_T << std::endl;

    const std::string config_path = default_configs_path();
    { std::ifstream c(config_path); if (!c) GTEST_SKIP() << "configs not available: " << config_path; }
    std::cout << "[decode_bts] configs=" << config_path << std::endl;
    auto parsed_configs = config_loader::parse_configs_json(
        config_loader::read_file_to_string(config_path));

    const std::string weights_path = default_weights_path();
    { std::ifstream w(weights_path); if (!w) GTEST_SKIP() << "weights not available: " << weights_path; }
    auto store = load_store(weights_path);

    const int N_blocks = parsed_configs.model.n_layers;

    // Oracle input embeddings = block-0 inputs for the T-token sequence.
    const std::string io_dir   = default_all_blocks_io_dir();
    const std::string io0_path = all_blocks_io_path(io_dir, /*L=*/0, T);
    if (!probe_io_file(io0_path))
        GTEST_SKIP() << "missing block-0 io ground truth: " << io0_path;
    auto io0 = read_io_arrays(io0_path);
    ASSERT_GE(static_cast<int>(io0.inp.size()), T);
    std::vector<std::vector<double>> oracle(io0.inp.begin(), io0.inp.begin() + T);

    // ---- build the decode facade (cachemir packing) ----
    const std::string m = env_or("GPT2_INFERENCE_MODE", "cached");
    const InferenceMode mode = parse_inference_mode(m);
    const bool cache_weights = env_or("GPT2_CACHE", (m == "cached") ? "1" : "0") != "0";

    // Named local: GPT2Model holds plans by const-ref (like parsed_configs), so it
    // must outlive the model.
    auto block_plans = default_block_plans(N_blocks);
    GPT2Model model = GPT2Model::load(
        store, parsed_configs, block_plans, mode, cache_weights,
        default_ckks_options());

    const int vocab = model.vocab();
    ASSERT_EQ(N_blocks, model.n_blocks());


    // Oracle next-token logits per step.
    const std::string steps_path = all_blocks_lm_head_steps_path(io_dir, steps_T);
    LmHeadSteps steps;
    bool have_steps = probe_io_file(steps_path);
    if (have_steps) {
        steps = read_lm_head_steps(steps_path);
        if (static_cast<int>(steps.steps.size()) < T) {
            std::cout << "[decode_bts] steps oracle covers " << steps.steps.size()
                      << " < T=" << T << "; running (no logit asserts)\n";
            have_steps = false;
        } else {
            ASSERT_EQ(steps.vocab, vocab);
        }
    } else {
        std::cout << "[decode_bts] no lm_head ground truth at " << steps_path
                  << "; running (no logit asserts)\n";
    }

    Sequence seq = model.start();
    EXPECT_EQ(seq.abs_pos, 0);

    // Planned mode: pre-encode every per-block selector plaintext (mask) for this
    // T-token decode "according to plan", so the timed loop never re-encodes a mask
    // online (strict, no online fallback). No-op in eager mode (masks stay lazy).
    model.generate_decode_masks(T);

    int top1_matches     = 0;
    int min_top5_overlap = 5;
    int completed        = 0;
    double total_decode_s = 0.0;   // per-token wall (advance + logits); always reported
    double tok0_decode_s  = 0.0;   // tok0 carries the one-time cold KV-arena alloc; excluded from per-block avg
    // Encrypted CutMax argmax is measured per token (validation; next token stays GT-forced),
    // but SKIPPED under graph capture (FHE_GRAPH_DIR) so decode captures stay byte-identical.
    const int  W_tile        = model.inference().slots;
    const bool measure_cutmax = std::getenv("FHE_GRAPH_DIR") == nullptr;

    for (int t = 0; t < T; ++t) {
        std::vector<double> logits;
        std::vector<PackedCtx> tiles;   // hoisted: reused for the CutMax measurement below
        const auto t0 = std::chrono::steady_clock::now();
        try {
            PackedCtx h = model.advance(seq, { oracle[t] });
            tiles  = model.logit_tiles(h);
            logits = decode_lm_head_logits(model.inference(), tiles, vocab, W_tile);
        } catch (const std::exception& e) {
            std::cout << "[decode_bts] tok" << t << " THREW: " << e.what()
                      << "  (completed " << completed << "/" << T << " tokens before throw)"
                      << std::endl;
            ADD_FAILURE() << "decode threw at token " << t << ": " << e.what();
            break;
        }
        const double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();   // pure decode wall (advance + logits)
        total_decode_s += dt;
        if (t == 0) tok0_decode_s = dt;
        ++completed;
        std::cout << std::fixed << std::setprecision(1)
                  << "[decode_bts] tok" << t << " perf=" << dt << " s/tok" << std::endl;

        // Encrypted CutMax argmax over the SAME FHE logit tiles — validates cutmax==fhe_argmax
        // across the whole GT trajectory. Teacher-forcing is preserved: the next iteration
        // advances on oracle[t+1], not on this argmax (no feedback; cutmax_step position=-1).
        if (measure_cutmax) {
            Inference& minf = model.inference();
            std::vector<PackedCtx> z; double argmax_s = 0.0;
            model.cutmax_step(tiles, /*position=*/-1, &z, &argmax_s);
            std::vector<double> zdec(vocab, 0.0);
            if (z.size() == 1 && vocab > W_tile) {
                auto pt = decrypt_pt(minf.cc(), z[0].ct, minf.fhe->sk());
                auto cv = pt->GetCKKSPackedValue();
                for (int m = 0; m < W_tile; ++m) {
                    const int col = cutmax_tile_col_of_slot(m, minf.size.hidDim, W_tile);
                    zdec[col] = cv[m].real();
                    if (W_tile + col < vocab) zdec[W_tile + col] = cv[m].imag();
                }
            } else {
                zdec = decode_lm_head_logits(minf, z, vocab, W_tile);
            }
            const int cm_am  = topk_indices(zdec, 1)[0];
            const int fhe_am = topk_indices(logits, 1)[0];
            std::cout << "[decode] pos=" << t << " cutmax=" << cm_am
                      << " fhe_argmax=" << fhe_am << (cm_am == fhe_am ? " OK" : " MISS")
                      << " z_mass=" << std::setprecision(4) << zdec[cm_am]
                      << " argmax=" << std::setprecision(1) << argmax_s << "s" << std::endl;
        }
        EXPECT_EQ(seq.abs_pos, t + 1) << "abs_pos did not advance at token " << t;
        ASSERT_EQ(static_cast<int>(logits.size()), vocab);

        if (!have_steps) {
            std::cout << "[decode_bts] tok" << t << " top1_fhe="
                      << topk_indices(logits, 1)[0] << " (no ground truth)" << std::endl;
            continue;
        }

        const auto& truth = steps.steps[t];
        ASSERT_EQ(static_cast<int>(truth.logits.size()), vocab);

        const int  top1_fhe = topk_indices(logits, 1)[0];
        const auto top5_fhe = topk_indices(logits, 5);
        const bool top5_hit =
            std::find(top5_fhe.begin(), top5_fhe.end(), truth.argmax) != top5_fhe.end();
        int top5_overlap = 0;
        for (int idx : topk_indices(truth.logits, 5))
            if (std::find(top5_fhe.begin(), top5_fhe.end(), idx) != top5_fhe.end())
                ++top5_overlap;
        const double kl_rf = kl_div(softmax(truth.logits), softmax(logits));

        std::cout << std::scientific << std::setprecision(4)
                  << "[decode_bts] tok" << t << " top1_fhe=" << top1_fhe
                  << " argmax_ref=" << truth.argmax
                  << " top5_overlap=" << top5_overlap << "/5"
                  << " top5_hit=" << top5_hit
                  << " KL(ref||fhe)=" << kl_rf << std::endl;

        if (top1_fhe == truth.argmax) ++top1_matches;
        min_top5_overlap = std::min(min_top5_overlap, top5_overlap);
    }

    const double avg_s = completed > 0 ? total_decode_s / completed : 0.0;
    std::cout << std::fixed << std::setprecision(1)
              << "[decode_bts] SUMMARY mode=" << (planned ? "PLANNED" : "EAGER")
              << " completed=" << completed << "/" << T
              << " top1_matches=" << top1_matches
              << " bootstraps=" << model.inference().fhe->total_bootstraps
              << " perf=" << avg_s << " s/tok"
              << " weight_relevels=" << model.inference().fhe->weight_relevel_count
              << " unplanned_bts=" << model.inference().fhe->unplanned_bootstrap_count
              << " anon_cts=" << model.inference().fhe->unnamed_ct_count
              << std::endl;
    // Two single end-of-run lines, collected sync-free (the per-token wall already rode the lm_head
    // decrypt sync): argmax agreement, and avg per-block wall. Per-token [blkperf] block timing is
    // opt-in under FHE_PROFILE (its bracketing device syncs are the only per-token interference).
    std::cout << "[decode_bts] argmax_agreement=" << top1_matches << "/" << completed << std::endl;
    // avg per-block over STEADY tokens (exclude tok0: it carries the one-time cold KV-arena alloc).
    const double steady_s = (completed > 1) ? (total_decode_s - tok0_decode_s) / (completed - 1) : avg_s;
    std::cout << std::fixed << std::setprecision(3)
              << "[decode_bts] avg_per_block=" << (N_blocks > 0 ? steady_s / N_blocks : steady_s) << " s  ("
              << std::setprecision(1) << steady_s << " s/tok steady / " << N_blocks << " blocks)" << std::endl;
    model.inference().fhe->dump_weight_relevel_report(std::cout);

    EXPECT_EQ(completed, T) << "decode did not complete all tokens (throw/timeout upstream)";

    if (have_steps && completed == T) {
        // Same soft oracle bar as the stateful decode identity test: reproduces
        // the model on the majority of tokens and never degenerates (each token
        // keeps >=2 of the model's top-5). Eager and planned must both clear it.
        EXPECT_GE(top1_matches, (T + 1) / 2)
            << "top1 matched the model on only " << top1_matches << "/" << T << " tokens";
        EXPECT_GE(min_top5_overlap, 2)
            << "a token's top-5 overlapped the model's by < 2";
    }

    model.inference().fhe->profile.dump(std::cout);
}
