#include "all_blocks_test_helpers.h"
#include "model/gpt2.h"
#include "model/gpt2_model.h"
#include "test_helpers.h"
#include "weight_loader.h"
#include "config_loader.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
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

TEST(StatefulGPT2Model, DecodeMatchesOracle) {
    const int T       = std::stoi(env_or("MULTI_T", "4"));
    const int steps_T = std::stoi(env_or("STEPS_T", "16"));

    const std::string config_path = default_configs_path();
    { std::ifstream c(config_path); if (!c) GTEST_SKIP() << "configs not available: " << config_path; }
    auto parsed_configs = config_loader::parse_configs_json(
        config_loader::read_file_to_string(config_path));

    const std::string weights_path = default_weights_path();
    { std::ifstream w(weights_path); if (!w) GTEST_SKIP() << "weights not available: " << weights_path; }
    auto store = load_store(weights_path);

    const int N_blocks = parsed_configs.model.n_layers;

    // Oracle input embeddings = block-0 inputs (same source as the chain test).
    const std::string io_dir   = default_all_blocks_io_dir();
    const std::string io0_path = all_blocks_io_path(io_dir, /*L=*/0, T);
    if (!probe_io_file(io0_path))
        GTEST_SKIP() << "missing block-0 io ground truth: " << io0_path;
    auto io0 = read_io_arrays(io0_path);
    ASSERT_GE(static_cast<int>(io0.inp.size()), T);
    std::vector<std::vector<double>> oracle(io0.inp.begin(), io0.inp.begin() + T);

    // ---- build the facade (decode/cachemir packing) ----
    const std::string m = env_or("GPT2_INFERENCE_MODE", "cached");
    const InferenceMode mode = parse_inference_mode(m);
    const bool cache_weights = env_or("GPT2_CACHE", (m == "cached") ? "1" : "0") != "0";

    // Named local: GPT2Model holds plans by const-ref (like parsed_configs), so it
    // must outlive the model.
    auto block_plans = default_block_plans(N_blocks);
    GPT2Model model = GPT2Model::load(
        store, parsed_configs, block_plans, mode, cache_weights,
        {.logN = default_logN(), .bts_iterations = default_bts_iterations()});

    const int vocab  = model.vocab();
    ASSERT_EQ(N_blocks, model.n_blocks());

    // Oracle next-token logits per step.
    const std::string steps_path = all_blocks_lm_head_steps_path(io_dir, steps_T);
    LmHeadSteps steps;
    bool have_steps = probe_io_file(steps_path);
    if (have_steps) {
        steps = read_lm_head_steps(steps_path);
        if (static_cast<int>(steps.steps.size()) < T) {
            std::cout << "[stateful] steps oracle covers " << steps.steps.size()
                      << " < T=" << T << "; running (no logit asserts)\n";
            have_steps = false;
        } else {
            ASSERT_EQ(steps.vocab, vocab);
        }
    } else {
        std::cout << "[stateful] no lm_head ground truth at " << steps_path
                  << "; running (no logit asserts)\n";
    }

    Sequence seq = model.start();
    EXPECT_EQ(seq.abs_pos, 0);

    int top1_matches = 0;
    int min_top5_overlap = 5;

    for (int t = 0; t < T; ++t) {
        // Stateful single-token advance; KV accumulates across iterations.
        PackedCtx h = model.advance(seq, { oracle[t] });
        EXPECT_EQ(seq.abs_pos, t + 1) << "abs_pos did not advance at token " << t;

        std::vector<double> logits = model.logits(h);
        ASSERT_EQ(static_cast<int>(logits.size()), vocab);

        if (!have_steps) continue;
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
                  << "[stateful] tok" << t << " top1_fhe=" << top1_fhe
                  << " argmax_ref=" << truth.argmax
                  << " top5_overlap=" << top5_overlap << "/5"
                  << " top5_hit=" << top5_hit
                  << " KL(ref||fhe)=" << kl_rf << std::endl;

        if (top1_fhe == truth.argmax) ++top1_matches;
        min_top5_overlap = std::min(min_top5_overlap, top5_overlap);
    }

    if (have_steps) {
        // Reproduces the model on the majority of tokens, and never degenerates
        // (each token keeps >=2 of the model's top-5) -> the facade runs real
        // decode correctly, not garbage.
        EXPECT_GE(top1_matches, (T + 1) / 2)
            << "facade top1 matched the model on only " << top1_matches << "/" << T << " tokens";
        EXPECT_GE(min_top5_overlap, 2)
            << "a token's facade top-5 overlapped the model's by < 2";
    }

    model.inference().fhe->profile.dump(std::cout);
}
