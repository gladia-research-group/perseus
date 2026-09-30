#include "all_blocks_test_helpers.h"
#include "config_loader.h"
#include "model/gpt2.h"
#include "model/gpt2_model.h"
#include "test_helpers.h"
#include "math/matrix_ops.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace test_helpers;

// Minimal counterpart to test_autoregressive_chain: feeds all T oracle tokens
// through the supported prefill facade and verifies the CachemirFilling prefill
// leaves the model ready for Cachemir decode. No per-block debug taps live on
// the production Inference state.
TEST(PrefillChainCachemirFilling, AllTokensInOneCt) {
    const int T = std::stoi(env_or("MULTI_T", "8"));

    const std::string config_path = default_configs_path();
    {
        std::ifstream c(config_path);
        if (!c) GTEST_SKIP() << "configs not available: " << config_path;
    }
    auto parsed_configs = config_loader::parse_configs_json(
        config_loader::read_file_to_string(config_path));

    const std::string weights_path = default_weights_path();
    {
        std::ifstream w(weights_path);
        if (!w) GTEST_SKIP() << "weights not available: " << weights_path;
    }
    auto store = load_store(weights_path);

    // Execution mode (env-selectable). Sync loads one block at a time — no async
    // prefetch — so encoded weights don't multi-buffer on host; Threaded overlaps
    // per-block encode with compute (faster, but ~2+ blocks of weights resident).
    // Resolved BEFORE load() because mode + cache_weights are construction params:
    // mode feeds the CKKS context build, cache_weights drives the block pre-encode.
    const std::string m = env_or("GPT2_INFERENCE_MODE", "threaded");
    const InferenceMode mode = parse_inference_mode(m);   // overlap; caching is moot for single-pass prefill
    const bool cache_weights = env_or("GPT2_CACHE", (m == "cached") ? "1" : "0") != "0";

    // Single construction entry: load() builds the Inference (decode/Cachemir
    // default) AND, with enable_prefill=true, the CachemirFilling aux rot keys.
    // This test drives the FULL API: model.prefill() owns the phase config
    // (CachemirFilling / n_tok / Linear granularity with per-pt-streamed MLP), the
    // final LN, the filling->cachemir KV handoff, and the post-prefill rot-key free.
    // Named local: GPT2Model holds plans by const-ref, so it must outlive the model.
    auto block_plans = default_block_plans(parsed_configs.model.n_layers);
    GPT2Model model = GPT2Model::load(
        store, parsed_configs, block_plans, mode, cache_weights,
        {.logN = default_logN(), .bts_iterations = default_bts_iterations()},
        /*enable_prefill=*/true);
    Inference& inf = model.inference();
    inf.use_cache  = true;

    const int N_blocks = parsed_configs.model.n_layers;   // API prefill runs the full stack
    const int d_real   = inf.size.getRealHidDim();
    const int d_pad    = inf.size.hidDim;
    const int t_stride = inf.slots / d_pad;
    std::cout << "[prefill_chain] T=" << T << " N_blocks=" << N_blocks
              << " d_real=" << d_real << " d_pad=" << d_pad
              << " t_stride=" << t_stride << " packing=cachemir_filling (via API)\n";

    const std::string io_dir   = default_all_blocks_io_dir();
    const std::string io0_path = all_blocks_io_path(io_dir, /*L=*/0, T);
    if (!probe_io_file(io0_path))
        GTEST_SKIP() << "missing block-0 io ground truth: " << io0_path;
    auto io0 = read_io_arrays(io0_path);
    ASSERT_GE(static_cast<int>(io0.inp.size()), T);
    ASSERT_EQ(static_cast<int>(io0.inp[0].size()), d_real);

    // Prompt = the T oracle block-0 input token vectors (d_real each). model.prefill
    // packs them into one CachemirFilling ciphertext internally (encode_prefill_input).
    std::vector<std::vector<double>> prompt(T);
    for (int tok = 0; tok < T; ++tok)
        prompt[tok].assign(io0.inp[tok].begin(), io0.inp[tok].begin() + d_real);

    // Full-API prefill: prefill (+ final LN + KV handoff + rot-key free) all T tokens.
    Sequence seq = model.start();
    bool threw = false;
    std::string err;
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    try {
        model.prefill(seq, prompt);
    } catch (const std::exception& e) {
        threw = true;
        err = e.what();
        std::cout << "[prefill_chain] model.prefill threw: " << err << "\n";
    }
    cudaDeviceSynchronize();
    const double prefill_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cout << std::fixed << std::setprecision(1)
              << "[prefill_chain] PERF: prefill wall = " << prefill_ms / 1000.0 << " s for "
              << T << " tokens (" << N_blocks << " blocks)" << (threw ? " [THREW]" : "")
              << "; amortized per-token = " << prefill_ms / std::max(T, 1) / 1000.0 << " s/tok\n";

    EXPECT_FALSE(threw) << err;
    EXPECT_EQ(seq.abs_pos, T) << "prefill should advance the sequence by the prompt length";
    EXPECT_EQ(inf.packing.kind, PackingKind::Cachemir)
        << "prefill should leave the facade ready for decode after KV handoff";

    // Per-step breakdown for the profiling run (empty/no-op unless FHE_PROFILE=wall).
    inf.fhe->profile.dump(std::cout);
}
