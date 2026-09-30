#include "all_blocks_test_helpers.h"
#include "ckks_primitives.h"
#include "model/gpt2.h"
#include "layernorm_test_helpers.h"
#include "nonlinear.h"
#include "test_helpers.h"
#include "weight_loader.h"
#include "math/matrix_ops.h"

#include <gtest/gtest.h>

#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

constexpr int LN_C_REAL                = 768;    // GPT-2 hidden_size
constexpr int LN_HID_PAD               = 1024;   // padded hidDim (next pow2)
constexpr int N_TOKENS_PER_T           = 16;     // sample stride per JSON file

struct TokenResult {
    AccStats stats;
    std::vector<double> got;
};

TokenResult run_token(Inference& inf,
                      const std::vector<double>& ln_weight,
                      const std::vector<double>& ln_bias,
                      const std::vector<double>& x_real,
                      const std::vector<double>& y_real) {
    auto x_pad = matrix::pad_vector(x_real, LN_HID_PAD);

    // Packing-agnostic I/O: encode/decode exactly as the model flow does
    // (encode_token_input → ln_1), so the same test exercises both packings.
    PackedCtx x = encode_linear_input(inf, x_pad, LN_HID_PAD, LN_HID_PAD);
    // x is fresh, so it sits above any level a real-flow input would have entering norm.
    PackedCtx y = norm(inf, x, "ln_1");
    auto raw = decrypt_slots(inf, y);
    auto normed = decode_linear_output(inf.packing, raw, inf.slots,
                                       LN_HID_PAD, LN_HID_PAD);
    normed.resize(LN_C_REAL);

    std::vector<double> got(LN_C_REAL);
    for (int k = 0; k < LN_C_REAL; ++k)
        got[k] = normed[k] * ln_weight[k] + ln_bias[k];

    return {compare_vec(got, y_real), std::move(got)};
}

} // namespace

class LayerNormTest : public ::testing::TestWithParam<PackingKind> {};

TEST_P(LayerNormTest, FirstLN_AgainstTorch) {
    const PackingKind kind = GetParam();
    const int logN = 16;

    std::cout << "[test_layernorm] packing=" << to_string(kind)
              << ", creating CKKS context (logN=" << logN << ")...\n";
    Inference inf = make_gpt2_inference({
        .ckks         = {.bts_iterations = default_bts_iterations()},
        .hidDim       = LN_HID_PAD,
        .bench_mode   = false,
        .packing_kind = kind,
    });
    std::cout << "[test_layernorm] context ready (slots=" << inf.slots << ")\n";

    const std::string weights_path = default_weights_path();
    const std::string io_dir       = default_all_blocks_io_dir();

    {
        std::ifstream probe(weights_path);
        if (!probe) GTEST_SKIP() << "weights not available: " << weights_path;
    }

    std::cout << "[test_layernorm] loading weights from " << weights_path << "\n";
    auto store = load_store(weights_path);

    auto ln_weight = store.tensor1d("transformer.h.0.ln_1.weight", LN_C_REAL);
    auto ln_bias   = store.tensor1d("transformer.h.0.ln_1.bias",   LN_C_REAL);

    const std::string configs_path = default_configs_path();
    std::cout << "[test_layernorm] loading calibrated configs from "
              << configs_path << "\n";
    auto parsed_configs = config_loader::parse_configs_json(
        config_loader::read_file_to_string(configs_path));
    weight_loader::prepare_gpt2_layer_configs(inf, parsed_configs, /*block_idx=*/0);
    std::cout << "[test_layernorm] configs installed for block 0\n";

    const std::array<int, 1> T_SWEEP = {default_t_sweep_val()};
    for (int T_val : T_SWEEP) {
        const std::string io_path = all_blocks_io_path(io_dir, /*block_idx=*/0, T_val);
        if (!probe_io_file(io_path)) continue;

        // LN_1 lives at block 0: inp -> ln_1_out.
        auto io = read_block0_io(io_path, "inp", "ln_1_out");
        const auto& inp = io.inp;
        const auto& res = io.res;
        ASSERT_FALSE(inp.empty()) << io_path;
        ASSERT_EQ(static_cast<int>(inp.size()), T_val) << io_path;
        ASSERT_EQ(static_cast<int>(inp[0].size()), LN_C_REAL) << io_path;
        ASSERT_EQ(static_cast<int>(res[0].size()), LN_C_REAL) << io_path;

        std::cout << "\n##### T=" << T_val
                  << " (" << N_TOKENS_PER_T << " tokens sampled) #####\n";
        print_token_header();

        constexpr double fail_thresh = 0.01;
        SweepSummary sum;
        const int step = std::max(1, T_val / N_TOKENS_PER_T);
        for (int tok_idx = 0; tok_idx < T_val; tok_idx += step) {
            ASSERT_EQ(static_cast<int>(inp[tok_idx].size()), LN_C_REAL);
            ASSERT_EQ(static_cast<int>(res[tok_idx].size()), LN_C_REAL);

            TokenResult r = run_token(inf, ln_weight, ln_bias,
                                      inp[tok_idx], res[tok_idx]);
            print_token_row(tok_idx, r.stats);
            sum.add(r.stats, fail_thresh);

            EXPECT_LT(r.stats.mean_rel, fail_thresh)
                << "T=" << T_val << " tok=" << tok_idx;
        }
        print_sweep_summary(T_val, sum, fail_thresh);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllPackings, LayerNormTest,
    ::testing::Values(PackingKind::Cachemir, PackingKind::CachemirFilling,
                      PackingKind::Diagonal),
    [](const ::testing::TestParamInfo<PackingKind>& info) {
        return std::string(to_string(info.param));
    });
