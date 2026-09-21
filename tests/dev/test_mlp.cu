#include "all_blocks_test_helpers.h"
#include "ckks_primitives.h"
#include "model/gpt2.h"
#include "model/mlp.h"
#include "test_helpers.h"
#include "weight_loader.h"

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

// Single T per run, picked via T_SWEEP_VAL env var (see default_t_sweep_val()).
// Input: post-LN_2 hidden state for one token (d_real).
// Reference: mlp_block output for that token (d_real).
// No KV cache / no MHA / no LN — mlp_block consumes already-normed input.
void run_mlp_for_T(Inference& inf, const std::string& io_dir, int T_val) {
    SCOPED_TRACE("T=" + std::to_string(T_val));
    std::cout << "\n##### Sweep T=" << T_val << " #####" << std::endl;

    const int d_real = inf.size.getRealHidDim();

    const std::string io_path = all_blocks_io_path(io_dir, /*block_idx=*/0, T_val);
    if (!probe_io_file(io_path)) return;

    // MLP consumes post-LN-2 hidden state; ground truth is the mlp_block output.
    auto io = read_block0_io(io_path, "ln_2_out", "mlp_out");
    const auto& inp = io.inp;
    const auto& res = io.res;

    ASSERT_FALSE(inp.empty()) << "empty inp";
    ASSERT_EQ((int)inp.size(),    T_val);
    ASSERT_EQ((int)inp[0].size(), d_real);
    ASSERT_EQ((int)res.size(),    T_val);
    ASSERT_EQ((int)res[0].size(), d_real);

    const int q = T_val - 1;
    std::cout << "  [T=" << T_val << "] running mlp_block on token "
              << q << std::endl;

    PackedCtx x_q = encode_token_input(inf, inp[q]);

    PackedCtx out = mlp_block(inf, x_q);
    std::cout << "  [T=" << T_val << "] mlp_block done; decrypting" << std::endl;

    auto y = decode_token_output(inf, out);

    auto s = compare_vec(y, res[q]);
    const std::string tag = "T=" + std::to_string(T_val) + " q=" + std::to_string(q);
    report_acc(tag, s);
    report_filtered_sweep(tag, y, res[q]);

    EXPECT_LT(s.mean_rel, 0.01) << "mean relative error too high";
}

} // namespace

TEST(MlpTest, FirstBlock_RealWeights_AgainstTorch) {
    const int logN = 16;

    std::cout << "[test_mlp] creating CKKS context (logN=" << logN << ")..." << std::endl;
    Inference inf = make_gpt2_inference({
        .ckks          = {.bts_iterations = default_bts_iterations()},
    });
    std::cout << "[test_mlp] context ready" << std::endl;

    const std::string weights_path = default_weights_path();
    const std::string io_dir       = default_all_blocks_io_dir();

    {
        std::ifstream probe(weights_path);
        if (!probe) GTEST_SKIP() << "weights not available: " << weights_path;
    }

    std::cout << "[test_mlp] loading weights from " << weights_path << std::endl;
    auto store = load_store(weights_path);

    auto names = weight_loader::gpt2_layer_names(0);
    std::cout << "[test_mlp] preparing GPT-2 layer-0 weights" << std::endl;
    weight_loader::prepare_gpt2_layer_weights(
        inf, store, names,
        /*d_real=*/inf.size.getRealHidDim(),
        /*d_exp_real=*/inf.size.getRealFfDim(),
        /*d_pad=*/inf.size.hidDim,
        /*d_exp_pad=*/inf.size.expDim,
        /*num_heads=*/inf.size.numHeads);
    std::cout << "[test_mlp] weights installed" << std::endl;

    const std::string configs_path = default_configs_path();
    std::cout << "[test_mlp] loading calibrated configs from "
              << configs_path << std::endl;
    auto parsed_configs = config_loader::parse_configs_json(
        config_loader::read_file_to_string(configs_path));
    weight_loader::prepare_gpt2_layer_configs(inf, parsed_configs, /*block_idx=*/0);
    std::cout << "[test_mlp] configs installed for block 0" << std::endl;

    const int T_val = default_t_sweep_val();
    run_mlp_for_T(inf, io_dir, T_val);
}
