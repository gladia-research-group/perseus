#include "all_blocks_test_helpers.h"
#include "attention.h"
#include "ckks_primitives.h"
#include "model/gpt2.h"
#include "gpt2_test_helpers.h"  // warmup_kv_cache (test-only progress loop)
#include "model/mha.h"
#include "test_helpers.h"
#include "weight_loader.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

// Reference dump (all_blocks_L00_T*.json) carries `ln_1_out` (post-LN-1 hidden
// state) and `attn_out` (attention layer output). The test feeds ln_1_out as
// inputs to a fresh K/V cache (no LN again) and compares the FHE mha_block
// output to attn_out. Hence the inline warmup using `linear()` directly rather
// than `prefill_kv_token` (which folds in LN).
void warmup_kv_no_ln(Inference& inf, const PackedCtx& x, int d_pad) {
    // Warm the cache via the SAME push path the query uses (cache_kv_push), so warmup/query scales
    // match under complex_payload. The two packings then differ ONLY in the projection (fused vs
    // separate), so the comparison is apples to apples.
    if (inf.complex) {
        // complex attention: fused projection emits P=K+iV → combined-bts packed push into the
        // complex K/V caches (the SAME path the query uses), so warmup/query scales match.
        auto qv = linear_multi(inf, x, {"kv"}, d_pad, d_pad);
        cache_kv_push_packed(inf, qv[0]);
    } else {
        auto kv = linear_multi(inf, x, {"k", "v"}, d_pad, d_pad);
        cache_kv_push(inf, kv[0], kv[1]);
    }
}

void run_mha_for_T(Inference& inf, const std::string& io_dir, int T_val) {
    SCOPED_TRACE("T=" + std::to_string(T_val));
    std::cout << "\n##### Sweep T=" << T_val << " #####" << std::endl;

    const int d_real = inf.size.getRealHidDim();
    const int d_pad  = inf.size.hidDim;

    const std::string io_path = all_blocks_io_path(io_dir, /*block_idx=*/0, T_val);
    if (!probe_io_file(io_path)) return;

    auto io = read_block0_io(io_path, "ln_1_out", "attn_out");
    const auto& inp = io.inp;
    const auto& res = io.res;

    ASSERT_FALSE(inp.empty()) << "empty inp";
    ASSERT_EQ((int)inp.size(),    T_val);
    ASSERT_EQ((int)inp[0].size(), d_real);
    ASSERT_EQ((int)res.size(),    T_val);
    ASSERT_EQ((int)res[0].size(), d_real);

    // Fresh KV cache for this T.
    prepare_mha_masks(inf);
    prepare_vcache(inf);

    // Warm up the K/V cache for tokens 0 .. T_val-2.
    warmup_kv_cache(T_val - 1, T_val, [&](int i) {
        PackedCtx x = encode_token_input(inf, inp[i]);
        warmup_kv_no_ln(inf, x, d_pad);
    });

    // Final query token: full attention forward, then compare.
    const int q  = T_val - 1;
    std::cout << "  [T=" << T_val << "] running mha_block on query token "
              << q << std::endl;
    PackedCtx x_q = encode_token_input(inf, inp[q]);
    PackedCtx out = mha_block(inf, x_q);
    Ptx out_extract_pt = inf.encode_stride_mask_at(
        inf.size.getRealHidDim(), inf.slots / inf.size.hidDim, out);
    inf.fhe->inplace_mult(out, out_extract_pt);
    std::cout << "  [T=" << T_val << "] mha_block done; decrypting" << std::endl;

    auto y = decode_token_output(inf, out);

    auto s = compare_vec(y, res[q]);
    const std::string tag = "T=" + std::to_string(T_val) + " q=" + std::to_string(q);
    report_acc(tag, s);
    report_filtered_sweep(tag, y, res[q]);

    EXPECT_LT(s.mean_rel, 0.01) << "mean relative error too high";
}

} // namespace

class MhaTest : public ::testing::TestWithParam<PackingKind> {};

TEST_P(MhaTest, EndToEnd_RealWeights_AgainstTorch) {
    const PackingKind kind = GetParam();
    const int logN = 16;

    std::cout << "[test_mha] packing=" << to_string(kind)
              << ", creating CKKS context (logN=" << logN << ")..." << std::endl;
    CKKSContextOptions ckks_opts{};
    ckks_opts.bts_iterations = default_bts_iterations();
    if (const char* cx = std::getenv("CKKS_COMPLEX"); cx && cx[0] == '1')
        ckks_opts.ckks_complex_payload = true;
    Inference inf = make_gpt2_inference({
        .ckks          = ckks_opts,
        .packing_kind  = kind,
    });
    std::cout << "[test_mha] context ready" << std::endl;

    const std::string weights_path = default_weights_path();
    const std::string io_dir       = default_all_blocks_io_dir();

    {
        std::ifstream probe(weights_path);
        if (!probe) GTEST_SKIP() << "weights not available: " << weights_path;
    }

    std::cout << "[test_mha] loading weights from " << weights_path << std::endl;
    auto store = load_store(weights_path);

    auto names = weight_loader::gpt2_layer_names(0);
    std::cout << "[test_mha] preparing GPT-2 layer-0 weights "
              << "(Q/K/V/Out + c_fc; γ/β NOT folded)" << std::endl;
    weight_loader::prepare_gpt2_layer_weights(
        inf, store, names,
        /*d_real=*/inf.size.getRealHidDim(),
        /*d_exp_real=*/inf.size.getRealFfDim(),
        /*d_pad=*/inf.size.hidDim,
        /*d_exp_pad=*/inf.size.expDim,
        /*num_heads=*/inf.size.numHeads);
    std::cout << "[test_mha] weights installed" << std::endl;

    const std::string configs_path = default_configs_path();
    std::cout << "[test_mha] loading calibrated configs from "
              << configs_path << std::endl;
    auto parsed_configs = config_loader::parse_configs_json(
        config_loader::read_file_to_string(configs_path));
    weight_loader::prepare_gpt2_layer_configs(inf, parsed_configs, /*block_idx=*/0);
    std::cout << "[test_mha] configs installed for block 0" << std::endl;

    const std::array<int, 1> T_SWEEP = {default_t_sweep_val()};
    for (int T_val : T_SWEEP) {
        run_mha_for_T(inf, io_dir, T_val);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllPackings, MhaTest,
    ::testing::Values(PackingKind::Cachemir, PackingKind::CachemirComplex),
    [](const ::testing::TestParamInfo<PackingKind>& info) {
        return std::string(to_string(info.param));
    });
