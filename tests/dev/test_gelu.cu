#include "all_blocks_test_helpers.h"
#include "ckks_primitives.h"
#include "model/gpt2.h"
#include "nonlinear.h"
#include "test_helpers.h"
#include "weight_loader.h"
#include "math/matrix_ops.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

void run_gelu_for_T(Inference& inf, const std::string& io_dir, int T_val) {
    SCOPED_TRACE("T=" + std::to_string(T_val));
    std::cout << "\n##### Sweep T=" << T_val << " #####" << std::endl;

    const int d_pad  = inf.size.hidDim;
    const int e_pad  = inf.size.expDim;
    const int e_real = inf.size.getRealFfDim();

    const std::string io_path = all_blocks_io_path(io_dir, /*block_idx=*/0, T_val);
    if (!probe_io_file(io_path)) return;

    // Elementwise GeLU: pre_gelu -> post_gelu (expanded MLP dim).
    auto io = read_block0_io(io_path, "pre_gelu", "post_gelu");
    const auto& inp = io.inp;
    const auto& res = io.res;

    ASSERT_FALSE(inp.empty()) << "empty inp";
    ASSERT_EQ((int)inp.size(),    T_val);
    ASSERT_EQ((int)inp[0].size(), e_real);
    ASSERT_EQ((int)res.size(),    T_val);
    ASSERT_EQ((int)res[0].size(), e_real);

    const int q = T_val - 1;
    std::cout << "  [step T=" << T_val << "] encode pre-GeLU x_q (token "
              << q << ")" << std::endl;

    auto xq_pad   = matrix::pad_vector(inp[q], e_pad);
    // pre_gelu lives on the e_pad (post-c_fc) side. The matching encoder is
    // the c_proj *input* layout (d_in=e_pad, d_out=d_pad) — same `m*p.tp` slot
    // pattern as a real c_fc output, so gelu_approx and decode see the same
    // layout they would in the real flow.
    PackedCtx x_q = encode_linear_input(inf, xq_pad, e_pad, d_pad);

    // x_q is fresh, so it enters GeLU a few levels above where a real flow would put it.

    std::cout << "  [step T=" << T_val << "] gelu_approx begin" << std::endl;
    PackedCtx gelu = gelu_approx(inf, x_q, "mlp.act");
    std::cout << "  [step T=" << T_val << "] gelu_approx end" << std::endl;

    auto raw = decrypt_slots(inf, gelu);
    auto y_pad = decode_linear_output(inf.packing, raw, inf.slots, d_pad, e_pad);
    std::vector<double> y(y_pad.begin(), y_pad.begin() + e_real);

    std::vector<int> active(e_real);
    std::iota(active.begin(), active.end(), 0);

    auto s = compare_vec(y, res[q]);
    const std::string tag = "T=" + std::to_string(T_val) + " q=" + std::to_string(q);
    report_acc(tag, s);
    report_filtered_sweep(tag, y, res[q]);
    report_acc_split(tag, y, res[q], active);
    report_filtered_sweep_split(tag, y, res[q], active);

    // Per-slot dump for offline plotting — only when the env var is set.
    const std::string dump_dir = env_or("GELU_DUMP_DIR", "");
    if (!dump_dir.empty()) {
        const std::string csv_path =
            dump_dir + "/gelu_T" + std::to_string(T_val) + ".csv";
        std::ofstream csv(csv_path);
        if (!csv) {
            std::cerr << "[gelu] WARN: cannot open " << csv_path << "\n";
        } else {
            csv << "x,ref,got,abs_err,rel_err\n";
            csv << std::setprecision(9);
            for (int i = 0; i < e_real; ++i) {
                const double x  = inp[q][i];
                const double r  = res[q][i];
                const double g  = y[i];
                const double ae = std::abs(g - r);
                const double re = ae / std::max(std::abs(r), 1e-6);
                csv << x << "," << r << "," << g << "," << ae << "," << re << "\n";
            }
            std::cout << "[gelu] wrote " << e_real << " rows -> " << csv_path << "\n";
        }
    }

    EXPECT_LT(s.mean_rel, 0.01) << "mean relative error too high";
}

} // namespace

TEST(GeluApproxTest, ElementwiseGeLU_AgainstTorch) {
    const int logN = 16;

    std::cout << "[test_gelu] creating CKKS context (logN=" << logN
              << ")..." << std::endl;
    Inference inf = make_gpt2_inference({
        .ckks          = {.bts_iterations = default_bts_iterations()},
    });
    std::cout << "[test_gelu] context ready" << std::endl;

    const std::string weights_path = default_weights_path();
    const std::string io_dir       = default_all_blocks_io_dir();

    {
        std::ifstream probe(weights_path);
        if (!probe) GTEST_SKIP() << "weights not available: " << weights_path;
    }

    const std::string configs_path = default_configs_path();
    std::cout << "[test_gelu] loading calibrated configs from "
              << configs_path << std::endl;
    auto parsed_configs = config_loader::parse_configs_json(
        config_loader::read_file_to_string(configs_path));
    weight_loader::prepare_gpt2_layer_configs(inf, parsed_configs, /*block_idx=*/0);
    std::cout << "[test_gelu] configs installed for block 0 (mlp.act)"
              << std::endl;

    const int T_val = default_t_sweep_val();
    run_gelu_for_T(inf, io_dir, T_val);
}
