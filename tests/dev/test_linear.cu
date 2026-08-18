#include "all_blocks_test_helpers.h"
#include "model/gpt2.h"
#include "test_helpers.h"
#include "math/matrix_ops.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

struct LinearCase {
    std::string label;        // "up", "down"
    std::string wname;        // tensor name in manifest (nn.Linear layout)
    std::string bname;
    int d_in_real;
    int d_out_real;
    int d_in_pad;
    int d_out_pad;
    std::string inp_key;      // all_blocks L=0 tap for the input
    std::string res_key;      // all_blocks L=0 tap for the expected output
};

double max_abs(const std::vector<double>& v) {
    double m = 0.0; for (double x : v) m = std::max(m, std::abs(x)); return m;
}

void run_case(Inference& inf,
              const weight_loader::WeightStore& store,
              const LinearCase& c,
              const std::string& io_dir,
              int T_val) {
    std::cout << "\n=== [" << c.label << "  T=" << T_val << "] "
              << c.d_in_real << "x" << c.d_out_real
              << "  (pad " << c.d_in_pad << "x" << c.d_out_pad << ") ===\n";

    auto W_linear = store.tensor2d(c.wname, c.d_out_real, c.d_in_real);
    auto W_real   = matrix::transpose(W_linear);
    auto W_pad    = matrix::pad_matrix(W_real, c.d_in_pad, c.d_out_pad);

    auto b_real = store.tensor1d(c.bname, c.d_out_real);
    auto b_pad  = matrix::pad_vector(b_real, c.d_out_pad);

    const std::string io_path = all_blocks_io_path(io_dir, /*block_idx=*/0, T_val);
    if (!probe_io_file(io_path)) return;

    auto io = read_block0_io(io_path, c.inp_key, c.res_key);
    const auto& inp = io.inp;
    const auto& res = io.res;
    ASSERT_FALSE(inp.empty()) << "empty inp in " << io_path;
    ASSERT_EQ(static_cast<int>(inp[0].size()), c.d_in_real)
        << "inp dim mismatch in " << io_path;
    ASSERT_EQ(static_cast<int>(res[0].size()), c.d_out_real)
        << "res dim mismatch in " << io_path;

    const auto& x_real = inp[0];
    auto x_pad = matrix::pad_vector(x_real, c.d_in_pad);

    // size metadata must match the shape being tested
    inf.size.hidDim = c.d_in_pad;
    inf.size.dim    = c.d_in_pad;

    constexpr int kEncodeLevel = 17;
    inf.w[c.label] = encode_weight_matrix(inf, W_pad, c.d_in_pad, c.d_out_pad, kEncodeLevel);
    inf.w.erase(c.label + "_bias");  // ensure linear() skips bias-add in stage 1

    const auto& y_pt = res[0];

    // Plaintext references: with and without bias.
    std::vector<double> y_ref_nobias(c.d_out_real, 0.0);
    for (int j = 0; j < c.d_out_real; ++j) {
        double s = 0.0;
        for (int i = 0; i < c.d_in_real; ++i) s += x_real[i] * W_real[i][j];
        y_ref_nobias[j] = s;
    }
    std::vector<double> y_ref_bias(c.d_out_real);
    for (int j = 0; j < c.d_out_real; ++j) y_ref_bias[j] = y_ref_nobias[j] + b_real[j];

    auto pt_vs_ref = compare_vec(y_pt, y_ref_bias);
    std::cout << std::scientific << std::setprecision(3)
              << "[sanity] torch_res vs (x@W+b)  "
              << "max_abs=" << pt_vs_ref.max_abs
              << "  max_rel=" << pt_vs_ref.max_rel << "\n";
    EXPECT_LT(pt_vs_ref.max_rel, 1e-3)
        << "torch ground truth disagrees with local matmul — likely a layout bug.";

    auto run_fhe_once = [&]() {
        // Encrypt input directly at kEncodeLevel so ct.level == pt.level
        // when linear() runs; no bootstrap needed (and none of its noise).
        PackedCtx x = encode_linear_input(inf, x_pad, c.d_in_pad, c.d_out_pad, kEncodeLevel);
        PackedCtx y = linear(inf, x, c.label, c.d_in_pad, c.d_out_pad);
        auto raw = decrypt_slots(inf, y);
        auto out = decode_linear_output(inf.packing, raw, inf.slots, c.d_in_pad, c.d_out_pad);
        out.resize(c.d_out_real);
        return out;
    };

    auto report = [&](const char* tag,
                      const std::vector<double>& got,
                      const std::vector<double>& ref) {
        auto s = compare_vec(got, ref);
        std::cout << std::scientific << std::setprecision(3)
                  << "[" << tag << "]  max_abs=" << s.max_abs
                  << "  max_rel=" << s.max_rel
                  << "  mean_rel=" << s.mean_rel
                  << "  w_mape=" << s.w_mape;
        if (s.worst_idx >= 0) {
            std::cout << "  worst_idx=" << s.worst_idx
                      << "  got=" << s.worst_got
                      << "  ref=" << s.worst_ref;
        }
        std::cout << "\n[range " << tag << "]  ref_max_abs=" << max_abs(ref)
                  << "  got_max_abs=" << max_abs(got) << "\n";
        report_filtered_sweep(tag, got, ref);
        return s;
    };

    auto y_fhe_nobias = run_fhe_once();
    auto s1 = report("stage1 no-bias  vs x@W", y_fhe_nobias, y_ref_nobias);

    inf.w[c.label + "_bias"] = { encode_bias_vector(inf, b_pad, c.d_in_pad, c.d_out_pad) };
    auto y_fhe_bias = run_fhe_once();
    auto s2 = report("stage2 w/ bias  vs torch", y_fhe_bias, y_pt);

    std::vector<double> bias_delta(c.d_out_real);
    for (int j = 0; j < c.d_out_real; ++j)
        bias_delta[j] = y_fhe_bias[j] - y_fhe_nobias[j];
    auto sb = report("stage2-stage1   vs b_real", bias_delta, b_real);

    std::cout << "\n=== Summary [" << c.label << "] ==="
              << std::scientific << std::setprecision(3)
              << "  stage1 max_rel=" << s1.max_rel << " mean_rel=" << s1.mean_rel
              << "  stage2 max_rel=" << s2.max_rel << " mean_rel=" << s2.mean_rel
              << "  biasD  max_rel=" << sb.max_rel << " mean_rel=" << sb.mean_rel
              << "\n";

    EXPECT_LT(s1.mean_rel, 0.01) << "stage1 (matmul-only)";
    EXPECT_LT(s2.mean_rel, 0.01) << "stage2 (matmul+bias)";

    // --- stage3: COMPLEX weight (W_re + i*W_im) through the REAL cachemir::linear (no body change).
    // Encode W + i*(0.5W); one linear must emit W·x in the Re lane and 0.5·W·x in the Im lane. This
    // proves the encode_weight_matrix_complex layout + the imag-preserving weights_at re-level + the
    // unpack, on the same linear pipeline. Cost: one complex linear = two real linears' worth of work.
    if (is_cachemir(inf.packing)) {
        std::vector<std::vector<double>> W_half = W_pad;
        for (auto& r : W_half) for (double& v : r) v *= 0.5;
        inf.w["cplx"] = encode_weight_matrix_complex(inf, W_pad, W_half,
                                                     c.d_in_pad, c.d_out_pad, kEncodeLevel);
        inf.complex_weight_names.insert("cplx");
        inf.w.erase("cplx_bias");

        PackedCtx x  = encode_linear_input(inf, x_pad, c.d_in_pad, c.d_out_pad, kEncodeLevel);
        PackedCtx y  = linear(inf, x, "cplx", c.d_in_pad, c.d_out_pad);     // W·x + i·0.5W·x
        PackedCtx cj = inf.fhe->conjugate(y);
        PackedCtx yr = inf.fhe->mult(inf.fhe->add(y, cj), 0.5);             // Re = (y+conj)/2
        Ptx nhi      = inf.encode_complex_const_at(0.0, -0.5, y);
        PackedCtx yi = inf.fhe->mult(inf.fhe->sub(y, cj), nhi);             // Im = (y-conj)·(-0.5i)
        auto out_re  = decode_linear_output(inf.packing, decrypt_slots(inf, yr), inf.slots, c.d_in_pad, c.d_out_pad);
        auto out_im  = decode_linear_output(inf.packing, decrypt_slots(inf, yi), inf.slots, c.d_in_pad, c.d_out_pad);
        out_re.resize(c.d_out_real); out_im.resize(c.d_out_real);
        std::vector<double> ref_half(c.d_out_real);
        for (int j = 0; j < c.d_out_real; ++j) ref_half[j] = 0.5 * y_ref_nobias[j];
        auto sre = report("stage3 cplx Re  vs x@W",     out_re, y_ref_nobias);
        auto sim = report("stage3 cplx Im  vs 0.5*x@W", out_im, ref_half);
        EXPECT_LT(sre.mean_rel, 0.01) << "complex Re lane (W·x) wrong";
        EXPECT_LT(sim.mean_rel, 0.01) << "complex Im lane (0.5·W·x) wrong — imag stripped at encode/relevel?";
        inf.complex_weight_names.erase("cplx");
        inf.w.erase("cplx");
    }

    // --- stage4: OUTPUT-ROW pack (S4). ONE matrix W, but pair its OWN output blocks (2k',2k'+1)
    // into complex weights so the contraction emits both in re/im (HALF the plaintext-mults);
    // apply_linear_outputpack unpacks each block then cascades. Output must EQUAL the real W·x
    // (just cheaper). r_o must be even (up/down: r_o=4). No bias key → compare to the no-bias matmul.
    if (is_cachemir(inf.packing)) {
        inf.w["opk"] = encode_weight_matrix_outputpack(inf, W_pad, c.d_in_pad, c.d_out_pad, kEncodeLevel);
        inf.complex_weight_names.insert("opk");
        inf.w.erase("opk_bias");

        PackedCtx x = encode_linear_input(inf, x_pad, c.d_in_pad, c.d_out_pad, kEncodeLevel);
        PackedCtx y = linear_outputpack(inf, x, "opk", c.d_in_pad, c.d_out_pad);
        auto out = decode_linear_output(inf.packing, decrypt_slots(inf, y), inf.slots,
                                        c.d_in_pad, c.d_out_pad);
        out.resize(c.d_out_real);
        auto so = report("stage4 outpack  vs x@W", out, y_ref_nobias);
        EXPECT_LT(so.mean_rel, 0.01) << "output-row pack wrong (block pairing / unpack)";
        inf.complex_weight_names.erase("opk");
        inf.w.erase("opk");
    }

    // --- Batched fill check (token-in-lane packings: Diagonal / CachemirFilling).
    // Pack n_tok DISTINCT tokens into one ciphertext and run a single linear;
    // every lane must independently match its own torch reference (res[tok]).
    // This is what single-token cachemir cannot do, and is the whole point of
    // cachemir_filling — so cachemir is skipped here. Bias is still loaded, so
    // linear() adds it and we compare against res (= linear + bias). ---
    if (!is_cachemir(inf.packing)) {
        const int t_out     = inf.slots / c.d_out_pad;
        const int t_in      = inf.slots / c.d_in_pad;
        const int max_n_tok = std::min(t_in, t_out);
        const int n_tok     = std::min<int>({ static_cast<int>(inp.size()),
                                              static_cast<int>(res.size()),
                                              max_n_tok });
        if (n_tok >= 2) {
            std::vector<double> x_batched;
            x_batched.reserve(static_cast<size_t>(n_tok) * c.d_in_pad);
            for (int tok = 0; tok < n_tok; ++tok) {
                auto xt = matrix::pad_vector(inp[tok], c.d_in_pad);
                x_batched.insert(x_batched.end(), xt.begin(), xt.end());
            }

            PackedCtx xb = encode_linear_input(inf, x_batched, c.d_in_pad, c.d_out_pad, kEncodeLevel);
            PackedCtx yb = linear(inf, xb, c.label, c.d_in_pad, c.d_out_pad);
            auto raw = decrypt_slots(inf, yb);

            double worst_mean = 0.0;
            for (int tok = 0; tok < n_tok; ++tok) {
                std::vector<double> y_tok(c.d_out_real);
                for (int j = 0; j < c.d_out_real; ++j)
                    y_tok[j] = raw[static_cast<size_t>(j) * t_out + tok];
                auto st = compare_vec(y_tok, res[tok]);
                std::cout << std::scientific << std::setprecision(3)
                          << "[batched " << c.label << " tok " << tok << "/" << n_tok
                          << "]  max_rel=" << st.max_rel
                          << "  mean_rel=" << st.mean_rel << "\n";
                worst_mean = std::max(worst_mean, st.mean_rel);
                EXPECT_LT(st.mean_rel, 0.01)
                    << "batched fill tok=" << tok << " (" << c.label << ", "
                    << to_string(inf.packing.kind) << ")";
            }
            std::cout << "[batched summary " << c.label << "]  n_tok=" << n_tok
                      << "  worst mean_rel=" << worst_mean << "\n";
        }
    }

    inf.w.erase(c.label);
    inf.w.erase(c.label + "_bias");
}

} // namespace

class LinearTest : public ::testing::TestWithParam<PackingKind> {};

TEST_P(LinearTest, EndToEnd_RealWeights_AgainstTorch) {
    const PackingKind kind = GetParam();
    const int logN  = 16;
    const int d     = 1024;   // GPT-2 hidden, padded to next pow2
    const int d_exp = 4096;   // GPT-2 expanded, padded to next pow2
    const int num_heads = 16;

    std::cout << "[test_linear] packing=" << to_string(kind)
              << ", creating CKKS context (logN=" << logN << ")...\n";
    CKKSContextOptions ckks_opts{};
    ckks_opts.bts_iterations = default_bts_iterations();
    // make_gpt2_inference takes the options struct directly (bypasses ckks_options_from_env), so
    // read CKKS_COMPLEX here — needed to enable the imag lane for the stage3 complex check.
    if (const char* cx = std::getenv("CKKS_COMPLEX"); cx && cx[0] == '1')
        ckks_opts.ckks_complex_payload = true;
    Inference inf = make_gpt2_inference({
        .ckks         = ckks_opts,
        .hidDim       = d,
        .expDim       = d_exp,
        .numHeads     = num_heads,
        .bench_mode   = false,
        .packing_kind = kind,
    });
    std::cout << "[test_linear] context ready\n";

    const std::string weights_path = default_weights_path();
    const std::string io_dir       = default_all_blocks_io_dir();

    {
        std::ifstream probe(weights_path);
        if (!probe) GTEST_SKIP() << "weights not available: " << weights_path;
    }

    std::cout << "[test_linear] loading weights from " << weights_path << "\n";
    auto store = load_store(weights_path);

    // 'square' (attn.c_proj) dropped — no direct input tap in all_blocks gather.
    const std::vector<LinearCase> cases = {
        {"up",
            "transformer.h.0.mlp.c_fc.weight",
            "transformer.h.0.mlp.c_fc.bias",
            768, 3072, 1024, 4096,
            "ln_2_out", "pre_gelu"},
        {"down",
            "transformer.h.0.mlp.c_proj.weight",
            "transformer.h.0.mlp.c_proj.bias",
            3072, 768, 4096, 1024,
            "post_gelu", "mlp_out"},
    };

    const std::array<int, 1> T_SWEEP = {default_t_sweep_val()};
    for (int T_val : T_SWEEP) {
        std::cout << "\n##### Sweep T=" << T_val << " #####\n";
        for (const auto& c : cases) {
            run_case(inf, store, c, io_dir, T_val);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllPackings, LinearTest,
    // CachemirFilling before Diagonal: the diagonal 4096->1024 down-proj OOMs
    // (4096 BSGS plaintexts) and kills the process, so run the new packing first.
    ::testing::Values(PackingKind::Cachemir, PackingKind::CachemirFilling,
                      PackingKind::Diagonal),
    [](const ::testing::TestParamInfo<PackingKind>& info) {
        return std::string(to_string(info.param));
    });
