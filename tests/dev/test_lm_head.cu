// FHE LM head (output-tiled) end-to-end test.
//
// Encodes the ground-truth post-ln_f hidden state of the LAST token, runs the
// encrypted output-tiled head (gpt2_lm_head), decrypts + decodes + concatenates
// the K tiles into 50257 logits, and scores them against the model's own
// next-token prediction (all_blocks_lm_head_T{T}.json: logits / argmax / top-k).
//
// The decrypt here is TEST-ONLY: in fully-ciphered generation the tiles stay
// encrypted and feed the (deferred) FHE argmax. Diagonal packing only — the
// head's diagonal BSGS keys depend on d_in (=1024), not d_out, so it reuses the
// MLP up-proj rotation keys and needs no bootstrap (the head is terminal).

#include "all_blocks_test_helpers.h"
#include "model/gpt2.h"
#include "test_helpers.h"
#include "weight_loader.h"
#include "math/matrix_ops.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

std::string all_blocks_lnf_path(const std::string& io_dir, int T_val) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s/all_blocks_ln_f_T%d.json", io_dir.c_str(), T_val);
    return std::string(buf);
}

std::string all_blocks_lm_head_path(const std::string& io_dir, int T_val) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s/all_blocks_lm_head_T%d.json", io_dir.c_str(), T_val);
    return std::string(buf);
}

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

double kl(const std::vector<double>& p, const std::vector<double>& q, double eps = 1e-12) {
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

class LmHeadTest : public ::testing::TestWithParam<PackingKind> {};

TEST_P(LmHeadTest, OutputTiled_AgainstTorch) {
    const PackingKind kind = GetParam();
    const int d     = 1024;   // GPT-2 hidden, padded to next pow2
    const int d_exp = 4096;   // GPT-2 expanded, padded to next pow2
    const int num_heads = 16;
    const int T_val = default_t_sweep_val();

    std::cout << "[test_lm_head] packing=" << to_string(kind)
              << ", creating CKKS context (logN=16)...\n";
    CKKSContextOptions ckks_opts{};
    ckks_opts.bts_iterations = default_bts_iterations();
    // make_gpt2_inference takes the options struct directly (bypasses ckks_options_from_env), so
    // read CKKS_COMPLEX here — cachemir_complex needs the imag lane to carry the second tile.
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

    const int d_real = inf.size.getRealHidDim();
    const int W_tile = inf.slots;  // widest tile = fewest pt-mults (n_diag = d_in)

    const std::string weights_path = default_weights_path();
    {
        std::ifstream probe(weights_path);
        if (!probe) GTEST_SKIP() << "weights not available: " << weights_path;
    }
    auto store = load_store(weights_path);

    const int vocab = static_cast<int>(store.meta(weight_loader::gpt2_lm_head_name()).shape[0]);
    const int K = (vocab + W_tile - 1) / W_tile;
    std::cout << "[test_lm_head] vocab=" << vocab << " W_tile=" << W_tile
              << " K=" << K << " d_real=" << d_real << "\n";

    const std::string io_dir = default_all_blocks_io_dir();
    const std::string lnf_path = all_blocks_lnf_path(io_dir, T_val);
    const std::string lm_path  = all_blocks_lm_head_path(io_dir, T_val);
    if (!probe_io_file(lnf_path) || !probe_io_file(lm_path)) {
        GTEST_SKIP() << "missing ground truth for T=" << T_val;
    }

    // Post-ln_f hidden of the LAST token (= the autoregressive prediction site).
    auto lnf = read_io_arrays(lnf_path);
    ASSERT_FALSE(lnf.res.empty());
    const auto& hidden = lnf.res.back();
    ASSERT_EQ(static_cast<int>(hidden.size()), d_real);

    auto truth = read_lm_head_truth(lm_path);
    ASSERT_EQ(static_cast<int>(truth.logits.size()), vocab);

    // Encode the head input once at the bootstrap output level — the level the
    // tied weights are encoded at inside gpt2_lm_head (and where the post-ln_f
    // input arrives in the real chain), so ct.level == pt.level at the mult.
    const int enc_lvl = static_cast<int>(inf.fhe->bootstrap_output_level());
    auto x_pad = matrix::pad_vector(hidden, d);
    PackedCtx x = encode_linear_input(inf, x_pad, d, W_tile, enc_lvl);

    const int K_eff = inf.complex ? (K + 1) / 2 : K;
    std::cout << "[test_lm_head] running gpt2_lm_head (" << K_eff << " tiles, K=" << K << ")...\n";
    auto tiles = gpt2_lm_head(inf, x, store, vocab, W_tile);
    ASSERT_EQ(static_cast<int>(tiles.size()), K_eff);

    // Decrypt + decode + concatenate → vocab logits via the production splitter (handles the
    // cachemir_complex real()/imag() tile split as well as the real path). Test-only decrypt.
    std::vector<double> logits = decode_lm_head_logits(inf, tiles, vocab, W_tile);
    ASSERT_EQ(static_cast<int>(logits.size()), vocab);

    // Compare distributions and argmax against the model's own next-token logits.
    auto probs_ref = softmax(truth.logits);
    auto probs_fhe = softmax(logits);
    const double kl_rf = kl(probs_ref, probs_fhe);

    const int top1_fhe = topk_indices(logits, 1)[0];
    const auto top5_fhe = topk_indices(logits, 5);
    const bool top5_hit =
        std::find(top5_fhe.begin(), top5_fhe.end(), truth.argmax) != top5_fhe.end();
    int top5_overlap = 0;
    for (int idx : topk_indices(truth.logits, 5))
        if (std::find(top5_fhe.begin(), top5_fhe.end(), idx) != top5_fhe.end())
            ++top5_overlap;

    auto s = compare_vec(logits, truth.logits);
    std::cout << std::scientific << std::setprecision(4)
              << "[test_lm_head] T=" << T_val
              << " top1_fhe=" << top1_fhe << " argmax_ref=" << truth.argmax
              << " top5_overlap=" << top5_overlap << "/5"
              << " KL(ref||fhe)=" << kl_rf
              << " logits max_rel=" << s.max_rel << " mean_rel=" << s.mean_rel
              << "\n";

    EXPECT_EQ(top1_fhe, truth.argmax) << "FHE argmax != model argmax";
    EXPECT_TRUE(top5_hit) << "model argmax not in FHE top-5";
    EXPECT_GE(top5_overlap, 4) << "FHE top-5 overlaps model top-5 in < 4 tokens";
    EXPECT_LT(kl_rf, 0.05) << "softmax KL(ref||fhe) too large";
}

INSTANTIATE_TEST_SUITE_P(
    AllPackings, LmHeadTest,
    ::testing::Values(PackingKind::Diagonal, PackingKind::Cachemir,
                      PackingKind::CachemirComplex),
    [](const ::testing::TestParamInfo<PackingKind>& info) {
        return std::string(to_string(info.param));
    });
