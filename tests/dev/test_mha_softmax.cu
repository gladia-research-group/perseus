#include "all_blocks_test_helpers.h"
#include "attention.h"
#include "ckks_primitives.h"
#include "model/gpt2.h"
#include "mha_softmax_test_helpers.h"
#include "nonlinear.h"
#include "test_helpers.h"
#include "weight_loader.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace test_helpers;

static void run_softmax_for_T(Inference& inf, int T_choice,
                              int H, int S, int t, int tH) {
    SCOPED_TRACE("T=" + std::to_string(T_choice));
    std::cout << "\n##### Sweep T=" << T_choice << " #####\n";

    const std::string data_path =
        all_blocks_io_path(default_all_blocks_io_dir(), /*block_idx=*/0, T_choice);
    if (!probe_io_file(data_path)) return;
    // pre/post_softmax are (H, T, T); KV-cache scenario only checks the
    // last query's row of attention scores → reshape to (H, T).
    const auto inp = read_tap_3d_last_row(data_path, "pre_softmax");
    const auto res = read_tap_3d_last_row(data_path, "post_softmax");

    ASSERT_FALSE(inp.empty()) << "inp ground-truth empty";
    ASSERT_FALSE(res.empty()) << "res ground-truth empty";
    ASSERT_EQ(inp.size(), res.size())
        << "input/output head counts disagree";

    int H_file = static_cast<int>(inp.size());
    int T = static_cast<int>(inp[0].size());
    for (int h = 0; h < H_file; ++h) {
        ASSERT_EQ(static_cast<int>(inp[h].size()), T) << "ragged inp at head " << h;
        ASSERT_EQ(static_cast<int>(res[h].size()), T) << "ragged res at head " << h;
    }
    ASSERT_EQ(T, T_choice)            << "loaded T differs from T_choice";
    ASSERT_LE(T, S)                   << "T exceeds crypt layout";
    ASSERT_LE(H_file, H)              << "H_file exceeds crypt layout";

    const int nk = T;
    inf.k_count() = nk;

    std::vector<double> active_mask(S, 0.0);
    for (int h = 0; h < H; ++h) {
        for (int tok = 0; tok < nk; ++tok) {
            active_mask[tok / t * tH + h * t + tok % t] = 1.0;
        }
    }
    inf.mask["active"] = inf.cc()->MakeCKKSPackedPlaintext(active_mask);
    inf.name_graph_pt_if_absent(inf.mask["active"], "mask.active");

    std::vector<double> scores_msg = encode_head_token_layout(inp, H, t, tH, nk, S,
                                                              /*replicate_phantom=*/true);
    std::vector<double> ref_softmax = encode_head_token_layout(res, H_file, t, tH, nk, S,
                                                               /*replicate_phantom=*/false);

    Ptx spt = inf.cc()->MakeCKKSPackedPlaintext(scores_msg);
    Ctx ct  = inf.cc()->Encrypt(spt, inf.fhe->pk());
    PackedCtx scores{ct, inf.make_packing(PackingKind::Cachemir)};
    // scores is fresh, so it sits above the level real attention scores would have
    // entering softmax.

    auto active = build_active_indices(H_file, nk, t, tH);

    const SoftmaxConfig& cfg = inf.sm_cfg.at("attn");
    std::cout << "\n=== attention_softmax_thor (end-to-end) ===\n"
              << "  cfg: log2delta1=" << cfg.log2delta1
              << " log2delta2=" << cfg.log2delta2
              << " clip=[" << cfg.clip_lo << "," << cfg.clip_hi << "]"
              << " H_file=" << H_file << " T=" << T << " H_layout=" << H << "\n";

    PackedCtx out = attention_softmax_thor(inf, {scores}, "attn")[0];
    auto got = decrypt(inf.cc(), out.ct, inf.fhe->sk());

    std::cout << "\n=== Analysis ===\n";

    auto slot = compute_slot_stats(got, ref_softmax, active);
    report_slot("attn_softmax_thor vs ground-truth", slot);

    {
        double sum = 0.0, mx = 0.0;
        int n_inact = 0;
        for (int i = 0; i < S; ++i) {
            if (active_mask[i] == 0.0) {
                double a = std::abs(got[i]);
                sum += a;
                mx = std::max(mx, a);
                ++n_inact;
            }
        }
        double mean = (n_inact > 0) ? sum / n_inact : 0.0;
        std::cout << std::scientific << std::setprecision(3)
                  << "[noise] inactive-slot |got|: mean=" << mean
                  << " max=" << mx
                  << " (n_inactive=" << n_inact << ")\n";
    }

    std::cout << "\n--- Filtered rel-err sweep (|ref| > thr) ---\n";
    for (double thr : {1e-4, 1e-3, 1e-2, 5e-2}) {
        auto f = compute_filtered_stats(got, ref_softmax, active, thr);
        report_filtered("above floor", thr, f);
    }

    std::cout << "\n--- Distribution-level error ---\n";
    auto kl = compute_kl_per_head(got, ref_softmax, H_file, nk, t, tH);
    report_kl("ciphertext vs ground-truth softmax", kl);

    auto tk = compute_topk_per_head(got, ref_softmax, H_file, nk, t, tH);
    report_topk("ciphertext vs ground-truth softmax", tk);

    std::cout << "\n=== Summary [T=" << T_choice << "] ===\n"
              << "  max_rel="    << std::scientific << std::setprecision(3) << slot.max_rel
              << "  mean_rel="   << slot.mean_rel
              << "  top1_acc="   << std::fixed      << std::setprecision(4) << tk.top1_acc
              << "  kl_fwd_mean="<< std::scientific << std::setprecision(3) << kl.mean_kl_fwd
              << "\n";

    EXPECT_LT(slot.mean_rel, 0.01) << "mean relative error on active slots too high";
    EXPECT_GE(tk.top1_acc,   0.99) << "top-1 accuracy below threshold";
}

TEST(MhaSoftmaxTest, EndToEnd_AccuracyAgainstGroundTruth) {
    const int logN = 16;
    const int d = 1024;
    const int H = 16;
    const int S = 1 << (logN - 1);
    const int t = S / d;
    const int tH = t * H;

    const std::array<int, 1> T_SWEEP = {default_t_sweep_val()};

    std::cout << "Creating CKKS context (logN=" << logN << ")...\n";

    // TODO: this should be raise to 2
    Inference inf = make_gpt2_inference({
        .ckks          = {.bts_iterations = default_bts_iterations()},
    });
    prepare_mha_masks(inf);

    const std::string configs_path = default_configs_path();
    std::cout << "[test_mha_softmax] loading calibrated configs from "
              << configs_path << "\n";
    auto parsed_configs = config_loader::parse_configs_json(
        config_loader::read_file_to_string(configs_path));
    weight_loader::prepare_gpt2_layer_configs(inf, parsed_configs, /*block_idx=*/0);
    std::cout << "[test_mha_softmax] configs installed for block 0\n";

    for (int T_choice : T_SWEEP) {
        run_softmax_for_T(inf, T_choice, H, S, t, tH);
    }
}
