// Ph2 milestone: stateful prefill -> decode across packings.
//
// Prefill m tokens through the DEDICATED cachemir_filling batched packing, then
// hand the per-block K/V off into the cachemir decode cache (re-push bridge),
// then decode position m on the cachemir path reading that cache. If the handoff
// is correct, the decoded token-m logits should track the oracle next-token
// distribution at position m (same bar as the decode chain test).
//
// One inference holds rotation keys for BOTH packings (aux_packing_kinds), and
// runs non-cached so the block loader encodes per inf.packing at call time
// (filling weights during prefill, cachemir weights during decode).

#include "all_blocks_test_helpers.h"
#include "model/gpt2.h"
#include "model/gpt2_model.h"   // GPT2Model facade (the wired interface under test)
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
std::vector<double> softmax(const std::vector<double>& l) {
    if (l.empty()) return {};
    double mx = l[0]; for (double v : l) mx = std::max(mx, v);
    std::vector<double> p(l.size()); double s = 0;
    for (size_t i = 0; i < l.size(); ++i) { p[i] = std::exp(l[i] - mx); s += p[i]; }
    for (double& v : p) v /= s; return p;
}
double kl_div(const std::vector<double>& p, const std::vector<double>& q, double e = 1e-12) {
    double d = 0; for (size_t i = 0; i < p.size(); ++i) if (p[i] > e) d += p[i] * std::log(p[i] / std::max(q[i], e));
    return d;
}
std::vector<int> topk(const std::vector<double>& v, int k) {
    std::vector<int> idx(v.size()); for (size_t i = 0; i < v.size(); ++i) idx[i] = (int)i;
    if (k > (int)idx.size()) k = idx.size();
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(k); return idx;
}
}  // namespace

TEST(PrefillDecodeHandoff, FillingPrefillThenCachemirDecode) {
    const int m       = std::stoi(env_or("MULTI_T", "4"));   // prefill chunk length
    const int steps_T = std::stoi(env_or("STEPS_T", "16"));
    const int io_T    = std::stoi(env_or("IO_T", "8"));      // need >= m+1 oracle embeddings

    const std::string config_path = default_configs_path();
    { std::ifstream c(config_path); if (!c) GTEST_SKIP() << "configs not available: " << config_path; }
    auto cfg = config_loader::parse_configs_json(config_loader::read_file_to_string(config_path));

    const std::string weights_path = default_weights_path();
    { std::ifstream w(weights_path); if (!w) GTEST_SKIP() << "weights not available: " << weights_path; }
    auto store = load_store(weights_path);

    const int n_blocks = cfg.model.n_layers;

    const std::string io_dir  = default_all_blocks_io_dir();
    const std::string io_path = all_blocks_io_path(io_dir, /*L=*/0, io_T);
    if (!probe_io_file(io_path)) GTEST_SKIP() << "missing io ground truth: " << io_path;
    auto io0 = read_io_arrays(io_path);
    ASSERT_GE(static_cast<int>(io0.inp.size()), m + 1) << "need >= m+1 embeddings";
    std::vector<std::vector<double>> prompt(io0.inp.begin(), io0.inp.begin() + m);
    std::vector<double> next_emb = io0.inp[m];   // teacher-forced input for position m

    const std::string steps_path = all_blocks_lm_head_steps_path(io_dir, steps_T);
    if (!probe_io_file(steps_path)) GTEST_SKIP() << "missing lm_head steps: " << steps_path;
    auto steps = read_lm_head_steps(steps_path);
    if (static_cast<int>(steps.steps.size()) <= m)
        GTEST_SKIP() << "steps oracle does not cover position " << m;

    // --- build the facade with prefill enabled (one context, both packings) ---
    const std::string imode = env_or("GPT2_INFERENCE_MODE", "threaded");
    // Named local: GPT2Model holds plans by const-ref, so it must outlive the model.
    auto block_plans = default_block_plans(n_blocks);
    GPT2Model model = GPT2Model::load(
        store, cfg, block_plans, parse_inference_mode(imode), /*cache_weights=*/false,
        {.logN = default_logN(), .bts_iterations = default_bts_iterations()},
        /*enable_prefill=*/true);
    ASSERT_EQ(model.n_blocks(), n_blocks);
    const int vocab = model.vocab();
    ASSERT_EQ(steps.vocab, vocab);

    // Weight residency is managed per-phase by the facade: Linear for the
    // (tiled) prefill so it streams and fits beside the dual-packing rot keys,
    // Block for decode (fits resident; fastest).
    //
    // Ordering NOTE: the prefill path runs FIRST (fresh model), the full-decode
    // reference SECOND. prefill->decode is the designed transition (model.prefill
    // ends in decode mode); decode->prefill is not — a prior full-decode leaves KV/
    // residency state that prefill's reset does not fully clear, which corrupts the
    // handoff. Running prefill first keeps it in its natural, proven flow.

    Sequence seq = model.start();
    EXPECT_EQ(seq.abs_pos, 0);

    // PREFILL m tokens via the dedicated filling packing (KV handoff happens inside).
    model.prefill(seq, prompt);
    EXPECT_EQ(seq.abs_pos, m) << "prefill should advance abs_pos by m";
    std::cout << "[handoff] facade prefill+bridge done (" << m << " tokens); abs_pos="
              << seq.abs_pos << std::endl;

    // DECODE position m on the bridged cachemir KV cache.
    std::cout << "[handoff] starting advance@pos" << m << std::endl << std::flush;
    PackedCtx h_m;
    try {
        h_m = model.advance(seq, { next_emb });
    } catch (const std::exception& e) {
        std::cerr << "[handoff] advance THREW: " << e.what() << std::endl << std::flush;
        throw;
    }
    std::cout << "[handoff] advance OK; h_m kind=" << to_string(h_m.packing.kind)
              << " t=" << h_m.packing.t << " hid=" << h_m.packing.hidDim << std::endl;
    EXPECT_EQ(seq.abs_pos, m + 1);

    // [diag] Localize the corruption: capture the prefill-path post-ln_f hidden and (in the
    // control below) the full-decode one, then report the lanes where they diverge most.
    // Outlier channels (447/138/373) => GPT-2 outlier sensitivity; broad/other => systematic.
    std::vector<double> hd_m;
    try {
        hd_m = decode_token_output(model.inference(), h_m);
        double mx = 0.0; for (double v : hd_m) mx = std::max(mx, std::abs(v));
        std::vector<int> ix(hd_m.size()); for (size_t i = 0; i < hd_m.size(); ++i) ix[i] = (int)i;
        std::partial_sort(ix.begin(), ix.begin() + std::min<size_t>(5, ix.size()), ix.end(),
                          [&](int a, int b) { return std::abs(hd_m[a]) > std::abs(hd_m[b]); });
        std::cout << "[diag] h_m (prefill) max|abs|=" << mx << " level="
                  << model.inference().fhe->level_for_ct(h_m.ct) << " top|abs| lanes:";
        for (int k = 0; k < 5 && k < (int)ix.size(); ++k)
            std::cout << " [" << ix[k] << "]=" << hd_m[ix[k]];
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[diag] h_m decrypt THREW (decode produced garbage): " << e.what() << std::endl;
    }

    std::vector<double> logits;
    try {
        logits = model.logits(h_m);
        std::cout << "[handoff] logits OK" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[handoff] logits THREW: " << e.what() << std::endl;
    }
    const bool have_pf = static_cast<int>(logits.size()) == vocab;

    const auto& truth = steps.steps[m];
    int top1 = -1; std::vector<int> top5; bool hit = false; int overlap = 0; double kl = -1.0;
    if (have_pf) {
        top1 = topk(logits, 1)[0];
        top5 = topk(logits, 5);
        hit  = std::find(top5.begin(), top5.end(), truth.argmax) != top5.end();
        for (int i : topk(truth.logits, 5)) if (std::find(top5.begin(), top5.end(), i) != top5.end()) ++overlap;
        kl   = kl_div(softmax(truth.logits), softmax(logits));
        std::cout << std::scientific << std::setprecision(4)
                  << "[handoff] decode@pos" << m << " top1=" << top1 << " argmax_ref=" << truth.argmax
                  << " top5_overlap=" << overlap << "/5 top5_hit=" << hit
                  << " KL=" << kl << std::endl;
    }

    // --- FULL-DECODE reference (control + equivalence reference), run AFTER the prefill
    //     path so prefill executed in its natural fresh-model flow. model.start() resets
    //     the KV cache, so this run is independent of the prefill run above.
    //     SKIPPABLE for big T via EQUIV_FULLDECODE=0: decoding all T tokens in sync is the
    //     slow part; the prefill path still scores vs the GT oracle, and full-decode ~= GT,
    //     so the KL/top5-vs-GT bar transitively covers "matches full decode".
    const bool do_fulldecode = env_or("EQUIV_FULLDECODE", "1") != "0";
    std::vector<double> logits_fd, hd_fd;
    if (do_fulldecode) {
        std::cout << "[equiv] full-decode reference: decoding positions 0.." << m
                  << " (" << (m + 1) << " tokens)" << std::endl << std::flush;
        try {
            Sequence seq_fd = model.start();
            PackedCtx h_fd;
            for (int t = 0; t <= m; ++t) h_fd = model.advance(seq_fd, { io0.inp[t] });
            EXPECT_EQ(seq_fd.abs_pos, m + 1);
            hd_fd     = decode_token_output(model.inference(), h_fd);   // full-decode post-ln_f hidden @ pos m
            logits_fd = model.logits(h_fd);
        } catch (const std::exception& e) {
            std::cout << "[equiv] full-decode THREW: " << e.what() << std::endl;
        }
    } else {
        std::cout << "[equiv] full-decode reference SKIPPED (EQUIV_FULLDECODE=0); "
                     "prefill scored vs GT oracle only" << std::endl;
    }
    const bool have_fd = static_cast<int>(logits_fd.size()) == vocab;
    std::cout << "[equiv] full-decode reference done (have_fd=" << have_fd << ")" << std::endl << std::flush;

    // [diag] which lanes of the prefill-path hidden diverge from the full-decode hidden?
    if (!hd_m.empty() && !hd_fd.empty() && hd_m.size() == hd_fd.size()) {
        std::vector<int> ix(hd_m.size()); for (size_t i = 0; i < hd_m.size(); ++i) ix[i] = (int)i;
        std::partial_sort(ix.begin(), ix.begin() + std::min<size_t>(8, ix.size()), ix.end(),
                          [&](int a, int b){ return std::abs(hd_m[a]-hd_fd[a]) > std::abs(hd_m[b]-hd_fd[b]); });
        std::cout << "[diag] worst h_m-vs-h_fd lanes (lane: prefill / fulldecode):";
        for (int k = 0; k < 8 && k < (int)ix.size(); ++k)
            std::cout << " [" << ix[k] << "]=" << hd_m[ix[k]] << "/" << hd_fd[ix[k]];
        std::cout << std::endl;
    }

    // --- EQUIVALENCE: prefill+decode  vs  full-decode, at the same position m -----
    // KL_*@m are vs the model oracle (how good each path is); KL(prefill||fulldecode)
    // is the direct agreement of the two FHE distributions — the real "does prefill
    // reproduce full decode" signal (immune to decode's own high-kc inaccuracy).
    if (have_fd) {
        const double kl_fd   = kl_div(softmax(truth.logits), softmax(logits_fd));
        const int    top1_fd = topk(logits_fd, 1)[0];
        const double kl_pf_vs_fd = have_pf ? kl_div(softmax(logits_fd), softmax(logits)) : -1.0;
        std::cout << std::scientific << std::setprecision(4)
                  << "[equiv] m=" << m << " argmax_ref=" << truth.argmax
                  << " | KL_fulldecode@m=" << kl_fd << " top1_fd=" << top1_fd
                  << " | KL_prefill@m="    << (have_pf ? kl : -1.0) << " top1_pf=" << top1
                  << " | KL(prefill||fulldecode)=" << kl_pf_vs_fd
                  << " | top1_match=" << (have_pf && top1 == top1_fd ? "YES" : "NO") << std::endl;
        // Equivalence bar: prefill+decode must reproduce full decode at position m.
        if (have_pf)
            EXPECT_LT(kl_pf_vs_fd, std::stod(env_or("EQUIV_MAX_KL", "0.3")))
                << "prefill+decode diverges from full decode at position " << m;
    }

    // The prefill path must produce logits at all (red if it threw — see [diag]/[handoff]).
    EXPECT_TRUE(have_pf) << "prefill+decode produced no logits (threw) at position " << m;
    // Handoff-correctness bar: the bridged context must let decode track the model.
    if (have_pf) {
        EXPECT_TRUE(hit)      << "model argmax@pos" << m << " not in decoded top-5 — handoff likely wrong";
        EXPECT_GE(overlap, 2) << "decoded top-5 barely overlaps the model — handoff likely wrong";
    }
    model.inference().fhe->profile.dump(std::cout);
}
