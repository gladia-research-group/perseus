// TRUE AUTOREGRESSIVE encrypted generation (Phase 3 of
// docs/fhe_argmax_cutmax.md): no teacher forcing after the prompt. Each
// step runs lm_head tiles -> CutMax argmax -> Z*wte + wpe feedback ->
// next advance, all under encryption; per-step decrypts are validation
// only. Submit via scripts/17_generate_cutmax.sh (eager, cachemir,
// BTS_ITERATIONS=2, gatem15 config).
//
// Env: GEN_PROMPT (4), GEN_TOKENS (4), STEPS_T (128), CONFIGS_PATH,
//      WEIGHTS_PATH, ALL_BLOCKS_IO_DIR.

#include "app/pipeline.h"

#include <gtest/gtest.h>

#include <cstdio>

TEST(GenerateCutmax, ShortAutoregressive) {
    app::RunConfig cfg = app::RunConfig::from_env();
    cfg.tokens = cfg.gen_prompt;   // read_teacher_forced_inputs row count
    std::vector<std::vector<double>> inputs;
    try {
        inputs = app::read_teacher_forced_inputs(cfg);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "no teacher-forced inputs: " << e.what();
    }

    app::RunResult r = app::run_generate(cfg, inputs);
    ASSERT_FALSE(r.threw) << r.error;
    EXPECT_EQ(r.completed, cfg.gen_tokens);

    std::printf("[generate] completed=%d/%d e2e=%.1fs/tok argmax=%.1fs/tok "
                "bts=%ld\n",
                r.completed, r.requested, r.avg_s_per_tok, r.avg_argmax_s,
                r.bootstraps);
    std::printf("[generate] tokens:");
    for (int t : r.top1) std::printf(" %d", t);
    std::printf("\n");
    std::fflush(stdout);

    // CutMax must extract the argmax of the model's own logits: compare
    // r.top1 vs the plaintext argmax of the decrypted FHE logits per step.
    int match = 0;
    for (int j = 0; j < r.completed; ++j) {
        int am = 0;
        for (size_t i = 1; i < r.logits[j].size(); ++i)
            if (r.logits[j][i] > r.logits[j][am]) am = static_cast<int>(i);
        match += (am == r.top1[j]);
    }
    std::printf("[generate] cutmax==fhe_argmax on %d/%d steps\n", match,
                r.completed);
    EXPECT_GE(match, r.completed - 1);   // near-ties may flip
}
