#pragma once

#include "inference.h"        // InferenceMode, PackedCtx

#include <memory>
#include <string>
#include <vector>

class GPT2Model;   // fwd: full type stays in pipeline.cu (DecodeSession pimpl)

namespace app {

struct RunConfig {
    int           tokens        = 1;
    int           prefill_tokens = -1;  // prefill subcommand only; -1 means derive from --tokens
    int           decode_tokens  = -1;  // prefill subcommand only; -1 means default tail decode
    int           gen_prompt    = 4;    // run_generate: teacher-forced prompt rows (GEN_PROMPT)
    int           gen_tokens    = 4;    // run_generate: tokens generated under encryption (GEN_TOKENS)
    int           steps_t       = 16;
    std::string   configs_path;
    std::string   weights_path;
    std::string   io_dir;                 // ALL_BLOCKS_IO_DIR (teacher-forced input source)
    std::string   plan_dir;
    std::string   decode_plan_dir;   // FHE_DECODE_PLACEMENTS_DIR: post-handoff decode-phase plan
    std::string   graph_dir;
    InferenceMode mode          = InferenceMode::Threaded;   // GPT2_INFERENCE_MODE
    bool          cache_weights = true;
    // CUT_MAX removed 2026-07-05: the encrypted CutMax argmax is ALWAYS computed.
    // teacher_forced (TEACHER_FORCED / --teacher-forced): generate advances on the GT token
    // instead of feeding back its own CutMax argmax — CutMax is still computed every step
    // (measured, not fed back). false => autoregressive (encrypted feedback).
    bool          teacher_forced = false;

    bool          prime_dummy_cache = false;

    static RunConfig from_env();
};

struct RunResult {
    std::vector<std::vector<double>> logits;   // [completed][vocab]
    std::vector<int>  top1;                     // argmax per completed token
    std::vector<int>  positions;                // absolute token position for each logits row
    int    completed        = 0;
    int    requested        = 0;
    long   bootstraps       = 0;
    long   unplanned_bts    = 0;
    long   weight_relevels  = 0;
    double avg_s_per_tok    = 0.0;   // tok0 (cold start) excluded from the average
    double avg_argmax_s     = 0.0;   // run_generate: cutmax stage s/tok (within e2e)
    bool   threw            = false;
    std::string error;               // populated iff threw (throwing token in the text)
};

std::vector<std::vector<double>> read_teacher_forced_inputs(const RunConfig& cfg);

struct GtSteps {
    int T = 0;
    std::vector<std::vector<double>> logits;   // [T][vocab]
};

GtSteps read_lm_head_steps(const RunConfig& cfg);

RunResult run_decode(const RunConfig& cfg,
                     const std::vector<std::vector<double>>& inputs);

// Reusable planned-decode session: builds the model (CKKS context + rotation/bootstrap keys +
// weight encoding) ONCE in the ctor, then decode() runs a single teacher-forced sample with a
// fresh KV cache (GPT2Model::start()). run_decode(cfg,inputs) == DecodeSession(cfg).decode(inputs).
// The 128-sample sweep (decode_multi) constructs ONE session and loops samples, amortizing the
// per-process FHE setup cold start across all samples in the job.
class DecodeSession {
public:
    explicit DecodeSession(const RunConfig& cfg);
    ~DecodeSession();
    RunResult decode(const std::vector<std::vector<double>>& inputs);
private:
    RunConfig cfg_;
    // Impl OWNS the WeightStore/ParsedConfigs/BlockPlans that GPT2Model holds BY REFERENCE
    // (store_/cfg_/plans_/decode_plans_) — they must outlive the model, so it can't be built
    // from a helper's locals (they'd dangle => tail plan .at() throws unordered_map::at).
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

RunResult run_prefill(const RunConfig& cfg,
                      const std::vector<std::vector<double>>& inputs);

RunResult run_generate(const RunConfig& cfg,
                       const std::vector<std::vector<double>>& inputs);

}  // namespace app
