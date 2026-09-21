#pragma once

#include "inference.h"

#include <memory>
#include <string>
#include <vector>

class GPT2Model;

namespace app {

struct RunConfig {
    int           tokens        = 1;
    int           prefill_tokens = -1;
    int           decode_tokens  = -1;
    int           gen_prompt    = 4;
    int           gen_tokens    = 4;
    int           steps_t       = 16;
    std::string   configs_path;
    std::string   weights_path;
    std::string   io_dir;
    std::string   plan_dir;
    std::string   decode_plan_dir;
    std::string   graph_dir;
    InferenceMode mode          = InferenceMode::Threaded;
    bool          cache_weights = true;
    bool          teacher_forced = false;

    bool          prime_dummy_cache = false;

    static RunConfig from_env();
};

struct RunResult {
    std::vector<std::vector<double>> logits;
    std::vector<int>  top1;
    std::vector<int>  positions;
    int    completed        = 0;
    int    requested        = 0;
    long   bootstraps       = 0;
    long   unplanned_bts    = 0;
    long   weight_relevels  = 0;
    double avg_s_per_tok    = 0.0;
    double avg_argmax_s     = 0.0;
    bool   threw            = false;
    std::string error;
};

std::vector<std::vector<double>> read_teacher_forced_inputs(const RunConfig& cfg);

struct GtSteps {
    int T = 0;
    std::vector<std::vector<double>> logits;   // [T][vocab]
};

GtSteps read_lm_head_steps(const RunConfig& cfg);

RunResult run_decode(const RunConfig& cfg,
                     const std::vector<std::vector<double>>& inputs);

class DecodeSession {
public:
    explicit DecodeSession(const RunConfig& cfg);
    ~DecodeSession();
    RunResult decode(const std::vector<std::vector<double>>& inputs);
private:
    RunConfig cfg_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

RunResult run_prefill(const RunConfig& cfg,
                      const std::vector<std::vector<double>>& inputs);

RunResult run_generate(const RunConfig& cfg,
                       const std::vector<std::vector<double>>& inputs);

}  // namespace app
