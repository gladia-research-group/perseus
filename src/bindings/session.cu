// perseus._core session layer: RunConfig/RunResult, run_* pipelines, DecodeSession.
#include "app/pipeline.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <string>

namespace py = pybind11;

namespace {

// __repr__ helpers: InferenceMode has no to_string; mirror the Python enum member names.
const char* mode_name(InferenceMode m) {
    switch (m) {
        case InferenceMode::Sync:     return "InferenceMode.Sync";
        case InferenceMode::Threaded: return "InferenceMode.Threaded";
        case InferenceMode::Prefetch: return "InferenceMode.Prefetch";
    }
    return "InferenceMode.?";
}

std::string fmt3(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3f", v);
    return buf;
}

std::string quoted(const std::string& s) { return "'" + s + "'"; }
const char* py_bool(bool b) { return b ? "True" : "False"; }

}  // namespace

void bind_session(py::module_& m) {
    py::class_<app::RunConfig>(m, "RunConfig")
        .def(py::init<>(), "Construct a RunConfig with the compiled-in defaults (app/pipeline.h).")
        .def_readwrite("tokens", &app::RunConfig::tokens,
                       "Tokens to run (MULTI_T); run_prefill derives prefill_tokens from it "
                       "when unset.")
        .def_readwrite("prefill_tokens", &app::RunConfig::prefill_tokens,
                       "run_prefill only: prefill row count (PREFILL_TOKENS); -1 = tokens - 1.")
        .def_readwrite("decode_tokens", &app::RunConfig::decode_tokens,
                       "run_prefill only: tail decode token count (DECODE_TOKENS); -1 = 1.")
        .def_readwrite("gen_prompt", &app::RunConfig::gen_prompt,
                       "run_generate: teacher-forced prompt rows (GEN_PROMPT).")
        .def_readwrite("gen_tokens", &app::RunConfig::gen_tokens,
                       "run_generate: tokens generated under encryption (GEN_TOKENS).")
        .def_readwrite("steps_t", &app::RunConfig::steps_t,
                       "Oracle step count T of the all_blocks IO files in io_dir (STEPS_T).")
        .def_readwrite("configs_path", &app::RunConfig::configs_path,
                       "Approximation configs.json path (CONFIGS_PATH).")
        .def_readwrite("weights_path", &app::RunConfig::weights_path,
                       "Model weight archive: a .zip file or a directory (WEIGHTS_PATH).")
        .def_readwrite("io_dir", &app::RunConfig::io_dir,
                       "ALL_BLOCKS_IO_DIR: teacher-forced input / oracle directory.")
        .def_readwrite("plan_dir", &app::RunConfig::plan_dir,
                       "Bootstrap placement plan directory (FHE_BOOTSTRAP_PLACEMENTS_DIR); "
                       "empty = eager.")
        .def_readwrite("decode_plan_dir", &app::RunConfig::decode_plan_dir,
                       "FHE_DECODE_PLACEMENTS_DIR: post-handoff decode-phase plan (run_prefill).")
        .def_readwrite("graph_dir", &app::RunConfig::graph_dir,
                       "Graph capture output directory (FHE_GRAPH_DIR).")
        .def_readwrite("mode", &app::RunConfig::mode,
                       "InferenceMode passed to GPT2Model::load "
                       "(GPT2_INFERENCE_MODE: sync|threaded|prefetch).")
        .def_readwrite("cache_weights", &app::RunConfig::cache_weights,
                       "Cache encoded weights across blocks (GPT2_CACHE).")
        .def_readwrite("teacher_forced", &app::RunConfig::teacher_forced,
                       "run_generate: advance on the GT token instead of the CutMax feedback "
                       "(TEACHER_FORCED).")
        .def_readwrite("prime_dummy_cache", &app::RunConfig::prime_dummy_cache,
                       "Not read by the C++ pipelines; from_env leaves it False.")
        .def_static("from_env", &app::RunConfig::from_env,
                    "Build a RunConfig from the environment (MULTI_T, STEPS_T, CONFIGS_PATH, "
                    "WEIGHTS_PATH, ALL_BLOCKS_IO_DIR, FHE_*_PLACEMENTS_DIR, FHE_GRAPH_DIR, "
                    "GPT2_INFERENCE_MODE, GPT2_CACHE, TEACHER_FORCED).")
        .def("__repr__", [](const app::RunConfig& c) {
            return "<RunConfig tokens=" + std::to_string(c.tokens) +
                   " prefill_tokens=" + std::to_string(c.prefill_tokens) +
                   " decode_tokens=" + std::to_string(c.decode_tokens) +
                   " gen_prompt=" + std::to_string(c.gen_prompt) +
                   " gen_tokens=" + std::to_string(c.gen_tokens) +
                   " steps_t=" + std::to_string(c.steps_t) +
                   " configs_path=" + quoted(c.configs_path) +
                   " weights_path=" + quoted(c.weights_path) +
                   " io_dir=" + quoted(c.io_dir) +
                   " plan_dir=" + quoted(c.plan_dir) +
                   " decode_plan_dir=" + quoted(c.decode_plan_dir) +
                   " graph_dir=" + quoted(c.graph_dir) +
                   " mode=" + mode_name(c.mode) +
                   " cache_weights=" + py_bool(c.cache_weights) +
                   " teacher_forced=" + py_bool(c.teacher_forced) +
                   " prime_dummy_cache=" + py_bool(c.prime_dummy_cache) + ">";
        }, "Repr listing every bound field.");

    py::class_<app::RunResult>(m, "RunResult")
        .def_readonly("logits", &app::RunResult::logits,
                      "Decrypted logits per completed token, [completed][vocab].")
        .def_readonly("top1", &app::RunResult::top1,
                      "Argmax token per completed row (run_generate: the CutMax argmax).")
        .def_readonly("positions", &app::RunResult::positions,
                      "Absolute token position of each logits row.")
        .def_readonly("completed", &app::RunResult::completed,
                      "Tokens completed (rows in logits).")
        .def_readonly("requested", &app::RunResult::requested,
                      "Tokens requested by the config.")
        .def_readonly("bootstraps", &app::RunResult::bootstraps,
                      "Total bootstraps performed during the run.")
        .def_readonly("unplanned_bts", &app::RunResult::unplanned_bts,
                      "Auto-bootstraps fired at a level ceiling the plan did not predict.")
        .def_readonly("weight_relevels", &app::RunResult::weight_relevels,
                      "Weight re-encodes at a new level (diagnostic; 0 = weight levels perfect).")
        .def_readonly("avg_s_per_tok", &app::RunResult::avg_s_per_tok,
                      "Average seconds per token; token 0 (cold start) excluded.")
        .def_readonly("avg_argmax_s", &app::RunResult::avg_argmax_s,
                      "Average CutMax argmax stage seconds per token (within e2e).")
        .def_readonly("threw", &app::RunResult::threw,
                      "True if a token threw and the run stopped there (see error).")
        .def_readonly("error", &app::RunResult::error,
                      "Error text naming the throwing token; empty unless threw.")
        .def("__repr__", [](const app::RunResult& r) {
            const size_t vocab = r.logits.empty() ? 0 : r.logits[0].size();
            return "<RunResult completed=" + std::to_string(r.completed) +
                   " requested=" + std::to_string(r.requested) +
                   " logits=[" + std::to_string(r.logits.size()) + "x" +
                   std::to_string(vocab) + "]" +
                   " top1=[" + std::to_string(r.top1.size()) + "]" +
                   " positions=[" + std::to_string(r.positions.size()) + "]" +
                   " bootstraps=" + std::to_string(r.bootstraps) +
                   " unplanned_bts=" + std::to_string(r.unplanned_bts) +
                   " weight_relevels=" + std::to_string(r.weight_relevels) +
                   " avg_s_per_tok=" + fmt3(r.avg_s_per_tok) +
                   " avg_argmax_s=" + fmt3(r.avg_argmax_s) +
                   " threw=" + py_bool(r.threw) +
                   " error=" + quoted(r.error) + ">";
        }, "Repr with the scalar fields and the sizes of logits/top1/positions.");

    py::class_<app::GtSteps>(m, "GtSteps")
        .def_readonly("T", &app::GtSteps::T,
                      "Number of oracle steps (rows in logits); 0 when no oracle file exists.")
        .def_readonly("logits", &app::GtSteps::logits,
                      "Ground-truth lm_head logits per step, [T][vocab].");

    m.def("read_teacher_forced_inputs", &app::read_teacher_forced_inputs, py::arg("config"),
          "Read max(1, config.tokens) block-0 input rows from config.io_dir/all_blocks_L00_T*.json.");
    m.def("read_lm_head_steps", &app::read_lm_head_steps, py::arg("config"),
          "Read the ground-truth lm_head logits per step from config.io_dir "
          "(all_blocks_lm_head_steps_T{steps_t}.json); T=0 when the file is missing.");

    auto checked = [](app::RunResult r, bool raise_on_error) {
        if (raise_on_error && r.threw) throw std::runtime_error(r.error);
        return r;
    };
    m.def("run_decode",
          [checked](const app::RunConfig& cfg, const std::vector<std::vector<double>>& inputs,
                    bool raise_on_error) {
              app::RunResult r;
              { py::gil_scoped_release nogil; r = app::run_decode(cfg, inputs); }
              return checked(std::move(r), raise_on_error);
          },
          py::arg("config"), py::arg("inputs"), py::arg("raise_on_error") = true,
          "Teacher-forced decode of config.tokens tokens; equals "
          "DecodeSession(config).decode(inputs). Raises on a token error unless "
          "raise_on_error=False (then see RunResult.threw / error).");
    m.def("run_prefill",
          [checked](const app::RunConfig& cfg, const std::vector<std::vector<double>>& inputs,
                    bool raise_on_error) {
              app::RunResult r;
              { py::gil_scoped_release nogil; r = app::run_prefill(cfg, inputs); }
              return checked(std::move(r), raise_on_error);
          },
          py::arg("config"), py::arg("inputs"), py::arg("raise_on_error") = true,
          "Prefill config.prefill_tokens rows, then a teacher-forced decode of "
          "config.decode_tokens tail tokens. Raises on a token error unless raise_on_error=False.");
    m.def("run_generate",
          [checked](const app::RunConfig& cfg, const std::vector<std::vector<double>>& inputs,
                    bool raise_on_error) {
              app::RunResult r;
              { py::gil_scoped_release nogil; r = app::run_generate(cfg, inputs); }
              return checked(std::move(r), raise_on_error);
          },
          py::arg("config"), py::arg("inputs"), py::arg("raise_on_error") = true,
          "Prompt with gen_prompt rows, then generate gen_tokens tokens feeding back the encrypted "
          "CutMax argmax (GT rows instead when teacher_forced). Raises on a token error unless "
          "raise_on_error=False.");

    py::class_<app::DecodeSession>(m, "DecodeSession")
        .def(py::init<const app::RunConfig&>(), py::arg("config"),
             py::call_guard<py::gil_scoped_release>(),
             "Build the model once (CKKS context, keys, encoded weights) from config; "
             "decode() reuses it.")
        .def("decode",
             [checked](app::DecodeSession& self, const std::vector<std::vector<double>>& inputs,
                       bool raise_on_error) {
                 app::RunResult r;
                 { py::gil_scoped_release nogil; r = self.decode(inputs); }
                 return checked(std::move(r), raise_on_error);
             },
             py::arg("inputs"), py::arg("raise_on_error") = true,
             "Teacher-forced decode of config.tokens tokens on a fresh KV cache; returns a "
             "RunResult, or raises on a token error unless raise_on_error=False.");
}
