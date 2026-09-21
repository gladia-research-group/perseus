// perseus._core session layer: RunConfig/RunResult, run_* pipelines, DecodeSession.
#include "app/pipeline.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_session(py::module_& m) {
    py::class_<app::RunConfig>(m, "RunConfig")
        .def(py::init<>())
        .def_readwrite("tokens", &app::RunConfig::tokens)
        .def_readwrite("prefill_tokens", &app::RunConfig::prefill_tokens)
        .def_readwrite("decode_tokens", &app::RunConfig::decode_tokens)
        .def_readwrite("gen_prompt", &app::RunConfig::gen_prompt)
        .def_readwrite("gen_tokens", &app::RunConfig::gen_tokens)
        .def_readwrite("steps_t", &app::RunConfig::steps_t)
        .def_readwrite("configs_path", &app::RunConfig::configs_path)
        .def_readwrite("weights_path", &app::RunConfig::weights_path)
        .def_readwrite("io_dir", &app::RunConfig::io_dir)
        .def_readwrite("plan_dir", &app::RunConfig::plan_dir)
        .def_readwrite("decode_plan_dir", &app::RunConfig::decode_plan_dir)
        .def_readwrite("graph_dir", &app::RunConfig::graph_dir)
        .def_readwrite("mode", &app::RunConfig::mode)
        .def_readwrite("cache_weights", &app::RunConfig::cache_weights)
        .def_readwrite("teacher_forced", &app::RunConfig::teacher_forced)
        .def_readwrite("prime_dummy_cache", &app::RunConfig::prime_dummy_cache)
        .def_static("from_env", &app::RunConfig::from_env);

    py::class_<app::RunResult>(m, "RunResult")
        .def_readonly("logits", &app::RunResult::logits)
        .def_readonly("top1", &app::RunResult::top1)
        .def_readonly("positions", &app::RunResult::positions)
        .def_readonly("completed", &app::RunResult::completed)
        .def_readonly("requested", &app::RunResult::requested)
        .def_readonly("bootstraps", &app::RunResult::bootstraps)
        .def_readonly("unplanned_bts", &app::RunResult::unplanned_bts)
        .def_readonly("weight_relevels", &app::RunResult::weight_relevels)
        .def_readonly("avg_s_per_tok", &app::RunResult::avg_s_per_tok)
        .def_readonly("avg_argmax_s", &app::RunResult::avg_argmax_s)
        .def_readonly("threw", &app::RunResult::threw)
        .def_readonly("error", &app::RunResult::error);

    py::class_<app::GtSteps>(m, "GtSteps")
        .def_readonly("T", &app::GtSteps::T)
        .def_readonly("logits", &app::GtSteps::logits);

    m.def("read_teacher_forced_inputs", &app::read_teacher_forced_inputs, py::arg("config"));
    m.def("read_lm_head_steps", &app::read_lm_head_steps, py::arg("config"));

    m.def("run_decode", &app::run_decode, py::arg("config"), py::arg("inputs"),
          py::call_guard<py::gil_scoped_release>());
    m.def("run_prefill", &app::run_prefill, py::arg("config"), py::arg("inputs"),
          py::call_guard<py::gil_scoped_release>());
    m.def("run_generate", &app::run_generate, py::arg("config"), py::arg("inputs"),
          py::call_guard<py::gil_scoped_release>());

    py::class_<app::DecodeSession>(m, "DecodeSession")
        .def(py::init<const app::RunConfig&>(), py::arg("config"),
             py::call_guard<py::gil_scoped_release>())
        .def("decode", &app::DecodeSession::decode, py::arg("inputs"),
             py::call_guard<py::gil_scoped_release>());
}
