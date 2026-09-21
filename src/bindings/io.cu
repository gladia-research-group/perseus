#include "config_loader.h"
#include "checks.h"

#include <algorithm>
#include "npconv.h"
#include "cutmax.h"
#include "encoded_block.h"
#include "model/gpt2.h"
#include "model/block_residency.h"
#include "weight_loader.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>

#include <filesystem>

namespace py = pybind11;

namespace {
constexpr auto kRelease = py::call_guard<py::gil_scoped_release>();
}

void bind_io(py::module_& m) {
    py::class_<weight_loader::WeightStore>(m, "WeightStore")
        .def_static("from_zip", &weight_loader::WeightStore::from_zip, py::arg("zip_path"), kRelease,
                    "Load a WeightStore from a zip export (manifest.json + tensor entries).")
        .def_static("from_dir", &weight_loader::WeightStore::from_dir, py::arg("dir_path"), kRelease,
                    "Load a WeightStore from an export directory (manifest.json + tensor files).")
        .def("names", &weight_loader::WeightStore::names, "Sorted tensor names in the store.")
        .def("has", &weight_loader::WeightStore::has, py::arg("name"), "Whether `name` is in the store.")
        .def("shape",
             [](const weight_loader::WeightStore& s, const std::string& name) {
                 if (!s.has(name)) throw py::key_error(name);
                 return s.meta(name).shape;
             },
             py::arg("name"), "Shape of tensor `name` (KeyError when absent).")
        .def("tensor",
             [](const weight_loader::WeightStore& s, const std::string& name) {
                 if (!s.has(name)) throw py::key_error(name);
                 const auto& m = s.meta(name);
                 py::array_t<double> out(m.shape);
                 const auto& v = s.tensor(name);
                 std::copy(v.begin(), v.end(), out.mutable_data());
                 return out;
             },
             py::arg("name"), "Tensor `name` as a float64 array in its manifest shape (KeyError when absent).")
        .def("__contains__", &weight_loader::WeightStore::has)
        .def("__len__", [](const weight_loader::WeightStore& s) { return s.names().size(); })
        .def("__repr__", [](const weight_loader::WeightStore& s) {
            return "<WeightStore " + std::to_string(s.names().size()) + " tensors>";
        });

    py::class_<config_loader::ModelConfig>(m, "ModelConfig")
        .def_readonly("n_layers", &config_loader::ModelConfig::n_layers)
        .def_readonly("n_embd", &config_loader::ModelConfig::n_embd)
        .def_readonly("n_head", &config_loader::ModelConfig::n_head)
        .def_readonly("n_inner", &config_loader::ModelConfig::n_inner);

    py::class_<CutMaxCalib>(m, "CutMaxCalib", "Parsed 'cutmax' calibration section (opaque).");
    py::class_<CutMaxConfig>(m, "CutMaxConfig",
                             "The encrypted-argmax schedule: per-iteration amplification powers, "
                             "range-reduction passes and Newton settings.")
        .def_readonly("newton_per_pass", &CutMaxConfig::newton_per_pass)
        .def_readonly("newton_polish", &CutMaxConfig::newton_polish)
        .def_property_readonly("n_iters", [](const CutMaxConfig& c) { return c.iters.size(); })
        .def_property_readonly("iters", [](const CutMaxConfig& c) {
            py::list out;
            for (const auto& it : c.iters) {
                py::dict d;
                d["p"] = it.p; d["c"] = it.c; d["m"] = it.m; d["s2_hi"] = it.s2_hi;
                d["passes"] = it.passes; d["ex2"] = it.ex2; d["ca"] = it.ca; d["cb"] = it.cb;
                d["casc_iters"] = it.casc_iters;
                out.append(d);
            }
            return out;
        }, "One dict per CutMax iteration (p, c, m, s2_hi, passes, ex2, ca, cb, casc_iters).")
        .def("__repr__", [](const CutMaxConfig& c) {
            return "<CutMaxConfig iters=" + std::to_string(c.iters.size()) +
                   " newton_per_pass=" + std::to_string(c.newton_per_pass) +
                   " newton_polish=" + std::to_string(c.newton_polish) + ">";
        });
    m.def("default_cutmax_config", &default_gpt2_cutmax_config,
          "The oracle-locked default GPT-2 CutMax schedule.");
    m.def("cutmax_config_from_calib", &cutmax_config_from_calib, py::arg("calib"),
          "Build the runtime CutMaxConfig from a configs.json 'cutmax' calibration.");

    py::class_<config_loader::ParsedConfigs>(m, "ParsedConfigs")
        .def_readonly("model", &config_loader::ParsedConfigs::model)
        .def_readonly("norm", &config_loader::ParsedConfigs::norm)
        .def_readonly("softmax", &config_loader::ParsedConfigs::softmax)
        .def_readonly("softgelu", &config_loader::ParsedConfigs::softgelu)
        .def_readonly("cutmax", &config_loader::ParsedConfigs::cutmax)
        .def_readonly("has_cutmax", &config_loader::ParsedConfigs::has_cutmax);

    m.def("load_configs",
          [](const std::string& path) {
              std::filesystem::path p(path);
              if (p.extension() == ".json") p = p.parent_path();
              return config_loader::load_configs(p.string());
          },
          py::arg("path"), "Parse configs.json (path = the file or its directory).");

    py::class_<BootstrapPlan>(m, "BootstrapPlan")
        .def(py::init<>(), "An empty plan (valid=False).")
        .def_readonly("valid", &BootstrapPlan::valid)
        .def_property_readonly("num_placements",
                               [](const BootstrapPlan& p) { return p.placement_after.size(); })
        .def_property_readonly("placements", [](const BootstrapPlan& p) {
            std::vector<std::string> v(p.placement_after.begin(), p.placement_after.end());
            std::sort(v.begin(), v.end());
            return v;
        }, "Sorted ct-variable names a bootstrap is planted after.")
        .def_property_readonly("expected_levels",
                               [](const BootstrapPlan& p) { return p.expected_levels; },
                               "var -> level the strict runtime checks (a copy).")
        .def_property_readonly("weight_levels",
                               [](const BootstrapPlan& p) { return p.weight_levels; })
        .def_property_readonly("hint_fire", [](const BootstrapPlan& p) {
            std::vector<std::string> v(p.hint_fire.begin(), p.hint_fire.end());
            std::sort(v.begin(), v.end());
            return v;
        })
        .def("__repr__", [](const BootstrapPlan& p) {
            return std::string("<BootstrapPlan ") + (p.valid ? "valid" : "empty") +
                   " placements=" + std::to_string(p.placement_after.size()) +
                   " expected_levels=" + std::to_string(p.expected_levels.size()) + ">";
        });
    m.def("parse_bootstrap_plan_file", &parse_bootstrap_plan_file, py::arg("path"), kRelease,
          "Parse a bootstrap placement JSON file into a BootstrapPlan (valid=False when missing).");

    py::class_<EncodedBlock>(m, "EncodedBlock")
        .def(py::init<>(), "An empty block state; fill it via set_weight/set_bias/set_*_cfg.")
        .def_readwrite("prefix", &EncodedBlock::prefix)
        .def_readwrite("plan", &EncodedBlock::plan)
        .def("set_weight",
             [](EncodedBlock& blk, Inference& inf, const std::string& name,
                const perseus_np::Arr2& W, int d_in, int d_out, int level) {
                 const auto M = perseus_np::to_mat(W);
                 perseus_checks::check_matrix("EncodedBlock.set_weight", name, M, d_in, d_out);
                 py::gil_scoped_release nogil;
                 blk.w[name] = encode_weight_matrix(inf, M, d_in, d_out, level);
                 inf.token_basis_weights.insert(name);
             },
             py::arg("inf"), py::arg("name"), py::arg("W"), py::arg("d_in"), py::arg("d_out"),
             py::arg("level") = 0, "numpy fast path: a (d_in, d_out) float array.")
        .def("set_weight",
             [](EncodedBlock& blk, Inference& inf, const std::string& name,
                const std::vector<std::vector<double>>& W, int d_in, int d_out, int level) {
                 perseus_checks::check_matrix("EncodedBlock.set_weight", name, W, d_in, d_out);
                 blk.w[name] = encode_weight_matrix(inf, W, d_in, d_out, level);
                 inf.token_basis_weights.insert(name);   // slot-layout checked (slot_layout.h)
             },
             py::arg("inf"), py::arg("name"), py::arg("W"), py::arg("d_in"), py::arg("d_out"),
             py::arg("level") = 0, kRelease)
        .def("set_weight_complex",
             [](EncodedBlock& blk, Inference& inf, const std::string& name,
                const std::vector<std::vector<double>>& W_re,
                const std::vector<std::vector<double>>& W_im, int d_in, int d_out, int level) {
                 perseus_checks::check_matrix("EncodedBlock.set_weight_complex", name + ".re", W_re, d_in, d_out);
                 perseus_checks::check_matrix("EncodedBlock.set_weight_complex", name + ".im", W_im, d_in, d_out);
                 blk.w[name] = encode_weight_matrix_complex(inf, W_re, W_im, d_in, d_out, level);
                 inf.complex_weight_names.insert(name);   // name-keyed registry, stable across stages
             },
             py::arg("inf"), py::arg("name"), py::arg("W_re"), py::arg("W_im"),
             py::arg("d_in"), py::arg("d_out"), py::arg("level") = 0, kRelease)
        .def("set_bias",
             [](EncodedBlock& blk, Inference& inf, const std::string& name,
                const perseus_np::Arr1& b, int d_in, int d_out, bool fill) {
                 const auto v = perseus_np::to_vec(b);
                 perseus_checks::check_vector_max("EncodedBlock.set_bias", name, v, d_out);
                 py::gil_scoped_release nogil;
                 blk.w[name] = {encode_bias_vector(inf, v, d_in, d_out, fill)};
             },
             py::arg("inf"), py::arg("name"), py::arg("b"), py::arg("d_in"), py::arg("d_out"),
             py::arg("fill") = true, "numpy fast path for EncodedBlock.set_bias.")
        .def("set_bias",
             [](EncodedBlock& blk, Inference& inf, const std::string& name,
                const std::vector<double>& b, int d_in, int d_out, bool fill) {
                 perseus_checks::check_vector_max("EncodedBlock.set_bias", name, b, d_out);
                 blk.w[name] = {encode_bias_vector(inf, b, d_in, d_out, fill)};
             },
             py::arg("inf"), py::arg("name"), py::arg("b"), py::arg("d_in"), py::arg("d_out"),
             py::arg("fill") = true, kRelease)
        .def("set_norm_cfg", [](EncodedBlock& blk, const std::string& name, NormConfig cfg) {
            blk.norm_cfg[name] = std::move(cfg);
        }, py::arg("name"), py::arg("cfg"),
             "Store the NormConfig for norm site `name` (copied into inf.norm_cfg at install).")
        .def("set_softmax_cfg", [](EncodedBlock& blk, const std::string& name, SoftmaxConfig cfg) {
            blk.sm_cfg[name] = std::move(cfg);
        }, py::arg("name"), py::arg("cfg"),
             "Store the SoftmaxConfig for softmax site `name` (copied into inf.sm_cfg at install).")
        .def("set_gelu_cfg", [](EncodedBlock& blk, const std::string& name, GeLUConfig cfg) {
            blk.gelu_cfg[name] = std::move(cfg);
        }, py::arg("name"), py::arg("cfg"),
             "Store the GeLUConfig for GeLU site `name` (copied into inf.gelu_cfg at install).");

    m.def("load_block_state",
          [](Inference& inf, const weight_loader::WeightStore& store,
             const config_loader::ParsedConfigs& parsed, const BootstrapPlan& plan, int block_idx) {
              return load_block_state(inf, store, parsed, plan, block_idx, nullptr);
          },
          py::arg("inf"), py::arg("store"), py::arg("configs"), py::arg("plan"), py::arg("block_idx"),
          kRelease,
          "Encode transformer block `block_idx` (weights, configs, plan) into an EncodedBlock.");
    m.def("load_final_ln_state",
          [](Inference& inf, const weight_loader::WeightStore& store,
             const config_loader::ParsedConfigs& parsed, const BootstrapPlan& plan) {
              return load_final_ln_state(inf, store, parsed, plan, nullptr);
          },
          py::arg("inf"), py::arg("store"), py::arg("configs"), py::arg("plan"), kRelease,
          "Encode the final LayerNorm (ln_f) weights + config into an EncodedBlock.");

    m.def("install_block_state", &install_block_state_copy, py::arg("inf"), py::arg("state"), kRelease,
          "Copy state's weights, configs and plan into inf; inf.weight_store then points at it.");
    m.def("load_block_to_device",
          [](Inference& inf, EncodedBlock& blk) { load_block_to_device(inf, blk, nullptr); },
          py::arg("inf"), py::arg("state"), kRelease,
          "Upload every weight plaintext of `state` to the GPU (default stream).");
    m.def("evict_block_from_device", &evict_block_from_device,
          py::arg("inf"), py::arg("state"), kRelease,
          "Evict every weight plaintext of `state` from the GPU.");
    m.def("block_scope", &block_scope, py::arg("block_idx"),
          "Per-block cache/weight scope prefix (\"transformer.h.{b}.\").");
    m.def("install_plan_live", [](Inference& inf, const BootstrapPlan& plan) {
        inf.fhe->install_plan_live(plan);   // no-op (clear) when plan.valid == false
    }, py::arg("inf"), py::arg("plan"),
       "Install a bootstrap plan on the live context (clears it when plan.valid is False).");
    m.def("free_rotation_steps", [](Inference& inf, const std::vector<int32_t>& steps) {
        return inf.fhe->free_rotation_steps(steps);
    }, py::arg("inf"), py::arg("steps"), kRelease,
       "Free the loaded rotation keys for `steps` to reclaim GPU memory; returns the count freed.");
    m.def("cutmax_argmax", &cutmax_argmax,
          py::arg("inf"), py::arg("tiles"), py::arg("vocab"), py::arg("config"), kRelease,
          "Encrypted CutMax argmax over the logit tiles; returns one-hot Z tiles (eager only).");
}
