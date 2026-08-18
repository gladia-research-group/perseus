// perseus._core io layer: weight store, parsed configs, per-block encoded state,
// lm_head tiles, encrypted CutMax argmax.
#include "config_loader.h"
#include "cutmax.h"
#include "encoded_block.h"
#include "model/gpt2.h"
#include "model/gpt2/internal.h"
#include "model/vit.h"
#include "model/bert.h"
#include "weight_loader.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>

#include <filesystem>

namespace py = pybind11;

namespace {
constexpr auto kRelease = py::call_guard<py::gil_scoped_release>();
}

// Persistent lm_head tile store (mirrors GPT2Model::lm_tiles_): encode once, reuse per token.
struct LMHeadCache {
    std::vector<EncodedBlock> tiles;
};

void bind_io(py::module_& m) {
    py::class_<weight_loader::WeightStore>(m, "WeightStore")
        .def_static("from_zip", &weight_loader::WeightStore::from_zip, py::arg("zip_path"), kRelease)
        .def_static("from_dir", &weight_loader::WeightStore::from_dir, py::arg("dir_path"), kRelease);

    py::class_<config_loader::ModelConfig>(m, "ModelConfig")
        .def_readonly("n_layers", &config_loader::ModelConfig::n_layers)
        .def_readonly("n_embd", &config_loader::ModelConfig::n_embd)
        .def_readonly("n_head", &config_loader::ModelConfig::n_head)
        .def_readonly("n_inner", &config_loader::ModelConfig::n_inner);

    py::class_<CutMaxCalib>(m, "CutMaxCalib");
    py::class_<CutMaxConfig>(m, "CutMaxConfig");
    m.def("default_cutmax_config", &default_gpt2_cutmax_config);
    m.def("cutmax_config_from_calib", &cutmax_config_from_calib, py::arg("calib"));

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
        .def(py::init<>())
        .def_readonly("valid", &BootstrapPlan::valid);
    m.def("parse_bootstrap_plan_file", &parse_bootstrap_plan_file, py::arg("path"));

    py::class_<EncodedBlock>(m, "EncodedBlock")
        .def_readonly("prefix", &EncodedBlock::prefix);

    m.def("load_block_state",
          [](Inference& inf, const weight_loader::WeightStore& store,
             const config_loader::ParsedConfigs& parsed, const BootstrapPlan& plan, int block_idx) {
              return load_block_state(inf, store, parsed, plan, block_idx, nullptr);
          },
          py::arg("inf"), py::arg("store"), py::arg("configs"), py::arg("plan"), py::arg("block_idx"),
          kRelease);
    m.def("load_final_ln_state",
          [](Inference& inf, const weight_loader::WeightStore& store,
             const config_loader::ParsedConfigs& parsed, const BootstrapPlan& plan) {
              return load_final_ln_state(inf, store, parsed, plan, nullptr);
          },
          py::arg("inf"), py::arg("store"), py::arg("configs"), py::arg("plan"), kRelease);

    m.def("install_block_state", &install_block_state_copy, py::arg("inf"), py::arg("state"), kRelease);
    m.def("load_block_to_device",
          [](Inference& inf, EncodedBlock& blk) { load_block_to_device(inf, blk, nullptr); },
          py::arg("inf"), py::arg("state"), kRelease);
    m.def("evict_block_from_device", &evict_block_from_device,
          py::arg("inf"), py::arg("state"), kRelease);
    m.def("kv_prefetch_first", &gpt2_kv_prefetch_first,
          py::arg("inf"), py::arg("n_blocks"), kRelease);
    m.def("kv_block_prologue", &gpt2_kv_block_prologue,
          py::arg("inf"), py::arg("block_idx"), py::arg("n_blocks"), kRelease);
    m.def("kv_finalize_last", &gpt2_kv_finalize_last,
          py::arg("inf"), py::arg("n_blocks"), kRelease);
    m.def("block_release", &gpt2_block_release, py::arg("inf"), py::arg("block_idx"), kRelease);
    m.def("apply_final_ln", &apply_final_ln, py::arg("inf"), py::arg("x"), py::arg("lnf"), kRelease);
    m.def("extract_token_i_cachemir", &extract_token_i_cachemir,
          py::arg("inf"), py::arg("filling_ct"), py::arg("i"), kRelease);
    m.def("make_vit_inference", &make_vit_inference,
          py::arg("options") = InferenceOptions{}, kRelease);
    m.def("vit_forward", &vit_forward,
          py::arg("inf"), py::arg("chunks"), py::arg("n_toks"), py::arg("store"),
          py::arg("configs"), py::arg("n_blocks"),
          py::arg("n_toks_imag") = std::vector<int>{}, kRelease);
    m.def("make_bert_inference", &make_bert_inference,
          py::arg("options") = InferenceOptions{}, kRelease);
    m.def("bert_forward", &bert_forward,
          py::arg("inf"), py::arg("chunks"), py::arg("n_toks"), py::arg("store"),
          py::arg("configs"), py::arg("n_blocks"),
          py::arg("n_toks_imag") = std::vector<int>{}, kRelease);
    m.def("block_scope", &block_scope, py::arg("block_idx"));
    m.def("reset_graph_runtime", &gpt2_reset_graph_runtime, py::arg("inf"));
    m.def("reset_kv_cache", &gpt2_reset_kv_cache, py::arg("inf"), py::arg("n_blocks"), kRelease);

    m.def("lm_head_vocab",
          [](const weight_loader::WeightStore& store) {
              return static_cast<int>(store.meta(weight_loader::gpt2_lm_head_name()).shape[0]);
          },
          py::arg("store"));
    py::class_<LMHeadCache>(m, "LMHeadCache").def(py::init<>());
    m.def("lm_head",
          [](Inference& inf, const PackedCtx& x, const weight_loader::WeightStore& store,
             int vocab, const BootstrapPlan& plan, LMHeadCache* cache) {
              return gpt2_lm_head(inf, x, store, vocab, inf.slots,
                                  cache ? &cache->tiles : nullptr, plan);
          },
          py::arg("inf"), py::arg("x"), py::arg("store"), py::arg("vocab"), py::arg("plan"),
          py::arg("cache") = static_cast<LMHeadCache*>(nullptr), kRelease);
    m.def("decode_lm_head_logits",
          [](Inference& inf, const std::vector<PackedCtx>& tiles, int vocab) {
              return decode_lm_head_logits(inf, tiles, vocab, inf.slots);
          },
          py::arg("inf"), py::arg("tiles"), py::arg("vocab"), kRelease);
    m.def("cutmax_argmax", &cutmax_argmax,
          py::arg("inf"), py::arg("tiles"), py::arg("vocab"), py::arg("config"), kRelease);
}
