#include "config_loader.h"
#include "model/vit.h"
#include "model/bert.h"
#include "weight_loader.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

namespace py = pybind11;

namespace {
constexpr auto kRelease = py::call_guard<py::gil_scoped_release>();
}

void bind_io_encoders(py::module_& m) {
    m.def("make_vit_inference", &make_vit_inference,
          py::arg("options") = InferenceOptions{}, kRelease,
          "Encoder Inference: CachemirFilling packing, bidirectional, explicit ViT rot-key set.");
    m.def("vit_forward", &vit_forward,
          py::arg("inf"), py::arg("chunks"), py::arg("n_toks"), py::arg("store"),
          py::arg("configs"), py::arg("n_blocks"),
          py::arg("n_toks_imag") = std::vector<int>{}, kRelease,
          "ViT forward over the packed chunks + CLS tail; returns the lm_head logit tiles.");
    m.def("make_bert_inference", &make_bert_inference,
          py::arg("options") = InferenceOptions{}, kRelease,
          "Encoder Inference for BERT: CachemirFilling packing, bidirectional, no KV cache.");
    m.def("bert_forward", &bert_forward,
          py::arg("inf"), py::arg("chunks"), py::arg("n_toks"), py::arg("store"),
          py::arg("configs"), py::arg("n_blocks"),
          py::arg("n_toks_imag") = std::vector<int>{}, kRelease,
          "BERT forward over the packed chunks; returns the CLS token as a cachemir ct.");
}
