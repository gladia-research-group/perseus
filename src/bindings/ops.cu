// perseus._core op layer: Inference, PackedCtx, data plane, weights, configs,
// packing-dispatched ops, GPT-2 composites, graph/plan controls.
#include "attention.h"
#include "model/gpt2.h"
#include "model/layer_norm.h"
#include "model/mha.h"
#include "model/mlp.h"
#include "nonlinear.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

constexpr auto kRelease = py::call_guard<py::gil_scoped_release>();

// `with inf.step("label"):` — mirrors the C++ WithStep scoping run_ops applies per op.
struct StepScope {
    Inference* inf;
    std::string label;
};

std::vector<double> roundtrip(CKKSContext& fhe, std::vector<double> values) {
    const size_t n = values.size();
    Ptx pt = fhe.cc->MakeCKKSPackedPlaintext(values, 1);
    Ctx ct = fhe.cc->Encrypt(fhe.keys.publicKey, pt);
    std::vector<double> out = decrypt(fhe.cc, ct, fhe.keys.secretKey);
    out.resize(n);
    return out;
}

}  // namespace

void bind_ops(py::module_& m) {
    // ── context ────────────────────────────────────────────────────────────
    py::class_<CKKSContext, std::shared_ptr<CKKSContext>>(m, "Context")
        .def("roundtrip", &roundtrip, py::arg("values"), kRelease)
        .def("level_limit", &CKKSContext::level_limit)
        .def("add", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&, const PackedCtx&)>(&CKKSContext::add), kRelease)
        .def("add", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&, double)>(&CKKSContext::add), kRelease)
        .def("sub", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&, const PackedCtx&)>(&CKKSContext::sub), kRelease)
        .def("sub", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&, double)>(&CKKSContext::sub), kRelease)
        .def("mult", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&, const PackedCtx&)>(&CKKSContext::mult), kRelease)
        .def("mult", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&, double)>(&CKKSContext::mult), kRelease)
        .def("square", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&)>(&CKKSContext::square), kRelease)
        .def("inplace_add", static_cast<void (CKKSContext::*)(PackedCtx&, const PackedCtx&)>(&CKKSContext::inplace_add), kRelease)
        .def("bootstrap", [](CKKSContext& c, PackedCtx& p) { c.bootstrap(p.ct); }, kRelease)
        .def("maybe_bootstrap", [](CKKSContext& c, PackedCtx& p) { c.maybe_bootstrap(p.ct); }, kRelease)
        .def("bootstrap_hint",
             static_cast<void (CKKSContext::*)(PackedCtx&, int, bool)>(&CKKSContext::bootstrap_hint),
             py::arg("ct"), py::arg("level_threshold"), py::arg("account_pending_rescale") = false)
        .def("level_hint", static_cast<void (CKKSContext::*)(PackedCtx&, int)>(&CKKSContext::level_hint));

    // ── data types ─────────────────────────────────────────────────────────
    py::enum_<InferenceMode>(m, "InferenceMode")
        .value("Sync", InferenceMode::Sync)
        .value("Threaded", InferenceMode::Threaded)
        .value("Prefetch", InferenceMode::Prefetch);

    py::enum_<PackingKind>(m, "PackingKind")
        .value("Cachemir", PackingKind::Cachemir)
        .value("Diagonal", PackingKind::Diagonal)
        .value("CachemirFilling", PackingKind::CachemirFilling)
        .value("CachemirComplex", PackingKind::CachemirComplex);

    py::class_<PackedCtx>(m, "PackedCtx")
        .def_property_readonly("level", [](const PackedCtx& p) { return level_of(p.ct); })
        .def_property_readonly("noise_deg", [](const PackedCtx& p) { return p.ct->GetNoiseScaleDeg(); })
        .def_property_readonly("packing", [](const PackedCtx& p) { return std::string(to_string(p.packing.kind)); })
        .def("__repr__", [](const PackedCtx& p) {
            return "<PackedCtx " + std::string(to_string(p.packing.kind)) +
                   " L" + std::to_string(level_of(p.ct)) +
                   " d" + std::to_string(p.ct->GetNoiseScaleDeg()) + ">";
        });

    py::class_<ModelSize>(m, "ModelSize")
        .def(py::init<>())
        .def_readwrite("dim", &ModelSize::dim)
        .def_readwrite("expanded", &ModelSize::expanded)
        .def_readwrite("hidDim", &ModelSize::hidDim)
        .def_readwrite("expDim", &ModelSize::expDim)
        .def_readwrite("numHeads", &ModelSize::numHeads)
        .def_readwrite("numHeadsReal", &ModelSize::numHeadsReal)
        .def_readwrite("seqLen", &ModelSize::seqLen);

    py::class_<InferenceOptions>(m, "InferenceOptions")
        .def(py::init<>())
        .def_readwrite("ckks", &InferenceOptions::ckks)
        .def_readwrite("dim", &InferenceOptions::dim)
        .def_readwrite("expanded", &InferenceOptions::expanded)
        .def_readwrite("hidDim", &InferenceOptions::hidDim)
        .def_readwrite("expDim", &InferenceOptions::expDim)
        .def_readwrite("numHeads", &InferenceOptions::numHeads)
        .def_readwrite("numHeadsReal", &InferenceOptions::numHeadsReal)
        .def_readwrite("seqLen", &InferenceOptions::seqLen)
        .def_readwrite("packing_kind", &InferenceOptions::packing_kind)
        .def_readwrite("mode", &InferenceOptions::mode);

    // ── approximation configs (calibration-carrying sites bind these) ─────
    py::enum_<NRInitMethod>(m, "NRInitMethod")
        .value("TAYLOR", NRInitMethod::TAYLOR)
        .value("REMEZ", NRInitMethod::REMEZ);
    py::enum_<GSInitMethod>(m, "GSInitMethod")
        .value("LINEAR", GSInitMethod::LINEAR)
        .value("CHEBYSHEV", GSInitMethod::CHEBYSHEV);
    py::enum_<GeLUMethod>(m, "GeLUMethod")
        .value("SOFTSIGN_INV_SQRT", GeLUMethod::SOFTSIGN_INV_SQRT)
        .value("CHEBYSHEV", GeLUMethod::CHEBYSHEV)
        .value("THOR_COMPOSITE", GeLUMethod::THOR_COMPOSITE);

    py::class_<NormConfig>(m, "NormConfig")
        .def(py::init<>())
        .def_readwrite("nr_init_method", &NormConfig::nr_init_method)
        .def_readwrite("nr_iters", &NormConfig::nr_iters)
        .def_readwrite("epsilon", &NormConfig::epsilon)
        .def_readwrite("taylor_z0", &NormConfig::taylor_z0)
        .def_readwrite("center_scale", &NormConfig::center_scale)
        .def_readwrite("inv_out_scale", &NormConfig::inv_out_scale)
        .def_readwrite("Ncoeffs", &NormConfig::Ncoeffs)
        .def_readwrite("Dcoeffs", &NormConfig::Dcoeffs)
        .def_readwrite("lin_alpha", &NormConfig::lin_alpha)
        .def_readwrite("lin_beta", &NormConfig::lin_beta)
        .def_readwrite("gs_lo", &NormConfig::gs_lo)
        .def_readwrite("gs_hi", &NormConfig::gs_hi)
        .def_readwrite("gs_iters", &NormConfig::gs_iters)
        .def_readwrite("center_scale_sq", &NormConfig::center_scale_sq);

    py::class_<SoftmaxConfig>(m, "SoftmaxConfig")
        .def(py::init<>())
        .def_readwrite("gs_init_method", &SoftmaxConfig::gs_init_method)
        .def_readwrite("log2delta1", &SoftmaxConfig::log2delta1)
        .def_readwrite("log2delta2", &SoftmaxConfig::log2delta2)
        .def_readwrite("clip_lo", &SoftmaxConfig::clip_lo)
        .def_readwrite("clip_hi", &SoftmaxConfig::clip_hi)
        .def_readwrite("poly_coeffs", &SoftmaxConfig::poly_coeffs)
        .def_readwrite("init_alpha", &SoftmaxConfig::init_alpha)
        .def_readwrite("init_beta", &SoftmaxConfig::init_beta)
        .def_readwrite("gs_iters_scaled", &SoftmaxConfig::gs_iters_scaled)
        .def_readwrite("refine_alpha", &SoftmaxConfig::refine_alpha)
        .def_readwrite("refine_beta", &SoftmaxConfig::refine_beta)
        .def_readwrite("gs_iters_refine_scaled", &SoftmaxConfig::gs_iters_refine_scaled)
        .def_readwrite("per_step_refine_iters", &SoftmaxConfig::per_step_refine_iters)
        .def_readwrite("sm_kc_r", &SoftmaxConfig::sm_kc_r)
        .def_readwrite("cheb_coeffs", &SoftmaxConfig::cheb_coeffs)
        .def_readwrite("cheb_a", &SoftmaxConfig::cheb_a)
        .def_readwrite("cheb_b", &SoftmaxConfig::cheb_b);

    py::class_<GeLUConfig>(m, "GeLUConfig")
        .def(py::init<>())
        .def_readwrite("method", &GeLUConfig::method)
        .def_readwrite("gate", &GeLUConfig::gate)
        .def_readwrite("exp_iters", &GeLUConfig::exp_iters)
        .def_readwrite("newton_iters", &GeLUConfig::newton_iters)
        .def_readwrite("gs_iters", &GeLUConfig::gs_iters)
        .def_readwrite("a", &GeLUConfig::a)
        .def_readwrite("b", &GeLUConfig::b)
        .def_readwrite("c", &GeLUConfig::c)
        .def_readwrite("xmax", &GeLUConfig::xmax)
        .def_readwrite("z_min", &GeLUConfig::z_min)
        .def_readwrite("z_max", &GeLUConfig::z_max)
        .def_readwrite("gs_lo", &GeLUConfig::gs_lo)
        .def_readwrite("gs_hi", &GeLUConfig::gs_hi)
        .def_readwrite("lin_alpha", &GeLUConfig::lin_alpha)
        .def_readwrite("lin_beta", &GeLUConfig::lin_beta)
        .def_readwrite("inv_out_scale", &GeLUConfig::inv_out_scale)
        .def_readwrite("Ncoeffs", &GeLUConfig::Ncoeffs)
        .def_readwrite("Dcoeffs", &GeLUConfig::Dcoeffs)
        .def_readwrite("cheb_coeffs", &GeLUConfig::cheb_coeffs)
        .def_readwrite("cheb_a", &GeLUConfig::cheb_a)
        .def_readwrite("cheb_b", &GeLUConfig::cheb_b)
        .def_readwrite("thor_p1", &GeLUConfig::thor_p1)
        .def_readwrite("thor_p2", &GeLUConfig::thor_p2)
        .def_readwrite("gate_cheb_coeffs", &GeLUConfig::gate_cheb_coeffs)
        .def_readwrite("gate_cheb_a", &GeLUConfig::gate_cheb_a)
        .def_readwrite("gate_cheb_b", &GeLUConfig::gate_cheb_b);

    // ── inference session ──────────────────────────────────────────────────
    py::class_<StepScope>(m, "StepScope")
        .def("__enter__", [](StepScope& s) { s.inf->fhe->push_step(s.label); })
        .def("__exit__", [](StepScope& s, py::object, py::object, py::object) {
            s.inf->fhe->pop_step();
            return false;
        });

    py::class_<Inference>(m, "Inference")
        .def_readonly("slots", &Inference::slots)
        .def_readonly("logN", &Inference::logN)
        .def_readwrite("size", &Inference::size)
        .def_readwrite("complex", &Inference::complex)
        .def_readwrite("n_tok", &Inference::n_tok)
        .def_readwrite("n_tok_imag", &Inference::n_tok_imag)
        .def_readwrite("token_pair", &Inference::token_pair)
        .def_readwrite("bidirectional", &Inference::bidirectional)
        .def_readwrite("use_cache", &Inference::use_cache)
        .def_readwrite("cache_weights", &Inference::cache_weights)
        .def_readwrite("block_prefix", &Inference::block_prefix)
        .def_readwrite("mlp_tile_dim", &Inference::mlp_tile_dim)
        .def_readwrite("strict_masks", &Inference::strict_masks)
        .def_property_readonly("fhe", [](Inference& inf) { return inf.fhe; })
        .def_property("capture_t",
                      [](const Inference& inf) { return inf.output.capture_t; },
                      [](Inference& inf, int t) { inf.output.capture_t = t; })
        .def_property("capture_b",
                      [](const Inference& inf) { return inf.output.capture_b; },
                      [](Inference& inf, int b) { inf.output.capture_b = b; })
        .def("scoped", &Inference::scoped)
        .def("step", [](Inference& inf, std::string label) { return StepScope{&inf, std::move(label)}; },
             py::arg("label"), py::keep_alive<0, 1>())
        .def("name_ct", [](Inference& inf, const PackedCtx& pc, const std::string& name) {
            inf.name_graph_ct(pc, name);
        })
        .def("name_ct_if_absent", [](Inference& inf, const PackedCtx& pc, const std::string& name) {
            inf.name_graph_ct_if_absent(pc, name);
        })
        .def("enable_graph_capture", &Inference::enable_graph_capture)
        .def("disable_graph_capture", &Inference::disable_graph_capture)
        .def("graph_capture_enabled", &Inference::graph_capture_enabled)
        .def("export_graph_json", &Inference::export_graph_json)
        .def("load_bootstrap_plan_json", &Inference::load_bootstrap_plan_json)
        .def("clear_bootstrap_plan", &Inference::clear_bootstrap_plan)
        .def("clear_enc_cache", &Inference::clear_enc_cache)
        .def("add_affine_term", &Inference::add_affine_term, kRelease)
        .def("set_norm_cfg", [](Inference& inf, const std::string& name, NormConfig cfg) {
            inf.norm_cfg[name] = std::move(cfg);
        })
        .def("set_softmax_cfg", [](Inference& inf, const std::string& name, SoftmaxConfig cfg) {
            inf.sm_cfg[name] = std::move(cfg);
        })
        .def("set_gelu_cfg", [](Inference& inf, const std::string& name, GeLUConfig cfg) {
            inf.gelu_cfg[name] = std::move(cfg);
        })
        .def("set_weight",
             [](Inference& inf, const std::string& name, const std::vector<std::vector<double>>& W,
                int d_in, int d_out, int level) {
                 inf.w[name] = encode_weight_matrix(inf, W, d_in, d_out, level);
             },
             py::arg("name"), py::arg("W"), py::arg("d_in"), py::arg("d_out"), py::arg("level") = 0,
             kRelease)
        .def("set_weight_complex",
             [](Inference& inf, const std::string& name,
                const std::vector<std::vector<double>>& W_re, const std::vector<std::vector<double>>& W_im,
                int d_in, int d_out, int level) {
                 inf.w[name] = encode_weight_matrix_complex(inf, W_re, W_im, d_in, d_out, level);
                 inf.complex_weight_names.insert(name);
             },
             py::arg("name"), py::arg("W_re"), py::arg("W_im"), py::arg("d_in"), py::arg("d_out"),
             py::arg("level") = 0, kRelease)
        .def("set_bias",
             [](Inference& inf, const std::string& name, const std::vector<double>& b,
                int d_in, int d_out, bool fill) {
                 inf.w[name] = {encode_bias_vector(inf, b, d_in, d_out, fill)};
             },
             py::arg("name"), py::arg("b"), py::arg("d_in"), py::arg("d_out"), py::arg("fill") = true,
             kRelease)
        .def("evict_weights", &Inference::evict_weights);

    m.def("device_free_gb", []() {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        return static_cast<double>(free_b) / (1 << 30);
    });
    m.def("make_inference", [](const InferenceOptions& o) { return make_inference(o); },
          py::arg("options") = InferenceOptions{}, kRelease);
    m.def("make_gpt2_inference",
          static_cast<Inference (*)(InferenceOptions)>(&make_gpt2_inference),
          py::arg("options") = InferenceOptions{}, kRelease);

    // ── data plane ─────────────────────────────────────────────────────────
    m.def("encode_token_input", &encode_token_input, py::arg("inf"), py::arg("x"), kRelease);
    m.def("decode_token_output", &decode_token_output, py::arg("inf"), py::arg("ct"), kRelease);
    m.def("encode_prefill_input", &encode_prefill_input, py::arg("inf"), py::arg("embeddings"),
          kRelease);
    m.def("decode_tokens_output", &decode_tokens_output, py::arg("inf"), py::arg("ct"),
          py::arg("n_tok"), kRelease);
    m.def("decode_linear_output",
          [](Inference& inf, const PackedCtx& pc, int d_in, int d_out) {
              auto raw = decrypt(inf.cc(), pc.ct, inf.fhe->sk());
              return decode_linear_output(inf.packing, raw, inf.slots, d_in, d_out);
          },
          py::arg("inf"), py::arg("ct"), py::arg("d_in"), py::arg("d_out"), kRelease);
    m.def("pack_tokens", &pack_tokens, py::arg("inf"), py::arg("embeddings"),
          py::arg("target_level") = 0, kRelease);
    m.def("unpack_tokens", &unpack_tokens, py::arg("inf"), py::arg("ct"), py::arg("T"), kRelease);

    // ── packing-dispatched ops ─────────────────────────────────────────────
    m.def("linear", &linear, py::arg("inf"), py::arg("x"), py::arg("wname"),
          py::arg("d_in"), py::arg("d_out"), py::arg("stream_pt") = false, kRelease);
    m.def("linear_multi", &linear_multi, py::arg("inf"), py::arg("x"), py::arg("wnames"),
          py::arg("d_in"), py::arg("d_out"), py::arg("stream_pt") = false, kRelease);
    m.def("linear_outputpack", &linear_outputpack, py::arg("inf"), py::arg("x"), py::arg("wname"),
          py::arg("d_in"), py::arg("d_out"), kRelease);
    m.def("norm",
          static_cast<PackedCtx (*)(Inference&, const PackedCtx&, const std::string&)>(&norm),
          py::arg("inf"), py::arg("x"), py::arg("cfg_name"), kRelease);
    m.def("layer_norm", &layer_norm, py::arg("inf"), py::arg("x"), py::arg("cfg_name"), kRelease);
    m.def("ln_affine", &ln_affine, py::arg("inf"), py::arg("normed"), py::arg("tag"), kRelease);
    m.def("fold_ln_affine", &fold_ln_affine, py::arg("cfg_name"));
    m.def("gelu_approx", &gelu_approx, py::arg("inf"), py::arg("x"), py::arg("cfg_name"), kRelease);
    m.def("exp_approx", &exp_approx, py::arg("inf"), py::arg("x"), py::arg("r"), kRelease);
    m.def("qkt", &qkt, py::arg("inf"), py::arg("query"), kRelease);
    m.def("attention_softmax_thor", &attention_softmax_thor, py::arg("inf"), py::arg("scores"),
          py::arg("cfg_name"), kRelease);
    m.def("softmax_v", &softmax_v, py::arg("inf"), py::arg("softmax_scores"), kRelease);
    m.def("head_reduce_sum", &head_reduce_sum, py::arg("inf"), py::arg("x"), kRelease);
    m.def("prepare_mha_masks", &prepare_mha_masks, py::arg("inf"), kRelease);
    m.def("prepare_vcache", &prepare_vcache, py::arg("inf"), kRelease);
    m.def("cache_k_push", &cache_k_push, py::arg("inf"), py::arg("key"), kRelease);
    m.def("cache_v_push", &cache_v_push, py::arg("inf"), py::arg("value"), kRelease);
    m.def("cache_kv_push", &cache_kv_push, py::arg("inf"), py::arg("key"), py::arg("value"), kRelease);
    m.def("cache_kv_push_packed", &cache_kv_push_packed, py::arg("inf"), py::arg("kv_packed"), kRelease);

    // ── GPT-2 composites ───────────────────────────────────────────────────
    m.def("mha_block", &mha_block, py::arg("inf"), py::arg("x"), kRelease);
    m.def("mlp_block", &mlp_block, py::arg("inf"), py::arg("x"), kRelease);
    m.def("transformer_block", &transformer_block, py::arg("inf"), py::arg("x"), kRelease);
}
