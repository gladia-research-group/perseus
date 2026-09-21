// perseus._core op layer: Inference, PackedCtx, data plane, weights, configs,
// packing-dispatched ops, GPT-2 composites, graph/plan controls.
#include "attention.h"
#include "checks.h"
#include "ckks_primitives.h"
#include "npconv.h"
#include "model/gpt2.h"
#include "model/layer_norm.h"
#include "model/mha.h"
#include "model/mlp.h"
#include "nonlinear.h"

#include <pybind11/numpy.h>
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

// __repr__ field formatting: scalars and enums print as their Python str
// (True, 1.0, GeLUMethod.CHEBYSHEV); vectors print as their length.
template <typename T>
std::string repr_field(const T& v) { return py::str(py::cast(v)); }
template <typename T>
std::string repr_field(const std::vector<T>& v) { return "[" + std::to_string(v.size()) + "]"; }

}  // namespace

void bind_ops(py::module_& m) {
    // ── context ────────────────────────────────────────────────────────────
    py::class_<CKKSContext, std::shared_ptr<CKKSContext>>(m, "Context")
        .def("roundtrip", &roundtrip, py::arg("values"), kRelease,
             "encode -> encrypt -> decrypt -> decode `values` (a context self-test).")
        .def("level_limit", &CKKSContext::level_limit,
             "The highest level a fresh ciphertext can carry in this context.")
        .def("add", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&, const PackedCtx&)>(&CKKSContext::add),
             py::arg("a"), py::arg("b"), kRelease, "a + b (ciphertext + ciphertext).")
        .def("add", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&, double)>(&CKKSContext::add),
             py::arg("a"), py::arg("scalar"), kRelease, "a + scalar.")
        .def("sub", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&, const PackedCtx&)>(&CKKSContext::sub),
             py::arg("a"), py::arg("b"), kRelease, "a - b (ciphertext - ciphertext).")
        .def("sub", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&, double)>(&CKKSContext::sub),
             py::arg("a"), py::arg("scalar"), kRelease, "a - scalar.")
        .def("mult", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&, const PackedCtx&)>(&CKKSContext::mult),
             py::arg("a"), py::arg("b"), kRelease, "a * b (ciphertext * ciphertext, relinearized).")
        .def("mult", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&, double)>(&CKKSContext::mult),
             py::arg("a"), py::arg("scalar"), kRelease, "a * scalar.")
        .def("square", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&)>(&CKKSContext::square),
             py::arg("a"), kRelease, "a * a.")
        .def("inplace_add", static_cast<void (CKKSContext::*)(PackedCtx&, const PackedCtx&)>(&CKKSContext::inplace_add),
             py::arg("a"), py::arg("b"), kRelease, "a += b.")
        .def("bootstrap", [](CKKSContext& c, PackedCtx& p) { c.bootstrap(p.ct); },
             py::arg("ct"), kRelease, "Refresh `ct` to the bootstrap output level (in place).")
        .def("maybe_bootstrap", [](CKKSContext& c, PackedCtx& p) { c.maybe_bootstrap(p.ct); },
             py::arg("ct"), kRelease, "Bootstrap `ct` only if its level is below the auto threshold.")
        .def("bootstrap_hint",
             static_cast<void (CKKSContext::*)(PackedCtx&, int, bool)>(&CKKSContext::bootstrap_hint),
             py::arg("ct"), py::arg("level_threshold"), py::arg("account_pending_rescale") = false,
             "Bootstrap `ct` if it has fewer than `level_threshold` levels left.")
        .def("level_hint", static_cast<void (CKKSContext::*)(PackedCtx&, int)>(&CKKSContext::level_hint),
             py::arg("ct"), py::arg("level"), "Drop `ct` to `level` if it is above it.")
        // ── leaf primitives for Python-authored ops ────────────────────────────
        .def("rotate", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&, int32_t)>(&CKKSContext::rotate),
             py::arg("ct"), py::arg("steps"), kRelease,
             "Cyclic slot rotation by `steps` (out[i] = in[i + steps]); the rotation key for "
             "`steps` must exist in the session's band.")
        .def("conjugate", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&)>(&CKKSContext::conjugate),
             py::arg("ct"), kRelease, "Complex conjugate of every slot (identity on real payloads).")
        .def("negate", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&)>(&CKKSContext::negate),
             py::arg("ct"), kRelease, "-ct.")
        .def_property_readonly("complex_payload",
                               [](const CKKSContext& c) { return c.complex_payload; })
        .def("bootstrap_output_level",
             [](CKKSContext& c) { return static_cast<int>(c.bootstrap_output_level()); })
        .def_property_readonly("has_secret_key",
                               [](const CKKSContext& c) { return static_cast<bool>(c.keys.secretKey); })
        .def_property_readonly("loaded_rot_steps",
                               [](const CKKSContext& c) { return c.loaded_rot_steps; },
                               "Rotation steps this session loaded (keygen band + load_rotation_steps - "
                               "free_rotation_steps); empty after close_session. The ones shared with the "
                               "bootstrap precomputation stay resident with the context.")
        .def("complete_setup", &CKKSContext::complete_setup, kRelease,
             "Run a deferred heavy setup (rot keygen/upload, bts precomps, LoadContext) "
             "now; idempotent no-op when nothing is pending. The C++ drivers call this "
             "defensively at every phase entry.")
        .def_property("magnitude_suppressed",
             [](const CKKSContext& c) { return c.magnitude_capture_suppressed; },
             [](CKKSContext& c, bool v) { c.magnitude_capture_suppressed = v; });

    // ── data types ─────────────────────────────────────────────────────────
    py::enum_<InferenceMode>(m, "InferenceMode")
        .value("Sync", InferenceMode::Sync)
        .value("Threaded", InferenceMode::Threaded)
        .value("Prefetch", InferenceMode::Prefetch);

    py::enum_<WeightGranularity>(m, "WeightGranularity")
        .value("Block", WeightGranularity::Block)
        .value("Sublayer", WeightGranularity::Sublayer)
        .value("Linear", WeightGranularity::Linear)
        .value("Plaintext", WeightGranularity::Plaintext);

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
        .def(py::init<>(),
             "Default model size: GPT-2 small (dim 768, expanded 3072) padded to hidDim 1024 / "
             "expDim 4096, 12 real heads in 16, seqLen 1024.")
        .def_readwrite("dim", &ModelSize::dim)
        .def_readwrite("expanded", &ModelSize::expanded)
        .def_readwrite("hidDim", &ModelSize::hidDim)
        .def_readwrite("expDim", &ModelSize::expDim)
        .def_readwrite("numHeads", &ModelSize::numHeads)
        .def_readwrite("numHeadsReal", &ModelSize::numHeadsReal)
        .def_readwrite("seqLen", &ModelSize::seqLen)
        .def("__repr__", [](const ModelSize& o) {
            return "<ModelSize dim=" + repr_field(o.dim) + " expanded=" + repr_field(o.expanded) +
                   " hidDim=" + repr_field(o.hidDim) + " expDim=" + repr_field(o.expDim) +
                   " numHeads=" + repr_field(o.numHeads) + " numHeadsReal=" + repr_field(o.numHeadsReal) +
                   " seqLen=" + repr_field(o.seqLen) + ">";
        });

    py::class_<InferenceOptions>(m, "InferenceOptions")
        .def(py::init<>(),
             "Default options: GPT-2-small sizes, Cachemir packing, Threaded mode, parallel on, "
             "bench_mode off, default CKKS options.")
        .def(py::init<const InferenceOptions&>(), py::arg("other"),
             "Copy constructor: InferenceOptions(other) is an exact copy (ckks included).")
        .def_readwrite("ckks", &InferenceOptions::ckks)
        .def_readwrite("dim", &InferenceOptions::dim)
        .def_readwrite("expanded", &InferenceOptions::expanded)
        .def_readwrite("hidDim", &InferenceOptions::hidDim)
        .def_readwrite("expDim", &InferenceOptions::expDim)
        .def_readwrite("numHeads", &InferenceOptions::numHeads)
        .def_readwrite("numHeadsReal", &InferenceOptions::numHeadsReal)
        .def_readwrite("seqLen", &InferenceOptions::seqLen)
        .def_readwrite("packing_kind", &InferenceOptions::packing_kind)
        .def_readwrite("aux_packing_kinds", &InferenceOptions::aux_packing_kinds)
        .def_readwrite("mode", &InferenceOptions::mode)
        .def_readwrite("parallel", &InferenceOptions::parallel)
        .def_readwrite("bench_mode", &InferenceOptions::bench_mode)
        .def("__repr__", [](const InferenceOptions& o) {
            return "<InferenceOptions ckks=<CKKSOptions logN=" + repr_field(o.ckks.logN) +
                   " depth=" + repr_field(o.ckks.depth) + " scale_bits=" + repr_field(o.ckks.scale_bits) +
                   " composite_degree=" + repr_field(o.ckks.composite_degree) + ">" +
                   " dim=" + repr_field(o.dim) + " expanded=" + repr_field(o.expanded) +
                   " hidDim=" + repr_field(o.hidDim) + " expDim=" + repr_field(o.expDim) +
                   " numHeads=" + repr_field(o.numHeads) + " numHeadsReal=" + repr_field(o.numHeadsReal) +
                   " seqLen=" + repr_field(o.seqLen) + " packing_kind=" + repr_field(o.packing_kind) +
                   " aux_packing_kinds=" + repr_field(o.aux_packing_kinds) + " mode=" + repr_field(o.mode) +
                   " parallel=" + repr_field(o.parallel) + " bench_mode=" + repr_field(o.bench_mode) + ">";
        });

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
        .def(py::init<>(),
             "Default LayerNorm config: TAYLOR NR init, 16 NR iterations, unit center/output scales, "
             "empty polynomials.")
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
        .def_readwrite("center_scale_sq", &NormConfig::center_scale_sq)
        .def("__repr__", [](const NormConfig& o) {
            return "<NormConfig nr_init_method=" + repr_field(o.nr_init_method) +
                   " nr_iters=" + repr_field(o.nr_iters) + " epsilon=" + repr_field(o.epsilon) +
                   " taylor_z0=" + repr_field(o.taylor_z0) + " center_scale=" + repr_field(o.center_scale) +
                   " inv_out_scale=" + repr_field(o.inv_out_scale) + " Ncoeffs=" + repr_field(o.Ncoeffs) +
                   " Dcoeffs=" + repr_field(o.Dcoeffs) + " lin_alpha=" + repr_field(o.lin_alpha) +
                   " lin_beta=" + repr_field(o.lin_beta) + " gs_lo=" + repr_field(o.gs_lo) +
                   " gs_hi=" + repr_field(o.gs_hi) + " gs_iters=" + repr_field(o.gs_iters) +
                   " center_scale_sq=" + repr_field(o.center_scale_sq) + ">";
        });

    py::class_<SoftmaxConfig>(m, "SoftmaxConfig")
        .def(py::init<>(),
             "Default softmax config: LINEAR GS init with every fit zero/empty (calibration fills them).")
        .def_readwrite("gs_init_method", &SoftmaxConfig::gs_init_method,
                       "Which formula the calibration used for the Goldschmidt seed. A record of "
                       "how init_alpha/init_beta were fitted; the evaluation reads those two, not "
                       "this, so changing it on a live config has no effect.")
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
        .def_readwrite("cheb_b", &SoftmaxConfig::cheb_b)
        .def("__repr__", [](const SoftmaxConfig& o) {
            return "<SoftmaxConfig gs_init_method=" + repr_field(o.gs_init_method) +
                   " log2delta1=" + repr_field(o.log2delta1) + " log2delta2=" + repr_field(o.log2delta2) +
                   " clip_lo=" + repr_field(o.clip_lo) + " clip_hi=" + repr_field(o.clip_hi) +
                   " poly_coeffs=" + repr_field(o.poly_coeffs) + " init_alpha=" + repr_field(o.init_alpha) +
                   " init_beta=" + repr_field(o.init_beta) + " gs_iters_scaled=" + repr_field(o.gs_iters_scaled) +
                   " refine_alpha=" + repr_field(o.refine_alpha) + " refine_beta=" + repr_field(o.refine_beta) +
                   " gs_iters_refine_scaled=" + repr_field(o.gs_iters_refine_scaled) +
                   " per_step_refine_iters=" + repr_field(o.per_step_refine_iters) +
                   " sm_kc_r=" + repr_field(o.sm_kc_r) + " cheb_coeffs=" + repr_field(o.cheb_coeffs) +
                   " cheb_a=" + repr_field(o.cheb_a) + " cheb_b=" + repr_field(o.cheb_b) + ">";
        });

    py::class_<GeLUConfig>(m, "GeLUConfig")
        .def(py::init<>(),
             "Default GELU config: SOFTSIGN_INV_SQRT with the gate on, exp_iters 12, newton_iters 2, "
             "gs_iters 14; fits zero/empty.")
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
        .def_readwrite("gate_cheb_b", &GeLUConfig::gate_cheb_b)
        .def("__repr__", [](const GeLUConfig& o) {
            return "<GeLUConfig method=" + repr_field(o.method) + " gate=" + repr_field(o.gate) +
                   " exp_iters=" + repr_field(o.exp_iters) + " newton_iters=" + repr_field(o.newton_iters) +
                   " gs_iters=" + repr_field(o.gs_iters) + " a=" + repr_field(o.a) + " b=" + repr_field(o.b) +
                   " c=" + repr_field(o.c) + " xmax=" + repr_field(o.xmax) + " z_min=" + repr_field(o.z_min) +
                   " z_max=" + repr_field(o.z_max) + " gs_lo=" + repr_field(o.gs_lo) +
                   " gs_hi=" + repr_field(o.gs_hi) + " lin_alpha=" + repr_field(o.lin_alpha) +
                   " lin_beta=" + repr_field(o.lin_beta) + " inv_out_scale=" + repr_field(o.inv_out_scale) +
                   " Ncoeffs=" + repr_field(o.Ncoeffs) + " Dcoeffs=" + repr_field(o.Dcoeffs) +
                   " cheb_coeffs=" + repr_field(o.cheb_coeffs) + " cheb_a=" + repr_field(o.cheb_a) +
                   " cheb_b=" + repr_field(o.cheb_b) + " thor_p1=" + repr_field(o.thor_p1) +
                   " thor_p2=" + repr_field(o.thor_p2) + " gate_cheb_coeffs=" + repr_field(o.gate_cheb_coeffs) +
                   " gate_cheb_a=" + repr_field(o.gate_cheb_a) + " gate_cheb_b=" + repr_field(o.gate_cheb_b) +
                   ">";
        });

    // ── inference session ──────────────────────────────────────────────────
    py::class_<StepScope>(m, "StepScope")
        .def("__enter__", [](StepScope& s) { s.inf->fhe->push_step(s.label); },
             "Push the scope's label onto the context's step stack.")
        .def("__exit__", [](StepScope& s, py::object, py::object, py::object) {
            s.inf->fhe->pop_step();
            return false;
        }, py::arg("exc_type"), py::arg("exc_value"), py::arg("traceback"),
           "Pop the step label; returns False so exceptions propagate.");

    py::class_<Inference>(m, "Inference")
        .def_readonly("slots", &Inference::slots)
        .def_readonly("logN", &Inference::logN)
        .def_readwrite("size", &Inference::size)
        .def_readwrite("complex", &Inference::complex)
        .def_readwrite("n_tok", &Inference::n_tok)
        .def_readwrite("n_tok_imag", &Inference::n_tok_imag)
        .def_readwrite("token_pair", &Inference::token_pair)
        .def_readwrite("use_cache", &Inference::use_cache)
        .def_readwrite("cache_weights", &Inference::cache_weights)
        .def_readwrite("block_prefix", &Inference::block_prefix)
        .def_readwrite("mode", &Inference::mode)
        .def_readwrite("weight_granularity", &Inference::weight_granularity)
        .def_readwrite("mlp_tile_dim", &Inference::mlp_tile_dim)
        .def_readwrite("strict_masks", &Inference::strict_masks)
        .def_property_readonly("fhe", [](Inference& inf) { return inf.fhe; })
        .def_property("capture_t",
                      [](const Inference& inf) { return inf.output.capture_t; },
                      [](Inference& inf, int t) { inf.output.capture_t = t; })
        .def_property("capture_b",
                      [](const Inference& inf) { return inf.output.capture_b; },
                      [](Inference& inf, int b) { inf.output.capture_b = b; })
        .def("scoped", &Inference::scoped, py::arg("name"),
             "block_prefix + name: the key a block-scoped tag resolves to.")
        .def("step", [](Inference& inf, std::string label) { return StepScope{&inf, std::move(label)}; },
             py::arg("label"), py::keep_alive<0, 1>(),
             "`with inf.step(label):` scopes the ops inside under a step label, as the C++ WithStep does.")
        .def("name_ct", [](Inference& inf, const PackedCtx& pc, const std::string& name) {
            inf.name_graph_ct(pc, name);
        }, py::arg("ct"), py::arg("name"), "Name `ct` in the captured graph, overwriting any existing name.")
        .def("name_ct_if_absent", [](Inference& inf, const PackedCtx& pc, const std::string& name) {
            inf.name_graph_ct_if_absent(pc, name);
        }, py::arg("ct"), py::arg("name"), "Name `ct` in the captured graph only if it has no name yet.")
        .def("enable_graph_capture", &Inference::enable_graph_capture,
             "Start graph capture: a fresh (or cleared) GraphBuilder attached to the context.")
        .def("disable_graph_capture", &Inference::disable_graph_capture,
             "Detach the graph builder from the context and drop it.")
        .def("graph_capture_enabled", &Inference::graph_capture_enabled,
             "True while a graph builder is attached and enabled.")
        .def("export_graph_json", &Inference::export_graph_json, py::arg("path"), kRelease,
             "Write the captured graph to `path` (plus capture_env.json beside it); no-op when not capturing.")
        .def("load_bootstrap_plan_json", &Inference::load_bootstrap_plan_json, py::arg("path"), kRelease,
             "Install the bootstrap placement plan at `path`; returns whether planned bootstraps are enabled.")
        .def("clear_bootstrap_plan", &Inference::clear_bootstrap_plan,
             "Drop the installed bootstrap placement plan.")
        .def("clear_enc_cache", &Inference::clear_enc_cache,
             "Evict and forget every cached encoded plaintext; returns how many were dropped.", kRelease)
        .def("add_affine_term", &Inference::add_affine_term, py::arg("ct"), py::arg("name"), kRelease,
             "ct += the stored per-feature affine term `name` (mirrored into the Im lane under token-pair "
             "packing).")
        .def("set_norm_cfg", [](Inference& inf, const std::string& name, NormConfig cfg) {
            inf.norm_cfg[name] = std::move(cfg);
        }, py::arg("name"), py::arg("cfg"), "Store `cfg` as the NormConfig norm/layer_norm look up by `name`.")
        .def("set_softmax_cfg", [](Inference& inf, const std::string& name, SoftmaxConfig cfg) {
            inf.sm_cfg[name] = std::move(cfg);
        }, py::arg("name"), py::arg("cfg"),
           "Store `cfg` as the SoftmaxConfig attention_softmax_thor looks up by `name`.")
        .def("set_gelu_cfg", [](Inference& inf, const std::string& name, GeLUConfig cfg) {
            inf.gelu_cfg[name] = std::move(cfg);
        }, py::arg("name"), py::arg("cfg"), "Store `cfg` as the GeLUConfig gelu_approx looks up by `name`.")
        .def("set_weight",
             [](Inference& inf, const std::string& name, const perseus_np::Arr2& W,
                int d_in, int d_out, int level) {
                 const auto M = perseus_np::to_mat(W);
                 perseus_checks::check_matrix("set_weight", name, M, d_in, d_out);
                 py::gil_scoped_release nogil;
                 inf.w[name] = encode_weight_matrix(inf, M, d_in, d_out, level);
                 inf.token_basis_weights.insert(name);
             },
             py::arg("name"), py::arg("W"), py::arg("d_in"), py::arg("d_out"), py::arg("level") = 0,
             "numpy fast path: a (d_in, d_out) float array is copied once from its buffer.")
        .def("set_weight",
             [](Inference& inf, const std::string& name, const std::vector<std::vector<double>>& W,
                int d_in, int d_out, int level) {
                 perseus_checks::check_matrix("set_weight", name, W, d_in, d_out);
                 inf.w[name] = encode_weight_matrix(inf, W, d_in, d_out, level);
                 inf.token_basis_weights.insert(name);   // slot-layout checked (slot_layout.h)
             },
             py::arg("name"), py::arg("W"), py::arg("d_in"), py::arg("d_out"), py::arg("level") = 0,
             kRelease,
             "Install a (d_in, d_out) weight matrix under `name` as CKKS plaintexts "
             "(y = x @ W). A wrong shape raises ValueError.")
        .def("set_weight_complex",
             [](Inference& inf, const std::string& name,
                const std::vector<std::vector<double>>& W_re, const std::vector<std::vector<double>>& W_im,
                int d_in, int d_out, int level) {
                 perseus_checks::check_matrix("set_weight_complex", name + ".re", W_re, d_in, d_out);
                 perseus_checks::check_matrix("set_weight_complex", name + ".im", W_im, d_in, d_out);
                 inf.w[name] = encode_weight_matrix_complex(inf, W_re, W_im, d_in, d_out, level);
                 inf.complex_weight_names.insert(name);
             },
             py::arg("name"), py::arg("W_re"), py::arg("W_im"), py::arg("d_in"), py::arg("d_out"),
             py::arg("level") = 0, kRelease)
        .def("set_bias",
             [](Inference& inf, const std::string& name, const perseus_np::Arr1& b,
                int d_in, int d_out, bool fill) {
                 const auto v = perseus_np::to_vec(b);
                 perseus_checks::check_vector_max("set_bias", name, v, d_out);
                 py::gil_scoped_release nogil;
                 inf.w[name] = {encode_bias_vector(inf, v, d_in, d_out, fill)};
             },
             py::arg("name"), py::arg("b"), py::arg("d_in"), py::arg("d_out"), py::arg("fill") = true,
             "numpy fast path for set_bias.")
        .def("set_bias",
             [](Inference& inf, const std::string& name, const std::vector<double>& b,
                int d_in, int d_out, bool fill) {
                 perseus_checks::check_vector_max("set_bias", name, b, d_out);
                 inf.w[name] = {encode_bias_vector(inf, b, d_in, d_out, fill)};
             },
             py::arg("name"), py::arg("b"), py::arg("d_in"), py::arg("d_out"), py::arg("fill") = true,
             kRelease,
             "Install a bias of up to d_out entries under `name` (shorter is zero-filled; "
             "longer raises ValueError).")
        .def("evict_weights", &Inference::evict_weights, py::arg("key"), kRelease,
             "Erase weight `key` (its device plaintexts freed unless weights are resident); no-op if absent.")
        .def_property_readonly("installed_weights",
                               [](const Inference& inf) {
                                   std::vector<std::string> keys;
                                   for (const auto& kv : inf.w) keys.push_back(kv.first);
                                   return keys;
                               },
                               "Names of the weights currently installed (bind order not preserved).")
        // ── plaintext-vector operands (encoded at the ciphertext's level) ─────
        .def("mult_pt",
             [](Inference& inf, const PackedCtx& ct, const perseus_np::Arr1& values) {
                 const auto v = perseus_np::to_vec(values);
                 py::gil_scoped_release nogil;
                 Ptx pt = inf.encode_at(v, ct);
                 return inf.fhe->mult(ct, pt);
             },
             py::arg("ct"), py::arg("values"),
             "ct * values (slot-wise): `values` is encoded as a plaintext at ct's level; "
             "shorter than the slot count is zero-filled.")
        .def("add_pt",
             [](Inference& inf, const PackedCtx& ct, const perseus_np::Arr1& values) {
                 const auto v = perseus_np::to_vec(values);
                 py::gil_scoped_release nogil;
                 Ptx pt = inf.encode_at(v, ct);
                 return inf.fhe->add(ct, pt);
             },
             py::arg("ct"), py::arg("values"), "ct + values (slot-wise plaintext add).")
        .def("eval_chebyshev",
             [](Inference& inf, const PackedCtx& ct, const std::vector<double>& coeffs,
                double a, double b) {
                 py::gil_scoped_release nogil;
                 return eval_chebyshev_series(*inf.fhe, ct, coeffs, a, b);
             },
             py::arg("ct"), py::arg("coeffs"), py::arg("a") = -1.0, py::arg("b") = 1.0,
             "Evaluate sum_k coeffs[k] * T_k(x) slot-wise for x in [a, b] (the runtime's "
             "Chebyshev evaluator — what the GELU/softmax composites use). Levels consumed "
             "grow with the degree; the argument must lie inside [a, b].")
        .def("sum_slots",
             [](Inference& inf, const PackedCtx& ct, int width) {
                 if (width < 1 || (width & (width - 1)))
                     throw std::invalid_argument("sum_slots: width must be a power of two");
                 py::gil_scoped_release nogil;
                 PackedCtx acc = ct;
                 for (int step = 1; step < width; step <<= 1)
                     acc = inf.fhe->add(acc, inf.fhe->rotate(acc, step));
                 return acc;
             },
             py::arg("ct"), py::arg("width"),
             "Rotate-and-add: slot i receives the sum of slots i .. i+width-1 (width a power "
             "of two; needs rotation keys for 1, 2, 4, ... width/2). The lane-0 slot of each "
             "width-aligned group holds that group's total.");

    m.def("close_session",
          [](Inference& inf) {
              py::gil_scoped_release nogil;
              std::vector<std::string> keys;
              for (const auto& kv : inf.w) keys.push_back(kv.first);
              for (const auto& k : keys) inf.evict_weights(k);
              inf.cache.clear();
              inf.cache_mask.clear();
              inf.clear_enc_cache();
              size_t freed = 0;
              if (inf.fhe && !inf.fhe->loaded_rot_steps.empty()) {
                  const std::vector<int> steps = inf.fhe->loaded_rot_steps;
                  freed = inf.fhe->free_rotation_steps(steps);
              }
              return freed;
          },
          py::arg("inf"),
          "Release what the session holds: every installed weight, the KV and mask caches, "
          "the encode cache and the loaded rotation keys. Returns the number of rotation keys "
          "freed (the ones shared with the bootstrap precomputation are protected and stay "
          "with the context; on the GPT-2 n32 band 87 of 133). The memory goes back to the runtime's device pool (FIDESlib keeps freed "
          "limbs in per-size free lists and the CUDA pool's release threshold is unbounded), "
          "so cudaMemGetInfo does not move: it is reused by the next session in this process "
          "and returned to the device at process exit. The CKKS context itself (keys, "
          "bootstrap precomputation) stays alive — ciphertexts still reference it — so a "
          "closed session cannot compute, but a new one can be created.");
    m.def("device_free_gb", []() {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        return static_cast<double>(free_b) / (1 << 30);
    }, "Free memory on the current CUDA device, in GiB (cudaMemGetInfo).");
    m.def("make_inference", [](const InferenceOptions& o) { return make_inference(o); },
          py::arg("options") = InferenceOptions{}, kRelease,
          "Build a generic Inference session from `options`: CKKS context, model sizes and packing.");
    m.def("make_gpt2_inference",
          static_cast<Inference (*)(InferenceOptions)>(&make_gpt2_inference),
          py::arg("options") = InferenceOptions{}, kRelease,
          "make_inference plus the GPT-2 rotation keys for `options`' packing (and aux packings).");

    // ── data plane ─────────────────────────────────────────────────────────
    m.def("encode_token_input",
          [](Inference& inf, const perseus_np::Arr1& x) {
              const auto v = perseus_np::to_vec(x);
              perseus_checks::check_max_len("encode_token_input", v, inf.size.getRealHidDim());
              py::gil_scoped_release nogil;
              return encode_token_input(inf, v);
          },
          py::arg("inf"), py::arg("x"), "numpy fast path for encode_token_input.");
    m.def("encode_token_input",
          [](Inference& inf, const std::vector<double>& x) {
              perseus_checks::check_max_len("encode_token_input", x, inf.size.getRealHidDim());
              return encode_token_input(inf, x);
          },
          py::arg("inf"), py::arg("x"), kRelease,
          "One token's real features (<= size.dim; shorter is zero-padded) -> fresh ciphertext. "
          "Longer raises ValueError instead of silently dropping the tail.");
    m.def("decode_token_output",
          [](Inference& inf, const PackedCtx& ct) {
              std::vector<double> v;
              { py::gil_scoped_release nogil; v = decode_token_output(inf, ct); }
              return perseus_np::from_vec(v);
          },
          py::arg("inf"), py::arg("ct"),
          "Decrypt one token's real features from `ct` (unpack_tokens with T=1).");
    m.def("encode_prefill_input",
          [](Inference& inf, const perseus_np::Arr2& embeddings) {
              const auto m = perseus_np::to_mat(embeddings);
              py::gil_scoped_release nogil;
              return encode_prefill_input(inf, m);
          },
          py::arg("inf"), py::arg("embeddings"), "numpy fast path: a [T][d] float array.");
    m.def("encode_prefill_input", &encode_prefill_input, py::arg("inf"), py::arg("embeddings"),
          kRelease,
          "Pack T token embeddings into one CachemirFilling input ciphertext for prefill "
          "(bootstrap output level; token-pair aware).");
    m.def("decode_tokens_output",
          [](Inference& inf, const PackedCtx& ct, int n_tok) {
              std::vector<std::vector<double>> m;
              { py::gil_scoped_release nogil; m = decode_tokens_output(inf, ct, n_tok); }
              return perseus_np::from_mat(m);
          },
          py::arg("inf"), py::arg("ct"), py::arg("n_tok"),
          "Decode the n_tok tokens packed in `ct` (slot[i*t + tok]) to [n_tok][d_real]; "
          "n_tok=1 equals decode_token_output.");
    // Debug: every slot of a ciphertext (the layout research tool — probe_chain_diag).
    m.def("decrypt_slots",
          [](Inference& inf, const PackedCtx& pc) { return decrypt(inf.cc(), pc.ct, inf.fhe->sk()); },
          py::arg("inf"), py::arg("x"), kRelease);
    // Custom LayerNorm affine (gamma/beta) under `tag`: the per-feature tiles ln_affine reads
    // as <tag>.weight / <tag>.bias when the fold is off for that name (custom names never fold).
    m.def("set_ln_affine",
          [](Inference& inf, const std::string& tag, const std::vector<double>& weight,
             const std::vector<double>& bias, int level) {
              // level<=0: encode at the bootstrap landing (the block loader's wl()
              // convention) instead of level 0 — the bottom-of-chain encode forced a
              // relevel to the ct's depth on every use.
              if (level <= 0) level = static_cast<int>(inf.fhe->bootstrap_output_level());
              const int d_pad = inf.size.hidDim, c_real = inf.size.getRealHidDim();
              inf.w[tag + ".weight"] = {encode_ln_affine_param(inf, weight, d_pad, c_real, level,
                                                               /*mask_inactive=*/false, nullptr)};
              inf.w[tag + ".bias"]   = {encode_ln_affine_param(inf, bias, d_pad, c_real, level,
                                                               /*mask_inactive=*/false, nullptr)};
          },
          py::arg("inf"), py::arg("tag"), py::arg("weight"), py::arg("bias"), py::arg("level") = 0,
          kRelease,
          "Install LayerNorm gamma/beta as <tag>.weight / <tag>.bias for ln_affine(tag) "
          "(level <= 0: encode at the bootstrap output level).");
    // Per-block subgraph capture (FHE_GRAPH_DIR-gated, no-op otherwise): the C++ drivers'
    // capture protocol, so a Python block loop captures the same block_<b>/graph.json layout.
    m.def("begin_subgraph_capture", &begin_subgraph_capture, py::arg("inf"), py::arg("block"),
          "Start capturing block `block` (FHE_GRAPH_DIR set, capture wanted, no graph.json there yet); "
          "returns whether capture began.");
    m.def("end_subgraph_capture", &end_subgraph_capture, py::arg("inf"), py::arg("block"),
          "Finish block `block`'s capture: write its graph.json and detach; no-op when not capturing.");
    m.def("decode_linear_output",
          [](Inference& inf, const PackedCtx& pc, int d_in, int d_out) {
              std::vector<double> v;
              {
                  py::gil_scoped_release nogil;
                  auto raw = decrypt(inf.cc(), pc.ct, inf.fhe->sk());
                  v = decode_linear_output(inf.packing, raw, inf.slots, d_in, d_out);
              }
              return perseus_np::from_vec(v);
          },
          py::arg("inf"), py::arg("ct"), py::arg("d_in"), py::arg("d_out"),
          "Decrypt `ct` and decode a (d_in, d_out) linear output to its d_out values (packing-aware).");
    m.def("pack_tokens",
          [](Inference& inf, const perseus_np::Arr2& embeddings, int target_level) {
              const auto m = perseus_np::to_mat(embeddings);
              py::gil_scoped_release nogil;
              return pack_tokens(inf, m, target_level);
          },
          py::arg("inf"), py::arg("embeddings"), py::arg("target_level") = 0,
          "numpy fast path: a [T][d] float array.");
    m.def("pack_tokens", &pack_tokens, py::arg("inf"), py::arg("embeddings"),
          py::arg("target_level") = 0, kRelease,
          "Encode T token embeddings (each <= size.dim, zero-padded to hidDim) into one ciphertext "
          "at `target_level`.");
    m.def("unpack_tokens", &unpack_tokens, py::arg("inf"), py::arg("ct"), py::arg("T"), kRelease,
          "Decrypt `ct` and decode its T packed tokens to [T][d_real].");

    // ── packing-dispatched ops ─────────────────────────────────────────────
    m.def("linear", &linear, py::arg("inf"), py::arg("x"), py::arg("wname"),
          py::arg("d_in"), py::arg("d_out"), py::arg("stream_pt") = false, kRelease,
          "y = x @ W for weight `wname` (d_in, d_out), dispatched on the packing; `stream_pt` "
          "loads/evicts each weight plaintext around its use.");
    m.def("linear_multi", &linear_multi, py::arg("inf"), py::arg("x"), py::arg("wnames"),
          py::arg("d_in"), py::arg("d_out"), py::arg("stream_pt") = false, kRelease,
          "Prepare `x` once, then apply every weight in `wnames` (d_in, d_out) to it; one output per name.");
    m.def("linear_outputpack", &linear_outputpack, py::arg("inf"), py::arg("x"), py::arg("wname"),
          py::arg("d_in"), py::arg("d_out"), kRelease,
          "Cachemir-only linear whose output blocks are paired into complex slots (output-row pack, S4).");
    m.def("norm",
          static_cast<PackedCtx (*)(Inference&, const PackedCtx&, const std::string&)>(&norm),
          py::arg("inf"), py::arg("x"), py::arg("cfg_name"), kRelease,
          "Normalize `x` under NormConfig `cfg_name`: mean-centred times inverse-sqrt variance, "
          "no gamma/beta.");
    m.def("layer_norm", &layer_norm, py::arg("inf"), py::arg("x"), py::arg("cfg_name"), kRelease,
          "norm(x, cfg_name) plus the LN affine: the folded shift when fold_ln_affine(cfg_name), "
          "else ln_affine.");
    m.def("realize_pending_rescale",
          [](Inference& inf, PackedCtx& x) { inf.fhe->realize_pending_rescale_raw(x.ct); },
          py::arg("inf"), py::arg("x"), kRelease,
          "Realize a deg-2 ciphertext's pending rescale in place (deg 1, level + d). "
          "FIDESlib multPt otherwise re-realizes a COPY inside every product.");
    m.def("ln_affine", &ln_affine, py::arg("inf"), py::arg("normed"), py::arg("tag"), kRelease,
          "normed * <tag>.weight + <tag>.bias: the LayerNorm gamma/beta stored under `tag`.");
    m.def("fold_ln_affine", &fold_ln_affine, py::arg("cfg_name"),
          "Whether this LN's affine is folded into its consumer (GPT2_FOLD_LN_AFFINE, per-tag "
          "GPT2_FOLD_LN1/LN2/LNF).");
    m.def("gelu_approx", &gelu_approx, py::arg("inf"), py::arg("x"), py::arg("cfg_name"), kRelease,
          "GELU(x) under GeLUConfig `cfg_name` (softsign, Chebyshev or THOR composite per cfg.method).");
    m.def("exp_approx", &exp_approx, py::arg("inf"), py::arg("x"), py::arg("r"), kRelease,
          "exp(x) ~= (1 + x / 2^r)^(2^r).");
    m.def("qkt", &qkt, py::arg("inf"), py::arg("query"), kRelease,
          "Attention scores query . K^T against this block's K cache (a list of ciphertexts, "
          "packing-dependent).");
    m.def("attention_softmax_thor", &attention_softmax_thor, py::arg("inf"), py::arg("scores"),
          py::arg("cfg_name"), kRelease,
          "Softmax over the score ciphertexts under SoftmaxConfig `cfg_name` (THOR approximation).");
    m.def("softmax_v", &softmax_v, py::arg("inf"), py::arg("softmax_scores"), kRelease,
          "Multiply the softmax probabilities by this block's V cache; one ciphertext out.");
    m.def("head_reduce_sum", &head_reduce_sum, py::arg("inf"), py::arg("x"), kRelease,
          "Cachemir-only: sum `x` over the t slots of each head lane and broadcast the sum back to all t.");
    m.def("prepare_mha_masks", &prepare_mha_masks, py::arg("inf"), kRelease,
          "Reset this block's K cache before the first push.");
    m.def("prepare_vcache", &prepare_vcache, py::arg("inf"), kRelease,
          "Reset this block's V cache before the first push.");
    m.def("cache_k_push", &cache_k_push, py::arg("inf"), py::arg("key"), kRelease,
          "Append `key` to this block's K cache.");
    m.def("cache_v_push", &cache_v_push, py::arg("inf"), py::arg("value"), kRelease,
          "Append `value` to this block's V cache.");
    m.def("cache_kv_push", &cache_kv_push, py::arg("inf"), py::arg("key"), py::arg("value"), kRelease,
          "Push `key` and `value` into this block's K and V caches.");
    m.def("cache_kv_push_packed", &cache_kv_push_packed, py::arg("inf"), py::arg("kv_packed"), kRelease,
          "Cachemir-only: push a pre-packed K + i*V ciphertext into this block's K and V caches.");

    // ── GPT-2 composites ───────────────────────────────────────────────────
    m.def("mha_block", &mha_block, py::arg("inf"), py::arg("x"), kRelease,
          "Multi-head attention sublayer on `x` (qkv, attention core, out-proj) run as an op sequence.");
    m.def("mlp_block", &mlp_block, py::arg("inf"), py::arg("x"), kRelease,
          "MLP sublayer on `x` (up-linear, GELU, down-linear); tiled when inf.tiled_mlp().");
    m.def("transformer_block", &transformer_block, py::arg("inf"), py::arg("x"), kRelease,
          "One GPT-2 block on `x`: ln_1, MHA, residual, ln_2, MLP, residual; writes the block's "
          "graph.json under FHE_GRAPH_DIR.");
}
