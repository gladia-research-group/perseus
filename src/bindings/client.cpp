#include "client_context.h"
#include "checks.h"
#include "fhe_errors.h"
#include "npconv.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <string>

namespace py = pybind11;
using namespace perseus_client;

namespace {

constexpr auto kRelease = py::call_guard<py::gil_scoped_release>();

template <typename T>
std::string repr_field(const T& v) { return py::str(py::cast(v)); }
template <typename T>
std::string repr_field(const std::vector<T>& v) { return "[" + std::to_string(v.size()) + "]"; }

void throw_test(const std::string& message) {
    throw std::runtime_error(message.empty() ? "[fhe_error] deliberate test throw from _client"
                                             : message);
}

PackingKind parse_packing_kind(const std::string& s) {
    if (s == "cachemir")         return PackingKind::Cachemir;
    if (s == "diagonal")         return PackingKind::Diagonal;
    if (s == "cachemir_filling") return PackingKind::CachemirFilling;
    if (s == "cachemir_complex") return PackingKind::CachemirComplex;
    throw std::invalid_argument("unknown packing '" + s + "'");
}

}  // namespace

PYBIND11_MODULE(_client, m) {
    m.doc() =
        "perseus client core: the CKKS client role (keygen, bundle, encode/encrypt, decrypt, "
        "ciphertext bytes) on OpenFHE alone — no CUDA driver, no FIDESlib. Keys and "
        "ciphertexts interchange with a perseus._core server. List-valued option fields are "
        "copied on access: assign a whole list.";

    static auto fhe_error = py::register_local_exception<std::runtime_error>(m, "FHEError", PyExc_RuntimeError);
    static auto plan_error   = py::register_local_exception<fhe::PlanError>(m, "PlanError", fhe_error.ptr());
    static auto mask_error   = py::register_local_exception<fhe::MaskError>(m, "MaskError", fhe_error.ptr());
    static auto layout_error = py::register_local_exception<fhe::LayoutError>(m, "LayoutError", fhe_error.ptr());
    py::register_local_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const lbcrypto::OpenFHEException& e) {
            PyErr_SetString(fhe_error.ptr(), (std::string("[openfhe] ") + e.what()).c_str());
        }
    });
    py::register_local_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::runtime_error& e) {
            const std::string w = e.what();
            auto has = [&](const char* s) { return w.find(s) != std::string::npos; };
            if (has("[plan_") || has("[plan]"))
                PyErr_SetString(plan_error.ptr(), w.c_str());
            else if (has("[mask_") || has("[mask_gen]"))
                PyErr_SetString(mask_error.ptr(), w.c_str());
            else if (has("[layout_error]"))
                PyErr_SetString(layout_error.ptr(), w.c_str());
            else
                throw;
        }
    });

    py::class_<CKKSContextOptions>(m, "CKKSOptions",
                                   "CKKS parameters for a context (same fields as perseus._core."
                                   "CKKSOptions). List fields are copied on read: set them whole.")
        .def(py::init<>())
        .def(py::init<const CKKSContextOptions&>(), py::arg("other"),
             "Copy constructor: CKKSOptions(other) is an exact copy, hidden fields included.")
        .def_readwrite("logN", &CKKSContextOptions::logN)
        .def_readwrite("composite_degree", &CKKSContextOptions::composite_degree)
        .def_readwrite("defer_heavy_setup", &CKKSContextOptions::defer_heavy_setup)
        .def_readwrite("depth", &CKKSContextOptions::depth)
        .def_readwrite("scale_bits", &CKKSContextOptions::scale_bits)
        .def_readwrite("enable_bootstrap", &CKKSContextOptions::enable_bootstrap)
        .def_readwrite("btp_depth_overhead", &CKKSContextOptions::btp_depth_overhead)
        .def_readwrite("level_budget", &CKKSContextOptions::level_budget)
        .def_readwrite("bootstrap_slots", &CKKSContextOptions::bootstrap_slots)
        .def_readwrite("sparse_bts_slots_list", &CKKSContextOptions::sparse_bts_slots_list)
        .def_property("sparse_bts_slots",
            [](const CKKSContextOptions& o) {
                return o.sparse_bts_slots_list.empty() ? 0u : o.sparse_bts_slots_list.front();
            },
            [](CKKSContextOptions& o, uint32_t v) {
                o.sparse_bts_slots_list = v ? std::vector<uint32_t>{v}
                                            : std::vector<uint32_t>{};
            })
        .def_readwrite("sparse_level_budget", &CKKSContextOptions::sparse_level_budget)
        .def_readwrite("btp_scale_bits", &CKKSContextOptions::btp_scale_bits)
        .def_readwrite("correction_factor", &CKKSContextOptions::correction_factor)
        .def_readwrite("first_mod_bits", &CKKSContextOptions::first_mod_bits)
        .def_readwrite("num_large_digits", &CKKSContextOptions::num_large_digits)
        .def_readwrite("auto_bts_level_override", &CKKSContextOptions::auto_bts_level_override)
        .def_readwrite("batch_size", &CKKSContextOptions::batch_size)
        .def_readwrite("ckks_complex_payload", &CKKSContextOptions::ckks_complex_payload)
        .def_readwrite("h_weight", &CKKSContextOptions::h_weight)
        .def_readwrite("extra_rot_steps", &CKKSContextOptions::extra_rot_steps)
        .def_readwrite("deferred_rot_steps", &CKKSContextOptions::deferred_rot_steps)
        .def_readwrite("bts_iterations", &CKKSContextOptions::bts_iterations)
        .def_readwrite("bts_precision", &CKKSContextOptions::bts_precision)
        .def_readwrite("keys_dir", &CKKSContextOptions::keys_dir)
        .def_readwrite("skip_gpu_load", &CKKSContextOptions::skip_gpu_load)
        .def_static("from_env", &ckks_options_from_env,
                    "The same 17 environment knobs perseus._core.CKKSOptions.from_env reads "
                    "(LOGN, CKKS_DEPTH, ..., SPARSE_BTS_SLOTS, LEVEL_BUDGET, CKKS_COMPLEX); silent.");

    py::enum_<InferenceMode>(m, "InferenceMode")
        .value("Sync", InferenceMode::Sync)
        .value("Threaded", InferenceMode::Threaded)
        .value("Prefetch", InferenceMode::Prefetch);

    py::enum_<PackingKind>(m, "PackingKind")
        .value("Cachemir", PackingKind::Cachemir)
        .value("Diagonal", PackingKind::Diagonal)
        .value("CachemirFilling", PackingKind::CachemirFilling)
        .value("CachemirComplex", PackingKind::CachemirComplex);

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

    py::class_<ClientContext, std::shared_ptr<ClientContext>>(m, "Context",
            "The client's CKKS context: lbcrypto context + key pair (secret key present after "
            "keygen or load_secret_key).")
        .def("bootstrap_output_level",
             [](const ClientContext& c) { return static_cast<int>(c.bootstrap_output_level()); },
             "The parameter formula composite_degree * (btp_depth_overhead + [bts_iterations >= 2]): "
             "what a GPU-less perseus._core session reports (its probe needs a bootstrap).")
        .def_property_readonly("has_secret_key", &ClientContext::has_secret_key)
        .def_property_readonly("loaded_rot_steps",
                               [](const ClientContext& c) { return c.loaded_rot_steps; },
                               "The sorted-unique rotation band the keys were generated for (or the "
                               "bundle sidecar's RotationIndexes when opened with keys_dir).")
        .def_property_readonly("complex_payload",
                               [](const ClientContext& c) { return c.complex_payload; })
        .def_property_readonly("from_keys", [](const ClientContext& c) { return c.from_keys; })
        .def_property_readonly("key_dist", [](const ClientContext& c) { return c.key_dist; })
        .def_property_readonly("key_tag",
                               [](const ClientContext& c) {
                                   return c.kp.publicKey ? c.kp.publicKey->GetKeyTag() : std::string();
                               },
                               "OpenFHE's key tag of this session's key pair (the rotkeys.bin map key).")
        .def_property_readonly("automorphism_key_indexes", &automorphism_key_indexes,
                               "Sorted automorphism indices OpenFHE's store holds for this key tag "
                               "(band + bootstrap rotations + conj + ENCAPS pair after keygen).");

    py::class_<ClientInference>(m, "Inference")
        .def_readonly("slots", &ClientInference::slots)
        .def_readonly("logN", &ClientInference::logN)
        .def_readwrite("size", &ClientInference::size)
        .def_readwrite("complex", &ClientInference::complex)
        .def_readwrite("n_tok", &ClientInference::n_tok)
        .def_readwrite("mode", &ClientInference::mode)
        .def_property_readonly("packing", [](const ClientInference& inf) { return std::string(to_string(inf.packing)); })
        .def_property_readonly("fhe", [](ClientInference& inf) { return inf.fhe; })
        .def("__repr__", [](const ClientInference& inf) {
            return "<perseus._client.Inference " + std::string(to_string(inf.packing)) +
                   " slots=" + std::to_string(inf.slots) + " logN=" + std::to_string(inf.logN) +
                   " dim=" + std::to_string(inf.size.dim) + ">";
        });

    py::class_<ClientCt>(m, "PackedCtx")
        .def_property_readonly("level", [](const ClientCt& p) { return static_cast<uint32_t>(p.ct->GetLevel()); })
        .def_property_readonly("noise_deg", [](const ClientCt& p) { return p.ct->GetNoiseScaleDeg(); })
        .def_property_readonly("packing", [](const ClientCt& p) { return std::string(to_string(p.packing)); })
        .def("__repr__", [](const ClientCt& p) {
            return "<PackedCtx " + std::string(to_string(p.packing)) +
                   " L" + std::to_string(p.ct->GetLevel()) +
                   " d" + std::to_string(p.ct->GetNoiseScaleDeg()) + ">";
        });

    m.def("make_inference", [](const InferenceOptions& o) { return make_inference(o); },
          py::arg("options") = InferenceOptions{}, kRelease,
          "Build a generic client session from `options`: CKKS context + keys, model sizes and packing.");
    m.def("make_gpt2_inference", [](const InferenceOptions& o) { return make_gpt2_inference(o); },
          py::arg("options") = InferenceOptions{}, kRelease,
          "make_inference plus the GPT-2 rotation keys for `options`' packing (and aux packings).");
    m.def("make_context", [](const CKKSContextOptions& o) { return make_client_context(o); },
          py::arg("options") = CKKSContextOptions{}, kRelease,
          "Create a bare CKKS context (keygen, or a bundle's context when keys_dir is set).");

    // ── data plane ─────────────────────────────────────────────────────────
    m.def("pack_tokens",
          [](ClientInference& inf, const perseus_np::Arr2& embeddings, int target_level) {
              const auto mm = perseus_np::to_mat(embeddings);
              py::gil_scoped_release nogil;
              return pack_tokens(inf, mm, target_level);
          },
          py::arg("inf"), py::arg("embeddings"), py::arg("target_level") = 0,
          "numpy fast path: a [T][d] float array.");
    m.def("pack_tokens", &pack_tokens, py::arg("inf"), py::arg("embeddings"),
          py::arg("target_level") = 0, kRelease,
          "Encode T token embeddings (each <= size.dim, zero-padded to hidDim) into one ciphertext "
          "at `target_level` (same slot layout as perseus._core.pack_tokens).");
    m.def("encode_token_input",
          [](ClientInference& inf, const perseus_np::Arr1& x) {
              const auto v = perseus_np::to_vec(x);
              perseus_checks::check_max_len("encode_token_input", v, inf.size.getRealHidDim());
              py::gil_scoped_release nogil;
              return encode_token_input(inf, v);
          },
          py::arg("inf"), py::arg("x"), "numpy fast path for encode_token_input.");
    m.def("encode_token_input",
          [](ClientInference& inf, const std::vector<double>& x) {
              perseus_checks::check_max_len("encode_token_input", x, inf.size.getRealHidDim());
              return encode_token_input(inf, x);
          },
          py::arg("inf"), py::arg("x"), kRelease,
          "One token's real features (<= size.dim; shorter is zero-padded) -> fresh ciphertext at "
          "the formula bootstrap output level. Longer raises ValueError.");
    m.def("decode_token_output",
          [](const ClientInference& inf, const ClientCt& ct) {
              std::vector<double> v;
              { py::gil_scoped_release nogil; v = decode_token_output(inf, ct); }
              return perseus_np::from_vec(v);
          },
          py::arg("inf"), py::arg("ct"),
          "Decrypt one token's real features from `ct` (unpack_tokens with T=1).");
    m.def("decode_tokens_output",
          [](const ClientInference& inf, const ClientCt& ct, int n_tok) {
              std::vector<std::vector<double>> mm;
              { py::gil_scoped_release nogil; mm = decode_tokens_output(inf, ct, n_tok); }
              return perseus_np::from_mat(mm);
          },
          py::arg("inf"), py::arg("ct"), py::arg("n_tok"),
          "Decode the n_tok tokens packed in `ct` (slot[i*t + tok]) to [n_tok][d_real].");
    m.def("unpack_tokens", &unpack_tokens, py::arg("inf"), py::arg("ct"), py::arg("T"), kRelease,
          "Decrypt `ct` and decode its T packed tokens to [T][d_real].");
    m.def("decrypt_slots",
          [](const ClientInference& inf, const ClientCt& pc) { return decrypt(inf, pc); },
          py::arg("inf"), py::arg("x"), kRelease,
          "Every slot of a ciphertext (the layout research tool).");
    m.def("decode_linear_output",
          [](const ClientInference& inf, const ClientCt& pc, int d_in, int d_out) {
              std::vector<double> v;
              {
                  py::gil_scoped_release nogil;
                  auto raw = decrypt(inf, pc);
                  v = decode_linear_output(inf.packing, raw, inf.slots, d_in, d_out);
              }
              return perseus_np::from_vec(v);
          },
          py::arg("inf"), py::arg("ct"), py::arg("d_in"), py::arg("d_out"),
          "Decrypt `ct` and decode a (d_in, d_out) linear output to its d_out values (packing-aware).");
    m.def("decode_lm_head_logits",
          [](const ClientInference& inf, const std::vector<ClientCt>& tiles, int vocab) {
              std::vector<double> v;
              { py::gil_scoped_release nogil; v = decode_lm_head_logits(inf, tiles, vocab); }
              return perseus_np::from_vec(v);
          },
          py::arg("inf"), py::arg("tiles"), py::arg("vocab"),
          "Decrypt lm_head logit tiles into a [vocab] float array (uses inf's secret key).");
    m.def("lm_head_tile_width", &lm_head_tile_width, py::arg("inf"), py::arg("vocab"),
          "The lm_head tile width: hidDim when vocab <= hidDim, else slots.");

    // ── bytes and files ────────────────────────────────────────────────────
    m.def("serialize_ct",
          [](const ClientInference&, const ClientCt& pc) {
              std::string s;
              { py::gil_scoped_release nogil; s = serialize_ct(pc); }
              return py::bytes(s);
          },
          py::arg("inf"), py::arg("x"),
          "Ciphertext -> bytes (OpenFHE binary; the same bytes perseus._core.serialize_ct writes).");
    m.def("deserialize_ct",
          [](const ClientInference& inf, const py::bytes& data) {
              std::string buf = data;
              py::gil_scoped_release nogil;
              return deserialize_ct(inf, buf);
          },
          py::arg("inf"), py::arg("data"),
          "bytes -> PackedCtx in this session's context (the session's packing is stamped).");
    m.def("save_keys", [](const ClientInference& inf, const std::string& dir) { save_keys(*inf.fhe, dir); },
          py::arg("inf"), py::arg("dir"), kRelease,
          "Write the SERVER bundle: context.bin, context.bin.dev, public.key, multkeys.bin, "
          "rotkeys.bin — the layout perseus._core's keys_dir loads; no secret material.");
    m.def("save_secret_key",
          [](const ClientInference& inf, const std::string& path) { save_secret_key(*inf.fhe, path); },
          py::arg("inf"), py::arg("path"), kRelease,
          "Write the CLIENT's secret key. This file never leaves the client.");
    m.def("load_secret_key",
          [](ClientInference& inf, const std::string& path) { load_secret_key(*inf.fhe, path); },
          py::arg("inf"), py::arg("path"), kRelease,
          "Load a secret key written by save_secret_key (either extension) into a session opened "
          "with keys_dir, so it can decrypt.");

    // ── identity taps ──────────────────────────────────────────────────────
    auto dbg = m.def_submodule("_debug",
        "Identity taps against perseus._core (bundle_meta, expected_automorphism_indexes, "
        "automorphism_indexes_in_file, bootstrap_indexes, rot_band, dev_sidecar) and the "
        "error-translation probes (throw_test, throw_typed).");
    dbg.def("bundle_meta",
            [](const InferenceOptions& o, const std::string& family) {
                std::pair<std::string, std::string> meta;
                {
                    py::gil_scoped_release nogil;
                    const InferenceOptions prepared = prepare_family_options(family, o);
                    meta = bundle_meta(prepared.ckks, {});
                }
                return py::make_tuple(py::bytes(meta.first), meta.second);
            },
            py::arg("options"), py::arg("family") = "gpt2",
            "(context.bin bytes, context.bin.dev text) a bundle for `options` / `family` carries: "
            "context + Enable set + bootstrap setups + band, no keygen.");
    dbg.def("expected_automorphism_indexes",
            [](const InferenceOptions& o, const std::string& family) {
                py::gil_scoped_release nogil;
                const InferenceOptions prepared = prepare_family_options(family, o);
                return expected_automorphism_indexes_for(prepared.ckks, {});
            },
            py::arg("options"), py::arg("family") = "gpt2",
            "Sorted automorphism indices the keygen for `options` / `family` writes into rotkeys.bin "
            "(band + every bootstrap precomp's rotations + M-1 conj + M-2/M-4 ENCAPS pair).");
    dbg.def("automorphism_indexes_in_file", &automorphism_indexes_in_file, py::arg("path"), kRelease,
            "{key tag: sorted automorphism indices} of a rotkeys.bin (reads the whole file).");
    dbg.def("bootstrap_indexes",
            [](const InferenceOptions& o, const std::string& family, int slots) {
                py::gil_scoped_release nogil;
                const InferenceOptions prepared = prepare_family_options(family, o);
                return bootstrap_indexes_for(prepared.ckks, slots);
            },
            py::arg("options"), py::arg("family") = "gpt2", py::arg("slots") = 0,
            "FIDESlib's GetBootstrapIndexes for the `slots` precomp (0 = the dense one).");
    dbg.def("rot_band",
            [](const std::string& kind, int slots, int hidDim, int ffDim, int numHeads) {
                return compute_gpt2_rot_indices(parse_packing_kind(kind), slots, hidDim, ffDim, numHeads);
            },
            py::arg("kind"), py::arg("slots"), py::arg("hidDim"), py::arg("ffDim"), py::arg("numHeads"),
            "The GPT-2 rotation band of one packing ('cachemir' | 'diagonal' | 'cachemir_filling').");
    dbg.def("family_rot_band",
            [](const InferenceOptions& o, const std::string& family) {
                return family_rot_band(family, prepare_family_options(family, o));
            },
            py::arg("options"), py::arg("family") = "gpt2",
            "The sorted-unique band make_<family>_inference generates keys for.");
    dbg.def("dev_sidecar", [](const ClientInference& inf) { return dev_sidecar_text(*inf.fhe); },
            py::arg("inf"), "The context.bin.dev text save_keys writes for this session.");
    dbg.def("formula_level",
            [](const CKKSContextOptions& o) {
                ClientContext c;
                c.composite_degree = static_cast<int>(o.composite_degree > 0 ? o.composite_degree : 1);
                c.btp_overhead     = o.enable_bootstrap ? o.btp_depth_overhead : 0u;
                c.bts_iterations   = o.bts_iterations;
                return static_cast<int>(c.bootstrap_output_level());
            },
            py::arg("options"),
            "Context.bootstrap_output_level() for `options` without building a context.");
    dbg.def("throw_test", &throw_test, py::arg("message") = "",
            "Raise a runtime error carrying `message` (default: an [fhe_error] marker).");
    dbg.def("throw_typed", [](const std::string& kind, const std::string& message) {
        if (kind == "plan") throw fhe::PlanError(message);
        if (kind == "mask") throw fhe::MaskError(message);
        if (kind == "layout") throw fhe::LayoutError(message);
        if (kind == "openfhe") OPENFHE_THROW(message);
        throw fhe::FHEError(message);
    }, py::arg("kind"), py::arg("message") = "typed test throw",
       "Raise the typed C++ error `kind` ('plan' | 'mask' | 'layout' | 'openfhe' | other = FHEError).");
    dbg.attr("decrypt_slots") = m.attr("decrypt_slots");
    m.attr("throw_test") = dbg.attr("throw_test");

    m.attr("__version__") = PERSEUS_VERSION;
    m.attr("chain") = stamp::kChain;
    m.attr("native_int_bits") = stamp::kNativeIntBits;
    m.def("build_info", []() {
        py::dict d;
        d["version"] = PERSEUS_VERSION;
        d["chain"] = stamp::kChain;
        d["native_int_bits"] = stamp::kNativeIntBits;
        d["pybind11"] = std::to_string(PYBIND11_VERSION_MAJOR) + "." +
                        std::to_string(PYBIND11_VERSION_MINOR) + "." +
                        std::to_string(PYBIND11_VERSION_PATCH);
        d["cuda_runtime"] = py::none();
        d["backend"] = "client";
        d["compiled"] = __DATE__ " " __TIME__;
        return d;
    }, "Version, chain (n32/n64), native integer width, pybind11 version, backend 'client' "
       "(cuda_runtime None) and the compile timestamp of this extension.");
}
