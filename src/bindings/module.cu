// perseus._core — pybind11 module over cachemir_lib.
#include "fideslib_wrapper.h"
#include "build_stamp.h"
#include "fhe_errors.h"
#include "interrupt.h"
#include "utils/exception.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

namespace py = pybind11;

using perseus_stamp::kChain;
using perseus_stamp::kNativeIntBits;

// Registration order (the calls in PYBIND11_MODULE below run in this sequence).
//
// pybind11 converts default-argument VALUES at def time: a `py::arg("x") = T{}` or
// `= static_cast<T*>(nullptr)` needs T's py::class_ registered already (otherwise
// pybind11_fail("arg(): could not convert default argument") at import — even for a
// nullptr), and an m.attr() lookup needs the name to exist. So types come before the
// functions that default to them, and the app layer comes last:
//
//   1 CKKSOptions (this file)      -> make_context(options=CKKSOptions{}) in step 8
//   2 bind_ops     Context, PackedCtx, ModelSize, InferenceOptions, Norm/Softmax/GeLUConfig,
//                  Inference, decrypt_slots -> make_inference / make_gpt2_inference (here)
//                  default to InferenceOptions{}; _debug aliases decrypt_slots (step 8)
//   3 bind_io      WeightStore, ModelConfig, CutMaxCalib/CutMaxConfig, ParsedConfigs,
//                  BootstrapPlan, EncodedBlock (set_*_cfg take the step-2 config types)
//                  + the block-state loaders
//   4 bind_io_gpt2 LMHeadCache first, then the defs defaulting to LMHeadCache* /
//                  BootstrapPlan* = nullptr (needs 3)
//   6 bind_pipeline Stage / run_stages (hold EncodedBlock: needs 3)
//   7 bind_serial  ciphertext / key / block-artifact transport (EncodedBlock, WeightStore,
//                  ParsedConfigs, BootstrapPlan arguments: needs 3)
//   8 make_context, _debug, stamps, build_info (this file)
//   9 bind_session RunConfig / RunResult / GtSteps / DecodeSession — the app layer, last
//
// A wrong order is an import-time failure, never a silent drift; the import is the test.
void bind_ops(py::module_& m);          // 2
void bind_io(py::module_& m);           // 3
void bind_io_gpt2(py::module_& m);      // 4
void bind_pipeline(py::module_& m);     // 6
void bind_serial(py::module_& m);       // 7
void bind_session(py::module_& m);      // 9

namespace {
// Tag types for the typed exception hierarchy (never thrown; py::exception keys on them).
struct PlanErrTag {};
struct MaskErrTag {};
struct LayoutErrTag {};
}  // namespace

namespace {

void throw_test(const std::string& message) {
    throw std::runtime_error(message.empty() ? "[fhe_error] deliberate test throw from _core"
                                             : message);
}

bool fatal_exit_disabled() {
    const char* e = std::getenv("PERSEUS_FATAL_EXIT");
    return e && std::strcmp(e, "0") == 0;
}

void install_fatal_exit_handler() {
    std::set_terminate([] {
        if (auto e = std::current_exception()) {
            try { std::rethrow_exception(e); }
            catch (const std::exception& ex) {
                std::fprintf(stderr, "[fatal] terminate: %s\n", ex.what());
            } catch (...) {
                std::fprintf(stderr, "[fatal] terminate: non-std exception\n");
            }
        } else {
            std::fprintf(stderr, "[fatal] terminate: no active exception\n");
        }
        std::fprintf(stderr, "[fatal] hard-exiting so the GPU allocation is reclaimed\n");
        std::fflush(stderr);
        std::_Exit(134);
    });
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() =
        "perseus CUDA core: FIDESlib/OpenFHE CKKS runtime bindings.\n\n"
        "Threading: one session (Inference) per process; every GPU-heavy call releases the "
        "GIL, but the runtime's state is process-global, so drive one session from one thread "
        "at a time (the residency pipeline owns its own worker). CUDA errors inside FIDESlib "
        "are fatal (the library exits the process); parameter and shape errors raise "
        "ValueError / FHEError before any device work. List-valued option fields "
        "(CKKSOptions.level_budget, extra_rot_steps, ...) are copied on access: assign a whole "
        "list, in-place mutation of the returned list does nothing.";

    if (!fatal_exit_disabled()) install_fatal_exit_handler();

    perseus_interrupt::hook() = [] {
        py::gil_scoped_acquire gil;
        if (PyErr_CheckSignals() != 0) throw py::error_already_set();
    };

    static auto fhe_error = py::register_exception<std::runtime_error>(m, "FHEError", PyExc_RuntimeError);

    static auto plan_error   = py::register_exception<fhe::PlanError>(m, "PlanError", fhe_error.ptr());
    static auto mask_error   = py::register_exception<fhe::MaskError>(m, "MaskError", fhe_error.ptr());
    static auto layout_error = py::register_exception<fhe::LayoutError>(m, "LayoutError", fhe_error.ptr());

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const lbcrypto::OpenFHEException& e) {
            PyErr_SetString(fhe_error.ptr(), (std::string("[openfhe] ") + e.what()).c_str());
        }
    });
    // Fallback for any runtime_error thrown with a bare marker rather than a typed
    // exception: matched by sniffing the message.
    py::register_exception_translator([](std::exception_ptr p) {
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
                throw;   // fall through to the FHEError registration
        }
    });

    py::class_<CKKSContextOptions>(m, "CKKSOptions",
                                   "CKKS parameters for a context. List fields are copied on "
                                   "read: set them whole (o.level_budget = [4, 3]).")
        .def(py::init<>())
        .def(py::init<const CKKSContextOptions&>(), py::arg("other"),
             "Copy constructor: CKKSOptions(other) is an exact copy, hidden fields included.")
        .def_readwrite("logN", &CKKSContextOptions::logN)
        // NATIVEINT=32 composite chain: primes per CKKS level (1 = plain chain). Set by
        // from_env (COMPOSITE_DEGREE); it must travel with every copy and manifest.
        .def_readwrite("composite_degree", &CKKSContextOptions::composite_degree)
        .def_readwrite("defer_heavy_setup", &CKKSContextOptions::defer_heavy_setup)
        .def_readwrite("depth", &CKKSContextOptions::depth)
        .def_readwrite("scale_bits", &CKKSContextOptions::scale_bits)
        .def_readwrite("enable_bootstrap", &CKKSContextOptions::enable_bootstrap)
        .def_readwrite("btp_depth_overhead", &CKKSContextOptions::btp_depth_overhead)
        .def_readwrite("level_budget", &CKKSContextOptions::level_budget)
        .def_readwrite("bootstrap_slots", &CKKSContextOptions::bootstrap_slots)
        .def_readwrite("sparse_bts_slots_list", &CKKSContextOptions::sparse_bts_slots_list)
        // back-compat scalar view: reads the routing default, writes a one-entry list
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
        .def_static("from_env", &ckks_options_from_env);

    bind_ops(m);          // 2
    bind_io(m);           // 3
    bind_io_gpt2(m);      // 4
    bind_pipeline(m);     // 6
    bind_serial(m);       // 7

    m.def("make_context", &make_ckks_context,
          py::arg("options") = CKKSContextOptions{},
          py::call_guard<py::gil_scoped_release>(),
          "Create a bare CKKS context (keygen) from CKKSOptions; model sessions use "
          "make_gpt2_inference / make_inference.");
    auto dbg = m.def_submodule("_debug",
        "Harness and research taps (hard_exit, throw_test, install_fatal_exit_handler, "
        "decrypt_slots): for probes, tests and diagnostics, not an application API.");
    dbg.def("throw_test", &throw_test, py::arg("message") = "",
            "Raise a runtime error carrying `message` (default: an [fhe_error] marker) so the "
            "Python-side exception translation can be tested without a context.");
    dbg.def("install_fatal_exit_handler", &install_fatal_exit_handler,
            "Install the std::terminate handler that _Exit(134)s the process after printing the "
            "escaped exception (done at import unless PERSEUS_FATAL_EXIT=0).");
    dbg.def("hard_exit", [](int code) { std::fflush(nullptr); std::_Exit(code); },
            py::arg("code") = 0,
            "Flush and std::_Exit(code), skipping Python and C++ teardown. A harness escape "
            "hatch for a crash during cross-library static destruction, not an API for "
            "applications.");
    dbg.def("throw_typed", [](const std::string& kind, const std::string& message) {
        if (kind == "plan") throw fhe::PlanError(message);
        if (kind == "mask") throw fhe::MaskError(message);
        if (kind == "layout") throw fhe::LayoutError(message);
        if (kind == "openfhe") OPENFHE_THROW(message);
        throw fhe::FHEError(message);
    }, py::arg("kind"), py::arg("message") = "typed test throw",
       "Raise the typed C++ error `kind` ('plan' | 'mask' | 'layout' | 'openfhe' | other = FHEError) "
       "so the translation is testable without a context.");
    dbg.attr("decrypt_slots") = m.attr("decrypt_slots");
    m.attr("throw_test") = dbg.attr("throw_test");
    m.attr("install_fatal_exit_handler") = dbg.attr("install_fatal_exit_handler");

    m.attr("__version__") = PERSEUS_VERSION;
    m.attr("chain") = kChain;                 // "n32" | "n64": the deps tree linked in
    m.attr("native_int_bits") = kNativeIntBits;
    m.def("build_info", []() {
        py::dict d;
        d["version"] = PERSEUS_VERSION;
        d["chain"] = kChain;
        d["native_int_bits"] = kNativeIntBits;
        d["pybind11"] = std::to_string(PYBIND11_VERSION_MAJOR) + "." +
                        std::to_string(PYBIND11_VERSION_MINOR) + "." +
                        std::to_string(PYBIND11_VERSION_PATCH);
        int rt = 0;
        cudaRuntimeGetVersion(&rt);
        d["cuda_runtime"] = rt;
        d["compiled"] = __DATE__ " " __TIME__;
        return d;
    }, "Version, chain (n32/n64), native integer width, pybind11 and CUDA runtime versions, "
       "and the compile timestamp of this extension.");
    m.attr("hard_exit") = dbg.attr("hard_exit");

    bind_session(m);      // 9
}
