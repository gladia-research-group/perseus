// perseus._core — pybind11 module over cachemir_lib.
#include "fideslib_wrapper.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_ops(py::module_& m);
void bind_io(py::module_& m);
void bind_session(py::module_& m);

namespace {

void throw_test() {
    throw std::runtime_error("[fhe_error] deliberate test throw from _core");
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "perseus CUDA core: FIDESlib/OpenFHE CKKS runtime bindings";

    py::register_exception<std::runtime_error>(m, "FHEError");

    py::class_<CKKSContextOptions>(m, "CKKSOptions")
        .def(py::init<>())
        .def_readwrite("logN", &CKKSContextOptions::logN)
        .def_readwrite("depth", &CKKSContextOptions::depth)
        .def_readwrite("scale_bits", &CKKSContextOptions::scale_bits)
        .def_readwrite("enable_bootstrap", &CKKSContextOptions::enable_bootstrap)
        .def_readwrite("btp_depth_overhead", &CKKSContextOptions::btp_depth_overhead)
        .def_readwrite("level_budget", &CKKSContextOptions::level_budget)
        .def_readwrite("bootstrap_slots", &CKKSContextOptions::bootstrap_slots)
        .def_readwrite("sparse_bts_slots", &CKKSContextOptions::sparse_bts_slots)
        .def_readwrite("sparse_level_budget", &CKKSContextOptions::sparse_level_budget)
        .def_readwrite("btp_scale_bits", &CKKSContextOptions::btp_scale_bits)
        .def_readwrite("correction_factor", &CKKSContextOptions::correction_factor)
        .def_readwrite("first_mod_bits", &CKKSContextOptions::first_mod_bits)
        .def_readwrite("num_large_digits", &CKKSContextOptions::num_large_digits)
        .def_readwrite("auto_bts_level_override", &CKKSContextOptions::auto_bts_level_override)
        .def_readwrite("chain_sizes_per_level", &CKKSContextOptions::chain_sizes_per_level)
        .def_readwrite("batch_size", &CKKSContextOptions::batch_size)
        .def_readwrite("ckks_complex_payload", &CKKSContextOptions::ckks_complex_payload)
        .def_readwrite("h_weight", &CKKSContextOptions::h_weight)
        .def_readwrite("extra_rot_steps", &CKKSContextOptions::extra_rot_steps)
        .def_readwrite("deferred_rot_steps", &CKKSContextOptions::deferred_rot_steps)
        .def_readwrite("bts_iterations", &CKKSContextOptions::bts_iterations)
        .def_readwrite("bts_precision", &CKKSContextOptions::bts_precision)
        .def_static("from_env", &ckks_options_from_env);

    bind_ops(m);
    bind_io(m);

    m.def("make_context", &make_ckks_context,
          py::arg("options") = CKKSContextOptions{},
          py::call_guard<py::gil_scoped_release>());
    m.def("throw_test", &throw_test);

    bind_session(m);
}
