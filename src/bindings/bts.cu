// perseus._core.bts: the GPU bootstrap as stages, for a Python orchestration (perseus/impl/bootstrap.py).
// The stage functions are FIDESlib's (CKKS/BootstrapStages.cuh); this file only adds the handles: a device
// ciphertext view (DeviceCt), the stage state (BtsState), the C++ references, a bit-exact comparison and the
// override hook that routes the runtime's own bootstraps to a Python callable.
#include "fideslib_wrapper.h"
#include "packing/packed_ctx.h"

#include <CKKS/BootstrapStages.cuh>
#include <CKKS/Bootstrap.cuh>
#include <CKKS/Ciphertext.cuh>
#include <CKKS/Spru.cuh>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

namespace py = pybind11;
namespace FC = FIDESlib::CKKS;

namespace {

constexpr auto kRelease = py::call_guard<py::gil_scoped_release>();

// A FIDESlib device ciphertext. `owner` keeps whatever owns the storage alive: the PackedCtx's API ciphertext for a
// borrowed view, nothing for a view handed in by the override hook (valid for the duration of that call only).
struct DeviceCt {
    FC::Ciphertext* ct = nullptr;
    std::shared_ptr<void> owner;
    FC::Ciphertext& get() const {
        if (!ct) throw std::runtime_error("[bts] empty DeviceCt");
        return *ct;
    }
};

DeviceCt device_of(CKKSContext& fhe, PackedCtx& p) {
    fhe.cc->LoadCiphertext(p.ct);
    auto sp = std::static_pointer_cast<FC::Ciphertext>(fhe.cc->GetDeviceCiphertext(p.ct->gpu));
    return DeviceCt{sp.get(), std::shared_ptr<void>(p.ct, (void*)p.ct.get())};
}

long mismatches(const DeviceCt& a, const DeviceCt& b) {
    auto& x = a.get();
    auto& y = b.get();
    if (x.getLevel() != y.getLevel() || x.NoiseLevel != y.NoiseLevel) return -1;
    cudaDeviceSynchronize();
    long bad = 0;
    for (int part = 0; part < 2; ++part) {
        std::vector<std::vector<uint64_t>> u, v;
        (part ? x.c1 : x.c0).store(u);
        (part ? y.c1 : y.c0).store(v);
        cudaDeviceSynchronize();
        for (size_t i = 0; i < u.size() && i < v.size(); ++i)
            for (size_t n = 0; n < u[i].size(); ++n) bad += u[i][n] != v[i][n];
    }
    return bad;
}

}  // namespace

void bind_bts(py::module_& root) {
    auto m = root.def_submodule("bts", "The GPU bootstrap as stages (FIDESlib BootstrapStages.cuh).");

    py::class_<DeviceCt>(m, "DeviceCt")
        .def_property_readonly("level", [](const DeviceCt& d) { return d.get().getLevel(); })
        .def_property_readonly("noise_level", [](const DeviceCt& d) { return d.get().NoiseLevel; })
        .def_property("slots", [](const DeviceCt& d) { return d.get().slots; },
                      [](DeviceCt& d, int s) { d.get().slots = s; })
        .def_property_readonly("noise_factor", [](const DeviceCt& d) { return d.get().NoiseFactor; })
        .def("__repr__", [](const DeviceCt& d) {
            return "<DeviceCt limbs=" + std::to_string(d.get().getLevel() + 1) + " deg=" +
                   std::to_string(d.get().NoiseLevel) + " slots=" + std::to_string(d.get().slots) + ">";
        });

    py::class_<FC::BtsState>(m, "BtsState")
        .def_readonly("slots", &FC::BtsState::slots)
        .def_readonly("old_slots", &FC::BtsState::oldSlots)
        .def_readonly("prescaled", &FC::BtsState::prescaled)
        .def_readonly("is_lt", &FC::BtsState::isLT)
        .def_readonly("mixed_chain", &FC::BtsState::mixedChain)
        .def_readonly("sparse_encaps", &FC::BtsState::sparseEncaps)
        .def_readonly("sparse_b", &FC::BtsState::sparseB)
        .def_readonly("stc_first", &FC::BtsState::stcFirst)
        .def_readonly("stc_folded", &FC::BtsState::stcFolded)
        .def_readonly("exact_const", &FC::BtsState::exactConst)
        .def_readonly("aks", &FC::BtsState::aksOn)
        .def_readonly("shifted", &FC::BtsState::shiftedFlow)
        .def_readonly("correction", &FC::BtsState::correction)
        .def_readonly("cor_factor", &FC::BtsState::corFactor)
        .def_readonly("eval_const", &FC::BtsState::constantEvalMult)
        .def_readonly("n_cts", &FC::BtsState::nCtS)
        .def_readonly("n_stc", &FC::BtsState::nStC)
        .def("__repr__", [](const FC::BtsState& s) {
            return "<BtsState slots=" + std::to_string(s.slots) + " lt=" + std::to_string(s.isLT) +
                   " stc_first=" + std::to_string(s.stcFirst) + " aks=" + std::to_string(s.aksOn) +
                   " exact=" + std::to_string(s.exactConst) + " cts=" + std::to_string(s.nCtS) +
                   " stc=" + std::to_string(s.nStC) + ">";
        });

    // ---- handles
    m.def("device", &device_of, py::arg("fhe"), py::arg("ct"), kRelease,
          "The device ciphertext of a PackedCtx (loaded if needed); the view keeps the PackedCtx's storage alive.");
    m.def("clone", [](PackedCtx& p) { PackedCtx q = p; q.ct = p.ct->Clone(); return q; }, py::arg("ct"), kRelease,
          "A deep copy (device copy) of a PackedCtx.");
    m.def("set_slots", [](PackedCtx& p, int s) { p.ct->SetSlots((size_t)s); }, py::arg("ct"), py::arg("slots"),
          "Set the ciphertext's slot count (the bootstrap route is chosen from it).");
    m.def("mismatches", &mismatches, py::arg("a"), py::arg("b"), kRelease,
          "Residues that differ between two device ciphertexts (both polynomials, every limb); -1 = different shape.");
    m.def("sync", [] { cudaDeviceSynchronize(); }, kRelease);

    // ---- C++ references
    m.def("cpp_bootstrap", [](DeviceCt& d, int slots, bool prescaled) { FC::Bootstrap(d.get(), slots, prescaled); },
          py::arg("ct"), py::arg("slots"), py::arg("prescaled") = false, kRelease,
          "FIDESlib's own Bootstrap() on the device ciphertext (in place).");
    m.def("cpp_bootstrap_staged",
          [](DeviceCt& d, int slots, bool prescaled) { FC::BootstrapStaged(d.get(), slots, prescaled); },
          py::arg("ct"), py::arg("slots"), py::arg("prescaled") = false, kRelease,
          "The stage sequence, driven from C++ (in place).");

    // ---- stages (in place on the DeviceCt)
    m.def("begin", [](DeviceCt& d, int slots, bool prescaled, bool allow_stc_first) {
              return FC::btsBegin(d.get(), slots, prescaled, allow_stc_first);
          },
          py::arg("ct"), py::arg("slots"), py::arg("prescaled") = false, py::arg("allow_stc_first") = true, kRelease);
    m.def("stc_first_input", [](DeviceCt& d, const FC::BtsState& s) { FC::btsStcFirstInput(d.get(), s); },
          py::arg("ct"), py::arg("st"), kRelease);
    m.def("mod_raise", [](DeviceCt& d, FC::BtsState& s) { FC::btsModRaise(d.get(), s); }, py::arg("ct"),
          py::arg("st"), kRelease);
    m.def("fold", [](DeviceCt& d, const FC::BtsState& s) { FC::btsFold(d.get(), s); }, py::arg("ct"), py::arg("st"),
          kRelease);
    m.def("cts_stage", [](DeviceCt& d, const FC::BtsState& s, int k) { FC::btsCtSStage(d.get(), s, k); },
          py::arg("ct"), py::arg("st"), py::arg("k"), kRelease);
    m.def("eval_mod", [](DeviceCt& d, const FC::BtsState& s) { FC::btsEvalMod(d.get(), s); }, py::arg("ct"),
          py::arg("st"), kRelease);
    m.def("stc_first_output", [](DeviceCt& d, const FC::BtsState& s) { FC::btsStcFirstOutput(d.get(), s); },
          py::arg("ct"), py::arg("st"), kRelease);
    m.def("stc_enter", [](DeviceCt& d, const FC::BtsState& s) { FC::btsStCEnter(d.get(), s); }, py::arg("ct"),
          py::arg("st"), kRelease);
    m.def("stc_stage", [](DeviceCt& d, const FC::BtsState& s, int k) { FC::btsStCStage(d.get(), s, k); },
          py::arg("ct"), py::arg("st"), py::arg("k"), kRelease);
    m.def("finish", [](DeviceCt& d, const FC::BtsState& s) { FC::btsFinish(d.get(), s); }, py::arg("ct"),
          py::arg("st"), kRelease);

    // ---- SPRU for the s = 1 route (CKKS/Spru.cuh)
    static std::shared_ptr<FC::SpruKey> g_spru;
    static struct { long n = 0; FC::SpruTimes t; } g_spruAcc;
    m.def("spru_times", [] {
        py::dict r;
        const double n = g_spruAcc.n ? (double)g_spruAcc.n : 1.0;
        r["n"] = g_spruAcc.n;
        r["adjust"] = g_spruAcc.t.adjust_ms / n; r["switch"] = g_spruAcc.t.switch_ms / n; r["extract_fft"] = g_spruAcc.t.host_ms / n;
        r["scatter_ntt"] = g_spruAcc.t.encode_ms / n; r["extmult"] = g_spruAcc.t.extmult_ms / n; r["trace"] = g_spruAcc.t.trace_ms / n;
        r["product"] = g_spruAcc.t.product_ms / n; r["finish"] = g_spruAcc.t.finish_ms / n; r["total"] = g_spruAcc.t.total_ms / n;
        return r;
    }, "Mean SPRU phase times (ms) over the calls made with PERSEUS_BTS_PROFILE set.");
    m.def("set_real_payload",
          [](CKKSContext& fhe, bool on) { std::any_cast<FC::Context&>(fhe.cc->gpu)->setBtsRealPayload(on); },
          py::arg("fhe"), py::arg("on"),
          "Route the following dense bootstraps as real payloads (one EvalMod chain; needs FIDESLIB_BTS_REAL=1 at "
          "session build). The planned runtime sets it per site from the plan.");
    m.def("spru_setup",
          [](CKKSContext& fhe, int h) {
              auto& lcc = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(fhe.cc->cpu);
              auto lsk = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(fhe.keys.secretKey->pimpl);
              auto lpk = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(fhe.keys.publicKey->pimpl);
              lbcrypto::KeyPair<lbcrypto::DCRTPoly> kp(lpk, lsk);
              auto& g = std::any_cast<FC::Context&>(fhe.cc->gpu);
              g_spru = FC::spruSetup(lcc, kp, g, h);
          },
          py::arg("fhe"), py::arg("h") = 64, kRelease,
          "Build the SPRU key material for the s = 1 route (block key, switching key, 4 key ciphertexts).");
    m.def("spru",
          [](CKKSContext& fhe, DeviceCt& d, uint32_t correction) {
              if (!g_spru) throw std::runtime_error("[bts] spru_setup first");
              auto& lcc = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(fhe.cc->cpu);
              static const bool timed = [] { const char* e = std::getenv("PERSEUS_BTS_PROFILE"); return e && *e && *e != '0'; }();
              if (!timed) { FC::spruBootstrap(d.get(), *g_spru, correction, lcc, nullptr); return; }
              FC::SpruTimes t;
              FC::spruBootstrap(d.get(), *g_spru, correction, lcc, &t);
              g_spruAcc.n++;
              g_spruAcc.t.adjust_ms += t.adjust_ms; g_spruAcc.t.switch_ms += t.switch_ms; g_spruAcc.t.host_ms += t.host_ms;
              g_spruAcc.t.encode_ms += t.encode_ms; g_spruAcc.t.extmult_ms += t.extmult_ms; g_spruAcc.t.trace_ms += t.trace_ms;
              g_spruAcc.t.product_ms += t.product_ms; g_spruAcc.t.finish_ms += t.finish_ms; g_spruAcc.t.total_ms += t.total_ms;
          },
          py::arg("fhe"), py::arg("ct"), py::arg("correction"), kRelease,
          "SPRU bootstrap of a ciphertext whose slots all hold one complex value (in place).");

    // ---- override hook: route every FIDESlib Bootstrap() of the runtime to fn(ct: DeviceCt, slots, prescaled)
    m.def("set_override",
          [](py::object fn) {
              if (fn.is_none()) {
                  FC::g_bootstrapOverride = nullptr;
                  return;
              }
              auto keep = std::make_shared<py::object>(fn);
              FC::g_bootstrapOverride = [keep](FC::Ciphertext& ct, int slots, bool prescaled) {
                  py::gil_scoped_acquire gil;
                  DeviceCt d{&ct, nullptr};
                  (*keep)(d, slots, prescaled);
              };
          },
          py::arg("fn"),
          "Install fn(ct, slots, prescaled) as the runtime's bootstrap (None restores FIDESlib's own). fn runs with "
          "the GIL held; the DeviceCt it receives is valid only during the call.");
}
