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
#include <CKKS/Ccmm.cuh>
#include <CKKS/Pcmm.cuh>
#include <CKKS/ApproxModEval.cuh>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
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

// PcmmKeys is opaque in FIDESlib: a complete handle for pybind
struct PcmmKeysHandle {
    std::shared_ptr<FC::PcmmKeys> k;
};

struct CcmmKeysHandle {
    std::shared_ptr<FC::CcmmKeys> k;
};

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

    // Coefficient residues of a device ciphertext and back. ContextData's layout differs in this TU (NCCL members):
    // moduli come from the OpenFHE parameters, N from the data.
    auto moduli_of = [](CKKSContext& fhe, int n) {
        const auto& lcc = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(fhe.cc->cpu);
        const auto& ps = lcc->GetCryptoParameters()->GetElementParams()->GetParams();
        std::vector<uint64_t> mod(n);
        for (int i = 0; i < n; ++i) mod[i] = ps.at(i)->GetModulus().ConvertToInt<uint64_t>();
        return mod;
    };
    m.def("export_limbs", [moduli_of](CKKSContext& fhe, DeviceCt& d) {
        auto& x = d.get();
        cudaDeviceSynchronize();
        std::vector<std::vector<uint64_t>> v[2];
        const int L = x.getLevel();
        for (int part = 0; part < 2; ++part) {
            FC::RNSPoly m(x.cc, L);
            m.copy(part ? x.c1 : x.c0);
            m.INTT(1, true);
            cudaDeviceSynchronize();
            m.store(v[part]);
        }
        cudaDeviceSynchronize();
        const int nl = (int)v[0].size();
        if ((int)v[1].size() != nl) throw std::runtime_error("export_limbs: c0 / c1 limb counts differ");
        const int N = nl ? (int)v[0][0].size() : 0;
        const std::vector<uint64_t> mod = moduli_of(fhe, nl);
        py::dict out;
        for (int part = 0; part < 2; ++part) {
            py::array_t<uint64_t> a({(py::ssize_t)nl, (py::ssize_t)N});
            auto w = a.mutable_unchecked<2>();
            for (int i = 0; i < nl; ++i) {
                if ((int)v[part][i].size() != N) throw std::runtime_error("export_limbs: ragged limbs");
                for (int k = 0; k < N; ++k) w(i, k) = v[part][i][k];
            }
            out[part ? "c1" : "c0"] = a;
        }
        out["moduli"] = mod;
        out["level"] = x.getLevel();
        out["noise_level"] = x.NoiseLevel;
        out["noise_factor"] = x.NoiseFactor;
        return out;
    }, py::arg("fhe"), py::arg("ct"), "c0 / c1 coefficient residues (limbs, N) uint64, moduli, level, noise level / factor.");
    m.def("import_limbs", [moduli_of](CKKSContext& fhe, DeviceCt& d, py::array_t<uint64_t, py::array::c_style> c0,
                             py::array_t<uint64_t, py::array::c_style> c1, double noise_factor, int noise_level) {
        auto& x = d.get();
        const int nl = x.getLevel() + 1;
        const py::ssize_t N = c0.ndim() == 2 ? c0.shape(1) : 0;
        if (c0.ndim() != 2 || c0.shape(0) != nl || c1.ndim() != 2 || c1.shape(0) != nl || c1.shape(1) != N)
            throw std::invalid_argument("import_limbs: c0 / c1 must be (limbs at the ciphertext's level, N)");
        const std::vector<uint64_t> mod = moduli_of(fhe, nl);
        for (int part = 0; part < 2; ++part) {
            auto r = (part ? c1 : c0).unchecked<2>();
            std::vector<std::vector<uint64_t>> v(nl, std::vector<uint64_t>(N));
            for (int i = 0; i < nl; ++i)
                for (py::ssize_t k = 0; k < N; ++k) v[i][k] = r(i, k);
            FC::RNSPoly m(x.cc, nl - 1);
            m.load(v, mod);
            m.NTT(1, true);
            cudaDeviceSynchronize();
            (part ? x.c1 : x.c0).copy(m);
        }
        cudaDeviceSynchronize();
        x.NoiseFactor = noise_factor;
        x.NoiseLevel = noise_level;
    }, py::arg("fhe"), py::arg("ct"), py::arg("c0"), py::arg("c1"), py::arg("noise_factor"), py::arg("noise_level"),
       "Overwrite a device ciphertext (at its current level) from coefficient residues (export_limbs' inverse).");
    py::class_<PcmmKeysHandle>(m, "PcmmKeys")
        .def_property_readonly("channels", [](const PcmmKeysHandle& k) { return FC::pcmmChannels(*k.k); });
    m.def("pcmm_keys", [](py::dict d) {
        FC::PcmmKeySource src;
        auto u32 = [](py::handle h) {
            auto a = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>::ensure(h);
            if (!a) throw std::invalid_argument("pcmm_keys: expected a uint32 array");
            return std::vector<uint32_t>(a.data(), a.data() + a.size());
        };
        auto prim = u32(d["primes"]);
        auto split = u32(d["split"]);
        if (prim.size() != 3 || split.size() != 4) throw std::invalid_argument("pcmm_keys: primes (3) / split (4)");
        src.q[0] = prim[0]; src.q[1] = prim[1]; src.P = prim[2];
        for (int i = 0; i < 4; ++i) src.split[i] = split[i];
        auto ni16 = u32(d["ninv16"]), ni12 = u32(d["ninv12"]);
        for (int i = 0; i < 3; ++i) {
            src.psi16[i] = u32(py::list(d["psi16"])[i]);
            src.ipsi16[i] = u32(py::list(d["ipsi16"])[i]);
            src.psi12[i] = u32(py::list(d["psi12"])[i]);
            src.ipsi12[i] = u32(py::list(d["ipsi12"])[i]);
            src.ninv16[i] = ni16[i]; src.ninv12[i] = ni12[i];
        }
        src.fwd = u32(d["fwd"]); src.back = u32(d["back"]); src.pack = u32(d["pack"]);
        return PcmmKeysHandle{FC::pcmmUploadKeys(src)};
    }, py::arg("keys"), "Upload the PCMM keys (Pcmm.cuh PcmmKeySource as a dict).");
    m.def("pcmm_multi", [](std::vector<DeviceCt> ins, std::vector<DeviceCt> outs, PcmmKeysHandle& k,
                           py::array_t<int16_t, py::array::c_style | py::array::forcecast> W, double noise_factor) {
        const py::ssize_t c = FC::pcmmChannels(*k.k);
        if (W.ndim() != 2 || W.shape(0) != c * (py::ssize_t)outs.size() || W.shape(1) != c * (py::ssize_t)ins.size())
            throw std::invalid_argument("pcmm_multi: W must be (nch len(outs), nch len(ins)) int16");
        std::vector<FC::Ciphertext*> pin, pout;
        for (auto& d : ins) pin.push_back(&d.get());
        for (auto& d : outs) pout.push_back(&d.get());
        std::vector<double> t(6, 0.0);
        {
            py::gil_scoped_release nogil;
            FC::pcmmMulti(pin, pout, *k.k, W.data(), t.data());
        }
        for (auto* c : pout) c->NoiseFactor = noise_factor;
        return t;
    }, py::arg("ins"), py::arg("outs"), py::arg("keys"), py::arg("W"), py::arg("noise_factor"),
       "Y = W X at the split point (Pcmm.cuh pcmmMulti); returns per-step times (ms).");
    m.def("pcmm_multi_planes", [](std::vector<DeviceCt> ins, std::vector<DeviceCt> outs, PcmmKeysHandle& k,
                                  py::array_t<int8_t, py::array::c_style | py::array::forcecast> Wd,
                                  double noise_factor) {
        const py::ssize_t c = FC::pcmmChannels(*k.k);
        if (Wd.ndim() != 3 || Wd.shape(1) != c * (py::ssize_t)outs.size() || Wd.shape(2) != c * (py::ssize_t)ins.size())
            throw std::invalid_argument("pcmm_multi_planes: Wd must be (nW, nch len(outs), nch len(ins)) int8");
        std::vector<FC::Ciphertext*> pin, pout;
        for (auto& d : ins) pin.push_back(&d.get());
        for (auto& d : outs) pout.push_back(&d.get());
        std::vector<double> t(6, 0.0);
        {
            py::gil_scoped_release nogil;
            FC::pcmmMultiPlanes(pin, pout, *k.k, Wd.data(), (int)Wd.shape(0), t.data());
        }
        for (auto* c : pout) c->NoiseFactor = noise_factor;
        return t;
    }, py::arg("ins"), py::arg("outs"), py::arg("keys"), py::arg("Wd"), py::arg("noise_factor"),
       "pcmm_multi with W as int8 digit planes (W = sum_u Wd[u] 256^u).");
    py::class_<CcmmKeysHandle>(m, "CcmmKeys");
    m.def("ccmm_keys", [](py::dict d) {
        FC::CcmmKeySource src;
        auto u32 = [](py::handle h) {
            auto a = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>::ensure(h);
            if (!a) throw std::invalid_argument("ccmm_keys: expected a uint32 array");
            return std::vector<uint32_t>(a.data(), a.data() + a.size());
        };
        auto prim = u32(d["primes"]);
        if (prim.size() != 5) throw std::invalid_argument("ccmm_keys: primes (5)");
        for (int i = 0; i < 5; ++i) src.p[i] = prim[i];
        const char* names[3] = {"16", "13", "6"};
        for (int s = 0; s < 3; ++s) {
            auto ni = u32(d[(std::string("ninv") + names[s]).c_str()]);
            for (int i = 0; i < 5; ++i) {
                src.psi[s][i] = u32(py::list(d[(std::string("psi") + names[s]).c_str()])[i]);
                src.ipsi[s][i] = u32(py::list(d[(std::string("ipsi") + names[s]).c_str()])[i]);
                src.ninv[s][i] = ni[i];
            }
        }
        src.fwd = u32(d["fwd"]); src.back = u32(d["back"]); src.autos = u32(d["autos"]); src.rlk = u32(d["rlk"]);
        return CcmmKeysHandle{FC::ccmmUploadKeys(src)};
    }, py::arg("keys"), "Upload the CCMM keys (Ccmm.cuh CcmmKeySource as a dict).");
    m.def("ccmm_transpose", [](std::vector<DeviceCt> ins, std::vector<DeviceCt> outs, CcmmKeysHandle& k) {
        std::vector<FC::Ciphertext*> pin, pout;
        for (auto& d : ins) pin.push_back(&d.get());
        for (auto& d : outs) pout.push_back(&d.get());
        std::vector<double> t(5, 0.0);
        {
            py::gil_scoped_release nogil;
            FC::ccmmTranspose(pin, pout, *k.k, t.data());
        }
        return t;
    }, py::arg("ins"), py::arg("outs"), py::arg("keys"), "Column-wise M (16 SinC cts) -> M^T; per-step times (ms).");
    m.def("ccmm_mul", [](std::vector<DeviceCt> a, std::vector<DeviceCt> b, std::vector<DeviceCt> outs,
                         CcmmKeysHandle& k, bool b_transposed) {
        std::vector<FC::Ciphertext*> pa, pb, pout;
        for (auto& d : a) pa.push_back(&d.get());
        for (auto& d : b) pb.push_back(&d.get());
        for (auto& d : outs) pout.push_back(&d.get());
        std::vector<double> t(5, 0.0);
        {
            py::gil_scoped_release nogil;
            FC::ccmmMul(pa, pb, pout, *k.k, t.data(), b_transposed);
        }
        return t;
    }, py::arg("m"), py::arg("mp"), py::arg("outs"), py::arg("keys"), py::arg("b_transposed") = false,
       "Batch CCMM M M' (CKL Alg. 4) on SinC column encryptions (mp = M'^T when b_transposed); per-step times (ms).");
    m.def("set_scale", [](DeviceCt& d, double nf, int noise_level) {
              d.get().NoiseFactor = nf;
              d.get().NoiseLevel = noise_level;
          },
          py::arg("ct"), py::arg("noise_factor"), py::arg("noise_level"),
          "Restamp a device ciphertext's NoiseFactor and NoiseLevel (no data change).");
    m.def("noise_factor", [](DeviceCt& d) { return d.get().NoiseFactor; }, py::arg("ct"),
          "NoiseFactor of a device ciphertext.");
    m.def("scale_noise_factor", [](DeviceCt& d, double f) { d.get().NoiseFactor *= f; }, py::arg("ct"),
          py::arg("factor"), "NoiseFactor *= factor.");
    m.def("mult_int", [](DeviceCt& d, uint64_t k) { FC::multIntScalar(d.get(), k); }, py::arg("ct"), py::arg("k"),
          kRelease, "Exact product by a positive integer (no level).");
    m.def("secret_key_coeffs", [](CKKSContext& fhe) {
        const auto& lsk = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(fhe.keys.secretKey->pimpl);
        auto sk = lsk->GetPrivateElement();
        sk.SetFormat(Format::COEFFICIENT);
        const auto& e0 = sk.GetElementAtIndex(0);
        const uint64_t q = e0.GetModulus().ConvertToInt<uint64_t>();
        const size_t n = e0.GetLength();
        py::array_t<int8_t> a((py::ssize_t)n);
        auto w = a.mutable_unchecked<1>();
        for (size_t k = 0; k < n; ++k) {
            const uint64_t v = e0[k].ConvertToInt<uint64_t>();
            w(k) = v == 0 ? 0 : (v == 1 ? 1 : (v == q - 1 ? -1 : 99));
        }
        return a;
    }, py::arg("fhe"), "Secret key coefficients (int8; 99 = non-ternary), for client-side PCMM keygen.");

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
    m.def("stc_first_entry", [](DeviceCt& d, int slots) { return FC::btsStcFirstEntry(d.get(), slots); },
          py::arg("ct"), py::arg("slots"), "StC-first input limb index (-1: none).");
    m.def("stc_first_input", [](DeviceCt& d, const FC::BtsState& s, bool own_cf) { FC::btsStcFirstInput(d.get(), s, own_cf); },
          py::arg("ct"), py::arg("st"), py::arg("own_cf") = false, kRelease);
    m.def("stc_high", [](DeviceCt& d, int slots, int out_limb, double factor, int skip_mask, double nf_ref) {
              FC::btsStcHigh(d.get(), slots, out_limb, factor, skip_mask, nf_ref);
          },
          py::arg("ct"), py::arg("slots"), py::arg("out_limb"), py::arg("factor") = 1.0, py::arg("skip_mask") = 0,
          py::arg("nf_ref") = 0.0, kRelease,
          "Slot -> coefficient transform outside a bootstrap, rescaled output at limb index out_limb.");
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
    m.def("set_correction_factor",
          [](CKKSContext& fhe, int cf) { std::any_cast<FC::Context&>(fhe.cc->gpu)->setCorrectionFactorOverride(cf); },
          py::arg("fhe"), py::arg("cf"),
          "Bootstrap the following ciphertexts with correction factor `cf` (-1 = the route's own). The planned "
          "runtime sets it per site from the plan.");
    m.def("set_raise_drop",
          [](CKKSContext& fhe, int drop) { std::any_cast<FC::Context&>(fhe.cc->gpu)->setBtsRaiseDrop(drop); },
          py::arg("fhe"), py::arg("drop"),
          "Raise the following bootstraps `drop` composite levels below the top (a variant FIDESLIB_BTS_RAISE_DROPS "
          "built at session start; 0 = the top). The planned runtime sets it per site from the plan.");
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
