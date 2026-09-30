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
#include "packing/pack_tag.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "residency_pipeline.h"
#include <future>
#include <chrono>
#include <tuple>

namespace py = pybind11;

namespace {

constexpr auto kRelease = py::call_guard<py::gil_scoped_release>();

// `with inf.step("label"):` — mirrors the C++ WithStep scoping run_ops applies per op.
struct StepScope {
    Inference* inf;
    std::string label;
};

// `with fhe.suppress_auto_bts():` — the runtime's AutoBtsSuppressScope (no reactive refresh
// inside; the fused reductions square at the ceiling and refresh inside the fold instead).
struct AutoBtsScope {
    CKKSContext* fhe;
    bool saved = false;
};

// The K/V cache residency of the Python implementation: ciphertexts parked in the runtime's
// pinned KV arena between the blocks of a token (KvStoreStaged / KvLoadStaged on one side
// stream, KvEvict once the store has landed), the C++ decode arm's offload_block_kv.
cudaStream_t impl_kv_stream() {
    static cudaStream_t s = [] {
        cudaStream_t st = nullptr;
        cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
        return st;
    }();
    return s;
}

// A batch of plaintexts encoded on the runtime's mask worker (residency_pipeline.h:
// mask_submit) while the main thread keeps issuing GPU work, adopted into inf.enc_cache
// afterwards -- the C++ decode arm's next-step mask staging (gpt2_decode.cu:138-150).
struct StagedPts {
    std::future<void> fut;
    std::vector<std::tuple<std::string, uint32_t, Ptx>> out;   // written by the worker only
    bool adopted = false;
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


// The named slot plaintext `name` at the level of `ct`, re-encoded (host side) and re-uploaded
// when the level differs — the weights_at contract without its weight_store canon pointer,
// which the block loader leaves dangling once a block state dies.
Ptx& slot_pt_at(Inference& inf, const std::string& name, const PackedCtx& ct) {
    auto it = inf.w.find(name);
    if (it == inf.w.end() || it->second.empty())
        throw std::out_of_range("slot plaintext '" + name + "' is not set (set_slot_pt)");
    Ptx& pt = it->second[0];
    // `encode_like_params` is chain-dependent: on a d=1 chain a plaintext meets the ct at the
    // ct's OWN level and degree, so adding it leaves a pending rescale pending; only on a d>1
    // chain does it go to level+pending at degree 1. Hardcoding the latter forced a realize
    // the composites do not do, which is level 17/deg 2 against 18/deg 1 on n64.
    const auto [lv, nsd] = inf.encode_like_params(ct.ct);
    // Level only: a plaintext does not carry its noiseScaleDeg, and it does not have to. The
    // caller names one slot per (level, degree) pair -- the composites likewise KEY their
    // encode cache by both and never query the plaintext -- so a given name holds exactly one
    // degree and a level match means the pair matches.
    if (static_cast<uint32_t>(pt->GetLevel()) != lv) {
        Ptx adapted = inf.fhe->complex_payload
            ? inf.cc()->MakeCKKSPackedPlaintext(pt->GetCKKSPackedValue(), nsd, lv)
            : inf.cc()->MakeCKKSPackedPlaintext(pt->GetRealPackedValue(), nsd, lv);
        inf.evict_plaintext(pt);
        pt = adapted;
        ++inf.fhe->weight_relevel_count;
    }
    inf.load_plaintext(pt);
    return pt;
}
// Encode a real or complex slot vector at ct's level (+ pending rescale), pack-tagged unless
// `tagged` is false (the C++ CutMax leaves its complex masks untagged: no sparse routing).
Ptx encode_any_at(Inference& inf, const perseus_np::AnyVec& v, const PackedCtx& ct, bool tagged = true) {
    const auto [lv, nsd] = inf.encode_like_params(ct.ct);
    if (v.cplx) return tagged ? inf.encode_tagged(v.c, lv, nsd)
                              : inf.cc()->MakeCKKSPackedPlaintext(v.c, nsd, lv);
    return tagged ? inf.encode_tagged(v.re, lv, nsd)
                  : inf.cc()->MakeCKKSPackedPlaintext(v.re, nsd, lv);
}

Ptx encode_any(Inference& inf, const perseus_np::AnyVec& v, uint32_t lv, bool tagged = true,
               uint32_t nsd = 1) {
    if (v.cplx) return tagged ? inf.encode_tagged(v.c, lv, nsd)
                              : inf.cc()->MakeCKKSPackedPlaintext(v.c, nsd, lv);
    return tagged ? inf.encode_tagged(v.re, lv, nsd)
                  : inf.cc()->MakeCKKSPackedPlaintext(v.re, nsd, lv);
}

// inf.enc_cache lookup keyed like encode_at_cached (tag + level), encoding on a miss.
Ptx cached_any_at(Inference& inf, const std::string& tag, const perseus_np::AnyVec& v,
                  const PackedCtx& ct, bool tagged) {
    const auto [lv, nsd] = inf.encode_like_params(ct.ct);
    const std::string key = Inference::enc_cache_key(tag, lv, nsd);
    auto it = inf.enc_cache.find(key);
    if (it != inf.enc_cache.end()) { ++inf.enc_cache_hit; return it->second; }
    ++inf.enc_cache_miss;
    if (inf.strict_masks) {
        ++inf.mask_strict_miss;
        throw fhe::MaskError("[mask_miss] strict planned-mask cache miss: '" + key + "'");
    }
    Ptx pt = encode_any(inf, v, lv, tagged);
    inf.enc_cache.emplace(key, pt);
    return pt;
}
}  // namespace

void bind_ops(py::module_& m) {
    // ── context ────────────────────────────────────────────────────────────
    py::class_<AutoBtsScope>(m, "AutoBtsScope")
        .def("__enter__", [](AutoBtsScope& s) {
            s.saved = s.fhe->auto_bts_suppressed; s.fhe->auto_bts_suppressed = true; return &s;
        }, py::return_value_policy::reference)
        .def("__exit__", [](AutoBtsScope& s, py::object, py::object, py::object) {
            s.fhe->auto_bts_suppressed = s.saved; return false;
        });
    py::class_<StagedPts, std::shared_ptr<StagedPts>>(m, "StagedPts",
        "A stage_pts job: plaintexts being encoded on the mask worker, adopted with adopt_pts.")
        .def_property_readonly("ready", [](const StagedPts& s) {
            return !s.fut.valid() || s.fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        })
        .def_property_readonly("adopted", [](const StagedPts& s) { return s.adopted; });
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
        .def("bootstrap",
             [](CKKSContext& c, PackedCtx& p, int iterations) {
                 if (iterations <= 1) { c.bootstrap(p.ct); return; }
                 CKKSContext::BtsItersScope scope(c, static_cast<uint32_t>(iterations));
                 c.bootstrap(p.ct);
             },
             py::arg("ct"), py::arg("iterations") = 1, kRelease,
             "Refresh `ct` to the bootstrap output level (in place). iterations=2 runs the "
             "runtime's two-iteration bootstrap (eval_bootstrap_iter: bootstrap the residual "
             "scaled by 2^bts_precision and add it back) as ONE recorded op, the way the C++ "
             "CutMax refreshes under its BtsItersScope.")
        .def("maybe_bootstrap", [](CKKSContext& c, PackedCtx& p) { c.maybe_bootstrap(p.ct); },
             py::arg("ct"), kRelease, "Bootstrap `ct` only if its level is below the auto threshold.")
        .def("bootstrap_hint",
             [](CKKSContext& c, PackedCtx& p, int thr, bool acct, int iterations) {
                 if (iterations <= 1) { c.bootstrap_hint(p, thr, acct); return; }
                 CKKSContext::BtsItersScope scope(c, static_cast<uint32_t>(iterations));
                 c.bootstrap_hint(p, thr, acct);
             },
             py::arg("ct"), py::arg("level_threshold"), py::arg("account_pending_rescale") = false,
             py::arg("iterations") = 1,
             "Bootstrap `ct` if its level exceeds `level_threshold`; iterations=2 as for bootstrap.")
        .def("level_hint", static_cast<void (CKKSContext::*)(PackedCtx&, int)>(&CKKSContext::level_hint),
             py::arg("ct"), py::arg("level"), "Drop `ct` to `level` if it is above it.")
        // ── leaf primitives for Python-authored ops ────────────────────────────
        .def("rotate", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&, int32_t)>(&CKKSContext::rotate),
             py::arg("ct"), py::arg("steps"), kRelease,
             "Cyclic slot rotation by `steps` (out[i] = in[i + steps]); the rotation key for "
             "`steps` must exist in the session's band.")
        .def("rotate_hoisted",
             [](CKKSContext& c, const PackedCtx& ct, const std::vector<int32_t>& steps) {
                 return c.rotate_hoisted(ct, steps);
             },
             py::arg("ct"), py::arg("steps"), kRelease,
             "Several rotations of one ciphertext sharing one key-switch decomposition (the "
             "runtime's hoisted rotation); one PackedCtx per step, recorded like plain rotations.")
        .def("rotate_and_sum",
             [](CKKSContext& c, const PackedCtx& ct, int32_t start, int32_t stop) {
                 if (start == 0) throw std::invalid_argument("rotate_and_sum: start must be non-zero");
                 if (stop <= 0) throw std::invalid_argument("rotate_and_sum: stop must be positive");
                 py::gil_scoped_release nogil;
                 const int32_t sign = start < 0 ? -1 : 1;
                 int32_t gap = start < 0 ? -start : start;
                 if (gap >= stop) return c.add(ct, 0.0);        // empty ladder: a clone
                 PackedCtx acc = c.add(ct, c.rotate(ct, sign * gap));
                 for (gap *= 2; gap < stop; gap *= 2)
                     c.inplace_add(acc, c.rotate(acc, sign * gap));
                 return acc;
             },
             py::arg("ct"), py::arg("start"), py::arg("stop"),
             "The rotate-and-sum ladder: x += rotate(x, g) for g = start, 2 start, 4 start, ... "
             "while |g| < stop (start < 0 rotates the other way). start=1, stop=slots is the "
             "all-slots total broadcast to every slot (rotate_and_sum_all); start=t sums the "
             "slots congruent mod t; start=-1, stop=t replicates slot 0 of each t-group "
             "rightwards. Issued in C++ (one Python call instead of 2 log2(stop/start)), "
             "recorded as the same rotate + add stream a Python loop would emit, so plans "
             "captured either way stay valid. Needs the rotation keys for every gap.")
        // ── folds: a sparse bootstrap that finishes a rotate-and-sum ladder ──────────────
        .def("fold_slots_for",
             [](const CKKSContext& c, uint32_t s_wanted) { return c.fold_slots_for(s_wanted); },
             py::arg("s_wanted"),
             "The slot count a fold wanting `s_wanted` can run at: `s_wanted` when a sparse "
             "bootstrap precomputation exists for it (SPARSE_BTS_SLOTS), else the smallest built "
             "count above it (pre-ladder the gap with rotate_and_sum(ct, s_wanted, s)), else the "
             "full slot count (a plain bootstrap; the caller ladders everything).")
        .def("fold_bootstrap",
             [](CKKSContext& c, PackedCtx& ct, uint32_t s, int n_live, double prescale) {
                 py::gil_scoped_release nogil;
                 c.fold_bootstrap(ct.ct, s, n_live, prescale);
             },
             py::arg("ct"), py::arg("s"), py::arg("n_live"), py::arg("prescale") = 1.0,
             "In place: bootstrap `ct` at `s` slots, folding the slots/s copies of every "
             "residue class into one and scaling by 1/n_live, so a ladder stopped at stride `s` "
             "comes out as the full class sum divided by n_live (the biased mean over n_live "
             "entries when the class holds them). `prescale` multiplies the input first (one "
             "level) and is undone in the recovery; use it to land a large sum inside the "
             "bootstrap's range. One recorded deliberate refresh (plannable).")
        .def("suppress_auto_bts",
             [](CKKSContext& c) { return AutoBtsScope{&c}; }, py::keep_alive<0, 1>(),
             "`with fhe.suppress_auto_bts():` no reactive refresh fires inside (the runtime's "
             "AutoBtsSuppressScope): a product may sit at the ceiling when a fold follows.")
        .def("ensure_const_one", [](CKKSContext& c, const PackedCtx& src, int slots) {
                 py::gil_scoped_release nogil; c.ensure_const_one(src.ct, slots);
             }, py::arg("src"), py::arg("slots"),
             "Seed the runtime's constant-one ciphertext from `src` (first call wins): the C++ "
             "decode does this from the freshest input, and its CutMax cascades start from it.")
        .def("const_one_available", &CKKSContext::const_one_available)
        .def("mult_i", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&)>(&CKKSContext::mult_i),
             py::arg("ct"), kRelease,
             "i * ct: the monomial x^(N/2), a level-free integer rotation of the complex plane "
             "(no rescale, no key switch). With conjugate it isolates the lanes of a complex "
             "payload: a = (z + conj z)/2, i b = (z - conj z)/2.")
        .def("mult_add_many",
             [](CKKSContext& c, PackedCtx& acc, const std::vector<PackedCtx>& vs,
                const std::vector<PackedCtx>& ss) {
                 if (vs.size() != ss.size())
                     throw std::invalid_argument("mult_add_many: vs and ss differ in length");
                 py::gil_scoped_release nogil;
                 std::vector<const PackedCtx*> vp, sp;
                 vp.reserve(vs.size()); sp.reserve(ss.size());
                 for (auto& v : vs) vp.push_back(&v);
                 for (auto& t : ss) sp.push_back(&t);
                 if (!vs.empty() && c.mult_add_many_usable(acc, vp, sp)) {
                     c.mult_add_many(acc, vp, sp);
                     return true;
                 }
                 for (size_t i = 0; i < vs.size(); ++i) c.inplace_add(acc, c.mult(vs[i], ss[i]));
                 return false;
             },
             py::arg("acc"), py::arg("vs"), py::arg("ss"),
             "acc += sum_j vs[j] * ss[j] with ONE relinearization when the runtime's fused lane "
             "accumulate applies (returns True), else the plain loop; the recorded op stream is "
             "the loop's either way.")
        .def("conjugate", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&)>(&CKKSContext::conjugate),
             py::arg("ct"), kRelease, "Complex conjugate of every slot (identity on real payloads).")
        .def("negate", static_cast<PackedCtx (CKKSContext::*)(const PackedCtx&)>(&CKKSContext::negate),
             py::arg("ct"), kRelease, "-ct.")
        .def_property_readonly("complex_payload",
                               [](const CKKSContext& c) { return c.complex_payload; })
        .def("bootstrap_output_level",
             [](CKKSContext& c) { return static_cast<int>(c.bootstrap_output_level()); })
        .def_property_readonly("total_bootstraps",
                               [](const CKKSContext& c) { return static_cast<long long>(c.total_bootstraps); },
                               "Bootstraps run so far in this context (deliberate + reactive + iterations).")
        .def_property_readonly("weight_relevel_count",
                               [](const CKKSContext& c) { return static_cast<long long>(c.weight_relevel_count); })
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
        .def_readwrite("bidirectional", &Inference::bidirectional)
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
             [](Inference& inf, const PackedCtx& ct, const py::array& values) {
                 auto v = perseus_np::to_any(values);
                 py::gil_scoped_release nogil;
                 v.resize(static_cast<size_t>(inf.slots));
                 Ptx pt = encode_any_at(inf, v, ct);
                 return inf.fhe->mult(ct, pt);
             },
             py::arg("ct"), py::arg("values"),
             "ct * values (slot-wise): `values` is encoded as a plaintext at ct's level; "
             "shorter than the slot count is zero-filled; a complex array encodes a complex "
             "plaintext.")
        .def("mult_const",
             [](Inference& inf, const PackedCtx& ct, double re, double im) {
                 py::gil_scoped_release nogil;
                 Ptx pt = inf.encode_complex_const_at(re, im, ct);
                 return inf.fhe->mult(ct, pt);
             },
             py::arg("ct"), py::arg("re"), py::arg("im"),
             "ct * (re + i im) as a plaintext product (the runtime's cached complex constant; "
             "a level like any mask). mult_const(ct, 0, 1) packs: a + i b = add(a, mult_const(b, 0, 1)).")
        .def("add_pt",
             [](Inference& inf, const PackedCtx& ct, const py::array& values) {
                 auto v = perseus_np::to_any(values);
                 py::gil_scoped_release nogil;
                 v.resize(static_cast<size_t>(inf.slots));
                 Ptx pt = encode_any_at(inf, v, ct);
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
        // The (level, noiseScaleDeg) a plaintext must carry to meet `ct` without forcing a
        // rescale -- chain-dependent, so the port asks rather than assuming (inference.h).
        .def("encode_like_level",
             [](Inference& inf, const PackedCtx& ct) {
                 const auto [lv, nsd] = inf.encode_like_params(ct.ct);
                 return py::make_tuple(lv, nsd);
             }, py::arg("ct"),
             "(level, noise_deg) a plaintext needs to meet this ciphertext as the composites "
             "encode it: the ct's own level and degree on a d=1 chain, level+pending at degree "
             "1 on a composite chain.")
        // ── named slot-vector plaintexts (encode once, reuse; the port's weight path) ─────
        .def("set_slot_pt",
             [](Inference& inf, const std::string& name, const py::array& values, int level,
                int deg) {
                 auto v = perseus_np::to_any(values);
                 if (static_cast<int>(v.size()) > inf.slots)
                     throw std::invalid_argument("set_slot_pt: more values than slots");
                 v.resize(static_cast<size_t>(inf.slots));
                 py::gil_scoped_release nogil;
                 const uint32_t lv = level <= 0 ? inf.fhe->bootstrap_output_level()
                                                : static_cast<uint32_t>(level);
                 inf.w[name] = {encode_any(inf, v, lv, true, static_cast<uint32_t>(deg))};
             },
             py::arg("name"), py::arg("values"), py::arg("level") = 0, py::arg("deg") = 1,
             "Encode a slot vector ONCE as the plaintext `name` at `level` (<= 0: the bootstrap "
             "output level), host-resident until loaded. mult_slot_pt / add_slot_pt re-level it "
             "on a level mismatch (weights_at), so pass the level it will be used at. A complex "
             "array (dtype kind 'c') encodes a complex plaintext (complex payload sessions).")
        .def("has_slot_pt", [](const Inference& inf, const std::string& name) {
                 return inf.w.find(name) != inf.w.end();
             }, py::arg("name"))
        .def("mult_slot_pt",
             [](Inference& inf, const PackedCtx& ct, const std::string& name) {
                 py::gil_scoped_release nogil;
                 Ptx& pt = slot_pt_at(inf, name, ct);
                 return inf.fhe->mult(ct, pt);
             },
             py::arg("ct"), py::arg("name"), "ct * the named slot plaintext (uploaded on first use).")
        .def("add_slot_pt",
             [](Inference& inf, const PackedCtx& ct, const std::string& name) {
                 py::gil_scoped_release nogil;
                 Ptx& pt = slot_pt_at(inf, name, ct);
                 return inf.fhe->add(ct, pt);
             },
             py::arg("ct"), py::arg("name"), "ct + the named slot plaintext.")
        .def("mult_slot_pt_many",
             [](Inference& inf, const PackedCtx& ct, const std::vector<std::string>& names) {
                 py::gil_scoped_release nogil;
                 std::vector<Ptx> pts;
                 pts.reserve(names.size());
                 for (const auto& n : names) pts.push_back(slot_pt_at(inf, n, ct));
                 std::vector<PackedCtx> out;
                 out.reserve(names.size());
                 if (!pts.empty() && inf.fhe->lane_batch_usable(ct.ct, pts)) {
                     auto raw = inf.fhe->mult_batch_exec(ct.ct, pts);
                     for (size_t i = 0; i < pts.size(); ++i)
                         out.push_back(inf.fhe->mult_finish(ct, pts[i], std::move(raw[i])));
                 } else {
                     for (auto& pt : pts) out.push_back(inf.fhe->mult(ct, pt));
                 }
                 return out;
             },
             py::arg("ct"), py::arg("names"),
             "ct * each named slot plaintext, as one fused batch of lane products when the "
             "runtime allows (the C++ V-push path), else one product each.")
        // ── slot-period stamps (what the C++ reductions do with packtag::t_reduce_*) ─────
        .def("tag_reduce",
             [](Inference& inf, PackedCtx& ct, int stride) {
                 const auto top = packtag::PackTag::top(inf.slots);
                 ct.tag = stride <= 1 ? packtag::t_reduce_all(top)
                                      : packtag::t_reduce_stride(top, stride);
                 inf.fhe->tag_ct(ct.ct, ct.tag);
             },
             py::arg("ct"), py::arg("stride"),
             "Stamp `ct` as the output of a rotate-and-sum reduction: constant over the slots "
             "(stride <= 1, e.g. an all-slot sum) or stride-periodic (a mod-stride class sum). "
             "The planner routes a refresh of such a ciphertext through a sparse bootstrap.")
        .def("load_slot_pts",
             [](Inference& inf, const std::string& prefix) {
                 py::gil_scoped_release nogil;
                 size_t n = 0;
                 for (auto& kv : inf.w)
                     if (kv.first.compare(0, prefix.size(), prefix) == 0)
                         for (auto& pt : kv.second) { inf.load_plaintext(pt); ++n; }
                 return n;
             },
             py::arg("prefix"), "Upload every slot plaintext whose name starts with `prefix`.")
        .def("evict_slot_pts",
             [](Inference& inf, const std::string& prefix) {
                 py::gil_scoped_release nogil;
                 size_t n = 0;
                 for (auto& kv : inf.w)
                     if (kv.first.compare(0, prefix.size(), prefix) == 0)
                         for (auto& pt : kv.second) { inf.evict_plaintext(pt); ++n; }
                 return n;
             },
             py::arg("prefix"), "Free the device copies of the slot plaintexts named `prefix*` (host copies stay).")
        .def("drop_slot_pts",
             [](Inference& inf, const std::string& prefix) {
                 py::gil_scoped_release nogil;
                 std::vector<std::string> keys;
                 for (auto& kv : inf.w)
                     if (kv.first.compare(0, prefix.size(), prefix) == 0) keys.push_back(kv.first);
                 for (auto& k : keys) inf.evict_weights(k);
                 return keys.size();
             },
             py::arg("prefix"), "Forget the slot plaintexts named `prefix*` entirely.")
        // ── K/V residency: park cache ciphertexts in the pinned KV arena between blocks ────
        .def("kv_store",
             [](Inference& inf, std::vector<PackedCtx>& cts, const std::vector<std::string>& keys) {
                 if (cts.size() != keys.size())
                     throw std::invalid_argument("kv_store: cts and keys differ in length");
                 py::gil_scoped_release nogil;
                 inf.cc()->PrewarmKvArena();
                 for (size_t i = 0; i < cts.size(); ++i)
                     if (cts[i].ct) inf.cc()->KvStoreStaged(cts[i].ct, keys[i], impl_kv_stream());
             },
             py::arg("cts"), py::arg("keys"),
             "Enqueue the device-to-host copy of each ciphertext into its pinned KV-arena slot "
             "`key` on the KV stream (async; no evict). kv_sync then kv_evict complete it.")
        .def("kv_load",
             [](Inference& inf, std::vector<PackedCtx>& cts, const std::vector<std::string>& keys) {
                 if (cts.size() != keys.size())
                     throw std::invalid_argument("kv_load: cts and keys differ in length");
                 py::gil_scoped_release nogil;
                 for (size_t i = 0; i < cts.size(); ++i)
                     if (cts[i].ct) inf.cc()->KvLoadStaged(cts[i].ct, keys[i], impl_kv_stream());
             },
             py::arg("cts"), py::arg("keys"),
             "Enqueue the host-to-device reload of each evicted ciphertext from its slot on the "
             "KV stream (async); kv_sync before the ciphertext is used.")
        .def("kv_evict",
             [](Inference& inf, std::vector<PackedCtx>& cts) {
                 py::gil_scoped_release nogil;
                 for (auto& pc : cts)
                     if (pc.ct) inf.cc()->KvEvict(pc.ct);
             },
             py::arg("cts"), "Free the device copy of stored ciphertexts (after kv_sync).")
        .def("kv_sync", [](Inference&) { py::gil_scoped_release nogil; cudaStreamSynchronize(impl_kv_stream()); },
             "Wait for the KV stream (every enqueued store or load has landed).")
        // ── the runtime's encode cache (inf.enc_cache: what the C++ masks go through) ──────
        // Keyed by tag AND level, like encode_at_cached. A miss encodes synchronously (and
        // counts); with strict_masks set a miss throws (planned-mask discipline).
        .def("mult_cached",
             [](Inference& inf, const PackedCtx& ct, const std::string& tag,
                const py::array& values, bool tagged) {
                 auto v = perseus_np::to_any(values);
                 py::gil_scoped_release nogil;
                 v.resize(static_cast<size_t>(inf.slots));
                 Ptx pt = cached_any_at(inf, tag, v, ct, tagged);
                 inf.load_plaintext(pt);
                 return inf.fhe->mult(ct, pt);
             },
             py::arg("ct"), py::arg("tag"), py::arg("values"), py::arg("tagged") = true,
             "ct * the plaintext cached under `tag` at ct's level (encode_at_cached: encoded "
             "on a miss, uploaded on first use, shared by every later use at that level).")
        .def("add_cached",
             [](Inference& inf, const PackedCtx& ct, const std::string& tag,
                const py::array& values, bool tagged) {
                 auto v = perseus_np::to_any(values);
                 py::gil_scoped_release nogil;
                 v.resize(static_cast<size_t>(inf.slots));
                 Ptx pt = cached_any_at(inf, tag, v, ct, tagged);
                 inf.load_plaintext(pt);
                 return inf.fhe->add(ct, pt);
             },
             py::arg("ct"), py::arg("tag"), py::arg("values"), py::arg("tagged") = true,
             "ct + the plaintext cached under `tag` at ct's level.")
        .def("mult_cached_many",
             [](Inference& inf, const PackedCtx& ct, const std::vector<std::string>& tags,
                const std::vector<py::array>& values) {
                 if (tags.size() != values.size())
                     throw std::invalid_argument("mult_cached_many: tags and values differ in length");
                 std::vector<perseus_np::AnyVec> vs;
                 vs.reserve(values.size());
                 for (const auto& a : values) vs.push_back(perseus_np::to_any(a));
                 py::gil_scoped_release nogil;
                 std::vector<Ptx> pts;
                 pts.reserve(tags.size());
                 for (size_t i = 0; i < tags.size(); ++i) {
                     vs[i].resize(static_cast<size_t>(inf.slots));
                     pts.push_back(cached_any_at(inf, tags[i], vs[i], ct, true));
                     inf.load_plaintext(pts.back());
                 }
                 std::vector<PackedCtx> out;
                 out.reserve(pts.size());
                 if (!pts.empty() && inf.fhe->lane_batch_usable(ct.ct, pts)) {
                     auto raw = inf.fhe->mult_batch_exec(ct.ct, pts);
                     for (size_t i = 0; i < pts.size(); ++i)
                         out.push_back(inf.fhe->mult_finish(ct, pts[i], std::move(raw[i])));
                 } else {
                     for (auto& pt : pts) out.push_back(inf.fhe->mult(ct, pt));
                 }
                 return out;
             },
             py::arg("ct"), py::arg("tags"), py::arg("values"),
             "ct * each cached plaintext, as one fused batch of lane products when the runtime "
             "allows (the C++ V-push path), else one product each.")
        .def("prime_pt",
             [](Inference& inf, const std::string& tag, const py::array& values, int level) {
                 auto v = perseus_np::to_any(values);
                 py::gil_scoped_release nogil;
                 v.resize(static_cast<size_t>(inf.slots));
                 if (v.cplx) inf.prime_enc_cache(tag, v.c, static_cast<uint32_t>(level));
                 else        inf.prime_enc_cache(tag, v.re, static_cast<uint32_t>(level));
             },
             py::arg("tag"), py::arg("values"), py::arg("level"),
             "Encode `values` under `tag` at `level` into the cache now, on this thread, if "
             "absent (the C++ prime_step_masks safety net).")
        .def("stage_pts",
             [](Inference& inf, const std::vector<std::tuple<std::string, int, py::array>>& items) {
                 auto st = std::make_shared<StagedPts>();
                 std::vector<std::tuple<std::string, uint32_t, perseus_np::AnyVec>> work;
                 work.reserve(items.size());
                 for (const auto& it : items) {
                     auto v = perseus_np::to_any(std::get<2>(it));
                     v.resize(static_cast<size_t>(inf.slots));
                     work.emplace_back(std::get<0>(it), static_cast<uint32_t>(std::get<1>(it)), std::move(v));
                 }
                 Inference* pinf = &inf;
                 py::gil_scoped_release nogil;
                 // The worker encodes only (no CUDA, no inf.enc_cache access): the same
                 // contract as gpt2_encode_shared_masks. Adoption happens on the main thread.
                 // The job owns a reference to the handle, so dropping it in Python early is safe.
                 st->fut = mask_submit([pinf, st, work = std::move(work)]() mutable {
                     st->out.reserve(work.size());
                     for (auto& w : work)
                         st->out.emplace_back(std::get<0>(w), std::get<1>(w),
                                              encode_any(*pinf, std::get<2>(w), std::get<1>(w)));
                 });
                 return st;
             },
             py::arg("items"),
             "Encode (tag, level, values) items on the runtime's mask worker while this thread "
             "keeps issuing GPU work; `adopt_pts(handle)` later moves them into the cache. The "
             "C++ decode arm stages the next token's masks this way under the argmax tail.")
        .def("adopt_pts",
             [](Inference& inf, std::shared_ptr<StagedPts> st) {
                 py::gil_scoped_release nogil;
                 if (!st || st->adopted) return static_cast<size_t>(0);
                 if (st->fut.valid()) st->fut.get();
                 size_t n = 0;
                 for (auto& e : st->out) {
                     inf.adopt_enc_cache(std::get<0>(e), std::get<1>(e), std::move(std::get<2>(e)));
                     ++n;
                 }
                 st->out.clear();
                 st->adopted = true;
                 return n;
             },
             py::arg("staged"),
             "Join a stage_pts job and adopt its plaintexts into the cache (a key already "
             "present wins; the staged copy is dropped). Returns the number adopted.")
        .def("erase_pts",
             [](Inference& inf, const std::vector<std::string>& tags) {
                 py::gil_scoped_release nogil;
                 size_t n = 0;
                 for (const auto& t : tags) n += inf.erase_enc_cache_all(t);
                 return n;
             },
             py::arg("tags"), "Drop these tags from the cache at every level (device copies freed).")
        .def("enc_cache_stats",
             [](const Inference& inf) {
                 py::dict d;
                 d["size"] = inf.enc_cache.size();
                 d["hits"] = inf.enc_cache_hit;
                 d["misses"] = inf.enc_cache_miss;
                 d["strict_misses"] = inf.mask_strict_miss;
                 return d;
             },
             "Cache size and the hit / miss counters of encode_at_cached.")
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
    m.def("device_sync", []() { py::gil_scoped_release nogil; cudaDeviceSynchronize(); },
          "cudaDeviceSynchronize: wait for every stream (a diagnostic fence).");
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
    m.def("decrypt_slots_complex",
          [](Inference& inf, const PackedCtx& pc) {
              std::vector<std::complex<double>> v;
              {
                  py::gil_scoped_release nogil;
                  auto pt = decrypt_pt(inf.cc(), pc.ct, inf.fhe->sk());
                  v = pt->GetCKKSPackedValue();
              }
              return perseus_np::from_cvec(v);
          },
          py::arg("inf"), py::arg("x"),
          "Every slot of a ciphertext as complex values (the imaginary lane of a complex payload).");
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
