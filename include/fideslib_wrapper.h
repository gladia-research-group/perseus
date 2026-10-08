#pragma once

#include "ckks_types.h"
#include "graph.h"
#include "packing/pack_signature.h"
#include "packing/packed_ctx.h"
#include "plan_json.h"
#include "slot_layout.h"
#include <CudaUtils.cuh>   // FIDESlib::CudaNvtxRange (FIDESLIB_NVTX-gated bookkeeping ranges)

#include <CKKS/Ciphertext.cuh>
#include <CKKS/Context.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>

// The exact Encode transform (slots -> plaintext coefficients), for the EvalMod-facing
// magnitude the placer prices (output_max_coeff). core-only header: no lbcrypto
// `duration` macro trap (that lives in the pke headers Bootstrap.cuh drags in).
#include <math/dftransform.h>

// NOT <CKKS/Bootstrap.cuh>: that header drags lbcrypto, whose `duration` macro breaks
// cuda::std::chrono in every downstream TU. Forward-declare, like setArcsineOverride below.
namespace FIDESlib::CKKS { int BootstrapPrecapture(Context& cc); }  // Bootstrap.cu

namespace FIDESlib::CKKS { void setArcsineOverride(int v); }  // ApproxModEval.cu

#include <any>
#include <cstdio>
#include <vector>
#include <complex>
#include <string>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <memory>
#include <cassert>
#include <iostream>
#include <fstream>
#include <regex>
#include <set>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <deque>
#include <type_traits>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <cuda_runtime.h>
#include "fhe_errors.h"

struct Inference;  // ctor below takes Inference&; full definition in inference.h
struct CKKSContext;

// BTS_DIM1="cts,stc": OpenFHE's BSGS baby-step sizes for the CtS and StC linear transforms
// (0 = OpenFHE's automatic split = the shipped default). Traffic campaign lever D.
inline std::vector<uint32_t> bts_dim1_from_env() {
    std::vector<uint32_t> d{0, 0};
    if (const char* e = std::getenv("BTS_DIM1")) {
        unsigned a = 0, b = 0;
        if (std::sscanf(e, "%u,%u", &a, &b) >= 1) {
            d[0] = a;
            d[1] = b;
        }
    }
    return d;
}

struct WithStep {
    CKKSContext* ctx_;
    inline WithStep(CKKSContext& c, const std::string& s);
    inline WithStep(Inference& inf, const std::string& s);  // defined in inference.h
    inline ~WithStep();
    WithStep(const WithStep&) = delete;
    WithStep& operator=(const WithStep&) = delete;
    inline void next(const std::string& s);
};


struct StepProfiler {
    enum class Mode { Off, Wall, Events };

    Mode mode = Mode::Off;
    bool initialized = false;

    struct Stat {
        uint64_t calls = 0;
        uint64_t inclusive_ns = 0;
        uint64_t exclusive_ns = 0;
    };
    std::unordered_map<std::string, Stat> stats;

    // Per-open-frame scratch; one entry per currently-open WithStep.
    std::vector<std::chrono::steady_clock::time_point> wall_t0;
    std::vector<cudaEvent_t> ev_start;
    std::vector<uint64_t> child_ns;

    void ensure_initialized() {
        if (initialized) return;
        initialized = true;
        const char* v = std::getenv("FHE_PROFILE");
        if (!v || !*v) { mode = Mode::Off; return; }
        const std::string s(v);
        if      (s == "wall" || s == "1" || s == "true" || s == "True") mode = Mode::Wall;
        else if (s == "events")                                          mode = Mode::Events;
        else                                                             mode = Mode::Off;
    }

    bool on() const { return mode != Mode::Off; }

    void on_push() {
        if (!on()) return;
        child_ns.push_back(0);
        if (mode == Mode::Wall) {
            cudaDeviceSynchronize();
            wall_t0.push_back(std::chrono::steady_clock::now());
        } else {
            cudaEvent_t ev;
            cudaEventCreate(&ev);
            cudaEventRecord(ev);
            ev_start.push_back(ev);
        }
    }

    void on_pop(const std::string& full_path) {
        if (!on()) return;
        if (child_ns.empty()) return;   // mode toggled mid-scope (FHE_PROFILE_TOKEN): push was skipped
        uint64_t ns = 0;
        if (mode == Mode::Wall) {
            cudaDeviceSynchronize();
            const auto t1 = std::chrono::steady_clock::now();
            ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - wall_t0.back()).count());
            wall_t0.pop_back();
        } else {
            cudaEvent_t ev_end;
            cudaEventCreate(&ev_end);
            cudaEventRecord(ev_end);
            cudaEventSynchronize(ev_end);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_start.back(), ev_end);
            ns = static_cast<uint64_t>(static_cast<double>(ms) * 1e6);
            cudaEventDestroy(ev_start.back());
            cudaEventDestroy(ev_end);
            ev_start.pop_back();
        }
        const uint64_t my_child = child_ns.back();
        child_ns.pop_back();
        auto& st = stats[full_path];
        st.calls         += 1;
        st.inclusive_ns  += ns;
        st.exclusive_ns  += (ns > my_child) ? (ns - my_child) : 0;
        if (!child_ns.empty()) child_ns.back() += ns;
    }

    void reset() { stats.clear(); wall_t0.clear(); ev_start.clear(); child_ns.clear(); }

    void dump(std::ostream& os) const {
        if (!initialized || mode == Mode::Off || stats.empty()) return;
        std::vector<std::pair<std::string, Stat>> rows(stats.begin(), stats.end());
        std::sort(rows.begin(), rows.end(),
                  [](const auto& a, const auto& b) {
                      return a.second.exclusive_ns > b.second.exclusive_ns;
                  });
        const auto ms = [](uint64_t ns) { return static_cast<double>(ns) / 1e6; };
        const char* name = (mode == Mode::Wall ? "wall" : "events");
        const std::ios::fmtflags saved = os.flags();
        const std::streamsize    prec  = os.precision();
        os << "\n[profile] mode=" << name
           << "  (sorted by self_ms; inc=inclusive)\n"
           << "  " << std::left  << std::setw(60) << "step"
                   << std::right << std::setw(10) << "calls"
                                 << std::setw(14) << "self_ms"
                                 << std::setw(14) << "inc_ms"
                                 << std::setw(14) << "self/call_us" << "\n";
        for (const auto& [path, st] : rows) {
            os << "  " << std::left  << std::setw(60) << path
                       << std::right << std::setw(10) << st.calls
                       << std::fixed << std::setprecision(3)
                                     << std::setw(14) << ms(st.exclusive_ns)
                                     << std::setw(14) << ms(st.inclusive_ns)
                                     << std::setw(14)
                       << (st.calls > 0
                               ? static_cast<double>(st.exclusive_ns) / st.calls / 1e3
                               : 0.0)
               << "\n";
        }
        os.flags(saved);
        os.precision(prec);
        os << std::flush;
    }
};

struct BootstrapPlan {
    std::unordered_set<std::string>              placement_after;
    std::unordered_map<std::string, uint32_t>    expected_levels;
    std::unordered_map<std::string, std::string> expected_producers;
    std::unordered_map<std::string, uint32_t>    weight_levels;
    std::unordered_map<std::string, std::vector<uint32_t>> mask_levels;
    // Plan-bound hint decisions: output vars of bootstrap_hint sites the planner's
    // final sim FIRES. When hints_bound (the plan carries the "hint_fire" key, even
    // empty), the planned runtime obeys these instead of re-evaluating the level
    // threshold — dynamic re-evaluation diverges from the sim on threshold-boundary
    // trajectories (deg-2 pending-rescale ±1).
    std::unordered_set<std::string>              hint_fire;
    bool                                         hints_bound = false;
    // Plan-bound rescale (realize) decisions, keyed like placements. Presence of the
    // "rescale_after" key — even empty — BINDS every landing-realize decision to the plan:
    // a refresh-target key realizes the refreshed output (per-site landing contract), any
    // other key realizes at the op that produced the var (block-boundary / pre-linear
    // anchors). An unbound plan realizes every planted landing.
    std::unordered_set<std::string>              rescale_after;
    bool                                         rescale_bound = false;
    // final_named_degs: per-var noise-degree pins beside final_named_levels. Checked
    // EXACTLY (both directions): a missing realize raises deg without deepening the
    // level, so the one-sided level check cannot see it.
    std::unordered_map<std::string, uint32_t>    expected_degs;
    int                                          cache_pin_level = -1;   // placer-chosen KV read level (-1 = unset)
    // Per-site sparse routing + input prescale (optional plan keys, keyed by the
    // bootstrap target var). sparse_bts_slots_rule mirrors rules.sparse_bts_slots — the
    // precomp set the planner assumed, validated against the built precomps at plan
    // install, where a mismatch is refused rather than clamped.
    std::unordered_map<std::string, uint32_t>    sparse_slots;
    std::unordered_map<std::string, double>      prescale;
    std::vector<uint32_t>                        sparse_bts_slots_rule;
    // Per-site EvalMod placement, same key and same parse shape as the two above.
    //   correction_factor — CF positions the usable band ([0.003, 0.03]·2^CF).
    //     Runtime-only: nothing precomputed depends on it, so it varies per call through
    //     CorrectionScope at ZERO level cost and without changing the schedule.
    //   offset            — the DC of `bootstrap(ct - c) + c`. Also level-free,
    //     and EXACT whatever the runtime value turns out to be: it is a plaintext add and
    //     its exact inverse, so a stale c loses benefit but can never corrupt.
    std::unordered_map<std::string, int>         correction_factor;
    // Level-aware ModRaise: composite levels below the chain top the site's raise stops at (0 = full).
    std::unordered_map<std::string, int>         raise_drop;
    std::unordered_map<std::string, double>      offset;
    bool valid = false;

    uint32_t weight_level(const std::string& key, uint32_t fallback) const {
        auto it = weight_levels.find(key);
        return it == weight_levels.end() ? fallback : it->second;
    }
    const std::vector<uint32_t>* mask_level_list(const std::string& site) const {
        auto it = mask_levels.find(site);
        return it == mask_levels.end() ? nullptr : &it->second;
    }
};

inline BootstrapPlan parse_bootstrap_plan_file(const std::string& path) {
    BootstrapPlan plan;

    std::ifstream in(path);
    if (!in) {
        return plan;   // valid=false
    }

    const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    // Iterative JSON parse (include/plan_json.h). A recursive scan overflows the default
    // 8 MB stack on a ~0.5 MB placement file, and the resulting segfault lands inside the
    // parser rather than at the caller, so it reads as a failure of the FHE forward.
    planjson::Value root;
    try {
        root = planjson::parse(content);
    } catch (const std::exception& e) {
        std::cerr << "[plan_parse] " << path << ": " << e.what() << " — plan ignored\n";
        return plan;   // valid=false
    }

    // Dict keys are searched ANYWHERE in the file: final_named_levels lives under
    // "summary", cache_pin_level under "rules".
    const planjson::Value* summary = root.find("summary");
    const planjson::Value* rules   = root.find("rules");
    auto find_any = [&](const char* key) -> const planjson::Value* {
        if (const auto* v = root.find(key)) return v;
        if (summary) if (const auto* v = summary->find(key)) return v;
        if (rules)   if (const auto* v = rules->find(key))   return v;
        return nullptr;
    };

    if (const auto* pl = root.find("placements"); pl && pl->is_array()) {
        for (const auto& e : pl->arr) {
            const auto* type = e.find("type");
            const auto* target = e.find("target_var");
            if (type && type->is_string() && type->str == "bootstrap_after_node"
                && target && target->is_string() && !target->str.empty())
                plan.placement_after.insert(target->str);
        }
    }

    if (const auto* d = find_any("final_named_levels"); d && d->is_object())
        for (const auto& kv : d->obj)
            if (kv.second.is_number())
                plan.expected_levels[kv.first] = static_cast<uint32_t>(kv.second.number);

    if (const auto* d = find_any("final_named_producers"); d && d->is_object())
        for (const auto& kv : d->obj)
            if (kv.second.is_string())
                plan.expected_producers[kv.first] = kv.second.str;

    if (const auto* d = find_any("weight_levels"); d && d->is_object())
        for (const auto& kv : d->obj)
            if (kv.second.is_number())
                plan.weight_levels[kv.first] = static_cast<uint32_t>(kv.second.number);

    if (const auto* d = find_any("mask_levels"); d && d->is_object()) {
        for (const auto& kv : d->obj) {
            if (!kv.second.is_array()) continue;
            std::vector<uint32_t> lvls;
            for (const auto& x : kv.second.arr)
                if (x.is_number()) lvls.push_back(static_cast<uint32_t>(x.number));
            if (!lvls.empty()) plan.mask_levels[kv.first] = std::move(lvls);
        }
    }

    // Plan-bound hint decisions: key presence (even an empty list) binds every
    // bootstrap_hint decision to the plan.
    if (const auto* d = find_any("hint_fire"); d && d->is_array()) {
        plan.hints_bound = true;
        for (const auto& x : d->arr)
            if (x.is_string()) plan.hint_fire.insert(x.str);
    }

    // Plan-bound rescale decisions: key presence (even an empty list) binds every
    // landing-realize decision to the plan (same rule as hint_fire).
    if (const auto* d = find_any("rescale_after"); d && d->is_array()) {
        plan.rescale_bound = true;
        for (const auto& x : d->arr)
            if (x.is_string()) plan.rescale_after.insert(x.str);
    }

    if (const auto* d = find_any("final_named_degs"); d && d->is_object())
        for (const auto& kv : d->obj)
            if (kv.second.is_number())
                plan.expected_degs[kv.first] = static_cast<uint32_t>(kv.second.number);

    if (const auto* d = find_any("cache_pin_level"); d && d->is_number())
        plan.cache_pin_level = static_cast<int>(d->number);

    // Per-site sparse routing + prescale (optional keys).
    if (const auto* d = find_any("sparse_slots"); d && d->is_object())
        for (const auto& kv : d->obj)
            if (kv.second.is_number())
                plan.sparse_slots[kv.first] = static_cast<uint32_t>(kv.second.number);

    if (const auto* d = find_any("prescale"); d && d->is_object())
        for (const auto& kv : d->obj)
            if (kv.second.is_number() && kv.second.number > 0.0)
                plan.prescale[kv.first] = kv.second.number;

    // Per-site correction factor + offset transform (optional keys).
    // CF is validated against the chain's deg guard at install, not here — the parser has
    // no context to compare against and a silent clamp would be worse than a loud refusal.
    if (const auto* d = find_any("correction_factor"); d && d->is_object())
        for (const auto& kv : d->obj)
            if (kv.second.is_number())
                plan.correction_factor[kv.first] = static_cast<int>(kv.second.number);

    if (const auto* d = find_any("offset"); d && d->is_object())
        for (const auto& kv : d->obj)
            if (kv.second.is_number())
                plan.offset[kv.first] = kv.second.number;

    if (const auto* d = find_any("raise_drop"); d && d->is_object())
        for (const auto& kv : d->obj)
            if (kv.second.is_number() && kv.second.number > 0)
                plan.raise_drop[kv.first] = static_cast<int>(kv.second.number);

    if (const auto* d = find_any("sparse_bts_slots"); d && d->is_array())
        for (const auto& x : d->arr)
            if (x.is_number())
                plan.sparse_bts_slots_rule.push_back(static_cast<uint32_t>(x.number));

    // valid = there is something actionable (bootstrap placements OR weight levels).
    plan.valid = !plan.placement_after.empty() || !plan.weight_levels.empty();
    return plan;
}

struct BlockPlans {
    std::vector<BootstrapPlan> blocks;

    const BootstrapPlan& at(int b) const {
        static const BootstrapPlan kEmpty;
        if (b < 0 || b >= static_cast<int>(blocks.size())) return kEmpty;
        return blocks[b];
    }
    bool any_valid() const {
        for (const auto& p : blocks) if (p.valid) return true;
        return false;
    }
    bool empty() const { return blocks.empty(); }
};

inline BlockPlans load_block_plans(const std::string& dir, int n_blocks) {
    BlockPlans plans;
    plans.blocks.reserve(n_blocks);
    for (int b = 0; b < n_blocks; ++b) {
        plans.blocks.push_back(
            parse_bootstrap_plan_file(dir + "/block_" + std::to_string(b) + "_placement.json"));
    }
    return plans;
}

// ── FIDESlib capability probes ──────────────────────────────────────────────────────
// `StoreRaw` / `DecryptStoredRaw` are used ONLY by the async magnitude capture. DETECT
// instead of require, so ONE tree builds on BOTH chains: where the symbols are missing the
// async path is compiled out and capture uses the synchronous magnitude probe (slower,
// identical values). Deliberately probed on the EXPRESSION, not a version macro: the deps
// trees carry no version stamp.
template <class C, class T, class = void>
struct fl_has_store_raw : std::false_type {};
template <class C, class T>
struct fl_has_store_raw<C, T,
    std::void_t<decltype(std::declval<C&>()->StoreRaw(std::declval<T&>()))>> : std::true_type {};

template <class C, class R, class K, class P, class = void>
struct fl_has_decrypt_stored_raw : std::false_type {};
template <class C, class R, class K, class P>
struct fl_has_decrypt_stored_raw<C, R, K, P,
    std::void_t<decltype(std::declval<C&>()->DecryptStoredRaw(
        std::declval<R&>(), std::declval<K&>(), std::declval<P>()))>> : std::true_type {};

// `if constexpr` only discards inside a TEMPLATE, so the guarded calls have to live in
// these helpers — an `if constexpr` in a plain member function would still require the
// dead branch to compile.
template <class C, class T>
inline std::shared_ptr<void> fl_store_raw(C& c, T& ct) {
    if constexpr (fl_has_store_raw<C, T>::value) {
        return c->StoreRaw(ct);
    } else {
        (void)c; (void)ct; return nullptr;
    }
}
template <class C, class R, class K, class P>
inline bool fl_decrypt_stored_raw(C& c, R& raw, K& sk, P pt) {
    if constexpr (fl_has_decrypt_stored_raw<C, R, K, P>::value) {
        c->DecryptStoredRaw(raw, sk, pt);
        return true;
    } else {
        (void)c; (void)raw; (void)sk; (void)pt; return false;
    }
}

struct CKKSContext {
    CC  cc;
    KP  keys;

    // ── Deferred heavy setup (pipeline-fill overlap) ─────────────────────────────────
    // With CKKSContextOptions.defer_heavy_setup, make_ckks_context returns after the CPU
    // context + mult keys exist (everything a host-side ENCODE needs) and stashes the
    // expensive tail — rotation keygen, bootstrap setups/keygens, LoadContext (the GPU
    // upload) — here. complete_setup() runs it; idempotent. The prefill driver encodes
    // block 0 on a worker while the main thread completes setup, so the pipeline fill
    // (~40 s of exposed encode) hides under key/precomp setup instead of the first block.
    std::function<void()> pending_heavy_setup;
    void complete_setup() {
        if (!pending_heavy_setup) return;
        auto work = std::move(pending_heavy_setup);
        pending_heavy_setup = nullptr;
        work();
    }
    bool setup_pending() const { return static_cast<bool>(pending_heavy_setup); }

    mutable std::unordered_map<uint64_t, Ptx> const_pt_cache;
    mutable std::unordered_map<uint64_t, Ptx> complex_const_pt_cache;   // {re,im}-const plaintexts (K/V pack i_pt/nhi)

    std::shared_ptr<GraphBuilder> graph_builder;
    std::unordered_map<const void*, std::string> ct_vars;
    std::unordered_map<const void*, std::string> pt_vars;
    std::unordered_map<std::string, uint32_t> expected_levels;
    std::unordered_map<std::string, std::string> expected_producers;
    std::unordered_set<std::string> placement_after;
    std::unordered_set<std::string> plan_hint_fire;    // hint outputs the live plan fires
    bool plan_hints_bound = false;                     // live plan carries hint decisions
    std::unordered_set<std::string> plan_rescale_after; // plan-bound realize anchors
    bool plan_rescale_bound = false;                    // live plan binds realize decisions
    std::unordered_map<std::string, uint32_t> expected_degs;  // final_named_degs pins
    // Per-site sparse routing + input prescale for the live block plan, keyed by the
    // bootstrap target var (placements and hint outputs alike). Absent key = dense,
    // no prescale.
    std::unordered_map<std::string, uint32_t> plan_sparse_slots;
    std::unordered_map<std::string, double>   plan_prescale;
    // Per-site correction factor and offset DC, same keying. Absent = the context-wide
    // CORRECTION_FACTOR and no offset.
    std::unordered_map<std::string, int>      plan_correction_factor;
    std::unordered_map<std::string, int>      plan_raise_drop;
    std::unordered_map<std::string, double>   plan_offset;
    int active_cache_pin_level = -1;   // placer-chosen KV read level for the live block plan (-1 = unset)
    uint64_t graph_ct_counter = 0;
    uint64_t graph_pt_counter = 0;
    uint64_t graph_anon_counter = 0;
    uint64_t unnamed_ct_count = 0;

    std::string graph_var_scope;
    uint64_t    graph_var_scope_counter = 0;
    uint64_t weight_relevel_count = 0;
    uint64_t unplanned_bootstrap_count = 0;

    struct WeightRelevelStat {
        uint32_t enc_level = 0;
        uint32_t ct_level  = 0;
        uint32_t enc_deg   = 0;
        uint32_t ct_deg    = 0;
        int32_t  delta     = 0;
        uint64_t hits      = 0;
    };
    std::map<std::string, WeightRelevelStat> weight_relevel_stats;

    // Report the weights whose pre-baked encode level did not match the level the runtime
    // met them at. An over-pin of exactly +1 on a deg-2 ciphertext is the expected
    // self-heal; anything else means the plan's weight levels and the run disagree.
    void dump_weight_relevel_report(std::ostream& os = std::cerr) const {
        if (weight_relevel_stats.empty()) {
            os << "[weight_relevel_report] clean: no pre-baked plaintext re-leveled "
                  "(plan levels == runtime levels)\n";
            return;
        }
        os << "[weight_relevel_report] " << weight_relevel_stats.size()
           << " weight(s) drifted (total re-encodes=" << weight_relevel_count << "):\n";
        for (const auto& kv : weight_relevel_stats) {
            const WeightRelevelStat& s = kv.second;
            const char* cls =
                (s.delta == 1 && s.ct_deg == 1) ? "INFO  over-pin+1 (deg-2->deg-1 self-heal, expected)" :
                (s.delta  > 1)                  ? "WARN  over-pin>+1 (unexpected; check plan)" :
                (s.delta  < 0)                  ? "ERROR under-pin (weight fell back to default; plan gap)" :
                                                  "INFO  drift";
            os << "  " << std::left << std::setw(16) << kv.first << std::right
               << " enc L" << s.enc_level << "/d" << s.enc_deg
               << " -> ct L" << s.ct_level << "/d" << s.ct_deg
               << "  delta=" << (s.delta >= 0 ? "+" : "") << s.delta
               << "  hits=" << s.hits
               << "   " << cls << "\n";
        }
    }

    bool placement_plan_enabled = false;
    std::unordered_set<std::string> planned_warned_;   // dedup keys for one-shot planned-mode warnings

    PublicKey<DCRTPoly>&  pk()  { return keys.publicKey; }
    PrivateKey<DCRTPoly>& sk()  { return keys.secretKey; }

    std::pair<double, double> debug_max_abs_re_im(const Ctx& ct) {
        Plaintext pt;
        Ctx c = ct;   // Decrypt takes a non-const Ctx&
        cc->Decrypt(c, keys.secretKey, &pt);
        const auto v = pt->GetCKKSPackedValue();
        double mr = 0.0, mi = 0.0;
        for (const auto& z : v) {
            mr = std::max(mr, std::abs(z.real()));
            mi = std::max(mi, std::abs(z.imag()));
        }
        return {mr, mi};
    }

    // TP_PROBE=1: print {max|Re(A)|, max|Im(B)|} of a ct with a label (token-pair lane trace).
    void tp_probe(const std::string& tag, const Ctx& ct) {
        if (!std::getenv("TP_PROBE")) return;
        auto ri = debug_max_abs_re_im(ct);
        fprintf(stderr, "[tp_probe:%s] |Re(A)|=%.5g |Im(B)|=%.5g\n", tag.c_str(), ri.first, ri.second);
    }

    void sync_ciphertext_cpu_from_device(Ctx& ct) {
        if (!ct || !ct->loaded) return;

        auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
        auto& ct_cpu  = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(
            cc->GetDeviceCiphertext(ct->gpu));

        FIDESlib::CKKS::RawCipherText raw_ct;
        ct_gpu->store(raw_ct);

        const size_t cpu_levels = ct_cpu->GetElements().empty()
                                      ? 0
                                      : ct_cpu->GetElements()[0].GetAllElements().size();
        const size_t gpu_levels = static_cast<size_t>(raw_ct.numRes);
        if (cpu_levels < gpu_levels) {
            std::vector<double> dummy(1, 0.0);
            // MakeCKKSPackedPlaintext's level = PRIMES dropped. FIDESlib getLevel() is the
            // top LIMB index; total q-limbs = composite_degree*(multDepth+1), so
            // primes_dropped = d*(multDepth+1) - 1 - getLevel()  (= multDepth - getLevel()
            // at d=1). On-grid GPU levels keep this a multiple of d (OpenFHE sentinel).
            auto pt_dummy = context->MakeCKKSPackedPlaintext(
                dummy, 1,
                composite_degree * (cc->multiplicative_depth + 1) - 1 - ct_gpu->getLevel());
            // The container only needs the right SHAPE (limb count); GetOpenFHECipherText
            // overwrites the polynomials. Encrypt with the PUBLIC key so a server session
            // built from a key bundle (no secret key in the process) can sync too.
            auto& pkImpl =
                std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(
                    keys.publicKey->pimpl);
            ct_cpu = context->Encrypt(pkImpl, pt_dummy);
        }

        FIDESlib::CKKS::GetOpenFHECipherText(ct_cpu, raw_ct);
    }

    // Every rotation step this session loaded (keygen band + load_rotation_steps, minus
    // free_rotation_steps): what close_session() releases. The subset shared with the
    // bootstrap precomputation (the DFT automorphism keys) is protected by
    // FreeRotationKeys and stays with the context until exit, so the freed count can be
    // smaller than the list.
    std::vector<int> loaded_rot_steps;

    size_t free_rotation_steps(const std::vector<int>& steps) {
        const size_t n = cc->FreeRotationKeys(steps, keys.publicKey);
        for (int s : steps)
            loaded_rot_steps.erase(std::remove(loaded_rot_steps.begin(), loaded_rot_steps.end(), s),
                                   loaded_rot_steps.end());
        return n;
    }

    void load_rotation_steps(const std::vector<int>& steps) {
        cc->LoadRotationKeys(steps, keys.publicKey);
        for (int s : steps)
            if (std::find(loaded_rot_steps.begin(), loaded_rot_steps.end(), s) == loaded_rot_steps.end())
                loaded_rot_steps.push_back(s);
    }

    int   total_depth = 25;
    int   btp_overhead = 15;
    // COMPOSITESCALING: primes per CKKS level (1 on classic chains). Wrapper levels are
    // PRIME-granular (level_of == OpenFHE GetLevel == primes dropped), so under composite
    // they move in steps of d — every ±1-level PREDICTION below must scale by this, while
    // comparisons against prime-granular thresholds (AUTO_BTS_LEVEL etc.) stay unchanged.
    int   composite_degree = 1;
    // Width of q0, the first (surviving) modulus — sets the coeff-encode centred-lift
    // bound. n64 uses 60, the n32 composite chain 56, so it cannot be hard-coded.
    int   first_mod_bits   = 60;
    // The scale the bootstrap runs at (BTP_SCALE_BITS, or SCALE_BITS with bootstrap off).
    int   bts_scale_bits   = 53;

    // The LOWEST legal correction factor on this chain. FIDESlib's Bootstrap throws
    // "deg=log2(q0/2^p)=N exceeds correctionFactor" whenever CF < deg, so a per-site CF
    // has a hard floor and it is a property of the chain, not a policy: n32 (56/54) floors
    // at 2, n64 (60/53) at 7. The planner is told this so it never emits an illegal value.
    int bootstrap_deg_floor() const {
        return std::max(0, first_mod_bits - bts_scale_bits);
    }

    // The definition of the pending-rescale level prediction. A ciphertext with a pending
    // FLEXIBLEAUTO rescale (noiseScaleDeg==2) must be met one CKKS LEVEL down, and one CKKS
    // level is `composite_degree` primes — so this reduces to `+1` exactly when d==1.
    //
    // For an ENCODE level an off-grid prediction is not a rounding error: OpenFHE fills
    // m_scalingFactorsReal only at prime indices k % d == 0 and stores a literal 1 in the
    // holes, so an off-grid level silently encodes against scalingFactor = 1 and throws
    // "Scaling factor too small" naming neither the level nor the caller.
    // Inference::pending_rescale_primes() forwards here; keep ONE definition.
    uint32_t pending_rescale_primes(const Ctx& ct) const {
        return (ct && ct->GetNoiseScaleDeg() == 2) ? static_cast<uint32_t>(composite_degree) : 0u;
    }

    // Cached post-bootstrap level (see bootstrap_output_level()).
    uint32_t bts_out_level_     = 0;
    bool     bts_out_level_set_ = false;

    uint32_t bts_iterations = 1;
    uint32_t bts_precision  = 0;
    // Sparse bootstrap precomps (0 = absent) + scope-driven routing. `sparse_bts_slots`
    // is the ROUTING DEFAULT (what SparseBtsScope arms — the head of the configured
    // list); `sparse_precomp_slots` is every slot count a precomp was actually built
    // for, which is what fold_bootstrap validates against.
    uint32_t sparse_bts_slots  = 0;
    uint32_t sparse_bts_active = 0;
    std::set<uint32_t> sparse_precomp_slots;

    // The slot count a fold wanting `s_wanted` can actually run at: `s_wanted` itself
    // when a precomp exists, else the smallest built s' > s_wanted (any larger power of
    // two is a multiple, so the caller can pre-ladder the gap: rotate+add at strides
    // s_wanted, 2·s_wanted, …, s'/2 — algebraically the same class sums), else S (the
    // dense degenerate: caller does the full ladder, the "fold" is a plain bootstrap
    // and recovery = 1/n_live — exactly the pre-fold code path).
    uint32_t fold_slots_for(uint32_t s_wanted) const {
        const uint32_t S = static_cast<uint32_t>(cc->GetRingDimension() / 2);
        if (s_wanted >= S || sparse_precomp_slots.count(s_wanted)) return std::min(s_wanted, S);
        for (uint32_t s : sparse_precomp_slots)   // std::set — ascending
            if (s > s_wanted) return s;
        return S;
    }
    uint32_t total_bootstraps = 0;
    // Of those, the ones taken with SparseBtsScope armed — i.e. on a lane the caller has
    // declared slot-periodic. This is the ROUTABLE fraction, and it is strictly smaller than
    // "bootstraps inside layernorm/softmax": in norm.cu the scope wraps only the inv_sqrt
    // Goldschmidt chain, while the centering and the final vector multiply bootstrap the full
    // vector outside it. Counting calls instead of scope-entries overstates the coverage.
    uint32_t total_bootstraps_sparse_lane = 0;

    // PLAINTEXT tag registry, keyed by the plaintext's identity. Ptx is an OpenFHE shared_ptr
    // we cannot add a field to, so the tag lives beside it. Registered at encode time, where
    // the values are in the clear and the tag is therefore EXACT — no conservatism needed.
    // Unregistered plaintexts return an unknown tag, which collapses the result to unknown:
    // opportunity lost, never soundness.
    mutable std::unordered_map<const void*, packtag::PackTag> pt_tags;

    // CIPHERTEXT tag registry. The tag lives on PackedCtx, but bootstraps happen on a raw Ctx,
    // so every PackedCtx op registers its result's tag here and inner_bootstrap looks it up —
    // no decryption, no tolerance.
    // Keyed by pointer, and ciphertext addresses ARE reused after free, so a stale entry
    // would silently mis-tag an unrelated ciphertext. Identity is therefore held by a
    // WEAK POINTER, which expires the moment the original is destroyed, so a reused address
    // fails the check deterministically and yields unknown (the conservative answer).
    struct CtTagEntry {
        std::weak_ptr<Ctx::element_type> wp;
        std::string var;
        packtag::PackTag tag;
    };
    std::unordered_map<const void*, CtTagEntry> ct_tags;

    void tag_ct(const Ctx& ct, const packtag::PackTag& t) {
        if (!ct) return;
        std::lock_guard<std::mutex> lk(tags_mtx);
        ct_tags[(const void*)ct.get()] = CtTagEntry{std::weak_ptr<Ctx::element_type>(ct),
                                                     var_for_ct(ct), t};
    }
    packtag::PackTag tag_of_ct(const Ctx& ct) {
        if (!ct) return packtag::PackTag{};
        std::lock_guard<std::mutex> lk(tags_mtx);
        auto it = ct_tags.find((const void*)ct.get());
        if (it == ct_tags.end()) return packtag::PackTag{};
        auto live = it->second.wp.lock();
        if (!live || live.get() != ct.get()) {   // original destroyed, address recycled
            ct_tags.erase(it);
            return packtag::PackTag{};
        }
        return it->second.tag;
    }
    // Build a PackedCtx and register its tag against the raw ciphertext in one step.
    PackedCtx tagged(Ctx c, const Packing& p, const packtag::PackTag& t) {
        tag_ct(c, t);
        return PackedCtx{std::move(c), p, t};
    }

    // The periodicity scan is O(S log S), sub-ms; encodes are memoised so it runs once per
    // distinct mask. pt_tags is written from worker threads too (the block-extraction
    // worker re-encodes weights while the main thread encodes masks): one mutex for both
    // tag registries.
    mutable std::mutex tags_mtx;

    // SUPPORT comes from the EXACT nonzero set (`*_exact`), never from the tolerance reading.
    // `live` counts slots above 1e-2*amax, which UNDER-reports — and an under-reported support
    // is unsound in the silent direction: the fold would relocate a value the tag says is not
    // there. On a plaintext the values are in the clear so exactness is free. A mask comes out
    // as its true AP; a weight vector comes out dense, which is the conservative answer.
    // The PERIOD stays the tolerance reading, which is what every ct-side rule assumes.
    void tag_plaintext(const Ptx& pt, const std::vector<double>& values) const {
        if (!pt) return;
        const PackSignature sig = analyze_packing(values);
        std::lock_guard<std::mutex> lk(tags_mtx);
        pt_tags[(const void*)pt.get()] = packtag::from_signature(
            sig.slots, sig.period, sig.live_exact, sig.stride_exact, sig.window_exact,
            sig.offset_exact);
    }
    void tag_plaintext(const Ptx& pt, const std::vector<std::complex<double>>& values) const {
        if (!pt) return;
        const PackSignature sig = analyze_packing(values);
        std::lock_guard<std::mutex> lk(tags_mtx);
        pt_tags[(const void*)pt.get()] = packtag::from_signature(
            sig.slots, sig.period, sig.live_exact, sig.stride_exact, sig.window_exact,
            sig.offset_exact);
    }
    packtag::PackTag tag_of(const Ptx& pt) const {
        if (!pt) return packtag::PackTag{};
        std::lock_guard<std::mutex> lk(tags_mtx);
        auto it = pt_tags.find((const void*)pt.get());
        return (it == pt_tags.end()) ? packtag::PackTag{} : it->second;
    }

    bool op_tally_active = false;
    std::map<std::string, uint64_t> op_tally;

    std::vector<std::string> step_stack;
    StepProfiler profile;
    Ptx complex_const_pt(double re, double im, int level = 0) const {
        uint64_t kre = 0, kim = 0;
        __builtin_memcpy(&kre, &re, sizeof(kre));
        __builtin_memcpy(&kim, &im, sizeof(kim));
        const uint64_t key = kre ^ (kim * 0x9E3779B97F4A7C15ull) ^ (static_cast<uint64_t>(level) << 1);
        auto it = complex_const_pt_cache.find(key);
        if (it != complex_const_pt_cache.end()) return it->second;
        const size_t slots = static_cast<size_t>(cc->GetRingDimension()) / 2;
        Ptx pt = cc->MakeCKKSPackedPlaintext(std::vector<std::complex<double>>(slots, {re, im}),
                                             /*noiseScaleDeg=*/1, (uint32_t)level);
        complex_const_pt_cache.emplace(key, pt);
        return pt;
    }
    void push_step(const std::string& s) {
        profile.ensure_initialized();
        step_stack.push_back(s);
        profile.on_push();
    }
    void pop_step() {
        if (step_stack.empty()) return;
        if (profile.on()) profile.on_pop(step_path());
        step_stack.pop_back();
    }

    std::string step_path() const {
        std::string out;
        for (const auto& s : step_stack) {
            if (!out.empty()) out += '.';
            out += s;
        }
        return out;
    }

    void check_packing(const Packing& a, const Packing& b) const {
        if (a != b)
            throw std::runtime_error(
                "PackedCtx packing mismatch [step=" + step_path() + "]: " +
                to_string(a.kind) + "(t=" + std::to_string(a.t) + ",hid=" +
                std::to_string(a.hidDim) + ") vs " + to_string(b.kind) +
                "(t=" + std::to_string(b.t) + ",hid=" + std::to_string(b.hidDim) + ")");
    }

    void attach_graph_builder(const std::shared_ptr<GraphBuilder>& builder) {
        graph_builder = builder;
        ct_vars.clear();
        pt_vars.clear();
        graph_ct_counter = 0;
        graph_pt_counter = 0;
    }

    void detach_graph_builder() {
        graph_builder.reset();
        ct_vars.clear();
        pt_vars.clear();
        graph_ct_counter = 0;
        graph_pt_counter = 0;
    }

    void clear_bootstrap_plan() {
        placement_after.clear();
        plan_hint_fire.clear();
        plan_hints_bound = false;
        plan_rescale_after.clear();
        plan_rescale_bound = false;
        expected_degs.clear();
        plan_sparse_slots.clear();
        plan_prescale.clear();
        plan_correction_factor.clear();
        plan_raise_drop.clear();
        plan_offset.clear();
        expected_levels.clear();
        expected_producers.clear();
        placement_plan_enabled = false;
        active_cache_pin_level = -1;
    }
    void verify_expected_level(const std::string& var_name,
                               const std::string& op_type,
                               uint32_t actual_level) const {
        FIDESlib::CudaNvtxRange _nvr_pw("pw::verify_level");
        if (!placement_plan_enabled || expected_levels.empty()) {
            return;
        }
        // A var with a PENDING bootstrap_after placement is checked here at naming time,
        // BEFORE maybe_apply_planned_bootstrap_after fires, but its expected level is the
        // planner's POST-bootstrap ledger. The downstream ops re-verify after the
        // bootstrap, so skipping the pre-fire check is correct by construction.
        if (placement_after.find(var_name) != placement_after.end()) {
            return;
        }
        if (!op_type.empty()) {
            auto pit = expected_producers.find(var_name);
            if (pit != expected_producers.end() && pit->second != op_type) {
                std::ostringstream oss;
                oss << "Producer mismatch for variable " << var_name
                    << ": expected " << pit->second
                    << ", actual " << op_type;
                throw std::runtime_error(oss.str());
            }
        }
        auto it = expected_levels.find(var_name);
        if (it != expected_levels.end()) {
            if (it->second < actual_level) {
                std::ostringstream oss;
                oss << "[plan_level_error] " << var_name << " expected " << it->second
                    << " actual " << actual_level << " step=" << step_path()
                    << " (planned mode is strict)";
                throw fhe::PlanError(oss.str());
            }
        }
    }
    void install_plan_live(const BootstrapPlan& plan) {
        clear_bootstrap_plan();
        if (!plan.valid) return;
        placement_after     = plan.placement_after;
        plan_hint_fire      = plan.hint_fire;
        plan_hints_bound    = plan.hints_bound;
        expected_levels     = plan.expected_levels;
        expected_producers  = plan.expected_producers;
        plan_rescale_after  = plan.rescale_after;
        plan_rescale_bound  = plan.rescale_bound;
        expected_degs       = plan.expected_degs;
        placement_plan_enabled = !plan.placement_after.empty();
        active_cache_pin_level = plan.cache_pin_level;
        if (plan.rescale_bound)
            std::cerr << "[plan_rescale] realize decisions plan-bound: "
                      << plan_rescale_after.size() << " anchor(s), deg pins="
                      << expected_degs.size() << "\n";

        // Chain exhaustion: refuse a plan that predicts a level with no usable limbs left.
        // `level` is PRIMES DROPPED and the chain carries d*(multDepth+1) of them, so a var
        // within d of the total has a fractional composite level and the plan is unrunnable.
        if (!plan.expected_levels.empty() && cc) {
            const uint32_t d     = static_cast<uint32_t>(composite_degree > 0 ? composite_degree : 1);
            const uint32_t total = d * static_cast<uint32_t>(cc->multiplicative_depth + 1);
            for (const auto& [var, lv] : plan.expected_levels) {
                if (lv + d <= total) continue;
                const std::string msg =
                    "[plan_level_error] plan predicts " + var + " at level " +
                    std::to_string(lv) + " but the chain holds only " + std::to_string(total) +
                    " primes (d=" + std::to_string(d) + "), leaving " +
                    std::to_string(lv >= total ? 0u : total - lv) +
                    " limb(s) — under one composite level, so the plan is unrunnable as emitted. "
                    "Re-plan with a cut that keeps the deep chains (ERASE_KEEP_STEPS), or "
                    "re-capture with FHE_ASYNC_MAG=0 if the graph magnitudes are garbage";
                throw fhe::PlanError(msg);
            }
        }

        // Sparse routing: validate EVERY planned s against the built precomps at
        // install time — a sparse bootstrap without its precomp is the silent-garbage
        // failure mode, so refuse loudly before any FHE work runs.
        if (!plan.sparse_slots.empty()) {
            const uint32_t S = static_cast<uint32_t>(cc->GetRingDimension() / 2);
            for (const auto& [var, s] : plan.sparse_slots) {
                if (s != S && !sparse_precomp_slots.count(s))
                    throw fhe::PlanError(
                        "[plan_sparse_error] plan routes " + var + " sparse at s=" +
                        std::to_string(s) + " but no such precomp is built "
                        "(SPARSE_BTS_SLOTS mismatch between plan and runtime)");
            }
            plan_sparse_slots = plan.sparse_slots;
            std::cerr << "[plan_sparse] " << plan_sparse_slots.size()
                      << " bootstrap site(s) sparse-routed, "
                      << plan.prescale.size() << " prescaled\n";
        }
        plan_prescale = plan.prescale;

        // Per-site correction factor: validate the whole map against the chain's deg guard
        // BEFORE any FHE work (`Bootstrap` throws "deg exceeds correctionFactor" per call,
        // which on the threaded path surfaces as a worker-thread segfault).
        if (!plan.correction_factor.empty()) {
            const int deg = bootstrap_deg_floor();
            std::map<int, int> hist;
            for (const auto& [var, cf] : plan.correction_factor) {
                if (cf < deg)
                    throw fhe::PlanError(
                        "[plan_cf_error] plan sets correction_factor=" + std::to_string(cf) +
                        " on " + var + " but this chain needs cf >= deg = " +
                        std::to_string(deg) + " (deg = FIRST_MOD_BITS - BTP_SCALE_BITS); "
                        "replan with --cf-min " + std::to_string(deg));
                if (cf == 1)
                    throw fhe::PlanError(
                        "[plan_cf_error] correction_factor=1 on " + var +
                        " is a known segfault; replan with --cf-min 2");
                ++hist[cf];
            }
            plan_correction_factor = plan.correction_factor;
            plan_raise_drop = plan.raise_drop;
            if (!plan_raise_drop.empty()) {
                std::map<int, int> h;
                for (const auto& kv : plan_raise_drop) ++h[kv.second];
                std::cerr << "[plan_raise] " << plan_raise_drop.size() << " site(s) raise below the chain top:";
                for (const auto& kv : h) std::cerr << " drop" << kv.first << "=" << kv.second;
                std::cerr << " (needs FIDESLIB_BTS_RAISE_DROPS to cover them)\n";
            }
            std::cerr << "[plan_cf] " << plan_correction_factor.size()
                      << " site(s) carry a per-site correction factor (chain deg=" << deg
                      << "), histogram:";
            for (const auto& [cf, n] : hist) std::cerr << " cf" << cf << "=" << n;
            std::cerr << "\n";
        }
        plan_offset = plan.offset;
        if (!plan_offset.empty())
            std::cerr << "[plan_offset] " << plan_offset.size()
                      << " site(s) carry an offset-transform DC\n";

        // Warn (once) when a planned plan is missing the level data it relies on.
        if (plan.expected_levels.empty() && planned_warned_.insert("no_expected_levels").second)
            std::cerr << "[plan_warn] planned mode enabled but no ct levels loaded "
                         "(final_named_levels empty); per-op level checks are disabled\n";
        if (plan.weight_levels.empty() && planned_warned_.insert("no_weight_levels").second)
            std::cerr << "[plan_warn] planned plan has no weight_levels; weights fall back to "
                         "bootstrap_output_level and will re-level on mismatch\n";
    }

    void warn_planned_weight_relevel(const std::string& name, uint32_t enc_lv, uint32_t ct_lv) {
        if (!placement_plan_enabled) return;
        std::ostringstream oss;
        oss << "[plan_weight_error] weight '" << name << "' encoded at level " << enc_lv
            << " but applied to ct at level " << ct_lv
            << " (planned mode is strict)";
        throw fhe::PlanError(oss.str());
    }

    void record_weight_relevel(const std::string& name,
                               uint32_t enc_lv, uint32_t ct_lv,
                               uint32_t enc_deg, uint32_t ct_deg) {
        auto& s = weight_relevel_stats[name];
        s.enc_level = enc_lv; s.ct_level = ct_lv;
        s.enc_deg   = enc_deg; s.ct_deg  = ct_deg;
        s.delta     = static_cast<int32_t>(enc_lv) - static_cast<int32_t>(ct_lv);
        ++s.hits;
    }
    bool load_bootstrap_plan_json(const std::string& path) {
        install_plan_live(parse_bootstrap_plan_file(path));
        return placement_plan_enabled;
    }

    bool planned_bootstraps_enabled() const {
        return placement_plan_enabled;
    }

    // The single point where a plan's per-site EvalMod decisions are applied. Both
    // plan-driven bootstrap paths — planted placements and plan-bound hints — must go
    // through it, or a site's decisions are parsed, installed and then silently ignored.
    //
    // The four knobs and why they compose in this order:
    //   offset  — removes the DC that puts the site out of range at all. Plaintext add,
    //             ZERO levels, and its inverse is exact, so a captured DC that no longer
    //             matches the runtime value costs accuracy and never correctness. It must
    //             wrap everything else: the prescale can only put the RESIDUAL in band
    //             once the DC is gone, and CF is chosen from that residual.
    //   CF      — positions the usable band. Zero levels, schedule-neutral.
    //   route   — lowers the noise floor; provably neutral above the band.
    //   prescale— the last resort: the only one of the four that costs a level.
    //
    //  RAW EvalAddInPlace on purpose, exactly as inner_bootstrap's prescale restore does:
    // the recording `inplace_add` would advance the sequential var counter, and these adds
    // exist only in PLANNED runs — an extra var name here shifts every later placement and
    // level lookup off by one against the captured graph.
    //
    //  THREADING: CorrectionScope mutates process-global ContextData, so a per-site CF is
    // only safe where bootstraps are serialised.
    void inner_bootstrap_planned(const std::string& var_name, Ctx& ct) {
        auto sit = plan_sparse_slots.find(var_name);
        auto pit = plan_prescale.find(var_name);
        auto cit = plan_correction_factor.find(var_name);
        auto oit = plan_offset.find(var_name);
        const double off = (oit != plan_offset.end()) ? oit->second : 0.0;

        // NO input-side deg normalization: the bootstrap handles deg-2 inputs natively, and
        // realizing first would make the sparse bootstrap clamp its landing; the OUTPUT-side
        // realize below is the landing contract.
        if (off != 0.0) cc->EvalAddInPlace(ct, -off);
        {
            // The planner proved s-periodicity from the captured static pack tag;
            // install_plan_live validated s against the built precomps and cf against
            // the chain's deg guard.
            FoldBtsScope fs(*this, sit != plan_sparse_slots.end()
                                       ? sit->second : sparse_bts_active);
            CorrectionScope cs(*this,
                               cit != plan_correction_factor.end() ? cit->second : -1,
                               cit != plan_correction_factor.end());
            auto rit = plan_raise_drop.find(var_name);
            RaiseScope rs(*this, rit != plan_raise_drop.end() ? rit->second : 0);
            inner_bootstrap(ct, pit != plan_prescale.end() ? pit->second : 1.0);
        }
        // Adding a broadcast constant back cannot lower the output's period (a constant is
        // period 1, and lcm(p, 1) == p), so the bootstrap's own tag stays sound and needs
        // no restamp.
        if (off != 0.0) cc->EvalAddInPlace(ct, off);
        // Landing contract: the bootstrap exits deg-2 on the uniform chain, and whether a
        // later check reads L or L+d depends on the consumer. Realize HERE so a planted
        // refresh lands deg-1 at one well-defined level (the level the planner is seeded
        // with). A rescale-bound plan decides per site: the realize fires iff this site's
        // target var is a rescale_after anchor, consumed here so the generic post-op hook
        // cannot double-report it. PLANNED path only.
        const bool landing_realize = plan_rescale_bound ? plan_rescale_take(var_name) : true;
        if (landing_realize && ct && ct->GetNoiseScaleDeg() == 2) cc->RescaleInPlace(ct);
    }

    // Consume a plan-bound realize anchor (fire-once, like placement_after).
    bool plan_rescale_take(const std::string& var_name) {
        if (var_name.empty()) return false;
        auto it = plan_rescale_after.find(var_name);
        if (it == plan_rescale_after.end()) return false;
        plan_rescale_after.erase(it);
        return true;
    }

    // Obey a non-refresh realize anchor (block-boundary / pre-linear sites). Refresh-target
    // anchors are consumed inside inner_bootstrap_planned instead, where the realize must
    // follow the bootstrap (input-side realization clamps the sparse landing).
    //  RAW RescaleInPlace on purpose (the prescale-restore rule): a recording op
    // would advance the sequential var counter and shift every later placement and
    // level lookup off by one against the captured graph.
    void maybe_apply_planned_rescale_after(const std::string& var_name, Ctx& ct) {
        if (!placement_plan_enabled || !plan_rescale_bound || !ct) return;
        if (!plan_rescale_take(var_name)) return;
        const int in_level = level_for_ct(ct);
        const int in_deg = static_cast<int>(ct->GetNoiseScaleDeg());
        if (in_deg == 2) cc->RescaleInPlace(ct);
        std::fprintf(stderr, "[planted_rsc] var=%s in=%d/d%d out=%d/d%d\n",
                     var_name.c_str(), in_level, in_deg,
                     (int)level_for_ct(ct), (int)ct->GetNoiseScaleDeg());
    }

    // Linear-input realize: FIDESlib's multPt rescales
    // a COPY of a deg-2 input inside every product, so a baby-step ciphertext that feeds G
    // giant-step products pays G rescales (INTT of the dropped primes + NTT-fused pass over
    // every remaining limb of c0/c1) where one would do. Realizing the pending rescale on the
    // (local, per-linear) rotated inputs ONCE is bit-identical -- the same rescale, on the
    // same values, reused instead of recomputed -- and leaves every recorded primitive's
    // OUTPUT level/degree unchanged (mult of deg-2@L and of deg-1@L+d both land deg-2@L+d),
    // so plans still bind. RAW on purpose, like maybe_apply_planned_rescale_after: a
    // recording op would shift the sequential var counter. Callers skip it under capture so
    // captured graphs keep the eager input levels.
    void realize_pending_rescale_raw(Ctx& ct) {
        if (ct && ct->GetNoiseScaleDeg() == 2) cc->RescaleInPlace(ct);
    }

    // final_named_degs pin: EXACT, both directions — a missing realize raises deg
    // without deepening the level, so the one-sided level check cannot see it.
    void verify_expected_deg(const std::string& var_name, const Ctx& ct) {
        if (expected_degs.empty() || !ct) return;
        auto it = expected_degs.find(var_name);
        if (it == expected_degs.end()) return;
        const uint32_t actual = static_cast<uint32_t>(ct->GetNoiseScaleDeg());
        if (actual == it->second) return;
        std::ostringstream oss;
        oss << "[plan_deg_error] " << var_name << " expected deg " << it->second
            << " actual " << actual << " step=" << step_path()
            << " (planned mode is strict)";
        throw fhe::PlanError(oss.str());
    }

    void maybe_apply_planned_bootstrap_after(const std::string& var_name, Ctx& ct) {
        FIDESlib::CudaNvtxRange _nvr_pw("pw::plan_after");
        if (!placement_plan_enabled || !ct) {
            return;
        }
        if (placement_after.find(var_name) == placement_after.end()) {
            // Plannable rescales: this hook already runs at every op exit with the
            // graph-consistent output var, so it is the one integration point for
            // non-refresh realize anchors and the deg pin — no per-site edits.
            maybe_apply_planned_rescale_after(var_name, ct);
            verify_expected_deg(var_name, ct);
            return;
        }
        // Fire-once: erasing the entry keeps a repeated var name from bootstrapping twice.
        placement_after.erase(var_name);

        const int in_level = level_for_ct(ct);
        const int in_deg = ct ? static_cast<int>(ct->GetNoiseScaleDeg()) : -1;
        const SparseRoute in_route = auto_sparse_route_for(ct);
        inner_bootstrap_planned(var_name, ct);
        const int out_deg = ct ? static_cast<int>(ct->GetNoiseScaleDeg()) : -1;
        const std::string out = var_name + "_planned_bootstrapped";
        name_ct(ct, out, true);
        // Ground-truth ledger for the planner's placement-output seeding: a sparse-routed
        // planted bootstrap does NOT land at the dense bts level (s=1 lands ~8 primes
        // richer, the clamped-DFT budget). One line per planted site.
        {
            auto sit = plan_sparse_slots.find(var_name);
            auto pit = plan_prescale.find(var_name);
            auto cit = plan_correction_factor.find(var_name);
            auto oit = plan_offset.find(var_name);
            std::fprintf(stderr,
                         "[planted_bts] var=%s s=%u prescale=%g cf=%d offset=%g "
                         "in=%d out=%d in_deg=%d out_deg=%d fold=%d drop=%d\n",
                         var_name.c_str(),
                         sit != plan_sparse_slots.end() ? sit->second : 0u,
                         pit != plan_prescale.end() ? pit->second : 1.0,
                         cit != plan_correction_factor.end() ? cit->second : -1,
                         oit != plan_offset.end() ? oit->second : 0.0,
                         in_level, (int)level_for_ct(ct),
                         in_deg, out_deg, (int)in_route.fold,
                         plan_raise_drop.count(var_name) ? plan_raise_drop.at(var_name) : 0);
        }
        record_primitive("auto_bootstrap", {var_name}, out, {in_level}, level_for_ct(ct), ct);
        // Post-refresh deg pin: final_named_degs keys are graph vars, and the sim's
        // deg for a refreshed var is the POST-refresh landing deg.
        verify_expected_deg(var_name, ct);
    }

    bool graph_enabled() const {
        return graph_builder && graph_builder->enabled();
    }

    bool naming_active() const {
        return graph_enabled() || placement_plan_enabled;
    }
    void name_ct(const Ctx& ct, const std::string& name, bool overwrite = true) {
        if (!naming_active()) return;
        if (!ct || name.empty()) {
            return;
        }
        const void* key = static_cast<const void*>(ct.get());
        if (!overwrite) {
            auto it = ct_vars.find(key);
            if (it != ct_vars.end()) {
                return;
            }
        }
        ct_vars[key] = name;
        verify_expected_level(name, "", level_of(ct));
    }

    void name_ct_if_absent(const Ctx& ct, const std::string& name) {
        name_ct(ct, name, false);
    }

    void name_pt(const Ptx& pt, const std::string& name, bool overwrite = true) {
        if (!naming_active()) return;
        if (!pt || name.empty()) {
            return;
        }
        const void* key = static_cast<const void*>(pt.get());
        if (!overwrite) {
            auto it = pt_vars.find(key);
            if (it != pt_vars.end()) {
                return;
            }
        }
        pt_vars[key] = name;
    }

    void name_pt_if_absent(const Ptx& pt, const std::string& name) {
        name_pt(pt, name, false);
    }

    std::string next_ct(const std::string& prefix = "v") {
        return prefix + "_" + std::to_string(++graph_ct_counter);
    }

    std::string next_pt(const std::string& prefix = "v") {
        return prefix + "_" + std::to_string(++graph_pt_counter);
    }

    void graph_scope_set(const std::string& prefix) { graph_var_scope = prefix; graph_var_scope_counter = 0; }
    void graph_scope_clear() { graph_var_scope.clear(); graph_var_scope_counter = 0; }
    bool graph_scope_active() const { return !graph_var_scope.empty(); }

    struct GraphScopeGuard {
        CKKSContext* ctx_ = nullptr;
        GraphScopeGuard() = default;
        explicit GraphScopeGuard(CKKSContext* c) : ctx_(c) {}
        GraphScopeGuard(const GraphScopeGuard&) = delete;
        GraphScopeGuard& operator=(const GraphScopeGuard&) = delete;
        GraphScopeGuard(GraphScopeGuard&& o) noexcept : ctx_(o.ctx_) { o.ctx_ = nullptr; }
        ~GraphScopeGuard() { if (ctx_) ctx_->graph_scope_clear(); }
    };
    GraphScopeGuard graph_scope_guard() { return GraphScopeGuard(this); }

    std::string var_for_ct(const Ctx& ct) {
        if (!naming_active()) return "x";
        if (!ct) {
            return "ct_null";
        }
        const void* key = static_cast<const void*>(ct.get());
        Ctx mutable_ct = ct;
        auto it = ct_vars.find(key);
        if (it != ct_vars.end()) {
            return it->second;
        }

        ++unnamed_ct_count;
        if (unnamed_ct_count == 1) {
            std::cout << "Warning: unnamed input ciphertext(s) at runtime (KV-cache reads "
                         "after per-block ct_vars.clear()); using anon names, "
                         "graph_ct_counter preserved. Further warnings suppressed.\n";
        }
        std::string name = "anon_" + std::to_string(++graph_anon_counter);
        ct_vars[key] = name;
        return name;
    }

    std::string set_new_var_for_ct(const Ctx& ct) {
        FIDESlib::CudaNvtxRange _nvr_pw("pw::new_var");
        if (!naming_active()) return "x";
        if (!ct) {
            return "ct_null";
        }
        const void* key = static_cast<const void*>(ct.get());
        std::string name = graph_var_scope.empty()
            ? next_ct("v")
            : (graph_var_scope + "_" + std::to_string(++graph_var_scope_counter));
        ct_vars[key] = name;
        return name;
    }

    // The name the NEXT set_new_var_for_ct will assign — must mirror its branches
    // exactly. Used by plan-bound hint decisions, which need the hint's output var
    // BEFORE deciding whether the hint bootstraps.
    std::string peek_new_var_for_ct(const Ctx& ct) const {
        if (!naming_active()) return "x";
        if (!ct) return "ct_null";
        return graph_var_scope.empty()
            ? "v_" + std::to_string(graph_ct_counter + 1)
            : (graph_var_scope + "_" + std::to_string(graph_var_scope_counter + 1));
    }

    std::string var_for_pt(const Ptx& pt) {
        if (!naming_active()) return "x";
        if (!pt) {
            return "pt_null";
        }
        const void* key = static_cast<const void*>(pt.get());
        auto it = pt_vars.find(key);
        if (it != pt_vars.end()) {
            return it->second;
        }
        std::string name = next_pt("pt");
        pt_vars[key] = name;
        return name;
    }

    std::string var_for_scalar(double scalar) {
        std::ostringstream oss;
        oss << "const(" << scalar << ")";
        return oss.str();
    }

    int level_for_ct(const Ctx& ct) const {
        return ct ? static_cast<int>(level_of(ct)) : -1;
    }

    int level_for_pt(const Ptx& pt) const {
        return pt ? static_cast<int>(pt->GetLevel()) : -1;
    }

    void record_primitive(const std::string& op_type,
                          const std::vector<std::string>& inputs,
                          const std::vector<int>& input_levels,
                          const std::string& output) {
        FIDESlib::CudaNvtxRange _nvr_pw("pw::record4");
        if (op_tally_active) ++op_tally[op_type];
        if (!graph_enabled()) {
            return;
        }
        graph_builder->add_node(op_type, inputs, output, input_levels, -1,
                                false, -1, false, 0.0, step_path());
    }

    void record_primitive(const std::string& op_type,
                          const std::vector<std::string>& inputs,
                          const std::string& output,
                          const std::vector<int>& input_levels,
                          int output_level,
                          const Ctx& output_ct) {
        FIDESlib::CudaNvtxRange _nvr_pw("pw::record6");
        if (op_tally_active) ++op_tally[op_type];
        if (!graph_enabled()) {
            return;
        }
        bool has_noise_level = false;
        int noise_level = -1;
        bool has_max_abs = false;
        double max_abs = 0.0;
        bool has_stats = false;          // output_mean / output_max_dev (offset transform)
        double node_mean = 0.0;
        double node_max_dev = 0.0;
        bool has_coeff = false;          // output_max_coeff / _ac (what EvalMod sees)
        double node_max_coeff = 0.0, node_max_coeff_ac = 0.0;
        bool defer_decrypt = false;   // async path: node added now, magnitude patched at drain
        bool defer_borrow  = false;
        const long ro_now = (output_ct && magnitude_reuse_active)
                                ? static_cast<long>(magnitude_reuse_ordinal++) : -1;
        if (output_ct) {
            noise_level = static_cast<int>(output_ct->GetNoiseScaleDeg());
            has_noise_level = true;
            static const bool capture_magnitude = true;
            const long ro = ro_now;
            if (capture_magnitude && magnitude_reuse_active && !magnitude_reuse_first) {
                if (async_mag_enabled()) {
                    defer_borrow = true;   // resolved from mag_ref_async at drain
                } else if (ro >= 0 && static_cast<size_t>(ro) < magnitude_reuse_ref.size()
                    && magnitude_reuse_has[static_cast<size_t>(ro)]) {
                    max_abs = magnitude_reuse_ref[static_cast<size_t>(ro)];
                    has_max_abs = true;   // borrowed reference: tractable, no decrypt
                }
            } else if (capture_magnitude && !magnitude_capture_suppressed) {
                if (async_mag_enabled()) {
                    defer_decrypt = true;   // D2H now (below), decrypt on the worker pool
                } else {
                Plaintext pt;
                Ctx ct_copy = output_ct;
                try {
                    cc->Decrypt(ct_copy, keys.secretKey, &pt);
                    // COMPLEX-aware magnitude: EvalMod's range constraint is on the encoded
                    // coefficients, which mix Re and Im (max|z| == max|Re| on real payloads).
                    const auto vz = pt->GetCKKSPackedValue();
                    // The recorded magnitude is the all-slot max; the live/dead lane split is
                    // reported only.
                    const std::vector<uint8_t>* lmask = live_lane_mask.get();
                    const LaneSplit split = lane_split(vz, lmask);
                    max_abs = split.all();
                    has_max_abs = true;
                    if (split.has_mask) {
                        const std::string sp_l = step_path();
                        lane_stat_add(sp_l.substr(0, sp_l.find_last_of('.')), split);
                    }
                    // EvalMod placement stats: mean = the DC the offset transform subtracts
                    // (Re mean); max_dev = max|z - mean|, what the bootstrap still carries.
                    // MEASURED, not inferred: (max, mean) alone only bounds the residual.
                    has_stats = slot_stats(vz, nullptr, node_mean, node_max_dev);
                    has_coeff = coeff_stats(vz, node_max_coeff, node_max_coeff_ac);
                } catch (const std::exception& e) {
                    has_max_abs = false;
                    // Report the decrypt failure: one line per distinct step family.
                    static std::set<std::string> seen_fams;
                    static std::mutex seen_mtx;
                    const std::string sp = step_path();
                    const std::string fam = sp.substr(0, sp.find_last_of('.'));
                    std::lock_guard<std::mutex> lk(seen_mtx);
                    if (seen_fams.insert(fam).second)
                        std::fprintf(stderr, "[mag_skip] %s: %s\n", sp.c_str(), e.what());
                } catch (...) {
                    has_max_abs = false;
                }
                if (magnitude_reuse_active && ro >= 0) {   // iter 0: store the reference
                    if (static_cast<size_t>(ro) >= magnitude_reuse_ref.size()) {
                        magnitude_reuse_ref.resize(ro + 1, 0.0);
                        magnitude_reuse_has.resize(ro + 1, false);
                    }
                    magnitude_reuse_ref[static_cast<size_t>(ro)] = max_abs;
                    magnitude_reuse_has[static_cast<size_t>(ro)] = has_max_abs;
                }
                }
            }
        }
        graph_builder->add_node(op_type, inputs, output, input_levels, output_level,
                                has_noise_level, noise_level, has_max_abs, max_abs,
                                step_path());
        // Sync path stats (the async worker emits the same four numbers at drain).
        if (has_stats)
            graph_builder->graph()->set_output_stats(
                graph_builder->graph()->size() - 1, node_mean, node_max_dev);
        if (has_coeff)
            graph_builder->graph()->set_output_max_coeff(
                graph_builder->graph()->size() - 1, node_max_coeff, node_max_coeff_ac);
        if (defer_decrypt || defer_borrow) {
            const std::size_t idx = graph_builder->graph()->size() - 1;
            mag_node_of[output] = idx;
            if (defer_borrow) {
                PendingMag b;
                b.node_idx = idx;
                b.borrow = ro_now;
                std::lock_guard<std::mutex> lk(mag_mtx);
                mag_borrows.push_back(std::move(b));
            } else if (mag_copy_op(op_type) && !inputs.empty()
                       && mag_node_of.count(inputs.front())
                       && !(magnitude_reuse_active && ro_now >= 0)) {
                // exact ma inheritance — no decrypt; resolved at drain. Reuse-scope
                // iter-0 nodes still decrypt (they must fill the ordinal reference).
                PendingMag c;
                c.node_idx = idx;
                c.copy_from = (long)mag_node_of[inputs.front()];
                c.negate = (op_type == "negate" || op_type == "negate_inplace");
                ++mag_stat_copies;
                std::lock_guard<std::mutex> lk(mag_mtx);
                mag_borrows.push_back(std::move(c));
            } else {
                PendingMag it;
                it.node_idx = idx;
                it.ordinal = (magnitude_reuse_active ? ro_now : -1);
                const auto st0 = std::chrono::steady_clock::now();
                it.raw = fl_store_raw(cc, output_ct);   // the D2H snapshot, at value time
                mag_stat_store_us += std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - st0).count();
                ++mag_stat_items;
                // byte estimate for the queue budget: 2 polys × remaining limbs × N × 8B
                it.bytes = size_t(2) * size_t(std::max(1, level_for_ct(output_ct) + 1))
                           * (cc->GetRingDimension()) * sizeof(uint64_t);
                it.live_mask = live_lane_mask;
                if (it.live_mask) {
                    const std::string sp_a = step_path();
                    it.fam = sp_a.substr(0, sp_a.find_last_of('.'));
                }
                mag_enqueue(std::move(it));
            }
        }
        // Stamp the STATIC pack tag of the output: this is what makes the planner's
        // sparse-slot choice decrypt-free. Unknown tags
        // emit nothing — absent means dense to the planner, the sound direction.
        if (output_ct) {
            const packtag::PackTag tg = tag_of_ct(output_ct);
            if (tg.known()) {
                if (GraphNode* n = graph_builder->last_node()) stamp_pack_tag(n, tg);
            }
        }
    }

    static void stamp_pack_tag(GraphNode* n, const packtag::PackTag& tg) {
        n->has_pack_tag = true;
        n->pack_period  = tg.period;
        n->pack_kind    = static_cast<int>(tg.support.kind);
        n->pack_offset  = tg.support.offset;
        n->pack_stride  = tg.support.stride;
        n->pack_count   = tg.support.count;
        n->pack_width   = tg.support.width;
    }

    // A tag stamped after its producing op was recorded (tag_reduce: a rotate-and-sum ladder's
    // output is periodic, which the ladder's own propagation does not derive) goes back onto
    // that op's graph node too, so a planned refresh of the reduction output itself can route
    // sparse instead of dense.
    void restamp_graph_tag(const Ctx& ct) {
        if (!ct || !graph_builder || !graph_builder->enabled()) return;
        const packtag::PackTag tg = tag_of_ct(ct);
        if (!tg.known()) return;
        if (GraphNode* n = graph_builder->last_node_for(var_for_ct(ct))) stamp_pack_tag(n, tg);
    }

    uint32_t auto_bts_level_override = 24;  // decode sweet spot; set from CKKSContextOptions at build
    bool     complex_payload = false;       // SetCKKSDataTypeComplex; set from CKKSContextOptions at build

    bool magnitude_capture_suppressed = false;
    struct MagnitudeSuppressScope {
        CKKSContext& cc_;
        bool prev_;
        explicit MagnitudeSuppressScope(CKKSContext& cc, bool armed = true)
            : cc_(cc), prev_(cc.magnitude_capture_suppressed) {
            if (armed) cc_.magnitude_capture_suppressed = true;
        }
        ~MagnitudeSuppressScope() { cc_.magnitude_capture_suppressed = prev_; }
    };

    // Live-lane magnitude pricing. A model region publishes which slots are live, for
    // ELEMENTWISE regions only — a rotation-mixing interior (any linear) must NOT install
    // one, since a dead lane there can transiently hold live data.
    // No scope installed => null mask => every slot live.
    std::shared_ptr<const std::vector<uint8_t>> live_lane_mask;   // null = all live
    struct LiveLaneScope {
        CKKSContext& cc_;
        std::shared_ptr<const std::vector<uint8_t>> prev_;
        LiveLaneScope(CKKSContext& cc, std::shared_ptr<const std::vector<uint8_t>> m)
            : cc_(cc), prev_(cc.live_lane_mask) { cc_.live_lane_mask = std::move(m); }
        ~LiveLaneScope() { cc_.live_lane_mask = prev_; }
    };
    // The recorded node magnitude is the all-slot max|z|; the live/dead split is only
    // reported ([mag_lanes]).
    struct LaneSplit {
        double live = 0.0;
        double dead = 0.0;
        bool   has_mask = false;
        double all() const { return std::max(live, dead); }
    };
    template <class Vec>
    static LaneSplit lane_split(const Vec& vz, const std::vector<uint8_t>* mask) {
        LaneSplit s;
        s.has_mask = mask && mask->size() == vz.size();
        for (std::size_t i = 0; i < vz.size(); ++i) {
            const double a = std::abs(vz[i]);
            if (s.has_mask && !(*mask)[i]) { if (a > s.dead) s.dead = a; }
            else if (a > s.live) s.live = a;
        }
        return s;
    }
    // Per-step-family live/dead report, printed at drain next to [mag].
    struct LaneStat {
        double live = 0.0, dead = 0.0;
        std::size_t nodes = 0, dead_dominant = 0;
    };
    std::map<std::string, LaneStat> mag_lane_stats;
    std::mutex mag_lane_mtx;
    void lane_stat_add(const std::string& fam, const LaneSplit& s) {
        if (!s.has_mask) return;
        std::lock_guard<std::mutex> lk(mag_lane_mtx);
        auto& st = mag_lane_stats[fam];
        st.live = std::max(st.live, s.live);
        st.dead = std::max(st.dead, s.dead);
        ++st.nodes;
        if (s.dead > 10.0 * std::max(s.live, 1e-300)) ++st.dead_dominant;
    }
    void lane_report() {
        std::lock_guard<std::mutex> lk(mag_lane_mtx);
        if (mag_lane_stats.empty()) return;
        std::fprintf(stderr, "[mag_lanes] priced=all-slot families=%zu\n", mag_lane_stats.size());
        for (const auto& kv : mag_lane_stats)
            std::fprintf(stderr,
                         "[mag_lanes] %-40s nodes=%6zu live_max=%.6g dead_max=%.6g dead_dominant=%zu\n",
                         kv.first.c_str(), kv.second.nodes, kv.second.live, kv.second.dead,
                         kv.second.dead_dominant);
        std::fflush(stderr);
        mag_lane_stats.clear();
    }

    // ── EvalMod placement stats, shared by the sync probe and the async worker ────────
    // slot_stats: mean (the DC the offset transform subtracts — a REAL subtraction, so the
    // Re mean) and max_dev = max|z - mean| (what the bootstrap still carries once the DC is
    // gone), over the live lanes when a mask is given (must be the SAME slot set as
    // max_abs, or the placer gets a CF it cannot honour). MEASURED, not inferred: (max,
    // mean) only BOUNDS the residual.
    static bool slot_stats(const std::vector<std::complex<double>>& vz,
                           const std::vector<uint8_t>* lmask,
                           double& mean, double& max_dev) {
        if (vz.empty()) return false;
        long double acc = 0.0L;
        std::size_t n_acc = 0;
        for (std::size_t i = 0; i < vz.size(); ++i) {
            if (lmask && !(*lmask)[i]) continue;
            acc += vz[i].real();
            ++n_acc;
        }
        if (!n_acc) return false;
        mean = static_cast<double>(acc / (long double)n_acc);
        max_dev = 0.0;
        for (std::size_t i = 0; i < vz.size(); ++i) {
            if (lmask && !(*lmask)[i]) continue;
            const double d = std::abs(vz[i] - std::complex<double>(mean, 0.0));
            if (d > max_dev) max_dev = d;
        }
        return true;
    }
    // coeff_stats: the max plaintext COEFFICIENT — the quantity EvalMod's range constraint is
    // actually on. Exactly OpenFHE's CKKSPackedEncoding::Encode: coefficients = [Re, Im] of
    // FFTSpecialInv(slots, 2N) (times the scale). ALL lanes, no mask: junk lanes are in the
    // polynomial EvalMod sees. `ac` excludes the X^0 (DC) coefficient. The 2N table is
    // read-only after the first Decode, so this is worker-safe.
    bool coeff_stats(const std::vector<std::complex<double>>& vz,
                     double& max_coeff, double& max_coeff_ac) const {
        if (vz.empty()) return false;
        std::vector<std::complex<double>> inv(vz);
        lbcrypto::DiscreteFourierTransform::FFTSpecialInv(inv, cc->GetRingDimension() * 2);
        max_coeff = 0.0; max_coeff_ac = 0.0;
        for (std::size_t i = 0; i < inv.size(); ++i) {
            const double re = std::abs(inv[i].real()), im = std::abs(inv[i].imag());
            const double m = std::max(re, im);
            if (m > max_coeff) max_coeff = m;
            const double m_ac = (i == 0) ? im : m;
            if (m_ac > max_coeff_ac) max_coeff_ac = m_ac;
        }
        return true;
    }

    bool   magnitude_reuse_active  = false;
    bool   magnitude_reuse_first   = false;
    size_t magnitude_reuse_ordinal = 0;
    std::vector<double> magnitude_reuse_ref;
    std::vector<bool>   magnitude_reuse_has;
    struct MagnitudeReuseScope {
        CKKSContext& cc_;
        bool   enabled_;
        bool   prev_active_;
        bool   prev_first_;
        size_t prev_ord_;
        MagnitudeReuseScope(CKKSContext& cc, bool first, bool enabled = true)
            : cc_(cc), enabled_(enabled), prev_active_(cc.magnitude_reuse_active),
              prev_first_(cc.magnitude_reuse_first), prev_ord_(cc.magnitude_reuse_ordinal) {
            if (!enabled_) return;
            cc_.magnitude_reuse_active  = true;
            cc_.magnitude_reuse_first   = first;
            cc_.magnitude_reuse_ordinal = 0;
            if (first) { cc_.magnitude_reuse_ref.clear(); cc_.magnitude_reuse_has.clear(); }
        }
        ~MagnitudeReuseScope() {
            if (!enabled_) return;
            cc_.magnitude_reuse_active  = prev_active_;
            cc_.magnitude_reuse_first   = prev_first_;
            cc_.magnitude_reuse_ordinal = prev_ord_;
        }
    };

    // ── ASYNC magnitude capture (FHE_ASYNC_MAG, default ON) ─────────────────────────
    // The sync magnitude probe decrypts EVERY node output on the main thread — D2H +
    // CPU decrypt + 32k-slot decode per node is ~95% of capture wall time. Split it:
    // the main thread does ONLY the D2H snapshot (StoreRaw — the value must be taken at
    // record time); a CPU worker pool does decrypt+decode+max-scan concurrently with
    // the GPU compute (pure CPU — no GPU concurrency, per the FIDESlib two-op hazard).
    // Results patch into the graph nodes at drain (export_graph_json). Byte-budgeted
    // queue gives backpressure. Magnitude-reuse borrows resolve at drain from the
    // async-filled reference table (ordinal order is main-thread sequential, unchanged).
    // Compiled out entirely against a FIDESlib without StoreRaw — the env then cannot turn
    // it on, and capture falls back to the synchronous probe.
    static constexpr bool kAsyncMagAvailable = fl_has_store_raw<CC, Ctx>::value;
    static bool async_mag_enabled() {
        if (!kAsyncMagAvailable) return false;
        static const bool on = [] {
            const char* e = std::getenv("FHE_ASYNC_MAG");
            return !(e && *e && std::atoi(e) == 0);
        }();
        return on;
    }
    struct PendingMag {
        std::size_t node_idx;
        long ordinal = -1;                 // reuse-store ordinal (iter 0), else -1
        long borrow  = -1;                 // borrow ordinal (iter>0), else -1
        long copy_from = -1;               // node idx whose ma this node inherits, else -1
        std::shared_ptr<void> raw;         // StoreRaw snapshot (null for borrows/copies)
        std::size_t bytes = 0;
        // The live mask is a property of the RECORD site (the scope is long gone by the
        // time a worker picks the item up), so it rides the item.
        std::shared_ptr<const std::vector<uint8_t>> live_mask;
        std::string fam;
        bool negate = false;               // copy_from through negate: the mean flips sign
    };
    // What the worker measures per node (the same four numbers the sync probe emits).
    struct MagStats {
        double max_abs = 0.0;
        bool has_stats = false; double mean = 0.0, max_dev = 0.0;
        bool has_coeff = false; double max_coeff = 0.0, max_coeff_ac = 0.0;
    };
    void apply_mag(ComputationGraph* g, std::size_t idx, const MagStats& s) {
        g->set_output_max_abs(idx, s.max_abs);
        if (s.has_stats) g->set_output_stats(idx, s.mean, s.max_dev);
        if (s.has_coeff) g->set_output_max_coeff(idx, s.max_coeff, s.max_coeff_ac);
    }
    // Permutation-class ops preserve max|Re| EXACTLY (rotation permutes slots,
    // conjugate/negate flip signs, clones/level moves copy values) — their magnitude is
    // inherited from the input at drain instead of decrypted. Rotation ladders are made
    // of exactly these nodes, so this cuts the capture's decode work roughly in half.
    static bool mag_copy_op(const std::string& op) {
        return op == "rotate" || op == "rotate_inplace" || op == "clone"
            || op == "conjugate" || op == "negate" || op == "negate_inplace"
            || op == "level_reduce" || op == "drop_to_level" || op == "level_hint";
    }
    std::unordered_map<std::string, std::size_t> mag_node_of;   // var -> node idx (capture)
    std::deque<PendingMag> mag_queue;
    std::vector<PendingMag> mag_borrows;   // resolved at drain, in order
    std::unordered_map<std::size_t, MagStats> mag_results;   // node_idx -> measured stats
    std::unordered_map<long, MagStats> mag_ref_async;        // ordinal -> stats
    std::mutex mag_mtx;
    std::condition_variable mag_cv_push, mag_cv_pop;
    std::vector<std::thread> mag_workers;
    std::atomic<size_t> mag_inflight{0};
    std::atomic<bool> mag_stop{false};
    std::atomic<bool> mag_first_done{false};
    size_t mag_queue_bytes = 0;
    static constexpr size_t kMagQueueBudget = size_t(6) << 30;   // 6 GB host backlog cap
    // telemetry (drained per block): where does capture time actually go?
    std::atomic<uint64_t> mag_stat_store_us{0}, mag_stat_backpressure_us{0};
    std::atomic<uint64_t> mag_stat_items{0}, mag_stat_copies{0};

    void mag_worker_main() {
        // OpenFHE's CPU decrypt/decode opens its own OMP parallel regions; N workers ×
        // full-width teams oversubscribes catastrophically (measured: 360 s/block vs the
        // 8 s/block sync baseline). One OMP thread per worker — the parallelism IS the
        // worker pool.
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        for (;;) {
            PendingMag item;
            {
                std::unique_lock<std::mutex> lk(mag_mtx);
                // The FIRST decode initializes OpenFHE's cached DFT/decode tables —
                // serialize it (mag_first_done) before going wide.
                mag_cv_pop.wait(lk, [&] {
                    return mag_stop.load()
                        || (!mag_queue.empty()
                            && (mag_first_done.load() || mag_inflight.load() == 0));
                });
                if (mag_queue.empty()) {
                    if (mag_stop.load()) return;
                    continue;
                }
                item = std::move(mag_queue.front());
                mag_queue.pop_front();
                mag_queue_bytes -= item.bytes;
                ++mag_inflight;
            }
            mag_cv_push.notify_all();
            MagStats m;
            bool ok = true;
            try {
                Plaintext pt;
                if (!fl_decrypt_stored_raw(cc, item.raw, keys.secretKey, &pt))
                    throw std::runtime_error("async mag: FIDESlib has no DecryptStoredRaw");
                // Same measurement as the sync probe: all-slot max|z|, lane split reported.
                const auto vz = pt->GetCKKSPackedValue();
                const LaneSplit split = lane_split(vz, item.live_mask.get());
                m.max_abs = split.all();
                if (split.has_mask) lane_stat_add(item.fam, split);
                m.has_stats = slot_stats(vz, nullptr, m.mean, m.max_dev);
                m.has_coeff = coeff_stats(vz, m.max_coeff, m.max_coeff_ac);
            } catch (...) {
                ok = false;
            }
            {
                std::lock_guard<std::mutex> lk(mag_mtx);
                if (ok) {
                    mag_results[item.node_idx] = m;
                    if (item.ordinal >= 0) mag_ref_async[item.ordinal] = m;
                } else if (item.ordinal >= 0) {
                    mag_ref_async.erase(item.ordinal);   // absent = no reference
                }
                --mag_inflight;
                mag_first_done.store(true);
            }
            mag_cv_pop.notify_all();
            mag_cv_push.notify_all();   // drain waits on queue-empty && inflight==0
        }
    }

    ~CKKSContext() {
        report_auto_sparse();   // no-op unless SPARSE_AUTO routed something
        {
            std::lock_guard<std::mutex> lk(mag_mtx);
            mag_stop.store(true);
        }
        mag_cv_pop.notify_all();
        for (auto& t : mag_workers)
            if (t.joinable()) t.join();
    }

    void mag_enqueue(PendingMag&& item) {
        {
            std::unique_lock<std::mutex> lk(mag_mtx);
            if (mag_workers.empty()) {
                const int n = [] {
                    const char* e = std::getenv("FHE_MAG_WORKERS");
                    const int v = (e && *e) ? std::atoi(e) : 28;
                    return v > 0 ? v : 28;
                }();
                mag_stop.store(false);
                for (int i = 0; i < n; ++i)
                    mag_workers.emplace_back([this] { mag_worker_main(); });
            }
            const auto bp0 = std::chrono::steady_clock::now();
            mag_cv_push.wait(lk, [&] {
                return mag_queue_bytes + item.bytes <= kMagQueueBudget || mag_queue.empty();
            });
            mag_stat_backpressure_us += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - bp0).count();
            mag_queue_bytes += item.bytes;
            mag_queue.push_back(std::move(item));
        }
        mag_cv_pop.notify_one();
    }

    // Wait for every queued decrypt, then patch magnitudes (and resolve borrows) into
    // the graph. Called from export_graph_json before serialization.
    void drain_magnitudes() {
        if (!graph_builder) return;
        {
            std::unique_lock<std::mutex> lk(mag_mtx);
            mag_cv_pop.notify_all();
            mag_cv_push.wait(lk, [&] { return mag_queue.empty() && mag_inflight.load() == 0; });
            auto g = graph_builder->graph();
            for (const auto& [idx, ms] : mag_results) apply_mag(g.get(), idx, ms);
            // Copy-inheritance (permutation-class ops) resolves transitively: chains of
            // rotate/clone all root at a decrypted node. mag_borrows is in record order,
            // so a single forward pass suffices (a copy's source precedes it).
            // Every stat is invariant under the permutation class (automorphism/conjugate/
            // level move keep |slot| max, the Re mean over lanes, max|v-mean| and every
            // coefficient magnitude); negate flips the mean only.
            for (const auto& b : mag_borrows) {
                if (b.copy_from >= 0) {
                    auto it = mag_results.find((std::size_t)b.copy_from);
                    if (it != mag_results.end()) {
                        MagStats s = it->second;
                        if (b.negate) s.mean = -s.mean;
                        mag_results[b.node_idx] = s;
                        apply_mag(g.get(), b.node_idx, s);
                    }
                } else {
                    auto it = mag_ref_async.find(b.borrow);
                    if (it != mag_ref_async.end()) {
                        mag_results[b.node_idx] = it->second;
                        apply_mag(g.get(), b.node_idx, it->second);
                    }
                }
            }
            std::fprintf(stderr,
                         "[mag] items=%llu copies=%llu store_ms=%llu backpressure_ms=%llu\n",
                         (unsigned long long)mag_stat_items.load(),
                         (unsigned long long)mag_stat_copies.load(),
                         (unsigned long long)(mag_stat_store_us.load() / 1000),
                         (unsigned long long)(mag_stat_backpressure_us.load() / 1000));
            mag_stat_store_us = 0; mag_stat_backpressure_us = 0;
            mag_stat_items = 0; mag_stat_copies = 0;
            mag_results.clear();
            mag_borrows.clear();
            lane_report();
            mag_node_of.clear();
            // NOTE: mag_ref_async intentionally survives the drain — reuse scopes can
            // span block boundaries only within one scope lifetime, and scopes reset
            // ordinals; stale entries are overwritten by the next iter-0 pass.
        }
        mag_cv_push.notify_all();
    }

    uint32_t level_limit() const {
        if (auto_bts_level_override != 0) {
            // env AUTO_BTS_LEVEL is PRIME-granular by convention (composite users pass
            // prime counts, e.g. 48 on a 54-tower d=2 chain).
            return auto_bts_level_override;
        }
        // total_depth counts CKKS LEVELS -> convert the 2-level headroom to primes.
        return static_cast<uint32_t>(composite_degree * (total_depth - 2));
    }

    // `k` CKKS LEVELS of headroom below the reactive ceiling, in the PRIME-granular units
    // every threshold here is expressed in. `level_limit()` counts primes, so a bare
    // `level_limit() - 3` would reserve 3 PRIMES = 1.5 CKKS levels on a d=2 chain rather
    // than 3 levels. Identity at d=1.
    int level_headroom(int levels) const {
        return static_cast<int>(level_limit()) - levels * composite_degree;
    }

    uint32_t bootstrap_output_level() {
        if (bts_out_level_set_) return bts_out_level_;

        // btp_overhead counts CKKS LEVELS; the returned value is a PRIME-granular level
        // (it feeds level_of comparisons and encode pins). The probe below is the
        // authoritative, unit-correct source; this formula is only its fallback.
        const uint32_t formula = static_cast<uint32_t>(composite_degree) *
            (static_cast<uint32_t>(btp_overhead) + (bts_iterations >= 2 ? 1u : 0u));

        uint32_t result = formula;
        try {
            auto pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(1, 0.0));
            Ctx  ct = cc->Encrypt(keys.publicKey, pt);
            inner_bootstrap(ct);  // applies the configured bts_iterations
            const uint32_t probed = level_of(ct);
            result = probed;
            std::cerr << "[bts-level] btp_overhead=" << btp_overhead
                      << " bts_iterations=" << bts_iterations
                      << " params_formula=" << formula << " probed=" << probed
                      << " -> encoding weights at level " << result << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[bts-level] probe failed (" << e.what()
                      << "); using params formula " << formula << "\n";
        } catch (...) {
            std::cerr << "[bts-level] probe failed; using params formula "
                      << formula << "\n";
        }
        bts_out_level_     = result;
        bts_out_level_set_ = true;
        return result;
    }

    // ct *= i via the monomial x^{N/2}: deg- and level-preserving, no keyswitch. Recorded so
    // token-pair captures see the lane swap (decode never calls it).
    Ctx mult_i(const Ctx& ct) {
        const packtag::PackTag _tg = tag_of_ct(ct);
        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        Ctx out_ct = ct->Clone();
        cc->EvalMultMonomialInPlace(out_ct, static_cast<uint32_t>(cc->GetRingDimension() / 2));
        const std::string out = set_new_var_for_ct(out_ct);
        tag_ct(out_ct, _tg);   // before the graph node is stamped and a planted bootstrap can fire
        const int out_level = level_for_ct(out_ct);
        record_primitive("mult_i", {in_a}, out, {in_a_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "mult_i", level_of(out_ct));
        tag_ct(out_ct, _tg);
        return out_ct;
    }

    Ctx pair_pack(const Ctx& a_re, const Ctx& b_im) {
        const packtag::PackTag _tg = packtag::t_add(tag_of_ct(a_re), tag_of_ct(b_im));
        Ctx b = mult_i(b_im);   // recorded ops (raw EvalAdd left the output var untracked -> anon graph inputs)
        return add(a_re, b);
    }

    std::pair<Ctx, Ctx> pair_unpack(const Ctx& c) {
        Ctx conj = cc->EvalConjugate(c);
        Ctx re   = cc->EvalAdd(c, conj);                                                    // 2a
        Ctx im   = cc->EvalSub(conj, c);                                                    // -2i*b
        cc->EvalMultMonomialInPlace(im, static_cast<uint32_t>(cc->GetRingDimension() / 2));  // *i -> 2b
        cc->EvalMultInPlace(re, 0.5);
        cc->EvalMultInPlace(im, 0.5);
        return {re, im};
    }

    void bootstrap_pair(Ctx& a, Ctx& b) {
        { bootstrap(a); bootstrap(b); return; }
        if (!complex_payload) { bootstrap(a); bootstrap(b); return; }
        const auto mono = static_cast<uint32_t>(cc->GetRingDimension() / 2);
        cc->EvalMultInPlace(a, 0.5);
        cc->EvalMultInPlace(b, 0.5);
        cc->EvalMultMonomialInPlace(b, mono);   // i·b/2
        cc->EvalAddInPlace(a, b);               // a := 0.5·(a + i·b)
        bootstrap(a);                           // the ONE recorded, planable op
        Ctx pc = cc->EvalConjugate(a);          // 0.5·(a − i·b)
        b = cc->EvalSub(pc, a);                 // −i·b
        cc->EvalMultMonomialInPlace(b, mono);   // ×i → b (deg-preserving)
        cc->EvalAddInPlace(a, pc);              // 2·Re → a (deg-preserving)
    }

    void bootstrap(Ctx& ct) {
        const int in_level = level_for_ct(ct);
        const std::string in = var_for_ct(ct);

        inner_bootstrap(ct);

        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("deliberate_bootstrap", {in}, out, {in_level}, out_level, ct);
    }

    // Per-THREAD override of bts_iterations (0 = none). Per-thread rather than saved and
    // restored on the shared `bts_iterations` member, which races as soon as two worker
    // threads open a scope concurrently. Setup-time readers (bootstrap_output_level(), cached
    // and computed before any scope opens) keep reading the member — that reservation is
    // process-wide and must NOT follow a per-thread scope.
    inline static thread_local uint32_t bts_iters_override_ = 0;
    uint32_t effective_bts_iters() const {
        return bts_iters_override_ ? bts_iters_override_ : bts_iterations;
    }

    struct BtsItersScope {
        uint32_t saved;
        BtsItersScope(CKKSContext&, uint32_t n)
            : saved(bts_iters_override_) { bts_iters_override_ = n; }
        ~BtsItersScope() { bts_iters_override_ = saved; }
    };

    // ── FUSED REDUCTION ─────────────────────────────────────────────────────────────────
    // A sparse bootstrap at slot count `s` opens with Accumulate, an orthogonal projection
    // onto the s-periodic subspace WITH MEAN NORMALISATION. That projection is mandatory and
    // already paid for. If the value you want downstream IS that mean, the bootstrap has
    // computed your reduction for free: a `rotate_and_sum` ladder plus a later bootstrap
    // collapse into one sparse bootstrap that is itself cheaper than the dense one it replaces.
    //
    // Arithmetic (n live values, S slots, copies = S/s):
    //     fold output = (Σ live)/copies      EvalMod sees `wanted · n/copies`
    //     recovery    = copies/n             <- a constant; restores (Σ live)/n
    // The shrink by `copies` is what buys the accuracy: EvalMod error goes as m³ (arcsine
    // off), so shrinking the input divides error by copies³ while recovery multiplies back
    // by copies.
    //
    // The fold is DESTRUCTIVE — everything except the window means is gone. Only call this
    // where the ciphertext carries the reduction target and NOTHING else live; wrong use
    // returns plausible values with nothing thrown.
    // It is NOT value-preserving, so it records its own op type. Never fold under
    // `deliberate_bootstrap`/`auto_bootstrap`: the planner's level accounting, the
    // [plan_level_error] gate and unplanned_bts all assume a bootstrap preserves the value.
    struct FoldBtsScope {
        CKKSContext& c;
        uint32_t saved;
        FoldBtsScope(CKKSContext& ctx, uint32_t s)
            : c(ctx), saved(ctx.sparse_bts_active) { c.sparse_bts_active = s; }
        ~FoldBtsScope() { c.sparse_bts_active = saved; }
    };

    // Fold `ct` at slot count `s`, reducing `n_live` values, and restore the wanted scale.
    // `s` must be a power of two in [1, slots]; s=1 is the full all-reduce (every slot
    // summed, broadcast to every slot).
    //
    // The recovery constant is derived HERE, from the same ring-derived slot count the fold
    // itself divides by, rather than taken from the caller: `copies` must match what
    // Accumulate actually did, and a caller computing it from `inf.slots` would be silently
    // wrong whenever batch_size != 0. Callers pass the one thing only they know — how many
    // live values the reduction covers (e.g. rD for the LN variance).
    // `prescale` (default 1) shrinks the FOLDED value in front of EvalMod for FREE (it
    // rides the bootstrap's own constantEvalMult) and its inverse rides the ONE recovery
    // mult — zero extra levels on either side. Needed when copies/n_live gives little
    // shrink (diagonal LN var: copies=1024 vs n_live=768 ⇒ EvalMod would see 0.75·var;
    // the arcsine-off EvalMod error goes as m³, so target folded·prescale ≈ 0.1–0.5,
    // staying above the kBtsSafeLo=0.01 precision floor).
    void fold_bootstrap(Ctx& ct, uint32_t s, int n_live, double prescale = 1.0) {
        if (n_live <= 0)
            throw std::runtime_error("fold_bootstrap: n_live must be > 0, got "
                                     + std::to_string(n_live));
        if (!(prescale > 0.0))
            throw std::runtime_error("fold_bootstrap: prescale must be > 0");
        const uint32_t S = static_cast<uint32_t>(cc->GetRingDimension() / 2);
        if (s == 0 || s > S || (s & (s - 1)) != 0)
            throw std::runtime_error("fold_bootstrap: slot count must be a power of two in [1, "
                                     + std::to_string(S) + "], got " + std::to_string(s));
        // A sparse bootstrap needs a PRECOMP at exactly that slot count (setup builds one
        // per SPARSE_BTS_SLOTS list entry). Folding without it returns garbage silently,
        // so refuse instead. Callers wanting a not-built s should ask fold_slots_for()
        // and pre-ladder the gap.
        if (s != S && !sparse_precomp_slots.count(s)) {
            std::string built;
            for (uint32_t b : sparse_precomp_slots) built += (built.empty() ? "" : ",") + std::to_string(b);
            throw std::runtime_error(
                "fold_bootstrap: no bootstrap precomputation for s=" + std::to_string(s) +
                " (built: {" + (built.empty() ? "none" : built) +
                "}). Add " + std::to_string(s) + " to SPARSE_BTS_SLOTS, or use "
                "fold_slots_for() and pre-ladder to a built slot count.");
        }

        const double recovery = (double)(S / s) / (double)n_live / prescale;   // copies/n/p

        const int in_level = level_for_ct(ct);
        const std::string in = var_for_ct(ct);
        // INPUT-SIDE prescale: the factor must multiply the ciphertext BEFORE ModRaise (the
        // bootstrap's own constant also normalizes the q·I multiples onto the sine period,
        // so an extra factor there would de-align EvalMod). Costs one level at the fold
        // input; the inverse rides the single recovery mult.
        if (prescale != 1.0) {
            cc->EvalMultInPlace(ct, prescale);
            tag_ct(ct, packtag::t_mult_scalar(tag_of_ct(ct), prescale));
        }
        {
            FoldBtsScope fs(*this, s);
            inner_bootstrap(ct);
        }
        // Costs the one level the reduction's own 1/n scaling would have cost — no extra.
        // RAW EvalMultInPlace on purpose: the recording inplace_mult would emit an orphan
        // mult node whose input sits at the deep pre-fold level, and the planner's budget
        // check would flag it as a violation. The level change lands in THIS fold node's
        // recorded output level instead (one node in = one node out, capture and planned
        // runs agree on the var sequence).
        if (recovery != 1.0) {
            cc->EvalMultInPlace(ct, recovery);
            tag_ct(ct, packtag::t_mult_scalar(tag_of_ct(ct), recovery));
        }
        const std::string out = set_new_var_for_ct(ct);
        record_primitive("fold_bootstrap", {in}, out, {in_level}, level_for_ct(ct), ct);
        if (graph_enabled()) {
            if (GraphNode* n = graph_builder->last_node()) {
                n->has_fold       = true;
                n->fold_stride    = static_cast<double>(s);
                n->recovery_const = recovery;
            }
        }
    }

    // Clears + arms the op tally for its lifetime (counts land in op_tally).
    struct OpTallyScope {
        CKKSContext& c;
        bool saved;
        explicit OpTallyScope(CKKSContext& ctx) : c(ctx), saved(ctx.op_tally_active) {
            c.op_tally.clear();
            c.op_tally_active = true;
        }
        ~OpTallyScope() { c.op_tally_active = saved; }
    };

    struct BtsPrecisionScope {
        CKKSContext& c;
        uint32_t saved;
        BtsPrecisionScope(CKKSContext& ctx, uint32_t p)
            : c(ctx), saved(ctx.bts_precision) { c.bts_precision = p; }
        ~BtsPrecisionScope() { c.bts_precision = saved; }
    };

    struct ArcsineScope {
        ArcsineScope(bool on) { FIDESlib::CKKS::setArcsineOverride(on ? 1 : 0); }
        ~ArcsineScope() { FIDESlib::CKKS::setArcsineOverride(-1); }
    };

    struct SparseBtsScope {
        CKKSContext& c;
        uint32_t saved;
        SparseBtsScope(CKKSContext& ctx, bool armed = true)
            : c(ctx), saved(ctx.sparse_bts_active) {
            if (armed) c.sparse_bts_active = c.sparse_bts_slots;
        }
        ~SparseBtsScope() { c.sparse_bts_active = saved; }
    };

    // Per-bootstrap CORRECTION_FACTOR override (range<->precision dial).
    // The correction factor is runtime-only — nothing precomputed (keys, CtS/StC matrices,
    // level accounting) depends on it, and its 2^-c raise adjust / 2^c restore pair is
    // self-contained inside one Bootstrap call — so different bootstraps in one execution
    // can safely use different values (e.g. CF=3 on |m|<=1 placements, CF=7 near the
    // |m|<=10 wall). Arms ContextData::correctionFactorOverride around the scoped calls;
    // same single-threaded scoping discipline as SparseBtsScope. cf < deg still throws
    // the Bootstrap deg-guard, per call. No-op before LoadContext.
    // Level-aware ModRaise for one planted bootstrap: the raise stops `drop` composite levels below the
    // chain top (plan 'raise_drop'), so every stage runs on fewer limbs and the refresh lands that much
    // deeper -- exactly the room the plan's simulation gave this site. Needs the matching DFT plaintext
    // variant built at LoadContext (FIDESLIB_BTS_RAISE_DROPS); FIDESlib throws otherwise.
    struct RaiseScope {
        FIDESlib::CKKS::ContextData* g = nullptr;
        int saved = 0;
        RaiseScope(CKKSContext& ctx, int drop) {
            if (drop <= 0 || !ctx.cc->gpu.has_value()) return;
            g = std::any_cast<FIDESlib::CKKS::Context&>(ctx.cc->gpu).get();
            saved = g->getBtsRaiseDrop();
            g->setBtsRaiseDrop(drop);
        }
        ~RaiseScope() {
            if (g) g->setBtsRaiseDrop(saved);
        }
    };

    struct CorrectionScope {
        FIDESlib::CKKS::ContextData* g = nullptr;
        int saved = -1;
        // NOTE: MUST go through the out-of-line accessors — ContextData has #ifdef NCCL
        // members, so direct field access from wrapper-compiled TUs can hit the wrong
        // offset when the NCCL define differs from the fideslib build.
        CorrectionScope(CKKSContext& ctx, int cf, bool armed = true) {
            if (!armed || !ctx.cc->gpu.has_value()) return;
            g = std::any_cast<FIDESlib::CKKS::Context&>(ctx.cc->gpu).get();
            saved = g->getCorrectionFactorOverride();
            g->setCorrectionFactorOverride(cf);
        }
        ~CorrectionScope() {
            if (g) g->setCorrectionFactorOverride(saved);
        }
    };

    void bootstrap_precise(Ctx& ct) {
        WithStep _w(*this, "bootstrap_precise");
        const int in_level = level_for_ct(ct);
        const std::string in = var_for_ct(ct);

        const uint32_t sp = sparse_bts_active;
        if (sp) ct->SetSlots(sp);
        ct = eval_bootstrap_iter(ct, 2, bts_precision);
        if (sp) ct->SetSlots(cc->GetRingDimension() / 2);

        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("deliberate_bootstrap", {in}, out, {in_level}, out_level, ct);
    }

    // SPARSE_AUTO — route EVERY bootstrap sparse wherever the static tag proves it legal,
    // not only inside the hand-declared scopes (SPARSE_LN_BTS / SPARSE_SM_BTS /
    // CUTMAX_SPARSE_BTS), which cannot cover the reactive bootstraps. Default 2.
    //
    //  The eligibility query is `periodic_at`, NOT `min_routable_s()`. min_routable_s()
    // qualifies by `periodic_at(s) || fold_collision_free_at(s)`, and those license DIFFERENT
    // operations: only periodicity makes a plain SetSlots(s) refresh valid, while the
    // collision-free case licenses a FOLD bootstrap (it sums residue classes and needs a
    // recovery factor). Routing a fold-only ciphertext through a plain sparse bootstrap
    // returns plausible garbage with nothing thrown.
    //
    // Data of period p is also kp-periodic, so any built precomp that is a MULTIPLE of p is a
    // legal route; the smallest one is best (cheapest, lowest EvalMod noise floor). An
    // explicit scope still wins: if a caller armed one, it knows something the tag does not.
    // `fold` must travel with `s` rather than living in a member: inner_bootstrap runs on
    // worker threads, and applying the fold restore (×copies + mask) to a ciphertext that
    // was routed PERIODIC would corrupt it by exactly `copies`. A shared flag would race,
    // and the failure would be silent.
    struct SparseRoute {
        uint32_t s = 0;
        bool fold = false;                 // false = periodic (identity refresh, no restore)
        packtag::Support support{};        // the input's support, for the output tag
        // An EXPLICIT SparseBtsScope / FoldBtsScope route, not a tag-driven one:
        // fold_bootstrap's input is supposed to be aperiodic and its caller applies its own
        // copies/n_live recovery.
        bool caller_managed = false;
    };

    SparseRoute auto_sparse_route_for(const Ctx& ct) {
        // explicit scope wins — and is flagged caller_managed so the gate does not judge it
        if (sparse_bts_active) return {sparse_bts_active, false, {}, true};
        // SPARSE_AUTO=0 (default) -> off; 1 -> periodic route only (identity refresh);
        // 2 -> min_routable_s(), i.e. ALSO the fold-collision-free route (the 32-bit preset);
        // 3 -> additionally the masked fold route.
        static const int mode = [] {
            const char* e = std::getenv("SPARSE_AUTO");
            return (e && *e) ? std::atoi(e) : 0;
        }();
        if (mode <= 0 || sparse_precomp_slots.empty()) return {};
        const packtag::PackTag t = tag_of_ct(ct);
        // Census FIRST, so the report explains where the UNROUTED bootstraps go. The three
        // populations are completely different problems: "no tag" is a propagation gap in
        // packtag, "dense" is genuinely aperiodic data, and "periodic but snapped to S" means
        // the period is real but no precomp was BUILT near it — a SPARSE_BTS_SLOTS config
        // limit, not a tag limit.
        if (!t.known()) {
            ++auto_sparse_untagged;
            // Which steps reach a bootstrap with an unknown tag (a ranked work list).
            if (auto_sparse_unknown_at.size() < 4096) ++auto_sparse_unknown_at[step_path()];
        } else if (t.period >= t.slots) {
            ++auto_sparse_dense;
        } else {
            ++auto_sparse_periodic_seen[t.period];
        }
        if (!t.known()) return {};                         // unknown == top == dense
        const uint32_t S = static_cast<uint32_t>(cc->GetRingDimension() / 2);
        uint32_t want;
        if (mode >= 2) {
            want = static_cast<uint32_t>(t.min_routable_s());
        } else {
            want = 1;
            while (want < static_cast<uint32_t>(t.period)) want <<= 1;   // next pow2 >= period
        }
        const uint32_t s = fold_slots_for(want);
        if (s >= S || !sparse_precomp_slots.count(s)) return {};
        // In mode 1 assert the multiple-of-period contract rather than trusting
        // fold_slots_for — a wrong route here is silent. Mode 2 deliberately admits the
        // fold route, which is NOT an identity refresh (it sums residue classes and lands
        // values at i mod s), so the histogram below separates the two populations.
        const bool periodic = t.periodic_at(static_cast<int>(s));
        if (mode < 2 && !periodic) return {};
        // MODE 3 ONLY — the masked fold route. Eligibility is COLLISION-FREEDOM: the fold
        // output is s-periodic, so each class's value is available at every congruent slot
        // including the one it came from, and a support-shaped mask picks it up in place
        // (fold_restore_pt). The price is structural: the fold returns v/copies and the
        // restore multiplies `copies` back, so the bootstrap's own error is amplified by S/s.
        if (mode >= 3 && !periodic && !t.support.is_empty() && !t.support.is_dense()) {
            const uint32_t fs = s;
            if (t.fold_collision_free_at(static_cast<int>(fs))) {
                ++auto_sparse_routed;
                ++auto_sparse_foldonly;
                ++auto_sparse_hist[fs];
                return {fs, true, t.support};
            }
        }
        // Refused: the support is dense or empty, or it collides mod s (two live values would
        // be summed into one, which no restore can separate). In mode < 3 every non-periodic
        // site lands here by design.
        if (!periodic) {
            ++auto_sparse_foldonly_refused;
            if (auto_sparse_refused_at.size() < 4096) ++auto_sparse_refused_at[step_path()];
            return {};
        }
        ++auto_sparse_routed;
        ++auto_sparse_periodic;
        ++auto_sparse_hist[s];
        return {s, false, t.support};
    }
    uint32_t auto_sparse_routed   = 0;   // how many bootstraps the tag alone re-routed
    uint32_t auto_sparse_periodic = 0;   // ... of which qualified by periodicity (identity)
    uint32_t auto_sparse_foldonly = 0;   // ... of which only by fold-collision-freedom
    uint32_t auto_sparse_foldonly_refused = 0;   // collision-free but not routed (see above)
    std::map<std::string, uint32_t> auto_sparse_refused_at;
    std::map<uint32_t, uint32_t> auto_sparse_hist;   // routed: s -> count
    uint32_t auto_sparse_untagged = 0;   // no usable tag at the bootstrap input
    uint32_t auto_sparse_dense    = 0;   // tagged, but period == slots (genuinely aperiodic)
    std::map<int, uint32_t> auto_sparse_periodic_seen;   // ALL sub-dense periods seen -> count
    std::map<std::string, uint32_t> auto_sparse_unknown_at;   // step_path -> unknown-tag count

    // ── the fold route's restore: ONE plaintext multiply doing BOTH jobs ────────────────────
    // A sparse bootstrap at s on fold-transparent data does two things that must be undone:
    //   (1) it divides by `copies = S/s` — the fold sums each residue class, and a transparent
    //       support puts exactly one live value in each, so slot i comes back as v[i]/copies;
    //   (2) it leaves REPLICAS above s — the output is s-periodic by construction, so the
    //       zeros the input had at i >= s are now copies of v[i mod s].
    // A scalar ×copies fixes (1) and not (2). A mask fixes (2) and not (1). One vector fixes
    // both, for the price of one plaintext multiply — the level a scalar recovery would have
    // cost on its own.
    //
    // ── WHICH SLOTS THE MASK KEEPS ──────────────────────────────────────────────────────────
    // The fold output is s-PERIODIC (t_bootstrap's contract), so the class mean sits at EVERY
    // slot congruent to r mod s; the mask keeps the slot the value CAME FROM. Keeping the
    // support itself admits any collision-free support wherever it sits in the ring, with no
    // rotation and no extra keys.
    //
    // Over-approximation stays safe: if the tag's support is a superset of the truth AND
    // is collision-free at s, then every extra slot the mask keeps
    // is alone in its residue class among tag-support slots, so its class contains NO live
    // value and it comes back 0 — not spurious data.
    Ptx fold_restore_pt(uint32_t s, uint32_t level, double prescale,
                        const packtag::Support& sup) {
        const uint32_t S = static_cast<uint32_t>(cc->GetRingDimension() / 2);
        const double copies = static_cast<double>(S) / static_cast<double>(s);
        const double val = copies / prescale;
        char key[176];
        std::snprintf(key, sizeof(key), "%u|%u|%.17g|%d|%d|%d|%d|%d", s, level, val,
                      (int)sup.kind, sup.offset, sup.stride, sup.count, sup.width);
        {
            std::lock_guard<std::mutex> lk(fold_pt_mtx);
            auto it = fold_restore_cache.find(key);
            if (it != fold_restore_cache.end()) return it->second;
        }
        std::vector<double> m(S, 0.0);
        if (sup.kind == packtag::Support::Kind::AP && sup.count > 0) {
            for (int k = 0; k < sup.count; ++k) {
                for (int j = 0; j < sup.width; ++j) {   // Block support: width slots per cell
                    const long long idx =
                        ((long long)sup.offset + (long long)k * sup.stride + j) % (long long)S;
                    m[(size_t)((idx % S + S) % S)] = val;
                }
            }
        } else {
            // No AP to place (dense support never reaches here — it is collision-free only at
            // s >= slots, which the router excludes). Fall back to the [0,s) window.
            for (uint32_t i = 0; i < s; ++i) m[i] = val;
        }
        Ptx pt = cc->MakeCKKSPackedPlaintext(m, /*noiseScaleDeg=*/1, level);
        tag_plaintext(pt, m);
        std::lock_guard<std::mutex> lk(fold_pt_mtx);
        fold_restore_cache.emplace(key, pt);
        return pt;
    }
    std::mutex fold_pt_mtx;
    std::unordered_map<std::string, Ptx> fold_restore_cache;

    // Sparse-routing coverage report at teardown. The sparse-lane total counts every
    // bootstrap that ran on a sparse precomp, via SPARSE_AUTO or a hand-declared scope:
    // routing changes cost, not the bootstrap count, so without it an armed scope and a
    // dead one would print identically.
    void report_auto_sparse() const {
        if (total_bootstraps_sparse_lane || sparse_bts_slots)
            std::fprintf(stderr,
                         "[sparse_lane] sparse_routed_bts=%u of total_bts=%u  scope_default_s=%u\n",
                         total_bootstraps_sparse_lane, total_bootstraps, sparse_bts_slots);
        if (!auto_sparse_routed && auto_sparse_hist.empty()) return;
        std::fprintf(stderr,
                     "[sparse_auto] routed=%u periodic=%u fold_only=%u of total_bts=%u  s:",
                     auto_sparse_routed, auto_sparse_periodic, auto_sparse_foldonly,
                     total_bootstraps);
        for (const auto& [s, n] : auto_sparse_hist)
            std::fprintf(stderr, " %u=%u", s, n);
        std::fprintf(stderr, "\n");
        // Where the UNROUTED ones went. `periods` lists every sub-dense period the tags
        // actually carried: any period with no built precomp near it is coverage lost to
        // SPARSE_BTS_SLOTS, not to the tagging.
        if (auto_sparse_foldonly_refused) {
            std::fprintf(stderr,
                         "[sparse_refused] fold_only=%u refused: collision-free but NOT "
                         "fold-transparent, i.e. the fold would RELOCATE these values to i mod s "
                         "and no restore multiply can undo a permutation. NOT a missing feature "
                         "— mostly rotated lanes living high in the ring.\n",
                         auto_sparse_foldonly_refused);
            std::vector<std::pair<uint32_t, std::string>> r;
            r.reserve(auto_sparse_refused_at.size());
            for (const auto& [path, n] : auto_sparse_refused_at) r.emplace_back(n, path);
            std::sort(r.rbegin(), r.rend());
            for (size_t i = 0; i < r.size() && i < 15; ++i)
                std::fprintf(stderr, "[sparse_refused] %6u  %s\n", r[i].first, r[i].second.c_str());
        }
        std::fprintf(stderr, "[sparse_census] untagged=%u dense=%u  periods:",
                     auto_sparse_untagged, auto_sparse_dense);
        for (const auto& [p, n] : auto_sparse_periodic_seen)
            std::fprintf(stderr, " %d=%u", p, n);
        std::fprintf(stderr, "  built:");
        for (uint32_t s : sparse_precomp_slots) std::fprintf(stderr, " %u", s);
        std::fprintf(stderr, "\n");
        // The work list: which steps reach a bootstrap with an UNKNOWN tag, worst first.
        std::vector<std::pair<uint32_t, std::string>> v;
        v.reserve(auto_sparse_unknown_at.size());
        for (const auto& [path, n] : auto_sparse_unknown_at) v.emplace_back(n, path);
        std::sort(v.rbegin(), v.rend());
        for (size_t i = 0; i < v.size() && i < 25; ++i)
            std::fprintf(stderr, "[sparse_unknown] %6u  %s\n", v[i].first, v[i].second.c_str());
        std::fflush(stderr);
    }

    void inner_bootstrap(Ctx& ct, double prescale = 1.0) {
        WithStep _w(*this, "bootstrap");
        const SparseRoute route = auto_sparse_route_for(ct);
        const uint32_t sp = route.s;
        // The prescale/restore pair assumes ONE bootstrap sees the factor; the iterative
        // path bootstraps residuals against the unscaled input, which would silently
        // desynchronise. Refuse loudly (planned prescale sites run BTS_ITERATIONS=1).
        const uint32_t eff_iters = effective_bts_iters();
        if (prescale != 1.0 && eff_iters > 1)
            throw std::runtime_error(
                "inner_bootstrap: prescale != 1 is unsupported with bts_iterations > 1");
        const uint32_t bts_before = total_bootstraps;
        // DEPTH GUARD. A bootstrap started past the measured envelope returns garbage
        // silently. Plan-time cannot catch the hint-fired ones (the level is decided here),
        // so this is the only place that sees every case.
        {
            // Chain-relative: ONE CKKS level above the reactive ceiling. On the 32-bit
            // composite chain that is 46 + 2 = 48 primes, the level the round-trip error was
            // measured at (rel_err 3.7e-1 at 48 -> 1.96e4 at 50). A bare 48 means nothing on
            // a d=1 chain whose levels stop at 24, so express it in the prime-granular units
            // `level_headroom` already exists for. BTS_MAX_INPUT_LEVEL overrides (primes).
            static const int kEnvOverride = [] {
                const char* e = std::getenv("BTS_MAX_INPUT_LEVEL");
                return (e && *e) ? std::atoi(e) : -1;
            }();
            const int kMaxBtsInputLevel =
                (kEnvOverride >= 0) ? kEnvOverride : level_headroom(-1);
            const int lvl = static_cast<int>(level_of(ct));
            if (lvl > kMaxBtsInputLevel) {
                std::ostringstream oss;
                oss << "[bts_depth_error] bootstrap input level " << lvl << " > " << kMaxBtsInputLevel
                    << " (measured envelope) at step=" << step_path()
                    << " var=" << var_for_ct(ct)
                    << " — a refresh started this deep returns garbage silently. "
                       "If this came from a HINT, re-plan disregarding it "
                       "(--hint-veto-steps / hint_force={var: false}); if from a placement, "
                       "lower --max-level.";
                throw std::runtime_error(oss.str());
            }
        }
        // INPUT-SIDE prescale — must precede ModRaise (a factor on the bootstrap's own
        // constant would de-align the q multiples from the sine period). Costs one level
        // at the bootstrap input; the planner only emits prescale where that level
        // exists. Raw mult: no var-counter advance (planned-mode name alignment).
        if (prescale != 1.0) {
            cc->EvalMultInPlace(ct, prescale);
            tag_ct(ct, packtag::t_mult_scalar(tag_of_ct(ct), prescale));
        }
        if (sp) ct->SetSlots(sp);
        if (eff_iters > 1) {
            ct = eval_bootstrap_iter(ct, eff_iters, bts_precision);
        } else {
            cc->EvalBootstrapInPlace(ct);
            total_bootstraps++;
        }
        if (sp) ct->SetSlots(cc->GetRingDimension() / 2);
        // Restore the value scale (the input side was free; this costs one level at the
        // refreshed output — the level the planner charges for a prescaled placement).
        // RAW EvalMultInPlace on purpose: the recording inplace_mult would advance the
        // sequential var counter, and this restore exists only in PLANNED runs — an
        // extra var name here would shift every later placement/level lookup off by one
        // against the captured graph. The level change is folded into the enclosing
        // bootstrap node's recorded output level instead.
        if (route.fold) {
            // The fold restore: one plaintext multiply carrying ×copies AND the mask (and the
            // 1/prescale if there is one) — see fold_restore_pt for why a scalar cannot do it.
            // `level_of(ct) + pending_rescale_primes(ct)`, NOT level_of alone: a plaintext
            // built at the un-adjusted level on a composite chain combines wrongly, invisibly
            // to any value-based probe. A fresh bootstrap output normally carries no pending
            // rescale, so this is expected to be a no-op here.
            Ptx rpt = fold_restore_pt(sp,
                                      static_cast<uint32_t>(level_of(ct))
                                          + pending_rescale_primes(ct),
                                      prescale, route.support);
            cc->EvalMultInPlace(ct, rpt);
        } else if (prescale != 1.0) {
            cc->EvalMultInPlace(ct, 1.0 / prescale);
            tag_ct(ct, packtag::t_mult_scalar(tag_of_ct(ct), 1.0 / prescale));
        }
        if (sp) total_bootstraps_sparse_lane += (total_bootstraps - bts_before);
        if (route.fold) {
            // NOT t_bootstrap. Its contract — "a sparse-routed bootstrap emits s-periodic
            // data whatever went in" — holds for the bare route but not here: the mask has
            // just zeroed everything at i >= s, so the output is not periodic at all. It is the
            // input's support, back in the input's slots. Claiming period=s on this would
            // under-estimate every downstream op's period.
            tag_ct(ct, packtag::PackTag{(int)(cc->GetRingDimension() / 2),
                                        (int)(cc->GetRingDimension() / 2),   // aperiodic = top
                                        route.support});
        } else {
            // A sparse-routed bootstrap emits s-periodic data whatever went in; a dense one is
            // value-preserving and carries the input tag through.
            tag_ct(ct, packtag::t_bootstrap(tag_of_ct(ct), (int)sp,
                                            (int)(cc->GetRingDimension() / 2)));
        }
    }

    Ctx eval_bootstrap_iter(const Ctx& ct, uint32_t numIterations = 1, uint32_t precision = 0) {
        // BTS_RAISE_VARIANT=k (harness experiments only): run this bootstrap on the route's raise variant k
        static const int raise_variant = [] { const char* e = std::getenv("BTS_RAISE_VARIANT"); return e && *e ? std::atoi(e) : 0; }();
        RaiseScope rs(*this, raise_variant);
        Ctx y = cc->EvalBootstrap(ct, /*numIterations=*/1, /*precision=*/0);
        ++total_bootstraps;
        if (numIterations <= 1) {
            return y;
        }

        const double scale_up   = std::pow(2.0, static_cast<double>(precision));
        const double scale_down = std::pow(2.0, -static_cast<double>(precision));

        for (uint32_t iter = 1; iter < numIterations; ++iter) {
            // Residual e = ct - y. Underlying EvalSub mod-aligns the operands.
            Ctx e         = cc->EvalSub(ct, y);
            Ctx e_scaled  = cc->EvalMult(e, scale_up);
            Ctx y_e       = cc->EvalBootstrap(e_scaled, /*numIterations=*/1, /*precision=*/0);
            ++total_bootstraps;
            Ctx y_e_back  = cc->EvalMult(y_e, scale_down);
            y             = cc->EvalAdd(y, y_e_back);
        }
        return y;
    }

    // Suppress the reactive safety-net bootstrap for a scope. ONLY legal where the
    // very next op is a fold_bootstrap on the same lane: the reactive would fire on
    // the (deep, often out-of-band) fold input and garble it.
    bool auto_bts_suppressed = false;
    struct AutoBtsSuppressScope {
        CKKSContext& c;
        bool saved;
        explicit AutoBtsSuppressScope(CKKSContext& ctx)
            : c(ctx), saved(ctx.auto_bts_suppressed) { c.auto_bts_suppressed = true; }
        ~AutoBtsSuppressScope() { c.auto_bts_suppressed = saved; }
    };

    void maybe_bootstrap(Ctx& ct) {
        FIDESlib::CudaNvtxRange _nvr_pw("pw::maybe_bts");
        if (auto_bts_suppressed) {
            return;
        }
        if (level_of(ct) < level_limit()) {
            return;
        }
        if (placement_plan_enabled && !graph_scope_active()) {
            // A ceiling-level ct the plan PREDICTED is not divergence: a planned
            // trajectory may legally touch level_limit when the next planned op
            // is the bootstrap itself (in the capture the auto fired here; the
            // cut carries it as the following deliberate/placement, e.g. the
            // fresh_recip deliberate after the GS chain). Only an UNPREDICTED
            // ceiling is a plan/run divergence.
            auto it = expected_levels.find(var_for_ct(ct));
            if (it != expected_levels.end() && it->second == level_of(ct)) {
                return;
            }
            ++unplanned_bootstrap_count;
            throw fhe::PlanError(
                "[plan_bts_error] unplanned safety-net bootstrap at step=" + step_path() +
                " var=" + var_for_ct(ct) + " level=" + std::to_string(level_of(ct)) +
                " (planned mode is strict)");
        }

        const int in_level = level_for_ct(ct);
        const std::string in = var_for_ct(ct);
        inner_bootstrap(ct);
        const std::string out = in + "_auto_bootstrapped";
        name_ct(ct, out, true);
        const int out_level = level_for_ct(ct);
        record_primitive("auto_bootstrap", {in}, out, {in_level}, out_level, ct);

        assert(level_of(ct) < level_limit());
    }

    void inplace_add(Ctx& ct, const Ctx& other) {
        const packtag::PackTag _tg = packtag::t_add(tag_of_ct(ct), tag_of_ct(other));
        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_ct(other);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_ct(other);
        cc->EvalAddInPlace(ct, other);
        tag_ct(ct, _tg);   // before the in-op bootstrap hook
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("add_inplace", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "add_inplace", level_of(ct));
        
        tag_ct(ct, _tg);
    }

    void inplace_add(Ctx& ct, Ptx& pt) {
        const packtag::PackTag _tg = packtag::t_add(tag_of_ct(ct), tag_of(pt));
        
        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_pt(pt);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_pt(pt);
        cc->EvalAddInPlace(ct, pt);
        tag_ct(ct, _tg);   // before the in-op bootstrap hook
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("add_inplace", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "add_inplace", level_of(ct));
        
        tag_ct(ct, _tg);
    }

    void inplace_add(Ctx& ct, double scalar) {
        const packtag::PackTag _tg = packtag::t_add_scalar(tag_of_ct(ct), scalar);
        
        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_scalar(scalar);
        cc->EvalAddInPlace(ct, scalar);
        tag_ct(ct, _tg);   // before the in-op bootstrap hook
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("add_inplace", {in_a, in_b}, out, {in_a_level, -1}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "add_inplace", level_of(ct));
        
        tag_ct(ct, _tg);
    }

    Ctx add(const Ctx& ct, const Ctx& other) {
        const packtag::PackTag _tg = packtag::t_add(tag_of_ct(ct), tag_of_ct(other));

        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_ct(other);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_ct(other);
        Ctx out_ct = cc->EvalAdd(ct, other);
        tag_ct(out_ct, _tg);   // before maybe_apply_planned_bootstrap_after can fire
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("add", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "add", level_of(out_ct));
        
        tag_ct(out_ct, _tg);
        return out_ct;
    }

    Ctx add(const Ctx& ct, Ptx& pt) {
        const packtag::PackTag _tg = packtag::t_add(tag_of_ct(ct), tag_of(pt));
        
        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_pt(pt);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_pt(pt);
        Ctx out_ct = cc->EvalAdd(ct, pt);
        tag_ct(out_ct, _tg);   // before maybe_apply_planned_bootstrap_after can fire
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("add", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "add", level_of(out_ct));
        
        tag_ct(out_ct, _tg);
        return out_ct;
    }

    Ctx add(const Ctx& ct, double scalar) {
        const packtag::PackTag _tg = packtag::t_add_scalar(tag_of_ct(ct), scalar);
        
        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_scalar(scalar);
        Ctx out_ct = cc->EvalAdd(ct, scalar);
        tag_ct(out_ct, _tg);   // before maybe_apply_planned_bootstrap_after can fire
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("add", {in_a, in_b}, out, {in_a_level, -1}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "add", level_of(out_ct));
        
        tag_ct(out_ct, _tg);
        return out_ct;
    }

    Ctx sub(const Ctx& ct, const Ctx& other) {
        const packtag::PackTag _tg = packtag::t_add(tag_of_ct(ct), tag_of_ct(other));

        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_ct(other);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_ct(other);
        Ctx out_ct = cc->EvalSub(ct, other);
        tag_ct(out_ct, _tg);   // before maybe_apply_planned_bootstrap_after can fire
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("sub_ct", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "sub_ct", level_of(out_ct));
        maybe_bootstrap(out_ct);
        
        tag_ct(out_ct, _tg);
        return out_ct;
    }

    Ctx sub(const Ctx& ct, Ptx& pt) {
        const packtag::PackTag _tg = packtag::t_add(tag_of_ct(ct), tag_of(pt));
        
        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_pt(pt);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_pt(pt);
        Ctx out_ct = cc->EvalSub(ct, pt);
        tag_ct(out_ct, _tg);   // before maybe_apply_planned_bootstrap_after can fire
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("sub", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "sub", level_of(out_ct));
        
        tag_ct(out_ct, _tg);
        return out_ct;
    }

    Ctx sub(const Ctx& ct, double scalar) {
        const packtag::PackTag _tg = packtag::t_add_scalar(tag_of_ct(ct), scalar);
        
        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_scalar(scalar);
        Ctx out_ct = cc->EvalSub(ct, scalar);
        tag_ct(out_ct, _tg);   // before maybe_apply_planned_bootstrap_after can fire
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("sub", {in_a, in_b}, out, {in_a_level, -1}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "sub", level_of(out_ct));
        
        tag_ct(out_ct, _tg);
        return out_ct;
    }

    void inplace_sub(Ctx& ct, const Ctx& other) {
        const packtag::PackTag _tg = packtag::t_add(tag_of_ct(ct), tag_of_ct(other));
        
        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_ct(other);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_ct(other);
        cc->EvalSubInPlace(ct, other);
        tag_ct(ct, _tg);   // before the in-op bootstrap hook
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("sub_inplace_ct", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "sub_inplace_ct", level_of(ct));
        maybe_bootstrap(ct);
        
        tag_ct(ct, _tg);
    }

    void inplace_sub(Ctx& ct, double scalar) {
        const packtag::PackTag _tg = packtag::t_add_scalar(tag_of_ct(ct), scalar);
        
        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_scalar(scalar);
        cc->EvalSubInPlace(ct, scalar);
        tag_ct(ct, _tg);   // before the in-op bootstrap hook
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("sub_inplace", {in_a, in_b}, out, {in_a_level, -1}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "sub_inplace", level_of(ct));
        
        tag_ct(ct, _tg);
    }

    Ctx mult(const Ctx& ct, const Ctx& other) {
        const packtag::PackTag _tg = packtag::t_mult(tag_of_ct(ct), tag_of_ct(other));

        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_ct(other);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_ct(other);
        if (op_tally_active) ++op_tally["mult_cc"];
        Ctx out_ct = cc->EvalMult(ct, other);
        tag_ct(out_ct, _tg);   // before maybe_apply_planned_bootstrap_after can fire
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("mult", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "mult", level_of(out_ct));
        maybe_bootstrap(out_ct);
        
        tag_ct(out_ct, _tg);
        return out_ct;
    }

    Ctx mult(const Ctx& ct, Ptx& pt) {
        const packtag::PackTag _tg = packtag::t_mult(tag_of_ct(ct), tag_of(pt));

        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_pt(pt);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_pt(pt);
        if (op_tally_active) ++op_tally["mult_pt"];
        Ctx out_ct = cc->EvalMult(ct, pt);
        tag_ct(out_ct, _tg);   // before maybe_apply_planned_bootstrap_after can fire
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("mult", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "mult", level_of(out_ct));
        maybe_bootstrap(out_ct);
        
        tag_ct(out_ct, _tg);
        return out_ct;
    }

    // ---- Batched lane ct×pt mults ----
    // Execution/recording split: mult_batch_exec issues ALL products in one fused facade
    // call (EvalMultPtBatch → one pointer-table kernel pass) with NO bookkeeping; the call
    // site then runs mult_finish per lane in the exact serial order, so var names, graph
    // nodes, packtags, planned-bootstrap application and level verification are
    // byte-identical to the serial loop. Guards live at the call site via lane_batch_usable().
    // Batch is usable only when it cannot change observable semantics: not during capture
    // (capture decrypts every node and must record magnitudes op-by-op), pts pre-encoded at
    // the input's level.
    bool lane_batch_usable(const Ctx& ct, const std::vector<Ptx>& pts) {
        static std::once_flag rej_announced;
        auto reject = [&](const std::string& why) {
            std::call_once(rej_announced, [&] { std::cerr << "[lane_batch] ct×pt fallback: " << why << "\n"; });
            return false;
        };
        if (graph_enabled()) return reject("capture mode");
        if (!ct) return reject("null ct");
        // NL2 inputs are fine: MultPtBatch mirrors serial multPt by rescaling ONE shared
        // copy, and encode_at_cached already targeted the post-rescale level.
        const int lvl = level_for_ct(ct) + static_cast<int>(pending_rescale_primes(ct));
        for (const auto& pt : pts)
            if (!pt || level_for_pt(pt) != lvl)
                return reject("pt level " + std::to_string(pt ? level_for_pt(pt) : -1) +
                              " != expected ct level " + std::to_string(lvl));
        return true;
    }
    std::vector<Ctx> mult_batch_exec(const Ctx& ct, std::vector<Ptx>& pts) {
        static std::once_flag announced;
        std::call_once(announced, [&] {
            std::cerr << "[lane_batch] active: fused ct×pt lane products, n=" << pts.size() << "\n";
        });
        return cc->EvalMultPtBatch(ct, pts);
    }
    // ---- Batched lane ct×ct accumulate ----
    // res += Σ_j vs[j]*ss[j] with ONE relinearization (EvalMultCtAccumBatch); the per-lane
    // node stream (mult + add_inplace) is replayed in serial order so var names and plan
    // verification are identical to the loop. Lane products are phantoms (never
    // materialized): their nodes are recorded with the no-ct overload at the analytic
    // level (== the operand level; mult does not rescale).
    bool mult_add_many_usable(const Ctx& res, const std::vector<const Ctx*>& vs,
                              const std::vector<const Ctx*>& ss) {
        static std::once_flag rej_announced;
        auto reject = [&](const std::string& why) {
            std::call_once(rej_announced, [&] { std::cerr << "[lane_batch] ct×ct fallback: " << why << "\n"; });
            return false;
        };
        if (graph_enabled()) return reject("capture mode");
        if (!res) return reject("null res");
        if (res->GetNoiseScaleDeg() != 2)
            return reject("res NoiseScaleDeg=" + std::to_string(res->GetNoiseScaleDeg()));
        // NL2 or shallower operands are fine: multAccumulateBatch runs the SERIAL
        // adjustForMult machinery per lane on copies (rescale + scalar scale-correction +
        // drop). Only an operand DEEPER than the seed cannot be closed — reject that.
        const int lvl = level_for_ct(res);
        auto eff_level = [&](const Ctx& x) {
            return level_for_ct(x) + static_cast<int>(pending_rescale_primes(x));
        };
        for (auto* v : vs)
            if (!v || !*v || eff_level(*v) > lvl)
                return reject("v lane NL=" + std::to_string(v && *v ? (int)(*v)->GetNoiseScaleDeg() : -1) +
                              " eff_lvl=" + std::to_string(v && *v ? eff_level(*v) : -1) +
                              " deeper than res lvl=" + std::to_string(lvl));
        for (auto* s : ss)
            if (!s || !*s || eff_level(*s) > lvl)
                return reject("score lane NL=" + std::to_string(s && *s ? (int)(*s)->GetNoiseScaleDeg() : -1) +
                              " eff_lvl=" + std::to_string(s && *s ? eff_level(*s) : -1) +
                              " deeper than res lvl=" + std::to_string(lvl));
        // Serial maybe_bootstrap would fire on a ceiling-level lane product — phantoms
        // cannot be bootstrapped, so fall back (never true for cache-level lanes).
        if (!auto_bts_suppressed && !vs.empty() && level_of(*vs.front()) >= level_limit()) return false;
        // Planned-bootstrap pre-flight: peek the 2n upcoming var names; only the FINAL add
        // output may carry a placement (it fires after the batch, exactly as in serial).
        if (placement_plan_enabled && naming_active()) {
            const size_t n2 = 2 * vs.size();
            for (size_t k = 1; k <= n2 - 1; ++k) {
                const std::string name = graph_var_scope.empty()
                    ? "v_" + std::to_string(graph_ct_counter + k)
                    : (graph_var_scope + "_" + std::to_string(graph_var_scope_counter + k));
                if (placement_after.find(name) != placement_after.end()) return false;
            }
        }
        return true;
    }
    void mult_add_many(Ctx& res, const std::vector<const Ctx*>& vs, const std::vector<const Ctx*>& ss) {
        static std::once_flag announced;
        std::call_once(announced, [&] {
            std::cerr << "[lane_batch] active: fused ct×ct accumulate, n=" << vs.size() << "\n";
        });
        std::vector<Ctx> as, bs;
        as.reserve(vs.size());
        bs.reserve(ss.size());
        for (auto* v : vs) as.push_back(*v);
        for (auto* s : ss) bs.push_back(*s);
        cc->EvalMultCtAccumBatch(res, as, bs);

        for (size_t j = 0; j < vs.size(); ++j) {
            // mult node (phantom product)
            const Ctx& v = *vs[j];
            const Ctx& s = *ss[j];
            const packtag::PackTag tmp_tg = packtag::t_mult(tag_of_ct(v), tag_of_ct(s));
            const int in_v_level = level_for_ct(v);
            const int in_s_level = level_for_ct(s);
            const std::string in_v = var_for_ct(v);
            const std::string in_s = var_for_ct(s);
            if (op_tally_active) ++op_tally["mult_cc"];
            const std::string tmp_var = graph_var_scope.empty()
                ? next_ct("v")
                : (graph_var_scope + "_" + std::to_string(++graph_var_scope_counter));
            record_primitive("mult", {in_v, in_s}, {in_v_level, in_s_level}, tmp_var);
            verify_expected_level(tmp_var, "mult", level_of(v));

            // add_inplace node (the real accumulator)
            const packtag::PackTag res_tg = packtag::t_add(tag_of_ct(res), tmp_tg);
            const int in_r_level = level_for_ct(res);
            const std::string in_r = var_for_ct(res);
            tag_ct(res, res_tg);
            const std::string out = set_new_var_for_ct(res);
            record_primitive("add_inplace", {in_r, tmp_var}, out, {in_r_level, in_v_level},
                             level_for_ct(res), res);
            maybe_apply_planned_bootstrap_after(out, res);   // pre-flight: only the final out can fire
            verify_expected_level(out, "add_inplace", level_of(res));
            tag_ct(res, res_tg);
        }
    }

    // The serial mult(ct, pt) body minus the EvalMult — records one already-computed product.
    Ctx mult_finish(const Ctx& ct, Ptx& pt, Ctx out_ct) {
        const packtag::PackTag _tg = packtag::t_mult(tag_of_ct(ct), tag_of(pt));

        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_pt(pt);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_pt(pt);
        if (op_tally_active) ++op_tally["mult_pt"];
        tag_ct(out_ct, _tg);   // before maybe_apply_planned_bootstrap_after can fire
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("mult", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "mult", level_of(out_ct));
        maybe_bootstrap(out_ct);

        tag_ct(out_ct, _tg);
        return out_ct;
    }

    Ctx mult(const Ctx& ct, double scalar) {
        const packtag::PackTag _tg = packtag::t_mult_scalar(tag_of_ct(ct), scalar);

        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_scalar(scalar);
        if (op_tally_active) ++op_tally["mult_sc"];
        Ctx out_ct = cc->EvalMult(ct, scalar);
        tag_ct(out_ct, _tg);   // before maybe_apply_planned_bootstrap_after can fire
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("mult", {in_a, in_b}, out, {in_a_level, -1}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "mult", level_of(out_ct));
        maybe_bootstrap(out_ct);
        
        tag_ct(out_ct, _tg);
        return out_ct;
    }

    void inplace_mult(Ctx& ct, Ptx& pt) {
        const packtag::PackTag _tg = packtag::t_mult(tag_of_ct(ct), tag_of(pt));

        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_pt(pt);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_pt(pt);
        cc->EvalMultInPlace(ct, pt);
        tag_ct(ct, _tg);   // before the in-op bootstrap hook
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("mult_inplace", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "mult_inplace", level_of(ct));
        maybe_bootstrap(ct);
        
        tag_ct(ct, _tg);
    }

    void inplace_mult(Ctx& ct, double scalar) {
        const packtag::PackTag _tg = packtag::t_mult_scalar(tag_of_ct(ct), scalar);

        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_scalar(scalar);
        cc->EvalMultInPlace(ct, scalar);
        tag_ct(ct, _tg);   // before the in-op bootstrap hook
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("mult_inplace", {in_a, in_b}, out, {in_a_level, -1}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "mult_inplace", level_of(ct));
        maybe_bootstrap(ct);
        
        tag_ct(ct, _tg);
    }

    void inplace_square(Ctx& ct) {
        const packtag::PackTag _tg = packtag::t_square(tag_of_ct(ct));

        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        cc->EvalSquareInPlace(ct);
        tag_ct(ct, _tg);   // before the in-op bootstrap hook
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("square_inplace", {in_a}, out, {in_a_level}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "square_inplace", level_of(ct));
        maybe_bootstrap(ct);
        
        tag_ct(ct, _tg);
    }

    Ctx square(const Ctx& ct) {
        const packtag::PackTag _tg = packtag::t_square(tag_of_ct(ct));

        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        Ctx out_ct = cc->EvalSquare(ct);
        tag_ct(out_ct, _tg);   // before maybe_apply_planned_bootstrap_after can fire
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("square", {in_a}, out, {in_a_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "square", level_of(out_ct));
        maybe_bootstrap(out_ct);
        
        tag_ct(out_ct, _tg);
        return out_ct;
    }

    void assert_rot_key(int32_t index) const {
        if (index == 0) return;   // rotate by 0 = identity; EvalRotate handles it, no key
        const auto& a = cc->rotation_indexes;
        if (std::find(a.begin(), a.end(), index) != a.end()) return;
        throw std::runtime_error(
            "rotate: no loaded rotation key for step " + std::to_string(index) +
            " (freed or deferred); " + std::to_string(a.size()) + " keys loaded");
    }

    Ctx rotate(const Ctx& ct, int32_t index) {
        const packtag::PackTag _tg = packtag::t_rotate(tag_of_ct(ct), index);

        const int in_a_level = level_for_ct(ct);
        assert_rot_key(index);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = "rot(" + std::to_string(index) + ")";

        Ctx out_ct = (index == 0) ? ct->Clone() : cc->EvalRotate(ct, index);
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("rotate", {in_a, in_b}, out, {in_a_level, -1}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "rotate", level_of(out_ct));
        
        tag_ct(out_ct, _tg);
        return out_ct;
    }

    std::vector<Ctx> rotate_hoisted(const Ctx& ct, const std::vector<int32_t>& steps) {
        // Hoisted batched rotations are always used; captured plans assume this node order.
        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        for (int32_t step : steps) assert_rot_key(step);

        auto precomp = cc->EvalFastRotationPrecompute(ct);  // null on GPU, cheap
        std::vector<Ctx> outs =
            cc->EvalFastRotation(ct, steps, 2u * cc->GetRingDimension(), precomp);

        for (size_t k = 0; k < outs.size(); ++k) {
            const int32_t index = steps[k];
            Ctx& out_ct = outs[k];
            const std::string in_b = "rot(" + std::to_string(index) + ")";
            const std::string out = set_new_var_for_ct(out_ct);
            const int out_level = level_for_ct(out_ct);
            record_primitive("rotate", {in_a, in_b}, out, {in_a_level, -1}, out_level, out_ct);
            maybe_apply_planned_bootstrap_after(out, out_ct);
            verify_expected_level(out, "rotate", level_of(out_ct));
        }
        return outs;
    }

    Ctx conjugate(const Ctx& ct) {
        const packtag::PackTag _tg = packtag::t_conjugate(tag_of_ct(ct));

        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        Ctx out_ct = cc->EvalConjugate(ct);
        tag_ct(out_ct, _tg);   // before maybe_apply_planned_bootstrap_after can fire
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("conjugate", {in_a}, out, {in_a_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "conjugate", level_of(out_ct));
        tag_ct(out_ct, _tg);
        return out_ct;
    }

    void inplace_rotate(Ctx& ct, int32_t index) {
        const packtag::PackTag _tg = packtag::t_rotate(tag_of_ct(ct), index);

        const int in_a_level = level_for_ct(ct);
        assert_rot_key(index);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = "rot(" + std::to_string(index) + ")";

        if (index != 0) cc->EvalRotateInPlace(ct, index);
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("rotate_inplace", {in_a, in_b}, out, {in_a_level, -1}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "rotate_inplace", level_of(ct));

        tag_ct(ct, _tg);
    }

    Ctx negate(const Ctx& ct) {
        const packtag::PackTag _tg = packtag::t_mult_scalar(tag_of_ct(ct), -1.0);
        
        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        Ctx out_ct = cc->EvalNegate(ct);
        tag_ct(out_ct, _tg);   // before maybe_apply_planned_bootstrap_after can fire
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("negate", {in_a}, out, {in_a_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "negate", level_of(out_ct));
        maybe_bootstrap(out_ct);
        
        tag_ct(out_ct, _tg);
        return out_ct;
    }

    void inplace_negate(Ctx& ct) {
        const packtag::PackTag _tg = packtag::t_mult_scalar(tag_of_ct(ct), -1.0);
        
        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        cc->EvalNegateInPlace(ct);
        tag_ct(ct, _tg);   // before the in-op bootstrap hook
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("negate_inplace", {in_a}, out, {in_a_level}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "negate_inplace", level_of(ct));
        maybe_bootstrap(ct);
        
        tag_ct(ct, _tg);
    }

    // dst <- src, GPU limbs + metadata only (CryptoContextImpl::CopyCiphertextDevice).
    // Same graph semantics as clone() — recorded as "clone", a level/deg passthrough —
    // but reuses the caller's ciphertext instead of minting one, so it skips the
    // copy-ctor's deep copy of the (stale, unused) OpenFHE CPU shadow: ~27 us vs
    // ~605 us measured. For hot loops that need many transient products from an
    // immutable source; dst must already be a loaded ciphertext of the same shape.
    void copy_into(Ctx& dst, const Ctx& src) {
        const packtag::PackTag _tg = tag_of_ct(src);
        const int in_a_level = level_for_ct(src);
        const std::string in_a = var_for_ct(src);
        cc->CopyCiphertextDevice(dst, src);
        const std::string out = set_new_var_for_ct(dst);
        const int out_level = level_for_ct(dst);
        record_primitive("clone", {in_a}, out, {in_a_level}, out_level, dst);
        maybe_apply_planned_bootstrap_after(out, dst);
        verify_expected_level(out, "clone", level_of(dst));
        tag_ct(dst, _tg);
    }
    void copy_into(PackedCtx& dst, const PackedCtx& src) { copy_into(dst.ct, src.ct); }

    // ---- the constant 1 as a CIPHERTEXT, derived once per process -----------------------
    // A Newton / Goldschmidt seed is an operand of ciphertext-ciphertext ops, and the
    // evaluator holds evaluation keys only — it evaluates, it does not encrypt.
    // `(x + 1) - x` is exactly 1.0 in every slot using only evaluation-key ops, but inherits
    // x's LEVEL, which for a seed is the depth budget of the whole chain. So derive it ONCE
    // from the freshest ciphertext in the process (the input embedding) and hand out clones.
    // Held on the context, so its lifetime ends with the CUDA context.
    Ctx const_one_;
    bool const_one_ready_ = false;

    bool const_one_available() const { return const_one_ready_; }

    // Call with the FRESHEST ciphertext available. First call wins; later ones are no-ops, so a
    // caller cannot accidentally downgrade the cached level by offering a spent ciphertext.
    void ensure_const_one(const Ctx& src, int slots) {
        if (const_one_ready_ || !src) return;
        Ctx one = sub(add(src, 1.0), src);
        tag_ct(one, packtag::PackTag::constant(slots));
        const_one_ = one;
        const_one_ready_ = true;
    }

    // A fresh clone the caller can mutate freely, at the level the constant was DERIVED at —
    // deliberately NOT dropped to the consumer's level (that would hand back the shallow seed
    // the cache exists to avoid; the arithmetic aligns operands internally). Pass `level >= 0`
    // only if a consumer genuinely needs it lower.
    // The tag is re-declared because the registry is keyed by ciphertext and `constant` is a
    // property of the VALUE, exact rather than an over-approximation.
    Ctx const_one_clone(int slots, int level = -1) {
        Ctx one = clone(const_one_);
        if (level >= 0) drop_to_level(one, level);
        tag_ct(one, packtag::PackTag::constant(slots));
        return one;
    }

    Ctx clone(const Ctx& ct) {
        const packtag::PackTag _tg = tag_of_ct(ct);

        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        Ctx out_ct = ct->Clone();
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("clone", {in_a}, out, {in_a_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "clone", level_of(out_ct));
        
        tag_ct(out_ct, _tg);
        return out_ct;
    }


    void bootstrap_hint(Ctx& ct, int level_threshold, bool account_pending_rescale = false) {

        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        std::string peeked_var;   // plan lookup key when hints are plan-bound
        bool fire;
        if (placement_plan_enabled && plan_hints_bound) {
            // Plan-bound decision: the planner's final sim already decided this
            // hint on the planned trajectory. Local re-evaluation flips fire/skip
            // on threshold-boundary trajectories (deg-2 pending-rescale ±1) and
            // silently diverges the run from every downstream pin.
            const std::string peek = peek_new_var_for_ct(ct);
            peeked_var = peek;
            fire = plan_hint_fire.count(peek) > 0;
        } else {
            int eff_level = static_cast<int>(level_of(ct));
            if (account_pending_rescale)
                eff_level += static_cast<int>(pending_rescale_primes(ct));
            fire = eff_level > level_threshold;
        }
        if (fire) {
            // Plan-bound hint sites carry the SAME per-site decisions as planted ones —
            // the plan keys them by the hint output var it also uses for hint_fire, so
            // they go through inner_bootstrap_planned like every other plan-driven site.
            inner_bootstrap_planned(peeked_var, ct);
            // A fired plan-bound hint IS this var's refresh: the sim models `fired or placed`
            // as ONE refresh (sim.py hint branch), but a plan can carry a placement on the
            // same var, and the op-exit hook below would then
            // refresh it a second time -- landing deg-2 with the fire-once realize anchor
            // already consumed, i.e. a [plan_deg_error] on a correct plan. Consume the
            // placement here so the hint's refresh is the only one.
            if (!peeked_var.empty()) placement_after.erase(peeked_var);
            auto sit = peeked_var.empty() ? plan_sparse_slots.end()
                                          : plan_sparse_slots.find(peeked_var);
            auto cit = peeked_var.empty() ? plan_correction_factor.end()
                                          : plan_correction_factor.find(peeked_var);
            std::fprintf(stderr, "[planted_bts] hint var=%s s=%u cf=%d in=%d out=%d\n",
                         peeked_var.c_str(),
                         sit != plan_sparse_slots.end() ? sit->second : 0u,
                         cit != plan_correction_factor.end() ? cit->second : -1,
                         in_a_level, (int)level_for_ct(ct));
        }
        const std::string out = set_new_var_for_ct(ct);
        const std::string in_b = "hint_lev(" + std::to_string(level_threshold) + ")";
        const int out_level = level_for_ct(ct);
        record_primitive("hint", {in_a, in_b}, out, {in_a_level, -1}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);   // plans may cut a hint output
        verify_expected_level(out, "hint", level_of(ct));
    }

    void level_hint(Ctx& ct, int level) {
        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        const std::string out = set_new_var_for_ct(ct);
        const std::string in_b = "lvl_cap(" + std::to_string(level) + ")";
        const int out_level = level_for_ct(ct);
        record_primitive("level_hint", {in_a, in_b}, out, {in_a_level, -1}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
    }

    void drop_to_level(Ctx& ct, int target_level) {
        const int cur = level_for_ct(ct);
        if (target_level <= cur) return;
        const std::string in_a = var_for_ct(ct);
        cc->DropToLevel(ct, static_cast<uint32_t>(target_level));
        const std::string out = set_new_var_for_ct(ct);
        record_primitive("level_reduce", {in_a}, out, {cur}, level_for_ct(ct), ct);
        maybe_apply_planned_bootstrap_after(out, ct);
    }
    void drop_to_level(PackedCtx& pc, int target_level) { drop_to_level(pc.ct, target_level); }

    //  PackedCtx overlay, ensures packing matching between operands and results.

    PackedCtx add(const PackedCtx& a, const PackedCtx& b) {
        check_packing(a.packing, b.packing);
        slotlayout::check_binary(a.ct, b.ct, "add");
        return slotlayout::keep(a.ct, b.ct, tagged(add(a.ct, b.ct), a.packing, packtag::t_add(a.tag, b.tag)));
    }
    // ct(+/x)pt: the ordinary algebra applies once the plaintext is tagged — identical tags
    // give lcm(P,P)=P and union/intersect of identical supports, i.e. the result keeps the
    // packing, which is the expected case for an aligned operand. An UNTAGGED pt yields
    // unknown, and we must NOT substitute "same as the ciphertext": a period-1 constant times
    // a per-position mask of period 32 has period 32, so assuming the ct's tag would
    // UNDER-estimate and route sparse into a silent fold corruption.
    PackedCtx add(const PackedCtx& a, Ptx& pt) {
        return slotlayout::keep(a.ct, tagged(add(a.ct, pt), a.packing, packtag::t_add(a.tag, tag_of(pt))));
    }
    PackedCtx add(const PackedCtx& a, double scalar){ return slotlayout::keep(a.ct, tagged(add(a.ct, scalar), a.packing, packtag::t_add_scalar(a.tag, scalar))); }

    PackedCtx sub(const PackedCtx& a, const PackedCtx& b) {
        check_packing(a.packing, b.packing);
        slotlayout::check_binary(a.ct, b.ct, "sub");
        // t_add is the sub rule too: period lcm + support union — sign is irrelevant.
        return slotlayout::keep(a.ct, b.ct, tagged(sub(a.ct, b.ct), a.packing, packtag::t_add(a.tag, b.tag)));
    }
    PackedCtx sub(const PackedCtx& a, Ptx& pt)      { return slotlayout::keep(a.ct, tagged(sub(a.ct, pt),     a.packing, packtag::t_add(a.tag, tag_of(pt)))); }
    PackedCtx sub(const PackedCtx& a, double scalar){ return slotlayout::keep(a.ct, tagged(sub(a.ct, scalar), a.packing, packtag::t_add_scalar(a.tag, scalar))); }

    PackedCtx mult(const PackedCtx& a, const PackedCtx& b) {
        check_packing(a.packing, b.packing);
        slotlayout::check_binary(a.ct, b.ct, "mult");
        return slotlayout::keep(a.ct, b.ct, tagged(mult(a.ct, b.ct), a.packing, packtag::t_mult(a.tag, b.tag)));
    }
    PackedCtx mult(const PackedCtx& a, Ptx& pt) {
        return slotlayout::keep(a.ct, tagged(mult(a.ct, pt), a.packing, packtag::t_mult(a.tag, tag_of(pt))));
    }
    // Batched-lane companion: record one product computed by mult_batch_exec (see the Ctx
    // overload above for the execution/recording split contract).
    PackedCtx mult_finish(const PackedCtx& a, Ptx& pt, Ctx out_ct) {
        return tagged(mult_finish(a.ct, pt, std::move(out_ct)), a.packing,
                      packtag::t_mult(a.tag, tag_of(pt)));
    }
    // Batched-accumulate companions: res += Σ vs[j]*ss[j], one relinearization (see Ctx overloads).
    bool mult_add_many_usable(const PackedCtx& res, const std::vector<const PackedCtx*>& vs,
                              const std::vector<const PackedCtx*>& ss) {
        std::vector<const Ctx*> v_raw, s_raw;
        v_raw.reserve(vs.size());
        s_raw.reserve(ss.size());
        for (auto* v : vs) v_raw.push_back(&v->ct);
        for (auto* s : ss) s_raw.push_back(&s->ct);
        return mult_add_many_usable(res.ct, v_raw, s_raw);
    }
    void mult_add_many(PackedCtx& res, const std::vector<const PackedCtx*>& vs,
                       const std::vector<const PackedCtx*>& ss) {
        std::vector<const Ctx*> v_raw, s_raw;
        v_raw.reserve(vs.size());
        s_raw.reserve(ss.size());
        for (auto* v : vs) v_raw.push_back(&v->ct);
        for (auto* s : ss) s_raw.push_back(&s->ct);
        mult_add_many(res.ct, v_raw, s_raw);
        res.tag = tag_of_ct(res.ct);   // the Ctx-level replay updated the registry tag per lane
    }
    PackedCtx mult(const PackedCtx& a, double scalar) { return slotlayout::keep(a.ct, tagged(mult(a.ct, scalar), a.packing, packtag::t_mult_scalar(a.tag, scalar))); }

    // The raw inplace ops compute the proper transfer tag in the ct registry; sync the
    // PackedCtx FIELD from it afterwards so the two views never diverge. A stale field
    // here is a soundness hazard for any later reader of pc.tag.
    void inplace_add(PackedCtx& a, const PackedCtx& b) {
        check_packing(a.packing, b.packing);
        inplace_add(a.ct, b.ct);
        a.tag = tag_of_ct(a.ct);
    }
    void inplace_add(PackedCtx& a, Ptx& pt)       { inplace_add(a.ct, pt);     a.tag = tag_of_ct(a.ct); }
    void inplace_add(PackedCtx& a, double scalar) { inplace_add(a.ct, scalar); a.tag = tag_of_ct(a.ct); }

    void inplace_sub(PackedCtx& a, const PackedCtx& b) {
        check_packing(a.packing, b.packing);
        inplace_sub(a.ct, b.ct);
        a.tag = tag_of_ct(a.ct);
    }
    void inplace_sub(PackedCtx& a, double scalar) { inplace_sub(a.ct, scalar); a.tag = tag_of_ct(a.ct); }

    void inplace_mult(PackedCtx& a, Ptx& pt)       { inplace_mult(a.ct, pt);     a.tag = tag_of_ct(a.ct); }
    void inplace_mult(PackedCtx& a, double scalar) { inplace_mult(a.ct, scalar); a.tag = tag_of_ct(a.ct); }

    PackedCtx square(const PackedCtx& a) { return slotlayout::keep(a.ct, tagged(square(a.ct), a.packing, packtag::t_square(a.tag))); }
    void inplace_square(PackedCtx& a)    { inplace_square(a.ct); a.tag = tag_of_ct(a.ct); }

    PackedCtx negate(const PackedCtx& a) { return tagged(negate(a.ct), a.packing, packtag::t_mult_scalar(a.tag, -1.0)); }
    void inplace_negate(PackedCtx& a)    { inplace_negate(a.ct); a.tag = tag_of_ct(a.ct); }

    PackedCtx clone(const PackedCtx& a) { return slotlayout::keep(a.ct, tagged(clone(a.ct), a.packing, a.tag)); }

    PackedCtx rotate(const PackedCtx& a, int32_t index) { return tagged(rotate(a.ct, index), a.packing, packtag::t_rotate(a.tag, index)); }
    std::vector<PackedCtx> rotate_hoisted(const PackedCtx& a, const std::vector<int32_t>& steps) {
        std::vector<Ctx> raw = rotate_hoisted(a.ct, steps);
        std::vector<PackedCtx> out;
        out.reserve(raw.size());
        for (size_t i = 0; i < raw.size(); ++i)
            out.push_back(tagged(std::move(raw[i]), a.packing,
                                 packtag::t_rotate(a.tag, steps[i])));
        return out;
    }
    PackedCtx conjugate(const PackedCtx& a) { return slotlayout::keep(a.ct, tagged(conjugate(a.ct), a.packing, packtag::t_conjugate(a.tag))); }
    void inplace_rotate(PackedCtx& a, int32_t index)    { inplace_rotate(a.ct, index); a.tag = tag_of_ct(a.ct); }

    PackedCtx im_cleanse(const PackedCtx& a) { return add(a, conjugate(a)); }
    void inplace_im_cleanse(PackedCtx& a) { inplace_add(a, conjugate(a)); }

    PackedCtx pack_ri(const PackedCtx& a, const PackedCtx& b, Ptx& i_pt) {
        return add(a, mult(b, i_pt));                         // a + i*b
    }
    std::pair<PackedCtx, PackedCtx> unpack_ri(const PackedCtx& P, Ptx& nhi_pt) {
        PackedCtx conj = conjugate(P);
        PackedCtx re   = mult(add(P, conj), 0.5);             // Re(P) = (P+conj)/2
        PackedCtx im   = mult(sub(P, conj), nhi_pt);          // Im(P) = (P-conj)*(-i/2)
        return {std::move(re), std::move(im)};
    }

    std::pair<PackedCtx, PackedCtx> conj_split(const PackedCtx& P) {
        PackedCtx conj = conjugate(P);
        return { add(P, conj), sub(P, conj) };
    }

    PackedCtx pair_pack(const PackedCtx& a_re, const PackedCtx& b_im) {
        return PackedCtx{ pair_pack(a_re.ct, b_im.ct), a_re.packing };
    }
    PackedCtx mult_i(const PackedCtx& a) { return tagged(mult_i(a.ct), a.packing, a.tag); }

    // Repack two independently-computed real halves (token-pair): strip the Im contamination each
    // half accrued, pack A + i*B, cancel the cleanses' 2x. Levels must already agree.
    PackedCtx pair_pack_cleansed(PackedCtx a_re, PackedCtx b_im) {
        inplace_im_cleanse(a_re);
        inplace_im_cleanse(b_im);
        if (level_for_ct(a_re.ct) != level_for_ct(b_im.ct))
            throw std::runtime_error("[pair_pack_cleansed] A/B level desync before repack");
        PackedCtx out = pair_pack(a_re, b_im);
        inplace_mult(out, 0.5);
        return out;
    }

    void bootstrap_hint(PackedCtx& pc, int level_threshold, bool account_pending_rescale = false) {
        bootstrap_hint(pc.ct, level_threshold, account_pending_rescale);
    }

    void level_hint(PackedCtx& pc, int level) {
        level_hint(pc.ct, level);
    }
};

inline WithStep::WithStep(CKKSContext& c, const std::string& s) : ctx_(&c) {
    ctx_->push_step(s);
}
inline WithStep::~WithStep() {
    if (ctx_) ctx_->pop_step();
}
inline void WithStep::next(const std::string& s) {
    if (ctx_) { ctx_->pop_step(); ctx_->push_step(s); }
}

// ── Chain-dependent defaults ────────────────────────────────────────────────────────────────
// The defaults below MUST follow the build's NATIVEINT. At NATIVEINT=32 OpenFHE enforces
// 15 < scalingModSize < 31 PER PRIME, so the n64 literals (scale 58, first_mod 60,
// composite_degree 1) make every context construction throw; a composite chain satisfies it
// by splitting the scale across `composite_degree` ~27-bit primes.
// NATIVEINT arrives via <CKKS/openfhe-interface/RawCiphertext.cuh> -> openfhe.h -> config_core.h.
#if !defined(NATIVEINT)
#error "NATIVEINT is not visible here; OpenFHE's config_core.h must be included before this point. \
Without it the n64 defaults would silently be used on an n32 build -- the exact bug this guards."
#endif
namespace chain_defaults {
#if NATIVEINT == 32
inline constexpr int      depth              = 10;
inline constexpr int      scale_bits         = 54;   // 2 x 27-bit primes
inline constexpr uint32_t composite_degree   = 2;
inline constexpr int      btp_scale_bits     = 54;
inline constexpr uint32_t correction_factor  = 6;    // CF positions the EvalMod band
inline constexpr int      first_mod_bits     = 56;
inline constexpr uint32_t num_large_digits   = 6;
// NOT 24*d: AUTO_BTS_LEVEL is a THRESHOLD, so remaining DEPTH is what must be preserved.
// n32 has 54 primes, so 54 - 4*d = 46 restores n64's 4-level headroom.
inline constexpr uint32_t auto_bts_level     = 46;
#else
inline constexpr int      depth              = 11;
inline constexpr int      scale_bits         = 58;
inline constexpr uint32_t composite_degree   = 1;
inline constexpr int      btp_scale_bits     = 53;
inline constexpr uint32_t correction_factor  = 0;    // OpenFHE auto (good at 53-bit scale)
inline constexpr int      first_mod_bits     = 60;
inline constexpr uint32_t num_large_digits   = 7;
inline constexpr uint32_t auto_bts_level     = 24;
#endif
}  // namespace chain_defaults

struct CKKSContextOptions {
    // Client/server split (serial.cu bundle): keys_dir non-empty => build the session FROM
    // a deserialized {context.bin, public.key, multkeys.bin, rotkeys.bin} bundle — no
    // KeyGen, no secret key in the process. skip_gpu_load => stop after the CPU keygen /
    // key load (a client that only encrypts/decrypts/serializes needs no GPU context).
    std::string keys_dir;
    bool skip_gpu_load = false;
    // CKKS scheme
    int logN       = 16;
    int depth      = chain_defaults::depth;   // usable compute levels (L); +btp_depth_overhead
    int scale_bits = chain_defaults::scale_bits;

    bool     enable_bootstrap   = true;
    uint32_t btp_depth_overhead = 16;
    std::vector<uint32_t> level_budget = {4, 3};
    uint32_t bootstrap_slots    = 0;   // 0 = N/2
    // Sparse bootstrap precomps (fold reductions + dual-slots sparse arcsine); empty =
    // off. SPARSE_BTS_SLOTS accepts a comma list ("512,32,1") — one precomp + keygen per
    // distinct entry; the FIRST entry is the routing default SparseBtsScope arms.
    // Arcsine coupling: under FIDESLIB_SPARSE_ARCSINE=1 only sparse setups reserve modall+3.
    std::vector<uint32_t> sparse_bts_slots_list = {};
    std::vector<uint32_t> sparse_level_budget = {};   // empty = level_budget

    // COMPOSITESCALING (NATIVEINT=32): primes per CKKS level. >1 selects
    // COMPOSITESCALINGMANUAL (scale = composite_degree x ~27-bit primes; e.g. depth 27,
    // BTP_SCALE_BITS=54, FIRST_MOD_BITS=56, d=2 => the 2x27 chain CPU-validated at
    // 10.8/19.2 bits). 1 = classic chains.
    uint32_t composite_degree   = chain_defaults::composite_degree;

    int      btp_scale_bits     = chain_defaults::btp_scale_bits;
    uint32_t correction_factor  = chain_defaults::correction_factor;
    int      first_mod_bits     = chain_defaults::first_mod_bits;
    uint32_t num_large_digits   = chain_defaults::num_large_digits;
    uint32_t auto_bts_level_override = chain_defaults::auto_bts_level;

    uint32_t batch_size     = 0;       // 0 = N/2
    bool ckks_complex_payload = false;

    int      h_weight         = 192;   // 0 = UNIFORM_TERNARY; >0 = SPARSE_TERNARY

    std::vector<int32_t> extra_rot_steps = {};
    std::vector<int32_t> deferred_rot_steps = {};

    uint32_t bts_iterations          = 1;
    uint32_t bts_precision           = 12;

    // Pipeline-fill overlap: return from make_ckks_context after the CPU
    // context + mult keys exist; the heavy tail (rot keygen, bts setups, LoadContext)
    // is stashed as CKKSContext::pending_heavy_setup for complete_setup(). The caller
    // OWNS calling it before any GPU op — see GPT2Model prefill wiring.
    bool defer_heavy_setup = false;
};

inline CKKSContextOptions ckks_options_from_env() {
    CKKSContextOptions o{};
    auto env_int = [](const char* name, auto dflt) {
        const char* e = std::getenv(name);
        return (e && *e) ? static_cast<decltype(dflt)>(std::stol(e)) : dflt;
    };
    o.logN               = env_int("LOGN",               o.logN);
    o.depth              = env_int("CKKS_DEPTH",         o.depth);
    o.first_mod_bits     = env_int("FIRST_MOD_BITS",     o.first_mod_bits);
    o.btp_scale_bits     = env_int("BTP_SCALE_BITS",     o.btp_scale_bits);
    o.scale_bits         = env_int("SCALE_BITS",         o.scale_bits);
    o.btp_depth_overhead = env_int("BTP_DEPTH_OVERHEAD", o.btp_depth_overhead);
    o.h_weight           = env_int("H_WEIGHT",           o.h_weight);
    o.num_large_digits   = env_int("NUM_LARGE_DIGITS",   o.num_large_digits);
    o.auto_bts_level_override = env_int("AUTO_BTS_LEVEL", o.auto_bts_level_override);
    o.correction_factor  = env_int("CORRECTION_FACTOR",  o.correction_factor);
    o.composite_degree   = env_int("COMPOSITE_DEGREE",   o.composite_degree);
    o.bts_iterations     = env_int("BTS_ITERATIONS",     o.bts_iterations);
    o.bts_precision      = env_int("BTS_PRECISION",      o.bts_precision);

    if (const char* sbs = std::getenv("SPARSE_BTS_SLOTS"); sbs && *sbs) {
        std::stringstream ss{std::string(sbs)};
        std::string tok;
        while (std::getline(ss, tok, ','))
            if (!tok.empty() && std::stoul(tok) > 0)
                o.sparse_bts_slots_list.push_back(std::stoul(tok));
    }

    if (const char* slb = std::getenv("SPARSE_LEVEL_BUDGET"); slb && *slb) {
        std::vector<uint32_t> budget;
        std::string s(slb);
        for (auto& ch : s) if (ch == ':') ch = ',';
        std::stringstream ss(s);
        std::string tok;
        while (std::getline(ss, tok, ',')) if (!tok.empty()) budget.push_back(std::stoul(tok));
        if (budget.size() >= 2) o.sparse_level_budget = budget;
        else std::cerr << "[ckks_options_from_env] SPARSE_LEVEL_BUDGET='" << slb
                       << "' needs two entries (cts:stc) — IGNORED\n";
    }

    if (const char* lb = std::getenv("LEVEL_BUDGET"); lb && *lb) {
        std::vector<uint32_t> budget;
        std::string s(lb);
        for (auto& ch : s) if (ch == ':') ch = ',';
        std::stringstream ss(s);
        std::string tok;
        while (std::getline(ss, tok, ',')) if (!tok.empty()) budget.push_back(std::stoul(tok));
        if (budget.size() >= 2) o.level_budget = budget;
        else std::cerr << "[ckks_options_from_env] LEVEL_BUDGET='" << lb
                       << "' needs two entries (cts:stc) — IGNORED\n";
    }

    if (const char* cx = std::getenv("CKKS_COMPLEX"); cx && cx[0] == '1')
        o.ckks_complex_payload = true;

    auto show = [](const char* name, auto val) {
        const char* e = std::getenv(name);
        std::cerr << "  " << std::left << std::setw(22) << name << "= " << val
                  << ((e && *e) ? "  [env]" : "  [def]") << "\n";
    };
    std::string lb;
    for (size_t i = 0; i < o.level_budget.size(); ++i)
        lb += (i ? ":" : "") + std::to_string(o.level_budget[i]);
    std::cerr << "[ckks_env] recognized environment knobs (effective values):\n";
    show("LOGN",                     o.logN);
    show("CKKS_DEPTH",               o.depth);
    show("BTP_DEPTH_OVERHEAD",       o.btp_depth_overhead);
    show("FIRST_MOD_BITS",           o.first_mod_bits);
    show("BTP_SCALE_BITS",           o.btp_scale_bits);
    show("SCALE_BITS",               o.scale_bits);
    show("CORRECTION_FACTOR",        o.correction_factor);
    show("COMPOSITE_DEGREE",         o.composite_degree);
    show("NUM_LARGE_DIGITS",         o.num_large_digits);
    show("AUTO_BTS_LEVEL",           o.auto_bts_level_override);
    show("H_WEIGHT",                 o.h_weight);
    show("BTS_ITERATIONS",           o.bts_iterations);
    show("BTS_PRECISION",            o.bts_precision);
    show("CKKS_COMPLEX",             (int)o.ckks_complex_payload);
    std::string sbs;
    for (size_t i = 0; i < o.sparse_bts_slots_list.size(); ++i)
        sbs += (i ? "," : "") + std::to_string(o.sparse_bts_slots_list[i]);
    if (sbs.empty()) sbs = "0";
    std::cerr << "  " << std::left << std::setw(22) << "SPARSE_BTS_SLOTS" << "= " << sbs
              << (std::getenv("SPARSE_BTS_SLOTS") ? "  [env]" : "  [def]") << "\n";
    std::cerr << "  " << std::left << std::setw(22) << "LEVEL_BUDGET" << "= " << lb
              << (std::getenv("LEVEL_BUDGET") ? "  [env]" : "  [def]") << "\n";

    return o;
}

inline void log_ckks_params(const CC& cc, const CKKSContextOptions& o, uint32_t slots) {
    const uint32_t ring_dim = cc->GetRingDimension();
    const int mult_depth = o.depth + (o.enable_bootstrap ? o.btp_depth_overhead : 0);
    const int scale      = o.enable_bootstrap ? o.btp_scale_bits : o.scale_bits;
    const int num_limbs  = mult_depth + 1;
    const double logQ = o.first_mod_bits + static_cast<double>(mult_depth) * scale;   // uniform towers

    auto budget128 = [](int lN) -> double {
        switch (lN) { case 13: return 218; case 14: return 438; case 15: return 881;
                      case 16: return 1761; case 17: return 3523; default: return -1.0; }
    };
    const double b      = budget128(o.logN);
    const bool   sparse = (o.h_weight > 0);

    std::string sec;
    if (b < 0)
        sec = "HEStd_128_classic enforced (budget for this logN unknown)";
    else
        sec = "HEStd_128_classic enforced & OK (logQ " + std::to_string((long)logQ)
            + " <= uniform-ternary budget " + std::to_string((long)b) + ")";

    std::cerr << "[ckks_params] ===== CKKS crypto context =====\n"
              << "  logN=" << o.logN << "  ringDim=" << ring_dim << "  slots=" << slots << "\n"
              << "  depth=" << o.depth << "  btp_overhead=" << o.btp_depth_overhead
              << "  mult_depth=" << mult_depth << "  limbs=" << num_limbs << "\n"
              << "  first_mod=" << o.first_mod_bits << "  scale(active)=" << scale
              << "  (btp_scale=" << o.btp_scale_bits << " scale_bits=" << o.scale_bits << ")"
              << "  logQ~=" << logQ << "\n"
              << "  bootstrap=" << (o.enable_bootstrap ? "on" : "off")
              << "  correction_factor=" << o.correction_factor
              << "  bts_iters=" << o.bts_iterations << "  bts_prec=" << o.bts_precision
              << "  auto_bts_level=" << o.auto_bts_level_override << "\n"
              << "  secret=" << (sparse ? "SPARSE_ENCAPSULATED" : "UNIFORM_TERNARY")
              << " h=" << o.h_weight << " (OpenFHE accounts UNIFORM_TERNARY)"
              << "  dnum=" << o.num_large_digits
              << "  complex_payload=" << o.ckks_complex_payload << "\n"
              << "  security: " << sec << "\n"
              << "[ckks_params] ===============================\n";
}

inline std::shared_ptr<CKKSContext>
make_ckks_context(const CKKSContextOptions& o = {})
{
    CCParams<CryptoContextCKKSRNS> params;

    std::vector<uint32_t> level_budget;
    int actual_scale_bits;
    uint32_t btp_depth_overhead = o.btp_depth_overhead;
    if (o.enable_bootstrap) {
        actual_scale_bits = o.btp_scale_bits;
        level_budget      = o.level_budget.empty() ? std::vector<uint32_t>{3, 3}
                                                   : o.level_budget;
    } else {
        actual_scale_bits  = o.scale_bits;
        level_budget       = {};
        btp_depth_overhead = 0;
    }

    const uint32_t slots = (o.batch_size == 0) ? (1u << (o.logN - 1)) : o.batch_size;

    params.SetMultiplicativeDepth(o.depth + btp_depth_overhead);
    params.SetScalingModSize(actual_scale_bits);

    if (o.ckks_complex_payload)
        params.SetCKKSDataTypeComplex();
    params.SetFirstModSize(o.first_mod_bits);
    if (o.composite_degree > 1) {
        // NATIVEINT=32 bootstrap path: bootstrap needs >=42 scale bits but 32-bit primes cap
        // at 28, so the scale is a PRODUCT of composite_degree primes per level. MANUAL takes
        // the explicit degree; the register word size caps the prime width the sampler uses.
        params.SetScalingTechnique(fideslib::COMPOSITESCALINGMANUAL);
        params.SetCompositeDegree(o.composite_degree);
        params.SetRegisterWordSize(32);
    } else {
        params.SetScalingTechnique(FLEXIBLEAUTO);
    }
    params.SetBatchSize(slots);
    params.SetSecretKeyDist(o.h_weight > 0 ? fideslib::SPARSE_ENCAPSULATED : UNIFORM_TERNARY);
    params.SetNumLargeDigits(o.num_large_digits);
    params.SetKeySwitchTechnique(HYBRID);
    params.SetSecurityLevel(HEStd_128_classic);  // always enforce 128-bit
    params.SetRingDim(1 << o.logN);

    const bool from_keys = !o.keys_dir.empty();
    CC cc;
    if (from_keys) {
        // The bundle's context carries the exact modulus chain the keys were made for;
        // regenerating from params would only reproduce it by luck of determinism.
        if (!fideslib::Serial::DeserializeFromFile(o.keys_dir + "/context.bin", cc,
                                                   fideslib::BINARY) || !cc)
            throw std::runtime_error("make_ckks_context: cannot deserialize " +
                                     o.keys_dir + "/context.bin");
        // GenCryptoContext fills this from the params; deserialization leaves it 0, and
        // the device->host sync's dummy-container level is derived from it.
        cc->multiplicative_depth = static_cast<uint32_t>(o.depth + btp_depth_overhead);
    } else {
        cc = GenCryptoContext(params);
    }
    log_ckks_params(cc, o, slots);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    if (o.enable_bootstrap) {
        cc->Enable(ADVANCEDSHE);
        cc->Enable(FHE);
    }

    KP kp;
    if (from_keys) {
        if (!fideslib::Serial::DeserializeFromFile(o.keys_dir + "/public.key", kp.publicKey,
                                                   fideslib::BINARY) || !kp.publicKey)
            throw std::runtime_error("make_ckks_context: cannot deserialize " +
                                     o.keys_dir + "/public.key");
        std::ifstream fm(o.keys_dir + "/multkeys.bin", std::ios::binary);
        if (!fm || !cc->DeserializeEvalMultKey(fm, fideslib::BINARY))
            throw std::runtime_error("make_ckks_context: cannot deserialize " +
                                     o.keys_dir + "/multkeys.bin");
        std::cerr << "[ckks_params] SERVER SESSION: keys from " << o.keys_dir
                  << " (no secret key in this process)\n";
    } else {
        kp = cc->KeyGen();
    }
    if (!from_keys)
        std::cerr << "[ckks_params] keygen: h_weight=" << o.h_weight << "  bts mode="
                  << (o.h_weight > 0 ? "ENCAPS(sparse-inside-bootstrap)" : "UNIFORM") << "\n";
    if (!from_keys) cc->EvalMultKeyGen(kp.secretKey);

    std::vector<int32_t> rot_steps;
    for (auto r : o.extra_rot_steps) rot_steps.push_back(r);
    std::sort(rot_steps.begin(), rot_steps.end());
    rot_steps.erase(std::unique(rot_steps.begin(), rot_steps.end()), rot_steps.end());

    auto ctx  = std::make_shared<CKKSContext>();
    ctx->cc   = cc;
    ctx->keys = std::move(kp);
    // A GPU-less session (client keygen/encrypt/decrypt) must not auto-load fresh
    // ciphertexts to a device it never loaded: Encrypt would throw "not loaded".
    if (o.skip_gpu_load) cc->auto_load_ciphertexts = false;

    // The HEAVY tail — rotation keygen, bootstrap setups/keygens (dense + every sparse
    // slot count), and LoadContext (the GPU upload of keys + bts plaintexts). With
    // o.defer_heavy_setup this is stashed on the context and run by complete_setup(),
    // so a caller can overlap host-side encode work with it (the CPU context above is
    // everything MakeCKKSPackedPlaintext needs). Captures cc/keys via ctx (shared_ptr).
    auto heavy = [cc, ctx, rot_steps, level_budget, slots, from_keys,
                  keys_dir = o.keys_dir, skip_gpu_load = o.skip_gpu_load,
                  enable_bootstrap = o.enable_bootstrap,
                  bootstrap_slots = o.bootstrap_slots,
                  correction_factor = o.correction_factor,
                  sparse_level_budget = o.sparse_level_budget,
                  sparse_bts_slots_list = o.sparse_bts_slots_list,
                  deferred_rot_steps = o.deferred_rot_steps]() {
        if (from_keys) {
            // Rotation AND bootstrap automorphism keys ride one store (OpenFHE's
            // automorphism map), so the client's EvalBootstrapKeyGen arrives here too.
            std::ifstream fr(keys_dir + "/rotkeys.bin", std::ios::binary);
            if (!fr || !cc->DeserializeEvalAutomorphismKey(fr, fideslib::BINARY))
                throw std::runtime_error("make_ckks_context: cannot deserialize " +
                                         keys_dir + "/rotkeys.bin");
        } else if (!rot_steps.empty()) {
            cc->EvalRotateKeyGen(ctx->keys.secretKey, rot_steps);
        }
        ctx->loaded_rot_steps.assign(rot_steps.begin(), rot_steps.end());

        if (enable_bootstrap) {
            uint32_t btp_slots = (bootstrap_slots == 0) ? slots : bootstrap_slots;
            // Setup is parameter-only precomputation (public): always re-run. KeyGen needs
            // the secret key: only on the keygen side. dim1={0,0} = OpenFHE's auto BSGS split.
            cc->EvalBootstrapSetup(level_budget, bts_dim1_from_env(), btp_slots, correction_factor);
            if (!from_keys) cc->EvalBootstrapKeyGen(ctx->keys.secretKey, btp_slots);
            // One extra precomp + keygen per distinct requested sparse slot count (fold
            // reductions want several: e.g. 512 for the softmax denominator, 32 for the
            // diagonal LN variance, 1 for the cachemir all-reduce). Under
            // FIDESLIB_SPARSE_ARCSINE=1 only these setups reserve modall+3.
            {
                std::set<uint32_t> built;
                const auto& slb = sparse_level_budget.empty()
                    ? level_budget : sparse_level_budget;
                for (uint32_t s : sparse_bts_slots_list) {
                    if (s == 0 || s >= slots || !built.insert(s).second) continue;
                    cc->EvalBootstrapSetup(slb, bts_dim1_from_env(), s, correction_factor);
                    if (!from_keys) cc->EvalBootstrapKeyGen(ctx->keys.secretKey, s);
                }
            }
        }

        // Lever 1b: hold the sparse encapsulation secret (regenerated keys) before the GPU load reads the keys
        if (!from_keys && enable_bootstrap) {
            const char* e = std::getenv("FIDESLIB_AKS");
            if (e && std::atoi(e) > 0) cc->RegenerateEncapsulationKeys(ctx->keys.secretKey);
        }
        cc->deferred_rotation_indexes = deferred_rot_steps;
        if (!skip_gpu_load) cc->LoadContext(ctx->keys.publicKey);
        // Lever 1b: aggregated key switching for CtS stage 0 (FIDESLIB_AKS=1; needs FIDESLIB_BTS_SHIFT>=1).
        if (!skip_gpu_load && !from_keys && enable_bootstrap) {
            const char* e = std::getenv("FIDESLIB_AKS");
            if (e && std::atoi(e) > 0) cc->LoadAksKeys(ctx->keys.secretKey);
            const char* d = std::getenv("FIDESLIB_DIAG_SK");  // noise-flooding-free diagnostic decryptions
            if (d && std::atoi(d) > 0) cc->LoadDiagSecret(ctx->keys.secretKey);
        }
        // Setup-time pre-capture of cached bootstrap graphs. Inside heavy so BOTH branches
        // get it; LoadContext just ran, so the GPU context and precomputes are live. No-op
        // unless FIDESlib's graph journal is configured (FIDESLIB_BTS_GRAPH / _JOURNAL).
        if (cc->gpu.has_value())
            FIDESlib::CKKS::BootstrapPrecapture(std::any_cast<FIDESlib::CKKS::Context&>(cc->gpu));
    };
    if (o.defer_heavy_setup)
        ctx->pending_heavy_setup = std::move(heavy);
    else
        heavy();
    ctx->total_depth  = o.depth + btp_depth_overhead;
    ctx->btp_overhead = btp_depth_overhead;
    ctx->composite_degree = static_cast<int>(o.composite_degree > 0 ? o.composite_degree : 1);
    ctx->bts_iterations          = o.bts_iterations;
    ctx->bts_precision           = o.bts_precision;
    ctx->auto_bts_level_override = o.auto_bts_level_override;
    ctx->complex_payload         = o.ckks_complex_payload;
    for (uint32_t s : o.sparse_bts_slots_list)
        if (s > 0 && s < slots) ctx->sparse_precomp_slots.insert(s);
    // Routing default = the FIRST configured entry (what SparseBtsScope arms).
    ctx->sparse_bts_slots = 0;
    for (uint32_t s : o.sparse_bts_slots_list)
        if (s > 0 && s < slots) { ctx->sparse_bts_slots = s; break; }
    if (!ctx->sparse_precomp_slots.empty()) {
        std::string built;
        for (uint32_t b : ctx->sparse_precomp_slots)
            built += (built.empty() ? "" : ",") + std::to_string(b);
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        std::cerr << "[sparse_bts] precomps={" << built << "} routing_default="
                  << ctx->sparse_bts_slots << " vram_free_gb="
                  << (double)free_b / (1 << 30) << "\n";
    }
    ctx->first_mod_bits          = o.first_mod_bits;   // q0 width — the coeff centred-lift bound
    // The bootstrap scale, cached so bootstrap_deg_floor() can state the CF floor without
    // re-reading the environment (a plan is validated long after the options are gone).
    ctx->bts_scale_bits          = o.enable_bootstrap ? o.btp_scale_bits : o.scale_bits;

    return ctx;
}

// COMPOSITE ENCODE-LEVEL DIAGNOSTIC. Under COMPOSITESCALING*, OpenFHE fills its scaling-factor
// table only at prime indices that are multiples of the composite degree ("holes" hold 1), so
// an off-grid encode level throws "Scaling factor too small" naming neither the level nor the
// caller. This names the level once per offending value.
inline void _check_composite_encode_level(int level) {
    static const int d = [] {
        const char* e = std::getenv("COMPOSITE_DEGREE");
        return (e && *e) ? std::max(1, std::atoi(e)) : 1;
    }();
    if (d <= 1 || level % d == 0) return;
    static std::set<int> seen;
    static std::mutex m;
    std::lock_guard<std::mutex> lk(m);
    if (seen.insert(level).second)
        std::fprintf(stderr,
                     "[encode_level] composite d=%d: level %d is NOT a multiple of d — "
                     "OpenFHE's scaling-factor table has a HOLE there (sf=1) and the encode "
                     "will throw 'Scaling factor too small'\n", d, level);
}

inline Ptx encode(const CC& cc, const std::vector<double>& values, int level = 0) {
    _check_composite_encode_level(level);
    return cc->MakeCKKSPackedPlaintext(values, /*noiseScaleDeg=*/1, (uint32_t)level);
}

inline Ptx encode(const CC& cc, const std::vector<std::complex<double>>& values,
                   int level = 0) {
    _check_composite_encode_level(level);
    return cc->MakeCKKSPackedPlaintext(values, /*noiseScaleDeg=*/1, (uint32_t)level);
}

inline Ptx encode_const(const CC& cc, double val, size_t slots, int level = 0) {
    std::vector<double> v(slots, val);
    return encode(cc, v, level);
}

inline Ctx encrypt(const CC& cc, Ptx pt, const PublicKey<DCRTPoly>& pk) {
    return cc->Encrypt(pk, pt);
}

inline Ctx encrypt_const(const CC& cc, double val, size_t slots,
                          const PublicKey<DCRTPoly>& pk, int level = 0) {
    return encrypt(cc, encode_const(cc, val, slots, level), pk);
}

// Decrypt-side tower drop: a value-preserving DropToLevel (OpenFHE LevelReduce) on a device
// CLONE before the host decrypt of a GPU-resident ciphertext — every step of that decrypt is
// proportional to the limb count. The caller's ciphertext is untouched. Exact as long as
// |m|*scale < Q'/2: keep 4 primes (deg-1) or 6 (deg-2, scale^2), no rounding introduced.
inline Ctx decrypt_view(const CC& cc, const Ctx& ct) {
    if (!ct || !cc->loaded || !ct->loaded) return ct;
    const auto& lcc = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
    const size_t total = lcc->GetCryptoParameters()->GetElementParams()->GetParams().size();
    const size_t dropped = ct->GetLevel();                 // OpenFHE convention: primes dropped
    const size_t keep = (ct->GetNoiseScaleDeg() >= 2) ? 6 : 4;
    if (total <= dropped + keep) return ct;                // already at or below the target
    Ctx view = ct->Clone();                                // device copy, metadata-only shadow
    cc->DropToLevel(view, static_cast<uint32_t>(total - keep));
    return view;
}

inline std::vector<double> decrypt(const CC& cc, Ctx ct,
                                    const PrivateKey<DCRTPoly>& sk) {
    if (!cc->loaded) {
        // GPU-less session (CKKSContextOptions::skip_gpu_load): FIDESlib's Decrypt wants a
        // device-resident ciphertext; go straight to the OpenFHE host path instead.
        auto& lcc = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
        auto& lct = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto& lsk = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
        lbcrypto::Plaintext lpt;
        lcc->Decrypt(lsk, lct, &lpt);
        return lpt->GetRealPackedValue();
    }
    Ctx view = decrypt_view(cc, ct);
    Plaintext pt;
    cc->Decrypt(view, sk, &pt);
    return pt->GetRealPackedValue();
}

inline Plaintext decrypt_pt(const CC& cc, Ctx ct,
                             const PrivateKey<DCRTPoly>& sk) {
    Ctx view = decrypt_view(cc, ct);
    Plaintext pt;
    cc->Decrypt(view, sk, &pt);
    return pt;
}
