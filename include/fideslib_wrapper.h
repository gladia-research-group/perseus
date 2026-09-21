#pragma once

#include "ckks_types.h"
#include "graph.h"
#include "packing/packed_ctx.h"

#include <CKKS/Ciphertext.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>

namespace FIDESlib::CKKS { void setArcsineOverride(int v); }  // ApproxModEval.cu

#include <any>
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

struct Inference;  // ctor below takes Inference&; full definition in inference.h
struct CKKSContext;

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

inline bool bts_debug_enabled() {
    static const bool enabled = []() {
        const char* v = std::getenv("DEBUG");
        if (!v || !*v) return false;
        const std::string s(v);
        return !(s == "0" || s == "false" || s == "False" || s == "FALSE");
    }();
    return enabled;
}


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
    // trajectories (deg-2 pending-rescale ±1) and was the gen/prefill bind breaker.
    std::unordered_set<std::string>              hint_fire;
    bool                                         hints_bound = false;
    int                                          cache_pin_level = -1;   // placer-chosen KV read level (-1 = unset)
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

    const std::regex entry_re(
        "\\{[^\\{\\}]*\\\"type\\\"\\s*:\\s*\\\"(bootstrap_after_node)\\\"[^\\{\\}]*\\}");
    const std::regex target_var_re("\\\"target_var\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");

    for (std::sregex_iterator it(content.begin(), content.end(), entry_re), end; it != end; ++it) {
        const std::string entry = it->str();

        std::smatch type_match;
        std::smatch target_match;

        if (!std::regex_search(entry, type_match,
                               std::regex("\\\"type\\\"\\s*:\\s*\\\"(bootstrap_after_node)\\\""))) {
            continue;
        }
        if (!std::regex_search(entry, target_match, target_var_re)) {
            continue;
        }

        const std::string type = type_match[1].str();
        const std::string target = target_match[1].str();

        if (type == "bootstrap_after_node") {
            plan.placement_after.insert(target);
        }
    }

    const std::regex levels_dict_re("\\\"final_named_levels\\\"\\s*:\\s*\\{([^{}]+)\\}");
    std::smatch match;
    if (std::regex_search(content, match, levels_dict_re)) {
        std::string dict_str = match[1].str();
        // Match keys like "v_1" and integers like 15
        const std::regex kv_re("\\\"([^\\\"]+)\\\"\\s*:\\s*([0-9]+)");
        for (std::sregex_iterator it(dict_str.begin(), dict_str.end(), kv_re), end; it != end; ++it) {
            plan.expected_levels[(*it)[1].str()] = static_cast<uint32_t>(std::stoul((*it)[2].str()));
        }
    }

    const std::regex producers_dict_re("\\\"final_named_producers\\\"\\s*:\\s*\\{([^{}]+)\\}");
    if (std::regex_search(content, match, producers_dict_re)) {
        std::string dict_str = match[1].str();
        const std::regex kv_re("\\\"([^\\\"]+)\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
        for (std::sregex_iterator it(dict_str.begin(), dict_str.end(), kv_re), end; it != end; ++it) {
            plan.expected_producers[(*it)[1].str()] = (*it)[2].str();
        }
    }

    // Optional per-weight encode levels: {"weight_levels": {"q": 21, "down": 22, ...}}.
    const std::regex wlvls_dict_re("\\\"weight_levels\\\"\\s*:\\s*\\{([^{}]+)\\}");
    if (std::regex_search(content, match, wlvls_dict_re)) {
        std::string dict_str = match[1].str();
        const std::regex kv_re("\\\"([^\\\"]+)\\\"\\s*:\\s*([0-9]+)");
        for (std::sregex_iterator it(dict_str.begin(), dict_str.end(), kv_re), end; it != end; ++it) {
            plan.weight_levels[(*it)[1].str()] = static_cast<uint32_t>(std::stoul((*it)[2].str()));
        }
    }

    const std::regex mlvls_dict_re("\\\"mask_levels\\\"\\s*:\\s*\\{([^{}]+)\\}");
    if (std::regex_search(content, match, mlvls_dict_re)) {
        const std::string dict_str = match[1].str();
        const std::regex site_re("\\\"([^\\\"]+)\\\"\\s*:\\s*\\[([^\\]]*)\\]");
        const std::regex num_re("([0-9]+)");
        for (std::sregex_iterator it(dict_str.begin(), dict_str.end(), site_re), end; it != end; ++it) {
            std::vector<uint32_t> lvls;
            const std::string arr = (*it)[2].str();
            for (std::sregex_iterator n(arr.begin(), arr.end(), num_re), nend; n != nend; ++n)
                lvls.push_back(static_cast<uint32_t>(std::stoul((*n)[1].str())));
            if (!lvls.empty()) plan.mask_levels[(*it)[1].str()] = std::move(lvls);
        }
    }

    // Plan-bound hint decisions: {"hint_fire": ["v_106", ...]}. Key presence (even
    // an empty list) binds every bootstrap_hint decision to the plan.
    const std::regex hint_fire_re("\\\"hint_fire\\\"\\s*:\\s*\\[([^\\]]*)\\]");
    if (std::regex_search(content, match, hint_fire_re)) {
        plan.hints_bound = true;
        const std::string arr = match[1].str();
        const std::regex var_re("\\\"([^\\\"]+)\\\"");
        for (std::sregex_iterator it(arr.begin(), arr.end(), var_re), end; it != end; ++it)
            plan.hint_fire.insert((*it)[1].str());
    }

    // Placer-chosen KV cache read/pin level (rules.cache_pin_level).
    std::smatch cpl_match;
    if (std::regex_search(content, cpl_match, std::regex("\\\"cache_pin_level\\\"\\s*:\\s*([0-9]+)")))
        plan.cache_pin_level = std::stoi(cpl_match[1].str());

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

struct CKKSContext {
    struct RuntimeGraphNode {
        uint64_t id = 0;
        std::string op_type;
        std::vector<std::string> inputs;
        std::string output;
    };

    CC  cc;
    KP  keys;

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
    int active_cache_pin_level = -1;   // placer-chosen KV read level for the live block plan (-1 = unset)
    std::vector<RuntimeGraphNode> expected_graph_nodes;
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
    uint64_t current_runtime_node_id = 0;
    bool placement_plan_enabled = false;
    bool graph_sync_checks_enabled = false;
    std::unordered_set<std::string> planned_warned_;   // dedup keys for one-shot planned-mode warnings

    PublicKey<DCRTPoly>&  pk()  { return keys.publicKey; }
    PrivateKey<DCRTPoly>& sk()  { return keys.secretKey; }

    // TP_PROBE=1: print {max|Re(A)|, max|Im(B)|} of a ct with a label (token-pair lane trace).
    void tp_probe(const std::string& tag, const Ctx& ct) {
        if (!std::getenv("TP_PROBE")) return;
        auto ri = debug_max_abs_re_im(ct);
        fprintf(stderr, "[tp_probe:%s] |Re(A)|=%.5g |Im(B)|=%.5g\n", tag.c_str(), ri.first, ri.second);
    }

    // DEBUG probe: decrypt a ct, return {max|Re|, max|Im|} over slots (token-pair A/B lane check).
    std::pair<double, double> debug_max_abs_re_im(const Ctx& ct) {
        Plaintext pt;
        Ctx c = ct;   // Decrypt takes a non-const Ctx& (mirror the magnitude-capture path)
        cc->Decrypt(c, keys.secretKey, &pt);
        const auto v = pt->GetCKKSPackedValue();
        double mr = 0.0, mi = 0.0;
        for (const auto& z : v) {
            mr = std::max(mr, std::abs(z.real()));
            mi = std::max(mi, std::abs(z.imag()));
        }
        return {mr, mi};
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
            auto pt_dummy = context->MakeCKKSPackedPlaintext(
                dummy, 1, cc->multiplicative_depth - ct_gpu->getLevel());
            auto& skImpl =
                std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(
                    keys.secretKey->pimpl);
            ct_cpu = context->Encrypt(skImpl, pt_dummy);
        }

        FIDESlib::CKKS::GetOpenFHECipherText(ct_cpu, raw_ct);
    }

    size_t free_rotation_steps(const std::vector<int>& steps) {
        return cc->FreeRotationKeys(steps, keys.publicKey);
    }

    void load_rotation_steps(const std::vector<int>& steps) {
        cc->LoadRotationKeys(steps, keys.publicKey);
    }

    int   total_depth = 25; // TODO: check this
    int   btp_overhead = 15;

    // Cached post-bootstrap level (see bootstrap_output_level()).
    uint32_t bts_out_level_     = 0;
    bool     bts_out_level_set_ = false;

    uint32_t bts_iterations = 1;
    uint32_t bts_precision  = 0;
    // dual-slots sparse arcsine precomp (0 = absent) + scope-driven routing
    uint32_t sparse_bts_slots  = 0;
    uint32_t sparse_bts_active = 0;
    // int   min_remaining = 5;  // bootstrap when remaining levels < threshold
    uint32_t total_bootstraps = 0;

    bool op_tally_active = false;
    std::map<std::string, uint64_t> op_tally;

    std::vector<std::string> step_stack;
    StepProfiler profile;

    Ptx const_pt(double val, int level = 0) const {
        uint64_t bits = 0; static_assert(sizeof(bits) == sizeof(val), "double != 64b");
        __builtin_memcpy(&bits, &val, sizeof(bits));
        const uint64_t key = bits ^ (static_cast<uint64_t>(level) << 1);
        auto it = const_pt_cache.find(key);
        if (it != const_pt_cache.end()) return it->second;
        const size_t slots = static_cast<size_t>(cc->GetRingDimension()) / 2;
        Ptx pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(slots, val),
                                             /*noiseScaleDeg=*/1, (uint32_t)level);
        const_pt_cache.emplace(key, pt);
        return pt;
    }

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

    size_t clear_const_pt_cache() {
        const size_t n = const_pt_cache.size() + complex_const_pt_cache.size();
        for (auto* cache : {&const_pt_cache, &complex_const_pt_cache}) {
            for (auto& kv : *cache) {
                Ptx& pt = kv.second;
                if (pt && pt->loaded && pt->gpu != 0) {
                    cc->EvictDevicePlaintext(pt->gpu);
                    pt->gpu = 0; pt->loaded = false;
                }
            }
            cache->clear();
        }
        return n;
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
        current_runtime_node_id = 0;
    }

    void detach_graph_builder() {
        graph_builder.reset();
        ct_vars.clear();
        pt_vars.clear();
        graph_ct_counter = 0;
        graph_pt_counter = 0;
        current_runtime_node_id = 0;
    }

    void clear_bootstrap_plan() {
        placement_after.clear();
        plan_hint_fire.clear();
        plan_hints_bound = false;
        expected_levels.clear();
        expected_producers.clear();
        current_runtime_node_id = 0;
        placement_plan_enabled = false;
        active_cache_pin_level = -1;
    }

    void clear_expected_graph_sync() {
        expected_graph_nodes.clear();
        current_runtime_node_id = 0;
        graph_sync_checks_enabled = false;
    }

    bool load_expected_graph_json(const std::string& path) {
        clear_expected_graph_sync();

        std::ifstream in(path);
        if (!in) {
            return false;
        }

        const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::regex node_re(
            "\\{\\s*\\\"id\\\"\\s*:\\s*([0-9]+)\\s*,\\s*\\\"op_type\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"\\s*,\\s*\\\"inputs\\\"\\s*:\\s*\\[(.*?)\\]\\s*,\\s*\\\"output\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"\\s*\\}");
        const std::regex input_re("\\\"([^\\\"]*)\\\"");

        for (std::sregex_iterator it(content.begin(), content.end(), node_re), end; it != end; ++it) {
            const std::smatch& m = *it;
            RuntimeGraphNode node;
            node.id = static_cast<uint64_t>(std::stoull(m[1].str()));
            node.op_type = m[2].str();

            const std::string inputs_blob = m[3].str();
            for (std::sregex_iterator in_it(inputs_blob.begin(), inputs_blob.end(), input_re), in_end;
                 in_it != in_end; ++in_it) {
                node.inputs.push_back((*in_it)[1].str());
            }

            node.output = m[4].str();
            expected_graph_nodes.push_back(std::move(node));
        }

        graph_sync_checks_enabled = !expected_graph_nodes.empty();
        return graph_sync_checks_enabled;
    }

    void enable_graph_sync_checks(bool enabled = true) {
        graph_sync_checks_enabled = enabled && !expected_graph_nodes.empty();
    }

    bool graph_sync_checks_active() const {
        return graph_sync_checks_enabled && !expected_graph_nodes.empty();
    }

    void verify_expected_level(const std::string& var_name,
                               const std::string& op_type,
                               uint32_t actual_level) const {
        if (!placement_plan_enabled || expected_levels.empty()) {
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
                if (plan_strict()) {
                    std::ostringstream oss;
                    oss << "[plan_level_error] " << var_name << " expected " << it->second
                        << " actual " << actual_level << " step=" << step_path()
                        << " (planned mode is strict)";
                    throw std::runtime_error(oss.str());
                }
                if (bts_debug_enabled()) {
                    std::cout << "[plan_level_warn] " << var_name
                              << " expected " << it->second
                              << " actual " << actual_level << std::endl;
                }
            }
        }
    }

    // A loaded plan is always authoritative: any deviation (level mismatch, weight
    // relevel, unplanned safety-net bootstrap) is a plan/run divergence and throws.
    bool plan_strict() const {
        return placement_plan_enabled;
    }

    void verify_runtime_node_against_expected(const std::string& op_type,
                                              const std::vector<std::string>& inputs,
                                              const std::string& output) const {
        if (!graph_sync_checks_active()) {
            return;
        }

        if (current_runtime_node_id >= expected_graph_nodes.size()) {
            std::ostringstream oss;
            oss << "Runtime/graph sync error: runtime node id " << current_runtime_node_id
                << " exceeds expected graph size " << expected_graph_nodes.size();
            throw std::runtime_error(oss.str());
        }

        const RuntimeGraphNode& expected = expected_graph_nodes[current_runtime_node_id];

        if (expected.id != current_runtime_node_id ||
            expected.op_type != op_type ||
            expected.output != output ||
            expected.inputs != inputs) {
            std::ostringstream oss;
            oss << "Runtime/graph sync mismatch at runtime node " << current_runtime_node_id << "\n"
                << "  expected: id=" << expected.id << ", op=" << expected.op_type
                << ", output=" << expected.output << "\n"
                << "  actual:   id=" << current_runtime_node_id << ", op=" << op_type
                << ", output=" << output << "\n"
                << "  expected_inputs=[";
            for (size_t i = 0; i < expected.inputs.size(); ++i) {
                if (i) oss << ", ";
                oss << expected.inputs[i];
            }
            oss << "] actual_inputs=[";
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (i) oss << ", ";
                oss << inputs[i];
            }
            oss << "]";
            throw std::runtime_error(oss.str());
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
        placement_plan_enabled = !plan.placement_after.empty();
        active_cache_pin_level = plan.cache_pin_level;

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
        throw std::runtime_error(oss.str());
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

    bool load_bootstrap_plan_json(const std::string& path) {
        install_plan_live(parse_bootstrap_plan_file(path));
        return placement_plan_enabled;
    }

    bool planned_bootstraps_enabled() const {
        return placement_plan_enabled;
    }

    void maybe_apply_planned_bootstrap_after(const std::string& var_name, Ctx& ct) {
        if (!placement_plan_enabled || !ct) {
            return;
        }
        if (placement_after.find(var_name) == placement_after.end()) {
            return;
        }
        // Optional: Remove it from the set once applied so it only ever triggers once globally
        placement_after.erase(var_name);

        if (bts_debug_enabled()) {
            std::cout << "Applying planned bootstrap after runtime node " << current_runtime_node_id
                      << " for variable " << var_name << "\n";
        }

        const int in_level = level_for_ct(ct);
        inner_bootstrap(ct);
        const std::string out = var_name + "_planned_bootstrapped";
        name_ct(ct, out, true);
        record_primitive("auto_bootstrap", {var_name}, out, {in_level}, level_for_ct(ct), ct);
    }

    bool graph_enabled() const {
        return graph_builder && graph_builder->enabled();
    }

    bool naming_active() const {
        return graph_enabled() || placement_plan_enabled
            || graph_sync_checks_enabled || bts_debug_enabled();
    }

    bool has_ct_name(const Ctx& ct) const {
        if (!ct) {
            return false;
        }
        const void* key = static_cast<const void*>(ct.get());
        return ct_vars.find(key) != ct_vars.end();
    }

    bool has_pt_name(const Ptx& pt) const {
        if (!pt) {
            return false;
        }
        const void* key = static_cast<const void*>(pt.get());
        return pt_vars.find(key) != pt_vars.end();
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
        if (op_tally_active) ++op_tally[op_type];
        verify_runtime_node_against_expected(op_type, inputs, output);
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
        if (op_tally_active) ++op_tally[op_type];
        verify_runtime_node_against_expected(op_type, inputs, output);
        if (!graph_enabled()) {
            return;
        }
        bool has_noise_level = false;
        int noise_level = -1;
        bool has_max_abs = false;
        double max_abs = 0.0;
        if (output_ct) {
            noise_level = static_cast<int>(output_ct->GetNoiseScaleDeg());
            has_noise_level = true;
            static const bool capture_magnitude = true;
            const long ro = magnitude_reuse_active
                                ? static_cast<long>(magnitude_reuse_ordinal++) : -1;
            if (capture_magnitude && magnitude_reuse_active && !magnitude_reuse_first) {
                if (ro >= 0 && static_cast<size_t>(ro) < magnitude_reuse_ref.size()
                    && magnitude_reuse_has[static_cast<size_t>(ro)]) {
                    max_abs = magnitude_reuse_ref[static_cast<size_t>(ro)];
                    has_max_abs = true;   // borrowed reference: tractable, no decrypt
                }
            } else if (capture_magnitude && !magnitude_capture_suppressed) {
                Plaintext pt;
                Ctx ct_copy = output_ct;
                try {
                    cc->Decrypt(ct_copy, keys.secretKey, &pt);
                    const auto& v = pt->GetRealPackedValue();
                    for (double x : v) {
                        const double a = std::abs(x);
                        if (a > max_abs) max_abs = a;
                    }
                    has_max_abs = true;
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
        graph_builder->add_node(op_type, inputs, output, input_levels, output_level,
                                has_noise_level, noise_level, has_max_abs, max_abs,
                                step_path());
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

    uint32_t level_limit() const {
        if (auto_bts_level_override != 0) {
            return auto_bts_level_override;
        }
        return static_cast<uint32_t>(total_depth - 2);
    }

    uint32_t bootstrap_output_level() {
        if (bts_out_level_set_) return bts_out_level_;

        const uint32_t formula =
            static_cast<uint32_t>(btp_overhead) + (bts_iterations >= 2 ? 1u : 0u);

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
        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        Ctx out_ct = ct->Clone();
        cc->EvalMultMonomialInPlace(out_ct, static_cast<uint32_t>(cc->GetRingDimension() / 2));
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("mult_i", {in_a}, out, {in_a_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "mult_i", level_of(out_ct));
        return out_ct;
    }

    Ctx pair_pack(const Ctx& a_re, const Ctx& b_im) {
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
        if (bts_debug_enabled()) {
            std::cout << "Deliberately bootstrapping ciphertext at level " << level_of(ct) << "\n";
        }

        const int in_level = level_for_ct(ct);
        const std::string in = var_for_ct(ct);

        inner_bootstrap(ct);

        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("deliberate_bootstrap", {in}, out, {in_level}, out_level, ct);
    }

    struct BtsItersScope {
        CKKSContext& c;
        uint32_t saved;
        BtsItersScope(CKKSContext& ctx, uint32_t n)
            : c(ctx), saved(ctx.bts_iterations) { c.bts_iterations = n; }
        ~BtsItersScope() { c.bts_iterations = saved; }
    };

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

    void bootstrap_precise(Ctx& ct) {
        WithStep _w(*this, "bootstrap_precise");
        if (bts_debug_enabled()) {
            std::cout << "Precise (2-iter) bootstrap at level " << level_of(ct) << "\n";
        }
        debug_log_bts_stats(ct, "[bts_input]");

        const int in_level = level_for_ct(ct);
        const std::string in = var_for_ct(ct);

        const uint32_t sp = sparse_bts_active;
        sparse_period_probe(ct, sp);
        if (sp) ct->SetSlots(sp);
        ct = eval_bootstrap_iter(ct, 2, bts_precision);
        if (sp) ct->SetSlots(cc->GetRingDimension() / 2);

        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("deliberate_bootstrap", {in}, out, {in_level}, out_level, ct);
    }

    int sparse_period_probe_mode() {
        static const int m = [] {
            const char* v = std::getenv("SPARSE_PERIOD_PROBE");
            return v && *v ? std::atoi(v) : 0;
        }();
        return m;
    }

    void sparse_period_probe(const Ctx& ct, uint32_t sp) {
        const int mode = sparse_period_probe_mode();
        if (mode == 0 || (mode == 1 && !sp)) return;
        const uint32_t period = sp ? sp : sparse_bts_slots;   // mode 2 on a dense-routed
        if (!period) return;                                  // bts: measure vs the 512 ref
        const uint32_t per = sp ? sp : sparse_bts_slots;   // mode 2 on dense-routed bts:
        if (!per) return;                                   // measure vs the sparse period
        try {
            Plaintext pt;
            Ctx c = ct;
            cc->Decrypt(c, keys.secretKey, &pt);
            const auto& v = pt->GetCKKSPackedValue();
            double am = 0.0, aper = 0.0, im_am = 0.0;
            for (size_t i = 0; i < v.size(); ++i) {
                const double re = v[i].real();
                if (std::abs(re) > am) am = std::abs(re);
                const double b = std::abs(v[i].imag());
                if (b > im_am) im_am = b;
                if (i >= period) {
                    const double d = std::abs(re - v[i % period].real());
                    if (d > aper) aper = d;
                }
            }
            const std::string spath = step_path();
            std::fprintf(stderr,
                         "[sparse_period] aper=%.4g absmax=%.4g rel=%.3g im=%.3g routed=%u step=%s\n",
                         aper, am, am > 0 ? aper / am : 0.0, im_am, sp,
                         spath.empty() ? "?" : spath.c_str());
            std::fflush(stderr);
        } catch (...) {}
    }

    void debug_log_bts_stats(const Ctx& ct, const char* tag) {
        if (!bts_debug_enabled()) return;
        Plaintext pt;
        Ctx ct_copy = ct;
        try {
            cc->Decrypt(ct_copy, keys.secretKey, &pt);
        } catch (const std::exception& e) {
            const std::string sp = step_path();
            std::cout << tag << " step=" << (sp.empty() ? std::string("?") : sp)
                      << " var=" << var_for_ct(ct)
                      << " level=" << level_of(ct)
                      << " deg=" << (ct ? static_cast<int>(ct->GetNoiseScaleDeg()) : -1)
                      << " DECODE_FAILED (" << e.what() << ")" << std::endl;
            return;
        }
        const auto& v = pt->GetRealPackedValue();
        double mn = std::numeric_limits<double>::infinity();
        double mx = -std::numeric_limits<double>::infinity();
        double am = 0.0, sum = 0.0;
        for (double x : v) {
            if (x < mn) mn = x;
            if (x > mx) mx = x;
            double a = std::abs(x);
            if (a > am) am = a;
            sum += x;
        }
        const double avg = v.empty() ? 0.0 : sum / static_cast<double>(v.size());
        constexpr double kBtsRangeWall = 10.0;
        constexpr double kBtsSafeLo    = 1e-2;
        if (am >= kBtsSafeLo && am <= kBtsRangeWall) return;   // in safe window → silent

        const std::ios::fmtflags saved_flags = std::cout.flags();
        const std::streamsize    saved_prec  = std::cout.precision();
        std::cout.unsetf(std::ios::floatfield);   // restore defaultfloat (sticky std::fixed elsewhere)
        std::cout.precision(3);
        const std::string sp = step_path();
        std::cout << tag << " step=" << (sp.empty() ? std::string("?") : sp)
                << " var=" << var_for_ct(ct)
                << " level=" << level_of(ct)
                << " deg=" << (ct ? static_cast<int>(ct->GetNoiseScaleDeg()) : -1)
                << " min="     << mn
                << " max="     << mx
                << " avg="     << avg
                << " |abs|max=" << am
                << std::endl;

        if (am > kBtsRangeWall) {
            std::cout << "[bts_warn] range overflow: |abs|max=" << am
                      << " exceeds EvalMod safe range (~10); bts precision degrades"
                      << std::endl;
        } else if (am > 0.0 && am < kBtsSafeLo) {
            std::cout << "[bts_warn] below safe range: |abs|max=" << am
                      << " < 1e-2 (rel-err >~15%; == noise below the ~1.5e-3 hard floor)"
                      << std::endl;
        }
        std::cout.flags(saved_flags);
        std::cout.precision(saved_prec);
    }

    void inner_bootstrap(Ctx& ct) {
        WithStep _w(*this, "bootstrap");
        debug_log_bts_stats(ct, "[bts_input]");
        const uint32_t sp = sparse_bts_active;
        sparse_period_probe(ct, sp);
        if (sp) ct->SetSlots(sp);
        if (bts_iterations > 1) {
            ct = eval_bootstrap_iter(ct, bts_iterations, bts_precision);
        } else {
            cc->EvalBootstrapInPlace(ct);
            total_bootstraps++;
        }
        if (sp) ct->SetSlots(cc->GetRingDimension() / 2);
    }

    Ctx eval_bootstrap_iter(const Ctx& ct, uint32_t numIterations = 1, uint32_t precision = 0) {
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

    void maybe_bootstrap(Ctx& ct) {
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
            const std::string sp = step_path();
            throw std::runtime_error(
                "[plan_bts_error] unplanned safety-net bootstrap at step=" + sp +
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
        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_ct(other);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_ct(other);
        cc->EvalAddInPlace(ct, other);
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("add_inplace", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "add_inplace", level_of(ct));
        
    }

    void inplace_add(Ctx& ct, Ptx& pt) {
        
        // add map-logic here?
        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_pt(pt);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_pt(pt);
        cc->EvalAddInPlace(ct, pt);
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("add_inplace", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "add_inplace", level_of(ct));
        
    }

    void inplace_add(Ctx& ct, double scalar) {
        
        // add map-logic here?
        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_scalar(scalar);
        cc->EvalAddInPlace(ct, scalar);
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("add_inplace", {in_a, in_b}, out, {in_a_level, -1}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "add_inplace", level_of(ct));
        
    }

    Ctx add(const Ctx& ct, const Ctx& other) {

        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_ct(other);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_ct(other);
        Ctx out_ct = cc->EvalAdd(ct, other);
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("add", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "add", level_of(out_ct));
        
        return out_ct;
    }

    Ctx add(const Ctx& ct, Ptx& pt) {
        
        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_pt(pt);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_pt(pt);
        Ctx out_ct = cc->EvalAdd(ct, pt);
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("add", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "add", level_of(out_ct));
        
        return out_ct;
    }

    Ctx add(const Ctx& ct, double scalar) {
        
        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_scalar(scalar);
        Ctx out_ct = cc->EvalAdd(ct, scalar);
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("add", {in_a, in_b}, out, {in_a_level, -1}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "add", level_of(out_ct));
        
        return out_ct;
    }

    Ctx sub(const Ctx& ct, const Ctx& other) {

        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_ct(other);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_ct(other);
        Ctx out_ct = cc->EvalSub(ct, other);
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("sub_ct", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "sub_ct", level_of(out_ct));
        maybe_bootstrap(out_ct);
        
        return out_ct;
    }

    Ctx sub(const Ctx& ct, Ptx& pt) {
        
        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_pt(pt);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_pt(pt);
        Ctx out_ct = cc->EvalSub(ct, pt);
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("sub", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "sub", level_of(out_ct));
        
        return out_ct;
    }

    Ctx sub(const Ctx& ct, double scalar) {
        
        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_scalar(scalar);
        Ctx out_ct = cc->EvalSub(ct, scalar);
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("sub", {in_a, in_b}, out, {in_a_level, -1}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "sub", level_of(out_ct));
        
        return out_ct;
    }

    void inplace_sub(Ctx& ct, const Ctx& other) {
        
        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_ct(other);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_ct(other);
        cc->EvalSubInPlace(ct, other);
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("sub_inplace_ct", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "sub_inplace_ct", level_of(ct));
        maybe_bootstrap(ct);
        
    }

    void inplace_sub(Ctx& ct, double scalar) {
        
        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_scalar(scalar);
        cc->EvalSubInPlace(ct, scalar);
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("sub_inplace", {in_a, in_b}, out, {in_a_level, -1}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "sub_inplace", level_of(ct));
        
    }

    Ctx mult(const Ctx& ct, const Ctx& other) {

        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_ct(other);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_ct(other);
        if (op_tally_active) ++op_tally["mult_cc"];
        Ctx out_ct = cc->EvalMult(ct, other);
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("mult", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "mult", level_of(out_ct));
        maybe_bootstrap(out_ct);
        
        return out_ct;
    }

    Ctx mult(const Ctx& ct, Ptx& pt) {

        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_pt(pt);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_pt(pt);
        if (op_tally_active) ++op_tally["mult_pt"];
        Ctx out_ct = cc->EvalMult(ct, pt);
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("mult", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "mult", level_of(out_ct));
        maybe_bootstrap(out_ct);
        
        return out_ct;
    }

    Ctx mult(const Ctx& ct, double scalar) {

        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_scalar(scalar);
        if (op_tally_active) ++op_tally["mult_sc"];
        Ctx out_ct = cc->EvalMult(ct, scalar);
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("mult", {in_a, in_b}, out, {in_a_level, -1}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "mult", level_of(out_ct));
        maybe_bootstrap(out_ct);
        
        return out_ct;
    }

    void inplace_mult(Ctx& ct, Ptx& pt) {

        const int in_a_level = level_for_ct(ct);
        const int in_b_level = level_for_pt(pt);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_pt(pt);
        cc->EvalMultInPlace(ct, pt);
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("mult_inplace", {in_a, in_b}, out, {in_a_level, in_b_level}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "mult_inplace", level_of(ct));
        maybe_bootstrap(ct);
        
    }

    void inplace_mult(Ctx& ct, double scalar) {

        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        const std::string in_b = var_for_scalar(scalar);
        cc->EvalMultInPlace(ct, scalar);
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("mult_inplace", {in_a, in_b}, out, {in_a_level, -1}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "mult_inplace", level_of(ct));
        maybe_bootstrap(ct);
        
    }

    void inplace_square(Ctx& ct) {

        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        cc->EvalSquareInPlace(ct);
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("square_inplace", {in_a}, out, {in_a_level}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "square_inplace", level_of(ct));
        maybe_bootstrap(ct);
        
    }

    Ctx square(const Ctx& ct) {

        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        Ctx out_ct = cc->EvalSquare(ct);
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("square", {in_a}, out, {in_a_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "square", level_of(out_ct));
        maybe_bootstrap(out_ct);
        
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
        
        return out_ct;
    }

    std::vector<Ctx> rotate_hoisted(const Ctx& ct, const std::vector<int32_t>& steps) {
        // FHE_HOIST removed 2026-07-05: hoisted batched rotations are ALWAYS used (the per-step
        // rotate fallback is deleted). All captured plans assume hoist-on (it was the default), so
        // the graph node order stays consistent.
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

        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        Ctx out_ct = cc->EvalConjugate(ct);
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("conjugate", {in_a}, out, {in_a_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "conjugate", level_of(out_ct));
        return out_ct;
    }

    void inplace_rotate(Ctx& ct, int32_t index) {

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

    }

    Ctx negate(const Ctx& ct) {
        
        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        Ctx out_ct = cc->EvalNegate(ct);
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("negate", {in_a}, out, {in_a_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "negate", level_of(out_ct));
        maybe_bootstrap(out_ct);
        
        return out_ct;
    }

    void inplace_negate(Ctx& ct) {
        
        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        cc->EvalNegateInPlace(ct);
        const std::string out = set_new_var_for_ct(ct);
        const int out_level = level_for_ct(ct);
        record_primitive("negate_inplace", {in_a}, out, {in_a_level}, out_level, ct);
        maybe_apply_planned_bootstrap_after(out, ct);
        verify_expected_level(out, "negate_inplace", level_of(ct));
        maybe_bootstrap(ct);
        
    }

    // dst <- src, GPU limbs + metadata only (CryptoContextImpl::CopyCiphertextDevice).
    // Same graph semantics as clone() — recorded as "clone", a level/deg passthrough —
    // but reuses the caller's ciphertext instead of minting one, so it skips the
    // copy-ctor's deep copy of the (stale, unused) OpenFHE CPU shadow: ~27 us vs
    // ~605 us measured. For hot loops that need many transient products from an
    // immutable source; dst must already be a loaded ciphertext of the same shape.
    void copy_into(Ctx& dst, const Ctx& src) {
        const int in_a_level = level_for_ct(src);
        const std::string in_a = var_for_ct(src);
        cc->CopyCiphertextDevice(dst, src);
        const std::string out = set_new_var_for_ct(dst);
        const int out_level = level_for_ct(dst);
        record_primitive("clone", {in_a}, out, {in_a_level}, out_level, dst);
        maybe_apply_planned_bootstrap_after(out, dst);
        verify_expected_level(out, "clone", level_of(dst));
    }
    void copy_into(PackedCtx& dst, const PackedCtx& src) { copy_into(dst.ct, src.ct); }

    Ctx clone(const Ctx& ct) {

        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        Ctx out_ct = ct->Clone();
        const std::string out = set_new_var_for_ct(out_ct);
        const int out_level = level_for_ct(out_ct);
        record_primitive("clone", {in_a}, out, {in_a_level}, out_level, out_ct);
        maybe_apply_planned_bootstrap_after(out, out_ct);
        verify_expected_level(out, "clone", level_of(out_ct));
        
        return out_ct;
    }


    void bootstrap_hint(Ctx& ct, int level_threshold, bool account_pending_rescale = false) {

        const int in_a_level = level_for_ct(ct);
        const std::string in_a = var_for_ct(ct);
        bool fire;
        if (placement_plan_enabled && plan_hints_bound) {
            // Plan-bound decision: the planner's final sim already decided this
            // hint on the planned trajectory. Local re-evaluation flips fire/skip
            // on threshold-boundary trajectories (deg-2 pending-rescale ±1) and
            // silently diverges the run from every downstream pin.
            const std::string peek = peek_new_var_for_ct(ct);
            fire = plan_hint_fire.count(peek) > 0;
            if (bts_debug_enabled())
                std::cerr << "[hint_bound] " << peek << " thr=" << level_threshold
                          << " lvl=" << level_of(ct) << " fire=" << fire
                          << " step=" << step_path() << "\n";
        } else {
            int eff_level = static_cast<int>(level_of(ct));
            if (account_pending_rescale && ct && ct->GetNoiseScaleDeg() == 2) eff_level += 1;
            fire = eff_level > level_threshold;
        }
        if (fire) {
            inner_bootstrap(ct);
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
        return PackedCtx{add(a.ct, b.ct), a.packing};
    }
    PackedCtx add(const PackedCtx& a, Ptx& pt)      { return PackedCtx{add(a.ct, pt),     a.packing}; }
    PackedCtx add(const PackedCtx& a, double scalar){ return PackedCtx{add(a.ct, scalar), a.packing}; }

    PackedCtx sub(const PackedCtx& a, const PackedCtx& b) {
        check_packing(a.packing, b.packing);
        return PackedCtx{sub(a.ct, b.ct), a.packing};
    }
    PackedCtx sub(const PackedCtx& a, Ptx& pt)      { return PackedCtx{sub(a.ct, pt),     a.packing}; }
    PackedCtx sub(const PackedCtx& a, double scalar){ return PackedCtx{sub(a.ct, scalar), a.packing}; }

    PackedCtx mult(const PackedCtx& a, const PackedCtx& b) {
        check_packing(a.packing, b.packing);
        return PackedCtx{mult(a.ct, b.ct), a.packing};
    }
    PackedCtx mult(const PackedCtx& a, Ptx& pt)       { return PackedCtx{mult(a.ct, pt),     a.packing}; }
    PackedCtx mult(const PackedCtx& a, double scalar) { return PackedCtx{mult(a.ct, scalar), a.packing}; }

    void inplace_add(PackedCtx& a, const PackedCtx& b) {
        check_packing(a.packing, b.packing);
        inplace_add(a.ct, b.ct);
    }
    void inplace_add(PackedCtx& a, Ptx& pt)       { inplace_add(a.ct, pt); }
    void inplace_add(PackedCtx& a, double scalar) { inplace_add(a.ct, scalar); }

    void inplace_sub(PackedCtx& a, const PackedCtx& b) {
        check_packing(a.packing, b.packing);
        inplace_sub(a.ct, b.ct);
    }
    void inplace_sub(PackedCtx& a, double scalar) { inplace_sub(a.ct, scalar); }

    void inplace_mult(PackedCtx& a, Ptx& pt)       { inplace_mult(a.ct, pt); }
    void inplace_mult(PackedCtx& a, double scalar) { inplace_mult(a.ct, scalar); }

    PackedCtx square(const PackedCtx& a) { return PackedCtx{square(a.ct), a.packing}; }
    void inplace_square(PackedCtx& a)    { inplace_square(a.ct); }

    PackedCtx negate(const PackedCtx& a) { return PackedCtx{negate(a.ct), a.packing}; }
    void inplace_negate(PackedCtx& a)    { inplace_negate(a.ct); }

    PackedCtx clone(const PackedCtx& a) { return PackedCtx{clone(a.ct), a.packing}; }

    PackedCtx rotate(const PackedCtx& a, int32_t index) { return PackedCtx{rotate(a.ct, index), a.packing}; }
    std::vector<PackedCtx> rotate_hoisted(const PackedCtx& a, const std::vector<int32_t>& steps) {
        std::vector<Ctx> raw = rotate_hoisted(a.ct, steps);
        std::vector<PackedCtx> out;
        out.reserve(raw.size());
        for (auto& r : raw) out.push_back(PackedCtx{std::move(r), a.packing});
        return out;
    }
    PackedCtx conjugate(const PackedCtx& a) { return PackedCtx{conjugate(a.ct), a.packing}; }
    void inplace_rotate(PackedCtx& a, int32_t index)    { inplace_rotate(a.ct, index); }

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
    PackedCtx mult_i(const PackedCtx& a) { return PackedCtx{ mult_i(a.ct), a.packing }; }

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

struct CKKSContextOptions {
    // CKKS scheme
    int logN       = 16;
    int depth      = 11;   // usable compute levels (L); +btp_depth_overhead = mult_depth 27
    int scale_bits = 58;

    bool     enable_bootstrap   = true;
    uint32_t btp_depth_overhead = 16;
    std::vector<uint32_t> level_budget = {4, 3};
    uint32_t bootstrap_slots    = 0;   // 0 = N/2
    // dual-slots sparse arcsine precomp (cutmax cascade); 0 = off. Requires
    // FIDESLIB_SPARSE_ARCSINE=1 so ONLY this precomp reserves modall+3.
    uint32_t sparse_bts_slots   = 0;
    std::vector<uint32_t> sparse_level_budget = {};   // empty = level_budget

    int      btp_scale_bits     = 53;
    uint32_t correction_factor  = 0;   // OpenFHE auto (good at 53-bit scale)
    int      first_mod_bits     = 60;
    uint32_t num_large_digits   = 7;
    uint32_t auto_bts_level_override = 24;
    std::vector<uint32_t> chain_sizes_per_level = {};

    uint32_t batch_size     = 0;       // 0 = N/2
    bool ckks_complex_payload = false;

    int      h_weight         = 192;   // 0 = UNIFORM_TERNARY; >0 = SPARSE_TERNARY

    std::vector<int32_t> extra_rot_steps = {};
    std::vector<int32_t> deferred_rot_steps = {};

    uint32_t bts_iterations          = 1;
    uint32_t bts_precision           = 12;
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
    o.bts_iterations     = env_int("BTS_ITERATIONS",     o.bts_iterations);
    o.bts_precision      = env_int("BTS_PRECISION",      o.bts_precision);
    o.sparse_bts_slots   = env_int("SPARSE_BTS_SLOTS",   o.sparse_bts_slots);

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


    // CHAIN_SIZES env removed (always uniform towers). The chain_sizes_per_level field + its
    // empty()-guards below stay dormant — to run a mixed-limb chain for research, set the field in
    // code (e.g. {41x11,45x3,59x10,56x4}); not OpenFHE-128-certifiable, see CLAUDE.md.

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
    show("NUM_LARGE_DIGITS",         o.num_large_digits);
    show("AUTO_BTS_LEVEL",           o.auto_bts_level_override);
    show("H_WEIGHT",                 o.h_weight);
    show("BTS_ITERATIONS",           o.bts_iterations);
    show("BTS_PRECISION",            o.bts_precision);
    show("CKKS_COMPLEX",             (int)o.ckks_complex_payload);
    show("SPARSE_BTS_SLOTS",         o.sparse_bts_slots);
    std::cerr << "  " << std::left << std::setw(22) << "LEVEL_BUDGET" << "= " << lb
              << (std::getenv("LEVEL_BUDGET") ? "  [env]" : "  [def]") << "\n";
    std::string ch;   // always uniform now (no env); dormant per-level field shown if ever set in code
    if (o.chain_sizes_per_level.empty()) ch = "(uniform " + std::to_string(o.btp_scale_bits) + ")";
    else for (size_t i = 0; i < o.chain_sizes_per_level.size(); ++i)
        ch += (i ? ":" : "") + std::to_string(o.chain_sizes_per_level[i]);
    std::cerr << "  " << std::left << std::setw(22) << "limb_chain" << "= " << ch << "  [def]\n";

    return o;
}

inline void log_ckks_params(const CC& cc, const CKKSContextOptions& o, uint32_t slots) {
    const uint32_t ring_dim = cc->GetRingDimension();
    const int mult_depth = o.depth + (o.enable_bootstrap ? o.btp_depth_overhead : 0);
    const int scale      = o.enable_bootstrap ? o.btp_scale_bits : o.scale_bits;
    const int num_limbs  = mult_depth + 1;
    // Mixed chain: logQ is the actual sum of per-level limb sizes (+ q0), not depth*scale.
    double logQ = o.first_mod_bits;
    if (o.chain_sizes_per_level.empty()) logQ += static_cast<double>(mult_depth) * scale;
    else for (uint32_t s : o.chain_sizes_per_level) logQ += s;

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

    if (o.enable_bootstrap && !o.chain_sizes_per_level.empty())
        params.SetScalingModSizePerLevel(o.chain_sizes_per_level);

    if (o.ckks_complex_payload)
        params.SetCKKSDataTypeComplex();
    params.SetFirstModSize(o.first_mod_bits);
    params.SetScalingTechnique(FLEXIBLEAUTO);
    params.SetBatchSize(slots);
    params.SetSecretKeyDist(o.h_weight > 0 ? fideslib::SPARSE_ENCAPSULATED : UNIFORM_TERNARY);
    params.SetNumLargeDigits(o.num_large_digits);
    params.SetKeySwitchTechnique(HYBRID);
    params.SetSecurityLevel(HEStd_128_classic);  // always enforce 128-bit (ENFORCE_128BIT escape removed)
    params.SetRingDim(1 << o.logN);

    auto cc = GenCryptoContext(params);
    log_ckks_params(cc, o, slots);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    if (o.enable_bootstrap) {
        cc->Enable(ADVANCEDSHE);
        cc->Enable(FHE);
    }

    auto kp = cc->KeyGen();
    {   // SECRET-KEY GROUND TRUTH (not options-derived): read the dist back from
        // the LIVE OpenFHE params AND count the generated key's nonzero coeffs.
        // Expected under this stack: dist=UNIFORM_TERNARY (the fideslib API layer
        // forces lbcrypto to uniform on the GPU path; SPARSE_ENCAPSULATED only
        // selects the ENCAPS bootstrap mode, whose sparse h-weight key lives
        // INSIDE bootstrapping), so nonzeros ~ 2N/3 — a sparse main key would
        // read ~h (192).
        auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(
            kp.secretKey->pimpl);
        auto s = skImpl->GetPrivateElement();
        s.SetFormat(Format::COEFFICIENT);
        const auto& t0 = s.GetElementAtIndex(0);
        size_t nz = 0;
        for (uint32_t i = 0; i < t0.GetLength(); ++i)
            if (t0[i] != 0) ++nz;
        int d = -1;
        if (auto rp = std::dynamic_pointer_cast<lbcrypto::CryptoParametersRLWE<lbcrypto::DCRTPoly>>(
                skImpl->GetCryptoParameters()))
            d = static_cast<int>(rp->GetSecretKeyDist());
        static const char* dist_names[] = {"GAUSSIAN", "UNIFORM_TERNARY", "SPARSE_TERNARY"};
        std::cerr << "[ckks_params] SECRET-KEY GROUND TRUTH: openfhe dist="
                  << (d >= 0 && d <= 2 ? dist_names[d] : "?") << " (" << d << ")"
                  << "  sk nonzero coeffs=" << nz << "/" << t0.GetLength()
                  << "  (uniform~2N/3=" << (2 * t0.GetLength() / 3)
                  << ", sparse~h=" << o.h_weight << ")"
                  << "  bts mode=" << (o.h_weight > 0 ? "ENCAPS(sparse-inside-bootstrap)"
                                                      : "UNIFORM") << "\n";
    }
    cc->EvalMultKeyGen(kp.secretKey);

    std::vector<int32_t> rot_steps;
    for (auto r : o.extra_rot_steps) rot_steps.push_back(r);
    std::sort(rot_steps.begin(), rot_steps.end());
    rot_steps.erase(std::unique(rot_steps.begin(), rot_steps.end()), rot_steps.end());
    if (!rot_steps.empty()) {
        cc->EvalRotateKeyGen(kp.secretKey, rot_steps);
    }

    if (o.enable_bootstrap) {
        uint32_t btp_slots = (o.bootstrap_slots == 0) ? slots : o.bootstrap_slots;
        cc->EvalBootstrapSetup(level_budget, {0, 0}, btp_slots, o.correction_factor);
        cc->EvalBootstrapKeyGen(kp.secretKey, btp_slots);
        if (o.sparse_bts_slots > 0 && o.sparse_bts_slots < slots) {
            // second precomp for the dual-slots sparse arcsine path; under
            // FIDESLIB_SPARSE_ARCSINE=1 only THIS setup reserves modall+3
            const auto& slb = o.sparse_level_budget.empty()
                ? level_budget : o.sparse_level_budget;
            cc->EvalBootstrapSetup(slb, {0, 0}, o.sparse_bts_slots,
                                   o.correction_factor);
            cc->EvalBootstrapKeyGen(kp.secretKey, o.sparse_bts_slots);
        }
    }

    cc->deferred_rotation_indexes = o.deferred_rot_steps;
    cc->LoadContext(kp.publicKey);

    auto ctx  = std::make_shared<CKKSContext>();
    ctx->cc   = cc;
    ctx->keys = std::move(kp);
    ctx->total_depth  = o.depth + btp_depth_overhead;
    ctx->btp_overhead = btp_depth_overhead;
    ctx->bts_iterations          = o.bts_iterations;
    ctx->bts_precision           = o.bts_precision;
    ctx->auto_bts_level_override = o.auto_bts_level_override;
    ctx->complex_payload         = o.ckks_complex_payload;
    ctx->sparse_bts_slots        = (o.sparse_bts_slots < slots) ? o.sparse_bts_slots : 0;

    return ctx;
}

inline Ptx encode(const CC& cc, const std::vector<double>& values, int level = 0) {
    return cc->MakeCKKSPackedPlaintext(values, /*noiseScaleDeg=*/1, (uint32_t)level);
}

inline Ptx encode(const CC& cc, const std::vector<std::complex<double>>& values,
                   int level = 0) {
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

inline std::vector<double> decrypt(const CC& cc, Ctx ct,
                                    const PrivateKey<DCRTPoly>& sk) {
    Plaintext pt;
    cc->Decrypt(ct, sk, &pt);
    return pt->GetRealPackedValue();
}

inline Plaintext decrypt_pt(const CC& cc, Ctx ct,
                             const PrivateKey<DCRTPoly>& sk) {
    Plaintext pt;
    cc->Decrypt(ct, sk, &pt);
    return pt;
}
