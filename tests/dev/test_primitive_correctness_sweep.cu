#include "ckks_fixture.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace test_helpers;

namespace {

using PrimitiveCorrectnessSweepTest = CkksFixture;

// Walk a ct to a target consumed level with value-preserving mults (deg-2
// FLEXIBLEAUTO states, like the model's working band).
Ctx at_level(CKKSContext& f, const std::vector<double>& v, int target) {
    Ctx ct = encrypt(f.cc, encode(f.cc, v), f.pk());
    int guard = 0;
    while (static_cast<int>(level_of(ct)) < target && guard++ < 64)
        ct = f.mult(ct, 1.0);
    return ct;
}

double max_err(CKKSContext& f, Ctx& ct, const std::vector<double>& ref) {
    auto out = decrypt_slots(f, ct);
    double e = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
        e = std::max(e, std::fabs(out[i] - ref[i]));
    return e;
}


TEST_F(PrimitiveCorrectnessSweepTest, PrimitiveVsOperatingLevel) {
    const int lo    = static_cast<int>(fhe().bootstrap_output_level());
    const int limit = static_cast<int>(fhe().level_limit());
    const int hi    = std::min(limit, static_cast<int>(fhe().total_depth) - 2);

    std::mt19937 gen(1234);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    std::vector<double> v(slots());
    for (auto& x : v) x = dist(gen);

    std::vector<double> half(slots(), 0.5), ones(slots(), 1.0);

    struct Prim {
        const char* name;
        std::function<Ctx(Ctx&)> run;
        std::function<std::vector<double>()> ref;
    };
    auto& f = fhe();
    // rotate-add over 1024 lanes sums whole random slots -> huge values; use a
    // small per-lane magnitude for tree-based prims so output stays O(1).
    std::vector<double> vsmall(v);
    for (auto& x : vsmall) x /= 1024.0;

    std::vector<Prim> prims = {
        {"mult_scalar", [&](Ctx& c) { return f.mult(c, 0.5); },
         [&] { auto r = v; for (auto& x : r) x *= 0.5; return r; }},
        {"mult_pt", [&](Ctx& c) {
             const int el = static_cast<int>(level_of(c)) +
                            (c->GetNoiseScaleDeg() == 2 ? 1 : 0);
             Ptx pt = encode(f.cc, half, el);
             return f.mult(c, pt);
         },
         [&] { auto r = v; for (auto& x : r) x *= 0.5; return r; }},
        {"mult_ct", [&](Ctx& c) {
             Ctx o = encrypt(f.cc, encode(f.cc, ones), f.pk());
             return f.mult(c, o);
         },
         [&] { return v; }},
        {"square", [&](Ctx& c) { return f.mult(c, c); },
         [&] { auto r = v; for (auto& x : r) x *= x; return r; }},
        {"rotate_pair", [&](Ctx& c) {
             Ctx r = f.rotate(c, 1);
             return f.rotate(r, -1);
         },
         [&] { return v; }},
        {"conj_fence", [&](Ctx& c) {
             Ctx cj = f.conjugate(c);
             f.inplace_add(c, cj);
             return f.mult(c, 0.5);
         },
         [&] { return v; }},
        {"add_pt", [&](Ctx& c) {
             const int el = static_cast<int>(level_of(c)) +
                            (c->GetNoiseScaleDeg() == 2 ? 1 : 0);
             Ptx pt = encode(f.cc, ones, el);
             return f.add(c, pt);
         },
         [&] { auto r = v; for (auto& x : r) x += 1.0; return r; }},
    };

    for (auto& p : prims) {
        const auto ref = p.ref();
        double ref_amp = 0.0;
        for (auto x : ref) ref_amp = std::max(ref_amp, std::fabs(x));
        for (int L = lo; L <= hi; ++L) {
            Ctx ct = at_level(f, v, L);
            const int in_lvl = static_cast<int>(level_of(ct));
            const int in_deg = static_cast<int>(ct->GetNoiseScaleDeg());
            if (in_lvl < L) break;  // walk stalled: chain exhausted
            double emax;
            double ms = -1.0;
            int out_lvl = -1, out_deg = -1;
            try {
                cudaDeviceSynchronize();
                auto t0 = std::chrono::steady_clock::now();
                Ctx out = p.run(ct);
                cudaDeviceSynchronize();
                ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                out_lvl = static_cast<int>(level_of(out));
                out_deg = static_cast<int>(out->GetNoiseScaleDeg());
                emax = max_err(f, out, ref);
            } catch (const std::exception& e) {
                std::cout << "[prim_sweep] prim=" << p.name << " in_lvl=" << in_lvl
                          << " THREW: " << e.what() << std::endl;
                continue;
            }
            std::cout << "[prim_sweep] prim=" << p.name << " in_lvl=" << in_lvl
                      << " in_deg=" << in_deg << " out_lvl=" << out_lvl
                      << " out_deg=" << out_deg << " ms=" << ms
                      << " err_max=" << emax
                      << " rel=" << emax / std::max(1.0, ref_amp) << std::endl;
            EXPECT_LT(emax / std::max(1.0, ref_amp), 0.05)
                << p.name << " fails at operating level " << in_lvl;
        }
    }

    // tree-based prims (rotate-add sum, mini-linear) with small lanes
    struct TreePrim {
        const char* name;
        bool with_weights;
    };
    for (TreePrim tp : {TreePrim{"rotadd_tree", false}, TreePrim{"linear_mini", true}}) {
        std::vector<double> ref(slots(), 0.0);
        for (int i = 0; i < slots(); ++i)
            for (int k = 0; k < 1024; ++k) ref[i] += vsmall[(i + k) % slots()];
        if (tp.with_weights)
            for (auto& x : ref) x *= 0.25;  // 0.5 weight * 0.5 mask
        double ref_amp = 0.0;
        for (auto x : ref) ref_amp = std::max(ref_amp, std::fabs(x));
        for (int L = lo; L <= hi; ++L) {
            Ctx ct = at_level(f, vsmall, L);
            const int in_lvl = static_cast<int>(level_of(ct));
            if (in_lvl < L) break;
            try {
                cudaDeviceSynchronize();
                auto t0 = std::chrono::steady_clock::now();
                Ctx s = ct;
                if (tp.with_weights) {
                    const int el = static_cast<int>(level_of(s)) +
                                   (s->GetNoiseScaleDeg() == 2 ? 1 : 0);
                    Ptx w = encode(f.cc, half, el);
                    s = f.mult(s, w);
                }
                for (int st = 1; st <= 512; st *= 2) {
                    Ctx r = f.rotate(s, st);
                    f.inplace_add(s, r);
                }
                if (tp.with_weights) {
                    const int el = static_cast<int>(level_of(s)) +
                                   (s->GetNoiseScaleDeg() == 2 ? 1 : 0);
                    Ptx m = encode(f.cc, half, el);
                    s = f.mult(s, m);
                }
                cudaDeviceSynchronize();
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                const double emax = max_err(f, s, ref);
                std::cout << "[prim_sweep] prim=" << tp.name << " in_lvl=" << in_lvl
                          << " out_lvl=" << level_of(s) << " ms=" << ms
                          << " err_max=" << emax
                          << " rel=" << emax / std::max(1.0, ref_amp) << std::endl;
                EXPECT_LT(emax / std::max(1.0, ref_amp), 0.05)
                    << tp.name << " fails at operating level " << in_lvl;
            } catch (const std::exception& e) {
                std::cout << "[prim_sweep] prim=" << tp.name << " in_lvl=" << in_lvl
                          << " THREW: " << e.what() << std::endl;
            }
        }
    }
}

TEST_F(PrimitiveCorrectnessSweepTest, BtsInputOvershoot) {
    const int total = static_cast<int>(fhe().total_depth);
    const int lo    = static_cast<int>(fhe().level_limit());
    std::mt19937 gen(99);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    std::vector<double> v(slots());
    for (auto& x : v) x = dist(gen);

    for (int L = lo; L <= total - 1; ++L) {
        Ctx ct = at_level(fhe(), v, L);
        const int lvl = static_cast<int>(level_of(ct));
        if (lvl < L) break;  // walk stalled (chain exhausted)
        Ctx bct = fhe().clone(ct);
        double emax;
        try {
            fhe().bootstrap(bct);
            emax = max_err(fhe(), bct, v);
        } catch (const std::exception& e) {
            std::cout << "[bts_overshoot] in_lvl=" << lvl
                      << " THREW: " << e.what() << std::endl;
            continue;
        }
        std::cout << "[bts_overshoot] in_lvl=" << lvl
                  << " deg=" << ct->GetNoiseScaleDeg()
                  << " err_max=" << emax << std::endl;
    }
}

// ---------------------------------------------------------------------------
// PrimitiveLevelAlignmentGain
//
// Cross-references the per-primitive cost/precision curve (measured here, by
// sweeping each isolated CKKS primitive over the operating band) against the
// levels at which those primitives ACTUALLY run in the current decode plan
// (bootstrap_placements/planned_head_diff, the cheb-fold + head_reduce +
// gpt2_diff "hoisted diff" plan), to answer three questions:
//
//   (plan dir default: bootstrap_placements/planned — the settled full optimised
//    stack: cheb-fold + head_reduce + gpt2_diff + mask_cache + rotation-hoist.)
//
//   1. CURRENT  — at which CKKS level does each primitive fire today, and how
//      often (count-weighted), summed over all 13 plan blocks.
//   2. BEST     — which level is cheapest while still correct (rel < 5%). Since
//      level = #levels-consumed (GetLevel), a HIGHER level == FEWER RNS limbs
//      == cheaper op; the bootstrap lands at the expensive end (lvl 16) and ops
//      drift up toward level_limit (24) getting progressively cheaper.
//   3. GAIN     — wall-time recoverable by aligning each primitive to its
//      cheapest correct level (theoretical floor), plus the marginal payoff of
//      shifting the whole operating band up one level (the bootstrap-landing
//      lever): sum_op count * dms/dL.
//
// Fully self-contained: reads only the test fixture + the plan JSON via
// json_utils; touches no model code.
// ---------------------------------------------------------------------------

// Parse a flat {"name": int, ...} map living under `key` in a JSON document.
std::unordered_map<std::string, int> parse_named_int_map(const std::string& text,
                                                         const char* key) {
    std::unordered_map<std::string, int> out;
    const std::string needle = std::string("\"") + key + "\"";
    size_t k = text.find(needle);
    if (k == std::string::npos) return out;
    size_t brace = text.find('{', k);
    if (brace == std::string::npos) return out;
    int depth = 0;
    size_t end = std::string::npos;
    for (size_t i = brace; i < text.size(); ++i) {
        if (text[i] == '{') ++depth;
        else if (text[i] == '}') { if (--depth == 0) { end = i; break; } }
    }
    if (end == std::string::npos) return out;
    size_t p = brace + 1;
    while (p < end) {
        size_t q = text.find('"', p);
        if (q == std::string::npos || q >= end) break;
        size_t q2 = text.find('"', q + 1);
        if (q2 == std::string::npos || q2 >= end) break;
        std::string name = text.substr(q + 1, q2 - q - 1);
        size_t colon = text.find(':', q2);
        if (colon == std::string::npos || colon >= end) break;
        size_t r = colon + 1;
        while (r < end && std::isspace(static_cast<unsigned char>(text[r]))) ++r;
        int sign = 1;
        if (r < end && text[r] == '-') { sign = -1; ++r; }
        int val = 0; bool any = false;
        while (r < end && std::isdigit(static_cast<unsigned char>(text[r]))) {
            val = val * 10 + (text[r] - '0'); ++r; any = true;
        }
        if (any) out[name] = sign * val;
        p = r;
    }
    return out;
}

// Extract (op_type, output) for every node in the "nodes" array of a var-graph.
std::vector<std::pair<std::string, std::string>> parse_var_graph_nodes(
    const std::string& text) {
    std::vector<std::pair<std::string, std::string>> out;
    size_t k = text.find("\"nodes\"");
    if (k == std::string::npos) return out;
    size_t arr = text.find('[', k);
    if (arr == std::string::npos) return out;
    int depth = 0;
    size_t obj_start = std::string::npos;
    for (size_t i = arr; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '{') { if (depth == 0) obj_start = i; ++depth; }
        else if (c == '}') {
            if (--depth == 0 && obj_start != std::string::npos) {
                const std::string obj = text.substr(obj_start, i - obj_start + 1);
                try {
                    out.emplace_back(json_utils::find_string_field(obj, "op_type"),
                                     json_utils::find_string_field(obj, "output"));
                } catch (...) { /* skip malformed node */ }
                obj_start = std::string::npos;
            }
        } else if (c == ']' && depth == 0) {
            break;  // end of nodes array
        }
    }
    return out;
}

// Map a captured graph op_type onto the isolated cost primitive that models it.
// Returns nullptr for zero-cost / metadata ops (hint, level_hint) that carry no
// homomorphic work.
const char* op_type_to_prim(const std::string& op) {
    if (op == "mult" || op == "mult_inplace")     return "mult_pt";   // BSGS linear bulk is pt-mults
    if (op == "square" || op == "square_inplace") return "square";
    if (op == "rotate")                            return "rotate";
    if (op == "conjugate")                         return "conjugate";
    if (op == "clone")                             return "clone";
    if (op == "add" || op == "add_inplace" ||
        op == "negate" || op == "sub_inplace_ct" ||
        op == "sub" || op == "sub_inplace")        return "add_pt";   // all ~free rescale-less adds
    return nullptr;  // hint, level_hint, ...
}

double median_of(std::vector<double> v) {
    if (v.empty()) return -1.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

TEST_F(PrimitiveCorrectnessSweepTest, PrimitiveLevelAlignmentGain) {
    auto& f = fhe();
    const int lo = static_cast<int>(f.bootstrap_output_level());
    const int hi = std::min(static_cast<int>(f.level_limit()),
                            static_cast<int>(f.total_depth) - 2);

    std::mt19937 gen(2024);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    std::vector<double> v(slots());
    for (auto& x : v) x = dist(gen);
    std::vector<double> half(slots(), 0.5), ones(slots(), 1.0);

    // --- isolated single-op cost primitives (one homomorphic op each) ---------
    // Operands (plaintext / ct) are PRE-BUILT once per level outside the timed
    // region (real decode uses pre-encoded weights), so we time only the op.
    // op_* are reassigned per level; the lambdas read them by reference.
    Ptx op_half, op_ones;
    Ctx op_ct_ones;
    struct CostPrim {
        const char* name;
        std::function<Ctx(Ctx&)> run;
        std::function<std::vector<double>(const std::vector<double>&)> ref;
    };
    const std::vector<CostPrim> prims = {
        {"mult_scalar", [&](Ctx& c) { return f.mult(c, 0.5); },
         [](const std::vector<double>& x){ auto r=x; for(auto&e:r)e*=0.5; return r; }},
        {"mult_pt", [&](Ctx& c) { return f.mult(c, op_half); },
         [](const std::vector<double>& x){ auto r=x; for(auto&e:r)e*=0.5; return r; }},
        {"mult_ct", [&](Ctx& c) { return f.mult(c, op_ct_ones); },
         [](const std::vector<double>& x){ return x; }},
        {"square", [&](Ctx& c) { return f.mult(c, c); },
         [](const std::vector<double>& x){ auto r=x; for(auto&e:r)e*=e; return r; }},
        {"rotate", [&](Ctx& c) { return f.rotate(c, 1); },
         [&](const std::vector<double>& x){ std::vector<double> r(x.size());
             for(size_t i=0;i<x.size();++i) r[i]=x[(i+1)%x.size()]; return r; }},
        {"conjugate", [&](Ctx& c) { return f.conjugate(c); },
         [](const std::vector<double>& x){ return x; }},  // real input -> conj == identity
        {"clone", [&](Ctx& c) { return f.clone(c); },
         [](const std::vector<double>& x){ return x; }},
        {"add_pt", [&](Ctx& c) { return f.add(c, op_ones); },
         [](const std::vector<double>& x){ auto r=x; for(auto&e:r)e+=1.0; return r; }},
    };

    constexpr int kReps = 7;
    // tab[prim][in_level] = one measured sample. A sample is "clean" only if it
    // is correct (rel<5%) AND no bootstrap fired (out_lvl stayed below the
    // level_limit). A consuming op (mult/square) run at in_level = limit-1 lands
    // at the limit and auto-bootstraps (~20x cost) -> that is a bootstrap probe,
    // NOT the primitive's own cost, so it must be excluded from the cost model.
    struct Sample { double ms = -1.0; double rel = 1e9; int out_lvl = -1; bool clean = false; };
    std::map<std::string, std::map<int, Sample>> tab;
    const int llimit = static_cast<int>(f.level_limit());

    std::cout << "\n[align] === per-primitive cost/precision sweep (in_level "
              << lo << ".." << hi << "; level=#levels-consumed; level_limit=" << llimit
              << "; out_L dropping back to " << lo
              << " => auto-bootstrap fired, excluded from cost model) ===\n";
    for (int L = lo; L <= hi; ++L) {
        Ctx base = at_level(f, v, L);
        if (static_cast<int>(level_of(base)) < L) break;  // chain exhausted
        // Pre-encode plaintext operands at the level the op will consume them at
        // (predict the lazy rescale: deg-2 ct rescales +1 before the op).
        const int el = static_cast<int>(level_of(base)) +
                       (base->GetNoiseScaleDeg() == 2 ? 1 : 0);
        op_half    = encode(f.cc, half, el);
        op_ones    = encode(f.cc, ones, el);
        op_ct_ones = encrypt(f.cc, encode(f.cc, ones), f.pk());  // mult_ct (informational)
        for (const auto& p : prims) {
            const std::vector<double> ref = p.ref(v);
            Sample s;
            std::vector<double> samples;
            try {
                Ctx warm = p.run(base);  // warm-up + correctness/level probe
                cudaDeviceSynchronize();
                s.out_lvl = static_cast<int>(level_of(warm));
                double emax = max_err(f, warm, ref), amp = 0.0;
                for (double x : ref) amp = std::max(amp, std::fabs(x));
                s.rel = emax / std::max(1.0, amp);
                // WARMED + BATCHED timing. A single op between two cudaDeviceSync is
                // floored at ~3.8 ms of launch+sync overhead -> swamps the real (limb-
                // bound) cost and falsely reports "level-flat" (clone==rotate). Amortize
                // the launch/sync over kBatch back-to-back ops (one sync per batch), like
                // test_linear_timing's tight loop. base is unmodified by p.run (each call
                // returns a fresh ct), so the batch is well-defined.
                constexpr int kBatch = 32;
                for (int w = 0; w < 3; ++w) { Ctx wo = p.run(base); (void)wo; }  // warmup
                cudaDeviceSynchronize();
                for (int r = 0; r < kReps; ++r) {
                    auto t0 = std::chrono::steady_clock::now();
                    for (int b = 0; b < kBatch; ++b) { Ctx out = p.run(base); (void)out; }
                    cudaDeviceSynchronize();
                    samples.push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count() / kBatch);
                }
            } catch (const std::exception& e) {
                std::cout << "[align] prim=" << p.name << " in_L=" << L
                          << " THREW: " << e.what() << "\n";
                continue;
            }
            s.ms = median_of(samples);
            // A bootstrap fired iff the output level DROPPED back to the
            // bootstrap-output level (lo) from a higher input (lo==bts_out_level):
            // a consuming op at in_L=limit-1 lands on the limit and auto-bts, which
            // resets to ~lo and costs ~20x. That is a bootstrap probe, not the op.
            const bool bts = (s.out_lvl <= lo && L > lo);
            s.clean = (s.rel < 0.05) && !bts;
            tab[p.name][L] = s;
            std::cout << "[align] prim=" << p.name << " in_L=" << L
                      << " out_L=" << s.out_lvl << " ms=" << s.ms << " rel=" << s.rel
                      << (s.clean ? "" : (bts ? "  (BOOTSTRAP-TRIGGERED)" : "  (INFEASIBLE)"))
                      << "\n";
        }
    }

    // Per-primitive cost model over CLEAN samples only (correct + no bootstrap).
    struct PrimModel {
        int best_level = -1, lo_key = -1, hi_key = -1;
        double best_ms = 0.0, med_ms = 0.0, min_ms = 0.0, max_ms = 0.0;
        double slope = 0.0, spread_pct = 0.0;
        bool consuming = false;  // op advances the level (mult/square) vs not (rotate/add)
    };
    std::map<std::string, PrimModel> model;
    for (const auto& p : prims) {
        PrimModel pm;
        std::vector<double> clean;
        double bm = 1e18, n=0, sx=0, sy=0, sxx=0, sxy=0;
        for (const auto& [L, s] : tab[p.name]) {
            if (s.out_lvl > L) pm.consuming = true;
            if (!s.clean) continue;
            clean.push_back(s.ms);
            if (pm.lo_key < 0) pm.lo_key = L;
            pm.hi_key = L;
            n+=1; sx+=L; sy+=s.ms; sxx+=double(L)*L; sxy+=double(L)*s.ms;
            if (s.ms < bm) { bm = s.ms; pm.best_level = L; pm.best_ms = s.ms; }
        }
        if (!clean.empty()) {
            pm.med_ms = median_of(clean);
            pm.min_ms = *std::min_element(clean.begin(), clean.end());
            pm.max_ms = *std::max_element(clean.begin(), clean.end());
            pm.spread_pct = pm.med_ms > 0 ? 100.0 * (pm.max_ms - pm.min_ms) / pm.med_ms : 0.0;
        }
        if (n >= 2 && (n*sxx - sx*sx) != 0.0) pm.slope = (n*sxy - sx*sy)/(n*sxx - sx*sx);
        model[p.name] = pm;
    }

    // Cost of `prim` whose plan OUTPUT level is histL. A consuming op ran with
    // input = histL-1; a non-consuming op with input = histL. Clamp to clean range.
    auto ms_at = [&](const std::string& prim, int histL) -> double {
        const PrimModel& pm = model[prim];
        if (pm.lo_key < 0) return 0.0;
        int key = pm.consuming ? histL - 1 : histL;
        key = std::max(pm.lo_key, std::min(pm.hi_key, key));
        const auto it = tab[prim].find(key);
        return (it != tab[prim].end()) ? it->second.ms : pm.med_ms;
    };

    // --- cost model summary (does cost actually vary with level?) --------------
    std::cout << "\n[align] === cost model (CLEAN samples only) — does cost track level? ===\n";
    std::cout << "[align] prim         clean_band  med_ms  min_ms  max_ms  spread%   slope(ms/lvl)  consumes_lvl\n";
    for (const auto& p : prims) {
        const PrimModel& pm = model[p.name];
        char ln[256];
        std::snprintf(ln, sizeof(ln),
            "%-11s  %2d..%-2d    %6.3f  %6.3f  %6.3f  %6.1f    %+9.4f      %s",
            p.name, pm.lo_key, pm.hi_key, pm.med_ms, pm.min_ms, pm.max_ms,
            pm.spread_pct, pm.slope, pm.consuming ? "yes" : "no");
        std::cout << "[align] " << ln << "\n";
    }

    // --- read the current plan's per-op level histogram (all blocks) ----------
    const char* env_dir = std::getenv("FHE_BOOTSTRAP_PLACEMENTS_DIR");
    const std::string plan_dir =
        (env_dir && *env_dir) ? std::string(env_dir)
                              : std::string("bootstrap_placements/planned");
    std::cout << "\n[align] plan dir = " << plan_dir << "\n";

    // op_type -> (level -> count), aggregated over every block file present.
    std::map<std::string, std::map<int, long>> op_level_count;
    int blocks_read = 0;
    for (int b = 0; b <= 12; ++b) {
        const std::string pl = plan_dir + "/block_" + std::to_string(b) + "_placement.json";
        const std::string vg = plan_dir + "/block_" + std::to_string(b) + "_var_graph.json";
        std::string pl_txt, vg_txt;
        try {
            pl_txt = json_utils::read_file_to_string(pl);
            vg_txt = json_utils::read_file_to_string(vg);
        } catch (...) { continue; }
        const auto levels = parse_named_int_map(pl_txt, "final_named_levels");
        const auto nodes  = parse_var_graph_nodes(vg_txt);
        for (const auto& [op, outv] : nodes) {
            auto it = levels.find(outv);
            if (it == levels.end()) continue;
            op_level_count[op][it->second] += 1;
        }
        ++blocks_read;
    }
    std::cout << "[align] blocks parsed = " << blocks_read << "\n";
    if (blocks_read == 0) {
        std::cout << "[align] no plan files found under '" << plan_dir
                  << "'; set FHE_BOOTSTRAP_PLACEMENTS_DIR (cost model above still valid).\n";
    }

    // --- Q3a: gain from LEVEL alignment (each op -> cheapest clean level) ------
    std::cout << "\n[align] === level-alignment gain: current level vs cheapest clean level ===\n";
    std::cout << "[align] op->prim            fires   meanL   ms_now    bestL  ms_best   gain_ms  gain%\n";
    double tot_now = 0.0, tot_best = 0.0;
    long tot_fires = 0;
    std::map<std::string, long> prim_fires;  // for the cost-by-count table below
    for (const auto& [op, hist] : op_level_count) {
        const char* prim = op_type_to_prim(op);
        long fires = 0; double sumL = 0.0;
        for (const auto& [L, c] : hist) { fires += c; sumL += double(L) * c; }
        const double meanL = fires ? sumL / fires : 0.0;
        if (!prim || model[prim].best_level < 0) {
            std::cout << "[align] " << op << " fires=" << fires << " meanL=" << meanL
                      << "  [metadata/no cost model -> excluded]\n";
            continue;
        }
        prim_fires[prim] += fires;
        const PrimModel& pm = model[prim];
        double now = 0.0;
        for (const auto& [L, c] : hist) now += double(c) * ms_at(prim, L);
        const double best = double(fires) * pm.best_ms;
        tot_now += now; tot_best += best; tot_fires += fires;
        char line[256];
        std::snprintf(line, sizeof(line),
            "%-18s %6ld  %5.1f  %9.1f  %5d  %8.1f  %8.1f  %5.1f",
            (std::string(op) + "->" + prim).c_str(), fires, meanL, now,
            pm.best_level, best, now - best, now > 0 ? 100.0 * (now - best) / now : 0.0);
        std::cout << "[align] " << line << "\n";
    }
    const double align_pct = tot_now > 0 ? 100.0 * (tot_now - tot_best) / tot_now : 0.0;
    std::cout << "[align] " << std::string(86, '-') << "\n";
    std::cout << "[align] LEVEL-ALIGNMENT TOTAL fires=" << tot_fires
              << "  ms_now=" << tot_now << "  ms_aligned=" << tot_best
              << "  gain=" << (tot_now - tot_best) << " ms (" << align_pct
              << "%, idealized floor: every op at its cheapest clean level)\n";

    // --- Q3b: where the time ACTUALLY goes (op-count x clean per-op cost) ------
    // This is the real lever once cost is shown to be level-flat above.
    std::cout << "\n[align] === cost attribution by op COUNT (the real lever) ===\n";
    double grand = 0.0;
    std::vector<std::pair<std::string, double>> ranked;
    for (const auto& [prim, fr] : prim_fires) {
        const double cost = double(fr) * model[prim].med_ms;
        ranked.emplace_back(prim, cost);
        grand += cost;
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    for (const auto& [prim, cost] : ranked) {
        char line[200];
        std::snprintf(line, sizeof(line),
            "%-11s fires=%6ld  med_ms=%6.3f  total=%9.1f ms  (%5.1f%%)",
            prim.c_str(), prim_fires[prim], model[prim].med_ms, cost,
            grand > 0 ? 100.0 * cost / grand : 0.0);
        std::cout << "[align] " << line << "\n";
    }

    // --- verdict (data-driven: warmed/batched timing reveals the limb-bound slope) --
    double max_spread = 0.0;
    for (const auto& p : prims) max_spread = std::max(max_spread, model[p.name].spread_pct);
    const bool limb_bound = max_spread > 15.0;   // a real RNS-limb gradient, not noise
    std::cout << "\n[align] === verdict ===\n";
    std::cout << "[align] per-op cost spread across the band up to " << max_spread << "%; "
              << (limb_bound ? "cost is LIMB-bound (cheaper at higher level / fewer RNS limbs)."
                             : "cost is ~level-FLAT (launch-bound).") << "\n";
    if (limb_bound) {
        std::cout << "[align] level placement IS a lever: aligning every op to its cheapest clean\n";
        std::cout << "[align] level recovers ~" << align_pct << "% (idealized floor). Realizable per\n";
        std::cout << "[align] op-type = gain_ms column above; rank there for the best 'launch'. The\n";
        std::cout << "[align] catch: cheap = high level = few limbs = near the bootstrap wall, so ops\n";
        std::cout << "[align] must be scheduled just-before-refresh, bounded by data deps.\n";
    } else {
        std::cout << "[align] aligning to a 'cheaper' level recovers only ~" << align_pct
                  << "% -- the lever is op COUNT (rotation hoisting / BSGS / bootstrap count).\n";
    }
    std::cout << "[align] top realizable op-type: "
              << (ranked.empty() ? "n/a" : ranked.front().first) << "\n";

    // sanity: we measured a non-trivial, clean cost model for the dominant op.
    EXPECT_GT(tab["mult_pt"].size(), 0u);
    EXPECT_GE(model["mult_pt"].best_level, 0);
}

}  // namespace
