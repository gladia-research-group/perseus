// Times the REAL (non-packed) decode ops in isolation at a CONTROLLED input CKKS
// level AND records the depth (levels) each op consumes. This is the REAL half of
// an apples-to-apples real-vs-complex comparison: test_complex_op_level_sweep.cu
// runs the SAME primitives in the complex (K/V-lane packed) regime under the same
// LEVEL_SWEEP, so the two summary grids line up op-by-op to expose (a) the latency
// advantage of complex packing and (b) any difference in each op's feasible level
// band / consumed depth.
//
// Comparable primitives (identical names/order in both tests):
//   qkv_lin    one d×d attention-shape linear  (complex: fused K+iV in ONE linear)
//   up_lin     MLP up d→d_exp linear           (complex: output-row pack, ½ pt-mults)
//   down_lin   MLP down d_exp→d linear         (NOT complex-packed; plain linear both paths)
//   qkt        Q·Kᵀ over the K cache           (complex: K cache holds 2 keys/ct)
//   softmax_v  P·V over the V cache            (complex: d_head/2 complex V buckets)
//   kv_push    cache_k_push + cache_v_push     (complex: ONE bootstrap for K+V)
// REAL-ONLY context rows (no complex analog — N/D packing is net-negative):
//   ln_1, gelu, softmax.
//
// Packing factor: the complex qkv_lin/up_lin/kv_push each do ~2× the logical work of
// the real op here (2 matrices / 2 output rows / K+V together), so the real per-op
// time must be ~doubled before comparing to its single complex counterpart.
//
// Two classes still show up among the weightless ops: SHALLOW keyswitch-heavy (qkt,
// softmax_v): few levels consumed -> movable up the band -> limb-bound cost curve.
// DEPTH-consuming nonlinearities (ln/gelu/softmax): many levels -> THROW high ->
// pinned low.
//
// Production THOR chain (default make_gpt2_inference). Configs from CONFIGS_PATH
// (gpt2_diff). Run with CKKS_COMPLEX=0 for the pure-real baseline. Each (op,level)
// is try/caught so a throw just prints THREW and the sweep continues.
//
// Knobs:  LEVEL_SWEEP (csv, default 16..24)   REPS (default 15)   KC (cached keys, default 8)

#include "model/gpt2.h"
#include "inference.h"
#include "attention.h"
#include "nonlinear.h"
#include "model/layer_norm.h"
#include "config_loader.h"
#include "weight_loader.h"
#include "packing/cachemir/cachemir_linear_utils.h"
#include "test_helpers.h"
#include "ckks_types.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace test_helpers;

// Pure (config-only) cachemir normalization dispatcher — defined in
// src/algorithms/norm/norm.cu. `layer_norm` (model/layer_norm.h) wraps this with
// the affine shift, which needs the ln_*.shift WEIGHT we don't load here, so we
// call `norm` directly like test_layernorm.cu does.
PackedCtx norm(Inference& inf, const PackedCtx& x, const std::string& cfg_name);

namespace {

std::vector<int> parse_csv_ints(const std::string& s) {
    std::vector<int> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') { if (!cur.empty()) out.push_back(std::stoi(cur)); cur.clear(); }
        else if (!std::isspace(static_cast<unsigned char>(c))) cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(std::stoi(cur));
    return out;
}

// cudaEvent timing: warm up, then median-ish mean over reps. Each composite op is
// dozens of kernels, so per-call launch overhead is naturally amortized.
template <typename F>
double time_ms(int reps, F&& fn) {
    for (int w = 0; w < 3; ++w) fn();                 // warmup
    cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    double sum = 0.0;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(s);
        fn();
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms = 0.0f; cudaEventElapsedTime(&ms, s, e);
        sum += ms;
    }
    cudaEventDestroy(s); cudaEventDestroy(e);
    return sum / reps;
}

// One sweep point: latency + the input/output CKKS level of the op (out-in = depth consumed).
struct OpResult { double ms; int in_lvl; int out_lvl; };
struct OpRow { std::string name; std::vector<double> ms; std::vector<int> depth; };

TEST(OpLevelSweep, CompositeOpsAcrossLevels) {
    // Honor CKKS_COMPLEX (else this test silently ran real-mode regardless of the env):
    // CKKS_COMPLEX=1 = complex payload, the K/V-pack regime the live decode runs in.
    Inference inf = make_gpt2_inference({
        .ckks = {.ckks_complex_payload = (env_or("CKKS_COMPLEX", "0") == "1"),
                 .bts_iterations = default_bts_iterations()},
    });
    const std::string configs_path = default_configs_path();
    std::cout << "[oplv] configs = " << configs_path << "\n";
    auto parsed = config_loader::parse_configs_json(
        config_loader::read_file_to_string(configs_path));
    weight_loader::prepare_gpt2_layer_configs(inf, parsed, /*block_idx=*/0);
    prepare_mha_masks(inf);
    prepare_vcache(inf);
    inf.output.capture_t = 0;   // LN center_scale_sq reads this position

    const int d           = inf.size.hidDim;                       // 1024
    const int d_exp       = inf.size.expDim;                        // 4096
    const int H           = inf.size.numHeads;
    const int d_head      = d / H;
    const int d_head_real = inf.size.getRealDHead();
    const int reps        = std::stoi(env_or("REPS", "15"));
    const int kc          = std::stoi(env_or("KC", "8"));
    std::vector<int> levels = parse_csv_ints(env_or("LEVEL_SWEEP", "8,12,16,17,18,19,20,21,22,23"));

    const int prod_lvl = static_cast<int>(inf.fhe->bootstrap_output_level());
    std::cout << "[oplv] d=" << d << " H=" << H << " d_head=" << d_head
              << " d_head_real=" << d_head_real << " kc=" << kc
              << " reps=" << reps << " bootstrap_output_level=" << prod_lvl
              << " level_limit=" << inf.fhe->level_limit() << "\n";
    std::cout << "[oplv] LEVEL_SWEEP =";
    for (int L : levels) std::cout << " " << L;
    std::cout << "  (L==" << prod_lvl << " is PRODUCTION post-bootstrap)\n\n";

    std::mt19937 g(7);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    auto rnd = [&](int n) { std::vector<double> v(n); for (auto& x : v) x = dist(g); return v; };
    auto rndmat = [&](int r, int c) {
        std::vector<std::vector<double>> m(r, std::vector<double>(c));
        for (auto& row : m) for (auto& x : row) x = dist(g);
        return m;
    };

    // Effective CKKS level of a PackedCtx = drops applied (GetLevel) + pending lazy
    // rescale (NoiseScaleDeg-1). FLEXIBLEAUTO defers the multPt rescale, so GetLevel
    // alone reads the same in/out for a linear (depth would falsely show 0); adding the
    // noise-scale degree recovers the true consumed depth. (test-side .ct access is OK.)
    auto lvl = [&](const PackedCtx& p) {
        return inf.fhe->level_for_ct(p.ct) + (static_cast<int>(p.ct->GetNoiseScaleDeg()) - 1);
    };

    std::vector<OpRow> rows;

    auto sweep = [&](const char* name, std::function<OpResult(int)> run_at) {
        OpRow row{name, {}, {}};
        for (int L : levels) {
            OpResult r{-1.0, -1, -1};
            try { r = run_at(L); }
            catch (const std::exception& ex) {
                std::cout << "[oplv] " << name << " L=" << L << " THREW: " << ex.what() << "\n";
                row.ms.push_back(-1.0); row.depth.push_back(-999);
                continue;
            }
            const int consumed = (r.in_lvl >= 0 && r.out_lvl >= 0) ? (r.out_lvl - r.in_lvl) : -999;
            std::cout << "[oplv] " << name << " L=" << L << " ms=" << r.ms
                      << " lvl " << r.in_lvl << "->" << r.out_lvl << " (depth=" << consumed << ")"
                      << (L == prod_lvl ? "  (PRODUCTION)" : "") << "\n";
            row.ms.push_back(r.ms); row.depth.push_back(consumed);
        }
        std::cout << "\n";
        rows.push_back(row);
    };

    // ===================== COMPARABLE PRIMITIVES (vs complex) =====================

    // ---- qkv_lin: ONE d×d attention-shape linear (complex packs K+iV into one) ----
    sweep("qkv_lin", [&](int L) -> OpResult {
        const std::string wn = "qline";
        inf.w[wn] = encode_weight_matrix(inf, rndmat(d, d), d, d, L);
        inf.w.erase(wn + "_bias");                       // pure matmul (no bias add)
        PackedCtx x = encode_linear_input(inf, rnd(d), d, d, L);
        PackedCtx y = linear(inf, x, wn, d, d);
        double ms = time_ms(reps, [&] { PackedCtx t = linear(inf, x, wn, d, d); (void)t; });
        return {ms, lvl(x), lvl(y)};
    });

    // ---- up_lin: MLP up d→d_exp linear (complex: output-row pack) ----
    sweep("up_lin", [&](int L) -> OpResult {
        const std::string wn = "up";
        inf.w[wn] = encode_weight_matrix(inf, rndmat(d, d_exp), d, d_exp, L);
        inf.w.erase(wn + "_bias");
        PackedCtx x = encode_linear_input(inf, rnd(d), d, d_exp, L);
        PackedCtx y = linear(inf, x, wn, d, d_exp);
        double ms = time_ms(reps, [&] { PackedCtx t = linear(inf, x, wn, d, d_exp); (void)t; });
        return {ms, lvl(x), lvl(y)};
    });

    // ---- down_lin: MLP down d_exp→d linear (no complex packing — plain linear both paths) ----
    sweep("down_lin", [&](int L) -> OpResult {
        const std::string wn = "down";
        inf.w[wn] = encode_weight_matrix(inf, rndmat(d_exp, d), d_exp, d, L);
        inf.w.erase(wn + "_bias");
        PackedCtx x = encode_linear_input(inf, rnd(d_exp), d_exp, d, L);
        PackedCtx y = linear(inf, x, wn, d_exp, d);
        double ms = time_ms(reps, [&] { PackedCtx t = linear(inf, x, wn, d_exp, d); (void)t; });
        return {ms, lvl(x), lvl(y)};
    });

    // ---- Q·Kᵀ (shallow keyswitch-heavy: replicate rots + per-group mult+reduce) -
    sweep("qkt", [&](int L) -> OpResult {
        inf.cache[inf.scoped("k")].clear();
        inf.cache[inf.scoped("k")].push_back(encode_linear_input(inf, rnd(d), d, d, L));  // 1 group (kc<=t)
        inf.k_count() = kc;
        PackedCtx query = encode_linear_input(inf, rnd(d), d, d, L);
        auto s0 = qkt(inf, query);
        double ms = time_ms(reps, [&] { auto s = qkt(inf, query); (void)s; });
        return {ms, lvl(query), s0.empty() ? -1 : lvl(s0[0])};
    });

    // ---- softmax·V / P·V (shallow keyswitch-heavy: lane mults + hoisted rots) -
    sweep("softmax_v", [&](int L) -> OpResult {
        prepare_vcache(inf);
        for (int i = 0; i < d_head; ++i)
            inf.cache[inf.scoped("v")][i] = encode_linear_input(inf, rnd(d), d, d, L);
        inf.k_count() = kc;
        PackedCtx scores = encode_linear_input(inf, rnd(d), d, d, L);
        PackedCtx o0 = softmax_v(inf, {scores});
        double ms = time_ms(reps, [&] { PackedCtx o = softmax_v(inf, {scores}); (void)o; });
        return {ms, lvl(scores), lvl(o0)};
    });

    // ---- kv_push: REAL = cache_k_push + cache_v_push (TWO bootstraps for K and V) -
    sweep("kv_push", [&](int L) -> OpResult {
        PackedCtx key = encode_linear_input(inf, rnd(d), d, d, L);
        PackedCtx val = encode_linear_input(inf, rnd(d), d, d, L);
        prepare_mha_masks(inf); prepare_vcache(inf);
        cache_k_push(inf, key);
        cache_v_push(inf, val);
        // K lands in "k" (real flat cache) or "k.pend" (complex, pre-pack) — probe both.
        auto& kflat = inf.cache[inf.scoped("k")];
        auto& kpend = inf.cache[inf.scoped("k.pend")];
        const int out_lvl = !kflat.empty() ? lvl(kflat[0]) : (!kpend.empty() ? lvl(kpend[0]) : -1);
        double ms = time_ms(reps, [&] {
            prepare_mha_masks(inf); prepare_vcache(inf);
            cache_k_push(inf, key);
            cache_v_push(inf, val);
        });
        return {ms, lvl(key), out_lvl};   // bootstraps -> out is the cache-read level (refresh)
    });

    // ================= REAL-ONLY context (no complex analog) ==================

    // ---- LayerNorm inv_sqrt (depth-consuming: Goldschmidt over hidden dim) -----
    sweep("ln_1", [&](int L) -> OpResult {
        PackedCtx x = encode_linear_input(inf, rnd(d), d, d, L);
        PackedCtx y = norm(inf, x, "ln_1");
        double ms = time_ms(reps, [&] { PackedCtx t = norm(inf, x, "ln_1"); (void)t; });
        return {ms, lvl(x), lvl(y)};
    });

    // ---- GELU (depth-consuming: softsign + inv_sqrt Goldschmidt) ---------------
    sweep("gelu", [&](int L) -> OpResult {
        PackedCtx x = encode_linear_input(inf, rnd(d), d, d, L);
        PackedCtx y = gelu_approx(inf, x, "mlp.act");
        double ms = time_ms(reps, [&] { PackedCtx t = gelu_approx(inf, x, "mlp.act"); (void)t; });
        return {ms, lvl(x), lvl(y)};
    });

    // ---- THOR softmax (depth-consuming: exp squares + GS reciprocal + refine) ---
    sweep("softmax", [&](int L) -> OpResult {
        inf.k_count() = kc;
        PackedCtx scores = encode_linear_input(inf, rnd(d), d, d, L);
        auto y0 = attention_softmax_thor(inf, {scores}, "attn");
        double ms = time_ms(reps, [&] { auto y = attention_softmax_thor(inf, {scores}, "attn"); (void)y; });
        return {ms, lvl(scores), y0.empty() ? -1 : lvl(y0[0])};
    });

    // ---- SUMMARY grids: op vs level (ms; "-" = threw / infeasible at that level) -
    auto print_grid = [&](const char* title, bool depth_grid) {
        std::cout << "\n" << title << "\n";
        std::cout << std::left << std::setw(12) << "op";
        for (int L : levels) std::cout << std::right << std::setw(9) << L;
        std::cout << "\n";
        for (const auto& row : rows) {
            std::cout << std::left << std::setw(12) << row.name;
            for (size_t i = 0; i < row.ms.size(); ++i) {
                if (row.ms[i] < 0) { std::cout << std::right << std::setw(9) << "-"; continue; }
                if (depth_grid) std::cout << std::right << std::setw(9) << row.depth[i];
                else            std::cout << std::right << std::setw(9) << std::fixed << std::setprecision(1) << row.ms[i];
            }
            std::cout << "\n";
        }
    };
    print_grid("============== OP LEVEL SWEEP SUMMARY (mean ms) ==============", false);
    std::cout << "(depth-consuming ops throw at high L = pinned low; shallow ops show the limb-bound curve)\n";
    print_grid("=========== OP LEVEL SWEEP SUMMARY (consumed depth, levels) ===========", true);
    std::cout << "(consumed depth = output_level - input_level; <0 = the op bootstraps/refreshes)\n";

    SUCCEED();
}

}  // namespace
