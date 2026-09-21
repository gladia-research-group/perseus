// Times the COMPLEX (K/V-lane packed) decode ops in isolation at a CONTROLLED input
// CKKS level AND records the depth (levels) each op consumes — the complex half of an
// apples-to-apples comparison with test_op_level_sweep.cu, which runs the SAME named
// primitives in the pure-real regime under the same LEVEL_SWEEP. Diff the two summary
// grids op-by-op to expose (a) the latency advantage of complex packing and (b) the
// difference in each op's feasible level band / consumed depth. This is also the data
// behind CACHE_READ_LEVEL + bts placement for the GPT2_PACKING=cachemir_complex path.
//
// Runs the production complex regime: packing_kind = CachemirComplex → inf.complex = true,
// so the un-prefixed dispatchers (linear/qkt/softmax_v/cache_kv_push_packed) route to the
// complex impls, and prepare_mha_masks/prepare_vcache build the complex caches.
//
// Comparable primitives (identical names/order to the real test):
//   qkv_lin    complex QKV linear (W_re=K + i·W_im=V → one packed P)            [keyswitch-heavy]
//   up_lin     complex MLP up, output-row pack (½ the pt-mults)                 [keyswitch-heavy]
//   down_lin   MLP down d_exp→d (NOT complex-packed in prod; plain linear, for parity)[keyswitch-heavy]
//   qkt        complex_qkt: q·K over the complex K cache + conj add/sub          [shallow]
//   softmax_v  complex_softmax_v: P·V over the d_head/2 complex V buckets        [shallow]
//   kv_push    cache_kv_push_packed_complex: ONE bootstrap, unpack, mask, realify[bootstrap+keyswitch]
// Each complex op does ~2× the logical work of its real counterpart (2 matrices / 2
// output rows / K+V together), so compare its time to ~2× the real per-op time.
//
// Configs from CONFIGS_PATH (gpt2_diff). Weightless ops read random caches; the linears
// encode random complex weights inline at each level. Each (op,level) is try/caught.
//
// Knobs: LEVEL_SWEEP (csv, default 16..24)   REPS (15)   KC (cached keys, 8)

#include "model/gpt2.h"
#include "inference.h"
#include "attention.h"
#include "config_loader.h"
#include "weight_loader.h"
#include "packing/cachemir/cachemir_attention.h"
#include "packing/cachemir/cachemir_linear_utils.h"
#include "test_helpers.h"
#include "ckks_types.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace test_helpers;

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

TEST(ComplexOpLevelSweep, ComplexOpsAcrossLevels) {
    // Full complex path: CachemirComplex packing → inf.complex = true (implies complex payload).
    Inference inf = make_gpt2_inference({
        .ckks = {.bts_iterations = default_bts_iterations()},
        .packing_kind = PackingKind::CachemirComplex,
    });
    ASSERT_TRUE(inf.complex) << "CachemirComplex packing must set inf.complex";

    const std::string configs_path = default_configs_path();
    std::cout << "[cxlv] configs = " << configs_path << "\n";
    auto parsed = config_loader::parse_configs_json(
        config_loader::read_file_to_string(configs_path));
    weight_loader::prepare_gpt2_layer_configs(inf, parsed, /*block_idx=*/0);
    prepare_mha_masks(inf);   // complex K cache (k + k.pend)
    prepare_vcache(inf);      // complex V cache (d_head/2 buckets)
    inf.output.capture_t = 0; // LN center_scale_sq reads this position

    const int d           = inf.size.hidDim;        // 1024
    const int d_exp       = inf.size.expDim;         // 4096
    const int H           = inf.size.numHeads;
    const int d_head      = d / H;
    const int d_head_real = inf.size.getRealDHead();
    const int reps        = std::stoi(env_or("REPS", "15"));
    const int kc          = std::stoi(env_or("KC", "8"));
    std::vector<int> levels = parse_csv_ints(env_or("LEVEL_SWEEP", "16,17,18,19,20,21,22,23,24"));

    const int prod_lvl = static_cast<int>(inf.fhe->bootstrap_output_level());
    std::cout << "[cxlv] d=" << d << " d_exp=" << d_exp << " H=" << H << " d_head=" << d_head
              << " d_head_real=" << d_head_real << " kc=" << kc << " reps=" << reps
              << " bootstrap_output_level=" << prod_lvl
              << " level_limit=" << inf.fhe->level_limit() << "\n";
    std::cout << "[cxlv] LEVEL_SWEEP =";
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
                std::cout << "[cxlv] " << name << " L=" << L << " THREW: " << ex.what() << "\n";
                row.ms.push_back(-1.0); row.depth.push_back(-999);
                continue;
            }
            const int consumed = (r.in_lvl >= 0 && r.out_lvl >= 0) ? (r.out_lvl - r.in_lvl) : -999;
            std::cout << "[cxlv] " << name << " L=" << L << " ms=" << r.ms
                      << " lvl " << r.in_lvl << "->" << r.out_lvl << " (depth=" << consumed << ")"
                      << (L == prod_lvl ? "  (PRODUCTION)" : "") << "\n";
            row.ms.push_back(r.ms); row.depth.push_back(consumed);
        }
        std::cout << "\n";
        rows.push_back(row);
    };

    // ---- qkv_lin: fused KV complex linear (W_re=K + i·W_im=V → one packed P) ----
    sweep("qkv_lin", [&](int L) -> OpResult {
        inf.w["kv"] = encode_weight_matrix_complex(inf, rndmat(d, d), rndmat(d, d), d, d, L);
        inf.complex_weight_names.insert("kv");
        inf.w.erase("kv_bias");
        PackedCtx x = encode_linear_input(inf, rnd(d), d, d, L);
        PackedCtx y = linear(inf, x, "kv", d, d);
        double ms = time_ms(reps, [&] { PackedCtx t = linear(inf, x, "kv", d, d); (void)t; });
        return {ms, lvl(x), lvl(y)};
    });

    // ---- up_lin: complex MLP up, output-row pack ----
    sweep("up_lin", [&](int L) -> OpResult {
        inf.w["up"] = encode_weight_matrix_outputpack(inf, rndmat(d, d_exp), d, d_exp, L);
        inf.complex_weight_names.insert("up");
        inf.w.erase("up_bias");
        PackedCtx x = encode_linear_input(inf, rnd(d), d, d_exp, L);
        PackedCtx y = linear_outputpack(inf, x, "up", d, d_exp);
        double ms = time_ms(reps, [&] { PackedCtx t = linear_outputpack(inf, x, "up", d, d_exp); (void)t; });
        return {ms, lvl(x), lvl(y)};
    });

    // ---- down_lin: MLP down d_exp→d (NOT complex-packed in production — plain linear,
    //       included for parity with the real sweep / to confirm no complex benefit) ----
    sweep("down_lin", [&](int L) -> OpResult {
        inf.w["down"] = encode_weight_matrix(inf, rndmat(d_exp, d), d_exp, d, L);
        inf.w.erase("down_bias");
        PackedCtx x = encode_linear_input(inf, rnd(d_exp), d_exp, d, L);
        PackedCtx y = linear(inf, x, "down", d_exp, d);
        double ms = time_ms(reps, [&] { PackedCtx t = linear(inf, x, "down", d_exp, d); (void)t; });
        return {ms, lvl(x), lvl(y)};
    });

    // ---- complex_qkt: q·K over the complex K cache + conj add/sub ----
    sweep("qkt", [&](int L) -> OpResult {
        inf.cache[inf.scoped("k")].clear();
        inf.cache[inf.scoped("k.pend")].clear();
        inf.cache[inf.scoped("k")].push_back(encode_linear_input(inf, rnd(d), d, d, L));  // 1 complex pair
        inf.k_count() = kc;
        PackedCtx query = encode_linear_input(inf, rnd(d), d, d, L);
        auto s0 = qkt(inf, query);
        double ms = time_ms(reps, [&] { auto s = qkt(inf, query); (void)s; });
        return {ms, lvl(query), s0.empty() ? -1 : lvl(s0[0])};
    });

    // ---- complex_softmax_v: P·V over the d_head/2 complex V buckets ----
    sweep("softmax_v", [&](int L) -> OpResult {
        prepare_vcache(inf);
        for (int i = 0; i < d_head / 2; ++i)
            inf.cache[inf.scoped("v")][i] = encode_linear_input(inf, rnd(d), d, d, L);
        inf.k_count() = kc;
        PackedCtx scores = encode_linear_input(inf, rnd(d), d, d, L);
        PackedCtx o0 = softmax_v(inf, {scores});
        double ms = time_ms(reps, [&] { PackedCtx o = softmax_v(inf, {scores}); (void)o; });
        return {ms, lvl(scores), lvl(o0)};
    });

    // ---- kv_push: COMPLEX = cache_kv_push_packed (ONE bootstrap for K+V) ----
    sweep("kv_push", [&](int L) -> OpResult {
        PackedCtx P = encode_linear_input(inf, rnd(d), d, d, L);
        prepare_mha_masks(inf); prepare_vcache(inf);
        cache_kv_push_packed(inf, P);   // dispatcher → cache_kv_push_packed_complex
        // K lands in "k" (real flat cache) or "k.pend" (complex, pre-pack) — probe both.
        auto& kflat = inf.cache[inf.scoped("k")];
        auto& kpend = inf.cache[inf.scoped("k.pend")];
        const int out_lvl = !kflat.empty() ? lvl(kflat[0]) : (!kpend.empty() ? lvl(kpend[0]) : -1);
        double ms = time_ms(reps, [&] {
            prepare_mha_masks(inf);
            prepare_vcache(inf);
            cache_kv_push_packed(inf, P);
        });
        return {ms, lvl(P), out_lvl};   // bootstraps -> out is the cache-read level (refresh)
    });

    // ---- SUMMARY grids: op vs level (ms; "-" = threw / infeasible at that level) ----
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
    print_grid("========== COMPLEX OP LEVEL SWEEP SUMMARY (mean ms) ==========", false);
    std::cout << "(- = threw / infeasible at that level → that op's feasibility band excludes it)\n";
    print_grid("======= COMPLEX OP LEVEL SWEEP SUMMARY (consumed depth, levels) =======", true);
    std::cout << "(consumed depth = output_level - input_level; <0 = the op bootstraps/refreshes)\n";

    SUCCEED();
}

}  // namespace
