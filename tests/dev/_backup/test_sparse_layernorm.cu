// RETIRED 2026-07-21: subject (the sparse_norm COPY of norm.cu) was unified into
// ln_inv_sqrt_tail (norm.cu) -- sparse==dense is now true BY CONSTRUCTION (one body,
// routing flag). Restore + port to the core entry if the scope routing ever needs a gate.
#include "all_blocks_test_helpers.h"
#include "ckks_primitives.h"
#include "fideslib_wrapper.h"
#include "layernorm_test_helpers.h"
#include "math/matrix_ops.h"
#include "model/gpt2.h"
#include "nonlinear.h"
#include "test_helpers.h"
#include "weight_loader.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace test_helpers;

// Isolation proof + timing for sparse-bts LayerNorm (sparse_norm.cu). The LN
// inv_sqrt chain is a pure function of the FULL-BROADCAST variance (constant in
// every slot after rotate-and-sum-all), so its bootstraps route to the sparse
// precomp; only the final mult(centered_x, .) is a vector op (full-slot).
//   - EqualsDenseUnderBootstrap: input dropped to a decode level so bootstraps
//     fire in-chain -> the PRIMARY proof is sparse_norm == norm head-to-head
//     (same result, same level, same bts count). vs-torch is informational:
//     eager placement degrades BOTH paths equally, so it is not the bar here.
//   - CorrectAndFaithfulFresh: fresh input (no bts) -> both match torch AND
//     sparse_norm reproduces norm bit-for-bit (the copy is a faithful refactor).
//   - Timing: wall-clock ms/LN, dense vs sparse, at the current s.
// One CKKS context is shared across the suite (building two OOMs a 64 GB GPU).
// Knobs: SPARSE_BTS_SLOTS (default 512), LN_TEST_DROP_LEVEL (16),
//        CONFIGS_PATH / WEIGHTS_PATH, T_SWEEP_VAL, CKKS_COMPLEX (complex payload).

namespace {

constexpr int LN_C_REAL      = 768;
constexpr int LN_HID_PAD     = 1024;
constexpr int N_TOKENS_PER_T = 16;

uint32_t sparse_slots_env() {
    const char* v = std::getenv("SPARSE_BTS_SLOTS");
    return (v && *v) ? static_cast<uint32_t>(std::atoi(v)) : 512u;
}
int drop_level_env() {
    const char* v = std::getenv("LN_TEST_DROP_LEVEL");
    return (v && *v) ? std::atoi(v) : 16;
}
// decode (Cachemir, var=global const) or prefill (CachemirFilling, var=period-t_stride)
PackingKind packing_env() {
    const char* v = std::getenv("GPT2_PACKING");
    if (v && std::string(v) == "cachemir_filling") return PackingKind::CachemirFilling;
    return PackingKind::Cachemir;
}

Inference make_ln_inference(uint32_t sparse_slots) {
    InferenceOptions opts;
    opts.ckks.bts_iterations   = default_bts_iterations();
    opts.ckks.sparse_bts_slots = sparse_slots;
    if (const char* cx = std::getenv("CKKS_COMPLEX"); cx && cx[0] == '1')
        opts.ckks.ckks_complex_payload = true;   // decode/token-pair operating point
    opts.hidDim       = LN_HID_PAD;
    opts.bench_mode   = false;
    opts.packing_kind = packing_env();
    return make_gpt2_inference(opts);
}

// One context for the whole process (a second full context OOMs the GPU).
Inference& g_inf() {
    static Inference inf = make_ln_inference(sparse_slots_env());
    return inf;
}

struct LnFix {
    std::vector<double> g, b;   // ln_1 gamma/beta
    IoArrays io;                // block-0 inp -> ln_1_out
    int T = 0;
    bool ok = false;
};

const LnFix& g_fix() {
    static LnFix f = [] {
        LnFix r;
        const std::string wp = default_weights_path();
        { std::ifstream probe(wp); if (!probe) return r; }
        auto store = load_store(wp);
        r.g = store.tensor1d("transformer.h.0.ln_1.weight", LN_C_REAL);
        r.b = store.tensor1d("transformer.h.0.ln_1.bias",   LN_C_REAL);
        auto parsed = config_loader::parse_configs_json(
            config_loader::read_file_to_string(default_configs_path()));
        weight_loader::prepare_gpt2_layer_configs(g_inf(), parsed, /*block_idx=*/0);
        r.T = default_t_sweep_val();
        const std::string io_path = all_blocks_io_path(default_all_blocks_io_dir(), 0, r.T);
        if (!probe_io_file(io_path)) return r;
        r.io = read_block0_io(io_path, "inp", "ln_1_out");
        r.ok = !r.io.inp.empty();
        return r;
    }();
    return f;
}

std::vector<double> decode_normed(Inference& inf, const PackedCtx& y) {
    auto raw = decrypt_slots(inf, y);
    auto normed = decode_linear_output(inf.packing, raw, inf.slots, LN_HID_PAD, LN_HID_PAD);
    normed.resize(LN_C_REAL);
    return normed;
}

std::vector<double> affine(const std::vector<double>& normed,
                           const std::vector<double>& g, const std::vector<double>& b) {
    std::vector<double> got(LN_C_REAL);
    for (int k = 0; k < LN_C_REAL; ++k) got[k] = normed[k] * g[k] + b[k];
    return got;
}

struct DualRun {
    std::vector<double> nd, ns;   // decoded raw-LN (dense, sparse)
    int lvl_d = -1, lvl_s = -1;
    uint64_t bts_d = 0, bts_s = 0;
};

// encode one input, drop to `drop` level, run BOTH norm() and sparse_norm() on
// it (neither mutates x, so the only difference is the bootstrap routing).
DualRun run_both(Inference& inf, const std::vector<double>& x_pad, int drop) {
    PackedCtx x = encode_linear_input(inf, x_pad, LN_HID_PAD, LN_HID_PAD);
    if (drop > 0) inf.fhe->drop_to_level(x, drop);
    DualRun r;
    uint64_t b = inf.fhe->total_bootstraps;
    PackedCtx yd = norm(inf, x, "ln_1");
    r.bts_d = inf.fhe->total_bootstraps - b; r.lvl_d = static_cast<int>(level_of(yd.ct));
    b = inf.fhe->total_bootstraps;
    PackedCtx ys = sparse_norm(inf, x, "ln_1");
    r.bts_s = inf.fhe->total_bootstraps - b; r.lvl_s = static_cast<int>(level_of(ys.ct));
    r.nd = decode_normed(inf, yd);
    r.ns = decode_normed(inf, ys);
    return r;
}

}  // namespace

// PRIMARY PROOF: a bootstrap fires in the inv_sqrt chain and is routed sparse;
// sparse_norm must equal the dense norm() head-to-head, at the same level and
// bootstrap count. (vs-torch is only reported: eager placement degrades both.)
TEST(SparseLayerNormTest, EqualsDenseUnderBootstrap) {
    if (!g_fix().ok) GTEST_SKIP() << "weights/io missing";
    Inference& inf = g_inf();
    const auto& fx = g_fix();
    const int drop = drop_level_env();
    std::cout << "[test_sparse_ln] packing=" << to_string(inf.packing.kind)
              << " SPARSE_BTS_SLOTS=" << sparse_slots_env()
              << " drop_level=" << drop << " sparse_precomp=" << inf.fhe->sparse_bts_slots << "\n";
    ASSERT_EQ(inf.fhe->sparse_bts_slots, sparse_slots_env())
        << "sparse precomp not built (s must be pow2 < slots=" << inf.slots << ")";

    constexpr double fail_thresh = 0.01;
    // sparse vs dense: at s=0 the scope is inert -> same ops, differ only by
    // GPU-kernel nondeterminism across two independent bootstrap invocations
    // (~1.4e-5, « bts floor 1e-3 « a real bug 1e-2); at s>0 they differ by the
    // sparse-vs-full bootstrap noise (~1e-4).
    const double head2head_thr = (sparse_slots_env() == 0) ? 1e-4 : 5e-3;
    const int step = std::max(1, fx.T / N_TOKENS_PER_T);
    int n_bts_exercised = 0;

    print_token_header();
    SweepSummary sum_sparse;
    for (int tok = 0; tok < fx.T; tok += step) {
        DualRun r = run_both(inf, matrix::pad_vector(fx.io.inp[tok], LN_HID_PAD), drop);
        auto got_dense  = affine(r.nd, fx.g, fx.b);
        auto got_sparse = affine(r.ns, fx.g, fx.b);
        AccStats st_torch_d = compare_vec(got_dense,  fx.io.res[tok]);
        AccStats st_torch_s = compare_vec(got_sparse, fx.io.res[tok]);
        AccStats st_sd      = compare_vec(got_sparse, got_dense);

        print_token_row(tok, st_torch_s);
        std::cout << "        [head2head] sparse-vs-dense max_abs=" << st_sd.max_abs
                  << " | bts d/s=" << r.bts_d << "/" << r.bts_s
                  << " | lvl d/s=" << r.lvl_d << "/" << r.lvl_s
                  << " | (info) vs-torch d/s mean_rel=" << st_torch_d.mean_rel
                  << "/" << st_torch_s.mean_rel << "\n";
        sum_sparse.add(st_torch_s, fail_thresh);
        if (r.bts_s > 0) ++n_bts_exercised;

        EXPECT_LT(st_sd.max_abs, head2head_thr) << "tok=" << tok << " sparse_norm != norm under bts";
        EXPECT_EQ(r.lvl_s, r.lvl_d)             << "tok=" << tok << " output level differs (plan break)";
        EXPECT_EQ(r.bts_s, r.bts_d)             << "tok=" << tok << " bootstrap count differs";
    }
    print_sweep_summary(fx.T, sum_sparse, fail_thresh);
    EXPECT_GT(n_bts_exercised, 0)
        << "no bootstrap fired — raise LN_TEST_DROP_LEVEL; sparse routing not exercised";
}

// NOTE: there is no "no-bootstrap" regime to test faithfulness against — the
// gatem15 LN inv_sqrt chain (12 Goldschmidt iters) is deep enough to bootstrap
// even from a fresh ct, and an EAGER bootstrap lands mid-chain at a bad
// magnitude, so even dense gatem15 LN misses torch in isolation (that is why
// decode PLANS the placements; vs-torch is the planned-decode gate, not this
// one). Faithfulness of the copy is instead proven by the s=0 arm of
// EqualsDenseUnderBootstrap (scope inert -> sparse_norm bit-identical to norm).

// Wall-clock: dense norm() vs sparse_norm() at the current s. Input dropped to a
// decode level so the LN bootstraps fire; the ms difference is purely the
// sparse-vs-full slot cost of those bootstraps (everything else is identical).
TEST(SparseLayerNormTest, Timing) {
    if (!g_fix().ok) GTEST_SKIP() << "weights/io missing";
    Inference& inf = g_inf();
    const auto& fx = g_fix();
    const int drop = drop_level_env();

    auto x_pad = matrix::pad_vector(fx.io.inp[0], LN_HID_PAD);
    PackedCtx x = encode_linear_input(inf, x_pad, LN_HID_PAD, LN_HID_PAD);
    if (drop > 0) inf.fhe->drop_to_level(x, drop);

    const uint64_t b0 = inf.fhe->total_bootstraps;
    { PackedCtx warm = sparse_norm(inf, x, "ln_1"); (void)warm; }
    const uint64_t bts_per_ln = inf.fhe->total_bootstraps - b0;

    auto bench = [&](bool sparse, int reps) {
        { PackedCtx w = sparse ? sparse_norm(inf, x, "ln_1") : norm(inf, x, "ln_1"); (void)w; }
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) {
            PackedCtx w = sparse ? sparse_norm(inf, x, "ln_1") : norm(inf, x, "ln_1");
            (void)w;
        }
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    };

    const int reps = 8;
    const double ms_dense  = bench(false, reps);
    const double ms_sparse = bench(true,  reps);
    const double d = ms_dense - ms_sparse;
    std::printf("[ln_timing] packing=%s s=%u  bts/ln=%lu | dense=%.1f ms  sparse=%.1f ms | "
                "delta=%.1f ms (%.1f%%)  per_bts_saving=%.2f ms\n",
                to_string(inf.packing.kind),
                sparse_slots_env(), static_cast<unsigned long>(bts_per_ln),
                ms_dense, ms_sparse, d, ms_dense > 0 ? 100.0 * d / ms_dense : 0.0,
                bts_per_ln ? d / bts_per_ln : 0.0);
    std::fflush(stdout);
    SUCCEED();
}
