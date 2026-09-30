// Head-to-head comparison of two head_reduce_sum formulations.
//
// cachemir::head_reduce_sum (src/algorithms/attention/cachemir/cachemir_attention.cu)
// implements a *segmented broadcast all-reduce*: every slot ends holding the sum
// of its length-t segment (the softmax denominator, broadcast across token slots
// so the elementwise z/s in goldschmidt_inv is correct everywhere), followed by a
// directed lane reduction over stride tH.
//
// The MASKED CYCLIC form of that first phase does, per doubling step, 2 rotations
// (step, step-t) + 2 plaintext-mults (complementary wrap masks). Over log2(t) steps
// that is 2*log2(t) plaintext-mults and burns log2(t) multiplicative levels.
//
// The cheaper form is REDUCE-then-BROADCAST (both halves already exist elsewhere in
// cachemir_attention.cu: softmax_v's tok_reduce and qkt's query replicate):
//   (A) mask-free directed reduce  -> sum into position 0 of each length-t segment
//   (B) one mask                   -> keep position 0, zero the discarded partials
//   (C) mask-free broadcast        -> copy position 0 across the segment
// Same rotation count, but ONE plaintext-mult (1 level) instead of 2*log2(t).
//
// Self-contained: it builds its own CKKS context (the operational THOR chain via
// default_ckks_options) with exactly the rotation keys both forms touch, defines both
// locally, and checks:
//   1. correctness  — both decrypt to a plaintext oracle, and agree with each other
//   2. op counts    — equal rotations/adds; masked does 2*log2(t) pt-mults, the other 1
//   3. levels       — reduce-then-broadcast consumes strictly fewer multiplicative levels
//   4. latency      — and is faster wall-clock (the 9 saved rescales)

#include "fideslib_wrapper.h"
#include "inference.h"
#include "packing/cachemir/cachemir_attention_utils.h"  // cachemir::mha_rot
#include "test_helpers.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

using namespace test_helpers;

namespace {

// Counters tallied at each FHE call site (no effect on the FHE computation).
struct OpCounts {
    int rotations = 0;   // EvalRotate (key-switch)
    int pt_mults  = 0;   // ciphertext x plaintext (the differentiator)
    int adds      = 0;   // ciphertext + ciphertext
    int clones    = 0;
};

// ── Variant A — VERBATIM copy of cachemir::head_reduce_sum (masked cyclic) ──────
PackedCtx head_reduce_sum_masked(Inference& inf, const PackedCtx& x, OpCounts& c) {
    int N  = inf.slots;
    int d  = inf.size.hidDim;
    int H  = inf.size.numHeads;
    int t  = N / d;
    int tH = t * H;

    PackedCtx out = inf.fhe->clone(x); ++c.clones;

    for (int step = 1; step < t; step *= 2) {
        Ptx pt_nw = inf.encode_at_cached("hrs.nw.s" + std::to_string(step), out,
            [&] {
                std::vector<double> mask_nw(N);
                for (int i = 0; i < N; ++i) mask_nw[i] = ((i % t) + step < t) ? 1.0 : 0.0;
                return mask_nw;
            });
        Ptx pt_w = inf.encode_at_cached("hrs.w.s" + std::to_string(step), out,
            [&] {
                std::vector<double> mask_w(N);
                for (int i = 0; i < N; ++i) mask_w[i] = ((i % t) + step < t) ? 0.0 : 1.0;
                return mask_w;
            });

        PackedCtx rot_fwd  = inf.fhe->rotate(out, cachemir::mha_rot(inf, step));      ++c.rotations;
        PackedCtx rot_wrap = inf.fhe->rotate(out, cachemir::mha_rot(inf, step - t));  ++c.rotations;
        PackedCtx mult1 = inf.fhe->mult(rot_fwd, pt_nw);  ++c.pt_mults;
        PackedCtx mult2 = inf.fhe->mult(rot_wrap, pt_w);  ++c.pt_mults;
        PackedCtx shifted = inf.fhe->add(mult1, mult2);   ++c.adds;
        inf.fhe->inplace_add(out, shifted);               ++c.adds;
    }

    for (int s = tH; s < N; s *= 2) {
        PackedCtx rot = inf.fhe->rotate(out, cachemir::mha_rot(inf, s)); ++c.rotations;
        inf.fhe->inplace_add(out, rot);                                 ++c.adds;
    }

    return out;
}

// ── reduce-then-broadcast: mask-free reduce + 1 mask ───────────────────────────
PackedCtx head_reduce_sum_reduce_bcast(Inference& inf, const PackedCtx& x, OpCounts& c) {
    int N  = inf.slots;
    int d  = inf.size.hidDim;
    int H  = inf.size.numHeads;
    int t  = N / d;
    int tH = t * H;

    PackedCtx out = inf.fhe->clone(x); ++c.clones;

    // (A) mask-free directed reduce: position 0 of each length-t segment ends
    //     holding the exact segment sum (Hillis-Steele; position 0 only ever
    //     reads within its own segment, so no cross-segment contamination).
    for (int step = 1; step < t; step *= 2) {
        PackedCtx rot = inf.fhe->rotate(out, cachemir::mha_rot(inf, step)); ++c.rotations;
        inf.fhe->inplace_add(out, rot);                                     ++c.adds;
    }

    // (B) one mask: keep position 0 of each segment, zero the discarded partials.
    Ptx pt0 = inf.encode_at_cached("hrs.pos0", out,
        [&] {
            std::vector<double> m(N, 0.0);
            for (int i = 0; i < N; i += t) m[i] = 1.0;
            return m;
        });
    inf.fhe->inplace_mult(out, pt0); ++c.pt_mults;

    // (C) mask-free broadcast: copy position 0 across the segment. For step<=t/2
    //     the boundary slot pulls a still-zero source, so no cross-segment leak.
    for (int step = 1; step < t; step *= 2) {
        PackedCtx rot = inf.fhe->rotate(out, cachemir::mha_rot(inf, -step)); ++c.rotations;
        inf.fhe->inplace_add(out, rot);                                      ++c.adds;
    }

    // Lane reduction over stride tH — identical to the masked variant.
    for (int s = tH; s < N; s *= 2) {
        PackedCtx rot = inf.fhe->rotate(out, cachemir::mha_rot(inf, s)); ++c.rotations;
        inf.fhe->inplace_add(out, rot);                                 ++c.adds;
    }

    return out;
}

// Plaintext oracle for the full head_reduce_sum map (both variants must match it).
//   O1[i] = sum over the length-t segment containing i          (broadcast)
//   O[i]  = sum_{m=0..R-1} O1[(i + m*tH) mod N], R = N/tH        (directed lanes)
std::vector<double> oracle_head_reduce_sum(const std::vector<double>& x,
                                           int N, int t, int tH) {
    std::vector<double> o1(N, 0.0);
    for (int g = 0; g * t < N; ++g) {
        double s = 0.0;
        for (int j = 0; j < t; ++j) s += x[g * t + j];
        for (int j = 0; j < t; ++j) o1[g * t + j] = s;
    }
    const int R = N / tH;
    std::vector<double> o(N, 0.0);
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int m = 0; m < R; ++m) s += o1[(i + m * tH) % N];
        o[i] = s;
    }
    return o;
}

using ReduceFn = PackedCtx (*)(Inference&, const PackedCtx&, OpCounts&);

// Mean / min wall-clock (ms) over `iters`, GPU-synced. Warmup is done by the
// caller so the encode cache is populated for both variants beforehand.
std::pair<double, double> time_variant(Inference& inf, const PackedCtx& x,
                                       ReduceFn fn, int iters) {
    cudaDeviceSynchronize();
    double total = 0.0, best = 1e300;
    volatile uint32_t sink = 0;
    for (int i = 0; i < iters; ++i) {
        OpCounts scratch;
        auto t0 = std::chrono::high_resolution_clock::now();
        PackedCtx o = fn(inf, x, scratch);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        sink ^= level_of(o.ct);   // keep the result live
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total += ms;
        best = std::min(best, ms);
    }
    (void)sink;
    return {total / iters, best};
}

}  // namespace

TEST(HeadReduceSumOpt, MaskedVsReduceBroadcast) {
    // ── operational context (production THOR chain) + exactly the rot keys used ──
    CKKSContextOptions opts = default_ckks_options();
    const int logN  = opts.logN;
    const int slots = (opts.batch_size == 0) ? (1 << (logN - 1))
                                             : static_cast<int>(opts.batch_size);
    const int d  = 1024;   // padded hidDim (GPT-2)
    const int H  = 16;     // padded numHeads
    const int t  = slots / d;
    const int tH = t * H;
    ASSERT_GT(t, 1) << "need t>1 for a meaningful segmented reduce";

    // Default keys are powers-of-two only; the masked variant's wrap keys
    // {step - t} are NOT powers of two, so add every index both variants touch.
    std::set<int32_t> rk;
    int log2t = 0;
    for (int step = 1; step < t; step *= 2) {
        rk.insert(step);        // reduce / masked-forward
        rk.insert(step - t);    // masked wrap-around
        rk.insert(-step);       // broadcast
        ++log2t;
    }
    int lane_steps = 0;
    for (int s = tH; s < slots; s *= 2) { rk.insert(s); ++lane_steps; }
    opts.extra_rot_steps.assign(rk.begin(), rk.end());

    std::cout << "[setup] building CKKS context (logN=" << logN << ", bootstrap chain)..."
              << std::endl;
    auto ctx = make_ckks_context(opts);

    Inference inf;
    inf.fhe              = ctx;
    inf.slots            = slots;
    inf.size.dim         = 768;
    inf.size.hidDim      = d;
    inf.size.numHeads    = H;
    inf.size.numHeadsReal = 12;
    inf.bench_mode       = false;
    inf.packing          = inf.make_packing(PackingKind::Cachemir);

    std::cout << "[setup] slots=" << slots << " hidDim=" << d << " numHeads=" << H
              << " -> t=" << t << " (log2 t=" << log2t << "), tH=" << tH
              << ", lane steps=" << lane_steps << std::endl;

    // ── random input over every slot (exercises the reduce everywhere) ──
    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<double> xv(slots);
    for (auto& v : xv) v = U(rng);

    Ptx xpt = encode(inf.cc(), xv, /*level=*/0);
    Ctx xct = encrypt(inf.cc(), xpt, inf.fhe->pk());
    PackedCtx x = inf.pack(xct, PackingKind::Cachemir);
    const uint32_t lvl_in = level_of(x.ct);

    // Warmup once each — populates the encode cache (cached for both) + warms GPU.
    { OpCounts w; head_reduce_sum_masked(inf, x, w); head_reduce_sum_reduce_bcast(inf, x, w); }
    cudaDeviceSynchronize();

    // ── run both, capture op counts + output level ──
    OpCounts cm, cb;
    PackedCtx out_m = head_reduce_sum_masked(inf, x, cm);
    PackedCtx out_b = head_reduce_sum_reduce_bcast(inf, x, cb);
    cudaDeviceSynchronize();

    const uint32_t lvl_m = level_of(out_m.ct) - lvl_in;
    const uint32_t lvl_b = level_of(out_b.ct) - lvl_in;

    // ── correctness ──
    std::vector<double> ym = decrypt_slots(inf, out_m);
    std::vector<double> yb = decrypt_slots(inf, out_b);
    std::vector<double> oracle = oracle_head_reduce_sum(xv, slots, t, tH);

    AccStats sm  = compare_vec(ym, oracle);   // masked   vs plaintext truth
    AccStats sb  = compare_vec(yb, oracle);   // reduce-bcast vs plaintext truth
    AccStats smb = compare_vec(yb, ym);       // reduce-bcast vs masked (equivalence)

    // ── latency ──
    const int iters = 50;
    auto [tm_mean, tm_min] = time_variant(inf, x, &head_reduce_sum_masked, iters);
    auto [tb_mean, tb_min] = time_variant(inf, x, &head_reduce_sum_reduce_bcast, iters);

    // ── report ──
    std::cout << "\n================ head_reduce_sum: masked-cyclic vs reduce-then-broadcast ================\n";
    std::cout << "[op counts]            rotations  pt_mults   ct_adds   clones\n";
    std::cout << "  masked-cyclic        " << std::setw(9) << cm.rotations
              << std::setw(10) << cm.pt_mults << std::setw(10) << cm.adds
              << std::setw(9) << cm.clones << "\n";
    std::cout << "  reduce-broadcast     " << std::setw(9) << cb.rotations
              << std::setw(10) << cb.pt_mults << std::setw(10) << cb.adds
              << std::setw(9) << cb.clones
              << "   <- " << (cm.pt_mults - cb.pt_mults) << " fewer plaintext-mults\n";
    std::cout << "[levels consumed]  masked=" << lvl_m << "  reduce-bcast=" << lvl_b
              << "   (deterministic rescales on the output path: masked=log2(t)=" << log2t
              << ", reduce-bcast=1; a measured 0 = the single rescale is still lazy)\n";
    report_acc("masked   vs oracle ", sm);
    report_acc("reduce-bcast vs oracle", sb);
    report_acc("reduce-bcast vs masked", smb);
    std::cout << std::fixed << std::setprecision(3)
              << "[latency over " << iters << " iters]  masked: mean=" << tm_mean
              << "ms min=" << tm_min << "ms   reduce-bcast: mean=" << tb_mean
              << "ms min=" << tb_min << "ms   speedup(mean)="
              << (tm_mean / std::max(tb_mean, 1e-9)) << "x\n";
    std::cout << "==========================================================================================\n\n";
    std::cout.unsetf(std::ios::fixed);

    // ── assertions ──
    // (1) correctness: both match the plaintext oracle, and each other. Magnitudes
    //     are O(t*R)~O(2000); a structural bug would mis-sum by O(magnitude), so a
    //     tight absolute bound cleanly separates "correct" from "broken".
    EXPECT_LT(sm.max_abs,  1e-2) << "masked variant disagrees with plaintext oracle";
    EXPECT_LT(sb.max_abs,  1e-2) << "reduce-then-broadcast disagrees with the plaintext oracle";
    EXPECT_LT(smb.max_abs, 1e-2) << "reduce-then-broadcast and masked are NOT equivalent";

    // (2) op counts (deterministic): equal rotations & adds; masked does
    //     2*log2(t) plaintext-mults, reduce-then-broadcast does exactly 1.
    EXPECT_EQ(cm.rotations, cb.rotations) << "rotation count must be unchanged";
    EXPECT_EQ(cm.adds,      cb.adds)      << "add count must be unchanged";
    EXPECT_EQ(cm.pt_mults,  2 * log2t)    << "masked first phase = 2*log2(t) pt-mults";
    EXPECT_EQ(cb.pt_mults,  1)            << "reduce-then-broadcast first phase = 1 pt-mult";

    // (3) levels: reduce-then-broadcast consumes strictly fewer multiplicative levels.
    EXPECT_LT(lvl_b, lvl_m) << "reduce-then-broadcast must consume fewer levels than masked";
    EXPECT_LE(lvl_b, 1u)    << "reduce-then-broadcast first phase costs at most 1 level";

    // (4) latency: reduce-then-broadcast is faster (it does strictly fewer ops).
    EXPECT_LT(tb_mean, tm_mean) << "reduce-then-broadcast should be faster wall-clock";
}
