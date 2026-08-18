#include "inference.h"
#include "packing/diagonal/diagonal_linear.h"
#include "packing/diagonal/diagonal_linear_utils.h"
#include "residency_pipeline.h"   // per-layer chunked-prefetch residency (prefill)

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace diagonal {

PackedCtx linear(Inference& inf, const PackedCtx& x_in,
                 const std::string& wname, int d_in, int d_out, bool stream_pt_arg) {
    auto p = compute_dg_params(inf.slots, d_in, d_out);

    if (inf.n_tok > p.max_n_tok)
        throw std::runtime_error(
            "diagonal::linear[" + wname + "]: n_tok=" + std::to_string(inf.n_tok) +
            " exceeds max_n_tok=" + std::to_string(p.max_n_tok) + " for " +
            std::to_string(d_in) + "x" + std::to_string(d_out) +
            " (prefill batch capped by slots/max(d_in,d_out); raise logN or reduce T)");

    auto& pts_W = inf.w.at(wname);
    if (static_cast<int>(pts_W.size()) != p.s * p.G)
        throw std::runtime_error("diagonal::linear: weight pt count != s*G");

    const bool stream_pt = stream_pt_arg || (inf.weight_granularity == WeightGranularity::Plaintext);

    // FHE_PT_PREFETCH=1 routes the streaming weight path through the residency pipeline's
    // worker-side CPU extraction (ResidencyStage::prefetch_cpu -> Inference::extract_plaintext
    // -> CryptoContextImpl::ExtractRawPlaintext). That moves GetRawPlainText's ~4 MB limb
    // flatten off the critical path AND, with FHE_PIN_STAGE (default on), stages it into the
    // pinned arena so the upload becomes a genuinely async H2D instead of a synchronous
    // pageable one. Measured serial cost it attacks: 947 us of load vs 35 us of multiply per
    // product (test_ptmult_isolation arms C/A). CPU-side only => graph-neutral, no re-capture.
    static const bool pt_prefetch = [] {
        const char* e = std::getenv("FHE_PT_PREFETCH");
        return e && *e && std::atoi(e) != 0;
    }();

    const bool chunked = is_cachemir_filling(inf.packing)
                      && (inf.weight_granularity == WeightGranularity::Linear
                          || inf.weight_granularity == WeightGranularity::Sublayer
                          || pt_prefetch);
    auto mult_pt = [&](const PackedCtx& xb, int idx) {
        if (stream_pt) inf.load_plaintext(pts_W[idx], nullptr);
        PackedCtx r = inf.fhe->mult(xb, pts_W[idx]);
        if (stream_pt) inf.evict_plaintext(pts_W[idx]);
        return r;
    };

    PackedCtx x_rep = inf.fhe->clone(x_in);
    if (p.is_up && p.alpha > 1) {
        std::vector<int32_t> steps;
        steps.reserve(p.alpha - 1);
        for (int g = 1; g < p.alpha; ++g) steps.push_back(dg_rot(inf, -g * p.t_out));
        std::vector<PackedCtx> rots = inf.fhe->rotate_hoisted(x_in, steps);
        for (auto& r : rots) inf.fhe->inplace_add(x_rep, r);
    }

    std::vector<PackedCtx> x_b(p.s);
    x_b[0] = x_rep;
    if (p.s > 1) {
        std::vector<int32_t> steps;
        steps.reserve(p.s - 1);
        for (int b = 1; b < p.s; ++b) steps.push_back(dg_rot(inf, b * p.t_in));
        std::vector<PackedCtx> rots = inf.fhe->rotate_hoisted(x_rep, steps);
        for (int b = 1; b < p.s; ++b) x_b[b] = std::move(rots[b - 1]);
    }

    // SCRATCH REUSE. `mult(xb, pt)` mints a fresh ciphertext per product, and its
    // copy-ctor deep-copies the OpenFHE CPU shadow — which for a GPU-resident ct is
    // stale and never read. Measured at 8 limbs: 605 us for that copy vs 8 us for
    // the actual multiply. BSGS needs n_diag = s*G separate products per layer, so
    // this dominated the linears (75% of the ViT block wall).
    // Instead: one scratch ciphertext per layer, refilled DEVICE-SIDE from the
    // (immutable) rotated input before each in-place multiply — 35 us/product
    // measured, 17x cheaper. Values are identical: copy_into duplicates limbs and
    // metadata, and mult-then-add == copy-mult-then-add.
    // The scratch is per-call (not static): concurrent linears must not share it.
    // SCRATCH REUSE (reverted 2026-07-20, do not re-land without a value gate):
    // replacing the per-product `mult` with copy_into(scratch, x_b[b]) +
    // inplace_mult is 17x cheaper in isolation (34.8 us vs 604.8 us — the saving is
    // the copy-ctor's deep copy of the stale OpenFHE CPU shadow), but the k=1 ViT
    // gate came back with "Decrypt: approximation error too high" (job 49898400).
    // The device copy is therefore NOT value-equivalent as used here; the primitive
    // (CryptoContextImpl::CopyCiphertextDevice + fhe->copy_into) is kept and is
    // unused until tests/test_ptmult_isolation.cu's correctness arm says which of
    // copy_into / inplace_mult diverges.
    auto inner_sum_for_giant = [&](int g) {
        PackedCtx acc = mult_pt(x_b[0], 0 * p.G + g);
        for (int b = 1; b < p.s; ++b) {
            PackedCtx tmp = mult_pt(x_b[b], b * p.G + g);
            inf.fhe->inplace_add(acc, tmp);
        }
        return acc;
    };

    PackedCtx y;
    if (chunked) {
        // 512 ≈ 3.2 GB/chunk @6.3 MB/pt (the baked GPT-2 prefill value). Under FHE_PT_PREFETCH
        // that is far too coarse: run_residency_pipeline extracts stages 0 and 1 on the MAIN
        // thread and only pipelines from stage 2, so at the ViT's n_chunks=2 nothing would
        // overlap at all. Smaller chunks put almost every extraction on the worker.
        static const int chunk_pts = [] {
            const char* e = std::getenv("FHE_PT_CHUNK_PTS");
            if (e && *e && std::atoi(e) > 0) return std::atoi(e);
            return pt_prefetch ? 64 : 512;   // fn-local static: usable in a non-capturing lambda
        }();
        const int gpc      = std::max(1, chunk_pts / std::max(1, p.s));   // giant-steps/block
        const int n_chunks = (p.G + gpc - 1) / gpc;

        auto y_sp   = std::make_shared<PackedCtx>();
        auto have_y = std::make_shared<bool>(false);

        std::vector<ResidencyStage> blocks;
        blocks.reserve(n_chunks);
        for (int c = 0; c < n_chunks; ++c) {
            const int g0 = c * gpc;
            const int g1 = std::min(p.G, g0 + gpc);
            ResidencyStage st;
            st.label   = "dg_chunk";
            // Worker-thread host extraction of the NEXT chunks' weights (runs two stages
            // ahead in run_residency_pipeline's prefetch_cpu branch). CPU + pinned-arena
            // memcpy only — no CUDA, no inf.w mutation — so it is safe off the main thread.
            if (pt_prefetch) {
                st.prefetch_cpu = [&pts_W, &p, g0, g1](Inference& i) {
                    for (int g = g0; g < g1; ++g)
                        for (int b = 0; b < p.s; ++b) i.extract_plaintext(pts_W[b * p.G + g]);
                };
            }
            st.acquire = [&pts_W, &p, g0, g1](Inference& i, cudaStream_t stream) {
                for (int g = g0; g < g1; ++g)
                    for (int b = 0; b < p.s; ++b) i.load_plaintext(pts_W[b * p.G + g], stream);
            };
            st.compute = [&pts_W, &x_b, &p, g0, g1, y_sp, have_y](Inference& i) {
                for (int g = g0; g < g1; ++g) {
                    PackedCtx acc = i.fhe->mult(x_b[0], pts_W[0 * p.G + g]);
                    for (int b = 1; b < p.s; ++b) {
                        PackedCtx tmp = i.fhe->mult(x_b[b], pts_W[b * p.G + g]);
                        i.fhe->inplace_add(acc, tmp);
                    }
                    if (!*have_y) { *y_sp = acc; *have_y = true; }
                    else {
                        PackedCtx rot = i.fhe->rotate(acc, dg_rot(i, g * p.s * p.t_in));
                        i.fhe->inplace_add(*y_sp, rot);
                    }
                }
            };
            st.release = [&pts_W, &p, g0, g1](Inference& i) {
                for (int g = g0; g < g1; ++g)
                    for (int b = 0; b < p.s; ++b) i.evict_plaintext(pts_W[b * p.G + g]);
            };
            blocks.push_back(std::move(st));
        }
        // Flip the double-buffered pinned staging arena once per LINEAR, not per chunk: the
        // ping-pong aliases stage i with stage i+2, and a per-chunk flip would let the worker
        // refill the arena a still-in-flight upload is reading. Per-linear is safe because the
        // previous linear's compute has fully drained. Chunks past the arena's capacity simply
        // fall back to the pageable upload (stage_raw keeps sub_0) — slower, never wrong.
        if (pt_prefetch) inf.begin_stage_block();
        run_residency_pipeline(inf, std::move(blocks), Overlap::Stream);
        y = *y_sp;
    } else {
        y = inner_sum_for_giant(0);
        for (int g = 1; g < p.G; ++g) {
            PackedCtx inner = inner_sum_for_giant(g);
            PackedCtx rotated = inf.fhe->rotate(inner, dg_rot(inf, g * p.s * p.t_in));
            inf.fhe->inplace_add(y, rotated);
        }
    }

    auto bias_it = inf.w.find(wname + "_bias");
    if (bias_it != inf.w.end() && !bias_it->second.empty()) {
        Ptx pt_at_y = inf.encode_additive_like(
            inf.scoped(wname + "_bias"), y,
            [&]{ return bias_it->second[0]->GetRealPackedValue(); });
        y = inf.fhe->add(y, pt_at_y);
    }

    return y;
}

}  // namespace diagonal
