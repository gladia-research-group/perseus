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

    // Prefill streams the diagonal plaintexts through the residency pipeline in chunks
    // when the weights are not block-resident.
    const bool chunked = is_cachemir_filling(inf.packing)
                      && (inf.weight_granularity == WeightGranularity::Linear
                          || inf.weight_granularity == WeightGranularity::Sublayer);
    auto mult_pt = [&](const PackedCtx& xb, int idx) {
        if (stream_pt) inf.load_plaintext(pts_W[idx], nullptr);
        PackedCtx r = inf.fhe->mult(xb, pts_W[idx]);
        if (stream_pt) inf.evict_plaintext(pts_W[idx]);
        return r;
    };
    auto mac_pt = [&](PackedCtx& acc, const PackedCtx& xb, int idx) {
        if (stream_pt) inf.load_plaintext(pts_W[idx], nullptr);
        PackedCtx tmp = inf.fhe->mult(xb, pts_W[idx]);
        inf.fhe->inplace_add(acc, tmp);
        if (stream_pt) inf.evict_plaintext(pts_W[idx]);
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
    // Realize the pending rescale on the rotated inputs once, before the pt-mult fan-out.
    if (!inf.graph_capture_enabled())
        for (auto& xb : x_b) inf.fhe->realize_pending_rescale_raw(xb.ct);

    auto inner_sum_for_giant = [&](int g) {
        PackedCtx acc = mult_pt(x_b[0], 0 * p.G + g);
        for (int b = 1; b < p.s; ++b) mac_pt(acc, x_b[b], b * p.G + g);
        return acc;
    };

    PackedCtx y;
    if (chunked) {
        constexpr int chunk_pts = 512;   // plaintexts per residency chunk
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
            st.acquire = [&pts_W, &p, g0, g1](Inference& i, cudaStream_t stream) {
                for (int g = g0; g < g1; ++g)
                    for (int b = 0; b < p.s; ++b) i.load_plaintext(pts_W[b * p.G + g], stream);
            };
            st.compute = [&pts_W, &x_b, &p, g0, g1, y_sp, have_y](Inference& i) {
                auto emit = [&](int g, PackedCtx acc) {
                    if (!*have_y) { *y_sp = std::move(acc); *have_y = true; }
                    else {
                        PackedCtx rot = i.fhe->rotate(acc, dg_rot(i, g * p.s * p.t_in));
                        i.fhe->inplace_add(*y_sp, rot);
                    }
                };
                for (int g = g0; g < g1; ++g) {
                    PackedCtx acc = i.fhe->mult(x_b[0], pts_W[0 * p.G + g]);
                    for (int b = 1; b < p.s; ++b) {
                        PackedCtx tmp = i.fhe->mult(x_b[b], pts_W[b * p.G + g]);
                        i.fhe->inplace_add(acc, tmp);
                    }
                    emit(g, std::move(acc));
                }
            };
            st.release = [&pts_W, &p, g0, g1](Inference& i) {
                for (int g = g0; g < g1; ++g)
                    for (int b = 0; b < p.s; ++b) i.evict_plaintext(pts_W[b * p.G + g]);
            };
            blocks.push_back(std::move(st));
        }
        run_residency_pipeline(inf, std::move(blocks), Overlap::Stream);
        y = *y_sp;
    } else {
        bool have_y = false;
        auto emit = [&](int g, PackedCtx acc) {
            if (!have_y) { y = std::move(acc); have_y = true; }
            else {
                PackedCtx rotated = inf.fhe->rotate(acc, dg_rot(inf, g * p.s * p.t_in));
                inf.fhe->inplace_add(y, rotated);
            }
        };
        for (int g = 0; g < p.G; ++g) emit(g, inner_sum_for_giant(g));
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
