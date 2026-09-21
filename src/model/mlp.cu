#include "model/mlp.h"
#include "model/gpt2.h"
#include "model/layer_norm.h"   // fold_ln_affine
#include "nonlinear.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>

std::vector<Op> mlp_ops() {
    return {
        { {}, [](Inference& i, PackedCtx& x) {
            i.name_graph_ct_if_absent(x, "mlp_block.x"); }, "up_linear" },
        { {}, [](Inference& i, PackedCtx& x) {
            const int up_hint = i.fhe->level_headroom(i.complex ? 2 : 1);
            i.fhe->bootstrap_hint(x, up_hint, /*account_pending_rescale=*/true);
            i.fhe->level_hint(x, up_hint);
            x = i.complex
                ? linear_outputpack(i, x, "up", i.size.hidDim, i.size.expDim)   // S4: half the pt-mults
                : linear(i, x, "up", i.size.hidDim, i.size.expDim,
                         /*stream_pt=*/is_cachemir_filling(i.packing));}, "up_linear" },
        { {}, [](Inference& i, PackedCtx& x) {

            CKKSContext::LiveLaneScope _ll(
                *i.fhe, i.graph_capture_enabled() ? i.live_lane_mask_expanded() : nullptr);
            x = gelu_approx(i, x, "mlp.act"); }, "gelu" },
        { {}, [](Inference& i, PackedCtx& x) {
            i.fhe->bootstrap_hint(x, i.fhe->level_headroom(1), /*account_pending_rescale=*/true);
            i.fhe->level_hint(x, i.fhe->level_headroom(1));   // ceiling-relative: hardcoded 23 confiscated the +1 level at ceiling 25
            x =  linear(i, x, "down", i.size.expDim, i.size.hidDim,
                         /*stream_pt=*/is_cachemir_filling(i.packing)); }, "down_linear" },
    };
}

std::vector<Op> mlp_tiled_ops(int n_tiles) {
    auto prep     = std::make_shared<PreparedLinearInput>();   // up-tile input rotations, hoisted
    auto cur      = std::make_shared<PackedCtx>();
    auto acc      = std::make_shared<PackedCtx>();
    auto have_acc = std::make_shared<bool>(false);
    std::vector<Op> ops;

    ops.push_back({ {}, [prep, have_acc](Inference& i, PackedCtx& x) {
        i.name_graph_ct_if_absent(x, "mlp_block.x");
        i.fhe->bootstrap_hint(x, i.fhe->level_headroom(3));
        *prep = prepare_linear_input(i, x, i.size.hidDim, i.size.hidDim);
        *have_acc = false; }, "up_linear" });

    for (int j = 0; j < n_tiles; ++j) {
        const std::string tj = ".t" + std::to_string(j);
        const std::string up = "up" + tj;
        const std::string dn = "down" + tj;

        ops.push_back({ {}, [up, cur, prep](Inference& i, PackedCtx&) {
            *cur = apply_linear(i, *prep, up,
                                /*stream_pt=*/is_cachemir_filling(i.packing)); }, "up_linear" });

        ops.push_back({ {}, [cur](Inference& i, PackedCtx& /*x*/) {
            i.mlp_tile_dim = i.size.hidDim;          // gelu half-mask -> square layout
            // Tiled: the tile stays in the square (hidDim) layout, so the token mask is
            // the live-lane set for this elementwise region. Capture-only, as above.
            CKKSContext::LiveLaneScope _ll(
                *i.fhe, i.graph_capture_enabled() ? i.live_lane_mask_token() : nullptr);
            *cur = gelu_approx(i, *cur, "mlp.act");
            i.mlp_tile_dim = 0;
            i.fhe->bootstrap_hint(*cur, i.fhe->level_headroom(3)); }, "gelu" });

        ops.push_back({ {}, [dn, cur, acc, have_acc](Inference& i, PackedCtx& /*x*/) {
            PackedCtx part = linear(i, *cur, dn, i.size.hidDim, i.size.hidDim,
                                    /*stream_pt=*/is_cachemir_filling(i.packing));
            if (!*have_acc) { *acc = std::move(part); *have_acc = true; }
            else i.fhe->inplace_add(*acc, part); }, "down_linear" });
    }

    ops.push_back({ {}, [acc](Inference& i, PackedCtx& x) {
        auto bit = i.w.find("down_bias");
        if (bit != i.w.end() && !bit->second.empty()) {
            Ptx pb = i.encode_additive_like(i.scoped("down_bias"), *acc,
                                            [&]{ return bit->second[0]->GetRealPackedValue(); });
            i.fhe->inplace_add(*acc, pb);
        }
        x = std::move(*acc); }, "down_linear" });

    return ops;
}

PackedCtx mlp_block(Inference& inf, PackedCtx& x) {
    WithStep _w(inf, "mlp_block");
    const bool stream = streams_within_block(inf.weight_granularity);
    const Overlap ov  = stream ? device_overlap(inf.mode) : Overlap::Sync;
    if (inf.tiled_mlp())
        return run_ops(inf, x, mlp_tiled_ops(inf.size.getRealFfDim() / inf.size.hidDim),
                       stream, ov);
    return run_ops(inf, x, mlp_ops(), stream, ov);
}
