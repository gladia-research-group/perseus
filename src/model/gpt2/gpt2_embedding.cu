#include "model/gpt2.h"
#include "model/gpt2/internal.h"
#include "cutmax.h"
#include "encoded_block.h"
#include "op_sequence.h"
#include "packing/cachemir/cachemir_linear_utils.h"
#include "weight_loader.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

void gpt2_add_positional(Inference& inf, PackedCtx& h,
                         const weight_loader::WeightStore& store, int position) {
    const int d_real = inf.size.getRealHidDim();
    const int t      = inf.slots / inf.size.hidDim;

    const std::string wname = weight_loader::gpt2_wpe_name();
    const auto& flat  = store.tensor(wname);
    const auto& shape = store.meta(wname).shape;
    if (shape.size() != 2)
        throw std::runtime_error("gpt2_add_positional: wpe is not 2D: " + wname);
    const int max_pos = static_cast<int>(shape[0]);
    const int d_cols  = static_cast<int>(shape[1]);
    if (position < 0 || position >= max_pos)
        throw std::runtime_error("gpt2_add_positional: position " +
                                 std::to_string(position) + " out of [0," +
                                 std::to_string(max_pos) + ")");

    std::vector<double> wpe_row(d_real);
    const double* row = flat.data() + static_cast<size_t>(position) * d_cols;
    for (int j = 0; j < d_real; ++j) wpe_row[j] = row[j];

    Ptx pt = inf.encode_stride_values_at(d_real, t, h.ct, wpe_row);
    h = inf.fhe->add(h, pt);
}

namespace {
std::string fb_tile_key(int k) { return "fb_tile_" + std::to_string(k); }
const char* graph_dir_env() {
    const char* v = std::getenv("FHE_GRAPH_DIR");
    return (v && *v) ? v : nullptr;
}
}  // namespace

void gpt2_prepare_feedback_weights(Inference& inf,
                                   const weight_loader::WeightStore& store,
                                   int vocab, int W_tile, bool packed_z,
                                   std::vector<EncodedBlock>& cached_tiles,
                                   const BootstrapPlan* plan14) {
    if (cached_tiles.empty()) cached_tiles.emplace_back();
    
    EncodedBlock& blk = cached_tiles.front();

    if (!blk.w.empty()) return;

    const int d_pad  = inf.size.hidDim;
    const int d_real = inf.size.getRealHidDim();
    const int K      = (vocab + W_tile - 1) / W_tile;

    const uint32_t enc_fallback = inf.fhe->bootstrap_output_level();
    auto enc_lv = [&](int k) {
        return static_cast<int>(plan14 && plan14->valid
            ? plan14->weight_level("fb_tile_" + std::to_string(k), enc_fallback)
            : enc_fallback);
    };
    auto W_lm = weight_loader::load_gpt2_lm_head_weight(
        inf, store, d_real, d_pad, vocab);
    auto slice = [&](int k, double scale) {
        const int col0  = k * W_tile;
        const int wreal = std::min(W_tile, vocab - col0);
        std::vector<std::vector<double>> W_fb(
            static_cast<size_t>(W_tile),
            std::vector<double>(static_cast<size_t>(d_pad), 0.0));
        for (int r = 0; r < wreal; ++r)
            for (int j = 0; j < d_real; ++j)
                W_fb[r][j] = scale * W_lm.W_pad[j][col0 + r];
        return W_fb;
    };
    if (packed_z) {
        blk.w[fb_tile_key(0)] = cachemir::encode_weight_matrix_complex(
            inf, slice(0, 0.5), slice(1, -0.5), W_tile, d_pad, enc_lv(0));
    } else {
        for (int k = 0; k < K; ++k)
            blk.w[fb_tile_key(k)] = cachemir::encode_weight_matrix(
                inf, slice(k, 1.0), W_tile, d_pad, enc_lv(k));
    }
    evict_block_from_device(inf, blk);   // host-cache; per-token load/evict
}

PackedCtx gpt2_feedback_embed(Inference& inf,
                              const std::vector<PackedCtx>& z_tiles,
                              int vocab, int W_tile) {
    const int d_pad = inf.size.hidDim;
    const int K     = (vocab + W_tile - 1) / W_tile;

    const bool packed_z = (static_cast<int>(z_tiles.size()) == 1 && K == 2);
    if (!packed_z && static_cast<int>(z_tiles.size()) != K)
        throw std::runtime_error(
            "gpt2_feedback_embed: expected " + std::to_string(K) +
            " REAL vocab tiles or one packed pair");

    PackedCtx emb;
    if (packed_z) {
        emb = inf.fhe->im_cleanse(linear(inf, z_tiles[0], fb_tile_key(0),
                                         /*d_in=*/W_tile, /*d_out=*/d_pad));
    } else
    for (int k = 0; k < K; ++k) {
        PackedCtx zk = z_tiles[k];
        PackedCtx yk = linear(inf, zk, fb_tile_key(k),
                              /*d_in=*/W_tile, /*d_out=*/d_pad);
        if (k == 0) emb = yk;
        else        inf.fhe->inplace_add(emb, yk);
    }
    return emb;
}

PackedCtx gpt2_cutmax_feedback(Inference& inf,
                               const std::vector<PackedCtx>& tiles,
                               const weight_loader::WeightStore& store,
                               int vocab, int W_tile,
                               const CutMaxConfig& cmc,
                               EncodedBlock* fb_blk, int position,
                               std::vector<PackedCtx>* z_out,
                               double* argmax_s,
                               const BootstrapPlan* plan13,
                               const BootstrapPlan* plan14) {
    auto z = std::make_shared<std::vector<PackedCtx>>(tiles);
    auto cm_s = std::make_shared<double>(0.0);
    const bool p13 = plan13 && plan13->valid;
    const bool p14 = plan14 && plan14->valid;

    std::vector<Op> ops;
    Op cm;
    cm.label = "cutmax";
    if (p13)
        cm.install = [plan13](Inference& i) {
            i.fhe->install_plan_live(*plan13);
        };
    cm.fwd = [z, vocab, cm_s, p13, &cmc](Inference& i, PackedCtx& x) {
        const auto t0 = std::chrono::steady_clock::now();
        // block-13 subgraph: capture (FHE_GRAPH_DIR, token 0) / strict plan
        if (p13 || graph_dir_env()) gpt2_reset_graph_runtime(i);
        begin_subgraph_capture(i, 13);
        if (z->size() == 2 && i.fhe->complex_payload) {
            Ptx i_pt = i.encode_complex_const_at(0.0, 1.0, (*z)[1].ct);
            *z = { i.fhe->pack_ri((*z)[0], (*z)[1], i_pt) };
        }
        *z = cutmax_argmax(i, *z, vocab, cmc);
        x = z->front();
        end_subgraph_capture(i, 13);

        cudaDeviceSynchronize(); // TODO: remove this sync once the graph capture is fixed to not leak across tokens
        *cm_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    };
    ops.push_back(std::move(cm));

    if (position >= 0) {
        if (!fb_blk || fb_blk->w.empty())
            throw std::runtime_error(
                "gpt2_cutmax_feedback: codebook not prepared "
                "(prepare_feedback_weights at model load)");
        Op fb;
        fb.label = "feedback";
        fb.acquire = [fb_blk](Inference& i, cudaStream_t s) {
            load_block_to_device(i, *fb_blk, s);
        };
        fb.install = [fb_blk, plan14, p14](Inference& i) {
            install_block_state_copy(i, *fb_blk);   // fb_blk plan invalid -> clears live
            if (p14) i.fhe->install_plan_live(*plan14);
        };
        fb.fwd = [z, &store, vocab, W_tile, position, p14](Inference& i,
                                                           PackedCtx& x) {
            // block-14 subgraph: fb linear + wpe + entry bts
            if (p14 || graph_dir_env()) gpt2_reset_graph_runtime(i);
            begin_subgraph_capture(i, 14);
            x = gpt2_feedback_embed(i, *z, vocab, W_tile);
            gpt2_add_positional(i, x, store, position);
            i.fhe->bootstrap(x.ct);   // fresh-input state for block 0
            end_subgraph_capture(i, 14);
        };
        fb.release = [fb_blk](Inference& i) {
            { WithStep _w(i, "block_sync"); cudaDeviceSynchronize(); }
            evict_block_from_device(i, *fb_blk);
        };
        ops.push_back(std::move(fb));
    }

    PackedCtx out = run_ops(inf, tiles.front(), std::move(ops),
                            /*stream_weights=*/false, device_overlap(inf.mode));
    if (p13 || p14) inf.clear_bootstrap_plan();   // tail plans die with the tail
    if (z_out) *z_out = *z;
    if (argmax_s) *argmax_s = *cm_s;
    return (position >= 0) ? out : PackedCtx{};
}
