#include "model/gpt2.h"
#include "encoded_block.h"         // EncodedBlock + load/evict/install
#include "op_sequence.h"           // Op / run_ops
#include "weight_loader.h"

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cmath>

std::vector<double> decode_lm_head_logits(Inference& inf,
                                          const std::vector<PackedCtx>& tiles,
                                          int vocab, int W_tile) {

    const int K = (vocab + W_tile - 1) / W_tile;
    std::vector<double> logits;
    logits.reserve(static_cast<size_t>(K) * W_tile);

    if (inf.complex) {
        for (size_t m = 0; m < tiles.size(); ++m) {
            auto pt   = decrypt_pt(inf.cc(), tiles[m].ct, inf.fhe->sk());
            auto cval = pt->GetCKKSPackedValue();
            std::vector<double> re(cval.size()), im(cval.size());
            for (size_t i = 0; i < cval.size(); ++i) {
                re[i] = cval[i].real();
                im[i] = cval[i].imag();
            }
            for (int lane = 0; lane < 2; ++lane) {
                const int k = 2 * static_cast<int>(m) + lane;
                if (k >= K) break;
                const std::vector<double>& src = (lane == 0) ? re : im;
                auto tile = decode_linear_output(tiles[m].packing, src, inf.slots,
                                                 inf.size.hidDim, W_tile);
                const int col0  = k * W_tile;
                const int wreal = std::min(W_tile, vocab - col0);
                logits.insert(logits.end(), tile.begin(), tile.begin() + wreal);
            }
        }
        return logits;
    }

    for (size_t k = 0; k < tiles.size(); ++k) {
        auto raw  = decrypt(inf.cc(), tiles[k].ct, inf.fhe->sk());
        auto tile = decode_linear_output(tiles[k].packing, raw, inf.slots,
                                         inf.size.hidDim, W_tile);
        const int col0  = static_cast<int>(k) * W_tile;
        const int wreal = std::min(W_tile, vocab - col0);
        logits.insert(logits.end(), tile.begin(), tile.begin() + wreal);
    }
    return logits;
}

static bool lm_head_streams() {
    const char* g = std::getenv("GPT2_LMHEAD_GRANULARITY");
    return g && *g && std::string(g) != "block";
}

static std::vector<Op> lm_head_ops(std::vector<PackedCtx>& tiles,
                                   const PreparedLinearInput& prep, int K) {
    std::vector<Op> ops;
    ops.reserve(static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) {
        const std::string key = weight_loader::gpt2_lm_head_tile_key(k);
        Op op{ {key, key + "_bias"},
            [&tiles, &prep, key, k](Inference& i, PackedCtx&) {
                tiles[static_cast<size_t>(k)] = apply_linear(i, prep, key);
            }, "lm_head_tile" };

        op.prefetch_cpu = [key](Inference& i) { cpu_extract_weight_keys(i, {key, key + "_bias"}); };
        ops.push_back(std::move(op));
    }
    return ops;
}

std::vector<PackedCtx> gpt2_lm_head(Inference& inf, const PackedCtx& x,
                                    const weight_loader::WeightStore& store,
                                    int vocab, int W_tile,
                                    std::vector<EncodedBlock>* cached_tiles,
                                    const BootstrapPlan& plan) {
    const int d_pad  = inf.size.hidDim;
    const int d_real = inf.size.getRealHidDim();
    const int K      = (vocab + W_tile - 1) / W_tile;
    const int K_eff  = inf.complex ? (K + 1) / 2 : K;
    const bool cached = (cached_tiles != nullptr);

    PackedCtx h = x;
    inf.fhe->sync_ciphertext_cpu_from_device(h.ct);
    inf.name_graph_ct_if_absent(h, "lm_head.x");

    EncodedBlock  local;
    EncodedBlock* blk = &local;
    if (cached) {
        if (cached_tiles->empty()) cached_tiles->emplace_back();
        blk = &cached_tiles->front();
    }
    if (blk->w.empty()) {
        auto enc = weight_loader::encode_gpt2_lm_head_weights(
            inf, store, d_real, d_pad, vocab, W_tile, plan);
        blk->w    = std::move(enc.w);
        blk->plan = plan;
        if (lm_head_streams())
            evict_block_from_device(inf, *blk);   // host-cached; tiles load per use
        else
            load_block_to_device(inf, *blk, /*stream=*/nullptr);   // resident; not evicted when cached
    }

    install_block_state_copy(inf, *blk);

    std::vector<PackedCtx> tiles(static_cast<size_t>(K_eff));
    { WithStep _w(inf, "lm_head");
      const PreparedLinearInput prep = prepare_linear_input(inf, h, d_pad, W_tile);
      inf.set_persistent_staging(true);
      run_ops(inf, h, lm_head_ops(tiles, prep, K_eff),
              /*stream_weights=*/lm_head_streams(), Overlap::Sync);
      inf.set_persistent_staging(false);

    }

    if (!cached) {
        evict_block_from_device(inf, *blk);
        inf.weight_store = nullptr;   // blk is the stack local here; don't leave weights_at a dangling canon map
    }
    return tiles;
}
