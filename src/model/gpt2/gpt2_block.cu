#include "model/gpt2.h"
#include "model/gpt2/internal.h"
#include "attention.h"
#include "encoded_block.h"    // run_ops / run_blocks / install/evict block state
#include "model/layer_norm.h"
#include "nonlinear.h"        // norm()
#include "model/mha.h"
#include "model/mlp.h"
#include "weight_loader.h"
#include "packing/cachemir/cachemir_attention_utils.h"

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <memory>
#include <vector>
#include <filesystem>

namespace {

const char* graph_dir_env() {
    const char* v = std::getenv("FHE_GRAPH_DIR");
    return (v && *v) ? v : nullptr;
}

inline int graph_capture_token() {
    static const int t = []() {
        const char* v = std::getenv("FHE_GRAPH_CAPTURE_TOKEN");
        return (v && *v) ? std::atoi(v) : 0;
    }();
    return t;
}

// Chunked prefill (capture_chunk >= 0) captures EVERY chunk into chunk_<c>/;
// flat captures (decode, generation) keep the capture_t gate and flat block dirs.
inline bool capture_wanted(const Inference& inf) {
    return inf.output.capture_chunk >= 0
        || inf.output.capture_t == graph_capture_token();
}

inline std::filesystem::path capture_block_dir(const Inference& inf, const char* graph_dir, int b) {
    std::filesystem::path p(graph_dir);
    if (inf.output.capture_chunk >= 0)
        p /= "chunk_" + std::to_string(inf.output.capture_chunk);
    return p / ("block_" + std::to_string(b));
}
}

static std::vector<Op> gpt2_block_ops(bool tiled_mlp, int n_tiles, bool token_pair) {
    auto skip = std::make_shared<PackedCtx>();   // residual skip (block input, then r1)
    std::vector<Op> ops;

    ops.push_back({ {}, [skip](Inference& i, PackedCtx& x) {
        const int res_lvl = i.fhe->level_for_ct(x.ct)
                          + static_cast<int>(i.pending_rescale_primes(x.ct));
        i.name_graph_ct_if_absent(x, "transformer_block.x-lvl=" + std::to_string(res_lvl));
        *skip = x; }, "block_in" });

    ops.push_back({ {}, [](Inference& i, PackedCtx& x) {
        PackedCtx normed = norm(i, x, "ln_1");
        if (fold_ln_affine("ln_1")) {   // gamma -> QKV; beta rides input as (beta/gamma) shift
            { WithStep _ws(i, "ln_shift");
              i.add_affine_term(normed, "ln_1.shift"); }
            x = normed;
        } else {
            x = ln_affine(i, normed, "ln_1");
        }
    }, "ln_1" });

    for (auto& op : (token_pair ? mha_ops_token_pair() : mha_ops())) ops.push_back(std::move(op));
    ops.push_back({ {}, [skip](Inference& i, PackedCtx& x) {   // attn residual (ln_2 re-masks)
        x = i.fhe->add(x, *skip);   // r1 = attn + block input
        *skip = x;                  // skip <- r1 for the MLP residual
    }, "attn_residual" });

    ops.push_back({ {}, [](Inference& i, PackedCtx& x) {
        x = layer_norm(i, x, "ln_2");
    }, "ln_2" });

    if (tiled_mlp)
        for (auto& op : mlp_tiled_ops(n_tiles)) ops.push_back(std::move(op));
    else
        for (auto& op : mlp_ops()) ops.push_back(std::move(op));
    ops.push_back({ {}, [skip](Inference& i, PackedCtx& x) {   // mlp residual (next ln_1/ln_f re-masks)
        x = i.fhe->add(x, *skip);   // out = mlp + r1
    }, "mlp_residual" });

    return ops;
}

PackedCtx transformer_block(Inference& inf, PackedCtx& x) {
    const char* graph_dir = graph_dir_env();

    bool graph_capture = (graph_dir != nullptr) && !inf.graph_capture_enabled()
                         && capture_wanted(inf);

    if (graph_capture) {
        std::filesystem::path existing =
            capture_block_dir(inf, graph_dir, inf.output.capture_b) / "graph.json";
        if (std::filesystem::exists(existing)) graph_capture = false;
    }

    if (graph_capture) {
        inf.enable_graph_capture();
    }
    WithStep _w(inf, "transformer_block");
    const bool stream = streams_within_block(inf.weight_granularity);
    const Overlap ov  = stream ? device_overlap(inf.mode) : Overlap::Sync;
    const bool tiled_mlp = inf.tiled_mlp();
    const int  n_tiles   = inf.size.getRealFfDim() / inf.size.hidDim;
    const bool token_pair = is_cachemir_filling(inf.packing) && inf.token_pair;
    auto out = run_ops(inf, x, gpt2_block_ops(tiled_mlp, n_tiles, token_pair), stream, ov);
    if (graph_capture) {
        std::filesystem::path out_dir = capture_block_dir(inf, graph_dir, inf.output.capture_b);
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);
        if (!ec) {
            inf.export_graph_json((out_dir / "graph.json").string());
        }
        inf.disable_graph_capture();
    }
    return out;
}

void gpt2_block_step(Inference& inf, PackedCtx& x, int b,
                            const std::function<void(Inference&)>& kv_prologue) {
    inf.block_prefix = block_scope(b);
    kv_prologue(inf);
    x = transformer_block(inf, x);
}

bool begin_subgraph_capture(Inference& inf, int capture_b) {
    const char* graph_dir = graph_dir_env();
    bool cap = (graph_dir != nullptr) && !inf.graph_capture_enabled()
               && capture_wanted(inf);
    if (cap) {
        std::filesystem::path existing =
            capture_block_dir(inf, graph_dir, capture_b) / "graph.json";
        if (std::filesystem::exists(existing)) cap = false;
    }
    if (cap) { inf.output.capture_b = capture_b; inf.enable_graph_capture(); }
    return cap;
}

void end_subgraph_capture(Inference& inf, int capture_b) {
    const char* graph_dir = graph_dir_env();
    if (!graph_dir || !inf.graph_capture_enabled()) return;
    std::filesystem::path out_dir = capture_block_dir(inf, graph_dir, capture_b);
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (!ec) inf.export_graph_json((out_dir / "graph.json").string());
    inf.disable_graph_capture();
}

PackedCtx apply_final_ln(Inference& inf, PackedCtx& x, EncodedBlock& lnf) {
    { WithStep _w(inf, "lnf_install");
      load_block_to_device(inf, lnf, nullptr);
      install_block_state_copy(inf, lnf); }
    inf.name_graph_ct_if_absent(x, "ln_f.x");
    PackedCtx h = layer_norm(inf, x, "ln_f");
    if (const char* cap = std::getenv("FHE_LMHEAD_CAP"); cap && *cap)
        inf.fhe->bootstrap_hint(h, std::atoi(cap), /*account_pending_rescale=*/true);
    { WithStep _w(inf, "lnf_sync"); cudaDeviceSynchronize(); }
    evict_block_from_device(inf, lnf);
    inf.weight_store = nullptr;
    return h;
}
