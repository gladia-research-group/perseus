#pragma once

#include "inference.h"
#include "residency_pipeline.h"
#include "op_sequence.h"          // Op / run_ops — a block is just an Op

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

struct EncodedBlock {
    std::string prefix;
    std::unordered_map<std::string, std::vector<Ptx>> w;
    std::unordered_map<std::string, std::vector<std::vector<double>>> raw_w;
    std::unordered_map<std::string, NormConfig> norm_cfg;
    std::unordered_map<std::string, SoftmaxConfig> sm_cfg;
    std::unordered_map<std::string, GeLUConfig> gelu_cfg;
    BootstrapPlan plan;
    std::vector<std::tuple<std::string, uint32_t, Ptx>> staged_masks;

    int stage_owner = -1;
};

void load_block_to_device(Inference& inf, EncodedBlock& blk, cudaStream_t stream);
void cpu_extract_block(Inference& inf, EncodedBlock& blk);
void evict_block_from_device(Inference& inf, EncodedBlock& blk);

void stage_plaintexts(Inference& inf, std::vector<Ptx>& pts, int n_threads);
void stage_block_weights(Inference& inf, EncodedBlock& blk, int n_threads);

inline int pt_stage_block_threads() {
    static const int v = [] {
        const char* e = std::getenv("FHE_PT_STAGE_BLOCK");
        const int n = (e && *e) ? std::atoi(e) : 0;
        return n <= 0 ? 0 : (n == 1 ? 12 : n);
    }();
    return v;
}

void load_weight_keys(Inference& inf, const std::vector<std::string>& keys,
                      cudaStream_t stream);
void evict_weight_keys(Inference& inf, const std::vector<std::string>& keys);
void cpu_extract_weight_keys(Inference& inf, const std::vector<std::string>& keys);

void install_block_state(Inference& inf, EncodedBlock&& state);
void install_block_state_copy(Inference& inf, const EncodedBlock& state);

void log_block(int b, int n_blocks, const char* label);

using BlockLoader = std::function<EncodedBlock(Inference&, int /*block*/, cudaStream_t)>;
using BlockRelease = std::function<void(Inference&, int /*block*/)>;


template <class BlockBody>
Op encode_block_op(int b, int n_blocks,
                   const BlockLoader& loader, BlockBody& body,
                   const BlockRelease& release, const char* label) {
    auto slot = std::make_shared<EncodedBlock>();
    Op op;
    op.label   = "block";
    op.acquire = [&loader, slot, b](Inference& i, cudaStream_t s) { *slot = loader(i, b, s); };
    op.install = [slot](Inference& i) { install_block_state(i, std::move(*slot)); };
    op.fwd     = [b, n_blocks, label, &body](Inference& i, PackedCtx& x) {
        log_block(b, n_blocks, label);
        body(i, x, b);
    };
    if (release) op.release = [release, b](Inference& i) { release(i, b); };
    return op;
}

template <class BlockBody>
Op cached_block_op(std::vector<EncodedBlock>& blocks, int b, int n_blocks,
                   BlockBody& body, bool stream_inner,
                   const BlockRelease& release, const char* label) {
    Op op;
    op.label   = "block";
    if (!stream_inner) {
        // Worker thread extracts the block's plaintexts into the staging arena
        // (prefetch_cpu); the main thread then uploads them (acquire).
        op.acquire = [&blocks, b](Inference& i, cudaStream_t s) { load_block_to_device(i, blocks[b], s); };
        op.prefetch_cpu = [&blocks, b](Inference& i) { cpu_extract_block(i, blocks[b]); };
    }
    op.install = [&blocks, b](Inference& i) { install_block_state_copy(i, blocks[b]); };

    // Circular staging: each block owns its staging half so the worker can
    // extract block b+1 while block b runs; the release happens after the op.
    if (!stream_inner) {
        op.stage_owner = b; blocks[b].stage_owner = b;
    } else {
        blocks[b].stage_owner = -1;
    }
    op.prefetch_next = !stream_inner;   // nothing to prefetch at block level when inner streams
    op.fwd     = [b, n_blocks, label, &body](Inference& i, PackedCtx& x) {
        log_block(b, n_blocks, label);
        body(i, x, b);
    };
    if (release) op.release = [release, b](Inference& i) { release(i, b); };
    return op;
}

// Run pre-encoded blocks reused across calls (decode): device-load per pass.
template <class BlockBody>
PackedCtx run_cached_blocks(Inference& inf, PackedCtx x,
                            std::vector<EncodedBlock>& blocks,
                            BlockBody body, BlockRelease release = {}) {
    const int n = static_cast<int>(blocks.size());
    if (n <= 0) return x;

    for (auto& blk : blocks) evict_block_from_device(inf, blk);

    const bool stream_inner = streams_within_block(inf.weight_granularity);
    const Overlap ov = stream_inner ? Overlap::Sync : device_overlap(inf.mode);
    const char* label = stream_inner            ? " (cached host + streamed weights)..."
                      : (ov == Overlap::Stream) ? " (cached host + async load)..."
                                                : " (cached host + sync load)...";
    std::vector<Op> ops;
    ops.reserve(static_cast<size_t>(n));
    for (int b = 0; b < n; ++b)
        ops.push_back(cached_block_op(blocks, b, n, body, stream_inner, release, label));
    PackedCtx out = run_ops(inf, std::move(x), std::move(ops), /*stream_weights=*/false, ov);
    { WithStep _w(inf, "enc_cache_evict"); inf.evict_enc_cache_device(); }
    return out;
}
template <class BlockBody>
PackedCtx run_blocks(Inference& inf, PackedCtx x, int n_blocks,
                     InferenceMode mode, const BlockLoader& loader, BlockBody body,
                     BlockRelease release = {}) {
    if (n_blocks <= 0) return x;
    std::vector<Op> ops;
    ops.reserve(static_cast<size_t>(n_blocks));

    const char* label = (mode == InferenceMode::Sync)     ? "..."
                      : (mode == InferenceMode::Prefetch) ? ""
                                                          : " (threaded)...";
    for (int b = 0; b < n_blocks; ++b)
        ops.push_back(encode_block_op(b, n_blocks, loader, body, release, label));
    PackedCtx out = run_ops(inf, std::move(x), std::move(ops), /*stream_weights=*/false, overlap_of(mode));
    // residency release of the token's transient mask device copies
    { WithStep _w(inf, "enc_cache_evict"); inf.evict_enc_cache_device(); }
    return out;
}
