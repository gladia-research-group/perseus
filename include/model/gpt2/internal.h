#pragma once

#include "inference.h"
#include "encoded_block.h"
#include "weight_loader.h"
#include "config_loader.h"
#include "model/block_residency.h"   // generic block residency: loader/evict/reclaim/block_scope

#include <cuda_runtime.h>
#include <functional>
#include <string>
#include <vector>

// Per-subgraph runtime-graph reset (naming vars/counters) for capture and
// planned strict tracking. Defined in gpt2_model_api.cu.
void gpt2_reset_graph_runtime(Inference& inf);

// Rotation steps exclusive to the CachemirFilling (prefill) packing — i.e. needed by
// filling but not by Cachemir (decode) nor the default rot-key set. Safe to free once
// prefill is done. Defined in gpt2_model.cu.
std::vector<int32_t> gpt2_filling_only_rot_steps(int slots, int hidDim, int ffDim, int numHeads);

// Rotation steps exclusive to Cachemir (decode) vs filling+default. Deferred off the
// device during prefill, GPU-loaded at the handoff. Defined in gpt2_model.cu.
std::vector<int32_t> gpt2_decode_only_rot_steps(int slots, int hidDim, int ffDim, int numHeads);

void offload_block_kv(Inference& inf);
void reload_block_kv(Inference& inf);
// Overlapped (FHE_KV_OVERLAP=1, default) staged-swap pipeline: seed reload(0) before the block loop,
// prefetch reload(b+1) / defer offload(b) at each block, drain+evict the last block after the loop.
void gpt2_kv_prefetch_first(Inference& inf, int n_blocks);
void gpt2_kv_block_prologue(Inference& inf, int b, int n_blocks);
void gpt2_kv_finalize_last(Inference& inf, int n_blocks);
void gpt2_reset_kv_cache(Inference& inf, int n_blocks);

// Planned-decode mask precompute: pre-encode the FIXED (token-invariant) cachemir selector
// plaintexts (masks) at the plan-supplied levels. The per-step (per-position) masks are NOT primed
// here — they go through the residency below to bound host memory at large T.
void gpt2_generate_decode_masks(Inference& inf, std::vector<EncodedBlock>& blocks,
                                int n_blocks, int T);
// Per-step mask residency: prime step `step`'s per-position masks (kc=step+1, right_rot=step%t,
// pos=step) just before token `step`; evict step `step-1`'s after. Keeps the host mask set ~1 step.
// Tail prefetch (2026-07-22): gpt2_prefill preps the lm_head tiles on the
// residency worker during the first chunk's last block; the tail consumer
// (GPT2Model::logit_tiles) takes them via this call (true once, then empty).
bool gpt2_tail_lm_take(std::vector<EncodedBlock>& out, std::vector<std::string>& complex_keys);
void gpt2_prime_step_masks(Inference& inf, std::vector<EncodedBlock>& blocks, int n_blocks, int step);
void gpt2_evict_step_masks(Inference& inf, std::vector<EncodedBlock>& blocks, int n_blocks, int step);
// Prefetch one block's `step` masks from the decode body (block_prefix already set to the block's scope).
void gpt2_prime_block_masks(Inference& inf, EncodedBlock& blk, int step);

// Bridge a cachemir_filling prefill's per-block K/V into the cachemir decode KV
// cache by re-pushing each of the m prefilled tokens through the verified
// cachemir push. inf.packing must be Cachemir on entry. See gpt2_model.cu.
void gpt2_kv_handoff_filling_to_cachemir(Inference& inf, int n_blocks, int m);

// Extract token i of a cachemir_filling group ct into a cachemir single-token ct
// (used by the filling->cachemir decode handoff).
PackedCtx extract_token_i_cachemir(Inference& inf, const PackedCtx& filling_ct, int i);

void gpt2_block_step(Inference& inf, PackedCtx& x, int b,
                     const std::function<void(Inference&)>& kv_prologue);
// Decode-arm pipeline release hook: generic evict_block_weights + the KV offload
// pipeline. Passed to run_blocks/run_cached_blocks as the per-block release; the
// prefill path additionally calls reclaim_host_async per block and
// finish_host_reclaim once (both in model/block_residency.h).
void gpt2_block_release(Inference& inf, int b);
PackedCtx apply_final_ln(Inference& inf, PackedCtx& x, EncodedBlock& lnf);

PackedCtx gpt2_decode_forward(Inference& inf, PackedCtx x, int t, int n_blocks,
                              std::vector<EncodedBlock>& blocks,
                              const BlockLoader& loader, EncodedBlock& lnf,
                              bool planned);
