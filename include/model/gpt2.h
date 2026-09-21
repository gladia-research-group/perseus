#pragma once

#include "inference.h"
#include "nonlinear.h"
#include "encoded_block.h"

#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace weight_loader {
class WeightStore;
}

namespace config_loader {
struct ParsedConfigs;
struct ModelConfig;
}

// Public dispatchers 

PackedCtx linear(Inference& inf, const PackedCtx& x,
                 const std::string& wname, int d_in, int d_out, bool stream_pt = false);

// Split linear: pay the packing-specific input prep once, then apply any number
// of weights over it. Pure compute — weight residency is never handled here
// (declare the keys on an Op::weights and run_ops moves them under the hood).
struct PreparedLinearInput {
    std::vector<PackedCtx> rotated;   // cachemir: interleaved input rotations
    PackedCtx x;                      // diagonal/cachemir_filling: input as-is (preps per apply)
    int d_in = 0, d_out = 0;
};
PreparedLinearInput prepare_linear_input(Inference& inf, const PackedCtx& x,
                                         int d_in, int d_out);
PackedCtx apply_linear(Inference& inf, const PreparedLinearInput& prep,
                       const std::string& wname, bool stream_pt = false);

std::vector<PackedCtx> linear_multi(Inference& inf, const PackedCtx& x,
                                    const std::vector<std::string>& wnames,
                                    int d_in, int d_out, bool stream_pt = false);

PackedCtx encode_linear_input(Inference& inf, const std::vector<double>& x,
                              int d_in, int d_out, int target_level = 0);

std::vector<double> decode_linear_output(const Packing& packing,
                                         const std::vector<double>& cy,
                                         int slots, int d_in, int d_out);

std::vector<std::vector<double>> decode_tokens(const Packing& packing,
                                               const std::vector<double>& cy,
                                               int slots, int d_pad, int d_real, int T);

// target_level: see cachemir/diagonal headers. Default 0 = full-level pts.
std::vector<Ptx> encode_weight_matrix(Inference& inf,
                                       const std::vector<std::vector<double>>& W,
                                       int d_in, int d_out,
                                       int target_level = 0);

std::vector<Ptx> encode_weight_matrix(Inference& inf,
                                       const std::vector<std::vector<double>>& W,
                                       int d_in, int d_out,
                                       int target_level,
                                       cudaStream_t stream);

// Complex weight (W_re + i*W_im) — cachemir-only; one linear emits W_re·x + i*W_im·x.
std::vector<Ptx> encode_weight_matrix_complex(Inference& inf,
                                              const std::vector<std::vector<double>>& W_re,
                                              const std::vector<std::vector<double>>& W_im,
                                              int d_in, int d_out,
                                              int target_level = 0,
                                              cudaStream_t stream = nullptr);

Ptx encode_bias_vector(Inference& inf, const std::vector<double>& b,
                        int d_in, int d_out, bool fill = true);

Ptx encode_bias_vector(Inference& inf, const std::vector<double>& b,
                        int d_in, int d_out, bool fill,
                        cudaStream_t stream);

// Complex bias (b_re + i*b_im) — cachemir-only; for the fused kv projection.
Ptx encode_bias_vector_complex(Inference& inf, const std::vector<double>& b_re,
                               const std::vector<double>& b_im, int d_in, int d_out,
                               bool fill = true, cudaStream_t stream = nullptr);

// Output-row pack (S4) — cachemir-only; pairs a single matrix's output blocks into complex.
std::vector<Ptx> encode_weight_matrix_outputpack(Inference& inf,
                                                 const std::vector<std::vector<double>>& W,
                                                 int d_in, int d_out, int target_level = 0,
                                                 cudaStream_t stream = nullptr);
PackedCtx linear_outputpack(Inference& inf, const PackedCtx& x,
                            const std::string& wname, int d_in, int d_out);

std::vector<Ptx> load_weight_txt(Inference& inf, const std::string& path,
                                  int d_in, int d_out,
                                  int target_level = 0);


std::vector<int32_t> compute_gpt2_rot_indices(
    const Packing& packing,
    int slots, int hidDim, int ffDim, int numHeads);

Inference make_gpt2_inference(InferenceOptions opts = {});

Inference make_gpt2_inference(const config_loader::ModelConfig& model,
                              CKKSContextOptions ckks = {});

PackedCtx transformer_block(Inference& inf, PackedCtx& x);

bool begin_subgraph_capture(Inference& inf, int capture_b);
void end_subgraph_capture(Inference& inf, int capture_b);

PackedCtx pack_tokens(Inference& inf, const std::vector<std::vector<double>>& embeddings,
                      int target_level = 0);
std::vector<std::vector<double>> unpack_tokens(Inference& inf, const PackedCtx& pc, int T);

PackedCtx encode_token_input(Inference& inf, const std::vector<double>& x_real);

std::vector<double> decode_token_output(Inference& inf, const PackedCtx& pc);

// Batched counterpart: decode T tokens packed in one ct (stride layout
// slot[i*t + tok]). Returns [T][d_real]. For T=1 it equals decode_token_output.
std::vector<std::vector<double>> decode_tokens_output(Inference& inf,
                                                      const PackedCtx& pc, int T);


struct EncodedBlock;  // fwd decl; cached_tiles holds the K pre-encoded vocab tiles
struct CutMaxConfig;  // fwd decl (cutmax.h); gpt2_cutmax_feedback takes it by ref

// `plan` (block n_blocks+1) is stamped on each vocab tile's EncodedBlock so run_cached_blocks'
// per-tile install keeps it live (an empty/invalid plan -> eager lm_head, backward compatible).
std::vector<PackedCtx> gpt2_lm_head(Inference& inf, const PackedCtx& x,
                                    const weight_loader::WeightStore& store,
                                    int vocab, int W_tile,
                                    std::vector<EncodedBlock>* cached_tiles = nullptr,
                                    const BootstrapPlan& plan = {});

std::vector<double> decode_lm_head_logits(Inference& inf,
                                          const std::vector<PackedCtx>& tiles,
                                          int vocab, int W_tile);


void gpt2_add_positional(Inference& inf, PackedCtx& h,
                         const weight_loader::WeightStore& store, int position);

// plan14: strict-tail encode level for fb_tile_k (weight_levels; the planned
// Z arrives deeper than the bts output and strict mode forbids the relevel)
void gpt2_prepare_feedback_weights(Inference& inf,
                                   const weight_loader::WeightStore& store,
                                   int vocab, int W_tile, bool packed_z,
                                   std::vector<EncodedBlock>& cached_tiles,
                                   const BootstrapPlan* plan14 = nullptr);

// Math-only: fb tiles must already be installed (encode lives in
// gpt2_prepare_feedback_weights; residency in gpt2_cutmax_feedback's ops).
PackedCtx gpt2_feedback_embed(Inference& inf,
                              const std::vector<PackedCtx>& z_tiles,
                              int vocab, int W_tile);

// Generation tail as model ops: cutmax argmax (block 13) + codebook feedback
// embed + wpe + entry bts (block 14) through run_ops, so the codebook rides
// the block residency discipline (acquire overlaps cutmax compute; release
// block_syncs before evict). position < 0 = argmax only (last token).
// z_out receives the one-hot Z tiles for validation decrypts.
// plan13/plan14 (nullable): strict tail plans (cutmax = block 13, feedback =
// block 14) installed per-op; capture writes the block_13/14 subgraphs when
// FHE_GRAPH_DIR is set (token 0).
PackedCtx gpt2_cutmax_feedback(Inference& inf,
                               const std::vector<PackedCtx>& tiles,
                               const weight_loader::WeightStore& store,
                               int vocab, int W_tile,
                               const CutMaxConfig& cmc,
                               EncodedBlock* fb_blk, int position,
                               std::vector<PackedCtx>* z_out,
                               double* argmax_s = nullptr,
                               const BootstrapPlan* plan13 = nullptr,
                               const BootstrapPlan* plan14 = nullptr);

// InferenceMode lives in inference.h (it is a property of the context now).
InferenceMode parse_inference_mode(const std::string& s);  // sync|threaded|prefetch|cached

enum class PrefillMode { SingleShot, Chunk };

PackedCtx gpt2_prefill(Inference& inf, PackedCtx x,
                       const weight_loader::WeightStore& store,
                       const config_loader::ParsedConfigs& parsed_configs,
                       const BlockPlans& plans,
                       int n_blocks, PrefillMode mode = PrefillMode::SingleShot);

// Pack T token embeddings into one CachemirFilling input ciphertext for prefill.
PackedCtx encode_prefill_input(Inference& inf,
                               const std::vector<std::vector<double>>& embeddings);
