#pragma once

#include "inference.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cachemir_filling {

std::vector<int32_t> compute_gpt2_rot_indices(int slots, int hidDim, int ffDim, int numHeads);

// Reset the K cache for a fresh prefill.
void prepare_mha_masks(Inference& inf);

void cache_k_push(Inference& inf, const PackedCtx& key);

std::vector<PackedCtx> qkt(Inference& inf, const PackedCtx& query);

std::vector<PackedCtx> attention_softmax_thor(
    Inference& inf, std::vector<PackedCtx> scores, const std::string& cfg_name);

// Reset the V cache for a fresh prefill.
void prepare_vcache(Inference& inf);

void cache_v_push(Inference& inf, const PackedCtx& value);

PackedCtx softmax_v(Inference& inf, std::vector<PackedCtx> probs);

// Token-pair prefill input encode: Re = tokens [0,t), Im = tokens [t,2t) at slot[i*t+tok].
PackedCtx encode_input_token_pair(Inference& inf,
                                  const std::vector<std::vector<double>>& embeddings,
                                  int target_level);

}  // namespace cachemir_filling
