#pragma once

#include "inference.h"
#include "nonlinear.h"
#include <string>
#include <utility>

namespace cachemir {

void      prepare_mha_masks(Inference& inf);
void      cache_k_push(Inference& inf, const PackedCtx& key);
PackedCtx qkt(Inference& inf, const PackedCtx& query);
PackedCtx head_reduce_sum(Inference& inf, const PackedCtx& x, double s0_expected = 0.0);
PackedCtx attention_softmax_thor(Inference& inf, const PackedCtx& scores, const std::string& cfg_name);
void      prepare_vcache(Inference& inf);
void      cache_v_push(Inference& inf, const PackedCtx& value);
void      cache_kv_push(Inference& inf, const PackedCtx& key, const PackedCtx& value);
void      cache_kv_push_packed(Inference& inf, const PackedCtx& kv_packed);
PackedCtx softmax_v(Inference& inf, const PackedCtx& softmax_scores);
// complex attention
PackedCtx prepare_pushed_k(Inference& inf, const PackedCtx& key);
PackedCtx pack_k_groups_complex(Inference& inf, const PackedCtx& K_even, const PackedCtx& K_odd);
PackedCtx complex_qkt(Inference& inf, const PackedCtx& query);
void      prepare_complex_k_cache(Inference& inf);
void      prepare_complex_v_cache(Inference& inf);
void      cache_kv_push_packed_complex(Inference& inf, const PackedCtx& kv_packed);
PackedCtx complex_softmax_v(Inference& inf, const PackedCtx& scores);

}  // namespace cachemir
