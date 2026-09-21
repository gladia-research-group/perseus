#pragma once

#include "inference.h"
#include "nonlinear.h"
#include <string>
#include <utility>

namespace cachemir {

void      prepare_mha_masks(Inference& inf);
void      cache_k_push(Inference& inf, const PackedCtx& key);
PackedCtx qkt(Inference& inf, const PackedCtx& query);
PackedCtx head_reduce_sum(Inference& inf, const PackedCtx& x);
PackedCtx attention_softmax_thor(Inference& inf, const PackedCtx& scores, const std::string& cfg_name);
void      prepare_vcache(Inference& inf);
void      cache_v_push(Inference& inf, const PackedCtx& value);
void      cache_v_push_pair(Inference& inf, const PackedCtx& va, const PackedCtx& vb);   // 2 tokens, 1 bts (complex payload; else 2 singles)
void      cache_kv_push(Inference& inf, const PackedCtx& key, const PackedCtx& value);
void      cache_kv_push_packed(Inference& inf, const PackedCtx& kv_packed);   // 2b: pre-packed K+iV
PackedCtx softmax_v(Inference& inf, const PackedCtx& softmax_scores);
// complex attention
PackedCtx prepare_pushed_k(Inference& inf, const PackedCtx& key);   // shared per-token K key-prep (real cache)
PackedCtx pack_k_groups_complex(Inference& inf, const PackedCtx& K_even, const PackedCtx& K_odd);
PackedCtx complex_qkt(Inference& inf, const PackedCtx& query);
void      prepare_complex_k_cache(Inference& inf);                  // reset complex K cache + pending
void      prepare_complex_v_cache(Inference& inf);                  // reset complex V cache (d_head/2 buckets)
void      cache_kv_push_packed_complex(Inference& inf, const PackedCtx& kv_packed);            // fused "kv" linear P=K+iV → complex K/V caches (the ONLY complex push)
PackedCtx complex_softmax_v(Inference& inf, const PackedCtx& scores);   // pair-reading P·V over the complex V cache (½ storage); emits the real 2·Re·(½ head-select) output

}  // namespace cachemir
