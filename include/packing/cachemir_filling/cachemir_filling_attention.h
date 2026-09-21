#pragma once

#include "inference.h"

#include <string>
#include <vector>

namespace cachemir_filling {

struct CfEntry { int g; int delta; int Lg; bool current; bool bd = false; };
std::vector<CfEntry> cf_score_schedule(Inference& inf);

std::vector<CfEntry> cf_score_schedule(Inference& inf, int K);

void prepare_mha_masks(Inference& inf);
void cache_k_push(Inference& inf, const PackedCtx& key);

std::vector<PackedCtx> qkt(Inference& inf, const PackedCtx& query);

std::vector<PackedCtx> attention_softmax_thor(
    Inference& inf, std::vector<PackedCtx> scores, const std::string& cfg_name,
    const std::string& stage_prefix = "cf.stg.");

void prepare_vcache(Inference& inf);
void cache_v_push(Inference& inf, const PackedCtx& value);

// Token-pair B-half (imaginary) K/V push
void cache_k_push_imag(Inference& inf, const PackedCtx& key);
void cache_v_push_imag(Inference& inf, const PackedCtx& value);

void mha_qkv_token_pair(Inference& inf, PackedCtx& x);
PackedCtx mha_attn_token_pair(Inference& inf, PackedCtx& q_cplx);

PackedCtx softmax_v(Inference& inf, std::vector<PackedCtx> probs,
                    const std::string& stage_prefix = "cf.stg.");
PackedCtx softmax_v_groups(Inference& inf, std::vector<PackedCtx> probs,
                           const std::vector<PackedCtx>& vgroups, int K,
                           const std::string& stage_prefix = "cf.stg.");

bool mask_values_for_tag(const Inference& inf, const SoftmaxConfig* cfg,
                         const std::string& tag, std::vector<double>& out);

}  // namespace cachemir_filling
