#pragma once

#include "inference.h"
#include "nonlinear.h"
#include <string>
#include <vector>

// Public dispatchers

void      prepare_mha_masks(Inference& inf);
void      cache_k_push(Inference& inf, const PackedCtx& key);
PackedCtx head_reduce_sum(Inference& inf, const PackedCtx& x);
void      prepare_vcache(Inference& inf);
void      cache_v_push(Inference& inf, const PackedCtx& value);
void      cache_v_push_pair(Inference& inf, const PackedCtx& va, const PackedCtx& vb);
void      cache_kv_push(Inference& inf, const PackedCtx& key, const PackedCtx& value);
void      cache_kv_push_packed(Inference& inf, const PackedCtx& kv_packed);   // 2b: pre-packed K+iV

std::vector<PackedCtx> qkt(Inference& inf, const PackedCtx& query);
std::vector<PackedCtx> attention_softmax_thor(
    Inference& inf, std::vector<PackedCtx> scores, const std::string& cfg_name);
PackedCtx softmax_v(Inference& inf, std::vector<PackedCtx> softmax_scores);

PackedCtx mha_attn_token_pair(Inference& inf, PackedCtx& q_cplx);

std::vector<PackedCtx> bi_attention(Inference& inf,
                                    std::vector<PackedCtx> qs,
                                    std::vector<PackedCtx> ks,
                                    std::vector<PackedCtx> vs,
                                    const std::vector<int>& ns);

std::vector<std::vector<double>> rearrange_qkv_weights(
    const Inference& inf, const std::vector<std::vector<double>>& W, int H);

std::vector<double> rearrange_qkv_biases(
    const Inference& inf, const std::vector<double>& b, int H);

std::vector<std::vector<double>> rearrange_wo_weights(
    const Inference& inf, const std::vector<std::vector<double>>& W, int H);
