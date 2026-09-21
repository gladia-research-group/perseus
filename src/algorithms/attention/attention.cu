#include "attention.h"
#include "inference.h"
#include "packing/cachemir/cachemir_attention.h"
#include "packing/cachemir/cachemir_attention_utils.h"
#include "packing/cachemir_filling/cachemir_filling_attention.h"
#include "packing/cachemir_filling/cachemir_filling_delta_attention.h"

#include <stdexcept>
#include <utility>

void prepare_mha_masks(Inference& inf) {
    if (is_cachemir(inf.packing)) {
        if (inf.complex) cachemir::prepare_complex_k_cache(inf);
        else                  cachemir::prepare_mha_masks(inf);
        return;
    }
    if (is_cachemir_filling(inf.packing)) { cachemir_filling::prepare_mha_masks(inf); return; }
    throw std::runtime_error("prepare_mha_masks: unsupported packing");
}

void prepare_vcache(Inference& inf) {
    if (is_cachemir(inf.packing)) {
        if (inf.complex) cachemir::prepare_complex_v_cache(inf);
        else                  cachemir::prepare_vcache(inf);
        return;
    }
    if (is_cachemir_filling(inf.packing)) { cachemir_filling::prepare_vcache(inf); return; }
    throw std::runtime_error("prepare_vcache: unsupported packing");
}

void cache_k_push(Inference& inf, const PackedCtx& key) {
    if (is_cachemir(key.packing)) {
        if (inf.complex) throw std::runtime_error("cache_k_push: complex uses fused cache_kv_push_packed");
        cachemir::cache_k_push(inf, key);
        return;
    }
    if (is_cachemir_filling(key.packing)) { cachemir_filling::cache_k_push(inf, key); return; }
    throw std::runtime_error("cache_k_push: unsupported packing");
}

void cache_v_push(Inference& inf, const PackedCtx& value) {
    if (is_cachemir(value.packing)) {
        if (inf.complex) throw std::runtime_error("cache_v_push: complex uses fused cache_kv_push_packed");
        cachemir::cache_v_push(inf, value);
        return;
    }
    if (is_cachemir_filling(value.packing)) { cachemir_filling::cache_v_push(inf, value); return; }
    throw std::runtime_error("cache_v_push: unsupported packing");
}

void cache_kv_push(Inference& inf, const PackedCtx& key, const PackedCtx& value) {
    if (is_cachemir(key.packing)) {
        if (inf.complex) throw std::runtime_error("cache_kv_push: complex uses fused cache_kv_push_packed");
        cachemir::cache_kv_push(inf, key, value);   // Mode-A merged push → flat caches
        return;
    }
    cache_k_push(inf, key);
    cache_v_push(inf, value);
}

void cache_kv_push_packed(Inference& inf, const PackedCtx& kv_packed) {
    if (is_cachemir(kv_packed.packing)) {
        if (inf.complex) cachemir::cache_kv_push_packed_complex(inf, kv_packed);  // fused P → complex caches
        else             cachemir::cache_kv_push_packed(inf, kv_packed);          // fused P → flat caches
        return;
    }
    throw std::runtime_error("cache_kv_push_packed: cachemir-only");
}

std::vector<PackedCtx> qkt(Inference& inf, const PackedCtx& query) {
    if (is_cachemir(query.packing))
        return { inf.complex ? cachemir::complex_qkt(inf, query) : cachemir::qkt(inf, query) };
    if (is_cachemir_filling(query.packing))
        return cachemir_filling::delta_block_enabled()
                   ? cachemir_filling::qkt_delta(inf, query)
                   : cachemir_filling::qkt(inf, query);
    throw std::runtime_error("qkt: unsupported packing");
}

PackedCtx head_reduce_sum(Inference& inf, const PackedCtx& x) {
    if (is_cachemir(x.packing)) return cachemir::head_reduce_sum(inf, x);
    throw std::runtime_error("head_reduce_sum: unsupported packing");
}

std::vector<PackedCtx> attention_softmax_thor(
        Inference& inf, std::vector<PackedCtx> scores, const std::string& cfg_name) {
    const Packing& p = scores.at(0).packing;
    if (is_cachemir(p))         return { cachemir::attention_softmax_thor(inf, scores[0], cfg_name) };
    if (is_cachemir_filling(p))
        return cachemir_filling::delta_block_enabled()
                   ? cachemir_filling::attention_softmax_thor_delta(inf, std::move(scores), cfg_name)
                   : cachemir_filling::attention_softmax_thor(inf, std::move(scores), cfg_name);
    throw std::runtime_error("attention_softmax_thor: unsupported packing");
}

PackedCtx mha_attn_token_pair(Inference& inf, PackedCtx& q_cplx) {
    if (!is_cachemir_filling(inf.packing) || !inf.token_pair)
        throw std::runtime_error("mha_attn_token_pair: filling token-pair only");
    return cachemir_filling::delta_block_enabled()
               ? cachemir_filling::mha_attn_token_pair_delta(inf, q_cplx)
               : cachemir_filling::mha_attn_token_pair(inf, q_cplx);
}

PackedCtx softmax_v(Inference& inf, std::vector<PackedCtx> softmax_scores) {
    const Packing& p = softmax_scores.at(0).packing;
    if (is_cachemir(p))
        return inf.complex ? cachemir::complex_softmax_v(inf, softmax_scores[0])
                                : cachemir::softmax_v(inf, softmax_scores[0]);
    if (is_cachemir_filling(p)) return cachemir_filling::softmax_v(inf, std::move(softmax_scores));
    throw std::runtime_error("softmax_v: unsupported packing");
}

std::vector<std::vector<double>> rearrange_qkv_weights(
        const Inference& inf, const std::vector<std::vector<double>>& W, int H) {
    if (is_cachemir(inf.packing) || is_cachemir_filling(inf.packing))
        return cachemir::rearrange_qkv_weights(W, H);
    throw std::runtime_error("rearrange_qkv_weights: unsupported packing");
}

std::vector<double> rearrange_qkv_biases(
        const Inference& inf, const std::vector<double>& b, int H) {
    if (is_cachemir(inf.packing) || is_cachemir_filling(inf.packing))
        return cachemir::rearrange_qkv_biases(b, H);
    throw std::runtime_error("rearrange_qkv_biases: unsupported packing");
}

std::vector<std::vector<double>> rearrange_wo_weights(
        const Inference& inf, const std::vector<std::vector<double>>& W, int H) {
    if (is_cachemir(inf.packing) || is_cachemir_filling(inf.packing))
        return cachemir::rearrange_wo_weights(W, H);
    throw std::runtime_error("rearrange_wo_weights: unsupported packing");
}
