#pragma once

#include "inference.h"

#include <string>
#include <vector>

// Disjoint batched causal-prefill attention for the CachemirFilling packing.
// Mirrors the cachemir attention API (cachemir_attention.h), but the N×N score
// matrix is represented as n_tok ciphertexts — one per causal key-offset
// δ = i−j — so qkt/softmax return vectors of PackedCtx. cachemir-ordered
// features (via the free-K weight rearrange) let the tH channel reductions and
// the K/V caches match cachemir's layout.
//
// Implementations are split cachemir-style:
//   - qkt / attention_softmax_thor / softmax_v  → algorithms/attention/cachemir_filling/
//   - prepare_*/cache_*_push                     → packing/cachemir_filling/cachemir_filling_kv_cache.cu

namespace cachemir_filling {

// One score-schedule entry: key group g, key offset delta, group row-length Lg, current-chunk flag,
// bidirectional flag. The schedule is shape-fixed (depends on k_count/n_tok only) — shared by
// qkt / softmax / softmax_v and the token-pair fused attention.
// inf.bidirectional (ViT): every group is enumerated f-style over the full active delta range
// [-(Lg-1), n_cur-1] with uniform kc = k_count; requires ALL K/V groups pushed before attention
// (two-phase driver) and is not combined with token_pair.
struct CfEntry { int g; int delta; int Lg; bool current; bool bd = false; };
std::vector<CfEntry> cf_score_schedule(Inference& inf);
// Explicit-key-count twin: K replaces inf.k_count() (cache-free callers — the
// bidirectional ViT attention — pass the chunk-sum directly).
std::vector<CfEntry> cf_score_schedule(Inference& inf, int K);

void prepare_mha_masks(Inference& inf);
void cache_k_push(Inference& inf, const PackedCtx& key);

// QKᵀ. scores[δ] holds score[h,i,i-δ] = (1/√d_head) Σ_c q[h,c,i]·k[h,c,i-δ] at
// the head-anchor slot h*t+i (c=0 block), causal (i ≥ δ), scaled.
std::vector<PackedCtx> qkt(Inference& inf, const PackedCtx& query);

// Causal softmax over keys: Σ_δ denominator, causal/active masking, per-row
// Goldschmidt init (causal row length i+1). Reuses the THOR exp/Goldschmidt core.
// stage_prefix = the host-staging key namespace this call's entries live in (StagedEntries keys
// are index-based; the token-pair fused MHA runs the A and B halves in distinct namespaces).
std::vector<PackedCtx> attention_softmax_thor(
    Inference& inf, std::vector<PackedCtx> scores, const std::string& cfg_name,
    const std::string& stage_prefix = "cf.stg.");

void prepare_vcache(Inference& inf);
void cache_v_push(Inference& inf, const PackedCtx& value);

// Token-pair B-half (imaginary) K/V push: complex -0.25i mask (tag .im) nets Im(K)/Im(V). Called
// only from the token-pair fused MHA (guarded is_cachemir_filling && inf.token_pair).
void cache_k_push_imag(Inference& inf, const PackedCtx& key);
void cache_v_push_imag(Inference& inf, const PackedCtx& value);

// Token-pair fused MHA (guarded is_cachemir_filling && inf.token_pair). qkv runs the complex QKV
// linear once and DEFERS the push (K,V -> tp.k/tp.v). attn pushes both halves, then runs the
// matmuls COMPLEX-PACKED over the union (B-view) schedule: one qkt pass (Re = A rows, Im = B rows;
// per-entry two-sided causal mask p·res + q·conj(res)), conj_split per entry, THOR softmax per
// half (nonlinear stays 2x; A runs under its pre-B k_count view), pair_pack the probs, and one
// softmax_v pass whose output is born packed (A + i·B). Value-exact vs the retired per-half
// split (qkt_imag) at ~40% fewer entry-matmuls.
void mha_qkv_token_pair(Inference& inf, PackedCtx& x);
PackedCtx mha_attn_token_pair(Inference& inf, PackedCtx& q_cplx);

// Causal P·V: out[h,c,i] = Σ_{δ≤i} p_δ[h,i]·v[h,c,i-δ], one cachemir-ordered ct.
PackedCtx softmax_v(Inference& inf, std::vector<PackedCtx> probs,
                    const std::string& stage_prefix = "cf.stg.");
// Cache-free twin: V groups + key count as arguments (bidirectional ViT attention).
PackedCtx softmax_v_groups(Inference& inf, std::vector<PackedCtx> probs,
                           const std::vector<PackedCtx>& vgroups, int K,
                           const std::string& stage_prefix = "cf.stg.");

// Rebuild the slot-vector of a filling attention mask from its UNSCOPED enc_cache tag
// (cf.qkt.m.* / cf.sm.{shift,amask,zscale,floor,rbeta,ralpha}.*). Single source of
// truth for the op bodies and the worker-side mask prefetch (gpt2_prefill). cfg is the
// block's softmax config — required for shift/rbeta/ralpha, ignored otherwise. Returns
// false when the tag doesn't parse as a filling attention mask.
bool mask_values_for_tag(const Inference& inf, const SoftmaxConfig* cfg,
                         const std::string& tag, std::vector<double>& out);

}  // namespace cachemir_filling
