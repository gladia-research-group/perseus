#pragma once

#include "inference.h"

#include <string>
#include <vector>

// δ-block single-ct score layout for CachemirFilling prefill attention (task #10,
// HANDOFF_delta_block_layout.txt VALIDATION ADDENDUM + scripts/delta_block_sim.py).
// score[h,i,Δ] (Δ = absolute causal key offset, δ-MAJOR) lives at slot
// Δ%64·tH + h·t + i of big ct Δ/64, so the THOR softmax chain (cheb exp + bts +
// squares + GS recips) runs ONCE per big ct instead of once per schedule entry.
// The qkt channel all-reduce already replicates each entry's reduced score into
// every tH-block, so the scatter is the existing qkt mask retargeted to block Δ
// (zero extra ops); the final refine folds the per-entry extraction into its
// zscale mask, so softmax_v consumes the per-entry probs UNCHANGED.
//
// Implementations split cachemir-style:
//   - real arm (qkt_delta / softmax core)      → cachemir_filling_delta_attention.cu
//   - token-pair fused (mha_attn_token_pair_delta) → cachemir_filling_delta_complex_attention.cu

namespace cachemir_filling {

// One union-schedule entry as the δ-block layout sees it: within-group key offset,
// absolute-diagonal target block (B-view), causal flavor for the active pattern,
// and whether this half consumes it (token-pair A half skips A-dead entries).
// Lg >= 0 marks a BIDIRECTIONAL entry and carries its group's key-lane bound.
struct DeltaEntry { int delta; int block; bool current; bool alive; int Lg = -1; };

// Per-half geometry for the shared softmax core. kc(i) = P + i + 1. A slot (b,h,i)
// of big ct ci is live iff 0 <= ci*NBLK + b - block_shift <= P + i (block_shift = 0
// for the real arm / TP B half; = t for the TP A half, whose diagonals sit t blocks
// up in the B-view packing — no realignment rotation needed, only mask targeting).
// BIDIRECTIONAL view (bd=true): P = total key count K (uniform kc), block_shift =
// Gtot*t - 1, entry block = (Gtot-1-g)*t + delta + (t-1) — query-chunk-INDEPENDENT
// (the group index cancels: slot (b,h,i) of ct ci is live iff key = i - D in [0, K),
// D = ci*NBLK + b - block_shift), so one mask set serves every chunk.
struct DeltaView {
    int P           = 0;
    int n_cur       = 0;
    int block_shift = 0;
    bool bd         = false;
    std::vector<DeltaEntry> entries;   // union order (== cf_score_schedule order)
};

// QKᵀ into δ-block cts: identical per-entry matmul/reduce as qkt(); the per-entry
// mask targets block Δ=(G−g)·t+δ and entries accumulate. Returns ceil((P+t)/64) cts.
std::vector<PackedCtx> qkt_delta(Inference& inf, const PackedCtx& query);
// Cache-free twin: K groups + key count as arguments (bidirectional ViT attention).
std::vector<PackedCtx> qkt_delta_groups(Inference& inf, const PackedCtx& query,
                                        const std::vector<PackedCtx>& kgroups, int K);

// Bidirectional DeltaView for an explicit key count (uniform kc = K, query-chunk-
// independent block targeting). n_cur is read from inf.n_tok.
DeltaView bd_delta_view(Inference& inf, int K);

// THOR softmax over the δ-block cts, one chain per ct; returns PER-ENTRY probs in
// schedule order (extraction folded into the final refine's zscale) — feed the
// unchanged softmax_v(). Real-arm wrapper builds the view from cf_score_schedule.
std::vector<PackedCtx> attention_softmax_thor_delta(
    Inference& inf, std::vector<PackedCtx> scores, const std::string& cfg_name,
    const std::string& stage_prefix = "cf.stg.");

std::vector<PackedCtx> attention_softmax_thor_delta_core(
    Inference& inf, std::vector<PackedCtx> scores, const std::string& cfg_name,
    const DeltaView& view, const std::string& stage_prefix);

// Token-pair fused δ-block MHA; same contract as mha_attn_token_pair (guarded
// is_cachemir_filling && inf.token_pair).
PackedCtx mha_attn_token_pair_delta(Inference& inf, PackedCtx& q_cplx);

// FHE_DELTA_BLOCK=1 routes the filling attention dispatchers (attention.cu qkt /
// attention_softmax_thor, mha.cu token-pair attn_core) to the δ-block arm.
// Default off — the shipping per-entry path is byte-identical when unset.
bool delta_block_enabled();

}  // namespace cachemir_filling
