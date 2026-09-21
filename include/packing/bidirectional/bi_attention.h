#pragma once

#include "inference.h"

#include <vector>

// Cache-free bidirectional attention (the ViT encoder path). The op takes q/k/v
// straight off the QKV linear and returns per-chunk context outputs — no KV cache
// (inf.cache / k_count bookkeeping), no push phase, no prefill→decode handoff.
// Both arms run the δ-block softmax layout (one THOR chain per big ct — the only
// bidirectional softmax arm; the per-entry path is a causal-prefill legacy).
//
// Implementations:
//   - delta_bi_attention.cu   → real arm (one real ct per chunk)
//   - complex_bi_attention.cu → token-pair arm (chunk A = Re, chunk B = Im of ONE
//                               packed ct; K/V half-extractions share a paired
//                               bootstrap, one packed QKᵀ + softmax_v pass)
// Callers use the un-prefixed bi_attention dispatcher (attention.h).

namespace bidirectional {

std::vector<PackedCtx> delta_bi_attention(Inference& inf,
                                          std::vector<PackedCtx> qs,
                                          std::vector<PackedCtx> ks,
                                          std::vector<PackedCtx> vs,
                                          const std::vector<int>& ns);

PackedCtx complex_bi_attention(Inference& inf, PackedCtx q, PackedCtx k, PackedCtx v,
                               int nA, int nB);

}  // namespace bidirectional
