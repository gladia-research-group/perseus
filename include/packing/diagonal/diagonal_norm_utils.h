#pragma once

#include "inference.h"
#include "packing/packed_ctx.h"

#include <vector>

// Diagonal-packing reductions and slot helpers used by norm / LN affine.
//
// Multi-token layout: feature i, token tok at slot i*t + tok (t = slots/hidDim).
// Reductions run over the feature axis only (stride t) so the per-token lanes
// stay independent — this is the prefill/batched counterpart of the cachemir
// reductions, which collapse every slot because they pack a single token.

namespace diagonal {

// Per-token sum across the feature (hidDim) axis: an all-reduce by feature
// stride (rotate by t, 2t, 4t, …, S/2 and add). Afterwards every feature block
// holds, at lane tok, the sum over features of token tok. Caller scales by 1/rD
// (e.g. via a stride mask) to obtain the mean.
PackedCtx compute_per_token_sum(Inference& inf, const PackedCtx& x);

Ptx encode_ln_center_mask(Inference& inf, const PackedCtx& x, int d, int t,
                          double scale, const std::vector<double>& per_pos_scale_sq);
void add_layernorm_epsilon(Inference& inf, PackedCtx& var_scaled,
                           double epsilon, double center_scale_sq,
                           const std::vector<double>& per_pos_scale_sq);

// Per-token biased variance: square the centered input, reduce over the feature
// axis (same stride-t all-reduce as the sum — NOT a full-slot sum, which would
// mix tokens), then multiply by 1/rD.
PackedCtx compute_per_token_var(Inference& inf, const PackedCtx& centered_x);

// Lift the variance in inactive (padding) token lanes (tok >= n_tok) to `fill`
// before the inverse-sqrt iteration. Those lanes carry var = 0 otherwise, and
// Taylor/Newton inv_sqrt would diverge on them; the real token lanes are left
// untouched (the mask is zero there). Counterpart of the cachemir no-op.
PackedCtx floor_inactive_token_lanes(Inference& inf, const PackedCtx& var_scaled,
                                     double fill);

// Pack a length-C_real per-feature vector (e.g. LN gamma/beta) into diagonal
// layout, broadcasting v[k] across all t token lanes of feature block k:
// out[k*t + tok] = v[k] for k in [0, C_real), tok in [0, t); zero elsewhere.
std::vector<double> pack_per_feature_vec(int slots, const std::vector<double>& v,
                                         int d_pad, int C_real);

}  // namespace diagonal
