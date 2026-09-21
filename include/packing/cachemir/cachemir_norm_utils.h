#pragma once

#include "inference.h"
#include "packing/packed_ctx.h"

#include <string>
#include <vector>

// Cachemir-packing reductions and slot helpers used by norm / LN affine.

namespace cachemir {

// LN centering mask is an active-token selector scaled by 0.5*c_eff. c_eff is
// per-block (center_scale / center_scale_sq[pos]) ⇒ block-scoped tag; pos < 0 means
// position-invariant (no center_scale_sq array). Shared by norm.cu and
// generate_decode_masks so the cached value never drifts.
// Prefix-taking form is the primitive; the Inference& form forwards to it —
// see the note on score_mask_tag: both MUST produce byte-identical tags.
inline std::string ln_center_mask_tag(const std::string& prefix, const std::string& cfg_name,
                                      int pos) {
    return Inference::scoped_in(prefix, "ln.center." + cfg_name +
                                (pos < 0 ? std::string(".base") : ".p" + std::to_string(pos)));
}
inline std::string ln_center_mask_tag(const Inference& inf, const std::string& cfg_name,
                                      int pos) {
    return ln_center_mask_tag(inf.block_prefix, cfg_name, pos);
}

PackedCtx compute_per_token_sum(Inference& inf, const PackedCtx& x);

Ptx encode_ln_center_mask(Inference& inf, const PackedCtx& x, const std::string& cfg_name,
                          int d, int t, double scale, int center_pos);
void add_layernorm_epsilon(Inference& inf, PackedCtx& var_scaled,
                           double epsilon, double center_scale_sq,
                           const std::vector<double>& per_pos_scale_sq);

PackedCtx compute_variance_interleaved(Inference& inf, const PackedCtx& centered_x);
PackedCtx compute_variance_interleaved(Inference& inf, const PackedCtx& x, PackedCtx neg_mean);

// No-op: single-token packing has no inactive token lanes (n_tok == 1 and the
// variance is broadcast across every slot by the full all-reduce), so there is
// nothing to floor. Present to mirror the diagonal primitive for dispatch.
PackedCtx floor_inactive_token_lanes(Inference& inf, const PackedCtx& var_scaled,
                                     double fill);

// Pack a length-C_real per-feature vector into single-token cachemir layout
// (slot k*t = v[k] for k in [0, C_real)), zero elsewhere.
std::vector<double> pack_per_feature_vec(int slots, const std::vector<double>& v,
                                         int d_pad, int C_real);

// Pack a length-d_pad padded vector into single-token cachemir layout and encrypt.
// C_real defaults to d_pad (all features carry data); pass a smaller value to
// drop trailing padding lanes.
PackedCtx encrypt_single_token(Inference& inf, const std::vector<double>& x_pad);
PackedCtx encrypt_single_token(Inference& inf, const std::vector<double>& x_pad,
                               int d_pad, int C_real);

}  // namespace cachemir
