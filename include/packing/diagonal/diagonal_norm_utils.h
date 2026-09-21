#pragma once

#include "inference.h"
#include "packing/packed_ctx.h"

#include <vector>

namespace diagonal {

PackedCtx compute_per_token_sum(Inference& inf, const PackedCtx& x);

Ptx encode_ln_center_mask(Inference& inf, const PackedCtx& x, int d, int t,
                          double scale, const std::vector<double>& per_pos_scale_sq);
void add_layernorm_epsilon(Inference& inf, PackedCtx& var_scaled,
                           double epsilon, double center_scale_sq,
                           const std::vector<double>& per_pos_scale_sq);

PackedCtx compute_per_token_var(Inference& inf, const PackedCtx& centered_x);

PackedCtx floor_inactive_token_lanes(Inference& inf, const PackedCtx& var_scaled,
                                     double fill);

std::vector<double> pack_per_feature_vec(int slots, const std::vector<double>& v,
                                         int d_pad, int C_real);

}  // namespace diagonal
