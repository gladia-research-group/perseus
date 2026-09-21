#include "packing/diagonal/diagonal_norm_utils.h"
#include "packing/diagonal/diagonal_linear_utils.h"
#include "inference.h"

#include <algorithm>
#include <cmath>

namespace diagonal {

static PackedCtx feature_reduce_sum(Inference& inf, const PackedCtx& x_in) {
    const int S = inf.slots;
    const int t = S / inf.size.hidDim;

    PackedCtx acc = inf.fhe->clone(x_in);
    for (int s = t; s < S; s *= 2) {
        PackedCtx tmp = inf.fhe->rotate(acc, dg_rot(inf, s));
        inf.fhe->inplace_add(acc, tmp);
    }
    return acc;
}

PackedCtx compute_per_token_sum(Inference& inf, const PackedCtx& x_in) {
    return feature_reduce_sum(inf, x_in);
}

Ptx encode_ln_center_mask(Inference& inf, const PackedCtx& x, int d, int t,
                          double scale, const std::vector<double>& per_pos_scale_sq) {
    if (!is_cachemir_filling(x.packing) || per_pos_scale_sq.empty())
        return inf.encode_active_token_mask_at(d, t, x, scale);

    std::vector<double> mask(inf.slots, 0.0);
    const auto active = stride_slots(x.packing, inf.slots, d, t, 0);
    const int n = static_cast<int>(per_pos_scale_sq.size());
    const int pos0 = inf.output.capture_t;
    for (int slot : active) {
        const int tok = slot % t;
        if (tok < inf.n_tok) {
            const int pos = std::max(0, std::min(pos0 + tok, n - 1));
            mask[slot] = 0.5 * std::sqrt(per_pos_scale_sq[pos]);
        }
    }
    return inf.encode_at(mask, x);
}

void add_layernorm_epsilon(Inference& inf, PackedCtx& var_scaled,
                           double epsilon, double center_scale_sq,
                           const std::vector<double>& per_pos_scale_sq) {
    if (!is_cachemir_filling(var_scaled.packing) || per_pos_scale_sq.empty()) {
        inf.fhe->inplace_add(var_scaled, epsilon * center_scale_sq);
        return;
    }

    const int t = inf.slots / inf.size.hidDim;
    const int n = static_cast<int>(per_pos_scale_sq.size());
    const int pos0 = inf.output.capture_t;
    std::vector<double> eps_vec(inf.slots, 0.0);
    for (int s = 0; s < inf.slots; ++s) {
        const int tok = s % t;
        if (tok < inf.n_tok) {
            const int pos = std::max(0, std::min(pos0 + tok, n - 1));
            eps_vec[s] = epsilon * per_pos_scale_sq[pos];
        }
    }
    Ptx eps_pt = inf.encode_at(eps_vec, var_scaled);
    inf.fhe->inplace_add(var_scaled, eps_pt);
}

PackedCtx compute_per_token_var(Inference& inf, const PackedCtx& centered_x_in) {
    const int rD = inf.size.getRealHidDim();

    PackedCtx var = inf.fhe->square(centered_x_in);
    var = feature_reduce_sum(inf, var);
    inf.fhe->inplace_mult(var, 1.0 / (double)rD);  // biased variance, matches nn.LayerNorm
    return var;
}

PackedCtx floor_inactive_token_lanes(Inference& inf, const PackedCtx& var_scaled,
                                     double fill) {
    const std::string tag = "ln.floor.nt" + std::to_string(inf.n_tok) +
                            ".f" + std::to_string(fill);
    Ptx floor_pt = inf.encode_inactive_token_mask_at_cached(tag, var_scaled, fill);
    return inf.fhe->add(var_scaled, floor_pt);
}

std::vector<double> pack_per_feature_vec(int slots, const std::vector<double>& v,
                                         int d_pad, int C_real) {
    const int t = slots / d_pad;
    std::vector<double> out(slots, 0.0);
    for (int k = 0; k < C_real; ++k)
        for (int tok = 0; tok < t; ++tok)
            out[k * t + tok] = v[k];
    return out;
}

}  // namespace diagonal
