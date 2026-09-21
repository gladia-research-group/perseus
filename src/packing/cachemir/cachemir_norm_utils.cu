#include "packing/cachemir/cachemir_norm_utils.h"
#include "inference.h"
#include "nonlinear.h"   // fused_ln_var_enabled()

#include <utility>

namespace cachemir {


PackedCtx compute_per_token_sum(Inference& inf, const PackedCtx& x_in) {
    const int S  = inf.slots;
    const int hD = inf.size.hidDim;
    const int t  = S / hD;

    PackedCtx mean = inf.fhe->clone(x_in);
    for (int i = t; i < S; i *= 2) {
        PackedCtx tmp = inf.fhe->rotate(mean, i);
        inf.fhe->inplace_add(mean, tmp);
    }
    // The ladder starts at gap=t, so this is a mod-t class sum (NOT an all-reduce):
    // t-periodic. Named stamp so a bootstrap placed here can route sparse.
    mean.tag = packtag::t_reduce_stride(packtag::PackTag::top(S), t);
    inf.fhe->tag_ct(mean.ct, mean.tag);
    return mean;
}

Ptx encode_ln_center_mask(Inference& inf, const PackedCtx& x, const std::string& cfg_name,
                          int d, int t, double scale, int center_pos) {
    return inf.encode_active_token_mask_at_cached(
        ln_center_mask_tag(inf, cfg_name, center_pos), d, t, x, scale);
}

void add_layernorm_epsilon(Inference& inf, PackedCtx& var_scaled,
                           double epsilon, double center_scale_sq,
                           const std::vector<double>& /*per_pos_scale_sq*/) {
    inf.fhe->inplace_add(var_scaled, epsilon * center_scale_sq);
}

PackedCtx compute_variance_interleaved(Inference& inf, const PackedCtx& centered_x_in) {
    const int S  = inf.slots;
    const int rD = inf.size.getRealHidDim();

    PackedCtx var;
    if (fused_ln_var_enabled()) {

        CKKSContext::AutoBtsSuppressScope no_auto(*inf.fhe);
        var = inf.fhe->square(centered_x_in);

        const uint32_t s_eff = inf.fhe->fold_slots_for(1);
        for (int gap = 1; gap < (int)s_eff; gap *= 2) {
            PackedCtx tmp = inf.fhe->rotate(var, gap);
            inf.fhe->inplace_add(var, tmp);
        }
        inf.fhe->fold_bootstrap(var.ct, s_eff, /*n_live=*/rD);
    } else {
        var = inf.fhe->square(centered_x_in);
        for (int gap = 1; gap < S; gap *= 2) {
            PackedCtx tmp = inf.fhe->rotate(var, gap);
            inf.fhe->inplace_add(var, tmp);
        }
        inf.fhe->inplace_mult(var, 1.0 / (double)rD);  // biased variance, matches nn.LayerNorm
    }

    var.tag = packtag::t_reduce_all(packtag::PackTag::top(S));

    inf.fhe->tag_ct(var.ct, var.tag);
    return var;
}

PackedCtx compute_variance_interleaved(Inference& inf, const PackedCtx& x_in, PackedCtx neg_mean) {
    assert_same_packing(x_in.packing, neg_mean.packing);
    return compute_variance_interleaved(inf, inf.fhe->add(x_in, neg_mean));
}

PackedCtx floor_inactive_token_lanes(Inference& /*inf*/, const PackedCtx& var_scaled,
                                     double /*fill*/) {
    return var_scaled;
}

std::vector<double> pack_per_feature_vec(int slots, const std::vector<double>& v,
                                         int d_pad, int C_real) {
    const int t = slots / d_pad;
    std::vector<double> out(slots, 0.0);
    for (int k = 0; k < C_real; ++k) out[k * t] = v[k];
    return out;
}
PackedCtx encrypt_single_token(Inference& inf, const std::vector<double>& x_pad,
                               int d_pad, int C_real) {
    auto slots = pack_per_feature_vec(inf.slots, x_pad, d_pad, C_real);
    Ctx ct = encrypt(inf.cc(), inf.cc()->MakeCKKSPackedPlaintext(slots), inf.fhe->pk());
    return inf.pack(ct, PackingKind::Cachemir);
}

PackedCtx encrypt_single_token(Inference& inf, const std::vector<double>& x_pad) {
    const int hD = inf.size.hidDim;
    return encrypt_single_token(inf, x_pad, hD, hD);
}

}  // namespace cachemir
