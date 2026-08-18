#include "model/layer_norm.h"

#include "nonlinear.h"
#include "packing/cachemir/cachemir_norm_utils.h"
#include "packing/diagonal/diagonal_norm_utils.h"

#include <stdexcept>

namespace {

std::vector<double> pack_param(const Inference& inf, const std::vector<double>& v,
                               int d_pad, int C_real) {
    if (is_cachemir(inf.packing))
        return cachemir::pack_per_feature_vec(inf.slots, v, d_pad, C_real);
    if (is_diagonal(inf.packing) || is_cachemir_filling(inf.packing))
        return diagonal::pack_per_feature_vec(inf.slots, v, d_pad, C_real);
    throw std::runtime_error("ln_affine: unsupported packing");
}

}  // namespace

Ptx encode_ln_affine_param(Inference& inf, const std::vector<double>& v,
                           int d_pad, int C_real, int target_level,
                           bool mask_inactive, cudaStream_t stream) {
    auto slots = pack_param(inf, v, d_pad, C_real);
    if (mask_inactive &&
        (is_diagonal(inf.packing) || is_cachemir_filling(inf.packing))) {
        const int t = inf.slots / inf.size.hidDim;
        for (size_t s = 0; s < slots.size(); ++s)
            if (static_cast<int>(s % t) >= inf.n_tok) slots[s] = 0.0;
    }
    const uint32_t lv = static_cast<uint32_t>(target_level);
    if (stream != nullptr)
        return inf.cc()->MakeCKKSPackedPlaintext(slots, /*noiseScaleDeg=*/1, lv,
                                                 nullptr, 0, stream);
    return inf.cc()->MakeCKKSPackedPlaintext(slots, /*noiseScaleDeg=*/1, lv);
}

PackedCtx ln_affine(Inference& inf, const PackedCtx& normed, const std::string& tag) {
    WithStep _w(inf, "ln_affine");
    PackedCtx y = inf.fhe->mult(normed, inf.weights_at(tag + ".weight", normed)[0]);
    inf.add_affine_term(y, tag + ".bias");
    return y;
}

PackedCtx layer_norm(Inference& inf, const PackedCtx& x, const std::string& cfg_name) {
    WithStep _w(inf, "layer_norm:" + cfg_name);
    PackedCtx normed = norm(inf, x, cfg_name);
    if (fold_ln_affine(cfg_name)) {   // gamma folded into consumer; beta rides input as (beta/gamma) shift
        WithStep _ws(inf, "ln_shift");
        inf.add_affine_term(normed, cfg_name + ".shift");
        return normed;
    }
    return ln_affine(inf, normed, cfg_name);
}
