#include "packing/cachemir_filling/cachemir_filling.h"
#include "nonlinear.h"
#include "inference.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Per-half LN body = THE shared core (norm.cu ln_inv_sqrt_tail); only the
// conj-split + per-half centering masks + repack below are token-pair-specific.
static PackedCtx ln_tail(Inference& inf, PackedCtx xs, const NormConfig& cfg,
                         int rD, int t, double c_eff_sq) {
    return ln_inv_sqrt_tail(inf, std::move(xs), cfg, rD, t, c_eff_sq, sparse_ln_enabled());
}

static Ptx ln_center_mask_imag(Inference& inf, const PackedCtx& ref, int d, int t,
                               const std::vector<double>& css, const std::string& cfg_name,
                               double scale_fb) {
    const auto active = stride_slots(ref.packing, inf.slots, d, t, 0);
    const int n      = static_cast<int>(css.size());
    const int pos0   = inf.output.capture_t;
    const int n_tok  = inf.n_tok;
    return inf.encode_at_cached_complex(
        inf.scoped("cf.ln.center.im." + cfg_name +
                   ".ct" + std::to_string(pos0) + ".nt" + std::to_string(n_tok)),
        ref,
        [&] {
            std::vector<std::complex<double>> mask(inf.slots, {0.0, 0.0});
            for (int slot : active) {
                const int tok = slot % t;
                if (tok < n_tok) {
                    const int pos = std::max(0, std::min(pos0 + tok, n - 1));
                    const double s = (n > 0) ? 0.5 * std::sqrt(css[pos]) : scale_fb;
                    mask[slot] = {0.0, -s};
                }
            }
            return mask;
        });
}

PackedCtx norm_token_pair(Inference& inf, const PackedCtx& x, const std::string& cfg_name) {
    if (inf.fhe->level_for_ct(x.ct) >= inf.fhe->level_limit())
        throw std::runtime_error("[tp.ln] conj_split input out of band (level >= limit)");
    const NormConfig& cfg = inf.norm_cfg.at(cfg_name);
    const int rD = inf.size.getRealHidDim();
    const int t  = inf.slots / inf.size.hidDim;
    const double c_eff_sq = cfg.center_scale * cfg.center_scale;   // filling: scalar (center_pos = -1)
    const int off = inf.output.capture_t;
    const int nA = inf.n_tok, nB = inf.n_tok_imag;

    auto [A, B] = inf.fhe->conj_split(x);   // A = 2*Re(x), B = 2i*Im(x)

    // A half (Re): A == the real body's leading im_cleanse; real center mask 0.5*sqrt(css[off+tok]).
    inf.output.capture_t = off; inf.n_tok = nA;
    Ptx cmaskA = encode_ln_center_mask(inf, A, cfg_name, rD, t, 0.5 * std::sqrt(c_eff_sq),
                                       -1, cfg.center_scale_sq);
    PackedCtx xsA = inf.fhe->mult(A, cmaskA);
    PackedCtx outA = ln_tail(inf, std::move(xsA), cfg, rD, t, c_eff_sq);

    PackedCtx out;
    if (nB > 0) {
        inf.output.capture_t = off + t; inf.n_tok = nB;
        Ptx cmaskB = ln_center_mask_imag(inf, B, rD, t, cfg.center_scale_sq, cfg_name,
                                         0.5 * std::sqrt(c_eff_sq));
        PackedCtx xsB = inf.fhe->mult(B, cmaskB);   // 2i*Im * (-0.5i*sqrt(css)) = sqrt(css)*Im (real)
        PackedCtx outB = ln_tail(inf, std::move(xsB), cfg, rD, t, c_eff_sq);
        out = inf.fhe->pair_pack_cleansed(std::move(outA), std::move(outB));   // LN(A) + i*LN(B)
    } else {
        out = std::move(outA);
    }
    inf.output.capture_t = off; inf.n_tok = nA;
    return out;
}

