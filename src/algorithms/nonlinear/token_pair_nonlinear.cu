#include "packing/cachemir_filling/cachemir_filling.h"
#include "nonlinear.h"          // GeLUConfig, GeLUMethod, gelu_*_core

#include <stdexcept>
#include <utility>
#include <vector>

// Per-half GELU bodies = THE shared cores (nonlinear.cu); only the entry prescale
// (done by the caller: the conj 2x rides the /2) and the output mult against the
// RAW half (replaces the real arm's im_cleanse — same 2x, same mask pairing) are
// token-pair-specific.
static PackedCtx gelu_half(Inference& inf, PackedCtx x2, const PackedCtx& half,
                           const GeLUConfig& cfg) {
    PackedCtx z = gelu_softsign_core(inf, std::move(x2), cfg);
    z = inf.fhe->mult(z, half);                        // output x = the raw half (replaces im_cleanse)
    Ptx quarter_mask = inf.gelu_half_mask(z, 0.25);
    return inf.fhe->mult(z, quarter_mask);
}

static PackedCtx gelu_half_thor(Inference& inf, PackedCtx x2, const PackedCtx& half,
                                const GeLUConfig& cfg) {
    PackedCtx g = gelu_thor_core(inf, std::move(x2), cfg);
    g = inf.fhe->mult(g, half);                        // output x = the raw half
    Ptx half_mask = inf.gelu_half_mask(g, 0.5);
    return inf.fhe->mult(g, half_mask);
}

PackedCtx gelu_token_pair(Inference& inf, const PackedCtx& x, const std::string& cfg_name) {
    const GeLUConfig& cfg = inf.gelu_cfg.at(cfg_name);
    PackedCtx (*half_fn)(Inference&, PackedCtx, const PackedCtx&, const GeLUConfig&);
    switch (cfg.method) {
        case GeLUMethod::SOFTSIGN_INV_SQRT: half_fn = gelu_half;      break;
        case GeLUMethod::THOR_COMPOSITE:    half_fn = gelu_half_thor; break;
        default:
            throw std::runtime_error("[tp.gelu] unsupported GeLU method for token-pair");
    }
    if (inf.fhe->level_for_ct(x.ct) >= inf.fhe->level_limit())
        throw std::runtime_error("[tp.gelu] conj_split input out of band (level >= limit)");
    const int nA = inf.n_tok, nB = inf.n_tok_imag;
    auto [A, B] = inf.fhe->conj_split(x);   // A = 2*Re(x), B = 2i*Im(x)

    inf.n_tok = nA;
    PackedCtx x2A  = inf.fhe->mult(A, 1.0 / (2.0 * cfg.xmax));    // A prescale: cancel the conj 2x
    PackedCtx outA = half_fn(inf, std::move(x2A), A, cfg);
    PackedCtx out;
    if (nB > 0) {
        inf.n_tok = nB;
        Ptx isc = inf.encode_complex_const_at(0.0, -1.0 / (2.0 * cfg.xmax), B);   // -i/(2xmax): realify + /2
        PackedCtx x2B  = inf.fhe->mult(B, isc);
        PackedCtx outB = half_fn(inf, std::move(x2B), B, cfg);   // i*gelu(rB)
        out = inf.fhe->add(outA, outB);                           // outB already imaginary
    } else {
        out = std::move(outA);
    }
    inf.n_tok = nA;
    inf.fhe->tp_probe("gelu", out.ct);
    return out;
}

