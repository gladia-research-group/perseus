#include "client_context.h"
#include "fhe_errors.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>

namespace perseus_client {

// ── cachemir ─────────────────────────────────────────────────────────────────────────
namespace cachemir {

struct CacheMirParams {
    bool is_up;
    int d, alpha, t, tp, tp_in, tp_out, r_i, r_o, n_pt;
    int bstep_c, gstep_c;
};

int interleave_idx(int m, int d, int dim) {
    int a = (dim > d) ? (dim / d) : 1;
    return (m / a + (m % a) * d) % dim;
}

CacheMirParams compute_cm_params(int N, int d_in, int d_out) {
    CacheMirParams p;
    p.is_up  = (d_in <= d_out);
    p.d      = p.is_up ? d_in : d_out;
    p.alpha  = std::max(d_in, d_out) / p.d;
    p.t      = N / p.d;
    p.tp     = N / (p.alpha * p.d);
    p.tp_in  = p.is_up ? p.t  : p.tp;
    p.tp_out = p.is_up ? p.tp : p.t;
    int d_   = p.is_up ? p.d : p.alpha * p.d;
    p.n_pt   = d_ / p.tp_out;
    p.r_i    = std::max(1, p.d * p.d / N);
    p.r_i    = std::min(p.r_i, p.n_pt);
    p.r_o    = p.n_pt / p.r_i;

    p.bstep_c = p.r_i;
    p.gstep_c = 1;
    for (int b = 1; b <= p.r_i; ++b) {
        if (p.r_i % b != 0) continue;
        const int g = p.r_i / b;
        if (b + g < p.bstep_c + p.gstep_c ||
            (b + g == p.bstep_c + p.gstep_c && b > p.bstep_c)) {
            p.bstep_c = b;
            p.gstep_c = g;
        }
    }
    return p;
}

static void collect_linear_rots(std::set<int32_t>& rots, int N, int d_in, int d_out) {
    auto p = compute_cm_params(N, d_in, d_out);

    // Input accumulation: step * (t - 1), step = 1,2,4,... while step < tp_in
    for (int step = 1; step < p.tp_in; step *= 2)
        rots.insert(step * (p.t - 1));

    // Input rotation: j * t^2, j = 1..r_i-1
    int rot2 = p.t * p.t;
    for (int j = 1; j < p.r_i; ++j)
        rots.insert(j * rot2);

    // Cascade rotation: t * tp
    rots.insert(p.t * p.tp);

    // Output accumulation: step = 1,2,4,... while step < tp_out
    for (int step = 1; step < p.tp_out; step *= 2)
        rots.insert(step);
}

static void collect_norm_rots(std::set<int32_t>& rots, int N, int hidDim) {
    const int t = N / hidDim;
    for (int s = t; s < N; s *= 2)
        rots.insert(s);
    for (int g = 1; g < N; g *= 2)
        rots.insert(g);
}

static void collect_mha_rots(std::set<int32_t>& rots, int N, int hidDim, int numHeads) {
    int t  = N / hidDim;
    int tH = t * numHeads;
    int d_head = hidDim / numHeads;

    // cache_k_push: rotate key into token slot
    for (int i = 1; i < t; ++i)
        rots.insert(-i);

    // filling->cachemir handoff (extract_token_i_cachemir): rotate a filling V/K
    for (int i = 1; i < t; ++i)
        rots.insert(i);

    // qkt query fill: replicate across token slots
    for (int step = 1; step < t; step *= 2)
        rots.insert(-step);

    // qkt sum_by_rot: reduce across dimension blocks
    for (int s = tH; s < N; s *= 2)
        rots.insert(s);

    // head_reduce_sum: intra-head masked rotations
    for (int step = 1; step < t; step *= 2) {
        rots.insert(step);
        rots.insert(step - t);  // wrap-around rotation
    }

    // softmax_v: direct score rotations by i*tH for each cached V lane
    for (int i = 1; i < d_head; ++i)
        rots.insert(i * tH);

    // softmax_v: intra-token reduction (same family as head_reduce_sum, included for clarity)
    for (int step = 1; step < t; step *= 2)
        rots.insert(step);
}

std::vector<int32_t> compute_gpt2_rot_indices(int slots, int hidDim, int ffDim, int numHeads) {
    std::set<int32_t> rots;

    // q/k/v/out: hidDim x hidDim
    collect_linear_rots(rots, slots, hidDim, hidDim);
    // up/gate: hidDim x ffDim
    collect_linear_rots(rots, slots, hidDim, ffDim);
    // down: ffDim x hidDim
    collect_linear_rots(rots, slots, ffDim, hidDim);
    // lm_head
    collect_linear_rots(rots, slots, hidDim, slots);

    // LayerNorm (ln_1 / ln_2 / ln_f) feature-axis mean + variance reductions.
    collect_norm_rots(rots, slots, hidDim);

    // MHA: KCache, qkt, head reduces, softmax-v lanes
    collect_mha_rots(rots, slots, hidDim, numHeads);

    return std::vector<int32_t>(rots.begin(), rots.end());
}

// The slot vector of encode_linear_input (cachemir_linear_utils.cu).
static std::vector<double> encode_input_slots(int N, const std::vector<double>& x, int d_in, int d_out) {
    auto p  = compute_cm_params(N, d_in, d_out);
    int d_x = p.is_up ? p.d : p.alpha * p.d;
    int M   = N / p.tp;
    std::vector<double> ptx(N, 0.0);
    if (p.is_up)
        for (int i = 0; i < p.d; ++i) ptx[i * p.t] = x.at(i);
    else
        for (int m = 0; m < M; ++m) ptx[m * p.tp] = x.at(interleave_idx(m, p.d, d_x));
    return ptx;
}

static std::vector<double> decode_linear_output(int slots, const std::vector<double>& cy,
                                                int d_in, int d_out) {
    auto p = compute_cm_params(slots, d_in, d_out);
    const int M = slots / p.tp;
    std::vector<double> y(d_out, 0.0);
    if (p.is_up && p.alpha > 1) {
        for (int m = 0; m < M; ++m) {
            const int idx = interleave_idx(m, p.d, d_out);
            if (idx < d_out) y[idx] = cy.at(m * p.tp);
        }
    } else {
        for (int i = 0; i < d_out; ++i) y[i] = cy.at(i * p.t);
    }
    return y;
}

static std::vector<std::vector<double>> decode_tokens(int slots, const std::vector<double>& cy,
                                                      int d_pad, int d_real, int T) {
    const int t = compute_cm_params(slots, d_pad, d_pad).t;
    std::vector<std::vector<double>> y(T, std::vector<double>(d_real));
    for (int tok = 0; tok < T; ++tok)
        for (int i = 0; i < d_real; ++i)
            y[tok][i] = cy.at(static_cast<size_t>(i) * t + tok);
    return y;
}

}  // namespace cachemir

// ── diagonal (and cachemir_filling, which shares the token-in-lane linear) ───────────
namespace diagonal {

struct DiagonalParams {
    int d_in, d_out;
    int t_in;
    int t_out;
    int alpha;
    bool is_up;
    int n_diag;
    int s, G;
    int max_n_tok;
};

DiagonalParams compute_dg_params(int N, int d_in, int d_out) {
    if (d_in <= 0 || d_out <= 0)
        throw fhe::FHEError("diagonal::compute_dg_params: d_in/d_out must be positive");
    const int d_max = std::max(d_in, d_out);
    const int d_min = std::min(d_in, d_out);
    if (d_max % d_min != 0)
        throw fhe::FHEError("diagonal::compute_dg_params: max(d_in, d_out) must be a multiple of min");
    if (N % d_max != 0)
        throw fhe::FHEError("diagonal::compute_dg_params: N must be a multiple of max(d_in, d_out)");

    DiagonalParams p;
    p.d_in   = d_in;
    p.d_out  = d_out;
    p.t_in   = N / d_in;
    p.t_out  = N / d_out;
    p.alpha  = d_max / d_min;
    p.is_up  = (d_in < d_out);
    p.n_diag = d_in;
    int s = static_cast<int>(std::sqrt(static_cast<double>(p.n_diag)));
    if (s < 1) s = 1;
    while (s > 1 && p.n_diag % s != 0) --s;
    p.s = s;
    p.G = p.n_diag / p.s;
    p.max_n_tok = std::min(p.t_in, p.t_out);
    return p;
}

static void collect_linear_rots(std::set<int32_t>& rots, int N, int d_in, int d_out) {
    auto p = compute_dg_params(N, d_in, d_out);
    // Baby steps: rotate x by b * t_in for b in [1, s)
    for (int b = 1; b < p.s; ++b)
        rots.insert(b * p.t_in);
    // Giant steps: rotate partial inner sum by g * s * t_in for g in [1, G)
    for (int g = 1; g < p.G; ++g)
        rots.insert(g * p.s * p.t_in);
}

static void collect_norm_rots(std::set<int32_t>& rots, int N, int hidDim) {
    const int t = N / hidDim;
    for (int s = t; s < N; s *= 2)
        rots.insert(s);
}

std::vector<int32_t> compute_gpt2_rot_indices(int slots, int hidDim, int ffDim, int /*numHeads*/) {
    std::set<int32_t> rots;

    collect_linear_rots(rots, slots, hidDim, hidDim);

    collect_linear_rots(rots, slots, hidDim, ffDim);

    {
        auto pu = compute_dg_params(slots, hidDim, ffDim);
        for (int g = 1; g < pu.alpha; ++g)
            rots.insert(-g * pu.t_out);
    }

    collect_linear_rots(rots, slots, ffDim, hidDim);

    collect_linear_rots(rots, slots, hidDim, slots);

    collect_norm_rots(rots, slots, hidDim);

    return std::vector<int32_t>(rots.begin(), rots.end());
}

// The slot vector of encode_linear_input (diagonal_linear_utils.cu); n_tok out.
static std::vector<double> encode_input_slots(int N, const std::vector<double>& x, int d_in, int d_out,
                                              int& n_tok_out) {
    auto p = compute_dg_params(N, d_in, d_out);
    if (x.size() % static_cast<size_t>(d_in) != 0)
        throw fhe::FHEError("diagonal::encode_linear_input: x.size() not a multiple of d_in");
    const int n_tok = static_cast<int>(x.size()) / d_in;
    if (n_tok < 1 || n_tok > p.max_n_tok)
        throw fhe::FHEError("diagonal::encode_linear_input: n_tok out of [1, max_n_tok]");
    n_tok_out = n_tok;  // norm() reads this to floor the empty token lanes

    std::vector<double> ptx(N, 0.0);
    for (int tok = 0; tok < n_tok; ++tok) {
        for (int i = 0; i < p.d_in; ++i) {
            const double val = x[tok * d_in + i];
            const int base = i * p.t_in;
            ptx[base + tok] = val;
        }
    }
    return ptx;
}

static std::vector<double> decode_linear_output(int slots, const std::vector<double>& cy,
                                                int d_in, int d_out) {
    auto p = compute_dg_params(slots, d_in, d_out);
    std::vector<double> y(d_out);
    for (int j = 0; j < d_out; ++j) y[j] = cy.at(j * p.t_out);
    return y;
}

static std::vector<std::vector<double>> decode_tokens(int slots, const std::vector<double>& cy,
                                                      int d_pad, int d_real, int T) {
    const int t = compute_dg_params(slots, d_pad, d_pad).t_out;   // = slots/d_pad
    std::vector<std::vector<double>> y(T, std::vector<double>(d_real));
    for (int tok = 0; tok < T; ++tok)
        for (int i = 0; i < d_real; ++i)
            y[tok][i] = cy.at(static_cast<size_t>(i) * t + tok);
    return y;
}

}  // namespace diagonal

namespace cachemir_filling {

static void collect_mha_rots(std::set<int32_t>& rots, int slots, int hidDim, int numHeads) {
    const int t = (hidDim > 0) ? slots / hidDim : 0;
    if (t <= 0) return;
    const int tH = t * numHeads;

    for (int i = 1; i < t; ++i) {
        rots.insert(i);
        rots.insert(-i);
    }

    for (int s = tH; s < slots; s *= 2) {
        rots.insert(s);
        rots.insert(-s);
    }
}

std::vector<int32_t> compute_gpt2_rot_indices(int slots, int hidDim, int ffDim, int numHeads) {
    std::set<int32_t> rots;
    for (int r : diagonal::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads)) rots.insert(r);
    collect_mha_rots(rots, slots, hidDim, numHeads);
    return std::vector<int32_t>(rots.begin(), rots.end());
}

}  // namespace cachemir_filling

// ── families ─────────────────────────────────────────────────────────────────────────
std::vector<int32_t> compute_gpt2_rot_indices(PackingKind kind, int slots, int hidDim, int ffDim,
                                              int numHeads) {
    if (is_cachemir(kind))
        return cachemir::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads);
    if (is_cachemir_filling(kind))
        return cachemir_filling::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads);
    if (is_diagonal(kind))
        return diagonal::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads);
    throw fhe::FHEError("compute_gpt2_rot_indices: unsupported packing");
}

// vit_model.cu / bert_model.cu: filling blocks ∪ the cachemir tail.
std::vector<int32_t> vit_rot_indices(int slots, int hidDim, int ffDim, int numHeads) {
    std::set<int32_t> rots;
    for (int32_t r : cachemir_filling::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads))
        rots.insert(r);
    for (int32_t r : cachemir::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads))
        rots.insert(r);
    return {rots.begin(), rots.end()};
}

std::vector<int32_t> bert_rot_indices(int slots, int hidDim, int ffDim, int numHeads) {
    return vit_rot_indices(slots, hidDim, ffDim, numHeads);
}

std::vector<int32_t> family_rot_steps(const std::string& family, const InferenceOptions& o) {
    const int slots = static_cast<int>(slots_of(o.ckks));
    std::vector<int32_t> steps;
    if (family == "gpt2") {
        // gpt2_model.cu: the main packing's band, then every aux packing's (!= main).
        for (int32_t r : compute_gpt2_rot_indices(o.packing_kind, slots, o.hidDim, o.expDim, o.numHeads))
            steps.push_back(r);
        for (PackingKind aux : o.aux_packing_kinds) {
            if (aux == o.packing_kind) continue;
            for (int32_t r : compute_gpt2_rot_indices(aux, slots, o.hidDim, o.expDim, o.numHeads))
                steps.push_back(r);
        }
    } else if (family == "vit") {
        steps = vit_rot_indices(slots, o.hidDim, o.expDim, o.numHeads);
    } else if (family == "bert") {
        steps = bert_rot_indices(slots, o.hidDim, o.expDim, o.numHeads);
    } else if (family == "generic") {
        // make_inference adds nothing
    } else {
        throw fhe::FHEError("family_rot_steps: unknown family '" + family + "'");
    }
    return steps;
}

std::vector<int32_t> family_rot_band(const std::string& family, const InferenceOptions& o) {
    std::vector<int32_t> band = o.ckks.extra_rot_steps;
    for (int32_t r : family_rot_steps(family, o)) band.push_back(r);
    std::sort(band.begin(), band.end());
    band.erase(std::unique(band.begin(), band.end()), band.end());
    return band;
}

// inference.h make_inference, minus the GPU-side state.
ClientInference make_inference(InferenceOptions o) {
    ClientInference inf;
    if (o.packing_kind == PackingKind::CachemirComplex)
        o.ckks.ckks_complex_payload = true;
    inf.fhe   = make_client_context(o.ckks);
    inf.logN  = o.ckks.logN;
    inf.slots = static_cast<int>(slots_of(o.ckks));
    inf.size.dim          = o.dim;
    inf.size.expanded     = o.expanded;
    inf.size.hidDim       = o.hidDim;
    inf.size.expDim       = o.expDim;
    inf.size.numHeads     = o.numHeads;
    inf.size.numHeadsReal = o.numHeadsReal;
    inf.size.seqLen       = o.seqLen;
    inf.mode              = o.mode;
    inf.complex           = (o.packing_kind == PackingKind::CachemirComplex);
    inf.packing           = inf.complex ? PackingKind::Cachemir : o.packing_kind;
    return inf;
}

InferenceOptions prepare_family_options(const std::string& family, InferenceOptions o) {
    if (family == "vit" || family == "bert")
        o.packing_kind = PackingKind::CachemirFilling;     // vit_model.cu:131 / bert_model.cu:154
    if (o.packing_kind == PackingKind::CachemirComplex)
        o.ckks.ckks_complex_payload = true;                // inference.h make_inference
    for (int32_t r : family_rot_steps(family, o)) o.ckks.extra_rot_steps.push_back(r);
    return o;
}

ClientInference make_gpt2_inference(InferenceOptions opts) {
    return make_inference(prepare_family_options("gpt2", std::move(opts)));
}

ClientInference make_vit_inference(InferenceOptions opts) {
    return make_inference(prepare_family_options("vit", std::move(opts)));
}

ClientInference make_bert_inference(InferenceOptions opts) {
    return make_inference(prepare_family_options("bert", std::move(opts)));
}

// ── tokens ───────────────────────────────────────────────────────────────────────────
namespace {

ClientCt encode_linear_input(ClientInference& inf, const std::vector<double>& x, int d_in, int d_out,
                             int target_level) {
    if (!inf.fhe || !inf.fhe->kp.publicKey)
        throw fhe::FHEError("encode: the session has no public key");
    std::vector<double> ptx;
    PackingKind kind = inf.packing;
    if (is_cachemir(inf.packing)) {
        ptx  = cachemir::encode_input_slots(inf.slots, x, d_in, d_out);
        kind = PackingKind::Cachemir;
    } else if (is_diagonal(inf.packing) || is_cachemir_filling(inf.packing)) {
        int n_tok = 0;
        ptx = diagonal::encode_input_slots(inf.slots, x, d_in, d_out, n_tok);
        inf.n_tok = n_tok;
    } else {
        throw fhe::FHEError("encode_linear_input: unsupported packing");
    }
    const auto& cc = inf.fhe->cc;
    lbcrypto::Plaintext pt = cc->MakeCKKSPackedPlaintext(ptx, /*noiseScaleDeg=*/1,
                                                         static_cast<uint32_t>(target_level));
    return ClientCt{cc->Encrypt(inf.fhe->kp.publicKey, pt), kind};
}

lbcrypto::Plaintext decrypt_pt(const ClientInference& inf, const ClientCt& pc) {
    if (!pc.ct) throw fhe::FHEError("decrypt: empty ciphertext");
    if (!inf.fhe || !inf.fhe->kp.secretKey)
        throw fhe::FHEError("decrypt: this session holds no secret key (load_secret_key first)");
    lbcrypto::Plaintext pt;
    inf.fhe->cc->Decrypt(inf.fhe->kp.secretKey, pc.ct, &pt);
    return pt;
}

}  // namespace

ClientCt pack_tokens(ClientInference& inf, const std::vector<std::vector<double>>& embeddings,
                     int target_level) {
    const int T      = static_cast<int>(embeddings.size());
    const int d_pad  = inf.size.hidDim;
    const int d_real = inf.size.getRealHidDim();
    std::vector<double> flat(static_cast<size_t>(T) * d_pad, 0.0);
    for (int tok = 0; tok < T; ++tok) {
        const int n = std::min<int>(d_real, static_cast<int>(embeddings[tok].size()));
        for (int i = 0; i < n; ++i)
            flat[static_cast<size_t>(tok) * d_pad + i] = embeddings[tok][i];
    }
    return encode_linear_input(inf, flat, d_pad, d_pad, target_level);   // dispatched per packing
}

std::vector<double> decrypt(const ClientInference& inf, const ClientCt& pc) {
    return decrypt_pt(inf, pc)->GetRealPackedValue();
}

std::vector<std::complex<double>> decrypt_complex(const ClientInference& inf, const ClientCt& pc) {
    return decrypt_pt(inf, pc)->GetCKKSPackedValue();
}

std::vector<double> decode_linear_output(PackingKind kind, const std::vector<double>& cy,
                                         int slots, int d_in, int d_out) {
    if (is_cachemir(kind)) return cachemir::decode_linear_output(slots, cy, d_in, d_out);
    if (is_diagonal(kind) || is_cachemir_filling(kind))
        return diagonal::decode_linear_output(slots, cy, d_in, d_out);
    throw fhe::FHEError("decode_linear_output: unsupported packing");
}

static std::vector<std::vector<double>> decode_tokens(PackingKind kind, const std::vector<double>& cy,
                                                      int slots, int d_pad, int d_real, int T) {
    if (is_cachemir(kind)) return cachemir::decode_tokens(slots, cy, d_pad, d_real, T);
    if (is_diagonal(kind) || is_cachemir_filling(kind))
        return diagonal::decode_tokens(slots, cy, d_pad, d_real, T);
    throw fhe::FHEError("decode_tokens: unsupported packing");
}

std::vector<std::vector<double>> unpack_tokens(const ClientInference& inf, const ClientCt& pc, int T) {
    auto raw = decrypt(inf, pc);
    return decode_tokens(inf.packing, raw, inf.slots, inf.size.hidDim, inf.size.getRealHidDim(), T);
}

ClientCt encode_token_input(ClientInference& inf, const std::vector<double>& x_real) {
    return pack_tokens(inf, {x_real}, static_cast<int>(inf.fhe->bootstrap_output_level()));
}

std::vector<double> decode_token_output(const ClientInference& inf, const ClientCt& pc) {
    return unpack_tokens(inf, pc, 1)[0];
}

std::vector<std::vector<double>> decode_tokens_output(const ClientInference& inf, const ClientCt& pc,
                                                      int T) {
    return unpack_tokens(inf, pc, T);
}

int lm_head_tile_width(const ClientInference& inf, int vocab) {
    return vocab <= inf.size.hidDim ? inf.size.hidDim : inf.slots;
}

std::vector<double> decode_lm_head_logits(const ClientInference& inf, const std::vector<ClientCt>& tiles,
                                          int vocab) {
    const int W_tile = lm_head_tile_width(inf, vocab);
    const int K = (vocab + W_tile - 1) / W_tile;
    std::vector<double> logits;
    logits.reserve(static_cast<size_t>(K) * W_tile);

    if (inf.complex) {
        for (size_t m = 0; m < tiles.size(); ++m) {
            auto cval = decrypt_complex(inf, tiles[m]);
            std::vector<double> re(cval.size()), im(cval.size());
            for (size_t i = 0; i < cval.size(); ++i) {
                re[i] = cval[i].real();
                im[i] = cval[i].imag();
            }
            for (int lane = 0; lane < 2; ++lane) {
                const int k = 2 * static_cast<int>(m) + lane;
                if (k >= K) break;
                const std::vector<double>& src = (lane == 0) ? re : im;
                auto tile = decode_linear_output(tiles[m].packing, src, inf.slots,
                                                 inf.size.hidDim, W_tile);
                const int col0  = k * W_tile;
                const int wreal = std::min(W_tile, vocab - col0);
                logits.insert(logits.end(), tile.begin(), tile.begin() + wreal);
            }
        }
        return logits;
    }

    for (size_t k = 0; k < tiles.size(); ++k) {
        auto raw  = decrypt(inf, tiles[k]);
        auto tile = decode_linear_output(tiles[k].packing, raw, inf.slots,
                                         inf.size.hidDim, W_tile);
        const int col0  = static_cast<int>(k) * W_tile;
        const int wreal = std::min(W_tile, vocab - col0);
        logits.insert(logits.end(), tile.begin(), tile.begin() + wreal);
    }
    return logits;
}

// ── bytes (serial.cu semantics) ────────────────────────────────────────────────
std::string serialize_ct(const ClientCt& pc) {
    if (!pc.ct) throw fhe::FHEError("serialize_ct: empty ciphertext");
    std::stringstream ss;
    lbcrypto::Serial::Serialize(pc.ct, ss, lbcrypto::SerType::BINARY);
    return ss.str();
}

ClientCt deserialize_ct(const ClientInference& inf, const std::string& data) {
    LbCt host;
    {
        std::stringstream ss(data);
        lbcrypto::Serial::Deserialize(host, ss, lbcrypto::SerType::BINARY);
    }
    if (!host) throw fhe::FHEError("deserialize_ct: no ciphertext in payload");
    return ClientCt{host, inf.packing};
}

}  // namespace perseus_client
