#include "packing/cachemir/cachemir_linear_utils.h"
#include "inference.h"
#include "io/weight_io.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <iostream>

namespace cachemir {

namespace {

struct CoeffPlan {
    bool     on = false;
    uint32_t lv1 = 0;
    double   ratio = 1.0;
    double   sf_target = 0.0;
};

// Coefficient staging needs |w| < 2, which GPT-2's LN-folded weights exceed, so it is off
// unless asked for. Read per call, not cached: EncGPT2.bind(coeff_encode=...) sets it around
// one bind, and a cached read would silently ignore every bind after the first.
bool cm_coeff_enabled() {
    const char* e = std::getenv("FHE_PT_COEFF_ENCODE");
    return e && *e && std::atoi(e) == 1;
}

// `mx` is the per-slot magnitude bound: max|w| for a real encode, max|z| (modulus) for a
// complex one — the canonical-embedding coefficients obey |c_k| <= max_j|z_j| either way.
// `what` names the matrix class in the refusal message, which is the only way to tell
// "the flag did nothing" from "the flag is off".
CoeffPlan cm_coeff_plan(Inference& inf, double mx, uint32_t lv, const char* what) {
    CoeffPlan c;
    if (!cm_coeff_enabled() || !inf.pt_stage_hook) return c;
    c.lv1 = static_cast<uint32_t>(inf.fhe->composite_degree * inf.fhe->total_depth);
    c.sf_target = inf.cc()->ScalingFactorReal(lv);
    c.ratio = c.sf_target / inf.cc()->ScalingFactorReal(c.lv1);
    // The 1-limb coefficient encode needs |w|*scale inside the centered-lift bound;
    // a matrix past it takes the full encode. GPT-2's LN-folded weights are past it on
    // both chains, so this is the usual outcome rather than an exceptional one.
    const double lift_bits = static_cast<double>(inf.fhe->first_mod_bits);
    c.on = (mx * c.sf_target < 0.45 * std::pow(2.0, lift_bits));
    if (!c.on) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "[coeff_encode] hybrid: |w|max=" << mx << " (" << what
                      << ") exceeds the centered-lift bound -> this matrix class "
                         "falls back to the full encode\n";
        }
        c.ratio = 1.0;
    }
    return c;
}

double abs_max(const std::vector<std::vector<double>>& W) {
    double mx = 0.0;
    for (const auto& row : W)
        for (double x : row) mx = std::max(mx, std::abs(x));
    return mx;
}

// Slot-modulus bound for the fused complex encode: max sqrt(re^2 + im^2) elementwise.
double abs_max_complex(const std::vector<std::vector<double>>& W_re,
                       const std::vector<std::vector<double>>& W_im) {
    double mx2 = 0.0;
    for (size_t r = 0; r < W_re.size(); ++r)
        for (size_t j = 0; j < W_re[r].size(); ++j) {
            const double re = W_re[r][j], im = W_im[r][j];
            mx2 = std::max(mx2, re * re + im * im);
        }
    return std::sqrt(mx2);
}
}  // namespace

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

PackedCtx encode_linear_input(Inference& inf, const std::vector<double>& x,
                              int d_in, int d_out, int target_level) {
    auto p  = compute_cm_params(inf.slots, d_in, d_out);
    int N   = inf.slots;
    int d_x = p.is_up ? p.d : p.alpha * p.d;
    int M   = N / p.tp;
    std::vector<double> ptx(N, 0.0);
    if (p.is_up)
        for (int i = 0; i < p.d; ++i) ptx[i * p.t] = x[i];
    else
        for (int m = 0; m < M; ++m) ptx[m * p.tp] = x[interleave_idx(m, p.d, d_x)];

    Ctx ct = encrypt(
        inf.cc(),
        inf.cc()->MakeCKKSPackedPlaintext(ptx, /*noiseScaleDeg=*/1,
                                          static_cast<uint32_t>(target_level)),
        inf.fhe->pk());
    return inf.pack(ct, PackingKind::Cachemir);
}

std::vector<Ptx> encode_weight_matrix(Inference& inf, const std::vector<std::vector<double>>& W,
                                       int d_in, int d_out, int target_level) {
    return encode_weight_matrix(inf, W, d_in, d_out, target_level, nullptr);
}

std::vector<Ptx> encode_weight_matrix(Inference& inf, const std::vector<std::vector<double>>& W,
                                       int d_in, int d_out, int target_level, cudaStream_t stream) {
    int N     = inf.slots;
    auto p    = compute_cm_params(N, d_in, d_out);
    int M_out         = N / p.tp_out;
    int cascade_shift = (p.t * p.tp) / p.tp_out;

    std::vector<std::vector<double>> pt(p.n_pt, std::vector<double>(N, 0.0));
    for (int j = 0; j < p.r_i; ++j) {
        const int  g   = j / p.bstep_c;
        const int  s_g = g * p.bstep_c * p.t * p.t;   // < N (g*bstep_c < r_i, r_i*t*t = N)
        for (int k = 0; k < p.r_o; ++k)
            for (int i = 0; i < N; ++i) {
                const int ip = ((i - s_g) % N + N) % N;
                int row = ((ip / p.t + j * p.t + ip % p.tp_in) % p.d)
                        + ((ip % p.t) / p.tp_in) * p.d;
                int ms  = ((ip / p.tp_out - k * cascade_shift) % M_out + M_out) % M_out;
                pt[j * p.r_o + k][i] = W[row][interleave_idx(ms, p.d, d_out)];
            }
    }

    const uint32_t lv = static_cast<uint32_t>(target_level);
    const CoeffPlan c = cm_coeff_plan(inf, abs_max(W), lv, "real");
    if (c.on)
        for (auto& v : pt)
            for (double& x : v) x *= c.ratio;
    const uint32_t enc_lv = c.on ? c.lv1 : lv;
    std::vector<Ptx> result(p.n_pt);
    for (int i = 0; i < p.n_pt; ++i) {
        if (stream != nullptr) {
            result[i] = inf.cc()->MakeCKKSPackedPlaintext(pt[i], /*noiseScaleDeg=*/1, enc_lv, nullptr, 0, stream);
        } else {
            result[i] = inf.cc()->MakeCKKSPackedPlaintext(pt[i], /*noiseScaleDeg=*/1, enc_lv);
        }
        if (c.on) inf.cc()->MarkCoeffStaged(result[i], lv, c.sf_target, /*prescale_log2=*/0);
    }
    return result;
}

std::vector<Ptx> encode_weight_matrix_complex(Inference& inf,
                                              const std::vector<std::vector<double>>& W_re,
                                              const std::vector<std::vector<double>>& W_im,
                                              int d_in, int d_out, int target_level,
                                              cudaStream_t stream) {
    int N     = inf.slots;
    auto p    = compute_cm_params(N, d_in, d_out);
    int M_out         = N / p.tp_out;
    int cascade_shift = (p.t * p.tp) / p.tp_out;

    std::vector<std::vector<std::complex<double>>> pt(
        p.n_pt, std::vector<std::complex<double>>(N, std::complex<double>(0.0, 0.0)));
    for (int j = 0; j < p.r_i; ++j) {
        const int  g   = j / p.bstep_c;
        const int  s_g = g * p.bstep_c * p.t * p.t;
        for (int k = 0; k < p.r_o; ++k)
            for (int i = 0; i < N; ++i) {
                const int ip = ((i - s_g) % N + N) % N;
                int row = ((ip / p.t + j * p.t + ip % p.tp_in) % p.d)
                        + ((ip % p.t) / p.tp_in) * p.d;
                int ms  = ((ip / p.tp_out - k * cascade_shift) % M_out + M_out) % M_out;
                int col = interleave_idx(ms, p.d, d_out);
                pt[j * p.r_o + k][i] = std::complex<double>(W_re[row][col], W_im[row][col]);
            }
    }

    const uint32_t lv = static_cast<uint32_t>(target_level);

    const CoeffPlan c = cm_coeff_plan(inf, abs_max_complex(W_re, W_im), lv, "complex");
    if (c.on)
        for (auto& v : pt)
            for (auto& z : v) z *= c.ratio;
    const uint32_t enc_lv = c.on ? c.lv1 : lv;
    std::vector<Ptx> result(p.n_pt);
    for (int i = 0; i < p.n_pt; ++i) {
        if (stream != nullptr)
            result[i] = inf.cc()->MakeCKKSPackedPlaintext(pt[i], /*noiseScaleDeg=*/1, enc_lv, nullptr, 0, stream);
        else
            result[i] = inf.cc()->MakeCKKSPackedPlaintext(pt[i], /*noiseScaleDeg=*/1, enc_lv);
        if (c.on) inf.cc()->MarkCoeffStaged(result[i], lv, c.sf_target, /*prescale_log2=*/0);
    }
    return result;
}

Ptx encode_bias_vector_complex(Inference& inf, const std::vector<double>& b_re,
                               const std::vector<double>& b_im, int d_in, int d_out, bool fill,
                               cudaStream_t stream) {
    int N = inf.slots;
    auto p = compute_cm_params(N, d_in, d_out);
    std::vector<std::complex<double>> pt(N, std::complex<double>(0.0, 0.0));

    auto put = [&](int slot, int idx) {
        pt[slot] = std::complex<double>(idx < (int)b_re.size() ? b_re[idx] : 0.0,
                                        idx < (int)b_im.size() ? b_im[idx] : 0.0);
    };
    if (p.is_up && p.alpha > 1) {
        int M = N / p.tp;
        for (int m = 0; m < M; ++m) {
            int idx = interleave_idx(m, p.d, d_out);
            if (idx < (int)b_re.size() || idx < (int)b_im.size()) {
                put(m * p.tp, idx);
                if (fill) for (int j = 1; j < p.tp; ++j) put(m * p.tp + j, idx);
            }
        }
    } else {
        const int n = std::max((int)b_re.size(), (int)b_im.size());
        for (int i = 0; i < n; ++i) {
            put(i * p.t, i);
            if (fill) for (int j = 1; j < p.t; ++j) put(i * p.t + j, i);
        }
    }

    if (stream != nullptr)
        return inf.cc()->MakeCKKSPackedPlaintext(pt, /*noiseScaleDeg=*/1, 0, nullptr, 0, stream);
    return inf.cc()->MakeCKKSPackedPlaintext(pt);
}

std::vector<Ptx> encode_weight_matrix_outputpack(Inference& inf,
                                                 const std::vector<std::vector<double>>& W,
                                                 int d_in, int d_out, int target_level,
                                                 cudaStream_t stream) {
    int N     = inf.slots;
    auto p    = compute_cm_params(N, d_in, d_out);
    if (p.r_o % 2 != 0)
        throw std::runtime_error("encode_weight_matrix_outputpack: r_o must be even (got " +
                                 std::to_string(p.r_o) + ")");
    const int rop = p.r_o / 2;
    int M_out         = N / p.tp_out;
    int cascade_shift = (p.t * p.tp) / p.tp_out;

    std::vector<std::vector<std::complex<double>>> pt(
        static_cast<size_t>(p.r_i) * rop, std::vector<std::complex<double>>(N, {0.0, 0.0}));
    auto col_at = [&](int ip, int k) {
        int ms = ((ip / p.tp_out - k * cascade_shift) % M_out + M_out) % M_out;
        return interleave_idx(ms, p.d, d_out);
    };
    for (int j = 0; j < p.r_i; ++j) {
        const int  g   = j / p.bstep_c;
        const int  s_g = g * p.bstep_c * p.t * p.t;
        for (int kp = 0; kp < rop; ++kp)
            for (int i = 0; i < N; ++i) {
                const int ip = ((i - s_g) % N + N) % N;
                int row = ((ip / p.t + j * p.t + ip % p.tp_in) % p.d)
                        + ((ip % p.t) / p.tp_in) * p.d;
                pt[j * rop + kp][i] = std::complex<double>(W[row][col_at(ip, 2 * kp)],
                                                           W[row][col_at(ip, 2 * kp + 1)]);
            }
    }

    const uint32_t lv = static_cast<uint32_t>(target_level);

    const CoeffPlan c =
        cm_coeff_plan(inf, std::sqrt(2.0) * abs_max(W), lv, "outputpack");
    if (c.on)
        for (auto& v : pt)
            for (auto& z : v) z *= c.ratio;
    const uint32_t enc_lv = c.on ? c.lv1 : lv;
    std::vector<Ptx> result(static_cast<size_t>(p.r_i) * rop);
    for (size_t i = 0; i < result.size(); ++i) {
        if (stream != nullptr)
            result[i] = inf.cc()->MakeCKKSPackedPlaintext(pt[i], /*noiseScaleDeg=*/1, enc_lv, nullptr, 0, stream);
        else
            result[i] = inf.cc()->MakeCKKSPackedPlaintext(pt[i], /*noiseScaleDeg=*/1, enc_lv);
        if (c.on) inf.cc()->MarkCoeffStaged(result[i], lv, c.sf_target, /*prescale_log2=*/0);
    }
    return result;
}

Ptx encode_bias_vector(Inference& inf, const std::vector<double>& b, int d_in, int d_out, bool fill) {
    int N = inf.slots;
    auto p = compute_cm_params(N, d_in, d_out);
    std::vector<double> pt(N, 0.0);

    if (p.is_up && p.alpha > 1) {
        int M = N / p.tp;
        for (int m = 0; m < M; ++m) {
            int idx = interleave_idx(m, p.d, d_out);
            if (idx < (int)b.size()) {
                pt[m * p.tp] = b[idx];
                if (fill)
                    for (int j = 1; j < p.tp; ++j)
                        pt[m * p.tp + j] = b[idx];
            }
        }
    } else {
        for (int i = 0; i < (int)b.size(); ++i) {
            pt[i * p.t] = b[i];
            if (fill)
                for (int j = 1; j < p.t; ++j)
                    pt[i * p.t + j] = b[i];
        }
    }

    return inf.cc()->MakeCKKSPackedPlaintext(pt);
}

Ptx encode_bias_vector(Inference& inf, const std::vector<double>& b, int d_in, int d_out, bool fill,
                       cudaStream_t stream) {
    int N = inf.slots;
    auto p = compute_cm_params(N, d_in, d_out);
    std::vector<double> pt(N, 0.0);

    if (p.is_up && p.alpha > 1) {
        int M = N / p.tp;
        for (int m = 0; m < M; ++m) {
            int idx = interleave_idx(m, p.d, d_out);
            if (idx < (int)b.size()) {
                pt[m * p.tp] = b[idx];
                if (fill)
                    for (int j = 1; j < p.tp; ++j)
                        pt[m * p.tp + j] = b[idx];
            }
        }
    } else {
        for (int i = 0; i < (int)b.size(); ++i) {
            pt[i * p.t] = b[i];
            if (fill)
                for (int j = 1; j < p.t; ++j)
                    pt[i * p.t + j] = b[i];
        }
    }

    if (stream != nullptr)
        return inf.cc()->MakeCKKSPackedPlaintext(pt, /*noiseScaleDeg=*/1, 0, nullptr, 0, stream);
    return inf.cc()->MakeCKKSPackedPlaintext(pt);
}

std::vector<Ptx> load_weight_txt(Inference& inf, const std::string& path,
                                  int d_in, int d_out, int target_level) {
    return cachemir::encode_weight_matrix(inf,
                                          weight_io::load_matrix_txt(path, d_in, d_out),
                                          d_in, d_out, target_level);
}

std::vector<double> decode_linear_output(int slots, const std::vector<double>& cy,
                                         int d_in, int d_out) {
    auto p = compute_cm_params(slots, d_in, d_out);
    const int M = slots / p.tp;
    std::vector<double> y(d_out, 0.0);
    if (p.is_up && p.alpha > 1) {
        for (int m = 0; m < M; ++m) {
            const int idx = interleave_idx(m, p.d, d_out);
            if (idx < d_out) y[idx] = cy[m * p.tp];
        }
    } else {
        for (int i = 0; i < d_out; ++i) y[i] = cy[i * p.t];
    }
    return y;
}

std::vector<std::vector<double>> decode_tokens(int slots, const std::vector<double>& cy,
                                               int d_pad, int d_real, int T) {
    const int t = compute_cm_params(slots, d_pad, d_pad).t;
    std::vector<std::vector<double>> y(T, std::vector<double>(d_real));
    for (int tok = 0; tok < T; ++tok)
        for (int i = 0; i < d_real; ++i)
            y[tok][i] = cy[static_cast<size_t>(i) * t + tok];
    return y;
}

}  // namespace cachemir
