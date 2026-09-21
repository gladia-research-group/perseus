#include "packing/cachemir/cachemir_linear_utils.h"
#include "inference.h"
#include "io/weight_io.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>

namespace cachemir {

namespace {
// Decode-side 1-limb COEFFICIENT encode of block weights (2026-09-15, GPT-2 large scratch patch).
// Mirror of diagonal_linear_utils.cu's coeff mode: encode at the q0 level (one limb, ~0.5 MB) with the
// values pre-scaled by sf(target)/sf(q0), then MarkCoeffStaged so the staged load expands the limbs on
// the GPU at the target level. The cached decode path (cached_block_op) always extracts (BeginStageBlock
// + ExtractRawPlaintext into the pinned arena) two blocks ahead of the upload, which is what a
// coeff-staged plaintext requires; the streamed loader (GPT2_CACHE=0) has no extract step, so this mode
// is for GPT2_CACHE=1 only. Env-gated: FHE_DECODE_COEFF_ENCODE=1 (default OFF = the pinned behaviour,
// full ~13 MB plaintexts at level 16, which is 660 GB of host memory for large's 36 x 1408).
struct CoeffPlan { bool on = false; uint32_t lv1 = 0; double ratio = 1.0; double sf_target = 0.0; };

bool decode_coeff_encode_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("FHE_DECODE_COEFF_ENCODE");
        return e && *e && std::atoi(e) != 0;
    }();
    return v;
}

CoeffPlan coeff_plan(Inference& inf, uint32_t lv, double wmax, const char* what) {
    CoeffPlan c;
    if (!decode_coeff_encode_enabled()) return c;
    c.lv1       = static_cast<uint32_t>(inf.fhe->total_depth);   // 1-limb (q0) encode level
    c.sf_target = inf.cc()->ScalingFactorReal(lv);
    c.ratio     = c.sf_target / inf.cc()->ScalingFactorReal(c.lv1);
    // Centered lift needs |round(w * sf_target)| < q0/2 (q0 = 2^FIRST_MOD_BITS-class), as in diagonal.
    if (wmax * c.sf_target >= 0.45 * std::pow(2.0, 60)) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "[decode_coeff_encode] hybrid: |w|max=" << wmax << " (" << what
                      << ") exceeds the centered-lift bound -> full encode for this matrix class\n";
        }
        return CoeffPlan{};
    }
    c.on = true;
    return c;
}

double abs_max(const std::vector<std::vector<double>>& W) {
    double mx = 0.0;
    for (const auto& row : W) for (double x : row) mx = std::max(mx, std::abs(x));
    return mx;
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
    std::vector<Ptx> result(p.n_pt);
    for (int i = 0; i < p.n_pt; ++i)
        result[i] = inf.cc()->MakeCKKSPackedPlaintext(pt[i], /*noiseScaleDeg=*/1, lv);
    return result;
}

std::vector<Ptx> encode_weight_matrix(Inference& inf, const std::vector<std::vector<double>>& W,
                                       int d_in, int d_out, int target_level, cudaStream_t stream) {
    int N     = inf.slots;
    auto p    = compute_cm_params(N, d_in, d_out);
    int M_out         = N / p.tp_out;
    int cascade_shift = (p.t * p.tp) / p.tp_out;
    const CoeffPlan cp = coeff_plan(inf, static_cast<uint32_t>(target_level), abs_max(W), "real");

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
                pt[j * p.r_o + k][i] = W[row][interleave_idx(ms, p.d, d_out)] * cp.ratio;
            }
    }

    const uint32_t lv = static_cast<uint32_t>(target_level);
    const uint32_t enc_lv = cp.on ? cp.lv1 : lv;
    std::vector<Ptx> result(p.n_pt);
    for (int i = 0; i < p.n_pt; ++i) {
        if (stream != nullptr) {
            result[i] = inf.cc()->MakeCKKSPackedPlaintext(pt[i], /*noiseScaleDeg=*/1, enc_lv, nullptr, 0, stream);
        } else {
            result[i] = inf.cc()->MakeCKKSPackedPlaintext(pt[i], /*noiseScaleDeg=*/1, enc_lv);
        }
        if (cp.on) inf.cc()->MarkCoeffStaged(result[i], lv, cp.sf_target);
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
    const CoeffPlan cp = coeff_plan(inf, static_cast<uint32_t>(target_level),
                                    std::max(abs_max(W_re), abs_max(W_im)), "complex");

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
                pt[j * p.r_o + k][i] = std::complex<double>(W_re[row][col] * cp.ratio, W_im[row][col] * cp.ratio);
            }
    }

    const uint32_t lv = static_cast<uint32_t>(target_level);
    const uint32_t enc_lv = cp.on ? cp.lv1 : lv;
    std::vector<Ptx> result(p.n_pt);
    for (int i = 0; i < p.n_pt; ++i) {
        if (stream != nullptr)
            result[i] = inf.cc()->MakeCKKSPackedPlaintext(pt[i], /*noiseScaleDeg=*/1, enc_lv, nullptr, 0, stream);
        else
            result[i] = inf.cc()->MakeCKKSPackedPlaintext(pt[i], /*noiseScaleDeg=*/1, enc_lv);
        if (cp.on) inf.cc()->MarkCoeffStaged(result[i], lv, cp.sf_target);
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
    const CoeffPlan cp = coeff_plan(inf, static_cast<uint32_t>(target_level), abs_max(W), "outputpack");

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
                pt[j * rop + kp][i] = std::complex<double>(W[row][col_at(ip, 2 * kp)] * cp.ratio,
                                                           W[row][col_at(ip, 2 * kp + 1)] * cp.ratio);
            }
    }

    const uint32_t lv = static_cast<uint32_t>(target_level);
    const uint32_t enc_lv = cp.on ? cp.lv1 : lv;
    std::vector<Ptx> result(static_cast<size_t>(p.r_i) * rop);
    for (size_t i = 0; i < result.size(); ++i) {
        if (stream != nullptr)
            result[i] = inf.cc()->MakeCKKSPackedPlaintext(pt[i], /*noiseScaleDeg=*/1, enc_lv, nullptr, 0, stream);
        else
            result[i] = inf.cc()->MakeCKKSPackedPlaintext(pt[i], /*noiseScaleDeg=*/1, enc_lv);
        if (cp.on) inf.cc()->MarkCoeffStaged(result[i], lv, cp.sf_target);
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
