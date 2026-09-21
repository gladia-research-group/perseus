#include "packing/diagonal/diagonal_linear_utils.h"
#include "inference.h"
#include "io/weight_io.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace diagonal {

DiagonalParams compute_dg_params(int N, int d_in, int d_out) {
    if (d_in <= 0 || d_out <= 0)
        throw std::runtime_error("diagonal::compute_dg_params: d_in/d_out must be positive");
    const int d_max = std::max(d_in, d_out);
    const int d_min = std::min(d_in, d_out);
    if (d_max % d_min != 0)
        throw std::runtime_error("diagonal::compute_dg_params: max(d_in, d_out) must be a multiple of min");
    if (N % d_max != 0)
        throw std::runtime_error("diagonal::compute_dg_params: N must be a multiple of max(d_in, d_out)");

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

PackedCtx encode_linear_input(Inference& inf, const std::vector<double>& x,
                               int d_in, int d_out, int target_level) {
    auto p  = compute_dg_params(inf.slots, d_in, d_out);
    const int N = inf.slots;
    if (x.size() % static_cast<size_t>(d_in) != 0)
        throw std::runtime_error("diagonal::encode_linear_input: x.size() not a multiple of d_in");
    const int n_tok = static_cast<int>(x.size()) / d_in;
    if (n_tok < 1 || n_tok > p.max_n_tok)
        throw std::runtime_error("diagonal::encode_linear_input: n_tok out of [1, max_n_tok]");
    inf.n_tok = n_tok;  // norm() reads this to floor the empty token lanes

    std::vector<double> ptx(N, 0.0);
    for (int tok = 0; tok < n_tok; ++tok) {
        for (int i = 0; i < p.d_in; ++i) {
            const double val = x[tok * d_in + i];
            const int base = i * p.t_in;
            ptx[base + tok] = val;
        }
    }
    Ctx ct = encrypt(
        inf.cc(),
        inf.cc()->MakeCKKSPackedPlaintext(ptx, /*noiseScaleDeg=*/1,
                                          static_cast<uint32_t>(target_level)),
        inf.fhe->pk());
    return inf.pack(ct, inf.packing.kind);
}

namespace {

// Same rule and the same polarity as the cachemir path (cachemir_linear_utils.cu): one
// name must not mean opposite things in one process.
bool coeff_encode_enabled() {
    const char* e = std::getenv("FHE_PT_COEFF_ENCODE");
    return e && *e && std::atoi(e) == 1;
}
}  // namespace

std::vector<Ptx> encode_weight_matrix(Inference& inf,
                                       const std::vector<std::vector<double>>& W,
                                       int d_in, int d_out, int target_level) {
    const int N = inf.slots;
    auto p = compute_dg_params(N, d_in, d_out);
    const uint32_t lv = static_cast<uint32_t>(target_level);

    bool coeff = coeff_encode_enabled() && static_cast<bool>(inf.pt_stage_hook);

    const uint32_t lv1 = static_cast<uint32_t>(
        inf.fhe->composite_degree * inf.fhe->total_depth);
    double ratio = 1.0, sf_target = 0.0;
    if (coeff) {
        sf_target = inf.cc()->ScalingFactorReal(lv);
        ratio     = sf_target / inf.cc()->ScalingFactorReal(lv1);
        double mx = 0.0;
        for (const auto& row : W)
            for (double x : row) mx = std::max(mx, std::abs(x));

        const double lift_bits = (double)inf.fhe->first_mod_bits;
        if (mx * sf_target >= 0.45 * std::pow(2.0, lift_bits)) {
            // Past the centered-lift bound: this matrix class takes the full encode.
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::cerr << "[coeff_encode] hybrid: |w|max=" << mx
                          << " exceeds the centered-lift bound -> this matrix class "
                             "falls back to the full encode\n";
            }
            coeff = false;
            ratio = 1.0;
        }
    }

    std::vector<Ptx> result(static_cast<size_t>(p.s) * p.G);

    #pragma omp parallel
    {
        std::vector<double> diag(N, 0.0);
        std::vector<double> shifted(N, 0.0);
        #pragma omp for collapse(2) schedule(static)
        for (int g = 0; g < p.G; ++g) {
            for (int b = 0; b < p.s; ++b) {
                const int k = g * p.s + b;
                for (int j = 0; j < p.d_out; ++j) {
                    const int base_j = p.is_up ? (j / p.alpha) : (j * p.alpha);
                    const int row    = (base_j + k) % p.d_in;
                    const double val = W[row][j] * ratio;   // ratio==1 outside coeff mode
                    const int slot0  = j * p.t_out;
                    for (int tok = 0; tok < p.t_out; ++tok)
                        diag[slot0 + tok] = val;
                }
                const int shift = static_cast<int>(
                    (static_cast<long long>(g) * p.s * p.t_in) % N);
                if (shift != 0)
                    for (int i = 0; i < N; ++i)
                        shifted[(i + shift) % N] = diag[i];
                Ptx pt = inf.cc()->MakeCKKSPackedPlaintext(
                    shift == 0 ? diag : shifted, /*noiseScaleDeg=*/1, coeff ? lv1 : lv);
                if (coeff) inf.cc()->MarkCoeffStaged(pt, lv, sf_target, /*prescale_log2=*/0);
                result[b * p.G + g] = std::move(pt);
            }
        }
    }
    return result;
}

std::vector<Ptx> encode_weight_matrix(Inference& inf,
                                       const std::vector<std::vector<double>>& W,
                                       int d_in, int d_out, int target_level,
                                       cudaStream_t stream) {
    (void)stream;   // encode is host-side; the stream overload only exists for API symmetry
    return encode_weight_matrix(inf, W, d_in, d_out, target_level);
}

Ptx encode_bias_vector(Inference& inf, const std::vector<double>& b,
                        int d_in, int d_out, bool fill) {
    const int N = inf.slots;
    auto p = compute_dg_params(N, d_in, d_out);
    std::vector<double> pt(N, 0.0);
    const int dim = std::min<int>(p.d_out, static_cast<int>(b.size()));
    for (int j = 0; j < dim; ++j) {
        if (fill) {
            const int base = j * p.t_out;
            for (int tok = 0; tok < p.t_out; ++tok)
                pt[base + tok] = b[j];
        } else {
            pt[j * p.t_out] = b[j];
        }
    }
    return inf.cc()->MakeCKKSPackedPlaintext(pt);
}

Ptx encode_bias_vector(Inference& inf, const std::vector<double>& b,
                        int d_in, int d_out, bool fill,
                        cudaStream_t stream) {
    const int N = inf.slots;
    auto p = compute_dg_params(N, d_in, d_out);
    std::vector<double> pt(N, 0.0);
    const int dim = std::min<int>(p.d_out, static_cast<int>(b.size()));
    for (int j = 0; j < dim; ++j) {
        if (fill) {
            const int base = j * p.t_out;
            for (int tok = 0; tok < p.t_out; ++tok)
                pt[base + tok] = b[j];
        } else {
            pt[j * p.t_out] = b[j];
        }
    }
    if (stream != nullptr)
        return inf.cc()->MakeCKKSPackedPlaintext(pt, /*noiseScaleDeg=*/1, 0, nullptr, 0, stream);
    return inf.cc()->MakeCKKSPackedPlaintext(pt);
}

std::vector<Ptx> load_weight_txt(Inference& inf, const std::string& path,
                                  int d_in, int d_out, int target_level) {
    return diagonal::encode_weight_matrix(
        inf, weight_io::load_matrix_txt(path, d_in, d_out), d_in, d_out, target_level);
}

std::vector<double> decode_linear_output(int slots, const std::vector<double>& cy,
                                         int d_in, int d_out) {
    auto p = compute_dg_params(slots, d_in, d_out);
    std::vector<double> y(d_out);
    for (int j = 0; j < d_out; ++j) y[j] = cy[j * p.t_out];
    return y;
}

std::vector<std::vector<double>> decode_tokens(int slots, const std::vector<double>& cy,
                                               int d_pad, int d_real, int T) {
    const int t = compute_dg_params(slots, d_pad, d_pad).t_out;   // = slots/d_pad
    std::vector<std::vector<double>> y(T, std::vector<double>(d_real));
    for (int tok = 0; tok < T; ++tok)
        for (int i = 0; i < d_real; ++i)
            y[tok][i] = cy[static_cast<size_t>(i) * t + tok];
    return y;
}

}  // namespace diagonal
