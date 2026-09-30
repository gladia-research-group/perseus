#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace periodic {

struct Params {
    int d = 0;       
    int slots = 0;   
    int T = 0;       
};

inline Params make_params(int d, int slots) {
    if (d <= 0 || slots <= 0)
        throw std::runtime_error("periodic::make_params: d and slots must be positive");
    if (slots % d != 0)
        throw std::runtime_error("periodic::make_params: slots must be a multiple of d (got d=" +
                                 std::to_string(d) + ", slots=" + std::to_string(slots) + ")");
    if ((d & (d - 1)) != 0)
        throw std::runtime_error("periodic::make_params: d must be a power of 2 for the subring "
                                 "argument to hold (got d=" + std::to_string(d) + ")");
    Params p;
    p.d = d;
    p.slots = slots;
    p.T = slots / d;
    return p;
}

inline int slot_of(int tok, int i, int d) { return tok * d + i; }

inline int slot_of_feature_major(int tok, int i, int t) { return i * t + tok; }

inline std::vector<double> expand(const std::vector<double>& period, int slots) {
    const int d = static_cast<int>(period.size());
    if (d == 0 || slots % d != 0)
        throw std::runtime_error("periodic::expand: slots must be a multiple of period length");
    std::vector<double> v(static_cast<size_t>(slots));
    for (int s = 0; s < slots; ++s) v[s] = period[s % d];
    return v;
}

inline bool is_periodic(const std::vector<double>& v, int d, double tol = 0.0) {
    const int n = static_cast<int>(v.size());
    if (d <= 0 || n % d != 0) return false;
    for (int s = d; s < n; ++s) {
        const double a = v[s], b = v[s % d];
        if (!(a == b || (a - b <= tol && b - a <= tol))) return false;
    }
    return true;
}

inline bool is_block_constant(const std::vector<double>& v, int blk) {
    const int n = static_cast<int>(v.size());
    if (blk <= 0 || n % blk != 0) return false;
    for (int s = 0; s < n; ++s)
        if (v[s] != v[(s / blk) * blk]) return false;
    return true;
}


inline std::vector<double> weight_diag_period(const std::vector<double>& W, int d, int k) {
    if (static_cast<int>(W.size()) != d * d)
        throw std::runtime_error("periodic::weight_diag_period: W must be d*d row-major");
    if (k < 0 || k >= d)
        throw std::runtime_error("periodic::weight_diag_period: k out of range");
    std::vector<double> diag(static_cast<size_t>(d));
    for (int i = 0; i < d; ++i) diag[i] = W[static_cast<size_t>(i) * d + ((i + k) % d)];
    return diag;
}

struct BlockRotTerm {
    int rot = 0;                        
    std::vector<double> mask_period;    
    bool empty = false;                 
};

inline std::vector<BlockRotTerm> block_rot_terms(int k, int d) {
    if (d <= 0) throw std::runtime_error("periodic::block_rot_terms: d must be positive");
    k = ((k % d) + d) % d;

    BlockRotTerm lo;
    lo.rot = k;
    lo.mask_period.assign(static_cast<size_t>(d), 0.0);
    for (int i = 0; i < d; ++i) if (i < d - k) lo.mask_period[i] = 1.0;

    BlockRotTerm hi;
    hi.rot = k - d;
    hi.mask_period.assign(static_cast<size_t>(d), 0.0);
    for (int i = 0; i < d; ++i) if (i >= d - k) hi.mask_period[i] = 1.0;
    hi.empty = (k == 0);   // k=0 is the identity: no wrap term, single plaintext

    return {lo, hi};
}

inline std::vector<BlockRotTerm> folded_diag_periods(const std::vector<double>& diag_period,
                                                     int k, int d) {
    if (static_cast<int>(diag_period.size()) != d)
        throw std::runtime_error("periodic::folded_diag_periods: diag_period must have length d");
    std::vector<BlockRotTerm> terms = block_rot_terms(k, d);
    for (auto& t : terms)
        for (int i = 0; i < d; ++i) t.mask_period[i] *= diag_period[i];
    return terms;
}

inline std::vector<double> rot_reference(const std::vector<double>& x, int r) {
    const int n = static_cast<int>(x.size());
    std::vector<double> out(static_cast<size_t>(n));
    for (int s = 0; s < n; ++s) out[s] = x[(((s + r) % n) + n) % n];
    return out;
}

inline std::vector<double> block_rot_reference(const std::vector<double>& x, int k, int d) {
    const int n = static_cast<int>(x.size());
    if (d <= 0 || n % d != 0)
        throw std::runtime_error("periodic::block_rot_reference: size must be a multiple of d");
    k = ((k % d) + d) % d;
    std::vector<double> out(static_cast<size_t>(n));
    for (int s = 0; s < n; ++s) {
        const int tok = s / d, i = s % d;
        out[s] = x[static_cast<size_t>(tok) * d + ((i + k) % d)];
    }
    return out;
}

inline std::vector<double> matvec_reference(const std::vector<double>& W,
                                            const std::vector<double>& x, int d) {
    const int n = static_cast<int>(x.size());
    if (n % d != 0) throw std::runtime_error("periodic::matvec_reference: size % d != 0");
    std::vector<double> y(static_cast<size_t>(n), 0.0);
    for (int tok = 0; tok < n / d; ++tok)
        for (int j = 0; j < d; ++j) {
            double acc = 0.0;
            for (int i = 0; i < d; ++i)
                acc += W[static_cast<size_t>(j) * d + i] * x[static_cast<size_t>(tok) * d + i];
            y[static_cast<size_t>(tok) * d + j] = acc;
        }
    return y;
}

inline int distinct_residue_bound(int d, int slots) {
    const int n_coeff = 2 * slots;                 // N
    const int bound = 2 * d;
    return bound < n_coeff ? bound : n_coeff;
}

inline double compression_ratio(int d, int slots) {
    return static_cast<double>(2 * slots) / static_cast<double>(distinct_residue_bound(d, slots));
}

}  // namespace periodic
