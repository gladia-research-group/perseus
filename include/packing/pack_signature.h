#pragma once

#include "ckks_types.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

struct PackSignature {
    int slots = 0;
    double amax = 0.0;        // max |Re|
    double im_amax = 0.0;     // max |Im|, a nonzero here means the Im lane carries data
    int live = 0;             
    int window = 0;           
    int stride = 0;           
    int period = 0;           
    bool im_is_data = false;  
    int fold_s = 0;           
    double folded_amax = 0.0; 
    int live_exact = 0;
    int stride_exact = 0;
    int window_exact = 0;
    int offset_exact = 0;
};

inline PackSignature analyze_packing(const std::vector<std::complex<double>>& v) {
    PackSignature s;
    s.slots = (int)v.size();
    if (s.slots == 0) return s;

    for (const auto& z : v) {
        s.amax = std::max(s.amax, std::fabs(z.real()));
        s.im_amax = std::max(s.im_amax, std::fabs(z.imag()));
    }
    s.im_is_data = s.im_amax > 1e-3 * std::max(s.amax, 1e-300);
    const double ref = s.im_is_data ? std::max(s.amax, s.im_amax) : s.amax;
    constexpr double rel_tol = 1e-2;   // live-slot threshold relative to the max magnitude
    const double tol = std::max(rel_tol * ref, 1e-12);

    std::vector<int> idx;
    idx.reserve(64);
    for (int i = 0; i < s.slots; ++i)
        if (std::fabs(v[i].real()) > tol || (s.im_is_data && std::fabs(v[i].imag()) > tol))
            idx.push_back(i);
    s.live = (int)idx.size();
    s.window = idx.empty() ? 0 : idx.back() + 1;

    if (idx.size() >= 2) {
        int g = 0;
        for (size_t k = 1; k < idx.size(); ++k) g = std::gcd(g, idx[k] - idx[0]);
        s.stride = g;
    } else if (idx.size() == 1) {
        s.stride = s.slots;
    }

    s.period = s.slots;
    for (int p = 1; p < s.slots; p <<= 1) {
        bool ok = true;
        for (int i = p; i < s.slots && ok; ++i) {
            if (std::fabs(v[i].real() - v[i % p].real()) > tol ||
                (s.im_is_data && std::fabs(v[i].imag() - v[i % p].imag()) > tol))
                ok = false;
        }
        if (ok) { s.period = p; break; }
    }

    s.fold_s = s.slots;
    if (!idx.empty()) {
        std::vector<char> seen;
        for (int p = 1; p < s.slots; p <<= 1) {
            seen.assign((size_t)p, 0);
            bool ok = true;
            for (int i : idx) {
                const int r = i % p;
                if (seen[(size_t)r]) { ok = false; break; }
                seen[(size_t)r] = 1;
            }
            if (ok) { s.fold_s = p; break; }
        }
    }
    if (s.fold_s > 0)
        s.folded_amax = s.amax * (double)s.fold_s / (double)s.slots;

    {
        std::vector<int> ex;
        ex.reserve(64);
        for (int i = 0; i < s.slots; ++i)
            if (v[i].real() != 0.0 || v[i].imag() != 0.0) ex.push_back(i);
        s.live_exact = (int)ex.size();
        s.window_exact = ex.empty() ? 0 : ex.back() + 1;
        s.offset_exact = ex.empty() ? 0 : ex.front();
        if (ex.size() >= 2) {
            int g = 0;
            for (size_t k = 1; k < ex.size(); ++k) g = std::gcd(g, ex[k] - ex[0]);
            s.stride_exact = g;
        } else if (ex.size() == 1) {
            s.stride_exact = s.slots;
        }
    }

    return s;
}

inline PackSignature analyze_packing(const std::vector<double>& v) {
    std::vector<std::complex<double>> c(v.size());
    for (size_t i = 0; i < v.size(); ++i) c[i] = {v[i], 0.0};
    return analyze_packing(c);
}


// The fields of analyze_packing a plaintext's pack tag reads (period, live_exact, stride_exact, window_exact,
// offset_exact), in two allocation-free passes over a raw row of n values (re[i * step], im[i * step] when im):
// the streamed-weight path tags ~150k rows per token, where analyze_packing's index vectors and complex copies
// dominated. Same semantics: the period uses the 1e-2 relative tolerance, the exact support every nonzero.
template <class T>
inline PackSignature analyze_tag_fields(const T* re, const T* im, int n, int step) {
    PackSignature s;
    s.slots = n;
    if (n == 0) return s;
    double amax = 0.0, imax = 0.0;
    int first = -1, last = -1, live = 0, g = 0;
    for (int i = 0; i < n; ++i) {
        const double a = (double)re[(size_t)i * step], b = im ? (double)im[(size_t)i * step] : 0.0;
        amax = std::max(amax, std::fabs(a));
        imax = std::max(imax, std::fabs(b));
        if (a != 0.0 || b != 0.0) {
            if (first < 0) first = i; else g = std::gcd(g, i - first);
            last = i;
            ++live;
        }
    }
    s.amax = amax; s.im_amax = imax;
    s.im_is_data = imax > 1e-3 * std::max(amax, 1e-300);
    const double ref = s.im_is_data ? std::max(amax, imax) : amax;
    const double tol = std::max(1e-2 * ref, 1e-12);
    s.period = n;
    for (int p = 1; p < n; p <<= 1) {
        bool ok = true;
        for (int i = p; i < n && ok; ++i) {
            const size_t a = (size_t)i * step, b = (size_t)(i % p) * step;
            if (std::fabs((double)re[a] - (double)re[b]) > tol ||
                (s.im_is_data && im && std::fabs((double)im[a] - (double)im[b]) > tol))
                ok = false;
        }
        if (ok) { s.period = p; break; }
    }
    s.live_exact = live;
    s.window_exact = live ? last + 1 : 0;
    s.offset_exact = live ? first : 0;
    s.stride_exact = live >= 2 ? g : (live == 1 ? n : 0);
    return s;
}
