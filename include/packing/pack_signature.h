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
