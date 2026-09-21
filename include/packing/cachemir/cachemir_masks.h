#pragma once

#include <stdexcept>
#include <vector>

namespace cachemir {

inline std::vector<int> stride_slots_cachemir(
    int slots, int d, int stride, int start_offset) {
    std::vector<int> out;
    out.reserve(static_cast<size_t>(d));
    for (int i = 0; i < d; ++i) {
        const int s = i * stride + start_offset;
        if (s < 0 || s >= slots)
            throw std::runtime_error(
                "stride_slots_cachemir: slot index out of range");
        out.push_back(s);
    }
    return out;
}

inline std::vector<int> active_expanded_slots_cachemir(
    int N, int d, int alpha, int e_real) {
    const int e_pad = alpha * d;
    const int tp    = N / e_pad;
    const int M     = N / tp;       // = α·d = e_pad
    auto interleave = [&](int m) {
        const int a = e_pad / d;    // = α (when e_pad > d)
        return (m / a + (m % a) * d) % e_pad;
    };
    std::vector<int> out;
    out.reserve(static_cast<size_t>(e_real));
    for (int m = 0; m < M; ++m) {
        if (interleave(m) < e_real) {
            out.push_back(m * tp);
        }
    }
    return out;
}

}  // namespace cachemir
