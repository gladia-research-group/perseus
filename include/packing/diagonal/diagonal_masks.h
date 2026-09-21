#pragma once

#include <stdexcept>
#include <vector>
        
namespace diagonal {

inline std::vector<int> stride_slots_diagonal(
    int slots, int d, int stride, int start_offset) {
    std::vector<int> out;
    out.reserve(static_cast<size_t>(d) * static_cast<size_t>(stride));
    for (int k = 0; k < d; ++k) {
        for (int tok = 0; tok < stride; ++tok) {
            const int s = start_offset + k * stride + tok;
            if (s < 0 || s >= slots)
                throw std::runtime_error(
                    "stride_slots_diagonal: slot index out of range");
            out.push_back(s);
        }
    }
    return out;
}

inline std::vector<int> active_expanded_slots_diagonal(
    int N, int d, int alpha, int e_real) {
    const int e_pad = alpha * d;
    const int t_out = N / e_pad;
    std::vector<int> out;
    out.reserve(static_cast<size_t>(e_real) * static_cast<size_t>(t_out));
    for (int j = 0; j < e_real; ++j) {
        for (int tok = 0; tok < t_out; ++tok) {
            const int s = j * t_out + tok;
            if (s < 0 || s >= N)
                throw std::runtime_error(
                    "active_expanded_slots_diagonal: slot index out of range");
            out.push_back(s);
        }
    }
    return out;
}

}  // namespace diagonal
