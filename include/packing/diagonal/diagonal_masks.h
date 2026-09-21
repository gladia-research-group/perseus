#pragma once

#include <stdexcept>
#include <vector>

// Diagonal-packing slot selectors used to build plaintext masks.
//
// In the diagonal layout a logical feature k does not occupy a single slot but
// a contiguous lane-block [k*stride, (k+1)*stride): the `stride` lanes carry the
// independent tokens of the prefill batch (stride == slots/hidDim == the
// per-feature lane width). A mask over feature k must therefore cover the whole
// block, not just its token-0 anchor — this is the multi-token counterpart of
// the cachemir helpers, which select only the anchor k*stride.

namespace diagonal {

// All token lanes of the first `d` feature blocks: {start_offset + k*stride + tok
// : k in [0,d), tok in [0,stride)}. For square layouts (stride == lane width)
// this is the contiguous range [start_offset, start_offset + d*stride).
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

// All token lanes of the first `e_real` expanded-feature blocks. The up-proj
// output lives in the standard diagonal layout for d_out = alpha*d (= e_pad),
// feature j at j*t_out + tok with t_out = N/e_pad — no cachemir interleave.
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
