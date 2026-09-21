#include "packing/diagonal/diagonal_rot_indices.h"
#include "packing/diagonal/diagonal_linear_utils.h"

#include <set>

namespace diagonal {

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

std::vector<int32_t> compute_gpt2_rot_indices(
    int slots, int hidDim, int ffDim, int /*numHeads*/) {
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

}  // namespace diagonal
