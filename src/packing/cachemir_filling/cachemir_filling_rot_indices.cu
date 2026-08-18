#include "packing/cachemir_filling/cachemir_filling_rot_indices.h"
#include "packing/diagonal/diagonal_rot_indices.h"

#include <set>

namespace cachemir_filling {

static void collect_mha_rots(std::set<int32_t>& rots, int slots, int hidDim, int numHeads) {
    const int t = (hidDim > 0) ? slots / hidDim : 0;
    if (t <= 0) return;
    const int tH = t * numHeads;

    for (int i = 1; i < t; ++i) {
        rots.insert(i);
        rots.insert(-i);
    }

    for (int s = tH; s < slots; s *= 2) {
        rots.insert(s);
        rots.insert(-s);
    }
}

std::vector<int32_t> compute_gpt2_rot_indices(int slots, int hidDim, int ffDim, int numHeads) {
    std::set<int32_t> rots;
    for (int r : diagonal::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads)) rots.insert(r);
    collect_mha_rots(rots, slots, hidDim, numHeads);
    return std::vector<int32_t>(rots.begin(), rots.end());
}

}  // namespace cachemir_filling
