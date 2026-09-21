#pragma once

#include <cstdint>
#include <vector>

// Cachemir-packing rotation indices needed by KeyGen for a GPT-2 block.

namespace cachemir {

std::vector<int32_t> compute_gpt2_rot_indices(
    int slots, int hidDim, int ffDim, int numHeads);

std::vector<int32_t> linear_rot_indices(int N, int d_in, int d_out);

}  // namespace cachemir
