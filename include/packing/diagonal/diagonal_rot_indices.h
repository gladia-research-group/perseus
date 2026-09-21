#pragma once

#include <cstdint>
#include <vector>

// Diagonal-packing rotation indices needed by KeyGen for a GPT-2 block.

namespace diagonal {

std::vector<int32_t> compute_gpt2_rot_indices(
    int slots, int hidDim, int ffDim, int numHeads);

}  // namespace diagonal
