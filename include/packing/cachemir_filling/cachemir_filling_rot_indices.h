#pragma once

#include <cstdint>
#include <vector>

namespace cachemir_filling {

std::vector<int32_t> compute_gpt2_rot_indices(int slots, int hidDim, int ffDim, int numHeads);

}  // namespace cachemir_filling
