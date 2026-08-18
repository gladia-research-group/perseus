#pragma once

#include "inference.h"
#include <string>
#include <vector>

namespace cachemir {

void      rotate_add_inplace(Inference& inf, PackedCtx& x, int step);
PackedCtx linear(Inference& inf, const PackedCtx& x,
                 const std::string& wname, int d_in, int d_out);

std::vector<PackedCtx> prepare_linear_input(Inference& inf, const PackedCtx& x,
                                            int d_in, int d_out);
PackedCtx apply_linear(Inference& inf, const std::vector<PackedCtx>& x_rotated,
                       const std::string& wname, int d_in, int d_out);

PackedCtx apply_linear_outputpack(Inference& inf, const std::vector<PackedCtx>& x_rotated,
                                  const std::string& wname, int d_in, int d_out);
PackedCtx linear_outputpack(Inference& inf, const PackedCtx& x,
                            const std::string& wname, int d_in, int d_out);

}  // namespace cachemir
