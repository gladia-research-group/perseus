#pragma once

#include "inference.h"

#include <vector>

namespace bidirectional {

std::vector<PackedCtx> delta_bi_attention(Inference& inf,
                                          std::vector<PackedCtx> qs,
                                          std::vector<PackedCtx> ks,
                                          std::vector<PackedCtx> vs,
                                          const std::vector<int>& ns);

PackedCtx complex_bi_attention(Inference& inf, PackedCtx q, PackedCtx k, PackedCtx v,
                               int nA, int nB);

}  // namespace bidirectional
