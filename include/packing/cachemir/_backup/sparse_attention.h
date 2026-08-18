#pragma once

#include "packing/packed_ctx.h"

struct Inference;

bool sparse_sm_enabled();

namespace cachemir {

PackedCtx softmax_recip(Inference& inf, const PackedCtx& z, const PackedCtx& s,
                        const PackedCtx& F_init, int iters);

}  // namespace cachemir
