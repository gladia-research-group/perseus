#pragma once

#include "inference.h"
#include "op_sequence.h"
#include "packing/packed_ctx.h"

std::vector<Op> mlp_ops();
std::vector<Op> mlp_tiled_ops(int n_tiles);
PackedCtx       mlp_block(Inference& inf, PackedCtx& x);
