#pragma once

#include "inference.h"
#include "op_sequence.h"
#include "packing/packed_ctx.h"

std::vector<Op> mha_ops();
std::vector<Op> mha_ops_token_pair();
PackedCtx       mha_block(Inference& inf, PackedCtx& x);
