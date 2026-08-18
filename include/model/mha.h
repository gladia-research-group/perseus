#pragma once

#include "inference.h"
#include "op_sequence.h"
#include "packing/packed_ctx.h"

// The MHA as a sequential op list (nn.Sequential-style): qkv (one fan-out op that
// caches k/v and yields q), qk^T + softmax + softmax·V (weight-free overlap window),
// out-proj. q/k/v share the input, so they're one op; the attention core needs no
// weights, so it's where the out-proj weights can be prefetched.
std::vector<Op> mha_ops();
std::vector<Op> mha_ops_token_pair();   // token-pair (2t-tok/chunk) MHA op-list; selected in gpt2_block_ops
PackedCtx       mha_block(Inference& inf, PackedCtx& x);
