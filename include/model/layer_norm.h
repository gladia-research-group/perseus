#pragma once

#include "inference.h"
#include "packing/packed_ctx.h"

#include <cstdlib>
#include <cuda_runtime.h>
#include <string>
#include <vector>

// Per-LN affine folding. Global GPT2_FOLD_LN_AFFINE sets the default (default ON);
// per-LN GPT2_FOLD_LN1 / GPT2_FOLD_LN2 / GPT2_FOLD_LNF (0/1) override for that tag.
inline bool fold_ln_affine(const std::string& tag) {
    auto off = [](const char* e) { return e && e[0] == '0' && e[1] == '\0'; };
    const char* g = std::getenv("GPT2_FOLD_LN_AFFINE");
    bool def = (g != nullptr) && !off(g);   // default OFF (opt-in); folding is net-negative in eager
    const char* per = nullptr;
    if      (tag == "ln_1") per = std::getenv("GPT2_FOLD_LN1");
    else if (tag == "ln_2") per = std::getenv("GPT2_FOLD_LN2");
    else if (tag == "ln_f") per = std::getenv("GPT2_FOLD_LNF");
    return per ? !off(per) : def;
}

// mask_inactive: zero the inactive (tok >= inf.n_tok) token lanes after packing,
// for diagonal / cachemir_filling. Use for the LN bias (added into the stream),
// not the weight. Reads inf.n_tok, so the encode must happen after the input set it.
Ptx encode_ln_affine_param(Inference& inf, const std::vector<double>& v,
                           int d_pad, int C_real, int target_level,
                           bool mask_inactive = false,
                           cudaStream_t stream = nullptr);

PackedCtx ln_affine(Inference& inf, const PackedCtx& normed, const std::string& tag);

PackedCtx layer_norm(Inference& inf, const PackedCtx& x, const std::string& cfg_name);
