#pragma once

#include "inference.h"
#include "packing/packed_ctx.h"

#include <cstdlib>
#include <cuda_runtime.h>
#include <string>
#include <vector>

inline bool fold_ln_affine(const std::string& tag) {
    auto off = [](const char* e) { return e && e[0] == '0' && e[1] == '\0'; };
    const char* g = std::getenv("GPT2_FOLD_LN_AFFINE");
    bool def = (g != nullptr) && !off(g);
    const char* per = nullptr;
    if      (tag == "ln_1") per = std::getenv("GPT2_FOLD_LN1");
    else if (tag == "ln_2") per = std::getenv("GPT2_FOLD_LN2");
    else if (tag == "ln_f") per = std::getenv("GPT2_FOLD_LNF");
    return per ? !off(per) : def;
}

Ptx encode_ln_affine_param(Inference& inf, const std::vector<double>& v,
                           int d_pad, int C_real, int target_level,
                           bool mask_inactive = false,
                           cudaStream_t stream = nullptr);

PackedCtx ln_affine(Inference& inf, const PackedCtx& normed, const std::string& tag);

PackedCtx layer_norm(Inference& inf, const PackedCtx& x, const std::string& cfg_name);
