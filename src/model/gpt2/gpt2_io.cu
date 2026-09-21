#include "model/gpt2.h"
#include "slot_layout.h"
#include "packing/cachemir_filling/cachemir_filling.h"

#include <algorithm>
#include <vector>


PackedCtx pack_tokens(Inference& inf, const std::vector<std::vector<double>>& embeddings,
                      int target_level) {
    const int T      = static_cast<int>(embeddings.size());
    const int d_pad  = inf.size.hidDim;
    const int d_real = inf.size.getRealHidDim();
    std::vector<double> flat(static_cast<size_t>(T) * d_pad, 0.0);
    for (int tok = 0; tok < T; ++tok) {
        const int n = std::min<int>(d_real, static_cast<int>(embeddings[tok].size()));
        for (int i = 0; i < n; ++i)
            flat[static_cast<size_t>(tok) * d_pad + i] = embeddings[tok][i];
    }
    return encode_linear_input(inf, flat, d_pad, d_pad, target_level);   // dispatched per packing
}

std::vector<std::vector<double>> unpack_tokens(Inference& inf, const PackedCtx& pc, int T) {
    auto raw = decrypt(inf.cc(), pc.ct, inf.fhe->sk());
    return decode_tokens(inf.packing, raw, inf.slots,
                         inf.size.hidDim, inf.size.getRealHidDim(), T);
}

// ---- named entry points (thin wrappers) ----

PackedCtx encode_token_input(Inference& inf, const std::vector<double>& x_real) {
    PackedCtx pc = pack_tokens(inf, {x_real},
                               static_cast<int>(inf.fhe->bootstrap_output_level()));
    slotlayout::set(pc.ct, slotlayout::Kind::Token);   // fresh-encode feature order
    return pc;
}

PackedCtx encode_prefill_input(Inference& inf,
                               const std::vector<std::vector<double>>& embeddings) {
    if (inf.token_pair)
        return cachemir_filling::encode_input_token_pair(
            inf, embeddings, static_cast<int>(inf.fhe->bootstrap_output_level()));
    return pack_tokens(inf, embeddings,
                       static_cast<int>(inf.fhe->bootstrap_output_level()));
}

std::vector<double> decode_token_output(Inference& inf, const PackedCtx& pc) {
    return unpack_tokens(inf, pc, 1)[0];
}

std::vector<std::vector<double>> decode_tokens_output(Inference& inf,
                                                      const PackedCtx& pc, int T) {
    return unpack_tokens(inf, pc, T);
}
