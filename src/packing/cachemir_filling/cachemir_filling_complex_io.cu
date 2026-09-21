#include "packing/cachemir_filling/cachemir_filling.h"
#include "inference.h"

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <vector>

namespace cachemir_filling {

PackedCtx encode_input_token_pair(Inference& inf,
                                  const std::vector<std::vector<double>>& embeddings,
                                  int target_level) {
    const int N      = inf.slots;
    const int d_pad  = inf.size.hidDim;
    const int d_real = inf.size.getRealHidDim();
    const int t      = N / d_pad;                     // token stride (=32 @ logN16)
    const int T      = static_cast<int>(embeddings.size());
    if (t <= 0 || 2 * t > N)
        throw std::runtime_error("encode_input_token_pair: bad token stride");
    if (T > 2 * t)
        throw std::runtime_error("encode_input_token_pair: chunk exceeds 2*t tokens");
    const int nA = std::min(t, T);                    // real-lane tokens [0, t)
    const int nB = std::max(0, T - t);                // imag-lane tokens [t, 2t)

    std::vector<std::complex<double>> ptx(N, {0.0, 0.0});
    for (int tok = 0; tok < nA; ++tok) {
        const int n = std::min<int>(d_real, static_cast<int>(embeddings[tok].size()));
        for (int i = 0; i < n; ++i)
            ptx[static_cast<size_t>(i) * t + tok].real(embeddings[tok][i]);
    }
    for (int tok = 0; tok < nB; ++tok) {
        const int n = std::min<int>(d_real, static_cast<int>(embeddings[t + tok].size()));
        for (int i = 0; i < n; ++i)
            ptx[static_cast<size_t>(i) * t + tok].imag(embeddings[t + tok][i]);
    }

    inf.n_tok      = nA;   // physical lane count (A half; B rides the same lanes)
    inf.n_tok_imag = nB;   // B-half active count -- read by the per-half token-pair wrappers

    Ctx ct = encrypt(
        inf.cc(),
        inf.cc()->MakeCKKSPackedPlaintext(ptx, /*noiseScaleDeg=*/1,
                                          static_cast<uint32_t>(target_level)),
        inf.fhe->pk());
    return inf.pack(ct, inf.packing.kind);
}

}  // namespace cachemir_filling
