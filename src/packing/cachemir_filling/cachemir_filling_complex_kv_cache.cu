#include "packing/cachemir_filling/cachemir_filling_attention.h"
#include "inference.h"

#include <complex>
#include <string>
#include <vector>

namespace cachemir_filling {

static Ptx active_token_mask_imag(Inference& inf, const PackedCtx& ref) {
    const int N  = inf.slots;
    const int t  = N / inf.size.hidDim;
    const int tH = t * inf.size.numHeads;
    const int d_head_real = inf.size.getRealDHead();
    const int H_real      = inf.size.getRealNumHeads();
    const int n_tok       = inf.n_tok;
    return inf.encode_at_cached_complex(
        "cf.kvpush.mask.q.im.nt" + std::to_string(n_tok), ref,
        [&] {
            std::vector<std::complex<double>> mask(N, {0.0, 0.0});
            for (int c = 0; c < d_head_real; ++c)
                for (int h = 0; h < H_real; ++h)
                    for (int i = 0; i < n_tok; ++i)
                        mask[c * tH + h * t + i] = {0.0, -0.25};   // -0.25i; 2 push cleanses realify to Im
            return mask;
        });
}

static PackedCtx fresh_masked_push_imag(Inference& inf, const PackedCtx& x) {
    Ptx mask_pt = active_token_mask_imag(inf, x);   // named lvalue: mult takes Ptx&
    PackedCtx masked = inf.fhe->mult(x, mask_pt);
    inf.fhe->inplace_im_cleanse(masked);
    inf.fhe->bootstrap(masked.ct);
    inf.fhe->inplace_im_cleanse(masked);
    return masked;
}

void cache_k_push_imag(Inference& inf, const PackedCtx& key) {
    WithStep _w(inf, "cf.cache_k_push.im");
    inf.cache[inf.scoped("k")].push_back(fresh_masked_push_imag(inf, key));
    inf.k_count() += inf.n_tok;
}

void cache_v_push_imag(Inference& inf, const PackedCtx& value) {
    WithStep _w(inf, "cf.cache_v_push.im");
    inf.cache[inf.scoped("v")].push_back(fresh_masked_push_imag(inf, value));
    inf.v_count() += inf.n_tok;
}

}  // namespace cachemir_filling
