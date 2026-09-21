#include "packing/cachemir_filling/cachemir_filling_attention.h"
#include "inference.h"

#include <string>
#include <utility>
#include <vector>


namespace cachemir_filling {

void prepare_mha_masks(Inference& inf) {
    inf.cache[inf.scoped("k")] = {};
    inf.k_count() = 0;
}

static Ptx active_token_mask(Inference& inf, const PackedCtx& ref) {
    const int N  = inf.slots;
    const int t  = N / inf.size.hidDim;
    const int tH = t * inf.size.numHeads;
    const int d_head_real = inf.size.getRealDHead();
    const int H_real      = inf.size.getRealNumHeads();
    const int n_tok       = inf.n_tok;
    return inf.encode_at_cached(
        "cf.kvpush.mask.q.nt" + std::to_string(n_tok), ref,
        [&] {
            std::vector<double> mask(N, 0.0);
            for (int c = 0; c < d_head_real; ++c)
                for (int h = 0; h < H_real; ++h)
                    for (int i = 0; i < n_tok; ++i)
                        mask[c * tH + h * t + i] = 0.25;   // 2 push cleanses below double it back
            return mask;
        });
}

static PackedCtx fresh_masked_push(Inference& inf, const PackedCtx& x) {
    Ptx mask_pt = active_token_mask(inf, x);   // named lvalue: mult takes Ptx&
    PackedCtx masked = inf.fhe->mult(x, mask_pt);
    inf.fhe->inplace_im_cleanse(masked);
    inf.fhe->bootstrap(masked.ct);
    inf.fhe->inplace_im_cleanse(masked);
    return masked;
}

void cache_k_push(Inference& inf, const PackedCtx& key) {
    WithStep _w(inf, "cf.cache_k_push");
    inf.cache[inf.scoped("k")].push_back(fresh_masked_push(inf, key));
    inf.k_count() += inf.n_tok;
}

void prepare_vcache(Inference& inf) {
    inf.cache[inf.scoped("v")] = {};
    inf.v_count() = 0;
}

void cache_v_push(Inference& inf, const PackedCtx& value) {
    WithStep _w(inf, "cf.cache_v_push");
    inf.cache[inf.scoped("v")].push_back(fresh_masked_push(inf, value));
    inf.v_count() += inf.n_tok;
}

}  // namespace cachemir_filling
