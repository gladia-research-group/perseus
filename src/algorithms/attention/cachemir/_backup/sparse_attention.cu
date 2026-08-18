#include "packing/cachemir/sparse_attention.h"

#include "ckks_primitives.h"    // goldschmidt_inv, goldschmidt_recip
#include "fideslib_wrapper.h"   // SparseBtsScope
#include "inference.h"

#include <cstdlib>

// SPARSE_SM_BTS gate (cached).
bool sparse_sm_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("SPARSE_SM_BTS");
        return v && *v && *v != '0';
    }();
    return on;
}

namespace cachemir {

// THOR softmax's z/s reciprocal, dispatched at the call boundary (the softmax body
// stays unified — it just calls this):
//   - dense (default): goldschmidt_inv(z,s,F) — the exact current graph, so the
//     planned decode is byte-identical when SPARSE_SM_BTS is off.
//   - sparse: r = 1/s via goldschmidt_recip on the BROADCAST track (s =
//     head_reduce_sum, period t·H = 512) routed to the sparse precomp, then z·r
//     multiplies the vector numerator back in OUTSIDE the scope. Value-identical +
//     depth-neutral to the dense form; F_init / divergence-cliff band untouched.
//     (recip changes graph node names -> planned decode needs a fresh capture+plan.)
PackedCtx softmax_recip(Inference& inf, const PackedCtx& z, const PackedCtx& s,
                        const PackedCtx& F_init, int iters) {
    if (!sparse_sm_enabled())
        return goldschmidt_inv(inf.cc_ctx(), z, s, F_init, iters);

    PackedCtx recip;
    {
        CKKSContext::SparseBtsScope ss(*inf.fhe);
        recip = goldschmidt_recip(inf.cc_ctx(), s, F_init, iters);
    }
    return inf.fhe->mult(z, recip);   // z (vector) · 1/s (broadcast) — outside the scope
}

}  // namespace cachemir
