#pragma once
#include "fideslib_wrapper.h"
#include <functional>
#include <algorithm>
#include <vector>

struct Inference;

struct CKKSFHECtx {
    std::shared_ptr<CKKSContext> ckks;  // FIDESlib context + keys
    int slots;                           // N/2

    CC&       cc()       { return ckks->cc; }
    const CC& cc() const { return ckks->cc; }

    PublicKey<DCRTPoly>&  pk() { return ckks->keys.publicKey; }
    PrivateKey<DCRTPoly>& sk() { return ckks->keys.secretKey; }
};

Ctx inv_sqrt_newton(CKKSContext& cc, const Ctx& x, const Ctx& ans_init, int iters, double x_scale = 1.0,
                    int real_d = 0, int real_stride = 0);

Ctx goldschmidt_inv_sqrt(CKKSContext& cc, const Ctx& x, const Ctx& ans_init, int iters);
Ctx goldschmidt_inv(CKKSContext& cc, const Ctx& a, const Ctx& x0_init, int iters);

// Repeated squaring, and Newton's iteration for 1/dnm from an initial guess.
Ctx exp_squaring(CKKSContext& cc, Ctx x, int iters);
Ctx newton_inverse(CKKSContext& cc, const Ctx& res, Ctx dnm, int iters);

// sparse_df routes the D_neg/F-track bootstraps to the sparse precomp — arm ONLY where
// D_init is proven slot-periodic (decode softmax s); the N-track always bootstraps dense.
Ctx goldschmidt_inv(CKKSContext& cc, const Ctx& N_init, const Ctx& D_init, const Ctx& F_init,
                    int iters, bool sparse_df = false);

Ctx goldschmidt_recip(CKKSContext& cc, const Ctx& D_init, const Ctx& F_init, int iters,
                      bool sparse_df = false);

Ctx eval_polynomial(CKKSContext& cc, const Ctx& x, const std::vector<double>& coeffs);

Ctx eval_polynomial_ps(CKKSContext& cc,
                           const Ctx& x,
                           const std::vector<double>& coeffs,
                           size_t slots);

Ctx eval_polynomial_deg4(CKKSContext& cc, const Ctx& x, const std::vector<double>& coeffs);

Ctx eval_polynomial_deg8(CKKSContext& cc, const Ctx& x, const std::vector<double>& coeffs);

Ctx eval_chebyshev_series(CKKSContext& cc, const Ctx& x,
                          const std::vector<double>& coeffs, double a, double b);

Ctx eval_remez_31(CKKSContext& cc, const Ctx& x, const std::vector<double>& Ncoeffs, const std::vector<double>& Dcoeffs, double alpha, double beta, int gs_iters);

std::vector<double> taylor_inv_sqrt_coeffs(double z0);

Ctx eval_taylor_inv_sqrt(CKKSContext& cc, const Ctx& x,
                          const std::vector<double>& coeffs, double z0);

Ctx eval_linear_wsum(CKKSContext& cc,
                     std::vector<Ctx>& cts,
                     const std::vector<double>& weights);

Ctx mask_slots(CKKSContext& cc, const Ctx& x, int slots, int active_dim);

//  PackedCtx overlays, to support both classical ctx operations and packing-aware.

inline PackedCtx inv_sqrt_newton(CKKSContext& cc, const PackedCtx& x, const PackedCtx& init, int iters, double x_scale = 1.0,
                                 int real_d = 0, int real_stride = 0) {
    assert_same_packing(x.packing, init.packing);
    return PackedCtx{inv_sqrt_newton(cc, x.ct, init.ct, iters, x_scale, real_d, real_stride), x.packing};
}

inline PackedCtx goldschmidt_inv(CKKSContext& cc, const PackedCtx& a, const PackedCtx& x0_init, int iters) {
    assert_same_packing(a.packing, x0_init.packing);
    return PackedCtx{goldschmidt_inv(cc, a.ct, x0_init.ct, iters), a.packing};
}

inline PackedCtx goldschmidt_inv(CKKSContext& cc,
                                 const PackedCtx& N_init,
                                 const PackedCtx& D_init,
                                 const PackedCtx& F_init,
                                 int iters,
                                 bool sparse_df = false) {
    assert_same_packing(N_init.packing, D_init.packing);
    assert_same_packing(N_init.packing, F_init.packing);
    return PackedCtx{goldschmidt_inv(cc, N_init.ct, D_init.ct, F_init.ct, iters, sparse_df),
                     N_init.packing};
}

inline PackedCtx goldschmidt_recip(CKKSContext& cc, const PackedCtx& D_init,
                                   const PackedCtx& F_init, int iters) {
    assert_same_packing(D_init.packing, F_init.packing);
    return PackedCtx{goldschmidt_recip(cc, D_init.ct, F_init.ct, iters), D_init.packing};
}

inline PackedCtx goldschmidt_inv_sqrt(CKKSContext& cc, const PackedCtx& x, const PackedCtx& ans_init, int iters) {
    assert_same_packing(x.packing, ans_init.packing);
    return PackedCtx{goldschmidt_inv_sqrt(cc, x.ct, ans_init.ct, iters), x.packing};
}

inline PackedCtx eval_polynomial(CKKSContext& cc, const PackedCtx& x, const std::vector<double>& coeffs) {
    return PackedCtx{eval_polynomial(cc, x.ct, coeffs), x.packing};
}

inline PackedCtx eval_polynomial_ps(CKKSContext& cc, const PackedCtx& x, const std::vector<double>& coeffs, size_t slots) {
    return PackedCtx{eval_polynomial_ps(cc, x.ct, coeffs, slots), x.packing};
}

inline PackedCtx eval_polynomial_deg4(CKKSContext& cc, const PackedCtx& x, const std::vector<double>& coeffs) {
    return PackedCtx{eval_polynomial_deg4(cc, x.ct, coeffs), x.packing};
}

inline PackedCtx eval_polynomial_deg8(CKKSContext& cc, const PackedCtx& x, const std::vector<double>& coeffs) {
    return PackedCtx{eval_polynomial_deg8(cc, x.ct, coeffs), x.packing};
}

inline PackedCtx eval_chebyshev_series(CKKSContext& cc, const PackedCtx& x,
                                       const std::vector<double>& coeffs, double a, double b) {
    return PackedCtx{eval_chebyshev_series(cc, x.ct, coeffs, a, b), x.packing};
}

inline PackedCtx eval_remez_31(CKKSContext& cc, const PackedCtx& x,
                               const std::vector<double>& Ncoeffs, const std::vector<double>& Dcoeffs,
                               double alpha, double beta, int gs_iters) {
    return PackedCtx{eval_remez_31(cc, x.ct, Ncoeffs, Dcoeffs, alpha, beta, gs_iters), x.packing};
}

inline PackedCtx eval_taylor_inv_sqrt(CKKSContext& cc, const PackedCtx& x,
                                       const std::vector<double>& coeffs, double z0) {
    return PackedCtx{eval_taylor_inv_sqrt(cc, x.ct, coeffs, z0), x.packing};
}

inline PackedCtx mask_slots(CKKSContext& cc, const PackedCtx& x, int slots, int active_dim) {
    return PackedCtx{mask_slots(cc, x.ct, slots, active_dim), x.packing};
}

Ctx rotate_and_sum_all(CKKSContext& cc, const Ctx& x, int slots);
Ctx pow_odd(CKKSContext& cc, const Ctx& y, int p);
Ctx inv_sqrt_newton_safe(CKKSContext& cc, const Ctx& x, const Ctx& y0, int iters);

inline PackedCtx rotate_and_sum_all(CKKSContext& cc, const PackedCtx& x, int slots) {
    return cc.tagged(rotate_and_sum_all(cc, x.ct, slots), x.packing,
                     packtag::t_reduce_all(packtag::PackTag::top(slots)));
}

inline PackedCtx pow_odd(CKKSContext& cc, const PackedCtx& y, int p) {
    Ctx out = pow_odd(cc, y.ct, p);
    return PackedCtx{out, y.packing, cc.tag_of_ct(out)};
}

inline PackedCtx inv_sqrt_newton_safe(CKKSContext& cc, const PackedCtx& x,
                                      const PackedCtx& y0, int iters) {
    assert_same_packing(x.packing, y0.packing);
    Ctx out = inv_sqrt_newton_safe(cc, x.ct, y0.ct, iters);
    return PackedCtx{out, x.packing, cc.tag_of_ct(out)};
}
