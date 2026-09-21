#include "ckks_primitives.h"
#include "inference.h"
#include <cmath>
#include <map>
#include <functional>
#include <string>

/// @brief Computes the inverse square root of a value using the Newton-Raphson iteration method.
/// @param cc, the crypto context
/// @param x, the input ciphertext for which we want to compute 1/sqrt(x)
/// @param ans_init, the initial guess for 1/sqrt(x)
/// @param iters, the number of iterations to perform
/// @return the i-th newton-raphson approximation of 1/sqrt(x)
Ctx inv_sqrt_newton(CKKSContext& cc, const Ctx& x, const Ctx& ans_init, int iters, double x_scale,
                    int real_d, int real_stride) {
    Ctx c = cc.mult(x, -0.5 * x_scale);

    Ctx ct = ans_init;

    for (int i = 0; i < iters; ++i) {
        auto a = cc.square(ct);
        auto b = cc.mult(c, a);   // (x_scale·x)·y² ≈ 1 first: every loop value bootstrap-safe
        b = cc.mult(b, ct);
        a = cc.mult(ct, 1.5);

        ct = cc.add(a, b);
    }
    // solving y' = 0.5 * (3y - x*y^3), as if y = 1/sqrt(x) we get
    // y' = 0.5 * (3/sqrt(x) - x * 1/x * 1/sqrt(x)) = 1/sqrt(x), so the fixed point is the solution to our problem.
    return ct;
}

/// @brief Computes the inverse square root of a value using the Goldschmidt iteration method.
/// @param cc, the crypto context
/// @param x, the input ciphertext for which we want to compute 1/sqrt(x)
/// @param ans_init, the initial guess for 1/sqrt(x)
/// @param iters, the number of iterations to perform
/// @return the i-th goldschmidt approximation of 1/sqrt(x)
/// @note The scalar mult by -0.5 is not rescaled here: the result feeds products with
///       sqrt_ct and ans, which rescale, so the rescale is applied there.
Ctx goldschmidt_inv_sqrt(CKKSContext& cc, const Ctx& x, const Ctx& ans_init, int iters) {
    Ctx x_copy = x;
    Ctx ans = cc.clone(ans_init);   // mutated in the loop below, so clone the caller's init

    Ctx sqrt_ct = cc.mult(x_copy, ans);

    for (int i = 0; i < iters; ++i) {
        Ctx res = cc.mult(sqrt_ct, ans);

        cc.inplace_mult(res, -0.5);
        cc.inplace_add(res, 1.5);

        if (i + 1 < iters) sqrt_ct = cc.mult(sqrt_ct, res);

        ans = cc.mult(ans, res);
    }
    return ans;
}

/// @brief Computes the inverse of a value using the Goldschmidt iteration method.
/// @param cc, the crypto context
/// @param a, the ciphertext a, for which we want to compute 1/a
/// @param x0_init, the initial guess for 1/a
/// @param iters, the number of iterations to perform
/// @return the i-th goldschmidt approximation of 1/a
/// @note Copies its inputs rather than mutating them, so the caller may reuse `a` and
///       `x0_init` after the call.
Ctx goldschmidt_inv(CKKSContext& cc, const Ctx& a, const Ctx& x0_init, int iters) {
    WithStep _w(cc, "goldschmidt");

    Ctx x0 = x0_init;
    Ctx a_l = a;

    Ctx E = cc.mult(a_l, x0);

    E = cc.negate(E);
    cc.inplace_add(E, 1.0);

    for (int i = 0; i < iters; ++i) {
        WithStep _wi(cc, "iter_" + std::to_string(i));
        auto e_add = cc.add(E, 1.0);
        x0 = cc.mult(x0, e_add);
        // E^2 only feeds the next iteration's `E+1`; the last square is dead. Skip it.
        if (i + 1 < iters) cc.inplace_square(E);
    }

    return x0;
}

Ctx goldschmidt_inv(CKKSContext& cc, const Ctx& N_init, const Ctx& D_init, const Ctx& F_init,
                    int iters, bool sparse_df) {
    WithStep _w(cc, "goldschmidt_nice");

    Ctx N = cc.mult(N_init, F_init);
    Ctx F, D_neg;
    {
        CKKSContext::SparseBtsScope ss(cc, sparse_df);
        F = cc.negate(F_init);
        D_neg = cc.mult(D_init, F);   // pay negate once, upfront, outside the loop
        F     = cc.add(D_neg, 2.0);   // F = 2 - D = 2 + D_neg, free
    }

    for (int i = 1; i < iters; ++i) {
        WithStep _wi(cc, "iter_" + std::to_string(i));
        N = cc.mult(N, F);
        if (i + 1 < iters) {
            CKKSContext::SparseBtsScope ss(cc, sparse_df);
            D_neg = cc.mult(D_neg, F);
            F = cc.add(D_neg, 2.0); // F = 2 - D = 2 + D_neg, free
        }
    }

    return N;
}

Ctx goldschmidt_recip(CKKSContext& cc, const Ctx& D_init, const Ctx& F_init, int iters,
                      bool sparse_df) {
    WithStep _w(cc, "goldschmidt_recip");

    // sparse_df: the D/F correction track is a broadcast quantity, so its reactive/planned
    // refreshes may route sparse. The R track stays dense: it carries the payload.
    Ctx R = F_init;                   // R_0 = 1·F_init
    Ctx F, D_neg;
    {
        CKKSContext::SparseBtsScope ss(cc, sparse_df);
        F = cc.negate(F_init);
        D_neg = cc.mult(D_init, F);   // = -D_init·F_init
        F  = cc.add(D_neg, 2.0);      // F_1 = 2 - D_init·F_init
    }

    for (int i = 1; i < iters; ++i) {
        WithStep _wi(cc, "iter_" + std::to_string(i));
        R = cc.mult(R, F);
        if (i + 1 < iters) {
            CKKSContext::SparseBtsScope ss(cc, sparse_df);
            D_neg = cc.mult(D_neg, F);
            F = cc.add(D_neg, 2.0);
        }
    }

    return R;
}


/// @brief Computes a weighted sum of ciphertexts, where the weights are given as plaintexts.
/// @param cc, the crypto context
/// @param cts, the vector of ciphertexts to be summed 
/// @param weights, the vector of plaintext weights corresponding to each ciphertext
/// @return the resulting ciphertext of the weighted sum
/// @note Assumes every input ciphertext has the same shape; this is not checked.
Ctx eval_linear_wsum(CKKSContext& cc,
                     std::vector<Ctx>& cts,
                     const std::vector<double>& weights) {
    Ctx result = cc.mult(cts[0], weights[0]);
    for (size_t i = 1; i < cts.size(); ++i) {
        Ctx term = cc.mult(cts[i], weights[i]);
        cc.inplace_add(result, term);
    }
    return result;
}

/// @brief Masks the first active_dim * (slots / active_dim) slots, zeroing the rest.
Ctx mask_slots(CKKSContext& cc, const Ctx& x, int slots, int active_dim) {
    int intRot = slots / active_dim;
    int active = active_dim * intRot;
    if (active >= slots) return x;

    std::vector<double> mask_vec(slots, 0.0);
    for (int i = 0; i < active; ++i) mask_vec[i] = 1.0;
    Ptx mask = cc.cc->MakeCKKSPackedPlaintext(
        mask_vec, /*noiseScaleDeg=*/1, (uint32_t)level_of(x));
    return cc.mult(x, mask);
}


/// @brief Full-ciphertext rotate-and-sum: the total of all `slots` slots,
/// broadcast to every slot (log2(slots) rotations, no levels consumed).
/// On a complex-payload ct the lanes reduce independently:
/// rotate_and_sum_all(a + i*b) = sum(a) + i*sum(b).
Ctx rotate_and_sum_all(CKKSContext& cc, const Ctx& x, int slots) {
    Ctx r = cc.clone(x);
    for (int gap = 1; gap < slots; gap *= 2)
        cc.inplace_add(r, cc.rotate(r, gap));
    return r;
}

/// @brief y^p for odd p in {3,5,7,9,11,13,15,19} via square-and-multiply
/// (<= 6 mult levels at p=19). Odd powers preserve sign and, on a purely
/// imaginary-axis ct, the axis (up to the deterministic i^p sign).
Ctx pow_odd(CKKSContext& cc, const Ctx& y, int p) {
    switch (p) {
        case 3: return cc.mult(cc.square(y), y);
        case 5: return cc.mult(cc.square(cc.square(y)), y);
        case 7: {
            Ctx s1 = cc.square(y);
            return cc.mult(cc.mult(cc.square(s1), s1), y);
        }
        case 9: return cc.mult(cc.square(cc.square(cc.square(y))), y);
        case 11: {
            Ctx s1 = cc.square(y);
            Ctx s3 = cc.square(cc.square(s1));
            return cc.mult(cc.mult(s3, s1), y);
        }
        case 13: {
            Ctx s2 = cc.square(cc.square(y));
            return cc.mult(cc.mult(cc.square(s2), s2), y);
        }
        case 15: {
            Ctx y5 = cc.mult(cc.square(cc.square(y)), y);
            return cc.mult(cc.square(y5), y5);
        }
        case 19: {
            Ctx y2 = cc.square(y);
            Ctx y16 = cc.square(cc.square(cc.square(y2)));
            return cc.mult(cc.mult(y16, y2), y);
        }
        default:
            throw std::runtime_error("pow_odd: unsupported p=" +
                                     std::to_string(p));
    }
}

/// @brief From-below Newton for 1/sqrt(x) with the magnitude safe product order
Ctx inv_sqrt_newton_safe(CKKSContext& cc, const Ctx& x, const Ctx& y0,
                         int iters) {
    Ctx xh = cc.mult(x, -0.5);
    Ctx y = cc.clone(y0);
    for (int i = 0; i < iters; ++i) {
        Ctx b = cc.mult(cc.mult(cc.mult(xh, y), y), y);   // -0.5*x*y^3
        y = cc.add(cc.mult(y, 1.5), b);
    }
    return y;
}
