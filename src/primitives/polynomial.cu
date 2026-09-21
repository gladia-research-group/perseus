#include "ckks_primitives.h"
#include "inference.h"
#include <cmath>
#include <map>
#include <functional>
#include <vector>


/// @brief Evaluates a polynomial at a given point using Horner's method.
/// @param cc, the crypto context
/// @param x, the input ciphertext at which to evaluate the polynomial
/// @param coeffs, the coefficients of the polynomial in standard basis [a0, a1, a2, ..., an] representing a0 + a1*x + a2*x^2 + ... + an*x^n
/// @return the ciphertext resulting from evaluating the polynomial at x
Ctx eval_polynomial(CKKSContext& cc, const Ctx& x, const std::vector<double>& coeffs) {
    size_t n = coeffs.size();
    Ctx result = cc.mult(x, coeffs[n - 1]);
    cc.inplace_add(result, coeffs[n - 2]);

    for (int i = static_cast<int>(n) - 3; i >= 0; --i) {
        result = cc.mult(result, x);
        cc.inplace_add(result, coeffs[i]);
    }

    return result;
}

/// @brief Evaluates a polynomial using the Power-sum tree algorithm. Achieves optimal multiplicative depth of ceil(log2(d+1)) where d is the degree.
/// @param cc, the crypto context
/// @param x, the input ciphertext
/// @param coeffs, coefficients [a0, a1, ..., an] for a0 + a1*x + ... + an*x^n
/// @param slots, number of slots
/// @return the ciphertext resulting from evaluating the polynomial at x
Ctx eval_polynomial_ps(CKKSContext& cc, const Ctx& x, const std::vector<double>& coeffs, size_t slots) {
    int d = static_cast<int>(coeffs.size()) - 1;

    if (d == 0) {
        // Constant polynomial: return coeffs[0] in every slot, at a level matching x.
        Ctx z = cc.sub(x, x);
        cc.inplace_add(z, coeffs[0]);
        return z;
    }

    Ctx result;

    int floor_log2 = std::floor(std::log2(d));

    std::vector<Ctx> powers(floor_log2 + 1);
    powers[0] = x;
    for (int p = 1; p <= floor_log2; ++p) {
        powers[p] = cc.square(powers[p - 1]); // squaring triggers rescaling
    }

    for (int p = 1; p < coeffs.size(); ++p) {
        Ctx curr_x = nullptr;
        int pow_idx = 0;
        for (int i = p; i > 0; i /= 2) {
            if (i % 2 == 1) {
                if (curr_x == nullptr) {
                    curr_x = cc.mult(powers[pow_idx], coeffs[p]);
                } else {
                    curr_x = cc.mult(curr_x, powers[pow_idx]);
                }
            }
            pow_idx++;
        }

        if (p > 1) {
            cc.inplace_add(result, curr_x);
        } else {
            // First term: fold in coeffs[0] via a scalar add. Using an
            // encoded Ptx here (encode_const + cc.add) drops coeffs[0]
            // when curr_x's noiseScaleDeg/level diverge from the ptx —
            // a scalar add stays robust under FLEXIBLEAUTO.
            result = cc.add(curr_x, coeffs[0]);
        }
    }

    return result;
}


/// @brief Evaluates a Chebyshev series sum_{i=0}^{n} coeffs[i]*T_i(y) over [a,b].
/// @param coeffs Chebyshev coefficients [c0, c1, ..., cn] in ascending T-order.
Ctx eval_chebyshev_series(CKKSContext& cc, const Ctx& x,
                          const std::vector<double>& coeffs, double a, double b) {
    WithStep _w(cc, "chebyshev_series");

    // Effective degree: ignore trailing exact-zero coefficients.
    int n = static_cast<int>(coeffs.size()) - 1;
    while (n > 0 && coeffs[n] == 0.0) --n;

    // Affine map y = (2x - (a+b))/(b-a) in [-1,1]; T_1(y) = y.
    const double alpha = 2.0 / (b - a);
    const double beta  = (a + b) / (b - a);
    Ctx y = cc.mult(x, alpha);
    cc.inplace_add(y, -beta);

    if (n <= 0) {
        Ctx z = cc.sub(y, y);                       // zero ct at y's level
        cc.inplace_add(z, coeffs.empty() ? 0.0 : coeffs[0]);
        return z;
    }
    if (n == 1) {
        Ctx r = cc.mult(y, coeffs[1]);
        cc.inplace_add(r, coeffs[0]);
        return r;
    }

    std::vector<Ctx> T(n + 1);                       // T[i] holds T_i; T[0] unused
    T[1] = y;
    for (int i = 2; i <= n; ++i) {
        if ((i & 1) == 0) {                          // T_{2j} = 2 T_j^2 - 1
            Ctx sq = cc.square(T[i / 2]);
            T[i] = cc.add(sq, sq);
            cc.inplace_add(T[i], -1.0);
        } else {                                     // T_{2j+1} = 2 T_{j+1} T_j - y
            int j = i / 2;
            Ctx prod = cc.mult(T[j + 1], T[j]);
            T[i] = cc.add(prod, prod);
            cc.inplace_sub(T[i], y);
        }
    }

    // result = c_0 + sum_{i=1}^{n} c_i T_i   (drop zero coefficients).
    std::vector<Ctx> terms;
    std::vector<double> w;
    for (int i = 1; i <= n; ++i) {
        if (coeffs[i] != 0.0) {
            terms.push_back(T[i]);
            w.push_back(coeffs[i]);
        }
    }
    Ctx r = eval_linear_wsum(cc, terms, w);
    cc.inplace_add(r, coeffs[0]);
    return r;
}

/// @brief Evaluates a degree-4 polynomial using hard-coded paterson-stockmeyer.
/// @param cc, the crypto context
/// @param x, the input ciphertext
/// @param coeffs, coefficients [a0, a1, ..., an] for a0 + a1*x + ... + an*x^n
/// @return the ciphertext resulting from evaluating the polynomial at x
Ctx eval_polynomial_deg4(CKKSContext& cc, const Ctx& x, const std::vector<double>& coeffs) {
    Ctx x2 = cc.square(x);
    Ctx result = cc.square(x2);

    cc.inplace_mult(result, coeffs[4]);

    Ctx A = cc.mult(x, coeffs[1]);
    cc.inplace_add(A, coeffs[0]);

    Ctx B = cc.mult(x, coeffs[3]);
    cc.inplace_add(B, coeffs[2]);
    B = cc.mult(B, x2);

    A = cc.add(A, B);

    return cc.add(A, result);
}

// Specialized fixed-degree-8 evaluator (Estrin form). As above for deg4.
Ctx eval_polynomial_deg8(CKKSContext& cc, const Ctx& x, const std::vector<double>& c) {
    Ctx x2 = cc.square(x);          // depth 1
    Ctx x4 = cc.square(x2);         // depth 2
    Ctx x8 = cc.square(x4);         // depth 3

    // low = (c0 + c1 x) + (c2 + c3 x) * x2
    Ctx lo0 = cc.mult(x, c[1]); cc.inplace_add(lo0, c[0]);
    Ctx lo1 = cc.mult(x, c[3]); cc.inplace_add(lo1, c[2]);
    lo1 = cc.mult(lo1, x2);
    Ctx low = cc.add(lo0, lo1);

    // mid = (c4 + c5 x) + (c6 + c7 x) * x2
    Ctx mi0 = cc.mult(x, c[5]); cc.inplace_add(mi0, c[4]);
    Ctx mi1 = cc.mult(x, c[7]); cc.inplace_add(mi1, c[6]);
    mi1 = cc.mult(mi1, x2);
    Ctx mid = cc.add(mi0, mi1);

    mid = cc.mult(mid, x4);         // mid * x^4
    Ctx hi = cc.mult(x8, c[8]);     // c8 * x^8

    Ctx out = cc.add(low, mid);
    return cc.add(out, hi);
}

/// @brief Computes the degree-3 Taylor coefficients of 1/sqrt(z) centered at z0.
/// Call once during initialization; pass the result to eval_taylor_inv_sqrt at runtime.
/// Returns coefficients in the shifted basis: f(z) ≈ a0 + a1*(z-z0) + a2*(z-z0)^2 + a3*(z-z0)^3
/// @param z0, the expansion point (typically midpoint of the interval)
/// @return vector {a0, a1, a2, a3}
std::vector<double> taylor_inv_sqrt_coeffs(double z0) {
    double z0_sqrt = std::sqrt(z0);
    double a0 =  1.0 / z0_sqrt;
    double a1 = -1.0 / (2.0 * z0 * z0_sqrt);
    double a2 =  3.0 / (8.0 * z0 * z0 * z0_sqrt);
    double a3 = -5.0 / (16.0 * z0 * z0 * z0 * z0_sqrt);
    return {a0, a1, a2, a3};
}

/// @brief Evaluates the degree-3 Taylor approximation of 1/sqrt(z) around z0.
/// @param coeffs, precomputed Taylor coefficients from taylor_inv_sqrt_coeffs (computed once at init)
/// @param z0, the expansion point used when computing coeffs
Ctx eval_taylor_inv_sqrt(CKKSContext& cc, const Ctx& x,
                          const std::vector<double>& coeffs, double z0) {
    Ctx u = cc.add(x, -z0);
    return eval_polynomial(cc, u, coeffs);
}


Ctx eval_remez_31(CKKSContext& cc, const Ctx& x, const std::vector<double>& Ncoeffs, const std::vector<double>& Dcoeffs, double alpha, double beta, int gs_iters) {
    WithStep _w(cc, "remez_31");
    Ctx D = cc.mult(x, Dcoeffs[1]);
    cc.inplace_add(D, Dcoeffs[0]);

    Ctx F_init = cc.mult(D, -beta);
    cc.inplace_add(F_init, alpha);

    double pos_coeff_3 = std::abs(Ncoeffs[3]);
    double coeff_3_sign = (Ncoeffs[3] >= 0) ? 1.0 : -1.0;

    Ctx N = cc.mult(x, coeff_3_sign * std::pow(pos_coeff_3, 1.0 / 3.0));
    cc.inplace_add(N, Ncoeffs[2] * std::pow(pos_coeff_3, -2.0 / 3.0));
    Ctx x2 = cc.mult(x, std::pow(pos_coeff_3, 1.0 / 3.0));
    N = cc.mult(N, x2);
    cc.inplace_add(N, Ncoeffs[1] * std::pow(pos_coeff_3, -1.0 / 3.0));
    N = cc.mult(N, x2);
    cc.inplace_add(N, Ncoeffs[0]);

    _w.next("gs_iters");
    Ctx N_frac_D = goldschmidt_inv(cc, N, D, F_init, gs_iters);

    return N_frac_D;
}