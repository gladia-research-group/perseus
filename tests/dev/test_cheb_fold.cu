// Isolated proof that folding the softmax `scale_factor` into the Chebyshev
// domain is harmless (Fusion B, cheb variant).
//
// Production (cachemir_attention.cu:171-172, then eval_chebyshev_series):
//     ct' = ct * scale_factor                 // explicit scalar mult, 1 level
//     out = cheb(ct', coeffs, a, b)           // affine map y = ct'*2/(b-a) - (a+b)/(b-a)
//
// Folded (no explicit mult; scale absorbed by the cheb affine map):
//     out = cheb(ct, coeffs, a/scale_factor, b/scale_factor)
//          // affine map y = ct*2/(b'-a') - (a'+b')/(b'-a')
//          //            = ct*scale_factor*2/(b-a) - (a+b)/(b-a)   == same y
//
// scale_factor = 2^-(log2delta1+log2delta2) = 2^-(n_squarings+refinement_iters)
//              = 2^-(2+4) = 2^-6  (config_loader.h:96-97, gpt2_base configs.json).
//
// The fold is an algebraic identity on the affine map. The ONLY behavioural
// change in FHE is:
//   (1) one fewer scalar mult  -> one fewer level consumed, and
//   (2) the value entering cheb() is the RAW centered score (|abs| up to
//       ~squeeze_bound ~= 50), not the pre-shrunk ~0.56. This test proves that
//       raw value is harmless: cheb()'s first op is the scalar affine mult,
//       which shrinks it to [-1,1] with NO bootstrap and NO square in between,
//       so it never reaches the bts range wall (~10) or a squaring.
//
// Claims asserted: (A) outputs match to the noise floor across the whole
// domain incl. the worst-case +/-squeeze_bound; (B) the folded path sits one
// level higher (a level is saved); (C) the folded output is finite & bounded.

#include "ckks_fixture.h"
#include "ckks_primitives.h"
#include "test_helpers.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

using ChebFoldTest = CkksFixture;

// Production softmax params (configs/model/approximation/gpt2_base/configs.json).
// scale_factor = 2^-6 for every block (n_squarings=2, refinement_iters=4).
struct BlockParams {
    const char* name;
    double a, b;                 // cheb domain (post-scale)
    std::vector<double> coeffs;  // cheb coefficients (ascending T-order)
};

constexpr double kScaleFactor = 1.0 / 64.0;  // 2^-6

const std::vector<BlockParams>& blocks() {
    static const std::vector<BlockParams> b = {
        {"h0", -0.5578325465321541, 0.5578325465321541,
         {1.0793204170464323, 0.5798138015007654, 0.07983132358559622,
          0.007374462262645712, 0.0005122271095283232, 2.849992735361374e-05,
          1.3224029012117316e-06, 5.2618366389924284e-08, 1.8325367180194995e-09}},
        {"h2", -0.7901022054255009, 0.7901022054255009,
         {1.162261098927262, 0.8533806939343992, 0.1643442156279813,
          0.021365724228170114, 0.0020938833434599553, 0.00016458514432863517,
          1.0796583314241762e-05, 6.076248697026358e-07, 2.9940573619873186e-08}},
        {"h11", -0.7333540946245194, 0.7333540946245194,
         {1.139039481249876, 0.7837717981209015, 0.1405800274942926,
          0.016993356302725204, 0.0015474120582887373, 0.00011297538299452384,
          6.882267528304665e-06, 3.5964805106034564e-07, 1.6453656106100028e-08}},
    };
    return b;
}

// Plaintext reference: same Chebyshev series the FHE path computes.
double cheb_ref(double x, double a, double b, const std::vector<double>& c) {
    const double y = (2.0 * x - (a + b)) / (b - a);
    double Tprev = 1.0, Tcur = y, acc = c[0] + (c.size() > 1 ? c[1] * y : 0.0);
    for (size_t i = 2; i < c.size(); ++i) {
        double Tnext = 2.0 * y * Tcur - Tprev;
        acc += c[i] * Tnext;
        Tprev = Tcur;
        Tcur = Tnext;
    }
    return acc;
}

Ctx enc_vec(CKKSContext& cc, const std::vector<double>& v) {
    auto pt = cc.cc->MakeCKKSPackedPlaintext(v);
    return encrypt(cc.cc, pt, cc.pk());
}

// Build the raw (pre-shrink) centered-score test vector: sweep the FULL raw
// domain [-squeeze_bound, +squeeze_bound] = [a/sf, b/sf], so the folded path is
// exercised exactly where it leaves values un-shrunk (the magnitude concern).
std::vector<double> raw_domain_sweep(const BlockParams& bp, int slots) {
    const double lo = bp.a / kScaleFactor;   // == -squeeze_bound
    const double hi = bp.b / kScaleFactor;   // == +squeeze_bound
    std::vector<double> v(slots, 0.0);
    const int n = std::min(slots, 64);
    for (int i = 0; i < n; ++i)
        v[i] = lo + (hi - lo) * (double)i / (double)(n - 1);
    return v;
}

TEST_F(ChebFoldTest, FoldMatchesProductionAndSavesALevel) {
    for (const auto& bp : blocks()) {
        const double a_raw = bp.a / kScaleFactor;
        const double b_raw = bp.b / kScaleFactor;

        std::vector<double> raw = raw_domain_sweep(bp, slots());
        Ctx x = enc_vec(fhe(), raw);
        const int lvl_in = level_of(x);

        // --- Path A: production (explicit scale_factor mult, then cheb on [a,b])
        Ctx xa   = fhe().mult(x, kScaleFactor);
        Ctx outA = eval_chebyshev_series(fhe(), xa, bp.coeffs, bp.a, bp.b);

        // --- Path B: folded (cheb directly on raw x over [a/sf, b/sf])
        Ctx outB = eval_chebyshev_series(fhe(), x, bp.coeffs, a_raw, b_raw);

        auto da = decrypt_slots(fhe(), outA);
        auto db = decrypt_slots(fhe(), outB);

        // (A) Path A == Path B to the noise floor, across the whole raw domain.
        double max_abs_diff = 0.0, max_rel_diff = 0.0, max_out = 0.0;
        double max_ref_err_B = 0.0;
        const int n = std::min((int)raw.size(), 64);
        for (int i = 0; i < n; ++i) {
            double diff = std::abs(da[i] - db[i]);
            max_abs_diff = std::max(max_abs_diff, diff);
            if (std::abs(da[i]) > 1e-9)
                max_rel_diff = std::max(max_rel_diff, diff / std::abs(da[i]));
            max_out = std::max(max_out, std::abs(db[i]));
            // (C) folded output tracks the intended plaintext function (finite, bounded)
            double ref = cheb_ref(raw[i] * kScaleFactor, bp.a, bp.b, bp.coeffs);
            EXPECT_TRUE(std::isfinite(db[i])) << bp.name << " slot " << i;
            max_ref_err_B = std::max(max_ref_err_B, std::abs(db[i] - ref));
        }

        // (B) the folded path consumed one fewer level. `level_of` counts
        // CONSUMED levels (increases down the modulus chain), so the cheaper
        // path has the SMALLER value: lvl_B == lvl_A - 1.
        const int lvl_A = level_of(outA);
        const int lvl_B = level_of(outB);
        const int levels_saved = lvl_A - lvl_B;

        std::cout << "[cheb_fold " << bp.name << "]"
                  << " raw_domain=[" << a_raw << "," << b_raw << "]"
                  << " (|x|max=" << std::max(std::abs(a_raw), std::abs(b_raw)) << ")"
                  << " max_abs_diff=" << max_abs_diff
                  << " max_rel_diff=" << max_rel_diff
                  << " max|outB-ref|=" << max_ref_err_B
                  << " | levels consumed: prodA=" << lvl_A << " foldB=" << lvl_B
                  << " saved=" << levels_saved << "\n";

        EXPECT_LT(max_abs_diff, 1e-3) << bp.name << ": fold changed the result";
        EXPECT_LT(max_ref_err_B, 5e-3) << bp.name << ": folded path off the function";
        EXPECT_LT(max_out, 1e3) << bp.name << ": folded output blew up";
        EXPECT_EQ(levels_saved, 1) << bp.name << ": fold did not save exactly one level";
    }
}

}  // namespace
