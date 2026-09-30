#include "ckks_fixture.h"
#include "ckks_primitives.h"
#include "test_helpers.h"

#include <gtest/gtest.h>

#include <cmath>

using namespace test_helpers;

namespace {

using PolyEvalTest = CkksFixture;

TEST_F(PolyEvalTest, EvalPolynomial_Horner) {
    Ctx x = enc_const(2.0);
    Ctx r = eval_polynomial(fhe(), x, {1.0, 0.0, 0.0, 1.0});
    EXPECT_NEAR(decrypt_slots(fhe(), r)[0], 9.0, 1e-2);
    EXPECT_LE(level_of(r) - level_of(x), 3u);

    Ctx x1 = enc_const(3.0);
    Ctx r1 = eval_polynomial(fhe(), x1, {1.0, 2.0});
    EXPECT_NEAR(decrypt_slots(fhe(), r1)[0], 7.0, 1e-3);
    EXPECT_LE(level_of(r1) - level_of(x1), 1u);
}

TEST_F(PolyEvalTest, EvalPolynomialPs) {
    Ctx x = enc_const(2.0);
    Ctx r = eval_polynomial_ps(fhe(), x, {1.0, 0.0, 0.0, 1.0}, slots());
    EXPECT_NEAR(decrypt_slots(fhe(), r)[0], 9.0, 1e-2);
    EXPECT_LE(level_of(r) - level_of(x), 2u);

    Ctx x4 = enc_const(1.0);
    Ctx r4 = eval_polynomial_ps(fhe(), x4, {1.0, 1.0, 1.0, 1.0, 1.0}, slots());
    EXPECT_NEAR(decrypt_slots(fhe(), r4)[0], 5.0, 1e-2);
    EXPECT_LE(level_of(r4) - level_of(x4), 3u);
}

TEST_F(PolyEvalTest, EvalPolynomialDeg4) {
    Ctx x = enc_const(2.0);
    Ctx r = eval_polynomial_deg4(fhe(), x, {1.0, 1.0, 1.0, 1.0, 1.0});
    EXPECT_NEAR(decrypt_slots(fhe(), r)[0], 31.0, 1e-2);
    EXPECT_LE(level_of(r) - level_of(x), 3u);
}

TEST_F(PolyEvalTest, TaylorInvSqrtCoeffs) {
    auto c = taylor_inv_sqrt_coeffs(1.0);
    ASSERT_EQ(c.size(), 4u);
    EXPECT_DOUBLE_EQ(c[0],  1.0);
    EXPECT_DOUBLE_EQ(c[1], -0.5);
    EXPECT_DOUBLE_EQ(c[2],  0.375);
    EXPECT_DOUBLE_EQ(c[3], -0.3125);
}

TEST_F(PolyEvalTest, EvalTaylorInvSqrt) {
    auto coeffs = taylor_inv_sqrt_coeffs(1.0);

    Ctx x = enc_const(1.04);
    Ctx r = eval_taylor_inv_sqrt(fhe(), x, coeffs, /*z0=*/1.0);
    EXPECT_NEAR(decrypt_slots(fhe(), r)[0], 1.0 / std::sqrt(1.04), 1e-3);
    EXPECT_LE(level_of(r) - level_of(x), 3u);

    Ctx xz = enc_const(1.0);
    Ctx rz = eval_taylor_inv_sqrt(fhe(), xz, coeffs, /*z0=*/1.0);
    EXPECT_NEAR(decrypt_slots(fhe(), rz)[0], 1.0, 1e-3);
}

}
