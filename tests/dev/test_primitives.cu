#include "ckks_fixture.h"
#include "ckks_primitives.h"
#include "packing/cachemir/cachemir_norm_utils.h"
#include "test_helpers.h"

#include <gtest/gtest.h>

#include <vector>

using namespace test_helpers;

namespace {

using PrimitivesTest = CkksFixture;

TEST_F(PrimitivesTest, ExpSquaring) {
    EXPECT_NEAR(decrypt_slots(fhe(), exp_squaring(fhe(), enc_const(1.5), 2))[0], 5.0625, 1e-2);
    EXPECT_NEAR(decrypt_slots(fhe(), exp_squaring(fhe(), enc_const(2.0), 0))[0], 2.0,    1e-4);
}

TEST_F(PrimitivesTest, MaskSlots) {
    Ctx x = enc_const(1.0);
    auto masked = decrypt_slots(fhe(), mask_slots(fhe(), x, slots(), /*active_dim=*/3));
    EXPECT_NEAR(masked[0],            1.0, 1e-3);
    EXPECT_NEAR(masked[slots() - 1],  0.0, 1e-3);

    Ctx noop = mask_slots(fhe(), x, slots(), /*active_dim=*/slots());
    EXPECT_EQ(level_of(noop), level_of(x));
}

TEST_F(PrimitivesTest, EvalLinearWsum) {
    std::vector<Ctx> two = {enc_const(0.5), enc_const(1.0)};
    EXPECT_NEAR(decrypt_slots(fhe(), eval_linear_wsum(fhe(), two, {2.0, 3.0}))[0], 4.0, 1e-3);

    std::vector<Ctx> one = {enc_const(0.7)};
    EXPECT_NEAR(decrypt_slots(fhe(), eval_linear_wsum(fhe(), one, {2.0}))[0], 1.4, 1e-3);
}

TEST_F(PrimitivesTest, NewtonInverse_Converges) {
    Ctx init = enc_const(0.4);
    Ctx res = newton_inverse(fhe(), init, enc_const(2.0), 3);
    EXPECT_NEAR(decrypt_slots(fhe(), res)[0], 0.5, 1e-3);
    EXPECT_LE(level_of(res) - level_of(init), 2u * 3u);
}

TEST_F(PrimitivesTest, GoldschmidtInv_Converges) {
    Ctx init = enc_const(0.4);
    Ctx res = goldschmidt_inv(fhe(), enc_const(2.0), init, 3);
    EXPECT_NEAR(decrypt_slots(fhe(), res)[0], 0.5, 1e-3);
    EXPECT_LE(level_of(res) - level_of(init), 4u);
}


TEST_F(PrimitivesTest, InvSqrtNewton_Converges) {
    Ctx init = enc_const(0.4);
    Ctx res = inv_sqrt_newton(fhe(), enc_const(4.0), init, 3);
    EXPECT_NEAR(decrypt_slots(fhe(), res)[0], 0.5, 1e-2);
    EXPECT_LE(level_of(res) - level_of(init), 7u);
}

TEST_F(PrimitivesTest, GoldschmidtInvSqrt_Converges) {
    Ctx init = enc_const(0.4);
    Ctx res = goldschmidt_inv_sqrt(fhe(), enc_const(4.0), init, 3);
    EXPECT_NEAR(decrypt_slots(fhe(), res)[0], 0.5, 1e-2);
    EXPECT_LE(level_of(res) - level_of(init), 10u);
}

TEST_F(PrimitivesTest, ComputePerTokenSum) {
    Inference inf = make_inf();
    PackedCtx x = inf.pack(enc_const(1.0), PackingKind::Cachemir);
    PackedCtx s = cachemir::compute_per_token_sum(inf, x);
    EXPECT_NEAR(decrypt_slots(inf, s)[0], 1024.0, 1e-2);
    EXPECT_EQ(level_of(s.ct), level_of(x.ct));
}

TEST_F(PrimitivesTest, ComputeVarianceInterleaved) {
    Inference inf = make_inf();
    PackedCtx centered = inf.pack(enc_const(1.0), PackingKind::Cachemir);
    PackedCtx var = cachemir::compute_variance_interleaved(inf, centered);
    EXPECT_NEAR(decrypt_slots(inf, var)[0], 32768.0 / 768.0, 1e-1);
    EXPECT_LE(level_of(var.ct) - level_of(centered.ct), 2u);
}

}
