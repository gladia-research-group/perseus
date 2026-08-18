#include "ckks_fixture.h"
#include "test_helpers.h"

#include <gtest/gtest.h>

using namespace test_helpers;

namespace {

using FideslibWrapperTest = CkksFixture;

// Round-trip — avg (non-zero const) + edge (exact zero, denom floor case).
TEST_F(FideslibWrapperTest, RoundTrip) {
    EXPECT_NEAR(decrypt_slots(fhe(), enc_const(3.14))[0], 3.14, 1e-3);
    EXPECT_NEAR(decrypt_slots(fhe(), enc_const(0.0))[0],  0.0,  1e-6);
}


TEST_F(FideslibWrapperTest, Arithmetic) {
    Ctx a = enc_const(2.5);
    Ctx b = enc_const(-0.4);
    const uint32_t l0 = level_of(a);

    Ctx p = fhe().mult(a, b);
    EXPECT_NEAR(decrypt_slots(fhe(), p)[0], -1.0, 1e-3);

    EXPECT_GE(level_of(p), l0);

    Ctx s = fhe().add(a, b);
    EXPECT_EQ(level_of(s), l0);
}

TEST_F(FideslibWrapperTest, ClonePreservesValueAndLevel) {
    Ctx a = enc_const(1.7);
    Ctx c = fhe().clone(a);
    EXPECT_EQ(level_of(c), level_of(a));
    EXPECT_NEAR(decrypt_slots(fhe(), c)[0], 1.7, 1e-4);
}

TEST_F(FideslibWrapperTest, NegationDoesNotConsumeLevel) {
    Ctx a = enc_const(1.2);
    const uint32_t l0 = level_of(a);

    Ctx n = fhe().negate(a);
    EXPECT_EQ(level_of(n), l0);
    EXPECT_NEAR(decrypt_slots(fhe(), n)[0], -1.2, 1e-4);

    Ctx b = enc_const(0.8);
    const uint32_t l1 = level_of(b);
    fhe().inplace_negate(b);
    EXPECT_EQ(level_of(b), l1);
    EXPECT_NEAR(decrypt_slots(fhe(), b)[0], -0.8, 1e-4);
}

TEST_F(FideslibWrapperTest, SubtractionDoesNotConsumeLevel) {
    Ctx a = enc_const(2.0);
    Ctx b = enc_const(0.5);
    const uint32_t la = level_of(a);
    const uint32_t lb = level_of(b);

    Ctx s = fhe().sub(a, b);
    EXPECT_EQ(level_of(s), la);
    EXPECT_EQ(level_of(a), la);
    EXPECT_EQ(level_of(b), lb);
    EXPECT_NEAR(decrypt_slots(fhe(), s)[0], 1.5, 1e-4);

    Ctx c = enc_const(1.25);
    Ctx d = enc_const(0.25);
    const uint32_t lc = level_of(c);
    const uint32_t ld = level_of(d);
    fhe().inplace_sub(c, d);
    EXPECT_EQ(level_of(c), lc);
    EXPECT_EQ(level_of(d), ld);
    EXPECT_NEAR(decrypt_slots(fhe(), c)[0], 1.0, 1e-4);
}

// De-risk: LevelReduce must run on the GPU ct, drop the level, and PRESERVE the value
// (unlike Rescale). Gates the KV-pin enforcement (drop_to_level).
TEST_F(FideslibWrapperTest, LevelReduceDropsLevelPreservesValue) {
    Ctx a = enc_const(2.5);
    const int l0 = static_cast<int>(level_of(a));

    fhe().drop_to_level(a, l0 + 2);
    EXPECT_EQ(static_cast<int>(level_of(a)), l0 + 2);
    EXPECT_NEAR(decrypt_slots(fhe(), a)[0], 2.5, 1e-3);

    // idempotent: target at-or-below current level is a no-op
    fhe().drop_to_level(a, l0 + 2);
    EXPECT_EQ(static_cast<int>(level_of(a)), l0 + 2);
    EXPECT_NEAR(decrypt_slots(fhe(), a)[0], 2.5, 1e-3);
}

}
