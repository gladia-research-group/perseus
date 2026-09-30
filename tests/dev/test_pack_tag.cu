// Soundness gate for the propagated packing tags.
//
// A PackTag is only useful if it NEVER under-estimates. Routing a ciphertext sparse when its
// true period does not divide s produces silent corruption (the fold averages and returns
// plausible garbage). Nothing throws, so a
// wrong transfer function cannot be caught by any normal test. This is the catch.
//
// Method: build a concrete slot vector, apply an operation to BOTH the vector (exactly, in the
// clear) and the tag (via its transfer function), then assert
//
//     tag.period >= analyze_packing(vector).period
//
// i.e. the tag is allowed to be pessimistic and forbidden to be optimistic. Same for support.
//
// CPU-only — no crypto context, no GPU. Runs in milliseconds.
#include "packing/pack_signature.h"
#include "packing/pack_tag.h"

#include <gtest/gtest.h>

#include <complex>
#include <random>
#include <vector>

using namespace packtag;

namespace {

constexpr int S = 1024;   // small slot count keeps the O(S^2) periodicity scan instant

std::vector<std::complex<double>> as_cplx(const std::vector<double>& v) {
    std::vector<std::complex<double>> c(v.size());
    for (size_t i = 0; i < v.size(); ++i) c[i] = {v[i], 0.0};
    return c;
}

// The soundness assertion, in one place so every case reads the same.
void assert_sound(const char* what, const std::vector<double>& v, const PackTag& tag) {
    const PackSignature sig = analyze_packing(as_cplx(v));
    EXPECT_GE(tag.period, sig.period)
        << what << ": tag claims period " << tag.period << " but the data is only periodic at "
        << sig.period << " — this would route sparse and corrupt SILENTLY";
    // Support: the tag must cover every live slot the data actually has.
    if (!tag.support.is_dense()) {
        const int covered = tag.fold_collision_free_at(sig.fold_s) ? 1 : 0;
        EXPECT_TRUE(covered || sig.fold_s >= S)
            << what << ": tag support claims a collision-free fold the data does not support";
    }
}

std::vector<double> periodic(int p, uint32_t seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<double> base(p);
    for (auto& x : base) x = d(g);
    std::vector<double> v(S);
    for (int i = 0; i < S; ++i) v[i] = base[i % p];
    return v;
}

std::vector<double> strided(int stride, int count, uint32_t seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<double> v(S, 0.0);
    for (int k = 0; k < count && k * stride < S; ++k) v[k * stride] = d(g);
    return v;
}

// ── the rule that intuition gets wrong ───────────────────────────────────────
TEST(PackTag, AddScalarDestroysSupportButKeepsPeriod) {
    auto v = strided(/*stride=*/8, /*count=*/32, 7);
    PackTag t{S, S, Support::ap(0, 8, 32)};
    ASSERT_LT(t.min_routable_s(), S) << "a stride-8 layout should be foldable before the add";

    for (auto& x : v) x += 0.5;                    // exact, in the clear
    t = t_add_scalar(t, 0.5);                      // transfer function

    assert_sound("add_scalar", v, t);
    EXPECT_TRUE(t.support.is_dense())
        << "adding a nonzero constant makes every zero slot live — the tag must say dense";
}

TEST(PackTag, MultScalarAndSquarePreserveBoth) {
    auto v = strided(8, 32, 11);
    PackTag t{S, S, Support::ap(0, 8, 32)};
    const int before = t.min_routable_s();

    for (auto& x : v) x *= 3.0;
    t = t_mult_scalar(t, 3.0);
    assert_sound("mult_scalar", v, t);

    for (auto& x : v) x = x * x;
    t = t_square(t);
    assert_sound("square", v, t);
    EXPECT_EQ(t.min_routable_s(), before) << "neither op should cost eligibility";
}

TEST(PackTag, MultTakesLcmOfPeriods) {
    auto a = periodic(4, 21);
    auto b = periodic(8, 22);
    std::vector<double> v(S);
    for (int i = 0; i < S; ++i) v[i] = a[i] * b[i];

    PackTag ta{S, 4, Support::dense()}, tb{S, 8, Support::dense()};
    const PackTag t = t_mult(ta, tb);
    assert_sound("mult", v, t);
    EXPECT_EQ(t.period, 8);
}

TEST(PackTag, AddUnionsSupportsConservatively) {
    auto a = strided(8, 16, 31);
    auto b = strided(8, 32, 32);
    std::vector<double> v(S);
    for (int i = 0; i < S; ++i) v[i] = a[i] + b[i];

    PackTag ta{S, S, Support::ap(0, 8, 16)}, tb{S, S, Support::ap(0, 8, 32)};
    assert_sound("add", v, t_add(ta, tb));
}

TEST(PackTag, RotationPreservesPeriod) {
    auto v = periodic(16, 41);
    std::vector<double> r(S);
    const int k = 3;
    for (int i = 0; i < S; ++i) r[i] = v[(i + k) % S];

    PackTag t{S, 16, Support::dense()};
    assert_sound("rotate", r, t_rotate(t, k));
}

// ── the case a compositional derivation gets wrong ───────────────────────────
TEST(PackTag, ReduceAllIsConstantNotDerivable) {
    auto v = periodic(S, 51);                       // fully aperiodic input
    double total = 0.0;
    for (double x : v) total += x;
    std::vector<double> out(S, total);              // rotate-and-sum-all result

    // Derived compositionally through rotate+add, the tag would still say "period S".
    PackTag naive{S, S, Support::dense()};
    for (int gap = 1; gap < S; gap *= 2) naive = t_add(naive, t_rotate(naive, gap));
    EXPECT_EQ(naive.period, S) << "compositional derivation cannot see that this is constant";

    // The primitive-level rule gets it right, and it is what makes the LN variance lane
    // recognisable as routable.
    const PackTag t = t_reduce_all(PackTag{S, S, Support::dense()});
    EXPECT_EQ(t.period, 1);
    assert_sound("reduce_all", out, t);
    EXPECT_EQ(t.min_routable_s(), 1) << "a constant is routable at the cheapest slot count";
}

TEST(PackTag, BootstrapSparseMakesOutputPeriodic) {
    PackTag t = PackTag::top(S);
    EXPECT_EQ(t.min_routable_s(), S);
    const PackTag out = t_bootstrap(t, /*routed_s=*/64);
    EXPECT_EQ(out.period, 64) << "a sparse-routed bootstrap emits s-periodic data by construction";
}

// ── the collision-free-fold predicate, which is the case (b) entry point ─────
TEST(PackTag, StridedSupportFoldsWithoutCollision) {
    // 32 live slots at stride 8 => residues mod 256 are 0,8,...,248: all distinct.
    PackTag t{S, S, Support::ap(0, 8, 32)};
    EXPECT_TRUE(t.fold_collision_free_at(256));
    EXPECT_TRUE(t.fold_collision_free_at(1024));
    // mod 128 the residues wrap at k=16 => collision.
    EXPECT_FALSE(t.fold_collision_free_at(128));
    EXPECT_FALSE(t.periodic_at(256)) << "case (b) must not be confused with case (a)";
}

// ── the case that forces plaintext tags to be REAL, not assumed ──────────────
// "if the ct and the pt have the same packing, the result keeps it" is true, and t_mult
// already computes it: lcm(P,P)=P, and intersecting identical supports is the identity. What
// is NOT safe is assuming the pt matches when we have not tagged it. This pins the
// counterexample that makes the assumption unsound.
TEST(PackTag, MaskedConstantTakesTheMaskPeriodNotTheCiphertextS) {
    // ct: a broadcast constant (what a reduce emits) — period 1, maximally routable.
    std::vector<double> ct(S, 0.75);
    PackTag t_ct{S, 1, Support::dense()};
    ASSERT_EQ(t_ct.min_routable_s(), 1);

    // pt: a per-position mask of period 32 — exactly the shape encode_*_mask_at_cached emits.
    std::vector<double> mask(S);
    for (int i = 0; i < S; ++i) mask[i] = (i % 32 == 0) ? 1.0 : 0.0;
    const PackSignature msig = analyze_packing(mask);
    const PackTag t_pt = from_signature(msig.slots, msig.period, msig.live_exact,
                                        msig.stride_exact, msig.window_exact,
                                        msig.offset_exact);

    std::vector<double> prod(S);
    for (int i = 0; i < S; ++i) prod[i] = ct[i] * mask[i];

    // Correct: the tagged product takes the mask's period.
    const PackTag t = t_mult(t_ct, t_pt);
    assert_sound("mult(ct, tagged pt)", prod, t);
    EXPECT_GT(t.period, 1) << "the product is NOT a period-1 constant any more";

    // Unsound shortcut, shown explicitly: inheriting the ciphertext's tag would claim period 1
    // on data whose real period is 32, and routing at s=1 would fold it to garbage silently.
    const PackSignature psig = analyze_packing(prod);
    EXPECT_GT(psig.period, 1);
    EXPECT_LT(t_ct.period, psig.period)
        << "inheriting the ct tag under-estimates — this is the silent-corruption path";
}

// ── fold-TRANSPARENCY: the predicate a generic router can actually use ───────
// The distinction this pins down is the whole reason for a third predicate. Collision-freedom
// says the fold LOSES NOTHING; transparency says it MOVES NOTHING. A router that masks and
// keeps reading its operands positionally needs the second, and `SPARSE_AUTO=2` grants only
// the first — so conflating them relocates data under a consumer that never agreed to it.
TEST(PackTag, FoldTransparencyIsStrictlyStrongerThanCollisionFreedom) {
    // 32 live slots at stride 8, all inside [0,256): 0,8,...,248. The fold at 256 is a no-op
    // on position, so masking the copies is enough.
    PackTag inside{S, S, Support::ap(0, 8, 32)};
    EXPECT_TRUE(inside.fold_transparent_at(256));
    EXPECT_TRUE(inside.fold_collision_free_at(256));

    // THE COUNTEREXAMPLE. Same stride, same count, but offset 512 puts every live slot ABOVE
    // 256. Residues mod 256 are still distinct, so the fold is collision-free and lossless --
    // and it lands every value at a DIFFERENT slot than it came from. Lossless, not transparent.
    PackTag above{S, S, Support::ap(512, 8, 32)};
    EXPECT_TRUE(above.fold_collision_free_at(256))
        << "collision-freedom holds: distinct residues";
    EXPECT_FALSE(above.fold_transparent_at(256))
        << "but the fold RELOCATES every value -- a positional consumer would read garbage";

    // Straddling the boundary is not transparent either: the tail wraps to the front and
    // overwrites nothing (no collision) but does move.
    PackTag straddle{S, S, Support::ap(0, 8, 64)};   // highest live index 504 >= 256
    EXPECT_FALSE(straddle.fold_transparent_at(256));
    EXPECT_TRUE(straddle.fold_transparent_at(512));   // 504 < 512

    // Degenerate ends, both directions.
    PackTag empty{S, 1, Support::empty()};
    EXPECT_TRUE(empty.fold_transparent_at(1)) << "nothing live: nothing to move";
    PackTag dense{S, S, Support::dense()};
    EXPECT_FALSE(dense.fold_transparent_at(S / 2)) << "dense occupies past s by definition";
    EXPECT_TRUE(dense.fold_transparent_at(S));
}

// Transparency must never be READ OFF a measurement. This is the shape the census actually
// showed at softmax_v.lane0_mult -- `live` drifting 10/11/12 across invocations of ONE step at
// the 1e-2 tolerance -- and it is why supports have to come from the mask that creates them.
TEST(PackTag, MeasuredSupportUnderReportsWhenTailIsBelowTolerance) {
    std::vector<double> v(S, 0.0);
    for (int k = 0; k < 12; ++k) v[k * 32] = 1.0;
    v[12 * 32] = 1e-3;          // a 13th live slot, three decades down: BELOW PACK_SIG_TOL

    const PackSignature sig = analyze_packing(v);
    EXPECT_EQ(sig.live, 12) << "the probe cannot see the 13th slot";

    // A tag built from that reading claims the support ends at slot 352 and is transparent at
    // 384 -- but slot 384 is live, so the claim is FALSE on the real data.
    const PackTag measured{S, S, Support::ap(0, 32, sig.live)};
    EXPECT_TRUE(measured.fold_transparent_at(384));
    const PackTag truth{S, S, Support::ap(0, 32, 13)};
    EXPECT_FALSE(truth.fold_transparent_at(384))
        << "the honest support is one slot wider, and that is the whole difference between "
           "a sound tag and silent corruption";
}

// ── a support tag must know WHERE it lives ───────────────────────────────────
// An offset-blind from_signature is invisible to fold_collision_free_at, which is
// offset-invariant, and fatal to fold_transparent_at, which asks where the support sits: it
// authorises folds on ciphertexts whose live slots start high in the ring. This pins the
// shape that exposes it: the rotated attention lanes, whose supports march UP the ring in
// steps of tH.
TEST(PackTag, SupportOffsetIsRecordedSoTransparencyCannotBeFaked) {
    constexpr int tH = 512;
    // A lane rotated to sit high in the ring: 12 live slots at stride 32, starting at 30049.
    std::vector<double> v(S * 32, 0.0);          // a ring big enough to hold the offset
    const int off = 30049;
    for (int k = 0; k < 12; ++k) v[off + k * 32] = 1.0;
    const PackSignature sig = analyze_packing(v);
    EXPECT_EQ(sig.offset_exact, off) << "the first live index must be recorded";

    const PackTag t = from_signature(sig.slots, sig.period, sig.live_exact, sig.stride_exact,
                                     sig.window_exact, sig.offset_exact);
    EXPECT_FALSE(t.fold_transparent_at(1024))
        << "support starts at " << off << " — a fold at 1024 RELOCATES it; claiming "
           "transparency here is the 167-violation bug";
    EXPECT_TRUE(t.fold_collision_free_at(1024))
        << "and note collision-freedom still holds — which is exactly why the offset bug hid: "
           "the old predicate never looked at the offset";

    // The same 12 slots low in the ring ARE transparent — the predicate is not just always-false.
    std::vector<double> lo(S * 32, 0.0);
    for (int k = 0; k < 12; ++k) lo[k * 32] = 1.0;
    const PackSignature slo = analyze_packing(lo);
    const PackTag tlo = from_signature(slo.slots, slo.period, slo.live_exact, slo.stride_exact,
                                       slo.window_exact, slo.offset_exact);
    EXPECT_EQ(slo.offset_exact, 0);
    EXPECT_TRUE(tlo.fold_transparent_at(1024));

    // And a rotation must carry the offset with it, or the same hole reopens one op later.
    const PackTag rot = t_rotate(tlo, -tH);      // rotate lane 0 up by one head stride
    EXPECT_EQ(rot.support.offset, tH);
    EXPECT_FALSE(rot.fold_transparent_at(tH)) << "rotated past s: no longer transparent";
    EXPECT_TRUE(rot.fold_transparent_at(2048));
}

// ── the fold + support-shaped restore, simulated in the clear ────────────────
// The route's whole claim in one test: for a COLLISION-FREE support the fold puts every value
// back at its ORIGINAL slot (because the output is s-periodic, so the class mean sits at every
// congruent slot), and a support-shaped mask ×copies recovers it exactly — no rotation, and no
// requirement that the support live inside [0,s). The contract this rests on is t_bootstrap's
// "output is s-periodic BY CONSTRUCTION", whose value at each slot is the arithmetic MEAN of
// the copies.
TEST(PackTag, CollisionFreeFoldRecoversValuesInPlaceWithASupportShapedMask) {
    constexpr int N = 32768, s = 1024, off = 30049, stride = 32, n = 12;
    const double copies = (double)N / s;

    std::vector<double> v(N, 0.0);
    for (int k = 0; k < n; ++k) v[off + k * stride] = 1.0 + 0.1 * k;

    const PackTag t{N, N, Support::ap(off, stride, n)};
    EXPECT_FALSE(t.fold_transparent_at(s)) << "the narrow predicate refuses this";
    EXPECT_TRUE(t.fold_collision_free_at(s)) << "the general one accepts it";

    // fold: slot i <- mean of its residue class mod s
    std::vector<double> folded(N, 0.0);
    for (int r = 0; r < s; ++r) {
        double sum = 0;
        for (int j = r; j < N; j += s) sum += v[j];
        for (int j = r; j < N; j += s) folded[j] = sum / copies;
    }
    // restore: keep exactly the support, scaled by copies
    std::vector<double> out(N, 0.0);
    for (int k = 0; k < n; ++k) { const int i = off + k * stride; out[i] = folded[i] * copies; }
    for (int i = 0; i < N; ++i)
        ASSERT_NEAR(out[i], v[i], 1e-12) << "slot " << i << " not recovered in place";

    // An OVER-APPROXIMATED mask stays sound: the extra slots are alone in their residue classes
    // (collision-freedom holds on the larger support too), so their classes hold no live value
    // and they come back 0 rather than as spurious data.
    std::vector<double> out2(N, 0.0);
    for (int k = 0; k < 2 * n; ++k) {
        const int i = (off + k * stride) % N;
        out2[i] = folded[i] * copies;
    }
    for (int i = 0; i < N; ++i)
        ASSERT_NEAR(out2[i], v[i], 1e-12) << "superset mask produced spurious data at " << i;
}

}  // namespace
