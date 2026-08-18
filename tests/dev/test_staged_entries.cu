// Isolated round-trip gate for the cf-attention entry staging (include/staged_entries.h).
//
// Context: the three T64 staging-active runs all threw at the decode-token decrypt while
// the staged prefill TRACE was clean vs baseline — so either staging corrupts state the
// trace can't see, or something else in the binary does. This test proves (or refutes),
// touching NO production flow, that the EXACT staging patterns the filling softmax uses
// round-trip ciphertexts bit-exactly through the pinned arena:
//   (1) push -> load -> value intact (all phases' levels/degs),
//   (2) the sum pass's load -> read -> drop -> LOAD AGAIN,
//   (3) the phase chain: overwrite the same per-index slot with a new value (s->e->y->z2),
//       including slot GROWTH (later phase at a lower level = MORE limbs),
//   (4) interleaving with block-KV-style KvStoreStaged/KvLoadStaged keys in the SAME arena.
// PASS => the staging container honours the FIDESlib staged-slot contract; look elsewhere.
// FAIL => the offending pattern is pinned to a single EXPECT.
//
// Run: cmake -S . -B build && cmake --build build --target test_staged_entries
//      (GPU node; 1 GPU, debug QOS is plenty)

#include "test_helpers.h"
#include "ckks_fixture.h"
#include "staged_entries.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using test_helpers::CkksFixture;
using test_helpers::decrypt_slots;

namespace {

constexpr double kTol = 1e-5;   // >> GPU decrypt nondeterminism (~7e-9), << any corruption

double slot0(Inference& inf, const PackedCtx& pc) {
    return decrypt_slots(inf, pc)[0];
}

PackedCtx make_ct(Inference& inf, double v, int level, int deg) {
    Ctx c = encrypt_const(inf.fhe->cc, v, inf.slots, inf.fhe->pk());
    inf.fhe->drop_to_level(c, level);
    PackedCtx pc{c, Packing{}};
    if (deg == 2) inf.fhe->inplace_mult(pc, 1.0);   // scalar mult -> deg-2, value unchanged
    return pc;
}

}  // namespace

class StagedEntriesTest : public CkksFixture {};

TEST_F(StagedEntriesTest, PhaseChainRoundTripAndArenaInterleave) {
    Inference inf = make_inf();
    const int n = 96;   // > the 64-entry auto gate; chunk-2-sized

    // ---- (4) pre-seed block-KV-style neighbours in the same arena ----
    std::vector<PackedCtx> kv_like;
    for (int i = 0; i < 8; ++i) {
        kv_like.push_back(make_ct(inf, 100.0 + i, 17, 2));
        inf.cc()->KvStoreStaged(kv_like.back().ct, "test.kv:" + std::to_string(i), nullptr);
        cudaDeviceSynchronize();
        inf.cc()->KvEvict(kv_like.back().ct);
    }

    // ---- (1) push (store+evict) then load: values intact ----
    StagedEntries s(inf, /*active=*/true);
    for (int i = 0; i < n; ++i)
        s.push(make_ct(inf, 1.0 + i * 0.001, 18, 2));   // "scores": L18 deg2
    s.seal();
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(slot0(inf, s.load(i)), 1.0 + i * 0.001, kTol) << "score slot " << i;
        s.drop(i);
    }

    // ---- (3) phase chain: overwrite each slot with the next phase's value ----
    StagedEntries e(inf, true);
    for (int i = 0; i < n; ++i) {
        PackedCtx& sc = s.load(i);
        PackedCtx ei = inf.fhe->mult(sc, 2.0);   // e = 2*s, L18->19-ish, deg2
        s.clear(i);
        e.push(std::move(ei));
    }
    e.seal();

    // ---- (2) the sum pass: load -> read -> drop -> LOAD AGAIN later ----
    PackedCtx acc = inf.fhe->clone(e.load(0));
    e.drop(0);
    for (int i = 1; i < n; ++i) {
        inf.fhe->inplace_add(acc, e.load(i));
        e.drop(i);
    }
    const double want_sum = [&] {
        double t = 0.0;
        for (int i = 0; i < n; ++i) t += 2.0 * (1.0 + i * 0.001);
        return t;
    }();
    EXPECT_NEAR(slot0(inf, acc), want_sum, kTol * n) << "sum over staged entries";

    // second full pass over the SAME slots (division-loop pattern)
    StagedEntries y(inf, true);
    for (int i = 0; i < n; ++i) {
        PackedCtx yi = inf.fhe->mult(e.load(i), 0.5);   // y = e/2 = s
        e.clear(i);
        y.push(std::move(yi));
    }
    y.seal();

    // ---- (3b) slot GROWTH: rebuild y at a LOWER level (more limbs) via set() ----
    for (int i = 0; i < n; ++i) {
        PackedCtx big = make_ct(inf, 3.0 + i * 0.001, 16, 2);   // L16 = the largest form
        y.load(i);
        y.set(i, std::move(big));
    }
    y.seal();
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(slot0(inf, y.load(i)), 3.0 + i * 0.001, kTol) << "grown slot " << i;
        y.drop(i);
    }

    // ---- (4b) the block-KV neighbours survived everything above ----
    for (int i = 0; i < 8; ++i) {
        inf.cc()->KvLoadStaged(kv_like[i].ct, "test.kv:" + std::to_string(i), nullptr);
        cudaDeviceSynchronize();
        EXPECT_NEAR(slot0(inf, kv_like[i]), 100.0 + i, kTol) << "kv-like slot " << i;
    }
}

TEST_F(StagedEntriesTest, MutateAfterLoadRefinePattern) {
    Inference inf = make_inf();
    const int n = 80;

    StagedEntries y(inf, true);
    for (int i = 0; i < n; ++i)
        y.push(make_ct(inf, 0.5 + i * 0.002, 19, 2));
    y.seal();

    // refine round: load -> im_cleanse in place -> square -> store derived into same slot
    StagedEntries z2(inf, true);
    for (int i = 0; i < n; ++i) {
        PackedCtx& yi = y.load(i);
        inf.fhe->inplace_im_cleanse(yi);            // 2*Re
        PackedCtx sq = inf.fhe->square(yi);         // (2v)^2
        y.clear(i);
        z2.push(inf.fhe->mult(sq, 0.25));           // back to v^2
    }
    z2.seal();
    for (int i = 0; i < n; ++i) {
        const double v = 0.5 + i * 0.002;
        EXPECT_NEAR(slot0(inf, z2.load(i)), v * v, kTol) << "refine slot " << i;
        z2.drop(i);
    }
}
