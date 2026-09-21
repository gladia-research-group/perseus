// Gate for the PERIODIC packing (include/packing/periodic/periodic_layout.h).
//
// test_ptmult_isolation showed one ct x pt multiply spends 0.51 % of its wall on
// arithmetic and 53.9 % (68 % at production limbs) shipping the plaintext. The
// periodic layout's claim is that a d-periodic weight slot vector encodes to
// limbs holding at most 2d distinct residues instead of N, so the shipped bytes
// can drop ~32x. That claim is a PREMISE about this library's encoder, not a
// theorem about our code — this file measures it before anything is built on it.
//
//   1. LayoutStructure — the two layouts are genuinely different structures:
//      the incumbent diagonal is block-constant and NOT periodic; the proposed
//      one is periodic and NOT block-constant. Host-only, no FHE.
//   2. EncodedFootprint  — THE GATE. Counts distinct residues per limb in the
//      limbs FIDESlib actually uploads (GetAllElements(), the same array
//      GetRawPlainText copies into RawPlainText::sub_0), for: today's
//      block-constant diagonal, the proposed periodic one, and a dense random
//      control. Green = periodic collapses to ~2d, control stays ~N.
//   3. LoadCost — LoadPlaintext A/B. Expected to be EQUAL today: sub_0 is
//      [numRes][N] regardless of the `slots` field, so the win is unrealised
//      until the transfer path stores the compressed form. This quantifies the
//      headroom rather than a saving.
//   4. BlockRotFold — the cost side. Token-major breaks per-token rotation;
//      verifies under FHE that the two-term decomposition with the masks folded
//      into the diagonal reproduces diag * blockrot_k(x) exactly.
//   5. ShipPathPrice — G0 of docs/STATUS_periodic_layout.md (2026-07-24): the
//      2026-07-22 coeff-staging ship replaced the transfer wall tests 2-3 were
//      aimed at, so the layout must be re-priced against the SHIP load path
//      (pinned staging + 1-limb coeff encode + GPU expand). Prints the kill
//      inequality; needs FHE_PIN_STAGE=1, FHE_STAGE_RELEASE_CPU unset.
//   6. SparseCoeffProbe — G0b: the periodic sparsity lives in the COEFFICIENT
//      domain (<= 2d nonzero coeffs on the X^(N/2d) subring stride). Measures
//      whether the 1-limb coeff encode of a periodic weight is exactly sparse
//      (a 2d-value ship + INTT skip would be exact) or rounding-dusty.
//
//   build: cmake --build build --parallel 16 --target test_periodic_layout
//   run:   D=1024 LEVEL=19 build/bin/test_periodic_layout

#include "fideslib_wrapper.h"
#include "inference.h"
#include "packing/periodic/periodic_layout.h"
#include "test_helpers.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_set>
#include <vector>

using namespace test_helpers;

namespace {

int env_int(const char* k, int dflt) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::atoi(v) : dflt;
}

double median_us(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

// The limbs FIDESlib uploads: PlaintextImpl::cpu holds the OpenFHE plaintext,
// whose DCRTPoly limbs are exactly what GetRawPlainText copies into sub_0.
// Same any_cast idiom as fideslib_wrapper.h:381.
struct LimbStats {
    int n_limbs = 0;
    int n_coeff = 0;
    int format = -1;
    std::vector<size_t> distinct;   // per limb
};

LimbStats limb_stats(const Ptx& pt) {
    const auto& ptImpl = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
    const auto& elems  = ptImpl->GetElement<lbcrypto::DCRTPoly>().GetAllElements();

    LimbStats st;
    st.n_limbs = static_cast<int>(elems.size());
    if (st.n_limbs == 0) return st;
    st.format = static_cast<int>(elems[0].GetFormat());
    st.n_coeff = static_cast<int>(elems[0].GetValues().GetLength());
    st.distinct.reserve(elems.size());
    for (const auto& limb : elems) {
        const auto& vals = limb.GetValues();
        std::unordered_set<uint64_t> uniq;
        uniq.reserve(static_cast<size_t>(st.n_coeff) * 2);
        for (size_t i = 0; i < vals.GetLength(); ++i)
            uniq.insert(vals[i].ConvertToInt());
        st.distinct.push_back(uniq.size());
    }
    return st;
}

size_t max_distinct(const LimbStats& s) {
    size_t m = 0;
    for (size_t d : s.distinct) m = std::max(m, d);
    return m;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b, int n) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int D_    = 0;   // feature dim / period
int K_    = 0;   // block-rotation step under test
int LEVEL_ = 0;

class PeriodicLayout : public ::testing::Test {
 protected:
    static void SetUpTestSuite() {
        D_     = env_int("D", 1024);
        K_     = env_int("K", 7);
        LEVEL_ = env_int("LEVEL", 19);

        CKKSContextOptions o = default_ckks_options();
        // the two rotations the block-rotation decomposition needs, and nothing else
        o.extra_rot_steps = {static_cast<int32_t>(K_), static_cast<int32_t>(K_ - D_)};
        ctx_   = make_ckks_context(o);
        slots_ = static_cast<int>(ctx_->cc->GetRingDimension() / 2);
    }
    static void TearDownTestSuite() { ctx_.reset(); }
    static std::shared_ptr<CKKSContext> ctx_;
    static int slots_;
};
std::shared_ptr<CKKSContext> PeriodicLayout::ctx_;
int PeriodicLayout::slots_ = 0;

// Today's ViT weight diagonal shape, verbatim from
// src/packing/diagonal/diagonal_linear_utils.cu:65-101 — block-constant.
std::vector<double> incumbent_block_constant(const std::vector<double>& period, int slots) {
    const int d = static_cast<int>(period.size());
    const int t_out = slots / d;
    std::vector<double> v(static_cast<size_t>(slots), 0.0);
    for (int j = 0; j < d; ++j)
        for (int tok = 0; tok < t_out; ++tok)
            v[static_cast<size_t>(j) * t_out + tok] = period[j];
    return v;
}

}  // namespace

// ---------------------------------------------------------------------------
TEST_F(PeriodicLayout, LayoutStructure) {
    auto p = periodic::make_params(D_, slots_);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> period(static_cast<size_t>(D_));
    for (auto& x : period) x = dist(rng);

    const std::vector<double> incumbent = incumbent_block_constant(period, slots_);
    const std::vector<double> proposed  = periodic::expand(period, slots_);

    std::printf("\n[periodic] d=%d slots=%d tokens/ct=%d t_out(incumbent)=%d\n",
                p.d, p.slots, p.T, slots_ / D_);

    // Same information, opposite orientation.
    EXPECT_TRUE(periodic::is_block_constant(incumbent, slots_ / D_));
    EXPECT_FALSE(periodic::is_periodic(incumbent, D_));
    EXPECT_TRUE(periodic::is_periodic(proposed, D_));
    EXPECT_FALSE(periodic::is_block_constant(proposed, slots_ / D_));

    std::printf("[periodic] incumbent: block-constant=%d periodic=%d\n",
                (int)periodic::is_block_constant(incumbent, slots_ / D_),
                (int)periodic::is_periodic(incumbent, D_));
    std::printf("[periodic] proposed : block-constant=%d periodic=%d\n",
                (int)periodic::is_block_constant(proposed, slots_ / D_),
                (int)periodic::is_periodic(proposed, D_));
    std::printf("[periodic] predicted distinct residues/limb: <= %d of %d (%.1fx)\n",
                periodic::distinct_residue_bound(D_, slots_), 2 * slots_,
                periodic::compression_ratio(D_, slots_));
}

// ---------------------------------------------------------------------------
// THE GATE.
TEST_F(PeriodicLayout, EncodedFootprint) {
    auto& fhe = *ctx_;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> period(static_cast<size_t>(D_));
    for (auto& x : period) x = dist(rng);
    std::vector<double> dense(static_cast<size_t>(slots_));
    for (auto& x : dense) x = dist(rng);

    const auto v_incumbent = incumbent_block_constant(period, slots_);
    const auto v_periodic  = periodic::expand(period, slots_);

    auto enc = [&](const std::vector<double>& v) {
        return fhe.cc->MakeCKKSPackedPlaintext(v, /*noiseScaleDeg=*/1,
                                               static_cast<uint32_t>(LEVEL_));
    };

    Ptx pt_inc  = enc(v_incumbent);
    Ptx pt_per  = enc(v_periodic);
    Ptx pt_rand = enc(dense);
    // The same period encoded as a genuinely sparse (D-slot) plaintext.
    Ptx pt_sparse = fhe.cc->MakeCKKSPackedPlaintext(period, 1, static_cast<uint32_t>(LEVEL_));

    const LimbStats s_inc  = limb_stats(pt_inc);
    const LimbStats s_per  = limb_stats(pt_per);
    const LimbStats s_rand = limb_stats(pt_rand);
    const LimbStats s_spa  = limb_stats(pt_sparse);

    std::printf("\n[periodic] --- encoded footprint (level=%d, %d limbs, %d coeffs/limb, format=%d) ---\n",
                LEVEL_, s_per.n_limbs, s_per.n_coeff, s_per.format);
    std::printf("[periodic] bound for d=%d: <= %d distinct residues/limb\n",
                D_, periodic::distinct_residue_bound(D_, slots_));

    auto report = [&](const char* name, const LimbStats& s) {
        if (s.n_limbs == 0 || s.n_coeff == 0) { std::printf("[periodic] %-28s (empty)\n", name); return; }
        const size_t mx = max_distinct(s);
        std::printf("[periodic] %-28s max distinct/limb = %8zu of %6d  (%6.1fx compressible)\n",
                    name, mx, s.n_coeff,
                    static_cast<double>(s.n_coeff) / static_cast<double>(std::max<size_t>(mx, 1)));
    };
    report("A block-constant (TODAY)", s_inc);
    report("B periodic (PROPOSED)", s_per);
    report("C dense random (control)", s_rand);
    report("D sparse D-slot encode", s_spa);

    ASSERT_GT(s_per.n_limbs, 0);
    ASSERT_GT(s_per.n_coeff, 0);

    // Control must be dense — if it is not, the distinct-count is measuring
    // something other than what we think and every other row is meaningless.
    EXPECT_GT(static_cast<double>(max_distinct(s_rand)),
              0.5 * static_cast<double>(s_rand.n_coeff))
        << "dense control collapsed: distinct-residue counting is not measuring encoder structure";

    // THE claim.
    const size_t bound = static_cast<size_t>(periodic::distinct_residue_bound(D_, slots_));
    EXPECT_LE(max_distinct(s_per), bound)
        << "periodic slot vector did NOT collapse to <= 2d distinct residues — the "
           "subring premise does not hold in this encoder; the periodic packing has no basis";

    // And the incumbent must NOT already have it, else there is nothing to gain.
    EXPECT_GT(max_distinct(s_inc), bound)
        << "block-constant is already as compressible as periodic — no reason to change layout";

    // Informational: is replication bit-identical to sparse packing here?
    if (s_spa.n_limbs == s_per.n_limbs && s_spa.n_coeff == s_per.n_coeff) {
        const auto& a = std::any_cast<const lbcrypto::Plaintext&>(pt_per->cpu)
                            ->GetElement<lbcrypto::DCRTPoly>().GetAllElements();
        const auto& b = std::any_cast<const lbcrypto::Plaintext&>(pt_sparse->cpu)
                            ->GetElement<lbcrypto::DCRTPoly>().GetAllElements();
        size_t same = 0, total = 0;
        for (size_t l = 0; l < a.size(); ++l)
            for (size_t i = 0; i < a[l].GetValues().GetLength(); ++i, ++total)
                if (a[l].GetValues()[i] == b[l].GetValues()[i]) ++same;
        std::printf("[periodic] replicated-vs-sparse-encode limbs identical: %zu/%zu (%.1f%%)\n",
                    same, total, total ? 100.0 * static_cast<double>(same) / static_cast<double>(total) : 0.0);
    } else {
        std::printf("[periodic] replicated-vs-sparse-encode: shape differs (%d x %d vs %d x %d)\n",
                    s_per.n_limbs, s_per.n_coeff, s_spa.n_limbs, s_spa.n_coeff);
    }
}

// ---------------------------------------------------------------------------
TEST_F(PeriodicLayout, LoadCost) {
    auto& fhe = *ctx_;
    const int iters = env_int("ITERS", 50);
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> period(static_cast<size_t>(D_));
    for (auto& x : period) x = dist(rng);

    auto time_load = [&](const std::vector<double>& v, const char* name) {
        Ptx pt = fhe.cc->MakeCKKSPackedPlaintext(v, 1, static_cast<uint32_t>(LEVEL_));
        std::vector<double> t;
        t.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            fhe.cc->LoadPlaintext(pt, nullptr);
            cudaDeviceSynchronize();
            t.push_back(std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count());
            if (pt->gpu) { fhe.cc->EvictDevicePlaintext(pt->gpu); pt->gpu = 0; pt->loaded = false; }
        }
        const double m = median_us(t);
        std::printf("[periodic] load %-26s median = %9.1f us\n", name, m);
        return m;
    };

    std::printf("\n[periodic] --- load cost (level=%d, iters=%d) ---\n", LEVEL_, iters);
    const double t_inc = time_load(incumbent_block_constant(period, slots_), "A block-constant (TODAY)");
    const double t_per = time_load(periodic::expand(period, slots_),         "B periodic (PROPOSED)");

    const double ratio = periodic::compression_ratio(D_, slots_);
    std::printf("[periodic] delta = %+.1f us (%.1f%%) — expected ~0: RawPlainText::sub_0 is\n"
                "[periodic]   [numRes][N] regardless of slots, so the structure is not yet exploited.\n",
                t_per - t_inc, t_inc > 0 ? 100.0 * (t_per - t_inc) / t_inc : 0.0);
    std::printf("[periodic] HEADROOM if the transfer stored only unique residues: %.1f us -> %.1f us (%.1fx)\n",
                t_per, t_per / ratio, ratio);
}

// ---------------------------------------------------------------------------
// The cost side: token-major breaks per-token rotation. Verify the fix.
TEST_F(PeriodicLayout, BlockRotFold) {
    auto& fhe = *ctx_;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // token-major activations: distinct per token, so NOT periodic
    std::vector<double> x(static_cast<size_t>(slots_));
    for (auto& v : x) v = dist(rng);

    // one weight diagonal
    std::vector<double> W(static_cast<size_t>(D_) * D_);
    for (auto& w : W) w = dist(rng) * 0.05;
    const std::vector<double> diag = periodic::weight_diag_period(W, D_, K_);

    // host truth: diag (replicated) * per-token-rotated x
    const std::vector<double> br = periodic::block_rot_reference(x, K_, D_);
    const std::vector<double> diag_full = periodic::expand(diag, slots_);
    std::vector<double> expect(static_cast<size_t>(slots_));
    for (int s = 0; s < slots_; ++s) expect[s] = diag_full[s] * br[s];

    // FHE: sum of the folded terms, i.e. exactly what the linear would run
    Ctx ct = encrypt(fhe.cc, fhe.cc->MakeCKKSPackedPlaintext(x, 1, LEVEL_), fhe.pk());
    const auto terms = periodic::folded_diag_periods(diag, K_, D_);

    Ctx acc;
    int n_rot = 0, n_pt = 0;
    for (const auto& t : terms) {
        if (t.empty) continue;
        Ctx r = (t.rot == 0) ? ct->Clone() : fhe.cc->EvalRotate(ct, t.rot);
        if (t.rot != 0) ++n_rot;
        Ptx p = fhe.cc->MakeCKKSPackedPlaintext(periodic::expand(t.mask_period, slots_),
                                                1, static_cast<uint32_t>(LEVEL_));
        ++n_pt;
        Ctx prod = fhe.cc->EvalMult(r, p);
        acc = acc ? fhe.cc->EvalAdd(acc, prod) : prod;

        // every folded plaintext must stay periodic, else it stops being compressible
        EXPECT_TRUE(periodic::is_periodic(periodic::expand(t.mask_period, slots_), D_));
    }
    ASSERT_TRUE(static_cast<bool>(acc));

    const std::vector<double> got = decrypt_slots(fhe, acc);
    const double err = max_abs_diff(got, expect, slots_);
    double scale = 0.0;
    for (int s = 0; s < slots_; ++s) scale = std::max(scale, std::fabs(expect[s]));

    std::printf("\n[periodic] --- block-rot fold (d=%d k=%d) ---\n", D_, K_);
    std::printf("[periodic] rotations=%d plaintexts=%d (incumbent: 1 and 1)\n", n_rot, n_pt);
    std::printf("[periodic] max|fhe - host| = %.3e  (signal %.3e, rel %.3e)\n",
                err, scale, scale > 0 ? err / scale : 0.0);

    EXPECT_LT(err, 1e-4 * std::max(scale, 1e-6))
        << "the two-term block-rotation fold does not reproduce diag * blockrot_k(x); "
           "token-major BSGS is not correct as decomposed";
}

// ---------------------------------------------------------------------------
TEST_F(PeriodicLayout, ShipPathPrice) {
    // G0 re-price (docs/STATUS_periodic_layout.md). A periodic diagonal costs
    // 2 folded plaintexts + 2 rotations + 2 pt-mults + 1 add against the
    // incumbent's 1 pt + 1 rot + 1 mult, so even a FREE periodic load leaves
    //     T_per_floor = 2*mult + 2*rot + add
    // against the incumbent's whole per-diagonal critical path
    //     T_inc = coeff_load(upload + expand) + mult + rot.
    // T_per_floor >= T_inc  ==>  the layout cannot pay under the ship config
    // no matter how well the sparse upload is engineered.
    const char* pin = std::getenv("FHE_PIN_STAGE");
    if (!(pin && *pin && std::atoi(pin) != 0))
        GTEST_SKIP() << "FHE_PIN_STAGE not set — the ship staged path is off";
    const char* rel = std::getenv("FHE_STAGE_RELEASE_CPU");
    if (rel && *rel && std::atoi(rel) != 0)
        GTEST_SKIP() << "FHE_STAGE_RELEASE_CPU=1 forbids the iterated re-extraction this bench needs";

    auto& fhe = *ctx_;
    const int iters = env_int("ITERS", 30);
    const uint32_t lv  = static_cast<uint32_t>(LEVEL_);
    const uint32_t lv1 = static_cast<uint32_t>(fhe.total_depth);   // 1-limb (q0) encode level
    const int numRes   = fhe.total_depth + 1 - LEVEL_;
    ASSERT_GT(numRes, 1) << "LEVEL must sit below total_depth for the price to mean anything";
    const double sf_target = fhe.cc->ScalingFactorReal(lv);
    const double ratio     = sf_target / fhe.cc->ScalingFactorReal(lv1);

    fideslib::PrewarmStageArenas();

    std::mt19937 rng(4242);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> period(static_cast<size_t>(D_));
    for (auto& w : period) w = dist(rng) * 0.05;
    const std::vector<double> v_per = periodic::expand(period, slots_);
    std::vector<double> v_scaled(v_per);
    for (auto& w : v_scaled) w *= ratio;

    std::vector<double> x(static_cast<size_t>(slots_));
    for (auto& v : x) v = dist(rng);
    Ctx ct = encrypt(fhe.cc, fhe.cc->MakeCKKSPackedPlaintext(x, 1, lv), fhe.pk());

    auto enc_full = [&] { return fhe.cc->MakeCKKSPackedPlaintext(v_per, 1, lv); };
    auto enc_coeff = [&] {
        Ptx pt = fhe.cc->MakeCKKSPackedPlaintext(v_scaled, 1, lv1);
        fhe.cc->MarkCoeffStaged(pt, lv, sf_target);
        return pt;
    };

    auto now = [] { return std::chrono::steady_clock::now(); };
    auto us  = [](auto t0, auto t1) {
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    };

    // Host encode cost (the loader-side term staging overlaps under compute).
    auto time_host = [&](auto&& fn) {
        std::vector<double> t; t.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            const auto t0 = now(); fn(); t.push_back(us(t0, now()));
        }
        return median_us(t);
    };
    const double t_enc_full  = time_host([&] { auto p = enc_full();  (void)p; });
    const double t_enc_coeff = time_host([&] { auto p = enc_coeff(); (void)p; });

    // Staged extract (worker-side, host-only) + staged load (H2D [+ GPU expand]).
    struct Split { double extract_us; double load_us; };
    auto time_staged = [&](Ptx pt) {
        std::vector<double> te, tl;
        te.reserve(iters); tl.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            fhe.cc->BeginStageBlock();
            const auto e0 = now();
            fhe.cc->ExtractRawPlaintext(pt);
            te.push_back(us(e0, now()));
            const auto l0 = now();
            fhe.cc->LoadPlaintext(pt, nullptr);
            cudaDeviceSynchronize();
            tl.push_back(us(l0, now()));
            if (pt->gpu) { fhe.cc->EvictDevicePlaintext(pt->gpu); pt->gpu = 0; pt->loaded = false; }
        }
        return Split{median_us(te), median_us(tl)};
    };
    Ptx pt_full  = enc_full();
    Ptx pt_coeff = enc_coeff();
    const Split s_full  = time_staged(pt_full);    // numRes pinned limbs, no expand
    const Split s_coeff = time_staged(pt_coeff);   // 1 limb + INTT->grow->broadcast->NTT

    // Compute ops at the same level, pt resident (the production steady state).
    fhe.cc->BeginStageBlock();
    fhe.cc->ExtractRawPlaintext(pt_coeff);
    fhe.cc->LoadPlaintext(pt_coeff, nullptr);
    cudaDeviceSynchronize();
    auto time_gpu = [&](auto&& fn) {
        for (int i = 0; i < 3; ++i) { fn(); }   // warmup (allocator, kernels)
        cudaDeviceSynchronize();
        std::vector<double> t; t.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            const auto t0 = now(); fn(); cudaDeviceSynchronize(); t.push_back(us(t0, now()));
        }
        return median_us(t);
    };
    const double t_mult = time_gpu([&] { auto r = fhe.cc->EvalMult(ct, pt_coeff); (void)r; });
    const double t_rot  = time_gpu([&] { auto r = fhe.cc->EvalRotate(ct, K_); (void)r; });
    Ctx ct2 = fhe.cc->EvalRotate(ct, K_);
    const double t_add  = time_gpu([&] { auto r = fhe.cc->EvalAdd(ct, ct2); (void)r; });

    const double upload1_est = s_full.load_us / numRes;         // ~linear in pinned limbs
    const double expand_est  = s_coeff.load_us - upload1_est;   // INTT + grow + broadcast + NTT

    const double T_inc       = s_coeff.load_us + t_mult + t_rot;
    const double T_per_floor = 2.0 * t_mult + 2.0 * t_rot + t_add;
    // Sparse-coeff best case: upload free, but BOTH folded pts still expand on GPU.
    const double T_per_best  = 2.0 * expand_est + T_per_floor;

    std::printf("\n[periodic] --- ship-path price (d=%d level=%d numRes=%d iters=%d) ---\n",
                D_, LEVEL_, numRes, iters);
    std::printf("[periodic] host encode   full=%9.1f us   coeff(1-limb)=%9.1f us\n", t_enc_full, t_enc_coeff);
    std::printf("[periodic] worker extract full=%8.1f us   coeff=%8.1f us\n", s_full.extract_us, s_coeff.extract_us);
    std::printf("[periodic] staged load   full(%d limbs)=%8.1f us   coeff(expand)=%8.1f us\n",
                numRes, s_full.load_us, s_coeff.load_us);
    std::printf("[periodic]   -> upload(1 limb) ~= %.1f us, GPU expand ~= %.1f us\n", upload1_est, expand_est);
    std::printf("[periodic] compute       mult=%8.1f us   rot=%8.1f us   add=%8.1f us\n", t_mult, t_rot, t_add);
    std::printf("[periodic] per diagonal: T_inc = load+mult+rot          = %9.1f us\n", T_inc);
    std::printf("[periodic]              T_per_floor = 2mult+2rot+add    = %9.1f us (load assumed FREE)\n", T_per_floor);
    std::printf("[periodic]              T_per_best  = floor + 2*expand  = %9.1f us (sparse upload, expand stays)\n", T_per_best);
    std::printf("[periodic] loader side/pt: coeff enc+extract = %.1f us; periodic ships 2x of these\n",
                t_enc_coeff + s_coeff.extract_us);
    if (T_per_floor >= T_inc) {
        std::printf("[periodic] VERDICT: CANNOT PAY — even a free periodic load loses %.1f us/diagonal "
                    "to the doubled mult+rot. Park permanently under the ship config.\n",
                    T_per_floor - T_inc);
    } else if (T_per_best >= T_inc) {
        std::printf("[periodic] VERDICT: CANNOT PAY at the measured expand cost — the doubled GPU expand "
                    "(%.1f us) eats the margin (%.1f us). Only an expand-side cut could revive it.\n",
                    2.0 * expand_est, T_inc - T_per_floor);
    } else {
        std::printf("[periodic] VERDICT: MARGIN EXISTS — best-case periodic saves %.1f us/diagonal "
                    "(%.1f%%). Proceed to G1 (BSGS giant-step price) before any wiring.\n",
                    T_inc - T_per_best, 100.0 * (T_inc - T_per_best) / T_inc);
    }

    EXPECT_GT(s_coeff.load_us, 0.0);
    EXPECT_GT(t_mult, 0.0);
}

// ---------------------------------------------------------------------------
TEST_F(PeriodicLayout, SparseCoeffProbe) {
    // G0b: is the 1-limb coeff encode of a periodic weight EXACTLY sparse in the
    // coefficient domain (<= 2d nonzeros, all on the X^(N/2d) subring stride)?
    // If yes, a sparse-coeff ship (2d values + scatter, INTT skippable) is exact.
    auto& fhe = *ctx_;
    const uint32_t lv  = static_cast<uint32_t>(LEVEL_);
    const uint32_t lv1 = static_cast<uint32_t>(fhe.total_depth);
    const double sf_target = fhe.cc->ScalingFactorReal(lv);
    const double ratio     = sf_target / fhe.cc->ScalingFactorReal(lv1);

    std::mt19937 rng(31337);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> period(static_cast<size_t>(D_));
    for (auto& w : period) w = dist(rng) * 0.05;

    auto enc1 = [&](const std::vector<double>& v) {
        std::vector<double> scaled(v);
        for (auto& w : scaled) w *= ratio;
        return fhe.cc->MakeCKKSPackedPlaintext(scaled, 1, lv1);   // 1-limb, ship recipe
    };

    const int stride = slots_ / D_;   // subring coeff stride N/(2d)
    struct CoeffStats {
        int format = -1; size_t n = 0;
        size_t zeros = 0, dust4 = 0, dust20 = 0, mid = 0, structural = 0, off_stride = 0;
    };
    auto coeff_stats = [&](const Ptx& pt) {
        const auto& ptImpl = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
        const auto& elems  = ptImpl->GetElement<lbcrypto::DCRTPoly>().GetAllElements();
        CoeffStats st;
        if (elems.empty()) return st;
        auto poly = elems[0];   // copy: SetFormat mutates
        st.format = static_cast<int>(poly.GetFormat());
        if (poly.GetFormat() != Format::COEFFICIENT)
            poly.SetFormat(Format::COEFFICIENT);
        const auto& vals = poly.GetValues();
        const uint64_t q = poly.GetModulus().ConvertToInt();
        st.n = vals.GetLength();
        for (size_t i = 0; i < st.n; ++i) {
            const uint64_t c = vals[i].ConvertToInt();
            const uint64_t m = std::min(c, q - c);   // centered lift magnitude
            if (m == 0)                 { ++st.zeros; }
            else if (m <= 4)            { ++st.dust4; }
            else if (m <= (1ull << 20)) { ++st.dust20; }
            else if (m <= (1ull << 32)) { ++st.mid; }
            else {
                ++st.structural;
                if (i % static_cast<size_t>(stride) != 0) ++st.off_stride;
            }
        }
        return st;
    };

    const CoeffStats s_per = coeff_stats(enc1(periodic::expand(period, slots_)));
    const CoeffStats s_inc = coeff_stats(enc1(incumbent_block_constant(period, slots_)));

    auto report = [&](const char* name, const CoeffStats& s) {
        std::printf("[periodic] %-26s coeffs=%zu  zero=%zu  |c|<=4:%zu  <=2^20:%zu  <=2^32:%zu  "
                    "structural=%zu (off-stride %zu)\n",
                    name, s.n, s.zeros, s.dust4, s.dust20, s.mid, s.structural, s.off_stride);
    };
    std::printf("\n[periodic] --- coeff-domain sparsity (d=%d, subring stride=%d, encode format=%d) ---\n",
                D_, stride, s_per.format);
    report("periodic (PROPOSED)", s_per);
    report("block-constant (TODAY)", s_inc);

    const size_t bound = 2 * static_cast<size_t>(D_);
    const size_t per_payload = s_per.n - s_per.zeros;   // everything a sparse ship must carry exactly
    std::printf("[periodic] sparse-ship payload (nonzero coeffs): periodic=%zu (bound %zu), incumbent=%zu\n",
                per_payload, bound, s_inc.n - s_inc.zeros);
    if (per_payload <= bound)
        std::printf("[periodic] G0b: EXACT — a %zu-value coeff ship reproduces the plaintext bit-for-bit.\n",
                    per_payload);
    else
        std::printf("[periodic] G0b: DUSTY — %zu coeffs beyond the structural %zu; a sparse ship needs a "
                    "threshold (value change!) or a denser index map.\n",
                    per_payload - s_per.structural, s_per.structural);

    ASSERT_GT(s_per.n, 0u);
    EXPECT_LE(s_per.structural, bound)
        << "more than 2d structural coeffs — the subring premise fails in the coefficient domain";
    EXPECT_EQ(s_per.off_stride, 0u)
        << "structural coeffs off the X^(N/2d) stride — not a subring element; sparse ship unsound";
    EXPECT_GT(s_inc.n - s_inc.zeros, bound * 4)
        << "the incumbent's coeff form is nearly as sparse — nothing to gain from the layout";
}
