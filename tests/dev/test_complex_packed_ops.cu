// =====================================================================================
// Isolation suite for the REMAINING complex-lane opportunities across the decode pipeline
// (sibling to test_complex_packed_vcache.cu, which covers the V path). Warm-up for the real
// implementation: each test proves the identity + measures the speedup with the wrapper
// primitives the model actually uses. Run both modes (CKKS_COMPLEX=0 then 1).
//
// Unifying lever (established by the vcache suite): per-op cost is ~flat, so a win needs FEWER
// ops, which complex packing delivers when the second operand is STATIC (pre-encoded weight /
// mask) OR when a linear/bilinear op + cheap unpack replaces two of them. Blocked through any
// nonlinearity (softmax/GELU/LN: squaring mixes the lanes).
//
//   1. KCachePackQKt    pack two keys K_a + i*K_b; Q real & shared -> <Q,K_a> + i<Q,K_b> in ONE
//                       mult + ONE reduction (vs two). Halves Q.K^T + K storage. Unpack scores
//                       before (nonlinear) softmax. Also exercises the reduction free-rider.
//   2. QKVOutputPack    fuse two projections sharing an input: (W_q + i*W_k)*x = Q + i*K in one
//                       matrix-vector (shared input rotations). Halves the diagonal mults.
//   3. BootstrapPackedPayload  DIAGNOSTIC: bootstrap each pack construction TWICE to separate a
//                       one-time first-complex-bootstrap warmup from a real construction failure
//                       (run-to-run the same pack->bts flipped between ~1e148 and ~1e-3).
//   4. ReductionFreeRider  one reduction over complex-packed data costs ~the same as one real
//                       reduction yet returns BOTH sums -> the 2nd lane rides every linear reduce free.
//   5. GeneralizedComplexLinear  output-row-pack a SINGLE BSGS linear (W_even + i*W_odd): halves the
//                       OUTPUT side (diagonal mults + giantstep rotations) for one conjugate, but does
//                       NOT halve the babystep (input/contraction) rotations. Measures the partial gain.
#include "ckks_fixture.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

using namespace test_helpers;

namespace {

using ComplexOpsTest = CkksFixture;

bool complex_mode() { const char* cx = std::getenv("CKKS_COMPLEX"); return cx && cx[0] == '1'; }
double now_ms() { using namespace std::chrono; return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count(); }
double max_abs_err(const std::vector<double>& ref, const std::vector<double>& got, int n) {
    double m = 0.0; for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(got[i] - ref[i])); return m;
}
int env_int(const char* k, int dflt) { const char* v = std::getenv(k); return (v && *v) ? std::atoi(v) : dflt; }

Ptx i_plaintext(CKKSContext& fhe, int n) { return encode(fhe.cc, std::vector<std::complex<double>>(n, {0.0, 1.0})); }

// (a + i*b) via the rescaling "x i" plaintext multiply -- bootstrap-safe (the lm_head tile pack
// is bootstrapped). Cheap: one plaintext mult + one add, no key-switch.
Ctx pack_ri(CKKSContext& fhe, const Ctx& a, const Ctx& b, Ptx& i_pt) { return fhe.add(a, fhe.mult(b, i_pt)); }

// x *= i via the monomial X^(N/2) (no key-switch / rescale; used only on NON-bootstrapped cts).
Ctx times_i_mono(CKKSContext& fhe, const Ctx& x) {
    Ctx t = fhe.clone(x);
    fhe.cc->EvalMultMonomialInPlace(t, static_cast<uint32_t>(fhe.cc->GetRingDimension() / 2));
    return t;
}
// Re(P) = im_cleanse(P)*0.5 = (P + conj(P))*0.5. The conjugate is the model's load-bearing primitive.
Ctx real_part(CKKSContext& fhe, const Ctx& P) { return fhe.mult(fhe.add(P, fhe.conjugate(P)), 0.5); }
// Im(P) carried in real slots = (conj(P) - P)*i*0.5.
Ctx imag_part(CKKSContext& fhe, const Ctx& P) { return fhe.mult(times_i_mono(fhe, fhe.sub(fhe.conjugate(P), P)), 0.5); }
// Recover (Re, Im) sharing ONE conjugate (mirrors fideslib_wrapper.h pair_unpack).
std::pair<Ctx, Ctx> unpack_ri(CKKSContext& fhe, const Ctx& P) {
    Ctx cj = fhe.conjugate(P);
    Ctx re = fhe.mult(fhe.add(P, cj), 0.5);
    Ctx im = fhe.mult(times_i_mono(fhe, fhe.sub(cj, P)), 0.5);
    return {re, im};
}

// time a thunk: warmup (out of timed region) then R reps, return ms/rep.
template <class F> double time_ms(int R, F&& f) {
    auto w = f(); (void)w; cudaDeviceSynchronize();
    const double t = now_ms();
    for (int r = 0; r < R; ++r) { auto c = f(); (void)c; }
    cudaDeviceSynchronize();
    return (now_ms() - t) / R;
}

// =====================================================================================
// 1) K-cache pack + Q.K^T. One query Q (real, shared) against two packed keys K_a + i*K_b:
//    reduce(Q (.) (K_a + i*K_b)) = <Q,K_a> + i<Q,K_b>. ONE ct*ct mult + ONE reduction give both
//    scores; unpack before softmax. (Full slot reduction stands in for head_reduce_sum -- the
//    real one is a d_head window, fewer rotations, but identically linear so it halves the same.)
// =====================================================================================
TEST_F(ComplexOpsTest, KCachePackQKt) {
    const int n = slots();
    const int R = env_int("REPS", 20);
    std::mt19937 gen(21);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<double> q(n), ka(n), kb(n);
    double ra = 0, rb = 0;
    for (int i = 0; i < n; ++i) { q[i] = d(gen); ka[i] = d(gen); kb[i] = d(gen); ra += q[i] * ka[i]; rb += q[i] * kb[i]; }
    const std::vector<double> refa(n, ra), refb(n, rb);   // full reduction broadcasts the scalar to every slot
    Ptx i_pt = i_plaintext(fhe(), n);
    Ctx cq  = encrypt(fhe().cc, encode(fhe().cc, q),  fhe().pk());
    Ctx cka = encrypt(fhe().cc, encode(fhe().cc, ka), fhe().pk());
    Ctx ckb = encrypt(fhe().cc, encode(fhe().cc, kb), fhe().pk());
    Ctx kpack = pack_ri(fhe(), cka, ckb, i_pt);   // K_a + i*K_b : built at cache push, amortized

    auto reduce_all = [&](Ctx x) { for (int s = 1; s < n; s *= 2) x = fhe().add(x, fhe().rotate(x, s)); return x; };
    auto qkt_baseline = [&]() { return std::make_pair(reduce_all(fhe().mult(cq, cka)), reduce_all(fhe().mult(cq, ckb))); };
    auto qkt_fused    = [&]() { return unpack_ri(fhe(), reduce_all(fhe().mult(cq, kpack))); };   // incl. unpack

    auto bp = qkt_baseline();
    auto fp = qkt_fused();
    const double e_base = std::max(max_abs_err(refa, decrypt_slots(fhe(), bp.first),  n), max_abs_err(refb, decrypt_slots(fhe(), bp.second), n));
    const double e_fa   = max_abs_err(refa, decrypt_slots(fhe(), fp.first),  n);
    const double e_fb   = max_abs_err(refb, decrypt_slots(fhe(), fp.second), n);
    const double T_base = time_ms(R, qkt_baseline);
    const double T_fus  = time_ms(R, qkt_fused);

    std::cout << std::fixed << std::setprecision(2)
              << "[kqkt] CKKS_COMPLEX=" << (complex_mode() ? 1 : 0) << " reps=" << R
              << "  baseline=" << T_base << " ms (2 mults + 2 reductions)"
              << "  fused=" << T_fus << " ms (1 mult + 1 reduction + unpack)"
              << "  speedup=" << (T_base / T_fus) << "x"
              << std::scientific << std::setprecision(2) << "  errs base=" << e_base << " a=" << e_fa << " b=" << e_fb << std::endl;
    EXPECT_LT(e_base, 1e-3) << "baseline Q.K^T wrong";
    if (complex_mode()) {
        EXPECT_LT(e_fa, 1e-3) << "fused score_a (Re) wrong";
        EXPECT_LT(e_fb, 1e-3) << "fused score_b (Im) wrong";
        EXPECT_LT(T_fus, 0.75 * T_base) << "two-key Q.K^T should roughly halve mult+reduction";
    } else {
        std::cout << "[kqkt] real mode: imag lane absent -> only score_a survives (informational)" << std::endl;
    }
}

// =====================================================================================
// 2) QKV output-packing: two weight matrices sharing one input. (W_q + i*W_k)*x = Q + i*K in ONE
//    matrix-vector (input rotations shared) -> halves the diagonal mults; unpack to split Q/K.
//    Modeled as a naive diagonal sum over power-of-2 strides (keys the fixture seeds). Correctness
//    is FHE-to-FHE (fused == baseline), so it is rotation-direction agnostic. A real BSGS linear
//    rotates only ~sqrt(ndiag) times, so the mult-halving counts for MORE there than here.
// =====================================================================================
TEST_F(ComplexOpsTest, QKVOutputPack) {
    const int n = slots();
    const int ND = env_int("NDIAG", 16);   // diagonals (strides 2^0..2^(ND-1), all power-of-2 keys)
    const int R  = env_int("REPS", 20);
    std::mt19937 gen(22);
    std::uniform_real_distribution<double> d(-0.5, 0.5);
    std::vector<double> x(n); for (double& v : x) v = d(gen);
    std::vector<std::vector<double>> Wq(ND), Wk(ND);
    for (int j = 0; j < ND; ++j) { Wq[j].resize(n); Wk[j].resize(n); for (int i = 0; i < n; ++i) { Wq[j][i] = d(gen); Wk[j][i] = d(gen); } }
    Ctx cx = encrypt(fhe().cc, encode(fhe().cc, x), fhe().pk());
    std::vector<Ptx> pwq, pwk, pwc;
    for (int j = 0; j < ND; ++j) {
        pwq.push_back(encode(fhe().cc, Wq[j]));
        pwk.push_back(encode(fhe().cc, Wk[j]));
        std::vector<std::complex<double>> w(n); for (int i = 0; i < n; ++i) w[i] = {Wq[j][i], Wk[j][i]};   // W_q + i*W_k
        pwc.push_back(encode(fhe().cc, w));
    }
    auto rotset = [&]() { std::vector<Ctx> rs; rs.reserve(ND); for (int j = 0; j < ND; ++j) rs.push_back(j == 0 ? cx : fhe().rotate(cx, 1 << j)); return rs; };

    auto proj_baseline = [&]() {                              // ND rotations (shared) + 2*ND mults
        auto rs = rotset();
        Ctx Q = fhe().mult(rs[0], pwq[0]), K = fhe().mult(rs[0], pwk[0]);
        for (int j = 1; j < ND; ++j) { Q = fhe().add(Q, fhe().mult(rs[j], pwq[j])); K = fhe().add(K, fhe().mult(rs[j], pwk[j])); }
        return std::make_pair(Q, K);
    };
    auto proj_fused = [&]() {                                 // ND rotations (shared) + ND complex mults + unpack
        auto rs = rotset();
        Ctx P = fhe().mult(rs[0], pwc[0]);
        for (int j = 1; j < ND; ++j) P = fhe().add(P, fhe().mult(rs[j], pwc[j]));
        return unpack_ri(fhe(), P);
    };

    auto bp = proj_baseline();
    auto fp = proj_fused();
    const double e_q = max_abs_err(decrypt_slots(fhe(), bp.first),  decrypt_slots(fhe(), fp.first),  n);
    const double e_k = max_abs_err(decrypt_slots(fhe(), bp.second), decrypt_slots(fhe(), fp.second), n);
    const double T_base = time_ms(R, proj_baseline);
    const double T_fus  = time_ms(R, proj_fused);

    std::cout << std::fixed << std::setprecision(2)
              << "[qkv] CKKS_COMPLEX=" << (complex_mode() ? 1 : 0) << " ndiag=" << ND << " reps=" << R
              << "  baseline=" << T_base << " ms (" << ND << " rot + " << 2 * ND << " mults)"
              << "  fused=" << T_fus << " ms (" << ND << " rot + " << ND << " complex-mults + unpack)"
              << "  speedup=" << (T_base / T_fus) << "x"
              << std::scientific << std::setprecision(2) << "  Q_err=" << e_q << " K_err=" << e_k << std::endl;
    if (complex_mode()) {
        EXPECT_LT(e_q, 1e-4) << "fused Q (Re) != baseline";
        EXPECT_LT(e_k, 1e-4) << "fused K (Im) != baseline";
        EXPECT_LT(T_fus, 0.9 * T_base) << "fusing two projections should beat computing them separately";
    } else {
        std::cout << "[qkv] real mode: imag lane absent -> K (Im) dropped (informational)" << std::endl;
    }
}

// =====================================================================================
// 3) DIAGNOSTIC: bootstrapping a complex/packed payload. Across runs the SAME pack-then-bootstrap
//    gave ~1e148 once and ~1e-3 another time -> not construction-dependent but STATEFUL. This probe
//    bootstraps each construction TWICE to separate a one-time first-complex-bootstrap warmup from a
//    real construction/level failure, so the implementation knows the safe rule. (Bootstrap PAIRING
//    speed ~1.8x + correctness are already proven by goldschmidt DualBootstrap & vcache
//    FusedMacThroughBootstrap; here we just nail down WHEN a packed bootstrap is trustworthy.)
// =====================================================================================
TEST_F(ComplexOpsTest, BootstrapPackedPayload) {
    const int n = slots();
    std::mt19937 gen(25);
    std::uniform_real_distribution<double> d(-0.5, 0.5);
    std::vector<double> a(n), b(n); for (int i = 0; i < n; ++i) { a[i] = d(gen); b[i] = d(gen); }
    Ptx i_pt = i_plaintext(fhe(), n);
    Ptx one_pt = encode(fhe().cc, std::vector<double>(n, 1.0));
    std::vector<std::complex<double>> wc(n); for (int i = 0; i < n; ++i) wc[i] = {1.0, b[i]};   // weight 1 + i*b
    Ptx pwc = encode(fhe().cc, wc);
    auto enc = [&](const std::vector<double>& v) { return encrypt(fhe().cc, encode(fhe().cc, v), fhe().pk()); };

    // each build() returns a ct whose REAL lane should decrypt to `a`; bootstrap twice to separate a
    // one-time first-complex-bootstrap warmup (first bad, second good) from a construction failure
    // (both bad). The number printed is the Re-lane error after the bootstrap.
    auto probe = [&](const char* tag, auto&& build) {
        Ctx P1 = build(); fhe().bootstrap(P1);
        const double r1 = max_abs_err(a, decrypt_slots(fhe(), real_part(fhe(), P1)), n);
        Ctx P2 = build(); fhe().bootstrap(P2);
        const double r2 = max_abs_err(a, decrypt_slots(fhe(), real_part(fhe(), P2)), n);
        std::cout << "[btsdiag] CKKS_COMPLEX=" << (complex_mode() ? 1 : 0) << " " << tag
                  << "  Re-err first=" << std::scientific << std::setprecision(2) << r1 << "  second=" << r2 << std::endl;
        return std::make_pair(r1, r2);
    };

    probe("real_single  a (no pack)        ", [&] { return enc(a); });                                                   // real, should always work
    probe("fresh_pack   add(a, i*b)        ", [&] { return pack_ri(fhe(), enc(a), enc(b), i_pt); });                     // goldschmidt-style (fresh)
    probe("product_pack add(a*1, i*(b*1))  ", [&] { return pack_ri(fhe(), fhe().mult(enc(a), one_pt), fhe().mult(enc(b), one_pt), i_pt); }); // lm_head-style (products)
    probe("cplx_weight  a (.) (1 + i*b)    ", [&] { return fhe().mult(enc(a), pwc); });                                  // single complex-weight mult

    // Real lane survives every construction (above) -> the lm_head 1e148 was NOT the bootstrap. Now
    // recover the IMAG lane after a bootstrap two ways: (i) MONOMIAL times_i_mono (what unpack_ri /
    // pair_unpack use) vs (ii) PLAINTEXT x(-0.5 i) (goldschmidt unpack_compat). kqkt/qkv use the
    // monomial unpack with NO bootstrap and are exact -> isolate monomial-AFTER-bootstrap here.
    Ptx nhi_pt = encode(fhe().cc, std::vector<std::complex<double>>(n, {0.0, -0.5}));
    // full 2x2: (fresh operands vs product operands) x (imag via monomial vs imag via plaintext).
    // lm_head's 1e148 = product-pack + bootstrap + monomial unpack -- the one cell untested so far.
    auto post_bts = [&](const char* tag, Ctx P) {
        fhe().bootstrap(P);
        Ctx cj = fhe().conjugate(P);
        const double re   = max_abs_err(a, decrypt_slots(fhe(), fhe().mult(fhe().add(P, cj), 0.5)), n);
        const double im_m = max_abs_err(b, decrypt_slots(fhe(), fhe().mult(times_i_mono(fhe(), fhe().sub(cj, P)), 0.5)), n);
        const double im_p = max_abs_err(b, decrypt_slots(fhe(), fhe().mult(fhe().sub(P, cj), nhi_pt)), n);
        std::cout << "[btsdiag] CKKS_COMPLEX=" << (complex_mode() ? 1 : 0) << " post-bts " << tag
                  << "  Re=" << std::scientific << std::setprecision(2) << re
                  << "  Im(monomial)=" << im_m << "  Im(plaintext -0.5i)=" << im_p << std::endl;
        return std::make_pair(re, im_p);
    };
    auto pf = post_bts("fresh_pack  ", pack_ri(fhe(), enc(a), enc(b), i_pt));
    auto pp = post_bts("product_pack", pack_ri(fhe(), fhe().mult(enc(a), one_pt), fhe().mult(enc(b), one_pt), i_pt));
    if (complex_mode()) {
        EXPECT_LT(pf.first, 5e-2);  EXPECT_LT(pf.second, 5e-2) << "fresh-pack plaintext imag extraction failed";
        EXPECT_LT(pp.first, 5e-2);  EXPECT_LT(pp.second, 5e-2) << "product-pack plaintext imag extraction failed";
        // monomial imag (im_m, printed) is the suspect; not asserted.
    }
}

// =====================================================================================
// 4) Reduction free-rider: ONE reduction over complex-packed data carries TWO sums at the cost of
//    ONE real reduction (reduce is linear: reduce(a+ib) = reduce(a) + i*reduce(b)). Proves any
//    upstream complex packing rides every head_reduce_sum / tok_reduce for free (ratio ~= 1.0).
// =====================================================================================
TEST_F(ComplexOpsTest, ReductionFreeRider) {
    const int n = slots();
    const int R = env_int("REPS", 20);
    std::mt19937 gen(24);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<double> a(n), b(n); double sa = 0, sb = 0;
    for (int i = 0; i < n; ++i) { a[i] = d(gen); b[i] = d(gen); sa += a[i]; sb += b[i]; }
    Ptx i_pt = i_plaintext(fhe(), n);
    Ctx ca = encrypt(fhe().cc, encode(fhe().cc, a), fhe().pk());
    Ctx cb = encrypt(fhe().cc, encode(fhe().cc, b), fhe().pk());
    Ctx cab = pack_ri(fhe(), ca, cb, i_pt);
    auto reduce_all = [&](Ctx x) { for (int s = 1; s < n; s *= 2) x = fhe().add(x, fhe().rotate(x, s)); return x; };

    Ctx rr = reduce_all(cab);
    const double e_a = max_abs_err(std::vector<double>(n, sa), decrypt_slots(fhe(), real_part(fhe(), rr)), n);
    const double e_b = max_abs_err(std::vector<double>(n, sb), decrypt_slots(fhe(), imag_part(fhe(), rr)), n);
    const double T_real = time_ms(R, [&]() { return reduce_all(ca);  });   // reduces ONE signal
    const double T_cplx = time_ms(R, [&]() { return reduce_all(cab); });   // reduces TWO, same op count

    std::cout << std::fixed << std::setprecision(2)
              << "[redfree] CKKS_COMPLEX=" << (complex_mode() ? 1 : 0) << " reps=" << R
              << "  real-only=" << T_real << " ms  complex(two-in-one)=" << T_cplx << " ms"
              << "  ratio=" << (T_cplx / T_real) << " (~1 => 2nd lane FREE)"
              << std::scientific << std::setprecision(2) << "  errs a=" << e_a << " b=" << e_b << std::endl;
    if (complex_mode()) {
        EXPECT_LT(e_a, 1e-3) << "real-lane sum wrong";
        EXPECT_LT(e_b, 1e-3) << "imag-lane sum wrong";
        EXPECT_LT(T_cplx, 1.3 * T_real) << "complex reduction should cost ~same as real (2nd lane free)";
    } else {
        std::cout << "[redfree] real mode: imag lane absent -> only sum a survives (informational)" << std::endl;
    }
}

// =====================================================================================
// 5) Generalized complex linear — output-row packing on ONE linear. Models a BSGS matrix-vector:
//    BABY babystep rotations of x (input/contraction side, SHARED) + GIANT giantstep rotations
//    (output side) + BABY*GIANT diagonal mults. Pairing output rows via complex weights
//    (W_even + i*W_odd) halves the OUTPUT side (mults + giantstep rotations) and costs ONE conjugate
//    to unpack, but the babystep (input) rotations are unchanged. So the gain is PARTIAL (not 2x) and
//    set by the baby/giant/mult balance. Correctness is checked on the pre-placement accumulator
//    (Re == baseline output row 2g, Im == row 2g+1); timing runs the full workload incl. the unpack.
// =====================================================================================
TEST_F(ComplexOpsTest, GeneralizedComplexLinear) {
    const int n = slots();
    const int BABY  = env_int("BABY", 8);     // babystep (input/contraction) rotations -- SHARED, not halved
    const int GIANT = env_int("GIANT", 16);   // giantstep (output) blocks (even) -- HALVED by output-row packing
    const int R = env_int("REPS", 20);
    std::mt19937 gen(26);
    std::uniform_real_distribution<double> d(-0.5, 0.5);
    std::vector<double> x(n); for (double& v : x) v = d(gen);
    Ctx cx = encrypt(fhe().cc, encode(fhe().cc, x), fhe().pk());

    auto rnd = [&]() { std::vector<double> v(n); for (double& e : v) e = d(gen); return v; };
    std::vector<std::vector<double>> W(static_cast<size_t>(GIANT) * BABY);
    for (auto& w : W) w = rnd();
    std::vector<Ptx> pw; pw.reserve(W.size());                         // real diagonals W[g][j]
    for (auto& w : W) pw.push_back(encode(fhe().cc, w));
    std::vector<Ptx> pwc; pwc.reserve(static_cast<size_t>(GIANT / 2) * BABY);   // complex pairs W[2gp][j] + i*W[2gp+1][j]
    for (int gp = 0; gp < GIANT / 2; ++gp)
        for (int j = 0; j < BABY; ++j) {
            const auto& wa = W[(2 * gp) * BABY + j];
            const auto& wb = W[(2 * gp + 1) * BABY + j];
            std::vector<std::complex<double>> wc(n);
            for (int i = 0; i < n; ++i) wc[i] = {wa[i], wb[i]};
            pwc.push_back(encode(fhe().cc, wc));
        }

    // BABY rotations of x (SHARED by both schemes). Steps cycle keyed powers of 2 in [2,2^14] so
    // BABY can model an input-heavy linear (MLP: big contraction) without exceeding the fixture keys.
    auto babysteps = [&]() {
        std::vector<Ctx> rx; rx.reserve(BABY);
        for (int j = 0; j < BABY; ++j) rx.push_back(j == 0 ? cx : fhe().rotate(cx, 1 << (1 + (j % 14))));
        return rx;
    };
    // baseline: GIANT output blocks (BABY real mults each) + GIANT-1 giantstep placement rotations.
    auto lin_baseline = [&]() {
        auto rx = babysteps();
        Ctx result = fhe().mult(rx[0], pw[0]);
        for (int j = 1; j < BABY; ++j) result = fhe().add(result, fhe().mult(rx[j], pw[j]));
        for (int g = 1; g < GIANT; ++g) {
            Ctx acc = fhe().mult(rx[0], pw[g * BABY + 0]);
            for (int j = 1; j < BABY; ++j) acc = fhe().add(acc, fhe().mult(rx[j], pw[g * BABY + j]));
            result = fhe().add(result, fhe().rotate(acc, 1 << (1 + (g % 14))));   // giantstep (output) rotation
        }
        return result;
    };
    // packed: GIANT/2 complex blocks (BABY complex mults each) + GIANT/2-1 giantstep rotations + 1 conjugate.
    auto lin_packed = [&]() {
        auto rx = babysteps();                                        // SAME BABY rotations (NOT halved)
        Ctx result = fhe().mult(rx[0], pwc[0]);
        for (int j = 1; j < BABY; ++j) result = fhe().add(result, fhe().mult(rx[j], pwc[j]));
        for (int gp = 1; gp < GIANT / 2; ++gp) {
            Ctx acc = fhe().mult(rx[0], pwc[gp * BABY + 0]);
            for (int j = 1; j < BABY; ++j) acc = fhe().add(acc, fhe().mult(rx[j], pwc[gp * BABY + j]));
            result = fhe().add(result, fhe().rotate(acc, 1 << (1 + (gp % 14))));
        }
        return real_part(fhe(), result);                              // one conjugate to unpack
    };

    // correctness on the pre-placement accumulator: complex block 0 carries baseline rows 0 (Re) and 1 (Im).
    auto rx = babysteps();
    auto acc_real = [&](int g) {
        Ctx a = fhe().mult(rx[0], pw[g * BABY + 0]);
        for (int j = 1; j < BABY; ++j) a = fhe().add(a, fhe().mult(rx[j], pw[g * BABY + j]));
        return a;
    };
    Ctx acc0 = acc_real(0), acc1 = acc_real(1);
    Ctx accc = fhe().mult(rx[0], pwc[0]);
    for (int j = 1; j < BABY; ++j) accc = fhe().add(accc, fhe().mult(rx[j], pwc[j]));
    const double e_re = max_abs_err(decrypt_slots(fhe(), acc0), decrypt_slots(fhe(), real_part(fhe(), accc)), n);
    const double e_im = max_abs_err(decrypt_slots(fhe(), acc1), decrypt_slots(fhe(), imag_part(fhe(), accc)), n);

    const double T_base = time_ms(R, lin_baseline);
    const double T_pack = time_ms(R, lin_packed);

    std::cout << std::fixed << std::setprecision(2)
              << "[genlin] CKKS_COMPLEX=" << (complex_mode() ? 1 : 0) << " baby=" << BABY << " giant=" << GIANT << " reps=" << R << "\n"
              << "  baseline = " << T_base << " ms (" << BABY << " baby-rot + " << (GIANT - 1) << " giant-rot + " << (GIANT * BABY) << " mults)\n"
              << "  packed   = " << T_pack << " ms (" << BABY << " baby-rot + " << (GIANT / 2 - 1) << " giant-rot + " << (GIANT / 2 * BABY) << " cplx-mults + 1 conj)\n"
              << "  speedup  = " << (T_base / T_pack) << "x  (output side halved; babystep rotations NOT halved)"
              << std::scientific << std::setprecision(2) << "  re_err=" << e_re << " im_err=" << e_im << std::endl;
    if (complex_mode()) {
        EXPECT_LT(e_re, 1e-4) << "packed block Re != baseline output row 2g";
        EXPECT_LT(e_im, 1e-4) << "packed block Im != baseline output row 2g+1";
        EXPECT_LT(T_pack, T_base) << "output-row packing should beat the unpacked linear";
    } else {
        std::cout << "[genlin] real mode: imag lane absent -> odd output rows dropped (informational)" << std::endl;
    }
}

}  // namespace
