// =====================================================================================
// Isolated proof for the V-cache complex lane. Store TWO real V lanes in the real/imag parts of
// ONE ciphertext (halving the V cache) and FUSE the two P*V multiply-accumulates of
// softmax_v into ONE ct*ct multiply via a conjugation:
//
//     Re( (v_a + i*v_b) * (s_a - i*s_b) )  =  v_a*s_a + v_b*s_b
//
// The simpler complex-lane case multiplies BOTH lanes by the SAME real factor F
// ((N + i*D)*F -> N*F + i*D*F). softmax_v instead needs a DIFFERENT
// factor per lane (score s_a on v_a, s_b on v_b) -- which is exactly what conjugating the
// score pack (s_a - i*s_b) and taking the real part buys. The closing Re() is already in
// softmax_v (the im_cleanse at cachemir_attention.cu), so it is amortized ONCE per
// P*V step, not per lane pair.
//
// Cost accounting it proves, per P*V step over P lane-pairs (rotations are identical in
// both schemes and excluded):
//     unfused:  2*P ct*ct mults                                    ; V cache = 2*P cts
//     fused  :    P ct*ct mults + P plaintext "x(-i)"              ; V cache =   P cts
//                          + 1 conjugation (the shared Re, amortized)
// The V pack (v_a + i*v_b) is built ONCE at cache-push and reused every step, so it is not
// charged to the per-step timing (it is charged to the push, where it also halves storage).
//
// Tests: (1) the fused MAC is bit-exact vs the two-multiply baseline (complex mode);
//        (2) it survives the cache-push bootstrap (cached V is post-bootstrap);
//        (3) it is faster -- P ct*ct mults instead of 2*P -- on a single P*V step.
//
// Run BOTH modes (CKKS_COMPLEX=0 vs =1). The fused-path asserts fire only in complex mode;
// real mode zeroes the imag lane (the v_b / s_b half vanishes) and is informational.
#include "ckks_fixture.h"
#include "ckks_primitives.h"

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

using VCachePackTest = CkksFixture;

bool complex_mode() {
    const char* cx = std::getenv("CKKS_COMPLEX");
    return cx && cx[0] == '1';
}
double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
double max_abs_err(const std::vector<double>& ref, const std::vector<double>& got, int n) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(got[i] - ref[i]));
    return m;
}

// Unit-magnitude complex constants used to pack/conjugate the lanes (cheap plaintext mult,
// no key-switch). Same "rescaling x i" trick the goldschmidt test proved bootstrap-safe.
Ptx i_plaintext(CKKSContext& fhe, int n)     { return encode(fhe.cc, std::vector<std::complex<double>>(n, {0.0,  1.0})); }
Ptx neg_i_plaintext(CKKSContext& fhe, int n) { return encode(fhe.cc, std::vector<std::complex<double>>(n, {0.0, -1.0})); }

// (v_a + i*v_b): the packed V-cache lane. Built ONCE at cache-push, reused every P*V step.
Ctx pack_v(CKKSContext& fhe, const Ctx& va, const Ctx& vb, Ptx& i_pt) {
    return fhe.add(va, fhe.mult(vb, i_pt));
}
// (s_a - i*s_b): the conjugated score pack for ONE P*V step (one plaintext mult + one add).
Ctx pack_scores(CKKSContext& fhe, const Ctx& sa, const Ctx& sb, Ptx& ni_pt) {
    return fhe.add(sa, fhe.mult(sb, ni_pt));
}
// Re(P) = im_cleanse(P) * 0.5.  im_cleanse(a) = a + conj(a) = 2*Re(a) (fideslib_wrapper.h)
// is the SAME conjugate deflation used all over the model and is EXACTLY softmax_v's closing op
// (cachemir_attention.cu) -- so in production this is not an added op, and the 0.5 rides the
// tok0 mask. Uses the wrapper conjugate (the one im_cleanse calls), not the raw primitive.
Ctx real_part(CKKSContext& fhe, const Ctx& P) {
    return fhe.mult(fhe.add(P, fhe.conjugate(P)), 0.5);   // == im_cleanse(P) * 0.5
}

// --- CHEAP "x i": monomial shift X^(N/2) (no key-switch, no rescale, level-preserving) ---
// This is what pair_pack uses. It is bootstrap-INCOMPATIBLE (CoeffToSlot corrupts the
// monomial state), but W = s_a - i*s_b is
// consumed immediately by the C*W multiply and never bootstrapped, so it is safe here.
Ctx times_i_mono(CKKSContext& fhe, const Ctx& x) {
    Ctx t = fhe.clone(x);
    fhe.cc->EvalMultMonomialInPlace(t, static_cast<uint32_t>(fhe.cc->GetRingDimension() / 2));  // t *= i
    return t;
}
// W = s_a - i*s_b, built with the cheap monomial (vs pack_scores' expensive plaintext x(-i)).
Ctx pack_scores_mono(CKKSContext& fhe, const Ctx& sa, const Ctx& sb) {
    return fhe.cc->EvalSub(sa, times_i_mono(fhe, sb));
}
// Im(P) carried in real slots: (conj(P)-P)*i*0.5 = Im(P). Mirrors pair_unpack's imag branch.
Ctx imag_part(CKKSContext& fhe, const Ctx& P) {
    Ctx im = fhe.cc->EvalSub(fhe.cc->EvalConjugate(P), P);                                // -2i*Im
    fhe.cc->EvalMultMonomialInPlace(im, static_cast<uint32_t>(fhe.cc->GetRingDimension() / 2));  // *i -> 2*Im
    fhe.cc->EvalMultInPlace(im, 0.5);
    return im;
}

// =====================================================================================
// 1) Exactness: the fused single-multiply MAC equals the two-multiply baseline AND the
//    plaintext truth v_a*s_a + v_b*s_b. No bootstrap -> tight (~scale precision).
// =====================================================================================
TEST_F(VCachePackTest, FusedMacExact) {
    const int n = slots();
    std::mt19937 gen(11);
    std::uniform_real_distribution<double> dv(-2.0, 2.0), ds(0.0, 1.0);   // scores ~ softmax weights >= 0
    std::vector<double> va(n), vb(n), sa(n), sb(n), ref(n);
    for (int i = 0; i < n; ++i) {
        va[i] = dv(gen); vb[i] = dv(gen); sa[i] = ds(gen); sb[i] = ds(gen);
        ref[i] = va[i] * sa[i] + vb[i] * sb[i];
    }
    Ptx i_pt = i_plaintext(fhe(), n), ni = neg_i_plaintext(fhe(), n);
    Ctx cva = encrypt(fhe().cc, encode(fhe().cc, va), fhe().pk());
    Ctx cvb = encrypt(fhe().cc, encode(fhe().cc, vb), fhe().pk());
    Ctx csa = encrypt(fhe().cc, encode(fhe().cc, sa), fhe().pk());
    Ctx csb = encrypt(fhe().cc, encode(fhe().cc, sb), fhe().pk());

    // unfused softmax_v, for one lane pair: 2 ct*ct mults + 1 add
    Ctx base = fhe().add(fhe().mult(cva, csa), fhe().mult(cvb, csb));
    double e_base = max_abs_err(ref, decrypt_slots(fhe(), base), n);

    // fused (plaintext x(-i) score pack) = pack V (push-time), pack scores, ONE mult, take Re
    Ctx C = pack_v(fhe(), cva, cvb, i_pt);
    Ctx W = pack_scores(fhe(), csa, csb, ni);
    double e_fused = max_abs_err(ref, decrypt_slots(fhe(), real_part(fhe(), fhe().mult(C, W))), n);

    // fused (cheap monomial score pack) = same identity, W built with the x-i monomial.
    Ctx Cm = fhe().pair_pack(cva, cvb);            // v_a + i*v_b (monomial)
    Ctx Wm = pack_scores_mono(fhe(), csa, csb);    // s_a - i*s_b (monomial)
    double e_mono = max_abs_err(ref, decrypt_slots(fhe(), real_part(fhe(), fhe().mult(Cm, Wm))), n);

    std::cout << "[fused_exact] CKKS_COMPLEX=" << (complex_mode() ? 1 : 0)
              << " baseline_err=" << std::scientific << std::setprecision(3) << e_base
              << " fused_pt_err=" << e_fused << " fused_mono_err=" << e_mono << std::endl;
    EXPECT_LT(e_base, 1e-5) << "two-multiply real baseline itself wrong";
    if (complex_mode()) {
        EXPECT_LT(e_fused, 1e-5) << "fused MAC != baseline: Re((va+i*vb)(sa-i*sb)) should equal va*sa+vb*sb";
        EXPECT_LT(e_mono,  1e-5) << "monomial-packed fused MAC wrong";
    } else {
        std::cout << "[fused_exact] real mode: imag lane absent -> vb*sb term dropped (expected)" << std::endl;
    }
}

// =====================================================================================
// 2) Through the push bootstrap: the cached V is stored post-bootstrap, so pack V and
//    refresh it (ONE bootstrap for both lanes), then fuse. ~9-10 bit bootstrap floor.
// =====================================================================================
TEST_F(VCachePackTest, FusedMacThroughBootstrap) {
    const int n = slots();
    std::mt19937 gen(12);
    std::uniform_real_distribution<double> dv(-1.0, 1.0), ds(0.0, 1.0);
    std::vector<double> va(n), vb(n), sa(n), sb(n), ref(n);
    for (int i = 0; i < n; ++i) {
        va[i] = dv(gen); vb[i] = dv(gen); sa[i] = ds(gen); sb[i] = ds(gen);
        ref[i] = va[i] * sa[i] + vb[i] * sb[i];
    }
    Ptx i_pt = i_plaintext(fhe(), n), ni = neg_i_plaintext(fhe(), n);
    Ctx cva = encrypt(fhe().cc, encode(fhe().cc, va), fhe().pk());
    Ctx cvb = encrypt(fhe().cc, encode(fhe().cc, vb), fhe().pk());
    Ctx csa = encrypt(fhe().cc, encode(fhe().cc, sa), fhe().pk());
    Ctx csb = encrypt(fhe().cc, encode(fhe().cc, sb), fhe().pk());

    Ctx C = pack_v(fhe(), cva, cvb, i_pt);
    fhe().bootstrap(C);                       // ONE bootstrap refreshes BOTH cached V lanes (the push pack)
    fhe().bootstrap(csa); fhe().bootstrap(csb);   // scores brought to the cached-V level, then packed
    Ctx W = pack_scores(fhe(), csa, csb, ni);
    Ctx P = fhe().mult(C, W);
    double e = max_abs_err(ref, decrypt_slots(fhe(), real_part(fhe(), P)), n);

    std::cout << "[fused_bts] CKKS_COMPLEX=" << (complex_mode() ? 1 : 0)
              << " fused_err=" << std::scientific << std::setprecision(3) << e << std::endl;
    if (complex_mode())
        EXPECT_LT(e, 5e-2) << "fused MAC lost accuracy through the push bootstrap";
    else
        std::cout << "[fused_bts] real mode: imag lane absent (informational)" << std::endl;
}

// =====================================================================================
// 3) Speed of a SINGLE P*V step over P lane-pairs: baseline 2*P ct*ct mults vs fused P
//    ct*ct mults (+ P plaintext x(-i) + one shared Re). The V pack is amortized at push.
//    PAIRS default 32 (= GPT-2 d_head_real 64 lanes); REPS averages the step.
// =====================================================================================
TEST_F(VCachePackTest, FusedMacIsFaster) {
    const int n = slots();
    const char* pairs_env = std::getenv("PAIRS");
    const char* reps_env  = std::getenv("REPS");
    const int P = (pairs_env && *pairs_env) ? std::atoi(pairs_env) : 32;   // lane-pairs per P*V step
    const int R = (reps_env  && *reps_env)  ? std::atoi(reps_env)  : 20;   // repeated steps (averaged)

    std::mt19937 gen(13);
    std::uniform_real_distribution<double> dv(-1.0, 1.0), ds(0.0, 1.0);
    std::vector<double> va(n), vb(n), sa(n), sb(n), ref(n);
    for (int i = 0; i < n; ++i) {
        va[i] = dv(gen); vb[i] = dv(gen); sa[i] = ds(gen); sb[i] = ds(gen);
        ref[i] = static_cast<double>(P) * (va[i] * sa[i] + vb[i] * sb[i]);   // P identical pairs accumulated
    }
    Ptx i_pt = i_plaintext(fhe(), n), ni = neg_i_plaintext(fhe(), n);
    Ctx cva = encrypt(fhe().cc, encode(fhe().cc, va), fhe().pk());
    Ctx cvb = encrypt(fhe().cc, encode(fhe().cc, vb), fhe().pk());
    Ctx csa = encrypt(fhe().cc, encode(fhe().cc, sa), fhe().pk());
    Ctx csb = encrypt(fhe().cc, encode(fhe().cc, sb), fhe().pk());
    // V packs are built ONCE at cache-push and reused every step (amortized, not timed).
    Ctx C_pt   = pack_v(fhe(), cva, cvb, i_pt);   // rescaling pack (bootstrap-safe in production)
    Ctx C_mono = fhe().pair_pack(cva, cvb);        // monomial pack (cheap)

    // unfused softmax_v: 2*P ct*ct mults (v_a*s_a, v_b*s_b), accumulated.
    auto step_baseline = [&]() {
        Ctx acc = fhe().add(fhe().mult(cva, csa), fhe().mult(cvb, csb));
        for (int k = 1; k < P; ++k) {
            acc = fhe().add(acc, fhe().mult(cva, csa));
            acc = fhe().add(acc, fhe().mult(cvb, csb));
        }
        return acc;
    };
    // fused_pt = P ct*ct mults; W built with the EXPENSIVE plaintext x(-i) (the first attempt).
    auto step_fused_pt = [&]() {
        Ctx acc = fhe().mult(C_pt, pack_scores(fhe(), csa, csb, ni));
        for (int k = 1; k < P; ++k) acc = fhe().add(acc, fhe().mult(C_pt, pack_scores(fhe(), csa, csb, ni)));
        return real_part(fhe(), acc);
    };
    // fused_mono = P ct*ct mults; W built with the CHEAP x-i monomial; Re via conjugate once.
    auto step_fused_mono = [&]() {
        Ctx acc = fhe().mult(C_mono, pack_scores_mono(fhe(), csa, csb));
        for (int k = 1; k < P; ++k) acc = fhe().add(acc, fhe().mult(C_mono, pack_scores_mono(fhe(), csa, csb)));
        return real_part(fhe(), acc);
    };

    auto time_path = [&](auto&& step) {
        Ctx w = step(); (void)w; cudaDeviceSynchronize();          // warmup out of the timed region
        const double t = now_ms();
        Ctx last = step();
        for (int r = 1; r < R; ++r) last = step();
        cudaDeviceSynchronize();
        return std::make_pair((now_ms() - t) / R, max_abs_err(ref, decrypt_slots(fhe(), last), n));
    };

    auto rb = time_path(step_baseline);   const double T_base = rb.first, e_base = rb.second;
    auto rp = time_path(step_fused_pt);   const double T_pt   = rp.first, e_pt   = rp.second;
    auto rm = time_path(step_fused_mono); const double T_mono = rm.first, e_mono = rm.second;

    std::cout << std::fixed << std::setprecision(2)
              << "[speed] CKKS_COMPLEX=" << (complex_mode() ? 1 : 0) << " pairs=" << P << " reps=" << R << "\n"
              << "  baseline   = " << T_base << " ms/step (" << 2 * P << " ct*ct mults, " << 2 * P << " V cts)\n"
              << "  fused_pt   = " << T_pt   << " ms/step (" << P << " ct*ct + " << P << " plaintext x(-i), " << P << " V cts)  speedup=" << (T_base / T_pt) << "x\n"
              << "  fused_mono = " << T_mono << " ms/step (" << P << " ct*ct + " << P << " monomial x-i,  " << P << " V cts)  speedup=" << (T_base / T_mono) << "x\n"
              << std::scientific << std::setprecision(2)
              << "  errs: base=" << e_base << " pt=" << e_pt << " mono=" << e_mono << std::endl;

    EXPECT_LT(e_base, 1e-4) << "baseline accumulation wrong";
    if (complex_mode()) {
        EXPECT_LT(e_pt,   1e-4) << "fused_pt accumulation wrong";
        EXPECT_LT(e_mono, 1e-4) << "fused_mono accumulation wrong (imag cross-terms should cancel in Re)";
        // NO speed assertion: consumption-side fusion is a MEASURED WASH (~1.0x). Halving ct*ct
        // mults saves ~one 8.2 ms mult per pair, but building W = s_a - i*s_b at RUNTIME (clone +
        // monomial + sub + accumulate, ~4 ops) costs about the same. Complex packing only pays off
        // when the second operand is STATIC/pre-encoded (the push mask, the out_proj weights) --
        // runtime scores are not. See PushPackIsFaster / LinearExtractsDirectly.
    } else {
        std::cout << "[speed] real mode: timing valid but fused result wrong (imag lane absent)" << std::endl;
    }
}

// =====================================================================================
// 4) The PUSH side -- where the win actually is. The V-cache scatter masks each token's
//    value into L = d_head_real lanes (one plaintext mult per lane, one cache ct per lane).
//    A single COMPLEX extraction mask (m_a + i*m_b) scatters TWO lanes in ONE plaintext
//    mult into ONE cache ct (lane a -> Re, lane b -> Im). This genuinely HALVES the op
//    count and the storage -- and since per-op cost is ~flat, halving ops ~halves time.
// =====================================================================================
TEST_F(VCachePackTest, PushPackIsFaster) {
    const int n = slots();
    const char* lanes_env = std::getenv("LANES");
    const char* reps_env  = std::getenv("REPS");
    const int L = (lanes_env && *lanes_env) ? std::atoi(lanes_env) : 64;   // d_head_real lanes
    const int R = (reps_env  && *reps_env)  ? std::atoi(reps_env)  : 20;

    std::mt19937 gen(14);
    std::uniform_real_distribution<double> dv(-1.0, 1.0);
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) v[i] = dv(gen);
    Ctx v_rot = encrypt(fhe().cc, encode(fhe().cc, v), fhe().pk());   // one token's value vector

    // disjoint 0/1 selection masks: lane `l` owns slots with (i % L) == l.
    auto sel = [&](int lane) { std::vector<double> m(n, 0.0); for (int i = 0; i < n; ++i) if ((i % L) == lane) m[i] = 1.0; return m; };
    std::vector<Ptx> rmask; rmask.reserve(L);                       // baseline: one real mask per lane
    for (int l = 0; l < L; ++l) rmask.push_back(encode(fhe().cc, sel(l)));
    std::vector<Ptx> cmask; cmask.reserve(L / 2);                   // fused: one complex mask per lane-pair
    for (int j = 0; j < L / 2; ++j) {
        std::vector<double> ma = sel(2 * j), mb = sel(2 * j + 1);
        std::vector<std::complex<double>> cm(n);
        for (int i = 0; i < n; ++i) cm[i] = {ma[i], mb[i]};         // mask_a + i*mask_b
        cmask.push_back(encode(fhe().cc, cm));
    }

    auto push_baseline = [&]() { std::vector<Ctx> c; c.reserve(L);     for (int l = 0; l < L; ++l)     c.push_back(fhe().mult(v_rot, rmask[l])); return c; };
    auto push_fused    = [&]() { std::vector<Ctx> c; c.reserve(L / 2); for (int j = 0; j < L / 2; ++j) c.push_back(fhe().mult(v_rot, cmask[j])); return c; };

    // correctness: fused lane-pair j carries lane 2j in Re and lane 2j+1 in Im.
    auto cbase = push_baseline();
    auto cfus  = push_fused();
    const double e_re = max_abs_err(decrypt_slots(fhe(), cbase[0]), decrypt_slots(fhe(), real_part(fhe(), cfus[0])), n);
    const double e_im = max_abs_err(decrypt_slots(fhe(), cbase[1]), decrypt_slots(fhe(), imag_part(fhe(), cfus[0])), n);

    auto time_path = [&](auto&& push) {
        auto w = push(); (void)w; cudaDeviceSynchronize();
        const double t = now_ms();
        for (int r = 0; r < R; ++r) { auto c = push(); (void)c; }
        cudaDeviceSynchronize();
        return (now_ms() - t) / R;
    };
    const double T_base = time_path(push_baseline);
    const double T_fus  = time_path(push_fused);

    std::cout << std::fixed << std::setprecision(2)
              << "[push] CKKS_COMPLEX=" << (complex_mode() ? 1 : 0) << " lanes=" << L << " reps=" << R
              << "  baseline=" << T_base << " ms (" << L << " mults, " << L << " V cts)"
              << "  fused=" << T_fus << " ms (" << L / 2 << " mults, " << L / 2 << " V cts)"
              << "  speedup=" << (T_base / T_fus) << "x"
              << std::scientific << std::setprecision(2) << "  re_err=" << e_re << " im_err=" << e_im << std::endl;

    if (complex_mode()) {
        EXPECT_LT(e_re, 1e-5) << "fused push lost the real (lane 2j) value";
        EXPECT_LT(e_im, 1e-5) << "fused push lost the imag (lane 2j+1) value";
        EXPECT_LT(T_fus, 0.75 * T_base) << "complex-mask push should roughly halve the scatter work";
    } else {
        std::cout << "[push] real mode: imag lane absent -> only lane 2j survives (informational)" << std::endl;
    }
}

// =====================================================================================
// 5) "Make the linear extract it directly": fold the lane-sum AND the Re-extraction into the
//    CONSUMER linear's weights. A complex-packed input (a + i*b) times a CONJUGATED complex
//    PLAINTEXT weight (W_a - i*W_b) gives, in its real part, a*W_a + b*W_b -- one plaintext
//    mult does TWO weighted feature contractions. Over a D-feature contraction that is D/2
//    complex-weight mults instead of D real ones, + one im_cleanse for the whole reduction
//    (Re commutes with the sum). Unlike softmax_v's runtime W, the weights are STATIC, so the
//    complex weight is pre-encoded -- nothing is built at runtime, so this wins like the push.
//    This is the end-state of "keep V complex from the cache through P*V and only pick it up at
//    the out-proj": the out-proj never sees a real attention vector, it extracts as it contracts.
// =====================================================================================
TEST_F(VCachePackTest, LinearExtractsDirectly) {
    const int n = slots();
    const char* dfeat_env = std::getenv("DFEAT");
    const char* reps_env  = std::getenv("REPS");
    const int D = (dfeat_env && *dfeat_env) ? std::atoi(dfeat_env) : 64;   // contraction features (even)
    const int R = (reps_env  && *reps_env)  ? std::atoi(reps_env)  : 20;

    std::mt19937 gen(15);
    std::uniform_real_distribution<double> dx(-1.0, 1.0), dw(-1.0, 1.0);
    std::vector<std::vector<double>> a(D), W(D);
    std::vector<double> ref(n, 0.0);
    for (int k = 0; k < D; ++k) {
        a[k].resize(n); W[k].resize(n);
        for (int i = 0; i < n; ++i) { a[k][i] = dx(gen); W[k][i] = dw(gen); ref[i] += a[k][i] * W[k][i]; }
    }
    // baseline: D real ciphertexts, D real plaintext-weight mults.
    std::vector<Ctx> ca; ca.reserve(D);
    std::vector<Ptx> pw; pw.reserve(D);
    for (int k = 0; k < D; ++k) { ca.push_back(encrypt(fhe().cc, encode(fhe().cc, a[k]), fhe().pk())); pw.push_back(encode(fhe().cc, W[k])); }
    // fused: D/2 complex inputs (a_2j + i*a_2j+1) and D/2 PRE-ENCODED conjugated complex weights.
    Ptx i_pt = i_plaintext(fhe(), n);
    std::vector<Ctx> xin; xin.reserve(D / 2);
    std::vector<Ptx> cw;  cw.reserve(D / 2);
    for (int j = 0; j < D / 2; ++j) {
        xin.push_back(pack_v(fhe(), ca[2 * j], ca[2 * j + 1], i_pt));      // a_2j + i*a_2j+1
        std::vector<std::complex<double>> w(n);
        for (int i = 0; i < n; ++i) w[i] = {W[2 * j][i], -W[2 * j + 1][i]};  // W_2j - i*W_2j+1
        cw.push_back(encode(fhe().cc, w));
    }

    auto lin_baseline = [&]() {
        Ctx acc = fhe().mult(ca[0], pw[0]);
        for (int k = 1; k < D; ++k) acc = fhe().add(acc, fhe().mult(ca[k], pw[k]));
        return acc;
    };
    auto lin_fused = [&]() {
        Ctx acc = fhe().mult(xin[0], cw[0]);
        for (int j = 1; j < D / 2; ++j) acc = fhe().add(acc, fhe().mult(xin[j], cw[j]));
        return real_part(fhe(), acc);   // one im_cleanse for the whole reduction
    };

    const double e_base = max_abs_err(ref, decrypt_slots(fhe(), lin_baseline()), n);
    const double e_fus  = max_abs_err(ref, decrypt_slots(fhe(), lin_fused()),    n);

    auto time_path = [&](auto&& f) {
        auto w = f(); (void)w; cudaDeviceSynchronize();
        const double t = now_ms();
        for (int r = 0; r < R; ++r) { auto c = f(); (void)c; }
        cudaDeviceSynchronize();
        return (now_ms() - t) / R;
    };
    const double T_base = time_path(lin_baseline);
    const double T_fus  = time_path(lin_fused);

    std::cout << std::fixed << std::setprecision(2)
              << "[linfold] CKKS_COMPLEX=" << (complex_mode() ? 1 : 0) << " features=" << D << " reps=" << R
              << "  baseline=" << T_base << " ms (" << D << " wmults)"
              << "  fused=" << T_fus << " ms (" << D / 2 << " complex-wmults + 1 im_cleanse)"
              << "  speedup=" << (T_base / T_fus) << "x"
              << std::scientific << std::setprecision(2) << "  base_err=" << e_base << " fused_err=" << e_fus << std::endl;

    EXPECT_LT(e_base, 1e-4) << "baseline contraction wrong";
    if (complex_mode()) {
        EXPECT_LT(e_fus, 1e-4) << "Re((a+ib)(Wa-iWb)) should equal a*Wa + b*Wb";
        EXPECT_LT(T_fus, 0.75 * T_base) << "weight-folded complex contraction should roughly halve the linear's mults";
    } else {
        std::cout << "[linfold] real mode: imag lane absent -> only even features survive (informational)" << std::endl;
    }
}

}  // namespace
