// =====================================================================================
// Disjoint proof: INPUT/contraction-axis complex packing is a SEPARATE lever from the
// shipped OUTPUT-row pack (apply_linear_outputpack / GeneralizedComplexLinear in
// test_complex_packed_ops.cu). Same BSGS matrix-vector model; the point of THIS file is to
// show, side by side, WHERE each scheme's gain comes from and where each one is the ONLY
// option.
//
// BSGS linear cost has two axes:
//   BABY  = babystep / INPUT / contraction rotations (expose each weight diagonal band)
//   GIANT = giantstep / OUTPUT blocks (placed by a rotation, then summed)
//   mults = BABY*GIANT plaintext-mults.
//
//   scheme        baby-rot | giant-rot   | mults          | conj | needs
//   --------------|---------|-------------|----------------|------|---------------------------
//   baseline      |  BABY   | GIANT-1     | GIANT*BABY     |  0   | -
//   OUTPUT-pack   |  BABY   | GIANT/2 - 1 | GIANT/2 * BABY |  1   | GIANT (r_o) EVEN
//   INPUT-pack    |  BABY   | GIANT-1     | GIANT * BABY/2 |  1   | BABY (r_i) EVEN  (<=> d^2>N)
//
// So BOTH halve the (cheap) plaintext-mults, but ONLY output-pack also halves the (expensive,
// keyswitch) giantstep rotations. Therefore:
//   * output-pack is the STRONGER lever WHEN AVAILABLE (it strictly dominates input-pack: same
//     mult saving + extra rotation saving), and its gain can exceed 2x when GIANT dominates.
//   * input-pack is the ONLY lever when GIANT is ODD -- in particular GIANT=1 (a SQUARE linear,
//     out_proj/q), where output-pack is structurally impossible. There the win is bounded by the
//     mult fraction and is a near-wash because the BABY rotations it CANNOT touch dominate.
//
// The new algebraic claim (proven in Algebra_ConjugatedWeightContraction): packing two input
// bands X = x_a + i*x_b against a CONJUGATED static weight P = W_a - i*W_b yields
//     Re(X*P) = x_a*W_a + x_b*W_b   (the exact two-band contraction; ONE conjugate to recover).
// The conjugate on the weight is load-bearing: the NON-conjugated weight gives the WRONG sign
// (x_a*W_a - x_b*W_b). The imaginary lane carries antisymmetric junk, NOT a second usable output
// -- which is exactly why input-pack and output-pack cannot compose to 4x (one imag lane, spent
// once -> the hard 2x-per-linear ceiling).
//
// Run BOTH modes via scripts/17_test_complex_packed_inputpack.sh (CKKS_COMPLEX 0 then 1).
// =====================================================================================
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

using InputPackTest = CkksFixture;

bool complex_mode() { const char* cx = std::getenv("CKKS_COMPLEX"); return cx && cx[0] == '1'; }
double now_ms() { using namespace std::chrono; return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count(); }
double max_abs_err(const std::vector<double>& ref, const std::vector<double>& got, int n) {
    double m = 0.0; for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(got[i] - ref[i])); return m;
}
int env_int(const char* k, int dflt) { const char* v = std::getenv(k); return (v && *v) ? std::atoi(v) : dflt; }

// x *= i via the monomial X^(N/2): NO key-switch / rescale -> the input pairing is depth-free
// (the same trick the K/V Mode-A pack uses). Valid on fresh / non-bootstrapped cts.
Ctx times_i_mono(CKKSContext& fhe, const Ctx& x) {
    Ctx t = fhe.clone(x);
    fhe.cc->EvalMultMonomialInPlace(t, static_cast<uint32_t>(fhe.cc->GetRingDimension() / 2));
    return t;
}
// Re(P) = (P + conj(P))*0.5. ONE conjugate (the load-bearing primitive) + an add + a foldable 0.5.
Ctx real_part(CKKSContext& fhe, const Ctx& P) { return fhe.mult(fhe.add(P, fhe.conjugate(P)), 0.5); }
// Im(P) carried into real slots = (conj(P) - P)*i*0.5.
Ctx imag_part(CKKSContext& fhe, const Ctx& P) { return fhe.mult(times_i_mono(fhe, fhe.sub(fhe.conjugate(P), P)), 0.5); }

template <class F> double time_ms(int R, F&& f) {
    auto w = f(); (void)w; cudaDeviceSynchronize();
    const double t = now_ms();
    for (int r = 0; r < R; ++r) { auto c = f(); (void)c; }
    cudaDeviceSynchronize();
    return (now_ms() - t) / R;
}

// =====================================================================================
// 0) The core algebra, on raw slots (no BSGS plumbing). Proves the conjugated-weight identity
//    that makes input-pack work, shows the conjugate is necessary, contrasts with output-pack,
//    and shows the imag lane is junk (NOT a free second output) => no 4x composition.
// =====================================================================================
TEST_F(InputPackTest, Algebra_ConjugatedWeightContraction) {
    const int n = slots();
    std::mt19937 gen(31);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<double> xa(n), xb(n), wa(n), wb(n);
    for (int i = 0; i < n; ++i) { xa[i] = d(gen); xb[i] = d(gen); wa[i] = d(gen); wb[i] = d(gen); }

    Ctx cxa = encrypt(fhe().cc, encode(fhe().cc, xa), fhe().pk());
    Ctx cxb = encrypt(fhe().cc, encode(fhe().cc, xb), fhe().pk());

    // INPUT-pack: X = x_a + i*x_b (free i-fold), weight CONJUGATE-packed P = W_a - i*W_b.
    Ctx X = fhe().add(cxa, times_i_mono(fhe(), cxb));
    std::vector<std::complex<double>> wconj(n), wplain(n), wout(n);
    for (int i = 0; i < n; ++i) {
        wconj[i]  = { wa[i], -wb[i] };   // P_conj  = W_a - i*W_b   (correct input-pack weight)
        wplain[i] = { wa[i],  wb[i] };   // P_plain = W_a + i*W_b   (WRONG: gives x_a*W_a - x_b*W_b)
        wout[i]   = { wa[i],  wb[i] };   // output-pack weight, applied to a REAL input
    }
    Ptx pconj  = encode(fhe().cc, wconj);
    Ptx pplain = encode(fhe().cc, wplain);
    Ptx pout   = encode(fhe().cc, wout);

    // (1) NEW point: Re(X * P_conj) == x_a*W_a + x_b*W_b (exact two-band contraction).
    Ctx prod_conj = fhe().mult(X, pconj);
    std::vector<double> got_re = decrypt_slots(fhe(), real_part(fhe(), prod_conj));
    std::vector<double> ref_contract(n), ref_wrong(n);
    for (int i = 0; i < n; ++i) {
        ref_contract[i] = xa[i] * wa[i] + xb[i] * wb[i];   // what we WANT
        ref_wrong[i]    = xa[i] * wa[i] - xb[i] * wb[i];   // what the non-conjugated weight gives
    }
    const double e_contract = max_abs_err(ref_contract, got_re, n);

    // (2) Why the conjugate is necessary: Re(X * P_plain) == x_a*W_a - x_b*W_b (WRONG sign).
    Ctx prod_plain = fhe().mult(X, pplain);
    std::vector<double> got_plain = decrypt_slots(fhe(), real_part(fhe(), prod_plain));
    const double e_plain_is_wrong   = max_abs_err(ref_wrong,    got_plain, n);   // small  -> it equals the WRONG form
    const double e_plain_vs_contract = max_abs_err(ref_contract, got_plain, n);  // LARGE  -> it is NOT the contraction

    // (3) Output-pack contrast: REAL input x_a against W_a + i*W_b -> Re=x_a*W_a, Im=x_a*W_b
    //     (TWO clean outputs, no cross terms, no conjugate needed to keep them).
    Ctx prod_out = fhe().mult(cxa, pout);
    std::vector<double> out_re = decrypt_slots(fhe(), real_part(fhe(), prod_out));
    std::vector<double> out_im = decrypt_slots(fhe(), imag_part(fhe(), prod_out));
    std::vector<double> ref_oa(n), ref_ob(n);
    for (int i = 0; i < n; ++i) { ref_oa[i] = xa[i] * wa[i]; ref_ob[i] = xa[i] * wb[i]; }
    const double e_out_re = max_abs_err(ref_oa, out_re, n);
    const double e_out_im = max_abs_err(ref_ob, out_im, n);

    // (4) No-4x: the input-pack imag lane is antisymmetric junk x_b*W_a - x_a*W_b, NOT a clean
    //     second output -> the imag lane is CONSUMED, so you cannot also output-pack on it.
    std::vector<double> junk = decrypt_slots(fhe(), imag_part(fhe(), prod_conj));
    std::vector<double> ref_junk(n), ref_clean(n);
    for (int i = 0; i < n; ++i) { ref_junk[i] = xb[i] * wa[i] - xa[i] * wb[i]; ref_clean[i] = xa[i] * wa[i]; }
    const double e_junk_is_antisym = max_abs_err(ref_junk,  junk, n);   // small -> it IS the junk
    const double e_junk_is_output  = max_abs_err(ref_clean, junk, n);   // LARGE -> it is NOT a usable output

    std::cout << std::scientific << std::setprecision(2)
              << "[algebra] CKKS_COMPLEX=" << (complex_mode() ? 1 : 0) << "\n"
              << "  (1) Re(X*Wconj) vs x_a W_a + x_b W_b   err=" << e_contract << "  (want ~0: input-pack identity)\n"
              << "  (2) Re(X*Wplain) == WRONG x_a W_a - x_b W_b err=" << e_plain_is_wrong
              << "  ; vs contraction err=" << e_plain_vs_contract << "  (want LARGE: conjugate necessary)\n"
              << "  (3) output-pack Re=x_a W_a err=" << e_out_re << "  Im=x_a W_b err=" << e_out_im << "  (two clean outputs)\n"
              << "  (4) input-pack Im is junk: vs antisym err=" << e_junk_is_antisym
              << "  ; vs clean output err=" << e_junk_is_output << "  (imag lane consumed -> no 4x)\n";

    if (complex_mode()) {
        // 1e-6 floor: fresh (non-bootstrapped) cts, but stay well above the ~7e-9 GPU-decrypt
        // nondeterminism floor. The contrast checks are O(1) apart, so 1e-2 separates them cleanly.
        EXPECT_LT(e_contract, 1e-6) << "input-pack: Re(X*Wconj) must be the exact two-band contraction";
        EXPECT_LT(e_plain_is_wrong, 1e-6) << "non-conjugated weight gives x_a W_a - x_b W_b";
        EXPECT_GT(e_plain_vs_contract, 1e-2) << "non-conjugated weight is NOT the contraction (conjugate is necessary)";
        EXPECT_LT(e_out_re, 1e-6) << "output-pack Re lane";
        EXPECT_LT(e_out_im, 1e-6) << "output-pack Im lane";
        EXPECT_LT(e_junk_is_antisym, 1e-6) << "input-pack imag lane is the antisymmetric cross-term";
        EXPECT_GT(e_junk_is_output, 1e-2) << "input-pack imag lane is junk, not a free second output (=> no 4x)";
    } else {
        std::cout << "[algebra] real mode: imag lane absent -> only the real lane survives (informational)\n";
    }
}

// =====================================================================================
// BSGS workload shared by both regimes. baseline / output-pack / input-pack run the SAME baby
// and giant rotations; only the contraction structure differs. Returns timings + a correctness
// check of input-pack against baseline (input-pack is OUTPUT-LAYOUT-PRESERVING: a drop-in cheaper
// contraction, so its Re lane must EQUAL the baseline output bit-for-bit modulo bts noise).
// =====================================================================================
struct RegimeResult { double t_base, t_out, t_in, in_err; bool out_ran; };

RegimeResult run_regime(CKKSContext& fhe, int n, int BABY, int GIANT, int R) {
    std::mt19937 gen(26);
    std::uniform_real_distribution<double> d(-0.5, 0.5);
    std::vector<double> x(n); for (double& v : x) v = d(gen);
    Ctx cx = encrypt(fhe.cc, encode(fhe.cc, x), fhe.pk());

    auto rnd = [&]() { std::vector<double> v(n); for (double& e : v) e = d(gen); return v; };
    std::vector<std::vector<double>> W(static_cast<size_t>(GIANT) * BABY);
    for (auto& w : W) w = rnd();

    // real diagonals
    std::vector<Ptx> pw; pw.reserve(W.size());
    for (auto& w : W) pw.push_back(encode(fhe.cc, w));
    // OUTPUT-pack weights: pair output blocks (2gp, 2gp+1), real input -> W[2gp][j] + i*W[2gp+1][j]
    std::vector<Ptx> pw_out;
    if (GIANT % 2 == 0) {
        pw_out.reserve(static_cast<size_t>(GIANT / 2) * BABY);
        for (int gp = 0; gp < GIANT / 2; ++gp)
            for (int j = 0; j < BABY; ++j) {
                std::vector<std::complex<double>> wc(n);
                for (int i = 0; i < n; ++i) wc[i] = { W[(2 * gp) * BABY + j][i], W[(2 * gp + 1) * BABY + j][i] };
                pw_out.push_back(encode(fhe.cc, wc));
            }
    }
    // INPUT-pack weights: pair babystep bands (2jp, 2jp+1), CONJUGATED -> W[g][2jp] - i*W[g][2jp+1]
    std::vector<Ptx> pw_in;
    pw_in.reserve(static_cast<size_t>(GIANT) * (BABY / 2));
    for (int g = 0; g < GIANT; ++g)
        for (int jp = 0; jp < BABY / 2; ++jp) {
            std::vector<std::complex<double>> wc(n);
            for (int i = 0; i < n; ++i) wc[i] = { W[g * BABY + 2 * jp][i], -W[g * BABY + 2 * jp + 1][i] };
            pw_in.push_back(encode(fhe.cc, wc));
        }

    auto baby_step  = [](int j) { return 1 << (1 + (j % 14)); };
    auto giant_step = [](int g) { return 1 << (1 + (g % 14)); };
    auto babysteps = [&]() {
        std::vector<Ctx> rx; rx.reserve(BABY);
        for (int j = 0; j < BABY; ++j) rx.push_back(j == 0 ? cx : fhe.rotate(cx, baby_step(j)));
        return rx;
    };

    // baseline: GIANT output blocks, each BABY real mults; GIANT-1 giantstep rotations.
    auto lin_baseline = [&]() {
        auto rx = babysteps();
        Ctx result = fhe.mult(rx[0], pw[0]);
        for (int j = 1; j < BABY; ++j) result = fhe.add(result, fhe.mult(rx[j], pw[j]));
        for (int g = 1; g < GIANT; ++g) {
            Ctx acc = fhe.mult(rx[0], pw[g * BABY + 0]);
            for (int j = 1; j < BABY; ++j) acc = fhe.add(acc, fhe.mult(rx[j], pw[g * BABY + j]));
            result = fhe.add(result, fhe.rotate(acc, giant_step(g)));
        }
        return result;
    };
    // OUTPUT-pack: GIANT/2 complex blocks (BABY cplx-mults each) + GIANT/2-1 giant-rot + 1 conj.
    // Halves the OUTPUT axis (mults AND giant rotations). Timing skeleton (real_part drops odd rows;
    // correctness of the layout is proven in test_complex_packed_ops.cu GeneralizedComplexLinear).
    auto lin_outputpack = [&]() {
        auto rx = babysteps();
        Ctx result = fhe.mult(rx[0], pw_out[0]);
        for (int j = 1; j < BABY; ++j) result = fhe.add(result, fhe.mult(rx[j], pw_out[j]));
        for (int gp = 1; gp < GIANT / 2; ++gp) {
            Ctx acc = fhe.mult(rx[0], pw_out[gp * BABY + 0]);
            for (int j = 1; j < BABY; ++j) acc = fhe.add(acc, fhe.mult(rx[j], pw_out[gp * BABY + j]));
            result = fhe.add(result, fhe.rotate(acc, giant_step(gp)));
        }
        return real_part(fhe, result);
    };
    // INPUT-pack: pair the BABY bands into BABY/2 complex inputs (free i-fold), CONJUGATED weights.
    // Halves the (cheap) mults; baby AND giant rotations UNCHANGED; +1 conjugate. OUTPUT-LAYOUT-
    // PRESERVING -> Re(result) EQUALS baseline. Works for ANY GIANT (incl. odd / GIANT=1).
    auto lin_inputpack = [&]() {
        auto rx = babysteps();
        std::vector<Ctx> X; X.reserve(BABY / 2);
        for (int jp = 0; jp < BABY / 2; ++jp) X.push_back(fhe.add(rx[2 * jp], times_i_mono(fhe, rx[2 * jp + 1])));
        const int H = BABY / 2;
        Ctx result = fhe.mult(X[0], pw_in[0]);
        for (int jp = 1; jp < H; ++jp) result = fhe.add(result, fhe.mult(X[jp], pw_in[jp]));
        for (int g = 1; g < GIANT; ++g) {
            Ctx acc = fhe.mult(X[0], pw_in[g * H + 0]);
            for (int jp = 1; jp < H; ++jp) acc = fhe.add(acc, fhe.mult(X[jp], pw_in[g * H + jp]));
            result = fhe.add(result, fhe.rotate(acc, giant_step(g)));
        }
        return real_part(fhe, result);   // ONE conjugate -> Re = full contraction == baseline
    };

    // Correctness: input-pack Re lane must equal baseline (drop-in cheaper contraction).
    std::vector<double> r_base = decrypt_slots(fhe, lin_baseline());
    std::vector<double> r_in   = decrypt_slots(fhe, lin_inputpack());
    const double in_err = max_abs_err(r_base, r_in, n);

    RegimeResult res{};
    res.out_ran = (GIANT % 2 == 0);
    res.t_base = time_ms(R, lin_baseline);
    res.t_out  = res.out_ran ? time_ms(R, lin_outputpack) : 0.0;
    res.t_in   = time_ms(R, lin_inputpack);
    res.in_err = in_err;
    return res;
}

void report_regime(const char* tag, int BABY, int GIANT, const RegimeResult& r) {
    const int base_mults = GIANT * BABY, base_grot = GIANT - 1;
    const int out_mults  = r.out_ran ? (GIANT / 2) * BABY : 0, out_grot = r.out_ran ? (GIANT / 2 - 1) : 0;
    const int in_mults   = GIANT * (BABY / 2), in_grot = GIANT - 1;
    std::cout << std::fixed << std::setprecision(3)
              << "[" << tag << "] CKKS_COMPLEX=" << (complex_mode() ? 1 : 0)
              << " baby=" << BABY << " giant=" << GIANT << "\n"
              << "  baseline    " << r.t_base << " ms  (baby-rot " << BABY << ", giant-rot " << base_grot
              << ", mults " << base_mults << ")\n";
    if (r.out_ran)
        std::cout << "  OUTPUT-pack " << r.t_out << " ms  (baby-rot " << BABY << ", giant-rot " << out_grot
                  << " [HALVED], cplx-mults " << out_mults << " [HALVED], 1 conj)   speedup="
                  << (r.t_base / r.t_out) << "x\n";
    else
        std::cout << "  OUTPUT-pack   N/A   (GIANT=" << GIANT << " is ODD -> cannot pair output blocks)\n";
    std::cout << "  INPUT-pack  " << r.t_in << " ms  (baby-rot " << BABY << " [unchanged], giant-rot " << in_grot
              << " [unchanged], cplx-mults " << in_mults << " [HALVED], 1 conj)   speedup="
              << (r.t_base / r.t_in) << "x"
              << std::scientific << std::setprecision(2) << "  re_err=" << r.in_err << "\n";
}

// =====================================================================================
// 1) OUTPUT-HEAVY (GIANT large & even, BABY small). Output-pack halves the dominant giantstep
//    rotations -> big win (can exceed 2x). Input-pack only halves the small mult mass, so it wins
//    much less. PROVES: output-pack draws its gain from the OUTPUT/giant axis; when GIANT is even
//    it strictly dominates input-pack.
// =====================================================================================
TEST_F(InputPackTest, Regime_OutputHeavy_OutputPackDominates) {
    const int BABY = env_int("OH_BABY", 4), GIANT = env_int("OH_GIANT", 32), R = env_int("REPS", 20);
    auto r = run_regime(fhe(), slots(), BABY, GIANT, R);
    report_regime("output-heavy", BABY, GIANT, r);
    if (complex_mode()) {
        EXPECT_LT(r.in_err, 1e-4) << "input-pack must reproduce the baseline output";
        EXPECT_LT(r.t_in,  r.t_base) << "input-pack should still beat baseline (mults halved)";
        EXPECT_LT(r.t_out, r.t_base) << "output-pack should beat baseline";
        EXPECT_LT(r.t_out, r.t_in)   << "OUTPUT-heavy: output-pack must beat input-pack (it also halves giant rotations)";
    }
}

// =====================================================================================
// 2) SQUARE (GIANT=1). Output-pack is IMPOSSIBLE (no output blocks to pair) -- this is out_proj/q
//    in the real model (r_o=1, odd). Input-pack is the ONLY complex lever, and since there are NO
//    giant rotations and the BABY contraction rotations dominate (input-pack cannot touch them),
//    halving only the cheap mults is a NEAR-WASH. PROVES: input-pack reaches a regime output-pack
//    cannot, but its square-matrix payoff is small -- the verified out_proj conclusion.
// =====================================================================================
TEST_F(InputPackTest, Regime_Square_OnlyInputPackApplies) {
    const int BABY = env_int("SQ_BABY", 32), GIANT = 1, R = env_int("REPS", 20);
    auto r = run_regime(fhe(), slots(), BABY, GIANT, R);
    report_regime("square", BABY, GIANT, r);
    if (complex_mode()) {
        EXPECT_FALSE(r.out_ran) << "output-pack must be unavailable for GIANT=1 (square)";
        EXPECT_LT(r.in_err, 1e-4) << "input-pack must reproduce the baseline output on the square linear";
        // Deliberately NO timing assertion: this regime is a NEAR-WASH. With the BABY contraction
        // rotations dominating (input-pack cannot touch them) and only the cheap mults halved (minus
        // one added conjugate), the speedup can land marginally either side of 1.0x. The printed
        // number is the deliverable -- it quantifies exactly how close to break-even out_proj is.
    }
}

// =====================================================================================
// 3) INPUT-HEAVY but GIANT EVEN (both large). Both schemes can run; both halve the (now dominant)
//    mult mass, but output-pack ALSO halves the giant rotations -> output-pack >= input-pack
//    everywhere GIANT is even. Confirms input-pack is never the better choice WHEN output-pack is
//    available; its value is strictly in the GIANT-odd regime above.
// =====================================================================================
TEST_F(InputPackTest, Regime_InputHeavy_OutputPackStillWins) {
    const int BABY = env_int("IH_BABY", 16), GIANT = env_int("IH_GIANT", 8), R = env_int("REPS", 20);
    auto r = run_regime(fhe(), slots(), BABY, GIANT, R);
    report_regime("input-heavy", BABY, GIANT, r);
    if (complex_mode()) {
        EXPECT_LT(r.in_err, 1e-4) << "input-pack must reproduce the baseline output";
        EXPECT_LT(r.t_in,  r.t_base) << "input-pack beats baseline";
        EXPECT_LT(r.t_out, r.t_base) << "output-pack beats baseline";
        EXPECT_LE(r.t_out, r.t_in * 1.05) << "GIANT even: output-pack should be <= input-pack (also halves giant rot)";
    }
}

// Recover (Re, Im) sharing ONE conjugate (mirrors fideslib_wrapper.h unpack_ri).
std::pair<Ctx, Ctx> unpack_ri_local(CKKSContext& fhe, const Ctx& P) {
    Ctx cj = fhe.conjugate(P);
    Ctx re = fhe.mult(fhe.add(P, cj), 0.5);
    Ctx im = fhe.mult(times_i_mono(fhe, fhe.sub(cj, P)), 0.5);
    return {re, im};
}

// =====================================================================================
// 5) AGGREGATION axis (answering: "complex halves the partial-sum aggregation, saving ~t rotations
//    for one conjugate"). The cachemir replication AND aggregation are LOG-TREE reductions
//    (cachemir_linear.cu:30 `for step *= 2`, :78 same): aggregating t partials costs log2(t)
//    rotate-adds, NOT t. So there are not ~t rotations to save -- only log2(t).
//    What complex packing can actually do to a reduction:
//      (A) baseline      reduce R real lanes                         -> log2(R) rotate-adds, 1 sum
//      (B) complex-halve pack pairs, reduce R/2 lanes + unpack       -> log2(R)-1 rot + 1 conj, 1 sum
//      (C) free-rider    reduce TWO reals packed in re/im + unpack   -> log2(R) rot + 1 conj, 2 sums
//    (B) saves exactly the TOP tree level (ONE rotation) and pays one conjugate ~= one rotation =>
//    a WASH on a SINGLE reduction (a sum-over-slots cannot span the imag lane; it rotates in
//    lockstep, not as extra slots). (C) is the genuine reduction win: TWO sums for ~one reduction
//    -- and that 2-for-1 is EXACTLY the mechanism by which output-pack halves the giantstep
//    rotations (its two output lanes share the placement/reduction). Input-pack's two lanes are
//    (real output, junk), so its reduction is NOT shared -> it does not save aggregation rotations.
// =====================================================================================
TEST_F(InputPackTest, Aggregation_ComplexShavesOneTreeLevelOnly) {
    const int n = slots();
    const int R = env_int("RED_LANES", 32);       // # partials to aggregate (= cachemir t for the square)
    const int REP = env_int("REPS", 20);
    const int LOGR = static_cast<int>(std::lround(std::log2(static_cast<double>(R))));

    std::mt19937 gen(41);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<double> p(n, 0.0), pe(n, 0.0), po(n, 0.0), u(n, 0.0), w(n, 0.0);
    double sp = 0, su = 0, sw = 0;
    for (int i = 0; i < R; ++i) { p[i] = d(gen); u[i] = d(gen); w[i] = d(gen); sp += p[i]; su += u[i]; sw += w[i]; }
    for (int k = 0; k < R / 2; ++k) { pe[k] = p[2 * k]; po[k] = p[2 * k + 1]; }   // pair partials onto R/2 lanes

    Ctx cp  = encrypt(fhe().cc, encode(fhe().cc, p), fhe().pk());
    Ctx cq  = fhe().add(encrypt(fhe().cc, encode(fhe().cc, pe), fhe().pk()),
                        times_i_mono(fhe(), encrypt(fhe().cc, encode(fhe().cc, po), fhe().pk())));  // p_even + i*p_odd
    Ctx cuw = fhe().add(encrypt(fhe().cc, encode(fhe().cc, u), fhe().pk()),
                        times_i_mono(fhe(), encrypt(fhe().cc, encode(fhe().cc, w), fhe().pk())));   // u + i*w

    auto reduce = [&](Ctx v, int lanes) { for (int s = 1; s < lanes; s *= 2) v = fhe().add(v, fhe().rotate(v, s)); return v; };
    auto agg_base = [&]() { return reduce(cp, R); };                                          // log2(R) rot
    auto agg_half = [&]() { auto ri = unpack_ri_local(fhe(), reduce(cq, R / 2)); return fhe().add(ri.first, ri.second); };  // log2(R)-1 rot + 1 conj
    auto agg_free = [&]() { return unpack_ri_local(fhe(), reduce(cuw, R)); };                 // log2(R) rot + 1 conj -> 2 sums

    const double e_base = std::fabs(decrypt_slots(fhe(), agg_base())[0] - sp);
    const double e_half = std::fabs(decrypt_slots(fhe(), agg_half())[0] - sp);
    auto fr = agg_free();
    const double e_fu = std::fabs(decrypt_slots(fhe(), fr.first)[0]  - su);
    const double e_fw = std::fabs(decrypt_slots(fhe(), fr.second)[0] - sw);

    const double T_base = time_ms(REP, agg_base);
    const double T_half = time_ms(REP, agg_half);
    const double T_free = time_ms(REP, agg_free);

    std::cout << std::fixed << std::setprecision(3)
              << "[aggregation] CKKS_COMPLEX=" << (complex_mode() ? 1 : 0) << " lanes R=" << R << " (log2R=" << LOGR << ")\n"
              << "  (A) baseline       " << T_base << " ms  (" << LOGR << " rot)                 1 sum\n"
              << "  (B) complex-halve  " << T_half << " ms  (" << (LOGR - 1) << " rot + 1 conj)        1 sum   ratio_vs_base="
              << (T_half / T_base) << "x  (want ~1.0 -> WASH, NOT a halving)\n"
              << "  (C) free-rider     " << T_free << " ms  (" << LOGR << " rot + 1 conj)        2 sums  per-sum_vs_base="
              << (T_free / 2.0 / T_base) << "x  (the real reduction win = output-pack's shared placement)\n"
              << std::scientific << std::setprecision(2)
              << "  err base=" << e_base << " half=" << e_half << " free_u=" << e_fu << " free_w=" << e_fw << "\n";

    if (complex_mode()) {
        EXPECT_LT(e_half, 1e-6) << "complex-halved single reduction must still equal the full sum";
        EXPECT_LT(e_fu, 1e-6) << "free-rider sum u";
        EXPECT_LT(e_fw, 1e-6) << "free-rider sum w";
        // A single log-tree reduction CANNOT be halved by complex packing: shaving one of log2(R)
        // rotations and paying a conjugate is a wash, not ~t rotations saved.
        EXPECT_GT(T_half, 0.7 * T_base) << "complex cannot meaningfully halve a single log-tree reduction";
        // The genuine win is 2-for-1 (free-rider) -> per-sum well under the baseline single reduction.
        EXPECT_LT(T_free, 1.6 * T_base) << "free-rider: TWO sums for ~one reduction";
    } else {
        std::cout << "[aggregation] real mode: imag lane absent -> (B)/(C) collapse (informational)\n";
    }
}

}  // namespace
