// Rung-1 token-pair prefill round-trip. Encode a 2t-token chunk into ONE complex ct (Re = tokens
// [0,t), Im = tokens [t,2t)) via the production encode_input_token_pair, split it with the
// production deg-preserving CKKSContext::conj_split (A = 2*Re, B = 2i*Im), and assert each half
// recovers the right tokens in the filling slot layout slot[i*t + tok]. Validates
// encode_input_token_pair (cachemir_filling_complex_io.cu) + conj_split (fideslib_wrapper.h).
// NEEDS the complex payload (forced in-code via ckks.ckks_complex_payload=true).

#include "model/gpt2.h"
#include "test_helpers.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace test_helpers;

TEST(PrefillTokenPairIO, EncodeUnpackRoundTrip) {
    CKKSContextOptions ckks{};
    ckks.bts_iterations       = default_bts_iterations();
    ckks.ckks_complex_payload = true;                       // token-pair requires the complex payload
    Inference inf = make_gpt2_inference({
        .ckks = ckks, .hidDim = 1024, .expDim = 4096, .numHeads = 16,
        .bench_mode = false, .packing_kind = PackingKind::CachemirFilling,
    });
    inf.token_pair = true;                                  // route encode_prefill_input -> token-pair encode

    const int t      = inf.slots / inf.size.hidDim;         // = 32
    const int d_real = inf.size.getRealHidDim();            // = 768
    const int T      = 48;                                  // nA=32 (full A) + nB=16 (partial B, exercises nB<t)
    const int nA     = std::min(t, T);
    const int nB     = std::max(0, T - t);

    std::mt19937 gen(11);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::vector<double>> prompt(T, std::vector<double>(d_real));
    for (auto& row : prompt) for (double& v : row) v = dist(gen);

    PackedCtx P = encode_prefill_input(inf, prompt);        // production dispatch on inf.token_pair
    EXPECT_EQ(inf.n_tok, nA);
    EXPECT_EQ(inf.n_tok_imag, nB);

    // production deg-preserving unpack: A = 2*Re(P), B = 2i*Im(P). Realify B by *(-i) (= production
    // V-realify): 2i*Im * (-i) = 2*Im (real). conj_split's B is +2i*Im, so -i (NOT +i) recovers +2*Im.
    auto [A, B] = inf.fhe->conj_split(P);
    Ptx neg_i = inf.encode_complex_const_at(0.0, -1.0, B);
    PackedCtx B_real = inf.fhe->mult(B, neg_i);

    auto a_slots = decrypt_slots(inf, A);                   // real: 2*Re
    auto b_slots = decrypt_slots(inf, B_real);              // real: 2*Im

    // half A carries tokens [0, nA); half B carries tokens [t, t+nB); slot = i*t + tok
    std::vector<double> a_got, a_ref, b_got, b_ref;
    for (int i = 0; i < d_real; ++i) {
        for (int tok = 0; tok < nA; ++tok) {
            a_got.push_back(a_slots[i * t + tok]);
            a_ref.push_back(2.0 * prompt[tok][i]);
        }
        for (int tok = 0; tok < nB; ++tok) {
            b_got.push_back(b_slots[i * t + tok]);
            b_ref.push_back(2.0 * prompt[t + tok][i]);
        }
    }
    auto sa = compare_vec(a_got, a_ref);
    auto sb = compare_vec(b_got, b_ref);
    std::cout << std::scientific << std::setprecision(3)
              << "[A=2Re tokens[0," << nA << ")]  max_abs=" << sa.max_abs
              << "  mean_rel=" << sa.mean_rel << "\n"
              << "[B=2Im tokens[" << t << "," << (t + nB) << ")]  max_abs=" << sb.max_abs
              << "  mean_rel=" << sb.mean_rel << "\n";

    // absolute gate (refs span near-zero -> a relative gate false-fails a correct unpack); values are
    // O(2), fresh non-bootstrapped encode, so abs err is well below 1e-6 and above the ~7e-9 floor.
    ASSERT_LT(sa.max_abs, 1e-6) << "A half != 2*Re(tokens[0,t))";
    ASSERT_LT(sb.max_abs, 1e-6) << "B half != 2*Im(tokens[t,2t)) -- if ~2*Re, the realify sign is wrong";

    // B imag lane is unset past nB -> that slot must be ~0 (its A/real half holds emb_A there)
    EXPECT_LT(std::abs(b_slots[3 * t + nB]), 1e-6) << "B imag lane leaked past nB";
}
