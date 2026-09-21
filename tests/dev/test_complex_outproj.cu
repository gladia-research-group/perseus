// Phase-3 out_proj INPUT-fold proof (the V-side latency win, gated on the composition gap).
//
// out_proj is square (d x d), so compute_cm_params gives r_o=1 — it is NOT an output-pack (S4),
// it is an input-CONTRACTION fold. Pair the contraction features: a complex-packed input
// (x[2j] + i*x[2j+1]) times a CONJUGATED complex weight (W[2j] - i*W[2j+1]) gives, in its real
// part, x[2j]*W[2j] + x[2j+1]*W[2j+1] — Re((a+ib)(Wa-iWb)) = a*Wa + b*Wb (test_complex_packed_vcache
// test 5, isolated). This halves the effective d_in (d -> d/2), so the cachemir linear runs at
// (d/2 x d): n_pt 32 -> 16 complex plaintexts. We prove it reproduces the real (d x d) linear AND
// report the n_pt halving. The weights are STATIC (pre-encoded) so this genuinely wins (unlike the
// runtime-score P*V fuse, which is a wash). NEEDS CKKS_COMPLEX=1.

#include "all_blocks_test_helpers.h"
#include "model/gpt2.h"
#include "packing/cachemir/cachemir_linear.h"
#include "packing/cachemir/cachemir_linear_utils.h"
#include "packing/cachemir/cachemir_rot_indices.h"
#include "test_helpers.h"

#include <gtest/gtest.h>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace test_helpers;

namespace {

// Complex cachemir linear-input for the folded (d_in x d_out) shape (is_up: feature j at slot j*t),
// pairing x[2j],x[2j+1] into Re/Im — the out_proj contraction fold's input.
PackedCtx encode_complex_linear_input_pair(Inference& inf, const std::vector<double>& x,
                                           int d_in, int d_out, int level) {
    auto p = cachemir::compute_cm_params(inf.slots, d_in, d_out);
    std::vector<std::complex<double>> ptx(inf.slots, {0.0, 0.0});
    for (int j = 0; j < p.d; ++j)
        ptx[j * p.t] = std::complex<double>(x[2 * j], x[2 * j + 1]);
    Ctx ct = encrypt(inf.cc(),
                     inf.cc()->MakeCKKSPackedPlaintext(ptx, 1, static_cast<uint32_t>(level)),
                     inf.fhe->pk());
    return inf.pack(ct, PackingKind::Cachemir);
}

}  // namespace

TEST(ComplexOutProjTest, InputFoldEqualsRealLinear) {
    const int d     = 1024;        // padded out_proj is d x d
    const int slots = 1 << 15;     // logN=16 → 32768
    CKKSContextOptions ckks{};
    ckks.bts_iterations       = default_bts_iterations();
    ckks.ckks_complex_payload = true;
    // Seed rotation keys for the folded (d/2 x d) contraction shape (not a model shape).
    ckks.extra_rot_steps = cachemir::linear_rot_indices(slots, d / 2, d);
    Inference inf = make_gpt2_inference({
        .ckks = ckks, .hidDim = d, .expDim = 4096, .numHeads = 16,
        .bench_mode = false, .packing_kind = PackingKind::Cachemir,
    });

    std::mt19937 gen(7);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::vector<double>> W(d, std::vector<double>(d));   // W[i=d_in][j=d_out]
    std::vector<double> x(d);
    for (int i = 0; i < d; ++i) {
        x[i] = dist(gen);
        for (int j = 0; j < d; ++j) W[i][j] = dist(gen);
    }
    std::vector<double> y_ref(d, 0.0);
    for (int j = 0; j < d; ++j) {
        double s = 0.0;
        for (int i = 0; i < d; ++i) s += x[i] * W[i][j];
        y_ref[j] = s;
    }

    constexpr int L = 17;

    // --- real baseline linear (d x d) ---
    inf.w["outref"] = cachemir::encode_weight_matrix(inf, W, d, d, L);
    inf.w.erase("outref_bias");
    {
        PackedCtx xr = cachemir::encode_linear_input(inf, x, d, d, L);
        PackedCtx yr = cachemir::linear(inf, xr, "outref", d, d);
        auto out = cachemir::decode_linear_output(inf.slots, decrypt_slots(inf, yr), d, d);
        out.resize(d);
        auto s = compare_vec(out, y_ref);
        std::cout << "[outproj real] " << d << "x" << d
                  << "  n_pt=" << cachemir::compute_cm_params(inf.slots, d, d).n_pt
                  << "  max_abs=" << s.max_abs << " max_rel=" << s.max_rel << "\n";
        EXPECT_LT(s.max_rel, 1e-3) << "real baseline linear wrong (harness/keys)";
    }

    // --- folded complex linear (d/2 x d): pair contraction features, conjugated weight ---
    const int dh = d / 2;
    std::vector<std::vector<double>> W_re(dh, std::vector<double>(d));
    std::vector<std::vector<double>> W_im(dh, std::vector<double>(d));
    for (int i = 0; i < dh; ++i)
        for (int j = 0; j < d; ++j) {
            W_re[i][j] =  W[2 * i][j];
            W_im[i][j] = -W[2 * i + 1][j];   // conjugated: W_2j - i*W_2j+1
        }

    inf.w["outfold"] = cachemir::encode_weight_matrix_complex(inf, W_re, W_im, dh, d, L);
    inf.w.erase("outfold_bias");
    inf.complex_weight_names.insert("outfold");   // weights_at must re-level preserving imag
    {
        PackedCtx xf = encode_complex_linear_input_pair(inf, x, dh, d, L);
        PackedCtx yf = cachemir::apply_linear(
            inf, cachemir::prepare_linear_input(inf, xf, dh, d), "outfold", dh, d);
        inf.fhe->inplace_im_cleanse(yf);             // 2*Re(contraction)
        yf = inf.fhe->mult(yf, 0.5);
        auto out = cachemir::decode_linear_output(inf.slots, decrypt_slots(inf, yf), dh, d);
        out.resize(d);
        auto s = compare_vec(out, y_ref);
        const auto pf = cachemir::compute_cm_params(inf.slots, dh, d);
        std::cout << "[outproj fold] " << dh << "x" << d
                  << "  n_pt=" << pf.n_pt << " (complex, vs 32 real → half the plaintext mults)"
                  << "  max_abs=" << s.max_abs << " max_rel=" << s.max_rel << "\n";
        EXPECT_LT(s.max_rel, 1e-3) << "complex input-fold != real out_proj contraction";
    }
}
