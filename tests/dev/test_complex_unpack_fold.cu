// INVST-01 proof: fold unpack_ri's Re-branch 0.5 scalar mult into the output-pack weight.
//
// Output-pack linear (apply_linear_outputpack) packs output cols (2k',2k'+1) into Re/Im of one
// complex contraction result P, then unpack_ri (fideslib_wrapper.h:1574) splits:
//     re = 0.5*(P + conj(P))          // Re(P)  -- a scalar mult
//     im = (P - conj(P)) * (-i/2)     // Im(P)  -- a plaintext mult (nhi = -0.5i)
// Both branches do ONE mult, so re/im exit at the SAME noiseScaleDeg and the downstream cascade
// add (cachemir_linear.cu:136-138, which sums cy[2k']=re and cy[2k'+1]=im) is balanced.
//
// The proposed fold bakes the 0.5 into the STATIC weight (W -> 0.5*W, encode-time, free) so:
//     re = (P' + conj(P'))            // = Re(P) since P' = 0.5*P  -- NO mult (a plain add)
//     im = (P' - conj(P')) * (-i)     // = Im(P)                   -- one plaintext mult (nhi=-i)
// dropping ONE scalar mult/token. The risk (raised by the audit's level-correctness verifier):
// re now skips a mult while im keeps one, so re/im exit at DIFFERENT noiseScaleDeg -> the cascade
// add mixes mismatched-deg ciphertexts -> a possible level desync vs the token-0 captured plan.
//
// This test is DISJOINT from production: it reimplements apply_linear_outputpack locally so the
// ONLY difference between the base and fold arms is the unpack, registers its own throwaway
// weights, and never edits any shipped .cu/.h. It proves:
//   (1) fold output == plaintext GT (correctness),
//   (2) fold output == production linear_outputpack output (bit-identity),
//   (3) measures Re/Im/y (level, deg) for base vs fold -> answers "is the saving free or does it
//       desync and need a realign op / re-plan?"
// NEEDS CKKS_COMPLEX=1.

#include "model/gpt2.h"
#include "packing/cachemir/cachemir_attention_utils.h"   // mha_rot
#include "packing/cachemir/cachemir_linear.h"            // prepare_linear_input, linear_outputpack
#include "packing/cachemir/cachemir_linear_utils.h"      // compute_cm_params, encode_*, decode_linear_output
#include "packing/cachemir/cachemir_rot_indices.h"       // linear_rot_indices
#include "test_helpers.h"

#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace test_helpers;

namespace {

struct LevelProbe {
    int P_lvl = -1, P_deg = -1;     // the packed contraction result (pre-unpack)
    int re_lvl = -1, re_deg = -1;   // Re branch (block 2k')
    int im_lvl = -1, im_deg = -1;   // Im branch (block 2k'+1)
    int y_lvl = -1, y_deg = -1;     // final unpacked+cascaded output
};

// Local clone of cachemir::apply_linear_outputpack, parametrized by `folded`. The base arm (folded
// =false) reproduces production exactly (validated against the real linear_outputpack below); the
// fold arm (folded=true) assumes the weight already carries the 0.5 and uses nhi=-i, dropping the
// Re-branch scalar mult. Captures level/deg probes for kp==0.
PackedCtx run_outputpack_local(Inference& inf, const PackedCtx& x_in,
                               const std::string& wname, int d_in, int d_out,
                               bool folded, LevelProbe& pr) {
    auto x_rotated = cachemir::prepare_linear_input(inf, x_in, d_in, d_out);
    const auto p = cachemir::compute_cm_params(inf.slots, d_in, d_out);
    if (p.r_o % 2 != 0) throw std::runtime_error("run_outputpack_local: r_o must be even");
    const int rop = p.r_o / 2;
    auto pts_W = inf.weights_at(wname, x_rotated[0]);

    const int giant_rot = p.bstep_c * p.t * p.t;
    Ptx nhi_half = inf.encode_complex_const_at(0.0, -0.5, x_rotated[0]);   // baseline (-i/2)
    Ptx nhi_full = inf.encode_complex_const_at(0.0, -1.0, x_rotated[0]);   // folded   (-i)

    std::vector<PackedCtx> cy(p.r_o);
    for (int kp = 0; kp < rop; ++kp) {
        PackedCtx acc;
        for (int g = 0; g < p.gstep_c; ++g) {
            const int j0  = g * p.bstep_c;
            PackedCtx tmp = inf.fhe->mult(x_rotated[0], pts_W[(j0 + 0) * rop + kp]);
            for (int b = 1; b < p.bstep_c; ++b) {
                PackedCtx t2 = inf.fhe->mult(x_rotated[b], pts_W[(j0 + b) * rop + kp]);
                inf.fhe->inplace_add(tmp, t2);
            }
            if (g > 0) tmp = inf.fhe->rotate(tmp, cachemir::mha_rot(inf, g * giant_rot));
            if (g == 0) acc = std::move(tmp);
            else        inf.fhe->inplace_add(acc, tmp);
        }

        PackedCtx conj = inf.fhe->conjugate(acc);
        PackedCtx re, im;
        if (folded) {
            re = inf.fhe->add(acc, conj);                               // (P'+conj) = Re(P), no mult
            im = inf.fhe->mult(inf.fhe->sub(acc, conj), nhi_full);      // (P'-conj)*(-i) = Im(P)
        } else {
            re = inf.fhe->mult(inf.fhe->add(acc, conj), 0.5);           // production unpack_ri
            im = inf.fhe->mult(inf.fhe->sub(acc, conj), nhi_half);
        }
        if (kp == 0) {
            pr.P_lvl  = inf.fhe->level_for_ct(acc.ct); pr.P_deg  = static_cast<int>(acc.ct->GetNoiseScaleDeg());
            pr.re_lvl = inf.fhe->level_for_ct(re.ct);  pr.re_deg = static_cast<int>(re.ct->GetNoiseScaleDeg());
            pr.im_lvl = inf.fhe->level_for_ct(im.ct);  pr.im_deg = static_cast<int>(im.ct->GetNoiseScaleDeg());
        }
        cy[2 * kp]     = std::move(re);
        cy[2 * kp + 1] = std::move(im);
    }

    int cascade_rot = p.t * p.tp;
    for (int k = p.r_o - 1; k > 0; --k) {                              // <-- the mismatched-deg add site
        PackedCtx tmp = inf.fhe->rotate(cy[k], cachemir::mha_rot(inf, cascade_rot));
        inf.fhe->inplace_add(cy[k - 1], tmp);
    }

    PackedCtx y = cy[0];
    for (int step = 1; step < p.tp_out; step *= 2) {
        PackedCtx t = inf.fhe->rotate(y, cachemir::mha_rot(inf, step));
        inf.fhe->inplace_add(y, t);
    }
    pr.y_lvl = inf.fhe->level_for_ct(y.ct); pr.y_deg = static_cast<int>(y.ct->GetNoiseScaleDeg());
    return y;
}

}  // namespace

TEST(ComplexUnpackFold, FoldEqualsBaselineAndProd) {
    const int d_in  = 1024;       // the real complex_mlp "up" shape (1024 -> 4096): r_o = 4 (even)
    const int d_out = 4096;
    const int slots = 1 << 15;    // logN=16
    constexpr int L = 17;

    CKKSContextOptions ckks{};
    ckks.bts_iterations       = default_bts_iterations();
    ckks.ckks_complex_payload = true;
    ckks.extra_rot_steps      = cachemir::linear_rot_indices(slots, d_in, d_out);
    Inference inf = make_gpt2_inference({
        .ckks = ckks, .hidDim = d_in, .expDim = d_out, .numHeads = 16,
        .bench_mode = false, .packing_kind = PackingKind::Cachemir,
    });

    const auto p = cachemir::compute_cm_params(slots, d_in, d_out);
    std::cout << "[shape] " << d_in << "x" << d_out
              << "  n_pt=" << p.n_pt << " r_i=" << p.r_i << " r_o=" << p.r_o
              << "  (output-pack: " << p.r_i * (p.r_o / 2) << " complex pts)\n";

    // --- random data + plaintext ground truth ------------------------------------------------
    std::mt19937 gen(11);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::vector<double>> W(d_in, std::vector<double>(d_out));   // W[d_in][d_out]
    std::vector<double> x(d_in);
    for (int i = 0; i < d_in; ++i) {
        x[i] = dist(gen);
        for (int j = 0; j < d_out; ++j) W[i][j] = dist(gen);
    }
    std::vector<double> y_ref(d_out, 0.0);
    for (int j = 0; j < d_out; ++j) {
        double s = 0.0;
        for (int i = 0; i < d_in; ++i) s += x[i] * W[i][j];
        y_ref[j] = s;
    }

    // half-scaled weight for the fold arm: bake the 0.5 into the static plaintext (encode-time).
    std::vector<std::vector<double>> W_half = W;
    for (auto& r : W_half) for (double& v : r) v *= 0.5;

    inf.w["opk"]      = cachemir::encode_weight_matrix_outputpack(inf, W,      d_in, d_out, L);
    inf.w["opk_half"] = cachemir::encode_weight_matrix_outputpack(inf, W_half, d_in, d_out, L);
    inf.complex_weight_names.insert("opk");
    inf.complex_weight_names.insert("opk_half");
    inf.w.erase("opk_bias");
    inf.w.erase("opk_half_bias");

    auto decode = [&](const PackedCtx& y) {
        auto out = cachemir::decode_linear_output(inf.slots, decrypt_slots(inf, y), d_in, d_out);
        out.resize(d_out);
        return out;
    };

    // --- 1. production output-pack (sanity that the harness + W layout are correct) -----------
    PackedCtx y_prod = cachemir::linear_outputpack(inf,
        cachemir::encode_linear_input(inf, x, d_in, d_out, L), "opk", d_in, d_out);
    auto out_prod = decode(y_prod);
    auto s_prod = compare_vec(out_prod, y_ref);

    // --- 2. local baseline (must match production) -------------------------------------------
    LevelProbe pr_base;
    PackedCtx y_base = run_outputpack_local(inf,
        cachemir::encode_linear_input(inf, x, d_in, d_out, L), "opk", d_in, d_out,
        /*folded=*/false, pr_base);
    auto out_base = decode(y_base);
    auto s_base = compare_vec(out_base, y_ref);

    // --- 3. local FOLD: 0.5*W weight + folded unpack (the candidate) --------------------------
    LevelProbe pr_fold;
    PackedCtx y_fold = run_outputpack_local(inf,
        cachemir::encode_linear_input(inf, x, d_in, d_out, L), "opk_half", d_in, d_out,
        /*folded=*/true, pr_fold);
    auto out_fold = decode(y_fold);
    auto s_fold = compare_vec(out_fold, y_ref);

    // fold-vs-prod bit-identity (both bts-free at L17, exact algebra -> expect ~decrypt-floor)
    auto s_identity = compare_vec(out_fold, out_prod);

    std::cout << std::scientific << std::setprecision(3)
              << "\n[vs GT]   prod  max_abs=" << s_prod.max_abs << " max_rel=" << s_prod.max_rel
              <<            "  mean_rel=" << s_prod.mean_rel << "\n"
              << "[vs GT]   base  max_abs=" << s_base.max_abs << " max_rel=" << s_base.max_rel
              <<            "  mean_rel=" << s_base.mean_rel << "\n"
              << "[vs GT]   fold  max_abs=" << s_fold.max_abs << " max_rel=" << s_fold.max_rel
              <<            "  mean_rel=" << s_fold.mean_rel << "\n"
              << "[fold==prod] max_abs=" << s_identity.max_abs << " max_rel=" << s_identity.max_rel
              <<            "  mean_rel=" << s_identity.mean_rel << "\n";

    auto lp = [](const char* tag, const LevelProbe& q) {
        std::cout << "[level " << tag << "]"
                  << "  P(lvl=" << q.P_lvl << ",deg=" << q.P_deg << ")"
                  << "  Re(lvl=" << q.re_lvl << ",deg=" << q.re_deg << ")"
                  << "  Im(lvl=" << q.im_lvl << ",deg=" << q.im_deg << ")"
                  << "  y(lvl=" << q.y_lvl << ",deg=" << q.y_deg << ")\n";
    };
    std::cout << "\n";
    lp("base", pr_base);
    lp("fold", pr_fold);

    const bool ri_desync_base = (pr_base.re_lvl != pr_base.im_lvl) || (pr_base.re_deg != pr_base.im_deg);
    const bool ri_desync_fold = (pr_fold.re_lvl != pr_fold.im_lvl) || (pr_fold.re_deg != pr_fold.im_deg);
    const bool y_shifted      = (pr_base.y_lvl != pr_fold.y_lvl) || (pr_base.y_deg != pr_fold.y_deg);
    std::cout << "\n[VERDICT]"
              << "  Re/Im desync  base=" << (ri_desync_base ? "YES" : "no")
              <<                 " fold=" << (ri_desync_fold ? "YES" : "no")
              << "  | final-y base-vs-fold " << (y_shifted ? "SHIFTED (re-plan needed)" : "IDENTICAL (free)")
              << "\n";

    EXPECT_LT(s_prod.mean_rel, 1e-3) << "production output-pack wrong (harness/keys/layout)";
    EXPECT_LT(s_base.mean_rel, 1e-3) << "local baseline diverged from GT (local copy is unfaithful)";
    EXPECT_LT(s_fold.mean_rel, 1e-3) << "FOLD diverged from GT -- fold is numerically WRONG";
    EXPECT_LT(s_identity.max_rel, 1e-3) << "FOLD != production output (not bit-identical)";
}
