#include "ckks_fixture.h"
#include "test_helpers.h"

#include <gtest/gtest.h>
#include <iomanip>
#include <cuda_runtime.h>

using namespace test_helpers;

namespace {

using FideslibWrapperTest = CkksFixture;

TEST_F(FideslibWrapperTest, RoundTrip) {
    auto diag_val = [&](double val, const char* label) {
        Ptx pt = encode(fhe().cc, std::vector<double>(slots(), val), 0);
        std::cerr << "[diag-pt] " << label << " val=" << val
                  << " level=" << pt->GetLevel() << std::endl;
        Ctx ct = encrypt(fhe().cc, pt, fhe().pk());
        std::vector<double> got = decrypt_slots(fhe(), ct);
        std::cerr << "[diag-dec] " << label << " got[0]="
                  << std::fixed << std::setprecision(12) << got[0]
                  << " got[1]=" << got[1]
                  << " got[last]=" << got.back() << std::endl;
    };
    diag_val(0.0,  "zero");
    diag_val(-1.0, "neg1");
    diag_val(3.14, "pi");
    EXPECT_NEAR(decrypt_slots(fhe(), enc_const(3.14))[0], 3.14, 1e-2);  // n32: 27-bit scale => ~9-bit noise (measured ~4e-3); 1e-3 was the 64-bit chain tolerance
    EXPECT_NEAR(decrypt_slots(fhe(), enc_const(0.0))[0],  0.0,  1e-2);
}

TEST_F(FideslibWrapperTest, MultDiagLevel0) {
    std::cerr << "[mult_diag] === Level-0 mult diagnostics ===" << std::endl;
    std::cerr << "[mult_diag] total_depth=" << fhe().total_depth
              << " level_limit=" << fhe().level_limit()
              << " bootstrap_output=" << fhe().bootstrap_output_level() << std::endl;

    Ctx a = enc_const(2.5);
    Ctx b = enc_const(-0.4);
    std::cerr << "[mult_diag] level_of(a)=" << level_of(a)
              << " level_of(b)=" << level_of(b) << std::endl;

    auto da = decrypt_slots(fhe(), a);
    auto db = decrypt_slots(fhe(), b);
    std::cerr << "[mult_diag] a[0]=" << da[0] << " b[0]=" << db[0] << std::endl;

    std::cerr << "[mult_diag] --- add (sanity) ---" << std::endl;
    {
        cudaDeviceSynchronize();
        Ctx s = fhe().add(a, b);
        cudaDeviceSynchronize();
        auto ds = decrypt_slots(fhe(), s);
        std::cerr << "[mult_diag] add: s[0]=" << ds[0] << " (expect 2.1)" << std::endl;
    }

    std::cerr << "[mult_diag] --- scalar mult a*1.0 ---" << std::endl;
    {
        cudaDeviceSynchronize();
        Ctx sc = fhe().mult(a, 1.0);
        cudaDeviceSynchronize();
        std::cerr << "[mult_diag] level_of(sc)=" << level_of(sc) << std::endl;
        auto dsc = decrypt_slots(fhe(), sc);
        std::cerr << "[mult_diag] sc[0]=" << dsc[0] << " (expect 2.5)" << std::endl;
        std::cerr << "[mult_diag] sc[1]=" << dsc[1] << std::endl;
    }

    std::cerr << "[mult_diag] --- CPU OpenFHE ct*ct mult (no GPU) ---" << std::endl;
    {
        // Same context/keys, pure lbcrypto path: splits OpenFHE-32 keyswitch (params, KSK,
        // n32 overlay) from the FIDESlib GPU keyswitch. If this is also garbage, the bug is
        // NOT in the GPU kernels.
        auto& occ = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(fhe().cc->cpu);
        auto& opk = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(fhe().pk()->pimpl);
        auto& osk = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(fhe().sk()->pimpl);
        auto pa = occ->MakeCKKSPackedPlaintext(std::vector<double>(slots(), 2.5));
        auto pb = occ->MakeCKKSPackedPlaintext(std::vector<double>(slots(), -0.4));
        auto ca = occ->Encrypt(opk, pa);
        auto cb = occ->Encrypt(opk, pb);
        auto cp = occ->EvalMult(ca, cb);
        lbcrypto::Plaintext out;
        occ->Decrypt(osk, cp, &out);
        out->SetLength(4);
        std::cerr << "[mult_diag] CPU mult p[0]=" << out->GetCKKSPackedValue()[0].real()
                  << " p[1]=" << out->GetCKKSPackedValue()[1].real() << " (expect -1.0)" << std::endl;

        // --- bit-exact GPU-vs-CPU on IDENTICAL inputs (deterministic given same cts+keys) ---
        // Any diverging limb pinpoints the broken GPU stage with no transform-convention
        // ambiguity. EvalAdd is the methodology control (must match bit-exact).
        auto cpu_add  = occ->EvalAdd(ca, cb);
        auto cpu_full = occ->Relinearize(occ->EvalMultNoRelin(ca, cb));

        auto inject = [&](const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& c) {
            auto ccc = fhe().cc;
            Ctx g = std::make_shared<CiphertextImpl<DCRTPoly>>(std::move(ccc));
            g->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(c->Clone());
            fhe().cc->LoadCiphertext(g);
            return g;
        };
        auto cmp = [&](const char* tag, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& cpu_ct, Ctx gpu_ct) {
            try {
                (void)decrypt_slots(fhe(), gpu_ct);  // forces download into gpu_ct->cpu
            } catch (const std::exception& e) {
                // Decode noise-check may throw on a wrong ct; the download into ->cpu already
                // happened, so the limb comparison below is still valid.
                std::cerr << "[bitcmp] " << tag << " decrypt threw: " << e.what() << std::endl;
            }
            auto& g = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(gpu_ct->cpu);
            auto& gel = g->GetElements();
            auto& cel = cpu_ct->GetElements();
            std::cerr << "[bitcmp] " << tag << " comps gpu=" << gel.size() << " cpu=" << cel.size()
                      << std::endl;
            for (size_t comp = 0; comp < std::min(gel.size(), cel.size()); ++comp) {
                auto ge = gel[comp].GetAllElements();
                auto ce = cel[comp].GetAllElements();
                std::cerr << "[bitcmp] " << tag << " comp" << comp << " limbs gpu=" << ge.size()
                          << " cpu=" << ce.size() << " badcounts:";
                for (size_t l = 0; l < std::min(ge.size(), ce.size()); ++l) {
                    auto& gv = ge[l].GetValues();
                    auto& cv = ce[l].GetValues();
                    size_t bad = 0, n = std::min((size_t)gv.GetLength(), (size_t)cv.GetLength());
                    for (size_t i = 0; i < n; ++i)
                        bad += (gv[i] != cv[i]);
                    std::cerr << " l" << l << ":" << bad;
                }
                std::cerr << std::endl;
            }
        };
        Ctx ga = inject(ca), gb = inject(cb);
        Ctx gadd = fhe().add(ga, gb);
        cmp("add", cpu_add, gadd);
        // per-index rotation VALUE check (bit-compare is meaningless here: the hoisted rotation
        // is a different-but-valid algorithm vs OpenFHE EvalRotate). Ramp payload exposes
        // slot-mapping errors; rotadd_tree fails index-dependently while rotate_pair passes.
        {
            std::vector<double> ramp(slots());
            for (size_t i = 0; i < ramp.size(); ++i)
                ramp[i] = ((i * 37) % 1000) / 1000.0;
            auto pr = occ->MakeCKKSPackedPlaintext(ramp);
            auto cr = occ->Encrypt(opk, pr);
            Ctx grr = inject(cr);
            // BASELINES: is the heavy per-slot tail the chain's own noise envelope?
            {
                auto rt = decrypt_slots(fhe(), grr);  // plain roundtrip, no ops
                double e = 0;
                for (size_t i = 0; i < rt.size(); ++i)
                    e = std::max(e, std::fabs(rt[i] - ramp[i]));
                std::cerr << "[rotval] baseline-roundtrip err_max=" << e << std::endl;
            }
            try {
                auto cpu_rot = occ->EvalRotate(cr, 16);  // pure CPU rotation
                lbcrypto::Plaintext prt;
                occ->Decrypt(osk, cpu_rot, &prt);
                auto& vv = prt->GetCKKSPackedValue();
                double e = 0;
                for (size_t i = 0; i < vv.size(); ++i)
                    e = std::max(e, std::fabs(vv[i].real() - ramp[(i + 16) % ramp.size()]));
                std::cerr << "[rotval] cpu-rot16 err_max=" << e << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[rotval] cpu-rot16 THREW: " << e.what() << std::endl;
            }
            for (int k : {1, 2, 16, 256, 512}) {
                try {
                    Ctx g = fhe().rotate(grr, k);
                    auto got = decrypt_slots(fhe(), g);
                    double e = 0;
                    for (size_t i = 0; i < got.size(); ++i)
                        e = std::max(e, std::fabs(got[i] - ramp[(i + k) % ramp.size()]));
                    std::cerr << "[rotval] k=" << k << " err_max=" << e << std::endl;
                    if (k == 1 && std::getenv("FHE_KS_TRACE_FULL")) {
                        std::cerr << "[rotfull] k=1 vals:";
                        for (double v : got)
                            std::cerr << " " << std::setprecision(6) << v;
                        std::cerr << std::endl;
                        // post-moddown final ct limbs (downloaded by the decrypt above)
                        auto& gg = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(g->cpu);
                        for (int comp = 0; comp < 2; ++comp) {
                            auto& e = gg->GetElements()[comp].GetAllElements()[0];
                            std::cerr << "[rotfinal] comp" << comp
                                      << " p=" << e.GetModulus().ConvertToInt<uint64_t>() << " vals:";
                            for (size_t i2 = 0; i2 < e.GetValues().GetLength(); ++i2)
                                std::cerr << " " << e.GetValues()[i2].ConvertToInt<uint64_t>();
                            std::cerr << std::endl;
                        }
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[rotval] k=" << k << " THREW: " << e.what() << std::endl;
                }
            }
            // and the tree pattern itself: s += rot(s, st) — catches sequence/state effects
            try {
                Ctx s = grr;
                for (int st = 1; st <= 4; st *= 2) {
                    Ctx r = fhe().rotate(s, st);
                    fhe().inplace_add(s, r);
                }
                auto got = decrypt_slots(fhe(), s);
                double e = 0;
                for (size_t i = 0; i < got.size(); ++i) {
                    double ref = 0;
                    for (int t = 0; t < 8; ++t)
                        ref += ramp[(i + t) % ramp.size()];
                    e = std::max(e, std::fabs(got[i] - ref));
                }
                std::cerr << "[rotval] tree8 err_max=" << e << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[rotval] tree8 THREW: " << e.what() << std::endl;
            }
        }
        // rescale validated via a second-level mult: under FLEXIBLEAUTO both sides rescale the
        // deg-2 input internally before the mult+keyswitch (explicit CPU Rescale is a no-op,
        // so a direct Rescale-vs-Rescale compare is meaningless).
        Ctx gmult_rs = fhe().mult(ga, gb);
        auto cpu_m2 = occ->EvalMult(cpu_full, ca);
        Ctx gm2 = fhe().mult(gmult_rs, ga);
        cmp("mult2-rescaled", cpu_m2, gm2);
        // CPU-side reference for the GPU keyswitch trace (FHE_KS_TRACE=1): c2 of the SAME mult,
        // per limb, in eval (must match the trace's "Input:") and true-coefficient form (must
        // match "Out INTT:" if the U32 INTT basis is right). Positions 0,1 only.
        {
            auto cpu_deg2 = occ->EvalMultNoRelin(ca, cb);
            auto evs = cpu_deg2->GetElements()[2].GetAllElements();
            const size_t half = evs[0].GetValues().GetLength() / 2;  // brev image of position 1
            std::cerr << "[cpuc2] eval:";
            for (auto& e : evs)
                std::cerr << " (" << e.GetModulus().ConvertToInt<uint64_t>() << ", "
                          << e.GetValues()[0].ConvertToInt<uint64_t>() << " "
                          << e.GetValues()[1].ConvertToInt<uint64_t>() << " "
                          << e.GetValues()[half].ConvertToInt<uint64_t>() << ")";
            std::cerr << std::endl << "[cpuc2] coeff:";
            for (auto& e : evs) {
                auto c = e;
                c.SetFormat(COEFFICIENT);
                std::cerr << " (" << c.GetModulus().ConvertToInt<uint64_t>() << ", "
                          << c.GetValues()[0].ConvertToInt<uint64_t>() << " "
                          << c.GetValues()[1].ConvertToInt<uint64_t>() << " "
                          << c.GetValues()[half].ConvertToInt<uint64_t>() << ")";
            }
            std::cerr << std::endl;
            if (std::getenv("FHE_KS_TRACE_FULL")) {
                // full limb-0 coefficient vector + root of unity: the offline analyzer derives
                // the U32 INTT's output permutation/twist from this.
                auto c = evs[0];
                c.SetFormat(COEFFICIENT);
                std::cerr << "[cpufull] p=" << c.GetModulus().ConvertToInt<uint64_t>()
                          << " root=" << c.GetRootOfUnity().ConvertToInt<uint64_t>() << " vals:";
                for (size_t i = 0; i < c.GetValues().GetLength(); ++i)
                    std::cerr << " " << c.GetValues()[i].ConvertToInt<uint64_t>();
                std::cerr << std::endl;
                // matching full EVAL vector: the (coeff, eval) pair calibrates OpenFHE's exact
                // negacyclic FTT convention for the offline forward-NTT verifier.
                std::cerr << "[cpufull-eval] p=" << evs[0].GetModulus().ConvertToInt<uint64_t>() << " vals:";
                for (size_t i = 0; i < evs[0].GetValues().GetLength(); ++i)
                    std::cerr << " " << evs[0].GetValues()[i].ConvertToInt<uint64_t>();
                std::cerr << std::endl;
            }
        }
        cmp("mult", cpu_full, fhe().mult(ga, gb));
    }

    std::cerr << "[mult_diag] --- ct*ct mult a*b ---" << std::endl;
    {
        cudaDeviceSynchronize();
        Ctx p = fhe().mult(a, b);
        cudaDeviceSynchronize();
        std::cerr << "[mult_diag] level_of(p)=" << level_of(p) << std::endl;
        auto dp = decrypt_slots(fhe(), p);
        std::cerr << "[mult_diag] p[0]=" << dp[0] << " (expect -1.0)" << std::endl;
        std::cerr << "[mult_diag] p[1]=" << dp[1] << std::endl;
        std::cerr << "[mult_diag] p[2]=" << dp[2] << std::endl;
    }
}

TEST_F(FideslibWrapperTest, Arithmetic) {
    Ctx a = enc_const(2.5);
    Ctx b = enc_const(-0.4);
    const uint32_t l0 = level_of(a);
    Ctx p = fhe().mult(a, b);
    auto result = decrypt_slots(fhe(), p);
    std::cerr << "[arith] mult result[0]=" << result[0] << std::endl;
    EXPECT_NEAR(result[0], -1.0, 1e-2);  // n32: 27-bit scale => ~9-bit noise (measured ~4e-3); 1e-3 was the 64-bit chain tolerance
    EXPECT_GE(level_of(p), l0);
    Ctx s = fhe().add(a, b);
    EXPECT_EQ(level_of(s), l0);
}

TEST_F(FideslibWrapperTest, ClonePreservesValueAndLevel) {
    Ctx a = enc_const(1.7);
    Ctx c = fhe().clone(a);
    EXPECT_EQ(level_of(c), level_of(a));
    EXPECT_NEAR(decrypt_slots(fhe(), c)[0], 1.7, 1e-2);  // n32: 27-bit scale => ~9-bit noise (measured ~4e-3); 1e-3 was the 64-bit chain tolerance
}

TEST_F(FideslibWrapperTest, NegationDoesNotConsumeLevel) {
    Ctx a = enc_const(1.2);
    const uint32_t l0 = level_of(a);
    Ctx n = fhe().negate(a);
    EXPECT_EQ(level_of(n), l0);
    EXPECT_NEAR(decrypt_slots(fhe(), n)[0], -1.2, 1e-2);  // n32: 27-bit scale => ~9-bit noise (measured ~4e-3); 1e-3 was the 64-bit chain tolerance
    Ctx b = enc_const(0.8);
    const uint32_t l1 = level_of(b);
    fhe().inplace_negate(b);
    EXPECT_EQ(level_of(b), l1);
    EXPECT_NEAR(decrypt_slots(fhe(), b)[0], -0.8, 1e-2);  // n32: 27-bit scale => ~9-bit noise (measured ~4e-3); 1e-3 was the 64-bit chain tolerance
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
    EXPECT_NEAR(decrypt_slots(fhe(), s)[0], 1.5, 1e-2);  // n32: 27-bit scale => ~9-bit noise (measured ~4e-3); 1e-3 was the 64-bit chain tolerance
}

TEST_F(FideslibWrapperTest, LevelReduceDropsLevelPreservesValue) {
    Ctx a = enc_const(2.5);
    const int l0 = static_cast<int>(level_of(a));
    fhe().drop_to_level(a, l0 + 2);
    EXPECT_EQ(static_cast<int>(level_of(a)), l0 + 2);
    EXPECT_NEAR(decrypt_slots(fhe(), a)[0], 2.5, 1e-2);  // n32: 27-bit scale => ~9-bit noise (measured ~4e-3); 1e-3 was the 64-bit chain tolerance
}

}
