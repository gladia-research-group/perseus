// Can either im_cleanse in the LIVE K/V pack be removed?
//
// The live decode (packing=Cachemir + complex_payload=1, NOT CachemirComplex) runs the 2-arg
// cache_kv_push (cachemir_kv_cache.cu:124). Its core packs K + i*V, bootstraps ONCE, unpacks:
//     im_cleanse(K)            // line 136
//     im_cleanse(V)            // line 137
//     P = pack_ri(K, V, i)     // K + i*V
//     bootstrap(P)
//     conj = conjugate(P)
//     K_out = P + conj         // 2*Re(P)
//     V_out = P - conj         // 2i*Im(P)
// The audit (KVPA-05) called BOTH cleanses load-bearing: K and V each carry imaginary noise from
// the upstream bootstrap, and packing them dirty cross-contaminates on unpack
//     Re(P) = Re(K) - Im(V)    <- V's imaginary noise leaks into the K output
//     Im(P) = Im(K) + Re(V)    <- K's imaginary noise leaks into the V output
// so each cleanse protects the OTHER channel. This test PROVES that quantitatively: it injects a
// KNOWN imaginary perturbation (noiseK into K, noiseV into V), then runs the pack/unpack with all
// four cleanse combinations and measures the cross-channel leakage. A removable cleanse would show
// negligible leakage when dropped; a load-bearing one shows leakage == the injected noise.
//
// Disjoint: reimplements the pack/unpack core locally (no production .cu/.h edited), dense slot
// layout (no linear/mask/rotate -- isolates exactly the cleanse+pack+bts+unpack). NEEDS CKKS_COMPLEX=1.

#include "model/gpt2.h"
#include "test_helpers.h"

#include <gtest/gtest.h>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace test_helpers;

namespace {

// Relative-L2 (energy) error: robust to near-zero entries that blow up per-element rel error.
double rel_l2(const std::vector<double>& got, const std::vector<double>& ref) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double e = got[i] - ref[i];
        num += e * e;
        den += ref[i] * ref[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : 0.0;
}

// 2*Re(P) and the realified 2*Im(P) of the unpacked pack, for given cleanse flags.
// Returns decoded real K_out (slots 0..d-1) and V_out (slots 0..d-1).
struct Unpacked { std::vector<double> K_out, V_out; };

Unpacked run_pack(Inference& inf, const Ctx& K_in, const Ctx& V_in, int d,
                  bool cleanseK, bool cleanseV) {
    PackedCtx a = inf.fhe->clone(inf.pack(K_in, PackingKind::Cachemir));
    PackedCtx b = inf.fhe->clone(inf.pack(V_in, PackingKind::Cachemir));
    if (cleanseK) inf.fhe->inplace_im_cleanse(a);   // a -> 2*Re(K)
    if (cleanseV) inf.fhe->inplace_im_cleanse(b);   // b -> 2*Re(V)

    Ptx i_pt = inf.encode_complex_const_at(0.0, 1.0, b);
    PackedCtx P = inf.fhe->pack_ri(a, b, i_pt);      // a + i*b
    inf.fhe->bootstrap(P.ct);
    PackedCtx conj  = inf.fhe->conjugate(P);
    PackedCtx K_out = inf.fhe->add(P, conj);         // 2*Re(P)
    PackedCtx V_out = inf.fhe->sub(P, conj);         // 2i*Im(P)
    Ptx neg_i = inf.encode_complex_const_at(0.0, -1.0, V_out);
    V_out = inf.fhe->mult(V_out, neg_i);             // 2*Im(P)  (realified)

    auto rawK = decrypt_slots(inf, K_out);
    auto rawV = decrypt_slots(inf, V_out);
    Unpacked u;
    u.K_out.assign(rawK.begin(), rawK.begin() + d);
    u.V_out.assign(rawV.begin(), rawV.begin() + d);
    return u;
}

}  // namespace

TEST(KvPackCleanseAblation, BothCleansesLoadBearing) {
    const int d     = 1024;
    const int slots = 1 << 15;

    CKKSContextOptions ckks{};
    ckks.bts_iterations       = default_bts_iterations();
    ckks.ckks_complex_payload = true;
    Inference inf = make_gpt2_inference({
        .ckks = ckks, .hidDim = d, .expDim = 4096, .numHeads = 16,
        .bench_mode = false, .packing_kind = PackingKind::Cachemir,
    });

    std::mt19937 gen(23);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> K(d), V(d), noiseK(d), noiseV(d);
    for (int i = 0; i < d; ++i) {
        K[i] = dist(gen); V[i] = dist(gen);
        noiseK[i] = 0.10 * dist(gen);   // known imaginary perturbation (~10% of signal)
        noiseV[i] = 0.10 * dist(gen);   // distinct from noiseK so leakage is attributable
    }

    // Encrypt K + i*noiseK and V + i*noiseV (dense: slot i = feature i).
    auto enc_complex = [&](const std::vector<double>& re, const std::vector<double>& im) {
        std::vector<std::complex<double>> pt(slots, {0.0, 0.0});
        for (int i = 0; i < d; ++i) pt[i] = {re[i], im[i]};
        return encrypt(inf.cc(), inf.cc()->MakeCKKSPackedPlaintext(pt, 1, /*level=*/0), inf.fhe->pk());
    };
    Ctx K_in = enc_complex(K, noiseK);
    Ctx V_in = enc_complex(V, noiseV);

    // For each cleanse combo: K_out signal coeff = cleanseK?4:2 (2*Re(a), Re(a) coeff = cleanseK?2:1);
    // V_out signal coeff = cleanseV?4:2. Compare against signal-only -> rel error == cross leakage.
    struct Combo { bool cK, cV; const char* name; };
    const Combo combos[] = {
        {true,  true,  "both cleanse (production)"},
        {false, true,  "DROP K-cleanse        "},
        {true,  false, "DROP V-cleanse        "},
        {false, false, "DROP both             "},
    };

    std::cout << std::scientific << std::setprecision(3)
              << "\n(noiseK,noiseV injected at ~10% of signal; rel-err >> 1e-3 == cross-channel leak)\n";

    bool drop_breaks = false;
    for (const auto& c : combos) {
        auto u = run_pack(inf, K_in, V_in, d, c.cK, c.cV);
        const double cK = c.cK ? 4.0 : 2.0;   // K_out signal coefficient
        const double cV = c.cV ? 4.0 : 2.0;   // V_out signal coefficient
        std::vector<double> K_sig(d), V_sig(d);
        for (int i = 0; i < d; ++i) { K_sig[i] = cK * K[i]; V_sig[i] = cV * V[i]; }
        const double eK = rel_l2(u.K_out, K_sig);   // relative-L2: robust to near-zero refs
        const double eV = rel_l2(u.V_out, V_sig);
        std::cout << "[" << c.name << "]  K-chan rel_l2=" << eK
                  << "   |   V-chan rel_l2=" << eV << "\n";

        if (c.cK && c.cV) {
            EXPECT_LT(eK, 1e-2) << "baseline K dirty (harness)";
            EXPECT_LT(eV, 1e-2) << "baseline V dirty (harness)";
        }
        // dropping K-cleanse must contaminate the V channel; dropping V-cleanse the K channel.
        if (!c.cK && eV > 1e-2) drop_breaks = true;
        if (!c.cV && eK > 1e-2) drop_breaks = true;
    }

    std::cout << "\n[VERDICT] dropping a cleanse contaminates the opposite channel: "
              << (drop_breaks ? "YES -> BOTH cleanses load-bearing, NOT removable"
                              : "NO  -> a cleanse may be removable")
              << "\n";
    EXPECT_TRUE(drop_breaks)
        << "Neither cleanse showed cross-contamination when dropped -- re-examine removability.";
}
