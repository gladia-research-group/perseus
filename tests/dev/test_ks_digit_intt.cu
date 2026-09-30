// Bit-identity gate for FIDESLIB_KS_DIGIT_INTT — the ct x ct keyswitch's redundant per-digit
// source INTT.
//
// The keyswitch INTTs its source polynomial twice: the merged pass over every live limb, and
// then again per digit over the same limbptr slice into the same GATHERptr slots. Skipping the
// second pass must change NOTHING about the result, only the launch count and the DRAM traffic.
// "Must" is not an argument, so this test proves it: the SAME ciphertext is run through both
// arms in one process (setKsDigitIntt flips the arm; the env knob is read once) and every RNS
// limb of every component is compared exactly, with a same-arm control run proving the path is
// deterministic in the first place.
//
// Two traps this test exists to avoid — both cost a measurement cycle when this was written:
//   * re-encrypting per arm compares fresh encryption NOISE, so every coefficient differs and
//     the comparison means nothing. Encrypt once, outside the arms.
//   * test_composite_bootstrap's err_max is not run-to-run deterministic, so a diff in it is
//     not evidence either way.
//
//   CHAIN=n32 source scripts/local_env.sh
//   bash scripts/local_build_devtest.sh test_ks_digit_intt
//   CUDA_VISIBLE_DEVICES=3 build_n32/bin/test_ks_digit_intt

#include "ckks_fixture.h"

#include "CKKS/Ciphertext.cuh"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

// Every RNS limb of every component of a ciphertext, downloaded from the GPU.
using LimbSnapshot = std::vector<std::vector<std::vector<uint64_t>>>;

LimbSnapshot snapshot(CKKSContext& fhe, const Ctx& ct) {
    (void)decrypt_slots(fhe, ct);  // forces the GPU->host download into ct->cpu
    auto& g = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
    auto& els = g->GetElements();
    LimbSnapshot out;
    out.reserve(els.size());
    for (size_t comp = 0; comp < els.size(); ++comp) {
        auto e = els[comp].GetAllElements();
        std::vector<std::vector<uint64_t>> limbs;
        limbs.reserve(e.size());
        for (size_t l = 0; l < e.size(); ++l) {
            auto& v = e[l].GetValues();
            std::vector<uint64_t> row((size_t)v.GetLength());
            for (size_t i = 0; i < row.size(); ++i)
                row[i] = (uint64_t)v[i].ConvertToInt();
            limbs.push_back(std::move(row));
        }
        out.push_back(std::move(limbs));
    }
    return out;
}

size_t count_diffs(const LimbSnapshot& a, const LimbSnapshot& b, std::string& where) {
    if (a.size() != b.size()) {
        where = "component count " + std::to_string(a.size()) + " vs " + std::to_string(b.size());
        return 1;
    }
    size_t bad = 0;
    for (size_t c = 0; c < a.size(); ++c) {
        if (a[c].size() != b[c].size()) {
            where = "limb count at comp " + std::to_string(c);
            return 1;
        }
        for (size_t l = 0; l < a[c].size(); ++l) {
            if (a[c][l].size() != b[c][l].size()) {
                where = "length at comp " + std::to_string(c) + " limb " + std::to_string(l);
                return 1;
            }
            for (size_t i = 0; i < a[c][l].size(); ++i) {
                if (a[c][l][i] != b[c][l][i]) {
                    if (bad == 0)
                        where = "comp " + std::to_string(c) + " limb " + std::to_string(l) + " coeff " +
                                std::to_string(i) + ": " + std::to_string(a[c][l][i]) + " vs " +
                                std::to_string(b[c][l][i]);
                    ++bad;
                }
            }
        }
    }
    return bad;
}

class KsDigitIntt : public CkksFixture {};

// mult / square / a two-deep chain: the ct x ct keyswitch shapes EvalMod actually runs.
TEST_F(KsDigitIntt, SkippingTheRedundantSourceInttIsBitIdentical) {
    std::vector<double> a(1024), b(1024);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = 0.29 * std::sin(0.017 * (double)i) + 0.04;
        b[i] = 0.33 * std::cos(0.023 * (double)i);
    }
    // ONE ciphertext for both arms: a fresh encryption carries fresh noise, which would make
    // every coefficient differ regardless of what the keyswitch does.
    Ctx ca = ::encrypt(fhe().cc, ::encode(fhe().cc, a), fhe().pk());
    Ctx cb = ::encrypt(fhe().cc, ::encode(fhe().cc, b), fhe().pk());

    auto run_mult = [&](bool redundant_pass) {
        FIDESlib::CKKS::setKsDigitIntt(redundant_pass);
        return snapshot(fhe(), fhe().mult(ca, cb));
    };
    auto run_sq = [&](bool redundant_pass) {
        FIDESlib::CKKS::setKsDigitIntt(redundant_pass);
        return snapshot(fhe(), fhe().mult(ca, ca));
    };
    auto run_chain = [&](bool redundant_pass) {
        FIDESlib::CKKS::setKsDigitIntt(redundant_pass);
        Ctx m1 = fhe().mult(ca, cb);
        return snapshot(fhe(), fhe().mult(m1, ca));
    };
    auto run_rot = [&](bool redundant_pass) {
        FIDESlib::CKKS::setKsDigitIntt(redundant_pass);
        return snapshot(fhe(), fhe().rotate(ca, 4));
    };

    const bool saved = FIDESlib::CKKS::ksDigitIntt();
    struct Case {
        const char* name;
        std::function<LimbSnapshot(bool)> f;
    };
    std::vector<Case> cases{{"mult", run_mult}, {"square", run_sq}, {"mult-chain", run_chain}, {"rotate", run_rot}};
    size_t total_bad = 0, total_ctl = 0;
    for (auto& c : cases) {
        const auto with = c.f(true);     // upstream: both source INTTs
        const auto control = c.f(true);  // same arm twice: is the path deterministic at all?
        const auto without = c.f(false); // shipped: merged pass only
        std::string w1, w2;
        const size_t ctl = count_diffs(with, control, w1);
        const size_t bad = count_diffs(with, without, w2);
        total_ctl += ctl;
        total_bad += bad;
        std::cerr << "[ksdigitintt] " << c.name << " comps=" << with.size()
                  << " limbs=" << (with.empty() ? 0 : with[0].size()) << " control_diffs=" << ctl
                  << " skip_diffs=" << bad << (bad ? ("  first: " + w2) : std::string{}) << std::endl;
    }
    FIDESlib::CKKS::setKsDigitIntt(saved);
    EXPECT_EQ(total_ctl, 0u) << "the keyswitch is not run-to-run deterministic - the comparison below "
                                "cannot mean anything";
    EXPECT_EQ(total_bad, 0u) << "skipping the redundant per-digit source INTT is NOT bit-identical";
}

}  // namespace
