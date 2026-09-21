// Isolated correctness gate for ROTATION-KEY LIMB-BAND PRUNING (FIDESLIB_ROT_KEY_BAND).
//
// The lever (sweep candidate "rotkey-limb-band-pruning"): model rotation keys are allocated
// full-chain (~28 towers, ~224 MB each, 163 keys => ~36.5 GB device) but decode ciphertexts
// only ever live in the L16->L24 band. A banded key (key_q_band=B, see FIDESlib
// KeySwitchingKey::Initialize / LimbPartition::generateAllDigitLimb) allocates only the
// Q-limb suffix it needs; a key-switch ABOVE the band THROWS (dotKSK guard,
// LimbPartition.cu:1175). The risk is a wrong row WITHIN the band -> silent decrypt
// corruption, so a decrypt gate is required before defaulting it on.
//
// This test PROVES, disjointly and touching NO production code, that with the band applied
// (via the FIDESLIB_ROT_KEY_BAND env, read in LoadContext) model rotations stay CORRECT
// across the decode level band. It builds exactly ONE context per process (band from env,
// the full 28-tower PRODUCTION chain so decode levels 16..24 exist and hybrid keyswitch
// dnum=7 divides the towers), so the SWEEP SCRIPT runs it once per band value as separate
// processes (scripts/13_rotkey_band_sweep.sh) -- no multi-context-per-process risk,
// matching the sbatch --export sweep convention.
//
// Each banded rotation is checked against the PLAINTEXT cyclic shift (stronger than
// banded==full: it confirms the rotation is actually correct). A band that is too small
// either throws (over-band guard) or diverges by O(1); a correct band matches to ~1e-5.
// PASS  => this band is safe across the decode levels (a candidate default).
// FAIL  => over-band/divergent at some decode level (band too small for this chain).
//
// Run: scripts/13_rotkey_band_sweep.sh   (sbatch; builds once, runs per band, 1 GPU, debug QOS)

#include "test_helpers.h"   // make_ckks_context, default_ckks_options, CKKSContextOptions,
                            // CKKSContext, Ctx, Ptx, encode, encrypt, decrypt_slots, env_or

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

// max |rotated - plaintext cyclic shift| over all slots, sign-agnostic (EvalRotate sign
// convention varies; accept whichever direction matches, like test_rotation_hoisting).
double rot_err_vs_plain(const std::vector<double>& got, const std::vector<double>& x,
                        int step, int slots) {
    double dl = 0.0, dr = 0.0;
    for (int s = 0; s < slots; ++s) {
        dl = std::max(dl, std::fabs(got[s] - x[(s + step) % slots]));
        dr = std::max(dr, std::fabs(got[s] - x[((s - step) % slots + slots) % slots]));
    }
    return std::min(dl, dr);
}

}  // namespace

TEST(RotKeyBand, BandedRotationsCorrectAcrossDecodeLevels) {
    const int logN   = std::atoi(env_or("LOGN", "16").c_str());
    const int slots  = (1 << logN) / 2;
    const int stride = std::max(1, slots / 64);   // tH at logN16 -> the P*V lane stride

    // Representative MODEL rotation steps (P*V lane fan-out is the dominant rotation user).
    // Band correctness is per-key-independent, so a handful proves the mechanism while
    // keeping device-key memory small.
    const std::vector<int> step_muls = {1, 8, 16, 24, 32, 48, 63};
    std::vector<int32_t> steps;
    for (int m : step_muls) steps.push_back(static_cast<int32_t>(stride) * m);

    const int lvl_min  = std::atoi(env_or("LEVEL_MIN", "16").c_str());   // decode level band
    const int lvl_max  = std::atoi(env_or("LEVEL_MAX", "24").c_str());
    const double tol   = std::atof(env_or("ROT_TOL", "1e-3").c_str());   // correct~1e-5, broken~O(1)
    const char* band_e = std::getenv("FIDESLIB_ROT_KEY_BAND");
    const int band     = (band_e && *band_e) ? std::atoi(band_e) : -1;   // -1 => full keys (baseline)

    // Build ONE context; FIDESLIB_ROT_KEY_BAND is read inside LoadContext().
    // MUST use the full PRODUCTION chain (enable_bootstrap=true => depth+btp_overhead = 28
    // towers): (a) decode levels 16..24 only exist in the 28-tower chain, and (b) hybrid
    // key switching needs num_large_digits (7) to divide the tower count (28/7=4) -- a
    // bootstrap-less 12-tower chain throws "can't distribute 12 towers into 7 digits".
    // The script sets SCALE_BITS=53 so the 28-tower chain fits the 128-bit budget.
    CKKSContextOptions o       = default_ckks_options();   // honours the decode chain env knobs
    o.extra_rot_steps          = steps;                    // only our probe keys (+ bootstrap); no default set
    auto ctx = make_ckks_context(o);
    ASSERT_EQ(static_cast<int>(ctx->cc->GetRingDimension() / 2), slots) << "slots mismatch (LOGN env?)";

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<double> x(slots);
    for (double& v : x) v = U(rng);

    std::printf("\n[rotkey-band] FIDESLIB_ROT_KEY_BAND=%s (band=%d, -1=full)  logN=%d slots=%d "
                "stride(tH)=%d keys=%zu  decode levels=%d..%d tol=%.1e\n",
                band_e ? band_e : "(unset)", band, logN, slots, stride, steps.size(),
                lvl_min, lvl_max, tol);
    std::printf("  per level: max|banded_rotate - plaintext_shift| over probe steps; THR=over-band guard threw\n\n");
    std::printf("  %-8s %14s   %s\n", "level", "max_err", "status");
    std::printf("  %s\n", std::string(36, '-').c_str());

    bool all_ok = true;
    for (int L = lvl_min; L <= lvl_max; ++L) {
        double cell = 0.0;
        bool   threw = false;
        std::string msg;
        try {
            Ctx ct = encrypt(ctx->cc, encode(ctx->cc, x, L), ctx->pk());
            for (int step : steps) {
                Ctx r = ctx->cc->EvalRotate(ct, step);
                cell  = std::max(cell, rot_err_vs_plain(decrypt_slots(*ctx, r), x, step, slots));
            }
        } catch (const std::exception& e) {
            threw = true; msg = e.what();
        }
        if (threw) {
            std::printf("  %-8d %14s   THR  (%s)\n", L, "-", msg.c_str());
            all_ok = false;
        } else {
            const bool ok = cell < tol;
            std::printf("  %-8d %14.3e   %s\n", L, cell, ok ? "ok" : "DIVERGED");
            if (!ok) all_ok = false;
        }
        std::fflush(stdout);
    }

    std::printf("\n[rotkey-band] band=%d across decode levels [%d,%d]: %s\n",
                band, lvl_min, lvl_max, all_ok ? "SAFE (candidate default)" : "UNSAFE");

    EXPECT_TRUE(all_ok)
        << "band=" << band << " is unsafe: a decode-band rotation threw (over-band) or diverged "
        << "(>=" << tol << ") vs the plaintext shift. Use a larger band (the sweep script tries a grid).";
}
