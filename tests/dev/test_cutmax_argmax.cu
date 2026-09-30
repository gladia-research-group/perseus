// Isolated HE validation of CutMax: encrypted argmax over the 2-tile GPT-2
// logit layout, validated against the
// T=128 head oracle. No model, no weights: oracle logits are encoded directly
// in the lm_head tile layout (interleave permutation, zero pads) at a
// tile-like level, then cutmax_argmax() runs the full pipeline op.
//
// The shift-point bootstraps need ~2e-4 absolute noise: submit with
// BTS_ITERATIONS=2. At BTS_ITERATIONS=1 the
// oracle predicts ~119-126/128 with runner-up flips on near-ties.
//
// Env:
//   CUTMAX_POSITIONS  csv of oracle positions   (default "0,1,5,83,50,96,97,105")
//   CUTMAX_SRC_LEVEL  tile encode level          (default 18)
//   STEPS_T           oracle horizon             (default 128)

#include "cutmax.h"
#include "config_loader.h"

#include "all_blocks_test_helpers.h"
#include "ckks_fixture.h"
#include "test_helpers.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace {

using test_helpers::CkksFixture;

std::vector<int> positions_from_env() {
    const char* v = std::getenv("CUTMAX_POSITIONS");
    std::string s = (v && *v) ? v : "0,1,5,83,50,96,97,105";
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(std::stoi(tok));
    return out;
}

int env_int(const char* k, int d) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::atoi(v) : d;
}

}  // namespace

TEST_F(CkksFixture, CutMaxArgmaxOracle) {
    const int W_tile = slots();
    const int src_level = env_int("CUTMAX_SRC_LEVEL", 18);
    const int steps_t = env_int("STEPS_T", 128);

    const std::string path = test_helpers::all_blocks_lm_head_steps_path(
        test_helpers::default_all_blocks_io_dir(), steps_t);
    test_helpers::LmHeadSteps oracle;
    try {
        oracle = test_helpers::read_lm_head_steps(path);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "no head oracle at " << path << ": " << e.what();
    }
    const int vocab = oracle.vocab;
    const int K = (vocab + W_tile - 1) / W_tile;

    Inference inf = make_inf(/*hidDim=*/1024, /*dim=*/768);
    // CONFIGS_PATH with a "cutmax" section overrides the baked schedule
    // (calibrate.py cutmax_calibrate=true) — the live gate for new calibs.
    CutMaxConfig cfg = default_gpt2_cutmax_config();
    if (const char* cp = std::getenv("CONFIGS_PATH"); cp && *cp) {
        auto parsed = config_loader::parse_configs_json(
            config_loader::read_file_to_string(cp));
        if (parsed.has_cutmax) cfg = cutmax_config_from_calib(parsed.cutmax);
    }
    const Packing packing{PackingKind::Cachemir, slots(), 1024, 768, 0, 0};

    int hard_fails = 0;
    for (int pos : positions_from_env()) {
        ASSERT_LT(pos, static_cast<int>(oracle.steps.size()));
        const auto& truth = oracle.steps[pos];

        // encode logits in the lm_head tile layout: slot m of tile k holds
        // logit column k*W_tile + interleave(m); pads (last tile) are 0.
        // CUTMAX_PACKED=1 (needs CKKS_COMPLEX=1): ONE complex tile
        // t0 + i*t1, the cachemir_complex lm_head output layout.
        const bool packed = env_int("CUTMAX_PACKED", 0) != 0;
        std::vector<PackedCtx> tiles;
        if (packed) {
            std::vector<std::complex<double>> v(
                static_cast<size_t>(W_tile), {0.0, 0.0});
            for (int m = 0; m < W_tile; ++m) {
                const int col = cutmax_tile_col_of_slot(m, 1024, W_tile);
                const double re = truth.logits[col];
                const double im = (W_tile + col < vocab)
                    ? truth.logits[W_tile + col] : 0.0;
                v[m] = {re, im};
            }
            Ptx pt = encode(fhe().cc, v, src_level);
            tiles.push_back(PackedCtx{encrypt(fhe().cc, pt, fhe().pk()),
                                      packing});
        } else
        for (int k = 0; k < K; ++k) {
            const int wreal = std::min(W_tile, vocab - k * W_tile);
            std::vector<double> v(static_cast<size_t>(W_tile), 0.0);
            for (int m = 0; m < W_tile; ++m) {
                const int col = cutmax_tile_col_of_slot(m, 1024, W_tile);
                if (col < wreal) v[m] = truth.logits[k * W_tile + col];
            }
            Ptx pt = encode(fhe().cc, v, src_level);
            tiles.push_back(PackedCtx{encrypt(fhe().cc, pt, fhe().pk()),
                                      packing});
        }

        const long bts0 = fhe().total_bootstraps;
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<PackedCtx> z_tiles;
        {
            CKKSContext::OpTallyScope _tally(fhe());
            z_tiles = cutmax_argmax(inf, tiles, vocab, cfg);
        }
        const double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        static bool ops_emitted = false;
        if (!ops_emitted) {   // schedule is fixed: counts identical per position
            ops_emitted = true;
            std::string s = "[cutmax_ops]";
            for (const auto& kv : fhe().op_tally)
                s += " " + kv.first + "=" + std::to_string(kv.second);
            std::cout << s << std::endl;
        }

        // decrypt + invert the interleave -> Z over the vocab
        std::vector<double> z(static_cast<size_t>(vocab), 0.0);
        if (packed) {
            auto pt = decrypt_pt(fhe().cc, z_tiles[0].ct, fhe().sk());
            auto cv = pt->GetCKKSPackedValue();
            for (int m = 0; m < W_tile; ++m) {
                const int col = cutmax_tile_col_of_slot(m, 1024, W_tile);
                z[col] = cv[m].real();
                if (W_tile + col < vocab) z[W_tile + col] = cv[m].imag();
            }
        } else
        for (int k = 0; k < K; ++k) {
            std::vector<double> dec =
                test_helpers::decrypt_slots(inf, z_tiles[k]);
            const int wreal = std::min(W_tile, vocab - k * W_tile);
            for (int m = 0; m < W_tile; ++m) {
                const int col = cutmax_tile_col_of_slot(m, 1024, W_tile);
                if (col < wreal) z[k * W_tile + col] = dec[m];
            }
        }

        int am = 0;
        for (int j = 1; j < vocab; ++j)
            if (z[j] > z[am]) am = j;
        const double top_mass = z[truth.argmax];

        // near-tie context: top-2 gap of the true logits
        std::vector<double> lg = truth.logits;
        std::nth_element(lg.begin(), lg.begin() + 1, lg.end(),
                         std::greater<double>());
        const double gap = lg[0] - lg[1];

        std::printf(
            "[cutmax] pos=%3d argmax=%6d truth=%6d %s gap=%.4f "
            "top_mass=%.4f bts=%ld %.1fs\n",
            pos, am, truth.argmax, am == truth.argmax ? "OK  " : "MISS",
            gap, top_mass, fhe().total_bootstraps - bts0, dt);
        std::fflush(stdout);

        if (am == truth.argmax) {
            EXPECT_GT(top_mass, 0.9)
                << "pos " << pos << ": argmax right but Z not one-hot";
        } else if (gap >= 0.05) {
            ++hard_fails;
            ADD_FAILURE() << "pos " << pos << ": argmax " << am << " != "
                          << truth.argmax << " (gap " << gap
                          << " is not a near-tie)";
        }
    }
    EXPECT_EQ(hard_fails, 0);
}
