#include "all_blocks_test_helpers.h"
#include "attention.h"
#include "model/gpt2.h"
#include "packing/cachemir_filling/cachemir_filling_attention.h"
#include "packing/cachemir_filling/cachemir_filling_delta_attention.h"
#include "test_helpers.h"
#include "weight_loader.h"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace test_helpers;

// Isolated gate for the BIDIRECTIONAL filling attention (ViT campaign, plan doc §5.7):
// two-phase driver (phase 1 = QKV linear + K/V push for ALL chunks, phase 2 = per-chunk
// qkt/softmax/softmax_v over the full cache), uniform kc = T. Validated against the exact
// plaintext bidirectional softmax computed from the decrypted q / K / V cache cts.
// CausalControl runs the SHIPPING causal path on the same data with the same metric —
// the honest baseline for the filling packing-parity noise (~0.2-0.4 worst-slot on
// multi-group rows); the bidirectional gates are set relative to it.
// Real ln_1_out rows from the block-0 GT dump serve as inputs (fold envs must be OFF).
// MULTI_T=37 exercises: full chunk attending FORWARD into a partial group, partial
// query chunk attending backward, and the partial-group delta pruning.

namespace {

Inference make_filling_inf() {
    CKKSContextOptions ckks{};
    ckks.bts_iterations = default_bts_iterations();
    return make_gpt2_inference({
        .ckks         = ckks,
        .packing_kind = PackingKind::CachemirFilling,
    });
}

void install_block0(Inference& inf) {
    auto store = load_store(default_weights_path());
    auto names = weight_loader::gpt2_layer_names(0);
    weight_loader::prepare_gpt2_layer_weights(
        inf, store, names,
        /*d_real=*/inf.size.getRealHidDim(),
        /*d_exp_real=*/inf.size.getRealFfDim(),
        /*d_pad=*/inf.size.hidDim,
        /*d_exp_pad=*/inf.size.expDim,
        /*num_heads=*/inf.size.numHeads);
    auto parsed = config_loader::parse_configs_json(
        config_loader::read_file_to_string(default_configs_path()));
    weight_loader::prepare_gpt2_layer_configs(inf, parsed, /*block_idx=*/0);
    // attention-only: drop the MLP/LN/out weights (largest host plaintext consumers)
    for (const char* k : {"up", "up_bias", "down", "down_bias", "out", "out_bias",
                          "ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias"})
        inf.w.erase(k);
}

void release_device(Inference& inf) {
    cudaDeviceSynchronize();
    for (auto& kv : inf.w)
        for (auto& p : kv.second) inf.evict_plaintext(p);
    inf.evict_enc_cache_device();
    cudaDeviceSynchronize();
}

struct Metrics { double max_d = 0, rmse = 0, mean_d = 0, ref_max = 0; int wh = -1, wi = -1; };

// Exact-softmax distance for one attended chunk. Row i attends keys [0, kc(i)):
// kc = base+i+1 (causal) or T (bidirectional). Layout slot[c*tH + h*t + lane].
Metrics dist_to_exact(Inference& inf, const std::vector<double>& raw,
                      const std::vector<double>& q_raw,
                      const std::vector<std::vector<double>>& kr,
                      const std::vector<std::vector<double>>& vr,
                      int n_cur, int base, int T, bool causal) {
    const int t  = inf.slots / inf.size.hidDim;
    const int tH = t * inf.size.numHeads;
    const int H_real = inf.size.getRealNumHeads();
    const int DH     = inf.size.getRealDHead();
    Metrics m;
    double sum_sq = 0, sum_d = 0;
    size_t cnt = 0;
    for (int h = 0; h < H_real; ++h) {
        for (int i = 0; i < n_cur; ++i) {
            const int kc = causal ? base + i + 1 : T;
            std::vector<double> sc(kc);
            for (int j = 0; j < kc; ++j) {
                double s = 0.0;
                for (int c = 0; c < DH; ++c)
                    s += q_raw[c * tH + h * t + i] * kr[j / t][c * tH + h * t + j % t];
                sc[j] = s / std::sqrt(static_cast<double>(DH));
            }
            const double mx = *std::max_element(sc.begin(), sc.end());
            double den = 0.0;
            for (double& s : sc) { s = std::exp(s - mx); den += s; }
            for (int c = 0; c < DH; ++c) {
                double o = 0.0;
                for (int j = 0; j < kc; ++j)
                    o += sc[j] / den * vr[j / t][c * tH + h * t + j % t];
                const double e = std::abs(raw[c * tH + h * t + i] - o);
                if (e > m.max_d) { m.max_d = e; m.wh = h; m.wi = i; }
                sum_sq += e * e;
                sum_d += e;
                ++cnt;
                m.ref_max = std::max(m.ref_max, std::abs(o));
            }
        }
    }
    m.rmse   = std::sqrt(sum_sq / std::max<size_t>(1, cnt));
    m.mean_d = sum_d / std::max<size_t>(1, cnt);
    return m;
}

void report(const std::string& tag, const Metrics& m) {
    std::cout << std::scientific << std::setprecision(3)
              << "[bdattn] " << tag << " dist-to-exact: max|d|=" << m.max_d
              << " (h=" << m.wh << ",i=" << m.wi << ") rmse=" << m.rmse
              << " mean=" << m.mean_d << " |ref|max=" << m.ref_max << "\n";
    ASSERT_TRUE(std::isfinite(m.max_d) && std::isfinite(m.rmse));
}

PackedCtx attend(Inference& inf, PackedCtx& q) {
    inf.fhe->level_hint(q, inf.fhe->level_limit() - 3);
    auto s = cachemir_filling::qkt(inf, q);
    auto p = cachemir_filling::attention_softmax_thor(inf, std::move(s), "attn");
    return cachemir_filling::softmax_v(inf, std::move(p));
}

PackedCtx attend_delta(Inference& inf, PackedCtx& q) {
    inf.fhe->level_hint(q, inf.fhe->level_limit() - 3);
    auto s = cachemir_filling::qkt_delta(inf, q);
    auto p = cachemir_filling::attention_softmax_thor_delta(inf, std::move(s), "attn");
    return cachemir_filling::softmax_v(inf, std::move(p));
}

template <typename F>
double timed_s(F&& f) {
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    f();
    cudaDeviceSynchronize();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

TEST(BidirectionalAttention, TwoPhaseVsExactSoftmax) {
    const int T = std::stoi(env_or("MULTI_T", "37"));

    {
        std::ifstream w(default_weights_path());
        if (!w) GTEST_SKIP() << "weights not available";
    }
    const std::string io_path = all_blocks_io_path(default_all_blocks_io_dir(), 0, 128);
    if (!probe_io_file(io_path)) GTEST_SKIP() << "missing GT: " << io_path;
    auto io = read_block0_io(io_path, "ln_1_out", "attn_out");
    ASSERT_GE(static_cast<int>(io.inp.size()), T);

    Inference inf = make_filling_inf();
    inf.bidirectional = true;
    install_block0(inf);
    const int d = inf.size.hidDim;
    const int t = inf.slots / d;

    prepare_mha_masks(inf);
    prepare_vcache(inf);

    // phase 1: QKV + K/V push for EVERY chunk before any attention
    std::vector<PackedCtx> qs;
    std::vector<int> ns;
    for (int base = 0; base < T; base += t) {
        const int n = std::min(t, T - base);
        inf.n_tok = n;
        std::vector<std::vector<double>> rows(io.inp.begin() + base,
                                              io.inp.begin() + base + n);
        PackedCtx x = encode_prefill_input(inf, rows);
        inf.fhe->bootstrap_hint(x, inf.fhe->level_limit() - 1, true);
        inf.fhe->level_hint(x, inf.fhe->level_limit() - 1);
        auto qkv = linear_multi(inf, x, {"k", "v", "q"}, d, d, /*stream_pt=*/true);
        cache_kv_push(inf, qkv[0], qkv[1]);
        qs.push_back(std::move(qkv[2]));
        ns.push_back(n);
    }
    ASSERT_EQ(inf.k_count(), T);
    const int G = static_cast<int>(qs.size());

    std::vector<std::vector<double>> kr, vr, qr;
    for (const auto& pc : inf.cache[inf.scoped("k")])
        kr.push_back(decrypt(inf.cc(), pc.ct, inf.fhe->sk()));
    for (const auto& pc : inf.cache[inf.scoped("v")])
        vr.push_back(decrypt(inf.cc(), pc.ct, inf.fhe->sk()));
    for (const auto& pc : qs)
        qr.push_back(decrypt(inf.cc(), pc.ct, inf.fhe->sk()));

    // phase 2: per-chunk attention over the full cache
    for (int c = 0; c < G; ++c) {
        const int n_cur = ns[c];
        inf.n_tok = n_cur;
        SCOPED_TRACE("chunk " + std::to_string(c) + " n=" + std::to_string(n_cur));
        std::cout << "\n##### bd chunk " << c << " n=" << n_cur << " T=" << T << " #####\n";

        // schedule sanity: every group, full active delta range
        size_t expected = 0;
        for (int g = 0; g * t < T; ++g) expected += std::min(t, T - g * t) - 1 + n_cur;
        auto sched = cachemir_filling::cf_score_schedule(inf);
        ASSERT_EQ(sched.size(), expected);

        release_device(inf);
        PackedCtx out = attend(inf, qs[c]);
        auto raw = decrypt(inf.cc(), out.ct, inf.fhe->sk());

        Metrics m = dist_to_exact(inf, raw, qr[c], kr, vr, n_cur, c * t, T,
                                  /*causal=*/false);
        report("bd chunk " + std::to_string(c), m);
        // rmse at the causal shipping-path level; worst-slot bounded vs |ref| (the
        // CausalControl below measures the inherited parity noise on the same data —
        // judge bd against its printed levels)
        EXPECT_LT(m.rmse, 0.05);
        EXPECT_LT(m.max_d, 1.0);

        release_device(inf);
        inf.clear_enc_cache();
    }
}

// δ-block bidirectional arm vs the per-entry bd path: exp/GS chain per big ct
// instead of per schedule entry (the dominant bootstrap cut). Comparative gates —
// B no farther from exact than A beyond margin (delta-test convention).
TEST(BidirectionalAttention, DeltaBlockVsPerEntry) {
    const int T = std::stoi(env_or("MULTI_T", "37"));

    {
        std::ifstream w(default_weights_path());
        if (!w) GTEST_SKIP() << "weights not available";
    }
    const std::string io_path = all_blocks_io_path(default_all_blocks_io_dir(), 0, 128);
    if (!probe_io_file(io_path)) GTEST_SKIP() << "missing GT: " << io_path;
    auto io = read_block0_io(io_path, "ln_1_out", "attn_out");
    ASSERT_GE(static_cast<int>(io.inp.size()), T);

    Inference inf = make_filling_inf();
    inf.bidirectional = true;
    install_block0(inf);
    const int d = inf.size.hidDim;
    const int t = inf.slots / d;

    prepare_mha_masks(inf);
    prepare_vcache(inf);

    std::vector<PackedCtx> qs;
    std::vector<int> ns;
    for (int base = 0; base < T; base += t) {
        const int n = std::min(t, T - base);
        inf.n_tok = n;
        std::vector<std::vector<double>> rows(io.inp.begin() + base,
                                              io.inp.begin() + base + n);
        PackedCtx x = encode_prefill_input(inf, rows);
        inf.fhe->bootstrap_hint(x, inf.fhe->level_limit() - 1, true);
        inf.fhe->level_hint(x, inf.fhe->level_limit() - 1);
        auto qkv = linear_multi(inf, x, {"k", "v", "q"}, d, d, /*stream_pt=*/true);
        cache_kv_push(inf, qkv[0], qkv[1]);
        qs.push_back(std::move(qkv[2]));
        ns.push_back(n);
    }
    ASSERT_EQ(inf.k_count(), T);

    std::vector<std::vector<double>> kr, vr, qr;
    for (const auto& pc : inf.cache[inf.scoped("k")])
        kr.push_back(decrypt(inf.cc(), pc.ct, inf.fhe->sk()));
    for (const auto& pc : inf.cache[inf.scoped("v")])
        vr.push_back(decrypt(inf.cc(), pc.ct, inf.fhe->sk()));
    for (const auto& pc : qs)
        qr.push_back(decrypt(inf.cc(), pc.ct, inf.fhe->sk()));

    for (int c = 0; c < static_cast<int>(qs.size()); ++c) {
        const int n_cur = ns[c];
        inf.n_tok = n_cur;
        SCOPED_TRACE("chunk " + std::to_string(c));
        std::cout << "\n##### dblk-bd chunk " << c << " n=" << n_cur << " T=" << T
                  << " #####\n";

        PackedCtx qa = inf.fhe->clone(qs[c]);
        release_device(inf);
        const auto btsA0 = inf.fhe->total_bootstraps;
        PackedCtx outA;
        const double a_s = timed_s([&] { outA = attend(inf, qa); });
        const auto btsA = inf.fhe->total_bootstraps - btsA0;
        auto rawA = decrypt(inf.cc(), outA.ct, inf.fhe->sk());
        outA = PackedCtx{};
        release_device(inf);

        PackedCtx qb = inf.fhe->clone(qs[c]);
        const auto btsB0 = inf.fhe->total_bootstraps;
        PackedCtx outB;
        const double b_s = timed_s([&] { outB = attend_delta(inf, qb); });
        const auto btsB = inf.fhe->total_bootstraps - btsB0;
        auto rawB = decrypt(inf.cc(), outB.ct, inf.fhe->sk());
        outB = PackedCtx{};
        release_device(inf);

        std::cout << std::fixed << std::setprecision(2)
                  << "[dblk-bd] TIMING chunk " << c << " A(per-entry)=" << a_s
                  << "s bts=" << btsA << " | B(δ-block)=" << b_s << "s bts=" << btsB
                  << " speedup=" << (b_s > 0 ? a_s / b_s : 0.0) << "x\n";
        std::cout.unsetf(std::ios::fixed);

        Metrics mA = dist_to_exact(inf, rawA, qr[c], kr, vr, n_cur, c * t, T, false);
        Metrics mB = dist_to_exact(inf, rawB, qr[c], kr, vr, n_cur, c * t, T, false);
        report("A(per-entry) chunk " + std::to_string(c), mA);
        report("B(δ-block)  chunk " + std::to_string(c), mB);
        EXPECT_LT(mB.rmse, 1.5 * mA.rmse + 1e-3);
        EXPECT_LT(mB.max_d, 1.5 * mA.max_d + 0.02);
        EXPECT_LT(btsB, btsA);

        inf.clear_enc_cache();
    }
}

// The shipping causal path on the SAME chunks + metric: the parity-noise baseline.
TEST(BidirectionalAttention, CausalControl) {
    const int T = std::stoi(env_or("MULTI_T", "37"));

    {
        std::ifstream w(default_weights_path());
        if (!w) GTEST_SKIP() << "weights not available";
    }
    const std::string io_path = all_blocks_io_path(default_all_blocks_io_dir(), 0, 128);
    if (!probe_io_file(io_path)) GTEST_SKIP() << "missing GT: " << io_path;
    auto io = read_block0_io(io_path, "ln_1_out", "attn_out");
    ASSERT_GE(static_cast<int>(io.inp.size()), T);

    Inference inf = make_filling_inf();
    install_block0(inf);
    const int d = inf.size.hidDim;
    const int t = inf.slots / d;

    prepare_mha_masks(inf);
    prepare_vcache(inf);

    for (int base = 0; base < T; base += t) {
        const int n = std::min(t, T - base);
        inf.n_tok = n;
        SCOPED_TRACE("chunk base=" + std::to_string(base));
        std::cout << "\n##### causal chunk base=" << base << " n=" << n << " #####\n";
        std::vector<std::vector<double>> rows(io.inp.begin() + base,
                                              io.inp.begin() + base + n);
        PackedCtx x = encode_prefill_input(inf, rows);
        inf.fhe->bootstrap_hint(x, inf.fhe->level_limit() - 1, true);
        inf.fhe->level_hint(x, inf.fhe->level_limit() - 1);
        auto qkv = linear_multi(inf, x, {"k", "v", "q"}, d, d, /*stream_pt=*/true);
        cache_kv_push(inf, qkv[0], qkv[1]);

        std::vector<std::vector<double>> kr, vr;
        for (const auto& pc : inf.cache[inf.scoped("k")])
            kr.push_back(decrypt(inf.cc(), pc.ct, inf.fhe->sk()));
        for (const auto& pc : inf.cache[inf.scoped("v")])
            vr.push_back(decrypt(inf.cc(), pc.ct, inf.fhe->sk()));
        auto q_raw = decrypt(inf.cc(), qkv[2].ct, inf.fhe->sk());

        release_device(inf);
        PackedCtx out = attend(inf, qkv[2]);
        auto raw = decrypt(inf.cc(), out.ct, inf.fhe->sk());

        Metrics m = dist_to_exact(inf, raw, q_raw, kr, vr, n, base, T, /*causal=*/true);
        report("causal chunk@" + std::to_string(base), m);

        release_device(inf);
        inf.clear_enc_cache();
    }
}
