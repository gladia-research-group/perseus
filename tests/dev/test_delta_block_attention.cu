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

// Isolated A/B validation of the δ-block prefill attention layout (task #10):
// pipeline A = the shipping per-entry qkt/attention_softmax_thor, pipeline B =
// qkt_delta/attention_softmax_thor_delta (+ the UNCHANGED softmax_v). Both run
// EAGER on the same block-0 weights, the same K/V cache state, and the same
// query ct, chunk by chunk; compared per-slot against each other and (after the
// out-proj) per-token against the exported attn_out ground truth.
// GT rows come from the T=128 dump — causal attention makes ln_1_out/attn_out
// rows for tokens < T identical across dump lengths, so any MULTI_T <= 128 slices.

namespace {

struct SlotDiff { double max_abs = 0.0, mean_abs = 0.0, ref_max = 0.0; };

SlotDiff slot_diff(const std::vector<double>& a, const std::vector<double>& b) {
    SlotDiff s;
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::abs(a[i] - b[i]);
        sum += d;
        if (d > s.max_abs) s.max_abs = d;
        if (std::abs(a[i]) > s.ref_max) s.ref_max = std::abs(a[i]);
    }
    s.mean_abs = sum / std::max<size_t>(1, a.size());
    return s;
}

void report_diff(const std::string& tag, const SlotDiff& s) {
    std::cout << std::scientific << std::setprecision(3)
              << "[dblk] " << tag << " max|A-B|=" << s.max_abs
              << " mean|A-B|=" << s.mean_abs << " |A|max=" << s.ref_max
              << " rel=" << (s.ref_max > 0 ? s.max_abs / s.ref_max : 0.0) << "\n";
}

double gt_compare(const std::string& tag, Inference& inf, const PackedCtx& proj,
                  const std::vector<std::vector<double>>& gt, int base, int n) {
    auto rows = decode_tokens_output(inf, proj, n);
    // gate on per-token w_mape (sum_abs_err / sum_abs_ref): plain per-element
    // mean_rel explodes on near-zero attn_out components (rel_eps floor), see the
    // 48834024 diagnosis — values matched GT to 3 decimals while mean_rel read 4.
    double worst = 0.0, best = 1e30;
    int worst_i = 0;
    for (int i = 0; i < n; ++i) {
        auto s = compare_vec(rows[i], gt[base + i]);
        if (s.w_mape > worst) { worst = s.w_mape; worst_i = i; }
        if (s.w_mape < best) best = s.w_mape;
    }
    std::cout << std::scientific << std::setprecision(3)
              << "[dblk] " << tag << " vs GT attn_out: worst tok" << worst_i << " w_mape="
              << worst << " best=" << best << " over " << n << " tokens\n";
    // no absolute gate here: the SHIPPING path itself reads ~0.20 worst on kc>=2
    // chunks (the known filling parity noise) — callers gate B against A.
    return worst;
}

// Plaintext exact-softmax attention from the DECRYPTED q / K / V cache cts vs the
// decrypted attn-core outputs of BOTH pipelines. Each pipeline carries the known
// filling packing-parity noise (~0.2-0.4 worst-slot on kc>=2 rows) independently,
// so the acceptance gate is COMPARATIVE: B's distance to the exact reference must
// not exceed A's beyond margin. Layout slot[c*tH + h*t + lane].
struct ExactDist { double max_d = 0.0, rmse = 0.0; };

std::pair<ExactDist, ExactDist> self_consistency(
        Inference& inf, const std::vector<double>& rawA, const std::vector<double>& rawB,
        const std::vector<double>& raw_q, int P, int n_cur) {
    const int N  = inf.slots;
    const int t  = N / inf.size.hidDim;
    const int H  = inf.size.numHeads;
    const int tH = t * H;
    const int H_real = inf.size.getRealNumHeads();
    const int DH = inf.size.getRealDHead();
    const auto& kg = inf.cache[inf.scoped("k")];
    const auto& vg = inf.cache[inf.scoped("v")];
    std::vector<std::vector<double>> kr, vr;
    for (const auto& pc : kg) kr.push_back(decrypt(inf.cc(), pc.ct, inf.fhe->sk()));
    for (const auto& pc : vg) vr.push_back(decrypt(inf.cc(), pc.ct, inf.fhe->sk()));
    ExactDist dA, dB;
    double sumA = 0.0, sumB = 0.0;
    size_t cnt = 0;
    double max_ref = 0.0;
    for (int h = 0; h < H_real; ++h) {
        for (int i = 0; i < n_cur; ++i) {
            const int kc = P + i + 1;
            std::vector<double> sc(kc);
            for (int j = 0; j < kc; ++j) {
                double s = 0.0;
                for (int c = 0; c < DH; ++c)
                    s += raw_q[c * tH + h * t + i] * kr[j / t][c * tH + h * t + j % t];
                sc[j] = s / std::sqrt(static_cast<double>(DH));
            }
            const double m = *std::max_element(sc.begin(), sc.end());
            double den = 0.0;
            for (double& s : sc) { s = std::exp(s - m); den += s; }
            for (int c = 0; c < DH; ++c) {
                double o = 0.0;
                for (int j = 0; j < kc; ++j)
                    o += sc[j] / den * vr[j / t][c * tH + h * t + j % t];
                const int slot = c * tH + h * t + i;
                const double ea = std::abs(rawA[slot] - o), eb = std::abs(rawB[slot] - o);
                dA.max_d = std::max(dA.max_d, ea);
                dB.max_d = std::max(dB.max_d, eb);
                sumA += ea * ea; sumB += eb * eb; ++cnt;
                max_ref = std::max(max_ref, std::abs(o));
            }
        }
    }
    dA.rmse = std::sqrt(sumA / std::max<size_t>(1, cnt));
    dB.rmse = std::sqrt(sumB / std::max<size_t>(1, cnt));
    std::cout << std::scientific << std::setprecision(3)
              << "[dblk] dist-to-exact-softmax (decrypted q/K/V): A max|d|=" << dA.max_d
              << " rmse=" << dA.rmse << " | B max|d|=" << dB.max_d
              << " rmse=" << dB.rmse << " |ref|max=" << max_ref << "\n";
    return {dA, dB};
}

std::vector<PackedCtx> clone_group(Inference& inf, const std::vector<PackedCtx>& v) {
    std::vector<PackedCtx> out;
    out.reserve(v.size());
    for (const auto& pc : v) out.push_back(inf.fhe->clone(pc));
    return out;
}

Inference make_filling_inf(bool complex_payload) {
    CKKSContextOptions ckks{};
    ckks.bts_iterations = default_bts_iterations();
    ckks.ckks_complex_payload = complex_payload;
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
    // attention-only test: drop the MLP/LN weights (diagonal-form up/down are the
    // biggest host plaintext consumers and are never touched here)
    for (const char* k : {"up", "up_bias", "down", "down_bias",
                          "ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias"})
        inf.w.erase(k);
}

// device hygiene between pipelines: the model evicts per block; this test runs
// TWO pipelines per chunk on a 64GB card that already holds ~46GB of keys, so
// weight/mask device copies must not accumulate (48834351 chunk-2 device OOM).
// Evicted plaintexts reload on demand.
void release_device(Inference& inf) {
    cudaDeviceSynchronize();
    for (auto& kv : inf.w)
        for (auto& p : kv.second) inf.evict_plaintext(p);
    inf.evict_enc_cache_device();
    cudaDeviceSynchronize();
}

// DBLK_PATH=A|B splits the two pipelines into separate processes (per-path peak
// halves; at 158+ entries the per-entry baseline + harness don't fit next to the
// 46GB key set — 48837043 chunk-64 device OOM). The A process saves its decrypted
// output + wall time; the B process loads them and runs every compare.
std::string dblk_io_file(const std::string& tag, int base) {
    return env_or("DBLK_IO", "/tmp") + "/dblk_" + tag + "_c" + std::to_string(base) + ".bin";
}

void save_vec(const std::string& f, const std::vector<double>& v, double t_s) {
    std::ofstream o(f, std::ios::binary);
    const size_t n = v.size();
    o.write(reinterpret_cast<const char*>(&t_s), sizeof(t_s));
    o.write(reinterpret_cast<const char*>(&n), sizeof(n));
    o.write(reinterpret_cast<const char*>(v.data()), sizeof(double) * n);
    ASSERT_TRUE(o.good()) << "failed to save " << f;
}

void load_vec(const std::string& f, std::vector<double>& v, double& t_s) {
    std::ifstream in(f, std::ios::binary);
    ASSERT_TRUE(in.good()) << "missing A-process artifact " << f;
    size_t n = 0;
    in.read(reinterpret_cast<char*>(&t_s), sizeof(t_s));
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    v.resize(n);
    in.read(reinterpret_cast<char*>(v.data()), sizeof(double) * n);
    ASSERT_TRUE(in.good()) << "truncated A-process artifact " << f;
}

// sync-bracketed wall timing of one attention path
template <typename F>
double timed_s(F&& f) {
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    f();
    cudaDeviceSynchronize();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void report_memfree(const std::string& tag) {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    std::cout << "[dblk] memfree " << tag << ": " << f / 1e9 << " / " << t / 1e9 << " GB\n";
}

void report_rss(const std::string& tag) {
    std::ifstream st("/proc/self/status");
    std::string line;
    while (std::getline(st, line))
        if (line.rfind("VmRSS", 0) == 0) { std::cout << "[dblk] " << tag << " " << line << "\n"; break; }
}

}  // namespace

TEST(DeltaBlockAttention, RealArmVsPerEntry) {
    const int T = std::stoi(env_or("MULTI_T", "64"));

    {
        std::ifstream w(default_weights_path());
        if (!w) GTEST_SKIP() << "weights not available";
    }
    const std::string io_path = all_blocks_io_path(default_all_blocks_io_dir(), 0, 128);
    if (!probe_io_file(io_path)) GTEST_SKIP() << "missing GT: " << io_path;
    auto io = read_block0_io(io_path, "ln_1_out", "attn_out");
    ASSERT_GE(static_cast<int>(io.inp.size()), T);

    Inference inf = make_filling_inf(/*complex_payload=*/false);
    install_block0(inf);
    const int d = inf.size.hidDim;
    const int t = inf.slots / d;

    prepare_mha_masks(inf);
    prepare_vcache(inf);

    // CHUNK_BASE >= 0: measure ONLY that chunk in this process (earlier chunks are
    // rebuilt as warm K/V pushes) — one chunk per process keeps the 64GB card clean
    // of the double-pipeline residue that OOM'd the in-process sweep (48835105).
    const int chunk_base = std::stoi(env_or("CHUNK_BASE", "-1"));

    for (int base = 0; base < T; base += t) {
        if (chunk_base >= 0 && base > chunk_base) break;
        const bool warm_only = (chunk_base >= 0 && base < chunk_base);
        const int n = std::min(t, T - base);
        inf.n_tok = n;
        SCOPED_TRACE("chunk base=" + std::to_string(base) + " n=" + std::to_string(n));
        std::cout << "\n##### chunk base=" << base << " n=" << n
                  << " kc_after_push=" << (base + n)
                  << (warm_only ? " (warm push only)" : "") << " #####\n";

        std::vector<std::vector<double>> rows(io.inp.begin() + base,
                                              io.inp.begin() + base + n);
        PackedCtx x = encode_prefill_input(inf, rows);
        inf.fhe->bootstrap_hint(x, inf.fhe->level_limit() - 1, true);
        inf.fhe->level_hint(x, inf.fhe->level_limit() - 1);
        auto qkv = linear_multi(inf, x, {"k", "v", "q"}, d, d, /*stream_pt=*/true);
        cache_kv_push(inf, qkv[0], qkv[1]);
        if (warm_only) {
            release_device(inf);
            inf.clear_enc_cache();
            continue;
        }
        PackedCtx q = std::move(qkv[2]);
        inf.fhe->level_hint(q, inf.fhe->level_limit() - 3);
        // the attention paths touch no weights; only the GT out-proj needs "out"
        for (auto it = inf.w.begin(); it != inf.w.end();)
            it = (it->first.rfind("out", 0) == 0) ? std::next(it) : inf.w.erase(it);
        release_device(inf);
        report_memfree("pre-attn chunk@" + std::to_string(base));

        const size_t n_entries = cachemir_filling::cf_score_schedule(inf).size();
        const std::string path_sel = env_or("DBLK_PATH", "");

        double a_s = -1.0, gtA = -1.0;
        std::vector<double> rawA;
        if (path_sel != "B") {   // A: shipping per-entry pipeline
            PackedCtx qa = inf.fhe->clone(q);
            PackedCtx outA;
            a_s = timed_s([&] {
                auto sA = cachemir_filling::qkt(inf, qa);
                auto pA = cachemir_filling::attention_softmax_thor(inf, std::move(sA), "attn");
                outA = cachemir_filling::softmax_v(inf, std::move(pA));
            });
            rawA = decrypt(inf.cc(), outA.ct, inf.fhe->sk());
            PackedCtx projA = linear(inf, outA, "out", d, d, /*stream_pt=*/true);
            gtA = gt_compare("A(per-entry) chunk@" + std::to_string(base), inf, projA,
                             io.res, base, n);
            release_device(inf);
        }
        if (path_sel == "A") {   // hand off to the B process
            save_vec(dblk_io_file("rawA", base), rawA, a_s);
            save_vec(dblk_io_file("gtA", base), {gtA}, a_s);
            report_rss("chunk@" + std::to_string(base) + " A end");
            inf.clear_enc_cache();
            continue;
        }
        if (path_sel == "B") {
            load_vec(dblk_io_file("rawA", base), rawA, a_s);
            std::vector<double> g;
            double dummy;
            load_vec(dblk_io_file("gtA", base), g, dummy);
            gtA = g.at(0);
        }

        // B: δ-block pipeline (softmax_v unchanged)
        PackedCtx qb = inf.fhe->clone(q);
        PackedCtx outB;
        const double b_s = timed_s([&] {
            auto sB = cachemir_filling::qkt_delta(inf, qb);
            auto pB = cachemir_filling::attention_softmax_thor_delta(inf, std::move(sB), "attn");
            outB = cachemir_filling::softmax_v(inf, std::move(pB));
        });
        std::cout << std::fixed << std::setprecision(2)
                  << "[dblk] TIMING chunk@" << base << " entries=" << n_entries
                  << " A(per-entry)=" << a_s << "s B(δ-block)=" << b_s
                  << "s speedup=" << (b_s > 0 ? a_s / b_s : 0.0) << "x\n";
        std::cout.unsetf(std::ios::fixed);
        release_device(inf);

        auto rawB = decrypt(inf.cc(), outB.ct, inf.fhe->sk());
        SlotDiff sd = slot_diff(rawA, rawB);
        report_diff("attn_core chunk@" + std::to_string(base), sd);
        {
            auto raw_q = decrypt(inf.cc(), q.ct, inf.fhe->sk());
            auto [dA, dB] = self_consistency(inf, rawA, rawB, raw_q, base, n);
            // comparative gate: δ-block no farther from exact softmax than the
            // shipping path (both carry the known parity noise independently)
            EXPECT_LT(dB.rmse, 1.5 * dA.rmse + 1e-3)
                << "δ-block rmse-to-exact exceeds per-entry's";
            EXPECT_LT(dB.max_d, 1.5 * dA.max_d + 0.02)
                << "δ-block worst-slot-to-exact exceeds per-entry's";
        }

        PackedCtx projB = linear(inf, outB, "out", d, d, /*stream_pt=*/true);
        const double gtB =
            gt_compare("B(δ-block) chunk@" + std::to_string(base), inf, projB, io.res, base, n);
        EXPECT_LT(gtB, 1.5 * gtA + 0.05) << "δ-block GT error exceeds per-entry's";
        EXPECT_LT(gtB, 0.5) << "δ-block GT error implausibly large";

        report_rss("chunk@" + std::to_string(base) + " end");
        inf.clear_enc_cache();   // masks/streamed pts rebuild per chunk; keep host RSS flat
    }
}

TEST(DeltaBlockAttention, TokenPairVsPerEntry) {
    const int T = std::stoi(env_or("MULTI_T", "128"));

    {
        std::ifstream w(default_weights_path());
        if (!w) GTEST_SKIP() << "weights not available";
    }
    const std::string io_path = all_blocks_io_path(default_all_blocks_io_dir(), 0, 128);
    if (!probe_io_file(io_path)) GTEST_SKIP() << "missing GT: " << io_path;
    auto io = read_block0_io(io_path, "ln_1_out", "attn_out");
    ASSERT_GE(static_cast<int>(io.inp.size()), T);

    Inference inf = make_filling_inf(/*complex_payload=*/true);
    inf.token_pair = true;
    install_block0(inf);
    const int d = inf.size.hidDim;
    const int t = inf.slots / d;

    prepare_mha_masks(inf);
    prepare_vcache(inf);

    const int chunk_base = std::stoi(env_or("CHUNK_BASE", "-1"));   // pair base, see RealArm

    for (int base = 0; base < T; base += 2 * t) {
        if (chunk_base >= 0 && base > chunk_base) break;
        const bool warm_only = (chunk_base >= 0 && base < chunk_base);
        const int n_pair = std::min(2 * t, T - base);
        SCOPED_TRACE("pair base=" + std::to_string(base) + " n=" + std::to_string(n_pair));
        std::cout << "\n##### token-pair chunk base=" << base << " n=" << n_pair
                  << (warm_only ? " (warm push only)" : "") << " #####\n";

        std::vector<std::vector<double>> rows(io.inp.begin() + base,
                                              io.inp.begin() + base + n_pair);
        PackedCtx x = encode_prefill_input(inf, rows);   // sets n_tok / n_tok_imag
        const int nA = inf.n_tok, nB = inf.n_tok_imag;
        inf.fhe->bootstrap_hint(x, inf.fhe->level_limit() - 1, true);
        inf.fhe->level_hint(x, inf.fhe->level_limit() - 1);
        cachemir_filling::mha_qkv_token_pair(inf, x);    // defers pushes into tp.k / tp.v

        if (warm_only) {   // both halves' pushes without attention (mirror the attn prologue)
            PackedCtx K = std::move(inf.cache[inf.scoped("tp.k")][0]);
            PackedCtx V = std::move(inf.cache[inf.scoped("tp.v")][0]);
            inf.n_tok = nA;
            cachemir_filling::cache_k_push(inf, K);
            cachemir_filling::cache_v_push(inf, V);
            if (nB > 0) {
                inf.n_tok = nB;
                cachemir_filling::cache_k_push_imag(inf, K);
                cachemir_filling::cache_v_push_imag(inf, V);
            }
            inf.n_tok = nA;
            inf.cache.erase(inf.scoped("tp.k"));
            inf.cache.erase(inf.scoped("tp.v"));
            release_device(inf);
            inf.clear_enc_cache();
            continue;
        }
        if (chunk_base >= 0) {   // split mode: last pair in this process — attention
            // touches no weights; only the GT out-proj needs "out"
            for (auto it = inf.w.begin(); it != inf.w.end();)
                it = (it->first.rfind("out", 0) == 0) ? std::next(it) : inf.w.erase(it);
            release_device(inf);
            report_memfree("tp pre-attn chunk@" + std::to_string(base));
        }

        const std::string path_sel = env_or("DBLK_PATH", "");
        double a_s = -1.0;
        std::vector<double> a_re, a_im;   // Re(A) and Re(i·A) = −Im(A)

        if (path_sel != "B") {
            // snapshot cache state so both paths see the identical entry state
            // (in the A-only process the restore is skipped — the loop ends here)
            auto k_snap  = clone_group(inf, inf.cache[inf.scoped("k")]);
            auto v_snap  = clone_group(inf, inf.cache[inf.scoped("v")]);
            auto tk_snap = clone_group(inf, inf.cache[inf.scoped("tp.k")]);
            auto tv_snap = clone_group(inf, inf.cache[inf.scoped("tp.v")]);
            const int kc_snap = inf.k_count(), vc_snap = inf.v_count();

            PackedCtx xa = inf.fhe->clone(x);
            release_device(inf);
            PackedCtx outA;
            a_s = timed_s([&] { outA = cachemir_filling::mha_attn_token_pair(inf, xa); });
            a_re = decrypt(inf.cc(), outA.ct, inf.fhe->sk());
            PackedCtx outA_i = inf.pack(inf.fhe->mult_i(outA.ct), inf.packing.kind);
            a_im = decrypt(inf.cc(), outA_i.ct, inf.fhe->sk());

            if (path_sel == "A") {
                save_vec(dblk_io_file("tpA_re", base), a_re, a_s);
                save_vec(dblk_io_file("tpA_im", base), a_im, a_s);
                inf.n_tok = nA;
                report_rss("tp chunk@" + std::to_string(base) + " A end");
                inf.clear_enc_cache();
                continue;
            }
            inf.cache[inf.scoped("k")]    = std::move(k_snap);
            inf.cache[inf.scoped("v")]    = std::move(v_snap);
            inf.cache[inf.scoped("tp.k")] = std::move(tk_snap);
            inf.cache[inf.scoped("tp.v")] = std::move(tv_snap);
            inf.k_count() = kc_snap;
            inf.v_count() = vc_snap;
            inf.n_tok = nA;
            inf.n_tok_imag = nB;
        } else {
            load_vec(dblk_io_file("tpA_re", base), a_re, a_s);
            double dummy;
            load_vec(dblk_io_file("tpA_im", base), a_im, dummy);
        }

        PackedCtx xb = inf.fhe->clone(x);
        release_device(inf);
        PackedCtx outB;
        const double b_s = timed_s([&] { outB = cachemir_filling::mha_attn_token_pair_delta(inf, xb); });
        std::cout << std::fixed << std::setprecision(2)
                  << "[dblk] TIMING tp chunk@" << base
                  << " A(per-entry)=" << a_s << "s B(δ-block)=" << b_s
                  << "s speedup=" << (b_s > 0 ? a_s / b_s : 0.0) << "x\n";
        std::cout.unsetf(std::ios::fixed);

        // packed output halves: Re from a plain decrypt, Im via Re(i·x) = −Im(x)
        auto b_re = decrypt(inf.cc(), outB.ct, inf.fhe->sk());
        PackedCtx outB_i = inf.pack(inf.fhe->mult_i(outB.ct), inf.packing.kind);
        auto b_im = decrypt(inf.cc(), outB_i.ct, inf.fhe->sk());
        SlotDiff sre, sim;
        double amax = 0.0;
        for (size_t i = 0; i < a_re.size(); ++i) {
            sre.max_abs = std::max(sre.max_abs, std::abs(a_re[i] - b_re[i]));
            sim.max_abs = std::max(sim.max_abs, std::abs(a_im[i] - b_im[i]));
            amax = std::max(amax, std::abs(a_re[i]));
        }
        std::cout << std::scientific << std::setprecision(3)
                  << "[dblk] tp attn_core@" << base << " max|Re(A-B)|=" << sre.max_abs
                  << " max|Im(A-B)|=" << sim.max_abs << " |Re(A)|max=" << amax << "\n";
        EXPECT_LT(sre.max_abs, 0.10 * std::max(1.0, amax));
        EXPECT_LT(sim.max_abs, 0.10 * std::max(1.0, amax));

        // GT anchor on the A (Re) half after out-proj (loose ceiling only — no
        // per-entry proj here; the slot-diff above is the A/B gate)
        PackedCtx projB = linear(inf, outB, "out", d, d, /*stream_pt=*/true);
        const double gtB = gt_compare("B(δ-block,tp Re-half) chunk@" + std::to_string(base),
                                      inf, projB, io.res, base, nA);
        EXPECT_LT(gtB, 0.5) << "δ-block TP GT error implausibly large";

        inf.n_tok = nA;   // pushes for the NEXT pair already happened inside path B
        report_rss("tp chunk@" + std::to_string(base) + " end");
        inf.clear_enc_cache();
    }
}
