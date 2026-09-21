// Granularity A/B for diagonal::linear — the k=1 matrices convicted the Linear-granularity
// CHUNKED residency branch of value corruption at eager (job 50045735: chunk_nostage fails the
// decode with no staging/release/coeff armed; every other failing arm also ran gran=linear).
// This isolates it: the SAME input × SAME weights through the per-mult path (Plaintext) and
// the chunked pipeline (Linear) must agree slotwise.
#include "fideslib_wrapper.h"
#include "inference.h"
#include "packing/diagonal/diagonal_linear.h"
#include "packing/diagonal/diagonal_linear_utils.h"
#include "packing/cachemir_filling/cachemir_filling_rot_indices.h"
#include "test_helpers.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

class DgLinearFixture : public ::testing::Test {
 protected:
    static void SetUpTestSuite() {
        CKKSContextOptions o = test_helpers::default_ckks_options();
        const int s = (o.batch_size == 0) ? (1 << (o.logN - 1)) : static_cast<int>(o.batch_size);
        for (int32_t r : cachemir_filling::compute_gpt2_rot_indices(s, 1024, 4096, 16))
            o.extra_rot_steps.push_back(r);
        ctx_   = make_ckks_context(o);
        slots_ = static_cast<int>(ctx_->cc->GetRingDimension() / 2);
    }
    static void TearDownTestSuite() { ctx_.reset(); }

    static Inference make_inf() {
        Inference inf;
        inf.fhe   = ctx_;
        inf.slots = slots_;
        inf.size.hidDim = 1024;
        inf.size.dim    = 768;
        inf.size.expDim = 4096;
        inf.size.expanded = 3072;
        inf.packing = inf.make_packing(PackingKind::CachemirFilling);
        return inf;
    }

    inline static std::shared_ptr<CKKSContext> ctx_;
    inline static int slots_ = 0;
};

}  // namespace

// Decrypt-probe matrix: the A/B below fails at its BASELINE decrypt, k=1 eager fails its
// plain control, k=12 eager passes — corner the failing (valve, level, deg) combination.
// A fresh-encrypted ct decrypts WITHOUT the lazy-shadow re-encrypt valve (cpu side intact);
// any GPU-computed ct decrypts THROUGH it.
TEST_F(DgLinearFixture, DecryptProbeMatrix) {
    Inference inf = make_inf();
    const int N = inf.slots;
    std::vector<double> v(N);
    for (int i = 0; i < N; ++i) v[i] = 0.0001 * ((i % 601) - 300);

    auto probe = [&](const char* tag, const Ctx& ct) {
        std::printf("[probe] %-28s level=%d deg=%d : ", tag,
                    static_cast<int>(ct->GetLevel()),
                    static_cast<int>(ct->GetNoiseScaleDeg()));
        try {
            const auto d = decrypt(inf.cc(), ct, inf.fhe->sk());
            double m = 0.0;
            int n_junk = 0;
            for (size_t i = 0; i < d.size(); ++i) {
                m = std::max(m, std::abs(d[i]));
                if (std::abs(d[i]) > 1.0) ++n_junk;
            }
            std::printf("OK  max|d|=%.3e n_junk=%d\n", m, n_junk);
            return true;
        } catch (const std::exception& e) {
            std::printf("THREW\n");
            return false;
        }
    };

    // fresh-encode ROUND-TRIP probe: encode+encrypt+decrypt with NO GPU involvement — a
    // dirty result here means host encode/decrypt state is already poisoned at this point.
    auto fresh_probe = [&](const char* tag) {
        Ptx fp = inf.cc()->MakeCKKSPackedPlaintext(v, 1, 16);
        Ctx fc = encrypt(inf.cc(), fp, inf.fhe->pk());
        probe(tag, fc);
    };

    fresh_probe("RT before any GPU op");
    for (uint32_t lv : {0u, 16u}) {
        Ptx pt  = inf.cc()->MakeCKKSPackedPlaintext(v, 1, lv);
        Ctx ct  = encrypt(inf.cc(), pt, inf.fhe->pk());
        char tag[64];
        std::snprintf(tag, sizeof tag, "fresh@%u (no valve)", lv);
        probe(tag, ct);

        Ctx sum = inf.cc()->EvalAdd(ct, ct);            // GPU ct, deg 1, same level
        std::snprintf(tag, sizeof tag, "add@%u (valve deg1)", lv);
        probe(tag, sum);
        std::snprintf(tag, sizeof tag, "RT after add@%u", lv);
        fresh_probe(tag);

        Ptx one = inf.cc()->MakeCKKSPackedPlaintext(std::vector<double>(N, 1.0), 1, lv);
        Ctx prod = inf.cc()->EvalMult(ct, one);         // GPU ct, deg 2 — FIRST pt-load op
        std::snprintf(tag, sizeof tag, "mult@%u (valve deg2)", lv);
        probe(tag, prod);
        std::snprintf(tag, sizeof tag, "RT after mult@%u", lv);
        fresh_probe(tag);

        Ctx rot = inf.cc()->EvalRotate(prod, 32);       // deg 2 + keyswitch
        std::snprintf(tag, sizeof tag, "mult+rot@%u (valve deg2)", lv);
        probe(tag, rot);
        std::snprintf(tag, sizeof tag, "RT after rotate@%u", lv);
        fresh_probe(tag);
    }
}

// Staged replication of diagonal::linear's pipeline with a decrypt probe at every step —
// the full linear's output fails decode while every primitive probes clean, so walk to the
// first corrupt stage: encode-input → clone → hoisted baby rotations → single ct×pt mult →
// 32-term inner sum → +giant rotate/accumulate.
TEST_F(DgLinearFixture, LinearPipelineWalk) {
    Inference inf = make_inf();
    const int d = 1024, T = 8;
    const int N = inf.slots;

    auto probe = [&](const char* tag, const Ctx& ct) {
        std::printf("[walk] %-24s level=%d deg=%d : ", tag,
                    static_cast<int>(ct->GetLevel()),
                    static_cast<int>(ct->GetNoiseScaleDeg()));
        try {
            const auto dv = decrypt(inf.cc(), ct, inf.fhe->sk());
            double m = 0.0;
            int first_junk = -1, n_junk = 0;
            for (size_t i = 0; i < dv.size(); ++i) {
                const double a = std::abs(dv[i]);
                m = std::max(m, a);
                if (a > 1.0) { ++n_junk; if (first_junk < 0) first_junk = (int)i; }
            }
            std::printf("OK  max|d|=%.3e n_junk=%d first=%d size=%zu\n",
                        m, n_junk, first_junk, dv.size());
            return true;
        } catch (const std::exception&) {
            std::printf("THREW\n");
            return false;
        }
    };

    // TRIGGER bisect: small-RT baseline → BIG-value raw encode FIRST (no eli) → small-RT →
    // eli → small-RT. Splits value-magnitude-vs-function as the trigger, and persistence.
    {
        auto small_rt = [&](const char* tag) {
            std::vector<double> vb(N);
            for (int i = 0; i < N; ++i) vb[i] = 0.0001 * ((i % 601) - 300);
            Ptx bp = inf.cc()->MakeCKKSPackedPlaintext(vb, 1, 16);
            Ctx bc = encrypt(inf.cc(), bp, inf.fhe->pk());
            probe(tag, bc);
        };
        small_rt("RT-1 (baseline)");

        std::vector<double> big(N, 0.0);
        for (int i = 0; i < 8192; ++i) big[i] = 0.001 * ((i % 611) - 300);
        Ptx gp = inf.cc()->MakeCKKSPackedPlaintext(big, 1, 16);
        Ctx gc = encrypt(inf.cc(), gp, inf.fhe->pk());
        probe("rawMake BIG contiguous", gc);

        small_rt("RT-2 (after big raw)");

        std::vector<double> sparse(N, 0.0);
        for (int tok = 0; tok < T; ++tok)
            for (int i = 0; i < d; ++i)
                sparse[i * 32 + tok] = 0.001 * ((int)(((size_t)tok * d + i) % 611) - 300);
        Ptx sp = inf.cc()->MakeCKKSPackedPlaintext(sparse, 1, 16);
        Ctx sc = encrypt(inf.cc(), sp, inf.fhe->pk());
        probe("stride32 (autoloaded)", sc);

        // Encrypt AUTO-LOADS the ct to the GPU — every probe so far decoded through the
        // device upload/store round trip. Evict => the true pure-CPU decrypt.
        if (sc->loaded) { inf.cc()->EvictDeviceCiphertext(sc->gpu); sc->gpu = 0; sc->loaded = false; }
        probe("stride32 EVICTED (pure CPU)", sc);

        // Re-upload the INTACT cpu ct and decode again: upload+store round trip, no compute.
        inf.cc()->LoadCiphertext(sc);
        probe("stride32 RELOADED (roundtrip)", sc);

        small_rt("RT-3 (after big sparse)");
    }

    // FUNCTION-vs-VALUES split: the junk carries a 0.001*2^64 signature (int64 wrap at encode)
    // yet a textually identical raw Make+Encrypt in DecryptProbeMatrix decodes clean. Cross the
    // two encode paths with the two value sets.
    {
        std::vector<double> x0(static_cast<size_t>(T) * d);
        for (size_t i = 0; i < x0.size(); ++i) x0[i] = 0.001 * ((int)(i % 611) - 300);
        PackedCtx pre = diagonal::encode_linear_input(inf, x0, d, d, /*target_level=*/16);
        probe("eli(walk vals)", pre.ct);

        std::vector<double> xs(static_cast<size_t>(T) * d);
        for (size_t i = 0; i < xs.size(); ++i) xs[i] = 0.0001 * ((int)(i % 601) - 300);
        PackedCtx pre2 = diagonal::encode_linear_input(inf, xs, d, d, /*target_level=*/16);
        probe("eli(probe vals)", pre2.ct);

        std::vector<double> raw(N, 0.0);
        for (size_t i = 0; i < x0.size(); ++i) raw[i] = x0[i];
        Ptx rp = inf.cc()->MakeCKKSPackedPlaintext(raw, 1, 16);
        Ctx rc = encrypt(inf.cc(), rp, inf.fhe->pk());
        probe("rawMake(walk vals)", rc);

        Ptx rp2 = inf.cc()->MakeCKKSPackedPlaintext(raw, 1, 0);
        Ctx rc2 = encrypt(inf.cc(), rp2, inf.fhe->pk());
        probe("rawMake(walk vals)@0", rc2);
    }

    std::vector<std::vector<double>> W(d, std::vector<double>(d));
    for (int r = 0; r < d; ++r)
        for (int c = 0; c < d; ++c)
            W[r][c] = 0.02 * (((r * 131 + c * 7) % 41) - 20);
    inf.w["w"] = diagonal::encode_weight_matrix(inf, W, d, d, /*target_level=*/16);
    auto& pts_W = inf.w.at("w");

    std::vector<double> x(static_cast<size_t>(T) * d);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.001 * ((int)(i % 611) - 300);
    PackedCtx xc = diagonal::encode_linear_input(inf, x, d, d, /*target_level=*/16);
    probe("input (fresh)", xc.ct);

    PackedCtx x_rep = inf.fhe->clone(xc);
    probe("clone", x_rep.ct);

    auto p = diagonal::compute_dg_params(N, d, d);
    std::printf("[walk] dg params: s=%d G=%d t_in=%d t_out=%d alpha=%d is_up=%d\n",
                p.s, p.G, p.t_in, p.t_out, p.alpha, (int)p.is_up);

    std::vector<int32_t> steps;
    for (int b = 1; b < p.s; ++b) steps.push_back(diagonal::dg_rot(inf, b * p.t_in));
    std::vector<PackedCtx> rots = inf.fhe->rotate_hoisted(x_rep, steps);
    probe("hoisted rot b=1", rots[0].ct);
    probe("hoisted rot b=31", rots[p.s - 2].ct);

    inf.load_plaintext(pts_W[0], nullptr);
    PackedCtx m0 = inf.fhe->mult(x_rep, pts_W[0]);
    probe("mult b=0 g=0", m0.ct);

    // inner sum for giant 0 (the per-mult path's first accumulation)
    PackedCtx acc = m0;
    for (int b = 1; b < p.s; ++b) {
        inf.load_plaintext(pts_W[static_cast<size_t>(b) * p.G], nullptr);
        PackedCtx tmp = inf.fhe->mult(rots[b - 1], pts_W[static_cast<size_t>(b) * p.G]);
        inf.fhe->inplace_add(acc, tmp);
    }
    probe("inner_sum g=0", acc.ct);

    // one giant step: inner sum for g=1, rotate, accumulate
    PackedCtx acc1;
    {
        inf.load_plaintext(pts_W[1], nullptr);
        acc1 = inf.fhe->mult(x_rep, pts_W[1]);
        for (int b = 1; b < p.s; ++b) {
            inf.load_plaintext(pts_W[static_cast<size_t>(b) * p.G + 1], nullptr);
            PackedCtx tmp = inf.fhe->mult(rots[b - 1], pts_W[static_cast<size_t>(b) * p.G + 1]);
            inf.fhe->inplace_add(acc1, tmp);
        }
    }
    probe("inner_sum g=1", acc1.ct);
    PackedCtx rot1 = inf.fhe->rotate(acc1, diagonal::dg_rot(inf, p.s * p.t_in));
    probe("giant rotate g=1", rot1.ct);
    inf.fhe->inplace_add(acc, rot1);
    probe("acc after g=1", acc.ct);
}

TEST_F(DgLinearFixture, ChunkedMatchesPerMult) {
    Inference inf = make_inf();
    const int d = 1024, T = 8;

    std::vector<std::vector<double>> W(d, std::vector<double>(d));
    for (int r = 0; r < d; ++r)
        for (int c = 0; c < d; ++c)
            W[r][c] = 0.02 * (((r * 131 + c * 7) % 41) - 20);

    inf.w["w"] = diagonal::encode_weight_matrix(inf, W, d, d, /*target_level=*/16);

    std::vector<double> x(static_cast<size_t>(T) * d);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.001 * ((int)(i % 611) - 300);
    PackedCtx xc = diagonal::encode_linear_input(inf, x, d, d, /*target_level=*/16);

    inf.weight_granularity = WeightGranularity::Plaintext;
    PackedCtx ya = diagonal::linear(inf, xc, "w", d, d, /*stream_pt=*/true);
    std::vector<double> da, db;
    try {
        da = test_helpers::decrypt_slots(inf, ya.ct);
        std::printf("[chunked_linear] PER-MULT decode OK\n");
    } catch (const std::exception& e) {
        std::printf("[chunked_linear] PER-MULT decode THREW: %s\n", e.what());
        FAIL() << "baseline (per-mult) arm failed decode — test setup issue, not the chunked branch";
    }

    inf.weight_granularity = WeightGranularity::Linear;   // routes the CHUNKED branch
    PackedCtx yb = diagonal::linear(inf, xc, "w", d, d, /*stream_pt=*/true);
    try {
        db = test_helpers::decrypt_slots(inf, yb.ct);
        std::printf("[chunked_linear] CHUNKED decode OK\n");
    } catch (const std::exception& e) {
        std::printf("[chunked_linear] CHUNKED decode THREW: %s\n", e.what());
        FAIL() << "chunked arm failed decode — corruption reproduced in isolation";
    }

    double max_ab = 0.0, max_a = 0.0;
    for (int i = 0; i < inf.slots; ++i) {
        max_ab = std::max(max_ab, std::abs(da[i] - db[i]));
        max_a  = std::max(max_a, std::abs(da[i]));
    }
    std::printf("[chunked_linear] max|permult|=%.3e  max|permult-chunked|=%.3e\n", max_a, max_ab);
    EXPECT_LT(max_ab, 1e-6);

    // and both against the true matmul for token 0
    std::vector<double> y0(d, 0.0);
    for (int j = 0; j < d; ++j)
        for (int i = 0; i < d; ++i)
            y0[j] += x[i] * W[i][j];
    const int t_out = inf.slots / d;
    double max_ref_err = 0.0;
    for (int j = 0; j < d; ++j)
        max_ref_err = std::max(max_ref_err, std::abs(da[static_cast<size_t>(j) * t_out] - y0[j]));
    std::printf("[chunked_linear] permult_vs_plain(tok0)=%.3e\n", max_ref_err);
    EXPECT_LT(max_ref_err, 1e-4);
}

// The square 1024x1024 above is alpha=1. The MLP up (1024->4096) and down (4096->1024)
// projections take the is_up alpha>1 replication branch and the wide-G chunked path — the
// E2E stage_lin fails at eager while ChunkedMatchesPerMult (square only) passed, so cover
// the shapes the square case never exercised. Per-mult vs chunked must agree slotwise.
TEST_F(DgLinearFixture, ChunkedShapesMatchPerMult) {
    struct Shape { const char* tag; int d_in; int d_out; };
    const Shape shapes[] = {{"up 1024->4096", 1024, 4096}, {"down 4096->1024", 4096, 1024}};
    const int T = 8;

    for (const auto& sh : shapes) {
        Inference inf = make_inf();
        const int d_in = sh.d_in, d_out = sh.d_out;

        std::vector<std::vector<double>> W(d_in, std::vector<double>(d_out));
        for (int r = 0; r < d_in; ++r)
            for (int c = 0; c < d_out; ++c)
                W[r][c] = 0.01 * (((r * 131 + c * 7) % 41) - 20);
        inf.w["w"] = diagonal::encode_weight_matrix(inf, W, d_in, d_out, /*target_level=*/16);

        std::vector<double> x(static_cast<size_t>(T) * d_in);
        for (size_t i = 0; i < x.size(); ++i) x[i] = 0.001 * ((int)(i % 611) - 300);
        PackedCtx xc = diagonal::encode_linear_input(inf, x, d_in, d_out, /*target_level=*/16);

        inf.weight_granularity = WeightGranularity::Plaintext;
        PackedCtx ya = diagonal::linear(inf, xc, "w", d_in, d_out, /*stream_pt=*/true);
        inf.weight_granularity = WeightGranularity::Linear;
        PackedCtx yb = diagonal::linear(inf, xc, "w", d_in, d_out, /*stream_pt=*/true);

        std::vector<double> da, db;
        bool a_ok = true, b_ok = true;
        try { da = test_helpers::decrypt_slots(inf, ya.ct); }
        catch (const std::exception&) { a_ok = false; }
        try { db = test_helpers::decrypt_slots(inf, yb.ct); }
        catch (const std::exception&) { b_ok = false; }
        std::printf("[shapes] %-18s per-mult=%s chunked=%s\n", sh.tag,
                    a_ok ? "OK" : "THREW", b_ok ? "OK" : "THREW");
        EXPECT_TRUE(a_ok) << sh.tag << ": per-mult (plaintext gran) threw";
        EXPECT_TRUE(b_ok) << sh.tag << ": chunked (linear gran) threw";
        if (a_ok && b_ok) {
            double mab = 0.0, ma = 0.0;
            for (int i = 0; i < inf.slots; ++i) {
                mab = std::max(mab, std::abs(da[i] - db[i]));
                ma  = std::max(ma, std::abs(da[i]));
            }
            std::printf("[shapes] %-18s max|permult|=%.3e max|permult-chunked|=%.3e\n",
                        sh.tag, ma, mab);
            EXPECT_LT(mab, 1e-6) << sh.tag;
        }
    }
}
