// Isolated value gate for COEFF-mode weight plaintexts (FHE_PT_COEFF_ENCODE).
//
// The host algebra is verified exactly (scratchpad verify_coeff.cpp: the 1-limb encode +
// centered-lift + mod-q reproduces the reference limbs to ±1 ulp). This test gates the GPU
// half: MarkCoeffStaged + ExtractRawPlaintext (staged q0 limb) + LoadPlaintext's
// loadCoeffExpand (upload → INTT → grow → broadcastLimb0 → NTT) must yield a device
// plaintext that multiplies identically to the reference full-limb encode.
#include "ckks_fixture.h"

#include <cmath>
#include <cstdio>
#include <vector>

using test_helpers::CkksFixture;

TEST_F(CkksFixture, CoeffExpandMatchesReference) {
    auto& f = fhe();
    const int N = slots();
    const uint32_t L = 17;   // ViT up/down weight level (11 limbs)

    std::vector<double> w(N), x(N);
    for (int i = 0; i < N; ++i) {
        w[i] = 0.003 * (i % 611) - 0.9;
        x[i] = 0.001 * (i % 997) - 0.4;
    }

    // Reference: the production full-limb encode at L.
    Ptx ref = f.cc->MakeCKKSPackedPlaintext(w, 1, L);

    // Coeff path: 1-limb (q0) encode of ratio-prescaled values, marked + staged.
    const uint32_t lv1  = static_cast<uint32_t>(f.total_depth);
    const double sfL    = f.cc->ScalingFactorReal(L);
    const double ratio  = sfL / f.cc->ScalingFactorReal(lv1);
    std::vector<double> wr(w);
    for (auto& v : wr) v *= ratio;
    Ptx cpt = f.cc->MakeCKKSPackedPlaintext(wr, 1, lv1);
    f.cc->MarkCoeffStaged(cpt, L, sfL);
    f.cc->BeginStageBlock();
    f.cc->ExtractRawPlaintext(cpt);   // stages the single q0 limb into the pinned arena

    // Same encrypted operand against both plaintexts.
    Ptx xpt = f.cc->MakeCKKSPackedPlaintext(x, 1, L);
    Ctx ct  = encrypt(f.cc, xpt, f.pk());

    Ctx y_ref = f.cc->EvalMult(ct, ref);
    Ctx y_cf  = f.cc->EvalMult(ct, cpt);   // internal load takes the staged coeff branch

    const auto d_ref = test_helpers::decrypt_slots(f, y_ref);
    const auto d_cf  = test_helpers::decrypt_slots(f, y_cf);

    double max_diff = 0.0, max_ref = 0.0;
    for (int i = 0; i < N; ++i) {
        max_diff = std::max(max_diff, std::abs(d_ref[i] - d_cf[i]));
        max_ref  = std::max(max_ref, std::abs(d_ref[i]));
    }
    std::printf("[coeff_expand] max|ref|=%.3e  max|ref-coeff|=%.3e\n", max_ref, max_diff);
    // Fresh ct×pt mult precision is ~1e-10; the ±1-ulp coeff rounding is ~2^-53-relative.
    // 1e-6 catches any structural corruption (permutation / wrong limb / wrong scale).
    EXPECT_LT(max_diff, 1e-6);

    // Sanity on the reference itself so a silently-zero mult cannot pass.
    double max_expect = 0.0, max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        const double e = w[i] * x[i];
        max_expect = std::max(max_expect, std::abs(e));
        max_err    = std::max(max_err, std::abs(d_ref[i] - e));
    }
    std::printf("[coeff_expand] max|w*x|=%.3e  ref_vs_plain=%.3e\n", max_expect, max_err);
    EXPECT_LT(max_err, 1e-6);
}

// Production-shaped reproduction: the e2e coeff arms fail the final decode while the
// single-pt case above is exact. Mimic the chunked pipeline: batches of coeff plaintexts
// staged together, loaded on a NON-NULL non-blocking stream ahead of use (upload+expansion
// enqueued before the consumer's mults), device-sync boundary, mult, evict (pool reuse for
// the next batch). Every product is checked, so one corrupted pt pinpoints the batch shape.
TEST_F(CkksFixture, CoeffExpandChunkPipeline) {
    auto& f = fhe();
    const int N = slots();
    const uint32_t L = 16;    // the EAGER weight level (12 limbs) — what the failing run used
    const int BATCH = 64, ROUNDS = 3;

    cudaStream_t astream = nullptr;
    cudaStreamCreateWithFlags(&astream, cudaStreamNonBlocking);

    const uint32_t lv1 = static_cast<uint32_t>(f.total_depth);
    const double sfL   = f.cc->ScalingFactorReal(L);
    const double ratio = sfL / f.cc->ScalingFactorReal(lv1);

    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = 0.001 * (i % 997) - 0.4;
    Ptx xpt = f.cc->MakeCKKSPackedPlaintext(x, 1, L);
    Ctx ct  = encrypt(f.cc, xpt, f.pk());

    double worst = 0.0;
    int bad_pts = 0;
    for (int r = 0; r < ROUNDS; ++r) {
        f.cc->BeginStageBlock();   // arena ping-pong per "block", as the loader does
        std::vector<Ptx> pts(BATCH);
        std::vector<std::vector<double>> ws(BATCH, std::vector<double>(N));
        for (int k = 0; k < BATCH; ++k) {
            for (int i = 0; i < N; ++i)
                ws[k][i] = 0.002 * ((i + 37 * k + r) % 811) - 0.8;
            std::vector<double> wr(ws[k]);
            for (auto& v : wr) v *= ratio;
            pts[k] = f.cc->MakeCKKSPackedPlaintext(wr, 1, lv1);
            f.cc->MarkCoeffStaged(pts[k], L, sfL);
        }
        // CONCURRENT staging — the production stage_plaintexts path is an OMP team on the
        // residency worker; gate the thread-safety of stage_into's atomic bump + the map.
        #pragma omp parallel for num_threads(16) schedule(dynamic, 4)
        for (int k = 0; k < BATCH; ++k)
            f.cc->ExtractRawPlaintext(pts[k]);
        // chunk ACQ: enqueue every load (upload on astream + expansion kernels) up front
        for (int k = 0; k < BATCH; ++k) f.cc->LoadPlaintext(pts[k], astream);
        cudaDeviceSynchronize();   // the pipeline's stage boundary
        for (int k = 0; k < BATCH; ++k) {
            Ctx y = f.cc->EvalMult(ct, pts[k]);
            const auto d = test_helpers::decrypt_slots(f, y);
            double md = 0.0;
            for (int i = 0; i < N; ++i)
                md = std::max(md, std::abs(d[i] - ws[k][i] * x[i]));
            if (md > 1e-6) {
                ++bad_pts;
                if (bad_pts <= 4)
                    std::printf("[coeff_chunk] round=%d pt=%d CORRUPT maxdiff=%.3e\n", r, k, md);
            }
            worst = std::max(worst, md);
        }
        for (int k = 0; k < BATCH; ++k) {   // REL: evict -> device pool reuse next round
            if (pts[k]->loaded) {
                f.cc->EvictDevicePlaintext(pts[k]->gpu);
                pts[k]->gpu = 0;
                pts[k]->loaded = false;
            }
        }
    }
    cudaStreamDestroy(astream);
    std::printf("[coeff_chunk] rounds=%d batch=%d bad_pts=%d worst=%.3e\n",
                ROUNDS, BATCH, bad_pts, worst);
    EXPECT_EQ(bad_pts, 0);
    EXPECT_LT(worst, 1e-6);
}
