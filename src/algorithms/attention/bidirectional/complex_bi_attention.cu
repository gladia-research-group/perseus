#include "packing/bidirectional/bi_attention.h"
#include "packing/cachemir_filling/cachemir_filling_attention.h"
#include "packing/cachemir_filling/cachemir_filling_delta_attention.h"
#include "packing/cachemir/cachemir_attention_utils.h"
#include "inference.h"
#include "staged_entries.h"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace bidirectional {

namespace {

using cachemir_filling::CfEntry;
using cachemir_filling::DeltaView;

struct BdDims { int N; int t; int tH; int nblk; };
BdDims bd_dims(const Inference& inf) {
    const int N  = inf.slots;
    const int t  = N / inf.size.hidDim;
    const int tH = t * inf.size.numHeads;
    return {N, t, tH, N / tH};
}

// bd active pattern for one half: query lane j < n_cur, key l = j − delta ∈ [0, Lg).
bool bd_active(const Inference& inf, int n_cur, int delta, int Lg, int s,
               const BdDims& dd) {
    if (s >= dd.tH) return false;
    const int h = s / dd.t, j = s % dd.t;
    if (h >= inf.size.getRealNumHeads() || j >= n_cur) return false;
    const int l = j - delta;
    return l >= 0 && l < Lg;
}

// Two-sided δ-block qkt mask: mA on the A pattern, mB on the B pattern, placed at
// the entry's ct-local block. plus = mA+mB rides res, minus = mA−mB rides conj(res).
std::vector<double> bd_tp_qkt_mask(const Inference& inf, const CfEntry& e,
                                   int nA, int nB, bool plus, int blk) {
    const auto dd = bd_dims(inf);
    const double scale = 0.5 / std::sqrt(static_cast<double>(inf.size.getRealDHead()));
    std::vector<double> out(dd.N, 0.0);
    for (int s = 0; s < dd.tH; ++s) {
        const double mA = bd_active(inf, nA, e.delta, e.Lg, s, dd) ? scale : 0.0;
        const double mB = bd_active(inf, nB, e.delta, e.Lg, s, dd) ? scale : 0.0;
        out[blk % dd.nblk * dd.tH + s] = plus ? (mA + mB) : (mA - mB);
    }
    return out;
}

// Half extraction masks (same tags as the legacy push path — encode memo shared).
Ptx half_mask_re(Inference& inf, const PackedCtx& ref, int n_tok) {
    const auto dd = bd_dims(inf);
    const int d_head_real = inf.size.getRealDHead();
    const int H_real      = inf.size.getRealNumHeads();
    return inf.encode_at_cached(
        "cf.kvpush.mask.q.nt" + std::to_string(n_tok), ref, [&] {
            std::vector<double> mask(dd.N, 0.0);
            for (int c = 0; c < d_head_real; ++c)
                for (int h = 0; h < H_real; ++h)
                    for (int i = 0; i < n_tok; ++i)
                        mask[c * dd.tH + h * dd.t + i] = 0.25;
            return mask;
        });
}

Ptx half_mask_im(Inference& inf, const PackedCtx& ref, int n_tok) {
    const auto dd = bd_dims(inf);
    const int d_head_real = inf.size.getRealDHead();
    const int H_real      = inf.size.getRealNumHeads();
    return inf.encode_at_cached_complex(
        "cf.kvpush.mask.q.im.nt" + std::to_string(n_tok), ref, [&] {
            std::vector<std::complex<double>> mask(dd.N, {0.0, 0.0});
            for (int c = 0; c < d_head_real; ++c)
                for (int h = 0; h < H_real; ++h)
                    for (int i = 0; i < n_tok; ++i)
                        mask[c * dd.tH + h * dd.t + i] = {0.0, -0.25};
            return mask;
        });
}

void split_halves_fresh(Inference& inf, const PackedCtx& packed, int nA, int nB,
                        PackedCtx& a_out, PackedCtx& b_out) {
    Ptx mA = half_mask_re(inf, packed, nA);
    PackedCtx a = inf.fhe->mult(packed, mA);
    inf.fhe->inplace_im_cleanse(a);
    Ptx mB = half_mask_im(inf, packed, nB);
    PackedCtx b = inf.fhe->mult(packed, mB);
    inf.fhe->inplace_im_cleanse(b);
    inf.fhe->bootstrap_pair(a.ct, b.ct);
    inf.fhe->inplace_im_cleanse(a);
    inf.fhe->inplace_im_cleanse(b);
    a_out = std::move(a);
    b_out = std::move(b);
}

// Fused packed QKᵀ into δ-block cts over the bd union schedule.
std::vector<PackedCtx> bd_qkt_pair(Inference& inf, const PackedCtx& query,
                                   const std::vector<PackedCtx>& kg,
                                   int K, int nA, int nB) {
    WithStep _w(inf, "bi.tp.qkt");
    const auto dd = bd_dims(inf);
    const int Gtot  = (dd.t > 0) ? (K + dd.t - 1) / dd.t : 0;
    const int n_cts = ((Gtot - 1) * dd.t + nA + dd.t - 2) / dd.nblk + 1;

    const std::vector<CfEntry> sched = cachemir_filling::cf_score_schedule(inf, K);

    PackedCtx q = inf.fhe->clone(query);
    inf.fhe->inplace_mult(q, 0.5);   // half-scale bts input; complex payload keeps q_A + i·q_B
    inf.fhe->bootstrap(q.ct);

    std::vector<PackedCtx> S(n_cts);

    size_t i0 = 0;
    while (i0 < sched.size()) {
        const int g = sched[i0].g;
        size_t i1 = i0;
        while (i1 < sched.size() && sched[i1].g == g) ++i1;

        const PackedCtx& k = kg[g];
        std::vector<int32_t> steps;
        steps.reserve(i1 - i0);
        for (size_t i = i0; i < i1; ++i)
            if (sched[i].delta != 0) steps.push_back(cachemir::mha_rot(inf, -sched[i].delta));
        std::vector<PackedCtx> krots = inf.fhe->rotate_hoisted(k, steps);

        size_t r = 0;
        for (size_t i = i0; i < i1; ++i) {
        CKKSContext::MagnitudeReuseScope _rm(*inf.fhe, i == 0, false);
        const CfEntry& e = sched[i];
        WithStep _wd(inf, "qkt_delta");

        PackedCtx krot = (e.delta == 0)
            ? inf.fhe->clone(k)
            : std::move(krots[r++]);

        PackedCtx res = inf.fhe->mult(q, krot);

        for (int s = dd.tH; s < dd.N; s *= 2) {
            PackedCtx rot = inf.fhe->rotate(res, cachemir::mha_rot(inf, s));
            inf.fhe->inplace_add(res, rot);
        }

        const int blk = (Gtot - 1 - e.g) * dd.t + e.delta + (dd.t - 1);
        const std::string suff = "d" + std::to_string(e.delta) +
                                 ".L" + std::to_string(e.Lg) +
                                 ".na" + std::to_string(nA) +
                                 ".nb" + std::to_string(nB) +
                                 ".b" + std::to_string(blk);
        Ptx p_pt = inf.encode_at_cached(
            "bi.tp.qkt.mp." + suff, res,
            [&] { return bd_tp_qkt_mask(inf, e, nA, nB, /*plus=*/true, blk); });
        PackedCtx masked = inf.fhe->mult(res, p_pt);
        if (nA != nB) {   // halves' query-lane bounds differ: + (mA−mB)·conj(res)
            PackedCtx cres = inf.fhe->conjugate(res);
            Ptx q_pt = inf.encode_at_cached(
                "bi.tp.qkt.mq." + suff, cres,
                [&] { return bd_tp_qkt_mask(inf, e, nA, nB, /*plus=*/false, blk); });
            PackedCtx corr = inf.fhe->mult(cres, q_pt);
            inf.fhe->inplace_add(masked, corr);
        }
        const int ci = blk / dd.nblk;
        if (!S[ci].ct) S[ci] = std::move(masked);
        else           inf.fhe->inplace_add(S[ci], masked);
        }
        i0 = i1;
    }

    return S;
}

constexpr const char* kStageA = "bi.tp.a.";
constexpr const char* kStageB = "bi.tp.b.";
constexpr const char* kStageZ = "bi.tp.z.";

}  // namespace

PackedCtx complex_bi_attention(Inference& inf, PackedCtx q, PackedCtx k, PackedCtx v,
                               int nA, int nB) {
    const auto dd = bd_dims(inf);
    if (nB <= 0)
        throw std::runtime_error("complex_bi_attention: nB > 0 required "
                                 "(single-chunk inputs take the real delta arm)");
    if (nA != dd.t)
        throw std::runtime_error("complex_bi_attention: chunk A must be full (nA == t_stride)");
    const int K = nA + nB;

    inf.fhe->level_hint(q, inf.fhe->level_headroom(3));   // mirror the real attn_core hint

    // K/V half extraction costs 4 bootstraps: bootstrap_pair returns early unconditionally
    // (fideslib_wrapper.h), so each split runs 2 real ones. Re-arming the pair would halve
    // that, but it changes the bootstrap count, so it needs a re-capture and a re-plan.
    PackedCtx kA, kB, vA, vB;
    { WithStep _wk(inf, "bi.tp.k_prep"); split_halves_fresh(inf, k, nA, nB, kA, kB); }
    k = PackedCtx{};
    { WithStep _wv(inf, "bi.tp.v_prep"); split_halves_fresh(inf, v, nA, nB, vA, vB); }
    v = PackedCtx{};
    std::vector<PackedCtx> kg{std::move(kA), std::move(kB)};
    std::vector<PackedCtx> vg{std::move(vA), std::move(vB)};

    // Union (A-view) schedule: n_cur = nA = t bounds the delta range for both halves.
    inf.n_tok = nA;
    std::vector<PackedCtx> packed = bd_qkt_pair(inf, q, kg, K, nA, nB);
    q = PackedCtx{};

    // conj_split on the big cts: ONE conjugate per ct serves both halves.
    const int n_cts = static_cast<int>(packed.size());
    std::vector<PackedCtx> A(n_cts), B(n_cts);
    {
        WithStep _ws(inf, "bi.tp.split");
        for (int ci = 0; ci < n_cts; ++ci) {
            PackedCtx c = inf.fhe->conjugate(packed[ci]);
            A[ci] = inf.fhe->add(packed[ci], c);                       // 2·Re = A rows
            B[ci] = inf.fhe->mult_i(inf.fhe->sub(c, packed[ci]));      // 2·Im = B rows
            packed[ci] = PackedCtx{};
        }
    }

    // Per-half views: same union entries/blocks; only the query-lane bound and the
    // alive flags differ (entries with delta ≥ nB have no active B rows).
    DeltaView va = cachemir_filling::bd_delta_view(inf, K);   // n_cur = nA
    DeltaView vb = va;
    vb.n_cur = nB;
    for (auto& e : vb.entries) e.alive = (e.delta < nB);

    std::vector<PackedCtx> sA = cachemir_filling::attention_softmax_thor_delta_core(
        inf, std::move(A), "attn", va, kStageA);
    std::vector<PackedCtx> sB = cachemir_filling::attention_softmax_thor_delta_core(
        inf, std::move(B), "attn", vb, kStageB);

    std::vector<PackedCtx> probs;
    {   // zip: pack the probs per union entry (B-dead entries ride pure-real)
        WithStep _wz(inf, "bi.tp.zip");
        size_t n_b = 0;
        for (const auto& e : vb.entries) n_b += e.alive ? 1 : 0;
        StagedEntries zipA(inf, StagedEntries::auto_active(va.entries.size()), kStageA);
        zipA.adopt(std::move(sA));
        StagedEntries zipB(inf, StagedEntries::auto_active(n_b), kStageB);
        zipB.adopt(std::move(sB));
        StagedEntries zipped(inf, StagedEntries::auto_active(va.entries.size()), kStageZ);
        size_t bi = 0;
        for (size_t idx = 0; idx < va.entries.size(); ++idx) {
            CKKSContext::MagnitudeReuseScope _rm(*inf.fhe, idx == 0, false);
            PackedCtx pa = zipA.consume(idx);
            if (vb.entries[idx].alive) {
                PackedCtx pb = zipB.consume(bi++);
                zipped.push(inf.fhe->pair_pack(pa, pb));
            } else {
                zipped.push(std::move(pa));
            }
            if (zipped.size() % 32 == 0) zipped.seal();
        }
        probs = zipped.into_vector();
    }

    PackedCtx out = cachemir_filling::softmax_v_groups(inf, std::move(probs), vg, K, kStageZ);
    inf.n_tok = nA;
    inf.n_tok_imag = nB;
    inf.fhe->tp_probe("bi_attn", out.ct);
    return out;
}

}  // namespace bidirectional
