#include "packing/cachemir_filling/cachemir_filling_delta_attention.h"
#include "packing/cachemir_filling/cachemir_filling_attention.h"
#include "packing/cachemir/cachemir_attention_utils.h"
#include "inference.h"
#include "staged_entries.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cachemir_filling {

namespace {

// Local twins of the split-path complex TU's helpers (file-local there by design).
enum class TpRole { Shared, AFresh, BFresh };

TpRole tp_role(const CfEntry& e, int g_a) {
    if (e.current) return TpRole::BFresh;
    return (e.g == g_a) ? TpRole::AFresh : TpRole::Shared;
}

bool tp_a_alive(const CfEntry& e, int g_a) {
    const TpRole r = tp_role(e, g_a);
    return r == TpRole::Shared || (r == TpRole::AFresh && e.delta >= 0);
}

std::string tp_mask_suffix(const CfEntry& e, int g_a, int nA, int nB) {
    const TpRole r = tp_role(e, g_a);
    const char role = (r == TpRole::Shared) ? 'f' : (r == TpRole::AFresh) ? 'a' : 'b';
    return std::string(1, role) + ".d" + std::to_string(e.delta) +
           ".na" + std::to_string(nA) + ".nb" + std::to_string(nB);
}

// tp_qkt_mask_values retargeted to δ-block blk (values identical, placed at
// blk*tH + s instead of block 0).
std::vector<double> tp_dqkt_mask_values(const Inference& inf, const CfEntry& e,
                                        int g_a, int nA, int nB, bool plus, int blk) {
    const int N  = inf.slots;
    const int t  = N / inf.size.hidDim;
    const int tH = t * inf.size.numHeads;
    const int H_real = inf.size.getRealNumHeads();
    const double scale = 0.5 / std::sqrt(static_cast<double>(inf.size.getRealDHead()));
    auto active = [&](bool current, int n_cur, int s) {
        if (s >= tH) return false;
        const int h = s / t, j = s % t;
        if (h >= H_real || j >= n_cur) return false;
        const int l = j - e.delta;
        return l >= 0 && l < (current ? n_cur : t);
    };
    const TpRole role = tp_role(e, g_a);
    const int nblk = N / tH;
    std::vector<double> out(N, 0.0);
    for (int s = 0; s < tH; ++s) {
        const double mA = (role == TpRole::Shared) ? (active(false, nA, s) ? scale : 0.0)
                        : (role == TpRole::AFresh && e.delta >= 0)
                              ? (active(true, nA, s) ? scale : 0.0) : 0.0;
        const double mB = (role == TpRole::BFresh) ? (active(true, nB, s) ? scale : 0.0)
                                                   : (active(false, nB, s) ? scale : 0.0);
        out[blk % nblk * tH + s] = plus ? (mA + mB) : (mA - mB);   // ct-LOCAL block
    }
    return out;
}

// Fused complex QKᵀ into δ-block cts: one matmul pass over the union schedule,
// per-entry two-sided mask p·res + q·conj(res) retargeted to block Δ, accumulated.
std::vector<PackedCtx> qkt_pair_delta(Inference& inf, const PackedCtx& query,
                                      int g_a, int nA, int nB) {
    WithStep _w(inf, "cf.dqkt.tp");
    const int N  = inf.slots;
    const int t  = N / inf.size.hidDim;
    const int tH = t * inf.size.numHeads;
    const int nblk = N / tH;
    const int P = inf.k_count() - inf.n_tok;
    const int G = (t > 0) ? P / t : 0;
    const int n_cts = (P + t + nblk - 1) / nblk;

    const auto& kc = inf.cache[inf.scoped("k")];
    const std::vector<CfEntry> sched = cf_score_schedule(inf);

    PackedCtx q = inf.fhe->clone(query);
    inf.fhe->inplace_mult(q, 0.5);   // half-scale bts input; complex payload keeps q_A + i*q_B
    inf.fhe->bootstrap(q.ct);

    std::vector<PackedCtx> S(n_cts);

    size_t i0 = 0;
    while (i0 < sched.size()) {
        const int g = sched[i0].g;
        size_t i1 = i0;
        while (i1 < sched.size() && sched[i1].g == g) ++i1;

        const PackedCtx& k = kc[g];
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

        for (int s = tH; s < N; s *= 2) {
            PackedCtx rot = inf.fhe->rotate(res, cachemir::mha_rot(inf, s));
            inf.fhe->inplace_add(res, rot);
        }

        const int blk = (G - e.g) * t + e.delta;
        const std::string suff = tp_mask_suffix(e, g_a, nA, nB) + ".b" + std::to_string(blk);
        Ptx p_pt = inf.encode_at_cached(
            "cf.dqkt.mp." + suff, res,
            [&] { return tp_dqkt_mask_values(inf, e, g_a, nA, nB, /*plus=*/true, blk); });
        PackedCtx masked = inf.fhe->mult(res, p_pt);
        if (tp_role(e, g_a) != TpRole::Shared) {   // halves' masks differ: + (mA−mB)·conj(res)
            PackedCtx cres = inf.fhe->conjugate(res);
            Ptx q_pt = inf.encode_at_cached(
                "cf.dqkt.mq." + suff, cres,
                [&] { return tp_dqkt_mask_values(inf, e, g_a, nA, nB, /*plus=*/false, blk); });
            PackedCtx corr = inf.fhe->mult(cres, q_pt);
            inf.fhe->inplace_add(masked, corr);
        }
        const int ci = blk / nblk;
        if (!S[ci].ct) S[ci] = std::move(masked);
        else           inf.fhe->inplace_add(S[ci], masked);
        }
        i0 = i1;
    }

    return S;
}

constexpr const char* kStageA = "cf.dtp.a.";
constexpr const char* kStageB = "cf.dtp.b.";
constexpr const char* kStageZ = "cf.dtp.z.";

int tp_a_count(const std::vector<CfEntry>& sched, int g_a) {
    int n = 0;
    for (const auto& e : sched) n += tp_a_alive(e, g_a) ? 1 : 0;
    return n;
}

}  // namespace

PackedCtx mha_attn_token_pair_delta(Inference& inf, PackedCtx& q_cplx) {
    inf.fhe->level_hint(q_cplx, inf.fhe->level_headroom(3));   // mirror the real attn_core level_hint
    PackedCtx K = std::move(inf.cache[inf.scoped("tp.k")][0]);
    PackedCtx V = std::move(inf.cache[inf.scoped("tp.v")][0]);
    const int nA = inf.n_tok, nB = inf.n_tok_imag;
    const int t  = inf.slots / inf.size.hidDim;

    inf.n_tok = nA;                                           // A (Re) group
    cachemir_filling::cache_k_push(inf, K);
    cachemir_filling::cache_v_push(inf, V);

    PackedCtx out;
    if (nB > 0) {
        const int kcA = inf.k_count();                        // A's causal view: B group not yet visible
        const int g_a = (kcA - nA) / t;                       // A-fresh group index (nA == t when nB > 0)
        inf.n_tok = nB;                                       // B (Im) group
        cachemir_filling::cache_k_push_imag(inf, K);
        cachemir_filling::cache_v_push_imag(inf, V);

        std::vector<PackedCtx> packed = qkt_pair_delta(inf, q_cplx, g_a, nA, nB);
        const std::vector<CfEntry> sched = cf_score_schedule(inf);   // union (B-view) schedule
        const int G = (inf.k_count() - nB) / t;

        // conj_split on the big cts: ONE conjugate per ct serves both halves
        // (per-entry split needed sched.size() of them).
        const int n_cts = static_cast<int>(packed.size());
        std::vector<PackedCtx> A(n_cts), B(n_cts);
        {
            WithStep _ws(inf, "cf.dtp.split");
            for (int ci = 0; ci < n_cts; ++ci) {
                PackedCtx c = inf.fhe->conjugate(packed[ci]);
                A[ci] = inf.fhe->add(packed[ci], c);                       // 2·Re = A rows
                B[ci] = inf.fhe->mult_i(inf.fhe->sub(c, packed[ci]));      // 2·Im = B rows
                packed[ci] = PackedCtx{};
            }
        }

        DeltaView va, vb;
        va.P = kcA - nA;  va.n_cur = nA;  va.block_shift = (G - g_a) * t;
        vb.P = inf.k_count() - nB;  vb.n_cur = nB;  vb.block_shift = 0;
        for (const CfEntry& e : sched) {
            const int blk = (G - e.g) * t + e.delta;
            va.entries.push_back({e.delta, blk, tp_role(e, g_a) == TpRole::AFresh,
                                  tp_a_alive(e, g_a)});
            vb.entries.push_back({e.delta, blk, e.current, true});
        }

        std::vector<PackedCtx> sA =
            attention_softmax_thor_delta_core(inf, std::move(A), "attn", va, kStageA);
        std::vector<PackedCtx> sB =
            attention_softmax_thor_delta_core(inf, std::move(B), "attn", vb, kStageB);

        std::vector<PackedCtx> probs;
        {   // zip: pack the probs per union entry (A-dead entries ride pure-imag)
            WithStep _wz(inf, "cf.dtp.zip");
            StagedEntries zipA(inf, StagedEntries::auto_active(tp_a_count(sched, g_a)), kStageA);
            zipA.adopt(std::move(sA));
            StagedEntries zipB(inf, StagedEntries::auto_active(sched.size()), kStageB);
            zipB.adopt(std::move(sB));
            StagedEntries zipped(inf, StagedEntries::auto_active(sched.size()), kStageZ);
            size_t ai = 0;
            for (size_t idx = 0; idx < sched.size(); ++idx) {
                CKKSContext::MagnitudeReuseScope _rm(*inf.fhe, idx == 0, false);
                PackedCtx pb = zipB.consume(idx);
                if (tp_a_alive(sched[idx], g_a)) {
                    PackedCtx pa = zipA.consume(ai++);
                    zipped.push(inf.fhe->pair_pack(pa, pb));
                } else {
                    zipped.push(inf.fhe->mult_i(pb));
                }
                if (zipped.size() % 32 == 0) zipped.seal();
            }
            probs = zipped.into_vector();
        }
        out = cachemir_filling::softmax_v(inf, std::move(probs), kStageZ);
    } else {                                                  // A-only chunk: the real δ path verbatim
        std::vector<PackedCtx> sA = cachemir_filling::qkt_delta(inf, q_cplx);
        std::vector<PackedCtx> pA =
            cachemir_filling::attention_softmax_thor_delta(inf, std::move(sA), "attn");
        out = cachemir_filling::softmax_v(inf, std::move(pA));
    }
    inf.n_tok = nA;
    inf.cache.erase(inf.scoped("tp.k"));
    inf.cache.erase(inf.scoped("tp.v"));
    return out;
}

}  // namespace cachemir_filling
