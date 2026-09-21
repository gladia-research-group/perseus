#include "packing/cachemir_filling/cachemir_filling_attention.h"
#include "packing/cachemir/cachemir_attention_utils.h"
#include "attention.h"
#include "model/gpt2.h"
#include "inference.h"
#include "staged_entries.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cachemir_filling {

void mha_qkv_token_pair(Inference& inf, PackedCtx& x) {
    const int d = inf.size.hidDim;
    auto qkv = linear_multi(inf, x, {"k", "v", "q"}, d, d,
                            /*stream_pt=*/is_cachemir_filling(inf.packing));
    inf.cache[inf.scoped("tp.k")] = { std::move(qkv[0]) };   // complex K = K_A + i*K_B
    inf.cache[inf.scoped("tp.v")] = { std::move(qkv[1]) };   // complex V
    x = std::move(qkv[2]);                                   // complex Q -> attn op
}

namespace {

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

std::vector<double> tp_qkt_mask_values(const Inference& inf, const CfEntry& e,
                                       int g_a, int nA, int nB, bool plus) {
    const int N  = inf.slots;
    const int t  = N / inf.size.hidDim;
    const int tH = t * inf.size.numHeads;
    const int H_real = inf.size.getRealNumHeads();
    const double scale = 0.5 / std::sqrt(static_cast<double>(inf.size.getRealDHead()));
    auto active = [&](bool current, int n_cur, int s) {   // == mask_values_for_tag's is_active
        if (s >= tH) return false;
        const int h = s / t, j = s % t;
        if (h >= H_real || j >= n_cur) return false;
        const int l = j - e.delta;
        return l >= 0 && l < (current ? n_cur : t);
    };
    const TpRole role = tp_role(e, g_a);
    std::vector<double> out(N, 0.0);
    for (int s = 0; s < tH; ++s) {
        const double mA = (role == TpRole::Shared) ? (active(false, nA, s) ? scale : 0.0)
                        : (role == TpRole::AFresh && e.delta >= 0)
                              ? (active(true, nA, s) ? scale : 0.0) : 0.0;
        const double mB = (role == TpRole::BFresh) ? (active(true, nB, s) ? scale : 0.0)
                                                   : (active(false, nB, s) ? scale : 0.0);
        out[s] = plus ? (mA + mB) : (mA - mB);
    }
    return out;
}

std::vector<PackedCtx> qkt_pair(Inference& inf, const PackedCtx& query,
                                int g_a, int nA, int nB) {
    WithStep _w(inf, "cf.qkt.tp");
    const int N  = inf.slots;
    const int d  = inf.size.hidDim;
    const int H  = inf.size.numHeads;
    const int t  = N / d;
    const int tH = t * H;

    const auto& kc = inf.cache[inf.scoped("k")];
    const std::vector<CfEntry> sched = cf_score_schedule(inf);

    PackedCtx q = inf.fhe->clone(query);
    inf.fhe->inplace_mult(q, 0.5);   // half-scale bts input; complex payload keeps q_A + i*q_B
    inf.fhe->bootstrap(q.ct);

    StagedEntries scores(inf, StagedEntries::auto_active(sched.size()));

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

        const std::string suff = tp_mask_suffix(e, g_a, nA, nB);
        Ptx p_pt = inf.encode_at_cached(
            "cf.qkt.mp." + suff, res,
            [&] { return tp_qkt_mask_values(inf, e, g_a, nA, nB, /*plus=*/true); });
        PackedCtx masked = inf.fhe->mult(res, p_pt);
        if (tp_role(e, g_a) != TpRole::Shared) {   // halves' masks differ: + (mA−mB)·conj(res)
            PackedCtx cres = inf.fhe->conjugate(res);
            Ptx q_pt = inf.encode_at_cached(
                "cf.qkt.mq." + suff, cres,
                [&] { return tp_qkt_mask_values(inf, e, g_a, nA, nB, /*plus=*/false); });
            PackedCtx corr = inf.fhe->mult(cres, q_pt);
            inf.fhe->inplace_add(masked, corr);
        }
        scores.push(std::move(masked));
        if (scores.size() % 32 == 0) scores.seal();
        }
        i0 = i1;
    }

    return scores.into_vector();
}

constexpr const char* kStageA = "cf.tp.a.";
constexpr const char* kStageB = "cf.tp.b.";
constexpr const char* kStageZ = "cf.tp.z.";

int tp_a_count(const std::vector<CfEntry>& sched, int g_a) {
    int n = 0;
    for (const auto& e : sched) n += tp_a_alive(e, g_a) ? 1 : 0;
    return n;
}

void split_scores(Inference& inf, std::vector<PackedCtx>&& packed,
                  const std::vector<CfEntry>& sched, int g_a,
                  std::vector<PackedCtx>& sA, std::vector<PackedCtx>& sB) {
    WithStep _w(inf, "cf.tp.split");
    StagedEntries src(inf, StagedEntries::auto_active(packed.size()));   // == qkt_pair's namespace
    src.adopt(std::move(packed));
    StagedEntries a_out(inf, StagedEntries::auto_active(tp_a_count(sched, g_a)), kStageA);
    StagedEntries b_out(inf, StagedEntries::auto_active(sched.size()), kStageB);
    for (size_t idx = 0; idx < sched.size(); ++idx) {
        CKKSContext::MagnitudeReuseScope _rm(*inf.fhe, idx == 0, false);
        PackedCtx m = src.consume(idx);
        PackedCtx c = inf.fhe->conjugate(m);
        if (tp_a_alive(sched[idx], g_a)) {
            a_out.push(inf.fhe->add(m, c));
            if (a_out.size() % 32 == 0) a_out.seal();
        }
        b_out.push(inf.fhe->mult_i(inf.fhe->sub(c, m)));
        if (b_out.size() % 32 == 0) b_out.seal();
    }
    sA = a_out.into_vector();   // seals+evicts under cf.tp.a. iff its own size warrants staging
    sB = b_out.into_vector();   // seals+evicts under cf.tp.b.
}

}  // namespace

PackedCtx mha_attn_token_pair(Inference& inf, PackedCtx& q_cplx) {
    inf.fhe->level_hint(q_cplx, inf.fhe->level_headroom(3));   // mirror the real attn_core level_hint
    PackedCtx K = std::move(inf.cache[inf.scoped("tp.k")][0]);
    PackedCtx V = std::move(inf.cache[inf.scoped("tp.v")][0]);
    const int nA = inf.n_tok, nB = inf.n_tok_imag;
    const int t  = inf.slots / inf.size.hidDim;

    inf.n_tok = nA;                                           // A (Re) group
    cachemir_filling::cache_k_push(inf, K);                   // fresh_masked_push -> Re(K); k_count += nA
    cachemir_filling::cache_v_push(inf, V);                   // (qualified: else ADL matches the global dispatcher too)

    PackedCtx out;
    if (nB > 0) {
        const int kcA = inf.k_count();                        // A's causal view: B group not yet visible
        const int g_a = (kcA - nA) / t;                       // A-fresh group index (nA == t when nB > 0)
        inf.n_tok = nB;                                       // B (Im) group
        cachemir_filling::cache_k_push_imag(inf, K);          // -0.25i mask -> Im(K); k_count += nB
        cachemir_filling::cache_v_push_imag(inf, V);

        std::vector<PackedCtx> packed = qkt_pair(inf, q_cplx, g_a, nA, nB);
        const std::vector<CfEntry> sched = cf_score_schedule(inf);   // union (B-view) schedule
        std::vector<PackedCtx> sA, sB;
        split_scores(inf, std::move(packed), sched, g_a, sA, sB);

        {   // A softmax under A's pre-B cache view (kc masks must see P = kcA − nA)
            int& kc_live = inf.k_count();
            const int kc_full = kc_live;
            kc_live = kcA;
            inf.n_tok = nA;
            sA = cachemir_filling::attention_softmax_thor(inf, std::move(sA), "attn", kStageA);
            kc_live = kc_full;
        }
        inf.n_tok = nB;
        sB = cachemir_filling::attention_softmax_thor(inf, std::move(sB), "attn", kStageB);

        std::vector<PackedCtx> probs;
        {   // zip: pack the probs per union entry (A-dead entries ride pure-imag)
            WithStep _wz(inf, "cf.tp.zip");
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
    } else {                                                  // A-only chunk: the real path verbatim
        std::vector<PackedCtx> sA = cachemir_filling::qkt(inf, q_cplx);
        sA = cachemir_filling::attention_softmax_thor(inf, std::move(sA), "attn");
        out = cachemir_filling::softmax_v(inf, std::move(sA));
    }
    inf.n_tok = nA;
    inf.cache.erase(inf.scoped("tp.k"));
    inf.cache.erase(inf.scoped("tp.v"));
    return out;
}

}  // namespace cachemir_filling
