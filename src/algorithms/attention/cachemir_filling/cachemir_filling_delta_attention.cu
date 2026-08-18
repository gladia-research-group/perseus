#include "packing/cachemir_filling/cachemir_filling_delta_attention.h"
#include "packing/cachemir_filling/cachemir_filling_attention.h"
#include "packing/cachemir/cachemir_attention_utils.h"
#include "ckks_primitives.h"
#include "nonlinear.h"
#include "inference.h"
#include "staged_entries.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cstdlib>

namespace cachemir_filling {

bool delta_block_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("FHE_DELTA_BLOCK");
        return e && e[0] == '1';
    }();
    return v;
}

namespace {

struct DeltaDims { int N; int t; int tH; int nblk; };
DeltaDims delta_dims(const Inference& inf) {
    const int N  = inf.slots;
    const int t  = N / inf.size.hidDim;
    const int tH = t * inf.size.numHeads;
    return {N, t, tH, N / tH};
}

// is_active twin of mask_values_for_tag: one entry's (h,i) pattern within a tH block.
// Lg >= 0 (bidirectional entry) overrides the key-lane bound.
bool entry_slot_active(const Inference& inf, bool current, int delta, int n_cur,
                       int s, int t, int tH, int Lg = -1) {
    if (s >= tH) return false;
    const int h = s / t, j = s % t;
    if (h >= inf.size.getRealNumHeads() || j >= n_cur) return false;
    const int l = j - delta;
    return l >= 0 && l < (Lg >= 0 ? Lg : (current ? n_cur : t));
}

// δ-block union activity: slot (b,h,i) of big ct ci is live iff its absolute
// diagonal D = ci*nblk + b − block_shift satisfies 0 <= D <= P + i (causal), or
// key = i − D in [0, P) (bidirectional; P carries the total key count K).
bool view_slot_active(const Inference& inf, const DeltaView& v, const DeltaDims& dd,
                      int ci, int slot) {
    const int b = slot / dd.tH, s = slot % dd.tH;
    const int h = s / dd.t, i = s % dd.t;
    if (h >= inf.size.getRealNumHeads() || i >= v.n_cur) return false;
    const int D = ci * dd.nblk + b - v.block_shift;
    if (v.bd) { const int key = i - D; return key >= 0 && key < v.P; }
    return D >= 0 && D <= v.P + i;
}

int view_kc(const DeltaView& v, int i) { return v.bd ? v.P : v.P + i + 1; }

std::vector<double> dqkt_mask_vec(const Inference& inf, bool current, int delta,
                                  int n_cur, int blk, int Lg = -1) {
    const auto dd = delta_dims(inf);
    const double scale = 0.5 / std::sqrt(static_cast<double>(inf.size.getRealDHead()));
    std::vector<double> out(dd.N, 0.0);
    for (int s = 0; s < dd.tH; ++s)
        if (entry_slot_active(inf, current, delta, n_cur, s, dd.t, dd.tH, Lg))
            out[blk % dd.nblk * dd.tH + s] = scale;   // ct-LOCAL block (mask targets ct blk/nblk)
    return out;
}

std::vector<double> dsm_shift_vec(const Inference& inf, const SoftmaxConfig& cfg,
                                  const DeltaView& v, int ci) {
    const auto dd = delta_dims(inf);
    const double mean = (cfg.clip_hi + cfg.clip_lo) / 2.0;
    std::vector<double> out(dd.N, cfg.clip_lo - mean);
    for (int slot = 0; slot < dd.N; ++slot)
        if (view_slot_active(inf, v, dd, ci, slot)) out[slot] = -mean;
    return out;
}

std::vector<double> dsm_amask_vec(const Inference& inf, const DeltaView& v, int ci) {
    const auto dd = delta_dims(inf);
    std::vector<double> out(dd.N, 0.0);
    for (int slot = 0; slot < dd.N; ++slot)
        if (view_slot_active(inf, v, dd, ci, slot))
            out[slot] = 0.5 / static_cast<double>(view_kc(v, slot % dd.t));
    return out;
}

std::vector<double> dsm_zscale_vec(const Inference& inf, const DeltaView& v, int ci) {
    const auto dd = delta_dims(inf);
    std::vector<double> out(dd.N, 0.0);
    for (int slot = 0; slot < dd.N; ++slot)
        if (view_slot_active(inf, v, dd, ci, slot))
            out[slot] = 0.5 * std::sqrt(static_cast<double>(view_kc(v, slot % dd.t))) * 0.25;
    return out;
}

// tH-periodic (every block): 1.0 on invalid (h,i) rows so the shared GS recip is
// well-posed there; valid rows carry the row denominator in EVERY block after the
// stride-tH all-reduce (genuinely 512-periodic — see the sim's periodicity assert).
std::vector<double> dsm_floor_vec(const Inference& inf, int n_cur) {
    const auto dd = delta_dims(inf);
    std::vector<double> out(dd.N, 0.0);
    for (int slot = 0; slot < dd.N; ++slot) {
        const int s = slot % dd.tH;
        if (s / dd.t >= inf.size.getRealNumHeads() || s % dd.t >= n_cur) out[slot] = 1.0;
    }
    return out;
}

double dsm_kc_r(const SoftmaxConfig& cfg, int step, int kc) {
    if (!cfg.sm_kc_r.empty() && cfg.log2delta2 > 0) {
        const int C = static_cast<int>(cfg.sm_kc_r.size()) / cfg.log2delta2;
        const int kpos = std::max(0, std::min(kc - 1, C - 1));
        return cfg.sm_kc_r[step * C + kpos];
    }
    if (kc > 0) { constexpr double kc_ref = 4.0; return std::min(1.0, kc_ref / kc); }
    return 1.0;
}

// Denominators run at the D2 = 2·D convention (row-sum cleanse doubles them), so
// the GS init constants carry 1/4 (beta: acts on D2) and 1/2 (alpha): the init
// F' = F/2 keeps the error trajectory e0 = 1 − D2·F' = 1 − D·F IDENTICAL to the
// per-entry path — calibration untouched.
std::vector<double> dsm_refine_vec(const Inference& inf, const SoftmaxConfig& cfg,
                                   const DeltaView& v, int r, bool beta) {
    const auto dd = delta_dims(inf);
    std::vector<double> out(dd.N, beta ? -0.25 * cfg.refine_beta[r]
                                       : 0.5 * cfg.refine_alpha[r]);
    for (int slot = 0; slot < dd.N; ++slot) {
        const int s = slot % dd.tH;
        const int h = s / dd.t, i = s % dd.t;
        if (h >= inf.size.getRealNumHeads() || i >= v.n_cur) continue;
        const double kr = dsm_kc_r(cfg, r, view_kc(v, i));
        out[slot] = beta ? -0.25 * cfg.refine_beta[r] * kr
                         : 0.5 * cfg.refine_alpha[r] * std::sqrt(kr);
    }
    return out;
}

// Per-entry extraction fold: the final refine's zscale values on the entry's own
// active pattern at its δ-block — same op count and level as the per-entry zscale.
std::vector<double> dsm_zext_vec(const Inference& inf, const DeltaView& v,
                                 const DeltaEntry& e) {
    const auto dd = delta_dims(inf);
    std::vector<double> out(dd.N, 0.0);
    for (int s = 0; s < dd.tH; ++s)
        if (entry_slot_active(inf, e.current, e.delta, v.n_cur, s, dd.t, dd.tH, e.Lg))
            out[e.block % dd.nblk * dd.tH + s] =   // ct-LOCAL block (mask applies to Ysq[ci])
                0.5 * std::sqrt(static_cast<double>(view_kc(v, s % dd.t))) * 0.25;
    return out;
}

// fresh_recip twin at the D2 = 2·D convention: GS(D2, F/2) emits 0.5·recip —
// ALREADY the half-scale bts input the per-entry fresh_recip creates with its
// 0.5 pre-mult — and the post-bts cleanse restores the 2×. Pairs the row-sum
// im_cleanse; one mult cheaper than the per-entry twin, level-identical output.
PackedCtx fresh_recip_2x(Inference& inf, const PackedCtx& D2, const PackedCtx& F_init, int iters) {
    PackedCtx R{goldschmidt_recip(inf.cc_ctx(), D2.ct, F_init.ct, iters), D2.packing};
    inf.fhe->bootstrap(R.ct);
    inf.fhe->inplace_im_cleanse(R);
    return R;
}

std::string entry_field_tag(const DeltaEntry& e) {
    return std::string(e.Lg >= 0 ? "b" : (e.current ? "c" : "f")) +
           ".d" + std::to_string(e.delta) + ".b" + std::to_string(e.block) +
           (e.Lg >= 0 ? ".L" + std::to_string(e.Lg) : "");
}

}  // namespace

std::vector<PackedCtx> qkt_delta(Inference& inf, const PackedCtx& query) {
    return qkt_delta_groups(inf, query, inf.cache[inf.scoped("k")], inf.k_count());
}

std::vector<PackedCtx> qkt_delta_groups(Inference& inf, const PackedCtx& query,
                                        const std::vector<PackedCtx>& kc, int K) {
    WithStep _w(inf, "cf.dqkt");
    const auto dd = delta_dims(inf);
    const int n_cur = inf.n_tok;
    const bool bd   = inf.bidirectional;
    const int P     = bd ? K : K - n_cur;
    const int G     = (dd.t > 0) ? P / dd.t : 0;
    const int Gtot  = (dd.t > 0) ? (K + dd.t - 1) / dd.t : 0;
    // shape-fixed ct count (partial chunks keep the full-32 schedule; entries past
    // the real key count carry all-zero masks, exactly like the per-entry path).
    // bd: max block = (Gtot-1)*t + (n_cur-1) + (t-1), blocks query-chunk-independent.
    const int n_cts = bd ? ((Gtot - 1) * dd.t + n_cur + dd.t - 2) / dd.nblk + 1
                         : (P + dd.t + dd.nblk - 1) / dd.nblk;

    const std::vector<CfEntry> sched = cf_score_schedule(inf, K);

    PackedCtx q = inf.fhe->clone(query);
    inf.fhe->inplace_mult(q, 0.25);
    inf.fhe->inplace_im_cleanse(q);
    inf.fhe->bootstrap(q.ct);
    inf.fhe->inplace_im_cleanse(q);

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

        for (int s = dd.tH; s < dd.N; s *= 2) {
            PackedCtx rot = inf.fhe->rotate(res, cachemir::mha_rot(inf, s));
            inf.fhe->inplace_add(res, rot);
        }

        // the all-reduce put the reduced score in EVERY tH-block; the retargeted
        // mask selects block Δ instead of block 0 — the scatter costs nothing.
        const int blk = bd ? (Gtot - 1 - e.g) * dd.t + e.delta + (dd.t - 1)
                           : (G - e.g) * dd.t + e.delta;
        const int Lg = bd ? e.Lg : -1;
        const std::string mtag = "cf.dqkt.m." +
            std::string(bd ? "b" : (e.current ? "c" : "f")) +
            ".d" + std::to_string(e.delta) + ".nt" + std::to_string(n_cur) +
            (bd ? ".L" + std::to_string(Lg) : "") + ".b" + std::to_string(blk);
        Ptx m_pt = inf.encode_at_cached(
            mtag, res, [&] { return dqkt_mask_vec(inf, e.current, e.delta, n_cur, blk, Lg); });
        PackedCtx masked = inf.fhe->mult(res, m_pt);
        {
            WithStep _wc(inf, "im_cleanse");
            inf.fhe->inplace_im_cleanse(masked);   // m_pt carries the 0.5 -> Re(score)
        }
        const int ci = blk / dd.nblk;
        if (!S[ci].ct) S[ci] = std::move(masked);
        else           inf.fhe->inplace_add(S[ci], masked);
        }
        i0 = i1;
    }

    return S;
}

DeltaView bd_delta_view(Inference& inf, int K) {
    const auto dd = delta_dims(inf);
    const int Gtot = (dd.t > 0) ? (K + dd.t - 1) / dd.t : 0;
    DeltaView v;
    v.n_cur       = inf.n_tok;
    v.bd          = true;
    v.P           = K;                     // uniform kc + key-range bound
    v.block_shift = Gtot * dd.t - 1;       // key = i - (D̂ - shift)
    for (const CfEntry& e : cf_score_schedule(inf, K))
        v.entries.push_back({e.delta,
                             (Gtot - 1 - e.g) * dd.t + e.delta + (dd.t - 1),
                             false, true, e.Lg});
    return v;
}

std::vector<PackedCtx> attention_softmax_thor_delta(
        Inference& inf, std::vector<PackedCtx> scores, const std::string& cfg_name,
        const std::string& stage_prefix) {
    const auto dd = delta_dims(inf);
    DeltaView v;
    v.n_cur = inf.n_tok;
    if (inf.bidirectional) {
        return attention_softmax_thor_delta_core(inf, std::move(scores), cfg_name,
                                                 bd_delta_view(inf, inf.k_count()),
                                                 stage_prefix);
    }
    v.P           = inf.k_count() - inf.n_tok;
    v.block_shift = 0;
    const int G = (dd.t > 0) ? v.P / dd.t : 0;
    for (const CfEntry& e : cf_score_schedule(inf))
        v.entries.push_back({e.delta, (G - e.g) * dd.t + e.delta, e.current, true});
    return attention_softmax_thor_delta_core(inf, std::move(scores), cfg_name, v,
                                             stage_prefix);
}

std::vector<PackedCtx> attention_softmax_thor_delta_core(
        Inference& inf, std::vector<PackedCtx> S, const std::string& cfg_name,
        const DeltaView& view, const std::string& stage_prefix) {
    WithStep _w(inf, "cf.dsm.softmax_thor:" + cfg_name);
    const SoftmaxConfig& cfg = inf.sm_cfg.at(cfg_name);
    const auto dd = delta_dims(inf);
    const int n_cts = static_cast<int>(S.size());

    const double scale_factor = std::pow(2.0, -cfg.log2delta1 - cfg.log2delta2);
    if (cfg.cheb_coeffs.empty())
        throw std::runtime_error("cf.dsm: cfg.cheb_coeffs empty -- Chebyshev exp required");
    if (cfg.log2delta2 < 1)
        throw std::runtime_error("cf.dsm: log2delta2 >= 1 required (per-entry extraction "
                                 "folds into the final refine)");

    const std::string vtag = ".nt" + std::to_string(view.n_cur) +
                             ".P" + std::to_string(view.P) +
                             ".sh" + std::to_string(view.block_shift);

    // exp chain ONCE per big ct (vs once per schedule entry)
    std::vector<PackedCtx> E(n_cts);
    for (int ci = 0; ci < n_cts; ++ci) {
        CKKSContext::MagnitudeReuseScope _rm(*inf.fhe, ci == 0, false);
        WithStep _wd(inf, "exp_delta");
        const std::string stag = "cf.dsm.shift" + vtag + ".ci" + std::to_string(ci);
        Ptx shift_pt = inf.encode_at_cached(
            inf.scoped(stag), S[ci], [&] { return dsm_shift_vec(inf, cfg, view, ci); });
        PackedCtx ct = inf.fhe->add(S[ci], shift_pt);
        S[ci] = PackedCtx{};
        PackedCtx z = eval_chebyshev_series(inf.cc_ctx(), ct, cfg.cheb_coeffs,
                                            cfg.cheb_a / scale_factor,
                                            cfg.cheb_b / scale_factor);
        {
            WithStep _wb(inf, "exp_refresh");
            inf.fhe->bootstrap(z.ct);
        }
        for (int i = 0; i < cfg.log2delta1; ++i) inf.fhe->inplace_square(z);
        {
            WithStep _wc(inf, "im_cleanse");
            inf.fhe->inplace_im_cleanse(z);   // amask below carries the 0.5
        }
        const std::string atag = "cf.dsm.amask" + vtag + ".ci" + std::to_string(ci);
        Ptx amask_pt = inf.encode_at_cached(
            atag, z, [&] { return dsm_amask_vec(inf, view, ci); });
        E[ci] = inf.fhe->mult(z, amask_pt);
    }

    // row-sum over diagonals: log2(N/tH) stride-tH rotations per ct + cross-ct adds
    // (replaces the per-entry free adds; the result is broadcast to every block).
    auto row_sum = [&](const std::vector<PackedCtx>& v) {
        PackedCtx acc;
        for (int ci = 0; ci < n_cts; ++ci) {
            PackedCtx r = inf.fhe->clone(v[ci]);
            for (int s = dd.tH; s < dd.N; s *= 2) {
                PackedCtx rot = inf.fhe->rotate(r, cachemir::mha_rot(inf, s));
                inf.fhe->inplace_add(r, rot);
            }
            if (ci == 0) acc = std::move(r);
            else         inf.fhe->inplace_add(acc, r);
        }
        return acc;
    };

    const std::string ftag = "cf.dsm.floor.nt" + std::to_string(view.n_cur);
    auto make_floor = [&] { return dsm_floor_vec(inf, view.n_cur); };

    PackedCtx sden = row_sum(E);
    Ptx floor_pt = inf.encode_at_cached(ftag, sden, make_floor);
    sden = inf.fhe->add(sden, floor_pt);
    {   // strip the row-sum rotations' imaginary keyswitch noise BEFORE the GS
        // chain squares it into Re (per-entry sums are noise-free adds). The 2×
        // rides the D2 convention (constants + fresh_recip_2x below).
        WithStep _wc(inf, "im_cleanse");
        inf.fhe->inplace_im_cleanse(sden);
    }

    PackedCtx F_init = inf.fhe->mult(sden, -0.25 * cfg.init_beta);
    inf.fhe->inplace_add(F_init, 0.5 * cfg.init_alpha);

    PackedCtx recip = fresh_recip_2x(inf, sden, F_init, cfg.gs_iters_scaled);
    std::vector<PackedCtx> Y(n_cts);
    for (int ci = 0; ci < n_cts; ++ci) {
        Y[ci] = inf.fhe->mult(E[ci], recip);
        E[ci] = PackedCtx{};
    }

    for (int r = 0; r < cfg.log2delta2; ++r) {
        WithStep _wr(inf, "refine_iter");
        const bool last = (r + 1 == cfg.log2delta2);
        std::vector<PackedCtx> Z2(n_cts), Ysq(n_cts);
        for (int ci = 0; ci < n_cts; ++ci) {
            CKKSContext::MagnitudeReuseScope _rm(*inf.fhe, ci == 0, false);
            {
                WithStep _wc(inf, "im_cleanse");
                inf.fhe->inplace_im_cleanse(Y[ci]);   // cleanse y BEFORE square
            }
            PackedCtx ysq = inf.fhe->square(Y[ci]);
            Y[ci] = PackedCtx{};
            const std::string ztag = "cf.dsm.zsc" + vtag + ".ci" + std::to_string(ci);
            Ptx zscale_pt = inf.encode_at_cached(
                ztag, ysq, [&] { return dsm_zscale_vec(inf, view, ci); });
            Z2[ci] = inf.fhe->mult(ysq, zscale_pt);
            if (last) Ysq[ci] = std::move(ysq);   // extraction reads the pre-zscale square
        }

        PackedCtx s2 = row_sum(Z2);
        Ptx floor_pt2 = inf.encode_at_cached(ftag, s2, make_floor);
        s2 = inf.fhe->add(s2, floor_pt2);
        {
            WithStep _wc(inf, "im_cleanse");
            inf.fhe->inplace_im_cleanse(s2);   // D2 convention, as sden above
        }

        const std::string rsuffix = ".r" + std::to_string(r) + vtag;
        const std::string btag = "cf.dsm.rbeta" + rsuffix;
        Ptx beta_pt = inf.encode_at_cached(
            inf.scoped(btag), s2, [&] { return dsm_refine_vec(inf, cfg, view, r, true); });
        PackedCtx Fr = inf.fhe->mult(s2, beta_pt);

        const std::string aftag = "cf.dsm.ralpha" + rsuffix;
        Ptx alpha_pt = inf.encode_at_cached(
            inf.scoped(aftag), Fr, [&] { return dsm_refine_vec(inf, cfg, view, r, false); });
        Fr = inf.fhe->add(Fr, alpha_pt);

        const int it_i = static_cast<int>(cfg.per_step_refine_iters.at(r));
        PackedCtx rrec = fresh_recip_2x(inf, s2, Fr, it_i);

        if (!last) {
            for (int ci = 0; ci < n_cts; ++ci) {
                Y[ci] = inf.fhe->mult(Z2[ci], rrec);
                Z2[ci] = PackedCtx{};
            }
            continue;
        }

        // FINAL refine: per-entry extraction folded into the zscale mask — z2_e at
        // the exact chain position and level of the per-entry path's zscale, then
        // y_e = z2_e·rrec. softmax_v consumes the result unchanged (its stride-tH
        // broadcast all-reduce is position-agnostic — block Δ broadcasts like block 0).
        size_t n_alive = 0;
        for (const DeltaEntry& e : view.entries) n_alive += e.alive ? 1 : 0;
        StagedEntries y(inf, StagedEntries::auto_active(n_alive), stage_prefix);
        for (const DeltaEntry& e : view.entries) {
            if (!e.alive) continue;
            CKKSContext::MagnitudeReuseScope _rm(*inf.fhe, y.size() == 0, false);
            const int ci = e.block / dd.nblk;
            const std::string etag = "cf.dsm.zext." + entry_field_tag(e) + vtag;
            Ptx zext_pt = inf.encode_at_cached(
                etag, Ysq[ci], [&] { return dsm_zext_vec(inf, view, e); });
            PackedCtx z2e = inf.fhe->mult(Ysq[ci], zext_pt);
            y.push(inf.fhe->mult(z2e, rrec));
            if (y.size() % 32 == 0) y.seal();
        }
        return y.into_vector();
    }
    throw std::runtime_error("cf.dsm: unreachable");
}

}  // namespace cachemir_filling
