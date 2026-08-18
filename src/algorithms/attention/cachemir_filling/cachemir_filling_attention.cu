#include "packing/cachemir_filling/cachemir_filling_attention.h"
#include "packing/cachemir/cachemir_attention_utils.h"
#include "ckks_primitives.h"
#include "nonlinear.h"
#include "inference.h"
#include "staged_entries.h"   // schedule-entry residency (host staging) lives OUTSIDE the op bodies

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cachemir_filling {

// CfEntry lives in cachemir_filling_attention.h — shared with the token-pair fused attention.
std::vector<CfEntry> cf_score_schedule(Inference& inf) {
    return cf_score_schedule(inf, inf.k_count());
}

std::vector<CfEntry> cf_score_schedule(Inference& inf, int K) {
    const int t     = inf.slots / inf.size.hidDim;
    const int n_cur = inf.n_tok;
    std::vector<CfEntry> sched;
    if (inf.bidirectional) {
        for (int g = 0; g * t < K; ++g) {
            const int Lg = std::min(t, K - g * t);
            for (int delta = -(Lg - 1); delta <= n_cur - 1; ++delta)
                sched.push_back({g, delta, Lg, false, true});
        }
        return sched;
    }
    const int P = K - n_cur;
    const int G = (t > 0) ? P / t : 0;
    sched.reserve((size_t)G * (2 * t - 1) + t);
    for (int g = 0; g < G; ++g)
        for (int delta = -(t - 1); delta <= t - 1; ++delta)
            sched.push_back({g, delta, t, false});
    for (int delta = 0; delta <= t - 1; ++delta)
        sched.push_back({G, delta, n_cur, true});
    return sched;
}

namespace {

std::string cf_entry_tag(const CfEntry& e, int n_cur) {
    if (e.bd)
        return "b.d" + std::to_string(e.delta) + ".nt" + std::to_string(n_cur) +
               ".L" + std::to_string(e.Lg);
    return std::string(e.current ? "c" : "f") + ".d" + std::to_string(e.delta) +
           ".nt" + std::to_string(n_cur);
}

std::vector<double> mask_values_checked(const Inference& inf, const SoftmaxConfig* cfg,
                                        const std::string& tag) {
    std::vector<double> v;
    if (!cachemir_filling::mask_values_for_tag(inf, cfg, tag, v))
        throw std::runtime_error("cf mask tag failed to rebuild: " + tag);
    return v;
}

double cf_sm_kc_r(const SoftmaxConfig& cfg, int step, int kc) {
    if (!cfg.sm_kc_r.empty() && cfg.log2delta2 > 0) {
        const int C = static_cast<int>(cfg.sm_kc_r.size()) / cfg.log2delta2;
        const int kpos = std::max(0, std::min(kc - 1, C - 1));
        return cfg.sm_kc_r[step * C + kpos];
    }
    if (kc > 0) { constexpr double kc_ref = 4.0; return std::min(1.0, kc_ref / kc); }
    return 1.0;
}


PackedCtx fresh_recip(Inference& inf, const PackedCtx& D, const PackedCtx& F_init, int iters) {
    PackedCtx R{goldschmidt_recip(inf.cc_ctx(), D.ct, F_init.ct, iters), D.packing};
    inf.fhe->inplace_mult(R, 0.5);     // half-scale bts input (|R|<=~3-10 -> <=~5, EvalMod-safe)
    inf.fhe->bootstrap(R.ct);
    inf.fhe->inplace_im_cleanse(R);    // 2*Re -> R; strips chain+bts imag, pairs the 0.5
    return R;
}

}  // namespace

std::vector<PackedCtx> qkt(Inference& inf, const PackedCtx& query) {
    WithStep _w(inf, "cf.qkt");
    const int N  = inf.slots;
    const int d  = inf.size.hidDim;
    const int H  = inf.size.numHeads;
    const int t  = N / d;
    const int tH = t * H;
    const int n_cur       = inf.n_tok;

    const auto& kc = inf.cache[inf.scoped("k")];
    const std::vector<CfEntry> sched = cf_score_schedule(inf);

    PackedCtx q = inf.fhe->clone(query);
    inf.fhe->inplace_mult(q, 0.25);
    inf.fhe->inplace_im_cleanse(q);
    inf.fhe->bootstrap(q.ct);
    inf.fhe->inplace_im_cleanse(q);

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

        const std::string mtag = "cf.qkt.m." + cf_entry_tag(e, n_cur);
        Ptx m_pt = inf.encode_at_cached(
            mtag, res, [&] { return mask_values_checked(inf, nullptr, mtag); });
        PackedCtx masked = inf.fhe->mult(res, m_pt);
        {
            WithStep _wc(inf, "im_cleanse");
            inf.fhe->inplace_im_cleanse(masked);   // m_pt carries the 0.5 -> Re(score)
        }
        scores.push(std::move(masked));
        }
        i0 = i1;
    }

    return scores.into_vector();
}

std::vector<PackedCtx> attention_softmax_thor(
        Inference& inf, std::vector<PackedCtx> scores, const std::string& cfg_name,
        const std::string& stage_prefix) {
    WithStep _w(inf, "cf.softmax_thor:" + cfg_name);
    const SoftmaxConfig& cfg = inf.sm_cfg.at(cfg_name);

    const int n_cur  = inf.n_tok;
    // causal: keys before this chunk (row-length offset). bidirectional: total key count
    // (uniform kc — the .P tag field carries it into the mask values).
    const int P      = inf.bidirectional ? inf.k_count() : inf.k_count() - n_cur;

    const std::vector<CfEntry> sched = cf_score_schedule(inf);
    const int n = static_cast<int>(sched.size());
    if (static_cast<int>(scores.size()) != n)
        throw std::runtime_error("cf.softmax: scores/schedule length mismatch");

    const double scale_factor = std::pow(2.0, -cfg.log2delta1 - cfg.log2delta2);

    if (cfg.cheb_coeffs.empty())
        throw std::runtime_error(
            "cf.softmax: cfg.cheb_coeffs empty -- Chebyshev exp required (mirror of decode)");

    const bool stage = StagedEntries::auto_active(sched.size());
    StagedEntries sc(inf, stage, stage_prefix);
    sc.adopt(std::move(scores));
    StagedEntries e(inf, stage, stage_prefix);

    auto exp_head = [&](int j) {   // shift + cheb for entry j (everything before the refresh)
        CKKSContext::MagnitudeReuseScope _rm(*inf.fhe, j == 0, false);
        WithStep _wd(inf, "exp_delta");
        const std::string tg = cf_entry_tag(sched[j], n_cur);
        PackedCtx& score = sc.load(j);
        const std::string stag = "cf.sm.shift." + tg;   // clip_lo/mean are per-block -> scoped
        Ptx shift_pt = inf.encode_at_cached(
            inf.scoped(stag), score, [&] { return mask_values_checked(inf, &cfg, stag); });
        PackedCtx ct = inf.fhe->add(score, shift_pt);
        sc.clear(j);   // release this score ct (highest level)
        return eval_chebyshev_series(inf.cc_ctx(), ct, cfg.cheb_coeffs,
                                     cfg.cheb_a / scale_factor,
                                     cfg.cheb_b / scale_factor);
    };
    auto exp_tail = [&](int j, PackedCtx z) {   // squares + cleanse + amask for entry j
        CKKSContext::MagnitudeReuseScope _rm(*inf.fhe, false, false);
        WithStep _wd(inf, "exp_delta");
        for (int i = 0; i < cfg.log2delta1; ++i) inf.fhe->inplace_square(z);
        {
            WithStep _wc(inf, "im_cleanse");
            inf.fhe->inplace_im_cleanse(z);   // amask below carries the 0.5
        }
        const std::string atag = "cf.sm.amask." + cf_entry_tag(sched[j], n_cur) +
                                 ".P" + std::to_string(P);   // block-independent
        Ptx amask_pt = inf.encode_at_cached(
            atag, z, [&] { return mask_values_checked(inf, nullptr, atag); });
        z = inf.fhe->mult(z, amask_pt);
        e.push(std::move(z));
    };
    for (int idx = 0; idx < n; idx += 2) {
        PackedCtx z0 = exp_head(idx);
        if (idx + 1 < n) {
            PackedCtx z1 = exp_head(idx + 1);
            {   // Deliberate option-1 refresh: keep the exp transients in range before squaring.
                WithStep _wb(inf, "exp_refresh");
                inf.fhe->bootstrap_pair(z0.ct, z1.ct);
            }
            exp_tail(idx, std::move(z0));
            exp_tail(idx + 1, std::move(z1));
        } else {
            {
                WithStep _wb(inf, "exp_refresh");
                inf.fhe->bootstrap(z0.ct);
            }
            exp_tail(idx, std::move(z0));
        }
    }
    e.seal();   // phase boundary: one sync, batch-stage, release device copies

    auto sum_over_entries = [&](StagedEntries& v) {
        PackedCtx acc = inf.fhe->clone(v.load(0));
        v.drop(0);
        for (int idx = 1; idx < n; ++idx) {
            inf.fhe->inplace_add(acc, v.load(idx));
            v.drop(idx);
        }
        return acc;
    };

    const std::string ftag = "cf.sm.floor.nt" + std::to_string(n_cur);
    auto make_floor = [&] { return mask_values_checked(inf, nullptr, ftag); };

    PackedCtx sden = sum_over_entries(e);
    Ptx floor_pt = inf.encode_at_cached(ftag, sden, make_floor);
    sden = inf.fhe->add(sden, floor_pt);

    PackedCtx F_init = inf.fhe->mult(sden, -cfg.init_beta);
    inf.fhe->inplace_add(F_init, cfg.init_alpha);

    PackedCtx recip = fresh_recip(inf, sden, F_init, cfg.gs_iters_scaled);
    StagedEntries y(inf, stage, stage_prefix);
    for (int idx = 0; idx < n; ++idx) {
        CKKSContext::MagnitudeReuseScope _rm(*inf.fhe, idx == 0, false);
        PackedCtx yi = inf.fhe->mult(e.load(idx), recip);   // shared denom: 1 mult/entry, not a GS chain
        e.clear(idx);   // e consumed (summed into sden + this product); refine uses y only
        y.push(std::move(yi));
    }
    y.seal();

    for (int r = 0; r < cfg.log2delta2; ++r) {
        WithStep _wr(inf, "refine_iter");
        StagedEntries z2(inf, stage, stage_prefix);
        for (int idx = 0; idx < n; ++idx) {
            CKKSContext::MagnitudeReuseScope _rm(*inf.fhe, idx == 0, false);
            const CfEntry& en = sched[idx];
            PackedCtx& yi = y.load(idx);
            {
                WithStep _wc(inf, "im_cleanse");
                inf.fhe->inplace_im_cleanse(yi);   // cleanse y BEFORE square -> z2 real
            }
            PackedCtx yi2 = inf.fhe->square(yi);
            // per-row z-scale 0.5·√kc·0.25 (the 0.25 absorbs the conj-doubling of y).
            const std::string ztag =
                "cf.sm.zscale." + cf_entry_tag(en, n_cur) + ".P" + std::to_string(P);
            Ptx zscale_pt = inf.encode_at_cached(
                ztag, yi2, [&] { return mask_values_checked(inf, nullptr, ztag); });
            y.clear(idx);   // y consumed (cleansed+squared into z2); rebuilt below
            z2.push(inf.fhe->mult(yi2, zscale_pt));
        }
        z2.seal();

        PackedCtx s2 = sum_over_entries(z2);
        Ptx floor_pt2 = inf.encode_at_cached(ftag, s2, make_floor);
        s2 = inf.fhe->add(s2, floor_pt2);

        // per-row sm_kc_r init: F = refine_alpha·√r - refine_beta·r·s2 (decode: sa=√r, sb=r).
        const std::string rsuffix = std::string(inf.bidirectional ? ".bd" : "") +
                                    ".r" + std::to_string(r) + ".nt" + std::to_string(n_cur) +
                                    ".P" + std::to_string(P);
        const std::string btag = "cf.sm.rbeta" + rsuffix;
        Ptx beta_pt = inf.encode_at_cached(
            inf.scoped(btag), s2, [&] { return mask_values_checked(inf, &cfg, btag); });
        PackedCtx Fr = inf.fhe->mult(s2, beta_pt);

        const std::string aftag = "cf.sm.ralpha" + rsuffix;
        Ptx alpha_pt = inf.encode_at_cached(
            inf.scoped(aftag), Fr, [&] { return mask_values_checked(inf, &cfg, aftag); });
        Fr = inf.fhe->add(Fr, alpha_pt);

        const int it_i = static_cast<int>(cfg.per_step_refine_iters.at(r));
        PackedCtx rrec = fresh_recip(inf, s2, Fr, it_i);
        for (int idx = 0; idx < n; ++idx) {
            PackedCtx yi = inf.fhe->mult(z2.load(idx), rrec);   // shared refined denom: 1 mult/entry
            z2.clear(idx);
            y.set(idx, std::move(yi));   // softmax_v reloads per entry
        }
        y.seal();
    }

    return y.into_vector();
}

PackedCtx softmax_v(Inference& inf, std::vector<PackedCtx> probs,
                    const std::string& stage_prefix) {
    return softmax_v_groups(inf, std::move(probs), inf.cache[inf.scoped("v")],
                            inf.k_count(), stage_prefix);
}

PackedCtx softmax_v_groups(Inference& inf, std::vector<PackedCtx> probs,
                           const std::vector<PackedCtx>& vc, int K,
                           const std::string& stage_prefix) {
    WithStep _w(inf, "cf.softmax_v");
    const int N  = inf.slots;
    const int d  = inf.size.hidDim;
    const int H  = inf.size.numHeads;
    const int t  = N / d;
    const int tH = t * H;

    const std::vector<CfEntry> sched = cf_score_schedule(inf, K);
    if (probs.size() != sched.size())
        throw std::runtime_error("cf.softmax_v: probs/schedule length mismatch");
    StagedEntries pe(inf, StagedEntries::auto_active(sched.size()), stage_prefix);
    pe.adopt(std::move(probs));

    PackedCtx out;
    size_t i0 = 0;
    while (i0 < sched.size()) {
        const int g = sched[i0].g;
        size_t i1 = i0;
        while (i1 < sched.size() && sched[i1].g == g) ++i1;

        const PackedCtx& v = vc[g];
        std::vector<int32_t> steps;
        steps.reserve(i1 - i0);
        for (size_t i = i0; i < i1; ++i)
            if (sched[i].delta != 0) steps.push_back(cachemir::mha_rot(inf, -sched[i].delta));
        std::vector<PackedCtx> vrots = inf.fhe->rotate_hoisted(v, steps);

        size_t r = 0;
        for (size_t idx = i0; idx < i1; ++idx) {
        WithStep _wd(inf, "pv_delta");
        const CfEntry& e = sched[idx];

        PackedCtx vrot = (e.delta == 0)
            ? inf.fhe->clone(v)
            : std::move(vrots[r++]);

        PackedCtx p_bcast = pe.consume(idx);   // load + take the prob ct: no clone, frees the slot
        for (int s = tH; s < N; s *= 2) {
            PackedCtx rot = inf.fhe->rotate(p_bcast, cachemir::mha_rot(inf, -s));
            inf.fhe->inplace_add(p_bcast, rot);
        }

        PackedCtx term = inf.fhe->mult(p_bcast, vrot);
        if (idx == 0) out = std::move(term);
        else inf.fhe->inplace_add(out, term);
        }
        i0 = i1;
    }
    return out;
}

namespace {

size_t parse_entry_tag(const std::string& s, size_t pos,
                       bool& current, int& delta, int& n_cur, bool& bd, int& Lg) {
    if (pos >= s.size() || (s[pos] != 'c' && s[pos] != 'f' && s[pos] != 'b'))
        return std::string::npos;
    current = (s[pos] == 'c');
    bd = (s[pos] == 'b');
    Lg = -1;
    if (s.compare(pos + 1, 2, ".d") != 0) return std::string::npos;
    const char* base = s.c_str();
    char* end = nullptr;
    delta = static_cast<int>(std::strtol(base + pos + 3, &end, 10));
    if (end == base + pos + 3) return std::string::npos;
    size_t i = static_cast<size_t>(end - base);
    if (s.compare(i, 3, ".nt") != 0) return std::string::npos;
    n_cur = static_cast<int>(std::strtol(base + i + 3, &end, 10));
    if (end == base + i + 3) return std::string::npos;
    i = static_cast<size_t>(end - base);
    if (bd) {
        if (s.compare(i, 2, ".L") != 0) return std::string::npos;
        Lg = static_cast<int>(std::strtol(base + i + 2, &end, 10));
        if (end == base + i + 2) return std::string::npos;
        i = static_cast<size_t>(end - base);
    }
    return i;
}

size_t parse_int_field(const std::string& s, size_t pos, const char* key, int& v) {
    const size_t kl = std::strlen(key);
    if (s.compare(pos, kl, key) != 0) return std::string::npos;
    const char* base = s.c_str();
    char* end = nullptr;
    v = static_cast<int>(std::strtol(base + pos + kl, &end, 10));
    if (end == base + pos + kl) return std::string::npos;
    return static_cast<size_t>(end - base);
}

}  // namespace

bool mask_values_for_tag(const Inference& inf, const SoftmaxConfig* cfg,
                         const std::string& tag, std::vector<double>& out) {
    const int N  = inf.slots;
    const int t  = N / inf.size.hidDim;
    const int tH = t * inf.size.numHeads;
    const int H_real = inf.size.getRealNumHeads();

    // Key-lane bound L: causal c-entries L = n_cur, f-entries L = t; bd entries carry Lg in-tag.
    auto is_active = [&](int delta, int n_cur, int L, int s) {
        if (s >= tH) return false;
        const int h = s / t, j = s % t;
        if (h >= H_real || j >= n_cur) return false;
        const int l = j - delta;
        return l >= 0 && l < L;
    };
    auto is_valid_row = [&](int n_cur, int s) {
        return s < tH && s / t < H_real && (s % t) < n_cur;
    };
    // causal: per-row visible-key count. bd: the .P field carries the uniform total.
    auto kc_of_slot = [&](bool bd, int P, int s) { return bd ? P : P + (s % t) + 1; };

    bool current = false, bd = false;
    int delta = 0, n_cur = 0, P = 0, r = 0, Lg = -1;
    const size_t npos = std::string::npos;
    auto lane_bound = [&] { return bd ? Lg : (current ? n_cur : t); };

    if (tag.rfind("cf.qkt.m.", 0) == 0) {
        if (parse_entry_tag(tag, 9, current, delta, n_cur, bd, Lg) != tag.size()) return false;
        const double scale = 0.5 / std::sqrt(static_cast<double>(inf.size.getRealDHead()));
        out.assign(N, 0.0);
        for (int s = 0; s < tH; ++s)
            if (is_active(delta, n_cur, lane_bound(), s)) out[s] = scale;
        return true;
    }
    if (tag.rfind("cf.sm.shift.", 0) == 0) {
        if (!cfg || parse_entry_tag(tag, 12, current, delta, n_cur, bd, Lg) != tag.size())
            return false;
        const double mean = (cfg->clip_hi + cfg->clip_lo) / 2.0;
        out.assign(N, cfg->clip_lo - mean);
        for (int s = 0; s < tH; ++s)
            if (is_active(delta, n_cur, lane_bound(), s)) out[s] = -mean;
        return true;
    }
    if (tag.rfind("cf.sm.amask.", 0) == 0) {
        size_t i = parse_entry_tag(tag, 12, current, delta, n_cur, bd, Lg);
        if (i == npos || parse_int_field(tag, i, ".P", P) != tag.size()) return false;
        out.assign(N, 0.0);
        for (int s = 0; s < tH; ++s)
            if (is_active(delta, n_cur, lane_bound(), s))
                out[s] = 0.5 / static_cast<double>(kc_of_slot(bd, P, s));
        return true;
    }
    if (tag.rfind("cf.sm.zscale.", 0) == 0) {
        size_t i = parse_entry_tag(tag, 13, current, delta, n_cur, bd, Lg);
        if (i == npos || parse_int_field(tag, i, ".P", P) != tag.size()) return false;
        out.assign(N, 0.0);
        for (int s = 0; s < tH; ++s)
            if (is_active(delta, n_cur, lane_bound(), s))
                out[s] = 0.5 * std::sqrt(static_cast<double>(kc_of_slot(bd, P, s))) * 0.25;
        return true;
    }
    if (tag.rfind("cf.sm.floor", 0) == 0) {
        if (parse_int_field(tag, 11, ".nt", n_cur) != tag.size()) return false;
        out.assign(N, 0.0);
        for (int s = 0; s < N; ++s)
            if (!is_valid_row(n_cur, s)) out[s] = 1.0;
        return true;
    }
    const bool rbeta  = tag.rfind("cf.sm.rbeta", 0) == 0;
    const bool ralpha = !rbeta && tag.rfind("cf.sm.ralpha", 0) == 0;
    if (rbeta || ralpha) {
        size_t i = rbeta ? 11 : 12;
        if (tag.compare(i, 3, ".bd") == 0) { bd = true; i += 3; }
        i = parse_int_field(tag, i, ".r", r);
        if (i == npos) return false;
        i = parse_int_field(tag, i, ".nt", n_cur);
        if (i == npos || parse_int_field(tag, i, ".P", P) != tag.size()) return false;
        if (!cfg || r < 0 || r >= static_cast<int>(rbeta ? cfg->refine_beta.size()
                                                         : cfg->refine_alpha.size()))
            return false;
        if (rbeta) {
            out.assign(N, -cfg->refine_beta[r]);
            for (int s = 0; s < tH; ++s)
                if (is_valid_row(n_cur, s))
                    out[s] = -cfg->refine_beta[r] * cf_sm_kc_r(*cfg, r, kc_of_slot(bd, P, s));
        } else {
            out.assign(N, cfg->refine_alpha[r]);
            for (int s = 0; s < tH; ++s)
                if (is_valid_row(n_cur, s))
                    out[s] = cfg->refine_alpha[r] *
                             std::sqrt(cf_sm_kc_r(*cfg, r, kc_of_slot(bd, P, s)));
        }
        return true;
    }
    return false;
}

}  // namespace cachemir_filling
