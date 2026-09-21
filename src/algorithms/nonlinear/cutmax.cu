#include "cutmax.h"

#include "ckks_primitives.h"
#include "fideslib_wrapper.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kCutmaxHintLevel = 16;

int cutmax_vec_bts_iters() {
    static const int v = [] {
        const char* e = std::getenv("CUTMAX_VEC_BTS_ITERS");
        return (e && *e) ? std::atoi(e) : 0;
    }();
    return v;
}

// per-pass u bootstrap respects the active cascade scope instead of hard 2-iter
bool cutmax_precise_scoped() {
    static const bool on = [] {
        const char* v = std::getenv("CUTMAX_PRECISE_SCOPED");
        return v && *v && *v != '0';
    }();
    return on;
}

bool cutmax_sparse_bts() {
    static const bool on = [] {
        const char* v = std::getenv("CUTMAX_SPARSE_BTS");
        if (!(v && *v && *v != '0')) return false;

        const char* a = std::getenv("FIDESLIB_SPARSE_ARCSINE");
        if (!(a && *a && *a != '0')) {

            const char* o = std::getenv("CUTMAX_SPARSE_NO_ARCSINE");
            if (o && *o && *o != '0') {
                std::fprintf(stderr,
                    "[cutmax]  CUTMAX_SPARSE_NO_ARCSINE=1: routing cutmax bootstraps sparse "
                    "WITHOUT the arcsine correction. The config's T5 cascade schedule was "
                    "calibrated WITH it — this is OUTSIDE its calibrated envelope. "
                    "Measurement only.\n");
                return true;
            }
            std::fprintf(stderr,
                "[cutmax] CUTMAX_SPARSE_BTS ignored: FIDESLIB_SPARSE_ARCSINE not armed "
                "(sparse cutmax routing is arcsine-envelope-calibrated; "
                "set CUTMAX_SPARSE_NO_ARCSINE=1 to measure without it)\n");
            return false;
        }
        return true;
    }();
    return on;
}

PackedCtx rotsum_all(Inference& inf, const PackedCtx& x) {
    return rotate_and_sum_all(inf.cc_ctx(), x, inf.slots);
}

// one line per argmax: bootstraps per stage (entry, i0..iN, sum) — the cost map
void print_bts_marks(const std::vector<uint64_t>& m) {
    std::string s = "[cutmax_bts]";
    for (size_t k = 1; k < m.size(); ++k) {
        const std::string tag = (k == 1) ? "entry"
                              : (k + 1 == m.size()) ? "sum"
                              : "i" + std::to_string(k - 2);
        s += " " + tag + "=" + std::to_string(m[k] - m[k - 1]);
    }
    s += " total=" + std::to_string(m.back() - m.front());
    std::fprintf(stderr, "%s\n", s.c_str());
    std::fflush(stderr);
}

PackedCtx rotsum_tiles(Inference& inf, const std::vector<PackedCtx>& tiles) {
    PackedCtx s = rotsum_all(inf, tiles[0]);
    for (size_t k = 1; k < tiles.size(); ++k)
        inf.fhe->inplace_add(s, rotsum_all(inf, tiles[k]));
    return s;
}

PackedCtx make_cutmax_ones(Inference& inf, const PackedCtx& x_in) {
    auto& F = *inf.fhe;
    if (F.const_one_available())
        return F.tagged(F.const_one_clone(inf.slots),
                        x_in.packing, packtag::PackTag::constant(inf.slots));
    PackedCtx ones = F.sub(F.add(x_in, 1.0), x_in);
    return F.tagged(ones.ct, x_in.packing, packtag::PackTag::constant(inf.slots));
}

PackedCtx inv_sigma_cascade(Inference& inf, const PackedCtx& x_in,
                            const CutMaxConfig& cfg,
                            const CutMaxConfig::Iter& it) {
    auto& F = *inf.fhe;

    WithStep _c(inf, "cascade");
    CKKSContext::BtsItersScope scalar_scope(
        F, it.casc_iters > 0 ? static_cast<uint32_t>(it.casc_iters) : 2);
    CKKSContext::SparseBtsScope sparse_scope(F, cutmax_sparse_bts());
    PackedCtx x = x_in;

    PackedCtx ones;
    { WithStep _o(inf, "ones_seed"); ones = make_cutmax_ones(inf, x_in); }
    PackedCtx u_prod;
    for (int j = 0; j < it.passes; ++j) {
        WithStep _p(inf, "p" + std::to_string(j));
        const int iters = cfg.newton_per_pass +
                          (j + 1 == it.passes ? cfg.newton_polish : 0);
        PackedCtx y0 = (j == 0 && it.ca != 0.0)
            ? F.add(F.mult(x, -it.cb), it.ca)
            : ones;
        PackedCtx u = inv_sqrt_newton_safe(inf.cc_ctx(), x, y0, iters);
        if (j + 1 < it.passes)
            x = F.mult(F.mult(x, u), u);

        if (cutmax_precise_scoped())
            F.bootstrap(u.ct);
        else
            F.bootstrap_precise(u.ct);
        u_prod = (j == 0) ? u : F.mult(u_prod, u);
    }
    return F.mult(u_prod, 0.5 / (std::sqrt(it.s2_hi) * it.c * it.m));
}

template <class Gen>
static Ptx cutmax_cached_pt(Inference& inf, const std::string& tag, uint32_t lv, bool tagged,
                            Gen&& gen) {
    const std::string key = Inference::enc_cache_key(tag, lv, 1);
    auto it = inf.enc_cache.find(key);
    if (it != inf.enc_cache.end()) { ++inf.enc_cache_hit; return it->second; }
    ++inf.enc_cache_miss;
    const auto vals = gen();
    Ptx pt = tagged ? inf.encode_tagged(vals, lv)
                    : encode(inf.fhe->cc, vals, static_cast<int>(lv));
    inf.enc_cache.emplace(key, pt);
    return pt;
}

PackedCtx cutmax_packed(Inference& inf, const PackedCtx& pair_in,
                        int vocab, const CutMaxConfig& cfg,
                        const std::vector<std::vector<double>>& mask) {
    auto& F = *inf.fhe;
    const int n = vocab;
    const std::string cfg_fp = [&] {
        size_t h = std::hash<double>{}(cfg.entry_scale) ^ (static_cast<size_t>(vocab) << 1);
        for (const auto& it : cfg.iters) {
            h ^= std::hash<double>{}(it.s2_hi) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<double>{}(it.m)     + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(it.p)     + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        char buf[20];
        std::snprintf(buf, sizeof buf, ".%llx",
                      static_cast<unsigned long long>(h & 0xffffffffULL));
        return std::string(buf);
    }();
    std::vector<uint64_t> bts_marks{F.total_bootstraps};


    // lanes: A real-axis (t0), B imag-axis (i*sgn*t1), canonical values
    PackedCtx pin, A, B;
    { WithStep _e(inf, "entry");
      pin = F.mult(pair_in, cfg.entry_scale);
      F.bootstrap_hint(pin.ct, kCutmaxHintLevel);
      A = F.add(pin, F.conjugate(pin));                      // 2*t0
      B = F.sub(pin, F.conjugate(pin));                      // 2i*t1
      A = F.mult(A, 0.5);                                    // entry only:

      const uint32_t _blv = static_cast<uint32_t>(level_of(B.ct)) + inf.pending_rescale_primes(B.ct);
      Ptx bmask_pt = cutmax_cached_pt(inf, "cutmax.bmask" + cfg_fp, _blv, /*tagged=*/true, [&]() {
          std::vector<double> bmask = mask[1];
          for (auto& x : bmask) x *= 0.5;
          return bmask;
      });
      B = F.mult(B, bmask_pt);                               // -> canonical, junk-free
    }
    bts_marks.push_back(F.total_bootstraps);
    double sgn = 1.0;

    for (size_t ii = 0; ii < cfg.iters.size(); ++ii) {
        const auto& it = cfg.iters[ii];
        WithStep _i(inf, "i" + std::to_string(ii));
        const double kf = 1.0 / std::sqrt(static_cast<double>(n) *
                                          it.s2_hi);
        // packed current state (free add: B lives on the imag axis)
        PackedCtx P = F.add(A, B);
        PackedCtx R_c = rotsum_all(inf, P);                  // S0 + i*sgn*S1
        Ptx ext = inf.encode_complex_const_at(0.5, -0.5 * sgn, R_c.ct);
        PackedCtx R = F.im_cleanse(F.mult(R_c, ext));        // S0 + S1

        PackedCtx kA = F.mult(A, kf);
        PackedCtx kB = F.mult(B, kf);
        if (!it.ex2) {
            const uint32_t _rlv = static_cast<uint32_t>(level_of(R.ct))
                                + inf.pending_rescale_primes(R.ct);
            Ptx m0 = cutmax_cached_pt(inf, "cutmax.m0.i" + std::to_string(ii) + cfg_fp, _rlv,
                                      /*tagged=*/true, [&]() {
                std::vector<double> mu0 = mask[0];
                for (auto& x : mu0) x *= kf / n;
                return mu0;
            });
            kA = F.sub(kA, F.mult(R, m0));

            Ptx m1 = cutmax_cached_pt(inf, "cutmax.m1.i" + std::to_string(ii) + cfg_fp, _rlv,
                                      /*tagged=*/false, [&]() {
                std::vector<std::complex<double>> mu1c(mask[1].size());
                for (size_t j = 0; j < mask[1].size(); ++j)
                    mu1c[j] = {0.0, sgn * mask[1][j] * kf / n};
                return mu1c;
            });
            kB = F.sub(kB, F.mult(R, m1));
        }
        PackedCtx S2 = rotsum_all(inf, F.sub(F.square(kA), F.square(kB)));
        PackedCtx f = inv_sigma_cascade(inf, S2, cfg, it);   // 0.5/(c*m) fold

        PackedCtx Rf = F.mult(R, f);
        PackedCtx a = F.mult(P, f);
        Ptx b_pt = cutmax_cached_pt(
            inf, "cutmax.b.i" + std::to_string(ii) + cfg_fp,
            static_cast<uint32_t>(level_of(Rf.ct)) + inf.pending_rescale_primes(Rf.ct),
            /*tagged=*/false, [&]() {
                std::vector<std::complex<double>> bc(mask[0].size());
                for (size_t j = 0; j < mask[0].size(); ++j)
                    bc[j] = {mask[0][j] / n, sgn * mask[1][j] / n};
                return bc;
            });
        PackedCtx b = F.mult(Rf, b_pt);
        Ptx sh_pt = cutmax_cached_pt(
            inf, "cutmax.sh.i" + std::to_string(ii) + cfg_fp,
            static_cast<uint32_t>(level_of(a.ct)) + inf.pending_rescale_primes(a.ct),
            /*tagged=*/false, [&]() {
                std::vector<std::complex<double>> sh(mask[0].size());
                for (size_t j = 0; j < mask[0].size(); ++j)
                    sh[j] = {mask[0][j] * 0.5 / it.m, sgn * mask[1][j] * 0.5 / it.m};
                return sh;
            });
        PackedCtx yP = F.add(F.sub(a, b), sh_pt);
        F.bootstrap_hint(yP.ct, kCutmaxHintLevel);   // ONE packed vector bts

        // deg-preserving unpack -> canonical lanes; odd power per lane
        A = pow_odd(inf.cc_ctx(), F.add(yP, F.conjugate(yP)), it.p);          // y0^p
        B = pow_odd(inf.cc_ctx(), F.sub(yP, F.conjugate(yP)), it.p);  // i^p*sgn*y1^p
        sgn *= (it.p % 4 == 1) ? 1.0 : -1.0;
        bts_marks.push_back(F.total_bootstraps);
    }

    // normalize: one packed reduction for the cross-lane sum
    PackedCtx Z;
    { WithStep _s(inf, "sum");
      PackedCtx P = F.add(A, B);
      PackedCtx S_c = rotsum_all(inf, P);
      Ptx ext = inf.encode_complex_const_at(0.5, -0.5 * sgn, S_c.ct);
      PackedCtx S = F.im_cleanse(F.mult(S_c, ext));
      const double g = 1.0 / std::sqrt(cfg.sum_lo * cfg.sum_hi);
      const double lo = cfg.sum_lo * g, hi = cfg.sum_hi * g;
      const double bsum = 8.0 / ((lo + hi) * (lo + hi) + 4.0 * lo * hi);
      const double alpha = bsum * (lo + hi);
      const double beta = bsum;
      PackedCtx r;
      { // broadcast-scalar segment: sparse-routable; Z (vector) stays outside
        CKKSContext::SparseBtsScope ss(F, cutmax_sparse_bts());
        PackedCtx Sn = F.mult(S, g);
        PackedCtx F_init = F.add(F.mult(Sn, -beta), alpha);
        r = goldschmidt_inv(inf.cc_ctx(), Sn, F_init, cfg.gs_sum_iters);
        r = F.mult(r, g);
      }
      // packed Z, canonical sign: fold sgn into the imag lane before return
      Z = F.mult(P, r);
      if (sgn < 0) {
          // realign: Z = t0 - i*t1 -> conj gives t0 + i*t1
          Z = F.conjugate(Z);
      }
    }
    bts_marks.push_back(F.total_bootstraps);
    print_bts_marks(bts_marks);
    return Z;
}

}  // namespace

CutMaxConfig default_gpt2_cutmax_config() {
    CutMaxConfig cfg;

    static const bool sparse_t5 = [] {
        const char* s = std::getenv("SPARSE_BTS_SLOTS");
        const char* a = std::getenv("FIDESLIB_SPARSE_ARCSINE");
        return cutmax_sparse_bts() && s && std::atoi(s) > 0 && a && *a && std::atoi(a) != 0;
    }();
    if (sparse_t5) {
        cfg.iters = {
            {9, 5.0, 0.381, 4.332795841420349e-4, 1, 0, 1.5051726133745356, 0.5051903772975017, 1},
            {3, 10.0, 2.997, 1.0551777780617426e+10, 2, 0, 0.9604179374470168, 0.13124323428218956, 1},
            {3, 10.0, 3.366, 6.450236359333247e-2, 3, 0, 0.3098089752721831, 0.004405319808908402, 1},
            {3, 10.0, 3.366, 7.427507058581279e-2, 3, 0, 0.33244606024757883, 0.005443265958490995, 1},
            {7, 10.0, 26.931, 0.410262961915094, 2, 0, 0.769537874098511, 0.06751281309075567, 1},
        };
        cfg.newton_per_pass = 5;
        cfg.newton_polish   = 0;
        cfg.gs_sum_iters    = 6;
        cfg.sum_lo          = 0.006374643501355827;
        cfg.sum_hi          = 0.414769321004312;
        cfg.entry_scale     = 0.00390625;
        std::fprintf(stderr, "[cutmax_cfg] default schedule: sparse-T5 (arcsine envelope armed)\n");
        return cfg;
    }
    static const bool bts1 = [] {
        const char* v = std::getenv("CUTMAX_BTS_ITERS");
        return v && *v && std::atoi(v) == 1;   // default = 2-iter T=6 (production)
    }();
    if (bts1) {
        cfg.iters = {
            {9, 5.0, 0.381, 5.487259e-4, 1, 0},   // s2_hi = 35.96133*kEntryScale^2
            {3, 10.0, 2.997, 3.645026e+10, 2, 0},
            {3, 10.0, 3.366, 2.255540, 3, 0},
            {7, 10.0, 3.366, 2.253690, 3, 0},
            {9, 10.0, 3.366, 1.238384e+07, 4, 0},
            {9, 10.0, 3.366, 2.901263e+10, 2, 0},
            {9, 10.0, 26.931, 2.901263e+10, 1, 0},
        };
        cfg.sum_lo = 0.25;   // oracle sum_band @2e-3 = [0.268, 0.305] + margin
        cfg.sum_hi = 0.33;
    } else {
        cfg.iters = {
            {9, 5.0, 0.381, 5.487259e-4, 1, 0},   // s2_hi = 35.9613*2^-16 (= *kEntryScale^2)
            {3, 10.0, 2.997, 3.64061e+10, 2, 0},
            {3, 10.0, 3.366, 2.25507, 3, 0},
            {3, 10.0, 3.366, 2.25369, 3, 0},
            {9, 10.0, 3.366, 2.25701, 2, 1},
            {19, 10.0, 26.93, 2.88937e+10, 3, 1},
        };
        cfg.sum_lo = 0.0183;  // final-sum band (chord-init GS converges over
        cfg.sum_hi = 0.0703;  // the whole band, worst init error 0.46)
    }
    return cfg;
}

CutMaxConfig cutmax_config_from_calib(const CutMaxCalib& c) {
    const size_t T = c.p.size();
    if (!T || c.c.size() != T || c.m.size() != T || c.s2_hi.size() != T ||
        c.passes.size() != T || c.ex2.size() != T ||
        c.chord_a.size() != T || c.chord_b.size() != T ||
        c.cascade_iters.size() != T)
        throw std::runtime_error("cutmax calib: ragged/empty schedule arrays");
    CutMaxConfig cfg;
    for (size_t i = 0; i < T; ++i)
        cfg.iters.push_back({static_cast<int>(c.p[i]), c.c[i], c.m[i],
                             c.s2_hi[i], static_cast<int>(c.passes[i]),
                             static_cast<int>(c.ex2[i]),
                             c.chord_a[i], c.chord_b[i],
                             static_cast<int>(c.cascade_iters[i])});
    cfg.newton_per_pass = c.newton_per_pass;
    cfg.newton_polish   = c.newton_polish;
    cfg.gs_sum_iters    = c.gs_sum_iters;
    cfg.sum_lo          = c.sum_lo;
    cfg.sum_hi          = c.sum_hi;
    cfg.entry_scale     = c.entry_scale;
    std::fprintf(stderr,
        "[cutmax_cfg] configs.json schedule: T=%zu entry_scale=%g sum=[%g,%g]\n",
        T, cfg.entry_scale, cfg.sum_lo, cfg.sum_hi);
    return cfg;
}

std::vector<PackedCtx> cutmax_argmax(Inference& inf,
                                     const std::vector<PackedCtx>& tiles_in,
                                     int vocab, const CutMaxConfig& cfg) {
    auto& F = *inf.fhe;
    const int W_tile = inf.slots;
    const int d = inf.size.hidDim;

    static const uint32_t cm_iters = [] {
        const char* v = std::getenv("CUTMAX_BTS_ITERS");
        return (v && *v) ? static_cast<uint32_t>(std::atoi(v)) : 2u;
    }();
    const uint32_t vec_iters = cutmax_vec_bts_iters() > 0
        ? static_cast<uint32_t>(cutmax_vec_bts_iters()) : cm_iters;
    CKKSContext::BtsItersScope bts_scope(F, vec_iters);

    static const bool cm_arcsine = [] {
        const char* v = std::getenv("CUTMAX_ARCSINE");
        return v && *v && *v != '0';
    }();
    static const uint32_t cm_prec = [] {
        const char* v = std::getenv("CUTMAX_BTS_PRECISION");
        return (v && *v) ? static_cast<uint32_t>(std::atoi(v)) : 0u;
    }();
    std::optional<CKKSContext::ArcsineScope> arc_scope;
    if (cm_arcsine)
        arc_scope.emplace(true);
    std::optional<CKKSContext::BtsPrecisionScope> prec_scope;
    if (cm_prec)
        prec_scope.emplace(F, cm_prec);

    const int K_logical = (vocab + W_tile - 1) / W_tile;
    std::vector<std::vector<double>> mask(K_logical);
    for (int k = 0; k < K_logical; ++k) {
        const int wreal = std::min(W_tile, vocab - k * W_tile);
        mask[k].assign(static_cast<size_t>(W_tile), 0.0);
        for (int m = 0; m < W_tile; ++m)
            if (cutmax_tile_col_of_slot(m, d, W_tile) < wreal)
                mask[k][m] = 1.0;
    }

    const bool packed_in = static_cast<int>(tiles_in.size()) < K_logical;
    if (inf.complex && !packed_in)
        throw std::runtime_error(
            "cutmax_argmax: complex arm expects the packed lm_head pair");
    if (packed_in && !F.complex_payload)
        throw std::runtime_error(
            "cutmax_argmax: packed pair requires CKKS_COMPLEX=1");
    if (packed_in) {
        if (K_logical != 2)
            throw std::runtime_error(
                "cutmax_argmax: packed path supports exactly 2 logical tiles");
        return { cutmax_packed(inf, tiles_in[0], vocab, cfg, mask) };
    }

    const int K = static_cast<int>(tiles_in.size());
    std::vector<uint64_t> bts_marks{F.total_bootstraps};

    std::vector<PackedCtx> tiles;
    { WithStep _e(inf, "entry");
      for (int k = 0; k < K; ++k) {
          std::vector<double> emask = mask[k];
          for (auto& x : emask) x *= cfg.entry_scale;
          Ptx emask_pt = inf.encode_at(emask, tiles_in[k]);
          PackedCtx x = F.mult(tiles_in[k], emask_pt);
          F.bootstrap_hint(x.ct, kCutmaxHintLevel);
          tiles.push_back(std::move(x));
      }
    }
    bts_marks.push_back(F.total_bootstraps);

    for (size_t ii = 0; ii < cfg.iters.size(); ++ii) {
        const auto& it = cfg.iters[ii];
        WithStep _i(inf, "i" + std::to_string(ii));
        const double kf = 1.0 / std::sqrt(static_cast<double>(vocab) *
                                          it.s2_hi);
        PackedCtx R = rotsum_tiles(inf, tiles);

        PackedCtx S2;
        for (int k = 0; k < K; ++k) {
            PackedCtx cen;
            if (it.ex2) {
                cen = F.mult(tiles[k], kf);
            } else {
                std::vector<double> mu_v = mask[k];
                for (auto& x : mu_v) x *= kf / vocab;
                Ptx mu_pt = inf.encode_at(mu_v, R);
                cen = F.sub(F.mult(tiles[k], kf), F.mult(R, mu_pt));
            }
            PackedCtx sq = rotsum_all(inf, F.square(cen));
            if (k == 0) S2 = sq;
            else        F.inplace_add(S2, sq);
        }
        PackedCtx f = inv_sigma_cascade(inf, S2, cfg, it);

        PackedCtx Rf = F.mult(R, f);   // scalar lane: n*mu * f
        for (int k = 0; k < K; ++k) {
            PackedCtx a = F.mult(tiles[k], f);
            std::vector<double> mu_v = mask[k];
            for (auto& x : mu_v) x /= vocab;
            Ptx mun_pt = inf.encode_at(mu_v, Rf);
            PackedCtx b = F.mult(Rf, mun_pt);
            std::vector<double> sh_v = mask[k];
            for (auto& x : sh_v) x *= 0.5 / it.m;
            Ptx shift_pt = inf.encode_at(sh_v, a);
            PackedCtx y = F.im_cleanse(F.add(F.sub(a, b), shift_pt));
            F.bootstrap_hint(y.ct, kCutmaxHintLevel);
            tiles[k] = pow_odd(inf.cc_ctx(), y, it.p);
        }
        bts_marks.push_back(F.total_bootstraps);
    }

    { WithStep _s(inf, "sum");
      PackedCtx S = rotsum_tiles(inf, tiles);
      const double g = 1.0 / std::sqrt(cfg.sum_lo * cfg.sum_hi);
      const double lo = cfg.sum_lo * g, hi = cfg.sum_hi * g;
      const double bsum = 8.0 / ((lo + hi) * (lo + hi) + 4.0 * lo * hi);
      const double alpha = bsum * (lo + hi);
      const double beta = bsum;
      PackedCtx r;
      { // broadcast-scalar segment: sparse-routable; tile mults stay outside
        CKKSContext::SparseBtsScope ss(F, cutmax_sparse_bts());
        PackedCtx Sn = F.mult(S, g);
        PackedCtx F_init = F.add(F.mult(Sn, -beta), alpha);
        r = goldschmidt_inv(inf.cc_ctx(), Sn, F_init, cfg.gs_sum_iters);
        r = F.mult(r, g);
      }
      for (int k = 0; k < K; ++k)
          tiles[k] = F.mult(tiles[k], r);
    }
    bts_marks.push_back(F.total_bootstraps);
    print_bts_marks(bts_marks);
    return tiles;
}
