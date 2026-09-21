// Bootstrap accuracy sweep — the tool that produces perseus/plan/data/bts_accuracy_<chain>.json,
// the measured error model the placer prices refresh sites with (perseus/plan/btserr.py).
//
// The correction factor is a runtime property (CorrectionScope), so one context sweeps the
// whole (correction factor x data period x sparse route x amplitude) grid and the bootstrap
// setup is paid once. AccuracyTable and OffsetTable emit the [bts_acc] / [bts_off] lines that
// scripts/utils/sweep_bts_accuracy.sh parses into the JSON table; the three WallVsAmplitude /
// OffsetTransform tests are single-axis views of the same measurement.
//
//   CHAIN=n32 bash scripts/utils/sweep_bts_accuracy.sh
// Env: BTS_ACC_CFS, BTS_ACC_AMPS, BTS_ACC_PERIODS, BTS_ACC_RIPPLE, BTS_ACC_ENCODE_CAP,
//      BTS_WALL_AMPS, BTS_WALL_LEVEL.

#include "fideslib_wrapper.h"
#include "test_helpers.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <limits>
#include <random>
#include <vector>

using namespace test_helpers;

namespace {

std::vector<double> amps_from_env() {
    const char* e = std::getenv("BTS_WALL_AMPS");
    const std::string s = (e && *e) ? e : "0.25,0.5,1,2,3,3.48909,4,5,6,8,10";
    std::vector<double> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        out.push_back(std::stod(s.substr(i, j - i)));
        i = j + 1;
    }
    return out;
}

// max |got - A| / A over the slots: a RELATIVE read, so amplitudes are comparable.
double rel_err(const std::vector<double>& got, double A) {
    double m = 0.0;
    for (double x : got) m = std::max(m, std::fabs(x - A));
    return m / std::fabs(A);
}

// max |got - ref| / A: same normalization as rel_err (error measured in units of the
// amplitude, NOT per-slot, so slots near zero do not blow the ratio up).
double err_vs_ref(const std::vector<double>& got, const std::vector<double>& ref, double A) {
    double m = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) m = std::max(m, std::fabs(got[i] - ref[i]));
    return m / std::fabs(A);
}

// An s-periodic slot vector: random in the first s lanes, repeated. s=0 means dense
// (independent random in every lane). A sparse-routed bootstrap is only VALID on data
// that is already s-periodic — this builds exactly that.
std::vector<double> periodic_vec(int slots, uint32_t s, double A, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-A, A);
    std::vector<double> v(slots);
    const size_t period = (s == 0) ? v.size() : s;
    for (size_t i = 0; i < period && i < v.size(); ++i) v[i] = dist(gen);
    for (size_t i = period; i < v.size(); ++i) v[i] = v[i % period];
    return v;
}

}  // namespace

TEST(BtsAccuracySweep, WallVsAmplitudeFreshAndDeep) {
    CKKSContextOptions o = default_ckks_options();   // honours the CHAIN=n32 env knobs
    auto ctx = make_ckks_context(o);                 // no extra_rot_steps: nothing rotates here
    CKKSContext& F = *ctx;
    const int slots = static_cast<int>(F.cc->GetRingDimension() / 2);
    const auto enc_const = [&](double v) { return ::encrypt_const(F.cc, v, slots, F.pk()); };

    const int level = [&] {
        const char* e = std::getenv("BTS_WALL_LEVEL");
        return (e && *e) ? std::atoi(e) : static_cast<int>(F.level_limit());
    }();

    std::cout << "[bts_wall] chain d=" << F.composite_degree
              << " total_depth=" << F.total_depth
              << " level_limit=" << F.level_limit()
              << " deep_level=" << level
              << " bts_out_level=" << F.bootstrap_output_level() << std::endl;

    // BTS_WALL_ROUTE_S=s routes these bootstraps through the sparse precomp for period s.
    // A CONSTANT vector is period-1, hence legally routable at ANY s — which makes this the
    // apples-to-apples dense-vs-sparse comparison the random-broadcast arm cannot give
    // (there the value is uniform in [-A,A] but the error is normalised by A, flattering
    // sparse by 2-3x). Same constant, same CF, only the route changes.
    const uint32_t route_s = [] {
        const char* e = std::getenv("BTS_WALL_ROUTE_S");
        return (e && *e) ? static_cast<uint32_t>(std::atoi(e)) : 0u;
    }();
    if (route_s && !F.sparse_precomp_slots.count(route_s)) {
        std::cout << "[bts_wall] route_s=" << route_s << " has no precomp — aborting"
                  << std::endl;
        FAIL();
    }
    F.sparse_bts_active = route_s;
    std::cout << "[bts_wall] route_s=" << route_s << " (0 = dense)" << std::endl;

    for (double A : amps_from_env()) {
        // --- fresh: full level, deg 1 -------------------------------------------------
        Ctx fresh = enc_const(A);
        F.bootstrap(fresh);
        const double e_fresh = rel_err(decrypt_slots(F, fresh), A);

        // --- deep: at the reactive threshold WITH a pending rescale --------------------
        // drop_to_level puts it at the threshold; the scalar multiply leaves
        // noiseScaleDeg = 2 under FLEXIBLEAUTO (x1.0 so the value is untouched), which is
        // the state 350 of 382 probed bootstraps are actually in on n32.
        Ctx deep = enc_const(A);
        F.drop_to_level(deep, level);
        F.inplace_mult(deep, 1.0);
        const int deep_deg = deep ? static_cast<int>(deep->GetNoiseScaleDeg()) : -1;
        const int deep_lvl = static_cast<int>(level_of(deep));
        F.bootstrap(deep);
        const double e_deep = rel_err(decrypt_slots(F, deep), A);

        std::cout << std::fixed << std::setprecision(5)
                  << "[bts_wall] A=" << A
                  << std::scientific << std::setprecision(3)
                  << "  fresh_rel_err=" << e_fresh
                  << "  deep_rel_err=" << e_deep
                  << " (deep_lvl=" << deep_lvl << " deg=" << deep_deg << ")"
                  << std::endl;
    }
    SUCCEED();
}

// Does the wall move when the bootstrap is SPARSE-ROUTED, and does it depend on the
// period s? Sparse routing changes which precomp (and, under FIDESLIB_SPARSE_ARCSINE,
// whether the arcsine correction runs at all) the EvalMod goes through, so the wall is
// not automatically the dense one. Data is genuinely s-periodic, which is the only input
// a sparse route is valid on.
//
// Arms: dense, then every s in F.sparse_precomp_slots (SPARSE_BTS_SLOTS, n32: 512,32,1).
// Env: BTS_WALL_SLOTS="512,32,1" overrides the s list.
TEST(BtsAccuracySweep, WallVsAmplitudeBySparsePeriod) {
    CKKSContextOptions o = default_ckks_options();
    auto ctx = make_ckks_context(o);
    CKKSContext& F = *ctx;
    const int slots = static_cast<int>(F.cc->GetRingDimension() / 2);

    std::vector<uint32_t> ss{0};   // 0 = dense
    if (const char* e = std::getenv("BTS_WALL_SLOTS")) {
        const std::string str = e;
        size_t i = 0;
        while (i < str.size()) {
            size_t j = str.find(',', i);
            if (j == std::string::npos) j = str.size();
            ss.push_back(static_cast<uint32_t>(std::stoul(str.substr(i, j - i))));
            i = j + 1;
        }
    } else {
        for (uint32_t s : F.sparse_precomp_slots) ss.push_back(s);
    }

    std::cout << "[bts_sparse] precomps={";
    for (uint32_t s : F.sparse_precomp_slots) std::cout << s << ",";
    std::cout << "} routing_default=" << F.sparse_bts_slots << std::endl;

    for (uint32_t s : ss) {
        if (s && !F.sparse_precomp_slots.count(s)) {
            std::cout << "[bts_sparse] s=" << s << " SKIPPED (no precomp built)" << std::endl;
            continue;
        }
        for (double A : amps_from_env()) {
            const std::vector<double> v = periodic_vec(slots, s, A, 7777);
            Ctx ct = encrypt(F.cc, encode(F.cc, v), F.pk());
            double e = std::numeric_limits<double>::quiet_NaN();
            try {
                const uint32_t saved = F.sparse_bts_active;
                F.sparse_bts_active = s;          // 0 = dense; SparseBtsScope only arms the default
                F.bootstrap(ct);
                F.sparse_bts_active = saved;
                auto out = decrypt_slots(F, ct);
                out.resize(v.size());
                e = err_vs_ref(out, v, A);
            } catch (const std::exception& ex) {
                std::cout << "[bts_sparse] s=" << s << " A=" << A << " THREW: " << ex.what()
                          << std::endl;
                continue;
            }
            std::cout << std::fixed << std::setprecision(5)
                      << "[bts_sparse] s=" << s << " A=" << A
                      << std::scientific << std::setprecision(3)
                      << " rel_err=" << e << std::endl;
        }
    }
    SUCCEED();
}

// THE OFFSET TRANSFORM: bootstrap(ct - c) + c, with c the site's DC offset (a plaintext
// constant known at plan time from magnitude capture).
//
// Rationale: the sine sees a plaintext COEFFICIENT, and a non-zero mean puts all
// of its energy into coefficient 0 — so a DC offset, not the spread, is what drives a site
// past the wall. Subtracting it is exact and costs no levels (a plaintext add before, a
// plaintext add back after, at the bootstrap output level). This is the ADDITIVE sibling of
// inner_bootstrap's existing multiplicative `prescale`, and strictly better for a
// DC-dominated site: prescale shrinks the spread too, the offset removes only the DC.
//
// Data: constant DC plus a small zero-mean ripple, i.e. the shape of a real LN/Goldschmidt
// input. Arms: refresh as-is vs refresh the residual. Env: BTS_WALL_RIPPLE (default 0.01).
TEST(BtsAccuracySweep, OffsetTransform) {
    CKKSContextOptions o = default_ckks_options();
    auto ctx = make_ckks_context(o);
    CKKSContext& F = *ctx;
    const int slots = static_cast<int>(F.cc->GetRingDimension() / 2);
    const double ripple = [] {
        const char* e = std::getenv("BTS_WALL_RIPPLE");
        return (e && *e) ? std::atof(e) : 0.01;
    }();

    std::cout << "[bts_offset] ripple=" << ripple << std::endl;

    for (double dc : amps_from_env()) {
        //  The ripple MUST be s-periodic, not per-slot random. A per-slot ripple makes v
        // aperiodic, the sparse arm below is then an ILLEGAL route, and it folds the ripple
        // away — yielding an error exactly equal to the ripple amplitude and independent of
        // CF. That is a precondition violation masquerading as "sparse is worse" (measured
        const uint32_t rs_period = [] {
            const char* e = std::getenv("BTS_OFFSET_ROUTE_S");
            return (e && *e) ? static_cast<uint32_t>(std::atoi(e)) : 32u;
        }();
        std::mt19937 gen(31337);
        std::uniform_real_distribution<double> d(-ripple, ripple);
        std::vector<double> v(slots);
        for (size_t i = 0; i < rs_period && i < v.size(); ++i) v[i] = dc + d(gen);
        for (size_t i = rs_period; i < v.size(); ++i) v[i] = v[i % rs_period];

        // arm A: refresh as-is
        Ctx plain = encrypt(F.cc, encode(F.cc, v), F.pk());
        F.bootstrap(plain);
        const double e_plain = err_vs_ref(decrypt_slots(F, plain), v, std::fabs(dc));

        // arm B: subtract the DC, refresh the residual, add it back
        Ctx off = encrypt(F.cc, encode(F.cc, v), F.pk());
        F.inplace_add(off, -dc);
        F.bootstrap(off);
        F.inplace_add(off, dc);
        const double e_off = err_vs_ref(decrypt_slots(F, off), v, std::fabs(dc));

        // arm C: offset THEN sparse-routed. The offset is what makes this pay: as-is the
        // input sits above the band where routing is provably neutral, but the
        // RESIDUAL is small, which is exactly where sparse's lower floor dominates. A
        // constant + ripple is period-1 data, so any built s is a legal route.
        double e_off_sparse = std::numeric_limits<double>::quiet_NaN();
        const uint32_t rs = [] {
            const char* e = std::getenv("BTS_OFFSET_ROUTE_S");
            return (e && *e) ? static_cast<uint32_t>(std::atoi(e)) : 32u;
        }();
        if (F.sparse_precomp_slots.count(rs)) {
            Ctx os = encrypt(F.cc, encode(F.cc, v), F.pk());
            F.inplace_add(os, -dc);
            const uint32_t saved = F.sparse_bts_active;
            F.sparse_bts_active = rs;
            F.bootstrap(os);
            F.sparse_bts_active = saved;
            F.inplace_add(os, dc);
            e_off_sparse = err_vs_ref(decrypt_slots(F, os), v, std::fabs(dc));
        }

        std::cout << std::fixed << std::setprecision(5) << "[bts_offset] dc=" << dc
                  << std::scientific << std::setprecision(3)
                  << " as_is=" << e_plain << " offset=" << e_off
                  << " offset+sparse(s=" << rs << ")=" << e_off_sparse
                  << " gain=" << (e_plain / e_off) << "x"
                  << " gain_sparse=" << (e_plain / e_off_sparse) << "x" << std::endl;
    }
    SUCCEED();
}

// ─────────────────────────────────────────────────────────────────────────────────────
// THE ACCURACY TABLE the placer plans against.
//
// The three tests above each vary ONE axis and read CF from the context, so a CF sweep
// costs one process (and one ~90 s bootstrap setup) per CF. CorrectionScope makes that
// unnecessary: CF is runtime-only — nothing precomputed depends on it — so a single
// context can sweep the whole (CF x data period x route x amplitude) grid. That is what
// these two tests do, in a machine-readable form scripts/utils/sweep_bts_accuracy.sh
// parses into perseus/plan/data/bts_accuracy_<chain>.json.
//
// Env: BTS_ACC_CFS     correction factors to sweep        (default "2,3,4,5,6,7,8,9")
//      BTS_ACC_AMPS    amplitude grid                     (default: a log grid 1e-3..256)
//      BTS_ACC_PERIODS data periods; 0 = dense            (default "1,512,0")
//      BTS_ACC_RIPPLE  offset-table ripple, relative to dc (default 0.01)
// ─────────────────────────────────────────────────────────────────────────────────────
namespace {

std::vector<double> csv_doubles(const char* name, const char* fallback) {
    const char* e = std::getenv(name);
    const std::string s = (e && *e) ? e : fallback;
    std::vector<double> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        out.push_back(std::stod(s.substr(i, j - i)));
        i = j + 1;
    }
    return out;
}

std::vector<int> csv_ints(const char* name, const char* fallback) {
    std::vector<int> out;
    for (double d : csv_doubles(name, fallback)) out.push_back(static_cast<int>(d));
    return out;
}

// The amplitude grid must span both ends of every CF's band. Bands are [0.003, 0.03]·2^CF,
// so CF=2 bottoms out at 0.012 and the CF=9 prescale reach (10x the band top) tops out at
// 154 — hence 1e-3 .. 256, ~3 points per decade.
constexpr const char* kDefaultAmps =
    "0.001,0.002,0.005,0.01,0.02,0.03,0.05,0.1,0.2,0.3,0.5,1,2,3,5,8,10,16,32,64,128,256";

// periodic_vec draws uniformly in [-A, A], so its realised max is only ~A in expectation
// — at period=1 it is ONE draw, which can land at 0.1A and silently understate the wall by
// a decade. The table is indexed by the amplitude, so anchor it: rescale so the realised
// max |value| is exactly A. Periodicity is preserved (a scalar multiple of a p-periodic
// vector is p-periodic).
void anchor_max(std::vector<double>& v, double A) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::fabs(x));
    if (m <= 0.0) { for (double& x : v) x = A; return; }
    const double g = A / m;
    for (double& x : v) x *= g;
}

}  // namespace

TEST(BtsAccuracySweep, AccuracyTable) {
    CKKSContextOptions o = default_ckks_options();
    auto ctx = make_ckks_context(o);
    CKKSContext& F = *ctx;
    const int slots = static_cast<int>(F.cc->GetRingDimension() / 2);

    const std::vector<int> cfs = csv_ints("BTS_ACC_CFS", "2,3,4,5,6,7,8,9");
    const std::vector<double> amps = csv_doubles("BTS_ACC_AMPS", kDefaultAmps);
    const std::vector<int> periods = csv_ints("BTS_ACC_PERIODS", "1,512,0");

    std::cout << "[bts_acc] meta chain_d=" << F.composite_degree
              << " total_depth=" << F.total_depth
              << " level_limit=" << F.level_limit()
              << " bts_out_level=" << F.bootstrap_output_level()
              << " slots=" << slots << " precomps={";
    for (uint32_t s : F.sparse_precomp_slots) std::cout << s << ",";
    std::cout << "}" << std::endl;

    for (int cf : cfs) {
        for (int period : periods) {
            // Legal routes for data of this period: dense, plus every BUILT precomp the
            // period divides. Routing at s requires period | s (include/packing/pack_tag.h) —
            // an illegal route folds residue classes and returns plausible garbage, so
            // the table must never contain one.
            std::vector<uint32_t> routes{0};
            if (period > 0) {
                for (uint32_t s : F.sparse_precomp_slots) {
                    if (s % static_cast<uint32_t>(period) == 0) routes.push_back(s);
                }
            }
            for (uint32_t route : routes) {
                for (double A : amps) {
                    // Same seed for every (cf, route) at a given (period, A): the data is
                    // held FIXED across the axes we are attributing to, which is exactly
                    // the confound this axis is easiest to introduce.
                    std::vector<double> v =
                        periodic_vec(slots, static_cast<uint32_t>(period), A, 7777);
                    anchor_max(v, A);
                    double e = std::numeric_limits<double>::quiet_NaN();
                    try {
                        // BTS_ACC_ENCODE_CAP: fresh encode stages every
                        // coefficient as int64 before the CRT split, so |m|*Delta must
                        // stay under ~2^61 — |m| ~ 128 at Delta=2^54. Bands for CF >= 16
                        // live ABOVE that, so encoding them directly measures the
                        // ENCODER, not the bootstrap (fingerprint: rel_err 0.5 -> 0.75 ->
                        // 0.875, a 1-2^-k top-bit-loss staircase). Build those amplitudes
                        // the way the RUNTIME does — by computation: encode A/2^k and
                        // double k times. Self-addition is exact, level-free and encode-
                        // free, so the measured cell is the bootstrap's own error.
                        const char* _cap = std::getenv("BTS_ACC_ENCODE_CAP");
                        const double cap = (_cap && *_cap) ? std::atof(_cap) : 0.0;
                        int dbl = 0;
                        std::vector<double> venc = v;
                        if (cap > 0.0 && A > cap) {
                            while (A / std::pow(2.0, dbl) > cap) ++dbl;
                            const double f = std::pow(2.0, dbl);
                            for (double& x : venc) x /= f;
                        }
                        Ctx ct = encrypt(F.cc, encode(F.cc, venc), F.pk());
                        for (int d = 0; d < dbl; ++d) F.cc->EvalAddInPlace(ct, ct);
                        CKKSContext::CorrectionScope cs(F, cf);
                        const uint32_t saved = F.sparse_bts_active;
                        F.sparse_bts_active = route;
                        F.bootstrap(ct);
                        F.sparse_bts_active = saved;
                        std::vector<double> out = decrypt_slots(F, ct);
                        out.resize(v.size());
                        e = err_vs_ref(out, v, A);
                    } catch (const std::exception& ex) {
                        // cf < deg throws the Bootstrap deg-guard, per call. Record the
                        // refusal rather than dropping the row: the placer needs to know
                        // the CF floor is a hard boundary, not a soft one.
                        std::cout << "[bts_acc] cf=" << cf << " period=" << period
                                  << " route=" << route << " A=" << A
                                  << " rel_err=nan threw=" << ex.what() << std::endl;
                        continue;
                    }
                    std::cout << std::scientific << std::setprecision(6)
                              << "[bts_acc] cf=" << cf << " period=" << period
                              << " route=" << route << " A=" << A
                              << " rel_err=" << e << std::endl;
                }
            }
        }
    }
    SUCCEED();
}

// The offset axis of the same table: dc + s-periodic ripple, as-is vs bootstrap(ct-c)+c,
// swept over CF. The two are ONE decision: after offsetting, the
// band must follow the RESIDUAL, so the best CF drops — and the placer needs both columns
// at the same (dc, cf) to make that choice.
TEST(BtsAccuracySweep, OffsetTable) {
    CKKSContextOptions o = default_ckks_options();
    auto ctx = make_ckks_context(o);
    CKKSContext& F = *ctx;
    const int slots = static_cast<int>(F.cc->GetRingDimension() / 2);

    const std::vector<int> cfs = csv_ints("BTS_ACC_CFS", "2,3,4,5,6,7,8,9");
    const std::vector<double> dcs = csv_doubles("BTS_ACC_AMPS", kDefaultAmps);
    const double ripple_frac = [] {
        const char* e = std::getenv("BTS_ACC_RIPPLE");
        return (e && *e) ? std::atof(e) : 0.01;
    }();
    // The ripple is s-periodic by construction at the largest built precomp, so every
    // built route stays legal on this data: a per-slot ripple would make the sparse arm an
    // illegal route, and it would then measure its own precondition rather than the route.
    const uint32_t rs_period = 512;

    std::cout << "[bts_off] meta ripple_frac=" << ripple_frac
              << " ripple_period=" << rs_period << std::endl;

    for (int cf : cfs) {
        for (double dc : dcs) {
            const double ripple = std::fabs(dc) * ripple_frac;
            std::mt19937 gen(31337);
            std::uniform_real_distribution<double> d(-ripple, ripple);
            std::vector<double> v(slots);
            for (size_t i = 0; i < rs_period && i < v.size(); ++i) v[i] = dc + d(gen);
            for (size_t i = rs_period; i < v.size(); ++i) v[i] = v[i % rs_period];

            double e_plain = std::numeric_limits<double>::quiet_NaN();
            double e_off = e_plain;
            try {
                Ctx plain = encrypt(F.cc, encode(F.cc, v), F.pk());
                CKKSContext::CorrectionScope cs(F, cf);
                F.bootstrap(plain);
                e_plain = err_vs_ref(decrypt_slots(F, plain), v, std::fabs(dc));

                Ctx off = encrypt(F.cc, encode(F.cc, v), F.pk());
                F.inplace_add(off, -dc);
                F.bootstrap(off);
                F.inplace_add(off, dc);
                e_off = err_vs_ref(decrypt_slots(F, off), v, std::fabs(dc));
            } catch (const std::exception& ex) {
                std::cout << "[bts_off] cf=" << cf << " dc=" << dc
                          << " as_is=nan offset=nan threw=" << ex.what() << std::endl;
                continue;
            }
            std::cout << std::scientific << std::setprecision(6)
                      << "[bts_off] cf=" << cf << " dc=" << dc
                      << " as_is=" << e_plain << " offset=" << e_off << std::endl;
        }
    }
    SUCCEED();
}
