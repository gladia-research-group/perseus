// Isolated FHE comparison of GELU approximations on REAL activation data.
//
// Layers under test: block 2 (ill-conditioned: widest range, z_max 15000),
// block 11 (second-worst, residual-outlier block), block 5 (well-conditioned
// control). Per layer, two separate 3072-slot encrypted inputs (a shared
// ciphertext mixes fates: one out-of-range slot poisons all slots through
// bootstrapping's global linear transforms; produced by
// he-aware-training/scripts/prep_gelu_competitor_inputs.py):
//   L*/input_obs.csv     logit-uniform quantiles of the OBSERVED pre-GELU
//                        distribution (in-distribution accuracy)
//   L*/input_stress.csv  ramp to ±0.98·xmax, the calibration-certified bound
//                        (range robustness)
// plus a NOISE-COHERENT series for the fixed-polynomial competitors: encode
// x/xmax -> deliberate bootstrap (realistic ~12-bit mid-pipeline floor) ->
// rescale, then evaluate. Ours' planned runs already pay an input refresh
// internally (plan-triggered hint on x2), so its standard numbers are the
// noise-coherent ones; this series measures how much each fixed polynomial
// AMPLIFIES input noise (ours' softsign x gate saturates: bounded derivative
// by construction).
//
// Methods, all as their actual FHE circuits under the production CKKS chain:
//   ours    gelu_approx (softsign x exp-gate, Goldschmidt+Newton, per-layer
//           calibrated config "mlp.act") — run with PLANNER-placed bootstraps
//   thor    Moon et al. 2024 (eprint 2024/1881): u*(P2(P1(u/64))+1/2),
//           deg-31 o deg-27 composite; full-precision coefficients from
//           github.com/crypto-starlab/THOR src/thor/nonlinear/gelu.py.
//           Fit domain |u| <= 64.
//   encllm  de Castro et al. ICML'25: PUMA piecewise (Dong et al. 2023) with
//           segment selection via the Cheon et al. 2020 composite sign
//           (this backend's lt_function / sign, same F4/G4 polynomials).
//   cheb    single Chebyshev-119 interpolant of GELU on the calibrated range.
//
// Phases (GELU_COMP_PHASE): ours' iterative circuit is deeper than the
// 24-level budget and NEEDS mid-circuit bootstraps; the eager safety-net
// corrupts them (backend bug: constant bias on the inv-sqrt), so ours runs
// under the range-aware bootstrap placer, exactly like production:
//   capture  run ours only (all layers), eagerly, with graph capture; export
//            block_0/graph.json (op DAG + per-node |max| magnitudes)
//   planned  load the placement json, run ours planner-placed, then clear the
//            plan and run competitors eagerly (they fit the budget: 0 bts)
// Capture and planned runs execute the identical op sequence from a fresh
// context so recorded var names line up.
//
// Per-run CSVs (x, ref, got, abs_err) -> $GELU_COMP_DIR/fhe_<run>.csv for
// he-aware-training/scripts/plot_gelu_fhe_competitors.py.
// Run: scripts/50_gelu_competitors.sh (capture -> planner -> planned).

#include "ckks_primitives.h"
#include "config_loader.h"
#include "model/gpt2.h"
#include "nonlinear.h"
#include "test_helpers.h"
#include "weight_loader.h"
#include "math/matrix_ops.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

const std::vector<int> LAYERS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

std::string comp_dir() {
    return env_or("GELU_COMP_DIR",
                  "/leonardo_work/IscrC_eff-SAM2/azirilli/he-aware-training/"
                  "data/gelu_competitors");
}

std::vector<double> read_column(const std::string& path) {
    std::ifstream f(path);
    std::vector<double> v;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        v.push_back(std::stod(line));
    }
    return v;
}

double gelu_exact(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// THOR composite coefficients, ascending power order (repo lists are
// descending; P2 halved as in their he_tanh_single).
std::vector<double> thor_p1_ascending() {
    std::vector<double> desc = {
        -1.06240033e-05,  1.64454894e-04, -5.83533517e-04, -3.80912692e-04,
         2.24431193e-03,  8.92295204e-03, -1.05277477e-02, -1.91827040e-02,
        -2.04634786e-01,  4.54014410e-01, -5.40759203e-01,  5.67745523e+00,
        -1.36433727e+01,  1.82574621e+01, -8.48849601e+01,  1.28686741e+02,
         3.66720281e+02, -1.01400159e+03, -1.26278856e+02,  2.21728878e+03,
        -9.95421415e+02, -2.31059465e+03,  1.73583957e+03,  1.27394360e+03,
        -1.27836230e+03, -3.66781716e+02,  4.79663919e+02,  4.94610178e+01,
        -9.06754761e+01, -2.36515790e+00,  8.74311855e+00,  1.62838703e-02};
    return {desc.rbegin(), desc.rend()};
}

std::vector<double> thor_p2_half_ascending() {
    std::vector<double> desc = {
        -1.70270667e+02,  6.81076279e+01,  1.79197364e+03, -6.81621043e+02,
        -8.49256169e+03,  3.05629446e+03,  2.39579397e+04, -8.10435126e+03,
        -4.48145152e+04,  1.41297616e+04,  5.86197512e+04, -1.70371505e+04,
        -5.51326382e+04,  1.45532495e+04,  3.77866438e+04, -8.87673890e+03,
        -1.89514802e+04,  3.84972853e+03,  6.94169727e+03, -1.16901058e+03,
        -1.84658407e+03,  2.41693754e+02,  3.54452276e+02, -3.24499570e+01,
        -4.91918227e+01,  2.58122977e+00,  5.78392852e+00, -9.45171527e-02};
    std::vector<double> asc(desc.rbegin(), desc.rend());
    for (auto& c : asc) c *= 0.5;
    return asc;
}

// PUMA segment polynomials (Dong et al. 2023), ascending.
const std::vector<double> PUMA_F0 = {
    -0.5054031199708174, -0.42226581151983866,
    -0.11807612951181953, -0.011034134030615728};
const std::vector<double> PUMA_F1 = {
    0.008526321541038084, 0.5, 0.3603292692789629, 0.0,
    -0.037688200365904236, 0.0, 0.0018067462606141187};

struct MethodResult {
    std::vector<double> y;
    double ms = 0.0;
};

struct LayerData {
    int idx = 0;
    std::vector<double> obs, stress;
    std::vector<double> cheb_coeffs;
    double cheb_a = 0.0, cheb_b = 0.0;
    double refit_S = 0.0;                 // range-refit composite (per layer)
    std::vector<double> refit_p1, refit_p2;
    std::string tag;   // "L02"
};

PackedCtx encode_input(Inference& inf, const std::vector<double>& x,
                       int e_pad, int d_pad) {
    auto padded = matrix::pad_vector(x, e_pad);
    return encode_linear_input(inf, padded, e_pad, d_pad);
}

// Noise-coherent input: normalize, refresh once (the realistic ~12-bit
// mid-pipeline floor), rescale. The competitors then run their unmodified
// circuits on a ciphertext whose absolute noise matches deployment
// (~xmax * 2^-12) instead of a pristine fresh encryption.
PackedCtx noisy_encode(Inference& inf, const std::vector<double>& x,
                       double xmax, int e_pad, int d_pad) {
    std::vector<double> xs(x);
    for (auto& v : xs) v /= xmax;
    PackedCtx ct = encode_input(inf, xs, e_pad, d_pad);
    inf.fhe->bootstrap(ct.ct);
    return PackedCtx{inf.fhe->mult(ct.ct, xmax), ct.packing};
}

std::vector<double> decode_output(Inference& inf, const PackedCtx& ct,
                                  int e_real, int e_pad, int d_pad) {
    auto raw = decrypt_slots(inf, ct);
    auto y_pad = decode_linear_output(inf.packing, raw, inf.slots, d_pad, e_pad);
    return {y_pad.begin(), y_pad.begin() + e_real};
}

MethodResult run_ours(Inference& inf, PackedCtx ct,
                      int e_real, int e_pad, int d_pad) {
    auto t0 = std::chrono::steady_clock::now();
    PackedCtx out = gelu_approx(inf, ct, "mlp.act");
    auto t1 = std::chrono::steady_clock::now();
    return {decode_output(inf, out, e_real, e_pad, d_pad),
            std::chrono::duration<double, std::milli>(t1 - t0).count()};
}

MethodResult run_thor(Inference& inf, PackedCtx ct,
                      int e_real, int e_pad, int d_pad) {
    const size_t S = (size_t)inf.slots;
    static const auto P1 = thor_p1_ascending();
    static const auto P2h = thor_p2_half_ascending();

    auto t0 = std::chrono::steady_clock::now();
    PackedCtx t{inf.fhe->mult(ct.ct, 1.0 / 64.0), ct.packing};
    PackedCtx p1 = eval_polynomial_ps(inf.cc_ctx(), t, P1, S);
    PackedCtx p2 = eval_polynomial_ps(inf.cc_ctx(), p1, P2h, S);
    Ctx gate = inf.fhe->add(p2.ct, 0.5);
    PackedCtx out{inf.fhe->mult(ct.ct, gate), ct.packing};
    auto t1 = std::chrono::steady_clock::now();
    return {decode_output(inf, out, e_real, e_pad, d_pad),
            std::chrono::duration<double, std::milli>(t1 - t0).count()};
}

MethodResult run_encllm(Inference& inf, PackedCtx ct, double xbound,
                        int e_real, int e_pad, int d_pad) {
    const size_t S = (size_t)inf.slots;
    auto t0 = std::chrono::steady_clock::now();

    // lt_function(x, c, r) = (1 - sign((x-c)*r))/2 = [x < c]
    PackedCtx lt4   = lt_function(inf, ct, -4.0,  1.0 / (xbound + 4.0));
    PackedCtx lt195 = lt_function(inf, ct, -1.95, 1.0 / (xbound + 1.95));
    PackedCtx lt3   = lt_function(inf, ct,  3.0,  1.0 / (xbound + 3.0));

    PackedCtx f0 = eval_polynomial_ps(inf.cc_ctx(), ct, PUMA_F0, S);
    PackedCtx f1 = eval_polynomial_ps(inf.cc_ctx(), ct, PUMA_F1, S);

    // out = (1-lt4)*lt195*F0 + (1-lt195)*lt3*F1 + (1-lt3)*x
    auto one_minus = [&](const Ctx& c) {
        Ctx r = inf.fhe->negate(c);
        inf.fhe->inplace_add(r, 1.0);
        return r;
    };
    Ctx not4   = one_minus(lt4.ct);
    Ctx not195 = one_minus(lt195.ct);
    Ctx not3   = one_minus(lt3.ct);
    Ctx seg0 = inf.fhe->mult(inf.fhe->mult(not4, lt195.ct), f0.ct);
    Ctx seg1 = inf.fhe->mult(inf.fhe->mult(not195, lt3.ct), f1.ct);
    Ctx seg2 = inf.fhe->mult(not3, ct.ct);
    Ctx out = inf.fhe->add(inf.fhe->add(seg0, seg1), seg2);
    auto t1 = std::chrono::steady_clock::now();
    return {decode_output(inf, PackedCtx{out, ct.packing},
                          e_real, e_pad, d_pad),
            std::chrono::duration<double, std::milli>(t1 - t0).count()};
}

MethodResult run_cheb(Inference& inf, PackedCtx ct,
                      const std::vector<double>& coeffs, double a, double b,
                      int e_real, int e_pad, int d_pad) {
    auto t0 = std::chrono::steady_clock::now();
    PackedCtx out = eval_chebyshev_series(inf.cc_ctx(), ct, coeffs, a, b);
    auto t1 = std::chrono::steady_clock::now();
    return {decode_output(inf, out, e_real, e_pad, d_pad),
            std::chrono::duration<double, std::milli>(t1 - t0).count()};
}

// ---- best-effort competitor variants -------------------------------------
// The as-published circuits die mid-pipeline because their depth exceeds the
// post-bootstrap budget and their power-basis interiors have no in-band
// refresh site. These variants give them the SAME treatment our placer gives
// our circuit: deliberate refreshes at their bounded intermediate points —
// THOR: after P1 (composite output, |.|<~1); EncLLM: between the sign
// composite's g4/f4 stages (all in [-1,1]). This is their best shot in this
// chain; any remaining error is structural (noise amplification, precision),
// not placement.

// composite tanh-form GELU with refreshes at its bounded interior points:
// t = x/S -> [bts] -> P1 -> [bts] -> P2 ; out = x*(P2+1/2). Used both for
// THOR's published coefficients (S=64) and for our range-refit of their
// two-stage construction (scripts/refit_thor_composite.py).
MethodResult run_thor_composite(Inference& inf, PackedCtx ct, double S,
                                const std::vector<double>& P1c,
                                const std::vector<double>& P2c,
                                int e_real, int e_pad, int d_pad) {
    const size_t NS = (size_t)inf.slots;
    auto t0 = std::chrono::steady_clock::now();
    PackedCtx t{inf.fhe->mult(ct.ct, 1.0 / S), ct.packing};
    inf.fhe->bootstrap(t.ct);                       // bounded: |x|/S <= ~1
    PackedCtx p1 = eval_polynomial_ps(inf.cc_ctx(), t, P1c, NS);
    inf.fhe->bootstrap(p1.ct);                      // bounded composite output
    PackedCtx p2 = eval_polynomial_ps(inf.cc_ctx(), p1, P2c, NS);
    Ctx gate = inf.fhe->add(p2.ct, 0.5);
    Ctx out = inf.fhe->mult(t.ct, gate);            // (x/S)*(tanh/2+1/2)
    inf.fhe->inplace_mult(out, S);
    auto t1 = std::chrono::steady_clock::now();
    return {decode_output(inf, PackedCtx{out, ct.packing},
                          e_real, e_pad, d_pad),
            std::chrono::duration<double, std::milli>(t1 - t0).count()};
}

MethodResult run_thor_best(Inference& inf, PackedCtx ct,
                           int e_real, int e_pad, int d_pad) {
    static const auto P1 = thor_p1_ascending();
    static const auto P2h = thor_p2_half_ascending();
    return run_thor_composite(inf, ct, 64.0, P1, P2h, e_real, e_pad, d_pad);
}

// sign composite with refreshes between stages (g4, g4, f4, f4 all map
// [-1,1] -> [-1,1]; every stage output is a legal refresh site)
Ctx sign_with_refresh(Inference& inf, Ctx t) {
    const size_t S = (size_t)inf.slots;
    static const std::vector<double> F4 = {
        0.0, 315.0/128, 0.0, -420.0/128, 0.0, 378.0/128, 0.0, -180.0/128,
        0.0, 35.0/128};
    static const std::vector<double> G4 = {
        0.0, 5850.0/1024, 0.0, -34974.0/1024, 0.0, 97015.0/1024, 0.0,
        -113492.0/1024, 0.0, 46623.0/1024};
    for (const auto* poly : {&G4, &G4, &F4, &F4}) {
        t = eval_polynomial_ps(inf.cc_ctx(), t, *poly, S);
        inf.fhe->bootstrap(t);
    }
    return t;
}

MethodResult run_encllm_best(Inference& inf, PackedCtx ct, double xbound,
                             int e_real, int e_pad, int d_pad) {
    const size_t S = (size_t)inf.slots;
    auto t0 = std::chrono::steady_clock::now();

    auto lt_best = [&](double c) {   // [x < c] with refresh-tolerant sign
        Ctx z = inf.fhe->add(ct.ct, -c);
        inf.fhe->inplace_mult(z, 1.0 / (xbound + std::abs(c)));
        Ctx s = sign_with_refresh(inf, z);
        Ctx r = inf.fhe->negate(s);
        inf.fhe->inplace_add(r, 1.0);
        inf.fhe->inplace_mult(r, 0.5);
        return r;
    };
    Ctx lt4 = lt_best(-4.0), lt195 = lt_best(-1.95), lt3 = lt_best(3.0);

    PackedCtx f0 = eval_polynomial_ps(inf.cc_ctx(), ct, PUMA_F0, S);
    PackedCtx f1 = eval_polynomial_ps(inf.cc_ctx(), ct, PUMA_F1, S);

    auto one_minus = [&](const Ctx& c) {
        Ctx r = inf.fhe->negate(c);
        inf.fhe->inplace_add(r, 1.0);
        return r;
    };
    Ctx seg0 = inf.fhe->mult(inf.fhe->mult(one_minus(lt4), lt195), f0.ct);
    Ctx seg1 = inf.fhe->mult(inf.fhe->mult(one_minus(lt195), lt3), f1.ct);
    Ctx seg2 = inf.fhe->mult(one_minus(lt3), ct.ct);
    Ctx out = inf.fhe->add(inf.fhe->add(seg0, seg1), seg2);
    auto t1 = std::chrono::steady_clock::now();
    return {decode_output(inf, PackedCtx{out, ct.packing},
                          e_real, e_pad, d_pad),
            std::chrono::duration<double, std::milli>(t1 - t0).count()};
}

// measured bootstrap noise floor: encode -> one refresh -> decode; the output
// IS the identity, so per-slot |got - x| is exactly the refresh noise at raw
// activation scale.
MethodResult run_floor(Inference& inf, const std::vector<double>& x,
                       double xmax, int e_real, int e_pad, int d_pad) {
    auto t0 = std::chrono::steady_clock::now();
    PackedCtx ct = noisy_encode(inf, x, xmax, e_pad, d_pad);
    auto t1 = std::chrono::steady_clock::now();
    auto y = decode_output(inf, ct, e_real, e_pad, d_pad);
    return {y, std::chrono::duration<double, std::milli>(t1 - t0).count()};
}

void dump_csv(const std::string& name, const std::vector<double>& x,
              const MethodResult& r, bool identity_ref = false) {
    const std::string path = comp_dir() + "/fhe_" + name + ".csv";
    std::ofstream csv(path);
    csv << "x,ref,got,abs_err\n" << std::setprecision(12);
    double worst = 0.0, worst_in = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double ref = identity_ref ? x[i] : gelu_exact(x[i]);
        const double got = r.y[i];
        const double ae = std::abs(got - ref);
        csv << x[i] << "," << ref << "," << got << "," << ae << "\n";
        worst = std::max(worst, ae);
        if (std::abs(x[i]) <= 10.0) worst_in = std::max(worst_in, ae);
    }
    std::cout << "[" << name << "]  " << std::scientific
              << std::setprecision(3) << "max|err|=" << worst
              << "  max|err| (|x|<=10)=" << worst_in << std::fixed
              << std::setprecision(0) << "  wall=" << r.ms << "ms"
              << std::endl;
}

// Run one method, catching decode failures: an undecryptable result (OpenFHE
// "approximation error too high") is itself a first-class experimental
// outcome — the circuit destroyed the ciphertext. Recorded in fhe_status.csv.
void run_and_dump(Inference& inf, std::ofstream& status,
                  const std::string& name, const std::vector<double>& x,
                  const std::function<MethodResult()>& fn,
                  bool identity_ref = false) {
    const uint32_t b0 = inf.fhe->total_bootstraps;
    try {
        MethodResult r = fn();
        dump_csv(name, x, r, identity_ref);
        status << name << ",ok," << r.ms << ","
               << (inf.fhe->total_bootstraps - b0) << "\n";
    } catch (const std::exception& e) {
        std::cout << "[" << name << "]  DECODE FAILED: " << e.what()
                  << std::endl;
        status << name << ",decode_failed,0,"
               << (inf.fhe->total_bootstraps - b0) << "\n";
    }
    status.flush();
}

}  // namespace

TEST(GeluCompetitorsTest, RealDistributionUnderCKKS) {
    const std::string dir = comp_dir();

    std::vector<LayerData> layers;
    for (int k : LAYERS) {
        LayerData L;
        L.idx = k;
        char buf[8];
        snprintf(buf, sizeof buf, "L%02d", k);
        L.tag = buf;
        const std::string ldir = dir + "/" + L.tag;
        L.obs = read_column(ldir + "/input_obs.csv");
        L.stress = read_column(ldir + "/input_stress.csv");
        auto cheb_raw = read_column(ldir + "/cheb119.csv");
        if (L.obs.empty() || L.stress.empty() || cheb_raw.size() < 3)
            GTEST_SKIP() << "missing inputs in " << ldir
                         << " (run scripts/prep_gelu_competitor_inputs.py)";
        L.cheb_a = cheb_raw[0];
        L.cheb_b = cheb_raw[1];
        L.cheb_coeffs.assign(cheb_raw.begin() + 2, cheb_raw.end());
        auto refit = read_column(ldir + "/thor_refit.csv");
        if (refit.size() == 1 + 32 + 28) {
            L.refit_S = refit[0];
            L.refit_p1.assign(refit.begin() + 1, refit.begin() + 33);
            L.refit_p2.assign(refit.begin() + 33, refit.end());
        }
        layers.push_back(std::move(L));
    }

    std::cout << "[competitors] creating CKKS context..." << std::endl;
    Inference inf = make_gpt2_inference({
        .ckks = {.bts_iterations = default_bts_iterations()},
    });

    const int d_pad  = inf.size.hidDim;
    const int e_pad  = inf.size.expDim;
    const int e_real = inf.size.getRealFfDim();
    for (const auto& L : layers) {
        ASSERT_EQ((int)L.obs.size(), e_real);
        ASSERT_EQ((int)L.stress.size(), e_real);
    }

    const std::string configs_path = default_configs_path();
    std::cout << "[competitors] configs: " << configs_path << std::endl;
    auto parsed = config_loader::parse_configs_json(
        config_loader::read_file_to_string(configs_path));

    auto install = [&](const LayerData& L) {
        weight_loader::prepare_gpt2_layer_configs(inf, parsed, L.idx);
        return inf.gelu_cfg.at("mlp.act").xmax;
    };

    const std::string phase = env_or("GELU_COMP_PHASE", "eager");
    std::cout << "[phase] " << phase << std::endl;

    if (phase == "capture") {
        // Optional: capture UNDER an existing plan (round 2). The eager
        // round-1 capture records corrupted magnitudes downstream of its
        // unplanned refreshes; re-capturing with the round-1 plan installed
        // yields faithful magnitudes for the final placement (the production
        // EXPORT_UNDER_PLAN pattern from 03_export_graph.sh).
        const std::string under = env_or("GELU_COMP_CAPTURE_UNDER_PLAN", "");
        if (!under.empty()) {
            ASSERT_TRUE(inf.load_bootstrap_plan_json(under))
                << "cannot load plan " << under;
            inf.fhe->expected_levels.clear();
            inf.fhe->expected_producers.clear();
            std::cout << "[capture] under plan " << under << " (placements="
                      << inf.fhe->placement_after.size() << ")" << std::endl;
        }
        inf.enable_graph_capture();
        for (const auto& L : layers) {
            install(L);
            (void)run_ours(inf, encode_input(inf, L.obs, e_pad, d_pad),
                           e_real, e_pad, d_pad);
            (void)run_ours(inf, encode_input(inf, L.stress, e_pad, d_pad),
                           e_real, e_pad, d_pad);
        }
        const std::string gdir = dir + "/graph/block_0";
        std::filesystem::create_directories(gdir);
        inf.export_graph_json(gdir + "/graph.json");
        inf.disable_graph_capture();
        std::cout << "[capture] graph exported -> " << gdir << "/graph.json"
                  << std::endl;
        SUCCEED();
        return;
    }

    ASSERT_EQ(phase, "planned") << "use GELU_COMP_PHASE=capture|planned "
                                   "(eager is broken for ours — backend bug)";

    const std::string plan_path =
        env_or("GELU_COMP_PLAN",
               "bootstrap_placements/gelu_iso/block_0_placement.json");
    ASSERT_TRUE(inf.load_bootstrap_plan_json(plan_path))
        << "cannot load plan " << plan_path;
    // Keep the planner's bootstrap POSITIONS but drop the per-op level
    // assertions: the planner's level model and the runtime differ by a
    // FLEXIBLEAUTO pending-rescale unit on early pt-mults, which is
    // bookkeeping, not divergence — placements are keyed by var name and
    // land at the same ops either way.
    inf.fhe->expected_levels.clear();
    inf.fhe->expected_producers.clear();
    std::cout << "[planned] plan installed from " << plan_path
              << " (placements=" << inf.fhe->placement_after.size()
              << ", level checks relaxed)" << std::endl;

    std::ofstream status(dir + "/fhe_status.csv");
    status << "run,status,wall_ms,bootstraps\n";

    // ours, planner-placed — same op order as the capture
    for (const auto& L : layers) {
        install(L);
        run_and_dump(inf, status, "ours_" + L.tag + "_obs", L.obs, [&] {
            return run_ours(inf, encode_input(inf, L.obs, e_pad, d_pad),
                            e_real, e_pad, d_pad); });
        run_and_dump(inf, status, "ours_" + L.tag + "_stress", L.stress, [&] {
            return run_ours(inf, encode_input(inf, L.stress, e_pad, d_pad),
                            e_real, e_pad, d_pad); });
    }
    inf.clear_bootstrap_plan();

    // competitors, eager (they fit the level budget: 0 bootstraps), plus the
    // noise-coherent series on the observed input
    for (const auto& L : layers) {
        const double xbound = install(L);
        struct InputSet { const char* tag; const std::vector<double>* x; bool noisy; };
        for (const auto& in : {InputSet{"obs", &L.obs, false},
                               InputSet{"stress", &L.stress, false},
                               InputSet{"noisy", &L.obs, true}}) {
            const auto& x = *in.x;
            const std::string t = "_" + L.tag + "_" + in.tag;
            auto enc = [&] {
                return in.noisy
                    ? noisy_encode(inf, x, xbound, e_pad, d_pad)
                    : encode_input(inf, x, e_pad, d_pad);
            };
            run_and_dump(inf, status, "thor" + t, x, [&] {
                return run_thor(inf, enc(), e_real, e_pad, d_pad); });
            run_and_dump(inf, status, "encllm" + t, x, [&] {
                return run_encllm(inf, enc(), xbound, e_real, e_pad, d_pad); });
            run_and_dump(inf, status, "cheb119" + t, x, [&] {
                return run_cheb(inf, enc(), L.cheb_coeffs, L.cheb_a, L.cheb_b,
                                e_real, e_pad, d_pad); });
        }

        // best-effort variants under deployment conditions: their circuits
        // with deliberate refreshes at their bounded interior points (the
        // same treatment our placer gives ours)
        run_and_dump(inf, status, "thor_best_" + L.tag, L.obs, [&] {
            return run_thor_best(inf,
                                 noisy_encode(inf, L.obs, xbound, e_pad, d_pad),
                                 e_real, e_pad, d_pad); });
        run_and_dump(inf, status, "encllm_best_" + L.tag, L.obs, [&] {
            return run_encllm_best(inf,
                                   noisy_encode(inf, L.obs, xbound, e_pad, d_pad),
                                   xbound, e_real, e_pad, d_pad); });
        // best-effort on the certified stress range: refreshes cannot fix a
        // fit-domain violation — measured, not asserted
        run_and_dump(inf, status, "thor_best_stress_" + L.tag, L.stress, [&] {
            return run_thor_best(inf,
                                 noisy_encode(inf, L.stress, xbound, e_pad, d_pad),
                                 e_real, e_pad, d_pad); });
        run_and_dump(inf, status, "encllm_best_stress_" + L.tag, L.stress, [&] {
            return run_encllm_best(inf,
                                   noisy_encode(inf, L.stress, xbound, e_pad, d_pad),
                                   xbound, e_real, e_pad, d_pad); });

        // range-refit composite (our reconstruction of their two-stage fit,
        // refit to this layer's calibrated bound) — the adversarial check
        if (L.refit_S > 0.0) {
            run_and_dump(inf, status, "thor_refit_" + L.tag, L.obs, [&] {
                return run_thor_composite(inf,
                    noisy_encode(inf, L.obs, xbound, e_pad, d_pad),
                    L.refit_S, L.refit_p1, L.refit_p2, e_real, e_pad, d_pad); });
            run_and_dump(inf, status, "thor_refit_stress_" + L.tag, L.stress, [&] {
                return run_thor_composite(inf,
                    noisy_encode(inf, L.stress, xbound, e_pad, d_pad),
                    L.refit_S, L.refit_p1, L.refit_p2, e_real, e_pad, d_pad); });
        }

        // measured bootstrap noise floor at this layer's raw scale
        run_and_dump(inf, status, "floor_" + L.tag, L.obs, [&] {
            return run_floor(inf, L.obs, xbound, e_real, e_pad, d_pad); },
            /*identity_ref=*/true);
    }

    SUCCEED();  // instrumentation run; the figure script judges the numbers
}
