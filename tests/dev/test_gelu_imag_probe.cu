// PROBE: is the imaginary lane ACTUALLY corrupted by the GELU gate?
//
// The gate arm (nonlinear.cu:101-110) does  gy = cheb(axsq); bootstrap_hint(gy,16); K*square(gy)
// with NO im_cleanse anywhere — unlike BOTH softmax exps (cachemir_attention.cu:150-162,
// cachemir_filling_attention.cu:167-178) which cleanse right after their squares. A full-slot
// bootstrap PRESERVES the imaginary lane and re-injects the ~9-bit imaginary floor; each square
// then folds Im forward ( (r+iε)^2 = r^2-ε^2 + i*2rε ), so K squares grow |Im| ~2^K and that
// corruption rides into z = z*gy and out through the down-proj.
//
// This probe MEASURES the imaginary lane directly (GetCKKSPackedValue().imag()), so it is real
// evidence, not inferred from a real-vs-exact error like test_gate_exp_cleanse.
//
//   Part A  — run the REAL production gelu_approx on real block-0 pre_gelu data, gate ON vs an
//             in-test gate-OFF copy of the same config, and report max|Im| / |Im|:|Re| of the
//             OUTPUT.  gate-ON >> gate-OFF  => the gate is what injects the imaginary corruption.
//   Part B  — reproduce the isolated gate-exp core faithfully and decrypt after EACH square, so
//             the |Im| growth vs K is visible; then repeat with a softmax-style im_cleanse AFTER
//             the squares, showing |Im| collapses to the bts floor (the proposed fix).
//
// Run with a gate-ON config, e.g. CONFIGS_PATH=.../gpt2_diff_gate/configs.json (scripts/09_*.sh).

#include "all_blocks_test_helpers.h"
#include "ckks_primitives.h"
#include "model/gpt2.h"
#include "nonlinear.h"
#include "weight_loader.h"
#include "math/matrix_ops.h"
#include "test_helpers.h"

#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace test_helpers;

namespace {

// Decrypt the full complex slot vector (real AND imaginary), the test-boundary .ct unwrap.
std::vector<std::complex<double>> decrypt_cplx(Inference& inf, const PackedCtx& pc) {
    Plaintext pt = decrypt_pt(inf.cc(), pc.ct, inf.fhe->sk());
    return pt->GetCKKSPackedValue();
}

struct ImStats {
    double max_re = 0.0, max_im = 0.0;   // over all scanned slots
    double sig_re = 0.0, sig_im = 0.0;   // summed over signal slots (|re| > floor)
    int    nsig   = 0;
    double ratio() const { return sig_re > 0.0 ? sig_im / sig_re : 0.0; }  // mean |Im|:|Re| on signal
};

ImStats im_stats(const std::vector<std::complex<double>>& v, int n, double floor = 1e-3) {
    ImStats s;
    for (int i = 0; i < n && i < (int)v.size(); ++i) {
        const double re = std::abs(v[i].real()), im = std::abs(v[i].imag());
        s.max_re = std::max(s.max_re, re);
        s.max_im = std::max(s.max_im, im);
        if (re > floor) { s.sig_re += re; s.sig_im += im; ++s.nsig; }
    }
    return s;
}

void print_im(const std::string& tag, const ImStats& s) {
    std::cout << std::scientific << std::setprecision(3)
              << "  " << std::left << std::setw(26) << tag << std::right
              << "  max|Re|=" << s.max_re
              << "  max|Im|=" << s.max_im
              << "  mean|Im|:|Re|=" << s.ratio()
              << "  (nsig=" << s.nsig << ")\n";
}

}  // namespace

TEST(GeluImagProbe, ImaginaryIsCorruptedByGate) {
    // Build the context in the LIVE decode's payload mode: default_ckks_options() reads
    // CKKS_COMPLEX from env. With complex payload the imaginary lane is live and
    // GetCKKSPackedValue().imag() is meaningful; with real payload (CKKS_COMPLEX=0) the
    // decoder zeroes imag by construction and this probe would measure nothing. Run 09_*.sh
    // (CKKS_COMPLEX=1) to see the real behaviour.
    auto ckks = default_ckks_options();
    ckks.bts_iterations = default_bts_iterations();
    Inference inf = make_gpt2_inference({.ckks = ckks});
    std::cout << "[imag_probe] complex_payload=" << (int)inf.fhe->complex_payload
              << " (0 => imaginary lane is not observable; run with CKKS_COMPLEX=1)\n";

    const std::string weights_path = default_weights_path();
    { std::ifstream probe(weights_path); if (!probe) GTEST_SKIP() << "weights: " << weights_path; }

    const std::string configs_path = default_configs_path();
    std::cout << "[imag_probe] configs = " << configs_path << "\n";
    auto parsed = config_loader::parse_configs_json(
        config_loader::read_file_to_string(configs_path));
    weight_loader::prepare_gpt2_layer_configs(inf, parsed, /*block_idx=*/0);

    const GeLUConfig cfg = inf.gelu_cfg.at("mlp.act");   // block-0 gate config
    std::cout << "[imag_probe] block0  gate=" << (cfg.gate ? "ON" : "OFF")
              << "  a=" << cfg.a << "  b=" << cfg.b << "  exp_iters=" << cfg.exp_iters
              << "  gate_cheb_a=" << cfg.gate_cheb_a << "  xmax=" << cfg.xmax << "\n";

    // Register a gate-OFF twin of the SAME config for attribution.
    inf.gelu_cfg["mlp.act.nogate"] = cfg;
    inf.gelu_cfg["mlp.act.nogate"].gate = false;

    // ---------------------------------------------------------------------------------------
    // Part A — production gelu_approx on real block-0 data: gate ON vs gate OFF, output imag.
    // ---------------------------------------------------------------------------------------
    const int d_pad  = inf.size.hidDim;
    const int e_pad  = inf.size.expDim;
    const int e_real = inf.size.getRealFfDim();
    const int T_val  = default_t_sweep_val();
    const std::string io_path = all_blocks_io_path(default_all_blocks_io_dir(), 0, T_val);

    if (probe_io_file(io_path)) {
        auto io = read_block0_io(io_path, "pre_gelu", "post_gelu");
        const int q = T_val - 1;
        auto xq_pad = matrix::pad_vector(io.inp[q], e_pad);

        std::cout << "\n[Part A] production gelu_approx output imaginary (real block-0 data, token "
                  << q << ")\n";

        // Fresh encode per run so neither call sees the other's level/state.
        auto run_gelu = [&](const std::string& name) -> ImStats {
            PackedCtx x = encode_linear_input(inf, xq_pad, e_pad, d_pad);
            {
                auto xin = decrypt_cplx(inf, x);
                print_im(std::string("input x (") + name + ")", im_stats(xin, inf.slots));
            }
            PackedCtx y = gelu_approx(inf, x, name);
            auto yc = decrypt_cplx(inf, y);
            ImStats st = im_stats(yc, inf.slots);
            print_im(std::string("gelu out [") + name + "]", st);
            // real accuracy vs torch GT, for context
            std::vector<double> yre(yc.size());
            for (size_t i = 0; i < yc.size(); ++i) yre[i] = yc[i].real();
            auto y_pad = decode_linear_output(inf.packing, yre, inf.slots, d_pad, e_pad);
            std::vector<double> yv(y_pad.begin(), y_pad.begin() + e_real);
            auto s = compare_vec(yv, io.res[q]);
            std::cout << "    (real vs GT: mean_abs=" << std::scientific << std::setprecision(3)
                      << s.mean_abs << "  max_abs=" << s.max_abs << ")\n";
            return st;
        };
        ImStats on  = run_gelu("mlp.act");          // gate as configured (ON for a gate config)
        ImStats off = run_gelu("mlp.act.nogate");   // gate forced OFF, same config otherwise
        const double gate_ratio = off.max_im > 0 ? on.max_im / off.max_im : 0.0;
        std::cout << std::scientific << std::setprecision(3)
                  << "  [A verdict] gate-ON max|Im|=" << on.max_im
                  << "  gate-OFF max|Im|=" << off.max_im
                  << "  (gate/nogate=" << std::fixed << std::setprecision(2) << gate_ratio << "x);"
                  << "  |Im|:|Re| on signal=" << std::scientific << std::setprecision(2) << on.ratio() << "\n"
                  << "              gate/nogate ~1x => the gate adds NO imaginary; any |Im| is shared-path\n"
                  << "              (softsign square / inv_sqrt). Judge severity by |Im|:|Re| vs the real error above.\n";
    } else {
        std::cout << "\n[Part A] SKIP (no IO file " << io_path << ")\n";
    }

    // ---------------------------------------------------------------------------------------
    // Part B — isolated gate-exp core, |Im| growth per square, no-cleanse vs cleanse-after.
    // ---------------------------------------------------------------------------------------
    std::cout << "\n[Part B] gate-exp core: imaginary lane per square (axsq swept in-domain)\n";
    if (cfg.gate_cheb_coeffs.empty()) {
        std::cout << "  SKIP: config has no gate_cheb_coeffs (gate-OFF calibration).\n";
        SUCCEED();
        return;
    }

    const int K  = std::max(1, cfg.exp_iters);
    const int NP = 256;
    // axsq = (h*a*x)^2 (nonlinear.cu:86-87); sweep x so axsq spans [0, |gate_cheb_a|] (in-domain).
    const double axsq_hi = std::abs(cfg.gate_cheb_a);
    std::vector<double> axsq(NP);
    for (int i = 0; i < NP; ++i) axsq[i] = axsq_hi * (double)i / (NP - 1);

    auto enc_real = [&](const std::vector<double>& val) {
        std::vector<double> slotv(inf.slots, 0.0);
        for (int i = 0; i < NP; ++i) slotv[i] = val[i];
        auto pt = inf.cc()->MakeCKKSPackedPlaintext(slotv, 1, 0);
        return inf.pack(encrypt(inf.cc(), pt, inf.fhe->pk()));
    };

    // Faithful to nonlinear.cu gate arm; `cleanse_after` adds the proposed softmax-style fix.
    // Returns {|Im| right after the bootstrap, |Im| after the last square}.
    auto run_core = [&](bool cleanse_after) -> std::pair<double, double> {
        PackedCtx gy = eval_chebyshev_series(inf.cc_ctx(), enc_real(axsq),
                                             cfg.gate_cheb_coeffs, cfg.gate_cheb_a, cfg.gate_cheb_b);
        std::cout << "  --- " << (cleanse_after ? "WITH cleanse-after-squares (fix)"
                                                : "NO cleanse (production)") << " ---\n";
        print_im("after cheb", im_stats(decrypt_cplx(inf, gy), NP));
        inf.fhe->bootstrap_hint(gy, 16);
        const double im_boot = im_stats(decrypt_cplx(inf, gy), NP).max_im;
        print_im("after bootstrap", im_stats(decrypt_cplx(inf, gy), NP));
        double im_last = im_boot;
        for (int i = 0; i < K; ++i) {
            inf.fhe->inplace_square(gy);
            ImStats st = im_stats(decrypt_cplx(inf, gy), NP);
            im_last = st.max_im;
            print_im("after square " + std::to_string(i + 1) + "/" + std::to_string(K), st);
        }
        if (cleanse_after) {
            inf.fhe->inplace_im_cleanse(gy);   // 2*Re: production would fold the 0.5 into quarter_mask
            print_im("after im_cleanse", im_stats(decrypt_cplx(inf, gy), NP));
        }
        return {im_boot, im_last};
    };
    auto nc = run_core(/*cleanse_after=*/false);
    run_core(/*cleanse_after=*/true);

    const double growth = nc.first > 0 ? nc.second / nc.first : 0.0;
    std::cout << std::scientific << std::setprecision(3)
              << "\n[B verdict] gate squares amplify |Im| " << std::fixed << std::setprecision(1) << growth
              << "x over " << K << " squares (post-bts " << std::scientific << nc.first
              << " -> post-square " << nc.second << "), max|Re|~1.\n"
              << "            Mechanism is real (~2^K), but judge whether the ABSOLUTE |Im| is large\n"
              << "            relative to the real signal before treating it as the breakage: a floor-\n"
              << "            level (~1e-10) |Im| is negligible; only a near-signal |Im| is a real bug.\n"
              << "            im_cleanse collapses it to the bts floor regardless (the softmax-style fix).\n";
    SUCCEED();
}
