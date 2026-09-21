#pragma once

#include "fideslib_wrapper.h"
#include "inference.h"
#include "weight_loader.h"
#include "io/json_utils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace test_helpers {

// Minimal JSON helpers shared by ground-truth tests (linear / mha / softmax /
// layernorm). They handle the (inp, res) 2D-array payloads produced by the
// preprocess/gather_*.py scripts — not a full JSON parser.

inline void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
}

inline double parse_number(const std::string& s, size_t& i) {
    skip_ws(s, i);
    if (i >= s.size()) throw std::runtime_error("Unexpected end while parsing number");
    const char* start = s.c_str() + i;
    char* end = nullptr;
    double v = std::strtod(start, &end);
    if (end == start) throw std::runtime_error("Failed to parse number");
    i = static_cast<size_t>(end - s.c_str());
    return v;
}

// Read an env var, falling back to a default if unset/empty.
inline std::string env_or(const char* key, const std::string& fallback) {
    const char* v = std::getenv(key);
    return (v && *v) ? std::string(v) : fallback;
}

// Open a WeightStore from either a directory or a .zip archive.
inline weight_loader::WeightStore load_store(const std::string& path) {
    const std::string suffix = ".zip";
    if (path.size() >= suffix.size() &&
        path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return weight_loader::WeightStore::from_zip(path);
    }
    return weight_loader::WeightStore::from_dir(path);
}

inline std::vector<double> parse_row_at(const std::string& text, size_t& i) {
    skip_ws(text, i);
    if (i >= text.size() || text[i] != '[') throw std::runtime_error("Expected '['");
    ++i;
    std::vector<double> row;
    while (true) {
        skip_ws(text, i);
        if (i >= text.size()) throw std::runtime_error("Unexpected end in row");
        if (text[i] == ']') { ++i; break; }
        row.push_back(parse_number(text, i));
        skip_ws(text, i);
        if (i >= text.size()) throw std::runtime_error("Unexpected end after number");
        if (text[i] == ',') { ++i; continue; }
        if (text[i] == ']') { ++i; break; }
        throw std::runtime_error("Malformed row");
    }
    return row;
}

inline std::vector<std::vector<double>> parse_2d_array(
        const std::string& text, const std::string& key) {
    const std::string quoted = "\"" + key + "\"";
    size_t pos = text.find(quoted);
    if (pos == std::string::npos) throw std::runtime_error("Missing key: " + key);
    pos = text.find('[', pos + quoted.size());
    if (pos == std::string::npos) throw std::runtime_error("Malformed array: " + key);

    size_t i = pos;
    skip_ws(text, i);
    if (i >= text.size() || text[i] != '[') throw std::runtime_error("Expected '['");
    ++i;

    std::vector<std::vector<double>> rows;
    while (true) {
        skip_ws(text, i);
        if (i >= text.size()) throw std::runtime_error("Unexpected end in outer array");
        if (text[i] == ']') { ++i; break; }
        rows.push_back(parse_row_at(text, i));
        skip_ws(text, i);
        if (i < text.size() && text[i] == ',') { ++i; continue; }
    }
    return rows;
}

// 3D parser for (H, T, T) taps like pre_softmax / post_softmax in the
// all_blocks gather payload. Outer dim is heads; inner 2D matrices are
// causal-masked attention scores / softmax probabilities.
inline std::vector<std::vector<std::vector<double>>> parse_3d_array(
        const std::string& text, const std::string& key) {
    const std::string quoted = "\"" + key + "\"";
    size_t pos = text.find(quoted);
    if (pos == std::string::npos) throw std::runtime_error("Missing key: " + key);
    pos = text.find('[', pos + quoted.size());
    if (pos == std::string::npos) throw std::runtime_error("Malformed array: " + key);

    size_t i = pos;
    skip_ws(text, i);
    if (i >= text.size() || text[i] != '[') throw std::runtime_error("Expected '['");
    ++i;

    std::vector<std::vector<std::vector<double>>> cube;
    while (true) {
        skip_ws(text, i);
        if (i >= text.size()) throw std::runtime_error("Unexpected end in outer array");
        if (text[i] == ']') { ++i; break; }
        if (text[i] != '[') throw std::runtime_error("Expected nested '[' for 3D");
        ++i;

        std::vector<std::vector<double>> mat;
        while (true) {
            skip_ws(text, i);
            if (i >= text.size()) throw std::runtime_error("Unexpected end in matrix");
            if (text[i] == ']') { ++i; break; }
            mat.push_back(parse_row_at(text, i));
            skip_ws(text, i);
            if (i < text.size() && text[i] == ',') { ++i; continue; }
        }
        cube.push_back(std::move(mat));

        skip_ws(text, i);
        if (i < text.size() && text[i] == ',') { ++i; continue; }
    }
    return cube;
}

// Bundle of (inp, res) ground-truth arrays parsed from a *_io.json file.
struct IoArrays {
    std::vector<std::vector<double>> inp;
    std::vector<std::vector<double>> res;
};

inline IoArrays read_io_arrays(const std::string& path) {
    std::string text = json_utils::read_file_to_string(path);
    return {parse_2d_array(text, "inp"), parse_2d_array(text, "res")};
}

// Probe a ground-truth JSON; prints a "(skip: missing ...)" line if absent.
// Tests use the return value to early-out of a per-T sweep.
inline bool probe_io_file(const std::string& path) {
    std::ifstream probe(path);
    if (!probe) {
        std::cout << "  (skip: missing ground truth " << path << ")\n";
        return false;
    }
    return true;
}

// Default checkpoint location for GPT-2 tests; overridable via WEIGHTS_PATH.
inline std::string default_weights_path() {
    return env_or(
        "WEIGHTS_PATH",
        "/leonardo_work/IscrC_eff-SAM2/azirilli/he-aware-training/"
        "checkpoints/openai-community/gpt2/lm_eval/classic/weights.bin.zip");
}

// Default bootstrap iterations for tests; overridable via BTS_ITERATIONS env var.
// Single point to bump from {1, 2, ...} across every CKKS-using test.
inline uint32_t default_bts_iterations() {
    return static_cast<uint32_t>(std::stoi(env_or("BTS_ITERATIONS", "1")));
}

// Default ring size exponent; overridable via LOGN env var. logN=16 is the
// production GPT-2 config; logN=17 doubles the ring (needed once the LM head's
// 65536-wide output must fit one ciphertext, and to clear 128-bit security).
inline int default_logN() {
    return std::stoi(env_or("LOGN", "16"));
}

// Default sweep T (sequence length) for tests; overridable via T_SWEEP_VAL env var.
// Single point to switch e.g. {16, 128, 256, ...} across every T-parameterized test.
inline int default_t_sweep_val() {
    return std::stoi(env_or("T_SWEEP_VAL", "128"));
}

// CKKS modulus-chain options for tests. Defaults reproduce the production
// CKKSContextOptions exactly (logN/bts_iterations from their own env helpers);
// the remaining knobs are env-overridable so an experiment can swap in a different
// crypto param set WITHOUT forking the test. Each falls back to the struct default
// when its env var is absent, so canonical runs are byte-for-byte unchanged.
//
// THOR param set (Moon et al. 2024): FIRST_MOD_BITS=53 (q0~53b),
// BTP_SCALE_BITS=41 (qi~41b), BTP_DEPTH_OVERHEAD=15 (K=15), level_budget {4,3}
// (CtS 4 / StC 3), H_WEIGHT=192 (sparse). depth (L=13) and logN (2^16) already match.
inline CKKSContextOptions default_ckks_options() {
    return ckks_options_from_env();   // production env→options plumbing (fideslib_wrapper.h)
}

// Default calibrated approximation configs (configs.json); overridable via CONFIGS_PATH.
inline std::string default_configs_path() {
    return env_or(
        "CONFIGS_PATH",
        "/leonardo_work/IscrC_eff-SAM2/azirilli/he-aware-training/"
        "configs/model/approximation/hybrid/configs.json");
}

// Per-block bts placement plans from FHE_BOOTSTRAP_PLACEMENTS_DIR (empty = eager mode).
// Parsed once at the boundary and passed to GPT2Model::load, mirroring default_configs_path.
inline BlockPlans default_block_plans(int n_blocks) {
    const char* dir = std::getenv("FHE_BOOTSTRAP_PLACEMENTS_DIR");
    if (!dir || !*dir) return {};
    // +1 MERGED tail stage: block_<n_blocks> = ln_f + lm_head (one subgraph). A missing file
    // parses to an invalid plan -> eager tail (backward compatible).
    return load_block_plans(dir, n_blocks + 1);
}

// Default ground-truth IO directory (data/<subdir>); overridable via `env_key`.
inline std::string default_io_dir(const char* env_key,
                                  const std::string& subdir) {
    return env_or(env_key,
        "/leonardo/pub/userexternal/azirilli/he-aware-training_data/" + subdir);
}

inline std::vector<double> decrypt_slots(CKKSContext& fhe, const Ctx& ct) {
    return decrypt(fhe.cc, ct, fhe.sk());
}

inline std::vector<double> decrypt_slots(Inference& inf, const Ctx& ct) {
    return decrypt(inf.cc(), ct, inf.fhe->sk());
}

// PackedCtx overload — the .ct boundary at decrypt time.
inline std::vector<double> decrypt_slots(Inference& inf, const PackedCtx& pc) {
    return decrypt_slots(inf, pc.ct);
}

// Per-slot accuracy stats restricted to the indices in `active`. Useful for
// MHA tests where most of the slot space is padding we want to ignore.
struct SlotStats {
    double max_abs  = 0.0;
    double mean_abs = 0.0;
    double max_rel  = 0.0;
    double mean_rel = 0.0;
    double max_ref  = 0.0;
    double max_got  = 0.0;
    double rmse     = 0.0;
    int    n        = 0;
};

inline SlotStats compute_slot_stats(const std::vector<double>& got,
                                    const std::vector<double>& ref,
                                    const std::vector<int>& active,
                                    double rel_eps = 1e-6) {
    SlotStats s;
    if (active.empty()) return s;
    double sum_abs = 0.0, sum_rel = 0.0, sum_sq = 0.0;
    for (int idx : active) {
        const double a = got[idx];
        const double r = ref[idx];
        const double abs_err = std::abs(a - r);
        const double rel_err = abs_err / std::max(std::abs(r), rel_eps);
        sum_abs += abs_err;
        sum_rel += rel_err;
        sum_sq  += abs_err * abs_err;
        s.max_abs = std::max(s.max_abs, abs_err);
        s.max_rel = std::max(s.max_rel, rel_err);
        s.max_ref = std::max(s.max_ref, std::abs(r));
        s.max_got = std::max(s.max_got, std::abs(a));
    }
    s.n        = static_cast<int>(active.size());
    s.mean_abs = sum_abs / s.n;
    s.mean_rel = sum_rel / s.n;
    s.rmse     = std::sqrt(sum_sq / s.n);
    return s;
}

// Same idea but restricted to slots where |ref| > threshold — drops the
// near-zero noise floor from the rel-err picture.
struct FilteredStats {
    double max_rel  = 0.0;
    double mean_rel = 0.0;
    double mean_abs = 0.0;
    int    n_above  = 0;
    double coverage = 0.0;
};

inline FilteredStats compute_filtered_stats(const std::vector<double>& got,
                                            const std::vector<double>& ref,
                                            const std::vector<int>& active,
                                            double threshold) {
    FilteredStats s;
    if (active.empty()) return s;
    double sum_rel = 0.0, sum_abs = 0.0;
    for (int idx : active) {
        const double r = ref[idx];
        if (std::abs(r) <= threshold) continue;
        const double abs_err = std::abs(got[idx] - r);
        const double rel_err = abs_err / std::abs(r);
        sum_abs += abs_err;
        sum_rel += rel_err;
        s.max_rel = std::max(s.max_rel, rel_err);
        s.n_above++;
    }
    if (s.n_above > 0) {
        s.mean_rel = sum_rel / s.n_above;
        s.mean_abs = sum_abs / s.n_above;
    }
    s.coverage = static_cast<double>(s.n_above) / active.size();
    return s;
}

// Dense overload: compares the first ref.size() entries by position (no active mask).
inline FilteredStats compute_filtered_stats(const std::vector<double>& got,
                                            const std::vector<double>& ref,
                                            double threshold) {
    FilteredStats s;
    const int n = static_cast<int>(ref.size());
    if (n == 0) return s;
    double sum_rel = 0.0, sum_abs = 0.0;
    for (int i = 0; i < n; ++i) {
        const double r = ref[i];
        if (std::abs(r) <= threshold) continue;
        const double abs_err = std::abs(got[i] - r);
        const double rel_err = abs_err / std::abs(r);
        sum_abs += abs_err;
        sum_rel += rel_err;
        s.max_rel = std::max(s.max_rel, rel_err);
        s.n_above++;
    }
    if (s.n_above > 0) {
        s.mean_rel = sum_rel / s.n_above;
        s.mean_abs = sum_abs / s.n_above;
    }
    s.coverage = static_cast<double>(s.n_above) / n;
    return s;
}

inline void report_filtered(const std::string& label, double thr,
                            const FilteredStats& s) {
    std::cout << std::scientific << std::setprecision(3)
              << "[filt] " << std::left << std::setw(28) << label
              << " thr=" << thr
              << " n=" << s.n_above
              << " cov=" << std::fixed << std::setprecision(3) << s.coverage
              << std::scientific << std::setprecision(3)
              << " mean_abs=" << s.mean_abs
              << " max_rel=" << s.max_rel
              << " mean_rel=" << s.mean_rel
              << "\n";
    std::cout.unsetf(std::ios::fixed);
}

// Run a threshold sweep against `active` indices and print one row per threshold.
inline void report_filtered_sweep(const std::string& label,
                                  const std::vector<double>& got,
                                  const std::vector<double>& ref,
                                  const std::vector<int>& active,
                                  std::initializer_list<double> thrs =
                                      {1e-3, 1e-2, 1e-1, 1.0}) {
    std::cout << "\n--- Filtered rel-err sweep (|ref| > thr) [" << label << "] ---\n";
    for (double t : thrs) {
        auto f = compute_filtered_stats(got, ref, active, t);
        report_filtered("above floor", t, f);
    }
}

// Split sweep by ref sign (>= 0 vs < 0) for activation-sensitive diagnostics.
inline void report_filtered_sweep_split(const std::string& label,
                                        const std::vector<double>& got,
                                        const std::vector<double>& ref,
                                        const std::vector<int>& active,
                                        std::initializer_list<double> thrs =
                                            {1e-3, 1e-2, 1e-1, 1.0}) {
    std::vector<int> pos;
    std::vector<int> neg;
    pos.reserve(active.size());
    neg.reserve(active.size());
    for (int idx : active) {
        if (ref[idx] >= 0.0) {
            pos.push_back(idx);
        } else {
            neg.push_back(idx);
        }
    }

    std::cout << "\n--- Filtered rel-err sweep (|ref| > thr) [" << label
              << "] ---\n";
    for (double t : thrs) {
        report_filtered("pos (ref >= 0)", t, compute_filtered_stats(got, ref, pos, t));
        report_filtered("neg (ref < 0)", t, compute_filtered_stats(got, ref, neg, t));
    }
}

// Dense overload of the sweep — uses every index in [0, ref.size()).
inline void report_filtered_sweep(const std::string& label,
                                  const std::vector<double>& got,
                                  const std::vector<double>& ref,
                                  std::initializer_list<double> thrs =
                                      {1e-3, 1e-2, 1e-1, 1.0}) {
    std::cout << "\n--- Filtered rel-err sweep (|ref| > thr) [" << label << "] ---\n";
    for (double t : thrs) {
        auto f = compute_filtered_stats(got, ref, t);
        report_filtered("above floor", t, f);
    }
}

// Dense rel/abs comparison shared by attn / mha / pre_gelu / full_block /
// layernorm / linear tests — single struct with every field anyone tracks
// (worst index + value, weighted MAPE, etc.).
struct AccStats {
    double max_abs   = 0.0;
    double mean_abs  = 0.0;
    double max_rel   = 0.0;
    double mean_rel  = 0.0;
    double min_ref   = 0.0;
    double max_ref   = 0.0;
    double min_got   = 0.0;
    double max_got   = 0.0;
    double w_mape    = 0.0;     // sum_abs_err / sum_abs_ref
    int    worst_idx = -1;
    double worst_got = 0.0;
    double worst_ref = 0.0;
    int    n         = 0;
};

inline AccStats compare_vec(const std::vector<double>& got,
                            const std::vector<double>& ref,
                            double rel_eps = 1e-6) {
    AccStats s;
    const int n = static_cast<int>(ref.size());
    if (n == 0) return s;
    double sum_abs = 0.0, sum_rel = 0.0, sum_ref = 0.0;
    s.min_ref = ref[0];
    s.max_ref = ref[0];
    s.min_got = got[0];
    s.max_got = got[0];
    for (int i = 0; i < n; ++i) {
        const double abs_err = std::abs(got[i] - ref[i]);
        const double denom   = std::max(std::abs(ref[i]), rel_eps);
        const double rel_err = abs_err / denom;
        if (abs_err > s.max_abs) {
            s.max_abs   = abs_err;
            s.worst_idx = i;
            s.worst_got = got[i];
            s.worst_ref = ref[i];
        }
        s.max_rel = std::max(s.max_rel, rel_err);
        s.min_ref = std::min(s.min_ref, ref[i]);
        s.max_ref = std::max(s.max_ref, ref[i]);
        s.min_got = std::min(s.min_got, got[i]);
        s.max_got = std::max(s.max_got, got[i]);
        sum_abs += abs_err;
        sum_rel += rel_err;
        sum_ref += std::abs(ref[i]);
    }
    s.n        = n;
    s.mean_abs = sum_abs / std::max(n, 1);
    s.mean_rel = sum_rel / std::max(n, 1);
    s.w_mape   = sum_abs / std::max(sum_ref, rel_eps);
    return s;
}

inline void report_acc(const std::string& tag, const AccStats& s) {
    std::cout << std::scientific << std::setprecision(3)
              << "[" << tag << "]"
              << "  max_abs="  << s.max_abs
              << "  mean_abs=" << s.mean_abs
              << "  max_rel="  << s.max_rel
              << "  mean_rel=" << s.mean_rel
              << "  ref_min="  << s.min_ref
              << "  ref_max="  << s.max_ref
              << "  got_min="  << s.min_got
              << "  got_max="  << s.max_got
              << "  worst_idx=" << s.worst_idx
              << std::endl;
}

// Split accuracy by ref sign (>= 0 vs < 0) for activation-sensitive diagnostics.
inline void report_acc_split(const std::string& tag,
                             const std::vector<double>& got,
                             const std::vector<double>& ref,
                             const std::vector<int>& active,
                             double rel_eps = 1e-6) {
    std::vector<double> got_low;
    std::vector<double> ref_low;
    std::vector<double> got_mid;
    std::vector<double> ref_mid;
    std::vector<double> got_high;
    std::vector<double> ref_high;
    got_low.reserve(active.size());
    ref_low.reserve(active.size());
    got_mid.reserve(active.size());
    ref_mid.reserve(active.size());
    got_high.reserve(active.size());
    ref_high.reserve(active.size());

    for (int idx : active) {
        const double r = ref[idx];
        if (r < -3.0) {
            got_low.push_back(got[idx]);
            ref_low.push_back(r);
        } else if (r < 3.0) {
            got_mid.push_back(got[idx]);
            ref_mid.push_back(r);
        } else {
            got_high.push_back(got[idx]);
            ref_high.push_back(r);
        }
    }

    report_acc(tag + " ref < -3", compare_vec(got_low, ref_low, rel_eps));
    report_acc(tag + " -3 <= ref < 3", compare_vec(got_mid, ref_mid, rel_eps));
    report_acc(tag + " ref >= 3", compare_vec(got_high, ref_high, rel_eps));
}

inline ::testing::AssertionResult RelErrLeFormat(
    const char* actual_expr,
    const char* ref_expr,
    const char* eps_expr,
    const std::vector<double>& actual,
    const std::vector<double>& ref,
    double eps) {
    if (actual.size() < ref.size()) {
        return ::testing::AssertionFailure()
            << actual_expr << ".size()=" << actual.size()
            << " < " << ref_expr << ".size()=" << ref.size();
    }
    constexpr double denom_floor = 1e-9;
    double max_rel = 0.0, sum_rel = 0.0;
    int worst_idx = -1;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double abs_err = std::abs(actual[i] - ref[i]);
        const double rel_err = abs_err / std::max(std::abs(ref[i]), denom_floor);
        sum_rel += rel_err;
        if (rel_err > max_rel) { max_rel = rel_err; worst_idx = static_cast<int>(i); }
    }
    if (max_rel <= eps) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
        << "relative error exceeds " << eps_expr << " (=" << eps << ")\n"
        << "  max_rel_err=" << max_rel << " at index " << worst_idx
        << "  (actual=" << actual[worst_idx] << ", ref=" << ref[worst_idx] << ")\n"
        << "  mean_rel_err=" << (sum_rel / static_cast<double>(ref.size()))
        << "  n=" << ref.size();
}

}

#define EXPECT_REL_LE(actual, ref, eps) \
    EXPECT_PRED_FORMAT3(::test_helpers::RelErrLeFormat, (actual), (ref), (eps))
#define ASSERT_REL_LE(actual, ref, eps) \
    ASSERT_PRED_FORMAT3(::test_helpers::RelErrLeFormat, (actual), (ref), (eps))
