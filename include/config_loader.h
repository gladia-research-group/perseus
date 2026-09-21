#pragma once

#include "io/json_utils.h"
#include "nonlinear.h"

#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace config_loader {

using namespace json_utils;

template <typename T>
struct Binder {
    const std::string& obj;
    T&                  out;

    void req(const char* k, double              T::* m) { out.*m = find_double_field(obj, k); }
    void req(const char* k, int                 T::* m) { out.*m = find_int_field(obj, k); }
    void req(const char* k, bool                T::* m) { out.*m = find_bool_field(obj, k); }
    void req(const char* k, std::string         T::* m) { out.*m = find_string_field(obj, k); }
    void req(const char* k, std::vector<double> T::* m) { out.*m = find_double_array_field(obj, k); }

    void opt(const char* k, double              T::* m) { load_or_warn(out.*m, obj, k); }
    void opt(const char* k, int                 T::* m) { load_or_warn(out.*m, obj, k); }
    void opt(const char* k, bool                T::* m) { load_or_warn(out.*m, obj, k); }
    void opt(const char* k, std::vector<double> T::* m) { load_or_warn(out.*m, obj, k); }

    // Required field mapping a JSON string onto an enum value.
    template <typename E>
    void req_enum(const char* k, E T::* m,
                  std::initializer_list<std::pair<const char*, E>> table) {
        const std::string s = find_string_field(obj, k);
        for (const auto& [name, val] : table)
            if (s == name) { out.*m = val; return; }
        throw std::runtime_error("unknown " + std::string(k) + " value: " + s);
    }
};

inline void describe(Binder<NormConfig>& b) {
    b.req("eps",     &NormConfig::epsilon);
    b.req("center_scale", &NormConfig::center_scale);

    const bool has_method = b.obj.find("\"method\"") != std::string::npos;
    if (has_method) {
        b.req_enum("method", &NormConfig::nr_init_method, {
            {"taylor", NRInitMethod::TAYLOR},
            {"remez",  NRInitMethod::REMEZ},
        });
    }
    
    b.opt("nr_iters", &NormConfig::nr_iters);

    switch (b.out.nr_init_method) {
        case NRInitMethod::TAYLOR:
            b.req("z0", &NormConfig::taylor_z0);
            break;
        case NRInitMethod::REMEZ:
            b.req("z0",        &NormConfig::taylor_z0);   // inactive-lane floor (median c²·var)
            b.req("inv_out_scale", &NormConfig::inv_out_scale);
            b.req("Ncoeffs",   &NormConfig::Ncoeffs);
            b.req("Dcoeffs",   &NormConfig::Dcoeffs);
            b.req("lin_alpha", &NormConfig::lin_alpha);
            b.req("lin_beta",  &NormConfig::lin_beta);
            b.req("gs_lo",     &NormConfig::gs_lo);
            b.req("gs_hi",     &NormConfig::gs_hi);
            b.req("gs_iters",  &NormConfig::gs_iters);
            b.opt("center_scale_sq", &NormConfig::center_scale_sq);  // per-token c_eff²; absent ⇒ plain c²
            b.opt("precise_var_bts", &NormConfig::precise_var_bts);  // absent ⇒ 1-iter (GPT-2 configs)
            break;
    }
}

inline void describe(Binder<SoftmaxConfig>& b) {
    b.req("n_squarings",            &SoftmaxConfig::log2delta1);
    b.req("refinement_iters",       &SoftmaxConfig::log2delta2);
    b.req("clip_lo",                &SoftmaxConfig::clip_lo);
    b.req("clip_hi",                &SoftmaxConfig::clip_hi);
    b.req("poly_coeffs",            &SoftmaxConfig::poly_coeffs);
    b.req("init_alpha",             &SoftmaxConfig::init_alpha);
    b.req("init_beta",              &SoftmaxConfig::init_beta);
    b.req("refine_alpha",           &SoftmaxConfig::refine_alpha);
    b.req("refine_beta",            &SoftmaxConfig::refine_beta);
    b.req("gs_iters_scaled",        &SoftmaxConfig::gs_iters_scaled);
    b.req("gs_iters_refine_scaled", &SoftmaxConfig::gs_iters_refine_scaled);
    b.req("per_step_refine_iters",  &SoftmaxConfig::per_step_refine_iters);
    b.opt("cheb_coeffs",            &SoftmaxConfig::cheb_coeffs);
    b.opt("cheb_a",                 &SoftmaxConfig::cheb_a);
    b.opt("cheb_b",                 &SoftmaxConfig::cheb_b);
    b.opt("sm_kc_r",                &SoftmaxConfig::sm_kc_r);   // per-step per-kc refine scaling; absent ⇒ kc_ref proxy
}

inline void describe(Binder<GeLUConfig>& b) {
    b.req_enum("method", &GeLUConfig::method, {
        {"softsign_inv_sqrt", GeLUMethod::SOFTSIGN_INV_SQRT},
        {"chebyshev",         GeLUMethod::CHEBYSHEV},
        {"thor_composite",    GeLUMethod::THOR_COMPOSITE},
    });

    switch (b.out.method) {
        case GeLUMethod::SOFTSIGN_INV_SQRT:
            b.opt("gate",      &GeLUConfig::gate);
            b.req("a",         &GeLUConfig::a);
            b.req("b",         &GeLUConfig::b);
            b.req("c",         &GeLUConfig::c);
            b.req("xmax",      &GeLUConfig::xmax);
            b.req("z_min",     &GeLUConfig::z_min);
            b.req("z_max",     &GeLUConfig::z_max);
            b.req("gs_lo",     &GeLUConfig::gs_lo);
            b.req("gs_hi",     &GeLUConfig::gs_hi);
            b.req("lin_alpha", &GeLUConfig::lin_alpha);
            b.req("lin_beta",  &GeLUConfig::lin_beta);
            b.req("inv_out_scale", &GeLUConfig::inv_out_scale);
            b.req("Ncoeffs",   &GeLUConfig::Ncoeffs);
            b.req("Dcoeffs",   &GeLUConfig::Dcoeffs);
            b.opt("exp_iters",    &GeLUConfig::exp_iters);
            b.opt("newton_iters", &GeLUConfig::newton_iters);
            b.opt("gs_iters",     &GeLUConfig::gs_iters);
            b.opt("gate_cheb_coeffs", &GeLUConfig::gate_cheb_coeffs);
            b.opt("gate_cheb_a",      &GeLUConfig::gate_cheb_a);
            b.opt("gate_cheb_b",      &GeLUConfig::gate_cheb_b);
            break;
        case GeLUMethod::CHEBYSHEV:
            b.req("cheb_coeffs", &GeLUConfig::cheb_coeffs);
            b.req("cheb_a",      &GeLUConfig::cheb_a);
            b.req("cheb_b",      &GeLUConfig::cheb_b);
            break;
        case GeLUMethod::THOR_COMPOSITE:
            b.req("xmax",    &GeLUConfig::xmax);
            b.req("thor_p1", &GeLUConfig::thor_p1);
            b.req("thor_p2", &GeLUConfig::thor_p2);
            b.opt("thor_p1_cheb", &GeLUConfig::thor_p1_cheb);
            b.opt("thor_p2_cheb", &GeLUConfig::thor_p2_cheb);
            b.opt("thor_p1_a",    &GeLUConfig::thor_p1_a);
            b.opt("thor_p1_b",    &GeLUConfig::thor_p1_b);
            b.opt("thor_p2_a",    &GeLUConfig::thor_p2_a);
            b.opt("thor_p2_b",    &GeLUConfig::thor_p2_b);
            break;
    }
}

struct ModelConfig {
    int n_layers = 0;
    int n_embd   = 0;
    int n_head   = 0;
    int n_inner  = 0;
};

inline void describe(Binder<ModelConfig>& b) {
    b.opt("n_layers", &ModelConfig::n_layers);
    b.opt("n_embd",   &ModelConfig::n_embd);
    b.opt("n_head",   &ModelConfig::n_head);
    b.opt("n_inner",  &ModelConfig::n_inner);
}

inline void describe(Binder<CutMaxCalib>& b) {
    b.req("entry_scale",     &CutMaxCalib::entry_scale);
    b.req("newton_per_pass", &CutMaxCalib::newton_per_pass);
    b.req("newton_polish",   &CutMaxCalib::newton_polish);
    b.req("gs_sum_iters",    &CutMaxCalib::gs_sum_iters);
    b.req("sum_lo",          &CutMaxCalib::sum_lo);
    b.req("sum_hi",          &CutMaxCalib::sum_hi);
    b.req("p",               &CutMaxCalib::p);
    b.req("c",               &CutMaxCalib::c);
    b.req("m",               &CutMaxCalib::m);
    b.req("s2_hi",           &CutMaxCalib::s2_hi);
    b.req("passes",          &CutMaxCalib::passes);
    b.req("ex2",             &CutMaxCalib::ex2);
    b.req("chord_a",         &CutMaxCalib::chord_a);
    b.req("chord_b",         &CutMaxCalib::chord_b);
    b.req("cascade_iters",   &CutMaxCalib::cascade_iters);
}

struct ParsedConfigs {
    ModelConfig                                     model;
    std::unordered_map<std::string, NormConfig>     norm;
    std::unordered_map<std::string, SoftmaxConfig>  softmax;
    std::unordered_map<std::string, GeLUConfig>     softgelu;
    CutMaxCalib                                     cutmax;
    bool                                            has_cutmax = false;
};

// Parse one JSON object into a T using its `describe(Binder<T>&)` schema.
template <typename T>
inline T parse_entry(const std::string& obj) {
    T c;
    Binder<T> b{obj, c};
    describe(b);
    return c;
}

// Parse a per-layer section ("norm", "softmax", …) into a name→T map.
template <typename T>
inline void parse_section(const std::string& text, const std::string& section,
                          std::unordered_map<std::string, T>& dst) {
    walk_section(text, section, [&](const std::string& k, const std::string& o) {
        dst[k] = parse_entry<T>(o);
    });
}

inline ParsedConfigs parse_configs_json(const std::string& text) {
    ParsedConfigs out;
    std::string model_obj = find_object_field(text, "model");
    if (!model_obj.empty()) out.model = parse_entry<ModelConfig>(model_obj);
    parse_section(text, "norm",     out.norm);
    parse_section(text, "softmax",  out.softmax);
    parse_section(text, "softgelu", out.softgelu);
    std::string cm_obj = find_object_field(text, "cutmax");
    if (!cm_obj.empty()) {
        out.cutmax = parse_entry<CutMaxCalib>(cm_obj);
        out.has_cutmax = true;
    }
    return out;
}

inline ParsedConfigs load_configs(const std::string& dir_path) {
    return parse_configs_json(read_file_to_string(
        (std::filesystem::path(dir_path) / "configs.json").string()));
}

} // namespace config_loader
