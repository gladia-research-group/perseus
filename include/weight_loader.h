#pragma once

#include "attention.h"
#include "model/gpt2.h"
#include "model/layer_norm.h"
#include "config_loader.h"
#include "io/weight_io.h"
#include "math/matrix_ops.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace weight_loader {

class WeightStore {
public:
    static WeightStore from_zip(const std::string& zip_path) {
#ifndef WEIGHT_LOADER_WITH_LIBARCHIVE
        (void)zip_path;
        throw std::runtime_error("LibArchive not available.");
#else
        WeightStore s;
        auto manifest_bytes = weight_io::read_zip_entry(zip_path, "manifest.json");
        s.load_manifest(std::string(manifest_bytes.begin(), manifest_bytes.end()));
        for (const auto& kv : s.meta_) {
            auto raw = weight_io::read_zip_entry(zip_path, kv.second.path);
            s.data_[kv.first] = weight_io::decode_tensor(raw, kv.second.dtype);
        }
        return s;
#endif
    }

    static WeightStore from_dir(const std::string& dir_path) {
        WeightStore s;
        s.load_manifest(json_utils::read_file_to_string(
            (std::filesystem::path(dir_path) / "manifest.json").string()));
        for (const auto& kv : s.meta_) {
            const auto path = (std::filesystem::path(dir_path) / kv.second.path).string();
            s.data_[kv.first] = weight_io::decode_tensor(weight_io::read_file_to_bytes(path),
                                                         kv.second.dtype);
        }
        return s;
    }

    bool                          has(const std::string& name) const { return data_.count(name) > 0; }
    const weight_io::TensorMeta&  meta(const std::string& name)   const { return meta_.at(name); }
    const std::vector<double>&    tensor(const std::string& name) const { return data_.at(name); }

    std::vector<double> tensor1d(const std::string& name, int n) const {
        const auto& m = meta_.at(name);
        if (m.shape.size() != 1 || m.shape[0] != n)
            throw std::runtime_error("Shape mismatch for " + name);
        return data_.at(name);
    }

    std::vector<double> tensor1d_any(const std::string& name) const {
        const auto& m = meta_.at(name);
        if (m.shape.size() != 1) throw std::runtime_error("Expected 1D tensor: " + name);
        return data_.at(name);
    }

    std::vector<std::vector<double>> tensor2d(const std::string& name,
                                              int rows, int cols) const {
        const auto& m = meta_.at(name);
        if (m.shape.size() != 2 || m.shape[0] != rows || m.shape[1] != cols)
            throw std::runtime_error("Shape mismatch for " + name);
        return reshape(data_.at(name), rows, cols);
    }

    std::vector<std::vector<double>> tensor2d_any(const std::string& name) const {
        const auto& m = meta_.at(name);
        if (m.shape.size() != 2) throw std::runtime_error("Expected 2D tensor: " + name);
        return reshape(data_.at(name), m.shape[0], m.shape[1]);
    }

private:
    void load_manifest(const std::string& text) {
        for (auto& t : weight_io::parse_manifest_json(text)) meta_[t.name] = std::move(t);
    }

    static std::vector<std::vector<double>> reshape(const std::vector<double>& flat,
                                                    int rows, int cols) {
        std::vector<std::vector<double>> out(rows, std::vector<double>(cols));
        size_t i = 0;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) out[r][c] = flat[i++];
        return out;
    }

    std::unordered_map<std::string, weight_io::TensorMeta> meta_;
    std::unordered_map<std::string, std::vector<double>>   data_;
};

struct MatrixPair {
    std::vector<std::vector<double>> real;
    std::vector<std::vector<double>> pad;
};

inline MatrixPair load_linear_weight(const WeightStore& store,
                                     const std::string& name,
                                     int d_in_real, int d_out_real,
                                     int d_in_pad,  int d_out_pad) {
    auto W = store.tensor2d_any(name);
    const int r = (int)W.size(), c = (int)W[0].size();
    MatrixPair out;
    if (r == d_out_real && c == d_in_real) {
        out.real = matrix::transpose(W);
        out.pad  = matrix::pad_matrix(out.real, d_in_pad, d_out_pad);
    } else if (r == d_out_pad && c == d_in_pad) {
        out.pad  = matrix::transpose(W);
        out.real = matrix::slice_matrix(out.pad, d_in_real, d_out_real);
    } else {
        throw std::runtime_error(
            "Unexpected shape for " + name + ": got " + matrix::shape_str(r, c) +
            " expected (d_out, d_in) = " + matrix::shape_str(d_out_real, d_in_real));
    }
    return out;
}

inline std::pair<std::vector<double>, std::vector<double>>
load_vector(const WeightStore& store, const std::string& name,
            int d_real, int d_pad) {
    auto v = store.tensor1d_any(name);
    if ((int)v.size() == d_real) return {v, matrix::pad_vector(v, d_pad)};
    if ((int)v.size() == d_pad)  return {matrix::slice_vec(v, 0, d_real), v};
    throw std::runtime_error("Unexpected vector size for " + name +
                             ": " + std::to_string(v.size()));
}

struct QKVMatrices {
    MatrixPair Q, K, V;
};

inline QKVMatrices load_qkv_weight(const WeightStore& store,
                                   const std::string& name,
                                   int d_real, int d_pad) {
    auto W = store.tensor2d_any(name);
    const int r = (int)W.size(), c = (int)W[0].size();
    const bool is_real = (r == 3 * d_real && c == d_real);
    const bool is_pad  = (r == 3 * d_pad  && c == d_pad);
    if (!is_real && !is_pad) {
        throw std::runtime_error(
            "Unexpected QKV shape for " + name + ": got " + matrix::shape_str(r, c) +
            " expected (3*d, d) = " + matrix::shape_str(3 * d_real, d_real));
    }
    const int d = is_real ? d_real : d_pad;
    auto Wt = matrix::transpose(W);
    auto Q  = matrix::slice_cols(Wt, 0,     d);
    auto K  = matrix::slice_cols(Wt, d,     d);
    auto V  = matrix::slice_cols(Wt, 2 * d, d);
    QKVMatrices out;
    if (is_real) {
        out.Q.real = std::move(Q); out.Q.pad = matrix::pad_matrix(out.Q.real, d_pad, d_pad);
        out.K.real = std::move(K); out.K.pad = matrix::pad_matrix(out.K.real, d_pad, d_pad);
        out.V.real = std::move(V); out.V.pad = matrix::pad_matrix(out.V.real, d_pad, d_pad);
    } else {
        out.Q.pad = std::move(Q); out.Q.real = matrix::slice_matrix(out.Q.pad, d_real, d_real);
        out.K.pad = std::move(K); out.K.real = matrix::slice_matrix(out.K.pad, d_real, d_real);
        out.V.pad = std::move(V); out.V.real = matrix::slice_matrix(out.V.pad, d_real, d_real);
    }
    return out;
}

struct QKVBiases {
    std::vector<double> Q_real, K_real, V_real;
    std::vector<double> Q_pad,  K_pad,  V_pad;
};

inline QKVBiases load_qkv_bias(const WeightStore& store, const std::string& name,
                               int d_real, int d_pad) {
    auto b = store.tensor1d_any(name);
    QKVBiases out;
    if ((int)b.size() == 3 * d_real) {
        out.Q_real = matrix::slice_vec(b, 0,          d_real);
        out.K_real = matrix::slice_vec(b, d_real,     d_real);
        out.V_real = matrix::slice_vec(b, 2 * d_real, d_real);
        out.Q_pad  = matrix::pad_vector(out.Q_real, d_pad);
        out.K_pad  = matrix::pad_vector(out.K_real, d_pad);
        out.V_pad  = matrix::pad_vector(out.V_real, d_pad);
    } else if ((int)b.size() == 3 * d_pad) {
        out.Q_pad  = matrix::slice_vec(b, 0,         d_pad);
        out.K_pad  = matrix::slice_vec(b, d_pad,     d_pad);
        out.V_pad  = matrix::slice_vec(b, 2 * d_pad, d_pad);
        out.Q_real = matrix::slice_vec(out.Q_pad, 0, d_real);
        out.K_real = matrix::slice_vec(out.K_pad, 0, d_real);
        out.V_real = matrix::slice_vec(out.V_pad, 0, d_real);
    } else {
        throw std::runtime_error("Unexpected QKV bias size for " + name +
                                 ": " + std::to_string(b.size()));
    }
    return out;
}

inline void fold_ln_gamma_into_linear(std::vector<std::vector<double>>& W_pad,
                                      const std::vector<double>& gamma_pad) {
    W_pad = matrix::mat_scale_rows(W_pad, gamma_pad);
}

inline double ln_gamma_descale(const NormConfig& cfg) {
    return cfg.nr_init_method == NRInitMethod::REMEZ ? 1.0 / cfg.inv_out_scale : 1.0;
}

inline std::vector<double> ln_input_shift(const std::vector<double>& gamma_pad,
                                          const std::vector<double>& beta_pad, int d_real) {
    std::vector<double> s(gamma_pad.size(), 0.0);
    for (int i = 0; i < d_real && i < (int)s.size(); ++i)
        if (gamma_pad[i] != 0.0) s[i] = beta_pad[i] / gamma_pad[i];
    return s;
}

struct GPT2WeightNames {
    std::string norm1_weight;
    std::string norm1_bias;
    std::string norm2_weight;
    std::string norm2_bias;

    std::string qkv_weight;
    std::string qkv_bias;

    std::string out_weight;
    std::string out_bias;

    std::string up_weight;
    std::string up_bias;
    std::string down_weight;
    std::string down_bias;
};

inline std::string gpt2_block_base(int block_idx) {
    return "transformer.h." + std::to_string(block_idx);
}

inline GPT2WeightNames gpt2_layer_names(int block_idx) {
    const std::string base = gpt2_block_base(block_idx);
    GPT2WeightNames n;
    n.norm1_weight = base + ".ln_1.weight";
    n.norm1_bias   = base + ".ln_1.bias";
    n.norm2_weight = base + ".ln_2.weight";
    n.norm2_bias   = base + ".ln_2.bias";

    n.qkv_weight = base + ".attn.c_attn.weight";
    n.qkv_bias   = base + ".attn.c_attn.bias";

    n.out_weight = base + ".attn.c_proj.weight";
    n.out_bias   = base + ".attn.c_proj.bias";

    n.up_weight   = base + ".mlp.c_fc.weight";
    n.up_bias     = base + ".mlp.c_fc.bias";
    n.down_weight = base + ".mlp.c_proj.weight";
    n.down_bias   = base + ".mlp.c_proj.bias";
    return n;
}

inline std::string gpt2_wte_name()     { return "transformer.wte.weight"; }
inline std::string gpt2_wpe_name()     { return "transformer.wpe.weight"; }
inline std::string gpt2_lm_head_name() { return "transformer.wte.weight"; }
// Pinned by plan weight_levels and live graphs — never rename.
inline std::string gpt2_lm_head_tile_key(int k) { return "lm_head_tile_" + std::to_string(k); }
inline std::string gpt2_final_ln_base() { return "transformer.ln_f"; }

struct EncodedGpt2Layer {
    std::unordered_map<std::string, std::vector<Ptx>> w;
    std::unordered_map<std::string, std::vector<std::vector<double>>> raw_w;
};

inline EncodedGpt2Layer encode_gpt2_layer_weights(
    Inference& inf,
    const WeightStore& store,
    const GPT2WeightNames& names,
    int d_real, int d_exp_real,
    int d_pad,  int d_exp_pad,
    int num_heads,
    cudaStream_t stream = nullptr,
    const BootstrapPlan& plan = {},
    double ln1_gamma_scale = 1.0,
    double ln2_gamma_scale = 1.0) {
    EncodedGpt2Layer out;

    auto [n1a, n1a_pad] = load_vector(store, names.norm1_weight, d_real, d_pad);
    auto  n1b           = load_vector(store, names.norm1_bias,   d_real, d_pad).first;
    auto [n2a, n2a_pad] = load_vector(store, names.norm2_weight, d_real, d_pad);
    auto  n2b           = load_vector(store, names.norm2_bias,   d_real, d_pad).first;
    for (auto& g : n1a)     g *= ln1_gamma_scale;
    for (auto& g : n1a_pad) g *= ln1_gamma_scale;
    for (auto& g : n2a)     g *= ln2_gamma_scale;
    for (auto& g : n2a_pad) g *= ln2_gamma_scale;

    auto qkv_w = load_qkv_weight(store, names.qkv_weight, d_real, d_pad);
    auto qkv_b = load_qkv_bias  (store, names.qkv_bias,   d_real, d_pad);
    
    if (fold_ln_affine("ln_1")) {   // gamma -> Q/K/V cols; beta rides input as ln_1.shift
        fold_ln_gamma_into_linear(qkv_w.Q.pad, n1a_pad);
        fold_ln_gamma_into_linear(qkv_w.K.pad, n1a_pad);
        fold_ln_gamma_into_linear(qkv_w.V.pad, n1a_pad);
    }
    auto W_Qr_pad = rearrange_qkv_weights(inf, qkv_w.Q.pad, num_heads);
    auto W_Kr_pad = rearrange_qkv_weights(inf, qkv_w.K.pad, num_heads);
    auto W_Vr_pad = rearrange_qkv_weights(inf, qkv_w.V.pad, num_heads);
    auto b_Qr_pad = rearrange_qkv_biases(inf, qkv_b.Q_pad, num_heads);
    auto b_Kr_pad = rearrange_qkv_biases(inf, qkv_b.K_pad, num_heads);
    auto b_Vr_pad = rearrange_qkv_biases(inf, qkv_b.V_pad, num_heads);

    auto W_O = load_linear_weight(store, names.out_weight,
                                  d_real, d_real, d_pad, d_pad);
    auto [b_out, b_out_pad] = load_vector(store, names.out_bias, d_real, d_pad);
    (void)b_out;
    auto W_Or_pad = rearrange_wo_weights(inf, W_O.pad, num_heads);

    auto W_up = load_linear_weight(store, names.up_weight,
                                   d_real, d_exp_real, d_pad, d_exp_pad);
    auto [b_up, b_up_pad] = load_vector(store, names.up_bias, d_exp_real, d_exp_pad);
    (void)b_up;

    if (fold_ln_affine("ln_2"))   // gamma -> up cols; beta rides input as ln_2.shift
        fold_ln_gamma_into_linear(W_up.pad, n2a_pad);

    auto W_down = load_linear_weight(store, names.down_weight,
                                     d_exp_real, d_real, d_exp_pad, d_pad);
    auto [b_down, b_down_pad] = load_vector(store, names.down_bias, d_real, d_pad);
    (void)b_down;

    const int wlvl = static_cast<int>(inf.fhe->bootstrap_output_level());
    auto wl = [&](const char* key) { return static_cast<int>(plan.weight_level(key, static_cast<uint32_t>(wlvl))); };

    // Family-interleaved staging: stage each family the moment it is encoded (worker-side
    // pinned staging, inf.pt_stage_hook) so the host never holds a whole block of freshly
    // encoded plaintexts — the hook (with FHE_STAGE_RELEASE_CPU) frees each family's OpenFHE
    // payload before the next family encodes. No-op when the hook is unset.
    auto put = [&](const std::string& key, std::vector<Ptx> v) {
        if (inf.pt_stage_hook) inf.pt_stage_hook(key, v);
        out.w[key] = std::move(v);
    };

    // LN gamma is inactive-lane-MASKED (filling/diagonal only, no-op for full
    // chunks): unmasked, padding-lane junk re-enters the stream at every LN with
    // gain c_eff·y_floor·gamma/s, compounds exponentially over blocks, and
    // detonates the deg-31 GELU once it crosses the basin (ViT k=12, block 6).
    if (fold_ln_affine("ln_1")) {   // beta rides the input as a (beta/gamma) shift
        auto s1 = ln_input_shift(n1a, n1b, d_real);
        put("ln_1.shift", {encode_ln_affine_param(inf, s1, d_pad, d_real, wl("ln_1.shift"), /*mask_inactive=*/true, stream)});
    } else {
        put("ln_1.weight", {encode_ln_affine_param(inf, n1a, d_pad, d_real, wl("ln_1.weight"), /*mask_inactive=*/true, stream)});
        put("ln_1.bias", {encode_ln_affine_param(inf, n1b, d_pad, d_real, wl("ln_1.bias"),   /*mask_inactive=*/true,  stream)});
    }
    if (fold_ln_affine("ln_2")) {
        auto s2 = ln_input_shift(n2a, n2b, d_real);
        put("ln_2.shift", {encode_ln_affine_param(inf, s2, d_pad, d_real, wl("ln_2.shift"), /*mask_inactive=*/true, stream)});
    } else {
        put("ln_2.weight", {encode_ln_affine_param(inf, n2a, d_pad, d_real, wl("ln_2.weight"), /*mask_inactive=*/true, stream)});
        put("ln_2.bias", {encode_ln_affine_param(inf, n2b, d_pad, d_real, wl("ln_2.bias"),   /*mask_inactive=*/true,  stream)});
    }

    put("q",         encode_weight_matrix(inf, W_Qr_pad,   d_pad,     d_pad,     /*target_level=*/wl("q"),   stream));
    put("q_bias",    {encode_bias_vector  (inf, b_Qr_pad,  d_pad,     d_pad,     /*fill=*/true, stream)});
    if (inf.complex) {   // fused KV linear: P = K_raw + i·V_raw, output-packed (one matmul for K & V)
        put("kv",      encode_weight_matrix_complex(inf, W_Kr_pad, W_Vr_pad, d_pad, d_pad,
                                                        /*target_level=*/wl("kv"), stream));
        put("kv_bias", {encode_bias_vector_complex(inf, b_Kr_pad, b_Vr_pad, d_pad, d_pad,
                                                       /*fill=*/true, stream)});
        inf.complex_weight_names.insert("kv");
        inf.complex_weight_names.insert("kv_bias");
    } else {
        put("k",         encode_weight_matrix(inf, W_Kr_pad,   d_pad,     d_pad,     /*target_level=*/wl("k"),   stream));
        put("k_bias",    {encode_bias_vector  (inf, b_Kr_pad,  d_pad,     d_pad,     /*fill=*/true, stream)});
        put("v",         encode_weight_matrix(inf, W_Vr_pad,   d_pad,     d_pad,     /*target_level=*/wl("v"),   stream));
        put("v_bias",    {encode_bias_vector  (inf, b_Vr_pad,  d_pad,     d_pad,     /*fill=*/true, stream)});
    }
    put("out",       encode_weight_matrix(inf, W_Or_pad,   d_pad,     d_pad,     /*target_level=*/wl("out"), stream));
    put("out_bias",  {encode_bias_vector  (inf, b_out_pad, d_pad,     d_pad,     /*fill=*/true, stream)});
    if (inf.tiled_mlp()) {
        if (inf.size.getRealFfDim() % d_pad != 0)
            throw std::runtime_error("tiled MLP: getRealFfDim must be a multiple of hidDim");
        const int N_TILES = inf.size.getRealFfDim() / d_pad;
        for (int j = 0; j < N_TILES; ++j) {
            const std::string tj = ".t" + std::to_string(j);
            auto W_up_j = matrix::slice_cols(W_up.pad, j * d_pad, d_pad);   // (d_pad x d_pad)
            auto b_up_j = matrix::slice_vec(b_up_pad,  j * d_pad, d_pad);
            put("up" + tj,          encode_weight_matrix(inf, W_up_j, d_pad, d_pad, /*target_level=*/wl("up"), stream));
            put("up" + tj + "_bias", {encode_bias_vector(inf, b_up_j, d_pad, d_pad, /*fill=*/true, stream)});

            auto W_down_j = matrix::slice_rows(W_down.pad, j * d_pad, d_pad); // (d_pad x d_pad)
            put("down" + tj,        encode_weight_matrix(inf, W_down_j, d_pad, d_pad, /*target_level=*/wl("down"), stream));
        }
        put("down_bias", {encode_bias_vector(inf, b_down_pad, d_pad, d_pad, /*fill=*/true, stream)});
    } else if (inf.complex) {
        // UP is output-packed (its +1 level is absorbed free by the following GELU). DOWN stays REAL:
        // mlp.cu reads "down" with the plain linear, so the weight must be plain too — and its
        // output-pack +1 would land on the residual (= +22 bts/token). out-proj likewise stays real.
        put("up",        encode_weight_matrix_outputpack(inf, W_up.pad,   d_pad,     d_exp_pad, /*target_level=*/wl("up"),   stream));
        put("up_bias",   {encode_bias_vector  (inf, b_up_pad,  d_pad,     d_exp_pad, /*fill=*/true, stream)});
        inf.complex_weight_names.insert("up");
        put("down",      encode_weight_matrix(inf, W_down.pad, d_exp_pad, d_pad,     /*target_level=*/wl("down"), stream));
        put("down_bias", {encode_bias_vector  (inf, b_down_pad,d_exp_pad, d_pad,     /*fill=*/true, stream)});
    } else {
        put("up",        encode_weight_matrix(inf, W_up.pad,   d_pad,     d_exp_pad, /*target_level=*/wl("up"),   stream));
        put("up_bias",   {encode_bias_vector  (inf, b_up_pad,  d_pad,     d_exp_pad, /*fill=*/true, stream)});
        put("down",      encode_weight_matrix(inf, W_down.pad, d_exp_pad, d_pad,     /*target_level=*/wl("down"), stream));
        put("down_bias", {encode_bias_vector  (inf, b_down_pad,d_exp_pad, d_pad,     /*fill=*/true, stream)});
    }

    return out;
}

inline void prepare_gpt2_layer_weights(
    Inference& inf,
    const WeightStore& store,
    const GPT2WeightNames& names,
    int d_real, int d_exp_real,
    int d_pad,  int d_exp_pad,
    int num_heads,
    cudaStream_t stream = nullptr) {
    auto enc = encode_gpt2_layer_weights(inf, store, names, d_real, d_exp_real,
                                         d_pad, d_exp_pad, num_heads, stream);
    for (auto& kv : enc.raw_w) inf.raw_w[kv.first] = std::move(kv.second);
    for (auto& kv : enc.w)     inf.w[kv.first]     = std::move(kv.second);
}

inline void prepare_gpt2_layer_configs(Inference& inf,
                                       const config_loader::ParsedConfigs& parsed,
                                       int block_idx) {
    const std::string base = gpt2_block_base(block_idx);
    inf.norm_cfg["ln_1"]   = parsed.norm.at(base + ".ln_1");
    inf.norm_cfg["ln_2"]   = parsed.norm.at(base + ".ln_2");
    inf.sm_cfg  ["attn"]   = parsed.softmax.at(base + ".attn");
    inf.gelu_cfg["mlp.act"] = parsed.softgelu.at(base + ".mlp.act");
}

inline void prepare_gpt2_final_ln_config(Inference& inf,
                                         const config_loader::ParsedConfigs& parsed) {
    inf.norm_cfg["ln_f"] = parsed.norm.at(gpt2_final_ln_base());
}

inline EncodedGpt2Layer encode_gpt2_final_ln_weights(
    Inference& inf, const WeightStore& store,
    int d_real, int d_pad, const BootstrapPlan& plan = {},
    cudaStream_t stream = nullptr,
    double gamma_scale = 1.0) {
    EncodedGpt2Layer out;
    const std::string base = gpt2_final_ln_base();
    auto w = load_vector(store, base + ".weight", d_real, d_pad).first;
    auto b = load_vector(store, base + ".bias",   d_real, d_pad).first;
    for (auto& g : w) g *= gamma_scale;
    const int wlvl = static_cast<int>(inf.fhe->bootstrap_output_level());
    auto wl = [&](const char* key) { return static_cast<int>(plan.weight_level(key, static_cast<uint32_t>(wlvl))); };
    if (fold_ln_affine("ln_f")) {   // gamma -> lm_head cols; beta rides input as ln_f.shift
        auto sf = ln_input_shift(w, b, d_real);
        out.w["ln_f.shift"] = {encode_ln_affine_param(inf, sf, d_pad, d_real, wl("ln_f.shift"), /*mask_inactive=*/true, stream)};
    } else {
        out.w["ln_f.weight"] = {encode_ln_affine_param(inf, w, d_pad, d_real, wl("ln_f.weight"), /*mask_inactive=*/true, stream)};
        out.w["ln_f.bias"]   = {encode_ln_affine_param(inf, b, d_pad, d_real, wl("ln_f.bias"),   /*mask_inactive=*/true,  stream)};
    }
    return out;
}

inline void prepare_gpt2_final_ln_weights(Inference& inf, const WeightStore& store,
                                          int d_real, int d_pad,
                                          cudaStream_t stream = nullptr) {
    auto enc = encode_gpt2_final_ln_weights(inf, store, d_real, d_pad, /*plan=*/BootstrapPlan{}, stream);
    for (auto& kv : enc.w) inf.w[kv.first] = std::move(kv.second);
}

struct LMHeadWeights {
    std::vector<std::vector<double>> W_pad;   // (d_pad, vocab), gamma-folded when folding
    std::vector<double> bias;                 // (vocab) ln_f beta logit bias; empty if not folding
};

inline LMHeadWeights load_gpt2_lm_head_weight(
    Inference& inf, const WeightStore& store, int d_real, int d_pad, int vocab) {

    auto W_lm = load_linear_weight(store, gpt2_lm_head_name(),
                                   /*d_in_real=*/d_real, /*d_out_real=*/vocab,
                                   /*d_in_pad=*/d_pad,   /*d_out_pad=*/vocab);
    LMHeadWeights out;
    if (fold_ln_affine("ln_f")) {   // gamma -> lm_head cols; beta rides input as ln_f.shift (no logit bias)
        const std::string lnf = gpt2_final_ln_base();
        auto [g, g_pad] = load_vector(store, lnf + ".weight", d_real, d_pad);
        (void)g;
        const double descale = ln_gamma_descale(inf.norm_cfg.at("ln_f"));
        for (auto& v : g_pad) v *= descale;
        W_lm.pad = matrix::mat_scale_rows(W_lm.pad, g_pad);   // scale input rows by gamma
    }
    out.W_pad = std::move(W_lm.pad);
    return out;
}

inline std::vector<Ptx> encode_gpt2_lm_head_tile(
    Inference& inf, const std::vector<std::vector<double>>& W_lm_pad,
    int tile_idx, int W_tile, int vocab, int d_pad,
    const BootstrapPlan& plan = {}, const std::string& wkey = "",
    cudaStream_t stream = nullptr) {
    const int col0  = tile_idx * W_tile;
    const int wreal = std::min(W_tile, vocab - col0);

    auto W_k = matrix::pad_matrix(matrix::slice_cols(W_lm_pad, col0, wreal),
                                  d_pad, W_tile);

    const int wlvl = static_cast<int>(inf.fhe->bootstrap_output_level());
    const int lvl  = wkey.empty() ? wlvl
                   : static_cast<int>(plan.weight_level(wkey, static_cast<uint32_t>(wlvl)));
    return encode_weight_matrix(inf, W_k, d_pad, W_tile, lvl, stream);
}

inline std::vector<Ptx> encode_gpt2_lm_head_tile_complex(
    Inference& inf, const std::vector<std::vector<double>>& W_lm_pad,
    int pair_idx, int W_tile, int vocab, int d_pad, int K,
    const BootstrapPlan& plan = {}, const std::string& wkey = "",
    cudaStream_t stream = nullptr) {
    auto slice_tile = [&](int k) {
        const int col0  = k * W_tile;
        const int wreal = std::min(W_tile, vocab - col0);
        return matrix::pad_matrix(matrix::slice_cols(W_lm_pad, col0, wreal), d_pad, W_tile);
    };
    const int a = 2 * pair_idx, b = 2 * pair_idx + 1;
    auto W_a = slice_tile(a);
    std::vector<std::vector<double>> W_b =
        (b < K) ? slice_tile(b)
                : std::vector<std::vector<double>>(d_pad, std::vector<double>(W_tile, 0.0));

    const int wlvl = static_cast<int>(inf.fhe->bootstrap_output_level());
    const int lvl  = wkey.empty() ? wlvl
                   : static_cast<int>(plan.weight_level(wkey, static_cast<uint32_t>(wlvl)));
    return encode_weight_matrix_complex(inf, W_a, W_b, d_pad, W_tile, lvl, stream);
}

// ln_f beta folded into lm_head: per-tile logit bias. Empty when not folding.
inline std::vector<Ptx> encode_gpt2_lm_head_bias_tile(
    Inference& inf, const std::vector<double>& lm_bias,
    int tile_idx, int W_tile, int vocab, int d_pad,
    cudaStream_t stream = nullptr) {
    if (lm_bias.empty()) return {};
    const int col0  = tile_idx * W_tile;
    const int wreal = std::min(W_tile, vocab - col0);
    auto bslice = matrix::pad_vector(matrix::slice_vec(lm_bias, col0, wreal), W_tile);
    return {encode_bias_vector(inf, bslice, d_pad, W_tile, /*fill=*/true, stream)};
}

inline EncodedGpt2Layer encode_gpt2_lm_head_weights(
    Inference& inf, const WeightStore& store,
    int d_real, int d_pad, int vocab, int W_tile,
    const BootstrapPlan& plan = {}, cudaStream_t stream = nullptr,
    // Tail prefetch: the worker preps tiles for the DECODE arm while inf is still
    // in the prefill phase — complex_override forces the arm, and complex_keys_out
    // (when set) collects the complex tile names INSTEAD of mutating
    // inf.complex_weight_names (a shared set; worker-side insert would race the
    // main thread). The consumer inserts them before first use.
    const bool* complex_override = nullptr,
    std::vector<std::string>* complex_keys_out = nullptr) {
    EncodedGpt2Layer out;
    auto lm = load_gpt2_lm_head_weight(inf, store, d_real, d_pad, vocab);
    const int K = (vocab + W_tile - 1) / W_tile;

    const bool as_complex = complex_override ? *complex_override : inf.complex;
    if (as_complex) {
        // cachemir_complex: ceil(K/2) complex tiles, each carrying a tile PAIR.
        if (!lm.bias.empty())
            throw std::runtime_error("complex lm_head + folded ln_f bias unsupported "
                                     "(set GPT2_FOLD_LNF=0)");
        const int Kc = (K + 1) / 2;
        for (int m = 0; m < Kc; ++m) {
            const std::string key = gpt2_lm_head_tile_key(m);
            out.w[key] = encode_gpt2_lm_head_tile_complex(inf, lm.W_pad, m, W_tile, vocab,
                                                          d_pad, K, plan, key, stream);
            if (complex_keys_out) complex_keys_out->push_back(key);
            else                  inf.complex_weight_names.insert(key);
        }
        return out;
    }

    for (int k = 0; k < K; ++k) {
        const std::string key = gpt2_lm_head_tile_key(k);
        out.w[key] = encode_gpt2_lm_head_tile(inf, lm.W_pad, k, W_tile, vocab,
                                              d_pad, plan, key, stream);
        auto bt = encode_gpt2_lm_head_bias_tile(inf, lm.bias, k, W_tile, vocab,
                                                d_pad, stream);
        if (!bt.empty()) out.w[key + "_bias"] = std::move(bt);
    }
    return out;
}

}  // namespace weight_loader
