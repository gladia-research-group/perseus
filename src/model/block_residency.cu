#include "model/block_residency.h"

#include <future>
#include <malloc.h>
#include <string>
#include <vector>

namespace {

// The canonical block-weight schema: what every arch's export adapter targets.
std::vector<std::string> ln_weight_keys() {
    return {"ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias",
            "ln_1.shift", "ln_2.shift"};   // shift present only when folded; evict_weights tolerates absent
}
std::vector<std::string> mha_weight_keys() {
    return {"q", "q_bias", "k", "k_bias", "v", "v_bias", "out", "out_bias"};
}
std::vector<std::string> mlp_weight_keys(const Inference& inf) {
    if (inf.tiled_mlp()) {
        // Tiled prefill MLP: per-tile up/down weights + a single down bias.
        const int n_tiles = inf.size.getRealFfDim() / inf.size.hidDim;
        std::vector<std::string> keys;
        for (int j = 0; j < n_tiles; ++j) {
            const std::string tj = ".t" + std::to_string(j);
            keys.push_back("up" + tj);
            keys.push_back("up" + tj + "_bias");
            keys.push_back("down" + tj);
        }
        keys.push_back("down_bias");
        return keys;
    }
    return {"up", "up_bias", "down", "down_bias"};
}

std::vector<std::string> block_weight_keys(const Inference& inf) {
    std::vector<std::string> keys = ln_weight_keys();
    for (const auto& group : {mha_weight_keys(), mlp_weight_keys(inf)})
        keys.insert(keys.end(), group.begin(), group.end());
    return keys;
}

std::future<void> g_host_reclaim;   // bounded: <=1 outstanding trim

}  // namespace

std::string block_scope(int b) {
    return weight_loader::gpt2_block_base(b) + ".";
}

void evict_block_weights(Inference& inf) {
    for (const auto& key : block_weight_keys(inf)) {
        inf.evict_weights(key);
        inf.raw_w.erase(key);   // raw host matrices are unused after encode — don't accumulate per block
    }
    inf.norm_cfg.erase("ln_1");
    inf.norm_cfg.erase("ln_2");
    inf.sm_cfg.erase("attn");
    inf.gelu_cfg.erase("mlp.act");
    inf.clear_bootstrap_plan();   // evict the live placement plan with the block's configs
}

EncodedBlock load_block_state(Inference& inf,
                              const weight_loader::WeightStore& store,
                              const config_loader::ParsedConfigs& parsed_configs,
                              const BootstrapPlan& plan,
                              int block_idx,
                              cudaStream_t stream) {
    EncodedBlock state;

    auto names = weight_loader::gpt2_layer_names(block_idx);
    const std::string cfg_base = weight_loader::gpt2_block_base(block_idx);

    auto enc = weight_loader::encode_gpt2_layer_weights(
        inf, store, names,
        /*d_real=*/inf.size.getRealHidDim(),
        /*d_exp_real=*/inf.size.getRealFfDim(),
        /*d_pad=*/inf.size.hidDim,
        /*d_exp_pad=*/inf.size.expDim,
        /*num_heads=*/inf.size.numHeads,
        stream,
        /*plan=*/plan,   // per-weight encode level from the bts plan
        weight_loader::ln_gamma_descale(parsed_configs.norm.at(cfg_base + ".ln_1")),
        weight_loader::ln_gamma_descale(parsed_configs.norm.at(cfg_base + ".ln_2")));
    state.w     = std::move(enc.w);
    state.raw_w = std::move(enc.raw_w);

    const std::string base = cfg_base;
    state.prefix = base + ".";
    state.norm_cfg["ln_1"]   = parsed_configs.norm.at(base + ".ln_1");
    state.norm_cfg["ln_2"]   = parsed_configs.norm.at(base + ".ln_2");
    state.sm_cfg  ["attn"]   = parsed_configs.softmax.at(base + ".attn");
    state.gelu_cfg["mlp.act"] = parsed_configs.softgelu.at(base + ".mlp.act");
    state.plan = plan;

    return state;
}

EncodedBlock load_final_ln_state(Inference& inf,
                                 const weight_loader::WeightStore& store,
                                 const config_loader::ParsedConfigs& parsed_configs,
                                 const BootstrapPlan& plan,
                                 cudaStream_t stream) {
    EncodedBlock state;
    const auto& lnf_cfg = parsed_configs.norm.at(weight_loader::gpt2_final_ln_base());
    auto enc = weight_loader::encode_gpt2_final_ln_weights(
        inf, store, inf.size.getRealHidDim(), inf.size.hidDim, plan, stream,
        weight_loader::ln_gamma_descale(lnf_cfg));
    state.w = std::move(enc.w);
    state.norm_cfg["ln_f"] = lnf_cfg;
    state.plan = plan;
    return state;
}

BlockLoader make_block_loader(const weight_loader::WeightStore& store,
                              const config_loader::ParsedConfigs& parsed_configs,
                              const BlockPlans& plans) {
    return [&store, &parsed_configs, &plans](Inference& i, int b, cudaStream_t s) {
        return load_block_state(i, store, parsed_configs, plans.at(b), b, s);
    };
}

void reclaim_host_async(Inference& inf) {
    if (inf.mode == InferenceMode::Threaded) {
        if (g_host_reclaim.valid()) g_host_reclaim.get();
        g_host_reclaim = std::async(std::launch::async, [] { malloc_trim(0); });
    } else {
        WithStep _w(inf, "malloc_trim");
        malloc_trim(0);
    }
}
void finish_host_reclaim() { if (g_host_reclaim.valid()) g_host_reclaim.get(); }
