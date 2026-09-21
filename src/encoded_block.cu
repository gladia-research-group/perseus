#include "encoded_block.h"

#include <iostream>
#include <malloc.h>
#include <utility>

void load_block_to_device(Inference& inf, EncodedBlock& blk, cudaStream_t stream) {
    for (auto& kv : blk.w)
        for (auto& pt : kv.second) inf.load_plaintext(pt, stream);
}

void cpu_extract_block(Inference& inf, EncodedBlock& blk) {
    inf.begin_stage_block();
    for (auto& kv : blk.w)
        for (auto& pt : kv.second) inf.extract_plaintext(pt);
}

void stage_plaintexts(Inference& inf, std::vector<Ptx>& pts, int n_threads) {
    #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 4)
    for (int i = 0; i < static_cast<int>(pts.size()); ++i)
        if (pts[i]) inf.extract_plaintext(pts[i]);
}

void stage_block_weights(Inference& inf, EncodedBlock& blk, int n_threads) {
    for (auto& kv : blk.w)
        stage_plaintexts(inf, kv.second, n_threads);
    malloc_trim(0);
}

void evict_block_from_device(Inference& inf, EncodedBlock& blk) {
    for (auto& kv : blk.w)
        for (auto& pt : kv.second) inf.evict_plaintext(pt);
}

void load_weight_keys(Inference& inf, const std::vector<std::string>& keys,
                      cudaStream_t stream) {
    for (const auto& key : keys) {
        auto it = inf.w.find(key);
        if (it == inf.w.end()) continue;
        for (auto& pt : it->second) inf.load_plaintext(pt, stream);
    }
}

void evict_weight_keys(Inference& inf, const std::vector<std::string>& keys) {
    for (const auto& key : keys) {
        auto it = inf.w.find(key);
        if (it == inf.w.end()) continue;
        for (auto& pt : it->second) inf.evict_plaintext(pt);
    }
}

void cpu_extract_weight_keys(Inference& inf, const std::vector<std::string>& keys) {
    for (const auto& key : keys) {
        auto it = inf.w.find(key);
        if (it == inf.w.end()) continue;
        for (auto& pt : it->second) inf.extract_plaintext(pt);
    }
}

void install_block_state(Inference& inf, EncodedBlock&& state) {
    for (auto& kv : state.w)        inf.w[kv.first]        = std::move(kv.second);
    for (auto& kv : state.raw_w)    inf.raw_w[kv.first]    = std::move(kv.second);
    for (auto& kv : state.norm_cfg) inf.norm_cfg[kv.first] = kv.second;
    for (auto& kv : state.sm_cfg)   inf.sm_cfg[kv.first]   = kv.second;
    for (auto& kv : state.gelu_cfg) inf.gelu_cfg[kv.first] = kv.second;
    for (auto& sm : state.staged_masks)   // worker-side mask prefetch (prefill; empty on decode)
        inf.adopt_enc_cache(std::get<0>(sm), std::get<1>(sm), std::move(std::get<2>(sm)));
    state.staged_masks.clear();
    inf.fhe->install_plan_live(state.plan);   // no-op (clear) when plan.valid == false
}

void install_block_state_copy(Inference& inf, const EncodedBlock& state) {
    for (const auto& kv : state.w)        inf.w[kv.first]        = kv.second;
    for (const auto& kv : state.raw_w)    inf.raw_w[kv.first]    = kv.second;
    for (const auto& kv : state.norm_cfg) inf.norm_cfg[kv.first] = kv.second;
    for (const auto& kv : state.sm_cfg)   inf.sm_cfg[kv.first]   = kv.second;
    for (const auto& kv : state.gelu_cfg) inf.gelu_cfg[kv.first] = kv.second;
    inf.fhe->install_plan_live(state.plan);   // no-op (clear) when plan.valid == false

    inf.weight_store =
        const_cast<std::unordered_map<std::string, std::vector<Ptx>>*>(&state.w);
}

void log_block(int b, int n_blocks, const char* label) {
    std::cout << "Processing block " << b << " / " << n_blocks << label << std::endl;
}
