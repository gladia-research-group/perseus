#pragma once

#include "inference.h"
#include "residency_pipeline.h"
#include "packing/packed_ctx.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

void load_weight_keys(Inference& inf, const std::vector<std::string>& keys, cudaStream_t stream);
void evict_weight_keys(Inference& inf, const std::vector<std::string>& keys);

inline Overlap overlap_of(InferenceMode mode) {
    switch (mode) {
        case InferenceMode::Sync:     return Overlap::Sync;
        case InferenceMode::Prefetch: return Overlap::Stream;
        case InferenceMode::Threaded: return Overlap::Threaded;
    }
    return Overlap::Sync;
}

inline Overlap device_overlap(InferenceMode mode) {
    const Overlap ov = overlap_of(mode);
    return (ov == Overlap::Threaded) ? Overlap::Stream : ov;
}

inline bool streams_within_block(WeightGranularity g) {
    return g == WeightGranularity::Sublayer || g == WeightGranularity::Linear;
}

struct Op {
    std::vector<std::string>                      weights; 
    std::function<void(Inference&, PackedCtx&)>   fwd;     
    const char*                                   label = "";
    std::function<void(Inference&, cudaStream_t)>  acquire;
    std::function<void(Inference&)>                install;
    std::function<void(Inference&)>                release;
    std::function<void(Inference&)>                prefetch_cpu;
    bool                                           prefetch_next = true;
    int stage_owner = -1;
};

inline PackedCtx run_ops(Inference& inf, PackedCtx x, std::vector<Op> ops,
                         bool stream_weights = false, Overlap mode = Overlap::Sync) {
    auto h = std::make_shared<PackedCtx>(std::move(x));
    std::vector<ResidencyStage> stages;
    stages.reserve(ops.size());
    for (auto& op : ops) {
        ResidencyStage st;
        st.label         = op.label;
        st.acquire       = std::move(op.acquire);
        st.install       = std::move(op.install);
        st.release       = std::move(op.release);
        st.prefetch_cpu  = std::move(op.prefetch_cpu);
        st.prefetch_next = op.prefetch_next;
        st.stage_owner = op.stage_owner;
        st.compute = [h, fwd = std::move(op.fwd), label = op.label](Inference& i) {
            WithStep _w(i, label);
            fwd(i, *h);
        };

        if (!st.acquire && stream_weights && !op.weights.empty()) {
            auto keys = op.weights;
            st.acquire = [keys](Inference& i, cudaStream_t s) { load_weight_keys(i, keys, s); };
            st.release = [keys](Inference& i) { evict_weight_keys(i, keys); };
            st.prefetch_next = false;
        }
        stages.push_back(std::move(st));
    }
    run_residency_pipeline(inf, std::move(stages), mode);
    return std::move(*h);
}
