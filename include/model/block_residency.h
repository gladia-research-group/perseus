#pragma once

#include "inference.h"
#include "encoded_block.h"
#include "weight_loader.h"
#include "config_loader.h"

#include <cuda_runtime.h>
#include <string>

std::string block_scope(int b);

EncodedBlock load_block_state(Inference& inf,
                              const weight_loader::WeightStore& store,
                              const config_loader::ParsedConfigs& parsed_configs,
                              const BootstrapPlan& plan,
                              int block_idx, cudaStream_t stream);
EncodedBlock load_final_ln_state(Inference& inf,
                                 const weight_loader::WeightStore& store,
                                 const config_loader::ParsedConfigs& parsed_configs,
                                 const BootstrapPlan& plan,
                                 cudaStream_t stream);
BlockLoader make_block_loader(const weight_loader::WeightStore& store,
                              const config_loader::ParsedConfigs& parsed_configs,
                              const BlockPlans& plans);

// Evict the current packing's block weights + per-block configs. Call before
// switching inf.packing so stale wrong-layout weights can't leak across packings.
void evict_block_weights(Inference& inf);

// Host allocator reclaim (malloc_trim): async off the compute thread in Threaded
// mode (bounded to one outstanding trim), synchronous otherwise.
void reclaim_host_async(Inference& inf);
void finish_host_reclaim();
