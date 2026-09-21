#pragma once

#include "inference.h"

#include <string>
#include <vector>

namespace cachemir_filling {

struct DeltaEntry { int delta; int block; bool current; bool alive; int Lg = -1; };

struct DeltaView {
    int P           = 0;
    int n_cur       = 0;
    int block_shift = 0;
    bool bd         = false;
    std::vector<DeltaEntry> entries;
};

std::vector<PackedCtx> qkt_delta(Inference& inf, const PackedCtx& query);

std::vector<PackedCtx> qkt_delta_groups(Inference& inf, const PackedCtx& query,
                                        const std::vector<PackedCtx>& kgroups, int K);

DeltaView bd_delta_view(Inference& inf, int K);

std::vector<PackedCtx> attention_softmax_thor_delta(
    Inference& inf, std::vector<PackedCtx> scores, const std::string& cfg_name,
    const std::string& stage_prefix = "cf.stg.");

std::vector<PackedCtx> attention_softmax_thor_delta_core(
    Inference& inf, std::vector<PackedCtx> scores, const std::string& cfg_name,
    const DeltaView& view, const std::string& stage_prefix);

PackedCtx mha_attn_token_pair_delta(Inference& inf, PackedCtx& q_cplx);

bool delta_block_enabled();

}  // namespace cachemir_filling
