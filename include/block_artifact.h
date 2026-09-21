#pragma once

#include "model/block_residency.h"

#include <string>

namespace block_artifact {

EncodedBlock encode_block_state_coeff(Inference& inf,
                                      const weight_loader::WeightStore& store,
                                      const config_loader::ParsedConfigs& parsed,
                                      const BootstrapPlan& plan, int block_idx);

void save_block_state(Inference& inf, const EncodedBlock& blk, const std::string& path);

EncodedBlock read_block_artifact(Inference& inf, const std::string& path);

std::string block_state_diff(const EncodedBlock& a, const EncodedBlock& b);

}  // namespace block_artifact
