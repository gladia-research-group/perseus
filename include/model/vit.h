#pragma once

#include "model/gpt2.h"

Inference make_vit_inference(InferenceOptions opts);

std::vector<PackedCtx> vit_forward(Inference& inf,
                                   std::vector<PackedCtx> chunks,
                                   const std::vector<int>& n_toks,
                                   const weight_loader::WeightStore& store,
                                   const config_loader::ParsedConfigs& parsed,
                                   int n_blocks,
                                   const std::vector<int>& n_toks_imag = {});
