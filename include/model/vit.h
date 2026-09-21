#pragma once

#include "model/gpt2.h"

// Encoder-specific Inference: CachemirFilling packing, bidirectional schedule,
// and the EXPLICIT ViT rotation-key set (filling encoder blocks + the cachemir
// CLS-tail reductions) instead of the GPT-2 union.
Inference make_vit_inference(InferenceOptions opts);

// n_toks_imag: token-pair arm only (one packed chunk carrying nA Re + nB Im
// tokens) — pass the per-chunk B-half counts. Empty = the real arm.
std::vector<PackedCtx> vit_forward(Inference& inf,
                                   std::vector<PackedCtx> chunks,
                                   const std::vector<int>& n_toks,
                                   const weight_loader::WeightStore& store,
                                   const config_loader::ParsedConfigs& parsed,
                                   int n_blocks,
                                   const std::vector<int>& n_toks_imag = {});
