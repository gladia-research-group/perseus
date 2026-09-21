#include "model/gpt2.h"
#include "model/gpt2/internal.h"
#include "packing/cachemir/cachemir_rot_indices.h"
#include "packing/cachemir_filling/cachemir_filling_rot_indices.h"
#include "packing/diagonal/diagonal_rot_indices.h"
#include "weight_loader.h"
#include "math/matrix_ops.h"

#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>

std::vector<int32_t> compute_gpt2_rot_indices(
    const Packing& packing,
    int slots, int hidDim, int ffDim, int numHeads) {
    if (is_cachemir(packing))
        return cachemir::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads);
    if (is_cachemir_filling(packing))
        return cachemir_filling::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads);
    if (is_diagonal(packing))
        return diagonal::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads);
    throw std::runtime_error("compute_gpt2_rot_indices: unsupported packing");
}

std::vector<int32_t> gpt2_filling_only_rot_steps(int slots, int hidDim, int ffDim, int numHeads) {
    const auto cm = cachemir::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads);
    const auto fl = cachemir_filling::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads);
    auto norm = [slots](int i) { i %= slots; if (i < 0) i += slots; return i; };

    std::set<int> keep;
    for (int r : cm) keep.insert(norm(r));
    for (int i = 1; i <= slots; i *= 2) { keep.insert(norm(i)); keep.insert(norm(-i)); }
    for (int j = 1; j <= 256; j *= 2) { keep.insert(norm(1024 / j)); keep.insert(norm(-(1024 / j))); }
    keep.insert(norm(5)); keep.insert(norm(-5));

    std::set<int> seen;
    std::vector<int32_t> out;
    for (int r : fl) {
        const int n = norm(r);
        if (n == 0 || keep.count(n) || seen.count(n)) continue;
        seen.insert(n);
        out.push_back(r);
    }
    return out;
}

std::vector<int32_t> gpt2_decode_only_rot_steps(int slots, int hidDim, int ffDim, int numHeads) {
    const auto cm = cachemir::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads);
    const auto fl = cachemir_filling::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads);
    auto norm = [slots](int i) { i %= slots; if (i < 0) i += slots; return i; };

    std::set<int> keep;
    for (int r : fl) keep.insert(norm(r));
    for (int i = 1; i <= slots; i *= 2) { keep.insert(norm(i)); keep.insert(norm(-i)); }
    for (int j = 1; j <= 256; j *= 2) { keep.insert(norm(1024 / j)); keep.insert(norm(-(1024 / j))); }
    keep.insert(norm(5)); keep.insert(norm(-5));

    std::set<int> seen;
    std::vector<int32_t> out;
    for (int r : cm) {
        const int n = norm(r);
        if (n == 0 || keep.count(n) || seen.count(n)) continue;
        seen.insert(n);
        out.push_back(r);
    }
    return out;
}

Inference make_gpt2_inference(InferenceOptions opts) {
    const int slots = (opts.ckks.batch_size == 0)
                    ? (1 << (opts.ckks.logN - 1))
                    : static_cast<int>(opts.ckks.batch_size);

    Packing packing;
    packing.kind     = opts.packing_kind;
    packing.slots    = slots;
    packing.hidDim   = opts.hidDim;
    packing.realDim  = opts.dim;
    packing.numHeads = opts.numHeads;
    packing.t        = (opts.hidDim > 0) ? slots / opts.hidDim : 0;

    auto model_rots = compute_gpt2_rot_indices(packing, slots, opts.hidDim,
                                               opts.expDim, opts.numHeads);
    for (auto r : model_rots) opts.ckks.extra_rot_steps.push_back(r);

    for (PackingKind aux : opts.aux_packing_kinds) {
        if (aux == opts.packing_kind) continue;
        Packing ap = packing;
        ap.kind = aux;
        auto aux_rots = compute_gpt2_rot_indices(ap, slots, opts.hidDim,
                                                 opts.expDim, opts.numHeads);
        for (auto r : aux_rots) opts.ckks.extra_rot_steps.push_back(r);
    }

    return make_inference(opts);
}

Inference make_gpt2_inference(const config_loader::ModelConfig& model,
                              CKKSContextOptions ckks) {
    InferenceOptions opts;
    opts.ckks         = ckks;
    opts.dim          = model.n_embd;
    opts.hidDim       = matrix::next_pow2(model.n_embd);
    opts.expanded     = model.n_inner;
    opts.expDim       = matrix::next_pow2(model.n_inner);
    opts.numHeadsReal = model.n_head;
    opts.numHeads     = matrix::next_pow2(model.n_head);
    return make_gpt2_inference(opts);
}

InferenceMode parse_inference_mode(const std::string& s) {
    if (s == "sync")     return InferenceMode::Sync;
    if (s == "threaded") return InferenceMode::Threaded;
    if (s == "prefetch") return InferenceMode::Prefetch;
    throw std::runtime_error("parse_inference_mode: unknown mode '" + s +
                             "' (expected sync|threaded|prefetch)");
}
