#pragma once

#include "inference.h"

#include <cuda_runtime.h>
#include <functional>
#include <string>
#include <vector>

// Single weight-residency runner. One stage = one unit of work (a transformer
// block, a sublayer, an lm_head tile). Each stage carries four hooks; the runner
// orchestrates *only* residency + overlap, the math lives in `compute`.
//
//   acquire(inf, stream) — make this stage's weights device-resident (encode if
//                          needed). May run off the main thread / on `stream`,
//                          so it MUST NOT touch inf.w or the FHE context.
//   install(inf)         — main-thread-only mutation of inf.w / context (e.g.
//                          install_block_state[_copy]). Empty when nothing to
//                          install (weights already resident).
//   compute(inf)         — the actual FHE math; data flows via captured locals.
//   release(inf)         — evict this stage's weights from device (empty when
//                          the compute body already evicts).
//
// Keeping acquire (worker-safe) and install (main-thread) separate is the core
// correctness invariant that makes Threaded overlap safe.
struct ResidencyStage {
    std::function<void(Inference&, cudaStream_t)> acquire;
    std::function<void(Inference&)>               install;
    std::function<void(Inference&)>               compute;
    std::function<void(Inference&)>               release;
    std::string                                   label;

    bool prefetch_next = true;

    std::function<void(Inference&)>               prefetch_cpu;
};

// How acquire(i+1) overlaps compute(i):
//   Sync     — acquire(i); install(i); compute(i); release(i).            (no overlap)
//   Stream   — prefetch acquire(i+1) on a side stream during compute(i).  (≈ old Prefetch/Cached)
//   Threaded — prefetch acquire(i+1) on a worker thread during compute(i). (≈ old Threaded)
enum class Overlap { Sync, Stream, Threaded };

void run_residency_pipeline(Inference& inf, std::vector<ResidencyStage> stages,
                            Overlap mode);
