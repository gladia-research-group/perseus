#pragma once

#include "inference.h"

#include <cuda_runtime.h>
#include <functional>
#include <future>
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

    // prefetch_cpu(inf) — host-only pre-extraction for stage i+2, run on the residency WORKER
    // while the main thread is inside INS(i)/ACQ(i+1)/CMP(i).
    //
    // This hook is the one exception to the `acquire` rule: it reaches the FHE context
    // through extract_plaintext / begin_stage_block only, which are thread-safe for it.
    // It must NOT encode, mutate inf.w, or touch CUDA — the main thread owns the streams.
    std::function<void(Inference&)>               prefetch_cpu;
    // Staging-arena half this stage owns while in flight (-1 = ungated).
    int                                           stage_owner = -1;
};

// How acquire(i+1) overlaps compute(i):
//   Sync     — acquire(i); install(i); compute(i); release(i).            (no overlap)
//   Stream   — prefetch acquire(i+1) on a side stream during compute(i).
//   Threaded — prefetch acquire(i+1) on a worker thread during compute(i).
enum class Overlap { Sync, Stream, Threaded };

// Hand a host-side job to the SAME persistent worker the pipeline uses. The decode loop
// uses it to prime the next token's first stages while the current token's argmax is still
// on the GPU (the circular block ring in gpt2_decode.cu).
std::future<void> residency_submit(std::function<void()> job);

// Hand a job to a SEPARATE persistent worker. Used solely for the scoped-mask staging: the
// ring worker's queue is FIFO with a same-iteration join, so a long job there would
// head-of-line block the per-block extractions. Do not add other callers.
std::future<void> mask_submit(std::function<void()> job);

void run_residency_pipeline(Inference& inf, std::vector<ResidencyStage> stages,
                            Overlap mode);
