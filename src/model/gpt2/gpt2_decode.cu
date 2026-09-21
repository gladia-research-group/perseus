#include "model/gpt2.h"
#include "model/gpt2/internal.h"
#include "encoded_block.h"    // run_blocks / run_cached_blocks
#include "packing/cachemir/cachemir_attention_utils.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <exception>
#include <future>
#include <vector>
#include <filesystem>
#include <cuda_runtime.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace {
    const char* graph_dir_env() {
        const char* v = std::getenv("FHE_GRAPH_DIR");
        return (v && *v) ? v : nullptr;
    }
}


namespace {

// The residency pipeline is a ~2-deep ring (REL(i) evicts block i while ACQ(i+1) uploads and
// the worker extracts i+2). The ring is kept turning ACROSS tokens: during the argmax/CutMax
// tail (GPU work needing no block weights) the worker extracts the next token's first two
// blocks, so the next token's EXTRACT(0)/EXTRACT(1) early-out instead of refilling from cold.
//
// The globals below are heap-allocated and INTENTIONALLY NEVER DESTROYED: they hold futures
// and Ptx (shared_ptr<PlaintextImpl>, which transitively references the CryptoContext). A
// plain global would be destroyed by __run_exit_handlers in a race with the CUDA driver's own
// exit handlers, and the loser dereferences a freed context. The process is exiting, so the
// leak is free. References, so every use site reads as a plain object.
std::future<void>& g_circular = *new std::future<void>();   // next token's fill, in flight during argmax

// Shared (unscoped) masks for the NEXT step, encoded by the residency worker in the same tail
// window the ring fill uses. WRITTEN ONLY by the worker while g_circular is in flight; READ
// ONLY by the main thread after g_circular.get(). The future is the happens-before edge.
std::vector<std::tuple<std::string, uint32_t, Ptx>>& g_staged_shared =
    *new std::vector<std::tuple<std::string, uint32_t, Ptx>>();

// Block-scoped masks for the next step, staged on a SECOND worker (mask_submit) so they do not
// sit behind the ring job in its FIFO. Same ownership contract as g_staged_shared, on g_scoped.
std::vector<std::tuple<std::string, uint32_t, Ptx>>& g_staged_scoped =
    *new std::vector<std::tuple<std::string, uint32_t, Ptx>>();
std::future<void> g_scoped;   // next token's scoped masks, in flight during this token's argmax

}  // namespace

PackedCtx gpt2_decode_forward(Inference& inf, PackedCtx x, int t, int n_blocks,
                                     std::vector<EncodedBlock>& blocks,
                                     const BlockLoader& loader, EncodedBlock& lnf,
                                     bool planned) {

    WithStep _tok(inf, "tok" + std::to_string(t));

    // Seed the process-wide constant-1 ciphertext from the input embedding, the freshest
    // ciphertext in the run (its level decides how many refreshes every Newton/Goldschmidt
    // consumer pays). First call wins. Plan-safe: planned mode resets graph_ct_counter at the
    // start of each block body, so ops emitted before block 0 cannot shift in-block var names.
    inf.fhe->ensure_const_one(x.ct, inf.slots);

    auto body = [t, planned, n_blocks, &blocks](Inference& i, PackedCtx& h, int b) {
        WithStep _blk(i, "blk" + std::to_string(b));
        i.output.capture_t = t;
        i.output.capture_b = b;
        if (planned || graph_dir_env()) {
            i.fhe->ct_vars.clear();
            i.fhe->pt_vars.clear();
            i.fhe->graph_ct_counter = 0;
            i.fhe->graph_pt_counter = 0;
        }
        gpt2_block_step(i, h, b, [b, n_blocks](Inference& j) {
            WithStep _w(j, "kv_reload"); gpt2_kv_block_prologue(j, b, n_blocks);
        });
    };

    // Join last token's circular prefetch BEFORE any main-thread access to blocks[], so the
    // worker is never inside GetRawPlainText on a Ptx the main thread is also extracting.
    if (g_circular.valid()) {
        WithStep _w(inf, "circular_join");
        g_circular.get();
    }
    // Adopt whatever the worker managed to encode. Strictly after the join, on the main thread,
    // so this is the only reader. adopt_enc_cache emplaces: a key the main thread already has
    // wins and the worker's copy is dropped here, which is why the worker can be speculative and
    // skip the (racy) cache check on its side.
    if (!g_staged_shared.empty()) {
        WithStep _w(inf, "mask_adopt");
        for (auto& sm : g_staged_shared)
            inf.adopt_enc_cache(std::get<0>(sm), std::get<1>(sm), std::move(std::get<2>(sm)));
        g_staged_shared.clear();
    }
    // Same join-then-adopt shape for the scoped masks, on its own future because it ran on
    // the other worker. If that worker overran, this join pays the remainder; the safety-net
    // prime below encodes anything still missing.
    if (g_scoped.valid()) { WithStep _w(inf, "scoped_join"); g_scoped.get(); }
    if (!g_staged_scoped.empty()) {
        WithStep _w(inf, "scoped_adopt");
        for (auto& sm : g_staged_scoped)
            inf.adopt_enc_cache(std::get<0>(sm), std::get<1>(sm), std::move(std::get<2>(sm)));
        g_staged_scoped.clear();
    }

    inf.strict_masks = inf.strict_masks_armed;

    // The two kv calls BLOCK on cudaStreamSynchronize(kv_stream()).
    { WithStep _w(inf, "mask_evict"); gpt2_evict_step_masks(inf, blocks, n_blocks, t - 1); }
    // Token 0 has no ring job to inherit from: run the same staged walk here, synchronously,
    // and adopt — identical keys and encodes, so the safety-net prime below then hits.
    if (t == 0 && inf.strict_masks_armed) {
        WithStep _w(inf, "mask_prime_tok0");
        std::vector<std::tuple<std::string, uint32_t, Ptx>> staged;
        gpt2_encode_shared_masks(inf, blocks, n_blocks, t, staged);
        for (auto& sm : staged)
            inf.adopt_enc_cache(std::get<0>(sm), std::get<1>(sm), std::move(std::get<2>(sm)));
    }
    { WithStep _w(inf, "mask_prime"); gpt2_prime_step_masks(inf, blocks, n_blocks, t); }
    { WithStep _w(inf, "kv_prefetch_first"); gpt2_kv_prefetch_first(inf, n_blocks); }   // seed reload(0) for block 0 attention

    x = inf.cache_weights
        ? run_cached_blocks(inf, std::move(x), blocks, body, gpt2_block_release)
        : run_blocks(inf, std::move(x), n_blocks, inf.mode, loader, body, gpt2_block_release);

    { WithStep _w(inf, "kv_finalize_last"); gpt2_kv_finalize_last(inf, n_blocks); }  // drain + evict last block's deferred offload
    inf.strict_masks = false;

    // Everything after this point (final LN, lm_head, then CutMax back in the driver) is GPU
    // work that needs no block weights — the window the ring fill hides in. ONE fused job on
    // the single ring worker: shared masks FIRST (if the window is short we lose only the tail
    // of the extraction, which the pipeline's own prefetch then redoes), then the ring fill.
    const bool want_ring = n_blocks > 1;
    g_circular = residency_submit([&inf, &blocks, n_blocks, t, want_ring] {
        gpt2_encode_shared_masks(inf, blocks, n_blocks, t + 1, g_staged_shared);
        if (want_ring) {
            cpu_extract_block(inf, blocks[0]);
            cpu_extract_block(inf, blocks[1]);
        }
    });
    // The SCOPED masks: same tail window, on the separate mask worker (the ring queue is FIFO
    // and a long job there would delay the ring's own join). Deadline is token t+1 block b,
    // so even a partial result is a win and the safety-net prime picks up whatever is missing.
    g_scoped = mask_submit([&inf, &blocks, n_blocks, t] {
        gpt2_encode_scoped_masks(inf, blocks, n_blocks, t + 1, g_staged_scoped);
    });
    // Everything below can throw (apply_final_ln is FHE work; a strict planned placement throws
    // live there), and the normal join for these jobs is at the START OF THE NEXT TOKEN, which
    // never runs if this one unwinds. Join on the exception path only; the happy path must not
    // wait here, since hiding these jobs under the argmax tail is the point of the ring.
    struct RingJoinOnUnwind {
        ~RingJoinOnUnwind() {
            if (std::uncaught_exceptions() == 0) return;
            if (g_circular.valid()) { try { g_circular.get(); } catch (...) {} }
            if (g_scoped.valid())   { try { g_scoped.get();   } catch (...) {} }
        }
    } _ring_join_on_unwind;

    if (planned || graph_dir_env()) {
        inf.fhe->ct_vars.clear();
        inf.fhe->pt_vars.clear();
        inf.fhe->graph_ct_counter = 0;
        inf.fhe->graph_pt_counter = 0;
    }
    inf.output.capture_t = t;

    begin_subgraph_capture(inf, n_blocks);
    PackedCtx h = apply_final_ln(inf, x, lnf);

    {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        // pool USED (live cudaMallocAsync allocations) vs RESERVED (what the pool holds from
        // the driver; it never shrinks on its own). `free` falling while `used` is flat is
        // pool bloat from stream-ordered reuse, not a leak; `used` climbing is a leak.
        size_t pool_used = 0, pool_res = 0;
        {
            int dev = 0; cudaGetDevice(&dev);
            cudaMemPool_t mp = nullptr;
            if (cudaDeviceGetDefaultMemPool(&mp, dev) == cudaSuccess && mp) {
                cuuint64_t u = 0, r = 0;
                cudaMemPoolGetAttribute(mp, cudaMemPoolAttrUsedMemCurrent, &u);
                cudaMemPoolGetAttribute(mp, cudaMemPoolAttrReservedMemCurrent, &r);
                pool_used = (size_t)u; pool_res = (size_t)r;
            }
            cudaGetLastError();
        }
        static uint64_t p_bts = 0, p_rlv = 0, p_miss = 0;
        static size_t   p_enc = 0;
        const uint64_t bts  = inf.fhe->total_bootstraps;
        const uint64_t rlv  = inf.fhe->weight_relevel_count;
        const uint64_t miss = inf.enc_cache_miss;
        const size_t   enc  = inf.enc_cache.size();
        std::printf("[tokstat] tok%d free=%.2fGB pool_used=%.2fGB pool_res=%.2fGB bts=%llu(+%llu) relevels=%llu(+%llu) "
                    "enc_cache=%zu(+%zu) miss=%llu(+%llu) lnf=%d/%d\n",
                    t, (double)free_b / 1073741824.0,
                    (double)pool_used / 1073741824.0, (double)pool_res / 1073741824.0,
                    (unsigned long long)bts, (unsigned long long)(bts - p_bts),
                    (unsigned long long)rlv, (unsigned long long)(rlv - p_rlv),
                    enc, enc - p_enc,
                    (unsigned long long)miss, (unsigned long long)(miss - p_miss),
                    inf.fhe->level_for_ct(h.ct),
                    h.ct ? (int)h.ct->GetNoiseScaleDeg() : -1);
        p_bts = bts; p_rlv = rlv; p_enc = enc; p_miss = miss;
    }
    std::fflush(stdout);
    return h;
}
