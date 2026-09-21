#include "model/gpt2.h"
#include "model/gpt2/internal.h"
#include "encoded_block.h"    // run_blocks / run_cached_blocks
#include "packing/cachemir/cachemir_attention_utils.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <string>
#include <exception>
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


PackedCtx gpt2_decode_forward(Inference& inf, PackedCtx x, int t, int n_blocks,
                                     std::vector<EncodedBlock>& blocks,
                                     const BlockLoader& loader, EncodedBlock& lnf,
                                     bool planned) {

    WithStep _tok(inf, "tok" + std::to_string(t));

    auto body = [t, planned, n_blocks, &blocks](Inference& i, PackedCtx& h, int b) {
        WithStep _blk(i, "blk" + std::to_string(b));
        i.output.capture_t = t;
        i.output.capture_b = b;
        if (planned || graph_dir_env()) {
            i.fhe->ct_vars.clear();
            i.fhe->pt_vars.clear();
            i.fhe->graph_ct_counter = 0;
            i.fhe->graph_pt_counter = 0;
            i.fhe->current_runtime_node_id = 0;
        }
        gpt2_block_step(i, h, b, [b, n_blocks](Inference& j) {
            WithStep _w(j, "kv_reload"); gpt2_kv_block_prologue(j, b, n_blocks);
        });
        gpt2_prime_block_masks(i, blocks[b], t + 1);
    };


    const bool _prof = inf.fhe->profile.on();
    std::chrono::steady_clock::time_point _blocks_t0;
    if (_prof) { cudaDeviceSynchronize(); _blocks_t0 = std::chrono::steady_clock::now(); }

    inf.strict_masks = inf.strict_masks_armed;
    
    gpt2_evict_step_masks(inf, blocks, n_blocks, t - 1);
    gpt2_prime_step_masks(inf, blocks, n_blocks, t);
    gpt2_kv_prefetch_first(inf, n_blocks);   // seed reload(0) so block 0 attention has its KV (overlap mode)

    x = inf.cache_weights
        ? run_cached_blocks(inf, std::move(x), blocks, body, gpt2_block_release)
        : run_blocks(inf, std::move(x), n_blocks, inf.mode, loader, body, gpt2_block_release);

    gpt2_kv_finalize_last(inf, n_blocks);    // drain + evict the last block's deferred offload (overlap mode)
    inf.strict_masks = false;

    if (planned || graph_dir_env()) {
        inf.fhe->ct_vars.clear();
        inf.fhe->pt_vars.clear();
        inf.fhe->graph_ct_counter = 0;
        inf.fhe->graph_pt_counter = 0;
        inf.fhe->current_runtime_node_id = 0;
    }
    inf.output.capture_t = t;

    begin_subgraph_capture(inf, n_blocks);
    PackedCtx h = apply_final_ln(inf, x, lnf);

    if (_prof) {
        cudaDeviceSynchronize();
        const double _blocks_dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - _blocks_t0).count();
        std::printf("[blkperf] tok%d blocks+lnf=%.3f s  avg_per_block=%.3f s (/%d)\n",
                    t, _blocks_dt, n_blocks > 0 ? _blocks_dt / n_blocks : _blocks_dt, n_blocks);
    }
    {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        static uint64_t p_bts = 0, p_rlv = 0, p_miss = 0;
        static size_t   p_enc = 0;
        const uint64_t bts  = inf.fhe->total_bootstraps;
        const uint64_t rlv  = inf.fhe->weight_relevel_count;
        const uint64_t miss = inf.enc_cache_miss;
        const size_t   enc  = inf.enc_cache.size();
        std::printf("[tokstat] tok%d free=%.2fGB bts=%llu(+%llu) relevels=%llu(+%llu) "
                    "enc_cache=%zu(+%zu) miss=%llu(+%llu) lnf=%d/%d\n",
                    t, (double)free_b / 1073741824.0,
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
