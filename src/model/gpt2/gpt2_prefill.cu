#include "model/gpt2.h"
#include "model/gpt2/internal.h"
#include "attention.h"        // prepare_mha_masks / prepare_vcache
#include "encoded_block.h"    // run_blocks
#include "packing/cachemir_filling/cachemir_filling_attention.h"   // mask_values_for_tag

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <malloc.h>
#include <omp.h>
#include <cuda_runtime.h>

namespace {

const char* graph_dir_env() {
    const char* v = std::getenv("FHE_GRAPH_DIR");
    return (v && *v) ? v : nullptr;
}


struct MaskObs {
    std::mutex m;
    std::map<std::string, std::vector<uint32_t>> tag_levels;   // unscoped tag -> levels seen
};

// Main thread (release hook): record block b's scoped cf.sm.* keys ("<scope>cf.sm.…#L<lv>:d1").
void record_block_mask_obs(Inference& inf, int b, MaskObs& obs) {
    const std::string scope = block_scope(b);
    const std::string want  = scope + "cf.sm.";
    std::lock_guard<std::mutex> lk(obs.m);
    for (const auto& kv : inf.enc_cache) {
        const std::string& key = kv.first;
        if (key.compare(0, want.size(), want) != 0) continue;
        const size_t l_at = key.rfind("#L");
        if (l_at == std::string::npos) continue;
        const std::string tag = key.substr(scope.size(), l_at - scope.size());
        const uint32_t lv = static_cast<uint32_t>(
            std::strtoul(key.c_str() + l_at + 2, nullptr, 10));
        auto& lvls = obs.tag_levels[tag];
        if (std::find(lvls.begin(), lvls.end(), lv) == lvls.end()) lvls.push_back(lv);
    }
}

// Worker thread (loader): encode block b's scoped masks at the observed (tag, level)s.
void prefetch_block_masks(Inference& inf, EncodedBlock& blk, int b, MaskObs& obs) {
    std::map<std::string, std::vector<uint32_t>> snapshot;
    {
        std::lock_guard<std::mutex> lk(obs.m);
        snapshot = obs.tag_levels;
    }
    if (snapshot.empty()) return;
    const auto cfg_it = blk.sm_cfg.find("attn");
    if (cfg_it == blk.sm_cfg.end()) return;
    const std::string scope = block_scope(b);
    std::vector<double> v;
    for (const auto& kv : snapshot) {
        if (!cachemir_filling::mask_values_for_tag(inf, &cfg_it->second, kv.first, v))
            continue;
        for (uint32_t lv : kv.second)
            blk.staged_masks.emplace_back(
                scope + kv.first, lv,
                inf.cc()->MakeCKKSPackedPlaintext(v, /*noiseScaleDeg=*/1, lv));
    }
}

void reset_graph_runtime(Inference& inf) {
    inf.fhe->ct_vars.clear();
    inf.fhe->pt_vars.clear();
    inf.fhe->graph_ct_counter = 0;
    inf.fhe->graph_pt_counter = 0;
}

}  // namespace

namespace {
// Tail-prefetch stash (one prep per process; chunk-independent — weights only).
struct TailLmPrep {
    std::vector<EncodedBlock>  lm;
    std::vector<std::string>   complex_keys;
    std::atomic<bool>          ready{false};
    std::atomic<bool>          attempted{false};
};
TailLmPrep g_tail_lm;
}  // namespace

bool gpt2_tail_lm_take(std::vector<EncodedBlock>& out, std::vector<std::string>& complex_keys) {
    if (!g_tail_lm.ready.load()) return false;
    out          = std::move(g_tail_lm.lm);
    complex_keys = std::move(g_tail_lm.complex_keys);
    g_tail_lm.ready.store(false);
    return true;
}

namespace {

struct EarlyBlock0 {
    std::atomic<bool>          attempted{false};
    std::future<EncodedBlock>  fut;
    double                     enc_s = 0.0;   // worker wall, reported separately
};
EarlyBlock0 g_early_b0;

int prefill_encode_threads() {
    const char* et = std::getenv("FHE_PREFILL_ENCODE_THREADS");
    return et ? std::atoi(et) : 12;
}
}  // namespace

void gpt2_prefill_early_block0(Inference& inf,
                               const weight_loader::WeightStore& store,
                               const config_loader::ParsedConfigs& parsed_configs,
                               const BlockPlans& plans) {
    if (g_early_b0.attempted.exchange(true)) return;
    { std::vector<double> w(16, 0.0); (void)inf.cc()->MakeCKKSPackedPlaintext(w, 1, 0u); }
    g_early_b0.fut = std::async(std::launch::async,
                                [&inf, &store, &parsed_configs, plans]() {
        omp_set_num_threads(prefill_encode_threads());
        const auto t0 = std::chrono::steady_clock::now();
        BlockLoader raw_loader = make_block_loader(store, parsed_configs, plans);

        const int st = pt_stage_block_threads();
        if (st > 0)
            inf.pt_stage_hook = [](const std::string&, std::vector<Ptx>&) {};
        EncodedBlock r = raw_loader(inf, 0, nullptr);
        if (st > 0) inf.pt_stage_hook = nullptr;
        g_early_b0.enc_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[prefill-early] block_0 encoded on the worker in %.1fs "
                             "(overlapped with key/bts setup)\n", g_early_b0.enc_s);
        return r;
    });
}

void gpt2_prefill_early_block0_join(Inference& inf) {
    if (!g_early_b0.fut.valid()) return;
    EncodedBlock r = g_early_b0.fut.get();
    const int st = pt_stage_block_threads();
    if (st > 0) {
        inf.begin_stage_block();
        stage_block_weights(inf, r, st);
    }
    std::promise<EncodedBlock> p;
    p.set_value(std::move(r));
    g_early_b0.fut = p.get_future();
}

namespace {

struct ChunkWCache {
    int  mode = 0;                       // 0=off, 1=fill (chunk 0), 2=reuse (chunk 1+)
    std::vector<EncodedBlock> blocks;
};
ChunkWCache g_wcache;
}  // namespace

void gpt2_prefill_wcache_configure(Inference& inf, int n_blocks, int mode) {
    if (mode != 0 && graph_dir_env()) mode = 0;   // capture: keep the recorded flow simple
    g_wcache.mode = mode;
    if (mode == 1) {
        g_wcache.blocks.assign(static_cast<size_t>(n_blocks), EncodedBlock{});
        inf.cc()->SuppressStageReleaseCpu(true);   // keep host payloads for the re-stage
    } else if (mode == 0) {
        inf.cc()->SuppressStageReleaseCpu(false);
    }
}

void gpt2_prefill_wcache_end(Inference& inf) {
    inf.cc()->SuppressStageReleaseCpu(false);
    g_wcache.blocks.clear();   // drops the payload-holding pt refs -> host RAM returns
    g_wcache.mode = 0;
}

PackedCtx gpt2_prefill(Inference& inf, PackedCtx x,
                       const weight_loader::WeightStore& store,
                       const config_loader::ParsedConfigs& parsed_configs,
                       const BlockPlans& plans,
                       int n_blocks, PrefillMode mode) {
    const bool chunk = (mode == PrefillMode::Chunk);
    const bool planned = plans.any_valid();
    BlockLoader raw_loader = make_block_loader(store, parsed_configs, plans);
    auto enc_s = std::make_shared<double>(0.0);   // [prefill-timing] weight-encode wall (load_block_state)

    const int kEncodeThreads = prefill_encode_threads();
    auto mask_obs = std::make_shared<MaskObs>();
    BlockLoader loader = [raw_loader, enc_s, mask_obs, n_blocks, &store, kEncodeThreads,
                          &plans](Inference& i, int b, cudaStream_t s) {

        if (g_wcache.mode == 2 && b < static_cast<int>(g_wcache.blocks.size())
            && !g_wcache.blocks[b].w.empty()) {
            EncodedBlock r = g_wcache.blocks[b];   // shallow: shared pts
            r.plan = plans.at(b);                  // THIS chunk's placements
            const int st2 = pt_stage_block_threads();
            if (st2 > 0) {
                i.begin_stage_block();
                stage_block_weights(i, r, st2);
            }
            r.staged_masks.clear();
            prefetch_block_masks(i, r, b, *mask_obs);
            return r;
        }

        if (b == 0 && g_early_b0.fut.valid()) {
            EncodedBlock r = g_early_b0.fut.get();
            if (g_wcache.mode == 1 && b < static_cast<int>(g_wcache.blocks.size()))
                g_wcache.blocks[b] = r;   // shallow copy into the chunk cache
            prefetch_block_masks(i, r, b, *mask_obs);
            return r;
        }
        omp_set_num_threads(kEncodeThreads);
        const auto t0 = std::chrono::steady_clock::now();
        const int st = pt_stage_block_threads();
        if (st > 0) {
            i.begin_stage_block();
            i.pt_stage_hook = [&i, st](const std::string&, std::vector<Ptx>& pts) {
                stage_plaintexts(i, pts, st);
            };
        }
        EncodedBlock r = raw_loader(i, b, s);
        if (st > 0) {
            i.pt_stage_hook = nullptr;
            stage_block_weights(i, r, st);
        }
        if (g_wcache.mode == 1 && b < static_cast<int>(g_wcache.blocks.size()))
            g_wcache.blocks[b] = r;   // shallow copy into the chunk cache (pts shared)
        prefetch_block_masks(i, r, b, *mask_obs);
        if (b >= n_blocks - 3
            && !i.fhe->complex_payload
            && !g_tail_lm.attempted.exchange(true)) {
            const auto tp0 = std::chrono::steady_clock::now();
            const bool dec_complex = i.fhe->complex_payload;
            const int vocab = static_cast<int>(
                store.meta(weight_loader::gpt2_lm_head_name()).shape[0]);
            g_tail_lm.lm.emplace_back();
            auto enc = weight_loader::encode_gpt2_lm_head_weights(
                i, store, i.size.getRealHidDim(), i.size.hidDim, vocab,
                i.slots, BootstrapPlan{}, nullptr, &dec_complex,
                &g_tail_lm.complex_keys);
            g_tail_lm.lm.front().w    = std::move(enc.w);
            g_tail_lm.lm.front().plan = BootstrapPlan{};
            g_tail_lm.ready.store(true);
            std::fprintf(stderr, "[tail-prefetch] lm_head (%s) encoded on the worker "
                         "in %.1fs (overlapped)\n", dec_complex ? "complex" : "real",
                         std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - tp0).count());
        }
        *enc_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return r;
    };

    if (!chunk) gpt2_reset_kv_cache(inf, n_blocks);

    auto cmp_s = std::make_shared<double>(0.0);   // [prefill-timing] block-compute wall (main thread)
    auto body = [planned, cmp_s](Inference& i, PackedCtx& h, int b) {
        size_t mem_free = 0, mem_total = 0;
        {

            WithStep _w(i, "body_prologue");
            cudaMemGetInfo(&mem_free, &mem_total);
        }
        std::fprintf(stderr, "[prefill-mem] block %d enter: free=%.2fGB / %.2fGB\n",
                     b, mem_free / 1e9, mem_total / 1e9);
        std::fflush(stderr);
        const auto t0 = std::chrono::steady_clock::now();
        i.output.capture_b = b;
        if (planned || graph_dir_env()) reset_graph_runtime(i);

        i.fhe->profile.ensure_initialized();
        if (i.fhe->profile.mode == StepProfiler::Mode::Wall) {
            WithStep _w(i, "block_entry_flush");
            cudaDeviceSynchronize();
        }
        gpt2_block_step(i, h, b, [](Inference& j) {
            WithStep _w(j, "kv_reload"); reload_block_kv(j);   // same prologue as decode
        });
        *cmp_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    auto release = [mask_obs](Inference& i, int b) {
        record_block_mask_obs(i, b, *mask_obs);
        gpt2_block_release(i, b);
        i.evict_enc_cache_device_scoped(block_scope(b));
        reclaim_host_async(i);
    };
    const auto t_all = std::chrono::steady_clock::now();
    x = run_blocks(inf, std::move(x), n_blocks, inf.mode, loader, body, release);
    finish_host_reclaim();
    const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all).count();
    const double hidden  = std::max(0.0, *enc_s + *cmp_s - total_s);
    const double exposed = std::max(0.0, total_s - *cmp_s);
    const int chunk_toks = inf.n_tok + inf.n_tok_imag;   // token-pair: B half rides the Im lanes
    std::fprintf(stderr,
        "[prefill-timing] blocks_total=%.1fs encode_wall=%.1fs compute_wall=%.1fs"
        " hidden=%.1fs exposed=%.1fs (s/tok=%.1f over %d tok)\n",
        total_s, *enc_s, *cmp_s, hidden, exposed,
        chunk_toks > 0 ? total_s / chunk_toks : 0.0, chunk_toks);
    std::fflush(stderr);

    if (chunk) return x;   // no final LN — caller LNs after the last chunk

    if (planned || graph_dir_env()) reset_graph_runtime(inf);
    begin_subgraph_capture(inf, n_blocks);
    EncodedBlock lnf = load_final_ln_state(inf, store, parsed_configs, plans.at(n_blocks), nullptr);
    PackedCtx h = apply_final_ln(inf, x, lnf);
    end_subgraph_capture(inf, n_blocks);
    return h;
}
