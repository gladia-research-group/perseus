#include "model/gpt2.h"
#include "model/gpt2/internal.h"
#include "model/block_residency.h"
#include "attention.h"
#include "encoded_block.h"
#include "weight_loader.h"
#include "packing/cachemir/cachemir_attention_utils.h"
#include "packing/cachemir/cachemir_norm_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <future>
#include <malloc.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>


static bool kv_resident() {
    static const bool v = [] {
        const char* e = std::getenv("FHE_KV_RESIDENT");
        return e && *e && std::atoi(e) != 0;
    }();
    return v;
}


static cudaStream_t kv_stream() {
    static cudaStream_t s = [] {
        cudaStream_t st = nullptr;
        cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
        return st;
    }();
    return s;
}

static std::string kv_pos_key(Inference& inf, const char* key, size_t lane) {
    return inf.scoped(key) + ":" + std::to_string(lane);
}

static bool kv_overlap() {
    static const bool v = [] { const char* e = std::getenv("FHE_KV_OVERLAP"); return !(e && *e && std::atoi(e) == 0); }();
    return v;
}

static void prefetch_reload_kv(Inference& inf, int b, int n_blocks) {
    if (kv_resident() || b < 0 || b >= n_blocks) return;
    const std::string saved = inf.block_prefix;
    inf.block_prefix = block_scope(b);
    const cudaStream_t s = kv_stream();
    for (const char* key : {"k", "v"}) {
        auto it = inf.cache.find(inf.scoped(key));
        if (it == inf.cache.end()) continue;
        for (size_t lane = 0; lane < it->second.size(); ++lane) {
            auto& pc = it->second[lane];
            if (pc.ct) inf.cc()->KvLoadStaged(pc.ct, kv_pos_key(inf, key, lane), s);
        }
    }
    inf.block_prefix = saved;
}

// Enqueue async D2H offload of block b's KV onto kv_stream (no sync, no evict). Scoped to block b.
static void enqueue_offload_kv(Inference& inf, int b) {
    if (kv_resident()) return;
    const std::string saved = inf.block_prefix;
    inf.block_prefix = block_scope(b);
    const cudaStream_t s = kv_stream();
    for (const char* key : {"k", "v"}) {
        auto it = inf.cache.find(inf.scoped(key));
        if (it == inf.cache.end()) continue;
        for (size_t lane = 0; lane < it->second.size(); ++lane) {
            auto& pc = it->second[lane];
            if (pc.ct) inf.cc()->KvStoreStaged(pc.ct, kv_pos_key(inf, key, lane), s);
        }
    }
    inf.block_prefix = saved;
}

static void finalize_offload_kv(Inference& inf, int b) {
    if (kv_resident() || b < 0) return;
    const std::string saved = inf.block_prefix;
    inf.block_prefix = block_scope(b);
    for (const char* key : {"k", "v"}) {
        auto it = inf.cache.find(inf.scoped(key));
        if (it == inf.cache.end()) continue;
        for (auto& pc : it->second)
            if (pc.ct) inf.cc()->KvEvict(pc.ct);
    }
    inf.block_prefix = saved;
}

void offload_block_kv(Inference& inf) {
    if (kv_resident()) return;   // keep KV on device — no device->host swap
    const cudaStream_t s = kv_stream();
    // 1) enqueue the async D2H of every cache ct into its pinned slot (no per-ct sync)
    for (const char* key : {"k", "v"}) {
        auto it = inf.cache.find(inf.scoped(key));
        if (it == inf.cache.end()) continue;
        for (size_t lane = 0; lane < it->second.size(); ++lane) {
            auto& pc = it->second[lane];
            if (pc.ct) inf.cc()->KvStoreStaged(pc.ct, kv_pos_key(inf, key, lane), s);
        }
    }
    // 2) one sync for the whole block (pinned slots now valid), then free the device copies
    cudaStreamSynchronize(s);
    for (const char* key : {"k", "v"}) {
        auto it = inf.cache.find(inf.scoped(key));
        if (it == inf.cache.end()) continue;
        for (auto& pc : it->second)
            if (pc.ct) inf.cc()->KvEvict(pc.ct);
    }
}

void reload_block_kv(Inference& inf) {
    if (kv_resident()) return;   // KV never left the device
    const cudaStream_t s = kv_stream();
    for (const char* key : {"k", "v"}) {
        auto it = inf.cache.find(inf.scoped(key));
        if (it == inf.cache.end()) continue;
        for (size_t lane = 0; lane < it->second.size(); ++lane) {
            auto& pc = it->second[lane];
            if (pc.ct) inf.cc()->KvLoadStaged(pc.ct, kv_pos_key(inf, key, lane), s);
        }
    }
    cudaStreamSynchronize(s);   // H2D complete before attention reads the cache
}

void gpt2_reset_kv_cache(Inference& inf, int n_blocks) {
    // if (!kv_resident()) inf.cc()->PrewarmKvArena();
    const std::string saved = inf.block_prefix;
    for (int b = 0; b < n_blocks; ++b) {
        inf.block_prefix = block_scope(b);
        prepare_mha_masks(inf);
        prepare_vcache(inf);
    }
    inf.block_prefix = saved;
}

// Pre-encode every cachemir per-block selector plaintext
void gpt2_generate_decode_masks(Inference& inf, std::vector<EncodedBlock>& blocks,
                                int n_blocks, int T) {
    using namespace cachemir;
    if (!is_cachemir(inf.packing)) return;            // decode masks are cachemir-only
    if (blocks.empty()) {                              // non-cached path: nothing to read
        std::fprintf(stderr, "[mask_gen] no cached blocks; planned masks stay online\n");
        return;
    }

    const int rD = inf.size.getRealHidDim();
    const int t  = inf.slots / inf.size.hidDim;
    const std::string saved = inf.block_prefix;

    uint64_t primed = 0, sites_missing = 0;

    auto at_levels = [&](const BootstrapPlan& plan, const std::string& site,
                         const std::string& tag, auto&& vec_fn) {
        const std::vector<uint32_t>* lvls = plan.mask_level_list(site);
        if (!lvls) { ++sites_missing; return; }
        const std::vector<double> vec = vec_fn();
        for (uint32_t lv : *lvls) { inf.prime_enc_cache(tag, vec, lv); ++primed; }
    };
    // complex variant — for the K/V-pack v.lane mask under CKKS_COMPLEX (same cache, complex pt)
    auto at_levels_cplx = [&](const BootstrapPlan& plan, const std::string& site,
                              const std::string& tag, auto&& vec_fn) {
        const std::vector<uint32_t>* lvls = plan.mask_level_list(site);
        if (!lvls) { ++sites_missing; return; }
        const std::vector<std::complex<double>> vec = vec_fn();
        for (uint32_t lv : *lvls) { inf.prime_enc_cache(tag, vec, lv); ++primed; }
    };

    for (int b = 0; b < n_blocks && b < static_cast<int>(blocks.size()); ++b) {
        const EncodedBlock&  blk  = blocks[b];
        const BootstrapPlan& plan = blk.plan;
        if (!plan.valid || plan.mask_levels.empty()) continue;   // eager-ish block: skip
        inf.block_prefix = block_scope(b);

        at_levels(plan, "tok0",        "tok0",        [&]{ return real_head_tok0_mask(inf); });
        at_levels(plan, "tok0.h",      "tok0.h",      [&]{ return real_head_half_mask(inf); });
        at_levels(plan, "kpush.tok0h", "kpush.tok0h", [&]{
            std::vector<double> m = real_head_half_mask(inf);
            if (inf.fhe->complex_payload && !inf.complex) for (double& x : m) x *= 0.5;
            return m;
        });
        at_levels(plan, "hrs.pos0",    "hrs.pos0",    [&]{ return hrs_pos0_vec(inf); });
        at_levels(plan, "ln.scalemask",
                  "ln.scalemask.d" + std::to_string(rD) + ".t" + std::to_string(t),
                  [&]{ return inf.stride_mask_vec(rD, t, -1.0 / static_cast<double>(rD)); });
        at_levels(plan, "gelu.half",
                  "gelu.half.s" + std::to_string(0.25),
                  [&]{ return inf.active_expanded_mask_vec(0.25, 0.0); });
        at_levels(plan, "gelu.half",                      // thor_composite exit masks at s0.5
                  "gelu.half.s" + std::to_string(0.5),
                  [&]{ return inf.active_expanded_mask_vec(0.5, 0.0); });

        for (const char* which : {"ln_1", "ln_2"}) {
            const NormConfig& nc = blk.norm_cfg.at(which);
            if (!nc.center_scale_sq.empty()) continue;
            const double ch = 0.5 * nc.center_scale;
            at_levels(plan, std::string("ln.center.") + which, ln_center_mask_tag(inf, which, -1),
                      [&]{ return inf.active_token_mask_vec(rD, t, ch); });
        }
    }

    inf.block_prefix = saved;
    inf.strict_masks_armed = (primed > 0);

    const size_t evicted = inf.evict_enc_cache_device();
    std::printf("[mask_gen] primed %llu mask plaintexts over %d blocks (T=%d, enc_cache=%zu, "
                "device_evicted=%zu, sites_skipped=%llu, strict=%d)\n",
                static_cast<unsigned long long>(primed), n_blocks, T, inf.enc_cache.size(),
                evicted, static_cast<unsigned long long>(sites_missing),
                static_cast<int>(inf.strict_masks_armed));
    std::fflush(stdout);
}

static void step_mask_walk_block(Inference& inf, EncodedBlock& blk, int step, bool prime) {
    using namespace cachemir;
    const BootstrapPlan& plan = blk.plan;
    if (!plan.valid || plan.mask_levels.empty()) return;
    const int rD = inf.size.getRealHidDim();
    const int t  = inf.slots / inf.size.hidDim;
    const int d_head_real = inf.size.getRealDHead();
    const int kc = step + 1;        // #cached keys at this step
    const int rr = step % t;        // V-push right_rot
    auto at = [&](const std::string& site, const std::string& tag, auto&& vec_fn) {
        const std::vector<uint32_t>* lvls = plan.mask_level_list(site);
        if (!lvls) return;
        for (uint32_t lv : *lvls) {
            if (prime) inf.prime_enc_cache(tag, vec_fn(), lv);
            else       inf.erase_enc_cache(tag, lv);
        }
    };
    const SoftmaxConfig& sm = blk.sm_cfg.at("attn");
    const double mean = (sm.clip_hi + sm.clip_lo) / 2.0;
    at("sm.score",  score_mask_tag(inf, kc), [&]{ return score_mask_vec(inf, sm.clip_lo, mean, kc); });
    at("sm.active", active_mask_tag(kc),     [&]{ return active_mask_vec(inf, kc); });
    if (inf.complex) {
        const int Nb = kc / (2 * t);             // completed complex buckets = floor(kc/64)
        const int G  = (kc + t - 1) / t;         // total groups
        const int M  = G - 2 * Nb;               // pending real tail groups
        for (int gc = 0; gc < Nb; ++gc) {
            const int ge = 2 * gc, go = 2 * gc + 1;
            at("qkt.gmask", qkt_group_mask_tag(kc, ge),
               [&]{ return qkt_group_mask_vec(inf, std::min(t, kc - ge * t), ge); });
            if (go * t < kc)
                at("qkt.gmask", qkt_complex_odd_mask_tag(kc, go),
                   [&]{ return qkt_complex_odd_mask_vec(inf, std::min(t, kc - go * t), go); });
        }
        for (int j = 0; j < M; ++j) {
            const int g = 2 * Nb + j;
            at("qkt.gmask", qkt_group_mask_tag(kc, g),
               [&]{ return qkt_group_mask_vec(inf, std::min(t, kc - g * t), g); });
        }
    } else {
        const int num_groups = (kc + t - 1) / t;
        for (int g = 0; g < num_groups; ++g) {
            const int num_tok = std::min(t, kc - g * t);
            at("qkt.gmask", qkt_group_mask_tag(kc, g), [&]{ return qkt_group_mask_vec(inf, num_tok, g); });
        }
    }
    if (inf.complex) {
        const int d_head = inf.size.hidDim / inf.size.numHeads;
        const int c      = step / t;
        for (int p = 0; p < d_head_real / 2; ++p) {
            const int i_re = ((c - 2 * p)     % d_head + d_head) % d_head;
            const int i_im = ((c - 2 * p - 1) % d_head + d_head) % d_head;
            at("v.lane", vpair_mask_complex_tag(i_re, i_im, rr),
               [&]{ return vpair_mask_complex_vec(inf, i_re, i_im, rr); });
        }
    } else {
        for (int i = 0; i < d_head_real; ++i) {
            if (inf.fhe->complex_payload)
                at("v.lane", vlane_mask_tag(i, rr), [&]{ return complex_vlane_mask_vec(inf, i, rr, -0.5); });
            else
                at("v.lane", vlane_mask_tag(i, rr), [&]{ return vlane_mask_vec(inf, i, rr); });
        }
    }
    for (const char* which : {"ln_1", "ln_2"}) {
        const NormConfig& nc = blk.norm_cfg.at(which);
        if (nc.center_scale_sq.empty()) continue;   // fixed variant primed in generate_decode_masks
        const int sz = static_cast<int>(nc.center_scale_sq.size());
        const int cpos = std::max(0, std::min(step, sz - 1));
        const double ch = 0.5 * std::sqrt(nc.center_scale_sq[cpos]);
        at(std::string("ln.center.") + which, ln_center_mask_tag(inf, which, cpos),
           [&]{ return inf.active_token_mask_vec(rD, t, ch); });
    }
}

static void step_mask_walk(Inference& inf, std::vector<EncodedBlock>& blocks, int n_blocks,
                           int step, bool prime) {
    if (!is_cachemir(inf.packing) || blocks.empty() || step < 0) return;
    const std::string saved = inf.block_prefix;
    for (int b = 0; b < n_blocks && b < static_cast<int>(blocks.size()); ++b) {
        inf.block_prefix = block_scope(b);
        step_mask_walk_block(inf, blocks[b], step, prime);
    }
    inf.block_prefix = saved;
}

void gpt2_prime_step_masks(Inference& inf, std::vector<EncodedBlock>& blocks, int n_blocks, int step) {
    if (inf.strict_masks_armed) step_mask_walk(inf, blocks, n_blocks, step, /*prime=*/true);
}

void gpt2_prime_block_masks(Inference& inf, EncodedBlock& blk, int step) {
    if (!inf.strict_masks_armed || !is_cachemir(inf.packing) || step < 0) return;
    step_mask_walk_block(inf, blk, step, /*prime=*/true);
}
void gpt2_evict_step_masks(Inference& inf, std::vector<EncodedBlock>& blocks, int n_blocks, int step) {
    if (inf.strict_masks_armed) step_mask_walk(inf, blocks, n_blocks, step, /*prime=*/false);
}

void gpt2_kv_prefetch_first(Inference& inf, int n_blocks) {
    if (!kv_overlap()) return;
    prefetch_reload_kv(inf, 0, n_blocks);
    if (!kv_resident()) cudaStreamSynchronize(kv_stream());
}

void gpt2_kv_block_prologue(Inference& inf, int b, int n_blocks) {
    if (!kv_overlap()) { reload_block_kv(inf); return; }
    prefetch_reload_kv(inf, b + 1, n_blocks);
}

// Post-loop (token end): drain + evict the last block's deferred offload (overlap mode only).
void gpt2_kv_finalize_last(Inference& inf, int n_blocks) {
    if (!kv_overlap() || kv_resident()) return;
    cudaStreamSynchronize(kv_stream());   // drain block n-1's offload D2H
    finalize_offload_kv(inf, n_blocks - 1);
}

void gpt2_block_release(Inference& inf, int b) {
    { WithStep _w(inf, "block_sync"); cudaDeviceSynchronize(); }
    evict_block_weights(inf);
    WithStep _w(inf, "kv_offload");
    if (kv_overlap()) {
        finalize_offload_kv(inf, b - 1);
        enqueue_offload_kv(inf, b);
    } else {
        offload_block_kv(inf);
    }
}

