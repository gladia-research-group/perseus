#include "model/gpt2.h"
#include "model/gpt2/internal.h"
#include "model/block_residency.h"
#include "attention.h"
#include "encoded_block.h"
#include "weight_loader.h"
#include "packing/cachemir/cachemir_attention_utils.h"
#include "packing/cachemir/cachemir_norm_utils.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <malloc.h>
#include <cuda_runtime.h>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>


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

static void prefetch_reload_kv(Inference& inf, int b, int n_blocks) {
    if (b < 0 || b >= n_blocks) return;
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
    if (b < 0) return;
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
    // Async once-only prewarm of the pinned KV arena (KV_ARENA_GB) and of the two pinned
    // stage-arena halves (FHE_STAGE_ARENA_GB each), so token 0 does not first-touch them on
    // its critical path. Both are mutex-raced against the lazy path (the loser frees).
    inf.cc()->PrewarmKvArena();
    fideslib::PrewarmStageArenas();
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

// ONE walker, three sinks (prime / erase / stage), two site classes (shared / scoped).
//
// Cache keys are scoped asymmetrically: `sm.score` and `ln.center.*` go through
// inf.scoped() and are per-block; `sm.active`, `qkt.gmask` and `v.lane` are UNSCOPED and
// shared across blocks. The worker-side stage sink encodes masks off the main thread and
// hands them over at the next token's join. The worker and the main thread must build the
// same vectors from the same code, so the classes are a filter over ONE walk rather than a
// second copy of the site table.
enum class MaskOp  { Prime, Erase, Stage };
enum class MaskSet { All, SharedOnly, ScopedOnly };

struct MaskWalkSink {
    MaskOp  op  = MaskOp::Prime;
    MaskSet set = MaskSet::All;
    // Stage only: the produced plaintexts, and the keys already produced this pass. Shared sites
    // are walked once per block so their per-block level lists union naturally; `seen` stops
    // that union from becoming duplicate encodes.
    std::vector<std::tuple<std::string, uint32_t, Ptx>>* out = nullptr;
    std::unordered_set<std::string>*                     seen = nullptr;
};

// `prefix` is the block scope ("gpt2.h.<b>.") passed EXPLICITLY rather than read from the mutable
// inf.block_prefix member — that is what lets the ScopedOnly walk run on a worker thread.
static void step_mask_walk_block(Inference& inf, EncodedBlock& blk, int step,
                                 const MaskWalkSink& sink, const std::string& prefix) {
    using namespace cachemir;
    const BootstrapPlan& plan = blk.plan;
    if (!plan.valid || plan.mask_levels.empty()) return;
    const int rD = inf.size.getRealHidDim();
    const int t  = inf.slots / inf.size.hidDim;
    const int d_head_real = inf.size.getRealDHead();
    const int kc = step + 1;        // #cached keys at this step
    const int rr = step % t;        // V-push right_rot
    // `shared` is stated per call site rather than inferred from the tag: it is the property the
    // whole split rests on, so it should be readable next to the site, and it must be kept true
    // if a tag builder ever gains or loses inf.scoped().
    auto at = [&](const std::string& site, bool shared, const std::string& tag, auto&& vec_fn) {
        if (sink.set == MaskSet::SharedOnly && !shared) return;
        if (sink.set == MaskSet::ScopedOnly &&  shared) return;
        const std::vector<uint32_t>* lvls = plan.mask_level_list(site);
        if (!lvls) return;
        // vec_fn() builds a full slot vector; it is evaluated lazily so a cache hit never
        // pays for it.
        for (uint32_t lv : *lvls) {
            switch (sink.op) {
            case MaskOp::Prime:
                inf.prime_enc_cache_lazy(tag, lv, vec_fn);
                break;
            case MaskOp::Erase:
                inf.erase_enc_cache(tag, lv);
                break;
            case MaskOp::Stage: {
                // Deliberately does NOT consult inf.enc_cache: this runs on the residency worker
                // while the main thread is still inserting into that map (apply_final_ln's
                // ln.center.ln_f, CutMax), and reading an unordered_map under concurrent insert
                // is a data race. Encoding a key that already exists is merely wasted work --
                // adopt_enc_cache emplaces, so the loser is dropped.
                const std::string key = Inference::enc_cache_key(tag, lv, 1);
                if (!sink.seen->insert(key).second) break;
                sink.out->emplace_back(tag, lv, inf.encode_tagged(vec_fn(), lv));
                break;
            }
            }
        }
    };
    const SoftmaxConfig& sm = blk.sm_cfg.at("attn");
    const double mean = (sm.clip_hi + sm.clip_lo) / 2.0;
    at("sm.score",  false, score_mask_tag(prefix, kc), [&]{ return score_mask_vec(inf, sm.clip_lo, mean, kc); });
    at("sm.active", true,  active_mask_tag(kc),     [&]{ return active_mask_vec(inf, kc); });
    if (inf.complex) {
        const int Nb = kc / (2 * t);             // completed complex buckets = floor(kc/64)
        const int G  = (kc + t - 1) / t;         // total groups
        const int M  = G - 2 * Nb;               // pending real tail groups
        for (int gc = 0; gc < Nb; ++gc) {
            const int ge = 2 * gc, go = 2 * gc + 1;
            at("qkt.gmask", true, qkt_group_mask_tag(kc, ge),
               [&]{ return qkt_group_mask_vec(inf, std::min(t, kc - ge * t), ge); });
            if (go * t < kc)
                at("qkt.gmask", true, qkt_complex_odd_mask_tag(kc, go),
                   [&]{ return qkt_complex_odd_mask_vec(inf, std::min(t, kc - go * t), go); });
        }
        for (int j = 0; j < M; ++j) {
            const int g = 2 * Nb + j;
            at("qkt.gmask", true, qkt_group_mask_tag(kc, g),
               [&]{ return qkt_group_mask_vec(inf, std::min(t, kc - g * t), g); });
        }
    } else {
        const int num_groups = (kc + t - 1) / t;
        for (int g = 0; g < num_groups; ++g) {
            const int num_tok = std::min(t, kc - g * t);
            at("qkt.gmask", true, qkt_group_mask_tag(kc, g), [&]{ return qkt_group_mask_vec(inf, num_tok, g); });
        }
    }
    if (inf.complex) {
        const int d_head = inf.size.hidDim / inf.size.numHeads;
        const int c      = step / t;
        for (int p = 0; p < d_head_real / 2; ++p) {
            const int i_re = ((c - 2 * p)     % d_head + d_head) % d_head;
            const int i_im = ((c - 2 * p - 1) % d_head + d_head) % d_head;
            at("v.lane", true, vpair_mask_complex_tag(i_re, i_im, rr),
               [&]{ return vpair_mask_complex_vec(inf, i_re, i_im, rr); });
        }
    } else {
        for (int i = 0; i < d_head_real; ++i) {
            if (inf.fhe->complex_payload)
                at("v.lane", true, vlane_mask_tag(i, rr), [&]{ return complex_vlane_mask_vec(inf, i, rr, -0.5); });
            else
                at("v.lane", true, vlane_mask_tag(i, rr), [&]{ return vlane_mask_vec(inf, i, rr); });
        }
    }
    for (const char* which : {"ln_1", "ln_2"}) {
        const NormConfig& nc = blk.norm_cfg.at(which);
        if (nc.center_scale_sq.empty()) continue;   // fixed variant primed in generate_decode_masks
        const int sz = static_cast<int>(nc.center_scale_sq.size());
        const int cpos = std::max(0, std::min(step, sz - 1));
        const double ch = 0.5 * std::sqrt(nc.center_scale_sq[cpos]);
        at(std::string("ln.center.") + which, false, ln_center_mask_tag(prefix, which, cpos),
           [&]{ return inf.active_token_mask_vec(rD, t, ch); });
    }
}

static void step_mask_walk(Inference& inf, std::vector<EncodedBlock>& blocks, int n_blocks,
                           int step, const MaskWalkSink& sink) {
    if (!is_cachemir(inf.packing) || blocks.empty() || step < 0) return;
    const std::string saved = inf.block_prefix;
    for (int b = 0; b < n_blocks && b < static_cast<int>(blocks.size()); ++b) {
        inf.block_prefix = block_scope(b);
        step_mask_walk_block(inf, blocks[b], step, sink, inf.block_prefix);
    }
    inf.block_prefix = saved;
}

void gpt2_prime_step_masks(Inference& inf, std::vector<EncodedBlock>& blocks, int n_blocks, int step) {
    // Walks ALL sites: this is the safety net. Anything the worker-side staging failed to
    // produce is simply encoded here, so the worker path can be purely speculative.
    if (inf.strict_masks_armed)
        step_mask_walk(inf, blocks, n_blocks, step, MaskWalkSink{MaskOp::Prime, MaskSet::All});
}

void gpt2_encode_shared_masks(Inference& inf, std::vector<EncodedBlock>& blocks, int n_blocks,
                              int step,
                              std::vector<std::tuple<std::string, uint32_t, Ptx>>& out) {
    // RUNS ON THE RESIDENCY WORKER. Three properties make that safe:
    //   * it never reads or writes inf.enc_cache -- it only appends to `out`;
    //   * it never touches inf.block_prefix, because every SHARED tag builder is a free function
    //     of run-constant state. That is the whole reason the split is shared-vs-scoped and not
    //     something more natural like per-block;
    //   * MakeCKKSPackedPlaintext reaches no CUDA (auto_load_plaintexts is false and nothing in
    //     this repo sets it) and tag_plaintext is mutex-guarded.
    if (!is_cachemir(inf.packing) || blocks.empty() || step < 0) return;
    if (!inf.strict_masks_armed) return;
    std::unordered_set<std::string> seen;
    MaskWalkSink sink{MaskOp::Stage, MaskSet::SharedOnly, &out, &seen};
    // NOT step_mask_walk: that assigns inf.block_prefix. Shared sites do not read it, so the
    // worker walks the blocks directly and leaves the main thread's prefix alone.
    for (int b = 0; b < n_blocks && b < static_cast<int>(blocks.size()); ++b)
        step_mask_walk_block(inf, blocks[b], step, sink, block_scope(b));
}

void gpt2_encode_scoped_masks(Inference& inf, std::vector<EncodedBlock>& blocks, int n_blocks,
                              int step,
                              std::vector<std::tuple<std::string, uint32_t, Ptx>>& out) {
    // RUNS ON THE SCOPED-MASK WORKER (not the ring worker, whose window is full). Safe because
    // `step_mask_walk_block` takes the block prefix as a parameter, so this walk never reads or
    // writes inf.block_prefix; everything else a scoped site reads (blk.plan, blk.sm_cfg,
    // blk.norm_cfg) is const for the duration of the window.
    if (!is_cachemir(inf.packing) || blocks.empty() || step < 0) return;
    if (!inf.strict_masks_armed) return;
    std::unordered_set<std::string> seen;
    MaskWalkSink sink{MaskOp::Stage, MaskSet::ScopedOnly, &out, &seen};
    for (int b = 0; b < n_blocks && b < static_cast<int>(blocks.size()); ++b)
        step_mask_walk_block(inf, blocks[b], step, sink, block_scope(b));
}

void gpt2_evict_step_masks(Inference& inf, std::vector<EncodedBlock>& blocks, int n_blocks, int step) {
    // ALL sites, always: the evict must undo whatever was primed, and the shared keys were
    // adopted from the worker rather than primed here. An evict narrowed to match the prime
    // would leak one step's shared masks per token.
    if (inf.strict_masks_armed)
        step_mask_walk(inf, blocks, n_blocks, step, MaskWalkSink{MaskOp::Erase, MaskSet::All});
}

void gpt2_kv_prefetch_first(Inference& inf, int n_blocks) {
    prefetch_reload_kv(inf, 0, n_blocks);
    cudaStreamSynchronize(kv_stream());
}

void gpt2_kv_block_prologue(Inference& inf, int b, int n_blocks) {
    prefetch_reload_kv(inf, b + 1, n_blocks);
}

// Post-loop (token end): drain + evict the last block's deferred offload.
void gpt2_kv_finalize_last(Inference& inf, int n_blocks) {
    cudaStreamSynchronize(kv_stream());   // drain block n-1's offload D2H
    finalize_offload_kv(inf, n_blocks - 1);
}

void gpt2_block_release(Inference& inf, int b) {
    { WithStep _w(inf, "block_sync"); cudaDeviceSynchronize(); }
    { WithStep _w(inf, "weight_evict"); evict_block_weights(inf); }
    WithStep _w(inf, "kv_offload");
    finalize_offload_kv(inf, b - 1);
    enqueue_offload_kv(inf, b);
}

