#include "model/vit.h"
#include "model/gpt2/internal.h"
#include "attention.h"
#include "encoded_block.h"
#include "model/layer_norm.h"
#include "model/mlp.h"
#include "nonlinear.h"
#include "packing/cachemir/cachemir_rot_indices.h"
#include "packing/cachemir_filling/cachemir_filling_rot_indices.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

// The encoder's working key set: filling blocks (diagonal linears + attention
// deltas/strides) ∪ the cachemir set (the CLS tail: extraction + ln_f + head).
std::vector<int32_t> vit_rot_indices(int slots, int hidDim, int ffDim, int numHeads) {
    std::set<int32_t> rots;
    for (int32_t r : cachemir_filling::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads))
        rots.insert(r);
    for (int32_t r : cachemir::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads))
        rots.insert(r);
    return {rots.begin(), rots.end()};
}

// Encoder release: the cache-free bi_attention leaves no K/V state behind — only
// weights and the block's scoped mask device copies need eviction.
void vit_block_release(Inference& inf, int b) {
    { WithStep _w(inf, "block_sync"); cudaDeviceSynchronize(); }
    evict_block_weights(inf);
    inf.evict_enc_cache_device_scoped(block_scope(b));
    reclaim_host_async(inf);
}

void encoder_block_body(Inference& inf, std::vector<PackedCtx>& xs,
                        const std::vector<int>& ns, const std::vector<int>& ns_im) {
    WithStep _w(inf, "encoder_block");
    const int d = inf.size.hidDim;
    // token-pair: ONE packed chunk carries nA (Re) + nB (Im) tokens; the ops read
    // both counts off inf, and bi_attention takes ns = {nA, nB}.
    const bool tp = inf.token_pair && !ns_im.empty() && ns_im[0] > 0;
    auto set_counts = [&](size_t c) {
        inf.n_tok      = ns[c];
        inf.n_tok_imag = tp ? ns_im[c] : 0;
    };

    const size_t C = xs.size();
    std::vector<PackedCtx> skips(C), qs(C), ks(C), vs(C);
    for (size_t c = 0; c < C; ++c) {
        set_counts(c);
        const int res_lvl = inf.fhe->level_for_ct(xs[c].ct)
                          + (xs[c].ct && xs[c].ct->GetNoiseScaleDeg() == 2 ? 1 : 0);
        inf.name_graph_ct_if_absent(
            xs[c], (c == 0 ? std::string("transformer_block.x")
                           : "transformer_block.x" + std::to_string(c)) +
                       "-lvl=" + std::to_string(res_lvl));
        skips[c] = xs[c];
        PackedCtx x;
        {
            WithStep _ws(inf, "ln_1");
            x = layer_norm(inf, xs[c], "ln_1");
        }
        WithStep _wq(inf, "qkv");
        inf.fhe->bootstrap_hint(x, inf.fhe->level_limit() - 1, true);
        inf.fhe->level_hint(x, inf.fhe->level_limit() - 1);
        auto qkv = linear_multi(inf, x, {"k", "v", "q"}, d, d, /*stream_pt=*/true);
        ks[c] = std::move(qkv[0]);
        vs[c] = std::move(qkv[1]);
        qs[c] = std::move(qkv[2]);
    }

    std::vector<PackedCtx> ats;
    {
        WithStep _wa(inf, "attn_core");
        if (tp && qs.size() != 1)
            throw std::runtime_error("[vit] token-pair supports exactly ONE packed chunk (got " +
                                     std::to_string(qs.size()) + ")");
        ats = bi_attention(inf, std::move(qs), std::move(ks), std::move(vs),
                           tp ? std::vector<int>{ns[0], ns_im[0]} : ns);
    }

    for (size_t c = 0; c < C; ++c) {
        set_counts(c);
        PackedCtx x = std::move(ats[c]);
        {
            WithStep _wo(inf, "out_proj");
            inf.fhe->bootstrap_hint(x, inf.fhe->level_limit() - 1, true);
            inf.fhe->level_hint(x, inf.fhe->level_limit() - 1);
            x = linear(inf, x, "out", d, d, /*stream_pt=*/true);
        }
        {
            WithStep _wr(inf, "attn_residual");
            x = inf.fhe->add(x, skips[c]);
            skips[c] = x;
        }
        {
            WithStep _wn(inf, "ln_2");
            x = layer_norm(inf, x, "ln_2");
        }
        x = mlp_block(inf, x);   // canonical MLP: tiled/plain + streaming dispatch
        {
            WithStep _wr(inf, "mlp_residual");
            xs[c] = inf.fhe->add(x, skips[c]);
            skips[c] = PackedCtx{};
        }
    }
}

}  // namespace

Inference make_vit_inference(InferenceOptions opts) {
    if (pt_stage_block_threads() > 0) fideslib::PrewarmStageArenas();
    const int slots = (opts.ckks.batch_size == 0)
                    ? (1 << (opts.ckks.logN - 1))
                    : static_cast<int>(opts.ckks.batch_size);
    opts.packing_kind = PackingKind::CachemirFilling;
    for (int32_t r : vit_rot_indices(slots, opts.hidDim, opts.expDim, opts.numHeads))
        opts.ckks.extra_rot_steps.push_back(r);
    Inference inf = make_inference(opts);
    inf.bidirectional = true;
    inf.use_cache = false;
    inf.token_pair = inf.fhe->complex_payload;
    return inf;
}

std::vector<PackedCtx> vit_forward(Inference& inf,
                                   std::vector<PackedCtx> chunks,
                                   const std::vector<int>& n_toks,
                                   const weight_loader::WeightStore& store,
                                   const config_loader::ParsedConfigs& parsed,
                                   int n_blocks,
                                   const std::vector<int>& n_toks_imag) {
    inf.weight_granularity = WeightGranularity::Plaintext;
    if (const char* g = std::getenv("VIT_WEIGHT_GRANULARITY"); g && *g) {
        const std::string gs(g);
        if      (gs == "linear")    inf.weight_granularity = WeightGranularity::Linear;
        else if (gs == "sublayer")  inf.weight_granularity = WeightGranularity::Sublayer;
        else if (gs == "block")     inf.weight_granularity = WeightGranularity::Block;
        else if (gs != "plaintext")
            throw std::runtime_error("VIT_WEIGHT_GRANULARITY: expected "
                                     "plaintext|linear|sublayer|block, got " + gs);
        std::fprintf(stderr, "[encvit] weight_granularity=%s\n", gs.c_str());
    }
    BlockPlans plans;   // loader captures by reference — must outlive run_blocks
    bool planned = false;
    if (const char* pd = std::getenv("FHE_BOOTSTRAP_PLACEMENTS_DIR"); pd && *pd) {
        plans = load_block_plans(pd, n_blocks);
        if (!plans.any_valid())
            throw std::runtime_error(std::string("vit_forward: FHE_BOOTSTRAP_PLACEMENTS_DIR set "
                                                 "but no valid block plans under ") + pd);
        std::fprintf(stderr, "[encvit] planned mode: %s\n", pd);
        planned = true;
    }
    BlockLoader loader = make_block_loader(store, parsed, plans);

    // TAIL PREFETCH (2026-07-22): the tail (ln_f + lm_head) was the last serial
    // term (~24s: encoded+loaded AFTER the loop with nothing to hide behind).
    // Piggyback its host-side encode onto the residency worker at the LAST
    // block's loader call — it runs during bodies 10-11, and the staging wrap
    // below is applied on top, so the tail plaintexts get staged/coeff'd like
    // block weights. FHE_TAIL_PREFETCH=0 restores the serial tail.
    // DEFAULT OFF — ViT tail prefetch does NOT pay (investigated to conclusion 2026-07-23).
    // Symptom: decoded logits come out ~ZERO (w_mape 1.000-1.004 == |0-ref|/|ref|, argmax
    // junk 512) because gpt2_lm_head skips its device load when blk->w is pre-populated and
    // the tiles are not device-resident at use time; a worker-side load does not survive the
    // block-release / enc-cache eviction between block 11 and the tail, and a tail-side load
    // did not upload either (tail_s stayed 13.5s). Four variants all failed identically:
    // staged (50091167), full/hook-disarmed (50092617), lm-only (50093090), worker-load
    // (50140014/15), sync-load (50140478), tail-load (50140677). It also ABORTS vit112 at
    // block 10 on the added residency pressure (50140016).
    // ECONOMICS: the ~10s "win" in those runs was FAKE — they were fast precisely because
    // they skipped the upload. Only the ~2-4s encode is safely hideable, so the honest
    // ceiling is ~2s on a 124s row (<2%), not the ~10-15s the handoff doc's §6 projected
    // (that projection rests on the same measurement error). Hiding the ~10s upload would
    // require keeping tail tiles device-resident across eviction — invasive, and vit112 has
    // no memory headroom for it. NOT worth risking the validated 124.0s baseline.
    // FHE_TAIL_PREFETCH=1 re-enables for anyone picking this up. GPT-2's tail prefetch is
    // UNAFFECTED and stays on: its presets set GPT2_LMHEAD_GRANULARITY, so lm_head_streams()
    // is true and the tiles load per-use (validated: prefill32 PASS, 131.5s).
    const bool tail_prefetch = [] {
        const char* e = std::getenv("FHE_TAIL_PREFETCH");
        return e && *e && std::atoi(e) == 1;
    }();
    struct TailPrep {
        EncodedBlock lnf;
        std::vector<EncodedBlock> lm;
        std::atomic<bool> ready{false};
    };
    auto tail_prep = std::make_shared<TailPrep>();
    const int vocab_pre = static_cast<int>(
        store.meta(weight_loader::gpt2_lm_head_name()).shape[0]);
    if (tail_prefetch) {
        BlockLoader pre_tail = std::move(loader);
        loader = [pre_tail, tail_prep, n_blocks, &store, &parsed, vocab_pre](
                     Inference& i, int b, cudaStream_t st) -> EncodedBlock {
            EncodedBlock blk = pre_tail(i, b, st);
            if (b != n_blocks - 1) return blk;
            const auto t0 = std::chrono::steady_clock::now();
            // Encode the tail with the staging hook DISARMED: full plaintexts on
            // the standard load path. Coeff/staged tail pts broke twice — 1-limb
            // unexpanded (50091167), then the released-payload second-extraction
            // guard in the tail's cpu prefetch (50091953). Full pts are ~1s more
            // upload, still fully overlapped here.
            auto saved_hook = i.pt_stage_hook;
            i.pt_stage_hook = nullptr;
            // lm-ONLY prep (the GPT-2-proven shape): prepping the lnf here breaks
            // the tail deterministically (w_mape 1.0, jobs 50091167/50092617 across
            // two staging treatments) — the lnf encodes serially at the tail (~1s).
            tail_prep->lm.emplace_back();
            auto enc = weight_loader::encode_gpt2_lm_head_weights(
                i, store, i.size.getRealHidDim(), i.size.hidDim, vocab_pre,
                i.slots, BootstrapPlan{});
            tail_prep->lm.front().w    = std::move(enc.w);
            tail_prep->lm.front().plan = BootstrapPlan{};
            // DEVICE-PREP ON THE WORKER — the actual fix (2026-07-23). gpt2_lm_head
            // does its load/evict INSIDE `if (blk->w.empty())`, so a prefetched block
            // (w already populated) SKIPS the device prep entirely: the tail then
            // multiplies against plaintexts that never reached the GPU -> deterministic
            // garbage (top1=512, w_mape~1.0; jobs 50091167/50092617/50093090). GPT-2 was
            // immune only because its presets set GPT2_LMHEAD_GRANULARITY, making
            // lm_head_streams() true (per-use loads). ViT leaves it unset -> resident load.
            // Doing it here (worker + loader stream, the same pattern the block loader
            // uses every block) also overlaps the ~10s upload, which is where the real
            // tail win lives — the encode alone is only ~2s.
            // ENCODE-ONLY on the worker. A worker-side device load does NOT survive to
            // the tail: block release / enc-cache eviction between block 11 and the tail
            // drops the device copies, and gpt2_lm_head skips its reload when blk->w is
            // already populated -> the tail multiplies against absent weights and decodes
            // ~ZERO logits (w_mape~1.000-1.004 == |0-ref|/|ref|, argmax junk 512; jobs
            // 50091167/50092617/50093090/50140014/15/50140478 across every variant).
            // The device load therefore stays at the tail (below); only the ~3.8s encode
            // is safely hideable.
            i.pt_stage_hook = saved_hook;
            tail_prep->ready.store(true);
            std::fprintf(stderr, "[tail-prefetch] lnf+lm_head encoded on the worker "
                         "in %.1fs (overlapped with the last bodies)\n",
                         std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count());
            return blk;
        };
    }
    if (const int stage_threads = pt_stage_block_threads(); stage_threads > 0) {
        BlockLoader raw_loader = std::move(loader);
        loader = [raw_loader, stage_threads](Inference& i, int b, cudaStream_t s) {
            using clk = std::chrono::steady_clock;
            auto secs = [](clk::time_point a, clk::time_point z) {
                return std::chrono::duration<double>(z - a).count();
            };
            i.begin_stage_block();
            // [ptstage-timing]: the hook fires between family encodes, so the delta since the
            // previous fire = that family's PREP+ENCODE wall (the first family also carries the
            // block's matrix load/transpose/rearrange). stage= is the pinned-arena memcpy time.
            // FHE_PT_STAGE_HOOK=0 disables the family-interleaved hook (sweep-only staging
            // after the full block encode — the round-2-validated protocol) to isolate it.
            static const bool use_hook = [] {
                const char* e = std::getenv("FHE_PT_STAGE_HOOK");
                return !(e && *e && std::atoi(e) == 0);
            }();
            double stage_s = 0.0;
            auto t_prev = clk::now();
            std::string fams;
            if (use_hook)
            i.pt_stage_hook = [&](const std::string& key, std::vector<Ptx>& pts) {
                const auto t0 = clk::now();
                const double enc = secs(t_prev, t0);
                stage_plaintexts(i, pts, stage_threads);
                stage_s += secs(t0, clk::now());
                if (enc > 0.25) {
                    char buf[64];
                    std::snprintf(buf, sizeof buf, " %s=%.2f", key.c_str(), enc);
                    fams += buf;
                }
                t_prev = clk::now();
            };
            const auto tb = clk::now();
            EncodedBlock blk = raw_loader(i, b, s);
            i.pt_stage_hook = nullptr;
            const auto ts = clk::now();
            stage_block_weights(i, blk, stage_threads);   // sweep leftovers + malloc_trim
            const auto te = clk::now();
            std::fprintf(stderr,
                         "[ptstage-timing] block=%d loader=%.2fs stage=%.2fs sweep+trim=%.2fs"
                         " enc_fams:%s\n",
                         b, secs(tb, te), stage_s, secs(ts, te), fams.c_str());
            return blk;
        };
        std::fprintf(stderr, "[encvit] pt-stage armed: worker-side pinned staging, threads=%d\n",
                     stage_threads);
    }

    const uint64_t bts0 = inf.fhe->total_bootstraps;
    const auto t_start = std::chrono::steady_clock::now();
    auto secs_since = [](const std::chrono::steady_clock::time_point& t0) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    StepProfiler& prof = inf.fhe->profile;
    prof.ensure_initialized();
    const bool prof_on = prof.on();

    auto body = [&chunks, &n_toks, &n_toks_imag, &secs_since, planned,
                 &prof, prof_on](Inference& i, PackedCtx&, int b) {
        const auto tb = std::chrono::steady_clock::now();
        if (prof_on) prof.reset();   // isolate THIS block's composition
        i.block_prefix = block_scope(b);
        i.output.capture_b = b;

        if (planned || std::getenv("FHE_GRAPH_DIR")) gpt2_reset_graph_runtime(i);
        const bool cap = begin_subgraph_capture(i, b);
        encoder_block_body(i, chunks, n_toks, n_toks_imag);
        if (cap) end_subgraph_capture(i, b);
        cudaDeviceSynchronize();
        std::fprintf(stderr, "[vit-time] block %d: %.2fs\n", b, secs_since(tb));
        std::fflush(stderr);
        if (prof_on) {
            std::ostringstream oss; oss << "[proftok] block=" << b;
            prof.dump(oss);
            std::fputs(oss.str().c_str(), stdout); std::fflush(stdout);
        }
    };
    auto release = [](Inference& i, int b) { vit_block_release(i, b); };
    PackedCtx carrier = inf.fhe->clone(chunks[0]);   // run_blocks threads one ct; state rides `chunks`
    run_blocks(inf, std::move(carrier), n_blocks, inf.mode, loader, body, release);
    finish_host_reclaim();
    const double blocks_s = secs_since(t_start);
    const auto t_tail = std::chrono::steady_clock::now();

    inf.block_prefix.clear();
    const bool tail_ready = tail_prefetch && tail_prep->ready.load();
    EncodedBlock lnf = load_final_ln_state(inf, store, parsed, BootstrapPlan{}, nullptr);
    PackedCtx h_fill = apply_final_ln(inf, chunks[0], lnf);   // filling ln_f (per-token css)
    if (inf.token_pair) {
        // CLS is token 0 of the A (Re) half; drop the B half before extraction.
        // im_cleanse doubles (2·Re), the scalar 0.5 undoes it level-free.
        WithStep _wc(inf, "tail_realify");
        inf.fhe->inplace_im_cleanse(h_fill);
        inf.fhe->inplace_mult(h_fill, 0.5);
        inf.token_pair = false;   // cachemir tail runs the real arm
    }
    PackedCtx cls = extract_token_i_cachemir(inf, h_fill, 0); // normed CLS -> cachemir
    inf.packing = inf.make_packing(PackingKind::Cachemir);
    inf.n_tok = 1;
    const int vocab = vocab_pre;
    // Prefetched tiles are encoded but NOT device-resident, and gpt2_lm_head skips its
    // load when blk->w is populated — do that load here, on the main thread, at use time.
    if (tail_ready) load_block_to_device(inf, tail_prep->lm.front(), nullptr);
    auto tiles = gpt2_lm_head(inf, cls, store, vocab, inf.slots,
                              tail_ready ? &tail_prep->lm : nullptr, BootstrapPlan{});
    if (tail_ready) {   // replicate gpt2_lm_head's uncached cleanup for the one-shot tail
        evict_block_from_device(inf, tail_prep->lm.front());
        inf.weight_store = nullptr;
    }
    cudaDeviceSynchronize();
    const double tail_s = secs_since(t_tail);
    const double e2e_s  = secs_since(t_start);
    const uint64_t bts  = inf.fhe->total_bootstraps - bts0;
    const char* gran = std::getenv("VIT_WEIGHT_GRANULARITY");
    std::printf("[encvit] SUMMARY blocks=%d gran=%s blocks_s=%.1f s_per_block=%.2f tail_s=%.1f "
                "e2e_s=%.1f bootstraps=%llu bts_per_block=%.1f unplanned_bts=%llu "
                "weight_relevels=%llu\n",
                n_blocks, (gran && *gran) ? gran : "plaintext",
                blocks_s, n_blocks > 0 ? blocks_s / n_blocks : 0.0, tail_s, e2e_s,
                static_cast<unsigned long long>(bts),
                n_blocks > 0 ? static_cast<double>(bts) / n_blocks : 0.0,
                static_cast<unsigned long long>(inf.fhe->unplanned_bootstrap_count),
                static_cast<unsigned long long>(inf.fhe->weight_relevel_count));
    std::fflush(stdout);
    return tiles;
}
