#include "model/bert.h"
#include "interrupt.h"
#include "model/gpt2/internal.h"
#include "attention.h"
#include "encoded_block.h"
#include "model/layer_norm.h"
#include "model/mlp.h"
#include "nonlinear.h"
#include "packing/cachemir/cachemir_rot_indices.h"
#include "packing/cachemir_filling/cachemir_filling_rot_indices.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

std::vector<int32_t> bert_rot_indices(int slots, int hidDim, int ffDim, int numHeads) {
    std::set<int32_t> rots;
    for (int32_t r : cachemir_filling::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads))
        rots.insert(r);
    for (int32_t r : cachemir::compute_gpt2_rot_indices(slots, hidDim, ffDim, numHeads))
        rots.insert(r);
    return {rots.begin(), rots.end()};
}

void bert_block_release(Inference& inf, int b) {
    { WithStep _w(inf, "block_sync"); cudaDeviceSynchronize(); }
    evict_block_weights(inf);
    inf.evict_enc_cache_device_scoped(block_scope(b));
    reclaim_host_async(inf);
}

void bert_block_body(Inference& inf, std::vector<PackedCtx>& xs,
                     const std::vector<int>& ns, const std::vector<int>& ns_im) {
    WithStep _w(inf, "encoder_block");
    const int d = inf.size.hidDim;

    const bool tp = inf.token_pair && !ns_im.empty() && ns_im[0] > 0;

    std::vector<int> chunk_base(ns.size(), 0);
    for (size_t c = 1; c < ns.size(); ++c)
        chunk_base[c] = chunk_base[c - 1] + ns[c - 1] + (tp && c - 1 < ns_im.size() ? ns_im[c - 1] : 0);
    auto set_counts = [&](size_t c) {
        inf.n_tok      = ns[c];
        inf.n_tok_imag = tp ? ns_im[c] : 0;
        inf.output.capture_t = chunk_base[c];
    };

    const size_t C = xs.size();
    std::vector<PackedCtx> skips(C), qs(C), ks(C), vs(C);
    for (size_t c = 0; c < C; ++c) {
        perseus_interrupt::poll();
        set_counts(c);
        const int res_lvl = inf.fhe->level_for_ct(xs[c].ct)
                          + static_cast<int>(inf.pending_rescale_primes(xs[c].ct));
        inf.name_graph_ct_if_absent(
            xs[c], (c == 0 ? std::string("transformer_block.x")
                           : "transformer_block.x" + std::to_string(c)) +
                       "-lvl=" + std::to_string(res_lvl));
        PackedCtx x = xs[c];
        WithStep _wq(inf, "qkv");
        inf.fhe->bootstrap_hint(x, inf.fhe->level_headroom(1), true);
        inf.fhe->level_hint(x, inf.fhe->level_headroom(1));
        skips[c] = x;
        auto qkv = linear_multi(inf, x, {"k", "v", "q"}, d, d, /*stream_pt=*/true);
        ks[c] = std::move(qkv[0]);
        vs[c] = std::move(qkv[1]);
        qs[c] = std::move(qkv[2]);
    }

    std::vector<PackedCtx> ats;
    {
        WithStep _wa(inf, "attn_core");
        if (tp && qs.size() != 1)
            throw std::runtime_error("[bert] token-pair supports exactly ONE packed chunk (got " +
                                     std::to_string(qs.size()) + ")");
        ats = bi_attention(inf, std::move(qs), std::move(ks), std::move(vs),
                           tp ? std::vector<int>{ns[0], ns_im[0]} : ns);
    }

    for (size_t c = 0; c < C; ++c) {
        perseus_interrupt::poll();
        set_counts(c);
        PackedCtx x = std::move(ats[c]);
        {
            WithStep _wo(inf, "out_proj");
            inf.fhe->bootstrap_hint(x, inf.fhe->level_headroom(1), true);
            inf.fhe->level_hint(x, inf.fhe->level_headroom(1));
            x = linear(inf, x, "out", d, d, /*stream_pt=*/true);
        }
        {
            WithStep _wr(inf, "attn_residual");
            x = inf.fhe->add(x, skips[c]);
        }
        {
            WithStep _wn(inf, "ln_1");          // POST-attention LayerNorm
            x = layer_norm(inf, x, "ln_1");
        }
        skips[c] = x;
        x = mlp_block(inf, x);   // canonical MLP: tiled/plain + streaming dispatch
        {
            WithStep _wr(inf, "mlp_residual");
            x = inf.fhe->add(x, skips[c]);
            skips[c] = PackedCtx{};
        }
        {
            WithStep _wn(inf, "ln_2");          // POST-MLP LayerNorm
            xs[c] = layer_norm(inf, x, "ln_2");
        }

        {
            const int t_cap = inf.slots / d;
            if (inf.n_tok < t_cap) {
                WithStep _wj(inf, "junk_mask");
                const std::string jtag = "bert.junk.n" + std::to_string(inf.n_tok);
                Ptx jm = inf.encode_at_cached(jtag, xs[c],
                    [&] { return inf.active_token_mask_vec(d, t_cap, 1.0); });
                xs[c] = inf.fhe->mult(xs[c], jm);
            }
        }
    }
}

}  // namespace

Inference make_bert_inference(InferenceOptions opts) {
    if (pt_stage_block_threads() > 0) fideslib::PrewarmStageArenas();
    const int slots = (opts.ckks.batch_size == 0)
                    ? (1 << (opts.ckks.logN - 1))
                    : static_cast<int>(opts.ckks.batch_size);
    opts.packing_kind = PackingKind::CachemirFilling;
    for (int32_t r : bert_rot_indices(slots, opts.hidDim, opts.expDim, opts.numHeads))
        opts.ckks.extra_rot_steps.push_back(r);
    Inference inf = make_inference(opts);
    inf.bidirectional = true;
    inf.use_cache = false;
    inf.token_pair = inf.fhe->complex_payload;
    return inf;
}

PackedCtx bert_forward(Inference& inf,
                       std::vector<PackedCtx> chunks,
                       const std::vector<int>& n_toks,
                       const weight_loader::WeightStore& store,
                       const config_loader::ParsedConfigs& parsed,
                       int n_blocks,
                       const std::vector<int>& n_toks_imag) {
    inf.weight_granularity = WeightGranularity::Plaintext;
    if (const char* g = std::getenv("BERT_WEIGHT_GRANULARITY"); g && *g) {
        const std::string gs(g);
        if      (gs == "linear")    inf.weight_granularity = WeightGranularity::Linear;
        else if (gs == "sublayer")  inf.weight_granularity = WeightGranularity::Sublayer;
        else if (gs == "block")     inf.weight_granularity = WeightGranularity::Block;
        else if (gs != "plaintext")
            throw std::runtime_error("BERT_WEIGHT_GRANULARITY: expected "
                                     "plaintext|linear|sublayer|block, got " + gs);
        std::fprintf(stderr, "[encbert] weight_granularity=%s\n", gs.c_str());
    }

    BlockPlans plans;   // loader captures by reference — must outlive run_blocks
    bool planned = false;
    if (const char* pd = std::getenv("FHE_BOOTSTRAP_PLACEMENTS_DIR"); pd && *pd) {
        plans = load_block_plans(pd, n_blocks);
        if (!plans.any_valid())
            throw std::runtime_error(std::string("bert_forward: FHE_BOOTSTRAP_PLACEMENTS_DIR set "
                                                 "but no valid block plans under ") + pd);
        std::fprintf(stderr, "[encbert] planned mode: %s\n", pd);
        planned = true;
    }
    BlockLoader loader = make_block_loader(store, parsed, plans);

    if (const int stage_threads = pt_stage_block_threads(); stage_threads > 0) {
        BlockLoader raw_loader = std::move(loader);
        static const bool use_hook = [] {
            const char* e = std::getenv("FHE_PT_STAGE_HOOK");
            return !(e && *e && std::atoi(e) == 0);
        }();
        loader = [raw_loader, stage_threads](Inference& i, int b, cudaStream_t s) {
            if (use_hook) {
                i.begin_stage_block();
                i.pt_stage_hook = [&i, stage_threads](const std::string&, std::vector<Ptx>& pts) {
                    stage_plaintexts(i, pts, stage_threads);
                };
            }
            EncodedBlock blk = raw_loader(i, b, s);
            i.pt_stage_hook = nullptr;
            stage_block_weights(i, blk, stage_threads);   // sweep leftovers + malloc_trim
            return blk;
        };
        std::fprintf(stderr, "[encbert] pt-stage armed: worker-side pinned staging, threads=%d\n",
                     stage_threads);
    }

    const bool multi_consume = pt_stage_block_threads() > 0 && n_toks.size() > 1;
    if (multi_consume) {
        inf.set_stage_multi_consume(true);
        std::fprintf(stderr, "[encbert] stage multi-consume: multi-chunk (%zu chunks)\n",
                     n_toks.size());
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
        bert_block_body(i, chunks, n_toks, n_toks_imag);
        if (cap) end_subgraph_capture(i, b);
        cudaDeviceSynchronize();
        std::fprintf(stderr, "[bert-time] block %d: %.2fs\n", b, secs_since(tb));
        std::fflush(stderr);
        if (prof_on) {
            std::ostringstream oss; oss << "[proftok] block=" << b;
            prof.dump(oss);
            std::fputs(oss.str().c_str(), stdout); std::fflush(stdout);
        }
    };
    auto release = [](Inference& i, int b) { bert_block_release(i, b); };
    PackedCtx carrier = inf.fhe->clone(chunks[0]);   // run_blocks threads one ct; state rides `chunks`
    run_blocks(inf, std::move(carrier), n_blocks, inf.mode, loader, body, release);
    finish_host_reclaim();
    if (multi_consume) inf.set_stage_multi_consume(false);
    const double blocks_s = secs_since(t_start);

    inf.block_prefix.clear();
    PackedCtx h_fill = std::move(chunks[0]);
    if (inf.token_pair) {
        WithStep _wc(inf, "tail_realify");
        inf.fhe->inplace_im_cleanse(h_fill);
        inf.fhe->inplace_mult(h_fill, 0.5);
        inf.token_pair = false;   // cachemir tail runs the real arm
    }
    PackedCtx cls = extract_token_i_cachemir(inf, h_fill, 0);
    inf.packing = inf.make_packing(PackingKind::Cachemir);
    inf.n_tok = 1;
    cudaDeviceSynchronize();

    const double e2e_s = secs_since(t_start);
    const uint64_t bts = inf.fhe->total_bootstraps - bts0;

    std::fprintf(stderr, "[encbert] blocks=%.1fs e2e=%.1fs bootstraps=%llu "
                 "bts_per_block=%.1f unplanned_bts=%llu weight_relevels=%llu\n",
                 blocks_s, e2e_s, static_cast<unsigned long long>(bts),
                 n_blocks > 0 ? static_cast<double>(bts) / n_blocks : 0.0,
                 static_cast<unsigned long long>(inf.fhe->unplanned_bootstrap_count),
                 static_cast<unsigned long long>(inf.fhe->weight_relevel_count));
    std::fflush(stderr);
    return cls;
}
