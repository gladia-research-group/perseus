#include "model/gpt2_model.h"
#include "model/gpt2.h"
#include "model/gpt2/internal.h"
#include "weight_loader.h"
#include "config_loader.h"
#include "math/matrix_ops.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

GPT2Model GPT2Model::load(const weight_loader::WeightStore& store,
                          const config_loader::ParsedConfigs& cfg,
                          const BlockPlans& plans,
                          InferenceMode mode,
                          bool cache_weights,
                          CKKSContextOptions ckks,
                          bool enable_prefill,
                          const BlockPlans* decode_plans,
                          const std::vector<BlockPlans>* prefill_chunk_plans,
                          const std::vector<BlockPlans>* decode_window_plans) {
    const auto& m = cfg.model;
    InferenceOptions opts;
    opts.ckks         = ckks;   // GPT2 supplies its own cachemir rot keys (make_gpt2_inference); no default set

    if (enable_prefill)
        opts.aux_packing_kinds = {PackingKind::CachemirFilling};

    opts.dim          = m.n_embd;
    opts.expanded     = m.n_inner;
    opts.hidDim       = matrix::next_pow2(m.n_embd);
    opts.expDim       = matrix::next_pow2(m.n_inner);
    opts.numHeadsReal = m.n_head;
    opts.numHeads     = matrix::next_pow2(m.n_head);
    opts.packing_kind = PackingKind::Cachemir;
    if (const char* pk = std::getenv("GPT2_PACKING"); pk && *pk)
        opts.packing_kind = parse_packing_kind(pk);
    opts.mode         = mode;

    const int slots0 = (opts.ckks.batch_size == 0) ? (1 << (opts.ckks.logN - 1))
                                                    : static_cast<int>(opts.ckks.batch_size);
    std::vector<int32_t> decode_steps;
    if (enable_prefill) {
        decode_steps = gpt2_decode_only_rot_steps(slots0, opts.hidDim, opts.expDim, opts.numHeads);
        opts.ckks.deferred_rot_steps.assign(decode_steps.begin(), decode_steps.end());
    }

    Inference inf      = make_gpt2_inference(opts);
    inf.cache_weights  = cache_weights;
    const int slots    = inf.slots;

    GPT2Model model(std::move(inf), store, cfg, plans, m.n_layers, decode_plans,
                    /*defer_block_cache=*/enable_prefill);
    model.prefill_enabled_ = enable_prefill;
    model.complex_decode_ = model.inf_.complex;
    model.prefill_chunk_plans_ = prefill_chunk_plans;
    model.decode_window_plans_ = decode_window_plans;
    if (enable_prefill) {
        model.filling_only_rot_steps_ =
            gpt2_filling_only_rot_steps(slots, opts.hidDim, opts.expDim, opts.numHeads);
        model.decode_only_rot_steps_  = decode_steps;
        model.decode_keys_loaded_     = false;
    }
    return model;
}

GPT2Model::GPT2Model(Inference inf,
                     const weight_loader::WeightStore& store,
                     const config_loader::ParsedConfigs& cfg,
                     const BlockPlans& plans,
                     int n_blocks,
                     const BlockPlans* decode_plans,
                     bool defer_block_cache)
    : inf_(std::move(inf)), store_(store), cfg_(cfg), plans_(plans),
      decode_plans_(decode_plans ? *decode_plans : plans),
      n_blocks_(n_blocks), defer_block_cache_(defer_block_cache) {
    vocab_  = static_cast<int>(store_.meta(weight_loader::gpt2_lm_head_name()).shape[0]);
    w_tile_ = inf_.slots;
    cutmax_cfg_ = cfg_.has_cutmax ? cutmax_config_from_calib(cfg_.cutmax)
                                  : default_gpt2_cutmax_config();
    std::fprintf(stderr, cfg_.has_cutmax
        ? "[cutmax_cfg] calibrated \"cutmax\" section from CONFIGS_PATH\n"
        : "[cutmax_cfg] WARNING: no \"cutmax\" section in CONFIGS_PATH -> BAKED 21s default "
          "(splice the frozen 9.2s T5 section; see CLAUDE.md hard-won rules)\n");
    if (inf_.cache_weights && !defer_block_cache_)
        for (int b = 0; b < n_blocks_; ++b)
            blocks_.push_back(load_block_state(inf_, store_, cfg_, decode_plans_.at(b), b, nullptr));
    lnf_    = load_final_ln_state(inf_, store_, cfg_, decode_plans_.at(n_blocks_), nullptr);
    loader_ = make_block_loader(store_, cfg_, decode_plans_);
}

void GPT2Model::ensure_decode_blocks() {
    if (!inf_.cache_weights || !defer_block_cache_ || !blocks_.empty()) return;
    configure_phase(Phase::Decode);
    const BlockPlans& p = active_decode_plans();
    for (int b = 0; b < n_blocks_; ++b)
        blocks_.push_back(load_block_state(inf_, store_, cfg_, p.at(b), b, nullptr));
    if (p.any_valid() && mask_T_ > 0)
        gpt2_generate_decode_masks(inf_, blocks_, n_blocks_, mask_T_);
}

void GPT2Model::select_decode_window(int abs_pos) {
    if (!decode_window_plans_) return;
    const int w = abs_pos / 32;
    if (w == active_window_) return;
    static const BlockPlans kEagerWindow;
    const BlockPlans* wp;
    if (w < static_cast<int>(decode_window_plans_->size())) {
        wp = &(*decode_window_plans_)[w];
    } else {
        std::fprintf(stderr, "[plan_phase] decode window %d beyond plan set (%zu) -> eager\n",
                     w, decode_window_plans_->size());
        wp = &kEagerWindow;
    }
    active_decode_plans_ = wp;
    active_window_ = w;
    std::fprintf(stderr, "[plan_phase] decode window %d installed (%s)\n",
                 w, wp->any_valid() ? "planned" : "eager");
    for (int b = 0; b < static_cast<int>(blocks_.size()); ++b)
        blocks_[b].plan = wp->at(b);
    lnf_.plan = wp->at(n_blocks_);
    loader_ = make_block_loader(store_, cfg_, *wp);
}

void gpt2_reset_graph_runtime(Inference& inf) {
    inf.fhe->ct_vars.clear();
    inf.fhe->pt_vars.clear();
    inf.fhe->graph_ct_counter = 0;
    inf.fhe->graph_pt_counter = 0;
    inf.fhe->current_runtime_node_id = 0;
}

namespace {

const char* graph_dir_env() {
    const char* v = std::getenv("FHE_GRAPH_DIR");
    return (v && *v) ? v : nullptr;
}

struct ScopeExit {
    std::function<void()> fn;
    bool armed = true;
    ~ScopeExit() { if (armed && fn) fn(); }
    void disarm() { armed = false; }
    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
    explicit ScopeExit(std::function<void()> f) : fn(std::move(f)) {}
};

}  // namespace

void GPT2Model::configure_phase(Phase phase, int n_tok) {
    switch (phase) {
        case Phase::Decode:
            inf_.packing.kind       = PackingKind::Cachemir;
            inf_.n_tok              = 1;
            inf_.weight_granularity = WeightGranularity::Block;     // decode: block-resident
            inf_.token_pair         = false;
            inf_.complex            = complex_decode_;               // full complex arm for decode
            break;
        case Phase::Prefill: {
            inf_.packing.kind       = PackingKind::CachemirFilling;
            inf_.complex            = false;   // filling arm; TP carries the complex payload
            const int t             = inf_.slots / inf_.size.hidDim;
            inf_.n_tok              = (inf_.token_pair && n_tok > t) ? t : n_tok;  // physical lanes <= t
            inf_.weight_store       = nullptr;
            const char* g = std::getenv("GPT2_PREFILL_GRANULARITY");
            inf_.weight_granularity =
                (g && std::string(g) == "linear")   ? WeightGranularity::Linear
              : (g && std::string(g) == "sublayer") ? WeightGranularity::Sublayer
                                                     : WeightGranularity::Plaintext;
            break;
        }
    }
}

void GPT2Model::free_filling_rot_keys() {
    if (filling_keys_freed_ || !prefill_enabled_ || filling_only_rot_steps_.empty())
        return;
    const size_t freed = inf_.fhe->free_rotation_steps(filling_only_rot_steps_);
    filling_keys_freed_ = true;
    std::fprintf(stderr,
        "[prefill] freed %zu/%zu filling-only rotation keys (decode-side headroom reclaimed)\n",
        freed, filling_only_rot_steps_.size());
    std::fflush(stderr);
}

void GPT2Model::ensure_decode_rot_keys() {
    if (decode_keys_loaded_) return;
    inf_.fhe->load_rotation_steps(decode_only_rot_steps_);
    decode_keys_loaded_ = true;
    std::fprintf(stderr, "[decode] GPU-loaded %zu deferred decode rotation keys\n",
                 decode_only_rot_steps_.size());
    std::fflush(stderr);
}

Sequence GPT2Model::start() {
    configure_phase(Phase::Decode);
    gpt2_reset_kv_cache(inf_, n_blocks_);
    lm_tiles_.clear();
    return Sequence{};
}

void GPT2Model::generate_decode_masks(int T) {
    mask_T_ = T;   // replayed by ensure_decode_blocks when the block cache is deferred
    if (!decode_plans_.any_valid()) return;   // eager mode: keep masks online (lazy on demand)
    configure_phase(Phase::Decode);
    gpt2_generate_decode_masks(inf_, blocks_, n_blocks_, T);
}

PackedCtx GPT2Model::advance(Sequence& s,
                             const std::vector<std::vector<double>>& embeddings) {
    if (embeddings.empty())
        throw std::runtime_error("GPT2Model::advance: empty embeddings");
    ensure_decode_rot_keys();
    configure_phase(Phase::Decode);   // encode_token_input reads the phase
    PackedCtx h;
    for (const auto& emb : embeddings) {
        PackedCtx x = entry_bts_
            ? pack_tokens(inf_, { emb }, /*target_level=*/24)
            : encode_token_input(inf_, emb);
        if (entry_bts_) {
            inf_.fhe->bootstrap(x.ct);
            cudaDeviceSynchronize();
        }
        h = advance_ct(s, std::move(x));
    }
    return h;
}

PackedCtx GPT2Model::advance_ct(Sequence& s, PackedCtx x) {
    ensure_decode_rot_keys();
    configure_phase(Phase::Decode);
    select_decode_window(s.abs_pos);
    ensure_decode_blocks();
    PackedCtx h = gpt2_decode_forward(inf_, std::move(x), s.abs_pos,
                                      n_blocks_, blocks_, loader_, lnf_,
                                      active_decode_plans().any_valid());
    ++s.abs_pos;
    return h;
}

PackedCtx GPT2Model::prefill(Sequence& s,
                             const std::vector<std::vector<double>>& prompt) {
    if (prompt.empty())
        throw std::runtime_error("GPT2Model::prefill: empty prompt");
    if (!prefill_enabled_)
        throw std::runtime_error(
            "GPT2Model::prefill: model loaded without the prefill packing "
            "(enable_prefill=false); decode-packed prompt loops are unsupported");

    const int m = static_cast<int>(prompt.size());
    const int t = inf_.slots / inf_.size.hidDim;   // real-lane token capacity per chunk

    auto chunk_arm = [&](int off) {
        const int r = m - off;
        const bool tp = inf_.fhe->complex_payload && r > t;
        return std::pair<bool, int>{tp, tp ? std::min(2 * t, r) : std::min(t, r)};
    };
    inf_.token_pair = chunk_arm(0).first;

    static const bool argmax_scan = [] {
        const char* v = std::getenv("PREFILL_ARGMAX_SCAN");
        return v && *v && *v != '0';
    }();
    prefill_lnf_chunks_.clear();

    ScopeExit phase_guard{ [this] { configure_phase(Phase::Decode); } };

    static const BlockPlans kEagerChunk;
    auto chunk_plan = [this](int c) -> const BlockPlans& {
        if (!prefill_chunk_plans_) return plans_;
        if (c < static_cast<int>(prefill_chunk_plans_->size()))
            return (*prefill_chunk_plans_)[c];
        std::fprintf(stderr, "[plan_phase] prefill chunk %d beyond plan set (%zu) -> eager\n",
                     c, prefill_chunk_plans_->size());
        return kEagerChunk;
    };

    configure_phase(Phase::Prefill, chunk_arm(0).second);
    gpt2_reset_kv_cache(inf_, n_blocks_);

    PackedCtx h;
    int last_chunk = 0;
    PackedCtx hidden;
    {
        static const bool ranged_capture = [] {
            const char* v = std::getenv("FHE_PREFILL_CAPTURE_RANGES");
            return v && *v && *v != '0';
        }();
        CKKSContext::MagnitudeSuppressScope _ms(*inf_.fhe, !ranged_capture);
        int chunk_idx = 0;
        for (int off = 0; off < m; ++chunk_idx) {
            const auto [tp_chunk, mj] = chunk_arm(off);
            inf_.token_pair = tp_chunk;
            if (!tp_chunk) inf_.n_tok_imag = 0;   // TP encode sets it; real chunks must clear
            last_chunk = chunk_idx;
            std::vector<std::vector<double>> chunk(prompt.begin() + off,
                                                   prompt.begin() + off + mj);
            configure_phase(Phase::Prefill, mj);
            inf_.output.capture_t = s.abs_pos + off;
            inf_.output.capture_chunk = chunk_idx;   // capture -> chunk_<c>/ template dirs
            PackedCtx x = encode_prefill_input(inf_, chunk);
            h = gpt2_prefill(inf_, std::move(x), store_, cfg_, chunk_plan(last_chunk),
                             n_blocks_, PrefillMode::Chunk);
            if (argmax_scan) prefill_lnf_chunks_.push_back(h);
            off += mj;
        }

        const BlockPlans& tail_plans = chunk_plan(last_chunk);   // final LN rides the last chunk's template
        if (tail_plans.any_valid() || graph_dir_env()) gpt2_reset_graph_runtime(inf_);
        begin_subgraph_capture(inf_, n_blocks_);
        EncodedBlock lnf = load_final_ln_state(inf_, store_, cfg_, tail_plans.at(n_blocks_), nullptr);
        hidden = apply_final_ln(inf_, h, lnf);
        end_subgraph_capture(inf_, n_blocks_);
        if (argmax_scan && !prefill_lnf_chunks_.empty()) {
            // ln_f every chunk while the filling rot keys are still live; tail reuses `hidden`
            for (size_t c = 0; c + 1 < prefill_lnf_chunks_.size(); ++c)
                prefill_lnf_chunks_[c] = apply_final_ln(inf_, prefill_lnf_chunks_[c], lnf);
            prefill_lnf_chunks_.back() = hidden;
        }
    }

    inf_.output.capture_chunk = -1;   // lnf rode the last chunk's template; decode captures flat
    free_filling_rot_keys();
    ensure_decode_rot_keys();
    configure_phase(Phase::Decode);
    static const bool kv_handoff = [] {
        const char* e = std::getenv("FHE_KV_HANDOFF");
        return !(e && *e == '0');
    }();
    if (kv_handoff) {
        const auto h_t0     = std::chrono::steady_clock::now();
        const uint32_t h_b0 = inf_.fhe->total_bootstraps;
        gpt2_kv_handoff_filling_to_cachemir(inf_, n_blocks_, m);
        std::fprintf(stderr, "[prefill] kv_handoff wall=%.1fs bts=%u\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - h_t0).count(),
                     inf_.fhe->total_bootstraps - h_b0);
        std::fflush(stderr);
    }
    else std::fprintf(stderr, "[prefill] FHE_KV_HANDOFF=0: K/V repack skipped (tail-only run)\n");
    phase_guard.disarm();
    s.abs_pos += m;
    return hidden;
}

std::vector<PackedCtx> GPT2Model::logit_tiles(const PackedCtx& hidden) {
    // Adopt worker-prepped lm_head tiles (tail prefetch, gpt2_prefill.cu): the
    // deferred complex-key registration happens HERE on the main thread.
    if (inf_.cache_weights && lm_tiles_.empty()) {
        std::vector<std::string> ckeys;
        if (gpt2_tail_lm_take(lm_tiles_, ckeys))
            for (const auto& k : ckeys) inf_.complex_weight_names.insert(k);
    }
    auto tiles = gpt2_lm_head(inf_, hidden, store_, vocab_, w_tile_,
                              inf_.cache_weights ? &lm_tiles_ : nullptr,
                              active_decode_plans().at(n_blocks_));
    end_subgraph_capture(inf_, n_blocks_);
    if (active_decode_plans().any_valid()) inf_.clear_bootstrap_plan();
    return tiles;
}

std::vector<double> GPT2Model::logits(const PackedCtx& hidden) {
    return decode_lm_head_logits(inf_, logit_tiles(hidden), vocab_, w_tile_);
}

std::vector<PackedCtx> GPT2Model::tail_logit_tiles(const PackedCtx& prefill_hidden, int prefill_len) {
    return tail_logit_tiles_at(prefill_hidden, prefill_len, prefill_len - 1);
}

std::vector<PackedCtx> GPT2Model::tail_logit_tiles_at(const PackedCtx& prefill_hidden,
                                                      int prefill_len, int pos) {
    const int cap  = inf_.slots / inf_.size.hidDim;    // real-lane tokens per filling chunk
    const bool tp   = inf_.fhe->complex_payload && prefill_len > cap;
    const int local = pos % (tp ? 2 * cap : cap);
    const int lane  = local % cap;                     // position within its half
    configure_phase(Phase::Decode);                    // cachemir readout state (already set post-prefill)
    PackedCtx h = prefill_hidden;
    if (tp) {
        auto halves = inf_.fhe->pair_unpack(prefill_hidden.ct);   // {A, B} realified, 0.5 folded
        h = inf_.pack(local >= cap ? halves.second : halves.first, prefill_hidden.packing.kind);
        inf_.fhe->inplace_im_cleanse(h);
        inf_.fhe->inplace_mult(h, 0.5);
    }
    PackedCtx tok = extract_token_i_cachemir(inf_, h, lane);
    return gpt2_lm_head(inf_, tok, store_, vocab_, w_tile_, nullptr, BootstrapPlan{});
}

void GPT2Model::prepare_feedback_weights(bool packed_z) {
    gpt2_prepare_feedback_weights(inf_, store_, vocab_, w_tile_, packed_z,
                                  fb_tiles_,
                                  &decode_plans_.at(n_blocks_ + 2));  // block 14
}

PackedCtx GPT2Model::cutmax_step(const std::vector<PackedCtx>& logit_tiles,
                                 int position,
                                 std::vector<PackedCtx>* z_out,
                                 double* argmax_s) {
    ensure_decode_rot_keys();
    configure_phase(Phase::Decode);
    return gpt2_cutmax_feedback(inf_, logit_tiles, store_, vocab_, w_tile_,
                                cutmax_cfg_,
                                fb_tiles_.empty() ? nullptr
                                                  : &fb_tiles_.front(),
                                position, z_out, argmax_s,
                                &decode_plans_.at(n_blocks_ + 1),   // block 13
                                &decode_plans_.at(n_blocks_ + 2));  // block 14
}

std::vector<double> GPT2Model::forward(Sequence& s,
                                       const std::vector<std::vector<double>>& embeddings) {
    return logits(advance(s, embeddings));
}

std::vector<int> GPT2Model::generate(
        Sequence& s,
        const std::vector<std::vector<double>>& prompt,
        const std::function<std::vector<double>(int)>& embed_of,
        int max_new_tokens) {
    if (prompt.empty())
        throw std::runtime_error("GPT2Model::generate: empty prompt");

    if (prompt.size() >= 2)
        prefill(s, std::vector<std::vector<double>>(prompt.begin(), prompt.end() - 1));
    PackedCtx h = advance(s, { prompt.back() });

    std::vector<int> out;
    out.reserve(max_new_tokens);
    for (int i = 0; i < max_new_tokens; ++i) {
        std::vector<double> lg = logits(h);
        const int tok = static_cast<int>(
            std::max_element(lg.begin(), lg.end()) - lg.begin());
        out.push_back(tok);
        if (i + 1 < max_new_tokens)
            h = advance(s, { embed_of(tok) });
    }
    return out;
}
