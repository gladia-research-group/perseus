#pragma once

#include "inference.h"
#include "encoded_block.h"
#include "cutmax.h"

#include <functional>
#include <vector>

namespace weight_loader { class WeightStore; }
namespace config_loader { struct ParsedConfigs; }

// Per-sequence mutable state. KV is stored in the model's Inference cache.
struct Sequence {
    int abs_pos = 0;   // tokens ingested so far (absolute position)
};

class GPT2Model {
public:
    static GPT2Model load(const weight_loader::WeightStore& store,
                          const config_loader::ParsedConfigs& cfg,
                          const BlockPlans& plans,
                          InferenceMode mode = InferenceMode::Threaded,
                          bool cache_weights = true,
                          CKKSContextOptions ckks = {},
                          bool enable_prefill = false,
                          const BlockPlans* decode_plans = nullptr,
                          const std::vector<BlockPlans>* prefill_chunk_plans = nullptr,
                          const std::vector<BlockPlans>* decode_window_plans = nullptr);

    // Adopt an already-configured Inference (caller set dims/mode/packing).
    GPT2Model(Inference inf,
              const weight_loader::WeightStore& store,
              const config_loader::ParsedConfigs& cfg,
              const BlockPlans& plans,
              int n_blocks,
              const BlockPlans* decode_plans = nullptr,
              bool defer_block_cache = false);

    Sequence start();

    void generate_decode_masks(int T);

    PackedCtx advance(Sequence& s,
                      const std::vector<std::vector<double>>& embeddings);

    PackedCtx advance_ct(Sequence& s, PackedCtx x);

    PackedCtx prefill(Sequence& s, const std::vector<std::vector<double>>& prompt);

    // Next-token logits over the vocab for a hidden state from advance().
    std::vector<double> logits(const PackedCtx& hidden);

    std::vector<PackedCtx> logit_tiles(const PackedCtx& hidden);

    std::vector<PackedCtx> tail_logit_tiles(const PackedCtx& prefill_hidden, int prefill_len);

    std::vector<PackedCtx> tail_logit_tiles_at(const PackedCtx& prefill_hidden, int prefill_len,
                                               int pos);

    // PREFILL_ARGMAX_SCAN=1: post-ln_f hidden of EVERY prefill chunk, oldest first (else empty).
    // Chunk pos/cap fed to tail_logit_tiles_at gives a full per-position prefill readout.
    const std::vector<PackedCtx>& prefill_lnf_chunks() const { return prefill_lnf_chunks_; }

    PackedCtx cutmax_step(const std::vector<PackedCtx>& logit_tiles,
                          int position,
                          std::vector<PackedCtx>* z_out = nullptr,
                          double* argmax_s = nullptr);

    void prepare_feedback_weights(bool packed_z);

    // Autoregressive generation bootstraps the prompt entry (advance) so it matches the feedback
    // entry level. Set to !teacher_forced by run_generate; off for decode/prefill (uniform entries).
    void set_entry_bts(bool v) { entry_bts_ = v; }

    // Convenience: advance then take logits of the last position.
    std::vector<double> forward(Sequence& s,
                                const std::vector<std::vector<double>>& embeddings);

    std::vector<int> generate(Sequence& s,
                              const std::vector<std::vector<double>>& prompt,
                              const std::function<std::vector<double>(int)>& embed_of,
                              int max_new_tokens);

    int n_blocks() const { return n_blocks_; }
    int vocab()    const { return vocab_; }
    Inference&       inference()       { return inf_; }
    const Inference& inference() const { return inf_; }

private:
    // Inference phases that own a coherent (packing, n_tok, weight_granularity) triple.
    enum class Phase { Decode, Prefill };
    void configure_phase(Phase phase, int n_tok = 1);
    void free_filling_rot_keys();
    void ensure_decode_rot_keys();
    void ensure_decode_blocks();
    void select_decode_window(int abs_pos);
    const BlockPlans& active_decode_plans() const {
        return active_decode_plans_ ? *active_decode_plans_ : decode_plans_;
    }

    Inference inf_;
    const weight_loader::WeightStore& store_;
    const config_loader::ParsedConfigs& cfg_;
    const BlockPlans& plans_;          // prefill-phase plan set (decode-only: the only set)
    const BlockPlans& decode_plans_;   // decode-phase plan set (defaults to plans_)
    const std::vector<BlockPlans>* prefill_chunk_plans_ = nullptr;   // per-chunk templates (compositional)
    const std::vector<BlockPlans>* decode_window_plans_ = nullptr;   // per-32-token decode windows (compositional)
    const BlockPlans* active_decode_plans_ = nullptr;                // current window (or decode_plans_)
    std::vector<PackedCtx> prefill_lnf_chunks_;                      // PREFILL_ARGMAX_SCAN only
    int active_window_ = -1;
    int n_blocks_ = 0;
    int vocab_    = 0;
    int w_tile_   = 0;
    bool entry_bts_ = false;                    // autoregressive: bootstrap the prompt entry (set by run_generate)
    bool prefill_enabled_ = false;
    bool complex_decode_ = false;               // cachemir_complex + prefill: decode PHASE runs the full complex
                                                // arm (prefill runs TP filling; the handoff complexifies the cache)
    bool filling_keys_freed_ = false;
    bool decode_keys_loaded_ = true;            // true unless prefill deferred them at load()
    bool defer_block_cache_ = false;            // cache_weights + prefill: encode decode blocks at first decode token
    int  mask_T_ = 0;                           // T recorded by generate_decode_masks (replayed after deferred load)
    std::vector<int> filling_only_rot_steps_;   // filling-exclusive rot steps (freed post-prefill)
    std::vector<int> decode_only_rot_steps_;    // decode-exclusive rot steps (deferred, loaded at handoff)
    std::vector<EncodedBlock> blocks_;
    EncodedBlock              lnf_;
    BlockLoader               loader_;
    std::vector<EncodedBlock> lm_tiles_;
    std::vector<EncodedBlock> fb_tiles_;   // feedback wte^T tiles (Z*wte)
    CutMaxConfig cutmax_cfg_;   // configs.json "cutmax" section when present, else baked default
};
