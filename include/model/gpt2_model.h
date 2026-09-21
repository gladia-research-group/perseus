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

    void overlap_setup_with_block0(int expected_tokens = 0);

    void generate_decode_masks(int T);

    PackedCtx advance(Sequence& s,
                      const std::vector<std::vector<double>>& embeddings);

    PackedCtx advance_ct(Sequence& s, PackedCtx x);

    PackedCtx prefill(Sequence& s, const std::vector<std::vector<double>>& prompt);

    std::vector<double> logits(const PackedCtx& hidden);

    std::vector<PackedCtx> logit_tiles(const PackedCtx& hidden);

    std::vector<PackedCtx> tail_logit_tiles(const PackedCtx& prefill_hidden, int prefill_len);

    std::vector<PackedCtx> tail_logit_tiles_at(const PackedCtx& prefill_hidden, int prefill_len,
                                               int pos);

    PackedCtx cutmax_step(const std::vector<PackedCtx>& logit_tiles,
                          int position,
                          std::vector<PackedCtx>* z_out = nullptr,
                          double* argmax_s = nullptr);

    void prepare_feedback_weights(bool packed_z);

    void set_entry_bts(bool v) { entry_bts_ = v; }

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
    const BlockPlans& plans_;          
    const BlockPlans& decode_plans_;   
    const std::vector<BlockPlans>* prefill_chunk_plans_ = nullptr;   
    const std::vector<BlockPlans>* decode_window_plans_ = nullptr;   
    const BlockPlans* active_decode_plans_ = nullptr;                
    int active_window_ = -1;
    int n_blocks_ = 0;
    int vocab_    = 0;
    int w_tile_   = 0;
    bool entry_bts_ = false;                    
    bool prefill_enabled_ = false;
    bool complex_decode_ = false;               
    bool filling_keys_freed_ = false;
    bool decode_keys_loaded_ = true;         
    bool defer_block_cache_ = false;         
    int  mask_T_ = 0;                        
    std::vector<int> filling_only_rot_steps_;
    std::vector<int> decode_only_rot_steps_;
    std::vector<EncodedBlock> blocks_;
    EncodedBlock              lnf_;
    BlockLoader               loader_;
    std::vector<EncodedBlock> lm_tiles_;
    std::vector<EncodedBlock> fb_tiles_;
    CutMaxConfig cutmax_cfg_;
};
