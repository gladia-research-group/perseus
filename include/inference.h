#pragma once
#include <filesystem>
#include <fstream>

#include "graph.h"
#include "fideslib_wrapper.h"
#include "fhe_errors.h"
#include "nonlinear.h"
#include "packing/packed_ctx.h"
#include "packing/cachemir/cachemir_masks.h"
#include "packing/diagonal/diagonal_masks.h"

#include <cuda_runtime.h>

#include <complex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <iostream>
#include <functional>
#include <cstdlib>

inline std::vector<int> stride_slots(const Packing& packing,
                                     int slots, int d, int stride,
                                     int start_offset) {
    if (is_cachemir(packing))
        return cachemir::stride_slots_cachemir(slots, d, stride, start_offset);
    if (is_diagonal(packing) || is_cachemir_filling(packing))
        return diagonal::stride_slots_diagonal(slots, d, stride, start_offset);
    throw std::runtime_error("stride_slots: unsupported packing");
}

inline std::vector<int> active_expanded_slots(const Packing& packing,
                                              int N, int d, int alpha, int e_real) {
    if (is_cachemir(packing))
        return cachemir::active_expanded_slots_cachemir(N, d, alpha, e_real);
    if (is_diagonal(packing) || is_cachemir_filling(packing))
        return diagonal::active_expanded_slots_diagonal(N, d, alpha, e_real);
    throw std::runtime_error("active_expanded_slots: unsupported packing");
}


  struct ModelSize {
      int dim          = 768;
      int expanded     = 3072;
      int hidDim       = 1024;
      int expDim       = 4096;
      int numHeads     = 16;   // padded
      int numHeadsReal = 12;   // real
      int seqLen       = 1024;

      int getRealHidDim()   const { return dim; }
      int getRealFfDim()    const { return expanded; }
      int getRealNumHeads() const { return numHeadsReal; }
      int getRealDHead()    const { return dim / numHeadsReal; }
  };

// Overlap policy only — caching is an independent axis (Inference::cache_weights).
enum class InferenceMode { Sync, Threaded, Prefetch };

enum class WeightGranularity { Block, Sublayer, Linear, Plaintext };

struct InferenceOutput {
    int capture_t = 0;
    int capture_b = 0;
    int capture_chunk = -1;
};

// Generic FHE inference context shared by all model types.
struct Inference {
    std::shared_ptr<CKKSContext> fhe;
    std::shared_ptr<GraphBuilder> graph;

    ModelSize size;
    int   logN        = 16;
    int   slots       = 0;

    bool parallel        = true;
    bool bench_mode      = false;

    InferenceMode mode   = InferenceMode::Threaded;
    bool cache_weights   = false;
    bool complex         = false;
    bool token_pair      = false;
    bool bidirectional   = false;

    Packing packing{};

    std::string weight_dir;

    std::unordered_map<std::string, std::vector<Ptx>> w;
    std::unordered_map<std::string, std::vector<std::vector<double>>> raw_w;
    std::unordered_map<std::string, std::vector<PackedCtx>> cache;
    std::unordered_map<std::string, Ptx> mask;
    std::vector<Ptx>                     cache_mask;

    std::unordered_map<std::string, std::vector<Ptx>>* weight_store = nullptr;

    std::string block_prefix;

    InferenceOutput output;   // per-token / per-block capture (test-only; see above)

    std::string scoped(const std::string& name) const { return block_prefix + name; }
    // Thread-safe form of scoped(): the caller supplies the prefix instead of reading the
    // mutable block_prefix member, so a worker thread can build scoped tags.
    static std::string scoped_in(const std::string& prefix, const std::string& name) {
        return prefix + name;
    }

    std::unordered_map<std::string, NormConfig>        norm_cfg;
    std::unordered_map<std::string, SoftmaxConfig> sm_cfg;
    std::unordered_map<std::string, GeLUConfig>     gelu_cfg;
    Packing make_packing(PackingKind kind) const {
        Packing p;
        p.kind     = kind;
        p.slots    = slots;
        p.hidDim   = size.hidDim;
        p.realDim  = size.getRealHidDim();
        p.numHeads = size.numHeads;
        p.t        = (size.hidDim > 0) ? slots / size.hidDim : 0;
        return p;
    }
    Packing make_packing() const { return make_packing(packing.kind); }

    PackedCtx pack(const Ctx& ct, PackingKind kind) const {
        return PackedCtx{ct, make_packing(kind)};
    }
    PackedCtx pack(const Ctx& ct) const { return PackedCtx{ct, make_packing()}; }

    void load_plaintext(Ptx& pt, cudaStream_t stream = nullptr) {
        if (!pt || pt->loaded) return;
        cc()->LoadPlaintext(pt, stream);
    }

    void extract_plaintext(Ptx& pt) {
        if (!pt || pt->loaded) return;
        cc()->ExtractRawPlaintext(pt);
    }

    void begin_stage_block() { cc()->BeginStageBlock(); }
    void begin_stage_block(int owner) { cc()->BeginStageBlockOwned(owner); }
    void release_stage_block(int owner) { cc()->ReleaseStageBlock(owner); }
    void set_persistent_staging(bool on) { cc()->SetPersistentStaging(on); }
    void evict_plaintext(Ptx& pt) {
        if (!pt || !pt->loaded || pt->gpu == 0) return;
        cc()->EvictDevicePlaintext(pt->gpu);
        pt->gpu    = 0;
        pt->loaded = false;
    }

    void evict_weights(const std::string& key) {
        auto it = w.find(key);
        if (it == w.end()) return;
        for (auto& pt : it->second) evict_plaintext(pt);
        w.erase(it);
    }

    uint32_t pending_rescale_primes(const Ctx& ct) const {
        if (fhe) return fhe->pending_rescale_primes(ct);
        return (ct && ct->GetNoiseScaleDeg() == 2) ? 1u : 0u;
    }

    template <class Vec>
    Ptx encode_tagged(const Vec& v, uint32_t lv) const {
        Ptx pt = cc()->MakeCKKSPackedPlaintext(v, /*noiseScaleDeg=*/1, lv);
        if (fhe) fhe->tag_plaintext(pt, v);   // real and complex overloads
        return pt;
    }

    Ptx encode_at(const std::vector<double>& v, const Ctx& ct) const {
        const uint32_t lv = static_cast<uint32_t>(level_of(ct))
                          + pending_rescale_primes(ct);
        return encode_tagged(v, lv);
    }

    Ptx encode_at(const std::vector<double>& v, const PackedCtx& pc) const {
        return encode_at(v, pc.ct);
    }

    Ptx encode_complex_const_at(double re, double im, const Ctx& ct) const {
        const uint32_t lv = static_cast<uint32_t>(level_of(ct))
                          + pending_rescale_primes(ct);
        return fhe->complex_const_pt(re, im, static_cast<int>(lv));
    }
    Ptx encode_complex_const_at(double re, double im, const PackedCtx& pc) const {
        return encode_complex_const_at(re, im, pc.ct);
    }

    std::vector<double> stride_mask_vec(int d, int stride, double scale = 1.0,
                                        double fill_value = 0.0,
                                        int start_offset = 0) const {
        std::vector<double> v(slots, fill_value);
        const auto active = stride_slots(packing, slots, d, stride, start_offset);
        for (int slot : active) v[slot] = scale;
        return v;
    }
    std::vector<double> active_token_mask_vec(int d, int t, double scale) const {
        std::vector<double> v(slots, 0.0);
        const auto active = stride_slots(packing, slots, d, t, 0);
        for (int slot : active)
            if (is_cachemir(packing) || (slot % t) < n_tok) v[slot] = scale;
        return v;
    }

    std::shared_ptr<const std::vector<uint8_t>> live_lane_mask_token() const {
        const int t = slots / size.hidDim;
        auto m = std::make_shared<std::vector<uint8_t>>(slots, uint8_t(0));
        for (int slot = 0; slot < slots; ++slot) {
            const int k = slot / t, i = slot % t;
            if (k % size.numHeads < size.getRealNumHeads() && i < n_tok) (*m)[slot] = 1;
        }
        return m;
    }

    std::shared_ptr<const std::vector<uint8_t>> live_lane_mask_expanded() const {
        const int t_out = slots / size.expDim;
        auto m = std::make_shared<std::vector<uint8_t>>(slots, uint8_t(0));
        for (int j = 0; j < size.getRealFfDim(); ++j)
            for (int tok = 0; tok < std::min(t_out, n_tok); ++tok)
                (*m)[j * t_out + tok] = 1;
        return m;
    }
    std::vector<double> active_expanded_mask_vec(double scale, double fill_value) const {
        std::vector<double> v(slots, fill_value);
        const int d     = size.hidDim;
        const int alpha = size.expDim / size.hidDim;
        const auto active = active_expanded_slots(packing, slots, d, alpha, size.expanded);
        for (int slot : active) v[slot] = scale;
        return v;
    }
    std::vector<double> active_residual_mask_vec(int d, int t) const {
        return active_token_mask_vec(d, t, 1.0);
    }

    Ptx encode_stride_mask_at(int d, int stride, const Ctx& ct,
                              double scale = 1.0, double fill_value = 0.0,
                              int start_offset = 0) const {
        return encode_at(stride_mask_vec(d, stride, scale, fill_value, start_offset), ct);
    }

    Ptx encode_stride_mask_at(int d, int stride, const PackedCtx& pc,
                              double scale = 1.0, double fill_value = 0.0,
                              int start_offset = 0) const {
        return encode_stride_mask_at(d, stride, pc.ct, scale, fill_value, start_offset);
    }

    // Memoised: static stride mask (f(d, stride, scale, fill) — block/token-independent).
    Ptx encode_stride_mask_at_cached(const std::string& tag, int d, int stride, const Ctx& ct,
                                     double scale = 1.0, double fill_value = 0.0,
                                     int start_offset = 0) const {
        return encode_at_cached(tag, stride_mask_vec(d, stride, scale, fill_value, start_offset), ct);
    }
    Ptx encode_stride_mask_at_cached(const std::string& tag, int d, int stride, const PackedCtx& pc,
                                     double scale = 1.0, double fill_value = 0.0,
                                     int start_offset = 0) const {
        return encode_stride_mask_at_cached(tag, d, stride, pc.ct, scale, fill_value, start_offset);
    }

    Ptx encode_stride_values_at(int d, int stride, const Ctx& ct,
                                const std::vector<double>& values,
                                double fill_value = 0.0,
                                int start_offset = 0) const {
        std::vector<double> v(slots, fill_value);
        const auto active = stride_slots(packing, slots, d, stride, start_offset);
        for (int k = 0; k < d; ++k) v[active[k]] = values[k];
        return encode_at(v, ct);
    }

    Ptx encode_stride_values_at(int d, int stride, const PackedCtx& pc,
                                const std::vector<double>& values,
                                double fill_value = 0.0,
                                int start_offset = 0) const {
        return encode_stride_values_at(d, stride, pc.ct, values, fill_value, start_offset);
    }

    Ptx encode_active_expanded_mask_at(const Ctx& ct,
                                       double scale = 1.0,
                                       double fill_value = 0.0) const {
        return encode_at(active_expanded_mask_vec(scale, fill_value), ct);
    }

    Ptx encode_active_expanded_mask_at(const PackedCtx& pc,
                                       double scale = 1.0,
                                       double fill_value = 0.0) const {
        return encode_active_expanded_mask_at(pc.ct, scale, fill_value);
    }

    // Memoised: the active-expanded mask is static (f(scale, fill, sizes)).
    Ptx encode_active_expanded_mask_at_cached(const std::string& tag, const Ctx& ct,
                                              double scale = 1.0,
                                              double fill_value = 0.0) const {
        return encode_at_cached(tag, active_expanded_mask_vec(scale, fill_value), ct);
    }
    Ptx encode_active_expanded_mask_at_cached(const std::string& tag, const PackedCtx& pc,
                                              double scale = 1.0,
                                              double fill_value = 0.0) const {
        return encode_active_expanded_mask_at_cached(tag, pc.ct, scale, fill_value);
    }

    Ptx encode_active_token_mask_at_cached(const std::string& tag, int d, int t,
                                           const Ctx& ct, double scale) const {
        return encode_at_cached(tag, active_token_mask_vec(d, t, scale), ct);
    }

    Ptx encode_active_token_mask_at(int d, int t, const Ctx& ct, double scale) const {
        return encode_at(active_token_mask_vec(d, t, scale), ct);
    }
    Ptx encode_active_token_mask_at(int d, int t, const PackedCtx& pc, double scale) const {
        return encode_active_token_mask_at(d, t, pc.ct, scale);
    }
    Ptx encode_active_token_mask_at_cached(const std::string& tag, int d, int t,
                                           const PackedCtx& pc, double scale) const {
        return encode_active_token_mask_at_cached(tag, d, t, pc.ct, scale);
    }

    Ptx gelu_half_mask(const PackedCtx& pc, double scale = 0.5) const {
        const std::string nt = token_pair ? ".nt" + std::to_string(n_tok) : "";
        if (mlp_tile_dim > 0) {
            const int t = slots / mlp_tile_dim;
            return encode_active_token_mask_at_cached(
                "gelu.half.tile" + std::to_string(mlp_tile_dim) + ".s" + std::to_string(scale) + nt,
                mlp_tile_dim, t, pc, scale);
        }
        return encode_active_expanded_mask_at_cached(
            "gelu.half.s" + std::to_string(scale) + nt, pc, scale);
    }

    Ptx encode_inactive_token_mask_at(const Ctx& ct, double fill) const {
        std::vector<double> v(slots, 0.0);
        const int t = (size.hidDim > 0) ? slots / size.hidDim : slots;
        for (int j = 0; j < slots; ++j)
            if (j % t >= n_tok) v[j] = fill;
        return encode_at(v, ct);
    }

    Ptx encode_inactive_token_mask_at(const PackedCtx& pc, double fill) const {
        return encode_inactive_token_mask_at(pc.ct, fill);
    }

    Ptx encode_inactive_token_mask_at_cached(const std::string& tag, const Ctx& ct,
                                             double fill) const {
        std::vector<double> v(slots, 0.0);
        const int t = (size.hidDim > 0) ? slots / size.hidDim : slots;
        for (int j = 0; j < slots; ++j)
            if (j % t >= n_tok) v[j] = fill;
        return encode_at_cached(tag, v, ct);
    }
    Ptx encode_inactive_token_mask_at_cached(const std::string& tag, const PackedCtx& pc,
                                             double fill) const {
        return encode_inactive_token_mask_at_cached(tag, pc.ct, fill);
    }

    Ptx encode_active_residual_mask_at(int d, int t, const Ctx& ct) const {
        return encode_at(active_residual_mask_vec(d, t), ct);
    }
    Ptx encode_active_residual_mask_at(int d, int t, const PackedCtx& pc) const {
        return encode_active_residual_mask_at(d, t, pc.ct);
    }

    // Memoised: static residual mask (f(d, t, n_tok, packing) — block/token-independent).
    Ptx encode_active_residual_mask_at_cached(const std::string& tag, int d, int t,
                                              const Ctx& ct) const {
        return encode_at_cached(tag, active_residual_mask_vec(d, t), ct);
    }
    Ptx encode_active_residual_mask_at_cached(const std::string& tag, int d, int t,
                                              const PackedCtx& pc) const {
        return encode_active_residual_mask_at_cached(tag, d, t, pc.ct);
    }

    std::pair<uint32_t, uint32_t> encode_like_params(const Ctx& ref) const {
        const uint32_t nsd_ref = static_cast<uint32_t>(ref->GetNoiseScaleDeg());
        const uint32_t lv_ref  = static_cast<uint32_t>(level_of(ref));
        if (!fhe || fhe->composite_degree <= 1) return {lv_ref, nsd_ref};
        return {lv_ref + pending_rescale_primes(ref), 1u};
    }

    // Re-encode a slot vector to exactly match a ct's level + noiseScaleDeg.

    Ptx encode_like(const std::vector<double>& v, const Ctx& ref) const {
        const auto [lv, nsd] = encode_like_params(ref);
        return cc()->MakeCKKSPackedPlaintext(v, nsd, lv);
    }
    Ptx encode_like(const std::vector<double>& v, const PackedCtx& ref) const {
        return encode_like(v, ref.ct);
    }

    Ptx encode_like(const Ptx& source, const Ctx& ref) const {
        return encode_like(source->GetRealPackedValue(), ref);
    }
    Ptx encode_like(const Ptx& source, const PackedCtx& ref) const {
        return encode_like(source, ref.ct);
    }

    mutable std::unordered_map<std::string, Ptx> enc_cache;
    mutable uint64_t enc_cache_hit = 0, enc_cache_miss = 0;

    bool strict_masks = false;        
    bool strict_masks_armed = false;  
    mutable uint64_t mask_strict_miss = 0;

    static std::string enc_cache_key(const std::string& tag, uint32_t lv, uint32_t nsd) {
        return tag + "#L" + std::to_string(lv) + ":d" + std::to_string(nsd);
    }

    size_t clear_enc_cache() {
        const size_t n = enc_cache.size();
        for (auto& kv : enc_cache) evict_plaintext(kv.second);
        enc_cache.clear();
        return n;
    }

    size_t evict_enc_cache_device() {
        size_t n = 0;
        for (auto& kv : enc_cache)
            if (kv.second && kv.second->loaded) { evict_plaintext(kv.second); ++n; }
        return n;
    }

    size_t evict_enc_cache_device_scoped(const std::string& prefix) {
        size_t n = 0;
        for (auto& kv : enc_cache)
            if (kv.second && kv.second->loaded &&
                kv.first.compare(0, prefix.size(), prefix) == 0) {
                evict_plaintext(kv.second);
                ++n;
            }
        return n;
    }

    bool erase_enc_cache(const std::string& tag, uint32_t lv) {
        auto it = enc_cache.find(enc_cache_key(tag, lv, 1));
        if (it == enc_cache.end()) return false;
        if (it->second) evict_plaintext(it->second);
        enc_cache.erase(it);
        return true;
    }

    size_t erase_enc_cache_all(const std::string& tag) {
        const std::string prefix = tag + "#L";
        size_t n = 0;
        for (auto it = enc_cache.begin(); it != enc_cache.end(); ) {
            if (it->first.compare(0, prefix.size(), prefix) == 0) {
                if (it->second) evict_plaintext(it->second);
                it = enc_cache.erase(it);
                ++n;
            } else {
                ++it;
            }
        }
        return n;
    }

    template <class Gen>
    void prime_enc_cache_gen(const std::string& tag, uint32_t lv, Gen&& gen) const {
        const std::string key = enc_cache_key(tag, lv, 1);
        if (enc_cache.find(key) != enc_cache.end()) return;
        Ptx pt = encode_tagged(gen(), lv);
        enc_cache.emplace(key, std::move(pt));
    }

    void prime_enc_cache(const std::string& tag, const std::vector<double>& v,
                         uint32_t lv) const {
        prime_enc_cache_gen(tag, lv, [&]() -> const std::vector<double>& { return v; });
    }

    template <class Gen>
    void prime_enc_cache_lazy(const std::string& tag, uint32_t lv, Gen&& gen) const {
        prime_enc_cache_gen(tag, lv, std::forward<Gen>(gen));
    }

    void adopt_enc_cache(const std::string& tag, uint32_t lv, Ptx pt) const {
        enc_cache.emplace(enc_cache_key(tag, lv, 1), std::move(pt));
    }

    void prime_enc_cache(const std::string& tag, const std::vector<std::complex<double>>& v,
                         uint32_t lv) const {
        prime_enc_cache_gen(tag, lv,
                            [&]() -> const std::vector<std::complex<double>>& { return v; });
    }

    template <class Build>
    Ptx encode_at_cached_impl(const std::string& tag, uint32_t lv, Build&& build) const {
        const std::string key = enc_cache_key(tag, lv, 1);
        auto it = enc_cache.find(key);
        if (it != enc_cache.end()) { ++enc_cache_hit; return it->second; }
        ++enc_cache_miss;
        if (strict_masks) {
            ++mask_strict_miss;
            throw fhe::MaskError(
                "[mask_miss] strict planned-mask cache miss: '" + key +
                "' not pre-generated by generate_decode_masks (plan mask_levels gap or "
                "stale capture/plan). Re-capture/re-plan to refresh mask_levels.");
        }
        const auto vals = build();
        Ptx pt = encode_tagged(vals, lv);
        enc_cache.emplace(key, pt);
        return pt;
    }

    // encode_at, memoised. `tag` must uniquely identify the slot-vector content.
    Ptx encode_at_cached(const std::string& tag, const std::vector<double>& v,
                         const Ctx& ct) const {
        const uint32_t lv = static_cast<uint32_t>(level_of(ct))
                          + pending_rescale_primes(ct);
        return encode_at_cached_impl(tag, lv, [&]() -> const std::vector<double>& { return v; });
    }
    Ptx encode_at_cached(const std::string& tag, const std::vector<double>& v,
                         const PackedCtx& pc) const {
        return encode_at_cached(tag, v, pc.ct);
    }

    template <class Gen>
    Ptx encode_at_cached(const std::string& tag, const Ctx& ct, Gen&& gen) const {
        const uint32_t lv = static_cast<uint32_t>(level_of(ct))
                          + pending_rescale_primes(ct);
        return encode_at_cached_impl(tag, lv, std::forward<Gen>(gen));
    }
    template <class Gen>
    Ptx encode_at_cached(const std::string& tag, const PackedCtx& pc, Gen&& gen) const {
        return encode_at_cached(tag, pc.ct, std::forward<Gen>(gen));
    }

    template <class Gen>
    Ptx encode_at_cached_complex(const std::string& tag, const Ctx& ct, Gen&& gen) const {
        const uint32_t lv = static_cast<uint32_t>(level_of(ct))
                          + pending_rescale_primes(ct);
        const std::string key = enc_cache_key(tag, lv, 1);
        auto it = enc_cache.find(key);
        if (it != enc_cache.end()) { ++enc_cache_hit; return it->second; }
        ++enc_cache_miss;
        if (strict_masks) {
            ++mask_strict_miss;
            throw fhe::MaskError("[mask_miss] strict planned-mask cache miss (complex): '" + key + "'");
        }
        const auto vals = gen();
        Ptx pt = cc()->MakeCKKSPackedPlaintext(vals, /*noiseScaleDeg=*/1, lv);
        if (fhe) fhe->tag_plaintext(pt, vals);
        enc_cache.emplace(key, pt);
        return pt;
    }
    template <class Gen>
    Ptx encode_at_cached_complex(const std::string& tag, const PackedCtx& pc, Gen&& gen) const {
        return encode_at_cached_complex(tag, pc.ct, std::forward<Gen>(gen));
    }

    template <class Build>
    Ptx encode_like_cached_impl(const std::string& tag, const Ctx& ref, Build&& build) const {
        const auto [lv, nsd] = encode_like_params(ref);
        const std::string key = enc_cache_key(tag, lv, nsd);
        auto it = enc_cache.find(key);
        if (it != enc_cache.end()) { ++enc_cache_hit; return it->second; }
        ++enc_cache_miss;
        const auto vals = build();
        Ptx pt = cc()->MakeCKKSPackedPlaintext(vals, nsd, lv);
        if (fhe) fhe->tag_plaintext(pt, vals);
        enc_cache.emplace(key, pt);
        return pt;
    }
    Ptx encode_like_cached(const std::string& tag, const std::vector<double>& v,
                           const Ctx& ref) const {
        return encode_like_cached_impl(tag, ref, [&]() -> const std::vector<double>& { return v; });
    }
    Ptx encode_like_cached(const std::string& tag, const std::vector<double>& v,
                           const PackedCtx& ref) const {
        return encode_like_cached(tag, v, ref.ct);
    }
    // Lazy overloads: the vector is built by `gen` only on a cache miss.
    template <class Gen>
    Ptx encode_like_cached(const std::string& tag, const Ctx& ref, Gen&& gen) const {
        return encode_like_cached_impl(tag, ref, std::forward<Gen>(gen));
    }
    template <class Gen>
    Ptx encode_like_cached(const std::string& tag, const PackedCtx& ref, Gen&& gen) const {
        return encode_like_cached_impl(tag, ref.ct, std::forward<Gen>(gen));
    }

    template <class Gen>
    Ptx encode_like_cached_complex_impl(const std::string& tag, const Ctx& ref, Gen&& gen) const {
        const auto [lv, nsd] = encode_like_params(ref);
        const std::string key = enc_cache_key(tag, lv, nsd);
        auto it = enc_cache.find(key);
        if (it != enc_cache.end()) { ++enc_cache_hit; return it->second; }
        ++enc_cache_miss;
        const auto vals = gen();
        Ptx pt = cc()->MakeCKKSPackedPlaintext(vals, nsd, lv);
        if (fhe) fhe->tag_plaintext(pt, vals);
        enc_cache.emplace(key, pt);
        return pt;
    }
    template <class Gen>
    Ptx encode_like_cached_complex(const std::string& tag, const PackedCtx& ref, Gen&& gen) const {
        return encode_like_cached_complex_impl(tag, ref.ct, std::forward<Gen>(gen));
    }

    template <class RealGen>
    Ptx encode_additive_like(const std::string& tag, const PackedCtx& ref, RealGen&& rv) const {
        if (!token_pair) return encode_like_cached(tag, ref, std::forward<RealGen>(rv));
        return encode_like_cached_complex(tag + ".ri", ref, [&] {
            auto v = rv();
            std::vector<std::complex<double>> c(v.size());
            for (size_t s = 0; s < v.size(); ++s) c[s] = {v[s], v[s]};
            return c;
        });
    }

    void add_affine_term(PackedCtx& ct, const std::string& name) {
        if (!token_pair) { fhe->inplace_add(ct, weights_at(name, ct)[0]); return; }
        Ptx s = encode_like_cached_complex(scoped(name) + ".ri", ct, [&] {
            auto v = w.at(name)[0]->GetRealPackedValue();
            std::vector<std::complex<double>> c(v.size());
            for (size_t k = 0; k < v.size(); ++k) c[k] = {v[k], v[k]};
            return c;
        });
        fhe->inplace_add(ct, s);
    }

    std::vector<Ptx>& weights_at(const std::string& name, const Ctx& ref) {
        std::vector<Ptx>& tiles = w.at(name);

        const uint32_t lv = static_cast<uint32_t>(level_of(ref))
                          + pending_rescale_primes(ref);

        std::vector<Ptx>* canon = nullptr;
        if (weight_store) {
            auto it = weight_store->find(name);
            if (it != weight_store->end() && it->second.size() == tiles.size())
                canon = &it->second;
        }
        for (size_t i = 0; i < tiles.size(); ++i) {
            const uint32_t enc_lv = static_cast<uint32_t>(tiles[i]->GetLevel());
            if (enc_lv == lv) continue;  // aligned
            if (enc_lv != lv + 1u) {
                std::fprintf(stderr, "[weight_dg] name=%s enc=%u eff=%u raw=%d deg=%d\n",
                             name.c_str(), enc_lv, lv, (int)level_of(ref),
                             (int)ref->GetNoiseScaleDeg());
                fhe->warn_planned_weight_relevel(name, enc_lv, lv);   // no-op unless planned
            }
            ++fhe->weight_relevel_count;   // total re-encodes (diagnostic; 0 = weight_levels perfect)

            fhe->record_weight_relevel(name, enc_lv, lv,
                                       /*enc_deg=*/1u,
                                       static_cast<uint32_t>(ref->GetNoiseScaleDeg()));

            Ptx adapted = complex_weight_names.count(name)
                ? cc()->MakeCKKSPackedPlaintext(tiles[i]->GetCKKSPackedValue(), /*noiseScaleDeg=*/1, lv)
                : cc()->MakeCKKSPackedPlaintext(tiles[i]->GetRealPackedValue(), /*noiseScaleDeg=*/1, lv);
            evict_plaintext(tiles[i]);
            tiles[i] = adapted;
            if (canon) (*canon)[i] = adapted;
        }
        return tiles;
    }
    std::vector<Ptx>& weights_at(const std::string& name, const PackedCtx& ref) {
        return weights_at(name, ref.ct);
    }

    std::unordered_set<std::string> complex_weight_names;
    std::unordered_set<std::string> token_basis_weights;

    std::unordered_map<std::string, int> cache_count;
    int& k_count() { return cache_count[scoped("k")]; }
    int& v_count() { return cache_count[scoped("v")]; }

    bool use_cache = true;
    bool suppress_kv_periodic_bts = false;

    WeightGranularity weight_granularity = WeightGranularity::Block;

    std::function<void(const std::string&, std::vector<Ptx>&)> pt_stage_hook;

    int n_tok = 1;
    int n_tok_imag = 0;
    int mlp_tile_dim = 0;

    // Prefill MLP is tiled (per-hidDim up/down tiles) when the chunk holds more tokens
    // than the expanded layout can pack.
    bool tiled_mlp() const {
        if (!(is_cachemir_filling(packing) && size.expDim > 0)) return false;
        return n_tok > slots / size.expDim;
    }

    CC& cc() { return fhe->cc; }
    const CC& cc() const { return fhe->cc; }

    CKKSContext& cc_ctx() { return *fhe; }
    const CKKSContext& cc_ctx() const { return *fhe; }

    void enable_graph_capture() {
        if (!graph) {
            graph = std::make_shared<GraphBuilder>();
        } else {
            graph->clear();
        }
        if (fhe) {
            fhe->attach_graph_builder(graph);
        }
    }

    void disable_graph_capture() {
        if (fhe) {
            fhe->detach_graph_builder();
        }
        graph.reset();
    }

    bool graph_capture_enabled() const {
        return graph && graph->enabled();
    }

    void name_graph_ct(const Ctx& ct, const std::string& name, bool overwrite = true) {
        if (fhe) {
            fhe->name_ct(ct, name, overwrite);
        }
    }

    void name_graph_ct(const PackedCtx& pc, const std::string& name, bool overwrite = true) {
        name_graph_ct(pc.ct, name, overwrite);
    }

    void name_graph_ct_if_absent(const Ctx& ct, const std::string& name) {
        if (fhe) {
            fhe->name_ct_if_absent(ct, name);
        }
    }

    void name_graph_ct_if_absent(const PackedCtx& pc, const std::string& name) {
        name_graph_ct_if_absent(pc.ct, name);
    }
    void export_graph_json(const std::string& path) const {
        if (graph) {
            if (fhe) fhe->drain_magnitudes();   // async magnitude capture: patch before write
            graph->export_json(path);
            write_capture_env_json(path);
        }
    }

    void write_capture_env_json(const std::string& graph_path) const {
        static const char* kLoadBearing[] = {
            "CHAIN", "LOGN", "AUTO_BTS_LEVEL", "BTS_ITERATIONS",
            "CKKS_COMPLEX", "GPT2_PACKING", "GPT2_CACHE",
            "GPT2_FOLD_LN1", "GPT2_FOLD_LN2", "GPT2_FOLD_LNF",
            "CACHE_READ_LEVEL_K", "CACHE_READ_LEVEL_V", "FHE_LMHEAD_CAP",
        };
        const std::filesystem::path dir = std::filesystem::path(graph_path).parent_path();
        std::ofstream f(dir / "capture_env.json");
        if (!f) return;
        f << "{\n  \"env\": {";
        bool first = true;
        for (const char* k : kLoadBearing) {
            const char* v = std::getenv(k);
            f << (first ? "\n" : ",\n") << "    \"" << k << "\": ";
            if (v) f << "\"" << v << "\""; else f << "null";
            first = false;
        }
        f << "\n  },\n  \"session\": {\"level_limit\": " << (fhe ? fhe->level_limit() : 0)
          << ", \"slots\": " << slots << ", \"logN\": " << logN << "}\n}\n";
    }

    bool load_bootstrap_plan_json(const std::string& path) {
        if (!fhe) {
            return false;
        }
        return fhe->load_bootstrap_plan_json(path);
    }

    void clear_bootstrap_plan() {
        if (fhe) {
            fhe->clear_bootstrap_plan();
        }
    }

    bool planned_bootstraps_enabled() const {
        return fhe && fhe->planned_bootstraps_enabled();
    }
};

struct InferenceOptions {
    CKKSContextOptions ckks{};

    // ModelSize
    int dim          = 768;
    int expanded     = 3072;
    int hidDim       = 1024;
    int expDim       = 4096;
    int numHeads     = 16;   // padded
    int numHeadsReal = 12;   // real
    int seqLen       = 1024;

    // Runtime flags
    bool parallel      = true;
    bool bench_mode    = false;

    PackingKind packing_kind = PackingKind::Cachemir;
    std::vector<PackingKind> aux_packing_kinds;
    InferenceMode mode       = InferenceMode::Threaded;  // full-inference execution strategy
};

inline WithStep::WithStep(Inference& inf, const std::string& s)
    : ctx_(inf.fhe.get()) { if (ctx_) ctx_->push_step(s); }

inline Inference make_inference(InferenceOptions o = {}) {
    Inference inf;

    inf.w.reserve(8192);
    inf.raw_w.reserve(8192);
    inf.cache.reserve(2048);
    inf.mask.reserve(2048);
    inf.enc_cache.reserve(65536);

    if (o.packing_kind == PackingKind::CachemirComplex)
        o.ckks.ckks_complex_payload = true;
    inf.fhe = make_ckks_context(o.ckks);
    inf.logN  = o.ckks.logN;
    inf.slots = (o.ckks.batch_size == 0)
                ? (1 << (o.ckks.logN - 1))
                : static_cast<int>(o.ckks.batch_size);
    inf.size.dim       = o.dim;
    inf.size.expanded  = o.expanded;
    inf.size.hidDim    = o.hidDim;
    inf.size.expDim    = o.expDim;
    inf.size.numHeads     = o.numHeads;
    inf.size.numHeadsReal = o.numHeadsReal;
    inf.size.seqLen       = o.seqLen;
    inf.parallel       = o.parallel;
    inf.bench_mode     = o.bench_mode;
    inf.mode           = o.mode;
    inf.complex        = (o.packing_kind == PackingKind::CachemirComplex);
    inf.packing        = inf.make_packing(inf.complex ? PackingKind::Cachemir : o.packing_kind);
    return inf;
}
