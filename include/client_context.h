#pragma once

#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include "config_core.h"

#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#ifndef PERSEUS_VERSION
#define PERSEUS_VERSION "0.1.0"
#endif

namespace perseus_client {

using LbCC  = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
using LbCt  = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
using LbKP  = lbcrypto::KeyPair<lbcrypto::DCRTPoly>;
using LbSK  = lbcrypto::PrivateKey<lbcrypto::DCRTPoly>;
using LbPK  = lbcrypto::PublicKey<lbcrypto::DCRTPoly>;
using LbEvalKeyMap = std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>;

// Chain defaults, mirrored from include/fideslib_wrapper.h (tests/test_client_layout.py checks
// the two option structs stay field-for-field identical).
namespace chain_defaults {
#if NATIVEINT == 32
inline constexpr int      depth              = 10;
inline constexpr int      scale_bits         = 54;
inline constexpr uint32_t composite_degree   = 2;
inline constexpr int      btp_scale_bits     = 54;
inline constexpr uint32_t correction_factor  = 6;
inline constexpr int      first_mod_bits     = 56;
inline constexpr uint32_t num_large_digits   = 6;
inline constexpr uint32_t auto_bts_level     = 46;
#else
inline constexpr int      depth              = 11;
inline constexpr int      scale_bits         = 58;
inline constexpr uint32_t composite_degree   = 1;
inline constexpr int      btp_scale_bits     = 53;
inline constexpr uint32_t correction_factor  = 0;
inline constexpr int      first_mod_bits     = 60;
inline constexpr uint32_t num_large_digits   = 7;
inline constexpr uint32_t auto_bts_level     = 24;
#endif
}  // namespace chain_defaults

struct CKKSContextOptions {
    std::string keys_dir;
    bool skip_gpu_load = false;
    int logN       = 16;
    int depth      = chain_defaults::depth;
    int scale_bits = chain_defaults::scale_bits;

    bool     enable_bootstrap   = true;
    uint32_t btp_depth_overhead = 16;
    std::vector<uint32_t> level_budget = {4, 3};
    uint32_t bootstrap_slots    = 0;
    std::vector<uint32_t> sparse_bts_slots_list = {};
    std::vector<uint32_t> sparse_level_budget = {};

    uint32_t composite_degree   = chain_defaults::composite_degree;

    int      btp_scale_bits     = chain_defaults::btp_scale_bits;
    uint32_t correction_factor  = chain_defaults::correction_factor;
    int      first_mod_bits     = chain_defaults::first_mod_bits;
    uint32_t num_large_digits   = chain_defaults::num_large_digits;
    uint32_t auto_bts_level_override = chain_defaults::auto_bts_level;

    uint32_t batch_size     = 0;
    bool ckks_complex_payload = false;

    int      h_weight         = 192;

    std::vector<int32_t> extra_rot_steps = {};
    std::vector<int32_t> deferred_rot_steps = {};

    uint32_t bts_iterations          = 1;
    uint32_t bts_precision           = 12;

    bool defer_heavy_setup = false;
};

enum class PackingKind {
    Cachemir,
    Diagonal,
    CachemirFilling,
    CachemirComplex,
};

inline const char* to_string(PackingKind k) {
    switch (k) {
        case PackingKind::Cachemir:        return "cachemir";
        case PackingKind::Diagonal:        return "diagonal";
        case PackingKind::CachemirFilling: return "cachemir_filling";
        case PackingKind::CachemirComplex: return "cachemir_complex";
    }
    return "unknown";
}

inline bool is_cachemir(PackingKind k) {
    return k == PackingKind::Cachemir || k == PackingKind::CachemirComplex;
}
inline bool is_diagonal(PackingKind k) { return k == PackingKind::Diagonal; }
inline bool is_cachemir_filling(PackingKind k) { return k == PackingKind::CachemirFilling; }

struct ModelSize {
    int dim          = 768;
    int expanded     = 3072;
    int hidDim       = 1024;
    int expDim       = 4096;
    int numHeads     = 16;
    int numHeadsReal = 12;
    int seqLen       = 1024;

    int getRealHidDim()   const { return dim; }
    int getRealFfDim()    const { return expanded; }
    int getRealNumHeads() const { return numHeadsReal; }
    int getRealDHead()    const { return dim / numHeadsReal; }
};

enum class InferenceMode { Sync, Threaded, Prefetch };

struct InferenceOptions {
    CKKSContextOptions ckks{};

    int dim          = 768;
    int expanded     = 3072;
    int hidDim       = 1024;
    int expDim       = 4096;
    int numHeads     = 16;
    int numHeadsReal = 12;
    int seqLen       = 1024;

    bool parallel      = true;
    bool bench_mode    = false;

    PackingKind packing_kind = PackingKind::Cachemir;
    std::vector<PackingKind> aux_packing_kinds;
    InferenceMode mode       = InferenceMode::Threaded;
};

struct ClientContext {
    LbCC cc;
    LbKP kp;
    std::vector<int32_t> loaded_rot_steps;
    uint32_t slots        = 0;
    uint32_t total_depth  = 0;
    uint32_t btp_overhead = 0;
    uint32_t bts_iterations = 1;
    int      composite_degree = 1;
    int      key_dist     = 1;
    bool     complex_payload = false;
    bool     from_keys    = false;

    uint32_t bootstrap_output_level() const {
        return static_cast<uint32_t>(composite_degree) *
               (btp_overhead + (bts_iterations >= 2 ? 1u : 0u));
    }
    bool has_secret_key() const { return static_cast<bool>(kp.secretKey); }
};

struct ClientInference {
    std::shared_ptr<ClientContext> fhe;
    ModelSize size;
    int slots = 0;
    int logN  = 0;
    int n_tok = 0;
    PackingKind packing = PackingKind::Cachemir;
    bool complex = false;
    InferenceMode mode = InferenceMode::Threaded;
};

struct ClientCt {
    LbCt ct;
    PackingKind packing = PackingKind::Cachemir;
};

CKKSContextOptions ckks_options_from_env();
std::shared_ptr<ClientContext> make_client_context(const CKKSContextOptions& o);
std::string dev_sidecar_text(const ClientContext& ctx);
void save_keys(const ClientContext& ctx, const std::string& dir);
void save_secret_key(const ClientContext& ctx, const std::string& path);
void load_secret_key(ClientContext& ctx, const std::string& path);
std::pair<std::string, std::string> bundle_meta(const CKKSContextOptions& o,
                                                const std::vector<int32_t>& band);
std::vector<uint32_t> expected_automorphism_indexes_for(const CKKSContextOptions& o,
                                                        const std::vector<int32_t>& band);
std::vector<int> bootstrap_indexes_for(const CKKSContextOptions& o, int slots);
uint32_t slots_of(const CKKSContextOptions& o);

std::vector<int> bootstrap_indexes(const LbCC& cc, int slots);
std::vector<int> accumulate_rotation_indices(int bStep, int stride, int size);
std::shared_ptr<LbEvalKeyMap> gen_rotation_keys(const LbSK& sk, const std::vector<int>& indexes);
void gen_bootstrap_keys(const LbCC& cc, const LbSK& sk, int slots);
std::vector<uint32_t> expected_automorphism_indexes(const LbCC& cc, const std::vector<int32_t>& band,
                                                    const std::vector<uint32_t>& slot_counts);

std::map<std::string, std::vector<uint32_t>> automorphism_indexes_in_file(const std::string& path);
std::vector<uint32_t> automorphism_key_indexes(const ClientContext& ctx);

namespace cachemir {
std::vector<int32_t> compute_gpt2_rot_indices(int slots, int hidDim, int ffDim, int numHeads);
}
namespace diagonal {
std::vector<int32_t> compute_gpt2_rot_indices(int slots, int hidDim, int ffDim, int numHeads);
}
namespace cachemir_filling {
std::vector<int32_t> compute_gpt2_rot_indices(int slots, int hidDim, int ffDim, int numHeads);
}
std::vector<int32_t> compute_gpt2_rot_indices(PackingKind kind, int slots, int hidDim, int ffDim,
                                              int numHeads);

std::vector<int32_t> family_rot_steps(const std::string& family, const InferenceOptions& o);

std::vector<int32_t> family_rot_band(const std::string& family, const InferenceOptions& o);

InferenceOptions prepare_family_options(const std::string& family, InferenceOptions o);

ClientInference make_inference(InferenceOptions o);
ClientInference make_gpt2_inference(InferenceOptions o);

ClientCt pack_tokens(ClientInference& inf, const std::vector<std::vector<double>>& embeddings,
                     int target_level);
std::vector<std::vector<double>> unpack_tokens(const ClientInference& inf, const ClientCt& pc, int T);
ClientCt encode_token_input(ClientInference& inf, const std::vector<double>& x_real);
std::vector<double> decode_token_output(const ClientInference& inf, const ClientCt& pc);
std::vector<std::vector<double>> decode_tokens_output(const ClientInference& inf, const ClientCt& pc,
                                                      int T);
std::vector<double> decrypt(const ClientInference& inf, const ClientCt& pc);
std::vector<std::complex<double>> decrypt_complex(const ClientInference& inf, const ClientCt& pc);
std::vector<double> decode_linear_output(PackingKind kind, const std::vector<double>& cy,
                                         int slots, int d_in, int d_out);
int lm_head_tile_width(const ClientInference& inf, int vocab);
std::vector<double> decode_lm_head_logits(const ClientInference& inf, const std::vector<ClientCt>& tiles,
                                          int vocab);
std::string serialize_ct(const ClientCt& pc);
ClientCt deserialize_ct(const ClientInference& inf, const std::string& data);

namespace stamp {
#if defined(NATIVEINT)
static constexpr int kNativeIntBits = NATIVEINT;
#else
static constexpr int kNativeIntBits = 0;
#endif
static constexpr const char* kChain =
    kNativeIntBits == 32 ? "n32" : (kNativeIntBits == 64 ? "n64" : "unknown");
}  // namespace stamp

}  // namespace perseus_client
