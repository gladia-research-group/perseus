#include "client_context.h"
#include "fhe_errors.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <unistd.h>

namespace perseus_client {

CKKSContextOptions ckks_options_from_env() {
    CKKSContextOptions o{};
    auto env_int = [](const char* name, auto dflt) {
        const char* e = std::getenv(name);
        return (e && *e) ? static_cast<decltype(dflt)>(std::stol(e)) : dflt;
    };
    o.logN               = env_int("LOGN",               o.logN);
    o.depth              = env_int("CKKS_DEPTH",         o.depth);
    o.first_mod_bits     = env_int("FIRST_MOD_BITS",     o.first_mod_bits);
    o.btp_scale_bits     = env_int("BTP_SCALE_BITS",     o.btp_scale_bits);
    o.scale_bits         = env_int("SCALE_BITS",         o.scale_bits);
    o.btp_depth_overhead = env_int("BTP_DEPTH_OVERHEAD", o.btp_depth_overhead);
    o.h_weight           = env_int("H_WEIGHT",           o.h_weight);
    o.num_large_digits   = env_int("NUM_LARGE_DIGITS",   o.num_large_digits);
    o.auto_bts_level_override = env_int("AUTO_BTS_LEVEL", o.auto_bts_level_override);
    o.correction_factor  = env_int("CORRECTION_FACTOR",  o.correction_factor);
    o.composite_degree   = env_int("COMPOSITE_DEGREE",   o.composite_degree);
    o.bts_iterations     = env_int("BTS_ITERATIONS",     o.bts_iterations);
    o.bts_precision      = env_int("BTS_PRECISION",      o.bts_precision);

    if (const char* sbs = std::getenv("SPARSE_BTS_SLOTS"); sbs && *sbs) {
        std::stringstream ss{std::string(sbs)};
        std::string tok;
        while (std::getline(ss, tok, ','))
            if (!tok.empty() && std::stoul(tok) > 0)
                o.sparse_bts_slots_list.push_back(std::stoul(tok));
    }

    if (const char* slb = std::getenv("SPARSE_LEVEL_BUDGET"); slb && *slb) {
        std::vector<uint32_t> budget;
        std::string s(slb);
        for (auto& ch : s) if (ch == ':') ch = ',';
        std::stringstream ss(s);
        std::string tok;
        while (std::getline(ss, tok, ',')) if (!tok.empty()) budget.push_back(std::stoul(tok));
        if (budget.size() >= 2) o.sparse_level_budget = budget;   // else: ignored, silently
    }

    if (const char* lb = std::getenv("LEVEL_BUDGET"); lb && *lb) {
        std::vector<uint32_t> budget;
        std::string s(lb);
        for (auto& ch : s) if (ch == ':') ch = ',';
        std::stringstream ss(s);
        std::string tok;
        while (std::getline(ss, tok, ',')) if (!tok.empty()) budget.push_back(std::stoul(tok));
        if (budget.size() >= 2) o.level_budget = budget;          // else: ignored, silently
    }

    if (const char* cx = std::getenv("CKKS_COMPLEX"); cx && cx[0] == '1')
        o.ckks_complex_payload = true;

    return o;
}

uint32_t slots_of(const CKKSContextOptions& o) {
    return (o.batch_size == 0) ? (1u << (o.logN - 1)) : o.batch_size;
}

namespace {

constexpr bool kPrecompute = false;

struct Derived {
    std::vector<uint32_t> level_budget;
    int actual_scale_bits = 0;
    uint32_t btp_depth_overhead = 0;
};

Derived derive(const CKKSContextOptions& o) {
    Derived d;
    d.btp_depth_overhead = o.btp_depth_overhead;
    if (o.enable_bootstrap) {
        d.actual_scale_bits = o.btp_scale_bits;
        d.level_budget      = o.level_budget.empty() ? std::vector<uint32_t>{3, 3} : o.level_budget;
    } else {
        d.actual_scale_bits  = o.scale_bits;
        d.level_budget       = {};
        d.btp_depth_overhead = 0;
    }
    return d;
}

// The 13 setters of make_ckks_context, in its order, on lbcrypto's CCParams.
LbCC gen_context(const CKKSContextOptions& o, const Derived& d, uint32_t slots) {
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(o.depth + d.btp_depth_overhead);
    params.SetScalingModSize(d.actual_scale_bits);
    if (o.ckks_complex_payload)
        params.SetCKKSDataType(lbcrypto::COMPLEX);
    params.SetFirstModSize(o.first_mod_bits);
    if (o.composite_degree > 1) {
        params.SetScalingTechnique(lbcrypto::COMPOSITESCALINGMANUAL);
        params.SetCompositeDegree(o.composite_degree);
        params.SetRegisterWordSize(32);
    } else {
        params.SetScalingTechnique(lbcrypto::FLEXIBLEAUTO);
    }
    params.SetBatchSize(slots);
    params.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
    params.SetNumLargeDigits(o.num_large_digits);
    params.SetKeySwitchTechnique(lbcrypto::HYBRID);
    params.SetSecurityLevel(lbcrypto::HEStd_128_classic);
    params.SetRingDim(1 << o.logN);

    LbCC cc = lbcrypto::GenCryptoContext(params);
    if (auto fhe = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE))
        fhe->m_bootPrecomMap.clear();
    return cc;
}

void enable_features(const LbCC& cc, bool bootstrap) {
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    if (bootstrap) {
        cc->Enable(lbcrypto::ADVANCEDSHE);
        cc->Enable(lbcrypto::FHE);
    }
}

std::vector<uint32_t> bootstrap_setups(const LbCC& cc, const CKKSContextOptions& o, const Derived& d,
                                       uint32_t slots, const LbSK* sk) {
    std::vector<uint32_t> built_slots;
    if (!o.enable_bootstrap) return built_slots;
    const uint32_t btp_slots = (o.bootstrap_slots == 0) ? slots : o.bootstrap_slots;
    cc->EvalBootstrapSetup(d.level_budget, {0, 0}, btp_slots, o.correction_factor, kPrecompute);
    if (sk) gen_bootstrap_keys(cc, *sk, static_cast<int>(btp_slots));
    built_slots.push_back(btp_slots);
    std::set<uint32_t> built;
    const auto& slb = o.sparse_level_budget.empty() ? d.level_budget : o.sparse_level_budget;
    for (uint32_t s : o.sparse_bts_slots_list) {
        if (s == 0 || s >= slots || !built.insert(s).second) continue;
        cc->EvalBootstrapSetup(slb, {0, 0}, s, o.correction_factor, kPrecompute);
        if (sk) gen_bootstrap_keys(cc, *sk, static_cast<int>(s));
        built_slots.push_back(s);
    }
    return built_slots;
}

std::vector<int32_t> sorted_unique(std::vector<int32_t> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

// The sidecar FIDESlib's DeserializeFromFile requires next to context.bin.
void parse_sidecar(const std::string& path, ClientContext& ctx) {
    std::ifstream dev(path, std::ios::binary);
    if (!dev)
        throw fhe::FHEError("make_client_context: missing device sidecar '" + path +
                            "' (a bundle written by save_keys carries it)");
    std::string line;
    ctx.loaded_rot_steps.clear();
    while (std::getline(dev, line)) {
        std::istringstream iss(line);
        std::string label;
        iss >> label;
        if (label == "RotationIndexes:") {
            char brace;
            iss >> brace;
            int index;
            while (iss >> index) ctx.loaded_rot_steps.push_back(index);
        } else if (label == "KeyDist:") {
            int dist = 0;
            iss >> dist;
            ctx.key_dist = dist;
        }
    }
}

}  // namespace

std::shared_ptr<ClientContext> make_client_context(const CKKSContextOptions& o) {
    const Derived d = derive(o);
    const uint32_t slots = slots_of(o);
    auto ctx = std::make_shared<ClientContext>();
    if (!o.keys_dir.empty()) {
        // A bundle written by save_keys (either extension): the context carries the exact
        // modulus chain, the public key encrypts, the secret key is loaded separately.
        LbCC cc;
        if (!lbcrypto::Serial::DeserializeFromFile(o.keys_dir + "/context.bin", cc,
                                                   lbcrypto::SerType::BINARY) || !cc)
            throw fhe::FHEError("make_client_context: cannot deserialize " + o.keys_dir +
                                "/context.bin");
        parse_sidecar(o.keys_dir + "/context.bin.dev", *ctx);
        enable_features(cc, o.enable_bootstrap);
        if (!lbcrypto::Serial::DeserializeFromFile(o.keys_dir + "/public.key", ctx->kp.publicKey,
                                                   lbcrypto::SerType::BINARY) || !ctx->kp.publicKey)
            throw fhe::FHEError("make_client_context: cannot deserialize " + o.keys_dir +
                                "/public.key");
        ctx->cc = cc;
        ctx->from_keys = true;
    } else {
        LbCC cc = gen_context(o, d, slots);
        enable_features(cc, o.enable_bootstrap);
        ctx->cc = cc;
        ctx->kp = cc->KeyGen();
        cc->EvalMultKeyGen(ctx->kp.secretKey);
        const std::vector<int32_t> rot_steps = sorted_unique(o.extra_rot_steps);
        if (!rot_steps.empty()) cc->EvalRotateKeyGen(ctx->kp.secretKey, rot_steps);
        ctx->loaded_rot_steps = rot_steps;
        bootstrap_setups(cc, o, d, slots, &ctx->kp.secretKey);
        ctx->key_dist = o.h_weight > 0 ? 3 : 1;
    }
    ctx->slots            = slots;
    ctx->total_depth      = static_cast<uint32_t>(o.depth) + d.btp_depth_overhead;
    ctx->btp_overhead     = d.btp_depth_overhead;
    ctx->composite_degree = static_cast<int>(o.composite_degree > 0 ? o.composite_degree : 1);
    ctx->bts_iterations   = o.bts_iterations;
    ctx->complex_payload  = o.ckks_complex_payload;
    return ctx;
}

std::pair<std::string, std::string> bundle_meta(const CKKSContextOptions& o,
                                                const std::vector<int32_t>& band) {
    CKKSContextOptions opts = o;
    opts.keys_dir.clear();
    for (int32_t r : band) opts.extra_rot_steps.push_back(r);
    const Derived d = derive(opts);
    const uint32_t slots = slots_of(opts);
    LbCC cc = gen_context(opts, d, slots);
    enable_features(cc, opts.enable_bootstrap);
    bootstrap_setups(cc, opts, d, slots, nullptr);
    std::stringstream ss;
    lbcrypto::Serial::Serialize(cc, ss, lbcrypto::SerType::BINARY);
    ClientContext meta;
    meta.loaded_rot_steps = sorted_unique(opts.extra_rot_steps);
    meta.key_dist = opts.h_weight > 0 ? 3 : 1;
    return {ss.str(), dev_sidecar_text(meta)};
}

std::vector<uint32_t> expected_automorphism_indexes_for(const CKKSContextOptions& o,
                                                        const std::vector<int32_t>& band) {
    CKKSContextOptions opts = o;
    opts.keys_dir.clear();
    for (int32_t r : band) opts.extra_rot_steps.push_back(r);
    const Derived d = derive(opts);
    const uint32_t slots = slots_of(opts);
    LbCC cc = gen_context(opts, d, slots);
    enable_features(cc, opts.enable_bootstrap);
    const auto built = bootstrap_setups(cc, opts, d, slots, nullptr);
    return expected_automorphism_indexes(cc, sorted_unique(opts.extra_rot_steps), built);
}

std::vector<int> bootstrap_indexes_for(const CKKSContextOptions& o, int slots) {
    CKKSContextOptions opts = o;
    opts.keys_dir.clear();
    const Derived d = derive(opts);
    const uint32_t all_slots = slots_of(opts);
    LbCC cc = gen_context(opts, d, all_slots);
    enable_features(cc, opts.enable_bootstrap);
    const auto built = bootstrap_setups(cc, opts, d, all_slots, nullptr);
    if (built.empty()) throw fhe::FHEError("bootstrap_indexes: bootstrapping is disabled in these options");
    return bootstrap_indexes(cc, slots == 0 ? static_cast<int>(built.front()) : slots);
}

std::string dev_sidecar_text(const ClientContext& ctx) {
    std::string s = "1 { 0 }\n";
    s += "AutoLoadCiphertexts: 0\n";
    s += "AutoLoadPlaintexts: 0\n";
    s += "RotationIndexes: { ";
    for (int32_t idx : ctx.loaded_rot_steps) s += std::to_string(idx) + " ";
    s += "}\n";
    s += "KeyDist: " + std::to_string(ctx.key_dist) + "\n";
    return s;
}

void save_keys(const ClientContext& ctx, const std::string& dir) {
    using CCI = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>;
    namespace fs = std::filesystem;
    if (!ctx.cc || !ctx.kp.publicKey)
        throw fhe::FHEError("save_keys: the session has no context / public key");
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir))
        throw fhe::FHEError("save_keys: cannot use bundle directory '" + dir + "': " +
                            (ec ? ec.message() : std::string("not a directory")));
    auto fail = [](const char* what, const std::string& path) {
        const int e = errno;
        std::string msg = std::string("save_keys: ") + what + " serialization failed for '" +
                          path + "'";
        if (e) msg += std::string(" (") + std::strerror(e) + ")";
        throw fhe::FHEError(msg);
    };
    const std::string tmp = ".partial-" + std::to_string(::getpid()) + "-";
    auto commit = [&](const std::string& name, const char* what) {
        std::error_code rc;
        fs::rename(dir + "/" + tmp + name, dir + "/" + name, rc);
        if (rc) throw fhe::FHEError(std::string("save_keys: cannot rename ") + what + " into '" +
                                    dir + "/" + name + "': " + rc.message());
    };
    errno = 0;
    if (!lbcrypto::Serial::SerializeToFile(dir + "/" + tmp + "context.bin", ctx.cc,
                                           lbcrypto::SerType::BINARY))
        fail("context", dir + "/" + tmp + "context.bin");
    {   // the device sidecar FIDESlib writes next to the context file under the same base name
        std::ofstream dev(dir + "/" + tmp + "context.bin.dev", std::ios::binary);
        const std::string text = dev_sidecar_text(ctx);
        if (!dev || !dev.write(text.data(), static_cast<std::streamsize>(text.size())))
            fail("context sidecar", dir + "/" + tmp + "context.bin.dev");
    }
    commit("context.bin", "context");
    commit("context.bin.dev", "context sidecar");
    errno = 0;
    if (!lbcrypto::Serial::SerializeToFile(dir + "/" + tmp + "public.key", ctx.kp.publicKey,
                                           lbcrypto::SerType::BINARY))
        fail("public key", dir + "/" + tmp + "public.key");
    commit("public.key", "public key");
    errno = 0;
    {
        std::ofstream fm(dir + "/" + tmp + "multkeys.bin", std::ios::binary);
        if (!fm || !CCI::SerializeEvalMultKey(fm, lbcrypto::SerType::BINARY, ""))
            fail("eval-mult key", dir + "/" + tmp + "multkeys.bin");
    }
    commit("multkeys.bin", "eval-mult key");
    errno = 0;
    {
        std::ofstream fr(dir + "/" + tmp + "rotkeys.bin", std::ios::binary);
        if (!fr || !CCI::SerializeEvalAutomorphismKey(fr, lbcrypto::SerType::BINARY, ""))
            fail("rotation key", dir + "/" + tmp + "rotkeys.bin");
    }
    commit("rotkeys.bin", "rotation key");
}

void save_secret_key(const ClientContext& ctx, const std::string& path) {
    if (!ctx.kp.secretKey) throw fhe::FHEError("save_secret_key: this session holds no secret key");
    if (!lbcrypto::Serial::SerializeToFile(path, ctx.kp.secretKey, lbcrypto::SerType::BINARY))
        throw fhe::FHEError("save_secret_key: serialization failed");
}

void load_secret_key(ClientContext& ctx, const std::string& path) {
    LbSK sk;
    if (!lbcrypto::Serial::DeserializeFromFile(path, sk, lbcrypto::SerType::BINARY) || !sk)
        throw fhe::FHEError("load_secret_key: cannot deserialize '" + path + "'");
    if (ctx.kp.publicKey && sk->GetKeyTag() != ctx.kp.publicKey->GetKeyTag())
        throw fhe::FHEError("load_secret_key: '" + path + "' belongs to key tag " + sk->GetKeyTag() +
                            ", this session's public key is " + ctx.kp.publicKey->GetKeyTag());
    ctx.kp.secretKey = sk;
}

}  // namespace perseus_client
