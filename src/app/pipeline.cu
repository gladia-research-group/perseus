#include "app/pipeline.h"

#include "cutmax.h"
#include "model/gpt2.h"          // parse_inference_mode
#include "model/gpt2_model.h"    // GPT2Model, Sequence
#include "config_loader.h"
#include "weight_loader.h"
#include "fideslib_wrapper.h"    // ckks_options_from_env, load_block_plans, BlockPlans
#include "io/json_utils.h"       // read_file_to_string

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace app {
namespace {

std::string env_or(const char* k, const std::string& d) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::string(v) : d;
}

bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

int argmax_index(const std::vector<double>& v) {
    if (v.empty()) return -1;
    int am = 0;
    for (int j = 1; j < static_cast<int>(v.size()); ++j)
        if (v[j] > v[am]) am = j;
    return am;
}

// --- CLASSIFY readout helpers (prefill-only tail; decode path never calls these) ---
std::vector<int> parse_int_list(const char* s) {
    std::vector<int> out;
    for (const char* p = s; p && *p;) {
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p) break;
        out.push_back(static_cast<int>(v));
        p = end;
        while (*p == ',' || *p == ' ') ++p;
    }
    return out;
}

void write_logits_bin(const char* path, const std::vector<double>& lg) {
    // full-vocab logits as float32 (compact) for post-hoc full-vocab KL / top-k, no FHE re-run.
    std::FILE* f = std::fopen(path, "wb");
    if (!f) return;
    const std::vector<float> buf(lg.begin(), lg.end());
    std::fwrite(buf.data(), sizeof(float), buf.size(), f);
    std::fclose(f);
}

std::vector<std::vector<double>> parse_2d_first(const std::string& text,
                                                const std::string& key, int rows) {
    size_t k = text.find("\"" + key + "\"");
    if (k == std::string::npos) throw std::runtime_error("missing field: " + key);
    size_t i = text.find('[', k);
    if (i == std::string::npos) throw std::runtime_error("malformed array: " + key);
    ++i;  // past outer '['
    const char* base = text.c_str();
    std::vector<std::vector<double>> out;
    while (static_cast<int>(out.size()) < rows) {
        while (i < text.size() && text[i] != '[' && text[i] != ']') ++i;
        if (i >= text.size() || text[i] == ']') break;   // end of outer array
        ++i;                                              // past inner '['
        std::vector<double> row;
        while (true) {
            while (i < text.size() &&
                   (text[i]==' '||text[i]==','||text[i]=='\n'||text[i]=='\t'||text[i]=='\r')) ++i;
            if (i >= text.size() || text[i] == ']') break;
            char* endp = nullptr;
            double v = std::strtod(base + i, &endp);
            if (endp == base + i) { ++i; continue; }      // not a number; skip
            row.push_back(v);
            i = static_cast<size_t>(endp - base);
        }
        ++i;                                              // past inner ']'
        out.push_back(std::move(row));
    }
    return out;
}


}  // namespace

RunConfig RunConfig::from_env() {
    RunConfig c;
    c.tokens  = std::stoi(env_or("MULTI_T", "1"));
    if (const char* p = std::getenv("PREFILL_TOKENS"); p && *p)
        c.prefill_tokens = std::stoi(p);
    if (const char* d = std::getenv("DECODE_TOKENS"); d && *d)
        c.decode_tokens = std::stoi(d);
    c.gen_prompt = std::stoi(env_or("GEN_PROMPT", "4"));
    c.gen_tokens = std::stoi(env_or("GEN_TOKENS", "4"));
    c.steps_t = std::stoi(env_or("STEPS_T", "16"));
    c.configs_path = env_or("CONFIGS_PATH",
        "/leonardo_work/IscrC_eff-SAM2/azirilli/he-aware-training/"
        "configs/model/approximation/hybrid/configs.json");
    c.weights_path = env_or("WEIGHTS_PATH",
        "/leonardo_work/IscrC_eff-SAM2/azirilli/he-aware-training/"
        "checkpoints/openai-community/gpt2/lm_eval/classic/weights.bin.zip");
    c.io_dir = env_or("ALL_BLOCKS_IO_DIR",
        "/leonardo/pub/userexternal/azirilli/he-aware-training_data/all_blocks_io");
    c.plan_dir  = env_or("FHE_BOOTSTRAP_PLACEMENTS_DIR", "");
    c.decode_plan_dir = env_or("FHE_DECODE_PLACEMENTS_DIR", "");
    c.graph_dir = env_or("FHE_GRAPH_DIR", "");
    c.mode = parse_inference_mode(env_or("GPT2_INFERENCE_MODE", "threaded"));
    c.cache_weights = env_or("GPT2_CACHE", "1") != "0";
    c.teacher_forced = env_or("TEACHER_FORCED", "0") != "0";   // generate: GT tokens vs argmax feedback
    return c;   // CUT_MAX removed 2026-07-05: encrypted CutMax always computed
}

std::vector<std::vector<double>> read_teacher_forced_inputs(const RunConfig& cfg) {
    const int rows = std::max(1, cfg.tokens);
    if (const char* cip = std::getenv("CLASSIFY_INPUT_JSON")) {
        auto inp = parse_2d_first(json_utils::read_file_to_string(cip), "inp", rows);
        if (static_cast<int>(inp.size()) < rows)
            throw std::runtime_error("CLASSIFY_INPUT_JSON has < requested rows: " +
                                     std::string(cip));
        return inp;
    }
    const int horizon = std::max(cfg.steps_t, rows);
    auto path_for = [&](int T) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s/all_blocks_L00_T%d.json",
                      cfg.io_dir.c_str(), T);
        return std::string(buf);
    };

    std::string path = path_for(horizon);
    std::string text;
    try {
        text = json_utils::read_file_to_string(path);
    } catch (const std::exception&) {
        if (horizon == rows) throw;
        path = path_for(rows);
        text = json_utils::read_file_to_string(path);
    }
    if (text.empty()) throw std::runtime_error("missing block-0 io: " + path);
    auto inp = parse_2d_first(text, "inp", rows);
    if (static_cast<int>(inp.size()) < rows)
        throw std::runtime_error("block-0 io has < requested rows: " + path);
    return inp;
}

GtSteps read_lm_head_steps(const RunConfig& cfg) {
    GtSteps out;
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/all_blocks_lm_head_steps_T%d.json",
                  cfg.io_dir.c_str(), cfg.steps_t);
    std::string text;
    try {
        text = json_utils::read_file_to_string(buf);
    } catch (const std::exception&) {
        return out;   // no oracle at this horizon -> "(no ground truth)"
    }
    const char* base = text.c_str();
    size_t pos = text.find("\"steps\"");
    if (pos == std::string::npos) return out;
    while (true) {
        size_t k = text.find("\"logits\"", pos);
        if (k == std::string::npos) break;
        size_t i = text.find('[', k);
        if (i == std::string::npos) break;
        ++i;
        std::vector<double> row;
        while (i < text.size() && text[i] != ']') {
            char* endp = nullptr;
            double v = std::strtod(base + i, &endp);
            if (endp == base + i) { ++i; continue; }
            row.push_back(v);
            i = static_cast<size_t>(endp - base);
        }
        out.logits.push_back(std::move(row));
        pos = i;
    }
    out.T = static_cast<int>(out.logits.size());
    return out;
}

struct DecodeSession::Impl {
    weight_loader::WeightStore   store;
    config_loader::ParsedConfigs parsed;
    BlockPlans                   plans;
    GPT2Model                    model;

    explicit Impl(const RunConfig& cfg)
        : store(ends_with(cfg.weights_path, ".zip")
                    ? weight_loader::WeightStore::from_zip(cfg.weights_path)
                    : weight_loader::WeightStore::from_dir(cfg.weights_path)),
          parsed(config_loader::parse_configs_json(
                     config_loader::read_file_to_string(cfg.configs_path))),
          // +1 MERGED tail stage (ln_f + lm_head). Empty plan dir => eager (BlockPlans{}).
          plans(cfg.plan_dir.empty()
                    ? BlockPlans{}
                    : load_block_plans(cfg.plan_dir, parsed.model.n_layers + 1)),
          model(GPT2Model::load(store, parsed, plans, cfg.mode,
                                cfg.cache_weights, ckks_options_from_env())) {}
};

DecodeSession::DecodeSession(const RunConfig& cfg)
    : cfg_(cfg), impl_(std::make_unique<Impl>(cfg)) {}
DecodeSession::~DecodeSession() = default;

RunResult DecodeSession::decode(const std::vector<std::vector<double>>& inputs) {
    GPT2Model& model = impl_->model;
    RunResult r;
    r.requested = cfg_.tokens;

    Sequence seq = model.start();
    model.generate_decode_masks(cfg_.tokens);

    const int W_tile = model.inference().slots;
    const bool measure_cutmax = std::getenv("FHE_GRAPH_DIR") == nullptr;   // off only under graph capture
    double total = 0.0, tok0 = 0.0, argmax_total = 0.0;

    auto& _prof = model.inference().fhe->profile;   // StepProfiler
    _prof.ensure_initialized();
    const auto _prof_mode = _prof.mode;
    const char* _pt = std::getenv("FHE_PROFILE_TOKEN");
    const int _prof_tok = (_pt && *_pt) ? std::atoi(_pt) : -1;

    for (int t = 0; t < cfg_.tokens; ++t) {
        const bool _prof_on = _prof_mode != StepProfiler::Mode::Off && (_prof_tok < 0 || t == _prof_tok);
        _prof.mode = _prof_on ? _prof_mode : StepProfiler::Mode::Off;
        if (_prof_on) _prof.reset();   // isolate THIS token's op composition (stack empty between tokens)
        const auto t0 = std::chrono::steady_clock::now();
        try {
            PackedCtx h = model.advance(seq, { inputs[t] });   // TEACHER-FORCED: GT token t
            auto tiles = model.logit_tiles(h);
            std::vector<double> lg = decode_lm_head_logits(     // == model.logits(h): decrypt+sync
                model.inference(), tiles, model.vocab(), W_tile);
            if (_prof_on) {   // pure-forward per-op table (advance+logits, BEFORE CutMax argmax stage)
                std::ostringstream _oss; _oss << "[proftok] t=" << t;
                _prof.dump(_oss);
                std::fputs(_oss.str().c_str(), stdout); std::fflush(stdout);
            }
            const int fhe_am = argmax_index(lg);
            const double dt = std::chrono::duration<double>(    // PURE decode wall (advance + logits)
                std::chrono::steady_clock::now() - t0).count();
            total += dt;
            if (t == 0) tok0 = dt;
            r.logits.push_back(lg);
            r.top1.push_back(fhe_am);
            r.positions.push_back(t);
            ++r.completed;

            if (measure_cutmax) {
                Inference& minf = model.inference();
                std::vector<PackedCtx> z; double argmax_s = 0.0;
                model.cutmax_step(tiles, /*position=*/-1, &z, &argmax_s);
                argmax_total += argmax_s;
                std::vector<double> zdec(model.vocab(), 0.0);
                if (z.size() == 1 && model.vocab() > W_tile) {
                    auto pt = decrypt_pt(minf.cc(), z[0].ct, minf.fhe->sk());
                    auto cv = pt->GetCKKSPackedValue();
                    for (int m = 0; m < W_tile; ++m) {
                        const int col = cutmax_tile_col_of_slot(m, minf.size.hidDim, W_tile);
                        zdec[col] = cv[m].real();
                        if (W_tile + col < model.vocab()) zdec[W_tile + col] = cv[m].imag();
                    }
                } else {
                    zdec = decode_lm_head_logits(minf, z, model.vocab(), W_tile);
                }
                const int cm_am = argmax_index(zdec);
                std::fprintf(stderr,
                    "[decode] pos=%d cutmax=%d fhe_argmax=%d %s z_mass=%.4f "
                    "decode=%.1fs argmax=%.1fs e2e=%.1fs\n",
                    t, cm_am, fhe_am, cm_am == fhe_am ? "OK" : "MISS",
                    zdec[cm_am], dt, argmax_s, dt + argmax_s);   // decode = forward only (e2e - argmax)
                std::fflush(stderr);
            }
        } catch (const std::exception& e) {
            r.threw = true;
            r.error = "token " + std::to_string(t) + ": " + e.what();
            break;
        }
    }

    r.avg_s_per_tok = (r.completed > 1)   // EXCLUDE tok0 (cold KV-arena alloc) from the average
        ? (total - tok0) / (r.completed - 1)
        : (r.completed ? total / r.completed : 0.0);
    r.avg_argmax_s    = r.completed ? argmax_total / r.completed : 0.0;   // decode cutmax stage s/tok
    r.bootstraps      = model.inference().fhe->total_bootstraps;
    r.unplanned_bts   = model.inference().fhe->unplanned_bootstrap_count;
    r.weight_relevels = model.inference().fhe->weight_relevel_count;
    return r;
}

RunResult run_decode(const RunConfig& cfg,
                     const std::vector<std::vector<double>>& inputs) {
    return DecodeSession(cfg).decode(inputs);   // byte-identical to the pre-refactor run_decode
}

RunResult run_generate(const RunConfig& cfg,
                       const std::vector<std::vector<double>>& inputs) {
    RunResult r;
    const int P = std::max(1, cfg.gen_prompt);
    const int M = std::max(1, cfg.gen_tokens);
    r.requested = M;

    if (static_cast<int>(inputs.size()) < P) {
        r.threw = true;
        r.error = "generate: prompt rows < GEN_PROMPT";
        return r;
    }
    if (cfg.teacher_forced && static_cast<int>(inputs.size()) < P + M) {
        r.threw = true;
        r.error = "generate --teacher-forced needs GEN_PROMPT+GEN_TOKENS GT rows (STEPS_T oracle "
                  "must cover them)";
        return r;
    }

    auto parsed = config_loader::parse_configs_json(
        config_loader::read_file_to_string(cfg.configs_path));

    weight_loader::WeightStore store = ends_with(cfg.weights_path, ".zip")
        ? weight_loader::WeightStore::from_zip(cfg.weights_path)
        : weight_loader::WeightStore::from_dir(cfg.weights_path);


    const bool use_prefill = P > 1;
    const int N_blocks = parsed.model.n_layers;

    BlockPlans plans = cfg.plan_dir.empty()
        ? BlockPlans{}
        : load_block_plans(cfg.plan_dir, N_blocks + 3);

    std::vector<BlockPlans> prefill_chunks;
    BlockPlans prefill_plans{};
    const std::string prefill_dir = env_or("FHE_PREFILL_PLACEMENTS_DIR", "");
    if (use_prefill && !prefill_dir.empty()) {
        if (std::filesystem::exists(prefill_dir + "/chunk_0")) {
            for (int c = 0;
                 std::filesystem::exists(prefill_dir + "/chunk_" + std::to_string(c)); ++c)
                prefill_chunks.push_back(load_block_plans(
                    prefill_dir + "/chunk_" + std::to_string(c), N_blocks + 1));
            prefill_plans = prefill_chunks.front();
        } else {
            prefill_plans = load_block_plans(prefill_dir, N_blocks + 1);
            prefill_chunks.push_back(prefill_plans);
        }
        std::fprintf(stderr, "[generate] planned prefill <- %s (%zu chunk%s)\n",
                     prefill_dir.c_str(), prefill_chunks.size(),
                     prefill_chunks.size() == 1 ? "" : "s");
        std::fflush(stderr);
    }

    GPT2Model model = use_prefill
        ? GPT2Model::load(store, parsed, prefill_plans, cfg.mode,
                          cfg.cache_weights, ckks_options_from_env(),
                          /*enable_prefill=*/true, &plans,
                          prefill_chunks.empty() ? nullptr : &prefill_chunks)
        : GPT2Model::load(store, parsed, plans, cfg.mode, cfg.cache_weights,
                          ckks_options_from_env());


    model.set_entry_bts(!cfg.teacher_forced ||
                        [] { const char* v = std::getenv("FHE_FORCE_ENTRY_BTS");
                             return v && *v && std::atoi(v) != 0; }());

    Sequence seq = model.start();
    model.generate_decode_masks(P + M);
    model.prepare_feedback_weights(   // CutMax always on -> encrypted feedback always used
        /*packed_z=*/model.inference().fhe->complex_payload);
    GtSteps gt = read_lm_head_steps(cfg);
    const int W_tile = model.inference().slots;

    PackedCtx h;
    double total = 0.0, tok0 = 0.0, argmax_total = 0.0;
    try {
        if (use_prefill)                          // prompt via prefill packing
            model.prefill(seq, std::vector<std::vector<double>>(
                                   inputs.begin(), inputs.begin() + (P - 1)));
        h = model.advance(seq, { inputs[P - 1] });   // last prompt token

        for (int j = 0; j < M; ++j) {
            const int pos = P - 1 + j;            // logits position
            const auto t0 = std::chrono::steady_clock::now();
            auto tiles = model.logit_tiles(h);

            // validation-only decrypt: plaintext argmax of the FHE logits
            std::vector<double> lg = decode_lm_head_logits(
                model.inference(), tiles, model.vocab(), W_tile);
            const int fhe_am = argmax_index(lg);
            // FHE_LOGIT_AUDIT=1: perpetrator-vs-victim discriminator for cutmax-side
            // explosions — are the logits already hot BEFORE cutmax runs?
            static const bool logit_audit = [] {
                const char* v = std::getenv("FHE_LOGIT_AUDIT");
                return v && *v && std::atoi(v) != 0;
            }();
            if (logit_audit) {
                double mx = 0.0, mn = 0.0;
                for (double x : lg) { mx = std::max(mx, x); mn = std::min(mn, x); }
                fprintf(stderr, "[logit_audit] pos=%d logit_max=%.6g logit_min=%.6g\n",
                        pos, mx, mn);
            }
            const int gt_am = (pos < gt.T && !gt.logits[pos].empty())
                ? argmax_index(gt.logits[pos]) : -1;
            const bool want_fb = (j + 1 < M);
            int top1 = fhe_am;
            double argmax_s = 0.0;

            {
                Inference& minf = model.inference();
                std::vector<PackedCtx> z;

                const bool need_fb = want_fb && !cfg.teacher_forced;
                PackedCtx emb = model.cutmax_step(tiles, need_fb ? pos + 1 : -1,
                                                  &z, &argmax_s);
                argmax_total += argmax_s;

                std::vector<double> zdec(model.vocab(), 0.0);
                if (z.size() == 1 && model.vocab() > W_tile) {
                    auto pt = decrypt_pt(minf.cc(), z[0].ct, minf.fhe->sk());
                    auto cv = pt->GetCKKSPackedValue();
                    for (int m = 0; m < W_tile; ++m) {
                        const int col =
                            cutmax_tile_col_of_slot(m, minf.size.hidDim, W_tile);
                        zdec[col] = cv[m].real();
                        if (W_tile + col < model.vocab())
                            zdec[W_tile + col] = cv[m].imag();
                    }
                } else {
                    zdec = decode_lm_head_logits(minf, z, model.vocab(), W_tile);
                }
                const int cm_am = argmax_index(zdec);

                double z_off_sum = 0.0, z_off_max = 0.0;
                int z_off_arg = -1;
                for (int m = 0; m < model.vocab(); ++m) {
                    if (m == cm_am) continue;
                    const double a = std::abs(zdec[m]);
                    z_off_sum += a;
                    if (a > z_off_max) { z_off_max = a; z_off_arg = m; }
                }

                std::fprintf(stderr,
                    "[generate] pos=%d cutmax=%d fhe_argmax=%d %s gt_argmax=%d "
                    "z_mass=%.4f z_off_sum=%.3e z_off_max=%.3e@%d argmax=%.1fs\n",
                    pos, cm_am, fhe_am, cm_am == fhe_am ? "OK" : "MISS",
                    gt_am, zdec[cm_am], z_off_sum, z_off_max, z_off_arg,
                    argmax_s);
                std::fflush(stderr);

                top1 = cm_am;
                if (want_fb)   // teacher-forced: GT token; autoregressive: the encrypted CutMax feedback
                    h = cfg.teacher_forced ? model.advance(seq, { inputs[pos + 1] })
                                           : model.advance_ct(seq, emb);
            }

            r.logits.push_back(std::move(lg));
            r.top1.push_back(top1);
            r.positions.push_back(pos);
            ++r.completed;
            const double dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            total += dt;
            if (j == 0) tok0 = dt;   // first generated token carries the cold-start cost
            std::fprintf(stderr,   // decode = forward+head only (e2e - argmax); e2e = full per-token wall
                "[gentime] pos=%d e2e=%.1fs decode=%.1fs argmax=%.1fs\n",
                pos, dt, dt - argmax_s, argmax_s);
            std::fflush(stderr);
        }
    } catch (const std::exception& e) {
        r.threw = true;
        r.error = "generate token " + std::to_string(r.completed) + ": " +
                  e.what();
    }

    r.avg_s_per_tok   = (r.completed > 1)   // EXCLUDE the first token (cold start) from the average
        ? (total - tok0) / (r.completed - 1)
        : (r.completed ? total / r.completed : 0.0);
    r.avg_argmax_s    = r.completed ? argmax_total / r.completed : 0.0;
    r.bootstraps      = model.inference().fhe->total_bootstraps;
    r.unplanned_bts   = model.inference().fhe->unplanned_bootstrap_count;
    r.weight_relevels = model.inference().fhe->weight_relevel_count;
    return r;
}

RunResult run_prefill(const RunConfig& cfg,
                      const std::vector<std::vector<double>>& inputs) {
    RunResult r;
    const int prefill_n = (cfg.prefill_tokens >= 0)
        ? cfg.prefill_tokens
        : std::max(0, cfg.tokens - 1);
    const int decode_n = (cfg.decode_tokens >= 0) ? cfg.decode_tokens : 1;
    const int total_n = prefill_n + decode_n;
    r.requested = total_n;

    if (prefill_n < 0 || decode_n < 0 || total_n < 1) {   // decode 0 = prefill-only (capture runs)
        r.threw = true;
        r.error = "prefill requires --prefill >= 0, --decode >= 0, total >= 1";
        return r;
    }
    if (static_cast<int>(inputs.size()) < total_n) {
        r.threw = true;
        r.error = "prefill input rows < requested tokens";
        return r;
    }

    auto parsed = config_loader::parse_configs_json(
        config_loader::read_file_to_string(cfg.configs_path));
    const int N_blocks = parsed.model.n_layers;

    weight_loader::WeightStore store = ends_with(cfg.weights_path, ".zip")
        ? weight_loader::WeightStore::from_zip(cfg.weights_path)
        : weight_loader::WeightStore::from_dir(cfg.weights_path);

    // Compositional prefill plans
    std::vector<BlockPlans> chunk_plans;
    BlockPlans plans;
    if (!cfg.plan_dir.empty()) {
        if (std::filesystem::exists(cfg.plan_dir + "/chunk_0")) {
            for (int c = 0; std::filesystem::exists(cfg.plan_dir + "/chunk_" + std::to_string(c)); ++c)
                chunk_plans.push_back(
                    load_block_plans(cfg.plan_dir + "/chunk_" + std::to_string(c), N_blocks + 1));
            plans = chunk_plans.front();
        } else {
            plans = load_block_plans(cfg.plan_dir, N_blocks + 1);
            chunk_plans.push_back(plans);
        }
    }

    std::vector<BlockPlans> decode_windows;
    BlockPlans decode_plans;
    if (!cfg.decode_plan_dir.empty()) {
        constexpr int kMaxWindows = 4;   // teacher-forced data ends at kc=128
        bool any_window = false;
        for (int w = 0; w < kMaxWindows; ++w)
            any_window |= std::filesystem::exists(cfg.decode_plan_dir + "/window_" + std::to_string(w));
        if (any_window) {
            int ref = -1;
            for (int w = 0; w < kMaxWindows; ++w) {
                const std::string d = cfg.decode_plan_dir + "/window_" + std::to_string(w);
                decode_windows.push_back(std::filesystem::exists(d)
                                             ? load_block_plans(d, N_blocks + 1)
                                             : BlockPlans{});
                if (decode_windows.back().any_valid() && ref < 0) ref = w;
            }
            for (size_t w = ref + 1; w < decode_windows.size(); ++w) {
                if (!decode_windows[w].any_valid()) continue;
                for (int b = 0; b <= N_blocks; ++b) {
                    const auto& a = decode_windows[ref].at(b);
                    const auto& c = decode_windows[w].at(b);
                    if (a.weight_levels != c.weight_levels || a.cache_pin_level != c.cache_pin_level)
                        throw std::runtime_error(
                            "decode window " + std::to_string(w) + " block " + std::to_string(b) +
                            ": weight_levels/cache_pin differ from window_" + std::to_string(ref) +
                            " (block weights are encoded once; windows must share pins)");
                }
            }
            if (ref >= 0) decode_plans = decode_windows[ref];
        } else {
            decode_plans = load_block_plans(cfg.decode_plan_dir, N_blocks + 1);
        }
    }
    std::fprintf(stderr, "[plan_phase] prefill=%s decode=%s%s\n",
                 cfg.plan_dir.empty() ? "(eager)" : cfg.plan_dir.c_str(),
                 cfg.decode_plan_dir.empty() ? "(eager)" : cfg.decode_plan_dir.c_str(),
                 decode_windows.empty() ? "" : (" (" + std::to_string(decode_windows.size()) + " windows)").c_str());

    GPT2Model model = GPT2Model::load(store, parsed, plans, cfg.mode,
                                      cfg.cache_weights, ckks_options_from_env(),
                                      /*enable_prefill=*/true, &decode_plans,
                                      chunk_plans.empty() ? nullptr : &chunk_plans,
                                      decode_windows.empty() ? nullptr : &decode_windows);

    Sequence seq = model.start();
    model.generate_decode_masks(total_n);

    double total = 0.0, decode_total = 0.0, first_decode = 0.0;
    double prefill_s = 0.0, tail_argmax_s = 0.0;   // per-phase wall for the end-of-run summary
    const auto run_t0 = std::chrono::steady_clock::now();
    try {
        PackedCtx tail_hidden;
        bool have_tail = false;
        if (prefill_n > 0) {
            std::vector<std::vector<double>> prompt(inputs.begin(),
                                                    inputs.begin() + prefill_n);
            const auto p_t0 = std::chrono::steady_clock::now();
            auto& _prof = model.inference().fhe->profile;
            _prof.ensure_initialized();
            if (_prof.on()) _prof.reset();
            tail_hidden = model.prefill(seq, prompt);   // batched prefill + filling->cachemir KV handoff
            if (_prof.on()) {
                std::ostringstream _oss; _oss << "[profpre] prefill_n=" << prefill_n;
                _prof.dump(_oss);
                std::fputs(_oss.str().c_str(), stdout); std::fflush(stdout);
            }
            prefill_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - p_t0).count();
            have_tail = true;
        }

        if (have_tail && decode_n == 0 && !std::getenv("FHE_GRAPH_DIR")) {
            const int W_tile = model.inference().slots;
            auto tiles = model.tail_logit_tiles(tail_hidden, prefill_n);
            std::vector<double> lg = decode_lm_head_logits(model.inference(), tiles,
                                                           model.vocab(), W_tile);
            int next_tok = argmax_index(lg);   // lm_head plaintext argmax (== full-vocab argmax of lg)

            if (const char* cont_start_s = std::getenv("CLASSIFY_CONT_START")) {
                const int cont_start = std::atoi(cont_start_s);
                const char* cont_ids_s = std::getenv("CLASSIFY_CONT_IDS");
                const std::vector<int> cont_ids =
                    cont_ids_s ? parse_int_list(cont_ids_s) : std::vector<int>{};
                double ll = 0.0;
                bool finite = !cont_ids.empty();
                std::string per;
                for (size_t j = 0; j < cont_ids.size(); ++j) {
                    const int ppos = cont_start - 1 + static_cast<int>(j);
                    auto tj = model.tail_logit_tiles_at(tail_hidden, prefill_n, ppos);
                    std::vector<double> lj =
                        decode_lm_head_logits(model.inference(), tj, model.vocab(), W_tile);
                    double mx = lj.empty() ? 0.0 : lj[0];
                    for (double v : lj) mx = std::max(mx, v);
                    double se = 0.0;
                    for (double v : lj) se += std::exp(v - mx);
                    const double lse = mx + std::log(se);
                    const int t = cont_ids[j];
                    const double lp = (t >= 0 && t < static_cast<int>(lj.size()))
                                          ? lj[t] - lse : std::nan("");
                    finite = finite && std::isfinite(lp);
                    ll += lp;
                    char b[48];
                    std::snprintf(b, sizeof(b), "%s%.6g", j ? "," : "", lp);
                    per += b;
                }
                std::fprintf(stderr,
                    "[classify_mc] cont_ll=%.8g ntok=%zu per_tok=%s finite=%d\n",
                    ll, cont_ids.size(), per.c_str(), finite ? 1 : 0);
                std::fflush(stderr);
            } else if (const char* classify_ids = std::getenv("CLASSIFY_CAND_IDS")) {
                const std::vector<int> ids = parse_int_list(classify_ids);
                double absmax = 0.0;
                bool finite = true;
                for (double v : lg) { absmax = std::max(absmax, std::fabs(v)); finite = finite && std::isfinite(v); }
                std::string cand;
                for (size_t k = 0; k < ids.size(); ++k) {
                    const bool ok = ids[k] >= 0 && ids[k] < static_cast<int>(lg.size());
                    char b[48];
                    std::snprintf(b, sizeof(b), "%s%.8g", k ? "," : "", ok ? lg[ids[k]] : std::nan(""));
                    cand += b;
                }
                std::fprintf(stderr,
                    "[classify] cand_logits=%s vocab_argmax=%d vocab_argmax_logit=%.8g absmax=%.6g finite=%d\n",
                    cand.c_str(), next_tok, (next_tok >= 0 ? lg[next_tok] : std::nan("")),
                    absmax, finite ? 1 : 0);
                std::fflush(stderr);
                if (const char* out = std::getenv("CLASSIFY_LOGITS_OUT")) write_logits_bin(out, lg);
            } else {
                Inference& minf = model.inference();
                std::vector<PackedCtx> z;
                double argmax_s = 0.0;
                model.cutmax_step(tiles, /*position=*/-1, &z, &argmax_s);   // argmax only
                std::vector<double> zdec(model.vocab(), 0.0);
                if (z.size() == 1 && model.vocab() > W_tile) {
                    auto pt = decrypt_pt(minf.cc(), z[0].ct, minf.fhe->sk());
                    auto cv = pt->GetCKKSPackedValue();
                    for (int m = 0; m < W_tile; ++m) {
                        const int col = cutmax_tile_col_of_slot(m, minf.size.hidDim, W_tile);
                        zdec[col] = cv[m].real();
                        if (W_tile + col < model.vocab()) zdec[W_tile + col] = cv[m].imag();
                    }
                } else {
                    zdec = decode_lm_head_logits(minf, z, model.vocab(), W_tile);
                }
                const int cm_am = argmax_index(zdec);
                tail_argmax_s = argmax_s;
                std::fprintf(stderr,
                    "[prefill] next-token cutmax=%d lm_head_argmax=%d %s argmax=%.1fs\n",
                    cm_am, next_tok, cm_am == next_tok ? "OK" : "MISS", argmax_s);
                std::fflush(stderr);
                next_tok = cm_am;
            }

            if (std::getenv("PREFILL_ARGMAX_SCAN")) {
                const auto& chunks = model.prefill_lnf_chunks();
                Inference& minf = model.inference();
                const int t_cap = minf.slots / minf.size.hidDim;
                const int cap = (minf.fhe->complex_payload && prefill_n > t_cap) ? 2 * t_cap
                                                                                 : t_cap;
                const char* st = std::getenv("PREFILL_ARGMAX_SCAN_STRIDE");
                const int stride = (st && atoi(st) > 0) ? atoi(st) : 1;
                for (int pos = 0; pos + 1 < prefill_n; pos += stride) {
                    const size_t c = static_cast<size_t>(pos / cap);
                    if (c >= chunks.size()) break;
                    auto tj = model.tail_logit_tiles_at(chunks[c], prefill_n, pos);
                    r.logits.push_back(decode_lm_head_logits(minf, tj, model.vocab(), W_tile));
                    r.top1.push_back(argmax_index(r.logits.back()));
                    r.positions.push_back(pos);
                    std::fprintf(stderr, "[prefill_scan] pos=%d/%d top1=%d\n",
                                 pos, prefill_n - 1, r.top1.back());
                    std::fflush(stderr);
                }
            }
            r.logits.push_back(lg);
            r.top1.push_back(next_tok);
            r.positions.push_back(prefill_n - 1);
            r.completed = seq.abs_pos;
        }

        for (int j = 0; j < decode_n; ++j) {
            const int pos = prefill_n + j;
            const auto t0 = std::chrono::steady_clock::now();
            PackedCtx h = model.advance(seq, { inputs[pos] });
            std::vector<double> lg = model.logits(h);
            const double dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            if (j == 0) first_decode = dt;
            decode_total += dt;
            r.logits.push_back(std::move(lg));
            r.top1.push_back(argmax_index(r.logits.back()));
            r.positions.push_back(pos);
            r.completed = seq.abs_pos;
        }
    } catch (const std::exception& e) {
        r.threw = true;
        r.error = "prefill/decode: " + std::string(e.what());
        r.completed = seq.abs_pos;
    }

    total = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - run_t0).count();
    r.avg_s_per_tok    = (r.logits.size() > 1)   // decode-phase avg, first decode token (cold) EXCLUDED
        ? (decode_total - first_decode) / (static_cast<int>(r.logits.size()) - 1)
        : (r.logits.empty() ? (r.completed ? total / r.completed : 0.0)
                            : decode_total / r.logits.size());

    // Clear end-of-run PHASE breakdown: prefill(+handoff) | tail argmax | decode | whole run.
    char tail_buf[64] = "";
    if (tail_argmax_s > 0.0)
        std::snprintf(tail_buf, sizeof(tail_buf), " | tail_argmax=%.1fs", tail_argmax_s);
    std::fprintf(stderr,
        "[prefill_timing] prefill+handoff=%.1fs (%d tok)%s"
        " | decode=%d tok: tok0=%.1fs steady=%.1fs/tok total=%.1fs | WHOLE RUN=%.1fs\n",
        prefill_s, prefill_n, tail_buf,
        decode_n, first_decode, r.avg_s_per_tok, decode_total, total);
    std::fflush(stderr);

    r.bootstraps       = model.inference().fhe->total_bootstraps;
    r.unplanned_bts    = model.inference().fhe->unplanned_bootstrap_count;
    r.weight_relevels  = model.inference().fhe->weight_relevel_count;
    return r;
}

}  // namespace app
