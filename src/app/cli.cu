#include "app/cli.h"
#include "app/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace app {
namespace {

void usage() {
    std::cerr <<
      "usage: cuda_cachemir <capture|decode|prefill|generate> [flags]\n"
      "  capture   trace the op-graph at token 0 (writes FHE_GRAPH_DIR); tokens forced to 1\n"
      "  decode    run a planned decode (requires FHE_BOOTSTRAP_PLACEMENTS_DIR, or --eager)\n"
      "  prefill   run prefill->decode handoff, then decode K teacher-forced tokens\n"
      "  generate  generation with encrypted CutMax argmax (ALWAYS on): prompt (prefill packing\n"
      "            when >1) -> per-token CutMax; --teacher-forced feeds the GT token next,\n"
      "            else the encrypted CutMax argmax is fed back (autoregressive)\n"
      "\n"
      "  flags (override the matching env var):\n"
      "    --tokens N        MULTI_T\n"
      "    --prefill N       PREFILL_TOKENS (prefill subcommand)\n"
      "    --decode N        DECODE_TOKENS (prefill subcommand; decode alias for --tokens)\n"
      "    --steps-t N       STEPS_T\n"
      "    --plan DIR        FHE_BOOTSTRAP_PLACEMENTS_DIR\n"
      "    --decode-plan DIR FHE_DECODE_PLACEMENTS_DIR (prefill subcommand: decode-phase plan)\n"
      "    --graph-dir DIR   FHE_GRAPH_DIR\n"
      "    --configs PATH    CONFIGS_PATH\n"
      "    --eager           decode: run without a plan (eager baseline)\n"
      "    --gen-prompt N    GEN_PROMPT (generate; >1 needs GPT2_PACKING=cachemir)\n"
      "    --gen-tokens N    GEN_TOKENS (generate)\n"
      "    --teacher-forced  TEACHER_FORCED (generate: advance on GT tokens; CutMax still measured)\n";
}

std::vector<double> softmax_of(const std::vector<double>& v) {
    std::vector<double> p(v.size());
    const double m = *std::max_element(v.begin(), v.end());
    double s = 0.0;
    for (size_t i = 0; i < v.size(); ++i) { p[i] = std::exp(v[i] - m); s += p[i]; }
    for (double& x : p) x /= s;
    return p;
}

double kl_div(const std::vector<double>& p, const std::vector<double>& q) {
    double kl = 0.0;
    for (size_t i = 0; i < p.size(); ++i)
        if (p[i] > 0.0 && q[i] > 0.0) kl += p[i] * std::log(p[i] / q[i]);
    return kl;
}

std::vector<int> topk_indices(const std::vector<double>& v, int k) {
    std::vector<int> idx(v.size());
    for (size_t i = 0; i < v.size(); ++i) idx[i] = static_cast<int>(i);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(k);
    return idx;
}

void report_decode(std::ostream& os, const std::string& cmd,
                   const RunConfig& cfg, const RunResult& r) {
    const GtSteps gt = r.top1.empty() ? GtSteps{} : read_lm_head_steps(cfg);
    for (size_t i = 0; i < r.top1.size(); ++i) {
        const int pos = (i < r.positions.size()) ? r.positions[i] : static_cast<int>(i);
        os << "[cuda_cachemir] tok" << pos << " top1=" << r.top1[i];
        if (pos < gt.T && i < r.logits.size()
            && r.logits[i].size() == gt.logits[pos].size()) {
            const auto& truth = gt.logits[pos];
            const auto top5_fhe = topk_indices(r.logits[i], 5);
            int overlap = 0;
            for (int idx : topk_indices(truth, 5))
                if (std::find(top5_fhe.begin(), top5_fhe.end(), idx) != top5_fhe.end())
                    ++overlap;
            os << " ref=" << topk_indices(truth, 1)[0]
               << " KL=" << kl_div(softmax_of(truth), softmax_of(r.logits[i]))
               << " top5_overlap=" << overlap << "/5";
        } else {
            os << " (no ground truth)";
        }
        os << "\n";
    }

    os << "[cuda_cachemir] SUMMARY cmd=" << cmd
       << " completed=" << r.completed << "/" << r.requested
       << " bootstraps=" << r.bootstraps
       << " unplanned_bts=" << r.unplanned_bts
       << " weight_relevels=" << r.weight_relevels
       << " s/tok=" << r.avg_s_per_tok;   // forward wall; tok0 (cold start) EXCLUDED from the average
    if (r.avg_argmax_s > 0.0) {
        os << " argmax_s/tok=" << r.avg_argmax_s;
        const double e2e = (cmd == "generate") ? r.avg_s_per_tok
                                               : r.avg_s_per_tok + r.avg_argmax_s;
        os << " e2e_s/tok=" << e2e;
    }
    os << std::endl;
}

int run_decode_multi_cli(const RunConfig& cfg) {
    const char* listp = std::getenv("HE128_SAMPLE_LIST");
    if (!listp || !*listp) {
        std::cerr << "decode_multi: env HE128_SAMPLE_LIST (file of '<id> <dir>' lines) required\n";
        return 2;
    }
    std::ifstream lf(listp);
    if (!lf) { std::cerr << "decode_multi: cannot open HE128_SAMPLE_LIST=" << listp << "\n"; return 2; }
    const char* od  = std::getenv("HE128_LOG_DIR");
    const std::string outdir = (od && *od) ? od : ".";
    const char* jid = std::getenv("SLURM_JOB_ID");
    const std::string jobid = (jid && *jid) ? jid : "0";

    std::vector<std::pair<std::string, std::string>> samples;
    std::string line;
    while (std::getline(lf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        std::string id, dir;
        if (is >> id >> dir) samples.emplace_back(id, dir);
    }
    if (samples.empty()) { std::cerr << "decode_multi: empty sample list " << listp << "\n"; return 2; }

    std::cout << "[decode_multi] samples=" << samples.size() << " tokens=" << cfg.tokens
              << " outdir=" << outdir << " job=" << jobid
              << " (model built ONCE, reused per sample)\n" << std::flush;

    DecodeSession sess(cfg);   // FHE context + rotation/bootstrap keys + weights: paid once here
    int failures = 0;
    for (const auto& s : samples) {
        const std::string& id  = s.first;
        const std::string& dir = s.second;
        const std::string path = outdir + "/decode_" + id + "_" + jobid + ".out";
        RunConfig rc = cfg;
        rc.io_dir = dir;
        std::vector<std::vector<double>> inputs;
        try {
            inputs = read_teacher_forced_inputs(rc);
        } catch (const std::exception& e) {
            std::cerr << "[decode_multi] sample=" << id << " input load FAILED: " << e.what()
                      << " (skipping)\n" << std::flush;
            ++failures;
            continue;
        }
        RunResult r = sess.decode(inputs);
        std::ofstream of(path);
        of << "[cuda_cachemir] cmd=decode_multi sample=" << id << " io=" << dir
           << " tokens=" << cfg.tokens << "\n";
        report_decode(of, "decode", rc, r);
        of.close();
        std::cout << "[decode_multi] sample=" << id
                  << " completed=" << r.completed << "/" << r.requested
                  << " relevels=" << r.weight_relevels
                  << " s/tok=" << r.avg_s_per_tok
                  << " argmax_s/tok=" << r.avg_argmax_s
                  << (r.threw ? " THREW" : "")
                  << " -> " << path << std::endl;
        if (r.threw || r.completed != r.requested) ++failures;
    }
    std::cout << "[decode_multi] DONE samples=" << samples.size()
              << " failures=" << failures << std::endl;
    return failures == 0 ? 0 : 1;
}

bool resolve(int argc, char** argv, int start, const std::string& cmd, RunConfig& cfg,
             bool& eager) {
    for (int i = start; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(2); }
            return argv[++i];
        };
        auto only_for = [&](const char* sub) {
            if (cmd != sub) {
                std::cerr << a << " is only valid for the " << sub << " subcommand\n";
                return false;
            }
            return true;
        };
        if      (a == "--tokens")    cfg.tokens         = std::stoi(next());
        else if (a == "--prefill" || a == "--prefill-tokens") {
            if (!only_for("prefill")) return false;
            cfg.prefill_tokens = std::stoi(next());
        }
        else if (a == "--decode" || a == "--decode-tokens") {
            if (cmd != "decode" && cmd != "prefill") {
                std::cerr << a << " is only valid for decode/prefill\n";
                return false;
            }
            cfg.decode_tokens = std::stoi(next());
        }
        else if (a == "--eager") {
            if (!only_for("decode")) return false;
            eager = true;
        }
        else if (a == "--teacher-forced" || a == "--teacher-forcing") {
            if (!only_for("generate")) return false;
            cfg.teacher_forced = true;   // GT tokens instead of CutMax feedback (CutMax still measured)
        }
        else if (a == "--gen-prompt") {
            if (!only_for("generate")) return false;
            cfg.gen_prompt = std::stoi(next());
        }
        else if (a == "--gen-tokens") {
            if (!only_for("generate")) return false;
            cfg.gen_tokens = std::stoi(next());
        }
        else if (a == "--cut-max") {   // removed 2026-07-05 (CutMax always on); accept+ignore
            std::cerr << "--cut-max is deprecated/removed (encrypted CutMax is always on); ignoring\n";
            next();
        }
        else if (a == "--steps-t")   cfg.steps_t        = std::stoi(next());
        else if (a == "--plan")      cfg.plan_dir       = next();
        else if (a == "--decode-plan") cfg.decode_plan_dir = next();
        else if (a == "--graph-dir") cfg.graph_dir      = next();
        else if (a == "--configs")   cfg.configs_path   = next();
        else if (a == "-h" || a == "--help") { usage(); std::exit(0); }
        else { std::cerr << "unknown flag: " << a << "\n"; usage(); return false; }
    }
    return true;
}

bool normalize_prefill_counts(RunConfig& cfg) {
    const bool has_prefill = cfg.prefill_tokens >= 0;
    const bool has_decode  = cfg.decode_tokens >= 0;

    if (!has_prefill && !has_decode) {
        cfg.prefill_tokens = std::max(0, cfg.tokens - 1);
        cfg.decode_tokens  = 1;
    } else {
        if (!has_prefill)
            cfg.prefill_tokens = std::max(0, cfg.tokens - cfg.decode_tokens);
        if (!has_decode)
            cfg.decode_tokens = 1;
    }

    // decode 0 = prefill-only (capture runs: T=128 prefill exhausts the teacher-forced rows)
    if (cfg.prefill_tokens < 0 || cfg.decode_tokens < 0 ||
        cfg.prefill_tokens + cfg.decode_tokens < 1) {
        std::cerr << "prefill: expected --prefill >= 0, --decode >= 0, total >= 1\n";
        return false;
    }
    cfg.tokens = cfg.prefill_tokens + cfg.decode_tokens;
    return true;
}

}  // namespace

int cli_main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }
    const std::string cmd = argv[1];
    if (cmd != "capture" && cmd != "decode" && cmd != "decode_multi" &&
        cmd != "prefill" && cmd != "generate") { usage(); return 2; }

    RunConfig cfg = RunConfig::from_env();
    bool eager = false;
    if (!resolve(argc, argv, /*start=*/2, cmd, cfg, eager)) return 2;
    if (!cfg.graph_dir.empty())
        setenv("FHE_GRAPH_DIR", cfg.graph_dir.c_str(), 1);

    if (cmd == "capture") {
        cfg.tokens = 1;                                       // token-0 trace
        if (cfg.graph_dir.empty()) {
            std::cerr << "capture: FHE_GRAPH_DIR / --graph-dir required\n";
            return 2;
        }
    } else if (cmd == "decode" || cmd == "decode_multi") {
        if (cfg.decode_tokens >= 0)
            cfg.tokens = cfg.decode_tokens;
        if (cfg.tokens < 1) {
            std::cerr << cmd << ": expected token count >= 1\n";
            return 2;
        }
        if (cfg.plan_dir.empty() && !eager) {
            std::cerr << cmd << ": a plan is required (FHE_BOOTSTRAP_PLACEMENTS_DIR / --plan), "
                         "or pass --eager for the eager baseline.\n";
            return 2;
        }
    } else if (cmd == "generate") {
        cfg.gen_prompt = std::max(1, cfg.gen_prompt);
        cfg.gen_tokens = std::max(1, cfg.gen_tokens);
        // rows to load: prompt only for autoregressive; prompt+tokens when teacher-forced (GT feed).
        cfg.tokens = cfg.teacher_forced ? cfg.gen_prompt + cfg.gen_tokens : cfg.gen_prompt;
    } else {  // prefill
        if (!normalize_prefill_counts(cfg)) return 2;
    }

    std::cout << "[cuda_cachemir] cmd=" << cmd
              << " tokens=" << cfg.tokens;
    if (cmd == "prefill")
        std::cout << " prefill=" << cfg.prefill_tokens
                  << " decode=" << cfg.decode_tokens;
    if (cmd == "generate")
        std::cout << " gen_prompt=" << cfg.gen_prompt
                  << " gen_tokens=" << cfg.gen_tokens;
    std::cout << " plan=" << (cfg.plan_dir.empty() ? "(eager)" : cfg.plan_dir);
    if (cmd == "prefill")
        std::cout << " decode_plan=" << (cfg.decode_plan_dir.empty() ? "(eager)"
                                                                     : cfg.decode_plan_dir);
    std::cout << " graph=" << (cfg.graph_dir.empty() ? "(none)" : cfg.graph_dir)
              << " configs=" << cfg.configs_path << std::endl;

    // Explicit operating-mode line — makes autoregressive vs teacher-forced unambiguous.
    const bool autoregressive = (cmd == "generate") && !cfg.teacher_forced;
    auto env_show = [](const char* k, const char* d) {
        const char* v = std::getenv(k); return std::string((v && *v) ? v : d);
    };
    std::cout << "[mode] " << cmd << ": "
              << (autoregressive
                    ? "AUTOREGRESSIVE (next token = encrypted CutMax feedback; token never revealed)"
                    : "TEACHER-FORCED (next token = GT oracle; CutMax measured, NOT fed back)")
              << " | CutMax=always-on"
              << " | entry_bts=" << (autoregressive ? "on" : "off")
              << " | packing=" << env_show("GPT2_PACKING", "cachemir")
              // +complex only adds info for Mode-A (cachemir + CKKS_COMPLEX); cachemir_complex is already complex
              << ((env_show("CKKS_COMPLEX", "0") != "0" &&
                   env_show("GPT2_PACKING", "cachemir") != "cachemir_complex") ? "+complex(Mode-A)" : "")
              << " | infer=" << env_show("GPT2_INFERENCE_MODE", "threaded")
              << " | plan=" << (cfg.plan_dir.empty() ? "eager" : "planned")
              << " | fold LN1/2/F=" << env_show("GPT2_FOLD_LN1", "0") << "/"
              << env_show("GPT2_FOLD_LN2", "0") << "/" << env_show("GPT2_FOLD_LNF", "0")
              << " | levels: auto_bts=" << env_show("AUTO_BTS_LEVEL", "24")
              << " kv_read=" << env_show("CACHE_READ_LEVEL_K", "17") << "/"
              << env_show("CACHE_READ_LEVEL_V", "17")
              << " lmhead_cap=" << env_show("FHE_LMHEAD_CAP", "22")
              << " rot_band=" << env_show("FIDESLIB_ROT_KEY_BAND", "-1")
              << std::endl;

    // Multi-sample sweep: build the model ONCE, loop samples (per-sample inputs/GT + report).
    if (cmd == "decode_multi") return run_decode_multi_cli(cfg);

    std::vector<std::vector<double>> inputs;
    try {
        inputs = read_teacher_forced_inputs(cfg);
    } catch (const std::exception& e) {
        std::cerr << "[cuda_cachemir] input load failed: " << e.what() << "\n";
        return 1;
    }

    const RunResult r = (cmd == "prefill")  ? run_prefill(cfg, inputs)
                      : (cmd == "generate") ? run_generate(cfg, inputs)
                                            : run_decode(cfg, inputs);

    report_decode(std::cout, cmd, cfg, r);   // per-token top1/ref/KL + SUMMARY (byte-identical)

    if (r.threw) {
        std::cerr << "[cuda_cachemir] THREW at " << r.error << std::endl;
        return 1;
    }
    if (r.completed != r.requested) {
        std::cerr << "[cuda_cachemir] incomplete: " << r.completed
                  << "/" << r.requested << std::endl;
        return 1;
    }
    return 0;
}

}  // namespace app
