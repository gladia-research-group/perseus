// Parity + correctness gate for the non-recursive plan parser (include/plan_json.h).
//
// Scanning placement JSON with recursive libstdc++ std::regex overflows an 8 MB stack on
// real ~0.5 MB files, and the segfault lands inside the parser rather than at the caller.
// The iterative parser must read EVERY real plan in the repo field-for-field identically to
// the regex one, plus the optional keys.
//
// The regex parser is preserved verbatim below as the golden reference; it runs on a
// dedicated 512 MB-stack pthread so the parity test itself cannot segfault at the
// default ulimit.
//
// CPU-only — no crypto context, no GPU.
#include "fideslib_wrapper.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <pthread.h>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ── the regex parser, verbatim (golden reference) ───────────────────────────────────
BootstrapPlan legacy_parse(const std::string& path) {
    BootstrapPlan plan;

    std::ifstream in(path);
    if (!in) {
        return plan;
    }

    const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    const std::regex entry_re(
        "\\{[^\\{\\}]*\\\"type\\\"\\s*:\\s*\\\"(bootstrap_after_node)\\\"[^\\{\\}]*\\}");
    const std::regex target_var_re("\\\"target_var\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");

    for (std::sregex_iterator it(content.begin(), content.end(), entry_re), end; it != end; ++it) {
        const std::string entry = it->str();
        std::smatch type_match;
        std::smatch target_match;
        if (!std::regex_search(entry, type_match,
                               std::regex("\\\"type\\\"\\s*:\\s*\\\"(bootstrap_after_node)\\\""))) {
            continue;
        }
        if (!std::regex_search(entry, target_match, target_var_re)) {
            continue;
        }
        if (type_match[1].str() == "bootstrap_after_node") {
            plan.placement_after.insert(target_match[1].str());
        }
    }

    const std::regex levels_dict_re("\\\"final_named_levels\\\"\\s*:\\s*\\{([^{}]+)\\}");
    std::smatch match;
    if (std::regex_search(content, match, levels_dict_re)) {
        std::string dict_str = match[1].str();
        const std::regex kv_re("\\\"([^\\\"]+)\\\"\\s*:\\s*([0-9]+)");
        for (std::sregex_iterator it(dict_str.begin(), dict_str.end(), kv_re), end; it != end; ++it) {
            plan.expected_levels[(*it)[1].str()] = static_cast<uint32_t>(std::stoul((*it)[2].str()));
        }
    }

    const std::regex producers_dict_re("\\\"final_named_producers\\\"\\s*:\\s*\\{([^{}]+)\\}");
    if (std::regex_search(content, match, producers_dict_re)) {
        std::string dict_str = match[1].str();
        const std::regex kv_re("\\\"([^\\\"]+)\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
        for (std::sregex_iterator it(dict_str.begin(), dict_str.end(), kv_re), end; it != end; ++it) {
            plan.expected_producers[(*it)[1].str()] = (*it)[2].str();
        }
    }

    const std::regex wlvls_dict_re("\\\"weight_levels\\\"\\s*:\\s*\\{([^{}]+)\\}");
    if (std::regex_search(content, match, wlvls_dict_re)) {
        std::string dict_str = match[1].str();
        const std::regex kv_re("\\\"([^\\\"]+)\\\"\\s*:\\s*([0-9]+)");
        for (std::sregex_iterator it(dict_str.begin(), dict_str.end(), kv_re), end; it != end; ++it) {
            plan.weight_levels[(*it)[1].str()] = static_cast<uint32_t>(std::stoul((*it)[2].str()));
        }
    }

    const std::regex mlvls_dict_re("\\\"mask_levels\\\"\\s*:\\s*\\{([^{}]+)\\}");
    if (std::regex_search(content, match, mlvls_dict_re)) {
        const std::string dict_str = match[1].str();
        const std::regex site_re("\\\"([^\\\"]+)\\\"\\s*:\\s*\\[([^\\]]*)\\]");
        const std::regex num_re("([0-9]+)");
        for (std::sregex_iterator it(dict_str.begin(), dict_str.end(), site_re), end; it != end; ++it) {
            std::vector<uint32_t> lvls;
            const std::string arr = (*it)[2].str();
            for (std::sregex_iterator n(arr.begin(), arr.end(), num_re), nend; n != nend; ++n)
                lvls.push_back(static_cast<uint32_t>(std::stoul((*n)[1].str())));
            if (!lvls.empty()) plan.mask_levels[(*it)[1].str()] = std::move(lvls);
        }
    }

    const std::regex hint_fire_re("\\\"hint_fire\\\"\\s*:\\s*\\[([^\\]]*)\\]");
    if (std::regex_search(content, match, hint_fire_re)) {
        plan.hints_bound = true;
        const std::string arr = match[1].str();
        const std::regex var_re("\\\"([^\\\"]+)\\\"");
        for (std::sregex_iterator it(arr.begin(), arr.end(), var_re), end; it != end; ++it)
            plan.hint_fire.insert((*it)[1].str());
    }

    std::smatch cpl_match;
    if (std::regex_search(content, cpl_match, std::regex("\\\"cache_pin_level\\\"\\s*:\\s*([0-9]+)")))
        plan.cache_pin_level = std::stoi(cpl_match[1].str());

    plan.valid = !plan.placement_after.empty() || !plan.weight_levels.empty();
    return plan;
}

// Run legacy_parse on a 512 MB pthread stack (recursive std::regex would SIGSEGV a
// default 8 MB stack on the real files — the very bug the new parser removes).
struct LegacyJob {
    std::string path;
    BootstrapPlan out;
};
void* legacy_thread_main(void* arg) {
    auto* job = static_cast<LegacyJob*>(arg);
    job->out = legacy_parse(job->path);
    return nullptr;
}
BootstrapPlan legacy_parse_bigstack(const std::string& path) {
    LegacyJob job;
    job.path = path;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 512ull << 20);
    pthread_t th;
    EXPECT_EQ(0, pthread_create(&th, &attr, legacy_thread_main, &job));
    pthread_join(th, nullptr);
    pthread_attr_destroy(&attr);
    return job.out;
}

void expect_plans_equal(const BootstrapPlan& a, const BootstrapPlan& b, const std::string& path) {
    EXPECT_EQ(a.valid, b.valid) << path;
    EXPECT_EQ(a.placement_after, b.placement_after) << path;
    EXPECT_EQ(a.expected_levels, b.expected_levels) << path;
    EXPECT_EQ(a.expected_producers, b.expected_producers) << path;
    EXPECT_EQ(a.weight_levels, b.weight_levels) << path;
    EXPECT_EQ(a.mask_levels, b.mask_levels) << path;
    EXPECT_EQ(a.hint_fire, b.hint_fire) << path;
    EXPECT_EQ(a.hints_bound, b.hints_bound) << path;
    EXPECT_EQ(a.cache_pin_level, b.cache_pin_level) << path;
}

std::vector<std::string> repo_plan_files() {
    std::vector<std::string> files;
    const char* root_env = std::getenv("REPO_ROOT");
    for (fs::path root : {fs::path(root_env ? root_env : "."),
                          fs::path(".."), fs::path("../.."), fs::path("../../..")}) {
        fs::path base = root / "bootstrap_placements";
        if (!fs::is_directory(base)) continue;
        for (const auto& e : fs::recursive_directory_iterator(base)) {
            const std::string name = e.path().filename().string();
            if (e.is_regular_file() && name.find("_placement.json") != std::string::npos)
                files.push_back(e.path().string());
        }
        if (!files.empty()) break;
    }
    return files;
}

}   // namespace

// Every real plan file in the repo parses field-for-field identically to the legacy
// regex parser.
TEST(PlanParse, ParityOnEveryRepoPlan) {
    const auto files = repo_plan_files();
    if (files.empty())
        GTEST_SKIP() << "no bootstrap_placements/*.json found (set REPO_ROOT)";
    size_t checked = 0;
    for (const auto& f : files) {
        const BootstrapPlan legacy = legacy_parse_bigstack(f);
        const BootstrapPlan fresh  = parse_bootstrap_plan_file(f);
        expect_plans_equal(legacy, fresh, f);
        ++checked;
    }
    std::printf("[plan_parse] parity checked on %zu plan files\n", checked);
    EXPECT_GT(checked, 0u);
}

// The new opt-in keys parse; legacy files without them leave the maps empty.
TEST(PlanParse, SparseAndPrescaleKeys) {
    const std::string json = R"({
      "version": 1,
      "rules": {"bootstrap_level": 16, "max_level": 24, "cache_pin_level": 17,
                "sparse_bts_slots": [1, 32, 512]},
      "placements": [
        {"type": "bootstrap_after_node", "node_op": "mult", "target_var": "v_10"},
        {"type": "bootstrap_after_node", "node_op": "square", "target_var": "v_20"}
      ],
      "weight_levels": {"q": 21},
      "mask_levels": {"ln.scalemask": [18, 19]},
      "hint_fire": ["v_30"],
      "sparse_slots": {"v_10": 1, "v_30": 512},
      "prescale": {"v_20": 0.015625},
      "summary": {"final_named_levels": {"v_10": 16, "v_20": 17}}
    })";
    const std::string path = fs::temp_directory_path() / "test_plan_parse_sparse.json";
    { std::ofstream(path) << json; }

    const BootstrapPlan p = parse_bootstrap_plan_file(path);
    EXPECT_TRUE(p.valid);
    EXPECT_EQ(p.placement_after.size(), 2u);
    ASSERT_EQ(p.sparse_slots.size(), 2u);
    EXPECT_EQ(p.sparse_slots.at("v_10"), 1u);
    EXPECT_EQ(p.sparse_slots.at("v_30"), 512u);
    ASSERT_EQ(p.prescale.size(), 1u);
    EXPECT_DOUBLE_EQ(p.prescale.at("v_20"), 0.015625);
    ASSERT_EQ(p.sparse_bts_slots_rule.size(), 3u);
    EXPECT_EQ(p.sparse_bts_slots_rule[2], 512u);
    EXPECT_EQ(p.expected_levels.at("v_20"), 17u);
    EXPECT_EQ(p.cache_pin_level, 17);
    EXPECT_TRUE(p.hints_bound);
    fs::remove(path);
}

// Malformed input must be refused (valid=false), never crash.
TEST(PlanParse, MalformedInputIsRefused) {
    const std::string path = fs::temp_directory_path() / "test_plan_parse_bad.json";
    { std::ofstream(path) << "{ \"placements\": [ {\"type\": \"boot"; }
    const BootstrapPlan p = parse_bootstrap_plan_file(path);
    EXPECT_FALSE(p.valid);
    fs::remove(path);
}
