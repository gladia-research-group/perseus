#pragma once

#include "test_helpers.h"
#include "io/json_utils.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace test_helpers {

inline std::string all_blocks_io_path(const std::string& io_dir,
                                      int block_idx, int T_val) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "%s/all_blocks_L%02d_T%d.json",
                  io_dir.c_str(), block_idx, T_val);
    return std::string(buf);
}

inline std::string default_all_blocks_io_dir() {
    return default_io_dir("ALL_BLOCKS_IO_DIR", "all_blocks_io");
}

inline std::vector<std::vector<double>> read_tap_2d(const std::string& path,
                                                    const std::string& key) {
    std::string text = json_utils::read_file_to_string(path);
    return parse_2d_array(text, key);
}

inline std::vector<std::vector<std::vector<double>>> read_tap_3d(
        const std::string& path, const std::string& key) {
    std::string text = json_utils::read_file_to_string(path);
    return parse_3d_array(text, key);
}

inline std::vector<std::vector<double>> read_tap_3d_last_row(
        const std::string& path, const std::string& key) {
    auto cube = read_tap_3d(path, key);
    std::vector<std::vector<double>> out;
    out.reserve(cube.size());
    for (auto& mat : cube) {
        if (mat.empty()) throw std::runtime_error("empty head matrix in " + key);
        out.push_back(std::move(mat.back()));
    }
    return out;
}

inline IoArrays read_block0_io(const std::string& path,
                               const std::string& inp_key,
                               const std::string& res_key) {
    std::string text = json_utils::read_file_to_string(path);
    return {parse_2d_array(text, inp_key), parse_2d_array(text, res_key)};
}

inline int parse_scalar_int(const std::string& text, const std::string& key) {
    const std::string quoted = "\"" + key + "\"";
    size_t pos = text.find(quoted);
    if (pos == std::string::npos) throw std::runtime_error("Missing key: " + key);
    pos = text.find(':', pos + quoted.size());
    if (pos == std::string::npos) throw std::runtime_error("Malformed scalar: " + key);
    size_t i = pos + 1;
    return static_cast<int>(parse_number(text, i));
}

inline std::vector<double> parse_1d_array(const std::string& text,
                                          const std::string& key) {
    const std::string quoted = "\"" + key + "\"";
    size_t pos = text.find(quoted);
    if (pos == std::string::npos) throw std::runtime_error("Missing key: " + key);
    pos = text.find('[', pos + quoted.size());
    if (pos == std::string::npos) throw std::runtime_error("Malformed array: " + key);
    size_t i = pos;
    return parse_row_at(text, i);
}

struct LmHeadTruth {
    std::vector<double> logits;     // [vocab]
    int                 argmax;
    std::vector<int>    topk_idx;
    std::vector<double> topk_val;
};

inline LmHeadTruth read_lm_head_truth(const std::string& path) {
    std::string text = json_utils::read_file_to_string(path);
    LmHeadTruth t;
    t.logits   = parse_1d_array(text, "logits");
    t.argmax   = parse_scalar_int(text, "argmax");
    t.topk_val = parse_1d_array(text, "topk_val");
    for (double v : parse_1d_array(text, "topk_idx"))
        t.topk_idx.push_back(static_cast<int>(v));
    return t;
}

// Per-position next-token logits from the gather script's
// all_blocks_lm_head_steps_T{T}.json: one LmHeadTruth per position 0..T-1, the
// ground truth for the multi-token autoregressive chain test.
struct LmHeadSteps {
    int                      T     = 0;
    int                      vocab = 0;
    std::vector<LmHeadTruth> steps;
};

inline std::string all_blocks_lm_head_steps_path(const std::string& io_dir, int T_val) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "%s/all_blocks_lm_head_steps_T%d.json", io_dir.c_str(), T_val);
    return std::string(buf);
}

inline LmHeadSteps read_lm_head_steps(const std::string& path) {
    std::string text = json_utils::read_file_to_string(path);
    LmHeadSteps out;
    out.T     = parse_scalar_int(text, "T");
    out.vocab = parse_scalar_int(text, "vocab");

    const std::string key = "\"steps\"";
    size_t pos = text.find(key);
    if (pos == std::string::npos) throw std::runtime_error("Missing key: steps");
    pos = text.find('[', pos + key.size());
    if (pos == std::string::npos) throw std::runtime_error("Malformed steps array");

    size_t i = pos + 1;  // just past the '['
    while (true) {
        skip_ws(text, i);
        if (i >= text.size()) throw std::runtime_error("Unexpected end in steps");
        if (text[i] == ']') { ++i; break; }
        if (text[i] == ',') { ++i; continue; }
        if (text[i] != '{') throw std::runtime_error("Expected '{' in steps");

        // Slice this step object by brace matching (values are numbers/arrays,
        // so no braces appear inside strings) and parse it with the 1D helpers.
        const size_t obj_start = i;
        int depth = 0;
        size_t j = i;
        for (; j < text.size(); ++j) {
            if (text[j] == '{') ++depth;
            else if (text[j] == '}' && --depth == 0) { ++j; break; }
        }
        const std::string obj = text.substr(obj_start, j - obj_start);

        LmHeadTruth t;
        t.logits   = parse_1d_array(obj, "logits");
        t.argmax   = parse_scalar_int(obj, "argmax");
        t.topk_val = parse_1d_array(obj, "topk_val");
        for (double v : parse_1d_array(obj, "topk_idx"))
            t.topk_idx.push_back(static_cast<int>(v));
        out.steps.push_back(std::move(t));
        i = j;
    }
    return out;
}

}  // namespace test_helpers
