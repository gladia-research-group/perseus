#pragma once

#include <cctype>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace json_utils {

inline std::string read_file_to_string(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open file: " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

inline bool has_field(const std::string& obj, const std::string& field) {
    return obj.find("\"" + field + "\"") != std::string::npos;
}

inline size_t value_start(const std::string& obj, const std::string& field) {
    const std::string key = "\"" + field + "\"";
    size_t pos = obj.find(key);
    if (pos == std::string::npos) throw std::runtime_error("Missing field: " + field);
    pos = obj.find(':', pos + key.size());
    if (pos == std::string::npos) throw std::runtime_error("Malformed field: " + field);
    return pos + 1;
}

inline int find_int_field(const std::string& obj, const std::string& field) {
    return std::stoi(obj.substr(value_start(obj, field)));
}

inline double find_double_field(const std::string& obj, const std::string& field) {
    return std::stod(obj.substr(value_start(obj, field)));
}

inline std::string find_string_field(const std::string& obj, const std::string& field) {
    size_t start = value_start(obj, field);
    size_t open  = obj.find('"', start);
    if (open == std::string::npos) throw std::runtime_error("Malformed string: " + field);
    size_t close = obj.find('"', open + 1);
    if (close == std::string::npos) throw std::runtime_error("Unterminated string: " + field);
    return obj.substr(open + 1, close - open - 1);
}

inline bool find_bool_field(const std::string& obj, const std::string& field) {
    size_t start = value_start(obj, field);
    while (start < obj.size() && std::isspace(static_cast<unsigned char>(obj[start]))) ++start;
    if (obj.compare(start, 4, "true")  == 0) return true;
    if (obj.compare(start, 5, "false") == 0) return false;
    return std::stoi(obj.substr(start)) != 0;   // tolerate numeric 0/1
}

inline std::vector<double> find_double_array_field(const std::string& obj, const std::string& field) {
    size_t start = value_start(obj, field);
    size_t open  = obj.find('[', start);
    if (open == std::string::npos) throw std::runtime_error("Malformed array: " + field);
    size_t close = obj.find(']', open + 1);
    if (close == std::string::npos) throw std::runtime_error("Unterminated array: " + field);
    std::vector<double> vals;
    size_t i = open + 1;
    while (i < close) {
        while (i < close && (std::isspace(static_cast<unsigned char>(obj[i])) || obj[i] == ',')) ++i;
        if (i >= close) break;
        size_t consumed = 0;
        vals.push_back(std::stod(obj.substr(i, close - i), &consumed));
        i += consumed;
    }
    return vals;
}

inline std::vector<int> find_int_array_field(const std::string& obj, const std::string& field) {
    size_t start = value_start(obj, field);
    size_t open  = obj.find('[', start);
    if (open == std::string::npos) throw std::runtime_error("Malformed array: " + field);
    size_t close = obj.find(']', open + 1);
    if (close == std::string::npos) throw std::runtime_error("Unterminated array: " + field);
    std::vector<int> vals;
    size_t i = open + 1;
    while (i < close) {
        while (i < close && (std::isspace(static_cast<unsigned char>(obj[i])) || obj[i] == ',')) ++i;
        if (i >= close) break;
        size_t consumed = 0;
        vals.push_back(std::stoi(obj.substr(i, close - i), &consumed));
        i += consumed;
    }
    return vals;
}

// Extract the "{...}" object value of a top-level field, or "" if absent.
inline std::string find_object_field(const std::string& text, const std::string& field) {
    if (!has_field(text, field)) return "";
    const std::string key = "\"" + field + "\"";
    size_t pos   = text.find(key);
    size_t brace = text.find('{', pos + key.size());
    if (brace == std::string::npos) return "";
    int depth = 1;
    size_t j = brace + 1;
    while (j < text.size() && depth > 0) {
        if (text[j] == '{') ++depth;
        else if (text[j] == '}') --depth;
        ++j;
    }
    return text.substr(brace, j - brace);
}

// Iterate the `"key": { ... }` entries of a named object section, invoking
// on_entry(entry_key, entry_object) for each. No-op if the section is absent.
template <typename F>
inline void walk_section(const std::string& text, const std::string& section, F&& on_entry) {
    const std::string key = "\"" + section + "\"";
    size_t pos = text.find(key);
    if (pos == std::string::npos) return;
    pos = text.find('{', pos + key.size());
    if (pos == std::string::npos) return;
    size_t i = pos + 1;
    while (i < text.size()) {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        if (i >= text.size() || text[i] == '}') break;
        if (text[i] != '"') { ++i; continue; }
        size_t key_end = text.find('"', i + 1);
        if (key_end == std::string::npos) break;
        std::string entry_key = text.substr(i + 1, key_end - i - 1);
        size_t brace = text.find('{', key_end + 1);
        if (brace == std::string::npos) break;
        int depth = 1;
        size_t j = brace + 1;
        while (j < text.size() && depth > 0) {
            if (text[j] == '{') ++depth;
            else if (text[j] == '}') --depth;
            ++j;
        }
        if (depth != 0) throw std::runtime_error("malformed JSON entry: " + entry_key);
        on_entry(entry_key, text.substr(brace, j - brace));
        i = j;
        while (i < text.size() && (text[i] == ',' || std::isspace(static_cast<unsigned char>(text[i])))) ++i;
    }
}

// Set `dst` from a JSON field, or keep its existing value and warn if absent.
// For required (calibration-derived) fields use the find_*_field throwers above.
inline void load_or_warn(int& dst, const std::string& obj, const std::string& field) {
    if (has_field(obj, field)) {
        dst = find_int_field(obj, field);
    } else {
        std::cerr << "[json_utils] '" << field << "' not in JSON; "
                  << "using struct default = " << dst << "\n";
    }
}
inline void load_or_warn(double& dst, const std::string& obj, const std::string& field) {
    if (has_field(obj, field)) {
        dst = find_double_field(obj, field);
    } else {
        std::cerr << "[json_utils] '" << field << "' not in JSON; "
                  << "using struct default = " << dst << "\n";
    }
}
inline void load_or_warn(bool& dst, const std::string& obj, const std::string& field) {
    if (has_field(obj, field)) {
        dst = find_bool_field(obj, field);
    } else {
        std::cerr << "[json_utils] '" << field << "' not in JSON; "
                  << "using struct default = " << (dst ? "true" : "false") << "\n";
    }
}
inline void load_or_warn(std::vector<double>& dst, const std::string& obj, const std::string& field) {
    if (has_field(obj, field)) {
        dst = find_double_array_field(obj, field);
    } else {
        std::ostringstream preview;
        preview << "[";
        for (size_t i = 0; i < dst.size() && i < 3; ++i) {
            if (i) preview << ", ";
            preview << dst[i];
        }
        if (dst.size() > 3) preview << ", …";
        preview << "]";
        std::cerr << "[json_utils] '" << field << "' not in JSON; "
                  << "using struct default (size=" << dst.size()
                  << ") = " << preview.str() << "\n";
    }
}

} // namespace json_utils
