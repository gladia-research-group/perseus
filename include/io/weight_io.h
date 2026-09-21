#pragma once

#include "json_utils.h"

#ifdef WEIGHT_LOADER_WITH_LIBARCHIVE
#include <archive.h>
#include <archive_entry.h>
#endif

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace weight_io {

struct TensorMeta {
    std::string name;
    std::string path;
    std::vector<int> shape;
    std::string dtype;
};

inline std::vector<char> read_file_to_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open file: " + path);
    f.seekg(0, std::ios::end);
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(size));
    if (size > 0 && !f.read(buf.data(), size))
        throw std::runtime_error("Failed to read: " + path);
    return buf;
}

inline std::vector<std::vector<double>> load_matrix_txt(const std::string& path,
                                                        int d_in, int d_out) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    std::vector<std::vector<double>> M(d_in, std::vector<double>(d_out));
    for (int i = 0; i < d_in; ++i)
        for (int j = 0; j < d_out; ++j)
            if (!(f >> M[i][j]))
                throw std::runtime_error(path + ": unexpected end of data");
    return M;
}

#ifdef WEIGHT_LOADER_WITH_LIBARCHIVE
inline std::vector<char> read_zip_entry(const std::string& zip_path,
                                        const std::string& entry_name) {
    struct archive* ar = archive_read_new();
    archive_read_support_format_zip(ar);
    archive_read_support_filter_all(ar);
    if (archive_read_open_filename(ar, zip_path.c_str(), 10240) != ARCHIVE_OK) {
        std::string err = archive_error_string(ar);
        archive_read_free(ar);
        throw std::runtime_error("Failed to open zip: " + zip_path + " error=" + err);
    }
    struct archive_entry* entry = nullptr;
    std::vector<char> data;
    bool found = false;
    while (archive_read_next_header(ar, &entry) == ARCHIVE_OK) {
        const char* name = archive_entry_pathname(entry);
        if (name && entry_name == name) {
            const size_t size = static_cast<size_t>(archive_entry_size(entry));
            data.resize(size);
            size_t offset = 0;
            while (offset < size) {
                const ssize_t n = archive_read_data(ar, data.data() + offset, size - offset);
                if (n <= 0) break;
                offset += static_cast<size_t>(n);
            }
            if (offset != size) {
                archive_read_free(ar);
                throw std::runtime_error("Failed to read zip entry: " + entry_name);
            }
            found = true;
            break;
        }
        archive_read_data_skip(ar);
    }
    archive_read_free(ar);
    if (!found) throw std::runtime_error("Zip entry not found: " + entry_name);
    return data;
}
#else
inline std::vector<char> read_zip_entry(const std::string&, const std::string&) {
    throw std::runtime_error("LibArchive not available. Use WEIGHTS_DIR.");
}
#endif

inline std::vector<TensorMeta> parse_manifest_json(const std::string& text) {
    const std::string key = "\"tensors\"";
    size_t pos = text.find(key);
    if (pos == std::string::npos) throw std::runtime_error("manifest: missing tensors");
    pos = text.find('[', pos + key.size());
    std::vector<TensorMeta> out;
    size_t i = pos + 1;
    while (i < text.size()) {
        if (text[i] == '{') {
            int depth = 1;
            size_t obj_start = i++;
            while (i < text.size() && depth > 0) {
                if (text[i] == '{') ++depth;
                else if (text[i] == '}') --depth;
                ++i;
            }
            std::string obj = text.substr(obj_start, i - obj_start);
            TensorMeta m;
            m.name  = json_utils::find_string_field(obj, "name");
            m.path  = json_utils::find_string_field(obj, "path");
            m.dtype = json_utils::find_string_field(obj, "dtype");
            m.shape = json_utils::find_int_array_field(obj, "shape");
            out.push_back(std::move(m));
        } else if (text[i] == ']') {
            break;
        } else {
            ++i;
        }
    }
    return out;
}

inline std::vector<double> decode_tensor(const std::vector<char>& bytes,
                                         const std::string& dtype) {
    const bool is_f4 = (dtype == "<f4" || dtype == "|f4" || dtype == "f4");
    const bool is_f8 = (dtype == "<f8" || dtype == "|f8" || dtype == "f8");
    if (!is_f4 && !is_f8) throw std::runtime_error("Unsupported dtype: " + dtype);
    std::vector<double> out;
    if (is_f4) {
        size_t n = bytes.size() / sizeof(float);
        out.resize(n);
        const float* p = reinterpret_cast<const float*>(bytes.data());
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(p[i]);
    } else {
        size_t n = bytes.size() / sizeof(double);
        out.resize(n);
        const double* p = reinterpret_cast<const double*>(bytes.data());
        for (size_t i = 0; i < n; ++i) out[i] = p[i];
    }
    return out;
}

} // namespace weight_io
