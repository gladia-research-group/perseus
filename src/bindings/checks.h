#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace perseus_checks {

inline void check_matrix(const char* fn, const std::string& name,
                         const std::vector<std::vector<double>>& W, int d_in, int d_out) {
    const size_t rows = W.size();
    const size_t cols = rows ? W[0].size() : 0;
    for (size_t r = 0; r < rows; ++r) {
        if (W[r].size() != cols)
            throw std::invalid_argument(std::string(fn) + "(" + name + "): ragged matrix (row " +
                                        std::to_string(r) + " has " + std::to_string(W[r].size()) +
                                        " entries, row 0 has " + std::to_string(cols) + ")");
    }
    if (static_cast<int>(rows) != d_in || static_cast<int>(cols) != d_out) {
        std::string msg = std::string(fn) + "(" + name + "): weight shape (" +
                          std::to_string(rows) + ", " + std::to_string(cols) +
                          ") != (d_in, d_out) = (" + std::to_string(d_in) + ", " +
                          std::to_string(d_out) + ")";
        if (static_cast<int>(rows) == d_out && static_cast<int>(cols) == d_in && d_in != d_out)
            msg += " -- that is torch's (out_features, in_features) layout; perseus computes "
                   "y = x @ W with W stored as (d_in, d_out): pass weight.T";
        throw std::invalid_argument(msg);
    }
}

inline void check_vector_max(const char* fn, const std::string& name, const std::vector<double>& v,
                             int n) {
    if (static_cast<int>(v.size()) > n)
        throw std::invalid_argument(std::string(fn) + "(" + name + "): " +
                                    std::to_string(v.size()) + " entries do not fit d_out = " +
                                    std::to_string(n) + " (a shorter vector is zero-filled)");
}

inline void check_max_len(const char* fn, const std::vector<double>& v, int max_len) {
    if (static_cast<int>(v.size()) > max_len)
        throw std::invalid_argument(std::string(fn) + ": " + std::to_string(v.size()) +
                                    " values do not fit the session's " +
                                    std::to_string(max_len) +
                                    "-wide token (values past it would be dropped)");
}

}  // namespace perseus_checks
