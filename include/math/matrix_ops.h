#pragma once

// Host-side (plaintext) vector/matrix helpers used when preparing weights for
// encoding: padding, slicing, transpose, row scaling. No CKKS/CUDA dependency.

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace matrix {

// Smallest power of two >= v (used to size padded ciphertext dimensions).
inline int next_pow2(int v) {
    if (v <= 0) throw std::runtime_error("next_pow2: non-positive input");
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

inline std::vector<double> pad_vector(const std::vector<double>& v, int n) {
    if ((int)v.size() > n) throw std::runtime_error("pad_vector: shrink not allowed");
    std::vector<double> out(n, 0.0);
    std::copy(v.begin(), v.end(), out.begin());
    return out;
}

inline std::vector<double> slice_vec(const std::vector<double>& v, int start, int len) {
    if (start + len > (int)v.size()) throw std::runtime_error("slice_vec: out of range");
    return std::vector<double>(v.begin() + start, v.begin() + start + len);
}

inline std::vector<std::vector<double>> pad_matrix(
    const std::vector<std::vector<double>>& W, int rows, int cols) {
    int r = (int)W.size(), c = (int)W[0].size();
    if (r > rows || c > cols) throw std::runtime_error("pad_matrix: shrink not allowed");
    std::vector<std::vector<double>> out(rows, std::vector<double>(cols, 0.0));
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j) out[i][j] = W[i][j];
    return out;
}

inline std::vector<std::vector<double>> slice_matrix(
    const std::vector<std::vector<double>>& W, int rows, int cols) {
    int r = (int)W.size(), c = (int)W[0].size();
    if (rows > r || cols > c) throw std::runtime_error("slice_matrix: out of range");
    std::vector<std::vector<double>> out(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) out[i][j] = W[i][j];
    return out;
}

inline std::vector<std::vector<double>> transpose(
    const std::vector<std::vector<double>>& W) {
    int r = (int)W.size(), c = (int)W[0].size();
    std::vector<std::vector<double>> out(c, std::vector<double>(r));
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j) out[j][i] = W[i][j];
    return out;
}

inline std::vector<std::vector<double>> slice_cols(
    const std::vector<std::vector<double>>& W, int c_start, int c_len) {
    int r = (int)W.size();
    std::vector<std::vector<double>> out(r, std::vector<double>(c_len));
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c_len; ++j) out[i][j] = W[i][c_start + j];
    return out;
}

inline std::vector<std::vector<double>> slice_rows(
    const std::vector<std::vector<double>>& W, int r_start, int r_len) {
    if (r_start + r_len > (int)W.size())
        throw std::runtime_error("slice_rows: out of range");
    return std::vector<std::vector<double>>(W.begin() + r_start,
                                            W.begin() + r_start + r_len);
}

inline std::vector<std::vector<double>> mat_scale_rows(
    const std::vector<std::vector<double>>& W, const std::vector<double>& scale) {
    int r = (int)W.size(), c = (int)W[0].size();
    if ((int)scale.size() != r) throw std::runtime_error("mat_scale_rows: size mismatch");
    std::vector<std::vector<double>> out(r, std::vector<double>(c));
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j) out[i][j] = W[i][j] * scale[i];
    return out;
}

inline std::vector<double> matmul_vec_mat(const std::vector<double>& x,
                                          const std::vector<std::vector<double>>& W) {
    int d_in = (int)x.size(), d_out = (int)W[0].size();
    std::vector<double> y(d_out, 0.0);
    for (int j = 0; j < d_out; ++j)
        for (int i = 0; i < d_in; ++i) y[j] += x[i] * W[i][j];
    return y;
}

inline std::string shape_str(int r, int c) {
    return std::to_string(r) + "x" + std::to_string(c);
}

} // namespace matrix
