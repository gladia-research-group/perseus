#include "packing/cachemir/cachemir_attention_utils.h"

#include <vector>

namespace cachemir {

std::vector<std::vector<double>> rearrange_qkv_weights(
        const std::vector<std::vector<double>>& W, int H) {
    int d_in  = (int)W.size();
    int d_out = (int)W[0].size();
    int d_head = d_out / H;
    std::vector<std::vector<double>> out(d_in, std::vector<double>(d_out));
    for (int r = 0; r < d_out; ++r) {
        int h  = r % H;
        int ld = r / H;
        int src_col = h * d_head + ld;
        for (int i = 0; i < d_in; ++i)
            out[i][r] = W[i][src_col];
    }
    return out;
}

std::vector<double> rearrange_qkv_biases(
        const std::vector<double>& b, int H) {
    int d_out = (int)b.size();
    int d_head = d_out / H;
    std::vector<double> out(d_out);
    for (int r = 0; r < d_out; ++r) {
        int h  = r % H;
        int ld = r / H;
        int src_idx = h * d_head + ld;
        out[r] = b[src_idx];
    }
    return out;
}

std::vector<std::vector<double>> rearrange_wo_weights(
        const std::vector<std::vector<double>>& W, int H) {
    int d_in  = (int)W.size();
    int d_out = (int)W[0].size();
    int d_head = d_in / H;
    std::vector<std::vector<double>> out(d_in, std::vector<double>(d_out));
    for (int r = 0; r < d_in; ++r) {
        int h  = r % H;
        int ld = r / H;
        int src_row = h * d_head + ld;
        out[r] = W[src_row];
    }
    return out;
}

}  // namespace cachemir
