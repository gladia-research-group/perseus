#pragma once

#include "inference.h"
#include "packing/packed_ctx.h"

#include <cuda_runtime.h>
#include <string>
#include <vector>

// Cachemir-packing parameters and weight/input encoding helpers for linear
// layers. The slot layout (interleaved heads / cascading t/tp/alpha) is the
// cachemir convention.

namespace cachemir {

struct CacheMirParams {
    bool is_up;
    int d, alpha, t, tp, tp_in, tp_out, r_i, r_o, n_pt;
    int bstep_c, gstep_c;
};

CacheMirParams compute_cm_params(int N, int d_in, int d_out);

int interleave_idx(int m, int d, int dim);

PackedCtx encode_linear_input(Inference& inf, const std::vector<double>& x,
                              int d_in, int d_out, int target_level = 0);

std::vector<Ptx> encode_weight_matrix(Inference& inf,
                                       const std::vector<std::vector<double>>& W,
                                       int d_in, int d_out,
                                       int target_level = 0);

std::vector<Ptx> encode_weight_matrix(Inference& inf,
                                       const std::vector<std::vector<double>>& W,
                                       int d_in, int d_out,
                                       int target_level,
                                       cudaStream_t stream);

// Complex weight (W_re + i*W_im) in the cachemir diagonal layout — one linear emits W_re·x + i*W_im·x.
std::vector<Ptx> encode_weight_matrix_complex(Inference& inf,
                                              const std::vector<std::vector<double>>& W_re,
                                              const std::vector<std::vector<double>>& W_im,
                                              int d_in, int d_out,
                                              int target_level = 0,
                                              cudaStream_t stream = nullptr);

Ptx encode_bias_vector(Inference& inf, const std::vector<double>& b,
                        int d_in, int d_out, bool fill = true);

Ptx encode_bias_vector(Inference& inf, const std::vector<double>& b,
                        int d_in, int d_out, bool fill,
                        cudaStream_t stream);

Ptx encode_bias_vector_complex(Inference& inf, const std::vector<double>& b_re,
                               const std::vector<double>& b_im, int d_in, int d_out,
                               bool fill = true, cudaStream_t stream = nullptr);

std::vector<Ptx> encode_weight_matrix_outputpack(Inference& inf,
                                                 const std::vector<std::vector<double>>& W,
                                                 int d_in, int d_out, int target_level = 0,
                                                 cudaStream_t stream = nullptr);

std::vector<Ptx> load_weight_txt(Inference& inf, const std::string& path,
                                  int d_in, int d_out,
                                  int target_level = 0);

std::vector<double> decode_linear_output(int slots, const std::vector<double>& cy,
                                         int d_in, int d_out);

std::vector<std::vector<double>> decode_tokens(int slots, const std::vector<double>& cy,
                                               int d_pad, int d_real, int T);

}  // namespace cachemir
