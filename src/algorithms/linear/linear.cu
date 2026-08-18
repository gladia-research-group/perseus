#include "model/gpt2.h"
#include "inference.h"
#include "packing/cachemir/cachemir_linear.h"
#include "packing/cachemir/cachemir_linear_utils.h"
#include "packing/diagonal/diagonal_linear.h"
#include "packing/diagonal/diagonal_linear_utils.h"

#include <stdexcept>

// CachemirFilling reuses the diagonal

PackedCtx linear(Inference& inf, const PackedCtx& x,
                 const std::string& wname, int d_in, int d_out, bool stream_pt) {
    if (is_cachemir(x.packing)) return cachemir::linear(inf, x, wname, d_in, d_out);
    if (is_diagonal(x.packing) || is_cachemir_filling(x.packing)) {
        CKKSContext::MagnitudeSuppressScope _ms(*inf.fhe);
        return diagonal::linear(inf, x, wname, d_in, d_out, stream_pt);
    }
    throw std::runtime_error("linear: unsupported packing");
}

PreparedLinearInput prepare_linear_input(Inference& inf, const PackedCtx& x,
                                         int d_in, int d_out) {
    if (is_cachemir(x.packing))
        return {cachemir::prepare_linear_input(inf, x, d_in, d_out), x, d_in, d_out};
    if (is_diagonal(x.packing) || is_cachemir_filling(x.packing))
        return {{}, x, d_in, d_out};   // diagonal preps per apply
    throw std::runtime_error("prepare_linear_input: unsupported packing");
}

PackedCtx apply_linear(Inference& inf, const PreparedLinearInput& prep,
                       const std::string& wname, bool stream_pt) {
    if (is_cachemir(prep.x.packing))
        return cachemir::apply_linear(inf, prep.rotated, wname, prep.d_in, prep.d_out);
    if (is_diagonal(prep.x.packing) || is_cachemir_filling(prep.x.packing)) {
        CKKSContext::MagnitudeSuppressScope _ms(*inf.fhe);
        return diagonal::linear(inf, prep.x, wname, prep.d_in, prep.d_out, stream_pt);
    }
    throw std::runtime_error("apply_linear: unsupported packing");
}

std::vector<PackedCtx> linear_multi(Inference& inf, const PackedCtx& x,
                                    const std::vector<std::string>& wnames,
                                    int d_in, int d_out, bool stream_pt) {
    const PreparedLinearInput prep = prepare_linear_input(inf, x, d_in, d_out);
    std::vector<PackedCtx> outs;
    outs.reserve(wnames.size());
    for (const auto& wname : wnames)
        outs.push_back(apply_linear(inf, prep, wname, stream_pt));
    return outs;
}

PackedCtx encode_linear_input(Inference& inf, const std::vector<double>& x,
                              int d_in, int d_out, int target_level) {
    if (is_cachemir(inf.packing)) return cachemir::encode_linear_input(inf, x, d_in, d_out, target_level);
    if (is_diagonal(inf.packing) || is_cachemir_filling(inf.packing))
        return diagonal::encode_linear_input(inf, x, d_in, d_out, target_level);
    throw std::runtime_error("encode_linear_input: unsupported packing");
}

std::vector<double> decode_linear_output(const Packing& packing,
                                         const std::vector<double>& cy,
                                         int slots, int d_in, int d_out) {
    if (is_cachemir(packing)) return cachemir::decode_linear_output(slots, cy, d_in, d_out);
    if (is_diagonal(packing) || is_cachemir_filling(packing))
        return diagonal::decode_linear_output(slots, cy, d_in, d_out);
    throw std::runtime_error("decode_linear_output: unsupported packing");
}

std::vector<std::vector<double>> decode_tokens(const Packing& packing,
                                               const std::vector<double>& cy,
                                               int slots, int d_pad, int d_real, int T) {
    if (is_cachemir(packing)) return cachemir::decode_tokens(slots, cy, d_pad, d_real, T);
    if (is_diagonal(packing) || is_cachemir_filling(packing))
        return diagonal::decode_tokens(slots, cy, d_pad, d_real, T);
    throw std::runtime_error("decode_tokens: unsupported packing");
}

std::vector<Ptx> encode_weight_matrix(Inference& inf,
                                       const std::vector<std::vector<double>>& W,
                                       int d_in, int d_out,
                                       int target_level) {
    if (is_cachemir(inf.packing)) return cachemir::encode_weight_matrix(inf, W, d_in, d_out, target_level);
    if (is_diagonal(inf.packing) || is_cachemir_filling(inf.packing))
        return diagonal::encode_weight_matrix(inf, W, d_in, d_out, target_level);
    throw std::runtime_error("encode_weight_matrix: unsupported packing");
}

std::vector<Ptx> encode_weight_matrix(Inference& inf,
                                       const std::vector<std::vector<double>>& W,
                                       int d_in, int d_out,
                                       int target_level,
                                       cudaStream_t stream) {
    if (is_cachemir(inf.packing)) return cachemir::encode_weight_matrix(inf, W, d_in, d_out, target_level, stream);
    if (is_diagonal(inf.packing) || is_cachemir_filling(inf.packing))
        return diagonal::encode_weight_matrix(inf, W, d_in, d_out, target_level, stream);
    throw std::runtime_error("encode_weight_matrix: unsupported packing");
}

std::vector<Ptx> encode_weight_matrix_complex(Inference& inf,
                                              const std::vector<std::vector<double>>& W_re,
                                              const std::vector<std::vector<double>>& W_im,
                                              int d_in, int d_out,
                                              int target_level, cudaStream_t stream) {
    if (is_cachemir(inf.packing))
        return cachemir::encode_weight_matrix_complex(inf, W_re, W_im, d_in, d_out, target_level, stream);
    throw std::runtime_error("encode_weight_matrix_complex: cachemir-only");
}

Ptx encode_bias_vector(Inference& inf, const std::vector<double>& b,
                        int d_in, int d_out, bool fill) {
    if (is_cachemir(inf.packing)) return cachemir::encode_bias_vector(inf, b, d_in, d_out, fill);
    if (is_diagonal(inf.packing) || is_cachemir_filling(inf.packing))
        return diagonal::encode_bias_vector(inf, b, d_in, d_out, fill);
    throw std::runtime_error("encode_bias_vector: unsupported packing");
}

Ptx encode_bias_vector(Inference& inf, const std::vector<double>& b,
                        int d_in, int d_out, bool fill,
                        cudaStream_t stream) {
    if (is_cachemir(inf.packing)) return cachemir::encode_bias_vector(inf, b, d_in, d_out, fill, stream);
    if (is_diagonal(inf.packing) || is_cachemir_filling(inf.packing))
        return diagonal::encode_bias_vector(inf, b, d_in, d_out, fill, stream);
    throw std::runtime_error("encode_bias_vector: unsupported packing");
}

Ptx encode_bias_vector_complex(Inference& inf, const std::vector<double>& b_re,
                               const std::vector<double>& b_im, int d_in, int d_out,
                               bool fill, cudaStream_t stream) {
    if (is_cachemir(inf.packing))
        return cachemir::encode_bias_vector_complex(inf, b_re, b_im, d_in, d_out, fill, stream);
    throw std::runtime_error("encode_bias_vector_complex: cachemir-only");
}

std::vector<Ptx> encode_weight_matrix_outputpack(Inference& inf,
                                                 const std::vector<std::vector<double>>& W,
                                                 int d_in, int d_out, int target_level,
                                                 cudaStream_t stream) {
    if (is_cachemir(inf.packing))
        return cachemir::encode_weight_matrix_outputpack(inf, W, d_in, d_out, target_level, stream);
    throw std::runtime_error("encode_weight_matrix_outputpack: cachemir-only");
}

PackedCtx linear_outputpack(Inference& inf, const PackedCtx& x,
                            const std::string& wname, int d_in, int d_out) {
    if (is_cachemir(inf.packing))
        return cachemir::linear_outputpack(inf, x, wname, d_in, d_out);
    throw std::runtime_error("linear_outputpack: cachemir-only");
}

std::vector<Ptx> load_weight_txt(Inference& inf, const std::string& path,
                                  int d_in, int d_out, int target_level) {
    if (is_cachemir(inf.packing)) return cachemir::load_weight_txt(inf, path, d_in, d_out, target_level);
    if (is_diagonal(inf.packing) || is_cachemir_filling(inf.packing))
        return diagonal::load_weight_txt(inf, path, d_in, d_out, target_level);
    throw std::runtime_error("load_weight_txt: unsupported packing");
}
