#pragma once

#include "inference.h"
#include "packing/packed_ctx.h"

#include <cuda_runtime.h>
#include <string>
#include <vector>

// Diagonal-packing parameters and weight/input encoding helpers for linear
// layers. Multi-token batched layout.

namespace diagonal {

struct DiagonalParams {
    int d_in, d_out;
    int t_in;        // = N / d_in   (input lane stride)
    int t_out;       // = N / d_out  (output lane stride)
    int alpha;       // = max(d_in, d_out) / min(d_in, d_out)
    bool is_up;      // (d_in < d_out)
    int n_diag;      // diagonals = d_in
    int s, G;        // BSGS split (n_diag = s * G), s = largest divisor ≤ √n_diag
    int max_n_tok;   // = min(t_in, t_out) — batch size cap
};

DiagonalParams compute_dg_params(int N, int d_in, int d_out);

// bench-mode rotation passthrough (matches cachemir::mha_rot semantics).
inline int dg_rot(const Inference& inf, int real_idx) {
    return inf.bench_mode ? 5 : real_idx;
}

// Encode a row-major (n_tok × d_in) matrix flattened into `x` (n_tok = x.size()/d_in).
// n_tok ∈ [1, max_n_tok]. For n_tok == 1 packs a single token; multi-token
// is the prefill / ViT batched case.
// `target_level` encrypts the input ct directly at the requested CKKS level
// (default 0 = full limbs). Useful to match a BTS-planner-determined level
// without bootstrap noise.
PackedCtx encode_linear_input(Inference& inf, const std::vector<double>& x,
                              int d_in, int d_out, int target_level = 0);

// BSGS pre-rotated d_in diagonals of W (d_in × d_out). Returns s*G plaintexts,
// flat; pt[b*G + g] = rot(diag_{g*s+b}, -g*s*t_in), where
//   diag_k[slot j*t_out + tok] = W[(input_idx_base(j) + k) mod d_in, j]
//   input_idx_base(j) = is_up ? (j / alpha) : (j * alpha)
// `target_level` controls how many CKKS RNS limbs each plaintext carries:
// level=0 (default) → all limbs (~9 MB/pt at logN=16, full-depth compatible);
// level=k         → (L-k) limbs (memory ∝ remaining limbs). Use the level
// the input ct will be at when the linear runs (e.g. post-bootstrap level)
// to minimize pt memory; the multiplication then has no level mismatch.
// Particularly important for diagonal down-proj (4096 pts at GPT-2 scale).
std::vector<Ptx> encode_weight_matrix(Inference& inf,
                                       const std::vector<std::vector<double>>& W,
                                       int d_in, int d_out,
                                       int target_level = 0);

std::vector<Ptx> encode_weight_matrix(Inference& inf,
                                       const std::vector<std::vector<double>>& W,
                                       int d_in, int d_out,
                                       int target_level,
                                       cudaStream_t stream);

Ptx encode_bias_vector(Inference& inf, const std::vector<double>& b,
                        int d_in, int d_out, bool fill = true);

Ptx encode_bias_vector(Inference& inf, const std::vector<double>& b,
                        int d_in, int d_out, bool fill,
                        cudaStream_t stream);

std::vector<Ptx> load_weight_txt(Inference& inf, const std::string& path,
                                  int d_in, int d_out,
                                  int target_level = 0);

// Inverse of the diagonal encode/eval slot layout: given a decrypted slot vector
// produced by a (d_in × d_out) linear layer, return a length-d_out plaintext.
// y[j] = cy[j * t_out] for both square and rectangular cases.
std::vector<double> decode_linear_output(int slots, const std::vector<double>& cy,
                                         int d_in, int d_out);

// Decode T token vectors from a square (d_pad) ciphertext: feature i, token tok at
// slot i*t_out + tok (t_out = slots/d_pad). Returns [T][d_real].
std::vector<std::vector<double>> decode_tokens(int slots, const std::vector<double>& cy,
                                               int d_pad, int d_real, int T);

}  // namespace diagonal
