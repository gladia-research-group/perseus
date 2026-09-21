#include "packing/cachemir/cachemir_rot_indices.h"
#include "packing/cachemir/cachemir_linear_utils.h"

#include <set>

namespace cachemir {

static void collect_linear_rots(std::set<int32_t>& rots, int N, int d_in, int d_out) {
    auto p = compute_cm_params(N, d_in, d_out);

    // Input accumulation: step * (t - 1), step = 1,2,4,... while step < tp_in
    for (int step = 1; step < p.tp_in; step *= 2)
        rots.insert(step * (p.t - 1));

    // Input rotation: j * t^2, j = 1..r_i-1
    int rot2 = p.t * p.t;
    for (int j = 1; j < p.r_i; ++j)
        rots.insert(j * rot2);

    // Cascade rotation: t * tp
    rots.insert(p.t * p.tp);

    // Output accumulation: step = 1,2,4,... while step < tp_out
    for (int step = 1; step < p.tp_out; step *= 2)
        rots.insert(step);
}

static void collect_norm_rots(std::set<int32_t>& rots, int N, int hidDim) {
    const int t = N / hidDim;
    for (int s = t; s < N; s *= 2)
        rots.insert(s);
    for (int g = 1; g < N; g *= 2)
        rots.insert(g);
}

static void collect_mha_rots(std::set<int32_t>& rots, int N, int hidDim, int numHeads) {
    int t  = N / hidDim;
    int tH = t * numHeads;
    int d_head = hidDim / numHeads;

    // cache_k_push: rotate key into token slot
    for (int i = 1; i < t; ++i)
        rots.insert(-i);

    // filling->cachemir handoff (extract_token_i_cachemir): rotate a filling V/K
    for (int i = 1; i < t; ++i)
        rots.insert(i);

    // qkt query fill: replicate across token slots
    for (int step = 1; step < t; step *= 2)
        rots.insert(-step);

    // qkt sum_by_rot: reduce across dimension blocks
    for (int s = tH; s < N; s *= 2)
        rots.insert(s);

    // head_reduce_sum: intra-head masked rotations
    for (int step = 1; step < t; step *= 2) {
        rots.insert(step);
        rots.insert(step - t);  // wrap-around rotation
    }

    // softmax_v: direct score rotations by i*tH for each cached V lane
    for (int i = 1; i < d_head; ++i)
        rots.insert(i * tH);

    // softmax_v: intra-token reduction (same family as head_reduce_sum, included for clarity)
    for (int step = 1; step < t; step *= 2)
        rots.insert(step);
}

std::vector<int32_t> linear_rot_indices(int N, int d_in, int d_out) {
    std::set<int32_t> rots;
    collect_linear_rots(rots, N, d_in, d_out);
    return std::vector<int32_t>(rots.begin(), rots.end());
}

std::vector<int32_t> compute_gpt2_rot_indices(
    int slots, int hidDim, int ffDim, int numHeads) {
    std::set<int32_t> rots;

    // q/k/v/out: hidDim x hidDim
    collect_linear_rots(rots, slots, hidDim, hidDim);
    // up/gate: hidDim x ffDim
    collect_linear_rots(rots, slots, hidDim, ffDim);
    // down: ffDim x hidDim
    collect_linear_rots(rots, slots, ffDim, hidDim);
    // lm_head
    collect_linear_rots(rots, slots, hidDim, slots);

    // LayerNorm (ln_1 / ln_2 / ln_f) feature-axis mean + variance reductions.
    collect_norm_rots(rots, slots, hidDim);

    // MHA: KCache, qkt, head reduces, softmax-v lanes
    collect_mha_rots(rots, slots, hidDim, numHeads);

    return std::vector<int32_t>(rots.begin(), rots.end());
}

}  // namespace cachemir
