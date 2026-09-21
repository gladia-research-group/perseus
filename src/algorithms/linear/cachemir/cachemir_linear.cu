#include "model/gpt2.h"
#include "inference.h"
#include "packing/cachemir/cachemir_attention_utils.h"
#include "packing/cachemir/cachemir_linear.h"
#include "packing/cachemir/cachemir_linear_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace cachemir {

static inline int rot(const Inference& inf, int real_idx) {
    return mha_rot(inf, real_idx);
}

void rotate_add_inplace(Inference& inf, PackedCtx& x, int step) {
    PackedCtx tmp = inf.fhe->rotate(x, rot(inf, step));
    inf.fhe->inplace_add(x, tmp);
}

std::vector<PackedCtx> prepare_linear_input(Inference& inf, const PackedCtx& x_in,
                                            int d_in, int d_out) {
    const auto p = compute_cm_params(inf.slots, d_in, d_out);
    // Populate the interleaved packing
    PackedCtx x = inf.fhe->clone(x_in);
    for (int step = 1; step < p.tp_in; step *= 2)
        rotate_add_inplace(inf, x, step * (p.t - 1));

    int rot2 = p.t * p.t;
    std::vector<PackedCtx> x_rotated(p.bstep_c);
    x_rotated[0] = x;

    if (p.bstep_c > 1) {
        std::vector<int32_t> steps;
        steps.reserve(p.bstep_c - 1);
        for (int b = 1; b < p.bstep_c; ++b) steps.push_back(rot(inf, b * rot2));
        std::vector<PackedCtx> rots = inf.fhe->rotate_hoisted(x, steps);
        for (int b = 1; b < p.bstep_c; ++b) x_rotated[b] = rots[b - 1];
    }
    // Realize the pending rescale on the rotated inputs once, before the pt-mult fan-out.
    if (!inf.graph_capture_enabled())
        for (auto& xr : x_rotated) inf.fhe->realize_pending_rescale_raw(xr.ct);

    return x_rotated;
}

PackedCtx apply_linear(Inference& inf, const std::vector<PackedCtx>& x_rotated,
                       const std::string& wname, int d_in, int d_out) {
    const auto p = compute_cm_params(inf.slots, d_in, d_out);
    auto pts_W = inf.weights_at(wname, x_rotated[0]);

    const int giant_rot = p.bstep_c * p.t * p.t;
    std::vector<PackedCtx> cy(p.r_o);
    for (int k = 0; k < p.r_o; ++k) {
        PackedCtx acc;
        for (int g = 0; g < p.gstep_c; ++g) {
            const int j0  = g * p.bstep_c;
            PackedCtx tmp = inf.fhe->mult(x_rotated[0], pts_W[(j0 + 0) * p.r_o + k]);
            for (int b = 1; b < p.bstep_c; ++b) {
                PackedCtx t2 = inf.fhe->mult(x_rotated[b], pts_W[(j0 + b) * p.r_o + k]);
                inf.fhe->inplace_add(tmp, t2);
            }
            if (g > 0) tmp = inf.fhe->rotate(tmp, rot(inf, g * giant_rot));
            if (g == 0) acc = std::move(tmp);
            else        inf.fhe->inplace_add(acc, tmp);
        }
        cy[k] = std::move(acc);
    }

    int cascade_rot = p.t * p.tp;
    for (int k = p.r_o - 1; k > 0; --k) {
        PackedCtx tmp = inf.fhe->rotate(cy[k], rot(inf, cascade_rot));
        inf.fhe->inplace_add(cy[k - 1], tmp);
    }

    PackedCtx y = cy[0];
    for (int step = 1; step < p.tp_out; step *= 2)
        rotate_add_inplace(inf, y, step);

    auto bias_it = inf.w.find(wname + "_bias");
    if (bias_it != inf.w.end() && !bias_it->second.empty()) {
        Ptx pt_at_y = inf.complex_weight_names.count(wname + "_bias")
            ? inf.encode_like_cached_complex(
                  inf.scoped(wname + "_bias"), y,
                  [&]{ return bias_it->second[0]->GetCKKSPackedValue(); })
            : inf.encode_like_cached(
                  inf.scoped(wname + "_bias"), y,
                  [&]{ return bias_it->second[0]->GetRealPackedValue(); });
        y = inf.fhe->add(y, pt_at_y);
    }

    return y;
}

PackedCtx linear(Inference& inf, const PackedCtx& x_in,
                 const std::string& wname, int d_in, int d_out) {
    return cachemir::apply_linear(inf,
                                  cachemir::prepare_linear_input(inf, x_in, d_in, d_out),
                                  wname, d_in, d_out);
}

PackedCtx apply_linear_outputpack(Inference& inf, const std::vector<PackedCtx>& x_rotated,
                                  const std::string& wname, int d_in, int d_out) {
    const auto p = compute_cm_params(inf.slots, d_in, d_out);
    if (p.r_o % 2 != 0)
        throw std::runtime_error("apply_linear_outputpack: r_o must be even");
    const int rop = p.r_o / 2;
    auto pts_W = inf.weights_at(wname, x_rotated[0]);

    const int giant_rot = p.bstep_c * p.t * p.t;
    const Ptx nhi = inf.encode_complex_const_at(0.0, -0.5, x_rotated[0]);   // -i/2 (Im extract)
    std::vector<PackedCtx> cy(p.r_o);
    for (int kp = 0; kp < rop; ++kp) {
        PackedCtx acc;
        for (int g = 0; g < p.gstep_c; ++g) {
            const int j0  = g * p.bstep_c;
            PackedCtx tmp = inf.fhe->mult(x_rotated[0], pts_W[(j0 + 0) * rop + kp]);
            for (int b = 1; b < p.bstep_c; ++b) {
                PackedCtx t2 = inf.fhe->mult(x_rotated[b], pts_W[(j0 + b) * rop + kp]);
                inf.fhe->inplace_add(tmp, t2);
            }
            if (g > 0) tmp = inf.fhe->rotate(tmp, rot(inf, g * giant_rot));
            if (g == 0) acc = std::move(tmp);
            else        inf.fhe->inplace_add(acc, tmp);
        }
        Ptx nhi_mut = nhi;

        WithStep _u(inf, "unpack_ri");
        auto ri = inf.fhe->unpack_ri(acc, nhi_mut);   // Re = block 2k', Im = block 2k'+1
        cy[2 * kp]     = std::move(ri.first);
        cy[2 * kp + 1] = std::move(ri.second);
    }

    int cascade_rot = p.t * p.tp;
    for (int k = p.r_o - 1; k > 0; --k) {
        PackedCtx tmp = inf.fhe->rotate(cy[k], rot(inf, cascade_rot));
        inf.fhe->inplace_add(cy[k - 1], tmp);
    }

    PackedCtx y = cy[0];
    for (int step = 1; step < p.tp_out; step *= 2)
        rotate_add_inplace(inf, y, step);

    auto bias_it = inf.w.find(wname + "_bias");
    if (bias_it != inf.w.end() && !bias_it->second.empty()) {
        Ptx pt_at_y = inf.encode_like_cached(
            inf.scoped(wname + "_bias"), y,
            [&]{ return bias_it->second[0]->GetRealPackedValue(); });
        y = inf.fhe->add(y, pt_at_y);
    }
    return y;
}

PackedCtx linear_outputpack(Inference& inf, const PackedCtx& x_in,
                            const std::string& wname, int d_in, int d_out) {
    return cachemir::apply_linear_outputpack(
        inf, cachemir::prepare_linear_input(inf, x_in, d_in, d_out), wname, d_in, d_out);
}

}  // namespace cachemir
