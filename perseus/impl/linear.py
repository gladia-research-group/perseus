"""The cachemir linear y = x @ W (src/algorithms/linear/cachemir/cachemir_linear.cu):
a BSGS over the interleaved packing whose weight plaintexts come from
layout.encode_weight_matrix. Hoisted rotations become plain rotations; plaintexts are
cached once per level (mult_pt with a key: the named-plaintext path)."""
from __future__ import annotations

import dataclasses

import numpy as np

from .layout import cm_params, encode_bias_vector, encode_weight_matrix, encode_weight_matrix_outputpack


@dataclasses.dataclass
class EncodedLinear:
    """A (d_in, d_out) weight as its n_pt slot vectors plus an optional bias slot vector."""
    d_in: int
    d_out: int
    pts: np.ndarray                  # (n_pt, N)
    bias: np.ndarray | None = None   # (N,)
    name: str | None = None          # plaintext cache prefix (None: encode at every use)

    outputpack: bool = False         # plaintexts pair output blocks (apply_linear_outputpack)

    @classmethod
    def encode(cls, W, N: int, d_in: int, d_out: int, bias=None, fill=True, name=None,
               outputpack=False):
        """A real or complex W (W_re + i W_im: the fused K+iV linear) as its slot vectors;
        `outputpack` pairs output blocks into one complex plaintext instead, with the
        unpack's 1/2 folded in (apply_linear_outputpack's unpack then costs no level)."""
        pts = (encode_weight_matrix_outputpack if outputpack else encode_weight_matrix)(W, N, d_in, d_out)
        if outputpack:
            pts = 0.5 * pts
        b = None if bias is None else encode_bias_vector(bias, N, d_in, d_out, fill)
        return cls(d_in, d_out, pts, b, name, outputpack)

    def key(self, i):
        return None if self.name is None else f"{self.name}.{i}"


def prepare_linear_input(rt, x, d_in: int, d_out: int):
    """cachemir_linear.cu: replicate ladder, baby-step rotations, realized rescale."""
    ops = rt.ops
    p = cm_params(rt.dims.N, d_in, d_out)
    if p.tp_in == 1:
        x = ops.copy(x)             # the C++ clones before the in-place ladder
    else:
        x = ops.rotate_and_sum(x, p.t - 1, p.tp_in * (p.t - 1))   # gaps step*(t-1), step < tp_in
    rot2 = p.t * p.t
    x_rot = [x] + ops.rotate_many(x, [b * rot2 for b in range(1, p.bstep_c)])   # rotate_hoisted
    for xr in x_rot:
        ops.realize(xr)
    return x_rot


def apply_linear(rt, x_rot, w: EncodedLinear):
    """cachemir_linear.cu."""
    ops = rt.ops
    p = cm_params(rt.dims.N, w.d_in, w.d_out)
    giant_rot = p.bstep_c * p.t * p.t
    cy = []
    for k in range(p.r_o):
        acc = None
        for g in range(p.gstep_c):
            j0 = g * p.bstep_c
            i0 = j0 * p.r_o + k
            tmp = ops.mult_pt(x_rot[0], w.pts[i0], key=w.key(i0))
            for b in range(1, p.bstep_c):
                i = (j0 + b) * p.r_o + k
                ops.inplace_add(tmp, ops.mult_pt(x_rot[b], w.pts[i], key=w.key(i)))
            if g > 0:
                tmp = ops.rotate(tmp, g * giant_rot)
            if acc is None:
                acc = tmp
            else:
                ops.inplace_add(acc, tmp)
        cy.append(acc)
    cascade_rot = p.t * p.tp
    for k in range(p.r_o - 1, 0, -1):
        ops.inplace_add(cy[k - 1], ops.rotate(cy[k], cascade_rot))
    y = cy[0]
    if p.tp_out > 1:
        y = ops.rotate_and_sum(y, 1, p.tp_out)
    if w.bias is not None:
        y = ops.add_pt(y, w.bias, key=None if w.name is None else w.name + ".bias")
    return y


def apply_linear_outputpack(rt, x_rot, w: EncodedLinear):
    """apply_linear_outputpack (cachemir_linear.cu): the BSGS over r_o/2 complex
    plaintexts, each accumulator unpacked into its two real output blocks before the cascade.
    The plaintexts carry the 1/2 (EncodedLinear.encode), so P = (A + iB)/2 unpacks as
    A = P + conj, B = i (conj - P): one conjugate, no constant product, no level."""
    ops = rt.ops
    p = cm_params(rt.dims.N, w.d_in, w.d_out)
    rop = p.r_o // 2
    giant_rot = p.bstep_c * p.t * p.t
    cy = [None] * p.r_o
    for kp in range(rop):
        acc = None
        for g in range(p.gstep_c):
            j0 = g * p.bstep_c
            i0 = j0 * rop + kp
            tmp = ops.mult_pt(x_rot[0], w.pts[i0], key=w.key(i0))
            for b in range(1, p.bstep_c):
                i = (j0 + b) * rop + kp
                ops.inplace_add(tmp, ops.mult_pt(x_rot[b], w.pts[i], key=w.key(i)))
            if g > 0:
                tmp = ops.rotate(tmp, g * giant_rot)
            if acc is None:
                acc = tmp
            else:
                ops.inplace_add(acc, tmp)
        # unpack_ri of acc = (A + iB)/2 (the 1/2 is in the weights): A = acc + conj,
        # B = i (conj - acc); one conjugate, no constant product, no level
        conj = ops.conjugate(acc)
        cy[2 * kp] = ops.add(acc, conj)
        cy[2 * kp + 1] = ops.mult_i(ops.sub(conj, acc))
    cascade_rot = p.t * p.tp
    for k in range(p.r_o - 1, 0, -1):
        ops.inplace_add(cy[k - 1], ops.rotate(cy[k], cascade_rot))
    y = cy[0]
    if p.tp_out > 1:
        y = ops.rotate_and_sum(y, 1, p.tp_out)
    if w.bias is not None:
        y = ops.add_pt(y, w.bias, key=None if w.name is None else w.name + ".bias")
    return y


def linear(rt, x, w: EncodedLinear):
    x_rot = prepare_linear_input(rt, x, w.d_in, w.d_out)
    return apply_linear_outputpack(rt, x_rot, w) if w.outputpack else apply_linear(rt, x_rot, w)


def linear_multi(rt, x, ws):
    """linear.cu: one prepared input, one output per weight (K, V, Q share it)."""
    prep = prepare_linear_input(rt, x, ws[0].d_in, ws[0].d_out)
    return [apply_linear_outputpack(rt, prep, w) if w.outputpack else apply_linear(rt, prep, w)
            for w in ws]
