"""One GPT-2 block (src/model/gpt2/gpt2_block.cu with src/model/mlp.cu):
ln_1 (folded: shift only) -> MHA -> residual -> ln_2 (folded) -> up -> GELU -> down ->
residual."""
from __future__ import annotations

from perseus.impl.attention import mha
from perseus.impl.activation import gelu
from perseus.impl.linear import linear
from perseus.impl.norm import norm, ln_shift, ln_affine


def transformer_block(rt, x, w, kv, cfgs, pos: int):
    """``w``: BlockWeights (q,k,v,out,up,down EncodedLinear; shift1/shift2 or the unfolded
    gamma/beta); ``cfgs``: (ln_1, ln_2, softmax, gelu) configs; ``pos``: token position."""
    ops = rt.ops
    ln1, ln2, sm, ge = cfgs
    skip = x
    with rt.step("ln_1"):
        n = norm(rt, x, ln1, pos)
        x = ln_shift(rt, n, w.shift1, w.tag + "ln_1") if w.shift1 is not None else \
            ln_affine(rt, n, w.gamma1, w.beta1, w.tag + "ln_1")
    x = mha(rt, x, w, kv, sm)
    x = ops.add(x, skip)
    skip = x
    with rt.step("ln_2"):
        n = norm(rt, x, ln2, pos)
        if w.shift2 is not None:
            x = ln_shift(rt, n, w.shift2, w.tag + "ln_2")
        else:
            ops.fhe.maybe_bootstrap(n)
            x = ln_affine(rt, n, w.gamma2, w.beta2, w.tag + "ln_2")
    with rt.step("up"):
        ops.bootstrap_hint(x, ops.headroom(1), True)   # mlp.cu (the output-pack unpack costs no level)
        x = linear(rt, x, w.up)
    with rt.step("gelu"):
        x = gelu(rt, x, ge)
    with rt.step("down"):
        ops.bootstrap_hint(x, ops.headroom(1), True)
        x = linear(rt, x, w.down)
    return ops.add(x, skip)
