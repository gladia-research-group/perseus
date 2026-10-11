"""One GPT-2 block (src/model/gpt2/gpt2_block.cu with src/model/mlp.cu):
ln_1 (folded: shift only) -> MHA -> residual -> ln_2 (folded) -> up -> GELU -> down ->
residual."""
from __future__ import annotations

from perseus.impl.attention import mha
from perseus.impl.activation import gelu
from perseus.impl.linear import linear
from perseus.impl.norm import norm, ln_shift, ln_affine


def transformer_block(rt, x, w, kv, cfgs, pos: int, last: bool = False):
    """``w``: BlockWeights (q,k,v,out,up,down EncodedLinear; shift1/shift2 or the unfolded
    gamma/beta); ``cfgs``: (ln_1, ln_2, softmax, gelu) configs; ``pos``: token position;
    ``last``: the model's final block (its exit feeds the tail, not another block)."""
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
        # mlp.cu (the output-pack unpack costs no level). QKV_LANES: the hint fires only past the limit plus one
        # unit, so the up-projection runs on the LayerNorm output as is and the plan refreshes its OUTPUT at the
        # GELU entry, at the top of the envelope
        ops.bootstrap_hint(x, ops.headroom(-1 if ops.qkv_lanes else 1), True)
        x = linear(rt, x, w.up)
    with rt.step("gelu"):
        x = gelu(rt, x, ge)
    with rt.step("down"):
        ops.bootstrap_hint(x, ops.headroom(1), True)
        x = linear(rt, x, w.down)
    out = ops.add(x, skip)
    if ops.qkv_lanes and not last:
        # QKV_LANES: the residual stream leaves the block at or under headroom(3) (40 primes on n32), so the next
        # block's LayerNorm output, six primes deeper, enters its qkv linear at or under the session limit and the
        # push refresh covers it with no hint. The planner reads the cap (lvl_cap), the runtime passes through. Not
        # on the last block: its exit feeds the tail, whose plan is cheaper from the deeper entry (+0.05 s/token
        # planned otherwise).
        ops.fhe.level_hint(out, ops.headroom(3))
    return out
