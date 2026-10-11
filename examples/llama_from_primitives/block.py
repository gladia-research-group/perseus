"""One Llama block on perseus.impl:
RMSNorm (gamma in the next linears) -> GQA attention with RoPE (K/V expanded to the query heads,
the rotary folded into the q / K selection masks) -> residual -> RMSNorm -> gate, up (one
prepared input) -> SiLU(gate) * up -> down -> residual."""
from __future__ import annotations

from perseus.impl.activation import gelu
from perseus.impl.attention import mha
from perseus.impl.linear import linear, linear_multi
from perseus.impl.norm import rms_norm

import os

# LLAMA_LINEAR_BOTTOM=1: every projection runs at the bottom of the chain (its input dropped to one
# level above the refresh limit), so its weight plaintexts are encoded at the fewest limbs; the
# projection's output is refreshed where it lands
LINEAR_BOTTOM = os.environ.get("LLAMA_LINEAR_BOTTOM", "0") not in ("", "0")


def _bottom(rt):
    if not LINEAR_BOTTOM:
        return None
    ops = rt.ops
    return lambda x: ops.level_drop(x, ops.level_limit() - ops.unit)


def _tap(rt, name, ct):
    """Debug hook: rt.tap(name, ct) when the model installed one (model.taps)."""
    f = getattr(rt, "tap", None)
    if f is not None:
        f(name, ct)


def llama_block(rt, x, w, kv, cfgs, pos: int, rope):
    """``w``: weights.BlockWeights; ``cfgs``: (ln_1, ln_2, softmax, silu) configs; ``pos``: the
    token's absolute position (the BOS prefix holds 0)."""
    ops = rt.ops
    ln1, ln2, sm, act = cfgs
    skip = x
    with rt.step("ln_1"):
        x = rms_norm(rt, x, ln1, pos)
    _tap(rt, "ln_1", x)
    pre = _bottom(rt)
    x = mha(rt, x, w, kv, sm, rope=rope, pos=pos, pre_linear=pre)
    x = ops.add(x, skip)
    _tap(rt, "attn_res", x)
    skip = x
    with rt.step("ln_2"):
        x = rms_norm(rt, x, ln2, pos)
    with rt.step("gate_up"):
        ops.bootstrap_hint(x, ops.headroom(1), True)
        if pre is not None:
            x = pre(x)
        g, u = linear_multi(rt, x, [w.gate, w.up])
    with rt.step("silu"):
        # the THOR composite x (p2(p1(x / xmax)) + 1/2) of activation.gelu with the SiLU's
        # coefficients; the gate carries 1/xmax (GELU_FOLD), the mask zeroes the padded lanes
        a = gelu(rt, g, act)
    _tap(rt, "gate", g); _tap(rt, "silu", a); _tap(rt, "up", u)
    with rt.step("glu"):
        h = ops.mult(a, u)
    _tap(rt, "glu", h)
    with rt.step("down"):
        ops.bootstrap_hint(h, ops.headroom(1), True)
        if pre is not None:
            h = pre(h)
        x = linear(rt, h, w.down)
    return ops.add(x, skip)
