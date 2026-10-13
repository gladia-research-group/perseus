"""One Llama block on perseus.impl:
RMSNorm (gamma in the next linears) -> GQA attention with RoPE (K/V expanded to the query heads,
the rotary folded into the q / K selection masks) -> residual -> RMSNorm -> gate, up (one
prepared input) -> SiLU(gate) * up -> down -> residual."""
from __future__ import annotations

from perseus.impl.activation import gelu
from perseus.impl.attention import mha, mha_gqa
from perseus.impl.linear import linear, linear_multi
from perseus.impl.norm import rms_norm

import os

# LLAMA_LINEAR_BOTTOM=<steps>: these projections (qkv, out_proj, gate_up, down; 1 = all) run at the bottom of
# the chain; their outputs are refreshed where they land
_LB = os.environ.get("LLAMA_LINEAR_BOTTOM", "0")
LINEAR_BOTTOM = (set() if _LB in ("", "0") else {"qkv", "out_proj", "gate_up", "down"} if _LB == "1"
                 else set(_LB.split(",")))


def _bottom(rt, step):
    if step not in LINEAR_BOTTOM:
        return None
    ops = rt.ops
    # two units under the limit: the products must land under it
    return lambda x: ops.level_drop(x, ops.level_limit() - 2 * ops.unit)


def _tap(rt, name, ct):
    """Debug hook: rt.tap(name, ct) when the model installed one (model.taps)."""
    f = getattr(rt, "tap", None)
    if f is not None:
        f(name, ct)


def llama_block(rt, x, w, kv, cfgs, pos: int, rope, last: bool = False, gqa=None):
    """``w``: weights.BlockWeights; ``cfgs``: (ln_1, ln_2, softmax, silu) configs; ``pos``: the
    token's absolute position (the BOS prefix holds 0); ``last``: the model's final block (its exit feeds the
    tail, not another block)."""
    ops = rt.ops
    ln1, ln2, sm, act = cfgs
    skip = x
    with rt.step("ln_1"):
        x = rms_norm(rt, x, ln1, pos)
    _tap(rt, "ln_1", x)
    if gqa is not None:        # (sub-runtime, G): LLAMA_GQA
        x = mha_gqa(rt, gqa[0], gqa[1], x, w, kv, sm, rope=rope, pos=pos, pre_linear=_bottom(rt, "qkv"),
                    pre_out=_bottom(rt, "out_proj"))
    else:
        x = mha(rt, x, w, kv, sm, rope=rope, pos=pos, pre_linear=_bottom(rt, "qkv"), pre_out=_bottom(rt, "out_proj"))
    x = ops.add(x, skip)
    _tap(rt, "attn_res", x)
    skip = x
    with rt.step("ln_2"):
        x = rms_norm(rt, x, ln2, pos)
    with rt.step("gate_up"):
        # QKV_LANES: as the GPT-2 block's up projection
        ops.bootstrap_hint(x, ops.headroom(-1 if ops.qkv_lanes else 1), True)
        pre = _bottom(rt, "gate_up")
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
        pre = _bottom(rt, "down")
        if pre is not None:
            h = pre(h)
        x = linear(rt, h, w.down)
    out = ops.add(x, skip)
    if ops.qkv_lanes and not last:
        # QKV_LANES: the GPT-2 block's exit cap
        ops.fhe.level_hint(out, ops.headroom(3))
    return out
