"""One decoder block on perseus.impl (arch.Arch: Llama, Gemma-2):
RMSNorm (gamma in the next linears) -> GQA attention with RoPE (K/V expanded to the query heads,
the rotary folded into the q / K selection masks; the logit soft-cap on the scores when the Arch has one)
[-> post RMSNorm] -> residual -> RMSNorm -> gate, up (one prepared input) -> act(gate) * up -> down
[-> post RMSNorm] -> residual. A post norm's gamma rides rms_norm's input product (no level)."""
from __future__ import annotations

from perseus.impl.activation import gelu
from perseus.impl.attention import mha, mha_gqa
from perseus.impl.linear import linear, linear_multi
from perseus.impl.norm import rms_norm
from perseus.impl.poly import eval_chebyshev, eval_chebyshev_masked

from .weights import mlp_mask

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


def softcap_map(sc):
    """The attention score map of a logit soft-cap (config.SoftcapCfg): cap tanh(s / cap) as its Chebyshev series in s
    on [lo, hi]. The map's output is alpha f(s), alpha the SM_FOLD scale (the exp's shift beta rides the score mask the
    softmax adds after the map), so alpha rides the coefficients. Its input arrives as `in_scale(alpha)` s: the q.K^T
    group masks carry 1 / hi on a symmetric interval, so the series runs on [-1, 1] with no affine level (as SM_FOLD
    folds the exp's input map), else alpha. `mask` (mult, add_mask): the series times a plaintext mask at its own depth
    (poly.eval_chebyshev_masked; attention._centre_mapped). `scale`: the scores come as scale s instead (the prefill's
    q.K^T: 1); the output is alpha f(s) all the same."""
    sym = sc.lo == -sc.hi and sc.hi > 0

    def in_scale(alpha):
        return 1.0 / sc.hi if sym else alpha

    def f(rt, scores, alpha, beta, mask=None, scale=None):
        if alpha <= 0:
            raise ValueError("softcap_map: a non-positive fold scale")
        k = in_scale(alpha) if scale is None else scale     # `scale`: the scores' own (the prefill's: 1)
        c = tuple(alpha * v for v in sc.coeffs)
        lo, hi = (-1.0, 1.0) if sym and scale is None else (k * sc.lo, k * sc.hi)   # exactly: no affine product
        if mask is not None:
            return eval_chebyshev_masked(rt.ops, scores, c, lo, hi, None, mask[0], mask[1], leaf_terms="budget")
        return eval_chebyshev(rt.ops, scores, c, lo, hi)
    f.in_scale = in_scale
    return f


def llama_block(rt, x, w, kv, cfgs, pos: int, rope, last: bool = False, gqa=None, post=None, softcap=None,
                act_step: str = "silu", k0=None):
    """``w``: weights.BlockWeights; ``cfgs``: (ln_1, ln_2, softmax, act) configs; ``pos``: the
    token's absolute position (the BOS prefix holds 0); ``last``: the model's final block (its exit feeds the
    tail, not another block). ``post``: the (attention, MLP) output norms' configs (Arch.post_norms; gammas w.post1 /
    w.post2); ``softcap``: the attention logit soft-cap's config; ``act_step``: the activation's step name; ``k0``:
    key 0's (the BOS's) K slot vector, for a softmax centred after the soft-cap (attention.mha)."""
    ops = rt.ops
    ln1, ln2, sm, act = cfgs
    smap = None if softcap is None or sm.cap_fold else softcap_map(softcap)     # cap_fold: in the exp's series
    skip = x
    with rt.step("ln_1"):
        x = rms_norm(rt, x, ln1, pos)
    _tap(rt, "ln_1", x)
    if gqa is not None:        # (sub-runtime, G): LLAMA_GQA
        x = mha_gqa(rt, gqa[0], gqa[1], x, w, kv, sm, rope=rope, pos=pos, pre_linear=_bottom(rt, "qkv"),
                    pre_out=_bottom(rt, "out_proj"), score_map=smap)
    else:
        x = mha(rt, x, w, kv, sm, rope=rope, pos=pos, pre_linear=_bottom(rt, "qkv"), pre_out=_bottom(rt, "out_proj"),
                score_map=smap, k0=k0)
    if post is not None:
        with rt.step("ln_1p"):
            x = rms_norm(rt, x, post[0], pos, gamma=w.post1, tag=f"{w.tag}post1")
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
        if w.mlp:       # LLAMA_MLP_SPLIT: each power-of-two slice of the MLP its own projections (weights.mlp_parts)
            gu = [linear_multi(rt, x, [prt.gate, prt.up]) for prt in w.mlp]
        else:
            gu = [linear_multi(rt, x, [w.gate, w.up])]
    with rt.step(act_step):
        # the THOR composite x (p2(p1(x / xmax)) + 1/2) of activation.gelu with the activation's
        # coefficients; the gate carries 1/xmax (GELU_FOLD), the mask zeroes the padded lanes (a slice's: its own)
        if w.mlp:
            acts = [gelu(rt, g, act, mask=(("mlp", prt.tag), lambda sc, prt=prt: mlp_mask(rt.dims, prt.width, prt.real, sc)))
                    for prt, (g, _u) in zip(w.mlp, gu)]
        else:
            acts = [gelu(rt, gu[0][0], act)]
    _tap(rt, "gate", gu[0][0]); _tap(rt, act_step, acts[0]); _tap(rt, "up", gu[0][1])
    with rt.step("glu"):
        hs = [ops.mult(a, u) for a, (_g, u) in zip(acts, gu)]
    _tap(rt, "glu", hs[0])
    with rt.step("down"):
        pre = _bottom(rt, "down")
        x = None
        for h, dn in zip(hs, [prt.down for prt in w.mlp] or [w.down]):
            ops.bootstrap_hint(h, ops.headroom(1), True)
            if pre is not None:
                h = pre(h)
            y = linear(rt, h, dn)
            x = y if x is None else ops.add(x, y)
    if post is not None:
        with rt.step("ln_2p"):
            x = rms_norm(rt, x, post[1], pos, gamma=w.post2, tag=f"{w.tag}post2")
    out = ops.add(x, skip)
    if ops.qkv_lanes and not last:
        # QKV_LANES: the GPT-2 block's exit cap
        ops.fhe.level_hint(out, ops.headroom(3))
    return out
