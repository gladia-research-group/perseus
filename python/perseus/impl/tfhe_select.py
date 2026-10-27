"""Argmax, top-k and top-k sampling over a slot vector by TFHE sign LUTs (impl/tfhe.py): a decode head's selection
without a polynomial sign's gap.

The vector: tiles `xs` of n real slots, tile 0 full, tile k > 0 valid on `valid[k]` (the GPT-2 logit tiles).

* Tournament (`tournament`): the tiles fold into tile 0 (one round each), then rounds at strides 1, 2, 4, ... compare
  slot i with slot i + h (i = 0 mod 2h): d = x[i + h] - x[i], s = [d >= 0] (one switch on the n / 2h heads),
  x += d s. After the round at stride h the maxima of the slot blocks of 2h sit at their first slots. n - 1 +
  sum |valid[k]| sign PBS for the maximum; the decisions are kept for the one-hot. Contiguous blocks, not strided
  ones: in the LM head's layout a block's tokens are ~1024 ids apart, while a stride class shares the id mod 64 and
  the frequent (low-id) tokens would collide in a few buckets.
* Signs (`msign`): a sign LUT reads d s_0 on (-2, 2) (|d| < R: no wrap). Late rounds (few heads) read more scales in
  the same switch, as extra slots: level j says "sure" when |d s_j| > TAU and hands the rest to level j + 1 at
  s_{j+1} = s_j / (2 TAU / 1.9) (~16x finer); s = p_0 + u_0 (p_1 + u_1 (... f)), p_j = [d s_j >= TAU],
  u_j = [d s_j >= -TAU] - p_j. 2 J + 1 PBS per comparison, the input noise / 16^J. Each s is cleaned as 3 s^2 - 2 s^3
  (the finish's ~1e-3 error times |d| would bias the running maxima).
* One-hot (`onehot`): the decisions backwards from the root: w <- w (1 - s) + rot(w, -h) rot(s, -h) (products of
  rotations, not rotations of products: the planner's degree rule).
* Top-k (`topk_sample`): approximate top-k over the maxima of L slot blocks (the scheme of JAX approx_max_k: recall ~
  1 - k / 2L), exact on them: M = their max, y = (m - M + W) [m - M + W >= 0] (W > the k-th's distance to the max),
  a binary search for the threshold t (count(y >= t) >= k: `probes` switches on L slots, the count compared to k by
  a 3-level sign on one slot), the mask [y >= t]; sampling = the Gumbel-max over the members: argmax mask (y + g + C)
  with g Gumbel(0, 1) per bucket (i.i.d. per member, as per token; public server randomness: a plaintext re-encoded
  per token, or a ciphertext).
"""
from __future__ import annotations

import hashlib

import numpy as np

from perseus.impl import tfhe

SPAN = 1.9          # |d s_0| <= SPAN: the sign LUT's (-2, 2) with room for the noise
TAU = 0.06          # a level's "sure" band (m units; the switch's input noise is ~5.3e-3 rms)
RHO = SPAN / (2 * TAU)


def sign01():
    return tfhe.register_sign("sign01", 0.0, 1.0)


def _mask(n, idx, v=1.0):
    out = np.zeros(n)
    out[np.asarray(idx, dtype=np.int64)] = v
    return out


def _clean(ops, s, n, heads):
    """3 s^2 - 2 s^3 on the heads (0 and 1 are fixed points; the error e -> 3 e^2). PF_SELECT_CLEAN=0: skipped."""
    import os
    if os.environ.get("PF_SELECT_CLEAN", "1") in ("0", ""):
        return s
    s2 = ops.square(s)
    return ops.mult(s2, ops.add_pt(ops.mult_pt(s, _mask(n, heads, -2.0), kind="mask"), _mask(n, heads, 3.0), kind="mask"))


# ── the CKKS baseline: the same selection with a composite polynomial sign (PF_SELECT_SIGN=ckks:<res>) ──────────────
#
# sign(x) on [-1, 1] \ (-eps, eps) as g3^a then f3^b (Cheon, Kim, Kim, Lee, Lee, arXiv 2005.00466: f3 the degree-7
# odd polynomial with f'(1) flat to order 3, g3 its fast-growing companion), the fewest iterations reaching 2^-10 on
# [eps, 1]; eps = res / R, `res` the decision resolution (d units) asked of every sign, the count's 0.5 at least.
_F3 = [0.0, 35 / 16, 0.0, -35 / 16, 0.0, 21 / 16, 0.0, -5 / 16]
_G3 = [0.0, 4589 / 1024, 0.0, -16577 / 1024, 0.0, 25614 / 1024, 0.0, -12860 / 1024]
_COMP = {}


def ckks_res():
    """The CKKS sign's resolution (PF_SELECT_SIGN=ckks:<res>), or None (the TFHE sign)."""
    import os
    v = os.environ.get("PF_SELECT_SIGN", "")
    return float(v.split(":", 1)[1]) if v.startswith("ckks") else None


def hybrid():
    """PF_SELECT_SIGN=hybrid:<pbs>:<res>[:<res2>]: a sign over at least <pbs> heads runs as the CKKS composite at
    resolution <res> instead (a CKKS round costs about the same at any width, a switch ~32 us per value: the big
    early rounds are cheaper in CKKS, the small late ones in TFHE); <res2>: the smaller ones too, as CKKS at that
    resolution (the hybrid's all-CKKS control). None otherwise."""
    import os
    v = os.environ.get("PF_SELECT_SIGN", "")
    if not v.startswith("hybrid"):
        return None
    f = v.split(":")
    return int(f[1]), float(f[2]), (float(f[3]) if len(f) > 3 else None)


def _composite(eps):
    """(g3 iterations, f3 iterations) for sign on [eps, 1] to 2^-10."""
    key = round(float(np.log2(eps)), 3)
    if key not in _COMP:
        x = np.linspace(eps, 1.0, 100001)
        P = np.polynomial.polynomial.polyval
        best = None
        for a in range(16):
            y = x.copy()
            for _ in range(a):
                y = P(y, _G3)
            for b in range(1, 8):
                y = P(y, _F3)
                if np.abs(1.0 - y).max() < 2.0 ** -10:
                    if best is None or a + b < sum(best):
                        best = (a, b)
                    break
        _COMP[key] = best
    return _COMP[key]


def _ckks_sign01(ops, d, n, heads, R, res):
    from perseus.impl.poly import eval_chebyshev
    cheb = np.polynomial.chebyshev.poly2cheb
    a, b = _composite(min(res / R, 0.5))
    x = ops.mult_pt(d, _mask(n, heads, 1.0 / R), kind="mask")
    for c in [_G3] * a + [_F3] * b:
        x = eval_chebyshev(ops, x, list(cheb(c)), -1.0, 1.0)
    return ops.add_pt(ops.mult_pt(x, _mask(n, heads, 0.5), kind="mask"), _mask(n, heads, 0.5), kind="mask")


def _ckks_sign01_const(ops, e, R, res):
    """[e >= 0] of an all-slot constant (the reduction stamp kept: its refreshes take the 1-slot route), by the
    composite: scalar products only."""
    from perseus.impl.poly import eval_chebyshev
    cheb = np.polynomial.chebyshev.poly2cheb
    a, b = _composite(min(res / R, 0.5))
    x = ops.mult(e, 1.0 / R)
    for c in [_G3] * a + [_F3] * b:
        x = eval_chebyshev(ops, x, list(cheb(c)), -1.0, 1.0)
    return ops.add(ops.mult(x, 0.5), 0.5)


def count_ckks():
    """PF_SELECT_COUNT=ckks: top-k's count comparisons as CKKS signs on an all-slot constant (1-slot refreshes)
    instead of a switch each (whose finish is a dense refresh)."""
    import os
    return os.environ.get("PF_SELECT_COUNT", "") == "ckks"


def msign(ops, d, n, heads, R, levels, tag, res=None):
    """[d >= 0] on the slots `heads` (0 elsewhere) for |d| < R: `levels` scales in one switch (module doc). Under
    PF_SELECT_SIGN=ckks:<r> a composite polynomial at resolution max(r, `res`) instead (the baseline)."""
    heads = np.asarray(heads, dtype=np.int64)
    r = ckks_res()
    if r is not None:
        return _ckks_sign01(ops, d, n, heads, R, max(r, res or 0.0))
    hy = hybrid()
    if hy is not None and len(heads) >= hy[0]:
        return _ckks_sign01(ops, d, n, heads, R, max(hy[1], res or 0.0))
    if hy is not None and hy[2] is not None:         # the all-CKKS control of a hybrid: the small signs at hy[2]
        return _ckks_sign01(ops, d, n, heads, R, max(hy[2], res or 0.0))
    groups = []                          # (scale, offset const, slot offset)
    s = SPAN / R
    for j in range(levels):
        if j + 1 < levels:
            groups += [(s, -TAU), (s, TAU)]
            s *= RHO
        else:
            groups += [(s, 0.0)]
    # the groups' slot offset: 1 when the heads are spaced at least the group count apart, else the span of heads
    # starting at slot 0
    gap = int(np.diff(heads).min()) if len(heads) > 1 else n
    h = 1 if gap >= len(groups) else int(heads.max()) + 1
    if len(groups) > 1 and h > 1 and len(groups) * h > n:
        raise ValueError(f"msign {tag}: {len(groups)} groups do not fit the heads (gap {gap}, span {h})")
    terms, const, slots = [], np.zeros(n), []
    rots = ops.rotate_many(d, [-g * h for g in range(len(groups))])
    for g, ((sc, c), r) in enumerate(zip(groups, rots)):
        terms.append(ops.mult_pt(r, _mask(n, heads + g * h, sc), kind="mask"))
        const[heads + g * h] = c
        slots.append(heads + g * h)
    D = terms[0]
    for t in terms[1:]:
        D = ops.add(D, t)
    if np.any(const):
        D = ops.add_pt(D, const, kind="mask")
    # named by content: equal sets (a search's probes) share one registration (the session holds 2^16 positions)
    idx = np.concatenate(slots)
    name = f"sel.{len(idx)}.{hashlib.blake2b(idx.astype(np.int64).tobytes(), digest_size=6).hexdigest()}"
    S = tfhe.tfhe_lut(ops, D, sign01(), tfhe.register_slots(name, idx))
    if len(groups) == 1:
        return _clean(ops, S, n, heads)
    m = _mask(n, heads)
    back = [S if g == 0 else ops.rotate(S, g * h) for g in range(len(groups))]
    b = [ops.mult_pt(x, m, kind="mask") for x in back]
    acc = b[-1]
    for j in range(levels - 2, -1, -1):
        p, q = b[2 * j], b[2 * j + 1]              # [d s_j >= TAU], [d s_j >= -TAU]
        acc = ops.add(p, ops.mult(ops.sub(q, p), acc))
    return _clean(ops, acc, n, heads)


def tournament(ops, xs, valid, *, first=1, upto=None, R=128.0, fine=512, dec=None, tag="t"):
    """Rounds at strides first, 2 first, ... < upto (default n): the maxima of the slot blocks of size `upto` at their
    first slots, and the decisions (module doc). `first` > 1: continue a vector already folded to blocks of `first`
    (no tile round)."""
    n = len(valid[0])
    upto = n if upto is None else int(upto)
    dec = [] if dec is None else dec
    x = xs[0]
    if first == 1:
        for k in range(1, len(xs)):
            heads = np.nonzero(valid[k])[0]
            d = ops.sub(xs[k], x)
            s = msign(ops, d, n, heads, R, 1, f"{tag}.tile{k}")
            x = ops.add(x, ops.mult(d, s))
            dec.append(("tile", k, s))
    h = int(first)
    while h < upto:
        heads = np.arange(0, n, 2 * h)
        d = ops.sub(ops.rotate(x, h), x)
        s = msign(ops, d, n, heads, R, 2 if len(heads) <= fine else 1, f"{tag}.h{h}")
        x = ops.add(x, ops.mult(d, s))
        dec.append(("stride", h, s))
        h *= 2
    return x, dec


def onehot(ops, dec, w, n, K):
    """The leaf indicator of the root indicator `w` (on the last tournament's block heads): K tiles."""
    tiles = [None] * K
    for kind, h, s in reversed(dec):
        if kind == "stride":
            w = ops.add(ops.mult(w, ops.add_pt(ops.negate(s), np.ones(n), kind="mask")),
                        ops.mult(ops.rotate(w, -h), ops.rotate(s, -h)))
        else:
            tiles[h] = ops.mult(w, s)
            w = ops.sub(w, tiles[h])
    tiles[0] = w
    return tiles


def _root(ops, ref, n, idx):
    """A ciphertext of 1 on the slots `idx` (0 elsewhere): ref - ref + mask (the planner's stage has no free constants)."""
    return ops.add_pt(ops.mult_pt(ref, np.zeros(n), kind="mask"), _mask(n, idx), kind="mask")


def argmax(ops, xs, valid, *, R=128.0, fine=512):
    """The one-hot (K tiles) of the argmax of the valid slots."""
    n = len(valid[0])
    x, dec = tournament(ops, xs, valid, R=R, fine=fine, tag="am")
    return onehot(ops, dec, _root(ops, x, n, [0]), n, len(xs))


def _bcast(ops, x, B, n):
    """Slot 0 to the slots j B (x zero elsewhere)."""
    return ops.rotate_and_sum(x, -B, n)


def topk_sample(ops, xs, valid, g, *, k, L, W=32.0, probes=8, R=128.0, fine=512, C=4.0):
    """The one-hot (K tiles) of a top-k sample: Gumbel-max over the top-k maxima of the L slot blocks of size B = n / L
    (module doc). `g`: Gumbel noise at the slots j B (a ciphertext, or a slot vector used as a plaintext). Returns
    (one-hot tiles, mask over the block heads j B)."""
    n = len(valid[0])
    K, B = len(xs), n // L
    m, dec = tournament(ops, xs, valid, upto=B, R=R, fine=fine, tag="tk")
    mx, _ = tournament(ops, [m], [np.ones(n)], first=B, R=R, fine=fine, tag="tkm")
    e0, heads = _mask(n, [0]), np.arange(0, n, B)
    M = _bcast(ops, ops.mult_pt(mx, e0, kind="mask"), B, n)
    hm = _mask(n, heads)
    y = ops.add_pt(ops.mult_pt(ops.sub(m, M), hm, kind="mask"), _mask(n, heads, W), kind="mask")
    y = ops.mult(y, msign(ops, y, n, heads, R, 1, "tk.clip"))
    # the threshold's lower end: public start 0, width W / 2^j; at slot 0 (a switch per count), or in every slot
    # (PF_SELECT_COUNT=ckks: an all-slot constant, no broadcast)
    const = count_ckks()
    lo = None
    lo_at = (lambda v: v) if const else (lambda v: _bcast(ops, v, B, n))
    w = W
    for j in range(probes):
        w /= 2
        t = _mask(n, heads, w)
        dt = ops.add_pt(y, -t, kind="mask") if lo is None else ops.sub(ops.add_pt(y, -t, kind="mask"), lo_at(lo))
        s = msign(ops, dt, n, heads, W + 1.0, 1, f"tk.p{j}")
        if const:
            cnt = ops.rotate_and_sum(s, 1, n)                         # every slot: count(y >= lo + w)
            if hasattr(ops, "tag_reduce"):                            # the stamp (the mirror has none)
                cnt = ops.tag_reduce(cnt, 1)
            b = _ckks_sign01_const(ops, ops.add(cnt, 0.5 - k), 2.0 * L, 0.25)
            step = ops.mult(b, w)
        else:
            cnt = ops.rotate_and_sum(s, B, n)                         # slot 0: count(y >= lo + w)
            e = ops.add_pt(ops.mult_pt(cnt, e0, kind="mask"), _mask(n, [0], 0.5 - k), kind="mask")
            b = msign(ops, e, n, [0], 2.0 * L, 3, f"tk.c{j}", res=0.25)   # [count >= k]: raise lo by w
            step = ops.mult_pt(b, _mask(n, [0], w), kind="mask")
        lo = step if lo is None else ops.add(lo, step)
    mask = msign(ops, ops.sub(y, lo_at(lo)), n, heads, W + 1.0, 1, "tk.mask")
    if isinstance(g, np.ndarray):        # the noise as a plaintext (public server randomness, encoded per token)
        z = ops.mult(mask, ops.add_pt(y, np.where(hm > 0, g + C, 0.0), kind="mask"))
    else:                                # g masked first: product + product (a product plus a fresh input is a degree
        # the planner and the runtime disagree on)
        z = ops.mult(mask, ops.add_pt(ops.add(y, ops.mult_pt(g, hm, kind="mask")), _mask(n, heads, C), kind="mask"))
    _, dz = tournament(ops, [z], [np.ones(n)], first=B, R=W + 24.0, fine=fine, tag="tks")
    wb = onehot(ops, dz, _root(ops, z, n, [0]), n, 1)[0]
    return onehot(ops, dec, wb, n, K), mask
