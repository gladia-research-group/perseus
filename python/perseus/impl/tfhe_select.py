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
* Native top-k (PF_SELECT_NATIVE=1, `_topk_native`): the maximum, threshold search and mask as native TFHE programs
  (impl/tfhe.py `Prog`); PF_SELECT_NAT_DRAW=1 also the Gumbel-max (`_prog_gumbel`).
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


def msign(ops, d, n, heads, R, levels, tag, res=None, pow2=False):
    """[d >= 0] on the slots `heads` (0 elsewhere) for |d| < R: `levels` scales in one switch (module doc). Under
    PF_SELECT_SIGN=ckks:<r> a composite polynomial at resolution max(r, `res`) instead (the baseline). `pow2`: the
    groups at slot offsets 0 and 2^j >= the heads' span (the rotations the tournaments already hold keys for)."""
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
    offs = [g * h for g in range(len(groups))]
    if pow2:
        b0 = 1 << int(np.ceil(np.log2(int(heads.max()) + 1)))
        offs = [0] + [b0 << j for j in range(len(groups) - 1)]
    if len(groups) > 1 and offs[-1] + int(heads.max()) >= n:
        raise ValueError(f"msign {tag}: {len(groups)} groups do not fit the heads (gap {gap}, span {h})")
    terms, const, slots = [], np.zeros(n), []
    rots = ops.rotate_many(d, [-o for o in offs])
    for g, ((sc, c), r) in enumerate(zip(groups, rots)):
        terms.append(ops.mult_pt(r, _mask(n, heads + offs[g], sc), kind="mask"))
        const[heads + offs[g]] = c
        slots.append(heads + offs[g])
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
    back = [S if g == 0 else ops.rotate(S, offs[g]) for g in range(len(groups))]
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


def native():
    """PF_SELECT_NATIVE=1: top-k's maximum, threshold search and mask as native TFHE programs."""
    import os
    return os.environ.get("PF_SELECT_NATIVE", "0") not in ("", "0")


NAT_S = 128.0       # _prog_max: the logits' distance to block 0's maximum per native unit (|m - m_0| < NAT_S / 2)
DEBUG: list = []    # the native path's last intermediates (m, u, max u, the search's input)
NAT_PAD = 8.0       # _prog_search: the window's room for _prog_max's error (its relu rounds' noise)


def _tree(pr, src, rows, bufs, until=1):
    """The sums of `src`'s first `rows` rows in `until` contiguous groups (rows / until a power of 2) into rows
    0 .. until - 1 of bufs[0] or bufs[1] (which is returned)."""
    cur, i = src, 0
    while rows > until:
        h = (rows + 1) // 2
        pr.lin(bufs[i], cur, cur, [(2 * r, 2 * r + 1 if 2 * r + 1 < rows else -1) for r in range(h)])
        cur, rows, i = bufs[i], h, 1 - i
    return cur


def nat_count():
    """PF_SELECT_NAT_COUNT: _prog_search's counts, `exact` (default: staircase-rounded group counts) or `plain`."""
    import os
    return os.environ.get("PF_SELECT_NAT_COUNT", "exact") or "exact"


def nat_draw():
    """PF_SELECT_NAT_DRAW=1 (with PF_SELECT_NATIVE): the Gumbel-max over the members as one native program."""
    import os
    return os.environ.get("PF_SELECT_NAT_DRAW", "0") not in ("", "0")


def nat_tau():
    """PF_SELECT_NAT_TAU: the native draw scales y by this, cancelling its comparison noise's temperature
    (softmax(y / sqrt(1 + 6 sigma^2 / pi^2)))."""
    import os
    return float(os.environ.get("PF_SELECT_NAT_TAU", "1.08") or 1.08)


NAT_GROUP = 32      # exact counts: rows per group (its count read by a staircase at 1.9 / NAT_GROUP per count)
NAT_TAU = 0.06      # exact counts: the first level's sure band (|e| > TAU), the second reads 16 e


def _prog_max(L, out):
    """max_j u_j of the loaded u = (m - m_0) / NAT_S (|u| < 1/2) on the one-slot set `out`: a relu tournament. One
    store row: a broadcast stored row by row would read the maximum with every row's own noise."""
    name = f"tk.nmax.{L}"
    pr = tfhe.Prog(name, 6)
    pr.lut("id", lambda m: m)
    pr.lut("relu", lambda m: np.maximum(m, 0.0))
    pr.lut("out", lambda m: m, warp=True)
    pr.load("id", 0)
    cur, rows, nxt = 0, L, 3
    while rows > 1:
        h = rows // 2
        pr.lin(1, cur, cur, [(2 * r + 1, 2 * r) for r in range(h)], ca=1, cb=-1)
        pr.pbs(1, h, "relu", [2])
        pr.lin(nxt, cur, 2, [(2 * r, r) for r in range(h)])
        cur, rows, nxt = nxt, h, (7 - nxt)
    pr.store(cur, 1, "out", out)
    return tfhe.register_prog(pr)


def _prog_search(L, k, W, probes, slots, offset=0.0):
    """The mask [y >= lo + offset] of the loaded u = y 1.9 / D - 0.95 (y in [0, D), D = W + NAT_PAD): `probes` halvings of
    the threshold's lower end lo, count(y >= lo + w) >= k raising it by w; the count's comparison is e = a (count - k +
    1/2), in (-2, 2) for any count. Output: +-1/2 (the caller adds 1/2)."""
    D = W + NAT_PAD
    exact = nat_count() == "exact"
    G = NAT_GROUP
    kc = min(G, k)                    # exact: a group's count capped at k (the capped sum reaches k iff the count does)
    a = 1.9 / ((L // G * kc if exact else L) - k + 0.5)
    a1 = 1.9 / G if exact else a
    name = f"tk.nsel.{L}.{k}.{W:g}.{probes}.{nat_count()}" + (f".{offset:.4g}" if offset else "")
    pr = tfhe.Prog(name, 11)
    pr.lut("id", lambda m: m)
    pr.lut("bit", lambda m: np.full_like(m, -a1 / 2))
    pr.lut("out", lambda m: np.full_like(m, -0.5), warp=True)
    if exact:
        pr.lut("stair", lambda m: a * (np.minimum(np.rint(m / a1) + G / 2, kc) - G / 2))   # a (min(count, k) - G / 2)
        pr.lut("half", lambda m: np.full_like(m, -0.25))                # bits +-1/4
    pr.load("id", 0)
    pr.lin(5, 0, 0, [(-1, -1)])                                       # lo = 0 (u units: lo 1.9 / D)
    w = W
    for j in range(probes):
        w /= 2
        wu = w * 1.9 / D
        pr.lut(f"step{j}", lambda m, wu=wu: np.full_like(m, -wu / 2))
        # p = u - u(lo + w) + 1: the sign's input
        pr.lin(1, 0, 5, [(r, 0) for r in range(L)], ca=1, cb=-1, c=1.0 - (wu - 0.95))
        pr.pbs(1, L, "bit", [2])
        if not exact:
            t = _tree(pr, 2, L, (3, 4))
            # e + 1 = a count - a (k - 1/2) + 1, the count = sum / a + L / 2
            pr.lin(6, t, t, [(0, -1)], c=a * L / 2 - a * (k - 0.5) + 1.0)
            pr.pbs(6, 1, f"step{j}", [1])                            # +-wu / 2
        else:
            t = _tree(pr, 2, L, (3, 4), until=L // G)
            pr.pbs(t, L // G, "stair", [7])
            t = _tree(pr, 7, L // G, (3, 4))
            off = a * (L / 2 - (k - 0.5))                            # e = sum + off
            pr.lin(8, t, t, [(0, -1)], c=off - NAT_TAU + 1.0)
            pr.lin(9, t, t, [(0, -1)], c=off + NAT_TAU + 1.0)
            pr.lin(10, t, t, [(0, -1)], ca=16, c=16 * off + 1.0)
            pr.lin(6, 8, 9, [(0, -1), (-1, 0)])
            pr.lin(8, 6, 10, [(0, -1), (1, -1), (-1, 0)])
            pr.pbs(8, 3, "half", [9])                                 # +-1/4 each: the sum is -3/4 .. 3/4
            pr.lin(6, 9, 9, [(0, 1)])
            pr.lin(6, 6, 9, [(0, 2)], c=1.0)
            pr.pbs(6, 1, f"step{j}", [1])                            # [two or three of p, u, b]: +-wu / 2
        pr.lin(5, 5, 1, [(0, 0)], c=wu / 2)
    pr.lin(1, 0, 5, [(r, 0) for r in range(L)], ca=1, cb=-1, c=1.0 + 0.95 - offset * 1.9 / D)
    pr.store(1, L, "out", slots)
    return tfhe.register_prog(pr)


def nat_arity():
    """PF_SELECT_NAT_ARITY: the native threshold search's split per step (2: binary; 3, 5 (default), 9:
    `_prog_search_kary`); PF_SELECT_NAT_STEPS: its steps (default: the binary search's resolution)."""
    import os
    return int(os.environ.get("PF_SELECT_NAT_ARITY", "5") or 5), int(os.environ.get("PF_SELECT_NAT_STEPS", "0") or 0)


def _prog_search_kary(L, k, W, probes, slots, offset=0.0):
    """`_prog_search` with arity - 1 = kt thresholds per step read by one many-LUT rotation (exact counts), lo raised
    by gs grid units per threshold whose count reaches k."""
    arity, steps = nat_arity()
    kt = arity - 1
    D = W + NAT_PAD
    G = NAT_GROUP
    kc = min(G, k)
    a = 1.9 / (L // G * kc - k + 0.5)
    a1 = 1.9 / G
    dg = 2.0 * kt / tfhe.GRID                                         # a table's grid unit
    g = int(round(W * 1.9 / D / dg))                                  # the window in grid units
    if not steps:
        steps = max(1, int(np.ceil(np.log(2.0 ** probes) / np.log(arity))))
    name = f"tk.nsk.{L}.{k}.{W:g}.{arity}.{steps}" + (f".{offset:.4g}" if offset else "")
    Y, P, B0, T0, T1, PK, ST, E0, E1, E2, E3, E4, LO, M = 0, 1, 2, 2 + kt, 3 + kt, 4 + kt, 5 + kt, 6 + kt, 7 + kt, \
        8 + kt, 9 + kt, 10 + kt, 11 + kt, 12 + kt
    pr = tfhe.Prog(name, 13 + kt)
    pr.lut("id", lambda m: m)
    pr.lut("out", lambda m: np.full_like(m, -0.5), warp=True)
    pr.lut("stair", lambda m: a * (np.minimum(np.rint(m / a1) + G / 2, kc) - G / 2))
    pr.lut("half", lambda m: np.full_like(m, -0.25))
    pr.load("id", Y)
    pr.lin(LO, Y, Y, [(-1, -1)])
    off = a * (L / 2 - (k - 0.5))
    for j in range(steps):
        gs = max(1, -(-g // arity))
        c0 = (kt + 1) // 2 * gs * dg                                  # the thresholds centred in the tables' (-1, 1)
        th = [l * gs * dg - c0 for l in range(1, kt + 1)]
        pr.lut(f"q{j}", *[lambda m, t=t: a1 * (np.where(m >= t - dg / 2, 1.0, 0.0) - 0.5) for t in th])
        pr.lut(f"step{j}", lambda m, d=gs * dg: np.full_like(m, -d / 2))
        pr.lin(P, Y, LO, [(r, 0) for r in range(L)], ca=1, cb=-1, c=0.95 - c0)
        pr.pbs(P, L, f"q{j}", list(range(B0, B0 + kt)))
        for l in range(kt):                                           # the group counts of threshold l at rows 32 l..
            t = _tree(pr, B0 + l, L, (T0, T1), until=L // G)
            n0 = l * (L // G)
            pr.lin(PK, PK if l else t, t, [(r, -1) for r in range(n0)] + [(-1, r) for r in range(L // G)])
        pr.pbs(PK, kt * (L // G), "stair", [ST])
        t = _tree(pr, ST, kt * (L // G), (T0, T1), until=kt)          # row l: threshold l's capped count
        pr.lin(E0, t, t, [(r, -1) for r in range(kt)], c=off - NAT_TAU + 1.0)
        pr.lin(E1, t, t, [(r, -1) for r in range(kt)], c=off + NAT_TAU + 1.0)
        pr.lin(E2, t, t, [(r, -1) for r in range(kt)], ca=16, c=16 * off + 1.0)
        pr.lin(E3, E0, E1, [(r, -1) for r in range(kt)] + [(-1, r) for r in range(kt)])
        pr.lin(E4, E3, E2, [(r, -1) for r in range(2 * kt)] + [(-1, r) for r in range(kt)])
        pr.pbs(E4, 3 * kt, "half", [E0])
        pr.lin(E1, E0, E0, [(l, kt + l) for l in range(kt)])
        pr.lin(E1, E1, E0, [(l, 2 * kt + l) for l in range(kt)], c=1.0)
        pr.pbs(E1, kt, f"step{j}", [M])                               # +-gs / 2 per threshold
        t = _tree(pr, M, kt, (T0, T1))
        pr.lin(LO, LO, t, [(0, 0)], c=kt * gs * dg / 2)
        g = gs
    # the mask at the steps' effective threshold (lo - dg / 2) less PF_SELECT_NAT_MARGIN logits
    import os
    offset -= float(os.environ.get("PF_SELECT_NAT_MARGIN", "") or (0.1 if arity == 3 else 0.2))
    pr.lin(P, Y, LO, [(r, 0) for r in range(L)], ca=1, cb=-1, c=1.0 + 0.95 + dg / 2 - offset * 1.9 / D)
    pr.store(P, L, "out", slots)
    return tfhe.register_prog(pr)


NAT_DZ = 56.0       # _prog_gumbel: the scores' span (mask (y + g + C) in [0, NAT_DZ))


def _prog_gumbel(L, slots):
    """The leaf one-hot of the maximum of the loaded x = z 0.95 / NAT_DZ - 0.475 (z = mask (y + g + C)): forward
    relu rounds keeping each decision s = [x_b >= x_a], then from the root w = b the children (w (1 - s), w s) by
    one two-table bootstrap of 2 w + s. Output: 0 / 1."""
    b = 0.3
    rounds = int(np.log2(L))
    name = f"tk.ngum.{L}"
    X0, X1, D, R, M, LF, RT, W0, W1 = range(9)
    S = 9                                              # decision buffers S .. S + rounds - 1
    pr = tfhe.Prog(name, S + rounds)
    pr.lut("id", lambda m: m)
    pr.lut("relu", lambda m: np.maximum(m, 0.0))
    pr.lut("bit", lambda m: np.full_like(m, -b / 2))
    pr.lut("kids", lambda m: np.where((m >= 1.5 * b) & (m < 2.5 * b), b, 0.0), lambda m: np.where(m >= 2.5 * b, b, 0.0))
    pr.lut("out", lambda m: np.where(m >= b / 2, 1.0, 0.0), warp=True)
    pr.load("id", X0)
    cur, nxt, rows = X0, X1, L
    for r in range(rounds):
        h = rows // 2
        pr.lin(D, cur, cur, [(2 * i + 1, 2 * i) for i in range(h)], ca=1, cb=-1)
        pr.pbs(D, h, "relu", [R])
        pr.lin(M, D, D, [(i, -1) for i in range(h)], c=1.0)
        pr.pbs(M, h, "bit", [S + r])                     # +-b / 2 (the caller of each bit adds b / 2)
        pr.lin(nxt, cur, R, [(2 * i, i) for i in range(h)])
        cur, nxt, rows = nxt, cur, h
    w, wn = W0, W1
    pr.lin(w, w, w, [(-1, -1)], c=b)
    rows = 1
    for r in reversed(range(rounds)):
        pr.lin(M, w, S + r, [(i, i) for i in range(rows)], ca=2, cb=1, c=b / 2)
        pr.pbs(M, rows, "kids", [LF, RT])
        pr.lin(wn, LF, RT, [p for i in range(rows) for p in ((i, -1), (-1, i))])
        w, wn, rows = wn, w, 2 * rows
    pr.store(w, L, "out", slots)
    return tfhe.register_prog(pr)


def _topk_native(ops, m, n, heads, *, k, L, W, probes, R, topp=None):
    """`topk_sample`'s y (clipped) and mask from the block maxima `m` (on the heads) by two native programs (`topp`:
    y only, mask None)."""
    B = n // L
    hm = _mask(n, heads)
    sl = tfhe.register_slots(f"tk.heads.{L}", heads)
    m0 = _bcast(ops, ops.mult_pt(m, _mask(n, [0]), kind="mask"), B, n)
    u = ops.mult_pt(ops.sub(m, m0), _mask(n, heads, 1.0 / NAT_S), kind="mask")
    mx = _bcast(ops, tfhe.tfhe_prog(ops, u, _prog_max(L, tfhe.register_slots("tk.slot0", [0])), sl), B, n)
    y = ops.add_pt(ops.mult_pt(ops.sub(u, mx), _mask(n, heads, NAT_S), kind="mask"), _mask(n, heads, W), kind="mask")
    y = ops.mult(y, msign(ops, y, n, heads, R, 1, "tk.clip"))
    if topp is not None:
        return y, None
    D = W + NAT_PAD
    uy = ops.add_pt(ops.mult_pt(y, _mask(n, heads, 1.9 / D), kind="mask"), _mask(n, heads, -0.95), kind="mask")
    search = _prog_search_kary if nat_arity()[0] > 2 else _prog_search
    prog = search(L, k, W, probes, sl)
    mask = ops.add_pt(tfhe.tfhe_prog(ops, uy, prog, sl), _mask(n, heads, 0.5), kind="mask")
    DEBUG[:] = [m, u, mx, uy]
    return y, mask


def nuc_offs():
    """PF_SELECT_TOPP_K (1, 2, 4: default): top-p's thresholds per step from one many-LUT bootstrap; a head's bits at
    the slot offsets (0, 1, 2, 4)[:k]."""
    import os
    return (0, 1, 2, 4)[:int(os.environ.get("PF_SELECT_TOPP_K", "4") or 4)]


def _exp_ckks(ops, x, lo, hi, sq=5, deg=10):
    """exp(x) for x in [lo, hi]: a Chebyshev series of exp(x / 2^sq), squared sq times (relative error ~2^sq times
    the series')."""
    from perseus.impl.poly import eval_chebyshev
    a, b = lo / 2 ** sq, hi / 2 ** sq
    c = np.polynomial.chebyshev.Chebyshev.interpolate(np.exp, deg, domain=[a, b]).coef
    y = eval_chebyshev(ops, ops.mult(x, 1.0 / 2 ** sq), list(c), a, b)
    for _ in range(sq):
        y = ops.square(y)
    return y


def _prog_bits(L, th, slots):
    """Bits [p >= theta_l] - 1/2 of the loaded p for the k = 1, 2, 4 thresholds `th` (one many-LUT load, the
    finish's encoding: stored as they are) on the slot set heads + nuc_offs() (row k i + l)."""
    k = len(th)
    dg = 2.0 * k / tfhe.GRID
    name = "tp.bits." + ".".join(f"{t:.5f}" for t in th)
    pr = tfhe.Prog(name, 7)
    pr.lut("b", *[lambda m, t=t: np.where(m >= t - dg / 2, 0.5, -0.5) for t in th], warp=True)
    pr.load("b", list(range(k)))
    out = 0
    if k >= 2:
        pr.lin(4, 0, 1, [p for i in range(L) for p in ((i, -1), (-1, i))])
        out = 4
    if k == 4:
        pr.lin(5, 2, 3, [p for i in range(L) for p in ((i, -1), (-1, i))])
        pr.lin(6, 4, 5, [p for i in range(L) for p in ((2 * i, -1), (2 * i + 1, -1), (-1, 2 * i), (-1, 2 * i + 1))])
        out = 6
    pr.store(out, k * L, None, slots)
    return tfhe.register_prog(pr)


def _topp_mask(ops, y, n, heads, *, p, W):
    """Top-p (nucleus; Holtzman et al., arXiv 1904.09751) over the block maxima: the masks [y >= t] for the largest t
    with mass(y >= t) >= p Z (e = exp(y - W), Z = sum e), by (k + 1)-way steps (`_prog_bits`, `nuc_offs`)."""
    L = len(heads)
    B = n // L
    offs = nuc_offs()
    D = W + NAT_PAD
    sc = 1.9 / D
    kt = len(offs)
    dg = 2.0 * kt / tfhe.GRID
    hm = _mask(n, heads)
    e = ops.mult_pt(_exp_ckks(ops, ops.add_pt(y, _mask(n, heads, -W) + (1.0 - hm) * -W, kind="mask"), -W, NAT_PAD),
                    hm, kind="mask")
    Z = ops.mult_pt(ops.rotate_and_sum(e, B, n), _mask(n, [0]), kind="mask")
    zr = [Z] + ops.rotate_many(Z, [-o for o in offs[1:]])
    pZ = zr[0]
    for v in zr[1:]:
        pZ = ops.add(pZ, v)
    pZ = ops.mult(pZ, float(p))
    er = [e] + ops.rotate_many(e, [-o for o in offs[1:]])
    erep = er[0]
    for v in er[1:]:
        erep = ops.add(erep, v)
    bits_set = tfhe.register_slots(f"tp.bits.{L}.{kt}", (heads[:, None] + np.asarray(offs)[None, :]).ravel())
    sl = tfhe.register_slots(f"tk.heads.{L}", heads)
    small = np.asarray(offs)
    g = int(round(W * sc / dg))                                       # the window in grid units
    lo = None                                                        # slot 0: the threshold's lower end (y units)
    while g > 1:
        gs = -(-g // (kt + 1))
        du = gs * dg
        c0 = (kt + 1) // 2 * du
        th = [l * du - c0 for l in range(1, kt + 1)]
        lob = None if lo is None else _bcast(ops, lo, B, n)
        dy = y if lob is None else ops.sub(y, lob)
        u = ops.add_pt(ops.mult_pt(dy, _mask(n, heads, sc), kind="mask"), _mask(n, heads, -c0), kind="mask")
        bits = ops.add_pt(tfhe.tfhe_prog(ops, u, _prog_bits(L, th, bits_set), sl),
                          _mask(n, (heads[:, None] + small[None, :]).ravel(), 0.5), kind="mask")
        mass = ops.rotate_and_sum(ops.mult(erep, bits), B, n)
        dm = ops.mult_pt(ops.sub(mass, pZ), _mask(n, small), kind="mask")
        b = msign(ops, dm, n, small, 2.0 ** 13, 5, f"tp.m{gs}", pow2=True)
        rs = [b] + ops.rotate_many(b, list(offs[1:]))
        cnt = rs[0]
        for v in rs[1:]:
            cnt = ops.add(cnt, v)
        step = ops.mult_pt(cnt, _mask(n, [0], gs * dg / sc), kind="mask")
        lo = step if lo is None else ops.add(lo, step)
        g = gs
    # the mask at the steps' effective threshold (lo - dg / 2) less PF_SELECT_TOPP_MARGIN logits: a bit read past its
    # bucket could lift lo past the top one (an empty nucleus)
    import os
    margin = float(os.environ.get("PF_SELECT_TOPP_MARGIN", "") or 0.35 * kt)
    lob = _bcast(ops, lo, B, n)
    dt = ops.add_pt(ops.sub(y, lob), _mask(n, heads, dg / 2 / sc + margin), kind="mask")
    return msign(ops, dt, n, heads, W + 1.0, 1, "tk.mask")


def topk_sample(ops, xs, valid, g, *, k, L, W=32.0, probes=8, R=128.0, fine=512, C=4.0, minp=None, topp=None):
    """The one-hot (K tiles) of a top-k sample: Gumbel-max over the top-k maxima of the L slot blocks of size B = n / L
    (module doc). `g`: Gumbel noise at the slots j B (a ciphertext, or a slot vector used as a plaintext). Returns
    (one-hot tiles, mask over the block heads j B). `minp`: min-p sampling instead (Nguyen et al., arXiv 2407.01082):
    the maxima within ln(minp) of the largest (-ln(minp) < W). `topp`: top-p sampling instead (`_topp_mask`; needs the
    native session). Both leave `k` unused."""
    n = len(valid[0])
    K, B = len(xs), n // L
    m, dec = tournament(ops, xs, valid, upto=B, R=R, fine=fine, tag="tk")
    e0, heads = _mask(n, [0]), np.arange(0, n, B)
    hm = _mask(n, heads)
    if native() and minp is None:
        y, mask = _topk_native(ops, m, n, heads, k=k, L=L, W=W, probes=probes, R=R, topp=topp)
        if topp is not None:
            mask = _topp_mask(ops, y, n, heads, p=topp, W=W)
        return _topk_draw(ops, y, mask, g, dec, hm, n, heads, K, C, W, fine)
    mx, _ = tournament(ops, [m], [np.ones(n)], first=B, R=R, fine=fine, tag="tkm")
    M = _bcast(ops, ops.mult_pt(mx, e0, kind="mask"), B, n)
    y = ops.add_pt(ops.mult_pt(ops.sub(m, M), hm, kind="mask"), _mask(n, heads, W), kind="mask")
    y = ops.mult(y, msign(ops, y, n, heads, R, 1, "tk.clip"))
    if minp is not None:                 # y = W at the maximum: no search
        mask = msign(ops, ops.add_pt(y, _mask(n, heads, -(W + float(np.log(minp)))), kind="mask"), n, heads, W + 1.0, 1,
                     "tk.mask")
        return _topk_draw(ops, y, mask, g, dec, hm, n, heads, K, C, W, fine)
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
    return _topk_draw(ops, y, mask, g, dec, hm, n, heads, K, C, W, fine)


def _topk_draw(ops, y, mask, g, dec, hm, n, heads, K, C, W, fine):
    """The Gumbel-max over the members of `mask` (`topk_sample`)."""
    B = n // len(heads)
    if native() and nat_draw():          # the native draw's noise acts as a temperature
        y = ops.mult_pt(y, _mask(n, heads, nat_tau()), kind="mask")
    if isinstance(g, np.ndarray):        # the noise as a plaintext (public server randomness, encoded per token)
        z = ops.mult(mask, ops.add_pt(y, np.where(hm > 0, g + C, 0.0), kind="mask"))
    else:                                # g masked first: product + product (a product plus a fresh input is a degree
        # the planner and the runtime disagree on)
        z = ops.mult(mask, ops.add_pt(ops.add(y, ops.mult_pt(g, hm, kind="mask")), _mask(n, heads, C), kind="mask"))
    if native() and nat_draw():
        L = len(heads)
        zx = ops.add_pt(ops.mult_pt(z, _mask(n, heads, 0.95 / NAT_DZ), kind="mask"), _mask(n, heads, -0.475), kind="mask")
        sl = tfhe.register_slots(f"tk.heads.{L}", heads)
        wb = tfhe.tfhe_prog(ops, zx, _prog_gumbel(L, sl), sl)
        return onehot(ops, dec, wb, n, K), mask
    _, dz = tournament(ops, [z], [np.ones(n)], first=B, R=W + 24.0, fine=fine, tag="tks")
    wb = onehot(ops, dz, _root(ops, z, n, [0]), n, 1)[0]
    return onehot(ops, dec, wb, n, K), mask
