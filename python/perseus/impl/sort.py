"""Sorting and selection on batched slot vectors (impl.stats.Batch: value j of vector b at slot j B + b, tiled).

Every algorithm is built on one primitive, the comparison `step(ops, d, heads, scale, ...)` = [scale d >= 0] on a set
of slots (0 elsewhere), by one of two back ends:

* ``tfhe``: the CKKS -> TFHE -> CKKS switch's sign LUT (impl/tfhe.py, as tfhe_select.msign: one PBS per compared
  value, exact at p_fail < 2^-128 per PBS with the default n 840 parameters, cleaned by 3 s^2 - 2 s^3). The switch is
  itself a refresh: its output lands fresh, so the compared data loses only the levels of the products around it.
  Cost grows with the number of values compared (~29 ms per 1024 values on the RTX PRO 6000, 2026-11-18).
  Decisions inside the switch's input noise band (rms ~5e-3 of the range) can go either way.
* ``ckks``: a composite minimax sign (impl.math.sign_ladder) at the asked resolution: one polynomial over the whole
  ciphertext, whatever the number of values, but deep (12-25 levels) and approximate (values within `res` of a tie
  are blended, not decided).

Algorithms (the comparisons each needs; the best back end differs by size, see the report):

* `vmax` / `vmin` / `argmax` / `argmin`: a tournament. CKKS: every slot at once (x <- max(x, rot(x, k B)) for
  k = 1, 2, .., n/2: log2 n signs, the maximum broadcast for free); TFHE: only the block heads (n - 1 PBS per vector
  in all), then a mask and a broadcast ladder. The one-hot is [x >= max - res / 2] (near-ties: several ones).
* `rank` (n <= sqrt(slots / 2B)): all pairs in one ciphertext (rows of 2n slots: x_i replicated along row i by a
  diagonal mask and two rotate-and-sum ladders, x_j already tiled along the columns): ONE comparison of n^2 values,
  then a ladder over the rows. CKKS: one sign for the whole ranking; TFHE: n (n - 1) PBS.
* From ranks: `topk_mask` ([rank >= n - k], one comparison), `kth` / `median` / `quantile` (rank indicators, two or
  three comparisons sharing one evaluation through the layout's tiled copies, a product and a sum), `sort_by_rank`
  (the scatter: two more n^2 comparisons).
* `bitonic` (any n): Batcher's bitonic network (log2 n (log2 n + 1) / 2 compare-exchange steps, each one comparison of
  the n/2 pairs at distance j, a product and two rotations), optionally carrying a payload (`argsort`);
  `rank_bitonic`: ranks for any n as the argsort of the argsort (two networks).
* `threshold` (any n): top-k / k-th value by a binary search on the value (the sampler's search, tfhe_select.
  topk_sample, generalised): per probe one comparison of the n values, a count ladder and a comparison of the count;
  the k-th value to (hi - lo) / 2^probes.
"""
from __future__ import annotations

import hashlib
import math as _m

import numpy as np

from . import math as M
from .stats import Batch


def _mask(n, idx, v=1.0):
    out = np.zeros(n)
    out[np.asarray(idx, dtype=np.int64)] = v
    return out


# ── the comparison ──────────────────────────────────────────────────────────────────────────────────────────────────


def _nslots(ops, x):
    n = getattr(getattr(ops, "inf", None), "slots", None)
    if n:
        return int(n)
    if isinstance(x, np.ndarray):
        return int(x.shape[0])
    return int(x.vec.shape[0])


DELTA = 2.0 ** -10      # the CKKS comparisons' default distance from 0 / 1 outside the gap (a swap / an indicator
                        # carries it times the compared difference: sorted values within bound DELTA per step)
IN_MARGIN = 1.0 / 16    # the CKKS comparisons scale |d| <= bound into [-1 / (1 + IN_MARGIN), ..]: values a network
                        # keeps rewriting drift by its refreshes' error, and a difference past the bound evaluated the
                        # ladder's first stage outside [-1, 1] (CKKS bitonic n 256: 1e19 on the device; the later
                        # stages' own margin is impl.math.LADDER_MARGIN)
NET_MARGIN = 1.0 / 4    # a sorting network's: its values carry every earlier step's refresh error (bitonic n 1024:
                        # differences up to 2.18 for a bound of 2 on the device, 1e19 at IN_MARGIN; 1/4 costs n 1024's
                        # comparisons one level, 18 for 17)


def ladder_for(bound: float, res: float, delta: float | None = None):
    """The composite sign deciding |d| >= res for |d| <= bound, within delta (default min(res / bound, DELTA)) of
    0 / 1."""
    eps = min(0.25, float(res) / float(bound))
    return M.sign_ladder(eps, min(0.25, float(delta) if delta is not None else min(eps, DELTA)))


def step(ops, d, heads, scale, *, method="tfhe", bound=1.0, res=1e-2, delta=None, tag="c", n=None, levels=1,
         margin=IN_MARGIN):
    """[scale_h d_h >= 0] (1 / 0) on the slots `heads`, 0 elsewhere. `scale`: a number or one per head (a sign
    folds the comparison's direction in); |scale d| <= bound. CKKS: within `delta` of 0 / 1 where |scale d| >= res;
    an exact 0 input gives 1/2. TFHE: exact outside the switch's noise band; `levels` > 1: tfhe_select.msign's
    finer scales in the same switch (2 levels - 1 more PBS per value, the band / 16 per extra level; needs
    2 levels - 1 slot groups beside the heads). `margin` (CKKS): how far past `bound` |scale d| may drift and still
    land inside the ladder's first stage."""
    n = _nslots(ops, d) if n is None else int(n)
    heads = np.asarray(heads, dtype=np.int64)
    sc = np.broadcast_to(np.asarray(scale, dtype=np.float64), heads.shape)
    if method == "tfhe" and levels > 1:
        from .tfhe_select import msign
        dd = d if np.all(sc == 1.0) else ops.mult_pt(d, _put(n, heads, sc), kind="mask")
        return msign(ops, dd, n, heads, float(bound), int(levels), f"sort.{tag}")
    if method == "tfhe":
        from . import tfhe
        from .tfhe_select import SPAN, _clean, sign01
        D = ops.mult_pt(d, _put(n, heads, sc * SPAN / bound), kind="mask")
        name = f"sort.{len(heads)}.{hashlib.blake2b(heads.tobytes(), digest_size=6).hexdigest()}"
        S = tfhe.tfhe_lut(ops, D, sign01(), tfhe.register_slots(name, heads))
        return _clean(ops, S, n, heads)
    if method == "fbs":
        return _step_fbs(ops, d, n, heads, sc, float(bound), float(res))
    if method != "ckks":
        raise ValueError(f"method {method!r}: 'tfhe', 'ckks' or 'fbs'")
    bound = float(bound) * (1.0 + margin)
    lad = ladder_for(bound, res, delta)
    t = ops.mult_pt(d, _put(n, heads, sc / bound), kind="mask")
    from .relu import odd_series
    for i, c in enumerate(lad.coefs):
        t = odd_series(ops, t, [0.5 * v for v in c] if i == len(lad.coefs) - 1 else list(c))
    return ops.add_pt(t, _mask(n, heads, 0.5), kind="mask")


# FBS sign (impl/fbs.py register_sign): x = 1.9 scale d / bound on |x| < 2, the 0 / 1 step exact to ~2e-5 outside
# |x| < gap (degree 63: gap 0.2, 127: 0.1, ... 1023: 0.0125, `FBS_SIGN`); the sine keeps the sign up to |x| < 4, so inputs may drift to
# twice `bound`. The resolution `res` (d units) maps to the gap 1.9 res / bound; one ciphertext costs one functional
# bootstrap whatever the number of heads (TFHE: one PBS per head), on the shallow EvalMod table (k32d3).
FBS_SIGN = ((63, 0.2), (127, 0.1), (255, 0.05), (511, 0.025), (1023, 0.0125))
FBS_X = 1.9


def fbs_sign_name(bound, res):
    """The registered FBS sign for a resolution `res` at `bound` (the lowest degree whose gap fits), or ValueError."""
    from . import fbs as F
    g = FBS_X * float(res) / float(bound)
    for deg, gap in FBS_SIGN:
        if gap <= g:
            name = f"sort.fbs.sign.{deg}"
            if name not in F.FUNCS:
                F.register_sign(name, 0.0, 1.0, deg=deg, gap=gap, table="k32d3")
            return name
    raise ValueError(f"fbs step: resolution {res} at bound {bound} is gap {g:.3g} < {FBS_SIGN[-1][1]} (use tfhe)")


def _step_fbs(ops, d, n, heads, sc, bound, res):
    from . import fbs as F
    name = fbs_sign_name(bound, res)
    x = ops.mult_pt(d, _put(n, heads, sc * FBS_X / bound), kind="mask")
    s = F.fbs(ops, x, name)
    return ops.mult_pt(s, _mask(n, heads, 1.0), kind="mask")


def step_method(n_heads, bound, res):
    """The measured rule for `step`'s backend (logs/fbs26_RUNLOG.md): 'fbs' for a wide round (>= FBS_MIN_HEADS heads in
    one ciphertext) whose resolution the FBS sign reaches, else 'tfhe'."""
    try:
        fbs_sign_name(bound, res)
    except ValueError:
        return "tfhe"
    return "fbs" if int(n_heads) >= FBS_MIN_HEADS else "tfhe"


FBS_MIN_HEADS = 256


def _put(n, idx, vals):
    out = np.zeros(n)
    out[np.asarray(idx, dtype=np.int64)] = vals
    return out


def _idx(lay: Batch, copies: int | None = None):
    """(slot, j, b) of the layout's live values in the first `copies` periods (default all)."""
    c = lay.slots // lay.period if copies is None else int(copies)
    s = np.arange(c * lay.period)
    j = (s // lay.B) % lay.span
    keep = j < lay.n
    return s[keep], j[keep], (s % lay.B)[keep]


def _tile(ops, v, lay: Batch, N: int):
    """The first period's values copied over the whole ciphertext (one mask level, log2(N / period) rotations): the
    TFHE paths compare the first period only."""
    if N == lay.period:
        return v
    v = ops.mult_pt(v, _put(N, np.arange(lay.period), 1.0), kind="mask")
    return ops.rotate_and_sum(v, -lay.period, N)


def tile_rotations(lay: Batch) -> list:
    out, g = [], lay.period
    while g < lay.slots:
        out.append(-g)
        g *= 2
    return out


# ── tournament: max / min / argmax ──────────────────────────────────────────────────────────────────────────────────


def vmax(ops, x, lay: Batch, bound: float, *, method="tfhe", res=None, err=1e-3, levels=1):
    """Each vector's maximum in every one of its slots (|x_i - x_j| <= bound). CKKS: log2 n all-slot rounds of
    max(x, rot(x)) (each within err; the gap res = err); TFHE: head rounds (n - 1 PBS per vector), a mask, a
    broadcast."""
    if lay.pad:
        raise ValueError("vmax: an unpadded layout")
    n, B = lay.n, lay.B
    N = _nslots(ops, x)
    if method == "ckks":
        k = 1
        e = float(err) / max(1, int(_m.log2(n)))
        while k < n:
            x = M.maximum(ops, ops.rotate(x, k * B), x, bound, e, "ckks")
            k *= 2
        return x
    s, j, _b = _idx(lay, 1)
    k = 1
    while k < n:
        heads = s[j % (2 * k) == 0]
        d = ops.sub(ops.rotate(x, k * B), x)
        sel = step(ops, d, heads, 1.0, method=method, bound=bound, tag=f"max{k}", n=N,
                   levels=levels if len(heads) * (2 * levels - 1) <= N // 2 else 1)
        x = ops.add(x, ops.mult(d, sel))
        k *= 2
    head0 = _mask(N, s[j == 0])
    return ops.rotate_and_sum(ops.mult_pt(x, head0, kind="mask"), -B, N)


def vmin(ops, x, lay: Batch, bound: float, **kw):
    """-vmax(-x)."""
    return ops.mult(vmax(ops, ops.mult(x, -1.0), lay, bound, **kw), -1.0)


def argmax(ops, x, lay: Batch, bound: float, *, method="tfhe", res=1e-2, err=1e-3, m=None):
    """The one-hot of each vector's maximum: [x >= max - res / 2] on the live slots (values within res / 2 of the
    maximum also read 1). `m`: the broadcast maximum if already computed."""
    m = vmax(ops, x, lay, bound, method=method, err=err) if m is None else m
    s, _j, _b = _idx(lay)
    d = ops.add_pt(ops.sub(x, m), _mask(_nslots(ops, x), s, 0.5 * res), kind="mask")
    return step(ops, d, s, 1.0, method=method, bound=bound + res, res=0.5 * res, tag="amax")


def argmin(ops, x, lay: Batch, bound: float, **kw):
    return argmax(ops, ops.mult(x, -1.0), lay, bound, **kw)


# ── all-pairs ranking ───────────────────────────────────────────────────────────────────────────────────────────────


def pairs_fit(n: int, B: int = 1, slots: int = 1 << 15) -> bool:
    """Whether `rank`'s all-pairs layout (rows of 2n slots, n rows, B vectors) fits one ciphertext."""
    return 2 * n * n * B <= slots


def _pair_slots(lay: Batch):
    n, B = lay.n, lay.B
    i, j, b = np.meshgrid(np.arange(n), np.arange(2 * n), np.arange(B), indexing="ij")
    return (b + B * (2 * n * i + j)), i, j, b


def rank(ops, x, lay: Batch, bound: float, *, method="ckks", res=1e-2, delta=None, compact=True):
    """rank_j = #{i: x_i < x_j} (0 .. n - 1, distinct values) on each vector's slots, by all n^2 comparisons in one
    ciphertext (module doc). `compact`: back on the Batch layout (one mask level, one rotation); else on the pair
    layout's columns (slot b + B (2 n i + j), j < n, any i). Ties within `res` (CKKS) / the noise band (TFHE) give
    fractional or shared ranks."""
    n, B = lay.n, lay.B
    N = _nslots(ops, x)
    if lay.pad or not pairs_fit(n, B, N):
        raise ValueError(f"rank: n = {n}, B = {B} needs 2 n^2 B <= {N} slots (use bitonic / threshold)")
    P = 2 * n * n * B
    sl, i, j, b = _pair_slots(lay)
    diag = np.tile(_put(P, sl[(j == i)], 1.0), N // P)
    D = ops.mult_pt(x, diag, kind="mask")                    # x_i at (i, i)
    R = ops.sub(ops.add(ops.rotate_and_sum(D, B, n * B), ops.rotate_and_sum(D, -B, n * B)), D)
    d = ops.sub(x, R)                                        # x_j - x_i at (i, j < n)
    if method == "tfhe":
        heads = sl[(j < n) & (j != i)]                       # the pairs, self-comparisons excluded
        s = step(ops, d, heads, 1.0, method="tfhe", bound=bound, tag="rank", n=N)
        r = ops.rotate_and_sum(s, 2 * n * B, P)              # sum over the rows
        offs = 0.0
    else:
        bound = float(bound) * (1.0 + IN_MARGIN)
        lad = ladder_for(bound, res, delta)
        live = np.tile(_put(P, sl[j < n], 1.0 / bound), N // P)
        t = ops.mult_pt(d, live, kind="mask")                # padding columns: an exact 0 (sgn 0)
        from .relu import odd_series
        for k, c in enumerate(lad.coefs):
            t = odd_series(ops, t, [0.5 * v for v in c] if k == len(lad.coefs) - 1 else list(c))
        r = ops.rotate_and_sum(t, 2 * n * B, P)              # 1/2 sum_i sgn(x_j - x_i)
        offs = 0.5 * (n - 1)
    if not compact:
        return ops.add(r, offs) if offs else r
    if method == "tfhe":                                     # valid on the first tile's row 0 = the first period
        return _tile(ops, r, lay, N)
    colm = np.tile(_put(P, sl[j < n], 1.0), N // P)
    r = ops.mult_pt(r, colm, kind="mask")
    r = ops.add(r, ops.rotate(r, -n * B))
    return ops.add_pt(r, lay.live_mask(offs), kind="mask") if offs else r


def rank_rotations(lay: Batch) -> list:
    n, B = lay.n, lay.B
    out, g = [-n * B], B
    while g < n * B:
        out += [g, -g]
        g *= 2
    g = 2 * n * B
    while g < 2 * n * n * B:
        out.append(g)
        g *= 2
    return out


def _rank_bound(lay):
    return lay.n + 1.0


def topk_mask(ops, r, lay: Batch, k: int, *, method="ckks"):
    """[rank >= n - k] (the k largest) from ranks (rank's compact output)."""
    s, _j, _b = _idx(lay)
    d = ops.add_pt(r, _mask(_nslots(ops, r), s, -(lay.n - k) + 0.5), kind="mask")
    return step(ops, d, s, 1.0, method=method, bound=_rank_bound(lay), res=0.5, tag=f"topk{k}")


def _indicators(ops, r, lay, ks, method):
    """[rank == k] for each k in ks (0 elsewhere): the comparisons [rank >= k - 1/2] for k in ks and k + 1 share one
    evaluation, one per tiled copy of the layout (copy t reads rank - edge_t), then the copies are differenced back
    onto copy 0. Valid on the first period."""
    edges = sorted({k - 0.5 for k in ks} | {k + 0.5 for k in ks})
    N = _nslots(ops, r)
    copies = N // lay.period
    if len(edges) > copies:
        raise ValueError(f"indicators: {len(edges)} edges > {copies} copies of the layout")
    s, j, b = _idx(lay, len(edges))
    t = s // lay.period
    off = _put(N, s, -np.asarray(edges)[t])
    d = ops.add_pt(r, off, kind="mask")
    st = step(ops, d, s, 1.0, method=method, bound=_rank_bound(lay), res=0.5, tag=f"ind{len(edges)}")
    by = {e: (st if q == 0 else ops.rotate(st, q * lay.period)) for q, e in enumerate(edges)}
    return {k: _tile(ops, ops.sub(by[k - 0.5], by[k + 0.5]), lay, N) for k in ks}


def kth(ops, x, r, lay: Batch, k: int, *, method="ckks"):
    """The k-th smallest value (k = 0 .. n - 1) of each vector, broadcast over its slots of the first period:
    sum_j x_j [rank_j == k]."""
    from .stats import total
    e = _indicators(ops, r, lay, [int(k)], method)[int(k)]
    return total(ops, ops.mult(x, e), lay)


def quantile(ops, x, r, lay: Batch, q: float, *, method="ckks"):
    """The q-quantile (numpy's linear interpolation between the order statistics floor / ceil of q (n - 1))."""
    from .stats import total
    pos = float(q) * (lay.n - 1)
    k0 = int(_m.floor(pos))
    f = pos - k0
    ks = [k0] if f == 0.0 or k0 + 1 >= lay.n else [k0, k0 + 1]
    ind = _indicators(ops, r, lay, ks, method)
    if len(ks) == 1:
        return total(ops, ops.mult(x, ind[k0]), lay)
    w = ops.add(ops.mult(ind[k0], 1.0 - f), ops.mult(ind[k0 + 1], f))
    return total(ops, ops.mult(x, w), lay)


def median(ops, x, r, lay: Batch, *, method="ckks"):
    return quantile(ops, x, r, lay, 0.5, method=method)


def sort_by_rank(ops, x, r, lay: Batch, *, method="ckks"):
    """The ascending sort from ranks: out_k = sum_i x_i [rank_i == k] on the pair layout (rank_i and x_i replicated
    along row i; columns j < n read [rank_i >= j - 1/2], columns n + j [rank_i >= j + 1/2]: both comparisons of all
    n^2 pairs in one evaluation), then a ladder over the rows. Same size limit as `rank`; `r` tiled (rank's compact
    output)."""
    n, B = lay.n, lay.B
    N = _nslots(ops, x)
    P = 2 * n * n * B
    sl, i, j, b = _pair_slots(lay)
    diag = np.tile(_put(P, sl[(j == i)], 1.0), N // P)
    colm = np.tile(_put(P, sl[j < n], 1.0), N // P)

    def rows(v):
        D = ops.mult_pt(v, diag, kind="mask")
        return ops.sub(ops.add(ops.rotate_and_sum(D, B, n * B), ops.rotate_and_sum(D, -B, n * B)), D)
    Rx = rows(x)                                              # x_i on row i, columns j < n
    Rr = ops.mult_pt(rows(r), colm, kind="mask")
    Rr = ops.add(Rr, ops.rotate(Rr, -n * B))                  # rank_i on all 2n columns of row i
    kk = np.where(j < n, j - 0.5, (j - n) + 0.5).astype(np.float64)
    first = 1 if method == "tfhe" else N // P
    heads = (sl.reshape(-1)[None, :] + P * np.arange(first)[:, None]).reshape(-1)
    off = np.tile(_put(P, sl.reshape(-1), -kk.reshape(-1)), N // P)
    st = step(ops, ops.add_pt(Rr, off, kind="mask"), heads, 1.0, method=method, bound=2.0 * n + 1.0, res=0.5,
              tag="srt", n=N)
    e = ops.sub(st, ops.rotate(st, n * B))                    # columns j < n: [rank_i == j]
    out = ops.rotate_and_sum(ops.mult(Rx, e), 2 * n * B, P)   # sum over the rows i
    if method == "tfhe":
        return _tile(ops, out, lay, N)
    out = ops.mult_pt(out, colm, kind="mask")
    return ops.add(out, ops.rotate(out, -n * B))


# ── bitonic network ─────────────────────────────────────────────────────────────────────────────────────────────────


def bitonic_steps(n: int):
    """[(k, j)] of Batcher's bitonic sort of n (a power of two)."""
    out, k = [], 2
    while k <= n:
        j = k // 2
        while j >= 1:
            out.append((k, j))
            j //= 2
        k *= 2
    return out


def bitonic_rotations(lay: Batch) -> list:
    js = sorted({j for _k, j in bitonic_steps(lay.n)})
    return [j * lay.B for j in js] + [-j * lay.B for j in js]


def bitonic(ops, x, lay: Batch, bound: float, *, method="tfhe", res=1e-2, delta=None, payload=None, descending=False,
            refresh=None, levels=1):
    """Batcher's bitonic sort of every vector (ascending; `descending`), returning (sorted, payload sorted along) on
    the first period (TFHE compares the first period's pairs only; CKKS all slots). `payload`: a ciphertext on the
    same layout permuted with the values (e.g. the indices: argsort). |x_i - x_j| <= bound."""
    if lay.pad:
        raise ValueError("bitonic: an unpadded layout")
    n, B = lay.n, lay.B
    N = _nslots(ops, x)
    s, jj, _b = _idx(lay, 1 if method == "tfhe" else None)
    for k, j in bitonic_steps(n):
        lower = (jj & j) == 0
        heads = s[lower]
        asc = ((jj[lower] & k) == 0) if k < n else np.ones(lower.sum(), dtype=bool)
        if descending:
            asc = ~asc
        sig = np.where(asc, -1.0, 1.0)            # swap = [-sigma d >= 0]: ascending swaps when d < 0
        d = ops.sub(ops.rotate(x, j * B), x)
        sw = step(ops, d, heads, sig, method=method, bound=bound, res=res, delta=delta, tag=f"b{k}.{j}", n=N,
                  levels=levels, margin=NET_MARGIN)
        dl = ops.mult(d, sw)
        x = ops.sub(ops.add(x, dl), ops.rotate(dl, -j * B))
        if payload is not None:
            dp = ops.mult(ops.sub(ops.rotate(payload, j * B), payload), sw)
            payload = ops.sub(ops.add(payload, dp), ops.rotate(dp, -j * B))
        if refresh is not None:
            refresh(x, payload)
    if method == "tfhe":
        x = _tile(ops, x, lay, N)
        payload = _tile(ops, payload, lay, N) if payload is not None else None
    return x, payload


def rank_bitonic(ops, x, lay: Batch, bound: float, *, method="tfhe", res=1e-2, delta=None):
    """rank_j for any n: the argsort of the argsort, two bitonic networks (the first carries the indices j / n along
    the values, the second sorts those (distinct, 1 / n apart) carrying j / n back: position p of the result holds
    rank_p / n), then the scale. Twice `bitonic`'s cost."""
    n = lay.n
    N = _nslots(ops, x)
    ramp = lay.encode(np.tile(np.arange(n, dtype=np.float64) / n, (lay.B, 1)))
    idx = ops.add_pt(ops.mult_pt(x, np.zeros(N), kind="mask"), ramp, kind="mask")
    _xs, p = bitonic(ops, x, lay, bound, method=method, res=res, delta=delta, payload=idx)
    _ps, r = bitonic(ops, p, lay, 1.0, method=method, res=0.25 / n, delta=delta, payload=idx)
    return ops.mult(r, float(n))


# ── threshold search ────────────────────────────────────────────────────────────────────────────────────────────────


def threshold(ops, x, lay: Batch, k: int, lo: float, hi: float, *, probes: int = 10, method="tfhe", count_method=None,
              levels=1):
    """The largest t on the grid lo + (hi - lo) i / 2^probes with count(x >= t) >= k (so the k-th largest value is in
    [t, t + (hi - lo) / 2^probes)), broadcast over each vector's slots, and the top-k mask [x >= t]. Per probe one
    comparison of the n values against t + w, a count ladder, and one comparison of the count with k (on the vector's
    slot 0 for TFHE, every slot for CKKS). `levels` (TFHE): the value comparisons' scales (msign; 2: the band / 16 at
    3x the PBS: a probe resolution below ~1e-2 of the range needs it)."""
    from .stats import total
    n, B = lay.n, lay.B
    N = _nslots(ops, x)
    count_method = method if count_method is None else count_method
    s, j, _b = _idx(lay, 1 if method == "tfhe" else None)
    s0 = s[j == 0]
    bound = float(hi) - float(lo)
    y = ops.add(x, -float(lo))                           # y in [0, bound]
    t = None
    w = bound
    for p in range(int(probes)):
        w /= 2.0
        dt = ops.add_pt(y, _mask(N, s, -w), kind="mask") if t is None else ops.sub(ops.add_pt(y, _mask(N, s, -w),
                                                                                          kind="mask"), t)
        c = step(ops, dt, s, 1.0, method=method, bound=bound + 1.0, res=bound / 2 ** probes / 4, tag=f"thr{p}", n=N,
                 levels=levels)
        # count(y >= t + w), every slot of the vector (CKKS: over n + 1 before the sum, so the comparison's bound is 1)
        cnt = total(ops, c if count_method == "tfhe" else ops.mult(c, 1.0 / (n + 1.0)), lay)
        e = ops.add_pt(cnt, _mask(N, s0, 0.5 - k) if count_method == "tfhe" else _mask(N, s, (0.5 - k) / (n + 1.0)),
                       kind="mask")
        if count_method == "tfhe":
            from .tfhe_select import msign
            bit = msign(ops, e, N, s0, 2.0 * n, 3, f"thr.c{p}", res=0.25)
            inc = ops.rotate_and_sum(ops.mult_pt(bit, _mask(N, s0, w), kind="mask"), -B, N)
        else:
            # resolution 1 / 2 (n + 1): a refresh between the sum and the ladder's first stage would cost ~1e-3 of
            # the bound (plan/btserr.py), past that resolution from n ~ 500 (n 4096: masks exact on the device, two
            # seeds; the ladders' margins keep a wrong decision bounded)
            bit = step(ops, e, s, 1.0, method="ckks", bound=1.0, res=0.5 / (n + 1.0), tag=f"thr.c{p}", n=N)
            inc = ops.mult(bit, w)
        t = inc if t is None else ops.add(t, inc)
    mask = step(ops, ops.sub(y, t), s, 1.0, method=method, bound=bound + 1.0, res=bound / 2 ** probes / 4, tag="thr.m",
                n=N, levels=levels)
    if method == "tfhe":
        mask = _tile(ops, mask, lay, N)
    return ops.add(t, float(lo)), mask
