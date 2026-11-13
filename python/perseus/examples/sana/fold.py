"""SANA_HIDDEN_FOLD=1 (read at import, like SANA_TIGHT_ROWS: a plan is cut for one of the two): the hidden width's third
channel block folded across chunk pairs.

The residual stream and every hidden-width tensor (1152 channels: blocks 0 and 1 full, block 2 holding 128 real
channels and 384 padding rows) is carried as FH: rows[g] = [block 0, block 1] per chunk pair, and rem[f] one ciphertext
per group of FOLD_PAIRS = 4 chunk pairs whose sub-block j (channel rows [128 j, 128 j + 128)) holds block 2's real
channels of chunk pair 4 f + j, on the same lanes. 15 -> 12 ciphertexts per hidden tensor at 768 px (G = 5: the second
folded ciphertext holds one pair), 12 -> 9 at 704 px (G = 4).

Nothing is unfolded on the way:
  - a PCMM reading a folded input takes pair g's block 2 as the folded ciphertext rotated by 128 j rows (no level: its
    other rows meet the padding's zero weight rows);
  - a PCMM writing a hidden output computes blocks 0 and 1 per pair as before and the folded block as ONE linear over
    the group's inputs with a block-diagonal weight (pair j's inputs -> sub-block j): one finish per folded ciphertext
    instead of one per pair; both share the unfolded linear's weight scale, shift and stamp;
  - the norms read the folded block through the masks they multiply by anyway (blocked.norm_blocked row0, p18);
  - the linear attention's heads (32 channels) stay inside a sub-block: the per-head token sums over pairs become a
    2-step row ladder over the sub-blocks (no level), the key's RMS scale reaches the folded ReLU as the sum of the
    pairs' scales on their sub-blocks (one mask product, where the packed block-2 chains masked it already).
The FFN (5760 / 2880 channels), the depthwise convolution, the cross-attention's x-head layout and the text side are
unchanged."""
from __future__ import annotations

import os

import numpy as np

from perseus.impl.prefill import join, split

from . import layout as L
from .layout import G, LANES, N, NCH

FOLD = os.environ.get("SANA_HIDDEN_FOLD", "0") not in ("", "0")
FOLD_PAIRS = 4
RB = NCH // FOLD_PAIRS                   # 128 rows per sub-block = block 2's real channels (1152 - 1024)
NF = -(-G // FOLD_PAIRS)                 # folded ciphertexts per hidden tensor
assert L.D - 2 * NCH == RB


def group(f):
    """The chunk pairs of folded ciphertext f."""
    return list(range(FOLD_PAIRS * f, min(G, FOLD_PAIRS * (f + 1))))


def rotations():
    """The row rotations the fold uses (pair j's sub-block to rows 0-127 and back; the sub-block ladder)."""
    s = {j * RB * LANES for j in range(1, FOLD_PAIRS)}
    return s | {-x for x in s} | {(N - x) % N for x in s}


def sub_mask(j, vals=None):
    """Slot vector: 1 (or `vals` (RB,) per channel) on sub-block j's rows, 0 elsewhere."""
    ch = np.zeros(NCH)
    ch[j * RB:(j + 1) * RB] = 1.0 if vals is None else np.asarray(vals)
    return L.lane_vec(np.ones(LANES), ch)


class FH:
    """A folded hidden-width tensor: rows[g] = [block 0, block 1], rem[f] (see the module docstring)."""

    def __init__(self, rows, rem):
        self.rows, self.rem = [list(r) for r in rows], list(rem)
        assert len(self.rows) == G and len(self.rem) == NF

    @staticmethod
    def n_cts():
        return 2 * G + NF

    def flat(self):
        return [x for r in self.rows for x in r] + list(self.rem)

    @classmethod
    def from_flat(cls, xs):
        xs = list(xs)
        assert len(xs) == cls.n_cts(), (len(xs), cls.n_cts())
        return cls([xs[2 * g:2 * g + 2] for g in range(G)], xs[2 * G:])

    def map(self, fn):
        return FH([[fn(x) for x in r] for r in self.rows], [fn(x) for x in self.rem])

    def add(self, ops, other):
        return FH([[ops.add(a, b) for a, b in zip(ra, rb)] for ra, rb in zip(self.rows, other.rows)],
                  [ops.add(a, b) for a, b in zip(self.rem, other.rem)])

    def pcmm_rows(self, rt, t):
        """Per pair g the PCMM's three input blocks: blocks 0, 1 and the folded ciphertext with pair g's sub-block
        rotated to rows 0-127 (its other rows meet zero weight rows). The folded ciphertext is hinted to the PCMM entry
        before its rotations, so its pairs' copies share one refresh."""
        from perseus.impl.pcmm import hint_entry
        ops = rt.ops
        cache = {}

        def down(f, j):
            if (f, j) not in cache:
                if (f, 0) not in cache:
                    cache[f, 0] = hint_entry(rt, self.rem[f], t)
                cache[f, j] = ops.rotate(cache[f, 0], j * RB * LANES) if j else cache[f, 0]
            return cache[f, j]
        return [list(self.rows[g]) + [down(g // FOLD_PAIRS, g % FOLD_PAIRS)] for g in range(G)]

    def norm_rows(self):
        """norm_blocked's Xs (each pair's block 2 = its folded ciphertext) and row0."""
        return ([list(self.rows[g]) + [self.rem[g // FOLD_PAIRS]] for g in range(G)],
                lambda g, k: RB * (g % FOLD_PAIRS) if k == 2 else 0)


def from_blocks(ops, out_rows):
    """FH from per-pair [block 0, block 1, block 2 on pair g's rows only (zero elsewhere)]: the folded block summed."""
    rem = []
    for f in range(NF):
        acc = None
        for g in group(f):
            acc = out_rows[g][2] if acc is None else ops.add(acc, out_rows[g][2])
        rem.append(acc)
    return FH([r[:2] for r in out_rows], rem)


def norm(rt, X: FH, cfg, ch, B):
    from perseus.impl.blocked import norm_blocked
    xs, row0 = X.norm_rows()
    return from_blocks(rt.ops, norm_blocked(rt, xs, cfg, ch, B, row0=row0))


def unscale(rt, X: FH, r, site):
    """model._unscale on a folded tensor (the folded block's channel scales tiled over its sub-blocks)."""
    from . import model as M
    if np.all(r == 1.0):
        return X
    rv = M._padv(r, 3 * NCH)
    rows = [[rt.mult_mask(x, (site, "unscale", k), lambda k=k: L.lane_vec(np.ones(LANES), rv[k * NCH:(k + 1) * NCH]))
             for k, x in enumerate(row)] for row in X.rows]
    tile = np.tile(rv[2 * NCH:2 * NCH + RB], FOLD_PAIRS)
    rem = [rt.mult_mask(x, (site, "unscale.fold"), lambda: L.lane_vec(np.ones(LANES), tile)) for x in X.rem]
    return FH(rows, rem)


# ── values (checks, tests) ──────────────────────────────────────────────────────────────────────────────────────────

def unfold_vals(vals):
    """Slot vectors of a folded tensor (FH.flat order) -> [g][3] slot vectors of the unfolded layout."""
    vals = [np.asarray(v) for v in vals]
    out = []
    for g in range(G):
        f, j = divmod(g, FOLD_PAIRS)
        b2 = np.roll(vals[2 * G + f], -j * RB * LANES).reshape(NCH, LANES).copy()
        b2[RB:] = 0.0
        out.append([vals[2 * g], vals[2 * g + 1], b2.reshape(N)])
    return out


def fold_vals(blocks):
    """[g][3] slot vectors (unfolded) -> FH.flat order slot vectors (the inverse of unfold_vals on the real rows)."""
    flat = [np.asarray(v) for g in range(G) for v in blocks[g][:2]]
    for f in range(NF):
        acc = np.zeros(N, dtype=complex)
        for g in group(f):
            j = g % FOLD_PAIRS
            b2 = np.asarray(blocks[g][2]).reshape(NCH, LANES).copy()
            b2[RB:] = 0.0
            acc += np.roll(b2.reshape(N), j * RB * LANES)
        flat.append(acc)
    return flat


# ── PCMM weights ────────────────────────────────────────────────────────────────────────────────────────────────────

def fold_out_cols(n_out):
    """The output blocks that are a hidden tensor's block 2 (one tensor: n_out 3; q, k, v: n_out 9)."""
    assert n_out % 3 == 0, n_out
    return [3 * t + 2 for t in range(n_out // 3)]


def fold_weight(W, b, n_in, nch, n_pairs, cols):
    """The folded block's linear for a group of n_pairs pairs: inputs = the pairs' n_in blocks each (pair-major), outputs
    = one block per tensor in `cols`, sub-block j <- pair j's channels [cols[t] nch, cols[t] nch + RB) of W."""
    Wf = np.zeros((n_pairs * n_in * nch, len(cols) * nch))
    bf = np.zeros(len(cols) * nch)
    for j in range(n_pairs):
        for t, c in enumerate(cols):
            Wf[j * n_in * nch:(j + 1) * n_in * nch, t * nch + j * RB:t * nch + (j + 1) * RB] = W[:, c * nch:c * nch + RB]
            bf[t * nch + j * RB:t * nch + (j + 1) * RB] = b[c * nch:c * nch + RB]
    return Wf, bf


# ── the linear attention on folded q / k / v ───────────────────────────────────────────────────────────────────────

def linear_attention(rt, Q: FH, K: FH, V: FH, wk, nl, site, delta=1e-15, out_scale=1.0):
    """kernels.linear_attention on folded inputs (the same maps and refresh rooms; block 2 = the folded ciphertexts)."""
    from . import kernels as KN
    ops = rt.ops
    inv_w = np.zeros(3 * NCH)
    inv_w[:KN.D_REAL] = 1.0 / np.asarray(wk)
    eps = 1e-5
    kh = [[split(ops, x) for x in K.rows[g]] for g in range(G)]
    qh = [[split(ops, x) for x in Q.rows[g]] for g in range(G)]
    khf = [split(ops, x) for x in K.rem]
    qhf = [split(ops, x) for x in Q.rem]
    with rt.step("la.knorm"):
        c = nl.stat_scale(site + ".knorm")
        stats = []
        for g in range(G):
            f, j = divmod(g, FOLD_PAIRS)
            for half in (0, 1):
                z = KN.token_stat(rt, [kh[g][0][half], kh[g][1][half], khf[f][half]], c * inv_w, site + ".knorm",
                                  row0=[0, 0, RB * j])
                stats.append(rt.add_mask(z, (site, "knorm.eps", c), lambda: np.full(N, c * c * eps)))
        rks = KN.packed_lane_map(rt, stats, lambda z: nl.inv_sqrt(rt, z, site + ".knorm"), site + ".knorm",
                                 lanes=[L.VALID[L.slot_of(g, half)].astype(float) for g in range(G) for half in (0, 1)])
        # the folded block's scale: each pair's rk on its sub-block (one mask product)
        rkf = []
        for f in range(NF):
            pair = []
            for half in (0, 1):
                acc = None
                for g in group(f):
                    j = g % FOLD_PAIRS
                    m = rt.mult_mask(rks[2 * g + half], (site, "rk.fold", j), lambda j=j: sub_mask(j))
                    acc = m if acc is None else ops.add(acc, m)
                pair.append(acc)
            rkf.append(pair)
    with rt.step("la.relu_k"):                                                    # a PCMM output
        fk = {}
        for g in range(G):
            for k in range(2):
                fk[g, k] = nl.relu_u2(rt, kh[g][k][0], kh[g][k][1], site + ".k", room=10,
                                      mul=(rks[2 * g], rks[2 * g + 1]))
        fkf = [nl.relu_u2(rt, khf[f][0], khf[f][1], site + ".k", room=10, mul=tuple(rkf[f])) for f in range(NF)]
    with rt.step("la.relu_q"):
        fq = {(g, k): nl.relu_u2(rt, qh[g][k][0], qh[g][k][1], site + ".q", room=10) for g in range(G) for k in range(2)}
        fqf = [nl.relu_u2(rt, qhf[f][0], qhf[f][1], site + ".q", room=10) for f in range(NF)]
    hint = KN.room_hint
    fK = [[hint(rt, join(ops, *fk[g, k]), 6) for k in range(2)] for g in range(G)]
    fQ = [[hint(rt, join(ops, *fq[g, k]), 4) for k in range(2)] for g in range(G)]
    fKf = [hint(rt, join(ops, *x), 6) for x in fkf]
    fQf = [hint(rt, join(ops, *x), 4) for x in fqf]
    return _la_tail(rt, fQ, fK, fQf, fKf, V, nl, site, delta, out_scale)


def _sub_total(rt, x):
    """Each sub-block's rows hold the sum over the sub-blocks (the pairs of a folded ciphertext): 2 rotations."""
    return rt.ops.rotate_and_sum(x, RB * LANES, NCH * LANES)


def _la_tail(rt, fQ, fK, fQf, fKf, V: FH, nl, site, delta, out_scale=1.0):
    from . import kernels as KN
    ops = rt.ops
    sigma = nl.la_sigma(site)
    DH = KN.DH
    S, Z = [], []
    st = rt.step("la.S")
    st.__enter__()
    for k in range(3):                       # k = 2: the folded block (its pairs' sums on every sub-block)
        if KN.LA_HOIST:                      # the value side conjugated once, its in-head rotations a grid
            pairs = [(fK[g][k], V.rows[g][k]) for g in range(G)] if k < 2 else list(zip(fKf, V.rem))
            sd = KN.s_sums(rt, pairs, site, sigma)
            S.append([_sub_total(rt, s) for s in sd] if k == 2 else sd)
            acc = None
            for a in ([fK[g][k] for g in range(G)] if k < 2 else fKf):
                p = ops.mult_const(a, 1.0, -1.0)
                acc = p if acc is None else ops.add(acc, p)
            z = KN.lane_total(rt, ops.add(acc, ops.conjugate(acc)), site + ".z", 0.5 / sigma)
            Z.append(_sub_total(rt, z) if k == 2 else z)
            continue
        sd = []
        for d in range(DH):
            acc = None
            srcs = ([(fK[g][k], V.rows[g][k]) for g in range(G)] if k < 2 else list(zip(fKf, V.rem)))
            for a, v in srcs:
                vd = KN.rotate_in_head(rt, v, d, site + ".v")
                p = ops.mult(a, ops.conjugate(vd))
                acc = p if acc is None else ops.add(acc, p)
            s = KN.lane_total(rt, ops.add(acc, ops.conjugate(acc)), site + ".S", 0.5 / sigma)
            sd.append(_sub_total(rt, s) if k == 2 else s)
        S.append(sd)
        acc = None
        for a in ([fK[g][k] for g in range(G)] if k < 2 else fKf):
            p = ops.mult_const(a, 1.0, -1.0)
            acc = p if acc is None else ops.add(acc, p)
        z = KN.lane_total(rt, ops.add(acc, ops.conjugate(acc)), site + ".z", 0.5 / sigma)
        Z.append(_sub_total(rt, z) if k == 2 else z)
    st.__exit__(None, None, None)
    st = rt.step("la.num")
    st.__enter__()
    nums, dens = [], []
    items = [(fQ[g][k], k, None) for g in range(G) for k in range(2)] + [(fQf[f], 2, f) for f in range(NF)]
    rhs = [KN.num_rhs(rt, S[k], site) for k in range(3)] if KN.LA_NUM_SUM else None
    for q, k, f in items:
        if KN.LA_NUM_SUM:
            num = KN.num_lhs(rt, q, S[k][0], rhs[k])
        else:
            num = None
            for d in range(DH):
                t = KN.rotate_in_head(rt, ops.mult(q, S[k][d]), -d, site + ".num")
                num = t if num is None else ops.add(num, t)
        den = KN.head_total(rt, ops.mult(q, Z[k]), site + ".den")
        if out_scale != 1.0 and KN.NUM_HINT:
            num = KN.room_hint(rt, num, 2)
        nums.append(split(ops, num))
        if KN.LA_QNORM:                      # p20's den normaliser; the folded block keyed by its live sub-blocks
            hq = KN.head_total(rt, q, site + ".hq")
            kkey = k if k < 2 else ("fold", len(group(f)))
            dens += [(dh, hh, kkey) for dh, hh in zip(split(ops, den), split(ops, hq))]
            continue
        if k < 2:
            live = np.ones(NCH, dtype=bool)
            key = (site, "den.delta", delta, k)
        else:
            n_live = len(group(f))
            live = np.arange(NCH) < n_live * RB
            key = (site, "den.delta.fold", delta, n_live)
        for dh in split(ops, den):
            # 2 den + 2 delta on live channels, a large value on the empty sub-blocks (zero num there)
            dens.append(rt.add_mask(dh, key, lambda live=live: L.lane_vec(np.ones(LANES), np.where(live, 2.0 * delta, 1.0))))
    st.__exit__(None, None, None)
    with rt.step("la.recip"):
        recips = KN._la_qnorm_map(rt, dens, nl, site, delta, out_scale, DH) if KN.LA_QNORM else KN.packed_group_map(
            rt, dens, lambda x: nl.recip(rt, x, site + ".den"), site + ".den", DH, out_scale=out_scale)
    outs = []
    for i in range(len(items)):
        na, nb = nums[i]
        outs.append(join(ops, ops.mult(na, recips[2 * i]), ops.mult(nb, recips[2 * i + 1])))   # (2 num) / (2 den)
    rows = [outs[2 * g:2 * g + 2] for g in range(G)]
    return FH(rows, outs[2 * G:])


# ── the block ───────────────────────────────────────────────────────────────────────────────────────────────────────

def block(rt, X: FH, bw, Kb, Vb, nl, lin, b):
    """model.block on the folded residual."""
    from . import kernels as KN
    from . import model as M
    ops = rt.ops
    _d, B, ch = L.geometry()
    site = f"b{b}"
    with rt.step("ln1"):
        N1 = norm(rt, unscale(rt, X, bw.r, site + ".ln1"), nl.ln(site + ".ln1"), ch, B)
    with rt.step("qkv"):
        Q, K, V = lin(rt, N1, bw.qkv, fold_out=True)
    if bw.outm is not None:                  # SANA_ATTN_MERGE (model.py): out1 merged into qx and out2
        from perseus.impl.pcmm import hint_entry
        t = N // NCH
        div = 1.0 / lin.in_div(bw.outm) if hasattr(lin, "in_div") else 1.0
        with rt.step("attn1"):
            O = linear_attention(rt, Q, K, V, bw.wk1, nl, site + ".la", nl.la_delta(site), out_scale=div)
        O = O.map(lambda o: hint_entry(rt, o, t))                        # read by both merged linears: hinted once
        orows = O.pcmm_rows(rt, t)
        with rt.step("qx"):
            Qx = lin(rt, [rx + ro for rx, ro in zip(X.pcmm_rows(rt, t), orows)], bw.qxm)
        with rt.step("attn2"):
            O2 = KN.cross_attention(rt, Qx, Kb, Vb, bw.wq2, nl, site + ".xa", out_scale=div)
        with rt.step("out2"):
            A = lin(rt, [ro + ro2 for ro, ro2 in zip(orows, O2)], bw.outm, fold_out=True)
        X = X.add(ops, A)
    else:
        with rt.step("attn1"):
            O = linear_attention(rt, Q, K, V, bw.wk1, nl, site + ".la", nl.la_delta(site),
                                 out_scale=1.0 / lin.in_div(bw.out1) if hasattr(lin, "in_div") else 1.0)
        with rt.step("out1"):
            A = lin(rt, O, bw.out1, fold_out=True)
        X = X.add(ops, A)
        with rt.step("qx"):
            Qx = lin(rt, X, bw.qx)
        with rt.step("attn2"):
            O = KN.cross_attention(rt, Qx, Kb, Vb, bw.wq2, nl, site + ".xa")
        with rt.step("out2"):
            A = lin(rt, O, bw.out2, fold_out=True)
        X = X.add(ops, A)
    with rt.step("ln2"):
        N2 = norm(rt, unscale(rt, X, bw.r, site + ".ln2"), nl.ln(site + ".ln2"), ch, B)
    with rt.step("ff_in"):
        U = lin(rt, N2, bw.ff_in)
    with rt.step("act1"):
        U = M._silu_blocks(rt, U, nl, site + ".ff.act1", in_room=M.PCMM_ROOM)
    with rt.step("dw"):
        U = KN.depthwise3x3(rt, U, bw.dw_w, bw.dw_b, site + ".dw")
    ff_div = 1.0 / lin.in_div(bw.ff_out) if hasattr(lin, "in_div") else 1.0
    with rt.step("glu"):
        P = [[join(ops, *nl.silu_u2(rt, row[6 + k], site + ".ff.act2", room=M.REFRESH_ROOM - 1,
                                    mul=split(ops, row[k]), out_scale=ff_div)) for k in range(6)] for row in U]
    with rt.step("ff_out"):
        F = lin(rt, P, bw.ff_out, fold_out=True)
    return X.add(ops, F)
