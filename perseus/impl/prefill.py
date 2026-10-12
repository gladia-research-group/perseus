"""Prefill: a T-token prompt through the decode kernels, 2t tokens per complex ciphertext.

Chunk k fills lanes l < t (token k t + l, feature i at slot i t + l); ciphertext c carries chunk 2c
(real) and chunk 2c+1 (imaginary).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from .attention import _recip_rounds, first_iters
from .layout import lanes_mask, lanes_periodic, stride_mask
from .layout import cm_params
from .linear import _baby_sum, linear_multi, prepare_linear_input
from .poly import eval_chebyshev, im_cleanse, rotsum
import os
_DBG = bool(os.environ.get("PF_DEBUG"))
# PF_ATTN_PAIR=1: two offsets per q.K^T / P.V product on the complex axis
_ATTN_PAIR = os.environ.get("PF_ATTN_PAIR", "1") not in ("", "0")
# PF_SM_JOINT=1: both chunks of a ciphertext share one refresh point after the softmax exp's Chebyshev
_SM_JOINT = os.environ.get("PF_SM_JOINT", "0") not in ("", "0")


@dataclasses.dataclass(frozen=True)
class Chunks:
    """A prompt of T tokens in chunks of t lanes; ciphertext c carries chunks 2c (real) and 2c+1 (imaginary)."""
    T: int
    t: int
    base: int = 0            # absolute position of token 0

    @property
    def n_chunks(self):
        return (self.T + self.t - 1) // self.t

    @property
    def n_ct(self):
        return (self.n_chunks + 1) // 2

    def idx(self, k):
        """Prompt indices of chunk k's filled lanes."""
        return list(range(k * self.t, min(self.T, (k + 1) * self.t)))

    def lanes(self, k):
        """Absolute positions of chunk k's filled lanes."""
        return [self.base + i for i in self.idx(k)]

    def has(self, k):
        return k < self.n_chunks

    @property
    def n_pairs(self):
        """Token pairs (lane l of ciphertext c = pair c t + l)."""
        return sum(len(self.lanes(2 * c)) for c in range(self.n_ct))


# ── complex payload: two real chunks per ciphertext ─────────────────────────────────────

def split(ops, X):
    """(2 Re X, 2 Im X): one conjugate, no level."""
    cj = ops.conjugate(X)
    return ops.add(X, cj), ops.mult_i(ops.sub(cj, X))


def join(ops, A, B):
    """A + i B (the monomial: no level); B None leaves A."""
    return A if B is None else ops.add(A, ops.mult_i(B))


def name_inputs(inf, Xs):
    """Name inputs after the first pf_in<c>: anon_<n> numbering differs between a capture and its planned run."""
    if hasattr(inf, "name_ct"):
        for c, X in enumerate(Xs[1:], 1):
            inf.name_ct(X, f"pf_in{c}")


def encode_prompt(rt, rows, ch: Chunks):
    """The prompt rows as ch.n_ct complex ciphertexts at the bootstrap output level (encrypted zero + slot vector)."""
    ops, d = rt.ops, rt.dims
    rows = np.asarray(rows, dtype=np.float64)
    out = []
    idx = np.arange(rows.shape[1]) * d.t
    for c in range(ch.n_ct):
        v = np.zeros(d.N, dtype=np.complex128)
        for half in (0, 1):
            if ch.has(2 * c + half):
                for lane, p in enumerate(ch.idx(2 * c + half)):
                    v[idx + lane] += (1j if half else 1.0) * rows[p]
        zero = ops.core.encode_token_input(ops.inf, np.zeros(d.dim))
        out.append(ops.add_pt(zero, v))
    return out


def decrypt_prompt(rt, Xs, ch: Chunks, d_real: int):
    """(T, d_real) rows of a hid-layout complex stream (stride t)."""
    ops, d = rt.ops, rt.dims
    rows = np.zeros((ch.T, d_real))
    idx = np.arange(d_real) * d.t
    for c, X in enumerate(Xs):
        v = ops.decrypt_slots_complex(X)
        for half, part in ((0, v.real), (1, v.imag)):
            for lane, p in enumerate(ch.idx(2 * c + half)):
                rows[p] = part[idx + lane]
    return rows


# ── dense <-> per-pair (the decode linears) ───────────────────────────────────────────

def _lane0(rt, stride, scale, site):
    N = rt.dims.N
    return (site, stride, scale), lambda: stride_mask(N, N // stride, stride, scale)


def unpack(rt, X, stride: int, n: int, scale: float = 1.0):
    """Lanes 0..n-1 of a dense ciphertext as n ciphertexts with the lane on lane 0 (stride mask times `scale`)."""
    ops = rt.ops
    rots = [X] + ops.rotate_many(X, list(range(1, n)))
    key, build = _lane0(rt, stride, scale, "pf.unpack")
    return [rt.mult_mask(r, key, build) for r in rots]


def pack(rt, ys, stride: int, scale: float = 1.0):
    """sum_l rotate(lane0(ys[l]), -l): the inverse of unpack (lane 0 of each, times `scale`)."""
    ops = rt.ops
    key, build = _lane0(rt, stride, scale, "pf.pack")
    acc = None
    for lane, y in enumerate(ys):
        m = rt.mult_mask(y, key, build)
        if lane:
            m = ops.rotate(m, -lane)
        if acc is None:
            acc = m
        else:
            ops.inplace_add(acc, m)
    return acc


def dense_linear(rt, Xs, ws, ch: Chunks, in_stride: int, out_stride: int, in_scale=1.0, out_scale=1.0,
                 headroom=3):
    """Decode linears `ws` on every token pair of dense Xs (layout in_stride); per weight, dense outputs (out_stride).
    Inputs are refreshed when fewer than `headroom` units are left."""
    ops = rt.ops
    P = ch.n_pairs
    pairs = []
    for X in Xs:
        if len(pairs) >= P:
            break
        ops.bootstrap_hint(X, ops.headroom(headroom), True)
        n = min(in_stride, P - len(pairs))
        pairs += unpack(rt, X, in_stride, n, in_scale)
    # a bias would land on the real chunk only: added to the dense outputs on both axes
    nobias = [dataclasses.replace(w, bias=None) if w.bias is not None else w for w in ws]
    if any(np.iscomplexobj(w.pts) for w in ws):
        ys = [pair_linears(rt, p, nobias) for p in pairs]
        # an unhalved fused weight's outputs carry a factor 2
        fac = [0.5 if _n_out(w) == 2 else 1.0 for w in ws for _ in range(_n_out(w))]
        ws = [w for w in ws for _ in range(_n_out(w))]
    else:
        ys = [linear_multi(rt, p, nobias) for p in pairs]
        fac = [1.0] * len(ws)
    outs = []
    for j, w in enumerate(ws):
        dense = []
        for g in range(0, P, out_stride):
            y = pack(rt, [y[j] for y in ys[g:g + out_stride]], out_stride, out_scale * fac[j])
            if w.bias is not None:
                y = _add_bias(rt, y, w, ch, g, out_stride, out_scale)
            dense.append(y)
        outs.append(dense)
    return outs


def _n_out(w):
    """Outputs per token pair: two for a complex weight that is not output-packed."""
    return 2 if np.iscomplexobj(w.pts) and not w.outputpack else 1


def _bsgs(rt, x_rot, w, ks, idx_of):
    """BSGS accumulators of output blocks `ks`; idx_of(g, k) gives the plaintext indices."""
    ops = rt.ops
    p = cm_params(rt.dims.N, w.d_in, w.d_out)
    giant_rot = p.bstep_c * p.t * p.t
    out = []
    for k in ks:
        acc = None
        for g in range(p.gstep_c):
            tmp = _baby_sum(ops, x_rot, w, idx_of(g, k))
            if g > 0:
                tmp = ops.rotate(tmp, g * giant_rot)
            if acc is None:
                acc = tmp
            else:
                ops.inplace_add(acc, tmp)
        out.append(acc)
    return out


def _tail(rt, w, cy):
    """apply_linear's cascade over the output blocks and its tp_out ladder."""
    ops = rt.ops
    p = cm_params(rt.dims.N, w.d_in, w.d_out)
    cascade_rot = p.t * p.tp
    for k in range(len(cy) - 1, 0, -1):
        ops.inplace_add(cy[k - 1], ops.rotate(cy[k], cascade_rot))
    y = cy[0]
    if p.tp_out > 1:
        y = ops.rotate_and_sum(y, 1, p.tp_out)
    return y


def pair_linears(rt, x, ws):
    """Decode linears on a token pair z = a + i b; a complex weight W0 + i W1 uses S = BSGS(z), S' = BSGS(conj z):
    S + conj S' = a W0 + i b W0, -i (S - conj S') = a W1 + i b W1."""
    ops = rt.ops
    w0 = ws[0]
    # conjugates before the realize: a conjugate's degree is its input's, as the planner predicts
    p0 = cm_params(rt.dims.N, w0.d_in, w0.d_out)
    xi = ops.copy(x) if p0.tp_in == 1 else ops.rotate_and_sum(x, p0.t - 1, p0.tp_in * (p0.t - 1))
    x_rot = [xi] + ops.rotate_many(xi, [b * p0.t * p0.t for b in range(1, p0.bstep_c)])
    x_cj = None
    if any(np.iscomplexobj(w.pts) for w in ws):
        x_cj = [ops.conjugate(r) for r in x_rot]
        for r in x_cj:
            ops.realize(r)
    for r in x_rot:
        ops.realize(r)
    outs = []
    for w in ws:
        p = cm_params(rt.dims.N, w.d_in, w.d_out)
        if not np.iscomplexobj(w.pts):
            outs.append(_tail(rt, w, _bsgs(rt, x_rot, w, range(p.r_o),
                                           lambda g, k: [(g * p.bstep_c + b) * p.r_o + k for b in range(p.bstep_c)])))
            continue
        if w.outputpack:
            rop = p.r_o // 2
            idx = lambda g, k: [(g * p.bstep_c + b) * rop + k for b in range(p.bstep_c)]
            Sa = _bsgs(rt, x_rot, w, range(rop), idx)
            Sb = _bsgs(rt, x_cj, w, range(rop), idx)
            cy = []
            for a, b in zip(Sa, Sb):
                cb = ops.conjugate(b)
                cy += [ops.add(a, cb), ops.mult_i(ops.sub(cb, a))]
            outs.append(_tail(rt, w, cy))
        else:
            idx = lambda g, k: [(g * p.bstep_c + b) * p.r_o + k for b in range(p.bstep_c)]
            Sa = _bsgs(rt, x_rot, w, range(p.r_o), idx)
            Sb = _bsgs(rt, x_cj, w, range(p.r_o), idx)
            c0, c1 = [], []
            for a, b in zip(Sa, Sb):
                cb = ops.conjugate(b)
                c0.append(ops.add(a, cb))
                c1.append(ops.mult_i(ops.sub(cb, a)))
            outs += [_tail(rt, w, c0), _tail(rt, w, c1)]
    return outs


def _add_bias(rt, y, w, ch: Chunks, g0: int, stride: int, scale: float):
    """Bias on the lanes of pairs g0.. of a dense output: scale (b + i b) where the real / imaginary token exists."""
    d = rt.dims
    P = ch.n_pairs
    fac = np.zeros(stride, dtype=np.complex128)
    for lane in range(stride):
        p = g0 + lane
        if p < P:
            c, l = divmod(p, ch.t)
            fac[lane] = scale * (1.0 + (1j if (2 * c + 1) * ch.t + l < ch.T else 0.0))
    key = ("pf.bias", w.name or id(w), g0, stride, scale, ch.T)
    return rt.add_mask(y, key, lambda: np.asarray(w.bias) * fac[np.arange(d.N) % stride])


# ── nonlinear sites on the two halves ───────────────────────────────────────────────

def shift_lanes(rt, X2, shift, ch: Chunks, c: int, d_real: int, scale: float = 1.0):
    """X2 + scale (shift + i shift) on every filled lane (the folded LayerNorm shift)."""
    d = rt.dims
    na, nb = len(ch.lanes(2 * c)), len(ch.lanes(2 * c + 1)) if ch.has(2 * c + 1) else 0
    sh = np.asarray(shift, dtype=np.float64).ravel()[:d_real]

    def build():
        out = np.zeros(d.N, dtype=np.complex128)
        idx = np.arange(sh.shape[0]) * d.t
        for lane in range(d.t):
            out[idx + lane] = scale * sh * ((1.0 if lane < na else 0.0) + (1j if lane < nb else 0.0))
        return out
    return rt.add_mask(X2, ("pf.shift", id(shift), na, nb, scale), build)


def norm_pair(rt, X, cfg, ch: Chunks, c: int, norm_fn):
    """2 (LN(A) + i LN(B)) of X = A + i B; each half is cleansed so its imaginary noise stays out of the other."""
    ops = rt.ops
    nA = im_cleanse(ops, norm_fn(rt, X, cfg, ch.lanes(2 * c)))
    if not ch.has(2 * c + 1):
        return nA
    nB = im_cleanse(ops, norm_fn(rt, ops.mult_i(X), cfg, ch.lanes(2 * c + 1)))
    return join(ops, nA, ops.negate(nB))


def gelu_pair(rt, H, cfg, gelu_fn):
    """2 (g(A) + i g(B)) of H = (A + i B) / 2."""
    ops, d = rt.ops, rt.dims
    A, B = split(ops, H)
    gA = im_cleanse(ops, gelu_fn(rt, A, cfg, lanes=d.tp_E))
    gB = im_cleanse(ops, gelu_fn(rt, B, cfg, lanes=d.tp_E))
    return join(ops, gA, gB)


# ── attention ──────────────────────────────────────────────────────────────────────────

def lane_rotate(rt, x, e: int, site: str):
    """Lane i <- lane (i - e) mod t in every feature group (one level)."""
    ops, d = rt.ops, rt.dims
    if e == 0:
        return x
    N, t = d.N, d.t
    hi = rt.mult_mask(ops.rotate(x, -e), (site + ".hi", e),
                      lambda: lanes_mask(N, N // t, t, [1.0 if i >= e else 0.0 for i in range(t)]))
    lo = rt.mult_mask(ops.rotate(x, t - e), (site + ".lo", e),
                      lambda: lanes_mask(N, N // t, t, [1.0 if i < e else 0.0 for i in range(t)]))
    return ops.add(hi, lo)


def _offsets(ch: Chunks, c: int):
    """(u, c', e) of query chunk c: every earlier chunk at all t offsets, its own chunk at e < its lane count."""
    out = []
    nq = len(ch.lanes(c))
    for cp in range(c, -1, -1):
        for e in range(nq if cp == c else ch.t):
            out.append(((c - cp) * ch.t + e, cp, e))
    return out


def _score_valid(d, ch: Chunks, c: int, m: int, prefix: bool = False):
    """(N,) bool: slots of score ciphertext m of query chunk c holding a real score (`prefix`: offset (c + 1) t)."""
    nq = len(ch.lanes(c))
    i = np.arange(d.N)
    blk, h, lane = i // d.tH, (i % d.tH) // d.t, i % d.t
    u = m * d.d_head + blk
    same = u < ch.t
    U = c * ch.t + ch.t + (1 if prefix else 0)
    ok = (h < d.H_real) & (lane < nq) & (u < U)
    ok &= np.where(same, lane >= u, True)
    return ok


def rope_lanes(rt, x, rope, positions, site):
    """RoPE of every lane at its own position: x cos + rotate(x, N/2) sin' (one level)."""
    ops, d = rt.ops, rt.dims
    pos = tuple(int(p) for p in positions)

    def vec(k):
        def build():
            out = np.zeros((d.d_head, d.H, d.t))
            for lane, p in enumerate(pos):
                out[:, :, lane] = rope.lane_cs(d, p)[k][:, None]
            return out.reshape(-1)
        return build
    a = rt.mult_mask(x, (site + ".c", pos), vec(0))
    b = rt.mult_mask(ops.rotate(x, d.N // 2), (site + ".s", pos), vec(1))
    return ops.add(a, b)


def _lane_vec(d, vals, fill=0.0):
    """A per-query-lane value on every slot (t-periodic)."""
    return lanes_periodic(d.N, d.t, vals, fill)


class _KcExtrapolated:
    """Softmax config whose kc_r scales by C / kc past the calibrated counts instead of holding the last r."""

    def __init__(self, cfg):
        self._c = cfg

    def __getattr__(self, name):
        return getattr(self._c, name)

    def kc_r(self, i, kc):
        r = self._c.kc_r(i, kc)
        if self._c.sm_kc_r and self._c.log2delta2 > 0:
            C = len(self._c.sm_kc_r) // self._c.log2delta2
            if kc > C and r < 1.0:
                r *= C / float(kc)
        return r


def recip_lanes(rt, D, a_vals, b_vals, iters: int, key):
    """1 / D by Goldschmidt with per-lane seed a - b D; D and the reciprocal share one complex ciphertext."""
    ops, d = rt.ops, rt.dims
    av = np.asarray(a_vals, dtype=np.float64)
    bv = np.asarray(b_vals, dtype=np.float64)
    hb = rt.mult_mask(D, ("pf.rcp.hb",) + key, lambda: _lane_vec(d, 0.5 * bv))
    G = rt.add_mask(hb, ("pf.rcp.ha",) + key, lambda: _lane_vec(d, -0.5 * av))
    nb = rt.mult_mask(D, ("pf.rcp.nb",) + key, lambda: _lane_vec(d, -0.5 * bv))
    im = rt.add_mask(nb, ("pf.rcp.pa",) + key, lambda: _lane_vec(d, 0.5 * av))
    P = ops.add(ops.mult(D, G), ops.mult_i(im))
    for _ in range(1, iters):
        P = ops.mult(P, ops.add(ops.add(P, ops.conjugate(P)), 2.0))
    return ops.mult_i(ops.sub(ops.conjugate(P), P))


def _sm_cheb(rt, S, cfg, ch: Chunks, c: int, prefix: bool, m0: int = 0):
    """Score masks and the exp's Chebyshev of chunk c's score ciphertexts m0, m0 + 1, ...: [(z, valid)]."""
    ops, d = rt.ops, rt.dims
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    out = []
    for m, s in enumerate(S, m0):
        valid = _score_valid(d, ch, c, m, prefix)
        s = rt.add_mask(s, ("pf.sm.score", c, m, cfg.clip_lo, mean, ch.T, ch.base, prefix),
                        lambda valid=valid: np.where(valid, -mean, cfg.clip_lo - mean))
        out.append((eval_chebyshev(ops, s, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf), valid))
    return out


def _sm_tail(rt, zs, cfg, ch: Chunks, c: int, prefix: bool, factor=None, m0: int = 0):
    """Squarings, cleanse and active mask (0.5 / kc per lane, times factor[j]) of chunk c's exps."""
    ops, d = rt.ops, rt.dims
    kcs = [p + 1 for p in ch.lanes(c)]
    Z = []
    for j, (z, valid) in enumerate(zs):
        m = m0 + j
        for _ in range(cfg.log2delta1):
            z = ops.square(z)
        z = im_cleanse(ops, z)
        f = 1.0 if factor is None else factor[j]
        lane_scale = np.array([f * 0.5 / kcs[i] if i < len(kcs) else 0.0 for i in range(d.t)])
        key = ("pf.sm.active", c, m, ch.T, ch.base, prefix) + ((f,) if f != 1.0 else ())
        Z.append(rt.mult_mask(z, key,
                              lambda valid=valid, ls=lane_scale: np.where(valid, ls[np.arange(d.N) % d.t], 0.0)))
    return Z


def _sm_joint(rt, za, zb, cfg):
    """A chunk pair's exp Chebyshevs joined into one refresh point and split back; returns the active masks' factors."""
    ops = rt.ops
    n = min(len(za), len(zb))
    fa, fb = [1.0] * len(za), [1.0] * len(zb)
    back = 2.0 ** (-(2 ** cfg.log2delta1))
    for m in range(n):
        X = ops.add(za[m][0], ops.mult_i(zb[m][0]))
        if _JOINT_FORCE:
            ops.bootstrap(X)            # binding under a plan
        else:
            ops.bootstrap_hint(X, ops.headroom(2), True)
        a, b = split(ops, X)
        za[m] = (a, za[m][1]); zb[m] = (b, zb[m][1])
        fa[m] = fb[m] = back
    return fa, fb


def prefill_softmax(rt, S, cfg, ch: Chunks, c: int, prefix: bool = False, Z=None):
    """softmax_thor over chunk c's packed scores S with per-lane (kc-dependent) reciprocal seeds.
    `Z`: precomputed masked exps (the joint path)."""
    ops, d = rt.ops, rt.dims
    pos = ch.lanes(c)
    kcs = [p + 1 for p in pos]
    if Z is None:     # cheb then tail per score ct: the captured op order
        Z = []
        for m in range(len(S)):
            (zc,) = _sm_cheb(rt, [S[m]], cfg, ch, c, prefix, m0=m)
            Z += _sm_tail(rt, [zc], cfg, ch, c, prefix, m0=m)
    lean = ops.sm_gs_first
    per_lane = [_recip_rounds(_KcExtrapolated(cfg), kc, lean) for kc in kcs]
    Y = Z
    for k in range(len(per_lane[0])):
        if k:
            Z = [ops.square(im_cleanse(ops, y)) for y in Y]
        D = Z[0] if len(Z) == 1 else ops.add(Z[0], Z[1])
        for z in Z[2:]:
            ops.inplace_add(D, z)
        D = ops.tag_reduce(rotsum(ops, D, d.tH, d.N), d.tH)
        rnd = [pl[k] for pl in per_lane]
        iters = rnd[0][2]
        a_vals = [al * sc for (al, _b, _i, sc) in rnd]
        b_vals = [be * sc * sc for (_a, be, _i, sc) in rnd]
        R = recip_lanes(rt, D, a_vals, b_vals, iters,
                        (tuple(round(v, 12) for v in a_vals), tuple(round(v, 15) for v in b_vals)))
        if _DBG:
            dv = np.real(D.vec).reshape(-1, d.t)[:, :len(pos)]; rv = np.real(R.vec).reshape(-1, d.t)[:, :len(pos)]
            hv = (np.arange(d.N) % d.tH) // d.t < d.H_real
            dvv = np.real(D.vec)[hv].reshape(-1)
            if k == 0:
                per = np.real(D.vec).reshape(-1, d.tH)[0].reshape(d.H, d.t)[:d.H_real, :len(pos)]
                print("      D>8 at (head,pos):", [(int(h), int(pos[l])) for h, l in zip(*np.nonzero(per > 8))][:20])
            print(f"      round {k} c={c} D [{dvv.min():.3g},{dvv.max():.3g}] a {min(a_vals):.3g}..{max(a_vals):.3g} b {min(b_vals):.3g}..{max(b_vals):.3g} iters {iters} R*D [{np.nanmin(np.real(R.vec*D.vec)[hv]):.3g},{np.nanmax(np.real(R.vec*D.vec)[hv]):.3g}]")
        Y = [ops.mult(z, R) for z in Z]
    return Y


def prefill_attention(rt, Qd, Kd, Vd, cfg, ch: Chunks, kv=None, rope=None, prefix=None):
    """Causal attention on dense complex Q/K/V (scale 1/2); returns dense outputs (scale 1) as the decode's P.V.
    `kv` receives the per-chunk real K/V; `prefix`: (k, v) slot vectors of one public key / value before the prompt."""
    ops, d = rt.ops, rt.dims
    nc = ch.n_chunks
    Q, K, V = [None] * nc, [None] * nc, [None] * nc
    for dst, src in ((Q, Qd), (K, Kd), (V, Vd)):
        for c, X in enumerate(src):
            ops.bootstrap_hint(X, ops.headroom(2), True)
            A, B = split(ops, X)
            dst[2 * c] = A
            if ch.has(2 * c + 1):
                dst[2 * c + 1] = B
    if rope is not None:
        for k in range(nc):
            Q[k] = rope_lanes(rt, Q[k], rope, ch.lanes(k), "pf.rope.q")
            K[k] = rope_lanes(rt, K[k], rope, ch.lanes(k), "pf.rope.k")
    if kv is not None:
        kv["K"], kv["V"] = K, V
    gscale = 1.0 / math.sqrt(d.d_head_real)
    rotK, rotV = {}, {}

    def k_rot(cp, e, c):
        if cp == c:
            return None
        if (cp, e) not in rotK:
            rotK[(cp, e)] = lane_rotate(rt, K[cp], e, "pf.krot")
        return rotK[(cp, e)]

    def v_rot(cp, e, c):
        if cp == c:
            return None
        if (cp, e) not in rotV:
            rotV[(cp, e)] = lane_rotate(rt, V[cp], e, "pf.vrot")
        return rotV[(cp, e)]

    def qkt_stage(c):
        """Chunk c's packed scores and what its P.V needs."""
        offs = _offsets(ch, c)
        nq = len(ch.lanes(c))
        own = [K[c]] + ops.rotate_many(K[c], [-e for e in range(1, nq)])
        u_pre = c * ch.t + ch.t                                    # prefix key's offset
        M = (max([u for u, _, _ in offs] + ([u_pre] if prefix is not None else [])) // d.d_head) + 1
        S = [None] * M
        pre = prefix is not None
        pair = _ATTN_PAIR
        half = 0.5 if pair else 1.0          # paired: scores at 1/2, one cleanse per score ct
        groups = [offs[i:i + 2] for i in range(0, len(offs), 2)] if pair else [[o] for o in offs]

        def smask(m, blk, coef):
            def build():
                valid = _score_valid(d, ch, c, m, pre)
                return np.where(valid & (np.arange(d.N) // d.tH == blk), coef, 0.0)
            return build

        def place(dst, ct, parts, tag):
            """ct times the block masks of `parts` [(m, blk, coef)], accumulated into dst[m]."""
            by_m = {}
            for m, blk, coef in parts:
                by_m.setdefault(m, []).append((blk, coef))
            for m, lst in by_m.items():
                builds = [smask(m, blk, coef) for blk, coef in lst]
                key = (tag, c, m, tuple((blk, complex(coef)) for blk, coef in lst), ch.T, ch.base, pre)
                v = rt.mult_mask(ct, key, lambda builds=builds: sum(b() for b in builds))
                if dst[m] is None:
                    dst[m] = v
                else:
                    ops.inplace_add(dst[m], v)

        with rt.step("qkt"):
            if pre:
                s = rotsum(ops, ops.mult_pt(Q[c], prefix[0]), d.tH, d.N)
                m, blk = divmod(u_pre, d.d_head)
                place(S, s, [(m, blk, half * gscale)], "pf.qkt")
            for grp in groups:
                krs = [own[e] if cp == c else k_rot(cp, e, c) for _, cp, e in grp]
                kc = krs[0] if len(krs) == 1 else ops.add(krs[0], ops.mult_i(krs[1]))
                s = rotsum(ops, ops.mult(Q[c], kc), d.tH, d.N)
                coefs = (half * gscale, -0.5j * gscale)
                place(S, s, [(*divmod(u, d.d_head), coefs[i]) for i, (u, _, _) in enumerate(grp)], "pf.qkt")
            if pair:
                S = [im_cleanse(ops, x) for x in S]
        return dict(S=S, offs=offs, nq=nq, u_pre=u_pre, pre=pre, groups=groups)

    def pv_stage(c, st, P):
        offs, nq, u_pre, pre, groups = st["offs"], st["nq"], st["u_pre"], st["pre"], st["groups"]
        with rt.step("softmax_v"):
            vown = [V[c]] + ops.rotate_many(V[c], [-e for e in range(1, nq)])
            acc = None
            if pre:
                m, blk = divmod(u_pre, d.d_head)
                pb = rt.mult_mask(P[m], ("pf.bsel", blk), lambda blk=blk: np.where(
                    np.arange(d.N) // d.tH == blk, 0.5, 0.0))
                acc = ops.mult_pt(rotsum(ops, pb, d.tH, d.N), prefix[1])
            for grp in groups:
                sel = [None] * len(P)
                coefs = (0.5, -0.5j)
                for i, (u, _, _) in enumerate(grp):
                    m, blk = divmod(u, d.d_head)
                    mk = lambda blk=blk, cf=coefs[i]: np.where(np.arange(d.N) // d.tH == blk, cf, 0.0)
                    v = rt.mult_mask(P[m], ("pf.bsel", blk, complex(coefs[i])), mk)
                    if sel[m] is None:
                        sel[m] = v
                    else:
                        ops.inplace_add(sel[m], v)
                pb = None
                for v in sel:
                    if v is not None:
                        pb = v if pb is None else ops.add(pb, v)
                pb = rotsum(ops, pb, d.tH, d.N)
                vrs = [vown[e] if cp == c else v_rot(cp, e, c) for _, cp, e in grp]
                vc = vrs[0] if len(vrs) == 1 else ops.add(vrs[0], ops.mult_i(vrs[1]))
                prod = ops.mult(pb, vc)
                if acc is None:
                    acc = prod
                else:
                    ops.inplace_add(acc, prod)
            return im_cleanse(ops, acc)

    outs = [None] * nc
    if _SM_JOINT:
        for k in range(0, nc, 2):
            cs = [c for c in (k, k + 1) if c < nc]
            sts = {c: qkt_stage(c) for c in cs}
            with rt.step("softmax"):
                zs = {c: _sm_cheb(rt, sts[c]["S"], cfg, ch, c, sts[c]["pre"]) for c in cs}
                fac = {c: None for c in cs}
                if len(cs) == 2:
                    fac[cs[0]], fac[cs[1]] = _sm_joint(rt, zs[cs[0]], zs[cs[1]], cfg)
                Ps = {c: prefill_softmax(rt, sts[c]["S"], cfg, ch, c, sts[c]["pre"],
                                         Z=_sm_tail(rt, zs[c], cfg, ch, c, sts[c]["pre"], fac[c])) for c in cs}
            for c in cs:
                outs[c] = pv_stage(c, sts[c], Ps[c])
    else:
        for c in range(nc):
            st = qkt_stage(c)
            with rt.step("softmax"):
                P = prefill_softmax(rt, st["S"], cfg, ch, c, st["pre"])
            outs[c] = pv_stage(c, st, P)
    return [join(ops, outs[2 * c], outs[2 * c + 1] if ch.has(2 * c + 1) else None) for c in range(ch.n_ct)]


def prefill_rot_steps(dims) -> set[int]:
    """Rotations the prefill adds to the decode band."""
    steps = set()
    for k in range(1, dims.t):
        steps |= {k, -k, dims.t - k}
    for k in range(1, dims.tp_E):
        steps |= {k, -k}
    r = dims.t
    while r < 512:                           # norm_all's region spread ladder
        steps.add(-r)
        r *= 2
    return {s % dims.N for s in steps if s % dims.N}


# ── dense diagonal linears ──────────────────────────────────────────────────────────────────────────────────────

class DiagLinear:
    """y = x W on dense token-layout ciphertexts, BSGS diagonal method (Halevi-Shoup) on B x B blocks, B = N / t:
        y_m = sum_g rot( sum_{n, j} P[n, m, g, j] * rot(x_n, j t), g b t ); `rows` (pinned) or `pts` (slots)."""

    def __init__(self, n_in, n_out, b, g, pts=None, rows=None, tags=None, bias=None, name=None):
        self.n_in, self.n_out, self.b, self.g = n_in, n_out, b, g
        self.pts, self.rows, self.tags, self.bias, self.name = pts, rows, tags, bias, name

    @staticmethod
    def split(B):
        b = 1 << ((B.bit_length() - 1 + 1) // 2)          # b ~ sqrt(B), a power of two, b >= g
        return b, B // b

    @classmethod
    def slot_vectors(cls, W, N, t, scale=1.0):
        """(n_in, n_out, g, b, N) plaintext slot vectors of W (n_in B, n_out B) in the order the call consumes."""
        W = np.asarray(W, dtype=np.float64) * scale
        B = N // t
        n_in, n_out = W.shape[0] // B, W.shape[1] // B
        b, g = cls.split(B)
        i = np.arange(B)
        out = np.empty((n_out, g, n_in, b, N))
        for m in range(n_out):
            for n in range(n_in):
                blk = W[n * B:(n + 1) * B, m * B:(m + 1) * B]
                for gg in range(g):
                    for j in range(b):
                        k = gg * b + j
                        v = np.repeat(blk[(i + k) % B, i], t)
                        out[m, gg, n, j] = np.roll(v, gg * b * t)
        return out

    @classmethod
    def encode(cls, W, N, t, scale=1.0, bias=None, name=None, pinned=None, core=None, cache=None):
        """From dense W (n_in B, n_out B): pinned coefficient rows (cached at `cache`) when `pinned`, else slots."""
        B = N // t
        n_in, n_out = W.shape[0] // B, W.shape[1] // B
        b, g = cls.split(B)
        bvec = None
        if bias is not None:
            bvec = [np.repeat(np.asarray(bias, dtype=np.float64)[m * B:(m + 1) * B] * scale, t) for m in range(n_out)]
        if pinned is None:
            return cls(n_in, n_out, b, g, pts=cls.slot_vectors(W, N, t, scale), bias=bvec, name=name)
        import os
        from .layout import coeff_rows
        n_rows = n_out * g * n_in * b
        if cache is not None and os.path.exists(cache + ".coeff.npy") and os.path.exists(cache + ".tags.npy"):
            src = np.load(cache + ".coeff.npy", mmap_mode="r")
            tags = np.load(cache + ".tags.npy")
        else:
            pts = cls.slot_vectors(W, N, t, scale).reshape(n_rows, N)
            src = coeff_rows(pts)
            tags = core.pt_tag_fields(np.ascontiguousarray(pts))
            if cache is not None:
                for path, arr in ((cache + ".coeff.npy", src), (cache + ".tags.npy", tags)):
                    tmp = path + f".{os.getpid()}.tmp.npy"
                    np.save(tmp, arr)
                    os.replace(tmp, path)
        rows = pinned(n_rows, 2 * N)
        rows[:] = src
        lin = cls(n_in, n_out, b, g, rows=rows, tags=np.ascontiguousarray(tags), bias=bvec, name=name)
        lin.src = (np.asarray(W, dtype=np.float64), N, t, scale)
        return lin

    def _sum(self, ops, xs, m, gg):
        """sum_{n, j} P[n, m, gg, j] * xs[n][j]."""
        cts = [x for xn in xs for x in xn]
        k = self.n_in * self.b
        r0 = (m * self.g + gg) * k
        if self.rows is not None and hasattr(ops, "mult_coeff_sum"):
            return ops.mult_coeff_sum(cts, self.rows[r0:r0 + k], self.tags[r0:r0 + k], _LazyRows(self, m, gg))
        vecs = self.pts[m, gg].reshape(k, -1)
        return ops.mult_pt_sum(cts, vecs)


class _LazyRows:
    """Slot vectors of one (m, g) group, built only when mult_coeff_sum falls back to mult_pt_sum."""

    def __init__(self, lin, m, gg):
        self.lin, self.m, self.gg = lin, m, gg

    def __array__(self, dtype=None, copy=None):
        lin = self.lin
        W, N, t, scale = lin.src
        B = N // t
        i = np.arange(B)
        out = np.empty((lin.n_in * lin.b, N))
        for n in range(lin.n_in):
            blk = W[n * B:(n + 1) * B, self.m * B:(self.m + 1) * B] * scale
            for j in range(lin.b):
                k = self.gg * lin.b + j
                out[n * lin.b + j] = np.roll(np.repeat(blk[(i + k) % B, i], t), self.gg * lin.b * t)
        return out if dtype is None else out.astype(dtype)


def diag_apply(rt, Xs, lins, scale_bias=None):
    """DiagLinears `lins` (same n_in) on input blocks Xs of one ciphertext group; per linear, its n_out blocks."""
    ops, d = rt.ops, rt.dims
    b = lins[0].b
    for x in Xs:
        ops.bootstrap_hint(x, ops.headroom(2), True)       # the products (one unit deeper) stay under the limit
    xs = [[x] + ops.rotate_many(x, [j * d.t for j in range(1, b)]) for x in Xs]
    for xn in xs:
        for x in xn:
            ops.realize(x)
    out = []
    for lin in lins:
        blocks = []
        for m in range(lin.n_out):
            acc = None
            for gg in range(lin.g):
                tmp = lin._sum(ops, xs, m, gg)
                if gg:
                    tmp = ops.rotate(tmp, gg * lin.b * d.t)
                if acc is None:
                    acc = tmp
                else:
                    ops.inplace_add(acc, tmp)
            blocks.append(acc)
        out.append(blocks)
    return out


def diag_bias(rt, y, lin, m, ch: Chunks, c: int):
    """Output block m of ciphertext c plus (b + i b) on the filled lanes."""
    if lin.bias is None:
        return y
    d = rt.dims
    na = len(ch.lanes(2 * c))
    nb = len(ch.lanes(2 * c + 1)) if ch.has(2 * c + 1) else 0
    lane = np.arange(d.N) % d.t
    fac = (lane < na).astype(np.float64) + 1j * (lane < nb)
    return rt.add_mask(y, ("pf.dbias", lin.name or id(lin), m, na, nb), lambda: lin.bias[m] * fac)


def diag_rot_steps(dims) -> set[int]:
    b, g = DiagLinear.split(dims.N // dims.t)
    steps = {j * dims.t for j in range(1, b)} | {k * b * dims.t for k in range(1, g)}
    return {s % dims.N for s in steps if s % dims.N}


# ── lockstep halves ─────────────────────────────────────────────────────────────────────────────────────────────

class Pair:
    """The two real halves (chunks 2c, 2c + 1) of a value under LockstepOps."""
    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a, self.b = a, b


class LockstepOps:
    """Ops on Pairs, each op on both halves; an operand near the ceiling is joined, bootstrapped once and split back."""

    total = 0                             # joint refreshes since import

    def __init__(self, ops):
        self._o = ops
        self.refreshes = 0

    def __getattr__(self, name):
        return getattr(self._o, name)

    def _two(self, f, x, y=None):
        if y is None:
            return Pair(f(x.a), f(x.b))
        if isinstance(y, Pair):
            return Pair(f(x.a, y.a), f(x.b, y.b))
        return Pair(f(x.a, y), f(x.b, y))

    def _ceiling(self):
        """Two units under the refresh envelope (PF_LOCKSTEP_TOP, BTS_MAX_INPUT_LEVEL, or the session limit)."""
        import os
        o = self._o
        top = int(os.environ.get("PF_LOCKSTEP_TOP") or os.environ.get("BTS_MAX_INPUT_LEVEL") or o.level_limit())
        return top - 2 * o.unit

    def _guard(self, x):
        o = self._o
        if isinstance(x, Pair) and max(o.lvl(x.a), o.lvl(x.b)) > self._ceiling():
            X = o.add(x.a, o.mult_i(x.b))
            if getattr(o, "folds", False):
                o.fold_bootstrap(X, self._o.inf.slots, 2)     # the split's 1/2 inside the refresh
                half = None
            else:
                o.bootstrap(X)
                half = 0.5
            o.device_sync()               # a conjugate right after a complex refresh must see it landed (poly.py)
            A, B = split(o, X)
            x = Pair(A, B) if half is None else Pair(o.mult(A, half), o.mult(B, half))
            self.refreshes += 1
            LockstepOps.total += 1
        return x

    def add(self, x, y): return self._two(self._o.add, x, y)
    def sub(self, x, y): return self._two(self._o.sub, x, y)
    def negate(self, x): return self._two(self._o.negate, x)
    def conjugate(self, x): return self._two(self._o.conjugate, x)
    def copy(self, x): return self._two(self._o.copy, x)

    def mult(self, x, y):
        x = self._guard(x)
        y = self._guard(y) if isinstance(y, Pair) else y
        return self._two(self._o.mult, x, y)

    def square(self, x):
        return self._two(self._o.square, self._guard(x))

    def inplace_add(self, x, y):
        if isinstance(y, Pair):
            self._o.inplace_add(x.a, y.a); self._o.inplace_add(x.b, y.b)
        else:
            self._o.inplace_add(x.a, y); self._o.inplace_add(x.b, y)

    def mult_pt(self, x, v, key=None, kind="weight", tagged=True):
        x = self._guard(x)
        return Pair(self._o.mult_pt(x.a, v, key=key, kind=kind, tagged=tagged),
                    self._o.mult_pt(x.b, v, key=key, kind=kind, tagged=tagged))

    def add_pt(self, x, v, key=None, kind="weight", tagged=True):
        return Pair(self._o.add_pt(x.a, v, key=key, kind=kind, tagged=tagged),
                    self._o.add_pt(x.b, v, key=key, kind=kind, tagged=tagged))

    def lvl(self, x): return max(self._o.lvl(x.a), self._o.lvl(x.b)) if isinstance(x, Pair) else self._o.lvl(x)
    def level(self, x): return max(self._o.level(x.a), self._o.level(x.b)) if isinstance(x, Pair) else self._o.level(x)

    def bootstrap_hint(self, x, thr, acct=False):
        return x                          # the refreshes are _guard's

    def tag_reduce(self, x, stride):
        return self._two(lambda c: self._o.tag_reduce(c, stride), x)

    def realize(self, x):
        return self._two(self._o.realize, x)


def lockstep_rt(rt):
    """rt with LockstepOps (shared mask caches)."""
    import copy
    r = copy.copy(rt)
    r.ops = LockstepOps(rt.ops)
    return r


# ── shared-chain norms ──────────────────────────────────────────────────────────────────────────────────────────

def _region_mask(d, R, k, val):
    """val on the slots of region k (slot s: (s // t) % R == k), zero elsewhere."""
    return np.where((np.arange(d.N) // d.t) % R == k, val, 0.0)


def norm_all(rt, Xs, cfg, ch: Chunks, rms: bool = False, max_period: int = 512):
    """norm_pair on every ciphertext with ONE inverse-sqrt chain: chunk k's statistic goes to region k of an
    (R t)-periodic ciphertext (R t <= max_period) and is spread back by a log R ladder."""
    ops, d = rt.ops, rt.dims
    N, t, dr = d.N, d.t, d.dim
    halves = []                              # (ct, half, lane scales, input, statistic)
    for c, X in enumerate(Xs):
        for half in (0, 1):
            if not ch.has(2 * c + half):
                continue
            pos = ch.lanes(2 * c + half)
            cs = tuple(round(math.sqrt(cfg.c_eff_sq(p)), 12) for p in pos)
            x2 = im_cleanse(ops, X if half == 0 else ops.mult_i(X))      # 2 Re X, or 2 Re(iX) = -2 B
            xs = rt.mult_mask(x2, ("ln.center.L", cs), lambda cs=cs: lanes_mask(N, dr, t, [0.5 * c_ for c_ in cs]))
            if rms:
                S = ops.tag_reduce(rotsum(ops, ops.square(xs), t, N), t)
                halves.append((c, half, cs, xs, S))
            else:
                mean = ops.tag_reduce(rotsum(ops, xs, t, N), t)
                mean = rt.mult_mask(mean, ("ln.scalemask.L", len(cs)),
                                    lambda n=len(cs): lanes_mask(N, dr, t, [-1.0 / dr] * n))
                centered = ops.add(xs, mean)
                var = ops.tag_reduce(rotsum(ops, ops.square(centered), t, N), t)
                halves.append((c, half, cs, centered, var))
    per = max(1, max_period // t)
    outs = {}
    for g0 in range(0, len(halves), per):
        grp = halves[g0:g0 + per]
        R = 1 << (len(grp) - 1).bit_length()
        V = None
        eps = np.zeros(N)
        for k, (_c, _h, cs, _x, st) in enumerate(grp):
            sc = 1.0 if rms else 1.0 / dr
            m = rt.mult_mask(st, ("pf.nreg", R, k, sc), lambda R=R, k=k, sc=sc: _region_mask(d, R, k, sc))
            V = m if V is None else ops.add(V, m)
            e = [(dr if rms else 1.0) * cfg.epsilon * c_ * c_ for c_ in cs]
            lane_e = lanes_periodic(N, t, e, (dr if rms else 1.0) * 0.5 * (cfg.cheb_lo + cfg.cheb_hi)
                                    if cfg.cheb_coeffs else 1.0)
            eps += np.where((np.arange(N) // t) % R == k, lane_e, 0.0)
        for k in range(len(grp), R):         # empty regions: a value in the seed's band
            eps += _region_mask(d, R, k, (dr if rms else 1.0) * 0.5 * (cfg.cheb_lo + cfg.cheb_hi)
                                if cfg.cheb_coeffs else 1.0)
        V = rt.add_mask(V, ("pf.neps", rms, R, tuple(g[2] for g in grp), cfg.epsilon), lambda eps=eps: eps)
        V = ops.tag_reduce(V, R * t)
        lo, hi = (dr * cfg.cheb_lo, dr * cfg.cheb_hi) if rms else (cfg.cheb_lo, cfg.cheb_hi)
        if not (cfg.cheb_coeffs and ops.ln_cheb):
            raise ValueError("norm_all: needs the Chebyshev inverse-sqrt seed (LN_CHEB=1, cheb_* config)")
        init = eval_chebyshev(ops, V, cfg.cheb_coeffs, lo, hi)
        xsc = 1.0 / ((dr if rms else 1.0) * cfg.inv_out_scale * cfg.inv_out_scale)
        from .poly import inv_sqrt_newton_d2
        inv = inv_sqrt_newton_d2(ops, V, init, cfg.cheb_nr_iters, xsc)
        ops.bootstrap_hint(inv, ops.headroom(4))
        for k, (c, half, _cs, x, _st) in enumerate(grp):
            ik = rt.mult_mask(inv, ("pf.nreg", R, k, 1.0), lambda R=R, k=k: _region_mask(d, R, k, 1.0))
            if R > 1:
                ik = ops.rotate_and_sum(ik, t, R * t)
            outs[(c, half)] = im_cleanse(ops, ops.mult(x, ik))
    res = []
    for c in range(len(Xs)):
        nA = outs[(c, 0)]
        res.append(nA if (c, 1) not in outs else join(ops, nA, ops.negate(outs[(c, 1)])))
    return res


# PF_JOINT_FORCE=1: the joint GELU / softmax refresh is a deliberate bootstrap instead of a hint
_JOINT_FORCE = os.environ.get("PF_JOINT_FORCE", "0") not in ("", "0")

def gelu_joint(rt, A, B, cfg, mask):
    """Folded THOR GELU of real halves A, B (x / xmax) with one refresh of p1(A) + i p1(B) between P1 and P2.
    Returns (g(A), g(B)) times the mask."""
    from .poly import eval_polynomial_ps
    ops = rt.ops
    if not (cfg.method == "thor_composite" and cfg.thor_p1_cheb and ops.gelu_fold):
        raise ValueError("gelu_joint: the folded THOR composite with Chebyshev P1 / P2 only")
    m = np.asarray(mask[1](1.0))
    if _JOINT_FORCE and np.all(m == m.flat[0]):
        # constant mask: P2's domain map and the split's 2 ride P1's coefficients, the mask P2's
        s_out = 0.5 * cfg.xmax * float(np.real(m.flat[0]))
        a2, b2 = cfg.thor_p2_a, cfg.thor_p2_b
        al, be = 2.0 / (b2 - a2), (a2 + b2) / (b2 - a2)
        c1 = [0.5 * al * float(c) for c in cfg.thor_p1_cheb]
        c1[0] -= 0.5 * be
        p1 = [eval_chebyshev(ops, x, c1, cfg.thor_p1_a, cfg.thor_p1_b) for x in (A, B)]
        X = ops.add(p1[0], ops.mult_i(p1[1]))
        ops.bootstrap(X)
        ya, yb = split(ops, X)                                 # P2's domain variable
        c2 = [s_out * float(c) for c in cfg.thor_p2_cheb]
        return [ops.mult(ops.add(eval_chebyshev(ops, y, c2, -1.0, 1.0), 0.5 * s_out), im_cleanse(ops, x))
                for x, y in ((A, ya), (B, yb))]
    p1 = [eval_chebyshev(ops, x, cfg.thor_p1_cheb, cfg.thor_p1_a, cfg.thor_p1_b) for x in (A, B)]
    X = ops.add(p1[0], ops.mult_i(p1[1]))
    ops.bootstrap_hint(X, ops.headroom(4), True)
    p1a, p1b = split(ops, X)                                   # 2 p1(A), 2 p1(B)
    out = []
    for x, p in ((A, p1a), (B, p1b)):
        g = eval_chebyshev(ops, p, cfg.thor_p2_cheb, 2.0 * cfg.thor_p2_a, 2.0 * cfg.thor_p2_b)
        g = ops.add(g, 0.5)
        g = ops.mult(g, im_cleanse(ops, x))
        out.append(rt.mult_mask(g, (mask[0], cfg.xmax), lambda: mask[1](0.5 * cfg.xmax)))
    return out
