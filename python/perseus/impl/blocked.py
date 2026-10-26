"""Channel-blocked prefill: the hidden over K ciphertexts of nch channels, each holding whole heads."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .layout import Dims, lanes_mask, lanes_periodic, ladder_chunks
from .poly import eval_chebyshev, im_cleanse, inv_sqrt_newton_d2, rotsum
from .prefill import Chunks, join, prefill_attention


@dataclass(frozen=True)
class Blocks:
    """K = hid / nch channel blocks of the model dims `d`; group k holds heads [k H', (k + 1) H')."""
    d: Dims
    nch: int = 512
    G: int = 1                  # query heads per KV head: K / V blocks hold each KV head once

    def __post_init__(self):
        if self.d.hid % self.nch or self.nch % self.d.d_head or self.t > self.d.d_head:
            raise ValueError("Blocks: nch must split hid into whole heads, with t <= d_head (score ct coverage)")
        if self.G > 1 and (self.K % self.G or self.d.H_real != self.d.H):
            raise ValueError("Blocks: GQA needs whole KV blocks and no padded heads")

    @property
    def K(self): return self.d.hid // self.nch
    @property
    def t(self): return self.d.N // self.nch
    @property
    def hg(self): return self.nch // self.d.d_head          # heads per block
    @property
    def Kkv(self): return self.K // self.G                  # K / V blocks; Q block q reads K / V block q % Kkv

    def real(self, k):
        return int(min(max(self.d.dim - k * self.nch, 0), self.nch))

    def group_dims(self, k):
        """The head group's own geometry (t lanes, H' heads)."""
        hr = int(min(max(self.d.H_real - k * self.hg, 0), self.hg))
        return Dims(self.d.N, self.nch, self.d.d_head_real * hr, self.hg, hr, self.d.N, self.d.N)

    def head_perm(self):
        """Head-grouped index of interleaved feature r = j H + h: block (h % G) Kkv + kv // H', local j H' + kv % H'."""
        r = np.arange(self.d.hid)
        H, hg = self.d.H, self.hg
        j, h = r // H, r % H
        kv, g = h // self.G, h % self.G
        return (g * self.Kkv + kv // hg) * self.nch + j * hg + (kv % hg)

    def kv_perm(self):
        """(source column, index in the K / V blocks) of each KV head once."""
        r = np.arange(self.d.hid)
        h = r % self.d.H
        keep = h % self.G == 0
        return r[keep], self.head_perm()[keep]

    def kv_cols(self, W):
        src, dst = self.kv_perm()
        out = np.zeros((W.shape[0], self.Kkv * self.nch), W.dtype)
        out[:, dst] = W[:, src]
        return out

    def kv_vec(self, v):
        src, dst = self.kv_perm()
        out = np.zeros(self.Kkv * self.nch, np.asarray(v).dtype)
        out[dst] = np.asarray(v)[src]
        return out

    def group_cols(self, W):
        out = np.zeros_like(W)
        out[:, self.head_perm()] = W
        return out

    def group_rows(self, W):
        out = np.zeros_like(W)
        out[self.head_perm(), :] = W
        return out

    def group_vec(self, v):
        out = np.zeros_like(v)
        out[self.head_perm()] = v
        return out


class SubRt:
    """rt with the dims of one head group; mask keys namespaced by those dims (their masks differ)."""

    def __init__(self, rt, dims, tag):
        self._rt, self.dims, self._tag = rt, dims, tag
        rt.sub_tags = frozenset(rt.sub_tags) | {tag}    # its masks keep their own sites (Rt.site)

    def __getattr__(self, name):
        return getattr(self._rt, name)

    def _key(self, key):
        return (self._tag,) + (key if isinstance(key, tuple) else (key,))

    def mask(self, key, build):
        return self._rt.mask(self._key(key), build)

    def mult_mask(self, ct, key, build):
        return self._rt.mult_mask(ct, self._key(key), build)

    def add_mask(self, ct, key, build):
        return self._rt.add_mask(ct, self._key(key), build)

    def mult_masks(self, ct, keys, builds):
        return self._rt.mult_masks(ct, [self._key(k) for k in keys], builds)


def sub_rt(rt, B: Blocks, k: int):
    g = B.group_dims(k)
    return SubRt(rt, g, f"hg{g.H}.{g.H_real}.{g.t}")


# ── prompt in / out ─────────────────────────────────────────────────────────────────────────────────────────────────

def blocked_slots(rows, ch: Chunks, B: Blocks):
    """encode_blocked's slot vectors, [g][k]."""
    N, t = B.d.N, B.t
    rows = np.asarray(rows, dtype=np.float64)
    full = np.zeros((rows.shape[0], B.d.hid))
    full[:, :rows.shape[1]] = rows
    idx = np.arange(B.nch) * t
    out = []
    for g in range(ch.n_ct):
        blk = []
        for k in range(B.K):
            v = np.zeros(N, dtype=np.complex128)
            for half in (0, 1):
                if ch.has(2 * g + half):
                    for lane, p in enumerate(ch.idx(2 * g + half)):
                        v[idx + lane] += (1j if half else 1.0) * full[p, k * B.nch:(k + 1) * B.nch]
            blk.append(v)
        out.append(blk)
    return out


def encode_blocked(rt, rows, ch: Chunks, B: Blocks):
    """Xs[g][k]: chunk pair g (2g real, 2g + 1 imaginary), channel block k (feature k nch + r at slot r t + lane)."""
    ops = rt.ops
    out = []
    for blk in blocked_slots(rows, ch, B):
        row = []
        for v in blk:
            zero = ops.core.encode_token_input(ops.inf, np.zeros(B.d.dim))
            row.append(ops.add_pt(zero, v))
        out.append(row)
    return out


def decrypt_blocked(rt, Xs, ch: Chunks, B: Blocks, d_real: int):
    ops = rt.ops
    rows = np.zeros((ch.T, B.d.hid))
    idx = np.arange(B.nch) * B.t
    for g, blk in enumerate(Xs):
        for k, X in enumerate(blk):
            v = ops.decrypt_slots_complex(X)
            for half, part in ((0, v.real), (1, v.imag)):
                for lane, p in enumerate(ch.idx(2 * g + half)):
                    rows[p, k * B.nch:(k + 1) * B.nch] = part[idx + lane]
    return rows[:, :d_real]


def token_of(rt, Xs, ch: Chunks, B: Blocks, p: int):
    """Position p of Xs[g][k] as one decode-layout token row (feature i at slot i d.t, real): its lane and axis are
    masked out and aligned, then each block's stride shrinks from B.t to d.t by one masked rotation per bit of the
    feature index (bit j moves its features (B.t - d.t) 2^j slots down), and block k is moved to feature k nch."""
    ops, d = rt.ops, B.d
    c = p // ch.t
    g, half, lane = c // 2, c % 2, p - c * ch.t
    out = None
    for k in range(B.K):
        n = B.real(k)
        if n == 0:
            continue
        r = np.arange(n)
        pos = r * B.t

        def at(slots):
            v = np.zeros(d.N)
            v[slots] = 1.0
            return v
        # w + conj(w) is the axis of chunk c: Re(x) for w = x/2, Im(x) for w = -i x/2
        w = rt.mult_mask(Xs[g][k], ("tok_of", B.nch, k, "sel", lane, half),
                         lambda: (-0.5 if half else 0.5) * at(pos + lane))
        w = ops.rotate(ops.mult_i(w) if half else w, lane)
        for j in range(max(1, int(n - 1).bit_length())):
            moved = (r >> j) & 1 == 1
            step = (B.t - d.t) << j
            keep = rt.mult_mask(w, ("tok_of", B.nch, k, "bit", j, 0), lambda: at(pos[~moved]))
            move = rt.mult_mask(w, ("tok_of", B.nch, k, "bit", j, 1), lambda: at(pos[moved]))
            w = ops.add(keep, ops.rotate(move, step))
            pos = np.where(moved, pos - step, pos)
        if k:
            w = ops.rotate(w, (-k * B.nch * d.t) % d.N)
        out = w if out is None else ops.add(out, w)
    return ops.add(out, ops.conjugate(out))


def token_of_rot_steps(B: Blocks, lane: int) -> set[int]:
    """The rotation steps token_of takes for a position on `lane`."""
    d = B.d
    steps = {lane} | {-k * B.nch * d.t for k in range(1, B.K)}
    steps |= {(B.t - d.t) << j for j in range(max(1, int(B.nch - 1).bit_length()))}
    return {s % d.N for s in steps if s % d.N}


def name_blocked(inf, Xs):
    """Name every input after the first (anon numbering differs between a capture and its planned run)."""
    if hasattr(inf, "name_ct"):
        flat = [x for blk in Xs for x in blk]
        for c, X in enumerate(flat[1:], 1):
            inf.name_ct(X, f"pf_in{c}")


# ── norms ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def _region(N, t, R, k, val):
    return np.where((np.arange(N) // t) % R == k, val, 0.0)


def norm_blocked(rt, Xs, cfg, ch: Chunks, B: Blocks, rms: bool = False, max_period: int = 512):
    """norm_all for Xs[g][k]: 2 (LN(A) + i LN(B)) per block, statistic summed over the K blocks."""
    ops, N, t, dr = rt.ops, B.d.N, B.t, B.d.dim
    halves = []                              # (g, half, lane scales, [x_k], statistic)
    for g, blk in enumerate(Xs):
        for half in (0, 1):
            if not ch.has(2 * g + half):
                continue
            pos = ch.lanes(2 * g + half)
            cs = tuple(round(math.sqrt(cfg.c_eff_sq(p)), 12) for p in pos)
            xs = []
            for k, X in enumerate(blk):
                if B.real(k) == 0:
                    xs.append(None)
                    continue
                x2 = im_cleanse(ops, X if half == 0 else ops.mult_i(X))
                xs.append(rt.mult_mask(x2, ("bl.center", k, cs),
                                       lambda k=k, cs=cs: lanes_mask(N, B.real(k), t, [0.5 * c_ for c_ in cs])))
            live = [x for x in xs if x is not None]
            if rms:
                sq = None
                for x in live:
                    s2 = ops.square(x)
                    sq = s2 if sq is None else ops.add(sq, s2)
                S = ops.tag_reduce(rotsum(ops, sq, t, N), t)
                halves.append((g, half, cs, xs, S))
            else:
                tot = live[0]
                for x in live[1:]:
                    tot = ops.add(tot, x)
                mean = ops.tag_reduce(rotsum(ops, tot, t, N), t)
                cen, sq = [], None
                for k, x in enumerate(xs):
                    if x is None:
                        cen.append(None)
                        continue
                    mk = rt.mult_mask(mean, ("bl.mean", k, len(cs)),
                                      lambda k=k, n=len(cs): lanes_mask(N, B.real(k), t, [-1.0 / dr] * n))
                    c_ = ops.add(x, mk)
                    cen.append(c_)
                    s2 = ops.square(c_)
                    sq = s2 if sq is None else ops.add(sq, s2)
                var = ops.tag_reduce(rotsum(ops, sq, t, N), t)
                halves.append((g, half, cs, cen, var))
    per = max(1, max_period // t)
    outs = {}
    for g0 in range(0, len(halves), per):
        grp = halves[g0:g0 + per]
        R = 1 << (len(grp) - 1).bit_length()
        V = None
        eps = np.zeros(N)
        mid = (dr if rms else 1.0) * 0.5 * (cfg.cheb_lo + cfg.cheb_hi) if cfg.cheb_coeffs else 1.0
        for k, (_g, _h, cs, _x, st) in enumerate(grp):
            sc = 1.0 if rms else 1.0 / dr
            m = rt.mult_mask(st, ("bl.nreg", R, k, sc), lambda R=R, k=k, sc=sc: _region(N, t, R, k, sc))
            V = m if V is None else ops.add(V, m)
            e = [(dr if rms else 1.0) * cfg.epsilon * c_ * c_ for c_ in cs]
            eps += np.where((np.arange(N) // t) % R == k, lanes_periodic(N, t, e, mid), 0.0)
        for k in range(len(grp), R):
            eps += _region(N, t, R, k, mid)
        V = rt.add_mask(V, ("bl.neps", rms, R, tuple(x[2] for x in grp), cfg.epsilon), lambda eps=eps: eps)
        V = ops.tag_reduce(V, R * t)
        lo, hi = (dr * cfg.cheb_lo, dr * cfg.cheb_hi) if rms else (cfg.cheb_lo, cfg.cheb_hi)
        if not (cfg.cheb_coeffs and ops.ln_cheb):
            raise ValueError("norm_blocked: needs the Chebyshev inverse-sqrt seed (LN_CHEB=1, cheb_* config)")
        init = eval_chebyshev(ops, V, cfg.cheb_coeffs, lo, hi)
        xsc = 1.0 / ((dr if rms else 1.0) * cfg.inv_out_scale * cfg.inv_out_scale)
        inv = inv_sqrt_newton_d2(ops, V, init, cfg.cheb_nr_iters, xsc)
        ops.bootstrap_hint(inv, ops.headroom(4))
        for k, (g, half, _cs, xs, _st) in enumerate(grp):
            ik = rt.mult_mask(inv, ("bl.nreg", R, k, 1.0), lambda R=R, k=k: _region(N, t, R, k, 1.0))
            if R > 1:
                ik = ops.rotate_and_sum(ik, t, R * t)
            outs[(g, half)] = [None if x is None else im_cleanse(ops, ops.mult(x, ik)) for x in xs]
    res = []
    for g, blk in enumerate(Xs):
        row = []
        for k in range(len(blk)):
            a = outs[(g, 0)][k]
            if a is None:
                row.append(ops.sub(blk[k], blk[k]))
                continue
            row.append(a if (g, 1) not in outs else join(ops, a, ops.negate(outs[(g, 1)][k])))
        res.append(row)
    return res


def shift_blocked(rt, X2, shift, ch: Chunks, g: int, k: int, B: Blocks, scale: float = 1.0):
    """X2 + scale (shift + i shift) of block k on every filled lane (the folded LayerNorm shift)."""
    N, t = B.d.N, B.t
    na = len(ch.lanes(2 * g))
    nb = len(ch.lanes(2 * g + 1)) if ch.has(2 * g + 1) else 0
    sh = np.zeros(B.d.hid)
    s0 = np.asarray(shift, dtype=np.float64).ravel()[:B.d.dim]
    sh[:s0.shape[0]] = s0
    sk = sh[k * B.nch:(k + 1) * B.nch]
    if not np.any(sk):
        return X2

    def build():
        out = np.zeros(N, dtype=np.complex128)
        idx = np.arange(B.nch) * t
        for lane in range(t):
            out[idx + lane] = scale * sk * ((1.0 if lane < na else 0.0) + (1j if lane < nb else 0.0))
        return out
    return rt.add_mask(X2, ("bl.shift", id(shift), k, na, nb, scale), build)


def attention_blocked(rt, Q, K, V, cfg, ch: Chunks, B: Blocks, rope=None, prefixes=None, kv=None, layer=0):
    """prefill_attention per head group: O[g][k]; GQA K / V block b serves query blocks b, Kkv + b, ...
    `kv`: a list receiving each K / V block's per-chunk real K (post-RoPE) / V (GQA only).
    PF_ATTN=ccmm: the CCMM path (ccmm_attn.py)."""
    from . import ccmm
    if ccmm.enabled():
        from .ccmm_attn import attention_ccmm
        if kv is None:
            kv = [dict() for _ in range(B.Kkv)]
        return attention_ccmm(rt, Q, K, V, cfg, ch, B, rope=rope, prefixes=prefixes, kv=kv, layer=layer)
    if B.G > 1:
        outs = [[None] * B.K for _ in Q]
        for b in range(B.Kkv):
            qs = [b + g * B.Kkv for g in range(B.G)]
            pre = prefixes[b] if prefixes is not None else None
            o = prefill_attention(sub_rt(rt, B, b), [[q[i] for q in Q] for i in qs], [x[b] for x in K],
                                  [x[b] for x in V], cfg, ch, rope=rope, prefix=pre, shared=True,
                                  kv=None if kv is None else kv[b])
            for i, oi in zip(qs, o):
                for g in range(len(Q)):
                    outs[g][i] = oi[g]
        return outs
    outs = []
    for k in range(B.K):
        if B.group_dims(k).H_real == 0:
            outs.append([None] * len(Q))
            continue
        pre = prefixes[k] if prefixes is not None else None
        outs.append(prefill_attention(sub_rt(rt, B, k), [q[k] for q in Q], [x[k] for x in K], [x[k] for x in V],
                                      cfg, ch, rope=rope, prefix=pre))
    ops = rt.ops
    res = []
    for g in range(len(Q)):
        row = []
        for k in range(B.K):
            o = outs[k][g]
            row.append(o if o is not None else ops.sub(Q[g][k], Q[g][k]))
        res.append(row)
    return res


# ── the last position alone: a block whose other positions feed nothing (Prefill-128's last block, before the tail) ──

def _last_lane(ch: Chunks, B: Blocks, p: int):
    """(half, lane) of position p, which must be the last lane of chunk pair 0."""
    c = p // ch.t
    if ch.n_ct != 1 or c // 2 or p - c * ch.t != B.t - 1:
        raise ValueError(f"last-position block: position {p} must sit on the last lane of the first chunk pair")
    return c % 2, B.t - 1


def token_bcast(rt, row, ch: Chunks, B: Blocks, p: int):
    """Position p of the blocked row (chunk pair 0) on every lane of both halves: one level."""
    ops, N = rt.ops, B.d.N
    half, lane = _last_lane(ch, B, p)
    out = []
    for k, x in enumerate(row):
        sel = np.where(np.arange(N) % B.t == lane, -0.5j if half else 0.5, 0.0)
        w = im_cleanse(ops, rt.mult_mask(x, ("bl.tok", B.nch, half, lane), lambda sel=sel: sel))
        w = rotsum(ops, w, 1, B.t)                  # w[i] = sum_k w[i + k]: the last lane reaches its group's lanes
        out.append(ops.add(w, ops.mult_i(w)))
    return out


def attention_last(rt, Q, K, V, cfg, ch: Chunks, B: Blocks, p: int):
    """attention_blocked for query position p = T - 1 alone (every key valid): O[0][k] holds p's output on every lane
    of both halves. Per head group k, q_p on every lane times the complex K (keys of chunk 0 real, chunk 1 imaginary)
    is summed over the head dimension; the groups' scores share one score ciphertext, keys of half h of group k on
    the blocks of class h K + k (period 2 K tH), so one softmax serves every head; its key sums run over the lanes
    (the halves first) and return to every lane from the first. P.V pairs the halves as P_0 - i P_1 against V."""
    from . import smcheb
    from .attention import _recip_rounds
    from .prefill import _KcExtrapolated, recip_lanes
    ops, N, t = rt.ops, B.d.N, B.t
    half, lane = _last_lane(ch, B, p)
    if ch.T != 2 * t or len(Q) != 1:
        raise ValueError("attention_last: T must fill both halves of one chunk pair")
    ks = [k for k in range(B.K) if B.group_dims(k).H_real]
    gd = B.group_dims(0)
    tH, R = gd.tH, 2 * B.K
    cls = (np.arange(N) // tH) % R
    head = (np.arange(N) % tH) // t
    valid = head < np.array([B.group_dims(r % B.K).H_real for r in range(R)])[cls]
    gscale = 1.0 / math.sqrt(gd.d_head_real)
    T = None
    for k in ks:
        sel = np.where(np.arange(N) % t == lane, -1j if half else 1.0, 0.0)
        q = im_cleanse(ops, rt.mult_mask(Q[0][k], ("bl.last.q", B.nch, half, lane), lambda sel=sel: sel))
        q = rotsum(ops, q, 1, t)                                    # q_p on every lane: real, scale 1
        s = rotsum(ops, ops.mult(q, K[0][k]), tH, N)                 # (s_0 + i s_1) / 2 on every block
        m = np.where(cls == k, gscale, 0.0) + np.where(cls == B.K + k, -1j * gscale, 0.0)
        s = rt.mult_mask(s, ("bl.last.s", B.nch, k, gscale), lambda m=m: m)
        T = s if T is None else ops.add(T, s)
    S = im_cleanse(ops, T)                                         # scores of half h of group k on class h K + k
    kc = p + 1
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    S = rt.add_mask(S, ("bl.last.score", B.nch, cfg.clip_lo, mean), lambda: np.where(valid, -mean, cfg.clip_lo - mean))
    z = eval_chebyshev(ops, S, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf)
    for _ in range(cfg.log2delta1):
        z = ops.square(z)
    Z = rt.mult_mask(im_cleanse(ops, z), ("bl.last.active", B.nch, kc), lambda: np.where(valid, 0.5 / kc, 0.0))
    cheb = getattr(ops, "sm_cheb_gs", False)
    rounds = smcheb.rounds(cfg, kc) if cheb else _recip_rounds(_KcExtrapolated(cfg), kc, ops.sm_gs_first)
    first = np.arange(t) == 0
    Y = Z
    for i, r in enumerate(rounds):
        if i:
            Z = ops.square(im_cleanse(ops, Y))
        D = ops.add(Z, ops.rotate(Z, B.K * tH))                     # both halves' keys
        D = rotsum(ops, D, 1, t)                                   # the key sum on each group's first lane
        a, b, iters = r[0] * r[3], r[1] * r[3] * r[3], r[2]
        cv = None if not cheb else ([[r[4][0][j]] * t for j in range(iters - 1)], [r[4][1]] * t)
        # seeds on the first lane only: the reciprocal is zero elsewhere, and the ladder spreads it to every lane
        rc = recip_lanes(sub_rt(rt, B, 0), D, np.where(first, a, 0.0), np.where(first, b, 0.0), iters,
                         ("last", round(a, 12), round(b, 15)), cv)
        Y = ops.mult(Z, rotsum(ops, rc, -1, t))
    outs = []
    for k in range(B.K):
        if k not in ks:
            outs.append(ops.sub(Q[0][k], Q[0][k]))
            continue
        m = np.where(cls == k, 1.0, 0.0) + np.where(cls == B.K + k, -1j, 0.0)
        pc = rotsum(ops, rt.mult_mask(Y, ("bl.last.pv", B.nch, k), lambda m=m: m), tH, R * tH)   # P_0 - i P_1
        o = im_cleanse(ops, rotsum(ops, ops.mult(pc, V[0][k]), -1, t))      # sum over the lanes, on the last lane
        o = rt.mult_mask(o, ("bl.last.o", B.nch, lane), lambda: (np.arange(N) % t == lane).astype(float))
        o = rotsum(ops, o, 1, t)
        outs.append(ops.add(o, ops.mult_i(o)))
    return [outs]


def blocked_rot_steps(B: Blocks, radix_bits: int = 0) -> set[int]:
    """Signed rotation steps (keys match exactly): lane moves, region spread (-r), radix ladder groups."""
    gd = B.group_dims(0)
    steps = {k for k in range(1, B.t)} | {-k for k in range(1, B.t)}
    r = B.t
    while r < 512:
        steps.add(-r)
        r *= 2
    if radix_bits >= 2:
        for a, b in ((B.t, gd.N), (gd.tH, gd.N), (B.t, 512)):
            for grp in ladder_chunks(a, b, radix_bits):
                if len(grp) > 1:
                    steps.update(grp)
    return steps
