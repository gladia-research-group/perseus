"""Prefill -> decode KV hand-off: the blocked GQA prefill's per-chunk K / V (post-RoPE, real, t' lanes) re-laid into
the decode's ComplexKVCache (cachemir_complex: K groups of t tokens over the expanded heads paired into complex
buckets, V in the d_head / 2 diagonal pair buckets). Masks and rotations only, no refresh (logs/prefill/HANDOFF.md)."""
from __future__ import annotations

import numpy as np

from .attention import ComplexKVCache
from .poly import im_cleanse


def rot_any(ops, x, s: int, keys=None):
    """rotate(x, s) by one loaded key, else two."""
    if s == 0:
        return x
    if keys is None or s in keys:
        return ops.rotate(x, s)
    for a in sorted(keys, key=abs, reverse=True):
        if s - a in keys:
            return ops.rotate(ops.rotate(x, a), s - a)
    raise KeyError(f"no rotation path for step {s}")


def k_pairing(n: int, t: int):
    """(buckets [(even, odd)], pending) group indices after n pushes (attention._pack_k_pair's rule)."""
    pend, buckets = [], []
    for c in range(1, n + 1):
        if (c - 1) % t == 0:
            pend.append((c - 1) // t)
        if len(pend) == 2 and c % t == 0:
            buckets.append(tuple(pend))
            pend = []
    return buckets, pend


def to_decode_cache(rt, K, V, T: int, bos_k, bos_v, t_src: int = 64, hg: int = 4, G: int = 4, keys=None,
                    tag: str = "ho", layer: int = 0, cleanse: bool = False):
    """K[b][c], V[b][c]: KV block b (KV heads hg b .. hg b + hg - 1 at local j hg + m), real chunk c (positions
    1 + t_src c + lane); bos_k / bos_v: position 0's decode-layout group vectors. Decode geometry from rt.dims.
    `cleanse`: keep only the real part of the sources (the pairing puts a group on the imaginary half)."""
    ops, d = rt.ops, rt.dims
    sc = 0.5 if cleanse else 1.0
    N, t, tH, dh = d.N, d.t, d.tH, d.d_head
    n = T + 1
    ng = (n + t - 1) // t
    sub = np.arange(N) % tH
    lane, m_of = sub % t_src, sub // t_src

    def groups(S, bos, kind):
        out = [None] * ng
        for b, chunks in enumerate(S):
            for c, x in enumerate(chunks):
                if x is None:
                    continue
                if cleanse:
                    x = im_cleanse(ops, x)
                last = min(t_src, T - t_src * c)
                for g in range(ng):
                    lo, hi = max(t * g - 1 - t_src * c, 0), min(t * g + t - 1 - t_src * c, last)
                    if lo >= hi:
                        continue
                    for m in range(hg):
                        mk = (tag + ".grp", lo, hi, m, sc)
                        y = rt.mult_mask(x, mk, lambda lo=lo, hi=hi, m=m:
                                         sc * ((m_of == m) & (lane >= lo) & (lane < hi)).astype(float))
                        s = t * g - 1 - t_src * c + m * (t_src - G * t) - b * hg * G * t
                        y = rot_any(ops, y, s, keys)
                        out[g] = y if out[g] is None else ops.add(out[g], y)
        for g in range(ng):
            r = t
            while r < G * t:
                out[g] = ops.add(out[g], rot_any(ops, out[g], -r, keys))
                r *= 2
        out[0] = rt.add_mask(out[0], (tag + ".bos", kind, layer), lambda: np.asarray(bos, dtype=np.float64))
        return out

    kv = ComplexKVCache(dh)
    GK = groups(K, bos_k, "k")
    bk, pend = k_pairing(n, t)
    kv.k_buckets = [ops.add(GK[a], ops.mult_i(GK[b])) for a, b in bk]
    kv.k_pend = [GK[a] for a in pend]
    GV = groups(V, bos_v, "v")
    blk = np.arange(N) // tH
    C = {}
    for g in range(-1, ng):
        re = GV[g] if g >= 0 else None
        im = ops.mult_i(GV[g + 1]) if g + 1 < ng else None
        C[g] = re if im is None else (im if re is None else ops.add(re, im))
    for p in range(dh // 2):
        acc = None
        for g, cg in C.items():
            j = (g - 2 * p) % dh
            y = rt.mult_mask(cg, (tag + ".vdiag", j), lambda j=j: (blk == j).astype(float))
            acc = y if acc is None else ops.add(acc, y)
        kv.v_buckets[p] = acc
    kv.k_count = kv.v_count = n
    return kv
