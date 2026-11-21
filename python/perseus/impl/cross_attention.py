"""Cross-attention in the decode (an encoder-decoder's): the token's query against a K / V cache the encoder's output
filled once (fill_cache), through mha's attention core (attention.attend).

The encoder hands K and V over in the decode's group layout (slot j tH + h t + lane: head dimension j, head h, the
key's lane in its group of t keys), groups 2c and 2c + 1 on the real and imaginary axes of pair c, which is how a
blocked prefill's PCMM leaves two chunks of t tokens when its output channels are in the decode's head order
(layout.rearrange_qkv_weights). K's pairs are the cache's K buckets as they are; V's go to the d_head / 2 pair buckets
by the diagonal selectors of handoff.to_decode_cache."""
from __future__ import annotations

import numpy as np

from .attention import ComplexKVCache, attend, complex_attention_step_masks
from .linear import linear


def cross_mha(rt, x, w, kv: ComplexKVCache, sm_cfg):
    """x -> the query linear (w.q) -> attend over the filled cache `kv` -> the out projection (w.out)."""
    ops = rt.ops
    with rt.step("q"):
        ops.bootstrap_hint(x, ops.headroom(1), True)
        q = linear(rt, x, w.q)
    tap = getattr(rt, "tap", None)
    if tap:
        tap("xattn.q", q)
    return attend(rt, q, kv, w, sm_cfg)


def fill_cache(rt, K, V, n: int, tag: str = "xkv"):
    """The ComplexKVCache of the n keys whose K / V pairs are K[c], V[c] (c < ceil(ceil(n / t) / 2)). Lanes past key
    n may hold anything: the q.K^T group masks and the softmax's active mask drop them.

    V bucket p holds, at head dimension j, groups g (real axis) and g + 1 (imaginary) for g = j + 2p mod d_head
    (complex_softmax_v's diagonal; g = -1 .. ng - 1, the groups past both ends empty). An even g's pair is V[g / 2]
    itself; an odd g's, (Im V[c], Re V[c + 1]) for c = (g - 1) / 2, comes from the two pairs' halves by one conjugate
    each: i / 2 (R[c + 1] - J[c]) with R = x + conj x, J = x - conj x, the i / 2 riding the selector."""
    ops, d = rt.ops, rt.dims
    t, dh = d.t, d.d_head
    ng = -(-n // t)
    nc = -(-ng // 2)
    if len(K) != nc or len(V) != nc:
        raise ValueError(f"fill_cache: {n} keys need {nc} K / V pairs, got {len(K)} / {len(V)}")
    if dh < ng + 1:
        raise ValueError(f"fill_cache: {ng} groups overlap on the {dh} head dimensions of a V bucket")
    kv = ComplexKVCache(dh)
    kv.k_buckets = list(K)
    kv.k_count = kv.v_count = n
    with rt.step("vpair"):
        conj = [ops.conjugate(v) for v in V]
        R = [ops.add(v, c) for v, c in zip(V, conj)] + [None]
        J = [None] + [ops.sub(v, c) for v, c in zip(V, conj)]        # J[c + 1] = J_c
        terms = [(g, V[g // 2], 1.0) for g in range(0, ng, 2)]          # (group, ciphertext, selector scale)
        for c in range(-1, nc):
            g = 2 * c + 1
            if g >= ng:
                continue
            r, j = R[c + 1], J[c + 1]
            terms.append((g, j, -0.5j) if r is None else (g, r if j is None else ops.sub(r, j), 0.5j))
    blk = np.arange(d.N) // d.tH
    with rt.step("vdiag"):
        for p in range(d.d_head_real // 2):
            keys = [(tag + ".vdiag", (g - 2 * p) % dh, s) for g, _x, s in terms]
            builds = [lambda j=(g - 2 * p) % dh, s=s: s * (blk == j).astype(np.complex128) for g, _x, s in terms]
            kv.v_buckets[p] = rt.mult_mask_sum([x for _g, x, _s in terms], keys, builds)
    return kv


def cross_masks(rt, cfg, kv: ComplexKVCache):
    """The masks of an attention over the filled cache `kv` (fixed: the cache never grows): the softmax's and the
    q.K^T group masks of its K buckets (every group paired, the last pair's odd group possibly short)."""
    return complex_attention_step_masks(rt, cfg, kv.k_count, None, buckets=len(kv.k_buckets))


def cache_cts(kv: ComplexKVCache):
    """A filled cache as a flat ciphertext list (a stage's output): its K buckets, then its V buckets."""
    return list(kv.k_buckets) + list(kv.v_buckets)


def cache_from(cts, n: int, dims):
    """`cache_cts`'s list back as the ComplexKVCache of n keys."""
    nk = -(-(-(-n // dims.t)) // 2)
    nv = dims.d_head // 2
    if len(cts) != nk + nv:
        raise ValueError(f"cache_from: {n} keys take {nk} K and {nv} V buckets, got {len(cts)} ciphertexts")
    kv = ComplexKVCache(dims.d_head)
    kv.k_buckets, kv.v_buckets = list(cts[:nk]), list(cts[nk:])
    kv.k_count = kv.v_count = n
    return kv
