"""Decode attention from primitives: the K/V cache pushes
(src/packing/cachemir/cachemir_kv_cache.cu), q.K^T, the THOR softmax with its head-wise
reductions and P.V (src/algorithms/attention/cachemir/cachemir_attention.cu), and the MHA
sublayer order with its bootstrap hints (src/model/mha.cu)."""
from __future__ import annotations

import contextlib
import math
import os

import numpy as np

from .config import SoftmaxCfg
from .layout import Dims
from .layout import (active_mask, complex_vlane_mask, hrs_pos0, qkt_complex_odd_mask,
                     qkt_group_mask, real_head_half_mask, real_head_tok0_mask, score_mask,
                     stride_mask, vlane_mask, vpair_mask_complex)
from .linear import linear, linear_multi
from . import smcheb
from .poly import (eval_chebyshev, goldschmidt_inv_ndf, goldschmidt_recip, im_cleanse, rotsum,
                   rotsum_neg)


def sm_periodic(rt, kc: int) -> int:
    """The scores' period in tH blocks, Gp (0: not periodic). SM_PERIODIC=1 while every cached token fits one group
    (kc <= t): q.K^T keeps the score of (h, tok) in every tH block instead of block 0 only, so the THOR softmax and
    its head sums run on tH-periodic ciphertexts (their refreshes route sparse, the head sum needs no lane-copy
    ladder) and P.V is one product with the summed V. SM_PERIODIC=2 carries this past one group: group g's scores sit
    in every block b with b % Gp == g, Gp the power of two >= the group count."""
    ops, d = rt.ops, rt.dims
    if not getattr(ops, "sm_periodic", False) or getattr(ops, "fused_sm_den", False):
        return 0                                     # FUSED_SM_DEN=1 keeps its block-0 ladder + fold
    if kc <= d.t:
        return 1
    if getattr(ops, "sm_periodic_general", False):
        gp = 1 << ((kc + d.t - 1) // d.t - 1).bit_length()
        return gp if gp * d.tH <= d.N else 0
    return 0


def _general(rt) -> bool:
    return bool(getattr(rt.ops, "sm_periodic_general", False))


def _gp_scope(rt, per: int):
    """SM_PERIODIC=2: the ops whose count depends on Gp run in a sub-step keyed by it, so every name a plan pins is
    the same at every cache length (one token-0 plan binds throughout; plan with FORBID_STEPS .gp)."""
    return rt.step("gp", shape=f"p{int(per)}")


def _gp_out(rt, x):
    """The sub-step's result named in the enclosing step (an identity add)."""
    return rt.ops.add(x, 0.0)


def sm_fold_affine(rt, cfg: SoftmaxCfg):
    """(alpha, beta) of the exp's Chebyshev input map y = alpha x - beta when SM_FOLD=1 folds it into the q.K^T
    group masks (alpha) and the score mask (alpha mask - beta); (1, 0) otherwise."""
    if not getattr(rt.ops, "sm_fold", False):
        return 1.0, 0.0
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    a, b = cfg.cheb_a / sf, cfg.cheb_b / sf
    return 2.0 / (b - a), (a + b) / (b - a)


def _group_mask_item(d, num_tok: int, g: int, per: int, scale: float = 1.0):
    if per:
        key = ("qkt.gmask.p", num_tok, g) + ((int(per),) if per > 1 else ())
        build = lambda: qkt_group_mask(d, num_tok, g, periodic=per)
    else:
        key, build = ("qkt.gmask", num_tok, g), lambda: qkt_group_mask(d, num_tok, g)
    if scale == 1.0:
        return key, build
    return key + ("x", scale), lambda: scale * build()


def _odd_mask_item(d, num_tok: int, g: int, scale: float = 1.0, per: int = 0):
    key = ("qkt.gmask.c", num_tok, g) + ((int(per),) if per else ())
    build = lambda: qkt_complex_odd_mask(d, num_tok, g, per)
    if scale == 1.0:
        return key, build
    return key + ("x", scale), lambda: scale * build()


def _fold_mask_items(rt, cfg: SoftmaxCfg, kc: int):
    """The per-step masks SM_FOLD adds: one seed-scaled head-sum mask per reciprocal round."""
    if not (getattr(rt.ops, "sm_fold", False) and getattr(rt.ops, "sm_den_recip", False)
            and not getattr(rt.ops, "fused_sm_den", False)):
        return []
    item = _hrs_pack_item if getattr(rt.ops, "hrs_pack", False) else _hrs_seed_item
    return [item(rt.dims, 0.5 * beta * scale * scale, _seed_site(cfg, k))
            for k, (_a, beta, _i, scale, _c) in enumerate(_recip_rounds_ops(rt.ops, cfg, kc))]


def _softmax_mask_items(d, cfg: SoftmaxCfg, kc: int, per: int, affine=(1.0, 0.0)):
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    sfx = (".p" if per == 1 else f".p{int(per)}") if per else ""
    al, be = affine
    if (al, be) == (1.0, 0.0):
        score = (("sm.score" + sfx, cfg.clip_lo, mean, kc), lambda: score_mask(d, cfg.clip_lo, mean, kc, periodic=per))
    else:
        score = (("sm.score" + sfx, cfg.clip_lo, mean, kc, "affine", al, be),
                 lambda: al * score_mask(d, cfg.clip_lo, mean, kc, periodic=per) - be)
    return [score, (("sm.active" + sfx, kc), lambda: active_mask(d, kc, periodic=per))]


def _recip_rounds(cfg: SoftmaxCfg, kc: int, lean: bool):
    """(alpha, beta, iters, scale) of each reciprocal round of _softmax_recip (`lean`: SM_GS_FIRST's first count)."""
    c = 0.5 * math.sqrt(kc) * 0.25
    rounds = [(cfg.init_alpha, cfg.init_beta, first_iters(cfg, lean), 1.0)]
    for i in range(cfg.log2delta2):
        r = cfg.kc_r(i, kc)
        rounds.append((cfg.refine_alpha[i] * math.sqrt(r), cfg.refine_beta[i] * r,
                       int(cfg.per_step_refine_iters[i]), c))
    return rounds


def _recip_rounds_ops(ops, cfg: SoftmaxCfg, kc: int):
    """(alpha, beta, iters, scale, cheb) of each reciprocal round under the session's switches: SM_CHEB_GS=1 takes the
    Chebyshev-scaled rounds of impl smcheb (cheb = (kappas, mu)), else _recip_rounds with cheb None."""
    if getattr(ops, "sm_cheb_gs", False):
        return smcheb.rounds(cfg, kc)
    return [r + (None,) for r in _recip_rounds(cfg, kc, getattr(ops, "sm_gs_first", False))]


def _seed_site(cfg: SoftmaxCfg, rnd: int) -> str:
    """The staging site of a reciprocal round's seed mask: one per (round, layer config), so a per-token seed is
    encoded only at the level its own block's round runs at (a shared site accumulates every block's level)."""
    tag = abs(hash((cfg.init_alpha, cfg.init_beta, tuple(cfg.refine_alpha), tuple(cfg.refine_beta)))) % 10 ** 8
    return f"hrs.seed{rnd}.{tag}"


def _hrs_seed_item(d, seed: float, site: str):
    return (site, seed), lambda: seed * hrs_pos0(d)


class Rope:
    """Rotary position embedding (RoPE, the rotate-half convention of HF Llama) on the
    head-interleaved cachemir layout. Feature r = lane * H + h sits at slot r * t, so lane j of
    every head lives in the slot block [j tH, (j + 1) tH) and its rotary partner, lane j +- d_head/2,
    is exactly N/2 slots away: rope(x) = x cos + rotate(x, N/2) sin', with sin' = -sin on the
    first half of the lanes and +sin on the second. The cos / sin' vectors fold into the selection
    mask the site multiplies by anyway (the q tok0 mask, the K push mask), so RoPE costs one
    rotation and one plaintext product, no level. `inv_freq`: the d_head/2 frequencies (HF's
    rotary_emb.inv_freq, scaling included)."""

    def __init__(self, inv_freq):
        self.inv_freq = tuple(float(f) for f in inv_freq)

    def lane_cs(self, d, pos: int):
        if d.d_head_real != d.d_head or 2 * len(self.inv_freq) != d.d_head:
            raise ValueError(f"RoPE needs unpadded heads and d_head/2 frequencies "
                             f"(d_head={d.d_head}, real {d.d_head_real}, {len(self.inv_freq)} freqs)")
        ang = pos * np.asarray(self.inv_freq)
        c, sn = np.cos(ang), np.sin(ang)
        return np.concatenate([c, c]), np.concatenate([-sn, sn])

    def slot_vecs(self, d, pos: int):
        """(cos, sin') per slot: lane j = slot // tH."""
        c, sn = self.lane_cs(d, pos)
        return np.repeat(c, d.tH), np.repeat(sn, d.tH)


def _rope_items(rt, rope, pos, site, base_build):
    d = rt.dims
    def build(k):
        return lambda: rope.slot_vecs(d, pos)[k] * base_build()
    return [((site + ".rope.c", pos), build(0)), ((site + ".rope.s", pos), build(1))]


def rope_mask(rt, x, site, base_build, rope=None, pos=None):
    """rt.mult_mask(x, site, base_build), with RoPE at position `pos` folded in when `rope` is
    given: x (base cos) + rotate(x, N/2) (base sin')."""
    if rope is None:
        return rt.mult_mask(x, site, base_build)
    (kc, bc), (ks, bs) = _rope_items(rt, rope, pos, site, base_build)
    a = rt.mult_mask(x, kc, bc)
    b = rt.mult_mask(rt.ops.rotate(x, rt.dims.N // 2), ks, bs)
    return rt.ops.add(a, b)


def rope_step_masks(rt, rope, pos: int, packing: str = "cachemir"):
    """The per-step RoPE masks of position `pos`: the q.K^T query mask and the K push mask of
    `packing` ("cachemir" on a complex payload pushes K + iV; "cachemir_complex"; "real")."""
    d = rt.dims
    if packing == "cachemir_complex" and getattr(rt.ops, "kv_lanes", False):
        packing = "cachemir_complex.lanes"
    push = {"cachemir_complex": ("kpush.tok0h", lambda: real_head_half_mask(d)),
            "cachemir_complex.lanes": ("kpush.tok0", lambda: real_head_tok0_mask(d)),
            "cachemir": ("kpush.tok0h.c", lambda: 0.5 * real_head_half_mask(d)),
            "real": ("kpush.tok0h", lambda: real_head_half_mask(d))}[packing]
    return (_rope_items(rt, rope, pos, "tok0", lambda: real_head_tok0_mask(d))
            + _rope_items(rt, rope, pos, *push))


def _hrs_pack_item(d, seed: float, site: str, scale: float = 1.0):
    """SM_HRS_PACK: both head-sum masks on one ciphertext, the sum on the real part and seed * the sum on the
    imaginary part, halved so that z + conj(z) and i (conj(z) - z) return them unscaled. `scale`: the mask times it
    (0.5 on SM_HRS_CLEANSE's 2 Re input)."""
    key = (site + ".pk", seed) if scale == 1.0 else (site + ".pk", seed, scale)
    return key, lambda: scale * (0.5 + 0.5j * seed) * hrs_pos0(d)


class KVCache:
    """One block's caches: K accumulated t tokens per group ciphertext, V scattered over
    d_head lane ciphertexts (cachemir_kv_cache.cu)."""

    def __init__(self, d_head: int):
        self.k_groups = []
        self.v_lanes = [None] * d_head
        self.k_count = 0
        self.v_count = 0

    def reset(self):
        self.k_groups.clear()
        self.v_lanes = [None] * len(self.v_lanes)
        self.k_count = self.v_count = 0

    def cts(self, prefix: str):
        """(ciphertexts, slot keys) of everything the cache holds: the residency ring parks
        them in the pinned KV arena between blocks (driver.py)."""
        cts, keys = [], []
        for i, g in enumerate(self.k_groups):
            cts.append(g); keys.append(f"{prefix}k.{i}")
        for i, v in enumerate(self.v_lanes):
            if v is not None:
                cts.append(v); keys.append(f"{prefix}v.{i}")
        return cts, keys


class ComplexKVCache:
    """The cachemir_complex caches (cachemir_complex_kv_cache.cu): K groups paired into
    complex buckets (group 2c on the real axis, 2c+1 on the imaginary axis; up to two real
    groups pending), V lanes paired into d_head/2 buckets (lane 2p real, 2p+1 imaginary)."""

    def __init__(self, d_head: int):
        self.k_buckets = []
        self.k_pend = []
        self.v_buckets = [None] * (d_head // 2)
        self.k_count = 0
        self.v_count = 0

    def reset(self):
        self.k_buckets.clear(); self.k_pend.clear()
        self.v_buckets = [None] * len(self.v_buckets)
        self.k_count = self.v_count = 0

    def cts(self, prefix: str):
        cts, keys = [], []
        for i, g in enumerate(self.k_buckets):
            cts.append(g); keys.append(f"{prefix}kb.{i}")
        for i, g in enumerate(self.k_pend):
            cts.append(g); keys.append(f"{prefix}kp.{i}")
        for i, v in enumerate(self.v_buckets):
            if v is not None:
                cts.append(v); keys.append(f"{prefix}vb.{i}")
        return cts, keys


def _vpair_item(d, i_re: int, i_im: int, rr: int, kv_lanes: bool):
    """The V pair selector of a push: on 2 V_raw (scale 0.5), or on V_raw itself under KV_LANES (scale 1)."""
    if kv_lanes:
        return ("v.cpc.l", i_re, i_im, rr), lambda: vpair_mask_complex(d, i_re, i_im, rr, 1.0)
    return ("v.cpc", i_re, i_im, rr), lambda: vpair_mask_complex(d, i_re, i_im, rr)


def _kv_bts_scope():
    """IMPL_STC_FIRST_KV=0: the K/V push refresh takes the shipped order under FIDESLIB_BTS_STC_FIRST (its payload is
    small in the slots but tiny in the coefficients, where the shipped order is ~4 bits more precise)."""
    if os.environ.get("IMPL_STC_FIRST_KV", "1") not in ("", "0"):
        return contextlib.nullcontext()
    from perseus.impl import bootstrap as _pyb
    return _pyb.stc_first(False)


def cache_kv_push_packed_complex(rt, kv: ComplexKVCache, P_in, q=None, rope=None, pos=None):
    """cache_kv_push_packed_complex: the fused K + iV linear output refreshed by ONE
    bootstrap, split by conjugation (2K on the real axis, -2iV on the imaginary one), K
    masked and bucketed (odd groups packed by the monomial i), V realified (times i, the
    monomial: no level) and scattered into the pair buckets.
    `q` (KV_LANES + QKV_LANES): the Q linear output rides the same real payload on token lane
    2 and comes back refreshed, on lane 0 (the one refresh serves Q, K and V)."""
    ops, d = rt.ops, rt.dims
    kvl = getattr(ops, "kv_lanes", False)
    q_out = None
    if kvl:
        # KV_LANES: realify first (both halves fill every token lane), keep K on lane 0 and V on lane 1, and refresh
        # that real payload by the real route; the K mask and the V pair masks below select the two lanes
        conj = ops.conjugate(P_in)
        Kc = rt.mult_mask(ops.add(P_in, conj), "kv.lane0", lambda: stride_mask(d.N, d.hid, d.t, 0.5))   # K_raw
        Vc = rt.mult_mask(ops.mult_i(ops.sub(conj, P_in)), "kv.lane0", lambda: stride_mask(d.N, d.hid, d.t, 0.5))
        X = ops.add(Kc, ops.rotate(Vc, -1))
        qs = None if q is None else (q if isinstance(q, list) else [q])
        if qs is not None:
            # QKV_LANES: Q (real; 2 Re by its conjugate, the lane mask's 0.5) on lane 2 of the same payload. The
            # three masks sit at one level, so the refresh input is the linear output plus one unit, as without Q;
            # a list (GQA, pre-rotated per query head) rides lanes 2, 3, ...
            for i, qi in enumerate(qs):
                Qc = rt.mult_mask(ops.add(qi, ops.conjugate(qi)), "kv.lane0",
                                  lambda: stride_mask(d.N, d.hid, d.t, 0.5))
                X = ops.add(X, ops.rotate(Qc, -2 - i))
        with _kv_bts_scope():
            ops.bootstrap_real(X)
        if qs is not None:
            # lane 2 (+ i) back to lane 0; complex_qkt's tok0 mask drops the K / V lanes
            q_out = [ops.rotate(X, 2 + i) for i in range(len(qs))]
            if not isinstance(q, list):
                q_out = q_out[0]
        K = rope_mask(rt, X, "kpush.tok0", lambda: real_head_tok0_mask(d), rope, pos)
    else:
        P = ops.copy(P_in)
        with _kv_bts_scope():
            ops.bootstrap(P)
        conj = ops.conjugate(P)
        K = ops.add(P, conj)                               # 2 K_raw
        V = ops.sub(conj, P)                               # -2i V_raw, realified by the monomial i
        K = rope_mask(rt, K, "kpush.tok0h", lambda: real_head_half_mask(d), rope, pos)
    K = ops.rotate(K, -(kv.k_count % d.t))
    # bucket_k_complex
    if kv.k_count % d.t == 0:
        kv.k_pend.append(ops.copy(K))
    else:
        ops.inplace_add(kv.k_pend[-1], K)
    kv.k_count += 1
    # with step-scoped names the pair pack (2 ops, every 2t tokens) runs last, so it cannot move
    # the names of the V half below
    late_pack = getattr(rt, "scoped_names_requested", False)
    if not late_pack:
        _pack_k_pair(ops, kv, d)
    rr = kv.v_count % d.t
    if kvl:
        v = ops.rotate(X, 1 - rr) if rr != 1 else ops.copy(X)   # V_raw, lane 1 -> lane rr
    else:
        v = ops.rotate(V, -rr) if rr else ops.copy(V)
        v = ops.mult_i(v)                                  # 2 V_raw, no level
    # bucket_v_complex
    c = kv.v_count // d.t
    keys, builds, lanes = [], [], []
    for p in range(d.d_head_real // 2):
        i_re = ((c - 2 * p) % d.d_head + d.d_head) % d.d_head
        i_im = ((c - 2 * p - 1) % d.d_head + d.d_head) % d.d_head
        key, build = _vpair_item(d, i_re, i_im, rr, kvl)
        keys.append(key); builds.append(build)
    tmps = rt.mult_masks(v, keys, builds)
    for p, tmp in enumerate(tmps):
        if kv.v_buckets[p] is None:
            kv.v_buckets[p] = ops.copy(tmp)
        else:
            ops.inplace_add(kv.v_buckets[p], tmp)
    kv.v_count += 1
    if late_pack:
        _pack_k_pair(ops, kv, d)
    return q_out


def _pack_k_pair(ops, kv: ComplexKVCache, d):
    """Pack the two pending K groups into one complex bucket (even + i odd, the monomial) once
    both are complete."""
    if len(kv.k_pend) == 2 and kv.k_count % d.t == 0:
        kv.k_buckets.append(ops.add(kv.k_pend[0], ops.mult_i(kv.k_pend[1])))
        kv.k_pend.clear()


def complex_qkt(rt, kv: ComplexKVCache, query, scale: float = 1.0, rope=None, pos=None, merge=None):
    """complex_qkt (cachemir_complex_attention.cu): one product per K bucket serves two
    token groups; the even group's scores come out of 2 Re, the odd group's out of the
    imaginary axis through the odd mask; pending real groups run the real q.K^T."""
    ops, d = rt.ops, rt.dims
    q = rope_mask(rt, query, "tok0", lambda: real_head_tok0_mask(d), rope, pos)
    q = rotsum_neg(ops, q, d.t)
    kc = kv.k_count
    attn = None
    def accum(r):
        nonlocal attn
        if attn is None:
            attn = r
        else:
            ops.inplace_add(attn, r)
    per = sm_periodic(rt, kc)
    # merge = (G, g): head g's group c goes to blocks b % (G Gp) == g Gp + c, so G heads share one ciphertext
    per_m, g0 = (merge[0] * per, merge[1] * per) if merge else (per, 0)
    for gc, bucket in enumerate(kv.k_buckets):
        g_even, g_odd = 2 * gc, 2 * gc + 1
        result = ops.mult(q, bucket)
        result = rotsum(ops, result, d.tH, d.N)
        if per:
            result = ops.tag_reduce(result, d.tH)
        conj = ops.conjugate(result)
        res_e = ops.add(result, conj)
        n_even = min(d.t, kc - g_even * d.t)
        accum(rt.mult_mask(res_e, *_group_mask_item(d, n_even, g0 + g_even, per_m, scale)))
        if g_odd * d.t < kc:
            res_o = ops.sub(result, conj)
            n_odd = min(d.t, kc - g_odd * d.t)
            accum(rt.mult_mask(res_o, *_odd_mask_item(d, n_odd, g0 + g_odd, scale, per_m)))
    for j, pend in enumerate(kv.k_pend):
        g = 2 * len(kv.k_buckets) + j
        result = ops.mult(q, pend)
        result = rotsum(ops, result, d.tH, d.N)
        if per:
            result = ops.tag_reduce(result, d.tH)             # the ladder's output is tH-periodic
        n = min(d.t, kc - g * d.t)
        result = rt.mult_mask(result, *_group_mask_item(d, n, g0 + g, per_m, scale))
        accum(im_cleanse(ops, result))
    return attn


def complex_softmax_v(rt, kv: ComplexKVCache, probs):
    """complex_softmax_v: pair the scores of lanes 2j and 2j+1 as s_2j - i s_2j+1, so one
    product with the V pair bucket sums both lanes on the real axis; hoisted rotations feed
    the pair products, then the token ladder, 2 Re and the tok0 half mask."""
    ops, d = rt.ops, rt.dims
    n_pairs = d.d_head_real // 2
    per = sm_periodic(rt, kv.k_count)
    if per and _general(rt):
        # SM_PERIODIC=2 (every Gp, 1 included): pair j's rotation 2j tH only matters mod Gp, so the buckets of each
        # class are summed and each class takes one product
        scores_b = ops.copy(probs)
        rot1 = ops.rotate(scores_b, d.tH)
        conj_S_all = ops.sub(scores_b, ops.mult_i(rot1))   # s_2j - i s_2j+1 (monomial)
        with _gp_scope(rt, per):
            classes = {}
            for j in range(n_pairs):
                r = (2 * j) % per
                if r in classes:
                    ops.inplace_add(classes[r], kv.v_buckets[j])
                else:
                    classes[r] = ops.copy(kv.v_buckets[j])
            rs = sorted(classes)
        res = ops.mult(classes[rs[0]], conj_S_all)
        with _gp_scope(rt, per):
            if len(rs) > 1:
                ops.mult_add_many(res, [classes[r] for r in rs[1:]],
                                  ops.rotate_many(conj_S_all, [r * d.tH for r in rs[1:]]))
        res = _gp_out(rt, res)
        res = rotsum(ops, res, 1, d.t)
        res = im_cleanse(ops, res)
        return rt.mult_mask(res, "tok0.h", lambda: real_head_half_mask(d))
    if per == 1:
        # tH-periodic scores: every pair's rotated score vector is (1 - i) P, so the pair products collapse into
        # one with the summed buckets; Re((V_re + i V_im)(1 - i) P) puts V P on both lanes of every pair
        vs = ops.copy(kv.v_buckets[0])
        for b in kv.v_buckets[1:n_pairs]:
            ops.inplace_add(vs, b)
        res = ops.mult(vs, ops.sub(probs, ops.mult_i(probs)))
        res = rotsum(ops, res, 1, d.t)
        res = im_cleanse(ops, res)
        return rt.mult_mask(res, "tok0.h", lambda: real_head_half_mask(d))
    scores_b = ops.copy(probs)
    rot1 = ops.rotate(scores_b, d.tH)
    conj_S_all = ops.sub(scores_b, ops.mult_i(rot1))       # s_2j - i s_2j+1 (monomial)
    pair_scores = ops.rotate_many(conj_S_all, [2 * j * d.tH for j in range(1, n_pairs)])
    res = ops.mult(kv.v_buckets[0], conj_S_all)
    if n_pairs > 1:
        ops.mult_add_many(res, kv.v_buckets[1:n_pairs], pair_scores)
    res = rotsum(ops, res, 1, d.t)
    res = im_cleanse(ops, res)
    return rt.mult_mask(res, "tok0.h", lambda: real_head_half_mask(d))


def cache_k_push(rt, kv: KVCache, key, rope=None, pos=None):
    """cachemir_kv_cache.cu: mask, rotate, bootstrap, im_cleanse, accumulate."""
    ops, d = rt.ops, rt.dims
    m = rope_mask(rt, key, "kpush.tok0h", lambda: real_head_half_mask(d), rope, pos)
    # the C++ rotates unconditionally (a rotation by 0 at the first token of a group), which
    # keeps the op sequence, hence the graph variable names a plan pins, identical per token
    r = ops.rotate(m, -(kv.k_count % d.t))
    ops.bootstrap(r)
    r = im_cleanse(ops, r)
    if kv.k_count % d.t == 0:
        kv.k_groups.append(ops.copy(r))      # the C++ clones: one recorded op either way,
    else:                                    # so token 0 and later tokens share a graph shape
        ops.inplace_add(kv.k_groups[-1], r)
    kv.k_count += 1


def cache_v_push(rt, kv: KVCache, value):
    """cachemir_kv_cache.cu."""
    ops, d = rt.ops, rt.dims
    rr = kv.v_count % d.t
    v = ops.rotate(value, -rr) if rr else ops.copy(value)
    ops.bootstrap(v)
    v = im_cleanse(ops, v)
    keys = [("v.lane", i, rr) for i in range(d.d_head_real)]
    tmps = rt.mult_masks(v, keys, [lambda i=i: vlane_mask(d, i, rr, 0.5) for i in range(d.d_head_real)])
    for i, tmp in enumerate(tmps):
        lane = ((kv.v_count // d.t) - i + d.d_head) % d.d_head
        if kv.v_lanes[lane] is None:
            kv.v_lanes[lane] = ops.copy(tmp)  # clone on first write (cachemir_kv_cache.cu)
        else:
            ops.inplace_add(kv.v_lanes[lane], tmp)
    kv.v_count += 1


def cache_kv_push_pair(rt, kv: KVCache, key, value, rope=None, pos=None):
    """cachemir_kv_cache.cu, the complex-payload arm (CKKS_COMPLEX=1, "Mode-A"): K
    and V are packed as K + iV, refreshed by ONE bootstrap and split by conjugation. The K
    mask carries an extra 0.5 (its 2 Re doubling happens before the pack), the V lanes are
    selected on the imaginary axis (2i V times -0.5 i = V)."""
    ops, d = rt.ops, rt.dims
    m = rope_mask(rt, key, "kpush.tok0h.c", lambda: 0.5 * real_head_half_mask(d), rope, pos)
    r = ops.rotate(m, -(kv.k_count % d.t))
    rr = kv.v_count % d.t
    v = ops.rotate(value, -rr) if rr else ops.copy(value)
    r = im_cleanse(ops, r)
    v = im_cleanse(ops, v)
    P = ops.add(r, ops.mult_i(v))                      # K + i V (monomial)
    ops.bootstrap(P)
    conj = ops.conjugate(P)
    r = ops.add(P, conj)                               # 2 Re(P) = K   (the mask was x0.5)
    v = ops.sub(P, conj)                               # 2i Im(P) = 2i V
    if kv.k_count % d.t == 0:
        kv.k_groups.append(ops.copy(r))
    else:
        ops.inplace_add(kv.k_groups[-1], r)
    kv.k_count += 1
    keys = [("v.lane.c", i, rr) for i in range(d.d_head_real)]
    tmps = rt.mult_masks(v, keys, [lambda i=i: complex_vlane_mask(d, i, rr) for i in range(d.d_head_real)])
    for i, tmp in enumerate(tmps):
        lane = ((kv.v_count // d.t) - i + d.d_head) % d.d_head
        if kv.v_lanes[lane] is None:
            kv.v_lanes[lane] = ops.copy(tmp)
        else:
            ops.inplace_add(kv.v_lanes[lane], tmp)
    kv.v_count += 1


def qkt(rt, kv: KVCache, query, scale: float = 1.0, rope=None, pos=None):
    """cachemir_attention.cu: scores at slot g*tH + h*t + tok%t."""
    ops, d = rt.ops, rt.dims
    q = rope_mask(rt, query, "tok0", lambda: real_head_tok0_mask(d), rope, pos)
    q = rotsum_neg(ops, q, d.t)
    attn = None
    per = sm_periodic(rt, kv.k_count)
    for g, kg in enumerate(kv.k_groups):
        num_tok = min(d.t, kv.k_count - g * d.t)
        res = ops.mult(q, kg)
        res = rotsum(ops, res, d.tH, d.N)
        if per:
            res = ops.tag_reduce(res, d.tH)
        res = rt.mult_mask(res, *_group_mask_item(d, num_tok, g, per, scale))
        res = im_cleanse(ops, res)
        if attn is None:
            attn = res
        else:
            ops.inplace_add(attn, res)
    return attn


def fold_sm_prescale(copies: float, s0_expected: float) -> float:
    """nonlinear.h fold_sm_prescale_for: land the folded denominator near 0.3 of the
    bootstrap's range; never amplifies."""
    if not (s0_expected > 0.0) or not math.isfinite(s0_expected) or copies <= 0.0:
        return 1.0
    return max(1e-6, min(1.0, 0.3 * copies / s0_expected))


def head_reduce_sum(rt, x, s0_expected: float = 0.0, periodic: bool = False, seed: float | None = None,
                    seed_site: str = "hrs.seed"):
    """cachemir_attention.cu: per-head total broadcast over the head's t slots and
    over the lane copies. With FUSED_SM_DEN the lane-copy ladder stops at the sparse slot
    count and a fold bootstrap finishes it (the denominator comes out refreshed);
    `s0_expected` sizes the fold's prescale. `periodic`: x is already tH-periodic (SM_PERIODIC), every lane copy
    holds the sum after the head ladder. `seed`: also return seed * the sum, from a second position mask on the same
    head ladder (SM_FOLD: the reciprocal's seed slope rides the mask instead of a ciphertext x constant product).
    SM_PERIODIC=2 past one group: the Gp groups' partial sums are added first (the rotations commute with the
    tH-periodic mask and the replicate), so the head ladder's named ops, and the refreshes a token-0 plan pins on
    them, hold the full tH-periodic sum at every cache length. Added after the ladder, those refreshes landed on the
    partial sums, which repeat only every Gp tH slots: the sparse refresh returned them wrong (Llama past 256 keys)."""
    ops, d = rt.ops, rt.dims
    if periodic and _general(rt):
        with _gp_scope(rt, periodic):
            if periodic > 1:
                x = ops.tag_reduce(rotsum(ops, x, d.tH, int(periodic) * d.tH), d.tH)
    out = rotsum(ops, x, 1, d.t)
    if seed is not None and getattr(ops, "hrs_pack", False) and ops.complex_payload:
        if ops.fused_sm_den:
            raise ValueError("SM_FOLD=1 does not combine with FUSED_SM_DEN=1")
        # SM_HRS_PACK: one replicate ladder for both sums (the input is real), split by one conjugation
        if getattr(ops, "hrs_cleanse", False):
            # SM_HRS_CLEANSE: the seed lane holds seed * sum, ~1e-5 of the sum on the Llama decode's rounds; the
            # imaginary noise of a refresh planned after the caller's im_cleanse (the exponentials' or a round's
            # square, at the refresh's precision) swamps it (the Goldschmidt seed returns garbage). 2 Re of the input
            # here, the mask halved; the sub-step `hrs` (FORBID_STEPS .hrs) keeps refreshes off the pack until the
            # split (a refresh of either split output is relative to its own value)
            with rt.step("hrs"):
                z = rotsum_neg(ops, rt.mult_mask(im_cleanse(ops, out), *_hrs_pack_item(d, seed, seed_site, 0.5)), d.t)
                cz = ops.conjugate(z)
            h = ops.mult_i(ops.sub(cz, z))
            out = ops.add(z, cz)
        else:
            z = rotsum_neg(ops, rt.mult_mask(out, *_hrs_pack_item(d, seed, seed_site)), d.t)
            cz = ops.conjugate(z)
            h = ops.mult_i(ops.sub(cz, z))
            out = ops.add(z, cz)
    else:
        if seed is not None:
            tot = out
        out = rt.mult_mask(out, "hrs.pos0", lambda: hrs_pos0(d))
        out = rotsum_neg(ops, out, d.t)
        if seed is not None:
            if ops.fused_sm_den:
                raise ValueError("SM_FOLD=1 does not combine with FUSED_SM_DEN=1")
            h = rotsum_neg(ops, rt.mult_mask(tot, *_hrs_seed_item(d, seed, seed_site)), d.t)
    if seed is not None:
        if not periodic:
            h = rotsum(ops, h, d.tH, d.N)
        elif _general(rt):
            h = _gp_out(rt, h)
        h = ops.tag_reduce(h, getattr(rt, "gq_merge", 1) * d.tH)
    if periodic:
        if ops.fused_sm_den:
            raise ValueError("SM_PERIODIC=1 does not combine with FUSED_SM_DEN=1")
        if _general(rt):     # the groups' partial sums were added before the head ladder
            out = _gp_out(rt, out)
    elif ops.fused_sm_den:
        s_eff = ops.fold_slots_for(d.tH)
        if s_eff > d.tH:
            out = rotsum(ops, out, d.tH, s_eff)
        p = fold_sm_prescale(d.N / s_eff, s0_expected)
        ops.fold_bootstrap(out, s_eff, 1, p)
    else:
        out = rotsum(ops, out, d.tH, d.N)
    # tH-periodic (G tH for merged GQA scores)
    out = ops.tag_reduce(out, getattr(rt, "gq_merge", 1) * d.tH)
    return out if seed is None else (out, h)


def _thor_exp(rt, scores, cfg: SoftmaxCfg, kc: int, folded: bool = False):
    """The masked exponentials of softmax_thor, before its divisions."""
    ops, d = rt.ops, rt.dims
    affine = sm_fold_affine(rt, cfg) if folded else (1.0, 0.0)   # `folded`: q.K^T already applied alpha (mha)
    score_item, active_item = _softmax_mask_items(d, cfg, kc, sm_periodic(rt, kc), affine)
    ct = rt.add_mask(scores, *score_item)
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    if not cfg.cheb_coeffs:
        raise ValueError("softmax: cfg.cheb_coeffs is empty (only the Chebyshev exp is ported)")
    if affine != (1.0, 0.0):    # the scores arrive as alpha x - beta (SM_FOLD): the identity map costs no level
        z = eval_chebyshev(ops, ct, cfg.cheb_coeffs, -1.0, 1.0)
    else:
        z = eval_chebyshev(ops, ct, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf)
    for _ in range(cfg.log2delta1):
        z = ops.square(z)
    z = im_cleanse(ops, z)
    return rt.mult_mask(z, *active_item)


def first_iters(cfg: SoftmaxCfg, lean: bool) -> int:
    """The first division's Goldschmidt count: the refine rounds renormalize after it, so
    SM_GS_FIRST runs it at the config's gs_iters_first when there is one."""
    return cfg.gs_iters_first if lean and cfg.gs_iters_first else cfg.gs_iters_scaled


def _softmax_recip(rt, z, cfg: SoftmaxCfg, kc: int):
    """softmax_thor's divisions as z * (1/s) (SM_DEN_RECIP=1): the Goldschmidt iterations run
    on the head sum's own tH-periodic ciphertext (goldschmidt_recip), whose refreshes route
    sparse, and the scores take one product per round instead of one per iteration. The
    refinement scalar c leaves the scores, z / s = (c z) / (c s): it only calibrates the seed.
    Same arithmetic as the default path."""
    ops = rt.ops
    fold = getattr(ops, "sm_fold", False) and not getattr(ops, "fused_sm_den", False)
    for k, (alpha, beta, iters, scale, cheb) in enumerate(_recip_rounds_ops(ops, cfg, kc)):
        if k:
            z = ops.square(im_cleanse(ops, y))
        if fold:
            s, sh = head_reduce_sum(rt, z, s0_expected=2.0 / (alpha * scale), periodic=sm_periodic(rt, kc),
                                    seed=0.5 * beta * scale * scale, seed_site=_seed_site(cfg, k))
            y = ops.mult(z, goldschmidt_recip(ops, s, alpha, beta, iters, scale, Dh=sh, cheb=cheb))
        else:
            s = head_reduce_sum(rt, z, s0_expected=2.0 / (alpha * scale), periodic=sm_periodic(rt, kc))
            y = ops.mult(z, goldschmidt_recip(ops, s, alpha, beta, iters, scale, cheb=cheb))
    return y


def softmax_thor(rt, scores, cfg: SoftmaxCfg, kc: int, folded: bool = False):
    """cachemir_attention.cu; with SM_DEN_RECIP=1 the divisions are _softmax_recip's. `folded`: the scores come from a
    q.K^T scaled by sm_fold_affine's alpha (SM_FOLD), the exp's input map rides the score mask."""
    ops = rt.ops
    z = _thor_exp(rt, scores, cfg, kc, folded)
    if ops.sm_den_recip:
        return _softmax_recip(rt, z, cfg, kc)
    if getattr(ops, "sm_cheb_gs", False):
        raise ValueError("SM_CHEB_GS=1 needs SM_DEN_RECIP=1")
    s = head_reduce_sum(rt, z, s0_expected=2.0 / cfg.init_alpha, periodic=sm_periodic(rt, kc))
    F_init = ops.mult(s, -cfg.init_beta)
    F_init = ops.add(F_init, cfg.init_alpha)
    y = goldschmidt_inv_ndf(ops, z, s, F_init, first_iters(cfg, ops.sm_gs_first))
    for i in range(cfg.log2delta2):
        y = im_cleanse(ops, y)
        z = ops.square(y)
        z = ops.mult(z, 0.5 * math.sqrt(kc) * 0.25)
        r = cfg.kc_r(i, kc)
        s = head_reduce_sum(rt, z, s0_expected=2.0 / (cfg.refine_alpha[i] * math.sqrt(r)),
                            periodic=sm_periodic(rt, kc))
        sa, sb = math.sqrt(r), r
        F_init = ops.mult(s, -cfg.refine_beta[i] * sb)
        F_init = ops.add(F_init, cfg.refine_alpha[i] * sa)
        y = goldschmidt_inv_ndf(ops, z, s, F_init, int(cfg.per_step_refine_iters[i]))
    return y


def softmax_v(rt, kv: KVCache, probs):
    """cachemir_attention.cu: sum_i V_lane_i * rotate(P, i*tH), token reduction,
    im_cleanse and the tok0 half mask."""
    ops, d = rt.ops, rt.dims
    per = sm_periodic(rt, kv.k_count)
    if per and _general(rt):
        # SM_PERIODIC=2 (every Gp): the lanes of each class i mod Gp summed, one product per class
        with _gp_scope(rt, per):
            vs = [None] * per
            for i, lane in enumerate(kv.v_lanes[:d.d_head_real]):
                r = i % per
                if vs[r] is None:
                    vs[r] = ops.copy(lane)
                else:
                    ops.inplace_add(vs[r], lane)
        res = ops.mult(vs[0], probs)
        with _gp_scope(rt, per):
            if per > 1:
                ops.mult_add_many(res, vs[1:], ops.rotate_many(probs, [r * d.tH for r in range(1, per)]))
        res = _gp_out(rt, res)
        res = rotsum(ops, res, 1, d.t)
        res = im_cleanse(ops, res)
        return rt.mult_mask(res, "tok0.h", lambda: real_head_half_mask(d))
    if per:
        # tH-periodic scores: every rotate(P, i tH) is P, so the lane products collapse into one with the summed lanes
        vs = ops.copy(kv.v_lanes[0])
        for lane in kv.v_lanes[1:d.d_head_real]:
            ops.inplace_add(vs, lane)
        res = ops.mult(vs, probs)
        res = rotsum(ops, res, 1, d.t)
        res = im_cleanse(ops, res)
        return rt.mult_mask(res, "tok0.h", lambda: real_head_half_mask(d))
    res = ops.mult(kv.v_lanes[0], probs)
    shifted = ops.rotate_many(probs, [i * d.tH for i in range(1, d.d_head_real)])  # hoisted
    ops.mult_add_many(res, kv.v_lanes[1:d.d_head_real], shifted)                   # one relin
    res = rotsum(ops, res, 1, d.t)
    res = im_cleanse(ops, res)
    return rt.mult_mask(res, "tok0.h", lambda: real_head_half_mask(d))


def kv_push(rt, kv, payload, q=None, rope=None, pos=None):
    """Push one token into the caches: `payload` is the fused K + iV linear output (a ComplexKVCache) or (k, v).
    `q` (QKV_LANES): the Q linear output riding the same refresh; the refreshed Q is returned (else None).
    A cache with a `payload_log` list (the fake's prefix run) also keeps a copy of the payload's slot vectors, to
    replay the pushes on another session (driver.ImplModel.replay_kv_pushes; a replay pushes without Q, which
    leaves the K / V lanes and their levels as they are). `rope` / `pos`: RoPE of K at position `pos`."""
    ops = rt.ops
    log = getattr(kv, "payload_log", None)
    if log is not None:
        import numpy as np
        vec = lambda c: np.array(getattr(c, "vec", c))
        log.append(vec(payload) if isinstance(kv, ComplexKVCache) else tuple(vec(c) for c in payload))
    if isinstance(kv, ComplexKVCache):
        return cache_kv_push_packed_complex(rt, kv, payload, q, rope=rope, pos=pos)
    elif ops.complex_payload:
        cache_kv_push_pair(rt, kv, *payload, rope=rope, pos=pos)   # one bootstrap for K and V
    else:
        cache_k_push(rt, kv, payload[0], rope=rope, pos=pos)
        cache_v_push(rt, kv, payload[1])


def mha(rt, x, w, kv, sm_cfg: SoftmaxCfg, rope: Rope | None = None, pos: int | None = None,
        pre_linear=None, pre_out=None):
    """src/model/mha.cu: qkv -> cache push -> attention core -> out projection.
    ``w`` has .q .k .v .out (EncodedLinear), or .kv (the fused K + iV linear) and .q for
    the cachemir_complex packing, whose ``kv`` is a ComplexKVCache. ``rope``: rotary position
    embedding of q and K at position ``pos`` (folded into their selection masks). ``pre_linear``:
    called on the qkv projection's input after its refresh hint (e.g. a level drop that runs the
    projection at the bottom of the chain), ``pre_out`` on the out projection's."""
    if rope is not None and pos is None:
        raise ValueError("mha: RoPE needs the token position")
    rp = dict(rope=rope, pos=pos)
    ops = rt.ops
    tap = getattr(rt, "tap", None) or (lambda name, ct: None)     # debug hook (the model's taps)
    cplx = isinstance(kv, ComplexKVCache)
    qkvl = cplx and getattr(ops, "kv_lanes", False) and getattr(ops, "qkv_lanes", False)
    with rt.step("qkv"):
        # QKV_LANES: the hint fires only past the session limit; at or under it the linear runs on the LayerNorm
        # output as is and its output, K, V and Q on three token lanes, is what the push refreshes (one unit deeper:
        # the lane masks), at the top of the envelope
        ops.bootstrap_hint(x, ops.headroom(0 if qkvl else 1), True)
        if pre_linear is not None:
            x = pre_linear(x)
        if cplx:
            P, q = linear_multi(rt, x, [w.kv, w.q])
        else:
            k, v, q = linear_multi(rt, x, [w.k, w.v, w.q])
    tap("attn.q", q)
    tap("attn.kv", P if cplx else k)
    with rt.step("kv_push"):
        if qkvl:
            q = kv_push(rt, kv, P, q, **rp)
        else:
            kv_push(rt, kv, P if cplx else (k, v), **rp)
    # the q.K^T issues one product per cache group: its op count is the cache's shape
    shape = (f"{len(kv.k_buckets)}x{len(kv.k_pend)}" if cplx else f"{len(kv.k_groups)}") \
        if getattr(rt, "scoped_names", False) else None
    with rt.step("qkt", shape=shape):
        a = sm_fold_affine(rt, sm_cfg)[0]
        scores = complex_qkt(rt, kv, q, a, **rp) if cplx else qkt(rt, kv, q, a, **rp)
        folded = bool(getattr(ops, "sm_fold", False))
    tap("attn.q_pushed", q)
    tap("attn.scores", scores)
    with rt.step("softmax"):
        probs = softmax_thor(rt, scores, sm_cfg, kv.k_count, folded)
    tap("attn.probs", probs)
    with rt.step("softmax_v"):
        x = complex_softmax_v(rt, kv, probs) if cplx else softmax_v(rt, kv, probs)
    tap("attn.pv", x)
    with rt.step("out_proj"):
        ops.bootstrap_hint(x, ops.headroom(1), True)
        if pre_out is not None:
            x = pre_out(x)
        return linear(rt, x, w.out)


_GQA_MERGE = os.environ.get("LLAMA_GQA_MERGE", "1") not in ("", "0")


def _gqa_psel(d: Dims, G: int, g: int):
    """tH blocks b with b % G == g (query head g of a merged GQA softmax)."""
    return ((np.arange(d.N) // d.tH) % G == g).astype(np.float64)


def gqa_dims(d: Dims, G: int) -> Dims:
    """Un-expanded K / V geometry for G query groups: hid / G channels, G times the token lanes."""
    return Dims(d.N, d.hid // G, d.dim // G, d.H // G, d.H_real // G, d.E, d.E_real)


def mha_gqa(rt, sub, G: int, x, w, kv, sm_cfg: SoftmaxCfg, rope: Rope | None = None, pos: int | None = None,
            pre_linear=None, pre_out=None):
    """mha on an un-expanded (GQA) K / V cache held by the sub-runtime `sub` (gqa_dims)."""
    if rope is not None and pos is None:
        raise ValueError("mha: RoPE needs the token position")
    rp = dict(rope=rope, pos=pos)
    ops, d = rt.ops, rt.dims
    qkvl = getattr(ops, "kv_lanes", False) and getattr(ops, "qkv_lanes", False)
    with rt.step("qkv"):
        ops.bootstrap_hint(x, ops.headroom(0 if qkvl else 1), True)
        if pre_linear is not None:
            x = pre_linear(x)
        P, q = linear_multi(rt, x, [w.kv, w.q])
    qs = [q] + [ops.rotate(q, g * d.t) for g in range(1, G)]
    with rt.step("kv_push"):
        if qkvl:
            qs = kv_push(sub, kv, P, qs, **rp)
        else:
            kv_push(sub, kv, P, **rp)
    a = sm_fold_affine(sub, sm_cfg)[0]
    folded = bool(getattr(ops, "sm_fold", False))
    outs = []
    per = sm_periodic(sub, kv.k_count)
    if _GQA_MERGE and per == 1 and G * sub.dims.tH <= sub.dims.N:
        # one softmax for all G heads (tH block b: head b % G, tags G tH-periodic); P split per head for P.V
        shape = f"{len(kv.k_buckets)}x{len(kv.k_pend)}" if getattr(rt, "scoped_names", False) else None
        with rt.step("qkt", shape=shape):
            scores = None
            for g in range(G):
                s_g = complex_qkt(sub, kv, qs[g], a, merge=(G, g), **rp)
                scores = s_g if scores is None else ops.add(scores, s_g)
            scores = ops.tag_reduce(scores, G * sub.dims.tH)
        sub.gq_merge = G
        try:
            with rt.step("softmax"):
                probs = softmax_thor(sub, scores, sm_cfg, kv.k_count, folded)
        finally:
            sub.gq_merge = 1
        for g in range(G):
            with rt.step(f"g{g}"):
                with rt.step("softmax_v"):
                    d_s = sub.dims
                    p = sub.mult_mask(probs, ("gqa.psel", G, g), lambda g=g: _gqa_psel(d_s, G, g))
                    p = ops.tag_reduce(rotsum(ops, p, d_s.tH, G * d_s.tH), d_s.tH)
                    outs.append(complex_softmax_v(sub, kv, p))
        G_loop = 0
    else:
        G_loop = G
    for g in range(G_loop):
        with rt.step(f"g{g}"):
            shape = f"{len(kv.k_buckets)}x{len(kv.k_pend)}" if getattr(rt, "scoped_names", False) else None
            with rt.step("qkt", shape=shape):
                scores = complex_qkt(sub, kv, qs[g], a, **rp)
            with rt.step("softmax"):
                probs = softmax_thor(sub, scores, sm_cfg, kv.k_count, folded)
            with rt.step("softmax_v"):
                outs.append(complex_softmax_v(sub, kv, probs))
    with rt.step("gather"):
        x = outs[0]
        for g in range(1, G):
            ops.inplace_add(x, ops.rotate(outs[g], -g * d.t))
    with rt.step("out_proj"):
        ops.bootstrap_hint(x, ops.headroom(1), True)
        if pre_out is not None:
            x = pre_out(x)
        return linear(rt, x, w.out)


def complex_attention_step_masks(rt, cfg: SoftmaxCfg, kc: int, pos_in_group: int):
    """The per-step masks of the cachemir_complex attention at cache count `kc`: the softmax
    masks, the even / odd group masks of the complete K buckets plus the pending groups',
    and the V pair masks of the push at `pos_in_group`."""
    d = rt.dims
    per = sm_periodic(rt, kc)
    affine = sm_fold_affine(rt, cfg)
    items = _softmax_mask_items(d, cfg, kc, per, affine) + _fold_mask_items(rt, cfg, kc)
    Nb = kc // (2 * d.t)
    G = (kc + d.t - 1) // d.t
    for gc in range(Nb):
        ge, go = 2 * gc, 2 * gc + 1
        n = min(d.t, kc - ge * d.t)
        items.append(_group_mask_item(d, n, ge, per, affine[0]))
        if go * d.t < kc:
            n = min(d.t, kc - go * d.t)
            items.append(_odd_mask_item(d, n, go, affine[0], per))
    for g in range(2 * Nb, G):
        n = min(d.t, kc - g * d.t)
        items.append(_group_mask_item(d, n, g, per, affine[0]))
    rr = pos_in_group
    c = (kc - 1) // d.t                     # the push's v_count // t
    for p in range(d.d_head_real // 2):
        i_re = ((c - 2 * p) % d.d_head + d.d_head) % d.d_head
        i_im = ((c - 2 * p - 1) % d.d_head + d.d_head) % d.d_head
        items.append(_vpair_item(d, i_re, i_im, rr, getattr(rt.ops, "kv_lanes", False)))
    return items


def attention_step_masks(rt, cfg: SoftmaxCfg, kc: int, pos_in_group: int):
    """The per-step masks the attention of a token at cache count `kc` (after its own push)
    uses: the score / active masks of the softmax, the q.K^T group masks and the V lane masks
    of the push at `pos_in_group` = v_count % t (cachemir_kv_cache.cu / cachemir_attention.cu;
    the C++ step_mask_walk_block)."""
    d = rt.dims
    per = sm_periodic(rt, kc)
    affine = sm_fold_affine(rt, cfg)
    items = _softmax_mask_items(d, cfg, kc, per, affine) + _fold_mask_items(rt, cfg, kc)
    n_groups = (kc + d.t - 1) // d.t
    for g in range(n_groups):
        num_tok = min(d.t, kc - g * d.t)
        items.append(_group_mask_item(d, num_tok, g, per, affine[0]))
    rr = pos_in_group
    if rt.ops.complex_payload:
        for i in range(d.d_head_real):
            items.append((("v.lane.c", i, rr), lambda i=i: complex_vlane_mask(d, i, rr)))
    else:
        for i in range(d.d_head_real):
            items.append((("v.lane", i, rr), lambda i=i: vlane_mask(d, i, rr, 0.5)))
    return items
