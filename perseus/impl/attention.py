"""Decode attention from primitives: the K/V cache pushes
(src/packing/cachemir/cachemir_kv_cache.cu), q.K^T, the THOR softmax with its head-wise
reductions and P.V (src/algorithms/attention/cachemir/cachemir_attention.cu), and the MHA
sublayer order with its bootstrap hints (src/model/mha.cu)."""
from __future__ import annotations

import math

from .config import SoftmaxCfg
from .layout import (active_mask, complex_vlane_mask, hrs_pos0, qkt_complex_odd_mask,
                     qkt_group_mask, real_head_half_mask, real_head_tok0_mask, score_mask,
                     vlane_mask, vpair_mask_complex)
from .linear import linear, linear_multi
from .poly import (eval_chebyshev, goldschmidt_inv_ndf, goldschmidt_recip, im_cleanse, rotsum,
                   rotsum_neg)


def sm_periodic(rt, kc: int) -> bool:
    """SM_PERIODIC=1 while every cached token fits one group (kc <= t): q.K^T keeps the score of (h, tok) in every
    tH block instead of block 0 only, so the THOR softmax and its head sums run on tH-periodic ciphertexts (their
    refreshes route sparse, the head sum needs no lane-copy ladder) and P.V is one product with the summed V."""
    return (bool(getattr(rt.ops, "sm_periodic", False)) and kc <= rt.dims.t
            and not getattr(rt.ops, "fused_sm_den", False))     # FUSED_SM_DEN=1 keeps its block-0 ladder + fold


def sm_fold_affine(rt, cfg: SoftmaxCfg):
    """(alpha, beta) of the exp's Chebyshev input map y = alpha x - beta when SM_FOLD=1 folds it into the q.K^T
    group masks (alpha) and the score mask (alpha mask - beta); (1, 0) otherwise."""
    if not getattr(rt.ops, "sm_fold", False):
        return 1.0, 0.0
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    a, b = cfg.cheb_a / sf, cfg.cheb_b / sf
    return 2.0 / (b - a), (a + b) / (b - a)


def _group_mask_item(d, num_tok: int, g: int, per: bool, scale: float = 1.0):
    if per:
        key, build = ("qkt.gmask.p", num_tok, g), lambda: qkt_group_mask(d, num_tok, g, periodic=True)
    else:
        key, build = ("qkt.gmask", num_tok, g), lambda: qkt_group_mask(d, num_tok, g)
    if scale == 1.0:
        return key, build
    return key + ("x", scale), lambda: scale * build()


def _odd_mask_item(d, num_tok: int, g: int, scale: float = 1.0):
    key, build = ("qkt.gmask.c", num_tok, g), lambda: qkt_complex_odd_mask(d, num_tok, g)
    if scale == 1.0:
        return key, build
    return key + ("x", scale), lambda: scale * build()


def _fold_mask_items(rt, cfg: SoftmaxCfg, kc: int):
    """The per-step masks SM_FOLD adds: one seed-scaled head-sum mask per reciprocal round."""
    if not (getattr(rt.ops, "sm_fold", False) and getattr(rt.ops, "sm_den_recip", False)
            and not getattr(rt.ops, "fused_sm_den", False)):
        return []
    return [_hrs_seed_item(rt.dims, 0.5 * beta * scale * scale, _seed_site(cfg, k))
            for k, (_a, beta, _i, scale) in enumerate(_recip_rounds(cfg, kc, getattr(rt.ops, "sm_gs_first", False)))]


def _softmax_mask_items(d, cfg: SoftmaxCfg, kc: int, per: bool, affine=(1.0, 0.0)):
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    sfx = ".p" if per else ""
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


def _seed_site(cfg: SoftmaxCfg, rnd: int) -> str:
    """The staging site of a reciprocal round's seed mask: one per (round, layer config), so a per-token seed is
    encoded only at the level its own block's round runs at (a shared site accumulates every block's level)."""
    tag = abs(hash((cfg.init_alpha, cfg.init_beta, tuple(cfg.refine_alpha), tuple(cfg.refine_beta)))) % 10 ** 8
    return f"hrs.seed{rnd}.{tag}"


def _hrs_seed_item(d, seed: float, site: str):
    return (site, seed), lambda: seed * hrs_pos0(d)


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


def cache_kv_push_packed_complex(rt, kv: ComplexKVCache, P_in):
    """cache_kv_push_packed_complex: the fused K + iV linear output refreshed by ONE
    bootstrap, split by conjugation (2K on the real axis, -2iV on the imaginary one), K
    masked and bucketed (odd groups packed by the monomial i), V realified (times i, the
    monomial: no level) and scattered into the pair buckets."""
    ops, d = rt.ops, rt.dims
    P = ops.copy(P_in)
    ops.bootstrap(P)
    conj = ops.conjugate(P)
    K = ops.add(P, conj)                                   # 2 K_raw
    V = ops.sub(conj, P)                                   # -2i V_raw, realified by the monomial i
    K = rt.mult_mask(K, "kpush.tok0h", lambda: real_head_half_mask(d))
    K = ops.rotate(K, -(kv.k_count % d.t))
    # bucket_k_complex
    if kv.k_count % d.t == 0:
        kv.k_pend.append(ops.copy(K))
    else:
        ops.inplace_add(kv.k_pend[-1], K)
    kv.k_count += 1
    if len(kv.k_pend) == 2 and kv.k_count % d.t == 0:      # even + odd pair complete
        bucket = ops.add(kv.k_pend[0], ops.mult_i(kv.k_pend[1]))   # pack: even + i odd (monomial)
        kv.k_buckets.append(bucket)
        kv.k_pend.clear()
    rr = kv.v_count % d.t
    v = ops.rotate(V, -rr) if rr else ops.copy(V)
    v = ops.mult_i(v)                                      # 2 V_raw, no level
    # bucket_v_complex
    c = kv.v_count // d.t
    keys, builds, lanes = [], [], []
    for p in range(d.d_head_real // 2):
        i_re = ((c - 2 * p) % d.d_head + d.d_head) % d.d_head
        i_im = ((c - 2 * p - 1) % d.d_head + d.d_head) % d.d_head
        keys.append(("v.cpc", i_re, i_im, rr))
        builds.append(lambda i_re=i_re, i_im=i_im: vpair_mask_complex(d, i_re, i_im, rr))
    tmps = rt.mult_masks(v, keys, builds)
    for p, tmp in enumerate(tmps):
        if kv.v_buckets[p] is None:
            kv.v_buckets[p] = ops.copy(tmp)
        else:
            ops.inplace_add(kv.v_buckets[p], tmp)
    kv.v_count += 1


def complex_qkt(rt, kv: ComplexKVCache, query, scale: float = 1.0):
    """complex_qkt (cachemir_complex_attention.cu): one product per K bucket serves two
    token groups; the even group's scores come out of 2 Re, the odd group's out of the
    imaginary axis through the odd mask; pending real groups run the real q.K^T."""
    ops, d = rt.ops, rt.dims
    q = rt.mult_mask(query, "tok0", lambda: real_head_tok0_mask(d))
    q = rotsum_neg(ops, q, d.t)
    kc = kv.k_count
    attn = None
    def accum(r):
        nonlocal attn
        if attn is None:
            attn = r
        else:
            ops.inplace_add(attn, r)
    for gc, bucket in enumerate(kv.k_buckets):
        g_even, g_odd = 2 * gc, 2 * gc + 1
        result = ops.mult(q, bucket)
        result = rotsum(ops, result, d.tH, d.N)
        conj = ops.conjugate(result)
        res_e = ops.add(result, conj)
        n_even = min(d.t, kc - g_even * d.t)
        accum(rt.mult_mask(res_e, *_group_mask_item(d, n_even, g_even, False, scale)))
        if g_odd * d.t < kc:
            res_o = ops.sub(result, conj)
            n_odd = min(d.t, kc - g_odd * d.t)
            accum(rt.mult_mask(res_o, *_odd_mask_item(d, n_odd, g_odd, scale)))
    per = sm_periodic(rt, kc)
    for j, pend in enumerate(kv.k_pend):
        g = 2 * len(kv.k_buckets) + j
        result = ops.mult(q, pend)
        result = rotsum(ops, result, d.tH, d.N)
        if per:
            result = ops.tag_reduce(result, d.tH)             # the ladder's output is tH-periodic
        n = min(d.t, kc - g * d.t)
        result = rt.mult_mask(result, *_group_mask_item(d, n, g, per, scale))
        accum(im_cleanse(ops, result))
    return attn


def complex_softmax_v(rt, kv: ComplexKVCache, probs):
    """complex_softmax_v: pair the scores of lanes 2j and 2j+1 as s_2j - i s_2j+1, so one
    product with the V pair bucket sums both lanes on the real axis; hoisted rotations feed
    the pair products, then the token ladder, 2 Re and the tok0 half mask."""
    ops, d = rt.ops, rt.dims
    n_pairs = d.d_head_real // 2
    if sm_periodic(rt, kv.k_count):
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


def cache_k_push(rt, kv: KVCache, key):
    """cachemir_kv_cache.cu: mask, rotate, bootstrap, im_cleanse, accumulate."""
    ops, d = rt.ops, rt.dims
    m = rt.mult_mask(key, "kpush.tok0h", lambda: real_head_half_mask(d))
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


def cache_kv_push_pair(rt, kv: KVCache, key, value):
    """cachemir_kv_cache.cu, the complex-payload arm (CKKS_COMPLEX=1, "Mode-A"): K
    and V are packed as K + iV, refreshed by ONE bootstrap and split by conjugation. The K
    mask carries an extra 0.5 (its 2 Re doubling happens before the pack), the V lanes are
    selected on the imaginary axis (2i V times -0.5 i = V)."""
    ops, d = rt.ops, rt.dims
    m = rt.mult_mask(key, "kpush.tok0h.c", lambda: 0.5 * real_head_half_mask(d))
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


def qkt(rt, kv: KVCache, query, scale: float = 1.0):
    """cachemir_attention.cu: scores at slot g*tH + h*t + tok%t."""
    ops, d = rt.ops, rt.dims
    q = rt.mult_mask(query, "tok0", lambda: real_head_tok0_mask(d))
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
    head ladder (SM_FOLD: the reciprocal's seed slope rides the mask instead of a ciphertext x constant product)."""
    ops, d = rt.ops, rt.dims
    out = rotsum(ops, x, 1, d.t)
    if seed is not None:
        tot = out
    out = rt.mult_mask(out, "hrs.pos0", lambda: hrs_pos0(d))
    out = rotsum_neg(ops, out, d.t)
    if seed is not None:
        if ops.fused_sm_den:
            raise ValueError("SM_FOLD=1 does not combine with FUSED_SM_DEN=1")
        h = rotsum_neg(ops, rt.mult_mask(tot, *_hrs_seed_item(d, seed, seed_site)), d.t)
        if not periodic:
            h = rotsum(ops, h, d.tH, d.N)
        h = ops.tag_reduce(h, d.tH)
    if periodic:
        if ops.fused_sm_den:
            raise ValueError("SM_PERIODIC=1 does not combine with FUSED_SM_DEN=1")
    elif ops.fused_sm_den:
        s_eff = ops.fold_slots_for(d.tH)
        if s_eff > d.tH:
            out = rotsum(ops, out, d.tH, s_eff)
        p = fold_sm_prescale(d.N / s_eff, s0_expected)
        ops.fold_bootstrap(out, s_eff, 1, p)
    else:
        out = rotsum(ops, out, d.tH, d.N)
    out = ops.tag_reduce(out, d.tH)                         # tH-periodic
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
    for k, (alpha, beta, iters, scale) in enumerate(_recip_rounds(cfg, kc, ops.sm_gs_first)):
        if k:
            z = ops.square(im_cleanse(ops, y))
        if fold:
            s, sh = head_reduce_sum(rt, z, s0_expected=2.0 / (alpha * scale), periodic=sm_periodic(rt, kc),
                                    seed=0.5 * beta * scale * scale, seed_site=_seed_site(cfg, k))
            y = ops.mult(z, goldschmidt_recip(ops, s, alpha, beta, iters, scale, Dh=sh))
        else:
            s = head_reduce_sum(rt, z, s0_expected=2.0 / (alpha * scale), periodic=sm_periodic(rt, kc))
            y = ops.mult(z, goldschmidt_recip(ops, s, alpha, beta, iters, scale))
    return y


def softmax_thor(rt, scores, cfg: SoftmaxCfg, kc: int, folded: bool = False):
    """cachemir_attention.cu; with SM_DEN_RECIP=1 the divisions are _softmax_recip's. `folded`: the scores come from a
    q.K^T scaled by sm_fold_affine's alpha (SM_FOLD), the exp's input map rides the score mask."""
    ops = rt.ops
    z = _thor_exp(rt, scores, cfg, kc, folded)
    if ops.sm_den_recip:
        return _softmax_recip(rt, z, cfg, kc)
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
    if sm_periodic(rt, kv.k_count):
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


def mha(rt, x, w, kv, sm_cfg: SoftmaxCfg):
    """src/model/mha.cu: qkv -> cache push -> attention core -> out projection.
    ``w`` has .q .k .v .out (EncodedLinear), or .kv (the fused K + iV linear) and .q for
    the cachemir_complex packing, whose ``kv`` is a ComplexKVCache."""
    ops = rt.ops
    cplx = isinstance(kv, ComplexKVCache)
    with rt.step("qkv"):
        ops.bootstrap_hint(x, ops.headroom(1), True)
        if cplx:
            P, q = linear_multi(rt, x, [w.kv, w.q])
        else:
            k, v, q = linear_multi(rt, x, [w.k, w.v, w.q])
    with rt.step("kv_push"):
        if cplx:
            cache_kv_push_packed_complex(rt, kv, P)
        elif ops.complex_payload:
            cache_kv_push_pair(rt, kv, k, v)           # one bootstrap for K and V
        else:
            cache_k_push(rt, kv, k)
            cache_v_push(rt, kv, v)
    with rt.step("qkt"):
        a = sm_fold_affine(rt, sm_cfg)[0]
        scores = complex_qkt(rt, kv, q, a) if cplx else qkt(rt, kv, q, a)
        folded = bool(getattr(ops, "sm_fold", False))
    with rt.step("softmax"):
        probs = softmax_thor(rt, scores, sm_cfg, kv.k_count, folded)
    with rt.step("softmax_v"):
        x = complex_softmax_v(rt, kv, probs) if cplx else softmax_v(rt, kv, probs)
    with rt.step("out_proj"):
        ops.bootstrap_hint(x, ops.headroom(1), True)
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
        items.append(_group_mask_item(d, n, ge, False, affine[0]))
        if go * d.t < kc:
            n = min(d.t, kc - go * d.t)
            items.append(_odd_mask_item(d, n, go, affine[0]))
    for g in range(2 * Nb, G):
        n = min(d.t, kc - g * d.t)
        items.append(_group_mask_item(d, n, g, per, affine[0]))
    rr = pos_in_group
    c = (kc - 1) // d.t                     # the push's v_count // t
    for p in range(d.d_head_real // 2):
        i_re = ((c - 2 * p) % d.d_head + d.d_head) % d.d_head
        i_im = ((c - 2 * p - 1) % d.d_head + d.d_head) % d.d_head
        items.append((("v.cpc", i_re, i_im, rr),
                      lambda i_re=i_re, i_im=i_im: vpair_mask_complex(d, i_re, i_im, rr)))
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
