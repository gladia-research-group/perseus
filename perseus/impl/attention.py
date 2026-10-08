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


def complex_qkt(rt, kv: ComplexKVCache, query):
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
        accum(rt.mult_mask(res_e, ("qkt.gmask", n_even, g_even),
                           lambda n=n_even, g=g_even: qkt_group_mask(d, n, g)))
        if g_odd * d.t < kc:
            res_o = ops.sub(result, conj)
            n_odd = min(d.t, kc - g_odd * d.t)
            accum(rt.mult_mask(res_o, ("qkt.gmask.c", n_odd, g_odd),
                               lambda n=n_odd, g=g_odd: qkt_complex_odd_mask(d, n, g)))
    for j, pend in enumerate(kv.k_pend):
        g = 2 * len(kv.k_buckets) + j
        result = ops.mult(q, pend)
        result = rotsum(ops, result, d.tH, d.N)
        n = min(d.t, kc - g * d.t)
        result = rt.mult_mask(result, ("qkt.gmask", n, g), lambda n=n, g=g: qkt_group_mask(d, n, g))
        accum(im_cleanse(ops, result))
    return attn


def complex_softmax_v(rt, kv: ComplexKVCache, probs):
    """complex_softmax_v: pair the scores of lanes 2j and 2j+1 as s_2j - i s_2j+1, so one
    product with the V pair bucket sums both lanes on the real axis; hoisted rotations feed
    the pair products, then the token ladder, 2 Re and the tok0 half mask."""
    ops, d = rt.ops, rt.dims
    n_pairs = d.d_head_real // 2
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


def qkt(rt, kv: KVCache, query):
    """cachemir_attention.cu: scores at slot g*tH + h*t + tok%t."""
    ops, d = rt.ops, rt.dims
    q = rt.mult_mask(query, "tok0", lambda: real_head_tok0_mask(d))
    q = rotsum_neg(ops, q, d.t)
    attn = None
    for g, kg in enumerate(kv.k_groups):
        num_tok = min(d.t, kv.k_count - g * d.t)
        res = ops.mult(q, kg)
        res = rotsum(ops, res, d.tH, d.N)
        res = rt.mult_mask(res, ("qkt.gmask", num_tok, g), lambda: qkt_group_mask(d, num_tok, g))
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


def head_reduce_sum(rt, x, s0_expected: float = 0.0):
    """cachemir_attention.cu: per-head total broadcast over the head's t slots and
    over the lane copies. With FUSED_SM_DEN the lane-copy ladder stops at the sparse slot
    count and a fold bootstrap finishes it (the denominator comes out refreshed);
    `s0_expected` sizes the fold's prescale."""
    ops, d = rt.ops, rt.dims
    out = rotsum(ops, x, 1, d.t)
    out = rt.mult_mask(out, "hrs.pos0", lambda: hrs_pos0(d))
    out = rotsum_neg(ops, out, d.t)
    if ops.fused_sm_den:
        s_eff = ops.fold_slots_for(d.tH)
        if s_eff > d.tH:
            out = rotsum(ops, out, d.tH, s_eff)
        p = fold_sm_prescale(d.N / s_eff, s0_expected)
        ops.fold_bootstrap(out, s_eff, 1, p)
    else:
        out = rotsum(ops, out, d.tH, d.N)
    return ops.tag_reduce(out, d.tH)                        # tH-periodic


def _thor_exp(rt, scores, cfg: SoftmaxCfg, kc: int):
    """The masked exponentials of softmax_thor, before its divisions."""
    ops, d = rt.ops, rt.dims
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    ct = rt.add_mask(scores, ("sm.score", cfg.clip_lo, mean, kc),
                     lambda: score_mask(d, cfg.clip_lo, mean, kc))
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    if not cfg.cheb_coeffs:
        raise ValueError("softmax: cfg.cheb_coeffs is empty (only the Chebyshev exp is ported)")
    z = eval_chebyshev(ops, ct, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf)
    for _ in range(cfg.log2delta1):
        z = ops.square(z)
    z = im_cleanse(ops, z)
    return rt.mult_mask(z, ("sm.active", kc), lambda: active_mask(d, kc))


def _softmax_recip(rt, z, cfg: SoftmaxCfg, kc: int):
    """softmax_thor's divisions as z * (1/s) (SM_DEN_RECIP=1): the Goldschmidt iterations run
    on the head sum's own tH-periodic ciphertext (goldschmidt_recip), whose refreshes route
    sparse, and the scores take one product per round instead of one per iteration. The
    refinement scalar c leaves the scores, z / s = (c z) / (c s): it only calibrates the seed.
    Same arithmetic as the default path."""
    ops = rt.ops
    c = 0.5 * math.sqrt(kc) * 0.25
    rounds = [(cfg.init_alpha, cfg.init_beta, cfg.gs_iters_scaled, 1.0)]
    for i in range(cfg.log2delta2):
        r = cfg.kc_r(i, kc)
        rounds.append((cfg.refine_alpha[i] * math.sqrt(r), cfg.refine_beta[i] * r,
                       int(cfg.per_step_refine_iters[i]), c))
    for k, (alpha, beta, iters, scale) in enumerate(rounds):
        if k:
            z = ops.square(im_cleanse(ops, y))
        s = head_reduce_sum(rt, z, s0_expected=2.0 / (alpha * scale))
        y = ops.mult(z, goldschmidt_recip(ops, s, alpha, beta, iters, scale))
    return y


def softmax_thor(rt, scores, cfg: SoftmaxCfg, kc: int):
    """cachemir_attention.cu; with SM_DEN_RECIP=1 the divisions are _softmax_recip's."""
    ops = rt.ops
    z = _thor_exp(rt, scores, cfg, kc)
    if ops.sm_den_recip:
        return _softmax_recip(rt, z, cfg, kc)
    s = head_reduce_sum(rt, z, s0_expected=2.0 / cfg.init_alpha)
    F_init = ops.mult(s, -cfg.init_beta)
    F_init = ops.add(F_init, cfg.init_alpha)
    y = goldschmidt_inv_ndf(ops, z, s, F_init, cfg.gs_iters_scaled)
    for i in range(cfg.log2delta2):
        y = im_cleanse(ops, y)
        z = ops.square(y)
        z = ops.mult(z, 0.5 * math.sqrt(kc) * 0.25)
        r = cfg.kc_r(i, kc)
        s = head_reduce_sum(rt, z, s0_expected=2.0 / (cfg.refine_alpha[i] * math.sqrt(r)))
        sa, sb = math.sqrt(r), r
        F_init = ops.mult(s, -cfg.refine_beta[i] * sb)
        F_init = ops.add(F_init, cfg.refine_alpha[i] * sa)
        y = goldschmidt_inv_ndf(ops, z, s, F_init, int(cfg.per_step_refine_iters[i]))
    return y


def softmax_v(rt, kv: KVCache, probs):
    """cachemir_attention.cu: sum_i V_lane_i * rotate(P, i*tH), token reduction,
    im_cleanse and the tok0 half mask."""
    ops, d = rt.ops, rt.dims
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
        scores = complex_qkt(rt, kv, q) if cplx else qkt(rt, kv, q)
    with rt.step("softmax"):
        probs = softmax_thor(rt, scores, sm_cfg, kv.k_count)
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
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    items = [(("sm.score", cfg.clip_lo, mean, kc), lambda: score_mask(d, cfg.clip_lo, mean, kc)),
             (("sm.active", kc), lambda: active_mask(d, kc))]
    Nb = kc // (2 * d.t)
    G = (kc + d.t - 1) // d.t
    for gc in range(Nb):
        ge, go = 2 * gc, 2 * gc + 1
        n = min(d.t, kc - ge * d.t)
        items.append((("qkt.gmask", n, ge), lambda n=n, g=ge: qkt_group_mask(d, n, g)))
        if go * d.t < kc:
            n = min(d.t, kc - go * d.t)
            items.append((("qkt.gmask.c", n, go), lambda n=n, g=go: qkt_complex_odd_mask(d, n, g)))
    for g in range(2 * Nb, G):
        n = min(d.t, kc - g * d.t)
        items.append((("qkt.gmask", n, g), lambda n=n, g=g: qkt_group_mask(d, n, g)))
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
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    items = [(("sm.score", cfg.clip_lo, mean, kc), lambda: score_mask(d, cfg.clip_lo, mean, kc)),
             (("sm.active", kc), lambda: active_mask(d, kc))]
    n_groups = (kc + d.t - 1) // d.t
    for g in range(n_groups):
        num_tok = min(d.t, kc - g * d.t)
        items.append((("qkt.gmask", num_tok, g), lambda n=num_tok, g=g: qkt_group_mask(d, n, g)))
    rr = pos_in_group
    if rt.ops.complex_payload:
        for i in range(d.d_head_real):
            items.append((("v.lane.c", i, rr), lambda i=i: complex_vlane_mask(d, i, rr)))
    else:
        for i in range(d.d_head_real):
            items.append((("v.lane", i, rr), lambda i=i: vlane_mask(d, i, rr, 0.5)))
    return items
