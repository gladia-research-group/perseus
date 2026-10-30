"""The decode tail: final LayerNorm (gpt2_block.cu), the tiled LM head
(gpt2_head.cu), logits decode, the CutMax encrypted argmax
(src/algorithms/nonlinear/cutmax.cu, real K-tile path) and the encrypted feedback
embedding (gpt2_embedding.cu)."""
from __future__ import annotations

import math

import numpy as np

from perseus.impl.config import CutMaxCfg, CutMaxIter, NormCfg
from perseus.impl.layout import cutmax_tile_col_of_slot, cutmax_tile_mask, lane_vec
from perseus.impl.linear import EncodedLinear, apply_linear, linear, prepare_linear_input
from perseus.impl.norm import layer_norm
from perseus.impl.poly import (bts2, goldschmidt_inv_x0, hint, im_cleanse, inv_sqrt_newton_safe,
                   inv_sqrt_newton_scaled, pow_odd, rotsum)


def final_ln(rt, x, cfg: NormCfg, params: dict, pos: int, cap: int | None = 44):
    """apply_final_ln: layer_norm(ln_f) then the FHE_LMHEAD_CAP hint."""
    h = layer_norm(rt, x, cfg, pos, tag="lnf.", **params)
    if cap is not None:
        rt.ops.bootstrap_hint(h, int(cap), True)
    return h


def lm_head(rt, h, tiles):
    """gpt2_head.cu: one prepared input, one apply per vocab tile."""
    prep = prepare_linear_input(rt, h, tiles[0].d_in, tiles[0].d_out)
    return [apply_linear(rt, prep, w) for w in tiles]


def decode_logits(rt, tiles, vocab: int, W_tile: int):
    """decode_lm_head_logits (gpt2_head.cu): real tiles, or paired complex tiles (tile 2m on
    the real axis, 2m+1 on the imaginary one)."""
    K = (vocab + W_tile - 1) // W_tile
    out = []
    if len(tiles) < K:                                        # paired tiles
        from perseus.impl.layout import decode_linear_output as decode_np
        for m, tile in enumerate(tiles):
            cv = rt.ops.decrypt_slots_complex(tile)
            for lane, src in ((2 * m, cv.real), (2 * m + 1, cv.imag)):
                if lane >= K:
                    break
                wreal = min(W_tile, vocab - lane * W_tile)
                out.append(decode_np(src, rt.dims.N, rt.dims.hid, W_tile)[:wreal])
        return np.concatenate(out)
    for k, tile in enumerate(tiles):
        wreal = min(W_tile, vocab - k * W_tile)
        out.append(rt.ops.decode_linear_output(tile, rt.dims.hid, W_tile)[:wreal])
    return np.concatenate(out)


# ── CutMax ───────────────────────────────────────────────────────────────────────────

# On 2-iteration bootstraps. The C++ runs the whole argmax under a 2-iteration bootstrap
# scope (cutmax.cu): its deliberate sites (the level-16 hints, bootstrap_precise in
# the cascade) AND its reactive refreshes are 2-iter. The port reproduces the deliberate
# sites with `bts2`; the reactive refreshes stay the runtime's 1-iteration ones. Refreshing
# an operand *before* a product (to emulate the scope) is wrong: the operands at those sites
# (R*f ~ 3.5e5, pow_odd outputs ~ 4e6) lie far outside the bootstrap envelope, whereas the
# product the runtime refreshes is moderate.


def make_ones(ops, x):
    """make_cutmax_ones (cutmax.cu) without the const-one cache: (x + 1) - x, stamped as a
    constant like the runtime's const_one_clone: the cascade passes seeded from it then
    route their refreshes sparse (1 slot), which is where the C++ lands them."""
    return ops.tag_reduce(ops.sub(ops.add(x, 1.0), x), 1)


def cutmax_iters():
    """The refresh iterations the C++ decode runs CutMax with: scripts/run_task.sh exports
    CUTMAX_VEC_BTS_ITERS=1 (the vector-region hints) and CUTMAX_PRECISE_SCOPED=1 (the cascade
    refresh follows that scope instead of the hard 2-iteration bootstrap_precise); the code
    defaults (2) are what a bare `cuda_cachemir` run would use."""
    import os
    vec = int(os.environ.get("CUTMAX_VEC_BTS_ITERS") or os.environ.get("CUTMAX_BTS_ITERS") or 1)
    scoped = os.environ.get("CUTMAX_PRECISE_SCOPED", "1") not in ("0", "")
    return vec, (vec if scoped else 2)


def inv_sigma_cascade(ops, x, cfg: CutMaxCfg, it: CutMaxIter, ones, tap=None, iters=2):
    """cutmax.cu: passes of from-below Newton inverse-sqrt, each refreshed
    (bootstrap_precise = 2 iterations, or the scope's count under CUTMAX_PRECISE_SCOPED),
    folded into 0.5/(sqrt(s2_hi) c m)."""
    # passes after the first (and the first without a chord) start from the seed 1, which
    # inv_sqrt_newton_safe folds into a closed-form first iteration: no constant-one
    # ciphertext is needed. Pass `ones` (the constant stamped as such) to reproduce the C++
    # op stream instead.
    u_prod = None
    # the pair rides re + i im of one payload: a real-slot payload (conjugate = identity) runs them unpaired
    pair = getattr(ops, "cutmax_pair", False) and getattr(ops, "complex_payload", False)
    pending, kpair = None, 1.0
    for j in range(it.passes):
        iters = cfg.newton_per_pass + (cfg.newton_polish if j + 1 == it.passes else 0)
        y0 = ops.add(ops.mult(x, -it.cb), it.ca) if (j == 0 and it.ca != 0.0) else ones
        u = inv_sqrt_newton_safe(ops, x, y0, iters)
        if tap: tap(f"casc.u{j}", u)
        if j + 1 < it.passes:
            x = ops.mult(ops.mult(x, u), u)
        if pair and iters != 2:
            # CUTMAX_PAIR: u_j's refresh is off the path to u_{j+1} (x above uses the unrefreshed u), so two of them
            # ride one 1-slot refresh as z = u_a + i u_b; u_a u_b = Im(z^2) / 2 = i (conj(z^2) - z^2) / 4
            if pending is None and j + 1 < it.passes:
                pending = u
                continue
            if pending is not None:
                z = ops.add(pending, ops.mult_i(u))
                pending = None
                ops.bootstrap(z)
                sq = ops.square(z)
                u = ops.mult_i(ops.sub(ops.conjugate(sq), sq))
                kpair *= 0.25
            else:
                u = ops.copy(u); ops.bootstrap(u)
        elif iters == 2:
            u = bts2(ops, u)
        else:
            u = ops.copy(u); ops.bootstrap(u)
        if tap: tap(f"casc.u{j}.bts", u)
        u_prod = u if u_prod is None else ops.mult(u_prod, u)
    return ops.mult(u_prod, kpair * 0.5 / (math.sqrt(it.s2_hi) * it.c * it.m))


def cutmax_argmax(rt, tiles_in, vocab: int, cfg: CutMaxCfg, ones=None, tap=None):
    """cutmax.cu (real tiles): returns the one-hot-like z tiles (same layout as the
    logit tiles). ``ones`` is the constant-one ciphertext (the runtime's const-one cache,
    seeded from the freshest input by the model; None: each cascade derives it from its
    input, as the C++ does without the cache)."""
    ops = rt.ops
    d, N = rt.dims, rt.dims.N
    vec_iters, casc_iters = cutmax_iters()
    W_tile = N
    K = len(tiles_in)
    masks = [rt.mask(("cutmax.mask", k), lambda k=k: cutmax_tile_mask(N, k, vocab, W_tile, d.hid))
             for k in range(K)]
    # the derived per-iteration masks, named so the runtime keeps their plaintexts too
    def mk(k, scale, what):
        return rt.mask(("cutmax", what, k, scale), lambda: masks[k] * scale)
    def mm(ct, k, scale, what):
        return rt.mult_mask(ct, ("cutmax", what, k, scale), lambda: masks[k] * scale)
    def am(ct, k, scale, what):
        return rt.add_mask(ct, ("cutmax", what, k, scale), lambda: masks[k] * scale)
    # the C++ rotate_and_sum_all stamps its output as an all-slot reduction (a constant), and
    # the runtime then routes the reactive refreshes of the scalar cascade SPARSE (1 slot,
    # landing deeper): without the stamp they run dense and the cascade pays extra refreshes
    rotsum_all = lambda x: ops.tag_reduce(rotsum(ops, x, 1, N), 1)

    tiles = []
    for k in range(K):
        x = mm(tiles_in[k], k, cfg.entry_scale, "entry")
        if not (cfg.iters and cfg.iters[0].newton):        # the scaled normalisers refresh the centred logits
            x = hint(ops, x, 16, iters=vec_iters)
        if tap: tap(f"entry{k}", x)
        tiles.append(x)

    for ii, it in enumerate(cfg.iters):
        R = None
        for k in range(K):
            r = rotsum_all(tiles[k])
            R = r if R is None else ops.add(R, r)
        if it.newton:
            # the scaled normaliser of cutmax_argmax_packed, per tile
            kd = 0.5 * math.sqrt(vocab * it.x_scale) / (it.c * it.m)
            D = [ops.sub(mm(tiles[k], k, kd, ("d", ii)), mm(R, k, kd / vocab, ("mu", ii))) for k in range(K)]
            if ii == 0:
                D = [hint(ops, x, 16, iters=vec_iters) for x in D]
            S2 = None
            for k in range(K):
                sq = rotsum_all(ops.square(D[k]))
                S2 = sq if S2 is None else ops.add(S2, sq)
            t = ops.mult(S2, it.x_scale / (kd * kd))
            ops.bootstrap(t)
            u = inv_sqrt_newton_scaled(ops, t, it.newton)
            if tap: tap(f"i{ii}.S2", S2); tap(f"i{ii}.f", u)
            for k in range(K):
                y = im_cleanse(ops, am(ops.mult(D[k], u), k, 0.5 / it.m, "shift"))
                tiles[k] = pow_odd(ops, hint(ops, y, 16, iters=vec_iters), it.p)
            continue
        kf = 1.0 / math.sqrt(vocab * it.s2_hi)
        S2 = None
        for k in range(K):
            if it.ex2:
                cen = ops.mult(tiles[k], kf)
            else:
                cen = ops.sub(ops.mult(tiles[k], kf), mm(R, k, kf / vocab, "mu"))
            sq = rotsum_all(ops.square(cen))
            S2 = sq if S2 is None else ops.add(S2, sq)
        if tap: tap(f"i{ii}.R", R); tap(f"i{ii}.S2", S2)
        f = inv_sigma_cascade(ops, S2, cfg, it, ones, tap, casc_iters)
        if tap: tap(f"i{ii}.f", f)
        Rf = ops.mult(R, f)
        if tap: tap(f"i{ii}.Rf", Rf)
        for k in range(K):
            a = ops.mult(tiles[k], f)
            b = mm(Rf, k, 1.0 / vocab, "mun")
            if tap: tap(f"i{ii}.a{k}", a); tap(f"i{ii}.b{k}", b)
            y = am(ops.sub(a, b), k, 0.5 / it.m, "shift")
            y = im_cleanse(ops, y)
            if tap: tap(f"i{ii}.y{k}", y)
            y = hint(ops, y, 16, iters=vec_iters)
            if tap: tap(f"i{ii}.y{k}.bts", y)
            tiles[k] = pow_odd(ops, y, it.p)
            if tap: tap(f"i{ii}.t{k}", tiles[k])

    S = None
    for k in range(K):
        s = rotsum_all(tiles[k])
        S = s if S is None else ops.add(S, s)
    if cfg.sum_newton:
        t = ops.mult(S, cfg.sum_scale)
        ops.bootstrap(t)
        r = ops.square(inv_sqrt_newton_scaled(ops, t, cfg.sum_newton, math.sqrt(cfg.sum_scale)))
        if tap: tap("sum.S", S); tap("sum.r", r)
        return [ops.mult(tiles[k], r) for k in range(K)]
    g = 1.0 / math.sqrt(cfg.sum_lo * cfg.sum_hi)
    lo, hi = cfg.sum_lo * g, cfg.sum_hi * g
    bsum = 8.0 / ((lo + hi) * (lo + hi) + 4.0 * lo * hi)
    alpha, beta = bsum * (lo + hi), bsum
    Sn = ops.mult(S, g)
    F_init = ops.add(ops.mult(Sn, -beta), alpha)
    r = goldschmidt_inv_x0(ops, Sn, F_init, cfg.gs_sum_iters)
    r = ops.mult(r, g)
    if tap: tap("sum.S", S); tap("sum.r", r)
    return [ops.mult(tiles[k], r) for k in range(K)]


def pack_tiles(rt, tiles):
    """pack_ri (gpt2_embedding.cu): the two real logit tiles as ONE complex-payload
    ciphertext, tile 0 on the real axis and tile 1 on the imaginary axis."""
    if len(tiles) != 2:
        raise ValueError("pack_tiles: the packed path supports exactly 2 logical tiles")
    ops = rt.ops
    return ops.add(tiles[0], ops.mult_i(tiles[1]))       # tile0 + i tile1 (monomial)


def cutmax_argmax_packed(rt, pair, vocab: int, cfg: CutMaxCfg, ones=None, tap=None):
    """cutmax.cu (cutmax_packed): the two-tile CutMax on one complex-payload
    ciphertext. Lane A (real axis) carries tile 0, lane B (imaginary axis) tile 1 times the
    running sign i^p of the odd powers (`sgn`); every vector refresh is ONE bootstrap for both
    tiles, the reductions run once on the packed vector and the complex constant
    (0.5, -0.5 sgn) followed by a 2 Re fold recombines S0 + S1."""
    ops = rt.ops
    d, N = rt.dims, rt.dims.N
    vec_iters, casc_iters = cutmax_iters()
    W_tile, n = N, vocab
    masks = [rt.mask(("cutmax.mask", k), lambda k=k: cutmax_tile_mask(N, k, vocab, W_tile, d.hid))
             for k in range(2)]
    # the C++ rotate_and_sum_all stamps its output as an all-slot reduction (a constant), and
    # the runtime then routes the reactive refreshes of the scalar cascade SPARSE (1 slot,
    # landing deeper): without the stamp they run dense and the cascade pays extra refreshes
    rotsum_all = lambda x: ops.tag_reduce(rotsum(ops, x, 1, N), 1)
    # the C++ leaves its complex CutMax masks untagged (no sparse routing off them)
    def mm(ct, key, build, tagged=True):
        return ops.mult_pt(ct, rt.mask(key, build), key=rt.mask_key(key), kind="mask", tagged=tagged)
    def am(ct, key, build, tagged=True):
        return ops.add_pt(ct, rt.mask(key, build), key=rt.mask_key(key), kind="mask", tagged=tagged)

    scaled = bool(cfg.iters) and bool(cfg.iters[0].newton)
    if not scaled:
        pin = ops.mult(pair, cfg.entry_scale)
        pin = hint(ops, pin, 16, iters=vec_iters)
        A = ops.add(pin, ops.conjugate(pin))                  # 2 t0
        B = ops.sub(pin, ops.conjugate(pin))                  # 2i t1
        A = ops.mult(A, 0.5)
        B = mm(B, ("cutmax.bmask",), lambda: 0.5 * masks[1])  # canonical, junk-free
        if tap: tap("entry.A", A); tap("entry.B", B)
    # Re((1 - i sgn) (S0 + i sgn S1)) = S0 + S1: the cleanse of that Gaussian-integer product is 2 (S0 + S1), with
    # no level and the reduction's stamp kept (its refreshes route sparse)
    sum2 = lambda R_c, sgn: im_cleanse(ops, ops.sub(R_c, ops.mult_i(R_c)) if sgn > 0
                                       else ops.add(R_c, ops.mult_i(R_c)))
    sgn = 1.0
    for ii, it in enumerate(cfg.iters):
        if it.newton:
            # the scaled normaliser: D = k (x - mu) on both tiles, k = 0.5 sqrt(n x_scale) / (c m), so that
            # y = 2 Re(D u) + 1 / m for u = 1 / sqrt(t), t = x_scale n sigma^2 = (x_scale / k^2) sum |D|^2 in (0, 1]
            # (one product for both tiles' squares); D, t and u are all of order one wherever the plan refreshes them
            k = 0.5 * math.sqrt(n * it.x_scale) / (it.c * it.m)
            M = lambda: masks[0] + 1j * sgn * masks[1]
            if ii == 0:
                # the logits as the LM head leaves them: the tile split, entry_scale, k and the centring are one
                # masked product per term, so the entry refresh sees the centred logits (their spread, not their
                # common offset) one level below the head
                ke = k * cfg.entry_scale
                # the raw pair's sum: the LM head leaves tile 1's slots past the vocabulary at zero (zero weight
                # columns, no refresh yet), so its imaginary part sums to S1 (the record masks B after its entry
                # refresh, whose noise fills them)
                R2 = sum2(rotsum_all(pair), sgn)
                D = ops.sub(ops.add(mm(pair, ("cutmax.e1", ke), lambda: ke * 0.5 * (1.0 + masks[1])),
                                    mm(ops.conjugate(pair), ("cutmax.e2", ke), lambda: ke * 0.5 * (1.0 - masks[1]))),
                            mm(R2, ("cutmax.mu", ii, sgn), lambda: M() * (0.5 * ke / n), tagged=False))
                D = hint(ops, D, 16, iters=vec_iters)
            else:
                P = ops.add(A, B)                         # free: B lives on the imaginary axis
                R2 = sum2(rotsum_all(P), sgn)
                D = ops.sub(ops.mult(P, k), mm(R2, ("cutmax.mu", ii, sgn), lambda: M() * (0.5 * k / n),
                                              tagged=False))
            S2 = rotsum_all(ops.mult(D, ops.conjugate(D)))
            if tap: tap(f"i{ii}.S2", S2)
            # t is refreshed where it is formed (one 1-slot refresh): every Newton step then reads it at the landing,
            # and u ends early enough for D u to reach the vector's refresh inside its envelope
            t = ops.mult(S2, it.x_scale / (k * k))
            ops.bootstrap(t)
            u = inv_sqrt_newton_scaled(ops, t, it.newton)
            yP = am(ops.mult(D, u), ("cutmax.sh", ii, it.m, sgn), lambda: M() * (0.5 / it.m), tagged=False)
        else:
            P = ops.add(A, B)                             # free: B lives on the imaginary axis
            R_c = rotsum_all(P)                           # S0 + i sgn S1
            kf = 1.0 / math.sqrt(n * it.s2_hi)
            R = im_cleanse(ops, ops.mult_const(R_c, 0.5, -0.5 * sgn))   # S0 + S1
            kA = ops.mult(A, kf)
            kB = ops.mult(B, kf)
            if not it.ex2:
                kA = ops.sub(kA, mm(R, ("cutmax.m0", ii, kf), lambda: masks[0] * (kf / n)))
                kB = ops.sub(kB, mm(R, ("cutmax.m1", ii, kf, sgn),
                                    lambda: 1j * sgn * masks[1] * (kf / n), tagged=False))
            S2 = rotsum_all(ops.sub(ops.square(kA), ops.square(kB)))
            if tap: tap(f"i{ii}.R", R); tap(f"i{ii}.S2", S2)
            f = inv_sigma_cascade(ops, S2, cfg, it, ones, tap, casc_iters)
            Rf = ops.mult(R, f)
            a = ops.mult(P, f)
            b = mm(Rf, ("cutmax.b", ii, sgn), lambda: (masks[0] + 1j * sgn * masks[1]) / n, tagged=False)
            yP = am(ops.sub(a, b), ("cutmax.sh", ii, it.m, sgn),
                    lambda: (masks[0] + 1j * sgn * masks[1]) * (0.5 / it.m), tagged=False)
        yP = hint(ops, yP, 16, iters=vec_iters)           # ONE packed vector refresh
        if tap: tap(f"i{ii}.yP", yP)
        A = pow_odd(ops, ops.add(yP, ops.conjugate(yP)), it.p)     # y0^p
        B = pow_odd(ops, ops.sub(yP, ops.conjugate(yP)), it.p)     # i^p sgn y1^p
        sgn *= 1.0 if it.p % 4 == 1 else -1.0
        if tap: tap(f"i{ii}.end", B)
    P = ops.add(A, B)
    S_c = rotsum_all(P)
    if cfg.sum_newton:
        # 1 / S = u^2, u = 1 / sqrt(S) from Newton steps on sum_scale S in (0, 1] (sum2 is 2 S)
        S = sum2(S_c, sgn)
        t = ops.mult(S, 0.5 * cfg.sum_scale)
        ops.bootstrap(t)
        r = ops.square(inv_sqrt_newton_scaled(ops, t, cfg.sum_newton, math.sqrt(cfg.sum_scale)))
    else:
        S = im_cleanse(ops, ops.mult_const(S_c, 0.5, -0.5 * sgn))
        g = 1.0 / math.sqrt(cfg.sum_lo * cfg.sum_hi)
        lo, hi = cfg.sum_lo * g, cfg.sum_hi * g
        bsum = 8.0 / ((lo + hi) * (lo + hi) + 4.0 * lo * hi)
        alpha, beta = bsum * (lo + hi), bsum
        Sn = ops.mult(S, g)
        F_init = ops.add(ops.mult(Sn, -beta), alpha)
        r = goldschmidt_inv_x0(ops, Sn, F_init, cfg.gs_sum_iters)
        r = ops.mult(r, g)
    Z = ops.mult(P, r)
    if sgn < 0:
        Z = ops.conjugate(Z)                              # realign: t0 - i t1 -> t0 + i t1
    if tap: tap("sum.S", S); tap("sum.Z", Z)
    return Z


def decode_z(rt, z_tiles, vocab: int, W_tile: int):
    """The validation decrypt of the CutMax output (pipeline.cu): real tiles, or the
    packed pair (tile 0 on the real axis, tile 1 on the imaginary axis)."""
    if len(z_tiles) == 1 and vocab > W_tile:
        cv = rt.ops.decrypt_slots_complex(z_tiles[0])
        d = rt.dims.hid
        out = np.zeros(vocab)
        m = np.arange(W_tile)
        col = cutmax_tile_col_of_slot(m, d, W_tile)
        out[col] = cv[m].real
        ok = W_tile + col < vocab
        out[W_tile + col[ok]] = cv[m[ok]].imag
        return out
    return decode_logits(rt, z_tiles, vocab, W_tile)


# ── encrypted feedback ───────────────────────────────────────────────────────────────

def feedback_embed(rt, z_tiles, fb_tiles, wpe_row, position=None):
    """gpt2_feedback_embed + gpt2_add_positional + the entry bootstrap: the next token's input ciphertext, fresh for block 0."""
    ops, d = rt.ops, rt.dims
    # CutMax hands z back deep (level 44 with a pending rescale): a codebook linear straight
    # off it refreshes every one of its 1024 products reactively (measured: 1025 bootstraps
    # and a 9% error on the embedding). One deliberate refresh per z tile first (the C++
    # runs the linear deep and pays the same reactive refreshes; this is the port's one
    # departure from its op stream, for precision and time).
    z_tiles = [hint(ops, z, ops.headroom(2), acct=True) for z in z_tiles]
    emb = None
    if len(z_tiles) == 1 and len(fb_tiles) == 1 and fb_tiles[0].pts.dtype.kind == "c":
        # the packed pair through the complex tile (0.5 W0 - 0.5 i W1): Re((t0 + i t1)(0.5 W0
        # - 0.5 i W1)) = 0.5 (t0 W0 + t1 W1), and the 2 Re fold restores the sum
        emb = im_cleanse(ops, linear(rt, z_tiles[0], fb_tiles[0]))
    else:
        for z, w in zip(z_tiles, fb_tiles):
            y = linear(rt, z, w)
            if emb is None:
                emb = y
            else:
                ops.inplace_add(emb, y)
    # the linear leaves its sums in every lane of a feature's t-slot group; the next forward reads lane 0 only and
    # takes the others for empty (a fresh token's encoding): keep lane 0
    emb = rt.mult_mask(emb, ("feedback.lane0",), lambda: lane_vec(np.ones(d.dim), d.N, d.t))
    emb = ops.add_pt(emb, lane_vec(wpe_row, d.N, d.t), key=None if position is None else f"wpe.{position}")
    ops.bootstrap(emb)
    return emb
