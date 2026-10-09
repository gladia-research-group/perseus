"""Polynomial and iterative kernels on ciphertexts, written against the "ops" protocol
(ops.py) so the SAME code evaluates on the FHE session, on the numpy fake and on plain numpy.

Ports: src/primitives/polynomial.cu (eval_polynomial, eval_polynomial_ps,
eval_chebyshev_series, eval_taylor_inv_sqrt, eval_remez_31) and src/primitives/primitives.cu
(inv_sqrt_newton, inv_sqrt_newton_safe, goldschmidt_inv (both forms), rotate_and_sum_all,
pow_odd); im_cleanse is include/fideslib_wrapper.h and the 2-iteration bootstrap
``bts2`` is eval_bootstrap_iter (fideslib_wrapper.h) with precision 0.
"""
from __future__ import annotations

import math

import numpy as np

from .ops import FheOps, NumpyOps  # noqa: F401  (re-exported for the older import path)

# ── ladders ──────────────────────────────────────────────────────────────────────────

def rotsum(ops, x, start: int, stop: int):
    """x += rotate(x, gap) for gap = start, 2 start, ... < stop (rotate_and_sum_all with
    start=1, stop=N: primitives.cu; the mod-t / tH ladders with other starts).
    One runtime call (Context.rotate_and_sum) when available."""
    return ops.rotate_and_sum(x, start, stop)


def rotsum_neg(ops, x, stop: int):
    """x += rotate(x, -gap) for gap = 1, 2, ... < stop (the replicate-right ladders,
    cachemir_attention.cu)."""
    return ops.rotate_and_sum(x, -1, stop)


def im_cleanse(ops, x):
    """x + conj(x) = 2 Re(x) (fideslib_wrapper.h)."""
    return ops.add(x, ops.conjugate(x))


# ── polynomials ──────────────────────────────────────────────────────────────────────

def eval_polynomial(ops, x, coeffs):
    """Horner (polynomial.cu)."""
    n = len(coeffs)
    r = ops.mult(x, coeffs[n - 1])
    r = ops.add(r, coeffs[n - 2])
    for i in range(n - 3, -1, -1):
        r = ops.mult(r, x)
        r = ops.add(r, coeffs[i])
    return r


def eval_polynomial_ps(ops, x, coeffs):
    """Power-sum tree (polynomial.cu)."""
    d = len(coeffs) - 1
    if d == 0:
        z = ops.sub(x, x)
        return ops.add(z, coeffs[0])
    floor_log2 = int(math.floor(math.log2(d)))
    powers = [x]
    for _ in range(floor_log2):
        powers.append(ops.square(powers[-1]))
    result = None
    for p in range(1, len(coeffs)):
        cur = None
        pow_idx = 0
        i = p
        while i > 0:
            if i % 2 == 1:
                cur = ops.mult(powers[pow_idx], coeffs[p]) if cur is None \
                    else ops.mult(cur, powers[pow_idx])
            i //= 2
            pow_idx += 1
        if p > 1:
            ops.inplace_add(result, cur)
        else:
            result = ops.add(cur, coeffs[0])
    return result


#: A Chebyshev coefficient at or below this fraction of the largest is treated as zero. The THOR
#: GELU polynomials are odd, but a least-squares fit leaves their even coefficients at 1e-10..1e-8
#: of the peak rather than exactly 0, so each is multiplied and added for nothing. The smallest
#: live coefficient is 1.6e-3 of the peak, so the threshold separates the two cleanly.
CHEB_COEFF_EPS = 1e-6


def eval_chebyshev(ops, x, coeffs, a: float, b: float):
    """sum_k c_k T_k(y), y = (2x - (a+b))/(b-a) (polynomial.cu), c0 unhalved."""
    coeffs = [float(c) for c in coeffs]
    cmax = max((abs(c) for c in coeffs), default=0.0)
    tol = CHEB_COEFF_EPS * cmax
    coeffs = [0.0 if abs(c) <= tol else c for c in coeffs]
    n = len(coeffs) - 1
    while n > 0 and coeffs[n] == 0.0:
        n -= 1
    alpha = 2.0 / (b - a)
    beta = (a + b) / (b - a)
    # An identity affine is a level: `mult(x, 1.0)` rescales like any other plaintext product,
    # and the THOR p1 range is exactly [-1, 1]. Only read `y` below, never mutate it -- when
    # the affine is the identity it IS the caller's ciphertext.
    y = x if alpha == 1.0 else ops.mult(x, alpha)
    if beta != 0.0:
        y = ops.add(y, -beta)
    if n <= 0:
        z = ops.sub(y, y)
        return ops.add(z, coeffs[0] if coeffs else 0.0)
    if n == 1:
        r = ops.mult(y, coeffs[1])
        return ops.add(r, coeffs[0])
    T = [None] * (n + 1)
    T[1] = y
    for i in range(2, n + 1):
        if i % 2 == 0:
            sq = ops.square(T[i // 2])
            Ti = ops.add(sq, sq)
            T[i] = ops.add(Ti, -1.0)
        else:
            j = i // 2
            prod = ops.mult(T[j + 1], T[j])
            Ti = ops.add(prod, prod)
            T[i] = ops.sub(Ti, y)
    r = None
    for i in range(1, n + 1):
        if coeffs[i] == 0.0:
            continue
        term = ops.mult(T[i], coeffs[i])
        if r is None:
            r = term
        else:
            ops.inplace_add(r, term)
    return ops.add(r, coeffs[0])


def taylor_inv_sqrt_coeffs(z0: float):
    """polynomial.cu."""
    s = math.sqrt(z0)
    return [1.0 / s, -1.0 / (2.0 * z0 * s), 3.0 / (8.0 * z0 * z0 * s),
            -5.0 / (16.0 * z0 * z0 * z0 * s)]


def eval_taylor_inv_sqrt(ops, x, coeffs, z0: float):
    """polynomial.cu."""
    return eval_polynomial(ops, ops.add(x, -z0), coeffs)


# ── iterative inverses ───────────────────────────────────────────────────────────────

def goldschmidt_inv_ndf(ops, N_init, D_init, F_init, iters: int):
    """N/D by Goldschmidt with a seeded F (primitives.cu)."""
    N = ops.mult(N_init, F_init)
    F = ops.negate(F_init)
    D_neg = ops.mult(D_init, F)
    F = ops.add(D_neg, 2.0)
    for i in range(1, iters):
        N = ops.mult(N, F)
        if i + 1 < iters:
            D_neg = ops.mult(D_neg, F)
            F = ops.add(D_neg, 2.0)
    return N


def goldschmidt_recip(ops, D, alpha: float, beta: float, iters: int, scale: float = 1.0, Dh=None):
    """1/D by the goldschmidt_inv_ndf recurrence with its numerator left out, the seed
    F_init = alpha - beta x calibrated for x = scale D (F_k = 2 - D_k, D_{k+1} = D_k F_k on
    scale D; the product of the F's, times scale, is 1/D). A caller dividing a wide payload by a
    narrow D then multiplies the payload once instead of at every iteration. On a complex payload
    the running D and the reciprocal share one ciphertext, P = D_neg/2 + i R/2, so one product
    (P <- P F, F = 2 + P + conj P) and one refresh serve both; R = i (conj P - P). A real payload
    carries them as two ciphertexts. `Dh` = (b/2) D supplied by the caller (a second mask on D's reduction,
    SM_FOLD) takes the seed's ciphertext x constant products off the chain: one level less per call."""
    a, b = alpha * scale, beta * scale * scale        # scale F_init(scale D) = a - b D
    if ops.complex_payload:
        if Dh is None:
            G = ops.add(ops.mult(D, 0.5 * b), -0.5 * a)                   # -scale F_init / 2
            P = ops.add(ops.mult(D, G), ops.mult_i(ops.add(ops.mult(D, -0.5 * b), 0.5 * a)))
        else:
            G = ops.add(Dh, -0.5 * a)
            P = ops.add(ops.mult(D, G), ops.mult_i(ops.add(ops.negate(Dh), 0.5 * a)))
        for _ in range(1, iters):
            P = ops.mult(P, ops.add(ops.add(P, ops.conjugate(P)), 2.0))
        return ops.mult_i(ops.sub(ops.conjugate(P), P))
    bD = ops.mult(D, b) if Dh is None else ops.add(Dh, Dh)
    R = ops.add(ops.negate(bD), a) if Dh is not None else ops.add(ops.mult(D, -b), a)   # scale F_init
    D_neg = ops.mult(D, ops.add(bD, -a))                                  # -scale D F_init
    for i in range(1, iters):
        F = ops.add(D_neg, 2.0)
        R = ops.mult(R, F)
        if i + 1 < iters:
            D_neg = ops.mult(D_neg, F)
    return R


def goldschmidt_inv_x0(ops, a, x0_init, iters: int):
    """1/a from an initial guess (primitives.cu)."""
    x0 = x0_init
    E = ops.mult(a, x0)
    E = ops.negate(E)
    E = ops.add(E, 1.0)
    for i in range(iters):
        e_add = ops.add(E, 1.0)
        x0 = ops.mult(x0, e_add)
        if i + 1 < iters:
            E = ops.square(E)
    return x0


def eval_remez_31(ops, x, Ncoeffs, Dcoeffs, alpha: float, beta: float, gs_iters: int):
    """polynomial.cu: rational (3,1) init + Goldschmidt."""
    D = ops.mult(x, float(Dcoeffs[1]))
    D = ops.add(D, float(Dcoeffs[0]))
    F_init = ops.mult(D, -beta)
    F_init = ops.add(F_init, alpha)
    pos3 = abs(float(Ncoeffs[3]))
    sign3 = 1.0 if Ncoeffs[3] >= 0 else -1.0
    N = ops.mult(x, sign3 * pos3 ** (1.0 / 3.0))
    N = ops.add(N, float(Ncoeffs[2]) * pos3 ** (-2.0 / 3.0))
    x2 = ops.mult(x, pos3 ** (1.0 / 3.0))
    N = ops.mult(N, x2)
    N = ops.add(N, float(Ncoeffs[1]) * pos3 ** (-1.0 / 3.0))
    N = ops.mult(N, x2)
    N = ops.add(N, float(Ncoeffs[0]))
    return goldschmidt_inv_ndf(ops, N, D, F_init, int(gs_iters))


def inv_sqrt_newton(ops, x, ans_init, iters: int, x_scale: float = 1.0):
    """primitives.cu (product order (c*a)*ct)."""
    c = ops.mult(x, -0.5 * x_scale)
    ct = ans_init
    for _ in range(iters):
        a = ops.square(ct)
        b = ops.mult(c, a)
        b = ops.mult(b, ct)
        a = ops.mult(ct, 1.5)
        ct = ops.add(a, b)
    return ct


def inv_sqrt_newton_d2(ops, x, y, iters: int, x_scale: float = 1.0):
    """inv_sqrt_newton with the product order (c*y)*y^2: two levels per iteration, not three."""
    c = ops.mult(x, -0.5 * x_scale)
    for _ in range(iters):
        a = ops.square(y)
        b = ops.mult(ops.mult(c, y), a)
        y = ops.add(ops.mult(y, 1.5), b)
    return y


def inv_sqrt_newton_safe(ops, x, y0, iters: int):
    """primitives.cu (inv_sqrt_newton_safe): the from-below Newton iteration for 1/sqrt(x).
    `y0=None` means the seed 1, and then the first iteration is closed-form: with y = 1 the
    cubic term is -0.5 x itself, so y1 = 1.5 - 0.5 x costs no ciphertext product and no
    level (the C++ spends a constant-one ciphertext and three products on it)."""
    xh = ops.mult(x, -0.5)
    if y0 is None:
        if iters <= 0:
            raise ValueError("inv_sqrt_newton_safe: iters must be >= 1 with the unit seed")
        y = ops.add(xh, 1.5)
        iters -= 1
    else:
        y = y0
    for _ in range(iters):
        b = ops.mult(ops.mult(ops.mult(xh, y), y), y)
        y = ops.add(ops.mult(y, 1.5), b)
    return y


def pow_odd(ops, y, p: int):
    """primitives.cu."""
    sq = ops.square
    mu = ops.mult
    if p == 3:
        return mu(sq(y), y)
    if p == 5:
        return mu(sq(sq(y)), y)
    if p == 7:
        s1 = sq(y)
        return mu(mu(sq(s1), s1), y)
    if p == 9:
        return mu(sq(sq(sq(y))), y)
    if p == 11:
        s1 = sq(y); s3 = sq(sq(s1))
        return mu(mu(s3, s1), y)
    if p == 13:
        s2 = sq(sq(y))
        return mu(mu(sq(s2), s2), y)
    if p == 15:
        y5 = mu(sq(sq(y)), y)
        return mu(sq(y5), y5)
    if p == 19:
        y2 = sq(y); y16 = sq(sq(sq(y2)))
        return mu(mu(y16, y2), y)
    raise ValueError(f"pow_odd: unsupported p={p}")


# ── bootstrapping helpers ────────────────────────────────────────────────────────────

def bts2(ops, ct, precision=None):
    """Two-iteration bootstrap (eval_bootstrap_iter, fideslib_wrapper.h). With a
    runtime whose `bootstrap` takes an iteration count it is ONE recorded op (plannable, the
    C++ CutMax form); otherwise it is emulated from the leaf ops: y = bts(ct);
    e = (ct - y) * 2^p; bts(e); y + e * 2^-p, p = bts_precision (the scalar product costs a
    level, skipped when the residual would reach the reactive ceiling). Returns a NEW ct."""
    if ops.native_bts_iters:
        y = ops.copy(ct)
        ops.fhe.bootstrap(y, 2)
        ops.n_bootstraps += 2
        return y
    if precision is None:
        precision = getattr(ops, "bts_precision", 0)
    y = ops.copy(ct)
    ops.bootstrap(y)
    e = ops.sub(ct, y)
    if precision and ops.lvl(e) + ops.unit < ops.level_limit():
        e = ops.mult(e, 2.0 ** precision)
        ops.bootstrap(e)
        e = ops.mult(e, 2.0 ** -precision)
    else:
        ops.bootstrap(e)
    return ops.add(y, e)


def hint(ops, ct, thr: int, acct: bool = False, iters: int = 1):
    """bootstrap_hint: refresh when the (pending-aware) level exceeds `thr`. iters=2 fires the
    two-iteration bootstrap instead of the runtime's one (CutMax's BtsItersScope)."""
    if iters <= 1:
        return ops.bootstrap_hint(ct, thr, acct)
    if iters == 2:
        if ops.native_bts_iters:
            before = ops.level(ct)
            ops.fhe.bootstrap_hint(ct, int(thr), bool(acct), 2)
            if ops.level(ct) < before:
                ops.n_bootstraps += 2
            return ct
        eff = ops.lvl(ct) if acct else ops.level(ct)
        if eff > thr:
            return bts2(ops, ct)
        return ct
    return ops.bootstrap_hint(ct, thr, acct)


# ── lane-wise algebra on a complex payload ───────────────────────────────────────────
#
# A ciphertext z = a + i b carries two real payloads. Additions, rotations, plaintext
# products and bootstraps act on both lanes at once; a product mixes them (z^2 = a^2 - b^2
# + 2ab i). The cross terms cancel against the conjugate:
#
#     z conj(z)   = a^2 + b^2
#     Re(z^2)     = a^2 - b^2                (z^2 + conj(z^2)) / 2
#     a^2 + i b^2 = (1+i)/2 (z conj z) + (1-i)/4 (z^2 + conj(z^2))
#
# and likewise for a product of two packed pairs. (1 +/- i) are Gaussian integers, so with
# the level-free multiply by i they cost no rescale; the 1/4 is one scalar level unless the
# caller folds it into what it multiplies next (`unscaled=True`). Per lane-wise product: two
# ciphertext products plus conjugations, against two products for two separate ciphertexts.
#
# Measured (converging Goldschmidt on two payloads, n32 chain): with the scaled form the
# packed pair spends 2 levels per lane product against 1, so it refreshes 1.5x MORE often
# than two separate chains (9 vs 6 bootstraps) and runs 1.75x slower; the shared refresh
# only pays when the 1/4 is folded away, i.e. when a polynomial's own scalars can absorb
# powers of two (a scaled Chebyshev recurrence). Nothing in decode packs two payloads through
# a polynomial, so the model does not use this; it is here as the tool for that experiment.


def _lane_guard(ops, z):
    """A packed operand two products from the ceiling (pending rescale included) is
    refreshed deliberately: ONE bootstrap for both lanes. A product whose realized inputs
    reach the ceiling would be refreshed reactively at the edge of the bootstrap envelope,
    where the refresh comes back wrong (measured: a 44-level operand with a pending rescale
    times another lands at 48)."""
    if hasattr(ops, "level_limit") and ops.lvl(z) > ops.level_limit() - 2 * ops.unit:
        z = ops.copy(z)                  # never refresh in place: the caller may still hold z
        ops.bootstrap_hint(z, ops.level_limit() - 2 * ops.unit, True)
        # Measured on the GPU runtime: a conjugate or monomial multiply issued right after a
        # refresh of a complex-payload ciphertext can read it before the refresh has landed
        # (the chain is exact with a device fence here and wrong without one). Fence.
        ops.device_sync()
    return z


def lane_square(ops, z, unscaled=False):
    """a^2 + i b^2 for z = a + i b (twice that with `unscaled`)."""
    z = _lane_guard(ops, z)
    zc = ops.conjugate(z)
    u = ops.mult(z, zc)                       # a^2 + b^2
    w = ops.square(z)                         # a^2 - b^2 + 2ab i
    r = ops.add(w, ops.conjugate(w))          # 2 (a^2 - b^2)
    # 2 (a^2 + i b^2) = (1+i) u + (1-i) r / 2  -> with r2 = r/2 folded: (1+i) u + (1-i) (a^2-b^2)
    # keep it integer: 4 (a^2 + i b^2) = 2 (1+i) u + (1-i) r
    t = ops.add(ops.add(u, ops.mult_i(u)), ops.add(u, ops.mult_i(u)))   # 2 (1+i) u
    t = ops.add(t, ops.sub(r, ops.mult_i(r)))                          # + (1-i) r
    return t if unscaled else ops.mult(t, 0.25)


def lane_mult(ops, z1, z2, unscaled=False):
    """a1 a2 + i b1 b2 for z1 = a1 + i b1, z2 = a2 + i b2 (four times that with `unscaled`)."""
    z1, z2 = _lane_guard(ops, z1), _lane_guard(ops, z2)
    p = ops.mult(z1, z2)                      # a1a2 - b1b2 + i (a1b2 + a2b1)
    q = ops.mult(z1, ops.conjugate(z2))       # a1a2 + b1b2 + i (a2b1 - a1b2)
    rp = ops.add(p, ops.conjugate(p))         # 2 (a1a2 - b1b2)
    rq = ops.add(q, ops.conjugate(q))         # 2 (a1a2 + b1b2)
    # 4 (a1a2 + i b1b2) = (1+i) rq + (1-i) rp
    t = ops.add(ops.add(rq, ops.mult_i(rq)), ops.sub(rp, ops.mult_i(rp)))
    return t if unscaled else ops.mult(t, 0.25)


class LanePackedOps:
    """The ops protocol on a packed pair: ciphertext products become lane-wise (lane_mult /
    lane_square), a scalar add lands on both lanes, everything else passes through, so any
    kernel written on the protocol
    (eval_chebyshev, the Goldschmidt and Newton iterations, pow_odd) evaluates on both
    payloads of one ciphertext at once."""

    def __init__(self, ops):
        self._ops = ops

    def __getattr__(self, name):
        return getattr(self._ops, name)

    def _add_both(self, a, s):
        """a + s on both lanes without touching a's level: the real scalar add is free, and
        the imaginary one is i times (s ones) with s ones = (a + s) - a at a's own level and
        degree (a plaintext add would realize a pending rescale and cost the level)."""
        o = self._ops
        a_s = o.add(a, s)
        return o.add(a_s, o.mult_i(o.sub(a_s, a)))

    def add(self, a, b):
        if isinstance(b, (int, float, np.floating)):
            return self._add_both(a, float(b))
        return self._ops.add(a, b)

    def sub(self, a, b):
        if isinstance(b, (int, float, np.floating)):
            return self._add_both(a, -float(b))
        return self._ops.sub(a, b)

    def mult(self, a, b):
        if isinstance(b, (int, float, np.floating)):
            return self._ops.mult(a, b)
        return lane_mult(self._ops, a, b)

    def square(self, a):
        return lane_square(self._ops, a)

    def inplace_add(self, a, b):
        return self._ops.inplace_add(a, b)
