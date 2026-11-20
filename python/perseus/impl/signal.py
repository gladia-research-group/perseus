"""Signal processing on top of impl.fourier: convolution and cross-correlation through the CKKS encoding itself,
finite differences, spectral derivatives and 2-D filtering, on the ops protocol (so a recorded ImplModel stage plans
them like any other code).

* Coefficient-encoded signals (`coeffs_to_slots` / `slots_to_coeffs`, the client's side: impl.fourier's canonical
  embedding and its inverse). A length-N real signal (N = 2 slots) held as a polynomial's coefficients is, as slots,
  its odd-frequency DFT; one plaintext product with the slots of h is the negacyclic convolution x * h (`cconv`: one
  level, no rotation), a ciphertext product of two such signals their negacyclic product, and the conjugation is the
  automorphism X -> X^-1, so x times conj(y) is the negacyclic cross-correlation (`ccorr`: one key switch, one
  level). The data must arrive (client encryption) and leave (client decryption, or a bootstrap's CoeffsToSlots) in
  coefficient form; inside a stage it is just a slot vector.
* `xcorr_dft`: the cyclic cross-correlation of two batched slot-domain signals (impl.fourier's stride layout) by
  DFT -> conj(X) Y -> inverse DFT, both transforms at the depth impl.fourier's Budget prices.
* `fdiff`: a central finite-difference stencil (Fornberg's weights, periodic) on the stride layout: one rotation per
  tap, the weights as scalars (one level); `spectral_diff`: DFT -> (i k)^m -> inverse DFT (exact for band-limited
  signals).
* `conv2d_direct` (a k x k stencil on r x r images: one rotation per tap, one plaintext level) and `conv2d_fft`
  (impl.fourier's 2-D DFT, the filter's spectrum as one plaintext product, the inverse 2-D DFT): periodic boundaries,
  `batch` images per ciphertext (stride layout)."""
from __future__ import annotations

import math

import numpy as np

from . import fourier as F


# ── coefficient encoding ────────────────────────────────────────────────────────────

def coeffs_to_slots(x) -> np.ndarray:
    """The slot vector a client coefficient encryption of real x (length N) decrypts to (impl.fourier)."""
    return F.canonical_embedding(x)


def slots_to_coeffs(v) -> np.ndarray:
    """The real coefficients (length N = 2 len(v)) whose canonical embedding is v: the inverse of coeffs_to_slots
    (the client's coefficient decryption)."""
    v = np.asarray(v, dtype=np.complex128)
    S = v.shape[-1]
    N = 2 * S
    e = np.ones(S, dtype=np.int64)
    for j in range(1, S):
        e[j] = e[j - 1] * 5 % (2 * N)
    Y = np.zeros(v.shape[:-1] + (N,), dtype=np.complex128)
    l = (e - 1) // 2
    Y[..., l] = v
    Y[..., N - 1 - l] = np.conj(v)                         # m real: m(conj z) = conj m(z)
    m = np.fft.fft(Y, axis=-1) / N
    return np.real(m * np.exp(-1j * np.pi * np.arange(N) / N))


def cconv(ops, x, h, tag: str = "signal.cconv"):
    """x * h (negacyclic, length N) of coefficient-encoded x and a PUBLIC filter h (length N): one plaintext
    product."""
    return ops.mult_pt(x, F.negacyclic_filter(h), key=tag, kind="weight")


def ccorr(ops, x, y):
    """The negacyclic cross-correlation c_k = sum_j x_(j + k) y_j (indices mod N with the sign of X^N = -1) of two
    coefficient-encoded signals: x times the conjugated y (the automorphism X -> X^-1)."""
    return ops.mult(x, ops.conjugate(y))


def negacyclic_corr(x, y) -> np.ndarray:
    """float64 reference of ccorr: x(X) y(X^-1) mod X^N + 1."""
    y = np.asarray(y, dtype=np.float64)
    N = y.shape[-1]
    yr = np.zeros_like(y)
    yr[..., 0] = y[..., 0]
    yr[..., 1:] = -y[..., :0:-1]                           # y(X^-1) = y_0 - sum_j y_j X^(N - j)
    return F.negacyclic_conv(x, yr)


# ── slot-domain transforms ──────────────────────────────────────────────────────────

def xcorr_dft(ops, x, y, fwd: F.Transform, inv: F.Transform):
    """Cyclic cross-correlation c_k = sum_j conj(x_j) y_(j + k) of batched signals (fwd = fourier.dft, inv =
    fourier.idft on the same layout, norm "backward" pair): IDFT(conj(X) Y)."""
    X, Y = fwd.apply(ops, x), fwd.apply(ops, y)
    return inv.apply(ops, ops.mult(ops.conjugate(X), Y))


def fornberg(m: int, offsets) -> np.ndarray:
    """Finite-difference weights of the m-th derivative at 0 on the integer `offsets` (Fornberg, Math. Comp. 51
    (1988))."""
    z = np.asarray(offsets, dtype=np.float64)
    n = len(z)
    c = np.zeros((n, m + 1))
    c1, c4 = 1.0, z[0]
    c[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, m)
        c2, c5, c4 = 1.0, c4, z[i]
        for j in range(i):
            c3 = z[i] - z[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c[:, m]


def fdiff(ops, x, lay: F.Layout, m: int = 1, order: int = 2, h: float = 1.0):
    """The m-th derivative of periodic signals (stride layout: sample j of signal b at slot j batch + b) by the
    central stencil of accuracy `order` (2 p + 1 taps, p = (m + order - 1) // 2), spacing h: hoisted rotations by
    t batch, the weights / h^m as scalars, one level."""
    assert lay.mode == "stride"
    p = (m + order - 1) // 2
    offs = list(range(-p, p + 1))
    w = fornberg(m, offs) / h ** m
    steps = [t * lay.batch for t in offs if t]
    rots = dict(zip([t for t in offs if t], ops.rotate_many(x, steps)))
    rots[0] = x
    terms = [(rots[t], float(c)) for t, c in zip(offs, w) if abs(c) > 1e-15]
    return ops.scalar_sum([t for t, _ in terms], [c for _, c in terms])


def spectral_diff(ops, x, fwd: F.Transform, inv: F.Transform, m: int = 1, L: float | None = None):
    """The m-th derivative of periodic signals of length n (period L, default n: unit spacing) as
    IDFT((i k)^m DFT(x)) (fwd / inv: fourier.dft / idft "backward" on the stride layout; the Nyquist mode dropped
    for odd m)."""
    n = fwd.layout.n
    L = L or float(n)
    k = np.fft.fftfreq(n, d=1.0 / n)
    mult = (2j * np.pi * k / L) ** m
    if m % 2:
        mult[n // 2] = 0.0
    X = fwd.apply(ops, x)
    d = np.repeat(mult, fwd.layout.batch).astype(np.complex128)
    return inv.apply(ops, ops.mult_pt(X, d, key=f"signal.sdiff{m}.{n}", kind="weight"))


def conv2d_direct(ops, x, ker, r: int, lay: F.Layout):
    """Periodic 2-D correlation y[i, j] = sum_(a, b) ker[a, b] x[i + a - ca, j + b - cb] of r x r row-major images
    (stride layout, n = r^2): one hoisted rotation per tap, the taps as scalars (one level)."""
    ker = np.asarray(ker, dtype=np.float64)
    ka, kb = ker.shape
    ca, cb = ka // 2, kb // 2
    ws = [float(ker[a, b]) for a in range(ka) for b in range(kb) if ker[a, b] != 0.0]
    # a shift of the flat index by (da r + db) is the periodic 2-D shift only when it does not cross a row: the
    # row wrap needs its own diagonal (mask) per tap
    out = []
    steps = sorted(conv2d_taps(ker, r))
    rot = dict(zip(steps, ops.rotate_many(x, [s * lay.batch for s in steps])))
    rot[0] = x
    j = np.arange(r * r) % r
    for (a, b), w in zip(((a, b) for a in range(ka) for b in range(kb) if ker[a, b] != 0.0), ws):
        db = b - cb
        da = a - ca
        base = (da * r + db) % (r * r)
        if db == 0:
            out.append((rot[base], np.full(lay.slots, w)))
            continue
        # columns j with 0 <= j + db < r read the same row; the others wrap within the row: flat shift - sign(db) r
        inside = ((j + db >= 0) & (j + db < r)).astype(np.float64)
        alt = ((da * r + db - np.sign(db) * r) % (r * r))
        out.append((rot[base], np.repeat(w * inside, lay.batch)))
        out.append((rot[alt], np.repeat(w * (1.0 - inside), lay.batch)))
    acc = None
    for t, m in out:
        term = ops.mult_pt(t, m, kind="mask")
        acc = term if acc is None else ops.add(acc, term)
    return acc


def conv2d_taps(ker, r: int):
    """The flat shifts a direct 2-D stencil reads (its rotation keys / batch)."""
    ker = np.asarray(ker)
    ka, kb = ker.shape
    out = set()
    for a in range(ka):
        for b in range(kb):
            if ker[a, b] != 0.0:
                da, db = a - ka // 2, b - kb // 2
                out.add((da * r + db) % (r * r))
                if db:
                    out.add((da * r + db - int(np.sign(db)) * r) % (r * r))
    return out - {0}


def conv2d_fft(ops, x, ker, r: int, fwd: F.Transform, inv: F.Transform):
    """The same correlation as conv2d_direct by fourier's 2-D DFT: Y = IDFT2(DFT2(x) conj-free H) with H the DFT of
    the kernel's periodic correlation image (one plaintext product between the transforms)."""
    ker = np.asarray(ker, dtype=np.float64)
    ka, kb = ker.shape
    img = np.zeros((r, r))
    for a in range(ka):
        for b in range(kb):
            img[(-(a - ka // 2)) % r, (-(b - kb // 2)) % r] += ker[a, b]       # correlation = convolution with the flip
    H = np.fft.fft2(img).reshape(-1)
    X = fwd.apply(ops, x)
    d = np.repeat(H, fwd.layout.batch).astype(np.complex128)
    return inv.apply(ops, ops.mult_pt(X, d, key=f"signal.conv2d.{r}.{abs(hash(ker.tobytes())) % 10**8}",
                                      kind="weight"))


def conv2d_ref(x, ker):
    """float64 periodic 2-D correlation of (B, r, r) images."""
    ker = np.asarray(ker, dtype=np.float64)
    ka, kb = ker.shape
    out = np.zeros_like(x, dtype=np.result_type(x, np.float64))
    for a in range(ka):
        for b in range(kb):
            out = out + ker[a, b] * np.roll(x, (-(a - ka // 2), -(b - kb // 2)), axis=(-2, -1))
    return out
