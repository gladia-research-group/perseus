"""impl/linalg.py on the plaintext mirror (NumpyOps) against numpy float64, a noisy mirror (iid slot noise on every
product, the CKKS rounding model) for the copy-consistency invariants, and the CPU twin for the level schedule."""
import numpy as np
import pytest

from perseus.impl import linalg as la
from perseus.impl.ops import NumpyOps
from perseus.examples.linalg import tasks as T

S = 1 << 15


class Noisy(NumpyOps):
    """iid Gaussian noise of std eps on every ciphertext and plaintext product output."""

    def __init__(self, eps, seed=0):
        self.eps, self.rng = eps, np.random.default_rng(seed)

    def _n(self, x):
        return x + self.eps * self.rng.standard_normal(np.shape(x))

    def mult(self, a, b):
        return self._n(super().mult(a, b))

    def mult_pt(self, a, v, **kw):
        return self._n(super().mult_pt(a, v))


@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_product_and_faces(n):
    rng = np.random.default_rng(n)
    cube = la.full_cube(n, n, S)
    lo, ll, lr = la.O(cube, (n, n)), la.L(cube, (n, n)), la.R(cube, (n, n))
    X, Y = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    ops = NumpyOps()
    z = la.mm(ops, la.relayout(ops, la.encode(X, lo), lo, ll), la.relayout(ops, la.encode(Y, lo), lo, lr), cube)
    np.testing.assert_allclose(la.decode(z, lo), X @ Y, atol=1e-12)
    np.testing.assert_allclose(z, la.encode(X @ Y, lo), atol=1e-12)          # every replicated copy
    t = la.relayout(ops, la.encode(X, lo), lo, lo, transpose=True, scale=0.5)
    np.testing.assert_allclose(la.decode(t, lo), 0.5 * X.T, atol=1e-14)
    # the face moves stay cheap: O -> R n diagonals, O -> L 2n - 1
    assert la.relayout_plan(lo, lr).n_diag == n and la.relayout_plan(lo, ll).n_diag == 2 * n - 1


def test_gathers():
    rng = np.random.default_rng(0)
    m, n = 32, 8
    lay = la.O(la.full_cube(m, n, S), (m, n))
    A = rng.standard_normal((m, n))
    ops = NumpyOps()
    for j in (0, 3, 7):
        np.testing.assert_allclose(la.decode(la.relayout(ops, la.encode(A, lay), lay, lay, gather=("col", j)), lay),
                                   np.outer(A[:, j], np.ones(n)))
        np.testing.assert_allclose(la.decode(la.relayout(ops, la.encode(A, lay), lay, lay, gather=("row", j)), lay),
                                   np.outer(np.ones(m), A[j]))


@pytest.mark.parametrize("spd", [False, True])
@pytest.mark.parametrize("method", ["ns", "gs"])
@pytest.mark.parametrize("kappa", [10.0, 1000.0])
def test_inverse_mirror(spd, method, kappa):
    t = T.Inverse(n=16, kappa=kappa, spd=spd, method=method)
    m = t.metrics(t.mirror())
    assert m["rel_err"] < 1e-6, m


def test_tiled_cube_is_unstable_full_cube_is_not():
    """The finding behind full_cube: on a tiled (n, n, n) cube the product's axis-0 sum crosses tiles, the copies of
    the iterate drift apart and Newton-Schulz amplifies a 1e-11 slot noise to O(1e-2); on the full cube the copies
    stay identical and the error stays at the noise floor."""
    n, s = 8, 100.0
    A = T.test_matrix(np.random.default_rng(0), n, n, s)
    Xi = np.linalg.inv(A)
    errs = {}
    for name, cube in (("tiled", la.Cube((n, n, n), S)), ("full", la.full_cube(n, n, S))):
        ops = Noisy(1e-11)
        lo, ll, lr = la.O(cube, (n, n)), la.L(cube, (n, n)), la.R(cube, (n, n))
        va = la.encode(A, lo)
        al = la.relayout(ops, va, lo, ll)
        it = la.ns_iters(s)
        sc = la.ns_scales(1.0 / s, 1.0, 1.0, it, False)
        zl = la.relayout(ops, va, lo, ll, scale=1 / sc[0], transpose=True)
        zr = la.relayout(ops, va, lo, lr, transpose=True)
        for k in range(it):
            zo = la._ns_step(ops, al, zl, zr, cube, sc[k] / sc[k + 1])
            zl, zr = la.relayout(ops, zo, lo, ll), la.relayout(ops, zo, lo, lr, scale=sc[k + 1])
        zo = sc[it] * zo / s
        errs[name] = np.abs(s * la.decode(zo, lo) - Xi).max() / np.abs(Xi).max()
    assert errs["full"] < 1e-6 and errs["tiled"] > 1e3 * errs["full"], errs


def test_consolidate_makes_copies_equal():
    n = 8
    cube = la.full_cube(n, n, S)
    lo = la.O(cube, (n, n))
    v = la.encode(np.arange(n * n, dtype=float).reshape(n, n), lo) + 1e-3 * np.random.default_rng(0).standard_normal(S)
    c = la.consolidate(NumpyOps(), v, lo)
    np.testing.assert_allclose(c, la.encode(la.decode(c, lo), lo), atol=1e-15)


@pytest.mark.parametrize("method,kw", [("ns", {}), ("cheb", {"kappa": 10.0}), ("ns", {"ridge": 1e-3}),
                                       ("ns", {"m": 1024, "sketch": 128})])
def test_lstsq_mirror(method, kw):
    t = T.Lstsq(method=method, **kw)
    m = t.metrics(t.mirror())
    if kw.get("sketch"):
        assert m["resid_ratio"] < 1.3, m                    # sketch-and-solve: (1 + eps) of optimal
    else:
        assert m["resid_ratio"] < 1.0 + 1e-2 and m["rel_err"] < 1e-3, m


@pytest.mark.parametrize("method,kw", [("mgs", {}), ("cqr2", {}), ("scqr3", {"kappa": 1000.0, "shift": 1e-4})])
def test_qr_mirror(method, kw):
    t = T.QR(m=64, n=8, method=method, **kw)
    m = t.metrics(t.mirror())
    assert m["orth"] < 1e-4 and m["resid"] < 1e-4, m


def test_pca_and_rsvd_mirror():
    p = T.PCA(N=256, d=8, k=2, iters=6)
    mp = p.metrics(p.mirror())
    assert mp["angle"] < 1e-3 and mp["expl_var_ratio"] > 0.999, mp
    r = T.RSVD(m=32, n=16, k=2, p=2, q=1)
    mr = r.metrics(r.mirror())
    assert mr["sv_rel_err"] < 1e-3 and mr["angle_V"] < 1e-2, mr


def test_inverse_twin_schedule():
    """The twin records one deliberate refresh per Newton-Schulz step and no reactive one."""
    from perseus.examples.linalg.run import main
    rows = []
    import perseus.examples.linalg.run as run
    orig = run._report
    run._report = lambda a, task, out, mode, extra: rows.append((task.metrics(out), extra)) or rows[-1]
    try:
        main(["--task", "inverse", "--set", "n=8", "kappa=100", "--fake"])
    finally:
        run._report = orig
    met, extra = rows[0]
    assert met["rel_err"] < 1e-6 and extra["bts_reactive"] == 0 and extra["bts"] == la.ns_iters(100.0) - 1


class RefreshNoise(NumpyOps):
    """A refresh adds iid Gaussian noise of std eps x the largest slot (the twin's FAKECAP_BTS_NOISE model)."""

    def __init__(self, eps, seed=0):
        self.eps, self.rng = eps, np.random.default_rng(seed)

    def bootstrap(self, a):
        a += self.eps * np.abs(a).max() * self.rng.standard_normal(a.shape)
        return a

    bootstrap_real = bootstrap


def test_pinv_beats_normal_equations_under_refresh_noise():
    """Least squares at cond(A) = 100 with a 1e-4 refresh error: through (A^T A)^{-1} the error is amplified by
    cond(A)^2 (second order: (eps cond^2)^2 ~ 1); Newton-Schulz on A^+ amplifies by cond(A)."""
    errs = {}
    for method in ("ns", "pinv"):
        t = T.Lstsq(m=128, n=16, kappa=100.0, method=method)
        out = t.outputs([np.asarray(v) for v in t.run(RefreshNoise(1e-4), t.encode())])
        errs[method] = t.metrics(out)["rel_err"]
    assert errs["pinv"] < 1e-3 and errs["ns"] > 30 * errs["pinv"], errs


def test_polar_ns_orthonormality_self_corrects():
    t = T.QR(m=128, n=16, kappa=100.0, method="polarns")
    out = t.outputs([np.asarray(v) for v in t.run(RefreshNoise(3e-4), t.encode())])
    m = t.metrics(out)
    assert m["orth"] < 1e-5 and m["resid"] < 3e-2, m


@pytest.mark.parametrize("n,B,spd", [(8, 64, False), (8, 64, True), (16, 8, False), (8, 1, False)])
def test_inverse_fast_batched_under_refresh_noise(n, B, spd):
    """inverse_fast: B problems per ciphertext, the accelerated (guarded) schedule, wrap faces; a 3e-4 refresh error
    leaves every problem's inverse within 1e-3 (the unguarded shift flips eigenvalues lifted past 1: errors O(1))."""
    t = T.Inverse(n=n, kappa=100.0, fast=True, batch=B, spd=spd)
    assert t.metrics(t.mirror())["rel_err"] < 1e-8
    out = t.outputs([np.asarray(v) for v in t.run(RefreshNoise(3e-4), t.encode())])
    assert t.metrics(out)["rel_err"] < 1e-3


def test_schedule_and_precond_counts():
    """The accelerated schedule: 12 steps at cond 100 (plain 18); the client-preconditioned least squares runs at
    cond(A R^-1) ~ 3 whatever cond(A): 6 refreshes, accurate at cond(A) = 1e4."""
    assert la.ops_count_schedule(100.0) == 12 and la.ops_count_schedule(100.0, accel=False) == 18
    t = T.Lstsq(kappa=1e4, method="precond")
    out = t.outputs([np.asarray(v) for v in t.run(RefreshNoise(3e-4), t.encode())])
    m = t.metrics(out)
    assert T.circuit(t)["bts"] <= 6 and m["resid_ratio"] < 1.001, m
