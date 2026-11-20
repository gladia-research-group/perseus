"""impl.solvers / matfun / cluster / learn / signal on the plaintext mirror (NumpyOps) against float64, and their
example tasks' schedules on the counting / noisy mirrors (edge cases, error targets)."""
import math

import numpy as np
import pytest

from perseus.impl import cluster as CL
from perseus.impl import learn as LN
from perseus.impl import linalg as la
from perseus.impl import math as M
from perseus.impl import matfun as MF
from perseus.impl import signal as SG
from perseus.impl import solvers as S
from perseus.impl import fourier as F
from perseus.impl.ops import NumpyOps
from perseus.examples.linalg.tasks import test_matrix as _matrix
from perseus.examples.wave2 import tasks as T

SL = 1 << 15
ops = NumpyOps()


def _spd(rng, n, kappa, B):
    return np.stack([_matrix(rng, n, n, kappa, spd=True) for _ in range(B)])


# ── solvers ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,B", [(8, 1), (16, 128), (8, 512), (32, 32)])
def test_vec_layout(n, B):
    lay = S.VecLay(n, B)
    rng = np.random.default_rng(n + B)
    A, u, v = rng.standard_normal((B, n, n)), rng.standard_normal((B, n)), rng.standard_normal((B, n))
    a, eu, ev = lay.encode_mat(A), lay.encode_vec(u), lay.encode_vec(v)
    np.testing.assert_allclose(lay.decode_vec(S.matvec(ops, a, eu, lay)).reshape(B, n), np.einsum("bij,bj->bi", A, u),
                               atol=1e-12)
    d = S.dot(ops, eu, ev, lay)
    np.testing.assert_allclose(d, lay.encode_vec(np.repeat((u * v).sum(1, keepdims=True), n, 1)), atol=1e-12)
    noisy = eu + 1e-3 * rng.standard_normal(eu.shape)
    c = S.consolidate(ops, noisy, lay, scale=2.0)
    np.testing.assert_allclose(c, lay.encode_vec(lay.decode_vec(c)), atol=0)        # every copy equal
    assert np.abs(lay.decode_vec(c) - 2 * u.reshape(lay.decode_vec(c).shape)).max() < 1e-2


def test_vec_layout_limits():
    with pytest.raises(AssertionError):
        S.VecLay(16, 256)                       # B n^2 > slots
    assert S.VecLay.fill(16).B == 128


@pytest.mark.parametrize("kappa", [10.0, 100.0, 1000.0])
def test_chebyshev(kappa):
    n, B = 16, 32
    rng = np.random.default_rng(1)
    A = _spd(rng, n, kappa, B)
    b = rng.standard_normal((B, n))
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    lay = S.VecLay(n, B)
    xs = np.linalg.solve(A, b[..., None])[..., 0]
    for rev in (0, 2):
        x = S.chebyshev(ops, lay.encode_mat(A), lay.encode_vec(b), lay, 1.0 / (1.05 * kappa), 1.0, eps=1e-6,
                        refresh_every=rev)
        e = np.linalg.norm(lay.decode_vec(x) - xs, axis=1) / np.linalg.norm(xs, axis=1)
        assert e.max() < 1e-5, (kappa, rev, e.max())


def test_chebyshev_refinement():
    """x0 given: the solve continues from it (the residual is the right-hand side)."""
    n = 8
    rng = np.random.default_rng(2)
    A = _spd(rng, n, 50.0, 1)[0]
    b = rng.standard_normal(n)
    b /= np.linalg.norm(b)
    lay = S.VecLay(n, 1)
    xs = np.linalg.solve(A, b)
    a, eb = lay.encode_mat(A), lay.encode_vec(b)
    x1 = S.chebyshev(ops, a, eb, lay, 1 / 52.5, 1.0, iters=8)
    x2 = S.chebyshev(ops, a, eb, lay, 1 / 52.5, 1.0, iters=8, x0=x1)
    e1 = np.linalg.norm(lay.decode_vec(x1) - xs) / np.linalg.norm(xs)
    e2 = np.linalg.norm(lay.decode_vec(x2) - xs) / np.linalg.norm(xs)
    assert e2 < 0.2 * e1


def test_cg_and_plan():
    n, B = 16, 16
    rng = np.random.default_rng(3)
    A = _spd(rng, n, 10.0, B)
    b = rng.standard_normal((B, n))
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    lay = S.VecLay(n, B)
    xs = np.linalg.solve(A, b[..., None])[..., 0]
    plan = S.cg_plan(1 / 10.5, 1.0, tau=0.1)
    assert plan.iters == S.cg_iters(10.5, 1e-3) and len(plan.env) == plan.iters + 1
    assert all(e2 <= e1 for e1, e2 in zip(plan.env, plan.env[1:]))
    x = S.cg(ops, lay.encode_mat(A), lay.encode_vec(b), lay, plan)
    e = np.linalg.norm(lay.decode_vec(x) - xs, axis=1) / np.linalg.norm(xs, axis=1)
    assert e.max() < 3e-3
    # the shifted reciprocal: its argument p.Ap + lo never leaves the band
    assert plan.recip.lo == pytest.approx(plan.lmin * 0.01) and plan.shift == 2 * plan.recip.lo
    with pytest.raises(M.TargetError):
        S.cg_plan(1e-3, 1.0, tau=1e-2, recip_err=1e-3)                       # a band of 1e7: refused


def test_cg_iters_and_envelope():
    assert S.cg_iters(1.0) == 1
    assert S.cg_iters(100.0, 1e-3) > S.cg_iters(10.0, 1e-3)
    env = S.cg_envelope(100.0, 40)
    assert env[0] == 1.0 and env[-1] < 0.05


def test_jacobi_scaled():
    t = T.Solve(method="pcheb", kappa=10.0, scaled=3, n=16, B=8, eps=1e-5)
    m = t.metrics(t.mirror())
    assert m["rel_err"] < 1e-3
    # the unscaled system's conditioning is far worse than the scaled one's
    assert t.lmax / t.lmin > 30 * t.hhi / t.hlo


@pytest.mark.parametrize("method", ["cheb", "cg"])
def test_lsq_sketch_preconditioned(method):
    t = T.Lsq(method=method, kappa=100.0, eps=1e-5)
    m = t.metrics(t.mirror())
    assert m["resid_ratio"] < 1.01 and m["rel_err"] < 5e-3
    assert t.phi / t.plo < 40                       # the sketch's band, whatever kappa


def test_power_iteration():
    t = T.Power(d=16, B=4, k=2, iters=12)
    m = t.metrics(t.mirror())
    assert m["angle_max"] < 1e-2 and m["lam_rel_err"] < 1e-3


# ── matrix functions ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kappa", [10.0, 100.0, 1000.0])
def test_inv_sqrt(kappa):
    n, B = 8, 8
    rng = np.random.default_rng(4)
    A = _spd(rng, n, kappa, B)
    Fc = MF.Faces(n, B)
    zi, ys = MF.inv_sqrt(ops, Fc.encode(A), Fc, bound=1.0, kappa=1.05 * kappa, want="both")
    for Z, Y, Ab in zip(Fc.decode(zi), Fc.decode(ys), A):
        w, V = np.linalg.eigh(Ab)
        np.testing.assert_allclose(Z, (V / np.sqrt(w)) @ V.T, rtol=0, atol=1e-6 * np.sqrt(kappa))
        np.testing.assert_allclose(Y, (V * np.sqrt(w)) @ V.T, rtol=0, atol=1e-6)


def test_inv_sqrt_schedule():
    st_a, env = MF.coupled_schedule(100.0, accel=True)
    st_p, _ = MF.coupled_schedule(100.0, accel=False)
    assert len(st_a) < len(st_p)
    assert env[-1][1] == pytest.approx(10.0, rel=1e-3)        # ||Z|| -> sqrt(kappa)


def test_polish_corrects_noise():
    """The coupled steps carry a refresh's error; one polar polish step squares the whitening residual Z A Z - I."""
    n = 8
    rng = np.random.default_rng(5)
    A = _spd(rng, n, 50.0, 1)
    Fc = MF.Faces(n, 1)

    class Noisy(NumpyOps):
        def __init__(self):
            self.r = np.random.default_rng(0)

        def bootstrap(self, a):
            a += 1e-4 * np.abs(a).max() * self.r.standard_normal(a.shape)
            return a

        bootstrap_real = bootstrap
    errs = []
    for polish in (0, 1):
        Z = Fc.decode(MF.inv_sqrt(Noisy(), Fc.encode(A), Fc, bound=1.0, kappa=52.5, polish=polish))[0]
        errs.append(np.linalg.norm(Z.T @ A[0] @ Z - np.eye(n), 2))
    assert errs[1] < 0.02 * errs[0]


def test_whiten():
    m = T.Whiten(m=128, d=8, kappa=50.0).mirror()
    t = T.Whiten(m=128, d=8, kappa=50.0)
    assert t.metrics(m)["cov_err"] < 1e-6


def test_sign_projector():
    t = T.Sign(n=16, mu=0.2, gap=0.05)
    m = t.metrics(t.mirror())
    assert m["proj_err"] < 1e-6 and m["idem_err"] < 1e-6


@pytest.mark.parametrize("kind,norm", [("sym", 2.0), ("gen", 1.0), ("sym", 8.0)])
def test_expm(kind, norm):
    t = T.Expm(n=8, B=4, norm=norm, kind=kind, err=1e-7)
    m = t.metrics(t.mirror())
    assert m["rel_err"] < 1e-6


def test_expm_plan():
    d1, s1 = MF.expm_plan(1.0, 1e-6)
    d2, s2 = MF.expm_plan(16.0, 1e-6)
    assert s2 > s1
    with pytest.raises(ValueError):
        MF.expm_plan(1e4, 1e-12, max_deg=4, max_s=2)
    assert MF.taylor_depth(8) < MF.taylor_depth(24)


# ── distances / clustering ───────────────────────────────────────────────────────────

def test_points_layout():
    lay = CL.Points(64, 3, 4)
    rng = np.random.default_rng(6)
    X, Mu = rng.uniform(-1, 1, (64, 3)), rng.uniform(-1, 1, (4, 3))
    D = CL.sqdist(ops, lay.encode_points(X), lay.encode_refs(Mu), lay)
    np.testing.assert_allclose(lay.decode_dist(D), ((X[None] - Mu[:, None]) ** 2).sum(-1), atol=1e-12)
    np.testing.assert_allclose(lay.decode_refs(lay.encode_refs(Mu)), Mu)
    a = np.eye(4)[rng.integers(0, 4, 64)].T                                     # (K, N) one-hot
    s = CL.assigned_sums(ops, _enc_dist(lay, a), lay.encode_points(X), lay)
    np.testing.assert_allclose(lay.decode_refs(s), a @ X, atol=1e-12)
    with pytest.raises(AssertionError):
        CL.Points(1024, 8, 4)                                                    # d + 1 > slots / (K N)


def _enc_dist(lay, a):
    f, c, i = lay._idx()
    return a[c, i]


@pytest.mark.parametrize("method", ["ckks"])
def test_kmeans(method):
    t = T.KMeans(N=256, K=4, d=2, iters=3, method=method)
    m = t.metrics(t.mirror())
    assert m["agree_clear"] == 1.0 and m["clear_frac"] > 0.6


def test_kmeans_iris():
    from perseus.examples.wave2 import data as DS
    if not DS.available("iris"):
        pytest.skip("iris: no scikit-learn / PERSEUS_DATA_DIR")
    t = T.KMeans(N=128, K=4, data="iris", iters=3, method="ckks")
    assert t.metrics(t.mirror())["agree_clear"] == 1.0


def test_knn():
    from perseus.examples.wave2 import data as DS
    if not DS.available("breast_cancer"):
        pytest.skip("breast_cancer: no scikit-learn / PERSEUS_DATA_DIR")
    for s in range(3):
        t = T.KNN(N=128, k=5, seed=s)
        assert t.metrics(t.mirror())["agree"] == 1.0


@pytest.mark.parametrize("kind", ["sqdist", "rbf", "poly"])
def test_kernels(kind):
    t = T.Kernel(N=32, d=8, kind=kind)
    m = t.metrics(t.mirror())
    assert m["max_err"] < (1e-4 if kind == "rbf" else 1e-12)


# ── training ─────────────────────────────────────────────────────────────────────────

def test_train_matches_float():
    """The encrypted schedule on the mirror equals train_numpy with the run's own sigmoid approximation."""
    rng = np.random.default_rng(7)
    N, d = 256, 6
    X = rng.uniform(-1, 1, (N, d)) / math.sqrt(d + 1)
    w0 = rng.standard_normal(d + 1)
    y = ((np.c_[X, np.ones(N)] @ w0) > 0).astype(float)
    lay = LN.Design(N, d, 1)
    for loss in ("logistic", "svm"):
        tr = LN.Trainer(loss, lr=4.0, epochs=5)
        w = lay.decode_w(LN.train(ops, lay.encode_points(X), lay.encode_labels(y), lay, tr))
        wa = LN.train_numpy(X, y, tr, link=tr.link().numpy)
        np.testing.assert_allclose(w, wa, atol=1e-10)
        acc = (((np.c_[X, np.ones(N)] @ w) > 0) == (y > 0)).mean()
        assert acc > 0.85, (loss, acc)


def test_nesterov_faster():
    rng = np.random.default_rng(8)
    N, d = 256, 4
    X = rng.uniform(-1, 1, (N, d)) / math.sqrt(d + 1)
    y = (X[:, 0] - X[:, 1] > 0).astype(float)
    loss = []
    for nest in (False, True):
        w = LN.train_numpy(X, y, LN.Trainer("logistic", lr=4.0, epochs=8, nesterov=nest))
        p = 1 / (1 + np.exp(-(np.c_[X, np.ones(N)] @ w)))
        loss.append(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    assert loss[1] < loss[0]


@pytest.mark.parametrize("method", ["ns", "cheb"])
def test_ridge(method):
    from perseus.examples.wave2 import data as DS
    if not DS.available("wine"):
        pytest.skip("wine: no scikit-learn / PERSEUS_DATA_DIR")
    t = T.Ridge(method=method, lam=0.1, eps=1e-6)
    assert t.metrics(t.mirror())["w_rel_err"] < 1e-4


# ── signal ───────────────────────────────────────────────────────────────────────────

def test_coeff_roundtrip_conv_corr():
    rng = np.random.default_rng(9)
    N = 2 * SL
    x, y = rng.standard_normal(N), rng.standard_normal(N)
    np.testing.assert_allclose(SG.slots_to_coeffs(SG.coeffs_to_slots(x)), x, atol=1e-10)
    h = np.zeros(N)
    h[:5] = rng.standard_normal(5)
    np.testing.assert_allclose(SG.slots_to_coeffs(SG.cconv(ops, SG.coeffs_to_slots(x), h)), F.negacyclic_conv(x, h),
                               atol=1e-9)
    c = SG.slots_to_coeffs(SG.ccorr(ops, SG.coeffs_to_slots(x), SG.coeffs_to_slots(y)))
    np.testing.assert_allclose(c, SG.negacyclic_corr(x, y), atol=1e-8)
    # the reference: c_k = sum_j x_(j+k) y_j with the negacyclic sign
    k = 3
    ref = sum((x[j + k] if j + k < N else -x[j + k - N]) * y[j] for j in range(N))
    assert abs(c[k] - ref) < 1e-8 * np.abs(c).max()


def test_fornberg():
    np.testing.assert_allclose(SG.fornberg(1, [-1, 0, 1]), [-0.5, 0, 0.5])
    np.testing.assert_allclose(SG.fornberg(2, [-1, 0, 1]), [1, -2, 1])
    np.testing.assert_allclose(SG.fornberg(1, [-2, -1, 0, 1, 2]), [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])


@pytest.mark.parametrize("kw,tol", [(dict(kind="diff", method="fd", order=2), 1e-3),
                                    (dict(kind="diff", method="fd", order=8), 1e-11),
                                    (dict(kind="diff", method="fd", m=2, order=4), 1e-7),
                                    (dict(kind="diff", method="spectral"), 1e-11),
                                    (dict(kind="conv2d", method="direct", ksize=5), 1e-13),
                                    (dict(kind="conv2d", method="direct", filt="sobel"), 1e-13),
                                    (dict(kind="conv2d", method="fft", ksize=5), 1e-12),
                                    (dict(kind="xcorr", method="dft", n=256), 1e-12),
                                    (dict(kind="cconv", method="dft", n=256), 1e-12)])
def test_signal_tasks(kw, tol):
    t = T.Signal(**kw)
    assert t.metrics(t.mirror())["err_max"] < tol


def test_conv2d_taps_keys():
    """The direct stencil's rotations are its taps (and their row wraps), nothing else."""
    t = T.Signal(kind="conv2d", method="direct", ksize=3)
    steps = set(t.rotations())
    want = {s * t.lay.batch % SL for s in SG.conv2d_taps(t.ker, t.r)}
    assert steps == want
