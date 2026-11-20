"""impl/stats.py and impl/sort.py on the plaintext mirror against numpy (both comparison back ends: the TFHE mirror is
the exact sign, the CKKS one the composite), the layouts' invariants, near-ties, and the CPU twin's level counts;
plus the example tasks (examples/mathlib) on the mirror and the twin capture -> plan -> compile path."""
import numpy as np
import pytest

from perseus.impl import sort as T
from perseus.impl import stats as S
from perseus.impl.ops import NumpyOps

OPS = NumpyOps()


def _distinct(rng, B, n, lo=-1.0, hi=1.0):
    """B rows of n values with neighbours >= (hi - lo) / (2 n) apart."""
    g = np.stack([rng.permutation(n) for _ in range(B)]).astype(float)
    return lo + (g + 0.5 + rng.uniform(-0.25, 0.25, (B, n))) * (hi - lo) / n


@pytest.mark.parametrize("n,B,pad", [(64, 4, False), (16, 1, False), (32, 8, True), (1024, 32, False)])
def test_batch_layout(n, B, pad):
    lay = S.Batch(n, B, pad=pad)
    X = np.random.default_rng(0).standard_normal((B, n))
    v = lay.encode(X)
    assert v.shape == (lay.slots,)
    np.testing.assert_array_equal(lay.decode(v), X)
    np.testing.assert_array_equal(lay.decode(np.roll(v, lay.period)), X)           # tiled
    with pytest.raises(ValueError):
        S.Batch(3, 1)
    with pytest.raises(ValueError):
        S.Batch(1 << 15, 2)


def test_reductions():
    rng = np.random.default_rng(1)
    lay = S.Batch(64, 8)
    X = rng.uniform(-2, 2, (8, 64))
    x = lay.encode(X)
    np.testing.assert_allclose(lay.decode(S.total(OPS, x, lay)), X.sum(1, keepdims=True) * np.ones((1, 64)))
    np.testing.assert_allclose(np.real(S.total(OPS, x, lay)), lay.encode(X.sum(1, keepdims=True) * np.ones((1, 64))))
    np.testing.assert_allclose(lay.decode(S.mean(OPS, x, lay)), X.mean(1, keepdims=True) * np.ones((1, 64)))
    np.testing.assert_allclose(lay.decode(S.variance(OPS, x, lay)), X.var(1, keepdims=True) * np.ones((1, 64)))
    np.testing.assert_allclose(lay.decode(S.variance(OPS, x, lay, ddof=1)), X.var(1, ddof=1, keepdims=True) *
                               np.ones((1, 64)))
    w = rng.uniform(0.1, 1.0, 64)
    mu = X @ w / w.sum()
    np.testing.assert_allclose(lay.decode(S.weighted_mean(OPS, x, lay, w))[:, 0], mu)
    np.testing.assert_allclose(lay.decode(S.weighted_sum(OPS, x, lay, w))[:, 0], X @ w)
    np.testing.assert_allclose(lay.decode(S.variance(OPS, x, lay, w=w))[:, 0], ((X - mu[:, None]) ** 2) @ w / w.sum())
    v = X.var(1)
    z = lay.decode(S.zscore(OPS, x, lay, 0.5 * v.min(), 2 * v.max(), 1e-5))
    np.testing.assert_allclose(z, (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True), atol=1e-4)
    s = lay.decode(S.std(OPS, x, lay, 0.5 * v.min(), 2 * v.max(), 1e-5))
    np.testing.assert_allclose(s[:, 0], X.std(1), atol=1e-5)


def test_prefix_sum():
    rng = np.random.default_rng(2)
    lay = S.Batch(128, 4, pad=True)
    X = rng.standard_normal((4, 128))
    x = lay.encode(X)
    np.testing.assert_allclose(lay.decode(S.prefix_sum(OPS, x, lay)), np.cumsum(X, 1), atol=1e-12)
    np.testing.assert_allclose(lay.decode(S.prefix_sum(OPS, x, lay, inclusive=False)), np.cumsum(X, 1) - X,
                               atol=1e-12)
    with pytest.raises(ValueError):
        S.prefix_sum(OPS, x, S.Batch(128, 4))


@pytest.mark.parametrize("m,d,B", [(32, 8, 2), (256, 16, 4), (8, 4, 1)])
def test_covariance_correlation(m, d, B):
    rng = np.random.default_rng(3)
    F = S.Features(m, d, B)
    X = 0.4 * (rng.standard_normal((B, m, d)) @ rng.standard_normal((d, d)) / np.sqrt(d)) + rng.uniform(-1, 1, d)
    C = F.assemble([F.decode_feature_row(v) for v in S.covariance(OPS, F.encode(X), F)])
    for b in range(B):
        np.testing.assert_allclose(C[b], np.cov(X[b].T), atol=1e-12)
    v = X.var(1)
    R = F.assemble([F.decode_feature_row(v_) for v_ in S.correlation(OPS, F.encode(X), F, 0.5 * v.min(),
                                                                       2 * v.max(), 1e-6)])
    for b in range(B):
        np.testing.assert_allclose(R[b], np.corrcoef(X[b].T), atol=1e-5)


@pytest.mark.parametrize("shift,method", [("max", "tfhe"), ("max", "ckks"), ("bound", "tfhe")])
def test_softmax_logsumexp(shift, method):
    rng = np.random.default_rng(4)
    lay = S.Batch(64, 4)
    X = _distinct(rng, 4, 64, -4.0, 4.0)
    x = lay.encode(X)
    e = np.exp(X - X.max(1, keepdims=True))
    sm = lay.decode(S.softmax(OPS, x, lay, -4.0, 4.0, 1e-3, shift=shift, method=method, s_lo=1.0))
    np.testing.assert_allclose(sm, e / e.sum(1, keepdims=True), atol=1e-3)
    lse = lay.decode(S.logsumexp(OPS, x, lay, -4.0, 4.0, 1e-3, shift=shift, method=method, s_lo=1.0))
    np.testing.assert_allclose(lse[:, 0], np.log(np.exp(X).sum(1)), atol=1e-3)


@pytest.mark.parametrize("method", ["tfhe", "ckks"])
def test_max_argmax(method):
    rng = np.random.default_rng(5)
    lay = S.Batch(64, 8)
    X = _distinct(rng, 8, 64)
    x = lay.encode(X)
    m = T.vmax(OPS, x, lay, 2.0, method=method, err=1e-4)
    np.testing.assert_allclose(np.real(m), lay.encode(X.max(1, keepdims=True) * np.ones((1, 64))), atol=1e-4)
    oh = lay.decode(T.argmax(OPS, x, lay, 2.0, method=method, res=1e-2, err=1e-4))
    np.testing.assert_allclose(oh, (X == X.max(1, keepdims=True)).astype(float), atol=2e-3)
    np.testing.assert_allclose(lay.decode(T.vmin(OPS, x, lay, 2.0, method=method, err=1e-4))[:, 0], X.min(1),
                               atol=1e-4)


@pytest.mark.parametrize("method", ["tfhe", "ckks"])
@pytest.mark.parametrize("n,B", [(16, 2), (32, 4), (8, 1)])
def test_rank_and_order_statistics(method, n, B):
    rng = np.random.default_rng(n + B)
    lay = S.Batch(n, B)
    X = _distinct(rng, B, n)
    x = lay.encode(X)
    res = 0.5 / n
    rk = np.argsort(np.argsort(X, 1), 1)
    r = T.rank(OPS, x, lay, 2.0, method=method, res=res)
    np.testing.assert_allclose(np.real(r), lay.encode(rk), atol=0.02)                 # tiled, every copy
    k = min(3, n - 1)
    np.testing.assert_allclose(lay.decode(T.topk_mask(OPS, r, lay, k, method=method)), (rk >= n - k), atol=0.01)
    np.testing.assert_allclose(lay.decode(T.kth(OPS, x, r, lay, 2, method=method))[:, 0], np.sort(X, 1)[:, 2],
                               atol=0.01)
    np.testing.assert_allclose(lay.decode(T.median(OPS, x, r, lay, method=method))[:, 0], np.median(X, 1), atol=0.01)
    np.testing.assert_allclose(lay.decode(T.quantile(OPS, x, r, lay, 0.3, method=method))[:, 0],
                               np.quantile(X, 0.3, 1), atol=0.01)
    np.testing.assert_allclose(lay.decode(T.sort_by_rank(OPS, x, r, lay, method=method)), np.sort(X, 1), atol=0.02)


def test_rank_limits_and_ties():
    with pytest.raises(ValueError):
        T.rank(OPS, np.zeros(1 << 15), S.Batch(256, 1), 2.0)
    assert T.pairs_fit(128, 1) and not T.pairs_fit(128, 2)
    lay = S.Batch(8, 1)
    X = np.array([[0.1, 0.5, 0.5, -0.3, 0.9, -0.8, 0.2, 0.0]])        # an exact tie: 0.5 twice
    r = lay.decode(T.rank(OPS, lay.encode(X), lay, 2.0, method="ckks", res=0.05))[0]
    assert r[1] == pytest.approx(5.5, abs=0.01) and r[2] == pytest.approx(5.5, abs=0.01)   # a shared half rank
    others = np.delete(r, [1, 2])
    np.testing.assert_allclose(others, np.delete(np.argsort(np.argsort(X[0])), [1, 2]).astype(float), atol=0.01)


@pytest.mark.parametrize("method", ["tfhe", "ckks"])
@pytest.mark.parametrize("n,B", [(16, 2), (64, 1)])
def test_bitonic(method, n, B):
    if method == "ckks" and n > 16:
        pytest.skip("the CKKS network's mirror is slow at n = 64; the twin test covers its schedule")
    rng = np.random.default_rng(7)
    lay = S.Batch(n, B)
    X = _distinct(rng, B, n)
    idx = lay.encode(np.tile(np.arange(n, dtype=float), (B, 1)))
    xs, p = T.bitonic(OPS, lay.encode(X), lay, 2.0, method=method, res=0.5 / n, payload=idx)
    tol = 1e-9 if method == "tfhe" else 0.005           # CKKS: |d| DELTA per step, blended swaps
    np.testing.assert_allclose(np.real(xs), lay.encode(np.sort(X, 1)), atol=tol)
    np.testing.assert_allclose(lay.decode(p), np.argsort(X, 1), atol=0.02)
    xd, _ = T.bitonic(OPS, lay.encode(X), lay, 2.0, method=method, res=0.5 / n, descending=True)
    np.testing.assert_allclose(lay.decode(xd), -np.sort(-X, 1), atol=tol)
    assert len(T.bitonic_steps(n)) == int(np.log2(n) * (np.log2(n) + 1) / 2)


@pytest.mark.parametrize("method", ["tfhe", "ckks"])
def test_threshold(method):
    rng = np.random.default_rng(8)
    n, B, k, P = 256, 2, 8, 10
    lay = S.Batch(n, B)
    X = _distinct(rng, B, n)
    t, mask = T.threshold(OPS, lay.encode(X), lay, k, -1.0, 1.0, probes=P, method=method)
    kth = np.sort(X, 1)[:, n - k]
    tt = lay.decode(t)[:, 0]
    assert np.all(tt <= kth + 1e-3) and np.all(kth < tt + 2.0 / 2 ** P + 1e-3)
    rk = np.argsort(np.argsort(X, 1), 1)
    np.testing.assert_allclose(lay.decode(mask), (rk >= n - k).astype(float), atol=2e-3)


# ── the twin: levels / refreshes of the CKKS paths, the TFHE switch recorded as a refresh ──────────────────────────


@pytest.fixture(scope="module")
def twin():
    from perseus.examples.gpt2 import env
    from perseus.impl import fakecap
    from perseus.impl.layout import Dims
    from perseus.impl.ops import FheOps
    over = {"AUTO_BTS_LEVEL": "50"}
    env.export_env(None, chain="n32", CKKS_COMPLEX="1", **over)
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
    steps = set()
    for lay in (S.Batch(64, 8), S.Batch(16, 2)):
        steps |= {s % lay.slots for s in lay.rotations() + T.rank_rotations(lay) + T.tile_rotations(lay) +
                  T.bitonic_rotations(lay)}
    _, inf = fakecap.make_session(Dims.gpt2(), steps - {0}, limit=200, bts_out=32)
    ops = FheOps(inf, fakecap.core)

    def fresh(v):
        z = ops.core.encode_token_input(ops.inf, np.zeros(Dims.gpt2().dim))
        return ops.add_pt(z, v)
    return ops, fresh


def test_twin_levels(twin):
    ops, fresh = twin
    lay = S.Batch(64, 8)
    X = np.random.default_rng(9).uniform(-2, 2, (8, 64))
    x = fresh(lay.encode(X))
    l0 = ops.lvl(x)
    t = S.total(ops, x, lay)
    assert ops.lvl(t) == l0                                       # a ladder costs no level
    v = S.variance(ops, x, lay)
    assert ops.lvl(v) - l0 == 2 * ops.unit
    np.testing.assert_allclose(lay.decode(ops.decrypt_slots_complex(v))[:, 0], X.var(1), atol=1e-12)


def test_twin_rank_values(twin):
    ops, fresh = twin
    lay = S.Batch(16, 2)
    X = _distinct(np.random.default_rng(10), 2, 16)
    r = T.rank(ops, fresh(lay.encode(X)), lay, 2.0, method="ckks", res=0.5 / 16)
    np.testing.assert_allclose(lay.decode(ops.decrypt_slots_complex(r)), np.argsort(np.argsort(X, 1), 1), atol=0.02)


# ── the example tasks ───────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name,kw", [
    ("fn", dict(fn="log", lo=0.5, hi=8.0)), ("fn", dict(fn="relu", lo=-1.0, hi=1.0, err=1e-3, method="tfhe")),
    ("stats", dict(kind="zscore", n=64, B=8)), ("stats", dict(kind="cov", m=32, d=8, B=2)),
    ("sort", dict(kind="rank", n=16, method="ckks")), ("sort", dict(kind="median", n=16, method="tfhe")),
    ("pipeline", dict(n=16, B=4, method="tfhe")),
])
def test_tasks_mirror(name, kw):
    from perseus.examples.mathlib import tasks
    t = tasks.make(name, **kw)
    met = t.metrics(t.mirror())
    errs = [v for k, v in met.items() if k.endswith("err") or k.startswith("max")]
    assert max(errs) < 5e-3, met
    for k in ("exact", "topk_exact"):
        if k in met:
            assert met[k] == 1.0, met


def test_pipeline_capture_plan_compile(tmp_path):
    """The planner-ready example end to end on the CPU: twin capture -> dense plan -> compiled program."""
    import os
    import subprocess
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=os.path.join(repo, "python"),
               PLAN_SPARSE_SLOTS="", PLAN_HARD_ENV_CAP="1", PLAN_DELIBERATE_BAND_CF="1",
               PC_WORK_DIR=str(tmp_path))
    graph = tmp_path / "graph"
    plan = os.path.join("bootstrap_placements", "_test_mathlib_pipeline")
    me = [sys.executable, "-m", "perseus.examples.mathlib.run", "--task", "pipeline", "--set", "n=16", "B=4"]
    try:
        subprocess.run(me + ["--fake", "--capture", str(graph)], check=True, cwd=repo, env=env,
                       stdout=subprocess.DEVNULL)
        subprocess.run(["bash", "python/perseus/examples/gpt2/make_plan.sh", str(graph), plan[len("bootstrap_placements/"):]],
                       check=True, cwd=repo, env=dict(env, PYTHON=sys.executable), stdout=subprocess.DEVNULL)
        out = subprocess.run(me + ["--plan", plan, "--compile-only"], check=True, cwd=repo, env=env,
                             capture_output=True, text=True).stdout
        assert "compiled:" in out
    finally:
        import shutil
        shutil.rmtree(os.path.join(repo, plan), ignore_errors=True)


@pytest.mark.parametrize("method", ["tfhe", "ckks"])
def test_rank_bitonic(method):
    """Ranks for n past the all-pairs limit: the argsort of the argsort (two networks)."""
    rng = np.random.default_rng(11)
    n = 32 if method == "ckks" else 256
    lay = S.Batch(n, 1)
    X = _distinct(rng, 1, n)
    r = lay.decode(T.rank_bitonic(OPS, lay.encode(X), lay, 2.0, method=method, res=0.25 / n))
    np.testing.assert_allclose(r, np.argsort(np.argsort(X, 1), 1), atol=0.1)


def test_tfhe_levels_resolve_near_ties():
    """msign's finer scales (levels=2) decide differences ~16x below the one-level band (the mirror is exact: here
    the comparisons see the switch's noise model)."""
    import importlib
    import os
    from perseus.impl import tfhe
    os.environ["PF_TFHE_NOISE"] = "5.3e-3"
    importlib.reload(tfhe)
    try:
        lay = S.Batch(256, 1)
        X = np.random.default_rng(12).uniform(-1, 1, (1, 256))
        e = {}
        for lv in (1, 2):
            out, _ = T.bitonic(OPS, lay.encode(X), lay, 2.0, method="tfhe", levels=lv)
            e[lv] = np.abs(lay.decode(out) - np.sort(X, 1)).max()
        assert e[2] < 2e-3 < e[1]
    finally:
        os.environ.pop("PF_TFHE_NOISE", None)
        importlib.reload(tfhe)
