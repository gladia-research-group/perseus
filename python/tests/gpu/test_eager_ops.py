"""Eager `_pc.Fhe` leaf ops on the GPU vs numpy.

Every leaf op on random slot vectors at several levels against the numpy result (relative error on the decoded
slots <= 1e-6 for one op), chains of 10 ops, the plaintext caches, the data plane, the debug refresh, and the
per-op Python -> device host overhead (median over a loop, reported and gated at 20 us).

    tools/gpu.sh out/eager_ops.log .venv/bin/python -m pytest -m gpu python/tests/gpu -q -s
"""
from __future__ import annotations

import time

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

N = 1 << 15                    # slots
STEPS = [1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56, -1, -2, -3, 64, 1024, 4096, 16384]
GROUPS = [[1, 2, 3, 4, 5, 6, 7], [8, 16, 24, 32, 40, 48, 56]]
LEVELS = [0, 12, 24, 36]
TOL = 1e-6


def rel(got, want):
    got, want = np.asarray(got), np.asarray(want)
    return float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-300))


@pytest.fixture(scope="module")
def sess():
    cfg = _pc.Config.preset("n32_gpt2_decode")
    keys = _pc.KeySource.generate(STEPS, groups=GROUPS, seed=11)
    s = _pc.Session(cfg, keys, pool=96, pt_slots=48, debug_refresh=True).complete_setup()
    print(f"\n[m5d] session: {s.mem_report}; setup / keygen {s.setup_seconds()} s; {s.n_keys()} keys")
    yield s
    s.close()


@pytest.fixture(scope="module")
def fhe(sess):
    return sess.fhe


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def cvec(rng, scale=1.0):
    return scale * (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)


def enc(fhe, v, level, deg=1):
    return fhe.encrypt(v, level, deg)


def dec(fhe, a):
    return fhe.decrypt_slots_complex(a)


# ── one op per call, several levels ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("level", LEVELS)
def test_additive(fhe, rng, level):
    x, y = cvec(rng), cvec(rng)
    a, b = enc(fhe, x, level), enc(fhe, y, level)
    assert rel(dec(fhe, fhe.add(a, b)), x + y) < TOL
    assert rel(dec(fhe, fhe.sub(a, b)), x - y) < TOL
    assert rel(dec(fhe, fhe.add(a, 0.75)), x + 0.75) < TOL
    assert rel(dec(fhe, fhe.sub(a, 0.75)), x - 0.75) < TOL
    assert rel(dec(fhe, fhe.negate(a)), -x) < TOL
    assert rel(dec(fhe, fhe.mult_i(a)), 1j * x) < TOL
    c = fhe.add(a, 0.0)
    fhe.inplace_add(c, b)
    assert rel(dec(fhe, c), x + y) < TOL
    fhe.inplace_add(c, c)                                  # aliasing: c += c
    assert rel(dec(fhe, c), 2 * (x + y)) < TOL
    assert (a.level, a.deg) == (level, 1)


@pytest.mark.parametrize("level", LEVELS)
def test_products(fhe, rng, level):
    x, y = cvec(rng), cvec(rng)
    a, b = enc(fhe, x, level), enc(fhe, y, level)
    p = fhe.mult(a, b)
    assert p.deg == 2
    assert rel(dec(fhe, p), x * y) < TOL
    assert rel(dec(fhe, fhe.square(a)), x * x) < TOL
    assert rel(dec(fhe, fhe.mult(a, -1.5)), -1.5 * x) < TOL
    assert rel(dec(fhe, fhe.mult(a, a)), x * x) < TOL
    w = rng.standard_normal(N)
    assert rel(dec(fhe, fhe.mult_pt(a, w)), x * w) < TOL
    assert rel(dec(fhe, fhe.add_pt(a, w)), x + w) < TOL
    assert rel(dec(fhe, fhe.mult_const(a, 0.5, -2.0)), x * (0.5 - 2.0j)) < TOL
    # a product of a product: the Rescale prelude runs in Ops::mult
    assert rel(dec(fhe, fhe.mult(p, b)), x * y * y) < TOL


def test_mixed_levels(fhe, rng):
    """Operands at different levels / degrees: the Recipe preludes (Rescale, DropTo, ScalarAdjust) run first."""
    x, y, z = cvec(rng), cvec(rng), cvec(rng)
    a, b = enc(fhe, x, 4), enc(fhe, y, 20)
    p = fhe.mult(a, enc(fhe, z, 4))                       # level 4, deg 2
    for got, want in ((fhe.add(a, b), x + y), (fhe.add(b, a), x + y), (fhe.sub(p, b), x * z - y),
                      (fhe.mult(p, b), x * z * y), (fhe.mult(b, p), x * z * y), (fhe.add(p, a), x * z + x)):
        assert rel(dec(fhe, got), want) < TOL


@pytest.mark.parametrize("level", [0, 30])
def test_rotations(fhe, rng, level):
    x = cvec(rng)
    a = enc(fhe, x, level)
    for k in (1, 7, -1, -3, 1024, 16384):
        assert rel(dec(fhe, fhe.rotate(a, k)), np.roll(x, -k)) < TOL, k
    assert rel(dec(fhe, fhe.conjugate(a)), np.conj(x)) < TOL
    steps = [0, 8, 16, 24, 32, 40, 48, 56]
    for k, r in zip(steps, fhe.rotate_hoisted(a, steps)):
        assert rel(dec(fhe, r), np.roll(x, -k)) < TOL, k
    want = x + sum(np.roll(x, -k) for k in GROUPS[0])
    assert rel(dec(fhe, fhe.rotate_sum(a, GROUPS[0])), want) < TOL
    for start, stop in ((1, 16), (8, 64), (-1, 4), (1, 3)):
        ref, gap, sign = x.copy(), abs(start), (1 if start > 0 else -1)
        while gap < stop:
            ref = ref + np.roll(ref, -sign * gap)
            gap *= 2
        assert rel(dec(fhe, fhe.rotate_and_sum(a, start, stop)), ref) < TOL, (start, stop)
    with pytest.raises(KeyError):
        fhe.rotate(a, 9)


def test_fused_sums(fhe, rng):
    xs = [cvec(rng) for _ in range(5)]
    cts = [enc(fhe, x, 10) for x in xs]
    cs = [0.5, -1.0, 2.0, 0.25, 3.0]
    assert rel(dec(fhe, fhe.mult_scalar_sum(cts, cs)), sum(c * x for c, x in zip(cs, xs))) < TOL
    ws = rng.standard_normal((5, N))
    assert rel(dec(fhe, fhe.mult_pt_sum(cts, ws)), sum(w * x for w, x in zip(ws, xs))) < TOL
    # terms at different levels: the eager loop (Ops::mult_add_many refuses terms with preludes)
    mixed = [enc(fhe, xs[0], 10), enc(fhe, xs[1], 20), fhe.mult(enc(fhe, xs[2], 6), 1.0)]
    assert rel(dec(fhe, fhe.mult_pt_sum(mixed, ws[:3])), sum(w * x for w, x in zip(ws[:3], xs[:3]))) < TOL
    acc = enc(fhe, xs[0], 10)
    fhe.mult_add_many(acc, cts[1:3], cts[3:5])
    assert rel(dec(fhe, acc), xs[0] + xs[1] * xs[3] + xs[2] * xs[4]) < 1e-5   # acc at degree 2 after the adds


@pytest.mark.parametrize("level", [10, 40])
def test_mult_sum(fhe, rng, level):
    """sum_j a_j * b_j relinearized once: 1, 2 (one launch), 3 (odd tail) and 40 pairs (d2 accumulated across launches)."""
    for n in (1, 2, 3, 40):
        xs, ys = [cvec(rng) for _ in range(n)], [cvec(rng) for _ in range(n)]
        a, b = [enc(fhe, x, level) for x in xs], [enc(fhe, y, level) for y in ys]
        want = sum(x * y for x, y in zip(xs, ys))
        p = fhe.mult_sum(a, b)
        assert p.deg == 2
        assert rel(dec(fhe, p), want) < TOL, n
        del a, b, p
    with pytest.raises(Exception):
        fhe.mult_sum([enc(fhe, xs[0], level)], [enc(fhe, ys[0], level + 2)])


def test_plaintext_caches(fhe, rng):
    x, w = cvec(rng), rng.standard_normal(N)
    a, b = enc(fhe, x, 8), enc(fhe, x, 20)
    fhe.set_slot_pt("impl.t.w0", w)
    assert fhe.has_slot_pt("impl.t.w0")
    for c in (a, b, a):
        assert rel(dec(fhe, fhe.mult_slot_pt(c, "impl.t.w0")), x * w) < TOL
    assert rel(dec(fhe, fhe.add_slot_pt(a, "impl.t.w0")), x + w) < TOL
    many = fhe.mult_slot_pt_many(a, ["impl.t.w0", "impl.t.w0"])
    assert len(many) == 2 and rel(dec(fhe, many[1]), x * w) < TOL
    assert fhe.evict_slot_pts("impl.t.") == 1
    assert rel(dec(fhe, fhe.mult_slot_pt(a, "impl.t.w0")), x * w) < TOL   # re-encoded on use
    assert fhe.drop_slot_pts("impl.t.") == 1 and not fhe.has_slot_pt("impl.t.w0")
    # the encode cache: mult_cached / prime / stage / adopt / erase
    m1, m2 = rng.standard_normal(N), rng.standard_normal(N)
    st0 = fhe.enc_cache_stats()
    assert rel(dec(fhe, fhe.mult_cached(a, "mask.t1", m1)), x * m1) < TOL
    assert rel(dec(fhe, fhe.mult_cached(a, "mask.t1", m1)), x * m1) < TOL
    st1 = fhe.enc_cache_stats()
    assert st1["misses"] == st0["misses"] + 1 and st1["hits"] == st0["hits"] + 1
    fhe.prime_pt("mask.t2", m2, a.level)
    h = fhe.stage_pts([("mask.t3", a.level, m2), ("mask.t3", b.level, m2)])
    assert fhe.adopt_pts(h) == 2
    st2 = fhe.enc_cache_stats()
    assert rel(dec(fhe, fhe.mult_cached(b, "mask.t3", m2)), x * m2) < TOL
    assert rel(dec(fhe, fhe.add_cached(a, "mask.t2", m2)), x + m2) < TOL
    assert fhe.enc_cache_stats()["hits"] == st2["hits"] + 2
    many = fhe.mult_cached_many(a, ["mask.t1", "mask.t2"], [m1, m2])
    assert rel(dec(fhe, many[1]), x * m2) < TOL
    assert fhe.erase_pts(["mask.t1", "mask.t2", "mask.t3"]) == 4


def test_levels_and_limbs(fhe, rng):
    x, y = cvec(rng), cvec(rng)
    a = enc(fhe, x, 10)
    p = fhe.mult(a, enc(fhe, y, 10))
    fhe.realize(p)
    assert (p.level, p.deg) == (12, 1) and rel(dec(fhe, p), x * y) < TOL
    fhe.drop_to_level(p, 30)
    assert p.level == 30 and rel(dec(fhe, p), x * y) < TOL
    assert fhe.encode_like_level(p) == (30, 1)
    lim = fhe.export_limbs(a)
    assert lim["c0"].shape == (54 - 10, 2 * N) and lim["level"] == 43
    b = enc(fhe, np.zeros(N), 10)
    fhe.import_limbs(b, lim["c0"], lim["c1"], lim["noise_factor"], lim["noise_level"])
    assert rel(dec(fhe, b), x) < TOL
    fhe.tag_reduce(a, 32)                                  # metadata only
    assert (a.level, a.deg) == (10, 1)


def test_data_plane(fhe, rng):
    size = fhe.size
    t = N // size.hidDim
    feat = rng.standard_normal(size.dim)
    a = fhe.encode_token_input(feat)
    assert a.level == fhe.bootstrap_output_level()
    np.testing.assert_allclose(fhe.decode_token_output(a), feat, atol=1e-8)
    np.testing.assert_allclose(_pc.decode_token_output(fhe, a), feat, atol=1e-8)
    rows = rng.standard_normal((3, size.dim))
    p = _pc.pack_tokens(fhe, rows, 6)
    v = fhe.decrypt_slots(p)
    assert p.level == 6
    for tok in range(3):
        np.testing.assert_allclose(v[np.arange(size.dim) * t + tok], rows[tok], atol=1e-8)
    q = _pc.encode_prefill_input(fhe, rows)
    assert q.level == fhe.bootstrap_output_level()
    # values beyond the bottom-pair range decode from 4 rows
    big = 50.0 * cvec(rng)
    assert rel(dec(fhe, enc(fhe, big, 20)), big) < TOL


def test_refresh(fhe, rng):
    x = cvec(rng)
    a = enc(fhe, x, 40)
    n0 = fhe.total_bootstraps
    fhe.bootstrap(a)
    assert a.level == fhe.bootstrap_output_level() and rel(dec(fhe, a), x) < TOL
    b = enc(fhe, x, 40)
    fhe.bootstrap_hint(b, 44)                              # below the threshold: no refresh
    assert b.level == 40
    fhe.bootstrap_hint(b, 38)
    assert b.level < 40
    c = fhe.mult(enc(fhe, x, 44), enc(fhe, x, 44))          # reaches the limit: reactive refresh
    assert c.level < 44 and rel(dec(fhe, c), x * x) < TOL
    with fhe.suppress_auto_bts():
        d = fhe.mult(enc(fhe, x, 44), 2.0)
    assert d.level >= 44
    assert fhe.total_bootstraps == n0 + 3


def test_bootstrap_not_implemented_without_debug():
    cfg = _pc.Config.preset("n32_gpt2_decode")
    s = _pc.Session(cfg, _pc.KeySource.generate([1], seed=3), pool=8, pt_slots=4).complete_setup()
    f = s.fhe
    a = f.encrypt(np.ones(4), 40)
    with pytest.raises(NotImplementedError, match="no bootstrap routes"):
        f.bootstrap(a)
    with pytest.raises(NotImplementedError):
        f.bootstrap_hint(a, 10)
    f.bootstrap_hint(a, 44)                                # does not fire: no error


@pytest.mark.parametrize("level", [0, 20])
def test_chain_of_ten(fhe, rng, level):
    """mult, rotate, add, mult_pt, square, conjugate, mult_i, sub, rotate_and_sum, mult_scalar: 10 ops."""
    x, y = cvec(rng, 0.5), cvec(rng, 0.5)
    w = rng.standard_normal(N)
    a, b = enc(fhe, x, level), enc(fhe, y, level)
    r = fhe.mult(a, b); v = x * y
    r = fhe.rotate(r, 3); v = np.roll(v, -3)
    r = fhe.add(r, a); v = v + x
    r = fhe.mult_pt(r, w); v = v * w
    r = fhe.square(r); v = v * v
    r = fhe.conjugate(r); v = np.conj(v)
    r = fhe.mult_i(r); v = 1j * v
    r = fhe.sub(r, b); v = v - y
    r = fhe.rotate_and_sum(r, 1, 8); v = v + np.roll(v, -1); v = v + np.roll(v, -2); v = v + np.roll(v, -4)
    r = fhe.mult(r, 0.125); v = 0.125 * v
    assert rel(dec(fhe, r), v) < TOL


# ── Python -> device overhead ────────────────────────────────────────────────────────────────────────────────────

def _median_us(fhe, fn, chunks=200, per=16):
    """Median host time per call over chunks of `per` calls, the device drained between chunks (no chunk can fill
    the launch queue, so the time is the issue cost, not the device's: pc-gpu benches/ops.rs method)."""
    for _ in range(per):
        fn()
    ts = np.empty(chunks)
    for i in range(chunks):
        fhe.device_sync()
        t0 = time.perf_counter()
        for _ in range(per):
            fn()
        ts[i] = (time.perf_counter() - t0) / per
    fhe.device_sync()
    return float(np.median(ts) * 1e6)


def test_host_overhead(fhe, sess, rng):
    """Median host wall per call of a Python op on the device (issue cost: see _median_us). The ops without a
    rescale or key switch are gated at 20 us; the multi-launch ones are reported."""
    x, y, w = cvec(rng), cvec(rng), rng.standard_normal(N)
    a, b = enc(fhe, x, 20), enc(fhe, y, 20)
    fhe.set_slot_pt("impl.bench.w", w)
    fhe.mult_slot_pt(a, "impl.bench.w")
    fhe.mult_cached(a, "mask.bench", w)
    p = fhe.mult(a, b)
    acc = fhe.add(a, 0.0)
    rows = {
        "add": lambda: fhe.add(a, b),
        "sub": lambda: fhe.sub(a, b),
        "add_scalar": lambda: fhe.add(a, 0.5),
        "mult_scalar": lambda: fhe.mult(a, 0.5),
        "negate": lambda: fhe.negate(a),
        "mult_i": lambda: fhe.mult_i(a),
        "inplace_add": lambda: fhe.inplace_add(acc, b),
        "mult_slot_pt (cached)": lambda: fhe.mult_slot_pt(a, "impl.bench.w"),
        "mult_cached (hit)": lambda: fhe.mult_cached(a, "mask.bench", w),
        "rotate": lambda: fhe.rotate(a, 1),
        "mult (ct x ct)": lambda: fhe.mult(a, b),
        "add (deg-2 + deg-1: Rescale prelude)": lambda: fhe.add(p, a),
    }
    single = {"add", "sub", "add_scalar", "mult_scalar", "negate", "mult_i", "inplace_add", "mult_slot_pt (cached)",
              "mult_cached (hit)"}
    res = {}
    for name, fn in rows.items():
        res[name] = _median_us(fhe, fn, chunks=50 if "mult (" in name or "rotate" in name else 200)
    print("\n[m5d] host overhead, median us per call (level 20, K=1):")
    for name, us in res.items():
        print(f"[m5d]   {name:<40} {us:7.1f}{'  (gated)' if name in single else ''}")
    worst = max(res[k] for k in single)
    assert worst <= 20.0, res
