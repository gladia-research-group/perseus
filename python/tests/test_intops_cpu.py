"""impl/intops.py on the CPU: chained-lookup integer arithmetic on the numpy mirror (exact), the tables' failure
model per lookup and over long chains, chaining slack per EvalMod table."""
from __future__ import annotations

import numpy as np
import pytest

from perseus.impl import fbs, intops
from perseus.impl.ops import NumpyOps

_pc = pytest.importorskip("perseus._pc")


@pytest.mark.parametrize("r", [2, 4])
def test_mirror_exact(r):
    rng = np.random.default_rng(r)
    io = intops.IntOps(NumpyOps(), r)
    for bits in (16, 32):
        m = (1 << bits) - 1
        a = rng.integers(0, 1 << bits, 4096, dtype=np.int64)
        b = rng.integers(0, 1 << bits, 4096, dtype=np.int64)
        b[:16] = a[:16]
        da, db = intops.digits(a, bits, r), intops.digits(b, bits, r)
        assert np.array_equal(intops.value(io.add(da, db), r), (a + b) & m)
        assert np.array_equal(np.round(io.cmp(da, db)).astype(np.int64), (a < b).astype(np.int64))
        s = rng.integers(0, 2, 4096).astype(float)
        assert np.array_equal(intops.value(io.mux(s, da, db), r), np.where(s == 1, a, b))
        for op, f in (("and", np.bitwise_and), ("or", np.bitwise_or), ("xor", np.bitwise_xor)):
            assert np.array_equal(intops.value(io.bitwise(op, da, db), r), f(a, b))


def test_lut_centering():
    name = fbs.register_lut("cpu.sq", lambda j: (j * j) % 7, 3, 17)
    j = np.arange(3, 18).astype(float)
    assert fbs.FUNCS[name].center == 3 + 8          # t = 16
    assert np.abs(fbs.FUNCS[name].f(j - fbs.FUNCS[name].center) - (j * j) % 7).max() < 1e-9   # the twin: centred


@pytest.mark.parametrize("table", fbs.TABLES)
def test_pfail_and_chains(table):
    """Every integer lookup table meets 2^-128 per lookup (per ciphertext); 2^20 chained lookups stay below 2^-100."""
    io = intops.IntOps(NumpyOps(), 2, table)
    io.bitwise("xor", [], [])
    for n in io.lookups():
        r = fbs.p_fail(n)
        assert r["ok"] and r["log2_ct"] < -128.0, (n, r)
    assert intops.chain_p_fail(io.lookups(), [2 ** 20] * len(io.lookups())) < -100.0


def test_chain_slack():
    """The degree-59 / 3-double-angle table lands two composite levels shallower than the K32 one."""
    k = fbs.register_lut("cpu.c16.k32", lambda j: j % 4, 0, 15, table="k32")
    d = fbs.register_lut("cpu.c16.k32d3", lambda j: j % 4, 0, 15, table="k32d3")
    assert fbs.chain_slack(d) == fbs.chain_slack(k) + 4
    assert fbs.chain_slack(d) >= 0


def test_sort_fbs_mirror():
    """sort.step's FBS backend on the numpy mirror: rank / top-k exact; the rule refuses gaps below degree 255's."""
    from perseus.examples.mathlib import tasks as T
    from perseus.impl import sort
    for kind in ("rank", "topk"):
        t = T.make("sort", kind=kind, method="fbs", n=16)
        assert t.metrics(t.mirror())["exact"] == 1.0
    assert sort.step_method(4096, 2.0, 0.1) == "fbs"
    assert sort.step_method(16, 2.0, 0.1) == "tfhe"
    assert sort.step_method(4096, 2.0, 0.01) == "tfhe"
