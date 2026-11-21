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


_WORD_OPS = ["add", "sub", "neg", "addf", "subf", "cmp", "lt", "le", "gt", "ge", "eq", "ne", "slt", "sle", "sgt",
             "sge", "seq", "sne", "sext4", "shl0", "shl1", "shl2", "shl3", "shlB", "shr1", "shr5", "shrB", "sar1",
             "sar4", "sar7", "sarM", "sarB", "rotl1", "rotl3", "rotr5", "rotrB", "shlv", "shrv", "sarv", "rotlv",
             "rotrv", "mul", "mulc0", "mulc1", "mulc7", "mulc12345", "mulcM", "min", "max", "smin", "smax", "abs",
             "popcount", "mux", "xor"]


@pytest.mark.parametrize("bits", [8, 16, 32])
@pytest.mark.parametrize("op", _WORD_OPS)
def test_word_ops_exact(op, bits):
    """WordOps (a word's digits in one ciphertext) on the mirror, every lane exact: edge pairs (0, -1, MIN, MAX, 1,
    -2), equal words, a + b = -1, shift amounts 0, 1, w - 1, w, w + 1, 2^w - 1, 2 w; flags / predicates at the top
    digit slot. Every table meets 2^-128 per ciphertext at its input noise, 2^20 chained lookups 2^-100."""
    import warnings
    o = op.replace("B", str(bits)).replace("M", str(bits - 1))
    n = bits // 2
    a, b, sel = intops.inputs(4096, bits, bits, o)
    w = intops.WordOps(NumpyOps(), bits, 2, "k28d1", 4096 * n)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)          # a registration over 2^-128 warns
        outs = intops.word_op(w, o, w.encode(a), w.encode(b), np.repeat(sel, n).astype(float))
    wrong, dev = intops.word_check(w, o, outs, a, b, sel, bits)
    assert wrong == 0 and dev < 1e-9, (o, wrong, dev)
    names = w.lookups()
    sg = w.lookup_sigmas()
    for nm in names:
        assert fbs.p_fail(nm, sigma_in=sg[nm])["log2_ct"] < -128.0, nm
    assert intops.chain_p_fail(names, [2 ** 20] * len(names)) < -100.0


def test_word_op_lookups():
    """Sequential lookups of the main 32-bit ops (the latency's unit on one ciphertext)."""
    calls = []
    orig = fbs.fbs

    def counting(ops, x, name):
        calls.append(name)
        return orig(ops, x, name)
    fbs.fbs = counting
    try:
        got = {}
        for o in ("add", "sub", "slt", "shl3", "shl2", "sar6", "shlv", "sarv", "mul", "mulc12345", "popcount"):
            calls.clear()
            w = intops.WordOps(NumpyOps(), 32, 2, "k28d1", 1024)
            a, b, sel = intops.inputs(64, 32, 0, o)
            intops.word_op(w, o, w.encode(a), w.encode(b), np.repeat(sel, 16).astype(float))
            got[o] = len(calls)
    finally:
        fbs.fbs = orig
    assert got == {"add": 4, "sub": 4, "slt": 5, "shl3": 1, "shl2": 0, "sar6": 1, "shlv": 6, "sarv": 7, "mul": 10,
                   "mulc12345": 9, "popcount": 4}, got


@pytest.mark.parametrize("bits", [8, 16, 32])
@pytest.mark.parametrize("op", _WORD_OPS)
def test_int_ops_exact(op, bits):
    """IntOps (a digit per ciphertext, ripple carries) on the mirror: the same ops and edge cases as WordOps."""
    import warnings
    o = op.replace("B", str(bits)).replace("M", str(bits - 1))
    a, b, sel = intops.inputs(4096, bits, bits + 1, o)
    io = intops.IntOps(NumpyOps(), 2, "k28d1")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        outs = intops.int_op(io, o, intops.digits(a, bits, 2), intops.digits(b, bits, 2), sel.astype(float))
    wrong, dev = intops.int_check(io, o, outs, a, b, sel, bits)
    assert wrong == 0 and dev < 1e-9, (o, wrong, dev)
    sg = io.lookup_sigmas()
    for nm in io.lookups():
        assert fbs.p_fail(nm, sigma_in=sg[nm])["log2_ct"] < -128.0, nm
