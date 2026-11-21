"""The native TFHE bitwise ops programs (impl/tfhe_bitwise.py `WordProg`) on the numpy mirror (impl/tfhe.py `Prog.run`, the
negacyclic tables with the native input noise): radix-4 add / compare at 8 / 16 / 32 bits with the fan-in 3 prefix
(whole-torus combine) and the radix-2 one, every word exact; the fan-in 3 programs take fewer PBS rounds."""
from __future__ import annotations

import numpy as np
import pytest

from perseus.impl import tfhe_bitwise


def _run(op, bits, fan, words=64, seed=0):
    rng = np.random.default_rng(seed)
    nd = bits // 2
    a = rng.integers(0, 1 << bits, words, dtype=np.int64)
    b = rng.integers(0, 1 << bits, words, dtype=np.int64)
    b[:words // 4] = a[:words // 4]                       # equal words: the all-propagate path
    b[words // 4:words // 2] = (~a[words // 4:words // 2]) & ((1 << bits) - 1)   # a + b = 2^bits - 1: the longest chain
    wp = tfhe_bitwise.WordProg("t", words, nd, fan=fan)
    getattr(wp, op)("o")
    out = np.asarray(wp.pr.run(np.concatenate([tfhe_bitwise.encode_digits(a, nd), tfhe_bitwise.encode_digits(b, nd)])))
    if op == "add":
        d = np.rint(out * 4).astype(np.int64).reshape(words, nd)
        got, want = (d * (4 ** np.arange(nd))).sum(1) & ((1 << bits) - 1), (a + b) & ((1 << bits) - 1)
    else:
        got, want = np.rint(out).astype(np.int64), (a < b).astype(np.int64)
    return int((got != want).sum()), sum(1 for st in wp.pr.steps if st["op"] == "pbs")


@pytest.mark.parametrize("bits", [8, 16, 32])
@pytest.mark.parametrize("op", ["add", "cmp"])
def test_bitwise_exact(op, bits):
    w3, r3 = _run(op, bits, 3)
    w2, r2 = _run(op, bits, 2)
    assert w3 == 0 and w2 == 0
    assert r3 <= r2
    if bits == 32:                                       # add: 1 + 3 rounds (+ the store), cmp: 1 + 2
        assert (r3, r2) == ((4, 5) if op == "add" else (3, 4))


_OPS = ["add", "sub", "neg", "addf", "subf", "cmp", "lt", "le", "gt", "ge", "eq", "ne", "slt", "sle", "sgt", "sge",
        "seq", "sne", "sext4", "shl0", "shl1", "shl2", "shl7", "shlB", "shr1", "shr3", "shrB", "sar1", "sar4", "sarM",
        "sarB", "rotl1", "rotl5", "rotr3", "rotrB", "shlv", "shrv", "sarv", "rotlv", "rotrv", "mul", "mulw", "mulc0",
        "mulc1", "mulc3", "mulc12345", "mulcM", "min", "max", "smin", "smax", "abs", "popcount", "mux", "xor"]


def _op(op, bits):
    """B: the width, M: width - 1 (shift amounts / constants at the edge)."""
    return op.replace("B", str(bits)).replace("M", str(bits - 1))


@pytest.mark.parametrize("bits", [8, 16, 32])
@pytest.mark.parametrize("op", _OPS)
def test_ops_exact_and_pfail(op, bits):
    """Every op on the mirror with the native input noise: exact on every word (edge pairs 0, -1, MIN, MAX, 1, -2;
    equal words; a + b = -1; shift amounts 0, 1, w - 1, w, w + 1, 2^w - 1, 2 w), and p_fail per lookup < 2^-128
    (2^20 chained lookups < 2^-100)."""
    from perseus.impl import tfhe_bitwise as T
    o = _op(op, bits)
    got, want, an = T.mirror(o, bits, 96, seed=bits)
    assert np.array_equal(got, want), (o, int((got != want).sum()))
    assert an["log2_pfail_lookup"] < -128.0 and an["log2_pfail_lookup"] + 20 < -100.0, (o, an)
    assert an["log2_pfail_op"] < -128.0, (o, an)


def test_buffers_and_rounds():
    """Compaction keeps programs small; the round counts of the main ops (32-bit) and the public shifts' zero-round
    cases (whole digits) hold."""
    from perseus.impl import tfhe_bitwise as T
    rounds = {o: T.analyze(T.build(o, "r", 4, 16).pr, T.load_values(T.slot_values(*T.inputs(4, 32, 0, o), 16)))["rounds"]
              for o in ("add", "sub", "subf", "slt", "shl4", "shl5", "sar4", "shlv", "mul", "min")}
    assert rounds == {"add": 4, "sub": 4, "subf": 5, "slt": 3, "shl4": 0, "shl5": 1, "sar4": 1, "shlv": 6,
                      "mul": 12, "min": 5}, rounds
    for o in ("sarv", "mulw", "abs"):
        assert T.build(o, "b", 16, 16).pr.bufs <= 9
