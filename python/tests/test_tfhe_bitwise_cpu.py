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
        "sarB", "rotl1", "rotl5", "rotr3", "rotrB", "shlv", "shrv", "sarv", "rotlv", "rotrv", "rotlvb", "rotrvb", "mul", "mulw", "mulb",
        "mulc0", "mulc1", "mulc3", "mulc12345", "mulcM", "min", "max", "smin", "smax", "abs", "popcount", "mux", "xor",
        "and", "or", "andn", "orn", "xnor", "clz", "ctz", "log2", "div", "rem", "divrem", "sdiv", "srem", "sdivrem",
        "divc7", "divc10", "remc7", "divremc12345", "divc0", "remc0", "divc16", "divc2654435769", "remc2654435769", "mulh", "mulhsu", "mulhu", "clmul",
        "clmulh", "clmulr", "addi12345", "addiM", "andi12345", "ori77", "xori99", "lti12345", "slti3", "eqi12345",
        "eqi0", "bset5", "bclr0", "binv3", "bextM", "bsetv", "bclrv", "binvv", "bextv", "shadd1", "shadd2", "shadd3",
        "orcb", "rev8", "zext16", "czeqz", "cznez"]
# the 64-bit set (the 32-bit-in-64 word forms only there)
_OPS64 = ["add", "sub", "subf", "cmp", "slt", "eq", "xor", "mux", "min", "smax", "abs", "popcount", "clz", "ctz", "log2",
          "shlv", "rotlv", "rotrv", "mul", "mulh", "divc7", "remc10", "divc11400714819323198485", "remc11400714819323198485", "addi12345", "lti12345", "eqi12345", "bsetv", "bextv",
          "shadd1", "shadduw2", "adduw", "orcb", "rev8", "czeqz", "clmulr", "w32add", "w32sub", "w32shl5", "w32sar7",
          "w32shrv", "w32mul", "w32div", "w32srem"]


def _op(op, bits):
    """B: the width, M: width - 1 (shift amounts / constants at the edge)."""
    return op.replace("B", str(bits)).replace("M", str(bits - 1))


@pytest.mark.parametrize("bits", [8, 16, 32])
@pytest.mark.parametrize("op", _OPS)
def test_ops_exact_and_pfail(op, bits):
    _check(op, bits)


@pytest.mark.parametrize("op", _OPS64)
def test_ops64_exact_and_pfail(op):
    _check(op, 64)


@pytest.mark.parametrize("op", ["div", "srem"])
def test_div64(op):
    """The 64-bit division (249 rounds), fewer words."""
    _check(op, 64, words=40)


def _check(op, bits, words=96):
    """Every op on the mirror with the native input noise: exact on every word (edge pairs 0, -1, MIN, MAX, 1, -2;
    equal words; a + b = -1; shift amounts 0, 1, w - 1, w, w + 1, 2^w - 1, 2 w), and p_fail per lookup < 2^-128
    (2^20 chained lookups < 2^-100)."""
    from perseus.impl import tfhe_bitwise as T
    o = _op(op, bits)
    got, want, an = T.mirror(o, bits, words, seed=bits)
    assert np.array_equal(got, want), (o, int((got != want).sum()))
    assert an["log2_pfail_lookup"] < -128.0 and an["log2_pfail_lookup"] + 20 < -100.0, (o, an)
    assert an["log2_pfail_op"] < -118.0, (o, an)        # the union over every lookup of the op


def test_buffers_and_rounds():
    """Compaction keeps programs small; the round counts of the main ops (32-bit; the result complete in the
    program, the store only reformats) and the public shifts' zero-round cases (whole digits) hold."""
    from perseus.impl import tfhe_bitwise as T
    rounds = {o: T.analyze(T.build(o, "r", 4, 16).pr, T.load_values(T.slot_values(*T.inputs(4, 32, 0, o), 16)))["rounds"]
              for o in ("add", "sub", "subf", "slt", "eq", "xor", "shl4", "shl5", "sar4", "shlv", "mul", "min", "clz",
                        "popcount", "addi12345", "lti12345")}
    assert rounds == {"add": 3, "sub": 3, "subf": 4, "slt": 4, "eq": 2, "xor": 1, "shl4": 0, "shl5": 1, "sar4": 1,
                      "shlv": 6, "mul": 7, "min": 5, "clz": 2, "popcount": 4, "addi12345": 3, "lti12345": 3}, rounds
    r64 = {o: T.analyze(T.build(o, "r", 2, 32).pr, T.load_values(T.slot_values(*T.inputs(2, 64, 0, o), 32)))["rounds"]
           for o in ("add", "cmp", "eq", "xor")}
    assert r64 == {"add": 4, "cmp": 4, "eq": 3, "xor": 1}, r64
    for o in ("sarv", "mulw", "abs", "div", "sdiv"):
        assert T.build(o, "b", 16, 16).pr.bufs <= 16


def test_one_launch_per_round():
    """The products and every carry-save round are one bootstrap launch (a table per row where they differ); the
    division's rounds too, but for its first two (b << 1 and b's thermometer: two launches each)."""
    from perseus.impl import tfhe_bitwise as T
    for o, extra in (("mul", 0), ("popcount", 0), ("divrem", 2)):
        pr = T.build(o, "m", 2, 8).pr
        an = T.analyze(pr, T.load_values(T.slot_values(*T.inputs(2, 16, 0, o), 8)))
        assert an["launches"] == an["rounds"] + 1 + extra, (o, an)          # + the store


def test_prefix_fans():
    """The prefix's combine fan-ins: the fewest levels (fan-in 4 at later levels where that saves one: 16 and 64
    digits), the old fan-in 3 plan otherwise; and the adder exact on the mirror with either."""
    from perseus.impl import tfhe_bitwise as T
    assert [T._prefix_fans(n) for n in (2, 4, 8, 16, 32, 64)] == [[2], [4], [3, 3], [4, 4], [4, 3, 3], [4, 4, 4]]
    assert T._prefix_fans(16, True, False) == [3, 3, 2]
    rng = np.random.default_rng(3)
    a = rng.integers(0, 1 << 32, 32, dtype=np.int64)
    b = (~a) & ((1 << 32) - 1)                        # a + b = 2^32 - 1: the longest chain
    b[:8] = rng.integers(0, 1 << 32, 8)
    for later in (True, False):
        wp = T.WordProg("p", 32, 16)
        wp.fan4all = later
        wp._adder("add", "o")
        out = np.asarray(wp.pr.run(np.concatenate([T.encode_digits(a, 16), T.encode_digits(b, 16)])))
        d = np.rint(out * 4).astype(np.int64).reshape(32, 16)
        assert np.array_equal((d * (4 ** np.arange(16))).sum(1) & ((1 << 32) - 1), (a + b) & ((1 << 32) - 1))
