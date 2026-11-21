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
