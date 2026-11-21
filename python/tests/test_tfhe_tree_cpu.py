"""Selection trees of private lookups (impl/tfhe.py `tree_steps`: `Prog.pack_rows` + `Prog.plut(each=True)`) on the
numpy mirror: tables of 2..4096 entries, every lookup returns its entry; the program's digits (`tree_index`)."""
from __future__ import annotations

import numpy as np
import pytest

from perseus.impl import tfhe


@pytest.mark.parametrize("t", [2, 8, 16, 64, 256, 4096])
def test_tree_lookup(t):
    rng = np.random.default_rng(t)
    b = 16
    table = rng.integers(-8, 8, t) / 8.0
    idx = rng.integers(0, t, b)
    L = len(tfhe.tree_digits(t))
    p = tfhe.Prog("tree", 4)
    p.lut("id", lambda m: m)
    p.lut("out", lambda m: m, warp=True)
    p.load("id", 0)
    p.lin(1, 0, 0, [(t + r, -1) for r in range(b * L)])
    res = tfhe.tree_steps(p, 0, t, 1, b, 2, 3, 0)
    p.store(res, b, "out", "o")
    m = np.concatenate([table, tfhe.tree_index(idx, t).reshape(-1)])
    out = np.asarray(p.run(m))
    assert np.array_equal(np.round(out * 8), np.round(table[idx] * 8))
    assert sum(1 for st in p.steps if st["op"] == "plut") == L


def test_tree_digits():
    assert tfhe.tree_digits(4096) == [8, 8, 8, 8]
    assert tfhe.tree_digits(1024) == [8, 8, 8, 2]
    with pytest.raises(ValueError):
        tfhe.tree_digits(12)
