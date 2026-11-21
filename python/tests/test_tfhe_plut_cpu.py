"""Encrypted lookup tables on the TFHE host reference (_pc.TfhePrivateLut, pc-tfhe `pack`): pack an encrypted table
with the functional packing key switch, select an entry by an encrypted index with the blind rotation of that
accumulator, decrypt the right entry every time; the noise model's p_fail is below 2^-128."""
from __future__ import annotations

import time

import numpy as np
import pytest

_pc = pytest.importorskip("perseus._pc")


@pytest.fixture(scope="module")
def keys():
    return _pc.TfhePrivateLut(7)


def test_pack_phases(keys):
    q_in, q, nl, n = keys.moduli()
    rng = np.random.default_rng(1)
    msgs = rng.integers(0, 16, 64)
    lwes = [keys.encrypt(int(m) * (q_in // 32)) for m in msgs]
    pos = list(range(0, n, n // 64))
    t = time.time()
    acc = keys.pack(lwes, pos, [3] * 64)                   # the functional variant: weight 3
    dt = time.time() - t
    ph = np.array(keys.phase_glwe(acc), dtype=np.float64)
    want = 3 * msgs * (q_in // 32) * (q / q_in)
    sd = keys.noise(64, 1, 1.0, 3.0)[0]
    err = ph[pos] - want
    err -= q * np.round(err / q)
    print(f"\n[tfhe] pack 64 LWEs: {dt * 1e3:.0f} ms host; err std {err.std():.3e}, model {sd:.3e}")
    assert np.abs(err).max() < 6 * sd
    assert np.all(np.abs(np.delete(ph, pos)) < 6 * keys.noise(0, 0, 1.0)[0] + 1e12)


def test_private_lut(keys):
    """8 entries of 4 bits, 32 encrypted-index lookups: 0 wrong; p_fail (index and output decisions) < 2^-128."""
    q_in, q, nl, n = keys.moduli()
    rng = np.random.default_rng(2)
    t_size, span = 8, n // 8
    table = rng.integers(0, 16, t_size)
    enc = [keys.encrypt(int(v) * (q_in // 32)) for v in table]
    lwes = [enc[j // span] for j in range(n)]
    acc = keys.pack(lwes, list(range(n)))
    spacing = q / 32.0
    sa, sb, li, lo = keys.noise(1, span, spacing)
    assert max(li, lo) < -128.0
    wrong = 0
    for trial in range(32):
        i = int(rng.integers(0, t_size))
        idx = keys.encrypt((i * span + span // 2) * q_in // (2 * n))
        out = keys.extract(keys.blind_rotate(acc, idx))
        got = int(round(keys.phase_out(out) / spacing))
        wrong += got != table[i]
    print(f"\n[tfhe] private LUT: 32 lookups, wrong {wrong}; std acc {sa:.2e} rotation {sb:.2e}; "
          f"log2 p_fail index {li:.1f} output {lo:.1f}")
    assert wrong == 0


def test_table_place_mirror():
    """A native program's encrypted table (Prog.pack on table_place, Prog.plut): the mirror reads entry i at
    table_index(i, t), on both halves of the accumulator (the upper one holds -v), with k = 2 outputs reading the
    next coefficient."""
    from perseus.impl import tfhe
    for t in (4, 8, 16):
        tab = np.random.default_rng(t).integers(-8, 8, t) / 8.0
        pr = tfhe.Prog(f"plut.test{t}", 4)
        pr.lut("id", lambda m: m)
        pr.lut("out", lambda m: m, warp=True)
        pr.load("id", 0)
        place, w = tfhe.table_place(t)
        assert len(place) == tfhe.NAT_N and all(0 <= r < t for r, _ in place)
        pr.pack(0, t, place, acc=0, w=w)
        idx = np.arange(t)
        pr.lin(1, 0, 0, [(t + r, -1) for r in range(t)])
        pr.plut(1, t, 0, [2])
        pr.store(2, t, "out", "s")
        x = np.concatenate([tab, tfhe.table_index(idx, t)])
        y = pr.run(x, noise=0.0)
        assert np.array_equal(np.round(y * 8), np.round(tab * 8)), (t, y, tab)
        st = [s for s in __import__("json").loads(pr.json())["steps"] if s["op"] == "pack"][0]
        assert len(st["place"]) == 2 * tfhe.NAT_N and len(st["w"]) == tfhe.NAT_N
