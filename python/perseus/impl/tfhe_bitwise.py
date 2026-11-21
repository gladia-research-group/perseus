"""Native TFHE bitwise ops on radix-4 words (impl/tfhe.py native programs: one switch in, PBS chains on LWEs, one finish
out). A word of `nd` digits d (base 4, least significant first) is nd native LWEs at value d / 8; every lookup table has
<= 8 bins on the native [-1, 1) (16 for the bitwise ops), so a lookup's input noise (the mod switch to 2 n_br and the
key switch: sd ~6e-3 at nu 1) leaves p_fail ~2^-317 per lookup (8 bins; 16 bins need PC_TFHE_NU=2 for < 2^-128).

- add: statuses S_i = sign(a_i + b_i - 3) / 4 (kill / propagate / generate), Kogge-Stone rounds of fan-in 3,
  S_i = sign(4 S_i + 2 S_(i - d) + S_(i - 2 d)) for d = 1, 3, 9, ... (the last one outputs the carry), then
  (a_i + b_i + c_(i - 1)) mod 4 in the store's table: 1 + ceil(log3 nd) PBS rounds + the store (32 bits: 4);
- cmp (a < b): statuses sign(a_i - b_i), a tree of the same fan-in 3 combine from the top digit, the last combine
  (fan-in <= 2) in the store's table: 1 + ceil(log3 nd) - 1 rounds + the store (32 bits: 16 -> 6 -> 2 -> store, 3);
  fan-in 3 inputs span (-2, 2): the tables' negacyclic half carries the sign, so the bins are only cut at the zero
  band and the wrap (margin 1/8: the 8-bin p_fail); `fan=2` keeps the radix-2 prefix (1 + log2 nd rounds);
- mux (sel ? a : b): o = f(2 (a - b) / 8 + 2 sel) = -+(a - b) / 16 (the negacyclic half turn flips it by sel), out =
  a + b + 2 o = 2 (sel ? a : b) / 8: one PBS round;
- xor / and / or: one PBS per digit on a + 4 b.
`WordProg` lays words out on a slot set (a's digits, b's, then one sel per word) and appends the steps; pc-check
tfhe-bitwise runs the same programs on the device's native ops directly.
"""
from __future__ import annotations

import numpy as np

from . import tfhe


def _r4(m):
    return np.round(4.0 * np.asarray(m, dtype=np.float64))


def _sgn(m):
    return np.sign(_r4(m)) / 4.0


def _carry(m):
    return np.where(_r4(m) > 0, 0.125, 0.0)


def _carry_pm(m):              # the whole-torus carry: c / 8 - 1/16 (antisymmetric)
    return np.where(_r4(m) > 0, 0.0625, -0.0625)


def _digit_out(m):            # (s + c) mod 4 on bins -7/8 + k/4
    return np.mod(np.floor((np.asarray(m) + 1.0) * 4.0), 4.0) / 8.0


def _lt(m):
    return np.where(_r4(m) < 0, 0.125, 0.0)


def _mux(m):
    return -_r4(m) / 16.0


def _bitwise(op):
    f = {"xor": np.bitwise_xor, "and": np.bitwise_and, "or": np.bitwise_or}[op]

    def g(m):
        k = np.clip(np.floor((np.asarray(m) + 1.0) * 8.0), 0, 15).astype(np.int64)
        return f(k % 4, k // 4) / 8.0
    return g


def _load(m):                  # slot value (2 d - 3) / 4 -> native d / 8 (snapped: half a bin is 1/4)
    return np.clip(np.round((4.0 * np.asarray(m) + 3.0) / 2.0), 0, 3) / 8.0


def encode_digits(x, nd):
    """Words -> the slot values a load reads: digit d at (2 d - 3) / 4."""
    x = np.asarray(x, dtype=np.int64)
    d = (x[:, None] >> (2 * np.arange(nd))[None, :]) & 3
    return ((2.0 * d - 3.0) / 4.0).reshape(-1)


class WordProg:
    """A native program over `words` words of `nd` radix-4 digits (`bits` = 2 nd): the loaded slot set holds a's
    digits ([word][digit]), then b's, then (mux) one sel per word at (2 sel - 1) / 4 ... read as a digit 0 / 3 -> sel
    0 / 1 is taken from the digit's top bit. Outputs: digits at value d / 4 (cmp: one value per word, 1 when a < b)."""

    def __init__(self, name, words, nd, fan=3):
        self.w, self.nd, self.fan = int(words), int(nd), int(fan)
        if self.fan not in (2, 3):
            raise ValueError("tfhe_bitwise: prefix fan-in 2 or 3")
        if self.nd & (self.nd - 1):
            raise ValueError("tfhe_bitwise: digits per word must be a power of two")
        self.R = self.w * self.nd
        self.pr = tfhe.Prog(name, 5)
        p = self.pr
        p.lut("load", _load)
        for k, f in (("sgn", _sgn), ("carry", _carry), ("carry_pm", _carry_pm), ("lt", _lt), ("mux", _mux)):
            p.lut(k, f)
        p.lut("out_w", lambda m: 2.0 * _digit_out(m), warp=True)      # d / 4 for the finish
        p.lut("lt_w", lambda m: 8.0 * _lt(m), warp=True)               # 1 when a < b
        p.lut("lt_pm", lambda m: np.where(_r4(m) < 0, 0.0625, -0.0625))
        p.lut("lt_id_w", lambda m: np.where(np.asarray(m) > 0.0625, 1.0, 0.0), warp=True)   # lt / 8 -> lt
        p.lut("dig_w", lambda m: np.clip(np.round(8.0 * np.asarray(m)), 0, 3) / 4.0, warp=True)    # d / 8 in
        p.lut("dig4_w", lambda m: np.clip(np.round(4.0 * np.asarray(m)), 0, 3) / 4.0, warp=True)   # d / 4 in
        p.load("load", 0)

    def rows_in(self, sel=False):
        return 2 * self.R + (self.w if sel else 0)

    def add(self, out_slots):
        if self.fan == 3:
            return self._add3(out_slots)
        p, R, nd = self.pr, self.R, self.nd
        p.lin(1, 0, 0, [(r, R + r) for r in range(R)], ca=2, cb=2, c=-0.75)
        p.pbs(1, R, "sgn", [2])
        cur, nxt, d = 2, 3, 1
        while d < nd:
            last = 2 * d >= nd
            p.lin(4, cur, cur, [(r, r - d if r % nd >= d else -1) for r in range(R)], ca=2, cb=1)
            p.pbs(4, R, "carry" if last else "sgn", [nxt])
            cur, nxt, d = nxt, cur, 2 * d
        p.lin(4, 1, cur, [(r, r - 1 if r % nd else -1) for r in range(R)], ca=1, cb=2, c=-0.125)
        p.store(4, R, "out_w", out_slots)

    def _add3(self, out_slots):
        p, R, nd = self.pr, self.R, self.nd
        p.lin(1, 0, 0, [(r, R + r) for r in range(R)], ca=2, cb=2, c=-0.75)
        p.pbs(1, R, "sgn", [2])
        cur, nxt, d = 2, 3, 1
        while d < nd:
            f = min(3, -(-nd // d))
            last = d * f >= nd
            p.lin(4, cur, cur, [(r, r - d if r % nd >= d else -1) for r in range(R)], ca=1 << (f - 1), cb=1 << (f - 2))
            if f == 3:          # in place: row r of 4 reads only row r of 4
                p.lin(4, 4, cur, [(r, r - 2 * d if r % nd >= 2 * d else -1) for r in range(R)], ca=1, cb=1)
            p.pbs(4, R, "carry_pm" if last else "sgn", [nxt])
            cur, nxt, d = nxt, cur, d * f
        # c = g + 1/16 in {0, 1/8} (digit 0: no carry in), then (s + c) mod 4 in the store's table
        p.lin(nxt, cur, cur, [(r, -1) for r in range(R)], ca=1, cb=0, c=0.0625)
        p.lin(4, 1, nxt, [(r, r - 1 if r % nd else -1) for r in range(R)], ca=1, cb=2, c=-0.125)
        p.store(4, R, "out_w", out_slots)

    def _cmp3(self, out_slots):
        p, R, w = self.pr, self.R, self.w
        p.lin(1, 0, 0, [(r, R + r) for r in range(R)], ca=2, cb=-2)
        p.pbs(1, R, "sgn", [2])
        cur, nxt, k = 2, 3, self.nd
        while True:
            f = min(3, k)
            h = -(-k // f)
            src = [[(q // h) * k + (q % h) * f + f - 1 - x if (q % h) * f + f - 1 - x < k else -1 for x in range(f)]
                   for q in range(w * h)]
            p.lin(4, cur, cur, [(v[0], v[1] if f > 1 else -1) for v in src], ca=1 << (f - 1), cb=1 << max(f - 2, 0))
            if f == 3:
                p.lin(4, 4, cur, [(q, v[2]) for q, v in enumerate(src)], ca=1, cb=1)
            if h == 1 and f <= 2:          # the last combine (inside [-1, 1)) rides the store
                p.store(4, w, "lt_w", out_slots)
                return
            if h == 1:                     # a fan-in 3 last combine: the whole-torus lt, then the store
                p.pbs(4, w, "lt_pm", [nxt])
                p.lin(4, nxt, nxt, [(q, -1) for q in range(w)], ca=1, cb=0, c=0.0625)
                p.store(4, w, "lt_id_w", out_slots)
                return
            p.pbs(4, w * h, "sgn", [nxt])
            cur, nxt, k = nxt, cur, h

    def cmp(self, out_slots):
        if self.fan == 3:
            return self._cmp3(out_slots)
        p, R, nd, w = self.pr, self.R, self.nd, self.w
        p.lin(1, 0, 0, [(r, R + r) for r in range(R)], ca=2, cb=-2)
        p.pbs(1, R, "sgn", [2])
        cur, nxt, k = 2, 3, nd
        while k > 1:
            h = k // 2
            p.lin(4, cur, cur, [(q // h * k + 2 * (q % h) + 1, q // h * k + 2 * (q % h)) for q in range(w * h)], ca=2, cb=1)
            if h == 1:
                p.store(4, w, "lt_w", out_slots)
                return
            p.pbs(4, w * h, "sgn", [nxt])
            cur, nxt, k = nxt, cur, h

    def mux(self, out_slots):
        p, R, nd = self.pr, self.R, self.nd
        p.lin(1, 0, 0, [(r, R + r) for r in range(R)], ca=2, cb=-2)
        # sel loaded as a digit 0 / 3 (d / 8): 16 x (3 / 8) = 6 = 2 (mod 4): the half turn; 0: none
        p.lin(4, 1, 0, [(r, 2 * R + r // nd) for r in range(R)], ca=1, cb=16)
        p.pbs(4, R, "mux", [2])
        p.lin(3, 0, 0, [(r, R + r) for r in range(R)], ca=1, cb=1)
        p.lin(4, 3, 2, [(r, r) for r in range(R)], ca=1, cb=2)
        p.store(4, R, "dig4_w", out_slots)

    def bitwise(self, op, out_slots):
        p, R = self.pr, self.R
        p.lut(op, lambda m: 2.0 * _bitwise(op)(m), warp=True)
        p.lin(1, 0, 0, [(r, R + r) for r in range(R)], ca=1, cb=4, c=-15.0 / 16.0)
        p.store(1, R, op, out_slots)

    def ident(self, out_slots):
        """The I/O baseline: load a, store it."""
        self.pr.store(0, self.R, "dig_w", out_slots)
