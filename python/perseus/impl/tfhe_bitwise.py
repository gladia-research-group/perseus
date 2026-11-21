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

The other integer / bitwise ops (fan-in 3 prefix; every table <= 8 bins on [-1, 1) or a negacyclic function of the whole
torus, so every lookup keeps the 8-bin margin 1/8 at nu 1):
- sub / neg: a - b = ~(~a + b) and -a = ~(a + 11..1) (~d = 3 - d is linear): the adder with complemented operands, the
  output complemented in the store's table; no carry-in. `flags=True`: the digits come out linear
  (a_i + b_i + c_(i - 1) - 4 c_i) and every word gains two rows, the carry / borrow (the prefix's top carry, free) and
  the signed overflow: one lookup on t = a_top + b_top + c_in (t / 4 - 7 / 8) turned by the half turn 2 (sa + sb)
  (the sign bits as half turns from one lookup on the top digits, beside the statuses): ovf = [2 <= t <= 5] without
  a turn, never with one (equal signs put t in [0, 3] or [4, 7], unequal ones in [2, 5]).
- cmp(pred, signed): statuses sign(a_i - b_i), the fan-in 3 tree, the last combine read by the predicate's table
  (lt / gt / le / ge / eq / ne); signed: the top group's first combine is turned by 2 (sa + sb) (a half turn when the
  signs differ: the top digit's status, never zero then, flips), no extra round.
- sext / public shifts and rotates: whole digits are a renumbering (lins); one bit is one round of the two-region bits
  table (d & 1 on 2 d / 8 + 1 / 16, d >> 1 on -2 d / 8 + 1 / 16: both functions of every digit in one batch), out =
  2 lo_i + hi_(i - 1) (left) / hi_i + 2 lo_(i + 1) (right; arithmetic: the sign 3 hi_top fills).
- shifts and rotates by an encrypted amount (a word in b's place): a barrel shifter, one stage per amount bit k
  (2^k bits: bit 0 from the bits table's lo / hi, then 1, 2, 4, ... digits renumbered); a stage is the AND mux
  and(sel, x_s) + and(~sel, x) in one batch (the and table on 2 x / 8 + sel - 7 / 8: x in the upper half iff sel = 1,
  so the output stays at d / 8 and stages chain); amounts >= the width (shifts) by an OR tree over the high amount bits
  folded into the last stage's selectors (sar: the sign fill as a third group).
- mul: b's bits (bits table), a << 1 (lo / hi), partial products and(bit_j, a or a << 1) at digit i + j // 2, a
  carry-save tree over the columns (groups of items summing to <= 7: one carry lookup, the digit s - 4 c linear),
  then the prefix adder on the last two rows; `wide=True` keeps all 2 nd digits. mulc (a public constant): the same
  tree on shifted copies (no and lookups). popcount: the digits' bit counts through the same tree.
- min / max: the compare's selector (1 / 0) and the AND mux; abs: the sign and the AND mux of -a and a.
`analyze` gives every program's rounds, PBS count and p_fail per lookup from the noise model (`LOOKUP_SD`,
`PBS_OUT_SD`); `case` emits a program with encrypted test inputs for pc-check tfhe-prog.
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
    digits ([word][digit]), then b's, then (mux) one sel per word at (4 sel - 3) / 4, read as a digit 0 / 2. Outputs:
    digits at value d / 4 (cmp: one value per word, 1 when a < b)."""

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
                p.lut("lt_pm8", TABLES["lt_pm8"][0])
                p.lut("id8_w", TABLES["id8_w"][0], warp=True)
                p.pbs(4, w, "lt_pm8", [nxt])
                p.lin(4, nxt, nxt, [(q, -1) for q in range(w)], ca=1, cb=0, c=0.125)
                p.store(4, w, "id8_w", out_slots)
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
        # sel loaded as a digit 0 / 2 (d / 8): 8 x (2 / 8) = 2: the half turn; 0: none (x 8, not the x 16 a digit 3
        # needs: the loaded sel's noise reaches the lookup x 8, 2^-215 per lookup instead of 2^-115)
        p.lin(4, 1, 0, [(r, 2 * R + r // nd) for r in range(R)], ca=1, cb=8)
        p.pbs(4, R, "mux", [2])
        p.lin(3, 0, 0, [(r, R + r) for r in range(R)], ca=1, cb=1)
        p.lin(4, 3, 2, [(r, r) for r in range(R)], ca=1, cb=2)
        p.store(4, R, "dig4_w", out_slots)

    def bitwise(self, op, out_slots):
        if tfhe.GRID == 2048:          # nu 1: the 16-bin table is 2^-82 per lookup; two rounds of <= 8-bin tables
            return self._bitwise2(op, out_slots)
        p, R = self.pr, self.R
        p.lut(op, lambda m: 2.0 * _bitwise(op)(m), warp=True)
        p.lin(1, 0, 0, [(r, R + r) for r in range(R)], ca=1, cb=4, c=-15.0 / 16.0)
        p.store(1, R, op, out_slots)

    def _bitwise2(self, op, out_slots):
        """and / or / xor at nu 1: the bits of a and b (one two-region batch), then one table on every bit pair's sum
        (lo and hi planes in one batch): 2 rounds, every lookup 8-bin."""
        R = self.R
        LH = self._lh(list(range(2 * R)))                 # lo of (a | b), then hi of (a | b): 4 R rows
        IN = self._lin(LH, LH, [(r, R + r) for r in range(R)] + [(2 * R + r, 3 * R + r) for r in range(R)], ca=2, cb=2)
        Y = self._pbs(IN, 2 * R, "bw_" + op)
        self._dig((Y, Y), [(r, R + r) for r in range(R)], ca=2, cb=4, out_slots=out_slots)

    def ident(self, out_slots):
        """The I/O baseline: load a, store it."""
        self.pr.store(0, self.R, "dig_w", out_slots)

    # ── the other integer / bitwise ops (module doc) ─────────────────────────────────────────────────────────────

    def _buf(self):
        """A fresh buffer (buffer 0 holds the load; the program's buffer count follows)."""
        if not hasattr(self, "_nb"):
            self._nb = 5
        b = self._nb
        self._nb += 1
        self.pr.bufs = max(self.pr.bufs, self._nb)
        return b

    def _tab(self, name):
        if name not in self.pr.luts:
            f, warp = TABLES[name]
            self.pr.lut(name, f, warp=warp)
        return name

    def _pbs(self, src, rows, lut, dst=None):
        dst = self._buf() if dst is None else dst
        self.pr.pbs(src, rows, self._tab(lut), [dst])
        return dst

    def _lin(self, a, b, pairs, ca=1, cb=1, c=0.0, dst=None):
        dst = self._buf() if dst is None else dst
        self.pr.lin(dst, a, b, pairs, ca=ca, cb=cb, c=c)
        return dst

    def _store(self, src, rows, lut, out_slots):
        self.pr.store(src, rows, self._tab(lut) if lut else None, out_slots)
        self.out_rows = rows

    def _row(self, op, w, i):
        """Row of digit i of word w of operand op (0: a, 1: b) in the loaded buffer."""
        return op * self.R + w * self.nd + i

    def _top(self, op):
        return [self._row(op, w, self.nd - 1) for w in range(self.w)]

    def _carries(self, X):
        """The fan-in 3 prefix on X = (s_i - 3) / 4 (R rows, s_i = x_i + y_i in 0..6): carries out of every digit,
        c_i in {0, 1/8}. 1 + ceil(log3 nd) rounds."""
        p, R, nd = self.pr, self.R, self.nd
        S, T, U = self._buf(), self._buf(), self._buf()
        self._pbs(X, R, "sgn", S)
        cur, nxt, d = S, T, 1
        while d < nd:
            f = min(3, -(-nd // d))
            last = d * f >= nd
            p.lin(U, cur, cur, [(r, r - d if r % nd >= d else -1) for r in range(R)], ca=1 << (f - 1), cb=1 << (f - 2))
            if f == 3:
                p.lin(U, U, cur, [(r, r - 2 * d if r % nd >= 2 * d else -1) for r in range(R)], ca=1, cb=1)
            self._pbs(U, R, "carry_pm" if last else "sgn", nxt)
            cur, nxt, d = nxt, cur, d * f
        p.lin(nxt, cur, cur, [(r, -1) for r in range(R)], ca=1, cb=0, c=0.0625)
        return nxt

    def _signs(self, extra=None):
        """The half turns 2 sa + 2 sb of every word (w rows; the sign bits as one lookup on the top digits, w + w
        rows); `extra` adds 2 (a half turn) to all."""
        w = self.w
        HI = self._lin(0, 0, [(r, -1) for r in self._top(0) + self._top(1)], ca=2, cb=0)
        H = self._pbs(HI, 2 * w, "hs")
        return self._lin(H, H, [(q, w + q) for q in range(w)], ca=1, cb=1, c=2.0 if extra else 0.0)

    def _adder(self, kind, out_slots, flags=False, X=None, xy=None):
        """add / sub / neg (module doc). `X`, `xy`: an adder input built elsewhere ((s - 3) / 4 and the two operand
        sources (buffer, pairs) of its linear output; mul's final add)."""
        p, R, nd, w = self.pr, self.R, self.nd, self.w
        if X is None:
            ab = [(r, R + r) for r in range(R)]
            if kind == "add":
                X = self._lin(0, 0, ab, ca=2, cb=2, c=-0.75)
            elif kind == "sub":                 # ~a + b
                X = self._lin(0, 0, ab, ca=-2, cb=2)
            else:                               # a + 11..1
                X = self._lin(0, 0, [(r, -1) for r in range(R)], ca=2, cb=0)
        H = self._signs(extra=kind == "sub") if flags else None
        C = self._carries(X)
        prev = [(r, r - 1 if r % nd else -1) for r in range(R)]
        if not flags:
            Z = self._lin(X, C, prev, ca=1, cb=2, c=-0.125)
            self._store(Z, R, "outn_w" if kind in ("sub", "neg") else "out_w", out_slots)
            return
        # linear digits at d / 4: 2 s / 8 + 2 c_(i - 1) - 8 c_i (X + 3/4 = 2 s / 8), complemented for sub / neg
        O = self._lin(X, C, prev, ca=1, cb=2, c=0.75)
        self._lin(O, C, [(r, r) for r in range(R)], ca=1, cb=-8, dst=O)
        if kind in ("sub", "neg"):
            self._lin(O, O, [(r, -1) for r in range(R)], ca=-1, cb=0, c=0.75, dst=O)
        tops = [r for r in range(R) if r % nd == nd - 1]
        V = self._lin(X, C, [(r, r - 1) for r in tops], ca=1, cb=2, c=-0.125)
        self._lin(V, H, [(q, q) for q in range(w)], ca=1, cb=1, dst=V)
        F = self._pbs(V, w, "ovf_pm")
        # rows: digits | carry (borrow for sub) | overflow, all at d / 4; the overflow rows 2 F + 1/8 ({0, 1/4}):
        # a per-row constant from a noiseless row (a lin of no terms: trivial LWEs of c)
        K = self._lin(0, 0, [(-1, -1)], ca=0, cb=0, c=0.125)
        G = self._lin(C, F, [(t, -1) for t in tops] + [(-1, q) for q in range(w)], ca=2, cb=2)
        self._lin(G, K, [(q, -1) for q in range(w)] + [(w + q, 0) for q in range(w)], ca=1, cb=1, dst=G)
        OUT = self._lin(O, G, [(r, -1) for r in range(R)] + [(-1, q) for q in range(2 * w)], ca=1, cb=1)
        self._store(OUT, R + 2 * w, "dig4_w", out_slots)

    def sub(self, out_slots, flags=False):
        """(a - b) mod 2^bits (flags: + borrow, overflow rows)."""
        self._adder("sub", out_slots, flags)

    def neg(self, out_slots):
        """(-a) mod 2^bits."""
        self._adder("neg", out_slots)

    def add_flags(self, out_slots):
        """a + b with the carry and the signed overflow rows."""
        self._adder("add", out_slots, True)

    def _tree(self, pred=None, signed=False, sel=False):
        """The compare tree (statuses sign(a_i - b_i), fan-in 3 combines from the top digit). Returns the buffer of
        w rows: the predicate (lt / gt / le / ge / eq / ne) at d / 8 x 8 = 1 when `sel` (unit 1, an AND mux
        selector), else stores nothing and returns (buffer, table) for the store."""
        p, R, w = self.pr, self.R, self.w
        X = self._lin(0, 0, [(r, R + r) for r in range(R)], ca=2, cb=-2)
        S = self._pbs(X, R, "sgn")
        HS = self._signs() if signed else None
        cur, k, first = S, self.nd, True
        U = self._buf()
        while True:
            f = min(3, k)
            h = -(-k // f)
            src = [[(q // h) * k + (q % h) * f + f - 1 - x if (q % h) * f + f - 1 - x < k else -1 for x in range(f)]
                   for q in range(w * h)]
            p.lin(U, cur, cur, [(v[0], v[1] if f > 1 else -1) for v in src], ca=1 << (f - 1), cb=1 << max(f - 2, 0))
            if f == 3:
                p.lin(U, U, cur, [(q, v[2]) for q, v in enumerate(src)], ca=1, cb=1)
            if first and signed:
                p.lin(U, U, HS, [(q, q // h if q % h == h - 1 else -1) for q in range(w * h)], ca=1, cb=1)
            first = False
            if h == 1:
                break
            nxt = self._buf()
            self._pbs(U, w * h, "sgn", nxt)
            cur, k = nxt, h
            U = self._buf()
        # the last combine's input U (w rows): inside [-1, 1) for fan-in <= 2, the whole torus for 3
        if f == 3 and pred in ("eq", "ne"):
            U = self._pbs(U, w, "sgn")          # the status, then the eq / ne table inside [-1, 1)
            f = 2
        if f <= 2:
            if sel:
                return self._pbs(U, w, pred + "1")
            return U, pred + "_w"
        V = self._pbs(U, w, pred + ("_pm1" if sel else "_pm8"))
        if sel:
            return self._lin(V, V, [(q, -1) for q in range(w)], ca=1, cb=0, c=0.5)
        return self._lin(V, V, [(q, -1) for q in range(w)], ca=1, cb=0, c=0.125), "id8_w"

    def cmp_pred(self, out_slots, pred="lt", signed=False):
        """[a pred b] per word (pred lt / gt / le / ge / eq / ne; signed: two's complement), 1 / 0."""
        if pred not in ("lt", "gt", "le", "ge", "eq", "ne"):
            raise ValueError(f"tfhe_bitwise cmp: predicate {pred}?")
        buf, lut = self._tree(pred, signed)
        self._store(buf, self.w, lut, out_slots)

    def _dig(self, src, pairs, ca=1, cb=1, out_slots=None, rows=None):
        """Output digits: a lin into d / 4 (`ca`, `cb` already doubled), then the store's identity table."""
        O = self._lin(src[0], src[1], pairs, ca=ca, cb=cb)
        self._store(O, len(pairs) if rows is None else rows, "dig4_w", out_slots)
        return O

    def _lh(self, rows, src=0):
        """The two-region bits table on rows `rows` of `src` (digits at d / 8): buffer [lo (d & 1) / 8 | hi (d >> 1)
        / 8], len(rows) each."""
        IN = self._lin(src, src, [(r, -1) for r in rows] + [(-1, r) for r in rows], ca=2, cb=-2, c=0.0625)
        return self._pbs(IN, 2 * len(rows), "bits")

    def sext(self, out_slots, from_bits):
        """Sign extension of a's low `from_bits` bits (a multiple of 2) to the word."""
        m, nd, w = int(from_bits) // 2, self.nd, self.w
        if from_bits % 2 or not 0 < m <= nd:
            raise ValueError("tfhe_bitwise sext: from_bits a multiple of 2, at most the width")
        LH = self._lh([self._row(0, q, m - 1) for q in range(w)])
        # out: a_i (i < m), 3 hi(a_(m - 1)) above: hi at d / 8 x 3 x 2
        pairs = [(self._row(0, q, i), -1) if i < m else (-1, w + q) for q in range(w) for i in range(nd)]
        self._dig((0, LH), pairs, ca=2, cb=6, out_slots=out_slots)

    def _src(self, kind, i, s):
        """The source digit of output digit i for a shift / rotate by s digits (None: the fill)."""
        nd = self.nd
        if kind in ("rotl", "rotr"):
            return (i - s) % nd if kind == "rotl" else (i + s) % nd
        j = i - s if kind == "shl" else i + s
        return j if 0 <= j < nd else None

    def shift(self, out_slots, kind, k):
        """a shifted / rotated by the public amount k (kind shl / shr / sar / rotl / rotr): whole digits renumbered,
        an odd bit through one round of the bits table; shifts by >= the width give 0 (sar: the sign)."""
        nd, w, R, bits = self.nd, self.w, self.R, 2 * self.nd
        if kind not in ("shl", "shr", "sar", "rotl", "rotr"):
            raise ValueError(f"tfhe_bitwise shift: {kind}?")
        k = int(k) % bits if kind.startswith("rot") else min(int(k), bits)
        q, t = divmod(k, 2)
        need_sign = kind == "sar" and (t or q)
        if not t:
            if need_sign:
                LH = self._lh(self._top(0))
                fill = lambda qq: (-1, w + qq)          # 3 hi(top) at d / 8: cb 6 at d / 4
            else:
                LH, fill = 0, lambda qq: (-1, -1)
            pairs = []
            for qq in range(w):
                for i in range(nd):
                    j = self._src(kind if kind != "sar" else "shr", i, q)
                    pairs.append((self._row(0, qq, j), -1) if j is not None else fill(qq))
            self._dig((0, LH), pairs, ca=2, cb=6, out_slots=out_slots)
            return
        # one bit (+ q digits): bit-shifted digits b_i, then renumbered: lo / hi of every digit (row r: lo, R + r: hi)
        LH = self._lh(list(range(R)))
        left = kind in ("shl", "rotl")
        rows = []                                # the bit-shifted word (before the digit renumbering): (lo, hi) rows
        for qq in range(w):
            for i in range(nd):
                r = qq * nd + i
                if left:                         # 2 lo_i + hi_(i - 1)
                    jj = i - 1 if i > 0 else (nd - 1 if kind == "rotl" else None)
                    rows.append((r, R + qq * nd + jj if jj is not None else -1))
                else:                            # hi_i + 2 lo_(i + 1)
                    jj = i + 1 if i < nd - 1 else (0 if kind == "rotr" else None)
                    if jj is not None:
                        rows.append((qq * nd + jj, R + r))
                    elif kind == "sar":          # 2 lo(fill) + hi_top = 3 hi_top
                        rows.append((-2, R + r))
                    else:
                        rows.append((-1, R + r))
        # 2 lo + hi at d / 8 (the sar top row: 2 hi + hi, hi through the lo slot too)
        BS = self._lin(LH, LH, [(b if a == -2 else a, b) for a, b in rows], ca=2, cb=1)
        if q == 0:
            self._store(self._lin(BS, BS, [(r, -1) for r in range(R)], ca=2, cb=0), R, "dig4_w", out_slots)
            return
        pairs = []
        for qq in range(w):
            for i in range(nd):
                j = self._src(kind if kind != "sar" else "shr", i, q)
                if j is not None:
                    pairs.append((qq * nd + j, -1))
                elif kind == "sar":
                    pairs.append((-1, R + qq * nd + nd - 1))     # 3 hi(top) at d / 8
                else:
                    pairs.append((-1, -1))
        self._dig((BS, LH), pairs, ca=2, cb=6, out_slots=out_slots)

    def _andmux(self, groups, dst_rows):
        """sum_g and(sel_g, x_g) per output row in one batch: `groups` = [(X buffer, x rows, SEL buffer, sel rows)],
        every group dst_rows long (sel at unit 1, x at d / 8). Returns the buffer of the sums (d / 8)."""
        n = dst_rows
        XX = self._buf()
        SS = self._buf()
        for g, (xb, xr, sb, sr) in enumerate(groups):
            if g == 0:
                self.pr.lin(XX, xb, xb, [(r, -1) for r in xr], ca=1, cb=0)
                self.pr.lin(SS, sb, sb, [(r, -1) for r in sr], ca=1, cb=0)
            else:                                # append: rows g n .. (g + 1) n (a fresh buffer each time)
                X2, S2 = self._buf(), self._buf()
                self.pr.lin(X2, XX, xb, [(r, -1) for r in range(g * n)] + [(-1, r) for r in xr], ca=1, cb=1)
                self.pr.lin(S2, SS, sb, [(r, -1) for r in range(g * n)] + [(-1, r) for r in sr], ca=1, cb=1)
                XX, SS = X2, S2
        G = len(groups)
        IN = self._lin(XX, SS, [(r, r) for r in range(G * n)], ca=2, cb=1, c=-0.875)
        A = self._pbs(IN, G * n, "and")
        O = self._lin(A, A, [(r, n + r if G > 1 else -1) for r in range(n)], ca=1, cb=1)
        for g in range(2, G):
            self._lin(O, A, [(r, g * n + r) for r in range(n)], ca=1, cb=1, dst=O)
        return O

    def _or_tree(self, buf, items):
        """OR of every word's items (rows of `buf`, (row, max) with max <= 3): groups summing to <= 7 through the
        threshold table, until one bit per word. Returns (buffer, rows) of the w bits at d / 8."""
        cur, its = buf, items                     # its[q] = [(row, max)]
        while any(len(x) > 1 or x[0][1] > 1 for x in its):
            groups, nxt = [], []
            for x in its:
                gs, g, s = [], [], 0
                for row, mx in sorted(x, key=lambda v: -v[1]):
                    if g and s + mx > 7:
                        gs.append(g)
                        g, s = [], 0
                    g.append(row)
                    s += mx
                gs.append(g)
                nxt.append(gs)
                groups.extend(gs)
            sums = self._gsum(cur, groups)
            B = self._pbs(sums, len(groups), "nz")
            k, its = 0, []
            for gs in nxt:
                its.append([(k + j, 1) for j in range(len(gs))])
                k += len(gs)
            cur = B
        return cur, [x[0][0] for x in its]

    def _gsum(self, buf, groups, ca=2, c=-0.875):
        """Row g = ca x sum of rows groups[g] of `buf` + c (the threshold tables' input 2 s / 8 - 7 / 8)."""
        m = max(len(g) for g in groups)
        S = self._lin(buf, buf, [(g[0], g[1] if len(g) > 1 else -1) for g in groups], ca=ca, cb=ca, c=c)
        # (a group [-1]: no term, the constant alone)
        for t in range(2, m):
            self._lin(S, buf, [(gi, g[t] if len(g) > t else -1) for gi, g in enumerate(groups)], ca=1, cb=ca, dst=S)
        return S

    def shift_enc(self, out_slots, kind):
        """a shifted / rotated by the encrypted amount b (a word; kind shl / shr / sar / rotl / rotr): a barrel
        shifter of log2(bits) AND-mux stages; shifts by >= the width give 0 (sar: the sign), rotates take b mod
        the width."""
        nd, w, R, bits = self.nd, self.w, self.R, 2 * self.nd
        if kind not in ("shl", "shr", "sar", "rotl", "rotr"):
            raise ValueError(f"tfhe_bitwise shift_enc: {kind}?")
        L = bits.bit_length() - 1
        na = (L + 1) // 2                          # amount digits holding bits 0 .. L - 1
        # round 1, one batch: lo / hi of a's digits (stage 0, the sign) and of the amount's low digits
        arows = [self._row(1, q, i) for q in range(w) for i in range(na)]
        LH = self._lh(list(range(R)) + arows)
        n1 = R + len(arows)                        # LH rows: lo of (a | amount), then hi of the same
        lo = lambda r: r
        hi = lambda r: n1 + r
        bit = lambda q, k: (lo if k % 2 == 0 else hi)(R + q * na + k // 2)      # bit k of word q's amount (/ 8)
        # selectors at unit 1: sel = 8 bit, ~sel = 1 - 8 bit (rows k w + q)
        SEL = self._lin(LH, LH, [(bit(q, k), -1) for k in range(L) for q in range(w)], ca=8, cb=0)
        NSEL = self._lin(LH, LH, [(bit(q, k), -1) for k in range(L) for q in range(w)], ca=-8, cb=0, c=1.0)
        shifting = not kind.startswith("rot")
        FILL = None
        if kind == "sar":                          # 3 hi(top) at d / 8, per word
            FILL = self._lin(LH, LH, [(hi(q * nd + nd - 1), -1) for q in range(w)], ca=3, cb=0)
        if shifting:
            # amount >= 2^L: any bit >= L: the digits above bit L (and hi of digit L // 2 when L is odd)
            q0 = L // 2
            items_rows = []
            ITEMS_pairs = []
            for q in range(w):
                its = []
                if L % 2:
                    ITEMS_pairs.append((-1, hi(R + q * na + q0) if q0 < na else -1))
                    its.append((len(ITEMS_pairs) - 1, 1))
                for i in range(q0 + (L % 2), nd):
                    ITEMS_pairs.append((self._row(1, q, i), -1))
                    its.append((len(ITEMS_pairs) - 1, 3))
                items_rows.append(its)
            IT = self._lin(0, LH, ITEMS_pairs, ca=1, cb=1)
            BIG, brow = self._or_tree(IT, items_rows)
            # last stage's selectors: s1 = [bit + ~big >= 2], s0 = [~bit + ~big >= 2] (unit 1), big (unit 1)
            kl = L - 1
            # (2 s / 8 - 7 / 8: s1's s = bit + 1 - big, s0's s = 2 - bit - big)
            IN = self._lin(LH, BIG, [(bit(q, kl), brow[q]) for q in range(w)], ca=2, cb=-2, c=-0.625)
            IN2 = self._lin(LH, BIG, [(bit(q, kl), brow[q]) for q in range(w)], ca=-2, cb=-2, c=-0.375)
            INS = self._lin(IN, IN2, [(q, -1) for q in range(w)] + [(-1, q) for q in range(w)], ca=1, cb=1)
            LS = self._pbs(INS, 2 * w, "ge2u")
            BIGU = self._lin(BIG, BIG, [(brow[q], -1) for q in range(w)], ca=8, cb=0)
        # stage 0: one bit; x_s from LH (as the public shift by 1), then the stages of 1, 2, 4 .. digits
        rows = []
        for q in range(w):
            for i in range(nd):
                r = q * nd + i
                if kind in ("shl", "rotl"):
                    jj = i - 1 if i > 0 else (nd - 1 if kind == "rotl" else None)
                    rows.append((lo(r), hi(q * nd + jj) if jj is not None else -1))
                else:
                    jj = i + 1 if i < nd - 1 else (0 if kind == "rotr" else None)
                    if jj is not None:
                        rows.append((lo(q * nd + jj), hi(r)))
                    elif kind == "sar":
                        rows.append((hi(r), hi(r)))
                    else:
                        rows.append((-1, hi(r)))
        XS = self._lin(LH, LH, rows, ca=2, cb=1)
        X = 0
        xr = list(range(R))
        last_k = L - 1
        for k in range(L):
            if k > 0:                              # shift by 2^(k - 1) digits
                s = 1 << (k - 1)
                pairs = []
                for q in range(w):
                    for i in range(nd):
                        j = self._src(kind if kind != "sar" else "shr", i, s)
                        if j is not None:
                            pairs.append((q * nd + j, -1))
                        else:
                            pairs.append((-1, q) if kind == "sar" else (-1, -1))
                XS = self._lin(X, FILL if FILL is not None else X, pairs, ca=1, cb=1)
                xr = list(range(R))
            sel_rows = [k * w + q for q in range(w) for _ in range(nd)]
            if shifting and k == last_k:
                g = [(XS, list(range(R)), LS, [q for q in range(w) for _ in range(nd)]),
                     (X, xr, LS, [w + q for q in range(w) for _ in range(nd)])]
                if kind == "sar":
                    g.append((FILL, [q for q in range(w) for _ in range(nd)], BIGU, [q for q in range(w) for _ in range(nd)]))
                X = self._andmux(g, R)
            else:
                X = self._andmux([(XS, list(range(R)), SEL, sel_rows), (X, xr, NSEL, sel_rows)], R)
            xr = list(range(R))
        self._store(self._lin(X, X, [(r, -1) for r in range(R)], ca=2, cb=0), R, "dig4_w", out_slots)

    def _reduce(self, ITEMS, cols, ncols):
        """Carry-save tree: `cols[q][c]` = [(row of ITEMS, max)] the items of word q's column c (digit position), until
        every column sums to <= 6 (the prefix adder's s). A round, low columns first: a column whose items plus the
        carries arriving this round could exceed 6 is cut into groups summing to <= 7, each one lookup pair (the
        digit s mod 4 and the carry s >= 4, both cleaned: two batches) -- every carry lands in a column that this
        same round compresses if it must, so carries never ripple. Returns (ITEMS, cols)."""
        while any(sum(m for _, m in x) > 6 for col in cols for x in col):
            groups, plan = [], []
            for q, col in enumerate(cols):
                inc = 0
                for c, x in enumerate(col):
                    tot = sum(m for _, m in x) + inc
                    ents, gs = [], []
                    if tot <= 6:
                        ents = [("pass", r, m) for r, m in x]
                    else:                        # first fit decreasing into groups of sum <= 7
                        for r, m in sorted(x, key=lambda v: -v[1]):
                            for g in gs:
                                if sum(v for _, v in g) + m <= 7:
                                    g.append((r, m))
                                    break
                            else:
                                gs.append([(r, m)])
                        for g in gs:
                            if len(g) == 1:
                                ents.append(("pass", g[0][0], g[0][1]))
                            else:
                                ents.append(("grp", len(groups), g))
                                groups.append(g)
                    inc = sum(1 for e in ents if e[0] == "grp" and sum(m for _, m in e[2]) >= 4)
                    plan.append((q, c, ents))
            S = self._gsum(ITEMS, [[r for r, _ in g] for g in groups], ca=2, c=-0.875)      # 2 s / 8 - 7 / 8
            CY = self._pbs(S, len(groups), "cy7")                                          # [s >= 4] / 8
            DG = self._pbs(S, len(groups), "md7")                                          # (s mod 4) / 8
            pass_pairs, dig_pairs = [], []
            ncol = [[[] for _ in range(ncols)] for _ in cols]
            for q, c, ents in plan:
                for e in ents:
                    if e[0] == "pass":
                        ncol[q][c].append((len(pass_pairs), e[2]))
                        pass_pairs.append((e[1], -1))
                        dig_pairs.append(-1)
                    else:
                        gi, g = e[1], e[2]
                        sm = sum(m for _, m in g)
                        ncol[q][c].append((len(pass_pairs), min(3, sm)))
                        pass_pairs.append((-1, -1))
                        dig_pairs.append(gi)
                        if sm >= 4 and c + 1 < ncols:
                            ncol[q][c + 1].append((len(pass_pairs), 1))
                            pass_pairs.append((-1, gi))
                            dig_pairs.append(-1)
            N = self._lin(ITEMS, CY, pass_pairs, ca=1, cb=1)
            self._lin(N, DG, [(r, d) for r, d in enumerate(dig_pairs)], ca=1, cb=1, dst=N)
            ITEMS, cols = N, ncol
        return ITEMS, cols

    def _final_add(self, ITEMS, cols, nout, out_slots):
        """Columns summing to <= 6 added by the prefix adder (nout digits per word, a word = nout rows)."""
        w = self.w
        if all(sum(m for _, m in x) <= 3 for col in cols for x in col):
            # no carries: every column is its digit (s <= 3)
            g = [[r for r, _ in col[c]] or [-1] for col in cols for c in range(nout)]
            O = self._gsum(ITEMS, g, ca=2, c=0.0)
            self._store(O, w * nout, "dig4_w", out_slots)
            return
        save = (self.nd, self.R)
        self.nd, self.R = nout, w * nout
        g = [[r for r, _ in col[c]] or [-1] for col in cols for c in range(nout)]
        X = self._gsum(ITEMS, g, ca=2, c=-0.75)                                            # (s - 3) / 4
        C = self._carries(X)
        prev = [(r, r - 1 if r % nout else -1) for r in range(self.R)]
        Z = self._lin(X, C, prev, ca=1, cb=2, c=-0.125)
        self._store(Z, self.R, "out_w", out_slots)
        self.nd, self.R = save

    def mul(self, out_slots, wide=False):
        """a b mod 2^bits (wide: the 2 bits-bit product, 2 nd digits per word)."""
        nd, w, R = self.nd, self.w, self.R
        nc = 2 * nd if wide else nd
        # round 1: lo / hi of a's and b's digits (a << 1 and b's bits)
        LH = self._lh(list(range(2 * R)))
        lo = lambda r: r
        hi = lambda r: 2 * R + r
        # a << 1 (nd + 1 digits when wide): 2 lo(a_i) + hi(a_(i - 1)); AA = [a | a << 1], a's rows q nd + i,
        # a << 1's rows R + q (nd + 1) + i
        n1 = nd + 1
        sh = [(lo(q * nd + i) if i < nd else -1, hi(q * nd + i - 1) if i > 0 else -1) for q in range(w) for i in range(n1)]
        A1 = self._lin(LH, LH, sh, ca=2, cb=1)
        AA = self._lin(0, A1, [(r, -1) for r in range(R)] + [(-1, r) for r in range(w * n1)], ca=1, cb=1)
        xr, br, cols = [], [], [[[] for _ in range(nc)] for _ in range(w)]
        for q in range(w):
            for j in range(4 * nd // 2):         # bit j of b
                bj = (lo if j % 2 == 0 else hi)(R + q * nd + j // 2)
                for i in range(nd + (j % 2)):
                    c = i + j // 2
                    if c >= nc or (j % 2 == 0 and i >= nd):
                        continue
                    cols[q][c].append((len(xr), 3))
                    xr.append(q * nd + i if j % 2 == 0 else R + q * n1 + i)
                    br.append(bj)
        IN = self._lin(AA, LH, list(zip(xr, br)), ca=2, cb=8, c=-0.875)
        PP = self._pbs(IN, len(xr), "and")
        ITEMS, cols = self._reduce(PP, cols, nc)
        self._final_add(ITEMS, cols, nc, out_slots)

    def mulc(self, out_slots, c):
        """a c mod 2^bits for a public constant c (shifted copies of a and a << 1, the same tree)."""
        nd, w, R = self.nd, self.w, self.R
        c = int(c) % (1 << (2 * nd))
        n1 = nd + 1
        odd = any((c >> j) & 1 for j in range(1, 2 * nd, 2))
        if odd:
            LH = self._lh(list(range(R)))
            sh = [(q * nd + i if i < nd else -1, R + q * nd + i - 1 if i > 0 else -1) for q in range(w) for i in range(n1)]
            A1 = self._lin(LH, LH, sh, ca=2, cb=1)
            AA = self._lin(0, A1, [(r, -1) for r in range(R)] + [(-1, r) for r in range(w * n1)], ca=1, cb=1)
        else:
            AA = 0
        cols = [[[] for _ in range(nd)] for _ in range(w)]
        for q in range(w):
            for j in range(2 * nd):
                if not (c >> j) & 1:
                    continue
                for i in range(nd):
                    cc = i + j // 2
                    if cc < nd:
                        cols[q][cc].append((q * nd + i if j % 2 == 0 else R + q * n1 + i, 3))
        if not any(cols[q][x] for q in range(w) for x in range(nd)):
            self._dig((0, 0), [(-1, -1)] * R, out_slots=out_slots)
            return
        ITEMS, cols = self._reduce(AA, cols, nd)
        self._final_add(ITEMS, cols, nd, out_slots)

    def popcount(self, out_slots):
        """The number of set bits of a, as a word."""
        nd, w, R = self.nd, self.w, self.R
        IN = self._lin(0, 0, [(r, -1) for r in range(R)], ca=2, cb=0)
        PC = self._pbs(IN, R, "pc")
        cols = [[[(q * nd + i, 2) for i in range(nd)]] + [[] for _ in range(nd - 1)] for q in range(w)]
        ITEMS, cols = self._reduce(PC, cols, nd)
        self._final_add(ITEMS, cols, nd, out_slots)

    def minmax(self, out_slots, kind="min", signed=False):
        """min / max of a, b per word."""
        nd, w, R = self.nd, self.w, self.R
        S = self._tree("lt", signed, sel=True)                     # [a < b] at unit 1, w rows
        NS = self._lin(S, S, [(q, -1) for q in range(w)], ca=-1, cb=0, c=1.0)
        first, second = (0, R) if kind == "min" else (R, 0)
        sel = [q for q in range(w) for _ in range(nd)]
        O = self._andmux([(0, [first + r for r in range(R)], S, sel), (0, [second + r for r in range(R)], NS, sel)], R)
        self._store(self._lin(O, O, [(r, -1) for r in range(R)], ca=2, cb=0), R, "dig4_w", out_slots)

    def abs(self, out_slots):
        """|a| (two's complement; abs(MIN) = MIN)."""
        nd, w, R = self.nd, self.w, self.R
        X = self._lin(0, 0, [(r, -1) for r in range(R)], ca=2, cb=0)      # -a = ~(a + 11..1)
        LH = self._lh(self._top(0))
        C = self._carries(X)
        prev = [(r, r - 1 if r % nd else -1) for r in range(R)]
        # -a at d / 8: 3/8 - (a + 3/8 + c_(i - 1) - 4 c_i) = -a - c_(i - 1) + 4 c_i
        N = self._lin(0, C, prev, ca=-1, cb=-1)
        self._lin(N, C, [(r, r) for r in range(R)], ca=1, cb=4, dst=N)
        S = self._lin(LH, LH, [(w + q, -1) for q in range(w)], ca=8, cb=0)        # sign (unit 1)
        NS = self._lin(LH, LH, [(w + q, -1) for q in range(w)], ca=-8, cb=0, c=1.0)
        sel = [q for q in range(w) for _ in range(nd)]
        O = self._andmux([(N, list(range(R)), S, sel), (0, list(range(R)), NS, sel)], R)
        self._store(self._lin(O, O, [(r, -1) for r in range(R)], ca=2, cb=0), R, "dig4_w", out_slots)


# ── tables of the other ops (native values on [-1, 1); module doc) ───────────────────────────────────────────────

def _k8(m):                  # the threshold tables' input 2 s / 8 - 7 / 8 -> s in 0..7
    return np.clip(np.round(4.0 * (np.asarray(m, dtype=np.float64) + 0.875)), 0, 7)


def _bits2(m):               # lo on 2 d / 8 + 1/16 (upper region), hi on -2 d / 8 + 1/16 (lower region)
    k = np.round((np.asarray(m, dtype=np.float64) - 0.0625) * 4.0).astype(np.int64)
    return np.where(k >= 0, np.clip(k, 0, 3) & 1, np.clip(-k, 0, 3) >> 1) / 8.0


def _and(m):                 # 2 x / 8 + sel - 7 / 8 -> sel x / 8
    k = _k8(m)
    return np.where(k >= 4, k - 4, 0) / 8.0


_PRED = {"lt": lambda r: r < 0, "gt": lambda r: r > 0, "le": lambda r: r <= 0, "ge": lambda r: r >= 0,
         "eq": lambda r: r == 0, "ne": lambda r: r != 0}

TABLES = {
    "sgn": (_sgn, False), "carry_pm": (_carry_pm, False),
    "out_w": (lambda m: 2.0 * _digit_out(m), True),
    "outn_w": (lambda m: 2.0 * (0.375 - _digit_out(m)), True),
    "dig4_w": (lambda m: np.clip(np.round(4.0 * np.asarray(m)), 0, 3) / 4.0, True),
    "id8_w": (lambda m: np.where(np.asarray(m) > 0.125, 1.0, 0.0), True),      # {0, 1/4} -> {0, 1}
    "hs": (lambda m: np.where(_r4(m) >= 2, 2.0, 0.0), False),          # a sign bit as a half turn (2 d / 8 in)
    "bits": (_bits2, False),
    "and": (_and, False),
    "cy7": (lambda m: np.where(_k8(m) >= 4, 0.125, 0.0), False),
    "md7": (lambda m: np.mod(_k8(m), 4) / 8.0, False),
    "nz": (lambda m: np.where(_k8(m) >= 1, 0.125, 0.0), False),
    "ge2u": (lambda m: np.where(_k8(m) >= 2, 1.0, 0.0), False),
    "ovf_pm": (lambda m: np.where((_k8(m) >= 2) & (_k8(m) <= 5), 0.0625, -0.0625), False),
    "bw_and": (lambda m: np.where(_r4(m) == 2, 0.125, 0.0), False),        # bit sums 0 / 1 / 2 at 1/4
    "bw_or": (lambda m: np.where(_r4(m) >= 1, 0.125, 0.0), False),
    "bw_xor": (lambda m: np.where(_r4(m) == 1, 0.125, 0.0), False),
    "pc": (lambda m: np.array([0, 1, 1, 2])[np.clip(_r4(m), 0, 3).astype(np.int64)] / 8.0, False),
}
for _p, _f in _PRED.items():
    TABLES[_p + "_w"] = (lambda m, f=_f: np.where(f(_r4(m)), 1.0, 0.0), True)
    TABLES[_p + "1"] = (lambda m, f=_f: np.where(f(_r4(m)), 1.0, 0.0), False)
    TABLES[_p + "_pm8"] = (lambda m, f=_f: np.where(f(_r4(m)), 0.125, -0.125), False)
    TABLES[_p + "_pm1"] = (lambda m, f=_f: np.where(f(_r4(m)), 0.5, -0.5), False)


# ── the noise model and the device cases ──────────────────────────────────────────────────────────────────────────

# a lookup's input noise sd (native units) at the shipped parameters (n_lwe 840, n_br 2048, sigma 2^-18 p0): the mod
# switch to 2 nu n_br and the key switch s_br -> s_lwe (pc-check tfhe-bitwise's model), by nu
LOOKUP_SD = {1: 6.0e-3, 2: 3.31e-3}
PBS_OUT_SD = 5.0e-4          # a PBS output's noise sd bound (measured rms 3.0e-4 .. 4.8e-4 at key-bit groups 5 .. 7)


def _log2_tail2(z):
    """log2 P(|N(0, 1)| > z)."""
    import math
    if z < 8.0:
        return math.log2(math.erfc(z / math.sqrt(2.0)))
    ln = -z * z / 2.0 - math.log(z) - 0.5 * math.log(2.0 * math.pi) + math.log1p(-1.0 / z**2 + 3.0 / z**4)
    return 1.0 + ln / math.log(2.0)


def _boundaries(f, pts):
    """The decision boundaries of a native table on the whole torus [-2, 2) (negacyclic beyond [-1, 1)): midpoints
    between neighbouring grid points of different output."""
    step = 2.0 / pts
    g = -2.0 + step * np.arange(2 * pts)
    v = tfhe._nat(f, g, pts)
    d = np.abs(np.roll(v, -1) - v) > 1e-9
    return (g[d] + step / 2.0)


def analyze(prog, m_in, nu=None, sp=PBS_OUT_SD):
    """Rounds, PBS count and p_fail per lookup of a native program on the noiseless loaded values `m_in` (every row of
    every lookup: the margin from its noiseless input to the table's nearest decision boundary on the torus, against
    sqrt(LOOKUP_SD^2 + the lins' propagated variance)). `rounds`: the PBS depth up to the store (not counting it);
    `launches`: pbs steps (+ the store's table)."""
    import math
    nu = tfhe.GRID // 2048 if nu is None else int(nu)
    sd = LOOKUP_SD[int(nu)]
    pts = tfhe.GRID if int(nu) == int(tfhe.GRID // 2048) else 2048 * int(nu)
    B, V, D = {}, {}, {}
    worst, tot, pbs, launches, store_rounds = -math.inf, -math.inf, 0, 0, 0
    bnd = {}

    def look(lut, x, var):
        nonlocal worst, tot
        fs, _ = prog.luts[lut]
        if lut not in bnd:
            bnd[lut] = _boundaries(fs[0], pts // len(fs))
        b = bnd[lut]
        xm = ((x + 2.0) % 4.0) - 2.0
        dist = np.abs(xm[:, None] - b[None, :])
        dist = np.minimum(dist, 4.0 - dist).min(1) if b.size else np.full(len(x), 4.0)
        z = dist / np.sqrt(sd * sd + var)
        lp = np.array([_log2_tail2(float(v)) for v in z])
        m = float(lp.max())
        worst = max(worst, m)
        here = m + math.log2(float(np.sum(2.0 ** (lp - m))))
        tot = here if tot == -math.inf else max(tot, here) + math.log2(1.0 + 2.0 ** (-abs(tot - here)))
        return [tfhe._nat(f, x, pts // len(fs)) for f in fs]

    for st in prog.steps:
        op = st["op"]
        if op == "load":
            fs = prog.luts[st["lut"]][0]
            for d, f in zip(st["dst"] if isinstance(st["dst"], list) else [st["dst"]], fs):
                B[d] = tfhe._nat(f, np.asarray(m_in, dtype=np.float64), pts // len(fs))
                V[d] = np.full(len(B[d]), sp * sp)
                D[d] = 0
        elif op == "lin":
            idx = np.asarray(st["idx"], dtype=np.int64).reshape(-1, 2)
            a, b = B.get(st["a"], np.zeros(0)), B.get(st["b"], np.zeros(0))
            va, vb = V.get(st["a"], np.zeros(0)), V.get(st["b"], np.zeros(0))
            ia, ib = idx[:, 0] >= 0, idx[:, 1] >= 0
            ta = np.where(ia, a[np.maximum(idx[:, 0], 0)] if a.size else 0.0, 0.0)
            tb = np.where(ib, b[np.maximum(idx[:, 1], 0)] if b.size else 0.0, 0.0)
            wa = np.where(ia, va[np.maximum(idx[:, 0], 0)] if va.size else 0.0, 0.0)
            wb = np.where(ib, vb[np.maximum(idx[:, 1], 0)] if vb.size else 0.0, 0.0)
            da = D.get(st["a"], 0) if ia.any() else 0
            db = D.get(st["b"], 0) if ib.any() else 0
            B[st["dst"]] = st["ca"] * ta + st["cb"] * tb + st["c"]
            V[st["dst"]] = st["ca"] ** 2 * wa + st["cb"] ** 2 * wb
            D[st["dst"]] = max(da, db)
        elif op in ("pbs", "store"):
            x = B[st["src"]][:st["rows"]]
            var = V[st["src"]][:st["rows"]]
            if op == "store":
                store_rounds = D[st["src"]]
                if st["lut"] is not None:
                    look(st["lut"], x, var)
                    launches += 1
                continue
            ys = look(st["lut"], x, var)
            pbs += st["rows"]
            launches += 1
            for d, y in zip(st["dst"], ys):
                B[d] = y
                V[d] = np.full(len(y), sp * sp)
                D[d] = D[st["src"]] + 1
        else:
            raise ValueError(f"analyze: step {op}")
    return {"rounds": int(store_rounds), "pbs": int(pbs), "launches": int(launches),
            "log2_pfail_lookup": worst, "log2_pfail_op": tot, "lookup_sd": sd, "pbs_out_sd": sp}


def load_values(slot_values):
    """The native values a load leaves (the load table on the slot values)."""
    return _load(np.asarray(slot_values, dtype=np.float64))


# ── the op registry (pc-check cases, the mathlib task, the mirror tests) ──────────────────────────────────────────
#
# op names: add sub neg addf subf (+ carry / borrow and overflow rows) | lt le gt ge eq ne, s-prefixed signed (slt ..)
# | sext<k> | shl<k> shr<k> sar<k> rotl<k> rotr<k> (public k) | shlv shrv sarv rotlv rotrv (amount = b) | mul mulw
# mulc<c> | min max smin smax abs popcount | mux xor and or (the round-2 ops) | id

import re as _re

_PREDS = ("lt", "le", "gt", "ge", "eq", "ne")


def parse(op):
    m = _re.fullmatch(r"(sext|shl|shr|sar|rotl|rotr|mulc)(\d+)", op)
    if m:
        return m.group(1), int(m.group(2))
    return op, None


def build(op, name, words, nd):
    """A WordProg computing `op` on `words` words of nd digits (inputs: a, b, sel as `WordProg.rows_in(True)`)."""
    wp = WordProg(name, words, nd)
    out = name + ".out"
    base, k = parse(op)
    if base == "add":
        wp.add(out)
    elif base in ("sub", "neg"):
        getattr(wp, base)(out)
    elif base == "addf":
        wp.add_flags(out)
    elif base == "subf":
        wp.sub(out, flags=True)
    elif base in _PREDS or (base[:1] == "s" and base[1:] in _PREDS):
        wp.cmp_pred(out, base.lstrip("s") if base not in _PREDS else base, signed=base not in _PREDS)
    elif base == "cmp":
        wp.cmp(out)
    elif base == "sext":
        wp.sext(out, k)
    elif base in ("shl", "shr", "sar", "rotl", "rotr"):
        wp.shift(out, base, k)
    elif base.endswith("v") and base[:-1] in ("shl", "shr", "sar", "rotl", "rotr"):
        wp.shift_enc(out, base[:-1])
    elif base in ("mul", "mulw"):
        wp.mul(out, wide=base == "mulw")
    elif base == "mulc":
        wp.mulc(out, k)
    elif base in ("min", "max", "smin", "smax"):
        wp.minmax(out, base.lstrip("s"), signed=base.startswith("s"))
    elif base == "abs":
        wp.abs(out)
    elif base == "popcount":
        wp.popcount(out)
    elif base == "mux":
        wp.mux(out)
    elif base in ("xor", "and", "or"):
        wp.bitwise(base, out)
    elif base == "id":
        wp.ident(out)
    else:
        raise ValueError(f"tfhe_bitwise: op {op}?")
    if not hasattr(wp, "out_rows"):
        wp.out_rows = wp.pr.steps[-1]["rows"]
    compact(wp.pr)
    return wp


def compact(prog):
    """Map a load / lin / pbs / store program's buffers onto as few as their live ranges allow (linear scan; a step's
    new buffer never takes one the step reads, so only a buffer's own in-place lins share rows)."""
    def uses(st):
        op = st["op"]
        if op == "lin":
            return [st["a"], st["b"]], [st["dst"]]
        if op == "pbs":
            return [st["src"]], list(st["dst"])
        if op == "store":
            return [st["src"]], []
        if op == "load":
            return [], st["dst"] if isinstance(st["dst"], list) else [st["dst"]]
        raise ValueError(f"compact: step {op}")
    last = {}
    for i, st in enumerate(prog.steps):
        rd, wr = uses(st)
        for v in rd + wr:
            last[v] = i
    phys, free, nphys = {}, [], 0
    for i, st in enumerate(prog.steps):
        rd, wr = uses(st)
        busy = {phys[v] for v in rd if v in phys}
        for v in wr:
            if v not in phys:
                pick = next((f for f in sorted(free) if f not in busy), None)
                if pick is None:
                    pick, nphys = nphys, nphys + 1
                else:
                    free.remove(pick)
                phys[v] = pick
        for v in set(rd + wr):
            if last[v] == i and v in phys and phys[v] not in free:
                free.append(phys[v])
        op = st["op"]
        if op == "lin":
            # a read of a buffer never written (a lin's unused operand): any physical buffer
            st["a"], st["b"] = (phys.get(st["a"], 0), phys.get(st["b"], 0))
            st["dst"] = phys[st["dst"]]
        elif op == "pbs":
            st["src"], st["dst"] = phys[st["src"]], [phys[d] for d in st["dst"]]
        elif op == "store":
            st["src"] = phys[st["src"]]
        else:
            st["dst"] = [phys[d] for d in st["dst"]] if isinstance(st["dst"], list) else phys[st["dst"]]
    prog.bufs = max(1, nphys)
    return prog


def _sgnd(x, bits):
    x = np.asarray(x, dtype=np.int64)
    return np.where(x >= 1 << (bits - 1), x - (1 << bits), x)


def reference(op, a, b, sel, bits):
    """The exact outputs of `op` per word: a list of integer arrays (a word value, or flags / predicates)."""
    a, b, sel = (np.asarray(v, dtype=np.int64) for v in (a, b, sel))
    msk = (1 << bits) - 1
    sa, sb = _sgnd(a, bits), _sgnd(b, bits)
    base, k = parse(op)
    if base in ("add", "addf"):
        r = [(a + b) & msk]
        if base == "addf":
            r += [((a + b) >> bits) & 1, ((sa + sb < -(1 << (bits - 1))) | (sa + sb >= 1 << (bits - 1))).astype(np.int64)]
        return r
    if base in ("sub", "subf"):
        r = [(a - b) & msk]
        if base == "subf":
            r += [(a < b).astype(np.int64), ((sa - sb < -(1 << (bits - 1))) | (sa - sb >= 1 << (bits - 1))).astype(np.int64)]
        return r
    if base == "neg":
        return [(-a) & msk]
    if base == "cmp":
        return [(a < b).astype(np.int64)]
    if base in _PREDS or (base[:1] == "s" and base[1:] in _PREDS):
        x, y = (sa, sb) if base not in _PREDS else (a, b)
        f = {"lt": x < y, "le": x <= y, "gt": x > y, "ge": x >= y, "eq": x == y, "ne": x != y}[base.lstrip("s") if base not in _PREDS else base]
        return [f.astype(np.int64)]
    if base == "sext":
        return [_sgnd(a & ((1 << k) - 1), k) & msk]
    if base in ("shl", "shr", "sar", "rotl", "rotr") or (base.endswith("v") and base[:-1] in ("shl", "shr", "sar", "rotl", "rotr")):
        kind = base.rstrip("v") if base.endswith("v") else base
        s_ = np.full_like(a, k) if k is not None else b
        if kind.startswith("rot"):
            s_ = s_ % bits
            sh = s_ if kind == "rotl" else (bits - s_) % bits
            return [((a << sh) | (a >> ((bits - sh) % bits) * (sh > 0))) & msk if False else
                    np.where(sh == 0, a, ((a << sh) | (a >> (bits - sh))) & msk)]
        big = s_ >= bits
        sc = np.minimum(s_, bits - 1)
        if kind == "shl":
            return [np.where(big, 0, (a << sc) & msk)]
        if kind == "shr":
            return [np.where(big, 0, a >> sc)]
        return [np.where(big, np.where(sa < 0, msk, 0), (sa >> sc) & msk)]
    if base == "mul":
        return [(a * b) & msk]
    if base == "mulw":
        return [(a.astype(object) * b.astype(object)).astype(np.int64) if bits < 32 else
                np.array([int(x) * int(y) for x, y in zip(a, b)], dtype=np.uint64).astype(np.int64)]
    if base == "mulc":
        return [(a * (k & msk)) & msk]
    if base in ("min", "max", "smin", "smax"):
        x, y = (sa, sb) if base.startswith("s") else (a, b)
        lt = x < y
        return [np.where(lt, a, b) if base.endswith("min") else np.where(lt, b, a)]
    if base == "abs":
        return [np.abs(sa) & msk]
    if base == "popcount":
        return [np.array([bin(int(v)).count("1") for v in a], dtype=np.int64)]
    if base == "mux":
        return [np.where(sel == 1, a, b)]
    if base in ("xor", "and", "or"):
        return [{"xor": a ^ b, "and": a & b, "or": a | b}[base]]
    if base == "id":
        return [a]
    raise ValueError(f"tfhe_bitwise: op {op}?")


def out_layout(op, nd):
    """(digits per word of the word result (0: none), number of 0 / 1 rows per word after them, value scale): how a
    program's store rows read (digits at d / 4 then flags at 1 / 4, or one predicate per word at 1)."""
    base, _ = parse(op)
    if base in ("addf", "subf"):
        return nd, 2, 4.0
    if base == "cmp" or base in _PREDS or (base[:1] == "s" and base[1:] in _PREDS):
        return 0, 1, 1.0
    if base == "mulw":
        return 2 * nd, 0, 4.0
    if base in ("xor", "and", "or"):
        return nd, 0, 4.0
    return nd, 0, 4.0


def want_rows(op, a, b, sel, nd):
    """The exact store rows (integers in the store's units: digits, then 0 / 1 flags)."""
    ref = reference(op, a, b, sel, 2 * nd)
    ndig, nfl, _ = out_layout(op, nd)
    w = len(np.asarray(a))
    rows = []
    if ndig:
        v = ref[0]
        rows.append(np.stack([(v >> (2 * i)) & 3 for i in range(ndig)], 1).reshape(-1))
        rows += [np.asarray(f) for f in ref[1:1 + nfl]]
    else:
        rows.append(np.asarray(ref[0]))
    return np.concatenate([r.reshape(-1) for r in rows]).astype(np.int64), ndig, nfl, w


def decode_rows(op, vals, nd, words):
    """Store rows (values) -> the integer rows `want_rows` gives (digits mod 4)."""
    ndig, nfl, sc = out_layout(op, nd)
    x = np.rint(np.asarray(vals, dtype=np.float64)[:words * ndig + words * nfl] * sc).astype(np.int64)
    if ndig:
        x[:words * ndig] = np.mod(x[:words * ndig], 4)
    return x


def inputs(words, bits, seed=0, op=None):
    """Random words with the edge cases first: 0, -1, MIN, MAX, MIN - 1, equal pairs, a + b = -1 (the longest carry
    chain); shift amounts 0, 1, bits - 1, bits, bits + 1, 2^bits - 1 and random below 2 bits."""
    rng = np.random.default_rng(seed)
    top = 1 << bits
    edge = np.array([0, top - 1, top >> 1, (top >> 1) - 1, 1, top - 2], dtype=np.int64)
    a = rng.integers(0, top, words, dtype=np.int64)
    b = rng.integers(0, top, words, dtype=np.int64)
    sel = rng.integers(0, 2, words).astype(np.int64)
    ne = len(edge)
    pairs = [(x, y) for x in edge for y in edge]
    for i, (x, y) in enumerate(pairs[:max(0, words // 2)]):
        a[i], b[i] = x, y
    q = min(words, len(pairs)) if words > 1 else 0
    if q < words:
        nq = (words - q) // 4
        b[q:q + nq] = a[q:q + nq]                                  # equal
        b[q + nq:q + 2 * nq] = (~a[q + nq:q + 2 * nq]) & (top - 1)   # a + b = -1
    base = parse(op)[0] if op else ""
    if base.endswith("v"):                                         # shift amounts
        amts = np.array([0, 1, bits - 1, bits, bits + 1, top - 1, 2 * bits, bits // 2], dtype=np.int64)
        b = rng.integers(0, 2 * bits, words, dtype=np.int64)
        b[:min(words, len(amts))] = amts[:min(words, len(amts))]
        if words > len(amts):
            b[len(amts):2 * len(amts)] = amts[:max(0, min(len(amts), words - len(amts)))]
    return a, b, sel


def slot_values(a, b, sel, nd):
    """The loaded slot set's values (a's digits, b's, sel) as the CKKS input encodes them."""
    return np.concatenate([encode_digits(a, nd), encode_digits(b, nd), (4.0 * np.asarray(sel) - 3.0) / 4.0])


def case(op, bits, words, seed=0, nu=None):
    """A pc-check tfhe-prog case: the program, its loaded native values, the exact store rows, the analysis."""
    import json
    nd = bits // 2
    a, b, sel = inputs(words, bits, seed, op)
    wp = build(op, f"c.{op}{bits}.w{words}", words, nd)
    m = load_values(slot_values(a, b, sel, nd))
    want, ndig, nfl, w = want_rows(op, a, b, sel, nd)
    an = analyze(wp.pr, m, nu=nu)
    prog = json.loads(wp.pr.json())
    return {"name": f"{op}{bits} x {words}", "op": op, "bits": bits, "words": words, "prog": prog,
            "bufs": wp.pr.bufs, "rows": max(rows_needed(wp.pr, len(m)), len(m)), "m_in": [float(v) for v in m],
            "want": [int(v) for v in want], "scale": out_layout(op, nd)[2], "ndig": ndig, "nflag": nfl,
            "analysis": an}


def rows_needed(prog, rows_in):
    """The largest buffer a program writes (rows)."""
    r = rows_in
    for st in prog.steps:
        if st["op"] == "lin":
            r = max(r, len(st["idx"]) // 2)
        elif st["op"] in ("pbs", "store"):
            r = max(r, st["rows"])
    return r


def mirror(op, bits, words, seed=0, noise=None):
    """Run `op` on the numpy mirror (with the native noise model unless noise=0): (got rows, want rows, analysis)."""
    nd = bits // 2
    a, b, sel = inputs(words, bits, seed, op)
    wp = build(op, f"m.{op}{bits}", words, nd)
    m = slot_values(a, b, sel, nd)
    out = wp.pr.run(m, noise=LOOKUP_SD[1] if noise is None else noise)
    want = want_rows(op, a, b, sel, nd)[0]
    return decode_rows(op, out, nd, words), want, analyze(wp.pr, load_values(m))


def main(argv=None):
    """python -m perseus.impl.tfhe_bitwise --cases FILE --ops add,sub --bits 8,16,32 --words 1,16,256: write the
    pc-check tfhe-prog cases (JSON lines)."""
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--ops", required=True)
    ap.add_argument("--bits", default="8,16,32")
    ap.add_argument("--words", default="1,16,256")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    with open(a.cases, "w") as fh:
        for op in a.ops.split(","):
            for bits in map(int, a.bits.split(",")):
                for words in map(int, a.words.split(",")):
                    c = case(op, bits, words, a.seed)
                    fh.write(json.dumps(c) + "\n")
                    an = c["analysis"]
                    print(f"{c['name']}: {an['rounds']} rounds, {an['pbs']} PBS, {an['launches']} launches, p_fail "
                          f"2^{an['log2_pfail_lookup']:.1f} / lookup, 2^{an['log2_pfail_op']:.1f} / op")


if __name__ == "__main__":
    main()
