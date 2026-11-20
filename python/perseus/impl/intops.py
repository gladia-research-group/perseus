"""Exact small-integer arithmetic by chained functional-bootstrap lookups (impl/fbs.py tables).

A word of `bits` bits is `n = bits / r` radix-B digits (B = 2^r), least significant first, one ciphertext per digit
position (every slot an independent word): digit-position ops are slot-wise, carries move between ciphertexts, never
across slots. Every lookup is one FBS (`fbs.register_lut`): its input is an integer combination of digits and earlier
lookup outputs (sums and integer multiples cost no lookup), its output lands on the integer grid with the model's
noise. Lookup outputs feed the next lookup directly (no plain refresh) when the FBS lands within the StC-first input
cap (`fbs.chain_slack`).

- `add(a, b)`: ripple carry, one carry lookup per digit (table on s = a_i + b_i + c in [0, 2B - 1]); the digit is
  s - B c (linear). n lookups per word, n sequential.
- `cmp(a, b)`: digit compare a_i + B b_i -> {0: a < b, 1: =, 2: >} (n lookups, independent), then the ripple combine
  e_i + 3 s -> (e_i if e_i != 1 else s) (n - 1 lookups, the last one emitting [a < b]).
- `mux(sel, x, y)`: y + sel (x - y) per digit (a product, no lookup).
- `bitwise(op, a, b)`: and / or / xor per digit, a bivariate table on a_i + B b_i.
- `bivariate(name, fn, a, b, B)`: any table of two digits through the packed index a + B b.

Exactness: a lookup is wrong when its input noise crosses half the integer spacing or its output noise reaches half
the output spacing; both are Gaussian in the FBS variance model and far below 2^-128 at the default input noise
(`fbs.p_fail`); the ModRaise overflow (exact, 2^-291.1 per bootstrap at K = 32, h = 32) dominates. `chain_p_fail`
gives the union bound of a chain of lookups.
"""
from __future__ import annotations

import math

import numpy as np

from . import fbs


def _lut(name, fn, lo, hi, table="k32"):
    return fbs.register_lut(name, fn, lo, hi, table=table)


class IntOps:
    """Radix-2^r digit arithmetic on `ops` (a planned session's ops, NumpyOps for the mirror). `table`: the FBS
    EvalMod table (`fbs.TABLES`)."""

    def __init__(self, ops, r=2, table="k32"):
        self.ops, self.r, self.base, self.table = ops, int(r), 1 << int(r), table
        b = self.base
        tag = f"r{self.r}.{table}"
        self.carry = _lut(f"int.carry.{tag}", lambda s: (s >= b).astype(float), 0, 2 * b - 1, table)
        self.dcmp = _lut(f"int.cmp.{tag}", lambda j: np.sign(j % b - j // b) + 1.0, 0, b * b - 1, table)
        self.comb = _lut(f"int.comb.{tag}", lambda j: np.where(j % 3 == 1, j // 3, j % 3).astype(float), 0, 8, table)
        self.lt = _lut(f"int.lt.{tag}", lambda j: (np.where(j % 3 == 1, j // 3, j % 3) == 0).astype(float), 0, 8,
                       table)
        self.logic = {}

    def lookups(self):
        """The registered function names (for p_fail)."""
        return [self.carry, self.dcmp, self.comb, self.lt] + list(self.logic.values())

    def _f(self, x, name):
        return fbs.fbs(self.ops, x, name)

    def add(self, a, b):
        """(a + b) mod B^n, digits least significant first."""
        o, out, c = self.ops, [], None
        for i, (ai, bi) in enumerate(zip(a, b)):
            s = o.add(ai, bi)
            if c is not None:
                s = o.add(s, c)
            if i == len(a) - 1:              # the last digit: its carry leaves the word
                c = self._f(o.copy(s), self.carry)
                out.append(o.sub(s, o.mult(c, float(self.base))))
                break
            c = self._f(o.copy(s), self.carry)
            out.append(o.sub(s, o.mult(c, float(self.base))))
        return out

    def cmp(self, a, b):
        """[a < b] (0 / 1) for words a, b."""
        o = self.ops
        e = [self._f(o.add(ai, o.mult(bi, float(self.base))), self.dcmp) for ai, bi in zip(a, b)]
        st = e[0]
        for i in range(1, len(e)):
            j = o.add(e[i], o.mult(st, 3.0))
            st = self._f(j, self.lt if i == len(e) - 1 else self.comb)
        if len(e) == 1:
            st = self._f(o.mult(st, 3.0), self.lt)     # j = 3 s + 0 is never the "=" branch: e = s
        return st

    def mux(self, sel, x, y):
        """sel ? x : y per digit (sel 0 / 1, one ciphertext)."""
        o = self.ops
        return [o.add(yi, o.mult(o.sub(xi, yi), sel)) for xi, yi in zip(x, y)]

    def bitwise(self, op, a, b):
        """and / or / xor of the digits."""
        if op not in self.logic:
            f = {"and": np.bitwise_and, "or": np.bitwise_or, "xor": np.bitwise_xor}[op]
            bb = self.base
            self.logic[op] = _lut(f"int.{op}.r{self.r}.{self.table}",
                                  lambda j, f=f, bb=bb: f(j % bb, j // bb).astype(float), 0, bb * bb - 1, self.table)
        o = self.ops
        return [self._f(o.add(ai, o.mult(bi, float(self.base))), self.logic[op]) for ai, bi in zip(a, b)]


def digits(x, bits, r):
    """Integers -> radix-2^r digit arrays, least significant first."""
    x = np.asarray(x, dtype=np.int64)
    return [((x >> (r * i)) & ((1 << r) - 1)).astype(float) for i in range(bits // r)]


def value(ds, r):
    """Digit arrays (rounded) -> integers."""
    return sum(np.round(np.real(d)).astype(np.int64) << (r * i) for i, d in enumerate(ds))


def chain_p_fail(names, counts, sigma_in=None):
    """log2 of the union bound over a chain: sum over lookups `names` of `counts` x p_fail per ciphertext."""
    tot = -math.inf
    for n, c in zip(names, counts):
        r = fbs.p_fail(n, sigma_in=sigma_in)
        tot = fbs._log2_add(tot, r["log2_ct"] + math.log2(c))
    return tot


class WordOps:
    """Words with their n = bits / r radix-2^r digits in n consecutive slots of ONE ciphertext (lane k = slots
    k n .. k n + n - 1, least significant first; slots / n lanes): every lookup of an op runs once for all digits and
    lanes, carries resolve by a parallel prefix (Kogge-Stone) over the digit slots with rotations, log2 n rounds.

    - `add(a, b)`: s = a + b; carry states {0 kill, 1 propagate (s = B - 1), 2 generate (s >= B)} combined by a
      parallel prefix st_i <- comb(st_i, st_{i-d}) (st_i unless it propagates; digits below a lane's 0 take kill),
      the first round merged with the states (a lookup on (s_i, s_{i-1})) and the last with the output digit (a lookup
      on (s_i, st_{i-1}, st_{i-1-d})): log2 n sequential lookups (32-bit radix 4: 4). `add_basic`: unmerged,
      log2 n + 2.
    - `cmp(a, b)`: digit compare a_i + B b_i -> {0 <, 1 =, 2 >}; log2 n rounds of the same combine (the higher digit
      wins unless equal), the last round emitting [a < b] (read at the lane's top digit slot). log2 n + 1 lookups.
    - `mux(sel, x, y)`: y + sel (x - y), sel 0 / 1 in every digit slot of its lane (one product, no lookup).
    - `bitwise(op, a, b)`: one bivariate lookup on a + B b.
    """

    def __init__(self, ops, bits, r=2, table="k28d1", slots=None):
        self.ops, self.r, self.base, self.table = ops, int(r), 1 << int(r), table
        self.n = int(bits) // self.r
        if self.n & (self.n - 1):
            raise ValueError(f"WordOps: {self.n} digits per word; a power of two is supported")
        self.slots = int(slots) if slots is not None else 1 << 15
        b, tag = self.base, f"r{self.r}.{table}"
        self.state = _lut(f"word.state.{tag}", lambda s: np.where(s >= b, 2.0, np.where(s == b - 1, 1.0, 0.0)), 0,
                          2 * b - 2, table)
        # combine: j = hi + 3 lo -> hi unless hi == 1 (propagate / equal), then lo
        self.comb = _lut(f"word.comb.{tag}", lambda j: np.where(j % 3 == 1, j // 3, j % 3).astype(float), 0, 8, table)
        self.lt = _lut(f"word.lt.{tag}", lambda j: (np.where(j % 3 == 1, j // 3, j % 3) == 0).astype(float), 0, 8,
                       table)
        # digit: j = s + (2B - 1) c, c = [st_{i-1} == 2] folded as st in {0, 1, 2}: j = s + (2B - 1) st
        w = 2 * b - 1
        self.digit = _lut(f"word.digit.{tag}", lambda j: ((j % w + (j // w == 2)) % b).astype(float), 0, 3 * w - 1,
                          table)
        self.dcmp = _lut(f"word.cmp.{tag}", lambda j: np.sign(j % b - j // b) + 1.0, 0, b * b - 1, table)
        self.logic = {}

    def lookups(self):
        self._merged()
        return [self.state, self.comb, self.lt, self.digit, self.dcmp, self.pair, self.final] + list(self.logic.values())

    def _mask(self, d):
        """1 on digit slots i >= d of every lane, 0 below."""
        return ((np.arange(self.slots) % self.n) >= d).astype(np.float64)

    def _shift(self, x, d, fill=0.0):
        """x moved up d digit slots inside each lane, `fill` below (rotation + mask: one level)."""
        o = self.ops
        y = o.mult_pt(o.rotate(x, -d), self._mask(d), kind="mask")
        return y if fill == 0.0 else o.add_pt(y, fill * (1.0 - self._mask(d)), kind="mask")

    def _prefix(self, st, last_lt=False, fill=0.0):
        """`fill`: the state below a lane's digit 0 (add: kill, cmp: equal)."""
        o, d = self.ops, 1
        while d < self.n:
            last = d * 2 >= self.n
            j = o.add(st, o.mult(self._shift(st, d, fill), 3.0))
            st = fbs.fbs(o, j, self.lt if (last and last_lt) else self.comb)
            d *= 2
        return st

    def add_basic(self, a, b):
        """`add` without the merged first / last rounds: log2 n + 2 lookups."""
        o = self.ops
        s = o.add(a, b)
        st = self._prefix(fbs.fbs(o, o.copy(s), self.state))
        j = o.add(s, o.mult(self._shift(st, 1), float(2 * self.base - 1)))
        return fbs.fbs(o, j, self.digit)

    def _merged(self):
        """The two merged tables of `add`: pair (s_i, s_{i-1}) -> comb(state s_i, state s_{i-1}); final
        (s_i, st_{i-1}, st_{i-1-d}) -> (s_i + [comb(st_{i-1}, st_{i-1-d}) = 2]) mod B."""
        if not hasattr(self, "pair"):
            b, w, tag = self.base, 2 * self.base - 1, f"r{self.r}.{self.table}"

            def stf(x):
                return np.where(x >= b, 2, np.where(x == b - 1, 1, 0))

            def comb(hi, lo):
                return np.where(hi == 1, lo, hi)
            self.pair = _lut(f"word.pair.{tag}", lambda j: comb(stf(j % w), stf(j // w)).astype(float), 0,
                             w * w - 1, self.table)
            self.final = _lut(f"word.final.{tag}",
                              lambda j: ((j % w + (comb((j // w) % 3, j // (3 * w)) == 2)) % b).astype(float), 0,
                              9 * w - 1, self.table)
        return self.pair, self.final

    def add(self, a, b):
        """(a + b) mod 2^bits per lane: the pair lookup (first round merged with the digit states), the middle
        combine rounds d = 2 .. n/4, the final lookup (last round merged with the digit): log2 n lookups for n >= 4
        (32-bit radix 4: 4)."""
        if self.n < 4 or 9 * (2 * self.base - 1) > 64:      # radix 4 only: the final table must stay t <= 64
            return self.add_basic(a, b)
        o, w = self.ops, float(2 * self.base - 1)
        pair, final = self._merged()
        s = o.add(a, b)
        st = fbs.fbs(o, o.add(s, o.mult(self._shift(s, 1), w)), pair)
        d = 2
        while d < self.n // 2:
            st = fbs.fbs(o, o.add(st, o.mult(self._shift(st, d), 3.0)), self.comb)
            d *= 2
        j = o.add(s, o.add(o.mult(self._shift(st, 1), w), o.mult(self._shift(st, 1 + d), 3.0 * w)))
        return fbs.fbs(o, j, final)

    def cmp(self, a, b):
        o = self.ops
        e = fbs.fbs(o, o.add(a, o.mult(b, float(self.base))), self.dcmp)
        if self.n == 1:
            return fbs.fbs(o, o.mult(e, 3.0), self.lt)
        return self._prefix(e, last_lt=True, fill=1.0)

    def mux(self, sel, x, y):
        o = self.ops
        return o.add(y, o.mult(o.sub(x, y), sel))

    def bitwise(self, op, a, b):
        if op not in self.logic:
            f = {"and": np.bitwise_and, "or": np.bitwise_or, "xor": np.bitwise_xor}[op]
            bb = self.base
            self.logic[op] = _lut(f"word.{op}.r{self.r}.{self.table}",
                                  lambda j, f=f, bb=bb: f(j % bb, j // bb).astype(float), 0, bb * bb - 1, self.table)
        o = self.ops
        if a is None:
            return None
        return fbs.fbs(o, o.add(a, o.mult(b, float(self.base))), self.logic[op])

    def encode(self, x, bits=None):
        """Integers (one per lane, slots / n of them) -> the slot vector of their digits."""
        ds = digits(x, self.n * self.r, self.r)
        return np.stack(ds, 1).reshape(-1)

    def decode(self, v):
        """Slot vector -> integers per lane."""
        d = np.round(np.real(np.asarray(v))).astype(np.int64).reshape(-1, self.n)
        return sum(d[:, i] << (self.r * i) for i in range(self.n))
