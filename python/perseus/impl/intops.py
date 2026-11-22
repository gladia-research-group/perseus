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

The other integer / bitwise ops, in both layouts (`IntOps`: digit lists; `WordOps`: one ciphertext per word set, the
digit slots of a lane; `word_op` / `int_op` dispatch by name, `reference` gives the exact results):
- sub / neg: a + ~b + 1 / ~a + 1 (~d = B - 1 - d is linear; the carry-in rides the adder: the first carry's input
  (IntOps) or the prefix's fills, a generate below digit 0 (WordOps)). `flags=True`: also the carry / borrow and the
  signed overflow (one lookup on the top digits a + B b + B^2 r; WordOps: at the lane's top digit slot).
- compare predicates lt / le / gt / ge / eq / ne, unsigned or signed (the top digit's own compare table; WordOps
  through a slot offset B^2 at the top slot, one table of 2 B^2), the last combine's table reading the predicate.
- sext(from_bits); shifts and rotates by a public amount: whole digits by renumbering (IntOps) / a move (rotation +
  mask, WordOps), the bits within a digit by one bivariate lookup on the digit and its neighbour (sar: the sign from
  one lookup fills); >= the width: 0 / the sign.
- shifts and rotates by an encrypted amount (a word): a barrel shifter, one lookup per amount bit on
  x + B x_s + B^2 b_digit (+ B^3 big); amounts >= the width give 0 (the last stage's `big`, a weighted lane sum of the
  amount's high digits through one threshold lookup) or the sign (sar: `big` forces every stage). WordOps folds the
  integer factors into the move / broadcast masks: no level of their own, the stages chain without plain refreshes.
- mul: the column sums S_i = sum_j a_(i - j) b_j by CKKS products (no lookup; WordOps broadcasts b's digits over the
  lane), then `_norm`: S's base-B digits (plain tables: S is a fresh product sum, one hop), the per-column sums T
  (<= B^2 - 1), one split T mod B / T div B, the add. mulc: the same on scalar multiples (no products).
- min / max (signed or not): the compare, then the mux; abs: the sign muxes -a and a; popcount: the digits' bit
  counts, their sum (WordOps: rotations to the lane's top slot), its digits.

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
        return [self.carry, self.dcmp, self.comb, self.lt] + list(self.logic.values()) + list(getattr(self, "_more", []))

    def lookup_sigmas(self):
        """name -> the input noise std its p_fail is evaluated at (None: fbs.SIGMA_IN)."""
        return {n: getattr(self, "_sig", {}).get(n) for n in self.lookups()}

    def _f(self, x, name):
        return fbs.fbs(self.ops, x, name)

    def add(self, a, b, cin=0, flags=False):
        """(a + b + cin) mod B^n, digits least significant first (b None: a alone; cin a public 0 / 1). `flags`:
        also (carry out, signed overflow) as ciphertexts."""
        out, c = self._ripple(a, b, cin)
        return (out, c, self._ovf(a, b, out)) if flags else out

    def _ripple(self, a, b, cin=0):
        o, out, c = self.ops, [], None
        for i, ai in enumerate(a):
            s = ai if b is None else o.add(ai, b[i])
            if c is not None:
                s = o.add(s, c)
            elif cin:
                s = o.add(s, float(cin))
            c = self._f(o.copy(s), self.carry)
            out.append(o.sub(s, o.mult(c, float(self.base))))
        return out, c

    def _ovf(self, a, b, r):
        """Signed overflow of r = a + b (+ cin): one lookup on the top digits a + B b + B^2 r."""
        o, B = self.ops, self.base
        t = _lut(f"int.ovf.r{self.r}.{self.table}", lambda j: (((j % B >= B // 2) == ((j // B) % B >= B // 2)) &
                                                              ((j // (B * B) >= B // 2) != (j % B >= B // 2))).astype(float),
                 0, B ** 3 - 1, self.table)
        self._extra(t)
        return self._f(o.add(a[-1], o.add(o.mult(b[-1], float(B)), o.mult(r[-1], float(B * B)))), t)

    def _extra(self, name):
        if not hasattr(self, "_more"):
            self._more = []
        if name not in self._more:
            self._more.append(name)
        return name

    def _not(self, a):
        o = self.ops
        return [o.add(o.mult(x, -1.0), float(self.base - 1)) for x in a]

    def sub(self, a, b, flags=False):
        """(a - b) mod B^n = a + ~b + 1; `flags`: also (borrow, signed overflow)."""
        nb = self._not(b)
        out, c = self._ripple(a, nb, 1)
        if not flags:
            return out
        return out, self.ops.add(self.ops.mult(c, -1.0), 1.0), self._ovf(a, nb, out)

    def neg(self, a):
        """(-a) mod B^n = ~a + 1."""
        return self._ripple(self._not(a), None, 1)[0]

    def cmp_pred(self, a, b, pred="lt", signed=False):
        """[a pred b] (lt / le / gt / ge / eq / ne; signed: the top digit compared as a signed digit)."""
        o, B = self.ops, self.base
        e = [self._f(o.add(ai, o.mult(bi, float(B))), self.dcmp) for ai, bi in zip(a[:-1], b[:-1])]
        if signed:
            def f(j):
                x, y = j % B, j // B
                return np.sign(np.where(x >= B // 2, x - B, x) - np.where(y >= B // 2, y - B, y)) + 1.0
            st = self._extra(_lut(f"int.scmp.r{self.r}.{self.table}", f, 0, B * B - 1, self.table))
        else:
            st = self.dcmp
        e.append(self._f(o.add(a[-1], o.mult(b[-1], float(B))), st))
        ok = {"lt": (0,), "le": (0, 1), "gt": (2,), "ge": (1, 2), "eq": (1,), "ne": (0, 2)}[pred]
        last = self.lt if pred == "lt" else self._extra(_lut(
            f"int.p{pred}.r{self.r}.{self.table}",
            lambda j, k=ok: np.isin(np.where(j % 3 == 1, j // 3, j % 3), k).astype(float), 0, 8, self.table))
        cur = e[0]
        for i in range(1, len(e)):
            cur = self._f(o.add(e[i], o.mult(cur, 3.0)), last if i == len(e) - 1 else self.comb)
        if len(e) == 1:
            cur = self._f(o.mult(cur, 3.0), last)
        return cur

    def sext(self, a, from_bits):
        """Sign extension of the low `from_bits` bits (a multiple of r)."""
        B, m = self.base, int(from_bits) // self.r
        if from_bits % self.r or not 0 < m <= len(a):
            raise ValueError("IntOps sext: from_bits a multiple of r, at most the width")
        if m == len(a):
            return list(a)
        t = self._extra(_lut(f"int.sx.r{self.r}.{self.table}", lambda d: (B - 1.0) * (d >= B // 2), 0, B - 1, self.table))
        sg = self._f(self.ops.copy(a[m - 1]), t)
        return list(a[:m]) + [sg] * (len(a) - m)

    def _bit(self, kind, t, fill_sign=False):
        """Within-digit shift by t bits on j = d + B nb (nb the neighbour digit moving in)."""
        B, r = self.base, self.r

        def f(j):
            d, nb = j % B, j // B
            if kind in ("shl", "rotl"):
                return (((d << t) | (nb >> (r - t))) & (B - 1)).astype(float)
            return ((d >> t) | ((nb << (r - t)) & (B - 1))).astype(float)
        return self._extra(_lut(f"int.b{kind}{t}.r{self.r}.{self.table}", f, 0, B * B - 1, self.table))

    def shift(self, a, kind, k):
        """a shifted / rotated by the public amount k (shl / shr / sar / rotl / rotr): digits renumbered, the bits
        within a digit one bivariate lookup per digit (independent); >= the width: 0 / the sign."""
        o, B, n, r = self.ops, self.base, len(a), self.r
        bits = n * r
        k = int(k) % bits if kind.startswith("rot") else min(int(k), bits)
        zero = o.mult(a[0], 0.0)
        sg = None
        if kind == "sar":                    # (B - 1) x the sign: the fill digit
            t = self._extra(_lut(f"int.sx.r{self.r}.{self.table}", lambda d: (B - 1.0) * (d >= B // 2), 0, B - 1,
                                 self.table))
            sg = self._f(o.copy(a[-1]), t)
        if k >= bits:
            return [sg if kind == "sar" else zero] * n
        q, t = divmod(k, r)
        fill = sg if kind == "sar" else zero

        def src(i, s, cyc):
            j = i - s if kind in ("shl", "rotl") else i + s
            return a_[j % n] if cyc else (a_[j] if 0 <= j < n else fill)
        a_ = list(a)
        cyc = kind.startswith("rot")
        if t:
            tab = self._bit(kind, t)
            a_ = [self._f(o.add(a_[i], o.mult(src(i, 1, cyc), float(B))), tab) for i in range(n)]
        return [src(i, q, cyc) for i in range(n)] if q else a_

    def shift_enc(self, a, b, kind):
        """a shifted / rotated by the encrypted amount b (a word: its digits are ciphertexts): one stage per amount bit
        k, every digit one lookup on x_i + B x_s + B^2 b_digit (+ B^3 big); >= the width: 0 / the sign (sar: every
        select forced); rotates take b mod the width."""
        o, B, n, r = self.ops, self.base, len(a), self.r
        bits = n * r
        L = bits.bit_length() - 1
        rot = kind.startswith("rot")
        big = sg = None
        if not rot:
            q0, p = divmod(L, r)
            T = 1 << p
            z = None
            for i in range(q0, n):
                t_ = b[i] if i == q0 else o.mult(b[i], float(T))
                z = t_ if z is None else o.add(z, t_)
            mx = int(T * (B - 1) * (n - 1 - q0) + (B - 1))
            big = self._f(z, self._extra(_lut(f"int.big{L}.{mx}.r{self.r}.{self.table}", lambda v, T=T: (v >= T).astype(float),
                                              0, mx, self.table)))
        if kind == "sar":
            t_ = self._extra(_lut(f"int.sx.r{self.r}.{self.table}", lambda d: (B - 1.0) * (d >= B // 2), 0, B - 1,
                                  self.table))
            sg = self._f(o.copy(a[-1]), t_)
        zero = o.mult(a[0], 0.0)
        x = list(a)
        for k in range(L):
            sh, bp = 1 << k, k % r
            use_big = big is not None and (kind == "sar" or k == L - 1)
            t_bits = sh if sh < r else 0
            s = 1 if t_bits else sh // r

            def f(j, t_bits=t_bits, bp=bp, use_big=use_big):
                xx, xs, dg, bg = j % B, (j // B) % B, (j // (B * B)) % B, j // (B ** 3)
                sel = ((dg >> bp) & 1).astype(bool)
                if use_big and kind == "sar":
                    sel = sel | (bg == 1)
                if t_bits:
                    v = ((((xx << t_bits) | (xs >> (r - t_bits))) & (B - 1)) if kind in ("shl", "rotl")
                         else ((xx >> t_bits) | ((xs << (r - t_bits)) & (B - 1))))
                else:
                    v = xs
                out = np.where(sel, v, xx)
                if use_big and kind != "sar":
                    out = np.where(bg == 1, 0, out)
                return out.astype(float)
            tab = self._extra(_lut(f"int.bs{kind}{sh}{'b' if use_big else ''}.r{self.r}.{self.table}", f, 0,
                                   (2 if use_big else 1) * B ** 3 - 1, self.table))
            fill = sg if kind == "sar" else zero
            nx = []
            for i in range(n):
                j_ = i - s if kind in ("shl", "rotl") else i + s
                xs = x[j_ % n] if rot else (x[j_] if 0 <= j_ < n else fill)
                j = o.add(x[i], o.add(o.mult(xs, float(B)), o.mult(b[k // r], float(B * B))))
                if use_big:
                    j = o.add(j, o.mult(big, float(B ** 3)))
                nx.append(self._f(j, tab))
            x = nx
        return x

    def _norm(self, S, smax):
        """Column sums S (column i at most smax[i]) -> the word: S's base-B digits (lookups), the per-column sums T,
        one split (T mod B, T div B), the ripple add."""
        o, B, r, n = self.ops, self.base, self.r, len(S)
        M = max(smax)
        if M <= B - 1:
            return list(S)
        K = 1
        while B ** K <= M:
            K += 1
        T = [None] * n
        for i in range(n):
            if smax[i] == 0:
                continue
            for k in range(K):
                if i + k >= n or (M >> (r * k)) == 0:
                    continue
                d = self._f(o.copy(S[i]), self._ndig(M, k))
                T[i + k] = d if T[i + k] is None else o.add(T[i + k], d)
        zero = o.mult(S[0], 0.0)
        T = [t if t is not None else zero for t in T]
        tmax = sum(min(B - 1, M >> (r * k)) for k in range(K))
        if tmax <= B - 1:
            return T
        lo = self._extra(_lut(f"int.t{tmax}lo.r{self.r}.{self.table}", lambda v: (v % B).astype(float), 0, tmax, self.table))
        hi = self._extra(_lut(f"int.t{tmax}hi.r{self.r}.{self.table}", lambda v: (v // B).astype(float), 0, tmax, self.table))
        e0 = [self._f(o.copy(t), lo) for t in T]
        e1 = [self._f(o.copy(t), hi) for t in T[:-1]]
        return self.add(e0, [zero] + e1)

    def _ndig(self, M, k):
        B, r = self.base, self.r
        name = f"int.nd{M}.{k}.r{self.r}.{self.table}.plain"
        if name not in fbs.FUNCS:
            fbs.register_lut(name, lambda v: ((v >> (r * k)) & (B - 1)).astype(float), 0, M, table=self.table,
                             flat=False, sigma_in=2.0 ** -14)
        self._extra(name)
        self._sig = getattr(self, "_sig", {})
        self._sig[name] = 2.0 ** -14
        return name

    def mul(self, a, b):
        """a b mod B^n: column sums by CKKS products (n (n + 1) / 2 of them, no lookup), then `_norm`."""
        o, n, B = self.ops, len(a), self.base
        S = []
        for i in range(n):
            t = None
            for j in range(i + 1):
                p = o.mult(a[i - j], b[j])
                t = p if t is None else o.add(t, p)
            S.append(t)
        return self._norm(S, [(B - 1) ** 2 * (i + 1) for i in range(n)])

    def mulc(self, a, c):
        """a c mod B^n for a public c (scalar products, then `_norm`)."""
        o, n, B, r = self.ops, len(a), self.base, self.r
        c = int(c) % (1 << (n * r))
        cd = [(c >> (r * j)) & (B - 1) for j in range(n)]
        S = []
        for i in range(n):
            t = None
            for j in range(i + 1):
                if cd[j]:
                    p = a[i - j] if cd[j] == 1 else o.mult(a[i - j], float(cd[j]))
                    t = p if t is None else o.add(t, p)
            S.append(t if t is not None else o.mult(a[0], 0.0))
        return self._norm(S, [(B - 1) * sum(cd[:i + 1]) for i in range(n)])

    def minmax(self, a, b, kind="min", signed=False):
        s = self.cmp_pred(a, b, "lt", signed)
        return self.mux(s, a, b) if kind == "min" else self.mux(s, b, a)

    def abs(self, a):
        B = self.base
        t = self._extra(_lut(f"int.sg1.r{self.r}.{self.table}", lambda d: (d >= B // 2).astype(float), 0, B - 1, self.table))
        return self.mux(self._f(self.ops.copy(a[-1]), t), self.neg(a), a)

    def popcount(self, a):
        """The set bits as a word: the digits' bit counts (one lookup each), their sum, its digits."""
        o, B, r, n = self.ops, self.base, self.r, len(a)
        t = self._extra(_lut(f"int.pc.r{self.r}.{self.table}",
                             lambda d: np.array([bin(int(v)).count("1") for v in np.ravel(d)], dtype=float).reshape(np.shape(d)),
                             0, B - 1, self.table))
        z = None
        for x in a:
            y = self._f(o.copy(x), t)
            z = y if z is None else o.add(z, y)
        M = r * n
        out, k = [], 0
        while (M >> (r * k)) > 0 and k < n:
            out.append(self._f(o.copy(z), self._extra(_lut(f"int.pcd{M}.{k}.r{self.r}.{self.table}",
                                                            lambda v, k=k: ((v >> (r * k)) & (B - 1)).astype(float), 0, M,
                                                            self.table))))
            k += 1
        zero = o.mult(a[0], 0.0)
        return out + [zero] * (n - len(out))

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
        """Every table this instance used (for p_fail); `lookup_sigmas` gives each one's input noise std."""
        self._merged()
        return ([self.state, self.comb, self.lt, self.digit, self.dcmp, self.pair, self.final] +
                list(self.logic.values()) + list(getattr(self, "_names", {})))

    def lookup_sigmas(self):
        """name -> the input noise std its p_fail is evaluated at (None: fbs.SIGMA_IN)."""
        return {n: getattr(self, "_names", {}).get(n) for n in self.lookups()}

    def _mask(self, d):
        """1 on digit slots i >= d of every lane, 0 below."""
        return ((np.arange(self.slots) % self.n) >= d).astype(np.float64)

    def _shift(self, x, d, fill=0.0):
        """x moved up d digit slots inside each lane, `fill` below (rotation + mask: one level)."""
        o = self.ops
        y = o.mult_pt(o.rotate(x, -d), self._mask(d), kind="mask")
        return y if fill == 0.0 else o.add_pt(y, fill * (1.0 - self._mask(d)), kind="mask")

    def _prefix(self, st, last_lt=False, fill=0.0, last=None):
        """`fill`: the state below a lane's digit 0 (add: kill, or generate for a carry-in; cmp: equal); `last`: the
        last round's table (default comb, or lt when last_lt)."""
        o, d = self.ops, 1
        lt = last if last is not None else (self.lt if last_lt else self.comb)
        while d < self.n:
            fin = d * 2 >= self.n
            j = o.add(st, o.mult(self._shift(st, d, fill), 3.0))
            st = fbs.fbs(o, j, lt if fin else self.comb)
            d *= 2
        return st

    def add_basic(self, a, b, cin=0):
        """`add` without the merged first / last rounds: log2 n + 2 lookups."""
        return self._add_basic_s(self.ops.add(a, b), cin)

    def _add_basic_s(self, s, cin=0):
        o, fst = self.ops, 2.0 if cin else 0.0
        st = self._prefix(fbs.fbs(o, o.copy(s), self.state), fill=fst)
        j = o.add(s, o.mult(self._shift(st, 1, fst), float(2 * self.base - 1)))
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

    def add(self, a, b, flags=False):
        """(a + b) mod 2^bits per lane: the pair lookup (first round merged with the digit states), the middle
        combine rounds d = 2 .. n/4, the final lookup (last round merged with the digit): log2 n lookups for n >= 4
        (32-bit radix 4: 4). `flags`: also (carry, signed overflow), 0 / 1 at the lane's top digit slot."""
        r = self._add_s(self.ops.add(a, b))
        return (r,) + self._flags(a, b, r) if flags else r

    def _add_s(self, s, cin=0):
        """The add on digit sums s (0 .. 2B - 2 per slot) with a public carry-in (the digit below a lane's 0 then
        generates: the pair table's s_(-1) = B, the shifted states' fill 2)."""
        if self.n < 4 or 9 * (2 * self.base - 1) > 64:      # radix 4 only: the final table must stay t <= 64
            return self._add_basic_s(s, cin)
        o, w = self.ops, float(2 * self.base - 1)
        fs, fst = (float(self.base), 2.0) if cin else (0.0, 0.0)
        pair, final = self._merged()
        st = fbs.fbs(o, o.add(s, o.mult(self._shift(s, 1, fs), w)), pair)
        d = 2
        while d < self.n // 2:
            st = fbs.fbs(o, o.add(st, o.mult(self._shift(st, d, fst), 3.0)), self.comb)
            d *= 2
        j = o.add(s, o.add(o.mult(self._shift(st, 1, fst), w), o.mult(self._shift(st, 1 + d, fst), 3.0 * w)))
        return fbs.fbs(o, j, final)

    def cmp(self, a, b, pred="lt", signed=False):
        """[a pred b] (pred lt / le / gt / ge / eq / ne; signed: two's complement) at the lane's top digit slot: the
        digit compare (signed: the top digit's own table, through a slot offset B^2 at the top slot: one lookup on
        t = 2 B^2), log2 n combine rounds, the last one emitting the predicate."""
        o, B = self.ops, self.base
        x = o.add(a, o.mult(b, float(B)))
        if signed:
            x = o.add_pt(x, float(B * B) * self._lane(lambda i: i == self.n - 1), kind="mask")
        e = fbs.fbs(o, x, self._sdcmp() if signed else self.dcmp)
        last = self._pred(pred)
        if self.n == 1:
            return fbs.fbs(o, o.mult(e, 3.0), last)
        return self._prefix(e, fill=1.0, last=last)

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

    # ── the other integer / bitwise ops ─────────────────────────────────────────────────────────────────────────────

    def _tab(self, key, fn, lo, hi, flat=True, sigma_in=None):
        name = f"word.{key}.r{self.r}.{self.table}" + ("" if flat else ".plain")
        if name not in fbs.FUNCS:
            fbs.register_lut(name, fn, lo, hi, table=self.table, flat=flat, sigma_in=sigma_in)
        if not hasattr(self, "_names"):
            self._names = {}
        self._names[name] = sigma_in
        return name

    def _lane(self, f):
        """The plaintext vector f(i mod n) over the slots."""
        return np.asarray(f(np.arange(self.slots) % self.n), dtype=np.float64)

    def _move(self, x, d, fill=None, scale=1.0):
        """`scale` x (x moved d digit slots up (d > 0) / down (d < 0) inside each lane): rotation + mask, one level
        (the scale rides the mask: an integer factor without a level of its own); the vacated slots take `fill`:
        None (0), a public scalar, or a ciphertext (its values there)."""
        o, n = self.ops, self.n
        if d == 0:
            return x if scale == 1.0 else o.mult(x, float(scale))
        keep = self._lane(lambda i: (i - d >= 0) & (i - d < n))
        y = o.mult_pt(o.rotate(x, -d), scale * keep, kind="mask")
        if fill is None or (np.isscalar(fill) and float(fill) == 0.0):
            return y
        if np.isscalar(fill):
            return o.add_pt(y, scale * float(fill) * (1.0 - keep), kind="mask")
        return o.add(y, o.mult_pt(fill, scale * (1.0 - keep), kind="mask"))

    def _cyc(self, x, d, scale=1.0):
        """`scale` x (x rotated d digit slots up inside each lane, cyclically): two rotations, two masks, one level."""
        o, n = self.ops, self.n
        d %= n
        if d == 0:
            return x if scale == 1.0 else o.mult(x, float(scale))
        hi = self._lane(lambda i: i >= d)
        return o.add(o.mult_pt(o.rotate(x, -d), scale * hi, kind="mask"),
                     o.mult_pt(o.rotate(x, n - d), scale * (1.0 - hi), kind="mask"))

    def _spread(self, x, k, lo, hi):
        """x's value at slot k of every lane (x zero at the lane's other slots) copied to slots lo .. hi of the lane:
        one rotation to lo, then a binary cover upwards (rotations only, never past the lane)."""
        o = self.ops
        y = o.rotate(x, k - lo) if k != lo else x
        cnt, acc, cur, off, b = hi - lo + 1, None, y, 0, 1
        while cnt:
            if cnt & 1:
                t = o.rotate(cur, -off) if off else cur
                acc = t if acc is None else o.add(acc, t)
                off += b
            cnt >>= 1
            if cnt:
                cur = o.add(cur, o.rotate(cur, -b))
                b *= 2
        return acc

    def _bcast(self, x, k, scale=1.0):
        """`scale` x slot k of every lane, over the whole lane (mask: one level, the scale in it; then rotations)."""
        o = self.ops
        y = o.mult_pt(x, scale * self._lane(lambda i: i == k), kind="mask")
        return self._spread(y, k, 0, self.n - 1)

    def _top_sum(self, y):
        """Every lane's sum at its top digit slot (rotations; the other slots hold partial sums)."""
        o, s = self.ops, 1
        while s < self.n:
            y = o.add(y, o.rotate(y, -s))
            s *= 2
        return y

    def _pred(self, pred):
        """The compare's last combine (j = hi + 3 lo -> the combined state s: 0 <, 1 =, 2 >) read as a predicate."""
        ok = {"lt": (0,), "le": (0, 1), "gt": (2,), "ge": (1, 2), "eq": (1,), "ne": (0, 2)}
        if pred not in ok:
            raise ValueError(f"WordOps cmp: predicate {pred}?")
        if pred == "lt":
            return self.lt
        return self._tab(f"p{pred}", lambda j, k=ok[pred]: np.isin(np.where(j % 3 == 1, j // 3, j % 3), k).astype(float),
                         0, 8)

    def _sdcmp(self):
        """The digit compare with the top digit signed: j = a + B b (+ B^2 at the top slot)."""
        B = self.base

        def f(j):
            x, y, top = j % B, (j // B) % B, j >= B * B
            sx, sy = np.where(top & (x >= B // 2), x - B, x), np.where(top & (y >= B // 2), y - B, y)
            return np.sign(sx - sy) + 1.0
        return self._tab("scmp", f, 0, 2 * B * B - 1)

    def _sign_top(self, unit):
        """`unit` x the sign bit of a's top digit, at the top slot only (0 elsewhere): one lookup on a + B at the top
        slot (t = 2 B)."""
        B = self.base
        name = self._tab(f"sgn{int(unit)}", lambda j: np.where(j >= B, float(unit) * ((j - B) >= B // 2), 0.0), 0,
                         2 * B - 1)
        return name

    def _flags(self, a, b, r, borrow=False):
        """(carry / borrow, signed overflow) of r = a + b (+ carry-in) at the lane's top digit slot: two lookups on
        j = a + B b + B^2 r (the carry-in to the top digit is (r - a - b) mod B)."""
        o, B = self.ops, self.base

        def parts(j):
            x, y, z = j % B, (j // B) % B, j // (B * B)
            return x, y, z, (z - x - y) % B

        def cy(j):
            x, y, z, c = parts(j)
            v = (x + y + c >= B).astype(float)
            return 1.0 - v if borrow else v

        def ov(j):
            x, y, z, _ = parts(j)
            sa, sb, sr = x >= B // 2, y >= B // 2, z >= B // 2
            return ((sa == sb) & (sr != sa)).astype(float)
        j = o.add(a, o.add(o.mult(b, float(B)), o.mult(r, float(B * B))))
        tc = self._tab("borrow" if borrow else "carry", cy, 0, B ** 3 - 1)
        return fbs.fbs(o, o.copy(j), tc), fbs.fbs(o, j, self._tab("ovf", ov, 0, B ** 3 - 1))

    def sub(self, a, b, flags=False):
        """(a - b) mod 2^bits = a + ~b + 1 (~d = B - 1 - d is linear; the carry-in through the add's fills).
        `flags`: also (borrow, signed overflow) at the top digit slot."""
        o = self.ops
        nb = o.add(o.mult(b, -1.0), float(self.base - 1))
        r = self._add_s(o.add(a, nb), cin=1)
        return (r,) + self._flags(a, nb, r, borrow=True) if flags else r

    def neg(self, a):
        """(-a) mod 2^bits = ~a + 1."""
        o = self.ops
        return self._add_s(o.add(o.mult(a, -1.0), float(self.base - 1)), cin=1)

    def sext(self, a, from_bits):
        """Sign extension of the low `from_bits` bits (a multiple of r) to the word: the sign at digit m - 1 (one
        lookup through a slot offset), copied to the digits above."""
        o, B, n = self.ops, self.base, self.n
        m = int(from_bits) // self.r
        if from_bits % self.r or not 0 < m <= n:
            raise ValueError("WordOps sext: from_bits a multiple of r, at most the width")
        if m == n:
            return a
        t = self._tab(f"sx{m}", lambda j: np.where(j >= B, (B - 1.0) * ((j - B) >= B // 2), 0.0), 0, 2 * B - 1)
        sg = fbs.fbs(o, o.add_pt(a, float(B) * self._lane(lambda i: i == m - 1), kind="mask"), t)
        return o.add(o.mult_pt(a, self._lane(lambda i: i < m), kind="mask"), self._spread(sg, m - 1, m, n - 1))

    def _signfill(self, a):
        """(B - 1) x the sign at the top slot (0 elsewhere): one lookup."""
        B = self.base
        t = self._tab("sxtop", lambda j: np.where(j >= B, (B - 1.0) * ((j - B) >= B // 2), 0.0), 0, 2 * B - 1)
        return fbs.fbs(self.ops, self.ops.add_pt(a, float(B) * self._lane(lambda i: i == self.n - 1), kind="mask"), t)

    def _bit_tab(self, kind, t, with_top=False):
        """The within-digit shift by t bits on j = d_i + B nb (nb the neighbour digit moving in; sar: + B^2 at the
        top slot, where the sign fills)."""
        B, r = self.base, self.r

        def f(j):
            d, nb, top = j % B, (j // B) % B, j >= B * B
            if kind in ("shl", "rotl"):
                return ((d << t) | (nb >> (r - t))) & (B - 1)
            fillb = np.where(top & (d >= B // 2), B - 1, nb) if kind == "sar" else nb
            return ((d >> t) | ((fillb << (r - t)) & (B - 1))).astype(np.int64)
        hi = 2 * B * B - 1 if with_top else B * B - 1
        return self._tab(f"b{kind}{t}" + ("t" if with_top else ""), lambda j: f(j).astype(float), 0, hi)

    def shift(self, a, kind, k):
        """a shifted / rotated by the public amount k (kind shl / shr / sar / rotl / rotr): whole digits by a move
        (rotation + mask, the sar fill from one sign lookup), the remaining bits in one bivariate lookup on the digit
        and its neighbour; shifts by >= the width give 0 (sar: the sign)."""
        o, B, n, r = self.ops, self.base, self.n, self.r
        bits = n * r
        if kind not in ("shl", "shr", "sar", "rotl", "rotr"):
            raise ValueError(f"WordOps shift: {kind}?")
        k = int(k) % bits if kind.startswith("rot") else min(int(k), bits)
        if k >= bits:
            if kind == "sar":
                return self._spread(self._signfill(a), n - 1, 0, n - 1)
            return o.mult_pt(a, np.zeros(self.slots), kind="mask")
        q, t = divmod(k, r)
        y = a
        if t:
            if kind in ("rotl", "rotr"):
                nb = self._cyc(a, 1 if kind == "rotl" else -1, scale=B)
            else:
                nb = self._move(a, 1 if kind == "shl" else -1, scale=B)
            j = o.add(a, nb)
            if kind == "sar":
                j = o.add_pt(j, float(B * B) * self._lane(lambda i: i == n - 1), kind="mask")
            y = fbs.fbs(o, j, self._bit_tab(kind, t, kind == "sar"))
        if q == 0:
            return y
        if kind in ("rotl", "rotr"):
            return self._cyc(y, q if kind == "rotl" else -q)
        if kind == "shl":
            return self._move(y, q)
        y = self._move(y, -q)
        if kind == "sar":
            y = o.add(y, self._spread(self._signfill(a), n - 1, n - q, n - 1))
        return y

    def shift_enc(self, a, b, kind):
        """a shifted / rotated by the encrypted amount b (a word in the same layout; kind shl / shr / sar / rotl /
        rotr): a barrel shifter, one lookup per amount bit k (j = x + B x_s + B^2 b_digit (+ B^3 big): bit k of the
        amount's digit selects x_s = x moved by 2^k bits); shifts by >= the width give 0 (the last stage's big) /
        the sign (sar: every stage's select forced by big, the amount saturating at the width - 1); rotates take b mod
        the width. The amount's digits are broadcast over the lane once (mask + rotations)."""
        o, B, n, r = self.ops, self.base, self.n, self.r
        bits = n * r
        if kind not in ("shl", "shr", "sar", "rotl", "rotr"):
            raise ValueError(f"WordOps shift_enc: {kind}?")
        L = bits.bit_length() - 1
        dig = [self._bcast(b, i, scale=B * B) for i in range(-(-L // r))]          # B^2 b_digit
        rot = kind.startswith("rot")
        big = sg = None
        if not rot:
            q0, p = divmod(L, r)
            T = 1 << p
            wv = self._lane(lambda i: np.where(i > q0, T, np.where(i == q0, 1, 0)))
            mx = int(T * (B - 1) * (n - 1 - q0) + (B - 1))
            tb = self._tab(f"big{L}", lambda v, T=T: (v >= T).astype(float), 0, mx)
            big = self._bcast(fbs.fbs(o, self._top_sum(o.mult_pt(b, wv, kind="mask")), tb), n - 1, scale=B ** 3)
        if kind == "sar":
            sg = self._spread(self._signfill(a), n - 1, 0, n - 1)

        def tab(k, t_bits, use_big):
            sh = 1 << k
            bitpos = k % r

            def f(j):
                x, xs, dg = j % B, (j // B) % B, (j // (B * B)) % B
                bg = j // (B ** 3)
                sel = ((dg >> bitpos) & 1).astype(bool)
                if use_big and kind == "sar":
                    sel = sel | (bg == 1)
                if t_bits:                              # within the digit: x_s is the neighbour
                    if kind in ("shl", "rotl"):
                        v = ((x << t_bits) | (xs >> (r - t_bits))) & (B - 1)
                    else:
                        v = (x >> t_bits) | ((xs << (r - t_bits)) & (B - 1))
                else:
                    v = xs
                out = np.where(sel, v, x)
                if use_big and kind != "sar":
                    out = np.where(bg == 1, 0, out)
                return out.astype(float)
            hi = (2 if use_big else 1) * B ** 3 - 1
            return self._tab(f"bs{kind}{sh}" + ("b" if use_big else ""), f, 0, hi)
        x = a
        for k in range(L):
            sh = 1 << k
            last = k == L - 1
            use_big = big is not None and (kind == "sar" or last)
            dd, t_bits = (1, sh) if sh < r else (sh // r, 0)     # within a digit: the neighbour moving in
            if rot:
                xs = self._cyc(x, dd if kind == "rotl" else -dd, scale=B)
            else:
                xs = self._move(x, dd if kind == "shl" else -dd, fill=sg if kind == "sar" else None, scale=B)
            j = o.add(x, o.add(xs, dig[k // r]))                 # x + B x_s + B^2 b_digit (+ B^3 big)
            if use_big:
                j = o.add(j, big)
            x = fbs.fbs(o, j, tab(k, t_bits, use_big))
        return x

    def mul(self, a, b):
        """a b mod 2^bits: the column sums S_i = sum_j a_(i - j) b_j by CKKS products (b's digits broadcast over the
        lane, a moved: no lookup), then `_norm` (S's base-B digits, one carry-save step, the add)."""
        o, n, B = self.ops, self.n, self.base
        S = None
        for j in range(n):
            t = o.mult(self._move(a, j), self._bcast(b, j))
            S = t if S is None else o.add(S, t)
        return self._norm(S, [(B - 1) ** 2 * (i + 1) for i in range(n)], "m")

    def mulc(self, a, c):
        """a c mod 2^bits for a public constant c: S = sum_j c_j (a moved by j) (scalar products, no lookup), then
        `_norm`."""
        o, n, B, r = self.ops, self.n, self.base, self.r
        c = int(c) % (1 << (n * r))
        cd = [(c >> (r * j)) & (B - 1) for j in range(n)]
        S = None
        for j, cj in enumerate(cd):
            if cj:
                t = self._move(a, j)
                t = t if cj == 1 else o.mult(t, float(cj))
                S = t if S is None else o.add(S, t)
        if S is None:
            return o.mult_pt(a, np.zeros(self.slots), kind="mask")
        return self._norm(S, [(B - 1) * sum(cd[:i + 1]) for i in range(n)], f"c{c}")

    def _norm(self, S, smax, key):
        """Digit sums S (column i at most smax[i]) -> the word sum_i S_i B^i mod B^n: S's base-B digits d_k (lookup
        k on S moved up k digits: K lookups) summed per column (T <= B^2 - 1), T split once (T mod B, T div B moved
        up: 2 lookups), the add on their sum (<= 2 B - 2)."""
        o, B, r = self.ops, self.base, self.r
        M = max(smax)
        if M <= B - 1:
            return S
        K = 1
        while B ** K <= M:
            K += 1
        tmax = sum(min(B - 1, M >> (r * k)) for k in range(K))
        if tmax > B * B - 1:
            raise ValueError(f"WordOps: column sums up to {M} need a second carry-save step")
        sig = 2.0 ** -14                                # S's noise: products and sums of encrypted digits
        T = None
        for k in range(K):
            tk = self._tab(f"nd{M}.{k}", lambda v, k=k: ((v >> (r * k)) & (B - 1)).astype(float), 0, M, flat=False,
                           sigma_in=sig)
            dk = fbs.fbs(o, self._move(S, k) if k else o.copy(S), tk)
            T = dk if T is None else o.add(T, dk)
        if tmax <= 2 * B - 2:
            return self._add_s(T)
        e0 = fbs.fbs(o, o.copy(T), self._tab(f"t{tmax}lo", lambda v: (v % B).astype(float), 0, tmax))
        e1 = fbs.fbs(o, self._move(T, 1), self._tab(f"t{tmax}hi", lambda v: (v // B).astype(float), 0, tmax))
        return self._add_s(o.add(e0, e1))

    def minmax(self, a, b, kind="min", signed=False):
        """min / max per lane: the compare's flag broadcast over the lane, then the mux."""
        s = self._bcast(self.cmp(a, b, "lt", signed), self.n - 1)
        return self.mux(s, a, b) if kind == "min" else self.mux(s, b, a)

    def abs(self, a):
        """|a| (two's complement; abs(MIN) = MIN): the sign (one lookup, broadcast) muxes -a and a."""
        s = self._bcast(fbs.fbs(self.ops, self.ops.add_pt(a, float(self.base) * self._lane(lambda i: i == self.n - 1),
                                                          kind="mask"), self._sign_top(1)), self.n - 1)
        return self.mux(s, self.neg(a), a)

    def popcount(self, a):
        """The number of set bits per lane, as a word: the digits' bit counts (one lookup), the lane sum (rotations),
        its base-B digits (lookups at the top slot) moved to digit slots 0, 1, ...."""
        o, B, n, r = self.ops, self.base, self.n, self.r
        pc = fbs.fbs(o, o.copy(a), self._tab("pc", lambda d: np.array([bin(int(v)).count("1") for v in np.ravel(d)],
                                                                        dtype=float).reshape(np.shape(d)), 0, B - 1))
        z = self._top_sum(pc)
        M = r * n
        K = 1
        while B ** K <= M:
            K += 1
        top = self._lane(lambda i: i == n - 1)
        out = None
        for k in range(K):
            dk = fbs.fbs(o, o.copy(z), self._tab(f"pcd{M}.{k}", lambda v, k=k: ((v >> (r * k)) & (B - 1)).astype(float),
                                                 0, M))
            t = o.rotate(o.mult_pt(dk, top, kind="mask"), n - 1 - k)
            out = t if out is None else o.add(out, t)
        return out

    def encode(self, x, bits=None):
        """Integers (one per lane, slots / n of them) -> the slot vector of their digits."""
        ds = digits(x, self.n * self.r, self.r)
        return np.stack(ds, 1).reshape(-1)

    def decode(self, v):
        """Slot vector -> integers per lane."""
        d = np.round(np.real(np.asarray(v))).astype(np.int64).reshape(-1, self.n)
        return sum(d[:, i] << (self.r * i) for i in range(self.n))


# ── the op registry shared with the TFHE path (impl/tfhe_bitwise.py) ─────────────────────────────────────────────
#
# op names: add sub neg addf subf (+ carry / borrow and signed overflow) | lt le gt ge eq ne, s-prefixed signed (slt ..)
# | sext<k> | shl<k> shr<k> sar<k> rotl<k> rotr<k> (public k) | shlv shrv sarv rotlv rotrv (amount = b) | mul mulw
# mulc<c> | min max smin smax abs popcount | mux xor and or | cmp (= lt) | id

import re as _re  # noqa: E402

PREDS = ("lt", "le", "gt", "ge", "eq", "ne")


def parse(op):
    """op -> (base, its integer parameter or None)."""
    m = _re.fullmatch(r"(sext|shl|shr|sar|rotl|rotr|mulcb|mulc|divc|remc|divremc|addi|andi|ori|xori|lti|slti|eqi|bset|bclr|binv|bext|"
                      r"shadd|shadduw|w32shl|w32shr|w32sar)(\d+)", op)
    if m:
        return m.group(1), int(m.group(2))
    return op, None


def is_pred(op):
    b = parse(op)[0]
    return b == "cmp" or b in PREDS or (b[:1] == "s" and b[1:] in PREDS)


def _sgnd(x, bits):
    x = np.asarray(x, dtype=np.int64)
    return np.where(x >= 1 << (bits - 1), x - (1 << bits), x)


def reference(op, a, b, sel, bits):
    """The exact outputs of `op` per word (two's complement words of `bits` bits): a list of integer arrays, the
    word value (or the predicate) first, then the flags."""
    op = _ALIAS.get(op, op)
    if op.startswith("mulcb"):
        op = "mulc" + op[5:]
    if bits > 32 or parse(op)[0] in _INT_OPS:
        return _reference_int(op, a, b, sel, bits)
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
    if base in PREDS or (base[:1] == "s" and base[1:] in PREDS):
        x, y = (sa, sb) if base not in PREDS else (a, b)
        f = {"lt": x < y, "le": x <= y, "gt": x > y, "ge": x >= y, "eq": x == y, "ne": x != y}[base.lstrip("s") if base not in PREDS else base]
        return [f.astype(np.int64)]
    if base == "sext":
        return [_sgnd(a & ((1 << k) - 1), k) & msk]
    if base in ("shl", "shr", "sar", "rotl", "rotr") or (base.endswith("v") and base[:-1] in ("shl", "shr", "sar", "rotl", "rotr")):
        kind = base.rstrip("v") if base.endswith("v") else base
        s_ = np.full_like(a, k) if k is not None else b
        if kind.startswith("rot"):
            s_ = s_ % bits
            sh = s_ if kind == "rotl" else (bits - s_) % bits
            return [np.where(sh == 0, a, ((a << sh) | (a >> (bits - sh))) & msk)]
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
        return [np.array([int(x) * int(y) for x, y in zip(a, b)], dtype=object)]       # up to 2 bits bits
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
    raise ValueError(f"intops: op {op}?")


# ops whose reference runs on Python integers (any width; the int64 path covers the rest up to 32 bits)
_INT_OPS = ("div", "rem", "divrem", "sdiv", "srem", "sdivrem", "divc", "remc", "divremc", "clz", "ctz", "log2",
            "andn", "orn", "xnor", "addi", "andi", "ori", "xori", "lti", "slti", "eqi", "bset", "bclr", "binv", "bext",
            "bsetv", "bclrv", "binvv", "bextv", "shadd", "shadduw", "adduw", "orcb", "rev8", "zext16", "czeqz", "cznez",
            "mulh", "mulhsu", "mulhu", "clmul", "clmulh", "clmulr", "w32add", "w32sub", "w32shl", "w32shr", "w32sar",
            "w32shlv", "w32shrv", "w32sarv", "w32mul", "w32div", "w32rem", "w32sdiv", "w32srem")
_ALIAS = {"mulb": "mul", "mulwb": "mulw", "popcountb": "popcount", "rotlvb": "rotlv", "rotrvb": "rotrv"}


def _reference_int(op, a, b, sel, bits):
    """`reference` on Python integers (words of any width; object arrays out). div / rem by 0: q = all ones, r = a
    (the common convention); signed division truncates toward 0 (MIN / -1 wraps to MIN, remainder 0); log2(0) = all
    ones (bits - 1 - clz)."""
    top, msk = 1 << bits, (1 << bits) - 1
    a, b, sel = ([int(x) for x in np.asarray(v).reshape(-1)] for v in (a, b, sel))
    sg = lambda x: x - top if x >= top >> 1 else x
    base, k = parse(op)
    O = lambda v: np.array(v, dtype=object)

    def udiv(x, y):
        return (msk, x) if y == 0 else (x // y, x % y)

    def sdiv(x, y):
        x, y = sg(x), sg(y)
        if y == 0:
            return ((1 if x < 0 else msk) & msk, x & msk)     # q = -1 for x >= 0, 1 for x < 0 (the sign-corrected all ones); r = x
        q = abs(x) // abs(y)
        q = -q if (x < 0) != (y < 0) else q
        return (q & msk, (x - q * y) & msk)
    if base in ("div", "rem", "divrem", "divc", "remc", "divremc"):
        ys = [k & msk] * len(a) if k is not None else b
        qr = [udiv(x, y) for x, y in zip(a, ys)]
        q, r = O([v[0] for v in qr]), O([v[1] for v in qr])
        return [q] if base in ("div", "divc") else [r] if base in ("rem", "remc") else [q + (r << bits)]
    if base in ("sdiv", "srem", "sdivrem"):
        qr = [sdiv(x, y) for x, y in zip(a, b)]
        q, r = O([v[0] for v in qr]), O([v[1] for v in qr])
        return [q] if base == "sdiv" else [r] if base == "srem" else [q + (r << bits)]
    if base.startswith("w32"):                       # the 32-bit op on the low halves, sign extended to the word
        inner = base[3:] + ("" if k is None else str(k))
        lo32 = lambda v: [x & 0xFFFFFFFF for x in v]
        r = _reference_int(inner, np.array(lo32(a), dtype=object), np.array(lo32(b), dtype=object), np.array(sel, dtype=object), 32)[0]
        return [O([(x - (1 << 32) if x >> 31 & 1 else x) & msk for x in r])]
    if base in ("addi", "andi", "ori", "xori"):
        f = {"addi": lambda x: (x + k) & msk, "andi": lambda x: x & k, "ori": lambda x: (x | k) & msk, "xori": lambda x: (x ^ k) & msk}[base]
        return [O([f(x) for x in a])]
    if base in ("lti", "eqi"):
        return [O([int(x < (k & msk) if base == "lti" else x == (k & msk)) for x in a])]
    if base == "slti":
        return [O([int(sg(x) < sg(k & msk)) for x in a])]
    if base in ("bset", "bclr", "binv", "bext", "bsetv", "bclrv", "binvv", "bextv"):
        idx = [k % bits] * len(a) if k is not None else [y % bits for y in b]
        op1 = base[:-1] if base in ("bsetv", "bclrv", "binvv", "bextv") else base
        f = {"bset": lambda x, j: x | 1 << j, "bclr": lambda x, j: x & ~(1 << j) & msk, "binv": lambda x, j: x ^ 1 << j,
             "bext": lambda x, j: x >> j & 1}[op1]
        return [O([f(x, j) for x, j in zip(a, idx)])]
    if base in ("shadd", "shadduw", "adduw"):
        kk = 0 if base == "adduw" else k
        src = [x & 0xFFFFFFFF for x in a] if base != "shadd" else a
        return [O([((x << kk) + y) & msk for x, y in zip(src, b)])]
    if base == "orcb":
        return [O([sum((0xFF if (x >> (8 * t)) & 0xFF else 0) << (8 * t) for t in range(bits // 8)) for x in a])]
    if base == "rev8":
        return [O([sum(((x >> (8 * t)) & 0xFF) << (8 * (bits // 8 - 1 - t)) for t in range(bits // 8)) for x in a])]
    if base == "zext16":
        return [O([x & 0xFFFF for x in a])]
    if base in ("czeqz", "cznez"):
        return [O([(0 if y == 0 else x) if base == "czeqz" else (0 if y != 0 else x) for x, y in zip(a, b)])]
    if base in ("mulh", "mulhsu", "mulhu"):
        xs = [sg(x) for x in a] if base in ("mulh", "mulhsu") else a
        ys = [sg(y) for y in b] if base == "mulh" else b
        return [O([((x * y) >> bits) & msk for x, y in zip(xs, ys)])]
    if base in ("clmul", "clmulh", "clmulr"):
        def cl(x, y):
            r = 0
            for i in range(bits):
                if y >> i & 1:
                    r ^= x << i
            return r
        sh = {"clmul": 0, "clmulh": bits, "clmulr": bits - 1}[base]
        return [O([(cl(x, y) >> sh) & msk for x, y in zip(a, b)])]
    if base == "clz":
        return [O([bits - x.bit_length() for x in a])]
    if base == "ctz":
        return [O([bits if x == 0 else (x & -x).bit_length() - 1 for x in a])]
    if base == "log2":
        return [O([(x.bit_length() - 1) & msk for x in a])]
    sa, sb = [sg(x) for x in a], [sg(x) for x in b]
    if base in ("add", "addf", "sub", "subf"):
        sub = base.startswith("sub")
        r = [O([((x - y) if sub else (x + y)) & msk for x, y in zip(a, b)])]
        if base.endswith("f"):
            cy = [int(x < y) if sub else (x + y) >> bits for x, y in zip(a, b)]
            ov = [int(not -(top >> 1) <= ((x - y) if sub else (x + y)) < top >> 1) for x, y in zip(sa, sb)]
            r += [O(cy), O(ov)]
        return r
    if base == "neg":
        return [O([(-x) & msk for x in a])]
    if base == "cmp" or base in PREDS or (base[:1] == "s" and base[1:] in PREDS):
        x_, y_ = (sa, sb) if base not in PREDS and base != "cmp" else (a, b)
        pr = "lt" if base == "cmp" else base if base in PREDS else base[1:]
        f = {"lt": lambda x, y: x < y, "le": lambda x, y: x <= y, "gt": lambda x, y: x > y, "ge": lambda x, y: x >= y,
             "eq": lambda x, y: x == y, "ne": lambda x, y: x != y}[pr]
        return [O([int(f(x, y)) for x, y in zip(x_, y_)])]
    if base == "sext":
        return [O([sg_k & msk for sg_k in ((x & ((1 << k) - 1)) - ((1 << k) if x >> (k - 1) & 1 else 0) for x in a)])]
    if base in ("shl", "shr", "sar", "rotl", "rotr") or (base.endswith("v") and base[:-1] in ("shl", "shr", "sar", "rotl", "rotr")):
        kind = base[:-1] if base.endswith("v") else base
        amts = [k] * len(a) if k is not None else b
        out = []
        for x, sx, s_ in zip(a, sa, amts):
            if kind.startswith("rot"):
                s_ %= bits
                sh = s_ if kind == "rotl" else (bits - s_) % bits
                out.append(((x << sh) | (x >> (bits - sh))) & msk if sh else x)
            elif s_ >= bits:
                out.append(msk if kind == "sar" and sx < 0 else 0)
            else:
                out.append((x << s_) & msk if kind == "shl" else x >> s_ if kind == "shr" else (sx >> s_) & msk)
        return [O(out)]
    if base == "mul":
        return [O([(x * y) & msk for x, y in zip(a, b)])]
    if base == "mulw":
        return [O([x * y for x, y in zip(a, b)])]
    if base == "mulc":
        return [O([(x * (k & msk)) & msk for x in a])]
    if base in ("min", "max", "smin", "smax"):
        xs, ys = (sa, sb) if base.startswith("s") else (a, b)
        mn = base.endswith("min")
        return [O([x if (xx < yy) == mn else y for x, y, xx, yy in zip(a, b, xs, ys)])]
    if base == "abs":
        return [O([abs(x) & msk for x in sa])]
    if base == "popcount":
        return [O([bin(x).count("1") for x in a])]
    if base == "mux":
        return [O([x if s_ == 1 else y for x, y, s_ in zip(a, b, sel)])]
    if base in ("xor", "and", "or", "andn", "orn", "xnor"):
        f = {"xor": lambda x, y: x ^ y, "and": lambda x, y: x & y, "or": lambda x, y: x | y,
             "andn": lambda x, y: x & ~y & msk, "orn": lambda x, y: (x | ~y) & msk, "xnor": lambda x, y: ~(x ^ y) & msk}[base]
        return [O([f(x, y) for x, y in zip(a, b)])]
    if base == "id":
        return [O(a)]
    raise ValueError(f"intops: op {op}?")


def inputs(words, bits, seed=0, op=None):
    """Random words with the edge cases first: 0, -1, MIN, MAX, MIN - 1, equal pairs, a + b = -1 (the longest carry
    chain); shift amounts 0, 1, bits - 1, bits, bits + 1, 2^bits - 1 and random below 2 bits."""
    rng = np.random.default_rng(seed)
    top = 1 << bits
    if bits > 32:                                                  # Python integers (object arrays)
        edge = np.array([0, top - 1, top >> 1, (top >> 1) - 1, 1, top - 2], dtype=object)
        rnd = lambda: np.array([int(x) | int(y) << 32 for x, y in rng.integers(0, 1 << 32, (words, 2), dtype=np.int64)], dtype=object) & (top - 1)
        a, b = rnd(), rnd()
    else:
        edge = np.array([0, top - 1, top >> 1, (top >> 1) - 1, 1, top - 2], dtype=np.int64)
        a = rng.integers(0, top, words, dtype=np.int64)
        b = rng.integers(0, top, words, dtype=np.int64)
    sel = rng.integers(0, 2, words).astype(np.int64)
    pairs = [(x, y) for x in edge for y in edge]
    q = min(len(pairs), words // 2)                                # every edge pair once words >= 72
    for i, (x, y) in enumerate(pairs[:q]):
        a[i], b[i] = x, y
    if q < words:
        nq = (words - q) // 4
        b[q:q + nq] = a[q:q + nq]                                  # equal
        b[q + nq:q + 2 * nq] = (~a[q + nq:q + 2 * nq]) & (top - 1)   # a + b = -1
    base = parse(op)[0] if op else ""
    if base.endswith("v"):                                         # shift amounts
        dt = object if bits > 32 else np.int64
        amts = np.array([0, 1, bits - 1, bits, bits + 1, top - 1, 2 * bits, bits // 2], dtype=dt)
        b = rng.integers(0, 2 * bits, words, dtype=np.int64).astype(dt)
        b[:min(words, len(amts))] = amts[:min(words, len(amts))]
        if words > len(amts):
            b[len(amts):2 * len(amts)] = amts[:max(0, min(len(amts), words - len(amts)))]
    return a, b, sel



def word_op(w, op, a, b, sel=None):
    """`op` on WordOps `w` (a, b the words' ciphertexts, sel the mux selector per digit slot): the outputs, the word
    first (a predicate: one flag at the lane's top digit slot), then the flags (top slot)."""
    base, k = parse(op)
    if base in ("add", "addf"):
        return list(w.add(a, b, flags=True)) if base == "addf" else [w.add(a, b)]
    if base in ("sub", "subf"):
        return list(w.sub(a, b, flags=True)) if base == "subf" else [w.sub(a, b)]
    if base == "neg":
        return [w.neg(a)]
    if is_pred(op):
        sg = base not in PREDS and base != "cmp"
        return [w.cmp(a, b, "lt" if base == "cmp" else base[1:] if sg else base, sg)]
    if base == "sext":
        return [w.sext(a, k)]
    if base in ("shl", "shr", "sar", "rotl", "rotr"):
        return [w.shift(a, base, k)]
    if base.endswith("v") and base[:-1] in ("shl", "shr", "sar", "rotl", "rotr"):
        return [w.shift_enc(a, b, base[:-1])]
    if base == "mul":
        return [w.mul(a, b)]
    if base == "mulc":
        return [w.mulc(a, k)]
    if base in ("min", "max", "smin", "smax"):
        return [w.minmax(a, b, base.lstrip("s"), base.startswith("s"))]
    if base == "abs":
        return [w.abs(a)]
    if base == "popcount":
        return [w.popcount(a)]
    if base == "mux":
        return [w.mux(sel, a, b)]
    if base in ("xor", "and", "or"):
        return [w.bitwise(base, a, b)]
    raise ValueError(f"WordOps: op {op}?")


def word_check(w, op, outs, a, b, sel, bits):
    """Decrypted outputs of `word_op` against `reference`: (wrong lanes, max deviation from the integers)."""
    n = w.n
    ref = reference(op, a, b, sel, bits)
    bad = np.zeros(len(np.asarray(a)), dtype=bool)
    dev = 0.0
    for i, (v, want) in enumerate(zip(outs, ref)):
        v = np.real(np.asarray(v))
        dev = max(dev, float(np.abs(v - np.round(v)).max()))
        got = (np.round(v).astype(np.int64).reshape(-1, n)[:, -1] if (i > 0 or is_pred(op)) else w.decode(v))
        bad |= got[:len(bad)] != want
    return int(bad.sum()), dev


def int_op(io, op, a, b, sel=None):
    """`op` on IntOps `io` (a, b: digit ciphertext lists, sel the mux selector): the outputs, a word (a list) first
    (a predicate: one ciphertext), then the flags."""
    base, k = parse(op)
    if base in ("add", "addf"):
        return list(io.add(a, b, flags=True)) if base == "addf" else [io.add(a, b)]
    if base in ("sub", "subf"):
        return list(io.sub(a, b, flags=True)) if base == "subf" else [io.sub(a, b)]
    if base == "neg":
        return [io.neg(a)]
    if is_pred(op):
        sg = base not in PREDS and base != "cmp"
        return [io.cmp_pred(a, b, "lt" if base == "cmp" else base[1:] if sg else base, sg)]
    if base == "sext":
        return [io.sext(a, k)]
    if base in ("shl", "shr", "sar", "rotl", "rotr"):
        return [io.shift(a, base, k)]
    if base.endswith("v") and base[:-1] in ("shl", "shr", "sar", "rotl", "rotr"):
        return [io.shift_enc(a, b, base[:-1])]
    if base == "mul":
        return [io.mul(a, b)]
    if base == "mulc":
        return [io.mulc(a, k)]
    if base in ("min", "max", "smin", "smax"):
        return [io.minmax(a, b, base.lstrip("s"), base.startswith("s"))]
    if base == "abs":
        return [io.abs(a)]
    if base == "popcount":
        return [io.popcount(a)]
    if base == "mux":
        return [io.mux(sel, a, b)]
    if base in ("xor", "and", "or"):
        return [io.bitwise(base, a, b)]
    raise ValueError(f"IntOps: op {op}?")


def int_check(io, op, outs, a, b, sel, bits):
    """Decrypted `int_op` outputs (each a digit list, or one array) against `reference`: (wrong words, max dev)."""
    ref = reference(op, a, b, sel, bits)
    bad = np.zeros(len(np.asarray(a)), dtype=bool)
    dev = 0.0
    for i, (v, want) in enumerate(zip(outs, ref)):
        if isinstance(v, list):
            dev = max(dev, max(float(np.abs(np.real(x) - np.round(np.real(x))).max()) for x in v))
            got = value([np.real(x) for x in v], io.r)
        else:
            x = np.real(np.asarray(v))
            dev = max(dev, float(np.abs(x - np.round(x)).max()))
            got = np.round(x).astype(np.int64)
        bad |= got[:len(bad)] != want
    return int(bad.sum()), dev
