"""Large-precision sign and exact word extraction of CKKS values by digit decomposition (in the spirit of Liu,
Micciancio, Song, Asiacrypt 2022: a high-precision phase resolved by small lookups), as native TFHE programs
(impl/tfhe.py `Prog`) behind one CKKS -> TFHE switch.

The switch reads its input to ~5.3e-3 rms of the native unit (the mod switch to 2 n_br and the LWE key switch), i.e.
~7.5 bits. Its CKKS side holds far more: at the StC-first split the value sits at phase m q0 / 4 mod q0 = p0 p1
(~2^56) with the CKKS noise far below 2^-30. A window load (`Prog.load(..., shifts=...)`) multiplies the split by 2^k
before the switch's rescale (exact, mod q0): the switch then reads 2^k m mod 4, the phase's bits k further down, at its
own precision. No new key and no new modulus: the windows only need the switch's ~7.5 bits each, every read is
independent (one launch for all of them) and the digits come out of the reads' overlaps.

Reads: Y_k = round(2^k r m + eta) (r bins per native unit, |eta| <~ 0.03 of a bin at 1 sigma: 5.3e-3 / bin), known
mod the window's bins. Consecutive windows overlap, so each lift is unique: with Y_(j+1) = s Y_j + c_j (s = 2^step),
|c_j| <= s (1/2 + |eta_j|) + 1/2 + |eta_(j+1)| stays inside the window's range when |eta| is below a third of a bin.

* Sign (`sign_prog`): windows of 16 bins (round(4 2^k m) mod 16) every 3 bits; the status of a window is the sign of
  its centred read (0 and 8 read 0: the negacyclic pair; 8 cannot be the deciding window). The topmost nonzero status
  is the sign of m (a window under an all-zero one has |Y| <= 4.5 + 9 eta < 8, so its centred read is its value): the
  compare tree of tfhe_bitwise (fan-in 3, power-of-two weights) over the windows, the last combine read by a whole-
  torus sign table in the store. Rounds: the load, ceil(log3 n) - 1 combines, the store. Resolution: correct for
  |m| >= (1/2 + eta) / (4 8^(n - 1)).
* Word (`word_load`): windows of 8 bins (round(2 2^k m) mod 8) every 2 bits, k = 0, 2, .., 2 (nd - 1): Y = round(m
  2^(2 nd - 1)) as an nd-digit radix-4 two's-complement word (tfhe_bitwise's format: digit d at d / 8), m in [-1, 1).
  With c_j = Y_j - 4 Y_(j-1) in [-3, 3] (|eta| < 0.1 bin: |c| <= 2, the lift needs only |c| < 4), c_j = centred((R_j
  - 4 p_(j-1)) mod 8), p = Y_(j-1) mod 2 = R_(j-1) mod 2, the word's digits are (R_j mod 4 - b_j) mod 4 with b_j the
  borrow of the lower part sum_(l > j) 4^(n-1-l) c_l: the sign of its topmost nonzero c (a prefix of statuses, the
  adder's). The load reads X = (-1)^b2 (2 u + 1) / 8 (u = R mod 4, b2 = R's top bit: full information, negacyclic)
  and T = (-1)^b2 / 4 (many-LUT k = 2); 8 X + 4 T = 2 (1 - p) mod 4 puts p as a half turn; a status table on X_j + 2
  p_(j-1) (its p = 1 half the negation: the ambiguous c = +-4 never happens) gives sign(c_j); the borrows' prefix;
  the digit from X_j + b_j / 8 - 1 / 16 (16 bins on [-1, 1)). Rounds: the load, the statuses, the prefix's levels,
  the digits.

Noise (`analyze_*`): every window read is a load (the switch's input noise, x 2 for k = 2); p_fail per read is the
Gaussian tail of the lift's margin; the later lookups are tfhe_bitwise's (8 / 16 bins on [-1, 1)).
"""
from __future__ import annotations

import math

import numpy as np

from . import tfhe

#: |m| bound of the sign's input (native units; the switch reads (-2, 2)): the top window's round(4 m) stays in (-8, 8)
SIGN_SPAN = 1.75
#: the switch's input noise, native units (pc-check tfhe --identity: rms 5.3e-3 of [-1, 1)); a k-table load: x k on
#: its mod-switch part (model 4.1e-3 of the 5.3e-3)
LOAD_SD = 5.3e-3
LOAD_SD_MS = 4.1e-3


def load_sd(k=1):
    """The read noise of a load of k interleaved tables (the mod switch k times coarser)."""
    rest = max(LOAD_SD ** 2 - LOAD_SD_MS ** 2, 0.0)
    return math.sqrt((k * LOAD_SD_MS) ** 2 + rest)


def _log2_tail2(z):
    """log2 P(|N(0, 1)| >= z)."""
    if z < 8.0:
        return math.log2(math.erfc(z / math.sqrt(2.0)))
    ln = -z * z / 2.0 - math.log(z) - 0.5 * math.log(2.0 * math.pi) + math.log1p(-1.0 / z ** 2 + 3.0 / z ** 4)
    return 1.0 + ln / math.log(2.0)


# ── sign ────────────────────────────────────────────────────────────────────────────────────────────────────────────


def sign_windows(bits):
    """Window shifts 0, 3, .., 3 (n - 1) for a sign exact at |x| >= 2^-bits of the input bound (`SIGN_SPAN` x)."""
    n = 1 + max(0, math.ceil((bits - math.log2(4.0 * SIGN_SPAN / 0.84)) / 3.0))
    return [3 * j for j in range(n)]


def sign_resolution(n, eta=0.34):
    """The smallest |x| / bound a sign of n windows decides (reads within eta of a bin)."""
    return (0.5 + eta) / (4.0 * SIGN_SPAN * 8.0 ** (n - 1))


def _sgn_status(m):
    """A window's status: sign(round(4 m)) / 4 on [-1, 1) (the whole torus by negacyclicity: 0 and 8 read 0)."""
    return np.sign(np.round(4.0 * np.asarray(m))) / 4.0


def _sgn_out(m):
    """The last combine on the whole torus: +1 for >= 0 (ties: all windows zero), -1 below (|m| <= 7/4)."""
    return np.where(np.round(4.0 * np.asarray(m)) >= 0, 1.0, -1.0)


def _tree(p, w, k, rowof, store):
    """The compare tree of tfhe_bitwise (`WordProg._cmp3`) over k statuses per word (row of digit i of word q:
    rowof(q, i), digit k - 1 the most significant) in buffer 0; `store(buf, rows)` takes the last combine."""
    cur, nxt, ro = 0, 3, rowof
    while True:
        f = min(3, k)
        h = -(-k // f)
        src = [[ro(q // h, (q % h) * f + f - 1 - x) if (q % h) * f + f - 1 - x < k else -1 for x in range(f)]
               for q in range(w * h)]
        p.lin(4, cur, cur, [(v[0], v[1] if f > 1 else -1) for v in src], ca=1 << (f - 1), cb=1 << max(f - 2, 0))
        if f == 3:
            p.lin(4, 4, cur, [(q, v[2]) for q, v in enumerate(src)], ca=1, cb=1)
        if h == 1:
            return store(4, w)
        p.pbs(4, w * h, "sgn", [nxt])
        cur, nxt, k = nxt, (2 if nxt == 3 else 3), h
        ro = (lambda hh: (lambda q, i: q * hh + i))(h)


def sign_prog(name, b, bits, out_slots, half=False):
    """[m >= 0] as +-1 (`half`: +-1/2) for the b loaded values m (|m| <= SIGN_SPAN), exact for |m| >= the resolution
    of `bits` (module doc); the output on `out_slots`."""
    sh = sign_windows(bits)
    n = len(sh)
    p = tfhe.Prog(name, 5)
    p.lut("sgn", _sgn_status)
    v = 0.5 if half else 1.0
    p.lut("sgn_w", lambda m: v * _sgn_out(m), warp=True)
    p.load("sgn", 0, shifts=sh)
    # window j (shift 3 j) is digit n - 1 - j of slot q: row j b + q
    _tree(p, b, n, lambda q, i: (n - 1 - i) * b + q, lambda buf, rows: p.store(buf, rows, "sgn_w", out_slots))
    return p


def analyze_sign(bits, k=1):
    """Windows, rounds, PBS per value, resolution and log2 p_fail per read / per sign."""
    sh = sign_windows(bits)
    n = len(sh)
    sd_bin = load_sd(k) / 0.25                  # 16 bins on the whole torus: a bin is 1/4 native
    # the lift: 8 |eta_a| + |eta_b| < 3 (a deciding window under a zero one stays inside (-8, 8))
    lp_read = _log2_tail2(3.0 / (math.sqrt(65.0) * sd_bin))
    levels, kk, pbs = 0, n, n + 1
    while kk > 1:
        kk = -(-kk // min(3, kk))
        levels += 1
        pbs += kk if kk > 1 else 0
    return {"windows": n, "shifts": sh, "rounds": levels + 1, "pbs_per_value": pbs,
            "resolution_log2": math.log2(sign_resolution(n)), "read_sd_bins": sd_bin,
            "log2_pfail_read": lp_read, "log2_pfail_sign": lp_read + math.log2(n)}


# ── word ────────────────────────────────────────────────────────────────────────────────────────────────────────────


def _xr(R):
    """The load's full-information read of R in Z_8: (-1)^b2 (2 (R mod 4) + 1) / 8."""
    R = np.asarray(R) % 8
    return np.where(R >= 4, -1.0, 1.0) * (2.0 * (R % 4) + 1.0) / 8.0


def _r8(m):
    return np.round(2.0 * np.asarray(m)).astype(np.int64) % 8


def _ld_x(m):
    return _xr(_r8(m))


def _ld_t(m):
    return np.where(_r8(m) >= 4, -0.25, 0.25)


def _stat(m):
    """-sign(c) (the borrow's status: generate when c < 0) from X_j (p_(j-1) = 0; p = 1 is the half turn): X = +-1/8
    (R = 0, 4) reads 0, X >= 3/8 (R = 1..3) -1/4, X <= -3/8 (R = 5..7) +1/4."""
    m = np.asarray(m)
    return np.where(np.abs(m) < 0.25, 0.0, -np.sign(m) / 4.0)


def _xdec(m):
    """X + b / 8 - 1 / 16 (16 points spaced 1/8 on (-1, 1)) -> (u, b)."""
    k = np.clip(np.round((np.asarray(m) + 15.0 / 16.0) * 8.0), 0, 15).astype(np.int64)
    x = (2 * (k // 2) - 7) / 8.0                        # the X grid point
    u = ((np.round(np.abs(x) * 8.0).astype(np.int64) - 1) // 2)
    return u, k % 2


def _digit(m):
    u, b = _xdec(m)
    return ((u - b) % 4) / 8.0


def word_windows(nd):
    return [2 * j for j in range(nd)]


def word_load(p, values, nd, dst=0, bufs=(1, 2, 3, 4)):
    """Steps of Prog `p` loading `values` CKKS values m in [-1, 1) as words of nd radix-4 digits (Y = round(m
    2^(2 nd - 1)), two's complement, digit d at d / 8) into buffer `dst`, rows [value][digit] (module doc). `bufs`:
    four scratch buffers (the X reads stay in bufs[0] to the last step)."""
    bx, bt, b2, b3 = bufs
    p.lut("dld", _ld_x, _ld_t)
    p.lut("dstat", _stat)
    p.lut("dcarry", lambda m: np.where(np.round(4.0 * np.asarray(m)) > 0, 0.0625, -0.0625))
    p.lut("sgn", _sgn_status)
    p.lut("ddig", _digit)
    V, n = int(values), int(nd)
    p.load("dld", [bx, bt], shifts=word_windows(n))
    R = V * n
    xrow = lambda r: ((n - 1) - (r % n)) * V + r // n            # window j = n - 1 - digit, value r // n
    up = lambda r: xrow(r) - V if r % n < n - 1 else -1          # the window above (digit + 1)
    # y = X_j + 2 p_(j-1), 2 p = 2 - 8 X - 4 T (mod 4) of the window above; the top digit's status is unused
    p.lin(b2, bx, bx, [(xrow(r), up(r)) for r in range(R)], ca=1, cb=-8)
    p.lin(b2, b2, bt, [(r, up(r)) for r in range(R)], ca=1, cb=-4, c=2.0)
    p.lin(b2, b2, b2, [(r, -1) if r % n < n - 1 else (-1, -1) for r in range(R)], ca=1, cb=0)
    p.pbs(b2, R, "dstat", [b3])
    # the borrows: the adder's prefix (statuses of the lower digits, the higher one dominating)
    cur, nxt, d = b3, b2, 1
    while d < n:
        f = min(3, -(-n // d))
        last = d * f >= n
        p.lin(dst, cur, cur, [(r, r - d if r % n >= d else -1) for r in range(R)], ca=1 << (f - 1), cb=1 << (f - 2))
        if f == 3:
            p.lin(dst, dst, cur, [(r, r - 2 * d if r % n >= 2 * d else -1) for r in range(R)], ca=1, cb=1)
        p.pbs(dst, R, "dcarry" if last else "sgn", [nxt])
        cur, nxt, d = nxt, cur, d * f
    if n > 1:
        p.lin(nxt, cur, cur, [(r, -1) for r in range(R)], ca=1, cb=0, c=0.0625)      # borrow out of r: 0 / 1/8
        p.lin(dst, bx, nxt, [(xrow(r), r - 1 if r % n else -1) for r in range(R)], ca=1, cb=1, c=-0.0625)
    else:
        p.lin(dst, bx, bx, [(xrow(r), -1) for r in range(R)], ca=1, cb=0, c=-0.0625)
    p.pbs(dst, R, "ddig", [dst])
    return R


def word_prog(name, values, nd, out_slots):
    """The word load alone, stored as digits d / 4 (one slot per digit on `out_slots`): the handoff's check."""
    p = tfhe.Prog(name, 5)
    R = word_load(p, values, nd, dst=0)
    p.lut("dig_w", lambda m: np.clip(np.round(8.0 * np.asarray(m)), 0, 3) / 4.0, warp=True)
    p.store(0, R, "dig_w", out_slots)
    return p


def word_reference(m, nd):
    """The exact words (digits, least significant first) of round(m 2^(2 nd - 1)) mod 4^nd."""
    y = np.round(np.asarray(m, dtype=np.float64) * 2.0 ** (2 * nd - 1)).astype(np.int64) % (4 ** nd)
    return np.stack([(y >> (2 * i)) & 3 for i in range(nd)], axis=-1)


def analyze_word(nd):
    """Rounds, PBS per word and log2 p_fail of the word load."""
    sd_bin = load_sd(2) / 0.5                   # 8 bins on the whole torus: a bin is 1/2 native, the load k = 2
    # the lift: |eta_j| + 4 |eta_(j-1)| < 1.5 (|c| < 4); the reads' tail at that margin
    lp_read = _log2_tail2(1.5 / (math.sqrt(17.0) * sd_bin))
    levels, d = 0, 1
    while d < nd:
        d *= min(3, -(-nd // d))
        levels += 1
    return {"windows": nd, "rounds": 1 + 1 + levels + 1, "pbs_per_word": nd * (2 + levels + 1),
            "read_sd_bins": sd_bin, "log2_pfail_read": lp_read}


# ── device cases (pc-check tfhe-digit --case FILE) ──────────────────────────────────────────────────────────────────


def sign_case(bits, values, seed=0, half=False):
    """A sign case: random values over 2^0 .. 2^-(bits + 2) of the bound, the edges (0, +-the resolution, the windows'
    boundaries, the range ends), as native loaded values (x SIGN_SPAN)."""
    rng = np.random.default_rng(seed)
    n = len(sign_windows(bits))
    res = sign_resolution(n)
    edge = [0.0, 1e-12, -1e-12, res, -res, 2.0 * res, -2.0 * res, 1.0, -1.0, 0.999, -0.999]
    for i in range(n):                         # a window's rounding boundary: (j + 1/2) / (4 8^i) of m
        for j in (-1, 0, 1, 3):
            edge.append((j + 0.5) / (4.0 * 8.0 ** i) / SIGN_SPAN)
    x = list(edge[:values]) if values >= 16 else []
    while len(x) < values:
        x.append(float(rng.uniform(-1, 1) * 2.0 ** -int(rng.integers(0, bits + 1))))
    x = np.asarray(x)
    p = sign_prog("dsign", values, bits, "out", half=half)
    return {"name": f"sign{bits}x{values}", "kind": "sign", "bits": bits, "resolution": res, "span": SIGN_SPAN,
            "half": half, "prog": __import__("json").loads(p.json()), "m": [float(v) for v in x * SIGN_SPAN]}


def word_case(nd, values, seed=0):
    """A word case: random values in [-1, 1), exact grid points and half-ulp ties, the range ends."""
    rng = np.random.default_rng(seed)
    ulp = 2.0 ** -(2 * nd - 1)
    k = rng.integers(-(2 ** (2 * nd - 1)) + 2, 2 ** (2 * nd - 1) - 2, 8)
    edge = [0.0, ulp / 2, -ulp / 2, ulp, -ulp, -1.0, 1.0 - ulp, 1.0 - 0.6 * ulp] + list(k * ulp) + list((k + 0.5) * ulp)
    x = list(edge[:values]) if values >= 16 else []
    while len(x) < values:
        x.append(float(rng.uniform(-1, 1) * 2.0 ** -int(rng.integers(0, 2 * nd))))
    p = word_prog("dword", values, nd, "out")
    return {"name": f"word{2 * nd}x{values}", "kind": "word", "nd": nd, "prog": __import__("json").loads(p.json()),
            "m": [float(v) for v in x]}


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(description="digit-decomposition cases for pc-check tfhe-digit, and the analysis")
    ap.add_argument("--case", help="write a case to this file")
    ap.add_argument("--kind", default="sign", choices=("sign", "word"))
    ap.add_argument("--bits", type=int, default=16, help="sign: resolution 2^-bits of the bound")
    ap.add_argument("--nd", type=int, default=8, help="word: radix-4 digits (2 nd bits)")
    ap.add_argument("--values", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--half", action="store_true", help="sign: outputs +-1/2")
    a = ap.parse_args(argv)
    if a.case:
        c = sign_case(a.bits, a.values, a.seed, a.half) if a.kind == "sign" else word_case(a.nd, a.values, a.seed)
        with open(a.case, "w") as f:
            json.dump(c, f)
        print(f"[digits] {c['name']}: {len(c['prog']['steps'])} steps -> {a.case}")
        return
    for bits in (12, 16, 20, 24):
        print("[digits] sign", bits, json.dumps(analyze_sign(bits)))
    for nd in (4, 8, 10, 16):
        print("[digits] word", 2 * nd, json.dumps(analyze_word(nd)))


if __name__ == "__main__":
    main()
