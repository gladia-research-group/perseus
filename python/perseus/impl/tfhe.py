"""Lookup tables by CKKS -> TFHE -> CKKS scheme switching (pc-tfhe; device: pc-gpu tfhe.rs; planned service:
pc-py tfhe.rs): y = f(x) on a set of real slots, every other slot an exact zero, as one refresh. The switch takes the
ciphertext at the StC-first split of a dense refresh, turns the chosen slots into LWEs, evaluates f by programmable
bootstrapping (exact for steps: no polynomial gap) and packs the outputs back for the refresh's finish.

`tfhe_lut(ops, x, lut, slots)`: `lut` names a function registered with `register_lut` (|f| <= 1.25 on [-1, 1): the
output warp's range at out_cf 3) or a sign registered with `register_sign` (inputs on (-2, 2)), `slots`
a slot set registered with `register_slots`; x must hold values in [-1, 1) on the set (they wrap beyond), and inputs
within ~1-2 % of the range of a discontinuity may land on either side (the switch's input noise). The capture records
one external refresh (`{op: "tfhe", lut, slots}`) landing where a PCMM output does; the planned driver registers the
tables and sets with the session (`setup_session`). The finish reads v / 2^(out_cf - cf) = v / 2: one ciphertext add
folds the 2 back. The numpy mirror and the fakes compute f exactly. Not on the eager device path (no refresh there):
plan it.
"""
from __future__ import annotations

import os

import numpy as np

# n_br nu: the test vector's points over [-1, 1) (PC_TFHE_NU=2: the extended blind rotation, pc-py tfhe.rs params)
GRID = 2048 * int(os.environ.get("PC_TFHE_NU", "1") or 1)
GAIN = 2.0           # 2^(out_cf - cf): the finish's output is f / GAIN
MAX_V = 1.25         # the largest |f| (pc-py tfhe.rs MAX_V)
_N32_CAP = 44        # the deepest input level that reaches the StC-first entry after a pending rescale (pcmm.py)
_N32_LAND = 26       # the PCMM / TFHE finish's raw landing (pc-shadow Finish::Tfhe; the fake lands here)

LUTS: dict = {}
MLUTS: dict = {}     # name -> [f_j]: k LUTs interleaved, one blind rotation (Session.tfhe_lut_multi)
SIGNS: dict = {}     # name -> (lo, hi): sign LUTs on the whole torus (inputs in (-2, 2))
SLOTS: dict = {}


def register_lut(name, f):
    """`f`: numpy-vectorized on [-1, 1), |f| <= MAX_V. Re-registering a name replaces it."""
    LUTS[name] = f
    return name


def register_lut_multi(name, fs):
    """k = len(fs) LUTs on one switch (many-LUT): each on a grid of GRID / k points, at k times the input's mod-switch
    noise; `tfhe_lut(..., part=j)` reads LUT j (part 0 runs the switch, part j > 0 must follow it on the same input)."""
    MLUTS[name] = list(fs)
    return name


def register_sign(name, lo, hi):
    """A sign LUT: `lo` for m < 0, `hi` for m >= 0, read on (-2, 2): the negacyclic sign needs no padding half (a
    constant test vector plus an offset), so its input range and resolution are twice a general LUT's."""
    if max(abs(lo), abs(hi)) > MAX_V:
        raise ValueError(f"tfhe sign {name}: |value| > {MAX_V}")
    SIGNS[name] = (float(lo), float(hi))
    return name


def register_slots(name, slots, copies=()):
    """The slots (real payload, < N/2) a LUT reads and writes, and `copies`: slot offsets every output is also
    written at (a layout's replicas: one bootstrap per value, the switch's repack scatters it again); the rest come
    back zero."""
    s = np.unique(np.asarray(slots, dtype=np.int64))
    c = tuple(int(k) for k in copies)
    if name in SLOTS and not (np.array_equal(SLOTS[name][0], s) and SLOTS[name][1] == c):
        raise ValueError(f"tfhe slots {name}: registered with another set")
    SLOTS[name] = (s, c)
    return name


def grid(f, pts=GRID):
    """f on the test vector's grid: m_j = 2 j / pts (j < pts / 2), 2 (j - pts) / pts above (pc-tfhe test_vector)."""
    j = np.arange(pts)
    m = np.where(j < pts // 2, 2.0 * j / pts, 2.0 * (j - pts) / pts)
    v = np.asarray(f(m), dtype=np.float64)
    if not np.all(np.abs(v) <= MAX_V):
        raise ValueError(f"tfhe lut: |f| > {MAX_V} on the grid")
    return v


def quantize(f, m, pts=GRID):
    """f as the switch evaluates it: at the grid point nearest m (|m| < 1)."""
    m = np.asarray(m, dtype=np.float64)
    j = np.rint(m * pts / 2.0)
    return np.asarray(f(2.0 * j / pts), dtype=np.float64)


def _sign_f(lo, hi):
    # the switch rounds m to the grid of 2 / GRID: m >= -1 / GRID reads as non-negative
    return lambda m: np.where(np.rint(np.asarray(m) * GRID / 2.0) >= 0, hi, lo)


# PF_TFHE_NOISE: sigma (in units of the LUT's input) of Gaussian noise the fakes add to the switched inputs, a model of
# the switch's input noise (the mod switch to 2 n_br and the LWE key switch: rms 5.3e-3 of [-1, 1) on the device,
# pc-check tfhe --identity; 7.9e-3 before the n 795 / binary-digit / mean-compensated set; half that in a sign LUT's
# (-2, 2) units)
_NOISE = float(os.environ.get("PF_TFHE_NOISE", "0") or 0)
# PF_TFHE_OUT_NOISE: sigma of the switch's output noise, added to every LUT output by the fakes
_OUT_NOISE = float(os.environ.get("PF_TFHE_OUT_NOISE", "0") or 0)
# PF_TFHE_MLUT=1: tfhe_lut_taylor's two LUTs from one blind rotation (many-LUT; Session.tfhe_lut_multi)
_MLUT = os.environ.get("PF_TFHE_MLUT", "0") not in ("", "0")
_RNG = np.random.default_rng(int(os.environ.get("PF_TFHE_NOISE_SEED", "0") or 0))


def _input_rng(m):
    import hashlib
    h = hashlib.blake2b(np.ascontiguousarray(m).tobytes(), digest_size=8).digest()
    return np.random.default_rng([int.from_bytes(h, "little"), int(os.environ.get("PF_TFHE_NOISE_SEED", "0") or 0)])


def _values(lut, slots, vec, part=None):
    if part is not None:     # LUT j of a many-LUT set, on its coarser grid
        j, k = part
        fj = MLUTS[lut][j]
        f = lambda m, fj=fj, k=k: quantize(fj, m, GRID // k)
    else:
        f = _sign_f(*SIGNS[lut]) if lut in SIGNS else LUTS[lut]
    idx, copies = SLOTS[slots]
    vec = np.asarray(vec)
    n = vec.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n > 1:            # CapCt.novec keeps one slot: the recorder needs the op stream only
        m = np.real(vec[idx])
        if _NOISE:
            # the device's switch is deterministic: LUTs on the same input read the same noise (tfhe_lut_taylor)
            m = m + _input_rng(m).normal(0.0, _NOISE, m.shape)
        v = f(m) if lut in SIGNS or part is not None else quantize(f, m)
        if _OUT_NOISE:
            v = v + _RNG.normal(0.0, _OUT_NOISE, v.shape)
        for c in (0,) + copies:
            out[(idx + c) % n] = v
    return out


def tfhe_lut(ops, x, lut, slots, part=None):
    """f(x) on the slot set `slots`, zero elsewhere (module doc); `part` = j: LUT j of the many-LUT set `lut`."""
    known = lut in MLUTS if part is not None else (lut in LUTS or lut in SIGNS)
    if not known or slots not in SLOTS:
        raise KeyError(f"tfhe_lut: lut {lut!r} (part {part}) / slots {slots!r} not registered")
    pk = None if part is None else (int(part), len(MLUTS[lut]))
    from .ops import NumpyOps
    if isinstance(ops, NumpyOps):
        return _values(lut, slots, x, pk)
    fhe = ops.fhe
    if getattr(ops.core, "bts", None) is not None:
        raise NotImplementedError("tfhe_lut runs in planned sessions (Session(tfhe=...)), not on the eager device path")
    ops.bootstrap_hint(x, _N32_CAP, True)
    fhe.level_hint(x, _N32_CAP)
    names, levels = [fhe.var_name(x)], [ops.level(x)]
    from .fake import FakeCt
    y = FakeCt(_values(lut, slots, x.vec, pk) / GAIN, _N32_LAND, 1, getattr(x, "packing", "cachemir"))
    if hasattr(x, "ptag"):
        y = type(x)(y.vec, y.meta, packing=y.packing)     # the capture's own ciphertext type (CapCt)
        y.ptag = x.ptag
    # "range": the finish's output bound (|f| / GAIN): the capture's magnitude floor, so a pass whose values happen to
    # be all zero (a sign LUT that read only negatives) is not a magnitude-free site the planner must refuse
    if pk is not None:
        rng = float(np.abs(grid(MLUTS[lut][pk[0]], GRID // pk[1])).max())
    else:
        rng = max(abs(v) for v in SIGNS[lut]) if lut in SIGNS else float(np.abs(grid(LUTS[lut])).max())
    desc = {"op": "tfhe", "lut": str(lut), "slots": str(slots), "gain": GAIN, "range": rng / GAIN}
    if pk is not None:
        desc["part"] = list(pk)
    fhe.record_external_bootstrap(y, names, levels, desc=desc)
    return ops.add(y, y)


# ── native programs (pc-py tfhe.rs TfheRt::prog; PC_TFHE_NATIVE=bufs,rows) ─────────────────────────────────────────
#
# One switch in, dependent bootstraps on LWEs (native encoding: v at phase v q / 4), one finish out. Native LUT inputs
# are read on [-1, 1), negacyclic beyond; the mirror's noise: PF_TFHE_NAT_NOISE (input), PF_TFHE_NAT_OUT (output).
PROGS: dict = {}
_NAT_NOISE = float(os.environ.get("PF_TFHE_NAT_NOISE", "") or _NOISE)
_NAT_OUT = float(os.environ.get("PF_TFHE_NAT_OUT", "0") or 0)


def _nat(f, m, pts):
    """f as a native bootstrap reads it: on the grid of `pts` points of [-1, 1), negacyclic beyond."""
    m = ((np.asarray(m, dtype=np.float64) + 1.0) % 4.0) - 1.0
    lo = m < 1.0
    return np.where(lo, quantize(f, np.where(lo, m, 0.0), pts), -quantize(f, np.where(lo, 0.0, m - 2.0), pts))


class Prog:
    """A native program over `bufs` LWE buffers (module section above). Steps run in order; `run` is the mirror."""

    def __init__(self, name, bufs):
        self.name, self.bufs = name, int(bufs)
        self.luts, self.steps = {}, []

    def lut(self, name, *fs, warp=False):
        """Tables f_j (numpy-vectorized on [-1, 1)): k = len(fs) from one rotation; warp: the finish's encoding (a
        store's table, or a pbs whose outputs a raw store keeps)."""
        self.luts[name] = (list(fs), bool(warp))
        return name

    def load(self, lut, dst):
        """`dst`: a buffer, or a list of one per table of `lut`."""
        self.steps.append({"op": "load", "lut": lut, "dst": dst})

    def lin(self, dst, a, b, pairs, ca=1, cb=1, c=0.0):
        """dst[r] = ca a[ia] + cb b[ib] + c for (ia, ib) = pairs[r] (-1: no term); ca, cb integers."""
        idx = [int(v) for p in pairs for v in p]
        self.steps.append({"op": "lin", "dst": dst, "a": a, "b": b, "idx": idx, "ca": int(ca), "cb": int(cb), "c": float(c)})

    def pbs(self, src, rows, lut, dst):
        self.steps.append({"op": "pbs", "src": src, "rows": int(rows), "lut": lut, "dst": list(dst)})

    def store(self, src, rows, lut, slots):
        """`lut` None: the rows as they are (a warp pbs's outputs)."""
        self.steps.append({"op": "store", "src": src, "rows": int(rows), "lut": lut, "slots": slots})

    def rows_out(self):
        return self.steps[-1]["rows"]

    def json(self):
        import json
        luts = {}
        for name, (fs, warp) in self.luts.items():
            pts = GRID // len(fs)
            j = np.arange(pts)
            m = np.where(j < pts // 2, 2.0 * j / pts, 2.0 * (j - pts) / pts)
            luts[name] = {"tables": [[float(v) for v in np.asarray(f(m), dtype=np.float64)] for f in fs], "warp": warp}
        return json.dumps({"luts": luts, "steps": self.steps})

    def run(self, m, noise=None):
        """The mirror on the loaded values `m`: the stored values."""
        noise = _NAT_NOISE if noise is None else noise
        rng = _input_rng(np.asarray(m))
        B = [np.zeros(0) for _ in range(self.bufs)]
        out = None
        for st in self.steps:
            op = st["op"]
            if op == "load":
                fs = self.luts[st["lut"]][0]
                x = np.asarray(m, dtype=np.float64)
                x = x + rng.normal(0.0, _NOISE * len(fs), x.shape) if _NOISE else x
                for d, f in zip(st["dst"] if isinstance(st["dst"], list) else [st["dst"]], fs):
                    B[d] = _nat(f, x, GRID // len(fs))
            elif op == "lin":
                idx = np.asarray(st["idx"], dtype=np.int64).reshape(-1, 2)
                a, b = B[st["a"]], B[st["b"]]
                ta = np.where(idx[:, 0] >= 0, a[np.maximum(idx[:, 0], 0)] if a.size else 0.0, 0.0)
                tb = np.where(idx[:, 1] >= 0, b[np.maximum(idx[:, 1], 0)] if b.size else 0.0, 0.0)
                B[st["dst"]] = st["ca"] * ta + st["cb"] * tb + st["c"]
            elif op == "store" and st["lut"] is None:
                x = B[st["src"]][:st["rows"]]
                out = x + (_RNG.normal(0.0, _OUT_NOISE, x.shape) if _OUT_NOISE else 0.0)
            else:
                fs, warp = self.luts[st["lut"]]
                x = B[st["src"]][:st["rows"]]
                x = x + rng.normal(0.0, noise * len(fs), x.shape) if noise else x
                ys = [_nat(f, x, GRID // len(fs)) for f in fs]
                if op == "pbs":
                    for d, y in zip(st["dst"], ys):
                        B[d] = y + rng.normal(0.0, _NAT_OUT, y.shape) if _NAT_OUT else y
                else:
                    out = ys[0] + (_RNG.normal(0.0, _OUT_NOISE, x.shape) if _OUT_NOISE else 0.0)
        return out


def register_prog(prog):
    PROGS[prog.name] = prog
    return prog.name


def tfhe_prog(ops, x, prog, slots):
    """Native program `prog` (`register_prog`) on the slot set `slots` (its load); the output on the set its store
    names, zero elsewhere."""
    if prog not in PROGS or slots not in SLOTS:
        raise KeyError(f"tfhe_prog: program {prog!r} / slots {slots!r} not registered")
    pr = PROGS[prog]
    out_set = pr.steps[-1]["slots"]

    def values(vec):
        vec = np.asarray(vec)
        n = vec.shape[0]
        out = np.zeros(n, dtype=np.float64)
        if n > 1:
            idx, _ = SLOTS[slots]
            v = pr.run(np.real(vec[idx]))
            oi, oc = SLOTS[out_set]
            for c in (0,) + oc:
                out[(oi + c) % n] = v
        return out
    from .ops import NumpyOps
    if isinstance(ops, NumpyOps):
        return values(x)
    fhe = ops.fhe
    if getattr(ops.core, "bts", None) is not None:
        raise NotImplementedError("tfhe_prog runs in planned sessions (Session(tfhe=...)), not on the eager device path")
    ops.bootstrap_hint(x, _N32_CAP, True)
    fhe.level_hint(x, _N32_CAP)
    names, levels = [fhe.var_name(x)], [ops.level(x)]
    from .fake import FakeCt
    y = FakeCt(values(x.vec) / GAIN, _N32_LAND, 1, getattr(x, "packing", "cachemir"))
    if hasattr(x, "ptag"):
        y = type(x)(y.vec, y.meta, packing=y.packing)
        y.ptag = x.ptag
    last = pr.steps[-1]["lut"]                 # a raw store: its range is the program's warp tables'
    rng = 0.0
    for name, (fs, warp) in pr.luts.items():
        if name == last or (last is None and warp):
            pts = GRID // len(fs)
            j = np.arange(pts)
            rng = max(rng, max(float(np.abs(f(np.where(j < pts // 2, 2.0 * j / pts, 2.0 * (j - pts) / pts))).max())
                               for f in fs))
    fhe.record_external_bootstrap(y, names, levels, desc={"op": "tfhe", "prog": str(prog), "slots": str(slots), "gain": GAIN, "range": rng / GAIN})
    return ops.add(y, y)


def tfhe_lut_taylor(ops, x, name, f, df, slots):
    """f(x) on `slots` with the switch's input error corrected to first order (after Crasson, Mehats, ePrint 2026/1533):
    f(x) ~ g(m~) + f'(m~) x with g = f - f' m, both LUTs reading the same rounded m~."""
    g, d, gd = f"{name}/g", f"{name}/d", f"{name}/gd"
    # f' / sd fits the LUT range; sd comes back as a constant on the product
    j = np.arange(GRID)
    sd = max(1.0, float(np.abs(df(np.where(j < GRID // 2, 2.0 * j / GRID, 2.0 * (j - GRID) / GRID))).max()) / MAX_V)
    if g not in LUTS:
        register_lut(g, lambda m: np.asarray(f(m)) - np.asarray(df(m)) * np.asarray(m))
        register_lut(d, lambda m: np.asarray(df(m)) / sd)
        register_lut_multi(gd, [LUTS[g], LUTS[d]])
    from .ops import NumpyOps
    if isinstance(ops, NumpyOps):
        # the mirror: one noise draw for both LUTs (they share m~)
        idx, copies = SLOTS[slots]
        vec = np.asarray(x)
        n = vec.shape[0]
        out = np.zeros(n, dtype=np.float64)
        m = np.real(vec[idx])
        mt = m + _input_rng(m).normal(0.0, _NOISE, m.shape) if _NOISE else m
        on = (lambda: _RNG.normal(0.0, _OUT_NOISE, m.shape)) if _OUT_NOISE else (lambda: 0.0)
        pts = GRID // 2 if _MLUT else GRID
        v = quantize(LUTS[g], mt, pts) + on() + (quantize(LUTS[d], mt, pts) + on()) * sd * m
        for c in (0,) + copies:
            out[(idx + c) % n] = v
        return out
    if _MLUT:      # both from one blind rotation: part 0 first (it runs the switch), part 1 reads its second output
        yg, yd = tfhe_lut(ops, x, gd, slots, part=0), tfhe_lut(ops, x, gd, slots, part=1)
    else:
        yg, yd = tfhe_lut(ops, x, g, slots), tfhe_lut(ops, x, d, slots)
    p = ops.mult(yd, x)
    return ops.add(yg, p if sd == 1.0 else ops.mult_const(p, sd, 0.0))


def setup_session(sess):
    """Register every LUT and slot set the capture used with a planned session (`Session(tfhe=...)`)."""
    for name, f in LUTS.items():
        sess.tfhe_lut(name, [float(v) for v in grid(f)])
    for name, fs in MLUTS.items():
        sess.tfhe_lut_multi(name, [[float(v) for v in grid(f, GRID // len(fs))] for f in fs])
    for name, (lo, hi) in SIGNS.items():
        sess.tfhe_lut_sign(name, lo, hi)
    for name, (s, c) in SLOTS.items():
        sess.tfhe_slots(name, [int(i) for i in s], list(c))
    for name, pr in PROGS.items():
        sess.tfhe_prog(name, pr.json())


# ── common tables ─────────────────────────────────────────────────────────────────────────────────────────────────

def step(m):
    """(1 + sign m) / 2: 1 on [0, 1), 0 on [-1, 0)."""
    return (np.asarray(m) >= 0).astype(np.float64)


def relu(m):
    return np.maximum(np.asarray(m), 0.0)


def scaled_step(g):
    """The sign LUT (0, g) (|g| <= MAX_V; inputs in (-2, 2)): a ReLU's gain folded into its sign."""
    return register_sign(f"step*{float(g):.9g}", 0.0, float(g))


register_lut("step", step)
register_lut("relu", relu)
