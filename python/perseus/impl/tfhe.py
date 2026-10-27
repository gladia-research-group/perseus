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

GRID = 2048          # n_br: the test vector's points over [-1, 1)
GAIN = 2.0           # 2^(out_cf - cf): the finish's output is f / GAIN
MAX_V = 1.25         # the largest |f| (pc-py tfhe.rs MAX_V)
_N32_CAP = 44        # the deepest input level that reaches the StC-first entry after a pending rescale (pcmm.py)
_N32_LAND = 26       # the PCMM / TFHE finish's raw landing (pc-shadow Finish::Tfhe; the fake lands here)

LUTS: dict = {}
SIGNS: dict = {}     # name -> (lo, hi): sign LUTs on the whole torus (inputs in (-2, 2))
SLOTS: dict = {}


def register_lut(name, f):
    """`f`: numpy-vectorized on [-1, 1), |f| <= MAX_V. Re-registering a name replaces it."""
    LUTS[name] = f
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


def grid(f):
    """f on the test vector's grid: m_j = 2 j / GRID (j < GRID / 2), 2 (j - GRID) / GRID above (pc-tfhe test_vector)."""
    j = np.arange(GRID)
    m = np.where(j < GRID // 2, 2.0 * j / GRID, 2.0 * (j - GRID) / GRID)
    v = np.asarray(f(m), dtype=np.float64)
    if not np.all(np.abs(v) <= MAX_V):
        raise ValueError(f"tfhe lut: |f| > {MAX_V} on the grid")
    return v


def quantize(f, m):
    """f as the switch evaluates it: at the grid point nearest m (|m| < 1)."""
    m = np.asarray(m, dtype=np.float64)
    j = np.rint(m * GRID / 2.0)
    return np.asarray(f(2.0 * j / GRID), dtype=np.float64)


def _sign_f(lo, hi):
    # the switch rounds m to the grid of 2 / GRID: m >= -1 / GRID reads as non-negative
    return lambda m: np.where(np.rint(np.asarray(m) * GRID / 2.0) >= 0, hi, lo)


# PF_TFHE_NOISE: sigma (in units of the LUT's input) of Gaussian noise the fakes add to the switched inputs, a model of
# the switch's input noise (the mod switch to 2 n_br and the LWE key switch: rms 5.3e-3 of [-1, 1) on the device,
# pc-check tfhe --identity; 7.9e-3 before the n 795 / binary-digit / mean-compensated set; half that in a sign LUT's
# (-2, 2) units)
_NOISE = float(os.environ.get("PF_TFHE_NOISE", "0") or 0)
_RNG = np.random.default_rng(int(os.environ.get("PF_TFHE_NOISE_SEED", "0") or 0))


def _values(lut, slots, vec):
    f = _sign_f(*SIGNS[lut]) if lut in SIGNS else LUTS[lut]
    idx, copies = SLOTS[slots]
    vec = np.asarray(vec)
    n = vec.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n > 1:            # CapCt.novec keeps one slot: the recorder needs the op stream only
        m = np.real(vec[idx])
        if _NOISE:
            m = m + _RNG.normal(0.0, _NOISE, m.shape)
        v = f(m) if lut in SIGNS else quantize(f, m)
        for c in (0,) + copies:
            out[(idx + c) % n] = v
    return out


def tfhe_lut(ops, x, lut, slots):
    """f(x) on the slot set `slots`, zero elsewhere (module doc)."""
    if (lut not in LUTS and lut not in SIGNS) or slots not in SLOTS:
        raise KeyError(f"tfhe_lut: lut {lut!r} / slots {slots!r} not registered")
    from .ops import NumpyOps
    if isinstance(ops, NumpyOps):
        return _values(lut, slots, x)
    fhe = ops.fhe
    if getattr(ops.core, "bts", None) is not None:
        raise NotImplementedError("tfhe_lut runs in planned sessions (Session(tfhe=...)), not on the eager device path")
    ops.bootstrap_hint(x, _N32_CAP, True)
    fhe.level_hint(x, _N32_CAP)
    names, levels = [fhe.var_name(x)], [ops.level(x)]
    from .fake import FakeCt
    y = FakeCt(_values(lut, slots, x.vec) / GAIN, _N32_LAND, 1, getattr(x, "packing", "cachemir"))
    if hasattr(x, "ptag"):
        y = type(x)(y.vec, y.meta, packing=y.packing)     # the capture's own ciphertext type (CapCt)
        y.ptag = x.ptag
    # "range": the finish's output bound (|f| / GAIN): the capture's magnitude floor, so a pass whose values happen to
    # be all zero (a sign LUT that read only negatives) is not a magnitude-free site the planner must refuse
    rng = max(abs(v) for v in SIGNS[lut]) if lut in SIGNS else float(np.abs(grid(LUTS[lut])).max())
    fhe.record_external_bootstrap(y, names, levels, desc={"op": "tfhe", "lut": str(lut), "slots": str(slots), "gain": GAIN,
                                                          "range": rng / GAIN})
    return ops.add(y, y)


def setup_session(sess):
    """Register every LUT and slot set the capture used with a planned session (`Session(tfhe=...)`)."""
    for name, f in LUTS.items():
        sess.tfhe_lut(name, [float(v) for v in grid(f)])
    for name, (lo, hi) in SIGNS.items():
        sess.tfhe_lut_sign(name, lo, hi)
    for name, (s, c) in SLOTS.items():
        sess.tfhe_slots(name, [int(i) for i in s], list(c))


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
