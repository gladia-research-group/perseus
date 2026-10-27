"""Lookup tables by CKKS -> TFHE -> CKKS scheme switching (pc-tfhe; device: pc-gpu tfhe.rs; planned service:
pc-py tfhe.rs): y = f(x) on a set of real slots, every other slot an exact zero, as one refresh. The switch takes the
ciphertext at the StC-first split of a dense refresh, turns the chosen slots into LWEs, evaluates f by programmable
bootstrapping (exact for steps: no polynomial gap) and packs the outputs back for the refresh's finish.

`tfhe_lut(ops, x, lut, slots)`: `lut` names a function registered with `register_lut` (|f| <= 1.25 on [-1, 1): the
output warp's range at out_cf 3), `slots`
a slot set registered with `register_slots`; x must hold values in [-1, 1) on the set (they wrap beyond), and inputs
within ~1-2 % of the range of a discontinuity may land on either side (the switch's input noise). The capture records
one external refresh (`{op: "tfhe", lut, slots}`) landing where a PCMM output does; the planned driver registers the
tables and sets with the session (`setup_session`). The finish reads v / 2^(out_cf - cf) = v / 2: one ciphertext add
folds the 2 back. The numpy mirror and the fakes compute f exactly. Not on the eager device path (no refresh there):
plan it.
"""
from __future__ import annotations

import numpy as np

GRID = 2048          # n_br: the test vector's points over [-1, 1)
GAIN = 2.0           # 2^(out_cf - cf): the finish's output is f / GAIN
MAX_V = 1.25         # the largest |f| (pc-py tfhe.rs MAX_V)
_N32_CAP = 44        # the deepest input level that reaches the StC-first entry after a pending rescale (pcmm.py)
_N32_LAND = 26       # the PCMM / TFHE finish's raw landing (pc-shadow Finish::Tfhe; the fake lands here)

LUTS: dict = {}
SLOTS: dict = {}


def register_lut(name, f):
    """`f`: numpy-vectorized on [-1, 1), |f| <= MAX_V. Re-registering a name replaces it."""
    LUTS[name] = f
    return name


def register_slots(name, slots):
    """The slots (real payload, < N/2) a LUT reads and writes; the rest come back zero."""
    s = np.unique(np.asarray(slots, dtype=np.int64))
    if name in SLOTS and not np.array_equal(SLOTS[name], s):
        raise ValueError(f"tfhe slots {name}: registered with another set")
    SLOTS[name] = s
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


def _values(lut, slots, vec):
    f, idx = LUTS[lut], SLOTS[slots]
    vec = np.asarray(vec)
    out = np.zeros(vec.shape[0], dtype=np.float64)
    if vec.shape[0] > 1:            # CapCt.novec keeps one slot: the recorder needs the op stream only
        out[idx] = quantize(f, np.real(vec[idx]))
    return out


def tfhe_lut(ops, x, lut, slots):
    """f(x) on the slot set `slots`, zero elsewhere (module doc)."""
    if lut not in LUTS or slots not in SLOTS:
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
    fhe.record_external_bootstrap(y, names, levels, desc={"op": "tfhe", "lut": str(lut), "slots": str(slots), "gain": GAIN})
    return ops.add(y, y)


def setup_session(sess):
    """Register every LUT and slot set the capture used with a planned session (`Session(tfhe=...)`)."""
    for name, f in LUTS.items():
        sess.tfhe_lut(name, [float(v) for v in grid(f)])
    for name, s in SLOTS.items():
        sess.tfhe_slots(name, [int(i) for i in s])


# ── common tables ─────────────────────────────────────────────────────────────────────────────────────────────────

def step(m):
    """(1 + sign m) / 2: 1 on [0, 1), 0 on [-1, 0)."""
    return (np.asarray(m) >= 0).astype(np.float64)


def relu(m):
    return np.maximum(np.asarray(m), 0.0)


def scaled_step(g):
    """The LUT g * step (registered once per g; |g| <= MAX_V): a ReLU's gain folded into its sign."""
    name = f"step*{float(g):.9g}"
    if name not in LUTS:
        register_lut(name, lambda m, g=float(g): g * step(m))
    return name


register_lut("step", step)
register_lut("relu", relu)
