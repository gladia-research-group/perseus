"""Plaintext replica of the runtime CutMax op (numpy only).

The mechanics behind the schedule calibration (`cutmax.py`): the cascaded
inv_sqrt (Newton passes, chord / const from-below inits, geo-mid prescale)
and the packed iteration simulator `Sim` — with the MEASURED bootstrap
envelopes (abs floor + value walls per cascade bts regime) injectable at
every scalar refresh and vector shift point via `noisy`.
"""
import numpy as np

np.seterr(over="ignore", invalid="ignore")  

# Unarmed until set_envelopes: the measured bts envelopes live in the yaml
# (cutmax_floor1/... knobs) — using the sim without arming them fails loudly.
FLOOR = WALL_X = WALL_Y = CONV = WALL_VEC = None


def set_envelopes(knobs):
    """Measured-envelope knobs (yaml cutmax_floor1/... fields) -> module state."""
    global FLOOR, WALL_X, WALL_Y, CONV, WALL_VEC
    FLOOR = {1: knobs["floor1"], 2: knobs["floor2"]}
    WALL_X = {1: knobs["wall_x1"], 2: knobs["wall_x2"]}
    WALL_Y = {1: knobs["wall_y1"], 2: knobs["wall_y2"]}
    CONV = knobs["conv"]
    WALL_VEC = knobs["wall_vec"]


def noisy(rng, x, rel, absf):
    """One bts refresh: relative AND absolute-floor noise (the absolute floor
    is what killed the two live bts1 attempts; relative-only blessed them)."""
    return x + rng.normal(0, rel, x.shape) * np.abs(x) \
             + rng.normal(0, absf, x.shape)


def pick_prescale(lo, hi, floor_safety):
    """Mixed-precision prescale: the scalar band must fit
    [floor_safety*floor, WALL_X] after x/g (floor_safety = the SNR margin
    against the ABS floor — the mixed-gate explosions were laggard x at
    SNR~1.4 flipping negative between passes; keep >= 5). Returns
    (g, cascade_iters). If even the 2-iter budget fails, ceiling wins.
    """
    for iters in (1, 2):
        g_min = hi / WALL_X[iters]                   # ceiling
        g_max = lo / (floor_safety * FLOOR[iters])   # floor
        if g_min <= g_max:
            return float(np.sqrt(g_min * g_max)), iters
    return float(hi / WALL_X[2]), 2


def chord(lo, hi):
    """From-below chord of x^-1/2 over [lo, hi]; returns folded (a, b) with
    y0 = a - b*x guaranteed <= x^-1/2 on the band."""
    va, vb = lo ** -0.5, hi ** -0.5
    beta = (va - vb) / (hi - lo)
    alpha = va + beta * lo
    grid = np.geomspace(lo, hi, 2048)
    gamma = (grid ** -0.5 / (alpha - beta * grid)).min()
    return gamma * alpha, gamma * beta


def cascade(s2, lo, hi, k, passes, g=None, chord_init=True, rng=None, rel=0.0,
            absf=0.0, wall_x=None, wall_y=None):
    """Explicit cascaded inv_sqrt on per-token scalars. `chord_init`: True =
    fit the chord here, False = const from-below 1/sqrt(xhi), (ca, cb) tuple =
    exact emitted init (ca==0 -> legacy ones). Returns (inv_sigma, max ct
    value, crossed): crossed = per-row wall breach at any cascade bts site
    (None when walls not given)."""
    if g is None:
        g = np.sqrt(lo * hi)
    x = s2 / g
    xlo, xhi = lo / g, hi / g
    if isinstance(chord_init, tuple):
        ca, cb = chord_init
        y = np.full_like(x, 1.0) if ca == 0.0 else ca - cb * x
    elif chord_init:
        ca, cb = chord(xlo, xhi)
        y = ca - cb * x
    else:
        y = np.full_like(x, 1.0 / np.sqrt(xhi))
    vmax = np.abs(y).max()
    ymax_r, xmax_r = np.abs(y), np.abs(x)
    u_prod = np.ones_like(x)
    for j in range(passes):
        for _ in range(k):
            y = y * (3.0 - x * y * y) / 2.0
            vmax = max(vmax, np.abs(y).max())
            ymax_r = np.maximum(ymax_r, np.abs(y))
        if rng is not None:                       # u refresh bts
            y = noisy(rng, y, rel, absf)
        u_prod = u_prod * y
        if j + 1 < passes:
            x = x * y * y
            if rng is not None:                   # chain autos
                x = noisy(rng, x, rel, absf)
            xmax_r = np.maximum(xmax_r, np.abs(x))
            y = np.ones_like(x)                   # const from-below (top ~1)
    crossed = None
    if wall_x is not None:
        crossed = (ymax_r > wall_y) | (xmax_r > wall_x)
    return u_prod / np.sqrt(g), vmax, crossed


class Sim:
    """Joint plaintext simulation of the packed cutmax over all rows."""

    def __init__(self, rows):
        self.vocab = rows.shape[1]
        self.ys = [r.astype(np.float64).copy() for r in rows]
        self.crossed = np.zeros(rows.shape[0], dtype=bool)   # wall-replay flags

    def s2_band(self):
        self.pre, s2s = [], []
        for y in self.ys:
            mu = y.sum() / self.vocab
            d = y - mu
            s2 = (d * d).sum() / self.vocab
            self.pre.append((d, s2))
            s2s.append(s2)
        return min(s2s), max(s2s)

    def step(self, p, c, m, band, k, passes, g=None, chord_init=True, rng=None,
             rel=0.0, absf=0.0, casc_rel=None, casc_absf=None,
             wall_x=None, wall_y=None):
        lo, hi = band
        s2 = np.array([s2_ for _, s2_ in self.pre])
        inv_sigma, vmax, crossed = cascade(
            s2, lo, hi, k, passes, g, chord_init, rng,
            rel if casc_rel is None else casc_rel,
            absf if casc_absf is None else casc_absf,
            wall_x=wall_x, wall_y=wall_y)
        if crossed is not None:
            self.crossed |= crossed
        for i in range(len(self.ys)):
            d, _ = self.pre[i]
            ynew = d * inv_sigma[i] / (c * m) + 1.0 / m
            if wall_x is not None and np.abs(ynew).max() > WALL_VEC:
                self.crossed[i] = True            # shift-point bts input breach
            if rng is not None:                   # shift-point vector bts
                ynew = noisy(rng, ynew, rel, absf)
            self.ys[i] = ynew ** p
        return vmax

    def masses(self):
        return np.array([np.sort(y)[-1] / y.sum() for y in self.ys])

    def finalize(self, rng=None, rel=0.0, absf=0.0):
        zs, sums = [], []
        for y in self.ys:
            tot = y.sum()
            if rng is not None:               # sum-lane bts abs noise (the GS
                tot += rng.normal(0.0, absf)  # chain bootstraps S itself)
            sums.append(tot)
            inv = 1.0 / tot
            if rng is not None:
                inv *= 1.0 + rng.normal(0.0, rel)
            z = y * inv
            if rng is not None:
                z = z + rng.normal(0, absf, z.shape)
            zs.append(z)
        return zs, (float(min(sums)), float(max(sums))), sums
