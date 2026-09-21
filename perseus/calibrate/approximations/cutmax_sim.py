import numpy as np

np.seterr(over="ignore", invalid="ignore")

FLOOR = WALL_X = WALL_Y = CONV = WALL_VEC = None


def set_envelopes(knobs):
    global FLOOR, WALL_X, WALL_Y, CONV, WALL_VEC
    FLOOR = {1: knobs["floor1"], 2: knobs["floor2"]}
    WALL_X = {1: knobs["wall_x1"], 2: knobs["wall_x2"]}
    WALL_Y = {1: knobs["wall_y1"], 2: knobs["wall_y2"]}
    CONV = knobs["conv"]
    WALL_VEC = knobs["wall_vec"]


def noisy(rng, x, rel, absf):
    return x + rng.normal(0, rel, x.shape) * np.abs(x) \
             + rng.normal(0, absf, x.shape)


def pick_prescale(lo, hi, floor_safety):
    for iters in (1, 2):
        g_min = hi / WALL_X[iters]                   # ceiling
        g_max = lo / (floor_safety * FLOOR[iters])   # floor
        if g_min <= g_max:
            return float(np.sqrt(g_min * g_max)), iters
    return float(hi / WALL_X[2]), 2


def chord(lo, hi):
    va, vb = lo ** -0.5, hi ** -0.5
    beta = (va - vb) / (hi - lo)
    alpha = va + beta * lo
    grid = np.geomspace(lo, hi, 2048)
    gamma = (grid ** -0.5 / (alpha - beta * grid)).min()
    return gamma * alpha, gamma * beta


def cascade(s2, lo, hi, k, passes, g=None, chord_init=True, rng=None, rel=0.0,
            absf=0.0, wall_x=None, wall_y=None):
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
