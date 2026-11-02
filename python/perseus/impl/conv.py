"""Conv2d on the gapped channel-major layout: each channel owns one GRID x GRID block of S slots, pixel (h, w) of a
feature map at gap g sits at block slot g h GRID + g w, so a stride-2 conv needs no repack (the next layer reads at
gap 2 g). A ciphertext holds `cpc` channel blocks (a power of two, replicated to fill the slots, so channel rotations
are cyclic); wider maps take several ciphertexts.

y = sum_d sum_tap P_{d,tap} * rot_{d S + tap}(x): the diagonal method over channel offsets d, BSGS d = a + b m with the
9 m baby rotations a S + tap hoisted once per input ciphertext. P carries the weight (BN folded), the padding mask and
the output-validity mask, so every slot that is not a valid output pixel is an exact 0 and one conv costs one level."""
from __future__ import annotations

import dataclasses

import numpy as np

GRID = 32
S = GRID * GRID


@dataclasses.dataclass(frozen=True)
class Geom:
    """C channels of H x H pixels at gap g in ciphertexts of `slots` slots."""
    C: int
    H: int
    g: int
    slots: int = 1 << 15

    @property
    def Q(self):
        """Channel blocks per ciphertext."""
        return self.slots // S

    @property
    def cpc(self):
        """Channels per ciphertext: the next power of two, at most Q (block q holds channel q mod cpc)."""
        return min(1 << max(self.C - 1, 0).bit_length(), self.Q)

    @property
    def n_ct(self):
        return -(-self.C // self.cpc)

    def pixel_mask(self):
        """(S,) 1 at the block slots of the H x H valid pixels."""
        m = np.zeros((GRID, GRID))
        m[:self.g * self.H:self.g, :self.g * self.H:self.g] = 1.0
        return m.reshape(-1)

    def pack(self, x):
        """(C, H, H) -> n_ct slot vectors (gaps and padded channels 0)."""
        x = np.asarray(x, dtype=np.float64)
        assert x.shape == (self.C, self.H, self.H), x.shape
        out = []
        for j in range(self.n_ct):
            v = np.zeros((self.Q, GRID, GRID))
            for q in range(self.Q):
                c = j * self.cpc + q % self.cpc
                if c < self.C:
                    v[q, :self.g * self.H:self.g, :self.g * self.H:self.g] = x[c]
            out.append(v.reshape(-1))
        return out

    def unpack(self, vecs):
        """n_ct slot vectors -> (C, H, H), read from each channel's first block."""
        x = np.zeros((self.C, self.H, self.H))
        for c in range(self.C):
            v = np.asarray(vecs[c // self.cpc]).real.reshape(self.Q, GRID, GRID)
            x[c] = v[c % self.cpc, :self.g * self.H:self.g, :self.g * self.H:self.g]
        return x


@dataclasses.dataclass
class ConvSpec:
    """A k x k conv (stride s, padding k // 2) from geometry `gi` to `go` (go.g = gi.g s, go.H = ceil(gi.H / s)).
    The diagonals are built on demand (`diag`), a stage-3 conv has ~1k of them."""
    W: np.ndarray            # (Co, Ci, k, k)
    b: np.ndarray            # (Co,)
    gi: Geom
    go: Geom
    stride: int
    m: int                   # baby steps over the channel offsets
    name: str | None = None  # plaintext cache prefix

    @classmethod
    def encode(cls, W, b, gi: Geom, stride=1, m=None, name=None, go: Geom | None = None):
        W = np.asarray(W, dtype=np.float64)
        Co, Ci, k, _ = W.shape
        assert Ci == gi.C, (W.shape, gi)
        if go is None:
            go = Geom(Co, -(-gi.H // stride), gi.g * stride, gi.slots)
        assert go.C == Co and go.g == gi.g * stride and go.H * stride >= gi.H, (gi, go, stride)
        D = gi.cpc
        if m is None:
            m = 1 << (D.bit_length() - 1) // 2       # ~sqrt(D), a power of two dividing D
        assert D % m == 0, (D, m)
        b = np.zeros(Co) if b is None else np.asarray(b, dtype=np.float64)
        return cls(W, b, gi, go, int(stride), int(m), name)

    @property
    def k(self):
        return self.W.shape[-1]

    @property
    def D(self):
        return self.gi.cpc

    def taps(self):
        """[(i, j, rotation)] of the kernel's taps (rotation in input-gap units)."""
        p, g = self.k // 2, self.gi.g
        return [(i, j, g * ((i - p) * GRID + (j - p))) for i in range(self.k) for j in range(self.k)]

    def _tap_mask(self, i, j):
        """(S,) 1 at the valid output pixels whose tap (i, j) input pixel is inside the map."""
        p, s, go = self.k // 2, self.stride, self.go
        hs = s * np.arange(go.H) + i - p
        ws = s * np.arange(go.H) + j - p
        ok = ((hs >= 0) & (hs < self.gi.H))[:, None] & ((ws >= 0) & (ws < self.gi.H))[None, :]
        m = np.zeros((GRID, GRID))
        m[:go.g * go.H:go.g, :go.g * go.H:go.g] = ok
        return m.reshape(-1)

    def diag(self, jo, ji, d, t):
        """P_{d, tap t} for output ciphertext jo / input ciphertext ji, pre-rotated for d's giant step."""
        go, gi = self.go, self.gi
        i, j, _ = self.taps()[t]
        q = np.arange(go.Q)
        co = jo * go.cpc + q % go.cpc
        ci = ji * gi.cpc + ((q + d) % gi.Q) % gi.cpc
        ok = (co < go.C) & (ci < gi.C)
        w = np.where(ok, self.W[np.minimum(co, go.C - 1), np.minimum(ci, gi.C - 1), i, j], 0.0)
        P = (w[:, None] * self._tap_mask(i, j)[None, :]).reshape(-1)
        return np.roll(P, (d // self.m) * self.m * S)

    def nonzero(self, jo, ji, d, t):
        """False when the diagonal is all zeros (padded channels / a tap off the map)."""
        go, gi = self.go, self.gi
        i, j, _ = self.taps()[t]
        q = np.arange(go.Q)
        co = jo * go.cpc + q % go.cpc
        ci = ji * gi.cpc + ((q + d) % gi.Q) % gi.cpc
        ok = (co < go.C) & (ci < gi.C)
        return bool(ok.any() and np.any(self.W[co[ok], ci[ok], i, j]) and self._tap_mask(i, j).any())

    def bias_vec(self, jo):
        go = self.go
        q = np.arange(go.Q)
        co = jo * go.cpc + q % go.cpc
        bb = np.where(co < go.C, self.b[np.minimum(co, go.C - 1)], 0.0)
        return (bb[:, None] * go.pixel_mask()[None, :]).reshape(-1)

    def key(self, *idx):
        return None if self.name is None else ".".join([self.name, *map(str, idx)])

    def rotations(self):
        """Every rotation step the conv issues (for the key scope)."""
        steps = {a * S + r for a in range(self.m) for _i, _j, r in self.taps()}
        steps |= {b * self.m * S for b in range(1, self.D // self.m)}
        return sorted(s % self.gi.slots for s in steps if s % self.gi.slots)


def conv2d(ops, xs, spec: ConvSpec):
    """`spec` applied to the input ciphertexts `xs` (spec.gi.n_ct of them); returns spec.go.n_ct ciphertexts."""
    assert len(xs) == spec.gi.n_ct, (len(xs), spec.gi)
    taps = spec.taps()
    m, G = spec.m, spec.D // spec.m
    babies = []
    for x in xs:
        steps = [a * S + r for a in range(m) for _i, _j, r in taps]
        rots = ops.rotate_many(x, [s for s in steps if s])
        it = iter(rots)
        babies.append([x if s == 0 else next(it) for s in steps])
    ys = []
    for jo in range(spec.go.n_ct):
        acc = None
        for gs in range(G):
            part = None
            for ji in range(len(xs)):
                for a in range(m):
                    d = gs * m + a
                    for t in range(len(taps)):
                        if not spec.nonzero(jo, ji, d, t):
                            continue
                        term = ops.mult_pt(babies[ji][a * len(taps) + t], spec.diag(jo, ji, d, t),
                                           key=spec.key(jo, ji, d, t))
                        if part is None:
                            part = term
                        else:
                            ops.inplace_add(part, term)
            if part is None:
                continue
            if gs:
                part = ops.rotate(part, gs * m * S)
            if acc is None:
                acc = part
            else:
                ops.inplace_add(acc, part)
        if np.any(spec.b):
            acc = ops.add_pt(acc, spec.bias_vec(jo), key=spec.key(jo, "bias"))
        ys.append(acc)
    return ys


def channel_sum(ops, xs):
    """Each channel's pixel sum at its block's slot 0 (the other slots are partial sums: read only slot 0)."""
    return [ops.rotate_and_sum(x, 1, S) for x in xs]


def fold_bn(W, b, bn_w, bn_b, mean, var, eps=1e-5):
    """conv -> BatchNorm (eval) as one conv."""
    s = np.asarray(bn_w) / np.sqrt(np.asarray(var) + eps)
    b = np.zeros(len(s)) if b is None else np.asarray(b)
    return np.asarray(W) * s[:, None, None, None], (b - np.asarray(mean)) * s + np.asarray(bn_b)


def head_geoms(C, n_cls, slots=1 << 15):
    """The pooled map (C channels, one pixel at block slot 0) and the logits geometry."""
    return Geom(C, 1, GRID, slots), Geom(n_cls, 1, GRID, slots)

