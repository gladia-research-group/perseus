"""ResNet-20 from the conv primitive (impl/conv.py): stem, 9 residual blocks, pooled 1x1 head, one stage each. The
activation is a hook `act(ops, cts, site, gain, geom=) -> gain * ReLU(cts)` (the exact max on the mirror, the
composite-sign ReLU or the TFHE sign encrypted, the gain folded into its last series / its LUT; `geom` the activation's
layout).

With `scales` ({site: B}) every ReLU input arrives divided by its own B (the producing conv carries the 1 / B) and
every activation is carried divided by a known scale (the consuming conv carries it back, the head the last one). An
activation that feeds an identity shortcut is carried at the scale of the add it meets (the next block's a2 B: the
producing ReLU's gain), so the shortcut needs no product. The ReLU then reads its input on [-1, 1] with no level for the
map (impl/relu.relu_unit); ReLU being positively homogeneous, the exact max is unchanged."""
from __future__ import annotations

import numpy as np

from perseus.impl.conv import Geom, ConvSpec, conv2d, channel_sum, head_geoms

from . import weights as R


def relu_exact(ops, xs, site, gain=1.0, geom=None):
    return [gain * np.maximum(x.real, 0.0) for x in xs]


class ResNet20:
    def __init__(self, sd, act=relu_exact, slots=1 << 15, scales=None, refresh=False, one_stage=False, argmax=False,
                 argmax_scale=1 / 16):
        """`refresh`: a deliberate bootstrap of every block / head stage's inputs on entry (the planner reads a stage's
        exit before a trailing refresh, so the refresh opens the next stage instead). `argmax`: a last stage that adds
        the encrypted one-hot of the logits' argmax (`argmax_stage`; the differences read at `argmax_scale`)."""
        self.argmax, self.argmax_scale = argmax, float(argmax_scale)
        stem, blks, (fw, fb) = R.blocks(sd)
        self.act, self.refresh, self.one_stage = act, refresh, one_stage
        B = (lambda site: 1.0) if scales is None else (lambda site: float(scales[site]))

        def sc(Wb, s_in, s_out):
            return Wb[0] * (s_in / s_out), Wb[1] / s_out
        # the scale each producer's activation is carried at: the next block's a2 B when that block adds it unchanged
        prod = ["stem"] + [name + ".a2" for name, *_r in blks]
        carried = {p: B(p) for p in prod}
        for p, (name, _st, _c1, _c2, short) in zip(prod, blks):
            if short is None:
                carried[p] = B(name + ".a2")
        self.gain = {p: B(p) / carried[p] for p in prod}
        self.g_in = Geom(3, 32, 1, slots)
        self.stem = ConvSpec.encode(*sc(stem, 1.0, B("stem")), self.g_in, name="stem")
        s, g = carried["stem"], self.stem.go
        self.blocks = []
        for name, stride, c1, c2, short in blks:
            b1, b2 = B(name + ".a1"), B(name + ".a2")
            s1 = ConvSpec.encode(*sc(c1, s, b1), g, stride, name=name + ".c1")
            s2 = ConvSpec.encode(*sc(c2, b1, b2), s1.go, 1, name=name + ".c2")
            sh = None if short is None else ConvSpec.encode(*sc(short, s, b2), g, stride, name=name + ".sc")
            assert sh is not None or s == b2, (name, s, b2)
            self.blocks.append((name, s1, s2, sh))
            s, g = carried[name + ".a2"], s2.go
        pooled, _ = head_geoms(g.C, len(fb), slots)
        # the pool sums g.H^2 pixels: its mean and the last activation's scale are folded into the head
        self.head = ConvSpec.encode(fw[:, :, None, None] * (s / g.H ** 2), fb, pooled, name="fc")

    def encode(self, img):
        """A normalized (3, 32, 32) image as the input slot vectors."""
        return self.g_in.pack(img)

    def logits(self, cts):
        return self.head.go.unpack(cts[:self.head.go.n_ct])[:, 0, 0]

    def onehot(self, cts):
        """The argmax stage's one-hot (its last ciphertext; class c at block slot 0 of block c)."""
        from perseus.impl.conv import S
        v = np.real(np.asarray(cts[-1]))
        return np.array([v[c * S] for c in range(self.head.go.C)])

    def argmax_stage(self, ops, xs):
        """The logits and the one-hot of their argmax, by two TFHE LUTs (impl/tfhe.py): a sampling-style decision
        inside the model. Logit c sits at slot c S (impl/conv.py head geometry: cpc channel blocks, one pixel each).
        D[c S + k] = (l_c - l_{(c + k) mod cpc}) argmax_scale for the k with (c + k) mod cpc a class (hoisted rotations
        of the logits, each times a mask carrying the scale); s = step(D) (one switch on those 90 slots); the wins of c
        = sum_k s[c S + k] (hoisted rotations, masked to the class slots, / cpc); one-hot = [wins = C - 1] (a second
        switch on the C class slots). The step is a sign LUT (inputs on (-2, 2): |l_c - l_d| < 2 / argmax_scale).
        Ties are undefined; differences within the switch's noise band (~1 % of 1 / argmax_scale) may flip."""
        from perseus.impl import tfhe
        from perseus.impl.conv import S
        g = self.head.go
        C, cpc, n = g.C, g.cpc, g.slots
        L = xs[0]
        pairs = [(c, k) for c in range(C) for k in range(1, cpc) if (c + k) % cpc < C]
        ks = sorted({k for _c, k in pairs})
        mask = {k: np.zeros(n) for k in ks}
        for c, k in pairs:
            mask[k][c * S + k] = self.argmax_scale
        # slot c S + k: rot(L, -k) holds l_c, rot(L, k S - k) holds l_{c + k}
        rots = ops.rotate_many(L, [-k for k in ks] + [k * S - k for k in ks])
        terms = [ops.mult_pt(rots[i], mask[k], kind="mask") for i, k in enumerate(ks)]
        terms += [ops.mult_pt(rots[len(ks) + i], -mask[k], kind="mask") for i, k in enumerate(ks)]
        D = terms[0]
        for t in terms[1:]:
            D = ops.add(D, t)
        win = tfhe.register_slots("argmax.pairs", [c * S + k for c, k in pairs])
        st = tfhe.tfhe_lut(ops, D, tfhe.register_sign("sign01", 0.0, 1.0), win)
        cls = np.zeros(n)
        cls[[c * S for c in range(C)]] = 1.0 / cpc
        rs = ops.rotate_many(st, ks)
        wins = ops.mult_pt(rs[0], cls, kind="mask")
        for r in rs[1:]:
            wins = ops.add(wins, ops.mult_pt(r, cls, kind="mask"))
        top = tfhe.register_lut("argmax.top", lambda m, t=(C - 1.5) / cpc: (np.asarray(m) >= t).astype(np.float64))
        oh = tfhe.tfhe_lut(ops, wins, top, tfhe.register_slots("argmax.classes", [c * S for c in range(C)]))
        return [L, oh]

    def specs(self):
        out = [self.stem]
        for _n, s1, s2, sh in self.blocks:
            out += [s1, s2] + ([sh] if sh is not None else [])
        return out + [self.head]

    def rotations(self):
        from perseus.impl.conv import S
        steps = {s for sp in self.specs() for s in sp.rotations()}
        return sorted(steps | {1 << i for i in range(S.bit_length() - 1)})

    def _in(self, ops, xs):
        # a copy is refreshed: the bootstrap is in place, and the stage input must keep its name (the recorder reads it
        # after the stage)
        return [ops.bootstrap(ops.copy(x)) for x in xs] if self.refresh else xs

    # ── stages ──
    def stem_stage(self, ops, xs):
        return self.act(ops, conv2d(ops, xs, self.stem), "stem", self.gain["stem"], geom=self.stem.go)

    def block_stage(self, ops, xs, b):
        name, s1, s2, sh = self.blocks[b]
        xs = self._in(ops, xs)
        h = self.act(ops, conv2d(ops, xs, s1), name + ".a1", geom=s1.go)
        y = conv2d(ops, h, s2)
        r = conv2d(ops, xs, sh) if sh is not None else xs
        return self.act(ops, [ops.add(a, c) for a, c in zip(y, r)], name + ".a2", self.gain[name + ".a2"], geom=s2.go)

    def head_stage(self, ops, xs):
        return conv2d(ops, channel_sum(ops, self._in(ops, xs)), self.head)

    def stages(self):
        """[(label, fn(ops, cts) -> cts)]; `one_stage`: the whole network as one (the planner places every refresh, no
        stage entry to fix)."""
        out = [("stem", self.stem_stage)]
        out += [(name, lambda ops, xs, b=b: self.block_stage(ops, xs, b)) for b, (name, *_r) in enumerate(self.blocks)]
        out += [("head", self.head_stage)]
        if self.argmax:
            out += [("argmax", self.argmax_stage)]
        if not self.one_stage:
            return out

        def net(ops, xs):
            for _lab, fn in out:
                xs = fn(ops, xs)
            return xs
        return [("net", net)]

    def forward(self, ops, xs):
        for _lab, fn in self.stages():
            xs = fn(ops, xs)
        return xs
