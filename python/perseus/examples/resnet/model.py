"""ResNet-20 from the conv primitive (impl/conv.py): stem, 9 residual blocks, pooled 1x1 head, one stage each. The
activation is a hook `act(ops, cts, site, gain) -> gain * ReLU(cts)` (the exact max on the mirror, the composite-sign
ReLU encrypted, the gain folded into its last series).

With `scales` ({site: B}) every ReLU input arrives divided by its own B (the producing conv carries the 1 / B) and
every activation is carried divided by a known scale (the consuming conv carries it back, the head the last one). An
activation that feeds an identity shortcut is carried at the scale of the add it meets (the next block's a2 B: the
producing ReLU's gain), so the shortcut needs no product. The ReLU then reads its input on [-1, 1] with no level for the
map (impl/relu.relu_unit); ReLU being positively homogeneous, the exact max is unchanged."""
from __future__ import annotations

import numpy as np

from perseus.impl.conv import Geom, ConvSpec, conv2d, channel_sum, head_geoms

from . import weights as R


def relu_exact(ops, xs, site, gain=1.0):
    return [gain * np.maximum(x.real, 0.0) for x in xs]


class ResNet20:
    def __init__(self, sd, act=relu_exact, slots=1 << 15, scales=None, refresh=False):
        """`refresh`: a deliberate bootstrap of every block / head stage's inputs on entry (the planner reads a stage's
        exit before a trailing refresh, so the refresh opens the next stage instead)."""
        stem, blks, (fw, fb) = R.blocks(sd)
        self.act, self.refresh = act, refresh
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
        return self.head.go.unpack(cts)[:, 0, 0]

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
        return self.act(ops, conv2d(ops, xs, self.stem), "stem", self.gain["stem"])

    def block_stage(self, ops, xs, b):
        name, s1, s2, sh = self.blocks[b]
        xs = self._in(ops, xs)
        h = self.act(ops, conv2d(ops, xs, s1), name + ".a1")
        y = conv2d(ops, h, s2)
        r = conv2d(ops, xs, sh) if sh is not None else xs
        return self.act(ops, [ops.add(a, c) for a, c in zip(y, r)], name + ".a2", self.gain[name + ".a2"])

    def head_stage(self, ops, xs):
        return conv2d(ops, channel_sum(ops, self._in(ops, xs)), self.head)

    def stages(self):
        """[(label, fn(ops, cts) -> cts)]."""
        out = [("stem", self.stem_stage)]
        out += [(name, lambda ops, xs, b=b: self.block_stage(ops, xs, b)) for b, (name, *_r) in enumerate(self.blocks)]
        return out + [("head", self.head_stage)]

    def forward(self, ops, xs):
        for _lab, fn in self.stages():
            xs = fn(ops, xs)
        return xs
