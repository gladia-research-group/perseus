"""impl/conv.py on the plaintext mirror (NumpyOps) against torch's conv2d in float64: every ResNet-20 conv shape, the
zero-slot invariant, and the pooled 1x1 head."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from perseus.impl import conv
from perseus.impl.ops import NumpyOps


def _ref(x, W, b, stride):
    return F.conv2d(torch.from_numpy(x)[None], torch.from_numpy(W), torch.from_numpy(b), stride=stride,
                    padding=W.shape[-1] // 2)[0].numpy()


@pytest.mark.parametrize("ci,co,H,g,k,stride", [
    (3, 16, 32, 1, 3, 1),       # stem
    (16, 16, 32, 1, 3, 1),
    (16, 32, 32, 1, 3, 2),      # stage 1 -> 2
    (16, 32, 32, 1, 1, 2),      # its shortcut
    (32, 32, 16, 2, 3, 1),
    (32, 64, 16, 2, 3, 2),      # stage 2 -> 3: two output ciphertexts
    (32, 64, 16, 2, 1, 2),
    (64, 64, 8, 4, 3, 1),       # two in, two out
])
def test_conv_matches_torch(ci, co, H, g, k, stride):
    rng = np.random.default_rng(ci * 1000 + co + k)
    x = rng.standard_normal((ci, H, H))
    W = rng.standard_normal((co, ci, k, k)) / np.sqrt(ci * k * k)
    b = rng.standard_normal(co)
    gi = conv.Geom(ci, H, g)
    spec = conv.ConvSpec.encode(W, b, gi, stride)
    ys = conv.conv2d(NumpyOps(), gi.pack(x), spec)
    assert len(ys) == spec.go.n_ct
    np.testing.assert_allclose(spec.go.unpack(ys), _ref(x, W, b, stride), atol=1e-9)
    # every slot that is not a valid output pixel is an exact 0, and the replicas agree
    go = spec.go
    for y, w, live in zip(ys, go.pack(_ref(x, W, b, stride)), go.pack(np.ones((go.C, go.H, go.H)))):
        assert np.all(y[live == 0] == 0.0)
        np.testing.assert_allclose(y, w, atol=1e-9)


def test_pool_head():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((64, 8, 8))
    W = rng.standard_normal((10, 64))
    b = rng.standard_normal(10)
    gi = conv.Geom(64, 8, 4)
    pooled, logits = conv.head_geoms(64, 10)
    spec = conv.ConvSpec.encode(W[:, :, None, None] / 64, b, pooled, 1)
    assert spec.go == logits
    ys = conv.conv2d(NumpyOps(), conv.channel_sum(NumpyOps(), gi.pack(x)), spec)
    np.testing.assert_allclose(logits.unpack(ys)[:, 0, 0], W @ x.mean(axis=(1, 2)) + b, atol=1e-9)


def test_rotation_scope():
    gi = conv.Geom(32, 16, 2)
    spec = conv.ConvSpec.encode(np.ones((32, 32, 3, 3)), None, gi, 1)
    used = []

    class Rec(NumpyOps):
        def rotate(self, a, k):
            used.append(int(k) % gi.slots)
            return super().rotate(a, k)

        def rotate_many(self, a, steps):
            used.extend(int(k) % gi.slots for k in steps)
            return super().rotate_many(a, steps)

    conv.conv2d(Rec(), gi.pack(np.zeros((32, 16, 16))), spec)
    assert set(used) <= set(spec.rotations())
