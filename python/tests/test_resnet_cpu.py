"""ResNet-20 (examples/resnet) on the plaintext mirror against the float64 torch reference."""
import numpy as np
import pytest

from perseus.examples.resnet import weights as R
from perseus.examples.resnet.model import ResNet20
from perseus.impl.ops import NumpyOps


@pytest.fixture(scope="module")
def sd():
    try:
        return R.state_dict()
    except Exception as e:      # offline, no cached checkpoint
        pytest.skip(f"ResNet-20 checkpoint: {e}")


def test_mirror_matches_reference(sd):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3, 32, 32))
    m = ResNet20(sd)
    ref = R.reference(sd, x)
    for i in range(len(x)):
        np.testing.assert_allclose(m.logits(m.forward(NumpyOps(), m.encode(x[i]))), ref[i], atol=1e-8)


def test_scales_fold(sd):
    """Activations carried divided by per-site scales: the exact max is unchanged (positively homogeneous)."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3, 32, 32))
    sites = ["stem"] + [f"layer{s}.{i}.a{j}" for s in (1, 2, 3) for i in range(3) for j in (1, 2)]
    m = ResNet20(sd, scales={n: rng.uniform(1, 8) for n in sites})
    ref = R.reference(sd, x)
    for i in range(len(x)):
        np.testing.assert_allclose(m.logits(m.forward(NumpyOps(), m.encode(x[i]))), ref[i], atol=1e-8)


def test_rotation_scope(sd):
    m = ResNet20(sd)
    assert len(m.rotations()) < 200
