"""The complex-lane constructions on the numpy fake: output-packed linears (up and down, the
unpack's 1/2 in the weights) equal the dense linear at the dense linear's level, and the
monomial packs equal a + i b."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perseus.impl import fake, layout, linear  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402
from examples.gpt2_from_primitives import head  # noqa: E402


def _rt():
    fhe, inf = fake.make_fake(complex_payload=True)
    return Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))


@pytest.mark.parametrize("shape", ["up", "down"])
def test_outputpack_equals_dense_at_its_level(shape):
    rt = _rt()
    d = rt.dims
    d_in, d_out = (d.hid, d.E) if shape == "up" else (d.E, d.hid)
    rng = np.random.default_rng(9)
    W = rng.standard_normal((d_in, d_out)) * 0.1
    b = rng.standard_normal(d_out) * 0.1
    xv = layout.encode_linear_input(rng.standard_normal(d_in), d.N, d_in, d_out)
    ref = linear.linear(rt, fake.FakeCt(xv.copy(), 34), linear.EncodedLinear.encode(W, d.N, d_in, d_out, b))
    y = linear.linear(rt, fake.FakeCt(xv.copy(), 34),
                      linear.EncodedLinear.encode(W, d.N, d_in, d_out, b, outputpack=True))
    np.testing.assert_allclose(np.real(y.vec), np.real(ref.vec), rtol=1e-10, atol=1e-10)
    assert y.level == ref.level          # the unpack costs no level


def test_pack_tiles_is_a_plus_ib():
    rt = _rt()
    rng = np.random.default_rng(5)
    a, b = rng.standard_normal(rt.dims.N), rng.standard_normal(rt.dims.N)
    P = head.pack_tiles(rt, [fake.FakeCt(a.copy(), 34), fake.FakeCt(b.copy(), 34)])
    np.testing.assert_allclose(P.vec, a + 1j * b, rtol=1e-12)
    assert P.level == 34                 # the monomial costs no level
