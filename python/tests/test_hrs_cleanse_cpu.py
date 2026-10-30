"""SM_HRS_CLEANSE: the packed softmax head sum (SM_HRS_PACK) carries seed * sum on the imaginary part of one complex
mask; imaginary noise on its input (a refresh planned between the caller's im_cleanse and the head sum) lands in that
lane at full scale. The cleansed pack takes 2 Re of its input first: exact on a real input, immune to that noise."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perseus.impl import attention, fake, layout  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402


def _rt():
    fhe, inf = fake.make_fake(complex_payload=True)
    return Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))


def _head_sum(monkeypatch, cleanse: str, x):
    monkeypatch.setenv("FUSED_SM_DEN", "0")
    monkeypatch.setenv("SM_HRS_PACK", "1")
    monkeypatch.setenv("SM_HRS_CLEANSE", cleanse)
    rt = _rt()
    s, h = attention.head_reduce_sum(rt, fake.FakeCt(np.array(x), 30), s0_expected=2.0, periodic=1, seed=3e-6)
    return np.real(s.vec).copy(), np.real(h.vec).copy()


def test_cleansed_pack_matches_on_real_input(monkeypatch):
    x = np.random.default_rng(3).standard_normal(_rt().dims.N) + 0j
    s0, h0 = _head_sum(monkeypatch, "0", x)
    s1, h1 = _head_sum(monkeypatch, "1", x)
    np.testing.assert_allclose(s1, s0, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(h1, h0, rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("noise", [1e-4, 1e-3])
def test_cleansed_pack_ignores_imaginary_noise(monkeypatch, noise):
    rng = np.random.default_rng(4)
    re = rng.standard_normal(_rt().dims.N)
    _, h_exact = _head_sum(monkeypatch, "0", re + 0j)
    noisy = re + 1j * noise * rng.standard_normal(re.shape[0])
    _, h_plain = _head_sum(monkeypatch, "0", noisy)
    _, h_clean = _head_sum(monkeypatch, "1", noisy)
    scale = np.abs(h_exact).max()
    assert np.abs(h_plain - h_exact).max() > 10 * scale        # the seed lane is the noise
    np.testing.assert_allclose(h_clean, h_exact, rtol=1e-9, atol=1e-12 * scale)
