"""PF_ATTN_WROT (the packed reductions' window masks on the K / V side) against the per-product window ladders, on the
numpy fake: the same attention outputs."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perseus._paths import PKG  # noqa: E402

from perseus.impl import config, fake, layout, prefill  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402

CFG = PKG / "configs/model/approximation/gpt2_base_n32/configs.json"


def _attend(T, pack, wrot, monkeypatch):
    fhe, inf = fake.make_fake(complex_payload=True, N=8192, hid=128, dim=128, H=2, H_real=2)
    fhe.slots = None
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    d = rt.dims
    rng = np.random.default_rng(T)
    ch = prefill.Chunks(T, d.t)
    sm = config.load_configs(str(CFG)).block(0)[2]
    q, k, v = (0.5 * rng.standard_normal((T, d.hid)) for _ in range(3))

    def dense(M):
        out = []
        for c in range(ch.n_ct):
            vec = np.zeros(d.N, dtype=np.complex128)
            for half in (0, 1):
                for lane, p in enumerate(ch.lanes(2 * c + half)):
                    vec[np.arange(d.hid) * d.t + lane] += 0.5 * (1j if half else 1.0) * M[p]
            out.append(fake.FakeCt(vec, 30))
        return out
    monkeypatch.setattr(prefill, "_PACK", pack)
    monkeypatch.setattr(prefill, "_BSGS", True)
    monkeypatch.setattr(prefill, "_WROT", wrot)
    O = prefill.prefill_attention(rt, dense(q), dense(k), dense(v), sm, ch)
    return np.concatenate([o.vec for o in O])


@pytest.mark.parametrize("T,pack,bits", [(128, 3, 1), (128, 3, 2), (128, 3, 4), (128, 3, 3), (128, 3, 11), (128, 2, 3), (64, 3, 3),
                                         (100, 3, 3), (100, 3, 5)])
def test_wrot_matches_ladders(T, pack, bits, monkeypatch):
    a = _attend(T, pack, 0, monkeypatch)
    b = _attend(T, pack, bits, monkeypatch)
    assert np.abs(a).max() > 1e-3
    assert np.abs(a - b).max() < 1e-9
