"""PF_ATTN_WROT bit 16 (q.K^T's window rotations on the operands, rot_w(q) rot_w(K~_w), one sum of products per giant
step) against the products rotated after (bit 1 alone), on the numpy fake: the same attention outputs, and fewer
products rotated."""
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


def _attend(T, bits, monkeypatch):
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
    monkeypatch.setattr(prefill, "_PACK", 3)
    monkeypatch.setattr(prefill, "_BSGS", True)
    monkeypatch.setattr(prefill, "_WROT", bits)
    n = {"mult": 0, "rot": 0}
    mult, rotate, rotate_many = rt.ops.mult, rt.ops.rotate, rt.ops.rotate_many

    def count(k, f, w=lambda *a: 1):
        def g(*a, **kw):
            n[k] += w(*a)
            return f(*a, **kw)
        return g
    monkeypatch.setattr(rt.ops, "mult", count("mult", mult))
    monkeypatch.setattr(rt.ops, "rotate", count("rot", rotate))
    monkeypatch.setattr(rt.ops, "rotate_many", count("rot", rotate_many, lambda x, steps, *a: len(steps)))
    O = prefill.prefill_attention(rt, dense(q), dense(k), dense(v), sm, ch)
    return np.concatenate([o.vec for o in O]), n


@pytest.mark.parametrize("T,base", [(128, 11), (128, 1), (128, 3), (64, 11)])
def test_qk_operand_rotation_matches(T, base, monkeypatch):
    a, na = _attend(T, base, monkeypatch)
    b, nb = _attend(T, base | 16, monkeypatch)
    assert np.abs(a).max() > 1e-3
    assert np.abs(a - b).max() < 1e-9 * np.abs(a).max()
    assert nb["mult"] == na["mult"]
    print(T, base, na, nb)
