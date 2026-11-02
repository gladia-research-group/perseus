"""PF_ATTN_WROT (one rotation per attention window) against the per-product window ladders on the encoders' and Llama's
attention shapes, on the numpy fake: bidirectional chunks (BERT / ViT), and query sets sharing K / V with a public
prefix key and d_head > t (Llama's GQA prefill). The same outputs, and the wrot path taken (fewer ladders)."""
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


def _attend(monkeypatch, wrot, *, T, dims, bidir=False, keys=0, n_q=1, prefix=False, pack=3):
    """The attention outputs (one array per query set) and the number of rotation-sum ladders run."""
    fhe, inf = fake.make_fake(complex_payload=True, **dims)
    fhe.slots = None
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    rt.ops.sm_cheb_gs = True
    d = rt.dims
    rng = np.random.default_rng(T + 13 * n_q + (7 if bidir else 0))
    ch = prefill.Chunks(T, d.t, bidir=bidir, keys=keys)
    sm = config.load_configs(str(CFG)).block(0)[2]

    def dense(M):
        out = []
        for c in range(ch.n_ct):
            vec = np.zeros(d.N, dtype=np.complex128)
            for half in (0, 1):
                for lane, p in enumerate(ch.lanes(2 * c + half)):
                    vec[np.arange(d.hid) * d.t + lane] += 0.5 * (1j if half else 1.0) * M[p]
            out.append(fake.FakeCt(vec, 30))
        return out
    qs = [dense(0.5 * rng.standard_normal((T, d.hid))) for _ in range(n_q)]
    k, v = (dense(0.5 * rng.standard_normal((T, d.hid))) for _ in range(2))
    pre = (np.repeat(rng.standard_normal(d.hid) * 0.3, d.t), np.repeat(rng.standard_normal(d.hid) * 0.3, d.t)) \
        if prefix else None
    monkeypatch.setattr(prefill, "_PACK", pack)
    monkeypatch.setattr(prefill, "_BSGS", True)
    monkeypatch.setattr(prefill, "_WROT", wrot)
    n = [0]
    ras = rt.ops.rotate_and_sum

    def counted(*a, **kw):
        n[0] += 1
        return ras(*a, **kw)
    monkeypatch.setattr(rt.ops, "rotate_and_sum", counted)
    if n_q > 1 or prefix:
        O = prefill.prefill_attention(rt, qs, k, v, sm, ch, prefix=pre, shared=True)
        return [np.concatenate([o.vec for o in Os]) for Os in O], n[0]
    O = prefill.prefill_attention(rt, qs[0], k, v, sm, ch)
    return [np.concatenate([o.vec for o in O])], n[0]


SHAPES = {
    # BERT / ViT: every query over every key (keys: the first `keys` tokens attended)
    "bidir": dict(T=128, dims=dict(N=8192, hid=128, dim=128, H=2, H_real=2), bidir=True),
    "bidir_keys": dict(T=100, dims=dict(N=8192, hid=128, dim=128, H=2, H_real=2), bidir=True, keys=77),
    # Llama prefill: query sets sharing K / V, a public prefix key, d_head (64) > t (32)
    "gqa_prefix_dhead": dict(T=96, dims=dict(N=4096, hid=128, dim=128, H=2, H_real=2), n_q=3, prefix=True),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("bits", [1, 2, 3, 4, 11, 27])
def test_wrot_matches_ladders_on_every_model_shape(shape, bits, monkeypatch):
    a, na = _attend(monkeypatch, 0, **SHAPES[shape])
    b, nb = _attend(monkeypatch, bits, **SHAPES[shape])
    for x, y in zip(a, b):
        assert np.abs(x).max() > 1e-3
        assert np.abs(x - y).max() < 1e-9
    assert nb < na, (shape, bits, na, nb)            # the wrot path ran (its ladders traded for window rotations)
