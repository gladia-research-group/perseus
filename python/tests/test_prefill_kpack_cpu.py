"""PF_SM_KPACK (prefill._kp*): with d_head past the chunk pair's key offsets (Gemma-2's 256 offset blocks for 128
tokens) chunk 1's scores ride the empty half of every BSGS window of chunk 0's, one softmax for the pair. On the numpy
fake, for a pair with a public prefix key, query sets sharing K / V (GQA) and a partial second chunk: the attention
outputs equal the unpacked ones, and one softmax runs per query set instead of two. A pair whose offsets fill more than
half a window is left unpacked."""
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
# t = N / hid = 16 lanes, d_head 128 = 8 t: a chunk pair's offsets (< 2 t, the prefix's included) fit half a window
DIMS = dict(N=4096, hid=256, dim=256, H=2, H_real=2)


def _attend(monkeypatch, kpack, *, T, n_q=1, prefix=True, pack=3, wrot=27, dims=DIMS):
    fhe, inf = fake.make_fake(complex_payload=True, **dims)
    fhe.slots = None
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    rt.ops.sm_cheb_gs = True
    d = rt.dims
    rng = np.random.default_rng(T + 7 * n_q)
    ch = prefill.Chunks(T, d.t, base=1)
    sm = config.load_configs(str(CFG)).block(0)[2]

    def dense(M):
        out = []
        for c in range(ch.n_ct):
            vec = np.zeros(d.N, dtype=np.complex128)
            for half in (0, 1):
                for lane, p in enumerate(ch.lanes(2 * c + half)):
                    vec[np.arange(d.hid) * d.t + lane] += 0.5 * (1j if half else 1.0) * M[p - 1]
            out.append(fake.FakeCt(vec, 30))
        return out
    qs = [dense(0.5 * rng.standard_normal((T, d.hid))) for _ in range(n_q)]
    k, v = (dense(0.5 * rng.standard_normal((T, d.hid))) for _ in range(2))
    pre = (np.repeat(rng.standard_normal(d.hid) * 0.3, d.t), np.repeat(rng.standard_normal(d.hid) * 0.3, d.t)) \
        if prefix else None
    monkeypatch.setattr(prefill, "_PACK", pack)
    monkeypatch.setattr(prefill, "_BSGS", True)
    monkeypatch.setattr(prefill, "_WROT", wrot)
    monkeypatch.setattr(prefill, "_SM_KPACK", kpack)
    n = [0]
    psm = prefill.prefill_softmax

    def counted(*a, **kw):
        n[0] += 1
        return psm(*a, **kw)
    monkeypatch.setattr(prefill, "prefill_softmax", counted)
    O = prefill.prefill_attention(rt, qs, k, v, sm, ch, prefix=pre, shared=True)
    return [np.concatenate([o.vec for o in Os]) for Os in O], n[0]


@pytest.mark.parametrize("T, n_q, prefix", [(31, 1, True), (32, 2, True), (27, 2, False), (31, 2, True)])
@pytest.mark.parametrize("wrot", [0, 27])
def test_kpack_matches_unpacked(T, n_q, prefix, wrot, monkeypatch):
    a, na = _attend(monkeypatch, False, T=T, n_q=n_q, prefix=prefix, wrot=wrot)
    b, nb = _attend(monkeypatch, True, T=T, n_q=n_q, prefix=prefix, wrot=wrot)
    for x, y in zip(a, b):
        assert np.abs(x).max() > 1e-3
        assert np.abs(x - y).max() < 1e-9 * np.abs(x).max()
    assert (na, nb) == (2 * n_q, n_q)


def test_kpack_needs_half_a_window(monkeypatch):
    """t = 32 lanes with d_head 64: chunk 1's offsets (< 64) fill whole windows: unpacked."""
    dims = dict(N=4096, hid=128, dim=128, H=2, H_real=2)
    a, na = _attend(monkeypatch, False, T=63, dims=dims)
    b, nb = _attend(monkeypatch, True, T=63, dims=dims)
    assert na == nb == 2
    for x, y in zip(a, b):
        assert np.abs(x - y).max() == 0.0
