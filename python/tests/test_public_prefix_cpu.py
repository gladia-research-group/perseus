"""prefill.PublicPrefix: P public keys / values before the prompt (positions 0 .. P - 1), their scores one PCMM on Q
onto P free key offsets of the score ciphertext and their values one PCMM on P (SANA's instruction prefix). On the
numpy fake, the attention of T prompt queries with the public prefix equals the same queries' rows of a P + T prompt
whose first P keys / values are the public ones (every lane attends as many keys, so the approximate softmax runs the
same seeds), for one chunk and two chunks, BSGS on or off."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perseus._paths import PKG  # noqa: E402

from perseus.impl import config, fake, layout, prefill  # noqa: E402
from perseus.impl import pcmm  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402

CFG = PKG / "configs/model/approximation/gpt2_base_n32/configs.json"
DIMS = dict(N=32768, hid=2048, dim=2048, H=8, H_real=8)     # t = 16 lanes, d_head 256 (SANA's Gemma group)


def _rt():
    fhe, inf = fake.make_fake(complex_payload=True, **DIMS)
    fhe.slots = None
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    rt.ops.sm_cheb_gs = True
    return rt


def _dense(d, ch, rows, base):
    out = []
    for c in range(ch.n_ct):
        vec = np.zeros(d.N, dtype=np.complex128)
        for half in (0, 1):
            for lane, p in enumerate(ch.lanes(2 * c + half)):
                vec[np.arange(d.hid) * d.t + lane] += 0.5 * (1j if half else 1.0) * rows[p - base]
        out.append(fake.FakeCt(vec, 30))
    return out


def _rows(d, ch, O):
    out = {}
    for c, o in enumerate(O):
        v = np.asarray(o.vec).reshape(d.hid, d.t)
        for half in (0, 1):
            for lane, p in enumerate(ch.lanes(2 * c + half)):
                out[p] = (v[:, lane].imag if half else v[:, lane].real)
    return out


@pytest.mark.parametrize("T, P, bsgs", [(12, 20, True), (12, 20, False), (16, 40, True), (27, 30, True)])
def test_public_prefix_equals_prompt_keys(T, P, bsgs, monkeypatch):
    monkeypatch.setattr(prefill, "_PACK", 3 if bsgs else 0)
    monkeypatch.setattr(prefill, "_BSGS", bsgs)
    monkeypatch.setattr(prefill, "_SM_KPACK", False)
    rng = np.random.default_rng(T + P)
    sm = config.load_configs(str(CFG)).block(0)[2]
    q, k, v = (0.5 * rng.standard_normal((P + T, DIMS["hid"])) for _ in range(3))

    # the reference: a P + T prompt (positions 0 ..), the last T queries
    rt = _rt()
    d = rt.dims
    chb = prefill.Chunks(P + T, d.t, base=0)
    Ob = prefill.prefill_attention(rt, _dense(d, chb, q, 0), _dense(d, chb, k, 0), _dense(d, chb, v, 0), sm, chb)
    want = _rows(d, chb, Ob)

    # the public prefix: positions 0 .. P - 1 public, the prompt at P ..
    rt = _rt()
    d = rt.dims
    ch = prefill.Chunks(T, d.t, base=P)
    gscale = 1.0 / np.sqrt(d.d_head_real)
    heads = lambda M: np.stack([M[:, h::d.H] for h in range(d.H)])           # (H, P, d_head): channel j H + h
    lins = {}

    def lin(c, i, which):
        if (c, which) not in lins:
            Ws, Wv = prefill.public_prefix_mats(d, ch, c, heads(k[:P]), heads(v[:P]), gscale)
            lins[(c, "s")] = pcmm.PcmmLinear.encode(Ws, d.t, name=f"pub.s{c}")
            lins[(c, "v")] = pcmm.PcmmLinear.encode(Wv, d.t, name=f"pub.v{c}")
        return lins[(c, which)]
    pre = prefill.PublicPrefix(P, lambda c, i: lin(c, i, "s"), lambda c, i: lin(c, i, "v"), pcmm.run)
    Oa = prefill.prefill_attention(rt, _dense(d, ch, q[P:], P), _dense(d, ch, k[P:], P), _dense(d, ch, v[P:], P), sm,
                                   ch, prefix=pre)
    got = _rows(d, ch, Oa)
    scale = max(np.abs(want[p]).max() for p in got)
    for p in got:
        np.testing.assert_allclose(got[p], want[p], atol=2e-4 * scale)
