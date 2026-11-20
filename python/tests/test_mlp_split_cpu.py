"""LLAMA_MLP_SPLIT (examples/llama/weights.mlp_parts): the MLP's real width as power-of-two slices instead of one width
padded to E, each slice its own gate / up / down. The slices of a padded MLP (Gemma-2-2B's 9216 of 16384 = 8192 +
1024), none where they would not save width (Llama-3-8B's 14336 in two slices pads back to 16384); and on the numpy fake
a hidden -> slice -> hidden MLP whose narrow slice reads the hidden through interleave_rows and writes it through
interleave_cols equals the dense product exactly, its activation masks (mlp_mask) on the slices' real slots."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perseus.examples.llama import weights as W  # noqa: E402
from perseus.impl import fake, layout  # noqa: E402
from perseus.impl.layout import decode_linear_output, linear_rot_steps  # noqa: E402
from perseus.impl.linear import EncodedLinear, linear  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402


@pytest.mark.parametrize("n, E, E_real, want", [
    ("2", 16384, 9216, [(0, 8192, 8192), (8192, 1024, 1024)]),     # Gemma-2-2B
    ("2", 16384, 14336, []),                                        # Llama-3-8B: two slices pad back to E
    ("3", 16384, 14336, [(0, 8192, 8192), (8192, 4096, 4096), (12288, 2048, 2048)]),
    ("2", 4096, 3072, [(0, 2048, 2048), (2048, 1024, 1024)]),       # GPT-2
    ("0", 16384, 9216, []),
    ("2", 16384, 9000, [(0, 8192, 8192), (8192, 1024, 808)]),       # the last slice padded
])
def test_mlp_parts(monkeypatch, n, E, E_real, want):
    monkeypatch.setenv("LLAMA_MLP_SPLIT", n)
    d = layout.Dims(32768, 4096, 2304, 16, 8, E, E_real)
    assert W.mlp_parts(d) == want


def test_split_mlp_matches_dense():
    """hid 512 (t = 8), E_real 1152 = 1024 + 128: Gemma-2-2B's shapes at N / 8."""
    N, hid, parts = 4096, 512, [(0, 1024), (1024, 128)]
    steps = set()
    for _o, w in parts:
        steps |= linear_rot_steps(N, hid, w) | linear_rot_steps(N, w, hid)
    fhe, inf = fake.make_fake(complex_payload=True, N=N, hid=hid, dim=320, H=8, H_real=4, E=2048, E_real=1152,
                              rot_steps=steps)
    rt = Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf))
    ops, d = rt.ops, rt.dims
    rng = np.random.default_rng(3)
    x = rng.standard_normal(d.dim)
    G = rng.standard_normal((hid, 1152)) * (np.arange(hid) < d.dim)[:, None]
    D = rng.standard_normal((1152, hid))
    want = (x @ G[:d.dim]) @ D
    xc = ops.encode_token(x)
    out = None
    for off, w in parts:
        narrow = w < hid
        g = W.interleave_rows(G[:, off:off + w], w) if narrow else G[:, off:off + w]
        dn = W.interleave_cols(D[off:off + w], w) if narrow else D[off:off + w]
        y = linear(rt, xc, EncodedLinear.encode(g, N, hid, w))
        y = ops.mult_pt(y, W.mlp_mask(d, w, w, 1.0))     # the activation's lane mask, at scale 1
        z = linear(rt, y, EncodedLinear.encode(dn, N, w, hid))
        out = z if out is None else ops.add(out, z)
    got = decode_linear_output(ops.decrypt_slots(out), N, hid, hid)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-10 * np.abs(want).max())
