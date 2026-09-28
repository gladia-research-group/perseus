"""perseus.impl.pairing on the numpy fake: every complex-lane construction equals the real
computation it replaces."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perseus.impl import fake, pairing  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.impl.poly import rotsum  # noqa: E402

N = 1024


@pytest.fixture
def ops():
    steps = {1 << k for k in range(11)}
    steps |= {(-s) % N for s in steps} | {s - N for s in steps}
    fhe, inf = fake.make_fake(N=N, complex_payload=True, rot_steps=steps)
    return FheOps(inf, fake.core, fhe.unit)


def ct(v, level=0):
    return fake.FakeCt(np.asarray(v, dtype=np.float64), level)


def close(a, b, tol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    assert np.max(np.abs(a - b)) <= tol * max(1.0, np.max(np.abs(b))), np.max(np.abs(a - b))


def steps(n):
    return [1 << (1 + (j % 9)) for j in range(1, n)]


def test_pack_unpack(ops):
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(N), rng.standard_normal(N)
    for P in (pairing.pack_ri(ops, ct(a), ct(b)), pairing.pair_pack(ops, ct(a), ct(b))):
        re, im = pairing.unpack_ri(ops, P)
        close(re.vec, a); close(im.vec, b)
        close(pairing.real_part(ops, P).vec, a)


@pytest.mark.parametrize("halved", [False, True])
@pytest.mark.parametrize("B,G", [(8, 16), (4, 32), (16, 8), (4, 4)])
def test_output_pairing(ops, B, G, halved):
    rng = np.random.default_rng(B * 100 + G)
    x = ct(rng.standard_normal(N))
    W = rng.standard_normal((G, B, N))
    bs, gs = steps(B), steps(G)
    ref = pairing.diag_linear(ops, x, W, bs, gs)
    on = pairing.diag_linear_outpack(ops, x, pairing.outpack_weights(W, halved), bs, gs, halved=halved)
    close(on.vec, ref.vec)


@pytest.mark.parametrize("B,G", [(32, 1), (8, 4)])
def test_input_pairing(ops, B, G):
    rng = np.random.default_rng(B + G)
    x = ct(rng.standard_normal(N))
    W = rng.standard_normal((G, B, N))
    bs, gs = steps(B), steps(G)
    ref = pairing.diag_linear(ops, x, W, bs, gs)
    on = pairing.diag_linear_inpack(ops, x, pairing.inpack_weights(W), bs, gs)
    close(on.vec, ref.vec)


@pytest.mark.parametrize("halved", [False, True])
def test_fused_pair(ops, halved):
    rng = np.random.default_rng(5)
    rs = [ct(rng.standard_normal(N)) for _ in range(16)]
    Wa, Wb = rng.standard_normal((16, N)), rng.standard_normal((16, N))
    A, B = pairing.fused_pair(ops, rs, pairing.pair_weights(Wa, Wb, halved), halved=halved)
    close(A.vec, sum(r.vec * w for r, w in zip(rs, Wa)))
    close(B.vec, sum(r.vec * w for r, w in zip(rs, Wb)))


def test_contract_pairs(ops):
    rng = np.random.default_rng(6)
    a, b = rng.standard_normal((32, N)), rng.standard_normal((32, N))
    Wa, Wb = rng.standard_normal((32, N)), rng.standard_normal((32, N))
    xs = [pairing.pack_ri(ops, ct(ai), ct(bi)) for ai, bi in zip(a, b)]
    close(pairing.contract_pairs(ops, xs, pairing.conj_pair_weights(Wa, Wb)).vec, (a * Wa + b * Wb).sum(0))


def test_paired_products(ops):
    rng = np.random.default_rng(7)
    va, vb, sa, sb = (rng.standard_normal((32, N)) for _ in range(4))
    C = [pairing.pair_pack(ops, ct(x), ct(y)) for x, y in zip(va, vb)]
    out = pairing.paired_products(ops, C, [ct(s) for s in sa], [ct(s) for s in sb])
    close(out.vec, (va * sa + vb * sb).sum(0))


def test_pair_masks(ops):
    rng = np.random.default_rng(8)
    v = rng.standard_normal(N)
    sel = lambda l: (np.arange(N) % 64 == l).astype(float)
    y = ops.mult_pt(ct(v), pairing.pair_masks(sel(2), sel(3)))
    close(y.vec.real, v * sel(2)); close(y.vec.imag, v * sel(3))


def test_paired_reduction(ops):
    rng = np.random.default_rng(9)
    q, ka, kb = (rng.standard_normal(N) for _ in range(3))
    sa, sb = pairing.paired_reduction(ops, ct(q), pairing.pack_ri(ops, ct(ka), ct(kb)), N)
    close(sa.vec, rotsum(ops, ct(q * ka), 1, N).vec)
    close(sb.vec, rotsum(ops, ct(q * kb), 1, N).vec)
