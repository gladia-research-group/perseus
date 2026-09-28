"""perseus.impl.pairing on the GPU (complex-payload session): every complex-lane
construction decrypts to the real computation it replaces.

    source scripts/local_env.sh; .venv/bin/python -m pytest -m gpu tests/gpu/test_pairing.py"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.gpt2_from_primitives import env as penv  # noqa: E402

pytestmark = pytest.mark.gpu
DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES", "3")
CHAIN = os.environ.get("CHAIN") or "n32"
penv.export_env(DEVICE, chain=CHAIN, CKKS_COMPLEX="1")

from perseus.impl import pairing  # noqa: E402
from perseus.impl.poly import rotsum  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402


@pytest.fixture(scope="module")
def S():
    from perseus import _core
    sess = penv.open_session(DEVICE, complex_payload=True, chain=CHAIN)
    rt = Rt.from_inf(sess.inf, _core, int(sess.options.ckks.composite_degree))
    ops = rt.ops
    N = sess.inf.slots
    loaded = {int(s) % N for s in ops.fhe.loaded_rot_steps}
    pow2 = [s for s in (1 << k for k in range(1, 15)) if s in loaded]
    fresh = ops.encode_token(np.zeros(768))
    yield dict(sess=sess, ops=ops, N=N, pow2=pow2, zero=ops.sub(fresh, fresh))
    sess.close()


def enc(S, v):
    return S["ops"].add_pt(S["zero"], np.asarray(v))


def dec(S, ct):
    return S["ops"].decrypt_slots_complex(ct)


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - b) / max(np.linalg.norm(b), 1e-12))


def strides(S, n):
    p = S["pow2"]
    return [p[j % len(p)] for j in range(1, n)]


@pytest.mark.parametrize("halved", [False, True])
@pytest.mark.parametrize("B,G", [(4, 4), (2, 8)])
def test_output_pairing(S, B, G, halved):
    ops, rng = S["ops"], np.random.default_rng(B * 10 + G)
    x = enc(S, rng.uniform(-0.5, 0.5, S["N"]))
    W = rng.uniform(-0.1, 0.1, (G, B, S["N"]))
    bs, gs = strides(S, B), strides(S, G)
    ref = dec(S, pairing.diag_linear(ops, x, W, bs, gs)).real
    on = dec(S, pairing.diag_linear_outpack(ops, x, pairing.outpack_weights(W, halved), bs, gs,
                                            halved=halved)).real
    assert rel(on, ref) < 1e-3


@pytest.mark.parametrize("B,G", [(8, 1), (4, 2)])
def test_input_pairing(S, B, G):
    ops, rng = S["ops"], np.random.default_rng(B + G)
    x = enc(S, rng.uniform(-0.5, 0.5, S["N"]))
    W = rng.uniform(-0.1, 0.1, (G, B, S["N"]))
    bs, gs = strides(S, B), strides(S, G)
    ref = dec(S, pairing.diag_linear(ops, x, W, bs, gs)).real
    on = dec(S, pairing.diag_linear_inpack(ops, x, pairing.inpack_weights(W), bs, gs)).real
    assert rel(on, ref) < 1e-3


@pytest.mark.parametrize("halved", [False, True])
def test_fused_pair(S, halved):
    ops, rng = S["ops"], np.random.default_rng(5)
    rs = [enc(S, rng.uniform(-0.5, 0.5, S["N"])) for _ in range(4)]
    Wa, Wb = rng.uniform(-0.2, 0.2, (2, 4, S["N"]))
    A, B = pairing.fused_pair(ops, rs, pairing.pair_weights(Wa, Wb, halved), halved=halved)
    xs = [dec(S, r).real for r in rs]
    assert rel(dec(S, A).real, sum(x * w for x, w in zip(xs, Wa))) < 1e-3
    assert rel(dec(S, B).real, sum(x * w for x, w in zip(xs, Wb))) < 1e-3


def test_contract_pairs(S):
    ops, rng = S["ops"], np.random.default_rng(6)
    a, b = rng.uniform(-0.5, 0.5, (2, 4, S["N"]))
    Wa, Wb = rng.uniform(-0.2, 0.2, (2, 4, S["N"]))
    xs = [pairing.pack_ri(ops, enc(S, ai), enc(S, bi)) for ai, bi in zip(a, b)]
    out = dec(S, pairing.contract_pairs(ops, xs, pairing.conj_pair_weights(Wa, Wb))).real
    assert rel(out, (a * Wa + b * Wb).sum(0)) < 1e-3


def test_paired_products(S):
    ops, rng = S["ops"], np.random.default_rng(7)
    va, vb, sa, sb = rng.uniform(-0.5, 0.5, (4, 4, S["N"]))
    C = [pairing.pair_pack(ops, enc(S, x), enc(S, y)) for x, y in zip(va, vb)]
    out = pairing.paired_products(ops, C, [enc(S, s) for s in sa], [enc(S, s) for s in sb])
    assert rel(dec(S, out).real, (va * sa + vb * sb).sum(0)) < 1e-3


def test_paired_reduction(S):
    ops, rng, N = S["ops"], np.random.default_rng(9), S["N"]
    q, ka, kb = rng.uniform(-0.5, 0.5, (3, N)) / 64
    sa, sb = pairing.paired_reduction(ops, enc(S, q), pairing.pack_ri(ops, enc(S, ka), enc(S, kb)), N)
    ra = dec(S, rotsum(ops, ops.mult(enc(S, q), enc(S, ka)), 1, N)).real
    rb = dec(S, rotsum(ops, ops.mult(enc(S, q), enc(S, kb)), 1, N)).real
    assert rel(dec(S, sa).real, ra) < 1e-3
    assert rel(dec(S, sb).real, rb) < 1e-3
