"""perseus.impl.handoff on plain slot vectors: the cache re-laid from prefill-layout K / V equals the one the decode
builds by pushing the same positions."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.llama_from_primitives.prefix import PlainOps, expanded_heads  # noqa: E402
from perseus.impl.attention import ComplexKVCache, Rope, cache_kv_push_packed_complex  # noqa: E402
from perseus.impl.handoff import to_decode_cache  # noqa: E402
from perseus.impl.layout import Dims, encode_linear_input  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402

CFG = ROOT / "configs/model/approximation/llama31_8b_n32/configs.json"
H, HKV, DH = 32, 8, 128


def _rope_float(k, p, inv):
    th = p * np.asarray(inv)
    c, s = np.cos(th), np.sin(th)
    a, b = k[..., :DH // 2], k[..., DH // 2:]
    return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)


@pytest.mark.parametrize("T", [1, 7, 8, 63, 64, 65, 120, 127])
@pytest.mark.parametrize("kv_lanes", [False, True])
def test_handoff_matches_pushes(T, kv_lanes):
    d = Dims(32768, 4096, 4096, 32, 32, 16384, 14336)
    N = d.N
    inv = json.load(open(CFG))["model"]["rope_inv_freq"]
    rng = np.random.default_rng(T)
    Kf, Vf = rng.standard_normal((T + 1, HKV, DH)), rng.standard_normal((T + 1, HKV, DH))
    ops = PlainOps()
    ops.kv_lanes = kv_lanes
    rt = Rt(ops, d)
    lin = lambda f: encode_linear_input(f, N, d.hid, d.hid)
    ref = ComplexKVCache(DH)
    rope = Rope(inv)
    for p in range(T + 1):
        k, v = (expanded_heads(F[p].ravel(), H, HKV, DH) for F in (Kf, Vf))
        cache_kv_push_packed_complex(rt, ref, lin(k) + 1j * lin(v), rope=rope, pos=p)
    Kr = np.stack([_rope_float(Kf[p], p, inv) for p in range(T + 1)])
    t_src, nc = 64, (T + 63) // 64

    def src(F, b, c):
        x = np.zeros(N)
        for lane in range(t_src):
            p = 1 + t_src * c + lane
            if p > T:
                break
            for m in range(4):
                x[np.arange(DH) * 256 + t_src * m + lane] = F[p, 4 * b + m]
        return x

    def bos(F):
        x = np.zeros(N)
        x[np.arange(d.hid) * d.t] = expanded_heads(F[0].ravel(), H, HKV, DH)
        return x
    K = [[src(Kr, b, c) for c in range(nc)] for b in range(2)]
    V = [[src(Vf, b, c) for c in range(nc)] for b in range(2)]
    got = to_decode_cache(Rt(PlainOps(), d), K, V, T, bos(Kr), bos(Vf))
    assert len(got.k_buckets) == len(ref.k_buckets) and len(got.k_pend) == len(ref.k_pend)
    for a, b in zip(got.k_buckets + got.k_pend, ref.k_buckets + ref.k_pend):
        assert np.allclose(a, b, atol=1e-9)
    for a, b in zip(got.v_buckets, ref.v_buckets):
        assert np.allclose(a, 0 if b is None else b, atol=1e-9)
    assert got.k_count == ref.k_count and got.v_count == ref.v_count


def test_handoff_cleanse_drops_imaginary_junk():
    d = Dims(32768, 4096, 4096, 32, 32, 16384, 14336)
    N, T = d.N, 70
    rng = np.random.default_rng(1)
    K = [[rng.standard_normal(N) for _ in range(2)] for _ in range(2)]
    V = [[rng.standard_normal(N) for _ in range(2)] for _ in range(2)]
    junk = lambda S: [[x + 1j * rng.standard_normal(N) for x in row] for row in S]
    bos = np.zeros(N)
    ref = to_decode_cache(Rt(PlainOps(), d), K, V, T, bos, bos)
    got = to_decode_cache(Rt(PlainOps(), d), junk(K), junk(V), T, bos, bos, cleanse=True)
    for a, b in zip(got.k_buckets + got.k_pend + got.v_buckets, ref.k_buckets + ref.k_pend + ref.v_buckets):
        assert np.allclose(a, b, atol=1e-9)
