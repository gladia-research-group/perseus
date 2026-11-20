"""impl/fourier.py encrypted on the GPU: DFT (direct, planned natural order, bit-reversed pair), the real-input FFTs,
DCT-II / DCT-III (real and pair packings), a 2-D DCT on 32 x 32 images, against numpy / scipy in float64 (relative
error on the decoded slots <= 1e-8; measured ~1e-10) and the levels each one spends.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -m gpu python/tests/gpu/test_fourier.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.fft as sf

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

S = 1 << 15
TOL = 1e-8


def _cases():
    from perseus.impl import fourier as F
    rng = np.random.default_rng(3)
    lay64 = F.Layout(64, S, S // 64, "stride")
    lay1k = F.Layout(1024, S, S // 1024, "stride")
    cx = rng.standard_normal((S // 64, 64)) + 1j * rng.standard_normal((S // 64, 64))
    ck = rng.standard_normal((S // 1024, 1024)) + 1j * rng.standard_normal((S // 1024, 1024))
    xr, yr = rng.standard_normal((S // 64, 64)), rng.standard_normal((S // 64, 64))
    xh = rng.standard_normal((S // 32, 64))
    img = rng.standard_normal((S // 1024, 32, 32))
    br = np.argsort(F.bit_reverse(1024))
    h = 32
    return {
        "dft64_direct": (F.dft(64, layout=lay64, direct=True, norm="ortho"), cx, np.fft.fft(cx, norm="ortho"), None),
        "dft1024_natural": (F.dft(1024, layout=lay1k, norm="ortho"), ck, np.fft.fft(ck, norm="ortho"),
                            F.idft(1024, layout=lay1k, norm="ortho")),
        "dft1024_bitrev": (F.dft(1024, layout=lay1k, norm="ortho", order="bitrev"), ck,
                           np.fft.fft(ck, norm="ortho")[:, F.bit_reverse(1024)],
                           F.idft(1024, layout=lay1k, norm="ortho", order="bitrev")),
        "rfft64_pair": (F.rfft_pair(64, layout=lay64, norm="ortho"), F.pack_real_pair(xr, yr),
                        np.concatenate([F.pack_rfft(np.fft.rfft(xr, norm="ortho"), 64),
                                        F.pack_rfft(np.fft.rfft(yr, norm="ortho"), 64)], 1),
                        F.rfft_pair(64, layout=lay64, norm="ortho", inverse=True)),
        "rfft64_half": (F.rfft_half(64, layout=F.Layout(h, S, S // h, "stride"), norm="ortho"), F.pack_real_half(xh),
                        F.pack_rfft(np.fft.rfft(xh, norm="ortho"), 64),
                        F.rfft_half(64, layout=F.Layout(h, S, S // h, "stride"), norm="ortho", inverse=True)),
        "dct64_real": (F.dct2(64, layout=lay64), xr + 0j, sf.dct(xr, 2, norm="ortho"), F.dct3(64, layout=lay64)),
        "dct64_pair": (F.dct2(64, layout=lay64, packing="pair"), F.pack_real_pair(xr, yr),
                       sf.dct(xr, 2, norm="ortho") + 1j * sf.dct(yr, 2, norm="ortho"),
                       F.dct3(64, layout=lay64, packing="pair")),
        "dct2d_32": (F.dct2_2d(32, layout=lay1k, separable=True), img.reshape(-1, 1024) + 0j,
                     sf.dctn(img, 2, axes=(1, 2), norm="ortho").reshape(-1, 1024),
                     F.dct3_2d(32, layout=lay1k, separable=True)),
    }


@pytest.fixture(scope="module")
def env():
    cases = _cases()
    steps = set()
    for T, _, _, inv in cases.values():
        steps |= T.rot_steps() | (inv.rot_steps() if inv else set())
    cfg = _pc.Config.preset("n32_gpt2_decode")
    keys = _pc.KeySource.generate(sorted(steps), seed=13)
    sess = _pc.Session(cfg, keys, pool=128, pt_slots=160).complete_setup()
    from perseus.impl.ops import FheOps
    yield cases, sess, FheOps.from_session(sess)
    sess.close()


def _rel(got, want):
    return float(np.max(np.abs(got - want)) / np.sqrt(np.mean(np.abs(want) ** 2)))


@pytest.mark.parametrize("name", ["dft64_direct", "dft1024_natural", "dft1024_bitrev", "rfft64_pair", "rfft64_half",
                                  "dct64_real", "dct64_pair", "dct2d_32"])
def test_encrypted(env, name):
    cases, sess, ops = env
    T, x, want, inv = cases[name]
    fhe = sess.fhe
    a = fhe.encrypt(T.pack(x), 0)
    y = T.apply(ops, a)
    assert ops.lvl(y) == T.levels * ops.unit
    got = T.unpack(fhe.decrypt_slots_complex(y))
    assert _rel(got, want) < TOL
    if inv is not None:
        z = inv.apply(ops, y)
        assert _rel(inv.unpack(fhe.decrypt_slots_complex(z)), x) < TOL
    from perseus.impl import fourier as F
    F.release(ops)


def test_coefficient_encoding_transform():
    """A client coefficient encryption decrypts to the canonical embedding of its coefficients, and one plaintext
    product is a length-N negacyclic convolution of them (one level, no rotation)."""
    from perseus.impl import fourier as F
    cfg = _pc.Config.preset("n32_gpt2_decode")
    sec = _pc.ClientSecret.generate(seed=5)
    cl = _pc.Client(sec, complex=True, seed=6)
    N = 2 * cl.slots
    sess = _pc.Session(cfg, _pc.KeySource.generate([1], seed=3, secret=sec.to_bytes()), pool=16,
                       pt_slots=4).complete_setup()
    try:
        rng = np.random.default_rng(1)
        x, h = rng.standard_normal(N), rng.standard_normal(N) / np.sqrt(N)
        ct = _pc.import_ct(sess, cl.encrypt_coeffs(list(x), 0))
        z = F.canonical_embedding(x)
        assert _rel(sess.fhe.decrypt_slots_complex(ct), z) < 1e-12
        p = sess.fhe.mult_pt(ct, F.negacyclic_filter(h))
        got = np.asarray(cl.decrypt_coeffs(_pc.export_ct(sess, p))[0])
        assert _rel(got, F.negacyclic_conv(x, h)) < 1e-10
    finally:
        sess.close()


@pytest.mark.parametrize("sets,bts,levels", [
    (["kind=dft", "n=1024"], 0, 5),                            # natural order, the depth the planner's budget prices
    (["kind=dct2", "n=1024"], 0, 9),                           # DCT-II, real packing
    (["kind=dft", "n=1024", "roundtrip=true"], 1, 10),         # forward + inverse: past one refresh window
])
def test_planned_stage(tmp_path, sets, bts, levels):
    """A transform recorded as an ImplModel stage (examples/linalg/run.py --auto: twin capture -> plan -> planned
    run): leaf ops only, the plaintexts by name, the session's rotation keys exactly the program's; a refresh where
    the levels need one (its error, ~1e-4 relative, is then the result's)."""
    import json
    import os
    import subprocess
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    out = str(tmp_path / "rows.jsonl")
    env = dict(os.environ, PYTHONPATH=os.path.join(repo, "python"), PC_WORK_DIR=str(tmp_path))
    subprocess.run([sys.executable, "-m", "perseus.examples.linalg.run", "--task", "fourier", "--set", *sets,
                    "--auto", str(tmp_path / "run"), "--reps", "1", "--out", out], check=True, env=env, cwd=repo)
    rows = [r for r in map(json.loads, open(out)) if r["mode"] == "planned"]
    print(f"\n[fourier] planned {sets}: " + " ".join(f"{r['err_max']:.2e}/{r['wall_s'] * 1e3:.1f}ms" for r in rows))
    assert rows and all(r["bts"] == bts and r["levels"] == levels and r["err_max"] < (1e-3 if bts else 1e-7)
                        for r in rows), rows
