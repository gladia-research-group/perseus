"""Whisper's encrypted ASR pieces on the CPU: the encoder window's client rows, the cross-attention cache the encoder's
last stage fills (impl.cross_attention.fill_cache on plain slot vectors against weights.cross_cache_ref), the
recorder's one-slot ciphertexts, the clip selection and the WER.

  .venv/bin/python -m pytest python/tests/test_whisper_cpu.py
"""
import contextlib
import os
import sys
import types
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perseus.examples.llama.prefix import PlainOps  # noqa: E402
from perseus.examples.whisper import weights as W  # noqa: E402
from perseus.examples.whisper.run_encoder import clip_seconds, clips, dims  # noqa: E402
from perseus.impl.cross_attention import cache_cts, fill_cache  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402


class _Ops(PlainOps):
    """PlainOps with the step scopes rt.step opens (no-ops)."""
    inf = types.SimpleNamespace(step=lambda label: contextlib.nullcontext())


def _wav(path, seconds, sr=W.SR):
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.zeros(int(seconds * sr), dtype=np.int16).tobytes())


def test_window_rows():
    """T encoder positions per window, the client's rows (T, 5 x 80): row t holds mel frames 2t-2 .. 2t+2."""
    assert W.N_FRAMES == 100 * W.WINDOW_S and W.T == W.N_FRAMES // 2
    assert W.MAX_CLIP_S <= W.WINDOW_S
    mel = np.random.default_rng(0).standard_normal((W.N_MELS, W.N_FRAMES))
    rows = W.mel_rows(mel)
    assert rows.shape == (W.T, W.WIN * W.N_MELS)
    t = W.T // 2
    assert np.array_equal(rows[t].reshape(W.WIN, W.N_MELS), mel[:, 2 * t - 2:2 * t + 3].T)
    assert not rows[0, :2 * W.N_MELS].any()           # frames -2, -1: the window's zero padding


@pytest.mark.parametrize("n", [W.T, 300])
def test_fill_cache_matches_reference(n):
    """The K buckets and the V diagonal buckets fill_cache builds from the encoder's K / V pairs are the float cache's
    on every real key's lane."""
    d = dims()
    rng = np.random.default_rng(n)
    K, V = rng.standard_normal((2, n, d.H * d.d_head))
    kv = fill_cache(Rt(_Ops(), d), W.decode_pairs(K, d), W.decode_pairs(V, d), n)
    got = cache_cts(kv)
    ref = W.cross_cache_ref(K, V, n, d)
    assert len(got) == len(ref)
    lanes = 0
    for x, (re, im) in zip(got, ref):
        x = np.asarray(x)
        for part, r in ((x.real, re), (x.imag, im)):       # an odd group count leaves the last pair's imaginary empty
            m = ~np.isnan(r)
            assert np.allclose(part[m], r[m], atol=1e-9)
            lanes += int(m.sum())
    assert lanes >= 2 * n * d.H * d.d_head                  # every key's K and V lanes compared (V's twice: diagonals)


def test_recorder_keeps_one_slot():
    """CapCt.novec keeps a copy of slot 0, not a view that would hold the whole slot vector."""
    from perseus.impl import fakecap
    old = fakecap.CapCt.novec
    fakecap.CapCt.novec = True
    try:
        ct = fakecap.CapCt(np.ones(1 << 12, dtype=np.complex128), 3, 1)
    finally:
        fakecap.CapCt.novec = old
    assert ct.vec.shape == (1,) and ct.vec.base is None


def test_shift_mask_names_stable():
    """A weight vector's mask tag is its values' digest: equal values under another object name the same plaintext
    (a run's compiled programs are found again), other values another."""
    a = np.random.default_rng(1).standard_normal(512)
    assert Rt.values_key(a) == Rt.values_key(a.copy())
    assert Rt.values_key(a) != Rt.values_key(a + 1e-12)


def test_clip_selection():
    """Clips past the window's limit left out; n clips spread over the sorted directory."""
    with tempfile.TemporaryDirectory() as tmp:
        for i, s in enumerate([1.0, 2.0, 9.5, 3.0, 4.0, 5.0]):
            _wav(Path(tmp) / f"c{i}.wav", s)
        assert clip_seconds(os.path.join(tmp, "c2.wav")) == pytest.approx(9.5)
        kept = [os.path.basename(c) for c in clips(tmp, max_s=8.0, log=lambda m: None)]
        assert kept == ["c0.wav", "c1.wav", "c3.wav", "c4.wav", "c5.wav"]
        spread = [os.path.basename(c) for c in clips(tmp, 3, spread=True, max_s=8.0, log=lambda m: None)]
        assert spread == ["c0.wav", "c3.wav", "c5.wav"]


def test_word_errors():
    from perseus.examples.whisper import score as S
    assert S.word_errors("a b c d", "a x c") == (2, 4)
    assert S.word_errors("a b", "a b") == (0, 2)
