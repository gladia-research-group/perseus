"""Audio input: a WAV file as float64 mono samples at a given rate (scipy only; FLAC / MP3 need decoders the runtime
does not carry: convert them to WAV first)."""
from __future__ import annotations

from math import gcd

import numpy as np


def read_wav(path, sr: int = 16000) -> np.ndarray:
    """Samples in [-1, 1) (integer PCM scaled by its full range), channels averaged, resampled to `sr`."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    rate, a = wavfile.read(path)
    a = np.asarray(a)
    if a.dtype.kind in "iu":
        a = a.astype(np.float64) / float(2 ** (8 * a.dtype.itemsize - 1))
    a = a.astype(np.float64)
    if a.ndim == 2:
        a = a.mean(axis=1)
    if rate != sr:
        g = gcd(int(rate), int(sr))
        a = resample_poly(a, int(sr) // g, int(rate) // g)
    return a
