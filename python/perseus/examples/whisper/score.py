"""Transcripts as text and their word error rate: Whisper's tokenizer and English normalizer (lower case, no
punctuation, spelled-out numbers and British spellings mapped), the edit distance over words."""
from __future__ import annotations

import functools
import os

import numpy as np


@functools.lru_cache(maxsize=1)
def tokenizer():
    from transformers import WhisperTokenizer
    from . import weights as W
    return WhisperTokenizer.from_pretrained(W.snapshot())


def text(tokens):
    """The transcript of emitted token ids (special tokens dropped)."""
    return tokenizer().decode(list(tokens), skip_special_tokens=True).strip()


def normalize(s):
    return tokenizer().normalize(s)


def word_errors(ref, hyp):
    """(edit distance in words, reference words) of two normalized transcripts."""
    r, h = ref.split(), hyp.split()
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for k in range(1, len(h) + 1):
            d[k] = min(prev[k] + 1, d[k - 1] + 1, prev[k - 1] + (r[i - 1] != h[k - 1]))
    return int(d[len(h)]), len(r)


def wer(pairs):
    """Corpus WER (%) of [(reference, hypothesis)] raw transcripts, both normalized."""
    e = n = 0
    for ref, hyp in pairs:
        a, b = word_errors(normalize(ref), normalize(hyp))
        e, n = e + a, n + b
    return 100.0 * e / max(n, 1)


def references(path):
    """{clip id: transcript} of a manifest (TSV: id, seconds, text; a header line allowed)."""
    out = {}
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 3 and f[1].replace(".", "", 1).isdigit():
            out[f[0]] = f[2]
    return out


def clip_id(path):
    return os.path.splitext(os.path.basename(path))[0]
