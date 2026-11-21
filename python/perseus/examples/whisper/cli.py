"""perseus-asr: encrypted speech-to-text from the command line (Whisper base.en).

The client computes each clip's log-mel and encrypts its rows; the server runs Whisper's encoder and greedy decoding
under CKKS on the GPU: the encoder's last stage fills the decoder's cross-attention caches, handed over inside the
server, and every token's argmax (CutMax) and its embedding fed back stay encrypted (examples/whisper/run_asr.py). The
client decrypts the tokens. Each transcript is printed with whether it matches the plaintext model's.

    perseus-asr clip.wav [more.wav ...]           # encrypted (the default plans, the first visible GPU)
    perseus-asr clip.wav --float                  # the plaintext model only

The 10-s window (weights.WINDOW_S) transcribes clips up to 8 s (it needs 2 s of silence after the speech; Whisper's
30-s window does not fit a 96 GB GPU encrypted). An utterance generates up to 62 tokens (one decode plan structure).
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None):
    from . import weights as W
    ap = argparse.ArgumentParser(prog="perseus-asr", description=__doc__.split("\n\n")[0])
    ap.add_argument("clips", nargs="+", help=f"16 kHz WAV clips (mono, up to {W.MAX_CLIP_S:g} s)")
    ap.add_argument("--float", dest="float_only", action="store_true", help="the plaintext model only (CPU)")
    ap.add_argument("--device", type=int, default=None, help="GPU index (default: the first visible)")
    ap.add_argument("--force", action="store_true", help="run on a GPU other processes use")
    a = ap.parse_args(argv)
    from .run_encoder import clip_seconds
    long = [f"{c} ({clip_seconds(c):.1f} s)" for c in a.clips if clip_seconds(c) > W.MAX_CLIP_S + 1e-6]
    if long:
        ap.error(f"clips longer than {W.MAX_CLIP_S:g} s (the {W.WINDOW_S}-s window's limit): {', '.join(long)}")
    if a.float_only:
        for c, t in zip(a.clips, float_transcripts(a.clips)):
            print(f"{os.path.basename(c)}: {t}", flush=True)
        return 0
    from . import run_asr as R
    argv = ["--audio", a.clips[0], "--enc-plan", R.ENC_PLAN] + (["--device", str(a.device)] if a.device is not None else [])
    argv += ["--force"] if a.force else []
    return R.main(argv, clips=a.clips)


def float_transcripts(clips):
    """The plaintext model's greedy transcripts in the same window (transformers, CPU)."""
    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration
    from . import hf, score as S, weights as W
    m = hf.fit_window(WhisperForConditionalGeneration.from_pretrained(W.snapshot()).eval())
    out = []
    for c in clips:
        x = torch.tensor(np.asarray(W.log_mel(W.load_audio(c)), dtype=np.float32)[None])
        with torch.no_grad():
            out.append(S.text(m.generate(x, max_new_tokens=128)[0].tolist()))
    return out


if __name__ == "__main__":
    sys.exit(main())
