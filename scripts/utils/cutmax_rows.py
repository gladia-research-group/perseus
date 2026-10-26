#!/usr/bin/env python
"""GPT-2's logit rows for CutMax's stage-13 magnitudes (capture_mags --argmax --rows): the float64 reference model
(speedrun/harness/gpt2_ref.py) at the last token of each window of a windows file (token ids per window).

    python scripts/utils/cutmax_rows.py graphs/gpt2_decode_python_n32_cm/cutmax_rows.json ROWS.npy \
        [--weights model.safetensors]

The weights default to GPT2_SAFETENSORS, else openai-community/gpt2 in the HF cache. Writes float32 rows
[windows, vocab].
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "speedrun", "harness"))
from gpt2_ref import GPT2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windows")
    ap.add_argument("out")
    ap.add_argument("--weights", default=os.environ.get("GPT2_SAFETENSORS"))
    a = ap.parse_args()
    if not a.weights:
        hub = os.path.join(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")), "hub")
        a.weights = sorted(glob.glob(os.path.join(hub, "models--openai-community--gpt2", "snapshots", "*",
                                                  "model.safetensors")))[0]
    ref = GPT2(a.weights)
    wins = json.load(open(a.windows))["windows"]
    rows = np.stack([ref.logits(ref.embed(w["ids"]))[-1] for w in wins]).astype(np.float32)
    np.save(a.out, rows)
    print(f"[cutmax_rows] {len(wins)} windows -> {a.out} {rows.shape}")


if __name__ == "__main__":
    main()
