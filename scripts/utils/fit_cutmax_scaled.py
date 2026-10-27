#!/usr/bin/env python
"""CutMax's scaled normalisers fitted on logit rows (perseus.calibrate.approximations.cutmax.fit_scaled): a base
configs.json in, the same configs with its cutmax section's normalisers fitted (outer schedule kept) out.

    python scripts/utils/fit_cutmax_scaled.py BASE_CONFIGS OUT_CONFIGS ROWS.npy [ROWS.npy ...]

gpt2_base_n32_cm is this on gpt2_base_n32 with six sets of calibration rows, 19,454 finite rows: GPT-2's logits (the
CPU twin with gpt2_base_n32's decode approximations) at positions 8-15 of 16-token windows of OpenWebText (384 document
starts, 384 random offsets) and of C4 validation shard 00001 (512 prompts), at positions 0-15 of 256 more document
starts and 256 more shard-00001 prompts, and the exact logits at the last of 128 tokens of 1,024 shard-00001 prompts;
never the dev prompts. gpt2_base_n32_pf_cm is this on gpt2_base_n32_pf with the same rows.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
from perseus.calibrate.approximations.cutmax import fit_scaled  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("rows", nargs="+")
    a = ap.parse_args()
    c = json.load(open(a.base))
    c["cutmax"] = fit_scaled([np.load(p, mmap_mode="r") for p in a.rows], c["cutmax"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(c, open(a.out, "w"), indent=1)
    print(a.out, {k: len(v) if isinstance(v, list) else v for k, v in c["cutmax"].items()})


if __name__ == "__main__":
    main()
