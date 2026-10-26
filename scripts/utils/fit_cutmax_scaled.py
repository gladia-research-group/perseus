#!/usr/bin/env python
"""CutMax's scaled normalisers fitted on logit rows (perseus.calibrate.approximations.cutmax.fit_scaled): a base
configs.json in, the same configs with its cutmax section's normalisers fitted (outer schedule kept) out.

    python scripts/utils/fit_cutmax_scaled.py BASE_CONFIGS OUT_CONFIGS ROWS.npy [ROWS.npy ...]

gpt2_base_n32_cm is this on gpt2_base_n32, gpt2_base_n32_pf_cm on gpt2_base_n32_pf, never on the evaluation prompts.
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
