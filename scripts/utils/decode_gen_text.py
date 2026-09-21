#!/usr/bin/env python3
"""Decode the cutmax (and gt_argmax) token streams of run_generate logs to text.
Usage: .venv/bin/python scripts/decode_gen_text.py logs/probes/<run>.err [...]"""
import os
import re
import sys

# force PAST the shell's IscrC_USAE HF env (PermissionError trap); gpt2 is cached
os.environ["HF_HOME"] = \
    "/leonardo_scratch/large/userexternal/azirilli/.cache/huggingface"
for k in ("HF_TOKEN_PATH", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_TOKEN"):
    os.environ.pop(k, None)
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
for path in sys.argv[1:]:
    steps = re.findall(r"\[generate\] pos=(\d+) cutmax=(\d+).*?gt_argmax=(-?\d+)",
                       open(path).read())
    if not steps:
        print(f"== {path}: no [generate] cutmax lines ==")
        continue
    cm = [int(c) for _, c, _ in steps]
    gt = [int(g) for _, _, g in steps if int(g) >= 0]
    print(f"== {path} (pos {steps[0][0]}..{steps[-1][0]}) ==")
    print(f"cutmax: {tok.decode(cm)!r}")
    if gt:
        print(f"gt    : {tok.decode(gt)!r}")
