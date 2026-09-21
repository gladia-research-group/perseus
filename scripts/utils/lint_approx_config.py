#!/usr/bin/env python3
"""Check an approximation config before a capture.

    python scripts/utils/lint_approx_config.py configs/model/approximation/gpt2_base_n32/configs.json [--tier base|heat|squeeze]

Base and squeeze tiers must carry adaptive per-site iteration counts (a uniform count means
the calibration did not run per site); GPT-2 configs must carry the CutMax section byte-equal
to the frozen one in configs/model/approximation/gpt2_base/configs.json (or CUTMAX_DONOR_CONFIG),
because the encrypted argmax plan is bound to it. Exit 0 on OK, 1 with the reasons otherwise.
"""
import collections
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DONOR = os.environ.get("CUTMAX_DONOR_CONFIG") or os.path.join(
    _HERE, "..", "..", "configs", "model", "approximation", "gpt2_base", "configs.json")

path = sys.argv[1]
tier = sys.argv[sys.argv.index("--tier") + 1] if "--tier" in sys.argv else \
    ("heat" if "heat" in path else "squeeze" if "squeeze" in path else "base")
d = json.load(open(path))
fail = []
prov = d.get("_provenance") or {}
certified = bool(prov.get("tool", "").startswith("recount") and prov.get("target"))
if tier in ("base", "squeeze") and not certified:
    gs = collections.Counter(v["gs_iters"] for v in d["norm"].values())
    si = collections.Counter(v["gs_iters_scaled"] for v in d["softmax"].values())
    if len(gs) == 1: fail.append(f"LN gs FIXED({list(gs)[0]}) — baselines must be adaptive")
    if len(si) == 1: fail.append(f"sm init FIXED({list(si)[0]}) — baselines must be adaptive")
if certified:
    for key in ("criterion", "cost_function", "count_convention", "source_config_md5"):
        if not prov.get(key): fail.append(f"_provenance missing '{key}'")
if "gpt2" in path:
    if not os.path.exists(_DONOR):
        print(f"[lint] {path}: CutMax donor {_DONOR} not found (set CUTMAX_DONOR_CONFIG)"); sys.exit(1)
    donor = json.load(open(_DONOR))["cutmax"]
    h = lambda x: hashlib.md5(json.dumps(x, sort_keys=True).encode()).hexdigest()   # noqa: E731
    if "cutmax" not in d: fail.append("cutmax section MISSING")
    elif h(d["cutmax"]) != h(donor): fail.append("cutmax differs from the frozen donor")
if fail:
    print(f"[lint] {path} tier={tier} FAIL:"); [print(f"  - {f}") for f in fail]; sys.exit(1)
print(f"[lint] {path} tier={tier} OK")
