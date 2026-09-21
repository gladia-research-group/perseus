"""Pre-capture config lint — run BEFORE any STAGE=capture (costed us a full
recapture round on 2026-07-28 when gpt2_base/gpt2_squeeze went to capture with
FIXED counts, violating the baselines-always-adaptive ruling).

  python scripts/utils/lint_approx_config.py <configs.json> [--tier base|squeeze|heat]

Checks: (1) base/squeeze tiers must carry ADAPTIVE per-site counts (spread, not
uniform — heat is exempt: learned counts may be uniform); (2) gpt2 configs must
carry the frozen CutMax section byte-equal to the donor.
"""
import json, os, sys, collections, hashlib

path = sys.argv[1]
tier = sys.argv[sys.argv.index("--tier") + 1] if "--tier" in sys.argv else \
    ("heat" if "heat" in path else "squeeze" if "squeeze" in path else "base")
d = json.load(open(path))
fail = []
# A config emitted by scripts/recount_1e4.py carries a machine-readable record of the criterion
# it was fitted under. Where that record exists it REPLACES the spread heuristic below, which
# only ever inferred "somebody forgot to fit this" from uniformity — a uniform count is the
# correct answer when the level-minimising fit lands there (e.g. vit_squeeze gs=8 at 25/25).
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
    # Repo-relative by default; LINT_CUTMAX_DONOR overrides. The old absolute /leonardo_work
    # path made every GPT-2 capture die at the gate on any other machine.
    _here = os.path.dirname(os.path.abspath(__file__))
    _default = os.path.normpath(os.path.join(
        _here, "..", "..", "..", "..", "configs", "model", "approximation",
        "gpt2_baseline", "configs.json"))
    donor_path = os.environ.get("LINT_CUTMAX_DONOR", _default)
    if not os.path.exists(donor_path):
        print(f"[lint] {path} tier={tier} FAIL:\n  - cutmax donor not found at {donor_path} "
              f"(set LINT_CUTMAX_DONOR)"); sys.exit(1)
    donor = json.load(open(donor_path))["cutmax"]
    h = lambda x: hashlib.md5(json.dumps(x, sort_keys=True).encode()).hexdigest()
    if "cutmax" not in d: fail.append("cutmax section MISSING (silent 21s fallback)")
    elif h(d["cutmax"]) != h(donor): fail.append("cutmax differs from frozen donor")
if fail:
    print(f"[lint] {path} tier={tier} FAIL:"); [print(f"  - {f}") for f in fail]; sys.exit(1)
print(f"[lint] {path} tier={tier} OK")
