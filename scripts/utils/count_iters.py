"""Iteration count for an approximation config — the EXECUTION convention.

    iterations = sum(norm.gs_iters) + sum(norm.nr_iters)
               + sum(softmax.gs_iters_scaled)
               + sum(sum(softmax.per_step_refine_iters))

Sites are ENUMERATED FROM THE CONFIG (25 norm keys: h.{0..11}.ln_{1,2} + ln_f;
12 softmax keys: h.{0..11}.attn) — never hard-coded, so a config with a different
depth still counts correctly.

THE TRAP this script exists to close: `per_step_refine_iters` is a LIST, one entry per
squaring-renormalisation pass, and must be DOUBLE-summed. Its lookalike scalar
`gs_iters_refine_scaled` is a legacy max-per-site aggregate — it equals
max(per_step_refine_iters) at every site (this script asserts that) and therefore
collapses a multi-pass site to its deepest single pass. Using it undercounts.

Worked example, vit_base's four multi-pass sites:
    h.0  [8,9,10,10,10] sum 47 max 10 -> undercounts 37
    h.1  [9,9]          sum 18 max  9 -> undercounts  9
    h.2  [8,9]          sum 17 max  9 -> undercounts  8
    h.11 [8,9]          sum 17 max  9 -> undercounts  8
    total undercount 62 = 617 - 555, exactly and with nothing left over.

The bug is invisible on any arm whose softmax sites are all single-pass — every GPT-2
arm — because there sum == max. It only bites on multi-pass refinement, i.e. the ViT
baseline. Compare COLUMNS, not just the total: that is how a mismatch gets localised.

Usage:  count_iters.py <config.json> [<config.json> ...]
        count_iters.py --arms vit_base vit_squeeze vit_heat
"""
import json
import sys
from pathlib import Path

CFG = Path(__file__).resolve().parents[2] / "configs" / "model" / "approximation"


def count(path):
    c = json.load(open(path))
    norm, sm = c.get("norm", {}), c.get("softmax", {})
    a = sum(v["gs_iters"] for v in norm.values())              # LN Goldschmidt
    b = sum(v["nr_iters"] for v in norm.values())              # LN Newton
    cc = sum(v["gs_iters_scaled"] for v in sm.values())        # softmax init GS
    per = {k: list(v["per_step_refine_iters"]) for k, v in sm.items()}
    d = sum(sum(p) for p in per.values())                      # refine, EXECUTED
    e = sum(v["gs_iters_refine_scaled"] for v in sm.values())  # refine, LEGACY max
    passes = sum(len(p) for p in per.values())
    # the legacy scalar is exactly the per-site max — verify, don't assume
    bad = [k for k, v in sm.items() if v["gs_iters_refine_scaled"] != max(per[k])]
    return dict(norm_sites=len(norm), sm_sites=len(sm), passes=passes,
                A=a, B=b, C=cc, D=d, E=e, per_step=a + b + cc + d, legacy=a + b + cc + e,
                multi={k: p for k, p in per.items() if len(p) > 1}, bad=bad)


def main(paths):
    hdr = f"{'arm':<14}{'passes':>7}{'A ln_gs':>9}{'B ln_nr':>9}{'C sm_gs':>9}" \
          f"{'D refine':>10}{'E legacy':>10}{'PER-STEP':>10}{'legacy':>9}"
    print(hdr)
    print("-" * len(hdr))
    for p in paths:
        p = Path(p)
        s = count(p)
        name = p.parent.name
        print(f"{name:<14}{s['passes']:>7}{s['A']:>9}{s['B']:>9}{s['C']:>9}"
              f"{s['D']:>10}{s['E']:>10}{s['per_step']:>10}{s['legacy']:>9}")
        if s["bad"]:
            print(f"  !! gs_iters_refine_scaled != max(per_step_refine_iters) at {s['bad']}")
        if s["passes"] == s["sm_sites"]:
            # One entry per site => the two conventions agree. That is EITHER a genuinely
            # single-pass arm (gpt2_squeeze 455, gpt2_heat 228 — verified) OR a config whose
            # lists were collapsed. The two are indistinguishable from the config alone, so
            # this is a flag to check provenance, not a verdict.
            print(f"  .. passes == {s['sm_sites']} softmax sites: every site is single-pass, so "
                  f"both conventions agree ({s['per_step']}). Genuine for gpt2_squeeze/heat; "
                  f"on any other arm check the config was not collapsed.")
        if s["multi"]:
            for k, v in sorted(s["multi"].items()):
                print(f"     multi-pass {k}: {v}  sum={sum(v)} max={max(v)} "
                      f"undercount={sum(v) - max(v)}")
            print(f"     total undercount = {s['per_step'] - s['legacy']}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--arms":
        args = [str(CFG / a / "configs.json") for a in args[1:]]
    sys.exit(main(args or [str(CFG / a / "configs.json")
                           for a in ("vit_base", "vit_squeeze", "vit_heat")]))
