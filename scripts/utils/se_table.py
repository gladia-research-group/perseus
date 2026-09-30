#!/usr/bin/env python3
"""Turn a test_sparse_envelope log into the dense-vs-sparse comparison tables.

    python3 scripts/utils/se_table.py <run.log> [--metric rel_bias|bits|abs_max|rms]

The grid test emits one `[se]` line per (structure, amplitude, rep, arm). Reps are
averaged.  The point of the table is the PAIRED comparison: the two arms saw the same
ciphertext, so `sparse - dense` in the gap column is the artifact the decode's
Goldschmidt margins were calibrated against, not a cross-run difference.
"""
import argparse
import math
import re
import sys
from collections import defaultdict

LINE = re.compile(
    r"\[se\] struct=(?P<struct>\S+) A=(?P<A>\S+) rep=(?P<rep>\d+) arm=(?P<arm>\S+) "
    r"bias=(?P<bias>\S+) rel_bias=(?P<rel_bias>\S+) bias_rel_ref=(?P<bias_rel_ref>\S+) "
    r"abs_max=(?P<abs_max>\S+) rms=(?P<rms>\S+) bits=(?P<bits>\S+) nonfinite=(?P<nf>\d+)"
)
THREW = re.compile(r"\[se\] struct=(?P<struct>\S+) A=(?P<A>\S+) rep=\d+ arm=(?P<arm>\S+) THREW")


def fmt(v, metric):
    if v is None:
        return "THREW"
    if not math.isfinite(v):
        return "inf"
    if metric in ("rel_bias", "bias_rel_ref"):
        return f"{100*v:+.2f}%"
    if metric == "bits":
        return f"{v:.1f}"
    return f"{v:.2e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--metric", default="rel_bias",
                    choices=["rel_bias", "bias_rel_ref", "bits", "abs_max", "rms", "bias"])
    args = ap.parse_args()

    vals = defaultdict(list)          # (struct, A, arm) -> [metric]
    threw = set()
    amps, structs = [], []
    with open(args.log) as fh:
        for ln in fh:
            m = LINE.search(ln)
            if m:
                s, A, arm = m["struct"], float(m["A"]), m["arm"]
                vals[(s, A, arm)].append(float(m[args.metric]))
                if A not in amps:
                    amps.append(A)
                if s not in structs:
                    structs.append(s)
                continue
            t = THREW.search(ln)
            if t:
                s, A, arm = t["struct"], float(t["A"]), t["arm"]
                threw.add((s, A, arm))
                if A not in amps:
                    amps.append(A)
                if s not in structs:
                    structs.append(s)

    if not structs:
        sys.exit(f"no [se] lines in {args.log}")
    amps.sort()

    def get(s, A, arm):
        v = vals.get((s, A, arm))
        if v:
            return sum(v) / len(v)
        return None

    hdr = f"metric = {args.metric}   (D = dense, S = sparse, reps averaged)"
    print(hdr)
    print("=" * len(hdr))
    for s in structs:
        print(f"\n## {s}")
        print("| A | dense | sparse | gap (S-D) | S/D |")
        print("|---|---|---|---|---|")
        for A in amps:
            d, sp = get(s, A, "dense"), get(s, A, "sparse")
            if d is None and sp is None and (s, A, "dense") not in threw \
                    and (s, A, "sparse") not in threw:
                continue
            gap = f"{100*(sp-d):+.2f}%" if (d is not None and sp is not None
                                            and args.metric.startswith(("rel", "bias_rel"))) \
                else (f"{sp-d:+.2e}" if (d is not None and sp is not None) else "—")
            ratio = f"{sp/d:.2f}x" if (d not in (None, 0.0) and sp is not None
                                       and math.isfinite(d) and d != 0) else "—"
            print(f"| {A:g} | {fmt(d, args.metric)} | {fmt(sp, args.metric)} | {gap} | {ratio} |")


if __name__ == "__main__":
    main()
