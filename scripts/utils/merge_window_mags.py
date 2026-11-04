#!/usr/bin/env python
"""Window magnitudes for the 512-context decode plans (`*_decode_ctx512_w<B>`): a copy of a recording (the structure
capture, block_<b>/graph.json) whose nodes take, by ciphertext name and op type, the largest output_max_* over
  * the window's captures (--win), for a node of a cache-length step (softmax, its .hrs / .gp, qkt: values that follow
    the cache length, ~1/kc to ~kc; another window's magnitudes would waste or overflow this window's refreshes);
  * every capture given (--win and --all), for any other node (values that follow the token: LayerNorm / linear /
    activation inputs, the more tokens sampled the safer).
A node no capture has (a scope carrying the cache shape) keeps the recording's statistics; the code's own refreshes
(deliberate_bootstrap), the cut's (auto_bootstrap) and the hints keep them too (the planner prices a code refresh
from the value it refreshes under PLAN_DELIB_PRICE_INPUT=1).
--lo: output_max_abs_lo / output_max_coeff_lo = the smallest over the same captures (the planner prices a refresh's
precision there). --periods CAP...: pack_period_max = the largest slot period those captures recorded, where it
exceeds the recording's (the softmax's period grows with the cache: PLAN_STEP_ROUTE_PRICING prices a refresh on the
routes it reaches).

The shipped windows (bootstrap_placements/README.md, "Long-context decode plans"), e.g. Llama's w512 from the
position-1 recording and captures at 384, 448 and 511:
    python scripts/utils/merge_window_mags.py graphs/<rec>_p1 graphs/llama31_8b_decode_ctx512_w512 \
        --win graphs/<rec>_p384 graphs/<rec>_p448 graphs/<rec>_p511 \
        --all graphs/<rec>_p384 graphs/<rec>_p448 graphs/<rec>_p511 --lo \
        --periods graphs/<rec>_p384 graphs/<rec>_p448 graphs/<rec>_p511
"""
import json
import os
import re
import shutil
import sys

CACHE_STEP = re.compile(r"\.(softmax|qkt)(\.|$)")
SKIP = ("auto_bootstrap", "hint", "level_hint", "deliberate_bootstrap")


def parse(argv):
    a = {"win": [], "all": [], "periods": [], "lo": False}
    if len(argv) < 2:
        sys.exit(__doc__)
    base, out, cur = argv[0], argv[1], None
    for x in argv[2:]:
        if x in ("--win", "--all", "--periods"):
            cur = x[2:]
        elif x == "--lo":
            a["lo"] = True
        elif cur is None:
            sys.exit(f"{x}: name --win, --all or --periods first")
        else:
            a[cur].append(x)
    return base, out, a


def load(cap):
    """{block number: {name: node}} of one capture (graph dir)."""
    out = {}
    for d in os.listdir(cap):
        if d.startswith("block_") and d[6:].isdigit():
            with open(os.path.join(cap, d, "graph.json")) as f:
                out[int(d[6:])] = {n["output"]: n for n in json.load(f)["nodes"] if n.get("output")}
    return out


def periods(caps):
    """{block number: {name: largest pack_period}} over the captures."""
    out = {}
    for c in caps:
        for b, nodes in load(c).items():
            o = out.setdefault(b, {})
            for nm, n in nodes.items():
                o[nm] = max(o.get(nm, 0), int(n.get("pack_period") or 0))
    return out


def main():
    base, out, a = parse(sys.argv[1:])
    if os.path.exists(out):
        sys.exit(f"{out} exists")
    win = [load(p) for p in a["win"]]
    allc = win + [load(p) for p in a["all"]]
    per = periods(a["periods"]) if a["periods"] else None
    shutil.copytree(base, out)
    for d in sorted(os.listdir(out)):
        if not (d.startswith("block_") and d[6:].isdigit()):
            continue
        b = int(d[6:])
        path = os.path.join(out, d, "graph.json")
        with open(path) as f:
            g = json.load(f)
        nw = na = 0
        for n in g["nodes"]:
            nm = n.get("output")
            if not nm or n.get("op_type") in SKIP:
                continue
            cache = bool(CACHE_STEP.search(n.get("step") or ""))
            ts = [c[b][nm] for c in (win if cache else allc) if b in c and nm in c[b]
                  and c[b][nm].get("op_type", n.get("op_type")) == n.get("op_type")]
            if not ts:
                continue
            nw += cache
            na += not cache
            keys = {k for t in ts for k, v in t.items() if k.startswith("output_max_") and not k.endswith("_lo")
                    and isinstance(v, (int, float))}
            for k in keys:
                n[k] = max(t[k] for t in ts if isinstance(t.get(k), (int, float)))
            if a["lo"]:
                for k in ("output_max_abs", "output_max_coeff"):
                    vals = [t[k] for t in ts if isinstance(t.get(k), (int, float))]
                    if vals:
                        n[k + "_lo"] = min(vals)
        if per is not None:
            pp = per.get(b, {})
            for n in g["nodes"]:
                v = pp.get(n.get("output"))
                if v and v > int(n.get("pack_period") or 0):
                    n["pack_period_max"] = int(v)
        with open(path, "w") as f:
            json.dump(g, f)
        print(f"{d}: {len(g['nodes'])} nodes; {nw} cache-step nodes from the window, {na} others from every capture",
              flush=True)


if __name__ == "__main__":
    main()
