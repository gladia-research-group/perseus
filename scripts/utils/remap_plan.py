"""Carry a served prefill plan onto a rewritten op stream (CPU only).

    python scripts/utils/remap_plan.py record OUT [WROT]     the served prefill's per-block graphs (speedrun/server.py's
        setup up to PlannedPrefill.record, PF_ATTN_WROT=WROT), OUT/block_<b>/graph.json, structure only
    python scripts/utils/remap_plan.py remap OLD NEW PLAN_IN PLAN_OUT --blocks 0-10 --unpin STEPS
        PLAN_IN's decisions (keyed by graph names) on OLD's graphs renamed onto NEW's: nodes aligned per block (difflib over
        op type, step, arity and rotation steps), a match kept only when its cipher inputs map onto the new node's; a
        matched consumer whose producers differ on both sides maps the old producer's name onto the new one (the same value
        computed another way). Every name-keyed field is renamed; a decision on a vanished node is an error; the level /
        degree pins of the nodes in STEPS (the rewritten steps) are dropped
    python scripts/utils/remap_plan.py pin PLAN CW --blocks 0-10 --steps STEPS
        CW = a keep-going compile of NEW against PLAN (programs CW/block_<b>.prog.json, graphs CW/block_<b>/graph.json):
        each new node of STEPS the compile reports at exactly the reactive ceiling ([plan_bts_error]) is pinned at its
        compiled level and degree, as the plan pins the old nodes it replaces; anything else is an error
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import sys

VAR = re.compile(r"^(v_\d+|anon_\d+|pf_in\d+)$")


def record(out, wrot):
    os.environ["PF_ATTN_WROT"] = str(wrot)
    sys.path.insert(0, os.getcwd())
    import numpy as np
    from speedrun.server import PREFILL_PLAN, PROMPT
    from perseus.examples.gpt2 import env, weights
    plan = os.environ.get("PLAN", PREFILL_PLAN)
    os.environ.setdefault("KV_ARENA_GB", "0")
    over = {**env.plan_runtime(plan, "n32"), "GPT2_PACKING": "cachemir"}
    for k, v in over.items():                  # as the server: the plan's PF_* switches
        if k.startswith("PF_"):
            os.environ.setdefault(k, str(v))
    os.environ.setdefault("PF_PCMM_CALIB_IN", os.path.join(plan, "pcmm_calib.json"))
    over = {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
            "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY": os.environ.get("PF_PCMM_STAGED_ONLY", "1"), **over}
    env.export_env(0, chain="n32", CKKS_COMPLEX="1", **over)
    from perseus import config as pcfg
    from perseus.impl import config, fakecap
    from perseus.impl.blocked import Blocks, blocked_rot_steps
    from perseus.impl.layout import Dims, ladder_radix_bits, ladder_radix_steps
    from perseus.impl.pcmm import rot_steps as pcmm_rot_steps
    from perseus.impl.prefill import PlannedPrefill, diag_rot_steps, prefill_rot_steps
    from perseus.examples.gpt2.prefill import Gpt2Prefill
    d = Dims.gpt2()
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
    rot = prefill_rot_steps(d) | diag_rot_steps(d) | pcmm_rot_steps() | blocked_rot_steps(
        Blocks(d, int(os.environ.get("PF_NCH", "512"))), ladder_radix_bits()) | fakecap.GPT2_BAND \
        | ladder_radix_steps(d, ladder_radix_bits())
    _fhe, inf = fakecap.make_session(d, rot)
    store = weights.RawStore(os.environ["WEIGHTS_PATH"])
    model = Gpt2Prefill(inf, store, config.load_configs(env.plan_configs(plan, "n32", env.CONFIGS_N32_LEGACY)),
                        core=fakecap.core)
    model.with_tail = True
    cfg = pcfg.plan_runtime(plan, "n32", base=pcfg.preset("n32_gpt2_prefill"))
    run = PlannedPrefill(model, plan, cfg, None, work_dir=out)
    ids = np.random.default_rng(0).integers(0, store.shape(weights.WTE)[0], PROMPT)
    rows = store.tensor(weights.WTE)[ids] + store.tensor(weights.WPE)[:PROMPT]
    if os.environ.get("COMPILE") == "1":       # record + compile against PLAN (keep going: every check error kept)
        from perseus import _pc
        orig = _pc.Program.compile
        _pc.Program.compile = staticmethod(lambda *x, **k: orig(*x, keep_going=True, **k))
        run.prepare(rows)
        for lab, p in run.progs.items():
            p.save(os.path.join(out, f"{lab}.prog.json"))
        return
    run.record(rows)
    for b in range(run.n_stages):
        dd = os.path.join(out, f"block_{b}")
        os.makedirs(dd, exist_ok=True)
        with open(os.path.join(dd, "graph.json"), "w") as f:
            json.dump({"version": 2, "nodes": run.recs[b].nodes}, f)


def sig(n):
    rots = tuple(x for x in n["inputs"] if isinstance(x, str) and x.startswith("rot("))
    return (n["op_type"], n.get("step", ""), len(n["inputs"]), rots)


def align(A, B):
    sm = difflib.SequenceMatcher(None, [sig(n) for n in A], [sig(n) for n in B], autojunk=False)
    pairs = [(i + k, j + k) for i, j, s in sm.get_matching_blocks() for k in range(s)]
    while True:
        m = {A[i]["output"]: B[j]["output"] for i, j in pairs}
        mv = set(m.values())
        inf, bad = {}, set()
        for i, j in pairs:
            for x, y in zip(A[i]["inputs"], B[j]["inputs"]):
                if not (isinstance(x, str) and VAR.match(x)):
                    continue
                if x in m:
                    ok = m[x] == y
                elif x in inf:
                    ok = inf[x] == y
                else:
                    ok = y not in mv and VAR.match(y) is not None
                    if ok:
                        inf[x] = y
                if not ok:
                    bad.add((i, j))
        if not bad:
            m.update(inf)
            return m
        pairs = [p for p in pairs if p not in bad]


def blocks(s):
    lo, hi = map(int, s.split("-"))
    return range(lo, hi + 1)


def dump(p, f):
    with open(f, "w") as fh:
        json.dump(p, fh, indent=2)


def remap(a):
    graph = lambda d, b: json.load(open(f"{d}/block_{b}/graph.json"))["nodes"]
    os.makedirs(a.plan_out, exist_ok=True)
    for f in os.listdir(a.plan_in):
        if not f.endswith("_placement.json") or not any(f == f"block_{b}_placement.json" for b in blocks(a.blocks)):
            shutil.copy(os.path.join(a.plan_in, f), a.plan_out)
    unpin = [x for x in a.unpin.split(",") if x]
    for b in blocks(a.blocks):
        p = json.load(open(f"{a.plan_in}/block_{b}_placement.json"))
        A, B = graph(a.old, b), graph(a.new, b)
        m = align(A, B)
        old_names = {n["output"] for n in A}
        gone = old_names - set(m)
        step = {n["output"]: n.get("step", "") for n in B}
        dropped = {}

        def ren(x, field):
            if x in m:
                return m[x]
            if x in gone:
                dropped.setdefault(field, []).append(x)
                return None
            return x
        q = dict(p)
        q["placements"] = [{**pl, "target_var": ren(pl["target_var"], "placements")} for pl in p.get("placements", [])]
        for f in ("hint_fire", "rescale_after", "real_route", "shipped_order"):
            if f in p:
                q[f] = [y for y in (ren(x, f) for x in p[f]) if y is not None]
        for f in ("sparse_slots", "correction_factor", "raise_drop", "prescale", "offset"):
            if f in p:
                q[f] = {y: v for y, v in ((ren(x, f), v) for x, v in p[f].items()) if y is not None}
        s = dict(p.get("summary", {}))
        for f in ("final_named_levels", "final_named_degs"):
            if f in s:
                s[f] = {y: v for y, v in ((ren(x, f), v) for x, v in s[f].items())
                        if y is not None and not any(u in step.get(y, "") for u in unpin)}
        if s.get("exit_var"):
            s["exit_var"] = ren(s["exit_var"], "exit_var")
        q["summary"] = s
        bad = {f: v for f, v in dropped.items() if f not in ("rescale_after", "final_named_levels", "final_named_degs")}
        if bad or any(pl["target_var"] is None for pl in q["placements"]):
            sys.exit(f"block {b}: decisions on vanished nodes {bad}")
        print(f"block {b}: {len(A)} -> {len(B)} nodes, {len(m)} names carried, {len(gone)} gone; dropped "
              + str({f: len(v) for f, v in dropped.items()}))
        dump(q, f"{a.plan_out}/block_{b}_placement.json")


def pin(a):
    steps = [x for x in a.steps.split(",") if x]
    for b in blocks(a.blocks):
        pr = json.load(open(f"{a.cw}/block_{b}.prog.json"))
        errs = pr["check"]["errors"]
        if not errs:
            continue
        g = {n["output"]: n.get("step", "") for n in json.load(open(f"{a.cw}/block_{b}/graph.json"))["nodes"]}
        f = f"{a.plan}/block_{b}_placement.json"
        p = json.load(open(f))
        for e in errs:
            x = re.match(r"\[plan_bts_error\] (\S+) at level (\d+) reaches the reactive ceiling (\d+) unplanned", e)
            if not x or x.group(2) != x.group(3) or not any(s in g.get(x.group(1), "") for s in steps):
                sys.exit(f"block {b}: {e}")
            v = pr["vars"][pr["names"][x.group(1)]]["meta"]
            p["summary"]["final_named_levels"][x.group(1)] = v["level"]
            p["summary"]["final_named_degs"][x.group(1)] = v["deg"]
        dump(p, f)
        print(f"block {b}: {len(errs)} new nodes pinned at the ceiling")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record"); r.add_argument("out"); r.add_argument("wrot", nargs="?", default="0")
    m = sub.add_parser("remap"); m.add_argument("old"); m.add_argument("new"); m.add_argument("plan_in")
    m.add_argument("plan_out"); m.add_argument("--blocks", default="0-13"); m.add_argument("--unpin", default="")
    p = sub.add_parser("pin"); p.add_argument("plan"); p.add_argument("cw"); p.add_argument("--blocks", default="0-13")
    p.add_argument("--steps", default="")
    a = ap.parse_args()
    if a.cmd == "record":
        os.makedirs(a.out, exist_ok=True)
        record(a.out, a.wrot)
    elif a.cmd == "remap":
        remap(a)
    else:
        pin(a)


if __name__ == "__main__":
    main()
