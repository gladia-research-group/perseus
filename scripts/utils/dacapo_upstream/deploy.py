"""Deploy the released DaCapo placer's bootstrap targets (sites.py) as a perseus plan.

hecate-opt plans each block standalone; this replays its targets through our planner so the
arm runs here. Nothing about the selection is ours: the DaCapo placer's choose_sites is
replaced by hecate's targets, and only the level bookkeeping, the chain hand-off and the
rescue repair come from perseus, and redundant refreshes are pruned (PLAN_PRUNE=0 keeps them;
the forced exit refresh is never pruned). As hecate assumes fresh block inputs, each block's exit is
refreshed so the next block enters at the landing (as the Orion deploy does). The argmax stage
(the last block, entered from the tail's exit) is planned like make_plan.sh's second stage:
hints dissolved and the refresh envelope hard. Every block plan is stamped with the capture
contract (a dense plan with the dense routing env).

    SITES=<sites.json> GRAPH_DIR=<graph> OUT=<name> [ML=48] [PLAN_DENSE=1] \\
        python scripts/utils/dacapo_upstream/deploy.py
"""
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, ".")

from perseus.plan.placer.planner import PlanConfig, plan_block          # noqa: E402
from perseus.plan.placer.place import PlanInfeasible                    # noqa: E402
from perseus.plan import contract                                       # noqa: E402
import perseus.plan.placer.baselines.dacapo as db                       # noqa: E402

ML = int(os.environ.get("ML", "48"))
DENSE = os.environ.get("PLAN_DENSE", "0") == "1"
GRAPHS = Path(os.environ.get("GRAPH_DIR", "graphs/gpt2_decode_python_n32"))
OUT = Path("bootstrap_placements") / os.environ["OUT"]
SITES = json.load(open(os.environ["SITES"]))
OUT.mkdir(parents=True, exist_ok=True)

# the DaCapo arm's recipe (bootstrap_placements/python/dacapo/PLAN_CMD.txt)
CFG = dict(bootstrap_level=36, max_level=ML, source_level=36, cache_read_level=36,
           level_unit=2, acc_chain="n32", cf_max=20, allow_prescale=False, mag_safety=2.0,
           prune=os.environ.get("PLAN_PRUNE", "1") == "1",
           sparse_precomps=() if DENSE else (512, 1),
           sparse_out_levels=() if DENSE else ((1, 26), (512, 36)), verbose=False)
if ML > 48:
    CFG["baseline_depth_cap"] = 48.0
# a dense plan runs every refresh on the dense route: the code's deliberate bootstraps land at
# the bootstrap level whatever route they took in the capture (make_plan.sh does the same)
if DENSE:
    CFG["deliberate_clamp0"] = True

_orig = db.DaCapoPlacer.choose_sites

el = ed = None
tot = feas = 0
n_blocks = len(SITES)
ARGMAX = n_blocks - 1 if (GRAPHS / f"block_{n_blocks - 1}").is_dir() and n_blocks > 13 else None
for bi in range(n_blocks):
    entry = {} if el is None else dict(entry_level=el, entry_deg=ed)
    sites = set(SITES[f"block_{bi}"]["sites"])
    stage = dict(dissolve_hints=True) if bi == ARGMAX else {}

    def run(extra=None, resc=False, _s=sites, _bi=bi, _entry=entry, _stage=stage):
        chosen = set(_s) | set(extra or ())
        db.DaCapoPlacer.choose_sites = lambda self, s0, _c=chosen: set(_c)
        hard = os.environ.get("PLAN_HARD_ENV_CAP")
        if _stage:
            os.environ["PLAN_HARD_ENV_CAP"] = "1"
        try:
            cfg = PlanConfig(placer="dacapo", baseline_rescue=resc, **CFG, **_stage,
                             prune_keep=tuple(extra or ()))
            return plan_block(GRAPHS / f"block_{_bi}" / "graph.json", cfg, **_entry)
        finally:
            db.DaCapoPlacer.choose_sites = _orig
            if _stage:
                if hard is None:
                    os.environ.pop("PLAN_HARD_ENV_CAP", None)
                else:
                    os.environ["PLAN_HARD_ENV_CAP"] = hard

    try:
        r = run()
    except PlanInfeasible:
        try:
            r = run(resc=True)
        except PlanInfeasible as e:
            print(f"block_{bi}: INFEASIBLE {str(e).splitlines()[0][:70]}")
            break
    ev = r["summary"].get("exit_var")
    if ev and bi < n_blocks - 1:
        for kw in (dict(extra={ev}), dict(extra={ev}, resc=True)):
            try:
                r = run(**kw)
                break
            except PlanInfeasible:
                continue
    s = r["summary"]
    (OUT / f"block_{bi}_placement.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    tot += s["total_bootstraps"]
    feas += 1
    el, ed = s.get("exit_level"), s.get("exit_deg") or 1
    print(f"block_{bi}: total={s['total_bootstraps']} exit={el}/d{ed} "
          f"placements={len(r.get('placements', []))}", flush=True)
cap = json.load(open(GRAPHS / "capture_env.json"))
if DENSE:
    cap["env"].update(SPARSE_AUTO="0", SPARSE_BTS_SLOTS="0")
for bi in range(feas):
    contract.stamp_file(str(OUT / f"block_{bi}_placement.json"), cap)
print(f"\nUPSTREAM DACAPO ARM: {feas}/{n_blocks} feasible, TOTAL={tot} -> {OUT}")
sys.exit(0 if feas == n_blocks else 1)
