"""Deploy the released Orion solver's site selection (marks.py) as a perseus plan.

marks.py marks steps on a graph; this replays those marks through our planner so the
resulting arm runs here. Nothing about the selection is ours: choose_sites is replaced by
upstream's marks, and only the level bookkeeping, the chain hand-off and the rescue repair
come from perseus. Every block plan is stamped with the capture contract (a dense plan with
the dense routing env).

  RESULTS=<marks.json> GRAPH_DIR=<graph> OUT=<name> [PLAN_DENSE=1] [ML=48] \
      python scripts/utils/orion_upstream/deploy.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PLAN_ACC_CHAIN", "n32")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, ".")

from perseus.plan.placer.planner import PlanConfig, plan_block          # noqa: E402
from perseus.plan.placer.place import PlanInfeasible                    # noqa: E402
from perseus.plan import contract                                       # noqa: E402
import perseus.plan.placer.baselines.orion as ob                        # noqa: E402

ML = int(os.environ.get("ML", "48"))
CF_MAX = int(os.environ.get("CF_MAX", "20"))
DENSE = os.environ.get("PLAN_DENSE", "0") == "1"
GRAPHS = Path(os.environ.get("GRAPH_DIR", "graphs/gpt2_decode_python_n32"))
OUT = Path("bootstrap_placements") / os.environ["OUT"]
UP = json.load(open(os.environ["RESULTS"]))
OUT.mkdir(parents=True, exist_ok=True)

CFG = dict(bootstrap_level=36, max_level=ML, source_level=34, cache_read_level=34,
           level_unit=2, acc_chain="n32", cf_min=2, cf_max=CF_MAX, allow_prescale=False,
           sparse_precomps=() if DENSE else (512, 1),
           sparse_out_levels=() if DENSE else ((1, 26), (512, 36)), verbose=False)


class _Caught(Exception):
    pass


_orig = ob.OrionPlacer.choose_sites


def upstream_sites(bi, entry):
    """Upstream's marked steps -> the vars whose chains cross them."""
    cap = {}

    def spy(self, s0, _c=cap):
        _c["p"], _c["s"] = self, s0
        raise _Caught

    ob.OrionPlacer.choose_sites = spy
    try:
        plan_block(GRAPHS / f"block_{bi}" / "graph.json",
                   PlanConfig(placer="orion", baseline_rescue=False, **CFG), **entry)
    except _Caught:
        pass
    finally:
        ob.OrionPlacer.choose_sites = _orig
    p, s0 = cap["p"], cap["s"]
    marks = set(UP[bi]["upstream"]["marked_steps"]) - {"__ENTRY__", "__SINK__"}
    _sg, _order, _cost, crossing, *_ = p._step_graph(s0)
    w = set()
    for s in marks & set(crossing):
        w |= set(crossing[s])
    return w


el = ed = None
tot = feas = 0
for bi in range(13):
    entry = {} if el is None else dict(entry_level=el, entry_deg=ed)
    w = upstream_sites(bi, entry)

    def run(extra=None, resc=False, _w=w, _bi=bi, _entry=entry):
        sites = set(_w) | set(extra or ())
        ob.OrionPlacer.choose_sites = lambda self, s0, _s=sites: set(_s)
        try:
            return plan_block(GRAPHS / f"block_{_bi}" / "graph.json",
                              PlanConfig(placer="orion", baseline_rescue=resc, **CFG),
                              **_entry)
        finally:
            ob.OrionPlacer.choose_sites = _orig

    r = None
    try:
        r = run()
    except PlanInfeasible:
        try:
            r = run(resc=True)
        except PlanInfeasible as e:
            print(f"block_{bi}: INFEASIBLE {str(e).splitlines()[0][:70]}")
            break
    # force the terminal exit refresh so the next block enters at the landing
    ev = r["summary"].get("exit_var")
    if ev and bi < 12:
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
print(f"\nUPSTREAM ORION ARM: {feas}/13 feasible, TOTAL={tot} -> {OUT}")
sys.exit(0 if feas == 13 else 1)
