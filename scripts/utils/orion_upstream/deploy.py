"""Deploy the released Orion solver's site selection (marks.py) as a perseus plan.

marks.py marks steps on a graph; this replays those marks through our planner so the
resulting arm runs here. Nothing about the selection is ours: choose_sites is replaced by
upstream's marks, and only the level bookkeeping, the chain hand-off and the rescue repair
come from perseus, and redundant refreshes are pruned (PLAN_PRUNE=0 keeps them; the forced
exit refresh is never pruned). Each block's exit is refreshed so the next block enters at the landing (the
solver plans every block from a fresh input). The argmax stage (the last block, entered from
the tail's exit) is planned like make_plan.sh's second stage: hints dissolved and the refresh
envelope hard. Every block plan is stamped with the capture contract (a dense plan with the
dense routing env).

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
           prune=os.environ.get("PLAN_PRUNE", "1") == "1",
           sparse_precomps=() if DENSE else (512, 1),
           sparse_out_levels=() if DENSE else ((1, 26), (512, 36)), verbose=False)
# a dense plan runs every refresh on the dense route: the code's deliberate bootstraps land at
# the bootstrap level whatever route they took in the capture (make_plan.sh does the same)
if DENSE:
    CFG["deliberate_clamp0"] = True


class _Caught(Exception):
    pass


_orig = ob.OrionPlacer.choose_sites


def upstream_sites(bi, entry, stage):
    """Upstream's marked steps -> the vars whose chains cross them."""
    cap = {}

    def spy(self, s0, _c=cap):
        _c["p"], _c["s"] = self, s0
        raise _Caught

    ob.OrionPlacer.choose_sites = spy
    try:
        plan_block(GRAPHS / f"block_{bi}" / "graph.json",
                   PlanConfig(placer="orion", baseline_rescue=False, **CFG, **stage), **entry)
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


class _Stage:
    """The argmax stage's planner settings (make_plan.sh's second stage)."""
    def __init__(self, on):
        self.kw = dict(dissolve_hints=True) if on else {}

    def __enter__(self):
        self.hard = os.environ.get("PLAN_HARD_ENV_CAP")
        if self.kw:
            os.environ["PLAN_HARD_ENV_CAP"] = "1"
        return self.kw

    def __exit__(self, *a):
        if self.kw:
            if self.hard is None:
                os.environ.pop("PLAN_HARD_ENV_CAP", None)
            else:
                os.environ["PLAN_HARD_ENV_CAP"] = self.hard


N = len(UP)
ARGMAX = N - 1 if N > 13 else None
el = ed = None
tot = feas = 0
for bi in range(N):
    entry = {} if el is None else dict(entry_level=el, entry_deg=ed)
    with _Stage(bi == ARGMAX) as stage:
        w = upstream_sites(bi, entry, stage)

    def run(extra=None, resc=False, _w=w, _bi=bi, _entry=entry):
        sites = set(_w) | set(extra or ())
        ob.OrionPlacer.choose_sites = lambda self, s0, _s=sites: set(_s)
        try:
            with _Stage(_bi == ARGMAX) as stage:
                cfg = PlanConfig(placer="orion", baseline_rescue=resc, **CFG, **stage,
                                 prune_keep=tuple(extra or ()))
                return plan_block(GRAPHS / f"block_{_bi}" / "graph.json", cfg, **_entry)
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
    if ev and bi < N - 1:
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
print(f"\nUPSTREAM ORION ARM: {feas}/{N} feasible, TOTAL={tot} -> {OUT}")
sys.exit(0 if feas == N else 1)
