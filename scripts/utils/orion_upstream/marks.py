"""Run the released Orion bootstrap solver (baahl-nyu/orion) on a captured graph.

The solver files orion/core/{network_dag,level_dag,auto_bootstrap}.py are loaded from a
clone at ORION_SRC (commit be8a827 with orion_be8a827.patch applied: the patch changes no
decision, it makes the solver finish and not crash on transformer-shaped graphs). Everything
else upstream imports is stubbed.

Adapter: the Orion placer's step graph (perseus.plan.placer.baselines.orion) becomes their
NetworkDAG. Each step is a network node with a dummy module: .depth = step cost (composite
levels), slots = 1 and fhe_input_shape.numel() = the step's ciphertext count, .level = None.
Their own find_residuals discovers the regions. A virtual __ENTRY__ (level pinned to
l_eff - entry_level through their user-specified-level feature) and __SINK__ normalize the
multi-source / multi-sink blocks, which their code assumes away.

    ORION_SRC=<clone> GRAPH_DIR=<graph> OUT_JSON=<marks.json> [PLAN_DENSE=1] \\
        python scripts/utils/orion_upstream/marks.py

Writes one record per block with the steps upstream marked for a bootstrap;
deploy.py turns them into a plan.
"""
import importlib.util
import json
import os
import signal
import sys
import types
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PLAN_HARD_ENV_CAP", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, ".")

import networkx as nx                                                    # noqa: E402

CLONE = Path(os.environ["ORION_SRC"])
GRAPHS = Path(os.environ.get("GRAPH_DIR", "graphs/gpt2_decode_python_n32"))
OUT_JSON = Path(os.environ["OUT_JSON"])
DENSE = os.environ.get("PLAN_DENSE", "0") == "1"


# ── 1. stub what upstream imports, then load their solver files ───────────────────────
def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


class _Placeholder:  # isinstance-only (BatchNormNd, LinearTransform) or unused (Bootstrap)
    def __init__(self, *a, **k):
        pass


_stub("matplotlib")
_stub("matplotlib.pyplot")
_stub("orion").__path__ = [str(CLONE / "orion")]
_stub("orion.core").__path__ = [str(CLONE / "orion" / "core")]
_stub("orion.nn").__path__ = [str(CLONE / "orion" / "nn")]
_stub("orion.nn.normalization", BatchNormNd=type("BatchNormNd", (_Placeholder,), {}))
_stub("orion.nn.linear", LinearTransform=type("LinearTransform", (_Placeholder,), {}))
_stub("orion.nn.operations", Bootstrap=type("Bootstrap", (_Placeholder,), {}))


def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(modname, CLONE / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


NetworkDAG = _load("orion.core.network_dag", "orion/core/network_dag.py").NetworkDAG
_load("orion.core.level_dag", "orion/core/level_dag.py")
BootstrapSolver = _load("orion.core.auto_bootstrap", "orion/core/auto_bootstrap.py").BootstrapSolver

# ── 2. our side: the Orion placer and its sim0 out of plan_block ──────────────────────
from perseus.plan.placer.planner import PlanConfig, plan_block          # noqa: E402
import perseus.plan.placer.baselines.orion as ob                        # noqa: E402
from perseus.plan.placer.ir import is_kv_cache_read                     # noqa: E402

# The solver's frame: 50 levels over a refresh landing at 36, i.e. 6 composite levels between
# refreshes. It cannot plan at 48 (a block input then arrives with no level left, which its
# user-level pin cannot express); deploy.py enforces the deployed chain's top.
CFG = dict(bootstrap_level=36, max_level=50, source_level=36, cache_read_level=36,
           level_unit=2, acc_chain="n32", cf_max=20,
           sparse_precomps=() if DENSE else (512, 1),
           sparse_out_levels=() if DENSE else ((1, 26), (512, 36)),
           allow_prescale=False, verbose=False,
           # a dense plan lands the code's deliberate bootstraps at the bootstrap level (as
           # deploy.py and make_plan.sh do), so the solver sees the same step graph
           deliberate_clamp0=DENSE)


class _Captured(Exception):
    pass


def get_placer(block):
    cap = {}

    def spy(self, sim0):
        cap["placer"], cap["sim0"] = self, sim0
        raise _Captured

    orig = ob.OrionPlacer.choose_sites
    ob.OrionPlacer.choose_sites = spy
    try:
        plan_block(GRAPHS / block / "graph.json", PlanConfig(placer="orion", **CFG))
    except _Captured:
        pass
    finally:
        ob.OrionPlacer.choose_sites = orig
    return cap["placer"], cap["sim0"]


# ── 3. adapter: our step graph -> their NetworkDAG ────────────────────────────────────
class _Shape:
    def __init__(self, n):
        self._n = n

    def numel(self):
        return self._n


class _Params:
    def get_slots(self):
        return 1


class _Scheme:
    params = _Params()


class Mod:
    """Dummy module carrying exactly the attributes their solver reads."""
    def __init__(self, depth, n_cts, level=None):
        self.depth = depth
        self.level = level
        self.scheme = _Scheme()
        self.fhe_input_shape = _Shape(n_cts)
        self.fhe_output_shape = _Shape(n_cts)


BIG = 10 ** 6   # price of a boundary the runtime cannot bootstrap


def build_network_dag(sg, order, cost, crossing, out_x, pseudo, entry_level, l_eff):
    nd = NetworkDAG(trace=None)
    w = {}
    for s in order:
        if crossing.get(s) and not (s in pseudo and not any(
                is_kv_cache_read(v) for v in crossing.get(s, ()))):
            w[s] = max(1, len(out_x.get(s) or crossing.get(s, ())))
    for s in order:
        # a layer consuming the whole chain is infeasible in their model (their
        # estimate_bootstrap_latency returns inf at prev_level - depth <= 0): cap it at
        # l_eff - 1, one level left to bootstrap from
        d = min(cost.get(s, 0), l_eff - 1)
        nd.add_node(s, op="call_module", module=Mod(d, w.get(s, BIG)))
    for u, v in sg.edges:
        nd.add_edge(u, v)
    entry_pin = l_eff - entry_level
    assert entry_pin > 0, "their user-level pin cannot express level 0 (falsy)"
    nd.add_node("__ENTRY__", op="call_module", module=Mod(0, BIG, level=entry_pin))
    nd.add_node("__SINK__", op="output", module=None)
    for s in order:
        if sg.in_degree(s) == 0:
            nd.add_edge("__ENTRY__", s)
        if sg.out_degree(s) == 0:
            nd.add_edge(s, "__SINK__")
    return nd


def _alarm(sig, frm):
    raise TimeoutError


def run_upstream(nd, l_eff, timeout=1800):
    """Their BootstrapSolver.solve(), unchanged."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        nd.find_residuals()
        solver = BootstrapSolver(net=None, network_dag=nd, l_eff=l_eff)
        input_level, num_boots, _slots = solver.solve()
        return dict(ok=True, input_level=input_level, total_boots_ctweighted=num_boots,
                    marked_steps=sorted(n for n in nd.nodes if nd.nodes[n].get("bootstrap")),
                    num_unmodeled=len(getattr(solver, "unmodeled_nodes", [])))
    except TimeoutError:
        return dict(ok=False, error="TIMEOUT")
    except Exception as e:
        return dict(ok=False, error=f"{type(e).__name__}: {e}")
    finally:
        signal.alarm(0)


def main(blocks):
    results = []
    for blk in blocks:
        placer, sim0 = get_placer(blk)
        sg, order, cost, crossing, entry_level, *_ = placer._step_graph(sim0)
        l_eff = max(1, int((placer.budget.L - 1) // max(1, placer.g.level_unit)))
        nd = build_network_dag(sg, order, cost, crossing, placer._out_x,
                               placer._pseudo_sources, entry_level, l_eff)
        up = run_upstream(nd, l_eff)
        results.append(dict(block=blk, l_eff=l_eff, num_steps=len(order),
                            entry_level=entry_level, upstream=up))
        print(f"[{blk}] steps={len(order)} entry={entry_level} "
              f"marked={len(up['marked_steps']) if up['ok'] else up['error']}", flush=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=1))
    print("wrote", OUT_JSON)
    return 0 if all(r["upstream"]["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or sorted(
        (d.name for d in GRAPHS.glob("block_*") if (d / "graph.json").exists()),
        key=lambda n: int(n.split("_")[1]))))
