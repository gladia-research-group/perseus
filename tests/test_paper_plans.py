"""Every shipped plan regenerates from its graph with the recipe in its PLAN_CMD.txt.

A plan whose PLAN_CMD.txt names ``tool=examples/gpt2_from_primitives/make_plan.sh`` (the
Python implementation's plans) is regenerated with that tool: the blocks, the argmax stage
(block 13, entered at the tail plan's exit) and, where the plan ships one, the feedback stage
(block 0 entered as a fed-back token), each stamped with the capture contract.

The main plan is always checked; the baselines and ablations (about 20 planner runs, a few
minutes) run when PERSEUS_ALL_PLANS=1. The python/ DaCapo and Orion plans are regenerated with the released tools
(scripts/utils/{dacapo,orion}_upstream/plan.sh) when they are present: hecate-opt at
HECATE_OPT or .cache/dacapo_upstream (build_hecate.sh), an Orion clone at ORION_SRC or
.cache/orion_upstream. The test neither clones nor builds them. Comparison: scripts/utils/plan_equiv.py (runtime content byte-identical;
summary additive only).
"""
import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BP = REPO / "bootstrap_placements"
MAIN = "gpt2_decode_python_n32"
# upstream-tool recipes: tool -> (env var, default location, what must exist there)
UPSTREAM = {
    "scripts/utils/orion_upstream/plan.sh": ("ORION_SRC", ".cache/orion_upstream", "orion/core"),
    "scripts/utils/dacapo_upstream/plan.sh": (
        "HECATE_OPT", ".cache/dacapo_upstream/build/bin/hecate-opt", ""),
}


def _plan_dirs():
    dirs = [MAIN]
    if os.environ.get("PERSEUS_ALL_PLANS") == "1":
        for p in sorted(BP.rglob("PLAN_CMD.txt")):
            rel = str(p.parent.relative_to(BP))
            if rel != MAIN:
                dirs.append(rel)
    return dirs


def _recipe(plan_dir: Path):
    """(graph, recipe flags, tool, route) from PLAN_CMD.txt; None for a measured artifact (no
    `recipe=`). tool is None for the plain run_bootstrap_all_blocks.sh recipe; route is
    `dense` for an upstream tool's dense plan."""
    graph = recipe = tool = None
    route = "sparse"
    for line in (plan_dir / "PLAN_CMD.txt").read_text().splitlines():
        if line.startswith("graph="):
            graph = line[len("graph="):].split(" ")[0]
        elif line.startswith("recipe="):
            recipe = shlex.split(line[len("recipe="):])
        elif line.startswith("tool="):
            tool = line[len("tool="):].split(" ")[0]
        elif line.startswith("route="):
            route = line[len("route="):].strip()
    if recipe is None:
        return None
    assert graph, f"{plan_dir}/PLAN_CMD.txt lacks graph="
    return graph, recipe, tool, route


@pytest.fixture(scope="module")
def equiv():
    spec = importlib.util.spec_from_file_location("plan_equiv", REPO / "scripts" / "utils" / "plan_equiv.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.equivalent


@pytest.mark.parametrize("name", _plan_dirs())
def test_plan_regenerates(name, equiv, tmp_path):
    plan_dir = BP / name
    parsed = _recipe(plan_dir)
    if parsed is None:
        pytest.skip(f"{name} is a measured artifact, not regenerated")
    graph, recipe, tool, route = parsed
    if not (REPO / graph).is_dir():
        pytest.skip(f"graph {graph} not present")
    out_name = f"_regen_{name.replace('/', '_')}"
    env = dict(os.environ, GRAPH_DIR=graph, OUT_NAME=out_name, PYTHON=sys.executable,
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    env.update(kv.split("=", 1) for kv in recipe)
    if tool is None:
        cmds = [["bash", "scripts/utils/run_bootstrap_all_blocks.sh"]]
    elif tool in UPSTREAM:
        var, default, inside = UPSTREAM[tool]
        src = os.environ.get(var) or str(REPO / default)
        if not (Path(src) / inside).exists():
            pytest.skip(f"no {var} at {src}")
        env[var] = src
        cmds = [["bash", tool, graph, out_name, route]]
    else:
        cmds = [["bash", tool, graph, out_name], ["bash", tool, graph, out_name, "argmax"]]
        if (plan_dir / "block_0_feedback_placement.json").exists():
            cmds.append(["bash", tool, graph, out_name, "feedback"])
    log = tmp_path / "plan.log"
    rc = 0
    with log.open("w") as f:
        for cmd in cmds:
            rc = rc or subprocess.run(cmd, cwd=REPO, env=env, stdout=f,
                                      stderr=subprocess.STDOUT).returncode
    regen = BP / out_name
    try:
        assert rc == 0, log.read_text()[-2000:]
        blocks = sorted(plan_dir.glob("block_*_placement.json"))
        assert blocks, f"no block files in {plan_dir}"
        for ref in blocks:
            ok, why = equiv(str(regen / ref.name), str(ref))
            assert ok, f"{name}/{ref.name}: {why}"
    finally:
        subprocess.run(["rm", "-rf", str(regen)])
