"""Every shipped plan regenerates from its graph with the recipe in its PLAN_CMD.txt.

A plan whose PLAN_CMD.txt names ``tool=examples/gpt2_from_primitives/make_plan.sh`` (the
Python implementation's plans) is regenerated with that tool: the blocks, then the argmax stage
(block 13, entered at the tail plan's exit), each stamped with the capture contract.

The main plan is always checked; the baselines and ablations (about 20 planner runs, a few
minutes) run when PERSEUS_ALL_PLANS=1. baselines/orion is the released Orion tool's output and
is skipped; python/orion is regenerated with scripts/utils/orion_upstream/plan.sh when an
upstream clone is present (ORION_SRC or .cache/orion_upstream; the test does not clone). Comparison: scripts/utils/plan_equiv.py (runtime content byte-identical;
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
ORION_TOOL = "scripts/utils/orion_upstream/plan.sh"


def _plan_dirs():
    dirs = [MAIN]
    if os.environ.get("PERSEUS_ALL_PLANS") == "1":
        for p in sorted(BP.rglob("PLAN_CMD.txt")):
            rel = str(p.parent.relative_to(BP))
            if rel != MAIN:
                dirs.append(rel)
    return dirs


def _recipe(plan_dir: Path):
    """(graph, recipe flags, tool) from PLAN_CMD.txt; None for a measured artifact (no
    `recipe=`). tool is None for the plain run_bootstrap_all_blocks.sh recipe."""
    graph = recipe = tool = None
    for line in (plan_dir / "PLAN_CMD.txt").read_text().splitlines():
        if line.startswith("graph="):
            graph = line[len("graph="):].split(" ")[0]
        elif line.startswith("recipe="):
            recipe = shlex.split(line[len("recipe="):])
        elif line.startswith("tool="):
            tool = line[len("tool="):].split(" ")[0]
    if recipe is None:
        return None
    assert graph, f"{plan_dir}/PLAN_CMD.txt lacks graph="
    return graph, recipe, tool


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
    graph, recipe, tool = parsed
    if not (REPO / graph).is_dir():
        pytest.skip(f"graph {graph} not present")
    out_name = f"_regen_{name.replace('/', '_')}"
    env = dict(os.environ, GRAPH_DIR=graph, OUT_NAME=out_name, PYTHON=sys.executable,
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    env.update(kv.split("=", 1) for kv in recipe)
    if tool is None:
        cmds = [["bash", "scripts/utils/run_bootstrap_all_blocks.sh"]]
    elif tool == ORION_TOOL:
        src = os.environ.get("ORION_SRC") or str(REPO / ".cache" / "orion_upstream")
        if not (Path(src) / "orion" / "core").is_dir():
            pytest.skip(f"no upstream Orion clone at {src}")
        env["ORION_SRC"] = src
        cmds = [["bash", tool, graph, out_name]]
    else:
        cmds = [["bash", tool, graph, out_name], ["bash", tool, graph, out_name, "argmax"]]
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
