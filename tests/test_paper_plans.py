"""Every shipped plan regenerates from its graph with the recipe in its PLAN_CMD.txt.

The main plan is always checked; the baselines and ablations (about 20 planner runs, a few
minutes) run when PERSEUS_ALL_PLANS=1. baselines/orion is the released Orion tool's output
and is skipped. Comparison: scripts/utils/plan_equiv.py (runtime content byte-identical;
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
MAIN = "gpt2_decode_n32"


def _plan_dirs():
    dirs = [MAIN]
    if os.environ.get("PERSEUS_ALL_PLANS") == "1":
        for p in sorted(BP.rglob("PLAN_CMD.txt")):
            rel = str(p.parent.relative_to(BP))
            if rel != MAIN:
                dirs.append(rel)
    return dirs


def _recipe(plan_dir: Path):
    """(graph, recipe flags) from PLAN_CMD.txt; None for a measured artifact (no `recipe=`)."""
    graph = recipe = None
    for line in (plan_dir / "PLAN_CMD.txt").read_text().splitlines():
        if line.startswith("graph="):
            graph = line[len("graph="):].split(" ")[0]
        elif line.startswith("recipe="):
            recipe = shlex.split(line[len("recipe="):])
    if recipe is None:
        return None
    assert graph, f"{plan_dir}/PLAN_CMD.txt lacks graph="
    return graph, recipe


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
    graph, recipe = parsed
    if not (REPO / graph).is_dir():
        pytest.skip(f"graph {graph} not present")
    out_name = f"_regen_{name.replace('/', '_')}"
    env = dict(os.environ, GRAPH_DIR=graph, OUT_NAME=out_name, PYTHON=sys.executable,
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    env.update(kv.split("=", 1) for kv in recipe)
    log = tmp_path / "plan.log"
    with log.open("w") as f:
        rc = subprocess.run(["bash", "scripts/utils/run_bootstrap_all_blocks.sh"],
                            cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
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
