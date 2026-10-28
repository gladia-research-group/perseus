"""Every shipped plan regenerates from its graph with the recipe in its PLAN_CMD.txt.

Plans made with ``perseus/examples/gpt2/make_plan.sh`` are regenerated with it (blocks, argmax stage, feedback
stage). The main plan is always checked; PERSEUS_ALL_PLANS=1 adds the baselines and ablations, PERSEUS_PLANS=a,b
names plans. Comparison: `PLAN_FIELDS` field by field, then the runtime body (perseus.plan.equiv) with the
contract stamps left out. The planner gate (`PC_PLAN_GATE=1`; by hand: `python python/tests/test_paper_plans.py
[name ...]`) replans `GATE`, whose recipes spell out what incomplete PLAN_CMD.txt files omit. DaCapo / Orion
upstream-tool plans skip (their tools are not ported).
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from perseus._paths import ASSETS, PKG
from perseus.plan import contract as _contract
from perseus.plan import equiv as _equiv

REPO = ASSETS
BP = REPO / "bootstrap_placements"
MAIN = "gpt2_decode"
# the released DaCapo / Orion drivers were repo scripts, not ported
UPSTREAM = ("scripts/utils/orion_upstream/plan.sh", "scripts/utils/dacapo_upstream/plan.sh")
GPT2_TOOL = "python/perseus/examples/gpt2/make_plan.sh"
LLAMA_TOOL = "python/perseus/examples/llama/make_plan.sh"
# tool spellings in PLAN_CMD.txt (the C++-runtime layout, then this one) -> the package's script
TOOLS = {
    "examples/gpt2_from_primitives/make_plan.sh": PKG / "examples" / "gpt2" / "make_plan.sh",
    "examples/llama_from_primitives/make_plan.sh": PKG / "examples" / "llama" / "make_plan.sh",
    GPT2_TOOL: PKG / "examples" / "gpt2" / "make_plan.sh",
    LLAMA_TOOL: PKG / "examples" / "llama" / "make_plan.sh",
}
LLAMA_TOOLS = (LLAMA_TOOL, "examples/llama_from_primitives/make_plan.sh")
# the stamp keys: a contract, not plan content
STAMPS = ("capture_env", "config")
# what the runtime binds of a plan, compared one by one (`summary.*` = nested summary keys)
PLAN_FIELDS = ("placements", "rescale_after", "sparse_slots", "correction_factor", "real_route", "raise_drop",
               "hint_fire", "summary.final_named_levels", "summary.final_named_degs")

# name -> (graph override, recipe beyond PLAN_CMD.txt's recipe= line, tool override, stages); None = as shipped
GATE: dict[str, tuple | None] = {
    "gpt2_decode": None,
    "gpt2_decode_ctx1024": None,
    # the encoders and the GPT-2 prefill (the tail and CutMax are its recording's blocks 12 and 13): the gpt2
    # make_plan.sh blocks stage
    "bert_sst2_128": (None, "", GPT2_TOOL, ("blocks",)),
    "vit_b16_224": (None, "", GPT2_TOOL, ("blocks",)),
    "gpt2_prefill128": (None, "", GPT2_TOOL, ("blocks",)),
    "llama31_8b_decode": None,
    # one planner run on the fake recording: the llama make_plan.sh prefill stage (its recipe= line)
    "llama31_8b_prefill128": (None, "", None, ("prefill",)),
    # the decode cut for the prefill's K / V (KV_ENTRY_LEVEL, PLAN_KEEP_PLACEMENTS, PLAN_SITE_CF in its recipe= line)
    "llama31_8b_handoff_decode": None,
}
# plans this planner does not regenerate, and why
RETIRED: dict[str, str] = {}
# the result plans: `python python/tests/test_paper_plans.py table` replans these
TABLE = ("llama31_8b_prefill128", "llama31_8b_decode", "gpt2_prefill128", "gpt2_decode_ctx1024", "gpt2_decode",
         "bert_sst2_128", "vit_b16_224")
# PLAN_KEEP_PLACEMENTS dirs that are not shipped: name -> the recipe changes that make them from the plan's own recipe
# (KV_ENTRY_LEVEL, PLAN_KEEP_PLACEMENTS and PLAN_SITE_CF dropped; PLAN_CMD.txt `sites=`), replanned first
KEEP_SITES = {"llama31_8b_handoff_decode": "PLAN_REFRESH_ENV_CAP=48"}
_KEEP_DROP = ("KV_ENTRY_LEVEL", "PLAN_KEEP_PLACEMENTS", "PLAN_SITE_CF")
# stages a separate tool plans from a graph that is not shipped (speedrun/make_tail_plan.sh): not regenerated here
TAIL_STAGES: dict[str, tuple[str, ...]] = {}


# summary diagnostics a later planner computes differently than the shipped file recorded: left out of the body
# comparison of that plan only
SUMMARY_DRIFT: dict[str, tuple[str, ...]] = {}
# summary lists the planner fills in set-iteration order (PYTHONHASHSEED): compared as sets, every plan
UNORDERED = ("bts_quality.env_cap.placed_past_cap",)


def _plan_dirs():
    if os.environ.get("PERSEUS_PLANS"):     # comma-separated plan names: those only
        return os.environ["PERSEUS_PLANS"].split(",")
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


def _get(doc, field: str):
    for k in field.split("."):
        if not isinstance(doc, dict) or k not in doc:
            return None
        doc = doc[k]
    return doc


def _count(ref, new) -> int:
    """Differing leaves between two values (dict keys / list items; type-strict leaves)."""
    if isinstance(ref, dict) and isinstance(new, dict):
        return sum(_count(ref.get(k), new.get(k)) for k in set(ref) | set(new))
    if isinstance(ref, list) and isinstance(new, list):
        return sum(_count(a, b) for a, b in zip(ref, new)) + abs(len(ref) - len(new))
    return 0 if (type(ref) is type(new) and ref == new) else 1


def field_diffs(new_path: Path, ref_path: Path) -> dict[str, int]:
    """Per PLAN_FIELDS entry, the number of differing leaves (0 = identical, including both absent)."""
    new, ref = (json.loads(Path(p).read_text()) for p in (new_path, ref_path))
    return {f: _count(_get(ref, f), _get(new, f)) for f in PLAN_FIELDS}


def _without_stamps(src: Path, dst: Path, drift: tuple[str, ...] = ()) -> Path:
    doc = json.loads(src.read_text())
    for k in STAMPS:
        doc.pop(k, None)
    for path in drift:
        *up, last = path.split(".")
        d = doc.get("summary", {})
        for k in up:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        if isinstance(d, dict):
            d.pop(last, None)
    for path in UNORDERED:
        *up, last = path.split(".")
        d = doc.get("summary", {})
        for k in up:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        if isinstance(d, dict) and isinstance(d.get(last), list):
            d[last] = sorted(d[last], key=json.dumps)
    dst.write_text(json.dumps(doc))
    return dst


def body_equiv(new_path: Path, ref_path: Path, tmp: Path, drift: tuple[str, ...] = ()):
    """perseus.plan.equiv with the contract stamps (and the plan's SUMMARY_DRIFT keys) left out of both."""
    return _equiv.equivalent(str(_without_stamps(new_path, tmp / "new.json", drift)),
                             str(_without_stamps(ref_path, tmp / "ref.json", drift)))


def _out_name(name: str) -> str:
    return f"_regen_{name.replace('/', '_')}"


def replan(name: str, log: Path) -> tuple[Path, float]:
    """Re-plan bootstrap_placements/<name> into bootstrap_placements/_regen_<name> (the caller removes it); returns
    (out dir, wall seconds). Skips (pytest) when the recipe or the graph is not available."""
    plan_dir = BP / name
    spec = GATE.get(name)
    parsed = _recipe(plan_dir)
    graph, recipe, tool = (parsed[0], list(parsed[1]), parsed[2]) if parsed else (None, [], None)
    stages = None
    if spec is not None:
        g2, extra, tool2, stages = spec
        graph = g2 or graph
        recipe += shlex.split(extra)
        tool = tool2 or tool
    elif parsed is None:
        pytest.skip(f"{name} is a measured artifact, not regenerated")
    if graph is None or not ((REPO / graph).is_dir() or Path(graph).is_dir()):
        pytest.skip(f"graph {graph} not present")
    out_name = _out_name(name)
    pypath = str(PKG.parent) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    env = dict(os.environ, GRAPH_DIR=graph, OUT_NAME=out_name, PYTHON=sys.executable,
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", PYTHONPATH=pypath)
    env.update(kv.split("=", 1) for kv in recipe)
    pre = []
    if name in KEEP_SITES:          # the kept-sites plan first, into the dir the recipe names
        keep = Path(env["PLAN_KEEP_PLACEMENTS"]).name
        base = [kv for kv in recipe if kv.split("=", 1)[0] not in _KEEP_DROP] + shlex.split(KEEP_SITES[name])
        benv = {k: v for k, v in env.items() if k not in _KEEP_DROP}
        benv.update(kv.split("=", 1) for kv in base)
        benv["OUT_NAME"] = keep
        pre = [(["bash", str(TOOLS[tool]), graph, keep], benv)]
    if tool is None:
        cmds = [["bash", str(PKG / "plan" / "run_bootstrap_all_blocks.sh")]]
    elif tool in UPSTREAM:
        pytest.skip(f"{tool}: the released-tool drivers are not ported")
    else:
        if stages is None:
            stages = ("blocks",)
            if tool not in LLAMA_TOOLS:
                stages += ("argmax",)
                if (plan_dir / "block_0_feedback_placement.json").exists():
                    stages += ("feedback",)
        cmds = [["bash", str(TOOLS[tool]), graph, out_name] + ([] if s == "blocks" else [s]) for s in stages]
    rc = 0
    t0 = time.perf_counter()
    try:
        with log.open("w") as f:
            for cmd, e in pre + [(c, env) for c in cmds]:
                rc = rc or subprocess.run(cmd, cwd=REPO, env=e, stdout=f, stderr=subprocess.STDOUT).returncode
    finally:
        for _, e in pre:
            subprocess.run(["rm", "-rf", str(BP / e["OUT_NAME"])], check=False)
    wall = time.perf_counter() - t0
    assert rc == 0, log.read_text()[-3000:]
    return BP / out_name, wall


def compare(name: str, regen: Path, tmp: Path) -> dict:
    """{block file: {field: differing leaves}}, plus "<file>:body" -> equiv's reason where the body differs and
    "<file>:stamp" where the replanned Config stamp is not the shipped legacy stamp read through the Config."""
    plan_dir = BP / name
    blocks = sorted(b for b in plan_dir.glob("block_*_placement.json") if b.name not in TAIL_STAGES.get(name, ()))
    assert blocks, f"no block files in {plan_dir}"
    out: dict = {}
    for ref in blocks:
        new = regen / ref.name
        assert new.exists(), f"{name}: no regenerated {ref.name}"
        out[ref.name] = field_diffs(new, ref)
        ok, why = body_equiv(new, ref, tmp, SUMMARY_DRIFT.get(name, ()))
        if not ok:
            out[f"{ref.name}:body"] = why
        # the replanned file carries the Config stamp, equal to the shipped legacy stamp read through the Config
        st_new, st_ref = (json.loads(p.read_text()) for p in (new, ref))
        if "capture_env" in st_new or (st_ref.get("capture_env") is not None
                                       and st_new.get("config") != _contract.stamp_of(st_ref["capture_env"])):
            out[f"{ref.name}:stamp"] = (f"stamp: {st_new.get('config')!r} vs shipped "
                                       f"{st_ref.get('capture_env')!r}")[:600]
    return out


def _assert_identical(name: str, res: dict):
    bad = {k: v for k, v in res.items() if isinstance(v, str) or any(v.values())}
    assert not bad, f"{name}: {json.dumps(bad, indent=1)[:3000]}"


def _run(name: str, tmp_path: Path):
    if name in RETIRED:
        pytest.skip(f"{name}: {RETIRED[name]}")
    try:
        regen, _ = replan(name, tmp_path / "plan.log")
        _assert_identical(name, compare(name, regen, tmp_path))
    finally:
        subprocess.run(["rm", "-rf", str(BP / _out_name(name))])


@pytest.mark.parametrize("name", _plan_dirs())
def test_plan_regenerates(name, tmp_path):
    _run(name, tmp_path)


def _gate_names():
    return [n for n in GATE if n != MAIN] if os.environ.get("PC_PLAN_GATE") == "1" else []


@pytest.mark.parametrize("name", _gate_names())
def test_gate_plan_regenerates(name, tmp_path):
    _run(name, tmp_path)


def main(names: list[str]) -> int:
    """Replan each of `names`, print per-plan field diffs and wall time; exit status = number of failing plans."""
    import tempfile
    failed = 0
    for name in names:
        if name in RETIRED:
            print(f"[gate] {name}: RETIRED ({RETIRED[name]})", flush=True)
            continue
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            logdir = Path(os.environ.get("PLAN_GATE_LOGDIR", td))
            try:
                regen, wall = replan(name, logdir / f"{name}.plan.log")
                res = compare(name, regen, tmp)
            except BaseException as e:  # noqa: BLE001  (pytest.skip is a BaseException)
                print(f"[gate] {name}: ERROR {type(e).__name__}: {str(e)[-1500:]}", flush=True)
                failed += 1
                continue
            finally:
                subprocess.run(["rm", "-rf", str(BP / _out_name(name))])
            files = {k: v for k, v in res.items() if ":" not in k}
            tot = {f: sum(v[f] for v in files.values()) for f in PLAN_FIELDS}
            body = [k for k in res if ":" in k]
            ok = not body and not any(tot.values())
            failed += not ok
            print(f"[gate] {name}: {'IDENTICAL' if ok else 'DIFFERS'} {len(files)} files, {wall:.1f} s, "
                  f"field diffs {tot}, body diffs {len(body)}", flush=True)
            for k in body:
                print(f"[gate]   {k}: {res[k]}", flush=True)
            for k, v in files.items():
                if any(v.values()):
                    print(f"[gate]   {k}: { {f: n for f, n in v.items() if n} }", flush=True)
    return failed


if __name__ == "__main__":
    args = sys.argv[1:] or list(GATE)
    sys.exit(main([n for a in args for n in (TABLE if a == "table" else (a,))]))
