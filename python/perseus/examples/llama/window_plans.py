"""The 512-context decode window plans of a decoder (arch.MODELS) from a reference model's shipped windows: the
reference's recipe and runtime (identical across its windows) with the route-sized switches moved to this model's
attention block tH, this model's configs, then the recordings and plans graphs/LLAMA.txt describes for Llama (CPU only).

    python -m perseus.examples.llama.window_plans --model gemma2_2b                # derive, capture, merge, plan
    python -m perseus.examples.llama.window_plans --model gemma2_2b --stage plan   # re-plan from the recordings
    python -m perseus.examples.llama.window_plans --model gemma2_2b --windows 512 --name NAME --configs CFG  # a variant

Stages: `derive` writes bootstrap_placements/<alias>_decode_ctx512_w<B>/PLAN_CMD.txt; `capture` records the structure
at position 1 (magnitudes from the fake's tape, capture_mags) and captures every window position from a plain prefix
(--jobs at a time, ~35 GB each for Gemma-2-2B); `merge` builds graphs/<alias>_decode_ctx512_w<B>
(scripts/utils/merge_window_mags.py); `plan` runs scripts/make_plans.sh on each window (its recipe, stages, rd= pass).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys

from perseus._paths import PKG

REPO = PKG.parent.parent
# bound -> the positions whose magnitudes size the window (Llama's shipped windows)
WINDOWS = {16: (1, 8, 15), 128: (16, 64, 127), 256: (128, 192, 255), 384: (256, 320, 383), 512: (384, 448, 511)}
# the switches whose values name a route by its slot count
ROUTE_KEYS = ("PLAN_SPARSE_BTS_OUT", "PLAN_SPARSE_SLOTS", "PLAN_RAISE_DROP_BASE", "PLAN_RAISE_ROUTES",
              "SPARSE_BTS_SLOTS", "FIDESLIB_BTS_RAISE_DROP")


def read_cmd(path):
    out = {}
    with open(path) as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition("=")
            out[k] = v
    return out


def retarget(switches: str, th_from: int, th_to: int) -> str:
    """`switches` (KEY=VALUE ...) with the route `th_from` renamed `th_to` in ROUTE_KEYS' values (a list of routes or of
    route:value pairs); every other switch as is."""
    out = []
    for kv in switches.split():
        k, _, v = kv.partition("=")
        if k in ROUTE_KEYS:
            v = ",".join(":".join([str(th_to)] + p.split(":")[1:]) if p.split(":")[0] == str(th_from) else p
                         for p in v.split(","))
        out.append(f"{k}={v}")
    return " ".join(out)


def route_of(switches: str) -> int:
    """The sparse route (the first SPARSE_BTS_SLOTS entry past 1) of a runtime line."""
    for kv in switches.split():
        if kv.startswith("SPARSE_BTS_SLOTS="):
            return max(int(s) for s in kv.split("=", 1)[1].split(",") if s)
    raise ValueError("no SPARSE_BTS_SLOTS in the reference runtime")


def step_ops(graph_json):
    """{step: [op types]} of a capture block, the shape-named steps (`~` in the name: q.K^T, the Gp sub-steps) left out:
    the op lists a window plan binds by name at every position."""
    import json
    from perseus.impl.magtape import _REFRESH            # refreshes: placed by magnitude, the plan's to decide
    out = {}
    for n in json.load(open(graph_json))["nodes"]:
        st = str(n.get("step"))
        if n["op_type"] in _REFRESH or "~" in st or "~" in str(n.get("output", "")) or st.endswith("kv_push"):
            continue                                       # the K / V push: dynamic per position (driver)
        out.setdefault(st, []).append(n["op_type"])
    return out


def check_steps(graphs, name, positions):
    """Every block's op lists outside the shape-named steps agree between the position-1 recording and each window
    position's: a plan made on one binds at all (else the device compile fails on a shifted name)."""
    import glob
    ref = graphs / f"{name}_p1"
    bad = []
    for gj in sorted(glob.glob(str(ref / "block_*" / "graph.json"))):
        blk = os.path.basename(os.path.dirname(gj))
        a = step_ops(gj)
        for p in positions:
            other = graphs / f"{name}_p{p}" / blk / "graph.json"
            if p == 1 or not other.exists():
                continue
            b = step_ops(str(other))
            for st in sorted(set(a) | set(b)):
                if a.get(st) != b.get(st):
                    bad.append(f"{blk} {st}: p1 {len(a.get(st, []))} ops, p{p} {len(b.get(st, []))}")
    if bad:
        sys.exit("[windows] op lists differ across positions (a plan would not bind):\n  " + "\n  ".join(bad[:12]))
    print(f"[windows] op lists agree across positions {positions}", flush=True)


def run(cmd, log, env=None):
    print(f"[windows] {' '.join(cmd)[:200]} > {log}", flush=True)
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO / "python", env={**os.environ, **(env or {})})
    if r.returncode:
        sys.exit(f"[windows] failed ({r.returncode}): see {log}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="arch.MODELS alias")
    ap.add_argument("--ref", default="llama31_8b_decode_ctx512", help="the reference windows' plan prefix")
    ap.add_argument("--stage", default="all", choices=("all", "derive", "capture", "merge", "plan"))
    ap.add_argument("--configs", default=None, help="the windows' config (default: the model's, arch.MODELS)")
    ap.add_argument("--name", default=None, help="the plans' prefix (default <alias>_decode_ctx512)")
    ap.add_argument("--windows", default=None, help="bounds to build (default all: 16,128,256,384,512)")
    ap.add_argument("--jobs", type=int, default=3, help="position captures at a time")
    ap.add_argument("--threads", type=int, default=16, help="OMP threads per capture")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="a runtime switch on top of the reference's (the windows' runtime= line)")
    a = ap.parse_args(argv)

    from . import hf
    from . import weights as W
    from .arch import MODELS
    hf.select(a.model)
    m = MODELS[a.model]
    dims = W.llama_dims(hf.config())
    name = a.name or f"{a.model}_decode_ctx512"
    configs = a.configs or m.configs
    wins = {b: WINDOWS[b] for b in (map(int, a.windows.split(",")) if a.windows else WINDOWS)}
    ref = read_cmd(REPO / "bootstrap_placements" / f"{a.ref}_w{min(WINDOWS)}" / "PLAN_CMD.txt")
    th_ref = route_of(ref["runtime"])
    recipe = retarget(ref["recipe"], th_ref, dims.tH)
    runtime = retarget(ref["runtime"], th_ref, dims.tH)
    if a.set:
        over = dict(kv.split("=", 1) for kv in a.set)
        runtime = " ".join([kv for kv in runtime.split() if kv.split("=", 1)[0] not in over]
                           + [f"{k}={v}" for k, v in over.items()])
    graphs = REPO / "graphs"
    logs = REPO / "logs" / "plans"
    logs.mkdir(parents=True, exist_ok=True)
    py = os.environ.get("PYTHON", sys.executable)
    rt = ["--set", "CORRECTION_FACTOR=6"] + [x for kv in runtime.split() for x in ("--set", kv)]
    common = ["--model", a.model, "--configs", configs, "--window", str(graphs / f"{name}_w0.npy"),
              "--window-tokens", "512"]
    cpu = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(a.threads)}

    if a.stage in ("all", "derive"):
        for b in wins:
            d = REPO / "bootstrap_placements" / f"{name}_w{b}"
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "PLAN_CMD.txt", "w") as f:
                f.write(f"graph=graphs/{name}_w{b}\nmodel={a.model}\ntool={ref['tool']}\nrecipe={recipe}\n"
                        f"runtime={runtime}\nconfigs={configs}\nrd={ref['rd']}\n")
        print(f"[windows] derived {name}_w{{{','.join(map(str, wins))}}} from {a.ref} (route {th_ref} -> {dims.tH})")

    if a.stage in ("all", "capture"):
        import numpy as np
        graphs.mkdir(exist_ok=True)
        np.save(graphs / f"{name}_w0.npy", hf.text_windows(1, 512)[0])
        p1 = str(graphs / f"{name}_p1")
        # a capture keeps any block it already holds (fakecap: the first position wins): a stale recording goes first
        for p in [1] + sorted({p for ps in wins.values() for p in ps if p > 1}):
            shutil.rmtree(graphs / f"{name}_p{p}", ignore_errors=True)
        run([py, "-m", "perseus.examples.llama.run_decode", "--fake", "--tokens", "1", "--capture", p1,
             "--structure-only", *common, *rt], logs / f"{name}_p1.log", cpu)
        run([py, "-m", "perseus.examples.llama.capture_mags", p1, "--model", a.model, "--configs", configs,
             "--window", str(graphs / f"{name}_w0.npy"), *rt], logs / f"{name}_p1_mags.log", cpu)
        pos = sorted({p for ps in wins.values() for p in ps if p > 1})
        with cf.ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(lambda p: run([py, "-m", "perseus.examples.llama.run_decode", "--fake", "--tokens", str(p),
                                       "--plain-prefix", str(p), "--capture", str(graphs / f"{name}_p{p}"),
                                       *common, *rt], logs / f"{name}_p{p}.log", cpu), pos))

    if a.stage in ("all", "merge"):
        check_steps(graphs, name, sorted({p for ps in wins.values() for p in ps}))
        for b, ps in wins.items():
            caps = [str(graphs / f"{name}_p{p}") for p in ps]
            shutil.rmtree(graphs / f"{name}_w{b}", ignore_errors=True)     # the merge refuses an existing window
            run([py, str(REPO / "scripts" / "utils" / "merge_window_mags.py"), str(graphs / f"{name}_p1"),
                 str(graphs / f"{name}_w{b}"), "--win", *caps, "--all", *caps, "--lo", "--periods", *caps],
                logs / f"{name}_w{b}_merge.log")

    if a.stage in ("all", "plan"):
        for b in wins:
            r = subprocess.run(["bash", str(REPO / "scripts" / "make_plans.sh"), f"{name}_w{b}"], cwd=REPO,
                               env={**os.environ, "PYTHON": py})
            if r.returncode:
                sys.exit(f"[windows] plan {name}_w{b} failed: logs/plans/{name}_w{b}.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
