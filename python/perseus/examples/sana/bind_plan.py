"""Bind a SANA / text-to-image plan input by input (CPU only). The stages differ, so no uniform entry binds the
chain: stage 0 is planned from the level each encrypted input enters at (its capture graph), every later stage from
its predecessor's exit map (the planner's PLAN_ENTRY_MAP: each input at its own level; a pass-through copy keeps the
level its source entered with), in Jacobi passes until each stage's planned map is its predecessor's exit map. Then
PLAN_CMD.txt (make_plan.sh's) and the chain compiled offline from the capture graphs, each stage against its
predecessor's compiled exit (as planned.SanaPlannedPrefill.prepare compiles it).

    python -m perseus.examples.sana.bind_plan <graph dir> <plan name> [--jobs 10] [--seed a:z,...] [--warm <plan>]
        [--write-only]

The stages' first plans (make_plan.sh's parallel pass) seed the exit maps; `--seed a:z[,a:z...]` (the first pass of
this invocation) has stages a+1..z take stage a's exit (a run of identical blocks has one fixed point, so a pass per
stage is spared: e.g. each sampler step's blocks from a converged one). `--warm <plan>` (make_plan.sh's WARM=): no
first plans; the first pass plans every stage from that bound plan's converged entry maps (logs/entry_maps.json: a
capture of the same layout with other levers exits close to it), a stage whose inputs differ from 32 / 1. Exported
PLAN_* values win over the recipe.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import levers

_NOT_CT = ("pt_", "const(", "rot(", "hint_lev(", "lvl_cap(")
SCHEDULE = "siblings"                    # the runs' (impl.prefill.PlannedPrefill's default): the saved programs are theirs


def recipe(G):
    r = {k: os.environ.get(k, v) for k, v in levers.RECIPE.items()}
    if (G / "pcmm_stamps.json").exists():
        r["PLAN_PCMM_STAMPS"] = str(G / "pcmm_stamps.json")
    return r


def entry0(G):
    """Stage 0's entry map: {position: [level, 1]}, each encrypted input at the level the capture first reads it."""
    from perseus.plan.placer.planner import stage_n_inputs
    seen = {}
    for nd in json.load(open(G / "block_0" / "graph.json"))["nodes"]:
        for x, lv in zip(nd["inputs"], nd.get("input_levels", [])):
            if not x.startswith(_NOT_CT) and x not in seen:
                seen[x] = int(lv)
    anon = [x for x in seen if x.startswith("anon_")]
    names = anon[:1] + [f"pf_in{c}" for c in range(1, stage_n_inputs(G / "block_0" / "graph.json"))]
    return {str(i): [seen[n], 1] for i, n in enumerate(names) if n in seen}


class Binder:
    def __init__(self, G, out, jobs):
        self.G, self.out, self.jobs = G, out, jobs
        self.repo = Path(__file__).resolve().parents[4]
        self.D = self.repo / "bootstrap_placements" / out
        self.tmp = G.parent / f"_bind_{out}"
        self.recipe = recipe(G)
        self.n = len(list(G.glob("block_*")))
        self._n_out = {}

    def graph(self, b):
        return self.G / f"block_{b}" / "graph.json"

    def n_out(self, b):
        """The values stage b returns (stage b + 1's input positions): those stage b + 1 reads, but a DiT block returns
        its input's shape (X, C and T in place), and the last step's head reads no T"""
        if b not in self._n_out:
            from perseus.plan.placer.planner import stage_n_inputs
            nxt = stage_n_inputs(self.graph(b + 1))
            step = json.load(open(self.graph(b)))["nodes"][-1].get("step", "")
            blk = re.fullmatch(r"(s\d+\.)?blk\d+", step) is not None
            self._n_out[b] = max(nxt, stage_n_inputs(self.graph(b))) if blk else nxt
        return self._n_out[b]

    def exit_map(self, b, n_next):
        from perseus.plan.placer.planner import stage_exit_map
        s = json.load(open(self.D / f"block_{b}_placement.json"))["summary"]
        return {k: list(v) for k, v in stage_exit_map(s, self.graph(b), 2, n_next).items()}

    def copy_srcs(self, b, n_next):
        """{output position: input position} of stage b's outputs that are copies of its inputs (pass-through)."""
        from perseus.plan.placer.planner import stage_outputs_ordered
        order, copies = stage_outputs_ordered(self.graph(b), 2, n_next)
        return {str(i): str(copies[v]) for i, v in enumerate(order) if v in copies}

    def plan_one(self, b, emap):
        """Stage b alone (block_0 of its own graph dir) from `emap`; its placement replaces the plan's. Returns its
        refreshes, or None."""
        t = self.tmp / f"s{b}"
        subprocess.run(["rm", "-rf", str(t)], check=True)
        t.mkdir(parents=True)
        (t / "block_0").symlink_to(self.G / f"block_{b}")
        for f in ("capture_env.json", "capture_config.json"):
            if (self.G / f).exists():
                subprocess.run(["cp", str(self.G / f), str(t)], check=True)
        (self.tmp / f"map{b}.json").write_text(json.dumps(emap))
        deep = max(emap.values(), key=lambda v: v[0] + (2 if v[1] == 2 else 0))
        o = f"{self.out}__m{b}"
        env = dict(os.environ, **self.recipe, PYTHONPATH=str(self.repo / "python"),
                   PYTHON=os.environ.get("PYTHON", sys.executable), BLOCKS_ENTRY_LEVEL=str(deep[0]),
                   BLOCKS_ENTRY_DEG=str(deep[1]), PLAN_ENTRY_MAP=str(self.tmp / f"map{b}.json"))
        with open(self.tmp / f"s{b}.log", "w") as log:
            r = subprocess.run(["bash", str(self.repo / "python/perseus/examples/gpt2/make_plan.sh"), str(t), o],
                               cwd=self.repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        src = self.repo / "bootstrap_placements" / o / "block_0_placement.json"
        if r.returncode or not src.exists():
            return None
        os.replace(src, self.D / f"block_{b}_placement.json")
        subprocess.run(["rm", "-rf", str(self.repo / "bootstrap_placements" / o)], check=True)
        subprocess.run(["cp", str(self.tmp / f"s{b}.log"), str(self.D / "logs" / f"s{b}.log")], check=True)
        return json.load(open(self.D / f"block_{b}_placement.json"))["summary"]["total_bootstraps"]

    def warm_maps(self, src, n_in):
        """Stage b's first-pass map: stage 0's capture entry, the others `src`'s converged map where it names the same
        input positions (else every input at 32 / 1, make_plan.sh's first-plan entry)."""
        d = Path(src) if os.path.isdir(src) else self.repo / "bootstrap_placements" / src
        w = json.loads((d / "logs" / "entry_maps.json").read_text())
        ex, hit, mode = {0: entry0(self.G)}, 0, 0
        for b in range(1, self.n):
            want = {str(i) for i in range(n_in[b])}
            m = w.get(str(b))
            if m is not None and set(m) == want:
                ex[b], hit = {k: list(v) for k, v in m.items()}, hit + 1
            elif m:                          # another layout (resolution, fold): its most common entry everywhere
                from collections import Counter
                top = Counter(tuple(v) for v in m.values()).most_common(1)[0][0]
                ex[b], mode = {k: list(top) for k in sorted(want, key=int)}, mode + 1
            else:
                ex[b] = {k: [32, 1] for k in sorted(want, key=int)}
        print(f"[bind] warm from {d.name}: {hit} of {self.n - 1} stages take its converged map, {mode} its most "
              f"common entry", flush=True)
        return ex

    def bind(self, seeds=(), warm=None):
        self.tmp.mkdir(exist_ok=True)
        (self.D / "logs").mkdir(parents=True, exist_ok=True)
        maps_f = self.D / "logs" / "entry_maps.json"          # {stage: the map it was planned from}
        maps = {int(k): v for k, v in json.loads(maps_f.read_text()).items()} if maps_f.exists() else {}
        n = self.n
        n_in = {b: self.n_out(b - 1) for b in range(1, n)}       # the positions stage b is handed (it may read fewer)
        srcs = {b: self.copy_srcs(b - 1, n_in[b]) for b in range(2, n)}
        def sweep(seeds):
            # forward sweep: a copied output carries the level its source input enters with in this pass (stage
            # b - 1's plan only echoes the level it was planned from)
            ex = {0: entry0(self.G), 1: self.exit_map(0, n_in[1])}
            for b in range(2, n):
                src = next((sa for sa, sz in seeds if sa < b - 1 and b <= sz and n_in[b] == n_in[sa + 1]), b - 1)
                e = self.exit_map(src, n_in[b])
                for i, c in srcs[b].items():
                    if c in ex[b - 1]:
                        e[i] = list(ex[b - 1][c])
                ex[b] = e
            return ex

        for p in range(40):
            if p == 0 and warm:                     # no first plans to sweep: every stage from the warm maps
                ex = self.warm_maps(warm, n_in)
                todo = list(range(n))
                print(f"[bind] pass 0: {n} stages from the warm maps", flush=True)
                with ThreadPoolExecutor(self.jobs) as pool:
                    for b, tot in zip(todo, pool.map(lambda b: self.plan_one(b, ex[b]), todo)):
                        print(f"[bind]   stage {b} from {sorted({tuple(v) for v in ex[b].values()})}: "
                              + (f"total {tot}" if tot is not None else "FAILED"), flush=True)
                        if tot is None:
                            return False
                        maps[b] = ex[b]
                maps_f.write_text(json.dumps(maps))
                continue
            ex = sweep(seeds if p == 0 else ())
            todo = [b for b in range(n) if maps.get(b) != ex[b]]
            # a ripple: stage h alone is off the chain because stage h - 1's exit moved in the last pass, and a pass
            # moves it one stage on (h + 1 waits for h's new exit). The stages after h that take what h takes (a run of
            # identical blocks has one fixed point) are planned in this pass from the same exit, as --seed does in the
            # first pass; the next pass keeps those the true chain confirms
            heads = [h for h in todo if h >= 2 and h - 1 not in todo and h + 1 not in todo]
            if p and heads:
                auto = []
                for h in heads:
                    z = h
                    while z + 1 < n and z + 1 not in todo and n_in[z + 1] == n_in[h]:
                        z += 1
                    auto.append((h - 1, z))
                ex = sweep(auto)
                todo = [b for b in range(n) if maps.get(b) != ex[b]]
                print(f"[bind] pass {p}: ripple at {heads}: seeded {auto}", flush=True)
            print(f"[bind] pass {p}: {len(todo)} stages off the chain: {todo}", flush=True)
            if not todo:
                break
            if 0 in todo:                       # stage 0 first: its exit feeds stage 1's map
                tot = self.plan_one(0, ex[0])
                print(f"[bind]   stage 0 from {sorted({tuple(v) for v in ex[0].values()})}: "
                      + (f"total {tot}" if tot is not None else "FAILED"), flush=True)
                if tot is None:
                    return False
                maps[0] = ex[0]
                maps_f.write_text(json.dumps(maps))
                continue
            with ThreadPoolExecutor(self.jobs) as pool:
                for b, tot in zip(todo, pool.map(lambda b: self.plan_one(b, ex[b]), todo)):
                    print(f"[bind]   stage {b} from {sorted({tuple(v) for v in ex[b].values()})}: "
                          + (f"total {tot}" if tot is not None else "FAILED"), flush=True)
                    if tot is not None:
                        maps[b] = ex[b]
            maps_f.write_text(json.dumps(maps))
        else:
            print("[bind] did not converge", flush=True)
            return False
        tot = [json.load(open(self.D / f"block_{b}_placement.json"))["summary"]["total_bootstraps"] for b in range(n)]
        print(f"[bind] chain binds input by input: {n} stages, {sum(tot)} refreshes ({tot})", flush=True)
        return True

    def write_cmd(self, bound=True):
        """PLAN_CMD.txt (graph, tool, recipe, runtime= the capture's levers, configs= its approximation configs) and
        the plan's inputs beside it: the PCMM stamps, the runtime, the calibration (pcmm_calib.json), the rotation
        steps (plan_rot_steps.json), so a --plan run needs neither the capture nor the inputs' names."""
        from perseus.impl.magtape import write_plan_rot_steps
        for f in ("pcmm_stamps.json", "sana_runtime.json"):
            if (self.G / f).exists():
                subprocess.run(["cp", str(self.G / f), str(self.D)], check=True)
        cap = json.load(open(self.G / "sana_capture.json")) if (self.G / "sana_capture.json").exists() else {}
        if cap.get("pcmm_calib"):
            subprocess.run(["cp", cap["pcmm_calib"], str(self.D / "pcmm_calib.json")], check=True)
        rt = json.load(open(self.G / "sana_runtime.json"))
        tool = "python/perseus/examples/sana/make_plan.sh" + (" (bind_plan.py: PLAN_ENTRY_MAP)" if bound else "")
        with open(self.D / "PLAN_CMD.txt", "w") as f:
            f.write(f"graph={self.G}\ntool={tool}\n")
            f.write("recipe=" + " ".join(f"{k}={shlex.quote(v)}" for k, v in self.recipe.items()) + "\n")
            f.write("runtime=" + " ".join(f"{k}={shlex.quote(str(v))}" for k, v in sorted(rt.items())) + "\n")
            if cap.get("configs"):
                f.write(f"configs={cap['configs']}\n")
        write_plan_rot_steps(str(self.D))

    def compile_chain(self):
        """Every stage compiled from its capture graph against its predecessor's compiled exit (stage 0: `entry0`).
        Returns the first failing stage, or None."""
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        levers.runtime(str(self.D), [])
        from perseus import _pc
        from perseus.examples.bert.planned import plan_config
        from perseus.plan.placer.planner import _stage_outputs, stage_n_inputs, stage_outputs_ordered
        from .planned import program_digest
        cfg = plan_config(str(self.D), False)
        work = self.tmp / "compile"
        prev, t0 = None, time.time()
        e0 = entry0(self.G)
        for b in range(self.n):
            lab = f"block_{b}"
            (work / lab).mkdir(parents=True, exist_ok=True)
            if not (work / lab / "graph.json").exists():
                (work / lab / "graph.json").symlink_to(self.graph(b))
            ins = []
            made = set()
            for nd in json.load(open(self.graph(b)))["nodes"]:
                for x in nd["inputs"]:
                    if not x.startswith(_NOT_CT) and x not in made and x not in ins:
                        ins.append(x)
                made.add(nd["output"])
            k = stage_n_inputs(self.graph(b))
            names = [x for x in ins if x.startswith("anon_")][:1] + [f"pf_in{c}" for c in range(1, k)]
            if prev is None:
                metas_l = [tuple(e0.get(str(i), (0, 1))) for i in range(k)]      # unread positions: unused
            else:
                got = {nm: (int(lv), int(dg)) for nm, lv, dg, _s in prev[0].outputs}
                metas_l = [got[nm] for nm in prev[1]]
            outs = stage_outputs_ordered(self.graph(b), 2, self.n_out(b))[0] if b + 1 < self.n \
                else _stage_outputs(self.graph(b), 2)
            one = work / "_one"                     # Program.compile compiles every block of its dir: this one only
            shutil.rmtree(one, ignore_errors=True)
            one.mkdir()
            (one / lab).symlink_to(work / lab)
            plan1 = work / "_plan1"                 # ... and reads every placement of its plan dir: this stage's only
            shutil.rmtree(plan1, ignore_errors=True)
            plan1.mkdir()
            for f in self.D.iterdir():
                if not f.name.endswith("_placement.json") or f.name == f"{lab}_placement.json":
                    (plan1 / f.name).symlink_to(f.resolve())
            metas = {nm: metas_l[names.index(nm)] for nm in ins}
            try:
                p = _pc.Program.compile(str(one), str(plan1), config=cfg, blocks=[lab], live_out={lab: list(outs)},
                                        input_metas={lab: metas}, schedule=SCHEDULE)[lab]
            except Exception as e:                                                  # noqa: BLE001
                print(f"[bind] compile stage {b}: FAILED {str(e)[:400]}", flush=True)
                return b
            # kept for the runs (planned.SanaPlannedPrefill.prepare loads it when its recorded stage gives this digest)
            (self.D / "programs").mkdir(exist_ok=True)
            p.save(str(self.D / "programs" / f"{lab}.json"))
            (self.D / "programs" / f"{lab}.json.digest").write_text(program_digest(
                json.load(open(self.graph(b)))["nodes"], plan1 / f"{lab}_placement.json", outs, metas, SCHEDULE))
            prev = (p, outs)
        print(f"[bind] all {self.n} stages compile ({time.time() - t0:.0f}s)", flush=True)
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs")
    ap.add_argument("out", help="the plan's name under bootstrap_placements/")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--seed", default="0:0", help="a:z[,a:z...]: stages a+1..z first take stage a's exit")
    ap.add_argument("--warm", default=None, help="a bound plan (name or dir): the first pass plans every stage from "
                    "its converged entry maps (no first plans needed)")
    ap.add_argument("--write-only", action="store_true", help="PLAN_CMD.txt and the plan's inputs only (not bound)")
    a = ap.parse_args(argv)
    bd = Binder(Path(a.graphs).resolve(), a.out, a.jobs)
    if a.write_only:
        bd.write_cmd(bound=False)
        return 0
    if not bd.bind([tuple(int(v) for v in r.split(":")) for r in a.seed.split(",") if r], warm=a.warm):
        return 1
    bd.write_cmd()
    return 1 if bd.compile_chain() is not None else 0


if __name__ == "__main__":
    sys.exit(main())
