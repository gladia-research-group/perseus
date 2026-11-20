"""impl.linalg / impl.fourier inside a recorded stage, on the CPU: the twin capture, the planner and the compiled
program's level check (examples/linalg/run.py --auto --compile-only), with no workaround in the stage code:

* a stage whose first op is a product of its fresh input plans and compiles (the planner seeded such an input one
  unit richer than it is: the lazily captured product's output level minus its cost);
* a refresh the code tags sparse compiles on the route it took (the twin lands it where that route of the compiled
  runtime lands: FAKECAP_LANDINGS=runtime);
* a Fourier transform and a least-squares solve in one stage compile, the transform's depth chosen by the planner's
  level budget (fourier.Budget), its rotation keys the ones the stage declares (keyset).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from perseus import _pc
from perseus.examples.linalg import tasks as T
from perseus.impl import fourier as F
from perseus.impl import keyset
from perseus.impl import linalg as la

S = 1 << 15


def _auto(tmp_path, task, sets, sparse=""):
    """run.py --auto --compile-only; returns the compiled bootstrap count and the plan's summary."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    d = tmp_path / task
    r = subprocess.run([sys.executable, "-m", "perseus.examples.linalg.run", "--task", task, "--set", *sets,
                        "--auto", str(d), "--compile-only", "--sparse", sparse],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    line = [x for x in r.stdout.splitlines() if x.startswith("[linalg] compiled:")][-1]
    plan = json.load(open(d / "plan" / "block_0_placement.json"))
    return int(line.split()[2]), plan


def _node(op, ins, lvls, out, lvl, deg, **kw):
    return {"op_type": op, "inputs": ins, "input_levels": lvls, "output": out, "output_level": lvl,
            "output_noise_level": deg, "step": "s", "output_max_abs": 0.5, "output_mean": 0.0, "output_max_dev": 0.5,
            **kw}


def test_stage_starting_with_a_product_compiles(tmp_path):
    """x (fresh, 32) -> y = 0.5 x (captured pending: 32, degree 2) read by two rotations and a product. The planner
    realizes y once (a shared deg-2 operand) and must seed x at its captured 32: seeded from y's level minus the
    product's cost (30) it pinned y at 32 and the compiled program, which starts x at 32, failed the level check
    ("v_1 expected 32 actual 34")."""
    from perseus.examples.bert.planned import plan_config
    from perseus._paths import ASSETS, PKG
    g = tmp_path / "graph"
    (g / "block_0").mkdir(parents=True)
    nodes = [
        _node("mult", ["anon_1", "const(0.5)"], [32, -1], "v_1", 32, 2, args={"scalar": 0.5}),
        _node("rotate", ["v_1", "rot(1)"], [32, -1], "v_2", 32, 2, args={"steps": [1]}),
        _node("rotate", ["v_1", "rot(2)"], [32, -1], "v_3", 32, 2, args={"steps": [2]}),
        _node("add", ["v_2", "v_3"], [32, 32], "v_4", 32, 2),
        _node("mult", ["v_4", "v_4"], [32, 32], "v_5", 32, 2),
    ]
    (g / "block_0" / "graph.json").write_text(json.dumps({"version": 2, "nodes": nodes}))
    plan = tmp_path / "plan"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PLAN_SPARSE_SLOTS="", PYTHON=sys.executable)
    r = subprocess.run(["bash", str(PKG / "examples" / "gpt2" / "make_plan.sh"), str(g),
                        os.path.relpath(plan, ASSETS / "bootstrap_placements")], env=env, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    s = json.load(open(plan / "block_0_placement.json"))["summary"]
    assert (s["final_named_levels"]["v_1"], s["final_named_degs"]["v_1"]) == (34, 1)
    progs = _pc.Program.compile(str(g), str(plan), config=plan_config(str(plan)), blocks=["block_0"],
                                live_out={"block_0": ["v_5"]}, input_metas={"block_0": {"anon_1": (32, 1)}})
    assert progs["block_0"].n_bootstraps == 0


def test_qr_polar_starts_with_a_product(tmp_path):
    """polar_ns' first op is a scalar product of the fresh input (it failed the level check before the fix)."""
    n_bts, _ = _auto(tmp_path, "qr", ["m=64", "n=8", "kappa=10", "method=polarns"])
    assert n_bts > 0


def test_sparse_tagged_refresh_compiles(tmp_path):
    """inverse_fast's refreshed iterate tagged with its period: the twin routes it sparse, the compile takes the
    recorded route (no `routes_built` needed) and the landings agree."""
    n_bts, plan = _auto(tmp_path, "inverse", ["n=8", "kappa=100", "fast=true", "batch=1", "sparse=true"],
                        sparse="512,1")
    assert n_bts == 11


def test_runtime_capture_landings_reproduce_the_presets():
    """FAKECAP_LANDINGS=runtime derives the twin's table from the compiled runtime's routes; on the shipped presets it
    is the table their captures were made with."""
    for p in ("n32_gpt2_decode", "n32_gpt2_prefill"):
        cfg = _pc.Config.preset(p)
        got = dict(x.split(":") for x in _pc.runtime_capture_landings(cfg).split(","))
        want = dict(x.split(":") for x in cfg["quirks.fakecap_landings"].split(","))
        assert {k: got[k] for k in want} == want, (p, got, want)


def test_fourier_depth_follows_the_budget():
    """plan(levels=None) prices a level at bts_cost / window: a tighter window (dearer levels) never deepens the
    transform, a cheaper one never makes it shallower; the planner's default window is (48 - 36) / 2."""
    assert F.Budget.planner().window == 6
    lay = F.Layout(1024, S, S // 1024, "stride")
    depth = [F.dft(1024, layout=lay, budget=F.Budget(window=w)).levels for w in (1, 3, 6, 12, 48)]
    assert depth == sorted(depth) and depth[0] < depth[-1], depth
    assert F.dft(1024, layout=lay, budget=F.Budget(window=48, max_levels=3)).levels <= 3


def test_stage_rotations_and_key_preference():
    """keyset.stage_rotations is the transform's own key list; under keyset.prefer a second transform reuses the
    first one's keys where an equally cheap split exists."""
    lay = F.Layout(256, S, S // 256, "stride")
    t1 = F.dft(256, 3, layout=lay, order="bitrev")
    x = np.zeros(S, complex)
    assert keyset.stage_rotations(lambda ops, xs: [t1.apply(ops, xs[0])], [x]) == sorted(
        k % S for k in t1.rot_steps())
    plain = F.dct2(256, 4, layout=lay)
    with keyset.prefer(t1.rot_steps()):
        pref = F.dct2(256, 4, layout=lay)
    new = lambda t: len({k % S for k in t.rot_steps()} - {k % S for k in t1.rot_steps()})
    assert new(pref) <= new(plain) and pref.rotations() == plain.rotations()


def test_fourier_and_lstsq_in_one_stage(tmp_path):
    """A DCT-II on the rows of A, pinv least squares on the features and a DCT-III back, recorded as one stage,
    planned and compiled: the solve's refreshes, none in the transforms (1-2 levels each)."""
    t = T.DctLstsq()
    m = t.metrics(t.mirror())
    assert m["rel_err"] < 1e-8 and m["levels_dct"] <= 2
    n_bts, _ = _auto(tmp_path, "dctls", [])
    assert n_bts == 11


def test_fourier_stage_compiles(tmp_path):
    n_bts, plan = _auto(tmp_path, "fourier", ["kind=dft", "n=1024"])
    assert n_bts == 0


def test_deliberate_refresh_past_the_default_band_is_priced(tmp_path):
    """A code refresh of a value of size 256 (a softmax sum) is far past the EvalMod band of the runtime's default CF
    (6: the device returned ~1e17): the planner gives it a CF from its captured magnitude; a unit-size refresh keeps
    the default (no CF in the plan), as every shipped plan's code refreshes do."""
    from perseus._paths import ASSETS, PKG
    g = tmp_path / "graph"
    (g / "block_0").mkdir(parents=True)
    route = {"route": {"slots": 0, "stc_first": False}}

    def big(n, mag):
        return {**n, "output_max_abs": mag, "output_mean": mag, "output_max_dev": 0.0, "output_max_coeff": mag}
    nodes = [
        big(_node("mult", ["anon_1", "const(256)"], [32, -1], "v_1", 32, 2, args={"scalar": 256.0}), 256.0),
        big(_node("deliberate_bootstrap", ["v_1"], [34], "v_2", 32, 2, args=route), 256.0),
        _node("mult", ["v_2", "const(0.00390625)"], [32, -1], "v_3", 32, 2, args={"scalar": 1.0 / 256}),
        _node("deliberate_bootstrap", ["v_3"], [34], "v_4", 32, 2, args=route),
        _node("mult", ["v_4", "v_4"], [32, 32], "v_5", 32, 2),
    ]
    (g / "block_0" / "graph.json").write_text(json.dumps({"version": 2, "nodes": nodes}))
    plan = tmp_path / "plan"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PLAN_SPARSE_SLOTS="", PYTHON=sys.executable)
    r = subprocess.run(["bash", str(PKG / "examples" / "gpt2" / "make_plan.sh"), str(g),
                        os.path.relpath(plan, ASSETS / "bootstrap_placements")], env=env, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    cf = json.load(open(plan / "block_0_placement.json"))["correction_factor"]
    assert cf.get("v_2", 0) >= 14 and "v_4" not in cf, cf
