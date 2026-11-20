"""Smooth functional bootstrapping on the device (impl/fbs_fit.py): the trigonometric form (f folded into EvalMod's
cosine and sine, P(t) + s Q(t)) and the sine form as planned FBS finishes on n32 (twin capture -> dense plan ->
planned Program; examples/mathlib `fn` task with fbs=true), every output against float64 and the largest error
against the noise model's bound.

    tools/gpu.sh out/fbs_smooth.log .venv/bin/python -m pytest -m gpu python/tests/gpu/test_fbs_smooth.py -q -s
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _auto(tmp_path, task, sets):
    out = str(tmp_path / "rows.jsonl")
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "python"), PC_WORK_DIR=str(tmp_path))
    subprocess.run([sys.executable, "-m", "perseus.examples.mathlib.run", "--task", task, "--set", *sets,
                    "--auto", str(tmp_path / "run"), "--seeds", "0", "1", "--reps", "2", "--out", out],
                   check=True, env=env, cwd=REPO)
    return [r for r in map(json.loads, open(out)) if r["mode"] == "planned"]


@pytest.mark.parametrize("sets", [
    ["fn=log", "lo=1", "hi=8", "err=1e-4"],             # a positive range: the asin warp's branch point at its end
    ["fn=sigmoid", "lo=-8", "hi=8", "err=1e-3"],
    ["fn=exp", "lo=-8", "hi=0", "err=1e-3"],
])
def test_planned_smooth(tmp_path, sets):
    """2 x 32768 outputs per case: the shallowest form (a trigonometric one: two to four levels above today's sine
    form at these targets), every output within the noise model's bound (2^-128 per output, met at selection)."""
    from perseus.examples.mathlib import tasks as T
    t = T.make("fn", **dict(s.split("=") for s in sets), fbs=True)
    a = t.approx()
    rows = _auto(tmp_path, "fn", sets + ["fbs=true"])
    print(f"\n[fbs smooth] {sets}: {a.method} {dict(a.params)} bound {a.err:.2e}: " + " ".join(
        f"max {r['max_err']:.2e} rel {r['max_rel']:.2e} rms {r['rms_err']:.2e} {r['wall_s'] * 1e3:.1f} ms"
        for r in rows))
    assert a.method.startswith("fbs-trig") and len(rows) == 2
    key = "max_rel" if t.fn in ("exp", "rsqrt", "reciprocal") else "max_err"
    for r in rows:
        assert r["method_chosen"] == a.method
        assert r[key] <= a.err <= t.err
