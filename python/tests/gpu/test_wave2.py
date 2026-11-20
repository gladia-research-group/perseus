"""The wave-2 modules encrypted end to end on the device: one representative task per module through
examples/wave2/run.py --auto (twin capture -> dense plan -> planned Program with its bootstraps), seeds 0 and 1.

    tools/gpu.sh out/wave2.log .venv/bin/python -m pytest -m gpu python/tests/gpu/test_wave2.py -q -s

Data sets (breast_cancer) come from scikit-learn or $PERSEUS_DATA_DIR (examples/wave2/data.py); without either the
training case is skipped."""
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
    subprocess.run([sys.executable, "-m", "perseus.examples.wave2.run", "--task", task, "--set", *sets,
                    "--auto", str(tmp_path / "run"), "--seeds", "0", "1", "--reps", "1", "--out", out],
                   check=True, env=env, cwd=REPO)
    return [r for r in map(json.loads, open(out)) if r["mode"] == "planned"]


CASES = [
    # (task, sets, metric, bound, max bootstraps)
    ("solve", ["method=chebr", "kappa=100", "n=16", "B=128", "eps=1e-4"], "rel_err", 1e-3, 110),
    ("invsqrt", ["method=coupled", "n=8", "B=64", "kappa=100"], "whiten_err", 1e-3, 30),
    ("expm", ["n=8", "B=64", "norm=2"], "rel_err", 1e-3, 10),
    ("kmeans", ["N=1024", "K=4", "d=2", "method=ckks"], "agree_clear", -0.999, 40),
    ("train", ["data=breast_cancer", "loss=logistic"], "w_err", 1e-2, 30),
    ("signal", ["kind=xcorr", "method=coeff"], "err_max", 1e-4, 0),
    ("signal", ["kind=conv2d", "method=fft", "ksize=7"], "err_max", 1e-4, 1),
]


@pytest.mark.parametrize("task,sets,key,bound,bts", CASES)
def test_planned(tmp_path, task, sets, key, bound, bts):
    """The encrypted run against float64: metric < bound (a negative bound: metric >= -bound), within the
    bootstrap budget."""
    if task == "train":
        from perseus.examples.wave2 import data as DS
        if not DS.available("breast_cancer"):
            pytest.skip("breast_cancer: no scikit-learn / PERSEUS_DATA_DIR")
    rows = _auto(tmp_path, task, sets)
    print(f"\n[wave2] planned {task} {sets}: " + " ".join(f"{r[key]:.2e}/{r['bts']}bts/{r['wall_s']:.3f}s" for r in rows))
    ok = (lambda v: v >= -bound) if bound < 0 else (lambda v: v < bound)
    assert len(rows) == 2 and all(ok(r[key]) and r["bts"] <= bts for r in rows), rows
