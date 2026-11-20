"""sort.step's FBS sign backend on the device: rank and top-k as planned runs (examples/mathlib `sort` task,
method=fbs), exact like the TFHE / CKKS backends.

    tools/gpu.sh out/sort_fbs.log .venv/bin/python -m pytest -m gpu python/tests/gpu/test_sort_fbs.py -q -s
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


@pytest.mark.parametrize("kind", ["rank", "topk"])
def test_planned_sort_fbs(tmp_path, kind):
    out = str(tmp_path / "rows.jsonl")
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "python"), PC_WORK_DIR=str(tmp_path))
    subprocess.run([sys.executable, "-m", "perseus.examples.mathlib.run", "--task", "sort", "--set", f"kind={kind}",
                    "method=fbs", "n=16", "--auto", str(tmp_path / "run"), "--seeds", "0", "--reps", "2", "--out", out],
                   check=True, env=env, cwd=REPO)
    rows = [r for r in map(json.loads, open(out)) if r["mode"] == "planned"]
    print(f"\n[fbs] sort {kind}: " + " ".join(f"exact {r['exact']} max {r['max_err']:.1e} {r['wall_s'] * 1e3:.1f} ms"
                                           for r in rows))
    assert len(rows) == 1 and rows[0]["exact"] == 1.0
