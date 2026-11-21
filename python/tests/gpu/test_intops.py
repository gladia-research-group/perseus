"""Chained-lookup integer arithmetic (impl/intops.py) on the device: 16 / 32-bit add, compare, mux and xor in radix-4
digits as planned FBS chains on n32 (examples/mathlib `intop` task: capture -> dense plan -> planned Program), no
plain refresh between lookups, every output rounding to the exact result, p_fail per lookup < 2^-128.

    tools/gpu.sh out/intops.log .venv/bin/python -m pytest -m gpu python/tests/gpu/test_intops.py -q -s
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


@pytest.mark.parametrize("sets", [
    ["op=add", "bits=16", "r=2", "table=k28d1"],
    ["op=cmp", "bits=32", "r=2", "table=k32d3"],
    ["op=mux", "bits=16", "r=2", "table=k28d1"],
    ["op=xor", "bits=16", "r=4", "table=k28d1"],
    ["op=add", "bits=32", "layout=tfhe", "words=16"],     # the native TFHE ALU (impl/tfhe_alu.py)
    ["op=cmp", "bits=32", "layout=tfhe", "words=16"],
])
def test_planned_intop(tmp_path, sets):
    out = str(tmp_path / "rows.jsonl")
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "python"), PC_WORK_DIR=str(tmp_path))
    subprocess.run([sys.executable, "-m", "perseus.examples.mathlib.run", "--task", "intop", "--set", *sets,
                    "--auto", str(tmp_path / "run"), "--seeds", "0", "--reps", "2", "--out", out],
                   check=True, env=env, cwd=REPO)
    rows = [r for r in map(json.loads, open(out)) if r["mode"] == "planned"]
    print(f"\n[intop] {sets}: " + " ".join(f"wrong {r['wrong']} dev {r['max_dev']:.1e} {r['wall_s'] * 1e3:.1f} ms "
                                          f"bts {r['bts']} p_fail 2^{r['log2_pfail_lookup_ct']:.1f}" for r in rows))
    assert len(rows) == 1
    r = rows[0]
    assert r["wrong"] == 0 and r["max_dev"] < 0.05
    assert r["bts"] == 0                         # lookups chain with no plain refresh
    assert r["log2_pfail_lookup_ct"] < -128.0
