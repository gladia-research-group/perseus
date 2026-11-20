"""CKKS functional bootstrapping (impl/fbs.py) on the device: table and sign lookups as planned FBS finishes on n32
(twin capture -> dense plan -> planned Program; examples/mathlib `lut` task), every output against the table, and the
largest error against the noise model's bound.

    tools/gpu.sh out/fbs.log .venv/bin/python -m pytest -m gpu python/tests/gpu/test_fbs.py -q -s
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


def _auto(tmp_path, sets):
    out = str(tmp_path / "rows.jsonl")
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "python"), PC_WORK_DIR=str(tmp_path))
    subprocess.run([sys.executable, "-m", "perseus.examples.mathlib.run", "--task", "lut", "--set", *sets,
                    "--auto", str(tmp_path / "run"), "--seeds", "0", "1", "--reps", "2", "--out", out],
                   check=True, env=env, cwd=REPO)
    return [r for r in map(json.loads, open(out)) if r["mode"] == "planned"]


@pytest.mark.parametrize("sets,name", [
    (["kind=table", "table=sign", "t=16"], "lut.sign.16"),
    (["kind=table", "table=rand", "t=64"], "lut.rand.64"),
    (["kind=table", "table=step", "t=256"], "lut.step.256"),
    (["kind=sign", "gap=0.1"], "lut.sign01.g0.1"),
])
def test_planned_lut(tmp_path, sets, name):
    """2 x 32768 lookups per case: none wrong (an error past half the output spacing), the largest error within the
    model's bound at 2^-40 per output (deviation + z sigma, the default input noise), p_fail per ciphertext < 2^-128."""
    from perseus.examples.mathlib import tasks as T
    from perseus.impl import fbs
    T.make("lut", **dict(s.split("=") for s in sets))        # registers the function in this process
    bound = fbs.error_bound(name, log2_tail=-40.0)["bound"]
    if name.startswith("lut.sign01"):            # the error is read against the step: + the fit's own
        bound += fbs.sign_error(name, 0.1)
    rows = _auto(tmp_path, sets)
    print(f"\n[fbs] {sets}: " + " ".join(f"wrong {r['wrong']} max {r['max_err']:.2e} rms {r['rms_err']:.2e} "
                                         f"{r['wall_s'] * 1e3:.1f} ms/{r['bts']} bts" for r in rows)
          + f"; bound {bound:.2e}, p_fail/ct 2^{rows[0]['log2_pfail_ct']:.1f}")
    assert len(rows) == 2
    for r in rows:
        assert r["wrong"] == 0
        assert r["max_err"] < bound
        assert r["log2_pfail_ct"] < -128.0
