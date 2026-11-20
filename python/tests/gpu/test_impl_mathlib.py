"""impl.math / impl.stats / impl.sort on the device: a few functions on the eager session (debug refresh: the device's
arithmetic noise, no bootstrap), and representative tasks end to end (twin capture -> dense plan -> planned Program
with its bootstraps and, for the TFHE comparisons, the session's scheme switch), against float64.

    tools/gpu.sh out/mathlib.log .venv/bin/python -m pytest -m gpu python/tests/gpu/test_impl_mathlib.py -q -s
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(scope="module")
def eager():
    from perseus.impl.ops import FheOps
    sess = _pc.Session(_pc.Config.preset("n32_gpt2_decode"), _pc.KeySource.generate([1, 2, 4], seed=3), pool=64,
                       pt_slots=8, debug_refresh=True).complete_setup()
    yield sess, FheOps.from_session(sess)
    sess.close()


@pytest.mark.parametrize("fn,lo,hi,err,kw", [
    ("exp", -8.0, 0.0, 1e-4, {}), ("log", 1.0, 100.0, 1e-4, {}), ("rsqrt", 0.0613, 0.9735, 1e-5, {}),
    ("reciprocal", 0.231, 27.55, 1e-4, {}), ("tanh", -4.0, 4.0, 1e-4, {}), ("sqrt", 0.1, 10.0, 1e-4, {}),
])
def test_eager_functions(eager, fn, lo, hi, err, kw):
    """The encrypted value against the float64 approximation (the CKKS noise alone) and the function."""
    from perseus.impl import math as M
    sess, ops = eager
    a = M.approx(fn, lo, hi, err, **kw)
    x = np.random.default_rng(0).uniform(lo, hi, sess.fhe.slots)
    y = np.real(np.asarray(sess.fhe.decrypt_slots_complex(a(ops, sess.fhe.encrypt(x, 0)))))
    noise = float(np.abs(y - a.numpy(x)).max())
    scale = float(np.abs(a.numpy(x)).max())
    print(f"\n[mathlib] eager {fn} [{lo}, {hi}] {a.method} depth {a.depth}: CKKS noise {noise:.2e} (|y| <= {scale:.3g})")
    assert noise < 1e-5 * max(1.0, scale)


def _auto(tmp_path, task, sets, seeds=("0", "1")):
    """run.py --auto: twin capture -> plan -> planned run on the device."""
    out = str(tmp_path / "rows.jsonl")
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "python"), PC_WORK_DIR=str(tmp_path))
    subprocess.run([sys.executable, "-m", "perseus.examples.mathlib.run", "--task", task, "--set", *sets,
                    "--auto", str(tmp_path / "run"), "--seeds", *seeds, "--reps", "2", "--out", out],
                   check=True, env=env, cwd=REPO)
    return [r for r in map(json.loads, open(out)) if r["mode"] == "planned"]


@pytest.mark.parametrize("task,sets,key,tol", [
    ("fn", ["fn=exp", "lo=-8", "hi=0", "err=1e-4"], "max_rel", 2e-4),
    ("fn", ["fn=log", "lo=0.01", "hi=100", "err=5e-2"], "max_err", 5e-2),     # refreshed inside: noise-held
    ("fn", ["fn=relu", "lo=-1", "hi=1", "err=1e-3", "method=ckks"], "max_err", 2e-3),
    ("fn", ["fn=relu", "lo=-1", "hi=1", "err=1e-3", "method=tfhe"], "rms_err", 1e-3),
    ("stats", ["kind=zscore", "n=256", "B=128"], "max_err", 5e-3),
    ("stats", ["kind=softmax", "n=256", "B=128", "method=tfhe"], "max_err", 2e-3),
    ("stats", ["kind=logsumexp", "n=256", "B=128", "method=tfhe"], "max_err", 3e-3),    # 3.9e-2 when u was refreshed
    ("stats", ["kind=cov", "m=256", "d=16", "B=4"], "max_err", 1e-4),
    ("sort", ["kind=rank", "n=16", "method=ckks"], "exact", None),
    ("sort", ["kind=rank", "n=16", "method=tfhe"], "exact", None),
    ("sort", ["kind=bitonic", "n=64", "method=tfhe"], "exact", None),
    ("sort", ["kind=threshold", "n=256", "k=8", "method=tfhe", "levels=2"], "mask_exact", None),
    ("sort", ["kind=threshold", "n=4096", "k=8", "method=ckks"], "mask_exact", None),       # 1e19 without ladder margins
    ("pipeline", ["n=16", "B=4", "method=tfhe"], "topk_exact", None),
])
def test_planned(tmp_path, task, sets, key, tol):
    rows = _auto(tmp_path, task, sets)
    print(f"\n[mathlib] planned {task} {sets}: " + " ".join(f"{r[key]:.3g}/{r['wall_s']:.3f}s/{r['bts']}bts"
                                                          for r in rows))
    assert rows
    for r in rows:
        assert (r[key] == 1.0) if tol is None else (r[key] < tol), r
