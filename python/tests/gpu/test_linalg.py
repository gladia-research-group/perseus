"""impl/linalg.py on the device: the face moves, the product and the consolidation on the eager session against
numpy, Newton-Schulz on the eager session (debug refresh: the device's arithmetic noise, no bootstrap), and the
encrypted inverse end to end (twin capture -> dense plan -> planned Program with its bootstraps).

    tools/gpu.sh out/linalg.log .venv/bin/python -m pytest -m gpu python/tests/gpu/test_linalg.py -q -s
"""
from __future__ import annotations

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
    from perseus.examples.linalg import tasks as T
    from perseus.impl.ops import FheOps
    steps = sorted(set(T.Inverse(n=8, kappa=100).rotations()) | set(T.Inverse(n=16, kappa=100).rotations()))
    keys = _pc.KeySource.generate(steps, seed=3)
    sess = _pc.Session(_pc.Config.preset("n32_gpt2_decode"), keys, pool=128, pt_slots=8,
                       debug_refresh=True).complete_setup()
    yield sess, FheOps.from_session(sess)
    sess.close()


def _dec(sess, c):
    return np.real(np.asarray(sess.fhe.decrypt_slots_complex(c)))


@pytest.mark.parametrize("n", [8, 16])
def test_faces_product_consolidate(eager, n):
    from perseus.impl import linalg as la
    from perseus.impl.ops import NumpyOps
    sess, ops = eager
    rng = np.random.default_rng(n)
    cube = la.full_cube(n, n, sess.fhe.slots)
    lo, ll, lr = la.O(cube, (n, n)), la.L(cube, (n, n)), la.R(cube, (n, n))
    X, Y = rng.uniform(-1, 1, (n, n)), rng.uniform(-1, 1, (n, n))
    cx, cy = sess.fhe.encrypt(la.encode(X, lo), 0), sess.fhe.encrypt(la.encode(Y, lo), 0)
    xl, yr = la.relayout(ops, cx, lo, ll), la.relayout(ops, cy, lo, lr)
    npo = NumpyOps()
    assert np.abs(_dec(sess, xl) - la.relayout(npo, la.encode(X, lo), lo, ll)).max() < 1e-8
    z = la.mm(ops, xl, yr, cube)
    assert np.abs(_dec(sess, z) - la.encode(X @ Y, lo)).max() < 1e-7 * n
    c = la.consolidate(ops, z, lo)
    d = _dec(sess, c)
    assert np.abs(d - la.encode(la.decode(d, lo), lo)).max() < 1e-8


def test_inverse_eager(eager):
    from perseus.examples.linalg import tasks as T
    sess, ops = eager
    t = T.Inverse(n=8, kappa=100)
    out = t.run(ops, [sess.fhe.encrypt(v, 0) for v in t.encode()])
    m = t.metrics(t.outputs([_dec(sess, c) for c in out]))
    print(f"\n[linalg] eager NS n=8 kappa=100: {m}")
    assert m["rel_err"] < 1e-7


@pytest.mark.parametrize("sets,bts,tol", [
    (["n=8", "kappa=100"], 17, 1e-4),                                   # inverse(): plain schedule, one matrix
    (["n=8", "kappa=100", "fast=true", "batch=64"], 11, 1e-3),        # inverse_fast(): 64 matrices per ciphertext
])
def test_inverse_planned(tmp_path, sets, bts, tol):
    """The encrypted run: one refresh per Newton-Schulz step, consolidated (impl/linalg.consolidate)."""
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "python"), CUDA_VISIBLE_DEVICES="")
    graph = str(tmp_path / "graph")
    out = str(tmp_path / "rows.jsonl")
    sets = ["--task", "inverse", "--set", *sets]
    subprocess.run([py, "-m", "perseus.examples.linalg.run", *sets, "--fake", "--capture", graph], check=True, env=env,
                   cwd=REPO)
    name = os.path.relpath(str(tmp_path / "plan"), os.path.join(REPO, "bootstrap_placements"))
    subprocess.run(["bash", "python/perseus/examples/gpt2/make_plan.sh", graph, name], check=True, cwd=REPO,
                   env=dict(env, PLAN_SPARSE_SLOTS="", PYTHON=py))
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    env["PC_WORK_DIR"] = str(tmp_path)
    subprocess.run([py, "-m", "perseus.examples.linalg.run", *sets, "--plan", str(tmp_path / "plan"), "--seeds", "0",
                    "1", "--reps", "1", "--out", out], check=True, env=env, cwd=REPO)
    import json
    rows = [json.loads(line) for line in open(out)]
    print(f"\n[linalg] planned {sets}: " + " ".join(f"{r['rel_err']:.2e}/{r['wall_s']:.3f}s" for r in rows))
    assert all(r["rel_err"] < tol and r["bts"] == bts for r in rows), rows
