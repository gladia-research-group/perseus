"""Encrypted linear algebra (perseus.impl.linalg) on the four back ends, gated against float64.

    python -m perseus.examples.linalg.run --task inverse --set n=16 kappa=100 --mirror          # plaintext mirror
    python -m perseus.examples.linalg.run --task inverse --set n=16 --fake --capture graphs/la   # CPU twin (+ capture)
    python -m perseus.examples.linalg.run --task inverse --set n=16 --eager                     # device, debug refresh
    python -m perseus.examples.linalg.run --task inverse --set n=16 --plan bootstrap_placements/la_inv16

`--eager` runs every op on the device as issued; the eager session has no bootstrap, so a ciphertext at the level
limit is refreshed by decrypt + re-encrypt (debug_refresh): the device's arithmetic noise without the bootstrap's.
`--plan` is the encrypted run: the plan's bootstraps on the device (the twin records the stage, the program compiles
against the plan). One stage per task; `--seeds` reruns the same program on other instances (same schedule).
Prints one JSON line per run (`[linalg] RESULT {...}`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from . import tasks as T


class TaskNet:
    """The task as the resnet driver's `net` (one stage)."""

    def __init__(self, task):
        self.task = task

    def stages(self):
        return [(self.task.name, lambda ops, xs: self.task.run(ops, xs))]

    def encode(self, _x=None):
        return self.task.encode()

    def logits(self, vals):
        return self.task.outputs(vals)

    def rotations(self):
        return self.task.rotations()


def _parse_set(items):
    kw = {}
    for it in items:
        k, v = it.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                continue
        else:
            v = {"true": True, "false": False}.get(v.lower(), v)
        kw[k] = v
    return kw


def _report(a, task, out, mode, extra):
    met = task.metrics(out)
    ref = task.metrics(task.mirror()) if mode != "mirror" else met
    row = {"task": task.name, "mode": mode, "tag": a.tag, **task.describe(), **{k: v for k, v in met.items()},
           **{"mirror_" + k: v for k, v in ref.items()}, **extra}
    print("[linalg] RESULT " + json.dumps(row, default=float), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")
    return row


def _twin(a, net):
    from perseus.examples.gpt2 import env
    from perseus.impl import fakecap
    from perseus.impl.layout import Dims
    from perseus.examples.resnet.impl import ResnetImpl
    over = dict(kv.split("=", 1) for kv in a.env)
    if a.plan:
        from perseus.config import _plan_cmd, _runtime_line
        line = _plan_cmd(a.plan).get("runtime")
        over = {**(_runtime_line(line) if line else {}), **over}
    over.setdefault("AUTO_BTS_LEVEL", str(a.twin_limit))
    env.export_env(None, chain="n32", CKKS_COMPLEX="1", **over)
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
    _, inf = fakecap.make_session(Dims.gpt2(), set(net.rotations()), limit=a.twin_limit, bts_out=a.twin_bts_out)
    model = ResnetImpl(inf, net, core=fakecap.core)
    if a.capture:
        model.set_capture(a.capture)
        from perseus.impl.magtape import save_rot_steps
        save_rot_steps(a.capture, inf.fhe.loaded_rot_steps)
    return model


def run_mirror(a, task):
    t0 = time.time()
    out = task.mirror()
    return _report(a, task, out, "mirror", {"wall_s": time.time() - t0})


def run_fake(a, task):
    net = TaskNet(task)
    model = _twin(a, net)
    t0 = time.time()
    xs, _, _ = model.prefill(np.zeros(1))
    vals = [np.real(model.rt.ops.decrypt_slots_complex(c)) for c in xs]
    fhe = model.rt.ops.fhe
    return _report(a, task, task.outputs(vals), "fake", {"wall_s": time.time() - t0,
                                                         "bts_reactive": int(getattr(fhe, "n_reactive", 0)),
                                                         "bts": int(getattr(fhe, "n_bootstraps", 0))})


def run_eager(a, task):
    from perseus import _pc
    from perseus.impl.ops import FheOps
    cfg = _pc.Config.preset(a.preset)
    t0 = time.time()
    keys = _pc.KeySource.generate(task.rotations(), layout=a.key_layout, seed=a.key_seed)
    sess = _pc.Session(cfg, keys, pool=a.pool or 256, pt_slots=8, debug_refresh=True).complete_setup()
    ops = FheOps.from_session(sess)
    fhe = sess.fhe
    setup = time.time() - t0
    rows = []
    for seed in a.seeds:
        task = T.make(a.task, **{**a.kw, "seed": seed})
        cts = [fhe.encrypt(v, 0) for v in task.encode()]
        t0 = time.time()
        out = task.run(ops, cts)
        ops.device_sync()
        wall = time.time() - t0
        vals = [np.real(fhe.decrypt_slots_complex(c)) for c in out]
        rows.append(_report(a, task, task.outputs(vals), "eager", {"wall_s": wall, "setup_s": setup,
                                                                    "n_keys": len(task.rotations())}))
    sess.close()
    return rows


def run_planned(a, task):
    from perseus import _pc
    from perseus.examples.bert.planned import plan_config
    from perseus.examples.resnet.impl import ResnetPlanned
    net = TaskNet(task)
    model = _twin(a, net)
    cfg = plan_config(a.plan)
    keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
    work = a.work_dir or os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), f"linalg_{os.getpid()}")
    run = ResnetPlanned(model, a.plan, cfg, keys, work_dir=work, pool=a.pool or 32,
                        log=lambda m: print(m, flush=True))
    t0 = time.time()
    run.prepare(np.zeros(1))
    progs = [run.progs[lab] for lab in run.labels]
    n_bts = sum(p.n_bootstraps for p in progs)
    run.setup(np.zeros(1))
    setup = time.time() - t0
    rows = []
    for seed in a.seeds:
        t_ = T.make(a.task, **{**a.kw, "seed": seed})
        for k in ("s",):              # values the recorded run fixed (the program is the first instance's)
            if hasattr(task, k):
                setattr(t_, k, getattr(task, k))
        net.task = t_
        vals_in = t_.encode()
        walls = []
        for rep in range(a.reps):
            t0 = time.time()
            regs = run.run(vals=vals_in)
            run.fhe.device_sync()
            walls.append(time.time() - t0)
        vals = [np.real(v) for v in run.decrypt(regs)]
        rows.append(_report(a, t_, t_.outputs(vals), "planned",
                            {"wall_s": float(np.median(walls)), "walls": walls, "setup_s": setup, "bts": n_bts}))
    run.close()
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="inverse", choices=sorted(T.TASKS))
    ap.add_argument("--set", nargs="*", default=[], help="task fields k=v")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--mirror", action="store_true")
    mode.add_argument("--fake", action="store_true")
    mode.add_argument("--eager", action="store_true")
    mode.add_argument("--plan", default=None)
    ap.add_argument("--capture", default=None)
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--twin-limit", type=int, default=50)
    ap.add_argument("--twin-bts-out", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--pool", type=int, default=None)
    ap.add_argument("--preset", default="n32_gpt2_decode")
    ap.add_argument("--key-layout", choices=("dense", "packed"), default="packed")
    ap.add_argument("--key-seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="append the JSON rows here")
    ap.add_argument("--tag", default=None, help="a label carried into the rows")
    a = ap.parse_args(argv)
    a.kw = _parse_set(a.set)
    task = T.make(a.task, **{**a.kw, "seed": a.seeds[0]})
    if a.mirror:
        run_mirror(a, task)
    elif a.fake:
        run_fake(a, task)
    elif a.eager:
        run_eager(a, task)
    elif a.plan:
        run_planned(a, task)
    else:
        ap.error("pick a mode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
