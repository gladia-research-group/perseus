"""ResNet-20 / CIFAR-10 encrypted inference, gated against the float64 model (top-1 agreement, logit error).

    python -m perseus.examples.resnet.run --mirror --n 1000          # plaintext mirror, composite ReLU
    tools/gpu.sh <log> python -m perseus.examples.resnet.run --eager --debug-refresh --n 4

`--eager --debug-refresh` runs every op on the device as it is issued (no plan); the eager session has no bootstrap,
so a ciphertext at the level limit is refreshed by decrypt + re-encrypt: a check of the device ops, not an encrypted
run (that is the planned path).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

DEFAULT_CONFIGS = "configs/model/approximation/resnet20_cifar10/configs.json"


def relu_act(section):
    """(act(ops, cts, site), stages, {site: B}) from the configs.json "relu" section; the act reads inputs already
    divided by their B (model.ResNet20 with scales=B)."""
    from perseus.impl.relu import relu_config, relu_unit, unit_stages
    stages, B = relu_config(section)
    coefs = unit_stages(stages)

    def act(ops, xs, site, gain=1.0):
        return [relu_unit(ops, x, coefs, gain) for x in xs]
    return act, stages, B


def _mirror(a, sd, act, x, y, ref):
    from perseus.impl.ops import NumpyOps
    from .model import ResNet20
    m = ResNet20(sd, act, scales=a.scales)
    ops = NumpyOps()
    lg = np.stack([m.logits(m.forward(ops, m.encode(xi))) for xi in x])
    return lg


def _eager(a, sd, act, x, y, ref):
    from perseus import _pc
    from perseus.impl.ops import FheOps
    from .model import ResNet20
    m = ResNet20(sd, act, scales=a.scales)
    cfg = _pc.Config.preset(a.preset)
    t0 = time.time()
    keys = _pc.KeySource.generate(m.rotations(), layout=a.key_layout, seed=0)
    sess = _pc.Session(cfg, keys, pool=a.pool, pt_slots=a.pt_slots, debug_refresh=a.debug_refresh).complete_setup()
    ops = FheOps.from_session(sess)
    fhe = sess.fhe
    print(f"[resnet] session {fhe.slots} slots, {len(m.rotations())} rotation keys, setup {time.time() - t0:.1f}s, "
          f"level limit {fhe.level_limit()}", flush=True)
    lgs, walls = [], []
    for i, xi in enumerate(x):
        cts = [fhe.encrypt(v, 0) for v in m.encode(xi)]
        t0 = time.time()
        out = m.forward(ops, cts)
        vals = [np.real(fhe.decrypt_slots_complex(c)) for c in out]
        walls.append(time.time() - t0)
        lg = m.logits(vals)
        lgs.append(lg)
        print(f"[resnet] image {i}: wall {walls[-1]:.2f}s label {y[i]} ours {lg.argmax()} ref {ref[i].argmax()} "
              f"max|dlogit| {np.abs(lg - ref[i]).max():.3g}", flush=True)
    sess.close()
    print(f"[resnet] s/image median {np.median(walls):.2f}", flush=True)
    return np.stack(lgs)


def _twin(a, net):
    """The CPU twin session (perseus.impl.fakecap) and the model on it; `--capture` writes the stage graphs."""
    from perseus.examples.gpt2 import env
    from perseus.impl import fakecap
    from perseus.impl.layout import Dims
    from .impl import ResnetImpl
    over = dict(kv.split("=", 1) for kv in a.set)
    over.setdefault("AUTO_BTS_LEVEL", str(a.twin_limit))     # the twin's ceiling, stamped in the capture's Config too
    env.export_env(None, chain="n32", CKKS_COMPLEX="1", **over)
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
    # the reactive ceiling at the planner's (make_plan.sh MAX_LEVEL): a stage output the twin refreshed reactively
    # would carry that refresh's name, which the compiled program (reactive refreshes erased) does not produce
    _, inf = fakecap.make_session(Dims.gpt2(), set(net.rotations()), limit=a.twin_limit, bts_out=a.twin_bts_out)
    model = ResnetImpl(inf, net, core=fakecap.core)
    if a.capture:
        model.set_capture(a.capture)
        from perseus.impl.magtape import save_rot_steps
        save_rot_steps(a.capture, inf.fhe.loaded_rot_steps)
        inf.fhe.magnitude_suppressed = a.structure_only     # the planner reads the twin's magnitudes
    return model


def _fake(a, sd, act, x, y, ref):
    from .model import ResNet20
    model = _twin(a, ResNet20(sd, act, scales=a.scales, refresh=not a.one_stage, one_stage=a.one_stage))
    lgs = []
    for i, xi in enumerate(x[:1] if a.capture else x):
        t0 = time.time()
        xs, _, _ = model.prefill(xi)
        lgs.append(model.logits(xs))
        print(f"[resnet] twin image {i}: {time.time() - t0:.1f}s ours {lgs[-1].argmax()} ref {ref[i].argmax()} "
              f"max|dlogit| {np.abs(lgs[-1] - ref[i]).max():.3g}", flush=True)
    if a.capture:
        print(f"[resnet] captured {model.n_layers} stages -> {a.capture}", flush=True)
    return np.stack(lgs + [ref[i] for i in range(len(lgs), len(x))])


def _planned(a, sd, act, x, y, ref):
    from perseus import _pc
    from perseus.examples.bert.planned import plan_config
    from .impl import ResnetPlanned
    from .model import ResNet20
    model = _twin(a, ResNet20(sd, act, scales=a.scales, refresh=not a.one_stage, one_stage=a.one_stage))
    cfg = plan_config(a.plan)
    keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
    work = a.work_dir or os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), f"resnet_{os.getpid()}")
    run = ResnetPlanned(model, a.plan, cfg, keys, work_dir=work, pt_rows=a.pt_rows, pool=a.pool,
                        log=lambda m: print(m, flush=True))
    t0 = time.time()
    run.prepare(x[0])
    progs = [run.progs[lab] for lab in run.labels]
    print(f"[resnet] compiled {len(progs)} stages: record {run.record_s:.1f}s compile {run.compile_s:.1f}s; "
          f"{sum(p.n_bootstraps for p in progs)} bootstraps, registers {run.n_regs}", flush=True)
    if a.compile_only:
        return None
    run.setup(x[0])
    print(f"[resnet] planned setup {time.time() - t0:.1f}s; device free {run.fhe.mem_info()[0] / 2 ** 30:.2f} GB",
          flush=True)
    lgs, walls = [], []
    for i, xi in enumerate(x):
        vals = run.input_values(xi)
        t0 = time.time()
        regs = run.run(vals=vals)
        run.fhe.device_sync()
        walls.append(time.time() - t0)
        lgs.append(run.logits(regs))
        print(f"[resnet] image {i}: wall {walls[-1]:.3f}s label {y[i]} ours {lgs[-1].argmax()} ref {ref[i].argmax()} "
              f"max|dlogit| {np.abs(lgs[-1] - ref[i]).max():.3g}", flush=True)
    print(f"[resnet] s/image median {np.median(walls):.3f} (walls {' '.join(f'{w:.3f}' for w in walls)})", flush=True)
    run.close()
    return np.stack(lgs)


def main(argv=None, results=None):
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mirror", action="store_true", help="plaintext mirror (NumpyOps)")
    mode.add_argument("--eager", action="store_true", help="GPU, eager (needs --debug-refresh)")
    mode.add_argument("--fake", action="store_true", help="the CPU twin (perseus.impl.fakecap)")
    mode.add_argument("--plan", default=None, help="GPU, planned: the bootstrap_placements dir")
    ap.add_argument("--capture", default=None, help="--fake: write the stage graphs here")
    ap.add_argument("--one-stage", action="store_true", help="twin / plan: the network as one stage")
    ap.add_argument("--twin-limit", type=int, default=50, help="the twin's reactive refresh ceiling")
    ap.add_argument("--twin-bts-out", type=int, default=32, help="the twin's refresh landing (the route the run takes)")
    ap.add_argument("--structure-only", action="store_true", help="--capture without magnitudes")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--pt-rows", type=int, default=28)
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--configs", default=None)
    ap.add_argument("--exact-relu", action="store_true", help="the exact max (mirror only)")
    ap.add_argument("--preset", default="n32_gpt2_decode")
    ap.add_argument("--debug-refresh", action="store_true",
                    help="eager: refresh by decrypt + re-encrypt (the eager session has no bootstrap; NOT encrypted end to end)")
    ap.add_argument("--key-layout", choices=("dense", "packed"), default="packed")
    ap.add_argument("--pool", type=int, default=256)
    ap.add_argument("--pt-slots", type=int, default=8)
    a = ap.parse_args(argv)
    from perseus._paths import resolve
    from . import weights as R
    from .model import relu_exact
    sd = R.state_dict()
    x, y = R.cifar10(a.n, a.start)
    a.scales = None
    if a.exact_relu:
        act, ref_relu = relu_exact, None
    else:
        from perseus.calibrate.approximations.relu import relu_apply
        import torch
        section = json.load(open(resolve(a.configs or DEFAULT_CONFIGS)))["relu"]
        act, stages, B = relu_act(section)
        a.scales = B
        ref_relu = lambda t, site: torch.from_numpy(relu_apply(t.numpy(), B[site], stages))
    exact = R.reference(sd, x)
    ref = R.reference(sd, x, ref_relu)          # what the encrypted run approximates
    if a.plan:
        a.plan = resolve(a.plan)
    lg = (_mirror if a.mirror else _eager if a.eager else _fake if a.fake else _planned)(a, sd, act, x, y, ref)
    if lg is None:
        return 0
    err = np.abs(lg - ref).max()
    agree = int((lg.argmax(1) == ref.argmax(1)).sum())
    print(f"[resnet] SUMMARY n={a.n} top1={int((lg.argmax(1) == y).sum())}/{a.n} (exact model "
          f"{int((exact.argmax(1) == y).sum())}/{a.n}) agree-with-ref={agree}/{a.n} max|dlogit|={err:.3g}", flush=True)
    if results is not None:
        results({"n": a.n, "top1": int((lg.argmax(1) == y).sum()), "exact_top1": int((exact.argmax(1) == y).sum()),
                 "agree": agree, "max_dlogit": float(err)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
