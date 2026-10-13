"""GPT-2 prefill from primitives: T prompt positions through the blocks, logits gated against the oracle.

    source scripts/local_env.sh
    .venv/bin/python -m examples.gpt2_from_primitives.run_prefill --tokens 64 [--layers 12] [--device 3]
        [--capture graphs/gpt2_prefill_python_n32] [--plan bootstrap_placements/gpt2_prefill_python_n32]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from .run_decode import gpu_busy


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--device", type=int, default=3)
    ap.add_argument("--capture", default=None)
    ap.add_argument("--structure-only", action="store_true")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--weights", default=os.environ.get("WEIGHTS_PATH"))
    ap.add_argument("--configs", default=os.environ.get("CONFIGS_PATH"))
    ap.add_argument("--cpp-bootstrap", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--check", action="store_true", help="also compare every block with the float64 reference")
    ap.add_argument("--taps", action="store_true", help="compare block 0's sub-steps with the float64 reference")
    ap.add_argument("--random-ids", type=int, default=None, metavar="SEED",
                    help="a random prompt (wte[ids] + wpe), gated against the float64 model instead of the oracle "
                         "(any T up to the 1024 positions)")
    a = ap.parse_args(argv)

    busy = gpu_busy(a.device)
    if busy and not a.force:
        print(f"GPU {a.device} has other processes: {busy}; refusing (use --force)", file=sys.stderr)
        return 2
    from . import env
    # the prefill keeps the host wait on device-encoded plaintexts that FIDESlib f12c06e restored for this path;
    # the decode runs without it (FIDESLIB_ENCODE_NOSYNC default 1). --set FIDESLIB_ENCODE_NOSYNC=1 overrides.
    over = {"FIDESLIB_ENCODE_NOSYNC": "0", **env.plan_runtime(a.plan, "n32"), "GPT2_PACKING": "cachemir",
            **dict(kv.split("=", 1) for kv in a.set)}
    for k, v in over.items():          # a plan's PF_* switches (the arm it was made for); the shell's win
        if k.startswith("PF_"):
            os.environ.setdefault(k, str(v))
    pcmm = os.environ.get("PF_LINEAR", "") == "pcmm"
    if pcmm and a.plan and not os.environ.get("PF_PCMM_CALIB_IN"):   # a PCMM plan carries its calibration
        cal = os.path.join(a.plan, "pcmm_calib.json")
        if os.path.exists(cal):
            os.environ["PF_PCMM_CALIB_IN"] = cal
    if pcmm:   # StC-first on the dense route, for the PCMM only
        over = {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
                "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY": os.environ.get("PF_PCMM_STAGED_ONLY", "1"), **over}
    env.export_env(a.device, chain="n32", CKKS_COMPLEX="1", **over)
    if not a.configs:   # the prefill keeps the composite GELU its plans were cut for
        a.configs = env.plan_configs(a.plan, "n32", env.CONFIGS_N32_LEGACY)
    if a.capture:
        os.environ["FHE_GRAPH_DIR"] = a.capture
    from perseus import _core
    from perseus.impl import config
    from perseus.impl.layout import Dims
    from perseus.impl.pcmm import rot_steps as pcmm_rot_steps
    from perseus.impl.prefill import diag_rot_steps, prefill_rot_steps
    from . import weights
    from .prefill import Gpt2Prefill, exact_blocks, plain_logits, report

    sess = env.open_session(a.device, complex_payload=True, chain="n32",
                            extra_rot_steps=prefill_rot_steps(Dims.gpt2()) | diag_rot_steps(Dims.gpt2())
                            | (pcmm_rot_steps() if pcmm else set()), **over)
    inf = sess.inf
    env.bootstrap_setup(inf.fhe, over, cpp_bootstrap=a.cpp_bootstrap)
    store = weights.RawStore(a.weights)
    cfgs = config.load_configs(a.configs)
    model = Gpt2Prefill(inf, store, cfgs, core=_core, n_layers=a.layers, profile=a.profile)
    d = model.rt.dims
    loaded = {int(s) % d.N for s in inf.fhe.loaded_rot_steps}
    missing = sorted(set(env.rotation_audit(inf.fhe, d)) | ((prefill_rot_steps(d) | diag_rot_steps(d)) - loaded))
    if missing:
        print(f"rotation keys missing: {missing}", file=sys.stderr)
        return 3
    if a.capture:
        model.set_capture(a.capture)
        from perseus.impl.magtape import save_rot_steps
        save_rot_steps(a.capture, inf.fhe.loaded_rot_steps)
        if a.structure_only:
            inf.fhe.magnitude_suppressed = True
    if a.plan:
        model.load_plans(a.plan, argmax_blocks=False)

    os.environ.setdefault("STEPS_T", "128")
    os.environ["MULTI_T"] = str(a.tokens)
    rc = _core.RunConfig.from_env()
    if a.random_ids is not None:
        ids = np.random.default_rng(a.random_ids).integers(0, store.shape(weights.WTE)[0], a.tokens)
        rows = store.tensor(weights.WTE)[ids] + store.tensor(weights.WPE)[:a.tokens]
        gt = plain_logits(store, exact_blocks(store, rows, 12)[-1])
    else:
        rows = np.asarray(_core.read_teacher_forced_inputs(rc)[:a.tokens])
        gt = np.asarray(_core.read_lm_head_steps(rc).logits)[:a.tokens]
    ref = exact_blocks(store, rows, model.n_layers) if a.check else None
    print(f"[prefill] T={a.tokens} layers={model.n_layers} cts={(a.tokens + 2 * d.t - 1) // (2 * d.t)} "
          f"capture={a.capture} plan={a.plan}", flush=True)
    b0 = [int(inf.fhe.total_bootstraps)]

    def on_block(b, Xs, s):
        tb = int(inf.fhe.total_bootstraps)
        line = f"[prefill] block {b} {s:.2f}s bts={tb - b0[0]}"
        b0[0] = tb
        if ref is not None:
            from perseus.impl.prefill import Chunks
            h = model.hidden(Xs, Chunks(a.tokens, d.t))
            err = np.abs(h - ref[b]).max(1)
            line += f" maxerr={err.max():.3g} (pos {int(err.argmax())}) |x|={np.abs(ref[b]).max():.3g}"
        print(line, flush=True)

    if a.taps:
        from .prefill import block0_taps, tap_rows
        from perseus.impl.prefill import Chunks
        want = block0_taps(store, rows, d)
        chk = Chunks(a.tokens, d.t)

        def tap(name, cts):
            if name in want and not getattr(model, "_tapped", set()) >= {name}:
                got = tap_rows(model.rt, cts, chk, want[name].shape[1])
                err = np.abs(got - want[name]).max(1)
                print(f"[tap] {name}: maxerr {err.max():.3g} at pos {int(err.argmax())} |x| {np.abs(want[name]).max():.3g} "
                      f"per-pos {' '.join(f'{e:.2g}' for e in err[:a.tokens])}", flush=True)
                model.__dict__.setdefault("_tapped", set()).add(name)
        model.rt.tap = tap
    t_pre = time.time()
    model.preload_prefill()
    if pcmm:
        from perseus.impl.pcmm import engine
        engine(model.rt).prepare(model.rt.dims.dim)
    print(f"[prefill] weights ready {time.time() - t_pre:.1f}s", flush=True)
    t0 = time.time()
    Xs, ch, rec = model.prefill(rows, on_block=on_block)
    wall = time.time() - t0
    h = model.hidden(Xs, ch)
    out = {"wall": wall, "bts": int(inf.fhe.total_bootstraps)}
    if model.n_layers == 12:
        rows_, s = report(plain_logits(store, h), gt)
        out.update(s)
        print("[prefill] per-position KL " + " ".join(f"{r['kl']:.3g}" for r in rows_), flush=True)
    from perseus.impl.prefill import LockstepOps
    o = model.rt.ops
    print(f"[prefill] levels: limit={o.level_limit()} bts_out={o.bts_out()} unit={o.unit} "
          f"joint_refreshes={LockstepOps.total} deliberate={o.n_bootstraps}", flush=True)
    pc = getattr(model.rt, "pcmm", None)
    if pc is not None:
        print(f"[pcmm] calls={pc.calls} cap={pc.cap} land={pc.land} keygen={pc.keygen_s:.1f}s split->out ms: "
              + " ".join(f"{k}={v:.0f}" for k, v in zip(("rescale", "ring_switch", "mlwe", "gemm", "pack", "back"),
                                                        pc.times))
              + f" | stc_in={pc.t_in:.2f}s pcmm={pc.t_mm:.2f}s finish={pc.t_out:.2f}s", flush=True)
    print(f"[prefill] SUMMARY T={a.tokens} wall={wall:.1f}s per_token={wall / a.tokens:.3f}s "
          f"bts={out['bts']} top1={out.get('top1')}/{a.tokens} medianKL={out.get('medianKL', float('nan')):.4g} "
          f"maxKL={out.get('maxKL', float('nan')):.4g} free={rec.get('free_gb')}GB", flush=True)
    if a.profile and hasattr(model.rt.ops, "report"):
        print("[prefill] time split:\n" + model.rt.ops.report())
        print("[prefill] per step:\n" + model.rt.ops.report_steps(1))
    ok = model.n_layers != 12 or (out["top1"] >= 0.9 * a.tokens and out["medianKL"] < 0.1)
    print("[prefill] PASS" if ok else "[prefill] FAIL")
    model.close()
    sess.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
