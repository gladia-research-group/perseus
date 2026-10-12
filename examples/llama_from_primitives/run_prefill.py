"""Llama prefill from primitives: BOS + T prompt tokens through the blocks, logits gated against the fp32 model.

    .venv/bin/python -m examples.llama_from_primitives.run_prefill --tokens 16 [--layers 32] [--window 0]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

DEFAULT_CONFIGS = "configs/model/approximation/llama31_8b_n32/configs.json"


def main(argv=None):
    from examples.gpt2_from_primitives.run_decode import gpu_busy
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16, help="prompt positions after the BOS prefix")
    ap.add_argument("--window", default="0")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--device", type=int, default=3)
    ap.add_argument("--configs", default=DEFAULT_CONFIGS)
    ap.add_argument("--capture", default=None)
    ap.add_argument("--structure-only", action="store_true")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--cpp-bootstrap", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--check", action="store_true", help="compare every block with the float64 model")
    ap.add_argument("--no-coeff-rows", action="store_true")
    ap.add_argument("--taps", action="store_true", help="compare block 0's sub-steps with the float64 model")
    ap.add_argument("--fake", action="store_true", help="capture on the numpy fake (perseus.impl.fakecap): no GPU")
    a = ap.parse_args(argv)
    busy = None if a.fake else gpu_busy(a.device)
    if busy and not a.force:
        print(f"GPU {a.device} has other processes: {busy}; refusing (use --force)", file=sys.stderr)
        return 2
    from . import env, hf
    from . import weights as W
    from .config import load_configs
    toks = hf.text_windows(int(a.window) + 1, 128)[int(a.window)][:a.tokens + 1]
    store = W.SafetensorsStore(hf.NAME)
    cache = os.path.join(os.environ.get("HF_HOME", "."), "perseus", "llama_oracle",
                         f"{store.fingerprint()}_{abs(hash(tuple(int(t) for t in toks))) % 10**12}.npy")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if a.fake else str(a.device)
    gt = None if a.fake else hf.oracle_logits(toks, cache)
    # the prefill keeps the host wait on device-encoded plaintexts that FIDESlib f12c06e restored for this path;
    # the decode runs without it (FIDESLIB_ENCODE_NOSYNC default 1). --set FIDESLIB_ENCODE_NOSYNC=1 overrides.
    over = {"FIDESLIB_ENCODE_NOSYNC": "0", **env.plan_runtime(a.plan, "n32"), "LLAMA_PACKING": "cachemir_complex",
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
    if a.capture:
        os.environ["FHE_GRAPH_DIR"] = a.capture
        os.environ.setdefault("FHE_GRAPH_CAPTURE_TOKEN", "1")
    cfgs = load_configs(a.configs)
    dims = W.llama_dims(store.config)
    from perseus.impl.prefill import prefill_rot_steps
    from perseus.impl.pcmm import rot_steps as pcmm_rot_steps
    from perseus.impl.layout import ladder_radix_bits, ladder_radix_steps
    rot = prefill_rot_steps(dims) | (pcmm_rot_steps() if pcmm else set()) | ladder_radix_steps(dims, ladder_radix_bits())
    if os.environ.get("PF_LAYOUT", "") == "blocked":
        from perseus.impl.blocked import Blocks, blocked_rot_steps
        rot |= blocked_rot_steps(Blocks(dims, int(os.environ.get("PF_NCH", "512"))), ladder_radix_bits())
    t0 = time.time()
    if a.fake:
        from perseus.impl import fakecap
        env.export_env(a.device, chain="n32", **{"CKKS_COMPLEX": "1", **over})
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
        _, inf = fakecap.make_session(dims, set(env.rot_steps(dims)) | rot)
        sess = fakecap.FakeSession(inf)
    else:
        sess = env.open_session(dims, a.device, complex_payload=True, extra_rot_steps=rot, **over)
        inf = sess.inf
        env.bootstrap_setup(inf.fhe, over, cpp_bootstrap=a.cpp_bootstrap)
    loaded = {int(s) % dims.N for s in inf.fhe.loaded_rot_steps}
    missing = [] if a.fake else sorted(set(env.rotation_audit(inf.fhe, dims)) | ({s % dims.N for s in rot} - loaded))
    if missing:
        print(f"rotation keys missing: {missing}", file=sys.stderr)
        return 3
    from perseus import _core
    from .prefill import LlamaPrefill, exact_hidden, plain_logits, report
    core = fakecap.core if a.fake else _core
    lw = W.LlamaWeights(store, cfgs, a.configs, dims, "cachemir_complex",
                        pinned=None if (a.no_coeff_rows or pcmm or a.fake) else _core.pinned_empty)
    model = LlamaPrefill(inf, lw, cfgs, core=core, n_layers=a.layers, packing="cachemir_complex", profile=a.profile)
    if a.capture:
        model.set_capture(a.capture)
        from perseus.impl.magtape import save_rot_steps
        save_rot_steps(a.capture, inf.fhe.loaded_rot_steps)
        if a.structure_only:
            inf.fhe.magnitude_suppressed = True
    if a.plan:
        model.load_plans(a.plan)
    ref = exact_hidden(model, toks, model.n_layers) if a.check else None
    print(f"[llama-prefill] session {time.time() - t0:.0f}s T={a.tokens} layers={model.n_layers} "
          f"cts={(a.tokens + 2 * dims.t - 1) // (2 * dims.t)}", flush=True)
    b0 = [int(inf.fhe.total_bootstraps)]

    def on_block(b, Xs, s):
        tb = int(inf.fhe.total_bootstraps)
        line = f"[llama-prefill] block {b} {s:.2f}s bts={tb - b0[0]}"
        b0[0] = tb
        if ref is not None:
            from perseus.impl.prefill import Chunks
            lt = model.blocks.t if os.environ.get("PF_LAYOUT", "") == "blocked" else dims.t
            h = model.hidden_rows(Xs, Chunks(len(toks) - 1, lt, base=1))
            err = np.abs(h - ref[b]).max(1)
            line += f" maxerr={err.max():.3g} (pos {int(err.argmax()) + 1}) |x|={np.abs(ref[b]).max():.3g}"
        print(line, flush=True)

    if a.taps and os.environ.get("PF_LAYOUT", "") != "blocked":
        from perseus.impl.prefill import Chunks
        from examples.gpt2_from_primitives.prefill import tap_rows
        from .prefill import block0_taps
        want = block0_taps(model, toks)
        chk = Chunks(a.tokens, dims.t, base=1)
        done = set()

        def tap(name, cts):
            if name in want and name not in done:
                done.add(name)
                got = tap_rows(model.rt, cts, chk, want[name].shape[1])
                err = np.abs(got - want[name]).max(1)
                print(f"[tap] {name}: maxerr {err.max():.3g} at pos {int(err.argmax()) + 1} |x| {np.abs(want[name]).max():.3g} "
                      f"per-pos {' '.join(f'{e:.2g}' for e in err)}", flush=True)
        model.rt.tap = tap
    if pcmm:
        from perseus.impl.pcmm import engine
        t_pre = time.time()
        for b in range(model.n_layers):
            model.pcmm_weights(b)
        engine(model.rt).prepare(dims.dim)
        print(f"[llama-prefill] pcmm weights + keys {time.time() - t_pre:.1f}s", flush=True)
    t1 = time.time()
    Xs, ch, rec = model.prefill(toks, on_block=on_block)
    wall = time.time() - t1
    pc = getattr(model.rt, "pcmm", None)
    if pc is not None:
        print(f"[pcmm] calls={pc.calls} cap={pc.cap} land={pc.land} keygen={pc.keygen_s:.1f}s split->out ms: "
              + " ".join(f"{k}={v:.0f}" for k, v in zip(("rescale", "ring_switch", "mlwe", "gemm", "pack", "back"),
                                                        pc.times))
              + f" | stc_in={pc.t_in:.2f}s pcmm={pc.t_mm:.2f}s finish={pc.t_out:.2f}s", flush=True)
    out = {}
    if model.n_layers == int(store.config["num_hidden_layers"]) and gt is not None:
        rows, out = report(plain_logits(model, model.hidden_rows(Xs, ch)), gt[1:a.tokens + 1])
        print("[llama-prefill] per-position KL " + " ".join(f"{r['kl']:.3g}" for r in rows), flush=True)
    print(f"[llama-prefill] SUMMARY T={a.tokens} wall={wall:.1f}s per_token={wall / a.tokens:.3f}s "
          f"bts={int(inf.fhe.total_bootstraps)} top1={out.get('top1')}/{a.tokens} "
          f"medianKL={out.get('medianKL', float('nan')):.4g} maxKL={out.get('maxKL', float('nan')):.4g} "
          f"free={rec.get('free_gb')}GB", flush=True)
    if a.profile and hasattr(model.rt.ops, "report"):
        print("[llama-prefill] time split:\n" + model.rt.ops.report())
        print("[llama-prefill] per step:\n" + model.rt.ops.report_steps(1))
    model.close()
    sess.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
