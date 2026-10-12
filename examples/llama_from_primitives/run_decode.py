"""Teacher-forced Llama decode from primitives: one encrypted forward per position after the BOS
prefix, logits decrypted and gated against the exact fp32 model.

    .venv/bin/python -m examples.llama_from_primitives.run_decode --tokens 8 \
        [--capture graphs/llama31_8b_decode_python_n32] [--plan bootstrap_placements/llama31_8b_decode_python_n32]

GPU 3 must be free of other users' processes (the run refuses otherwise)."""
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
    ap.add_argument("--tokens", type=int, default=8, help="encrypted positions after the BOS prefix")
    ap.add_argument("--window", default="0", help="openwebtext window index (or a .npy of token ids)")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--device", type=int, default=3)
    ap.add_argument("--configs", default=DEFAULT_CONFIGS)
    ap.add_argument("--capture", default=None, help="graph dir: capture block graphs at the first position")
    ap.add_argument("--structure-only", action="store_true",
                    help="capture without the per-node magnitude decrypts; fill them with capture_mags.py")
    ap.add_argument("--plan", default=None, help="plan dir: run under block_<b>_placement.json")
    ap.add_argument("--packing", choices=("cachemir", "cachemir_complex"), default="cachemir_complex")
    ap.add_argument("--cpp-bootstrap", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--kv-offload", action="store_true", help="park K/V caches in the pinned KV arena")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fake", action="store_true", help="capture on the numpy fake (perseus.impl.fakecap): no GPU")
    ap.add_argument("--tap-blocks", default="", help="blocks whose sub-steps are tapped too (comma list)")
    ap.add_argument("--taps", default=None, help="decrypt every block output; save {pos_block: vec} to this .npz")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--no-coeff-rows", action="store_true",
                    help="encode the weight plaintexts from their slot vectors (not the pinned coefficient rows)")
    a = ap.parse_args(argv)
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)    # kill -USR1 <pid>: a stack dump

    busy = None if a.fake else gpu_busy(a.device)
    if busy and not a.force:
        print(f"GPU {a.device} has other processes: {busy}; refusing (use --force)", file=sys.stderr)
        return 2
    from . import env, hf
    from . import weights as W
    from .config import load_configs

    if a.window.endswith(".npy"):
        toks = np.load(a.window)
        toks = toks if toks.ndim == 1 else toks[0]
    else:
        toks = hf.text_windows(int(a.window) + 1, 128)[int(a.window)]
    toks = toks[:a.tokens + 1]
    store = W.SafetensorsStore(hf.NAME)
    cache = os.path.join(os.environ.get("HF_HOME", "."), "perseus", "llama_oracle",
                         f"{store.fingerprint()}_{abs(hash(tuple(int(t) for t in toks))) % 10**12}.npy")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if a.fake else str(a.device)
    gt = None if a.fake else hf.oracle_logits(toks, cache)   # before the session: the fp32 model needs the GPU

    over = {**env.plan_runtime(a.plan, "n32"), "LLAMA_PACKING": a.packing,
            **dict(kv.split("=", 1) for kv in a.set)}
    if a.capture:
        os.environ["FHE_GRAPH_DIR"] = a.capture
        # the first encrypted position is 1 (the BOS prefix is plaintext)
        os.environ.setdefault("FHE_GRAPH_CAPTURE_TOKEN", "1")
    cfgs = load_configs(a.configs)
    dims = W.llama_dims(store.config)
    t0 = time.time()
    if a.fake:
        from perseus.impl import fakecap
        env.export_env(a.device, chain="n32", **{"CKKS_COMPLEX": "1", **over})
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from perseus.impl.layout import ladder_radix_bits, ladder_radix_steps
        _, inf = fakecap.make_session(dims, set(env.rot_steps(dims)) | (
            ladder_radix_steps(dims, ladder_radix_bits()) if ladder_radix_bits() >= 2 else set()))
        sess = fakecap.FakeSession(inf)
    else:
        sess = env.open_session(dims, a.device, complex_payload=True, **over)
        inf = sess.inf
        env.bootstrap_setup(inf.fhe, over, cpp_bootstrap=a.cpp_bootstrap)
    print(f"[llama] session {time.time() - t0:.0f}s", flush=True)
    missing = [] if a.fake else env.rotation_audit(inf.fhe, dims)
    if missing:
        print(f"rotation keys missing: {missing}", file=sys.stderr)
        return 3
    from perseus import _core
    from .model import LlamaModel
    lw = W.LlamaWeights(store, cfgs, a.configs, dims, a.packing,
                        pinned=None if (a.no_coeff_rows or a.fake) else _core.pinned_empty)
    model = LlamaModel(inf, lw, cfgs, core=fakecap.core if a.fake else _core, n_layers=a.layers,
                       packing=a.packing, profile=a.profile, kv_offload=a.kv_offload)
    print(f"[llama] model {time.time() - t0:.0f}s", flush=True)
    model.preload()
    print(f"[llama] session+weights {time.time() - t0:.0f}s; decode {len(toks) - 1} positions, "
          f"layers={model.n_layers} packing={a.packing} capture={a.capture} plan={a.plan}", flush=True)
    if a.capture:
        model.set_capture(a.capture)
        if a.structure_only:
            inf.fhe.magnitude_suppressed = True
    if a.plan:
        model.load_plans(a.plan)
    if a.taps:
        model.taps = {}
        model.tap_blocks = tuple(int(b) for b in a.tap_blocks.split(",")) if a.tap_blocks else ()

    def on_token(r):
        line = (f"[llama] pos{r['pos']} top1={r['top1']} decode={r['decode_s']:.1f}s bts={r['bootstraps']}")
        if "kl" in r:
            line += f" ref={r['ref']} rank={r['ref_rank']} KL={r['kl']:.4g} {'OK' if r['hit'] else 'MISS'}"
        for k in ("free_gb", "rss_gb", "cpu_s"):
            if k in r:
                line += f" {k}={r[k]}"
        if r.get("masks"):
            m = r["masks"]
            line += f" masks={m.get('adopted', 0)}/{m.get('evicted', 0)}/{m.get('misses', 0)}"
        print(line, flush=True)
        if a.profile and hasattr(model.rt.ops, "report_steps"):
            print("[llama] steps: " + " | ".join(t.split("%")[0].strip() for t in model.rt.ops.report_steps(1).split("\n")[:8]), flush=True)
            print("[llama] primitives: " + " | ".join(t.split("%")[0].strip() for t in model.rt.ops.report(10).split("\n")[:10]), flush=True)
            if os.environ.get("LLAMA_PROFILE_STEPS"):
                for st in sorted(model.rt.ops.t_step, key=lambda k: -model.rt.ops.t_step[k]):
                    print(f"[llama] step {st}: " + model.rt.ops.report_step_split(st, 1, 6), flush=True)
            model.rt.ops.reset()

    t1 = time.time()
    res = model.run_decode(toks, gt if model.n_layers == int(store.config["num_hidden_layers"]) else None,
                           on_token=on_token)
    kls = sorted(r["kl"] for r in res if "kl" in r)
    hits = sum(1 for r in res if r.get("hit"))
    print(f"[llama] SUMMARY positions={len(res)} top1={hits}/{len(res)} "
          f"medianKL={kls[len(kls) // 2] if kls else float('nan'):.4g} maxKL={kls[-1] if kls else float('nan'):.4g} "
          f"deliberate_bts={model.rt.ops.n_bootstraps} total_bts={getattr(inf.fhe, 'total_bootstraps', None)} "
          f"wall={time.time() - t1:.0f}s")
    if a.profile:
        print("[llama] time split:\n" + model.rt.ops.report())
    if a.taps:
        np.savez(a.taps, **{f"p{p}_b{b}": v for (p, b), v in model.taps.items()})   # b: block or "<b>.<site>"
    model.close()
    sess.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
