"""GPT-2 generation from primitives: prompt rows through decode steps, then the encrypted
CutMax argmax and the encrypted feedback embedding (src/app/pipeline.cu).

    .venv/bin/python -m examples.gpt2_from_primitives.run_generate --prompt 1 --tokens 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from .run_decode import gpu_busy


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=int(os.environ.get("GEN_PROMPT", "1")))
    ap.add_argument("--tokens", type=int, default=int(os.environ.get("GEN_TOKENS", "4")))
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--device", type=int, default=3)
    ap.add_argument("--capture", default=None)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--no-plan-argmax", action="store_true")
    ap.add_argument("--weights", default=os.environ.get("WEIGHTS_PATH"))
    ap.add_argument("--configs", default=os.environ.get("CONFIGS_PATH"))
    ap.add_argument("--io-dir", default=os.environ.get("ALL_BLOCKS_IO_DIR"))
    ap.add_argument("--packing", choices=("cachemir", "cachemir_complex"), default="cachemir_complex",
                    help="cachemir_complex (default): fused K+iV linear, output-packed up/down "
                         "projections, complex K/V buckets and paired LM-head tile (needs the "
                         "complex payload); cachemir: real linears, the complex payload only in the "
                         "K/V push, CutMax and the feedback (the C++ decode's Mode-A)")
    ap.add_argument("--payload", choices=("complex", "real"), default="complex",
                    help="complex: CKKS_COMPLEX=1, the C++ decode configuration (K/V pair bootstrap, "
                         "packed CutMax); real: real slots only (no shipped plan)")
    ap.add_argument("--chain", choices=("n32", "n64"),
                    default=os.environ.get("CHAIN") or "n32",
                    help="CKKS chain; source scripts/local_env.sh for the SAME chain")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one of the port's session env values (env.ENV), repeatable")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    busy = gpu_busy(a.device)
    if busy and not a.force:
        print(f"GPU {a.device} has other processes: {busy}; refusing", file=sys.stderr)
        return 2
    from . import env
    over = {"GPT2_PACKING": a.packing, **dict(kv.split("=", 1) for kv in a.set)}
    env.export_env(a.device, chain=a.chain,
                   CKKS_COMPLEX="1" if a.payload == "complex" else "0", **over)
    if not a.configs:
        _cfg = "gpt2_base_n32" if a.chain == "n32" else "gpt2_base"
        a.configs = f"configs/model/approximation/{_cfg}/configs.json"
    if a.capture:
        os.environ["FHE_GRAPH_DIR"] = a.capture
    from perseus import _core
    from perseus.impl import config
    from . import weights
    from .model import Gpt2Primitives

    sess = env.open_session(a.device, complex_payload=(a.payload == "complex"),
                            chain=a.chain, **over)
    model = Gpt2Primitives(sess.inf, weights.RawStore(a.weights), config.load_configs(a.configs),
                           core=_core, n_layers=a.layers, packing=a.packing)
    if a.capture:
        model.set_capture(a.capture)
    if a.plan:
        model.load_plans(a.plan, argmax_blocks=not a.no_plan_argmax)
    os.environ["ALL_BLOCKS_IO_DIR"] = a.io_dir or ""   # --io-dir wins over an exported default
    os.environ.setdefault("STEPS_T", "128")   # the shipped per-step logits oracle horizon
    os.environ["MULTI_T"] = str(a.prompt)
    cfg = _core.RunConfig.from_env()
    rows = _core.read_teacher_forced_inputs(cfg)[:a.prompt]
    gt = _core.read_lm_head_steps(cfg)
    gt_logits = gt.logits if gt.T else None

    def on_step(r):
        print(f"[generate] pos={r['pos']} cutmax={r['cutmax']} fhe_argmax={r['fhe_argmax']} "
              f"{'OK' if r['cutmax'] == r['fhe_argmax'] else 'MISS'} gt_argmax={r.get('gt_argmax', -1)} "
              f"z_mass={r['z_mass']:.4f} z_off_sum={r['z_off_sum']:.3e} z_off_max={r['z_off_max']:.3e}",
              flush=True)

    t0 = time.time()
    res = model.generate(rows, a.tokens, gt_logits, on_step=on_step)
    ok = all(r["cutmax"] == r["fhe_argmax"] for r in res)
    print(f"[generate] tokens={[r['cutmax'] for r in res]} bootstraps={model.rt.ops.n_bootstraps} "
          f"wall={time.time() - t0:.0f}s {'PASS' if ok else 'FAIL'}")
    model.close()
    sess.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
