"""Teacher-forced GPT-2 decode from primitives, gated like scripts/modes_baseline.py.

    source scripts/local_env.sh
    .venv/bin/python -m examples.gpt2_from_primitives.run_decode --tokens 16 [--argmax]
        [--capture graphs/gpt2_decode_python_n32] [--plan bootstrap_placements/gpt2_decode_python_n32]
        [--layers 12] [--device 3]

GPU 3 must be free of other users' processes (the run refuses otherwise)."""
from __future__ import annotations

import argparse
import os
import sys
import time


def gpu_busy(device=3):
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader", "-i", str(device)],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None
    return [l for l in out.splitlines() if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=int(os.environ.get("MULTI_T", "16")))
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--device", type=int, default=3)
    ap.add_argument("--argmax", action="store_true", help="also run the encrypted CutMax per token")
    ap.add_argument("--capture", default=None, help="graph dir: capture block graphs at token 0")
    ap.add_argument("--plan", default=None, help="plan dir: run under block_<b>_placement.json")
    ap.add_argument("--no-plan-argmax", action="store_true", help="run CutMax / feedback eager even if planned")
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
    ap.add_argument("--force", action="store_true", help="run even if GPU is busy")
    ap.add_argument("--profile", action="store_true", help="per-primitive wall-time split")
    a = ap.parse_args(argv)

    busy = gpu_busy(a.device)
    if busy and not a.force:
        print(f"GPU {a.device} has other processes: {busy}; refusing (use --force)", file=sys.stderr)
        return 2

    from . import env
    # GPT2_PACKING is only read back by the plan contract: a plan cut under one packing
    # must not load under the other
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
    from .model import Gpt2Primitives, dist_gate

    # `over` has to reach open_session as well: it calls export_env a second time, and a
    # second call re-applies env.ENV over anything set here.
    sess = env.open_session(a.device, complex_payload=(a.payload == "complex"),
                            chain=a.chain, **over)
    inf = sess.inf
    store = weights.RawStore(a.weights)
    cfgs = config.load_configs(a.configs)
    model = Gpt2Primitives(inf, store, cfgs, core=_core, n_layers=a.layers, profile=a.profile, packing=a.packing)
    missing = env.rotation_audit(inf.fhe, model.rt.dims)
    if missing:
        print(f"rotation keys missing from the band: {missing}", file=sys.stderr)
        return 3
    if a.capture:
        model.set_capture(a.capture)
    if a.plan:
        model.load_plans(a.plan, argmax_blocks=not a.no_plan_argmax)

    os.environ.setdefault("ALL_BLOCKS_IO_DIR", a.io_dir or "")
    os.environ.setdefault("STEPS_T", "128")   # the shipped per-step logits oracle horizon
    os.environ["MULTI_T"] = str(a.tokens)
    cfg = _core.RunConfig.from_env()
    inputs = _core.read_teacher_forced_inputs(cfg)[:a.tokens]
    gt = _core.read_lm_head_steps(cfg)
    gt_logits = gt.logits if gt.T else None
    print(f"[impl] decode {len(inputs)} tokens, layers={model.n_layers}, "
          f"capture={a.capture} plan={a.plan}", flush=True)

    def on_token(r):
        if a.profile and hasattr(model.rt.ops, "report_steps") and r["pos"] > 0:
            top = model.rt.ops.report_steps(1).split("\n")[:7]
            print(f"[impl] tok{r['pos']} steps: " + " | ".join(t.split("%")[0].strip() for t in top), flush=True)
            for step in ("cutmax", "softmax"):
                print(f"[impl] tok{r['pos']} {step}: " + model.rt.ops.report_step_split(step), flush=True)
        if a.profile and hasattr(model.rt.ops, "reset"):
            model.rt.ops.reset()          # per-token tables; token 0 pays the encodes
        line = f"[impl] tok{r['pos']} top1={r['top1']} decode={r['decode_s']:.1f}s"
        if "kl" in r:
            line += f" ref={r['ref']} ref_rank={r['ref_rank']} KL={r['kl']:.4g} {'OK' if r['hit'] else 'MISS'}"
        if "free_gb" in r:
            line += f" free={r['free_gb']}GB"
        if "rss_gb" in r:
            line += f" rss={r['rss_gb']}GB"
        if "cpu_s" in r:
            line += (f" cpu={r['cpu_s']}s thread={r['thread_s']}s ctx={r['ctx_vol']}/{r['ctx_invol']}"
                     f" flt={r['minflt']}/{r['majflt']}")
        if r.get("masks"):
            m = r["masks"]
            line += f" masks={m.get('adopted', 0)}/{m.get('evicted', 0)}/{m.get('misses', 0)} wait={r.get('mask_wait_s', 0):.2f}s"
        if "enc_cache" in r:
            line += f" enc={r['enc_cache']['size']} miss={r['enc_cache']['misses']} hit={r['enc_cache']['hits']}"
        if "ring_gap_s" in r:
            line += f" ring_gap={r['ring_gap_s']}s gaps={r['ring_gaps']}"
        if "cutmax" in r:
            line += f" cutmax={r['cutmax']} {'OK' if r['cutmax'] == r['top1'] else 'MISS'} z_mass={r['z_mass']:.4f} argmax={r['argmax_s']:.1f}s bts={r.get('argmax_bts', '?')}"
        print(line, flush=True)

    t0 = time.time()
    res = model.run_decode(inputs, gt_logits, on_token=on_token, argmax=a.argmax)
    ok, bad = dist_gate(res)
    hits = sum(1 for r in res if r.get("hit"))
    kls = sorted(r["kl"] for r in res if "kl" in r)
    med = kls[len(kls) // 2] if kls else float("nan")
    total_bts = getattr(inf.fhe, "total_bootstraps", None)
    print(f"[impl] SUMMARY completed={len(res)}/{len(inputs)} top1={hits}/{len(res)} "
          f"medianKL={med:.4g} deliberate_bts={model.rt.ops.n_bootstraps} "
          f"total_bts={total_bts} relevels={getattr(inf.fhe, 'weight_relevel_count', None)} "
          f"wall={time.time() - t0:.0f}s")
    if a.profile:
        print("[impl] time split over the run:\n" + model.rt.ops.report())
        print("[impl] per step (steady tokens, token 0 excluded):\n" + model.rt.ops.report_steps(max(1, len(res) - 1)))
    print("[decode] PASS" if ok else f"[decode] FAIL {[(r['pos'], r['kl'], r['ref_rank']) for r in bad]}")
    model.close()
    sess.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
