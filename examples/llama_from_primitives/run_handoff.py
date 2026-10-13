"""Encrypted prefill of T prompt tokens, its K / V handed to the decode's caches (perseus.impl.handoff), then N
decoded positions, logits gated against the fp32 model.

    .venv/bin/python -m examples.llama_from_primitives.run_handoff --tokens 120 --decode 8 \
        --prefill-plan bootstrap_placements/llama31_8b_prefill_python_n32_blocked_gqa \
        --decode-plan bootstrap_placements/llama31_8b_decode_python_n32
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def main(argv=None):
    from examples.gpt2_from_primitives.run_decode import gpu_busy
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--window", default="0")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--device", type=int, default=3)
    ap.add_argument("--prefill-plan", default=None)
    ap.add_argument("--decode-plan", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--no-handoff", action="store_true", help="decode every position from the BOS prefix (control)")
    ap.add_argument("--keys", choices=("all", "plan"), default="all", help="plan: the prefill plan's keys only (diag)")
    ap.add_argument("--prefill-only", action="store_true", help="no decode model (diag)")
    ap.add_argument("--cache-level", type=int, default=None, help="drop the handed K / V to this level")
    ap.add_argument("--taps", action="store_true", help="decoded block outputs vs the float model (diag)")
    ap.add_argument("--set", action="append", default=[], help="KEY=VAL over the plans' runtime")
    ap.add_argument("--launder", action="store_true", help="re-encrypt the handed cts from their values (diag)")
    ap.add_argument("--cleanse", action="store_true", help="drop the imaginary part of the prefill K / V first")
    a = ap.parse_args(argv)
    os.environ["KV_ARENA_GB"] = os.environ.get("LLAMA_KV_ARENA_GB", "72")
    busy = None if a.fake else gpu_busy(a.device)
    if busy and not a.force:
        print(f"GPU {a.device} has other processes: {busy}; refusing (use --force)", file=sys.stderr)
        return 2
    from . import env, hf
    from . import weights as W
    from .config import load_configs
    from .run_prefill import DEFAULT_CONFIGS
    n = a.tokens + 1 + a.decode
    if a.window.endswith(".npy"):
        toks = np.load(a.window)
        toks = (toks if toks.ndim == 1 else toks[0])[:n]
    else:
        toks = hf.text_windows(int(a.window) + 1, n)[int(a.window)][:n]
    store = W.SafetensorsStore(hf.NAME)
    cache = os.path.join(os.environ.get("HF_HOME", "."), "perseus", "llama_oracle",
                         f"{store.fingerprint()}_{abs(hash(tuple(int(t) for t in toks))) % 10**12}.npy")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if a.fake else str(a.device)
    gt = None if a.fake else hf.oracle_logits(toks, cache)
    over = {"FIDESLIB_ENCODE_NOSYNC": "0", **env.plan_runtime(a.decode_plan, "n32"),
            **env.plan_runtime(a.prefill_plan, "n32"), "LLAMA_PACKING": "cachemir_complex"}
    for k, v in over.items():
        if k.startswith("PF_"):
            os.environ.setdefault(k, str(v))
    over = {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
            "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY": "1", "KV_ARENA_GB": os.environ["KV_ARENA_GB"], **over,
            **dict(kv.split("=", 1) for kv in a.set)}
    cal = os.path.join(a.prefill_plan or "", "pcmm_calib.json")
    if a.prefill_plan and os.path.exists(cal):
        os.environ.setdefault("PF_PCMM_CALIB_IN", cal)
    def plan_configs(plan):
        for line in open(os.path.join(plan, "PLAN_CMD.txt")).read().splitlines() if plan else []:
            if line.startswith("configs="):
                return line[len("configs="):].split()[0]
        return DEFAULT_CONFIGS
    cfgs_path, dcfgs_path = plan_configs(a.prefill_plan), plan_configs(a.decode_plan)
    cfgs, dcfgs = load_configs(cfgs_path), load_configs(dcfgs_path)
    dims = W.llama_dims(store.config)
    from perseus.impl.blocked import Blocks, blocked_rot_steps
    from perseus.impl.layout import ladder_radix_bits
    from perseus.impl.magtape import plan_rot_steps
    from perseus.impl.pcmm import rot_steps as pcmm_rot_steps
    rot = set() if a.keys == "plan" else set(env.rot_steps(dims)) | pcmm_rot_steps() | {-64, -128, 64, 128}
    if a.prefill_plan:
        rot |= plan_rot_steps(a.prefill_plan)
    # the hand-off's lane / group moves decompose over the blocked layout's keys (a CCMM plan's graph has none)
    rot |= blocked_rot_steps(Blocks(dims, 512), ladder_radix_bits())
    t0 = time.time()
    if a.fake:
        from perseus.impl import fakecap
        env.export_env(a.device, chain="n32", **{"CKKS_COMPLEX": "1", **over})
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
        _, inf = fakecap.make_session(dims, rot)
        sess = fakecap.FakeSession(inf)
    else:
        sess = env.open_session(dims, a.device, complex_payload=True, extra_rot_steps=rot, base=a.keys == "all", **over)
        inf = sess.inf
        env.bootstrap_setup(inf.fhe, over, cpp_bootstrap=False)
    keys = {int(s) for s in inf.fhe.loaded_rot_steps}
    from perseus import _core
    from perseus.impl import fakecap as FC
    from perseus.impl.handoff import to_decode_cache
    from . import prefix as PX
    from .model import LlamaModel
    from .prefill import LlamaPrefill, report
    core = FC.core if a.fake else _core
    pre = LlamaPrefill(inf, W.LlamaWeights(store, cfgs, cfgs_path, dims, "cachemir_complex", pinned=None), cfgs,
                       core=core, n_layers=a.layers, packing="cachemir_complex")
    dec = None if a.prefill_only else LlamaModel(inf, W.LlamaWeights(store, dcfgs, dcfgs_path, dims, "cachemir_complex",
                                         pinned=None if a.fake else _core.pinned_empty),
                     dcfgs, core=core, n_layers=a.layers, packing="cachemir_complex", kv_offload=not a.fake)
    if dec is not None and a.decode_plan:
        dec.rt.ops.sm_cheb_gs = env.plan_runtime(a.decode_plan, "n32").get("SM_CHEB_GS", "0") not in ("", "0")
    if a.prefill_plan and not a.fake:
        pre.load_plans(a.prefill_plan)
    print(f"[handoff] session {time.time() - t0:.0f}s T={a.tokens} decode={a.decode} keys={len(keys)}", flush=True)
    from perseus.impl.pcmm import engine
    for b in range(pre.n_layers):
        pre.pcmm_weights(b)
    engine(pre.rt).prepare(dims.dim)
    bos = pre.w.bos_kv(pre.n_layers)
    dh = dims.dim // pre.H

    def bos_vec(f):
        x = np.zeros(dims.N)
        x[np.arange(dims.hid) * dims.t] = PX.expanded_heads(f, pre.H, pre.H_kv, dh)
        return x
    t_ho = [0.0]

    def on_kv(b, K, V):
        t = time.time()
        with pre.rt.step("handoff"):
            if b == 0 and not a.fake:
                for nm, S in (("K", K), ("V", V)):
                    z = pre.rt.ops.decrypt_slots_complex(S[0][0])
                    print(f"[handoff] layer 0 {nm}[0][0] max|re|={np.abs(z.real).max():.3g} "
                          f"max|im|={np.abs(z.imag).max():.3g}", flush=True)
            kv = to_decode_cache(pre.rt, K, V, a.tokens, bos_vec(bos[b][0]), bos_vec(bos[b][1]), keys=keys, layer=b,
                                 cleanse=a.cleanse)
            if a.launder:
                ops = pre.rt.ops

                def fresh(c):
                    y = ops.add_pt(core.encode_token_input(inf, np.zeros(dims.dim)), ops.decrypt_slots_complex(c))
                    return ops.level_drop(y, a.cache_level or 34)
                kv.k_buckets = [fresh(c) for c in kv.k_buckets]
                kv.k_pend = [fresh(c) for c in kv.k_pend]
                kv.v_buckets = [None if c is None else fresh(c) for c in kv.v_buckets]
            cts, _ = kv.cts("")
            if a.cache_level is not None:
                for c in cts:
                    pre.rt.ops.level_drop(c, a.cache_level)
            if b == 0 and not a.fake:
                lv = sorted({(pre.rt.ops.level(c), pre.rt.ops.noise_deg(c)) for c in cts})
                from perseus._core import bts as BT
                dl = sorted({BT.device(inf.fhe, c).level for c in cts})
                print(f"[handoff] layer 0 cache cts={len(cts)} (level, deg)={lv} device levels={dl}", flush=True)
        dec.adopt_cache(b, kv)
        t_ho[0] += time.time() - t

    def on_block(b, Xs, s):
        print(f"[handoff] prefill block {b} {s:.2f}s bts={int(inf.fhe.total_bootstraps)}", flush=True)
    t1 = time.time()
    Xs, ch, _ = pre.prefill(toks[:a.tokens + 1], on_block=on_block,
                            on_kv=None if (a.no_handoff or dec is None) else on_kv)
    ttft = time.time() - t1
    from .prefill import plain_logits
    from perseus.impl.prefill import Chunks
    if gt is not None and pre.n_layers == int(store.config["num_hidden_layers"]):
        rows, out = report(plain_logits(pre, pre.hidden_rows(Xs, ch)), gt[1:a.tokens + 1])
        print(f"[handoff] prefill top1={out.get('top1')}/{a.tokens} medianKL={out.get('medianKL', float('nan')):.4g}",
              flush=True)
    print(f"[handoff] prefill+handoff {ttft:.1f}s (handoff {t_ho[0]:.1f}s) bts={int(inf.fhe.total_bootstraps)}",
          flush=True)
    if dec is None:
        sess.close()
        return 0
    del Xs
    pre.__dict__.pop("_pcmm_w", None)
    if hasattr(_core, "gpu_pool_trim") and not a.fake:
        _core.gpu_pool_trim()
    dec.preload()
    if a.decode_plan and not a.fake:
        dec.load_plans(a.decode_plan)
        if a.layers:
            dec.plans.pop(dec.n_layers, None)
    b0 = int(inf.fhe.total_bootstraps)

    def on_token(r):
        line = f"[handoff] pos{r['pos']} top1={r['top1']} decode={r['decode_s']:.1f}s bts={r['bootstraps'] - b0}"
        if "kl" in r:
            line += f" ref={r['ref']} KL={r['kl']:.4g} {'OK' if r['hit'] else 'MISS'}"
        print(line, flush=True)
    if a.fake or a.taps:
        dec.taps, dec.tap_blocks = {}, ()
    t2 = time.time()
    try:
        res = dec.run_decode(toks, gt, on_token=on_token, start=None if a.no_handoff else a.tokens + 1)
    except Exception as e:
        if not a.taps:
            raise
        print(f"[handoff] decode stopped: {str(e)[:160]}", flush=True)
        res = []
    kls = sorted(r["kl"] for r in res if "kl" in r)
    hits = sum(1 for r in res if r.get("hit"))
    print(f"[handoff] SUMMARY T={a.tokens} ttft={ttft:.1f}s decode={len(res)} top1={hits}/{len(res)} "
          f"medianKL={kls[len(kls) // 2] if kls else float('nan'):.4g} maxKL={kls[-1] if kls else float('nan'):.4g} "
          f"decode_wall={time.time() - t2:.0f}s", flush=True)
    if a.fake or a.taps:
        from .prefill import exact_hidden
        ref = exact_hidden(pre, toks, pre.n_layers)
        for (pos, b), h in sorted(dec.taps.items(), key=lambda kv: str(kv[0])):
            if isinstance(b, int):
                r = ref[b][pos - 1]
                print(f"[handoff] fake pos{pos} blk{b} maxerr={np.abs(np.asarray(h)[:r.size] - r).max():.3g} "
                      f"|x|={np.abs(r).max():.3g}", flush=True)
    dec.close()
    sess.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
