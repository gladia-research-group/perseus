"""Teacher-forced Llama decode from primitives: one encrypted forward per position after the BOS
prefix, logits decrypted and gated against the exact fp32 model.

    .venv/bin/python -m perseus.examples.llama.run_decode --tokens 8 \
        [--capture graphs/llama31_8b_decode] [--plan bootstrap_placements/llama31_8b_decode]

The GPU must be free of other processes (the run refuses otherwise; --force)."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from . import env

DEFAULT_CONFIGS = "configs/model/approximation/llama31_8b_n32/configs.json"


def plan_config(plan):
    """The session Config of a planned run: the Llama decode preset, the plan's stamp, its runtime= switches the
    Config has fields for (the rest is listed: old-stack levers without meaning here)."""
    import shlex
    from perseus import _pc, config as pcfg
    cfg = pcfg.preset("n32_llama_decode")
    stamp = pcfg._plan_stamp(plan)
    if stamp is not None:
        cfg = cfg.overlay(stamp)
    line = pcfg._plan_cmd(plan).get("runtime", "")
    rt, _pl, rest = _pc.Config.split_legacy_env(dict(kv.split("=", 1) for kv in shlex.split(line)))
    if rest:
        print(f"[llama] runtime switches without a Config field (not applied): {' '.join(sorted(rest))}", flush=True)
    return cfg.overlay(rt)


def logits_of(fhe, tiles, dims, vocab):
    """The LM head tiles (complex: two vocab tiles on the real / imaginary axis) decrypted into logits."""
    import numpy as np
    from perseus.impl.layout import decode_linear_output
    N, out = dims.N, []
    for t in tiles:
        cv = np.asarray(fhe.decrypt_slots_complex(t))
        out += [decode_linear_output(cv.real, N, dims.hid, N), decode_linear_output(cv.imag, N, dims.hid, N)]
    return np.concatenate(out)[:vocab]


def streamed_run(a, twin, lw, plan, *, log=print):
    """The planned run on the new runtime (impl.driver.StreamedPlannedRun) of the twin's model: streamed weights
    (every linear's float32 coefficient rows pinned on the host, one block's in a device frame), K/V in pinned units
    from the BOS prefix, one program class per stream structure."""
    from perseus import _pc
    from perseus.impl.driver import Recorder, StreamedPlannedRun
    n = twin.n_layers
    n_total = int(lw.store.config["num_hidden_layers"])
    rec = Recorder(twin, n)
    rec.first = 1
    labels = [f"block_{b}" for b in range(n)] + [f"block_{n_total}"]
    names = [x for x in lw.linear_names(n_total) if not x.startswith("b") or int(x[1:].split(".")[0]) < n]
    for b in range(n):      # the slot cache entries the streams below mmap: built here on a fresh cache
        lw.block(b)
    lw.lm_tiles(paired=lw.packing == "cachemir_complex")
    streams = {x: int(np.load(lw.dir / f"{x}.npy", mmap_mode="r").shape[0]) for x in names}

    def prefix(b):
        pk = twin.prefix_caches()[b]
        out = {f"kb.{j}": v for j, v in enumerate(pk.k_buckets)}
        out.update({f"kp.{j}": v for j, v in enumerate(pk.k_pend)})
        out.update({f"vb.{j}": v for j, v in enumerate(pk.v_buckets) if v is not None})
        if pk.k_ref is not None:
            out["kref"] = pk.k_ref
        return out

    def routes_for(pos):
        r = twin.routes_for(pos)
        return set() if r is None else set(r[1])
    cfg = plan_config(plan)
    cfg["exec.use_graphs"] = bool(a.graphs)
    keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
    work = a.work_dir or os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), f"m9e_decode_{os.getpid()}")
    d = twin.rt.dims
    size = {"hidDim": d.hid, "dim": d.dim, "numHeads": d.H, "numHeadsReal": d.H_real, "expDim": d.E,
            "expanded": d.E_real, "seqLen": 1024}
    windows = getattr(a, "windows", None)
    run = StreamedPlannedRun(rec, plan, cfg, labels, keys, work_dir=work, streams=streams, stream_rows=lw.coeff_rows,
                             prefix=prefix, input_vec=lw.embedding, routes_for=routes_for, stc_cache=a.stc_cache,
                             size=size, slim=not (a.debug_blocks or a.tap_block is not None), log=log,
                             schedule=getattr(a, "schedule", "program"), plan_windows=windows)
    run.debug = a.debug_blocks or a.tap_block is not None
    run.tail = n == n_total
    if a.tap_block is not None:
        run.tap = (a.tap_token, a.tap_block)
        run.tap_steps = {x for x in a.tap_steps.split(",") if x}
    return run


def _planned(a, toks, gt, lw, cfgs, dims, store):
    """PLANNED on the new runtime: setup (record every token, compile the classes, session, weights, prefix), then
    the decode loop, logits decrypted per token."""
    from perseus._paths import resolve
    from perseus.impl import fakecap
    from .model import LlamaModel, dist_report
    plan = resolve(a.plan)
    _, tinf = fakecap.make_session(dims, set(env.rot_steps(dims)))
    twin = LlamaModel(tinf, lw, cfgs, core=fakecap.core, n_layers=a.layers, packing=a.packing, kv_offload=False)
    twin.stream_twin()
    first = 1
    if a.plain_prefix:
        # the plain prefix: positions 0 .. N-1 from the exact model (K / V encrypted as public values), the encrypted
        # decode from N; only the program classes of positions >= N are recorded
        first = a.plain_prefix
        twin.plain_prompt = a.plain_kv
        print(f"[llama] plain prefix: positions 0..{first - 1} computed in plaintext (exact fp32 model, pre-RoPE "
              f"K / V pushed into the caches and encrypted as public values); encrypted decode from position {first}",
              flush=True)
    run = streamed_run(a, twin, lw, plan, log=lambda m: print(m, flush=True))
    run.rec.first = first
    rows = [int(x) for x in toks[first:]]
    t0 = time.time()
    run.setup(rows)
    print(f"[llama] planned setup {time.time() - t0:.0f}s", flush=True)
    vocab = int(store.config["vocab_size"])
    full = twin.n_layers == int(store.config["num_hidden_layers"])
    res = []
    b0 = run.sess.stream_stats()["bytes"]
    for t in range(len(rows)):
        pos = t + first
        tt = time.time()
        regs, info = run.token(t, rows)
        lg = logits_of(run.fhe, [run.ct(r) for r in regs], dims, vocab) if run.tail else np.zeros(vocab)
        r = {"pos": pos, "token_s": time.time() - tt, **info, "top1": int(np.argmax(lg)), "logits": lg}
        st = run.sess.stream_stats()
        r["pcie_gb"] = (st["bytes"] - b0) / 2**30
        b0 = st["bytes"]
        if gt is not None and full and pos < len(gt):
            r.update(dist_report(lg, np.asarray(gt[pos])))
        res.append(r)
        if run.tap and run.tap[0] == t:
            for nm, err, den in run.tap_report():
                print(f"[llama:tap] {nm}: rel {err:.3g} (max {den:.3g})", flush=True)
        line = (f"[llama] pos{pos} top1={r['top1']} token={r['token_s']:.2f}s (record {r['record_s']} patch "
                f"{r['patch_s']} issue {r['issue_s']}) class={r['class']} staged={r['staged']} pcie={r['pcie_gb']:.2f}GB")
        if "kl" in r:
            line += f" ref={r['ref']} rank={r['ref_rank']} KL={r['kl']:.4g} {'OK' if r['hit'] else 'MISS'}"
        print(line, flush=True)
    st = run.sess.stream_stats()
    print(f"[llama] planned stats {run.sess.planned_stats()} stream {st}", flush=True)
    run.close()
    return res


def main(argv=None):
    from perseus.examples.gpt2.run_decode import gpu_busy
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8, help="encrypted positions after the BOS prefix")
    ap.add_argument("--plain-prefix", type=int, default=0, metavar="N",
                    help="planned: positions 0..N-1 computed in plaintext (exact fp32 model), their K / V pushed into "
                         "the caches and encrypted as public values; the encrypted decode runs positions N..--tokens")
    ap.add_argument("--window", default="0", help="openwebtext window index (or a .npy of token ids)")
    ap.add_argument("--window-tokens", type=int, default=128, help="the length of the indexed --window")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--device", type=int, default=None,
                    help="GPU index (default: the first of CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--configs", default=None, help="default: the plan's configs= line, else llama31_8b_n32")
    ap.add_argument("--capture", default=None, help="graph dir: capture block graphs at the first position")
    ap.add_argument("--structure-only", action="store_true",
                    help="capture without the per-node magnitude decrypts; fill them with capture_mags.py")
    ap.add_argument("--plan", default=None, help="plan dir: run under block_<b>_placement.json")
    ap.add_argument("--plan-windows", default=None, metavar="B1:DIR1,B2:DIR2,...",
                    help="planned: a plan per position window (DIRi serves the positions below Bi), the plans sharing "
                         "one runtime (only their magnitudes differ); --plan defaults to the first window's")
    ap.add_argument("--packing", choices=("cachemir", "cachemir_complex"), default="cachemir_complex")
    ap.add_argument("--cpp-bootstrap", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--kv-offload", action=argparse.BooleanOptionalAction, default=True,
                    help="park K/V caches in the pinned KV arena (default: resident K/V leave ~16 GB free and the "
                         "decode turns host-bound, ~2.5x slower)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fake", action="store_true", help="capture on the numpy fake (perseus.impl.fakecap): no GPU")
    ap.add_argument("--tap-blocks", default="", help="blocks whose sub-steps are tapped too (comma list)")
    ap.add_argument("--taps", default=None, help="decrypt every block output; save {pos_block: vec} to this .npz")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--no-coeff-rows", action="store_true",
                    help="encode the weight plaintexts from their slot vectors (not the pinned coefficient rows)")
    ap.add_argument("--graphs", action="store_true", help="planned: CUDA graphs per block (default eager replay)")
    ap.add_argument("--key-layout", default="packed", choices=("dense", "packed"))
    ap.add_argument("--stc-cache", type=int, default=8)
    ap.add_argument("--schedule", choices=("program", "siblings", "lockstep"), default="program",
                    help="planned: op order before K-batching (siblings: refresh pairs (PC_BTS_PAIR))")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--debug-blocks", action="store_true", help="planned: decrypt every block output vs the twin")
    ap.add_argument("--tap-block", type=int, default=None, help="planned debug: block whose intermediates (token 0) are compared with the twin")
    ap.add_argument("--tap-steps", default="", help="with --tap-block: every output of these steps (comma list)")
    ap.add_argument("--tap-token", type=int, default=0,
                    help="with --tap-block: the token (position - 1) tapped; the first token of its program class")
    a = ap.parse_args(argv)
    from perseus._paths import resolve
    from perseus.examples.defaults import default_windows
    a.windows = None
    if a.plan_windows:
        from perseus.impl.driver import parse_plan_windows
        a.windows = [(b, resolve(d)) for b, d in parse_plan_windows(a.plan_windows)]
    elif a.plan is None and not (a.fake or a.capture):
        # up to a 512-token context: a plan per 128-position window
        a.windows = [(b, resolve(d)) for b, d in default_windows("llama_decode")]
    if a.windows and a.plan is None:
        a.plan = a.windows[0][1]
    if not a.configs:
        a.configs = DEFAULT_CONFIGS
        from perseus._paths import resolve
        cmd = os.path.join(resolve(a.plan), "PLAN_CMD.txt") if a.plan else ""
        if cmd and os.path.exists(cmd):
            for line in open(cmd).read().splitlines():
                if line.startswith("configs="):
                    a.configs = line[len("configs="):].split()[0]
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
        toks = hf.text_windows(int(a.window) + 1, a.window_tokens)[int(a.window)]
    toks = toks[:a.tokens + 1]
    store = W.SafetensorsStore(hf.NAME)
    cache = os.path.join(os.environ.get("HF_HOME", "."), "perseus", "llama_oracle",
                         f"{store.fingerprint()}_{abs(hash(tuple(int(t) for t in toks))) % 10**12}.npy")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if a.fake else env.gpt2_env.physical_device(a.device)
    gt = None if a.fake else hf.oracle_logits(toks, cache)   # before the session: the fp32 model needs the GPU
    a.plain_kv = None
    if a.plain_prefix:
        # a fake run from a plain prefix decodes (and captures) its first position alone: the magnitudes of a late
        # position at the cost of one
        if not (a.fake or a.plan):
            print("--plain-prefix: planned device runs or fake runs only", file=sys.stderr)
            return 2
        a.plain_kv = hf.prompt_kv(toks, a.plain_prefix, cache[:-4] + f"_pkv{a.plain_prefix}.npz",
                                  device="cpu" if a.fake else "cuda")

    over = {**env.plan_runtime(a.plan, "n32"), "LLAMA_PACKING": a.packing,
            **dict(kv.split("=", 1) for kv in a.set)}
    if a.kv_offload and "KV_ARENA_GB" not in over:
        over["KV_ARENA_GB"] = os.environ.get("LLAMA_KV_ARENA_GB", "56")     # the offloaded K/V of 32 layers
    if a.capture:
        os.environ["FHE_GRAPH_DIR"] = a.capture
        # the first encrypted position is 1 (the BOS prefix is plaintext), or the plain prefix's length
        os.environ.setdefault("FHE_GRAPH_CAPTURE_TOKEN", str(a.plain_prefix or 1))
    cfgs = load_configs(a.configs)
    dims = W.llama_dims(store.config)
    t0 = time.time()
    if not a.fake:
        # the new runtime runs planned only: every block a compiled Program, its per-position stream recorded on
        # the CPU twin (impl.driver.StreamedPlannedRun)
        if not a.plan or a.capture:
            print("the device path runs planned (--plan); capture with --fake", file=sys.stderr)
            return 2
        from perseus.impl import fakecap
        env.export_env(a.device, chain="n32", **{"CKKS_COMPLEX": "1", **over})
        fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
        lw = W.LlamaWeights(store, cfgs, a.configs, dims, a.packing)
        res = _planned(a, toks, gt, lw, cfgs, dims, store)
        kls = sorted(r["kl"] for r in res if "kl" in r)
        hits = sum(1 for r in res if r.get("hit"))
        tok_s = sorted(r["token_s"] for r in res)
        steady = sorted(r["token_s"] for r in res[1:]) or tok_s
        print(f"[llama] SUMMARY positions={len(res)} top1={hits}/{len(res)} "
              f"medianKL={kls[len(kls) // 2] if kls else float('nan'):.4g} maxKL={kls[-1] if kls else float('nan'):.4g} "
              f"median_token_s={tok_s[len(tok_s) // 2]:.3f} steady_median_s={steady[len(steady) // 2]:.3f} "
              f"pcie_gb_per_token={sorted(r['pcie_gb'] for r in res)[len(res) // 2]:.2f} wall={sum(r['token_s'] for r in res):.0f}s")
        return 0
    if a.fake:
        from perseus.impl import fakecap
        env.export_env(a.device, chain="n32", **{"CKKS_COMPLEX": "1", **over})
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
        _, inf = fakecap.make_session(dims, set(env.rot_steps(dims)))      # the GPU session's keys exactly
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
    from perseus import _pc as _core
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
    if os.environ.get("LLAMA_GC_FREEZE", "0") not in ("", "0"):
        import gc
        gc.collect(); gc.freeze(); gc.disable()
    if a.taps:
        model.taps = {}
        model.tap_blocks = tuple(int(b) for b in a.tap_blocks.split(",")) if a.tap_blocks else ()

    def _cuda_profile_range(pos):
        """CUDA_PROFILE_POS=p: the cudaProfiler range covers position p + 1 (for nsys --capture-range)."""
        p = int(os.environ.get("CUDA_PROFILE_POS", "0") or 0)
        if p and pos in (p, p + 1):
            import ctypes
            rt = ctypes.CDLL("libcudart.so")
            (rt.cudaProfilerStart if pos == p else rt.cudaProfilerStop)()

    def on_token(r):
        line = (f"[llama] pos{r['pos']} top1={r['top1']} decode={r['decode_s']:.1f}s bts={r['bootstraps']}")
        if "kl" in r:
            line += f" ref={r['ref']} rank={r['ref_rank']} KL={r['kl']:.4g} {'OK' if r['hit'] else 'MISS'}"
        for k in ("free_gb", "rss_gb", "cpu_s"):
            if k in r:
                line += f" {k}={r[k]}"
        if os.environ.get("LLAMA_KV_LEVELS", "0") not in ("", "0"):
            cts, _ = model.kv[0].cts("")
            line += f" kv0={sorted({(model.rt.ops.level(c), model.rt.ops.noise_deg(c)) for c in cts})}"
        if r.get("masks"):
            m = r["masks"]
            line += f" masks={m.get('adopted', 0)}/{m.get('evicted', 0)}/{m.get('misses', 0)}"
        if os.environ.get("FIDESLIB_POOL_TRACE") and hasattr(_core, "gpu_pool_trace"):
            for site, nb in _core.gpu_pool_trace(6)[:8]:
                line += f"\n  [trace] {nb / 2**30:.2f} GB  {site[:260]}"
        print(line, flush=True)
        _cuda_profile_range(r["pos"])
        if a.profile and hasattr(model.rt.ops, "report_steps"):
            print("[llama] steps: " + " | ".join(t.split("%")[0].strip() for t in model.rt.ops.report_steps(1).split("\n")[:8]), flush=True)
            print("[llama] primitives: " + " | ".join(t.split("%")[0].strip() for t in model.rt.ops.report(10).split("\n")[:10]), flush=True)
            if os.environ.get("LLAMA_PROFILE_STEPS"):
                for st in sorted(model.rt.ops.t_step, key=lambda k: -model.rt.ops.t_step[k]):
                    print(f"[llama] step {st}: " + model.rt.ops.report_step_split(st, 1, 6), flush=True)
            model.rt.ops.reset()

    t1 = time.time()
    start = None
    if a.plain_kv is not None:
        model.plain_prompt = a.plain_kv
        model.start()
        start = a.plain_prefix
    res = model.run_decode(toks, gt if model.n_layers == int(store.config["num_hidden_layers"]) else None,
                           on_token=on_token, start=start)
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
