"""Llama prefill from primitives: BOS + T prompt tokens through the blocks, logits gated against the fp32 model.

    .venv/bin/python -m perseus.examples.llama.run_prefill --fake --tokens 127 --capture graphs/<name>
    GPU_JOB_TIMEOUT=7200 tools/gpu.sh <log> .venv/bin/python -m perseus.examples.llama.run_prefill --tokens 127 \
        --plan bootstrap_placements/llama31_8b_prefill128 [--check]

The device path runs planned only (examples.llama.planned.LlamaPlannedPrefill): the forward's op stream
recorded once on the CPU twin, every block a compiled Program run in Rust, the PCMM linears and the CCMM attention
`Finish` ops served by the session's matmul services.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

DEFAULT_CONFIGS = "configs/model/approximation/llama31_8b_n32/configs.json"


def _plan_config(plan):
    """The session Config of a planned run: the Llama prefill preset, the plan's stamp, its runtime= switches the
    Config has fields for (FIDESlib async / cache levers, e.g. FIDESLIB_PCMM_ASYNC, have no meaning here and
    are listed, not applied: the same split pc-program's `PlanDir::config` makes)."""
    import shlex
    from perseus import _pc, config as pcfg
    cfg = pcfg.preset("n32_llama_prefill")
    stamp = pcfg._plan_stamp(plan)
    if stamp is not None:
        cfg = cfg.overlay(stamp)
    line = pcfg._plan_cmd(plan).get("runtime", "")
    rt, _pl, rest = _pc.Config.split_legacy_env(dict(kv.split("=", 1) for kv in shlex.split(line)))
    if rest:
        print(f"[llama-prefill] runtime switches without a Config field (old-stack levers, not applied): "
              f"{' '.join(sorted(rest))}", flush=True)
    return cfg.overlay(rt)


def _planned(a, model, toks, gt, ref, results=None):
    """PLANNED on the new runtime: record + compile + session, then `--repeat` timed forwards (each: block 0's inputs
    encrypted, every block issued, the device drained); gated on the last one."""
    from perseus import _pc, config as pcfg
    from perseus._paths import resolve
    from .planned import LlamaPlannedPrefill
    from .prefill import plain_logits, report
    cfg = _plan_config(a.plan)
    cfg["exec.use_graphs"] = bool(a.graphs)
    keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
    work = a.work_dir or os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), f"llama_prefill_{os.getpid()}")
    run = LlamaPlannedPrefill(model, resolve(a.plan), cfg, keys, work_dir=work, pt_rows=a.pt_rows,
                              stc_cache=a.stc_cache, check_meta=a.check_meta, log=lambda m: print(m, flush=True),
                              tap_block=a.tap_block, ccmm_kappa=[float(x) for x in a.ccmm_kappa.split(",")],
                              ccmm_sweep=[float(x) for x in a.ccmm_sweep.split(",") if x])
    run.tap_steps = {x for x in a.tap_steps.split(",") if x}
    t_set = time.time()
    if a.compile_only:
        run.prepare(toks)
        progs = [run.progs[lab] for lab in run.labels]
        print(f"[llama-prefill] compiled {len(progs)} blocks: record {run.record_s:.1f}s compile {run.compile_s:.1f}s; "
              f"{sum(p.n_bootstraps for p in progs)} refresh sites, finishes {[p.n_finishes for p in progs][:2]}..., "
              f"slots {max(p.mem[1] for p in progs)}, registers {run.n_regs}, distinct plaintexts {len(run.uniq)}",
              flush=True)
        return 0
    run.prepare(toks)
    if a.cts_alt is not None:
        # a runtime-only lever (same landings: the compiled programs do not change), set after the compile's
        # plan-stamp check
        run.cfg["bts.cts_alt_budget"] = a.cts_alt
    if a.pcmm_batch is not None:
        run.cfg["model.prefill.pcmm_batch"] = a.pcmm_batch      # likewise runtime-only (the finish groups)
    run.setup(toks)
    progs = [run.progs[lab] for lab in run.labels]
    n_bts = sum(p.n_bootstraps for p in progs)
    n_fin = [sum(p.n_finishes[i] for p in progs) for i in (0, 1)]
    free0 = run.fhe.mem_info()[0] / 2 ** 30
    print(f"[llama-prefill] planned setup {time.time() - t_set:.1f}s; {n_bts} refresh sites + {n_fin[1]} PCMM/CCMM "
          f"outputs ({n_fin[0]} finishes) = {n_bts + n_fin[1]} bts; device free {free0:.2f} GB", flush=True)
    # PF_WARMUP (model.prefill.warmup; the llama128 plan sets it): one untimed forward of a different prompt of the
    # same length first (the second openwebtext window of 512, cut to this length, BOS kept) warms the route caches
    # and the services. The timed forwards follow.
    if a.warmup if a.warmup is not None else bool(cfg["model.prefill.warmup"]):
        from . import hf
        wtoks = [int(t) for t in hf.text_windows(2, 512)[1][:len(toks)]]
        if wtoks[0] != int(toks[0]) or wtoks == [int(t) for t in toks]:
            raise RuntimeError("warm-up prompt: expected a different document with the same BOS")
        t_w = time.time()
        run.run(vals=run.input_values(wtoks))
        run.fhe.device_sync()
        print(f"[llama-prefill] warm-up forward (other prompt) {time.time() - t_w:.2f}s", flush=True)
    walls, frees, blk_free, errs = [], [], [], {}
    for k in range(max(1, a.repeat)):
        t0 = time.time()
        info, fr = [], []

        def on_block(b, i):
            info.append(i)
            if ref is not None:     # the registers are reused two blocks later: check now (synchronous)
                h = run.hidden(b)
                n = min(h.shape[0], a.tokens)
                errs[b] = (np.abs(h[:n] - ref[b][:n]).max(1), np.abs(ref[b][:n]).max())
            if a.mem_per_block:
                run.fhe.device_sync()
                fr.append(round(run.fhe.mem_info()[0] / 2 ** 30, 3))
        run.run(on_block=on_block)
        run.fhe.device_sync()
        walls.append(time.time() - t0)
        frees.append(round(run.fhe.mem_info()[0] / 2 ** 30, 3))
        blk_free.append(fr)
        host = sum(i["host_us"] for i in info) / 1e6
        mm = [(b, i["meta_mismatches"], i["first_mismatches"][:2]) for b, i in enumerate(info) if i["meta_mismatches"]]
        if a.check_meta:
            print(f"[llama-prefill] metadata vs compile: {sum(m[1] for m in mm)} mismatches {mm[:3]}", flush=True)
        print(f"[llama-prefill] forward {k}: wall {walls[-1]:.2f}s ({walls[-1] / a.tokens:.3f} s/tok), host issue "
              f"{host:.2f}s, served {''.join(i['served'][0] for i in info)}, free {frees[-1]} GB"
              + (f", per block {min(fr)}..{max(fr)} GB" if fr else ""), flush=True)
    wall = sorted(walls)[len(walls) // 2]
    for b, (err, mx) in sorted(errs.items()):
        print(f"[llama-prefill] block {b} maxerr={err.max():.3g} (pos {int(err.argmax()) + 1}) |x|={mx:.3g}", flush=True)
    if a.tap_block is not None:
        for nm, i, rel, mx, mdev in run.tap_report():
            print(f"[tap] blk{a.tap_block} #{i} {nm}: rel err {rel:.3g} ({np.log2(1 / max(rel, 1e-300)):.1f} bits) "
                  f"max|x| {mx:.4g} device {mdev:.4g}", flush=True)
    out = {"wall": wall}
    n_rep = a.tokens
    if gt is not None and model.n_layers == int(model.w.store.config["num_hidden_layers"]):
        hid = run.hidden()[:a.tokens]
        # PF_LAST_QUERY: the last position alone is computed past the last block's K / V
        from perseus.impl.prefill import Chunks
        lo = a.tokens - 1 if model.last_position(Chunks(a.tokens, model.blocks.t)) is not None else 0
        n_rep = a.tokens - lo
        rows_, s = report(plain_logits(model, hid[lo:]), gt[1 + lo:a.tokens + 1])
        out.update(s)
        print("[llama-prefill] per-position KL " + " ".join(f"{r['kl']:.3g}" for r in rows_), flush=True)
    st = run.sess.pcmm_stats()
    cst = run.sess.ccmm_stats() if run.ccmm else {}
    print(f"[pcmm] calls={st.get('calls')} keygen={st.get('keygen_s', 0):.1f}s linears={st.get('linears')} "
          f"missing={st.get('missing')}; [ccmm] calls={cst.get('calls')} keygen={cst.get('keygen_s', 0):.1f}s; "
          f"planned {run.sess.planned_stats()}", flush=True)
    print(f"[llama-prefill] SUMMARY T={a.tokens} wall={wall:.2f}s per_token={wall / a.tokens:.3f}s "
          f"bts={n_bts + n_fin[1]} top1={out.get('top1')}/{n_rep} medianKL={out.get('medianKL', float('nan')):.4g} "
          f"maxKL={out.get('maxKL', float('nan')):.4g} free={frees}GB (setup {free0:.2f}) "
          f"walls={[round(w, 2) for w in walls]}", flush=True)
    if results is not None:
        results({**out, "walls": walls, "free_gb": frees, "block_free_gb": blk_free, "bts": n_bts + n_fin[1],
                 "pcmm_calls": st.get("calls"), "ccmm_calls": cst.get("calls")})
    run.close()
    return 0


def main(argv=None, results=None):
    from perseus.examples.gpt2.run_decode import gpu_busy
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16, help="prompt positions after the BOS prefix")
    ap.add_argument("--window", default="0")
    ap.add_argument("--seq", type=int, default=128, help="window length (PF_PAGED=1 runs prompts past 128)")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--device", type=int, default=None,
                    help="GPU index (default: the first of CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--configs", default=None)
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
    ap.add_argument("--key-layout", choices=("dense", "packed"), default="packed",
                    help="planned: rotation keys resident (dense) or 28-bit b + seed regenerated per use (packed)")
    ap.add_argument("--pt-rows", type=int, default=28, help="planned: rows per plaintext slot (0 = every prime)")
    ap.add_argument("--stc-cache", type=int, default=8, help="planned: StC-first last-stage cache entries")
    ap.add_argument("--graphs", action="store_true", help="planned: capture each block as CUDA graphs")
    ap.add_argument("--check-meta", action="store_true", help="planned: compare every step's metadata with the compile")
    ap.add_argument("--repeat", type=int, default=1, help="planned: run the forward this many times (wall = median)")
    ap.add_argument("--mem-per-block", action="store_true", help="planned: device free memory after every block")
    ap.add_argument("--work-dir", default=None, help="planned: where the recorded stream is written")
    ap.add_argument("--compile-only", action="store_true", help="planned: record + compile on the CPU, then stop")
    ap.add_argument("--cts-alt", type=int, default=None,
                    help="planned: CtS level budget of the dropped raise variants (0 = the shipped CtS; default: the "
                         "plan's FIDESLIB_CTS_ALT_BUDGET)")
    ap.add_argument("--pcmm-batch", type=int, default=None,
                    help="planned: PCMM finish group size (default: the plan's PF_PCMM_BATCH, else 4)")
    ap.add_argument("--warmup", type=int, default=None, choices=(0, 1),
                    help="planned: an untimed forward of another prompt first (default: the plan's PF_WARMUP)")
    ap.add_argument("--tap-block", type=int, default=None, help="planned: compare this block's intermediates with the twin")
    ap.add_argument("--ccmm-kappa", default="256,16", help="planned: CCMM operating points QK,PV (SinCBoot input |M| x kappa)")
    ap.add_argument("--tap-steps", default="", help="planned: with --tap-block, every op output of these steps")
    ap.add_argument("--ccmm-sweep", default="", help="planned: comma list of kappas: QK / PV probe precision per kappa")
    a = ap.parse_args(argv)
    from perseus.examples.defaults import default_plan
    if a.plan is None and not (a.fake or a.capture):
        a.plan = default_plan("llama_prefill", a.tokens)
    planned = not a.fake
    if planned and (not a.plan or a.capture):
        # the new runtime has no eager device path: a forward is a planned run (its stream recorded on the CPU twin)
        print("the device path runs planned (--plan); capture with --fake", file=sys.stderr)
        return 2
    if not a.configs:
        a.configs = DEFAULT_CONFIGS
        from perseus._paths import resolve
        cmd = os.path.join(resolve(a.plan), "PLAN_CMD.txt") if a.plan else ""
        if cmd and os.path.exists(cmd):
            for line in open(cmd).read().splitlines():
                if line.startswith("configs="):
                    a.configs = line[len("configs="):].split()[0]
    # prefill K / V stay on the device: a pinned KV arena would only lock host memory
    os.environ.setdefault("KV_ARENA_GB", "0")
    busy = None if (a.fake or a.compile_only) else gpu_busy(a.device)
    if busy and not a.force:
        print(f"GPU {a.device} has other processes: {busy}; refusing (use --force)", file=sys.stderr)
        return 2
    from . import env, hf
    from . import weights as W
    from .config import load_configs
    toks = hf.text_windows(int(a.window) + 1, a.seq)[int(a.window)][:a.tokens + 1]
    store = W.SafetensorsStore(hf.NAME)
    cache = os.path.join(os.environ.get("HF_HOME", "."), "perseus", "llama_oracle",
                         f"{store.fingerprint()}_{abs(hash(tuple(int(t) for t in toks))) % 10**12}.npy")
    if a.fake:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
        from perseus._paths import resolve
        cal = os.path.join(resolve(a.plan), "pcmm_calib.json")
        if os.path.exists(cal):
            os.environ["PF_PCMM_CALIB_IN"] = cal
    ccmm_on = os.environ.get("PF_ATTN", "") == "ccmm"
    if pcmm or ccmm_on:
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
    if ccmm_on:
        from perseus.impl.sinc import rot_steps as sinc_rot_steps
        rot |= sinc_rot_steps() | {16, -16}
    if os.environ.get("PF_LAYOUT", "") == "blocked":
        from perseus.impl.blocked import Blocks, blocked_rot_steps
        rot |= blocked_rot_steps(Blocks(dims, int(os.environ.get("PF_NCH", "512"))), ladder_radix_bits())
    t0 = time.time()
    pruned = None
    if planned:
        # the CPU twin: the fake session under the plan's runtime; its PCMM / CCMM finishes land where the Shadow
        # says the device's do (`finish_landing`; the CCMM calibration probe checks it)
        from perseus.impl import fakecap
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        env.export_env(a.device, chain="n32", **{"CKKS_COMPLEX": "1", **over})
        os.environ["CUDA_VISIBLE_DEVICES"] = cvd
        fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
        _, inf = fakecap.make_session(dims, set(env.rot_steps(dims)) | rot)
        sess = fakecap.FakeSession(inf)
    elif a.fake:
        from perseus.impl import fakecap
        env.export_env(a.device, chain="n32", **{"CKKS_COMPLEX": "1", **over})
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
        _, inf = fakecap.make_session(dims, set(env.rot_steps(dims)) | rot)
        sess = fakecap.FakeSession(inf)
    else:
        from perseus.impl.magtape import plan_rot_steps
        pruned = plan_rot_steps(a.plan) if a.plan and os.environ.get("PF_ROT_PRUNE", "1") != "0" else None
        if pruned:          # a planned run needs only its graph's rotation keys (VRAM)
            rot = pruned
            print(f"[llama-prefill] rotation keys from the plan's graph: {len(rot)}", flush=True)
        sess = env.open_session(dims, a.device, complex_payload=True, extra_rot_steps=rot, base=not pruned, **over)
        inf = sess.inf
        env.bootstrap_setup(inf.fhe, over, cpp_bootstrap=a.cpp_bootstrap)
    loaded = {int(s) % dims.N for s in inf.fhe.loaded_rot_steps}
    missing = [] if (a.fake or planned) else sorted((set() if pruned else set(env.rotation_audit(inf.fhe, dims)))
                                       | ({s % dims.N for s in rot} - loaded))
    if missing:
        print(f"rotation keys missing: {missing}", file=sys.stderr)
        return 3
    from perseus import _pc as _core
    from .prefill import LlamaPrefill, exact_hidden, plain_logits, report
    core = fakecap.core if (a.fake or planned) else _core
    lw = W.LlamaWeights(store, cfgs, a.configs, dims, "cachemir_complex",
                        pinned=None if (a.no_coeff_rows or pcmm or a.fake or planned) else _core.pinned_empty)
    model = LlamaPrefill(inf, lw, cfgs, core=core, n_layers=a.layers, packing="cachemir_complex", profile=a.profile)
    if a.capture:
        model.set_capture(a.capture)
        from perseus.impl.magtape import save_rot_steps
        save_rot_steps(a.capture, inf.fhe.loaded_rot_steps)
        if a.structure_only:
            inf.fhe.magnitude_suppressed = True
    if a.plan and not planned:
        model.load_plans(a.plan)
    ref = exact_hidden(model, toks, model.n_layers) if a.check else None
    if planned:
        print(f"[llama-prefill] twin {time.time() - t0:.0f}s T={a.tokens} layers={model.n_layers} plan={a.plan}",
              flush=True)
        rc_ = _planned(a, model, toks, gt, ref, results)
        model.close()
        sess.close()
        return rc_
    print(f"[llama-prefill] session {time.time() - t0:.0f}s T={a.tokens} layers={model.n_layers} "
          f"cts={(a.tokens + 2 * dims.t - 1) // (2 * dims.t)}", flush=True)
    b0 = [int(inf.fhe.total_bootstraps)]

    def on_block(b, Xs, s):
        tb = int(inf.fhe.total_bootstraps)
        line = f"[llama-prefill] block {b} {s:.2f}s bts={tb - b0[0]}"
        if not a.fake and hasattr(_core, "device_free_gb"):
            line += f" free={_core.device_free_gb():.1f}GB"
            if hasattr(inf, "enc_cache_stats"):
                line += f" enc={inf.enc_cache_stats().get('size')}"
            if hasattr(_core, "gpu_pool_stats"):
                line += f" pool={_core.gpu_pool_stats()}"
            if os.environ.get("FIDESLIB_POOL_TRACE") and hasattr(_core, "gpu_pool_trace"):
                for site, nb in _core.gpu_pool_trace(6)[:6]:
                    line += f"\n  [trace] {nb / 2**30:.2f} GB  {site[:300]}"
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
        from perseus.examples.gpt2.prefill import tap_rows
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
    if ccmm_on:
        from perseus.impl.ccmm import engine as ccmm_engine
        t_pre = time.time()
        ce = ccmm_engine(model.rt)
        ce.prepare(dims.dim)
        print(f"[llama-prefill] ccmm keys + calibration {time.time() - t_pre:.1f}s c_unit={ce.c_unit} land={ce.land}",
              flush=True)
    t1 = time.time()
    if os.environ.get("PF_PAGED", "0") not in ("", "0"):
        def on_page_block(p, b, Xs, ch, s):
            tb = int(inf.fhe.total_bootstraps)
            line = f"[llama-prefill] page {p} block {b} {s:.2f}s bts={tb - b0[0]}"
            if not a.fake and hasattr(_core, "device_free_gb"):
                line += f" free={_core.device_free_gb():.1f}GB"
            b0[0] = tb
            if ref is not None:
                h = model.hidden_rows(Xs, ch)
                lo = ch.base - 1
                n = min(ch.T, a.tokens - lo)
                err = np.abs(h[:n] - ref[b][lo:lo + n]).max(1)
                line += f" maxerr={err.max():.3g} (pos {int(err.argmax()) + 1 + lo}) |x|={np.abs(ref[b][lo:lo + n]).max():.3g}"
            print(line, flush=True)
        pages, rec = model.prefill_paged(toks, on_block=on_page_block)
        wall = time.time() - t1
        hid = np.concatenate([model.hidden_rows(X, c) for X, c in pages])[:a.tokens]
    else:
        Xs, ch, rec = model.prefill(toks, on_block=on_block)
        wall = time.time() - t1
        hid = None
    pc = getattr(model.rt, "pcmm", None)
    if pc is not None:
        print(f"[pcmm] calls={pc.calls} cap={pc.cap} land={pc.land} keygen={pc.keygen_s:.1f}s split->out ms: "
              + " ".join(f"{k}={v:.0f}" for k, v in zip(("rescale", "ring_switch", "mlwe", "gemm", "pack", "back"),
                                                        pc.times))
              + f" | stc_in={pc.t_in:.2f}s pcmm={pc.t_mm:.2f}s finish={pc.t_out:.2f}s", flush=True)
    ce = getattr(model.rt, "ccmm", None)
    if ce is not None:
        print(f"[ccmm] calls={ce.calls} cap={ce.cap} land={ce.land} in={ce.t_in:.2f}s mm={ce.t_mm:.2f}s "
              f"out={ce.t_out:.2f}s C++ ms " + " ".join(f"{v:.0f}" for v in ce.times), flush=True)
    out = {}
    n_rep = a.tokens
    if model.n_layers == int(store.config["num_hidden_layers"]) and gt is not None:
        h = model.hidden_rows(Xs, ch) if hid is None else hid
        lo = a.tokens - 1 if hid is None and model.last_position(ch) is not None else 0   # PF_LAST_QUERY
        n_rep = a.tokens - lo
        rows, out = report(plain_logits(model, h[lo:]), gt[1 + lo:a.tokens + 1])
        print("[llama-prefill] per-position KL " + " ".join(f"{r['kl']:.3g}" for r in rows), flush=True)
    print(f"[llama-prefill] SUMMARY T={a.tokens} wall={wall:.1f}s per_token={wall / a.tokens:.3f}s "
          f"bts={int(inf.fhe.total_bootstraps)} top1={out.get('top1')}/{n_rep} "
          f"medianKL={out.get('medianKL', float('nan')):.4g} maxKL={out.get('maxKL', float('nan')):.4g} "
          f"free={rec.get('free_gb')}GB", flush=True)
    if a.profile and hasattr(model.rt.ops, "report"):
        print("[llama-prefill] time split:\n" + model.rt.ops.report())
        print("[llama-prefill] per step:\n" + model.rt.ops.report_steps(1))
        for st in sorted(model.rt.ops.t_step, key=lambda k: -model.rt.ops.t_step[k])[:8]:
            print(f"[llama-prefill] split {st}: " + model.rt.ops.report_step_split(st))
    model.close()
    sess.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
