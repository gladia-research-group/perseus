"""ViT-B/16 encrypted inference: images -> patch rows (encrypted) -> 12 blocks -> CLS logits (final LN + classifier on
the decrypted CLS row), gated against the float64 model (top-1 / top-5 agreement, KL).

    .venv/bin/python -m perseus.examples.vit.run_prefill --fake --images logs/vit/img [--layers 12]
        [--capture graphs/X --structure-only] [--plan bootstrap_placements/X] [--check]
    GPU_JOB_TIMEOUT=7200 tools/gpu.sh <log> .venv/bin/python -m perseus.examples.vit.run_prefill \
        --images <dir of .npy / images> --n 4 [--plan bootstrap_placements/vit_b16_224] [--check]

The device path runs planned only (examples.vit.planned.VitPlannedPrefill): the op stream recorded once on the CPU
twin (any image: the stream depends on the token count only), every block a compiled Program run in Rust, the PCMM
linears and the CCMM attention `Finish` ops served by the session's matmul services. Inputs: images (PIL) or
pre-normalized (224, 224, 3) .npy arrays.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

DEFAULT_CONFIGS = "configs/model/approximation/vit_base_n32/configs.json"


def _pixels(V, path):
    return np.load(path) if path.endswith(".npy") else V.pixels(path)


def _planned(a, model, store, files, names, W1, results=None):
    """PLANNED on the new runtime: record + compile + session once, then one timed forward per image (block 0's
    inputs encrypted, every block issued, the device drained); s/image = the median of the image walls."""
    from perseus import _pc
    from perseus._paths import resolve
    from perseus.examples.bert.planned import plan_config
    from perseus.examples.gpt2.prefill import report
    from . import weights as V
    from .planned import VitPlannedPrefill
    cfg = plan_config(a.plan, a.graphs)
    keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
    work = a.work_dir or os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), f"m10_vit_{os.getpid()}")
    run = VitPlannedPrefill(model, resolve(a.plan), cfg, keys, work_dir=work, pt_rows=a.pt_rows,
                            stc_cache=a.stc_cache, check_meta=a.check_meta, log=lambda m: print(m, flush=True),
                            ccmm_kappa=[float(x) for x in a.ccmm_kappa.split(",")],
                            ccmm_sweep=[float(x) for x in a.ccmm_sweep.split(",") if x])
    rows0 = V.patch_rows(_pixels(V, files[0]))
    t_set = time.time()
    run.prepare(rows0)
    progs = [run.progs[lab] for lab in run.labels]
    n_bts = sum(p.n_bootstraps for p in progs)
    n_fin = [sum(p.n_finishes[i] for p in progs) for i in (0, 1)]
    print(f"[vit] compiled {len(progs)} blocks: record {run.record_s:.1f}s compile {run.compile_s:.1f}s; {n_bts} refresh "
          f"sites + {n_fin[1]} PCMM/CCMM outputs ({n_fin[0]} finishes) = {n_bts + n_fin[1]} bts, registers {run.n_regs}",
          flush=True)
    if a.compile_only:
        return 0
    run.setup(rows0)
    free0 = run.fhe.mem_info()[0] / 2 ** 30
    print(f"[vit] planned setup {time.time() - t_set:.1f}s; device free {free0:.2f} GB", flush=True)
    res, walls, frees = [], [], []
    for f in files:
        rows = V.patch_rows(_pixels(V, f))
        vals = run.input_values(rows)
        ref = V.exact_blocks(store, V.embed(store, rows), 12)
        gt = V.logits(store, ref[-1])[0]
        info = []
        t0 = time.time()
        run.run(on_block=lambda b, i: info.append(i), vals=vals)
        run.fhe.device_sync()
        walls.append(time.time() - t0)
        frees.append(round(run.fhe.mem_info()[0] / 2 ** 30, 2))
        host = sum(i["host_us"] for i in info) / 1e6
        mm = sum(i["meta_mismatches"] for i in info)
        if a.check:
            for b in range(model.n_layers):
                h = V.unscale(run.hidden(b))
                err = np.abs(h - ref[b]).max(1)
                print(f"[vit] block {b} maxerr={err.max():.3g} (tok {int(err.argmax())}) |x|={np.abs(ref[b]).max():.3g}",
                      flush=True)
        line = f"[vit] {os.path.basename(f)}: wall={walls[-1]:.2f}s host issue {host:.2f}s free {frees[-1]} GB" + \
            (f" metadata mismatches {mm}" if a.check_meta else "")
        if model.n_layers == 12:
            lg = V.logits(store, V.unscale(run.hidden()))[0]
            _, s = report([lg], [gt])
            top5 = set(np.argsort(-gt)[:5].tolist())
            r = dict(file=os.path.basename(f), hit=s["top1"] == 1, kl=s["medianKL"], top5=int(np.argmax(lg)) in top5,
                     ours=names.get(str(int(np.argmax(lg)))), ref=names.get(str(int(np.argmax(gt)))))
            res.append(r)
            line += f" ours={r['ours']} ref={r['ref']} top1={'OK' if r['hit'] else 'MISS'} in_top5={r['top5']} " \
                    f"KL={r['kl']:.3g}"
        print(line, flush=True)
    stp, stc = run.sess.pcmm_stats(), (run.sess.ccmm_stats() if run.ccmm else {})
    print(f"[pcmm] calls={stp.get('calls')} linears={stp.get('linears')} missing={stp.get('missing')}; "
          f"[ccmm] calls={stc.get('calls')}; planned {run.sess.planned_stats()}", flush=True)
    kls = sorted(r["kl"] for r in res)
    print(f"[vit] SUMMARY images={len(files)} s/image={np.median(walls):.2f} (walls {' '.join(f'{w:.2f}' for w in walls)}) "
          f"bts/image={n_bts + n_fin[1]} top1={sum(r['hit'] for r in res)}/{len(res)} "
          f"top5={sum(r['top5'] for r in res)}/{len(res)} medianKL={kls[len(kls) // 2] if kls else float('nan'):.4g} "
          f"maxKL={kls[-1] if kls else float('nan'):.4g} free={frees}GB (setup {free0:.2f})", flush=True)
    if results is not None:
        results({"walls": walls, "s_per_image": float(np.median(walls)), "top1": sum(r["hit"] for r in res),
                 "top5": sum(r["top5"] for r in res), "n": len(res), "kls": kls, "free_gb": frees,
                 "bts": n_bts + n_fin[1]})
    run.close()
    return 0


def main(argv=None, results=None):
    from perseus.examples.gpt2.run_decode import gpu_busy
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="logs/vit/img")
    ap.add_argument("--n", type=int, default=None, help="first n images")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--device", type=int, default=None,
                    help="GPU index (default: the first of CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--configs", default=None)
    ap.add_argument("--capture", default=None)
    ap.add_argument("--structure-only", action="store_true")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--graphs", action="store_true", help="planned: replay each block as CUDA graphs")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--check", action="store_true", help="compare every block with the float64 model")
    ap.add_argument("--fake", action="store_true", help="the numpy fake (perseus.impl.fakecap): no GPU")
    ap.add_argument("--key-layout", choices=("dense", "packed"), default="packed",
                    help="planned: rotation keys resident (dense) or 28-bit b + seed regenerated per use (packed)")
    ap.add_argument("--pt-rows", type=int, default=28, help="planned: rows per plaintext slot (0 = every prime)")
    ap.add_argument("--stc-cache", type=int, default=16, help="planned: StC-first last-stage cache entries")
    ap.add_argument("--check-meta", action="store_true", help="planned: compare every step's metadata with the compile")
    ap.add_argument("--work-dir", default=None, help="planned: where the recorded stream is written")
    ap.add_argument("--compile-only", action="store_true", help="planned: record + compile on the CPU, then stop")
    ap.add_argument("--ccmm-kappa", default="256,16", help="planned: CCMM operating points QK,PV (SinCBoot input x kappa)")
    ap.add_argument("--ccmm-sweep", default="", help="planned: comma list of kappas: QK / PV probe precision per kappa")
    a = ap.parse_args(argv)
    from perseus.examples.defaults import default_plan
    if a.plan is None and not (a.fake or a.capture):
        a.plan = default_plan("vit")
    planned = not a.fake
    if planned and (not a.plan or a.capture):
        # the new runtime has no eager device path: a forward is a planned run (its stream recorded on the CPU twin)
        print("the device path runs planned (--plan); capture with --fake", file=sys.stderr)
        return 2
    os.environ.setdefault("KV_ARENA_GB", "0")
    busy = None if (a.fake or a.compile_only) else gpu_busy(a.device)
    if busy and not a.force:
        print(f"GPU {a.device} has other processes: {busy}; refusing (use --force)", file=sys.stderr)
        return 2
    from perseus._paths import resolve
    from perseus.examples.gpt2 import env
    if a.plan:
        a.plan = resolve(a.plan)
    # the sync-free bundle + power-of-two StC-first cache keys (Llama prefill's, 2026-10-18: 129.4 vs 132.2-134.1
    # s/image here); the shell overrides each, the plan runtime and --set win after them
    fast = {k: os.environ.get(k, v) for k, v in (("FIDESLIB_ENCODE_NOSYNC", "1"), ("FIDESLIB_CCMM_ASYNC", "1"),
                                                 ("FIDESLIB_PCMM_NOSYNC", "1"), ("FIDESLIB_ENCODE_PINNED_RING", "256"),
                                                 ("FIDESLIB_STC_POW2_KEY", "1"))}
    over = {**fast, **env.plan_runtime(a.plan, "n32"), "GPT2_PACKING": "cachemir",
            **dict(kv.split("=", 1) for kv in a.set)}
    for k, v in over.items():
        if k.startswith("PF_"):
            os.environ.setdefault(k, str(v))
    if not a.configs:
        cmd = os.path.join(a.plan, "PLAN_CMD.txt") if a.plan else ""
        a.configs = DEFAULT_CONFIGS
        if cmd and os.path.exists(cmd):
            for line in open(cmd).read().splitlines():
                if line.startswith("configs="):
                    a.configs = line[len("configs="):].split()[0]
    pcmm = os.environ.get("PF_LINEAR", "") == "pcmm"
    if pcmm and a.plan and not os.environ.get("PF_PCMM_CALIB_IN"):
        cal = os.path.join(a.plan, "pcmm_calib.json")
        if os.path.exists(cal):
            os.environ["PF_PCMM_CALIB_IN"] = cal
    if pcmm:
        over = {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
                "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY": os.environ.get("PF_PCMM_STAGED_ONLY", "1"), **over}
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    env.export_env(a.device, chain="n32", CKKS_COMPLEX="1", **over)
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if a.fake else cvd
    if a.capture:
        os.environ["FHE_GRAPH_DIR"] = a.capture
    from perseus.impl import config
    from perseus.impl.layout import Dims, ladder_radix_bits, ladder_radix_steps
    from perseus.impl.pcmm import rot_steps as pcmm_rot_steps
    from perseus.impl.blocked import Blocks, blocked_rot_steps
    from perseus.impl.prefill import Chunks, diag_rot_steps, prefill_rot_steps
    from perseus.examples.gpt2.prefill import report
    from . import weights as V
    from .prefill import VitPrefill

    d0 = Dims.gpt2()
    extra = prefill_rot_steps(d0) | diag_rot_steps(d0) | (pcmm_rot_steps() if pcmm else set()) \
        | blocked_rot_steps(Blocks(d0, int(os.environ.get("PF_NCH", "512"))), ladder_radix_bits())
    ccmm_on = os.environ.get("PF_ATTN", "") == "ccmm"
    if ccmm_on:
        from perseus.impl import sinc_mha
        extra |= sinc_mha.rot_steps(2) | {16, -16}
    # the fake session: the capture (--fake) or the planned run's CPU twin
    from perseus.impl import fakecap
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
    rot = extra | fakecap.GPT2_BAND | (ladder_radix_steps(d0, ladder_radix_bits())
                                       if ladder_radix_bits() >= 2 else set())
    _, inf = fakecap.make_session(d0, rot)
    sess = fakecap.FakeSession(inf)
    core = fakecap.core
    store = V.VitStore()
    cfgs = config.load_configs(a.configs)
    model = VitPrefill(inf, store, cfgs, core=core, n_layers=a.layers, profile=a.profile)
    d = model.rt.dims
    if a.capture:
        model.set_capture(a.capture)
        from perseus.impl.magtape import save_rot_steps
        save_rot_steps(a.capture, inf.fhe.loaded_rot_steps)
        if a.structure_only:
            inf.fhe.magnitude_suppressed = True
    if a.plan and a.fake:
        model.load_plans(a.plan, argmax_blocks=False)
    os.environ.setdefault("STEPS_T", "128")
    os.environ["MULTI_T"] = str(V.T)
    files = sorted(glob.glob(os.path.join(a.images, "*")))[:a.n]
    t_pre = time.time()
    model.preload_prefill()
    if pcmm:
        from perseus.impl.pcmm import engine
        engine(model.rt).prepare(d.dim)
    if ccmm_on:
        from perseus.impl.ccmm import engine as ccmm_engine
        ce = ccmm_engine(model.rt)
        ce.prepare(d.dim)
        print(f"[vit] ccmm keys + calibration c_unit={ce.c_unit} land={ce.land}", flush=True)
    print(f"[vit] T={V.T} layers={model.n_layers} images={len(files)} weights ready {time.time() - t_pre:.1f}s "
          f"capture={a.capture} plan={a.plan}", flush=True)
    W1 = np.asarray(store.r("classifier.weight"))
    names = store.config.get("id2label", {})
    if planned:
        rc = _planned(a, model, store, files, names, W1, results)
        model.close()
        sess.close()
        return rc
    res, walls = [], []
    for f in files:
        rows = V.patch_rows(_pixels(V, f))
        ref = V.exact_blocks(store, V.embed(store, rows), 12)
        gt = V.logits(store, ref[-1])[0]
        b0 = [int(inf.fhe.total_bootstraps)]

        def on_block(b, Xs, s, ref=ref):
            tb = int(inf.fhe.total_bootstraps)
            line = f"[vit] block {b} {s:.2f}s bts={tb - b0[0]}"
            b0[0] = tb
            if a.check:
                h = V.unscale(model.hidden(Xs, model.last_ch))
                err = np.abs(h - ref[b]).max(1)
                line += f" maxerr={err.max():.3g} (tok {int(err.argmax())}) |x|={np.abs(ref[b]).max():.3g}"
            print(line, flush=True)
        if a.check and os.environ.get("VIT_TAPS"):
            x0 = V.embed(store, rows)
            tb_ = int(os.environ["VIT_TAPS"])
            want = V.block0_taps(store, x0 if tb_ == 0 else ref[tb_ - 1], tb_)
            model.tap_block = tb_

            def tap(name, got, want=want):
                w_ = want[name]
                g = np.asarray(got)[:w_.shape[0], :w_.shape[1]]
                c = float(np.vdot(w_, g) / np.vdot(w_, w_))
                e = np.abs(g / c - w_).max(1)
                tk = int(e.argmax())
                print(f"[tap] {name}: tok {tk} err {e[tk]:.3g} |x_tok| {np.abs(w_[tk]).max():.3g} | tok18 err {e[18]:.3g} "
                      f"|x18| {np.abs(w_[18]).max():.3g}", flush=True)
                print(f"[tap] {name}: scale {c:.4g} maxerr {e.max():.3g} (tok {int(e.argmax())}) "
                      f"median {np.median(e):.3g} |x| {np.abs(w_).max():.3g} (u_max {want['u_max']:.3g})", flush=True)
            if os.environ.get("VIT_TAPS"):
                model.tap = tap
            model.dbg_embed = lambda h, x0=x0: print(f"[vit] embed maxerr={np.abs(h[:len(x0)] - x0).max():.3g} "
                                                     f"|x|={np.abs(x0).max():.3g}", flush=True)
        t0 = time.time()
        Xs, ch, rec = model.prefill(rows, on_block=on_block)
        walls.append(time.time() - t0)
        if model.n_layers == 12:
            lg = V.logits(store, V.unscale(model.hidden(Xs, ch)))[0]
            _, s = report([lg], [gt])
            top5 = set(np.argsort(-gt)[:5].tolist())
            r = dict(file=os.path.basename(f), hit=s["top1"] == 1, kl=s["medianKL"], top5=int(np.argmax(lg)) in top5,
                     ours=names.get(str(int(np.argmax(lg)))), ref=names.get(str(int(np.argmax(gt)))))
            res.append(r)
            print(f"[vit] {r['file']}: ours={r['ours']} ref={r['ref']} top1={'OK' if r['hit'] else 'MISS'} "
                  f"in_top5={r['top5']} KL={r['kl']:.3g} wall={walls[-1]:.1f}s", flush=True)
    ce = getattr(model.rt, "ccmm", None)
    if ce is not None:
        print(f"[ccmm] calls={ce.calls} cap={ce.cap} land={ce.land} in={ce.t_in:.2f}s mm={ce.t_mm:.2f}s "
              f"out={ce.t_out:.2f}s", flush=True)
    pc = getattr(model.rt, "pcmm", None)
    if pc is not None:
        print(f"[pcmm] calls={pc.calls} cap={pc.cap} land={pc.land} | stc_in={pc.t_in:.2f}s pcmm={pc.t_mm:.2f}s "
              f"finish={pc.t_out:.2f}s", flush=True)
    kls = sorted(r["kl"] for r in res)
    print(f"[vit] SUMMARY images={len(files)} s/image={np.median(walls):.1f} (walls {' '.join(f'{w:.1f}' for w in walls)}) "
          f"bts/image={int(inf.fhe.total_bootstraps) // max(1, len(files))} top1={sum(r['hit'] for r in res)}/{len(res)} "
          f"top5={sum(r['top5'] for r in res)}/{len(res)} medianKL={kls[len(kls) // 2] if kls else float('nan'):.4g} "
          f"maxKL={kls[-1] if kls else float('nan'):.4g}", flush=True)
    if a.profile and hasattr(model.rt.ops, "report"):
        print("[vit] time split:\n" + model.rt.ops.report())
        print("[vit] per step:\n" + model.rt.ops.report_steps(1))
    model.close()
    sess.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
