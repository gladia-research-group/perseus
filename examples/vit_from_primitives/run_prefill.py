"""ViT-B/16 encrypted inference: images -> patch rows (encrypted) -> 12 blocks -> CLS logits (final LN + classifier on
the decrypted CLS row), gated against the float64 model (top-1 / top-5 agreement, KL).

    .venv/bin/python -m examples.vit_from_primitives.run_prefill --images logs/vit/img [--layers 12] [--fake]
        [--capture graphs/X --structure-only] [--plan bootstrap_placements/X] [--check]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

DEFAULT_CONFIGS = "configs/model/approximation/vit_base_n32/configs.json"


def main(argv=None):
    from examples.gpt2_from_primitives.run_decode import gpu_busy
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="logs/vit/img")
    ap.add_argument("--n", type=int, default=None, help="first n images")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--device", type=int, default=3)
    ap.add_argument("--configs", default=None)
    ap.add_argument("--capture", default=None)
    ap.add_argument("--structure-only", action="store_true")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--check", action="store_true", help="compare every block with the float64 model")
    ap.add_argument("--fake", action="store_true", help="the numpy fake (perseus.impl.fakecap): no GPU")
    a = ap.parse_args(argv)
    os.environ.setdefault("KV_ARENA_GB", "0")
    busy = None if a.fake else gpu_busy(a.device)
    if busy and not a.force:
        print(f"GPU {a.device} has other processes: {busy}; refusing (use --force)", file=sys.stderr)
        return 2
    from examples.gpt2_from_primitives import env
    over = {"FIDESLIB_ENCODE_NOSYNC": "0", **env.plan_runtime(a.plan, "n32"), "GPT2_PACKING": "cachemir",
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
    env.export_env(a.device, chain="n32", CKKS_COMPLEX="1", **over)
    if a.fake:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.capture:
        os.environ["FHE_GRAPH_DIR"] = a.capture
    from perseus import _core
    from perseus.impl import config
    from perseus.impl.layout import Dims, ladder_radix_bits, ladder_radix_steps
    from perseus.impl.pcmm import rot_steps as pcmm_rot_steps
    from perseus.impl.blocked import Blocks, blocked_rot_steps
    from perseus.impl.prefill import Chunks, diag_rot_steps, prefill_rot_steps
    from examples.gpt2_from_primitives.prefill import report
    from . import weights as V
    from .prefill import VitPrefill

    d0 = Dims.gpt2()
    extra = prefill_rot_steps(d0) | diag_rot_steps(d0) | (pcmm_rot_steps() if pcmm else set()) \
        | blocked_rot_steps(Blocks(d0, int(os.environ.get("PF_NCH", "512"))), ladder_radix_bits())
    pruned = None
    if a.fake:
        from perseus.impl import fakecap
        fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
        rot = extra | fakecap.GPT2_BAND | (ladder_radix_steps(d0, ladder_radix_bits())
                                           if ladder_radix_bits() >= 2 else set())
        _, inf = fakecap.make_session(d0, rot)
        sess = fakecap.FakeSession(inf)
        core = fakecap.core
    else:
        from perseus.impl.magtape import plan_rot_steps
        pruned = plan_rot_steps(a.plan) if a.plan and os.environ.get("PF_ROT_PRUNE", "1") != "0" else None
        if pruned:
            extra = pruned
        sess = env.open_session(a.device, complex_payload=True, chain="n32", extra_rot_steps=extra, base=not pruned,
                                **over)
        inf = sess.inf
        env.bootstrap_setup(inf.fhe, over)
        core = _core
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
    if a.plan:
        model.load_plans(a.plan, argmax_blocks=False)
    os.environ.setdefault("STEPS_T", "128")
    os.environ["MULTI_T"] = str(V.T)
    files = sorted(glob.glob(os.path.join(a.images, "*")))[:a.n]
    t_pre = time.time()
    model.preload_prefill()
    if pcmm:
        from perseus.impl.pcmm import engine
        engine(model.rt).prepare(d.dim)
    print(f"[vit] T={V.T} layers={model.n_layers} images={len(files)} weights ready {time.time() - t_pre:.1f}s "
          f"capture={a.capture} plan={a.plan}", flush=True)
    W1 = np.asarray(store.r("classifier.weight"))
    names = store.config.get("id2label", {})
    res, walls = [], []
    for f in files:
        rows = V.patch_rows(V.pixels(f))
        ref = V.exact_blocks(store, V.embed(store, rows), 12)
        gt = V.logits(store, ref[-1])[0]
        b0 = [int(inf.fhe.total_bootstraps)]

        def on_block(b, Xs, s, ref=ref):
            tb = int(inf.fhe.total_bootstraps)
            line = f"[vit] block {b} {s:.2f}s bts={tb - b0[0]}"
            b0[0] = tb
            if a.check:
                h = V.unscale(model.hidden(Xs, Chunks(V.T, model.blocks.t)))
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
            model.dbg_embed = lambda h, x0=x0: print(f"[vit] embed maxerr={np.abs(h - x0).max():.3g} "
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
