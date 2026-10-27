"""GPT-2 prefill from primitives: T prompt positions through the blocks, logits gated against the oracle.

    .venv/bin/python -m perseus.examples.gpt2.run_prefill --fake --tokens 64 --capture graphs/<name>
    tools/gpu.sh <log> .venv/bin/python -m perseus.examples.gpt2.run_prefill --tokens 128 \
        --plan bootstrap_placements/gpt2_prefill128 --check

The device path runs planned only (impl.prefill.PlannedPrefill): the forward's op stream recorded once on the CPU
twin, every block a compiled Program run in Rust, the PCMM linears served by the session's PCMM engine.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from .run_decode import gpu_busy


def _planned(a, model, store, rows, gt, ref, results=None):
    """PLANNED on the new runtime (impl.prefill.PlannedPrefill): record + compile + session, then `--repeat` timed
    forwards (each: block 0's inputs encrypted, every block issued, the device drained); gated on the last one."""
    from perseus import _pc, config as pcfg
    from perseus._paths import resolve
    from perseus.impl.prefill import PlannedPrefill
    from .prefill import plain_logits, report
    cfg = pcfg.plan_runtime(a.plan, "n32", base=pcfg.preset("n32_gpt2_prefill"))
    cfg["exec.use_graphs"] = bool(a.graphs)
    keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
    work = a.work_dir or os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), f"m8_prefill_{os.getpid()}")
    run = PlannedPrefill(model, resolve(a.plan), cfg, keys, work_dir=work, pt_rows=a.pt_rows, stc_cache=a.stc_cache,
                         check_meta=a.check_meta, log=lambda m: print(m, flush=True), schedule=a.schedule)
    t_set = time.time()
    if a.compile_only:
        run.prepare(rows)
        progs = [run.progs[lab] for lab in run.labels]
        print(f"[prefill] compiled {len(progs)} blocks: record {run.record_s:.1f}s compile {run.compile_s:.1f}s; "
              f"{sum(p.n_bootstraps for p in progs)} refresh sites, {sum(p.n_launches for p in progs)} launches, "
              f"registers {run.n_regs}", flush=True)
        for lab in run.labels[:2]:
            print(f"[prefill] {lab} refresh batching (k_max, launches, sizes, per route): {run.progs[lab].bts_batches}",
                  flush=True)
        return 0
    run.setup(rows)
    progs = [run.progs[lab] for lab in run.labels]
    n_bts = sum(p.n_bootstraps for p in progs)
    n_fin = [sum(p.n_finishes[i] for p in progs) for i in (0, 1)]
    free0 = run.fhe.mem_info()[0] / 2 ** 30
    print(f"[prefill] planned setup {time.time() - t_set:.1f}s; {n_bts} refresh sites + {n_fin[1]} PCMM outputs "
          f"({n_fin[0]} linears) = {n_bts + n_fin[1]} bts; device free {free0:.2f} GB", flush=True)
    # GPT2_PF_WARMUP (the old driver's semantics, default 1; 0 = time the cold first pass, "same" = warm on the
    # timed prompt): one untimed forward of another prompt of the same length first (the same tokens with positions
    # 1.. reversed, the position embeddings kept in place), then the `--repeat` timed forwards
    warm = os.environ.get("GPT2_PF_WARMUP", "1")
    if warm not in ("", "0"):
        from . import weights
        wpe = np.asarray(store.tensor(weights.WPE)[:a.tokens])
        tok = np.asarray(rows) - wpe
        if warm != "same":
            tok[1:] = tok[1:][::-1].copy()
        t_w = time.time()
        run.run(vals=run.input_values(tok + wpe))
        run.fhe.device_sync()
        print(f"[prefill] warm-up forward (other prompt) {time.time() - t_w:.2f}s", flush=True)
    walls, frees = [], []
    for k in range(max(1, a.repeat)):
        t0 = time.time()
        info = []
        regs = run.run(on_block=lambda b, i: info.append(i))
        run.fhe.device_sync()
        walls.append(time.time() - t0)
        frees.append(round(run.fhe.mem_info()[0] / 2 ** 30, 2))
        host = sum(i["host_us"] for i in info) / 1e6
        mm = [(b, i["meta_mismatches"], i["first_mismatches"][:2]) for b, i in enumerate(info) if i["meta_mismatches"]]
        if a.check_meta:
            print(f"[prefill] metadata vs compile: {sum(m[1] for m in mm)} mismatches {mm[:3]}", flush=True)
        print(f"[prefill] forward {k}: wall {walls[-1]:.2f}s ({walls[-1] / a.tokens:.3f} s/tok), host issue {host:.2f}s, "
              f"served {''.join(i['served'][0] for i in info)}, free {frees[-1]} GB", flush=True)
    wall = sorted(walls)[len(walls) // 2]
    if ref is not None:
        for b in range(model.n_layers):
            h = run.hidden(b)
            err = np.abs(h - ref[b]).max(1)
            print(f"[prefill] block {b} maxerr={err.max():.3g} (pos {int(err.argmax())}) |x|={np.abs(ref[b]).max():.3g}",
                  flush=True)
    h = run.hidden()
    out = {"wall": wall}
    if model.n_layers == 12:
        rows_, s = report(plain_logits(store, h[-1:]), gt[-1:]) if a.tail else report(plain_logits(store, h), gt)
        out.update(s)
        print("[prefill] per-position KL " + " ".join(f"{r['kl']:.3g}" for r in rows_), flush=True)
    st = run.sess.pcmm_stats() if n_fin[0] else {}
    print(f"[pcmm] calls={st.get('calls')} keygen={st.get('keygen_s', 0):.1f}s linears={st.get('linears')} "
          f"missing={st.get('missing')}; planned {run.sess.planned_stats()}", flush=True)
    print(f"[prefill] SUMMARY T={a.tokens} wall={wall:.2f}s per_token={wall / a.tokens:.3f}s bts={n_bts + n_fin[1]} "
          f"top1={out.get('top1')}/{a.tokens} medianKL={out.get('medianKL', float('nan')):.4g} "
          f"maxKL={out.get('maxKL', float('nan')):.4g} free={frees}GB (setup {free0:.2f}) walls={[round(w, 2) for w in walls]} "
          f"warmup={warm}",
          flush=True)
    ok = model.n_layers != 12 or (out["top1"] >= 0.9 * (1 if a.tail else a.tokens) and out["medianKL"] < 0.1)
    print("[prefill] PASS" if ok else "[prefill] FAIL")
    if results is not None:
        results({**out, "walls": walls, "free_gb": frees, "bts": n_bts + n_fin[1], "pcmm_calls": st.get("calls")})
    run.close()
    return 0 if ok else 1


def main(argv=None, results=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--device", type=int, default=None,
                    help="GPU index (default: the first of CUDA_VISIBLE_DEVICES)")
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
    ap.add_argument("--fake", action="store_true", help="capture on the numpy fake (perseus.impl.fakecap): no GPU")
    ap.add_argument("--random-ids", type=int, default=None, metavar="SEED",
                    help="a random prompt (wte[ids] + wpe), gated against the float64 model instead of the oracle "
                         "(any T up to the 1024 positions)")
    ap.add_argument("--key-layout", choices=("dense", "packed"), default="packed",
                    help="planned: rotation keys resident (dense) or 28-bit b + seed regenerated per use (packed)")
    ap.add_argument("--pt-rows", type=int, default=28, help="planned: rows per plaintext slot (0 = every prime)")
    ap.add_argument("--stc-cache", type=int, default=16, help="planned: StC-first last-stage cache entries")
    ap.add_argument("--graphs", action="store_true", help="planned: capture each block as CUDA graphs (exec.use_graphs; "
                                                          "a one-shot forward only pays the capture)")
    ap.add_argument("--check-meta", action="store_true", help="planned: compare every step's metadata with the compile")
    ap.add_argument("--tail", action="store_true", help="the last position's encrypted tail as two more stages (ln_f + "
                                                        "LM head, then CutMax), gated on its decrypted logits")
    ap.add_argument("--repeat", type=int, default=1, help="planned: run the forward this many times (wall = median)")
    ap.add_argument("--work-dir", default=None, help="planned: where the recorded stream is written")
    ap.add_argument("--compile-only", action="store_true", help="planned: record + compile on the CPU, then stop")
    ap.add_argument("--schedule", choices=("siblings", "program", "lockstep"), default="siblings",
                    help="planned: op order before K-batching (siblings: refresh pairs (PC_BTS_PAIR) and pulled sibling "
                         "batches; lockstep: more batched refreshes)")
    a = ap.parse_args(argv)
    from perseus.examples.defaults import default_plan
    if a.plan is None and not (a.fake or a.capture):
        a.plan = default_plan("gpt2_prefill", a.tokens)

    # prefill K / V stay on the device: a pinned KV arena would only lock host memory
    os.environ.setdefault("KV_ARENA_GB", "0")
    busy = None if (a.fake or a.compile_only) else gpu_busy(a.device)
    if busy and not a.force:
        print(f"GPU {a.device} has other processes: {busy}; refusing (use --force)", file=sys.stderr)
        return 2
    from . import env
    # the fastest measured config is the default: sync-free encodes / PCMM / CCMM + StC-first power-of-two cache keys
    # (2026-10-18, prefill-128 45.9 -> 40.5 s, same top-1 / KL; it replaces the host wait on device-encoded plaintexts
    # that FIDESlib f12c06e had restored for this path). The shell or --set overrides each (FIDESLIB_ENCODE_NOSYNC=0:
    # the wait).
    fast = {"FIDESLIB_ENCODE_NOSYNC": "1", "FIDESLIB_CCMM_ASYNC": "1", "FIDESLIB_PCMM_NOSYNC": "1",
            "FIDESLIB_ENCODE_PINNED_RING": "256", "FIDESLIB_STC_POW2_KEY": "1"}
    over = {**{k: os.environ.get(k, v) for k, v in fast.items()}, **env.plan_runtime(a.plan, "n32"),
            "GPT2_PACKING": "cachemir", **dict(kv.split("=", 1) for kv in a.set)}
    for k, v in over.items():          # a plan's PF_* switches (the arm it was made for); the shell's win
        if k.startswith("PF_"):
            os.environ.setdefault(k, str(v))
    pcmm = os.environ.get("PF_LINEAR", "") == "pcmm"
    if pcmm and a.plan and not os.environ.get("PF_PCMM_CALIB_IN"):   # a PCMM plan carries its calibration
        from perseus._paths import resolve
        cal = os.path.join(resolve(a.plan), "pcmm_calib.json")
        if os.path.exists(cal):
            os.environ["PF_PCMM_CALIB_IN"] = cal
    if pcmm:   # StC-first on the dense route, for the PCMM only
        over = {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
                "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY": os.environ.get("PF_PCMM_STAGED_ONLY", "1"), **over}
    env.export_env(a.device, chain="n32", CKKS_COMPLEX="1", **over)
    if a.fake:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if not a.configs:   # the prefill keeps the composite GELU its plans were cut for
        a.configs = env.plan_configs(a.plan, "n32", env.CONFIGS_N32_LEGACY)
    if a.capture:
        os.environ["FHE_GRAPH_DIR"] = a.capture
    from perseus.impl import config
    from perseus.impl.layout import Dims
    from perseus.impl.pcmm import rot_steps as pcmm_rot_steps
    blocked_steps = set()
    if os.environ.get("PF_LAYOUT", "") == "blocked":
        from perseus.impl.blocked import Blocks, blocked_rot_steps
        from perseus.impl.layout import ladder_radix_bits
        blocked_steps = blocked_rot_steps(Blocks(Dims.gpt2(), int(os.environ.get("PF_NCH", "512"))), ladder_radix_bits())
    from perseus.impl.prefill import diag_rot_steps, prefill_rot_steps
    from . import weights
    from .prefill import Gpt2Prefill, exact_blocks, plain_logits, report

    extra = prefill_rot_steps(Dims.gpt2()) | diag_rot_steps(Dims.gpt2()) | (pcmm_rot_steps() if pcmm else set()) \
        | blocked_steps
    if not a.fake and (not a.plan or a.capture):
        # the new runtime has no eager device path: a forward is a planned run (its stream recorded on the CPU twin)
        print("the device path runs planned (--plan); capture with --fake", file=sys.stderr)
        return 2
    # the fake session: the capture (--fake) or the planned run's CPU twin
    from perseus.impl import fakecap
    from perseus.impl.layout import ladder_radix_bits, ladder_radix_steps
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
    rot = extra | fakecap.GPT2_BAND | (ladder_radix_steps(Dims.gpt2(), ladder_radix_bits())
                                       if ladder_radix_bits() >= 2 else set())
    _fhe_f, inf = fakecap.make_session(Dims.gpt2(), rot)
    sess = fakecap.FakeSession(inf)
    core = fakecap.core
    store = weights.RawStore(a.weights)
    cfgs = config.load_configs(a.configs)
    model = Gpt2Prefill(inf, store, cfgs, core=core, n_layers=a.layers, profile=a.profile)
    model.with_tail = a.tail
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
    os.environ["MULTI_T"] = str(a.tokens)
    from . import oracle_io
    rc = oracle_io.RunConfig.from_env()
    if a.random_ids is not None:
        ids = np.random.default_rng(a.random_ids).integers(0, store.shape(weights.WTE)[0], a.tokens)
        rows = store.tensor(weights.WTE)[ids] + store.tensor(weights.WPE)[:a.tokens]
        gt = plain_logits(store, exact_blocks(store, rows, 12)[-1])
    else:
        rows = np.asarray(oracle_io.read_teacher_forced_inputs(rc)[:a.tokens])
        gt = np.asarray(oracle_io.read_lm_head_steps(rc).logits)[:a.tokens]
    ref = exact_blocks(store, rows, model.n_layers) if a.check else None
    print(f"[prefill] T={a.tokens} layers={model.n_layers} cts={(a.tokens + 2 * d.t - 1) // (2 * d.t)} "
          f"capture={a.capture} plan={a.plan}", flush=True)
    if not a.fake:
        rc_ = _planned(a, model, store, rows, gt, ref, results)
        model.close()
        sess.close()
        return rc_
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

    if a.taps and os.environ.get("PF_LAYOUT", "") != "blocked":
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
    # GPT2_PF_WARMUP (pers/dev 31ab85f's warm-up) is the device path's (`_planned`): the fake has no caches to warm
    t0 = time.time()
    Xs, ch, rec = model.prefill(rows, on_block=on_block)
    wall = time.time() - t0
    out = {"wall": wall, "bts": int(inf.fhe.total_bootstraps)}
    if a.tail:
        from .head import decode_logits, decode_z
        lg = decode_logits(model.rt, model.tail_tiles, model.vocab, model.W_tile)
        rows_, s = report(lg[None, :], gt[-1:])
        out.update(s)
        print(f"[prefill] tail: position {a.tokens - 1} KL {rows_[0]['kl']:.3g} top1 {int(np.argmax(lg))} "
              f"CutMax {int(np.argmax(decode_z(model.rt, Xs, model.vocab, model.W_tile)))}", flush=True)
    elif model.n_layers == 12:
        h = model.hidden(Xs, ch)
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
    ok = model.n_layers != 12 or (out["top1"] >= 0.9 * (1 if a.tail else a.tokens) and out["medianKL"] < 0.1)
    print("[prefill] PASS" if ok else "[prefill] FAIL")
    model.close()
    sess.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
