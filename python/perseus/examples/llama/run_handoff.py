"""Encrypted prefill of T prompt tokens, its K / V handed to the decode's caches (perseus.impl.handoff), then N
decoded positions, logits gated against the fp32 model.

    .venv/bin/python -m perseus.examples.llama.run_handoff --tokens 127 --decode 8 \
        --prefill-plan bootstrap_placements/llama31_8b_prefill128 --decode-plan <a decode plan cut for the hand-off>

No hand-off decode plan ships: one is cut for the hand-off's cache level (python/perseus/examples/llama/make_plan.sh
with KV_ENTRY_LEVEL and PLAN_KEEP_PLACEMENTS) on the prefill's chain. --handoff-only checks the hand-off itself.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def handoff_keys(dims):
    """Rotation steps the hand-off's moves decompose over (impl.handoff.rot_any: one key, else two): every shift
    8g - 1 - 64c + 32m - 128b of the blocked prefill's chunks lies in [-208, 240] and is 16q + r, |r| < 16."""
    return sorted({16 * q for q in range(-13, 16) if q} | {r for r in range(-15, 16) if r})


def _handoff_check(a, dims, n, T, pending, dec, kv_init, run, log, prefill_s, hstate):
    """--handoff-only: the hand-off on an eager session (no decode programs), every handed state decrypted against the
    decode twin's (the twin ran the same hand-off on the twin prefill's K / V). Reports TTFT = prefill + hand-off."""
    from perseus import _pc
    size = {"hidDim": dims.hid, "dim": dims.dim, "numHeads": dims.H, "numHeadsReal": dims.H_real, "expDim": dims.E,
            "expanded": dims.E_real, "seqLen": 1024}
    sess = _pc.Session(run.cfg, run.keys, size=size, pool=320, pt_slots=1024)
    sess.complete_setup()
    run.fhe = sess.fhe
    errs = []
    for b in range(n):
        got = kv_init(run, b)
        cts, keys = dec.kv[b].cts("")
        want = dict(zip(keys, cts))
        e = 0.0
        for k, ct in got.items():
            w = np.asarray(want[k].vec)
            g = np.asarray(sess.fhe.decrypt_slots_complex(ct))
            e = max(e, float(np.abs(g[:w.shape[0]] - w).max() / max(np.abs(w).max(), 1e-30)))
            if (int(ct.level), int(ct.deg)) != (int(want[k].level), int(want[k].noise_deg)):
                raise RuntimeError(f"block {b} {k}: device ({ct.level}, {ct.deg}) vs twin ({want[k].level}, {want[k].noise_deg})")
        errs.append(e)
        del got
    ttft = prefill_s + hstate.get("s", 0.0)
    log(f"[handoff] handed K/V vs the twin: max rel err per block {[f'{x:.2g}' for x in errs]} (max {max(errs):.3g})")
    log(f"[handoff] SUMMARY T={T} ttft={ttft:.1f}s = prefill {prefill_s:.1f}s + hand-off {hstate.get('s', 0.0):.1f}s; decode "
        f"not run (--handoff-only)")
    sess.close()
    return 0


def _planned(a, toks, gt, store, dims, over, cfgs_path, dcfgs_path, log=print):
    """PLANNED on the device: the planned paged prefill (LlamaPlannedPrefill) with its K / V kept
    as outputs and exported per block; the prefill session closed; the streamed decode session
    (StreamedPlannedRun) whose K / V units start from the hand-off (impl.handoff.to_decode_cache on the device's
    imported K / V); `--decode` positions decoded from T + 1. The CPU twin runs the same: the prefill twin's K / V
    handed to the decode twin's caches (one fakecap session for both)."""
    import gc
    from perseus import _pc
    from perseus._paths import resolve
    from perseus.impl import fakecap
    from perseus.impl.driver import Recorder
    from perseus.impl.handoff import to_decode_cache
    from . import env, prefix as PX, weights as W
    from .config import load_configs
    from .model import LlamaModel, dist_report
    from .planned import LlamaPlannedPrefill
    from .prefill import LlamaPrefill
    from .run_decode import logits_of, plan_config as decode_config, streamed_run
    from .run_prefill import _plan_config as prefill_config
    T, D = a.tokens, a.decode
    cfgs, dcfgs = load_configs(cfgs_path), load_configs(dcfgs_path)
    keys_h = set(handoff_keys(dims))
    # the twin: one fake session for the prefill and the decode models (the plans' merged runtime)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    env.export_env(a.device, chain="n32", **{"CKKS_COMPLEX": "1", **over})
    os.environ["CUDA_VISIBLE_DEVICES"] = cvd
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
    from perseus.impl.blocked import Blocks, blocked_rot_steps
    from perseus.impl.layout import ladder_radix_bits, ladder_radix_steps
    from perseus.impl.pcmm import rot_steps as pcmm_rot_steps
    from perseus.impl.prefill import prefill_rot_steps
    from perseus.impl.sinc import rot_steps as sinc_rot_steps
    rot = (prefill_rot_steps(dims) | pcmm_rot_steps() | ladder_radix_steps(dims, ladder_radix_bits()) | sinc_rot_steps()
           | {16, -16} | blocked_rot_steps(Blocks(dims, 512), ladder_radix_bits()) | set(env.rot_steps(dims)) | keys_h)
    _, tinf = fakecap.make_session(dims, rot)
    lw_p = W.LlamaWeights(store, cfgs, cfgs_path, dims, "cachemir_complex")
    lw_d = W.LlamaWeights(store, dcfgs, dcfgs_path, dims, "cachemir_complex")
    pre = LlamaPrefill(tinf, lw_p, cfgs, core=fakecap.core, n_layers=a.layers, packing="cachemir_complex")
    dec = LlamaModel(tinf, lw_d, dcfgs, core=fakecap.core, n_layers=a.layers, packing="cachemir_complex", kv_offload=False)
    # the decode keeps its own Chebyshev-GS switch (the merged runtime carries the prefill plan's)
    dec.rt.ops.sm_cheb_gs = env.plan_runtime(a.decode_plan, "n32").get("SM_CHEB_GS", "0") not in ("", "0")
    n = pre.n_layers
    bos = lw_p.bos_kv(n)
    dh = dims.dim // pre.H

    def bos_vec(f):
        x = np.zeros(dims.N)
        x[np.arange(dims.hid) * dims.t] = PX.expanded_heads(f, pre.H, pre.H_kv, dh)
        return x
    # the twin's hand-off: on the prefill twin's K / V once their last page is done (the old run_handoff's on_kv)
    orig_pp = pre.prefill_paged
    acc = {}

    def prefill_paged(tokens, on_block=None, on_kv=None):
        from perseus.impl.ccmm_paged import page_tokens
        n_pages, cov = 0, 0
        while cov < T:
            cov += page_tokens(n_pages)
            n_pages += 1

        def kv(p, b, K, V):
            if on_kv is not None:
                on_kv(p, b, K, V)
            Ks, Vs, bs = acc.setdefault(b, ([[] for _ in K], [[] for _ in V], []))
            for i in range(len(K)):
                Ks[i] += K[i]
                Vs[i] += V[i]
            base = 1 if p == 0 else 128 * p
            bs += [base, base + 64]
            if p == n_pages - 1:
                with pre.rt.step("handoff"):
                    dec.adopt_cache(b, to_decode_cache(pre.rt, Ks, Vs, T, bos_vec(bos[b][0]), bos_vec(bos[b][1]),
                                                       keys=keys_h, layer=b, bases=bs))
                acc[b] = (Ks, Vs, bs)
        return orig_pp(tokens, on_block=on_block, on_kv=kv)
    pre.prefill_paged = prefill_paged
    pcfg = prefill_config(resolve(a.prefill_plan))
    pcfg["exec.use_graphs"] = False     # the one-shot prefill runs eager (run_prefill's default)
    seed = int.from_bytes(os.urandom(8), "little")
    keys_p = _pc.KeySource.generate([], conj=True, layout="packed", seed=seed)
    work = os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), f"m9e_handoff_{os.getpid()}")
    pp = LlamaPlannedPrefill(pre, resolve(a.prefill_plan), pcfg, keys_p, kv_out=True, work_dir=os.path.join(work, "prefill"),
                             pt_rows=28, stc_cache=8, ccmm_kappa=(256.0, 16.0), log=log)
    t0 = time.time()
    pp.prepare(toks[:T + 1])
    warm_vals = None
    if a.warmup:
        # packed now: the decode twin below shares the fake session and changes its packing state
        from . import hf
        wtoks = [int(t) for t in hf.text_windows(2, 512)[1][:T + 1]]
        if wtoks[0] != int(toks[0]) or wtoks == [int(t) for t in toks[:T + 1]]:
            raise RuntimeError("warm-up prompt: expected a different document with the same BOS")
        warm_vals = pp.input_values(wtoks)
    pre.prefill_paged = orig_pp
    log(f"[handoff] prefill twin + programs {time.time() - t0:.0f}s ({len(acc)} layers handed to the decode twin)")
    # the decode twin from the handed caches: positions T + 1 .. T + D
    dec.stream_twin()
    dcfg_ns = argparse.Namespace(graphs=False, key_layout="packed", stc_cache=8, work_dir=os.path.join(work, "decode"),
                                 debug_blocks=a.debug_blocks, tap_block=a.tap_block, tap_steps="")
    rows = [int(x) for x in toks[T + 1:T + 1 + D]]
    pending = {}

    hstate = {}

    def kv_init(run, b):
        """The hand-off of block b on the decode session: the prefill's exported K / V imported and re-laid by
        to_decode_cache with the session's eager ops; {key suffix: Ct}."""
        t1 = time.time()
        if "rt" not in hstate:
            dm = LlamaModel(run.fhe, lw_d, dcfgs, core=_pc, n_layers=n, packing="cachemir_complex", kv_offload=False)
            hstate["rt"], hstate["s"] = dm.rt, 0.0
        rt, fhe = hstate["rt"], run.fhe

        def imp(x):
            ct = fhe.encrypt(np.zeros(dims.N, dtype=np.complex128), 53 - int(x["level"]), int(x["noise_level"]))
            fhe.import_limbs(ct, x["c0"], x["c1"], float(x["noise_factor"]), int(x["noise_level"]))
            return ct
        K, V, bs = pending.pop(b)
        K = [[imp(x) for x in ch] for ch in K]
        V = [[imp(x) for x in ch] for ch in V]
        kv = to_decode_cache(rt, K, V, T, bos_vec(bos[b][0]), bos_vec(bos[b][1]), keys=keys_h, layer=b, bases=bs)
        cts, ks = kv.cts("")
        fhe.device_sync()
        hstate["s"] += time.time() - t1
        return dict(zip(ks, cts))
    run = streamed_run(dcfg_ns, dec, lw_d, resolve(a.decode_plan), log=log)
    run.rec.first, run.rec.restart = T + 1, False
    run.prefix, run.kv_init = None, kv_init
    # SM_PERIODIC=2: positions T + 1 .. read a cache of T + 2 .. entries; the softmax values outgrow the planned
    # 256-slot refreshes and run on the class's Gp x tH route (8192 past 128 entries)
    def routes_for(pos):
        r = dec.routes_for(pos)
        return set() if r is None else set(r[1])
    run.routes_for = routes_for
    run.dense_fallback = False
    # the handed K / V states sit deeper than the decode's own cache writes (34): the decode plan must be cut for the
    # hand-off's level (make_plan.sh KV_ENTRY_LEVEL + PLAN_KEEP_PLACEMENTS); the class compile checks every pin
    # strictly
    run.extra_pt_slots = 1024
    run.pool = 256                      # the hand-off's eager ciphertexts (one layer's at a time)
    run.cfg = decode_config(resolve(a.decode_plan))
    run.cfg["exec.use_graphs"] = False  # eager replay, as run_decode (per-token constants would pin the ring)
    run.keys = _pc.KeySource.generate(sorted(keys_h), conj=True, layout="packed", seed=seed)
    run.check_meta = a.check_meta
    t0 = time.time()
    if not a.handoff_only:
        run.prepare(rows)
        log(f"[handoff] decode twin + classes {time.time() - t0:.0f}s")
    if a.compile_only:
        return 0
    # the device prefill: its K / V exported per block (the registers ping-pong over blocks); every device object of
    # the prefill session dies with this function (the decode session needs the memory)
    def device_prefill():
        pp.setup(toks[:T + 1])
        kv_regs = {}
        for b in range(n):
            reg_of = {pp.recs[b].out_names[j]: r for j, (_o, r) in enumerate(pp.reg[b])}
            Kn, Vn = pp.kv_names[b]
            kv_regs[b] = ([[reg_of[x] for x in ch] for ch in Kn], [[reg_of[x] for x in ch] for ch in Vn])
        if warm_vals is not None:
            # warm: one untimed prefill of another document of the same length (same BOS) in this session first
            t_w = time.time()
            pp.run(vals=warm_vals)
            pp.fhe.device_sync()
            log(f"[handoff] warm-up prefill (other prompt) {time.time() - t_w:.1f}s")
        t_pre = time.time()

        def on_block(b, info):
            K, V = kv_regs[b]
            exp = lambda r: dict(pp.fhe.export_limbs(pp.ct(r)))
            pending[b] = ([[exp(r) for r in ch] for ch in K], [[exp(r) for r in ch] for ch in V], acc[b][2])
        pp.run(on_block=on_block)
        pp.fhe.device_sync()
        s_ = time.time() - t_pre
        pp.sess.close()
        return s_
    prefill_s = device_prefill()
    log(f"[handoff] device prefill {prefill_s:.1f}s ({prefill_s / T:.3f} s/tok) incl. the K / V export")
    pp = None
    gc.collect()
    if a.handoff_only:
        return _handoff_check(a, dims, n, T, pending, dec, kv_init, run, log, prefill_s, hstate)
    # the decode session; its setup runs the hand-off (kv_init)
    run.setup(rows)
    handoff_s = hstate.get("s", 0.0)
    ttft = prefill_s + handoff_s
    log(f"[handoff] TTFT {ttft:.1f}s = prefill {prefill_s:.1f}s + hand-off {handoff_s:.1f}s (sessions' setup excluded)")
    vocab = int(store.config["vocab_size"])
    full = n == int(store.config["num_hidden_layers"])
    res = []
    for t in range(len(rows)):
        pos = T + 1 + t
        tt = time.time()
        regs, info = run.token(t, rows)
        lg = logits_of(run.fhe, [run.ct(r) for r in regs], dims, vocab)
        r = {"pos": pos, "token_s": time.time() - tt, "top1": int(np.argmax(lg)), **info}
        if run.tap and run.tap[0] == t:
            for nm, err, den in run.tap_report():
                log(f"[handoff:tap] {nm}: rel {err:.3g} (max {den:.3g})")
        if gt is not None and full:
            r.update(dist_report(lg, np.asarray(gt[pos])))
        res.append(r)
        log(f"[handoff] pos{pos} top1={r['top1']} token={r['token_s']:.2f}s class={r['class']}"
            + (f" ref={r['ref']} KL={r['kl']:.4g} {'OK' if r['hit'] else 'MISS'}" if "kl" in r else ""))
    ttft_tok = ttft + res[0]["token_s"] if res else float("nan")
    log(f"[handoff] TTFT incl. the first decoded token {ttft_tok:.1f}s = prefill + K/V export {prefill_s:.1f}s + hand-off "
        f"{handoff_s:.1f}s + position {T + 1} {res[0]['token_s'] if res else float('nan'):.1f}s ({'warm' if a.warmup else 'cold'})")
    kls = sorted(r["kl"] for r in res if "kl" in r)
    hits = sum(1 for r in res if r.get("hit"))
    log(f"[handoff] SUMMARY T={T} ttft={ttft:.1f}s decode={len(res)} top1={hits}/{len(res)} "
        f"medianKL={kls[len(kls) // 2] if kls else float('nan'):.4g} maxKL={kls[-1] if kls else float('nan'):.4g} "
        f"median_token_s={sorted(r['token_s'] for r in res)[len(res) // 2]:.2f}")
    run.close()
    return 0


def main(argv=None):
    from perseus.examples.gpt2.run_decode import gpu_busy
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--window", default="0")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--device", type=int, default=None,
                    help="GPU index (default: the first of CUDA_VISIBLE_DEVICES)")
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
    ap.add_argument("--old-path", action="store_true", help="the eager device path of the old port (no planned run)")
    ap.add_argument("--warmup", type=int, default=0, choices=(0, 1),
                    help="1: an untimed prefill of another 128-token window in the prefill session first")
    ap.add_argument("--compile-only", action="store_true", help="planned: the twins and the programs only (CPU)")
    ap.add_argument("--debug-blocks", action="store_true", help="planned decode: every block output vs the twin (diag)")
    ap.add_argument("--tap-block", type=int, default=None, help="planned decode: block whose intermediates (first "
                    "decoded position) are compared with the twin (diag)")
    ap.add_argument("--check-meta", action="store_true", help="planned decode: device vs compiled metadata per op (diag)")
    ap.add_argument("--handoff-only", action="store_true",
                    help="planned: prefill + hand-off on the device, the handed K/V checked against the twin (no decode)")
    a = ap.parse_args(argv)
    from perseus.examples.defaults import default_plan
    if a.prefill_plan is None and a.decode_plan is None and not (a.fake or a.old_path):
        # BOS + tokens = the prefill's T
        a.prefill_plan = default_plan("llama_prefill", a.tokens + 1)
    if a.prefill_plan and a.decode_plan is None and not a.handoff_only:
        print("no hand-off decode plan ships: pass --decode-plan (a plan cut for the hand-off's cache level), or "
              "--handoff-only", file=sys.stderr)
        return 2
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if a.fake else env.gpt2_env.physical_device(a.device)
    gt = None if (a.fake or a.compile_only) else hf.oracle_logits(toks, cache)
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
    if not (a.fake or a.old_path):
        # the new runtime: planned prefill -> hand-off -> streamed planned decode (two sessions, one secret)
        from perseus._paths import resolve
        if not (a.prefill_plan and a.decode_plan):
            print("the device path runs planned (--prefill-plan, --decode-plan)", file=sys.stderr)
            return 2
        cal = os.path.join(resolve(a.prefill_plan), "pcmm_calib.json")
        if os.path.exists(cal):
            os.environ["PF_PCMM_CALIB_IN"] = cal

        def plan_cfgs(plan):
            for line in open(os.path.join(resolve(plan), "PLAN_CMD.txt")).read().splitlines():
                if line.startswith("configs="):
                    return line[len("configs="):].split()[0]
            return DEFAULT_CONFIGS
        dims = W.llama_dims(store.config)
        return _planned(a, toks, gt, store, dims, over, plan_cfgs(a.prefill_plan), plan_cfgs(a.decode_plan),
                        log=lambda m: print(m, flush=True))
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
    from perseus import _pc as _core
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

    def on_kv(b, K, V, bases=None):
        t = time.time()
        with pre.rt.step("handoff"):
            if b == 0 and not a.fake:
                for nm, S in (("K", K), ("V", V)):
                    z = pre.rt.ops.decrypt_slots_complex(S[0][0])
                    print(f"[handoff] layer 0 {nm}[0][0] max|re|={np.abs(z.real).max():.3g} "
                          f"max|im|={np.abs(z.imag).max():.3g}", flush=True)
            kv = to_decode_cache(pre.rt, K, V, a.tokens, bos_vec(bos[b][0]), bos_vec(bos[b][1]), keys=keys, layer=b,
                                 cleanse=a.cleanse, bases=bases)
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
                from perseus._pc import bts as BT
                dl = sorted({BT.device(inf.fhe, c).level for c in cts})
                print(f"[handoff] layer 0 cache cts={len(cts)} (level, deg)={lv} device levels={dl}", flush=True)
        dec.adopt_cache(b, kv)
        t_ho[0] += time.time() - t

    def on_block(b, Xs, s):
        print(f"[handoff] prefill block {b} {s:.2f}s bts={int(inf.fhe.total_bootstraps)}", flush=True)
    t1 = time.time()
    paged = os.environ.get("PF_PAGED", "0") not in ("", "0")
    if paged:
        # PF_PAGED=1: each layer's K / V chunks gathered over the pages, handed off after the last page
        from perseus.impl.ccmm_paged import page_tokens
        n_pages, cov = 0, 0
        while cov < a.tokens:
            cov += page_tokens(n_pages)
            n_pages += 1
        acc = {}

        def on_kv_paged(p, b, K, V):
            Ks, Vs, bs = acc.setdefault(b, ([[] for _ in K], [[] for _ in V], []))
            for i in range(len(K)):
                Ks[i] += K[i]
                Vs[i] += V[i]
            base = 1 if p == 0 else 128 * p
            bs += [base, base + 64]
            if p == n_pages - 1:
                on_kv(b, Ks, Vs, bases=bs)
                del acc[b]
        pages, _ = pre.prefill_paged(toks[:a.tokens + 1], on_block=lambda p, b, X, c, s: on_block(b, X, s),
                                     on_kv=None if (a.no_handoff or dec is None) else on_kv_paged)
        hid = np.concatenate([pre.hidden_rows(X, c) for X, c in pages])[:a.tokens]
        Xs = None
    else:
        Xs, ch, _ = pre.prefill(toks[:a.tokens + 1], on_block=on_block,
                                on_kv=None if (a.no_handoff or dec is None) else on_kv)
        hid = None
    ttft = time.time() - t1
    from .prefill import plain_logits
    from perseus.impl.prefill import Chunks
    if gt is not None and pre.n_layers == int(store.config["num_hidden_layers"]):
        rows, out = report(plain_logits(pre, pre.hidden_rows(Xs, ch) if hid is None else hid), gt[1:a.tokens + 1])
        print(f"[handoff] prefill top1={out.get('top1')}/{a.tokens} medianKL={out.get('medianKL', float('nan')):.4g}",
              flush=True)
    print(f"[handoff] prefill+handoff {ttft:.1f}s (handoff {t_ho[0]:.1f}s) bts={int(inf.fhe.total_bootstraps)}",
          flush=True)
    if dec is None:
        sess.close()
        return 0
    del Xs
    if paged:
        del pages
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
