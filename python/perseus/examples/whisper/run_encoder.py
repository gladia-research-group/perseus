"""Whisper base.en's encoder, encrypted: a 30-s clip -> log-mel rows (encrypted) -> the front-end and 6 blocks ->
the hidden states, gated against the float64 encoder (per-block max error).

    .venv/bin/python -m perseus.examples.whisper.run_encoder --fake --audio DIR [--n 1] [--layers 6] [--check]
        [--capture graphs/X --structure-only] [--plan bootstrap_placements/X]
    .venv/bin/python -m perseus.examples.whisper.run_encoder --audio DIR --n 2 --plan bootstrap_placements/X --check

The device path runs planned only (EncoderPlannedPrefill): the op stream recorded once on the CPU twin (any clip:
the stream depends on the 30-s window only), every block a compiled Program run in Rust.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

from . import weights as _W
DEFAULT_CONFIGS = f"configs/model/approximation/whisper_base_en_{_W.T}/configs.json"     # per window (T positions)
# ViT's plan runtime (the encoder plans share its planner path), less what is ViT's own
TEMPLATE_PLAN = "bootstrap_placements/vit_b16_224"
TEMPLATE_DROP = ("PF_PCMM_RAISE_OUT",)
TEMPLATE_SET = {"PF_LAST_QUERY": "0"}      # every position's output feeds the decoder's cross-attention


def dims():
    """Whisper base.en's geometry: width 512 (no padding), 8 heads of 64, MLP 2048."""
    from perseus.impl.layout import Dims
    return Dims(1 << 15, 512, 512, 8, 8, 2048, 2048)


def clip_seconds(path):
    import wave
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def clips(path, n=None, spread=False, max_s=None, log=print):
    """The WAV clips of `path` (a file or a directory, sorted): the first n, or n spread evenly over the directory
    (`spread`); clips longer than `max_s` seconds left out (a window transcribes those it leaves silence after)."""
    fs = sorted(glob.glob(os.path.join(path, "*.wav"))) if os.path.isdir(path) else [path]
    if max_s is not None:
        keep = [f for f in fs if clip_seconds(f) <= max_s + 1e-6]
        if len(keep) < len(fs):
            log(f"[whisper] {len(fs) - len(keep)} of {len(fs)} clip(s) longer than {max_s:g} s left out (the window's "
                f"limit)")
        fs = keep
    if n is not None and spread and len(fs) > n:
        fs = [fs[i] for i in np.linspace(0, len(fs) - 1, n).round().astype(int)]
    return fs[:n]


def runtime_env(plan):
    """The plan's runtime switches; with no plan, the template's (TEMPLATE_PLAN less TEMPLATE_DROP, plus TEMPLATE_SET)."""
    from perseus.examples.gpt2 import env
    if plan:
        return env.plan_runtime(plan, "n32")
    rt = {k: v for k, v in env.plan_runtime(TEMPLATE_PLAN, "n32").items() if k not in TEMPLATE_DROP}
    return {**rt, **TEMPLATE_SET}


def setup(a):
    """The env, the fake session (capture, or the planned run's CPU twin) and the encoder model: (model, inf, sess)."""
    from perseus._paths import resolve
    from perseus.examples.gpt2 import env
    if a.plan:
        a.plan = resolve(a.plan)
    fast = {k: os.environ.get(k, v) for k, v in (("FIDESLIB_ENCODE_NOSYNC", "1"), ("FIDESLIB_CCMM_ASYNC", "1"),
                                                 ("FIDESLIB_PCMM_NOSYNC", "1"), ("FIDESLIB_ENCODE_PINNED_RING", "256"),
                                                 ("FIDESLIB_STC_POW2_KEY", "1"))}
    over = {**fast, **runtime_env(a.plan), "GPT2_PACKING": "cachemir", **dict(kv.split("=", 1) for kv in a.set)}
    for k, v in over.items():
        if k.startswith("PF_"):
            os.environ.setdefault(k, str(v))
    if not a.configs:
        a.configs = DEFAULT_CONFIGS
        cmd = os.path.join(a.plan, "PLAN_CMD.txt") if a.plan else ""
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
    from perseus.impl import config, fakecap
    from perseus.impl.blocked import Blocks, blocked_rot_steps
    from perseus.impl.layout import ladder_radix_bits, ladder_radix_steps
    from perseus.impl.pcmm import rot_steps as pcmm_rot_steps
    from perseus.impl.prefill import diag_rot_steps, prefill_rot_steps
    from . import weights as W
    from .encoder import WhisperEncoder
    d0 = dims()
    rot = prefill_rot_steps(d0) | diag_rot_steps(d0) | (pcmm_rot_steps() if pcmm else set()) \
        | blocked_rot_steps(Blocks(d0, int(os.environ.get("PF_NCH", "512"))), ladder_radix_bits()) \
        | fakecap.GPT2_BAND | (ladder_radix_steps(d0, ladder_radix_bits()) if ladder_radix_bits() >= 2 else set())
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
    _, inf = fakecap.make_session(d0, rot)
    sess = fakecap.FakeSession(inf)
    store = W.WhisperStore("encoder")
    cfgs = config.load_configs(a.configs)
    model = WhisperEncoder(inf, store, cfgs, core=fakecap.core, n_layers=a.layers, profile=a.profile)
    if a.capture:
        model.set_capture(a.capture)
        from perseus.impl.magtape import save_rot_steps
        save_rot_steps(a.capture, inf.fhe.loaded_rot_steps)
        if a.structure_only:
            inf.fhe.magnitude_suppressed = True
    if a.plan and a.fake:
        model.load_plans(a.plan, argmax_blocks=False)
    os.environ.setdefault("STEPS_T", "128")
    os.environ["MULTI_T"] = str(W.T)
    t0 = time.time()
    model.preload_prefill()
    if pcmm:
        from perseus.impl.pcmm import engine
        engine(model.rt).prepare(model.rt.dims.dim)
    print(f"[whisper] T={W.T} layers={model.n_layers} weights ready {time.time() - t0:.1f}s capture={a.capture} "
          f"plan={a.plan} configs={a.configs}", flush=True)
    return model, inf, sess


def report_block(b, got, ref):
    err = np.abs(got - ref).max(1)
    return f"maxerr={err.max():.3g} (tok {int(err.argmax())}) median={np.median(err):.3g} |x|={np.abs(ref).max():.3g}"


def report_xkv(model, Xs, ref):
    """The xkv stage's caches (decrypted) against their float values (weights.cross_cache_ref) on the real keys' lanes:
    max |err| of the K and V buckets, and their max |value|."""
    dec = model.rt.ops.decrypt_slots_complex
    nk = len(ref) // model.n_dec - model.rt.dims.d_head_real // 2
    err, mag = {"K": 0.0, "V": 0.0}, {"K": 0.0, "V": 0.0}
    for i, (x, (re, im)) in enumerate(zip(Xs, ref)):
        v, kind = dec(x), "K" if i % (len(ref) // model.n_dec) < nk else "V"
        for got, want in ((v.real, re), (v.imag, im)):
            ok = ~np.isnan(want)
            err[kind] = max(err[kind], float(np.abs(got[ok] - want[ok]).max(initial=0.0)))
            mag[kind] = max(mag[kind], float(np.abs(want[ok]).max(initial=0.0)))
    return " ".join(f"{k} maxerr={err[k]:.3g} |x|={mag[k]:.3g}" for k in ("K", "V"))


def run_fake(a, model, inf):
    from . import weights as W
    store = model.store
    dec = W.WhisperStore("decoder", store.path)
    model.n_dec = dec.n_layers
    for f in clips(a.audio, a.n, max_s=W.MAX_CLIP_S):
        rows = W.mel_rows(W.log_mel(W.load_audio(f), store.path))
        hs, out = W.encoder_ref(store, rows)
        xref = [v for K, V in W.cross_kv(dec, out) for v in W.cross_cache_ref(K, V, W.T, model.rt.dims)]
        x0 = W.stem(store, rows)
        b0 = [int(inf.fhe.total_bootstraps)]
        if a.check:
            model.dbg_embed = lambda h: print(f"[whisper] front-end {report_block(0, h[:W.T], x0)}", flush=True)

        def on_block(b, Xs, s):
            tb = int(inf.fhe.total_bootstraps)
            line = f"[whisper] block {b} {s:.1f}s bts={tb - b0[0]}"
            b0[0] = tb
            if a.check:
                line += " " + (report_block(b, model.hidden(Xs, model.last_ch), hs[b]) if b < model.n_layers else
                               report_xkv(model, Xs, xref))
            print(line, flush=True)
        t0 = time.time()
        Xs, ch, _ = model.prefill(rows, on_block=on_block)
        print(f"[whisper] {os.path.basename(f)}: wall {time.time() - t0:.1f}s bts {int(inf.fhe.total_bootstraps)}",
              flush=True)
        if a.save:                         # the decoder's cross caches decrypted, for its twin (run_asr --enc)
            os.makedirs(a.save, exist_ok=True)
            np.save(os.path.join(a.save, os.path.basename(f)[:-4] + ".xkv.npy"),
                    np.stack([model.rt.ops.decrypt_slots_complex(x) for x in Xs]))
    if a.capture:                          # the linears' stamps for the planner (PF_PCMM_STAMPS_OUT -> PLAN_PCMM_STAMPS)
        from perseus.impl.pcmm import PcmmLinear, session_specs
        lins = model.extra_linears() + [w for b in range(model.n_layers) for w in model.diag_weights(b).values()
                                       if isinstance(w, PcmmLinear)]
        session_specs(lins, model.blocks.t)


XKV_METAS = "xkv_metas.json"


def stage_out_metas(run, b=-1):
    """[(level, deg)] of stage b's outputs as its compiled program leaves them, in the stage's output order."""
    b = b % run.n_stages
    got = {n: (int(lv), int(dg)) for n, lv, dg, _s in run.progs[run.labels[b]].outputs}
    return [got[n] for n in run.recs[b].out_names]


def save_xkv_metas(run, plan, log=print):
    """The cross caches' (level, deg) as the compiled last stage hands them, kept in the plan (`XKV_METAS`): the
    decoder's twin adopts them at these (run_asr --enc-plan), so its plan reads what the encoder gives."""
    import json
    metas = stage_out_metas(run)
    path = os.path.join(plan, XKV_METAS)
    old = json.load(open(path)) if os.path.exists(path) else None
    if old != [list(m) for m in metas]:
        with open(path, "w") as f:
            json.dump(metas, f)
        log(f"[whisper] {path}: {len(metas)} cross cache states, (level, deg) {sorted(set(metas))}")
    return metas


def export_caches(plan, files, level=None, device=None, sets=(), log=print, work_dir=None, secret=None, graphs=True):
    """The encrypted encoder planned on the device over `files`: per clip its last stage's ciphertexts (the decoder's
    cross caches, WhisperEncoder.cross_caches' order) exported as limbs ({"c0", "c1": (limbs, N) uint32, "level" (top
    limb), "noise_level", "noise_factor"}), each first dropped to `level` (device level) when given; the session
    closed after (`graphs`: each stage replayed as CUDA graphs, as run_encoder --graphs). The decoder imports them into
    its own session on the same chain (run_asr). Run it in a process of its own: the encoder's env (its module-level
    switches) must not reach the decoder's."""
    import argparse
    from perseus import _pc
    from . import weights as W
    a = argparse.Namespace(plan=plan, set=list(sets), configs=None, device=device, fake=False, capture=None,
                           structure_only=False, layers=None, profile=False)
    model, _inf, _sess = setup(a)
    # after setup: the prefill modules read the plan's runtime switches (PF_*) when they are imported
    from perseus.examples.bert.planned import plan_config
    from perseus.examples.vit.planned import VitPlannedPrefill
    rows = [W.mel_rows(W.log_mel(W.load_audio(f), model.store.path)) for f in files]
    keys = _pc.KeySource.generate([], conj=True, layout="packed", secret=secret)    # the client's (run_asr.client_secret)
    work = work_dir or os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), f"whisper_enc_{os.getpid()}")
    run = VitPlannedPrefill(model, a.plan, plan_config(a.plan, graphs), keys, work_dir=work, pt_rows=28, stc_cache=16,
                            log=log)
    run.prepare(rows[0])
    save_xkv_metas(run, a.plan, log)
    run.setup(rows[0])
    out = {}
    for f, r in zip(files, rows):
        t0 = time.time()
        regs = run.run(vals=run.input_values(r))
        run.fhe.device_sync()
        wall = time.time() - t0
        res = []
        for g in regs:                  # one at a time: run.ct draws on the session's small ciphertext pool
            ct = run.ct(g)
            if level is not None:
                run.fhe.drop_to_level(ct, int(level))
            x = dict(run.fhe.export_limbs(ct))
            del ct
            res.append({"c0": np.asarray(x["c0"], dtype=np.uint32), "c1": np.asarray(x["c1"], dtype=np.uint32),
                        "level": int(x["level"]), "noise_level": int(x["noise_level"]),
                        "noise_factor": float(x["noise_factor"])})
        out[f] = res
        log(f"[whisper] {os.path.basename(f)}: encoder {wall:.1f}s, {len(res)} cache states exported")
    run.close()
    return out


def run_planned(a, model):
    """PLANNED on the device: record + compile + session once, then one timed forward per clip (block 0's inputs
    encrypted, every block issued, the device drained); s/clip = the median of the clip walls."""
    from perseus import _pc
    from perseus.examples.bert.planned import plan_config
    from perseus.examples.vit.planned import VitPlannedPrefill
    from . import weights as W
    store = model.store
    files = clips(a.audio, a.n, max_s=W.MAX_CLIP_S)
    rows = [W.mel_rows(W.log_mel(W.load_audio(f), store.path)) for f in files]
    keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
    work = a.work_dir or os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), f"whisper_enc_{os.getpid()}")
    run = VitPlannedPrefill(model, a.plan, plan_config(a.plan, a.graphs), keys, work_dir=work, pt_rows=a.pt_rows,
                            stc_cache=a.stc_cache, log=lambda m: print(m, flush=True))
    t_set = time.time()
    run.prepare(rows[0])
    save_xkv_metas(run, a.plan)
    progs = [run.progs[lab] for lab in run.labels]
    n_bts = sum(p.n_bootstraps for p in progs) + sum(p.n_finishes[1] for p in progs)
    print(f"[whisper] compiled {len(progs)} stages: record {run.record_s:.1f}s compile {run.compile_s:.1f}s; "
          f"{n_bts} bts", flush=True)
    if a.compile_only:
        return 0
    run.setup(rows[0])
    print(f"[whisper] planned setup {time.time() - t_set:.1f}s; device free {run.fhe.mem_info()[0] / 2 ** 30:.2f} GB",
          flush=True)
    walls = []
    for f, r in zip(files, rows):
        vals = run.input_values(r)
        t0 = time.time()
        run.run(vals=vals)
        run.fhe.device_sync()
        walls.append(time.time() - t0)
        line = f"[whisper] {os.path.basename(f)}: wall={walls[-1]:.2f}s free {run.fhe.mem_info()[0] / 2 ** 30:.2f} GB"
        if a.check:
            hs, _ = W.encoder_ref(store, r)
            for b in range(model.n_layers):
                print(f"[whisper]   block {b} {report_block(b, run.hidden(b), hs[b])}", flush=True)
        print(line, flush=True)
    print(f"[whisper] SUMMARY clips={len(files)} s/clip={np.median(walls):.2f} "
          f"(walls {' '.join(f'{w:.2f}' for w in walls)}) bts/clip={n_bts}", flush=True)
    run.close()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="a WAV file or a directory of them")
    ap.add_argument("--n", type=int, default=None, help="first n clips")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--configs", default=None)
    ap.add_argument("--capture", default=None)
    ap.add_argument("--structure-only", action="store_true")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--check", action="store_true", help="compare every block with the float64 encoder")
    ap.add_argument("--fake", action="store_true", help="the numpy fake (perseus.impl.fakecap): no GPU")
    ap.add_argument("--save", default=None, help="fake: write each clip's cross caches, decrypted (<clip>.xkv.npy), here")
    ap.add_argument("--graphs", action="store_true", help="planned: replay each stage as CUDA graphs")
    ap.add_argument("--key-layout", choices=("dense", "packed"), default="packed")
    ap.add_argument("--pt-rows", type=int, default=28, help="planned: rows per plaintext slot (0 = every prime)")
    ap.add_argument("--stc-cache", type=int, default=16, help="planned: StC-first last-stage cache entries")
    ap.add_argument("--work-dir", default=None, help="planned: where the recorded stream is written")
    ap.add_argument("--compile-only", action="store_true", help="planned: record + compile on the CPU, then stop")
    ap.add_argument("--force", action="store_true", help="run on a GPU other processes use")
    a = ap.parse_args(argv)
    if not a.fake and not a.plan:
        print("the device path runs planned (--plan); capture with --fake", file=sys.stderr)
        return 2
    if not (a.fake or a.compile_only or a.force):
        from perseus.examples.gpt2.run_decode import gpu_busy
        busy = gpu_busy(a.device)
        if busy:
            print(f"GPU {a.device} has other processes: {busy}; refusing (use --force)", file=sys.stderr)
            return 2
    os.environ.setdefault("KV_ARENA_GB", "0")
    model, inf, sess = setup(a)
    rc = run_fake(a, model, inf) if a.fake else run_planned(a, model)
    model.close()
    sess.close()
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
