"""Speech to text with Whisper base.en: the decoder's greedy transcription on the encrypted runtime, from the encoder's
cross-attention caches (examples/whisper/encoder.py's last stage), checked token for token against the float64 greedy
decoding.

    .venv/bin/python -m perseus.examples.whisper.run_asr --fake --audio CLIP [--max-new 128] [--tokens 8] [--enc DIR]

The caches: the encrypted encoder's (--enc: its twin's, run_encoder --fake --save DIR), or the float encoder's K / V
filled in the decoder's session.

--fake: the CPU twin (perseus.impl.fakecap), no GPU. The decoder runs on GPT-2's decode runtime switches
(DECODE_TEMPLATE's PLAN_CMD.txt runtime) until it has its own plan.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np

from . import weights as _W

_ENV0 = dict(os.environ)          # the launch env: the encoder's process starts from it, not from the decoder's switches
_DEC = "whisper_base_en_dec" + ("" if _W.T == 1500 else f"_{_W.T}")      # the decode's configs / plan per encoder window
DEC_CONFIGS = f"configs/model/approximation/{_DEC}/configs.json"
DECODE_TEMPLATE = "bootstrap_placements/gpt2_decode"
DEC_PLAN = f"bootstrap_placements/{_DEC}"
ENC_PLAN = f"bootstrap_placements/whisper_base_en_{_W.T}"     # the encoder's plan for the window
# every token enters block 0 at the fed-back token's (level, deg): planner.fed_back_entry on DEC_PLAN's capture
DEC_ENTRY = (26, 2)


def decoder_env(plan=None, sets=(), device=None):
    """The decode's session env: the plan's runtime switches (default DECODE_TEMPLATE's), the complex payload, the
    cachemir_complex packing; `sets` KEY=VALUE overrides. Returns the overrides applied."""
    from perseus.examples.gpt2 import env
    over = {**env.plan_runtime(plan or DECODE_TEMPLATE, "n32"), "GPT2_PACKING": "cachemir_complex",
            **dict(kv.split("=", 1) for kv in sets)}
    env.export_env(device, chain="n32", CKKS_COMPLEX="1", **over)
    return over


def fake_decoder(a):
    """The decoder on the CPU twin: (model, inf); a capture's session when a.capture (at position a.capture_pos)."""
    from perseus.examples.gpt2 import env
    from perseus.impl import config, fake, fakecap
    from .decoder import WhisperDecoder
    from .run_encoder import dims
    from . import weights as W
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    over = decoder_env(a.plan, a.set, getattr(a, "device", None))
    # the twin alone (a check, a capture) hides the GPUs; a planned run's twin keeps the device the run uses
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if (a.fake or getattr(a, "capture", None)) else cvd
    if getattr(a, "capture", None):
        os.environ["FHE_GRAPH_DIR"] = a.capture
        os.environ["FHE_GRAPH_CAPTURE_TOKEN"] = str(a.capture_pos)
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over}, "n32", True)
    d = dims()
    band, _ = fake.make_fake(N=d.N, hid=d.hid, dim=d.dim, H=d.H, H_real=d.H_real, E=d.E, E_real=d.E_real)
    _, inf = fakecap.make_session(d, set(band.loaded_rot_steps) | fakecap.GPT2_BAND)
    model = WhisperDecoder(inf, W.WhisperStore("decoder"), config.load_configs(a.configs), core=fakecap.core,
                           kv_offload=False)
    model.cross_metas = cross_metas(getattr(a, "enc_plan", None))
    model.entry = a.entry
    missing = env.rotation_audit(inf.fhe, model.rt.dims)
    if missing:
        raise RuntimeError(f"rotation keys missing from the band: {missing}")
    if getattr(a, "capture", None):
        from perseus.impl.magtape import save_rot_steps
        model.set_capture(a.capture)
        save_rot_steps(a.capture, inf.fhe.loaded_rot_steps)
        if a.structure_only:
            inf.fhe.magnitude_suppressed = True
    return model, inf


def utterance(model, clip, max_new):
    """(the float encoder's output, the float greedy transcript) of one clip."""
    from . import weights as W
    enc = W.WhisperStore("encoder", model.store.path)
    _, enc_out = W.encoder_ref(enc, W.mel_rows(W.log_mel(W.load_audio(clip), enc.path)))
    return enc_out, W.greedy(model.store, enc_out, max_new, model.gen)


def teacher_rows(model, tokens):
    """The client's rows of the forced start then `tokens`, one per position (a teacher-forced decode)."""
    from perseus.examples.gpt2 import weights as GW
    return [GW.token_embedding(model.store, tok, p, model.rt.dims.dim)
            for p, tok in enumerate(list(model.gen.start) + list(tokens))]


def capture(a, log=lambda *x: print(*x, flush=True)):
    """The decode's plan capture: positions 0 .. a.capture_pos teacher-forced on the first clip's float transcript
    (CutMax on each), the blocks, the tail and CutMax recorded at a.capture_pos."""
    model, _inf = fake_decoder(a)
    enc_out, ref = utterance(model, a.clips[0], a.max_new)
    rows = teacher_rows(model, ref)
    if a.capture_pos >= len(rows):
        raise ValueError(f"position {a.capture_pos}: the transcript holds {len(rows)}")
    model.fill_plain(enc_out)
    t0 = time.time()
    model.run_decode(rows[:a.capture_pos + 1], argmax=True)
    log(f"[asr] captured position {a.capture_pos} into {a.capture} in {time.time() - t0:.0f}s")
    return 0


def run_fake(a, log=lambda *x: print(*x, flush=True)):
    from . import weights as W
    model, _inf = fake_decoder(a)
    gen = model.gen
    res = []
    for clip in a.clips:
        enc_out, ref = utterance(model, clip, a.max_new)
        t0 = time.time()
        if a.enc:                          # the encrypted encoder's caches (run_encoder --fake --save DIR)
            model.adopt_values(np.load(os.path.join(a.enc, os.path.basename(clip)[:-4] + ".xkv.npy")))
        else:
            model.fill_plain(enc_out)
        log(f"[asr] {os.path.basename(clip)}: cross caches {'from ' + a.enc if a.enc else 'of the float encoder'} "
            f"in {time.time() - t0:.1f}s; float greedy {len(ref)} tokens")
        out = []

        def on_step(r):
            tok = model.token_of(r["cutmax"])
            out.append(tok)
            j = len(out) - 1
            want = ref[j] if j < len(ref) else None
            log(f"[asr]   pos {r['pos']}: token {tok} (logits argmax {model.token_of(r['fhe_argmax'])}, float "
                f"{want}) z_mass {r['z_mass']:.3f} {'OK' if tok == want else 'MISS'} {time.time() - t0:.0f}s")
        n = min(a.tokens or a.max_new, a.max_new)
        model.generate(model.prompt_rows(), n, on_step=on_step, stop=lambda r: out[-1] == gen.eot)
        k = min(len(out), len(ref))
        same = out[:k] == ref[:k]
        log(f"[asr] {os.path.basename(clip)}: {len(out)} tokens {'IDENTICAL' if same else 'DIFFER'} to the float "
            f"greedy over {k}; {time.time() - t0:.0f}s")
        res.append(same)
    return 0 if all(res) else 1


def planned_run(model, plan, rows, work, log=print, debug=False, secret=None):
    """The decoder's planned generation (gpt2.run_generate.GenerateRun) on the twin `model` whose cross caches are
    filled: recorded on `rows` (any rows: the twin records op streams, not tokens), compiled; the device session set
    up and every block's cross caches given before token 0 at the twin's metadata (`cross_cts`)."""
    from perseus import _pc, config as pcfg
    from perseus.examples.gpt2.run_generate import GenRecorder, GenerateRun

    class Run(GenerateRun):
        def input_ct(self, t, rows_, p):
            """Row t encrypted at the program's input (level, deg): every token enters block 0 at the fed-back
            token's metadata (WhisperDecoder.entry)."""
            _nm, lv, dg, _nf = p.inputs[0]
            d = model.rt.dims
            v = np.zeros(d.N)
            v[np.arange(d.dim) * d.t] = np.asarray(rows_[t], dtype=np.float64)[:d.dim]
            return self.fhe.encrypt(v, lv, dg)

    cfg = pcfg.plan_runtime(plan, "n32", base=pcfg.preset("n32_gpt2_decode"))
    keys = _pc.KeySource.generate([], conj=True, layout="packed", secret=secret)
    n_fixed = sum(len(model.fixed_cache_keys(b)) for b in range(model.n_layers))
    run = Run(GenRecorder(model, model.n_layers), plan, cfg, keys, work_dir=work, pt_rows=28, stc_cache=8,
              pool=16 + n_fixed, log=log,        # the pool holds the cross caches' states, given once
              dyn_rot=[1] + [-i for i in range(model.rt.dims.t)])   # the push's lane moves over a t-token group
    run.debug = debug                    # every block's output and states decrypted against the twin's values
    t0 = time.time()
    run.prepare(rows, argmax=True)
    log(f"[asr] decoder compiled {len(run.progs)} programs in {time.time() - t0:.1f}s; refreshes per block "
        f"{[run.progs[run.labels[b]].n_bootstraps for b in range(model.n_layers + 3)]}")
    run.setup(rows, argmax=True)
    return run


def cross_cts(run, model, b, vecs=None):
    """Block b's cross cache states on the device, {cache key: Ct}: `vecs` (one utterance's cache values in cache_cts'
    order; default the twin's own) encrypted at the twin's (level, deg) (public values: the decoder without the
    encrypted encoder's hand-off)."""
    cts, keys = model.xkv[b].cts(f"impl.b{b}.x")
    vecs = [c.vec for c in cts] if vecs is None else vecs
    return {k: run.fhe.encrypt(np.asarray(v), int(c.level), int(c.noise_deg)) for c, k, v in zip(cts, keys, vecs)}


def cross_metas(enc_plan):
    """[(level, deg)] the encoder plan hands the cross cache states at (its run_encoder.XKV_METAS, written when it is
    compiled), None without an encoder plan: the decoder's twin adopts them there (its capture and planned run)."""
    import json
    from perseus._paths import resolve
    from .run_encoder import XKV_METAS
    if not enc_plan:
        return None
    path = os.path.join(resolve(enc_plan), XKV_METAS)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: compile the encoder plan first (run_encoder --plan {enc_plan} --compile-only)")
    return [tuple(m) for m in json.load(open(path))]


def _export_child(plan, files, device, work_dir, secret):
    from .run_encoder import export_caches
    return export_caches(plan, files, None, device, log=lambda m: print(m, flush=True), work_dir=work_dir,
                         secret=secret)


def max_new(model):
    """The tokens one utterance may generate: the decode plan's positions hold one program structure while the keys
    fit one lane group (position < t, the cachemir layout's), the forced start's included."""
    return model.rt.dims.t - len(model.gen.start)


def client_secret():
    """The client's secret (ClientSecret.to_bytes()): the encoder's and the decoder's sessions generate their keys from
    it, so the encoder's ciphertexts are the decoder's (one client, one key)."""
    from perseus import _pc
    return _pc.ClientSecret.generate(chain=os.environ.get("CHAIN_PRESET") or "n32").to_bytes()


def encoder_caches(plan, files, device=None, work_dir=None, secret=None):
    """The encrypted encoder's cross caches of `files` (run_encoder.export_caches) from a spawned process (its env
    and its session apart from the decoder's): {file: [limbs]}, each state as the plan hands it (XKV_METAS)."""
    import multiprocessing as mp
    cur = dict(os.environ)
    os.environ.clear()
    os.environ.update(_ENV0)         # a spawned child inherits os.environ at its start: the launch env, not the decoder's
    try:
        with mp.get_context("spawn").Pool(1) as pool:
            return pool.apply(_export_child, (plan, list(files), device, work_dir, secret))
    finally:
        os.environ.clear()
        os.environ.update(cur)


def import_cts(run, model, b, limbs):
    """Block b's cross cache states imported into the decoder's session from the encoder's limbs, {cache key: Ct}
    (the same chain: the limbs are residues of the same primes)."""
    from perseus import _pc
    top = _pc.chain_limbs(run.cfg["chain.preset"]) - 1
    cts, keys = model.xkv[b].cts(f"impl.b{b}.x")
    out = {}
    for c, k, x in zip(cts, keys, limbs):
        ct = run.fhe.encrypt(np.zeros(model.rt.dims.N, dtype=np.complex128), top - x["level"], x["noise_level"])
        run.fhe.import_limbs(ct, x["c0"].astype(np.uint64), x["c1"].astype(np.uint64), x["noise_factor"],
                             x["noise_level"])
        if (int(ct.level), int(ct.deg)) != (int(c.level), int(c.noise_deg)):
            raise RuntimeError(f"block {b} {k}: the encoder hands it at ({ct.level}, {ct.deg}), the decoder reads it at "
                               f"({c.level}, {c.noise_deg}): re-capture the decoder with --enc-plan")
        out[k] = ct
    return out


def transcribe(run, model, dmodel, rows, n_new, log=print, ref=None, check=None):
    """One utterance on the planned run (its cross caches given): the forced start's rows, then per token CutMax's
    argmax fed back encrypted until end-of-text or n_new tokens. Returns (the tokens, the client's decryptions of z;
    each token's server seconds: its forward from the feedback's issue to drained on the device, plus its argmax drained,
    the client's decryptions apart); `ref`: the float greedy transcript to report against, `check(pos, logits, out)`:
    a debug hook."""
    gen, P = model.gen, len(model.gen.start)
    for p in range(P - 1):
        run.token(p, rows, argmax=True)
    run.fhe.device_sync()
    ts = time.time()
    reg, _ = run.token(P - 1, rows, argmax=True)
    out, server = [], []
    for j in range(n_new):
        pos = P - 1 + j
        run.fhe.device_sync()
        fwd = time.time() - ts
        lg = dmodel.logits([run.ct(reg)])
        ta = time.time()
        z = run.argmax(pos)
        run.fhe.device_sync()
        server.append(fwd + time.time() - ta)
        zdec = dmodel.decode_z([run.ct(z)])
        cm, am = int(np.argmax(zdec)), int(np.argmax(lg))
        tok = model.token_of(cm)
        if check is not None:
            check(pos, np.asarray(lg), out)
        out.append(tok)
        if j + 1 < n_new and tok != gen.eot:
            ts = time.time()
            x = run.feedback(pos)
            reg, _ = run.token(pos + 1, rows, argmax=True, entry=x, variant=run.FB_VARIANT)
        want = ref[j] if ref is not None and j < len(ref) else None
        log(f"[asr]   pos {pos}: token {tok} (logits argmax {model.token_of(am)}, float {want}) z_mass "
            f"{zdec[cm]:.3f} {'OK' if tok == want else 'MISS'} server {server[-1]:.2f}s")
        if tok == gen.eot:
            break
    return out, server


def run_planned(a, log=lambda *x: print(*x, flush=True)):
    """The decoder planned on the device over the clips, the cross caches the encrypted encoder's (--enc-plan) or the
    float encoder's (encrypted as public values); each transcript checked against the float greedy (and scored against
    --refs). With the encoder, the clips go in chunks of --chunk: the encoder's session over a chunk, closed, then the
    decoder's over it (both do not fit on one GPU; a chunk's handed states stay in host memory)."""
    from perseus import _pc
    from perseus._paths import resolve
    from . import score as S
    from . import weights as W
    from .decoder import WhisperDecoder
    work = a.work_dir or os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), f"whisper_dec_{os.getpid()}")
    secret = client_secret() if a.enc_plan else None
    model, _inf = fake_decoder(a)
    gen = model.gen
    first = utterance(model, a.clips[0], a.max_new)
    model.fill_plain(first[0])                          # the twin's caches: the op streams (any utterance's)
    n_new = min(a.tokens or a.max_new, a.max_new, max_new(model))
    rows = teacher_rows(model, [gen.eot] * n_new)      # the forced start's rows, then placeholders the twin records
    refs = S.references(a.refs) if a.refs else None
    res, rates, texts = [], [], []
    step = a.chunk if a.enc_plan else len(a.clips)
    for c0 in range(0, len(a.clips), step):
        chunk = a.clips[c0:c0 + step]
        handed = None
        if a.enc_plan:      # the encrypted encoder first, in a process and a device session of its own
            t0 = time.time()
            handed = encoder_caches(resolve(a.enc_plan), chunk, a.device, os.path.join(work, "encoder"), secret)
            log(f"[asr] encoder: {len(handed)} clips' cross caches handed over in {time.time() - t0:.0f}s (incl. setup)")
        run = planned_run(model, a.plan, rows, work, log, debug=a.debug, secret=secret)
        # the client's decrypt helpers on the device session, at the model's geometry (the session's default: GPT-2's)
        dmodel = WhisperDecoder(run.fhe, model.store, model.cfgs, core=_pc, kv_offload=False, dims=model.rt.dims)
        for clip in chunk:
            enc_out, ref = first if clip == a.clips[0] else utterance(model, clip, a.max_new)
            if handed is not None:
                vecs = handed[clip]
            else:
                vecs = [np.nan_to_num(re) + 1j * np.nan_to_num(im) for K, V in W.cross_kv(model.store, enc_out)
                        for re, im in W.cross_cache_ref(K, V, len(enc_out), model.rt.dims)]
            per = len(vecs) // model.n_layers
            for b in range(model.n_layers):
                run.kv_first.pop(b, None)        # the last utterance's states freed first: the pool holds one set
                part = vecs[b * per:(b + 1) * per]
                run.give_kv(b, import_cts(run, model, b, part) if handed is not None else cross_cts(run, model, b, part))
            check = None
            if a.debug:                  # the device logits' top 3 against the float model's at each position
                def check(pos, lg, out, enc_out=enc_out):
                    fl = W.decoder_ref(model.store, enc_out, list(gen.start) + out)[-1][model.tokens]
                    top = lambda v: [(model.token_of(c), round(float(v[c]), 3)) for c in np.argsort(v)[::-1][:3]]
                    log(f"[asr:debug] pos {pos} device top {top(lg)} float top {top(fl)}")
            log(f"[asr] {os.path.basename(clip)}: float greedy {len(ref)} tokens; device free "
                f"{run.fhe.mem_info()[0] / 2 ** 30:.1f} GB")
            out, server = transcribe(run, model, dmodel, rows, n_new, log, ref, check)
            k = min(len(out), len(ref))
            same = out[:k] == ref[:k]
            rates.extend(server)
            log(f"[asr] {os.path.basename(clip)}: {len(out)} tokens {'IDENTICAL' if same else 'DIFFER'} to the float "
                f"greedy over {k}; {np.median(server):.2f} s/token (server, median)")
            cid = S.clip_id(clip)
            texts.append(((refs or {}).get(cid), S.text(out), S.text(ref)))
            log(f"[asr] {cid} encrypted: {texts[-1][1]}")
            if not same:
                log(f"[asr] {cid} float    : {texts[-1][2]}")
            if texts[-1][0] is not None:
                log(f"[asr] {cid} reference: {texts[-1][0]}")
            res.append(same)
        run.close()
        del run, dmodel, handed        # the session's device memory goes with its last holder: the next chunk's
        gc.collect()                   # encoder session needs the GPU
    scored = [t for t in texts if t[0] is not None]
    wers = (f" WER encrypted {S.wer([(r, e) for r, e, _ in scored]):.2f} % float "
            f"{S.wer([(r, f) for r, _, f in scored]):.2f} % over {len(scored)} clips" if scored else "")
    log(f"[asr] SUMMARY clips={len(res)} identical={sum(res)} s/token median={np.median(rates):.2f} (server, {len(rates)} tokens){wers}")
    return 0 if all(res) else 1


def main(argv=None, clips=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="a WAV clip or a directory of them")
    ap.add_argument("--n", type=int, default=1, help="clips (a directory's first n; --spread: n spread over it)")
    ap.add_argument("--spread", action="store_true", help="the n clips spread evenly over the directory's sorted list")
    ap.add_argument("--chunk", type=int, default=8, help="--enc-plan: clips per encoder session (host memory: ~1.6 GB "
                    "of handed states per clip, up to 3 copies while the pool hands them over)")
    ap.add_argument("--refs", default=None, metavar="TSV",
                    help="reference transcripts (id, seconds, text): each clip's printed, the WERs in the summary")
    ap.add_argument("--fake", action="store_true", help="the CPU twin (no GPU)")
    ap.add_argument("--enc", default=None, metavar="DIR",
                    help="the encrypted encoder's cross caches (run_encoder --fake --save DIR); default: the float "
                         "encoder's K / V, encrypted")
    ap.add_argument("--enc-plan", default=None, metavar="PLAN",
                    help="planned: the encrypted encoder on the device under PLAN hands the caches over (default: the "
                         "float encoder's, encrypted as public values)")
    ap.add_argument("--configs", default=DEC_CONFIGS)
    ap.add_argument("--plan", default=DEC_PLAN, help="the decode plan (its runtime; 'template': DECODE_TEMPLATE's)")
    ap.add_argument("--entry", default=None, metavar="LEVEL:DEG",
                    help="block 0's entry for every token (default: DEC_ENTRY on DEC_PLAN, else a fresh encryption's)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--max-new", type=int, default=128, help="tokens the server generates (public)")
    ap.add_argument("--tokens", type=int, default=0, help="stop after this many (a check; 0: --max-new)")
    ap.add_argument("--capture", default=None, metavar="GRAPH_DIR",
                    help="fake: capture the decode's stages at --capture-pos for the planner")
    ap.add_argument("--capture-pos", type=int, default=0)
    ap.add_argument("--structure-only", action="store_true", help="capture without magnitudes (capture_mags fills)")
    ap.add_argument("--device", type=int, default=None, help="planned: the GPU (correctness runs on 0)")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--debug", action="store_true", help="planned: each block decrypted against the twin (slow)")
    ap.add_argument("--force", action="store_true", help="planned: run on a busy GPU")
    a = ap.parse_args(argv)
    from perseus._paths import resolve
    from .run_encoder import clips as select
    from . import weights as W
    clips_given = clips
    a.configs = resolve(a.configs)
    if a.plan == "template":
        a.plan = None
    a.entry = tuple(int(x) for x in a.entry.split(":")) if a.entry else \
        (DEC_ENTRY if a.plan and os.path.abspath(resolve(a.plan)) == os.path.abspath(resolve(DEC_PLAN)) else None)
    a.clips = list(clips_given) if clips_given else select(a.audio, a.n, spread=a.spread, max_s=W.MAX_CLIP_S)
    if a.fake or a.capture:
        return capture(a) if a.capture else run_fake(a)
    from perseus.examples.gpt2.run_decode import gpu_busy
    busy = gpu_busy(a.device)
    if busy and not a.force:
        print(f"GPU {a.device} has other processes: {busy}; refusing (use --force)", file=sys.stderr)
        return 2
    a.plan = resolve(a.plan)
    return run_planned(a)


if __name__ == "__main__":
    sys.exit(main())
