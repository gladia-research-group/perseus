"""Fill a structure-only encoder capture with the numpy fake's magnitudes (examples/vit/capture_mags_prefill.py's
recipe on Whisper's encoder): the planner sizes every refresh on them.

    .venv/bin/python -m perseus.examples.whisper.run_encoder --fake --audio CLIP --capture graphs/X --structure-only
    .venv/bin/python -m perseus.examples.whisper.capture_mags graphs/X --audio CLIP --configs CFG

Several clips: one capture each, merged by scripts/utils/merge_mags.py (the planner takes each value's largest).

The decoder (--part decoder), as examples/gpt2/capture_mags.py on run_asr's capture: positions 0 .. --pos teacher-forced
on the clip's float transcript, the cross caches the float encoder's; --rows sizes CutMax as the per-node max over
logit rows:

    .venv/bin/python -m perseus.examples.whisper.run_asr --fake --audio CLIP --capture graphs/D --structure-only
    .venv/bin/python -m perseus.examples.whisper.capture_mags graphs/D --part decoder --audio CLIP [--rows R.npy]
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_dir")
    ap.add_argument("--audio", required=True, nargs="+", help="WAV clip(s): the encoder's statistics are the max "
                    "over them (one process each); the decoder's: the first")
    ap.add_argument("--configs", default=None, help="default: the encoder's (run_encoder) or the decoder's (run_asr)")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--part", choices=("encoder", "decoder"), default="encoder")
    ap.add_argument("--pos", type=int, default=0, help="decoder: the captured position (run_asr --capture-pos)")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--enc-plan", default=None, help="decoder: the encoder plan whose cross caches it reads (run_asr)")
    ap.add_argument("--rows", nargs="+", default=[], metavar="NPY", help="decoder: logit rows sizing CutMax")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    if a.part == "decoder":
        return decoder_mags(a)
    if not a.configs:
        from .run_encoder import DEFAULT_CONFIGS
        a.configs = DEFAULT_CONFIGS
    from perseus.examples.gpt2 import env
    from .run_encoder import runtime_env
    over = {**runtime_env(a.plan), "GPT2_PACKING": "cachemir", **dict(kv.split("=", 1) for kv in a.set)}
    for k, v in {"CHAIN": "n32", **env.ENV, **over}.items():
        os.environ[k] = str(v)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from perseus.impl import magtape
    clips = list(a.audio)
    t0 = time.time()
    if len(clips) == 1:
        tapes = encoder_tapes(a.graph_dir, a.configs, clips[0])
    else:                   # one process per clip (the env above is theirs), the stages' statistics merged by max
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(a.procs, len(clips))) as pool:
            per = pool.starmap(encoder_tapes, [(a.graph_dir, a.configs, c) for c in clips])
        tapes = {b: magtape.merge_max([t[b] for t in per]) for b in per[0]}
    print(f"[mags] fake tape: {len(clips)} clip(s), {len(tapes)} stages, {sum(map(len, tapes.values()))} records, "
          f"{time.time() - t0:.0f}s", flush=True)
    magtape.fill_dir(a.graph_dir, tapes)
    return 0


def encoder_tapes(graph_dir, configs, clip):
    """{stage: magtape records} of the encoder's prefill of one clip on the numpy fake."""
    from perseus.impl import config, fake, magtape
    from . import weights as W
    from .encoder import WhisperEncoder
    from .run_encoder import dims
    d = dims()
    fhe, inf = fake.make_fake(N=d.N, hid=d.hid, dim=d.dim, H=d.H, H_real=d.H_real, E=d.E, E_real=d.E_real,
                              rot_steps=magtape.load_rot_steps(graph_dir, d.N), complex_payload=True)
    store = W.WhisperStore("encoder")
    model = WhisperEncoder(inf, store, config.load_configs(configs), core=fake.core)
    tape = magtape.attach(fhe, inf)
    tapes = magtape.tape_stages(model, tape, 0)
    os.environ.setdefault("STEPS_T", "128")
    os.environ["MULTI_T"] = str(W.T)
    model.preload_prefill()
    model.prefill(W.mel_rows(W.log_mel(W.load_audio(clip), store.path)))
    return {b: list(r) for b, r in tapes.items()}


def decoder_mags(a):
    from perseus._paths import resolve
    from perseus.examples.gpt2 import env
    from .run_asr import DEC_CONFIGS, decoder_env, teacher_rows, utterance
    from .run_encoder import dims
    over = decoder_env(a.plan, a.set)
    for k, v in {"CHAIN": "n32", **env.ENV, **over}.items():
        os.environ[k] = str(v)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from perseus.impl import config, fake, magtape
    from . import weights as W
    from .decoder import WhisperDecoder
    d = dims()
    fhe, inf = fake.make_fake(N=d.N, hid=d.hid, dim=d.dim, H=d.H, H_real=d.H_real, E=d.E, E_real=d.E_real,
                              rot_steps=magtape.load_rot_steps(a.graph_dir, d.N), complex_payload=True)
    model = WhisperDecoder(inf, W.WhisperStore("decoder"), config.load_configs(resolve(a.configs or DEC_CONFIGS)),
                           core=fake.core, kv_offload=False)
    if a.rows:            # the all-slot ladders' radix groups (as gpt2.capture_mags --argmax)
        from perseus.impl.layout import ladder_chunks, ladder_radix_bits
        for g in ladder_chunks(1, inf.slots, ladder_radix_bits()):
            fhe._rot |= {int(k) for k in g}
    from .run_asr import cross_metas
    model.cross_metas = cross_metas(a.enc_plan)
    from .run_asr import DEC_ENTRY, DEC_PLAN
    model.entry = DEC_ENTRY if a.plan and os.path.abspath(resolve(a.plan)) == os.path.abspath(resolve(DEC_PLAN)) else None
    enc_out, ref = utterance(model, a.audio[0], a.max_new)
    model.fill_plain(enc_out)
    tape = magtape.attach(fhe, inf)
    tapes = magtape.tape_stages(model, tape, a.pos)
    entry, argmax_encrypted = {}, model.argmax_encrypted

    def argmax_entry(tiles, pos):                 # CutMax's input meta, for the rows
        entry.setdefault("meta", tiles[0].meta)
        return argmax_encrypted(tiles, pos)
    model.argmax_encrypted = argmax_entry
    t0 = time.time()
    model.run_decode(teacher_rows(model, ref)[:a.pos + 1], argmax=True)
    model.argmax_encrypted = argmax_encrypted
    print(f"[mags] fake tape: {len(tapes)} stages, {sum(map(len, tapes.values()))} records, {time.time() - t0:.0f}s",
          flush=True)
    if a.rows:
        import multiprocessing
        import numpy as np
        from perseus.examples.gpt2 import capture_mags as G
        b, recs = model.n_layers + 1, []
        for path in a.rows:
            G._ROWS.update(model=model, tapes=tapes, meta=entry["meta"], pos=a.pos, rows=np.load(path, mmap_mode="r"))
            with multiprocessing.get_context("fork").Pool(a.procs) as pool:
                recs += pool.map(G._cutmax_row, range(len(G._ROWS["rows"])), chunksize=1)
            print(f"[mags] {path}: {len(G._ROWS['rows'])} CutMax rows", flush=True)
        tapes[b] = magtape.merge_max(recs)
    magtape.fill_dir(a.graph_dir, tapes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
