"""Fill a structure-only ViT capture with the numpy fake's magnitudes (one image):

    .venv/bin/python -m perseus.examples.vit.run_prefill --n 1 --capture graphs/X --structure-only
    .venv/bin/python -m perseus.examples.vit.capture_mags_prefill graphs/X --image logs/vit/img/0.jpg
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_dir")
    ap.add_argument("--image", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    from perseus.examples.gpt2 import env
    over = {**env.plan_runtime(a.plan, "n32"), "GPT2_PACKING": "cachemir", **dict(kv.split("=", 1) for kv in a.set)}
    for k, v in {"CHAIN": "n32", **env.ENV, **over}.items():
        os.environ[k] = str(v)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from perseus.impl import config, fake, magtape
    from perseus.impl.layout import Dims
    from . import weights as V
    from .prefill import VitPrefill
    d = Dims.gpt2()
    fhe, inf = fake.make_fake(N=d.N, hid=d.hid, dim=d.dim, H=d.H, H_real=d.H_real, E=d.E, E_real=d.E_real,
                              rot_steps=magtape.load_rot_steps(a.graph_dir, d.N), complex_payload=True)
    model = VitPrefill(inf, V.VitStore(), config.load_configs(a.configs), core=fake.core)
    tape = magtape.attach(fhe, inf)
    tapes = magtape.tape_stages(model, tape, 0)
    os.environ.setdefault("STEPS_T", "128")
    os.environ["MULTI_T"] = str(V.T)
    t0 = time.time()
    model.preload_prefill()
    model.prefill(V.patch_rows(np.load(a.image) if a.image.endswith(".npy") else V.pixels(a.image)))
    print(f"[mags] fake tape: {len(tapes)} stages, {sum(map(len, tapes.values()))} records, {time.time() - t0:.0f}s",
          flush=True)
    magtape.fill_dir(a.graph_dir, tapes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
