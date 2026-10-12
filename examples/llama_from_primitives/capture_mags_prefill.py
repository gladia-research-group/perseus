"""Fill a structure-only Llama prefill capture with the numpy fake's magnitudes:

    .venv/bin/python -m examples.llama_from_primitives.run_prefill --tokens 16 --capture graphs/X --structure-only
    .venv/bin/python -m examples.llama_from_primitives.capture_mags_prefill graphs/X --tokens 16
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_dir")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--window", default="0")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--configs", default="configs/model/approximation/llama31_8b_n32/configs.json")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    from . import env as E
    over = {**E.plan_runtime(a.plan, "n32"), "LLAMA_PACKING": "cachemir_complex",
            **dict(kv.split("=", 1) for kv in a.set)}
    for k, v in {"CHAIN": "n32", **E.ENV, **over}.items():
        os.environ[k] = str(v)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from perseus.impl import fake, magtape
    from . import hf
    from . import weights as W
    from .config import load_configs
    from .prefill import LlamaPrefill
    store = W.SafetensorsStore(hf.NAME)
    cfgs = load_configs(a.configs)
    d = W.llama_dims(store.config)
    fhe, inf = fake.make_fake(N=d.N, hid=d.hid, dim=d.dim, H=d.H, H_real=d.H_real, E=d.E, E_real=d.E_real,
                              rot_steps=magtape.load_rot_steps(a.graph_dir, d.N),
                              complex_payload=True)
    model = LlamaPrefill(inf, W.LlamaWeights(store, cfgs, a.configs, d, "cachemir_complex"), cfgs, core=fake.core,
                         n_layers=a.layers, packing="cachemir_complex")
    tape = magtape.attach(fhe, inf)
    tapes = magtape.tape_stages(model, tape, 1)
    toks = hf.text_windows(int(a.window) + 1, 128)[int(a.window)][:a.tokens + 1]
    t0 = time.time()
    model.prefill(toks)
    print(f"[mags] fake tape: {len(tapes)} stages, {sum(map(len, tapes.values()))} records, {time.time() - t0:.0f}s",
          flush=True)
    magtape.fill_dir(a.graph_dir, tapes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
