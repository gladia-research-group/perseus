"""Fill a structure-only BERT capture with the numpy fake's magnitudes (the 128-token text):

    .venv/bin/python -m examples.bert_from_primitives.run_bert --capture graphs/X --structure-only
    .venv/bin/python -m examples.bert_from_primitives.capture_mags graphs/X
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_dir")
    ap.add_argument("--configs", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    from examples.gpt2_from_primitives import env
    from .run_bert import ARM, CONFIGS
    over = {**ARM, "GPT2_PACKING": "cachemir", **dict(kv.split("=", 1) for kv in a.set)}
    for k, v in {"CHAIN": "n32", **env.ENV, **over}.items():
        os.environ[k] = str(v)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from perseus.impl import config, fake, magtape
    from perseus.impl.layout import Dims
    from . import data as D
    from .model import BertPrefill
    d = Dims.gpt2()
    fhe, inf = fake.make_fake(N=d.N, hid=d.hid, dim=d.dim, H=d.H, H_real=d.H_real, E=d.E, E_real=d.E_real,
                              rot_steps=magtape.load_rot_steps(a.graph_dir, d.N), complex_payload=True)
    model = BertPrefill(inf, D.store(), config.load_configs(a.configs or CONFIGS), core=fake.core)
    tape = magtape.attach(fhe, inf)
    tapes = magtape.tape_stages(model, tape, 0)
    ids, n = D.text128()
    t0 = time.time()
    model.forward(D.embed(ids), n)
    print(f"[mags] fake tape: {len(tapes)} stages, {sum(map(len, tapes.values()))} records, {time.time() - t0:.0f}s",
          flush=True)
    magtape.fill_dir(a.graph_dir, tapes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
