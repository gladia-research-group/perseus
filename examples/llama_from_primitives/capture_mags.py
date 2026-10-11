"""Fill a magnitude-suppressed capture with the fake's exact magnitudes (perseus/impl/magtape.py).

    # 1. the device capture, structure only (fast: no per-node decrypt)
    .venv/bin/python -m examples.llama_from_primitives.run_decode --tokens 1 --capture graphs/X --structure-only
    # 2. the same position on the numpy fake (CPU), its tape merged into graphs/X/block_<b>/graph.json
    .venv/bin/python -m examples.llama_from_primitives.capture_mags graphs/X

`--check REF_DIR` compares against a capture taken with magnitudes (the device decrypts) instead
of writing: the relative difference of every statistic per node, summarized.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def run_tape(cfgs, configs_path, toks, pos, n_layers=None, packing="cachemir_complex", overrides=None):
    """{stage b: tape records} of the stages of position `pos` on the fake."""
    # the switches that shape the op stream must be the device session's (env.ENV wins over a
    # sourced scripts/local_env.sh exactly as in open_session), or the tape describes another graph
    from . import env as E
    for k, v in {**E.ENV, **(overrides or {})}.items():
        os.environ[k] = str(v)
    from perseus.impl import fake, magtape

    from . import hf
    from . import weights as W
    from .model import LlamaModel
    store = W.SafetensorsStore(hf.NAME)
    d = W.llama_dims(store.config)
    fhe, inf = fake.make_fake(N=d.N, hid=d.hid, dim=d.dim, H=d.H, H_real=d.H_real, E=d.E, E_real=d.E_real,
                              complex_payload=True)
    tape = magtape.attach(fhe, inf)
    lw = W.LlamaWeights(store, cfgs, configs_path, d, packing)
    model = LlamaModel(inf, lw, cfgs, core=fake.core, n_layers=n_layers, packing=packing, kv_offload=False)
    per_stage = {}
    enter, exit_ = model.enter_stage, model.exit_stage

    def enter_stage(b, p):
        if p == pos:
            tape.records = []
            tape.active = True
        return enter(b, p)

    def exit_stage(b, cap):
        if tape.active:
            per_stage[b] = tape.records
            tape.active = False
        return exit_(b, cap)
    model.enter_stage, model.exit_stage = enter_stage, exit_stage
    model.run_decode(toks[:pos + 1])
    return per_stage


def compare(ref_graph, filled_graph):
    """Per statistic: the median and max relative difference over the nodes both carry."""
    a = json.load(open(ref_graph))["nodes"]; b = json.load(open(filled_graph))["nodes"]
    out = {}
    for k in ("output_max_abs", "output_max_coeff", "output_max_coeff_ac", "output_max_dev"):
        r = [abs(x[k] - y[k]) / max(abs(x[k]), 1e-12) for x, y in zip(a, b)
             if k in x and k in y and x["op_type"] not in ("auto_bootstrap", "hint", "deliberate_bootstrap")]
        if r:
            r = np.asarray(r)
            out[k] = (float(np.median(r)), float(np.quantile(r, 0.99)), float(r.max()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph_dir")
    ap.add_argument("--configs", default="configs/model/approximation/llama31_8b_n32/configs.json")
    ap.add_argument("--window", required=True, help=".npy token window (the device capture's)")
    ap.add_argument("--pos", type=int, default=1, help="the captured position (FHE_GRAPH_CAPTURE_TOKEN)")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="the device capture's --set overrides (the switches that shape the op stream)")
    ap.add_argument("--check", default=None, help="a capture with device magnitudes to compare against")
    a = ap.parse_args(argv)
    from perseus.impl import magtape

    from .config import load_configs
    cfgs = load_configs(a.configs)
    toks = np.load(a.window)
    toks = toks if toks.ndim == 1 else toks[0]
    t0 = time.time()
    tapes = run_tape(cfgs, a.configs, toks, a.pos, a.layers,
                     overrides=dict(kv.split("=", 1) for kv in a.set))
    print(f"[mags] fake tape: {len(tapes)} stages, {sum(map(len, tapes.values()))} records, {time.time() - t0:.0f}s",
          flush=True)
    for b, rec in sorted(tapes.items()):
        g = os.path.join(a.graph_dir, f"block_{b}", "graph.json")
        if not os.path.exists(g):
            print(f"[mags] block {b}: no graph, skipped")
            continue
        out = g + ".filled" if a.check else None
        used = magtape.fill_graph(g, rec, 0, out)
        print(f"[mags] block {b}: {used}/{len(rec)} records merged", flush=True)
        if a.check:
            print(f"[mags] block {b} vs {a.check}: (median, p99, max) rel diff",
                  compare(os.path.join(a.check, f"block_{b}", "graph.json"), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
