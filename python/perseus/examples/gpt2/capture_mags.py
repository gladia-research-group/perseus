"""Fill a magnitude-suppressed GPT-2 capture with the numpy fake's exact magnitudes (perseus/impl/magtape.py):
the device capture records the structure only, the fake runs the same kernels on the same oracle rows.

    .venv/bin/python -m perseus.examples.gpt2.run_decode --tokens 1 --capture graphs/X --structure-only
    .venv/bin/python -m perseus.examples.gpt2.capture_mags graphs/X [--argmax]

The env is env.ENV plus the plan runtime and the same --set overrides. `--check REF_DIR` compares with a capture
that carries the device's own magnitudes. `--argmax --rows R.npy` sizes CutMax (block 13) as the per-node max over
logit rows (graphs/gpt2_decode/cutmax_rows.json, scripts/utils/cutmax_rows.py); `--blocks 13` writes that block.
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def fake_model(store_path, configs_path, packing):
    """The GPT-2 port on the numpy fake at full size: (model, fhe, inf)."""
    from perseus.impl import config, fake
    from perseus.impl.layout import Dims
    from . import weights
    from .model import Gpt2Model
    d = Dims.gpt2()
    fhe, inf = fake.make_fake(N=d.N, hid=d.hid, dim=d.dim, H=d.H, H_real=d.H_real, E=d.E, E_real=d.E_real,
                              complex_payload=True)
    model = Gpt2Model(inf, weights.RawStore(store_path), config.load_configs(configs_path), core=fake.core,
                      packing=packing, kv_offload=False)
    return model, fhe, inf


_ROWS = {}


def _cutmax_row(i):
    """One logit row through the CutMax code on the fake (a forked worker): its tape records."""
    import numpy as np
    from perseus.impl.fake import FakeCt
    from perseus.impl.layout import cutmax_tile_col_of_slot
    model, tapes, meta, rows = _ROWS["model"], _ROWS["tapes"], _ROWS["meta"], _ROWS["rows"]
    d, row = model.rt.dims, np.asarray(rows[i], dtype=np.float64)
    col = cutmax_tile_col_of_slot(np.arange(d.N), d.hid, d.N)            # the LM head's two tiles, packed t0 + i t1
    t1 = np.zeros(d.N)
    ok = d.N + col < row.shape[0]
    t1[ok] = row[d.N + col[ok]]
    model.argmax_encrypted([FakeCt(row[col] + 1j * t1, meta)], _ROWS["pos"])
    from perseus.impl import magtape
    return [(o, p, magtape._stats(s)) for o, p, s in tapes[model.n_layers + 1]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph_dir")
    ap.add_argument("--packing", choices=("cachemir", "cachemir_complex"), default="cachemir_complex")
    ap.add_argument("--argmax", action="store_true", help="also tape CutMax (block 13), as run_decode --argmax")
    ap.add_argument("--weights", default=os.environ.get("WEIGHTS_PATH"))
    ap.add_argument("--configs", default=os.environ.get("CONFIGS_PATH"))
    ap.add_argument("--plan", default=None, help="the plan dir whose runtime the device capture ran under, if any")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--pos", type=int, default=0, help="the captured position (FHE_GRAPH_CAPTURE_TOKEN)")
    ap.add_argument("--check", default=None, help="a capture with device magnitudes to compare against")
    ap.add_argument("--rows", nargs="+", default=[], metavar="NPY",
                    help="with --argmax: logit rows [N, vocab] whose CutMax also runs; block 13 takes the per-node max")
    ap.add_argument("--blocks", default=None, help="comma-separated stages to write (default: every taped stage)")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--keep-differing", action="store_true",
                    help="a step whose op list differs at this position keeps the capture's statistics (position 1's "
                         "K / V push)")
    a = ap.parse_args(argv)
    from . import env
    over = {**env.plan_runtime(a.plan, "n32"), "GPT2_PACKING": a.packing,
            **dict(kv.split("=", 1) for kv in a.set)}
    for k, v in {"CHAIN": "n32", **env.ENV, **over}.items():
        os.environ[k] = str(v)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if not a.configs:
        a.configs = env.plan_configs(a.plan, "n32", env.CONFIGS_N32)
    from . import oracle_io
    from perseus.impl import magtape
    model, fhe, inf = fake_model(a.weights, a.configs, a.packing)
    if a.argmax:
        # the all-slot ladders' radix groups (the device has their keys; the fake's band lacks the top ones)
        from perseus.impl.layout import ladder_chunks, ladder_radix_bits
        for g in ladder_chunks(1, inf.slots, ladder_radix_bits()):
            fhe._rot |= {int(k) for k in g}
    tape = magtape.attach(fhe, inf)
    tapes = magtape.tape_stages(model, tape, a.pos)
    entry, argmax_encrypted = {}, model.argmax_encrypted

    def argmax_entry(tiles, pos):                 # CutMax's input meta (levels, not values), for the rows
        entry.setdefault("meta", tiles[0].meta)
        return argmax_encrypted(tiles, pos)
    model.argmax_encrypted = argmax_entry
    os.environ.setdefault("STEPS_T", "128")
    os.environ["MULTI_T"] = str(a.pos + 1)
    rows = oracle_io.read_teacher_forced_inputs(oracle_io.RunConfig.from_env())[:a.pos + 1]
    t0 = time.time()
    model.run_decode(rows, argmax=a.argmax)
    model.argmax_encrypted = argmax_encrypted
    print(f"[mags] fake tape: {len(tapes)} stages, {sum(map(len, tapes.values()))} records, {time.time() - t0:.0f}s",
          flush=True)
    if a.argmax and a.rows:
        import multiprocessing
        import numpy as np
        b, recs = model.n_layers + 1, []
        for path in a.rows:
            _ROWS.update(model=model, tapes=tapes, meta=entry["meta"], pos=a.pos, rows=np.load(path, mmap_mode="r"))
            t0 = time.time()
            with multiprocessing.get_context("fork").Pool(a.procs) as pool:
                recs += pool.map(_cutmax_row, range(len(_ROWS["rows"])), chunksize=1)
            print(f"[mags] {path}: {len(_ROWS['rows'])} CutMax rows, {time.time() - t0:.0f}s", flush=True)
        tapes[b] = magtape.merge_max(recs)
    if a.blocks:
        tapes = {b: r for b, r in tapes.items() if b in {int(x) for x in a.blocks.split(",")}}
    done = magtape.fill_dir(a.graph_dir, tapes, ".filled" if a.check else "", keep_differing=a.keep_differing)
    if a.check:
        for b in done:
            print(f"[mags] block {b} vs {a.check} (median, p99, max rel diff):",
                  magtape.compare(os.path.join(a.check, f"block_{b}", "graph.json"),
                                  os.path.join(a.graph_dir, f"block_{b}", "graph.json.filled")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
