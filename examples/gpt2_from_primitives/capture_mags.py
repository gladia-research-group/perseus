"""Fill a magnitude-suppressed GPT-2 capture with the numpy fake's exact magnitudes (perseus/impl/magtape.py):
the device capture records the structure only (no per-node decrypt), the fake runs the same kernels on the same
oracle rows and writes each node's statistics.

    # 1. the device capture, structure only
    .venv/bin/python -m examples.gpt2_from_primitives.run_decode --tokens 1 --capture graphs/X --structure-only
    # 2. the same token on the fake (CPU), merged into graphs/X/block_<b>/graph.json
    .venv/bin/python -m examples.gpt2_from_primitives.capture_mags graphs/X [--argmax]

The switches that shape the op stream must be the device run's: the env is env.ENV plus the plan runtime and the
same --set overrides. `--check REF_DIR` compares with a capture that carries the device's own magnitudes instead
of writing.
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
    a = ap.parse_args(argv)
    from . import env
    over = {**env.plan_runtime(a.plan, "n32"), "GPT2_PACKING": a.packing,
            **dict(kv.split("=", 1) for kv in a.set)}
    for k, v in {"CHAIN": "n32", **env.ENV, **over}.items():
        os.environ[k] = str(v)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if not a.configs:
        a.configs = env.plan_configs(a.plan, "n32", env.CONFIGS_N32)
    from perseus import _core
    from perseus.impl import magtape
    model, fhe, inf = fake_model(a.weights, a.configs, a.packing)
    tape = magtape.attach(fhe, inf)
    tapes = magtape.tape_stages(model, tape, a.pos)
    os.environ.setdefault("STEPS_T", "128")
    os.environ["MULTI_T"] = str(a.pos + 1)
    rows = _core.read_teacher_forced_inputs(_core.RunConfig.from_env())[:a.pos + 1]
    t0 = time.time()
    model.run_decode(rows, argmax=a.argmax)
    print(f"[mags] fake tape: {len(tapes)} stages, {sum(map(len, tapes.values()))} records, {time.time() - t0:.0f}s",
          flush=True)
    done = magtape.fill_dir(a.graph_dir, tapes, ".filled" if a.check else "")
    if a.check:
        for b in done:
            print(f"[mags] block {b} vs {a.check} (median, p99, max rel diff):",
                  magtape.compare(os.path.join(a.check, f"block_{b}", "graph.json"),
                                  os.path.join(a.graph_dir, f"block_{b}", "graph.json.filled")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
