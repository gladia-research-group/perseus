"""Fill a structure-only GPT-2 prefill capture with the numpy fake's magnitudes:

    .venv/bin/python -m examples.gpt2_from_primitives.run_prefill --tokens 64 --capture graphs/X --structure-only
    .venv/bin/python -m examples.gpt2_from_primitives.capture_mags_prefill graphs/X --tokens 64
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_dir")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--weights", default=os.environ.get("WEIGHTS_PATH"))
    ap.add_argument("--configs", default="configs/model/approximation/gpt2_base_n32/configs.json")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--check", default=None)
    a = ap.parse_args(argv)
    from . import env
    over = {**env.plan_runtime(a.plan, "n32"), "GPT2_PACKING": "cachemir", **dict(kv.split("=", 1) for kv in a.set)}
    for k, v in {"CHAIN": "n32", **env.ENV, **over}.items():
        os.environ[k] = str(v)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    from perseus import _core
    from perseus.impl import config, fake, magtape
    from perseus.impl.layout import Dims
    from . import weights
    from .prefill import Gpt2Prefill
    d = Dims.gpt2()
    fhe, inf = fake.make_fake(N=d.N, hid=d.hid, dim=d.dim, H=d.H, H_real=d.H_real, E=d.E, E_real=d.E_real,
                              rot_steps=magtape.load_rot_steps(a.graph_dir, d.N),
                              complex_payload=True)
    model = Gpt2Prefill(inf, weights.RawStore(a.weights), config.load_configs(a.configs), core=fake.core)
    tape = magtape.attach(fhe, inf)
    tapes = magtape.tape_stages(model, tape, 0)
    os.environ.setdefault("STEPS_T", "128")
    os.environ["MULTI_T"] = str(a.tokens)
    rows = np.asarray(_core.read_teacher_forced_inputs(_core.RunConfig.from_env())[:a.tokens])
    t0 = time.time()
    model.prefill(rows)
    print(f"[mags] fake tape: {len(tapes)} stages, {sum(map(len, tapes.values()))} records, {time.time() - t0:.0f}s",
          flush=True)
    done = magtape.fill_dir(a.graph_dir, tapes, ".filled" if a.check else "")
    if a.check:
        for b in done:
            print(f"[mags] block {b} vs {a.check}:", magtape.compare(os.path.join(a.check, f"block_{b}", "graph.json"),
                                                                       os.path.join(a.graph_dir, f"block_{b}", "graph.json.filled")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
