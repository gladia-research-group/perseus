"""GPT-2 block 0 forward, eager on the GPU (`_pc.Session` + `_pc.Fhe` under `examples/gpt2`), vs the float
reference (gate: relative error < 1e-3).

The reference is the same `transformer_block` on the numpy fake (`impl/fake.py`): the float64 mirror of the exact op
stream the GPU runs (the approximations of `ref.py` included), so the difference is FHE noise only. Bootstraps are
stubbed by the TEST-ONLY decrypt + re-encrypt refresh (`Session(debug_refresh=True)`).
The distances of both to the PyTorch oracle's block output (`res` of all_blocks_L00_T1.json; approximation error)
are printed for information.

Weights and the oracle rows: WEIGHTS_PATH / ALL_BLOCKS_IO_DIR, defaulting to the old repo's `.cache`.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

OLD = "/data02/users/mithrillo/projects/perseus/.cache"
WEIGHTS = os.environ.get("WEIGHTS_PATH") or f"{OLD}/models/openai-community/gpt2/classic/weights.bin.zip"
IO_DIR = os.environ.get("ALL_BLOCKS_IO_DIR") or f"{OLD}/oracle/gpt2/all_blocks_io"


def _band(d):
    """The rotation steps the GPT-2 port needs (examples/gpt2/env.rotation_audit + the radix ladder groups)."""
    from perseus.impl.layout import attention_rot_steps, ladder_radix_bits, ladder_radix_steps, linear_rot_steps
    steps = set()
    for d_in, d_out in ((d.hid, d.hid), (d.hid, d.E), (d.E, d.hid), (d.hid, d.N), (d.N, d.hid)):
        steps |= linear_rot_steps(d.N, d_in, d_out)
    steps |= attention_rot_steps(d)
    steps |= ladder_radix_steps(d, ladder_radix_bits())
    return sorted({int(s) % d.N for s in steps} - {0})


def _block0(model, core, row):
    model.start()
    model.token_begin(0)
    x = model.encode_input(row)
    label, _prefix, fn = model.token_stages(x, 0)[0]
    assert label == "blk0"
    y = fn(x)
    return np.asarray(core.decode_token_output(model.inf, y), dtype=np.float64)


@pytest.mark.skipif(not (os.path.exists(WEIGHTS) and os.path.exists(os.path.join(IO_DIR, "all_blocks_L00_T1.json"))),
                    reason="GPT-2 weights / oracle io (all_blocks_L00_T1.json) missing")
def test_eager_gpt2_block0(monkeypatch):
    from perseus.examples.gpt2 import env, oracle_io, weights
    from perseus.examples.gpt2.model import Gpt2Model
    from perseus.impl import config, fake
    from perseus.impl.layout import Dims

    # the session switches the impl layer reads (the eager n32 decode: env.ENV + the plan-free runtime levers)
    for k, v in {"CHAIN": "n32", **env.ENV, **env.plan_runtime(None, "n32"), "GPT2_PACKING": "cachemir_complex",
                 "PERSEUS_SLOTVEC_CACHE": "0"}.items():
        monkeypatch.setenv(k, str(v))
    rows = oracle_io.read_teacher_forced_inputs(oracle_io.RunConfig(tokens=1, steps_t=1, io_dir=IO_DIR))
    with open(os.path.join(IO_DIR, "all_blocks_L00_T1.json")) as f:
        oracle = np.asarray(json.load(f)["res"][0], dtype=np.float64)
    cfgs = config.load_configs(env.CONFIGS_N32)
    store = weights.RawStore(WEIGHTS)
    d = Dims.gpt2()

    # the float reference: the numpy fake
    t0 = time.time()
    ffhe, finf = fake.make_fake(N=d.N, hid=d.hid, dim=d.dim, H=d.H, H_real=d.H_real, E=d.E, E_real=d.E_real,
                                complex_payload=True)
    fmodel = Gpt2Model(finf, store, cfgs, core=fake.core, packing="cachemir_complex", kv_offload=False, n_layers=1)
    ref = _block0(fmodel, fake.core, rows[0])
    t_fake = time.time() - t0

    # the GPU, eager
    t0 = time.time()
    steps = _band(d)
    keys = _pc.KeySource.generate(steps, seed=5)
    sess = _pc.Session(_pc.Config.preset("n32_gpt2_decode"), keys, pool=320, pt_slots=320,
                       debug_refresh=True).complete_setup()
    fhe = sess.fhe
    t_setup = time.time() - t0
    model = Gpt2Model(fhe, store, cfgs, core=_pc, packing="cachemir_complex", kv_offload=False, n_layers=1)
    assert not env.rotation_audit(fhe, model.rt.dims)
    model._bw = fmodel._bw                       # the same encoded weights (host numpy)
    t0 = time.time()
    out = _block0(model, _pc, rows[0])
    fhe.device_sync()
    t_gpu = time.time() - t0

    err = float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))
    o_fake = float(np.max(np.abs(ref - oracle)) / np.max(np.abs(oracle)))
    o_gpu = float(np.max(np.abs(out - oracle)) / np.max(np.abs(oracle)))
    print(f"\n[m5d] GPT-2 block 0 eager: {len(steps)} rotation keys, session setup {t_setup:.1f} s "
          f"({sess.setup_seconds()[1]:.1f} s keygen); fake {t_fake:.1f} s, GPU {t_gpu:.1f} s; refreshes "
          f"{fhe.total_bootstraps} (reactive {fhe.n_reactive}; fake {ffhe.n_bootstraps}); deliberate "
          f"{model.rt.ops.n_bootstraps}")
    print(f"[m5d] block-0 output max |gpu - fake| / max |fake| = {err:.3e}  (gate < 1e-3); vs the PyTorch oracle: "
          f"fake {o_fake:.3e}, gpu {o_gpu:.3e}; |out|max {np.max(np.abs(out)):.3f}")
    assert np.isfinite(out).all()
    assert err < 1e-3
