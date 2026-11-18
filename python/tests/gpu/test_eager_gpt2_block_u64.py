"""GPT-2 block 0 forward, eager on the GPU with 64-bit words (`_pc.w64.Session`: `Context<u64, Barrett64>` over the
n64 chain), vs the numpy fake (gate: relative error < 1e-3). `test_eager_gpt2_block.py` with the
word switched: the n64 chain is the perseus 64-bit chain (28 x 53-bit + 60-bit q0, one prime per level), so the impl
layer runs with `unit=1` and the n64 profile's levels (CHAIN=n64, AUTO_BTS_LEVEL 24, the legacy runtime levers that
`env.plan_runtime(None, "n64")` gives, with their THOR configs); the fake runs the same op stream with the same ceiling and landing.
Bootstraps are the TEST-ONLY decrypt + re-encrypt refresh (`debug_refresh=True`).

    tools/gpu.sh out/blk_u64.log .venv/bin/python -m pytest -m gpu python/tests/gpu/test_eager_gpt2_block_u64.py -q -s
(the extension built with `maturin develop --release -F pc-gpu/w64`).
"""
from __future__ import annotations

import json
import os
import time
import types

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

OLD = "/data02/users/mithrillo/projects/perseus/.cache"
WEIGHTS = os.environ.get("WEIGHTS_PATH") or f"{OLD}/models/openai-community/gpt2/classic/weights.bin.zip"
IO_DIR = os.environ.get("ALL_BLOCKS_IO_DIR") or f"{OLD}/oracle/gpt2/all_blocks_io"

LIMIT = 24        # the old n64 profile's AUTO_BTS_LEVEL (CKKS levels = primes on n64)
BTS_OUT = 14      # the w64 session's dense landing (word64.rs shadow_config_n64)


def _band(d):
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


def _core64():
    """`_pc` with the w64 module functions (decode_token_output, ...) in place of the u32 ones."""
    ns = {k: getattr(_pc, k) for k in dir(_pc) if not k.startswith("__")}
    ns.update({k: getattr(_pc.w64, k) for k in dir(_pc.w64) if not k.startswith("__")})
    return types.SimpleNamespace(**ns)


@pytest.mark.skipif(not (os.path.exists(WEIGHTS) and os.path.exists(os.path.join(IO_DIR, "all_blocks_L00_T1.json"))),
                    reason="GPT-2 weights / oracle io (all_blocks_L00_T1.json) missing")
def test_eager_gpt2_block0_u64(monkeypatch):
    from perseus.examples.gpt2 import env, oracle_io, weights
    from perseus.examples.gpt2.model import Gpt2Model
    from perseus.impl import config, fake
    from perseus.impl.layout import Dims

    for k, v in {"CHAIN": "n64", **env.ENV, **env.plan_runtime(None, "n64"), "GPT2_PACKING": "cachemir_complex",
                 "PERSEUS_SLOTVEC_CACHE": "0", "AUTO_BTS_LEVEL": str(LIMIT), "CACHE_READ_LEVEL_K": "17",
                 "CACHE_READ_LEVEL_V": "17", "FHE_LMHEAD_CAP": "22"}.items():
        monkeypatch.setenv(k, str(v))
    rows = oracle_io.read_teacher_forced_inputs(oracle_io.RunConfig(tokens=1, steps_t=1, io_dir=IO_DIR))
    with open(os.path.join(IO_DIR, "all_blocks_L00_T1.json")) as f:
        oracle = np.asarray(json.load(f)["res"][0], dtype=np.float64)
    # the legacy levers take the configs they were cut for (THOR GELU; env.plan_configs' n32-plan-free default
    # CONFIGS_N32 under the legacy levers diverges, fake included)
    cfgs = config.load_configs(env.CONFIGS_N32_LEGACY)
    store = weights.RawStore(WEIGHTS)
    d = Dims.gpt2()

    t0 = time.time()
    ffhe, finf = fake.make_fake(N=d.N, hid=d.hid, dim=d.dim, H=d.H, H_real=d.H_real, E=d.E, E_real=d.E_real,
                                limit=LIMIT, bts_out=BTS_OUT, unit=1, complex_payload=True)
    fmodel = Gpt2Model(finf, store, cfgs, core=fake.core, packing="cachemir_complex", kv_offload=False, n_layers=1,
                       unit=1)
    ref = _block0(fmodel, fake.core, rows[0])
    t_fake = time.time() - t0

    t0 = time.time()
    steps = _band(d)
    keys = _pc.KeySource.generate(steps, seed=5)
    cfg = _pc.Config.preset("n32_gpt2_decode")
    cfg["chain.chain"] = "n64"
    cfg["chain.composite_degree"] = 1
    cfg["bts.auto_bts_level"] = LIMIT
    try:
        sess = _pc.w64.Session(cfg, keys, pool=320, pt_slots=320, debug_refresh=True).complete_setup()
    except RuntimeError as e:
        if "is not built" in str(e):        # a build without the u64 kernels (maturin ... -F pc-gpu/w64)
            pytest.skip(str(e))
        raise
    fhe = sess.fhe
    t_setup = time.time() - t0
    core = _core64()
    model = Gpt2Model(fhe, store, cfgs, core=core, packing="cachemir_complex", kv_offload=False, n_layers=1, unit=1)
    assert not env.rotation_audit(fhe, model.rt.dims)
    model._bw = fmodel._bw
    t0 = time.time()
    out = _block0(model, core, rows[0])
    fhe.device_sync()
    t_gpu = time.time() - t0

    err = float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))
    o_fake = float(np.max(np.abs(ref - oracle)) / np.max(np.abs(oracle)))
    o_gpu = float(np.max(np.abs(out - oracle)) / np.max(np.abs(oracle)))
    print(f"\n[m11a] GPT-2 block 0 eager on u64 / n64: {len(steps)} rotation keys, session setup {t_setup:.1f} s "
          f"({sess.setup_seconds()[1]:.1f} s keygen); {sess.mem_report}; fake {t_fake:.1f} s, GPU {t_gpu:.1f} s; "
          f"refreshes {fhe.total_bootstraps} (reactive {fhe.n_reactive}; fake {ffhe.n_bootstraps}); deliberate "
          f"{model.rt.ops.n_bootstraps}")
    print(f"[m11a] block-0 output max |gpu - fake| / max |fake| = {err:.3e}  (gate < 1e-3); vs the PyTorch oracle: "
          f"fake {o_fake:.3e}, gpu {o_gpu:.3e}; |out|max {np.max(np.abs(out)):.3f}")
    assert np.isfinite(out).all()
    assert err < 1e-3
