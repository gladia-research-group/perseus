"""examples/gpt2_from_primitives on a real n32 session: every primitive port against the C++
composite it mirrors, on the SAME ciphertext (the composites are the reference here and
nowhere else). Runs under the port's env (dense, real payload, folded ln_1/ln_2)."""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.gpt2_from_primitives import env as penv  # noqa: E402

DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES", "3")
CHAIN = os.environ.get("CHAIN") or "n32"   # source scripts/local_env.sh for the same chain
penv.export_env(DEVICE, chain=CHAIN, CKKS_COMPLEX="0")   # real slots; before perseus._core is imported

from perseus.impl import attention, config, layout, linear, norm  # noqa: E402
from perseus.impl import activation as gelu  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402
from examples.gpt2_from_primitives import block, head, weights  # noqa: E402
from examples.gpt2_from_primitives.model import Gpt2Model  # noqa: E402

CFG = ROOT / ("configs/model/approximation/"
              + ("gpt2_base_n32" if CHAIN == "n32" else "gpt2_base")
              + "/configs.json")
WEIGHTS = os.environ.get("WEIGHTS_PATH",
                         str(ROOT / ".cache/models/openai-community/gpt2/classic/weights.bin.zip"))


@pytest.fixture(scope="module")
def S():
    from perseus import _core
    sess = penv.open_session(DEVICE, complex_payload=False, chain=CHAIN)
    inf = sess.inf
    rt = Rt.from_inf(inf, _core, int(sess.options.ckks.composite_degree))
    parsed = _core.load_configs(str(CFG))
    cfgs = config.load_configs(str(CFG))
    store = weights.RawStore(WEIGHTS)
    yield dict(sess=sess, inf=inf, core=_core, rt=rt, parsed=parsed, cfgs=cfgs, store=store)
    sess.close()


def _slots(S, ct):
    return S["rt"].ops.decrypt_slots(ct)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def test_rotation_band_covers_the_port(S):
    assert penv.rotation_audit(S["inf"].fhe, S["rt"].dims) == []


@pytest.mark.parametrize("d_in,d_out", [(1024, 1024), (1024, 4096), (4096, 1024)])
def test_linear_vs_core(S, d_in, d_out):
    inf, core, rt = S["inf"], S["core"], S["rt"]
    rng = np.random.default_rng(d_in + d_out)
    W = rng.standard_normal((d_in, d_out)) * 0.05
    b = rng.standard_normal(d_out) * 0.1
    name = f"tw_{d_in}_{d_out}"
    inf.set_weight(name, W, d_in, d_out)
    inf.set_bias(name + "_bias", b, d_in, d_out)
    if d_in == 1024:
        x = rt.ops.encode_token(rng.standard_normal(768) * 0.3)
    else:
        # a down-linear input: make one from an up-linear output
        xin = rt.ops.encode_token(rng.standard_normal(768) * 0.3)
        Wu = rng.standard_normal((1024, 4096)) * 0.05
        x = linear.linear(rt, xin, linear.EncodedLinear.encode(Wu, rt.dims.N, 1024, 4096))
    y_ref = core.linear(inf, x, name, d_in, d_out)
    y = linear.linear(rt, x, linear.EncodedLinear.encode(W, rt.dims.N, d_in, d_out, b))
    a, r = _slots(S, y), _slots(S, y_ref)
    assert _rel(a, r) < 1e-3, _rel(a, r)
    assert y.level == y_ref.level, (y, y_ref)
    inf.evict_weights(name); inf.evict_weights(name + "_bias")


def test_norm_vs_core(S):
    inf, core, rt = S["inf"], S["core"], S["rt"]
    name = "transformer.h.2.ln_1"
    inf.set_norm_cfg("ln_1", S["parsed"].norm[name])
    cfg = S["cfgs"].norm[name]
    rng = np.random.default_rng(2)
    x = rt.ops.encode_token(rng.standard_normal(768) * 0.5)
    inf.capture_t = 4
    y_ref = core.norm(inf, x, "ln_1")
    y = norm.norm(rt, x, cfg, 4)
    a, r = _slots(S, y), _slots(S, y_ref)
    assert _rel(a, r) < 2e-2, _rel(a, r)
    assert abs(y.level - y_ref.level) <= 2, (y, y_ref)


def test_gelu_vs_core(S):
    inf, core, rt = S["inf"], S["core"], S["rt"]
    name = "transformer.h.0.mlp.act"
    inf.set_gelu_cfg("mlp.act", S["parsed"].softgelu[name])
    cfg = S["cfgs"].gelu[name]
    rng = np.random.default_rng(3)
    xin = rt.ops.encode_token(rng.standard_normal(768) * 0.5)
    Wu = rng.standard_normal((1024, 4096)) * 0.2
    x = linear.linear(rt, xin, linear.EncodedLinear.encode(Wu, rt.dims.N, 1024, 4096))
    y_ref = core.gelu_approx(inf, x, "mlp.act")
    y = gelu.gelu(rt, x, cfg)
    a, r = _slots(S, y), _slots(S, y_ref)
    assert _rel(a, r) < 2e-2, _rel(a, r)


def test_attention_vs_core(S):
    inf, core, rt = S["inf"], S["core"], S["rt"]
    d = rt.dims
    name = "transformer.h.0.attn"
    inf.set_softmax_cfg("attn", S["parsed"].softmax[name])
    cfg = S["cfgs"].softmax[name]
    inf.block_prefix = core.block_scope(0)
    core.prepare_mha_masks(inf); core.prepare_vcache(inf)
    kv = attention.KVCache(d.d_head)
    rng = np.random.default_rng(4)
    q = None
    zero = None
    for tok in range(3):
        def tokvec():
            v = rng.standard_normal(1024) * 0.5
            v[(np.arange(1024) % d.H) >= d.H_real] = 0.0      # padded heads carry nothing
            return v
        k, v, qv = tokvec(), tokvec(), tokvec()
        if zero is None:
            fresh = rt.ops.encode_token(np.zeros(768))
            zero = rt.ops.sub(fresh, fresh)
        # 1024-lane (head-interleaved) token cts: a plaintext add onto a zero ct
        kct = rt.ops.add_pt(zero, layout.lane_vec(k, d.N, d.t))
        vct = rt.ops.add_pt(zero, layout.lane_vec(v, d.N, d.t))
        qct = rt.ops.add_pt(zero, layout.lane_vec(qv, d.N, d.t))
        core.cache_kv_push(inf, kct, vct)
        attention.cache_k_push(rt, kv, kct)
        attention.cache_v_push(rt, kv, vct)
        q = qct
    s_ref = core.qkt(inf, q)
    s = attention.qkt(rt, kv, q)
    assert _rel(_slots(S, s), _slots(S, s_ref[0])) < 1e-2
    p_ref = core.attention_softmax_thor(inf, s_ref, "attn")
    p = attention.softmax_thor(rt, s, cfg, kv.k_count)
    assert _rel(_slots(S, p), _slots(S, p_ref[0])) < 2e-2
    o_ref = core.softmax_v(inf, p_ref)
    o = attention.softmax_v(rt, kv, p)
    assert _rel(_slots(S, o), _slots(S, o_ref)) < 2e-2


def _block_state(S, b):
    inf, core = S["inf"], S["core"]
    st = core.load_block_state(inf, core.WeightStore.from_zip(WEIGHTS), S["parsed"],
                               core.BootstrapPlan(), b)
    core.load_block_to_device(inf, st)
    core.install_block_state(inf, st)
    return st


def test_block_vs_core(S):
    inf, core, rt = S["inf"], S["core"], S["rt"]
    d = rt.dims
    st = _block_state(S, 0)
    core.reset_kv_cache(inf, 1)
    inf.block_prefix = core.block_scope(0)
    model = Gpt2Model(inf, S["store"], S["cfgs"], core=core, n_layers=1)
    model.start()
    w = model.block_weights(0)
    os.environ["MULTI_T"] = "4"
    cfg = core.RunConfig.from_env()
    inputs = core.read_teacher_forced_inputs(cfg)
    assert len(inputs) >= 2, cfg
    errs = []
    for t in range(2):
        x = rt.ops.encode_token(inputs[t])
        inf.capture_t = t
        core.load_block_to_device(inf, st)      # EncGPT2.run_block: (re)install per token
        core.install_block_state(inf, st)
        inf.block_prefix = core.block_scope(0)
        core.kv_block_prologue(inf, 0, 1)
        y_ref = core.transformer_block(inf, x)
        core.block_release(inf, 0)
        y = block.transformer_block(rt, x, w, model.kv[0], S["cfgs"].block(0), t)
        a, r = rt.ops.decode_token(y), rt.ops.decode_token(y_ref)
        errs.append(_rel(a, r))
    core.evict_block_from_device(inf, st)
    assert max(errs) < 5e-2, errs


def test_lm_head_and_cutmax_vs_core(S):
    inf, core, rt = S["inf"], S["core"], S["rt"]
    model = Gpt2Model(inf, S["store"], S["cfgs"], core=core, n_layers=1)
    # a hidden state that reproduces the oracle's token-0 logits (CutMax's schedule is
    # calibrated for GPT-2 logit statistics, so a random vector is outside its envelope)
    os.environ["STEPS_T"] = "128"
    gt = core.read_lm_head_steps(core.RunConfig.from_env())
    assert gt.T > 0, "no per-step logits oracle"
    Wlm = weights.lm_head_matrix(S["store"], S["cfgs"].ln_f, rt.dims, model.vocab)
    target = np.asarray(gt.logits[0])
    hvec = np.linalg.lstsq(Wlm[:768].T, target, rcond=None)[0]
    h = rt.ops.encode_token(hvec)
    tiles = head.lm_head(rt, h, model.lm_tiles())
    lg = model.logits(tiles)
    want = hvec @ Wlm[:768]
    assert _rel(lg, want) < 1e-2 and int(np.argmax(want)) == int(np.argmax(target))
    z_ref = core.cutmax_argmax(inf, tiles, model.vocab, core.cutmax_config_from_calib(S["parsed"].cutmax))
    z = head.cutmax_argmax(rt, tiles, model.vocab, S["cfgs"].cutmax, model.ones)
    zr, zp = model.decode_z(z_ref), model.decode_z(z)
    assert int(np.argmax(zr)) == int(np.argmax(zp)) == int(np.argmax(lg))
    assert abs(zr.max() - zp.max()) < 0.05


@pytest.mark.slow
def test_capture_plan_planned_one_block(S, tmp_path):
    """The production loop from primitives: capture block graphs (token 0) -> plan (CPU)
    -> run under the plan (strict). One block plus the tail, two tokens; the planned
    forward must match the eager one."""
    import subprocess
    inf, core, rt = S["inf"], S["core"], S["rt"]
    model = Gpt2Model(inf, S["store"], S["cfgs"], core=core, n_layers=1)
    os.environ["MULTI_T"] = "4"
    inputs = core.read_teacher_forced_inputs(core.RunConfig.from_env())
    graph_dir = tmp_path / "graphs"
    model.set_capture(str(graph_dir))
    eager = model.run_decode(inputs[:2])
    os.environ.pop("FHE_GRAPH_DIR", None)
    assert (graph_dir / "block_0" / "graph.json").exists() and (graph_dir / "block_1" / "graph.json").exists()
    out_name = f"_prim_test_{os.getpid()}"
    r = subprocess.run(["bash", str(ROOT / "examples/gpt2_from_primitives/make_plan.sh"),
                        str(graph_dir), out_name], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    plan_dir = ROOT / "bootstrap_placements" / out_name
    try:
        model2 = Gpt2Model(inf, S["store"], S["cfgs"], core=core, n_layers=1)
        model2.load_plans(str(plan_dir))
        planned = model2.run_decode(inputs[:2])
        for e, p in zip(eager, planned):
            assert _rel(p["logits"], e["logits"]) < 5e-2 and p["top1"] == e["top1"]
    finally:
        import shutil
        shutil.rmtree(plan_dir, ignore_errors=True)
