"""The complex-payload arm (CKKS_COMPLEX=1, the C++ decode configuration) of the primitives port
against the C++ composites on one complex session: the K + iV pair push feeding the same
q.K^T / softmax / P.V composites, the packed two-tile CutMax, and the complex feedback tile.
One live session per process: run this module on its own."""
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
penv.export_env(DEVICE, chain=CHAIN, CKKS_COMPLEX="1")

from perseus.impl import attention, config, layout  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402
from examples.gpt2_from_primitives import head, weights  # noqa: E402
from examples.gpt2_from_primitives.model import Gpt2Primitives  # noqa: E402

CFG = ROOT / ("configs/model/approximation/"
              + ("gpt2_base_n32" if CHAIN == "n32" else "gpt2_base")
              + "/configs.json")
WEIGHTS = os.environ.get("WEIGHTS_PATH",
                         str(ROOT / ".cache/models/openai-community/gpt2/classic/weights.bin.zip"))


@pytest.fixture(scope="module")
def S():
    from perseus import _core
    sess = penv.open_session(DEVICE, complex_payload=True, chain=CHAIN)
    inf = sess.inf
    assert inf.fhe.complex_payload and not inf.complex      # Mode-A: cachemir + complex slots
    rt = Rt.from_inf(inf, _core, int(sess.options.ckks.composite_degree))
    yield dict(sess=sess, inf=inf, core=_core, rt=rt, parsed=_core.load_configs(str(CFG)),
               cfgs=config.load_configs(str(CFG)), store=weights.RawStore(WEIGHTS))
    sess.close()


def _slots(S, ct):
    return S["rt"].ops.decrypt_slots(ct)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def test_kv_pair_push_and_attention_vs_core(S):
    inf, core, rt = S["inf"], S["core"], S["rt"]
    d = rt.dims
    name = "transformer.h.0.attn"
    inf.set_softmax_cfg("attn", S["parsed"].softmax[name])
    cfg = S["cfgs"].softmax[name]
    inf.block_prefix = core.block_scope(0)
    core.prepare_mha_masks(inf); core.prepare_vcache(inf)
    kv = attention.KVCache(d.d_head)
    rng = np.random.default_rng(4)
    zero = None
    n0 = rt.ops.n_bootstraps
    for tok in range(3):
        def tokvec():
            v = rng.standard_normal(1024) * 0.5
            v[(np.arange(1024) % d.H) >= d.H_real] = 0.0
            return v
        k, v, qv = tokvec(), tokvec(), tokvec()
        if zero is None:
            fresh = rt.ops.encode_token(np.zeros(768))
            zero = rt.ops.sub(fresh, fresh)
        kct = rt.ops.add_pt(zero, layout.lane_vec(k, d.N, d.t))
        vct = rt.ops.add_pt(zero, layout.lane_vec(v, d.N, d.t))
        q = rt.ops.add_pt(zero, layout.lane_vec(qv, d.N, d.t))
        core.cache_kv_push(inf, kct, vct)                 # Mode-A: the pair bootstrap
        attention.cache_kv_push_pair(rt, kv, kct, vct)
    assert rt.ops.n_bootstraps - n0 == 3                  # one deliberate refresh per token
    s_ref = core.qkt(inf, q)
    s = attention.qkt(rt, kv, q)
    assert _rel(_slots(S, s), _slots(S, s_ref[0])) < 1e-2
    p_ref = core.attention_softmax_thor(inf, s_ref, "attn")
    p = attention.softmax_thor(rt, s, cfg, kv.k_count)
    assert _rel(_slots(S, p), _slots(S, p_ref[0])) < 2e-2
    o_ref = core.softmax_v(inf, p_ref)
    o = attention.softmax_v(rt, kv, p)
    assert _rel(_slots(S, o), _slots(S, o_ref)) < 2e-2


def test_cutmax_packed_vs_core(S):
    inf, core, rt = S["inf"], S["core"], S["rt"]
    model = Gpt2Primitives(inf, S["store"], S["cfgs"], core=core, n_layers=1)
    assert model.complex
    os.environ["STEPS_T"] = "128"
    gt = core.read_lm_head_steps(core.RunConfig.from_env())
    assert gt.T > 0, "no per-step logits oracle"
    Wlm = weights.lm_head_matrix(S["store"], S["cfgs"].ln_f, rt.dims, model.vocab)
    target = np.asarray(gt.logits[0])
    hvec = np.linalg.lstsq(Wlm[:768].T, target, rcond=None)[0]
    h = rt.ops.encode_token(hvec)
    tiles = head.lm_head(rt, h, model.lm_tiles())
    lg = model.logits(tiles)
    assert int(np.argmax(lg)) == int(np.argmax(target))
    cmc = core.cutmax_config_from_calib(S["parsed"].cutmax)
    n0 = inf.fhe.total_bootstraps
    z_ref = core.cutmax_argmax(inf, [head.pack_tiles(rt, tiles)], model.vocab, cmc)   # cutmax_packed
    n_ref = inf.fhe.total_bootstraps - n0
    marks = {}
    tap = lambda name, ct: marks.__setitem__(name, inf.fhe.total_bootstraps)
    n1 = inf.fhe.total_bootstraps
    z = [head.cutmax_argmax_packed(rt, head.pack_tiles(rt, tiles), model.vocab, S["cfgs"].cutmax,
                                   model.ones, tap=tap)]
    n_port = inf.fhe.total_bootstraps - n1
    phases = ["entry.B"] + [f"i{i}.end" for i in range(len(S["cfgs"].cutmax.iters))] + ["sum.Z"]
    prev, port_marks = n1, []
    for ph in phases:
        port_marks.append((ph, marks[ph] - prev)); prev = marks[ph]
    print("[port_cutmax_bts]", port_marks, "total", n_port, flush=True)
    assert len(z_ref) == 1 and len(z) == 1
    zr, zp = model.decode_z(z_ref), model.decode_z(z)
    assert int(np.argmax(zr)) == int(np.argmax(zp)) == int(np.argmax(lg))
    assert abs(zr.max() - zp.max()) < 0.05
    # No more refreshes than cutmax_packed. Equality held while the port seeded the cascade
    # from a unit ciphertext like the C++; the closed-form first Newton iteration dropped
    # the port below the reference, and being cheaper than it is not a regression.
    assert n_port <= n_ref, (n_port, n_ref, port_marks)
    # the complex feedback tile on the packed z equals the two real tiles on the unpacked z
    z_re = rt.ops.mult(rt.ops.add(z[0], rt.ops.conjugate(z[0])), 0.5)
    z_im = rt.ops.mult_const(rt.ops.sub(z[0], rt.ops.conjugate(z[0])), 0.0, -0.5)
    real_tiles = weights.feedback_tiles(Wlm, rt.dims, model.vocab, model.W_tile)
    wpe = weights.wpe_row(S["store"], 1, rt.dims.dim)
    print(f"[feedback] z level={rt.ops.level(z[0])} deg={rt.ops.noise_deg(z[0])} "
          f"z_re level={rt.ops.level(z_re)} z_im level={rt.ops.level(z_im)}", flush=True)
    b0 = inf.fhe.total_bootstraps
    a = head.feedback_embed(rt, [z_re, z_im], real_tiles, wpe)
    b1 = inf.fhe.total_bootstraps
    b = head.feedback_embed(rt, z, model.fb_tiles(), wpe)
    print(f"[feedback] bootstraps real-tiles={b1 - b0} packed={inf.fhe.total_bootstraps - b1}", flush=True)
    # the plaintext expectation from the decrypted z: both paths evaluate the same linear map
    # on the same (noisy) z, so each must sit within CKKS noise of it
    want = zp @ Wlm[:768].T + wpe
    ea, eb = _rel(rt.ops.decode_token(a), want), _rel(rt.ops.decode_token(b), want)
    print(f"[feedback] real-tiles rel={ea:.4f} packed rel={eb:.4f} |want|={np.linalg.norm(want):.3f}", flush=True)
    assert eb < 0.05 and ea < 0.05, (ea, eb)


# ── the cachemir_complex packing ─────────────────────────────────────────────────────

def test_outputpack_linear_on_gpu(S):
    inf, core, rt = S["inf"], S["core"], S["rt"]
    d = rt.dims
    rng = np.random.default_rng(21)
    W = rng.standard_normal((d.hid, d.E)) * 0.05; b = rng.standard_normal(d.E) * 0.1
    x = rng.standard_normal(768) * 0.3
    from perseus.impl import linear
    y = linear.linear(rt, rt.ops.encode_token(x), linear.EncodedLinear.encode(W, d.N, d.hid, d.E, b, outputpack=True))
    got = rt.ops.decode_linear_output(y, d.hid, d.E)[:d.E_real]
    want = (x @ W[:768] + b)[:d.E_real]
    assert _rel(got, want) < 1e-2


def test_outputpack_down_projection_on_gpu(S):
    """The down-projection (E -> hid, four output blocks) output-packed equals the dense
    linear on the same input, at the dense linear's level."""
    rt = S["rt"]
    d = rt.dims
    rng = np.random.default_rng(22)
    W = rng.standard_normal((d.E, d.hid)) * 0.02; b = rng.standard_normal(d.hid) * 0.05
    from perseus.impl import layout, linear
    fresh = rt.ops.encode_token(np.zeros(768)); zero = rt.ops.sub(fresh, fresh)
    xv = layout.encode_linear_input(rng.standard_normal(d.E) * 0.3, d.N, d.E, d.hid)
    ref = linear.linear(rt, rt.ops.add_pt(zero, xv), linear.EncodedLinear.encode(W, d.N, d.E, d.hid, b))
    y = linear.linear(rt, rt.ops.add_pt(zero, xv),
                      linear.EncodedLinear.encode(W, d.N, d.E, d.hid, b, outputpack=True))
    assert _rel(_slots(S, y).real, _slots(S, ref).real) < 1e-3
    assert y.level == ref.level


def test_complex_kv_pending_group_vs_core(S):
    """cachemir_complex on the C++ side (inf.complex), three tokens: the packed push into
    the pending real group, then q.K^T and P.V through the complex composites."""
    inf, core, rt = S["inf"], S["core"], S["rt"]
    d = rt.dims
    name = "transformer.h.0.attn"
    inf.set_softmax_cfg("attn", S["parsed"].softmax[name])
    cfg = S["cfgs"].softmax[name]
    inf.block_prefix = core.block_scope(0)
    inf.complex = True
    try:
        core.prepare_mha_masks(inf); core.prepare_vcache(inf)
        kv = attention.ComplexKVCache(d.d_head)
        rng = np.random.default_rng(22)
        fresh = rt.ops.encode_token(np.zeros(768)); zero = rt.ops.sub(fresh, fresh)
        for tok in range(3):
            k, v, qv = (rng.standard_normal(1024) * 0.5 for _ in range(3))
            for arr in (k, v, qv):
                arr[(np.arange(1024) % d.H) >= d.H_real] = 0.0
            P = rt.ops.add_pt(zero, layout.lane_vec(k, d.N, d.t) + 1j * layout.lane_vec(v, d.N, d.t))
            q = rt.ops.add_pt(zero, layout.lane_vec(qv, d.N, d.t))
            core.cache_kv_push_packed(inf, P)
            attention.cache_kv_push_packed_complex(rt, kv, P)
        s_ref = core.qkt(inf, q)
        s = attention.complex_qkt(rt, kv, q)
        assert _rel(_slots(S, s), _slots(S, s_ref[0])) < 1e-2
        p_ref = core.attention_softmax_thor(inf, s_ref, "attn")
        p = attention.softmax_thor(rt, s, cfg, kv.k_count)
        assert _rel(_slots(S, p), _slots(S, p_ref[0])) < 2e-2
        o_ref = core.softmax_v(inf, p_ref)
        o = attention.complex_softmax_v(rt, kv, p)
        assert _rel(_slots(S, o), _slots(S, o_ref)) < 2e-2
    finally:
        inf.complex = False


def test_complex_kv_buckets_vs_real_path(S):
    """Two complete K buckets plus a pending group (kc = 2t + 1) on the GPU: the complex
    caches equal the real caches on the same K/V/Q (the C++ complex composite refuses this
    mix of bucket and pending levels on synthetic inputs, so the reference is the real path)."""
    inf, core, rt = S["inf"], S["core"], S["rt"]
    d = rt.dims
    cfg = S["cfgs"].softmax["transformer.h.0.attn"]
    kvc, kvr = attention.ComplexKVCache(d.d_head), attention.KVCache(d.d_head)
    rng = np.random.default_rng(23)
    fresh = rt.ops.encode_token(np.zeros(768)); zero = rt.ops.sub(fresh, fresh)
    kc = 2 * d.t + 1
    for tok in range(kc):
        k, v = (rng.standard_normal(1024) * 0.5 for _ in range(2))
        for arr in (k, v):
            arr[(np.arange(1024) % d.H) >= d.H_real] = 0.0
        kvec, vvec = layout.lane_vec(k, d.N, d.t), layout.lane_vec(v, d.N, d.t)
        attention.cache_kv_push_packed_complex(rt, kvc, rt.ops.add_pt(zero, kvec + 1j * vvec))
        attention.cache_k_push(rt, kvr, rt.ops.add_pt(zero, kvec))
        attention.cache_v_push(rt, kvr, rt.ops.add_pt(zero, vvec))
    assert len(kvc.k_buckets) == 1 and len(kvc.k_pend) == 1
    qv = rng.standard_normal(1024) * 0.5; qv[(np.arange(1024) % d.H) >= d.H_real] = 0.0
    q = rt.ops.add_pt(zero, layout.lane_vec(qv, d.N, d.t))
    sc, sr = attention.complex_qkt(rt, kvc, q), attention.qkt(rt, kvr, q)
    assert _rel(_slots(S, sc), _slots(S, sr)) < 1e-2
    pc, pr = attention.softmax_thor(rt, sc, cfg, kc), attention.softmax_thor(rt, sr, cfg, kc)
    oc, orr = attention.complex_softmax_v(rt, kvc, pc), attention.softmax_v(rt, kvr, pr)
    assert _rel(_slots(S, oc), _slots(S, orr)) < 2e-2


def test_paired_lm_head_and_cutmax(S):
    inf, core, rt = S["inf"], S["core"], S["rt"]
    model = Gpt2Primitives(inf, S["store"], S["cfgs"], core=core, n_layers=1, packing="cachemir_complex")
    os.environ["STEPS_T"] = "128"
    gt = core.read_lm_head_steps(core.RunConfig.from_env())
    Wlm = weights.lm_head_matrix(S["store"], S["cfgs"].ln_f, rt.dims, model.vocab)
    target = np.asarray(gt.logits[0])
    hvec = np.linalg.lstsq(Wlm[:768].T, target, rcond=None)[0]
    tiles = head.lm_head(rt, rt.ops.encode_token(hvec), model.lm_tiles())
    assert len(tiles) == 1
    lg = model.logits(tiles)
    assert _rel(lg, hvec @ Wlm[:768]) < 1e-2 and int(np.argmax(lg)) == int(np.argmax(target))
    z = model.argmax_encrypted(tiles, 0)
    zdec = model.decode_z(z)
    assert int(np.argmax(zdec)) == int(np.argmax(lg))


def test_kv_offload_matches_resident(S):
    """The K/V residency (caches parked in the pinned arena between blocks) leaves the
    logits unchanged: two blocks, three tokens, offload on vs off."""
    inf, core, rt = S["inf"], S["core"], S["rt"]
    os.environ["STEPS_T"] = "128"
    cfg = core.RunConfig.from_env()
    inputs = core.read_teacher_forced_inputs(cfg)[:3]
    out = {}
    for offload in (False, True):
        model = Gpt2Primitives(inf, S["store"], S["cfgs"], core=core, n_layers=2, kv_offload=offload)
        assert model.kv_offload == offload
        res = model.run_decode(inputs)
        out[offload] = [r["logits"] for r in res]
        model.close()
    for a, b in zip(out[False], out[True]):
        assert _rel(b, a) < 2e-2
