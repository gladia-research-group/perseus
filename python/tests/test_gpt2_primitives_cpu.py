"""perseus.impl + perseus/examples/gpt2 on the CPU: the cachemir layout against x @ W, every mask
against the C++ formulas, the polynomial kernels against numpy, and each op chain on the
numpy fake session against its plaintext mirror (ref.py) — so the GPU tier only adds noise."""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perseus._paths import PKG  # noqa: E402

from perseus.impl import (attention, config, fake, layout, linear, norm, poly)  # noqa: E402
from perseus.impl import activation as gelu  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from perseus.examples.gpt2 import block, head, ref, weights  # noqa: E402

CFG = PKG / "configs/model/approximation/gpt2_base_n32/configs.json"


def _rt(strict=False, **kw):
    fhe, inf = fake.make_fake(strict=strict, **kw)
    return Rt(FheOps(inf, fake.core, fhe.unit), layout.Dims.from_inf(inf)), fhe, inf


# ── layout ───────────────────────────────────────────────────────────────────────────

def test_cm_params_full_size():
    N = 32768
    p = layout.cm_params(N, 1024, 1024)
    assert (p.t, p.tp, p.n_pt, p.r_i, p.r_o, p.bstep_c, p.gstep_c) == (32, 32, 32, 32, 1, 8, 4)
    p = layout.cm_params(N, 1024, 4096)
    assert (p.tp, p.tp_in, p.tp_out, p.n_pt, p.r_o) == (8, 32, 8, 128, 4)
    p = layout.cm_params(N, 4096, 1024)
    assert (not p.is_up) and (p.tp_in, p.tp_out, p.n_pt, p.r_o) == (8, 32, 128, 4)
    for s in ((1024, 32768), (32768, 1024)):
        p = layout.cm_params(N, *s)
        assert (p.tp, p.n_pt, p.r_o, p.bstep_c, p.gstep_c) == (1, 1024, 32, 8, 4)


@pytest.mark.parametrize("N,d_in,d_out", [(1024, 32, 32), (1024, 32, 128), (1024, 128, 32),
                                          (1024, 32, 1024), (1024, 1024, 32), (4096, 64, 256)])
def test_linear_matches_matmul(N, d_in, d_out):
    rng = np.random.default_rng(N + d_in)
    fhe, inf = fake.make_fake(N=N, hid=d_in if d_in <= d_out else d_out)
    fhe._rot = set(range(-N, N)); fhe.slots = None      # any step on this shape probe
    rt = Rt(FheOps(inf, fake.core, 2), layout.Dims.from_inf(inf))
    x = rng.standard_normal(d_in); W = rng.standard_normal((d_in, d_out)); b = rng.standard_normal(d_out)
    ct = fake.FakeCt(layout.encode_linear_input(x, N, d_in, d_out), 34)
    w = linear.EncodedLinear.encode(W, N, d_in, d_out, b)
    y = linear.linear(rt, ct, w)
    got = layout.decode_linear_output(y.vec, N, d_in, d_out)
    np.testing.assert_allclose(got, x @ W + b, atol=1e-9)


def test_masks_match_cpp_loops():
    d = layout.Dims.gpt2()
    N, t, tH = d.N, d.t, d.tH
    m = layout.real_head_tok0_mask(d)
    idx = np.flatnonzero(m)
    assert len(idx) == d.d_head_real * d.H_real and idx[1] == t and idx[d.H_real] == tH
    assert layout.real_head_half_mask(d)[0] == 0.5
    sm = layout.score_mask(d, -3.0, 1.0, kc=33)
    assert sm[0] == -1.0 and sm[32] == -1.0 and sm[tH] == -1.0 and sm[tH + 1] == -4.0  # tok 32 -> group 1
    assert layout.score_mask(d, -3.0, 1.0, kc=3)[3] == -4.0
    am = layout.active_mask(d, 3)
    assert am[0] == pytest.approx(0.5 / 3) and am[3] == 0 and am[t] == pytest.approx(0.5 / 3)
    gm = layout.qkt_group_mask(d, 2, 0)
    assert gm[0] == pytest.approx(0.5 / 8) and gm[2] == 0 and gm[12 * t] == 0 and gm[tH] == 0
    assert layout.hrs_pos0(d)[::t].all() and layout.hrs_pos0(d).sum() == N // t
    vl = layout.vlane_mask(d, 5, 3)
    assert vl[5 * tH + 3] == 0.5 and vl[5 * tH + 11 * t + 3] == 0.5 and vl[5 * tH + 12 * t + 3] == 0
    em = layout.active_expanded_mask(d, 0.5)
    assert em.sum() == pytest.approx(0.5 * d.E_real) and em[0] == 0.5 and em[d.tp_E * 3] == 0.0
    cm = layout.cutmax_tile_mask(N, 1, 50257, N, d.hid)
    assert cm.sum() == 50257 - N


def test_rearrange_inverse_of_head_grouping():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((16, 32)); H = 4
    R = layout.rearrange_qkv_weights(W, H)
    # column r of R is head r%H, lane r//H
    for r in range(32):
        np.testing.assert_array_equal(R[:, r], W[:, (r % H) * 8 + r // H])
    O = layout.rearrange_wo_weights(W.T, H)
    for r in range(32):
        np.testing.assert_array_equal(O[r], W.T[(r % H) * 8 + r // H])


# ── polynomials on numpy ─────────────────────────────────────────────────────────────

def test_chebyshev_and_ps_match_numpy():
    ops = poly.NumpyOps()
    x = np.linspace(-0.9, 0.9, 7)
    from numpy.polynomial import chebyshev as C
    c = [0.3, -0.2, 0.5, 0.1, -0.05, 0.02, 0.0, 0.01]
    np.testing.assert_allclose(poly.eval_chebyshev(ops, x, c, -1, 1), C.chebval(x, c), atol=1e-12)
    a, b = 0.2, 3.0
    y = (2 * x * 0.5 + 1.7) * 0.5  # some points in [a, b]
    xx = a + (b - a) * (np.linspace(0, 1, 9))
    np.testing.assert_allclose(poly.eval_chebyshev(ops, xx, c, a, b),
                               C.chebval((2 * xx - (a + b)) / (b - a), c), atol=1e-12)
    p = [1.0, -2.0, 0.5, 0.25, -0.125, 0.0625]
    np.testing.assert_allclose(poly.eval_polynomial_ps(ops, x, p), np.polyval(p[::-1], x), atol=1e-12)
    np.testing.assert_allclose(poly.eval_polynomial(ops, x, p), np.polyval(p[::-1], x), atol=1e-12)
    for pp in (3, 5, 7, 9, 11, 13, 15, 19):
        np.testing.assert_allclose(poly.pow_odd(ops, x, pp), x ** pp, rtol=1e-10)


def test_iterative_inverses_converge():
    ops = poly.NumpyOps()
    a = np.array([0.5, 1.0, 1.7])
    y = poly.goldschmidt_inv_x0(ops, a, np.full(3, 0.8), 6)
    np.testing.assert_allclose(y, 1 / a, rtol=1e-6)
    x = np.array([0.4, 1.0, 2.5])
    y = poly.inv_sqrt_newton(ops, x, 1 / np.sqrt(x) * 0.9, 6, 1.0)
    np.testing.assert_allclose(y, 1 / np.sqrt(x), rtol=1e-8)
    y = poly.inv_sqrt_newton_safe(ops, x, 1 / np.sqrt(x) * 0.9, 6)
    np.testing.assert_allclose(y, 1 / np.sqrt(x), rtol=1e-8)
    cfg = config.load_configs(str(CFG)).norm["transformer.h.0.ln_1"]
    z = np.linspace(cfg.gs_lo * 1.1, cfg.gs_hi, 5)
    r = poly.eval_remez_31(ops, z, cfg.Ncoeffs, cfg.Dcoeffs, cfg.lin_alpha, cfg.lin_beta, cfg.gs_iters)
    r = poly.inv_sqrt_newton(ops, z, r, cfg.nr_iters, 1 / cfg.inv_out_scale ** 2)
    np.testing.assert_allclose(r, cfg.inv_out_scale / np.sqrt(z), rtol=2e-2)


class _CountingOps(poly.NumpyOps):
    """NumpyOps counting ciphertext x ciphertext products (mult of two arrays, square)."""
    def __init__(self, bsgs):
        self.cheb_bsgs = bsgs; self.ct_products = 0
    def mult(self, a, b):
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            self.ct_products += 1
        return a * b
    def square(self, a):
        self.ct_products += 1
        return a * a


@pytest.mark.parametrize("deg", [4, 5, 7, 8, 12, 15, 16, 18, 23, 27, 31, 32, 40])
def test_chebyshev_bsgs_matches_direct(deg):
    rng = np.random.default_rng(deg)
    c = rng.standard_normal(deg + 1) * (0.8 ** np.arange(deg + 1))   # above the 1e-6 coefficient trim
    x = rng.uniform(-3.0, 5.0, 64)
    a, b = -3.0, 5.0
    from numpy.polynomial import chebyshev as C
    direct, bsgs = _CountingOps(False), _CountingOps(True)
    ref = poly.eval_chebyshev(direct, x, c, a, b)
    got = poly.eval_chebyshev(bsgs, x, c, a, b)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got, C.chebval((2 * x - (a + b)) / (b - a), c), rtol=1e-11, atol=1e-11)
    if deg >= 8:
        assert bsgs.ct_products < direct.ct_products, (deg, bsgs.ct_products, direct.ct_products)
    if deg == 31:
        assert bsgs.ct_products == 11 and direct.ct_products == 30


def test_chebyshev_bsgs_depth_matches_direct():
    """Same multiplicative depth on the fake's level accounting (the plan windows do not move)."""
    rt, fhe, inf = _rt()
    g = next(v for v in _cfgs().gelu.values() if v.thor_p1_cheb)
    for coeffs, a, b in ((g.thor_p1_cheb, g.thor_p1_a, g.thor_p1_b), (g.thor_p2_cheb, g.thor_p2_a, g.thor_p2_b)):
        levels = []
        for flag in (False, True):
            rt.ops.cheb_bsgs = flag
            ct = fake.FakeCt(np.linspace(-1, 1, inf.slots), 34)
            levels.append(poly.eval_chebyshev(rt.ops, ct, coeffs, a, b).level)
        rt.ops.cheb_bsgs = False
        assert levels[0] == levels[1], levels


def test_bts2_is_a_refresh_on_the_fake():
    rt, fhe, inf = _rt()
    ct = fake.FakeCt(np.arange(inf.slots, dtype=float), 44)
    out = poly.bts2(rt.ops, ct)
    assert out.level == 34 and fhe.n_bootstraps == 2
    np.testing.assert_allclose(out.vec, ct.vec)


# ── configs ──────────────────────────────────────────────────────────────────────────

def test_config_name_map():
    c = config.load_configs(str(CFG))
    ln1, ln2, sm, ge = c.block(0)
    assert ln1.method == "remez" and ln1.epsilon == 1e-5 and len(ln1.center_scale_sq) == 1024
    assert sm.log2delta1 == 2 and sm.log2delta2 == 3 and len(sm.cheb_coeffs) == 9
    assert ge.method == "thor_composite" and len(ge.thor_p1_cheb) == 32 and ge.thor_p1_a == -1.0
    assert c.cutmax.iters[0].p == 9 and c.cutmax.entry_scale == 1 / 256
    assert sm.kc_r(0, 1) == sm.sm_kc_r[0] and sm.kc_r(2, 10 ** 6) == sm.sm_kc_r[-1]


# ── op chains on the fake vs the numpy mirror ────────────────────────────────────────

def _cfgs():
    return config.load_configs(str(CFG))


@pytest.mark.parametrize("ln_cheb", ["0", "1"])
def test_norm_fake_matches_ref(monkeypatch, ln_cheb):
    monkeypatch.setenv("LN_CHEB", ln_cheb)
    rt, fhe, inf = _rt()
    cfg = _cfgs().norm["transformer.h.3.ln_1"]
    rng = np.random.default_rng(3)
    x = rng.standard_normal(inf.size.dim) * 2.5     # a residual-stream scale: inside both seeds' bands
    ct = fake.core.encode_token_input(inf, x)
    y = norm.norm(rt, ct, cfg, pos=5)
    got = fake.core.decode_token_output(inf, y)
    want = ref.norm_ref(x, cfg, 5)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)
    # inside the calibrated window the approximation tracks the closed form
    assert np.abs(want - ref.norm_exact(x, cfg)).max() < 0.05 * np.abs(want).max()
    # off-lane slots stay zero
    d = rt.dims
    off = np.ones(d.N, bool); off[np.arange(d.dim) * d.t] = False
    assert np.abs(y.vec[off]).max() < 1e-12


def test_layer_norm_unfolded_and_shift():
    rt, fhe, inf = _rt()
    cfg = _cfgs().ln_f
    rng = np.random.default_rng(4)
    x = rng.standard_normal(inf.size.dim) * 0.3
    g, b = rng.standard_normal(inf.size.dim), rng.standard_normal(inf.size.dim)
    ct = fake.core.encode_token_input(inf, x)
    y = norm.layer_norm(rt, ct, cfg, 0, gamma_desc=g * cfg.descale, beta=b)
    np.testing.assert_allclose(fake.core.decode_token_output(inf, y),
                               ref.layer_norm_ref(x, cfg, 0, g, b), rtol=1e-9, atol=1e-9)
    y2 = norm.layer_norm(rt, ct, cfg, 0, shift=b / g)
    np.testing.assert_allclose(fake.core.decode_token_output(inf, y2),
                               ref.norm_ref(x, cfg, 0) + b / g, rtol=1e-9, atol=1e-9)


def test_gelu_fake_matches_ref():
    rt, fhe, inf = _rt()
    cfg = _cfgs().gelu["transformer.h.0.mlp.act"]
    d = rt.dims
    rng = np.random.default_rng(5)
    x = rng.standard_normal(d.E_real) * 2.0
    # an up-linear output layout: feature j at slot m*tp_E with interleave(m) == j
    m = np.arange(d.E); feat = layout.interleave_idx(m, d.hid, d.E)
    v = np.zeros(d.N)
    ok = feat < d.E_real
    v[m[ok] * d.tp_E] = x[feat[ok]]
    ct = fake.FakeCt(v / cfg.xmax if rt.ops.gelu_fold else v, 34)   # GELU_FOLD: the up-projection carries 1/xmax
    y = gelu.gelu(rt, ct, cfg)
    got = np.zeros(d.E_real); got[feat[ok]] = y.vec[m[ok] * d.tp_E]
    np.testing.assert_allclose(got, ref.gelu_ref(x, cfg), rtol=1e-9, atol=1e-9)
    assert np.abs(ref.gelu_ref(x, cfg) - ref.gelu_exact(x)).max() < 0.1


def test_attention_fake_matches_ref():
    rt, fhe, inf = _rt()
    d = rt.dims
    cfg = _cfgs().softmax["transformer.h.0.attn"]
    rng = np.random.default_rng(6)
    kv = attention.KVCache(d.d_head)
    kc = 3
    Ks, Vs, Qs = [], [], []
    for tok in range(kc):
        k = rng.standard_normal(d.hid); v = rng.standard_normal(d.hid); q = rng.standard_normal(d.hid)
        # padded heads carry nothing
        for arr in (k, v, q):
            r = np.arange(d.hid); arr[(r % d.H) >= d.H_real] = 0.0
        Ks.append(k); Vs.append(v); Qs.append(q)
        kct = fake.FakeCt(layout.lane_vec(k, d.N, d.t), 34)
        vct = fake.FakeCt(layout.lane_vec(v, d.N, d.t), 34)
        attention.cache_k_push(rt, kv, kct)
        attention.cache_v_push(rt, kv, vct)
    assert kv.k_count == kc and len(kv.k_groups) == 1
    q = Qs[-1]
    qct = fake.FakeCt(layout.lane_vec(q, d.N, d.t), 34)
    scores = attention.qkt(rt, kv, qct)
    # score(h, tok) at slot h*t + tok (group 0)
    K = np.stack(Ks)
    got = np.array([[scores.vec[h * d.t + tok] for tok in range(kc)] for h in range(d.H_real)])
    want = np.array([[K[tok, h + d.H * np.arange(d.d_head)] @ q[h + d.H * np.arange(d.d_head)]
                      for tok in range(kc)] for h in range(d.H_real)]) / math.sqrt(d.d_head_real)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)
    probs = attention.softmax_thor(rt, scores, cfg, kc)
    out = attention.softmax_v(rt, kv, probs)
    got = fake.core.decode_token_output(inf, out)[:d.hid] if d.dim >= d.hid else out.vec[np.arange(d.hid) * d.t]
    want = ref.attention_ref(q, K, np.stack(Vs), cfg, d.H_real, d.d_head)
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9)


def test_cutmax_fake_matches_ref_and_picks_argmax():
    """The CutMax schedule is calibrated for GPT-2's vocabulary (its sum band), so the
    probe uses vocab 50257 over 50 tiles of the fake's 1024 slots."""
    rt, fhe, inf = _rt()
    cfg = _cfgs().cutmax
    d = rt.dims
    vocab = 50257
    rng = np.random.default_rng(7)
    logits = rng.standard_normal(vocab) * 3.0
    logits[123] = logits.max() + 5.0
    K = (vocab + d.N - 1) // d.N
    tiles = []
    for k in range(K):
        col0 = k * d.N; wreal = min(d.N, vocab - col0)
        v = np.zeros(d.N)
        m = np.arange(d.N); col = layout.cutmax_tile_col_of_slot(m, d.hid, d.N)
        ok = col < wreal
        v[m[ok]] = logits[col0 + col[ok]]
        tiles.append(fake.FakeCt(v, 34))
    assert getattr(head.make_ones(rt.ops, tiles[0]), "period", None) == 1   # a constant, for sparse routing
    z = head.cutmax_argmax(rt, tiles, vocab, cfg)
    zdec = head.decode_z(rt, z, vocab, d.N)
    want = ref.cutmax_ref(logits, cfg)
    np.testing.assert_allclose(zdec, want, rtol=1e-6, atol=1e-9)
    assert int(np.argmax(zdec)) == 123 and zdec[123] > 0.99


@pytest.mark.parametrize("pair", ["0", "1"])
def test_cutmax_pair_complex_matches_ref(monkeypatch, pair):
    """CUTMAX_PAIR packs two cascade passes as u_a + i u_b on one refresh: on a complex payload the result is the
    unpaired one (ref); on a real-slot payload the switch is a no-op (test above)."""
    monkeypatch.setenv("CUTMAX_PAIR", pair)
    rt, fhe, inf = _rt(complex_payload=True)
    cfg = _cfgs().cutmax
    d = rt.dims
    vocab = 50257
    rng = np.random.default_rng(7)
    logits = rng.standard_normal(vocab) * 3.0
    logits[123] = logits.max() + 5.0
    tiles = []
    for k in range((vocab + d.N - 1) // d.N):
        col0 = k * d.N; wreal = min(d.N, vocab - col0)
        v = np.zeros(d.N)
        m = np.arange(d.N); col = layout.cutmax_tile_col_of_slot(m, d.hid, d.N)
        ok = col < wreal
        v[m[ok]] = logits[col0 + col[ok]]
        tiles.append(fake.FakeCt(v, 34))
    z = head.cutmax_argmax(rt, tiles, vocab, cfg)
    zdec = head.decode_z(rt, z, vocab, d.N)
    np.testing.assert_allclose(zdec, ref.cutmax_ref(logits, cfg), rtol=1e-6, atol=1e-9)
    assert int(np.argmax(zdec)) == 123


@pytest.mark.parametrize("complex_payload", [False, True])
def test_head_reduce_sum_pack_matches_unpacked(monkeypatch, complex_payload):
    """SM_HRS_PACK: the head sum and seed * the sum from one complex ladder equal the two unpacked sums."""
    monkeypatch.setenv("FUSED_SM_DEN", "0")
    rng = np.random.default_rng(3)
    outs = {}
    for pack in ("0", "1"):
        monkeypatch.setenv("SM_HRS_PACK", pack)
        rt, fhe, inf = _rt(complex_payload=complex_payload)
        x = fake.FakeCt(rng.standard_normal(rt.dims.N) if pack == "0" else outs["x"], 30)
        outs["x"] = x.vec.copy()
        s, h = attention.head_reduce_sum(rt, x, s0_expected=2.0, periodic=1, seed=0.37)
        outs[pack] = (np.real(s.vec).copy(), np.real(h.vec).copy())
    np.testing.assert_allclose(outs["1"][0], outs["0"][0], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(outs["1"][1], outs["0"][1], rtol=1e-9, atol=1e-9)


def test_block_fake_runs_and_matches_ref_shape(tmp_path):
    """A whole block on the fake with random small weights: finite output on the lanes and
    the level accounting alive (the eager C++ relies on reactive refreshes inside the
    LayerNorm/softmax chains exactly like the fake does)."""
    rt, fhe, inf = _rt()
    d = rt.dims
    cfgs = _cfgs()
    rng = np.random.default_rng(8)
    N, dp, ep, dr, er = d.N, d.hid, d.E, d.dim, d.E_real
    def W(r, c, rr, cc):
        M = np.zeros((r, c)); M[:rr, :cc] = rng.standard_normal((rr, cc)) * 0.05
        return M
    enc = linear.EncodedLinear.encode
    w = weights.BlockWeights(
        q=enc(layout.rearrange_qkv_weights(W(dp, dp, dr, dr), d.H), N, dp, dp, np.zeros(dp)),
        k=enc(layout.rearrange_qkv_weights(W(dp, dp, dr, dr), d.H), N, dp, dp, np.zeros(dp)),
        v=enc(layout.rearrange_qkv_weights(W(dp, dp, dr, dr), d.H), N, dp, dp, np.zeros(dp)),
        out=enc(W(dp, dp, dr, dr), N, dp, dp, np.zeros(dp)),
        up=enc(W(dp, ep, dr, er), N, dp, ep, np.zeros(ep)),
        down=enc(W(ep, dp, er, dr), N, ep, dp, np.zeros(dp)),
        shift1=np.zeros(dr), shift2=np.zeros(dr))
    kv = attention.KVCache(d.d_head)
    x = fake.core.encode_token_input(inf, rng.standard_normal(dr) * 0.3)
    y = block.transformer_block(rt, x, w, kv, cfgs.block(0), 0)
    out = fake.core.decode_token_output(inf, y)
    assert np.isfinite(out).all() and fhe.n_bootstraps > fhe.n_reactive > 0


# ── the runtime services: ladders and the mask discipline ────────────────────────────

def test_rotate_and_sum_matches_loop():
    rt, fhe, inf = _rt()
    x = fake.FakeCt(np.random.default_rng(0).standard_normal(inf.slots), 34)
    d = rt.dims
    for start, stop in ((1, d.t), (d.t, d.N), (d.tH, d.N), (-1, d.t)):   # the band's ladders
        got = rt.ops.rotate_and_sum(x, start, stop).vec
        ref = x.vec.copy(); gap, sign = abs(start), (1 if start > 0 else -1)
        while gap < stop:
            ref = ref + np.roll(ref, -sign * gap); gap *= 2
        np.testing.assert_allclose(got, ref, atol=1e-9)
        np.testing.assert_allclose(poly.NumpyOps().rotate_and_sum(x.vec, start, stop), ref, atol=1e-9)


def test_mask_cache_stage_adopt_evict():
    rt, fhe, inf = _rt()
    N = inf.slots
    x = fake.FakeCt(np.ones(N), 34)
    build = lambda kc: (lambda: np.arange(N) % kc == 0)
    # step 0: first use = a synchronous encode (a miss), level recorded per site
    rt.declare_step(0, [(("sm.active", 1), build(1))])
    rt.begin_step(0)
    rt.mult_mask(x, ("sm.active", 1), build(1))
    st = inf.enc_cache_stats()
    assert st["misses"] == 1 and rt.site_levels["sm.active"] == {34}
    # step 1 staged at the tail: encoded "on the worker", adopted at begin_step, no miss
    rt.end_step(0, [(("sm.active", 2), build(2))])
    stats = rt.begin_step(1)
    assert stats["adopted"] == 1 and stats["evicted"] == 1        # step 0's mask evicted
    assert "mask.sm.active.1#L34" not in inf.enc_cache
    rt.mult_mask(x, ("sm.active", 2), build(2))
    st = inf.enc_cache_stats()
    assert st["misses"] == 1 and st["hits"] == 1 and rt.mask_misses == 0
    # a per-step mask used at a level nobody staged counts as a miss
    y = fake.FakeCt(np.ones(N), 36)
    rt.mult_mask(y, ("sm.active", 2), build(2))
    assert rt.mask_misses == 1
    # a mask shared by consecutive steps is not evicted
    rt.end_step(1, [(("sm.active", 2), build(2)), (("sm.active", 3), build(3))])
    stats = rt.begin_step(2)
    assert stats["evicted"] == 0 and "mask.sm.active.2#L34" in inf.enc_cache


# ── the complex payload (CKKS_COMPLEX=1, the C++ decode configuration) ─────────────────────

def test_kv_pair_push_matches_real_pushes():
    """One K + iV bootstrap (cache_kv_push_pair) fills the same real K groups and V lanes as
    the two real pushes (cachemir_kv_cache.cu, cache_k_push / cache_v_push)."""
    rtc, fhe, inf = _rt(complex_payload=True)
    rtr, _, _ = _rt()
    d = rtc.dims
    rng = np.random.default_rng(11)
    kvc, kvr = attention.KVCache(d.d_head), attention.KVCache(d.d_head)
    for tok in range(3):
        k, v = rng.standard_normal(d.hid), rng.standard_normal(d.hid)
        for arr in (k, v):
            arr[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
        kc = fake.FakeCt(layout.lane_vec(k, d.N, d.t), 34); vc = fake.FakeCt(layout.lane_vec(v, d.N, d.t), 34)
        attention.cache_kv_push_pair(rtc, kvc, kc, vc)
        attention.cache_k_push(rtr, kvr, fake.FakeCt(kc.vec.copy(), 34))
        attention.cache_v_push(rtr, kvr, fake.FakeCt(vc.vec.copy(), 34))
    assert fhe.n_bootstraps == 3                      # one refresh per token, not two
    for g_c, g_r in zip(kvc.k_groups, kvr.k_groups):
        np.testing.assert_allclose(np.real(g_c.vec), g_r.vec, atol=1e-9)
        assert np.abs(np.imag(g_c.vec)).max() < 1e-9
    for lc, lr in zip(kvc.v_lanes, kvr.v_lanes):
        assert (lc is None) == (lr is None)
        if lc is not None:
            np.testing.assert_allclose(np.real(lc.vec), lr.vec, atol=1e-9)
            assert np.abs(np.imag(lc.vec)).max() < 1e-9


def _logit_tiles(rng, d, vocab, boost=123):
    logits = rng.standard_normal(vocab) * 3.0
    logits[boost] = logits.max() + 5.0
    K = (vocab + d.N - 1) // d.N
    tiles = []
    for k in range(K):
        col0 = k * d.N; wreal = min(d.N, vocab - col0)
        v = np.zeros(d.N)
        m = np.arange(d.N); col = layout.cutmax_tile_col_of_slot(m, d.hid, d.N)
        ok = col < wreal
        v[m[ok]] = logits[col0 + col[ok]]
        tiles.append(fake.FakeCt(v, 34))
    return logits, tiles


@pytest.mark.parametrize("cfgs", ["gpt2_base_n32", "gpt2_base_n32_cm"])
def test_cutmax_packed_matches_real_path(cfgs):
    """The packed two-tile CutMax (cutmax.cu) equals the real path on the same
    logits, at the production geometry (the schedule needs vocab 50257 = 2 tiles of 32768); the record's
    cascade and the scaled normalisers (gpt2_base_n32_cm) alike."""
    rt, fhe, inf = _rt(N=32768, hid=1024, dim=768, H=16, H_real=12, E=4096, E_real=3072,
                       complex_payload=True)
    cfg = config.load_configs(str(PKG / f"configs/model/approximation/{cfgs}/configs.json")).cutmax
    d = rt.dims
    logits, tiles = _logit_tiles(np.random.default_rng(7), d, 50257)
    pair = head.pack_tiles(rt, tiles)
    n0 = fhe.n_bootstraps
    Z = head.cutmax_argmax_packed(rt, pair, 50257, cfg)
    n_packed = fhe.n_bootstraps - n0
    zdec = head.decode_z(rt, [Z], 50257, d.N)
    want = ref.cutmax_ref(logits, cfg)
    np.testing.assert_allclose(zdec, want, rtol=1e-6, atol=1e-9)
    z_real = head.cutmax_argmax(rt, [fake.FakeCt(t.vec.copy(), 34) for t in tiles], 50257, cfg)
    n_real = fhe.n_bootstraps - n0 - n_packed
    np.testing.assert_allclose(head.decode_z(rt, z_real, 50257, d.N), zdec, rtol=1e-6, atol=1e-9)
    assert n_packed < n_real                            # one refresh for both tiles
    assert int(np.argmax(zdec)) == 123 and zdec[123] > 0.99


# the band ratios (top / bottom) gpt2_base_n32_cm's normalisers were fitted on: five iterations, then the sum
CUTMAX_BANDS = (93.7, 3.13e4, 4.66e4, 4.64e4, 1.93e4, 7.11e3)


@pytest.mark.parametrize("cfgs", ["gpt2_base_n32_cm", "gpt2_base_n32_pf_cm"])
def test_cutmax_scaled_steps_on_their_bands(cfgs):
    """The served scaled normalisers: on its band each gives 1 / sqrt(t) to within [1, 1 + tol] of the exact value,
    never below, and past the band top it stays positive up to three times the top (the first step's zero)."""
    import json
    raw = json.load(open(PKG / f"configs/model/approximation/{cfgs}/configs.json"))["cutmax"]
    sets = [(s, r, 5e-3) for s, r in zip(raw["newton"], CUTMAX_BANDS)] + [(raw["sum_newton"], CUTMAX_BANDS[-1], 1e-5)]
    for steps, ratio, tol in sets:
        t = np.geomspace(1.0 / ratio, 1.0, 4001)
        r = poly.inv_sqrt_newton_scaled(poly.NumpyOps(), t, steps) * np.sqrt(t)
        assert r.min() >= 1.0 - 1e-7 and r.max() <= 1.0 + 1.01 * tol
        assert (poly.inv_sqrt_newton_scaled(poly.NumpyOps(), np.linspace(1.0, 2.99, 400), steps) > 0).all()


def test_fit_scaled_round_trips_through_the_config():
    """calibrate.approximations.cutmax.fit_scaled on a few logit rows: a cutmax section parse_cutmax takes, with the
    base's outer schedule, whose CutMax picks each row's top logit."""
    import json
    from perseus.calibrate.approximations.cutmax import fit_scaled
    rng = np.random.default_rng(3)
    rows = rng.standard_normal((8, 50257)) * 3.0
    top = rng.integers(0, 50257, 8)
    rows[np.arange(8), top] = rows.max(axis=1) + 5.0
    base = json.load(open(CFG))["cutmax"]
    cfg = config.parse_cutmax(json.loads(json.dumps(fit_scaled([rows], base))))
    assert [it.p for it in cfg.iters] == base["p"] and all(it.newton for it in cfg.iters) and cfg.sum_newton
    for row, k in zip(rows, top):
        z = ref.cutmax_ref(row, cfg)
        assert int(np.argmax(z)) == k and z[k] > 0.99


@pytest.mark.parametrize("ratio", [10.0, 1e3, 5e4])
def test_scaled_newton_steps_from_below(ratio):
    """poly.scaled_newton_steps: 1 / sqrt(x) from below to the tolerance on [1 / ratio, 1], the first step's zero at
    three times the band top (past the band the result stays positive, never sign-flipped)."""
    steps = poly.scaled_newton_steps(ratio, 5e-3)
    x = np.geomspace(1.0 / ratio, 1.0, 4001)
    r = poly.inv_sqrt_newton_scaled(poly.NumpyOps(), x, steps) * np.sqrt(x)
    assert r.max() <= 1.0 + 1e-12 and r.min() >= 1.0 - 5e-3
    assert steps[0][0] / steps[0][1] == pytest.approx(3.0)


def test_feedback_packed_matches_real_tiles():
    rt, fhe, inf = _rt(complex_payload=True)
    d = rt.dims
    vocab, W_tile = 1500, d.N
    rng = np.random.default_rng(3)
    W_lm = np.zeros((d.hid, vocab)); W_lm[:d.dim] = rng.standard_normal((d.dim, vocab)) * 0.1
    real = weights.feedback_tiles(W_lm, d, vocab, W_tile)
    packed = weights.feedback_tiles(W_lm, d, vocab, W_tile, packed=True)
    _, z = _logit_tiles(rng, d, vocab, boost=1300)   # any two vectors: the map is linear
    wpe = rng.standard_normal(d.dim)
    a = head.feedback_embed(rt, [fake.FakeCt(t.vec.copy(), 34) for t in z], real, wpe)
    b = head.feedback_embed(rt, [head.pack_tiles(rt, z)], packed, wpe)
    np.testing.assert_allclose(fake.core.decode_token_output(inf, b),
                               fake.core.decode_token_output(inf, a), rtol=1e-8, atol=1e-9)


# ── the fused reductions (FUSED_SM_DEN / FUSED_LN_VAR): a ladder finished inside a fold ──

def test_fused_ln_var_matches_dense():
    rt, fhe, inf = _rt()
    cfg = _cfgs().norm["transformer.h.3.ln_1"]
    x = np.random.default_rng(3).standard_normal(inf.size.dim) * 0.3
    dense = norm.norm(rt, fake.core.encode_token_input(inf, x), cfg, pos=5)
    rt.ops.fused_ln_var = True
    fhe.calls.clear()
    fused = norm.norm(rt, fake.core.encode_token_input(inf, x), cfg, pos=5)
    assert fhe.calls.count("fold_bootstrap") == 1 and rt.ops.fold_slots_for(1) == 1
    np.testing.assert_allclose(fake.core.decode_token_output(inf, fused),
                               fake.core.decode_token_output(inf, dense), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(fake.core.decode_token_output(inf, fused), ref.norm_ref(x, cfg, 5),
                               rtol=1e-9, atol=1e-9)


def test_fused_softmax_den_matches_dense():
    rt, fhe, inf = _rt()
    d = rt.dims
    cfg = _cfgs().softmax["transformer.h.0.attn"]
    rng = np.random.default_rng(6)
    def run(fused):
        rt.ops.fused_sm_den = fused
        kv = attention.KVCache(d.d_head)
        Ks, Vs = [], []
        for tok in range(3):
            k, v, q = (rng.standard_normal(d.hid) for _ in range(3))
            for arr in (k, v, q):
                arr[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
            Ks.append(k); Vs.append(v)
            attention.cache_k_push(rt, kv, fake.FakeCt(layout.lane_vec(k, d.N, d.t), 34))
            attention.cache_v_push(rt, kv, fake.FakeCt(layout.lane_vec(v, d.N, d.t), 34))
        scores = attention.qkt(rt, kv, fake.FakeCt(layout.lane_vec(q, d.N, d.t), 34))
        fhe.calls.clear()
        probs = attention.softmax_thor(rt, scores, cfg, kv.k_count)
        return attention.softmax_v(rt, kv, probs), q, np.stack(Ks), np.stack(Vs)
    rng_state = rng.bit_generator.state
    out_d, q, K, V = run(False)
    rng.bit_generator.state = rng_state
    out_f, _, _, _ = run(True)
    # one fold per denominator (init + refinements), pre-laddered to the built slot count
    assert fhe.calls.count("fold_bootstrap") == 1 + cfg.log2delta2
    assert rt.ops.fold_slots_for(d.tH) == 512 and d.tH < 512
    want = ref.attention_ref(q, K, V, cfg, d.H_real, d.d_head)
    got = out_f.vec[np.arange(d.hid) * d.t]
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(out_f.vec, out_d.vec, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("complex_payload", [False, True])
def test_softmax_den_recip_matches_goldschmidt(complex_payload):
    """SM_DEN_RECIP: the reciprocal of the head sum built on its own ciphertext (packed as
    D/2 + iR/2 on a complex payload) and the scores multiplied once per round compute the
    same thing as the per-iteration Goldschmidt products of the default path."""
    cfg = _cfgs().softmax["transformer.h.0.attn"]

    def run(recip):
        rt, fhe, inf = _rt(complex_payload=complex_payload)
        rt.ops.sm_den_recip = recip
        d = rt.dims
        rng = np.random.default_rng(11)
        kv = attention.KVCache(d.d_head)
        for tok in range(5):
            k, v, q = (rng.standard_normal(d.hid) for _ in range(3))
            for arr in (k, v, q):
                arr[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
            attention.cache_k_push(rt, kv, fake.FakeCt(layout.lane_vec(k, d.N, d.t), 34))
            attention.cache_v_push(rt, kv, fake.FakeCt(layout.lane_vec(v, d.N, d.t), 34))
        scores = attention.qkt(rt, kv, fake.FakeCt(layout.lane_vec(q, d.N, d.t), 34))
        return attention.softmax_thor(rt, scores, cfg, kv.k_count).vec

    np.testing.assert_allclose(run(True), run(False), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("complex_packing", [False, True])
@pytest.mark.parametrize("recip", [False, True])
@pytest.mark.parametrize("kc", [1, 5, 32])
def test_softmax_periodic_matches_block0(complex_packing, recip, kc):
    """SM_PERIODIC: scores kept tH-periodic (periodic group / score / active masks, no lane-copy ladder in the head
    sum, P.V as one product with the summed V) give the attention output of the block-0 layout."""
    cfg = _cfgs().softmax["transformer.h.0.attn"]

    def run(periodic):
        rt, fhe, inf = _rt(complex_payload=True)
        rt.ops.sm_den_recip, rt.ops.sm_periodic = recip, periodic
        d = rt.dims
        rng = np.random.default_rng(17)
        kv = attention.ComplexKVCache(d.d_head) if complex_packing else attention.KVCache(d.d_head)
        for tok in range(kc):
            k, v = rng.standard_normal(d.hid) * 0.5, rng.standard_normal(d.hid) * 0.5
            for arr in (k, v):
                arr[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
            kct, vct = layout.lane_vec(k, d.N, d.t), layout.lane_vec(v, d.N, d.t)
            if complex_packing:
                attention.cache_kv_push_packed_complex(rt, kv, fake.FakeCt(kct + 1j * vct, 34))
            else:
                attention.cache_k_push(rt, kv, fake.FakeCt(kct, 34))
                attention.cache_v_push(rt, kv, fake.FakeCt(vct, 34))
        q = rng.standard_normal(d.hid); q[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
        qct = fake.FakeCt(layout.lane_vec(q, d.N, d.t), 34)
        scores = (attention.complex_qkt if complex_packing else attention.qkt)(rt, kv, qct)
        probs = attention.softmax_thor(rt, scores, cfg, kc)
        out = (attention.complex_softmax_v if complex_packing else attention.softmax_v)(rt, kv, probs)
        return scores.vec, probs.vec, out.vec, d.tH

    s0, p0, o0, tH = run(False)
    s1, p1, o1, _ = run(True)
    np.testing.assert_allclose(s1, np.tile(s0[:tH], s0.shape[0] // tH), rtol=1e-12, atol=1e-12)   # block 0, tiled
    np.testing.assert_allclose(p1[:tH], p0[:tH], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.real(o1), np.real(o0), rtol=1e-9, atol=1e-12)
    assert np.abs(np.imag(o1)).max() < 1e-9


@pytest.mark.parametrize("complex_packing", [False, True])
@pytest.mark.parametrize("recip", [False, True])
@pytest.mark.parametrize("kc", [33, 70, 140, 256])
def test_softmax_periodic_general_matches_block0(complex_packing, recip, kc):
    """SM_PERIODIC=2: past one group the scores keep period Gp tH (group g in every block b with b % Gp == g, Gp the
    power of two >= the group count): the head sums ladder over one period and P.V takes Gp products, and the
    attention output is the block layout's."""
    cfg = _cfgs().softmax["transformer.h.0.attn"]

    def run(mode):
        rt, fhe, inf = _rt(complex_payload=True)
        rt.ops.sm_den_recip = recip
        rt.ops.sm_periodic, rt.ops.sm_periodic_general = mode > 0, mode == 2
        d = rt.dims
        rng = np.random.default_rng(23)
        kv = attention.ComplexKVCache(d.d_head) if complex_packing else attention.KVCache(d.d_head)
        for tok in range(kc):
            k, v = rng.standard_normal(d.hid) * 0.5, rng.standard_normal(d.hid) * 0.5
            for arr in (k, v):
                arr[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
            kct, vct = layout.lane_vec(k, d.N, d.t), layout.lane_vec(v, d.N, d.t)
            if complex_packing:
                attention.cache_kv_push_packed_complex(rt, kv, fake.FakeCt(kct + 1j * vct, 34))
            else:
                attention.cache_k_push(rt, kv, fake.FakeCt(kct, 34))
                attention.cache_v_push(rt, kv, fake.FakeCt(vct, 34))
        q = rng.standard_normal(d.hid); q[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
        qct = fake.FakeCt(layout.lane_vec(q, d.N, d.t), 34)
        a = attention.sm_fold_affine(rt, cfg)[0]
        scores = (attention.complex_qkt if complex_packing else attention.qkt)(rt, kv, qct, a)
        probs = attention.softmax_thor(rt, scores, cfg, kc, folded=rt.ops.sm_fold)
        out = (attention.complex_softmax_v if complex_packing else attention.softmax_v)(rt, kv, probs)
        return scores.vec, out.vec, attention.sm_periodic(rt, kc), d.tH

    s0, o0, per0, tH = run(0)
    s2, o2, per, _ = run(2)
    assert per0 == 0 and per == 1 << ((kc + 31) // 32 - 1).bit_length()
    np.testing.assert_allclose(np.real(s2), np.real(np.tile(s0[:per * tH], s0.shape[0] // (per * tH))),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.real(o2), np.real(o0), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("complex_packing", [False, True])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("kc", [1, 5, 32, 70])
def test_softmax_fold_matches_unfolded(complex_packing, periodic, kc):
    """SM_FOLD: the exp's Chebyshev affine carried by the q.K^T group and score masks and each reciprocal round's
    seed slope by a second head-sum mask give the unfolded softmax (scores x probabilities x P.V)."""
    cfg = _cfgs().softmax["transformer.h.0.attn"]

    def run(fold):
        rt, fhe, inf = _rt(complex_payload=True)
        rt.ops.sm_den_recip, rt.ops.sm_periodic, rt.ops.sm_fold = True, periodic, fold
        d = rt.dims
        rng = np.random.default_rng(19)
        kv = attention.ComplexKVCache(d.d_head) if complex_packing else attention.KVCache(d.d_head)
        for tok in range(kc):
            k, v = rng.standard_normal(d.hid) * 0.5, rng.standard_normal(d.hid) * 0.5
            for arr in (k, v):
                arr[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
            kct, vct = layout.lane_vec(k, d.N, d.t), layout.lane_vec(v, d.N, d.t)
            if complex_packing:
                attention.cache_kv_push_packed_complex(rt, kv, fake.FakeCt(kct + 1j * vct, 34))
            else:
                attention.cache_k_push(rt, kv, fake.FakeCt(kct, 34))
                attention.cache_v_push(rt, kv, fake.FakeCt(vct, 34))
        q = rng.standard_normal(d.hid); q[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
        qct = fake.FakeCt(layout.lane_vec(q, d.N, d.t), 34)
        a = attention.sm_fold_affine(rt, cfg)[0]
        scores = (attention.complex_qkt if complex_packing else attention.qkt)(rt, kv, qct, a)
        probs = attention.softmax_thor(rt, scores, cfg, kc, folded=fold)
        out = (attention.complex_softmax_v if complex_packing else attention.softmax_v)(rt, kv, probs)
        return probs.vec, out.vec

    p0, o0 = run(False)
    p1, o1 = run(True)
    np.testing.assert_allclose(np.real(p1), np.real(p0), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.real(o1), np.real(o0), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("kc", [1, 5, 33, 70])
def test_kv_lanes_push_matches_complex_refresh(kc):
    """KV_LANES: K on token lane 0 and V on lane 1 of one real payload fill the same K buckets and V pair buckets as
    the complex K + iV refresh (any position in the group: the V lane rotation is 1 - rr)."""
    def run(kvl):
        rt, fhe, inf = _rt(complex_payload=True)
        rt.ops.kv_lanes = kvl
        d = rt.dims
        rng = np.random.default_rng(23)
        kv = attention.ComplexKVCache(d.d_head)
        for tok in range(kc):
            k, v = rng.standard_normal(d.hid) * 0.5, rng.standard_normal(d.hid) * 0.5
            for arr in (k, v):
                arr[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
            # the fused linear's output: K + iV on every token lane
            full = np.repeat(k, d.t) + 1j * np.repeat(v, d.t)
            attention.cache_kv_push_packed_complex(rt, kv, fake.FakeCt(full, 34))
        return ([b.vec for b in kv.k_buckets] + [b.vec for b in kv.k_pend]
                + [b.vec for b in kv.v_buckets if b is not None])

    for a, b in zip(run(True), run(False)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kc", [1, 3])
def test_qkv_lanes_q_rides_push_refresh(kc):
    """QKV_LANES: the Q linear output on token lane 2 of the K/V push payload leaves the K buckets and V pair buckets
    untouched and comes back on lane 0, where complex_qkt's tok0 mask reads it."""
    from perseus.impl.layout import real_head_tok0_mask
    def run(with_q):
        rt, fhe, inf = _rt(complex_payload=True)
        rt.ops.kv_lanes = True
        d = rt.dims
        rng = np.random.default_rng(29)
        kv = attention.ComplexKVCache(d.d_head)
        qs = []
        for tok in range(kc):
            k, v, q = (rng.standard_normal(d.hid) * 0.5 for _ in range(3))
            for arr in (k, v, q):
                arr[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
            full = np.repeat(k, d.t) + 1j * np.repeat(v, d.t)
            qfull = np.repeat(q, d.t) + 1j * rng.standard_normal(d.N) * 1e-3   # a real output with im noise
            out = attention.cache_kv_push_packed_complex(rt, kv, fake.FakeCt(full, 34),
                                                         fake.FakeCt(qfull, 34) if with_q else None)
            qs.append((q, out))
        return ([b.vec for b in kv.k_buckets] + [b.vec for b in kv.k_pend]
                + [b.vec for b in kv.v_buckets if b is not None]), qs, d
    bufs_q, qs, d = run(True)
    bufs_0, qs0, _ = run(False)
    for a, b in zip(bufs_q, bufs_0):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    assert all(o is None for _, o in qs0)
    m = real_head_tok0_mask(d) > 0
    for q, out in qs:
        assert out is not None
        got = (out.vec * (real_head_tok0_mask(d))).real
        np.testing.assert_allclose(got[m], np.repeat(q, d.t)[m], rtol=1e-12, atol=1e-12)


def test_gelu_fold_matches_gelu():
    """GELU_FOLD: fed x / xmax (the up-projection carries the scaling) and with xmax on the
    mask, the GELU returns what the unfolded form returns on x."""
    rt, fhe, inf = _rt(complex_payload=True)
    d = rt.dims
    cfg = _cfgs().gelu["transformer.h.3.mlp.act"]
    x = np.random.default_rng(12).normal(0.0, cfg.xmax / 4, d.N) * (layout.active_expanded_mask(d, 1.0) != 0)
    rt.ops.gelu_fold = False
    want = gelu.gelu(rt, fake.FakeCt(x.astype(complex), 34), cfg).vec
    rt.ops.gelu_fold = True
    got = gelu.gelu(rt, fake.FakeCt((x / cfg.xmax).astype(complex), 34), cfg).vec
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


# ── the cachemir_complex packing ─────────────────────────────────────────────────────

def test_outputpack_linear_matches_dense():
    rt, fhe, inf = _rt(complex_payload=True)
    d = rt.dims
    rng = np.random.default_rng(12)
    W = rng.standard_normal((d.hid, d.E)) * 0.1; b = rng.standard_normal(d.E) * 0.1
    x = rng.standard_normal(d.dim) * 0.3
    dense = linear.linear(rt, fake.core.encode_token_input(inf, x), linear.EncodedLinear.encode(W, d.N, d.hid, d.E, b))
    packed_w = linear.EncodedLinear.encode(W, d.N, d.hid, d.E, b, outputpack=True)
    assert packed_w.outputpack and packed_w.pts.shape[0] * 2 == dense.vec.shape[0] // d.N * 0 + linear.cm_params(d.N, d.hid, d.E).n_pt
    fhe.calls.clear()
    packed = linear.linear(rt, fake.core.encode_token_input(inf, x), packed_w)
    assert fhe.calls.count("mult_pt") == packed_w.pts.shape[0]     # half the plaintext products
    np.testing.assert_allclose(np.real(packed.vec), np.real(dense.vec), rtol=1e-9, atol=1e-9)
    assert np.abs(np.imag(packed.vec)).max() < 1e-9
    want = fake.core.decode_linear_output(inf, dense, d.hid, d.E)
    np.testing.assert_allclose(want[:d.E_real], (x @ W[:d.dim] + b)[:d.E_real], rtol=1e-9, atol=1e-9)


def test_complex_attention_matches_real_over_two_buckets():
    """Two complete K buckets plus a pending group (kc = 2t + 1): the complex q.K^T and
    P.V equal the real path on the same K/V/Q."""
    rtc, fhec, infc = _rt(complex_payload=True)
    rtr, fher, infr = _rt()
    d = rtc.dims
    cfg = _cfgs().softmax["transformer.h.0.attn"]
    rng = np.random.default_rng(13)
    kvc, kvr = attention.ComplexKVCache(d.d_head), attention.KVCache(d.d_head)
    kc = 2 * d.t + 1
    for tok in range(kc):
        k, v = rng.standard_normal(d.hid) * 0.5, rng.standard_normal(d.hid) * 0.5
        for arr in (k, v):
            arr[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
        kct = layout.lane_vec(k, d.N, d.t); vct = layout.lane_vec(v, d.N, d.t)
        attention.cache_kv_push_packed_complex(rtc, kvc, fake.FakeCt(kct + 1j * vct, 34))   # P = K + iV
        attention.cache_k_push(rtr, kvr, fake.FakeCt(kct, 34))
        attention.cache_v_push(rtr, kvr, fake.FakeCt(vct, 34))
    assert len(kvc.k_buckets) == 1 and len(kvc.k_pend) == 1 and kvc.k_count == kc
    q = rng.standard_normal(d.hid); q[(np.arange(d.hid) % d.H) >= d.H_real] = 0.0
    sc = attention.complex_qkt(rtc, kvc, fake.FakeCt(layout.lane_vec(q, d.N, d.t), 34))
    sr = attention.qkt(rtr, kvr, fake.FakeCt(layout.lane_vec(q, d.N, d.t), 34))
    np.testing.assert_allclose(np.real(sc.vec), sr.vec, rtol=1e-8, atol=1e-9)
    assert np.abs(np.imag(sc.vec)).max() < 1e-9
    pc = attention.softmax_thor(rtc, sc, cfg, kc); pr = attention.softmax_thor(rtr, sr, cfg, kc)
    oc = attention.complex_softmax_v(rtc, kvc, pc); orr = attention.softmax_v(rtr, kvr, pr)
    np.testing.assert_allclose(np.real(oc.vec), orr.vec, rtol=1e-7, atol=1e-9)
    assert np.abs(np.imag(oc.vec)).max() < 1e-8


def test_paired_lm_head_logits_match_real():
    rt, fhe, inf = _rt(complex_payload=True)
    d = rt.dims
    vocab, W_tile = 1500, d.N
    rng = np.random.default_rng(14)
    W_lm = np.zeros((d.hid, vocab)); W_lm[:d.dim] = rng.standard_normal((d.dim, vocab)) * 0.1
    h = rng.standard_normal(d.dim) * 0.3
    real = head.lm_head(rt, fake.core.encode_token_input(inf, h), weights.lm_head_tiles(W_lm, d, vocab, W_tile))
    paired = head.lm_head(rt, fake.core.encode_token_input(inf, h), weights.lm_head_tiles(W_lm, d, vocab, W_tile, paired=True))
    assert len(real) == 2 and len(paired) == 1
    a = head.decode_logits(rt, real, vocab, W_tile); b = head.decode_logits(rt, paired, vocab, W_tile)
    np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(a, h @ W_lm[:d.dim], rtol=1e-9, atol=1e-9)


# ── lane-wise algebra on a packed pair ───────────────────────────────────────────────

def test_lane_square_and_mult_are_lanewise():
    rt, fhe, inf = _rt(complex_payload=True)
    rng = np.random.default_rng(31)
    a, b = rng.standard_normal(inf.slots), rng.standard_normal(inf.slots)
    c, e = rng.standard_normal(inf.slots), rng.standard_normal(inf.slots)
    z1, z2 = fake.FakeCt(a + 1j * b, 34), fake.FakeCt(c + 1j * e, 34)
    s = poly.lane_square(rt.ops, z1)
    np.testing.assert_allclose(s.vec, a * a + 1j * (b * b), rtol=1e-12, atol=1e-12)
    m = poly.lane_mult(rt.ops, z1, z2)
    np.testing.assert_allclose(m.vec, a * c + 1j * (b * e), rtol=1e-12, atol=1e-12)
    assert s.level == 34 + 2 * fhe.unit      # the product and the 1/4: two levels; the (1+-i) are free
    u = poly.lane_square(rt.ops, fake.FakeCt(a + 1j * b, 34), unscaled=True)
    np.testing.assert_allclose(u.vec, 4 * (a * a + 1j * (b * b)), rtol=1e-12, atol=1e-12)
    assert u.level == 34 + fhe.unit          # one level, like a plain square


def test_lane_packed_chebyshev_evaluates_both_lanes():
    rt, fhe, inf = _rt(complex_payload=True)
    cfg = _cfgs().softmax["transformer.h.0.attn"]
    rng = np.random.default_rng(32)
    a = rng.uniform(cfg.cheb_a, cfg.cheb_b, inf.slots); b = rng.uniform(cfg.cheb_a, cfg.cheb_b, inf.slots)
    lane = poly.LanePackedOps(rt.ops)
    z = poly.eval_chebyshev(lane, fake.FakeCt(a + 1j * b, 34), cfg.cheb_coeffs, cfg.cheb_a, cfg.cheb_b)
    ya = poly.eval_chebyshev(rt.ops, fake.FakeCt(a, 34), cfg.cheb_coeffs, cfg.cheb_a, cfg.cheb_b)
    yb = poly.eval_chebyshev(rt.ops, fake.FakeCt(b, 34), cfg.cheb_coeffs, cfg.cheb_a, cfg.cheb_b)
    np.testing.assert_allclose(z.vec, ya.vec + 1j * yb.vec, rtol=1e-9, atol=1e-9)


def test_unit_seed_newton_matches_ones_ciphertext():
    rt, fhe, inf = _rt()
    x = fake.FakeCt(np.random.default_rng(41).uniform(0.5, 2.0, inf.slots), 34)
    ones = head.make_ones(rt.ops, x)
    a = poly.inv_sqrt_newton_safe(rt.ops, x, ones, 5)
    fhe.calls.clear()
    b = poly.inv_sqrt_newton_safe(rt.ops, x, None, 5)
    np.testing.assert_allclose(b.vec, a.vec, rtol=1e-12, atol=1e-12)
    assert b.level == a.level - fhe.unit                     # the folded first iteration: one level fewer
    assert fhe.calls.count("mult_cc") == 3 * 4               # three products per remaining iteration


@pytest.mark.parametrize("complex_packing", [False, True])
def test_replay_kv_pushes(complex_packing):
    """ImplModel.replay_kv_pushes: the payloads one session logs (attention.kv_push) rebuild the same caches,
    vectors and counts, when replayed on another."""
    from perseus.impl.driver import ImplModel
    rt, fhe, inf = _rt(complex_payload=True)
    d = rt.dims
    rng = np.random.default_rng(29)
    Cache = attention.ComplexKVCache if complex_packing else attention.KVCache
    src = [Cache(d.d_head) for _ in range(2)]
    for kv in src:
        kv.payload_log = []
    for _ in range(d.t + 3):
        for kv in src:
            k, v = (layout.lane_vec(rng.standard_normal(d.hid) * 0.5, d.N, d.t) for _ in range(2))
            p = fake.FakeCt(k + 1j * v, 34) if complex_packing else (fake.FakeCt(k, 34), fake.FakeCt(v, 34))
            attention.kv_push(rt, kv, p)
    _fhe2, inf2 = fake.make_fake(complex_payload=True)
    m = ImplModel(inf2, core=fake.core, kv_offload=False)
    m.kv = [Cache(d.d_head) for _ in range(2)]
    assert m.replay_kv_pushes([kv.payload_log for kv in src], level=36) == d.t + 3
    for a, b in zip(src, m.kv):
        assert (a.k_count, a.v_count) == (b.k_count, b.v_count)
        ca, ka = a.cts("p."); cb, kb = b.cts("p.")
        assert ka == kb
        for x, y in zip(ca, cb):
            np.testing.assert_allclose(y.vec, x.vec, atol=1e-9)


def test_ensure_routes_drops_before_it_builds(monkeypatch):
    """ImplModel.ensure_routes: at a length-class change the routes no longer needed go first, then the missing ones
    are built (the device holds the larger set, never both); the session's base routes are never dropped."""
    from perseus.impl.driver import ImplModel
    monkeypatch.delenv("SM_ROUTE_AHEAD", raising=False)
    log = []

    class Inf:
        sparse_routes = [1, 512]

        def add_sparse_route(self, s):
            log.append(("add", s)); self.sparse_routes = sorted(set(self.sparse_routes) | {s})

        def drop_sparse_route(self, s):
            log.append(("drop", s)); self.sparse_routes = sorted(set(self.sparse_routes) - {s}); return 1

    m = ImplModel.__new__(ImplModel)
    m.inf, m._route_class, m._base_routes = Inf(), None, None
    gp = lambda pos: 1 << ((pos // 32).bit_length())                       # 1, 2, 4 ... by 32-token group
    m.routes_for = lambda pos: (gp(pos), {gp(pos) * 512} if gp(pos) > 1 else set())
    for pos in range(0, 130):
        m.ensure_routes(pos)
    assert log == [("add", 1024), ("drop", 1024), ("add", 2048), ("drop", 2048), ("add", 4096)]
    assert m.inf.sparse_routes == [1, 512, 4096]
