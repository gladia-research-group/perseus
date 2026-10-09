"""perseus.impl + examples/gpt2_from_primitives on the CPU: the cachemir layout against x @ W, every mask
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

from perseus.impl import (attention, config, fake, layout, linear, norm, poly)  # noqa: E402
from perseus.impl import activation as gelu  # noqa: E402
from perseus.impl.rt import Rt  # noqa: E402
from perseus.impl.ops import FheOps  # noqa: E402
from examples.gpt2_from_primitives import block, head, ref, weights  # noqa: E402

CFG = ROOT / "configs/model/approximation/gpt2_base_n32/configs.json"


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


def test_cutmax_packed_matches_real_path():
    """The packed two-tile CutMax (cutmax.cu) equals the real path on the same
    logits, at the production geometry (the schedule needs vocab 50257 = 2 tiles of 32768)."""
    rt, fhe, inf = _rt(N=32768, hid=1024, dim=768, H=16, H_real=12, E=4096, E_real=3072,
                       complex_payload=True)
    cfg = _cfgs().cutmax
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
