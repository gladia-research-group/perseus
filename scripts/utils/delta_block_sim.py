#!/usr/bin/env python3
"""Slot-exact numpy validation of the delta-block prefill attention layout (task #10,
HANDOFF_delta_block_layout.txt).

Pipeline A = the shipping per-entry flow (cachemir_filling_attention.cu, real arm):
one score ct per union-schedule entry (g, delta), THOR softmax chain PER ENTRY.
Pipeline B = delta-block: score[h,i,Delta] at slot (Delta%64)*tH + h*t + i of ct
Delta//64; the softmax chain runs ONCE per big ct; row-sums are the log2(N/tH)=6
stride-tH rotations; entry extraction for P.V is folded into the FINAL refine's
zscale mask (same op count / level as today's per-entry zscale).

Both pipelines emulate CKKS at slot level (cyclic N-vector, plaintext masks, the
exact goldschmidt_recip recurrence, chebval == eval_chebyshev_series) in VALUE form
(im_cleanse 2x / paired 0.5 factors cancelled analytically: qkt mask 1/sqrt(dh),
amask 1/kc, zscale sqrt(kc)/2). Asserts:
  A == B exactly; A == dense slotless THOR reference; entry->Delta coverage is a
  disjoint partition; B's denominator is tH-periodic (=> sparse-bts routable).
Reports op counts, bootstrap-input magnitudes, GS-basin stats, and THOR-vs-exact
softmax context error. Optional per-bootstrap noise injection compares robustness.
"""

import json
import os
import sys
import numpy as np
from numpy.polynomial import chebyshev

N = 32768
HID = 1024
T = N // HID          # 32 token lanes
H = 16                # padded heads
H_REAL = 12
DH_REAL = 64
tH = T * H            # 512
NBLK = N // tH        # 64 tH-blocks per ct
LOG2_NBLK = 6

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(REPO, "..", "..", "configs", "model", "approximation",
                        "gpt2_gatem15_cm_sparse", "configs.json")


def load_cfg():
    with open(CFG_PATH) as f:
        return json.load(f)["softmax"]["transformer.h.0.attn"]


class Ops:
    """CKKS slot emulator + op counters. rot(v, r): out[s] = v[(s+r) % N]."""

    def __init__(self, noise=0.0, seed=0):
        self.c = {"rot": 0, "bts": 0, "ctmult": 0, "ptmult": 0, "add": 0}
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.bts_max_abs = []

    def rot(self, v, r):
        self.c["rot"] += 1
        return np.roll(v, -r)

    def ctmult(self, a, b):
        self.c["ctmult"] += 1
        return a * b

    def ptmult(self, a, pt):
        self.c["ptmult"] += 1
        return a * pt

    def add(self, a, b):
        self.c["add"] += 1
        return a + b

    def bts(self, v, half_scaled=True):
        # runtime feeds 0.5-scaled inputs at the recip sites; record the value-form
        # magnitude the EvalMod actually sees.
        self.c["bts"] += 1
        self.bts_max_abs.append(np.max(np.abs(v)) * (0.5 if half_scaled else 1.0))
        if self.noise:
            return v + self.noise * self.rng.uniform(-1.0, 1.0, size=v.shape)
        return v

    def allreduce_stride_tH(self, v, sign=+1):
        # qkt reduce / softmax_v broadcast: every slot ends with the sum of its
        # stride-tH class (cyclic, direction-independent).
        for s in [tH << k for k in range(LOG2_NBLK)]:
            v = self.add(v, self.rot(v, sign * s))
        return v


def goldschmidt_recip(ops, D, F_init, iters):
    R = F_init
    D_neg = ops.ctmult(D, -F_init)
    F = D_neg + 2.0
    for i in range(1, iters):
        R = ops.ctmult(R, F)
        if i + 1 < iters:
            D_neg = ops.ctmult(D_neg, F)
            F = D_neg + 2.0
    return R


def fresh_recip(ops, D, F_init, iters):
    R = goldschmidt_recip(ops, D, F_init, iters)
    return ops.bts(R)          # value form: the 0.5 mult + post-bts cleanse cancel


def cheb_exp(ops, x, cfg, sf):
    a, b = cfg["cheb_a"] / sf, cfg["cheb_b"] / sf
    y = (2.0 * x - (a + b)) / (b - a)
    ops.c["ptmult"] += 1
    ops.c["ctmult"] += 7       # T2..T8 tower (deg-8 series), as in polynomial.cu
    return chebyshev.chebval(y, np.asarray(cfg["cheb_coeffs"]))


def sm_kc_r(cfg, step, kc):
    r = cfg["sm_kc_r"]
    C = len(r) // cfg["refinement_iters"]
    return r[step * C + max(0, min(kc - 1, C - 1))]


# ---------------------------------------------------------------- geometry

def schedule(P, n_cur):
    """cf_score_schedule: f-groups full delta range, current chunk delta in [0, T)."""
    G = P // T
    sched = [(g, d, False) for g in range(G) for d in range(-(T - 1), T)]
    sched += [(G, d, True) for d in range(T)]
    return sched, G


def entry_active(g, d, cur, n_cur, P):
    """(h, i) activity of one entry, as a (H, T) bool grid (== is_active)."""
    act = np.zeros((H, T), dtype=bool)
    for h in range(H_REAL):
        for i in range(n_cur):
            l = i - d
            if 0 <= l < (n_cur if cur else T):
                act[h, i] = True
    return act


def delta_abs(g, d, G):
    return (G - g) * T + d


def grid(fn):
    """Build an N-vector from fn(blk, h, i)."""
    v = np.zeros(N)
    m = v.reshape(NBLK, H, T)
    for b in range(NBLK):
        for h in range(H):
            for i in range(T):
                m[b, h, i] = fn(b, h, i)
    return v


# ---------------------------------------------------------------- pipeline A

def pipeline_A(ops, q_ct, k_group, v_group, sched, G, P, n_cur, cfg, exact_recip):
    sf = 2.0 ** (-cfg["n_squarings"] - cfg["refinement_iters"])
    mean = (cfg["clip_hi"] + cfg["clip_lo"]) / 2.0
    kc_i = np.array([P + i + 1 for i in range(T)])

    valid_row = grid(lambda b, h, i: b == 0 and h < H_REAL and i < n_cur)
    floor = 1.0 - valid_row                                  # cf.sm.floor: 1 on invalid

    # --- qkt: per-entry matmul, reduce, mask (value form: scale = 1/sqrt(dh))
    q = ops.bts(q_ct)                     # q entry bts (runtime feeds 0.5q via 0.25+cleanse)
    scores = []
    for (g, d, cur) in sched:
        krot = ops.rot(k_group[g], -d)
        res = ops.ctmult(q, krot)
        res = ops.allreduce_stride_tH(res, +1)
        act = entry_active(g, d, cur, n_cur, P)
        m = np.zeros(N)
        m.reshape(NBLK, H, T)[0] = act / np.sqrt(DH_REAL)    # cf.qkt.m -> block 0
        scores.append(ops.ptmult(res, m))

    # --- softmax (per entry)
    e = []
    for idx, (g, d, cur) in enumerate(sched):
        act = entry_active(g, d, cur, n_cur, P)
        shift = np.full(N, cfg["clip_lo"] - mean)
        shift.reshape(NBLK, H, T)[0][act] = -mean            # cf.sm.shift
        z = ops.add(scores[idx], shift)
        z = cheb_exp(ops, z, cfg, sf)
        z = ops.bts(z, half_scaled=False)                    # exp_refresh
        for _ in range(cfg["n_squarings"]):
            z = ops.ctmult(z, z)
        am = np.zeros(N)                                     # cf.sm.amask: 1/kc
        am.reshape(NBLK, H, T)[0][act] = (1.0 / kc_i)[None, :].repeat(H, 0)[act]
        e.append(ops.ptmult(z, am))

    sden = e[0].copy()
    for x in e[1:]:
        sden = ops.add(sden, x)
    sden = sden + floor
    if exact_recip:
        recip = 1.0 / sden
    else:
        F0 = cfg["init_alpha"] - cfg["init_beta"] * sden
        recip = fresh_recip(ops, sden, F0, cfg["gs_iters_scaled"] + 1)
    y = [ops.ctmult(ei, recip) for ei in e]

    zsc_row = np.sqrt(kc_i) / 2.0                            # value-form zscale
    for r in range(cfg["refinement_iters"]):
        z2 = []
        for idx, (g, d, cur) in enumerate(sched):
            act = entry_active(g, d, cur, n_cur, P)
            zs = np.zeros(N)
            zs.reshape(NBLK, H, T)[0][act] = zsc_row[None, :].repeat(H, 0)[act]
            z2.append(ops.ptmult(ops.ctmult(y[idx], y[idx]), zs))
        s2 = z2[0].copy()
        for x in z2[1:]:
            s2 = ops.add(s2, x)
        s2 = s2 + floor
        if exact_recip:
            rrec = 1.0 / s2
        else:
            kr = np.array([sm_kc_r(cfg, r, kc) for kc in kc_i])
            al = grid(lambda b, h, i: cfg["refine_alpha"][r] *
                      (np.sqrt(kr[i]) if (b == 0 and h < H_REAL and i < n_cur) else 1.0))
            be = grid(lambda b, h, i: -cfg["refine_beta"][r] *
                      (kr[i] if (b == 0 and h < H_REAL and i < n_cur) else 1.0))
            Fr = ops.ptmult(s2, be) + al
            rrec = fresh_recip(ops, s2, Fr, int(cfg["per_step_refine_iters"][r]) + 1)
        y = [ops.ctmult(z2i, rrec) for z2i in z2]

    # --- softmax_v: per-entry broadcast + rotated V
    out = np.zeros(N)
    for idx, (g, d, cur) in enumerate(sched):
        p = ops.allreduce_stride_tH(y[idx], -1)
        vrot = ops.rot(v_group[g], -d)
        out = ops.add(out, ops.ctmult(p, vrot))
    return out


# ---------------------------------------------------------------- pipeline B

def pipeline_B(ops, q_ct, k_group, v_group, sched, G, P, n_cur, cfg, exact_recip):
    sf = 2.0 ** (-cfg["n_squarings"] - cfg["refinement_iters"])
    mean = (cfg["clip_hi"] + cfg["clip_lo"]) / 2.0
    kc_i = np.array([P + i + 1 for i in range(T)])
    n_delta = P + n_cur                                      # Delta in [0, P+n_cur-1]
    n_cts = (n_delta + NBLK - 1) // NBLK

    # delta-block activity: slot (b,h,i) of ct ci is live iff Delta=ci*64+b <= P+i
    def blk_active(ci):
        return grid(lambda b, h, i: h < H_REAL and i < n_cur and ci * NBLK + b <= P + i)

    act_ct = [blk_active(ci) for ci in range(n_cts)]
    valid_row = grid(lambda b, h, i: h < H_REAL and i < n_cur)   # tH-periodic
    floor = 1.0 - valid_row

    # coverage assert: every active (Delta,h,i) is written by EXACTLY one entry
    cover = [np.zeros(N) for _ in range(n_cts)]
    for (g, d, cur) in sched:
        D = delta_abs(g, d, G)
        ci, b = D // NBLK, D % NBLK
        cover[ci].reshape(NBLK, H, T)[b] += entry_active(g, d, cur, n_cur, P)
    for ci in range(n_cts):
        assert np.array_equal(cover[ci], act_ct[ci]), "entry->Delta cover not a partition"

    # --- qkt: identical per-entry matmul; the ONLY change is the mask target block
    q = ops.bts(q_ct)                     # q entry bts (runtime feeds 0.5q via 0.25+cleanse)
    S = [np.zeros(N) for _ in range(n_cts)]
    for (g, d, cur) in sched:
        krot = ops.rot(k_group[g], -d)
        res = ops.ctmult(q, krot)
        res = ops.allreduce_stride_tH(res, +1)
        D = delta_abs(g, d, G)
        ci, b = D // NBLK, D % NBLK
        m = np.zeros(N)
        m.reshape(NBLK, H, T)[b] = entry_active(g, d, cur, n_cur, P) / np.sqrt(DH_REAL)
        S[ci] = ops.add(S[ci], ops.ptmult(res, m))           # scatter = mask retarget + add

    # --- softmax: ONCE per big ct
    shift = [np.where(act_ct[ci] > 0, -mean, cfg["clip_lo"] - mean) for ci in range(n_cts)]
    amask = [act_ct[ci] * np.tile((1.0 / kc_i), NBLK * H) for ci in range(n_cts)]
    zsc = [act_ct[ci] * np.tile(np.sqrt(kc_i) / 2.0, NBLK * H) for ci in range(n_cts)]

    E = []
    for ci in range(n_cts):
        z = ops.add(S[ci], shift[ci])
        z = cheb_exp(ops, z, cfg, sf)
        z = ops.bts(z, half_scaled=False)                    # ONE exp bts per ct
        for _ in range(cfg["n_squarings"]):
            z = ops.ctmult(z, z)
        E.append(ops.ptmult(z, amask[ci]))

    def row_sum(cts):                                        # 6 rots/ct + cross-ct add
        acc = ops.allreduce_stride_tH(cts[0].copy(), +1)
        for x in cts[1:]:
            acc = ops.add(acc, ops.allreduce_stride_tH(x.copy(), +1))
        return acc

    sden = row_sum(E) + floor
    assert np.allclose(sden.reshape(NBLK, tH), sden.reshape(NBLK, tH)[0], atol=0.0), \
        "delta-block denominator is not tH-periodic"
    if exact_recip:
        recip = 1.0 / sden
    else:
        F0 = cfg["init_alpha"] - cfg["init_beta"] * sden
        recip = fresh_recip(ops, sden, F0, cfg["gs_iters_scaled"] + 1)
    Y = [ops.ctmult(E[ci], recip) for ci in range(n_cts)]

    last = cfg["refinement_iters"] - 1
    y_entry = {}
    for r in range(cfg["refinement_iters"]):
        Z2 = []
        for ci in range(n_cts):
            Z2.append(ops.ptmult(ops.ctmult(Y[ci], Y[ci]), zsc[ci]))
        s2 = row_sum(Z2) + floor
        if exact_recip:
            rrec = 1.0 / s2
        else:
            kr = np.array([sm_kc_r(cfg, r, kc) for kc in kc_i])
            al = np.where(valid_row > 0, cfg["refine_alpha"][r] * np.tile(np.sqrt(kr), NBLK * H),
                          cfg["refine_alpha"][r])
            be = np.where(valid_row > 0, -cfg["refine_beta"][r] * np.tile(kr, NBLK * H),
                          -cfg["refine_beta"][r])
            Fr = ops.ptmult(s2, be) + al
            rrec = fresh_recip(ops, s2, Fr, int(cfg["per_step_refine_iters"][r]) + 1)
        if r < last:
            Y = [ops.ctmult(Z2[ci], rrec) for ci in range(n_cts)]
        else:
            # FINAL refine: entry extraction folded into the per-entry zscale mask —
            # ysq once per ct, then per-entry ptmult (same count/level as A's zscale).
            ysq = [ops.ctmult(Y[ci], Y[ci]) for ci in range(n_cts)]
            zsc_row = np.sqrt(kc_i) / 2.0
            for (g, d, cur) in sched:
                D = delta_abs(g, d, G)
                ci, b = D // NBLK, D % NBLK
                act = entry_active(g, d, cur, n_cur, P)
                zs = np.zeros(N)
                zs.reshape(NBLK, H, T)[b][act] = zsc_row[None, :].repeat(H, 0)[act]
                z2e = ops.ptmult(ysq[ci], zs)
                y_entry[(g, d)] = ops.ctmult(z2e, rrec)
            # NOTE: Z2 above already counted a packed square per ct; reuse ysq for it in
            # a real impl (here we keep both for op-count honesty: +1 ptmult/ct, not /entry).

    # --- softmax_v: UNCHANGED per-entry leg; broadcast is position-agnostic, so the
    # prob living in block Delta (not block 0) broadcasts identically.
    out = np.zeros(N)
    for (g, d, cur) in sched:
        p = ops.allreduce_stride_tH(y_entry[(g, d)], -1)
        vrot = ops.rot(v_group[g], -d)
        out = ops.add(out, ops.ctmult(p, vrot))
    return out


# ---------------------------------------------------------------- references

def dense_refs(qv, kv, vv, P, n_cur, cfg, exact_recip):
    """Slotless per-row THOR chain + exact softmax. qv[h,c,i], kv/vv[h,c,j]."""
    sf = 2.0 ** (-cfg["n_squarings"] - cfg["refinement_iters"])
    mean = (cfg["clip_hi"] + cfg["clip_lo"]) / 2.0
    a, b = cfg["cheb_a"] / sf, cfg["cheb_b"] / sf
    out_thor = np.zeros((H_REAL, DH_REAL, n_cur))
    out_exact = np.zeros((H_REAL, DH_REAL, n_cur))
    basin = []
    for h in range(H_REAL):
        for i in range(n_cur):
            kc = P + i + 1
            x = (qv[h, :, i] @ kv[h, :, :kc]) / np.sqrt(DH_REAL)
            out_exact[h, :, i] = vv[h, :, :kc] @ (np.exp(x - x.max()) /
                                                  np.exp(x - x.max()).sum())
            t = (2.0 * (x - mean) - (a + b)) / (b - a)
            z = chebyshev.chebval(t, np.asarray(cfg["cheb_coeffs"]))
            for _ in range(cfg["n_squarings"]):
                z = z * z
            e = z / kc
            s = e.sum()
            basin.append(s)
            if exact_recip:
                y = e / s
            else:
                y = e * scalar_gs(s, cfg["init_alpha"] - cfg["init_beta"] * s,
                                  cfg["gs_iters_scaled"] + 1)
            for r in range(cfg["refinement_iters"]):
                z2 = y * y * (np.sqrt(kc) / 2.0)
                s2 = z2.sum()
                if exact_recip:
                    y = z2 / s2
                else:
                    kr = sm_kc_r(cfg, r, kc)
                    F = cfg["refine_alpha"][r] * np.sqrt(kr) - cfg["refine_beta"][r] * kr * s2
                    y = z2 * scalar_gs(s2, F, int(cfg["per_step_refine_iters"][r]) + 1)
            out_thor[h, :, i] = vv[h, :, :kc] @ y
    return out_thor, out_exact, np.array(basin)


def scalar_gs(D, F0, iters):
    R, D_neg = F0, -D * F0
    F = D_neg + 2.0
    for i in range(1, iters):
        R = R * F
        if i + 1 < iters:
            D_neg = D_neg * F
            F = D_neg + 2.0
    return R


# ---------------------------------------------------------------- driver

def pack(fn3):
    """Pack x[h,c,·] -> slot c*tH + h*t + lane."""
    v = np.zeros(N)
    m = v.reshape(NBLK, H, T)
    for c in range(DH_REAL):
        for h in range(H_REAL):
            for lane in range(T):
                m[c, h, lane] = fn3(h, c, lane)
    return v


def run_case(name, P, n_cur, cfg, exact_recip=True, noise=0.0, score_gain=2.55):
    rng = np.random.default_rng(7)
    G = P // T
    K_tot = P + n_cur
    qv = rng.normal(size=(H_REAL, DH_REAL, T)) * score_gain
    qv[:, :, n_cur:] = 0.0
    kv = rng.normal(size=(H_REAL, DH_REAL, G * T + T)) * score_gain
    kv[:, :, K_tot:] = 0.0
    vv = rng.normal(size=(H_REAL, DH_REAL, G * T + T))
    vv[:, :, K_tot:] = 0.0

    # band guard: shifted scores must stay inside the cheb domain (squeeze regime)
    mean = (cfg["clip_hi"] + cfg["clip_lo"]) / 2.0
    wall = -cfg["cheb_a"] / (2.0 ** (-cfg["n_squarings"] - cfg["refinement_iters"]))
    smax = 0.0
    for h in range(H_REAL):
        x = (qv[h].T @ kv[h]) / np.sqrt(DH_REAL)
        smax = max(smax, np.abs(x[:n_cur, :K_tot] - mean).max())
    assert smax < wall, f"synthetic scores out of cheb band: {smax:.1f} >= {wall:.1f}"

    q_ct = pack(lambda h, c, i: qv[h, c, i])
    k_group = [pack(lambda h, c, l, g=g: kv[h, c, g * T + l]) for g in range(G + 1)]
    v_group = [pack(lambda h, c, l, g=g: vv[h, c, g * T + l]) for g in range(G + 1)]
    sched, _ = schedule(P, n_cur)

    opsA = Ops(noise=noise, seed=1)
    outA = pipeline_A(opsA, q_ct, k_group, v_group, sched, G, P, n_cur, cfg, exact_recip)
    opsB = Ops(noise=noise, seed=2)
    outB = pipeline_B(opsB, q_ct, k_group, v_group, sched, G, P, n_cur, cfg, exact_recip)

    thor, exact, basin = dense_refs(qv, kv, vv, P, n_cur, cfg, exact_recip)

    sel = np.zeros(N, dtype=bool)
    sel.reshape(NBLK, H, T)[:DH_REAL, :H_REAL, :n_cur] = True
    ref_thor = np.zeros(N)
    ref_thor.reshape(NBLK, H, T)[:DH_REAL, :H_REAL, :n_cur] = \
        np.transpose(thor, (1, 0, 2))
    ref_exact = np.zeros(N)
    ref_exact.reshape(NBLK, H, T)[:DH_REAL, :H_REAL, :n_cur] = \
        np.transpose(exact, (1, 0, 2))

    dAB = np.abs(outA - outB)[sel].max()
    dAT = np.abs(outA - ref_thor)[sel].max()
    dBT = np.abs(outB - ref_thor)[sel].max()
    dAE = np.abs(outA - ref_exact)[sel].max()
    scale = np.abs(ref_exact[sel]).max()

    n_e = len(sched)
    n_cts = (P + n_cur + NBLK - 1) // NBLK

    print(f"\n=== {name}: P={P} n_cur={n_cur} entries={n_e} delta-cts={n_cts} "
          f"(recip={'exact' if exact_recip else 'GS'}, noise={noise:g})")
    print(f"  A vs B (layout equivalence) : max|d| = {dAB:.3e}  (scale {scale:.2f})")
    print(f"  A vs dense THOR reference   : max|d| = {dAT:.3e}")
    print(f"  B vs dense THOR reference   : max|d| = {dBT:.3e}")
    print(f"  A vs exact softmax attention: max|d| = {dAE:.3e}  "
          f"(THOR approx error, layout-independent)")
    print(f"  GS-basin sden range (valid rows): [{basin.min():.4f}, {basin.max():.4f}] "
          f"(init y0>0 wall: {cfg['init_alpha']/cfg['init_beta']:.1f})")
    for tag, o in (("A", opsA), ("B", opsB)):
        est = o.c["bts"] * 0.086
        print(f"  ops[{tag}]: bts={o.c['bts']:4d} (~{est:5.1f}s @86ms)  "
              f"rot={o.c['rot']:5d}  ctmult={o.c['ctmult']:5d}  "
              f"ptmult={o.c['ptmult']:4d}  add={o.c['add']:5d}  "
              f"bts|in|max={max(o.bts_max_abs):.2f}")
    if noise == 0.0:
        assert dAB < 1e-6 * max(1.0, scale), "delta-block pipeline diverges from per-entry"
        assert dAT < 1e-6 * max(1.0, scale), "slot pipeline diverges from dense reference"
    return opsA, opsB


if __name__ == "__main__":
    cfg = load_cfg()
    noise = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

    # T=64 chunk-1 (P=32): 95 union entries -> exactly 1 delta-block ct (64 diagonals)
    run_case("T64-c1", P=32, n_cur=32, cfg=cfg, exact_recip=True, noise=noise)
    # T=128 last chunk (P=96): 221 entries -> 2 delta-block cts (128 diagonals)
    run_case("T128-c3", P=96, n_cur=32, cfg=cfg, exact_recip=True, noise=noise)
    # partial chunk (T=58: P=32, n_cur=26): shape-fixed schedule, nt<t masks
    run_case("T58-c1(partial)", P=32, n_cur=26, cfg=cfg, exact_recip=True, noise=noise)

    # live-GS runs (config alpha/beta + sm_kc_r in-slot): value fidelity of the
    # calibrated reciprocal chain in both layouts
    run_case("T64-c1 GS", P=32, n_cur=32, cfg=cfg, exact_recip=False, noise=noise)
    run_case("T128-c3 GS", P=96, n_cur=32, cfg=cfg, exact_recip=False, noise=noise)
