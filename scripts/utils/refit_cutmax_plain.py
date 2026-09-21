#!/usr/bin/env python
"""Schedule-CONSTRAINED cutmax refit for a PLAIN HF GPT-2 (medium / large / xl).

WHY (measured, job 50185142): perseus.calibrate FREE-FITS the argmax schedule.
On gpt2-medium that produced p=[9,3,5,9,9] passes=[1,2,3,3,3] — both SLOW
(17.3 s/argmax vs base's 9.4) and FRAGILE (encrypted argmax MISSED at pos=3:
cutmax=1 vs fhe_argmax=326, z_mass=0.9951). CutMax runs over the vocab (50257,
identical for every GPT-2 size), so its cost must NOT scale with the model —
17.3s is purely the free-fit's extra passes.

DOCTRINE (mirrors he-aware-training/.../refit_cutmax_constrained.py): never
free-fit on a new checkpoint. Pin the frozen T5 schedule SHAPE — p, passes,
cascade_iters, and iteration 0 in full — from the reference config (gpt2_base,
the shipped ~9.4 s/argmax fit), and refit ONLY the bands / chords / prescales on
the target model's own logit rows. Shape frozen => runtime unchanged by
construction; bands refit => correct on THIS model's logit statistics.

Iteration 0 is pinned whole (not just its shape): it separates the tail lm_head's
Im-residue twin, and a gentler refit iter-0 amplification mass-splits it. The
reference iteration-0 constants were fit on PLAIN GPT-2 rows — and medium/large/xl
are plain GPT-2 too (raw HF, no HE-aware fine-tuning), the same family.

Run (login node, offline, ~minutes):
  HF_HOME=$SCRATCH/.cache HF_HUB_OFFLINE=1 PYTHONPATH=$REPO \\
    python scripts/utils/refit_cutmax_plain.py \\
      --model openai-community/gpt2-medium \\
      --pool $SCRATCH/.cache/perseus/pools/openwebtext_gpt2_2000000.npy \\
      --ref-config configs/model/approximation/gpt2_base/configs.json \\
      --out-config configs/model/approximation/gpt2_medium/configs.json
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# the fitting simulator lives with the calibration tooling in he-aware-training
_CALIB = os.environ.get("CALIB_DIR", "/data02/users/azirilli/he-aware-training/src/he_aware_training/scripts/preprocess")
sys.path.insert(0, _CALIB)
import _cutmax_calib as cm  # noqa: E402

T = 5   # frozen schedule length

# == perseus/configs/approximation/gpt2_cutmax.yaml (the knobs the shipped fits used)
KNOBS = dict(
    entry_scale=0.00390625, margin=1.15, c0=5.0, c_late=10.0, shift_max=8.0,
    spread_cap=1.0e8, mass_target=0.97, gap_floor=0.75, sum_margin=0.08,
    newton_per_pass=5, gs_sum_iters=6, band_margin=2.0, floor_safety=5.0,
    floor1=2.0e-4, floor2=2.0e-4, wall_x1=150.0, wall_x2=150.0,
    wall_y1=150.0, wall_y2=150.0, conv=0.05, t_max=T,
)


def load_rows(model_name, pool_path, n_rows, window, cache):
    """(n_rows, vocab) plaintext lm_head logit rows from the model's own forward."""
    if cache and os.path.exists(cache):
        rows = np.load(cache)
        print(f"[rows] cached {rows.shape} <- {cache}")
        return rows.astype(np.float64)

    import torch
    from perseus.hub import load_model

    pool = np.load(pool_path, mmap_mode="r")
    model = load_model(model_name, device="cpu")
    model.eval()

    n_win = int(np.ceil(n_rows / window))
    out = []
    # spread windows across the pool so the bands see varied text, not one document
    stride = max(window, (len(pool) - window) // max(n_win, 1))
    for w in range(n_win):
        off = 4096 + w * stride
        toks = np.asarray(pool[off:off + window], dtype=np.int64)
        if len(toks) < window:
            break
        with torch.no_grad():
            lg = model(torch.tensor(toks).unsqueeze(0)).logits[0]
        out.append(lg.to(torch.float64).numpy())
        print(f"[rows] window {w} @tok{off}: +{lg.shape[0]} rows")
    rows = np.concatenate(out)[:n_rows]
    print(f"[rows] total {rows.shape}")
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, rows)
        print(f"[rows] cached -> {cache}")
    return rows


def frozen_shape(ref_cfg_path):
    """The reference config's schedule shape + its whole iteration 0."""
    cm_ref = json.load(open(ref_cfg_path))["cutmax"]
    es = cm_ref["entry_scale"]
    it0 = dict(
        p=cm_ref["p"][0], c=cm_ref["c"][0], m=cm_ref["m"][0],
        passes=cm_ref["passes"][0], cascade_iters=cm_ref["cascade_iters"][0],
        chord_a=cm_ref["chord_a"][0], chord_b=cm_ref["chord_b"][0],
        # runtime s2_hi[0] carries the entry^2 fold; the sim wants the raw prescale
        g=cm_ref["s2_hi"][0] / (es * es),
    )
    print(f"[frozen] ref={ref_cfg_path}")
    print(f"[frozen] p={cm_ref['p']} passes={cm_ref['passes']} "
          f"cascade_iters={cm_ref['cascade_iters']}")
    return cm_ref["p"], cm_ref["passes"], cm_ref["cascade_iters"], it0


def constrained_derive(rows, knobs, P, PASSES, CI, IT0):
    """cm.derive() with the schedule shape pinned; bands/chords/prescales refit."""
    sim = cm._Sim(rows)
    k, bm = knobs["newton_per_pass"], knobs["band_margin"]
    gaps = np.sort(rows, axis=1)
    mass_mask = (gaps[:, -1] - gaps[:, -2]) >= knobs["gap_floor"]
    sched = []
    for it in range(T):
        if it == 0:
            e0 = dict(IT0)
            lo, hi = sim.s2_band()
            e0["lo"], e0["hi"] = lo / bm, hi * bm
            sim.step(e0["p"], e0["c"], e0["m"], (e0["lo"], e0["hi"]), k,
                     e0["passes"], e0["g"], (e0["chord_a"], e0["chord_b"]))
            sched.append(e0)
            print(f"[iter 0] PINNED to reference constants (twin separation); "
                  f"observed s2 band=[{lo:.4g},{hi:.4g}]")
            continue
        c = knobs["c_late"]
        lo, hi = sim.s2_band()
        lo, hi = lo / bm, hi * bm
        s2 = np.array([s2_ for _, s2_ in sim.pre])
        g_pre, ci_auto = cm.pick_prescale(lo, hi, knobs["floor_safety"])
        ci = CI[it]
        if ci_auto > ci:
            raise RuntimeError(f"iter {it}: band needs cascade_iters={ci_auto}, "
                               f"frozen shape has {ci} — shape cannot hold")
        wy = cm.WALL_Y[ci]
        chord_ok = (lo / g_pre) >= 1.0 / (wy * wy)
        min_passes, vmax = cm._calib_passes(s2, lo, hi, k, g_pre, chord_ok, ci)
        if min_passes is None or min_passes > PASSES[it]:
            raise RuntimeError(f"iter {it}: needs >= {min_passes} passes "
                               f"(vmax={vmax}), frozen shape has {PASSES[it]}")
        passes = PASSES[it]
        w_hi = max((d / (np.sqrt(s2_) * c)).max() for d, s2_ in sim.pre)
        m_final = round(knobs["margin"] * (1.0 + w_hi), 3)
        m = m_final if it + 1 == T else round(m_final / knobs["shift_max"], 3)
        p = P[it]
        ca, cb = (cm._chord(lo / g_pre, hi / g_pre) if chord_ok
                  else (1.0 / np.sqrt(hi / g_pre), 0.0))
        sim.step(p, c, m, (lo, hi), k, passes, g_pre, chord_ok)
        sched.append({"p": p, "c": c, "m": m, "lo": lo, "hi": hi, "g": g_pre,
                      "passes": passes, "chord_a": float(ca), "chord_b": float(cb),
                      "cascade_iters": ci})
        masses = sim.masses()
        gate = masses[mass_mask] if mass_mask.any() else masses
        print(f"[iter {it}] band=[{lo:.4g},{hi:.4g}] g={g_pre:.4g} ci={ci} "
              f"passes={passes} (min {min_passes}) chord={chord_ok} "
              f"min_gated_mass={gate.min():.4f}")
    return sched, mass_mask


def bad_mask(rows, sched, k, rel, absf, sum_band, seed=0):
    """cm.validate's per-row verdict (it only returns the count)."""
    rng = np.random.default_rng(seed)
    sim = cm._Sim(rows)
    for e in sched:
        sim.s2_band()
        ci = e.get("cascade_iters", 2)
        sim.step(e["p"], e["c"], e["m"], (e["lo"], e["hi"]), k, e["passes"],
                 e.get("g"), (e.get("chord_a", 0.0), e.get("chord_b", 0.0)),
                 rng, rel, absf, casc_rel=cm.FLOOR[ci] / 3.0, casc_absf=cm.FLOOR[ci],
                 wall_x=cm.WALL_X[ci], wall_y=cm.WALL_Y[ci])
    zs, _, sums = sim.finalize(rng, rel, absf)
    out = np.zeros(len(rows), dtype=bool)
    for i, (r, z) in enumerate(zip(rows, zs)):
        top = int(np.argsort(r)[-1])
        out[i] = (int(np.argmax(z)) != top or bool(sim.crossed[i])
                  or not (sum_band[0] <= sums[i] <= sum_band[1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai-community/gpt2-medium")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--ref-config", default="configs/model/approximation/gpt2_base/configs.json")
    ap.add_argument("--out-config", required=True)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--rows-cache", default="")
    ap.add_argument("--band-margin", type=float, default=None)
    ap.add_argument("--passes", default="",
                    help="override the frozen passes list, e.g. 1,3,3,3,2. The SANCTIONED "
                         "minimal deviation when an iteration's band cannot converge in the "
                         "frozen pass count (+1 pass ~= +0.8 s argmax); p/T/cascade_iters "
                         "stay frozen so the schedule shape — and thus the runtime — holds.")
    ap.add_argument("--gate-rows", type=int, default=4,
                    help="how many leading rows the decode gate actually decodes")
    ap.add_argument("--dry-run", action="store_true", help="fit + validate, do not write")
    args = ap.parse_args()

    if args.band_margin is not None:
        KNOBS["band_margin"] = args.band_margin
    cm.set_envelopes(KNOBS)

    P, PASSES, CI, IT0 = frozen_shape(args.ref_config)
    if args.passes:
        override = [int(x) for x in args.passes.split(",")]
        if len(override) != T:
            raise SystemExit(f"--passes needs {T} entries, got {len(override)}")
        print(f"[frozen] passes OVERRIDE {PASSES} -> {override} "
              f"(p/cascade_iters still frozen)")
        PASSES = override
    rows = load_rows(args.model, args.pool, args.rows, args.window, args.rows_cache)
    sched, mass_mask = constrained_derive(rows, KNOBS, P, PASSES, CI, IT0)
    k = KNOBS["newton_per_pass"]

    # sum band from the mass-gated rows (the runtime GS chain divides by this sum)
    sim = cm._Sim(rows[mass_mask])
    for e in sched:
        sim.s2_band()
        sim.step(e["p"], e["c"], e["m"], (e["lo"], e["hi"]), k, e["passes"],
                 e["g"], (e["chord_a"], e["chord_b"]))
    _, sum_band, _ = sim.finalize()
    if not sum_band[0] > 0.0:
        raise RuntimeError(f"degenerate noise-free sum band {sum_band}")
    smr = KNOBS["sum_margin"]
    emit = (sum_band[0] * (1 - smr), sum_band[1] * (1 + smr))
    print(f"[sum band] noise-free {sum_band} -> emit {emit} "
          f"kappa={emit[1]/emit[0]:.2f}")

    # Widen for the live tail: the mass-gated sim never sees sub-gap (near-tie) rows,
    # and the tail lm_head row carries a twin-doubled sum. The runtime geo-mid
    # prescales then minimax-inits, so ANY kappa converges in-band => widening is
    # free correctness margin (gs_sum_iters is fixed, so it costs no time).
    emit = (min(emit[0], sum_band[0] * 0.11), max(emit[1], 1.05))
    print(f"[sum band] live-tail widened -> {emit} kappa={emit[1]/emit[0]:.2f}")

    # twin-augmented validation: inject the tail lm_head Im-residue (rival lane at
    # 0.9x the top logit, +slots offset) — separation must survive it
    twin = rows[mass_mask].copy()
    ti = np.argmax(twin, axis=1)
    tv = twin[np.arange(len(twin)), ti]
    rival = (ti + 32768) % twin.shape[1]
    twin[np.arange(len(twin)), rival] = np.maximum(
        twin[np.arange(len(twin)), rival], 0.9 * np.abs(tv) * np.sign(tv))
    ok = True
    for name, ci, rws in (("ambient1", 1, rows), ("ambient2", 2, rows),
                          ("twin@0.9", 1, twin)):
        fails, mass, _ = cm.validate(rws, sched, k, cm.FLOOR[ci] / 3.0,
                                     cm.FLOOR[ci], sum_band=emit)
        print(f"[validate {name}] {len(rws)-fails}/{len(rws)} "
              f"min_mass={min(mass):.4f}")
        ok &= fails == 0

    # Per-row detail: are the failures inherent near-ties (which no schedule can
    # separate, and where the plaintext argmax is itself ambiguous), and do the
    # positions the decode gate actually runs survive?
    bad = bad_mask(rows, sched, k, cm.FLOOR[1] / 3.0, cm.FLOOR[1], emit)
    srt = np.sort(rows, axis=1)
    gap = srt[:, -1] - srt[:, -2]
    if bad.any():
        print(f"[detail] {bad.sum()}/{len(rows)} fail — median top-2 gap: "
              f"fail={np.median(gap[bad]):.4f} vs pass={np.median(gap[~bad]):.4f}; "
              f"fails with gap<{KNOBS['gap_floor']}: "
              f"{int((gap[bad] < KNOBS['gap_floor']).sum())}/{int(bad.sum())}")
    g = args.gate_rows
    print(f"[detail] decode-gate positions rows[0:{g}] "
          f"{'ALL PASS' if not bad[:g].any() else 'FAIL ' + str(np.where(bad[:g])[0].tolist())} "
          f"(gaps {np.round(gap[:g], 3).tolist()})")

    es = KNOBS["entry_scale"]
    section = {
        "entry_scale": es, "newton_per_pass": k, "newton_polish": 0,
        "gs_sum_iters": KNOBS["gs_sum_iters"],
        "sum_lo": emit[0], "sum_hi": emit[1],
        "p": [e["p"] for e in sched], "c": [e["c"] for e in sched],
        "m": [e["m"] for e in sched],
        "s2_hi": [e["g"] * (es * es if i == 0 else 1.0) for i, e in enumerate(sched)],
        "passes": [e["passes"] for e in sched], "ex2": [0] * T,
        "chord_a": [e["chord_a"] for e in sched],
        "chord_b": [e["chord_b"] for e in sched],
        "cascade_iters": [e["cascade_iters"] for e in sched],
        "oracle_rows": int(rows.shape[0]), "oracle_noise": 0.0002,
    }
    print(f"[section] p={section['p']} passes={section['passes']} "
          f"cascade_iters={section['cascade_iters']}")
    if not ok:
        print("[warn] some validation rows FAILED — inspect before deploying")
    if args.dry_run:
        print("[dry-run] not written")
        return

    out = Path(args.out_config)
    shutil.copy(out, out.with_suffix(".json.pre_cmrefit"))
    cfg = json.load(open(out))
    cfg["cutmax"] = section
    out.write_text(json.dumps(cfg, indent=2))
    print(f"[emit] {out} — cutmax section replaced "
          f"(backup: {out.name}.pre_cmrefit; forward sections untouched)")


if __name__ == "__main__":
    main()
