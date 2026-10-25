"""CutMax schedule calibration — the configs.json "cutmax" section.

Derives the argmax schedule from the model's own lm_head logits rows, searching over the plaintext runtime replica
(`cutmax_sim`): outer iterations (p, c, m) from the shift_max wall map with spread-capped powers; a geometric-mid
prescale per iteration (s2_hi := sqrt(lo * hi), consumed unchanged by the runtime's fold); a pass-0 chord init per
iteration (from-below secant of x^-1/2 over the calibrated band, gamma-shrunk so Newton stays monotone), the const
from-below init on later passes; and per-iteration pass counts (the smallest that converges the worst derivation token
to 5 %). Validation replays the measured bootstrap envelopes at every noise site (`cutmax_sim.noisy`) plus a hard-wall
breach check per row. `fit_scaled` fits the scaled normalisers (head.cutmax_argmax_packed) on such a schedule.
"""
import logging
import math

import numpy as np

from perseus.calibrate.approximations import cutmax_sim as cs
from perseus.calibrate.registry import Approximation, register

log = logging.getLogger(__name__)


def _calib_passes(s2, lo, hi, k, g, chord_ok, iters):
    truth = 1.0 / np.sqrt(s2)
    for passes in range(1, 8):
        est, vmax, _ = cs.cascade(s2, lo, hi, k, passes, g, chord_ok)
        if vmax > cs.WALL_Y[iters]:
            return None, vmax
        if np.abs(est / truth - 1.0).max() <= cs.CONV:
            return passes, vmax
    return None, vmax


def derive(rows, knobs, mass_mask=None):
    """Outer schedule (p/c/m per the oracle logic) + calibrated cascade.

    Bands, chord inits and passes come from ALL rows, widened by band_margin
    (a chord init is LETHAL out-of-band: y0 = ca - cb*x < 0 diverges Newton;
    the runtime distribution also shifts vs the calib rows). mass_mask (the
    gap-floored rows) gates only the STOP trigger — near-ties are semantic
    ties that never reach mass_target and must not run the schedule long.
    """
    sim = cs.Sim(rows)
    k = knobs["newton_per_pass"]
    bm = knobs["band_margin"]
    sched = []
    for it in range(knobs["t_max"]):
        c = knobs["c0"] if it == 0 else knobs["c_late"]
        lo, hi = sim.s2_band()
        lo, hi = lo / bm, hi * bm            # out-of-band safety
        s2 = np.array([s2_ for _, s2_ in sim.pre])
        g_pre, casc_iters = cs.pick_prescale(lo, hi, knobs["floor_safety"])
        wy = cs.WALL_Y[casc_iters]
        chord_ok = (lo / g_pre) >= 1.0 / (wy * wy)
        passes, vmax = _calib_passes(s2, lo, hi, k, g_pre, chord_ok, casc_iters)
        if passes is None:
            raise RuntimeError(f"cutmax calib iter {it}: cascade won't "
                               f"converge under the wall (vmax={vmax:.2f})")
        probe = [(d / (np.sqrt(s2_) * c)).max() for d, s2_ in sim.pre]
        w_lo, w_hi = min(probe), max(probe)
        masses = sim.masses()
        gate = masses[mass_mask] if mass_mask is not None else masses
        final = gate.min() >= knobs["mass_target"] or it + 1 == knobs["t_max"]
        m_final = round(knobs["margin"] * (1.0 + w_hi), 3)
        m = m_final if final else round(m_final / knobs["shift_max"], 3)
        ratio = (1.0 + w_hi) / (1.0 + w_lo)
        p = knobs["p_candidates"][-1]
        for cand in knobs["p_candidates"]:
            if ratio ** (2 * cand) <= knobs["spread_cap"]:
                p = cand
                break
        ca, cb = (cs.chord(lo / g_pre, hi / g_pre) if chord_ok
                  else (1.0 / np.sqrt(hi / g_pre), 0.0))
        sim.step(p, c, m, (lo, hi), k, passes, g_pre, chord_ok)
        sched.append({"p": p, "c": c, "m": m, "lo": lo, "hi": hi, "g": g_pre,
                      "passes": passes, "ex2": 0, "chord_a": float(ca),
                      "chord_b": float(cb), "cascade_iters": casc_iters})
        if final:
            break
    return sched


def validate(rows, sched, k, rel, absf, seed=0, sum_band=None):
    """rel/absf = the AMBIENT (vector-lane) regime; each iteration's cascade
    uses its own cascade_iters regime. Hard-wall replay: a bts-site wall
    breach fails the row (Gaussian noise alone is blind to divergence — the
    18/34 live lesson); sum_band = the EMITTED [sum_lo, sum_hi]: noisy sums
    outside it fail (GS reciprocal divergence, the top_mass +-1e3..1e5 live
    fingerprint). Inits replay the emitted (chord_a, chord_b) exactly."""
    rng = np.random.default_rng(seed)
    sim = cs.Sim(rows)
    for e in sched:
        sim.s2_band()
        ci = e.get("cascade_iters", 2)
        sim.step(e["p"], e["c"], e["m"], (e["lo"], e["hi"]), k, e["passes"],
                 e.get("g"), (e.get("chord_a", 0.0), e.get("chord_b", 0.0)),
                 rng, rel, absf,
                 casc_rel=cs.FLOOR[ci] / 3.0, casc_absf=cs.FLOOR[ci],
                 wall_x=cs.WALL_X[ci], wall_y=cs.WALL_Y[ci])
    zs, sum_band_obs, sums = sim.finalize(rng, rel, absf)
    fails, top_mass = 0, []
    for i, (r, z) in enumerate(zip(rows, zs)):
        order = np.argsort(r)
        bad = int(np.argmax(z)) != int(order[-1]) or bool(sim.crossed[i])
        if sum_band is not None and not (sum_band[0] <= sums[i] <= sum_band[1]):
            bad = True
        fails += bad
        top_mass.append(float(z[order[-1]]))
    return fails, top_mass, sum_band_obs


def fit_cutmax_section(rows, knobs):
    """rows: [N, vocab] float array of lm_head logits. knobs: the approximation
    set's `cutmax` subtree. Returns the configs.json "cutmax" section."""
    cs.set_envelopes(knobs)
    gaps = np.sort(rows, axis=1)
    gaps = gaps[:, -1] - gaps[:, -2]
    mass_mask = gaps >= knobs["gap_floor"]
    if not mass_mask.any():
        mass_mask = np.ones(len(rows), dtype=bool)
    sched = derive(rows, knobs, mass_mask=mass_mask)
    k = knobs["newton_per_pass"]

    sim = cs.Sim(rows[mass_mask])
    for e in sched:
        sim.s2_band()
        sim.step(e["p"], e["c"], e["m"], (e["lo"], e["hi"]), k, e["passes"],
                 e.get("g"), (e.get("chord_a", 0.0), e.get("chord_b", 0.0)))
    _, sum_band, _ = sim.finalize()
    if not (sum_band[0] > 0.0):
        raise RuntimeError(f"cutmax calib: degenerate noise-free sum band "
                           f"{sum_band}")
    sm = knobs["sum_margin"]
    emit_band = (sum_band[0] * (1.0 - sm), sum_band[1] * (1.0 + sm))
    if emit_band[1] / emit_band[0] > 4.0:
        raise RuntimeError(f"cutmax calib: sum band kappa "
                           f"{emit_band[1]/emit_band[0]:.2f} > 4 breaks the "
                           f"geo-mid GS init (needs den*w0 in (0,2))")

    lines = []
    for name, ci in (("ambient1", 1), ("ambient2", 2)):
        fails, mass, _ = validate(rows, sched, k, cs.FLOOR[ci] / 3.0, cs.FLOOR[ci],
                                  sum_band=emit_band)
        lines.append(f"{name}: {len(rows)-fails}/{len(rows)} "
                     f"min_mass={min(mass):.3f}")
    log.info(f"[cutmax] T={len(sched)} passes={sum(e['passes'] for e in sched)} "
          f"casc_iters={[e['cascade_iters'] for e in sched]} "
          f"(mass gate on {int(mass_mask.sum())}/{rows.shape[0]} rows, "
          f"gap_floor={knobs['gap_floor']}) | " + " | ".join(lines))

    es = knobs["entry_scale"]
    return {
        "entry_scale": es,
        "newton_per_pass": k,
        "newton_polish": knobs["newton_polish"],
        "gs_sum_iters": knobs["gs_sum_iters"],
        "sum_lo": emit_band[0],
        "sum_hi": emit_band[1],
        "p":       [e["p"] for e in sched],
        "c":       [e["c"] for e in sched],
        "m":       [e["m"] for e in sched],
        "s2_hi":   [e["g"] * (es * es if i == 0 else 1.0)
                    for i, e in enumerate(sched)],
        "passes":  [e["passes"] for e in sched],
        "ex2":     [e["ex2"] for e in sched],
        "chord_a": [e["chord_a"] for e in sched],
        "chord_b": [e["chord_b"] for e in sched],
        "cascade_iters": [e["cascade_iters"] for e in sched],
        "oracle_rows": int(rows.shape[0]),
        "oracle_noise": knobs["noise"],
    }


def _scaled_forward(x, sec, upto):
    """The scaled CutMax (head.cutmax_argmax_packed, exact arithmetic) on rows x [B, n] times entry_scale, through
    iteration upto - 1; returns iteration upto's sum (x - mu)^2, or the final sum S when upto is the last + 1."""
    from perseus.impl.poly import NumpyOps, inv_sqrt_newton_scaled
    n = x.shape[1]
    for i in range(upto):
        d = x - x.mean(axis=1, keepdims=True)
        xs, c, m = sec["x_scale"][i], sec["c"][i], sec["m"][i]
        f = inv_sqrt_newton_scaled(NumpyOps(), (d * d).sum(axis=1) * xs, sec["newton"][i],
                                   0.5 * math.sqrt(n * xs) / (c * m))
        x = (2.0 * (d * f[:, None] + 0.5 / m)) ** sec["p"][i]
    if upto == len(sec["p"]):
        return x.sum(axis=1)
    d = x - x.mean(axis=1, keepdims=True)
    return (d * d).sum(axis=1)


def _steps_not_below(ratio, tol, zero):
    """poly.scaled_newton_steps on [1 / ratio, 1] (from below, to tol), the last step scaled so the result is never
    below the exact 1 / sqrt(x) on the band: within [1, 1 + tol] of it (a normaliser never softer than the exact
    one)."""
    from perseus.impl.poly import NumpyOps, inv_sqrt_newton_scaled, scaled_newton_steps
    steps = scaled_newton_steps(ratio, tol, zero)
    x = np.geomspace(1.0 / ratio, 1.0, 20001)
    beta = 1.0 / float((inv_sqrt_newton_scaled(NumpyOps(), x, steps) * np.sqrt(x)).min())
    return [list(s) for s in steps[:-1]] + [[steps[-1][0] * beta, steps[-1][1] * beta]]


def fit_scaled(rows, section, tol=5e-3, sum_tol=1e-5, lo_margin=2.0, hi_margin=1.25, heavy_margin=4.0,
               sum_margin=1.25, zero=3.0, batch=512):
    """The scaled normalisers (configs.json "cutmax" with x_scale / newton / sum_scale / sum_newton) on the outer
    schedule of `section` (its entry_scale, p, c, m kept). rows: a list of [N, vocab] logit arrays (memmaps are read
    `batch` rows at a time; rows with a non-finite logit are skipped). Per iteration, in order, the band of
    sum (x - mu)^2 over the rows (the earlier iterations run with their fitted steps) is widened by lo_margin below and
    hi_margin above (heavy_margin after the schedule's largest power: there sum (x - mu)^2 goes as the top value to
    twice that power, the heaviest tail), x_scale = 1 / its top, and the Newton steps reach a relative error tol on
    it, never below the exact value; the final 1 / S the same with sum_tol and sum_margin."""
    sec = {"entry_scale": section["entry_scale"], "p": list(section["p"]), "c": list(section["c"]),
           "m": list(section["m"]), "x_scale": [], "newton": []}
    heavy = 1 + int(np.argmax(sec["p"]))

    def band(upto):
        vals = []
        for r in rows:
            for s in range(0, len(r), batch):
                x = np.asarray(r[s:s + batch], dtype=np.float64)
                vals.append(_scaled_forward(x[np.isfinite(x).all(axis=1)] * sec["entry_scale"], sec, upto))
        v = np.concatenate(vals)
        return float(v.min()), float(v.max())

    for i in range(len(sec["p"])):
        lo, hi = band(i)
        lo, hi = lo / lo_margin, hi * (heavy_margin if i == heavy else hi_margin)
        sec["x_scale"].append(1.0 / hi)
        sec["newton"].append(_steps_not_below(hi / lo, tol, zero))
    lo, hi = band(len(sec["p"]))
    lo, hi = lo / lo_margin, hi * sum_margin
    sec["sum_scale"], sec["sum_newton"] = 1.0 / hi, _steps_not_below(hi / lo, sum_tol, zero)
    log.info(f"[cutmax] scaled normalisers: {[len(s) for s in sec['newton']]} Newton steps per iteration, "
             f"{len(sec['sum_newton'])} for the sum")
    return sec


class _CutmaxCollector:
    """Full-vocab lm_head logit rows from the model outputs (the fit oracle)."""

    def __init__(self, cfg):
        self.target = cfg.rows
        self._bufs = []
        self._collected = 0

    def attach(self, name, module):
        raise RuntimeError("cutmax is model-level; it attaches to outputs")

    def on_output(self, output):
        import torch
        if self._collected >= self.target:
            return
        lg = output.logits if hasattr(output, "logits") else output[0]
        lg2d = lg.float().reshape(-1, lg.size(-1))
        good = torch.isfinite(lg2d).all(dim=-1) & (lg2d.abs().amax(dim=-1) < 1e4)
        lg2d = lg2d[good]   # non-finite rows poison the cutmax bands
        keep = min(self.target - self._collected, lg2d.size(0))
        idx = torch.randperm(lg2d.size(0), device=lg2d.device)[:keep]
        self._bufs.append(lg2d[idx].detach().cpu())
        self._collected += keep

    def finalize(self):
        import torch
        return torch.cat(self._bufs) if self._bufs else None


APPROX = register(Approximation(
    kind="cutmax",
    section="cutmax",
    matches=lambda m: False,
    make_collector=_CutmaxCollector,
    fit_section=lambda rows, cfg: fit_cutmax_section(rows.numpy(), cfg),
    model_level=True,
))
