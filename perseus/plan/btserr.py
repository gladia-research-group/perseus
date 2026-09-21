"""The bootstrap accuracy model the placer plans against.

The min-cut optimises bootstrap count against a level budget; this module prices each
refresh site so the cut also respects the EvalMod band. The model:

* the bootstrap computes `C·sin(m/C)` with `C = 2^CF/2π`: a correction factor does not
  buy accuracy, it slides a fixed-quality band `[0.003, 0.03]·2^CF` (centre `0.01·2^CF`,
  ~2e-3 relative inside it) by 2x per step;
* the quantity the sine sees is the max plaintext COEFFICIENT after CtS, not the slot
  max — a broadcast constant and spread data differ by orders at the same slot max;
* sparse routing lowers the noise floor and is neutral above it (orthogonal to CF);
* the offset transform `bootstrap(ct−c)+c` costs no level and lets CF follow the residual.

`data/bts_accuracy_<chain>.json` is measured on data of known period, indexed by slot
amplitude, so a lookup keyed by `(cf, period, route, slot |m|)` already prices the
coefficient effect from what the capture records (`output_max_abs`, `pack_period`).
When a capture carries the coefficient itself (`output_max_coeff`) the period-1 column
is the coefficient-indexed law and is used instead.

Unknown period defaults are deliberately opposite: routing stays DENSE (an unproven
sparse route folds residue classes silently), while the error model prices it as
period 1 (all energy in one coefficient — the pessimistic reading).

Pure python: no GPU, no FIDESlib, no import of the runtime.
"""
from __future__ import annotations

import bisect
import json
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

log = logging.getLogger(__name__)

# ── the measured band ──────────────────────────────────────────────────────
BAND_LO_FRAC = 0.003
BAND_HI_FRAC = 0.03
BAND_CENTRE_FRAC = 0.01

# The prescale reach: a site is admissible while its magnitude is within this multiple of
# the band TOP of the chosen CF, prescaled down to the band centre (|m| <= 30720 at CF=9).
# Reach is a safety knob, not an accuracy one: the prescale/restore pair is scale-invariant
# in relative terms, so what a deep prescale buys is exposure to magnitude drift between
# the captured trajectory and the runtime one; `mag_safety` (κ) prices that drift.
DEFAULT_PRESCALE_REACH = 2000.0

# CF's legal window. The floor is the chain's `deg = FIRST_MOD_BITS - BTP_SCALE_BITS`
# (2 on n32; Bootstrap throws `deg exceeds correctionFactor` below it). The ceiling is a
# search-window bound, not a quality claim: an explicit correctionFactor bypasses OpenFHE's
# AUTO clamp, CF slides the band 2x per step, and a higher ceiling lets the worst site use
# a shallower prescale (|m|=615 needs 6.9 bits at cf=9, 1.9 bits at cf=14). Every placer
# plans in the same 2..20 window so arms differ by model, not by window.
DEFAULT_CF_MIN = 2
DEFAULT_CF_MAX = 20

# `pack_period` sentinel for aperiodic data: the capture writes the full slot count.
DENSE_PERIOD = 0
DEFAULT_SLOTS = 32768


def query_period(period: int | None) -> int:
    """The period the error model is queried with: a known proper period, else 1
    (all energy in one coefficient — the conservative reading of an unknown packing)."""
    return 1 if (period is None or period <= 0) else int(period)


def band(cf: float) -> tuple[float, float]:
    """The measured EvalMod usable band for a correction factor."""
    return BAND_LO_FRAC * 2.0 ** cf, BAND_HI_FRAC * 2.0 ** cf


def band_centre(cf: float) -> float:
    return BAND_CENTRE_FRAC * 2.0 ** cf


def cf_for_magnitude(mag: float) -> float:
    """`CF* ≈ log2(|m|/0.01)` — the band-centring CF. Unclamped."""
    return math.log2(max(mag, 1e-300) / BAND_CENTRE_FRAC)


# ── analytic model (the fallback, and the extrapolation outside the measured grid) ────

def rho(period: int | None, slots: int = 32768) -> float:
    """Max plaintext COEFFICIENT / max slot value, for data of period `p`.

    CtS moves coefficients into slots, so p-periodic data has non-zero coefficients only at
    the p multiples of N/p — its energy is spread over p of them and the max falls as
    `1/sqrt(p)`. MEASURED on the n32 sweep by sliding each period's bend branch onto the
    period-1 branch: p=512 gives 0.0440 against 1/sqrt(512) = 0.0442, and full-slot dense
    gives 0.0063 against 1/sqrt(32768) = 0.0055. Two independent points, one free parameter,
    and `rho(1) = 1` by construction — which is why a broadcast constant is the worst case.

    `period` None or <= 0 means aperiodic, i.e. the full slot count.
    """
    p = slots if (period is None or period <= 0) else int(period)
    return 1.0 / math.sqrt(max(p, 1))


def bend(mag: float, cf: float) -> float:
    """Relative error from the sine bend: `|1 - sin(x)/x|`, `x = 2π·m/2^CF`.

    One-parameter law. Above the first zero of `sin` the value is not
    merely inaccurate, it is SIGN-INVERTED — which is why the placer treats anything past
    the band as a refusal rather than as a graded cost.
    """
    x = 2.0 * math.pi * max(mag, 0.0) / 2.0 ** cf
    if x < 1e-12:
        return 0.0
    return abs(1.0 - math.sin(x) / x)


# ABSOLUTE noise floor, in units of `2^CF`, per ROUTE. The message is pre-scaled by 2^-corr,
# so the floor grows with CF exactly as the band moves with it — which is why no global CF
# can serve both a broadcast constant and a small-|m| lane, and why the two knobs are
# orthogonal: CF moves the band, the route lowers the floor.
#
# MEASURED (n32 sweep, period-1 data at A=1e-3, i.e. deep in the floor). The proportionality
# to 2^CF holds to 11% across the whole CF 2..9 range, which is what makes this one constant
# rather than a table:
#     dense  2.11e-5 .. 2.35e-5   (mean 2.2e-5)
#     s=512  1.52e-7 .. 2.01e-7   (mean 1.6e-7,  ~135x below dense)
#     s=1    1.6e-9  .. 6.6e-9    (mean 3.0e-9,  ~7000x below dense)
_FLOOR_ABS_PER_2CF = {0: 2.2e-5, 512: 1.6e-7, 1: 3.0e-9}


def _floor_abs(cf: float, route: int) -> float:
    r = int(route)
    if r in _FLOOR_ABS_PER_2CF:
        return _FLOOR_ABS_PER_2CF[r] * 2.0 ** cf
    # Unmeasured route: interpolate log-log in s between the measured anchors, and never
    # below the best measured floor. Being optimistic here would license a route the
    # measurement does not support.
    known = sorted(k for k in _FLOOR_ABS_PER_2CF if k > 0)
    lo = max((k for k in known if k <= r), default=known[0])
    hi = min((k for k in known if k >= r), default=known[-1])
    if lo == hi:
        return _FLOOR_ABS_PER_2CF[lo] * 2.0 ** cf
    t = (math.log(r) - math.log(lo)) / (math.log(hi) - math.log(lo))
    v = math.exp((1 - t) * math.log(_FLOOR_ABS_PER_2CF[lo])
                 + t * math.log(_FLOOR_ABS_PER_2CF[hi]))
    return v * 2.0 ** cf


def analytic_rel_err(mag: float, cf: float, route: int = 0,
                     period: int | None = 1, slots: int = 32768) -> float:
    """`bend(coeff) + floor/slot` — the closed form.

    `mag` is the SLOT magnitude (what the capture records); the bend is evaluated on the
    coefficient `mag·rho(period)`, because that is what EvalMod sees. Passing `period=1`
    makes `mag` the coefficient directly.
    """
    if mag <= 0.0:
        return float("inf")
    return bend(mag * rho(period, slots), cf) + _floor_abs(cf, route) / mag


# ── the measured table ───────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent / "data"


def default_table_path(chain: str) -> Path:
    return DATA_DIR / f"bts_accuracy_{chain}.json"


class AccuracyTable:
    """Measured `rel_err` curves, keyed by `(cf, period, route)` over amplitude.

    Lookups interpolate log-log in amplitude (the law is a power law below the band and a
    smooth sine above it, so log-log is the faithful interpolant) and fall back to the
    analytic form outside the measured amplitude range or for an unmeasured key.
    """

    def __init__(self, doc: dict | None = None):
        self.meta: dict = (doc or {}).get("meta", {})
        # (cf, period, route) -> (sorted amps, errs)
        self._curves: dict[tuple[int, int, int], tuple[list[float], list[float]]] = {}
        # (cf) -> sorted [(dc, as_is, offset)]
        self._offset: dict[int, list[tuple[float, float, float]]] = {}
        raw: dict[tuple[int, int, int], list[tuple[float, float]]] = {}
        for row in (doc or {}).get("wall", []):
            key = (int(row["cf"]), int(row["period"]), int(row["route"]))
            raw.setdefault(key, []).append((float(row["amp"]), float(row["rel_err"])))
        for key, pts in raw.items():
            pts.sort()
            self._curves[key] = ([p[0] for p in pts], [p[1] for p in pts])
        for row in (doc or {}).get("offset", []):
            self._offset.setdefault(int(row["cf"]), []).append(
                (float(row["dc"]), float(row["as_is"]), float(row["offset"])))
        for cf in self._offset:
            self._offset[cf].sort()

    # -- construction ---------------------------------------------------------------
    @classmethod
    def load(cls, path: Path | str | None) -> AccuracyTable:
        if path is None:
            return cls(None)
        p = Path(path)
        if not p.is_file():
            return cls(None)
        return cls(json.loads(p.read_text(encoding="utf-8")))

    @classmethod
    def for_chain(cls, chain: str | None) -> AccuracyTable:
        return cls.load(default_table_path(chain)) if chain else cls(None)

    @property
    def measured(self) -> bool:
        return bool(self._curves)

    def cf_range(self) -> tuple[int, int] | None:
        """(min, max) CF the table actually MEASURED, or None when there is no table.

        Anything outside this is the analytic fallback — an extrapolation — and a caller
        that emits a CF outside this range should say so rather than let it look measured.
        """
        if not self._curves:
            return None
        cfs = sorted({k[0] for k in self._curves})
        return cfs[0], cfs[-1]

    def periods(self) -> list[int]:
        return sorted({k[1] for k in self._curves})

    def routes_for(self, period: int) -> list[int]:
        return sorted({k[2] for k in self._curves if k[1] == period})

    # -- lookup ---------------------------------------------------------------------
    def _interp(self, key: tuple[int, int, int], mag: float) -> float | None:
        curve = self._curves.get(key)
        if not curve:
            return None
        amps, errs = curve
        if mag < amps[0] or mag > amps[-1]:
            return None                      # outside the grid: caller falls back
        i = bisect.bisect_left(amps, mag)
        if i == 0 or amps[i] == mag:
            return errs[i]
        x0, x1 = amps[i - 1], amps[i]
        y0, y1 = errs[i - 1], errs[i]
        if min(x0, x1, y0, y1) <= 0.0:
            t = (mag - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
        t = (math.log(mag) - math.log(x0)) / (math.log(x1) - math.log(x0))
        return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))

    def rel_err(self, mag: float, cf: int, period: int, route: int = 0) -> float:
        """Predicted relative error of one bootstrap. `period` 0 = dense/aperiodic."""
        if mag <= 0.0:
            return float("inf")
        for p in self._period_candidates(period):
            v = self._interp((int(cf), p, int(route)), mag)
            if v is not None:
                return v
            if route:                        # unmeasured route: fall back to dense, which
                v = self._interp((int(cf), p, 0), mag)   # is the CONSERVATIVE direction —
                if v is not None:            # a sparse route only ever lowers the floor.
                    return v
        return analytic_rel_err(mag, cf, route, period)

    @staticmethod
    def _period_candidates(period: int) -> list[int]:
        """Periods to try, most specific first.

        An unmeasured period falls back to **1**, not to dense: period 1 concentrates the
        whole plaintext into one coefficient and is the worst case, so an unknown or
        unmeasured shape is priced pessimistically rather than optimistically.
        """
        p = int(period)
        return [p, 1] if p != 1 else [1]

    def offset_gain(self, cf: int, dc: float) -> float | None:
        """Measured `as_is / offset` at this DC, or None if unmeasured."""
        rows = self._offset.get(int(cf))
        if not rows:
            return None
        dcs = [r[0] for r in rows]
        mag = abs(dc)
        if mag < dcs[0] or mag > dcs[-1]:
            return None
        i = min(range(len(dcs)), key=lambda j: abs(math.log(max(dcs[j], 1e-300)) -
                                                   math.log(max(mag, 1e-300))))
        _, as_is, off = rows[i]
        return (as_is / off) if off > 0 else None


# ── the per-site decision ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SiteChoice:
    """What the placer will emit for one bootstrap site, and what it predicts."""
    feasible: bool
    rel_err: float
    cf: int | None = None
    route: int = 0                  # sparse slot count; 0 = dense
    prescale: float | None = None
    offset: float | None = None
    extra_levels: int = 0           # the prescale restore; offset and CF are level-free
    reason: str = ""

    # A site whose best predicted error exceeds `err_hopeless`: the refresh is not merely
    # lossy, it is destructive (past the first zero of `sin` the value comes back
    # SIGN-INVERTED). These are the only ones the cut refuses outright.
    hopeless: bool = False

    def quality(self, target: float) -> float:
        """0 (at the band floor) .. 1 (at the error target), for the min-cut tie-break."""
        if self.rel_err <= 0.0:
            return 0.0
        if not math.isfinite(self.rel_err):
            return 1.0
        return min(1.0, max(0.0, self.rel_err / max(target, 1e-300)))

    def overshoot(self, target: float) -> float:
        """How far above the target, in decades, 0 when the site meets it."""
        if self.feasible or not math.isfinite(self.rel_err) or self.rel_err <= 0:
            return 0.0
        return max(0.0, math.log10(self.rel_err / max(target, 1e-300)))


@dataclass(frozen=True)
class SitePolicy:
    """Placer policy. Defaults reproduce nothing on their own — the caller opts in."""
    cf_min: int = DEFAULT_CF_MIN
    cf_max: int = DEFAULT_CF_MAX
    err_target: float = 1.0e-2
    prescale_reach: float = DEFAULT_PRESCALE_REACH
    allow_prescale: bool = False
    allow_offset: bool = False
    allow_sparse: bool = True
    # κ: captured maxima are one trajectory's sample; κ is the drift headroom applied to
    # the magnitude every decision is taken on (CF, route, prescale). 2.0 matches upstream
    # Orion's own margin, so the baseline comparison is margin-matched.
    mag_safety: float = 2.0
    # ── the two-tier refusal ─────────────────────────────────────────────────────────
    # A chain MUST be refreshed somewhere: if every candidate on it misses the target, a
    # hard refusal does not produce a safer plan, it produces NO plan. So only genuinely
    # DESTRUCTIVE sites are refused outright, and merely-lossy ones are priced.
    #   err_hopeless — above this the sine has bent past its first zero and the refresh
    #                  returns a sign-inverted value (rel_err ~1.0 is the saturation
    #                  plateau; 0.5 is inside the destroyed region).
    #   miss_penalty — extra cut capacity, in BOOTSTRAPS, charged per decade above the
    #                  target. At 4.0 the cut will happily spend four extra refreshes to
    #                  route around one decade-over site.  Unlike MAG_TIEBREAK_LAMBDA
    #                  this is NOT a tie-break: it trades count for accuracy, on purpose.
    err_hopeless: float = 0.5
    miss_penalty: float = 4.0
    # ── the prescale restore is not free ─────────────────────────────────────────────
    # A prescale by f shrinks the message to f·m for EvalMod and the restore multiplies by
    # 1/f afterwards, amplifying the ciphertext's own absolute noise by 1/f. Every CF lands
    # on its own band centre with ~2e-3 error, but the depth 1/f = |m| / (0.01 · 2^CF) it
    # demands varies 16x across CF 2..9. Measured round-trips are exact up to 9.1 bits and
    # destroyed at 10.9; 9.5 sits inside that bracket. Scale-invariance holds for the EvalMod
    # bend and fails for the ciphertext's own noise, so `rel_err(f·m)` alone is not enough.
    prescale_bits_max: float = 9.5
    # Preference among prescale candidates that all clear the budget: shallower is safer,
    # and since every CF lands on its own centre with equal predicted error, without this
    # the choice is arbitrary. In units of the error target, so it can never outrank a
    # genuine accuracy difference — it only breaks the ties the table cannot.
    prescale_depth_weight: float = 0.05


def choose_site(
    table: AccuracyTable,
    policy: SitePolicy,
    *,
    slot_mag: float | None,
    period: int | None = None,
    coeff_mag: float | None = None,
    coeff_mag_ac: float | None = None,
    mean: float | None = None,
    residual_mag: float | None = None,
    routes_avail: Sequence[int] = (),
    levels_free: int = 0,
) -> SiteChoice:
    """Pick `(offset, CF, sparse route, prescale)` for one bootstrap site.

    The order is the priority order: the offset removes the DC that puts a
    site out of range at all, CF then positions the band on what is left, sparse lowers the
    floor, and prescale is the last resort because it is the only one that costs a level.

    `slot_mag`  captured `output_max_abs`; `None` ⇒ infeasible (a magnitude-free node
                cannot be judged).
    `period`    captured `pack_period`; `None`/full-slots ⇒ `query_period` (period 1);
                routing stays dense either way (see the module note).
    `coeff_mag` captured `output_max_coeff`, when the capture carries it. Exact, and it
                supersedes the period heuristic (the module note: NO period heuristic
                serves both the ≈constant and the spread vectors).
    `coeff_mag_ac` captured `output_max_coeff_ac` — the coefficient max with the DC
                excluded: what the offset arm bootstraps. Used only with `coeff_mag`.
    `mean`/`residual_mag`  captured `output_mean` / `output_max_dev`, for the offset arm.
    `routes_avail`  the runtime's BUILT sparse precomps, already filtered to routes this
                site's tag proves legal. Never guessed here.
    `levels_free`  spare levels at the bootstrap output; a prescale needs one.
    """
    # A node with no captured magnitude cannot be judged, so it must be UNCUTTABLE — which
    # means `hopeless`, not merely "misses the target". Returning a non-hopeless choice here
    # would make every magnitude-free node placeable.
    if slot_mag is None or not (slot_mag > 0.0) or not math.isfinite(slot_mag):
        return SiteChoice(False, float("inf"), hopeless=True,
                          reason="no captured magnitude")

    kappa = max(policy.mag_safety, 1.0)
    # Query key. A captured coefficient is exact, and for period-1 data slot == coefficient,
    # so the period-1 column IS the coefficient-indexed law.
    use_coeff = coeff_mag is not None and coeff_mag > 0.0 and math.isfinite(coeff_mag)
    if use_coeff:
        q_mag, q_period = coeff_mag * kappa, 1
        # The site's error is judged RELATIVE TO THE SLOT SIGNAL, but rel_err(coeff) is
        # relative to the coefficient — for spread data (coeff << slot) that overstates
        # the site error by slot/coeff (~1000x at the refine iterates), which read as
        # "miss target" and made the min-cut route around perfectly good sites (the G17
        # 506-vs-418 inflation). Same physics as the offset arm's rescale below. The
        # factor is CF- and route-constant, so the CF/route argmin is unchanged by it.
        q_rescale = min(1.0, coeff_mag / slot_mag)
    else:
        q_mag = slot_mag * kappa
        q_period = query_period(period)
        q_rescale = 1.0

    routes = [0]
    if policy.allow_sparse:
        routes += [int(s) for s in routes_avail if int(s) > 0]

    best: SiteChoice | None = None

    best_score = float("inf")

    def consider(cand: SiteChoice, score: float | None = None) -> None:
        """Rank by `score`, which defaults to the predicted error.

        The two differ only for prescale candidates, where the table cannot separate the
        CFs (each lands on its own band centre at ~2e-3) but the restore depth can.
        """
        nonlocal best, best_score
        s = cand.rel_err if score is None else score
        if best is None or s < best_score:
            best, best_score = cand, s

    # ── arm 1+2+3: offset (optional) then CF then route, all level-free ──────────────
    # Each arm is (offset DC, magnitude the bootstrap actually sees, period, rescale).
    #
    #  `rescale` is what makes the offset arm comparable to the others, and getting it
    # wrong silently buries the single best accuracy lever. `table.rel_err(r)` is an error
    # RELATIVE TO r — but after the DC is added back, the site still carries ~slot_mag, so
    # the site's relative error is `rel_err(r)·r/slot_mag`, not `rel_err(r)`. Without this
    # a DC-dominated site looks WORSE offset than not, and the measurement says the exact
    # opposite: even with CF free to pick its best value per site, the offset still wins
    # 26x at DC=1, 150x at DC=5 and 2467x at DC=16 on the n32 sweep.
    offset_arms: list[tuple[float | None, float, int, float]] = [
        (None, q_mag, q_period, q_rescale)]
    if policy.allow_offset and mean is not None and residual_mag is not None:
        # Apply only where it pays: below the band centre the transform is neutral to
        # slightly negative (0.8-1.0x below DC=0.05).
        if abs(mean) > band_centre(policy.cf_min) and residual_mag > 0.0:
            # Subtracting a plaintext scalar preserves the data's period, so the query
            # period is unchanged; only the magnitude drops to the residual. With a
            # captured coefficient the residual's load is the DC-excluded coefficient max
            # (the scalar removes exactly the X^0 term), priced on the same period-1 law.
            if use_coeff and coeff_mag_ac is not None and coeff_mag_ac > 0.0:
                r = coeff_mag_ac * kappa
            else:
                r = residual_mag * kappa
            # Rescale against the SLOT signal (what the site carries after the DC is
            # added back), not against q_mag — with coeff pricing q_mag is the
            # coefficient, and r/q_mag would compare the two arms on different bases.
            offset_arms.append((float(mean), r, q_period, r / (slot_mag * kappa)))

    for off, mag, per, rescale in offset_arms:
        for cf in range(policy.cf_min, policy.cf_max + 1):
            for route in routes:
                e = table.rel_err(mag, cf, per, route) * rescale
                consider(SiteChoice(e <= policy.err_target, e, cf=cf, route=route,
                                    offset=off, extra_levels=0,
                                    reason="offset+cf+route" if off is not None
                                           else "cf+route"))

    if best is not None and best.feasible:
        return best

    # ── arm 4: prescale, the last resort — it is the only knob that costs a level ─────
    n_over_budget = 0
    if policy.allow_prescale and levels_free >= 1:
        for off, mag, per, rescale in offset_arms:
            for cf in range(policy.cf_min, policy.cf_max + 1):
                lo, hi = band(cf)
                if mag > policy.prescale_reach * hi:
                    continue                 # past the reach: refuse, do not stretch
                f = band_centre(cf) / mag
                if f >= 1.0:
                    continue                 # already at or below the centre
                # THE RESTORE BUDGET. log2(1/f) bits of the chain's precision are spent
                # amplifying noise back up; past the measured cliff the refresh returns
                # garbage however good its EvalMod position is.
                depth_bits = math.log2(1.0 / f)
                if depth_bits > policy.prescale_bits_max:
                    n_over_budget += 1
                    continue
                for route in routes:
                    e = table.rel_err(mag * f, cf, per, route) * rescale
                    # Rank by predicted error PLUS a depth preference. Every CF lands on
                    # its own band centre with near-identical measured error, so without
                    # this term the winner is arbitrary — and the arbitrary winner was the
                    # DEEPEST prescale, which is the one that detonates.
                    score = e + (policy.prescale_depth_weight * policy.err_target
                                 * depth_bits / max(policy.prescale_bits_max, 1e-9))
                    consider(SiteChoice(e <= policy.err_target, e, cf=cf, route=route,
                                        prescale=f, offset=off, extra_levels=1,
                                        reason=f"prescale+cf+route ({depth_bits:.1f} bits)"),
                             score=score)

    if best is None:
        why = "no admissible (cf, route)"
        if policy.allow_prescale and levels_free >= 1 and n_over_budget:
            why = (f"every prescale needs more than {policy.prescale_bits_max:g} bits of "
                   f"restore amplification ({n_over_budget} candidate(s) refused) — past "
                   f"the measured cliff the refresh returns garbage")
        return SiteChoice(False, float("inf"), hopeless=True, reason=why)
    if not best.feasible:
        hopeless = not (best.rel_err < policy.err_hopeless)
        best = SiteChoice(False, best.rel_err, cf=best.cf, route=best.route,
                          prescale=best.prescale, offset=best.offset,
                          extra_levels=best.extra_levels, hopeless=hopeless,
                          reason=("DESTRUCTIVE: best predicted rel_err "
                                  f"{best.rel_err:.3g} >= err_hopeless "
                                  f"{policy.err_hopeless:.3g} (the sine has bent past its "
                                  "first zero; the refresh returns a sign-inverted value)"
                                  if hopeless else
                                  f"lossy: best predicted rel_err {best.rel_err:.3g} "
                                  f"> target {policy.err_target:.3g}"))
    return best


def ceiling_binds(table: AccuracyTable, policy: SitePolicy, cf: int | None, **site) -> bool:
    """Is `cf` (what `choose_site(table, policy, **site)` returned) at `policy.cf_max`
    BECAUSE the search ran out of range?

    `choose_site` is an exhaustive argmin over `cf_min..cf_max`; the ceiling is only the
    loop bound, so `cf == cf_max` cannot say whether cf_max was cheapest or merely last.

    CLAMPED := `cf == policy.cf_max` AND the identical search with `cf_max + 1` returns
    `cf_max + 1` — the ceiling binds at one step. This is a deterministic lower bound on
    ceiling-bound sites: the CF objective is not unimodal (e.g. cf 13 < cf 14 > cf 15 at
    |m| ~ 8-9), so a site whose global argmin lies past
    the ceiling but that was NOT pinned at cf_max is deliberately outside the count, and a
    pinned site that only improves two steps up is reported as pinned, not clamped. When
    `cf_max` equals the table's measured maximum (`table.cf_range()[1]`) the probe compares
    a measured cf_max against an ANALYTIC cf_max + 1. Pure and side-effect free; the wider
    choice is never used for planning, so the plan itself is unchanged by asking.
    """
    if cf is None or cf != policy.cf_max:
        return False
    wider = replace(policy, cf_max=policy.cf_max + 1)
    return choose_site(table, wider, **site).cf == wider.cf_max


# ── self-check: does the closed form agree with what was measured? ───────────────────

def check(table: AccuracyTable, verbose: bool = True) -> dict:
    """Replay `analytic_rel_err` against every measured row; report the deviation.

    This is what makes the planner runnable with no table present: if the closed form
    tracks the measurement, an unmeasured chain or an out-of-grid amplitude is an
    extrapolation rather than a guess.
    """
    rows: list[tuple[float, dict]] = []
    for (cf, period, route), (amps, errs) in sorted(table._curves.items()):
        for a, e in zip(amps, errs):
            if e <= 0.0:
                continue
            pred = analytic_rel_err(a, cf, route, period)
            ratio = pred / e if e > 0 else float("inf")
            rows.append((abs(math.log10(ratio)) if ratio > 0 else float("inf"),
                         {"cf": cf, "period": period, "route": route, "amp": a,
                          "measured": e, "analytic": pred, "ratio": ratio}))
    if not rows:
        return {"n": 0}
    rows.sort(key=lambda t: -t[0])
    decades = [r[0] for r in rows]
    decades.sort()
    out = {
        "n": len(rows),
        "median_abs_log10_ratio": decades[len(decades) // 2],
        "p90_abs_log10_ratio": decades[int(0.9 * len(decades))],
        "worst": [r[1] for r in rows[:10]],
    }
    if verbose:
        log.info(f"[btserr] {out['n']} measured rows; analytic-vs-measured |log10 ratio| "
              f"median={out['median_abs_log10_ratio']:.3f} "
              f"p90={out['p90_abs_log10_ratio']:.3f}")
        log.info("[btserr] worst 10 (the closed form is a FALLBACK; the table is the truth):")
        for w in out["worst"]:
            log.info(f"  cf={w['cf']} period={w['period']:>5} route={w['route']:>3} "
                  f"A={w['amp']:<9.4g} measured={w['measured']:.3e} "
                  f"analytic={w['analytic']:.3e} ratio={w['ratio']:.3g}")
    return out


def _main(argv: Iterable[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m perseus.plan.btserr",
                                 description="inspect / validate the accuracy table")
    ap.add_argument("--chain", default="n32")
    ap.add_argument("--table", help="explicit path (default: data/bts_accuracy_<chain>.json)")
    ap.add_argument("--check", action="store_true", help="analytic-vs-measured deviation")
    ap.add_argument("--band", type=float, metavar="CF", help="print the band for a CF")
    ap.add_argument("--site", type=float, metavar="MAG", help="choose for one magnitude")
    ap.add_argument("--period", type=int, default=1)
    ap.add_argument("--routes", default="", help="comma list of built precomps")
    args = ap.parse_args(list(argv) if argv is not None else None)

    table = AccuracyTable.load(args.table) if args.table else AccuracyTable.for_chain(args.chain)
    log.info(f"[btserr] table measured={table.measured} meta={table.meta}")
    if table.measured:
        log.info(f"[btserr] periods={table.periods()} "
              f"routes(period=1)={table.routes_for(1)}")
    if args.band is not None:
        lo, hi = band(args.band)
        log.info(f"[btserr] CF={args.band:g}: band [{lo:.4g}, {hi:.4g}] "
              f"centre {band_centre(args.band):.4g} "
              f"prescale reach {DEFAULT_PRESCALE_REACH * hi:.4g}")
    if args.site is not None:
        routes = tuple(int(s) for s in args.routes.split(",") if s)
        pol = SitePolicy(allow_prescale=True, allow_offset=False)
        c = choose_site(table, pol, slot_mag=args.site, period=args.period,
                        routes_avail=routes, levels_free=1)
        log.info(f"[btserr] |m|={args.site:g} period={args.period}: {c}")
    if args.check:
        check(table)


if __name__ == "__main__":
    _main()
