"""Unit tests for the bootstrap accuracy model (perseus/plan/btserr.py).

These are the properties a CF/band-aware placer must not silently lose.

  python -m pytest tests/test_btserr.py -q        (or: python tests/test_btserr.py)
"""
from __future__ import annotations

import math
import unittest

from perseus.plan import btserr

ANALYTIC = btserr.AccuracyTable(None)      # no table: the closed form alone
MEASURED = btserr.AccuracyTable.for_chain("n32")


class BandTest(unittest.TestCase):
    def test_band_slides_by_two_per_cf_step(self):
        """CF does not buy accuracy, it slides a fixed-quality band."""
        for cf in range(2, 9):
            lo0, hi0 = btserr.band(cf)
            lo1, hi1 = btserr.band(cf + 1)
            self.assertAlmostEqual(lo1 / lo0, 2.0, places=9)
            self.assertAlmostEqual(hi1 / hi0, 2.0, places=9)

    def test_band_matches_the_documented_numbers(self):
        lo, hi = btserr.band(6)
        self.assertAlmostEqual(lo, 0.192, places=6)
        self.assertAlmostEqual(hi, 1.92, places=6)

    def test_cf_for_magnitude_centres_the_band(self):
        for mag in (0.05, 0.5, 3.489, 40.0):
            cf = btserr.cf_for_magnitude(mag)
            self.assertAlmostEqual(btserr.band_centre(cf), mag, places=6)


class BendTest(unittest.TestCase):
    def test_bend_reproduces_the_measured_sine_law(self):
        """`got = C·sin(m/C)`, C = 2^CF/2π: `bend` returns the relative error
        |got - m|/m = |0.496 - 3.489|/3.489 = 0.858 at A=3.48909/CF=3, and the measured
        table reads 0.827 at that cell, so the one-parameter law is good to ~4%."""
        self.assertAlmostEqual(btserr.bend(3.48909, 3), 0.858, places=2)
        # nearest measured grid points, same cell shape
        self.assertAlmostEqual(MEASURED.rel_err(3.0, 3, period=1), 0.671, places=2)
        self.assertAlmostEqual(btserr.bend(3.0, 3), 0.700, places=2)

    def test_bend_is_negligible_inside_the_band(self):
        for cf in range(2, 10):
            self.assertLess(btserr.bend(btserr.band_centre(cf), cf), 1e-2)

    def test_bend_saturates_far_above_the_band(self):
        # Past the first zero of sin the value comes back sign-inverted; that is the
        # `hopeless` region, and it must read as ~1 rather than growing without bound.
        self.assertGreater(btserr.bend(1e4, 6), 0.5)


class RhoTest(unittest.TestCase):
    def test_period_one_is_the_worst_case(self):
        """A broadcast constant puts all its energy in one coefficient."""
        self.assertEqual(btserr.rho(1), 1.0)
        for p in (2, 32, 512, 32768):
            self.assertLess(btserr.rho(p), 1.0)

    def test_rho_matches_the_measured_sweep(self):
        # Measured on the n32 grid by sliding each period's bend branch onto period 1:
        # p=512 -> 0.0440 (1/sqrt(512) = 0.04419).
        self.assertAlmostEqual(btserr.rho(512), 0.0440, places=3)

    def test_unknown_period_is_treated_as_aperiodic_for_rho(self):
        self.assertEqual(btserr.rho(None, slots=32768), btserr.rho(0, slots=32768))


class TableTest(unittest.TestCase):
    def test_measured_table_is_present_and_covers_the_cf_range(self):
        self.assertTrue(MEASURED.measured, "perseus/plan/data/bts_accuracy_n32.json missing")
        self.assertEqual(MEASURED.meta["cf_min"], 2)
        self.assertGreaterEqual(MEASURED.meta["cf_max"], 14)
        self.assertIn(1, MEASURED.periods())
        self.assertIn(512, MEASURED.periods())

    def test_table_carries_its_provenance(self):
        """A table is only valid for the binary that produced it — plans are binary-bound."""
        for key in ("repo_sha", "fideslib_sha"):
            self.assertTrue(MEASURED.meta.get(key), f"table has no {key}")

    def test_unknown_period_falls_back_to_period_one_not_to_dense(self):
        """The pessimistic direction. Dense would UNDER-estimate the error."""
        mag, cf = 8.0, 6
        self.assertAlmostEqual(MEASURED.rel_err(mag, cf, period=999_999),
                               MEASURED.rel_err(mag, cf, period=1), places=12)
        self.assertGreater(MEASURED.rel_err(mag, cf, period=1),
                           MEASURED.rel_err(mag, cf, period=0))

    def test_analytic_tracks_the_measurement_at_low_cf(self):
        """The closed form is a fallback that tracks the table only at low CF: its
        deviation grows monotonically with CF and becomes optimistic at high CF, so this
        pins the range the fallback may be trusted over, not a global average."""
        rows = [(cf, p, r, a, e) for (cf, p, r), (amps, errs) in MEASURED._curves.items()
                for a, e in zip(amps, errs) if e > 0]
        self.assertGreater(len(rows), 500)
        for lo_cf, hi_cf, limit in ((2, 6, 0.10), (7, 9, 0.20)):
            dev = sorted(abs(math.log10(btserr.analytic_rel_err(a, cf, r, p) / e))
                         for cf, p, r, a, e in rows if lo_cf <= cf <= hi_cf)
            self.assertLess(dev[len(dev) // 2], limit,
                            f"analytic median deviation too large for cf {lo_cf}..{hi_cf}")

    def test_measured_table_covers_the_full_cf_window(self):
        """The shipped n32 table covers CF 2..20 (the planner's window), and at a fixed
        CF the error still climbs with amplitude past the band top."""
        cfs = sorted({cf for (cf, _p, _r) in MEASURED._curves})
        self.assertEqual((cfs[0], cfs[-1]), (btserr.DEFAULT_CF_MIN, btserr.DEFAULT_CF_MAX))
        self.assertLess(MEASURED.rel_err(256.0, 14, period=1, route=0), 0.05)
        self.assertGreater(MEASURED.rel_err(4096.0, 14, period=1, route=0),
                           MEASURED.rel_err(256.0, 14, period=1, route=0))

    def test_sparse_route_only_ever_lowers_the_floor(self):
        """CF moves the band, the route lowers the floor — orthogonal."""
        # deep in the floor, where routing is the whole story
        self.assertLess(MEASURED.rel_err(0.001, 6, period=1, route=1),
                        MEASURED.rel_err(0.001, 6, period=1, route=0))
        # above the band, all routes agree to a few digits
        hi = MEASURED.rel_err(16.0, 6, period=1, route=0)
        for route in (1, 512):
            self.assertAlmostEqual(MEASURED.rel_err(16.0, 6, period=1, route=route) / hi,
                                   1.0, delta=0.05)


class ChooseSiteTest(unittest.TestCase):
    # cf_max pinned to 9: these exercise the prescale/CF machinery, and with the ceiling
    # at 15 several of these magnitudes simply fall inside a band and need no prescale.
    # mag_safety pinned to 1.0: these test the prescale/CF MECHANICS, and κ inflates the
    # query magnitude, which shifts every derived factor. κ has its own test below.
    POLICY = btserr.SitePolicy(allow_prescale=True, allow_sparse=True, cf_max=9,
                               mag_safety=1.0)

    def choose(self, mag, **kw):
        kw.setdefault("levels_free", 1)
        return btserr.choose_site(MEASURED, self.POLICY, slot_mag=mag, **kw)

    def test_no_magnitude_is_not_placeable(self):
        c = self.choose(None)
        self.assertFalse(c.feasible)
        self.assertTrue(c.hopeless)

    def test_cf_tracks_the_site_magnitude(self):
        """The point of the whole exercise: one CF per site, not one per run."""
        cf_small = self.choose(0.05, period=1).cf
        cf_large = self.choose(8.0, period=1).cf
        self.assertLess(cf_small, cf_large)

    def test_cf_stays_inside_the_legal_window(self):
        for mag in (1e-3, 1e-2, 0.3, 3.0, 30.0, 3000.0):
            c = self.choose(mag, period=1)
            if c.cf is not None:
                self.assertGreaterEqual(c.cf, self.POLICY.cf_min)
                self.assertLessEqual(c.cf, self.POLICY.cf_max)

    def test_in_band_sites_need_no_prescale(self):
        c = self.choose(btserr.band_centre(6), period=1)
        self.assertTrue(c.feasible)
        self.assertIsNone(c.prescale)
        self.assertEqual(c.extra_levels, 0)

    def test_prescale_is_a_last_resort_and_costs_a_level(self):
        c = self.choose(500.0, period=1)
        self.assertIsNotNone(c.prescale)
        self.assertEqual(c.extra_levels, 1)
        # it lands the site at its CF's band centre
        self.assertAlmostEqual(500.0 * c.prescale, btserr.band_centre(c.cf), places=6)

    def test_prescale_is_refused_without_a_spare_level(self):
        c = btserr.choose_site(MEASURED, self.POLICY, slot_mag=500.0, period=1,
                               levels_free=0)
        self.assertIsNone(c.prescale)

    def test_reach_bounds_how_far_a_prescale_may_go(self):
        pol = btserr.SitePolicy(allow_prescale=True, prescale_reach=10.0)
        far = 100.0 * 10.0 * btserr.band(pol.cf_max)[1]
        c = btserr.choose_site(MEASURED, pol, slot_mag=far, period=1, levels_free=1)
        self.assertIsNone(c.prescale)
        self.assertTrue(c.hopeless)

    def test_ceiling_binds_separates_clamped_from_merely_pinned(self):
        """cf == cf_max is a clamp only when cf_max + 1 would win the same search."""
        pol = btserr.SitePolicy(cf_max=14, allow_sparse=True, allow_prescale=False)
        # the shipping n32 arms' one pinned site (block_0 v_1156, |m|=9.11, period 1,
        # sparse s=1): 14 -> 1.1e-5, 15 -> 8.2e-6, so the ceiling binds
        site = dict(slot_mag=9.110810754329055, period=1, routes_avail=(1,), levels_free=0)
        c = btserr.choose_site(MEASURED, pol, **site)
        self.assertEqual(c.cf, 14)
        self.assertTrue(btserr.ceiling_binds(MEASURED, pol, c.cf, **site))
        # a site inside a band is not at the ceiling at all
        inside = dict(slot_mag=1.0, period=1, routes_avail=(), levels_free=0)
        c = btserr.choose_site(MEASURED, pol, **inside)
        self.assertLess(c.cf, pol.cf_max)
        self.assertFalse(btserr.ceiling_binds(MEASURED, pol, c.cf, **inside))
        self.assertFalse(btserr.ceiling_binds(MEASURED, pol, None, **inside))
        # pinned but NOT clamped: |m|=100 (x1.2) takes cf 14 at 1.9e-3 and cf 15 prices
        # ~3.4e-3, so raising the ceiling one step changes nothing
        dense = btserr.SitePolicy(cf_max=14, allow_sparse=False, allow_prescale=False)
        pinned = dict(slot_mag=100.0, period=1, routes_avail=(), levels_free=0)
        c = btserr.choose_site(MEASURED, dense, **pinned)
        self.assertEqual(c.cf, 14)
        self.assertFalse(btserr.ceiling_binds(MEASURED, dense, c.cf, **pinned))
        # asking never moves the choice itself
        self.assertEqual(btserr.choose_site(MEASURED, dense, **pinned), c)

    def test_destructive_and_lossy_are_different_verdicts(self):
        """A chain must be refreshed SOMEWHERE: only destructive sites are refused."""
        pol = btserr.SitePolicy(allow_prescale=False, allow_sparse=False,
                                err_target=1e-4, err_hopeless=0.5)
        lossy = btserr.choose_site(MEASURED, pol, slot_mag=1.0, period=1)
        self.assertFalse(lossy.feasible)
        self.assertFalse(lossy.hopeless)          # placeable, just priced
        dead = btserr.choose_site(MEASURED, pol, slot_mag=1e6, period=1)
        self.assertTrue(dead.hopeless)            # refused outright

    def test_overshoot_is_zero_when_the_target_is_met(self):
        c = self.choose(btserr.band_centre(6), period=1)
        self.assertEqual(c.overshoot(self.POLICY.err_target), 0.0)
        bad = btserr.SiteChoice(False, 1e-1)
        self.assertAlmostEqual(bad.overshoot(1e-2), 1.0, places=9)

    def test_sparse_route_is_only_offered_when_it_is_legal(self):
        """Routing must come from the tag, never be guessed — an unproven route corrupts."""
        c = self.choose(0.001, period=1, routes_avail=())
        self.assertEqual(c.route, 0)
        c = self.choose(0.001, period=1, routes_avail=(1,))
        self.assertEqual(c.route, 1)

    def test_offset_needs_both_the_mean_and_the_residual(self):
        pol = btserr.SitePolicy(allow_offset=True, allow_prescale=True)
        no_stats = btserr.choose_site(MEASURED, pol, slot_mag=8.0, period=1, levels_free=1)
        self.assertIsNone(no_stats.offset)
        with_stats = btserr.choose_site(MEASURED, pol, slot_mag=8.0, period=1,
                                        mean=8.0, residual_mag=0.08, levels_free=1)
        self.assertIsNotNone(with_stats.offset)
        # after offsetting, the band follows the residual, so CF comes down
        self.assertLess(with_stats.cf, no_stats.cf)
        self.assertLess(with_stats.rel_err, no_stats.rel_err)

    def test_offset_is_skipped_when_the_dc_is_already_small(self):
        pol = btserr.SitePolicy(allow_offset=True)
        c = btserr.choose_site(MEASURED, pol, slot_mag=0.02, period=1,
                               mean=1e-6, residual_mag=0.02)
        self.assertIsNone(c.offset)

    def test_mag_safety_never_leaves_a_site_less_protected(self):
        """κ models drift: at κx drift the site must still be inside the band. Not
        asserted: that κ raises CF or worsens the predicted error — a κ-triggered
        prescale can land the site dead-centre with a better prediction."""
        base = btserr.SitePolicy(allow_prescale=True, mag_safety=1.0)
        kappa = btserr.SitePolicy(allow_prescale=True, mag_safety=8.0)
        for mag in (0.5, 4.0, 40.0, 400.0):
            a = btserr.choose_site(MEASURED, base, slot_mag=mag, period=1, levels_free=1)
            b = btserr.choose_site(MEASURED, kappa, slot_mag=mag, period=1, levels_free=1)
            fb = b.prescale if b.prescale is not None else 1.0
            # THE invariant: a runtime value κx the captured one still lands in band.
            if b.cf is not None:
                self.assertLessEqual(mag * kappa.mag_safety * fb,
                                     btserr.band(b.cf)[1] * 1.001,
                                     f"κ-inflated |m| above the band top at |m|={mag}")
            # sanity: κ=1 protects only the captured value, which is the weaker promise
            self.assertIsNotNone(a)


class PrescaleRestoreTest(unittest.TestCase):
    """The prescale restore is not free: a |m|=610.77 site prescaled to CF=5's band
    centre needs 10.9 bits of restore amplification and the refresh is destroyed
    (0.343x its input); the same site at cf=9 (6.9 bits) round-trips."""
    POL = btserr.SitePolicy(allow_prescale=True, err_target=1e-2, prescale_reach=2000.0,
                            cf_max=9, mag_safety=1.0)

    def test_the_deep_site_does_not_pick_the_deepest_prescale(self):
        c = btserr.choose_site(MEASURED, self.POL, slot_mag=610.7723, period=None,
                               levels_free=1)
        self.assertIsNotNone(c.prescale)
        depth = math.log2(1.0 / c.prescale)
        self.assertLessEqual(depth, self.POL.prescale_bits_max)
        # the shallow-prescale preference must beat a plain cf=5 choice
        self.assertGreater(c.cf, 5)

    def test_depth_budget_is_enforced(self):
        # a magnitude far enough out that EVERY cf needs more than the budget
        deep = 0.01 * 2 ** 9 * 2 ** (self.POL.prescale_bits_max + 2)
        c = btserr.choose_site(MEASURED, self.POL, slot_mag=deep, period=1, levels_free=1)
        if c.prescale is not None:
            self.assertLessEqual(math.log2(1.0 / c.prescale),
                                 self.POL.prescale_bits_max)

    def test_budget_does_not_refuse_depths_measured_to_work(self):
        """Depths of 8.3 / 8.9 / 9.1 bits are measured to round-trip exactly."""
        self.assertGreaterEqual(self.POL.prescale_bits_max, 9.1)

    def test_shallowest_prescale_wins_among_equal_error(self):
        """Every cf lands on its own band centre at ~2e-3, so without the depth term the
        winner would be arbitrary."""
        mag = 610.7723
        c = btserr.choose_site(MEASURED, self.POL, slot_mag=mag, period=1, levels_free=1)
        depth = math.log2(1.0 / c.prescale)
        for cf in range(self.POL.cf_min, self.POL.cf_max + 1):
            f = btserr.band_centre(cf) / mag
            if f < 1.0 and math.log2(1.0 / f) <= self.POL.prescale_bits_max:
                self.assertLessEqual(depth, math.log2(1.0 / f) + 1e-9)


class MagSafetyDefaultTest(unittest.TestCase):
    def test_kappa_defaults_to_headroom_not_none(self):
        """κ=1.0 means every decision rides on a point estimate with no margin."""
        self.assertGreater(btserr.SitePolicy().mag_safety, 1.0)

    def test_kappa_shrinks_the_prescale_it_derives(self):
        mag = 500.0
        a = btserr.choose_site(MEASURED, btserr.SitePolicy(allow_prescale=True, cf_max=9,
                                                           mag_safety=1.0),
                               slot_mag=mag, period=1, levels_free=1)
        b = btserr.choose_site(MEASURED, btserr.SitePolicy(allow_prescale=True, cf_max=9,
                                                           mag_safety=1.2),
                               slot_mag=mag, period=1, levels_free=1)
        self.assertLess(b.prescale, a.prescale)   # κ shrinks harder => more headroom


class CfCeilingTest(unittest.TestCase):
    """9 is OpenFHE's AUTO clamp, not a limit; raising the ceiling removes deep prescales."""

    def test_ceiling_default_is_above_the_openfhe_auto_clamp(self):
        self.assertGreater(btserr.DEFAULT_CF_MAX, 9)

    def test_raising_the_ceiling_removes_the_deep_prescale(self):
        mag = 610.7723            # the block-1 residual that detonated the arm
        at9 = btserr.choose_site(MEASURED, btserr.SitePolicy(allow_prescale=True, cf_max=9),
                                 slot_mag=mag, period=None, levels_free=1)
        at15 = btserr.choose_site(MEASURED, btserr.SitePolicy(allow_prescale=True, cf_max=15),
                                  slot_mag=mag, period=None, levels_free=1)
        self.assertIsNotNone(at9.prescale)       # at cf<=9 it must prescale
        depth9 = math.log2(1.0 / at9.prescale)
        if at15.prescale is None:
            self.assertGreater(at15.cf, 9)       # it fits a band instead
        else:
            self.assertLess(math.log2(1.0 / at15.prescale), depth9)

    def test_table_reports_what_it_actually_measured(self):
        """So a CF outside the measured range can never look measured."""
        rng = MEASURED.cf_range()
        self.assertIsNotNone(rng)
        self.assertEqual(rng[0], 2)
        self.assertGreaterEqual(rng[1], 9)
        self.assertIsNone(ANALYTIC.cf_range())


class AnalyticFallbackTest(unittest.TestCase):
    def test_planner_works_with_no_table_at_all(self):
        """A missing table must degrade to the closed form, never to a crash."""
        self.assertFalse(ANALYTIC.measured)
        pol = btserr.SitePolicy(allow_prescale=True)
        c = btserr.choose_site(ANALYTIC, pol, slot_mag=3.489, period=1, levels_free=1)
        self.assertIsNotNone(c.cf)
        self.assertTrue(math.isfinite(c.rel_err))


if __name__ == "__main__":
    unittest.main(verbosity=2)
