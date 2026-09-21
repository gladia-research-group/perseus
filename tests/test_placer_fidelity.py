"""Port-fidelity tests for the baseline placers (Orion / DaCapo / Fhelipe).

Ground truth, vendored in tests/fixtures/placer_fidelity/:
  * graphs/          nine synthetic capture graphs (chains, fork-join, skip,
                     fan-out, diamond) exported to the real upstream tools
  * upstream.json    the site sets the real upstream Orion / DaCapo / Fhelipe
                     chose on those graphs (provenance inside the file)

The tests re-run our ports on the same graphs with the same config and assert,
per placer:
  FIDELITY  — on every case the recorded comparison found exact, our port still
              equals the real upstream tool's site set (fhelipe 9/9, dacapo 8/9,
              orion 6/9; Orion's validation basis is its objective, so its three
              site-divergent shapes are pinned as known, not asserted exact).
  NO DRIFT  — on the known-divergent cases the divergence is exactly the recorded
              one (a change in either direction fails).

A failure here means a planner/sim change moved a baseline port off the validated
upstream semantics.
"""
import json
import unittest
from pathlib import Path
from unittest import mock

FIX = Path(__file__).resolve().parent / "fixtures" / "placer_fidelity"
GT = json.load(open(FIX / "upstream.json"))
CASES = sorted(k for k in GT if not k.startswith("_"))

CFG = dict(bootstrap_level=16, max_level=22, source_level=16, cache_read_level=16,
           level_unit=1, acc_chain="n32", allow_prescale=True, verbose=False)
ENTRY = dict(entry_level=16, entry_deg=2)


def _run(case: str, placer: str):
    from perseus.plan.placer.baselines.orion import OrionPlacer
    from perseus.plan.placer.place import PlanInfeasible
    from perseus.plan.placer.planner import PlanConfig, plan_block
    # Upstream Orion plans whole programs and has no hand-off exit cap: compare
    # against it with the cap disabled.
    with mock.patch.object(OrionPlacer, "EXIT_CAP_SLACK", 0):
        try:
            r = plan_block(FIX / "graphs" / case / "block_0" / "graph.json",
                           PlanConfig(placer=placer, baseline_rescue=False, **CFG), **ENTRY)
        except PlanInfeasible as e:
            return None, str(e)
    return sorted(p["target_var"] for p in r["placements"]), None


class SyntheticFidelity(unittest.TestCase):
    """Our ports vs the real upstream tools' recorded outputs, case by case."""

    def _check(self, placer):
        for case in CASES:
            up = GT[case]["upstream"][placer]
            era_exact = GT[case]["era_exact"][placer]
            era_ours = GT[case]["era_ours"][placer]
            ours, err = _run(case, placer)
            with self.subTest(case=case):
                self.assertIsNotNone(
                    ours, f"{placer} refused {case} ({err}); the era run placed "
                          f"{era_ours} — the port lost a case it could solve")
                if era_exact:
                    self.assertEqual(
                        set(ours), set(up),
                        f"{placer}/{case}: port no longer matches the REAL upstream "
                        f"tool (upstream {sorted(up)}, ours {sorted(ours)}). "
                        f"Fidelity to published semantics is broken — do not trust "
                        f"new baseline numbers until the full harness is re-run.")
                else:
                    self.assertEqual(
                        set(ours), set(era_ours),
                        f"{placer}/{case}: known-divergent case DRIFTED "
                        f"(era ours {sorted(era_ours)}, now {sorted(ours)}, "
                        f"upstream {sorted(up)}). Either direction of drift "
                        f"invalidates the recorded comparison.")

    def test_fhelipe_matches_upstream_9_of_9(self):
        self._check("fhelipe")

    def test_dacapo_matches_upstream_8_of_9(self):
        self._check("dacapo")

    def test_orion_6_of_9_exact_rest_pinned(self):
        self._check("orion")


if __name__ == "__main__":
    unittest.main()
