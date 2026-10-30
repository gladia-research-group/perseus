"""PLAN_STC_FIRST_STAGED_ONLY: with the StC-first order on the staged finishes only, no refresh site of the plan takes
the slim (slot) law of PLAN_SLIM_CUBIC; its CF is the shipped order's choice."""
from perseus import _pc
from perseus.plan import btserr
from perseus.plan.placer.ir import Graph
from perseus.plan.placer.refresh import RefreshPlanner
from perseus.plan.placer.sim import PlannerShadow


def _planner(env, bl):
    # a spread value: slot max 20, coefficient max 0.03 (the shipped order's EvalMod sees the coefficients)
    g = Graph.from_nodes([{"op_type": "mult", "inputs": ["x", "x"], "input_levels": [None, None], "output": "a",
                           "output_level": None, "step": "blk0.mlp", "output_max_abs": 20.0,
                           "output_max_coeff": 0.03}], level_unit=2)
    sh = PlannerShadow(bl, 2, _pc.PlanConfig.from_legacy_env(env))
    return lambda slim: RefreshPlanner(table=btserr.AccuracyTable.for_chain("n32l"),
                                       policy=btserr.SitePolicy(slim_cubic=slim), graph=g, bootstrap_level=bl,
                                       shadow=sh).spec("a")


def test_staged_only_prices_the_shipped_order(monkeypatch):
    spec = _planner({}, 34)
    shipped, slim = spec(0.0), spec(1.0)
    assert slim.cf > shipped.cf + 4 and not slim.shipped          # the slot law holds 2 x 20 in the band
    monkeypatch.setenv("PLAN_STC_FIRST_STAGED_ONLY", "1")
    staged = spec(1.0)
    assert (staged.cf, staged.rel_err) == (shipped.cf, shipped.rel_err) and staged.shipped
    assert (staged.out_consumed, staged.out_deg) == (shipped.out_consumed, shipped.out_deg)   # no fallback named


def test_staged_only_lands_at_the_named_shipped_landing(monkeypatch):
    spec = _planner({"PLAN_STC_FIRST_ENTRY": "46", "PLAN_STC_FIRST_FALLBACK": "34:1"}, 28)
    assert spec(1.0).out_consumed == 0.0
    monkeypatch.setenv("PLAN_STC_FIRST_STAGED_ONLY", "1")
    s = spec(1.0)
    assert s.shipped and (s.out_consumed, s.out_deg) == (6.0, 1) and s.cf == spec(0.0).cf
