"""ImplModel plan variants: block_<b>_<v>_placement.json replaces block b's plan while
``plan_variant`` is v (block 0 for a fed-back token in generation)."""
import json

import pytest

from perseus.impl.driver import ImplModel


class _Core:
    def parse_bootstrap_plan_file(self, p):
        return ("plan", p.rsplit("/", 1)[-1])


def _model(tmp_path, names):
    for n in names:
        (tmp_path / n).write_text(json.dumps({"placements": []}))
    m = object.__new__(ImplModel)          # the plan bookkeeping needs no session
    m.core, m.plans, m.plan_variants, m.plan_variant = _Core(), None, {}, None
    return m.load_plans(str(tmp_path), range(3), validate=False, variants=("feedback",))


def test_variant_replaces_its_block_only_while_selected(tmp_path):
    m = _model(tmp_path, ["block_0_placement.json", "block_1_placement.json",
                          "block_0_feedback_placement.json"])
    assert m._plan_for(0) == ("plan", "block_0_placement.json")
    m.plan_variant = "feedback"
    assert m._plan_for(0) == ("plan", "block_0_feedback_placement.json")
    assert m._plan_for(1) == ("plan", "block_1_placement.json")   # no variant: the block's plan
    assert m._plan_for(2) is None                                  # no plan: eager
    m.plan_variant = None
    assert m._plan_for(0) == ("plan", "block_0_placement.json")


def test_unplanned_model_has_no_plans(tmp_path):
    m = object.__new__(ImplModel)
    m.plans, m.plan_variants, m.plan_variant = None, {}, "feedback"
    assert m._plan_for(0) is None


def test_generate_refuses_a_plan_without_the_feedback_variant(tmp_path):
    pytest.importorskip("examples.gpt2_from_primitives.model")
    from examples.gpt2_from_primitives.model import Gpt2Primitives
    m = _model(tmp_path, ["block_0_placement.json"])
    m.__class__ = Gpt2Primitives
    with pytest.raises(RuntimeError, match="block_0_feedback_placement.json"):
        m.generate([[0.0]], 1)
