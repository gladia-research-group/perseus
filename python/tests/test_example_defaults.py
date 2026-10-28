"""Every default plan of the example drivers resolves to a plan dir with per-block placements."""
import os

from perseus._paths import resolve
from perseus.examples.defaults import PLANS, WINDOWS, default_plan, default_windows


def test_default_plans_exist():
    plans = [(model, t, plan) for model, table in PLANS.items() for t, plan in table.items()]
    plans += [(model, b, plan) for model, windows in WINDOWS.items() for b, plan in windows]
    for model, t, plan in plans:
        d = resolve(plan)
        assert os.path.isdir(d), (model, t, plan)
        assert any(f.startswith("block_") for f in os.listdir(d)), (model, t, plan)


def test_shape_lookup():
    assert default_plan("gpt2_prefill", 128).endswith("bootstrap_placements/gpt2_prefill128")
    assert default_plan("gpt2_prefill", 64) is None
    assert default_plan("gpt2_decode", 16).endswith("bootstrap_placements/gpt2_decode")
    assert default_plan("nope") is None


def test_window_lookup():
    assert [b for b, _d in default_windows("gpt2_decode", 512)] == [128, 256, 384, 512]
    assert default_windows("gpt2_decode", 513) is None
    assert default_windows("llama_decode")[-1][1].endswith("bootstrap_placements/llama31_8b_decode_ctx512_w512")
    assert default_windows("bert") is None
