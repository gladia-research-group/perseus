"""GPT-2 prefill, planned on the device: every block a compiled `Program` run in Rust
(`impl.prefill.PlannedPrefill`), the PCMM linears `Finish` ops served by the session's PCMM engine (keys made at
setup, weights resident), T=128 tokens to the first token: the last position's logits vs the PyTorch oracle's.

Gates (the full gate is `run_prefill --tokens 128 --tail --check`): the last position's top-1, median KL < 0.1,
every PCMM linear served (48 per forward), device free memory flat over two forwards.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

OLD = "/data02/users/mithrillo/projects/perseus/.cache"
WEIGHTS = os.environ.get("WEIGHTS_PATH") or f"{OLD}/models/openai-community/gpt2/classic/weights.bin.zip"
IO_DIR = os.environ.get("ALL_BLOCKS_IO_DIR") or f"{OLD}/oracle/gpt2/all_blocks_io"
PLAN = "bootstrap_placements/gpt2_prefill128"


@pytest.mark.skipif(not (os.path.exists(WEIGHTS) and os.path.isdir(IO_DIR)), reason="GPT-2 weights / oracle io missing")
def test_prefill_planned_t128(tmp_path, monkeypatch):
    from perseus.examples.gpt2 import run_prefill
    monkeypatch.setenv("PERSEUS_SLOTVEC_CACHE", "0")
    monkeypatch.setenv("WEIGHTS_PATH", WEIGHTS)
    monkeypatch.setenv("ALL_BLOCKS_IO_DIR", IO_DIR)
    monkeypatch.setenv("FAKECAP_LANDINGS", _pc.Config.preset("n32_gpt2_prefill")["quirks.fakecap_landings"])
    got = {}
    rc = run_prefill.main(["--tokens", "128", "--tail", "--plan", PLAN, "--repeat", "2",
                           "--work-dir", str(tmp_path / "ref")],
                          results=got.update)
    assert rc == 0
    assert got["top1"] == 1, got["top1"]
    assert got["medianKL"] < 0.1, got["medianKL"]
    assert got["pcmm_calls"] == 3 * 48, got["pcmm_calls"]      # the warm-up and two forwards
    assert len(set(got["free_gb"])) == 1, got["free_gb"]
