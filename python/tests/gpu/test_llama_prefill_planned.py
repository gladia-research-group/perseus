"""Llama-3.1-8B prefill, planned on the device: blocks 0-1 of the 128-token plan (BOS + 127 tokens),
every block a compiled `Program` with its PCMM linears and CCMM attention served as `Finish` ops inside the replay.

Gates (the full gate is `run_prefill --tokens 127 --plan bootstrap_placements/llama31_8b_prefill128
--check`): every finish served, compiled metadata == executed metadata, block outputs within the FIDESlib
reference's per-block error class vs the float64 model (1.5e-3 / 1.2e-3 at |x| 0.41 / 0.49), device memory flat.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

PLAN = "bootstrap_placements/llama31_8b_prefill128"


@pytest.mark.skipif(not os.environ.get("HF_HOME"), reason="Llama weights / oracle cache (HF_HOME) missing")
def test_llama_prefill_planned_two_blocks(tmp_path, monkeypatch, capsys):
    from perseus.examples.llama import run_prefill
    monkeypatch.setenv("PERSEUS_SLOTVEC_CACHE", "0")
    got = {}
    # blocks 0-1 are full blocks in the plan: its PF_LAST_QUERY tail is block 31's, not the last of a --layers run
    rc = run_prefill.main(["--tokens", "127", "--plan", PLAN, "--layers", "2", "--check", "--check-meta",
                           "--mem-per-block", "--set", "PF_LAST_QUERY=0", "--work-dir", str(tmp_path / "w")],
                          results=got.update)
    assert rc == 0
    out = capsys.readouterr().out
    assert "metadata vs compile: 0 mismatches" in out
    errs = [float(line.split("maxerr=")[1].split()[0]) for line in out.splitlines() if "] block " in line and "maxerr=" in line]
    assert len(errs) == 2 and errs[0] < 5e-3 and errs[1] < 1e-2, errs
    assert (got["pcmm_calls"], got["ccmm_calls"]) == (2 * 8, 1 + 2 * 4), (got["pcmm_calls"], got["ccmm_calls"])
    assert len(set(got["block_free_gb"][0])) == 1, got["block_free_gb"]
