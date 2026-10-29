"""ViT-B/16 planned on the new runtime (`examples.vit.planned.VitPlannedPrefill`): patch embedding as a PCMM in
block 0, two-page bidirectional CCMM attention, every block a compiled `Program` run in Rust; the 4 EuroSAT images of
the old repro set vs the float64 model.

Gates (the full gate is the paired `vit.run_prefill --n 4 --plan bootstrap_placements/vit_b16_224`):
top-1 and top-5 4/4, device free memory flat. Long: ~10 min (GPU_JOB_TIMEOUT 7200 under tools/gpu.sh).
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

PLAN = "bootstrap_placements/vit_b16_224"
IMAGES = "scripts/repro/data/vit_eurosat"


@pytest.mark.skipif(not (os.path.isdir(IMAGES) and os.environ.get("HF_HOME")), reason="repro images / HF_HOME missing")
def test_vit_planned_eurosat4(tmp_path, monkeypatch):
    from perseus.examples.vit import run_prefill
    monkeypatch.setenv("PERSEUS_SLOTVEC_CACHE", "0")
    got = {}
    rc = run_prefill.main(["--images", IMAGES, "--n", "4", "--plan", PLAN, "--work-dir", str(tmp_path / "w")],
                          results=got.update)
    assert rc == 0
    assert got["top1"] == got["n"] == 4, got
    assert got["top5"] == 4, got
    assert len(set(got["free_gb"])) == 1, got["free_gb"]
