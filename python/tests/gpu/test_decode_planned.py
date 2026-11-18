"""GPT-2 decode, planned on the device: every block a compiled `Program` run in Rust
(`impl.driver.PlannedRun`), 4 teacher-forced tokens + CutMax vs the PyTorch oracle.

Gates (the 16-token gate is `run_decode --tokens 16 --plan ... --check`): every token's top-1 equals the
oracle's, KL < 0.01, CutMax == top-1, device free memory flat over the tokens, every block replays its graphs after
the first token. Weights and oracle rows: WEIGHTS_PATH / ALL_BLOCKS_IO_DIR, defaulting to the old repo's `.cache`.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.gpu

_pc = pytest.importorskip("perseus._pc")

OLD = "/data02/users/mithrillo/projects/perseus/.cache"
WEIGHTS = os.environ.get("WEIGHTS_PATH") or f"{OLD}/models/openai-community/gpt2/classic/weights.bin.zip"
IO_DIR = os.environ.get("ALL_BLOCKS_IO_DIR") or f"{OLD}/oracle/gpt2/all_blocks_io"
PLAN = "bootstrap_placements/gpt2_decode"


@pytest.mark.skipif(not (os.path.exists(WEIGHTS) and os.path.isdir(IO_DIR)), reason="GPT-2 weights / oracle io missing")
def test_decode_planned_4_tokens(tmp_path, monkeypatch):
    from perseus.examples.gpt2 import run_decode
    monkeypatch.setenv("PERSEUS_SLOTVEC_CACHE", "0")
    got = {}

    def on_token(res):
        got["res"] = res

    rc = run_decode.main(["--tokens", "4", "--plan", PLAN, "--argmax", "--check", "--weights", WEIGHTS,
                          "--io-dir", IO_DIR, "--work-dir", str(tmp_path / "ref")], results=on_token)
    res = got["res"]
    assert rc == 0
    assert len(res) == 4
    for r in res:
        assert r["hit"], (r["pos"], r["top1"], r["ref"])
        assert r["kl"] < 0.01, (r["pos"], r["kl"])
        assert r["cutmax"] == r["top1"], (r["pos"], r["cutmax"], r["top1"])
    assert len({r["free_gb"] for r in res}) == 1, [r["free_gb"] for r in res]
    # the plan binds at every cache length (SM_PERIODIC=2): the classed run replays eagerly, its plaintexts
    # retargeted per token
    assert all(set(r["served"]) == {"e"} for r in res), [r["served"] for r in res]
