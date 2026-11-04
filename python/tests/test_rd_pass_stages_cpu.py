"""scripts/utils/rd_pass.sh: the raise-drop re-pass plans every block of the graph but the argmax stage, found by its
steps (CutMax), not by its number: GPT-2's block_13 is the argmax, Llama's block_13 a block whose exit block_14 enters
at (skipped, block_14 was re-planned at block_12's exit)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _node(step):
    return {"op_type": "add", "inputs": ["a", "b"], "input_levels": [30, 30], "output": "o", "step": step,
            "output_level": 30}


def _stages(blocks):
    """The stages rd_pass.sh plans for a graph whose block_<b> records `step` (RD_PASS_STAGES=1)."""
    name = f"_t_rdstages_{os.getpid()}"
    plan = REPO / "bootstrap_placements" / name
    with tempfile.TemporaryDirectory() as g:
        for b, step in blocks.items():
            os.makedirs(f"{g}/block_{b}")
            with open(f"{g}/block_{b}/graph.json", "w") as f:
                json.dump({"version": 2, "nodes": [_node(step)]}, f)
        Path(g, "capture_env.json").write_text("{}")
        plan.mkdir()
        try:
            (plan / "PLAN_CMD.txt").write_text(f"graph={g}\nrecipe=\n")
            out = subprocess.run(["bash", str(REPO / "scripts/utils/rd_pass.sh"), name], cwd=REPO, check=True,
                                 env=dict(os.environ, RD_PASS_STAGES="1", PYTHON="python3"),
                                 capture_output=True, text=True).stdout
        finally:
            shutil.rmtree(plan)
    return sorted(out.split(), key=lambda s: int(s.split("_")[1]))


class RdPassStages(unittest.TestCase):
    def test_gpt2_argmax_skipped(self):
        blocks = {b: f"blk{b}.ln_1" for b in range(12)} | {12: "tail", 13: "cutmax"}
        self.assertEqual(_stages(blocks), [f"block_{b}" for b in range(13)])

    def test_llama_block_13_kept(self):
        blocks = {b: f"blk{b}.ln_1" for b in range(32)} | {32: "tail"}
        self.assertEqual(_stages(blocks), [f"block_{b}" for b in range(33)])

    def test_argmax_found_by_steps(self):
        blocks = {b: f"blk{b}.ln_1" for b in range(32)} | {32: "tail", 33: "cutmax"}
        self.assertEqual(_stages(blocks), [f"block_{b}" for b in range(33)])


if __name__ == "__main__":
    unittest.main()
