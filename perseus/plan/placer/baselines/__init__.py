"""Baseline bootstrap placers for the benchmark (see base.py for the design).

  orion    — MIT-licensed port of baahl-nyu/orion's level-DAG shortest path
             (orion/core/level_dag.py + auto_bootstrap.py), adapted to our IR.
  dacapo   — MIT-licensed port of corelab-src/dacapo's segment DP
             (DaCapoPlanner + CandidateSelection + BypassDetection), adapted.
  fhelipe  — CLEAN-ROOM reimplementation of Fhelipe's depth-boundary DP from the
             PLDI 2024 paper (Sec. 6). Upstream is GPL-3; no upstream code was
             read into this module.

Selection: PlanConfig.placer / --placer.
"""

from __future__ import annotations

import json
from pathlib import Path

from .dacapo import DaCapoPlacer
from .fhelipe import FhelipePlacer
from .orion import OrionPlacer

PLACERS = {
    FhelipePlacer.NAME_KEY: FhelipePlacer,
    OrionPlacer.NAME_KEY: OrionPlacer,
    DaCapoPlacer.NAME_KEY: DaCapoPlacer,
}

#: Flat per-op latency weights in ms for the DaCapo segment DP, on the n32 chain.
#: `bootstrap` is the measured dense 4:3 wall; the rest are order-of-magnitude op costs.
#: The bootstrap dominates by 60-500x, so per-level refinement of the others is
#: second-order. Swap the file via --latency-table to test sensitivity.
_DEFAULT_LATENCY = Path(__file__).with_name("latency_n32.json")


def make_placer(name: str, *, rescue: bool = False, depth_cap: float | None = None,
                latency_table_path: str | None = None, **placer_kwargs):
    if name not in PLACERS:
        raise ValueError(f"unknown placer '{name}' (have: {sorted(PLACERS)} "
                         "+ the default 'min_cut')")
    cls = PLACERS[name]
    extra = {}
    if cls is DaCapoPlacer:
        path = Path(latency_table_path) if latency_table_path else _DEFAULT_LATENCY
        extra["latency"] = json.loads(path.read_text(encoding="utf-8"))
        extra["latency_table_name"] = path.name
    return cls(rescue=rescue, depth_cap=depth_cap, **extra, **placer_kwargs)
