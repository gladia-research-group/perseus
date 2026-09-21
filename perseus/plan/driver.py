"""Graph-dir planning driver: plan every captured block with entry chaining.

Typed replacement for the scripts/run_bootstrap_all_blocks.sh env interface. Blocks
are planned in numeric order; each block's planned exit (level, deg) seeds the next
block's entry (planned exits differ from the eager capture's). Per-weight encode
levels, mask levels and the KV cache pin are emitted into each placement json.

Keep max_level / cache_read_level locked to the runtime AUTO_BTS_LEVEL /
CACHE_READ_LEVEL so plan and runtime never diverge.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import framework


@dataclass(frozen=True)
class PlanOptions:
    """Planner policy for one graph dir.

    Defaults reproduce the verified decode baseline (the smart cut): range-aware
    min-cut with erased auto-bts, forbid discretionary cuts on LN .var (outlier)
    and .mean (sub-floor), erase out-of-band auto-bts onto clean in-band ancestors.
    """

    max_level: int = 24
    bootstrap_level: int = 16
    source_level: int = 16
    cache_read_level: int = 17
    bts_out_deg: int = 2
    hint_shift: float = 0.0
    max_abs: float = 50.0
    min_abs: float = 0.01
    emit_deliberate_placements: bool = True
    erase_autobts: bool = True
    erase_oob_inband: bool = True
    erase_keep_steps: tuple[str, ...] = ()
    no_relocate: bool = False
    reloc_min_abs: float = 0.0
    reloc_safety_margin: float = 1.0
    forbid_steps: tuple[str, ...] = (".var", ".mean")
    write_var_graphs: bool = True

    bts_level_b13: Optional[int] = None
    min_abs_b13: Optional[float] = None

    first_entry_level: Optional[int] = None
    first_entry_deg: Optional[int] = None

    @classmethod
    def eager_mirror(cls, **overrides) -> "PlanOptions":
        """The precise eager-mirror recipe (shell: `ERASE_AUTOBTS= MAX_ABS=10`).

        Reproduces the captured auto-bts exactly; reads no magnitudes, so
        magnitude-free graphs plan fine. Filling chunks with f-entries additionally
        need max_level=28: the violation detector is 1-2 levels conservative on the
        capture-proven refine chains, and the budget is not consumed by the runtime.
        Do NOT raise max_level for cut recipes.
        """
        base = dict(erase_autobts=False, erase_oob_inband=False, max_abs=10.0)
        base.update(overrides)
        return cls(**base)


def plan_block(graph_file: Path, out_file: Path, opts: PlanOptions, *,
               entry_level: Optional[int] = None,
               entry_deg: Optional[int] = None,
               var_graph_out: Optional[Path] = None) -> dict:
    """Plan one captured block graph; returns the placement dict (also written)."""
    framework.HINT_LEVEL_SHIFT = opts.hint_shift
    framework.BTS_OUT_DEG = opts.bts_out_deg
    graph = json.loads(Path(graph_file).read_text(encoding="utf-8"))

    result = framework.optimize_global(
        graph=graph,
        bootstrap_level=opts.bootstrap_level,
        max_level=opts.max_level,
        source_level=opts.source_level,
        cache_read_level=opts.cache_read_level,
        max_abs_threshold=opts.max_abs,
        min_abs_threshold=opts.min_abs,
        entry_level_override=entry_level,
        entry_deg_override=entry_deg,
        emit_deliberate_placements=opts.emit_deliberate_placements,
        erase_autobts=opts.erase_autobts,
        erase_keep_steps=list(opts.erase_keep_steps),
        erase_oob_inband=opts.erase_oob_inband,
        no_relocate=opts.no_relocate,
        reloc_min_abs=opts.reloc_min_abs,
        reloc_safety_margin=opts.reloc_safety_margin,
        forbid_steps=list(opts.forbid_steps),
    )

    var_graph = result.pop("var_participation_graph", None)
    if var_graph is not None and var_graph_out is not None:
        var_graph_out.parent.mkdir(parents=True, exist_ok=True)
        var_graph_out.write_text(json.dumps(var_graph, indent=2), encoding="utf-8")

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _block_index(block_dir: Path) -> int:
    return int(block_dir.name.rsplit("_", 1)[1])


def plan_graph_dir(graph_dir: Path, out_dir: Path,
                   opts: PlanOptions = PlanOptions()) -> list[dict]:
    """Plan all block_*/graph.json under graph_dir; returns per-block summaries."""
    graph_dir, out_dir = Path(graph_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out={out_dir} max_abs={opts.max_abs} max_level={opts.max_level} "
          f"cache_read_level={opts.cache_read_level}")

    summaries: list[dict] = []
    entry_level: Optional[int] = opts.first_entry_level
    entry_deg: Optional[int] = opts.first_entry_deg
    blocks = sorted((d for d in graph_dir.glob("block_*") if d.is_dir()), key=_block_index)
    for block_dir in blocks:
        graph_file = block_dir / "graph.json"
        if not graph_file.is_file():
            print(f"Skipping {block_dir} (no graph.json)")
            continue

        block_opts = opts
        if block_dir.name == "block_13":
            overrides = {}
            if opts.bts_level_b13 is not None:
                overrides["bootstrap_level"] = opts.bts_level_b13
            if opts.min_abs_b13 is not None:
                overrides["min_abs"] = opts.min_abs_b13
            if overrides:
                block_opts = dataclasses.replace(opts, **overrides)

        out_file = out_dir / f"{block_dir.name}_placement.json"
        entry_desc = (f"--entry-level {entry_level} --entry-deg {entry_deg}"
                      if entry_level is not None else "capture-derived")
        print(f"[{block_dir.name}] -> {out_file} (entry: {entry_desc}, "
              f"bts_level={block_opts.bootstrap_level})")

        var_graph_out = (out_dir / f"{block_dir.name}_var_graph.json"
                         if opts.write_var_graphs else None)
        result = plan_block(graph_file, out_file, block_opts,
                            entry_level=entry_level, entry_deg=entry_deg,
                            var_graph_out=var_graph_out)

        summary = result["summary"]
        print(f"  bootstraps={summary['total_bootstraps']} "
              f"placements={summary['num_placements']}")
        summaries.append({"block": block_dir.name, "output": str(out_file), **summary})

        entry_level = summary.get("exit_level")
        entry_deg = (summary.get("exit_deg") or 1) if entry_level is not None else None

    return summaries
