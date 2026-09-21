"""CLI for the graph-dir planning driver.

  python -m perseus.plan --graph-dir .cache/graph_<name> \
      --out-dir bootstrap_placements/planned_<name> [--mirror] [overrides]

Defaults = PlanOptions defaults (the verified smart-cut decode baseline).
--mirror applies PlanOptions.eager_mirror() before the explicit overrides.
"""

from __future__ import annotations

import argparse
import dataclasses

from .driver import PlanOptions, plan_graph_dir


def _parser() -> argparse.ArgumentParser:
    base = PlanOptions()
    p = argparse.ArgumentParser(prog="python -m perseus.plan", description=__doc__)
    p.add_argument("--graph-dir", required=True, help="captured block_*/graph.json dir")
    p.add_argument("--out-dir", required=True, help="placement output dir")
    p.add_argument("--mirror", action="store_true",
                   help="eager-mirror preset (reproduce captured auto-bts; no magnitudes)")

    opt = p.add_argument_group("PlanOptions overrides (unset = preset/baseline default)")
    opt.add_argument("--max-level", type=int)
    opt.add_argument("--bootstrap-level", type=int)
    opt.add_argument("--source-level", type=int)
    opt.add_argument("--cache-read-level", type=int)
    opt.add_argument("--bts-out-deg", type=int, choices=[1, 2])
    opt.add_argument("--hint-shift", type=float)
    opt.add_argument("--max-abs", type=float)
    opt.add_argument("--min-abs", type=float)
    opt.add_argument("--emit-deliberate", action=argparse.BooleanOptionalAction,
                     dest="emit_deliberate_placements")
    opt.add_argument("--erase-autobts", action=argparse.BooleanOptionalAction)
    opt.add_argument("--erase-oob-inband", action=argparse.BooleanOptionalAction)
    opt.add_argument("--erase-keep-steps", type=str,
                     help="comma-separated step substrings (default: none)")
    opt.add_argument("--no-relocate", action="store_true", default=None)
    opt.add_argument("--reloc-min-abs", type=float)
    opt.add_argument("--reloc-safety-margin", type=float)
    opt.add_argument("--forbid-steps", type=str,
                     help=f"comma-separated step substrings; '' disables "
                          f"(default: {','.join(base.forbid_steps)})")
    opt.add_argument("--var-graphs", action=argparse.BooleanOptionalAction,
                     dest="write_var_graphs")
    opt.add_argument("--bts-level-b13", type=int)
    opt.add_argument("--min-abs-b13", type=float)
    opt.add_argument("--first-entry-level", type=int,
                     help="block-0 entry level override (autoregressive gen: 16)")
    opt.add_argument("--first-entry-deg", type=int,
                     help="block-0 entry degree override (autoregressive gen bootstrapped entry: 2)")
    return p


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s for s in value.split(",") if s)


def main() -> None:
    args = vars(_parser().parse_args())
    graph_dir, out_dir = args.pop("graph_dir"), args.pop("out_dir")
    mirror = args.pop("mirror")
    for key in ("erase_keep_steps", "forbid_steps"):
        if args[key] is not None:
            args[key] = _csv(args[key])
    overrides = {k: v for k, v in args.items() if v is not None}
    opts = PlanOptions.eager_mirror(**overrides) if mirror else PlanOptions(**overrides)
    summaries = plan_graph_dir(graph_dir, out_dir, opts)
    total = sum(s["total_bootstraps"] for s in summaries)
    print(f"planned {len(summaries)} blocks, {total} bootstraps -> {out_dir}")


if __name__ == "__main__":
    main()
