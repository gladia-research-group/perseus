"""Bootstrap-placement planning: captured graph JSON -> strict placement plans.

- `framework` — the min-cut / range-aware / white-box / level-pinning planner core
  (single block; parameter-agnostic, the captured graph is the source of truth).
- `driver` — per-graph-dir orchestration with block entry chaining (`PlanOptions`,
  `plan_graph_dir`); CLI: `python -m perseus.plan`.
- `annotate` — retrofit `-lvl=N` names onto anonymous KV-cache reads in filling
  captures (mandatory before planning them); CLI: `python -m perseus.plan.annotate`.

Only third-party dependency: networkx.
"""

from .annotate import annotate_graph_dir
from .driver import PlanOptions, plan_block, plan_graph_dir

__all__ = ["PlanOptions", "annotate_graph_dir", "plan_block", "plan_graph_dir"]
