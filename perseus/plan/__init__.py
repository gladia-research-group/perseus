"""Bootstrap-placement planning: captured graph JSON -> strict placement plans.

- `placer` — the planner (`PlanConfig`, `plan_block`, `plan_graph_dir`): min-cut placement
  with the correction-factor and band accuracy model, exact-degree simulation, and the
  ported baseline placers; CLI: `perseus-plan` / `python -m perseus.plan`.
- `btserr` — the bootstrap accuracy model the planner prices refreshes with.
- `contract` — the capture-time env contract stamped into plans and validated at load.

Only third-party dependency: networkx.
"""

from .contract import PlanContractError, capture_contract, read_stamp, validate_contract
from .placer.planner import PlanConfig, PlanDiagnostics, plan_block, plan_graph_dir

__all__ = ["PlanConfig", "PlanContractError", "PlanDiagnostics", "capture_contract", "plan_block", "plan_graph_dir", "read_stamp",
           "validate_contract"]
