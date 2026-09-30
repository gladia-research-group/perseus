#!/usr/bin/env bash
# run_bootstrap_all_blocks.sh — env-driven wrapper over `python -m perseus.plan`.
#   GRAPH_DIR=graphs/gpt2_decode_python_n32 OUT_NAME=<name> <PLAN_* knobs> bash scripts/utils/run_bootstrap_all_blocks.sh
# Every PLAN_* variable below maps to one planner flag; unset = the planner's default
# (`python -m perseus.plan --help`). Each shipped plan's recipe is in its PLAN_CMD.txt.
set -euo pipefail
GRAPH_DIR="${GRAPH_DIR:?set GRAPH_DIR=<dir with block_N/graph.json>}"
OUT_NAME="${OUT_NAME:?set OUT_NAME=<bootstrap_placements/<name>>}"
PYTHON="${PYTHON:-python3}"

args=(
  --graph-dir "$GRAPH_DIR"
  --out-dir "./bootstrap_placements/${OUT_NAME}"
  --max-level "${MAX_LEVEL:-24}"
  --bootstrap-level "${BTS_LEVEL:-16}"
  --source-level "${SRC_LEVEL:-16}"
  --cache-read-level "${CACHE_READ_LEVEL:-17}"
)
if [[ -n "${PLAN_LEVEL_UNIT:-}"     ]]; then args+=(--level-unit "$PLAN_LEVEL_UNIT");         fi
if [[ -n "${PLAN_SPARSE_SLOTS:-}"   ]]; then args+=(--sparse-slots "$PLAN_SPARSE_SLOTS");     fi
if [[ -n "${PLAN_SPARSE_BTS_OUT:-}" ]]; then args+=(--sparse-bts-out "$PLAN_SPARSE_BTS_OUT"); fi
if [[ -n "${BTS_OUT_DEG:-}"         ]]; then args+=(--bts-out-deg "$BTS_OUT_DEG");            fi
# set-but-empty is meaningful: it selects the analytic error model (no measured table)
if [[ -n "${PLAN_ACC_CHAIN+x}"      ]]; then args+=(--acc-chain "$PLAN_ACC_CHAIN");           fi
if [[ -n "${PLAN_ACC_TABLE:-}"      ]]; then args+=(--acc-table "$PLAN_ACC_TABLE");           fi
if [[ -n "${PLAN_CF_MIN:-}"         ]]; then args+=(--cf-min "$PLAN_CF_MIN");                 fi
if [[ -n "${PLAN_CF_MAX:-}"         ]]; then args+=(--cf-max "$PLAN_CF_MAX");                 fi
if [[ -n "${PLAN_ERR_TARGET:-}"     ]]; then args+=(--err-target "$PLAN_ERR_TARGET");         fi
if [[ -n "${PLAN_ERR_HOPELESS:-}"   ]]; then args+=(--err-hopeless "$PLAN_ERR_HOPELESS");     fi
if [[ -n "${PLAN_PRESCALE_REACH:-}" ]]; then args+=(--prescale-reach "$PLAN_PRESCALE_REACH"); fi
if [[ -n "${PLAN_MAG_SAFETY:-}"     ]]; then args+=(--mag-safety "$PLAN_MAG_SAFETY");         fi
if [[ -n "${PLAN_EMIT_OFFSET:-}"    ]]; then args+=(--emit-offset);                           fi
if [[ -n "${PLAN_NO_PRESCALE:-}"    ]]; then args+=(--no-prescale);                           fi
if [[ -n "${PLAN_MISS_PENALTY:-}"   ]]; then args+=(--miss-penalty "$PLAN_MISS_PENALTY");     fi
if [[ -n "${PLAN_PRESCALE_BITS:-}"  ]]; then args+=(--prescale-bits-max "$PLAN_PRESCALE_BITS"); fi
if [[ -n "${PLAN_QUALITY_WEIGHT:-}" ]]; then args+=(--quality-weight "$PLAN_QUALITY_WEIGHT"); fi
if [[ -n "${PLAN_RESCALE_OPT:-}"    ]]; then args+=(--rescale-opt);                           fi
if [[ -n "${PLAN_SITE_BTS_OUT_FILE:-}" ]]; then args+=(--site-bts-out-file "$PLAN_SITE_BTS_OUT_FILE"); fi
if [[ -n "${PLAN_BOUNDARY_REALIZE:-}" ]]; then args+=(--boundary-realize);                    fi
if [[ -n "${FORBID_STEPS+x}"        ]]; then args+=(--forbid-steps "$FORBID_STEPS");          fi
if [[ -n "${FIRST_ENTRY_LEVEL:-}"   ]]; then args+=(--first-entry-level "$FIRST_ENTRY_LEVEL");fi
if [[ -n "${FIRST_ENTRY_DEG:-}"     ]]; then args+=(--first-entry-deg "$FIRST_ENTRY_DEG");    fi
if [[ -n "${PLAN_BIND_HINTS:-}"     ]]; then args+=(--bind-hints);                            fi
if [[ -n "${PLAN_DISSOLVE_HINTS:-}" ]]; then args+=(--dissolve-hints);                        fi
if [[ -n "${PLAN_DELIBERATE_CLAMP0:-}" ]]; then args+=(--deliberate-clamp0);                 fi
if [[ -n "${PLAN_PLACER:-}"         ]]; then args+=(--placer "$PLAN_PLACER");                 fi
if [[ -n "${PLAN_BASELINE_RESCUE:-}"   ]]; then args+=(--baseline-rescue);                          fi
if [[ -n "${PLAN_BASELINE_DEPTH_CAP:-}" ]]; then args+=(--baseline-depth-cap "$PLAN_BASELINE_DEPTH_CAP"); fi
if [[ -n "${PLAN_LATENCY_TABLE:-}"  ]]; then args+=(--latency-table "$PLAN_LATENCY_TABLE");   fi
if [[ "${PLAN_PRUNE:-}" == 1        ]]; then args+=(--prune);                                 fi
exec "$PYTHON" -m perseus.plan.placer "${args[@]}"
