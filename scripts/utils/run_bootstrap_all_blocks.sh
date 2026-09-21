#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper over `python -m perseus.plan` (perseus/plan/driver.py owns the
# defaults and full knob documentation). Env interface unchanged:
#   GRAPH_DIR OUT_NAME MAX_ABS MIN_ABS MAX_LEVEL BTS_LEVEL SRC_LEVEL
#   CACHE_READ_LEVEL BTS_OUT_DEG HINT_SHIFT BTS_LEVEL_B13 MIN_ABS_B13
#   ERASE_KEEP_STEPS NO_RELOCATE RELOC_MIN_ABS RELOC_SAFETY_MARGIN
#   FORBID_STEPS EMIT_DELIBERATE ERASE_AUTOBTS ERASE_OOB_INBAND  (single-dash: VAR= disables)
# Defaults = the verified smart-cut decode baseline.
# Eager-mirror recipe: ERASE_AUTOBTS= MAX_ABS=10 (+ MAX_LEVEL=28 for f-entry chunks).
# Requires python3 >= 3.9 with networkx (repo venv).

GRAPH_DIR="${GRAPH_DIR:-.cache/graph_cheb_l24}"
OUT_NAME="${OUT_NAME:-planned_cheb_l24_smart_fvar}"
PYTHON="${PYTHON:-python3}"

args=(
  --graph-dir "$GRAPH_DIR"
  --out-dir "./bootstrap_placements/${OUT_NAME}"
  --max-abs "${MAX_ABS:-50}"
  --min-abs "${MIN_ABS:-0.01}"
  --max-level "${MAX_LEVEL:-24}"
  --bootstrap-level "${BTS_LEVEL:-16}"
  --source-level "${SRC_LEVEL:-16}"
  --cache-read-level "${CACHE_READ_LEVEL:-17}"
  --bts-out-deg "${BTS_OUT_DEG:-2}"
  --hint-shift "${HINT_SHIFT:-0}"
  --forbid-steps "${FORBID_STEPS-.var,.mean}"
)

if [[ -n "${EMIT_DELIBERATE-1}"   ]]; then args+=(--emit-deliberate);   else args+=(--no-emit-deliberate);   fi
if [[ -n "${ERASE_AUTOBTS-1}"     ]]; then args+=(--erase-autobts);     else args+=(--no-erase-autobts);     fi
if [[ -n "${ERASE_OOB_INBAND-1}"  ]]; then args+=(--erase-oob-inband);  else args+=(--no-erase-oob-inband);  fi
if [[ -n "${ERASE_KEEP_STEPS:-}"     ]]; then args+=(--erase-keep-steps "$ERASE_KEEP_STEPS");         fi
if [[ -n "${NO_RELOCATE:-}"          ]]; then args+=(--no-relocate);                                  fi
if [[ -n "${RELOC_MIN_ABS:-}"        ]]; then args+=(--reloc-min-abs "$RELOC_MIN_ABS");               fi
if [[ -n "${RELOC_SAFETY_MARGIN:-}"  ]]; then args+=(--reloc-safety-margin "$RELOC_SAFETY_MARGIN");   fi
if [[ -n "${BTS_LEVEL_B13:-}"        ]]; then args+=(--bts-level-b13 "$BTS_LEVEL_B13");               fi
if [[ -n "${MIN_ABS_B13:-}"          ]]; then args+=(--min-abs-b13 "$MIN_ABS_B13");                   fi
if [[ -n "${FIRST_ENTRY_LEVEL:-}"    ]]; then args+=(--first-entry-level "$FIRST_ENTRY_LEVEL");       fi
if [[ -n "${FIRST_ENTRY_DEG:-}"      ]]; then args+=(--first-entry-deg "$FIRST_ENTRY_DEG");           fi

"$PYTHON" -m perseus.plan "${args[@]}"
