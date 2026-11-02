#!/bin/bash
# rd_drops.sh PLAN: the raise-drop re-pass (rd_pass.sh) with the code's own refreshes lowered where they have slack
# (CPU only). Each round runs the pass with the deliberate-drop probe (PLAN_RAISE_DELIB=1) and turns what it finds
# into PF_PCMM_RAISE_OUT (PCMM finishes) and KV_PUSH_RAISE (K/V push refreshes) with scripts/utils/delib_drops.py;
# the next round lands them, until the probe finds nothing more (the probe takes at most 5 levels per refresh and
# round). The plan's rd= line ends with the switches of the last round, which regenerate it in one pass.
#   bash scripts/utils/rd_drops.sh gpt2_prefill128       -> bootstrap_placements/gpt2_prefill128 in place
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/local_env.sh
PLAN=${1:?plan dir under bootstrap_placements/}
GR=$(sed -n 's/^graph=\([^ ]*\).*/\1/p' "bootstrap_placements/$PLAN/PLAN_CMD.txt")
# the attention the plan runs (delib_drops.py keeps a CCMM's operands at their drops)
ATTN="$(sed -n 's/^runtime=.*\bPF_ATTN=\([^ ]*\).*/\1/p' "bootstrap_placements/$PLAN/PLAN_CMD.txt")"
LOGS=${RD_LOGS:-logs/plans}; mkdir -p "$LOGS"
cur_pcmm="" cur_kv=""
for round in 1 2 3 4 5 6; do
    log="$LOGS/rd_drops_${PLAN}_$round.log"
    PLAN_RAISE_DELIB=1 PF_PCMM_RAISE_OUT="$cur_pcmm" KV_PUSH_RAISE="$cur_kv" bash scripts/utils/rd_pass.sh "$PLAN" > "$log" 2>&1
    out=$(PF_ATTN="$ATTN" PF_PCMM_RAISE_OUT="$cur_pcmm" KV_PUSH_RAISE="$cur_kv" "$PYTHON" scripts/utils/delib_drops.py "$log" "$GR" 2>>"$log")
    nxt_pcmm=$(sed -n 's/^PF_PCMM_RAISE_OUT=//p' <<< "$out")
    nxt_kv=$(sed -n 's/^KV_PUSH_RAISE=//p' <<< "$out")
    echo "round $round: $(grep -h '^rd_pass: [0-9]* refreshes lower' "$log") PF_PCMM_RAISE_OUT=${nxt_pcmm:-(none)} KV_PUSH_RAISE=${nxt_kv:-(none)}"
    if [ "$nxt_pcmm" = "$cur_pcmm" ] && [ "$nxt_kv" = "$cur_kv" ]; then
        exit 0
    fi
    cur_pcmm=$nxt_pcmm cur_kv=$nxt_kv
done
echo "rd_drops.sh: the probe still finds slack after 6 rounds" >&2
exit 1
