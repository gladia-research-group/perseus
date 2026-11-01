#!/bin/bash
# Prefill-128 with the packed attention reductions' window ladders traded for one rotation per window (CPU only):
# impl/prefill.py PF_ATTN_WROT (1: q.K^T with its window masks on the K side; 2: P.V with y rotated once per window and
# the masks on the probabilities' side). The op stream computes the same values as BASE's (the served plan's) with
# other ops inside attn.qkt (and attn.softmax_v), so BASE's decisions carry over by name: both streams are recorded
# as the server records them (scripts/utils/remap_plan.py record), aligned per block, BASE's placements, hints, factors,
# drops, routes and realize anchors renamed onto the new stream (remap), the new nodes left lazy (no realize anchor)
# and unpinned except where they sit at the reactive ceiling as the nodes they replace did (pin, from a keep-going
# compile); a last compile checks every remaining pin (level one-sided, degree exact) with no error.
# PAIR_WINDOW: the runtime line's PC_BTS_PAIR_WINDOW (the attention's extra ops move same-depth refreshes further apart
# than the default pair window: 6144 keeps WROT=3's refresh pairs, 4096 WROT=1's).
# MULT_SUM=1: the runtime line's PC_MULT_SUM=1 (P.V's products of a giant step relinearized once: compile::msum).
#   WROT=3 PAIR_WINDOW=6144 MULT_SUM=1 bash speedrun/make_prefill_wrot_plan.sh   -> bootstrap_placements/gpt2_prefill128_n32a4_pfbhrd_cmw
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/local_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
BASE=${BASE:-gpt2_prefill128_n32a4_pfbhrd_cm} OUT=${OUT:-gpt2_prefill128_n32a4_pfbhrd_cmw} W=${WROT:-1}
STEPS=attn.qkt
if [ $((W & 6)) != 0 ]; then STEPS=attn.qkt,attn.softmax_v; fi
R="$PYTHON scripts/utils/remap_plan.py"
T="$(mktemp -d graphs/_wrot_XXXX)"
PLAN=bootstrap_placements/$BASE $R record "$T/old" 0
PLAN=bootstrap_placements/$BASE $R record "$T/new" "$W"
rm -rf "bootstrap_placements/$OUT"
# blocks 0-10 run prefill_attention; block 11 (the last position alone) and the tail stages keep BASE's stream
$R remap "$T/old" "$T/new" "bootstrap_placements/$BASE" "bootstrap_placements/$OUT" --blocks 0-10 --unpin "$STEPS"
sed -i "s/^runtime=\(.*\)$/runtime=\1 PF_ATTN_WROT=$W${PAIR_WINDOW:+ PC_BTS_PAIR_WINDOW=$PAIR_WINDOW}${MULT_SUM:+ PC_MULT_SUM=1}/" "bootstrap_placements/$OUT/PLAN_CMD.txt"
PLAN=bootstrap_placements/$OUT COMPILE=1 $R record "$T/cw" "$W"
$R pin "bootstrap_placements/$OUT" "$T/cw" --blocks 0-10 --steps "$STEPS"
PLAN=bootstrap_placements/$OUT COMPILE=1 $R record "$T/cw2" "$W"
"$PYTHON" - "$T/cw2" <<'PY'
import glob, json, sys
bad = {f: p["check"]["errors"][:2] for f in sorted(glob.glob(f"{sys.argv[1]}/block_*.prog.json"))
       for p in [json.load(open(f))] if p["check"]["errors"]}
sys.exit(f"check errors left: {bad}" if bad else 0)
PY
rm -rf "$T"
echo "wrot=blocks 0-10 by speedrun/make_prefill_wrot_plan.sh (WROT=$W): $BASE's decisions carried by name onto the op \
stream with PF_ATTN_WROT=$W ($STEPS rewritten: their new nodes lazy, unpinned but at the reactive ceiling); blocks 11-13 \
are $BASE's${PAIR_WINDOW:+; refresh pairs looked for $PAIR_WINDOW launches ahead (PC_BTS_PAIR_WINDOW)}${MULT_SUM:+; added products relinearized once (PC_MULT_SUM)}" >> "bootstrap_placements/$OUT/PLAN_CMD.txt"
