#!/bin/bash
# Plan Decode-8 on the lean chain n32a4 (pc-core ChainSpec::n32a4; CPU only), from an n32 plan's captures with its
# recipe (by default the served plan's base, gpt2_decode_python_n32_kc16r: its PLAN_CMD.txt, the ILP with one HiGHS
# thread) and every value and refresh at level 48 or below: n32a4's deepest user level, the composite pair of its 2nd
# and 3rd bottom primes (n32's is 50). Then:
#   - stage 12 once more, alone at block 11's exit, its exit capped at effective level 46 (PLAN_ILP_EXIT_CAP; 44/d2):
#     CutMax's stage 13 takes its first refresh two levels below its entry, past the scaled normaliser (n32: at 50 from
#     46/d2), so on n32a4 it needs that entry one level higher;
#   - block 0's feedback variant (make_plan.sh feedback);
#   - stage 13, CutMax with the scaled normalisers, by speedrun/make_cutmax_plan.sh at stage 12's exit on the ILP
#     placer (CUTMAX_PLACER=ilp): the one refresh n32a4 adds there goes where the ILP's quality weight puts it (CF 15
#     on CutMax's last iteration at input level 24, not the min-cut's CF-20 square at its end), which keeps CutMax's
#     one-hot mass at the record's. The ILP also takes the recipe's raise drops into stage 13 (that refresh and the
#     record's hint v_54 at raise drop 2).
# The blocks pass also plans the base capture's stage 13 (the argmax before CutMax) at the old exit; it is dropped,
# as the served plan's stage 13 is CutMax's. The runtime line gains CHAIN_PRESET=n32a4 (the server's chain) and
# FIDESLIB_BTS_SPARSE_B_K=16: the SparseB refreshes' EvalMod on the dense route's degree-14 series (Prefill-128's), same
# levels and landings as K = 24's degree-20 table, about 0.7 ms less per refresh on the RTX PRO 6000.
# RESUME=1 keeps the blocks already planned (a preempted run). BASE (an n32 Decode-8 plan with a `_cm` CutMax plan
# beside it), GR (its stages 0-12 capture), CUTMAX_GRAPH (its stage-13 capture) and CONFIGS (its CutMax configs) name
# another n32 plan to take the recipe and captures of; OUT is BASE with n32a4 for n32.
#   bash speedrun/make_n32a4_decode_plan.sh   -> bootstrap_placements/gpt2_decode_python_n32a4_kc16r{,_cm} (served)
#   BASE=gpt2_decode_python_n32_kc16 GR=graphs/gpt2_decode_python_n32_kc16m \
#   CUTMAX_GRAPH=graphs/gpt2_decode_python_n32_kc16_cm \
#   CONFIGS=configs/model/approximation/gpt2_base_n32_kc16_cm/configs.json bash speedrun/make_n32a4_decode_plan.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/local_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1
BASE=${BASE:-gpt2_decode_python_n32_kc16r} GR=${GR:-graphs/gpt2_decode_python_n32_kc16rm}
CUTMAX_GRAPH=${CUTMAX_GRAPH:-graphs/gpt2_decode_python_n32_kc16r_cm}
CONFIGS=${CONFIGS:-configs/model/approximation/gpt2_base_n32_kc16r_cm/configs.json}
OUT=${BASE/_n32_/_n32a4_}
[ "$OUT" != "$BASE" ] || { echo "BASE=$BASE: not an n32 plan name" >&2; exit 2; }
CAP=46
RECIPE="$(sed -n 's/^recipe=//p' "bootstrap_placements/$BASE/PLAN_CMD.txt") MAX_LEVEL=48 PLAN_REFRESH_ENV_CAP=48"
[ "${RESUME:-0}" = 1 ] || rm -rf "bootstrap_placements/$OUT"
# shellcheck disable=SC2086
env $RECIPE PLAN_RESUME="${RESUME:-0}" bash python/perseus/examples/gpt2/make_plan.sh "$GR" "$OUT" blocks
read -r L D < <("$PYTHON" -c 'import json, sys; s = json.load(open(sys.argv[1]))["summary"]
print(s["exit_level"], s.get("exit_deg") or 1)' "bootstrap_placements/$OUT/block_11_placement.json")
TMP="$(mktemp -d graphs/_b12_XXXX)"
ln -s "$PWD/$GR/block_12" "$TMP/block_12" && cp "$GR/capture_env.json" "$TMP/"
# shellcheck disable=SC2086
env $RECIPE BLOCKS_ENTRY_LEVEL="$L" BLOCKS_ENTRY_DEG="$D" PLAN_ILP_EXIT_CAP="$CAP" \
    bash python/perseus/examples/gpt2/make_plan.sh "$TMP" "${OUT}_b12" blocks
rm -rf "$TMP"
"$PYTHON" - "bootstrap_placements/${OUT}_b12/block_12_placement.json" "$CAP" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
eff = s["exit_level"] + (2 if (s.get("exit_deg") or 1) == 2 else 0)
if eff > float(sys.argv[2]):
    sys.exit(f"stage 12 exits at {s['exit_level']}/d{s.get('exit_deg')}, past the cap {sys.argv[2]}")
PY
cp "bootstrap_placements/${OUT}_b12/block_12_placement.json" "bootstrap_placements/$OUT/"
rm -rf "bootstrap_placements/${OUT}_b12" "bootstrap_placements/$OUT/block_13_placement.json"
# shellcheck disable=SC2086
env $RECIPE bash python/perseus/examples/gpt2/make_plan.sh "$GR" "$OUT" feedback
TOOL="speedrun/make_n32a4_decode_plan.sh (make_plan.sh blocks; stage 12 alone at block 11's exit with \
PLAN_ILP_EXIT_CAP=$CAP; feedback)"
sed -e "s|^tool=.*|tool=$TOOL|" -e "s|^recipe=.*|recipe=$RECIPE|" -e "s|^runtime=\(.*\)|runtime=\1 CHAIN_PRESET=n32a4 FIDESLIB_BTS_SPARSE_B_K=16|" \
    -e "s|make_plan.sh's chain (blocks, argmax, feedback)|make_n32a4_decode_plan.sh's chain|" \
    "bootstrap_placements/$BASE/PLAN_CMD.txt" > "bootstrap_placements/$OUT/PLAN_CMD.txt"
grep "^capture13=" "bootstrap_placements/${BASE}_cm/PLAN_CMD.txt" >> "bootstrap_placements/$OUT/PLAN_CMD.txt" || true
CUTMAX_PLACER=ilp CUTMAX_GRAPH="$CUTMAX_GRAPH" bash speedrun/make_cutmax_plan.sh "$OUT" "${OUT}_cm" "$CONFIGS"
"$PYTHON" - bootstrap_placements "${BASE}_cm" "${OUT}_cm" <<'PY'
import json, sys
root, *names = sys.argv[1:]
for name in names:
    tot = [json.load(open(f"{root}/{name}/block_{b}_placement.json"))["summary"]["total_bootstraps"] for b in range(14)]
    print(f"{name}: refreshes {tot}, stages 0-12 {sum(tot[:13])}, all {sum(tot)}")
PY
