#!/bin/bash
# Plan Decode-8 on the alternating chain n32sp (pc-core ChainSpec::n32sp: n32a4 with its 9 user levels replaced by 12
# levels that drop one prime and two in turn; CPU only), from scratch: the served n32a4 plan's configs and recipe
# (gpt2_decode_python_n32a4_kc16r_lh's PLAN_CMD.txt: the ILP with one HiGHS thread, its tie-break) on n32sp's level grid.
# There a level counts 2 per CKKS level (pc-shadow ChainShape; the planner's PLAN_LEVEL_UNIT), the 14 levels above the
# user region (CtS, EvalMod) are n32a4's and the user region has 12 levels instead of 9, so every landing stays n32a4's
# and every level counted from the bottom is 6 deeper: MAX_LEVEL 54, the refresh envelopes 54 (StC-first's inputs 52),
# raise-dropped landings to 48, StC-first's entry 52. Raise drops keep EvalMod on its pairs: SparseB to a total drop of
# 3 and SPRU to 6 (deeper, EvalMod would run on the user region's 2^36 / 2^44.15 levels); no dense drop, so no K/V push
# raise. The runtime line is _lh's with CHAIN_PRESET=n32sp, the reactive ceiling and the input guard 6 deeper
# (AUTO_BTS_LEVEL=50, BTS_MAX_INPUT_LEVEL=54), SparseB's raise variants 1-3, no KV_PUSH_RAISE or
# FIDESLIB_CTS_ALT_BUDGET (dense drops only), and the CPU twin's StC-first entry at n32sp's (FAKECAP_STC_ENTRY=52):
# the twin lands a deliberate dense refresh (the K/V push) where the runtime's StC-first route does, not at the
# shipped order's landing the n32 entry (46) would give an input at 48, which the compile would then reach by a dense
# raise drop into the user region. Steps:
#   1. the position-0 capture under that runtime (run_decode --fake, as _lh's: FAKECAP_SPARSE_EXACT=1, the switches that
#      do not shape the op stream left out) and the merged capture, speedrun/make_decode_capture.sh (4 rows of the
#      OpenWebText calibration pool x positions 0-15) -> graphs/gpt2_decode_python_n32sp_kc16r_lhm
#   2. blocks 0-11; stage 12 alone at block 11's exit, its exit capped at effective level 52 (n32a4's 46, 6 deeper:
#      CutMax takes its first refresh two levels below its entry); block 0's feedback variant
#   3. stage 13's capture: the position-0 capture's CutMax with its magnitudes over the logit rows of
#      graphs/gpt2_decode_python_n32_cm/cutmax_rows.json (capture_mags --argmax --rows) and the merged capture's stage 12
#      -> graphs/gpt2_decode_python_n32sp_kc16r_lh_cm; stage 13 planned by speedrun/make_cutmax_plan.sh on the ILP
#   4. speedrun/make_n32a4_decode_rd.sh: the raise-drop search once more on the final program (no K/V push raise)
# RESUME=1 keeps each step's product and the blocks already planned (a stopped run).
#   bash speedrun/make_n32sp_decode_plan.sh      -> bootstrap_placements/gpt2_decode_python_n32sp_kc16r_lh (served)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/local_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1
REC=bootstrap_placements/gpt2_decode_python_n32a4_kc16r_lh
OUT=gpt2_decode_python_n32sp_kc16r_lh
CFG=configs/model/approximation/gpt2_base_n32_kc16r_cmfl_lh/configs.json
G0=graphs/${OUT}_p0 GM=graphs/${OUT}m G13=graphs/${OUT}_cm
CAP=52
RUNTIME="$(sed -n 's/^runtime=//p' "$REC/PLAN_CMD.txt" | sed -e 's/CHAIN_PRESET=n32a4/CHAIN_PRESET=n32sp/' \
    -e 's/AUTO_BTS_LEVEL=44/AUTO_BTS_LEVEL=50/' -e 's/BTS_MAX_INPUT_LEVEL=48/BTS_MAX_INPUT_LEVEL=54/' \
    -e 's/FIDESLIB_BTS_RAISE_DROPS=[^ ]*/FIDESLIB_BTS_RAISE_DROPS=1,2,3/' -e 's/ KV_PUSH_RAISE=[^ ]*//' \
    -e 's/ FIDESLIB_CTS_ALT_BUDGET=[^ ]*//' -e 's/  */ /g') FAKECAP_STC_ENTRY=52"
# the twin reads its quirks from the environment (a --set reaches the session's Config only): the captures' twin too
export FAKECAP_STC_ENTRY=52
LEVELS="MAX_LEVEL=54 PLAN_REFRESH_ENV_CAP=54 PLAN_REFRESH_ENV_CAP_STEPS=cutmax:54,tail:54 PLAN_REFRESH_ENV_CAP_ROUTES=0:52 \
PLAN_RAISE_LANDING_MAX=48 PLAN_STC_FIRST_ENTRY=52 PLAN_RAISE_ROUTES=512,1 PLAN_RAISE_DROP_SET=1,2,3 PLAN_RAISE_DROP_MAX=6"
RECIPE="$(sed -n 's/^recipe=//p' "$REC/PLAN_CMD.txt") $LEVELS"
resume() { [ "${RESUME:-0}" = 1 ]; }
# 1. captures
SETS=()
for kv in $RUNTIME; do
    case $kv in FIDESLIB_BTS_SPARSE_B_K=*|PC_BTS_MERGE_S2D=*|PC_BTS_STC_PS=*|PC_EM_LEVEL_KEYS=*) ;; *) SETS+=(--set "$kv") ;; esac
done
if ! resume || [ ! -f "$G0/block_13/graph.json" ]; then
    rm -rf "$G0"
    FAKECAP_SPARSE_EXACT=1 "$PYTHON" -m perseus.examples.gpt2.run_decode --fake --tokens 1 --argmax --capture "$G0" \
        --configs "$CFG" "${SETS[@]}"
fi
STUB="$(mktemp -d bootstrap_placements/_stub_XXXX)"
trap 'rm -rf "$STUB"' EXIT
printf 'runtime=%s\nconfigs=%s\n' "$RUNTIME" "$CFG" > "$STUB/PLAN_CMD.txt"
if ! resume || [ ! -f "$GM/block_13/graph.json" ]; then
    BASE="$G0" PLAN="$STUB" bash speedrun/make_decode_capture.sh "$GM"
fi
# 2. blocks 0-11, stage 12 alone, feedback
resume || rm -rf "bootstrap_placements/$OUT"
TMP="$(mktemp -d graphs/_b011_XXXX)"
for b in $(seq 0 11); do ln -s "$PWD/$GM/block_$b" "$TMP/block_$b"; done
cp "$GM/capture_env.json" "$TMP/"
# shellcheck disable=SC2086
env $RECIPE PLAN_RESUME="${RESUME:-0}" bash python/perseus/examples/gpt2/make_plan.sh "$TMP" "$OUT" blocks
rm -rf "$TMP"
if ! resume || [ ! -f "bootstrap_placements/$OUT/block_12_placement.json" ]; then
    read -r L D < <("$PYTHON" -c 'import json, sys; s = json.load(open(sys.argv[1]))["summary"]
print(s["exit_level"], s.get("exit_deg") or 1)' "bootstrap_placements/$OUT/block_11_placement.json")
    TMP="$(mktemp -d graphs/_b12_XXXX)"
    ln -s "$PWD/$GM/block_12" "$TMP/block_12" && cp "$GM/capture_env.json" "$TMP/"
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
    rm -rf "bootstrap_placements/${OUT}_b12"
fi
# shellcheck disable=SC2086
env $RECIPE bash python/perseus/examples/gpt2/make_plan.sh "$GM" "$OUT" feedback
# the provenance lines (tool, the configs' description, acc_table, oracle, capture, configs_from, solver, capture13)
{ echo "graph=$GM"
  cat <<'EOF'
tool=speedrun/make_n32sp_decode_plan.sh: blocks 0-11 by make_plan.sh blocks (the recipe below, ILP) on graphs/gpt2_decode_python_n32sp_kc16r_lhm's blocks 0-11; stage 12 alone at block 11's exit (48:1) with PLAN_ILP_EXIT_CAP=52 (exit 44:2); make_plan.sh feedback; stage 13 by make_cutmax_plan.sh (the cutmax= line); then the rd= pass. The recipe is gpt2_decode_python_n32a4_kc16r_lh's with the levels counted from the bottom 6 units deeper on n32sp's level grid (its last 9 knobs): every landing is n32a4's
EOF
  echo "recipe=$RECIPE"
  echo "runtime=$RUNTIME"
  cat <<'EOF'
configs=configs/model/approximation/gpt2_base_n32_kc16r_cmfl_lh/configs.json  (_cmfl's with key_lanes 16 on every softmax; the LayerNorm statistics from one 1-slot refresh (spru) on ln_2 of blocks 1, 2, 3, 7, 10 and ln_1 of blocks 2, 3, 4, 10, 11, on one complex ladder (zladder) on the other 14 block LNs; ln_f as _cmfl)
acc_table=8dc653fa405969c75f3804a0edfdfe80
oracle=the capture's input (ALL_BLOCKS_IO_DIR): the repo's oracle row (scripts/utils/gen_gpt2_oracle.py, $PERSEUS_DATA/oracle/gpt2/all_blocks_io), as for gpt2_decode_python_n32_kc16
capture=graphs/gpt2_decode_python_n32sp_kc16r_lh_p0 (not kept): FAKECAP_SPARSE_EXACT=1 FAKECAP_STC_ENTRY=52 run_decode --fake --tokens 1 --argmax --capture with gpt2_base_n32_kc16r_cmfl_lh and this runtime= line less FIDESLIB_BTS_SPARSE_B_K and the key-set switches (PC_BTS_MERGE_S2D, PC_BTS_STC_PS, PC_EM_LEVEL_KEYS) as --set (position 0 of the oracle row; the same command on gpt2_decode_python_n32a4_kc16r_lh's runtime gives graphs/gpt2_decode_python_n32_kc16r_lh's structure, node for node); graphs/gpt2_decode_python_n32sp_kc16r_lhm: FAKECAP_STC_ENTRY=52 BASE=<that capture> PLAN=<a plan dir with this runtime= and configs=> bash speedrun/make_decode_capture.sh (max over the 4 pool rows x positions 0-15)
configs_from=python scripts/utils/fit_gpt2_decode16_ref.py --out python/perseus/configs/model/approximation/gpt2_base_n32_kc16r_cm/configs.json (gpt2_base_n32_kc16_cm with the softmax of blocks 0-7, 9 and 10 centred on per-head reference keys, fitted on OpenWebText shard 0042)
solver=scipy 1.18.1 milp (HiGHS 1.12.0; numpy 2.5.4, Python 3.12.15), one HiGHS thread, real _pc; make_n32sp_decode_plan.sh's chain with the recipe= line. In every block the ILP and its tie-break are proven optimal (an unproven solve stops the planner); on one Xeon core stage 12 takes ~2 min of solver time, every other stage < 30 s, inside the recipe's PLAN_ILP_TIME_LIMIT=3600. Tested with these versions only (pyproject allows scipy>=1.17): another scipy / HiGHS may stop the planner on an unproven solve or, untested, return another plan; regenerate with these
capture13=graphs/gpt2_decode_python_n32sp_kc16r_lh_cm: the position-0 capture's stage 13, its magnitudes capture_mags --argmax --rows (the logit rows of graphs/gpt2_decode_python_n32_cm/cutmax_rows.json) --blocks 13, and the merged capture's stage 12
EOF
} > "bootstrap_placements/$OUT/PLAN_CMD.txt"
# 3. stage 13
if ! resume || [ ! -f "$G13/block_13/graph.json" ]; then
    rm -rf "$G13" && mkdir -p "$G13"
    cp -r "$G0/block_13" "$G0/capture_env.json" "$G13/"
    [ ! -f "$G0/capture_config.json" ] || cp "$G0/capture_config.json" "$G13/"
    cp -r "$GM/block_12" "$G13/"
    ROWS="$(mktemp -d)/rows.npy"
    "$PYTHON" scripts/utils/cutmax_rows.py graphs/gpt2_decode_python_n32_cm/cutmax_rows.json "$ROWS"
    "$PYTHON" -m perseus.examples.gpt2.capture_mags "$G13" --argmax --rows "$ROWS" --blocks 13 \
        --plan "bootstrap_placements/$OUT" --configs "$CFG"
    rm -rf "$(dirname "$ROWS")"
fi
CUTMAX_PLACER=ilp CUTMAX_GRAPH="$G13" bash speedrun/make_cutmax_plan.sh "$OUT" "${OUT}_cm" "$CFG"
rm -rf "bootstrap_placements/$OUT" && mv "bootstrap_placements/${OUT}_cm" "bootstrap_placements/$OUT"
# 4. the raise-drop re-pass
bash speedrun/make_n32a4_decode_rd.sh "$OUT"
"$PYTHON" - bootstrap_placements "$REC" "$OUT" <<'PY'
import json, sys
root, *names = sys.argv[1:]
for name in names:
    d = name if name.startswith(root) else f"{root}/{name}"
    tot = [json.load(open(f"{d}/block_{b}_placement.json"))["summary"]["total_bootstraps"] for b in range(14)]
    print(f"{name}: refreshes {tot}, stages 0-12 {sum(tot[:13])}, all {sum(tot)}")
PY
