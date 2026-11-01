#!/bin/bash
# A Prefill-128 plan's decisions carried onto a new capture (CPU only), the chain of _cmw_mf, _cmw_mfsab: the planning
# capture of make_prefill_plan.sh's four rows with CONFIGS (the served capture's PCMM band shifts reused: the configs
# change no linear), BASE's decisions (a pfbh-level plan on BASE_GRAPH) mapped onto it by scripts/utils/map_plan_decisions.py
# (each step's ops aligned; an unmapped decision stops), the served raise-drop pass (RD_SRC's recipe: the decisions kept
# exactly, the drops searched) with the PCMM finishes landed where PF_PCMM_RAISE / PF_PCMM_RAISE_OUT drop them
# (RD_SRC's runtime values by default; the plan's runtime line gets the ones used), then speedrun/make_prefill_wrot_plan.sh
# with the served switches. Stages 12-13 are RD_SRC's. ROUTES_FREE=1 lets the pass re-pick a refresh's route (reported).
#   NAME=mfsab2 CONFIGS=gpt2_base_n32_pf_mf_s16_ab2 CM_CONFIGS=gpt2_base_n32_pf_cm_mf_s16_ab2 \
#   PF_PCMM_RAISE=up:3:2-2,up:3:11-11,up:2:0-1,up:2:3-10,up:1,qkv:5:11-11,qkv:1:3-3 bash speedrun/make_prefill_carry_plan.sh
#   -> graphs/gpt2_prefill128_pfbh_$NAME, bootstrap_placements/gpt2_prefill128_n32a4_pfbh{,rd_cm,rd_cmw}_$NAME
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/local_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
: "${NAME:?}" "${CONFIGS:?}" "${CM_CONFIGS:?}"
BASE=${BASE:-gpt2_prefill128_n32a4_pfbh_mfs} BASE_GRAPH=${BASE_GRAPH:-graphs/gpt2_prefill128_pfbh_mfs}
RD_SRC=${RD_SRC:-gpt2_prefill128_n32a4_pfbhrd_cm_mf}
GR=graphs/gpt2_prefill128_pfbh_$NAME KEEP=gpt2_prefill128_n32a4_pfbh_$NAME
RD=gpt2_prefill128_n32a4_pfbhrd_cm_$NAME OUT=gpt2_prefill128_n32a4_pfbhrd_cmw_$NAME
A=configs/model/approximation
src_rt() { sed -n 's/^runtime=//p' "bootstrap_placements/$RD_SRC/PLAN_CMD.txt" | tr ' ' '\n' | sed -n "s/^$1=//p"; }
RAISE=${PF_PCMM_RAISE:-$(src_rt PF_PCMM_RAISE)} RAISE_OUT=${PF_PCMM_RAISE_OUT:-$(src_rt PF_PCMM_RAISE_OUT)}
# the drops are the raise-drop pass's alone: the capture and the WROT recording run on the undropped landings (the
# latter takes them from the plan's runtime line)
unset PF_PCMM_RAISE PF_PCMM_RAISE_OUT

# 1. the capture (make_prefill_plan.sh's runtime and rows)
RUNTIME="$(sed -n 's/^runtime=//p' bootstrap_placements/gpt2_prefill128_m16s0/PLAN_CMD.txt) PF_LAST_QUERY=1 \
FIDESLIB_BTS_RAISE_DROP=512:1 SM_CHEB_GS=1 PF_QKV_PACK=1 PF_ATTN_PACK=3 PF_ATTN_BSGS=1 PF_SM_HPACK=1"
SETS=$(echo "$RUNTIME" | tr ' ' '\n' | grep = | sed 's/^/--set /')
ROWS=${ROWS:-4096 8192 12288 16384}
CALIB=graphs/gpt2_prefill128_pfbh/pcmm_calib.json
TMP="$(mktemp -d graphs/_carry_XXXX)"
pids=()
for r in $ROWS; do
    [ -d "$PERSEUS_DATA/oracle/gpt2/owt_$r" ] || { echo "no oracle row $r: speedrun/make_prefill_plan.sh makes it" >&2; exit 1; }
    ALL_BLOCKS_IO_DIR="$PERSEUS_DATA/oracle/gpt2/owt_$r" PF_PCMM_CALIB_IN=$CALIB "$PYTHON" -m perseus.examples.gpt2.run_prefill \
        --tokens 128 --fake --tail --capture "$TMP/r$r" --configs "$A/$CONFIGS/configs.json" $SETS > "$TMP/r$r.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || { echo "a row's capture failed: $TMP/r*.log" >&2; exit 1; }; done
rm -rf "$GR"
first=${ROWS%% *}
"$PYTHON" scripts/utils/merge_mags.py "$TMP/r$first" "$GR" $(for r in $ROWS; do echo "$TMP/r$r"; done) > /dev/null
cp "$CALIB" "$GR/"
rm -rf "$TMP"

# 2. BASE's decisions on it
rm -rf "bootstrap_placements/$KEEP"
"$PYTHON" scripts/utils/map_plan_decisions.py "$BASE_GRAPH" "bootstrap_placements/$BASE" "$GR" "bootstrap_placements/$KEEP"

# 3. the raise-drop pass on the capture with the PCMM finishes landed lower
DROP="$GR.pcmmrd" OUTB=_carry_$NAME
rm -rf "$DROP" && cp -r "$GR" "$DROP"
PF_PCMM_RAISE=$RAISE PF_PCMM_RAISE_OUT=$RAISE_OUT "$PYTHON" - "$DROP" <<'PY'
import json, sys, types
from perseus.impl import pcmm
d, n_ed = sys.argv[1], 0
for b in range(12):
    f = f"{d}/block_{b}/graph.json"
    g = json.load(open(f))
    for n in g["nodes"]:
        ext = (n.get("args") or {}).get("external") or {}
        if n["op_type"] != "deliberate_bootstrap" or ext.get("op") != "pcmm":
            continue
        k = pcmm._drop_of(types.SimpleNamespace(name=ext["linear"], n_out=ext["n_out"]), ext["out"])
        if k:
            ext["raise_drop"] = k
            n["output_level"] += 2 * k
            n["args"]["landing"][0] += 2 * k
            n["shadow"]["level"] += 2 * k
            n_ed += 1
    json.dump(g, open(f, "w"))
print(f"{d}: {n_ed} PCMM finishes dropped")
PY
RECIPE="$(sed -n 's/^recipe=//p' "bootstrap_placements/$RD_SRC/PLAN_CMD.txt")"
rm -rf "bootstrap_placements/$OUTB"
B="$(mktemp -d graphs/_blocks_XXXX)"
for b in $(seq 0 11); do ln -s "$PWD/$DROP/block_$b" "$B/block_$b"; done
cp graphs/gpt2_prefill128_pfbh/capture_env.json "$B/"
# shellcheck disable=SC2086
env $RECIPE PF_PCMM_RAISE="$RAISE" PF_PCMM_RAISE_OUT="$RAISE_OUT" PLAN_KEEP_PLACEMENTS="bootstrap_placements/$KEEP" bash python/perseus/examples/gpt2/make_plan.sh "$B" "$OUTB" blocks \
    > "bootstrap_placements/$OUTB.log" 2>&1 || { tail -5 "bootstrap_placements/$OUTB.log" >&2; exit 1; }
rm -rf "$B" "bootstrap_placements/$OUTB.log"
"$PYTHON" - "bootstrap_placements/$KEEP" "bootstrap_placements/$OUTB" "bootstrap_placements/$RD_SRC" "${ROUTES_FREE:-0}" <<'PY'
import json, sys
k, b, s, free = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
keys = ("placements", "hint_fire", "real_route") + (() if free else ("sparse_slots",))
for i in range(12):
    x, y, z = (json.load(open(f"{d}/block_{i}_placement.json")) for d in (k, b, s))
    bad = [f for f in keys if x.get(f) != y.get(f)]
    ex = lambda q: (q["summary"]["exit_level"], q["summary"].get("exit_deg") or 1)
    if bad or ex(y) != ex(z):
        sys.exit(f"block {i}: changed {bad or ('exit', ex(y), ex(z))}")
    rr = {v: (x["sparse_slots"].get(v), y["sparse_slots"].get(v)) for v in set(x["sparse_slots"]) | set(y["sparse_slots"])
          if x["sparse_slots"].get(v) != y["sparse_slots"].get(v)}
    print(f"block {i}: drops {sum((y.get('raise_drop') or {}).values())}" + (f", routes re-picked {rr}" if rr else ""))
PY
rm -rf "bootstrap_placements/$RD" && cp -r "bootstrap_placements/$RD_SRC" "bootstrap_placements/$RD"
for b in $(seq 0 11); do cp "bootstrap_placements/$OUTB/block_${b}_placement.json" "bootstrap_placements/$RD/"; done
rm -rf "bootstrap_placements/$OUTB" "$DROP"
sed -i -e "s#^graph=$(sed -n 's/^graph=\([^ ]*\).*/\1/p' "bootstrap_placements/$RD_SRC/PLAN_CMD.txt") #graph=$GR #" \
    -e "s#^configs=.*#configs=$A/$CM_CONFIGS/configs.json#" \
    -e "/^runtime=/s#PF_PCMM_RAISE=[^ ]*#PF_PCMM_RAISE=$RAISE#" \
    -e "/^runtime=/s#PF_PCMM_RAISE_OUT=[^ ]*#PF_PCMM_RAISE_OUT=$RAISE_OUT#" "bootstrap_placements/$RD/PLAN_CMD.txt"
echo "carry=blocks 0-11 by speedrun/make_prefill_carry_plan.sh: $BASE's decisions carried onto $GR (configs $CONFIGS, \
make_prefill_plan.sh's rows, $CALIB's band shifts) by scripts/utils/map_plan_decisions.py, then $RD_SRC's raise-drop pass \
(decisions kept exact$([ "${ROUTES_FREE:-0}" = 1 ] && echo ", routes free")) with PF_PCMM_RAISE=$RAISE PF_PCMM_RAISE_OUT=$RAISE_OUT\
${NOTE:+; $NOTE}" >> "bootstrap_placements/$RD/PLAN_CMD.txt"

# 4. the WROT carry-over with the served switches
BASE=$RD OUT=$OUT WROT=11 PAIR_WINDOW=6144 MULT_SUM=1 EXTRA="PC_BTS_PAIR_LOOSE=1 PC_BTS_PAIR_CF_MAX=15" \
    bash speedrun/make_prefill_wrot_plan.sh | tail -2
echo "-> bootstrap_placements/$OUT"
