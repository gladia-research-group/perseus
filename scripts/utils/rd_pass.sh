#!/bin/bash
# rd_pass.sh PLAN: the planner's raise-drop search once more on a finished plan's program (CPU only). Every stage keeps
# PLAN's program (PLAN_KEEP_PLACEMENTS_EXACT: its placements, fired hints and realize anchors, on the min-cut placer)
# and the search runs on the final program, where the slack the realize-anchor trials leave shows: refreshes land
# lower, none moves and no stage exit changes (checked). The stages are the graph's blocks but the argmax (block_13,
# planned on its own without raise drops), then block 0's feedback variant when PLAN has one. PLAN_CMD.txt gains an
# rd= line; python/tests/test_paper_plans.py runs the pass after a plan's regeneration when the line is there.
#   bash scripts/utils/rd_pass.sh gpt2_decode_ctx512_w512        -> bootstrap_placements/gpt2_decode_ctx512_w512 in place
# Optional, both landed on a copy of the graph before the pass (where the runtime then lands those refreshes):
#   KV_PUSH_RAISE=b:d,...     the K/V push refresh of block b raised d levels below its captured landing (the runtime
#                             switch of the same name; the plan's runtime= line gains it)
#   PF_PCMM_RAISE_OUT=...     the PCMM finishes' raise drop per linear output (impl/pcmm.py; the runtime= line gains it)
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/local_env.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
PLAN=${1:?plan dir under bootstrap_placements/}
CMD="bootstrap_placements/$PLAN/PLAN_CMD.txt"
GR=$(sed -n 's/^graph=\([^ ]*\).*/\1/p' "$CMD")
RECIPE="$(sed -n 's/^recipe=//p' "$CMD")"
[ -d "$GR" ] || { echo "rd_pass.sh: no graph $GR"; exit 1; }
OUT=_rd_$PLAN
rm -rf "bootstrap_placements/$OUT"
TMP="$(mktemp -d graphs/_rd_XXXX)"
trap 'rm -rf "$TMP" "bootstrap_placements/$OUT"' EXIT
# the graph the pass reads: links to GR's blocks, or edited copies of the blocks whose deliberate refreshes it lands
for b in "$GR"/block_*; do
    n=$(basename "$b")
    [ "$n" = block_13 ] && continue
    ln -s "$PWD/$b" "$TMP/$n"
done
cp "$GR/capture_env.json" "$TMP/"
if [ -n "${KV_PUSH_RAISE:-}${PF_PCMM_RAISE_OUT:-}" ]; then
    "$PYTHON" - "$TMP" "${KV_PUSH_RAISE:-}" <<'PY'
import json, os, shutil, sys, types
d, kv = sys.argv[1:]
from perseus.impl import pcmm
kvr = dict(tuple(int(x) for x in e.split(":")) for e in kv.split(",") if e)
n_kv = n_pc = 0
for name in sorted(os.listdir(d)):
    if not name.startswith("block_"):
        continue
    b = int(name.split("_")[1])
    f = os.path.join(d, name, "graph.json")
    g = json.load(open(f))
    hit = False
    for n in g["nodes"]:
        if n["op_type"] != "deliberate_bootstrap":
            continue
        ext = (n.get("args") or {}).get("external") or {}
        if b in kvr and n["step"].endswith("kv_push"):
            k = kvr[b]
            n["output_level"] += 2 * k
            n["shadow"]["level"] += 2 * k
            n["args"]["route"]["raise_drop"] = k
            n_kv += 1
            hit = True
        elif ext.get("op") == "pcmm" and os.environ.get("PF_PCMM_RAISE_OUT"):
            k = pcmm._drop_of(types.SimpleNamespace(name=ext["linear"], n_out=ext["n_out"]), ext["out"])
            if k and k != ext.get("raise_drop", 0):
                k0 = ext.get("raise_drop", 0)
                ext["raise_drop"] = k
                n["output_level"] += 2 * (k - k0)
                n["args"]["landing"][0] += 2 * (k - k0)
                n["shadow"]["level"] += 2 * (k - k0)
                n_pc += 1
                hit = True
    if hit:                       # a real copy of this block, the link replaced
        src = os.path.realpath(os.path.join(d, name))
        os.unlink(os.path.join(d, name))
        shutil.copytree(src, os.path.join(d, name))
        json.dump(g, open(f, "w"))
print(f"rd_pass: landed {n_kv} K/V push refreshes, {n_pc} PCMM finishes")
if kvr and n_kv != len(kvr):
    sys.exit(f"rd_pass: KV_PUSH_RAISE names {len(kvr)} blocks, {n_kv} K/V push refreshes found")
PY
fi
KEEP="PLAN_KEEP_PLACEMENTS=bootstrap_placements/$PLAN PLAN_KEEP_PLACEMENTS_EXACT=1 PLAN_PLACER=min_cut"
# shellcheck disable=SC2086
env $RECIPE $KEEP bash python/perseus/examples/gpt2/make_plan.sh "$TMP" "$OUT" blocks
if [ -f "bootstrap_placements/$PLAN/block_0_feedback_placement.json" ]; then
    # shellcheck disable=SC2086
    env $RECIPE $KEEP bash python/perseus/examples/gpt2/make_plan.sh "$TMP" "$OUT" feedback
fi
"$PYTHON" - "bootstrap_placements/$PLAN" "bootstrap_placements/$OUT" <<'PY'
import glob, json, os, sys
a, b = sys.argv[1:]
keys = ("placements", "hint_fire", "sparse_slots", "correction_factor", "real_route", "rescale_after")
ex = lambda s: (s["exit_level"], s.get("exit_deg") or 1)
tot = 0
for fb in sorted(glob.glob(os.path.join(b, "block_*_placement.json"))):
    x, y = (json.load(open(os.path.join(d, os.path.basename(fb)))) for d in (a, b))
    bad = [f for f in keys if x.get(f) != y.get(f)]
    if bad or ex(x["summary"]) != ex(y["summary"]):
        sys.exit(f"{os.path.basename(fb)}: the pass changed {bad or 'the exit'}")
    rx, ry = x.get("raise_drop", {}), y.get("raise_drop", {})
    moved = {k: (rx.get(k, 0), v) for k, v in ry.items() if rx.get(k, 0) != v}
    tot += len(moved)
    print(f"{os.path.basename(fb)}: {len(moved)} refreshes lower {moved if moved else ''}")
print(f"rd_pass: {tot} refreshes lower")
PY
for f in "bootstrap_placements/$OUT"/block_*_placement.json; do cp "$f" "bootstrap_placements/$PLAN/"; done
sed -i -e "/^rd=/d" -e "/^runtime=/s/ KV_PUSH_RAISE=[^ ]*//" -e "/^runtime=/s/ PF_PCMM_RAISE_OUT=[^ ]*//" "$CMD"
[ -z "${KV_PUSH_RAISE:-}" ] || sed -i "/^runtime=/s|\$| KV_PUSH_RAISE=$KV_PUSH_RAISE|" "$CMD"
[ -z "${PF_PCMM_RAISE_OUT:-}" ] || sed -i "/^runtime=/s|\$| PF_PCMM_RAISE_OUT=$PF_PCMM_RAISE_OUT|" "$CMD"
echo "rd=scripts/utils/rd_pass.sh${KV_PUSH_RAISE:+ KV_PUSH_RAISE=$KV_PUSH_RAISE}${PF_PCMM_RAISE_OUT:+ PF_PCMM_RAISE_OUT=$PF_PCMM_RAISE_OUT}" >> "$CMD"
