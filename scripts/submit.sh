#!/usr/bin/env bash
# =============================================================================
#  submit.sh — submit matrix jobs (login-side, CLEAN shell: no modules loaded).
#
#    bash scripts/submit.sh [--model gpt2|vit|all] [--task LIST] [--stage S]
#                           [--runner python|cuda] [--time H:M:S] [--cpus N]
#                           [--dry-run]
#
#    --task    which case(s): decode gen gen_prefill handoff prefill32
#              prefill96 prefill128 vit80 vit112  (comma/space list; aliases:
#              genp, pfx32, pfx96, pfx128)
#    --model   all tasks of a family when --task is omitted: gpt2 | vit | all
#    --stage   run (planned, default) | eager | capture
#    --runner  python (default, gate-authoritative) | cuda (native CLI, GPT-2 only)
#    --time    sbatch walltime override      --cpus  cores-per-gpu override
#    --dry-run print the sbatch commands instead of submitting
#
#  Examples:
#    bash scripts/submit.sh                                # full planned matrix
#    bash scripts/submit.sh --model vit                    # vit80 + vit112 runs
#    bash scripts/submit.sh --task decode,gen              # two runs
#    bash scripts/submit.sh --task prefill128 --stage capture   # ranged pair (chained)
#    bash scripts/submit.sh --task decode --runner cuda --stage eager
#
#  Per-task knobs live as env presets in run_task.sh; this script only sizes and
#  submits. prefill128 capture auto-chains the resumable ranged pair (T=64
#  chunk_0 template, then T=128 with chunk_0 pre-seeded).
# =============================================================================
set -e -o pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
ACCOUNT="-A euhpc_d34_099"

GPT2_TASKS="decode gen prefill128 prefill96 prefill32 handoff gen_prefill"
VIT_TASKS="vit80 vit112"

MODEL=all; TASKS=""; STAGE=run; RUNNER=python; TIME_OVR=""; CPUS_OVR=""; DRY=0
usage() { sed -n '3,27p' "$0" | sed 's/^# \{0,2\}//'; exit "${1:-0}"; }
while [ $# -gt 0 ]; do case "$1" in
    --model)  MODEL="$2"; shift 2 ;;
    --task)   TASKS="$TASKS ${2//,/ }"; shift 2 ;;
    --stage)  STAGE="$2"; shift 2 ;;
    --runner) RUNNER="$2"; shift 2 ;;
    --time)   TIME_OVR="$2"; shift 2 ;;
    --cpus)   CPUS_OVR="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1"; usage 1 ;;
esac; done

if [ -z "$TASKS" ]; then case "$MODEL" in
    gpt2) TASKS="$GPT2_TASKS" ;;
    vit)  TASKS="$VIT_TASKS" ;;
    all)  TASKS="$GPT2_TASKS $VIT_TASKS" ;;
    *) echo "unknown --model $MODEL (gpt2|vit|all)"; exit 1 ;;
esac; fi

# canonical name + default sizing per task: "<task> <time> <cpus> <qos>"
# Post-adoption walls are all well under 30 min except gen_prefill -> everything
# else rides boost_qos_dbg (30-min cap, fast scheduling). Captures always run
# qos=normal (hours). --time overrides > 30 min force qos=normal automatically.
sizing() { case "$1" in
    genp|gen_prefill)   echo "gen_prefill 01:00:00 8  normal" ;;
    decode)             echo "decode      00:30:00 8  boost_qos_dbg" ;;
    gen)                echo "gen         00:30:00 8  boost_qos_dbg" ;;
    handoff)            echo "handoff     00:30:00 8  boost_qos_dbg" ;;
    pfx32|prefill32)    echo "prefill32   00:30:00 16 boost_qos_dbg" ;;
    pfx96|prefill96)    echo "prefill96   00:30:00 16 boost_qos_dbg" ;;
    pfx128|prefill128)  echo "prefill128  00:30:00 16 boost_qos_dbg" ;;
    vit80)              echo "vit80       00:30:00 16 boost_qos_dbg" ;;
    vit112)             echo "vit112      00:30:00 16 boost_qos_dbg" ;;
    *) return 1 ;;
esac; }

over_30min() {  # true when H:M:S exceeds 00:30:00
    local t="${1:-}"; [ -n "$t" ] || return 1
    local h="${t%%:*}" rest="${t#*:}"; local m="${rest%%:*}"
    [ $((10#$h * 60 + 10#$m)) -gt 30 ]
}

launch() {  # launch <jobname> <time> <cpus> <qos> [env…]  -> prints job id (or dry line)
    local name="$1" time="$2" cpus="$3" qos="$4"; shift 4
    local t="${TIME_OVR:-$time}"
    [ "$STAGE" = "capture" ] && qos=normal
    over_30min "$t" && qos=normal
    local cmd=(sbatch --parsable $ACCOUNT --qos="$qos" --time="$t" \
               --cpus-per-gpu="${CPUS_OVR:-$cpus}" --job-name="$name" scripts/run_task.sh)
    if [ "$DRY" = 1 ]; then echo "DRY: env $* ${cmd[*]}"; return; fi
    local jid
    if jid=$(env "$@" "${cmd[@]}" 2>/dev/null); then echo "$jid"; return; fi
    # boost_qos_dbg caps submitted jobs per user (QOSMaxSubmitJobPerUserLimit) —
    # overflow falls back to the normal queue transparently.
    if [ "$qos" = "boost_qos_dbg" ]; then
        cmd=(sbatch --parsable $ACCOUNT --qos=normal --time="$t" \
             --cpus-per-gpu="${CPUS_OVR:-$cpus}" --job-name="$name" scripts/run_task.sh)
        env "$@" "${cmd[@]}"
    else
        return 1
    fi
}

for raw in $TASKS; do
    read -r task time cpus qos <<< "$(sizing "$raw")" || { echo "unknown task: $raw"; exit 1; }
    if [ "$task" = "prefill96" ] && [ "$STAGE" = "capture" ]; then
        # T96 capture: pre-seed the packed chunk_0 from the T128 RANGED graph
        # (identical template — the greedy chunking makes 32<T<=64 and T96/T128
        # chunk_0 the same shape), then capture only the REAL tail chunk_1.
        SRC=.cache/graph_gpt2_prefill_delta_T128
        [ -d "$SRC/chunk_0" ] || SRC=.cache/graph_gpt2_prefill_delta_T128_rng
        [ -d "$SRC/chunk_0" ] || { echo "prefill96 capture needs the T128 ranged chunk_0 first"; exit 1; }
        rm -rf .cache/graph_gpt2_prefill_delta_T96
        mkdir -p .cache/graph_gpt2_prefill_delta_T96
        cp -r "$SRC/chunk_0" .cache/graph_gpt2_prefill_delta_T96/chunk_0
        echo "$task(capture): pre-seeded chunk_0 from $SRC; $(launch fhe_capture_prefill96 05:00:00 16 normal \
             TASK=prefill96 STAGE=capture RUNNER="$RUNNER" RESUME_GRAPH=1)"
    elif [ "$task" = "prefill128" ] && [ "$STAGE" = "capture" ]; then
        C0=$(launch fhe_capture_prefill128_c0 05:00:00 16 normal \
             TASK=prefill128 STAGE=capture RUNNER="$RUNNER" PREFILL_T=64)
        C1=$(launch fhe_capture_prefill128_c1 05:00:00 16 normal \
             TASK=prefill128 STAGE=capture RUNNER="$RUNNER" PREFILL_T=128 RESUME_GRAPH=1 \
             SBATCH_DEPENDENCY="afterany:$C0")
        # SBATCH_DEPENDENCY is honored by sbatch natively via the environment
        echo "prefill128(capture): chunk_0=$C0 chunk_1=$C1 (chained)"
    else
        # env overrides that change WHAT runs show up in the job name (a T64 run
        # under the prefill128 preset is named fhe_run_prefill128_T64, not _128)
        echo "$task($STAGE): $(launch "fhe_${STAGE}_${task}${PREFILL_T:+_T$PREFILL_T}" "$time" "$cpus" "$qos" \
             TASK="$task" STAGE="$STAGE" RUNNER="$RUNNER")"
    fi
done
