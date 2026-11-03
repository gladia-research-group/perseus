#!/bin/bash
# A/B-only stand-in for speedrun/runner/sandbox.sh, for a container rental (RunPod pods: no systemd user manager, user
# namespaces blocked by Docker's seccomp), copied over the runner of an out-of-repo copy of speedrun/ by run_gpu.sh
# (NOSANDBOX=1). Same arguments; it runs CMD directly, as close to the sandbox's view as a process can get without
# namespaces: an empty environment (PATH, HOME, LANG and the --setenv pairs), one GPU visible (CUDA_VISIBLE_DEVICES=N,
# so the server's --device 0 is GPU N, as the sandbox's one device node; none for the client), its own scratch
# directory as cwd and HOME (the sandbox's /work), the /out paths in CMD rewritten to --out, the wall-time limit, and
# NUMA pinning. NO isolation, no memory/task caps, no seccomp: the times and the KL are the harness's own code paths,
# but the runs are NOT harness runs and must never be reported as such.
set -euo pipefail
GPU= OUT= TIME=3600 NUMA= ENV=()
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --ro|--rw) shift 2 ;;
        --setenv) ENV+=("$2=$3"); shift 3 ;;
        --time) TIME=$2; shift 2 ;;
        --numa) NUMA=$2; shift 2 ;;
        --mem|--work|--cpus|--unit) shift 2 ;;
        --) shift; break ;;
        *) echo "sandbox.sh (A/B stand-in): unknown option $1" >&2; exit 2 ;;
    esac
done
[ -n "$GPU" ] && [ -n "$OUT" ] && [ $# -gt 0 ] || { echo "usage: sandbox.sh --gpu N --out DIR [--ro PATH]... -- CMD" >&2; exit 2; }
mkdir -p "$OUT"
WORK=$(mktemp -d "$(dirname "$OUT")/work-$(basename "$OUT").XXXX")
CMD=()
for x in "$@"; do
    case "$x" in
        /out) CMD+=("$OUT") ;;
        /out/*) CMD+=("$OUT/${x#/out/}") ;;
        *) CMD+=("$x") ;;
    esac
done
[ "$GPU" = none ] && VIS="" || VIS=$GPU
PIN=()
[ -n "$NUMA" ] && PIN=(numactl --cpunodebind="${NUMA%%:*}" --membind="${NUMA#*:}")
cd "$WORK"
# CMD in a session of its own, so a stop (run.py terminates this script) ends its whole process tree, as the sandbox's
# --die-with-parent and systemd unit do: a leftover server would hold the GPU through the next run
setsid env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin HOME="$WORK" LANG=C.UTF-8 TMPDIR="$WORK" \
    CUDA_VISIBLE_DEVICES="$VIS" "${ENV[@]}" timeout --signal=TERM --kill-after=30 "$TIME" "${PIN[@]}" "${CMD[@]}" &
CHILD=$!
trap 'kill -TERM -- -"$CHILD" 2>/dev/null; sleep 5; kill -KILL -- -"$CHILD" 2>/dev/null; exit 143' TERM INT
wait "$CHILD"
