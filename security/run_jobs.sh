#!/bin/bash
# run_jobs.sh <jobs file> <out.jsonl> [parallel jobs (default 1)]
# Each line of the jobs file is one estimate, single-threaded, no timeout:
#     [VAR=value ...] <estimate.py | hj_estimate.py> <arguments>
# ({out} in a line is replaced by the out file's path without .jsonl, e.g. SCANOUT={out}.scan.jsonl). Each job appends
# its JSON line to <out.jsonl>, its stderr to <out>.err and {"job", "rc", "wall_s"} to <out>.wall; <out>.DONE is
# written at the end. A job already listed in <out>.wall is skipped, so a killed run can be restarted.
# env SAGE: the sage executable (default: sage).
set -u
[ $# -ge 2 ] || { sed -n 2,8p "$0"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
JOBS="$(realpath "$1")"
OUT="$(realpath -m "$2")"
PAR="${3:-1}"
mkdir -p "$(dirname "$OUT")"
export HERE OUT SAGE="${SAGE:-sage}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

one() {
    local line="${1//\{out\}/${OUT%.jsonl}}"
    grep -qF "\"job\": \"$line\"" "$OUT.wall" 2>/dev/null && return 0
    local ev=() args
    read -r -a args <<< "$line"
    while [[ "${args[0]}" == *=* ]]; do ev+=("${args[0]}"); args=("${args[@]:1}"); done
    local st=$(date +%s) res rc
    res=$(cd "$HERE" && env "${ev[@]}" "$SAGE" -python "${args[@]}" 2>>"$OUT.err")
    rc=$?
    [ -n "$res" ] && echo "$res" >> "$OUT"
    echo "{\"job\": \"$line\", \"rc\": $rc, \"wall_s\": $(($(date +%s) - st))}" >> "$OUT.wall"
}
export -f one
grep -v '^\s*\(#\|$\)' "$JOBS" | tr '\n' '\0' | xargs -0 -P "$PAR" -I{} bash -c 'one "$1"' _ {}
echo "DONE $(date -Is)" > "$OUT.DONE"
