#!/usr/bin/env bash
# Wait for SLURM job(s) to leave the queue — WITHOUT mistaking a controller outage
# for completion.
#
# The naive poll
#     while squeue -j $J -h | grep -q .; do sleep 30; done
# is wrong: when slurmctld is unreachable (it happens on this cluster — `squeue`
# hangs and `sacct` returns "No route to host"), squeue prints NOTHING and exits
# non-zero, so the loop reads "job gone" and returns instantly. A monitor built on
# it reports a job as finished seconds after submission and the caller then reads a
# stale/absent output file (hit 2026-07-25 on the BERT recalibration, job 50203377).
#
# Contract: only a SUCCESSFUL squeue that does not list the job counts as done.
# A failed/timed-out squeue is "unknown" -> keep waiting.
#
# Usage: bash scripts/utils/wait_job.sh <jobid> [jobid...]
#   env: POLL=30 (seconds)  TIMEOUT=0 (seconds; 0 = no cap)  QTMO=30 (squeue timeout)
set -u
POLL="${POLL:-30}"; TIMEOUT="${TIMEOUT:-0}"; QTMO="${QTMO:-30}"
start=$SECONDS
unknown=0

still_queued() {   # 0 = queued, 1 = definitely gone, 2 = unknown (controller down)
    local out rc
    out=$(timeout "$QTMO" squeue -j "$1" -h -o "%i" 2>/dev/null); rc=$?
    [ $rc -ne 0 ] && return 2
    [ -n "$(printf '%s' "$out" | tr -d '[:space:]')" ] && return 0
    return 1
}

for J in "$@"; do
    while :; do
        still_queued "$J"; s=$?
        if   [ $s -eq 1 ]; then echo "[wait] job $J left the queue"; break
        elif [ $s -eq 2 ]; then
            unknown=$((unknown+1))
            [ $((unknown % 10)) -eq 1 ] && \
                echo "[wait] job $J: squeue unavailable (slurmctld?), still waiting" >&2
        fi
        if [ "$TIMEOUT" -gt 0 ] && [ $((SECONDS-start)) -ge "$TIMEOUT" ]; then
            echo "[wait] TIMEOUT after ${TIMEOUT}s waiting on $J" >&2; exit 2
        fi
        sleep "$POLL"
    done
done
[ $unknown -gt 0 ] && echo "[wait] note: $unknown poll(s) could not reach slurmctld" >&2
exit 0
