#!/bin/bash
# macro_run_retry.sh TAG [ENV=VAL ...]: scripts/macro_run.sh with a retry on the shared-worktree
# hazard -- another session's rebuild swaps perseus/_core.*.so for a few seconds and a launch in
# that window dies with `ImportError: cannot import name '_core'`. Never kill processes from
# here with a pattern that appears in this command line (pkill -f self-matches: exit 144).
TAG=$1; shift
for attempt in 1 2 3 4 5 6; do
  until [ -e perseus/_core.cpython-312-x86_64-linux-gnu.so ]; do sleep 10; done; sleep 15
  env "$@" TAG=$TAG bash scripts/macro_run.sh > /dev/null 2>&1
  grep -aq 'ImportError' "logs/macro/$TAG.err" 2>/dev/null || break
  echo "[$TAG] ImportError on attempt $attempt (module being relinked), retrying"; sleep 30
done
grep -a 'k=12 top1\|s/tok=\|forward\] PASS\|decode\] PASS' "logs/macro/$TAG.out" | tail -2
grep -a -o 'blocks_s=[0-9.]*.*unplanned_bts=[0-9]*' "logs/macro/$TAG.out" | tail -1
