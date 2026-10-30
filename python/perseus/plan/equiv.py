#!/usr/bin/env python3
"""Compare a regenerated plan file against a committed one under the baseline contract.

    plan_equiv.py NEW.json REF.json      -> exit 0 if equivalent, 1 (with a reason) if not

Every top-level key except `summary` must be identical (canonical `json.dumps(sort_keys=True)` text: also catches
`1` vs `1.0` and `-0.0`); `summary` may only GROW: every committed key present with an identical value, recursively,
type-strict leaves. Diagnostics can thus be added without re-emitting the shipped plans.
"""
import json
import sys

# summary leaves that record wall time (the ILP's solve), never equal across two runs
WALL_CLOCK = {"solve_s"}
# the solver's float reports (placer_meta): the same optimum is summed in another order on another solver path, so they
# compare to a relative 1e-9
SOLVER_FLOATS = {"objective", "bound", "gap", "mincut_cost"}


def _diff_subset(ref, new, path="summary"):
    """First (path, reason) where `ref` is not a value-identical subset of `new`."""
    if isinstance(ref, dict):
        if not isinstance(new, dict):
            return path, "was an object"
        for k, v in ref.items():
            if k in WALL_CLOCK:
                continue
            if k not in new:
                return f"{path}.{k}", "missing"
            if (k in SOLVER_FLOATS and isinstance(v, float) and isinstance(new[k], float)
                    and abs(v - new[k]) <= 1e-9 * max(abs(v), abs(new[k]), 1.0)):
                continue
            r = _diff_subset(v, new[k], f"{path}.{k}")
            if r:
                return r
        return None
    if type(ref) is not type(new) or ref != new:
        return path, f"{ref!r} -> {new!r}"
    return None


def equivalent(new_path: str, ref_path: str):
    with open(new_path, encoding="utf-8") as f:
        new = json.load(f)
    with open(ref_path, encoding="utf-8") as f:
        ref = json.load(f)
    ref_body = {k: v for k, v in ref.items() if k != "summary"}
    new_body = {k: v for k, v in new.items() if k != "summary"}
    if json.dumps(ref_body, sort_keys=True) != json.dumps(new_body, sort_keys=True):
        keys = sorted(k for k in set(ref_body) | set(new_body)
                      if ref_body.get(k) != new_body.get(k))
        return False, f"runtime content differs: {', '.join(keys)}"
    r = _diff_subset(ref.get("summary", {}), new.get("summary", {}))
    if r:
        return False, f"summary regressed at {r[0]}: {r[1]}"
    return True, "ok"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ok, why = equivalent(sys.argv[1], sys.argv[2])
    if not ok:
        print(why)
    sys.exit(0 if ok else 1)
