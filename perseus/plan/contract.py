"""Plan/env contract: stamp capture-time environment into plan artifacts, validate at load.

A placement plan is strict against the op sequence and level schedule it was captured
under, but the knobs that shape both live in the environment (LN folding flips which
ops exist; AUTO_BTS_LEVEL moves every level). A plan loaded under a different env dies
deep in the run with a bare [plan_weight_error]/[plan_level_error]. The contract makes
the mismatch a load-time message that names the knob instead.

Scheduling-only knobs (GPT2_INFERENCE_MODE, the arena sizes, the rotation-key band) are
deliberately NOT in the contract: they move WHEN work happens, never what is computed.
"""
import json
import os

CONTRACT_KEY = "capture_env"

# Env vars that shape the captured op sequence or its level schedule.
LOAD_BEARING_ENV = (
    "CHAIN", "LOGN", "AUTO_BTS_LEVEL", "BTS_ITERATIONS",
    "CKKS_COMPLEX", "GPT2_PACKING", "GPT2_CACHE",
    "GPT2_FOLD_LN1", "GPT2_FOLD_LN2", "GPT2_FOLD_LNF",
    "CACHE_READ_LEVEL_K", "CACHE_READ_LEVEL_V", "FHE_LMHEAD_CAP",
    # routing and fold decisions: they change which bootstraps exist and at which degree
    "SPARSE_BTS_SLOTS", "SPARSE_AUTO", "FUSED_SM_DEN", "FUSED_LN_VAR", "CORRECTION_FACTOR",
    "FHE_PT_COEFF_ENCODE",
    # the softmax division form (impl attention.softmax_thor): a different op sequence
    "SM_DEN_RECIP",
)


class PlanContractError(RuntimeError):
    """A plan's capture-time contract does not match the current session."""


def capture_contract(inf=None):
    """The current session's contract dict (stamp this at capture time).

    Unset vars are stamped as null: set<->unset transitions change the op
    sequence exactly like value changes do (an unset fold flag IS a value)."""
    c = {"env": {k: os.environ.get(k) for k in LOAD_BEARING_ENV}}
    if inf is not None:
        c["session"] = {"level_limit": inf.fhe.level_limit(),
                        "slots": inf.slots, "logN": inf.logN}
    return c


def contract_mismatches(stamped, inf=None):
    """List of human-readable mismatches between a stamped contract and now."""
    out = []
    for k, v in (stamped.get("env") or {}).items():
        cur = os.environ.get(k)
        if cur != v:
            out.append(f"{k}: captured {v!r}, session has {cur!r}")
    sess = stamped.get("session") or {}
    if inf is not None:
        for key, cur in (("level_limit", inf.fhe.level_limit()),
                         ("slots", inf.slots), ("logN", inf.logN)):
            if key in sess and sess[key] != cur:
                out.append(f"{key}: captured {sess[key]}, session has {cur}")
    return out


def validate_contract(stamped, inf=None, source="plan", strict=True):
    """Raise (strict) or return the mismatch list. None/absent contract passes."""
    if not stamped:
        return []
    bad = contract_mismatches(stamped, inf)
    if bad and strict:
        raise PlanContractError(
            f"{source}: capture-time contract mismatch — the plan is op-sequence-"
            f"strict and will fail deep in the run. Fix the env or re-plan:\n  "
            + "\n  ".join(bad))
    return bad


def read_stamp(path):
    """The contract stamped in a placement/plan JSON file, or None."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get(CONTRACT_KEY)
    except (OSError, ValueError):
        return None


def stamp_file(path, contract):
    """Inject the contract into an existing plan/placement JSON, in place."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    doc[CONTRACT_KEY] = contract
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
