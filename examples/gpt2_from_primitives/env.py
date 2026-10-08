"""Session setup for the port: the C++ n32 decode preset (scripts/run_task.sh), sync mode;
sparse bootstrap routes on, the softmax denominator reduced by its rotation ladder (no fold).
Exported BEFORE ``perseus._core`` is imported so the runtime and the C++ composites used as
references in tests see the same graph.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path

# GPT2_FOLD_LN1/LN2 = 1, LNF = 0: the scripts/run_task.sh defaults.
# CHAIN is not here: it is `export_env`'s argument, because everything chain-specific (level
# counts, correction factor, rotation band, lm-head cap) comes from scripts/local_env.sh, which
# has to be sourced for the SAME chain.
ENV = {
    "CKKS_COMPLEX": "1",
    # sparse bootstrap routes as in the C++ decode: keys for 512 and 1 slots, automatic
    # routing of periodic payloads at reactive sites, plan-driven routing at planted ones
    "SPARSE_AUTO": "2",
    "SPARSE_BTS_SLOTS": "512,1",
    "SPARSE_LN_BTS": "0",
    "SPARSE_SM_BTS": "0",
    "CUTMAX_SPARSE_BTS": "0",
    # the fused reductions (a ladder finished inside a sparse refresh) are off: the softmax
    # denominator is reduced by its rotation ladder, the LayerNorm variance likewise
    "FUSED_SM_DEN": "0",
    "FUSED_LN_VAR": "0",
    # the softmax divides through 1/denominator built on the denominator's own ciphertext
    # (attention.softmax_thor); off by default, a plan cut for it turns it on in its runtime= line
    "SM_DEN_RECIP": "0",
    # the GELU's 1/xmax folded into the up-projection weights, its refreshes left to the plan
    # (activation.gelu); off by default, a plan cut for it turns it on in its runtime= line
    "GELU_FOLD": "0",
    "FHE_PT_COEFF_ENCODE": "0",
    "GPT2_FOLD_LN1": "1",
    "GPT2_FOLD_LN2": "1",
    "GPT2_FOLD_LNF": "0",
    "GPT2_INFERENCE_MODE": "sync",
    # the C++ decode's CutMax refresh policy (scripts/run_task.sh): single-iteration bootstraps
    "CUTMAX_VEC_BTS_ITERS": "1",
    "CUTMAX_PRECISE_SCOPED": "1",
    "OMP_NUM_THREADS": "16",
}


# Bootstrap levers that move where a refresh LANDS are bound to the plan, which was cut for those landings: the
# exact post-raise scaling (FIDESLIB_BTS_SHIFT, eprint 2025/1403) and SPRU on the 1-slot route (FIDESLIB_SPRU = h,
# Coron-Koestler arXiv 2607.27401; Python bootstrap only). A plan declares them in a `runtime=` line of its
# PLAN_CMD.txt; a plan without one predates them and runs with both off. Eager (no plan) on n32: both on.
RUNTIME_N32 = {"FIDESLIB_BTS_SHIFT": "1", "FIDESLIB_SPRU": "64"}
RUNTIME_LEGACY = {"FIDESLIB_BTS_SHIFT": "0", "FIDESLIB_SPRU": "0"}


def plan_runtime(plan=None, chain="n32"):
    """The landing-bound bootstrap levers for `plan` (a plan dir or None) on `chain`."""
    if chain != "n32":
        return dict(RUNTIME_LEGACY)
    if not plan:
        return dict(RUNTIME_N32)
    cmd = Path(plan) / "PLAN_CMD.txt"
    if cmd.exists():
        for line in cmd.read_text().splitlines():
            if line.startswith("runtime="):
                return dict(kv.split("=", 1) for kv in shlex.split(line[len("runtime="):]))
    return dict(RUNTIME_LEGACY)


def bootstrap_setup(fhe, runtime, cpp_bootstrap=False):
    """Install the Python-orchestrated bootstrap (the default) with SPRU on the 1-slot route when the runtime asks
    for it. A SPRU plan cannot run on the C++ bootstrap (its 1-slot refreshes land 16 primes lower there)."""
    spru = int(runtime.get("FIDESLIB_SPRU", "0") or 0)
    if cpp_bootstrap:
        if spru:
            raise SystemExit("this plan needs SPRU on the 1-slot route, which runs only on the Python bootstrap: "
                             "drop --cpp-bootstrap or pass --set FIDESLIB_SPRU=0 with a non-SPRU plan")
        return
    from perseus.impl import bootstrap as _pyb
    _pyb.install(fhe=fhe, spru_h=spru)


def export_env(device=3, chain=None, **overrides):
    """Force the port's env (exported values WIN over scripts/local_env.sh).

    `chain` selects the CKKS chain ("n32" or "n64"); unset takes whatever CHAIN the shell
    already exported, so a run that sourced local_env.sh for a chain keeps it.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    chain = chain or os.environ.get("CHAIN") or "n32"
    for k, v in {"CHAIN": chain, **ENV,
                 **{k: str(v) for k, v in overrides.items()}}.items():
        os.environ[k] = v
    # never exceed the shared-box thread cap
    if int(os.environ.get("OMP_NUM_THREADS", "16")) > 36:
        os.environ["OMP_NUM_THREADS"] = "36"


def open_session(device=3, complex_payload=True, chain=None, **overrides):
    """perseus.session on the gpt2_decode_n32 preset with the overrides above.
    `complex_payload`: CKKS_COMPLEX=1, the C++ decode configuration (K/V pair bootstrap, packed
    CutMax, complex feedback tile; weights and linears stay real); False: real slots only."""
    chain = chain or os.environ.get("CHAIN") or "n32"
    overrides = {"CKKS_COMPLEX": "1" if complex_payload else "0", **overrides}
    export_env(device, chain=chain, **overrides)
    from perseus import session
    from perseus.profile import SessionProfile
    prof = (SessionProfile.gpt2_decode_n64() if chain == "n64"
            else SessionProfile.gpt2_decode_n32())
    prof.ckks_complex = bool(complex_payload)
    prof.mode = "sync"
    prof.extra = dict(prof.extra)
    prof.extra.update({k: v for k, v in ENV.items()})
    prof.extra.update({k: str(v) for k, v in overrides.items()})
    return session(profile=prof, mode="sync")


def rotation_audit(fhe, dims):
    """Which of the port's rotation steps are missing from the loaded band (empty = ok)."""
    from perseus.impl.layout import attention_rot_steps, linear_rot_steps
    need = set()
    for d_in, d_out in ((dims.hid, dims.hid), (dims.hid, dims.E), (dims.E, dims.hid),
                        (dims.hid, dims.N), (dims.N, dims.hid)):
        need |= linear_rot_steps(dims.N, d_in, d_out)
    need |= attention_rot_steps(dims)
    loaded = {int(s) % dims.N for s in fhe.loaded_rot_steps}
    return sorted(need - loaded)
