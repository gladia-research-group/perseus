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
    # (attention.softmax_thor); a plan cut before it runs it off (RUNTIME_LEGACY)
    "SM_DEN_RECIP": "1",
    # the GELU's 1/xmax folded into the up-projection weights, its refreshes left to the plan
    # (activation.gelu); a plan cut before it runs it off (RUNTIME_LEGACY)
    "GELU_FOLD": "1",
    # the LayerNorm inverse sqrt from the config's Chebyshev seed (norm.norm) and the softmax's
    # first division at gs_iters_first; a plan cut before them runs them off (RUNTIME_LEGACY)
    "LN_CHEB": "1",
    "SM_GS_FIRST": "1",
    # plaintexts encoded on the GPU (FIDESlib DeviceEncode.cu): bit-identical to a reload of the host encode within
    # 1e-14, an evicted weight kept as its first composite group and lifted back on reload
    "FIDESLIB_GPU_ENCODE": "1",
    # the scores kept tH-periodic while the cache fits one group (attention.sm_periodic); a plan cut before it runs it
    # off (RUNTIME_LEGACY)
    "SM_PERIODIC": "1",
    # the softmax's constant products folded into masks (attention.sm_fold_affine, the reciprocal's seed); a plan cut
    # before it runs it off (RUNTIME_LEGACY)
    "SM_FOLD": "1",
    # the K + iV refresh as one real payload on two token lanes (attention.cache_kv_push_packed_complex); a plan cut
    # before it runs it off (RUNTIME_LEGACY)
    "KV_LANES": "1",
    # the Q linear output on the K/V push refresh and the qkv / up input hints moved past the limit (attention.mha,
    # block.transformer_block); a plan cut before it runs it off (RUNTIME_LEGACY)
    "QKV_LANES": "1",
    # the Chebyshev series by baby-step giant-step (poly.eval_chebyshev_bsgs); a plan cut before it runs it off
    # (RUNTIME_LEGACY)
    "CHEB_BSGS": "1",
    # the softmax head sum and its seed-scaled copy on one complex mask and one replicate ladder (attention.head_reduce_sum),
    # two CutMax cascade passes on one 1-slot refresh (head.inv_sigma_cascade), no dead graph ops (the odd GELU series'
    # T_6, the LayerNorm Newton constant at 0 iterations), rotate-and-sum ladders as hoisted radix-8 groups
    # (layout.rotsum); a plan cut before them runs them off (RUNTIME_LEGACY)
    "SM_HRS_PACK": "1",
    "CUTMAX_PAIR": "1",
    "IMPL_DEAD_OPS": "1",
    "LADDER_RADIX_BITS": "3",
    # step-scoped ciphertext names (perseus.impl Rt): a plan captured at one token stays bound at every cache
    # length (the q.K^T's per-group ops no longer renumber the block); with SM_PERIODIC=0, whose general softmax
    # path does not switch at t tokens. Off by default: the shipped plans name by the block-wide counter
    "IMPL_SCOPED_NAMES": "0",
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


# Switches a plan is bound to, because it was cut for them: the bootstrap levers that move where a refresh LANDS
# (the exact post-raise scaling FIDESLIB_BTS_SHIFT, eprint 2025/1403; SPRU on the 1-slot route FIDESLIB_SPRU = h,
# Coron-Koestler arXiv 2607.27401, Python bootstrap only), the real-payload dense route (FIDESLIB_BTS_REAL), the
# depth-1 CtS/StC of the 512-slot route (FIDESLIB_BTS_SPARSE_B, eprint 2026/1023; it lands 10 primes higher, so the
# model rotation keys are banded at 32, FIDESLIB_ROT_KEY_BAND), a route raised below the top modulus
# (FIDESLIB_BTS_RAISE_DROP, e.g. 512:1: that route lands one composite level lower) and the op sequences of the softmax,
# the GELU and the LayerNorm (SM_DEN_RECIP, SM_GS_FIRST, GELU_FOLD, LN_CHEB, KV_LANES). A plan declares them in a
# `runtime=` line of its PLAN_CMD.txt; one it does not declare predates it and runs at its RUNTIME_LEGACY value (a plan
# without the line runs all of them legacy). Eager (no plan) on n32: all on except the raise drop (plan-only).
RUNTIME_N32 = {"FIDESLIB_BTS_SHIFT": "1", "FIDESLIB_SPRU": "64", "FIDESLIB_BTS_REAL": "1",
               "FIDESLIB_BTS_SPARSE_B": "1", "FIDESLIB_ROT_KEY_BAND": "32"}
RUNTIME_LEGACY = {"FIDESLIB_BTS_SHIFT": "0", "FIDESLIB_SPRU": "0", "FIDESLIB_BTS_REAL": "0",
                  "SM_DEN_RECIP": "0", "GELU_FOLD": "0", "LN_CHEB": "0", "SM_GS_FIRST": "0", "SM_PERIODIC": "0",
                  "SM_FOLD": "0", "FIDESLIB_BTS_SPARSE_B": "0", "KV_LANES": "0", "CHEB_BSGS": "0",
                  "QKV_LANES": "0", "IMPL_SCOPED_NAMES": "0", "SM_HRS_PACK": "0", "CUTMAX_PAIR": "0",
                  "IMPL_DEAD_OPS": "0", "LADDER_RADIX_BITS": "0", "FIDESLIB_BTS_RAISE_DROP": "0",
                  "SM_CHEB_GS": "0"}


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
                return {**RUNTIME_LEGACY, **dict(kv.split("=", 1) for kv in shlex.split(line[len("runtime="):]))}
    return dict(RUNTIME_LEGACY)


# The approximation configs a plan was cut for. The n32 default (gpt2_base_n32) runs the GELU as one Chebyshev series
# per block; a plan declares its configs in a `configs=` line of its PLAN_CMD.txt, and an n32 plan without the line
# predates it and runs the composite (THOR) GELU of gpt2_base_n32_thor.
CONFIGS_N32 = "configs/model/approximation/gpt2_base_n32/configs.json"
CONFIGS_N32_LEGACY = "configs/model/approximation/gpt2_base_n32_thor/configs.json"


def plan_configs(plan=None, chain="n32", default=None):
    """The configs.json for `plan` (a plan dir or None) on `chain`; `default` when the plan is None or not n32."""
    if chain != "n32" or not plan:
        return default
    cmd = Path(plan) / "PLAN_CMD.txt"
    if cmd.exists():
        for line in cmd.read_text().splitlines():
            if line.startswith("configs="):
                return line[len("configs="):].split()[0]
    return CONFIGS_N32_LEGACY


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


def open_session(device=3, complex_payload=True, chain=None, extra_rot_steps=(), base=True, **overrides):
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
    from perseus.impl.layout import Dims, ladder_radix_bits, ladder_radix_steps
    steps = {int(s) for s in extra_rot_steps}
    if base and ladder_radix_bits() >= 2:
        steps |= ladder_radix_steps(Dims.gpt2(), ladder_radix_bits())
    if not steps:
        return session(profile=prof, mode="sync")
    prof.apply()
    opts = prof.options()
    ck = opts.ckks
    ck.extra_rot_steps = sorted(set(ck.extra_rot_steps) | steps)
    opts.ckks = ck
    return session(options=opts, mode="sync")


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
