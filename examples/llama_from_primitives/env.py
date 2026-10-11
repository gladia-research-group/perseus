"""Session setup for the Llama port: the GPT-2 port's n32 decode env (examples/gpt2_from_primitives/
env.py: sparse routes, the softmax / norm switches, the plan-bound bootstrap levers) on a session
sized for Llama (InferenceOptions dims) with exactly the rotation keys the port issues (the
"generic" family: the GPT-2 band would add a key per baby step j t^2, j < 512, at hid 4096).
"""
from __future__ import annotations

import os

from examples.gpt2_from_primitives import env as gpt2_env
from examples.gpt2_from_primitives.env import RUNTIME_LEGACY, RUNTIME_N32, bootstrap_setup, plan_runtime  # noqa: F401

from perseus.impl.layout import Dims, attention_rot_steps, linear_rot_steps

# What differs from the GPT-2 port's env: no residency ring (the weights are encoded at use, not
# staged per block), so its pinned stage arenas stay small.
ENV = {**gpt2_env.ENV,
       "FHE_STAGE_ARENA_GB": "1",
       "LLAMA_PACKING": "cachemir_complex",
       "LLAMA_LINEAR_BOTTOM": "0",
       # plans that stay bound at any context length: step-scoped ciphertext names, and the softmax
       # periodic at every length (SM_PERIODIC=2: Gp x tH-periodic scores, the Gp-dependent ops in `gp`
       # sub-steps, the Gp x tH-slot route built as the cache grows)
       "IMPL_SCOPED_NAMES": "1",
       "SM_PERIODIC": "2",
       # the attention's per-head block is tH = t H = 8 x 32 = 256 slots (GPT-2's is 32 x 16 = 512): the
       # periodic softmax values refresh on the 256-slot route
       "SPARSE_BTS_SLOTS": "256,1",
       # the GPT-2 long-context route setup (gpt2-long-context): SparseB up to 1024 slots, the multi-stage
       # big routes on a {3, 3} budget
       "FIDESLIB_BTS_SPARSE_B_MAX": "1024",
       "SPARSE_LEVEL_BUDGET": "3,3"}


def rot_steps(dims: Dims) -> list[int]:
    """Every rotation the port issues, as the signed step the runtime keys it by (a key for -1 is
    not one for N - 1): the linears (hid->hid, hid->E, E->hid, the LM-head tile hid->N), the
    attention / norm ladders, RoPE's N/2, and the negative steps -1 .. -(t-1) of the K/V push and
    the replicate-right ladders."""
    N = dims.N
    steps = set()
    for d_in, d_out in ((dims.hid, dims.hid), (dims.hid, dims.E), (dims.E, dims.hid), (dims.hid, N)):
        steps |= linear_rot_steps(N, d_in, d_out)
    steps |= attention_rot_steps(dims)
    steps.add(N // 2)
    pos = {s % N for s in steps if s % N and s % N > N - dims.t}
    steps = {s % N for s in steps if s % N} - pos
    return sorted(steps | {-k for k in range(1, dims.t)})


def export_env(device=3, chain="n32", **overrides):
    # every Llama value that differs from the GPT-2 port's (a key both set, e.g. SM_PERIODIC, too)
    gpt2_env.export_env(device, chain=chain, **{**{k: v for k, v in ENV.items() if gpt2_env.ENV.get(k) != v},
                                                **overrides})


def open_session(dims: Dims, device=3, complex_payload=True, chain="n32", **overrides):
    """perseus.session on the n32 decode preset, Llama dims, the port's rotation keys."""
    overrides = {"CKKS_COMPLEX": "1" if complex_payload else "0", **overrides}
    export_env(device, chain=chain, **overrides)
    from perseus import session
    from perseus.profile import SessionProfile
    prof = SessionProfile.gpt2_decode_n32()
    prof.ckks_complex = bool(complex_payload)
    prof.mode = "sync"
    prof.extra = dict(prof.extra)
    prof.extra.update(ENV)
    prof.extra.update({k: str(v) for k, v in overrides.items()})
    prof.apply()
    opts = prof.options()
    opts.dim, opts.hidDim = dims.dim, dims.hid
    opts.expanded, opts.expDim = dims.E_real, dims.E
    opts.numHeads, opts.numHeadsReal = dims.H, dims.H_real
    opts.ckks.extra_rot_steps = rot_steps(dims)      # a whole list: the binding copies on access
    return session(options=opts, family="generic", mode="sync")


def rotation_audit(fhe, dims: Dims):
    loaded = {int(s) for s in fhe.loaded_rot_steps}
    return sorted(set(rot_steps(dims)) - loaded)
