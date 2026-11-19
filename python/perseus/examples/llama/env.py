"""Session setup for the Llama port: the GPT-2 port's n32 decode env (perseus/examples/gpt2/
env.py: sparse routes, the softmax / norm switches, the plan-bound bootstrap levers) on a session
sized for Llama (InferenceOptions dims) with exactly the rotation keys the port issues (the
"generic" family: the GPT-2 band would add a key per baby step j t^2, j < 512, at hid 4096).
"""
from __future__ import annotations

import os

from perseus.examples.gpt2 import env as gpt2_env
from perseus.examples.gpt2.env import RUNTIME_LEGACY, RUNTIME_N32, bootstrap_setup, plan_runtime  # noqa: F401

from perseus.impl.layout import Dims, attention_rot_steps, linear_rot_steps

# What differs from the GPT-2 port's env: no residency ring (the weights are encoded at use, not
# staged per block), so its pinned stage arenas stay small.
ENV = {**gpt2_env.ENV,
       "FHE_STAGE_ARENA_GB": "1",
       "LLAMA_PACKING": "cachemir_complex",
       "LLAMA_LINEAR_BOTTOM": "0",
       "LLAMA_LMHEAD_BOTTOM": "0",
       "LLAMA_MLP_SPLIT": "0",
       "SM_QKT_SHARE": "0",
       "SM_HEAD_PACK": "0",
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


def dims_env(dims: Dims) -> dict:
    """The switches that follow the model's layout: the periodic softmax refreshes on the attention's per-head block,
    tH = t H slots (Llama-3-8B 8 x 32 = 256: ENV's value; Gemma-2-2B 8 x 16 = 128). The CPU twin's refresh landings
    (FAKECAP_LANDINGS, the preset's) list GPT-2's and Llama's sparse routes only: a sparse route lands by its level budget
    (SPARSE_LEVEL_BUDGET), not its slot count, so a tH route missing there lands where the next listed sparse route does."""
    out = {"SPARSE_BTS_SLOTS": f"{dims.tH},1"}
    from perseus import _pc
    spec = _pc.Config.from_legacy_env(dict(os.environ))["quirks.fakecap_landings"] or "0:32,1:14,256:22,512:22"
    lands = dict(kv.split(":", 1) for kv in spec.split(",") if kv)
    sparse = sorted((int(s), v) for s, v in lands.items() if int(s) > 1)
    if str(dims.tH) not in lands and sparse:
        ref = next((v for s, v in sparse if s >= dims.tH), sparse[-1][1])
        out["FAKECAP_LANDINGS"] = f"{spec},{dims.tH}:{ref}"
    return out


def rot_steps(dims: Dims) -> list[int]:
    """Every rotation the port issues, as the signed step the runtime keys it by (a key for -1 is
    not one for N - 1): the linears (hid->hid, hid->E, E->hid, the LM-head tile hid->N), the
    attention / norm ladders, RoPE's N/2, and the negative steps -1 .. -(t-1) of the K/V push and
    the replicate-right ladders."""
    N = dims.N
    steps = set()
    from .weights import mlp_parts
    shapes = [(dims.hid, dims.hid), (dims.hid, dims.E), (dims.E, dims.hid), (dims.hid, N)]
    shapes += [s for _o, w, _r in mlp_parts(dims) for s in ((dims.hid, w), (w, dims.hid))]   # LLAMA_MLP_SPLIT
    for d_in, d_out in shapes:
        steps |= linear_rot_steps(N, d_in, d_out)
    steps |= attention_rot_steps(dims)
    steps.add(N // 2)
    pos = {s % N for s in steps if s % N and s % N > N - dims.t}
    steps = {s % N for s in steps if s % N} - pos
    steps |= {-k for k in range(1, dims.t)}
    steps |= _radix_steps(dims)
    from .weights import q_pad_heads
    if q_pad_heads(dims):
        steps.add(q_pad_heads(dims) * dims.t)      # Q off K's padding heads (the K / V push)
    if any(os.environ.get(k, "0") not in ("", "0") for k in ("SM_QKT_SHARE", "SM_HEAD_PACK")) \
            and 0 < dims.H_real and 2 * dims.H_real <= dims.H:
        steps |= {dims.H_real * dims.t, N - dims.H_real * dims.t}   # attention._qkt_share's shift down and back up
        if os.environ.get("SM_HEAD_PACK", "0") not in ("", "0"):    # the push's lane + half rotations (K, V)
            sh = dims.H_real * dims.t
            steps |= {(-r - sh) % N for r in range(dims.t)} | {(1 - r - sh) % N for r in range(dims.t)}
    if os.environ.get("LLAMA_GQA", "0") not in ("", "0") and dims.H % 8 == 0 and dims.H > 8:
        # mha_gqa: the sub-geometry's core/push steps, the queries' g t moves and their push lanes 2 .. G + 1
        from perseus.impl.attention import gqa_dims
        G = dims.H // 8
        sub = gqa_dims(dims, G)
        steps |= {s % N for s in attention_rot_steps(sub) if s % N} | {-k for k in range(1, sub.t)} | {N // 2}
        steps -= {N - k for k in range(1, sub.t)}      # rotated by -k: N - k would be a second key
        steps |= {s * g * dims.t for g in range(1, G) for s in (1, -1)} | {s * (2 + i) for i in range(G) for s in (1, -1)}
        steps |= _radix_steps(sub)
    return sorted(steps)


def _radix_steps(dims: Dims) -> set[int]:
    """LADDER_RADIX_BITS group steps (keyed as -k near N): without them plans fail plan_deg_error."""
    from perseus.impl.layout import ladder_radix_bits, ladder_radix_steps
    bits = ladder_radix_bits()
    if bits < 2:
        return set()
    N = dims.N
    return {s - N if s > N - dims.t else s for s in ladder_radix_steps(dims, bits)}


def export_env(device=None, chain="n32", **overrides):
    # every Llama value that differs from the GPT-2 port's (a key both set, e.g. SM_PERIODIC, too)
    gpt2_env.export_env(device, chain=chain, **{**{k: v for k, v in ENV.items() if gpt2_env.ENV.get(k) != v},
                                                **overrides})


def open_session(dims: Dims, device=None, complex_payload=True, chain="n32", extra_rot_steps=(), base=True, **overrides):
    """perseus.session on the n32 decode preset, Llama dims, the port's rotation keys."""
    overrides = {"CKKS_COMPLEX": "1" if complex_payload else "0", **overrides}
    export_env(device, chain=chain, **overrides)
    from perseus.session import session
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
    # a whole list: the binding copies on access
    opts.ckks.extra_rot_steps = sorted((set(rot_steps(dims)) if base else set()) | {int(s) for s in extra_rot_steps})
    return session(options=opts, family="generic", mode="sync")


def rotation_audit(fhe, dims: Dims):
    loaded = {int(s) for s in fhe.loaded_rot_steps}
    return sorted(set(rot_steps(dims)) - loaded)
