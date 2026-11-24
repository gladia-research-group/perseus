"""The plan each example driver runs when `--plan` is not given (device path only; `--fake` / `--capture` stay
plan-free). One entry per (model, token count): a plan is cut for one shape, so a driver asked for another count
still needs an explicit `--plan`. The 512-token decodes run a plan per position window (`WINDOWS`).
"""
from __future__ import annotations

import os

# model -> {T or None (any): plan dir, resolved by perseus._paths.resolve}
PLANS: dict[str, dict[int | None, str]] = {
    # up to 32 cached keys (the server, generation); past them the window plans up to 512 (WINDOWS)
    "gpt2_decode": {None: "bootstrap_placements/gpt2_decode"},
    # past 512 cached keys: one plan for every cache length up to GPT-2's 1,024
    "gpt2_decode_ctx1024": {None: "bootstrap_placements/gpt2_decode_ctx1024"},
    # to the first token: the blocks, the last position's LM head and the encrypted argmax
    "gpt2_prefill": {128: "bootstrap_placements/gpt2_prefill128"},
    "bert": {128: "bootstrap_placements/bert_sst2_128"},
    # StC-first dense refreshes, their inputs capped at its entry level (46)
    "vit": {None: "bootstrap_placements/vit_b16_224"},
    "llama_prefill": {128: "bootstrap_placements/llama31_8b_prefill128"},
    # one stage (run.py --one-stage), the folded ReLU
    "resnet": {None: "bootstrap_placements/resnet20_folded"},
    # CCMM attention (Gemma's 256 heads as two Llama half-heads, 4-way SinC merge, half pack, causal zeroing)
    "gemma2_prefill": {128: "bootstrap_placements/gemma2_2b_prefill128"},
}

# model -> [(bound, plan dir)]: up to a 512-token context, a plan per position window (`--plan-windows`: a window's plan
# serves the positions below its bound): the first 16 positions (the attention's lanes over the keys' window), then
# one per 128 positions, each sized on its window's magnitudes, one runtime for all
WINDOWS: dict[str, list[tuple[int, str]]] = {
    "gpt2_decode": [(b, f"bootstrap_placements/gpt2_decode_ctx512_w{b}") for b in (16, 128, 256, 384, 512)],
    "llama_decode": [(b, f"bootstrap_placements/llama31_8b_decode_ctx512_w{b}") for b in (16, 128, 256, 384, 512)],
    "gemma2_decode": [(b, f"bootstrap_placements/gemma2_2b_decode_ctx512_w{b}") for b in (16, 128, 256, 384, 512)],
}


def default_plan(model: str, tokens: int | None = None) -> str | None:
    """The default plan of `model` at `tokens` (None when the model has no plan for that shape)."""
    table = PLANS.get(model, {})
    return table.get(tokens, table.get(None))


def default_windows(model: str, tokens: int | None = None) -> list[tuple[int, str]] | None:
    """The default window plans of `model` for a `tokens`-token context (None past the last window)."""
    windows = WINDOWS.get(model)
    return windows if windows and (tokens is None or tokens <= windows[-1][0]) else None


# model -> switches every run of the model takes unless the environment sets them (each won a both-order A/B on it)
RUN_ENV: dict[str, dict[str, str]] = {
    # 11-24 (graphs, 2 forwards): the rotation pairs around the SiLU-GLU dropped (-0.19 s on 1 GPU, -0.135 s on 2) and
    # the softmax reciprocal chain's dense refreshes, 4096-periodic, on the 4096-slot route (-0.10 s / -0.12 s, 1.5 GB
    # more device memory)
    "llama_prefill": {"PC_ROT_CANCEL": "1", "PC_SHRINK_SLOTS": "4096", "PC_SHRINK_MATCH": "attn_softmax"},
    # 11-24 (2 GPUs, graphs): the rotation pairs dropped, -0.05 s and 0.8 GB less device memory
    "gemma2_prefill": {"PC_ROT_CANCEL": "1"},
}

_SCHED2 = {"PC_SCHED2": "1", "PC_SCHED2_FIN": "async"}
_PART = {"PC_SCHED2_PART": "1", "PC_SCHED2_PART_ALL": "1"}
# model -> switches of a two-GPU run (PC_REMOTE_DEVICE set) unless the environment sets them: the two-device compile
# schedule where it beats the replica lane alone
TWO_GPU_ENV: dict[str, dict[str, str]] = {
    "gpt2_prefill": {**_SCHED2, **_PART},
    "vit": {**_SCHED2, **_PART},
    "bert": dict(_SCHED2),
    "gemma2_prefill": {**_SCHED2, "PC_SCHED2_MEM": "1.1"},
}


def apply_env(model: str) -> None:
    """Set `model`'s default switches (RUN_ENV, and TWO_GPU_ENV on two GPUs) that the environment leaves unset; call
    before the runtime starts (the switches are read once)."""
    two = os.environ.get("PC_REMOTE_DEVICE", "").strip() not in ("", "0")
    for k, v in {**RUN_ENV.get(model, {}), **(TWO_GPU_ENV.get(model, {}) if two else {})}.items():
        os.environ.setdefault(k, v)
