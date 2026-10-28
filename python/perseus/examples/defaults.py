"""The plan each example driver runs when `--plan` is not given (device path only; `--fake` / `--capture` stay
plan-free). One entry per (model, token count): a plan is cut for one shape, so a driver asked for another count
still needs an explicit `--plan`. The 512-token decodes run a plan per position window (`WINDOWS`).
"""
from __future__ import annotations

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
    # the decode plan cut for K / V handed over by the 128-token prefill, its softmax refreshes one CF lower (past
    # 128 entries they run the 8192-slot route, where the priced CF sits above the band)
    "llama_handoff_decode": {None: "bootstrap_placements/llama31_8b_handoff_decode"},
}

# model -> [(bound, plan dir)]: up to a 512-token context, one plan per 128-position window (`--plan-windows`: a
# window's plan serves the positions below its bound), each sized on its window's magnitudes, one runtime for all
WINDOWS: dict[str, list[tuple[int, str]]] = {
    "gpt2_decode": [(b, f"bootstrap_placements/gpt2_decode_ctx512_w{b}") for b in (128, 256, 384, 512)],
    "llama_decode": [(b, f"bootstrap_placements/llama31_8b_decode_ctx512_w{b}") for b in (128, 256, 384, 512)],
}


def default_plan(model: str, tokens: int | None = None) -> str | None:
    """The default plan of `model` at `tokens` (None when the model has no plan for that shape)."""
    table = PLANS.get(model, {})
    return table.get(tokens, table.get(None))


def default_windows(model: str, tokens: int | None = None) -> list[tuple[int, str]] | None:
    """The default window plans of `model` for a `tokens`-token context (None past the last window)."""
    windows = WINDOWS.get(model)
    return windows if windows and (tokens is None or tokens <= windows[-1][0]) else None
