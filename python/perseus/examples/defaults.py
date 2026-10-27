"""The plan each example driver runs when `--plan` is not given (device path only; `--fake` / `--capture` stay
plan-free). One entry per (model, token count): a plan is cut for one shape, so a driver asked for another count
still needs an explicit `--plan`.
"""
from __future__ import annotations

# model -> {T or None (any): plan dir, resolved by perseus._paths.resolve}
PLANS: dict[str, dict[int | None, str]] = {
    # up to 32 cached keys (the server, generation); past them bootstrap_placements/gpt2_decode_ctx1024
    "gpt2_decode": {None: "bootstrap_placements/gpt2_decode"},
    # to the first token: the blocks, the last position's LM head and the encrypted argmax
    "gpt2_prefill": {128: "bootstrap_placements/gpt2_prefill128"},
    "bert": {128: "bootstrap_placements/bert_sst2_128"},
    # StC-first dense refreshes where the input fits the entry, the shipped order past it (code-owned refreshes too)
    "vit": {None: "bootstrap_placements/vit_b16_224"},
    # up to a 512-token context
    "llama_decode": {None: "bootstrap_placements/llama31_8b_decode"},
    "llama_prefill": {128: "bootstrap_placements/llama31_8b_prefill128"},
    # the decode plan cut for K / V handed over by the 128-token prefill, its softmax refreshes one CF lower (past
    # 128 entries they run the 8192-slot route, where the priced CF sits above the band)
    "llama_handoff_decode": {None: "bootstrap_placements/llama31_8b_handoff_decode"},
}


def default_plan(model: str, tokens: int | None = None) -> str | None:
    """The default plan of `model` at `tokens` (None when the model has no plan for that shape)."""
    table = PLANS.get(model, {})
    return table.get(tokens, table.get(None))
