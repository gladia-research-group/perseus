"""The plan each example driver runs when `--plan` is not given (device path only; `--fake` / `--capture` stay
plan-free). One entry per (model, token count): a plan is cut for one shape, so a driver asked for another count
still needs an explicit `--plan`.
"""
from __future__ import annotations

# model -> {T or None (any): plan dir, resolved by perseus._paths.resolve}
PLANS: dict[str, dict[int | None, str]] = {
    # 16 tokens (Decode-8, speedrun/server.py's plan): softmax and GELU fitted to positions 0-15, the softmax of blocks
    # 0-7, 9 and 10 centred on per-head reference keys, planned on the capture merged over four oracle rows at every
    # position, CutMax's scaled normalisers with ln_f folded into the LM head, the LayerNorm statistics from one refresh
    # or one complex ladder, on the alternating chain n32sp with record 5's key set. Any other length: the
    # shipped decode capture's plan, 133 refreshes / token (CutMax: 12 of them); those that run the shipped order at its CF
    "gpt2_decode": {16: "bootstrap_placements/gpt2_decode_python_n32sp_kc16r_gnabd7_z5",
                    None: "bootstrap_placements/gpt2_decode_python_n32_m16i_fcf_cm"},
    # StC-first dense refreshes + the runtime's measured landings
    "gpt2_prefill": {128: "bootstrap_placements/gpt2_prefill128_m16s0"},
    "bert": {128: "bootstrap_placements/bert_base_sst2_128_m16s0"},
    # StC-first dense refreshes where the input fits the entry, the shipped order past it (code-owned refreshes too)
    "vit": {None: "bootstrap_placements/vit_b16_224_m18s3"},
    "llama_decode": {None: "bootstrap_placements/llama31_8b_decode_python_n32"},
    "llama_prefill": {128: "bootstrap_placements/llama31_8b_prefill128"},
    # the decode plan cut for K / V handed over by the 128-token prefill, its softmax refreshes one CF lower (past
    # 128 entries they run the 8192-slot route, where the priced CF sits above the band)
    "llama_handoff_decode": {None: "bootstrap_placements/llama31_8b_decode_python_n32_dsw48_ho38_m18e"},
}


def default_plan(model: str, tokens: int | None = None) -> str | None:
    """The default plan of `model` at `tokens` (None when the model has no plan for that shape)."""
    table = PLANS.get(model, {})
    return table.get(tokens, table.get(None))
