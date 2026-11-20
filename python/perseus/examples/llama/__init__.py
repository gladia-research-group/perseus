"""Decoder-only LLMs (Llama-3 8B, Gemma-2 2B) encrypted on ``perseus.impl``: the GPT-2 port's kernels, packing and
runtime levers (perseus/examples/gpt2) applied to the RMSNorm / RoPE / GQA / gated-MLP decoder.

Shared: RMSNorm (impl.norm.rms_norm), RoPE folded into the selection masks (impl.attention.Rope: one
rotation, no level), GQA expanded in the K / V weights, a gated MLP with a THOR composite activation, a plaintext BOS
prefix (prefix.py; the encrypted decode starts at position 1), streamed coefficient-row weights (~40 GB pinned host
memory for Llama-3-8B, encoded on the GPU per baby-step sum), attention routes {tH, 1}. What a family changes (norm
offset, post norms, activation, soft-caps, head width, embedding scale, tied head) is its arch.Arch, read from the
checkpoint; the shipped checkpoints and their configs are arch.MODELS (--model). Module map and the capture -> plan ->
planned decode recipe: docs/ARCHITECTURE.md.
"""
