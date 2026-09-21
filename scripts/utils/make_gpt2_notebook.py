"""Regenerate notebooks/gpt2_torch_forward.ipynb — custom prompt, both phases planned.

Two things this fixes versus the previous version:

1. CUSTOM SENTENCE. It used to read block-0 embeddings straight out of the
   teacher-forced oracle, so the "prompt" was whatever phrase sat at that offset in
   the dataset. `read_teacher_forced_inputs` checks CLASSIFY_INPUT_JSON first, so the
   client can embed its own text (wte + wpe, exactly the split ViT/BERT use) and hand
   those rows in instead.

2. PLANNED. `run_generate` takes ONE plan dir (load_block_plans, flat block_*.json) and
   applies it across prefill and decode; it ignores FHE_DECODE_PLACEMENTS_DIR, and the
   chunked prefill dirs load as EMPTY -> silent eager. planned_gpt2_gen is the only
   correctly-shaped plan, and it binds while the run stays inside one attention group
   (GEN_PROMPT+GEN_TOKENS <= 32). Verified by `[mask_gen] ... strict=1`, which eager
   never prints.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# GPT-2 as a torch module — encrypted, planned, autoregressive

A prompt in, a continuation out, under CKKS. Prefill packs the prompt, hands off to the
decode phase, and the model then generates token by token with **encrypted argmax
feedback** — nothing is decrypted mid-stream, the sampled token never leaves the
ciphertext domain until the end.

Both phases run on `planned_gpt2_gen` — `run_generate` takes a single plan dir and
applies it across prefill and decode.

Only the embedding (`wte` + `wpe`) runs client-side, as in the ViT and BERT notebooks.
""")

code('''
import json, os, time
from pathlib import Path

import numpy as np
import torch                                   # torch before _core (NCCL load order)
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from perseus import _core

# ── write your prompt here ────────────────────────────────────────────────────
PROMPT = ("The history of computing begins long before the first electronic machine, "
          "with mechanical devices built to automate arithmetic, the patient work of "
          "human calculators, and the slow realisation that calculation itself could "
          "be described precisely enough for a machine to carry it out.")
# ──────────────────────────────────────────────────────────────────────────────
P = int(os.environ.get("GEN_PROMPT", "8"))     # prompt tokens (P+N must stay <= 32)
N = int(os.environ.get("GEN_TOKENS", "10"))    # tokens to generate

tok = GPT2TokenizerFast.from_pretrained("gpt2")
hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()

ids = tok(PROMPT)["input_ids"]
if len(ids) < P:
    raise ValueError(f"PROMPT is {len(ids)} tokens, need >= GEN_PROMPT={P}")
ids = ids[:P]
print(f"prompt ({P} tokens): {tok.decode(ids)!r}")
''')

md("""
## Client side: embed the prompt

`wte[token] + wpe[position]` — the block-0 input. Written to JSON and handed to the
runtime via `CLASSIFY_INPUT_JSON`, which `read_teacher_forced_inputs` honours ahead of
the oracle.

The rows past the prompt are placeholders: generation is not teacher-forced, so it feeds
back its own encrypted argmax and never reads them. They exist only because the reader
wants `gen_prompt + gen_tokens` rows.
""")

code('''
with torch.no_grad():
    emb = (hf.transformer.wte(torch.tensor(ids)) +
           hf.transformer.wpe(torch.arange(P))).numpy()

rows = emb.tolist() + [[0.0] * emb.shape[1] for _ in range(N)]   # tail unused (argmax feedback)
inp_path = Path(os.environ.get("TMPDIR", "/tmp")) / "gpt2_custom_prompt.json"
inp_path.write_text(json.dumps({"inp": rows}))
os.environ["CLASSIFY_INPUT_JSON"] = str(inp_path)
print(f"embedded {P} tokens x {emb.shape[1]} -> {inp_path}")

# plaintext reference: greedy continuation from the same prompt
with torch.no_grad():
    ref = hf.generate(torch.tensor([ids]), max_new_tokens=N, do_sample=False,
                      pad_token_id=tok.eos_token_id)[0, P:].tolist()
print(f"plaintext continuation: {tok.decode(ref)!r}")
''')

md("""
## Encrypted prefill → hand-off → generate

`RunConfig.from_env()` picks up the plan directory.

`unplanned_bts` is NOT proof of planning: it reads 0 in eager mode too, since nothing is
unplanned when there is no plan. The signal that the plan actually bound is
`[mask_gen] primed ... strict=1` in the job's stderr — eager never emits it.
""")

code('''
cfg = _core.RunConfig.from_env()
cfg.gen_prompt, cfg.gen_tokens, cfg.teacher_forced = P, N, False
cfg.tokens = P + N

print(f"plan: {os.environ.get('FHE_BOOTSTRAP_PLACEMENTS_DIR', '(eager)').split('/')[-1]}  "
      f"(P+N = {P + N} positions, must be <= 32)")

inputs = _core.read_teacher_forced_inputs(cfg)      # -> our CLASSIFY_INPUT_JSON rows
t0 = time.perf_counter()
res = _core.run_generate(cfg, inputs)
assert not res.threw, res.error
dt = time.perf_counter() - t0

gen_ids = [int(t) for t in res.top1]
print(f"\\ngenerated {res.completed} tokens in {dt:.0f}s ({dt/max(res.completed,1):.0f}s/tok)")
print(f"unplanned_bts = {res.unplanned_bts}   (necessary, not sufficient — see above)")
''')

md("""
## The continuation

Decrypted only now. The encrypted run feeds back its own argmax each step, so any
divergence from the plaintext model compounds — matching for the first tokens and then
drifting is the expected behaviour, not a failure.
""")

code('''
print(f"prompt           : {tok.decode(ids)!r}")
print(f"FHE continuation : {tok.decode(gen_ids)!r}")
print(f"ref continuation : {tok.decode(ref)!r}")
agree = sum(a == b for a, b in zip(gen_ids, ref))
print(f"\\ntoken agreement  : {agree}/{len(ref)}")
print(f"full FHE text    : {tok.decode(ids + gen_ids)!r}")
assert res.completed == N
assert res.unplanned_bts == 0, f"unplanned bootstraps fired: {res.unplanned_bts}"
''')

md("""
## What this showed

- **Encrypted GPT-2** generated a continuation from a prompt of your choosing, with the
  sampled token never leaving the ciphertext domain (encrypted argmax feedback).
- It ran **planned** on `planned_gpt2_gen` across prefill and decode — confirmed by
  `[mask_gen] ... strict=1` in stderr, which eager never prints (`unplanned_bts=0` alone
  proves nothing, since eager reports 0 too).
- Only `wte`+`wpe` ran client-side.

Change `PROMPT` in the first cell to prompt it differently — it must be at least
`GEN_PROMPT` tokens.

`GEN_PROMPT + GEN_TOKENS` must stay within **32 total positions**: that is
`slots / hidDim`, one attention group. `planned_gpt2_gen` was captured from position 0
over 2 tokens, so it only recorded group 0; past position 32 the runtime runs
`qkt_group.g1` ops the capture never saw and the plan cannot place them. Longer runs need
a decode-phase capture at a hand-off position, which the tooling does not currently
emit.
""")

nb.cells = C
nb.metadata = {"language_info": {"name": "python"}}
nbf.write(nb, "notebooks/gpt2_torch_forward.ipynb")
print("wrote notebooks/gpt2_torch_forward.ipynb")
