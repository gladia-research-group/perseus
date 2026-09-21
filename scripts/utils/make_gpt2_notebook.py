"""Generate notebooks/gpt2_torch_forward.ipynb — a custom prompt generated under encryption.

The notebook drives the C++ driver (`_core.RunConfig.from_env()`, `_core.run_generate`)
in eager mode: the client embeds its own prompt (`wte + wpe`) and passes the rows straight
to the driver; the server prefills the prompt and generates with encrypted argmax feedback.

    python scripts/utils/make_gpt2_notebook.py      # from the repo root

`tests/test_notebooks.py` checks that this generator reproduces the tracked file byte for
byte, which is why every cell id is pinned (nbformat mints random ids otherwise).
"""
import nbformat as nbf

OUT = "notebooks/gpt2_torch_forward.ipynb"

PREREQ_MARK = "<!-- prereq-cell -->"
PREREQ_NOTE = PREREQ_MARK + """
> **Prerequisites.** This notebook loads artifacts it does not create: the exported
> weights (`weights.bin.zip`, `client.npz`), the calibrated `configs.json`, and the decode
> oracle (`ALL_BLOCKS_IO_DIR`), which every run of the driver requires even when, as here,
> it is not read. Run **`setup_artifacts.ipynb`** first if you do not have them.
"""


def build() -> nbf.NotebookNode:
    """The notebook as a NotebookNode (no file I/O)."""
    cells = []

    def md(slug, s):
        cells.append((slug, nbf.v4.new_markdown_cell(s.strip())))

    def code(slug, s):
        cells.append((slug, nbf.v4.new_code_cell(s.strip())))

    md("intro", """
# GPT-2 generation under encryption — the C++ driver

A prompt in, a continuation out, under CKKS. The prompt is prefilled, the model hands off
to the decode phase and then generates token by token with **encrypted argmax feedback**:
the sampled token never leaves the ciphertext domain until the end.

The run is **eager**: the shipped bootstrap plans (`bootstrap_placements/gpt2_decode_n32`,
`gpt2_decode_n64`) are for the paper's oracle-fed decode row (`TASK=decode`), and eager
generation refreshes reactively whenever a ciphertext runs out of levels.

Only the embedding (`wte` + `wpe`) runs client-side. The environment comes from the runner
(`NB=gpt2_torch bash scripts/run_notebooks.sh`).
""")

    md("prereq", PREREQ_NOTE)

    code("setup", '''
import os
import time

import torch                                   # torch before _core (NCCL load order)
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from perseus import _core

# ── write your prompt here ────────────────────────────────────────────────────
PROMPT = ("The history of computing begins long before the first electronic machine, "
          "with mechanical devices built to automate arithmetic, the patient work of "
          "human calculators, and the slow realisation that calculation itself could "
          "be described precisely enough for a machine to carry it out.")
# ──────────────────────────────────────────────────────────────────────────────
P = int(os.environ.get("GEN_PROMPT", "8"))     # prompt tokens (the prefilled prefix)
N = int(os.environ.get("GEN_TOKENS", "10"))    # tokens to generate

tok = GPT2TokenizerFast.from_pretrained("gpt2")
hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()

ids = tok(PROMPT)["input_ids"]
if len(ids) < P:
    raise ValueError(f"PROMPT is {len(ids)} tokens, need >= GEN_PROMPT={P}")
ids = ids[:P]
print(f"prompt ({P} tokens): {tok.decode(ids)!r}")
''')

    md("client-md", """
## Client side: embed the prompt

`wte[token] + wpe[position]` — the block-0 input, which is what the encrypted run
consumes. `run_generate` takes these rows directly, so the prompt below is the one the
model actually continues.
""")

    code("client", '''
with torch.no_grad():
    emb = (hf.transformer.wte(torch.tensor(ids)) +
           hf.transformer.wpe(torch.arange(P))).numpy()

rows = emb.tolist()                                 # generation feeds back its own argmax
print(f"embedded {P} tokens x {emb.shape[1]}")

# plaintext reference: greedy continuation from the same prompt
with torch.no_grad():
    ref = hf.generate(torch.tensor([ids]), max_new_tokens=N, do_sample=False,
                      pad_token_id=tok.eos_token_id)[0, P:].tolist()
print(f"plaintext continuation: {tok.decode(ref)!r}")
''')

    md("run-md", """
## Encrypted prefill → hand-off → generate

`RunConfig.from_env()` reads the model paths and the CKKS environment; `plan_dir` is
cleared explicitly so the run is eager whatever the shell exported. The prompt rows go in
directly. `res.bootstraps` is the number of reactive refreshes the run executed.
""")

    code("run", '''
cfg = _core.RunConfig.from_env()
cfg.plan_dir = ""                                   # eager: no plan
cfg.gen_prompt, cfg.gen_tokens, cfg.teacher_forced = P, N, False
cfg.tokens = P + N

t0 = time.perf_counter()
res = _core.run_generate(cfg, rows)                 # raises on a token error
dt = time.perf_counter() - t0

gen_ids = [int(t) for t in res.top1]
print(f"\\ngenerated {res.completed} tokens in {dt:.0f}s ({dt / max(res.completed, 1):.0f}s/tok)")
print(f"bootstraps executed = {res.bootstraps}")
''')

    md("result-md", """
## The continuation

Decrypted only now. The encrypted run feeds back its own argmax each step, so any
divergence from the plaintext model compounds — matching for the first tokens and then
drifting is the expected behaviour, not a failure.
""")

    code("result", '''
print(f"prompt           : {tok.decode(ids)!r}")
print(f"FHE continuation : {tok.decode(gen_ids)!r}")
print(f"ref continuation : {tok.decode(ref)!r}")
agree = sum(a == b for a, b in zip(gen_ids, ref))
print(f"\\ntoken agreement  : {agree}/{len(ref)}")
print(f"full FHE text    : {tok.decode(ids + gen_ids)!r}")
assert res.completed == N
''')

    md("showed", """
## What this showed

- **Encrypted GPT-2** generated a continuation from a prompt of your choosing, with the
  sampled token never leaving the ciphertext domain (encrypted argmax feedback).
- It ran **eager**: every bootstrap was placed reactively by the runtime. The planned
  configuration of the paper is the oracle-fed decode row (`TASK=decode CHAIN=n32 bash
  scripts/run_task.sh`, 552 bootstraps per token against 1310 eager).
- Only `wte`+`wpe` ran client-side.

Change `PROMPT` in the first cell to prompt it differently — it must be at least
`GEN_PROMPT` tokens; `GEN_TOKENS` sets the length of the continuation.
`gpt2_perseus_nn.ipynb` does the same through the Python modules, with the client and the
server split across the trust boundary.
""")

    nb = nbf.v4.new_notebook()
    nb.cells = [c for _, c in cells]
    for i, (slug, c) in enumerate(cells):
        c.id = f"gpt2torch-{i:02d}-{slug}"     # pinned: regeneration is byte-identical
    nb.metadata = {"language_info": {"name": "python"}}
    return nb


if __name__ == "__main__":
    nbf.write(build(), OUT)
    print(f"wrote {OUT}")
