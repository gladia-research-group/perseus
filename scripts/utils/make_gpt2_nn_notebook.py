"""Generate notebooks/gpt2_perseus_nn.ipynb — encrypted GPT-2 generation through perseus.nn.

The `gpt2_torch_forward` notebook drives the C++ driver (`_core.RunConfig.from_env()`,
`_core.run_generate`). This one walks the Python API instead: `EncGPT2.from_pretrained`,
the client/server roles (`EncClient` / `EncServer`, `EncGenerationClient` /
`EncGenerationServer`) and `generate`. It is never executed in-tree (it needs a GPU);
`tests/test_notebooks.py` validates the file and checks that this generator reproduces it
byte for byte, which is why every cell id is pinned below (nbformat mints random ids
otherwise).

    python scripts/utils/make_gpt2_nn_notebook.py      # from the repo root
"""
import nbformat as nbf

OUT = "notebooks/gpt2_perseus_nn.ipynb"

PREREQ_MARK = "<!-- prereq-cell -->"
PREREQ_NOTE = PREREQ_MARK + """
> **Prerequisites.** This notebook loads artifacts it does not create: the exported
> weights (`weights.bin.zip`, `client.npz`) and the calibrated `configs.json`. Run
> **`setup_artifacts.ipynb`** first if you do not have them.
"""


def build() -> nbf.NotebookNode:
    """The notebook as a NotebookNode (no file I/O)."""
    cells = []

    def md(slug, s):
        cells.append((slug, nbf.v4.new_markdown_cell(s.strip())))

    def code(slug, s):
        cells.append((slug, nbf.v4.new_code_cell(s.strip())))

    md("intro", """
# Encrypted GPT-2 generation through `perseus.nn`

A prompt in, a continuation out, under CKKS — composed from the Python modules and split
across the trust boundary:

- the **client** owns the keys, the embedding tables (`client.npz`) and the argmax over
  decrypted logit tiles; it never sees a weight;
- the **server** holds the bound `EncGPT2` and maps bytes to bytes — encrypted prompt
  embeddings in, encrypted logit tiles out. With encrypted feedback its CutMax argmax
  re-embeds the next token on the server side, so no token id is ever visible there.

This notebook is **not executed in-tree**: keygen, the bundle and the bound model need a
GPU box and tens of minutes. It is validated (`tests/test_notebooks.py`) as a valid
notebook whose code cells parse and whose generator reproduces it.
""")

    md("prereq", PREREQ_NOTE)

    md("differs", """
## How this differs from `gpt2_torch_forward.ipynb`

`gpt2_torch_forward.ipynb` drives the **C++ driver**: `_core.RunConfig.from_env()`,
`_core.read_teacher_forced_inputs`, `_core.run_generate`, with the encrypted prefill
hand-off — the vehicle the paper's rows were measured with.

This notebook composes the same model from the **Python modules** (`perseus.nn`) and splits
client and server across the trust boundary described in `docs/SECURITY_MODEL.md`. It is the
API path — the one a deployment builds on. Both run **eager**: the shipped plans are for the
oracle-fed decode row of the paper (`TASK=decode`), and eager generation refreshes
reactively.
""")

    code("setup", '''
import os
import tempfile
import time

import numpy as np
import torch                                   # torch before _core (NCCL load order)
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from perseus.nn import EncClient, EncGenerationClient, EncGenerationServer, EncGPT2, EncServer
from perseus.profile import SessionProfile

MODEL    = os.environ.get("GPT2_MODEL", "openai-community/gpt2")
DATA     = os.environ.get("PERSEUS_DATA", ".cache")
WEIGHTS  = os.environ.get("WEIGHTS_PATH", f"{DATA}/models/{MODEL}/classic/weights.bin.zip")
CONFIGS  = os.environ.get("CONFIGS_PATH", "configs/model/approximation/gpt2_base_n32/configs.json")
# CKKS parameters: the runner's env by default; PIN_PROFILE=1 pins the paper's n32 recipe
PROFILE  = SessionProfile.gpt2_decode_n32() if os.environ.get("PIN_PROFILE") else None
PROMPT   = os.environ.get("PERSEUS_PROMPT", "The capital of France is")
N        = int(os.environ.get("GEN_TOKENS", "6"))

tok = GPT2TokenizerFast.from_pretrained(MODEL)
prompt_ids = tok.encode(PROMPT)
print(f"{MODEL}: {len(prompt_ids)} prompt tokens, {N} to generate")
print(f"weights: {WEIGHTS}    configs: {CONFIGS}    "
      f"profile: {PROFILE.chain if PROFILE else 'env'}")
''')

    md("client-md", """
## Client: keys and embeddings

`EncClient` generates the keys on the CPU (no GPU on the client side) and writes the
server's **bundle**: context, public and evaluation keys, and a manifest of the CKKS
parameters — never the secret key. The CKKS parameters come from the runner's environment
(the chain, the levels); `PIN_PROFILE=1` passes `profile=SessionProfile.gpt2_decode_n32()`
instead.

`EncGenerationClient` adds the GPT-2 embedding tables from `client.npz` (written by
`perseus-export` next to `weights.bin.zip`): token ids become `wte[id] + wpe[pos]` rows,
encrypted one per position.
""")

    code("client", '''
t0 = time.perf_counter()
client = EncClient(family="gpt2", profile=PROFILE)   # CPU keygen
bundle = client.save_bundle(os.path.join(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")), "bundle"))
print(f"{client!r}: keys + bundle in {time.perf_counter() - t0:.0f} s -> {bundle}")

npz = np.load(os.path.join(os.path.dirname(WEIGHTS), "client.npz"))
gen_client = EncGenerationClient(client, wte=npz["wte"], wpe=npz["wpe"])
print(f"wte {npz['wte'].shape}, wpe {npz['wpe'].shape}, vocab {gen_client.vocab}")

blob = client.encrypt(gen_client.embed(prompt_ids[0], 0))
print(f"one encrypted position on the wire: {len(blob) / 1e6:.1f} MB")
''')

    md("server-md", """
## Server: a session from the bundle, the model from the artifacts

`EncServer(bundle)` builds the GPU session from the client's bundle; it holds no secret key
and refuses to start with one. `EncGPT2.from_pretrained` loads the exported weights and the
calibrated `configs.json`, with a provenance cross-check between the two (without
`weights=` it looks in the HuggingFace cache home, `<HF_HOME>/perseus/models/<name>/classic/`).

`bind(server.inf, cache_states=False)` is the loader-shaped bind: nothing is held host-side
and each block is re-encoded per forward on the residency worker (seconds to bind).
`cache_states=True` pre-encodes every block at bind (the decode shape: minutes to bind,
faster steady tokens).
""")

    code("server", '''
t0 = time.perf_counter()
server = EncServer(bundle)                     # session from the bundle: cannot decrypt
model = EncGPT2.from_pretrained(MODEL, weights=WEIGHTS, configs=CONFIGS)
model.bind(server.inf, cache_states=False)     # eager bind: reactive bootstraps
gen_server = EncGenerationServer(server, model)
print(f"{server!r} bound in {time.perf_counter() - t0:.0f} s")
print(model)
''')

    md("generate-md", """
## Generate

`feedback="encrypted"`: after each step the server's CutMax argmax turns the logit tiles
into a one-hot and re-embeds it through the codebook, on the server, under encryption —
the client decrypts the logit tiles only to *report* the token. `feedback="client"` sends
the decrypted token back as a fresh encrypted embedding instead (the path that allows
sampling).
""")

    code("generate", '''
FEEDBACK = os.environ.get("GEN_FEEDBACK", "encrypted")
emitted = []

def on_token(tok_id, j):
    emitted.append(tok_id)
    print(f"  token {j}: {tok_id:6d}  {tok.decode([tok_id])!r}", flush=True)

t0 = time.perf_counter()
ids = gen_client.generate(gen_server, prompt_ids, max_new_tokens=N, feedback=FEEDBACK,
                          on_token=on_token)
dt = time.perf_counter() - t0
print(f"{len(ids)} tokens in {dt:.0f} s  ({dt / len(ids):.1f} s/token, feedback={FEEDBACK})")
print(repr(PROMPT + tok.decode(ids)))
assert ids == emitted
assert len(ids) == N or ids[-1] == tok.eos_token_id
''')

    md("reference-md", """
## Plaintext reference

Greedy decoding with the HuggingFace model on the same prompt. Agreement is reported, not
asserted: every approximated nonlinearity adds a little drift and it compounds over steps,
so the encrypted continuation is expected to track the plaintext one for the first tokens
and may diverge later.
""")

    code("reference", '''
hf = GPT2LMHeadModel.from_pretrained(MODEL).eval()
with torch.no_grad():
    out = hf.generate(torch.tensor([prompt_ids]), max_new_tokens=N, do_sample=False,
                      pad_token_id=tok.eos_token_id)
ref_ids = out[0, len(prompt_ids):].tolist()

print("encrypted:", repr(tok.decode(ids)))
print("plaintext:", repr(tok.decode(ref_ids)))
agree = sum(a == b for a, b in zip(ids, ref_ids))
print(f"{agree}/{min(len(ids), len(ref_ids))} tokens agree")
''')

    md("notes", """
## Profile, chain, plans and GPU notes

**Chain and profile.** `EncClient()` takes the CKKS parameters from the environment, which
the runner exports (`scripts/run_notebooks.sh`) exactly as `client_server_minimal.ipynb`
does. To pin them in code use `EncClient(profile=SessionProfile.gpt2_decode_n32())` — chain
n32, `AUTO_BTS_LEVEL` 46, rotation-key band 22. The bundle's manifest carries the
parameters; `EncServer` refuses a bundle that disagrees with the options it was given, and a
`SessionProfile` for another chain than the loaded `perseus._core` build warns.

**One live context per process.** The client is CPU-only (`skip_gpu_load`) and coexists with
the server session here, as in `client_server_minimal.ipynb`; a real deployment runs the
client in its own process (`scripts/utils/probe_client_server.py` is the two-process form).

**Plans.** The bind is eager: the shipped plans are the Python implementation's
(`examples/gpt2_from_primitives`), captured on its own op sequence. Planning a session of these
modules is a capture of its own (`FHE_GRAPH_DIR`) followed by `perseus-plan`.

**Cost.** Keygen is about a minute of CPU; the bundle is tens of GB on `TMPDIR`; the server
session takes about a minute to build; the eager, loader-shaped decode is slower per token
than the planned decode row (15.44 s/token on n32, README Table 1). Nothing here allocates
a GPU until `EncServer(bundle)`.
""")

    md("showed", """
## What this showed

- `EncGPT2.from_pretrained` + `bind(server.inf)` is the whole server: the artifacts the
  setup notebook produced, a session built from a key bundle, no secret key anywhere.
- `EncGenerationClient` / `EncGenerationServer` are the two halves of `EncGPT2.generate()`
  split at the wire: bytes cross, token ids do not (with encrypted feedback).
- The plaintext reference is the yardstick, not a gate: agreement on the first tokens,
  drift allowed afterwards.

To change: `PERSEUS_PROMPT` / `GEN_TOKENS` for the text, `GEN_FEEDBACK=client` for sampling,
`cache_states=True` for the decode-shaped bind.
""")

    nb = nbf.v4.new_notebook()
    nb.cells = [c for _, c in cells]
    for i, (slug, c) in enumerate(cells):
        c.id = f"gpt2nn-{i:02d}-{slug}"     # pinned: regeneration is byte-identical
    nb.metadata = {"language_info": {"name": "python"}}
    return nb


if __name__ == "__main__":
    nbf.write(build(), OUT)
    print(f"wrote {OUT}")
