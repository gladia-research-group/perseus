"""Generate notebooks/setup_artifacts.ipynb — the artifacts every GPT-2 notebook loads.

Artifacts only: export, token pool, calibration, decode oracle, graph capture and plan.
It never runs the encrypted model; the model notebooks do that. Every step is guarded by
an existence check, so re-running the notebook on a prepared checkout is free.

    python scripts/utils/make_setup_notebook.py      # from the repo root

`tests/test_notebooks.py` checks that this generator reproduces the tracked file byte for
byte, which is why every cell id is pinned (nbformat mints random ids otherwise).
"""
import nbformat as nbf

OUT = "notebooks/setup_artifacts.ipynb"


def build() -> nbf.NotebookNode:
    """The notebook as a NotebookNode (no file I/O)."""
    cells = []

    def md(slug, s):
        cells.append((slug, nbf.v4.new_markdown_cell(s.strip())))

    def code(slug, s):
        cells.append((slug, nbf.v4.new_code_cell(s.strip())))

    md("intro", """
# Setting up encrypted GPT-2 inference

Running GPT-2 under CKKS needs a few artifacts besides the HuggingFace checkpoint: the
weights in the runtime's packed format, the calibrated polynomial approximations of every
nonlinearity, a plaintext oracle the decode gate compares against, and — for the planned
row — a captured operation graph and the bootstrap plan computed from it.

| # | artifact | made by | cost |
|---|---|---|---|
| 1 | `weights.bin.zip`, `client.npz` | `perseus-export` | seconds, CPU |
| 2 | calibration token pool (`.npy`) | `perseus.calibrate.data` | needs network once |
| 3 | `configs.json` — the fitted approximations | `perseus-calibrate` (shipped) | minutes, GPU |
| 4 | the decode oracle (`all_blocks_io/`) | `scripts/utils/gen_gpt2_oracle.py` | seconds, CPU |
| 5 | the captured graph (`graphs/gpt2_decode_python_n32`) | `examples.gpt2_from_primitives.run_decode --capture` (shipped) | an hour, GPU |
| 6 | the bootstrap plan (`bootstrap_placements/gpt2_decode_python_n32`) | `scripts/make_plans.sh` (shipped) | seconds, CPU |

Steps 1-4 land under `PERSEUS_DATA` (default `.cache/` in the checkout), which is where
`scripts/local_env.sh` and the runner look for them; the graph and the plan land in the
repository, under `graphs/` and `bootstrap_placements/`. Each cell skips its step when the
artifact already exists.
""")

    code("setup", '''
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path.cwd()
DATA = Path(os.environ.get("PERSEUS_DATA", REPO / ".cache"))
MODEL = "openai-community/gpt2"

MODEL_DIR = DATA / "models" / MODEL / "classic"           # weights.bin.zip + client.npz
WEIGHTS = MODEL_DIR / "weights.bin.zip"
POOL = DATA / "pools" / "openwebtext_gpt2.npy"            # calibration token pool
CONFIGS = Path(os.environ.get("CONFIGS_PATH",
                              REPO / "configs/model/approximation/gpt2_base_n32/configs.json"))
ORACLE = DATA / "oracle" / "gpt2" / "all_blocks_io"       # the teacher-forced decode oracle
GRAPHS = REPO / "graphs" / "gpt2_decode_python_n32"              # captured graph of one forward
PLAN = REPO / "bootstrap_placements" / "gpt2_decode_python_n32"  # the main plan

PYTHON = sys.executable


def run(*args, **env):
    subprocess.run([str(a) for a in args], check=True, cwd=REPO, env={**os.environ, **env})


print(f"model   {MODEL}")
print(f"data    {DATA}")
print(f"configs {CONFIGS}")
''')

    md("export-md", """
## 1. Export the weights

Splits the checkpoint in two:
- `weights.bin.zip` — the tensors the server evaluates under encryption
- `client.npz` — what stays client-side in the clear (the token and position embeddings)

This is `perseus-export`; `--out` is the model directory, the export adds the `classic/`
tag below it.
""")

    code("export", '''
if WEIGHTS.exists():
    print("already exported")
else:
    run(PYTHON, "-m", "perseus.export", "--model", MODEL, "--out", MODEL_DIR.parent)

for f in ("weights.bin.zip", "client.npz"):
    print(f"  {f:<16} {(MODEL_DIR / f).stat().st_size / 1e6:8.1f} MB")
''')

    md("pool-md", """
## 2. Calibration token pool

Tokens of what the model will actually see, streamed once from the calibration dataset
(`perseus/configs/dataset/openwebtext.yaml`: OpenWebText, 2M tokens) and cached as an
`.npy`. It feeds both the calibration and the decode oracle below. Building it needs
network; everything after this cell is offline.
""")

    code("pool", '''
import numpy as np
from omegaconf import OmegaConf

from perseus.calibrate import data

DATASET = OmegaConf.load(REPO / "perseus/configs/dataset/openwebtext.yaml")
if POOL.exists():
    pool = np.load(POOL, mmap_mode="r")
else:
    pool = data.load_token_pool(MODEL, DATASET)      # streams + caches under the HF home
    POOL.parent.mkdir(parents=True, exist_ok=True)
    np.save(POOL, np.asarray(pool))
print(f"token pool: {len(pool):,} tokens -> {POOL}")
''')

    md("calibrate-md", """
## 3. Calibrate the approximations

`perseus-calibrate` fits every nonlinearity (LayerNorm inverse-sqrt, softmax, GELU, the
CutMax argmax) over the ranges the model visits on the pool and writes `configs.json`.

The checkout ships the calibration the paper ran with
(`configs/model/approximation/gpt2_base_n32/configs.json`), and the shipped graph and plan
are bound to it: a re-calibration changes the polynomials, so it needs a new capture and a
new plan (steps 5–6). The cell therefore reuses the shipped file and only calibrates when
`CONFIGS_PATH` points somewhere new. Calibration runs on the GPU by default
(`device=cpu` works, slowly).
""")

    code("calibrate", '''
if CONFIGS.exists():
    print(f"using {CONFIGS}")
else:
    run(PYTHON, "-m", "perseus.calibrate", "model=gpt2", "dataset=openwebtext",
        f"calib_out_path={CONFIGS}")

cfg = json.load(open(CONFIGS))
print(f"  {len(cfg['norm'])} norm, {len(cfg['softgelu'])} gelu, {len(cfg['softmax'])} softmax"
      " sites + cutmax")
''')

    md("oracle-md", """
## 4. The decode oracle

The decode gate (`TASK=decode`) feeds the encrypted model a fixed token sequence and compares
its per-position argmax and next-token distribution against the plaintext model. The
reference is the raw model's own forward on a slice of the token pool: block-0 inputs
(`all_blocks_L00_T<T>.json`) and final logits (`all_blocks_lm_head_steps_T<T>.json`) for
each horizon `T`. The runner reads the `T = 128` pair (`STEPS_T`).
""")

    code("oracle", '''
if all((ORACLE / f).exists() for f in ("all_blocks_L00_T128.json",
                                        "all_blocks_lm_head_steps_T128.json")):
    print("oracle already generated")
else:
    run(PYTHON, "scripts/utils/gen_gpt2_oracle.py", "--model", MODEL, "--pool", POOL,
        "--out", ORACLE, "--T", "16", "32", "64", "128")
print(f"oracle: {sorted(p.name for p in ORACLE.glob('*.json'))}")
''')

    md("capture-md", """
## 5. Capture the graph

The planner works on a record of one forward: every ciphertext edge with its packing period
and largest coefficient. `run_decode --capture` runs the first token eagerly on the GPU and
writes `graphs/gpt2_decode_python_n32/block_<b>/graph.json`, the argmax stage included; the
checkout ships the capture the plans were computed on, so this cell normally does nothing. A
capture is bound to the runtime build and to `configs.json`: change either and re-capture.
""")

    code("capture", '''
graphs = sorted(GRAPHS.glob("block_*/graph.json"))
if graphs:
    print(f"captured graph present: {len(graphs)} blocks in {GRAPHS.relative_to(REPO)}")
else:
    run(PYTHON, "-m", "examples.gpt2_from_primitives.run_decode", "--tokens", "1", "--argmax",
        "--capture", GRAPHS)
    graphs = sorted(GRAPHS.glob("block_*/graph.json"))

eager = sum(sum(n["op_type"] == "auto_bootstrap" for n in json.load(open(g))["nodes"])
            for g in graphs)
# one forward of the 12 blocks, the LM head and the encrypted argmax; the eager decode row
# of the README runs 912 per token
print(f"  the capture fired {eager} reactive bootstraps")
''')

    md("plan-md", """
## 6. Plan the bootstraps

`scripts/make_plans.sh gpt2_decode_python_n32` runs the min-cut placer over the captured graph
with the recipe in the directory's `PLAN_CMD.txt` (blocks, then the argmax stage) and writes one
`block_<b>_placement.json` per block. Pure Python, seconds on the CPU. `bash
scripts/make_plans.sh` (no argument) regenerates every shipped plan, the baselines included;
`bootstrap_placements/README.md` maps each directory to its table row.
""")

    code("plan", '''
plans = sorted(PLAN.glob("block_*_placement.json"))
if plans:
    print(f"plan present: {len(plans)} blocks in {PLAN.relative_to(REPO)}")
else:
    run("bash", "scripts/make_plans.sh", "gpt2_decode_python_n32")
    plans = sorted(PLAN.glob("block_*_placement.json"))

# what the paper counts: cut placements + hint-triggered + deliberate refreshes
s = [json.load(open(p))["summary"] for p in plans]
placed = sum(x["num_placements"] for x in s)
total = sum(x["total_bootstraps"] for x in s)
print(f"  {total} planned bootstraps ({placed} from the cut, the rest hint-fired or deliberate)")
''')

    md("run", """
## Run it

The artifacts are in place. The measured row is one command:

```bash
python -m examples.gpt2_from_primitives.run_decode --tokens 16 --argmax \
    --plan bootstrap_placements/gpt2_decode_python_n32     # planned decode, prints [decode] PASS
python -m examples.gpt2_from_primitives.run_decode --tokens 16 --argmax   # the same without a plan
```

The notebooks run the model interactively (`NB=<name> bash scripts/run_notebooks.sh`
executes one headless):

| notebook | |
|---|---|
| `gpt2_torch_forward.ipynb` | a prompt of your choosing, generated under encryption through the C++ driver |
| `gpt2_perseus_nn.ipynb` | the same through the Python modules, split into client and server |
| `custom_encrypted_model.ipynb` | your own model from `Enc*` modules: capture, plan, planned rerun |
| `client_server_minimal.ipynb` | the key bundle protocol in three cells |
""")

    nb = nbf.v4.new_notebook()
    nb.cells = [c for _, c in cells]
    for i, (slug, c) in enumerate(cells):
        c.id = f"setup-{i:02d}-{slug}"       # pinned: regeneration is byte-identical
    nb.metadata = {"language_info": {"name": "python"}}
    return nb


if __name__ == "__main__":
    nbf.write(build(), OUT)
    print(f"wrote {OUT}")
