"""Generate notebooks/setup_artifacts.ipynb — the artifacts every model notebook loads.

Artifacts only: export, pool, calibrate, plan. It never runs the encrypted model; the
model notebooks do that. Model-agnostic via the FAMILIES registry (bert|vit|gpt2), which
holds nothing but names — export dispatches on config.model_type and calibration is
hydra configs, so steps 1-3 are uniform.

Also injects an idempotent "Prerequisites" cell into the model notebooks.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Setting up encrypted inference

Encrypted models inference requires producing several artifacts to complete the porting to the encrypted world. Apart from exporting weights from the HuggingFace format to the CUDA one we need to:
- calibrate approximation
- create a plan for a fixed bts schedule.

| # | artifact | made by | ETA |
|---|---|---|---|
| 1 | `weights.bin.zip`, `client.npz` | `perseus.export` | seconds, CPU |
| 2 | calibration pool (tokens or images) | `perseus.calibrate.data` | needs network |
| 3 | `configs.json` — the fitted approximations | `perseus.calibrate` | ~5 min, GPU |
| 4 | the **plan** — where bootstraps go | `perseus.plan.plan_graph_dir` | ~2 min, CPU |
""")

code('''
import os, json, time, subprocess, sys
from pathlib import Path

import numpy as np
import torch                                   # torch before _core (NCCL load order)
from omegaconf import OmegaConf

FAMILY  = os.environ.get("FAMILY", "bert")
REPO    = Path.cwd()
SCRATCH = os.environ["SCRATCH"]
PYTHON  = sys.executable
os.environ.setdefault("HF_HOME", f"{SCRATCH}/.cache")
os.environ.setdefault("HF_HUB_CACHE", f"{SCRATCH}/.cache")

FAMILIES = {
    "bert": dict(
        hf_name="textattack/bert-base-uncased-SST-2",
        hydra=dict(model="bert_base", dataset="sst2", approximation="bert_base"),
        calib_extra=["model.block_size=32"],   # == the deployed T (real arm caps at 32)
        configs_name="bert_base",
        graph_dir=".cache/graph_bert_base",
        plan_dir="bootstrap_placements/planned_bert_base",
        keep=("goldschmidt", "inv_sqrt_newton", ".var", "ln_affine", "remez"),  # BERT needs keep
        probe=["scripts/utils/bert_operating_point.py", "--block-size", "32"],
        notebook="bert_torch_forward.ipynb",
    ),
    "vit": dict(
        hf_name="google/vit-base-patch16-224",
        hydra=dict(model="vit_base_224", dataset="tiny_imagenet", approximation="vit_base"),
        calib_extra=["model.resolution=80"],
        configs_name="vit_base",
        graph_dir=".cache/graph_vit_base",
        plan_dir="bootstrap_placements/planned_vit_base",
        keep=(".var",),                        # relax — the shipped ViT recipe (2026-07-27)
        probe=None,
        notebook="vit_torch_forward.ipynb",
    ),
    "gpt2": dict(
        hf_name="openai-community/gpt2",
        hydra=dict(model="gpt2", dataset="openwebtext", approximation="gpt2_cutmax"),
        calib_extra=[],                        # block_size null = inferred from the config
        configs_name="gpt2_base",
        graph_dir=".cache/graph_gpt2_base",
        plan_dir="bootstrap_placements/planned_gpt2_base",
        keep=(),                               # decode = bare smart cut, no keeps
        probe=None,
        notebook="gpt2_torch_forward.ipynb",
    ),
}

F = FAMILIES[FAMILY]
MODEL_DIR = f"{SCRATCH}/.cache/perseus/models/{F['hf_name']}/classic"
CONFIGS   = f"{REPO}/configs/model/approximation/{F['configs_name']}/configs.json"
DATASET   = OmegaConf.load(REPO / f"perseus/configs/dataset/{F['hydra']['dataset']}.yaml")
run = lambda *a: subprocess.run([str(x) for x in a], check=True, cwd=REPO)

print(f"FAMILY = {FAMILY}   ({F['hf_name']})")
print(f"  weights -> {MODEL_DIR}")
print(f"  configs -> {CONFIGS}")
print(f"  dataset -> {DATASET.name} ({DATASET.kind})")
''')

md("""
## 1. Export the weights

Splits the checkpoint in two:
- `weights.bin.zip` — the tensors the server evaluates under encryption
- `client.npz` — what stays client-side in the clear (embeddings, and any head with no
  registered FHE approximation)

`export.adapter_for` dispatches on `config.model_type`, so this is the same call for
every architecture.
""")

code("""
if not Path(f"{MODEL_DIR}/weights.bin.zip").exists():
    run(PYTHON, "-m", "perseus.export", "--model", F["hf_name"],
        "--out", str(Path(MODEL_DIR).parent))
else:
    print("already exported")

for f in ("weights.bin.zip", "client.npz"):
    p = Path(f"{MODEL_DIR}/{f}")
    print(f"  {f:<18} {p.stat().st_size/1e6:8.1f} MB")
""")

md("""
## 2. Calibration pool

Samples of what the model will actually see, cached as an `.npy` next to the HF cache.
Text families stream tokens, image families stream preprocessed tensors.

Calibrate on the **deployment distribution**, not generic corpora. BERT calibrated on
openwebtext instead of SST-2 lands its LayerNorm inverse-sqrt 3.9x outside the
Goldschmidt basin, and the encrypted run dies at block 2.

Building needs network. Compute nodes here have none, so run this cell once on a login
node; everything after it is offline-safe.
""")

code('''
from perseus.calibrate import data

if DATASET.kind == "image":
    pool = data.load_image_pool(F["hf_name"], DATASET)
    print(f"image pool: {pool.shape}")
else:
    pool = data.load_token_pool(F["hf_name"], DATASET)
    print(f"token pool: {len(pool):,} tokens")
''')

md("""
## 3. Calibrate

Fits every nonlinearity (LayerNorm inverse-sqrt, softmax, GELU) over the ranges the
model actually visits and writes `configs.json`.

`block_size` must equal the deployed sequence length: it sets the softmax denominator
range and the per-position LayerNorm rescale. The registry pins it per family.
""")

code('''
if not Path(CONFIGS).exists():
    t0 = time.perf_counter()
    run(PYTHON, "-m", "perseus.calibrate",
        f"model={F['hydra']['model']}", f"dataset={F['hydra']['dataset']}",
        f"approximation={F['hydra']['approximation']}",
        *F["calib_extra"], f"calib_out_path={CONFIGS}")
    print(f"calibrated in {time.perf_counter()-t0:.0f} s")
else:
    print(f"already calibrated (delete {Path(CONFIGS).name} to refit)")

cfg = json.load(open(CONFIGS))
print(f"  {len(cfg['norm'])} norm, {len(cfg['softgelu'])} gelu, {len(cfg['softmax'])} softmax sites")
''')

md("""
### Check the operating point first

Runs the plaintext model on what the calibrator saw and on what you will encrypt, then
prints where each approximation lands inside its fitted band. Two minutes on CPU.

The first out-of-band site by depth is the block the encrypted run will die at. Cheaper
to find here than as a `Decode(): approximation error is too high` ten blocks in.
""")

code('''
if F["probe"]:
    run(PYTHON, *F["probe"])
else:
    print(f"no probe wired for {FAMILY} — scripts/utils/bert_operating_point.py is the template")
''')

md("""
## Run it

Artifacts done — the model runs now, in eager mode. Running it belongs to the model
notebooks:

| FAMILY | notebook | |
|---|---|---|
| `bert` | `bert_torch_forward.ipynb` | sentence in, sentiment out |
| `vit` | `vit_torch_forward.ipynb` | image in, class out |
| `gpt2` | `gpt2_torch_forward.ipynb` | prefill, hand-off, generate |

Current limits: embeddings and unapproximated heads run client-side; BERT is capped at
32 tokens (~76% of SST-2 validation) until the multi-chunk arm is validated; encrypted
logits come out compressed toward zero, so argmax is reliable but magnitudes are not
calibrated.
""")

md("""
## 4. Capture and plan — optional, slow

Only buys speed: eager places bootstraps reactively, a plan places them deliberately.
BERT 96.5s -> 83.4s, ViT-80 118s -> 110s.

Needs a capture — one forward in `Sync` with full rotation keys, so it gets its own job
(~35 min for BERT-base):

```bash
TASK=bert STAGE=capture GATE_BLOCKS=12 sbatch scripts/run_task.sh
```

Planning is then ~2 min on CPU. The default cut is **relax** (`.var`); BERT is the
exception and keeps the LayerNorm bootstraps, since relax leaves its last blocks with no
feasible placement. Output goes to `_staged` — diff the placements before promoting.
""")

code('''
from perseus.plan import PlanOptions, plan_graph_dir

graphs = sorted(Path(F["graph_dir"]).glob("block_*/graph.json"))
staged = f"{F['plan_dir']}_staged"

if not graphs:
    print(f"no capture at {F['graph_dir']}")
    print(f"  make one:  TASK=<task> STAGE=capture sbatch scripts/run_task.sh")
else:
    summaries = plan_graph_dir(F["graph_dir"], staged,
                               PlanOptions(erase_keep_steps=F["keep"]))

    # Count from the WRITTEN files, not summary["num_placements"] — the summary counts
    # only the min-cut deliberate placements (a fraction), while the plan the runtime
    # loads is the full `placements` array. Mixing the two makes an identical plan look
    # like a 90% regression.
    def n_placements(d):
        return sum(len(json.load(open(p)).get("placements", []))
                   for p in sorted(Path(d).glob("block_*_placement.json")))

    eager = sum(sum(n["op_type"] == "auto_bootstrap" for n in json.load(open(g))["nodes"])
                for g in graphs)
    print(f"\\n{len(summaries)} blocks -> {staged}")
    print(f"  placements {n_placements(staged)}   (capture fired {eager} eager auto-bootstraps)")

    live = Path(F["plan_dir"])
    if live.is_dir():
        print(f"  live plan {live.name}: {n_placements(live)} placements — compare before promoting")
''')

md("""
Three things that decide whether a plan binds:

1. **Config- and binary-bound.** Recalibrate or rebuild and you re-capture and re-plan.
   A stale plan throws `[plan_level_error]`, it does not degrade quietly.
2. **Chain-bound.** One recipe per plan directory. A block's plan pins the level its
   predecessors produce, so plans from two recipes cannot be spliced.
3. **`unplanned_bts=0` is the gate.** A plan that fails to bind still returns correct
   output, reactively — the counter is the only signal. The exit code is not evidence;
   read the `PASS` marker.

Point `FHE_BOOTSTRAP_PLACEMENTS_DIR` at the staged plan and run `Threaded` to use it.
""")

nb.cells = C
nb.metadata = {"language_info": {"name": "python"}}
nbf.write(nb, "notebooks/setup_artifacts.ipynb")
print("wrote notebooks/setup_artifacts.ipynb")

MARK = "<!-- prereq-cell -->"
NOTE = MARK + """
> **Prerequisites.** This notebook loads artifacts it does not create: the exported
> weights (`weights.bin.zip`, `client.npz`), the calibrated `configs.json`, and — for the
> planned path — a bootstrap plan. Run **`setup_artifacts.ipynb`** first if you do not
> have them.
"""

for path in ("notebooks/bert_torch_forward.ipynb",
             "notebooks/vit_torch_forward.ipynb",
             "notebooks/gpt2_torch_forward.ipynb",
             "notebooks/custom_encrypted_model.ipynb"):
    try:
        n = nbf.read(path, as_version=4)
    except FileNotFoundError:
        print(f"  (skip {path}: not found)")
        continue
    n.cells = [c for c in n.cells if MARK not in c.source]      # idempotent
    n.cells.insert(1, nbf.v4.new_markdown_cell(NOTE.strip()))
    nbf.write(n, path)
    print(f"  prereq note -> {path}")
