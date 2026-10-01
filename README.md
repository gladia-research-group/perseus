<p align="center">
  <a href="https://www.uniroma1.it/en"><img src="assets/sapienza-logo.svg" alt="Sapienza University of Rome" height="76"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://gladia.di.uniroma1.it">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/gladia-logo-white.svg">
      <source media="(prefers-color-scheme: light)" srcset="assets/gladia-logo.svg">
      <img src="assets/gladia-logo.svg" alt="GLADIA Research Group" height="84">
    </picture>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.gmu.edu"><img src="assets/GM-monogramRGB-r.png" alt="George Mason University" height="76"></a>
</p>

<h1 align="center">Perseus: Faster FHE Transformer Inference via Complex-Packing and Sparse Bootstraps</h1>

<p align="center">
  <a href="https://github.com/gladia-research-group/FIDESlib32bits">FIDESlib32bits runtime</a> ·
  <a href="#0-install">Quickstart</a> ·
  <a href="#citation">Citation</a> ·
  <a href="LICENSE">BUSL-1.1 license</a>
</p>

Fully homomorphic encryption (FHE) lets a server run a model on data it cannot read. Under the
CKKS scheme, every multiplication uses up part of an encrypted value's budget of remaining
multiplications (its *levels*). When the budget runs out, the value has to be refreshed by a
*bootstrap*, which costs far more than any other operation in the model. How many bootstraps
run, where they sit and which kind is used therefore decide most of the inference time. Existing
tools choose where to bootstrap from the multiplication count alone, without looking at the
values being refreshed.

**Perseus** records one encrypted forward pass of the model: every encrypted value, how its data
is laid out across the ciphertext, and how large it gets. From that record it plans where and
how to bootstrap. A minimum-cut algorithm chooses the positions; each bootstrap gets the
scaling setting its values allow, and a cheaper bootstrap is used wherever a ciphertext holds
only a few distinct values. On encrypted GPT-2 generation, Perseus runs 416 bootstraps per
token against 584–894 for the other tools, and takes 11.78 s per token against 16.09–21.40 s.

This repository contains the planner; GPT-2 written in Python on the Perseus runtime
(`perseus.impl`, `examples/gpt2_from_primitives`); the recorded forward passes and every plan
behind the results; and the scripts that produce the comparison plans with the released DaCapo
and Orion tools. The runtime is [FIDESlib32bits](https://github.com/gladia-research-group/FIDESlib32bits),
our 32-bit port of the GPU CKKS library [FIDESlib](https://github.com/CAPS-UMU/FIDESlib),
included as a submodule. It also shrinks the key-switching keys, the large public keys that
rotations and multiplications need, so that they fit in GPU memory.

## Results

Encrypted GPT-2 small generating text on one NVIDIA RTX PRO 6000 Blackwell GPU, with a ring
dimension of 2<sup>16</sup> and 128-bit security.

<p align="center">
  <a href="assets/results-table-light.png">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/results-table-dark.png">
      <source media="(prefers-color-scheme: light)" srcset="assets/results-table-light.png">
      <img src="assets/results-table-light.png" alt="Encrypted GPT-2. Perseus 32-bit: 393 planned and 416 executed bootstraps per token, 11.78 s/token, KL 0.026, top-1 93.8%. Perseus 64-bit: 384, 407, 16.38 s, 0.002, 100%. DaCapo: 595, 618, 16.09 s, 0.067, 68.8%. Orion: 871, 894, 21.40 s, 0.142, 68.8%. Fhelipe: 561, 584, 16.10 s, 0.034, 87.5%. No plan, 32-bit: 912 bootstraps, 23.49 s; no plan, 64-bit: 552, 20.74 s." width="707">
    </picture>
  </a>
</p>

Perseus runs on two CKKS parameter sets. The 32-bit one, the default, represents each level
with a pair of 27-bit primes (27 levels); the 64-bit reference uses 28 levels of 53-bit primes.

Half of every key-switching key is pseudo-random: the runtime recomputes it on the GPU from a
seed instead of storing it, and stores the other half bit-packed. The effect on the 32-bit row:

| key storage | s / token | GPU memory used |
|---|---|---|
| recomputed half + bit-packed half (default) | 11.86 | 32.55 GiB |
| recomputed half only (`FIDESLIB_KSK_PACK=0`) | 12.19 (+3%) | 34.49 GiB |
| bit-packed half only (`FIDESLIB_KSK_REGEN=0`) | 14.18 (+20%) | 41.61 GiB |

[`bootstrap_placements/README.md`](bootstrap_placements/README.md) lists every plan in the
repository, including the variants that use only full-size bootstraps and the ablations, with
the recipe that regenerates each one.

## Layout

- `perseus/`
  - `impl/` the Python implementation: encrypted linear layers, attention, normalization and
    activations, plus the two execution modes, recording a forward pass and following a plan.
  - `plan/` the planner: it reads a recorded forward pass, simulates how levels are spent,
    places the bootstraps and picks each one's type. It also holds our ports of the DaCapo,
    Orion and Fhelipe planners.
  - `nn/`, `calibrate/`, `export.py` a PyTorch-like module API, the fitting of the polynomial
    approximations used for nonlinear functions, and the export of HuggingFace weights.
- `examples/gpt2_from_primitives/` GPT-2 decoding and generation on `perseus.impl`, with its
  planning script `make_plan.sh`.
- `graphs/gpt2_decode_python_{n32,n64}/` the recorded forward passes the plans are computed from.
- `bootstrap_placements/` one directory per plan, each with the recipe that regenerates it
  (`PLAN_CMD.txt`).
- `configs/model/approximation/` the fitted approximations (`gpt2_base_n32` for 32-bit,
  `gpt2_base` for 64-bit).
- `scripts/` build, run and planning scripts; `scripts/utils/{dacapo,orion}_upstream/` run the
  released DaCapo and Orion tools.
- `src/`, `include/` the CUDA runtime and its Python bindings: `perseus._core`, and
  `perseus._client`, the client side that runs without a GPU.
- `third_party/` the FIDESlib32bits submodule and our OpenFHE patches.
- `docs/` the Python API reference and the security model.
- `assets/` the logos and the results table (`results-table.tex`, rendered by `render.sh`).

## Python API

There are two Python layers, and both run in one encrypted session. `perseus.impl` is the one
the results use: encrypted models written in Python, which can record a forward pass for the
planner and then run under the resulting plan. `examples/gpt2_from_primitives` is GPT-2
written with it. From the repository root, after the install below and
`source scripts/local_env.sh`, which points at the weights and the test inputs:

```python
import os
from examples.gpt2_from_primitives import env, weights
from examples.gpt2_from_primitives.model import Gpt2Model

PACKING = "cachemir_complex"                    # the data layout the shipped plans assume
sess = env.open_session(device=0, chain="n32", GPT2_PACKING=PACKING)   # parameters + keys
from perseus import _core                       # after the session has set its environment
from perseus.impl import config

model = Gpt2Model(sess.inf, weights.RawStore(os.environ["WEIGHTS_PATH"]),
                  config.load_configs("configs/model/approximation/gpt2_base_n32/configs.json"),
                  core=_core, packing=PACKING)
model.load_plans("bootstrap_placements/gpt2_decode_python_n32")      # without it: no plan
cfg = _core.RunConfig.from_env(); cfg.tokens = 4                    # the first 4 test tokens
out = model.run_decode(_core.read_teacher_forced_inputs(cfg), argmax=True)   # one record per token
print([(r["top1"], r["cutmax"]) for r in out])  # top token after decryption, top token picked encrypted
model.close(); sess.close()
```

Calling `model.set_capture(dir)` instead of `load_plans` records the forward pass the planner
reads (step 2 below). A new model subclasses `ImplModel` and lists its stages, the masks each
step needs and its weights; `perseus/impl/__init__.py` describes the modules, and
`docs/PYTHON_API.md` shows how to write a new operation.

`perseus.nn` is a PyTorch-like module API built on the runtime's C++ layers; the polynomial
approximations of its nonlinear functions are fitted on your data:

```python
import numpy as np
from perseus import session
from perseus.profile import SessionProfile
from perseus.nn import EncGELU, EncLinear, EncSequential, calibrate_sequential

d, e, real = 1024, 4096, 768                       # padded widths; 768 real features
rng = np.random.default_rng(0)
W1 = rng.standard_normal((d, e)) * 0.5 / np.sqrt(real); W1[real:, :] = 0; W1[:, 3072:] = 0
W2 = rng.standard_normal((e, d)) * 0.5 / np.sqrt(3072); W2[3072:, :] = 0; W2[:, real:] = 0

with session(profile=SessionProfile.custom_n32()) as s:     # parameters, keys, GPU context
    model = EncSequential(EncLinear("fc1", d, e, weight=W1), EncGELU("act"),
                          EncLinear("fc2", e, d, weight=W2)).bind(s)
    calibrate_sequential(model, rng.standard_normal((32, real)) * 0.3)   # fit GELU on your data
    x = rng.standard_normal(real) * 0.3
    y = s.decrypt(model(s.encrypt(x)), d=real)               # encrypted forward
```

`EncClient` and `EncServer` split the key holder from the server that computes
(`docs/SECURITY_MODEL.md`); `docs/PYTHON_API.md` is the full reference.

## Pipeline

The complete sequence for GPT-2 on the 32-bit parameters, with the 64-bit differences. Every
file a step produces is already in the repository, so any step can be skipped.

### 0. Install

You need Linux, CUDA 12.6 or newer, a GPU with 48 GB or more of memory (the 32-bit run peaks at
about 33 GiB, the 64-bit one at about 46 GiB), Python 3.11+, CMake 3.24+, gcc 12+,
`libarchive-dev`, and NCCL (`libnccl2` and `libnccl-dev`, or `NCCL_HOME` pointing at a prefix
with `lib/libnccl.so` and `include/nccl.h`).

```bash
git clone --recurse-submodules https://github.com/gladia-research-group/perseus
cd perseus
uv sync                                  # or: python -m venv .venv && .venv/bin/pip install -e ".[hf,dev]"
```

That installs the parts that need no GPU: the weight export, the approximation fitting and the
planner. Encrypted runs also need the CUDA extension:

```bash
NATIVE_SIZE=32 bash scripts/install_deps.sh      # patched OpenFHE + FIDESlib32bits -> deps_n32/
CHAIN=n32 bash scripts/local_build_core.sh       # perseus/_core.n32.so
.venv/bin/python -c "import perseus.nn"          # otherwise reports exactly what is missing
```

`NATIVE_SIZE=64` and `CHAIN=n64` build the 64-bit version next to it (`deps_n64/`,
`_core.n64.so`); the importable module is a symlink to whichever is active.
`scripts/local_env.sh` sets every default, and a variable you export yourself always wins:

| variable | what it points at | default |
|---|---|---|
| `CHAIN` | the parameter set: `n32` (32-bit) or `n64` (64-bit) | `n32` |
| `PERSEUS_DATA` | where the weights, the calibration text and the reference outputs live | `.cache/` |
| `WEIGHTS_PATH` | the exported weights (`weights.bin.zip`) | under `PERSEUS_DATA` |
| `CONFIGS_PATH` | the fitted approximations | unset: the scripts take `gpt2_base_n32` (`gpt2_base` for 64-bit) |
| `ALL_BLOCKS_IO_DIR` | the reference outputs that encrypted runs are checked against | under `PERSEUS_DATA` |

Without root, the prefix `NCCL_HOME` points at must hold **both** `lib/libnccl.so` (to link) and
`lib/libnccl.so.2` (to load), with `$NCCL_HOME/lib` on `LD_LIBRARY_PATH`. The
`nvidia-nccl-cu12` wheel ships only the second name, so symlink both from
`.venv/lib/python3.11/site-packages/nvidia/nccl/lib/`. `apt-get download libarchive-dev &&
dpkg-deb -x libarchive-dev_*.deb prefix` unpacks LibArchive's headers without installing
anything (pass `LibArchive_INCLUDE_DIR` and `LibArchive_LIBRARY`).

### 1. Export the weights, fit the approximations, compute the reference outputs

```bash
.venv/bin/perseus-export --model openai-community/gpt2 --out .cache/models/openai-community/gpt2
.venv/bin/python -c 'import numpy as np; from omegaconf import OmegaConf; from perseus.calibrate import data; \
    np.save(".cache/pools/openwebtext_gpt2.npy", np.asarray(data.load_token_pool("openai-community/gpt2", \
    OmegaConf.load("perseus/configs/dataset/openwebtext.yaml"))))'
.venv/bin/perseus-calibrate model=gpt2 dataset=openwebtext
.venv/bin/python scripts/utils/gen_gpt2_oracle.py --model openai-community/gpt2 \
    --pool .cache/pools/openwebtext_gpt2.npy --out .cache/oracle/gpt2/all_blocks_io
```

The export packs the weights for the runtime. The calibration fits a polynomial approximation
to every nonlinear function (LayerNorm, softmax, GELU and the encrypted argmax) on the value
ranges it sees on OpenWebText; the shipped `gpt2_base_n32/configs.json` is the one the results
use. The last command runs the plaintext model on a fixed piece of text and saves its inputs
and output logits: these reference outputs are what encrypted runs are compared against.

A model fine-tuned for encryption, such as the HEAT GPT-2, is exported on top of its base model
with `--checkpoint` (a `model.pt` file or a HuggingFace repository id):

```bash
.venv/bin/perseus-export --model openai-community/gpt2 \
    --checkpoint gladia/heat-gpt2-small-openwebtext --tag heat --out .cache/models/gladia/heat-gpt2
.venv/bin/python scripts/utils/gen_gpt2_oracle.py --model openai-community/gpt2 \
    --checkpoint gladia/heat-gpt2-small-openwebtext \
    --pool .cache/pools/openwebtext_gpt2.npy --out .cache/oracle/gpt2_heat/all_blocks_io
```

### 2. Record a forward pass

The planner reads a record of one encrypted forward pass: every encrypted value, how its data
is laid out and the largest magnitude it reaches. The first token runs without a plan and is
recorded block by block, the encrypted argmax included:

```bash
source scripts/local_env.sh
.venv/bin/python -m examples.gpt2_from_primitives.run_decode --tokens 1 --argmax \
    --capture graphs/gpt2_decode_python_n32
```

A recording belongs to one runtime build and one approximation config: change either and
record again.

### 3. Plan

The planner runs on the CPU, in seconds per plan (the comparison tools take minutes). Each plan
directory's `PLAN_CMD.txt` holds its recipe, and `make_plans.sh` replays them:

```bash
bash scripts/make_plans.sh gpt2_decode_python_n32     # the 32-bit plan
bash scripts/make_plans.sh                            # every plan in the repository
```

`python -m perseus.plan --help` lists the planner's options, among them the safety margin on
recorded magnitudes (`--mag-safety`), the accuracy target (`--err-target`), the largest scaling
setting (`--cf-max`), the sizes of the cheaper bootstrap (`--sparse-slots`), removing bootstraps
that turn out redundant (`--prune`) and the placement algorithm
(`--placer {min_cut,orion,dacapo,fhelipe}`). The 32-bit plans estimate each bootstrap's error
from measurements (`perseus/plan/data/bts_accuracy_n32.json`), the 64-bit plans from a formula.

The comparison plans come from the released tools. DaCapo needs its compiler, `hecate-opt`,
built once (LLVM/MLIR 18); Orion's solver is downloaded and patched on first use:

```bash
bash scripts/utils/dacapo_upstream/build_hecate.sh
bash scripts/make_plans.sh python/dacapo python/orion python/fhelipe
```

### 4. Run encrypted

```bash
source scripts/local_env.sh
DECODE=".venv/bin/python -m examples.gpt2_from_primitives.run_decode --tokens 16 --argmax"
$DECODE --plan bootstrap_placements/gpt2_decode_python_n32     # the 32-bit row: 416 bootstraps per token
$DECODE --plan bootstrap_placements/python/dacapo              # a comparison plan (also python/orion, python/fhelipe)
$DECODE                                                        # no plan: bootstrap whenever a value runs out of levels
$DECODE --plan bootstrap_placements/gpt2_decode_python_n32_dense --set SPARSE_AUTO=0 --set SPARSE_BTS_SLOTS=0   # full-size bootstraps only
.venv/bin/python -m examples.gpt2_from_primitives.run_generate --prompt 4 --tokens 8 \
    --plan bootstrap_placements/gpt2_decode_python_n32          # generation: each chosen token is fed back encrypted
```

In generation, each new token re-enters the model as the output of a bootstrap instead of being
freshly encrypted, so a plan that supports generation also holds a second plan for block 0,
`block_0_feedback_placement.json`, used for those tokens (`bootstrap_placements/README.md`
lists which plans have one).

For the 64-bit row, export `CHAIN=n64` before `source scripts/local_env.sh`, point the import
symlink at `_core.n64.so`, and pass `--chain n64 --plan bootstrap_placements/gpt2_decode_python_n64`.
`FIDESLIB_KSK_REGEN=0` or `FIDESLIB_KSK_PACK=0` reproduce the key-storage rows.

A run passes when it prints `[decode] PASS`: at every position the reference token stays among
the top few of the encrypted output (`GATE_REF_RANK_MAX`), and the divergence from the plaintext
output stays small (`GATE_KL_MAX`; a broken run shows a KL above 50). Matching the plaintext top
token is reported but not required; the 32-bit row matches 15 of 16. The times above were
measured on an otherwise idle machine, with the process pinned to the GPU's NUMA node
(`numactl --cpunodebind=<node> --preferred=<node>`).

The runtime also has a native C++ driver for the same model (`scripts/run_task.sh`,
`RUNNER=cuda`). No recording or plan ships for it; it needs its own (`STAGE=capture`, then
`perseus-plan`).

## Citation

If you use Perseus, please cite the paper (GitHub's *Cite this repository* button reads
[`CITATION.cff`](CITATION.cff)):

```bibtex
@misc{zirilli2026perseus,
  title  = {Perseus: A Bootstrap Placer for Faster Encrypted Transformer Inference},
  author = {Zirilli, Alessandro and Marincione, Davide and Kornaropoulos, Evgenios M. and Ateniese, Giuseppe and Rodol{\`a}, Emanuele},
  year   = {2026},
  url    = {https://github.com/gladia-research-group/perseus}
}
```

## License

Perseus is released under the [Business Source License 1.1](LICENSE): free for any
non-commercial use (research, academic, evaluation); commercial use requires prior written
permission from the authors. On the Change Date (2030-08-18) the license converts to GPLv2.
`third_party/FIDESlib` keeps its own license.
