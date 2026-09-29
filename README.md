# Perseus

Code for *Perseus: A Bootstrap Placer for Faster Encrypted Transformer Inference*
(Zirilli, Marincione, Kornaropoulos, Ateniese, Rodolà).

Perseus runs GPT-2 under CKKS fully homomorphic encryption on one GPU and decides **where** and
**how** to bootstrap from a single profiling pass of the model: every ciphertext edge is recorded
with its packing period and the largest coefficient of its polynomial, and an iterated minimum
cut places the refreshes, each typed with the correction factor its data admits and, where the
packing is periodic, with a cheaper sparse bootstrap. On encrypted GPT-2 decoding it executes
416 bootstraps per token against 584–894 for prior placers and decodes at 11.78 s/token against
16.09–21.40 s/token. The runtime is a 32-bit composite-scaling port of
[FIDESlib](https://github.com/CAPS-UMU/FIDESlib)
([FIDESlib32bits](https://github.com/gladia-research-group/FIDESlib32bits)) that regenerates the
random half of every key-switching key in-kernel and bit-packs the rest.

## Results

Results of the Python implementation (`perseus.impl`, `examples/gpt2_from_primitives`) in its
default configuration, measured on one NVIDIA RTX PRO 6000 Blackwell with logN = 16, 128-bit
security, 27 composite levels over 54 primes (pairs of 27-bit primes per level, q0 a pair of 28-bit primes). The 64-bit
reference chain uses 28 levels of 53-bit primes, with automatic sparse routing (`SPARSE_AUTO=2`)
and `CORRECTION_FACTOR=7` — the chain default of 0 is safe for a dense refresh and not for a
sparse one. Workload: GPT-2 (124M) decoding through all 12 blocks, encrypted argmax included;
KL is the median over decoded tokens of the divergence between encrypted and plaintext
next-token distributions; latency is the mean (± 95% confidence interval) over the steady-state
tokens of two isolated 16-token sessions.

GPT-2 decode, per token. DaCapo is the released DaCapo compiler's (`hecate-opt`) own site
selection and Orion the released Orion solver's (patched to run on transformer graphs), both on
every block including the encrypted argmax; Fhelipe is our reimplementation (the released tool
does not produce a plan on these graphs). Every baseline runs with our correction-factor
selection, sparse dispatch and level bookkeeping; the planner repairs what a baseline's model
misses with counted refreshes (Orion 68 per token, DaCapo none). Perseus and Fhelipe plans are
pruned of redundant refreshes (Fhelipe's published post-pass); DaCapo and Orion run unpruned, as
released (`PLAN_PRUNE=1` gives their pruned variants). The three baselines run with 48 levels:
at 50 their refreshes can start past the measured bootstrap envelope, which the runtime refuses.
Orion misses the reference-rank gate at one token (rank 9, limit 8).

| placer | planned bts | executed bts | median KL | e2e s/token | plan directory |
|---|---|---|---|---|---|
| eager, 64-bit | — | 552 | 0.024 | 20.74 ± 0.04 | (no plan, `--chain n64`) |
| eager, 32-bit | — | 912 | 0.064 | 23.49 ± 0.05 | (no plan) |
| Fhelipe | 561 | 584 | 0.034 | 16.10 ± 0.04 | `python/fhelipe` |
| Orion | 871 | 894 | 0.141 | 21.40 ± 0.06 | `python/orion` |
| DaCapo | 595 | 618 | 0.067 | 16.09 ± 0.18 | `python/dacapo` |
| **Perseus, 64-bit** | 384 | 407 | 0.002 | 16.38 ± 0.06 | `gpt2_decode_python_n64` |
| **Perseus, 32-bit** | **393** | **416** | **0.025** | **11.78 ± 0.11** | `gpt2_decode_python_n32` |

Key-switching keys (GPT-2 decode, the Perseus 32-bit plan):

| configuration | s/token | used VRAM |
|---|---|---|
| in-kernel `a` regeneration + `b` packing (shipping) | 11.86 | 32.55 GiB |
| regeneration only (`FIDESLIB_KSK_PACK=0`) | 12.19 (+3%) | 34.49 GiB |
| packing only (`FIDESLIB_KSK_REGEN=0`) | 14.18 (+20%) | 41.61 GiB |

`bootstrap_placements/README.md` maps every plan directory to its paper row and records the
recipe that regenerates it; `tests/test_paper_plans.py` checks the regeneration.

## Install

Linux, CUDA 12.6 or newer, a GPU with 64 GB or more (the decode peaks at about 33 GiB, the C++
decode at about 48 GiB),
Python 3.11+, CMake 3.24+, gcc 12+, `libarchive-dev`, and NCCL (`libnccl2` + `libnccl-dev`,
or `NCCL_HOME` pointing at a prefix with `lib/libnccl.so` and `include/nccl.h`). Clone with the FIDESlib32bits submodule:

```bash
git clone --recurse-submodules https://github.com/gladia-research-group/perseus
cd perseus
uv sync                                  # or: python -m venv .venv && .venv/bin/pip install -e ".[hf,dev]"
```

That is the pure-Python half: HF export (`perseus-export`), calibration (`perseus-calibrate`)
and the planner (`perseus-plan`). Everything under `perseus.nn` needs the CUDA extension:

```bash
NATIVE_SIZE=32 bash scripts/install_deps.sh      # patched OpenFHE + FIDESlib32bits -> deps_n32/
CHAIN=n32 bash scripts/local_build_core.sh       # perseus/_core.n32.so + build_py_n32/bin/cuda_cachemir
.venv/bin/python -c "import perseus.nn"          # says exactly what is missing otherwise
```

`NATIVE_SIZE=64` / `CHAIN=n64` build the 64-bit reference chain next to it (`deps_n64/`,
`_core.n64.so`); the import name is a symlink to the active chain. The GPU-less client role
(`perseus._client`, OpenFHE only) builds with `CHAIN=n32 bash scripts/local_build_client.sh`.
`scripts/local_env.sh` holds every default (CUDA_HOME, thread count, data paths); an exported
variable always wins.

On a machine without root, two of those dependencies need spelling out. The prefix `NCCL_HOME`
points at must hold **both** `lib/libnccl.so` (to link against) and `lib/libnccl.so.2` (to load at
run time), and `$NCCL_HOME/lib` has to be on `LD_LIBRARY_PATH` or the built `_core` and
`cuda_cachemir` will not start; the `nvidia-nccl-cu12` wheel `uv sync` installs ships only the
second name, so symlink both from `.venv/lib/python3.11/site-packages/nvidia/nccl/lib/`. For
LibArchive, `apt-get download libarchive-dev && dpkg-deb -x libarchive-dev_*.deb prefix` unpacks
the headers without installing anything; pass them as `LibArchive_INCLUDE_DIR` and
`LibArchive_LIBRARY`, and repoint the unpacked `libarchive.so` symlink at the system runtime,
which it does not resolve to once extracted.

## Artifacts

Three artifacts turn the HuggingFace checkpoint into the encrypted model
(`notebooks/setup_artifacts.ipynb` walks through the same steps, and skips any that exist):

```bash
.venv/bin/perseus-export --model openai-community/gpt2 --out .cache/models/openai-community/gpt2   # weights.bin.zip + client.npz (CPU)
.venv/bin/python -c 'import numpy as np; from omegaconf import OmegaConf; from perseus.calibrate import data; \
    np.save(".cache/pools/openwebtext_gpt2.npy", np.asarray(data.load_token_pool("openai-community/gpt2", \
    OmegaConf.load("perseus/configs/dataset/openwebtext.yaml"))))'   # the calibration token pool (network, once)
.venv/bin/perseus-calibrate model=gpt2 dataset=openwebtext                                       # -> configs/model/approximation/gpt2_base*/configs.json (shipped; GPU, device=cpu works)
.venv/bin/python scripts/utils/gen_gpt2_oracle.py --model openai-community/gpt2 \
    --pool .cache/pools/openwebtext_gpt2.npy --out .cache/oracle/gpt2/all_blocks_io              # the teacher-forced decode oracle (CPU)
```

The shipped `configs/model/approximation/gpt2_base_n32/configs.json` is the calibration the paper
ran with; the plans are bound to it and to the runtime, so a re-calibration needs a new capture
and plan (below). `PERSEUS_DATA` (default `.cache/`) is where the runner looks for the weights
and the oracle.

## Reproduce a row

```bash
source scripts/local_env.sh                                           # CHAIN=n32 unless exported
DECODE=".venv/bin/python -m examples.gpt2_from_primitives.run_decode --tokens 16 --argmax"
$DECODE --plan bootstrap_placements/gpt2_decode_python_n32            # the 32-bit row: 416 bts/token
$DECODE                                                               # eager (no plan)
$DECODE --plan bootstrap_placements/python/dacapo                     # a baseline (python/orion, python/fhelipe)
FIDESLIB_KSK_REGEN=0 $DECODE --plan bootstrap_placements/gpt2_decode_python_n32   # stored keys, no in-kernel regeneration
```

The 64-bit row runs the same command with `CHAIN=n64` exported before `source scripts/local_env.sh`,
the import symlink pointed at `_core.n64.so`, and
`--chain n64 --plan bootstrap_placements/gpt2_decode_python_n64`.

The C++ decode runs the same model through the runtime's composites, with its own plans
(`gpt2_decode_n32`, `gpt2_decode_n64`, `baselines/`, `ablations/`):

```bash
TASK=decode CHAIN=n32 RUNNER=cuda bash scripts/run_task.sh
FHE_BOOTSTRAP_PLACEMENTS_DIR=bootstrap_placements/baselines/dacapo TASK=decode RUNNER=cuda bash scripts/run_task.sh
```

A run passes when the driver prints `[decode] PASS` (python) or `SUMMARY … completed=16/16`
(cuda) with `unplanned_bts=0 weight_relevels=0`. The verdict rides the quantities that measure
the encrypted computation: the reference token stays near the top of our distribution
(`ref_rank`, gated by `GATE_REF_RANK_MAX`) and the divergence stays far below the value a
broken chain saturates at (`GATE_KL_MAX`; a detonated run reads KL > 50). Exact top1 agreement
is printed, not required, because it depends on the reference sequence: with the shipped
oracle the 32-bit row reads 15/16 and the 64-bit row 15–16/16, but a self-generated oracle
resolves near-ties differently. `GATE_MIN_TOP1=15` demands an exact count. Latency depends on
the host: the walls above were taken on an idle machine with the process pinned to the GPU's NUMA node (`numactl --cpunodebind=<node> --preferred=<node>`).

## Capture and plan

The planner works on a captured graph of one forward:

```bash
TASK=decode STAGE=capture bash scripts/run_task.sh      # graphs/gpt2_decode_n32/block_<b>/graph.json (sync forward)
bash scripts/make_plans.sh main                          # -> bootstrap_placements/gpt2_decode_n32 (CPU, seconds)
bash scripts/make_plans.sh                               # every plan in the paper
FUSED_SM_DEN=0 SPARSE_AUTO=0 SPARSE_BTS_SLOTS=0 FHE_GRAPH_DIR=graphs/gpt2_decode_n32_dense \
  TASK=decode STAGE=capture bash scripts/run_task.sh     # the dense capture the two dense 32-bit arms are planned from
```

`python -m perseus.plan --help` lists the planner options (`--mag-safety` κ, `--err-target` τ,
`--cf-max`, `--sparse-slots`, `--placer {min_cut,orion,dacapo,fhelipe}`). The planner is pure Python
(`perseus/plan`), needs no GPU, and prices every site with a measured accuracy table,
`perseus/plan/data/bts_accuracy_<chain>.json`, selected by `--acc-chain`. The 32-bit rows are
priced from the measured 32-bit table; the 64-bit rows are priced by the analytic error model
(`--acc-chain ''`), which is how the paper planned them. A measured 64-bit table ships as well:
it reproduces the 64-bit sparse arm exactly and moves the dense arm by one bootstrap. Either
table is re-measured on the GPU with `CHAIN=n32 bash scripts/utils/sweep_bts_accuracy.sh`.

## Python API

```python
import numpy as np
from perseus import session
from perseus.profile import SessionProfile
from perseus.nn import EncGELU, EncLinear, EncSequential, calibrate_sequential

d, e, real = 1024, 4096, 768                       # packed widths; 768 real features
rng = np.random.default_rng(0)
W1 = rng.standard_normal((d, e)) * 0.5 / np.sqrt(real); W1[real:, :] = 0; W1[:, 3072:] = 0
W2 = rng.standard_normal((e, d)) * 0.5 / np.sqrt(3072); W2[3072:, :] = 0; W2[:, real:] = 0

with session(profile=SessionProfile.custom_n32()) as s:     # keygen + GPU context
    model = EncSequential(EncLinear("fc1", d, e, weight=W1), EncGELU("act"),
                          EncLinear("fc2", e, d, weight=W2)).bind(s)
    calibrate_sequential(model, rng.standard_normal((32, real)) * 0.3)   # fit GELU on your data
    x = rng.standard_normal(real) * 0.3
    y = s.decrypt(model(s.encrypt(x)), d=real)               # encrypted forward
```

`EncGPT2.from_pretrained(...)` runs the whole model; `EncClient` / `EncServer` split the roles
across a trust boundary (`docs/SECURITY_MODEL.md`); `docs/PYTHON_API.md` is the reference.
Notebooks: `setup_artifacts`, `gpt2_torch_forward`, `gpt2_perseus_nn`, `custom_encrypted_model`,
`client_server_minimal` (`NB=<name> bash scripts/run_notebooks.sh` executes one headless).

## Layout

```
perseus/           python: nn modules, sessions, calibration, the planner (perseus/plan)
src/, include/     the CUDA runtime (GPT-2 driver, cachemir packing, plan-bound bootstrapping)
src/bindings/      the pybind11 layer (perseus._core, perseus._client)
third_party/       FIDESlib32bits submodule; openfhe-n32/ holds the OpenFHE patch series
bootstrap_placements/, graphs/   the paper's plans and the captured graphs behind them
configs/           calibrated approximation configs (gpt2_base = 64-bit, gpt2_base_n32 = 32-bit)
scripts/           install_deps.sh, local_build_*.sh, run_task.sh, make_plans.sh, run_notebooks.sh
tests/             CPU tier (pytest), GPU tier (pytest -m gpu tests/gpu)
docs/              PYTHON_API.md, SECURITY_MODEL.md
```

## Citation

```bibtex
@article{perseus2026,
  title   = {Perseus: A Bootstrap Placer for Faster Encrypted Transformer Inference},
  author  = {Zirilli, Alessandro and Marincione, Davide and Kornaropoulos, Evgenios M. and Ateniese, Giuseppe and Rodol{\`a}, Emanuele},
  year    = {2026}
}
```

## License

[Business Source License 1.1](LICENSE): free for any non-commercial use (research, academic,
evaluation); commercial use requires prior written permission from the authors. On the Change
Date (2030-08-18) the license converts to GPLv2. `third_party/FIDESlib` keeps its own license.
