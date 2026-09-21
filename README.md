# ⚠️ HEAT REPRODUCIBILITY SNAPSHOT ⚠️

# **This commit exists ONLY to reproduce the HEAT publication.**

### It is the exact, frozen runtime state the HEAT work was measured on (development line `dab6cf7`, branch `bert-heat`, 2026-08-15; FIDESlib at the matching `heat-baseline` snapshot of [FIDESlib32bits](https://github.com/gladia-research-group/FIDESlib32bits)). **Do not base new work on it** — current development is the tip of [`main`](../../tree/main).

---

# perseus

Transformer inference under CKKS fully homomorphic encryption, on GPU. Models run as
`torch.nn.Module`s — a sentence or an image goes in, a prediction comes out, and nothing
is decrypted in between.

Built on [FIDESlib](https://github.com/alexzilligmm/pyFIDESlib) + OpenFHE, driven from Python via `perseus._core`.

## What runs today

| model | task | encrypted | wall |
|---|---|---|---|
| **GPT-2** (124M) | generation, prefill → hand-off → autoregressive decode | 12 blocks + LM head + **argmax** | 53.6 s/token |
| **GPT-2 medium** (355M) | generation | 24 blocks + head + argmax | 93.9 s/token |
| **ViT-B/16** (86M) | image classification | 12 blocks + final LN + classifier | 110.1 s |
| **BERT-base** (110M) | SST-2 sentiment | 12 encoder blocks | 83.4 s |

Encrypted argmax means the sampled token never leaves the ciphertext domain during
generation — the model feeds back its own encrypted prediction.

## Quick start

```bash
# 1. build the dependencies (patched OpenFHE + FIDESlib -> deps/), then the python module
bash scripts/install_deps.sh
bash scripts/build_core.sh

# 2. make the artifacts for a model (export, calibrate, plan)
NB=setup FAMILY=bert sbatch scripts/run_notebooks.sh

# 3. run it
NB=bert_torch sbatch scripts/run_notebooks.sh
```

The notebooks in `notebooks/` are the readable entry point:

| notebook | |
|---|---|
| `setup_artifacts` | HF checkpoint → encrypted model, in Python (`FAMILY=bert\|vit\|gpt2`) |
| `bert_torch_forward` | sentence in, sentiment out |
| `vit_torch_forward` | image in, class out |
| `gpt2_torch_forward` | prompt in, continuation out |
| `custom_encrypted_model` | build your own encrypted module, capture it, plan it |

## How it works

Four artifacts turn a HuggingFace checkpoint into an encrypted model:

1. **export** — split the weights into what the server evaluates encrypted
   (`weights.bin.zip`) and what stays client-side in the clear (`client.npz`)
2. **calibrate** — fit polynomial approximations for every nonlinearity (LayerNorm
   inverse-sqrt, softmax, GELU) over the ranges the model actually visits
3. **capture** — record the op-graph of one forward
4. **plan** — place bootstraps deliberately by min-cut instead of reactively

Only 1–3 are required. Without a plan the model still runs, in *eager* mode; the plan is
a performance optimization worth 10–70% depending on the row.

**Calibrate on the deployment distribution.** This is not a nicety: calibrating BERT on
generic web text instead of its own SST-2 fine-tuning set put the LayerNorm inverse-sqrt
3.9× outside its convergence basin and the encrypted model failed outright from block 2.
`scripts/utils/bert_operating_point.py` checks this in two minutes on CPU, before any GPU
time.

## What stays in the clear

The embedding lookup always runs client-side. Beyond that it differs by model:

- **ViT** — final LayerNorm and the classifier run **encrypted**; the client adds one bias
- **GPT-2** — fully **encrypted**
- **BERT** — the head (pooler → `tanh` → classifier) runs **client-side**, as `tanh` is not yet supported in this library.


## Layout

```
perseus/            python: nn modules, calibration, planner, export
src/model/           CUDA drivers, one TU per architecture
scripts/run_task.sh  the benchmark/gate entry point (TASK=…, STAGE=…)
notebooks/           the demos above
docs/STATUS_matrix   frozen numbers and what produced them
CLAUDE.md            operational recipe — build, envs, failure triage
```
