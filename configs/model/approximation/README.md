# Approximation configs

One `configs.json` per chain, produced by `perseus-calibrate` on OpenWebText and used as-is by
the runtime (`CONFIGS_PATH`):

| config | chain | notes |
|---|---|---|
| `gpt2_base_n32` | 32-bit composite (the paper's) | Chebyshev-basis GELU; the CutMax section drives the encrypted argmax |
| `gpt2_base` | 64-bit reference | same calibration, monomial GELU |

Each file carries per-site LayerNorm inverse-sqrt, softmax and GELU approximations (intervals,
degrees, iteration counts) fitted on the ranges the model visits, plus provenance under `meta`.
The shipped plans are bound to these files: change a value and the graph must be recaptured and
re-planned (README → Capture and plan). `scripts/utils/lint_approx_config.py` checks a config
before a capture.
