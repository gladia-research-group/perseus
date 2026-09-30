# Approximation configs

Encrypted models can only add and multiply, so every nonlinear function (LayerNorm's inverse
square root, softmax, GELU and the encrypted argmax) runs as a polynomial or an iterative
approximation. Each `configs.json` holds those approximations for one model and one parameter
set: for every place the function appears, the input interval, the polynomial degree and the
number of iterations, fitted by `perseus-calibrate` on the value ranges the model reaches on
OpenWebText, plus a record of how it was produced under `meta`. The runtime reads the file given
by `CONFIGS_PATH` as it is.

| config | parameters | notes |
|---|---|---|
| `gpt2_base_n32` | 32-bit (the default) | GELU as a Chebyshev series; its `cutmax` section drives the encrypted argmax |
| `gpt2_base` | 64-bit | the same fit, GELU as a plain polynomial |

The shipped plans are computed for these files: change a value and the forward pass must be
recorded and planned again (README, steps 2 and 3). `scripts/utils/lint_approx_config.py`
checks a config before recording.
