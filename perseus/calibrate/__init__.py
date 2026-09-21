"""FHE approximation calibration: any HF causal LM → the runtime's configs.json.

A registry-driven framework. Each supported op kind is an `Approximation`
(site matcher + sample collector + fitter) registered in
`perseus.calibrate.approximations`; the engine is model-agnostic — a model
calibrates iff every nonlinearity it contains is matched, and refuses with a
precise report otherwise. Mirrors the CUDA backend's generality
(nn.Sequential-style models and transformers alike).

Approximation SETS are hydra group configs (`configs/approximation/*.yaml`):
one subtree per kind, presence = participation, each kind sees only its own
subtree. Scale to new sets (models, packings, ablations) by adding files and
selecting `approximation=<name>`.

    registry.py          the Approximation contract + registry
    discovery.py         classify a module tree against the registry
    engine.py            collect + fit orchestration (calibrate_model)
    approximations/      built-ins: norm (LN/RMS), gelu, softmax, cutmax
                         (+ cutmax_sim: plaintext replica of the runtime op)
    numerics.py          shared primitives (Remez engine, Chebyshev exp series,
                         quantiles, GS inits + reciprocal convergence core)
    data.py              calibration token pools from HF datasets
    cli.py               the hydra entry point (`perseus-calibrate`)

Extending: register a new `Approximation` — nothing else changes.
"""

from perseus.calibrate.registry import Approximation, register  # noqa: F401
