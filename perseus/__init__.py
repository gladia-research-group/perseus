"""perseus — GPU FHE inference for transformer LMs under CKKS.

Python side of the pipeline: HF model acquisition (`perseus.hub`), approximation
calibration (`perseus.calibrate`), weight packing (`perseus.export`). The CUDA/C++
runtime (capture / plan / decode / prefill / generate) is the `cuda_cachemir` binary;
native bindings (`perseus._core`) come with the scikit-build-core phase.

Heavy deps (torch/transformers/hydra) are imported inside submodules, not here.
"""

__version__ = "0.1.0"
