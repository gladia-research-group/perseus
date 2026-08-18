"""Calibration CLI: any HF causal LM → the FHE runtime's configs.json.

    perseus-calibrate model.name=EleutherAI/gpt-neo-125m approximation.name=neo125m
    perseus-calibrate approximation=gpt2_cutmax        # set with the argmax schedule

The model is instantiated from the hub, its architecture discovered against
the approximation registry, and — if every nonlinearity is supported — the
fitted config is written, ready for the CUDA backend. Config tree:
perseus/configs/ (hydra); approximation sets live in configs/approximation/.
"""

import functools
import json
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

import perseus.calibrate.approximations  # noqa: F401  (registers built-ins)
from perseus import export
from perseus.calibrate import data
from perseus.calibrate.discovery import infer_block_size
from perseus.calibrate.engine import calibrate_model

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="calibrate")
def main(cfg: DictConfig):
    torch.manual_seed(42)

    model = instantiate(cfg.model.instance)
    model.to(device=cfg.device, dtype=torch.float32).eval()

    if cfg.dataset.kind == "image":
        if cfg.model.resolution:   # sub-native patch grid: interpolate pos embeddings
            model.forward = functools.partial(model.forward,
                                              interpolate_pos_encoding=True)
        get_batch = data.compose_image_get_batch(
            model_name=cfg.model.name,
            dataset=cfg.dataset,
            device=cfg.device,
            micro_batch=cfg.model.batch_size,
            resolution=cfg.model.resolution,
        )
    else:
        block_size = cfg.model.block_size or infer_block_size(model)
        if not block_size:
            raise ValueError("cannot infer the context length — set model.block_size")
        get_batch = data.compose_get_batch(
            model_name=cfg.model.name,
            dataset=cfg.dataset,
            device=cfg.device,
            block_size=block_size,
            micro_batch=cfg.model.batch_size,
        )

    calib = calibrate_model(model, get_batch, cfg)
    calib = export.remap_sections(calib, model.config.model_type)   # runtime section names

    out_path = Path(
        cfg.calib_out_path
        or Path.cwd() / "configs" / "model" / "approximation"
           / cfg.approximation.name / "configs.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():                       # merge into an existing config
        calib = {**json.load(open(out_path)), **calib}
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"[save] wrote calibration to {out_path}")


if __name__ == "__main__":
    main()
