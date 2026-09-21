import functools
import json
import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

import perseus.calibrate.approximations  # noqa: F401  (registers built-ins)
from perseus import artifacts, export
from perseus._log import configure_cli_logging
from perseus.calibrate import data
from perseus.calibrate.discovery import infer_block_size
from perseus.calibrate.engine import calibrate_model

log = logging.getLogger(__name__)

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="calibrate")
def main(cfg: DictConfig):
    configure_cli_logging()
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
    inherited = []
    if out_path.exists():                       # merge into an existing config
        with open(out_path, encoding="utf-8") as f:
            previous = json.load(f)
        inherited = sorted(k for k in previous if k not in calib and k != "meta")
        calib = {**previous, **calib}
        if inherited:
            log.warning("calibrate: %s already existed; keeping its sections %s untouched "
                        "(only %s were refit this run)", out_path, inherited,
                        sorted(k for k in calib if k not in inherited and k != "meta"))
    calib["meta"] = artifacts.calibration_meta(
        model=cfg.model.name, dataset=str(cfg.dataset.get("name", cfg.dataset.kind)),
        approximation=cfg.approximation.name, n_batches=int(cfg.n_calib_batches),
        sections=sorted(k for k in calib if k not in ("meta", "model")),
        inherited=inherited)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)
    log.info(f"[save] wrote calibration to {out_path}")


if __name__ == "__main__":
    main()
