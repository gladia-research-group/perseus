import os

from .. import _core, artifacts
from ..hub import cache_dir

WEIGHTS = "weights.bin.zip"


def resolve_artifacts(name, tag="classic", weights=None, configs=None):
    """(weights_zip, configs_json) for `name`, from explicit paths or the perseus cache.
    Raises FileNotFoundError naming the command that produces a missing file."""
    weights = weights or os.path.join(cache_dir("models", name, tag), WEIGHTS)
    if not os.path.isfile(weights):
        raise FileNotFoundError(
            f"no packed weights at {weights}\n  produce them with:  "
            f"perseus-export --model {name} --tag {tag}   (or pass weights=...)")
    if configs is None:
        raise FileNotFoundError(
            "configs= is required: the calibrated approximations (configs.json) are fitted "
            "on YOUR data — produce them with:  perseus-calibrate model=<model> "
            "dataset=<dataset> approximation=<set>   and pass the resulting directory")
    cfg = configs if os.path.isfile(configs) else os.path.join(configs, "configs.json")
    if not os.path.isfile(cfg):
        raise FileNotFoundError(f"no configs.json at {configs}")
    return weights, cfg


def load_artifacts(name, tag="classic", weights=None, configs=None, model_type=None,
                   strict=True):
    """(WeightStore, ParsedConfigs) with the provenance cross-check applied."""
    weights, cfg = resolve_artifacts(name, tag=tag, weights=weights, configs=configs)
    problems = artifacts.check(weights_zip=weights, configs_path=cfg, model_type=model_type)
    errors = [p for p in problems if p.startswith("error:")]
    if errors and strict:
        raise ValueError("artifact mismatch for " + name + ":\n  " + "\n  ".join(errors))
    store = _core.WeightStore.from_zip(weights)
    parsed = _core.load_configs(cfg)
    return store, parsed, problems


def from_pretrained(cls, name, *, tag="classic", weights=None, configs=None, n_layers=None,
                    strict=True, model_type=None):
    """Build `cls(store, configs, n_layers)` from resolved artifacts (see module doc)."""
    store, parsed, _ = load_artifacts(name, tag=tag, weights=weights, configs=configs,
                                      model_type=model_type, strict=strict)
    return cls(store, parsed, n_layers=n_layers)
