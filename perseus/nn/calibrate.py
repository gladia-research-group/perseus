import json
import os
import tempfile

import numpy as np

from .. import _core
from .._env import current_chain
from .container import EncSequential

_SECTIONS = ("softgelu", "norm", "softmax")   # configs.json sections a module can carry


def _mirror_items(model):
    """(index-name, enc module, torch mirror) per depth-1 child; raises on gaps."""
    import torch.nn as nn

    items, missing = [], []
    for i, mod in enumerate(model):
        tm = mod.torch_mirror()
        if tm is None:
            missing.append(f"{i}: {type(mod).__name__}")
        items.append((str(i), mod, tm))
    if missing:
        raise ValueError(
            "calibrate_sequential: these children provide no torch_mirror() — a "
            "module either mirrors itself or cannot be calibrated:\n  "
            + "\n  ".join(missing))
    return items, nn.Sequential(*(tm for _, _, tm in items))


def _approx_set(name_or_cfg):
    from omegaconf import OmegaConf
    if not isinstance(name_or_cfg, str):
        return OmegaConf.create(name_or_cfg)
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "approximation",
                        f"{name_or_cfg}.yaml")
    return OmegaConf.load(os.path.abspath(path))


def _thor_to_cheb_module():
    from ..calibrate import thor_to_cheb
    return thor_to_cheb


def calibrate_sequential(model, samples, approximation="gpt2", n_batches=8,
                         out_path=None, apply=True, cheb_basis=None):
    import torch
    from omegaconf import OmegaConf

    import perseus.calibrate.approximations  # noqa: F401  (registers built-ins)

    from ..calibrate.engine import calibrate_model

    if not isinstance(model, EncSequential):
        raise TypeError("calibrate_sequential expects an EncSequential")
    items, seq = _mirror_items(model)
    seq = seq.eval()

    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    d_in = None
    for _, mod, _ in items:
        d_in = getattr(mod, "d_in", None)
        if d_in is not None:
            break
    if d_in is not None and x.shape[1] < d_in:   # zero-pad to the packed width
        x = np.concatenate([x, np.zeros((x.shape[0], d_in - x.shape[1]))], axis=1)

    batches = np.array_split(x, min(n_batches, len(x)))
    it = iter(batches)

    def get_batch():
        return torch.as_tensor(next(it), dtype=torch.float32)

    cfg = OmegaConf.create({"approximation": OmegaConf.to_container(
                                _approx_set(approximation), resolve=True),
                            "n_calib_batches": len(batches)})
    with torch.no_grad():
        calib = calibrate_model(seq, get_batch, cfg)

    if cheb_basis is None:
        cheb_basis = current_chain() == "n32"
    if cheb_basis and "softgelu" in calib:
        conv = _thor_to_cheb_module()
        for site, g in calib["softgelu"].items():
            if isinstance(g, dict) and "thor_p1" in g:
                conv.convert_site(g, site, m1=1.10)   # the validated p1 domain (B16)
    if cheb_basis and "norm" in calib:
        for ncfg in calib["norm"].values():
            if isinstance(ncfg, dict) and int(ncfg.get("nr_iters", 1)) < 3:
                ncfg["nr_iters"] = 3

    calib.setdefault("model", {"n_layers": 1, "n_embd": d_in or 0,
                               "n_head": 1, "n_inner": d_in or 0})
    if out_path is None:
        out_path = os.path.join(tempfile.mkdtemp(prefix="enc_calib_"), "configs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)
    parsed = _core.load_configs(out_path)

    if apply:
        seen = {}
        for name, mod, _ in items:
            cfg_name = getattr(mod, "cfg_name", None)
            if cfg_name is None:
                continue
            for section in _SECTIONS:
                table = getattr(parsed, section)
                # a mirror may wrap its calibrated leaf ("0.ln"): match the child prefix
                hits = [k for k in table if k == name or k.startswith(name + ".")]
                if hits:
                    key = hits[0]
                    table = {name: table[key]}
                if name in table:
                    if seen.get(cfg_name) not in (None, name):
                        import warnings
                        warnings.warn(f"calibrate_sequential: cfg name {cfg_name!r} is "
                                      f"shared by sites {seen[cfg_name]} and {name}; "
                                      f"the later fit wins", stacklevel=2)
                    seen[cfg_name] = name
                    mod.apply_calibration(section, table[name], probe=x[0])
    return parsed
