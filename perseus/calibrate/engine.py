"""The calibration engine: registry-driven, model-agnostic.

    calibrate_model(model, get_batch, cfg)
        discovery  -> match every site against the set's approximations
        collection -> one hooked pass over n_calib_batches
        fitting    -> each approximation turns its samples into a config section

The engine knows nothing about architectures or op kinds — both live entirely
in the registry entries (perseus.calibrate.approximations).
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from perseus.calibrate.discovery import ModelMap, discover
from perseus.calibrate.registry import Approximation, registered


@torch.no_grad()
def _collect(model, mm: ModelMap, approxes: list[Approximation], collectors,
             get_batch, n_batches: int) -> None:
    modules = dict(model.named_modules())
    handles = []
    for approx in approxes:
        if approx.model_level:
            continue
        for site in mm.by_kind(approx.kind):
            handles.extend(collectors[approx.kind].attach(site.name, modules[site.name]))
    try:
        for _ in tqdm(range(n_batches), desc="collect", unit="batch"):
            out = model(get_batch())
            for approx in approxes:
                if approx.model_level:
                    collectors[approx.kind].on_output(out)
    finally:
        for h in handles:
            h.remove()


def calibrate_model(model, get_batch, cfg) -> dict:
    """Calibrate an instantiated model; returns the configs.json dict.

    An approximation participates iff its kind has a subtree in the selected
    approximation set (`cfg.approximation`), and receives ONLY that subtree —
    discovery still aborts on any uncovered nonlinearity, so a set can never
    silently skip a required op."""
    sets = cfg.approximation
    approxes = [a for a in registered().values() if a.kind in sets]
    mm = discover(model, approxes)

    collectors = {a.kind: a.make_collector(sets[a.kind]) for a in approxes}
    _collect(model, mm, approxes, collectors, get_batch, cfg.n_calib_batches)

    calib: dict = {}
    if mm.dims is not None:
        calib["model"] = mm.dims
    for approx in approxes:
        collected = collectors[approx.kind].finalize()
        if collected is None:
            continue
        calib[approx.section] = approx.fit_section(collected, sets[approx.kind])
    return calib
