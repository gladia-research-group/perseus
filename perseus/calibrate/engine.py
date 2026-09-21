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
