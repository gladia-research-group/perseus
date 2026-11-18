"""The composite-sign ReLU calibration of ResNet-20 (calibrate/approximations/relu.py): per-site max |x| over
CIFAR-10 training images, the ladder fitted once, written as the configs.json "relu" section.

    python -m perseus.examples.resnet.calibrate --degs 15,15,27 --eps 0.0078125 [--n 2000] [--margin 1.2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from omegaconf import OmegaConf


def site_max(sd, x, batch=500):
    import torch
    from . import weights as R
    mx = {}

    def act(t, site):
        mx[site] = max(mx.get(site, 0.0), float(t.abs().max()))
        return torch.clamp_min(t, 0)
    for i in range(0, len(x), batch):
        R.reference(sd, x[i:i + batch], act)
    return mx


def main(argv=None):
    from perseus._paths import CONFIGS
    from perseus.calibrate.approximations.relu import fit_relu_section
    from . import weights as R
    ap = argparse.ArgumentParser()
    ap.add_argument("--degs", default="15,15,27")
    ap.add_argument("--eps", type=float, default=2 ** -7)
    ap.add_argument("--margin", type=float, default=1.2)
    ap.add_argument("--n", type=int, default=2000, help="training images")
    ap.add_argument("--name", default="resnet20_cifar10")
    a = ap.parse_args(argv)
    sd = R.state_dict()
    x, _ = R.cifar10(a.n, split="train")
    cfg = OmegaConf.create({"degs": [int(d) for d in a.degs.split(",")], "eps": a.eps, "margin": a.margin})
    section = fit_relu_section(site_max(sd, x), cfg)
    out = os.path.join(CONFIGS, a.name, "configs.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"relu": section, "meta": {"model": "chenyaofo/cifar10_resnet20", "dataset": "cifar10/train",
                                         "n_images": a.n}}, open(out, "w"), indent=2)
    print(f"[calibrate] {len(section['sites'])} sites, depth {section['depth']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
