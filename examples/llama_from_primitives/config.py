"""configs.json of the Llama port (calibrate.py) -> the perseus.impl config structs:
``norm`` (RMSNorm sites layers.<b>.ln1 / ln2 and lnf, Chebyshev seed only), ``silu`` (THOR
composite, the GeluCfg the shared activation kernel takes), ``softmax`` and ``cutmax``."""
from __future__ import annotations

import dataclasses
import json

from perseus.impl.config import CutMaxCfg, GeluCfg, NormCfg, SoftmaxCfg, parse_cutmax, parse_gelu, parse_softmax


def parse_rmsnorm(j: dict) -> NormCfg:
    """An RMSNorm site: the centering scale, the output scale and the Chebyshev seed band of
    the inverse sqrt (norm.rms_norm runs the seed + Newton path only)."""
    return NormCfg(method="remez", epsilon=float(j["eps"]), center_scale=float(j["center_scale"]),
                   taylor_z0=0.0, nr_iters=0, inv_out_scale=float(j["inv_out_scale"]),
                   cheb_coeffs=tuple(float(c) for c in j["cheb_coeffs"]), cheb_lo=float(j["cheb_lo"]),
                   cheb_hi=float(j["cheb_hi"]), cheb_nr_iters=int(j["cheb_nr_iters"]))


@dataclasses.dataclass
class LlamaConfigs:
    model: dict
    norm: dict                  # site -> NormCfg
    act: list                   # per layer GeluCfg (the SiLU's THOR composite)
    softmax: list               # per layer SoftmaxCfg
    cutmax: CutMaxCfg | None

    @property
    def n_layers(self):
        return int(self.model["n_layers"])

    def block(self, b: int):
        return (self.norm[f"layers.{b}.ln1"], self.norm[f"layers.{b}.ln2"], self.softmax[b], self.act[b])

    @property
    def ln_f(self):
        return self.norm["lnf"]


def load_configs(path: str) -> LlamaConfigs:
    with open(path) as f:
        j = json.load(f)
    L = int(j["model"]["n_layers"])
    return LlamaConfigs(model=j["model"],
                        norm={k: parse_rmsnorm(v) for k, v in j["norm"].items()},
                        act=[parse_gelu(j["silu"][f"layers.{b}.mlp.act"]) for b in range(L)],
                        softmax=[parse_softmax(j["softmax"][f"layers.{b}.attn"]) for b in range(L)],
                        cutmax=parse_cutmax(j["cutmax"]) if "cutmax" in j else None)
