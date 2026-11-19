"""configs.json of the decoder port (calibrate.py) -> the perseus.impl config structs:
``norm`` (RMSNorm sites layers.<b>.ln1 / ln2 [/ ln1p / ln2p, the post norms] and lnf, Chebyshev seed only), ``act``
(the MLP activation's THOR composite, the GeluCfg the shared activation kernel takes; older files name it ``silu`` or
``gelu``), ``softmax``, ``softcap`` (the attention logit soft-cap's Chebyshev series, Arch.attn_softcap) and ``cutmax``."""
from __future__ import annotations

import dataclasses
import json

from perseus.impl.config import CutMaxCfg, NormCfg, parse_cutmax, parse_gelu, parse_softmax


def parse_rmsnorm(j: dict) -> NormCfg:
    """An RMSNorm site: the centering scale, the output scale and the Chebyshev seed band of
    the inverse sqrt (norm.rms_norm runs the seed + Newton path only)."""
    return NormCfg(method="remez", epsilon=float(j["eps"]), center_scale=float(j["center_scale"]),
                   taylor_z0=0.0, nr_iters=0, inv_out_scale=float(j["inv_out_scale"]),
                   cheb_coeffs=tuple(float(c) for c in j["cheb_coeffs"]), cheb_lo=float(j["cheb_lo"]),
                   cheb_hi=float(j["cheb_hi"]), cheb_nr_iters=int(j["cheb_nr_iters"]))


@dataclasses.dataclass(frozen=True)
class SoftcapCfg:
    """cap tanh(s / cap) as a Chebyshev series in the score s on [lo, hi]."""
    cap: float
    lo: float
    hi: float
    coeffs: tuple


@dataclasses.dataclass
class LlamaConfigs:
    model: dict
    norm: dict                  # site -> NormCfg
    act: list                   # per layer GeluCfg (the activation's THOR composite)
    softmax: list               # per layer SoftmaxCfg
    cutmax: CutMaxCfg | None
    softcaps: list = dataclasses.field(default_factory=list)    # per layer SoftcapCfg (empty: no soft-cap)

    def post(self, b: int):
        """(attention output norm, MLP output norm) of block b, or None without post norms."""
        k = f"layers.{b}.ln1p"
        return (self.norm[k], self.norm[f"layers.{b}.ln2p"]) if k in self.norm else None

    def softcap(self, b: int):
        return self.softcaps[b] if self.softcaps else None

    @property
    def n_layers(self):
        return int(self.model["n_layers"])

    def block(self, b: int):
        return (self.norm[f"layers.{b}.ln1"], self.norm[f"layers.{b}.ln2"], self.softmax[b], self.act[b])

    @property
    def ln_f(self):
        return self.norm["lnf"]


def load_configs(path: str) -> LlamaConfigs:
    from perseus._paths import resolve
    with open(resolve(path)) as f:
        j = json.load(f)
    L = int(j["model"]["n_layers"])
    act = next(j[k] for k in ("act", "silu", "gelu") if k in j)
    sc = [SoftcapCfg(float(c["cap"]), float(c["lo"]), float(c["hi"]), tuple(float(v) for v in c["coeffs"]))
          for c in (j["softcap"][f"layers.{b}.attn"] for b in range(L))] if j.get("softcap") else []
    return LlamaConfigs(model=j["model"],
                        norm={k: parse_rmsnorm(v) for k, v in j["norm"].items()},
                        act=[parse_gelu(act[f"layers.{b}.mlp.act"]) for b in range(L)],
                        softmax=[parse_softmax(j["softmax"][f"layers.{b}.attn"]) for b in range(L)],
                        cutmax=parse_cutmax(j["cutmax"]) if "cutmax" in j else None, softcaps=sc)
