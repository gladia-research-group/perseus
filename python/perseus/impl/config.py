"""configs.json -> the per-site approximation configs, with the JSON->struct name map of
include/config_loader.h (eps -> epsilon, z0 -> taylor_z0, n_squarings -> log2delta1,
refinement_iters -> log2delta2, thor_p1_a/b default -1/1) and the CutMax schedule
(cutmax_config_from_calib, src/algorithms/nonlinear/cutmax.cu).
"""
from __future__ import annotations

import dataclasses
import json
from typing import Optional


@dataclasses.dataclass
class NormCfg:
    method: str                      # "remez" | "taylor"
    epsilon: float
    center_scale: float
    taylor_z0: float
    nr_iters: int = 16
    inv_out_scale: float = 1.0
    Ncoeffs: tuple = ()
    Dcoeffs: tuple = ()
    lin_alpha: float = 0.0
    lin_beta: float = 0.0
    gs_lo: float = 0.0
    gs_hi: float = 0.0
    gs_iters: int = 0
    center_scale_sq: tuple = ()
    precise_var_bts: bool = False
    cheb_coeffs: tuple = ()          # Chebyshev seed of inv_out_scale/sqrt on [cheb_lo, cheb_hi]
    cheb_lo: float = 0.0
    cheb_hi: float = 0.0
    cheb_nr_iters: int = 0

    @property
    def descale(self) -> float:
        """ln_gamma_descale (include/weight_loader.h)."""
        return 1.0 / self.inv_out_scale if self.method == "remez" else 1.0

    def c_eff_sq(self, pos: int) -> float:
        """norm.cu: the per-position centering scale squared."""
        if self.center_scale_sq:
            p = max(0, min(int(pos), len(self.center_scale_sq) - 1))
            return float(self.center_scale_sq[p])
        return float(self.center_scale) ** 2


@dataclasses.dataclass
class SoftmaxCfg:
    log2delta1: int
    log2delta2: int
    clip_lo: float
    clip_hi: float
    poly_coeffs: tuple
    init_alpha: float
    init_beta: float
    refine_alpha: tuple
    refine_beta: tuple
    gs_iters_scaled: int
    gs_iters_refine_scaled: int
    per_step_refine_iters: tuple
    cheb_coeffs: tuple = ()
    cheb_a: float = -1.0
    cheb_b: float = 1.0
    sm_kc_r: tuple = ()
    gs_iters_first: int = 0          # the first division's count under SM_GS_FIRST
    # SM_CHEB_GS (impl smcheb): Chebyshev-scaled Goldschmidt counts per round and the mid-round residual targets
    recip_iters_cheb: tuple = ()
    recip_targets: tuple = ()
    # reference-key centring (attention.complex_qkt): per head the key whose score is subtracted before the exp
    # (REF_SINK key 0, REF_SELF the query's own key, REF_PREV the one before it); empty = the mid-centred exp.
    # ref_window: the most keys the centred config covers (the broadcast ladder's length)
    ref_keys: tuple = ()
    ref_window: int = 0

    def kc_r(self, i: int, kc: int) -> float:
        """cachemir_attention.cu: the per-step per-kc refine scaling."""
        if self.sm_kc_r and self.log2delta2 > 0:
            C = len(self.sm_kc_r) // self.log2delta2
            kpos = max(0, min(int(kc) - 1, C - 1))
            return float(self.sm_kc_r[i * C + kpos])
        if kc > 0:
            return min(1.0, 4.0 / kc)
        return 1.0


REF_SINK, REF_SELF, REF_PREV = 0, 1, 2
_REF_NAMES = {"sink": REF_SINK, "self": REF_SELF, "prev": REF_PREV}


def ref_key(kind: int, kc: int) -> int:
    """The cache index of a reference key at cache count kc (the previous key of the first query is its own)."""
    return 0 if kind == REF_SINK else kc - 1 if kind == REF_SELF else max(kc - 2, 0)


@dataclasses.dataclass
class GeluCfg:
    method: str
    xmax: float = 1.0
    thor_p1: tuple = ()
    thor_p2: tuple = ()
    thor_p1_cheb: tuple = ()
    thor_p2_cheb: tuple = ()
    thor_p1_a: float = -1.0
    thor_p1_b: float = 1.0
    thor_p2_a: float = -1.0
    thor_p2_b: float = 1.0
    cheb_coeffs: tuple = ()
    cheb_a: float = -1.0
    cheb_b: float = 1.0
    cheb_eps: float | None = None     # coefficient trim (relative to the largest), None = poly.CHEB_COEFF_EPS


@dataclasses.dataclass
class CutMaxIter:
    p: int
    c: float
    m: float
    s2_hi: float
    passes: int
    ex2: int
    ca: float = 0.0
    cb: float = 0.0
    casc_iters: int = 0
    # the scaled normaliser (head.cutmax_argmax_packed): 1/sigma from x_scale * sum (x - mu)^2 in (0, 1] by the
    # Newton steps (a, b) of poly.inv_sqrt_newton_scaled; empty = the cascade above
    x_scale: float = 0.0
    newton: tuple = ()


@dataclasses.dataclass
class CutMaxCfg:
    iters: tuple
    newton_per_pass: int = 4
    newton_polish: int = 0
    sum_lo: float = 0.0
    sum_hi: float = 0.0
    gs_sum_iters: int = 4
    entry_scale: float = 1.0 / 256.0
    # the final 1/S as (1/sqrt(sum_scale S))^2 by poly.inv_sqrt_newton_scaled; empty = Goldschmidt on [sum_lo, sum_hi]
    sum_scale: float = 0.0
    sum_newton: tuple = ()


@dataclasses.dataclass
class ModelCfg:
    n_layers: int
    n_embd: int
    n_head: int
    n_inner: int


@dataclasses.dataclass
class Configs:
    model: ModelCfg
    norm: dict
    softmax: dict
    gelu: dict
    cutmax: Optional[CutMaxCfg]

    def block(self, b: int):
        base = f"transformer.h.{b}"
        return (self.norm[base + ".ln_1"], self.norm[base + ".ln_2"],
                self.softmax[base + ".attn"], self.gelu[base + ".mlp.act"])

    @property
    def ln_f(self):
        return self.norm["transformer.ln_f"]


def _tup(v):
    return tuple(float(x) for x in v) if v is not None else ()


def parse_norm(j: dict) -> NormCfg:
    method = j.get("method", "taylor")
    cfg = NormCfg(method=method, epsilon=float(j["eps"]), center_scale=float(j["center_scale"]),
                  taylor_z0=float(j["z0"]), nr_iters=int(j.get("nr_iters", 16)))
    if method == "remez":
        cfg.inv_out_scale = float(j["inv_out_scale"])
        cfg.Ncoeffs = _tup(j["Ncoeffs"])
        cfg.Dcoeffs = _tup(j["Dcoeffs"])
        cfg.lin_alpha = float(j["lin_alpha"])
        cfg.lin_beta = float(j["lin_beta"])
        cfg.gs_lo = float(j["gs_lo"])
        cfg.gs_hi = float(j["gs_hi"])
        cfg.gs_iters = int(j["gs_iters"])
        cfg.center_scale_sq = _tup(j.get("center_scale_sq"))
        cfg.precise_var_bts = bool(j.get("precise_var_bts", False))
        if "cheb_coeffs" in j:
            cfg.cheb_coeffs = _tup(j["cheb_coeffs"])
            cfg.cheb_lo, cfg.cheb_hi = float(j["cheb_lo"]), float(j["cheb_hi"])
            cfg.cheb_nr_iters = int(j["cheb_nr_iters"])
    return cfg


def parse_softmax(j: dict) -> SoftmaxCfg:
    return SoftmaxCfg(
        log2delta1=int(j["n_squarings"]), log2delta2=int(j["refinement_iters"]),
        clip_lo=float(j["clip_lo"]), clip_hi=float(j["clip_hi"]),
        poly_coeffs=_tup(j["poly_coeffs"]), init_alpha=float(j["init_alpha"]),
        init_beta=float(j["init_beta"]), refine_alpha=_tup(j["refine_alpha"]),
        refine_beta=_tup(j["refine_beta"]), gs_iters_scaled=int(j["gs_iters_scaled"]),
        gs_iters_refine_scaled=int(j["gs_iters_refine_scaled"]),
        per_step_refine_iters=_tup(j["per_step_refine_iters"]),
        cheb_coeffs=_tup(j.get("cheb_coeffs")), cheb_a=float(j.get("cheb_a", -1.0)),
        cheb_b=float(j.get("cheb_b", 1.0)), sm_kc_r=_tup(j.get("sm_kc_r")),
        gs_iters_first=int(j.get("gs_iters_first", 0)),
        recip_iters_cheb=_tup(j.get("recip_iters_cheb")), recip_targets=_tup(j.get("recip_targets")),
        ref_keys=tuple(_REF_NAMES[r] for r in j.get("ref_keys", ())), ref_window=int(j.get("ref_window", 0)))


def parse_gelu(j: dict) -> GeluCfg:
    method = j["method"]
    if method == "thor_composite":
        return GeluCfg(method=method, xmax=float(j["xmax"]), thor_p1=_tup(j["thor_p1"]),
                       thor_p2=_tup(j["thor_p2"]), thor_p1_cheb=_tup(j.get("thor_p1_cheb")),
                       thor_p2_cheb=_tup(j.get("thor_p2_cheb")),
                       thor_p1_a=float(j.get("thor_p1_a", -1.0)), thor_p1_b=float(j.get("thor_p1_b", 1.0)),
                       thor_p2_a=float(j.get("thor_p2_a", -1.0)), thor_p2_b=float(j.get("thor_p2_b", 1.0)))
    if method == "chebyshev":
        # xmax: with GELU_FOLD the up-projection carries 1/xmax and the output mask xmax (as for thor_composite)
        return GeluCfg(method=method, xmax=float(j.get("xmax", 1.0)), cheb_coeffs=_tup(j["cheb_coeffs"]),
                       cheb_a=float(j["cheb_a"]), cheb_b=float(j["cheb_b"]),
                       cheb_eps=(float(j["cheb_eps"]) if j.get("cheb_eps") is not None else None))
    raise ValueError(f"GELU method {method!r} is not ported (thor_composite / chebyshev only)")


def _steps(v):
    return tuple((float(a), float(b)) for a, b in v)


def parse_cutmax(j: dict) -> CutMaxCfg:
    T = len(j["p"])
    if "newton" in j:                     # the scaled normalisers (calibrate.approximations.cutmax.fit_scaled)
        for k in ("c", "m", "x_scale", "newton"):
            if len(j[k]) != T:
                raise ValueError("cutmax calib: ragged/empty schedule arrays")
        iters = tuple(CutMaxIter(int(j["p"][i]), float(j["c"][i]), float(j["m"][i]), 0.0, 0, 0,
                                 x_scale=float(j["x_scale"][i]), newton=_steps(j["newton"][i])) for i in range(T))
        return CutMaxCfg(iters=iters, entry_scale=float(j["entry_scale"]), sum_scale=float(j["sum_scale"]),
                         sum_newton=_steps(j["sum_newton"]))
    for k in ("c", "m", "s2_hi", "passes", "ex2", "chord_a", "chord_b", "cascade_iters"):
        if len(j[k]) != T:
            raise ValueError("cutmax calib: ragged/empty schedule arrays")
    iters = tuple(CutMaxIter(int(j["p"][i]), float(j["c"][i]), float(j["m"][i]),
                             float(j["s2_hi"][i]), int(j["passes"][i]), int(j["ex2"][i]),
                             float(j["chord_a"][i]), float(j["chord_b"][i]),
                             int(j["cascade_iters"][i])) for i in range(T))
    return CutMaxCfg(iters=iters, newton_per_pass=int(j["newton_per_pass"]),
                     newton_polish=int(j["newton_polish"]), sum_lo=float(j["sum_lo"]),
                     sum_hi=float(j["sum_hi"]), gs_sum_iters=int(j["gs_sum_iters"]),
                     entry_scale=float(j["entry_scale"]))


def load_configs(path: str) -> Configs:
    from perseus._paths import resolve
    with open(resolve(path)) as f:
        j = json.load(f)
    m = j.get("model", {})
    model = ModelCfg(int(m.get("n_layers", 0)), int(m.get("n_embd", 0)),
                     int(m.get("n_head", 0)), int(m.get("n_inner", 0)))
    return Configs(model=model,
                   norm={k: parse_norm(v) for k, v in j.get("norm", {}).items()},
                   softmax={k: parse_softmax(v) for k, v in j.get("softmax", {}).items()},
                   gelu={k: parse_gelu(v) for k, v in j.get("softgelu", {}).items()},
                   cutmax=parse_cutmax(j["cutmax"]) if "cutmax" in j else None)
