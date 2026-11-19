"""The decoder architectures this port runs: what differs between them, read from the checkpoint's config.json.

The port (this package) is a decoder-only transformer on the cachemir decode path: RMSNorm -> GQA attention with RoPE ->
residual -> RMSNorm -> gated MLP -> residual, a final RMSNorm and an LM head. An `Arch` names everything a family changes
in that shape; every module of the port reads it (weights, the BOS prefix, the block, the model, the configs, the
calibration, the runners), so a new family is an `Arch` entry in `arch_of`, a new checkpoint of a known family a `MODELS`
entry (its calibrated configs and plan windows):

- `head_dim`: the attention head width (Llama: hidden / heads; Gemma-2: 256 explicitly). Heads sit in hidden / head_dim
  slots of the padded hidden (`model_dims`), the slots past the model's heads zero.
- `norm_offset`: RMSNorm's gamma is offset + weight (Gemma: 1).
- `pre_mlp_norm`: the MLP's input norm module (Llama: post_attention_layernorm; Gemma-2: pre_feedforward_layernorm).
- `post_norms`: norms on the attention's and the MLP's outputs before their residual add (Gemma-2), or None.
- `act`: the MLP gate's activation, `silu` or `gelu_tanh`.
- `attn_softcap` / `final_softcap`: cap tanh(x / cap) on the attention scores / the logits, or None.
- `embed_scale`: the embedding's scale (Gemma: sqrt(hidden)); `tied`: the LM head is the embedding matrix.
- `query_scale`: the scores' 1/sqrt(.) (Gemma-2: query_pre_attn_scalar); the runtime applies 1/sqrt(head_dim), the
  builder folds the ratio into Q.
- `sliding_window`: the local layers' attention window (Gemma-2); the port runs contexts within it (global attention).
"""
from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class Arch:
    family: str
    head_dim: int
    norm_offset: float = 0.0
    pre_mlp_norm: str = "post_attention_layernorm"
    post_norms: tuple[str, str] | None = None
    act: str = "silu"
    attn_softcap: float | None = None
    final_softcap: float | None = None
    embed_scale: float = 1.0
    tied: bool = False
    query_scale: float | None = None
    sliding_window: int | None = None
    hf_module: str = "transformers.models.llama.modeling_llama"

    @property
    def q_fold(self) -> float:
        """What the builder multiplies Q by: query_scale over the runtime's 1/sqrt(head_dim) (1 when they agree)."""
        return 1.0 if self.query_scale is None else self.query_scale * math.sqrt(self.head_dim)

    @property
    def norm_sites(self) -> tuple[str, ...]:
        """A layer's RMSNorm sites (configs.json `norm` keys layers.<b>.<site>): ln1 the attention's input, ln2 the
        MLP's, ln1p / ln2p their outputs' post norms."""
        return ("ln1", "ln2") + (("ln1p", "ln2p") if self.post_norms else ())

    def norm_module(self, site: str) -> str:
        """The HF module name of a layer's norm site."""
        names = {"ln1": "input_layernorm", "ln2": self.pre_mlp_norm}
        if self.post_norms:
            names.update(ln1p=self.post_norms[0], ln2p=self.post_norms[1])
        return names[site]

    @property
    def act_step(self) -> str:
        """The MLP activation's step name (plans key their recipes on step names)."""
        return {"silu": "silu", "gelu_tanh": "gelu"}[self.act]


def arch_of(config: dict) -> Arch:
    """The Arch of an HF config (its model_type)."""
    mt = config.get("model_type")
    d, H = int(config["hidden_size"]), int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim") or d // H)
    if mt == "llama":
        return Arch("llama", head_dim, tied=bool(config.get("tie_word_embeddings", False)))
    if mt == "gemma2":
        qs = config.get("query_pre_attn_scalar")
        return Arch("gemma2", head_dim, norm_offset=1.0, pre_mlp_norm="pre_feedforward_layernorm",
                    post_norms=("post_attention_layernorm", "post_feedforward_layernorm"), act="gelu_tanh",
                    attn_softcap=config.get("attn_logit_softcapping"), final_softcap=config.get("final_logit_softcapping"),
                    embed_scale=math.sqrt(d), tied=bool(config.get("tie_word_embeddings", True)),
                    query_scale=None if qs is None or int(qs) == head_dim else float(qs) ** -0.5,
                    sliding_window=config.get("sliding_window"), hf_module="transformers.models.gemma2.modeling_gemma2")
    raise ValueError(f"no decoder Arch for model_type {mt!r} (llama, gemma2): add one in examples/llama/arch.py")


@dataclasses.dataclass(frozen=True)
class Model:
    """A shipped checkpoint: its HF name, calibrated approximation configs and defaults.py plan keys (the decode's
    window plans, the prefill's plans by token count)."""
    name: str
    configs: str
    windows: str
    prefill: str


# alias -> Model; --model / $PERSEUS_DECODER take an alias or an HF name (an unlisted one runs with --configs given)
MODELS = {
    "llama31_8b": Model("meta-llama/Llama-3.1-8B", "configs/model/approximation/llama31_8b_n32/configs.json",
                        "llama_decode", "llama_prefill"),
    "gemma2_2b": Model("google/gemma-2-2b", "configs/model/approximation/gemma2_2b_n32_ctx512/configs.json", "gemma2_decode",
                       "gemma2_prefill"),
}


def resolve_model(name: str | None = None) -> str:
    """The HF name of an alias or name (default $PERSEUS_DECODER, else $LLAMA_MODEL, else Llama-3.1-8B)."""
    import os
    name = name or os.environ.get("PERSEUS_DECODER") or os.environ.get("LLAMA_MODEL") or "llama31_8b"
    return MODELS[name].name if name in MODELS else name


def model_entry(name: str) -> Model | None:
    """The MODELS entry of an HF name or alias, or None."""
    hf = resolve_model(name)
    return next((m for m in MODELS.values() if m.name == hf), None)


def model_dims(config: dict, N: int = 32768):
    """The slot geometry: the hidden and MLP widths padded to powers of two, hid / head_dim head slots of which the model's
    heads are the first (Llama: as many slots as heads; Gemma-2: 16 slots of 256, 8 live)."""
    from perseus.impl.layout import Dims
    d, E, H = int(config["hidden_size"]), int(config["intermediate_size"]), int(config["num_attention_heads"])
    dh = arch_of(config).head_dim
    hid, Ep = 1 << (d - 1).bit_length(), 1 << (E - 1).bit_length()
    if hid % dh or H * dh > hid:
        raise ValueError(f"{H} heads of {dh} do not fit a {hid}-wide hidden")
    if H * dh == d and hid // dh == H:
        return Dims(N, hid, d, H, H, Ep, E)
    return Dims(N, hid, d, hid // dh, H, Ep, E, head_real=dh)


def softcap(x, cap):
    """cap tanh(x / cap) (numpy), or x when cap is None."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    return x if not cap else cap * np.tanh(x / cap)
