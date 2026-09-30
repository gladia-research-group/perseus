"""Model acquisition: HuggingFace download / instantiation.

Mirrors the model-factory convention of the calibration configs
(``perseus/configs/model/*.yaml``): eager attention (hooks need the exact
non-fused path), no remote code, fp32 for calibration/export fidelity.

All derived artifacts (token pools, packed weights) live in `cache_dir` under
the HF cache home — relocate them the HF way (the ``HF_HOME`` env).
"""

import os


def cache_dir(*parts: str) -> str:
    """<HF_HOME>/perseus/<parts...>, created on first use."""
    from huggingface_hub.constants import HF_HOME

    path = os.path.join(HF_HOME, "perseus", *parts)
    os.makedirs(path, exist_ok=True)
    return path


def load_model(name: str = "openai-community/gpt2", vocab_size: int | None = None,
               device: str = "cpu", dtype=None, eval_mode: bool = True,
               encoder_only: bool = False):
    """Download/instantiate a hub model (causal LM, image classifier, or text encoder).
    You can extend this to other model types by adding to the `auto` dict below.
    """
    import torch
    import transformers
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForImageClassification,
        AutoModelForSequenceClassification,
    )

    config = AutoConfig.from_pretrained(name, trust_remote_code=False)
    want = torch.float32 if dtype is None else dtype      # fp32 for calibration/export fidelity
    # transformers 5 renamed torch_dtype -> dtype (the old name only warns, for now)
    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    kw = {"device_map": None, dtype_kw: want, "trust_remote_code": False,
          "attn_implementation": "eager"}
    if encoder_only and config.model_type == "bert":
        model = AutoModel.from_pretrained(name, add_pooling_layer=False, **kw)
    else:
        auto = {"vit": AutoModelForImageClassification,
                "bert": AutoModelForSequenceClassification}.get(
                    config.model_type, AutoModelForCausalLM)
        model = auto.from_pretrained(name, **kw)
    if vocab_size is not None and model.config.vocab_size != vocab_size:
        model.resize_token_embeddings(vocab_size)
    model.to(device=device, dtype=want)
    if eval_mode:
        model.eval()
    return model
