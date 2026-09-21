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
               device: str = "cpu", dtype=None, eval_mode: bool = True):
    """Download/instantiate a hub causal LM (GPT-2 family) in fp32."""
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.from_pretrained(name, trust_remote_code=False)   # validates the checkpoint name early
    want = torch.float32 if dtype is None else dtype      # fp32 for calibration/export fidelity
    # transformers 5 renamed torch_dtype -> dtype (the old name only warns, for now)
    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    kw = {"device_map": None, dtype_kw: want, "trust_remote_code": False,
          "attn_implementation": "eager"}
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    if vocab_size is not None and model.config.vocab_size != vocab_size:
        model.resize_token_embeddings(vocab_size)
    model.to(device=device, dtype=want)
    if eval_mode:
        model.eval()
    return model
