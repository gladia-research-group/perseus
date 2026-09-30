"""Weight packing: HF model → weights.bin.zip consumed by the CKKS runtime.

Format (parsed by include/io/weight_io.h): a ZIP with ``manifest.json``
(``{"model": tag, "tensors": [{name, path, shape, dtype}...]}``) plus one raw
C-order binary per parameter at ``tensors/NNNNN.bin``.

The runtime's naming grammar is the loader's contract (GPT-2 block naming:
``transformer.h.{b}.attn.c_attn`` / ``mlp.c_fc`` / ``ln_1`` …, head =
``transformer.wte``). One ADAPTER per architecture maps a hub checkpoint into
that grammar — tensors for the store, optional plaintext client-side tensors
(``client.npz``), and the calibration site→section renaming — so no per-model
bridge scripts exist. GPT-2 is the identity (HF names ARE the grammar; Conv1D
weights transpose to the nn.Linear ``(d_out, d_in)`` convention).

An HE-aware-TRAINED backbone (he-aware-training's ``model.pt``, e.g. the HEAT GPT-2 on the
hub) is exported with ``--checkpoint``: its weights are loaded into the stock hub model by
``load_trained_backbone`` (the trainer's own ``save_weights`` rules) and exported as usual.
"""

import argparse
import json
import logging
import os
import zipfile

import numpy as np

from perseus import artifacts
from perseus._log import configure_cli_logging

log = logging.getLogger(__name__)


def _gpt2_tensors(model):
    from transformers.pytorch_utils import Conv1D

    modules = dict(model.named_modules())
    out = {}
    for name, param in model.named_parameters():
        arr = param.detach().cpu().numpy()
        parent = modules.get(name.rpartition(".")[0])
        if isinstance(parent, Conv1D) and name.endswith(".weight"):
            arr = arr.T
        out[name] = arr
    return out


def _gpt2_client(model):
    """What the GPT-2 client keeps in the clear: the token and position embedding tables,
    so it can build the next input itself (perseus.nn.serve.EncGenerationClient)."""
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    return {"wte": sd["transformer.wte.weight"], "wpe": sd["transformer.wpe.weight"]}


# Trained checkpoints keep some Conv1D weights as nn.Linear (the trainer's attention surgery).
# A rectangular one is recognised by its reversed shape; a square one cannot be, so the adapter
# names it (he-aware-training: transpose_keys=(".attn.c_proj.weight",)).
SQUARE_LINEAR = {"gpt2": (".attn.c_proj.weight",)}


def _resolve_checkpoint(ref):
    """A local file, or a hub repo id (``org/name`` -> its ``model.pt``, ``org/name:file``)."""
    if os.path.exists(ref):
        return ref
    from huggingface_hub import hf_hub_download
    repo, _, fname = ref.partition(":")
    return hf_hub_download(repo, fname or "model.pt")


def load_trained_backbone(model, checkpoint):
    """Load an HE-aware-trained state dict into the stock hub ``model``, in place.

    The rules of he-aware-training's checkpoint loader: ``model_state_dict`` is unwrapped;
    a weight stored transposed is transposed back (by shape, or by name when square); the
    training-only extras (halting logits, approximation parameters and buffers) are dropped.
    Anything else that does not fit is an error, never a silent re-initialisation.
    """
    import torch

    path = _resolve_checkpoint(checkpoint)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    ref = model.state_dict()
    square = SQUARE_LINEAR.get(model.config.model_type, ())
    out, dropped = {}, 0
    for k, t in sd.items():
        if k not in ref:
            dropped += 1
            continue
        want = tuple(ref[k].shape)
        if t.dim() == 2 and want[0] == want[1] and k.endswith(square):
            t = t.t()
        elif tuple(t.shape) != want and t.dim() == 2 and tuple(t.shape) == want[::-1]:
            t = t.t()
        if tuple(t.shape) != want:
            raise ValueError(f"{k}: checkpoint shape {tuple(t.shape)} != model shape {want}")
        out[k] = t.contiguous()
    missing, _ = model.load_state_dict(out, strict=False)
    missing = [k for k in missing if not k.endswith((".attn.bias", ".attn.masked_bias"))]
    if missing:
        raise ValueError(f"{path}: no weights for {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    log.info(f"Loaded {len(out)} trained tensors from {path} ({dropped} training-only entries dropped)")
    return path


ADAPTERS = {
    "gpt2": {"tensors": _gpt2_tensors, "client": _gpt2_client, "section": lambda s: s},
}


def adapter_for(model_type):
    if model_type not in ADAPTERS:
        raise ValueError(f"no export adapter for model_type {model_type!r} "
                         f"(have {sorted(ADAPTERS)})")
    return ADAPTERS[model_type]


def remap_sections(calib, model_type):
    """Calibration dict → runtime section names (identity for gpt2)."""
    section = adapter_for(model_type)["section"]
    return {sec: ({section(k): v for k, v in entries.items()}
                  if isinstance(entries, dict) else entries)
            for sec, entries in calib.items()}


def export_weights(model, out_dir: str, tag: str = "classic") -> str:
    """Serialize ``model`` to ``<out_dir>/<tag>/weights.bin.zip`` (+ client.npz)."""
    model.eval()
    ad = adapter_for(model.config.model_type)
    tensors = ad["tensors"](model)
    out_dir = os.path.join(out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    zip_path = os.path.join(out_dir, "weights.bin.zip")
    manifest = {"model": tag, "tensors": [],
                **artifacts.weights_meta(model_id=getattr(model.config, "_name_or_path", None),
                                         model_type=model.config.model_type)}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for index, (name, arr) in enumerate(tensors.items()):
            arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
            tensor_path = f"tensors/{index:05d}.bin"
            zf.writestr(tensor_path, arr.tobytes(order="C"))
            manifest["tensors"].append(
                {"name": name, "path": tensor_path, "shape": list(arr.shape),
                 "dtype": arr.dtype.str})
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    log.info(f"Zipped -> {zip_path} ({len(tensors)} tensors)")

    if ad["client"] is not None:
        np.savez(os.path.join(out_dir, "client.npz"), **ad["client"](model))
        log.info(f"Wrote -> {out_dir}/client.npz")
    return zip_path


def main():
    configure_cli_logging()
    p = argparse.ArgumentParser(description="Export an HF model to weights.bin.zip")
    p.add_argument("--model", default="openai-community/gpt2",
                   help="HF model name or local path")
    p.add_argument("--out", default=None,
                   help="output dir (default: <HF_HOME>/perseus/models/<model>)")
    p.add_argument("--tag", default="classic", help="subdir/tag for this export")
    p.add_argument("--vocab-size", type=int, default=None,
                   help="resize token embeddings (e.g. 50257 for GPT-2)")
    p.add_argument("--checkpoint", default=None,
                   help="HE-aware-trained weights for --model: a model.pt, or a hub repo id "
                        "(org/name[:file])")
    args = p.parse_args()

    from perseus.hub import cache_dir, load_model

    model = load_model(args.model, vocab_size=args.vocab_size)
    if args.checkpoint:
        load_trained_backbone(model, args.checkpoint)
    out = args.out or cache_dir("models", args.model)
    export_weights(model, out, args.tag)


if __name__ == "__main__":
    main()
