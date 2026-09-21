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

Loading an HE-aware-TRAINED backbone checkpoint into the export is not vendored
(that path lives in he-aware-training's save_weights.py).
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
    args = p.parse_args()

    from perseus.hub import cache_dir, load_model

    model = load_model(args.model, vocab_size=args.vocab_size)
    out = args.out or cache_dir("models", args.model)
    export_weights(model, out, args.tag)


if __name__ == "__main__":
    main()
