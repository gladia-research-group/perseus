"""Calibration data: token pools built straight from HuggingFace datasets.

No pre-tokenized `.bin` corpora: the first run streams the configured HF
dataset, tokenizes it with the MODEL'S OWN tokenizer (architecture-agnostic),
and materializes exactly `pool_tokens` tokens into a small cached `.npy` in
the HF cache home (`hub.cache_dir("pools")`, relocatable via `HF_HOME`).
Later runs — including offline compute nodes — memory-map that pool. Batches
are random `block_size` windows over it.

On clusters whose compute nodes have no internet, run any calibration command
once on a login node first (or call `build_token_pool` directly) to populate
the pool cache.
"""

import os

import numpy as np
import torch

from perseus.hub import cache_dir


def _pool_path(dataset_name, model_name, n_tokens):
    model_slug = model_name.split("/")[-1]
    return os.path.join(cache_dir("pools"), f"{dataset_name}_{model_slug}_{n_tokens}.npy")


def build_token_pool(model_name, dataset_path, split, n_tokens, text_field="text"):
    """Stream the HF dataset and tokenize until `n_tokens` tokens are collected.
    Documents are EOS-separated; dtype follows the tokenizer's vocab size."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = np.uint16 if len(tok) < (1 << 16) else np.uint32
    sep_id = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
    eos = [sep_id] if sep_id is not None else []

    print(f"[data] streaming {dataset_path}[{split}] -> {n_tokens} tokens "
          f"({tok.name_or_path} tokenizer, {np.dtype(dtype).name})")
    stream = load_dataset(dataset_path, split=split, streaming=True)
    buf = np.empty(n_tokens, dtype=dtype)
    fill = 0
    for doc in stream:
        ids = tok(doc[text_field], add_special_tokens=False)["input_ids"] + eos
        take = min(len(ids), n_tokens - fill)
        buf[fill:fill + take] = ids[:take]
        fill += take
        if fill >= n_tokens:
            break
    if fill < n_tokens:
        raise RuntimeError(f"dataset exhausted at {fill}/{n_tokens} tokens")
    return buf


def load_token_pool(model_name, dataset):
    """Return the cached token pool for (dataset, model tokenizer), building it
    on first use (needs network — do that once on a login node)."""
    path = _pool_path(dataset.name, model_name, dataset.pool_tokens)
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")

    pool = build_token_pool(model_name, dataset.path, dataset.split, dataset.pool_tokens,
                           text_field=getattr(dataset, "text_field", "text"))
    np.save(path, pool)
    print(f"[data] cached token pool -> {path}")
    return pool


def build_image_pool(model_name, dataset_path, split, n_images, image_field="image"):
    """Stream the HF image dataset through the MODEL'S OWN processor until
    `n_images` preprocessed tensors are collected — the image twin of
    `build_token_pool` (any RGB image dataset works)."""
    from datasets import load_dataset
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(model_name)
    print(f"[data] streaming {dataset_path}[{split}] -> {n_images} images "
          f"({proc.__class__.__name__})")
    stream = load_dataset(dataset_path, split=split, streaming=True)
    out = []
    for ex in stream:
        img = ex[image_field]
        if getattr(img, "mode", "RGB") != "RGB":
            img = img.convert("RGB")
        out.append(proc(img, return_tensors="np")["pixel_values"][0])
        if len(out) >= n_images:
            break
    if len(out) < n_images:
        raise RuntimeError(f"dataset exhausted at {len(out)}/{n_images} images")
    return np.stack(out).astype(np.float32)


def load_image_pool(model_name, dataset):
    path = _pool_path(dataset.name, model_name, dataset.pool_images)
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    pool = build_image_pool(model_name, dataset.path, dataset.split,
                            dataset.pool_images, dataset.image_field)
    np.save(path, pool)
    print(f"[data] cached image pool -> {path}")
    return pool


def compose_image_get_batch(model_name, dataset, device, micro_batch, resolution=None):
    """resolution: run the model below its native input size (fewer patches);
    pooled tensors are stored at native size and interpolated per batch — the
    caller must arm the model's position-embedding interpolation."""
    pool = load_image_pool(model_name, dataset)

    def get_batch():
        ix = torch.randint(len(pool), (micro_batch,))
        x = torch.from_numpy(np.stack([pool[i] for i in ix.tolist()]))
        if resolution:
            x = torch.nn.functional.interpolate(
                x, size=(resolution, resolution), mode="bilinear", antialias=True)
        if "cuda" in device:
            return x.pin_memory().to(device, non_blocking=True)
        return x.to(device)

    return get_batch


def compose_get_batch(model_name, dataset, device, block_size, micro_batch):
    pool = load_token_pool(model_name, dataset)
    n = len(pool)
    if n <= block_size:
        raise RuntimeError(f"token pool ({n}) smaller than block_size ({block_size})")

    def get_batch():
        ix = torch.randint(n - block_size, (micro_batch,))
        x = torch.stack(
            [torch.from_numpy(pool[i : i + block_size].astype(np.int64)) for i in ix]
        )
        if "cuda" in device:
            return x.pin_memory().to(device, non_blocking=True)
        return x.to(device)

    return get_batch
