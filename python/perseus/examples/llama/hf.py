"""The Hugging Face side of the decoder port: the checkpoint, the calibration / evaluation text
windows and the per-site statistics hooks (fp32 model on the visible GPU).

`NAME` is the checkpoint every runner uses: $PERSEUS_DECODER (an arch.MODELS alias or an HF name), else $LLAMA_MODEL,
else Llama-3.1-8B; `select(name)` sets it (the runners' --model). Its architecture (arch.Arch) decides the norm sites,
the HF attention module the score hook patches and the soft-cap."""
from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import json
import os

import numpy as np

from .arch import arch_of, resolve_model

NAME = resolve_model()

# text_windows caches per tokenizer (the sha of its tokenizer.json); Llama-3's windows kept their pre-key file names
_LEGACY_TOKENIZER = "76e48799b099"


def select(name: str | None) -> str:
    """Set NAME to an alias or HF name (None keeps it); returns the HF name."""
    global NAME
    if name:
        NAME = resolve_model(name)
    return NAME


@functools.cache
def _config(name: str) -> dict:
    from .weights import snapshot_dir
    with open(os.path.join(snapshot_dir(name), "config.json")) as f:
        return json.load(f)


def config() -> dict:
    """The checkpoint's config.json."""
    return _config(NAME)


def arch():
    return arch_of(config())


def load_model(dtype="float32", device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(NAME, dtype=getattr(torch, dtype), attn_implementation="eager")
    return m.to(device).eval()


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(NAME)


def _tokenizer_key() -> str:
    from .weights import snapshot_dir
    f = os.path.join(snapshot_dir(NAME), "tokenizer.json")
    if not os.path.exists(f):
        return NAME.replace("/", "_")
    with open(f, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


def text_windows(n, T, skip=0):
    """`n` windows of `T` tokens (BOS first) from the first openwebtext documents that are long
    enough, after skipping `skip` such documents. Cached on disk per tokenizer: the stream can stall for hours."""
    key = _tokenizer_key()
    stem = f"owt_{T}_{skip}" if key == _LEGACY_TOKENIZER else f"owt_{key}_{T}_{skip}"
    path = os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "windows", stem + ".npy")
    if os.path.exists(path):
        cached = np.load(path)
        if len(cached) >= n:
            return cached[:n]
    from datasets import load_dataset
    tok = tokenizer()
    out = []
    for ex in load_dataset("Skylion007/openwebtext", split="train", streaming=True):
        ids = tok(ex["text"])["input_ids"]
        if len(ids) < T:
            continue
        if skip:
            skip -= 1
            continue
        out.append(ids[:T])
        if len(out) >= n:
            break
    out = np.asarray(out, dtype=np.int64)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path + ".tmp.npy", out)
    os.replace(path + ".tmp.npy", path)
    return out


def norm_sites(L, ar=None):
    """configs.json's RMSNorm sites: layers.<b>.<site> per the Arch's sites, then lnf."""
    ar = ar or arch()
    return [f"layers.{b}.{s}" for b in range(L) for s in ar.norm_sites] + ["lnf"]


def norm_module(model, site, ar=None):
    if site == "lnf":
        return model.model.norm
    _, b, s = site.split(".")
    return getattr(model.model.layers[int(b)], (ar or arch()).norm_module(s))


def collect_sites(model, toks, rows_per_layer=8192, seed=0, rot=None, first=1):
    """Per-site samples over the positions from `first` (1: every position but the BOS; an encoder after a public
    prefix: the prompt's, toks then ragged windows): '<norm>.ms' / '<norm>.absmax' per token,
    'layers.<b>.gate.absmax' per token, 'layers.<b>.scores' = (rows, row_q) of the causal score
    matrix the softmax takes (after the Arch's soft-cap; query rows >= 1, sampled) and 'layers.<b>.logit.absmax' the
    largest |score| before the soft-cap per query (its series' range). `rot`: the residual rotation (d, d) the port
    applies: a norm's absmax is taken on x rot (its mean square is rotation-invariant)."""
    import torch
    ar = arch_of(model.config.to_dict())
    M = importlib.import_module(ar.hf_module)
    L = model.config.num_hidden_layers
    rec, handles = {}, []
    R = None if rot is None else torch.as_tensor(rot, dtype=torch.float64, device=model.device)

    def put(k, v):
        rec.setdefault(k, []).append(v.detach().double().cpu().numpy())

    for site in norm_sites(L, ar):
        def f(m, inp, out, site=site):
            x = inp[0][0, first:]
            put(site + ".ms", (x * x).mean(-1))
            xr = x if R is None else x.double() @ R
            put(site + ".absmax", xr.abs().amax(-1))
        handles.append(norm_module(model, site, ar).register_forward_hook(f))
    for b, layer in enumerate(model.model.layers if hasattr(model, "model") else model.layers):
        def g(m, inp, out, b=b):
            put(f"layers.{b}.gate.absmax", out[0, first:].abs().amax(-1))
        handles.append(layer.mlp.gate_proj.register_forward_hook(g))

    gen = torch.Generator().manual_seed(seed)
    orig = M.eager_attention_forward      # the attention looks the eager path up as a module global
    sig = inspect.signature(orig)
    per_seq = max(1, rows_per_layer // len(toks))

    def eager(*args, **kw):
        ba = sig.bind(*args, **kw)
        ba.apply_defaults()
        module, query, key = ba.arguments["module"], ba.arguments["query"], ba.arguments["key"]
        scaling = ba.arguments.get("scaling")
        scaling = module.scaling if scaling is None else scaling
        cap = ba.arguments.get("softcap")
        k = M.repeat_kv(key, module.num_key_value_groups)
        raw = (torch.matmul(query, k.transpose(2, 3)) * scaling)[0]        # (H, T, T)
        H, T, _ = raw.shape
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=raw.device))[first:]
        put(f"layers.{module.layer_idx}.logit.absmax", raw[:, first:].abs().masked_fill(~causal, 0).amax((0, 2)))
        s = raw if cap is None else cap * torch.tanh(raw / cap)
        q = torch.arange(first, T).repeat(H)
        h = torch.arange(H).repeat_interleave(T - first)
        pick = torch.randperm(len(q), generator=gen)[:per_seq]
        r = s[h[pick], q[pick]]
        rec.setdefault(f"layers.{module.layer_idx}.scores", []).append(
            (r.float().cpu().numpy(), q[pick].numpy()))
        return orig(*args, **kw)

    M.eager_attention_forward = eager
    try:
        with torch.no_grad():
            for s in toks:
                model(torch.as_tensor(s[None]).to(model.device))
    finally:
        M.eager_attention_forward = orig
        for hd in handles:
            hd.remove()
    out = {}
    for k, v in rec.items():
        if k.endswith(".scores"):
            w = max(r.shape[1] for r, _ in v)        # ragged windows: rows padded past their keys (row_q bounds them)
            out[k] = (np.concatenate([np.pad(r, ((0, 0), (0, w - r.shape[1]))) for r, _ in v]),
                      np.concatenate([q for _, q in v]))
        else:
            out[k] = np.concatenate(v)
    return out


def prompt_kv(toks, n: int, path=None, device="cuda"):
    """The plain prefix: exact fp32 pre-RoPE K / V of the first `n` positions of a teacher-forced window,
    (k, v) each (layers, n, H_kv * d_head) float64 (the k_proj / v_proj outputs), cached at `path` (.npz)."""
    if path and os.path.exists(path):
        z = np.load(path)
        return z["k"], z["v"]
    import torch
    m = load_model(device=device)
    out = {}

    def hook(b, kind):
        def f(_mod, _inp, o):
            out[(b, kind)] = o[0].double().cpu().numpy()
        return f
    hs = []
    for b, layer in enumerate(m.model.layers):
        hs.append(layer.self_attn.k_proj.register_forward_hook(hook(b, "k")))
        hs.append(layer.self_attn.v_proj.register_forward_hook(hook(b, "v")))
    with torch.no_grad():
        m(torch.as_tensor(np.asarray(toks[:n])[None]).to(m.device))
    for h in hs:
        h.remove()
    L = len(m.model.layers)
    k = np.stack([out[(b, "k")] for b in range(L)])
    v = np.stack([out[(b, "v")] for b in range(L)])
    del m
    torch.cuda.empty_cache()
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, k=k, v=v)
    return k, v


def oracle_logits(toks, path=None, device="cuda"):
    """Exact fp32 logits (T, vocab) of one teacher-forced window (the Arch's final soft-cap included), cached at
    `path` (.npy)."""
    if path and os.path.exists(path):
        return np.load(path)
    import torch
    m = load_model(device=device)
    with torch.no_grad():
        lg = m(torch.as_tensor(np.asarray(toks)[None]).to(m.device)).logits[0].float().cpu().numpy()
    del m
    torch.cuda.empty_cache()
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, lg)
    return lg
