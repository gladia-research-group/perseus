"""The Hugging Face side of the port: the checkpoint, the calibration / evaluation text
windows and the per-site statistics hooks (fp32 model on the visible GPU)."""
from __future__ import annotations

import os

import numpy as np

NAME = os.environ.get("LLAMA_MODEL", "meta-llama/Llama-3.1-8B")


def load_model(dtype="float32", device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(NAME, dtype=getattr(torch, dtype), attn_implementation="eager")
    return m.to(device).eval()


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(NAME)


def text_windows(n, T, skip=0):
    """`n` windows of `T` tokens (BOS first) from the first openwebtext documents that are long
    enough, after skipping `skip` such documents. Cached on disk: the stream can stall for hours."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "windows", f"owt_{T}_{skip}.npy")
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


def norm_sites(L):
    return [f"layers.{b}.{s}" for b in range(L) for s in ("ln1", "ln2")] + ["lnf"]


def norm_module(model, site):
    if site == "lnf":
        return model.model.norm
    _, b, s = site.split(".")
    layer = model.model.layers[int(b)]
    return layer.input_layernorm if s == "ln1" else layer.post_attention_layernorm


def collect_sites(model, toks, rows_per_layer=8192, seed=0, rot=None):
    """Per-site samples over every position but 0: '<norm>.ms' / '<norm>.absmax' per token,
    'layers.<b>.gate.absmax' per token, 'layers.<b>.scores' = (rows, row_q) of the causal score
    matrix (query rows >= 1, sampled). `rot`: the residual rotation (d, d) the port applies: a
    norm's absmax is taken on x rot (its mean square is rotation-invariant)."""
    import torch
    import transformers.models.llama.modeling_llama as M
    L = model.config.num_hidden_layers
    rec, handles = {}, []
    R = None if rot is None else torch.as_tensor(rot, dtype=torch.float64, device=model.device)

    def put(k, v):
        rec.setdefault(k, []).append(v.detach().double().cpu().numpy())

    for site in norm_sites(L):
        def f(m, inp, out, site=site):
            x = inp[0][0, 1:]
            put(site + ".ms", (x * x).mean(-1))
            xr = x if R is None else x.double() @ R
            put(site + ".absmax", xr.abs().amax(-1))
        handles.append(norm_module(model, site).register_forward_hook(f))
    for b, layer in enumerate(model.model.layers):
        def g(m, inp, out, b=b):
            put(f"layers.{b}.gate.absmax", out[0, 1:].abs().amax(-1))
        handles.append(layer.mlp.gate_proj.register_forward_hook(g))

    gen = torch.Generator().manual_seed(seed)
    orig = M.eager_attention_forward      # LlamaAttention looks the eager path up as a module global
    per_seq = max(1, rows_per_layer // len(toks))

    def eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        k = M.repeat_kv(key, module.num_key_value_groups)
        s = (torch.matmul(query, k.transpose(2, 3)) * scaling)[0]        # (H, T, T)
        H, T, _ = s.shape
        q = torch.arange(1, T).repeat(H)
        h = torch.arange(H).repeat_interleave(T - 1)
        pick = torch.randperm(len(q), generator=gen)[:per_seq]
        r = s[h[pick], q[pick]]
        rec.setdefault(f"layers.{module.layer_idx}.scores", []).append(
            (r.float().cpu().numpy(), q[pick].numpy()))
        return orig(module, query, key, value, attention_mask, scaling, dropout, **kw)

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
            out[k] = (np.concatenate([r for r, _ in v]), np.concatenate([q for _, q in v]))
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


def oracle_logits(toks, path=None):
    """Exact fp32 logits (T, vocab) of one teacher-forced window, cached at `path` (.npy)."""
    if path and os.path.exists(path):
        return np.load(path)
    import torch
    m = load_model()
    with torch.no_grad():
        lg = m(torch.as_tensor(np.asarray(toks)[None]).to(m.device)).logits[0].float().cpu().numpy()
    del m
    torch.cuda.empty_cache()
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, lg)
    return lg
