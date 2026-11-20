"""The decoder port's architecture switch (perseus/examples/llama/arch.py) on tiny checkpoints, on the CPU:

- the slot cache's builder is pinned: the matrices and slot vectors of a tiny Llama and a tiny Gemma-2 hash to golden
  values, so an edit that changes what a cached slot vector holds fails here until weights.BUILDER_VERSION is bumped
  with the goldens (the Llama golden is the pre-Arch builder's output: the Llama caches on disk stay valid);
- the plaintext BOS prefix (prefix.bos_kv) matches the HF model's K / V at position 0 for both families;
- the layouts: Llama-3-8B's Dims as before the Arch, Gemma-2-2B's 16 head slots of 256;
- Q on K's padding heads: the packed fused linear is K + iV with Q's heads moved onto K's zero padding heads.
"""
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perseus.examples.llama import arch as A
from perseus.examples.llama import weights as W
from perseus.impl.layout import Dims

LLAMA = {"model_type": "llama", "hidden_size": 32, "intermediate_size": 48, "num_attention_heads": 4,
         "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 64, "rms_norm_eps": 1e-5, "bos_token_id": 1,
         "tie_word_embeddings": False, "rope_theta": 10000.0, "max_position_embeddings": 64}
GEMMA2 = {"model_type": "gemma2", "hidden_size": 24, "intermediate_size": 40, "num_attention_heads": 2,
          "num_key_value_heads": 1, "head_dim": 8, "num_hidden_layers": 2, "vocab_size": 64, "rms_norm_eps": 1e-6,
          "bos_token_id": 2, "tie_word_embeddings": True, "rope_theta": 10000.0, "max_position_embeddings": 64,
          "query_pre_attn_scalar": 16, "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0,
          "sliding_window": 32, "hidden_activation": "gelu_pytorch_tanh"}

# sha256[:16] of the tiny checkpoints' block matrices and slot vectors (see the module docstring before editing).
# gemma2 with Q on K's padding heads (weights.q_pad_heads, the default: 2 heads in 4 slots) caches one entry "kvq" in
# place of "q" and "kv"; with LLAMA_Q_PAD=0 it keeps those two, unchanged
GOLDEN = {"llama": ("c9c39daf908946b4", "b270163bbb1207a8"), "gemma2": ("21a5993edd1d4a80", "a6df0b85b96ce0b7"),
          "gemma2_qpad": ("21a5993edd1d4a80", "f8037434715474ee")}


def _exact(shape, k):
    """Deterministic weights exact in float32 (multiples of 1/1024 in [-1, 1)): platform-independent hashes."""
    n = int(np.prod(shape))
    return (((np.arange(n, dtype=np.int64) * 7919 + k * 104729) % 2048 - 1024) / 1024.0).reshape(shape).astype(np.float32)


def _tensors(c):
    d, E, L, V = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], c["vocab_size"]
    H, Hk = c["num_attention_heads"], c["num_key_value_heads"]
    dh = c.get("head_dim") or d // H
    a = A.arch_of(c)
    t = {"model.embed_tokens.weight": _exact((V, d), 1), "model.norm.weight": _exact((d,), 2)}
    if not a.tied:
        t["lm_head.weight"] = _exact((V, d), 3)
    norms = ("input_layernorm", a.pre_mlp_norm) + (a.post_norms or ())
    for b in range(L):
        p, k = f"model.layers.{b}.", 10 * (b + 1)
        t.update({p + "self_attn.q_proj.weight": _exact((H * dh, d), k), p + "self_attn.k_proj.weight": _exact((Hk * dh, d), k + 1),
                  p + "self_attn.v_proj.weight": _exact((Hk * dh, d), k + 2), p + "self_attn.o_proj.weight": _exact((d, H * dh), k + 3),
                  p + "mlp.gate_proj.weight": _exact((E, d), k + 4), p + "mlp.up_proj.weight": _exact((E, d), k + 5),
                  p + "mlp.down_proj.weight": _exact((d, E), k + 6)})
        for i, m in enumerate(norms):
            # norm weights near their identity (Gemma's offset 1 + w: w small)
            t[p + m + ".weight"] = (1.0 - a.norm_offset + _exact((d,), k + 7 + i) / 8).astype(np.float32)
    return t


def _checkpoint(tmp_path, c):
    from safetensors.numpy import save_file
    save_file(_tensors(c), str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(c))
    return W.SafetensorsStore(str(tmp_path))


def _cfgs(c):
    a = A.arch_of(c)
    n = lambda s: types.SimpleNamespace(descale=s)
    L = c["num_hidden_layers"]
    norm = {f"layers.{b}.{s}": n(0.75 + 0.125 * i) for b in range(L) for i, s in enumerate(a.norm_sites)}
    norm["lnf"] = n(0.625)
    act = [types.SimpleNamespace(xmax=8.0 + b) for b in range(L)]
    post = (lambda b: (norm[f"layers.{b}.ln1p"], norm[f"layers.{b}.ln2p"])) if a.post_norms else (lambda b: None)
    return types.SimpleNamespace(model={}, norm=norm, act=act, ln_f=norm["lnf"], post=post,
                                 block=lambda b: (norm[f"layers.{b}.ln1"], norm[f"layers.{b}.ln2"], None, act[b]))


def _sha(arrays):
    h = hashlib.sha256()
    for x in arrays:
        x = np.ascontiguousarray(x)
        h.update(str((x.dtype, x.shape)).encode())
        h.update(x.tobytes())
    return h.hexdigest()[:16]


def builder_hashes(tmp_path, c):
    """(matrices, slot vectors) hashes of the tiny checkpoint `c`: block_matrices of every layer, then the cached
    slot vectors of every block and the LM head (cachemir_complex)."""
    store = _checkpoint(tmp_path, c)
    cfgs = _cfgs(c)
    dims = A.model_dims(c, N=1024)
    mats = []
    for b in range(c["num_hidden_layers"]):
        m = W.block_matrices(store, cfgs, b, dims)
        mats += [x for x in (m.Q, m.K, m.V, m.O, m.Gate, m.Up, m.Down, getattr(m, "post1", None),
                             getattr(m, "post2", None)) if x is not None]
    lw = W.LlamaWeights(store, cfgs, "", dims, "cachemir_complex", cache_dir=str(tmp_path / "slots"))
    for b in range(c["num_hidden_layers"]):
        lw.block(b)
    lw.lm_tiles()
    slots = [np.load(f) for f in sorted(lw.dir.glob("*.npy")) if f.name != "bos_kv.npy"]
    return _sha(mats), _sha(slots)


@pytest.mark.parametrize("family,c,qpad", [("llama", LLAMA, "1"), ("gemma2", GEMMA2, "0"), ("gemma2_qpad", GEMMA2, "1")])
def test_builder_golden(tmp_path, monkeypatch, family, c, qpad):
    monkeypatch.setenv("LLAMA_Q_PAD", qpad)
    got = builder_hashes(tmp_path, c)
    assert got == GOLDEN[family], (f"{family}: the slot cache's builder output changed ({got}); bump "
                                   f"weights.BUILDER_VERSION and update GOLDEN")


def test_builder_key_keeps_llama_caches():
    # version 1 is the pre-version source hash: the Llama-3-8B caches on disk keep their directory
    assert W.BUILDER_VERSION == 1 and W.builder_key() == "4131b6a248f56d4c"


def test_dims():
    llama8b = {"model_type": "llama", "hidden_size": 4096, "intermediate_size": 14336, "num_attention_heads": 32,
               "num_key_value_heads": 8}
    d = A.model_dims(llama8b)
    assert d == Dims(32768, 4096, 4096, 32, 32, 16384, 14336) and repr(d) == repr(Dims(32768, 4096, 4096, 32, 32, 16384, 14336))
    assert (d.tH, d.d_head, d.d_head_real) == (256, 128, 128)
    gemma2b = {"model_type": "gemma2", "hidden_size": 2304, "intermediate_size": 9216, "num_attention_heads": 8,
               "num_key_value_heads": 4, "head_dim": 256, "query_pre_attn_scalar": 256}
    g = A.model_dims(gemma2b)
    assert (g.hid, g.dim, g.H, g.H_real, g.E, g.E_real) == (4096, 2304, 16, 8, 16384, 9216)
    assert (g.tH, g.d_head, g.d_head_real) == (128, 256, 256)
    assert A.arch_of(gemma2b).q_fold == 1.0


def _hf_bos_kv(tmp_path, c):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    store = _checkpoint(tmp_path, c)
    cfg = transformers.AutoConfig.for_model(**{k: v for k, v in c.items() if k != "model_type"},
                                            model_type=c["model_type"])
    cfg._attn_implementation = "eager"
    m = transformers.AutoModelForCausalLM.from_config(cfg).float().eval()
    m.load_state_dict({k: torch.from_numpy(v) for k, v in _tensors(c).items()}, strict=False)
    if A.arch_of(c).tied:
        m.tie_weights()
    out = {}
    hs = []
    for b, layer in enumerate(m.model.layers):
        for kind in ("k", "v"):
            hs.append(getattr(layer.self_attn, f"{kind}_proj").register_forward_hook(
                lambda _m, _i, o, b=b, kind=kind: out.__setitem__((b, kind), o[0, 0].double().numpy())))
    with torch.no_grad():
        m(torch.tensor([[c["bos_token_id"]]]))
    for h in hs:
        h.remove()
    return store, [(out[(b, "k")], out[(b, "v")]) for b in range(c["num_hidden_layers"])]


@pytest.mark.parametrize("c", [LLAMA, GEMMA2], ids=["llama", "gemma2"])
def test_bos_prefix_matches_hf(tmp_path, c):
    from perseus.examples.llama.prefix import bos_kv
    store, ref = _hf_bos_kv(tmp_path, c)
    for (k, v), (rk, rv) in zip(bos_kv(store), ref):
        np.testing.assert_allclose(k, rk, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(v, rv, rtol=1e-4, atol=1e-5)


def test_registry():
    assert A.resolve_model("gemma2_2b") == "google/gemma-2-2b"
    assert A.model_entry("google/gemma-2-2b").windows == "gemma2_decode"
    assert A.resolve_model("org/unlisted") == "org/unlisted" and A.model_entry("org/unlisted") is None
    with pytest.raises(ValueError):
        A.arch_of({"model_type": "mistral", "hidden_size": 8, "num_attention_heads": 2})


def test_q_pad_heads():
    from perseus.impl.layout import Dims
    llama = A.model_dims({"model_type": "llama", "hidden_size": 4096, "intermediate_size": 14336,
                          "num_attention_heads": 32, "num_key_value_heads": 8})
    gemma = A.model_dims({"model_type": "gemma2", "hidden_size": 2304, "intermediate_size": 9216,
                          "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256})
    assert W.q_pad_heads(llama) == 0 and W.q_pad_heads(gemma) == 8
    d = Dims(64, 16, 12, 4, 2, 64, 48, head_real=4)
    rng = np.random.default_rng(0)
    Q = np.zeros((16, 16))
    real = (np.arange(16) % 4) < 2
    Q[:, real] = rng.standard_normal((16, int(real.sum())))
    P = W.pad_q(Q, d, W.q_pad_heads(d))
    assert not P[:, real].any()                                   # K's real heads untouched
    np.testing.assert_array_equal(P[:, np.flatnonzero(real) + 2], Q[:, real])   # two head slots up
