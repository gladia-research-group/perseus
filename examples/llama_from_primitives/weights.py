"""HF Llama checkpoint -> the folded, head-rearranged, GQA-expanded matrices of the cachemir
packing and their slot vectors (EncodedLinear), cached on disk as memory-mapped arrays.

Per block (the matrices are (d_in, d_out), y = x @ W):
  q, k, v   ln_1's gamma / inv_out_scale folded into the rows (RMSNorm has no beta); K and V
            expanded from the 8 KV heads to the 32 query heads (head h reads KV head
            h // (H / H_kv)), so the attention core is plain MHA; columns head-interleaved
            (rearrange_qkv_weights). cachemir_complex: one fused K + iV linear.
  out       rows head-interleaved (rearrange_wo_weights)
  gate, up  ln_2's gamma / inv_out_scale folded into the rows, columns padded to E; the gate
            also carries the SiLU's 1/xmax (the THOR composite's input scaling, GELU_FOLD)
  down      rows padded to E
  lm head   lnf's gamma / inv_out_scale folded into the rows, vocab tiles of N columns

Slot vectors are stored as float32 / complex64: the bf16 weights are exact there and the folded
products round at ~6e-8, far under the CKKS noise. A layer is ~1 GB; the 32 layers + the LM head
~40 GB under PERSEUS_LLAMA_CACHE (default $HF_HOME/perseus/llama_slots), keyed by the checkpoint,
the configs and the builder sources."""
from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from perseus.impl.layout import Dims, rearrange_qkv_weights, rearrange_wo_weights
from perseus.impl.linear import EncodedLinear



def snapshot_dir(name: str) -> str:
    """The local snapshot of an HF repo id (or a directory as is)."""
    if os.path.isdir(name):
        return name
    from huggingface_hub import snapshot_download
    return snapshot_download(name, allow_patterns=["*.json", "*.safetensors"])


class SafetensorsStore:
    """Lazy tensor access over a checkpoint's safetensors shards (float32 numpy)."""

    def __init__(self, path: str):
        self.dir = snapshot_dir(path)
        idx = os.path.join(self.dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                self._where = {k: os.path.join(self.dir, v) for k, v in json.load(f)["weight_map"].items()}
        else:
            from safetensors import safe_open
            self._where = {}
            for fn in glob.glob(os.path.join(self.dir, "*.safetensors")):
                with safe_open(fn, "pt") as f:
                    self._where.update({k: fn for k in f.keys()})
        with open(os.path.join(self.dir, "config.json")) as f:
            self.config = json.load(f)

    def has(self, name):
        return name in self._where

    def tensor(self, name) -> np.ndarray:
        from safetensors import safe_open
        with safe_open(self._where[name], "pt") as f:
            return f.get_tensor(name).float().numpy()

    def fingerprint(self) -> str:
        """The checkpoint's identity: shard names and sizes (the content is hashed by HF)."""
        h = hashlib.sha256()
        for fn in sorted(set(self._where.values())):
            h.update(os.path.basename(fn).encode()); h.update(str(os.path.getsize(fn)).encode())
        h.update(json.dumps(self.config, sort_keys=True).encode())
        return h.hexdigest()[:16]


def llama_dims(cfg: dict, N: int = 32768) -> Dims:
    d, E = int(cfg["hidden_size"]), int(cfg["intermediate_size"])
    H = int(cfg["num_attention_heads"])
    hid = 1 << (d - 1).bit_length()
    return Dims(N, hid, d, H, H, 1 << (E - 1).bit_length(), E)


def expand_kv(W, H: int, H_kv: int, d_head: int):
    """(d_in, H_kv d_head) -> (d_in, H d_head): query head h reads KV head h // (H / H_kv)."""
    g = H // H_kv
    cols = (np.arange(H)[:, None] // g * d_head + np.arange(d_head)[None, :]).ravel()
    return W[:, cols]


def residual_rotation(d: int, seed: int) -> np.ndarray:
    """The orthogonal Q of the rotated residual stream (QuaRot, arXiv 2404.00456): a Walsh-Hadamard
    matrix with random signs, Q = diag(s) H / sqrt(d). Exact (orthogonal), so the model's function
    is unchanged: RMSNorm without its gamma commutes with Q and gamma is folded into the next
    linears; per-dimension outliers of the residual spread over every dimension."""
    if d & (d - 1):
        raise ValueError(f"residual rotation needs a power-of-two width, got {d}")
    H = np.array([[1.0]])
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    s = np.random.default_rng(seed).choice([-1.0, 1.0], size=d)
    return (s[:, None] * H) / np.sqrt(d)


def rotation_of(cfgs, d: int):
    """Q for the configs' `residual_rotation` seed (model section), or None."""
    seed = cfgs.model.get("residual_rotation")
    return None if seed is None else residual_rotation(d, int(seed))


def pad(W, r, c):
    if W.shape == (r, c):
        return W
    out = np.zeros((r, c), dtype=W.dtype)
    out[:W.shape[0], :W.shape[1]] = W
    return out


@dataclasses.dataclass
class BlockWeights:
    q: EncodedLinear
    k: EncodedLinear | None
    v: EncodedLinear | None
    out: EncodedLinear
    gate: EncodedLinear
    up: EncodedLinear
    down: EncodedLinear
    tag: str = ""
    kv: EncodedLinear | None = None   # cachemir_complex: the fused K + iV linear (k, v unused)


@dataclasses.dataclass
class BlockMatrices:
    Q: np.ndarray; K: np.ndarray; V: np.ndarray; O: np.ndarray
    Gate: np.ndarray; Up: np.ndarray; Down: np.ndarray


def block_matrices(store: SafetensorsStore, cfgs, b: int, dims: Dims) -> BlockMatrices:
    c = store.config
    H, H_kv = int(c["num_attention_heads"]), int(c["num_key_value_heads"])
    dh = dims.dim // H
    ln1, ln2, _sm, act = cfgs.block(b)
    p = f"model.layers.{b}."
    t = lambda n: store.tensor(p + n).astype(np.float64)
    g1 = t("input_layernorm.weight") * ln1.descale
    g2 = t("post_attention_layernorm.weight") * ln2.descale
    R = rotation_of(cfgs, dims.dim)                      # the residual stream is x R (or x)
    rin = (lambda W: R.T @ W) if R is not None else (lambda W: W)
    rout = (lambda W: W @ R) if R is not None else (lambda W: W)
    Q = rin(t("self_attn.q_proj.weight").T * g1[:, None])
    K = rin(expand_kv(t("self_attn.k_proj.weight").T, H, H_kv, dh) * g1[:, None])
    V = rin(expand_kv(t("self_attn.v_proj.weight").T, H, H_kv, dh) * g1[:, None])
    Q, K, V = (rearrange_qkv_weights(pad(M, dims.hid, dims.hid), dims.H) for M in (Q, K, V))
    O = rearrange_wo_weights(pad(rout(t("self_attn.o_proj.weight").T), dims.hid, dims.hid), dims.H)
    Gate = pad(rin(t("mlp.gate_proj.weight").T * (g2 / act.xmax)[:, None]), dims.hid, dims.E)
    Up = pad(rin(t("mlp.up_proj.weight").T * g2[:, None]), dims.hid, dims.E)
    Down = pad(rout(t("mlp.down_proj.weight").T), dims.E, dims.hid)
    return BlockMatrices(Q, K, V, O, Gate, Up, Down)


def _sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def _folded_config(cfgs) -> str:
    """The config values the slot vectors carry: each norm's output scale (gamma / inv_out_scale)
    and each SiLU's xmax (the gate's 1/xmax). A refit of anything else keeps the cache."""
    norms = sorted((k, round(v.descale, 15)) for k, v in cfgs.norm.items())
    key = (norms, [round(a.xmax, 15) for a in cfgs.act])
    rot = cfgs.model.get("residual_rotation")
    return repr(key if rot is None else key + (("residual_rotation", rot),))


def _builder_sha():
    """The source of what produces a slot vector (an edit elsewhere keeps the ~40 GB cache)."""
    import inspect

    from perseus.impl import layout, linear
    fns = (block_matrices, expand_kv, pad, LlamaWeights._linear, LlamaWeights.block, LlamaWeights.lm_tiles,
           layout.cm_params, layout.interleave_idx, layout.weight_index, layout.encode_weight_matrix_fast,
           layout.rearrange_qkv_weights, layout.rearrange_wo_weights, linear.EncodedLinear.encode)
    return hashlib.sha256("".join(inspect.getsource(f) for f in fns).encode()).hexdigest()[:16]


class LlamaWeights:
    """The slot vectors of every block and of the LM head, built once per (checkpoint, configs,
    packing, builder sources) into the disk cache and memory-mapped from there."""

    def __init__(self, store: SafetensorsStore, cfgs, configs_path: str, dims: Dims,
                 packing: str = "cachemir_complex", cache_dir: str | None = None, pinned=None):
        """`pinned(rows, cols)`: a float32 array in pinned host memory (perseus._core.pinned_empty); given, every
        linear also carries its plaintexts as coefficient rows there (EncodedLinear.coeffs, the device-encoded
        inner sums), built once from the slot vectors into the same disk cache."""
        self.store, self.cfgs, self.dims, self.packing = store, cfgs, dims, packing
        self.pinned = pinned
        root = cache_dir or os.environ.get("PERSEUS_LLAMA_CACHE") or os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "perseus", "llama_slots")
        key = hashlib.sha256(f"{store.fingerprint()}|{_folded_config(cfgs)}|{_builder_sha()}|{dims}|{packing}"
                             .encode()).hexdigest()[:20]
        self.dir = Path(root) / key
        self.dir.mkdir(parents=True, exist_ok=True)

    # ── the disk cache ──
    def _cached(self, name, build):
        f = self.dir / f"{name}.npy"
        if not f.exists():
            arr = build()
            arr = arr.astype(np.complex64 if np.iscomplexobj(arr) else np.float32)
            tmp = self.dir / f".{name}.{os.getpid()}.tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, f)
        return np.load(f, mmap_mode="r")

    def _linear(self, name, W_build, d_in, d_out, outputpack=False):
        N = self.dims.N
        pts = self._cached(name, lambda: EncodedLinear.encode(W_build(), N, d_in, d_out,
                                                               outputpack=outputpack).pts)
        return EncodedLinear(d_in, d_out, pts, None, None, outputpack)

    def attach_coeffs(self, lin: EncodedLinear, name: str):
        """Give `lin` (the slot cache entry `name`) its coefficient rows and tags (see _coeffs) when pinned memory
        is configured. Separate from _linear, whose source keys the slot cache."""
        if self.pinned is not None and lin.coeffs is None:
            lin.coeffs, lin.tags = self._coeffs(name, lin.pts)
        return lin

    def _coeffs(self, name, pts):
        """The slot vectors' coefficient rows (layout.coeff_rows) in pinned memory and their plaintext tags, from
        the disk cache next to the slot vectors (built at the first use)."""
        from perseus import _core
        from perseus.impl.layout import coeff_rows
        f, ft = self.dir / f"{name}.coeff.npy", self.dir / f"{name}.tags.npy"
        if not (f.exists() and ft.exists()):
            for path, build in ((f, lambda: coeff_rows(pts)), (ft, lambda: _core.pt_tag_fields(np.ascontiguousarray(pts)))):
                if not path.exists():
                    tmp = self.dir / f".{path.stem}.{os.getpid()}.tmp.npy"
                    np.save(tmp, build())
                    os.replace(tmp, path)
        src = np.load(f, mmap_mode="r")
        pin = self.pinned(*src.shape)
        pin[:] = src
        return pin, np.load(ft)

    def block(self, b: int) -> BlockWeights:
        d = self.dims
        hid, E = d.hid, d.E
        m = {}
        def mats():
            if not m:
                m["x"] = block_matrices(self.store, self.cfgs, b, d)
            return m["x"]
        pre = f"b{b}."
        lin = lambda n, f, i, o, op=False: self._linear(pre + n, f, i, o, op)
        cplx = self.packing == "cachemir_complex"
        w = BlockWeights(
            q=lin("q", lambda: mats().Q, hid, hid),
            k=None if cplx else lin("k", lambda: mats().K, hid, hid),
            v=None if cplx else lin("v", lambda: mats().V, hid, hid),
            out=lin("out", lambda: mats().O, hid, hid),
            gate=lin("gate", lambda: mats().Gate, hid, E, cplx),
            up=lin("up", lambda: mats().Up, hid, E, cplx),
            down=lin("down", lambda: mats().Down, E, hid, cplx),
            tag=pre,
            kv=lin("kv", lambda: mats().K + 1j * mats().V, hid, hid) if cplx else None)
        return w

    def lm_tiles(self, paired: bool = True):
        """encode_gpt2_lm_head_tile: ceil(vocab / N) tiles of (hid, N); `paired`: tiles 2m and
        2m+1 as one complex tile (real and imaginary axis)."""
        d = self.dims
        N = d.N
        vocab = int(self.store.config["vocab_size"])
        K = (vocab + N - 1) // N
        m = {}
        def W():
            if not m:
                g = self.store.tensor("model.norm.weight").astype(np.float64) * self.cfgs.ln_f.descale
                m["W"] = self.store.tensor("lm_head.weight").T.astype(np.float64) * g[:, None]
                R = rotation_of(self.cfgs, d.dim)
                if R is not None:
                    m["W"] = R.T @ m["W"]
            return m["W"]
        def tile(k):
            if k >= K:
                return np.zeros((d.hid, N))
            c0 = k * N
            return pad(W()[:, c0:min(c0 + N, vocab)], d.hid, N)
        if paired:
            return [self._linear(f"lm.c{j}", lambda j=j: tile(2 * j) + 1j * tile(2 * j + 1), d.hid, N)
                    for j in range((K + 1) // 2)]
        return [self._linear(f"lm.{k}", lambda k=k: tile(k), d.hid, N) for k in range(K)]

    def linear_names(self, n_layers: int):
        """The slot cache entries of every linear the model uses."""
        per = ("q", "out", "gate", "up", "down") + (("kv",) if self.packing == "cachemir_complex" else ("k", "v"))
        K = (int(self.store.config["vocab_size"]) + self.dims.N - 1) // self.dims.N
        lm = [f"lm.c{j}" for j in range((K + 1) // 2)] if self.packing == "cachemir_complex" else [f"lm.{k}" for k in range(K)]
        return [f"b{b}.{n}" for b in range(n_layers) for n in per] + lm

    def build_coeff_caches(self, names, workers: int = 8):
        """The coefficient rows of these (already built) slot cache entries, in parallel processes."""
        import concurrent.futures as cf
        todo = [n for n in names if (self.dir / f"{n}.npy").exists() and not (self.dir / f"{n}.coeff.npy").exists()]
        if not todo:
            return 0
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_build_coeff_file, [str(self.dir)] * len(todo), todo))
        return len(todo)

    def bos_kv(self, n_layers: int):
        """prefix.bos_kv, cached next to the slot vectors (one exact fp64 pass per layer)."""
        from .prefix import bos_kv
        f = self.dir / "bos_kv.npy"
        have = np.load(f) if f.exists() else np.zeros((0, 2, 0))
        if len(have) < n_layers:
            have = np.asarray([np.stack(kv) for kv in bos_kv(self.store, n_layers=n_layers)])
            tmp = self.dir / f".bos_kv.{os.getpid()}.tmp.npy"
            np.save(tmp, have)
            os.replace(tmp, f)
        return [(kv[0], kv[1]) for kv in have[:n_layers]]

    def embedding(self, token: int) -> np.ndarray:
        if not hasattr(self, "_emb"):
            self._emb = self.store.tensor("model.embed_tokens.weight")
            self._R = rotation_of(self.cfgs, self.dims.dim)
        e = np.asarray(self._emb[int(token)], dtype=np.float64)
        return e if self._R is None else e @ self._R


def _build_coeff_file(d: str, name: str):
    """Worker: <name>.coeff.npy from <name>.npy (LlamaWeights._coeffs builds the same at first use)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from perseus.impl.layout import coeff_rows
    d = Path(d)
    pts = np.load(d / f"{name}.npy", mmap_mode="r")
    tmp = d / f".{name}.coeff.{os.getpid()}.tmp.npy"
    np.save(tmp, coeff_rows(pts))
    os.replace(tmp, d / f"{name}.coeff.npy")
