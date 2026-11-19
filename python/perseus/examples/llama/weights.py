"""A decoder checkpoint (arch.Arch: Llama, Gemma-2) -> the folded, head-rearranged, GQA-expanded matrices of the
cachemir packing and their slot vectors (EncodedLinear), cached on disk as memory-mapped arrays.

Per block: q/k/v, gate/up and the LM head carry their norm's gamma / inv_out_scale folded into the rows (gamma = the
Arch's norm_offset + weight); K and V expanded from the KV heads to the query heads (plain MHA core) in the head slots
of arch.model_dims; columns head-interleaved; gate also carries the activation's 1/xmax. The post norms' gammas (Gemma-2)
stay vectors: no linear follows them. Slot vectors are float32 / complex64 (bf16 weights are exact there); ~40 GB for
Llama-3-8B under PERSEUS_LLAMA_CACHE (default $HF_HOME/perseus/llama_slots), keyed by checkpoint, configs and builder.
"""
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

from .arch import arch_of, model_dims

# The slot cache's builder key. Bump BUILDER_VERSION whenever what a slot vector holds changes (the matrices, their folds,
# the encoding): python/tests/test_decoder_arch.py pins the builder's output on tiny checkpoints, so such an edit fails it
# until the golden hashes and this version move together. Version 1 is the source hash the caches built before the
# version existed carry (main 0d5c85b2): they stay valid.
BUILDER_VERSION = 1
_BUILDER_KEYS = {1: "4131b6a248f56d4c"}


def builder_key() -> str:
    return _BUILDER_KEYS.get(BUILDER_VERSION, f"v{BUILDER_VERSION}")



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

    def _key(self, name):
        """The checkpoint's own name for `name`: a bare decoder (Gemma2Model, a pipeline's text encoder) has no
        "model." prefix."""
        if name not in self._where and name.startswith("model.") and name[6:] in self._where:
            return name[6:]
        return name

    def has(self, name):
        return self._key(name) in self._where

    def tensor(self, name) -> np.ndarray:
        from safetensors import safe_open
        k = self._key(name)
        with safe_open(self._where[k], "pt") as f:
            return f.get_tensor(k).float().numpy()

    def fingerprint(self) -> str:
        """The checkpoint's identity: shard names and sizes (the content is hashed by HF)."""
        h = hashlib.sha256()
        for fn in sorted(set(self._where.values())):
            h.update(os.path.basename(fn).encode()); h.update(str(os.path.getsize(fn)).encode())
        h.update(json.dumps(self.config, sort_keys=True).encode())
        return h.hexdigest()[:16]


def llama_dims(cfg: dict, N: int = 32768) -> Dims:
    """arch.model_dims (the name the runners used before the Arch)."""
    return model_dims(cfg, N)


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
    q: EncodedLinear | None          # None: Q rides K's padding heads in `kv` (q_pad)
    k: EncodedLinear | None
    v: EncodedLinear | None
    out: EncodedLinear
    gate: EncodedLinear
    up: EncodedLinear
    down: EncodedLinear
    tag: str = ""
    kv: EncodedLinear | None = None   # cachemir_complex: the fused K + iV linear (k, v unused)
    post1: np.ndarray | None = None   # Arch.post_norms: the attention output norm's gamma / inv_out_scale, (dim,)
    post2: np.ndarray | None = None   # the MLP output norm's
    q_pad: int = 0                    # Q's head slots in `kv`: its heads moved this many slots up (q_pad_heads)
    kv_entry: str = "kv"              # the slot cache entry of `kv` ("kvq" with Q on it)
    mlp: tuple = ()                   # LLAMA_MLP_SPLIT: the MLP's parts (MlpPart), gate / up / down = part 0's


@dataclasses.dataclass
class MlpPart:
    """One power-of-two slice [off, off + width) of the MLP's real width (mlp_parts), its own gate / up / down."""
    off: int
    width: int
    real: int                          # its real columns (the last part's may stop short of width)
    gate: EncodedLinear
    up: EncodedLinear
    down: EncodedLinear

    @property
    def tag(self):
        return f"s{self.off}x{self.width}"


def mlp_parts(dims: Dims) -> list[tuple[int, int, int]]:
    """LLAMA_MLP_SPLIT=<n>: the MLP's real width E_real as at most n power-of-two slices (off, width, real), largest
    first, instead of one padded to E (Gemma-2-2B: 9216 = 8192 + 1024 of 16384, 44 % fewer plaintexts in gate / up /
    down). [] (one padded MLP) when off, when it fits E already, or when n slices do not cover it."""
    n = int(os.environ.get("LLAMA_MLP_SPLIT", "0") or 0)
    Er, E = dims.E_real, dims.E
    if n < 2 or Er >= E:
        return []
    out, off = [], 0
    while off < Er and len(out) < n:
        rest = Er - off
        w = 1 << (rest.bit_length() - 1)                    # the largest power of two within what is left
        if len(out) == n - 1 and w < rest:
            w <<= 1                                         # the last slice covers the rest, padded
        out.append((off, w, min(w, rest)))
        off += w
    return out if off >= Er and sum(w for _, w, _ in out) < E else []


def interleave_rows(W, d: int):
    """W's rows in the input order a (rows, d) cachemir linear with d < rows reads the hidden layout in
    (encode_linear_input: slot m tp holds feature interleave_idx(m, d, rows)): the projection onto an MLP slice
    narrower than the hidden reads the hidden as it is."""
    from perseus.impl.layout import interleave_idx
    out = np.zeros_like(W)
    out[interleave_idx(np.arange(W.shape[0]), d, W.shape[0])] = W
    return out


def interleave_cols(W, d: int):
    """W's columns in the output order a (d, cols) cachemir linear with d < cols writes (decode_linear_output), so
    the down projection of an MLP slice narrower than the hidden lands in the hidden layout."""
    from perseus.impl.layout import interleave_idx
    out = np.zeros_like(W)
    out[:, interleave_idx(np.arange(W.shape[1]), d, W.shape[1])] = W
    return out


def mlp_mask(dims: Dims, width: int, real: int, scale: float) -> np.ndarray:
    """The slots of an MLP slice's activation input that hold its real features (the gate projection hid -> width
    lands them as decode_linear_output reads them), at `scale`; 0 elsewhere."""
    from perseus.impl.layout import cm_params, interleave_idx
    p = cm_params(dims.N, dims.hid, width)
    out = np.zeros(dims.N)
    if p.is_up and p.alpha > 1:
        m = np.arange(dims.N // p.tp)
        ok = interleave_idx(m, p.d, width) < real
        out[m[ok] * p.tp] = scale
    else:
        out[np.arange(real) * p.t] = scale
    return out


def q_pad_heads(dims: Dims) -> int:
    """Q rides K's zero padding heads in the fused K + iV linear (cachemir_complex), as GPT-2's prefill puts V on K's
    padding heads (PF_QKV_PACK): the head slots Q moves up by when the model's heads fill at most half of them
    (Gemma-2-2B: 8 of 16), so the QKV step is one complex linear; 0 (Q its own linear) when they fill more (Llama,
    GPT-2). LLAMA_Q_PAD=0 keeps Q separate."""
    if os.environ.get("LLAMA_Q_PAD", "1") in ("", "0"):
        return 0
    return dims.H_real if 0 < dims.H_real and 2 * dims.H_real <= dims.H else 0


def pad_q(Q, dims: Dims, off: int):
    """Q's real head columns (rearranged: column r = h + H i) moved `off` head slots up, onto K's zero padding heads."""
    r = np.arange(Q.shape[1])
    src = r[(r % dims.H) < dims.H_real]
    out = np.zeros_like(Q)
    out[:, src + off] = Q[:, src]
    return out


@dataclasses.dataclass
class BlockMatrices:
    Q: np.ndarray; K: np.ndarray; V: np.ndarray; O: np.ndarray
    Gate: np.ndarray; Up: np.ndarray; Down: np.ndarray
    post1: np.ndarray | None = None
    post2: np.ndarray | None = None


def block_matrices(store: SafetensorsStore, cfgs, b: int, dims: Dims) -> BlockMatrices:
    c = store.config
    arch = arch_of(c)
    H, H_kv = int(c["num_attention_heads"]), int(c["num_key_value_heads"])
    dh = dims.d_head_real
    ln1, ln2, _sm, act = cfgs.block(b)
    p = f"model.layers.{b}."
    t = lambda n: store.tensor(p + n).astype(np.float64)
    g1 = (arch.norm_offset + t("input_layernorm.weight")) * ln1.descale
    g2 = (arch.norm_offset + t(f"{arch.pre_mlp_norm}.weight")) * ln2.descale
    R = rotation_of(cfgs, dims.dim)                      # the residual stream is x R (or x)
    rin = (lambda W: R.T @ W) if R is not None else (lambda W: W)
    rout = (lambda W: W @ R) if R is not None else (lambda W: W)
    Q = rin(t("self_attn.q_proj.weight").T * g1[:, None])
    if arch.q_fold != 1.0:          # the Arch's score scale over the runtime's 1/sqrt(head_dim)
        Q = Q * arch.q_fold
    K = rin(expand_kv(t("self_attn.k_proj.weight").T, H, H_kv, dh) * g1[:, None])
    V = rin(expand_kv(t("self_attn.v_proj.weight").T, H, H_kv, dh) * g1[:, None])
    Q, K, V = (rearrange_qkv_weights(pad(M, dims.hid, dims.hid), dims.H) for M in (Q, K, V))
    O = rearrange_wo_weights(pad(rout(t("self_attn.o_proj.weight").T), dims.hid, dims.hid), dims.H)
    Gate = pad(rin(t("mlp.gate_proj.weight").T * (g2 / act.xmax)[:, None]), dims.hid, dims.E)
    Up = pad(rin(t("mlp.up_proj.weight").T * g2[:, None]), dims.hid, dims.E)
    Down = pad(rout(t("mlp.down_proj.weight").T), dims.E, dims.hid)
    post = [None, None]
    if arch.post_norms:
        if R is not None:
            raise ValueError("post norms with a rotated residual: their gamma would need the rotation too")
        post = [(arch.norm_offset + t(f"{m}.weight")) * n.descale for m, n in zip(arch.post_norms, cfgs.post(b))]
    return BlockMatrices(Q, K, V, O, Gate, Up, Down, *post)


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


class LlamaWeights:
    """The slot vectors of every block and of the LM head, built once per (checkpoint, configs,
    packing, builder sources) into the disk cache and memory-mapped from there."""

    def __init__(self, store: SafetensorsStore, cfgs, configs_path: str, dims: Dims,
                 packing: str = "cachemir_complex", cache_dir: str | None = None, pinned=None):
        """`pinned(rows, cols)`: a float32 array in pinned host memory (perseus._core.pinned_empty); given, every
        linear also carries its plaintexts as coefficient rows there (EncodedLinear.coeffs, the device-encoded
        inner sums), built once from the slot vectors into the same disk cache."""
        self.store, self.cfgs, self.dims, self.packing = store, cfgs, dims, packing
        self.arch = arch_of(store.config)
        self.pinned = pinned
        root = cache_dir or os.environ.get("PERSEUS_LLAMA_CACHE") or os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "perseus", "llama_slots")
        key = hashlib.sha256(f"{store.fingerprint()}|{_folded_config(cfgs)}|{builder_key()}|{dims}|{packing}"
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

    def stream_tags(self, name, pts):
        """The plaintext tags of slot-cache entry `name` ((n, 7) int32 pt_tag_fields), cached next to it."""
        ft = self.dir / f"{name}.tags.npy"
        if not ft.exists():
            tmp = self.dir / f".{name}.tags.{os.getpid()}.tmp.npy"
            np.save(tmp, np.asarray([pt_tag_fields(v) for v in pts], dtype=np.int32))
            os.replace(tmp, ft)
        return np.load(ft)

    def coeff_rows(self, name):
        """The float32 coefficient rows of slot-cache entry `name` ((n, 2 slots), memory-mapped): built from the slot
        vectors (layout.coeff_rows) at the first use."""
        from perseus.impl.layout import coeff_rows
        f = self.dir / f"{name}.coeff.npy"
        if not f.exists():
            pts = np.load(self.dir / f"{name}.npy", mmap_mode="r")
            tmp = self.dir / f".{name}.coeff.{os.getpid()}.tmp.npy"
            np.save(tmp, coeff_rows(pts))
            os.replace(tmp, f)
        return np.load(f, mmap_mode="r")

    def _coeffs(self, name, pts):
        """The slot vectors' coefficient rows (layout.coeff_rows) in pinned memory and their plaintext tags, from
        the disk cache next to the slot vectors (built at the first use)."""
        from perseus import _pc as _core
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
        post = [None, None]
        if self.arch.post_norms:
            post = [np.asarray(v, dtype=np.float64)
                    for v in self._cached(pre + "post", lambda: np.stack([mats().post1, mats().post2]))]
        off = q_pad_heads(d) if cplx else 0
        w = BlockWeights(
            q=None if off else lin("q", lambda: mats().Q, hid, hid),
            k=None if cplx else lin("k", lambda: mats().K, hid, hid),
            v=None if cplx else lin("v", lambda: mats().V, hid, hid),
            out=lin("out", lambda: mats().O, hid, hid),
            gate=None if mlp_parts(d) else lin("gate", lambda: mats().Gate, hid, E, cplx),
            up=None if mlp_parts(d) else lin("up", lambda: mats().Up, hid, E, cplx),
            down=None if mlp_parts(d) else lin("down", lambda: mats().Down, E, hid, cplx),
            tag=pre,
            kv=(lin("kvq", lambda: mats().K + pad_q(mats().Q, d, off) + 1j * mats().V, hid, hid) if off
                else lin("kv", lambda: mats().K + 1j * mats().V, hid, hid)) if cplx else None,
            post1=post[0], post2=post[1], q_pad=off, kv_entry="kvq" if off else "kv")
        parts = mlp_parts(d)
        if parts:
            from perseus.impl.layout import cm_params
            mp = []
            for o, wd, real in parts:
                sl = slice(o, o + wd)
                narrow = wd < hid
                g = (lambda M: interleave_rows(M, wd)) if narrow else (lambda M: M)
                dn = (lambda M: interleave_cols(M, wd)) if narrow else (lambda M: M)
                pk_up = cplx and cm_params(d.N, hid, wd).r_o % 2 == 0
                pk_dn = cplx and cm_params(d.N, wd, hid).r_o % 2 == 0
                tg = f"s{o}x{wd}"
                mp.append(MlpPart(o, wd, real,
                                  lin(f"gate.{tg}", lambda sl=sl, g=g: g(mats().Gate[:, sl]), hid, wd, pk_up),
                                  lin(f"up.{tg}", lambda sl=sl, g=g: g(mats().Up[:, sl]), hid, wd, pk_up),
                                  lin(f"down.{tg}", lambda sl=sl, dn=dn: dn(mats().Down[sl, :]), wd, hid, pk_dn)))
            w = dataclasses.replace(w, gate=mp[0].gate, up=mp[0].up, down=mp[0].down, mlp=tuple(mp))
        return w

    def lm_tiles(self, paired: bool = True):
        """encode_gpt2_lm_head_tile: ceil(vocab / N) tiles of (hid, N); `paired`: tiles 2m and
        2m+1 as one complex tile (real and imaginary axis). A tied head (Arch.tied) is the embedding matrix."""
        d = self.dims
        N = d.N
        vocab = int(self.store.config["vocab_size"])
        K = (vocab + N - 1) // N
        m = {}
        def W():
            if not m:
                g = (self.arch.norm_offset + self.store.tensor("model.norm.weight").astype(np.float64)) \
                    * self.cfgs.ln_f.descale
                head = "model.embed_tokens.weight" if self.arch.tied else "lm_head.weight"
                m["W"] = self.store.tensor(head).T.astype(np.float64) * g[:, None]
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
        cplx = self.packing == "cachemir_complex"
        qkv = (("kvq",) if q_pad_heads(self.dims) else ("q", "kv")) if cplx else ("q", "k", "v")
        mlp = [f"{f}.s{o}x{w}" for o, w, _ in mlp_parts(self.dims) for f in ("gate", "up", "down")]
        per = qkv + ("out",) + (tuple(mlp) or ("gate", "up", "down"))
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
        if self.arch.embed_scale != 1.0:
            e = e * self.arch.embed_scale
        return e if self._R is None else e @ self._R


def pt_tag_fields(values):
    """The packing signature of a plaintext's slot values as the 7 ints of the old `_core.pt_tag_fields` (slots,
    period, live, stride, window, offset, field 0 real / 1 imag / 2 complex): fakecap.pt_tag's measurement, which
    `Tag.from_signature` + the field rebuild."""
    v = np.asarray(values)
    cplx = np.iscomplexobj(v)
    re_ = v.real.astype(np.float64)
    im = v.imag.astype(np.float64) if cplx else np.zeros_like(re_)
    n = re_.shape[0]
    amax, imax = float(np.abs(re_).max(initial=0.0)), float(np.abs(im).max(initial=0.0))
    im_data = imax > 1e-3 * max(amax, 1e-300)
    tol = max(1e-2 * (max(amax, imax) if im_data else amax), 1e-12)
    period, p = n, 1
    while p < n:
        a = re_.reshape(-1, p)
        ok = np.abs(a - a[0]).max() <= tol
        if ok and im_data:
            b = im.reshape(-1, p)
            ok = np.abs(b - b[0]).max() <= tol
        if ok:
            period = p
            break
        p <<= 1
    nz = np.flatnonzero((re_ != 0.0) | (im != 0.0))
    live = int(nz.size)
    window = int(nz[-1]) + 1 if live else 0
    offset = int(nz[0]) if live else 0
    stride = int(np.gcd.reduce(nz - nz[0])) if live >= 2 else (n if live == 1 else 0)
    if cplx:
        r, i = bool((re_ != 0).any()), bool((im != 0).any())
        field = (2 if r else 1) if i else 0
    else:
        field = 0
    return (n, period, live, stride, window, offset, field)


def _build_coeff_file(d: str, name: str):
    """Worker: <name>.coeff.npy from <name>.npy (LlamaWeights._coeffs builds the same at first use)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from perseus.impl.layout import coeff_rows
    d = Path(d)
    pts = np.load(d / f"{name}.npy", mmap_mode="r")
    tmp = d / f".{name}.coeff.{os.getpid()}.tmp.npy"
    np.save(tmp, coeff_rows(pts))
    os.replace(tmp, d / f"{name}.coeff.npy")
