"""HF GPT-2 weights (the perseus-export zip) -> the padded, transposed, LN-folded,
head-rearranged matrices the cachemir runtime encodes (include/weight_loader.h, src/model/block_residency.cu) and their slot vectors (EncodedLinear).

``RawStore`` reads the export zip directly (manifest.json + '<f4' blobs) so the CPU tier
needs no CUDA extension; a ``_core.WeightStore`` works too (same ``tensor(name)``)."""
from __future__ import annotations

import dataclasses
import json
import zipfile

import numpy as np

from perseus.impl.config import Configs, NormCfg
from perseus.impl.layout import Dims, rearrange_qkv_biases, rearrange_qkv_weights, rearrange_wo_weights
from perseus.impl.linear import EncodedLinear

WTE = "transformer.wte.weight"
WPE = "transformer.wpe.weight"
LNF = "transformer.ln_f"


class RawStore:
    def __init__(self, zip_path: str):
        self._zip = zipfile.ZipFile(zip_path)
        man = [n for n in self._zip.namelist() if n.endswith("manifest.json")][0]
        m = json.loads(self._zip.read(man))
        self._meta = {e["name"]: e for e in m["tensors"]}
        self._cache = {}

    def names(self):
        return sorted(self._meta)

    def has(self, name):
        return name in self._meta

    def shape(self, name):
        return list(self._meta[name]["shape"])

    def tensor(self, name):
        if name not in self._cache:
            e = self._meta[name]
            buf = self._zip.read(e["path"])
            self._cache[name] = np.frombuffer(buf, dtype=np.dtype(e["dtype"])).astype(np.float64) \
                .reshape(e["shape"])
        return self._cache[name]


def block_names(b: int):
    base = f"transformer.h.{b}"
    return dict(ln1_w=f"{base}.ln_1.weight", ln1_b=f"{base}.ln_1.bias",
                ln2_w=f"{base}.ln_2.weight", ln2_b=f"{base}.ln_2.bias",
                qkv_w=f"{base}.attn.c_attn.weight", qkv_b=f"{base}.attn.c_attn.bias",
                out_w=f"{base}.attn.c_proj.weight", out_b=f"{base}.attn.c_proj.bias",
                up_w=f"{base}.mlp.c_fc.weight", up_b=f"{base}.mlp.c_fc.bias",
                down_w=f"{base}.mlp.c_proj.weight", down_b=f"{base}.mlp.c_proj.bias")


def pad_matrix(W, r, c):
    out = np.zeros((r, c))
    out[:W.shape[0], :W.shape[1]] = W
    return out


def pad_vector(v, n):
    out = np.zeros(n)
    out[:v.shape[0]] = v
    return out


def load_linear(store, name, d_in_real, d_out_real, d_in_pad, d_out_pad):
    """load_linear_weight (weight_loader.h): (d_out, d_in) storage -> (d_in, d_out) padded."""
    W = np.asarray(store.tensor(name), dtype=np.float64)
    r, c = W.shape
    if (r, c) == (d_out_real, d_in_real):
        return pad_matrix(W.T, d_in_pad, d_out_pad)
    if (r, c) == (d_out_pad, d_in_pad):
        return W.T.copy()
    raise ValueError(f"{name}: unexpected shape {W.shape}, expected ({d_out_real}, {d_in_real})")


def load_vector(store, name, d_real, d_pad):
    """load_vector (weight_loader.h): (real, padded)."""
    v = np.asarray(store.tensor(name), dtype=np.float64).ravel()
    if v.shape[0] == d_real:
        return v, pad_vector(v, d_pad)
    if v.shape[0] == d_pad:
        return v[:d_real].copy(), v
    raise ValueError(f"{name}: unexpected vector size {v.shape[0]}")


def load_qkv(store, name, d_real, d_pad):
    """load_qkv_weight (weight_loader.h): (3d, d) -> three (d_pad, d_pad)."""
    W = np.asarray(store.tensor(name), dtype=np.float64)
    r, c = W.shape
    if (r, c) == (3 * d_real, d_real):
        d = d_real
    elif (r, c) == (3 * d_pad, d_pad):
        d = d_pad
    else:
        raise ValueError(f"{name}: unexpected QKV shape {W.shape}")
    Wt = W.T
    parts = [Wt[:, i * d:(i + 1) * d] for i in range(3)]
    return [pad_matrix(P, d_pad, d_pad) for P in parts]


def load_qkv_bias(store, name, d_real, d_pad):
    b = np.asarray(store.tensor(name), dtype=np.float64).ravel()
    if b.shape[0] == 3 * d_real:
        d = d_real
    elif b.shape[0] == 3 * d_pad:
        d = d_pad
    else:
        raise ValueError(f"{name}: unexpected QKV bias size {b.shape[0]}")
    return [pad_vector(b[i * d:(i + 1) * d], d_pad) for i in range(3)]


def ln_input_shift(gamma, beta, d_real):
    """weight_loader.h: beta/gamma where gamma != 0 (zeros elsewhere)."""
    s = np.zeros(gamma.shape[0])
    g = gamma[:d_real]
    nz = g != 0
    s[:d_real][nz] = beta[:d_real][nz] / g[nz]
    return s


@dataclasses.dataclass
class BlockWeights:
    q: EncodedLinear
    k: EncodedLinear | None
    v: EncodedLinear | None
    out: EncodedLinear
    up: EncodedLinear
    down: EncodedLinear
    shift1: np.ndarray | None = None
    shift2: np.ndarray | None = None
    gamma1: np.ndarray | None = None
    beta1: np.ndarray | None = None
    gamma2: np.ndarray | None = None
    beta2: np.ndarray | None = None
    tag: str = ""                     # plaintext-name prefix of the block ("b3.")
    kv: EncodedLinear | None = None   # cachemir_complex: the fused K + iV linear (k, v unused)


@dataclasses.dataclass
class BlockMatrices:
    """The padded / folded / rearranged matrices before slot encoding (for tests)."""
    Q: np.ndarray; K: np.ndarray; V: np.ndarray; bq: np.ndarray; bk: np.ndarray; bv: np.ndarray
    O: np.ndarray; bo: np.ndarray; Up: np.ndarray; bup: np.ndarray; Down: np.ndarray; bdown: np.ndarray
    g1: np.ndarray; b1: np.ndarray; g2: np.ndarray; b2: np.ndarray


def block_matrices(store, cfgs: Configs, b: int, dims: Dims, fold_ln1=True, fold_ln2=True):
    """encode_gpt2_layer_weights (weight_loader.h) up to the slot encoding."""
    n = block_names(b)
    dr, dp, er, ep, H = dims.dim, dims.hid, dims.E_real, dims.E, dims.H
    ln1, ln2, _, _ = cfgs.block(b)
    g1, g1_pad = load_vector(store, n["ln1_w"], dr, dp)
    b1 = load_vector(store, n["ln1_b"], dr, dp)[0]
    g2, g2_pad = load_vector(store, n["ln2_w"], dr, dp)
    b2 = load_vector(store, n["ln2_b"], dr, dp)[0]
    g1, g1_pad = g1 * ln1.descale, g1_pad * ln1.descale       # block_residency.cu
    g2, g2_pad = g2 * ln2.descale, g2_pad * ln2.descale
    Q, K, V = load_qkv(store, n["qkv_w"], dr, dp)
    bq, bk, bv = load_qkv_bias(store, n["qkv_b"], dr, dp)
    if fold_ln1:
        Q, K, V = (M * g1_pad[:, None] for M in (Q, K, V))   # mat_scale_rows
    Q, K, V = (rearrange_qkv_weights(M, H) for M in (Q, K, V))
    bq, bk, bv = (rearrange_qkv_biases(v, H) for v in (bq, bk, bv))
    O = rearrange_wo_weights(load_linear(store, n["out_w"], dr, dr, dp, dp), H)
    bo = load_vector(store, n["out_b"], dr, dp)[1]
    Up = load_linear(store, n["up_w"], dr, er, dp, ep)
    bup = load_vector(store, n["up_b"], er, ep)[1]
    if fold_ln2:
        Up = Up * g2_pad[:, None]
    Down = load_linear(store, n["down_w"], er, dr, ep, dp)
    bdown = load_vector(store, n["down_b"], dr, dp)[1]
    return BlockMatrices(Q, K, V, bq, bk, bv, O, bo, Up, bup, Down, bdown, g1, b1, g2, b2)


def block_weights(store, cfgs: Configs, b: int, dims: Dims, fold_ln1=True, fold_ln2=True,
                  complex_packing=False):
    """The block's EncodedLinears (weight_loader.h, untiled) plus the LN shifts (folded) or
    descaled gamma/beta (unfolded). `complex_packing` (cachemir_complex): one fused K + iV
    linear with a complex bias, and the up- and down-projections output-packed."""
    m = block_matrices(store, cfgs, b, dims, fold_ln1, fold_ln2)
    N, dp, ep = dims.N, dims.hid, dims.E
    enc = EncodedLinear.encode
    pre = f"b{b}."
    if complex_packing:
        w = BlockWeights(
            q=enc(m.Q, N, dp, dp, m.bq, name=pre + "q"), k=None, v=None,
            out=enc(m.O, N, dp, dp, m.bo, name=pre + "out"),
            up=enc(m.Up, N, dp, ep, m.bup, name=pre + "up", outputpack=True),
            down=enc(m.Down, N, ep, dp, m.bdown, name=pre + "down", outputpack=True), tag=pre,
            kv=enc(m.K + 1j * m.V, N, dp, dp, m.bk + 1j * m.bv, name=pre + "kv"))
    else:
        w = BlockWeights(
            q=enc(m.Q, N, dp, dp, m.bq, name=pre + "q"), k=enc(m.K, N, dp, dp, m.bk, name=pre + "k"),
            v=enc(m.V, N, dp, dp, m.bv, name=pre + "v"), out=enc(m.O, N, dp, dp, m.bo, name=pre + "out"),
            up=enc(m.Up, N, dp, ep, m.bup, name=pre + "up"),
            down=enc(m.Down, N, ep, dp, m.bdown, name=pre + "down"), tag=pre)
    if fold_ln1:
        w.shift1 = ln_input_shift(m.g1, m.b1, dims.dim)
    else:
        w.gamma1, w.beta1 = m.g1, m.b1
    if fold_ln2:
        w.shift2 = ln_input_shift(m.g2, m.b2, dims.dim)
    else:
        w.gamma2, w.beta2 = m.g2, m.b2
    return w


def final_ln_params(store, cfg_lnf: NormCfg, dims: Dims, fold_lnf=False):
    """encode_gpt2_final_ln_weights (weight_loader.h): (gamma_desc, beta) or a shift."""
    g = load_vector(store, LNF + ".weight", dims.dim, dims.hid)[0] * cfg_lnf.descale
    b = load_vector(store, LNF + ".bias", dims.dim, dims.hid)[0]
    if fold_lnf:
        return {"shift": ln_input_shift(g, b, dims.dim)}
    return {"gamma_desc": g, "beta": b}


def lm_head_matrix(store, cfg_lnf: NormCfg, dims: Dims, vocab: int, fold_lnf=False):
    """load_gpt2_lm_head_weight (weight_loader.h): (hid, vocab), gamma-folded when
    ln_f is folded."""
    W = load_linear(store, WTE, dims.dim, vocab, dims.hid, vocab)
    if fold_lnf:
        g_pad = load_vector(store, LNF + ".weight", dims.dim, dims.hid)[1] * cfg_lnf.descale
        W = W * g_pad[:, None]
    return W


def lm_head_tiles(W_lm_pad, dims: Dims, vocab: int, W_tile: int, paired: bool = False):
    """encode_gpt2_lm_head_tile (weight_loader.h): K = ceil(vocab / W_tile) tiles of
    (hid, W_tile), no bias (ln_f unfolded); `paired` (cachemir_complex): ceil(K/2) complex
    tiles, each carrying tiles 2m (real axis) and 2m+1 (imaginary axis)."""
    K = (vocab + W_tile - 1) // W_tile
    def slice_tile(k):
        if k >= K:
            return np.zeros((dims.hid, W_tile))
        col0 = k * W_tile
        wreal = min(W_tile, vocab - col0)
        return pad_matrix(W_lm_pad[:, col0:col0 + wreal], dims.hid, W_tile)
    if paired:
        return [EncodedLinear.encode(slice_tile(2 * m) + 1j * slice_tile(2 * m + 1), dims.N,
                                     dims.hid, W_tile, name=f"lm.c{m}") for m in range((K + 1) // 2)]
    return [EncodedLinear.encode(slice_tile(k), dims.N, dims.hid, W_tile, name=f"lm.{k}")
            for k in range(K)]


def feedback_tiles(W_lm_pad, dims: Dims, vocab: int, W_tile: int, packed: bool = False):
    """gpt2_prepare_feedback_weights (gpt2_embedding.cu): W_fb[k][r][j] =
    W_lm_pad[j][k*W_tile + r], a (W_tile, hid) linear per tile; `packed` (the complex-payload
    arm, K = 2): ONE complex tile 0.5 W_fb[0] - 0.5 i W_fb[1] applied to the packed pair."""
    K = (vocab + W_tile - 1) // W_tile
    def slice_(k, scale):
        col0 = k * W_tile
        wreal = min(W_tile, vocab - col0)
        Wk = np.zeros((W_tile, dims.hid))
        Wk[:wreal, :dims.dim] = scale * W_lm_pad[:dims.dim, col0:col0 + wreal].T
        return Wk
    if packed:
        if K != 2:
            raise ValueError("packed feedback needs exactly 2 vocab tiles")
        Wc = slice_(0, 0.5) + 1j * slice_(1, -0.5)
        return [EncodedLinear.encode(Wc, dims.N, W_tile, dims.hid, name="fb.c")]
    return [EncodedLinear.encode(slice_(k, 1.0), dims.N, W_tile, dims.hid, name=f"fb.{k}")
            for k in range(K)]


def wpe_row(store, position: int, d_real: int):
    """gpt2_add_positional (gpt2_embedding.cu)."""
    wpe = store.tensor(WPE)
    if not 0 <= position < wpe.shape[0]:
        raise ValueError(f"position {position} out of [0, {wpe.shape[0]})")
    return np.asarray(wpe[position, :d_real], dtype=np.float64)


def token_embedding(store, token: int, position: int, d_real: int):
    """wte[token] + wpe[position] (the client-side embedding, _core.token_embedding)."""
    return np.asarray(store.tensor(WTE)[token, :d_real], dtype=np.float64) + wpe_row(store, position, d_real)


def vocab_size(store):
    return int(store.shape(WTE)[0])
