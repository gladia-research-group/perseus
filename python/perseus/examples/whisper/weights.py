"""Whisper (openai/whisper-base.en) weights under the GPT-2 export's tensor names, the client's audio -> encrypted-input
rows, the convolutional front-end as the two linears the server runs, and the float64 reference.

The front-end, rewritten so the server runs the shape of an MLP (no token shifts, no stride):

    client:  log-mel (80, N_FRAMES) -> rows (T, 400), row t = mel frames 2t-2 .. 2t+2 (zero outside the window)
    server:  conv1 as one linear 400 -> 3 x 512 (conv1's outputs at frames 2t-1, 2t, 2t+1), GELU,
             conv2 as one linear 3 x 512 -> 512, GELU, + the sinusoidal position embedding.

conv2's zero padding is frame -1 of conv1's output: row 0's first 512 GELU outputs are zeroed (`STEM_PAD`).
"""
from __future__ import annotations

import json
import os
import types

import numpy as np

MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-base.en")
SR = 16000                      # samples per second
# The encoder's window: Whisper's 30-s log-mel cut to its first WINDOW_S seconds, the position embedding to its first T
# rows. A shorter window keeps the 30-s window's accuracy on clips that leave it >= 2 s of silence (test-clean, float:
# 10-s window on clips <= 8 s WER 3.99 % vs 4.08 %; clips filling it loop), at ~(WINDOW_S / 30)^2 of the attention.
WINDOW_S = int(os.environ.get("WHISPER_WINDOW_S", "10"))
N_MELS, N_FRAMES = 80, 100 * WINDOW_S     # the window's log-mel frames (10 ms hop)
T = N_FRAMES // 2               # encoder positions (conv2's stride)
MAX_CLIP_S = WINDOW_S - 2 if WINDOW_S < 30 else 30.0     # the longest clip a window transcribes
WIN = 5                         # mel frames per row: 2t-2 .. 2t+2
STEM_PAD = (0, 0)               # (row, conv1 output) zeroed after the first GELU: conv2's padding frame -1


def snapshot():
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt"])


_BLOCK = {                     # GPT-2 block tensor -> Whisper layer tensor (self-attention sublayer and MLP)
    "ln_1": "self_attn_layer_norm", "ln_2": "final_layer_norm", "attn.c_proj": "self_attn.out_proj",
    "mlp.c_fc": "fc1", "mlp.c_proj": "fc2",
    # the decoder's cross-attention sublayer (the GPT-2 grammar has none: these two names are the port's)
    "ln_x": "encoder_attn_layer_norm", "xattn.c_proj": "encoder_attn.out_proj",
}


class WhisperStore:
    """tensor(name) for the GPT-2 names the GPT-2 machinery reads (float64), over one half of the model: `part`
    "encoder" (its blocks, ln_f = the encoder's final LayerNorm) or "decoder" (its blocks with `ln_x` / `xattn.*`
    for the cross-attention, wte / wpe the token and position embeddings, ln_f the decoder's final LayerNorm)."""

    def __init__(self, part="encoder", path=None):
        from safetensors.numpy import load_file
        if part not in ("encoder", "decoder"):
            raise ValueError(f"part {part!r}")
        self.part = part
        self.path = path or snapshot()
        self.raw = load_file(os.path.join(self.path, "model.safetensors"))
        hf = json.load(open(os.path.join(self.path, "config.json")))
        self.hf_config = hf
        # the keys the shared float64 references read (examples.vit.weights.exact_blocks)
        self.config = {"num_attention_heads": hf[f"{part}_attention_heads"], "hidden_size": hf["d_model"],
                       "layer_norm_eps": 1e-5, "num_hidden_layers": hf[f"{part}_layers"]}
        self._zip = types.SimpleNamespace(filename=os.path.join(self.path, "model.safetensors"))
        self._cache = {}

    @property
    def n_layers(self):
        return self.config["num_hidden_layers"]

    def r(self, n):
        return np.asarray(self.raw[n], dtype=np.float64)

    def _qkv(self, L, att, s):
        if s == "weight":
            return np.concatenate([self.r(f"{L}{att}.{x}_proj.weight") for x in ("q", "k", "v")], axis=0)
        q = self.r(f"{L}{att}.q_proj.bias")                 # Whisper's key projection has no bias
        return np.concatenate([q, np.zeros_like(q), self.r(f"{L}{att}.v_proj.bias")])

    def _map(self, name):
        P = f"model.{self.part}."
        p = name.split(".")
        if name == "transformer.wte.weight":           # the model's vocabulary (the encoder half never reads it)
            return self.r("model.decoder.embed_tokens.weight")
        if name == "transformer.wpe.weight":
            return self.r(P + "embed_positions.weight")
        if name.startswith("transformer.ln_f."):
            return self.r(P + "layer_norm." + p[-1])
        if not name.startswith("transformer.h."):
            raise KeyError(name)
        b, rest, s = int(p[2]), ".".join(p[3:-1]), p[-1]
        L = f"{P}layers.{b}."
        if rest == "attn.c_attn":
            return self._qkv(L, "self_attn", s)
        if rest == "xattn.c_attn" and self.part == "decoder":
            return self._qkv(L, "encoder_attn", s)
        if rest in _BLOCK and (self.part == "decoder" or not rest.startswith(("ln_x", "xattn"))):
            return self.r(f"{L}{_BLOCK[rest]}.{s}")
        raise KeyError(name)

    def tensor(self, name):
        if name not in self._cache:
            self._cache[name] = self._map(name)
        return self._cache[name]

    def has(self, name):
        try:
            self.tensor(name)
            return True
        except KeyError:
            return False

    def shape(self, name):
        return list(self.tensor(name).shape)

    def names(self):
        return sorted(self.raw)


# ── client: audio -> rows ────────────────────────────────────────────────────────────────────────────────────────────

def load_audio(path):
    """A WAV file as float64 mono at 16 kHz."""
    from perseus.audio import read_wav
    return read_wav(path, SR)


def log_mel(audio, path=None):
    """Whisper's log-mel of one clip in the window: (80, N_FRAMES) (transformers' feature extractor's 30-s log-mel, its
    first N_FRAMES frames: the same values a shorter clip's own padding gives)."""
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained(path or snapshot())
    return np.asarray(fe(np.asarray(audio, dtype=np.float32), sampling_rate=SR, return_tensors="np")
                      .input_features[0, :, :N_FRAMES], dtype=np.float64)


def mel_rows(mel):
    """(T, WIN * 80): row t = mel frames 2t-2 .. 2t+2 (frame-major), zero outside [0, N_FRAMES)."""
    mel = np.asarray(mel, dtype=np.float64)
    pad = np.zeros((N_MELS, N_FRAMES + 4))
    pad[:, 2:2 + N_FRAMES] = mel
    idx = 2 * np.arange(T)[:, None] + np.arange(WIN)[None, :]          # into the padded frames: 2t-2+s -> 2t+s
    return pad[:, idx].transpose(1, 2, 0).reshape(T, WIN * N_MELS)


# ── server: the front-end as two linears ─────────────────────────────────────────────────────────────────────────────

def stem_matrices(store):
    """(A (400, 3 d), a (3 d,), C (3 d, d), c (d,)): conv1 on a row's five frames -> its outputs at frames 2t-1, 2t,
    2t+1 (column block i: frame 2t-1+i), and conv2 (stride 2) on those three."""
    W1, b1 = store.r("model.encoder.conv1.weight"), store.r("model.encoder.conv1.bias")      # (d, 80, 3)
    W2, b2 = store.r("model.encoder.conv2.weight"), store.r("model.encoder.conv2.bias")      # (d, d, 3)
    d = W1.shape[0]
    A = np.zeros((WIN * N_MELS, 3 * d))
    for i in range(3):                      # conv1 output at frame 2t-1+i reads row frames s = i + j, j = 0, 1, 2
        for j in range(3):
            s = i + j
            A[s * N_MELS:(s + 1) * N_MELS, i * d:(i + 1) * d] = W1[:, :, j].T
    C = np.concatenate([W2[:, :, i].T for i in range(3)], axis=0)
    return A, np.tile(b1, 3), C, b2


def positions(store):
    """The window's position embedding: (T, d)."""
    return store.r("model.encoder.embed_positions.weight")[:T]


def gelu(x):
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def stem(store, rows, taps=None):
    """The front-end in float64 on the client's rows: the encoder blocks' input (T, d). `taps`: a dict receiving the
    two GELU inputs (h1, h2), the calibration's ranges."""
    A, a, C, c = stem_matrices(store)
    h1 = np.asarray(rows, dtype=np.float64) @ A + a
    g1 = gelu(h1)
    d = C.shape[1]
    r, i = STEM_PAD
    g1[r, i * d:(i + 1) * d] = 0.0
    h2 = g1 @ C + c
    if taps is not None:
        taps["h1"], taps["h2"] = h1, h2
    return gelu(h2) + positions(store)


def encoder_ref(store, rows):
    """Float64 encoder on the client's rows: (per-block hidden states, the output after the final LayerNorm)."""
    from perseus.examples.vit.weights import exact_blocks
    hs = exact_blocks(store, stem(store, rows), store.n_layers)
    return hs, layer_norm(hs[-1], store.tensor("transformer.ln_f.weight"), store.tensor("transformer.ln_f.bias"))


def layer_norm(h, g, b, eps=1e-5):
    mu = h.mean(axis=-1, keepdims=True)
    return (h - mu) / np.sqrt(((h - mu) ** 2).mean(axis=-1, keepdims=True) + eps) * g + b


# ── the decoder (float64) ────────────────────────────────────────────────────────────────────────────────────────────

class Generation:
    """Whisper base.en's greedy decoding rules (its generation_config.json): the forced start, the end-of-text token,
    the tokens never emitted (`suppress`) and those not emitted first (`begin_suppress`)."""

    def __init__(self, path=None):
        g = json.load(open(os.path.join(path or snapshot(), "generation_config.json")))
        forced = [t for _, t in sorted(g.get("forced_decoder_ids") or [])]
        self.start = [int(g["decoder_start_token_id"])] + forced
        self.eot = int(g["eos_token_id"])
        self.suppress = sorted(set(g.get("suppress_tokens", [])) | {int(g["no_timestamps_token_id"])})
        self.begin_suppress = sorted(g.get("begin_suppress_tokens", []))
        self.max_length = int(g.get("max_length", 448))

    def allowed(self, vocab, first=False):
        """The tokens an argmax may pick (first: at the first free position)."""
        ok = np.ones(vocab, dtype=bool)
        ok[self.suppress] = False
        if first:
            ok[self.begin_suppress] = False
        return ok


def _attn(q, k, v, H, causal):
    dh = q.shape[1] // H
    o = np.zeros_like(q)
    for h in range(H):
        sl = slice(h * dh, (h + 1) * dh)
        s = q[:, sl] @ k[:, sl].T / np.sqrt(dh)
        if causal:
            s = s + np.triu(np.full(s.shape, -np.inf), 1)
        p = np.exp(s - s.max(axis=1, keepdims=True))
        o[:, sl] = (p / p.sum(axis=1, keepdims=True)) @ v[:, sl]
    return o


def decoder_ref(store, enc_out, tokens):
    """Teacher-forced decoder in float64 (store: WhisperStore("decoder")): the logits of every position (T, vocab)."""
    from perseus.examples.gpt2 import weights as GW
    t, H = store.tensor, store.config["num_attention_heads"]
    n = len(tokens)
    x = t("transformer.wte.weight")[tokens] + t("transformer.wpe.weight")[:n]
    for b in range(store.n_layers):
        nm = GW.block_names(b)
        p = f"transformer.h.{b}."
        q, k, v = np.split(layer_norm(x, t(nm["ln1_w"]), t(nm["ln1_b"])) @ t(nm["qkv_w"]).T + t(nm["qkv_b"]), 3, 1)
        x = x + _attn(q, k, v, H, True) @ t(nm["out_w"]).T + t(nm["out_b"])
        Wx, bx = t(p + "xattn.c_attn.weight"), t(p + "xattn.c_attn.bias")
        d = x.shape[1]
        q = layer_norm(x, t(p + "ln_x.weight"), t(p + "ln_x.bias")) @ Wx[:d].T + bx[:d]
        k, v = enc_out @ Wx[d:2 * d].T + bx[d:2 * d], enc_out @ Wx[2 * d:].T + bx[2 * d:]
        x = x + _attn(q, k, v, H, False) @ t(p + "xattn.c_proj.weight").T + t(p + "xattn.c_proj.bias")
        u = layer_norm(x, t(nm["ln2_w"]), t(nm["ln2_b"])) @ t(nm["up_w"]).T + t(nm["up_b"])
        x = x + gelu(u) @ t(nm["down_w"]).T + t(nm["down_b"])
    x = layer_norm(x, t("transformer.ln_f.weight"), t("transformer.ln_f.bias"))
    return x @ t("transformer.wte.weight").T


def greedy(store, enc_out, max_new=128, gen=None):
    """Greedy transcription in float64 (quadratic: the reference, not a fast decoder): the token ids after the forced
    start, up to and including end-of-text."""
    gen = gen or Generation(store.path)
    toks = list(gen.start)
    vocab = store.shape("transformer.wte.weight")[0]
    for i in range(max_new):
        lg = decoder_ref(store, enc_out, toks)[-1]
        lg[~gen.allowed(vocab, first=(i == 0))] = -np.inf
        toks.append(int(np.argmax(lg)))
        if toks[-1] == gen.eot:
            break
    return toks[len(gen.start):]


# ── the cross-attention's K / V: the encoder's output projected once per utterance ───────────────────────────────────

def cross_kv(store, enc_out):
    """[(K, V)] per decoder layer in float64 (store: WhisperStore("decoder"); enc_out: ln_post's output (T, d)), heads
    in HF's order (h d_head + j)."""
    t, d = store.tensor, enc_out.shape[1]
    out = []
    for b in range(store.n_layers):
        Wx, bx = t(f"transformer.h.{b}.xattn.c_attn.weight"), t(f"transformer.h.{b}.xattn.c_attn.bias")
        out.append((enc_out @ Wx[d:2 * d].T + bx[d:2 * d], enc_out @ Wx[2 * d:].T + bx[2 * d:]))
    return out


def cross_kv_matrices(store, H):
    """Every decoder layer's K and V projections as one (d, 2 L d) matrix and bias (columns 2 l d .. : layer l's K,
    then its V), the output channels in the decode's head order (layout.rearrange_qkv_weights: j H + h)."""
    from perseus.impl.layout import rearrange_qkv_biases, rearrange_qkv_weights
    t = store.tensor
    Ws, bs = [], []
    for b in range(store.n_layers):
        Wx, bx = t(f"transformer.h.{b}.xattn.c_attn.weight"), t(f"transformer.h.{b}.xattn.c_attn.bias")
        d = Wx.shape[1]
        for i in (1, 2):                               # K, V
            Ws.append(rearrange_qkv_weights(Wx[i * d:(i + 1) * d].T, H))
            bs.append(rearrange_qkv_biases(bx[i * d:(i + 1) * d], H))
    return np.hstack(Ws), np.concatenate(bs)


def decode_pairs(X, dims):
    """Rows X (T, H d_head; HF's head order) as the cross cache's pairs: pair c's slot (j H + h) t + lane holds
    X[2c t + lane, h d_head + j] + i X[(2c + 1) t + lane, h d_head + j] (impl.cross_attention.fill_cache's input)."""
    from perseus.impl.layout import rearrange_qkv_weights
    t = dims.t
    Xr = rearrange_qkv_weights(np.asarray(X, dtype=np.float64), dims.H)       # columns j H + h
    G = -(-len(Xr) // t)
    G += G % 2
    pad = np.zeros((G * t, Xr.shape[1]))
    pad[:len(Xr)] = Xr
    grp = pad.reshape(G, t, -1).transpose(0, 2, 1).reshape(G, -1)             # group g: slot (j H + h) t + lane
    return [grp[2 * c] + 1j * grp[2 * c + 1] for c in range(G // 2)]


def cross_cache_ref(K, V, n, dims):
    """The filled cross cache in float (impl.cross_attention.fill_cache's slot values, cache_cts' order: the K buckets,
    then the V buckets) as (real, imaginary) pairs, NaN on the lanes of keys >= n (the encoder's padding positions,
    which the attention masks)."""
    t, dh, H = dims.t, dims.d_head, dims.H
    from perseus.impl.layout import rearrange_qkv_weights

    def groups(X):
        Xr = rearrange_qkv_weights(np.asarray(X, dtype=np.float64), H)
        G = -(-n // t)
        G += G % 2
        pad = np.full((G * t, Xr.shape[1]), np.nan)
        pad[:n] = Xr[:n]
        return pad.reshape(G, t, -1).transpose(0, 2, 1).reshape(G, -1)
    gk, gv = groups(K), groups(V)
    ng = -(-n // t)
    out = [(gk[2 * c], gk[2 * c + 1]) for c in range(len(gk) // 2)]
    blk = np.arange(dims.N) // dims.tH
    zero = np.zeros(dims.N)
    for p in range(dims.d_head_real // 2):
        re, im = np.full(dims.N, np.nan), np.full(dims.N, np.nan)
        for g in range(-1, ng):
            sel = blk == (g - 2 * p) % dh
            re[sel] = (gv[g] if g >= 0 else zero)[sel]
            im[sel] = (gv[g + 1] if g + 1 < ng else zero)[sel]
        out.append((re, im))
    return out
