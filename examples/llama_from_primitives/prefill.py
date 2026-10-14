"""Llama prefill on ``perseus.impl.prefill``: the prompt after BOS as complex chunk pairs, BOS K/V a public prefix."""
from __future__ import annotations

import os
import time

import numpy as np

from perseus.impl.activation import gelu
from perseus.impl.layout import cm_params
from perseus.impl.linear import EncodedLinear
from perseus.impl.norm import rms_norm_lanes
from perseus.impl.poly import im_cleanse
from perseus.impl.prefill import (name_inputs, gelu_joint, Chunks, decrypt_prompt, dense_linear, encode_prompt, join, norm_all, norm_pair,
                                  prefill_attention, split)

from perseus.impl import blocked as BL
from perseus.impl.prefill import evict_block_masks
from perseus.impl.pcmm import run as pcmm_run

from . import prefix as PX
from .model import LlamaModel, dist_report
from .weights import rotation_of


def real_linear(lin: EncodedLinear, name: str, part=None) -> EncodedLinear:
    """A real EncodedLinear from a slot cache entry: `part` "re" / "im" of a complex linear, or output-unpacked."""
    N = lin.pts.shape[1]
    if part is not None:
        pts = np.asarray(lin.pts.real if part == "re" else lin.pts.imag, dtype=np.float64)
    elif lin.outputpack:
        p = cm_params(N, lin.d_in, lin.d_out)
        rop = p.r_o // 2
        src = np.asarray(lin.pts)
        pts = np.empty((p.r_i * p.r_o, N))
        for j in range(p.r_i):
            blk = src[j * rop:(j + 1) * rop]
            pts[j * p.r_o:(j + 1) * p.r_o:2] = 2.0 * blk.real
            pts[j * p.r_o + 1:(j + 1) * p.r_o:2] = 2.0 * blk.imag
    else:
        pts = np.asarray(lin.pts, dtype=np.float64)
    return EncodedLinear(lin.d_in, lin.d_out, pts, None, name, False)


# PF_NORM_SHARE=1: one RMSNorm inverse-sqrt chain for every chunk (period <= the Llama sparse route's 256)
_NORM_SHARE = os.environ.get("PF_NORM_SHARE", "1") not in ("", "0")
_NORM_PERIOD = 256
# PF_LINEAR=pcmm: every linear a PCMM inside a refresh
_PCMM = os.environ.get("PF_LINEAR", "") == "pcmm"
# PF_GELU_JOINT=1: one refresh point for both halves between the SiLU's P1 and P2
_GELU_JOINT = os.environ.get("PF_GELU_JOINT", "0") not in ("", "0")


def glu_pair(rt, G, U, cfg, dense=False):
    """2 (silu(g) u) per chunk of G, U = (A + i B) / 2, in the MLP layout or (`dense`) the PCMM's token layout."""
    ops, d = rt.ops, rt.dims
    gA, gB = split(ops, G)
    uA, uB = split(ops, U)
    if dense:
        mask = ("pf.silu.dense", lambda s: np.full(d.N, s))
    else:
        from perseus.impl.layout import active_expanded_lanes
        mask = ("pf.silu.lanes", lambda s: active_expanded_lanes(d, s, d.tp_E))
    if _GELU_JOINT:
        aA, aB = gelu_joint(rt, gA, gB, cfg, mask)
        hA = im_cleanse(ops, ops.mult(aA, uA))
        hB = im_cleanse(ops, ops.mult(aB, uB))
    elif dense:
        hA = im_cleanse(ops, ops.mult(gelu(rt, gA, cfg, mask=mask), uA))
        hB = im_cleanse(ops, ops.mult(gelu(rt, gB, cfg, mask=mask), uB))
    else:       # captured op order: each half's product right after its SiLU
        hA = im_cleanse(ops, ops.mult(gelu(rt, gA, cfg, lanes=d.tp_E), uA))
        hB = im_cleanse(ops, ops.mult(gelu(rt, gB, cfg, lanes=d.tp_E), uB))
    return join(ops, hA, hB)


def llama_prefill_block(rt, Xs, w, cfgs, ch: Chunks, rope, prefix_kv):
    ops, d = rt.ops, rt.dims
    ln1, ln2, sm, act = cfgs
    tap = getattr(rt, "tap", None) or (lambda name, cts: None)
    skip = Xs
    with rt.step("ln_1"):
        Y = norm_all(rt, Xs, ln1, ch, rms=True, max_period=_NORM_PERIOD) if _NORM_SHARE else \
            [norm_pair(rt, X, ln1, ch, c, rms_norm_lanes) for c, X in enumerate(Xs)]
    with rt.step("qkv"):
        K, V, Q = dense_linear(rt, Y, [w.kv, w.q], ch, d.t, d.t, 0.5, 0.5)
    tap("q", Q); tap("k", K); tap("v", V)
    with rt.step("attn"):
        O = prefill_attention(rt, Q, K, V, sm, ch, rope=rope, prefix=prefix_kv)
    tap("attn", O)
    with rt.step("out_proj"):
        (A,) = dense_linear(rt, O, [w.out], ch, d.t, d.t, 1.0, 1.0)
    Xs = [ops.add(a, s) for a, s in zip(A, skip)]
    tap("res1", Xs)
    skip = Xs
    with rt.step("ln_2"):
        Y = norm_all(rt, Xs, ln2, ch, rms=True, max_period=_NORM_PERIOD) if _NORM_SHARE else \
            [norm_pair(rt, X, ln2, ch, c, rms_norm_lanes) for c, X in enumerate(Xs)]
    with rt.step("gate_up"):
        G, U = dense_linear(rt, Y, [w.gate, w.up], ch, d.t, d.tp_E, 0.5, 0.5)
    with rt.step("glu"):
        H = [glu_pair(rt, g, u, act) for g, u in zip(G, U)]
    with rt.step("down"):
        (Dn,) = dense_linear(rt, H, [w.down], ch, d.tp_E, d.t, 0.5, 1.0)
    return [ops.add(a, s) for a, s in zip(Dn, skip)]


def llama_prefill_block_pcmm(rt, Xs, P, cfgs, ch: Chunks, rope, prefix_kv):
    """llama_prefill_block with every linear a PCMM (P: LlamaPrefill.pcmm_weights), all in the token layout."""
    ops, d = rt.ops, rt.dims
    ln1, ln2, sm, act = cfgs
    tap = getattr(rt, "tap", None) or (lambda name, cts: None)
    skip = Xs
    with rt.step("ln_1"):
        Y = norm_all(rt, Xs, ln1, ch, rms=True, max_period=_NORM_PERIOD) if _NORM_SHARE else \
            [norm_pair(rt, X, ln1, ch, c, rms_norm_lanes) for c, X in enumerate(Xs)]
    with rt.step("qkv"):
        K, V, Q = [], [], []
        for y in Y:
            k, v, q = pcmm_run(rt, [y], P["qkv"])
            K.append(k); V.append(v); Q.append(q)
    tap("q", Q); tap("k", K); tap("v", V)
    with rt.step("attn"):
        O = prefill_attention(rt, Q, K, V, sm, ch, rope=rope, prefix=prefix_kv)
    tap("attn", O)
    with rt.step("out_proj"):
        A = [pcmm_run(rt, [o], P["out"])[0] for o in O]
    Xs = [ops.add(a, s) for a, s in zip(A, skip)]
    tap("res1", Xs)
    skip = Xs
    with rt.step("ln_2"):
        Y = norm_all(rt, Xs, ln2, ch, rms=True, max_period=_NORM_PERIOD) if _NORM_SHARE else \
            [norm_pair(rt, X, ln2, ch, c, rms_norm_lanes) for c, X in enumerate(Xs)]
    out = []
    for y in Y:
        with rt.step("gate_up"):
            GU = pcmm_run(rt, [y], P["gate_up"])
            nb = len(GU) // 2
        with rt.step("glu"):
            H = [glu_pair(rt, g, u, act, dense=True) for g, u in zip(GU[:nb], GU[nb:])]
        with rt.step("down"):
            out.append(pcmm_run(rt, H, P["down"])[0])
    return [ops.add(a, s) for a, s in zip(out, skip)]


def _memdbg(rt, tag):
    """PF_MEMDBG=1: print device free / pool reserved after a stage."""
    if not os.environ.get("PF_MEMDBG"):
        return
    core = rt.ops.core
    if hasattr(core, "device_free_gb"):
        rt.ops.device_sync()
        st = core.gpu_pool_stats() if hasattr(core, "gpu_pool_stats") else {}
        print(f"[memdbg] {tag}: free={core.device_free_gb():.2f}GB pool={st.get('reserved_gb', 0):.2f}GB", flush=True)


def llama_prefill_block_blocked(rt, Xs, P, cfgs, ch: Chunks, rope, prefixes, B, kv=None, layer=0):
    """llama_prefill_block_pcmm on the channel-blocked layout (Xs[g][k]; whole heads per block)."""
    ops, K = rt.ops, B.Kkv
    ln1, ln2, sm, act = cfgs
    _memdbg(rt, "in")
    skip = Xs
    with rt.step("ln_1"):
        Y = BL.norm_blocked(rt, Xs, ln1, ch, B, rms=True, max_period=_NORM_PERIOD)
    with rt.step("qkv"):
        Q, Kc, V = [], [], []
        for y in Y:
            o = pcmm_run(rt, y, P["qkv"])
            Kc.append(o[:K]); V.append(o[K:2 * K]); Q.append(o[2 * K:])
    _memdbg(rt, "qkv")
    with rt.step("attn"):
        O = BL.attention_blocked(rt, Q, Kc, V, sm, ch, B, rope=rope, prefixes=prefixes, kv=kv, layer=layer)
    _memdbg(rt, "attn")
    with rt.step("out_proj"):
        A = [pcmm_run(rt, o, P["out"]) for o in O]
    Xs = [[ops.add(a, s) for a, s in zip(ra, rs)] for ra, rs in zip(A, skip)]
    _memdbg(rt, "out_proj")
    skip = Xs
    with rt.step("ln_2"):
        Y = BL.norm_blocked(rt, Xs, ln2, ch, B, rms=True, max_period=_NORM_PERIOD)
    _memdbg(rt, "ln_2")
    out = []
    for y in Y:
        with rt.step("gate_up"):
            GU = pcmm_run(rt, y, P["gate_up"])
            nb = len(GU) // 2
        with rt.step("glu"):
            H = [glu_pair(rt, g, u, act, dense=True) for g, u in zip(GU[:nb], GU[nb:])]
        with rt.step("down"):
            out.append(pcmm_run(rt, H, P["down"]))
    _memdbg(rt, "mlp")
    return [[ops.add(a, s) for a, s in zip(ra, rs)] for ra, rs in zip(out, skip)]


def _layout_blocked():
    return os.environ.get("PF_LAYOUT", "") == "blocked"


class LlamaPrefill(LlamaModel):
    def __init__(self, inf, weights, cfgs, **kw):
        kw.setdefault("kv_offload", False)
        super().__init__(inf, weights, cfgs, **kw)

    def prefix_kv(self, b):
        """Block b's BOS key / value as slot vectors on every lane (RoPE at position 0 is the identity)."""
        d = self.rt.dims
        if not hasattr(self, "_bos"):
            self._bos = self.w.bos_kv(self.n_layers)
        k, v = self._bos[b]
        dh = d.dim // self.H
        kf = PX.expanded_heads(k, self.H, self.H_kv, dh)
        vf = PX.expanded_heads(v, self.H, self.H_kv, dh)
        if _layout_blocked():
            B = self.blocks
            kg, vg = B.kv_vec(kf), B.kv_vec(vf)
            return [(np.repeat(kg[k * B.nch:(k + 1) * B.nch], B.t), np.repeat(vg[k * B.nch:(k + 1) * B.nch], B.t))
                    for k in range(B.Kkv)]
        return np.repeat(kf, d.t), np.repeat(vf, d.t)

    @property
    def blocks(self):
        G = self.H // self.H_kv if os.environ.get("PF_GQA", "") == "1" else 1
        return BL.Blocks(self.rt.dims, int(os.environ.get("PF_NCH", "512")), G)

    def prefill(self, tokens, on_block=None, on_kv=None):
        """tokens[0] = BOS (public prefix), tokens[1:] the prompt -> the dense residual stream after the blocks.
        `on_kv(b, K, V)` (blocked GQA): block b's K[kv block][chunk] (post-RoPE) / V, after its planned stage."""
        d = self.rt.dims
        T = len(tokens) - 1
        blk_l = _layout_blocked()
        B = self.blocks if blk_l else None
        ch = Chunks(T, B.t if blk_l else d.t, base=1)
        self.token_begin(0)
        rows = np.stack([self.w.embedding(t) for t in tokens[1:]])
        if blk_l:
            Xs = BL.encode_blocked(self.rt, rows, ch, B)
            for b in range(self.n_layers):
                t0 = time.time()
                cap = self.enter_stage(b, 1)
                BL.name_blocked(self.inf, Xs)
                kvs = None if on_kv is None else [{} for _ in range(B.Kkv)]
                with self.rt.step(f"blk{b}"):
                    Xs = llama_prefill_block_blocked(self.rt, Xs, self.pcmm_weights(b), self.cfgs.block(b), ch,
                                                     self.rope, self.prefix_kv(b), B, kvs, layer=b)
                self.exit_stage(b, cap)
                if on_kv is not None:
                    on_kv(b, [x["K"] for x in kvs], [x["V"] for x in kvs])
                evict_block_masks(self.rt)
                if on_block is not None:
                    on_block(b, Xs, time.time() - t0)
            return Xs, ch, self.token_end(0)
        Xs = encode_prompt(self.rt, rows, ch)
        ops = self.rt.ops
        for b in range(self.n_layers):
            t0 = time.time()
            w = self.pcmm_weights(b) if _PCMM else self.block_weights(b)
            blk = llama_prefill_block_pcmm if _PCMM else llama_prefill_block
            cap = self.enter_stage(b, 1)
            name_inputs(self.inf, Xs)
            with self.rt.step(f"blk{b}"):
                Xs = blk(self.rt, Xs, w, self.cfgs.block(b), ch, self.rope, self.prefix_kv(b))
            self.exit_stage(b, cap)
            evict_block_masks(self.rt)
            if on_block is not None:
                on_block(b, Xs, time.time() - t0)
        return Xs, ch, self.token_end(0)

    def pcmm_weights(self, b):
        """Block b's PcmmLinears, scaled as the pair path's outputs (PF_PCMM_LLAMA_SCALES), cached next to the slots."""
        from perseus.impl.pcmm import PcmmLinear, band_shift, pinned_planes, weight_bits
        from .weights import block_matrices
        cache = self.__dict__.setdefault("_pcmm_w", {})
        if b in cache:
            return cache[b]
        d = self.rt.dims
        m = {}

        def mats():
            if not m:
                m["x"] = block_matrices(self.w.store, self.cfgs, b, d)
            return m["x"]

        wb = weight_bits()

        blk_l = _layout_blocked()
        B = self.blocks if blk_l else None
        t_enc = B.t if blk_l else d.t
        tag = f"blk{t_enc}." if blk_l else ""

        def enc(name, build, scale):
            # cached at scale 1: a power-of-two scale only moves w_scale
            tg = f"blk{t_enc}g{B.G}." if blk_l and B.G > 1 and name in ("qkv", "out") else tag
            f = self.w.dir / f"pcmm3.w{wb}.{tg}b{b}.{name}.npz"
            if not f.exists():
                lin = PcmmLinear.encode(build(), t_enc, 1.0, name=name, wbits=wb)
                tmp = self.w.dir / f".pcmm3.w{wb}.{tg}b{b}.{name}.{os.getpid()}.tmp.npz"
                np.savez(tmp, Wd=lin.Wd, ws=lin.w_scale, n=np.array([lin.n_in, lin.n_out]))
                os.replace(tmp, f)
            z = np.load(f)
            nm = f"b{b}.{name}"
            return PcmmLinear(int(z["n"][0]), int(z["n"][1]), pinned_planes(self.core, z["Wd"]), float(z["ws"]) / scale,
                              band_shift(nm), None, nm, t_enc if blk_l else 0)

        sc = [float(x) for x in os.environ.get("PF_PCMM_LLAMA_SCALES", "0.25,1,0.25,0.5").split(",")]
        if blk_l:
            Er = d.E_real
            cache[b] = {"qkv": enc("qkv", lambda: np.hstack([B.kv_cols(mats().K), B.kv_cols(mats().V),
                                                             B.group_cols(mats().Q)]), sc[0]),
                        "out": enc("out", lambda: B.group_rows(mats().O), sc[1]),
                        "gate_up": enc("gate_up", lambda: np.hstack([mats().Gate[:, :Er], mats().Up[:, :Er]]), sc[2]),
                        "down": enc("down", lambda: mats().Down[:Er], sc[3])}
            return cache[b]
        cache[b] = {"qkv": enc("qkv", lambda: np.hstack([mats().K, mats().V, mats().Q]), sc[0]),
                    "out": enc("out", lambda: mats().O, sc[1]),
                    "gate_up": enc("gate_up", lambda: np.hstack([mats().Gate, mats().Up]), sc[2]),
                    "down": enc("down", lambda: mats().Down, sc[3])}
        return cache[b]

    def hidden_rows(self, Xs, ch):
        if _layout_blocked():
            return BL.decrypt_blocked(self.rt, Xs, ch, self.blocks, self.rt.dims.dim)
        return decrypt_prompt(self.rt, Xs, ch, self.rt.dims.dim)


def plain_logits(model, hidden):
    """The final RMSNorm + LM head in float64 on decrypted hidden rows (T, dim), the residual rotation undone."""
    st = model.w.store
    R = rotation_of(model.cfgs, model.rt.dims.dim)
    h = np.asarray(hidden, dtype=np.float64)
    if R is not None:
        h = h @ R.T
    eps = float(st.config["rms_norm_eps"])
    n = h / np.sqrt((h * h).mean(axis=1, keepdims=True) + eps) * st.tensor("model.norm.weight").astype(np.float64)
    return n @ st.tensor("lm_head.weight").astype(np.float32).T.astype(np.float64)


def exact_hidden(model, tokens, n_layers):
    """Float64 Llama blocks on the prompt (BOS included): per-block hidden rows of positions 1.., rotated basis."""
    st = model.w.store
    c = st.config
    H, H_kv = int(c["num_attention_heads"]), int(c["num_key_value_heads"])
    D = int(c["hidden_size"]); dh = D // H
    eps = float(c["rms_norm_eps"])
    inv = np.asarray(model.cfgs.model["rope_inv_freq"], dtype=np.float64)
    t = lambda n: st.tensor(n).astype(np.float64)
    x = t("model.embed_tokens.weight")[np.asarray(tokens)]
    T = x.shape[0]
    pos = np.arange(T)[:, None] * inv[None, :]
    cos, sin = np.cos(np.concatenate([pos, pos], 1)), np.sin(np.concatenate([pos, pos], 1))

    def rope(a):                       # (T, heads, dh)
        h = dh // 2
        rot = np.concatenate([-a[..., h:], a[..., :h]], -1)
        return a * cos[:, None, :] + rot * sin[:, None, :]
    rms = lambda a, g: a / np.sqrt((a * a).mean(1, keepdims=True) + eps) * g
    R = rotation_of(model.cfgs, D)
    out = []
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for b in range(n_layers):
        p = f"model.layers.{b}."
        n = rms(x, t(p + "input_layernorm.weight"))
        q = rope((n @ t(p + "self_attn.q_proj.weight").T).reshape(T, H, dh))
        k = rope((n @ t(p + "self_attn.k_proj.weight").T).reshape(T, H_kv, dh))
        v = (n @ t(p + "self_attn.v_proj.weight").T).reshape(T, H_kv, dh)
        o = np.zeros((T, H, dh))
        for hh in range(H):
            kh = hh // (H // H_kv)
            s = q[:, hh] @ k[:, kh].T / np.sqrt(dh) + mask
            pr = np.exp(s - s.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
            o[:, hh] = pr @ v[:, kh]
        x = x + o.reshape(T, D) @ t(p + "self_attn.o_proj.weight").T
        n = rms(x, t(p + "post_attention_layernorm.weight"))
        g = n @ t(p + "mlp.gate_proj.weight").T
        x = x + (g / (1 + np.exp(-g)) * (n @ t(p + "mlp.up_proj.weight").T)) @ t(p + "mlp.down_proj.weight").T
        out.append((x if R is None else x @ R)[1:].copy())
    return out


def report(logits, gt):
    rows = [dist_report(lg, g) for lg, g in zip(logits, gt)]
    kls = sorted(r["kl"] for r in rows)
    return rows, {"top1": sum(r["hit"] for r in rows), "n": len(rows), "medianKL": kls[len(kls) // 2],
                  "maxKL": kls[-1]}


def block0_taps(model, tokens):
    """Float64 block-0 tap values in the prefill layouts, positions 1.. (q / k / v pre-RoPE at 1/2)."""
    from perseus.impl.layout import rearrange_qkv_biases
    st = model.w.store
    c = st.config
    H, H_kv = int(c["num_attention_heads"]), int(c["num_key_value_heads"])
    D = int(c["hidden_size"]); dh = D // H
    eps = float(c["rms_norm_eps"])
    inv = np.asarray(model.cfgs.model["rope_inv_freq"], dtype=np.float64)
    t = lambda n: st.tensor(n).astype(np.float64)
    x = t("model.embed_tokens.weight")[np.asarray(tokens)]
    T = x.shape[0]
    p = "model.layers.0."
    n = x / np.sqrt((x * x).mean(1, keepdims=True) + eps) * t(p + "input_layernorm.weight")
    q = n @ t(p + "self_attn.q_proj.weight").T
    k = n @ t(p + "self_attn.k_proj.weight").T
    v = n @ t(p + "self_attn.v_proj.weight").T
    exp = lambda a: np.repeat(a.reshape(T, H_kv, dh), H // H_kv, axis=1).reshape(T, D)
    k, v = exp(k), exp(v)
    pos = np.arange(T)[:, None] * inv[None, :]
    cos, sin = np.cos(np.concatenate([pos, pos], 1)), np.sin(np.concatenate([pos, pos], 1))

    def rope(a):
        a = a.reshape(T, H, dh)
        h2 = dh // 2
        return (a * cos[:, None] + np.concatenate([-a[..., h2:], a[..., :h2]], -1) * sin[:, None])
    qr, kr = rope(q), rope(k)
    o = np.zeros((T, H, dh))
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for hh in range(H):
        s = qr[:, hh] @ kr[:, hh].T / np.sqrt(dh) + mask
        pr = np.exp(s - s.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        o[:, hh] = pr @ v.reshape(T, H, dh)[:, hh]
    res1 = x + o.reshape(T, D) @ t(p + "self_attn.o_proj.weight").T
    R = rotation_of(model.cfgs, D)
    if R is not None:
        res1 = res1 @ R
    inter = lambda a: np.stack([rearrange_qkv_biases(r, H) for r in a])
    return {"q": 0.5 * inter(q)[1:], "k": 0.5 * inter(k)[1:], "v": 0.5 * inter(v)[1:],
            "attn": inter(o.reshape(T, D))[1:], "res1": res1[1:]}
