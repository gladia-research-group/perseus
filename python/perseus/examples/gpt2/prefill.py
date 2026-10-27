"""GPT-2 prefill on ``perseus.impl.prefill``: the prompt as complex chunk pairs, transformer_block's dataflow."""
from __future__ import annotations

import copy
import dataclasses
import os
import time
from pathlib import Path

import numpy as np

from perseus.impl.activation import gelu
from perseus.impl.norm import norm_lanes
from perseus.impl.poly import im_cleanse
from perseus.impl.prefill import (name_inputs, gelu_joint, Chunks, DiagLinear, Pair, decrypt_prompt, dense_linear, diag_apply, diag_bias,
                                  encode_prompt, gelu_pair, join, lockstep_rt, norm_all, norm_pair, prefill_attention,
                                  shift_lanes, split)

from perseus.impl import blocked as BL
from perseus.impl.prefill import evict_block_masks
from perseus.impl.pcmm import PcmmLinear, band_shift, run as pcmm_run

from .head import final_ln, lm_head
from .model import Gpt2Model, dist_report
from . import weights as W


def prefill_block(rt, Xs, w, cfgs, ch: Chunks):
    """One block on the dense complex residual stream Xs (scale 1)."""
    ops, d = rt.ops, rt.dims
    ln1, ln2, sm, ge = cfgs
    tap = getattr(rt, "tap", None) or (lambda name, cts: None)
    skip = Xs
    with rt.step("ln_1"):
        Y = [shift_lanes(rt, norm_pair(rt, X, ln1, ch, c, norm_lanes), w.shift1, ch, c, d.dim, 2.0)
             for c, X in enumerate(Xs)]                                  # 2 (LN + shift)
    tap("ln_1", Y)
    with rt.step("qkv"):
        K, V, Q = dense_linear(rt, Y, [w.k, w.v, w.q], ch, d.t, d.t, 0.5, 0.5)
    tap("q", Q); tap("k", K); tap("v", V)
    with rt.step("attn"):
        O = prefill_attention(rt, Q, K, V, sm, ch)
    tap("attn", O)
    with rt.step("out_proj"):
        (A,) = dense_linear(rt, O, [w.out], ch, d.t, d.t, 1.0, 1.0)
    Xs = [ops.add(a, s) for a, s in zip(A, skip)]
    tap("res1", Xs)
    skip = Xs
    with rt.step("ln_2"):
        Y = [shift_lanes(rt, norm_pair(rt, X, ln2, ch, c, norm_lanes), w.shift2, ch, c, d.dim, 2.0)
             for c, X in enumerate(Xs)]
    with rt.step("up"):
        (H,) = dense_linear(rt, Y, [w.up], ch, d.t, d.tp_E, 0.5, 0.5)
    tap("up", H)
    with rt.step("gelu"):
        G = [gelu_pair(rt, h, ge, gelu) for h in H]                      # 2 gelu
    tap("gelu", G)
    with rt.step("down"):
        (Dn,) = dense_linear(rt, G, [w.down], ch, d.tp_E, d.t, 0.5, 1.0)
    return [ops.add(a, s) for a, s in zip(Dn, skip)]


# PF_LOCKSTEP=1: the GELU's two halves in lockstep, one refresh per pair
_LOCKSTEP = os.environ.get("PF_LOCKSTEP", "0") not in ("", "0")
# PF_NORM_SHARE=1: one LayerNorm inverse-sqrt chain for every chunk (prefill.norm_all)
_NORM_SHARE = os.environ.get("PF_NORM_SHARE", "1") not in ("", "0")
# PF_GELU_JOINT=1: one refresh point for both halves between the GELU's P1 and P2 (a THOR block; a Chebyshev
# block's single series has none)
_GELU_JOINT = os.environ.get("PF_GELU_JOINT", "0") not in ("", "0")


# PF_LAST_QUERY=1 (with the tail): the last block computes the last position alone past its K / V (the tail reads
# nothing else)
_LAST_QUERY = os.environ.get("PF_LAST_QUERY", "0") not in ("", "0")


# PF_QKV_PACK=1: the last head group's K and V share one PCMM output when its real heads fill at most half of it
# (GPT-2: heads 8-11 in a group of 8): V's heads ride K's padding heads, a rotation by half a group brings them home
_QKV_PACK = os.environ.get("PF_QKV_PACK", "0") not in ("", "0")


def _kv_packed(B):
    return _QKV_PACK and os.environ.get("PF_ATTN", "") != "ccmm" and 2 * B.group_dims(B.K - 1).H_real <= B.hg


def _pack_kv(B, k_cols, v_cols, k_vec, v_vec):
    """K's and V's last channel blocks as one: V's heads half a group up, on K's (zero) padding heads."""
    n, h2 = B.nch, B.hg // 2
    lo = (B.K - 1) * n
    r = np.arange(n)
    src = lo + r[r % B.hg < h2]
    k_cols, k_vec = k_cols.copy(), k_vec.copy()
    k_cols[:, src + h2] += v_cols[:, src]
    k_vec[src + h2] += v_vec[src]
    return k_cols, v_cols[:, :lo], k_vec, v_vec[:lo]


def qkv_encode(B, m):
    """The fused QKV linear's columns and bias in head-grouped order, K | V | Q (PF_QKV_PACK: V's last block on K's)."""
    k_cols, v_cols = B.group_cols(m.K), B.group_cols(m.V)
    k_vec, v_vec = B.group_vec(m.bk), B.group_vec(m.bv)
    if _kv_packed(B):
        k_cols, v_cols, k_vec, v_vec = _pack_kv(B, k_cols, v_cols, k_vec, v_vec)
    return np.hstack([k_cols, v_cols, B.group_cols(m.Q)]), np.concatenate([k_vec, v_vec, B.group_vec(m.bq)])


def qkv_split(rt, o, B):
    """(K, V, Q) blocks of the fused QKV linear's output (PF_QKV_PACK: V's last block rides K's, half a group up)."""
    K = B.K
    if _kv_packed(B):
        o = o[:2 * K - 1] + [rt.ops.rotate(o[K - 1], B.hg // 2 * B.t)] + o[2 * K - 1:]
    return o[:K], o[K:2 * K], o[2 * K:]


# PF_PCMM_CALIB=<json>: record every block linear's max |output| (the PCMM band shifts)
_CALIB_PATH = os.environ.get("PF_PCMM_CALIB", "")
_calib: dict[str, float] = {}


def _dump_calib():
    import json
    with open(_CALIB_PATH, "w") as f:
        json.dump(_calib, f, indent=1, sort_keys=True)


def lin_apply(rt, xs, lin):
    """Bias-free output blocks of one linear on dense blocks xs: PCMM or diagonal BSGS."""
    if isinstance(lin, PcmmLinear):
        return pcmm_run(rt, xs, lin)
    (blocks,) = diag_apply(rt, xs, [lin])
    if _CALIB_PATH and lin.name:
        m = max(float(np.abs(rt.ops.core.decrypt_slots_complex(rt.ops.inf, y)).max()) for y in blocks)
        _calib[lin.name] = max(_calib.get(lin.name, 0.0), m)
        _dump_calib()
    return blocks


def prefill_block_diag(rt, Xs, w, cfgs, ch: Chunks):
    """prefill_block with the dense diagonal linears (w: diag_weights)."""
    ops, d = rt.ops, rt.dims
    ln1, ln2, sm, ge = cfgs
    tap = getattr(rt, "tap", None) or (lambda name, cts: None)
    skip = Xs
    with rt.step("ln_1"):
        N1 = norm_all(rt, Xs, ln1, ch) if _NORM_SHARE else [norm_pair(rt, X, ln1, ch, c, norm_lanes)
                                                             for c, X in enumerate(Xs)]
        Y = [shift_lanes(rt, n, w["shift1"], ch, c, d.dim, 2.0) for c, n in enumerate(N1)]
    with rt.step("qkv"):
        K, V, Q = [], [], []
        for c, y in enumerate(Y):
            blocks = lin_apply(rt, [y], w["qkv"])
            k, v, q = (diag_bias(rt, blk, w["qkv"], m, ch, c) for m, blk in enumerate(blocks))
            K.append(k); V.append(v); Q.append(q)
    tap("q", Q); tap("k", K); tap("v", V)
    with rt.step("attn"):
        O = prefill_attention(rt, Q, K, V, sm, ch)
    tap("attn", O)
    with rt.step("out_proj"):
        A = [diag_bias(rt, lin_apply(rt, [o], w["out"])[0], w["out"], 0, ch, c) for c, o in enumerate(O)]
    Xs = [ops.add(a, s) for a, s in zip(A, skip)]
    tap("res1", Xs)
    skip = Xs
    with rt.step("ln_2"):
        N2 = norm_all(rt, Xs, ln2, ch) if _NORM_SHARE else [norm_pair(rt, X, ln2, ch, c, norm_lanes)
                                                             for c, X in enumerate(Xs)]
        Y = [shift_lanes(rt, n, w["shift2"], ch, c, d.dim, 2.0) for c, n in enumerate(N2)]
    out = []
    for c, y in enumerate(Y):
        with rt.step("up"):
            H = lin_apply(rt, [y], w["up"])
            H = [diag_bias(rt, h, w["up"], m, ch, c) for m, h in enumerate(H)]
        with rt.step("gelu"):
            G = []
            mask = ("pf.gelu.dense", lambda s: np.full(d.N, s))
            for h in H:
                a, b = split(ops, h)
                if _GELU_JOINT and ge.method == "thor_composite":
                    ga, gb = (im_cleanse(ops, g) for g in gelu_joint(rt, a, b, ge, mask))
                elif _LOCKSTEP:
                    g = gelu(lockstep_rt(rt), Pair(a, b), ge, mask=mask)
                    ga, gb = im_cleanse(ops, g.a), im_cleanse(ops, g.b)
                else:
                    ga = im_cleanse(ops, gelu(rt, a, ge, mask=mask))
                    gb = im_cleanse(ops, gelu(rt, b, ge, mask=mask))
                G.append(join(ops, ga, gb))
        with rt.step("down"):
            Dn = lin_apply(rt, G, w["down"])
            out.append(diag_bias(rt, Dn[0], w["down"], 0, ch, c))
    return [ops.add(a, s) for a, s in zip(out, skip)]


def _bias_blocks(rt, ys, lin, ch: Chunks, g: int, B):
    sub = BL.sub_rt(rt, B, 0)
    return [diag_bias(sub, y, lin, m, ch, g) for m, y in enumerate(ys)]


def _gelu_blocks(rt, H, ge):
    ops, d = rt.ops, rt.dims
    mask = ("pf.gelu.dense", lambda s: np.full(d.N, s))
    out = []
    for h in H:
        a, b = split(ops, h)
        if _GELU_JOINT and ge.method == "thor_composite":
            ga, gb = (im_cleanse(ops, g) for g in gelu_joint(rt, a, b, ge, mask))
        else:
            ga = im_cleanse(ops, gelu(rt, a, ge, mask=mask))
            gb = im_cleanse(ops, gelu(rt, b, ge, mask=mask))
        out.append(join(ops, ga, gb))
    return out


def prefill_block_blocked(rt, Xs, w, cfgs, ch: Chunks, B, last=None):
    """prefill_block_diag on the channel-blocked layout (Xs[g][k]; PCMM linears in head-grouped order). `last`: the
    only position the block's output must hold (every position's K / V is still computed): its attention alone
    (blocked.attention_last), then it fills every lane of the residual stream."""
    ops = rt.ops
    ln1, ln2, sm, ge = cfgs
    skip = Xs
    with rt.step("ln_1"):
        N1 = BL.norm_blocked(rt, Xs, ln1, ch, B)
        Y = [[BL.shift_blocked(rt, n, w["shift1"], ch, g, k, B, 2.0) for k, n in enumerate(row)]
             for g, row in enumerate(N1)]
    with rt.step("qkv"):
        Q, Kc, V = [], [], []
        for g, y in enumerate(Y):
            k, v, q = qkv_split(rt, _bias_blocks(rt, pcmm_run(rt, y, w["qkv"]), w["qkv"], ch, g, B), B)
            Kc.append(k); V.append(v); Q.append(q)
    with rt.step("attn"):
        O = (BL.attention_blocked(rt, Q, Kc, V, sm, ch, B) if last is None
             else BL.attention_last(rt, Q, Kc, V, sm, ch, B, last))
        if last is not None:
            skip = [BL.token_bcast(rt, row, ch, B, last) for row in skip]
    with rt.step("out_proj"):
        A = [_bias_blocks(rt, pcmm_run(rt, o, w["out"]), w["out"], ch, g, B) for g, o in enumerate(O)]
    Xs = [[ops.add(a, s) for a, s in zip(ra, rs)] for ra, rs in zip(A, skip)]
    skip = Xs
    with rt.step("ln_2"):
        N2 = BL.norm_blocked(rt, Xs, ln2, ch, B)
        Y = [[BL.shift_blocked(rt, n, w["shift2"], ch, g, k, B, 2.0) for k, n in enumerate(row)]
             for g, row in enumerate(N2)]
    out = []
    for g, y in enumerate(Y):
        with rt.step("up"):
            H = _bias_blocks(rt, pcmm_run(rt, y, w["up"]), w["up"], ch, g, B)
        with rt.step("gelu"):
            G = _gelu_blocks(rt, H, ge)
        with rt.step("down"):
            out.append(_bias_blocks(rt, pcmm_run(rt, G, w["down"]), w["down"], ch, g, B))
    return [[ops.add(a, s) for a, s in zip(ra, rs)] for ra, rs in zip(out, skip)]


def _layout_blocked():
    return os.environ.get("PF_LAYOUT", "") == "blocked"


class Gpt2Prefill(Gpt2Model):
    """Gpt2Model with the prefill block; real linears (the complex payload carries the second chunk)."""

    def __init__(self, inf, store, cfgs, **kw):
        kw.setdefault("packing", "cachemir")
        kw.setdefault("kv_offload", False)
        super().__init__(inf, store, cfgs, **kw)

    def stage_caches(self, i):
        return None

    def prefill_weights(self, b):
        """block_weights(b) from pinned coefficient rows (cached on disk); PF_STREAM=0 keeps named plaintexts."""
        w = self.block_weights(b)
        if os.environ.get("PF_STREAM", "1") in ("", "0") or not hasattr(self.core, "pinned_empty"):
            return w
        cache = self.__dict__.setdefault("_pf_stream", {})
        if b not in cache:
            from perseus.impl.layout import coeff_rows
            root = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "perseus" / \
                "gpt2_coeff" / _coeff_key(self, b)
            root.mkdir(parents=True, exist_ok=True)
            sw = copy.copy(w)
            for f in ("q", "k", "v", "out", "up", "down"):
                lin = getattr(w, f)
                pts = np.ascontiguousarray(lin.pts)
                fr, ft = root / f"{f}.coeff.npy", root / f"{f}.tags.npy"
                if not (fr.exists() and ft.exists()):
                    for path, arr in ((fr, coeff_rows(pts)), (ft, self.core.pt_tag_fields(pts))):
                        tmp = root / f".{path.stem}.{os.getpid()}.tmp.npy"
                        np.save(tmp, arr)
                        os.replace(tmp, path)
                src = np.load(fr, mmap_mode="r")
                rows = self.core.pinned_empty(*src.shape)
                rows[:] = src
                setattr(sw, f, dataclasses.replace(lin, name=None, coeffs=rows, tags=np.load(ft)))
            cache[b] = sw
        return cache[b]

    def diag_weights(self, b):
        """The block's dense diagonal (or PCMM) linears: qkv, out, up, down, plus the LayerNorm shifts."""
        cache = self.__dict__.setdefault("_pf_diag", {})
        if b not in cache:
            d = self.rt.dims
            ge = self.cfgs.block(b)[3]
            fold = self.rt.ops.gelu_fold and ge.method in ("thor_composite", "chebyshev")
            us = 1.0 / ge.xmax if fold else 1.0
            m = W.block_matrices(self.store, self.cfgs, b, d, self.fold[0], self.fold[1])
            pinned = None if _use_pcmm(self) else getattr(self.core, "pinned_empty", None)
            root = None
            if pinned is not None:
                root = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "perseus" / \
                    "gpt2_diag" / _coeff_key(self, b)
                root.mkdir(parents=True, exist_ok=True)
            enc = lambda name, Wm, sc, bias: DiagLinear.encode(Wm, d.N, d.t, sc, bias, name=f"b{b}.{name}",
                                                               pinned=pinned, core=self.core,
                                                               cache=None if root is None else str(root / name))
            if _use_pcmm(self):
                enc = lambda name, Wm, sc, bias: PcmmLinear.encode(Wm, d.t, sc, bias, shift=band_shift(f"b{b}.{name}"),
                                                                   name=f"b{b}.{name}")
            Er = d.E_real
            if _layout_blocked():
                B = self.blocks
                enc = lambda name, Wm, sc, bias: PcmmLinear.encode(Wm, B.t, sc, bias, shift=band_shift(f"b{b}.{name}"),
                                                                   name=f"b{b}.{name}")
                qkv_cols, qkv_bias = qkv_encode(B, m)
                cache[b] = {
                    "qkv": enc("qkv", qkv_cols, 0.25, 2.0 * qkv_bias),
                    "out": enc("out", B.group_rows(m.O), 1.0, m.bo),
                    "up": enc("up", (m.Up * us)[:, :Er], 0.25, 2.0 * (m.bup * us)[:Er]),
                    "down": enc("down", m.Down[:Er], 0.5, 2.0 * m.bdown),
                    "shift1": self.block_weights(b).shift1, "shift2": self.block_weights(b).shift2}
                return cache[b]
            cache[b] = {
                "qkv": enc("qkv", np.hstack([m.K, m.V, m.Q]), 0.25, 2.0 * np.concatenate([m.bk, m.bv, m.bq])),
                "out": enc("out", m.O, 1.0, m.bo),
                "up": enc("up", (m.Up * us)[:, :Er], 0.25, 2.0 * (m.bup * us)[:Er]),
                "down": enc("down", m.Down[:Er], 0.5, 2.0 * m.bdown),
                "shift1": self.block_weights(b).shift1, "shift2": self.block_weights(b).shift2}
        return cache[b]

    def prefill_stages(self, ch: Chunks):
        diag = os.environ.get("PF_LINEAR", "pairs") in ("diag", "pcmm")

        def block_fn(b):
            def compute(Xs):
                cap = self.enter_stage(b, 0)
                name_inputs(self.inf, Xs)
                with self.rt.step(f"blk{b}"):
                    if _layout_blocked():
                        B = self.blocks
                        Xg = [Xs[g * B.K:(g + 1) * B.K] for g in range(len(Xs) // B.K)]
                        tail = getattr(self, "with_tail", False)
                        last = ch.T - 1 if _LAST_QUERY and tail and b == self.n_layers - 1 else None
                        Xg = prefill_block_blocked(self.rt, Xg, self.diag_weights(b), self.cfgs.block(b), ch, B, last)
                        Xs = [x for row in Xg for x in row]
                    elif diag:
                        Xs = prefill_block_diag(self.rt, Xs, self.diag_weights(b), self.cfgs.block(b), ch)
                    else:
                        Xs = prefill_block(self.rt, Xs, self.prefill_weights(b), self.cfgs.block(b), ch)
                self.exit_stage(b, cap)
                evict_block_masks(self.rt)
                return Xs
            return compute
        stages = [(f"blk{b}", f"b{b}.", block_fn(b)) for b in range(self.n_layers)]
        return stages + self.tail_stages(ch) if getattr(self, "with_tail", False) else stages

    def tail_stages(self, ch: Chunks, p=None):
        """Prefill-128's encrypted tail as two more stages (with_tail): stage n_layers moves prompt position p (default
        the last) out of the blocked layout into the decode's (blocked.token_of), then ln_f and the LM head as the
        decode's tail runs them; stage n_layers + 1 is the decode's CutMax on those tiles."""
        p = ch.T - 1 if p is None else p
        B, n = self.blocks, self.n_layers

        def tail(Xs):
            cap = self.enter_stage(n, p)
            with self.rt.step("tail"):
                x = BL.token_of(self.rt, [Xs[g * B.K:(g + 1) * B.K] for g in range(len(Xs) // B.K)], ch, B, p)
                h = final_ln(self.rt, x, self.cfgs.ln_f, self.lnf_params(), p, self.lmhead_cap)
                tiles = lm_head(self.rt, h, self.lm_tiles_paired())
            self.exit_stage(n, cap)
            self.tail_tiles = tiles               # the logits, read before CutMax (the quality check)
            return tiles

        return [("tail", None, tail), ("cutmax", None, lambda tiles: self.argmax_encrypted(tiles, p))]

    def lm_tiles_paired(self):
        """The decode's paired complex LM-head tile (cachemir_complex) whatever this model's packing: the tail's
        CutMax then runs on one tile, as the decode's does."""
        if getattr(self, "_lm_paired", None) is None:
            Wlm = W.lm_head_matrix(self.store, self.cfgs.ln_f, self.rt.dims, self.vocab, self.fold[2])
            self._lm_paired = W.lm_head_tiles(Wlm, self.rt.dims, self.vocab, self.W_tile, paired=True)
        return self._lm_paired

    def preload_prefill(self):
        """Build every block's prefill weights before the clock."""
        diag = os.environ.get("PF_LINEAR", "pairs") in ("diag", "pcmm")
        for b in range(self.n_layers):
            self.diag_weights(b) if diag else self.prefill_weights(b)

    def prefill(self, rows, on_block=None):
        """Embedding rows (T, 768) -> the dense residual stream after the last block."""
        blk = _layout_blocked()
        ch = Chunks(len(rows), self.blocks.t if blk else self.rt.dims.t)
        self.start()
        self.token_begin(0)
        if blk:
            Xs = [x for row in BL.encode_blocked(self.rt, rows, ch, self.blocks) for x in row]
        else:
            Xs = encode_prompt(self.rt, rows, ch)
        stages = self.prefill_stages(ch)
        if on_block is not None:
            stages = [(lab, pre, _tapped(fn, b, on_block)) for b, (lab, pre, fn) in enumerate(stages)]
        Xs = self.run_stages(Xs, 0, stages)
        rec = self.token_end(0)
        return Xs, ch, rec

    @property
    def blocks(self):
        return BL.Blocks(self.rt.dims, int(os.environ.get("PF_NCH", "512")))

    def hidden(self, Xs, ch):
        if _layout_blocked():
            B = self.blocks
            return BL.decrypt_blocked(self.rt, [Xs[g * B.K:(g + 1) * B.K] for g in range(len(Xs) // B.K)], ch, B,
                                      self.rt.dims.dim)
        return decrypt_prompt(self.rt, Xs, ch, self.rt.dims.dim)


def _use_pcmm(model):
    """PF_LINEAR=pcmm."""
    return os.environ.get("PF_LINEAR", "") == "pcmm"


def _coeff_key(model, b):
    """Coefficient-row cache key of block b."""
    import hashlib
    st = model.store._zip.filename
    ge = model.cfgs.block(b)[3]
    fold = model.rt.ops.gelu_fold and ge.method in ("thor_composite", "chebyshev")
    h = hashlib.sha256(f"{os.path.abspath(st)}|{os.path.getmtime(st)}|{model.cfgs.block(b)}|{model.fold}|"
                       f"{fold}|{model.complex_packing}|{model.rt.dims}".encode()).hexdigest()[:20]
    return f"{h}_b{b}"


def _tapped(fn, b, cb):
    def compute(x):
        t0 = time.time()
        x = fn(x)
        cb(b, x, time.time() - t0)
        return x
    return compute


def plain_logits(store, hidden):
    """ln_f + the tied LM head in float64 on decrypted hidden rows (T, 768)."""
    g = store.tensor(W.LNF + ".weight"); b = store.tensor(W.LNF + ".bias")
    h = np.asarray(hidden, dtype=np.float64)
    mu = h.mean(axis=1, keepdims=True)
    var = ((h - mu) ** 2).mean(axis=1, keepdims=True)
    n = (h - mu) / np.sqrt(var + 1e-5) * g + b
    return n @ store.tensor(W.WTE).T


def report(logits, gt_logits):
    """Per-position dist_report rows and the summary (top1 hits, median / max KL)."""
    rows = [dist_report(lg, gt) for lg, gt in zip(logits, gt_logits)]
    kls = sorted(r["kl"] for r in rows)
    return rows, {"top1": sum(r["hit"] for r in rows), "n": len(rows),
                  "medianKL": kls[len(kls) // 2], "maxKL": kls[-1]}


def exact_blocks(store, rows, n_layers=12):
    """Causal HF GPT-2 blocks in float64 on embedding rows (T, 768): the per-block hidden states."""
    x = np.asarray(rows, dtype=np.float64)
    T = x.shape[0]

    def ln(h, g, b):
        mu = h.mean(axis=1, keepdims=True)
        return (h - mu) / np.sqrt(((h - mu) ** 2).mean(axis=1, keepdims=True) + 1e-5) * g + b

    out = []
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for blk in range(n_layers):
        n = W.block_names(blk)
        t = store.tensor
        h = ln(x, t(n["ln1_w"]), t(n["ln1_b"]))
        qkv = h @ t(n["qkv_w"]).T + t(n["qkv_b"])
        q, k, v = np.split(qkv, 3, axis=1)
        H, dh = 12, 64
        o = np.zeros_like(q)
        for hh in range(H):
            sl = slice(hh * dh, (hh + 1) * dh)
            s = q[:, sl] @ k[:, sl].T / np.sqrt(dh) + mask
            p = np.exp(s - s.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
            o[:, sl] = p @ v[:, sl]
        x = x + o @ t(n["out_w"]).T + t(n["out_b"])
        h = ln(x, t(n["ln2_w"]), t(n["ln2_b"]))
        u = h @ t(n["up_w"]).T + t(n["up_b"])
        u = 0.5 * u * (1 + np.tanh(np.sqrt(2 / np.pi) * (u + 0.044715 * u ** 3)))
        x = x + u @ t(n["down_w"]).T + t(n["down_b"])
        out.append(x.copy())
    return out


def block0_taps(store, rows, d):
    """Float64 block-0 tap values in the prefill layouts (q / k / v at 1/2, head-interleaved); (T, features) each."""
    T = rows.shape[0]
    t = store.tensor; n = W.block_names(0)

    def ln(h, g, b):
        mu = h.mean(axis=1, keepdims=True)
        return (h - mu) / np.sqrt(((h - mu) ** 2).mean(axis=1, keepdims=True) + 1e-5) * g + b
    h = ln(rows, t(n["ln1_w"]), t(n["ln1_b"]))
    q, k, v = np.split(h @ t(n["qkv_w"]).T + t(n["qkv_b"]), 3, axis=1)
    o = np.zeros_like(q)
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for hh in range(d.H_real):
        sl = slice(hh * d.d_head, (hh + 1) * d.d_head)
        s = q[:, sl] @ k[:, sl].T / np.sqrt(d.d_head) + mask
        p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        o[:, sl] = p @ v[:, sl]

    def inter(a):
        out = np.zeros((T, d.hid))
        for hh in range(d.H_real):
            for j in range(d.d_head):
                out[:, j * d.H + hh] = a[:, hh * d.d_head + j]
        return out
    return {"q": 0.5 * inter(q), "k": 0.5 * inter(k), "v": 0.5 * inter(v), "attn": inter(o),
            "res1": rows + o @ t(n["out_w"]).T + t(n["out_b"])}


def tap_rows(rt, cts, ch: Chunks, n_feat):
    """(T, n_feat) rows of dense complex hid-layout ciphertexts (decrypted)."""
    d = rt.dims
    out = np.zeros((ch.T, n_feat))
    idx = np.arange(n_feat) * d.t
    for c, X in enumerate(cts):
        v = rt.ops.decrypt_slots_complex(X)
        for half, part in ((0, v.real), (1, v.imag)):
            for lane, p in enumerate(ch.idx(2 * c + half)):
                out[p] = part[idx + lane]
    return out
