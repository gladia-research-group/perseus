"""Llama on ``perseus.impl``: streamed weights + configs + per-block K/V caches over one session,
the BOS prefix, the decode step (n_layers blocks -> final RMSNorm -> LM head tiles) and the
teacher-forced decode driver. Modes and mask staging come from ``perseus.impl.driver.ImplModel``.

Weights are not resident: a Llama block is ~4.6k plaintexts (~1 GB of slot vectors), so every
weight product encodes its plaintext at use (the GPU encoder) from the memory-mapped slot cache
(weights.LlamaWeights); only the masks go through the runtime's encode cache.

Block numbering: 0..n_layers-1 the transformer blocks, n_layers the tail (final norm + LM head).
"""
from __future__ import annotations

import time

import numpy as np

from perseus.impl.attention import ComplexKVCache, KVCache, Rope, attention_step_masks, \
    complex_attention_step_masks, rope_step_masks
from perseus.impl.driver import ImplModel
from perseus.impl.linear import apply_linear, prepare_linear_input
from perseus.impl.norm import rms_norm

from . import prefix as P
from .block import llama_block


class LlamaModel(ImplModel):
    def __init__(self, inf, weights, cfgs, *, core=None, n_layers=None, packing="cachemir_complex",
                 lmhead_cap=44, unit=None, profile=False, kv_offload=None, dims=None):
        super().__init__(inf, core=core, unit=unit, profile=profile, dims=dims or weights.dims,
                         kv_offload=kv_offload)
        if packing not in ("cachemir", "cachemir_complex"):
            raise ValueError(f"packing must be cachemir or cachemir_complex, got {packing!r}")
        if not self.rt.ops.gelu_fold:
            raise ValueError("the Llama port folds the SiLU's 1/xmax into the gate: run with GELU_FOLD=1")
        if packing == "cachemir_complex" and not self.rt.ops.complex_payload:
            raise ValueError("cachemir_complex needs a complex payload session (CKKS_COMPLEX=1)")
        self.packing = packing
        self.w, self.cfgs = weights, cfgs
        self.n_layers = int(n_layers if n_layers is not None else cfgs.n_layers)
        self.lmhead_cap = lmhead_cap
        c = weights.store.config
        self.vocab = int(c["vocab_size"])
        self.H, self.H_kv = int(c["num_attention_heads"]), int(c["num_key_value_heads"])
        self.rope = Rope(cfgs.model["rope_inv_freq"])
        Cache = ComplexKVCache if packing == "cachemir_complex" else KVCache
        self.kv = [Cache(self.rt.dims.d_head) for _ in range(self.n_layers)]
        self._prefix = None
        self.taps = None                # {(pos, block): decrypted block output} when a dict
        self._bw = {}
        self._lm = None

    # ── weights ──
    def block_weights(self, b):
        if b not in self._bw:
            w = self.w.block(b)
            for f in ("q", "k", "v", "out", "gate", "up", "down", "kv"):
                if getattr(w, f) is not None:
                    self.w.attach_coeffs(getattr(w, f), f"b{b}.{f}")
            self._bw[b] = w
        return self._bw[b]

    def lm_tiles(self):
        if self._lm is None:
            paired = self.rt.ops.complex_payload
            self._lm = self.w.lm_tiles(paired=paired)
            for j, t in enumerate(self._lm):
                self.w.attach_coeffs(t, f"lm.c{j}" if paired else f"lm.{j}")
        return self._lm

    def preload(self):
        """Build (or open) every slot cache entry."""
        for b in range(self.n_layers):
            self.block_weights(b)
        self.lm_tiles()

    def load_plans(self, plan_dir, validate=True):
        return super().load_plans(plan_dir, range(self.n_layers + 1), validate=validate)

    # ── the BOS prefix ──
    def prefix_caches(self):
        """The plain slot vectors of every block's cache after the BOS push (prefix.py)."""
        if self._prefix is None:
            d = self.rt.dims
            dh = d.dim // self.H
            out = []
            for k, v in self.w.bos_kv(self.n_layers):
                kf = P.expanded_heads(k, self.H, self.H_kv, dh)
                vf = P.expanded_heads(v, self.H, self.H_kv, dh)
                out.append(P.plain_cache(d, kf, vf, self.packing, self.rope))
            self._prefix = out
        return self._prefix

    def _encrypt_public(self, vec):
        """An encryption of the public slot vector `vec`: a fresh encryption of zero plus it."""
        z = self.core.encode_token_input(self.inf, np.zeros(self.rt.dims.dim))
        return self.rt.ops.add_pt(z, vec)

    def start(self):
        """Reset the caches to the BOS prefix (cache count 1 everywhere)."""
        enc = self._encrypt_public
        self._parked.clear()
        for i, (kv, pk) in enumerate(zip(self.kv, self.prefix_caches())):
            kv.reset()
            if isinstance(kv, ComplexKVCache):
                kv.k_buckets = [enc(c) for c in pk.k_buckets]
                kv.k_pend = [enc(c) for c in pk.k_pend]
                kv.v_buckets = [None if c is None else enc(c) for c in pk.v_buckets]
            else:
                kv.k_groups = [enc(c) for c in pk.k_groups]
                kv.v_lanes = [None if c is None else enc(c) for c in pk.v_lanes]
            kv.k_count, kv.v_count = pk.k_count, pk.v_count
            if self.kv_offload:
                # ~65 fresh ciphertexts per block: parked in the pinned KV arena at once (the whole
                # prefix does not fit the device next to the keys), reloaded by the stage prefetch
                cts, keys = self.stage_caches(i)
                self.rt.ops.device_sync()
                self.inf.kv_store(cts, keys)
                self.inf.kv_sync()
                self.inf.kv_evict(cts)
                self._parked[i] = set(keys)

    # ── the hooks ──
    def step_masks(self, pos):
        """Per block the attention masks at cache count pos + 1 (the prefix included) and the
        RoPE masks of position `pos` (shared by every block)."""
        d = self.rt.dims
        step = complex_attention_step_masks if self.packing == "cachemir_complex" else attention_step_masks
        items = []
        for b in range(self.n_layers):
            items += step(self.rt, self.cfgs.block(b)[2], pos + 1, pos % d.t)
        kind = self.packing if self.rt.ops.complex_payload else "real"
        items += rope_step_masks(self.rt, self.rope, pos, kind)
        seen, out = set(), []
        for key, build in items:
            if key not in seen:
                seen.add(key); out.append((key, build))
        return out

    def routes_for(self, pos):
        """SM_PERIODIC=2: the Gp x tH-slot route of the softmax's (Gp tH)-periodic values (cache count pos + 1)."""
        from perseus.impl.attention import sm_periodic
        if not getattr(self.rt.ops, "sm_periodic_general", False):
            return None
        gp = max(1, sm_periodic(self.rt, pos + 1))
        return gp, ({gp * self.rt.dims.tH} if gp > 1 else set())

    def stage_caches(self, i):
        if i < self.n_layers:
            return self.kv[i].cts(f"impl.b{i}.")
        return None

    def token_stages(self, x, pos, head=True):
        def block_fn(b):
            def compute(x):
                if self.taps is not None and b in self.tap_blocks:
                    self.rt.tap = lambda name, ct: self.taps.__setitem__((pos, f"{b}.{name}"), self._decode_any(ct))
                else:
                    self.rt.tap = None
                cap = self.enter_stage(b, pos)
                with self.rt.step(f"blk{b}"):
                    x = llama_block(self.rt, x, self.block_weights(b), self.kv[b], self.cfgs.block(b),
                                    pos, self.rope)
                self.exit_stage(b, cap)
                if self.taps is not None:           # debug: every block's output, decrypted
                    self.taps[(pos, b)] = self.hidden(x)
                return x
            return compute

        def tail_fn(x):
            b = self.n_layers
            cap = self.enter_stage(b, pos)
            with self.rt.step("tail"):
                h = rms_norm(self.rt, x, self.cfgs.ln_f, pos)
                if self.lmhead_cap is not None:
                    self.rt.ops.bootstrap_hint(h, int(self.lmhead_cap), True)
                if head:
                    tiles = self.lm_tiles()
                    prep = prepare_linear_input(self.rt, h, tiles[0].d_in, tiles[0].d_out)
                    h = [apply_linear(self.rt, prep, t) for t in tiles]
            self.exit_stage(b, cap)
            return h

        stages = [(f"blk{b}", None, block_fn(b)) for b in range(self.n_layers)]
        stages.append(("tail", None, tail_fn))
        return stages

    # ── the forward ──
    def encode_input(self, token):
        return self.rt.ops.encode_token(self.w.embedding(token))

    def decode_token(self, x, pos, head=True):
        return self.run_stages(x, pos, self.token_stages(x, pos, head))

    def logits(self, tiles):
        from perseus.impl.layout import decode_linear_output
        d, N = self.rt.dims, self.rt.dims.N
        out = []
        if self.rt.ops.complex_payload:
            for m, tile in enumerate(tiles):
                cv = self.rt.ops.decrypt_slots_complex(tile)
                out += [decode_linear_output(cv.real, N, d.hid, N), decode_linear_output(cv.imag, N, d.hid, N)]
        else:
            out = [self.rt.ops.decode_linear_output(t, d.hid, N) for t in tiles]
        return np.concatenate(out)[:self.vocab]

    tap_blocks = ()

    def _decode_any(self, ct):
        """A tapped ciphertext's real slot values (token or MLP layout, decoded by the caller)."""
        return self.rt.ops.decrypt_slots_complex(ct).real.copy()

    def hidden(self, x):
        """Decrypt a block's output (token features, for debugging against the mirror)."""
        return self.rt.ops.decode_token(x)

    # ── drivers ──
    def run_decode(self, tokens, gt_logits=None, on_token=None):
        """Teacher-forced decode of tokens[1:] after the BOS prefix (tokens[0] must be BOS):
        one forward per position, logits decrypted per token. Returns per-token dicts."""
        self.start()
        results = []
        for pos in range(1, len(tokens)):
            self.token_begin(pos)
            x = self.encode_input(tokens[pos])
            tiles = self.decode_token(x, pos)
            self.token_tail(pos)
            rec = self.token_end(pos)
            lg = self.logits(tiles)
            rec.update(logits=lg, top1=int(np.argmax(lg)), bootstraps=int(getattr(self.inf.fhe, "total_bootstraps", 0)))
            if gt_logits is not None and pos < len(gt_logits):
                rec.update(dist_report(lg, np.asarray(gt_logits[pos])))
            results.append(rec)
            if on_token:
                on_token(rec)
        return results


def kl(p_logits, q_logits):
    p = np.asarray(p_logits, dtype=np.float64); q = np.asarray(q_logits, dtype=np.float64)
    lp = p - p.max(); lp -= np.log(np.exp(lp).sum())
    lq = q - q.max(); lq -= np.log(np.exp(lq).sum())
    return float(np.sum(np.exp(lp) * (lp - lq)))


def dist_report(ours, ref):
    ours = np.asarray(ours, dtype=np.float64); ref = np.asarray(ref, dtype=np.float64)
    top1, rtop = int(np.argmax(ours)), int(np.argmax(ref))
    return {"ref": rtop, "ref_rank": int((ours > ours[rtop]).sum()), "kl": kl(ref, ours), "hit": top1 == rtop}
