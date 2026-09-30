"""GPT-2 on ``perseus.impl``: weights + configs + per-block K/V caches over one session, the
decode step (12 blocks -> final LN -> LM head), the encrypted argmax + feedback, the
teacher-forced decode driver and the generation loop. Modes, ring and mask staging come from
``perseus.impl.driver.ImplModel``.

Block numbering follows the C++/EncGPT2 convention: 0..n_layers-1 the transformer blocks,
n_layers the tail (final LN + LM head), n_layers+1 CutMax, n_layers+2 the feedback embed.
"""
from __future__ import annotations

import os
import time

import numpy as np

from perseus.impl.attention import (ComplexKVCache, KVCache, attention_step_masks,
                                    complex_attention_step_masks)
from perseus.impl.config import Configs
from perseus.impl.driver import ImplModel
from perseus.impl.norm import norm_step_masks

from .block import transformer_block
from .head import (cutmax_argmax, cutmax_argmax_packed, decode_logits, decode_z, feedback_embed,
                   final_ln, lm_head, make_ones, pack_tiles)
from . import weights as W


class Gpt2Primitives(ImplModel):
    def __init__(self, inf, store, cfgs: Configs, *, core=None, n_layers=None,
                 fold_ln1=True, fold_ln2=True, fold_lnf=False, lmhead_cap=44,
                 unit=None, profile=False, packing="cachemir", kv_offload=None):
        super().__init__(inf, core=core, unit=unit, profile=profile, kv_offload=kv_offload)
        # "cachemir": real weights and linears (with the K/V pair bootstrap and the packed
        # CutMax on a complex payload); "cachemir_complex": the fused K + iV linear, the
        # output-packed up-projection, complex K/V buckets and the paired LM-head tile
        if packing not in ("cachemir", "cachemir_complex"):
            raise ValueError(f"packing must be cachemir or cachemir_complex, got {packing!r}")
        self.packing = packing
        self.complex_packing = packing == "cachemir_complex"
        if self.complex_packing and not self.rt.ops.complex_payload:
            raise ValueError("cachemir_complex needs a complex payload session (CKKS_COMPLEX=1)")
        self.store, self.cfgs = store, cfgs
        self.n_layers = int(n_layers if n_layers is not None else cfgs.model.n_layers)
        self.fold = (fold_ln1, fold_ln2, fold_lnf)
        self.lmhead_cap = lmhead_cap
        self.vocab = W.vocab_size(store)
        self.W_tile = self.rt.dims.N
        # CKKS_COMPLEX=1 (the C++ decode configuration): real weights and linears, but the K/V
        # push refreshes K + iV with one bootstrap, CutMax runs on the packed tile pair and
        # the feedback goes through the complex tile
        self.complex = bool(self.rt.ops.complex_payload)
        Cache = ComplexKVCache if self.complex_packing else KVCache
        self.kv = [Cache(self.rt.dims.d_head) for _ in range(self.n_layers)]
        self._bw = {}
        self._lnf = None
        self._lm = None
        self._fb = None
        self.ones = None
        self.stats = {"bootstraps": 0, "tok_s": []}

    # ── weights (host numpy, lazily encoded) ──
    def block_weights(self, b):
        if b not in self._bw:
            self._bw[b] = W.block_weights(self.store, self.cfgs, b, self.rt.dims,
                                          self.fold[0], self.fold[1], self.complex_packing)
        return self._bw[b]

    def lnf_params(self):
        if self._lnf is None:
            self._lnf = W.final_ln_params(self.store, self.cfgs.ln_f, self.rt.dims, self.fold[2])
        return self._lnf

    def lm_tiles(self):
        if self._lm is None:
            Wlm = W.lm_head_matrix(self.store, self.cfgs.ln_f, self.rt.dims, self.vocab, self.fold[2])
            self._lm = W.lm_head_tiles(Wlm, self.rt.dims, self.vocab, self.W_tile,
                                       paired=self.complex_packing)
        return self._lm

    def fb_tiles(self):
        if self._fb is None:
            Wlm = W.lm_head_matrix(self.store, self.cfgs.ln_f, self.rt.dims, self.vocab, self.fold[2])
            self._fb = W.feedback_tiles(Wlm, self.rt.dims, self.vocab, self.W_tile,
                                        packed=self.complex)
        return self._fb

    def preload(self):
        for b in range(self.n_layers):
            self.block_weights(b)
        self.lnf_params(); self.lm_tiles()

    def load_plans(self, plan_dir, validate=True, argmax_blocks=True):
        """block_<b>_placement.json for the transformer blocks, the tail and, when present,
        CutMax (n_layers+1, planned from the same capture with the tail plan's exit as its
        entry: make_plan.sh ... argmax) and the feedback (n_layers+2); a stage without a
        plan file runs eager. block_0_feedback_placement.json (make_plan.sh ... feedback) is
        block 0 planned for a fed-back token, which enters at the landing of the feedback's
        bootstrap instead of as the fresh encryption the capture recorded; generate() uses it."""
        return super().load_plans(plan_dir, range(self.n_layers + (3 if argmax_blocks else 1)),
                                  validate=validate, variants=("feedback",))

    # ── the hooks ──
    def start(self):
        for kv in self.kv:
            kv.reset()
        self.ones = None

    def step_masks(self, pos):
        """The masks of position `pos` that change with the step (the C++ step_mask_walk):
        per block the attention masks at cache count pos+1 and the LN centering scales."""
        items = []
        for b in range(self.n_layers):
            ln1, ln2, sm, _ge = self.cfgs.block(b)
            step_masks = complex_attention_step_masks if self.complex_packing else attention_step_masks
            items += step_masks(self.rt, sm, pos + 1, pos % self.rt.dims.t)
            for cfg in (ln1, ln2):
                if cfg.center_scale_sq:
                    items += norm_step_masks(self.rt, cfg, pos)
        if self.cfgs.ln_f.center_scale_sq:
            items += norm_step_masks(self.rt, self.cfgs.ln_f, pos)
        # one item per key (the same mask is shared by every block that uses it)
        seen, out = set(), []
        for key, build in items:
            if key not in seen:
                seen.add(key); out.append((key, build))
        return out

    def stage_caches(self, i):
        if i < self.n_layers:
            return self.kv[i].cts(f"impl.b{i}.")
        return None

    def token_stages(self, x, pos, head=True):
        def block_fn(b):
            def compute(x):
                cap = self.enter_stage(b, pos)
                with self.rt.step(f"blk{b}"):
                    x = transformer_block(self.rt, x, self.block_weights(b), self.kv[b],
                                          self.cfgs.block(b), pos)
                self.exit_stage(b, cap)
                return x
            return compute

        def tail_fn(x):
            b = self.n_layers
            cap = self.enter_stage(b, pos)
            with self.rt.step("tail"):
                h = final_ln(self.rt, x, self.cfgs.ln_f, self.lnf_params(), pos, self.lmhead_cap)
                out = lm_head(self.rt, h, self.lm_tiles()) if head else h
            self.exit_stage(b, cap)
            return out

        stages = [(f"blk{b}", f"b{b}.", block_fn(b)) for b in range(self.n_layers)]
        stages.append(("tail", None, tail_fn))
        return stages

    # ── the forward ──
    def encode_input(self, row):
        return self.rt.ops.encode_token(row)

    def decode_token(self, x, pos, head=True):
        """12 blocks -> final LN (-> LM head tiles). x: the token's input ciphertext."""
        if self.ones is None:
            self.ones = make_ones(self.rt.ops, x)       # ensure_const_one: from the freshest ct
        return self.run_stages(x, pos, self.token_stages(x, pos, head))

    def logits(self, tiles):
        return decode_logits(self.rt, tiles, self.vocab, self.W_tile)

    def argmax_encrypted(self, tiles, pos):
        b = self.n_layers + 1
        cap = self.enter_stage(b, pos)
        marks, tap = {}, None
        if os.environ.get("IMPL_CUTMAX_MARKS"):          # refreshes per phase, like [cutmax_bts]
            tap = lambda name, ct: marks.__setitem__(name, int(self.inf.fhe.total_bootstraps))
            b0 = int(self.inf.fhe.total_bootstraps)
        with self.rt.step("cutmax"):
            if self.complex and len(tiles) == 1:              # the paired LM-head tile
                z = [cutmax_argmax_packed(self.rt, tiles[0], self.vocab, self.cfgs.cutmax, None, tap=tap)]
            elif self.complex and len(tiles) == 2:
                z = [cutmax_argmax_packed(self.rt, pack_tiles(self.rt, tiles), self.vocab,
                                          self.cfgs.cutmax, None, tap=tap)]
            else:
                z = cutmax_argmax(self.rt, tiles, self.vocab, self.cfgs.cutmax, None, tap=tap)
        if tap is not None:
            phases = ["entry.B"] + [f"i{i}.end" for i in range(len(self.cfgs.cutmax.iters))] + ["sum.Z"]
            prev, out = b0, []
            for ph in phases:
                if ph in marks:
                    out.append(f"{ph}={marks[ph] - prev}"); prev = marks[ph]
            print(f"[impl] cutmax_bts tok{pos} " + " ".join(out) + f" total={prev - b0}", flush=True)
        self.exit_stage(b, cap)
        return z

    def feedback(self, z, next_pos):
        b = self.n_layers + 2
        cap = self.enter_stage(b, next_pos)
        with self.rt.step("feedback"):
            x = feedback_embed(self.rt, z, self.fb_tiles(),
                               W.wpe_row(self.store, next_pos, self.rt.dims.dim), next_pos)
        self.exit_stage(b, cap)
        return x

    def decode_z(self, z):
        return decode_z(self.rt, z, self.vocab, self.W_tile)

    # ── drivers ──
    def run_decode(self, inputs, gt_logits=None, on_token=None, argmax=False):
        """Teacher-forced decode (src/app/pipeline.cu): one forward per oracle row,
        logits decrypted per token, optional CutMax on the tiles. Returns per-token dicts."""
        self.start()
        results = []
        for t, row in enumerate(inputs):
            self.token_begin(t)
            x = self.encode_input(row)
            tiles = self.decode_token(x, t)
            self.token_tail(t)                 # next step's masks: staged under what follows
            rec = self.token_end(t)            # decode_s = the forward (the C++ [tokstat] cut)
            lg = self.logits(tiles)
            rec.update(logits=lg, top1=int(np.argmax(lg)))
            if gt_logits is not None and t < len(gt_logits) and len(gt_logits[t]):
                rec.update(dist_report(lg, np.asarray(gt_logits[t])))
            if argmax:
                t1 = time.time()
                b0 = int(getattr(self.inf.fhe, "total_bootstraps", 0))
                z = self.argmax_encrypted(tiles, t)
                zdec = self.decode_z(z)
                rec.update(cutmax=int(np.argmax(zdec)), z_mass=float(zdec.max()),
                           argmax_s=time.time() - t1,
                           argmax_bts=int(getattr(self.inf.fhe, "total_bootstraps", 0)) - b0)
            results.append(rec)
            if on_token:
                on_token(rec)
        return results

    def generate(self, prompt_rows, n_tokens, gt_logits=None, on_step=None):
        """src/app/pipeline.cu / perseus.nn.gpt2.stream: the prompt through decode
        steps (LM head on the last), then CutMax argmax + encrypted feedback."""
        if self.plans and 0 in self.plans and 0 not in self.plan_variants.get("feedback", {}):
            raise RuntimeError("planned generation needs block_0_feedback_placement.json in the "
                               "plan dir: make_plan.sh <graph> <plan> feedback")
        self.start()
        P = len(prompt_rows)
        for p, row in enumerate(prompt_rows):
            self.token_begin(p)
            x = self.encode_input(row)
            tiles = self.decode_token(x, p, head=(p == P - 1))
            self.token_tail(p)
        out = []
        for j in range(n_tokens):
            pos = P - 1 + j
            t0 = time.time()
            lg = self.logits(tiles)
            fhe_am = int(np.argmax(lg))
            z = self.argmax_encrypted(tiles, pos)
            zdec = self.decode_z(z)
            cm = int(np.argmax(zdec))
            off = np.abs(np.delete(zdec, cm))
            rec = {"pos": pos, "cutmax": cm, "fhe_argmax": fhe_am, "z_mass": float(zdec[cm]),
                   "z_off_sum": float(off.sum()), "z_off_max": float(off.max()), "logits": lg}
            if gt_logits is not None and pos < len(gt_logits) and len(gt_logits[pos]):
                rec["gt_argmax"] = int(np.argmax(gt_logits[pos]))
            out.append(rec)
            if on_step:
                on_step(rec)
            if j + 1 < n_tokens:
                self.token_begin(pos + 1)
                x = self.feedback(z, pos + 1)
                self.plan_variant = "feedback"
                try:
                    tiles = self.decode_token(x, pos + 1)
                finally:
                    self.plan_variant = None
                self.token_tail(pos + 1)
            rec["step_s"] = time.time() - t0
        return out


# ── the gate (scripts/modes_baseline.py) ───────────────────────────────────────

def kl(p_logits, q_logits):
    p = np.asarray(p_logits, dtype=np.float64); q = np.asarray(q_logits, dtype=np.float64)
    p = np.exp(p - p.max()); p /= p.sum()
    q = np.exp(q - q.max()); q /= q.sum()
    return float(np.sum(p * (np.log(np.maximum(p, 1e-30)) - np.log(np.maximum(q, 1e-30)))))


def dist_report(ours, ref):
    ours = np.asarray(ours, dtype=np.float64); ref = np.asarray(ref, dtype=np.float64)
    top1, rtop = int(np.argmax(ours)), int(np.argmax(ref))
    ref_rank = int((ours > ours[rtop]).sum())
    return {"ref": rtop, "ref_rank": ref_rank, "kl": kl(ref, ours), "hit": top1 == rtop}


def dist_gate(results, kl_max=5.0, rank_max=8):
    bad = [r for r in results if "kl" in r and (r["kl"] > kl_max or r["ref_rank"] > rank_max)]
    return not bad, bad
