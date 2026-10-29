"""Llama prefill PLANNED on the device: the paged blocked-GQA CCMM prefill (plans pr9 / pr2) as one compiled `Program`
per block, run in Rust.

The op stream depends on the prompt length and layout only, so it is recorded once on the CPU twin (fakecap
`LlamaPrefill`), compiled per block against the plan and bound on one planned Session that also holds the PCMM and
CCMM services the blocks' `Finish` ops call inside the replay. The paged prefill (PF_PAGED=1) runs page 0 (BOS + 127
tokens) through every block with the one token-0 plan. Details: docs/ARCHITECTURE.md (examples/llama/planned.py).
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np

from perseus.impl.pcmm import session_specs
from perseus.impl.prefill import PlannedPrefill


def _key(v):
    return "v/" + hashlib.blake2b(np.ascontiguousarray(v).tobytes(), digest_size=12).hexdigest()


class LlamaPlannedPrefill(PlannedPrefill):
    """`PlannedPrefill` for `examples.llama.prefill.LlamaPrefill` (blocked layout, PCMM linears, CCMM attention,
    paged). `kv_out`: keep every block's K / V (post-RoPE, the hand-off's input) as program outputs."""

    def __init__(self, model, plan_dir, config, keys, *, kv_out=False, tap_block=None, taps_per_step=4,
                 ccmm_kappa=(256.0, 16.0), ccmm_sweep=(), **kw):
        super().__init__(model, plan_dir, config, keys, **kw)
        # the CCMM operating points (QK, PV): matmul.rs `CcmmRt::kappa`; measured with `ccmm_sweep`
        self.ccmm_kappa, self.ccmm_sweep = tuple(float(k) for k in ccmm_kappa), tuple(ccmm_sweep)
        self.kv_out = kv_out
        # debug: block `tap_block`'s intermediates (up to `taps_per_step` op outputs per step) kept as program outputs
        # and compared with the twin's values after the run (`tap_report`)
        self.tap_block, self.taps_per_step = tap_block, taps_per_step
        self.tap_steps = set()
        self.snaps, self.taps = {}, []
        self.ccmm = os.environ.get("PF_ATTN", "") == "ccmm"
        if self.ccmm:       # the calibration probe's eager ciphertexts: a token, 2 x 16 operands, 16 outputs
            self.pool = max(self.pool, 56)

    # ── setup (CPU) ──
    def record(self, toks):
        """The forward's per-block streams on the CPU twin (one page: BOS + T tokens). Plaintext values are kept once
        per distinct value (`self.uniq`); a block's record maps its plaintext names to those keys."""
        from perseus.impl.driver import Recorder
        from . import prefill as LP
        t0 = time.time()
        m = self.model
        self.toks = [int(t) for t in toks]
        rec = Recorder(m, self.n_layers)
        orig = LP.llama_prefill_block_blocked
        io, kv_io = {}, {}
        self.uniq = {}

        def block(rt, Xs, P, cfgs, ch, rope, prefixes, B, kv=None, layer=0, paged=None, last=None):
            flat = [x for blk in Xs for x in blk]
            metas = [(int(c.level), int(c.noise_deg)) for c in flat]
            vals = [np.array(c.vec) for c in flat] if layer == 0 else None
            Y = orig(rt, Xs, P, cfgs, ch, rope, prefixes, B, kv=kv, layer=layer, paged=paged, last=last)
            # graph names as this block's capture wrote them (the next block renames what it reads)
            io[layer] = ([c.name for c in flat], [c.name for blk in Y for c in blk], metas, vals, ch, [len(b) for b in Y])
            return Y

        def on_kv(p, b, K, V):
            kv_io[b] = ([[c.name for c in ks] for ks in K], [[c.name for c in vs] for vs in V])

        def on_block(p, b, Xs, ch, s):
            r = rec.blocks[b]
            for nm in list(r.pts):
                v = r.pts[nm]
                k = _key(v)
                self.uniq.setdefault(k, np.ascontiguousarray(v))
                r.pts[nm] = k
            r.objs = {}

        fhe = m.inf.fhe
        new_var = fhe.new_var
        tapping = [False]

        def snap_var(ct):
            nm = new_var(ct)
            if tapping[0] and getattr(ct, "vec", None) is not None:
                self.snaps[nm] = np.array(ct.vec)
            return nm

        def block_t(rt, Xs, P, cfgs, ch, rope, prefixes, B, kv=None, layer=0, paged=None, last=None):
            tapping[0] = layer == self.tap_block
            try:
                return block(rt, Xs, P, cfgs, ch, rope, prefixes, B, kv=kv, layer=layer, paged=paged, last=last)
            finally:
                tapping[0] = False

        LP.llama_prefill_block_blocked = block_t
        if self.tap_block is not None:
            fhe.new_var = snap_var
        try:
            if os.environ.get("PF_PAGED", "0") not in ("", "0"):
                pages, _ = m.prefill_paged(toks, on_block=on_block, on_kv=on_kv if self.kv_out else None)
            else:           # the unpaged blocked prefill (one chunk-pair prompt)
                Xs_, ch_, _ = m.prefill(toks, on_block=lambda b, Xs, s: on_block(0, b, Xs, None, s),
                                        on_kv=(lambda b, K, V: on_kv(0, b, K, V)) if self.kv_out else None)
                pages = [(Xs_, ch_)]
        finally:
            LP.llama_prefill_block_blocked = orig
            fhe.new_var = new_var
        if len(pages) != 1:
            raise RuntimeError(f"planned prefill: {len(pages)} pages; the token-0 plan runs page 0 only here")
        self.ch = pages[0][1]
        self.in_metas, self.in_names, self.kv_names = {}, {}, kv_io
        for i, r in rec.blocks.items():
            names_in, names_out, metas, vals, _ch, groups = io[i]
            outs = list(names_out)
            if self.kv_out and i in kv_io:
                outs += [n for part in kv_io[i] for grp in part for n in grp]
            if i == self.tap_block:
                self.taps = self._pick_taps(r.nodes, set(outs))
                outs += self.taps
            r.out_names = outs
            self.in_names[i] = names_in
            self.in_metas[i] = metas
            if i == 0:
                self.in_vals = vals
        self.n_hidden = len(io[0][1])
        self.recs = rec.blocks
        self.record_s = time.time() - t0
        return self.recs

    def _pick_taps(self, nodes, have):
        """Per step, up to `taps_per_step` op outputs spread over the step (the last one always), snapped by the twin."""
        by_step = {}
        for n in nodes:
            o = n.get("output")
            if o in self.snaps and o not in have and not n["op_type"].endswith("hint"):
                by_step.setdefault(n["step"], []).append(o)
        out = []
        for st, names in by_step.items():
            if self.tap_steps and st not in self.tap_steps:
                continue
            k = len(names) if self.tap_steps else min(self.taps_per_step, len(names))
            idx = sorted({int(round(x)) for x in np.linspace(0, len(names) - 1, k)})
            out += [names[i] for i in idx]
        return out

    def tap_report(self):
        """[(name, step-order index, max |device - twin| / max |twin|)] of the tapped intermediates."""
        b = self.tap_block
        reg_of = {self.recs[b].out_names[j]: r for j, (_o, r) in enumerate(self.reg[b])}
        out = []
        for i, nm in enumerate(self.taps):
            got = np.asarray(self.decrypt([reg_of[nm]])[0])
            want = self.snaps[nm]
            if want.shape != got.shape:
                want = np.resize(want, got.shape)
            den = max(float(np.abs(want).max()), 1e-300)
            out.append((nm, i, float(np.abs(got - want).max()) / den, den, float(np.abs(got).max())))
        return out

    def prepare(self, toks):
        super().prepare(toks)
        # registers ping-pong between two sets (block b writes set b % 2, reads set (b - 1) % 2): 32 blocks of 8
        # outputs would hold 8 GB of full-layout ciphertexts; a block's outputs are read before b + 2 overwrites them
        # (per-block checks decrypt right after their block: `on_block`)
        width = max(len(self.reg[b]) for b in range(self.n_layers))
        for b in range(self.n_layers):
            self.reg[b] = [(j, (b % 2) * width + k) for k, (j, _r) in enumerate(self.reg[b])]
        self.n_regs = min(self.n_layers, 2) * width
        # the hidden registers of block b are its first `n_hidden` outputs (then its K / V when kv_out)
        self.hid_reg = {b: [r for _j, r in self.reg[b][:self.n_hidden]] for b in range(self.n_layers)}
        return self

    def input_values(self, toks):
        """Block 0's input slot vectors for another prompt of the recorded length (the PF_WARMUP runner: pers/dev
        31ab85f's untimed prefill of a different document first), packed as `prefill_paged` packs page 0 on the twin;
        checked against the recorded prompt's own."""
        from perseus.impl import blocked as BL
        from perseus.impl.ccmm_paged import page_tokens
        from perseus.impl.prefill import Chunks
        m = self.model

        def pack(tk):
            prompt = list(tk[1:])
            n = page_tokens(0)
            if len(prompt) > n:
                raise ValueError(f"input_values: {len(prompt)} prompt tokens, page 0 holds {n} (the token-0 plan)")
            chunk = prompt + [prompt[-1]] * (n - len(prompt))
            rows = np.stack([m.w.embedding(t) for t in chunk])
            Xs = BL.encode_blocked(m.rt, rows, Chunks(len(chunk), m.blocks.t, base=1), m.blocks)
            return [np.array(c.vec) for blk in Xs for c in blk]
        if not hasattr(self, "_toks_ok"):
            mine = pack(self.toks)
            if len(mine) != len(self.in_vals) or not all(np.allclose(a, b, atol=1e-9) for a, b in zip(mine, self.in_vals)):
                raise RuntimeError("LlamaPlannedPrefill.input_values: the packing does not reproduce the recorded block-0 inputs")
            self._toks_ok = True
        if len(toks) != len(self.toks):
            raise ValueError(f"warm-up prompt of {len(toks)} tokens: the programs were compiled for {len(self.toks)}")
        return pack(toks)

    def linears(self):
        out = {}
        for b in range(self.n_layers):
            for lin in self.model.pcmm_weights(b).values():
                out[lin.name] = lin
        return out

    def setup(self, toks):
        """`prepare` (CPU) + the planned Session: routes, keys (+ PCMM / CCMM keys), slab, registers, the services;
        every block bound, its plaintexts staged by value; the CCMM calibrated; then the secret is dropped."""
        from perseus import _pc
        if not hasattr(self, "progs"):
            self.prepare(toks)
        t0 = time.time()
        kw = self.session_kwargs()
        progs, uniq, lins, specs = kw["programs"], self._uniq, self._lins, kw["pcmm"] or []
        n_pts = len(uniq)
        self.sess = self.open_session(kw)
        self.store = self.sess.pt_store
        for b, p in enumerate(progs):
            p.bind(self.sess, self.store)
            p.alias_pts(self.alias.get(self.labels[b], {}))
        self.sess.complete_setup()
        self.fhe = self.sess.fhe
        tw = time.time()
        for lin in lins.values():
            self.sess.pcmm_upload(lin.name, np.ascontiguousarray(lin.Wd, dtype=np.int8).tobytes())
        self.upload_s = time.time() - tw
        if self.ccmm:
            self.calibrate_ccmm()
        self.sess.drop_secret()
        staged = self.store.stage([(key, lv, v) for (key, lv), v in uniq.items()])
        n_refs = sum(len({(nm, lv) for nm, lv, _d, _f in p.pt_refs}) for p in progs)
        self.setup_s = time.time() - t0
        self.log(f"[planned] {len(progs)} programs, {self.n_regs} registers, {n_refs} plaintext keys = {n_pts} distinct "
                 f"(values, level) ({staged} staged), {len(specs)} PCMM linears (upload {self.upload_s:.1f} s); "
                 f"record {self.record_s:.1f} s, compile {self.compile_s:.1f} s, session "
                 f"{self.sess.setup_seconds()[0]:.1f} s (keygen {self.sess.setup_seconds()[1]:.1f}), setup total "
                 f"{self.setup_s:.1f} s")
        self.log("[planned] " + self.sess.mem_report.replace("\n", " | "))
        return self

    def session_kwargs(self):
        """The planned Session's arguments after (config, keys) (`Session.key_manifest` / `from_bundle` take the
        same); keeps the plaintext aliases and the distinct (value, level) plaintexts for `setup`."""
        progs = [self.progs[lab] for lab in self.labels]
        self.alias, uniq = {}, {}
        for b, p in enumerate(progs):
            r, lab = self.recs[b], self.labels[b]
            for nm, lv, _d, form in p.pt_refs:
                key = r.pts[nm]
                store = f"{lab}/{nm}" if (form == "at_use" or nm.startswith("pt_")) else nm
                self.alias.setdefault(lab, {})[store] = key
                uniq.setdefault((key, lv), self.uniq[key])
        pt_rows = self.pt_rows and max(self.pt_rows, 54 - min(lv for _k, lv in uniq))
        lins = self.linears()
        t = self.model.rt.dims.t
        specs = session_specs(lins.values(), t)
        self._uniq, self._lins = uniq, lins
        from perseus.impl.driver import register_levels
        reg_levels = register_levels(self.n_regs, [(progs[b], j, r) for b in range(self.n_layers) for j, r in self.reg[b]])
        return dict(pool=self.pool, pt_slots=len(uniq) + 8, programs=progs, regs=self.n_regs, pt_rows=pt_rows,
                    stc_cache=self.stc_cache, pcmm=specs or None, pcmm_cf=self.pcmm_cf, check_meta=self.check_meta,
                    ccmm=self.ccmm, reg_levels=reg_levels, pt_levels=[lv for _k, lv in uniq])

    def open_session(self, kw):
        from perseus import _pc
        return _pc.Session(self.cfg, self.keys, **kw)

    def calibrate_ccmm(self):
        """ccmm.py `prepare` + `_calibrate` on the session's CCMM service: nf_ref / nf_split from a fresh token at the
        bootstrap output level, then c_unit from known probe matrices (shift 0, factors 2^-3), fitted like the old one."""
        from perseus.impl import ccmm as CC
        from perseus.impl import sinc as sc
        t0 = time.time()
        fhe, sess = self.fhe, self.sess
        L = int(self.in_metas[0][0][0])
        S = len(self.in_vals[0])
        tok = fhe.encrypt(np.zeros(S, dtype=np.complex128), L, 1)
        nf_ref, nf_split = sess.ccmm_prepare(tok)
        del tok
        rng = np.random.default_rng(1)
        A = np.zeros((sc.D, sc.D, sc.BATCH), complex)
        B = np.zeros((sc.D, sc.D, sc.BATCH))
        A[:, :, :16] = 0.05 * (rng.standard_normal((sc.D, sc.D, 16)) + 1j * rng.standard_normal((sc.D, sc.D, 16)))
        B[:, :, :16] = 0.05 * rng.standard_normal((sc.D, sc.D, 16))
        xa = [fhe.encrypt(v, L, 1) for v in CC.from_mats(A)]
        xb = [fhe.encrypt(v.astype(np.complex128), L, 1) for v in CC.from_mats(B)]
        f = 2.0 ** -3
        ys = sess.ccmm_probe(xa, xb, False, f, f, 0, False)
        got = CC.to_mats([np.asarray(fhe.decrypt_slots_complex(y)) for y in ys])[:, :, :16]
        want = np.einsum("ijl,jkl->ikl", A, B)[:, :, :16]
        c_unit = float(np.real(np.vdot(want, got) / np.vdot(want, want))) / (f * f)
        err = np.abs(got - c_unit * f * f * want).max() / (np.abs(want).max() * c_unit * f * f)
        del xa, xb
        sess.ccmm_set(c_unit, *self.ccmm_kappa)
        land = (int(ys[0].level), int(ys[0].deg))
        fake_land = CC.fake_landing(getattr(getattr(self.model, "inf", None), "fhe", None))
        self.ccmm_cal = {"c_unit": c_unit, "nf_ref": nf_ref, "nf_split": nf_split, "probe_rel_err": float(err),
                         "land": land, "s": time.time() - t0}
        self.log(f"[ccmm] calibrated: c_unit {c_unit:.6g} (probe rel. err {err:.2e}, "
                 f"{np.log2(1 / max(err, 1e-300)):.1f} bits), nf_ref 2^{np.log2(nf_ref):.3f}, nf_split "
                 f"2^{np.log2(nf_split):.3f}, landing {land} (the twin's {fake_land}), "
                 f"{time.time() - t0:.1f} s")
        if land[0] != fake_land:
            raise RuntimeError(f"CCMM lands at level {land[0]} on the device, the CPU twin recorded {fake_land} "
                               f"(ccmm.fake_landing)")
        for kappa in self.ccmm_sweep:
            for case in ("qk", "pv"):
                bits = self.ccmm_case(case, kappa, L)
                self.log(f"[ccmm] sweep {case} kappa 2^{np.log2(kappa):.1f}: {bits:.2f} bits")

    def ccmm_case(self, case, kappa, L):
        """Precision (bits, max |want| / max err) of one merged CCMM shaped like Llama's: "qk" (Q complex, K real,
        b transposed, amax 4, bmax 24, out_scale 1/sqrt(512), shift 5) or "pv" (P >= 0, V real, amax 1.5, bmax 8,
        shift 3); operands duplicated on batch l + 16 (PF_CCMM_MERGE)."""
        from perseus.impl import ccmm as CC
        from perseus.impl import sinc as sc
        fhe, sess = self.fhe, self.sess
        rng = np.random.default_rng(7)
        D, BT = sc.D, sc.BATCH
        A = np.zeros((D, D, BT), complex)
        Bm = np.zeros((D, D, BT))
        if case == "qk":
            A[:, :, :16] = rng.standard_normal((D, D, 16)) + 1j * rng.standard_normal((D, D, 16))
            A[:, :, :16] *= 4.0 / np.abs(A[:, :, :16]).max()
            Bm[:, :, :16] = rng.standard_normal((D, D, 16))
            Bm[:, :, :16] *= 24.0 / np.abs(Bm[:, :, :16]).max()
            bt, amax, bmax, osc, shift = True, 4.0, 24.0, 1 / np.sqrt(512), 5
        else:
            A[:, :, :16] = rng.random((D, D, 16)) ** 4
            A[:, :, :16] *= 1.5 / A[:, :, :16].max()
            Bm[:, :, :16] = rng.standard_normal((D, D, 16))
            Bm[:, :, :16] *= 8.0 / np.abs(Bm[:, :, :16]).max()
            bt, amax, bmax, osc, shift = False, 1.5, 8.0, 1.0, 3
        A[:, :, 16:], Bm[:, :, 16:] = A[:, :, :16], Bm[:, :, :16]
        c_unit = sess.ccmm_stats()["c_unit"]
        p = osc * 2.0 ** -shift / c_unit * 0.5
        fa, fb = np.sqrt(p * bmax / amax), np.sqrt(p * amax / bmax)
        xa = [fhe.encrypt(v, L, 1) for v in CC.from_mats(A)]
        xb = [fhe.encrypt(v.astype(np.complex128), L, 1) for v in CC.from_mats(Bm)]
        ys = sess.ccmm_probe(xa, xb, bt, fa, fb, shift, True, kappa)
        got = np.array([np.asarray(fhe.decrypt_slots_complex(y)) for y in ys])
        P = osc * CC.exact(CC.from_mats(A), CC.from_mats(Bm), bt)
        want = np.array([CC.merged(P[2 * i], P[2 * i + 1], CC.w_theory()) for i in range(sc.NCT // 2)])
        ls = (np.arange(sc.SLOTS) - sc.OFFSET) % 32
        sel = ls < 16
        g, w = got[:, sel], want[:, sel]
        err = np.abs(g - w).max()
        sc_ = np.real(np.vdot(w, g) / np.vdot(w, w))
        efit = np.abs(g / sc_ - w).max()
        self.log(f"[ccmm] {case} kappa 2^{np.log2(kappa):.1f}: output scale {sc_:.4g}, fitted {np.log2(np.abs(w).max() / max(efit, 1e-300)):.2f} bits")
        return float(np.log2(np.abs(w).max() / max(err, 1e-300)))

    def hidden(self, b=None):
        """(T, dim) rows of block b's output (default: the last)."""
        import types
        from perseus.impl.blocked import decrypt_blocked
        b = self.n_layers - 1 if b is None else b
        vals = self.decrypt(self.hid_reg[b])
        shim = types.SimpleNamespace(ops=types.SimpleNamespace(decrypt_slots_complex=lambda v: v),
                                     dims=self.model.rt.dims)
        Bk = self.model.blocks
        return decrypt_blocked(shim, [vals[g * Bk.K:(g + 1) * Bk.K] for g in range(len(vals) // Bk.K)], self.ch, Bk,
                               self.model.rt.dims.dim)
