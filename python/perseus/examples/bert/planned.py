"""BERT (and the ViT subclass) PLANNED on the new runtime: the encoder's blocked PCMM + CCMM prefill as one compiled
`Program` per block, run in Rust.

The op stream depends on the prompt length (and, for BERT, the number of real keys) only, so it is recorded once on
the model's CPU twin (fakecap session, `impl.driver.Recorder`), compiled per block against the plan, and bound on one
planned Session holding the two matmul services the blocks' `Finish` ops call inside the replay: the PCMM linears
(every linear's digit planes resident) and the CCMM attention (impl/ccmm_mha.py, ccmm_mha_paged.py: Q K^T / P V with
their SinCBoot refreshes; keys from the secret, `c_unit` calibrated on the device before the secret is dropped).
"""
from __future__ import annotations

import os
import time

import numpy as np

from perseus.impl.pcmm import session_specs
from perseus.impl.prefill import PlannedPrefill


def plan_config(plan, graphs=False):
    """The session Config of a planned encoder run: the GPT-2 prefill preset (the encoder plans come from its planner
    path), the plan's capture env (the load-bearing fields its blocks were cut under, e.g. the reactive ceiling 46),
    then its runtime switches. `graphs`: replay each block as CUDA graphs (exec.use_graphs; the first forward captures)."""
    import json
    from pathlib import Path
    from perseus import config as pcfg
    cfg = pcfg.preset("n32_gpt2_prefill")
    first = sorted(Path(plan).glob("block_*_placement.json"))[:1]
    if first:
        env = (json.loads(first[0].read_text()).get("capture_env") or {}).get("env") or {}
        cfg = cfg.with_env({k: str(v) for k, v in env.items() if v is not None})
    cfg = pcfg.plan_runtime(plan, "n32", base=cfg)
    cfg["exec.use_graphs"] = bool(graphs)
    return cfg


def _flat(xs):
    """The stage's ciphertexts in program order: a flat list, or [g][k] blocked rows row-major."""
    return [c for x in xs for c in (x if isinstance(x, (list, tuple)) else [x])]


class EncoderPlannedPrefill(PlannedPrefill):
    """`PlannedPrefill` for a bidirectional encoder with CCMM attention (PF_ATTN=ccmm). Subclasses give `_forward(rows)`
    (the twin's forward: (Xs, ch, rec)) and `_stages_attr` (the model method that lists the stages)."""
    _stages_attr = "prefill_stages"

    def __init__(self, model, plan_dir, config, keys, *, ccmm_kappa=(256.0, 16.0), ccmm_sweep=(), **kw):
        super().__init__(model, plan_dir, config, keys, **kw)
        # CCMM operating points (QK sites, PV sites): matmul.rs `CcmmRt::kappa`; `ccmm_sweep` measures them
        self.ccmm_kappa, self.ccmm_sweep = tuple(float(k) for k in ccmm_kappa), tuple(ccmm_sweep)
        self.ccmm = os.environ.get("PF_ATTN", "") == "ccmm"
        if self.ccmm:       # the calibration probe's eager ciphertexts: a token, 2 x 16 operands, 16 outputs
            self.pool = max(self.pool, 56)

    def _forward(self, rows):
        return self.model.prefill(rows)

    # ── setup (CPU) ──
    def record(self, rows):
        """`PlannedPrefill.record` for stages whose inputs / outputs are blocked [g][k] rows (flattened row-major)."""
        from perseus.impl.driver import Recorder
        t0 = time.time()
        m = self.model
        rec = Recorder(m, self.n_layers)
        orig = getattr(m, self._stages_attr)
        io = {}

        def stages(ch):
            out = []
            for i, (lab, pre, fn) in enumerate(orig(ch)):
                def wrap(x, i=i, fn=fn):
                    flat = _flat(x)
                    metas = [(int(c.level), int(c.noise_deg)) for c in flat]
                    vals = [np.array(c.vec) for c in flat] if i == 0 else None
                    y = fn(x)
                    io[i] = ([c.name for c in flat], [c.name for c in _flat(y)], metas, vals)
                    return y
                out.append((lab, pre, wrap))
            return out
        setattr(m, self._stages_attr, stages)
        self.rows = np.asarray(rows, dtype=np.float64)
        try:
            _xs, self.ch, _ = self._forward(self.rows)
        finally:
            delattr(m, self._stages_attr)
        self.in_metas, self.in_names = {}, {}
        for i, r in rec.blocks.items():
            names_in, names_out, metas, vals = io[i]
            r.out_names = names_out
            self.in_names[i] = names_in
            self.in_metas[i] = metas
            if i == 0:
                self.in_vals = vals
        self.recs = rec.blocks
        self.record_s = time.time() - t0
        return self.recs

    def prepare(self, rows):
        """`record` + the programs compiled block by block: block b > 0 starts from block b - 1's compiled output
        metadata (the plan refreshes a block's outputs, landing them above the twin's unplanned levels)."""
        import json
        from perseus import _pc
        if not hasattr(self, "recs"):
            self.record(rows)
        t0 = time.time()
        os.makedirs(self.work_dir, exist_ok=True)
        self.progs, self.in_pos = {}, {}
        prev = None
        for b in range(self.n_layers):
            r, lab = self.recs[b], self.labels[b]
            d = os.path.join(self.work_dir, f"block_{b}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "graph.json"), "w") as f:
                json.dump({"version": 2, "nodes": r.nodes}, f)
            names = self.in_names[b]
            ins = r.inputs()
            missing = [n for n in ins if n not in names]
            if missing:
                raise RuntimeError(f"block {b}: inputs {missing} are not the stage's input ciphertexts {names}")
            self.in_pos[b] = [names.index(n) for n in ins]
            if prev is not None:        # stage input i = the previous stage's output i
                p = self.progs[self.labels[b - 1]]
                got = {n: (int(lv), int(dg)) for n, lv, dg, _s in p.outputs}
                self.in_metas[b] = [got[n] for n in self.recs[b - 1].out_names]
            metas = {lab: {n: self.in_metas[b][names.index(n)] for n in ins}}
            self.progs.update(_pc.Program.compile(self.work_dir, self.plan_dir, config=self.cfg, blocks=[lab],
                                                  live_out={lab: list(r.out_names)}, input_metas=metas,
                                                  schedule=self.schedule))
            prev = b
        self.reg, n = {}, 0
        for b in range(self.n_layers):
            p = self.progs[self.labels[b]]
            self.reg[b] = []
            for o in self.recs[b].out_names:
                self.reg[b].append((p.output_index(o), n))
                n += 1
        self.n_regs = n
        self.compile_s = time.time() - t0
        return self

    def setup(self, rows):
        """`prepare` (CPU) + the planned Session with the PCMM and CCMM services; every block bound, plaintexts staged
        by value, the CCMM calibrated, then the secret dropped."""
        import hashlib
        from perseus import _pc
        from perseus.impl.driver import plan_rot_scope
        if not hasattr(self, "progs"):
            self.prepare(rows)
        t0 = time.time()
        progs = [self.progs[lab] for lab in self.labels]
        self.alias, uniq = {}, {}
        for b, p in enumerate(progs):
            r, lab = self.recs[b], self.labels[b]
            for nm, lv, _d, form in p.pt_refs:
                v = np.ascontiguousarray(r.pts[nm])
                key = "v/" + hashlib.blake2b(v.tobytes(), digest_size=12).hexdigest()
                store = f"{lab}/{nm}" if (form == "at_use" or nm.startswith("pt_")) else nm
                self.alias.setdefault(lab, {})[store] = key
                uniq.setdefault((key, lv), v)
        n_pts = len(uniq)
        pt_rows = self.pt_rows and max(self.pt_rows, _pc.chain_limbs(self.cfg["chain.preset"]) - min(lv for _k, lv in uniq))
        lins = self.linears()
        t = self.model.rt.dims.t
        specs = session_specs(lins.values(), t)
        self.rot_keys = plan_rot_scope(self.plan_dir, progs)
        from perseus.impl.driver import register_levels
        reg_levels = register_levels(self.n_regs, [(progs[b], j, r) for b in range(self.n_layers) for j, r in self.reg[b]])
        self.sess = _pc.Session(self.cfg, self.keys, pool=self.pool, pt_slots=n_pts + 8, programs=progs,
                                regs=self.n_regs, pt_rows=pt_rows, stc_cache=self.stc_cache,
                                pcmm=specs or None, pcmm_cf=self.pcmm_cf, check_meta=self.check_meta, ccmm=self.ccmm,
                                reg_levels=reg_levels, pt_levels=[lv for _k, lv in uniq])
        self.store = self.sess.pt_store
        for b, p in enumerate(progs):
            p.bind(self.sess, self.store)
            p.alias_pts(self.alias.get(self.labels[b], {}))
        self.sess.complete_setup()
        self.fhe = self.sess.fhe
        for lin in lins.values():
            self.sess.pcmm_upload(lin.name, np.ascontiguousarray(lin.Wd, dtype=np.int8).tobytes())
        if self.ccmm:
            self.calibrate_ccmm()
        self.sess.drop_secret()
        staged = self.store.stage([(key, lv, v) for (key, lv), v in uniq.items()])
        n_refs = sum(len({(nm, lv) for nm, lv, _d, _f in p.pt_refs}) for p in progs)
        self.setup_s = time.time() - t0
        self.log(f"[planned] {len(progs)} programs, {self.n_regs} registers, {n_refs} plaintext keys = {n_pts} distinct "
                 f"(values, level) ({staged} staged), {len(specs)} PCMM linears, rotation keys {self.rot_keys[0]}; "
                 f"record {self.record_s:.1f} s, compile {self.compile_s:.1f} s, session "
                 f"{self.sess.setup_seconds()[0]:.1f} s (keygen {self.sess.setup_seconds()[1]:.1f}), setup total "
                 f"{self.setup_s:.1f} s")
        self.log("[planned] " + self.sess.mem_report.replace("\n", " | "))
        return self

    def calibrate_ccmm(self):
        """ccmm.py `prepare` + `_calibrate` on the session's CCMM service: nf_ref / nf_split from a token at the
        block-0 input level, c_unit from known probe matrices (shift 0, factors 2^-3), then the operating points."""
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
                         "land": land}
        self.log(f"[ccmm] calibrated: c_unit {c_unit:.6g} (probe {np.log2(1 / max(err, 1e-300)):.1f} bits), nf_ref "
                 f"2^{np.log2(nf_ref):.3f}, nf_split 2^{np.log2(nf_split):.3f}, landing {land} (twin "
                 f"{fake_land}), kappa QK {self.ccmm_kappa[0]:g} PV {self.ccmm_kappa[1]:g}, "
                 f"{time.time() - t0:.1f} s")
        if land[0] != fake_land:
            raise RuntimeError(f"CCMM lands at level {land[0]} on the device, the CPU twin recorded {fake_land} "
                               f"(ccmm.fake_landing)")
        for kappa in self.ccmm_sweep:
            for case in ("qk", "pv"):
                bits = self.ccmm_case(case, kappa, L)
                self.log(f"[ccmm] sweep {case} kappa 2^{np.log2(kappa):.1f}: {bits:.2f} bits")

    def ccmm_case(self, case, kappa, L, fill=1.0):
        """Precision (bits: max |want| / max err) of one merged CCMM at this encoder's operating point
        (impl/ccmm_mha.py constants): "qk" (Q complex at |Q| <= QMAX, K real at KMAX, b transposed, out_scale
        1/sqrt(64), SHIFT_QK) or "pv" (P >= 0 at 1.5, V real at VMAX, SHIFT_PV); `fill` scales the operands below
        their bounds (the input-range check: the SinCBoot input grows with fill^2)."""
        from perseus.impl import ccmm as CC
        from perseus.impl import ccmm_mha as MH
        from perseus.impl import sinc as sc
        fhe, sess = self.fhe, self.sess
        rng = np.random.default_rng(7)
        D = sc.D
        A = np.zeros((D, D, sc.BATCH), complex)
        Bm = np.zeros((D, D, sc.BATCH))
        if case == "qk":
            amax, bmax, osc, shift, bt = MH.QMAX, MH.KMAX, 1 / 8.0, MH.SHIFT_QK, True
            A[:, :, :16] = rng.standard_normal((D, D, 16)) + 1j * rng.standard_normal((D, D, 16))
        else:
            amax, bmax, osc, shift, bt = 1.5, MH.VMAX, 1.0, MH.SHIFT_PV, False
            A[:, :, :16] = rng.random((D, D, 16)) ** 4
        A[:, :, :16] *= fill * amax / np.abs(A[:, :, :16]).max()
        Bm[:, :, :16] = rng.standard_normal((D, D, 16))
        Bm[:, :, :16] *= fill * bmax / np.abs(Bm[:, :, :16]).max()
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
        sel = ((np.arange(sc.SLOTS) - sc.OFFSET) % 32) < 16
        g, w = got[:, sel], want[:, sel]
        return float(np.log2(np.abs(w).max() / max(np.abs(g - w).max(), 1e-300)))

    def _rows_in(self, rows):
        """The rows as the model's forward packs them (ViT pads to whole pages)."""
        return rows

    def input_values(self, rows):
        """Block 0's input slot vectors for another input of the recorded length (blocked layout, the recorded
        chunks); checked against the recorded input's own."""
        from perseus.impl.blocked import blocked_slots

        def pack(r):
            return [v for row in blocked_slots(self._rows_in(r), self.ch, self.model.blocks) for v in row]
        if not hasattr(self, "_rows_ok"):
            mine = pack(self.rows)
            if len(mine) != len(self.in_vals) or not all(np.allclose(a, b, atol=1e-9) for a, b in zip(mine, self.in_vals)):
                raise RuntimeError("input_values: the packing does not reproduce the recorded block-0 inputs")
            self._rows_ok = True
        if np.asarray(rows).shape != self.rows.shape:
            raise ValueError(f"input rows {np.asarray(rows).shape}: the programs were compiled for {self.rows.shape}")
        return pack(np.asarray(rows, dtype=np.float64))


class BertPlannedPrefill(EncoderPlannedPrefill):
    """BERT-base: `n_keys` real tokens (the rest [PAD], masked as keys) fix the recorded stream."""
    _stages_attr = "stages"

    def __init__(self, model, plan_dir, config, keys, *, n_keys, **kw):
        super().__init__(model, plan_dir, config, keys, **kw)
        self.n_keys = int(n_keys)

    def _forward(self, rows):
        return self.model.forward(rows, self.n_keys)

    def linears(self):
        from perseus.impl.pcmm import PcmmLinear
        out = {}
        for b in range(self.n_layers):
            for lin in self.model.weights(b).values():
                if isinstance(lin, PcmmLinear):
                    out[lin.name] = lin
        return out
