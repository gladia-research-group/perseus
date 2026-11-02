"""Device graph capture (fideslib_wrapper.h record_primitive) replayed on the numpy fake: no GPU.

The recorder of the old fake capture, with its bookkeeping moved to perseus: levels and degrees from
``_pc.Shadow`` in "fakecap" mode (the C++ rules + the fake capture's quirks ``AddPtRealizes``,
``CapStcEntryNominal``, ``CapMultPtMax``; raw degree-2 landings from ``quirks.fakecap_landings``) and the pack-tag
algebra from ``_pc.tag`` (pc-shadow's 1:1 port). It writes graph.json version 2: the v1 node fields the planner reads
(op_type, inputs, input_levels, output, step, output_level, output_noise_level, pack_*, the magnitudes) plus, per
node, ``tag`` (the full pack tag), ``shadow`` {level, deg, nf} and ``args`` for the ops whose parameters the v1 names
only hint at (rotations, plaintext operands, hints, refreshes and their routes, level caps, external finishes).
Each graph dir also gets ``capture_config.json`` (the typed Config stamp) next to ``capture_env.json``.
"""
from __future__ import annotations

import contextlib
import json
import os
import re

import numpy as np

from perseus import _pc

from .fake import FakeCt, FakeFhe, FakeInf, _Core

T = _pc.tag
Tag = _pc.Tag
EMPTY, AP, DENSE = T.EMPTY, T.AP, T.DENSE
F_UNKNOWN, F_REAL, F_IMAG, F_COMPLEX = T.F_UNKNOWN, T.F_REAL, T.F_IMAG, T.F_COMPLEX

GRAPH_VERSION = 2


def pt_tag(values):
    """tag_plaintext(analyze_packing(values)) (include/packing/pack_signature.h): the measured signature -> Tag."""
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
    t = T.from_signature(n, period, live, stride, window, offset)
    if cplx:
        r, i = bool((re_ != 0).any()), bool((im != 0).any())
        t.field = (F_COMPLEX if r else F_IMAG) if i else F_REAL
    else:
        t.field = F_REAL
    return t


def tag_json(tg):
    sp = tg.support
    return {"slots": int(tg.slots), "period": int(tg.period), "kind": int(sp.kind), "offset": int(sp.offset),
            "stride": int(sp.stride), "count": int(sp.count), "width": int(sp.width), "field": int(tg.field)}


# ── the recorder ───────────────────────────────────────────────────────────────────────────────────────────────────

def _fmt_scalar(x):
    """std::ostream << double (precision 6, %g)."""
    return f"const({float(x):g})"


_uids = iter(range(1, 1 << 62))


def _uid(ct):
    """The ciphertext id a conjugate's tag points at (the old tag held the object and compared identities)."""
    u = getattr(ct, "uid", None)
    if u is None:
        u = ct.uid = next(_uids)
    return u


class CapCt(FakeCt):
    """A recorded ciphertext: ``tag`` is the Ciphertext's own tag (``meta.tag``), ``ptag`` the PackedCtx's.
    ``CapCt.novec``: keep one slot of every ciphertext value (the planned driver's recorder needs the op stream and
    the plaintexts, not the message; ~4x faster)."""
    novec = False

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        if CapCt.novec and self.vec.shape[0] > 1:
            self.vec = self.vec[:1]


class Pt:
    """Plaintext named afresh by identity (device pt_N names can go stale and carry no plan semantics).
    `src` = (kind, name) for the graph's args: ("values"|"const"|"slot"|"cached", name or None)."""
    __slots__ = ("vec", "level", "deg", "tag", "name", "gen", "src")

    def __init__(self, vec, level, deg=1, tag=None, src=("values", None)):
        self.vec, self.level, self.deg = vec, int(level), int(deg)
        self.tag = tag if tag is not None else Tag()
        self.name, self.gen, self.src = None, -1, src

    def meta(self):
        return _pc.PtMeta(self.level, self.deg, 0.0, self.tag)


def parse_landings(spec):
    """route s -> (level, deg) of a refresh landing; 0 = dense (FAKECAP_LANDINGS)."""
    out = {}
    for kv in spec.split(","):
        s, lv = kv.split(":")
        out[int(s)] = (int(lv), 2)
    return out


def session_config(env=None):
    """The capture session's Config from the legacy env (the bridge until the examples build a Config)."""
    return _pc.Config.from_legacy_env(dict(os.environ) if env is None else env)


class CapFhe(FakeFhe):
    """FakeFhe with the device recorder's bookkeeping."""
    MODE = "fakecap"

    def __init__(self, *a, config=None, **k):
        cfg = config if config is not None else session_config()
        super().__init__(*a, config=cfg, **k)
        self.gen = 0
        self.nodes = None                  # list while a block is captured
        self.ct_counter = self.pt_counter = self.anon_counter = 0
        self.graph_scope, self.graph_scope_counter, self.graph_scope_strict = "", 0, False
        self.steps = []
        self.landing = parse_landings(cfg["quirks.fakecap_landings"] or "0:32,1:14,256:22,512:22")
        self.sparse_mode = cfg["bts.sparse_auto"] or 0
        self.sparse_precomp_slots = sorted({int(x) for x in (cfg["bts.sparse_bts_slots"] or []) if x})
        self.pcmm_landing = None           # (level, deg) override of a finish's drop-0 landing (None: the Shadow's)
        self.stc_drop = 6 if cfg["quirks.fakecap_stc_drop"] is None else cfg["quirks.fakecap_stc_drop"]
        self.stc_entry = 46 if cfg["quirks.fakecap_stc_entry"] is None else cfg["quirks.fakecap_stc_entry"]
        self._stc_first_cfg = ((cfg["bts.stc_first"] or 0) > 0 and not cfg["bts.stc_first_staged_only"]
                               and "dense" in (cfg["bts.stc_first_routes"] or ["dense"]))
        self._sparse_exact = bool(cfg["quirks.fakecap_sparse_exact"])
        self._check_field = bool(cfg["exec.debug.fakecap_check_field"])
        self.stats = True
        self._pend, self._batch = [], cfg["quirks.fakecap_batch"] or 1
        self.total_bootstraps = 0
        self.magnitude_suppressed = False

    # ── names (ct_vars / pt_vars keyed by identity, cleared per block) ──
    def var(self, ct):
        if ct.gen == self.gen and ct.name:
            return ct.name
        self.anon_counter += 1
        ct.name, ct.gen = f"anon_{self.anon_counter}", self.gen
        return ct.name

    def new_var(self, ct):
        if self.graph_scope:
            self.graph_scope_counter += 1
            ct.name = f"{self.graph_scope}_{self.graph_scope_counter}"
        else:
            self.ct_counter += 1
            ct.name = f"v_{self.ct_counter}"
        ct.gen = self.gen
        return ct.name

    def name_ct(self, ct, name, overwrite=True):
        if not name:
            return
        if not overwrite and ct.gen == self.gen and ct.name:
            return
        ct.name, ct.gen = name, self.gen

    def pt_var(self, pt):
        if pt.gen == self.gen and pt.name:
            return pt.name
        self.pt_counter += 1
        pt.name, pt.gen = f"pt_{self.pt_counter}", self.gen
        return pt.name

    def tag_ct(self, ct, tag):
        self.var(ct)                       # tag_ct keys its registry entry by var_for_ct (an anon name if unnamed)
        ct.tag = tag

    def path(self):
        return ".".join(self.steps)

    def record(self, op, ins, out, in_levels, out_ct=None, out_level=None, args=None):
        if self.nodes is None:
            return
        n = {"op_type": op, "inputs": list(ins), "input_levels": [int(x) for x in in_levels], "output": out,
             "step": self.path(), "output_level": int(out_ct.level if out_ct is not None else out_level)}
        if args:
            n["args"] = args
        if out_ct is not None:
            n["output_noise_level"] = int(out_ct.noise_deg)
            tg = out_ct.tag
            if self._check_field and tg.field in (F_REAL, F_IMAG) and np.iscomplexobj(out_ct.vec):
                v = np.asarray(out_ct.vec)
                bad = np.abs(v.imag if tg.field == F_REAL else v.real).max()
                if bad > 1e-6 * max(float(np.abs(v).max()), 1e-30):
                    print(f"[fakecap] field tag {tg.field} unsound: {op} {out} @ {self.path()} "
                          f"off-axis {bad:.3g} of {float(np.abs(v).max()):.3g}", flush=True)
            if self.stats and not self.magnitude_suppressed:
                self._pend.append((n, np.array(out_ct.vec, dtype=np.complex128)))
                if len(self._pend) >= self._batch:
                    self.flush_stats()
            if tg.known():
                self._stamp(n, tg)
            if tg.field:
                n["pack_field"] = int(tg.field)
            n["tag"] = tag_json(tg)
            m = out_ct.meta
            n["shadow"] = {"level": int(m.level), "deg": int(m.deg), "nf": float(m.nf)}
        self.nodes.append(n)

    def flush_stats(self):
        """Pending nodes' magnitudes (magtape.node_stats), one FFT per batch."""
        if self._pend:
            from .magtape import node_stats_batch
            for (n, _), st in zip(self._pend, node_stats_batch(np.stack([v for _, v in self._pend]))):
                n.update(st)
                ext = n.get("args", {}).get("external")
                if isinstance(ext, dict) and "range" in ext:     # a TFHE finish: its table's range (impl/tfhe.py)
                    n["output_max_abs"] = max(float(n.get("output_max_abs", 0.0)), float(ext["range"]))
            self._pend = []

    @staticmethod
    def _stamp(n, tg):
        sp = tg.support
        n.update(pack_period=int(tg.period), pack_kind=int(sp.kind), pack_offset=int(sp.offset),
                 pack_stride=int(sp.stride), pack_count=int(sp.count), pack_width=int(sp.width))
        if tg.field:
            n["pack_field"] = int(tg.field)

    def restamp(self, ct):
        tg = ct.tag
        if self.nodes is None or not tg.known():
            return
        name = self.var(ct)
        for n in reversed(self.nodes):
            if n["output"] == name:
                self._stamp(n, tg)
                n["tag"] = tag_json(tg)
                return

    # ── levels ──
    def eff(self, ct):
        return ct.meta.effective_level(self.unit)

    # ── refreshes ──
    def auto_route(self, ct):
        if self.sparse_mode <= 0 or not self.sparse_precomp_slots:
            return 0
        t = ct.tag
        if not t.known():
            return 0
        S = self.slots
        if self.sparse_mode >= 2:
            want = t.min_routable_s()
        else:
            want = 1
            while want < t.period:
                want <<= 1
        s = self.fold_slots_for(want)
        if s >= S or s not in self.sparse_precomp_slots:
            return 0
        if not t.periodic_at(s):
            if self.sparse_mode >= 3:
                raise NotImplementedError("fakecap: SPARSE_AUTO=3 fold route")
            return 0
        return s

    def stc_first(self):
        """Whether a dense refresh now takes the StC-first order (bts.stc_first*, inside an StcFirst scope)."""
        if not self._stc_first_cfg:
            return False
        from . import bootstrap as _pyb
        return bool(_pyb._stc_first["on"])

    def bootstrap_output_level(self):
        return self.landing[0][0] - (self.stc_drop if self.stc_first() else 0)

    def inner_bootstrap(self, ct):
        """Refresh in place on the auto-routed slot count; returns the route's args for the graph."""
        s = self.auto_route(ct)
        if s and self._sparse_exact and not CapCt.novec:       # a sparse route returns the s-periodic projection
            v = np.asarray(ct.vec).reshape(-1, s).mean(0)
            ct.vec = np.tile(v, self.slots // s)
        stc = s == 0 and self.stc_first()
        # a deliberate refresh under a raise variant (PF_SM_R_RAISE, CapFhe.raise_drop) lands that much deeper
        drop = int(getattr(self, "raise_drop", 0) or 0) if s == 0 else 0
        route = _pc.Route(slots=s, stc_first=stc, raise_drop=drop)
        tag = T.t_bootstrap(ct.tag, s, self.slots)
        ct.meta = self.shadow.bootstrap(ct.meta, route).out.replace(tag=tag)
        self.n_bootstraps += 1
        self.total_bootstraps += 1
        return {"route": {"slots": s, "stc_first": stc}}

    def maybe_bootstrap(self, ct):
        if self.auto_bts_suppressed or ct.level < self._limit:
            return
        lin, nin = ct.level, self.var(ct)
        args = self.inner_bootstrap(ct)
        self.n_reactive += 1
        out = nin + "_auto_bootstrapped"
        self.name_ct(ct, out)
        self.record("auto_bootstrap", [nin], out, [lin], ct, args=args)

    def bootstrap(self, a, iterations=1):
        lin, nin = a.level, self.var(a)
        args = self.inner_bootstrap(a)
        if iterations != 1:
            args["iterations"] = int(iterations)
        out = self.new_var(a)
        self.record("deliberate_bootstrap", [nin], out, [lin], a, args=args)

    def bootstrap_real(self, a, iterations=1):
        """`bootstrap` of a payload the code knows is real (FheOps.bootstrap_real): recorded with
        args.route.real so the compiled program takes the real-payload route there."""
        lin, nin = a.level, self.var(a)
        args = self.inner_bootstrap(a)
        args["route"]["real"] = True
        if iterations != 1:
            args["iterations"] = int(iterations)
        out = self.new_var(a)
        self.record("deliberate_bootstrap", [nin], out, [lin], a, args=args)

    def bootstrap_hint(self, a, thr, account_pending_rescale=False, iterations=1):
        lin, nin = a.level, self.var(a)
        args = {"thr": int(thr), "acct": bool(account_pending_rescale)}
        if self.shadow.hint(a.meta, int(thr), bool(account_pending_rescale)).fired:
            args.update(self.inner_bootstrap(a))
            a.meta = self.shadow.realize(a.meta).out      # inner_bootstrap_planned's landing realize
        args["fired"] = "route" in args
        out = self.new_var(a)
        self.record("hint", [nin, f"hint_lev({int(thr)})"], out, [lin, -1], a, args=args)

    def level_hint(self, a, level):
        lin, nin = a.level, self.var(a)
        a.meta = self.shadow.hint_level(a.meta).out
        out = self.new_var(a)
        self.record("level_hint", [nin, f"lvl_cap({int(level)})"], out, [lin, -1], a, args={"cap": int(level)})

    def drop_to_level(self, a, level):
        lin, nin = a.level, self.var(a)
        a.meta = self.shadow.drop_to_level(a.meta, int(level)).out
        out = self.new_var(a)
        self.record("level_reduce", [nin], out, [lin], a, args={"target": int(level)})

    def record_external_bootstrap(self, a, inputs, input_levels, landing=None, desc=None):
        """A refresh computed outside the recorded primitives (the PCMM / CCMM finishes): `desc` names it. It lands
        where the Shadow says (`finish_landing`: the StC-first route under desc's raise drop, + the CCMM's StC
        stage), what the runtime pins and its device run lands at; `landing` / the session's `pcmm_landing` override
        the drop-0 landing (the drop still applies)."""
        self.n_bootstraps += 1
        self.total_bootstraps += 1
        drop = int(desc.get("raise_drop", 0) or 0) if isinstance(desc, dict) else 0
        op = desc.get("op") if isinstance(desc, dict) else None
        base = landing if landing is not None else self.pcmm_landing
        if base is not None:
            land = (int(base[0]) + self.unit * drop, int(base[1]))
        else:
            land = self.shadow.finish_landing(op if op in ("pcmm", "ccmm") else "pcmm", drop)
        if land is not None:
            a.meta = a.meta.replace(level=int(land[0]), deg=int(land[1]))
        out = self.new_var(a)
        a.tag = Tag()
        args = {"external": desc or "finish", "n_inputs": len(inputs)}
        if land is not None:
            args["landing"] = [int(land[0]), int(land[1])]
        self.record("deliberate_bootstrap", list(inputs), out, list(input_levels), a, args=args)

    def var_name(self, a):
        return self.var(a)

    def fold_bootstrap(self, a, s, n_live, prescale=1.0):
        raise NotImplementedError("fakecap: fold_bootstrap")

    # ── arithmetic ──
    def _binary(self, a, vec, recipe, tag, rec_op, b_name, b_level, product=False, args=None):
        """`product`: the op ends in maybe_bootstrap (the products, negate, and ct - ct)."""
        la, na = a.level, self.var(a)
        out_ct = CapCt(vec, recipe.out, packing=a.packing)
        self.tag_ct(out_ct, tag)
        out = self.new_var(out_ct)
        self.record(rec_op, [na, b_name], out, [la, b_level], out_ct, args=args)
        if product:
            self.maybe_bootstrap(out_ct)
        out_ct.tag = tag
        return out_ct

    def _unary(self, a, vec, recipe, tag, rec_op, product=False, when="tag_ct", b=None, args=None):
        """`when` the output gets its tag, as the C++ recorder sees it: "tag_ct" (registered before naming),
        "named" (after naming, before recording), "after" (recorded untagged: the PackedCtx overlay tags it)."""
        la, na = a.level, self.var(a)
        out_ct = CapCt(vec, recipe.out.replace(tag=Tag()), packing=a.packing)
        if when == "tag_ct":
            self.tag_ct(out_ct, tag)
        out = self.new_var(out_ct)
        if when == "named":
            out_ct.tag = tag
        ins, lvs = ([na], [la]) if b is None else ([na, b], [la, -1])
        self.record(rec_op, ins, out, lvs, out_ct, args=args)
        if product:
            self.maybe_bootstrap(out_ct)
        out_ct.tag = tag
        return out_ct

    @staticmethod
    def _pt_args(pt):
        kind, name = pt.src
        d = {"pt": {"kind": kind, "level": pt.level, "deg": pt.deg}}
        if name is not None:
            d["pt"]["name"] = name
        return d

    def _c_add(self, a, b):
        if isinstance(b, FakeCt):
            ta, tb = a.tag, b.tag
            tag = T.conj_pair(T.t_add(ta, tb), _uid(a), _uid(b), ta, tb, False)
            nb = self.var(b)
            return self._binary(a, a.vec + b.vec, self.shadow.add(a.meta, b.meta), tag, "add", nb, b.level)
        if isinstance(b, Pt):
            return self._add_pt(a, b, a.vec + b.vec)
        c = float(b)
        return self._binary(a, a.vec + c, self.shadow.add_scalar(a.meta, c), T.t_add_scalar(a.tag, c), "add",
                            _fmt_scalar(c), -1, args={"scalar": c})

    def _add_pt(self, a, pt, vec, op="add"):
        tag = T.t_add(a.tag, pt.tag)
        return self._binary(a, vec, self.shadow.add_pt(a.meta, pt.meta()), tag, op, self.pt_var(pt), pt.level,
                            args=self._pt_args(pt))

    def _c_sub(self, a, b):
        if isinstance(b, FakeCt):
            ta, tb = a.tag, b.tag
            tag = T.conj_pair(T.t_add(ta, tb), _uid(a), _uid(b), ta, tb, True)
            nb = self.var(b)
            return self._binary(a, a.vec - b.vec, self.shadow.sub(a.meta, b.meta), tag, "sub_ct", nb, b.level,
                                product=True)
        c = float(b)
        return self._binary(a, a.vec - c, self.shadow.add_scalar(a.meta, -c), T.t_add_scalar(a.tag, c), "sub",
                            _fmt_scalar(c), -1, args={"scalar": c})

    def _c_mult(self, a, b):
        if isinstance(b, FakeCt):
            nb = self.var(b)
            return self._binary(a, a.vec * b.vec, self.shadow.mult(a.meta, b.meta), T.t_mult(a.tag, b.tag), "mult",
                                nb, b.level, product=True)
        if isinstance(b, Pt):
            return self._binary(a, a.vec * b.vec, self.shadow.mult_pt(a.meta, b.meta()), T.t_mult(a.tag, b.tag),
                                "mult", self.pt_var(b), b.level, product=True, args=self._pt_args(b))
        c = float(b)
        return self._binary(a, a.vec * c, self.shadow.mult_scalar(a.meta, c), T.t_mult_scalar(a.tag, c), "mult",
                            _fmt_scalar(c), -1, product=True, args={"scalar": c})

    def _c_square(self, a):
        return self._unary(a, a.vec * a.vec, self.shadow.square(a.meta), T.t_square(a.tag), "square", product=True)

    def _c_negate(self, a):
        return self._unary(a, -a.vec, self.shadow.negate(a.meta), T.t_mult_scalar(a.tag, -1.0), "negate",
                           product=True)

    def inplace_add(self, a, b):
        ta, tb = a.tag, b.tag
        tag = T.conj_pair(T.t_add(ta, tb), _uid(a), _uid(b), ta, tb, False)
        la, lb, na, nb = a.level, b.level, self.var(a), self.var(b)
        a.meta = self.shadow.add(a.meta, b.meta).out
        a.vec = a.vec + b.vec
        a.tag = tag
        out = self.new_var(a)
        self.record("add_inplace", [na, nb], out, [la, lb], a)
        a.tag = tag
        a.ptag = tag

    def _check_rot(self, steps):
        if self.slots is not None:
            k = steps % self.slots
            if k and k not in self._rot and (k - self.slots) not in self._rot and steps not in self._rot:
                raise KeyError(f"no rotation key for {steps}")

    def _c_rotate(self, a, steps):
        steps = int(steps)
        self._check_rot(steps)
        tag = T.t_rotate(a.tag, steps)
        return self._unary(a, np.roll(a.vec, -steps), self.shadow.rotate(a.meta, steps), tag, "rotate",
                           when="after", b=f"rot({steps})", args={"steps": [steps]})

    def _c_rotate_hoisted(self, a, steps):
        la, na = a.level, self.var(a)
        outs = []
        for k in steps:
            o = CapCt(np.roll(a.vec, -int(k)), self.shadow.rotate(a.meta, int(k)).out.replace(tag=Tag()),
                      packing=a.packing)
            out = self.new_var(o)
            self.record("rotate", [na, f"rot({int(k)})"], out, [la, -1], o, args={"steps": [int(k)]})
            outs.append(o)
        for k, o in zip(steps, outs):      # the PackedCtx overlay tags each after the loop
            o.tag = T.t_rotate(a.tag, int(k))
        return outs

    def _c_rotate_sum(self, a, steps):
        if not self.has_rot_keys(steps):
            raise KeyError(f"no rotation keys for {steps}")
        steps = [int(k) for k in steps]
        r = self.shadow.rotate_sum(a.meta, steps)
        vec = a.vec.copy()
        for k in steps:
            vec = vec + np.roll(a.vec, -k)
        return self._unary(a, vec, r, r.out.tag, "rotate_sum", b="rot(" + ",".join(str(k) for k in steps) + ")",
                           args={"steps": steps})

    def rotate_and_sum(self, a, start, stop):
        from .layout import ladder_radix_bits
        sign = -1 if start < 0 else 1
        gap = abs(int(start))
        if gap >= stop:
            return self.add(a, 0.0)
        bits = ladder_radix_bits()
        if bits >= 2:
            gaps = []
            g = gap
            while g < stop:
                gaps.append(g)
                g *= 2
            acc, have = a, False
            for i0 in range(0, len(gaps), bits):
                n = min(bits, len(gaps) - i0)
                steps = [sign * j * gaps[i0] for j in range(1, 1 << n)]
                rungs = [sign * gaps[i0 + r] for r in range(n)]
                if n > 1 and self.has_rot_keys(steps):
                    acc = self.rotate_sum(acc if have else a, steps)
                    have = True
                    continue
                for g in rungs:
                    if not have:
                        acc = self.add(a, self.rotate(a, g))
                        have = True
                    else:
                        self.inplace_add(acc, self.rotate(acc, g))
            return acc
        acc = self.add(a, self.rotate(a, sign * gap))
        gap *= 2
        while gap < stop:
            self.inplace_add(acc, self.rotate(acc, sign * gap))
            gap *= 2
        return acc

    def _c_conjugate(self, a):
        tag = T.t_conjugate(a.tag)
        tag.conj_of = _uid(a)
        vec = np.conj(a.vec) if self.complex_payload else a.vec.copy()
        return self._unary(a, vec, self.shadow.conjugate(a.meta), tag, "conjugate")

    def _c_mult_i(self, a):
        return self._unary(a, a.vec * 1j, self.shadow.mult_i(a.meta), T.t_mult_i(a.tag), "mult_i", when="named")

    def mult_add_many(self, acc, vs, ss):
        for v, t in zip(vs, ss):
            self.inplace_add(acc, self.mult(v, t))
        return False

    # ── PackedCtx overlay: Ctx op, then retagged from the handles' own tags ──
    def _tagged(self, out, tag):
        out.tag = tag
        out.ptag = tag
        return out

    def add(self, a, b):
        if isinstance(b, FakeCt):
            return self._tagged(self._c_add(a, b),
                                T.conj_pair(T.t_add(a.ptag, b.ptag), _uid(a), _uid(b), a.ptag, b.ptag, False))
        if isinstance(b, Pt):
            return self._tagged(self._c_add(a, b), T.t_add(a.ptag, b.tag))
        return self._tagged(self._c_add(a, b), T.t_add_scalar(a.ptag, float(b)))

    def sub(self, a, b):
        if isinstance(b, FakeCt):
            return self._tagged(self._c_sub(a, b),
                                T.conj_pair(T.t_add(a.ptag, b.ptag), _uid(a), _uid(b), a.ptag, b.ptag, True))
        return self._tagged(self._c_sub(a, b), T.t_add_scalar(a.ptag, float(b)))

    def mult(self, a, b):
        if isinstance(b, FakeCt):
            return self._tagged(self._c_mult(a, b), T.t_mult(a.ptag, b.ptag))
        if isinstance(b, Pt):
            return self._tagged(self._c_mult(a, b), T.t_mult(a.ptag, b.tag))
        return self._tagged(self._c_mult(a, b), T.t_mult_scalar(a.ptag, float(b)))

    def square(self, a):
        return self._tagged(self._c_square(a), T.t_square(a.ptag))

    def negate(self, a):
        return self._tagged(self._c_negate(a), T.t_mult_scalar(a.ptag, -1.0))

    def rotate(self, a, steps):
        return self._tagged(self._c_rotate(a, steps), T.t_rotate(a.ptag, int(steps)))

    def rotate_hoisted(self, a, steps):
        return [self._tagged(o, T.t_rotate(a.ptag, int(k))) for o, k in zip(self._c_rotate_hoisted(a, steps), steps)]

    def rotate_sum(self, a, steps):
        o = self._c_rotate_sum(a, steps)
        return self._tagged(o, o.tag)

    def conjugate(self, a):
        t = T.t_conjugate(a.ptag)
        t.conj_of = _uid(a)
        return self._tagged(self._c_conjugate(a), t)

    def mult_i(self, a):
        return self._tagged(self._c_mult_i(a), T.t_mult_i(a.ptag))


class CapInf(FakeInf):
    """FakeInf with plaintext objects (identity-named, tagged, encoded at the ciphertext's level)."""

    def __init__(self, fhe, *a, **k):
        super().__init__(fhe, *a, **k)
        self.complex = True
        self.cache = {}
        self.const_pts = {}
        self.named = {}

    def step(self, label):
        fhe = self.fhe

        @contextlib.contextmanager
        def cm():
            fhe.steps.append(str(label))
            try:
                yield
            finally:
                fhe.steps.pop()
        return cm()

    def name_ct(self, ct, name):
        self.fhe.name_ct(ct, name)

    def name_ct_if_absent(self, ct, name):
        self.fhe.name_ct(ct, name, overwrite=False)

    def _like(self, a):
        return self.fhe.eff(a), 1

    def _pt(self, values, a, tagged=True):
        v = self._full(values)
        lv, dg = self._like(a)
        return Pt(v, lv, dg, pt_tag(v) if tagged else Tag())

    def mult_pt(self, a, values):
        return self.fhe.mult(a, self._pt(values, a))

    def add_pt(self, a, values):
        return self.fhe.add(a, self._pt(values, a))

    def mult_const(self, a, re_, im):
        lv = self.fhe.eff(a)
        key = (float(re_), float(im), lv)
        pt = self.const_pts.get(key)
        if pt is None:
            pt = self.const_pts[key] = Pt(np.full(self.slots, complex(re_, im)), lv, 1, Tag(),
                                          src=("const", f"{float(re_):g}{float(im):+g}j"))
        return self.fhe.mult(a, pt)

    def set_slot_pt(self, name, values, level=0, deg=1):
        v = self._full(values)
        lv = int(level) if level > 0 else self.fhe.bootstrap_output_level()
        self.named[name] = Pt(v, lv, deg, pt_tag(v), src=("slot", name))
        self.slot_pts[name] = v

    def _slot_pt(self, name, a):
        pt = self.named[name]
        lv, dg = self._like(a)
        if pt.level != lv:                  # slot_pt_at re-levels on a mismatch: a fresh, untagged plaintext
            pt = self.named[name] = Pt(pt.vec, lv, dg, Tag(), src=("slot", name))
        return pt

    def mult_slot_pt(self, a, name):
        return self.fhe.mult(a, self._slot_pt(name, a))

    def add_slot_pt(self, a, name):
        return self.fhe.add(a, self._slot_pt(name, a))

    def mult_slot_pt_many(self, a, names):
        return [self.mult_slot_pt(a, n) for n in names]

    def tag_reduce(self, a, stride):
        a.tag = T.t_reduce(a.tag, self.slots, int(stride))
        a.ptag = a.tag
        self.fhe.var(a)
        self.fhe.restamp(a)

    def _cached_pt(self, tag, a, values, tagged=True):
        lv, dg = self._like(a)
        key = f"{tag}#L{lv}:d{dg}"
        pt = self.cache.get(key)
        if pt is None:
            self.enc_misses += 1
            v = self._full(values)
            pt = self.cache[key] = Pt(v, lv, dg, pt_tag(v) if tagged else Tag(), src=("cached", tag))
        else:
            self.enc_hits += 1
        return pt

    def mult_cached(self, a, tag, values, tagged=True):
        return self.fhe.mult(a, self._cached_pt(tag, a, values, tagged))

    def add_cached(self, a, tag, values, tagged=True):
        return self.fhe.add(a, self._cached_pt(tag, a, values, tagged))

    def mult_cached_many(self, a, tags, values):
        return [self.mult_cached(a, t, v) for t, v in zip(tags, values)]

    def prime_pt(self, tag, values, level):
        key = f"{tag}#L{int(level)}:d1"
        if key not in self.cache:
            v = self._full(values)
            self.cache[key] = Pt(v, int(level), 1, pt_tag(v), src=("cached", tag))

    def adopt_pts(self, handle):
        n = 0
        for tag, level, values in handle:
            key = f"{tag}#L{int(level)}:d1"
            if key not in self.cache:
                self.prime_pt(tag, values, level)
                n += 1
        return n

    def erase_pts(self, tags):
        n = 0
        for tag in tags:
            ks = [k for k in self.cache if k.startswith(tag + "#L")]
            for k in ks:
                del self.cache[k]
            n += len(ks)
        return n

    def drop_slot_pts(self, prefix):
        ks = [k for k in self.named if k.startswith(prefix)]
        for k in ks:
            del self.named[k]
            self.slot_pts.pop(k, None)
        return len(ks)


def _write_config(graph_dir, cfg):
    with open(os.path.join(graph_dir, "capture_config.json"), "w", encoding="utf-8") as f:
        f.write(cfg.to_json())


class CapCore(_Core):
    """fake.core with the capture controls (begin / end_subgraph_capture write block_<b>/graph.json)."""

    @staticmethod
    def encode_token_input(inf, x):
        from .layout import lane_vec
        t = inf.slots // inf.size.hidDim
        x = np.asarray(x, dtype=np.float64).ravel()
        ct = CapCt(lane_vec(x, inf.slots, t), inf.fhe.bootstrap_output_level())
        ct.tag = ct.ptag = Tag(field=F_REAL)
        return ct

    @staticmethod
    def pack_tokens(inf, rows, target_level=0):
        t = inf.slots // inf.size.hidDim
        rows = np.asarray(rows, dtype=np.float64)
        v = np.zeros(inf.slots)
        for tok, row in enumerate(rows):
            v[np.arange(row.shape[0]) * t + tok] = row
        ct = CapCt(v, target_level or inf.fhe.bootstrap_output_level())
        ct.tag = ct.ptag = Tag(field=F_REAL)
        return ct

    @staticmethod
    def graph_dir():
        # set by ImplModel.set_capture after the session exists (the one env read left in the recorder)
        return os.environ.get("FHE_GRAPH_DIR")

    @staticmethod
    def begin_subgraph_capture(inf, b):
        d = CapCore.graph_dir()
        if not d or os.path.exists(os.path.join(d, f"block_{b}", "graph.json")):
            return False
        tok = os.environ.get("FHE_GRAPH_CAPTURE_TOKEN")
        if tok and getattr(inf, "capture_t", None) is not None and inf.capture_t != int(tok):
            return False
        if not os.path.exists(os.path.join(d, "capture_config.json")):
            os.makedirs(d, exist_ok=True)
            _write_config(d, inf.fhe.config)
        fhe = inf.fhe
        fhe.gen += 1
        fhe.nodes, fhe.ct_counter, fhe.pt_counter = [], 0, 0
        return True

    @staticmethod
    def end_subgraph_capture(inf, b):
        fhe = inf.fhe
        if fhe.nodes is None:
            return
        fhe.flush_stats()
        top = CapCore.graph_dir()
        d = os.path.join(top, f"block_{b}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "graph.json"), "w") as f:
            json.dump({"version": GRAPH_VERSION, "nodes": fhe.nodes}, f)
        import shutil
        for stamp in ("capture_env.json", "capture_config.json"):
            if os.path.exists(os.path.join(top, stamp)):
                shutil.copy(os.path.join(top, stamp), os.path.join(d, stamp))
        fhe.nodes = None


core = CapCore()

# the GPT-2 session's own band beyond the port's steps (lm-head tiles: 512 + 1024 k)
GPT2_BAND = frozenset(512 + 1024 * k for k in range(4, 32))


class FakeSession:
    def __init__(self, inf):
        self.inf = inf

    def close(self):
        pass


def session_env(env_defaults, overrides, chain="n32", complex_payload=True):
    """Export a device session's profile env, so a fake capture stamps the same contract."""
    from perseus.profile import SessionProfile
    prof = SessionProfile.gpt2_decode_n64() if chain == "n64" else SessionProfile.gpt2_decode_n32()
    prof.ckks_complex = bool(complex_payload)
    prof.extra = dict(prof.extra)
    prof.extra.update({k: v for k, v in env_defaults.items()})
    prof.extra.update({k: str(v) for k, v in overrides.items()})
    prof.apply()


def make_session(dims, rot_steps, limit=None, bts_out=32, unit=2, complex_payload=True, pcmm_landing=None,
                 config=None):
    """(fhe, inf) recording fakes at `dims` with the session's rotation keys. `config` (default: the legacy env
    through Config.from_legacy_env) sets the reactive ceiling (bts.auto_bts_level), the routes and the landings
    (`pcmm_landing`: override of the finishes' drop-0 landing; None = the Shadow's `finish_landing`)."""
    N = dims.N
    cfg = config if config is not None else session_config()
    limit = int(limit or cfg["bts.auto_bts_level"] or 46)
    cfg = cfg.copy()
    cfg["bts.auto_bts_level"] = limit
    cfg["chain.composite_degree"] = cfg["chain.composite_degree"] or int(unit)
    steps = {int(s) % N for s in rot_steps}
    steps = steps | {s - N for s in steps}
    sparse = cfg["bts.sparse_bts_slots"]
    if sparse is None:
        sparse = ["512", "1"]
    fhe = CapFhe(steps, limit=limit, bts_out=bts_out, unit=unit, slots=N, complex_payload=complex_payload,
                 sparse_slots=tuple(int(x) for x in sparse if x), config=cfg)
    fhe.pcmm_landing = pcmm_landing
    inf = CapInf(fhe, N, dims.hid, dims.dim, dims.H, dims.H_real, dims.E, dims.E_real)
    return fhe, inf


COMPARED = ("op_type", "inputs", "input_levels", "output", "step", "output_level", "output_noise_level",
            "pack_period", "pack_kind", "pack_offset", "pack_stride", "pack_count", "pack_width", "pack_field")


def _canon(nodes):
    """anon_N names numbered in order of first appearance, pt_N names collapsed (both carry no plan meaning)."""
    m = {}
    out = []
    for n in nodes:
        c = dict(n)
        c["inputs"] = [m.setdefault(x, f"anon#{len(m)}") if x.startswith("anon_") else
                       "pt" if x.startswith("pt_") else x for x in c["inputs"]]
        if c["output"].startswith("anon_"):
            c["output"] = m.setdefault(c["output"], f"anon#{len(m)}")
        c["inputs"] = [re.sub(r"^(anon_\d+)(_auto_bootstrapped)$", lambda mm: m.get(mm.group(1), mm.group(1))
                              + mm.group(2), x) for x in c["inputs"]]
        out.append(c)
    return out


def compare_graphs(ref_dir, got_dir, fields=COMPARED, log=print, show=3):
    """Field-wise diff of every block_<b>/graph.json in `ref_dir` against `got_dir`: {block: differences}, a
    difference = one (node, field) pair that differs, plus |len(ref) - len(got)| for missing / extra nodes (a
    missing block counts its node count). A fused inner sum's replayed product (output_level -1) is compared on
    structure only. Logs the first `show` differences per block."""
    import glob
    out = {}
    for p in sorted(glob.glob(os.path.join(ref_dir, "block_*", "graph.json")),
                    key=lambda s: int(s.split("block_")[-1].split("/")[0])):
        b = p.split("block_")[-1].split("/")[0]
        A = _canon(json.load(open(p))["nodes"])
        q = os.path.join(got_dir, f"block_{b}", "graph.json")
        if not os.path.exists(q):
            log(f"[compare] block {b}: missing ({len(A)} nodes)")
            out[b] = len(A)
            continue
        B = _canon(json.load(open(q))["nodes"])
        n, shown = abs(len(A) - len(B)), 0
        for i, (x, y) in enumerate(zip(A, B)):
            fs = ("op_type", "inputs", "output", "step") if x.get("output_level") == -1 else fields
            diff = [f for f in fs if x.get(f) != y.get(f)]
            n += len(diff)
            if diff and shown < show:
                shown += 1
                log(f"[compare] block {b} node {i}/{len(A)}: "
                    + "; ".join(f"{f}: {x.get(f)!r} vs {y.get(f)!r}" for f in diff))
        out[b] = n
        log(f"[compare] block {b}: {len(A)} vs {len(B)} nodes, {n} differences")
    return out
