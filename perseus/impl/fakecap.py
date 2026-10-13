"""Structure capture on the numpy fake: the device recorder (include/fideslib_wrapper.h record_primitive,
include/graph.h) replayed on fake.FakeFhe, so a capture needs no GPU.

Per op the wrapper's bookkeeping: graph var names (v_N / pt_N counters reset per block, anon_N global, names keyed
by ciphertext identity), physical level + noise degree (lazy rescale on the composite chain: a product lands at the
max effective input level with degree 2; a plaintext add realizes), the pack tag (include/packing/pack_tag.h) and
its stamp, the reactive / deliberate / hint refreshes with their sparse routes and landings, and the magnitudes of
every node (magtape.node_stats). The planner reads the result exactly like a device capture.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from . import fake
from .fake import FakeCt, FakeFhe, FakeInf, _Core

# ── pack tags (include/packing/pack_tag.h) ─────────────────────────────────────────────────────────────────────────

EMPTY, AP, DENSE = 0, 1, 2
F_UNKNOWN, F_REAL, F_IMAG, F_COMPLEX = 0, 1, 2, 3


class Support:
    __slots__ = ("kind", "offset", "stride", "count", "width")

    def __init__(self, kind=DENSE, offset=0, stride=1, count=0, width=1):
        self.kind, self.offset, self.stride, self.count, self.width = kind, offset, stride, count, width

    @staticmethod
    def empty(): return Support(EMPTY, 0, 1, 0, 1)

    @staticmethod
    def dense(): return Support(DENSE, 0, 1, 0, 1)

    @staticmethod
    def ap(off, st, cnt):
        if cnt <= 0:
            return Support.empty()
        if st <= 1 and off == 0:
            return Support.dense()
        return Support(AP, off, st, cnt, 1)

    @staticmethod
    def block(off, st, w, cnt):
        if cnt <= 0 or w <= 0:
            return Support.empty()
        if w == 1:
            return Support.ap(off, st, cnt)
        if w >= st:
            return Support.dense()
        return Support(AP, off, st, cnt, w)

    def is_dense(self): return self.kind == DENSE

    def is_empty(self): return self.kind == EMPTY


class Tag:
    __slots__ = ("slots", "period", "support", "field", "conj_of")

    def __init__(self, slots=0, period=0, support=None, field=F_UNKNOWN, conj_of=None):
        self.slots, self.period = slots, period
        self.support = support if support is not None else Support()
        self.field, self.conj_of = field, conj_of

    def known(self): return self.slots > 0 and self.period > 0

    def periodic_at(self, s): return self.period > 0 and s % self.period == 0

    def fold_collision_free_at(self, s):
        sp = self.support
        if sp.is_empty():
            return True
        if sp.is_dense():
            return s >= self.slots
        if sp.width > 1:
            if sp.stride > 0 and s % sp.stride == 0:
                return sp.count <= s // sp.stride and sp.width <= sp.stride
            return False
        return sp.count <= s // math.gcd(sp.stride, s)

    def min_routable_s(self):
        s = 1
        while s < self.slots:
            if self.periodic_at(s) or self.fold_collision_free_at(s):
                return s
            s <<= 1
        return self.slots


def top(slots): return Tag(slots, slots, Support.dense())


def _lcm_capped(a, b, cap):
    if a <= 0 or b <= 0:
        return cap
    l = a * b // math.gcd(a, b)
    return cap if l >= cap else l


def _union(a, b):
    if a.is_empty():
        return b
    if b.is_empty():
        return a
    if a.is_dense() or b.is_dense():
        return Support.dense()
    if a.stride == b.stride and a.offset == b.offset and a.width == b.width:
        return Support.block(a.offset, a.stride, a.width, max(a.count, b.count))
    if a.stride == b.stride:
        lo, hi = min(a.offset, b.offset), max(a.offset + a.width, b.offset + b.width)
        if hi - lo <= a.stride:
            return Support.block(lo, a.stride, hi - lo, max(a.count, b.count))
    return Support.dense()


def _intersect(a, b):
    if a.is_empty() or b.is_empty():
        return Support.empty()
    if a.is_dense():
        return b
    if b.is_dense():
        return a
    if a.stride == b.stride and a.offset == b.offset:
        return Support.block(a.offset, a.stride, min(a.width, b.width), min(a.count, b.count))
    return a if a.count * a.width <= b.count * b.width else b


def _f_add(a, b):
    if a == F_UNKNOWN or b == F_UNKNOWN:
        return F_UNKNOWN
    return a if a == b else F_COMPLEX


def _f_mult(a, b):
    if a == F_UNKNOWN or b == F_UNKNOWN:
        return F_UNKNOWN
    if a == F_COMPLEX or b == F_COMPLEX:
        return F_COMPLEX
    return F_REAL if a == b else F_IMAG


def with_field(t, f):
    return Tag(t.slots, t.period, t.support, f, None)


def t_mult_scalar(x, c):
    if not x.known():
        return with_field(Tag(), x.field)
    if c == 0.0:
        return with_field(Tag(x.slots, 1, Support.empty()), x.field)
    return with_field(x, x.field)


def t_add_scalar(x, c):
    f = x.field if c == 0.0 else _f_add(x.field, F_REAL)
    if not x.known():
        return with_field(Tag(), f)
    if c == 0.0:
        return with_field(x, f)
    return with_field(Tag(x.slots, x.period, Support.dense()), f)


def t_square(x): return with_field(x, _f_mult(x.field, x.field))


def t_add(a, b):
    f = _f_add(a.field, b.field)
    if not a.known() or not b.known():
        return with_field(Tag(), f)
    return with_field(Tag(a.slots, _lcm_capped(a.period, b.period, a.slots), _union(a.support, b.support)), f)


def t_mult(a, b):
    f = _f_mult(a.field, b.field)
    if not a.known() and not b.known():
        return with_field(Tag(), f)
    if not a.known():
        return with_field(Tag(b.slots, b.slots, b.support), f)
    if not b.known():
        return with_field(Tag(a.slots, a.slots, a.support), f)
    return with_field(Tag(a.slots, _lcm_capped(a.period, b.period, a.slots), _intersect(a.support, b.support)), f)


def t_mult_i(x):
    f = F_IMAG if x.field == F_REAL else F_REAL if x.field == F_IMAG else x.field
    return with_field(x, f)


def t_rotate(x, k):
    if not x.known():
        return with_field(Tag(), x.field)
    r = with_field(x, x.field)
    if x.support.kind == AP:
        off = (x.support.offset - k) % x.slots
        r.support = Support.block(off, x.support.stride, x.support.width, x.support.count)
    return r


def t_conjugate(x):
    if not x.known():
        return with_field(Tag(), x.field)
    r = with_field(x, x.field)
    if x.support.kind == AP:
        r.support = Support.dense()
    return r


def t_bootstrap(x, s, slots):
    if s <= 0:
        return with_field(x, x.field)
    if slots <= 0:
        return with_field(Tag(), x.field)
    return with_field(Tag(slots, min(s, slots), Support.dense()), x.field)


def conj_pair(t, a, b, ta, tb, sub):
    if (tb.conj_of is not None and tb.conj_of is a) or (ta.conj_of is not None and ta.conj_of is b):
        t.field = F_IMAG if sub else F_REAL
    return t


def from_signature(slots, period, live, stride, window, offset):
    t = Tag(slots, period if period > 0 else slots)
    if live <= 0:
        t.support = Support.empty()
    elif live >= slots:
        t.support = Support.dense()
    elif stride > 1:
        t.support = Support.ap(offset, stride, live)
    else:
        t.support = Support.ap(offset, 1, window - offset if window > offset else live)
    return t


def pt_tag(values):
    """tag_plaintext(analyze_packing(values)) (include/packing/pack_signature.h)."""
    v = np.asarray(values)
    cplx = np.iscomplexobj(v)
    re = v.real.astype(np.float64)
    im = v.imag.astype(np.float64) if cplx else np.zeros_like(re)
    n = re.shape[0]
    amax, imax = float(np.abs(re).max(initial=0.0)), float(np.abs(im).max(initial=0.0))
    im_data = imax > 1e-3 * max(amax, 1e-300)
    tol = max(1e-2 * (max(amax, imax) if im_data else amax), 1e-12)
    period, p = n, 1
    while p < n:
        a = re.reshape(-1, p)
        ok = np.abs(a - a[0]).max() <= tol
        if ok and im_data:
            b = im.reshape(-1, p)
            ok = np.abs(b - b[0]).max() <= tol
        if ok:
            period = p
            break
        p <<= 1
    nz = np.flatnonzero((re != 0.0) | (im != 0.0))
    live = int(nz.size)
    window = int(nz[-1]) + 1 if live else 0
    offset = int(nz[0]) if live else 0
    stride = int(np.gcd.reduce(nz - nz[0])) if live >= 2 else (n if live == 1 else 0)
    t = from_signature(n, period, live, stride, window, offset)
    if cplx:
        r, i = bool((re != 0).any()), bool((im != 0).any())
        t.field = (F_COMPLEX if r else F_IMAG) if i else F_REAL
    else:
        t.field = F_REAL
    return t


# ── the recorder ───────────────────────────────────────────────────────────────────────────────────────────────────

def _fmt_scalar(x):
    """std::ostream << double (precision 6, %g)."""
    return f"const({float(x):g})"


class CapCt(FakeCt):
    def __init__(self, vec, level, noise_deg=1, packing="cachemir"):
        super().__init__(vec, level, noise_deg, packing)
        self.name, self.gen, self.tag = None, -1, Tag()
        self.ptag = Tag()           # the PackedCtx's own tag (the registry's `tag` is the Ciphertext's)


_CHECK_FIELD = bool(os.environ.get("FAKECAP_CHECK_FIELD"))

# ciphertexts made outside the recorder (the PCMM's fake products) start unnamed and untagged
FakeCt.name, FakeCt.gen, FakeCt.tag, FakeCt.ptag = None, -1, Tag(), Tag()


class Pt:
    """A plaintext object: its identity names it (pt_vars). The device keys pt_vars by address, so a recycled
    temporary can inherit a stale pt_N; plaintext names carry no plan semantics (the planner reads only that an input
    is a plaintext, and its level), so the fake names every plaintext afresh."""
    __slots__ = ("vec", "level", "deg", "tag", "name", "gen")

    def __init__(self, vec, level, deg=1, tag=None):
        self.vec, self.level, self.deg = vec, int(level), int(deg)
        self.tag = tag if tag is not None else Tag()
        self.name, self.gen = None, -1


def _landings():
    """route s -> (level, deg) of a refresh landing; 0 = dense (FAKECAP_LANDINGS, the n32 chain's by default)."""
    spec = os.environ.get("FAKECAP_LANDINGS", "0:32,1:14,256:22,512:22")
    out = {}
    for kv in spec.split(","):
        s, lv = kv.split(":")
        out[int(s)] = (int(lv), 2)
    return out


class CapFhe(FakeFhe):
    """FakeFhe with the device recorder's bookkeeping."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.gen = 0
        self.nodes = None                  # list while a block is captured
        self.ct_counter = self.pt_counter = self.anon_counter = 0
        self.graph_scope, self.graph_scope_counter, self.graph_scope_strict = "", 0, False
        self.steps = []
        self.landing = _landings()
        self.sparse_mode = int(os.environ.get("SPARSE_AUTO", "0") or 0)
        self.sparse_precomp_slots = sorted({int(x) for x in os.environ.get("SPARSE_BTS_SLOTS", "").split(",") if x})
        self.pcmm_landing = None           # (level, deg) of a PCMM output, set by the session
        self.stc_drop = int(os.environ.get("FAKECAP_STC_DROP", "6"))      # StC-first lands this much lower (n32)
        self.stc_entry = int(os.environ.get("FAKECAP_STC_ENTRY", "46"))   # deepest input with room for StC-first
        self.stats = True
        self._pend, self._batch = [], int(os.environ.get("FAKECAP_BATCH", "1"))
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

    def record(self, op, ins, out, in_levels, out_ct=None, out_level=None):
        if self.nodes is None:
            return
        n = {"op_type": op, "inputs": list(ins), "input_levels": [int(x) for x in in_levels], "output": out,
             "step": self.path(), "output_level": int(out_ct.level if out_ct is not None else out_level)}
        if out_ct is not None:
            n["output_noise_level"] = int(out_ct.noise_deg)
            if _CHECK_FIELD and out_ct.tag.field in (F_REAL, F_IMAG) and np.iscomplexobj(out_ct.vec):
                v = np.asarray(out_ct.vec)
                bad = np.abs(v.imag if out_ct.tag.field == F_REAL else v.real).max()
                if bad > 1e-6 * max(float(np.abs(v).max()), 1e-30):
                    print(f"[fakecap] field tag {out_ct.tag.field} unsound: {op} {out} @ {self.path()} "
                          f"off-axis {bad:.3g} of {float(np.abs(v).max()):.3g}", flush=True)
            if self.stats and not self.magnitude_suppressed:
                self._pend.append((n, np.array(out_ct.vec, dtype=np.complex128)))
                if len(self._pend) >= self._batch:
                    self.flush_stats()
            tg = out_ct.tag
            if tg.known():
                self._stamp(n, tg)
            if tg.field:
                n["pack_field"] = int(tg.field)
        self.nodes.append(n)

    def flush_stats(self):
        """The pending nodes' magnitudes (magtape.node_stats), one multi-worker FFT per batch."""
        if self._pend:
            from .magtape import node_stats_batch
            for (n, _), st in zip(self._pend, node_stats_batch(np.stack([v for _, v in self._pend]))):
                n.update(st)
            self._pend = []

    @staticmethod
    def _stamp(n, tg):
        n.update(pack_period=int(tg.period), pack_kind=int(tg.support.kind), pack_offset=int(tg.support.offset),
                 pack_stride=int(tg.support.stride), pack_count=int(tg.support.count),
                 pack_width=int(tg.support.width))
        if tg.field:
            n["pack_field"] = int(tg.field)

    def restamp(self, ct):
        if self.nodes is None or not ct.tag.known():
            return
        name = self.var(ct)
        for n in reversed(self.nodes):
            if n["output"] == name:
                self._stamp(n, ct.tag)
                return

    # ── levels ──
    def eff(self, ct):
        return ct.level + (self.unit if ct.noise_deg == 2 else 0)

    def _new(self, vec, level, deg, a):
        return CapCt(vec, level, deg, a.packing)

    def _sum_level(self, a, b):
        """(level, deg) of a ct + ct (the planner's sim rule, OpenFHE's level/degree adjustment)."""
        if a.noise_deg == b.noise_deg:
            return max(a.level, b.level), a.noise_deg
        d2 = max(self.eff(x) for x in (a, b) if x.noise_deg == 2)
        d1 = max(x.level for x in (a, b) if x.noise_deg != 2)
        return (d2 - self.unit, 2) if d2 > d1 else (d1, 1)

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
        """Whether a dense refresh now takes the StC-first order (FIDESLIB_BTS_STC_FIRST on the dense route, allowed by
        perseus.impl.bootstrap.stc_first, not staged-only)."""
        if int(os.environ.get("FIDESLIB_BTS_STC_FIRST", "0") or 0) <= 0:
            return False
        if os.environ.get("FIDESLIB_BTS_STC_FIRST_STAGED_ONLY", "0") not in ("", "0"):
            return False
        if "dense" not in os.environ.get("FIDESLIB_BTS_STC_FIRST_ROUTES", "dense"):
            return False
        from . import bootstrap as _pyb
        return bool(_pyb._stc_first["on"])

    def bootstrap_output_level(self):
        return self.landing[0][0] - (self.stc_drop if self.stc_first() else 0)

    def inner_bootstrap(self, ct):
        self._in_level = ct.level
        s = self.auto_route(ct)
        if s and os.environ.get("FAKECAP_SPARSE_EXACT"):   # a sparse route returns the s-periodic projection
            v = np.asarray(ct.vec).reshape(-1, s).mean(0)
            ct.vec = np.tile(v, self.slots // s)
        ct.level, ct.noise_deg = self.landing[s]
        if s == 0 and ct.level and self.stc_first() and self._in_level <= self.stc_entry:
            ct.level -= self.stc_drop
        ct.tag = t_bootstrap(ct.tag, s, self.slots)
        self.n_bootstraps += 1
        self.total_bootstraps += 1

    def maybe_bootstrap(self, ct):
        if self.auto_bts_suppressed or ct.level < self._limit:
            return
        lin, nin = ct.level, self.var(ct)
        self.inner_bootstrap(ct)
        self.n_reactive += 1
        out = nin + "_auto_bootstrapped"
        self.name_ct(ct, out)
        self.record("auto_bootstrap", [nin], out, [lin], ct)

    def bootstrap(self, a, iterations=1):
        lin, nin = a.level, self.var(a)
        self.inner_bootstrap(a)
        out = self.new_var(a)
        self.record("deliberate_bootstrap", [nin], out, [lin], a)

    def bootstrap_hint(self, a, thr, account_pending_rescale=False, iterations=1):
        lin, nin = a.level, self.var(a)
        eff = a.level + (self.unit if (account_pending_rescale and a.noise_deg == 2) else 0)
        if eff > int(thr):
            self.inner_bootstrap(a)
            if a.noise_deg == 2:           # inner_bootstrap_planned's landing realize
                a.level += self.unit
                a.noise_deg = 1
        out = self.new_var(a)
        self.record("hint", [nin, f"hint_lev({int(thr)})"], out, [lin, -1], a)

    def level_hint(self, a, level):
        lin, nin = a.level, self.var(a)
        out = self.new_var(a)
        self.record("level_hint", [nin, f"lvl_cap({int(level)})"], out, [lin, -1], a)

    def drop_to_level(self, a, level):
        lin, nin = a.level, self.var(a)
        if int(level) > a.level:
            a.level = int(level)
        out = self.new_var(a)
        self.record("level_reduce", [nin], out, [lin], a)

    def record_external_bootstrap(self, a, inputs, input_levels):
        self.n_bootstraps += 1
        self.total_bootstraps += 1
        if self.pcmm_landing is not None:
            a.level, a.noise_deg = self.pcmm_landing
        out = self.new_var(a)
        a.tag = Tag()
        self.record("deliberate_bootstrap", list(inputs), out, list(input_levels), a)

    def var_name(self, a):
        return self.var(a)

    def fold_bootstrap(self, a, s, n_live, prescale=1.0):
        raise NotImplementedError("fakecap: fold_bootstrap")

    # ── arithmetic ──
    def _binary(self, op, a, b, vec, tag, level, deg, rec_op, b_name, b_level, product=False):
        """`product`: the op ends in maybe_bootstrap (the products, negate, and ct - ct)."""
        la, na = a.level, self.var(a)
        out_ct = self._new(vec, level, deg, a)
        self.tag_ct(out_ct, tag)
        out = self.new_var(out_ct)
        self.record(rec_op, [na, b_name], out, [la, b_level], out_ct)
        if product:
            self.maybe_bootstrap(out_ct)
        out_ct.tag = tag
        return out_ct

    def _c_add(self, a, b):
        if isinstance(b, CapCt):
            ta, tb = a.tag, b.tag
            tag = conj_pair(t_add(ta, tb), a, b, ta, tb, False)
            lv, dg = self._sum_level(a, b)
            nb = self.var(b)
            return self._binary("add", a, b, a.vec + b.vec, tag, lv, dg, "add", nb, b.level)
        if isinstance(b, Pt):
            return self._add_pt(a, b, a.vec + b.vec)
        c = float(b)
        return self._binary("add", a, b, a.vec + c, t_add_scalar(a.tag, c), a.level, a.noise_deg, "add",
                            _fmt_scalar(c), -1)

    def _add_pt(self, a, pt, vec, op="add"):
        tag = t_add(a.tag, pt.tag)
        if a.noise_deg == 2 and pt.deg == 1:
            lv, dg = max(self.eff(a), pt.level), 1
        else:
            lv, dg = max(a.level, pt.level), a.noise_deg
        return self._binary(op, a, pt, vec, tag, lv, dg, op, self.pt_var(pt), pt.level)

    def _c_sub(self, a, b):
        if isinstance(b, CapCt):
            ta, tb = a.tag, b.tag
            tag = conj_pair(t_add(ta, tb), a, b, ta, tb, True)
            lv, dg = self._sum_level(a, b)
            nb = self.var(b)
            return self._binary("sub", a, b, a.vec - b.vec, tag, lv, dg, "sub_ct", nb, b.level, product=True)
        c = float(b)
        return self._binary("sub", a, b, a.vec - c, t_add_scalar(a.tag, c), a.level, a.noise_deg, "sub",
                            _fmt_scalar(c), -1)

    def _c_mult(self, a, b):
        if isinstance(b, CapCt):
            lv = max(self.eff(a), self.eff(b))
            nb = self.var(b)
            return self._binary("mult", a, b, a.vec * b.vec, t_mult(a.tag, b.tag), lv, 2, "mult", nb, b.level,
                                product=True)
        if isinstance(b, Pt):
            lv = max(self.eff(a), b.level)
            return self._binary("mult", a, b, a.vec * b.vec, t_mult(a.tag, b.tag), lv, 2, "mult", self.pt_var(b),
                                b.level, product=True)
        c = float(b)
        return self._binary("mult", a, b, a.vec * c, t_mult_scalar(a.tag, c), self.eff(a), 2, "mult",
                            _fmt_scalar(c), -1, product=True)

    def _c_square(self, a):
        la, na = a.level, self.var(a)
        tag = t_square(a.tag)
        out_ct = self._new(a.vec * a.vec, self.eff(a), 2, a)
        self.tag_ct(out_ct, tag)
        out = self.new_var(out_ct)
        self.record("square", [na], out, [la], out_ct)
        self.maybe_bootstrap(out_ct)
        out_ct.tag = tag
        return out_ct

    def _c_negate(self, a):
        la, na = a.level, self.var(a)
        tag = t_mult_scalar(a.tag, -1.0)
        out_ct = self._new(-a.vec, self.eff(a), 2, a)
        self.tag_ct(out_ct, tag)
        out = self.new_var(out_ct)
        self.record("negate", [na], out, [la], out_ct)
        self.maybe_bootstrap(out_ct)
        out_ct.tag = tag
        return out_ct

    def inplace_add(self, a, b):
        ta, tb = a.tag, b.tag
        tag = conj_pair(t_add(ta, tb), a, b, ta, tb, False)
        la, lb, na, nb = a.level, b.level, self.var(a), self.var(b)
        a.level, a.noise_deg = self._sum_level(a, b)
        a.vec = a.vec + b.vec
        a.tag = tag
        out = self.new_var(a)
        self.record("add_inplace", [na, nb], out, [la, lb], a)
        a.tag = tag
        a.ptag = tag

    def _c_rotate(self, a, steps):
        steps = int(steps)
        if self.slots is not None:
            k = steps % self.slots
            if k and k not in self._rot and (k - self.slots) not in self._rot and steps not in self._rot:
                raise KeyError(f"no rotation key for {steps}")
        tag = t_rotate(a.tag, steps)
        la, na = a.level, self.var(a)
        out_ct = self._new(np.roll(a.vec, -steps), a.level, a.noise_deg, a)
        out = self.new_var(out_ct)
        self.record("rotate", [na, f"rot({steps})"], out, [la, -1], out_ct)
        out_ct.tag = tag
        return out_ct

    def _c_rotate_hoisted(self, a, steps):
        la, na = a.level, self.var(a)
        outs = []
        for k in steps:
            o = self._new(np.roll(a.vec, -int(k)), a.level, a.noise_deg, a)
            out = self.new_var(o)
            self.record("rotate", [na, f"rot({int(k)})"], out, [la, -1], o)
            outs.append(o)
        for k, o in zip(steps, outs):      # the PackedCtx overlay tags each after the loop
            o.tag = t_rotate(a.tag, int(k))
        return outs

    def _c_rotate_sum(self, a, steps):
        if not self.has_rot_keys(steps):
            raise KeyError(f"no rotation keys for {steps}")
        tag = a.tag
        i = 0
        while (1 << i) - 1 < len(steps):
            tag = t_add(tag, t_rotate(tag, steps[(1 << i) - 1]))
            i += 1
        la, na = a.level, self.var(a)
        vec = a.vec.copy()
        for k in steps:
            vec = vec + np.roll(a.vec, -int(k))
        out_ct = self._new(vec, a.level, a.noise_deg, a)
        self.tag_ct(out_ct, tag)
        out = self.new_var(out_ct)
        self.record("rotate_sum", [na, "rot(" + ",".join(str(int(k)) for k in steps) + ")"], out, [la, -1], out_ct)
        out_ct.tag = tag
        return out_ct

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
        tag = t_conjugate(a.tag)
        tag.conj_of = a
        la, na = a.level, self.var(a)
        out_ct = self._new(np.conj(a.vec) if self.complex_payload else a.vec.copy(), a.level, a.noise_deg, a)
        self.tag_ct(out_ct, tag)
        out = self.new_var(out_ct)
        self.record("conjugate", [na], out, [la], out_ct)
        out_ct.tag = tag
        return out_ct

    def _c_mult_i(self, a):
        tag = t_mult_i(a.tag)
        la, na = a.level, self.var(a)
        out_ct = self._new(a.vec * 1j, a.level, a.noise_deg, a)
        out = self.new_var(out_ct)
        out_ct.tag = tag
        self.record("mult_i", [na], out, [la], out_ct)
        return out_ct

    def mult_add_many(self, acc, vs, ss):
        for v, t in zip(vs, ss):
            self.inplace_add(acc, self.mult(v, t))
        return False

    # ── the PackedCtx overlay (what Python calls): Ctx op, then the result tagged from the handles' own tags ──
    def _tagged(self, out, tag):
        out.tag = tag
        out.ptag = tag
        return out

    def add(self, a, b):
        if isinstance(b, CapCt):
            return self._tagged(self._c_add(a, b), conj_pair(t_add(a.ptag, b.ptag), a, b, a.ptag, b.ptag, False))
        if isinstance(b, Pt):
            return self._tagged(self._c_add(a, b), t_add(a.ptag, b.tag))
        return self._tagged(self._c_add(a, b), t_add_scalar(a.ptag, float(b)))

    def sub(self, a, b):
        if isinstance(b, CapCt):
            return self._tagged(self._c_sub(a, b), conj_pair(t_add(a.ptag, b.ptag), a, b, a.ptag, b.ptag, True))
        return self._tagged(self._c_sub(a, b), t_add_scalar(a.ptag, float(b)))

    def mult(self, a, b):
        if isinstance(b, CapCt):
            return self._tagged(self._c_mult(a, b), t_mult(a.ptag, b.ptag))
        if isinstance(b, Pt):
            return self._tagged(self._c_mult(a, b), t_mult(a.ptag, b.tag))
        return self._tagged(self._c_mult(a, b), t_mult_scalar(a.ptag, float(b)))

    def square(self, a):
        return self._tagged(self._c_square(a), t_square(a.ptag))

    def negate(self, a):
        return self._tagged(self._c_negate(a), t_mult_scalar(a.ptag, -1.0))

    def rotate(self, a, steps):
        return self._tagged(self._c_rotate(a, steps), t_rotate(a.ptag, int(steps)))

    def rotate_hoisted(self, a, steps):
        return [self._tagged(o, t_rotate(a.ptag, int(k))) for o, k in zip(self._c_rotate_hoisted(a, steps), steps)]

    def rotate_sum(self, a, steps):
        o = self._c_rotate_sum(a, steps)
        return self._tagged(o, o.tag)

    def conjugate(self, a):
        t = t_conjugate(a.ptag)
        t.conj_of = a
        return self._tagged(self._c_conjugate(a), t)

    def mult_i(self, a):
        return self._tagged(self._c_mult_i(a), t_mult_i(a.ptag))


class CapInf(FakeInf):
    """FakeInf with plaintext objects (identity-named, tagged, encoded at the ciphertext's level)."""

    def __init__(self, fhe, *a, **k):
        super().__init__(fhe, *a, **k)
        self.complex = True
        self.cache = {}
        self.const_pts = {}
        self.named = {}

    def step(self, label):
        import contextlib
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

    def mult_const(self, a, re, im):
        lv = self.fhe.eff(a)
        key = (float(re), float(im), lv)
        pt = self.const_pts.get(key)
        if pt is None:
            pt = self.const_pts[key] = Pt(np.full(self.slots, complex(re, im)), lv, 1, Tag())
        return self.fhe.mult(a, pt)

    def set_slot_pt(self, name, values, level=0, deg=1):
        v = self._full(values)
        lv = int(level) if level > 0 else self.fhe.bootstrap_output_level()
        self.named[name] = Pt(v, lv, deg, pt_tag(v))
        self.slot_pts[name] = v

    def _slot_pt(self, name, a):
        pt = self.named[name]
        lv, dg = self._like(a)
        if pt.level != lv:                  # slot_pt_at re-levels on a mismatch: a fresh, untagged plaintext
            pt = self.named[name] = Pt(pt.vec, lv, dg, Tag())
        return pt

    def mult_slot_pt(self, a, name):
        return self.fhe.mult(a, self._slot_pt(name, a))

    def add_slot_pt(self, a, name):
        return self.fhe.add(a, self._slot_pt(name, a))

    def mult_slot_pt_many(self, a, names):
        return [self.mult_slot_pt(a, n) for n in names]

    def tag_reduce(self, a, stride):
        t = top(self.slots)
        t.field = a.tag.field
        a.tag = with_field(Tag(self.slots, 1, Support.dense()), t.field) if stride <= 1 else \
            with_field(Tag(self.slots, min(int(stride), self.slots), Support.dense()), t.field)
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
            pt = self.cache[key] = Pt(v, lv, dg, pt_tag(v) if tagged else Tag())
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
            self.cache[key] = Pt(v, int(level), 1, pt_tag(v))

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
    def begin_subgraph_capture(inf, b):
        d = os.environ.get("FHE_GRAPH_DIR")
        if not d or os.path.exists(os.path.join(d, f"block_{b}", "graph.json")):
            return False
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
        d = os.path.join(os.environ["FHE_GRAPH_DIR"], f"block_{b}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "graph.json"), "w") as f:
            json.dump({"version": 1, "nodes": fhe.nodes}, f)
        top_env = os.path.join(os.environ["FHE_GRAPH_DIR"], "capture_env.json")
        if os.path.exists(top_env):
            import shutil
            shutil.copy(top_env, os.path.join(d, "capture_env.json"))
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
    """The env a device session's profile exports (examples/*/env.open_session: the decode preset's extras, the port's
    ENV, the overrides; setdefault), so a fake capture stamps the same contract."""
    from perseus.profile import SessionProfile
    prof = SessionProfile.gpt2_decode_n64() if chain == "n64" else SessionProfile.gpt2_decode_n32()
    prof.ckks_complex = bool(complex_payload)
    prof.extra = dict(prof.extra)
    prof.extra.update({k: v for k, v in env_defaults.items()})
    prof.extra.update({k: str(v) for k, v in overrides.items()})
    prof.apply()


def make_session(dims, rot_steps, limit=None, bts_out=32, unit=2, complex_payload=True, pcmm_landing=(26, 2)):
    """(fhe, inf) recording fakes at `dims` with the session's rotation keys; the reactive ceiling is AUTO_BTS_LEVEL."""
    N = dims.N
    limit = int(limit or os.environ.get("AUTO_BTS_LEVEL") or 46)
    steps = {int(s) % N for s in rot_steps}
    steps = steps | {s - N for s in steps}
    fhe = CapFhe(steps, limit=limit, bts_out=bts_out, unit=unit, slots=N, complex_payload=complex_payload,
                 sparse_slots=tuple(int(x) for x in os.environ.get("SPARSE_BTS_SLOTS", "512,1").split(",") if x))
    fhe.pcmm_landing = pcmm_landing
    inf = CapInf(fhe, N, dims.hid, dims.dim, dims.H, dims.H_real, dims.E, dims.E_real)
    return fhe, inf


def compare_graphs(ref_dir, got_dir, fields=("op_type", "inputs", "input_levels", "output", "step", "output_level",
                                             "output_noise_level", "pack_period", "pack_kind", "pack_offset",
                                             "pack_stride", "pack_count", "pack_width", "pack_field"), log=print):
    """First structural divergence per block (anon names canonicalized by first appearance, plaintext names as a
    class). Returns #bad blocks."""
    import glob
    import re
    bad = 0

    def canon(nodes):
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
    for p in sorted(glob.glob(os.path.join(ref_dir, "block_*", "graph.json")),
                    key=lambda s: int(s.split("block_")[1].split("/")[0])):
        b = p.split("block_")[1].split("/")[0]
        q = os.path.join(got_dir, f"block_{b}", "graph.json")
        if not os.path.exists(q):
            log(f"[fakecap] block {b}: missing")
            bad += 1
            continue
        A, B = canon(json.load(open(p))["nodes"]), canon(json.load(open(q))["nodes"])
        for i, (x, y) in enumerate(zip(A, B)):
            # a fused inner sum's replayed product (output_level -1, no degree / tag): the loop's node, unannotated
            fs = ("op_type", "inputs", "output", "step") if x.get("output_level") == -1 else fields
            diff = [f for f in fs if x.get(f) != y.get(f)]
            if diff:
                log(f"[fakecap] block {b} node {i}/{len(A)}: " + "; ".join(f"{f}: {x.get(f)!r} vs {y.get(f)!r}"
                                                                          for f in diff))
                log(f"           ref {json.dumps({k: x.get(k) for k in fields})}")
                bad += 1
                break
        else:
            if len(A) != len(B):
                log(f"[fakecap] block {b}: {len(A)} vs {len(B)} nodes")
                bad += 1
            else:
                log(f"[fakecap] block {b}: {len(A)} nodes identical")
    return bad
