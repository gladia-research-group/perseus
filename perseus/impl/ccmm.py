"""Batch ciphertext-ciphertext matrix products on SinC operands (Cheon-Kang-Lee, ePrint 2025/1957; FIDESlib Ccmm.cu),
each output refreshed by SinCBoot. Recorded like the PCMM: one external bootstrap per output."""
from __future__ import annotations

import math
import os
import time

import numpy as np

from . import sinc as sc
from .pcmm import _N32_CAP, KeyRng, _ks_key, _twiddles, find_prime

NB, NS, KY, SW = 1 << 16, 1 << 13, 64, 8
# PF_CCMM_MERGE=1: operands duplicated on batch l + 16 (sinc.dup) make every output even in Y = X^1024, so output pairs
# merge at the split as e + X^1024 o (no key switch) and share one SinCBoot; unmerge() splits them in the slot domain
MERGE = os.environ.get("PF_CCMM_MERGE", "0") not in ("", "0")


def _auto_poly(s, h):
    n = len(s)
    t = np.arange(n) * h % (2 * n)
    out = np.zeros(n, dtype=np.int64)
    out[np.where(t < n, t, t - n)] = s * np.where(t < n, 1, -1)
    return out


def _negacyclic_square(s):
    n = len(s)
    f = np.fft.fft(np.r_[s, np.zeros(n)].astype(float))
    sq = np.rint(np.real(np.fft.ifft(f * f))).astype(np.int64)
    return sq[:n] - sq[n:]


def make_keys(sk, moduli):
    """bts.ccmm_keys dict: ring switch sk <-> s'(X^8) at 2^16, 128 automorphism keys and s'^2 -> s' at 2^13, over the
    four bottom moduli + one 31-bit special prime."""
    rng = KeyRng()
    q = [int(m) for m in moduli[:4]]
    P = find_prime(31, 2 * NB, set(q))
    big = q + [P]
    sk = np.asarray(sk, dtype=np.int64)
    s1 = rng.integers(-1, 2, size=NS).astype(np.int64)
    emb = np.zeros(NB, dtype=np.int64)
    emb[::SW] = s1
    d = dict(primes=np.array(big, dtype=np.uint32))
    for name, n in (("16", NB), ("13", NS), ("6", KY)):
        tw = [_twiddles(p, n) for p in big]
        d["psi" + name] = [t[0] for t in tw]
        d["ipsi" + name] = [t[1] for t in tw]
        d["ninv" + name] = np.array([t[2] for t in tw], dtype=np.uint32)
    d["fwd"] = _ks_key(sk, emb, q, P, rng)
    d["back"] = _ks_key(emb, sk, q, P, rng)
    d["autos"] = np.stack([_ks_key(_auto_poly(s1, 2 * KY * t + 1), s1, q, P, rng) for t in range(128)])
    d["rlk"] = _ks_key(_negacyclic_square(s1), s1, q, P, rng)
    return d


_IDX = None


def _index():
    global _IDX
    if _IDX is None:
        r, c, l = np.meshgrid(np.arange(sc.D), np.arange(sc.D), np.arange(sc.BATCH), indexing="ij")
        _IDX = sc.sinc_slot(r, c, l)
    return _IDX


def w_theory():
    """Per batch slot l, the R_64 evaluation point of Y (slot l <-> zeta^(5^l), zeta = exp(i pi / 64))."""
    return np.exp(1j * np.pi * np.array([pow(5, l, 128) for l in range(sc.BATCH)]) / 64)


def w_slots(w):
    """w per batch slot -> per SinC slot."""
    return np.asarray(w)[(np.arange(sc.SLOTS) - sc.OFFSET) % 32]


def merged(e, o, w):
    """Slot vectors of the merged SinCBoot output (e + w o) / 2."""
    return 0.5 * (np.asarray(e) + w_slots(w) * np.asarray(o))


def to_mats(vecs):
    """16 SinC slot vectors -> (128 rows, 128 cols, 32 batch) complex."""
    ct, sl = _index()
    return np.asarray(vecs)[ct, sl]


def from_mats(M):
    ct, sl = _index()
    out = np.zeros((sc.NCT, sc.SLOTS), dtype=np.complex128)
    out[ct, sl] = M
    return out


def exact(a_vecs, b_vecs, b_transposed):
    A, B = to_mats(a_vecs), to_mats(b_vecs)
    return from_mats(np.einsum("ijl,kjl->ikl" if b_transposed else "ijl,jkl->ikl", A, B))


class Ccmm:
    def __init__(self, ops):
        self.ops = ops
        self.fhe = ops.fhe
        self.bts = getattr(ops.core, "bts", None)
        self.keys = None
        self.cap = int(os.environ.get("PF_CCMM_CAP", "0")) or None
        self.land = int(os.environ.get("PF_CCMM_LAND", "0")) or None
        if self.bts is None:
            self.cap = self.cap or _N32_CAP
            self.land = self.land or int(os.environ.get("PF_CCMM_FAKE_LAND", "28"))
        self.c_unit = None
        self._carriers = None
        self.w = w_theory()
        self.nf_ref = self.nf_split = None
        self.times = np.zeros(5)
        self.t_in = self.t_mm = self.t_out = self.keygen_s = 0.0
        self.calls = 0

    def prepare(self, dim):
        if self.bts is None or self.keys is not None:
            return
        ops, bts, S = self.ops, self.bts, self.ops.inf.slots
        t0 = time.perf_counter()
        x = ops.encode_token(np.zeros(dim))
        d = bts.device(self.fhe, x)
        ex = bts.export_limbs(self.fhe, d)
        self.keys = bts.ccmm_keys(make_keys(bts.secret_key_coeffs(self.fhe), ex["moduli"][:4]))
        self.nf_ref = bts.noise_factor(d)
        y = ops.encode_token(np.zeros(dim))
        dy = bts.device(self.fhe, y)
        st = bts.begin(dy, S, False, True)
        if not st.stc_first:
            raise RuntimeError("ccmm: the dense route is not StC-first (FIDESLIB_BTS_STC_FIRST=2)")
        bts.stc_first_input(dy, st)
        self.nf_split = bts.noise_factor(dy)
        self.keygen_s = time.perf_counter() - t0
        self._calibrate(dim)

    def _calibrate(self, dim):
        """c_unit: output value per unit operand factors, from known probe matrices (shift 0)."""
        rng = np.random.default_rng(1)
        A = np.zeros((sc.D, sc.D, sc.BATCH), complex)
        B = np.zeros((sc.D, sc.D, sc.BATCH))
        A[:, :, :16] = 0.05 * (rng.standard_normal((sc.D, sc.D, 16)) + 1j * rng.standard_normal((sc.D, sc.D, 16)))
        B[:, :, :16] = 0.05 * rng.standard_normal((sc.D, sc.D, 16))
        enc = [self.ops.encode_token(np.zeros(dim)) for _ in range(2 * sc.NCT)]
        xa = [self.ops.add_pt(e, v) for e, v in zip(enc[:sc.NCT], from_mats(A))]
        xb = [self.ops.add_pt(e, v) for e, v in zip(enc[sc.NCT:], from_mats(B))]
        f = 2.0 ** -3
        ys = self._device(xa, xb, False, f, f, 0)
        got = to_mats([self.ops.decrypt_slots_complex(y) for y in ys])[:, :, :16]
        want = np.einsum("ijl,jkl->ikl", A, B)[:, :, :16]
        self.c_unit = float(np.real(np.vdot(want, got) / np.vdot(want, want))) / (f * f)
        if self.land is None:
            self.land = self.ops.level(ys[0])
        if MERGE:
            self._calibrate_w(dim, A, B, f)

    def _calibrate_w(self, dim, A, B, f):
        """Fit w (merged = (e + w o) / 2 per batch slot) on the device from duplicated probe operands."""
        A2, B2 = A.copy(), B.copy()
        A2[:, :, 16:], B2[:, :, 16:] = A[:, :, :16], B[:, :, :16]
        enc = [self.ops.encode_token(np.zeros(dim)) for _ in range(2 * sc.NCT)]
        xa = [self.ops.add_pt(e, v) for e, v in zip(enc[:sc.NCT], from_mats(A2))]
        xb = [self.ops.add_pt(e, v) for e, v in zip(enc[sc.NCT:], from_mats(B2))]
        g = f / math.sqrt(2.0)
        ys = self._device(xa, xb, False, g, g, 0, True)
        got = np.array([self.ops.decrypt_slots_complex(y) for y in ys]) / (self.c_unit * f * f)
        P = from_mats(np.einsum("ijl,jkl->ikl", A2, B2))
        e, o = P[0::2], P[1::2]
        ls = (np.arange(sc.SLOTS) - sc.OFFSET) % 32
        w = np.zeros(sc.BATCH, complex)
        for l in range(sc.BATCH):
            sel = ls == l
            ov = o[:, sel].ravel()
            w[l] = np.vdot(ov, (2 * got[:, sel] - e[:, sel]).ravel()) / np.vdot(ov, ov)
        res = np.abs(2 * got - e - w_slots(w) * o).max() / np.abs(P).max()
        print(f"[ccmm] merge w fit: |w|-1 max {np.abs(np.abs(w) - 1).max():.2e}, vs theory {np.abs(w - w_theory()).max():.2e}"
              f", vs conj {np.abs(w - np.conj(w_theory())).max():.2e}, residual {res:.2e}", flush=True)
        self.w = w

    def _measure_cap(self, x):
        ops, bts = self.ops, self.bts
        d = bts.device(self.fhe, x)
        entry = bts.stc_first_entry(d, ops.inf.slots)
        return ops.level(x) + d.level - entry - ops.unit

    def _factors(self, amax, bmax, shift, out_scale, merge):
        p = out_scale * 2.0 ** -shift / self.c_unit * (0.5 if merge else 1.0)
        return math.sqrt(p * bmax / amax), math.sqrt(p * amax / bmax)

    def prep_b(self, xb, amax=1.0, bmax=1.0, shift=0, out_scale=1.0, merge=False, keep=False, export=True):
        """The right operands of later apply calls (same amax / bmax / shift / out_scale / merge), taken to the
        CCMM's input form once (consumed): the paged prefill keeps every page's K / V this way (4 limbs). `keep`:
        the next apply reads the device copies as they are (no import); `export=False`: no host copy (last page)."""
        ops = self.ops
        if self.cap is None:
            self.cap = self._measure_cap(xb[0])
        for x in xb:
            ops.bootstrap_hint(x, self.cap, True)
            self.fhe.level_hint(x, self.cap)
        prep = PreparedB(list(xb), [self.fhe.var_name(x) for x in xb], [ops.level(x) for x in xb])
        if self.bts is None:
            prep.vecs = [np.array(x.vec) for x in xb]
            return prep
        fb = self._factors(amax, bmax, shift, out_scale, merge)[1]
        bts, S = self.bts, ops.inf.slots
        bts.sync()
        t0 = time.perf_counter()
        # a dropped device ciphertext keeps its full allocation: host copies of the 4 live limbs, imported by
        # apply into one shared set of carriers (the first operands prepared)
        carry = self._carriers is None
        if carry:
            self._carriers = ([], [])
        if not keep:
            prep.cts = []
        for x in xb:
            d = bts.device(self.fhe, x)
            bts.stc_high(d, S, 3, fb, 1, self.nf_ref)
            if keep:
                prep.dev.append(d)
            if export:
                ex = bts.export_limbs(self.fhe, d)
                prep.host.append((ex["c0"].astype(np.uint32), ex["c1"].astype(np.uint32), float(ex["noise_factor"]),
                                  int(ex["noise_level"])))
            if carry:
                self._carriers[0].append(x)
                self._carriers[1].append(d)
        bts.sync()
        self.t_in += time.perf_counter() - t0
        return prep

    def apply(self, xa, xb, b_transposed=False, amax=1.0, bmax=1.0, shift=0, out_scale=1.0, site="ccmm", merge=False):
        """out_scale M M' (M' = xb^T when b_transposed) for SinC slot ciphertexts; amax / bmax bound |entries|
        (balance the operand factors), 2^shift bounds |output| (the refresh's range). Inputs are consumed. With
        `merge` (operands duplicated on l + 16) the 8 merged outputs (e + w o) / 2 of the pairs (2 p, 2 p + 1)."""
        ops = self.ops
        prep = xb if isinstance(xb, PreparedB) else None
        assert len(xa) == sc.NCT and (prep is not None or len(xb) == sc.NCT)
        if self.cap is None:
            self.cap = self._measure_cap(xa[0])
        mine = list(xa) if prep is not None else list(xa) + list(xb)
        for x in mine:
            ops.bootstrap_hint(x, self.cap, True)
            self.fhe.level_hint(x, self.cap)
        names = [self.fhe.var_name(x) for x in mine] + ([] if prep is None else prep.names)
        levels = [ops.level(x) for x in mine] + ([] if prep is None else prep.levels)
        want = None
        if self.bts is not None and os.environ.get("PF_CCMM_VERIFY") and prep is None:
            want = out_scale * exact([ops.decrypt_slots_complex(x) for x in xa],
                                     [ops.decrypt_slots_complex(x) for x in xb], b_transposed)
            if merge:
                want = np.array([merged(want[2 * i], want[2 * i + 1], self.w) for i in range(sc.NCT // 2)])
        if self.bts is None:
            outs = self._fake(xa, xb if prep is None else prep.vecs, b_transposed, out_scale, merge)
        else:
            fa, fb = self._factors(amax, bmax, shift, out_scale, merge)
            outs = self._device(xa, xb if prep is None else prep, b_transposed, fa, fb, shift, merge)
        if want is not None:
            got = np.array([ops.decrypt_slots_complex(y) for y in outs])
            err = float(np.abs(got - want).max())
            print(f"[ccmm_verify] {site} max|y|={np.abs(want).max():.3g} err={err:.3g} "
                  f"({math.log2(np.abs(want).max() / max(err, 1e-300)):.1f} bits)", flush=True)
        for y in outs:
            if self.bts is None:
                self.fhe.record_external_bootstrap(y, names, levels, landing=(self.land, 1))
            else:
                self.fhe.record_external_bootstrap(y, names, levels)
        self.calls += 1
        return outs

    def _fake(self, xa, xb, b_transposed, out_scale, merge=False):
        Y = out_scale * exact([x.vec for x in xa], [getattr(x, "vec", x) for x in xb], b_transposed)
        if merge:
            Y = np.array([merged(Y[2 * i], Y[2 * i + 1], self.w) for i in range(sc.NCT // 2)])
        cls = type(xa[0])
        outs = []
        for m, y in enumerate(Y):
            o = cls(y, self.land, 1, xa[0].packing)
            if hasattr(xa[m], "ptag"):
                o.ptag = xa[m].ptag
            outs.append(o)
        return outs

    def _device(self, xa, xb, b_transposed, fa, fb, shift, merge=False):
        ops, bts, S = self.ops, self.bts, self.ops.inf.slots
        bts.sync()
        t0 = time.perf_counter()
        da, db, states = [], [], []
        if isinstance(xb, PreparedB) and xb.dev:
            db, xb.dev, xb.cts, xb = list(xb.dev), [], [], []
        elif isinstance(xb, PreparedB):
            db = self._carriers[1]
            for d, (c0, c1, nf, nl) in zip(db, xb.host):
                bts.import_limbs(self.fhe, d, c0.astype(np.uint64), c1.astype(np.uint64), nf, nl)
            xb = []
        for xs, f, ds in ((xa, fa, da), (xb, fb, db)):
            for x in xs:
                d = bts.device(self.fhe, x)
                if ds is da:
                    states.append(bts.begin(d, S, False, True))
                bts.stc_high(d, S, 3, f, 1, self.nf_ref)
                ds.append(d)
        t1 = time.perf_counter()
        self.t_in += t1 - t0
        tm = bts.ccmm_mul(da, db, da, self.keys, b_transposed)
        self.times += np.asarray(tm)
        t2 = time.perf_counter()
        self.t_mm += t2 - t1
        if merge:
            for i in range(sc.NCT // 2):
                bts.merge_monomial(da[2 * i], da[2 * i + 1], NB // KY)
            da, states, xa = da[0::2], states[0::2], xa[0::2]
        for d, st in zip(da, states):
            bts.set_scale(d, self.nf_split, 2)
            bts.mod_raise(d, st)
            bts.fold(d, st)
            for k in range(st.n_cts):
                bts.cts_stage(d, st, k)
            bts.eval_mod(d, st)
            bts.stc_first_output(d, st)
            bts.stc_high(d, S, d.level - 2 - (2 if d.noise_level == 2 else 0), 1.0, 6)
            if shift:
                bts.mult_int(d, 1 << shift)
        bts.sync()
        self.t_out += time.perf_counter() - t2
        return list(xa)


class PreparedB:
    """Ccmm.prep_b's operands: host limbs (device) or slot vectors (fake), graph names / levels."""

    def __init__(self, cts, names, levels):
        self.cts, self.names, self.levels = cts, names, levels
        self.host, self.vecs, self.dev = [], None, []


def unmerge(rt, Ms, w, scale=1.0, site="ccmm.um"):
    """Merged outputs (e + w o) / 2 (duplicated on l + 16) -> scale [e, o, ...] on l < 16 (zero above), one level."""
    ops = rt.ops
    lo = scale * (((np.arange(sc.SLOTS) - sc.OFFSET) % 32) < 16)
    out = []
    for m in Ms:
        r = ops.rotate(m, 16)
        out.append(rt.mult_mask(ops.add(m, r), (site, "e", scale), lambda: lo))
        out.append(rt.mult_mask(ops.sub(m, r), (site, "o", scale), lambda: lo / w_slots(w)))
    return out


def engine(rt):
    e = rt.__dict__.get("_ccmm_engine")
    if e is None:
        e = rt.__dict__["_ccmm_engine"] = Ccmm(rt.ops)
        rt.ccmm = e
    return e


def enabled():
    return os.environ.get("PF_ATTN", "") == "ccmm"


def rot_steps():
    return sc.rot_steps()
