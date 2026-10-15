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

    def _measure_cap(self, x):
        ops, bts = self.ops, self.bts
        d = bts.device(self.fhe, x)
        entry = bts.stc_first_entry(d, ops.inf.slots)
        return ops.level(x) + d.level - entry - ops.unit

    def apply(self, xa, xb, b_transposed=False, amax=1.0, bmax=1.0, shift=0, out_scale=1.0, site="ccmm"):
        """out_scale M M' (M' = xb^T when b_transposed) for SinC slot ciphertexts; amax / bmax bound |entries|
        (balance the operand factors), 2^shift bounds |output| (the refresh's range). Inputs are consumed."""
        ops = self.ops
        assert len(xa) == len(xb) == sc.NCT
        if self.cap is None:
            self.cap = self._measure_cap(xa[0])
        for x in list(xa) + list(xb):
            ops.bootstrap_hint(x, self.cap, True)
            self.fhe.level_hint(x, self.cap)
        names = [self.fhe.var_name(x) for x in list(xa) + list(xb)]
        levels = [ops.level(x) for x in list(xa) + list(xb)]
        want = None
        if self.bts is not None and os.environ.get("PF_CCMM_VERIFY"):
            want = out_scale * exact([ops.decrypt_slots_complex(x) for x in xa],
                                     [ops.decrypt_slots_complex(x) for x in xb], b_transposed)
        if self.bts is None:
            outs = self._fake(xa, xb, b_transposed, out_scale)
        else:
            p = out_scale * 2.0 ** -shift / self.c_unit
            fa, fb = math.sqrt(p * bmax / amax), math.sqrt(p * amax / bmax)
            outs = self._device(xa, xb, b_transposed, fa, fb, shift)
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

    def _fake(self, xa, xb, b_transposed, out_scale):
        Y = out_scale * exact([x.vec for x in xa], [x.vec for x in xb], b_transposed)
        cls = type(xa[0])
        outs = []
        for m, y in enumerate(Y):
            o = cls(y, self.land, 1, xa[0].packing)
            if hasattr(xa[m], "ptag"):
                o.ptag = xa[m].ptag
            outs.append(o)
        return outs

    def _device(self, xa, xb, b_transposed, fa, fb, shift):
        ops, bts, S = self.ops, self.bts, self.ops.inf.slots
        bts.sync()
        t0 = time.perf_counter()
        da, db, states = [], [], []
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
