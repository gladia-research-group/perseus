"""Plaintext-ciphertext matrix products at the StC-first split point of the bootstrap (ePrint 2024/1284 Alg. 2;
Sylph, arXiv 2601.18511): y = x W on the dense token layout via ring switch + int8 GEMM (FIDESlib Pcmm.cuh).
Needs FIDESLIB_BTS_STC_FIRST=2 and FIDESLIB_BTS_STC_FIRST_STAGED_ONLY=1."""
from __future__ import annotations

import hashlib
import math
import os
import secrets
import time
from dataclasses import dataclass

import numpy as np

N, S_SWITCH = 1 << 16, 16
N1 = N // S_SWITCH
SIGMA = 3.2


def brev(x, bits):
    x = np.asarray(x, dtype=np.int64)
    out = np.zeros_like(x)
    for b in range(bits):
        out |= ((x >> b) & 1) << (bits - 1 - b)
    return out


@dataclass(frozen=True)
class Shape:
    """MLWE shape for t lanes: pieces of degree np = 2t, rank r = N1 / np, nch = 16 r channels."""
    t: int

    @property
    def np(self): return 2 * self.t
    @property
    def r(self): return N1 // self.np
    @property
    def nch(self): return S_SWITCH * self.r

    def ch_of_row(self):
        """Channel held by GEMM row sub * r + comp (folded into W)."""
        idx = np.arange(self.nch)
        return brev(idx // self.r + S_SWITCH * (idx % self.r), self.nch.bit_length() - 1)


# ── primes / negacyclic NTT (numpy, key generation only) ──────────────────────────────────────────────────────────

def _is_prime(p):
    if p < 2:
        return False
    for f in (2, 3, 5, 7, 11, 13):
        if p % f == 0:
            return p == f
    d, s = p - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23):
        x = pow(a, d, p)
        if x in (1, p - 1):
            continue
        for _ in range(s - 1):
            x = x * x % p
            if x == p - 1:
                break
        else:
            return False
    return True


def find_prime(bits, m, avoid):
    p = ((1 << bits) - 1) // m * m + 1
    while p > (1 << (bits - 1)):
        if p not in avoid and _is_prime(p):
            return p
        p -= m
    raise ValueError("no prime")


def _primitive_root(p):
    x, fs, f = p - 1, [], 2
    while f * f <= x:
        if x % f == 0:
            fs.append(f)
            while x % f == 0:
                x //= f
        f += 1
    if x > 1:
        fs.append(x)
    g = 2
    while not all(pow(g, (p - 1) // f, p) != 1 for f in fs):
        g += 1
    return g


def _powers(w, n, p):
    B = 256
    base = np.empty(min(B, n), dtype=np.int64)
    acc = 1
    for j in range(len(base)):
        base[j] = acc
        acc = acc * w % p
    out = np.empty(n, dtype=np.int64)
    blk = 1
    for j0 in range(0, n, B):
        m = min(B, n - j0)
        out[j0:j0 + m] = base[:m] * blk % p
        blk = blk * acc % p
    return out


_TAB = {}


def _tables(p, n):
    if (p, n) not in _TAB:
        assert (p - 1) % (2 * n) == 0
        psi = pow(_primitive_root(p), (p - 1) // (2 * n), p)
        rev = brev(np.arange(n), n.bit_length() - 1)
        om = psi * psi % p
        iom = pow(om, p - 2, p)
        st, ist, ln = [], [], 2
        while ln <= n:
            st.append(_powers(pow(om, n // ln, p), ln // 2, p))
            ist.append(_powers(pow(iom, n // ln, p), ln // 2, p))
            ln *= 2
        _TAB[(p, n)] = (psi, _powers(psi, n, p), _powers(pow(psi, p - 2, p), n, p), rev, st, ist, pow(n, p - 2, p))
    return _TAB[(p, n)]


def _cyclic(a, p, rev, stages):
    n = a.shape[-1]
    a = a[..., rev]
    ln = 2
    for w in stages:
        a = a.reshape(a.shape[:-1] + (n // ln, ln))
        u, v = a[..., : ln // 2], a[..., ln // 2:] * w % p
        a = np.concatenate([(u + v) % p, (u - v) % p], axis=-1).reshape(a.shape[:-2] + (n,))
        ln *= 2
    return a


def _polymul(a, b, p):
    _, pp, ip, rev, st, ist, ninv = _tables(p, a.shape[-1])
    fa = _cyclic(a % p * pp % p, p, rev, st)
    fb = _cyclic(b % p * pp % p, p, rev, st)
    return _cyclic(fa * fb % p, p, rev, ist) * ninv % p * ip % p


def _twiddles(p, n):
    """psi^brev(k), psi^-brev(k) (k < n) and n^-1: the device's NTT tables."""
    _, pp, ip, rev, _, _, ninv = _tables(p, n)
    return pp[rev].astype(np.uint32), ip[rev].astype(np.uint32), ninv


class KeyRng:
    """Key-material sampler: SHAKE-256 in counter mode under a fresh 256-bit OS key (numpy's generators are not
    cryptographic, and a seeded one publishes the keys)."""

    def __init__(self):
        self._key, self._ctr = secrets.token_bytes(32), 0

    def _bytes(self, n):
        self._ctr += 1
        return hashlib.shake_256(self._key + self._ctr.to_bytes(8, "little")).digest(n)

    def integers(self, lo, hi, size, dtype=np.int64):
        """Uniform on [lo, hi), hi - lo <= 2^32, by rejection."""
        m = hi - lo
        lim = (1 << 32) // m * m
        out, k = [], 0
        while k < size:
            u = np.frombuffer(self._bytes(4 * ((size - k) * (1 << 32) // lim + 64)), dtype=np.uint32).astype(np.int64)
            u = u[u < lim][: size - k] % m
            out.append(u)
            k += u.size
        return (np.concatenate(out) + lo).astype(dtype)

    def normal(self, mu, sigma, size):
        """Box-Muller on 53-bit uniforms."""
        h = (size + 1) // 2
        u = (np.frombuffer(self._bytes(16 * h), dtype=np.uint64) >> np.uint64(11)).astype(np.float64) / 2.0 ** 53
        r = np.sqrt(-2.0 * np.log(1.0 - u[:h]))
        z = np.concatenate([r * np.cos(2 * np.pi * u[h:]), r * np.sin(2 * np.pi * u[h:])])
        return mu + sigma * z[:size]


def _ks_key(s_from, s_to, q, P, rng):
    """[digit][prime][a, b][n] coefficient form: key_j = (a, -a s_to + e + P (Q / q_j) s_from) over q + [P]."""
    n = s_from.shape[-1]
    Q = math.prod(q)
    big = list(q) + [P]
    out = np.empty((len(q), len(big), 2, n), dtype=np.uint32)
    for j, qj in enumerate(q):
        g = P * (Q // qj)
        e = np.rint(rng.normal(0, SIGMA, size=n)).astype(np.int64)
        for i, p in enumerate(big):
            a = rng.integers(0, p, size=n, dtype=np.int64)
            b = (-_polymul(a, s_to % p, p) + e + s_from % p * (g % p)) % p
            out[j, i, 0], out[j, i, 1] = a, b
    return out


def make_keys(sk, split_moduli, t):
    """Device key dict for bts.pcmm_keys: ring-switch keys to s'(X^16) and back, r pack keys, over {q0, q1, P}."""
    sh = Shape(t)
    rng = KeyRng()
    q = [int(m) for m in split_moduli[:2]]
    P = find_prime(31, 2 * N, set(int(m) for m in split_moduli))
    sk = np.asarray(sk, dtype=np.int64)
    s1 = rng.integers(-1, 2, size=N1).astype(np.int64)
    emb = np.zeros(N, dtype=np.int64)
    emb[::S_SWITCH] = s1
    comp = s1.reshape(sh.np, sh.r).T                      # s_l = component l of s'
    pack = []
    for l in range(sh.r):
        e = np.zeros(N1, dtype=np.int64)
        e[::sh.r] = comp[l]
        pack.append(_ks_key(e, s1, q, P, rng))
    big = q + [P]
    d = dict(primes=np.array(big, dtype=np.uint32), split=np.array([int(m) for m in split_moduli], dtype=np.uint32),
             psi16=[], ipsi16=[], psi12=[], ipsi12=[], ninv16=[], ninv12=[])
    for p in big:
        a, b, ni = _twiddles(p, N); d["psi16"].append(a); d["ipsi16"].append(b); d["ninv16"].append(ni)
        a, b, ni = _twiddles(p, N1); d["psi12"].append(a); d["ipsi12"].append(b); d["ninv12"].append(ni)
    d["ninv16"] = np.array(d["ninv16"], dtype=np.uint32)
    d["ninv12"] = np.array(d["ninv12"], dtype=np.uint32)
    d["fwd"] = _ks_key(sk, emb, q, P, rng).ravel()
    d["back"] = _ks_key(emb, sk, q, P, rng).ravel()
    d["pack"] = np.stack(pack).ravel()
    return d


# ── weights ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def digit_planes(Wi, n_planes):
    """Balanced base-256 digits of an integer matrix: (n_planes, ...) int8 with Wi = sum_u planes[u] 256^u."""
    v = np.asarray(Wi, dtype=np.int64).copy()
    out = np.empty((n_planes,) + v.shape, dtype=np.int8)
    for u in range(n_planes):
        r = ((v + 128) % 256) - 128
        out[u] = r
        v = (v - r) // 256
    if np.any(v):
        raise ValueError("digit_planes: the matrix needs more planes")
    return out


def from_planes(Wd):
    acc = np.zeros(Wd.shape[1:], dtype=np.int64)
    for u in reversed(range(Wd.shape[0])):
        acc = acc * 256 + Wd[u]
    return acc


def weight_bits():
    """PF_PCMM_WBITS (default 16): integer weight width (8 bits per digit plane)."""
    return int(os.environ.get("PF_PCMM_WBITS", "16"))


@dataclass
class PcmmLinear:
    """y = x W (+ bias), W (n_in nch, n_out nch) as int8 digit planes Wd (nW, out, in) in channel order; w_scale is a
    power of two, `shift` puts |y| / 2^shift in the bootstrap input band."""
    n_in: int
    n_out: int
    Wd: np.ndarray
    w_scale: float
    shift: int
    bias: list | None = None
    name: str | None = None
    t: int = 0                  # token lanes (0: the runtime's)

    @classmethod
    def encode(cls, W, t, scale=1.0, bias=None, shift=0, name=None, wbits=None):
        sh = Shape(t)
        wbits = wbits or weight_bits()
        W = np.asarray(W, dtype=np.float64) * scale
        n_in, n_out = W.shape[0] // sh.nch, W.shape[1] // sh.nch
        assert W.shape == (n_in * sh.nch, n_out * sh.nch), "W must be whole nch blocks"
        ws = 2.0 ** math.floor(math.log2((2 ** (wbits - 1) - 1) / np.abs(W).max()))
        Wt = np.rint(W.T * ws).astype(np.int64)
        perm = sh.ch_of_row()
        Wi = np.empty_like(Wt)
        for o in range(n_out):
            for c in range(n_in):
                blk = Wt[o * sh.nch:(o + 1) * sh.nch, c * sh.nch:(c + 1) * sh.nch]
                Wi[o * sh.nch:(o + 1) * sh.nch, c * sh.nch:(c + 1) * sh.nch] = blk[perm[:, None], perm[None, :]]
        bvec = None
        if bias is not None:
            bias = np.asarray(bias, dtype=np.float64) * scale
            bvec = [np.repeat(bias[m * sh.nch:(m + 1) * sh.nch], t) for m in range(n_out)]
        return cls(n_in, n_out, digit_planes(Wi, (wbits + 7) // 8), ws, int(shift), bvec, name, int(t))


# ── band shifts ─────────────────────────────────────────────────────────────────────────────────────────────────────
# PF_PCMM_CALIB=<json>: a fake run records each linear's max |y|; PF_PCMM_CALIB_IN=<json>: shifts from it
# (PF_PCMM_MARGIN, PF_PCMM_BAND); PF_PCMM_SHIFT forces one shift.
_calib: dict = {}
_shifts = None


def band_shift(name):
    global _shifts
    import json
    if os.environ.get("PF_PCMM_SHIFT"):
        return int(os.environ["PF_PCMM_SHIFT"])
    if os.environ.get("PF_PCMM_CALIB"):
        return 0                                   # calibrating: the fake is exact whatever the shift
    if _shifts is None:
        path = os.environ.get("PF_PCMM_CALIB_IN", "")
        _shifts = json.load(open(path)) if path else {}
    if name not in _shifts:
        raise KeyError(f"pcmm: no calibrated max |y| for {name} (PF_PCMM_CALIB_IN / PF_PCMM_SHIFT)")
    band = float(os.environ.get("PF_PCMM_BAND", "0.25"))
    ymax = _shifts[name] * float(os.environ.get("PF_PCMM_MARGIN", "1.5"))
    return max(0, math.ceil(math.log2(ymax / band)))


def _record_calib(name, ys):
    import json
    path = os.environ.get("PF_PCMM_CALIB", "")
    if not path or not name:
        return
    m = max(float(np.abs(y).max()) for y in ys)
    _calib[name] = max(_calib.get(name, 0.0), m)
    with open(path, "w") as f:
        json.dump(_calib, f, indent=1, sort_keys=True)


def engine(rt, t=None):
    """The runtime's PCMM engine for t token lanes (made on first use)."""
    t = int(t or rt.dims.t)
    engines = rt.__dict__.setdefault("_pcmm_engines", {})
    if t not in engines:
        engines[t] = Pcmm(rt.ops, t)
    rt.pcmm = engines[t]
    return engines[t]


def run(rt, xs, lin):
    """lin on the dense blocks xs through the runtime's PCMM engine."""
    return engine(rt, lin.t).apply(xs, lin)


def pinned_planes(core, a):
    """`a` (int8 digit planes) copied into pinned host memory; `a` itself when that is unavailable."""
    alloc = getattr(core, "pinned_empty", None)
    a = np.ascontiguousarray(a, dtype=np.int8)
    if alloc is None or a.shape[-1] % 4:
        return a
    out = np.asarray(alloc(int(np.prod(a.shape[:-1])), a.shape[-1] // 4)).view(np.int8).reshape(a.shape)
    out[:] = a
    return out


# ── the op ──────────────────────────────────────────────────────────────────────────────────────────────────────────

class Pcmm:
    """The session's PCMM engine, recorded as rotate(x, -1) + hint/level_hint(cap) per input, one external bootstrap
    per output, rotate(y, 1); the fake computes the exact integer products."""

    def __init__(self, ops, t):
        self.ops, self.t = ops, t
        self.core = ops.core
        self.fhe = ops.fhe
        self.bts = getattr(ops.core, "bts", None)
        self.sh = Shape(t)
        self.keys = None
        self.times = np.zeros(6)
        self.calls = 0
        self.keygen_s = self.t_in = self.t_mm = self.t_out = 0.0
        # StC-first entry / landing levels: measured on the GPU, n32 values on the fake (PF_PCMM_CAP / _LAND)
        self.cap = int(os.environ.get("PF_PCMM_CAP", "0")) or None
        self.land = int(os.environ.get("PF_PCMM_LAND", "0")) or None
        if self.bts is None:
            self.cap = self.cap or _N32_CAP
            self.land = self.land or _N32_LAND

    def _ensure_keys(self, dev, moduli_first=None):
        if self.keys is None:
            t0 = time.perf_counter()
            ex = self.bts.export_limbs(self.fhe, dev)
            sk = self.bts.secret_key_coeffs(self.fhe)
            mod = ex["moduli"][:moduli_first] if moduli_first else ex["moduli"]
            self.keys = self.bts.pcmm_keys(make_keys(sk, mod, self.t))
            assert self.keys.channels == self.sh.nch
            self.keygen_s = time.perf_counter() - t0

    def prepare(self, dim):
        """GPU keys from the chain's lowest four moduli; `dim`: model width (for a probe ciphertext)."""
        if self.bts is not None and self.keys is None:
            x = self.ops.encode_token(np.zeros(dim))
            self._ensure_keys(self.bts.device(self.fhe, x), moduli_first=4)

    def _measure_cap(self, x):
        """The deepest level that still reaches the StC-first entry after a pending rescale."""
        ops, bts = self.ops, self.bts
        d = bts.device(self.fhe, x)
        entry = bts.stc_first_entry(d, ops.inf.slots)
        if entry < 0:
            raise RuntimeError("pcmm: the dense route has no StC-first precomputation (FIDESLIB_BTS_STC_FIRST=2)")
        total = ops.level(x) + d.level              # constant on the chain
        return total - entry - ops.unit             # a pending rescale drops a level first

    def apply(self, xs, lin: PcmmLinear):
        """The n_out outputs of lin on the n_in blocks xs, bias-free; the inputs are left unchanged."""
        ops = self.ops
        assert len(xs) == lin.n_in
        xr = [ops.rotate(x, -1) for x in xs]
        if self.cap is None:
            self.cap = self._measure_cap(xr[0])
        for x in xr:
            ops.bootstrap_hint(x, self.cap, True)
            self.fhe.level_hint(x, self.cap)
        names = [self.fhe.var_name(x) for x in xr]
        levels = [ops.level(x) for x in xr]
        want = None
        if self.bts is not None and os.environ.get("PF_PCMM_VERIFY"):     # the exact products of the decrypted inputs
            want = [_exact(self, [ops.decrypt_slots_complex(x) for x in xr], lin, m) for m in range(lin.n_out)]
        outs = self._device(xr, lin) if self.bts is not None else self._fake(xr, lin)
        if want is not None:
            got = [ops.decrypt_slots_complex(y) for y in outs]
            err = max(float(np.abs(g - w).max()) for g, w in zip(got, want))
            print(f"[pcmm_verify] {lin.name} n_in={lin.n_in} n_out={lin.n_out} shift={lin.shift} "
                  f"max|y|={max(float(np.abs(w).max()) for w in want):.3g} err={err:.3g}", flush=True)
        for y in outs:
            self.fhe.record_external_bootstrap(y, names, levels)
        self.calls += 1
        return [ops.rotate(y, 1) for y in outs]

    def _fake(self, xr, lin):
        from .fake import FakeCt
        t, nch = self.t, self.sh.nch
        inv = np.argsort(self.sh.ch_of_row())
        X = [np.roll(np.asarray(x.vec, dtype=np.complex128), -1).reshape(nch, t) for x in xr]
        Wi = from_planes(lin.Wd)
        outs = []
        for m in range(lin.n_out):
            y = np.zeros((nch, t), dtype=np.complex128)
            for c in range(lin.n_in):
                blk = Wi[m * nch:(m + 1) * nch, c * nch:(c + 1) * nch].astype(np.float64)[inv[:, None], inv[None, :]]
                y += blk @ X[c]
            o = FakeCt(np.roll(y.reshape(-1) / lin.w_scale, 1), self.land, 1, xr[0].packing)
            if hasattr(xr[0], "ptag"):       # fakecap: outputs reuse the input handles (spares: input 0)
                o.ptag = xr[m if m < lin.n_in else 0].ptag
            outs.append(o)
        _record_calib(lin.name, [o.vec for o in outs])
        return outs

    def _device(self, xr, lin):
        cf = os.environ.get("PF_PCMM_CF", "")    # the PCMM refresh's own correction factor (its outputs are band-scaled)
        if not cf:
            return self._device_cf(xr, lin)
        self.bts.set_correction_factor(self.fhe, int(cf))
        try:
            return self._device_cf(xr, lin)
        finally:
            self.bts.set_correction_factor(self.fhe, -1)

    def _device_cf(self, xr, lin):
        ops, bts, S = self.ops, self.bts, self.ops.inf.slots
        stamp = lin.w_scale * (1 << lin.shift)
        bts.sync()
        t0 = time.perf_counter()
        devs, outs, out_devs, states = [], [], [], []
        spare = []
        for x in xr:
            d = bts.device(self.fhe, x)
            st = bts.begin(d, S, False, True)
            if not spare and lin.n_out > lin.n_in:   # the spare outputs' states, taken at the input level
                spare = [bts.begin(d, S, False, True) for _ in range(lin.n_out - lin.n_in)]
            if not st.stc_first:
                raise RuntimeError("pcmm: the dense route is not StC-first (FIDESLIB_BTS_STC_FIRST=2)")
            bts.scale_noise_factor(d, stamp)
            bts.stc_first_input(d, st, True)
            devs.append(d)
            if len(outs) < lin.n_out:
                outs.append(x)
                out_devs.append(d)
                states.append(st)
        bts.sync()
        t1 = time.perf_counter()
        self.t_in += t1 - t0
        self._ensure_keys(devs[0])
        t1 = time.perf_counter()
        nf_out = bts.noise_factor(devs[0]) / lin.w_scale
        while len(outs) < lin.n_out:                            # spare outputs: clones of a split-level input
            y = bts.clone(outs[0])
            outs.append(y)
            out_devs.append(bts.device(self.fhe, y))
            states.append(spare.pop())
        # outputs overwrite inputs only after every input is consumed (pcmmMulti's contract)
        tm = bts.pcmm_multi_planes(devs, out_devs[:lin.n_out], self.keys, lin.Wd, nf_out)
        self.times += np.asarray(tm)
        t2 = time.perf_counter()
        self.t_mm += t2 - t1
        for m in range(lin.n_out):
            d, st = out_devs[m], states[m]
            bts.mod_raise(d, st)
            bts.fold(d, st)
            for k in range(st.n_cts):
                bts.cts_stage(d, st, k)
            bts.eval_mod(d, st)
            bts.stc_first_output(d, st)
            if lin.shift:
                bts.mult_int(d, 1 << lin.shift)
        bts.sync()
        self.t_out += time.perf_counter() - t2
        if self.land is None:
            self.land = ops.level(outs[0])
        return outs[:lin.n_out]


def _exact(pc, xs, lin, m):
    """Output block m of lin on the slot vectors xs (the split point's right-shifted layout)."""
    t, nch = pc.t, pc.sh.nch
    inv = np.argsort(pc.sh.ch_of_row())
    Wi = from_planes(lin.Wd)
    y = np.zeros((nch, t), dtype=np.complex128)
    for c, x in enumerate(xs):
        blk = Wi[m * nch:(m + 1) * nch, c * nch:(c + 1) * nch].astype(np.float64)[inv[:, None], inv[None, :]]
        y += blk @ np.roll(np.asarray(x, dtype=np.complex128), -1).reshape(nch, t)
    return np.roll(y.reshape(-1) / lin.w_scale, 1)


# the fake's cap / landing (n32 chain, FIDESLIB_BTS_STC_FIRST=2)
_N32_CAP = 44
_N32_LAND = 26


def rot_steps():
    """Rotations the op needs."""
    return {1, (1 << 15) - 1}


def enabled():
    return os.environ.get("PF_LINEAR", "") == "pcmm"
