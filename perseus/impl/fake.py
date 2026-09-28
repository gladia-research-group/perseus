"""A numpy fake of the ``perseus._core`` surface the port uses, with the runtime's level
accounting (tests/test_rmsnorm_example.py pattern, extended to the whole leaf surface).

Levels are prime counts; every product (ct*ct, ct*pt, ct*scalar, square, negate) costs
``unit`` primes and, when the result reaches ``limit``, triggers the reactive bootstrap the
C++ ``maybe_bootstrap`` applies after products (fideslib_wrapper.h).
Adds, scalar adds/subs, rotations and conjugation are free. ``bootstrap`` lands at
``bts_out``. A ``strict`` fake raises instead of bootstrapping reactively, so the CPU tests
catch missing hints. ``rotate`` checks the rotation-key band. ``conjugate`` is the identity
(real payload).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class FakeCt:
    def __init__(self, vec, level, noise_deg=1, packing="cachemir"):
        vec = np.asarray(vec)
        self.vec = np.asarray(vec, dtype=np.complex128 if np.iscomplexobj(vec) else np.float64)
        self.level = int(level)
        self.noise_deg = int(noise_deg)
        self.packing = packing

    def __repr__(self):
        return f"<FakeCt L{self.level} d{self.noise_deg}>"


class FakeFhe:
    def __init__(self, rot_steps, limit=46, bts_out=34, unit=2, strict=False, slots=None,
                 complex_payload=False, sparse_slots=(512, 1)):
        self.loaded_rot_steps = sorted({int(s) for s in rot_steps})
        self.sparse_precomp_slots = sorted(int(x) for x in sparse_slots)   # the fold's built counts
        self.auto_bts_suppressed = False
        self._rot = set(self.loaded_rot_steps)
        self._limit = int(limit)
        self._bts_out = int(bts_out)
        self.unit = int(unit)
        self.strict = strict
        self.slots = slots
        self.n_bootstraps = 0
        self.n_reactive = 0
        self.calls = []
        self.complex_payload = bool(complex_payload)   # conjugate is the identity when False
        self.has_secret_key = True

    # ── level bookkeeping ──
    def level_limit(self): return self._limit
    def bootstrap_output_level(self): return self._bts_out
    def complete_setup(self): pass

    def _product(self, vec, level, packing):
        level = level + self.unit
        out = FakeCt(vec, level, 1, packing)
        if level >= self._limit and not self.auto_bts_suppressed:
            if self.strict:
                raise RuntimeError(f"reactive bootstrap at level {level} (strict fake)")
            self.n_reactive += 1
            self.bootstrap(out)
        return out

    @staticmethod
    def _lvl(a, b):
        return max(a.level, b.level)

    # ── arithmetic ──
    def add(self, a, b):
        if isinstance(b, FakeCt):
            return FakeCt(a.vec + b.vec, self._lvl(a, b), max(a.noise_deg, b.noise_deg), a.packing)
        return FakeCt(a.vec + float(b), a.level, a.noise_deg, a.packing)

    def sub(self, a, b):
        if isinstance(b, FakeCt):
            self.calls.append("sub_cc")
            return FakeCt(a.vec - b.vec, self._lvl(a, b), max(a.noise_deg, b.noise_deg), a.packing)
        return FakeCt(a.vec - float(b), a.level, a.noise_deg, a.packing)

    def mult(self, a, b):
        if isinstance(b, FakeCt):
            self.calls.append("mult_cc")
            return self._product(a.vec * b.vec, self._lvl(a, b), a.packing)
        self.calls.append("mult_sc")
        return self._product(a.vec * float(b), a.level, a.packing)

    def square(self, a):
        self.calls.append("square")
        return self._product(a.vec * a.vec, a.level, a.packing)

    def negate(self, a):
        return FakeCt(-a.vec, a.level, a.noise_deg, a.packing)

    def inplace_add(self, a, b):
        a.vec = a.vec + b.vec
        a.level = self._lvl(a, b)

    def rotate(self, a, steps):
        steps = int(steps)
        if self.slots is not None:
            k = steps % self.slots
            if k and k not in self._rot and (k - self.slots) not in self._rot and steps not in self._rot:
                raise KeyError(f"no rotation key for {steps}")
        self.calls.append(f"rotate{steps}")
        return FakeCt(np.roll(a.vec, -steps), a.level, a.noise_deg, a.packing)

    def rotate_and_sum(self, a, start, stop):
        gap, sign = abs(int(start)), (1 if start > 0 else -1)
        x = self.add(a, 0.0)
        while gap < stop:
            x = self.add(x, self.rotate(x, sign * gap))
            gap *= 2
        return x

    def conjugate(self, a):
        self.calls.append("conjugate")
        return FakeCt(np.conj(a.vec) if self.complex_payload else a.vec.copy(),
                      a.level, a.noise_deg, a.packing)

    def mult_i(self, a):
        self.calls.append("mult_i")
        return FakeCt(a.vec * 1j, a.level, a.noise_deg, a.packing)

    def rotate_hoisted(self, a, steps):
        return [self.rotate(a, k) for k in steps]

    def mult_add_many(self, acc, vs, ss):
        for v, t in zip(vs, ss):
            self.inplace_add(acc, self.mult(v, t))
        return True

    # ── bootstrapping (in place) ──
    def bootstrap(self, a):
        self.n_bootstraps += 1
        self.calls.append("bootstrap")
        a.level = self._bts_out
        a.noise_deg = 1

    # folds (fideslib_wrapper.h fold_slots_for / fold_bootstrap)
    def fold_slots_for(self, s_wanted):
        S = self.slots
        if s_wanted >= S or s_wanted in self.sparse_precomp_slots:
            return min(s_wanted, S)
        for s in self.sparse_precomp_slots:
            if s > s_wanted:
                return s
        return S

    def fold_bootstrap(self, a, s, n_live, prescale=1.0):
        S = self.slots
        if s != S and s not in self.sparse_precomp_slots:
            raise RuntimeError(f"fold_bootstrap: no precomputation for s={s}")
        if a.level >= self._limit + self.unit:
            raise RuntimeError("fold_bootstrap: input past the envelope")
        v = a.vec * prescale
        gap = s
        while gap < S:                       # the fold: the class sums over the S/s copies
            v = v + np.roll(v, -gap); gap *= 2
        a.vec = v * (1.0 / n_live / prescale)
        self.calls.append("fold_bootstrap")
        self.bootstrap(a)

    def suppress_auto_bts(self):
        fhe = self
        class _S:
            def __enter__(s_):
                s_.saved = fhe.auto_bts_suppressed; fhe.auto_bts_suppressed = True
            def __exit__(s_, *a):
                fhe.auto_bts_suppressed = s_.saved
        return _S()

    def maybe_bootstrap(self, a):
        if a.level >= self._limit:
            self.bootstrap(a)

    def bootstrap_hint(self, a, thr, account_pending_rescale=False):
        eff = a.level + (self.unit if (account_pending_rescale and a.noise_deg == 2) else 0)
        if eff > int(thr):
            self.bootstrap(a)

    def level_hint(self, a, level):
        pass


class FakeInf:
    """Inference fake: ``size`` fields, ``mult_pt``/``add_pt`` and the module-level data-plane
    functions the port reaches through ``fake.core``."""

    def __init__(self, fhe, N, hid, dim, H, H_real, E, E_real):
        self.fhe = fhe
        self.slots = int(N)
        self.size = SimpleNamespace(hidDim=hid, dim=dim, numHeads=H, numHeadsReal=H_real,
                                    expDim=E, expanded=E_real, seqLen=1024)
        self.complex = False
        self.slot_pts = {}
        self.enc_cache = {}
        self.enc_hits = self.enc_misses = 0
        self.capture_t = 0
        self.capture_b = 0
        self.block_prefix = ""
        self.logN = int(N).bit_length()

    def step(self, label):
        import contextlib
        return contextlib.nullcontext()

    def name_ct_if_absent(self, ct, name):
        pass

    def name_ct(self, ct, name):
        pass

    def _full(self, values):
        vals = np.asarray(values).ravel()
        v = np.zeros(self.slots, dtype=np.complex128 if np.iscomplexobj(vals) else np.float64)
        v[:vals.shape[0]] = vals
        return v

    def mult_pt(self, a, values):
        self.fhe.calls.append("mult_pt")
        return self.fhe._product(a.vec * self._full(values), a.level, a.packing)

    def add_pt(self, a, values):
        return FakeCt(a.vec + self._full(values), a.level, a.noise_deg, a.packing)

    def mult_i(self, a):                      # level-free, like the runtime's monomial multiply
        self.fhe.calls.append("mult_i")
        return FakeCt(a.vec * 1j, a.level, a.noise_deg, a.packing)

    def mult_const(self, a, re, im):
        self.fhe.calls.append("mult_const")
        return self.fhe._product(a.vec * complex(re, im), a.level, a.packing)

    # the named-plaintext extension (set_slot_pt & co.), level-agnostic on the fake
    def set_slot_pt(self, name, values, level=0, deg=1):
        self.slot_pts[name] = self._full(values)

    def has_slot_pt(self, name):
        return name in self.slot_pts

    def mult_slot_pt(self, a, name):
        self.fhe.calls.append("mult_pt")
        return self.fhe._product(a.vec * self.slot_pts[name], a.level, a.packing)

    def add_slot_pt(self, a, name):
        return FakeCt(a.vec + self.slot_pts[name], a.level, a.noise_deg, a.packing)

    def tag_reduce(self, a, stride):
        a.period = int(stride)

    def mult_slot_pt_many(self, a, names):
        return [self.mult_slot_pt(a, n) for n in names]

    # the encode cache (mult_cached & co.): tag + level keyed like inf.enc_cache
    def _cache_key(self, tag, a):
        lv = a.level + (self.fhe.unit if a.noise_deg == 2 else 0)
        return f"{tag}#L{lv}"

    def _cached(self, tag, a, values):
        k = self._cache_key(tag, a)
        v = self.enc_cache.get(k)
        if v is None:
            self.enc_misses += 1
            v = self._full(values)
            self.enc_cache[k] = v
        else:
            self.enc_hits += 1
        return v

    def mult_cached(self, a, tag, values, tagged=True):
        self.fhe.calls.append("mult_pt")
        return self.fhe._product(a.vec * self._cached(tag, a, values), a.level, a.packing)

    def add_cached(self, a, tag, values, tagged=True):
        return FakeCt(a.vec + self._cached(tag, a, values), a.level, a.noise_deg, a.packing)

    def mult_cached_many(self, a, tags, values):
        return [self.mult_cached(a, t, v) for t, v in zip(tags, values)]

    def prime_pt(self, tag, values, level):
        k = f"{tag}#L{int(level)}"
        if k not in self.enc_cache:
            self.enc_cache[k] = self._full(values)

    def stage_pts(self, items):
        return list(items)          # "encoded on the worker": adopted later

    def adopt_pts(self, handle):
        n = 0
        for tag, level, values in handle:
            k = f"{tag}#L{int(level)}"
            if k not in self.enc_cache:
                self.prime_pt(tag, values, level)
                n += 1
        return n

    def erase_pts(self, tags):
        n = 0
        for tag in tags:
            ks = [k for k in self.enc_cache if k.startswith(tag + "#L")]
            for k in ks:
                del self.enc_cache[k]
            n += len(ks)
        return n

    def enc_cache_stats(self):
        return {"size": len(self.enc_cache), "hits": self.enc_hits, "misses": self.enc_misses,
                "strict_misses": 0}

    def load_slot_pts(self, prefix):
        return sum(1 for k in self.slot_pts if k.startswith(prefix))

    def evict_slot_pts(self, prefix):
        return sum(1 for k in self.slot_pts if k.startswith(prefix))

    def drop_slot_pts(self, prefix):
        ks = [k for k in self.slot_pts if k.startswith(prefix)]
        for k in ks:
            del self.slot_pts[k]
        return len(ks)


class _Core:
    """The data-plane module functions (``_core.*``) on the fake."""

    @staticmethod
    def encode_token_input(inf, x):
        from .layout import lane_vec
        t = inf.slots // inf.size.hidDim
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] > inf.size.dim:
            raise ValueError("encode_token_input: too many features")
        return FakeCt(lane_vec(x, inf.slots, t), inf.fhe.bootstrap_output_level())

    @staticmethod
    def pack_tokens(inf, rows, target_level=0):
        t = inf.slots // inf.size.hidDim
        rows = np.asarray(rows, dtype=np.float64)
        v = np.zeros(inf.slots)
        for tok, row in enumerate(rows):
            v[np.arange(row.shape[0]) * t + tok] = row
        return FakeCt(v, target_level)

    @staticmethod
    def decrypt_slots(inf, a):
        return np.real(a.vec).copy()

    @staticmethod
    def decrypt_slots_complex(inf, a):
        return np.asarray(a.vec, dtype=np.complex128).copy()

    @staticmethod
    def decode_token_output(inf, a):
        t = inf.slots // inf.size.hidDim
        return np.real(a.vec[np.arange(inf.size.dim) * t]).copy()

    @staticmethod
    def decode_linear_output(inf, a, d_in, d_out):
        from .layout import decode_linear_output
        return decode_linear_output(np.real(a.vec), inf.slots, d_in, d_out)

    # graph / plan controls: no-ops on the fake
    @staticmethod
    def block_scope(b):
        return f"block_{b}."

    @staticmethod
    def reset_graph_runtime(inf):
        pass

    @staticmethod
    def begin_subgraph_capture(inf, b):
        return False

    @staticmethod
    def end_subgraph_capture(inf, b):
        pass

    @staticmethod
    def install_plan_live(inf, plan):
        pass

    @staticmethod
    def realize_pending_rescale(inf, a):
        if a.noise_deg == 2:
            a.noise_deg = 1
            a.level += inf.fhe.unit


core = _Core()


def make_fake(N=1024, hid=32, dim=24, H=4, H_real=3, E=128, E_real=96, limit=46, bts_out=34,
              unit=2, strict=False, rot_steps=None, complex_payload=False, sparse_slots=(512, 1)):
    """A small fake session: (fhe, inf). The rotation band covers every step the port
    needs at these dims (linear shapes hid->hid, hid->E, E->hid, hid->N, N->hid, the
    attention/norm ladders)."""
    from .layout import Dims, attention_rot_steps, linear_rot_steps
    dims = Dims(N, hid, dim, H, H_real, E, E_real)
    if rot_steps is None:
        steps = set()
        for d_in, d_out in ((hid, hid), (hid, E), (E, hid), (hid, N), (N, hid)):
            steps |= linear_rot_steps(N, d_in, d_out)
        steps |= attention_rot_steps(dims)
        rot_steps = steps | {(-s) % N for s in steps} | {s - N for s in steps}
    fhe = FakeFhe(rot_steps, limit=limit, bts_out=bts_out, unit=unit, strict=strict, slots=N,
                  complex_payload=complex_payload, sparse_slots=sparse_slots)
    inf = FakeInf(fhe, N, hid, dim, H, H_real, E, E_real)
    return fhe, inf
