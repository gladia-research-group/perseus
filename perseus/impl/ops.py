"""The leaf-primitive "ops" protocol: the same kernel code (poly.py, linear.py, norm.py,
attention.py, activation.py) runs on the FHE session (``FheOps`` over ``perseus._core``), on
the numpy fake session (``FheOps`` over ``fake``) and on plain numpy arrays (``NumpyOps``,
the plaintext mirror). ``TimedOps`` (profile.py) wraps ``FheOps`` with per-call timing.

Plaintext operands. ``mult_pt(a, v)`` / ``add_pt(a, v)`` encode ``v`` at every call (the leaf
primitive). With ``key=`` they go through the runtime's caches instead:
  * ``kind="weight"`` (the default for keyed calls): ``set_slot_pt`` + ``mult_slot_pt`` — a
    named, host-resident plaintext encoded ONCE at the level of its first use, streamed
    through the residency ring as part of a block state (see driver.py);
  * ``kind="mask"``: ``mult_cached`` — the runtime's encode cache (``inf.enc_cache``, keyed by
    tag and level), the path the C++ masks take: primed / staged ahead on a worker / evicted
    per step by the runtime (see rt.py).
"""
from __future__ import annotations

import contextlib
import math
import os

import numpy as np


def _accepts_iterations(fhe):
    """Whether this runtime's bootstrap takes an iteration count (the binding extension)."""
    doc = getattr(getattr(fhe, "bootstrap", None), "__doc__", "") or ""
    return "iterations" in doc


class FheOps:
    """The leaf-primitive surface of a session: ``inf`` is a ``_core.Inference`` (or the
    fake), ``core`` the module holding the data-plane functions (``perseus._core`` or
    ``fake.core``)."""

    def __init__(self, inf, core, unit: int = 2, bts_precision: int = 12):
        self.inf = inf
        self.fhe = inf.fhe
        self.core = core
        self.unit = int(unit)          # primes per CKKS level (composite_degree)
        self.bts_precision = int(bts_precision)
        self.n_bootstraps = 0
        self.n_encodes = 0
        self._slots = set()
        self._slot_name = {}      # (key, level) -> slot name: the f-string, built once
        self.slot_pts = hasattr(inf, "set_slot_pt")     # the named-plaintext extension
        self.enc_cache = hasattr(inf, "mult_cached")    # the runtime's encode cache (masks)
        self.native_bts_iters = _accepts_iterations(inf.fhe)   # bootstrap(ct, iterations)
        self.native_rotsum = hasattr(inf.fhe, "rotate_and_sum")  # the C++ ladder
        # CKKS_COMPLEX=1: the slots carry complex values (conjugate is a real operation, a
        # ciphertext can hold two real payloads as its real and imaginary lanes)
        self.complex_payload = bool(getattr(inf.fhe, "complex_payload", False))
        # the fused reductions (FUSED_SM_DEN / FUSED_LN_VAR, the C++ switches of the same
        # name): a rotate-and-sum ladder stopped at the sparse slot count and finished inside
        # a sparse bootstrap ("the fold"), one refresh instead of a ladder plus a refresh
        self.folds = hasattr(inf.fhe, "fold_bootstrap")
        self.fused_sm_den = self.folds and os.environ.get("FUSED_SM_DEN", "0") not in ("", "0")
        self.fused_ln_var = self.folds and os.environ.get("FUSED_LN_VAR", "0") not in ("", "0")
        # SM_DEN_RECIP=1: the softmax divides by building 1/denominator on the denominator's own
        # (tH-periodic, sparse-routed) ciphertext and multiplying the scores by it once per round,
        # instead of multiplying the scores at every Goldschmidt iteration (attention.softmax_thor)
        self.sm_den_recip = os.environ.get("SM_DEN_RECIP", "1") not in ("", "0")
        # GELU_FOLD=1: the GELU's 1/xmax rides the up-projection weights and its refreshes are
        # placed by the planner instead of its own hints (activation.gelu)
        self.gelu_fold = os.environ.get("GELU_FOLD", "1") not in ("", "0")
        # LN_CHEB=1: the LayerNorm inverse sqrt from the config's Chebyshev seed (norm.norm);
        # SM_GS_FIRST=1: the softmax's first division at its gs_iters_first count
        self.ln_cheb = os.environ.get("LN_CHEB", "1") not in ("", "0")
        self.sm_gs_first = os.environ.get("SM_GS_FIRST", "1") not in ("", "0")
        # SM_PERIODIC=1: while every cached token fits one group (kc <= t) the scores, the softmax and its
        # denominators stay tH-periodic instead of living in block 0, so their refreshes route sparse and P.V is one
        # product (attention.qkt / softmax_thor / softmax_v)
        self.sm_periodic = os.environ.get("SM_PERIODIC", "1") not in ("", "0")
        # SM_FOLD=1: the softmax's ciphertext x constant products folded into masks (the exp's Chebyshev affine into
        # the q.K^T group and score masks; each reciprocal round's seed slope into a second head-sum mask)
        self.sm_fold = os.environ.get("SM_FOLD", "1") not in ("", "0")
        # KV_LANES=1: the K + iV linear output refreshed as one REAL payload (K on token lane 0, V on lane 1) by the
        # real-payload route instead of a complex refresh (attention.cache_kv_push_packed_complex)
        self.kv_lanes = os.environ.get("KV_LANES", "0") not in ("", "0")

    # arithmetic
    def add(self, a, b):
        return self.fhe.add(a, b) if not isinstance(b, (int, float, np.floating)) \
            else self.fhe.add(a, float(b))

    def sub(self, a, b):
        return self.fhe.sub(a, b) if not isinstance(b, (int, float, np.floating)) \
            else self.fhe.sub(a, float(b))

    def mult(self, a, b):
        return self.fhe.mult(a, b) if not isinstance(b, (int, float, np.floating)) \
            else self.fhe.mult(a, float(b))

    def square(self, a): return self.fhe.square(a)
    def negate(self, a): return self.fhe.negate(a)
    def rotate(self, a, k): return self.fhe.rotate(a, int(k))

    def rotate_and_sum(self, a, start, stop):
        """x += rotate(x, g) for g = start, 2 start, ... |g| < stop: one C++ call when the
        runtime has it (Context.rotate_and_sum), else the loop of leaf ops (same op stream)."""
        start, stop = int(start), int(stop)
        if self.native_rotsum:
            return self.fhe.rotate_and_sum(a, start, stop)
        gap, sign = abs(start), (1 if start > 0 else -1)
        if gap >= stop:
            return self.copy(a)
        x = self.add(a, self.rotate(a, sign * gap))
        gap *= 2
        while gap < stop:
            self.inplace_add(x, self.rotate(x, sign * gap))
            gap *= 2
        return x
    def conjugate(self, a): return self.fhe.conjugate(a)
    def inplace_add(self, a, b): self.fhe.inplace_add(a, b)
    def copy(self, a): return self.fhe.add(a, 0.0)
    # Plaintext operands. Without a key the vector is encoded at every call (the leaf
    # primitive). With a key and a runtime that has `set_slot_pt` (the named-plaintext
    # extension), the vector is encoded ONCE at the level of its first use and reused —
    # what the C++ does with its pre-encoded weights and its encode cache.
    def _slot(self, key, a, v):
        # keyed by name AND level: a site whose level alternates between tokens keeps one
        # host copy per level instead of re-encoding (the runtime's relevel) at every visit
        # The level AND degree a plaintext must carry to meet `a` without forcing a rescale is
        # chain-dependent, so ask the runtime rather than assuming level+pending at degree 1:
        # that assumption is only right on a composite chain, and on a d=1 one it realizes a
        # rescale the C++ composites leave pending.
        lv, deg = self._encode_like(a)
        ck = (key, lv, deg)
        name = self._slot_name.get(ck)
        if name is not None:
            return name
        name = f"impl.{key}@{lv}d{deg}"
        if name not in self._slots:
            if not self.inf.has_slot_pt(name):
                self.inf.set_slot_pt(name, self._vec(v), lv, deg)
                self.n_encodes += 1
            self._slots.add(name)
        self._slot_name[ck] = name
        return name

    def _encode_like(self, a):
        """(level, noise_deg) a plaintext needs to meet `a`, by the runtime's own rule."""
        f = getattr(self.inf, "encode_like_level", None)
        if f is None:                      # older core: the composite-chain rule
            return self.lvl(a), 1
        lv, deg = f(a)
        return int(lv), int(deg)

    @staticmethod
    def _vec(v):
        """A contiguous float64 vector, or complex128 for a complex mask / weight."""
        v = np.asarray(v)
        return np.ascontiguousarray(v, dtype=np.complex128 if np.iscomplexobj(v) else np.float64)

    # `_vec` is applied where the vector is actually handed to the runtime: on the slot path
    # a registered plaintext never looks at `v` again, and that path is every weight product.
    def mult_pt(self, a, v, key=None, kind="weight", tagged=True):
        if key is not None:
            if kind == "mask" and self.enc_cache:
                return self.inf.mult_cached(a, key, self._vec(v), tagged)
            if self.slot_pts:
                return self.inf.mult_slot_pt(a, self._slot(key, a, v))
        return self.inf.mult_pt(a, self._vec(v))

    def add_pt(self, a, v, key=None, kind="weight", tagged=True):
        if key is not None:
            if kind == "mask" and self.enc_cache:
                return self.inf.add_cached(a, key, self._vec(v), tagged)
            if self.slot_pts:
                return self.inf.add_slot_pt(a, self._slot(key, a, v))
        return self.inf.add_pt(a, self._vec(v))

    def mult_const(self, a, re, im):
        """a * (re + i im): a plaintext product with a complex constant (the runtime's cached
        complex constant; costs a level like a mask). The packing primitive: a + i b =
        add(a, mult_const(b, 0, 1)); the realify: mult_const(x, 0, -1) turns 2i b into 2 b."""
        if hasattr(self.inf, "mult_const"):
            return self.inf.mult_const(a, float(re), float(im))
        return self.inf.mult_pt(a, np.full(self.inf.slots, complex(re, im), dtype=np.complex128))

    def mult_i(self, a):
        """i * a, level-free (the runtime's monomial multiply); falls back to a constant
        product (one level) on a runtime without it."""
        if hasattr(self.fhe, "mult_i"):
            return self.fhe.mult_i(a)
        return self.mult_const(a, 0.0, 1.0)

    def device_sync(self):
        """A device fence (every stream), when the runtime exposes one."""
        if hasattr(self.core, "device_sync"):
            self.core.device_sync()

    def decrypt_slots_complex(self, a):
        return np.asarray(self.core.decrypt_slots_complex(self.inf, a), dtype=np.complex128)

    # Batched forms of the primitives above. Each one records the SAME op stream as the loop
    # it replaces (the runtime replays the per-lane nodes), so plans stay valid.
    def rotate_many(self, a, steps):
        """Hoisted rotations: one key-switch decomposition shared by every step."""
        steps = [int(k) for k in steps]
        if not steps:
            return []
        if hasattr(self.fhe, "rotate_hoisted"):
            return list(self.fhe.rotate_hoisted(a, steps))
        return [self.fhe.rotate(a, k) for k in steps]

    def mult_add_many(self, acc, vs, ss):
        """acc += sum_j vs[j]*ss[j], one relinearization when the runtime can fuse."""
        if hasattr(self.fhe, "mult_add_many"):
            return self.fhe.mult_add_many(acc, list(vs), list(ss))
        for v, t in zip(vs, ss):
            self.fhe.inplace_add(acc, self.fhe.mult(v, t))
        return False

    def mult_pt_many(self, a, vecs, keys, kind="weight"):
        """ct * each plaintext vector, as one fused lane batch when named and cached."""
        if all(k is not None for k in keys):
            if kind == "mask" and self.enc_cache and hasattr(self.inf, "mult_cached_many"):
                vecs = [self._vec(v) for v in vecs]
                return list(self.inf.mult_cached_many(a, list(keys), vecs))
            if kind != "mask" and self.slot_pts and hasattr(self.inf, "mult_slot_pt_many"):
                names = [self._slot(k, a, v) for k, v in zip(keys, vecs)]
                return list(self.inf.mult_slot_pt_many(a, names))
        return [self.mult_pt(a, v, key=k, kind=kind) for v, k in zip(vecs, keys)]

    # the encode cache's staging surface (rt.py drives it; no-ops without the extension)
    def prime_pt(self, tag, v, level):
        if self.enc_cache:
            self.inf.prime_pt(tag, self._vec(v), int(level))
            self.n_encodes += 1

    def stage_pts(self, items):
        """items: [(tag, level, vec)] -> a handle for adopt_pts (None without the extension)."""
        if not self.enc_cache or not items:
            return None
        self.n_encodes += len(items)
        return self.inf.stage_pts([(t, int(l), self._vec(v)) for t, l, v in items])

    def adopt_pts(self, handle):
        return self.inf.adopt_pts(handle) if (self.enc_cache and handle is not None) else 0

    def erase_pts(self, tags):
        if self.enc_cache and tags:
            return self.inf.erase_pts(list(tags))
        return 0

    def enc_cache_stats(self):
        return dict(self.inf.enc_cache_stats()) if self.enc_cache else {}

    def tag_reduce(self, a, stride):
        """Stamp a reduction output's slot period (cachemir_norm_utils.cu,
        cachemir_attention.cu) so a planned refresh of it can route sparse."""
        if hasattr(self.inf, "tag_reduce"):
            self.inf.tag_reduce(a, int(stride))
        return a

    def load_pts(self, prefix):
        return self.inf.load_slot_pts("impl." + prefix) if self.slot_pts else 0

    def evict_pts(self, prefix):
        return self.inf.evict_slot_pts("impl." + prefix) if self.slot_pts else 0

    def drop_pts(self, prefix):
        return self.inf.drop_slot_pts("impl." + prefix) if self.slot_pts else 0

    # levels / bootstrapping
    def level(self, a): return int(a.level)
    def noise_deg(self, a): return int(a.noise_deg)
    def lvl(self, a):
        """Level including a pending rescale (a deg-2 ct hides `unit` primes)."""
        return self.level(a) + (self.unit if self.noise_deg(a) == 2 else 0)
    def level_limit(self): return int(self.fhe.level_limit())
    def bts_out(self): return int(self.fhe.bootstrap_output_level())
    def headroom(self, k): return self.level_limit() - k * self.unit

    def bootstrap(self, a):
        self.fhe.bootstrap(a); self.n_bootstraps += 1
        return a

    def bootstrap_real(self, a):
        """A deliberate dense refresh of a payload known to be real: the real-payload route (one EvalMod chain) when
        the session was built with FIDESLIB_BTS_REAL=1, else the ordinary refresh."""
        bts = getattr(self.core, "bts", None)
        if bts is None or os.environ.get("FIDESLIB_BTS_REAL", "0") in ("", "0"):
            return self.bootstrap(a)
        bts.set_real_payload(self.fhe, True)
        try:
            return self.bootstrap(a)
        finally:
            bts.set_real_payload(self.fhe, False)

    def fold_slots_for(self, s_wanted):
        return int(self.fhe.fold_slots_for(int(s_wanted))) if self.folds else self.inf.slots

    def fold_bootstrap(self, a, s, n_live, prescale=1.0):
        """In place: the sparse bootstrap that finishes a ladder stopped at stride `s`: the
        class sums over the slots/s copies, divided by n_live (fideslib_wrapper.h
        fold_bootstrap). Counts as one deliberate refresh."""
        self.fhe.fold_bootstrap(a, int(s), int(n_live), float(prescale))
        self.n_bootstraps += 1
        return a

    def suppress_auto_bts(self):
        """`with ops.suppress_auto_bts():` no reactive refresh inside (a fold follows)."""
        if hasattr(self.fhe, "suppress_auto_bts"):
            return self.fhe.suppress_auto_bts()
        return contextlib.nullcontext()

    def bootstrap_hint(self, a, thr, acct=False):
        before = self.level(a)
        self.fhe.bootstrap_hint(a, int(thr), bool(acct))
        if self.level(a) < before:
            self.n_bootstraps += 1
        return a

    def realize(self, a):
        self.core.realize_pending_rescale(self.inf, a)
        return a

    # data plane
    def encode_token(self, x): return self.core.encode_token_input(self.inf, np.asarray(x, dtype=np.float64))
    def pack_tokens(self, rows, level): return self.core.pack_tokens(self.inf, np.asarray(rows, dtype=np.float64), int(level))
    def decrypt_slots(self, a): return np.asarray(self.core.decrypt_slots(self.inf, a), dtype=np.float64)
    def decode_linear_output(self, a, d_in, d_out):
        return np.asarray(self.core.decode_linear_output(self.inf, a, int(d_in), int(d_out)), dtype=np.float64)
    def decode_token(self, a): return np.asarray(self.core.decode_token_output(self.inf, a), dtype=np.float64)


class NumpyOps:
    """Plain float64 arrays: the plaintext mirror. Levels are meaningless (0), bootstraps
    are the identity, conjugation is the identity (real payload)."""
    unit = 2

    def add(self, a, b): return a + b
    def sub(self, a, b): return a - b
    def mult(self, a, b): return a * b
    def square(self, a): return a * a
    def negate(self, a): return -a
    def rotate(self, a, k): return np.roll(a, -int(k))
    def rotate_and_sum(self, a, start, stop):
        gap, sign = abs(int(start)), (1 if start > 0 else -1)
        x = np.array(a)
        while gap < stop:
            x = x + np.roll(x, -sign * gap)
            gap *= 2
        return x
    def conjugate(self, a): return np.conj(a)
    def inplace_add(self, a, b): a += b
    def copy(self, a): return np.array(a)
    def mult_pt(self, a, v, key=None, kind="weight", tagged=True): return a * np.asarray(v)
    def add_pt(self, a, v, key=None, kind="weight", tagged=True): return a + np.asarray(v)
    def mult_const(self, a, re, im): return a * complex(re, im)
    def mult_i(self, a): return a * 1j
    def device_sync(self): pass
    complex_payload = True
    def level(self, a): return 0
    def noise_deg(self, a): return 1
    def lvl(self, a): return 0
    def level_limit(self): return 1 << 30
    def bts_out(self): return 0
    def headroom(self, k): return 1 << 30
    def bootstrap(self, a): return a
    def bootstrap_real(self, a): return a
    def bootstrap_hint(self, a, thr, acct=False): return a
    folds = fused_sm_den = fused_ln_var = sm_den_recip = gelu_fold = sm_periodic = sm_fold = kv_lanes = False
    def fold_slots_for(self, s_wanted): return int(s_wanted)
    def fold_bootstrap(self, a, s, n_live, prescale=1.0):
        out = self.rotate_and_sum(a, s, a.shape[0]) / n_live   # the fold: the remaining ladder, /n_live
        a[...] = out
        return a
    def suppress_auto_bts(self): return contextlib.nullcontext()
    def realize(self, a): return a


