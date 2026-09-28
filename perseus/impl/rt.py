"""The per-session bundle every kernel takes (``Rt``): the ops protocol, the geometry, a step
scope, and the MASK discipline of the C++ decode arm.

Masks are the plaintexts that depend on nothing but the geometry and, for some, the step
(token position): selection masks, the softmax clip/active masks, the K/V lane masks, the
LayerNorm centering scale. The C++ keeps them in the runtime's encode cache (``inf.enc_cache``,
keyed by tag AND level) and, per token: adopts the next step's masks that a worker encoded
under the previous token's argmax tail, evicts the previous step's, and primes synchronously
whatever is still missing (gpt2_decode.cu, gpt2_residency.cu). ``Rt`` does the
same through ``Inference.{mult_cached, add_cached, prime_pt, stage_pts, adopt_pts, erase_pts}``:

    rt.mult_mask(ct, key, build)   ct * mask (cached by tag + level; encoded on first use)
    rt.declare_step(s, items)      the per-step masks of step s: [(key, build)]
    rt.stage_step(s)               encode them on the worker at the levels their sites used
    rt.begin_step(s)               adopt step s's staged masks, evict step s-1's

A mask's level is not known before its first use, so token 0 pays synchronous encodes and
records, per SITE (the first element of the key), the levels it was used at; every later
step's masks are staged at those levels. A staged mask that is never used costs one encode;
a used mask that was not staged is primed on the spot and counted (``rt.mask_misses``).
"""
from __future__ import annotations

import numpy as np

from .layout import Dims


class Rt:
    def __init__(self, ops, dims: Dims):
        self.ops = ops
        self.dims = dims
        self._masks = {}          # key -> numpy vector (memoised builds)
        self._kinfo = {}          # key -> (tag, the site's level set): derived once per key
        self.site_levels = {}     # site -> set of levels the site's masks were used at
        self._step_items = {}     # step -> [(key, build)]
        self._staged = {}         # step -> (handle, [tags])
        self.mask_misses = 0      # uses of a per-step mask that was not staged
        self.mask_stats = {}      # last begin_step's numbers

    # ── mask vectors ──
    def mask(self, key, build):
        """Memoised numpy mask; `mask_key(key)` is the tag the runtime caches its plaintext
        under (the C++ encode_at_cached tag)."""
        m = self._masks.get(key)
        if m is None:
            m = np.asarray(build())
            m = np.ascontiguousarray(m, dtype=np.complex128 if np.iscomplexobj(m) else np.float64)
            self._masks[key] = m
        return m

    @staticmethod
    def mask_key(key):
        return "mask." + (key if isinstance(key, str) else ".".join(str(k) for k in key))

    @staticmethod
    def site(key):
        return key if isinstance(key, str) else str(key[0])

    def _key_info(self, key):
        """(tag, the site's level set) for `key`, derived once. `mask_key`/`site` build a
        string per call and `_use` runs on every masked leaf op, so the derivation is cached
        next to the mask vector rather than repeated. The level set is held by reference:
        `site_levels` only ever grows sets in place."""
        info = self._kinfo.get(key)
        if info is None:
            info = (self.mask_key(key), self.site_levels.setdefault(self.site(key), set()))
            self._kinfo[key] = info
        return info

    def _use(self, key, ct):
        tag, levels = self._key_info(key)
        lv = self.ops.lvl(ct)
        if key in self._declared_now and lv not in levels:
            self.mask_misses += 1     # a per-step mask used at a level nobody staged
        levels.add(lv)
        return tag

    def mult_mask(self, ct, key, build):
        return self.ops.mult_pt(ct, self.mask(key, build), key=self._use(key, ct), kind="mask")

    def add_mask(self, ct, key, build):
        return self.ops.add_pt(ct, self.mask(key, build), key=self._use(key, ct), kind="mask")

    def mult_masks(self, ct, keys, builds):
        """ct * each mask, as one fused lane batch when the runtime allows."""
        vecs = [self.mask(k, b) for k, b in zip(keys, builds)]
        tags = [self._use(k, ct) for k in keys]
        return self.ops.mult_pt_many(ct, vecs, tags, kind="mask")

    # ── per-step discipline ──
    _declared_now = frozenset()

    def declare_step(self, step, items):
        """Register the per-step masks of `step`: items = [(key, build)]. Idempotent."""
        self._step_items[step] = list(items)

    def stage_step(self, step):
        """Encode step `step`'s declared masks on the runtime's worker, at every level their
        sites were used at so far. Returns the number of plaintexts submitted."""
        items = self._step_items.get(step, [])
        work, tags = [], []
        for key, build in items:
            tag, levels = self._key_info(key)
            tags.append(tag)
            if not levels:
                continue
            vec = self.mask(key, build)
            for lv in sorted(levels):
                work.append((tag, lv, vec))
        handle = self.ops.stage_pts(work) if work else None
        self._staged[step] = (handle, tags)
        return len(work)

    def begin_step(self, step):
        """Adopt what was staged for `step`, evict step-1's masks. Call before the step's ops."""
        cur = frozenset(k for k, _ in self._step_items.get(step, []))
        # evict what step-1 declared and this step does not use (a mask shared by both, e.g.
        # a full q.K^T group's mask, stays: the C++ evicts-then-reprimes it instead)
        prev = [k for k, _ in self._step_items.pop(step - 1, []) if k not in cur]
        evicted = self.ops.erase_pts([self.mask_key(k) for k in prev]) if prev else 0
        for k in prev:
            self._masks.pop(k, None)          # the numpy copy too
        handle, _tags = self._staged.pop(step, (None, None))
        adopted = self.ops.adopt_pts(handle) if handle is not None else 0
        self._declared_now = cur
        self.mask_stats = {"adopted": adopted, "evicted": evicted, "misses": self.mask_misses}
        self.mask_misses = 0
        return self.mask_stats

    def drain(self):
        """Join every staged job still in flight (before the session closes: a job that
        outlives the Inference it encodes for is a use-after-free at exit)."""
        n = 0
        for step in sorted(self._staged):
            handle, _tags = self._staged.pop(step)
            n += self.ops.adopt_pts(handle) if handle is not None else 0
        return n

    def end_step(self, step, next_items=None):
        """Declare and stage step+1 (the tail of a token: overlaps the argmax like the C++)."""
        if next_items is not None:
            self.declare_step(step + 1, next_items)
        return self.stage_step(step + 1)

    def step(self, label):
        """`with rt.step("label"):` scopes ops under a step label in the captured graph and,
        for the timing layer, attributes the time of the ops inside to `label`."""
        return _StepScope(self, label)

    @classmethod
    def from_inf(cls, inf, core, unit=2, bts_precision=12):
        from .ops import FheOps
        return cls(FheOps(inf, core, unit, bts_precision), Dims.from_inf(inf))


class _StepScope:
    def __init__(self, rt, label):
        self.rt, self.label = rt, label
        self.inner = rt.ops.inf.step(label)

    def __enter__(self):
        self.inner.__enter__()
        ops = self.rt.ops
        self.prev = getattr(ops, "step_label", None)
        ops.step_label = self.label
        return self

    def __exit__(self, *a):
        self.rt.ops.step_label = self.prev
        return self.inner.__exit__(*a)
