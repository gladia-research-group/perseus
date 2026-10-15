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

import os
import re

import numpy as np

from .layout import Dims


# IMPL_MASK_SUM=1: masked inner sums (the SinC conversions) as one fused call (rt.mult_mask_sum)
_MASK_SUM = os.environ.get("IMPL_MASK_SUM", "0") not in ("", "0")

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
        self._used = set()        # keys used since the last evict_unused
        # IMPL_SCOPED_NAMES=1: every step names its ciphertexts <step path>_<n> on its own counter
        # (Context.graph_scope), so a plan made at one token stays bound at tokens where an
        # earlier step issues more ops (the q.K^T over a cache with more groups)
        # (the op ORDER changes the scoping implies, e.g. attention's late pair pack, follow the
        # request even on a session without the binding, so the fake mirrors the device's stream)
        self.scoped_names_requested = os.environ.get("IMPL_SCOPED_NAMES", "0") not in ("", "0")
        self.scoped_names = (self.scoped_names_requested
                             and hasattr(getattr(ops, "fhe", None), "graph_scope"))
        if self.scoped_names and hasattr(ops.fhe, "graph_scope_strict"):
            ops.fhe.graph_scope_strict = True       # naming only: planned mode stays strict
        self._scope_path = []
        self._scope_visits = {}

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

    sub_tags = frozenset()      # SubRt namespaces: a key's site is the one under the tag

    def site(self, key):
        if isinstance(key, str):
            return key
        if len(key) > 1 and key[0] in self.sub_tags:
            return str(key[1])
        return str(key[0])

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
        self._used.add(key)
        lv = self.ops.lvl(ct)
        if key in self._declared_now and lv not in levels:
            self.mask_misses += 1     # a per-step mask used at a level nobody staged
        levels.add(lv)
        return tag

    def mult_mask(self, ct, key, build):
        return self.ops.mult_pt(ct, self.mask(key, build), key=self._use(key, ct), kind="mask")

    def add_mask(self, ct, key, build):
        return self.ops.add_pt(ct, self.mask(key, build), key=self._use(key, ct), kind="mask")

    def mult_mask_sum(self, cts, keys, builds):
        """sum_i cts[i] * mask_i: one fused call when the runtime has mult_cached_sum and IMPL_MASK_SUM=1 (the loop's
        op stream replayed), else mult_mask + add per term."""
        if _MASK_SUM and len(cts) > 1 and hasattr(getattr(self.ops, "inf", None), "mult_cached_sum") \
                and getattr(self.ops, "enc_cache", False):
            vecs = [self.ops._vec(self.mask(k, b)) for k, b in zip(keys, builds)]
            tags = [self._use(k, c) for k, c in zip(keys, cts)]
            out = self.ops.inf.mult_cached_sum(list(cts), tags, vecs, True)
            if out is not None:
                return out
        acc = None
        for c, k, b in zip(cts, keys, builds):
            t = self.mult_mask(c, k, b)
            acc = t if acc is None else self.ops.add(acc, t)
        return acc

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
        self._scope_visits.clear()
        self._declared_now = cur
        self.mask_stats = {"adopted": adopted, "evicted": evicted, "misses": self.mask_misses}
        self.mask_misses = 0
        return self.mask_stats

    def evict_masks(self, prefixes):
        """Free the encoded plaintexts of masks whose site starts with one of `prefixes`; returns the count."""
        keys = [k for k in self._kinfo if self.site(k).startswith(tuple(prefixes))]
        if not keys:
            return 0
        return self.ops.erase_pts([self._kinfo[k][0] for k in keys])

    def evict_unused(self):
        """Free the encoded plaintexts of masks not used since the last call; returns the count."""
        keys = [k for k in self._kinfo if k not in self._used]
        self._used = set()
        return self.ops.erase_pts([self._kinfo[k][0] for k in keys]) if keys else 0

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

    def step(self, label, shape=None):
        """`with rt.step("label"):` scopes ops under a step label in the captured graph and,
        for the timing layer, attributes the time of the ops inside to `label`. `shape`: what
        the step's op count depends on (e.g. the cache's group count), part of its name scope."""
        return _StepScope(self, label, shape)

    @classmethod
    def from_inf(cls, inf, core, unit=2, bts_precision=12):
        from .ops import FheOps
        return cls(FheOps(inf, core, unit, bts_precision), Dims.from_inf(inf))


# PF_NVTX_STEPS=1 (diagnostic): an NVTX range per step (torch's NVTX binding), for nsys per-step GPU time
_NVTX = None
if os.environ.get("PF_NVTX_STEPS", "0") not in ("", "0"):
    from torch.cuda import nvtx as _NVTX


class _StepScope:
    def __init__(self, rt, label, shape=None):
        self.rt, self.label, self.shape = rt, label, shape
        self.inner = rt.ops.inf.step(label)
        self.saved = None

    def __enter__(self):
        self.inner.__enter__()
        if _NVTX is not None:
            _NVTX.range_push(str(self.label))
        rt, ops = self.rt, self.rt.ops
        self.prev = getattr(ops, "step_label", None)
        ops.step_label = self.label
        if rt.scoped_names:
            fhe = ops.fhe
            self.saved = (fhe.graph_scope, fhe.graph_scope_counter)
            rt._scope_path.append(str(self.label) + ("" if self.shape is None else f"~{self.shape}"))
            path = "_".join(re.sub(r"[^A-Za-z0-9~]+", "", p) for p in rt._scope_path)
            n = rt._scope_visits.get(path, 0)
            rt._scope_visits[path] = n + 1
            fhe.graph_scope = f"sc_{path}" + (f"~{n}" if n else "")
            fhe.graph_scope_counter = 0
        return self

    def __exit__(self, *a):
        rt = self.rt
        rt.ops.step_label = self.prev
        if self.saved is not None:
            rt._scope_path.pop()
            rt.ops.fhe.graph_scope, rt.ops.fhe.graph_scope_counter = self.saved
        if _NVTX is not None:
            _NVTX.range_pop()
        return self.inner.__exit__(*a)
