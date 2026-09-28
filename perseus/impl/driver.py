"""``ImplModel``: the model-independent driver of a primitives model — one session, the
three execution modes, the residency ring and the per-token bookkeeping. A model subclasses
it and supplies its stages (``token_stages``), its per-step masks (``step_masks``) and its
weights; see examples/gpt2_from_primitives/model.py.

Modes (the same loop the C++ / EncGPT2 runs):
    eager     reactive bootstraps plus the C++ hint sites (default)
    capture   ``set_capture(graph_dir)``: block_<b>/graph.json per stage for the capture token
              (FHE_GRAPH_CAPTURE_TOKEN, default 0) plus capture_env.json (the plan contract)
    planned   ``load_plans(plan_dir)``: block_<b>_placement.json installed live per stage,
              strict (the op sequence must match the capture)

Stages. ``token_stages(x, pos)`` returns ``[(label, prefix, fn)]``: ``fn(x) -> x`` runs the
stage's ops; ``prefix`` names the block's cached weight plaintexts (``set_slot_pt`` names
``impl.<prefix>*``) or is None for a stage without streamed weights. With the runtime's
residency ring available, a stage with a prefix becomes a ``Stage(state=EncodedBlock)`` whose
plaintexts are extracted on the worker and uploaded on the side stream while the previous
stage computes, installed for the compute and evicted after it — the C++ block loader's path.
Without the ring (the fake, or the first token before the names exist) the stages run as a
plain loop with load/evict around each.

Masks. ``step_masks(pos)`` returns the per-step mask items ``[(key, build)]`` of position
``pos``; ``token_begin`` / ``token_tail`` drive ``Rt``'s adopt / stage / evict cycle around it.
"""
from __future__ import annotations

import json
import os
import resource
import time

from .layout import Dims
from .ops import FheOps
from .rt import Rt


class ImplModel:
    def __init__(self, inf, *, core=None, unit=None, profile=False, dims=None, kv_offload=None):
        if core is None:
            from perseus import _core as core
        self.inf, self.core = inf, core
        if unit is None:
            unit = int(os.environ.get("COMPOSITE_DEGREE", "2"))
        prec = int(os.environ.get("BTS_PRECISION", "12"))
        if profile:
            from .profile import TimedOps
            ops = TimedOps(inf, core, unit, prec)
        else:
            ops = FheOps(inf, core, unit, prec)
        self.rt = Rt(ops, dims or Dims.from_inf(inf))
        self.plans = None
        self.graph_dir = None
        # weight-upload overlap through the runtime's residency ring (needs the real core)
        self.overlap = (hasattr(core, "run_stages") and hasattr(core, "Stage")
                        and hasattr(getattr(core, "EncodedBlock", None), "adopt_slot_pts"))
        self._states = {}
        # Pin the ring's stage arenas before the plaintext copies fill the GPU's NUMA node
        # (the C++ decode arm does the same in reset_kv_cache; a late first-touch lands them
        # on a remote node and every token then stalls on remote pinned memory).
        if self.overlap and hasattr(core, "prewarm_stage_arenas"):
            core.prewarm_stage_arenas(inf)
        # K/V residency (the C++ offload_block_kv): a stage's cache ciphertexts are copied to
        # the pinned KV arena on a side stream after the stage and reloaded ahead of its next
        # use, so only two blocks' caches sit on the device at any time
        if kv_offload is None:
            kv_offload = os.environ.get("IMPL_KV_OFFLOAD", "1") not in ("", "0")
        self.kv_offload = bool(kv_offload) and hasattr(inf, "kv_store")
        self._parked = {}              # stage -> keys whose ciphertexts live in the arena
        self._ring_order = []          # prefixes of the ring's stages, in order
        self._ring_prefetch = None     # the circular ring's in-flight extraction
        self.stage_marks = []          # (pos, stage, t_enter, t_leave): ring overhead = the gaps
        self._names_ready = None       # the stage-prefix tuple whose slot names are registered
        self.overlap_mode = getattr(getattr(core, "InferenceMode", None), "Prefetch", None)
        self._tok = None

    # ── hooks a model overrides ──
    def token_stages(self, x, pos):
        """[(label, prefix | None, fn)] for one token; see the module docstring."""
        raise NotImplementedError

    def step_masks(self, pos):
        """The per-step mask items [(key, build)] of position `pos` (default: none)."""
        return []

    def stage_caches(self, i):
        """(ciphertexts, keys) a stage keeps across tokens (its K/V cache), or None. With
        `kv_offload` they are parked in the pinned KV arena between the stage's uses."""
        return None

    def start(self):
        """Reset per-sequence state (K/V caches, seeds)."""

    # ── modes ──
    def set_capture(self, graph_dir):
        """CAPTURE: write block_<b>/graph.json under graph_dir for the token
        FHE_GRAPH_CAPTURE_TOKEN (default 0), plus capture_env.json (the plan contract)."""
        from perseus.plan import contract as _contract
        os.makedirs(graph_dir, exist_ok=True)
        os.environ["FHE_GRAPH_DIR"] = str(graph_dir)
        with open(os.path.join(graph_dir, "capture_env.json"), "w", encoding="utf-8") as f:
            json.dump(_contract.capture_contract(self.inf), f, indent=2)
        self.graph_dir = str(graph_dir)

    def load_plans(self, plan_dir, blocks, validate=True):
        """PLANNED: block_<b>_placement.json for b in `blocks`, contract-checked against this
        session. A stage without a plan file runs eager under the planned model."""
        from perseus.plan import contract as _contract
        self.plans = {}
        for b in blocks:
            p = os.path.join(plan_dir, f"block_{b}_placement.json")
            if not os.path.exists(p):
                continue
            if validate:
                _contract.validate_contract(_contract.read_stamp(p), self.inf, source=p)
            self.plans[b] = self.core.parse_bootstrap_plan_file(p)
        return self

    def enter_stage(self, b, pos):
        """Per-stage graph / plan setup (the C++ decode body: reset the naming state, scope,
        install the plan, open the capture)."""
        inf, core = self.inf, self.core
        if self.plans is not None or os.environ.get("FHE_GRAPH_DIR"):
            core.reset_graph_runtime(inf)
            if self.plans is not None and b not in self.plans:
                inf.clear_bootstrap_plan()      # this stage runs eager under a planned model
        inf.block_prefix = core.block_scope(b)
        inf.capture_b = b
        inf.capture_t = pos
        plan = (self.plans or {}).get(b)
        if plan is not None:
            core.install_plan_live(inf, plan)
        return core.begin_subgraph_capture(inf, b)

    def exit_stage(self, b, cap):
        if cap:
            self.core.end_subgraph_capture(self.inf, b)
        if self.plans is not None:
            self.inf.clear_bootstrap_plan()

    # ── the ring ──
    def run_stages(self, x, pos, stages):
        """Run [(label, prefix, fn)] through the residency ring (or a plain loop)."""
        ops = self.rt.ops
        marks = self.stage_marks

        def timed(i, fn):
            def compute(x):
                t_in = time.time()
                x = fn(x)
                marks.append((pos, i, t_in, time.time()))
                return x
            return compute

        if self.kv_offload:
            stages = self._with_kv_residency(stages)
        keyed = [(lab, pre, timed(i, fn)) for i, (lab, pre, fn) in enumerate(stages)]
        prefixed = [pre for _, pre, _ in keyed if pre is not None]
        self._ring_order = prefixed
        # `_stage_keys` scans every named slot in the model per prefix; the answer only ever
        # flips false->true (slots are never dropped), so latch it against the prefix tuple.
        ring_key = tuple(prefixed)
        if self._names_ready != ring_key and all(self._stage_keys(pre) for pre in prefixed):
            self._names_ready = ring_key
        if self.overlap and prefixed and self._names_ready == ring_key:
            Stage = self.core.Stage
            for pre in prefixed:
                if pre not in self._states:
                    st = self.core.EncodedBlock()
                    st.adopt_slot_pts(self.inf, "impl." + pre)
                    self._states[pre] = st
            return self.core.run_stages(
                self.inf, x,
                [Stage(compute=fn, state=self._states[pre], label=lab) if pre is not None
                 else Stage(compute=fn, label=lab) for lab, pre, fn in keyed],
                self.overlap_mode)
        for lab, pre, fn in keyed:         # plain loop: first token (names not yet
            if pre is not None:            # registered), the fake, or overlap off
                ops.load_pts(pre)
            x = fn(x)
            if pre is not None:
                ops.evict_pts(pre)         # block_release: host copies stay
        return x

    def _kv_prefetch(self, i):
        """Enqueue the reload of stage i's parked cache ciphertexts on the KV stream."""
        keys = self._parked.get(i)
        if not keys:
            return
        cts, ks = self.stage_caches(i)
        sel = [(c, k) for c, k in zip(cts, ks) if k in keys]
        if sel:
            self.inf.kv_load([c for c, _ in sel], [k for _, k in sel])

    def _with_kv_residency(self, stages):
        """Wrap the stage computes with the cache choreography of the C++ decode arm: before
        stage i, wait for its reload and prefetch stage i+1's; after it, enqueue its offload
        and finalize (sync + evict) the previous stage's."""
        inf = self.inf
        n = len(stages)

        load = self._kv_prefetch

        def wrap(i, fn):
            def compute(x):
                if i == 0:
                    load(0)                             # a no-op when the token tail did it
                if self._parked.get(i):
                    inf.kv_sync()                       # stage i's reload has landed
                    self._parked.pop(i, None)
                if i + 1 < n:
                    load(i + 1)                         # overlaps this stage's compute
                x = fn(x)
                caches = self.stage_caches(i)
                if caches and caches[0]:
                    cts, ks = caches
                    # the D2H runs on its own stream and does not wait for the ops that wrote
                    # the caches: fence first, like the C++ block_sync before its offload
                    self.rt.ops.device_sync()
                    inf.kv_store(cts, ks)               # async D2H; evicted after the next stage
                    self._pending = (i, cts, ks)
                prev = getattr(self, "_pending_prev", None)
                if prev is not None:                    # the previous stage's store has landed
                    inf.kv_sync()
                    inf.kv_evict(prev[1])
                    self._parked[prev[0]] = set(prev[2])
                self._pending_prev = getattr(self, "_pending", None)
                self._pending = None
                if i + 1 == n:                          # last stage: finalize its own store too
                    fin = self._pending_prev
                    if fin is not None:
                        inf.kv_sync(); inf.kv_evict(fin[1]); self._parked[fin[0]] = set(fin[2])
                    self._pending_prev = None
                return x
            return compute

        return [(lab, pre, wrap(i, fn)) for i, (lab, pre, fn) in enumerate(stages)]

    def _stage_keys(self, prefix):
        """The inf.w names of a stage's cached plaintexts (known after their first use)."""
        return sorted(k for k in getattr(self.rt.ops, "_slots", ()) if k.startswith("impl." + prefix))

    # ── per-token bookkeeping ──
    def token_begin(self, pos):
        """Adopt this step's staged masks, evict the previous step's, start the counters."""
        if pos not in self.rt._step_items:
            self.rt.declare_step(pos, self.step_masks(pos))
        t_wait = time.time()
        rp = getattr(self, "_ring_prefetch", None)
        if rp is not None:
            rp.wait()                            # before anything touches the block states
            self._ring_prefetch = None
        self.rt.begin_step(pos)
        self._mask_wait = time.time() - t_wait
        self._tok = {"t0": time.time(), "cpu": time.process_time(), "thr": time.thread_time(),
                     "ru": resource.getrusage(resource.RUSAGE_SELF), "pos": pos}

    def token_tail(self, pos):
        """Stage the next step's masks on the worker and start the next token's ring warm:
        the first two stages' weights are extracted on the residency worker and their caches
        reloaded on the KV stream, under whatever GPU tail follows (the C++ circular ring)."""
        n = self.rt.end_step(pos, self.step_masks(pos + 1))
        self._ring_prefetch = None
        if self.overlap and hasattr(self.core, "prefetch_states") and self._ring_order:
            first = [self._states[pre] for pre in self._ring_order[:2] if pre in self._states]
            if first:
                self._ring_prefetch = self.core.prefetch_states(self.inf, first)
        if self.kv_offload:
            # the reloads allocate device ciphertexts on the KV stream, which is not ordered
            # with the compute still in flight: fence first
            self.rt.ops.device_sync()
            for i in (0, 1):
                self._kv_prefetch(i)
        return n

    def close(self):
        """Join the staging worker's pending jobs; call before the session closes."""
        return self.rt.drain()

    def token_end(self, pos):
        """Timing / resource / ring numbers of the token as a dict."""
        tk = self._tok or {}
        ru0, ru1 = tk.get("ru"), resource.getrusage(resource.RUSAGE_SELF)
        rec = {"pos": pos, "decode_s": time.time() - tk.get("t0", time.time())}
        if ru0 is not None:
            rec.update(cpu_s=round(time.process_time() - tk["cpu"], 1),
                       thread_s=round(time.thread_time() - tk["thr"], 1),
                       ctx_vol=ru1.ru_nvcsw - ru0.ru_nvcsw, ctx_invol=ru1.ru_nivcsw - ru0.ru_nivcsw,
                       minflt=ru1.ru_minflt - ru0.ru_minflt, majflt=ru1.ru_majflt - ru0.ru_majflt)
        if hasattr(self.core, "device_free_gb"):
            rec["free_gb"] = round(float(self.core.device_free_gb()), 1)
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rec["rss_gb"] = round(int(line.split()[1]) / 1048576, 1)
        except OSError:
            pass
        ms = [m for m in self.stage_marks if m[0] == pos]
        if ms:
            inside = sum(m[3] - m[2] for m in ms)
            rec["ring_gap_s"] = round(rec["decode_s"] - inside, 2)
            rec["ring_gaps"] = [round(ms[i + 1][2] - ms[i][3], 2) for i in range(len(ms) - 1)]
        rec["masks"] = dict(self.rt.mask_stats)
        rec["mask_wait_s"] = round(getattr(self, "_mask_wait", 0.0), 2)
        st = self.rt.ops.enc_cache_stats()
        if st:
            rec["enc_cache"] = st
        return rec
