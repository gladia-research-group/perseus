"""``ImplModel``: the model-independent driver of a primitives model — one session, the
three execution modes, the residency ring and the per-token bookkeeping. A model subclasses
it and supplies its stages (``token_stages``), its per-step masks (``step_masks``) and its
weights; see perseus/examples/gpt2/model.py.

Modes: ``eager`` (reactive bootstraps + hint sites), ``capture`` (``set_capture``: graph.json per stage of the
capture token), ``planned`` (``load_plans``: placements installed live per stage, strict; per-state variants).
Stages with a weight prefix run as residency-ring ``Stage``s (uploaded on the side stream while the previous stage
computes); masks follow ``Rt``'s adopt / stage / evict cycle. Details: docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import json
import os
import resource
import threading
import time

import numpy as np

from .layout import Dims
from .ops import FheOps
from .rt import Rt


class ImplModel:
    def __init__(self, inf, *, core=None, unit=None, profile=False, dims=None, kv_offload=None):
        if core is None:
            from perseus import _pc as core
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
        self.plan_variants = {}
        self._route_class = None       # ensure_routes
        self._base_routes = None
        self.plan_variant = None
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
        # K/V residency (the C++ offload_block_kv), IMPL_KV_OFFLOAD=1: a stage's cache ciphertexts are copied
        # to the pinned KV arena on a side stream after the stage and reloaded ahead of its next use, so only two
        # blocks' caches sit on the device at any time. Off by default: the caches stay resident (2.2 GB on the
        # n32 decode) and the per-stage device fences the offload needs go away (-0.08 s/token with the other
        # barrier removals, 2026-10-11)
        if kv_offload is None:
            kv_offload = os.environ.get("IMPL_KV_OFFLOAD", "0") not in ("", "0")
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

    def replay_kv_pushes(self, payloads, level=36):
        """Fill the caches with positions 0..n-1 computed elsewhere (a numpy fake run, a public prefix): every
        position's pushes are replayed on this session from its payloads (attention.kv_push's `payload_log`),
        each encrypted as a public value (a fresh encryption of zero plus it) at `level`. The pushes themselves
        (refresh, masks, bucketing) run here, so the caches carry this session's levels, degrees and pack tags.
        `payloads[b]` is block b's log. Run after start(); decode then begins at position n."""
        import numpy as np
        from .attention import kv_push
        ops = self.rt.ops
        zero = np.zeros(self.rt.dims.dim)

        def enc(v):
            ct = ops.add_pt(self.core.encode_token_input(self.inf, zero), np.asarray(v))
            return ops.level_drop(ct, level)
        n = len(payloads[0]) if payloads else 0
        for pos in range(n):
            self.rt.declare_step(pos, self.step_masks(pos))
            self.rt.begin_step(pos)
            for b, (kv, log) in enumerate(zip(self.kv, payloads)):
                p = log[pos]
                with self.rt.step(f"blk{b}"), self.rt.step("kv_push"):
                    kv_push(self.rt, kv, tuple(enc(c) for c in p) if isinstance(p, tuple) else enc(p))
        if self.kv_offload:
            # parked as a decode leaves them: in the pinned KV arena, reloaded by the stage loop
            ops.device_sync()
            for i in range(len(self.kv)):
                caches = self.stage_caches(i)
                if caches and caches[0]:
                    cts, ks = caches
                    self.inf.kv_store(cts, ks)
                    self.inf.kv_sync()
                    self.inf.kv_evict(cts)
                    self._parked[i] = set(ks)
        return n

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

    def load_plans(self, plan_dir, blocks, validate=True, variants=()):
        """PLANNED: block_<b>_placement.json for b in `blocks`, contract-checked against this
        session, and block_<b>_<v>_placement.json for each of `variants` where present. A
        stage without a plan file runs eager under the planned model."""
        from perseus._paths import resolve
        from perseus.plan import contract as _contract

        plan_dir = resolve(plan_dir)

        def load(p):
            if validate:
                _contract.validate_contract(_contract.read_stamp(p), self.inf, source=p)
            return self.core.parse_bootstrap_plan_file(p)

        self.plans, self.plan_variants = {}, {v: {} for v in variants}
        for b in blocks:
            p = os.path.join(plan_dir, f"block_{b}_placement.json")
            if os.path.exists(p):
                self.plans[b] = load(p)
            for v in variants:
                p = os.path.join(plan_dir, f"block_{b}_{v}_placement.json")
                if os.path.exists(p):
                    self.plan_variants[v][b] = load(p)
        return self

    def routes_for(self, pos):
        """(class, {slots}): the sparse routes position `pos` needs beyond the session's own (hook; None: none)."""
        return None

    def ensure_routes(self, pos):
        """Make the resident sparse routes those position `pos` (+ SM_ROUTE_AHEAD) needs: unneeded routes are dropped
        first, then missing ones built, so the device never holds both sets. Routes are not part of the plan: the
        runtime refreshes a planned site on the route its value's period needs (planned_route_for). Building needs
        the secret key (in a deployment: the client's keys, asked for ahead of the boundary)."""
        if not hasattr(self.inf, "add_sparse_route"):
            return
        if self._base_routes is None:
            self._base_routes = set(self.inf.sparse_routes)
        ahead = int(os.environ.get("SM_ROUTE_AHEAD", "0") or 0)
        want = self.routes_for(pos + ahead)
        if want is None:
            return
        cls, slots = want
        if cls == self._route_class:
            return
        target = self._base_routes | set(slots)
        current = set(self.inf.sparse_routes)
        t0 = time.time()
        dropped = sorted(current - target)
        for s in dropped:
            self.inf.drop_sparse_route(int(s))
        built = sorted(target - current)
        for s in built:
            self.inf.add_sparse_route(int(s))
        self._route_class = cls
        print(f"[routes] position {pos}: class {cls}, routes {sorted(self.inf.sparse_routes)} "
              f"(dropped {dropped}, built {built}) in {time.time() - t0:.1f}s", flush=True)

    def _plan_for(self, b):
        if self.plans is None:
            return None
        plan = self.plan_variants.get(self.plan_variant, {}).get(b)
        return plan if plan is not None else self.plans.get(b)

    def enter_stage(self, b, pos):
        """Per-stage graph / plan setup (the C++ decode body: reset the naming state, scope,
        install the plan, open the capture)."""
        inf, core = self.inf, self.core
        if self.plans is not None or os.environ.get("FHE_GRAPH_DIR"):
            core.reset_graph_runtime(inf)
            if self.plans is not None and self._plan_for(b) is None:
                inf.clear_bootstrap_plan()      # this stage runs eager under a planned model
        inf.block_prefix = core.block_scope(b)
        inf.capture_b = b
        inf.capture_t = pos
        plan = self._plan_for(b)
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
        """Adopt this step's staged masks, evict the previous step's, start the counters, build its routes."""
        self.ensure_routes(pos)
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
        if hasattr(self.core, "gpu_pool_stats"):
            st = self.core.gpu_pool_stats()
            rc = self.inf.recycled_polys() if hasattr(self.inf, "recycled_polys") else (0, 0.0)
            rec["pool_gb"] = (round(st["reserved_gb"] - st["idle_gb"] - rc[1], 1), round(st["reserved_gb"], 1),
                              round(st["trimmable_gb"], 1), round(rc[1], 1))
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


# ── planned mode on the new runtime: per-stage Program.run ─────────────────────────────────────────────────────────
#
# A compiled block program fixes the op stream; what changes with the position is public and data-independent: a
# few scalars (the LayerNorm centering constant, the softmax's kc-dependent constants), the K/V lane rotations
# (rot(-pos)), the per-step masks, and, at the first tokens, the K/V push into an empty cache (`x + 0` instead of
# `x + cache`, a rotation by 0 written as `x + 0`). `Recorder` runs the model on the CPU recorder (fakecap) one
# token ahead and gives, per block, the op stream, the plaintext values by graph name and which ciphertext objects
# were read as inputs. `PlannedRun` compiles the blocks once from the reference token's stream (a token whose K/V
# push reads a non-empty cache), keeps the K/V states live out, binds them on a planned Session, and per token:
# checks the token's stream against the reference (strict, like the old runtime's plan binding), sets the dynamic
# scalars / rotations, re-encodes the plaintexts whose values changed (in place: the graphs stay valid), and runs
# every block in Rust (Program.run: graphs between the dynamic steps, GIL released).

class BlockRec:
    __slots__ = ("nodes", "pts", "objs", "metas", "x_in", "out", "out_names", "final", "_ins", "kv_pre", "kv_post")

    def __init__(self, nodes, pts, objs, metas):
        self.nodes, self.pts, self.objs = nodes, pts, objs      # v2 nodes; pt_N -> values; anon -> ct object
        self.metas = metas                                      # anon -> (level, deg) when first read
        self.x_in = self.out = None                             # the stage's input / output object(s)
        self.out_names = []                                     # the output objects' graph names at block end
        self._ins = None
        self.final = {}                                         # id(K/V object) -> its graph name at block end
        self.kv_pre = {}                                        # id(K/V object) -> its cache key at block entry
        self.kv_post = {}                                       # cache key -> its graph name at block end

    def inputs(self):
        """Names read and never produced, in first-use order (= the compiled program's input order)."""
        if self._ins is not None:
            return self._ins
        made, out = set(), []
        for n in self.nodes:
            if n["op_type"] == "auto_bootstrap":
                made.add(n["output"])
                continue
            for x in n["inputs"]:
                if not x.startswith(("pt_", "const(", "rot(", "hint_lev(", "lvl_cap(")) and x not in made and x not in out:
                    out.append(x)
            made.add(n["output"])
        self._ins = out
        return out

    def pos(self):
        return {n: i for i, n in enumerate(self.inputs())}


class Recorder:
    """The model on the CPU recorder, one token at a time (structure only: no magnitudes)."""

    def __init__(self, model, n_layers):
        from . import fakecap
        self.model, self.n_layers = model, n_layers
        fhe = model.inf.fhe
        fhe.magnitude_suppressed = True
        self.blocks = {}
        rec = self
        # wrap the model's own core and naming hooks, never an earlier Recorder's: its Core class and hooks close
        # over that Recorder, which would keep every block it recorded alive (BERT: ~22 GB per token count)
        base = getattr(model.core, "_rec_base", type(model.core))
        if not hasattr(fhe, "_rec_orig"):
            fhe._rec_orig = (fhe.pt_var, fhe.var)

        class Core(base):
            _rec_base = base

            def begin_subgraph_capture(self, inf, b):
                f = inf.fhe
                f.gen += 1
                f.nodes, f.ct_counter, f.pt_counter = [], 0, 0
                f._rec_pts, f._rec_objs, f._rec_metas = {}, {}, {}
                return True

            def end_subgraph_capture(self, inf, b):
                f = inf.fhe
                rec.blocks[b] = BlockRec(f.nodes, f._rec_pts, f._rec_objs, f._rec_metas)
                f.nodes = None
        model.core = Core()
        pt_var, var = fhe._rec_orig

        def rec_pt_var(pt):
            new = not (pt.gen == fhe.gen and pt.name)
            name = pt_var(pt)
            if new and fhe.nodes is not None:
                fhe._rec_pts[name] = pt.vec
            return name

        def rec_var(ct):
            new = not (ct.gen == fhe.gen and ct.name)
            name = var(ct)
            if new and fhe.nodes is not None:
                fhe._rec_objs[name] = ct
                fhe._rec_metas[name] = (int(ct.level), int(ct.noise_deg))
            return name
        fhe.pt_var, fhe.var = rec_pt_var, rec_var
        self._fakecap = fakecap
        self.values = True
        self.first = 0                     # the position of token 0 (Llama decode: 1, the BOS prefix holds 0)
        self.restart = True                # token 0 resets the model (start()); False: it continues from its state
                                           # (a decode after a prefill hand-off)

    def token(self, t, row, argmax=False):
        """{block: BlockRec} of token t (position first + t; blocks 0..n_layers-1, the tail, CutMax when `argmax`)."""
        m, self.blocks = self.model, {}
        pos = t + self.first
        self._fakecap.CapCt.novec = not self.values
        if t == 0 and self.restart:
            m.start()
        m.token_begin(pos)
        x = m.encode_input(row)
        caches = [m.stage_caches(i) for i in range(self.n_layers)]
        for i, (_label, pre, fn) in enumerate(m.token_stages(x, pos)):
            if pre is not None:
                m.rt.ops.load_pts(pre)
            kv_pre = {}
            if i < self.n_layers and m.stage_caches(i):
                kv_pre = {id(c): k for c, k in zip(*m.stage_caches(i))}
            x_in, x = x, fn(x)
            if pre is not None:
                m.rt.ops.evict_pts(pre)
            r = self.blocks[i]
            r.x_in, r.out = x_in, x
            r.out_names = [o.name for o in (x if isinstance(x, (list, tuple)) else [x])]
            if i < self.n_layers:
                cts, keys = m.stage_caches(i)
                r.final = {id(c): c.name for c in cts}
                r.kv_pre = kv_pre
                r.kv_post = {k: c.name for c, k in zip(cts, keys)}
        m.token_tail(pos)
        if argmax:
            b = self.n_layers + 1
            z = m.argmax_encrypted(x, t)
            r = self.blocks[b]
            r.x_in, r.out = x, z
            r.out_names = [o.name for o in (z if isinstance(z, (list, tuple)) else [z])]
        del caches
        return self.blocks


def _canon(pos, names):
    """Input names as their position among the block's inputs (anon names differ from token to token)."""
    return [f"in{pos[x]}" if x in pos else ("const" if x.startswith("const(") else "rot" if x.startswith("rot(") else x)
            for x in names]


def diff_block(ref, rec):
    """(scalars {name: v}, rotations {name: k}, zero-cache input positions) that turn the reference program into
    this token's stream; raises on any other difference (the plan does not bind)."""
    if len(ref.nodes) != len(rec.nodes):
        raise RuntimeError(f"op stream of {len(rec.nodes)} nodes, the program's has {len(ref.nodes)}")
    scal, rots, zero = {}, {}, set()
    rin, yin = ref.pos(), rec.pos()
    for x, y in zip(ref.nodes, rec.nodes):
        xi, yi = _canon(rin, x["inputs"]), _canon(yin, y["inputs"])
        same = x["op_type"] == y["op_type"] and xi == yi and x["output"] == y["output"]
        ax, ay = x.get("args", {}), y.get("args", {})
        if same:
            if ax.get("scalar") != ay.get("scalar"):
                scal[x["output"]] = float(ay["scalar"])
            if x["op_type"] == "rotate" and x["inputs"][1] != y["inputs"][1]:
                rots[x["output"]] = int(y["inputs"][1][4:-1])
            continue
        if x["output"] == y["output"] and y["op_type"] == "add" and ay.get("scalar") == 0.0:
            if x["op_type"] == "rotate" and xi[0] == yi[0]:            # a rotation by 0 written as x + 0
                rots[x["output"]] = 0
                continue
            if x["op_type"] == "add_inplace" and xi[1] == yi[0] and x["inputs"][0] in rin:
                zero.add(rin[x["inputs"][0]])                         # x + (empty cache)
                continue
        raise RuntimeError(f"node {x['output']} ({x.get('step')}): {x['op_type']} {xi} vs {y['op_type']} {yi}")
    return scal, rots, zero


def plan_rot_scope(plan_dir, progs, dyn_rot=(), half=1 << 15):
    """The rotation steps a planned Session generates keys for (its programs' rotations + `dyn_rot`, mod N/2; the
    refresh routes' LT keys are added by the Session per route) checked against the plan's `plan_rot_steps.json` (the
    key set the plan was made for): a step outside it raises, naming it. Returns
    (program steps, plan steps or None when the plan ships no list)."""
    steps = {int(s) % half for p in progs for s in p.rot_steps()} | {int(s) % half for s in dyn_rot}
    steps.discard(0)
    f = os.path.join(str(plan_dir), "plan_rot_steps.json")
    if not os.path.exists(f):
        return len(steps), None
    with open(f) as fh:
        scope = {int(s) % half for s in json.load(fh)} - {0}
    out = sorted(steps - scope)
    if out:
        raise RuntimeError(f"{f}: the programs rotate by {out[:8]} ({len(out)} steps) outside the plan's rotation keys")
    return len(steps), len(scope)


def register_levels(n_regs, writes):
    """`Session(reg_levels=)`: per register the richest level any program output written to it carries (a register
    is sized for the Q rows of that level); `writes` = (program, output index, register)."""
    lv = [None] * n_regs
    for p, j, r in writes:
        l = p.outputs[j][1]
        lv[r] = l if lv[r] is None else min(lv[r], l)
    return [0 if l is None else l for l in lv]


class PlannedRun:
    """PLANNED on the new runtime (see above). `recorder`: a Recorder on the model's CPU twin; `labels[i]`: the
    program label of stage i (+ CutMax after the stages). `variants` {label: stage}: more programs compiled from a
    stage's stream under the plan's `block_<b>_<variant>_placement.json` (e.g. block 0 for a token fed back by
    generation); they share the stage's registers, dynamic steps and plaintext values."""

    def __init__(self, recorder, plan_dir, config, labels, keys, *, ref_token=2, probe=4, work_dir, pt_rows=0,
                 dyn_rot=None, stc_cache=8, pool=16, log=print, variants=None, schedule="program"):
        self.rec, self.plan_dir, self.cfg, self.labels, self.keys = recorder, plan_dir, config, labels, keys
        self.variants = dict(variants or {})
        self.schedule = schedule          # Program.compile's op order (siblings: refresh pairs)
        self.ref_token, self.probe, self.work_dir, self.pt_rows = ref_token, probe, work_dir, pt_rows
        self.dyn_rot = list(dyn_rot) if dyn_rot is not None else [-i for i in range(-1, 32)]
        self.stc_cache, self.pool, self.log = stc_cache, pool, log
        self.debug = False
        self.records, self._next = {}, None
        self.stats = []
        self._pre, self._staged, self._keys, self._alias = {}, None, {}, {}

    # ── setup ──
    def _record(self, t, rows, argmax):
        if t not in self.records:
            self.records[t] = self.rec.token(t, rows[t], argmax)
        return self.records[t]

    def prepare(self, rows, argmax=False):
        """CPU part of the setup: the probe tokens' streams, the programs compiled from the reference token's, the
        dynamic steps, the register map."""
        from perseus import _pc
        t0 = time.time()
        self.argmax_stage = argmax             # the stages recorded (CutMax included): `token`'s default
        self.rec.values = self.debug           # the debug check compares the recorder's float values
        for t in range(min(self.probe, len(rows))):
            self._record(t, rows, argmax)
        if self.ref_token not in self.records:
            raise RuntimeError(f"the reference token {self.ref_token} needs {self.ref_token + 1} input rows")
        ref = self.records[self.ref_token]
        self.ref = ref
        # the reference stream as a graph dir; the K/V states (and every stage output) stay live out
        live, self.kv = {}, {}
        os.makedirs(self.work_dir, exist_ok=True)
        for b, r in ref.items():
            lab = self.labels[b]
            d = os.path.join(self.work_dir, f"block_{lab.split('_')[1]}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "graph.json"), "w") as f:
                json.dump({"version": 2, "nodes": r.nodes}, f)
            prev = self.records[self.ref_token - 1][b]
            ins = r.inputs()
            kv = [(i, prev.final[id(r.objs[n])]) for i, n in enumerate(ins) if n in r.objs and id(r.objs[n]) in prev.final]
            live[lab] = list(r.out_names) + [nm for _, nm in kv]
            self.kv[b] = kv
        # the K/V inputs enter as the previous token left them (the recorder's state when read); the stage input
        # enters at the previous block's compiled exit (compile_dir chains them)
        metas = {self.labels[b]: {r.inputs()[i]: r.metas[r.inputs()[i]] for i, _ in self.kv[b]} for b, r in ref.items()}
        for lab, b in self.variants.items():
            live[lab], metas[lab] = live[self.labels[b]], metas[self.labels[b]]
        self.progs = _pc.Program.compile(self.work_dir, str(self.plan_dir), config=self.cfg,
                                         blocks=[self.labels[b] for b in ref] + list(self.variants), live_out=live,
                                         input_metas=metas, schedule=self.schedule)
        # dynamic steps: everything a probe token changes
        dyn = {b: set() for b in ref}
        for t, rec in self.records.items():
            for b in ref:
                s, k, _ = diff_block(ref[b], rec[b])
                dyn[b] |= set(s) | set(k)
        # registers: per block its K/V states, then the stage outputs (carry)
        self.reg = {}
        n = 0
        for b in sorted(ref):
            p = self.progs[self.labels[b]]
            self.reg[b] = {"kv": [], "out": []}
            for i, nm in self.kv[b]:
                self.reg[b]["kv"].append((i, p.output_index(nm), n)); n += 1
            for o in ref[b].out_names:
                self.reg[b]["out"].append((p.output_index(o), n)); n += 1
        for lab, b in self.variants.items():
            p = self.progs[lab]
            self.reg[lab] = {"kv": [(i, p.output_index(nm), r) for (i, nm), (_i, _j, r) in zip(self.kv[b], self.reg[b]["kv"])],
                             "out": [(p.output_index(o), r) for o, (_j, r) in zip(ref[b].out_names, self.reg[b]["out"])]}
        # the reference values of the dynamic steps: a token that matches the reference there sets them back
        self.ref_dyn = {}
        for b, r in ref.items():
            byname = {nd["output"]: nd for nd in r.nodes}
            vals = {}
            for nm in dyn[b]:
                nd = byname[nm]
                if nd["op_type"] == "rotate":
                    vals[nm] = ("r", int(nd["inputs"][1][4:-1]))
                else:
                    vals[nm] = ("s", float(nd["args"]["scalar"]))
            self.ref_dyn[b] = vals
        self.dyn, self.n_regs = dyn, n
        self.prepare_s = time.time() - t0
        return self

    def setup(self, rows, argmax=False):
        """`prepare` (CPU) + the planned Session: routes, keys, slab, registers; every block bound."""
        from perseus import _pc
        t0 = time.time()
        if not hasattr(self, "progs"):
            self.prepare(rows, argmax)
        ref, dyn, n = self.ref, self.dyn, self.n_regs
        kw = self.session_kwargs()
        progs, n_pts = kw["programs"], kw["pt_slots"] - 8
        self.rot_keys = plan_rot_scope(self.plan_dir, progs, self.dyn_rot)
        self.sess = self.open_session(kw)
        self.store = self.sess.pt_store
        for b, lab in [(b, self.labels[b]) for b in sorted(ref)] + [(b, lab) for lab, b in self.variants.items()]:
            p = self.progs[lab]
            p.bind(self.sess, self.store)
            if dyn[b]:
                p.mark_dynamic(sorted(dyn[b]))
        self.sess.complete_setup()
        self.fhe = self.sess.fhe
        self._pt_vals = {}
        self.setup_s = time.time() - t0 + getattr(self, "prepare_s", 0.0)
        self.log(f"[planned] {len(progs)} programs from token {self.ref_token}'s stream, "
                 f"{sum(len(v) for v in dyn.values())} dynamic steps, {n} registers, {n_pts} plaintexts; "
                 f"rotation keys {self.rot_keys[0]} (plan_rot_steps.json {self.rot_keys[1]}); "
                 f"session setup {self.sess.setup_seconds()[0]:.1f} s (keygen {self.sess.setup_seconds()[1]:.1f}), "
                 f"total {self.setup_s:.1f} s")
        self.log("[planned] " + self.sess.mem_report.replace("\n", " | "))
        return self

    def session_kwargs(self):
        """The planned Session's arguments after (config, keys); `Session.key_manifest` / `from_bundle` take the same."""
        keys = [(self.labels[b], b) for b in sorted(self.ref)] + [(lab, lab) for lab in self.variants]
        progs = [self.progs[lab] for lab, _k in keys]
        pt_levels = [l for p in progs for _nm, l in sorted({(nm, l) for nm, l, _d, _f in p.pt_refs})]
        writes = []
        for lab, k in keys:
            p = self.progs[lab]
            writes += [(p, j, r) for _i, j, r in self.reg[k]["kv"]] + [(p, j, r) for j, r in self.reg[k]["out"]]
        reg_levels = register_levels(self.n_regs, writes)
        return dict(pool=self.pool, pt_slots=len(pt_levels) + 8, programs=progs, regs=self.n_regs, pt_rows=self.pt_rows,
                    dyn_rot=self.dyn_rot, stc_cache=self.stc_cache, reg_levels=reg_levels, pt_levels=pt_levels)

    def open_session(self, kw):
        from perseus import _pc
        return _pc.Session(self.cfg, self.keys, **kw)

    def input_ct(self, t, rows, p):
        """Block 0's input of token t, encrypted at the program's input level."""
        return self.fhe.pack_tokens(np.asarray([rows[t]], dtype=np.float64), p.inputs[0][1])

    def precompute(self, rows, argmax=False):
        """Setup: every position's patch (its dynamic-step values, the K/V inputs it reads as an empty cache, the
        plaintexts whose values differ from the previous position's), so that a token's in-clock work is setting
        and staging them. The twin records structure only, on the session's public rows, so a position's patch
        depends on the position alone. Plaintexts that hold equal values at every position (the blocks' step masks)
        share one slot (`alias_pts`), so a position stages each changed value once. Once, before the first run."""
        import hashlib
        if self.debug:
            return self
        if self._alias:
            raise RuntimeError("precompute: the plaintexts are already aliased")
        t0, n = time.time(), len(rows)
        self._keys = {lab: p.pt_keys() for lab, p in self.progs.items()}
        steps, pts = [], []
        for t in range(n):
            calls, zero, vals = self._steps(t, self.rec.token(t, rows[t], argmax))
            if pts:                        # equal values share the previous position's array
                last = dict(pts[-1])
                vals = [(key, last[key] if key in last and np.array_equal(last[key], v) else v) for key, v in vals]
            steps.append((calls, zero))
            pts.append(vals)
        # a plaintext's name in the store: the digest of its values at every position (checked equal within a name)
        digest, traj, first = {}, {}, {}
        for i, (key, _v) in enumerate(pts[0]):
            h = hashlib.blake2b(digest_size=12)
            for t in range(n):
                v = pts[t][i][1]
                if id(v) not in digest:
                    a = np.ascontiguousarray(v)
                    digest[id(v)] = hashlib.blake2b(f"{a.dtype.str}{a.shape}".encode() + a.tobytes(),
                                                    digest_size=12).digest()
                h.update(digest[id(v)])
            name = traj[key] = "v/" + h.hexdigest()
            j = first.setdefault(name, i)
            if j != i and not all(np.array_equal(pts[t][j][1], pts[t][i][1]) for t in range(n)):
                raise RuntimeError(f"precompute: digest collision between {pts[0][j][0]} and {key}")
        self._alias = {nm: name for (nm, _lv), name in traj.items()}
        for lab, p in self.progs.items():
            p.alias_pts({nm: self._alias[nm] for nm, _lv in self._keys[lab]})
        for t in range(n):
            for prev in (None, n - 1) if t == 0 else (t - 1,):
                staged = dict(pts[prev]) if prev is not None else {}
                items = []
                for key, v in pts[t]:
                    old = staged.get(key)
                    if old is None or (old is not v and not np.array_equal(old, v)):
                        items.append((*key, v))
                        staged[key] = v
                self._pre[(prev, t)] = {"calls": steps[t][0], "zero": steps[t][1], "items": self._shared(items)}
        self.log(f"[planned] {n} positions precomputed in {time.time() - t0:.1f} s: {len(set(traj.values()))} "
                 f"plaintext slots for {len(traj)} keys, {len(self._pre[(0, 1)]['items']) if n > 1 else 0} staged "
                 f"per position")
        return self

    def _shared(self, items):
        """Staged items [(name, level, values)] under their shared store names, one per name and level."""
        if not self._alias:
            return items
        out = {}
        for nm, lv, v in items:
            out.setdefault((self._alias[nm], lv), v)
        return [(nm, lv, v) for (nm, lv), v in out.items()]

    # ── per token ──
    def _steps(self, t, recs):
        """Token t's streams checked against the reference: the dynamic-step calls [(program, kind, name, value)],
        the K/V inputs read as an empty cache {block: positions}, and the plaintext values [((name, level), v)] by
        the programs' own plaintext names."""
        calls, zero, vals = [], {}, []
        keys = self._keys
        for b, r in recs.items():
            labs = [self.labels[b]] + [lab for lab, vb in self.variants.items() if vb == b]
            s, k, z = diff_block(self.ref[b], r)
            for nm, (kind, v0) in self.ref_dyn[b].items():
                v = s.pop(nm, v0) if kind == "s" else k.pop(nm, v0)
                calls += [(self.progs[lab], kind, nm, v) for lab in labs]
            if s or k:
                raise RuntimeError(f"token {t} block {b}: steps the probe tokens kept constant change: {sorted(s) + sorted(k)}")
            zero[b] = z
            vals += [((nm, lv), r.pts[nm.split("/", 1)[1]]) for lab in labs
                     for nm, lv in keys.get(lab) or self.progs[lab].pt_keys()]
        return calls, zero, vals

    def _apply(self, calls, items):
        for p, kind, nm, v in calls:
            if kind == "s":
                p.set_scalar(nm, v)
            else:
                p.set_rotation(nm, v)
        if items:
            self.store.stage(items)

    def _patch(self, t, recs):
        """Check token t's streams, set the dynamic arguments, stage the plaintexts whose values changed."""
        calls, zero, vals = self._steps(t, recs)
        items = []
        for (nm, lv), v in vals:
            old = self._pt_vals.get((nm, lv))
            if old is None or (old is not v and not np.array_equal(old, v)):
                items.append((nm, lv, v))
                self._pt_vals[(nm, lv)] = v
        items = self._shared(items)
        self._apply(calls, items)
        return zero, len(items)

    def _run_block(self, b, t, zero, first_in, variant=None):
        key = variant or b
        p = self.progs[variant or self.labels[b]]
        ins = [None] * len(p.inputs)
        ins[0] = first_in
        for i, _j, r in self.reg[key]["kv"]:
            ins[i] = None if (t == 0 or i in zero.get(b, ())) else r
        outs = [(j, r) for _i, j, r in self.reg[key]["kv"]] + list(self.reg[key]["out"])
        return p.run(ins, outs)

    def token(self, t, rows, argmax=None, entry=None, variant=None):
        """Issue token t's forward (the blocks and the tail) on the device; returns (the tail output's register,
        record of host times). Asynchronous: decrypting the register waits for it. `entry`: block 0's input (a
        register, e.g. a fed-back token) instead of row t encrypted, run on block 0's program `variant`. `argmax`
        (default: as prepared) records CutMax too: token 0 re-records, and its record stages CutMax's plaintexts."""
        tt = time.time()
        if argmax is None:
            argmax = getattr(self, "argmax_stage", False)
        if self._next is not None:
            self._next.join()
            self._next = None
        pre = self._pre.get((self._staged, t))
        if pre is not None and not self.debug:
            t_rec = 0.0
            self._apply(pre["calls"], pre["items"])
            self._zero, n_staged = pre["zero"], len(pre["items"])
            self._pt_vals.clear()          # the existing path, if ever taken next, restages every plaintext
        else:
            if t == 0:
                # a new sequence: the twin replays it from its start, the reference token included (a stream cached
                # from an earlier sequence would skip that position's K/V push, and every later position would be
                # recorded against a cache one entry short); `self.ref` keeps the compiled reference
                self.records.clear()
            recs = self._record(t, rows, argmax)
            t_rec = time.time() - tt
            # the next token's stream is recorded on the CPU while this one is patched, issued and run (started
            # before the issue, which the device paces: started after it, the record runs into the next token)
            if t + 1 < len(rows) and (t + 1) not in self.records and (t, t + 1) not in self._pre and not self.debug:
                self._next = threading.Thread(target=self._record, args=(t + 1, rows, argmax))
                self._next.start()
            self._zero, n_staged = self._patch(t, recs)
        self._staged = t
        t_patch = time.time() - tt - t_rec
        info = []
        n_blocks = self.rec.n_layers + 1
        for b in range(n_blocks):
            if b == 0:
                x = self.input_ct(t, rows, self.progs[self.labels[0]]) if entry is None else entry
            else:
                x = self.reg[b - 1]["out"][0][1]
            info.append(self._run_block(b, t, self._zero, x, variant if b == 0 else None))
            if self.debug:
                self._check(t, b, recs[b])
        t_issue = time.time() - tt - t_rec - t_patch
        for old in [k for k in list(self.records) if k < t and k != self.ref_token]:
            self.records.pop(old)
        rec = {"record_s": round(t_rec, 3), "patch_s": round(t_patch, 3), "issue_s": round(t_issue, 3),
               "staged": n_staged, "served": "".join(i["served"][0] for i in info)}
        return self.reg[n_blocks - 1]["out"][0][1], rec

    def _check(self, t, b, r):
        """Debug: the block's output and K/V states decrypted vs the recorder's float values (synchronous)."""
        def rel(reg, obj):
            got = np.asarray(self.fhe.decrypt_slots_complex(self.ct(reg)))
            ref = np.asarray(obj.vec)
            return float(np.max(np.abs(got[:ref.shape[0]] - ref)) / max(np.max(np.abs(ref)), 1e-30))
        outs = r.out if isinstance(r.out, (list, tuple)) else [r.out]
        e_out = max(rel(reg, o) for (_j, reg), o in zip(self.reg[b]["out"], outs))
        kv = self.stage_kv_objs(b, r)
        errs = [(nm, rel(reg, o) if o is not None else None) for (_i, _j, reg), o, (_k, nm) in zip(self.reg[b]["kv"], kv, self.kv[b])]
        e_kv = max([e for _n, e in errs if e is not None] or [0.0])
        self.log(f"[planned:check] tok{t} block {b}: out {e_out:.3g} kv {e_kv:.3g}")
        if b == 0:
            self.log("[planned:check]   kv " + " ".join(f"{n}:{'-' if e is None else f'{e:.2g}'}" for n, e in errs))

    def stage_kv_objs(self, b, r):
        """The recorder's K/V objects of block b at the end of its token, in register order (debug)."""
        caches = self.rec.model.stage_caches(b)
        objs = caches[0] if caches else []
        return [next((o for o in objs if r.final.get(id(o)) == nm), None) for _i, nm in self.kv[b]]

    def argmax(self, t):
        """Issue the CutMax block on the last forward's tail output; returns the z register."""
        b = self.rec.n_layers + 1
        self._run_block(b, t, self._zero, self.reg[b - 1]["out"][0][1])
        return self.reg[b]["out"][0][1]

    def ct(self, reg):
        from perseus import _pc
        return _pc.Program.reg_ct(self.sess, reg)

    def close(self):
        if self._next is not None:
            self._next.join()
        self.sess.close()


# ── planned mode with streamed weights and pinned K/V (Llama decode) ───────────────────────────────────────────────
#
# What differs from PlannedRun (GPT-2):
#  * weights: a block's linears are streamed, not resident. The CPU twin records each weight sum as the loop of
#    `mult` + `add_inplace` with plaintexts named by their streamed row (`<linear>#<row>`); the compiler fuses each
#    loop into one `CoeffSum`, and the session expands the float32 coefficient rows on the device inside the sum.
#    The rows live in pinned host memory (`Session.stream_weights`) and one block's linears are uploaded into a
#    device frame on the Copy lane while the block before runs (`Session.stream_order`).
#  * K/V: ~65 states per block x 32 blocks do not fit as registers: they live in pinned host units
#    (`Session(kv=...)`), joined into device pages for their block and copied back after it, the next block's
#    reloaded while one runs. They start from the BOS prefix (encrypted at setup).
#  * program classes: the per-position stream changes structure as the cache grows (SM_PERIODIC=2: a new score
#    period Gp, a new K bucket). Every token of the run is recorded at setup, the tokens grouped by structure (and
#    K/V entry state), and each class compiled from its first token's stream (the plan binds by names, the sparse
#    refresh routes follow the values' period: `routes_built`) and bound under its own labels; the other members'
#    position-dependent arguments are dynamic steps.
#  * plaintexts (masks): recorded per token; equal values (by object, the twin's mask caches) share one store slot
#    (`Program.retarget_pts`, eager replay); a value seen for the first time is encoded, one not read by this token
#    is evicted.

def _sig(r):
    return tuple((n["op_type"], len(n["inputs"]), n.get("step")) for n in r.nodes)


class StreamedPlannedRun:
    """PLANNED with streamed weights and pinned K/V (see above). `recorder`: a Recorder on the model's CPU twin whose
    weight sums record streamed rows (LlamaModel.stream_twin); `labels[i]`: the plan block of stage i (the tail is
    the plan's last block); `streams`: {linear: rows}; `stream_rows(name)`: its float32 coefficient rows;
    `routes_for(pos)`: the extra sparse routes position `pos` holds (SM_PERIODIC=2), or None; `prefix(b)`: {cache
    key suffix: slot vector} of block b's caches before the first token (the BOS prefix; None: the caches start
    empty, token 0 pushes into them). `argmax`: the stages include CutMax after the tail (`argmax(t)` runs it);
    `pt_slots`: the plaintext slots (default: the distinct values a token reads, + 64); `slim`: keep only the
    plaintext values of the non-reference tokens' records once their diffs are taken (512-position runs)."""

    def __init__(self, recorder, plan_dir, config, labels, keys, *, work_dir, streams, stream_rows, prefix=None,
                 kv_init=None, input_vec=None, routes_for=None, frames=None, stc_cache=8, pool=16, base_routes=(256, 1),
                 size=None, dense_fallback=False, extra_pt_slots=0, argmax=False, pt_slots=None, slim=False, log=print,
                 schedule="program"):
        self.rec, self.plan_dir, self.cfg, self.labels, self.keys = recorder, plan_dir, config, labels, keys
        self.schedule = schedule          # Program.compile's op order (siblings: refresh pairs)
        self.work_dir, self.streams, self.stream_rows, self.prefix = work_dir, dict(streams), stream_rows, prefix
        self.input_vec = input_vec or (lambda row: row)
        self.size = size                   # the model geometry (Session size=: token packing, inf.size)
        # the K/V states before token 0: `prefix(b)` {key suffix: slot vector} encrypted at the twin's entry metadata
        # (the BOS prefix), or `kv_init(run, b)` {key suffix: device Ct} of block b (a prefill's hand-off)
        self.kv_init = kv_init
        self.dense_fallback, self.extra_pt_slots = dense_fallback, extra_pt_slots
        self.argmax_stage, self.pt_slots, self.slim = argmax, pt_slots, slim
        self.routes_for, self.frames = routes_for, frames
        self.stc_cache, self.pool, self.log = stc_cache, pool, log
        self.dyn_rot = []
        self.base_routes = tuple(base_routes)
        self.records = {}
        self.n = recorder.n_layers
        self.debug = False
        # debug: (token, block) whose intermediates (up to `tap_per_step` op outputs per step, or every output of
        # `tap_steps`) are kept as program outputs and compared with the twin's values (`tap_report`)
        self.tap, self.tap_per_step, self.tap_steps = None, 4, set()
        self.snaps, self.taps = {}, []
        self.tail = True                   # run the tail (False: a truncated model's test run, blocks only)
        self.check_meta = False            # debug: every op's device metadata vs the compiled one (eager replay)

    # ── setup (CPU) ──
    def _ukey(self, b, key):
        """A cache key `impl.b<b>.vb.3` as its suffix `vb.3`."""
        pre = f"impl.b{b}."
        return key[len(pre):] if key.startswith(pre) else key

    def prepare(self, rows):
        """Record every token, group them into program classes, compile each class. A token joins its class as it is
        recorded: its diff against the class's reference is taken then (and, `slim`, its op stream dropped)."""
        t0 = time.time()
        self.rec.values = self.debug
        self.classes, self.class_of = [], {}
        keyidx, by_hash = {}, {}
        seen, self.n_pts, self.pt_new = None, 0, 0
        for t in range(len(rows)):
            if t in self.records:
                continue
            tt = time.time()
            rec = self.records[t] = self._record_tapped(t, rows) if self.tap and self.tap[0] == t else \
                self.rec.token(t, rows[t], self.argmax_stage)
            if self.slim:
                self._canon_pts(rec, self.records.get(t - 1), by_hash)
            # plaintext slots: the distinct values a token reads, the most a token reads that the one before did not
            have = {(id(rec[b].pts[nm]), lv_) for b in rec for nm, lv_ in self._pt_names(rec, b)}
            self.n_pts = max(self.n_pts, len(have))
            fresh = len(have - seen) if seen is not None else 0
            self.pt_new = max(self.pt_new, fresh)
            seen = have
            # classes: structure + K/V entry state + the class's routes
            pos = t + self.rec.first
            extra = tuple(sorted(self.routes_for(pos))) if self.routes_for else ()
            key = (tuple((_sig(rec[b]), tuple(rec[b].metas.get(x) for x in rec[b].inputs())) for b in sorted(rec)), extra)
            if key not in keyidx:
                if self.slim:          # a new structure: the classes before it take no more members (else reloaded)
                    for k0, c0 in enumerate(self.classes):
                        self._close_class(k0, c0)
                keyidx[key] = len(self.classes)
                self.classes.append({"ref": t, "members": [], "routes": sorted(set(self.base_routes) | set(extra)),
                                     "diffs": {}})
            c = self.classes[keyidx[key]]
            c["members"].append(t)
            self.class_of[t] = keyidx[key]
            ref = self.records[c["ref"]]
            if c.get("closed"):
                self._reload_ref(keyidx[key], c)
            c["diffs"][t] = {}
            for b in ref:
                sc, kk, z = diff_block(ref[b], rec[b])
                if z:
                    raise RuntimeError(f"token {t} block {b}: an empty-cache push in a streamed run (the caches start "
                                       f"from the prefix)")
                c["diffs"][t][b] = (sc, kk)
            if c.get("closed"):
                for r in ref.values():
                    r.nodes = None
            if self.slim and t != c["ref"] and t != 0:
                for r in rec.values():
                    r.nodes, r.objs, r.metas, r.final, r._ins = None, None, None, None, None
            self.log(f"[planned] recorded token {t} (position {pos}) in {time.time() - tt:.1f} s, class {keyidx[key]}, "
                     f"{fresh} new plaintexts")
        self.record_s = time.time() - t0
        n = self.n
        # K/V units: block b's cache keys (suffixes) over every recorded token, one unit each
        suffixes = set()
        for rec in self.records.values():
            for b in range(n):
                suffixes |= {self._ukey(b, k) for k in rec[b].kv_post}
                suffixes |= {self._ukey(b, k) for k in rec[b].kv_pre.values()}
        self.kv_keys = sorted(suffixes, key=lambda k: (k.split(".")[0], int(k.split(".")[1]) if k.split(".")[-1].isdigit() else 0))
        self.U = len(self.kv_keys)
        self.unit = {k: i for i, k in enumerate(self.kv_keys)}
        self.log(f"[planned] {len(rows)} tokens in {len(self.classes)} program classes: "
                 + " ".join(f"{[x + self.rec.first for x in c['members']]}" for c in self.classes))
        tc = time.time()
        for k, c in enumerate(self.classes):
            self._compile_class(k, c)
        self.compile_s = time.time() - tc
        self.prepare_s = time.time() - t0
        return self

    @staticmethod
    def _canon_pts(rec, prev, by_hash):
        """Equal plaintext values as one object (the twin re-creates masks per token, and its weights' slot vectors
        when the cache crosses a group boundary): a value equal to the one the previous token read under the same
        name becomes that object, any other is looked up by a hash of its bytes. One copy per distinct value, staged
        once."""
        import hashlib
        for b, r in rec.items():
            old = prev[b].pts if prev is not None and b in prev else {}
            for nm, v in r.pts.items():
                u = old.get(nm)
                if u is v or (u is not None and np.shape(u) == np.shape(v) and np.array_equal(u, v)):
                    r.pts[nm] = u
                    continue
                a = np.ascontiguousarray(v)
                key = (a.shape, a.dtype.str, hashlib.blake2b(a.tobytes(), digest_size=16).digest())
                r.pts[nm] = by_hash.setdefault(key, v)

    def _record_tapped(self, t, rows):
        """Record token t with the tap block's intermediate values snapshot (twin values on)."""
        fhe = self.rec.model.inf.fhe
        new_var, tb = fhe.new_var, self.tap[1]
        prefix = f"blk{tb}" if tb < self.n else "tail"

        def snap(ct):
            nm = new_var(ct)
            if fhe.path().startswith(prefix) and getattr(ct, "vec", None) is not None and ct.vec.shape[0] > 1:
                self.snaps[nm] = np.array(ct.vec)
            return nm
        fhe.new_var = snap
        try:
            return self.rec.token(t, rows[t])
        finally:
            fhe.new_var = new_var

    def _pick_taps(self, nodes, have):
        by_step = {}
        for nd in nodes:
            o = nd.get("output")
            if o in self.snaps and o not in have and not nd["op_type"].endswith("hint"):
                by_step.setdefault(nd["step"], []).append(o)
        out = []
        for st, names in by_step.items():
            if self.tap_steps and st not in self.tap_steps:
                continue
            k = len(names) if self.tap_steps else min(self.tap_per_step, len(names))
            out += [names[i] for i in sorted({int(round(x)) for x in np.linspace(0, len(names) - 1, k)})]
        return out

    def tap_report(self):
        """[(name, max |device - twin| / max |twin|, max |twin|)] of the tapped intermediates (after the tap token)."""
        out = []
        for nm, reg in zip(self.taps, self.tap_regs):
            got = np.asarray(self.fhe.decrypt_slots_complex(self.ct(reg)))
            want = self.snaps[nm]
            if want.shape != got.shape:
                want = np.resize(want, got.shape)
            den = max(float(np.abs(want).max()), 1e-300)
            out.append((nm, float(np.abs(got - want).max()) / den, den))
        return out

    def _close_class(self, k, c):
        """`slim`: class k's reference stream to disk (its graph dir, which the compile reads anyway) and out of memory,
        keeping what the compile takes from it: inputs, K/V entries, their metadata, the scalar / rotation steps. A
        512-position run opens a class every few positions, and each reference stream holds GBs (token 0 stays whole:
        setup reads its K/V entry states)."""
        if c.get("closed") or c["ref"] == 0:
            return
        c["graphs"] = self._write_graphs(k, c)
        c["ref_info"] = {}
        for b, r in self.records[c["ref"]].items():
            ins = r.inputs()
            kv_in = [(i, self._ukey(b, r.kv_pre[id(r.objs[x])])) for i, x in enumerate(ins)
                     if x in r.objs and id(r.objs[x]) in r.kv_pre]
            steps = {nd["output"]: (("r", int(nd["inputs"][1][4:-1])) if nd["op_type"] == "rotate"
                                    else ("s", nd["args"]["scalar"]))
                     for nd in r.nodes if nd["op_type"] == "rotate" or nd.get("args", {}).get("scalar") is not None}
            c["ref_info"][b] = (kv_in, {ins[i]: r.metas[ins[i]] for i, _ in kv_in}, steps)
            r.nodes, r.objs, r.metas = None, None, None
        c["closed"] = True

    def _reload_ref(self, k, c):
        """A closed class's reference stream back from its graph dir (a later token of the same structure)."""
        for b, r in self.records[c["ref"]].items():
            with open(os.path.join(c["graphs"], f"block_{self.labels[b].split('_')[1]}", "graph.json")) as f:
                r.nodes = json.load(f)["nodes"]

    def _write_graphs(self, k, c):
        """Class k's reference token as a graph dir (block_<label>/graph.json); returns the dir."""
        wd = os.path.join(self.work_dir, f"c{k}")
        for b, r in self.records[c["ref"]].items():
            d = os.path.join(wd, f"block_{self.labels[b].split('_')[1]}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "graph.json"), "w") as f:
                json.dump({"version": 2, "nodes": r.nodes}, f)
        return wd

    def _compile_class(self, k, c):
        from perseus import _pc
        ref = self.records[c["ref"]]
        closed = c.get("closed", False)
        wd = c["graphs"] if closed else self._write_graphs(k, c)
        live, metas, io = {}, {}, {}
        for b, r in ref.items():
            lab = self.labels[b]
            ins = r.inputs()
            if closed:
                kv_in = c["ref_info"][b][0]
            else:
                kv_in = [(i, self._ukey(b, r.kv_pre[id(r.objs[x])])) for i, x in enumerate(ins) if x in r.objs and id(r.objs[x]) in r.kv_pre]
            live[lab] = list(r.out_names) + list(r.kv_post.values())
            if self.tap and c["ref"] == self.tap[0] and b == self.tap[1]:
                self.taps = self._pick_taps(r.nodes, set(live[lab]))
                live[lab] += self.taps
            metas[lab] = c["ref_info"][b][1] if closed else {ins[i]: r.metas[ins[i]] for i, _ in kv_in}
            io[b] = kv_in
        progs = _pc.Program.compile(wd, str(self.plan_dir), config=self.cfg, blocks=[self.labels[b] for b in ref],
                                    live_out=live, input_metas=metas, routes_built=list(c["routes"]),
                                    dense_fallback=self.dense_fallback, schedule=self.schedule)
        c["progs"] = {}
        c["kv_in"], c["kv_out"], c["out"] = {}, {}, {}
        for b, r in ref.items():
            p = progs[self.labels[b]]
            p.relabel(f"c{k}.{self.labels[b]}")
            c["progs"][b] = p
            c["kv_in"][b] = [(i, b * self.U + self.unit[key]) for i, key in io[b]]
            c["kv_out"][b] = [(p.output_index(nm), b * self.U + self.unit[self._ukey(b, key)]) for key, nm in r.kv_post.items()]
            c["out"][b] = [p.output_index(o) for o in r.out_names]
        # dynamic steps: everything another member token changes (diffs taken when the tokens were recorded)
        dyn = {b: set() for b in ref}
        for t in c["members"]:
            for b in ref:
                s, kk = c["diffs"][t][b]
                dyn[b] |= set(s) | set(kk)
                self.dyn_rot = sorted(set(self.dyn_rot) | set(kk.values()))
        c["dyn"] = dyn
        c["ref_dyn"] = {}
        for b, r in ref.items():
            if closed:
                steps = c["ref_info"][b][2]
                c["ref_dyn"][b] = {nm: (steps[nm][0], float(steps[nm][1]) if steps[nm][0] == "s" else steps[nm][1])
                                   for nm in dyn[b]}
            else:
                byname = {nd["output"]: nd for nd in r.nodes}
                c["ref_dyn"][b] = {nm: (("r", int(byname[nm]["inputs"][1][4:-1])) if byname[nm]["op_type"] == "rotate"
                                        else ("s", float(byname[nm]["args"]["scalar"]))) for nm in dyn[b]}
            self.dyn_rot = sorted(set(self.dyn_rot) | {v for kind, v in c["ref_dyn"][b].values() if kind == "r"})
        ops = sum(p.n_ops for p in c["progs"].values())
        sums = sum(len(p.coeff_sums) for p in c["progs"].values())
        bts = sum(p.n_bootstraps for p in c["progs"].values())
        self.log(f"[planned] class {k} (positions {[t + self.rec.first for t in c['members']]}, routes {c['routes']}): "
                 f"{len(c['progs'])} programs, {ops} ops, {sums} streamed sums, {bts} refreshes, "
                 f"{sum(len(v) for v in dyn.values())} dynamic steps")

    def setup(self, rows):
        """`prepare` + the Session: routes, keys, slab, weight frames, pinned K/V; every class bound; the weights
        uploaded to pinned memory; the K/V units filled with the encrypted BOS prefix."""
        from perseus import _pc
        t0 = time.time()
        if not hasattr(self, "classes"):
            self.prepare(rows)
        n = self.n
        progs = [c["progs"][b] for c in self.classes for b in sorted(c["progs"])]
        # registers: block b's output -> b; the tail's outputs after
        n_tail = max(len(c["out"][n]) for c in self.classes)
        self.reg_out = {b: [b] for b in range(n)}
        self.reg_out[n] = [n + j for j in range(n_tail)]
        regs = n + n_tail
        if self.argmax_stage:
            n_z = max(len(c["out"][n + 1]) for c in self.classes)
            self.reg_out[n + 1] = [regs + j for j in range(n_z)]
            regs += n_z
        self.tap_regs = [regs + j for j in range(len(self.taps))]
        regs += len(self.taps)
        # K/V rows: the deepest state any program writes or reads
        lv = []
        for c in self.classes:
            for b in range(n):
                p = c["progs"][b]
                lv += [p.outputs[j][1] for j, _u in c["kv_out"][b]] + [p.inputs[i][1] for i, _u in c["kv_in"][b]]
        kv_rows = 54 - min(lv) if lv else 1
        pages = 2 * self.U
        # plaintext slots: the most (value, level) pairs one token's class programs read (`_patch`'s keys). `_patch`
        # evicts the values a token no longer reads before it stages the new ones, so the store never holds more than
        # one token reads; a staged upload waits on the compute lane's hand-off either way, so reusing an evicted slot
        # costs nothing (2 x the values: ~17 GB at 511 Llama positions)
        peak, at = 0, 0
        for t in sorted(self.records):
            r, progs_t = self.records[t], self.classes[self.class_of[t]]["progs"]
            k = len({(id(r[b].pts[nm]), lv_) for b in r for nm, lv_, _d, f in progs_t[b].pt_refs if f != "stream"})
            if k > peak:
                peak, at = k, t
        n_slots = self.pt_slots if self.pt_slots is not None else peak + 64
        self.log(f"[planned] plaintext slots {n_slots + self.extra_pt_slots}: the most one token reads {peak} (position "
                 f"{at + self.rec.first}; recorded {self.n_pts})")
        pt_level = min(lev for c in self.classes for p in c["progs"].values() for _nm, lev, _d, f in p.pt_refs if f != "stream")
        # weight frames: the Config's host.n_in_flight (default 2), raised until the last block and the next token's
        # first one sit in different frames
        frames = self.frames or self._frames(n + 1, int(self.cfg["host.n_in_flight"] or 2))
        self.sess = _pc.Session(self.cfg, self.keys, size=self.size, pool=self.pool, pt_slots=n_slots + self.extra_pt_slots, programs=progs,
                                regs=regs, pt_rows=54 - pt_level, dyn_rot=self.dyn_rot, stc_cache=self.stc_cache,
                                streams=self.streams, stream_frames=frames, kv=(n * self.U, kv_rows, pages),
                                check_meta=self.check_meta)
        self.store = self.sess.pt_store
        for c in self.classes:
            for b, p in c["progs"].items():
                p.bind(self.sess, self.store)
                if c["dyn"][b]:
                    p.mark_dynamic(sorted(c["dyn"][b]))
                c.setdefault("pt_idx", {})[b] = [(i, nm, lv_) for i, (nm, lv_, _d, f) in enumerate(p.pt_refs) if f != "stream"]
        c0 = self.classes[0]
        self.sess.stream_order([c0["progs"][b] for b in sorted(c0["progs"]) if c0["progs"][b].coeff_sums])
        self.sess.complete_setup()
        self.fhe = self.sess.fhe
        tw = time.time()
        for name in self.streams:
            self.sess.stream_weights(name, self.stream_rows(name))
        self.weights_s = time.time() - tw
        # the K/V states token 0 reads: the BOS prefix (each state encrypted at the twin's entry metadata of the first
        # token) or a hand-off's device states
        rec0, kp = self.records[0], 0
        ti = time.time()
        for b in range(n):
            r = rec0[b]
            given = self.kv_init(self, b) if self.kv_init is not None else None
            vecs = self.prefix(b) if given is None and self.prefix is not None else None
            for x in r.inputs():
                o = r.objs.get(x)
                if o is None or id(o) not in r.kv_pre:
                    continue
                key = self._ukey(b, r.kv_pre[id(o)])
                if given is None and vecs is None:
                    raise RuntimeError(f"block {b}: token 0 reads the cache state {key}, and the caches start empty")
                lvl, deg = r.metas[x]
                if given is None:
                    ct = self.fhe.encrypt(np.asarray(vecs[key]), lvl, deg)
                else:
                    ct = given[key]
                    if (int(ct.level), int(ct.deg)) != (lvl, deg):
                        raise RuntimeError(f"block {b} {key}: the handed state is at ({ct.level}, {ct.deg}), the twin read it at ({lvl}, {deg})")
                self.sess.kv_put(b * self.U + self.unit[key], ct)
                kp += 1
            del given
        self.fhe.device_sync()
        self.kv_init_s = time.time() - ti
        self._resident = {}          # (id(values), level) -> (values, store name)
        self._pt_serial = 0
        self.setup_s = time.time() - t0 + getattr(self, "prepare_s", 0.0)
        st = self.sess.stream_stats()
        self.log(f"[planned] {len(progs)} programs in {len(self.classes)} classes, {regs} registers, K/V {n} x {self.U} "
                 f"units of {kv_rows} rows ({pages} device pages, {kp} prefix states), {frames} weight frames of "
                 f"{st.get('frame_rows')} rows, {st['pinned_bytes'] / 2**30:.1f} GB pinned weights (upload "
                 f"{self.weights_s:.1f} s); record {self.record_s:.1f} s, compile {self.compile_s:.1f} s, session "
                 f"{self.sess.setup_seconds()[0]:.1f} s (keygen {self.sess.setup_seconds()[1]:.1f}), total {self.setup_s:.1f} s")
        self.log("[planned] " + self.sess.mem_report.replace("\n", " | "))
        return self

    @staticmethod
    def _frames(n_blocks, at_least=2):
        """Weight frames: >= at_least (>= 2), and the last block and the first one (the next token's) in different
        frames."""
        f = max(2, at_least)
        while n_blocks > 1 and (n_blocks - 1) % f == 0:
            f += 1
        return f

    @staticmethod
    def _pt_names(rec, b):
        """(graph pt name, level) of block b's non-streamed plaintext reads."""
        out = []
        for nd in rec[b].nodes:
            a = nd.get("args", {}).get("pt")
            if a and a.get("kind") != "stream":
                out.append((nd["inputs"][1], a["level"]))
        return out

    # ── per token ──
    def _patch(self, t, c, recs):
        """Token t's dynamic arguments and plaintexts on class c's programs (eager: slots by value)."""
        need, items = {}, []
        for b, r in recs.items():
            p = c["progs"][b]
            # the token's stream was checked against the class's reference when the classes were made (prepare)
            s, k = (dict(x) for x in c["diffs"][t][b]) if t in c["diffs"] else diff_block(self.records[c["ref"]][b], r)[:2]
            for nm, (kind, v0) in c["ref_dyn"][b].items():
                if kind == "s":
                    p.set_scalar(nm, s.pop(nm, v0))
                else:
                    p.set_rotation(nm, k.pop(nm, v0))
            if s or k:
                raise RuntimeError(f"token {t} block {b}: steps the class's tokens kept constant change: {sorted(s) + sorted(k)}")
            pairs = []
            for i, nm, lv in c["pt_idx"][b]:
                v = r.pts[nm]
                key = (id(v), lv)
                if key not in need:
                    have = self._resident.get(key)
                    if have is None:
                        self._pt_serial += 1
                        have = (v, f"v{self._pt_serial}")
                        items.append((have[1], lv, v))
                    need[key] = have
                pairs.append((i, need[key][1]))
            p.retarget_pts(pairs)
        gone = [(name, lv) for key, (_v, name) in self._resident.items() if key not in need for lv in (key[1],)]
        if gone:
            self.store.evict(gone)
        if items:
            self.store.stage(items)
        self._resident = need
        return len(items)

    def token(self, t, rows, argmax=None):
        """Issue token t's forward; returns (the tail's output registers, host times). Asynchronous. `argmax`: as
        constructed (every record stages CutMax from setup)."""
        tt = time.time()
        if t not in self.records:
            self.records[t] = self.rec.token(t, rows[t], self.argmax_stage)
        recs = self.records[t]
        c = self.classes[self.class_of[t]]
        t_rec = time.time() - tt
        n_staged = self._patch(t, c, recs)
        t_patch = time.time() - tt - t_rec
        info = []
        n, U = self.n, self.U
        units = lambda b: [b * U + j for j in range(U)] if 0 <= b < n else []
        for b in range(n + 1 if self.tail else n):
            p = c["progs"][b]
            ins = [None] * len(p.inputs)
            if b == 0:
                vec = np.asarray(self.input_vec(rows[t]), dtype=np.float64)
                ins[0] = self.fhe.pack_tokens(np.asarray([vec]), p.inputs[0][1])
            else:
                ins[0] = self.reg_out[b - 1][0]
            outs = [(j, reg) for j, reg in zip(c["out"][b], self.reg_out[b])]
            if self.tap and (t, b) == tuple(self.tap):
                outs += [(p.output_index(nm), reg) for nm, reg in zip(self.taps, self.tap_regs)]
            if b < n:
                nxt = units(b + 1) if b + 1 < n else units(0)
                info.append(p.run(ins, outs, kv_in=c["kv_in"][b], kv_out=c["kv_out"][b], kv_next=nxt,
                                  kv_done=units(b - 1 if b > 0 else n - 1)))
            else:
                info.append(p.run(ins, outs))
            if self.check_meta and info[-1].get("meta_mismatches"):
                self.log(f"[planned:meta] tok{t} block {b}: {info[-1]['meta_mismatches']} mismatches, first "
                         f"{info[-1]['first_mismatches'][:3]}")
            if self.debug:
                self._check(t, b, recs[b])
        t_issue = time.time() - tt - t_rec - t_patch
        rec = {"record_s": round(t_rec, 3), "patch_s": round(t_patch, 3), "issue_s": round(t_issue, 3),
               "staged": n_staged, "class": self.class_of[t], "served": "".join(i["served"][0] for i in info)}
        self._last = t
        return self.reg_out[n], rec

    def argmax(self, t):
        """Issue the CutMax stage on token t's tail output (after `token(t)`); returns its output registers."""
        n = self.n
        p = self.classes[self.class_of[t]]["progs"][n + 1]
        ins = [None] * len(p.inputs)
        ins[0] = self.reg_out[n][0]
        p.run(ins, [(j, reg) for j, reg in zip(self.classes[self.class_of[t]]["out"][n + 1], self.reg_out[n + 1])])
        return self.reg_out[n + 1]

    def _check(self, t, b, r):
        """Debug: the block's output decrypted vs the recorder's float values (synchronous)."""
        outs = r.out if isinstance(r.out, (list, tuple)) else [r.out]
        errs = []
        for reg, o in zip(self.reg_out[b], outs):
            got = np.asarray(self.fhe.decrypt_slots_complex(self.ct(reg)))
            ref = np.asarray(o.vec)
            errs.append(float(np.max(np.abs(got[:ref.shape[0]] - ref)) / max(np.max(np.abs(ref)), 1e-30)))
        self.log(f"[planned:check] tok{t} block {b}: out {max(errs):.3g}")

    def ct(self, reg):
        from perseus import _pc
        return _pc.Program.reg_ct(self.sess, reg)

    def close(self):
        self.sess.close()
