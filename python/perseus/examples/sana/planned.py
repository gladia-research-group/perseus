"""SANA-Sprint (and the text-to-image forward: Gemma-2 -> caption -> SANA) PLANNED on the device: one compiled `Program`
per stage, run in Rust, on the Llama prefill's pipeline (examples/llama/planned.py): the op stream recorded once on the
CPU twin for the set prompt length, its plaintexts kept once per distinct value, the stage outputs in two register
sets, the PCMM linears served by the session.

Unlike Llama's blocks the stages differ (the Gemma layers, the caption, embed, the blocks, the head) and their plans
bind input by input (the planner's PLAN_ENTRY_MAP), so stage b compiles against stage b - 1's compiled exit, as the
encoder prefill does (examples/bert/planned.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from perseus.impl.pcmm import deferred_planes, session_specs
from perseus.impl.prefill import PlannedPrefill


def record_workers():
    """PERSEUS_RECORD_WORKERS, else the process's CPUs less two (at most 32): the threads encoding PCMM weights
    while a record runs."""
    if os.environ.get("PERSEUS_RECORD_WORKERS"):
        return int(os.environ["PERSEUS_RECORD_WORKERS"])
    return max(1, min(32, len(os.sched_getaffinity(0)) - 2))


# switches that change how a record runs, not what it records
_KEY_SKIP = {"PERSEUS_RECORD_WORKERS", "PERSEUS_T2I_THREADS", "PERSEUS_T2I_PLANS", "CUDA_VISIBLE_DEVICES",
             "OMP_NUM_THREADS"}


def record_key(n_stages, files=()):
    """What a record is a function of: the perseus package's Python and its runtime build, every environment switch
    the package reads (by name, or by the SANA_ / PF_ / FIDESLIB_ / PC_ / PLAN_ / FAKECAP_ / CKKS_ prefixes), the
    `files` (configs, PCMM calibration, the prompt's dumps) and the stage count."""
    import re
    import perseus
    pkg = os.path.dirname(os.path.abspath(perseus.__file__))
    h = hashlib.blake2b(digest_size=16)
    names = set()
    pat = re.compile(r"""(?:environ(?:\.get)?\(|environ\[|getenv\()\s*["']([A-Z][A-Z0-9_]+)["']""")
    for root, dirs, fs in os.walk(pkg):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(fs):
            if f.endswith(".py") or (f.startswith("_pc.") and f.endswith(".so")):
                p = os.path.join(root, f)
                with open(p, "rb") as fh:
                    b = fh.read()
                h.update(os.path.relpath(p, pkg).encode() + hashlib.blake2b(b, digest_size=16).digest())
                if f.endswith(".py"):
                    names.update(pat.findall(b.decode("utf-8", "replace")))
    pre = ("SANA_", "PF_", "FIDESLIB_", "PC_", "PLAN_", "FAKECAP_", "CKKS_")
    env = sorted((k, v) for k, v in os.environ.items() if (k in names or k.startswith(pre)) and k not in _KEY_SKIP)
    h.update(json.dumps(env).encode())
    for p in files:
        with open(p, "rb") as fh:
            h.update(os.path.basename(p).encode() + hashlib.blake2b(fh.read(), digest_size=16).digest())
    h.update(str(int(n_stages)).encode())
    return h.hexdigest()


def _key(v):
    return "v/" + hashlib.blake2b(np.ascontiguousarray(v).tobytes(), digest_size=12).hexdigest()


def program_digest(nodes, placement, live_out, metas, schedule):
    """What a stage's compiled program is a function of: its op stream (the capture's magnitude statistics left out:
    a recorded stream has none), its placement, the outputs kept, the inputs' entry metadata and the schedule.
    bind_plan saves each program with this digest (<plan>/programs/); a run whose recorded stage gives the same one
    loads the program instead of compiling it."""
    h = hashlib.blake2b(digest_size=16)
    for nd in nodes:
        h.update(json.dumps({k: v for k, v in nd.items() if not k.startswith("output_m")}, sort_keys=True).encode())
    with open(placement, "rb") as f:
        h.update(hashlib.blake2b(f.read(), digest_size=16).digest())
    h.update(json.dumps([list(live_out), sorted((n, list(m)) for n, m in metas.items()), schedule]).encode())
    return h.hexdigest()


def stage_levels(sets, nq):
    """`Session(pt_levels=)` for a store any one of `sets` ({(key, level)}) fits: per row count r (nq - level) as many
    slots of at least r rows as the most keys of at least r rows one set holds (a key takes the smallest free slot it
    fits)."""
    rows = sorted({nq - lv for s in sets for _k, lv in s}, reverse=True)
    out, have = [], 0
    for r in rows:
        need = max(sum(1 for _k, lv in s if nq - lv >= r) for s in sets)
        out += [nq - r] * (need - have)
        have = max(have, need)
    return out


def store_spare(sets, nq, pt_levels, rounds=3):
    """Full-height store slots `stage_pts` needs beside the classes (`stage_levels`): a key consecutive stages share
    keeps the slot it took, which can be a larger class than a fresh placement gives it, so a later stage can find a
    class short. Replays the staging over `rounds` forwards (evict what a stage does not read, stage the rest in its
    set's order into the smallest free slot that fits, a full-height slot when none does) and returns the most
    full-height slots in use at once."""
    import bisect
    free = sorted(nq - lv for lv in pt_levels)          # rows of the free class slots
    held, prev, spare, most = {}, set(), 0, 0
    for need in list(sets) * rounds:
        for k in prev - need:
            r = held.pop(k)
            if r is None:
                spare -= 1
            else:
                bisect.insort(free, r)
        for k in need:
            if k in held:
                continue
            i = bisect.bisect_left(free, nq - k[1])
            if i < len(free):
                held[k] = free.pop(i)
            else:
                held[k] = None
                spare += 1
                most = max(most, spare)
        prev = need
    return most


class SanaPlannedPrefill(PlannedPrefill):
    """`PlannedPrefill` for `SanaImpl` / `SanaTextImpl` (their `stages()`: flat ciphertext lists in, out)."""

    def __init__(self, model, plan_dir, config, keys, **kw):
        super().__init__(model, plan_dir, config, keys, **kw)
        if os.environ.get("SANA_PCMM_BATCH"):         # finish group size (runtime only; the plan's is 4): 2 measured 4% faster per output
            self.cfg["model.prefill.pcmm_batch"] = int(os.environ["SANA_PCMM_BATCH"])
        self.n_stages = len(model.stages())
        self.labels = [f"block_{b}" for b in range(self.n_stages)]

    # ── setup (CPU) ──
    def record(self, rows=None):
        """The forward's per-stage streams on the CPU twin (the model's set prompt; `rows` unused). Plaintext values
        are kept once per distinct value (`self.pt_vals`); a stage's record maps its plaintext names to those keys and
        drops its ciphertext objects (a SANA block makes ~76k). The twin runs on one slot per ciphertext
        (fakecap.CapCt.novec, as the Llama and GPT-2 planned drivers record): the op stream and the plaintexts do not
        depend on the message."""
        from perseus.impl import fakecap
        from perseus.impl.driver import Recorder
        t0 = time.time()
        m = self.model
        rec = Recorder(m, self.n_stages)
        orig = m.prefill_stages
        io, self.pt_vals = {}, {}

        def stages(ch):
            out = []
            for i, (lab, pre, fn) in enumerate(orig(ch)):
                def wrap(x, i=i, fn=fn):
                    metas = [(int(c.level), int(c.noise_deg)) for c in x]
                    vals = None
                    y = fn(x)
                    # graph names as this stage's capture wrote them (the next stage renames what it reads)
                    entry = getattr(m, "_entry_names", {}).get(i)
                    names_in = [e if e else c.name for e, c in zip(entry, x)] if entry else [c.name for c in x]
                    io[i] = (names_in, [c.name for c in y], metas, vals)
                    r = rec.blocks[i]
                    for nm, v in list(r.pts.items()):
                        r.pts[nm] = k = _key(v)
                        self.pt_vals.setdefault(k, np.ascontiguousarray(v))
                    r.objs = {}
                    if getattr(self, "progress", None) is not None:      # a CLI's progress bar (perseus.t2i)
                        self.progress("record", i + 1, self.n_stages)
                    return y
                out.append((lab, pre, wrap))
            return out
        m.prefill_stages = stages
        fakecap.CapCt.novec = True
        try:
            # the PCMM weights' digit planes (most of a record's time) on worker threads, beside the op stream
            with deferred_planes(record_workers()):
                m.prefill()
        finally:
            del m.prefill_stages
            fakecap.CapCt.novec = False
        self.in_metas, self.in_names = {}, {}
        for i, r in rec.blocks.items():
            names_in, names_out, metas, vals = io[i]
            r.out_names = names_out
            self.in_names[i] = names_in
            self.in_metas[i] = metas
            if i == 0:
                self.in_vals = [np.asarray(v) for v in m.input_values()]
        self.recs = rec.blocks
        self.record_s = time.time() - t0
        return self.recs

    # ── the record cache (a CLI's next launch loads the record instead of running the twin) ──
    def record_cached(self, cache_dir, files=()):
        """`record`, or the one saved in `cache_dir` when its key (`record_key`: the package's code, the switches,
        `files`) matches; a fresh record is saved there for the next run. Returns True when it was loaded."""
        key = record_key(self.n_stages, files)
        if self.load_record(cache_dir, key):
            return True
        self.record()
        self.save_record(cache_dir, key)
        return False

    def save_record(self, d, key):
        """The record (each stage's stream, the stage inputs, the distinct plaintexts, the PCMM linears with their
        digit planes) under directory d: written beside it, then swapped in."""
        import copy
        import pickle
        import shutil
        t0 = time.time()
        tmp, old = f"{d}.tmp{os.getpid()}", f"{d}.old{os.getpid()}"
        os.makedirs(tmp)

        def blob(name, arrays):
            out, off = [], 0
            with open(os.path.join(tmp, name), "wb") as f:
                for k, v in arrays:
                    v = np.ascontiguousarray(v)
                    v.tofile(f)
                    out.append((k, v.dtype.str, v.shape, off))
                    off += v.nbytes
                    pad = -off % 64
                    f.write(b"\0" * pad)
                    off += pad
            return out
        pts = blob("pt.bin", self.pt_vals.items())
        lins = self.model.lin._enc
        wds = blob("wd.bin", ((nm, lin.Wd) for nm, lin in lins.items()))
        bare = {}
        for nm, lin in lins.items():
            bare[nm] = copy.copy(lin)
            bare[nm].Wd = None
        meta = dict(version=1, key=key, n_stages=self.n_stages, record_s=self.record_s,
                    recs={b: (r.nodes, r.pts, r.metas, r.out_names) for b, r in self.recs.items()},
                    in_names=self.in_names, in_metas=self.in_metas, in_vals=self.in_vals, pts=pts, wds=wds, lins=bare)
        with open(os.path.join(tmp, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f, protocol=5)
        with open(os.path.join(tmp, "key.txt"), "w") as f:
            f.write(key)
        if os.path.exists(d):
            os.rename(d, old)
        os.rename(tmp, d)
        shutil.rmtree(old, ignore_errors=True)
        self.save_s = time.time() - t0

    def load_record(self, d, key):
        """The record saved by `save_record` under d if its key is `key` (else False, nothing changed)."""
        import pickle
        from perseus.impl.driver import BlockRec
        try:
            with open(os.path.join(d, "key.txt")) as f:
                if f.read() != key:
                    return False
            t0 = time.time()
            with open(os.path.join(d, "meta.pkl"), "rb") as f:
                meta = pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError):
            return False
        if meta.get("key") != key or meta.get("n_stages") != self.n_stages:
            return False

        def views(name, entries):
            buf = np.fromfile(os.path.join(d, name), dtype=np.uint8)
            return {k: buf[off:off + int(np.prod(shape)) * np.dtype(dt).itemsize].view(dt).reshape(shape)
                    for k, dt, shape, off in entries}
        self.pt_vals = views("pt.bin", meta["pts"])
        wd = views("wd.bin", meta["wds"])
        for nm, lin in meta["lins"].items():
            lin.Wd = wd[nm]
        self.model.lin._enc = dict(meta["lins"])
        self.recs = {}
        for b, (nodes, pts, metas, out_names) in meta["recs"].items():
            r = BlockRec(nodes, pts, {}, metas)
            r.out_names = out_names
            self.recs[b] = r
        self.in_names, self.in_metas, self.in_vals = meta["in_names"], meta["in_metas"], meta["in_vals"]
        self.recorded_s = meta["record_s"]
        self.record_s = time.time() - t0
        return True

    def prepare(self, rows=None):
        """`record` + the programs compiled stage by stage: stage b > 0 enters at stage b - 1's compiled exit. The
        registers ping-pong between two sets (Llama's): stage b writes set b % 2 and reads set (b - 1) % 2, so its
        outputs live until stage b + 2 overwrites them (per-stage checks decrypt in `on_block`)."""
        from perseus import _pc
        if not hasattr(self, "recs"):
            self.record(rows)
        t0 = time.time()
        os.makedirs(self.work_dir, exist_ok=True)
        # Program.compile compiles every block of the graph dir and keeps `blocks`: stage b compiles from a dir that
        # holds only its own graph (a link), else stage b recompiles stages 0..b (quadratic: hours at 87 stages)
        one = os.path.join(self.work_dir, "_one")
        os.makedirs(one, exist_ok=True)
        # ... and reads every placement of the plan dir on each call (4 passes over 247 MB of JSON at 87 stages: 9 s
        # a call, 13 min a run): stage b compiles from a plan dir that holds only its own placement (links)
        plan1 = os.path.join(self.work_dir, "_plan1")
        os.makedirs(plan1, exist_ok=True)
        shared = [f for f in os.listdir(self.plan_dir) if not f.endswith("_placement.json")]
        self.progs, self.in_pos = {}, {}
        for b in range(self.n_stages):
            r, lab = self.recs[b], self.labels[b]
            d = os.path.join(self.work_dir, lab)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "graph.json"), "w") as f:
                json.dump({"version": 2, "nodes": r.nodes}, f)
            for x in os.listdir(one):
                os.unlink(os.path.join(one, x))
            os.symlink(os.path.join("..", lab), os.path.join(one, lab))
            for x in os.listdir(plan1):
                os.unlink(os.path.join(plan1, x))
            for f in shared + [f"{lab}_placement.json"]:
                os.symlink(os.path.join(os.path.abspath(self.plan_dir), f), os.path.join(plan1, f))
            names, ins = self.in_names[b], r.inputs()
            missing = [n for n in ins if n not in names]
            if missing:
                raise RuntimeError(f"stage {b}: inputs {missing} are not the stage's input ciphertexts {names}")
            self.in_pos[b] = [names.index(n) for n in ins]          # program input i = stage input in_pos[i]
            if b:                                                   # stage input i = stage b - 1's output i
                got = {n: (int(lv), int(dg)) for n, lv, dg, _s in self.progs[self.labels[b - 1]].outputs}
                self.in_metas[b] = [got[n] for n in self.recs[b - 1].out_names]
            metas = {lab: {n: self.in_metas[b][names.index(n)] for n in ins}}
            saved = os.path.join(self.plan_dir, "programs", lab + ".json")
            dg = program_digest(r.nodes, os.path.join(plan1, f"{lab}_placement.json"), r.out_names, metas[lab],
                                self.schedule)
            if os.path.exists(saved + ".digest") and open(saved + ".digest").read() == dg:
                self.progs[lab] = _pc.Program.load(saved, plan1)       # bind_plan's (the same compile)
                self.n_loaded = getattr(self, "n_loaded", 0) + 1
            else:
                self.progs.update(_pc.Program.compile(one, plan1, config=self.cfg, blocks=[lab],
                                                      live_out={lab: list(r.out_names)}, input_metas=metas,
                                                      schedule=self.schedule))
                if getattr(self, "save_programs", False):          # kept for the next run (as bind_plan saves)
                    os.makedirs(os.path.dirname(saved), exist_ok=True)
                    self.progs[lab].save(saved + ".tmp")
                    os.replace(saved + ".tmp", saved)
                    with open(saved + ".digest.tmp", "w") as f:
                        f.write(dg)
                    os.replace(saved + ".digest.tmp", saved + ".digest")
            r.nodes, r.metas = None, None                           # compiled (graph.json keeps them)
            if getattr(self, "progress", None) is not None:
                self.progress("compile", b + 1, self.n_stages)
        width = max(len(self.recs[b].out_names) for b in range(self.n_stages))
        self.reg = {b: [(self.progs[self.labels[b]].output_index(o), (b % 2) * width + k)
                        for k, o in enumerate(self.recs[b].out_names)] for b in range(self.n_stages)}
        self.n_regs = min(self.n_stages, 2) * width
        self.compile_s = time.time() - t0
        return self

    def linears(self):
        return self.model.lin.linears()

    def planes(self, nm):
        """Linear `nm`'s digit planes as the bytes `pcmm_upload` takes, made once: pooled weights upload every stage
        of every forward (16.7 GB a forward at 768 px), so the copy is not repeated (host memory: the planes twice)."""
        if nm not in self._planes:
            self._planes[nm] = np.ascontiguousarray(self.linears()[nm].Wd, dtype=np.int8).tobytes()
        return self._planes[nm]

    def session_kwargs(self):
        """The planned Session's arguments after (config, keys); keeps the plaintext aliases, the distinct
        (value, level) plaintexts (`self.uniq`) and each stage's (`self.stage_keys`). The store holds the most one
        stage reads (+ 64), its slot classes the most keys of each size one stage reads."""
        from perseus import _pc
        from perseus.impl.driver import register_levels
        progs = [self.progs[lab] for lab in self.labels]
        self.alias, self.uniq, self.stage_keys = {}, {}, []
        for b, p in enumerate(progs):
            r, lab = self.recs[b], self.labels[b]
            keys = set()
            for nm, lv, _d, form in p.pt_refs:
                key = r.pts[nm]
                store = f"{lab}/{nm}" if (form == "at_use" or nm.startswith("pt_")) else nm
                self.alias.setdefault(lab, {})[store] = key
                self.uniq.setdefault((key, lv), self.pt_vals[key])
                keys.add((key, lv))
            self.stage_keys.append(keys)
        nq = _pc.chain_limbs(self.cfg["chain.preset"])
        pt_rows = self.pt_rows and max(self.pt_rows, nq - min(lv for _k, lv in self.uniq))
        specs = session_specs(self.linears().values(), self.model.rt.dims.t)
        writes = [(progs[b], j, r) for b in range(self.n_stages) for j, r in self.reg[b]]
        reg_levels = register_levels(self.n_regs, writes)
        pt_levels = stage_levels(self.stage_keys, nq)
        # the full-height slots a replay of the staging needs (+ 64): 64 alone ran out at 768 px (11-11)
        self.pt_spare = store_spare(self.stage_keys, nq, pt_levels)
        return dict(pool=self.pool, pt_slots=len(pt_levels) + self.pt_spare + 64, programs=progs, regs=self.n_regs, pt_rows=pt_rows,
                    stc_cache=self.stc_cache, pcmm=specs or None, pcmm_cf=self.pcmm_cf, check_meta=self.check_meta,
                    reg_levels=reg_levels, pt_levels=pt_levels)

    def setup(self, rows=None):
        """`prepare` (CPU) + the planned Session: routes, keys (+ the PCMM keys, then the secret is dropped), slab,
        registers, the PCMM service; every stage bound, stage 0's plaintexts staged (`run` stages each stage's before
        it issues). The PCMM weights: every linear resident, or pooled per stage (PC_PCMM_WEIGHTS_PER_PROGRAM=1: the
        session holds the most one stage finishes; `run` uploads each stage's before it issues)."""
        if not hasattr(self, "progs"):
            self.prepare(rows)
        t0 = time.time()
        kw = self.session_kwargs()
        progs, lins, specs = kw["programs"], self.linears(), kw["pcmm"] or []
        from perseus.impl.driver import plan_rot_scope
        self.rot_keys = plan_rot_scope(self.plan_dir, progs)
        self.sess = self.open_session(kw)
        self.store = self.sess.pt_store
        for b, p in enumerate(progs):
            p.bind(self.sess, self.store)
            p.alias_pts(self.alias.get(self.labels[b], {}))
        self.sess.complete_setup()
        self.pooled = bool(specs) and bool(self.sess.pcmm_stats()["pooled"])
        self.stage_lins = [set(p.pcmm_linears) for p in progs]
        self._planes = {}
        if not self.pooled:
            for lin in lins.values():
                self.sess.pcmm_upload(lin.name, np.ascontiguousarray(lin.Wd, dtype=np.int8).tobytes())
        self.sess.drop_secret()
        self.fhe = self.sess.fhe
        self._resident, self._wres = set(), set()
        self.stage_pts(0)
        self.stage_weights(0)
        n_refs = sum(len({(nm, lv) for nm, lv, _d, _f in p.pt_refs}) for p in progs)
        peak = max(range(self.n_stages), key=lambda b: len(self.stage_keys[b]))
        pool = f" (weights per stage: the most one finishes {max(map(len, self.stage_lins))})" if self.pooled else ""
        self.setup_s = time.time() - t0
        self.log(f"[planned] {len(progs)} programs, {self.n_regs} registers, {n_refs} plaintext keys = "
                 f"{len(self.uniq)} distinct (values, level), staged per stage (the most one reads: "
                 f"{len(self.stage_keys[peak])}, stage {peak}; {kw['pt_slots']} slots, {self.pt_spare} + 64 full-height), "
                 f"{len(specs)} PCMM linears"
                 f"{pool}, "
                 f"rotation keys {self.rot_keys[0]} (plan_rot_steps.json {self.rot_keys[1]}); record "
                 f"{self.record_s:.1f} s, compile {self.compile_s:.1f} s, session {self.sess.setup_seconds()[0]:.1f} s "
                 f"(keygen {self.sess.setup_seconds()[1]:.1f}), setup total {self.setup_s:.1f} s")
        self.log("[planned] " + self.sess.mem_report.replace("\n", " | "))
        return self

    def stage_pts(self, b):
        """Stage b's plaintexts (Llama decode's `_patch`): evict what it does not read, then stage what is not
        resident (the device encodes in stream order, after the stages issued before). Returns the number staged."""
        need = self.stage_keys[b]
        gone = [k for k in self._resident if k not in need]
        if gone:
            self.store.evict(gone)
        new = [(key, lv) for key, lv in need if (key, lv) not in self._resident]
        if new and os.environ.get("SANA_PT_PINNED", "") not in ("", "0"):
            # SANA_PT_PINNED=1: the session keeps each plaintext's values pinned (copied the first time its key is
            # staged) and encodes from there: no host copy and no lane wait per plaintext
            seen = self.__dict__.setdefault("_pt_pinned", set())
            self.store.stage_pinned([(key, lv, None if key in seen else self.uniq[(key, lv)]) for key, lv in new])
            seen.update(key for key, _lv in new)
            self._resident = need
            return len(new)
        items = [(key, lv, self.uniq[(key, lv)]) for key, lv in new]
        if items:
            self.store.stage(items)
        self._resident = need
        return len(items)

    def stage_weights(self, b):
        """Pooled PCMM weights: evict the linears stage b does not finish, then upload those it does that are not
        resident (synchronous on the compute lane: after the stages issued before). Returns the number uploaded."""
        if not self.pooled:
            return 0
        need = self.stage_lins[b]
        gone = sorted(self._wres - need)
        if gone:
            self.sess.pcmm_evict(gone)
        new = sorted(need - self._wres)
        pin = os.environ.get("SANA_PCMM_PINNED", "") not in ("", "0")
        for nm in new:
            if pin:     # SANA_PCMM_PINNED=1: the session keeps the planes pinned and uploads without a lane sync
                seen = self.__dict__.setdefault("_pinned", set())
                self.sess.pcmm_upload(nm, b"" if nm in seen else np.ascontiguousarray(
                    self.linears()[nm].Wd, dtype=np.int8).tobytes(), True)
                seen.add(nm)
            else:
                self.sess.pcmm_upload(nm, self.planes(nm))
        self._wres = set(need)
        return len(new)

    # ── the device run ──
    def run(self, on_block=None, vals=None, x0=None):
        """`PlannedPrefill.run` with each stage's plaintexts (and pooled PCMM weights) staged before it issues."""
        if x0 is None:
            vals = self.in_vals if vals is None else vals
            x0 = [self.fhe.encrypt(v, lv, dg) for v, (lv, dg) in zip(vals, self.in_metas[0])]
        times = [] if os.environ.get("PC_STAGE_TIMES", "") not in ("", "0") else None
        for b in range(self.n_stages):
            t0 = time.perf_counter()
            n_pts = self.stage_pts(b)
            t1 = time.perf_counter()
            n_w = self.stage_weights(b)
            t2 = time.perf_counter()
            src = x0 if b == 0 else [reg for _j, reg in self.reg[b - 1]]
            info = self.progs[self.labels[b]].run([src[j] for j in self.in_pos[b]], list(self.reg[b]))
            if times is not None:       # PC_STAGE_TIMES=1: the host's seconds per stage (staging, weights, issue)
                times.append((b, n_pts, n_w, t1 - t0, t2 - t1, time.perf_counter() - t2))
            if on_block is not None:
                on_block(b, info)
        if times is not None:
            self.stage_times = times
        return [reg for _j, reg in self.reg[self.n_stages - 1]]

    def input_values(self, x_in=None, txt_rows=None):
        """Stage 0's input slot vectors for another prompt of the recorded length (the model's packing); the model's
        own prompt checked once against the recorded inputs."""
        if not hasattr(self, "_vals_ok"):
            mine = self.model.input_values()
            ok = [np.allclose(a, b, atol=1e-9) for a, b in zip(mine, self.in_vals)]
            same = len(mine) == len(self.in_vals) and all(ok)
            if not same:
                raise RuntimeError("input_values: the packing does not reproduce the recorded stage-0 inputs")
            self._vals_ok = True
        return self.model.input_values(x_in, txt_rows)


def device_run(a, model, check, tag):
    """A runner's planned device run (its --plan, --graphs, --key-layout, --work-dir, --compile-only, --repeat), as
    examples/llama/run_prefill.py runs one: record + compile on the twin, the planned Session, `--repeat` timed
    forwards (each: stage 0's inputs encrypted, every stage issued, the device drained), then the checking forward
    (`check(b, slot vectors)` -> a line or ""). Returns {run, out (the last stage's slot vectors), walls, lines
    ({stage: check line}), bts, free (GB after the last forward)}, or None with --compile-only."""
    from perseus import _pc
    from perseus._paths import resolve
    from perseus.examples.bert.planned import plan_config
    plan = resolve(a.plan)
    keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
    work = a.work_dir or os.path.join(os.environ.get("PC_WORK_DIR", "/tmp"), f"{tag}_{os.getpid()}")
    run = SanaPlannedPrefill(model, plan, plan_config(plan, a.graphs), keys, work_dir=work,
                             log=lambda m: print(m, flush=True))
    t0 = time.time()
    if a.graphs and os.environ.get("PC_STC_CACHE_PER_PROGRAM", "") not in ("", "0") and not a.compile_only:
        # the session refuses it only at its open, after record + compile (~77 min at 768 px): fail before them.
        # A captured graph needs every StC-first key resident (~25 GB at 768 px) and every PCMM weight (~16.7 GB):
        # ~112 GB with the rest, past a 96 GB card (11-11)
        raise SystemExit("--graphs with PC_STC_CACHE_PER_PROGRAM=1: a captured graph needs every key resident "
                         "(the session refuses it at its open); set PC_STC_CACHE_PER_PROGRAM=0 if the memory fits")
    run.prepare()
    progs = [run.progs[lab] for lab in run.labels]
    bts = sum(p.n_bootstraps for p in progs) + sum(p.n_finishes[1] for p in progs)
    print(f"[{tag}] compiled {len(progs)} stages ({getattr(run, 'n_loaded', 0)} loaded from the plan): "
          f"record {run.record_s:.1f}s compile {run.compile_s:.1f}s; "
          f"{sum(p.n_bootstraps for p in progs)} refresh sites + {sum(p.n_finishes[1] for p in progs)} PCMM outputs = "
          f"{bts} bts, slots {max(p.mem[1] for p in progs)}, registers {run.n_regs}, distinct plaintexts "
          f"{len(run.pt_vals)}", flush=True)
    if a.compile_only:
        return None
    if getattr(a, "wait_free", False):                 # record + compile overlapped with another run on the GPU
        from perseus.examples.gpt2.run_decode import gpu_busy
        while gpu_busy(a.device):
            time.sleep(30)
        print(f"[{tag}] GPU {a.device} free, setup starts", flush=True)
    if a.graphs and os.environ.get("PC_PCMM_WEIGHTS_PER_PROGRAM", "") not in ("", "0"):
        # a captured PCMM finish bakes its weight buffer: with CUDA graphs every linear stays resident (the session
        # refuses pooled weights; ~16.7 GB of weights at 768 px instead of the most one stage finishes)
        os.environ["PC_PCMM_WEIGHTS_PER_PROGRAM"] = "0"
        print(f"[{tag}] --graphs: PCMM weights resident (PC_PCMM_WEIGHTS_PER_PROGRAM=0)", flush=True)
    run.setup()
    print(f"[{tag}] planned setup {time.time() - t0:.1f}s; device free {run.fhe.mem_info()[0] / 2 ** 30:.2f} GB",
          flush=True)
    vals = run.input_values()
    walls, graphs = [], bool(a.graphs)
    for k in range(max(1, a.repeat)):
        info = []
        t1 = time.time()
        try:
            run.run(vals=vals, on_block=lambda b, i: info.append(i))
            run.fhe.device_sync()
        except Exception as e:  # noqa: BLE001 -- a graph capture that does not fit: the same session, eager
            if not graphs:
                raise
            print(f"[{tag}] forward {k} with CUDA graphs failed ({type(e).__name__}: {e}); eager from here "
                  f"(set_graphs(False), the same session)", flush=True)
            run.sess.set_graphs(False)
            graphs = False
            info = []
            t1 = time.time()
            run.run(vals=vals, on_block=lambda b, i: info.append(i))
            run.fhe.device_sync()
        walls.append(time.time() - t1)
        print(f"[{tag}] forward {k}: wall {walls[-1]:.2f}s, host issue {sum(i['host_us'] for i in info) / 1e6:.2f}s, "
              f"served {''.join(i['served'][0] for i in info)}, free {run.fhe.mem_info()[0] / 2 ** 30:.2f} GB"
              + (f", graphs {dict(run.sess.planned_stats())}" if a.graphs else ""), flush=True)
    # the checking forward: the device synced after each stage (its seconds, the decryption excluded) and its outputs
    # checked before the register sets reuse them
    secs, lines, t = [], {}, [0.0]

    def on(b, _info):
        run.fhe.device_sync()
        secs.append(time.time() - t[0])
        line = check(b, run.decrypt([reg for _j, reg in run.reg[b]])).strip()
        if line:
            lines[b] = line
        t[0] = time.time()
    t[0] = time.time()
    out = run.decrypt(run.run(vals=vals, on_block=on))
    print(f"[{tag}] per stage (synced): {' '.join(f'{s:.2f}' for s in secs)}", flush=True)
    for b, line in sorted(lines.items()):
        print(f"[{tag}] stage {b}: {line}", flush=True)
    return {"run": run, "out": out, "walls": walls, "lines": lines, "bts": bts,
            "free": round(run.fhe.mem_info()[0] / 2 ** 30, 3)}


def summary(tag, res, err):
    """The run's SUMMARY line (examples/llama/run_prefill.py's): the median wall, the refreshes, the gate."""
    w = res["walls"]
    print(f"[{tag}] SUMMARY wall={sorted(w)[len(w) // 2]:.2f}s bts={res['bts']} {err} free={res['free']}GB "
          f"walls={[round(x, 2) for x in w]}", flush=True)
