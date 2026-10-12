"""Capture magnitudes from the numpy fake instead of decrypting on the device.

A capture records, per graph node, the op's structure (names, levels, pack tags, the refreshes the
runtime fired: what a plan binds to) and its output's magnitudes (what the planner prices a refresh
by: max |slot|, the mean / max deviation the offset transform sees, the max plaintext coefficient
EvalMod sees). The second half is a decrypt + FFT per node on the host, which dominates a capture of
a weight-heavy model (Llama-8B: hours, the GPU idle). This module splits the two:

  1. the device capture runs with magnitudes suppressed (``Context.magnitude_capture_suppressed``):
     exact structure, no decrypts;
  2. the fake runs the same kernels on the same input with a TAPE: per leaf op, its graph op type,
     its step path and the exact (noise-free) output statistics;
  3. ``fill_graph`` walks the device graph and the tape in order and writes the tape's statistics into
     the nodes. Refresh nodes (auto / deliberate bootstraps, hints) are the runtime's decisions, not
     ops of the kernels: they are skipped on both sides and inherit their input's statistics.

The walk checks every pair (op type and step path) and stops at the first divergence, so a tape that
does not describe the device's op stream is never merged silently.
"""
from __future__ import annotations

import functools
import json

import numpy as np

# the runtime's own nodes, not the kernels' ops: refreshes, level drops and level hints (value-preserving)
_REFRESH = ("auto_bootstrap", "deliberate_bootstrap", "hint", "fold_bootstrap", "level_reduce", "level_hint")


@functools.lru_cache(maxsize=4)
def _coeff_perm(n: int):
    """Slot j evaluates at zeta^(5^j mod 4n) = zeta^(4 m_j + 1): m_j per slot, and zeta^-k."""
    e = np.empty(n, dtype=np.int64)
    x = 1
    for j in range(n):
        e[j] = x
        x = (x * 5) % (4 * n)
    m = (e - 1) // 4
    k = np.arange(n)
    return m, np.exp(-2j * np.pi * k / (4 * n))


def node_stats(vec) -> dict:
    """The capture's per-node statistics (fideslib_wrapper.h: lane_split / slot_stats /
    coeff_stats) of a slot vector: the coefficients are the canonical embedding's inverse,
    u_k = 1/n sum_j z_j zeta^-(5^j k), polynomial = sum_k Re(u_k) X^k + Im(u_k) X^(k+n)."""
    z = np.asarray(vec, dtype=np.complex128)
    n = z.shape[0]
    m, tw = _coeff_perm(n)
    y = np.empty(n, dtype=np.complex128)
    y[m] = z
    u = np.fft.fft(y) * tw / n
    mag = np.maximum(np.abs(u.real), np.abs(u.imag))
    mean = float(z.real.mean())
    ac = mag.copy()
    ac[0] = abs(u[0].imag)
    return {"output_max_abs": float(np.abs(z).max()), "output_mean": mean,
            "output_max_dev": float(np.abs(z - mean).max()),
            "output_max_coeff": float(mag.max()), "output_max_coeff_ac": float(ac.max())}


def node_stats_batch(V) -> list:
    """node_stats of every row of V (B, n), vectorized (one multi-worker FFT)."""
    import os
    import scipy.fft
    Z = np.asarray(V, dtype=np.complex128)
    B, n = Z.shape
    m, tw = _coeff_perm(n)
    Y = np.empty_like(Z)
    Y[:, m] = Z
    U = scipy.fft.fft(Y, axis=1, workers=int(os.environ.get("MAGTAPE_THREADS", "8"))) * (tw / n)
    mag = np.maximum(np.abs(U.real), np.abs(U.imag))
    mean = Z.real.mean(axis=1)
    mx = mag.max(axis=1)
    ac = np.maximum(mag[:, 1:].max(axis=1), np.abs(U[:, 0].imag))
    amax = np.abs(Z).max(axis=1)
    dev = np.abs(Z - mean[:, None]).max(axis=1)
    return [{"output_max_abs": float(amax[i]), "output_mean": float(mean[i]), "output_max_dev": float(dev[i]),
             "output_max_coeff": float(mx[i]), "output_max_coeff_ac": float(ac[i])} for i in range(B)]


class _Pending:
    """A record's statistics, computed with its batch."""
    __slots__ = ("value",)

    def result(self):
        return self.value


class Tape:
    """Records (op_type, step path, stats) per leaf op of a fake session while `active`; MAGTAPE_BATCH > 1 batches
    the stats (a record holds a _Pending until flush)."""

    def __init__(self):
        import os
        self.active = False
        self.records = []
        self.steps = []
        self._batch = int(os.environ.get("MAGTAPE_BATCH", "1"))
        self._pend, self._vecs = [], []

    def path(self):
        return ".".join(self.steps)

    def flush(self):
        if self._pend:
            for p, st in zip(self._pend, node_stats_batch(np.stack(self._vecs))):
                p.value = st
            self._pend, self._vecs = [], []

    def add(self, op_type, ct):
        if self.active:
            if self._batch <= 1:          # default: stats per record
                self.records.append((op_type, self.path(), node_stats(ct.vec)))
                return ct
            p = _Pending()
            self._pend.append(p)
            self._vecs.append(np.array(ct.vec, dtype=np.complex128))
            self.records.append((op_type, self.path(), p))
            if len(self._pend) >= self._batch:
                self.flush()
        return ct


def attach(fhe, inf) -> Tape:
    """Wrap a fake session (fake.make_fake) so its leaf ops write to a Tape. The graph op types:
    add / sub_ct / mult / square / negate / add_inplace / rotate / conjugate / mult_i; a plaintext
    add or a scalar add/sub records as add, a plaintext / scalar / constant product as mult."""
    import contextlib
    tape = Tape()

    def wrap(obj, name, op_type):
        f = getattr(obj, name)
        def g(*a, **k):
            out = f(*a, **k)
            ot = op_type(a) if callable(op_type) else op_type
            return tape.add(ot, out)
        setattr(obj, name, g)

    from .fake import FakeCt
    wrap(fhe, "add", "add")
    wrap(fhe, "sub", lambda a: "sub_ct" if isinstance(a[1], FakeCt) else "add")
    wrap(fhe, "mult", "mult")
    wrap(fhe, "square", "square")
    wrap(fhe, "negate", "negate")
    wrap(fhe, "rotate", "rotate")
    wrap(fhe, "conjugate", "conjugate")
    wrap(fhe, "mult_i", "mult_i")
    f_ip = fhe.inplace_add

    def inplace_add(a, b):
        f_ip(a, b)
        tape.add("add_inplace", a)
    fhe.inplace_add = inplace_add

    from .layout import ladder_chunks, ladder_radix_bits
    f_rs = fhe.rotate_sum

    def rotate_and_sum(a, start, stop):
        # the runtime's ladder records add (first rung) then inplace_add (FheOps' fallback stream)
        gap, sign = abs(int(start)), (1 if start > 0 else -1)
        if gap >= stop:
            return fhe.add(a, 0.0)
        bits = ladder_radix_bits()
        if bits >= 2:     # src/bindings/ops.cu rotate_and_sum: radix groups record one rotate_sum each
            x = None
            for g in ladder_chunks(start, stop, bits):
                if len(g) > 1 and fhe.has_rot_keys(g):
                    x = tape.add("rotate_sum", f_rs(a if x is None else x, g))
                    continue
                for k in (g if len(g) == 1 else [g[(1 << i) - 1] for i in range(len(g).bit_length())]):
                    if x is None:
                        x = fhe.add(a, fhe.rotate(a, k))
                    else:
                        fhe.inplace_add(x, fhe.rotate(x, k))
            return x
        x = fhe.add(a, fhe.rotate(a, sign * gap))
        gap *= 2
        while gap < stop:
            fhe.inplace_add(x, fhe.rotate(x, sign * gap))
            gap *= 2
        return x
    fhe.rotate_and_sum = rotate_and_sum
    for name in ("mult_pt", "mult_slot_pt", "mult_cached", "mult_const"):
        wrap(inf, name, "mult")
    for name in ("add_pt", "add_slot_pt", "add_cached"):
        wrap(inf, name, "add")
    wrap(inf, "mult_i", "mult_i")

    @contextlib.contextmanager
    def step(label):
        tape.steps.append(str(label))
        try:
            yield
        finally:
            tape.steps.pop()
    inf.step = step
    return tape


def fill_graph(graph_path: str, records, start: int = 0, out_path: str | None = None):
    """Write the tape's statistics, from records[start:], into the nodes of one capture graph.
    Returns the index after the last record consumed (the next block's start)."""
    with open(graph_path) as f:
        g = json.load(f)
    nodes = g["nodes"]
    by_out = {}
    i = start
    for n in nodes:
        op = n["op_type"]
        if op in _REFRESH:
            src = by_out.get(n["inputs"][0]) if n["inputs"] else None
            if src:
                n.update(src)
            by_out[n["output"]] = src
            continue
        if i >= len(records):
            raise ValueError(f"{graph_path}: tape exhausted at node {n['output']} ({op}, {n.get('step')})")
        r_op, r_step, st = records[i]
        if hasattr(st, "result"):
            st = st.result()
        if r_op != op or r_step != n.get("step", r_step):
            raise ValueError(f"{graph_path}: node {n['output']} is {op} @ {n.get('step')} but tape[{i}] is "
                             f"{r_op} @ {r_step}")
        n.update(st)
        by_out[n["output"]] = st
        i += 1
    with open(out_path or graph_path, "w") as f:
        json.dump(g, f)
    return i


def tape_stages(model, tape, pos: int):
    """Route `tape` per stage of position `pos`: wraps model.enter_stage / exit_stage (perseus.impl.driver
    ImplModel) so each stage's records land in their own list. Returns {stage: records}, filled as the model runs."""
    per_stage = {}
    enter, exit_ = model.enter_stage, model.exit_stage

    def enter_stage(b, p):
        if p == pos:
            tape.records = []
            tape.active = True
        return enter(b, p)

    def exit_stage(b, cap):
        if tape.active:
            tape.flush()
            per_stage[b] = tape.records
            tape.active = False
        return exit_(b, cap)
    model.enter_stage, model.exit_stage = enter_stage, exit_stage
    return per_stage


def fill_dir(graph_dir: str, tapes: dict, suffix: str = "", log=print):
    """fill_graph for every captured block_<b>/graph.json with its stage's tape (suffix: write graph.json<suffix>
    instead of in place)."""
    import os
    done = []
    for b, rec in sorted(tapes.items()):
        g = os.path.join(graph_dir, f"block_{b}", "graph.json")
        if not os.path.exists(g):
            log(f"[mags] block {b}: no graph, skipped")
            continue
        used = fill_graph(g, rec, 0, g + suffix if suffix else None)
        log(f"[mags] block {b}: {used}/{len(rec)} records merged")
        done.append(b)
    return done


def compare(ref_graph: str, filled_graph: str):
    """Per statistic: (median, p99, max) relative difference over the kernel nodes both carry."""
    a = json.load(open(ref_graph))["nodes"]
    b = json.load(open(filled_graph))["nodes"]
    out = {}
    for k in ("output_max_abs", "output_max_coeff", "output_max_coeff_ac", "output_max_dev"):
        r = np.asarray([abs(x[k] - y[k]) / max(abs(x[k]), 1e-12) for x, y in zip(a, b)
                        if k in x and k in y and x["op_type"] not in _REFRESH])
        if r.size:
            out[k] = (float(np.median(r)), float(np.quantile(r, 0.99)), float(r.max()))
    return out


def plan_rot_steps(plan_dir: str):
    """The rotation steps the captured graph of `plan_dir` (PLAN_CMD.txt graph=) issues: a planned run repeats that op
    stream, so these are all the keys it needs (None: no graph found)."""
    import glob
    import os
    try:
        line = next(l for l in open(os.path.join(plan_dir, "PLAN_CMD.txt")) if l.startswith("graph="))
        g = line[len("graph="):].split(" (md5")[0].strip()
    except (OSError, StopIteration):      # the pipelines name the graph like the plan
        g = os.path.join("graphs", os.path.basename(os.path.normpath(plan_dir)))
    paths = glob.glob(os.path.join(g, "block_*", "graph.json"))
    if not paths:
        return None
    used = set()
    for p in paths:
        for n in json.load(open(p))["nodes"]:
            for x in n["inputs"]:
                if x.startswith("rot("):
                    used |= {int(v) for v in x[4:-1].split(",")}
    return used


def save_rot_steps(graph_dir: str, steps):
    """The capturing session's rotation keys, so the fake takes the same ladder forms (rot_steps.json)."""
    import os
    with open(os.path.join(graph_dir, "rot_steps.json"), "w") as f:
        json.dump(sorted(int(s) for s in steps), f)


def load_rot_steps(graph_dir: str, N: int):
    """save_rot_steps' keys in every form FakeFhe matches (None: not saved)."""
    import os
    path = os.path.join(graph_dir, "rot_steps.json")
    if not os.path.exists(path):
        return None
    steps = {int(s) % N for s in json.load(open(path))}
    return steps | {s - N for s in steps}
