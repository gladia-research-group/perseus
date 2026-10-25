"""Capture magnitudes from the numpy fake instead of decrypting on the device.

A capture records per node the op's structure (what a plan binds to) and its output magnitudes (what the planner
prices refreshes by). The device capture runs with magnitudes suppressed (exact structure, no decrypts); the fake
runs the same kernels on the same input with a TAPE of exact per-op statistics; ``fill_graph`` walks graph and tape
in order and writes the statistics into the nodes (refresh nodes skipped on both sides, inheriting their input's).
Every pair is checked (op type, step path): a tape that does not describe the device's op stream stops the walk.
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


def _runs(steps):
    """[(step, first index, end index)] of the consecutive runs of equal steps."""
    out = []
    for i, st in enumerate(steps):
        if out and out[-1][0] == st:
            out[-1][2] = i + 1
        else:
            out.append([st, i, i + 1])
    return out


def fill_graph(graph_path: str, records, start: int = 0, out_path: str | None = None,
               keep_differing: bool = False):
    """Write the tape's statistics, from records[start:], into the nodes of one capture graph.
    Returns the index after the last record consumed (the next block's start).

    `keep_differing`: a step whose op list differs from the tape's (a later decode position pushes its K / V through
    another op sequence) keeps the capture's statistics instead of failing; every other step is filled. An in-place op
    is the same kernel on the same operands as its out-of-place twin (a later position accumulates in place)."""
    with open(graph_path) as f:
        g = json.load(f)
    nodes = g["nodes"]
    kernel = [n for n in nodes if n["op_type"] not in _REFRESH]
    recs = records[start:start + len(kernel)]
    if len(recs) < len(kernel) and not keep_differing:
        raise ValueError(f"{graph_path}: tape exhausted at node {kernel[len(recs)]['output']}")
    stats: dict[str, dict] = {}
    g_runs, r_runs = _runs([n.get("step") for n in kernel]), _runs([r[1] for r in records[start:]])
    i = start
    for k, (st, a, b) in enumerate(g_runs):
        if k >= len(r_runs) or r_runs[k][0] != st:
            raise ValueError(f"{graph_path}: step {st} at node {kernel[a]['output']} has no tape run "
                             f"(tape: {r_runs[k][0] if k < len(r_runs) else 'exhausted'})")
        ra, rb = r_runs[k][1] + start, r_runs[k][2] + start
        ops = [n["op_type"].removesuffix("_inplace") for n in kernel[a:b]]
        r_ops = [records[j][0].removesuffix("_inplace") for j in range(ra, rb)]
        if ops != r_ops:
            if keep_differing:
                print(f"[mags] {graph_path}: step {st} keeps the capture's statistics ({len(ops)} nodes, "
                      f"{len(r_ops)} taped ops)", flush=True)
            else:
                j = next((j for j, (x, y) in enumerate(zip(ops, r_ops)) if x != y), min(len(ops), len(r_ops)))
                raise ValueError(f"{graph_path}: step {st}: node {kernel[min(a + j, b - 1)]['output']} is "
                                 f"{ops[j] if j < len(ops) else 'past the step'} but the tape has "
                                 f"{r_ops[j] if j < len(r_ops) else 'nothing'}")
            i = rb
            continue
        for n, j in zip(kernel[a:b], range(ra, rb)):
            st_j = records[j][2]
            stats[n["output"]] = st_j.result() if hasattr(st_j, "result") else st_j
        i = rb
    by_out = {}
    for n in nodes:
        if n["op_type"] in _REFRESH:
            src = by_out.get(n["inputs"][0]) if n["inputs"] else None
            if src:
                n.update(src)
            by_out[n["output"]] = src
            continue
        src = stats.get(n["output"])
        if src is not None:
            n.update(src)
        by_out[n["output"]] = src
    with open(out_path or graph_path, "w") as f:
        json.dump(g, f)
    return i


def _stats(st):
    return st.result() if hasattr(st, "result") else st


def fold_visits(records, repeated, skip, visits: int):
    """Records of a stage whose repeated steps (`repeated(path)`) ran `visits` times -> one visit's records, each
    record's output_max_* the max over the visits; records under `skip(path)` (the visit accumulators) dropped."""
    recs = [(o, p, _stats(st)) for o, p, st in records if not skip(p)]
    out, i = [], 0
    while i < len(recs):
        if not repeated(recs[i][1]):
            out.append(recs[i])
            i += 1
            continue
        j = i
        while j < len(recs) and recs[j][1] == recs[i][1]:
            j += 1
        run = recs[i:j]
        n = len(run) // visits
        if n * visits != len(run) or any(run[t][0] != run[t % n][0] for t in range(len(run))):
            raise ValueError(f"fold_visits: {len(run)} records at {run[0][1]} are not {visits} equal visits")
        for t in range(n):
            st = dict(run[t][2])
            for v in range(t + n, len(run), n):
                for key, val in run[v][2].items():
                    if key.startswith("output_max"):
                        st[key] = max(st[key], val)
            out.append((run[t][0], run[t][1], st))
        i = j
    return out


def merge_max(tapes):
    """Element-wise max of the output_max_* statistics of equally shaped record lists."""
    out = []
    for recs in zip(*tapes):
        st = dict(_stats(recs[0][2]))
        for r in recs[1:]:
            if r[0] != recs[0][0] or r[1] != recs[0][1]:
                raise ValueError(f"merge_max: {r[0]} @ {r[1]} vs {recs[0][0]} @ {recs[0][1]}")
            for key, val in _stats(r[2]).items():
                if key.startswith("output_max"):
                    st[key] = max(st[key], val)
        out.append((recs[0][0], recs[0][1], st))
    return out


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


def fill_dir(graph_dir: str, tapes: dict, suffix: str = "", log=print, keep_differing: bool = False):
    """fill_graph for every captured block_<b>/graph.json with its stage's tape (suffix: write graph.json<suffix>
    instead of in place)."""
    import os
    done = []
    for b, rec in sorted(tapes.items()):
        g = os.path.join(graph_dir, f"block_{b}", "graph.json")
        if not os.path.exists(g):
            log(f"[mags] block {b}: no graph, skipped")
            continue
        used = fill_graph(g, rec, 0, g + suffix if suffix else None, keep_differing)
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
    """Rotation steps issued by the captured graph of `plan_dir` (None: no graph found). The plan's own
    plan_rot_steps.json (write_plan_rot_steps) stands in for the graph, so a shipped plan needs no capture."""
    import glob
    import os
    f = os.path.join(plan_dir, "plan_rot_steps.json")
    if os.path.exists(f):
        return {int(s) for s in json.load(open(f))}
    try:
        line = next(l for l in open(os.path.join(plan_dir, "PLAN_CMD.txt")) if l.startswith("graph="))
        g = line[len("graph="):].split(" (md5")[0].strip()
    except (OSError, StopIteration):      # graph named like the plan
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


def write_plan_rot_steps(plan_dir: str):
    """plan_rot_steps(plan_dir) from its graph -> plan_dir/plan_rot_steps.json (None: no graph found)."""
    import os
    steps = plan_rot_steps(plan_dir)
    if steps is not None:
        with open(os.path.join(plan_dir, "plan_rot_steps.json"), "w") as f:
            json.dump(sorted(steps), f)
    return steps


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
