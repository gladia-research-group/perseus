"""Per-primitive wall-time accounting: `TimedOps` wraps an FheOps and sums the time spent in
each leaf call (mult_pt counts the plaintext encode + upload + product; bootstraps are
separate). GPU work is stream-ordered, so a call's time includes whatever it had to wait for;
the split is still the right first look at where a token goes."""
from __future__ import annotations

import collections
import functools
import time

from .ops import FheOps

_TIMED = ("add", "sub", "mult", "square", "negate", "rotate", "conjugate", "inplace_add", "copy",
          "mult_pt", "add_pt", "mult_pt_many", "rotate_and_sum", "rotate_many", "mult_add_many",
          "bootstrap", "bootstrap_hint", "realize", "encode_token",
          "decrypt_slots", "decode_linear_output", "decode_token", "load_pts", "evict_pts")


class TimedOps(FheOps):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.t = collections.defaultdict(float)
        self.n = collections.defaultdict(int)
        self.t_step = collections.defaultdict(float)
        self.n_step = collections.defaultdict(int)
        self.t_sp = collections.defaultdict(float)     # (step, primitive) -> time
        self.n_sp = collections.defaultdict(int)
        self.step_label = None

    def reset(self):
        self.t.clear(); self.n.clear(); self.t_step.clear(); self.n_step.clear()
        self.t_sp.clear(); self.n_sp.clear()

    def report_step_split(self, step, tokens=1, top=10):
        """The primitives inside one step label (e.g. "cutmax"): time and calls per token."""
        rows = sorted(((k[1], v) for k, v in self.t_sp.items() if k[0] == step), key=lambda kv: -kv[1])[:top]
        return " | ".join(f"{k} {v / tokens:.2f}s n={self.n_sp[(step, k)] // tokens}" for k, v in rows)

    def report_steps(self, tokens=1):
        tot = sum(self.t_step.values())
        rows = sorted(self.t_step.items(), key=lambda kv: -kv[1])
        lines = [f"{k:16s} {v / tokens:8.2f}s/tok {100 * v / max(tot, 1e-9):5.1f}%  ops/tok={self.n_step[k] // tokens}"
                 for k, v in rows]
        return "\n".join(lines) + f"\ntotal {tot / tokens:.1f}s/tok"

    def report(self, top=12):
        tot = sum(self.t.values())
        rows = sorted(self.t.items(), key=lambda kv: -kv[1])[:top]
        lines = [f"{k:14s} {v:8.1f}s {100 * v / max(tot, 1e-9):5.1f}%  n={self.n[k]}" for k, v in rows]
        return "\n".join(lines) + f"\ntotal {tot:.1f}s"


def _wrap(name):
    base = getattr(FheOps, name)

    @functools.wraps(base)
    def f(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return base(self, *a, **k)
        finally:
            dt = time.perf_counter() - t0
            self.t[name] += dt
            self.n[name] += 1
            lab = self.step_label or "?"
            self.t_step[lab] += dt
            self.n_step[lab] += 1
            self.t_sp[(lab, name)] += dt
            self.n_sp[(lab, name)] += 1
    return f


for _name in _TIMED:
    setattr(TimedOps, _name, _wrap(_name))
