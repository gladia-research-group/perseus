"""Rotation keys of a stage: what a module's code rotates by, found by running it on the plaintext mirror, and the
key set a module's own choices (baby-step giant-step splits) should prefer.

A planned session generates keys for exactly the rotations its compiled programs issue (driver.plan_rot_scope); the
capture twin and an eager session need the set before the code runs. `stage_rotations(fn, inputs)` runs a stage on
`RotationRecorder` (NumpyOps that notes every step, rotate-and-sum ladders rung by rung as FheOps issues them) and
returns the steps. `prefer(steps)` makes the steps a model already holds keys for the cheaper choice wherever a
module picks among equivalent rotation sets (linalg's face moves, fourier's BSGS splits): a step outside them costs
a new key (memory, key generation), not run time, so it only breaks ties between splits of equal cost."""
from __future__ import annotations

import contextlib
import contextvars

import numpy as np

from .ops import NumpyOps

_PREFERRED: contextvars.ContextVar = contextvars.ContextVar("perseus_impl_preferred_keys", default=frozenset())

# The price of one rotation key the model does not hold already, in rotations (fourier's COST_ROT units): a tie
# breaker between BSGS splits, below any real difference in rotations.
COST_NEW_KEY = 0.25


@contextlib.contextmanager
def prefer(steps, slots: int = 1 << 15):
    """Within the block, modules prefer rotation steps in `steps` (mod `slots`, both signs)."""
    s = {int(k) % slots for k in steps}
    tok = _PREFERRED.set(frozenset(s | {k - slots for k in s}))
    try:
        yield
    finally:
        _PREFERRED.reset(tok)


def preferred() -> frozenset:
    """The steps `prefer` set (empty: no preference)."""
    return _PREFERRED.get()


def new_keys(steps, slots: int = 1 << 15) -> int:
    """How many of `steps` (non-zero) fall outside the preferred set; 0 when none is set."""
    pref = _PREFERRED.get()
    if not pref:
        return 0
    return sum(1 for k in {int(k) % slots for k in steps if int(k) % slots} if k not in pref and k - slots not in pref)


class RotationRecorder(NumpyOps):
    """The plaintext mirror noting every rotation step (FheOps' forms: hoisted rotations one step each, a
    rotate-and-sum ladder one step per rung)."""

    def __init__(self, slots: int = 1 << 15):
        self.slots = slots
        self.steps: set[int] = set()
        self.conj = False

    def _note(self, k):
        k = int(k) % self.slots
        if k:
            self.steps.add(k)

    def rotate(self, a, k):
        self._note(k)
        return super().rotate(a, k)

    def rotate_many(self, a, steps):
        for k in steps:
            self._note(k)
        return super().rotate_many(a, steps)

    def rotate_and_sum(self, a, start, stop):
        g = abs(int(start))
        while g < stop:
            self._note(g if start > 0 else -g)
            g *= 2
        return super().rotate_and_sum(a, start, stop)

    def conjugate(self, a):
        self.conj = True
        return super().conjugate(a)


def stage_rotations(fn, inputs, slots: int = 1 << 15) -> list[int]:
    """The rotation steps (mod `slots`, sorted) `fn(ops, xs)` issues on the slot vectors `inputs` (any values of the
    right shape: the steps of a fixed schedule do not depend on the data)."""
    rec = RotationRecorder(slots)
    fn(rec, [np.array(x) for x in inputs])
    return sorted(rec.steps)
