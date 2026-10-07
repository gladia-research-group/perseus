"""The GPU bootstrap orchestrated from Python, one call per stage.

The stages are FIDESlib's (``CKKS/BootstrapStages.cuh``, bound as ``perseus._core.bts``); this module only
decides their order, which is the production path of FIDESlib's own ``Bootstrap()``:

    begin -> [stc_first_input] -> mod_raise -> fold -> cts_stage(0..n_cts-1) -> eval_mod ->
        ( stc_first_output | stc_enter -> stc_stage(0..n_stc-1) -> finish )

``bootstrap(ct, slots)`` is bit-identical to ``_core.bts.cpp_bootstrap`` (scripts/dev/bts_python_gate.py checks
it). ``install()`` routes every bootstrap of the runtime through this module; ``trace`` / ``probe`` give a
per-stage view (wall time with a device sync per stage, level/degree, or an arbitrary callback).
"""
from __future__ import annotations

import atexit
import os
import sys
import time

from perseus import _core

bts = _core.bts


_spru = {"fhe": None}


def bootstrap(ct, slots: int, prescaled: bool = False, *, trace: list | None = None, probe=None):
    """In place on a ``bts.DeviceCt``. ``trace``: append (stage, limbs, deg, ms) per stage (syncs the device,
    so it perturbs the timing it reports less than a profiler but more than nothing). ``probe(stage, ct)``:
    called after every stage."""
    timed = trace is not None
    t0 = [time.perf_counter()]

    def mark(name):
        if timed:
            bts.sync()
            t = time.perf_counter()
            trace.append((name, ct.level + 1, ct.noise_level, (t - t0[0]) * 1e3))
            t0[0] = t
        if probe is not None:
            probe(name, ct)

    if timed:
        bts.sync()
        t0[0] = time.perf_counter()
    st = bts.begin(ct, slots, prescaled, True)
    if slots == 1 and _spru["fhe"] is not None:  # SPRU for the single-value route (CKKS/Spru.cuh)
        bts.spru(_spru["fhe"], ct, st.correction)
        mark("spru")
        return ct
    if st.stc_first:
        bts.stc_first_input(ct, st)
        mark("stc_first_input")
    bts.mod_raise(ct, st)
    mark("mod_raise")
    bts.fold(ct, st)
    mark("fold")
    for k in range(st.n_cts):
        bts.cts_stage(ct, st, k)
        mark(f"cts_{k}")
    bts.eval_mod(ct, st)
    mark("eval_mod")
    if st.stc_first:
        bts.stc_first_output(ct, st)
        mark("stc_first_output")
        return ct
    bts.stc_enter(ct, st)
    for k in range(st.n_stc):
        bts.stc_stage(ct, st, k)
        mark(f"stc_{k}")
    bts.finish(ct, st)
    mark("finish")
    return ct


_installed = {"n": 0}


def install(enable: bool = True, *, fhe=None, spru_h: int = 0):
    """Route every FIDESlib Bootstrap() of the runtime (planned refreshes, auto bootstraps, folds) through
    ``bootstrap``. ``enable=False`` restores FIDESlib's own. ``spru_h > 0`` (with ``fhe``) bootstraps the s = 1 route
    with SPRU (h blocks; the session must have been built with FIDESLIB_SPRU=h for its rotation keys)."""
    if not enable:
        bts.set_override(None)
        _spru["fhe"] = None
        return
    if spru_h > 0:
        bts.spru_setup(fhe, int(spru_h))
        _spru["fhe"] = fhe

    prof = os.environ.get("PERSEUS_BTS_PROFILE", "0") not in ("", "0")
    stats = {}

    def _hook(ct, slots, prescaled):
        _installed["n"] += 1
        if not prof:
            bootstrap(ct, int(slots), bool(prescaled))
            return
        bts.sync()
        t = time.perf_counter()
        lin = ct.level + 1
        bootstrap(ct, int(slots), bool(prescaled))
        bts.sync()
        key = (int(slots), lin, ct.level + 1, ct.noise_level)
        n, ms = stats.get(key, (0, 0.0))
        stats[key] = (n + 1, ms + (time.perf_counter() - t) * 1e3)

    if prof:
        def _report():
            for (sl, li, lo, dg), (n, ms) in sorted(stats.items()):
                print(f"[bts_prof] slots={sl} in={li} out={lo} deg={dg} n={n} mean={ms / n:.2f}ms total={ms:.0f}ms",
                      file=sys.stderr)
            if _spru["fhe"] is not None:
                t = bts.spru_times()
                print("[bts_prof] spru phases " + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                                         for k, v in t.items()), file=sys.stderr)
        atexit.register(_report)

    bts.set_override(_hook)
    # release the hook (a Python callable held by C++) before interpreter finalization
    atexit.register(bts.set_override, None)


def calls() -> int:
    """Bootstraps routed through Python since install()."""
    return _installed["n"]
