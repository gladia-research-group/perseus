#!/usr/bin/env python3
"""Gate for the StC-first dense bootstrap (FIDESLIB_BTS_STC_FIRST=2, dense route) with the real-payload route.

StC-first runs SlotsToCoeffs on the input, before the raise, so EvalMod sees the SLOT values, not the coefficients:
its range is set by the correction factor chosen from the slot magnitude (CF ~ 10 + log2 max|x|, what the planner's
PLAN_SLIM_CUBIC prices). In one session, against the shipped order at its best CF:
  * precision on token lanes of max 1/4/16/64 and on outlier channels (the payloads where a coefficient-sized CF
    fails), on the real route, at the slot-magnitude CF,
  * the landing (StC-first lands higher) and the deep-input fallback (an input without room for the StC levels takes
    the shipped order),
  * the 512-slot route keeps its raise drops, and the real route's wall time.

    python scripts/dev/bts_stc_first_gate.py [--iters 15] [--device 4]
"""
import argparse
import math
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--device", type=int, default=3)
    a = ap.parse_args()

    from examples.gpt2_from_primitives import env
    sess = env.open_session(a.device, complex_payload=True, FIDESLIB_BTS_SHIFT="1", FIDESLIB_BTS_REAL="1",
                            FIDESLIB_BTS_STC_FIRST="2", FIDESLIB_BTS_STC_FIRST_ROUTES="dense",
                            FIDESLIB_BTS_RAISE_DROPS="1,3,5")
    from perseus import _core
    from perseus.impl import bootstrap as pyb
    bts = _core.bts
    inf, fhe = sess.inf, sess.inf.fhe
    S = inf.slots

    def run(base, stc_first, real=True, cf=-1, slots=S, drop=0):
        c = bts.clone(base)
        if slots < S:
            bts.set_slots(c, slots)
        d = bts.device(fhe, c)
        bts.set_real_payload(fhe, real)
        bts.set_correction_factor(fhe, cf)
        bts.set_raise_drop(fhe, drop)
        try:
            bts.sync()
            t = time.perf_counter()
            st = bts.begin(d, slots, False, stc_first)
            if st.stc_first:
                bts.stc_first_input(d, st)
            bts.mod_raise(d, st)
            bts.fold(d, st)
            for k in range(st.n_cts):
                bts.cts_stage(d, st, k)
            bts.eval_mod(d, st)
            if st.stc_first:
                bts.stc_first_output(d, st)
            else:
                bts.stc_enter(d, st)
                for k in range(st.n_stc):
                    bts.stc_stage(d, st, k)
                bts.finish(d, st)
            bts.sync()
            ms = (time.perf_counter() - t) * 1e3
        finally:
            bts.set_real_payload(fhe, False)
            bts.set_correction_factor(fhe, -1)
            bts.set_raise_drop(fhe, 0)
        return c, d.level + 1, bool(st.stc_first), ms

    def at(ct, level):  # scalar products down to an effective level (fhe.mult leaves its rescale pending)
        with fhe.suppress_auto_bts():           # no reactive refresh on the way down
            while ct.level < level - 2:
                ct = fhe.mult(ct, 1.0)
        return ct

    bits = lambda e, m: -math.log2(max(e, 1e-300) / m)
    rng = np.random.default_rng(5)
    lane = rng.uniform(-1, 1, 768)
    outl = rng.normal(0, 0.3, 768)
    outl[[138, 447]] = [50.0, -30.0]
    ok = True
    for name, v in [(f"token lane max {k}", lane * k) for k in (1, 4, 16, 64)] + [("outlier channels 50/-30", outl)]:
        base = at(_core.encode_token_input(inf, v), 46)
        want = np.real(np.asarray(_core.decrypt_slots_complex(inf, base)))
        m = float(np.max(np.abs(want)))
        err = lambda c: float(np.max(np.abs(np.real(np.asarray(_core.decrypt_slots_complex(inf, c))) - want)))
        shipped = max(bits(err(run(base, False, cf=cf)[0]), m) for cf in range(2, 9))
        cf = min(20, round(10 + math.log2(m)))
        c, lim, stcf, _ = run(base, True, cf=cf)
        got = bits(err(c), m)
        print(f"[stc_first] {name:24s} shipped (best CF) {shipped:5.1f} bits | StC-first CF {cf:2d} {got:5.1f} bits "
              f"at {lim} limbs", flush=True)
        ok &= stcf and got >= min(shipped, 14.0) - 0.5

    deep = at(_core.encode_token_input(inf, lane), 50)
    c, lim, stcf, _ = run(deep, True, cf=4)
    e = bits(float(np.max(np.abs(np.real(np.asarray(_core.decrypt_slots_complex(inf, c)))[:768]
                                 - np.real(np.asarray(_core.decrypt_slots_complex(inf, deep)))[:768]))), 1.0)
    print(f"[stc_first] input at level 50 (no room for the StC levels): stc_first={int(stcf)}, {e:.1f} bits at {lim} "
          "limbs (the shipped order)", flush=True)
    ok &= not stcf and e > 8

    x512 = np.tile(rng.uniform(-1, 1, 512), S // 512)
    b512 = at(inf.add_pt(_core.encode_token_input(inf, np.zeros(768)), x512), 46)
    w512 = np.real(np.asarray(_core.decrypt_slots_complex(inf, b512)))[:512]
    for drop in (0, 1, 3, 5):
        c, lim, _, _ = run(b512, True, real=False, slots=512, drop=drop)
        e = float(np.max(np.abs(np.real(np.asarray(_core.decrypt_slots_complex(inf, c)))[:512] - w512)))
        print(f"[stc_first] 512-slot route, raise drop {drop}: {bits(e, 1.0):5.1f} bits at {lim} limbs", flush=True)
        ok &= bits(e, 1.0) >= 12

    real_ct = at(_core.encode_token_input(inf, lane), 46)
    ts, tf = [], []
    for i in range(a.iters + 2):
        for stc_first in ((False, True) if i % 2 == 0 else (True, False)):
            ms = run(real_ct, stc_first, cf=10 if stc_first else 6)[3]
            if i >= 2:
                (tf if stc_first else ts).append(ms)
    ms, mf = statistics.median(ts), statistics.median(tf)
    print(f"[stc_first] real route wall median (n={len(ts)}): shipped {ms:.2f} ms, StC-first {mf:.2f} ms "
          f"({100 * (mf - ms) / ms:+.1f}%)", flush=True)
    ok &= mf < ms
    print("[stc_first] PASS" if ok else "[stc_first] FAIL", flush=True)
    sess.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
