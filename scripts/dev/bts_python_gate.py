#!/usr/bin/env python3
"""Gate for perseus/impl/bootstrap.py: bit-identical to FIDESlib's Bootstrap() on every route, and its wall cost.

    python scripts/dev/bts_python_gate.py [--iters 10] [--routes 32768,512,1] [--depth 3]

Per route: the same input is bootstrapped by (a) FIDESlib's Bootstrap(), (b) the stage sequence driven from C++,
(c) the stage sequence driven from Python; (b) and (c) must match (a) residue for residue. Then alternating timed
runs of (a) and (c), a per-stage trace of (c), and the decoded error of (c) against the plaintext.
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--routes", default="32768,512,1")
    ap.add_argument("--depth", type=int, default=3, help="scalar products applied before bootstrapping")
    ap.add_argument("--device", type=int, default=3)
    a = ap.parse_args()

    from examples.gpt2_from_primitives import env
    sess = env.open_session(a.device, complex_payload=True)
    from perseus import _core
    from perseus.impl import bootstrap as pyb
    bts = _core.bts
    inf, fhe = sess.inf, sess.inf.fhe

    rng = np.random.default_rng(7)
    x = rng.uniform(-1, 1, 768)
    base = _core.encode_token_input(inf, x)
    for _ in range(a.depth):
        base = fhe.mult(base, 0.999)
    print(f"[gate] input {base!r}", flush=True)

    ok = True
    for slots in [int(s) for s in a.routes.split(",") if s]:
        def fresh():
            c = bts.clone(base)
            if slots < 32768:
                bts.set_slots(c, slots)
            return c
        A, B, C = fresh(), fresh(), fresh()
        dA, dB, dC = bts.device(fhe, A), bts.device(fhe, B), bts.device(fhe, C)
        bts.cpp_bootstrap(dA, slots)
        bts.cpp_bootstrap_staged(dB, slots)
        pyb.bootstrap(dC, slots)
        mB, mC = bts.mismatches(dA, dB), bts.mismatches(dA, dC)
        st = bts.begin(bts.device(fhe, fresh()), slots)
        print(f"[gate] slots={slots} {st!r}: out {dA!r} | staged-C++ mismatches {mB} | python mismatches {mC}",
              flush=True)
        ok &= (mB == 0 and mC == 0)

        # wall: alternate C++ and Python, sync around each
        tc, tp = [], []
        for i in range(a.iters + 2):
            for which in ((0, 1) if i % 2 == 0 else (1, 0)):
                c = fresh()
                d = bts.device(fhe, c)
                bts.sync()
                t = time.perf_counter()
                if which == 0:
                    bts.cpp_bootstrap(d, slots)
                else:
                    pyb.bootstrap(d, slots)
                bts.sync()
                if i >= 2:
                    (tc if which == 0 else tp).append((time.perf_counter() - t) * 1e3)
        mc, mp = statistics.median(tc), statistics.median(tp)
        print(f"[gate] slots={slots} wall median: C++ {mc:.3f} ms  python {mp:.3f} ms  diff {mp - mc:+.3f} ms "
              f"(n={len(tc)})", flush=True)

        tr = []
        c = fresh()
        pyb.bootstrap(bts.device(fhe, c), slots, trace=tr)
        print("[gate]   stages: " + "  ".join(f"{n} {l}L/d{g} {ms:.2f}ms" for n, l, g, ms in tr), flush=True)
        if slots == 32768:
            y = np.asarray(_core.decode_token_output(inf, C))[:768]
            want = x * 0.999 ** a.depth
            err = float(np.max(np.abs(y - want)))
            print(f"[gate]   python output vs plaintext: max err {err:.3g} ({-np.log2(err):.1f} bits)", flush=True)
    print("[gate] PASS" if ok else "[gate] FAIL", flush=True)
    sess.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
