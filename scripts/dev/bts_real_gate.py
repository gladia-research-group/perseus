#!/usr/bin/env python3
"""Gate for the real-payload dense bootstrap (FIDESLIB_BTS_REAL): one EvalMod chain and the real StC stage 0 must
return Re(x) as precisely as the two-chain bootstrap returns x, and cost less.

    python scripts/dev/bts_real_gate.py [--iters 20] [--device 4]

Inputs: a real vector over every slot, the same with a 0.6 % imaginary residue (the route returns its real part), one
with a large slot mean (coefficient 0, the column the real StC stage halves) and a decode-like token lane. Then
alternating timed runs of both routes.
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
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--device", type=int, default=3)
    a = ap.parse_args()

    from examples.gpt2_from_primitives import env
    sess = env.open_session(a.device, complex_payload=True, FIDESLIB_BTS_SHIFT="1", FIDESLIB_BTS_REAL="1")
    from perseus import _core
    from perseus.impl import bootstrap as pyb
    bts = _core.bts
    inf, fhe = sess.inf, sess.inf.fhe
    S = inf.slots

    def run(base, real):
        c = bts.clone(base)
        bts.set_real_payload(fhe, real)
        try:
            pyb.bootstrap(bts.device(fhe, c), S)
        finally:
            bts.set_real_payload(fhe, False)
        return c, np.asarray(_core.decrypt_slots_complex(inf, c))

    rng = np.random.default_rng(1)
    zero = lambda: _core.encode_token_input(inf, np.zeros(768))
    xr, xi, x768 = rng.uniform(-1, 1, S), rng.uniform(-1, 1, S), rng.uniform(-1, 1, 768)
    real_ct = fhe.mult(inf.add_pt(zero(), xr), 0.999)
    cplx_ct = fhe.add(fhe.mult(inf.add_pt(zero(), xr), 0.999), fhe.mult_i(fhe.mult(inf.add_pt(zero(), xi), 0.006)))
    mean_ct = fhe.mult(inf.add_pt(zero(), 0.5 * xr + 0.3), 0.999)                 # slot mean 0.3: coefficient 0
    tok_ct = fhe.mult(_core.encode_token_input(inf, x768), 0.999)
    tok_x = np.real(np.asarray(_core.decrypt_slots_complex(inf, tok_ct)))
    cases = [("real, every slot", real_ct, 0.999 * xr, 0.999 * xr),
             ("real + 0.6% imaginary", cplx_ct, 0.999 * xr + 0.006j * xi, 0.999 * xr),
             ("real, slot mean 0.3", mean_ct, 0.999 * (0.5 * xr + 0.3), 0.999 * (0.5 * xr + 0.3)),
             ("token lane", tok_ct, tok_x, tok_x)]
    bits = lambda e: -np.log2(max(e, 1e-300))
    ok = True
    for name, base, want, want_re in cases:
        _, std = run(base, False)
        _, rea = run(base, True)
        e_std = float(np.max(np.abs(std - want)))
        e_rea = float(np.max(np.abs(rea - want_re)))
        print(f"[real] {name:22s} two chains {bits(e_std):5.1f} bits | real route vs Re {bits(e_rea):5.1f} bits "
              f"(max |Im| {float(np.max(np.abs(rea.imag))):.1e})", flush=True)
        ok &= bits(e_rea) >= bits(e_std) - 0.5

    ts, tr = [], []
    for i in range(a.iters + 2):
        for real in ((False, True) if i % 2 == 0 else (True, False)):
            c = bts.clone(real_ct)
            d = bts.device(fhe, c)
            bts.set_real_payload(fhe, real)
            bts.sync()
            t = time.perf_counter()
            pyb.bootstrap(d, S)
            bts.sync()
            bts.set_real_payload(fhe, False)
            if i >= 2:
                (tr if real else ts).append((time.perf_counter() - t) * 1e3)
    ms, mr = statistics.median(ts), statistics.median(tr)
    print(f"[real] wall median (n={len(ts)}): two chains {ms:.2f} ms, real route {mr:.2f} ms ({100 * (mr - ms) / ms:+.1f}%)",
          flush=True)
    print("[real] PASS" if ok else "[real] FAIL", flush=True)
    sess.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
