"""THOR composite GELU error at BERT's per-layer ranges (diagnostic)."""
import sys
import time

from perseus.calibrate.approximations.gelu import _fit_thor_composite


def main():
    for S in (26.0, 77.0, 111.0, 173.0):
        for d1, d2 in ((31, 27), (47, 31), (63, 31)):
            t0 = time.time()
            p1, p2, e, med = _fit_thor_composite(S, d1, d2)
            print(f"S {S:6.1f} d1 {d1} d2 {d2}: max err {e:.3e} median {med:.3e} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
