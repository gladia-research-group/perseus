"""Crown-jewel aggregation: per-arm encrypted-classification stats from cj sweep logs.

usage:
  cj_aggregate.py baseline='logs/core/cjr_cal_*.out' squeeze='logs/core/cjr_sq_*.out' ...

Conventions (the paper's):
  * end-to-end accuracy counts a DECRYPTION FAILURE AS A MISS. A failure is an image with
    "[cj] image N FAILED" and no "[forward] img=N" line (a retry that later produced a
    forward is not a failure).
  * "decodable acc" = accuracy over the images that did decrypt.
  * "ref acc" = that arm's own PLAINTEXT accuracy on the same images (exact arithmetic),
    so encryption damage = decodable-vs-ref, and weight damage = ref-vs-teacher.
  * "flips" = decodable images whose encrypted top-1 differs from their own plaintext top-1.
  * Wilson 95% intervals, because n is a few hundred and the point estimates get compared
    across arms.
Bootstraps/unplanned come from the per-forward [encvit] SUMMARY lines; unplanned_bts must be
0 in every arm or the run was off-plan and the latency is not quotable.
"""
import glob
import math
import re
import statistics
import sys

FWD = re.compile(r"\[forward\] img=(\d+) label=(\d+) enc_top1=(-?\d+) ref_top1=(-?\d+)"
                 r".*?mape=([\d.]+) fwd_s=([\d.]+)")
FAIL = re.compile(r"\[cj\] image (\d+) FAILED")
SUMM = re.compile(r"bootstraps=(\d+) bts_per_block=[\d.]+ unplanned_bts=(\d+)")


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def arm(pattern):
    fwd, failed, bts, unplanned = {}, set(), [], []
    for f in sorted(glob.glob(pattern)):
        txt = open(f, errors="ignore").read()
        for m in FWD.finditer(txt):
            fwd[int(m.group(1))] = dict(label=int(m.group(2)), enc=int(m.group(3)),
                                        ref=int(m.group(4)), mape=float(m.group(5)),
                                        s=float(m.group(6)))
        failed |= {int(m.group(1)) for m in FAIL.finditer(txt)}
        for m in SUMM.finditer(txt):
            bts.append(int(m.group(1)))
            unplanned.append(int(m.group(2)))
    failed -= set(fwd)
    return fwd, failed, bts, unplanned


print(f"{'arm':<10} {'n':>4} {'fail':>10} {'decodable':>10} {'e2e (95% CI)':>22} "
      f"{'ref':>7} {'flip':>5} {'bts':>5} {'mean s':>8} {'med s':>8} {'mape':>6}")
for spec in sys.argv[1:]:
    name, pattern = spec.split("=", 1)
    fwd, failed, bts, unplanned = arm(pattern)
    n = len(fwd) + len(failed)
    if not n:
        print(f"{name:<10} no data for {pattern}")
        continue
    dec_ok = sum(1 for v in fwd.values() if v["enc"] == v["label"])
    ref_ok = sum(1 for v in fwd.values() if v["ref"] == v["label"])
    flips = sum(1 for v in fwd.values() if v["enc"] != v["ref"])
    lo, hi = wilson(dec_ok, n)
    ubad = sum(unplanned)
    print(f"{name:<10} {n:>4} {len(failed):>4} {100*len(failed)/n:>4.1f}% "
          f"{100*dec_ok/len(fwd):>9.2f}% {100*dec_ok/n:>9.2f}% [{lo:5.1f},{hi:5.1f}] "
          f"{100*ref_ok/len(fwd):>6.2f}% {flips:>5} "
          f"{statistics.mode(bts) if bts else 0:>5} "
          f"{statistics.mean(v['s'] for v in fwd.values()):>8.1f} "
          f"{statistics.median(v['s'] for v in fwd.values()):>8.1f} "
          f"{statistics.median(v['mape'] for v in fwd.values()):>6.3f}"
          + ("" if ubad == 0 else f"   !! unplanned_bts={ubad} — OFF PLAN, latency not quotable"))
