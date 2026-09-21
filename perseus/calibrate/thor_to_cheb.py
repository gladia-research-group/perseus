import argparse
import json
import logging

import mpmath as mp

from perseus._log import configure_cli_logging

log = logging.getLogger(__name__)

mp.mp.dps = 80


def poly_eval(coeffs, x):
    acc = mp.mpf(0)
    for c in reversed(coeffs):
        acc = acc * x + mp.mpf(c)
    return acc


def to_cheb(coeffs, a, b):
    n = len(coeffs) - 1
    m = n + 1
    # Chebyshev-Gauss nodes on [-1,1] mapped to [a,b]
    ys = [mp.cos(mp.pi * (k + mp.mpf(0.5)) / m) for k in range(m)]
    xs = [(mp.mpf(b) - mp.mpf(a)) / 2 * y + (mp.mpf(b) + mp.mpf(a)) / 2 for y in ys]
    vals = [poly_eval(coeffs, x) for x in xs]
    out = []
    for j in range(m):
        s = mp.mpf(0)
        for k in range(m):
            s += vals[k] * mp.cos(mp.pi * j * (k + mp.mpf(0.5)) / m)
        cj = s * 2 / m
        if j == 0:
            cj /= 2
        out.append(float(cj))
    return out


def cheb_eval(cheb, a, b, x):
    y = (2 * mp.mpf(x) - (mp.mpf(a) + mp.mpf(b))) / (mp.mpf(b) - mp.mpf(a))
    b1 = b2 = mp.mpf(0)
    for c in reversed(cheb[1:]):
        b1, b2 = 2 * y * b1 - b2 + mp.mpf(c), b1
    return y * b1 - b2 + mp.mpf(cheb[0])


def fit_p1_domain(p1, m_max, c_cap):
    m = round(float(m_max), 4)
    best = None
    while m >= 1.0 - 1e-12:
        m = round(m, 4)                   # keep the emitted domain free of fp drift
        cb = to_cheb(p1, -m, m)
        if max(abs(c) for c in cb) <= c_cap:
            best = (m, cb)
            break
        m -= 0.005
    if best is None:                      # even [-1,1] is over the cap: keep it (status quo)
        best = (1.0, to_cheb(p1, -1.0, 1.0))
    return best


def fit_p2_pad(p2, lo, hi, pad_max, c_cap):
    p = float(pad_max)
    while p > 0.0:
        d = (hi - lo) * mp.mpf(repr(p)) + mp.mpf("1e-9")
        a, b = float(lo - d), float(hi + d)
        cb = to_cheb(p2, a, b)
        if max(abs(c) for c in cb) <= c_cap:
            return a, b, p, cb
        p = round(p - 0.01, 4)
    d = mp.mpf("1e-9")
    a, b = float(lo - d), float(hi + d)
    return a, b, 0.0, to_cheb(p2, a, b)


def convert_site(g, name, m1=1.0, c_cap=10.0, pad_max=0.10):
    p1, p2 = g["thor_p1"], g["thor_p2"]
    m_fit, p1_cheb = fit_p1_domain(p1, m1, c_cap)
    a1, b1 = -m_fit, m_fit

    grid = [mp.mpf(-1) + mp.mpf(2) * k / 4096 for k in range(4097)]
    p1_vals = [poly_eval(p1, x) for x in grid]
    lo, hi = min(p1_vals), max(p1_vals)
    a2, b2, pad, p2_cheb = fit_p2_pad(p2, lo, hi, pad_max, c_cap)

    wgrid = [mp.mpf(a1) + (mp.mpf(b1) - mp.mpf(a1)) * k / 512 for k in range(513)]
    e1 = max(abs(cheb_eval(p1_cheb, a1, b1, x) - poly_eval(p1, x)) for x in wgrid)
    p1w = [poly_eval(p1, x) for x in wgrid]
    grid2 = [mp.mpf(a2) + (mp.mpf(b2) - mp.mpf(a2)) * k / 256 for k in range(257)]
    e2 = max(abs(cheb_eval(p2_cheb, a2, b2, x) - poly_eval(p2, x)) for x in grid2)
    c1max = max(abs(c) for c in p1_cheb)
    c2max = max(abs(c) for c in p2_cheb)
    c1sum = sum(abs(c) for c in p1_cheb)
    log.info(f"[thor_cheb] {name}: p1 deg {len(p1)-1} |mono|max={max(abs(c) for c in p1):.3g} "
          f"-> |T|max={c1max:.3g} sum|T|={c1sum:.3g} err={float(e1):.3e} on "
          f"[{a1:.3f},{b1:.3f}] (p1 out on full domain "
          f"[{float(min(p1w)):.3f},{float(max(p1w)):.3f}]) | "
          f"p2 domain [{a2:.4f},{b2:.4f}] pad={pad:.0%} | "
          f"p2 |T|max={c2max:.3g} err={float(e2):.3e}")
    assert float(e1) < 1e-9 and float(e2) < 1e-9, f"{name}: conversion error too high"
    assert c1max < 1e3 and c2max < 1e3, f"{name}: T coefficients not O(1) — domain wrong?"

    g["thor_p1_cheb"] = p1_cheb
    g["thor_p2_cheb"] = p2_cheb
    g["thor_p1_a"] = a1
    g["thor_p1_b"] = b1
    g["thor_p2_a"] = a2
    g["thor_p2_b"] = b2


def main():
    configure_cli_logging()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--p1-domain", type=float, default=1.0, metavar="M",
                    help="MAXIMUM domain [-M,M] for thor_p1's T-basis (default 1.0 = the "
                         "historical behaviour). Each site gets the largest M ≤ this "
                         "whose T coefficients stay under --p1-coeff-cap; do not ask "
                         "for more than ~1.12, p1 diverges out there.")
    ap.add_argument("--p1-coeff-cap", type=float, default=10.0, metavar="C",
                    help="per-site cap on max|T-coeff| when fitting the p1 domain and "
                         "the p2 pad (default 10.0 = the EvalMod wall).")
    ap.add_argument("--p2-pad", type=float, default=0.10, metavar="P",
                    help="MAXIMUM pad on p2's domain as a fraction of p1's certified "
                         "output range (default 0.10; was a flat 0.01). Fitted down "
                         "per site under --p1-coeff-cap.")
    args = ap.parse_args()
    src, dst, m1, ccap = args.src, args.dst, args.p1_domain, args.p1_coeff_cap
    pad_max = args.p2_pad
    assert 0.5 <= m1 <= 1.2, f"--p1-domain {m1} out of sane range"
    assert 0.0 <= pad_max <= 0.5, f"--p2-pad {pad_max} out of sane range"
    cfg = json.load(open(src))
    n = 0

    def walk(o, path=""):
        nonlocal n
        if isinstance(o, dict):
            if "thor_p1" in o and isinstance(o.get("thor_p1"), list):
                convert_site(o, path, m1, ccap, pad_max)
                n += 1
            else:
                for k, v in o.items():
                    walk(v, f"{path}/{k}")

    walk(cfg)
    assert n > 0, "no THOR gelu sites found"
    import os
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(cfg, open(dst, "w"), indent=1)
    log.info(f"[thor_cheb] {n} sites converted -> {dst}")


if __name__ == "__main__":
    main()
