"""Replace an encoder config's THOR composite GELUs by one Chebyshev series per block, the gpt2_base_n32_pf form:
GELU(X t) / X on [-1, 1], X = 1.25 S (S the THOR site's calibrated xmax), at the smallest degree of DEGS whose error
over the whole [-X, X] (after eval_chebyshev's coefficient trim) is at most FLOOR (the THOR composite level of the
repo's GELU calibrator, GPT-2's and ViT's sites) or the site's own THOR error on [-S, S] if larger. A block whose series
would need more than 383 (the THOR depth) keeps THOR.
    .venv/bin/python scripts/utils/fit_gelu_series.py CONFIGS.json [--write]"""
import json, math, sys
import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.special import erf

DEGS = (63, 95, 127, 191, 255, 383)
FLOOR = 4.7e-4
EPS = 1e-9                                     # the entries' cheb_eps (the runtime trim; default 1e-6)


def gelu(x):
    return 0.5 * x * (1 + erf(x / math.sqrt(2)))


def trim(c):
    c = np.where(np.abs(c) <= EPS * np.abs(c).max(), 0.0, c)
    n = len(c) - 1
    while n > 0 and c[n] == 0.0:
        n -= 1
    return c[:n + 1]


def thor_err(g, t):
    S = g["xmax"]
    u = (2 * t - (g["thor_p1_a"] + g["thor_p1_b"])) / (g["thor_p1_b"] - g["thor_p1_a"])
    p1 = C.chebval(u, g["thor_p1_cheb"])
    v = (2 * p1 - (g["thor_p2_a"] + g["thor_p2_b"])) / (g["thor_p2_b"] - g["thor_p2_a"])
    comp = C.chebval(v, g["thor_p2_cheb"])
    return np.abs(S * t * (comp + 0.5) - gelu(S * t)).max()


def main():
    path = sys.argv[1]
    cfg = json.load(open(path))
    t = np.linspace(-1, 1, 400001)
    for name in sorted(cfg["softgelu"], key=lambda n: int(n.split(".")[2])):
        g = cfg["softgelu"][name]
        if g["method"] != "thor_composite":
            print(f"{name}: {g['method']}, kept")
            continue
        S = g["xmax"]
        X = 1.25 * S
        e_t = thor_err(g, t)
        pick = None
        for deg in DEGS:
            c = C.chebinterpolate(lambda u: gelu(X * u) / X, deg)
            c[3::2] = 0.0                       # GELU(x) - x/2 is even: no odd terms past T_1
            c = trim(c)
            e = np.abs(C.chebval(t, c) * X - gelu(X * t)).max()
            if e <= max(e_t, FLOOR):
                pick = (deg, c, e)
                break
        if pick is None:
            print(f"{name}: S={S:.2f} THOR err {e_t:.2e}; series needs > {DEGS[-1]}: THOR kept")
            continue
        deg, c, e = pick
        print(f"{name}: S={S:.2f} THOR err {e_t:.2e} -> degree {deg} (trimmed {len(c) - 1}), err {e:.2e}")
        cfg["softgelu"][name] = {
            "method": "chebyshev", "xmax": X, "cheb_coeffs": [float(x) for x in c], "cheb_a": -1.0, "cheb_b": 1.0,
            "cheb_eps": EPS, "note": f"degree-{deg} Chebyshev of GELU(X t)/X (erf) on [-1,1], X = 1.25 x calibrated xmax {S:.4f}; "
                    f"max abs err {e:.1e} on [-X, X] after the runtime trim; replaces thor_composite "
                    f"(abs err {e_t:.1e} on [-xmax, xmax]); target max(that, {FLOOR:g})"}
    if "--write" in sys.argv:
        with open(path, "w") as f:
            json.dump(cfg, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
