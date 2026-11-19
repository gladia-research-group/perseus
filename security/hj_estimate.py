# hj_estimate.py: Hou-Jiang enhanced hybrid decoding estimate (ePrint 2026/366, merged into CRYPTO'26 "Careful with
# the Ring!"), Sec. 5.3 / Eq. (3.2):  T = (T_BKZ(beta, d-k) + N |S| T_bdd(d-k)) / (p_bdd p_mitm (1 - (1 - p_S)^N)).
#
# Estimator = ext/est_hj/estimator (setup.sh; env HJ_EST_PATH overrides): lattice-estimator e35f45b (the commit HJ
# forked) with lwe_primal.py and cost.py replaced verbatim by HJ's published ones
# (github.com/identitymapping/enhanced_lattice-estimator f8968ad9). No other HJ file is used. Their cost model, in
# their code (PrimalHybrid.cost):
#   bound = N, capped at int(T_BKZ / T_svp) + 1 when T_BKZ / T_svp < N (rotations used);
#   hw grows while T_svp * bound * ssf(|S|) < T_BKZ (ssf = sqrt with MITM), |S| = sum_{i<=hw} C(k, i) 2^i;
#   p = 1 - (1 - p_S)^bound (1024-bit), then x p_mitm (Babai MITM) x p_babai (eta <= 20);  rop = (BKZ + svp) x amplify(0.99, p).
#   d = max(beta, min(ceil(sqrt((n-k) log(q/xi) / log delta)), m-k)) (lattice-estimator 1e28f669 formula).
#
# usage: sage -python hj_estimate.py <label> <logn> <logq> <secret> <mode> [sigma] [m]
#   secret: t<h> SparseTernary(h - h//2, h//2), b<h> sparse binary, dense Uniform(-1,1), bin Uniform(0,1),
#           spru = SPRU block key (63 unknown blocks of 1024, one 1 each, strided guess model of estimate.py spru_set)
#   mode:   usvp | gv | egv | hg | ehg
#           gv  = primal_hybrid(mitm=False, babai=False)              (guess-and-verify, Table 5.1 T_gvDec)
#           hg  = primal_hybrid(mitm=True,  babai=True)               (Howgrave-Graham + MITM, T_hgDec)
#           egv / ehg = the same with structure_leverage=True, deg_ring=N (T_egvDec / T_ehgDec)
#   m: "oo" (HJ's tables), "n" (= N, one RLWE sample), or an integer.
#   env SEARCH unset -> HJ's own search (zeta grid step n//100 over [0, n], refine +-25 step 10, d by formula).
#   env ZETA=z      -> their cost_zeta at fixed zeta (beta local_minimum, then their two-stage d search).
#   env SCAN=shard:nshards SCANOUT=path -> full optimisation: EVERY zeta of this shard's residue class, ascending, with
#       their cost_zeta(optimize_d=False); stop at the first zeta whose rigorous lower bound LB(zeta) exceeds the best
#       cost found by any shard (LB non-decreasing in zeta, so the cut is final). See lb_log2.
#   env DOPT=z1,z2,... -> d search (dfull) at those zetas.
#   env SEARCH=fine -> multi-resolution full zeta search + d search (see the SEARCH == fine branch).
import sys, json, time, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("HJ_EST_PATH") or os.path.join(HERE, "ext", "est_hj"))
from estimator import LWE, ND, RC  # noqa
from estimator import lwe_primal
from estimator.lwe_primal import PrimalHybrid, PrimalUSVP
from estimator.lwe_parameters import LWEParameters
from estimator.cost import Cost
from functools import partial
from sage.all import oo, RR, log, ZZ, RealField, binomial

label, logn, logq, sec, mode = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
N = int(logn[1:]) if logn.startswith("n") else 2 ** int(logn)
sigma = float(sys.argv[6]) if len(sys.argv) > 6 else 3.19
mm = sys.argv[7] if len(sys.argv) > 7 else "oo"
F = RealField(4000)
q = ZZ((F(2) ** F(logq)).round()) if logq != int(logq) else ZZ(2) ** int(logq)

NBLK, BLKLEN = 63, 1024
if sec == "dense":
    Xs = ND.Uniform(-1, 1, n=N)
elif sec == "bin":
    Xs = ND.Uniform(0, 1, n=N)
elif sec.startswith("t"):
    h = int(sec[1:]); Xs = ND.SparseTernary(h - h // 2, h // 2, n=N)
elif sec.startswith("b"):
    h = int(sec[1:]); Xs = ND.SparseTernary(h, 0, n=N)
elif sec == "spru":
    # block-binary key: block 0 known (s'_0 = 1) and removed, so the LWE dimension is 63 * 1024; weight 63, base 1
    N_ring = N
    Xs = ND.SparseTernary(NBLK, 0, n=NBLK * BLKLEN)
Xe = ND.DiscreteGaussian(sigma)
n_lwe = Xs.n
m = oo if mm == "oo" else (n_lwe if mm == "n" else int(mm))
params = LWEParameters(n=n_lwe, q=q, Xs=Xs, Xe=Xe, m=m)

if sec == "spru":
    # SPRU strided guess model (same as estimate.py spru_set): zeta = NBLK c + r guessed coordinates, r blocks hold
    # c+1, the rest c; the guessed piece of a block holds its 1 w.p. c_i / BLKLEN independently; a pattern = which
    # pieces are nonzero and where. HJ's hw loop asks for (#patterns of exactly weight hw) via binomial(zeta, hw) * base^hw
    # (base = 1 here) and P(exactly hw nonzero) via prob_drop(n, h, zeta, fail=hw): both are swapped for the block model.
    # Rotations: X^j maps a periodic (strided) guess set to one with the same distribution, so HJ's Assumption 1
    # (p_shift = 1 - (1 - p_S)^N) is the attacker-favourable reading, N = 65536 rotations (as the r65536 rows of estimate.py).
    R2 = RealField(256)

    def _split(zeta):
        c, r = divmod(zeta, NBLK)
        return c, r

    def spru_count(zeta, hw):
        c, r = _split(zeta)
        if c >= BLKLEN:
            return ZZ(1) if hw == 0 else ZZ(0)
        S = ZZ(0)
        for j in range(max(0, hw - (NBLK - r)), min(r, hw) + 1):
            S += binomial(r, j) * ZZ(c + 1) ** j * binomial(NBLK - r, hw - j) * ZZ(c) ** (hw - j)
        return S

    def spru_prob(n_, h_, zeta, fail=0):
        c, r = _split(zeta)
        if c >= BLKLEN:
            return 1.0 if fail == 0 else 0.0
        pa, pb = R2(c + 1) / BLKLEN, R2(c) / BLKLEN
        P = R2(0)
        w = fail
        for j in range(max(0, w - (NBLK - r)), min(r, w) + 1):
            P += binomial(r, j) * pa ** j * (1 - pa) ** (r - j) * binomial(NBLK - r, w - j) * pb ** (w - j) * (1 - pb) ** (NBLK - r - w + j)
        return RR(P)

    lwe_primal.binomial = lambda zeta, hw: spru_count(zeta, hw)
    lwe_primal.prob_drop = spru_prob

if mode == "usvp":
    t = time.time()
    c = LWE.primal_usvp(params, red_cost_model=RC.MATZOV)
    print(json.dumps({"label": label, "mode": mode, "logq": logq, "secret": sec, "m": mm,
                      "bits": float(RR(log(c["rop"], 2))) if c["rop"] < oo else None,
                      "beta": int(c.get("beta", 0)), "d": int(c.get("d", 0)), "s": round(time.time() - t, 1)}), flush=True)
    sys.exit(0)

enh = mode.startswith("e")
kind = mode[1:] if enh else mode
mitm, babai = {"gv": (False, False), "hg": (True, True)}[kind]
deg = N
kw = dict(mitm=mitm, babai=babai, red_cost_model=RC.MATZOV)
if enh:
    kw.update(deg_ring=deg, structure_leverage=True)


def fl(c, k):
    v = c.get(k)
    if v is None:
        return None
    if k in ("rop", "|S|", "red", "svp"):
        return float(RR(log(v, 2))) if v < oo else None
    if k == "prob":
        return float(RR(log(v, 2))) if v > 0 else None
    return int(v)


def rec_of(c):
    return {"bits": fl(c, "rop"), "zeta": fl(c, "zeta"), "beta": fl(c, "beta"), "d": fl(c, "d"), "eta": fl(c, "eta"),
            "hw": fl(c, "hw"), "logS": fl(c, "|S|"), "logprob": fl(c, "prob"), "red": fl(c, "red"), "svp": fl(c, "svp"),
            "reps": float(RR(log(c["repetitions"], 2))) if c.get("repetitions") else None}


T0 = time.time()
P = LWEParameters.normalize(params)
m_ = P.m + P.n if P.Xs <= P.Xe else P.m       # PrimalHybrid.__call__'s m (Bai-Galbraith)
czeta = partial(PrimalHybrid.cost_zeta, params=P, red_shape_model="gsa", m=m_, **kw)


def lb_log2(zeta):
    """Rigorous lower bound on log2(rop) of HJ's PrimalHybrid.cost at this zeta, any beta/d/hw/bound:
    rop >= bound * ssf(|S|) * svp * reps, svp >= 1, reps = amplify(0.99, p) >= max(1, 0.046 / p) and
    p <= 1 - (1 - p_S)^bound <= bound * p_S <= bound * |S| * pmax(zeta) (union bound; p_mitm, p_babai <= 1).
    No MITM: rop >= x max(1, 0.046/(x pmax)) >= 0.046/pmax  (x = bound |S|).
    MITM: rop >= sqrt(bound) sqrt(x) max(1, 0.046/(x pmax)) >= sqrt(0.046/pmax).
    pmax(zeta) = largest probability of a single zeta-prefix pattern in the estimator's model (prob_drop / patterns),
    non-increasing in zeta, so LB is non-decreasing. SPRU: pmax = prod over pieces of max(1 - c_i/B, 1/B)."""
    if sec == "spru":
        c, r = divmod(zeta, NBLK)
        if c >= BLKLEN:
            return float("inf")
        R2 = RealField(256)
        lp = R2(0)
        for ci, cnt in ((c + 1, r), (c, NBLK - r)):
            lp += cnt * max(1 - R2(ci) / BLKLEN, R2(1) / BLKLEN).log()
        best = RR(lp)
    else:
        n, hh = P.Xs.n, P.Xs.hamming_weight
        base = P.Xs.bounds[1] - P.Xs.bounds[0]
        lo, hi = max(0, zeta - n + hh), min(zeta, hh)
        lg = lambda x: RR(x + 1).log_gamma()
        lC = lambda a, b: lg(a) - lg(b) - lg(a - b)
        best = None
        for w in range(lo, hi + 1):
            lp = lC(n - zeta, hh - w) - lC(n, hh) - w * log(RR(base))
            best = lp if best is None or lp > best else best
    l = float(log(RR(0.046), 2)) - float(best / log(RR(2)))
    return l / 2 if mitm else l


def dfull(zeta):
    """d search at the zeta's optimal beta (their cost_zeta step 1): every 20th d over [n - zeta, d_formula + 1], then every d
    within +-20 of the best (their two-stage d search, refined to step 1 over a wider window)."""
    c = czeta(zeta=zeta, optimize_d=False)
    if c["rop"] == oo:
        return c, c
    f = partial(PrimalHybrid.cost, params=P, zeta=zeta, red_shape_model="gsa", m=m_, **kw)
    best = c
    lo, hi = max(P.n - zeta, c["beta"]), c["d"] + 1
    for d in list(range(lo, hi + 1, 20)) + [hi]:
        x = f(beta=c["beta"], d=d)
        if x["rop"] < best["rop"]:
            best = x
    d0 = best["d"]
    for d in range(max(lo, d0 - 20), min(hi, d0 + 20) + 1):
        x = f(beta=c["beta"], d=d)
        if x["rop"] < best["rop"]:
            best = x
    return best, c


def _cz(z):
    t = time.time()
    c = czeta(zeta=z, optimize_d=False)
    r = rec_of(c)
    r["zeta"] = z
    r["s"] = round(time.time() - t, 1)
    return r


def _dz(z):
    t = time.time()
    b, c = dfull(z)
    r = rec_of(b)
    r.update({"zeta": z, "bits_nod": fl(c, "rop"), "d_nod": fl(c, "d"), "s": round(time.time() - t, 1)})
    return r


scan = os.environ.get("SCAN")
if os.environ.get("SEARCH") == "fine":
    # Full optimisation of HJ's cost over zeta (their cost_zeta per zeta: beta local_minimum, hw rule, bound cap):
    #  1. HJ's coarse grid: every n//100 over [0, n] (their stage 1; the profile is logged to check for one basin)
    #  2. every 25 within +-n//100 of the coarse best      3. every zeta within +-25 of that best
    #  4. d search (dfull) at the 3 best zetas of stage 3.
    from multiprocessing import Pool
    trace = os.environ.get("TRACE")
    npar = int(os.environ.get("FINEP", "4"))
    pool = Pool(npar)
    seen = {}

    def run(zs, fn=_cz, tag="z"):
        zs = [z for z in sorted(set(zs)) if 0 <= z <= P.n and (tag, z) not in seen]
        for r in pool.map(fn, zs, chunksize=1):
            seen[(tag, r["zeta"])] = r
            if trace:
                with open(trace, "a") as f:
                    f.write(json.dumps({"label": label, "mode": mode, "stage": tag, **r}) + "\n")

    def bestz(tag="z"):
        cand = [(r["bits"], z) for (t_, z), r in seen.items() if t_ == tag and r["bits"] is not None]
        return min(cand)[1] if cand else 0

    step = max(1, P.n // 100)
    run(range(0, P.n + 1, step))
    prof = [(z, seen[("z", z)]["bits"]) for z in range(0, P.n + 1, step) if ("z", z) in seen]
    fin = [b for z, b in prof if b is not None]
    nloc = sum(1 for i in range(1, len(prof) - 1) if prof[i][1] is not None and all(
        prof[j][1] is None or prof[i][1] < prof[j][1] for j in (i - 1, i + 1)))
    z1 = bestz()
    run(range(z1 - step, z1 + step + 1, 25))
    z2 = bestz()
    run(range(z2 - 25, z2 + 26))
    top = sorted([(r["bits"], z) for (t_, z), r in seen.items() if t_ == "z" and r["bits"] is not None])[:3]
    run([z for _, z in top], fn=_dz, tag="d")
    dbest = min([(r["bits"], z) for (t_, z), r in seen.items() if t_ == "d" and r["bits"] is not None])
    rb = seen[("d", dbest[1])]
    print(json.dumps({"label": label, "mode": mode, "logq": logq, "secret": sec, "m": mm, "search": "fine",
                      **rb, "bits_grid": seen[("z", z1)]["bits"], "zeta_grid": z1,
                      "bits_nod": seen[("z", bestz())]["bits"], "zeta_nod": bestz(), "grid_local_minima": nloc,
                      "n_evals": len(seen), "s": round(time.time() - T0, 1)}), flush=True)
    pool.close()
elif os.environ.get("DOPT"):
    for z in [int(x) for x in os.environ["DOPT"].split(",")]:
        t = time.time()
        b, c = dfull(z)
        print(json.dumps({"label": label, "mode": mode, "logq": logq, "secret": sec, "m": mm, "dopt": True,
                          **rec_of(b), "bits_nod": fl(c, "rop"), "d_nod": fl(c, "d"), "zeta": z,
                          "s": round(time.time() - t, 1)}), flush=True)
elif scan:
    shard, nsh = [int(x) for x in scan.split(":")]
    out_path = os.environ["SCANOUT"]

    def emit(r):
        with open(out_path, "a") as f:
            f.write(json.dumps(r) + "\n")

    c0 = czeta(zeta=0, optimize_d=False)
    U = fl(c0, "rop") if c0["rop"] < oo else float("inf")
    if shard == 0:
        emit({"label": label, "mode": mode, "logq": logq, "secret": sec, "zeta": 0, **{k: v for k, v in rec_of(c0).items() if k != "zeta"}})

    def best():
        B = U
        try:
            for ln in open(out_path):
                r_ = json.loads(ln)
                if r_.get("label") == label and r_.get("mode") == mode and r_.get("bits") is not None:
                    B = min(B, r_["bits"])
        except FileNotFoundError:
            pass
        return B

    z = shard if shard > 0 else nsh
    nb, B = 0, best()
    while z <= P.n:
        if nb % 50 == 0:
            B = best()
        nb += 1
        lb = lb_log2(z)
        if lb > B:
            B = best()
            if lb > B:
                emit({"label": label, "mode": mode, "stop": z, "shard": shard, "lb": lb, "B": B})
                break
        tz = time.time()
        c = czeta(zeta=z, optimize_d=False)
        r = {"label": label, "mode": mode, "logq": logq, "secret": sec, **rec_of(c), "zeta": z, "lb": lb,
             "s": round(time.time() - tz, 2)}
        emit(r)
        z += nsh
    else:
        emit({"label": label, "mode": mode, "stop": P.n + 1, "shard": shard, "lb": None, "B": best(), "end": True})
    print(json.dumps({"label": label, "mode": mode, "shard": shard, "done": True, "s": round(time.time() - T0, 1)}), flush=True)
else:
    zf = os.environ.get("ZETA")
    t = time.time()
    if zf is not None:
        c = PrimalHybrid.cost_zeta(int(zf), P, red_shape_model="gsa", m=m_, optimize_d=True, **kw)
    else:
        c = LWE.primal_hybrid(params, **kw)
    print(json.dumps({"label": label, "mode": mode, "logq": logq, "secret": sec, "m": mm, "search": "zeta" if zf else "hj",
                      **rec_of(c), "s": round(time.time() - t, 1)}), flush=True)
