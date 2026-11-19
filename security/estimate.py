# estimate.py: one cost estimate on lattice-estimator 6019056 (MATZOV cost model, GSA, the estimator's defaults):
# the stock attacks, Ogilvie's rotation hybrid (ePrint 2026/279, Lemma 18), the exhaustive zeta scan with a proven
# lower-bound stop, two attacker-favourable stand-ins for Hou-Jiang (ePrint 2026/366) and the SPRU block-key model.
# Prints one JSON line (with SCAN: appends one line per zeta to SCANOUT).
#
# usage: sage -python estimate.py <label> <logn> <logq> <secret> <mode> [sigma] [m]
#   logn:   k -> ring degree 2^k; "n795" -> plain LWE of dimension 795 (no ring: the rot modes do not apply)
#   logq:   log2 of the modulus (q = round(2^logq))
#   secret: t<h> sparse ternary (h - h//2 ones, h//2 minus ones), b<h> sparse binary, dense (uniform ternary),
#           bin (uniform binary)
#   sigma:  error standard deviation (default 3.19);  m: "n" (N samples = one RLWE sample; default), "oo", or a number
#   mode:
#     est:<algo>   stock LWE.estimate restricted to one algorithm (usvp bdd bdd_hybrid bdd_mitm_hybrid dual dual_hybrid
#                  arora-gb bkw); the same code path as LWE.estimate(deny_list=<all others>)
#     plain_nomitm plain_mitm plain_cvp  primal hybrid, (babai, mitm) = (T,F) (T,T) (F,F)  [plain_cvp == est:bdd_hybrid]
#     rot_nomitm rot_h1 rot_h2 rot_cvp   Ogilvie's rotated guessing set (Lemma 18): |S_rot| = N |S|, hit 1 - (1 - p)^N
#                  in 256-bit arithmetic; H2 = sqrt(|S_rot|) NP calls, H1 = N sqrt(|S|); rot_cvp = projected-CVP NP
#     hjfree_*     stand-in for Hou-Jiang: the hit probability amplified by all rotations, the guessing set (so the
#                  decoding work) kept at |S|: the rotations are decoded for free
#     hjunit_*     hjfree plus unit-cost decoding (an extreme bound)
#     spru_*       SPRU block-binary key (env NBLK unknown blocks of BLKLEN, one 1 per block), strided guess model:
#                  spru_mitm / spru_nomitm plain, spru_rot_* rotated (NROT rotations, Lemma 18), spru_free_* as hjfree
#   env NROT: rotations in the rotated guessing set (default N)
#   env SCAN=lo:hi:shard:nshards SCANOUT=path -> exhaustive zeta scan instead of the stock zeta search: every zeta in
#     [lo, hi] with zeta % nshards == shard, all hw (no early break), beta by the estimator's local_minimum, d
#     optimised; a shard stops at the first zeta whose lower bound exceeds the best cost any shard has found (shards
#     share SCANOUT). hi = "auto": no upper limit. See lb_log2.
#   env ZETA=z -> the stock primal hybrid at a fixed zeta;  FLOATHIT=1 -> Lemma 18's hit in IEEE double (reproduces
#     the paper's underflow; validation only);  PROGRESS=path -> a line per zeta the stock search evaluates.
#   env LE_PATH: the lattice-estimator checkout (default: ext/lattice-estimator next to this file, see setup.sh)
import sys, json, time, os
sys.path.insert(0, os.environ.get("LE_PATH") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "ext", "lattice-estimator"))
from estimator import *
from estimator import lwe_primal, prob
from estimator.lwe_primal import PrimalHybrid, primal_usvp
from estimator.lwe_parameters import LWEParameters
from estimator.util import local_minimum
from estimator.cost import Cost
from estimator.conf import max_beta as max_beta_global
from functools import partial
from sage.all import oo, RR, log, ZZ, RealField

label, logn, logq, sec, mode = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
# logn: k -> ring degree 2^k; "n795" -> plain LWE dimension 795 (no ring: rot modes do not apply)
N = int(logn[1:]) if logn.startswith("n") else 2 ** int(logn)
sigma = float(sys.argv[6]) if len(sys.argv) > 6 else 3.19
mm = sys.argv[7] if len(sys.argv) > 7 else "n"
F = RealField(4000)
q = ZZ((F(2) ** F(logq)).round())
if sec == "dense":
    Xs = ND.Uniform(-1, 1, n=N)
elif sec == "bin":
    Xs = ND.Uniform(0, 1, n=N)
elif sec.startswith("t"):
    h = int(sec[1:]); Xs = ND.SparseTernary(h - h // 2, h // 2, n=N)
elif sec.startswith("b"):
    h = int(sec[1:]); Xs = ND.SparseTernary(h, 0, n=N)
Xe = ND.DiscreteGaussian(sigma)
m = N if mm == "n" else (oo if mm == "oo" else int(mm))
p = LWE.Parameters(n=N, q=q, Xs=Xs, Xe=Xe, m=m)
PROG = os.environ.get("PROGRESS")
T0 = time.time()


def plog(msg):
    if PROG:
        with open(PROG, "a") as f:
            f.write(f"{time.time() - T0:9.1f}s {label} {mode} {msg}\n")


orig = prob.guessing_set_and_hit_probability
R = RealField(256)


def rot_set(zeta, dist, hw):
    S, pr = orig(zeta, dist, hw)
    if zeta == 0:
        return S, pr
    return rot_amp(S, pr, mode)            # H1: NROT sqrt(|S_plain|) NP calls = sqrt(NROT^2 |S_plain|)


NROT = int(os.environ.get("NROT", N))   # rotations used by the rotated guessing set (default: all N)
NBLK = int(os.environ.get("NBLK", "0"))  # SPRU: number of unknown blocks, block length = BLKLEN
BLKLEN = int(os.environ.get("BLKLEN", "1024"))


FLOATHIT = os.environ.get("FLOATHIT") == "1"   # diagnostic: hit = 1 - (1 - p)^N in IEEE double (no-MitM gap study)


def rot_amp(S, pr, sub):
    if FLOATHIT:
        hit = 1.0 - (1.0 - float(pr)) ** NROT
        return (S * NROT * NROT if sub.endswith("h1") else S * NROT), RR(hit)
    pr = R(pr)
    hit = -(NROT * (-pr).log1p()).expm1() if pr < 1 else R(1)
    if sub.endswith("h1"):
        return S * NROT * NROT, RR(hit)
    return S * NROT, RR(hit)


def spru_set(zeta, dist, hw):
    """SPRU block-binary key (NBLK unknown blocks of BLKLEN, one 1 per block; the known block 0 removed from the LWE
    dimension). Guessed coordinates spread over the blocks: r blocks contribute c+1 coordinates, NBLK-r contribute c
    (zeta = NBLK c + r). The guessed piece of a block holds its 1 with probability c_i/BLKLEN, independently across
    blocks; candidates for a piece: its c_i positions. Guessing set: all patterns with <= hw nonzero pieces."""
    if zeta == 0:
        return 1, 1.0
    c, r = divmod(zeta, NBLK)
    if c >= BLKLEN:
        return ZZ(1), 1.0
    from sage.all import binomial as Bn
    S, hit = ZZ(0), R(0)
    pa, pb = R(c + 1) / BLKLEN, R(c) / BLKLEN
    for w in range(0, hw + 1):
        for j in range(max(0, w - (NBLK - r)), min(r, w) + 1):
            S += Bn(r, j) * ZZ(c + 1) ** j * Bn(NBLK - r, w - j) * ZZ(c) ** (w - j)
            hit += Bn(r, j) * pa ** j * (1 - pa) ** (r - j) * Bn(NBLK - r, w - j) * pb ** (w - j) * (1 - pb) ** (NBLK - r - w + j)
    if hit > 1:
        hit = R(1)
    if mode.startswith("spru_rot"):
        return rot_amp(S, hit, mode)
    if mode.startswith("spru_free"):          # Hou-Jiang proxy: rotations amplify the hit for free
        return rot_amp(S, hit, mode)[0] // NROT, RR(-(NROT * (-hit).log1p()).expm1() if hit < 1 else 1)
    return S, RR(hit)




def free_set(zeta, dist, hw):
    """Hou-Jiang proxy (attacker-favourable bound): all NROT rotations decoded for the price of one, i.e. the hit
    probability is amplified by the rotations (Lemma 18) but the NP-call count stays |S_plain|: an improvement linear
    in N over the plain hybrid when p is small (the abstract of ePrint 2026/366 claims 'linear in N')."""
    S, pr = orig(zeta, dist, hw)
    if zeta == 0:
        return S, pr
    pr = R(pr)
    hit = -(NROT * (-pr).log1p()).expm1() if pr < 1 else R(1)
    return S, RR(hit)


if mode.startswith("hj"):
    lwe_primal.guessing_set_and_hit_probability = free_set
if mode.startswith("hjunit"):
    PrimalHybrid.babai_cost = classmethod(lambda cls, d: Cost(rop=1))   # decoding for free as well (extreme bound)
if mode.startswith("rot"):
    lwe_primal.guessing_set_and_hit_probability = rot_set
if mode.startswith("spru"):
    lwe_primal.guessing_set_and_hit_probability = spru_set
babai = mode not in ("plain_cvp", "rot_cvp")
mitm = mode in ("plain_mitm", "rot_h1", "rot_h2", "spru_mitm", "spru_rot_h1", "spru_rot_h2", "hjfree_h2", "hjunit_h2", "spru_free_h2")

# progress hook on the stock search
_cz = PrimalHybrid.cost_zeta.__func__


def _cz_log(cls, zeta, params, *a, **k):
    t = time.time()
    c = _cz(cls, zeta, params, *a, **k)
    plog(f"zeta={zeta} rop=2^{float(RR(log(c['rop'], 2))) if c['rop'] < oo else 'oo'} beta={c.get('beta')} "
         f"({time.time() - t:.1f}s)")
    return c


PrimalHybrid.cost_zeta = classmethod(_cz_log)


def lb_log2(zeta, dist, mitm_):
    """Rigorous lower bound on log2(rop) of ANY primal-hybrid cost the estimator can report at this zeta (plain or
    Lemma 18 rotated guessing set, any hw, beta, d, Babai or projected CVP):
      rop >= ssf(|S|) * amplify(0.99, hit),  hit <= |S| * pmax(zeta)   (union bound; for rotations
      hit_rot = 1 - (1 - p)^N <= N p <= N |S_plain| pmax = |S_rot| pmax, and H1's |S| is larger still),
      amplify(0.99, p) >= max(1, 0.046 / p).
    => rop >= 0.046 / pmax (no MitM), sqrt(0.046 / pmax) (MitM).  pmax(zeta) = largest probability of one zeta-prefix
    in the estimator's own secret model (uniform over weight-h vectors, nonzeros uniform over 'base' values); it is
    non-increasing in zeta (marginalisation), so LB is non-decreasing and a cut-off is final."""
    n, hh = dist.n, dist.hamming_weight
    base = dist.bounds[1] - dist.bounds[0]
    lo, hi = max(0, zeta - n + hh), min(zeta, hh)
    best = None
    lg = lambda x: RR(x + 1).log_gamma()
    lC = lambda a, b: lg(a) - lg(b) - lg(a - b)
    for w in range(lo, hi + 1):
        # P(segment = v) for one v of weight w: C(n - zeta, h - w) / C(n, h) * base^-w
        lp = lC(n - zeta, hh - w) - lC(n, hh) - w * log(RR(base))
        best = lp if best is None or lp > best else best
    lpmax = float(best / log(RR(2)))
    l = float(log(RR(0.046), 2)) - lpmax
    if mode.startswith("hj"):
        # free rotations: |S| = |S_plain| but hit <= NROT |S_plain| pmax, so the bound loses log2(NROT)
        l -= float(log(RR(NROT), 2))
    return l / 2 if mitm_ else l


def cost_zeta_full(zeta, params, m_):
    """PrimalHybrid.cost_zeta without the early break over hw (all hw tried), then d optimised."""
    baseline = primal_usvp(params, optimize_d=False)
    max_beta = max_beta_global if baseline["rop"] == oo else baseline["beta"]
    f = partial(PrimalHybrid.cost, params=params, zeta=zeta, babai=babai, mitm=mitm, m=m_)
    min_hw = max(0, zeta - params.Xs.n + params.Xs.hamming_weight)
    max_hw = min(zeta, params.Xs.hamming_weight)
    cost = Cost(rop=oo)
    for hw in range(min_hw, max_hw + 1):
        S, hit = lwe_primal.guessing_set_and_hit_probability(zeta, params.Xs, hw)
        precision = 2
        with local_minimum(40, max_beta + precision, precision=precision) as it:
            for beta in it:
                it.update(f(beta, search_space=S, hit_probability=hit))
            for beta in it.neighborhood:
                it.update(f(beta, search_space=S, hit_probability=hit))
            c = it.y
        if c is not None and c["rop"] < cost["rop"]:
            cost = c
    nod = cost
    if cost["rop"] < oo:
        # d optimisation at the chosen beta (estimator step 2)
        with local_minimum(params.n - zeta, cost["d"] + 1) as it:
            for d in it:
                it.update(f(beta=cost["beta"], d=d))
            c = it.y
        if c is not None and c["rop"] < cost["rop"]:
            cost = c
    return cost, nod


def fl(c, k):
    v = c.get(k)
    return None if v is None else (float(RR(log(v, 2))) if k in ("rop", "|S|") else (int(v) if k != "prob" else float(v)))


t = time.time()
scan = os.environ.get("SCAN")
if mode.startswith("est:"):
    algo = mode[4:]
    allalg = ("arora-gb", "bkw", "usvp", "bdd", "bdd_hybrid", "bdd_mitm_hybrid", "dual", "dual_hybrid")
    if algo in ("bdd_hybrid", "bdd_mitm_hybrid"):
        # LWE.estimate's own partials for these two (lwe.py Estimate.__call__); called directly because its report
        # loop indexes res["bdd"] / res["bdd_hybrid"] and raises KeyError when those are denied.
        from estimator.lwe_primal import primal_hybrid
        from estimator.conf import red_cost_model as red_cost_model_default, red_shape_model as red_shape_model_default
        c = primal_hybrid(p.normalize(), mitm=(algo == "bdd_mitm_hybrid"), babai=(algo == "bdd_mitm_hybrid"),
                          red_cost_model=red_cost_model_default, red_shape_model=red_shape_model_default)
    else:
        r = LWE.estimate(p, deny_list=tuple(a for a in allalg if a != algo), jobs=1, quiet=True, catch_exceptions=False)
        c = r[algo]
    out = {"label": label, "logn": logn, "logq": logq, "secret": sec, "mode": mode, "sigma": sigma, "m": mm,
           "bits": float(RR(log(c["rop"], 2))) if c["rop"] < oo else None,
           "cost": {k: (str(v)) for k, v in c.items() if k != "problem"}, "s": round(time.time() - t, 1)}
    print(json.dumps(out), flush=True)
elif scan:
    # Exhaustive scan, complete by construction: zetas in increasing order (this shard's residue class); stop at the
    # first zeta with LB(zeta) > B, B = best cost found so far by ANY shard (SCANOUT shared) or cost(0). LB is
    # non-decreasing, so every later zeta costs > B >= the final minimum.
    lo, hi, shard, nsh = scan.split(":")
    lo, shard, nsh = int(lo), int(shard), int(nsh)
    hi = 10**9 if hi == "auto" else int(hi)
    out_path = os.environ["SCANOUT"]
    params = LWEParameters.normalize(p)
    m_ = params.m + params.n if params.Xs <= params.Xe else params.m
    c0, _ = cost_zeta_full(0, params, m_)
    U = float(RR(log(c0["rop"], 2))) if c0["rop"] < oo else float("inf")

    def emit(rec):
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    if shard == 0:
        emit({"label": label, "logq": logq, "secret": sec, "mode": mode, "zeta": 0, "bits": U, "beta": fl(c0, "beta"),
              "d": fl(c0, "d"), "s": round(time.time() - t, 1)})

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

    z = max(lo, 1)
    while z % nsh != shard:
        z += 1
    while z <= hi:
        B, lb = best(), lb_log2(z, params.Xs, mitm)
        if lb > B:
            emit({"label": label, "mode": mode, "stop": z, "shard": shard, "lb": lb, "B": B})
            plog(f"shard {shard} stops at zeta={z}: LB {lb:.2f} > B {B:.2f}")
            break
        tz = time.time()
        c, nod = cost_zeta_full(z, params, m_)
        rec = {"label": label, "logq": logq, "secret": sec, "mode": mode, "zeta": z,
               "bits": fl(c, "rop") if c["rop"] < oo else None, "bits_nod": fl(nod, "rop") if nod["rop"] < oo else None,
               "beta": fl(c, "beta"), "d": fl(c, "d"), "logS": fl(c, "|S|"), "lb": lb, "s": round(time.time() - tz, 1)}
        emit(rec)
        plog(f"zeta={z} {rec['bits']} (B {B:.2f}, LB {lb:.2f})")
        z += nsh
    print(json.dumps({"label": label, "mode": mode, "shard": shard, "done": True, "s": round(time.time() - t, 1)}),
          flush=True)
else:
    zeta_fix = int(os.environ["ZETA"]) if os.environ.get("ZETA") else None
    r = LWE.primal_hybrid(p, babai=babai, mitm=mitm, zeta=zeta_fix)
    out = {"label": label + (f"_z{zeta_fix}" if zeta_fix else ""), "logn": logn, "logq": logq, "secret": sec,
           "mode": mode, "sigma": sigma, "m": mm, "bits": float(RR(log(r["rop"], 2))), "beta": int(r.get("beta", 0)),
           "zeta": int(r.get("zeta", 0)), "logS": float(RR(log(r["|S|"], 2))) if "|S|" in r else None,
           "d": int(r.get("d", 0)), "s": round(time.time() - t, 1)}
    print(json.dumps(out), flush=True)
