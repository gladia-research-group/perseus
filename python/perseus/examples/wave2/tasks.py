"""Problems for the wave-2 modules: inputs, the encrypted computation as one stage, the float64 reference and the
error metrics (examples/mathlib's Task protocol: `encode()`, `run(ops, cts)`, `outputs(vals)`, `metrics(out)`)."""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from perseus.examples.linalg.tasks import test_matrix, _lsq_problem, gaussian_sketch
from perseus.examples.mathlib.tasks import Task
from perseus.impl import linalg as la
from perseus.impl import solvers as S

SLOTS = 1 << 15


def _rel(x, y):
    return float(np.linalg.norm(np.asarray(x) - np.asarray(y)) / max(np.linalg.norm(y), 1e-300))


# ── 1. iterative solvers ─────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Solve(Task):
    """B SPD systems A_b x = b_b (n x n, eigenvalues geometric in [1/kappa, 1], ||b|| = 1) in one ciphertext.
    `method`: "cg" (CG, cg_plan on the public band, `tau`), "cgr" (CG + `refine` restarts on the recomputed residual),
    "cheb" (Chebyshev semi-iteration), "chebr" (+ `refine` passes of iterative refinement, solvers.refined), "pcg" / "pcheb" (the Jacobi-scaled system; `scaled` decades of diagonal
    scaling make A = D^1/2 A0 D^1/2 badly scaled), "ns" (linalg.inverse_fast, SPD start, then X b: the Newton-Schulz
    baseline, B <= slots / n^3). `eps`: the iteration count's target (cg_iters / cheb_iters); `iters` overrides."""
    n: int = 16
    B: int = 1
    kappa: float = 10.0
    method: str = "cheb"
    eps: float = 1e-3
    iters: int = 0
    tau: float = 0.1
    refine: int = 1
    recip_err: float = 1e-2
    scaled: float = 0.0
    rev: int = 2             # cheb: refresh (consolidated) every `rev` steps by hand; 0: the planner's placement
    rbound: float = 0.05     # chebr: the public bound of a pass's relative residual
    seed: int = 0

    name = "solve"

    def __post_init__(self):
        if self.B == 0:
            self.B = SLOTS // (self.n ** (3 if self.method == "ns" else 2))
        rng = np.random.default_rng(self.seed)
        As, D = [], []
        for _ in range(self.B):
            A0 = test_matrix(rng, self.n, self.n, self.kappa, spd=True)
            if self.scaled:
                d = 10.0 ** -rng.permutation(np.linspace(0.0, self.scaled, self.n))
                A = np.sqrt(np.outer(d, d)) * A0
                A /= np.linalg.eigvalsh(A).max() * 1.0001
                D.append(np.diag(A))
            else:
                A = A0
            As.append(A)
        self.As = np.stack(As)
        b = rng.standard_normal((self.B, self.n))
        self.bs = b / np.linalg.norm(b, axis=1, keepdims=True)
        self.lay = S.VecLay(self.n, self.B, SLOTS)
        ev = np.linalg.eigvalsh(self.As)
        self.lmin, self.lmax = float(ev.min()) / 1.05, float(ev.max()) * 1.0001   # the client's public band
        if self.scaled:
            D = np.stack(D)
            self.dlo, self.dhi = float(D.min()) / 1.05, float(D.max()) * 1.05
            Ah = self.As / np.sqrt(D[:, :, None] * D[:, None, :])
            evh = np.linalg.eigvalsh(Ah)
            self.hlo, self.hhi = float(evh.min()) / 1.05, float(evh.max()) * 1.05
            self.hb = float(np.max(np.linalg.norm(self.bs / np.sqrt(D), axis=1))) * 1.05
        if self.method == "ns":
            self.cube = la.batched_cube(self.n, self.B, SLOTS)

    def encode(self):
        if self.method == "ns":
            c = self.cube
            return [la.encode(self.As, la.O(c, (self.n, self.n), self.B)),
                    la.encode(self.bs[:, :, None], la.R(c, (self.n, 1), self.B, wrap=True))]
        return [self.lay.encode_mat(self.As), self.lay.encode_vec(self.bs)]

    def _plan(self, lmin, lmax, bnorm=1.0):
        return S.cg_plan(lmin, lmax, bnorm, iters=self.iters or None, eps=self.eps, tau=self.tau,
                         recip_err=self.recip_err)

    def run(self, ops, xs):
        a, b = xs
        m, lay = self.method, self.lay
        if m == "ns":
            n, c = self.n, self.cube
            X = la.inverse_fast(ops, a, n, B=self.B, bound=self.lmax, kappa=self.lmax / self.lmin, spd=True,
                                slots=SLOTS)
            xl = la.relayout(ops, X, la.O(c, (n, n), self.B), la.L(c, (n, n), self.B, wrap=True))
            return [la.mm(ops, xl, b, c, n)]
        if m == "cg":
            return [S.cg(ops, a, b, lay, self._plan(self.lmin, self.lmax))]
        if m == "cgr":
            p2 = self._plan(self.lmin, self.lmax, bnorm=0.05)
            return [S.cg_refined(ops, a, b, lay, self._plan(self.lmin, self.lmax), self.refine, p2)]
        if m == "chebr":
            def solve(o, rhs, bn):
                return S.chebyshev(o, a, rhs, lay, self.lmin, self.lmax, bnorm=bn, iters=self.iters or None,
                                   eps=self.eps, refresh_every=self.rev)
            return [S.refined(ops, solve, a, b, lay, self.refine, rbound=self.rbound)]
        if m == "cheb":
            return [S.chebyshev(ops, a, b, lay, self.lmin, self.lmax, iters=self.iters or None, eps=self.eps,
                                refresh_every=self.rev)]
        if m == "pcg":
            return [S.pcg_jacobi(ops, a, b, lay, dlo=self.dlo, dhi=self.dhi, lmin=self.hlo, lmax=self.hhi,
                                 bnorm=self.hb, iters=self.iters or None, eps=self.eps, tau=self.tau,
                                 recip_err=self.recip_err)]
        if m == "pcheb":
            return [S.pcg_jacobi(ops, a, b, lay, dlo=self.dlo, dhi=self.dhi, lmin=self.hlo, lmax=self.hhi,
                                 bnorm=self.hb, method="chebyshev", iters=self.iters or None, eps=self.eps,
                                 refresh_every=self.rev)]
        raise ValueError(m)

    def outputs(self, vals):
        if self.method == "ns":
            x = la.decode(np.real(vals[0]), la.O(self.cube, (self.n, 1), self.B))
            return {"x": np.asarray(x).reshape(self.B, self.n)}
        return {"x": self.lay.decode_vec(np.real(vals[0])).reshape(self.B, self.n)}

    def metrics(self, out):
        xs = np.linalg.solve(self.As, self.bs[:, :, None])[:, :, 0]
        errs = [_rel(x, t) for x, t in zip(out["x"], xs)]
        res = [float(np.linalg.norm(A @ x - b)) for A, x, b in zip(self.As, out["x"], self.bs)]
        return {"rel_err": float(np.max(errs)), "rel_err_med": float(np.median(errs)), "resid": float(np.max(res))}

    def describe(self):
        d = super().describe()
        if self.method in ("cg", "cgr"):
            p = self._plan(self.lmin, self.lmax)
            d.update(steps=p.iters, recip=f"{p.recip.method} depth {p.recip.depth}")
        elif self.method == "cheb":
            d.update(steps=self.iters or S.cheb_iters(self.lmax / self.lmin, self.eps))
        return d


@dataclasses.dataclass
class Lsq(Task):
    """min ||A x - b|| (m x n, cond `kappa`) through its normal equations G y = c solved iteratively:
    `method` "cg" / "cheb" with the client's sketch preconditioner (G = R^-T A^T A R^-1, x = R^-1 y: cond(G) ~ 3-10
    whatever kappa), "cg0" / "cheb0" on A^T A itself (cond kappa^2), against linalg's "pinv" / "precond" (Newton-Schulz
    on the pseudoinverse; the latter with the same client preconditioner)."""
    m: int = 128
    n: int = 16
    kappa: float = 100.0
    method: str = "cg"
    eps: float = 1e-3
    iters: int = 0
    tau: float = 0.1
    recip_err: float = 1e-2
    noise: float = 1e-3
    seed: int = 0

    name = "lsq"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.A, self.b = _lsq_problem(rng, self.m, self.n, self.kappa, self.noise)
        self.cube = la.gram_cube(self.n, SLOTS)
        self.lay = S.VecLay(self.n, 1, SLOTS)
        sk = np.random.default_rng(7000 + self.seed)
        Rk = np.linalg.qr(gaussian_sketch(sk, 4 * self.n, self.m) @ self.A)[1]
        self.Ri = np.linalg.inv(Rk)
        Bm = self.A @ self.Ri
        ev = np.linalg.eigvalsh(Bm.T @ Bm)
        # the client's public band of the preconditioned Gram: the sketch's (1 +- eps)^2 distortion, eps ~ 1/sqrt(4)
        self.plo, self.phi = float(ev.min()) / 1.2, float(ev.max()) * 1.2
        self.pb = float(np.linalg.norm(Bm.T @ self.b)) * 1.2
        sv = np.linalg.svd(self.A, compute_uv=False)
        self.glo, self.ghi = float(sv.min() ** 2) / 1.05, float(sv.max() ** 2) * 1.0001
        self.gb = float(np.linalg.norm(self.A.T @ self.b)) * 1.05

    def encode(self):
        g, pc = self.cube, la.full_cube(la.p2(self.m), self.n, SLOTS)
        a_r = la.encode(self.A, la.R(g, (self.m, self.n)))
        b_l = la.encode(self.b.reshape(1, -1), la.L(g, (1, self.m)))
        if self.method in ("cg", "cheb"):
            return [a_r, b_l, la.encode(self.Ri, la.R(pc, (self.n, self.n))), self.lay.encode_mat(self.Ri)]
        if self.method == "precond":
            return [la.encode(self.A, la.L(pc, (self.m, self.n))), la.encode(self.Ri, la.R(pc, (self.n, self.n))),
                    la.encode(self.Ri.T, la.R(g, (self.n, self.n))), b_l]
        return [a_r, b_l]

    def run(self, ops, xs):
        m, n, lay = self.m, self.n, self.lay
        if self.method == "pinv":
            return [la.pinv_lstsq(ops, xs[0], xs[1], m, n, kappa=self.kappa * 1.05, slots=SLOTS)]
        if self.method == "precond":
            return [la.precond_lstsq(ops, *xs, m, n, slots=SLOTS)]
        pre = self.method in ("cg", "cheb")
        G, c = S.normal_system(ops, xs[0], xs[1], m, n, lay, rinv_r=xs[2] if pre else None)
        lo, hi, bn = (self.plo, self.phi, self.pb) if pre else (self.glo, self.ghi, self.gb)
        if self.method in ("cg", "cg0"):
            y = S.cg(ops, G, c, lay, S.cg_plan(lo, hi, bn, iters=self.iters or None, eps=self.eps, tau=self.tau,
                                                recip_err=self.recip_err))
        else:
            y = S.chebyshev(ops, G, c, lay, lo, hi, bnorm=bn, iters=self.iters or None, eps=self.eps)
        if pre:          # refreshed by hand before the last product (planned in place, it drifts past the envelope)
            y = S.refresh(ops, y, lay)
        return [S.precond_back(ops, y, xs[3], lay) if pre else y]

    def outputs(self, vals):
        v = np.real(vals[0])
        if self.method in ("pinv", "precond"):
            return {"x": la.decode(v, la.O(self.cube, (1, self.n)))[0]}
        return {"x": self.lay.decode_vec(v)}

    def metrics(self, out):
        xs = np.linalg.lstsq(self.A, self.b, rcond=None)[0]
        x = out["x"]
        res, res_opt = np.linalg.norm(self.A @ x - self.b), np.linalg.norm(self.A @ xs - self.b)
        return {"rel_err": _rel(x, xs), "resid_ratio": float(res / res_opt)}


# ── 2. matrix functions ──────────────────────────────────────────────────────────────

def _spd(rng, n, kappa, lo=None):
    """SPD with eigenvalues geometric in [1 / kappa, 1]."""
    return test_matrix(rng, n, n, kappa, spd=True)


def _sqrtm_inv(A):
    w, V = np.linalg.eigh(A)
    return (V / np.sqrt(w)) @ V.T


@dataclasses.dataclass
class InvSqrt(Task):
    """A^{-1/2} of B SPD n x n matrices (eigenvalues geometric in [1/kappa, 1]): `method` "coupled" (matfun.inv_sqrt:
    scaled coupled Newton-Schulz, consolidated refreshes, `polish` Newton steps), "plain" (the same, unaccelerated,
    no polish) or "linalg" (linalg.inv_sqrt_spd as it was: B = 1, no consolidation)."""
    n: int = 8
    B: int = 1
    kappa: float = 100.0
    method: str = "coupled"
    polish: int = 1
    seed: int = 0

    name = "invsqrt"

    def __post_init__(self):
        from perseus.impl import matfun as MF
        if self.method == "linalg":
            self.B = 1
        rng = np.random.default_rng(self.seed)
        self.As = np.stack([_spd(rng, self.n, self.kappa) for _ in range(self.B)])
        self.F = MF.Faces(self.n, self.B, SLOTS)
        self.kb = self.kappa * 1.05

    def encode(self):
        return [self.F.encode(self.As)]

    def run(self, ops, xs):
        from perseus.impl import matfun as MF
        if self.method == "linalg":
            z = la.inv_sqrt_spd(ops, xs[0], self.n, bound=1.0, kappa=self.kb, slots=SLOTS, refresh_iterate=True)
            return [z]
        acc = self.method == "coupled"
        return [MF.inv_sqrt(ops, xs[0], self.F, bound=1.0, kappa=self.kb, accel=acc,
                            polish=self.polish if acc else 0)]

    def outputs(self, vals):
        return {"Z": self.F.decode(vals[0])}

    def metrics(self, out):
        errs, wh = [], []
        for Z, A in zip(out["Z"], self.As):
            errs.append(_rel(Z, _sqrtm_inv(A)))
            wh.append(float(np.linalg.norm(Z @ A @ Z - np.eye(self.n), 2)))
        return {"rel_err": float(np.max(errs)), "rel_err_med": float(np.median(errs)), "whiten_err": float(np.max(wh))}


@dataclasses.dataclass
class Whiten(Task):
    """Whitening of m samples x d features (centred, correlated, covariance cond `kappa`): X C^{-1/2}, C = X^T X / m
    (matfun.whiten). Metrics: ||cov(X W) - I||, ||W - C^{-1/2}|| relative."""
    m: int = 128
    d: int = 8
    kappa: float = 100.0
    polish: int = 1
    seed: int = 0

    name = "whiten"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        C = _spd(rng, self.d, self.kappa)
        Z = rng.standard_normal((self.m, self.d))
        Z -= Z.mean(0)
        Z = Z @ _sqrtm_inv(Z.T @ Z / self.m)                                  # exactly white
        self.X = Z @ np.linalg.cholesky(C).T                                  # cov = C exactly
        self.C = self.X.T @ self.X / self.m
        self.g = la.gram_cube(self.d, SLOTS)

    def encode(self):
        return [la.encode(self.X, la.R(self.g, (self.m, self.d)))]

    def run(self, ops, xs):
        from perseus.impl import matfun as MF
        xw, W = MF.whiten(ops, xs[0], self.m, self.d, bound=1.0001, kappa=self.kappa * 1.05, slots=SLOTS,
                          polish=self.polish)
        return [xw, W]

    def outputs(self, vals):
        pc = la.full_cube(la.p2(self.m), self.d, SLOTS)
        return {"XW": la.decode(np.real(vals[0]), la.O(pc, (self.m, self.d))),
                "W": la.decode(np.real(vals[1]), la.O(self.g, (self.d, self.d)))}

    def metrics(self, out):
        XW = out["XW"]
        return {"cov_err": float(np.linalg.norm(XW.T @ XW / self.m - np.eye(self.d), 2)),
                "W_rel_err": _rel(out["W"], _sqrtm_inv(self.C))}


@dataclasses.dataclass
class Sign(Task):
    """The spectral projector onto the eigenvalues of a symmetric n x n A above `mu` (matfun.spectral_projector, i.e.
    (I + sign(A - mu I)) / 2 by accelerated polar Newton-Schulz). Eigenvalues uniform in [-1, 1] outside a gap
    `gap` around mu (PUBLIC); `accel` false: the unaccelerated iteration."""
    n: int = 16
    mu: float = 0.0
    gap: float = 0.05
    accel: bool = True
    seed: int = 0

    name = "sign"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        k = self.n // 2
        lo = np.sort(rng.uniform(-1.0, self.mu - self.gap, k))
        hi = np.sort(rng.uniform(self.mu + self.gap, 1.0, self.n - k))
        lo[0], hi[-1], lo[-1], hi[0] = -1.0, 1.0, self.mu - self.gap, self.mu + self.gap
        U, _ = np.linalg.qr(rng.standard_normal((self.n, self.n)))
        self.w = np.r_[lo, hi]
        self.A = (U * self.w) @ U.T
        self.P = U[:, k:] @ U[:, k:].T
        self.lay = la.O(la.gram_cube(self.n, SLOTS), (self.n, self.n))

    def encode(self):
        return [la.encode(self.A, self.lay)]

    def run(self, ops, xs):
        from perseus.impl import matfun as MF
        g = self.gap / 1.05
        return [MF.spectral_projector(ops, xs[0], self.n, self.mu, gap=g, bound=1.0001, slots=SLOTS,
                                      accel=self.accel)]

    def outputs(self, vals):
        return {"P": la.decode(np.real(vals[0]), self.lay)}

    def metrics(self, out):
        P = out["P"]
        return {"proj_err": float(np.linalg.norm(P - self.P, 2)), "idem_err": float(np.linalg.norm(P @ P - P, 2))}


@dataclasses.dataclass
class Expm(Task):
    """exp(A) of B n x n matrices, ||A||_2 = `norm` (public bound 1.0001 x): `kind` "sym" (symmetric, eigenvalues
    uniform in [-norm, norm]) or "gen" (Gaussian, scaled to the norm). `method` "ss" (matfun.expm: scaling and
    squaring, Taylor degree / s from expm_plan at `err`) or "taylor" (no squaring: s = 0, the degree for `err`)."""
    n: int = 8
    B: int = 1
    norm: float = 2.0
    kind: str = "sym"
    method: str = "ss"
    err: float = 1e-6
    seed: int = 0

    name = "expm"

    def __post_init__(self):
        from perseus.impl import matfun as MF
        rng = np.random.default_rng(self.seed)
        As = []
        for _ in range(self.B):
            if self.kind == "sym":
                U, _ = np.linalg.qr(rng.standard_normal((self.n, self.n)))
                w = rng.uniform(-1, 1, self.n)
                w[0], w[-1] = -1.0, 1.0
                A = (U * (self.norm * w)) @ U.T
            else:
                A = rng.standard_normal((self.n, self.n))
                A *= self.norm / np.linalg.norm(A, 2)
            As.append(A)
        self.As = np.stack(As)
        self.F = MF.Faces(self.n, self.B, SLOTS)
        self.bound = self.norm * 1.0001
        if self.method == "taylor":
            self.deg, self.s = MF.expm_plan(self.bound, self.err, max_deg=40, max_s=0)
        else:
            self.deg, self.s = MF.expm_plan(self.bound, self.err)

    def encode(self):
        return [self.F.encode(self.As)]

    def run(self, ops, xs):
        from perseus.impl import matfun as MF
        return [MF.expm(ops, xs[0], self.F, bound=self.bound, deg=self.deg, s=self.s)]

    def outputs(self, vals):
        return {"E": self.F.decode(vals[0])}

    def metrics(self, out):
        from scipy.linalg import expm as ref
        errs = [_rel(E, ref(A)) for E, A in zip(out["E"], self.As)]
        return {"rel_err": float(np.max(errs)), "rel_err_med": float(np.median(errs))}

    def describe(self):
        return {**super().describe(), "deg": self.deg, "s": self.s}


@dataclasses.dataclass
class Power(Task):
    """Top-k eigenpairs of B symmetric PSD d x d matrices with a spiked spectrum (k eigenvalues `spikes` x a geometric
    ratio, the rest <= `floor`): `method` "power" (matfun.power_iteration with deflation, `iters` steps per pair) or
    "block" (linalg.block_power: subspace iteration with polar_ns, B = 1)."""
    d: int = 16
    B: int = 1
    k: int = 2
    iters: int = 12
    method: str = "power"
    floor: float = 0.1
    seed: int = 0

    name = "power"

    def __post_init__(self):
        from perseus.impl import solvers as S
        if self.method == "block":
            self.B = 1
        rng = np.random.default_rng(self.seed)
        As, V, W = [], [], []
        for _ in range(self.B):
            U, _ = np.linalg.qr(rng.standard_normal((self.d, self.d)))
            w = np.r_[[1.0 / 2 ** j for j in range(self.k)], self.floor * rng.uniform(0, 1, self.d - self.k)]
            As.append((U * w) @ U.T)
            V.append(U[:, :self.k])
            W.append(w[:self.k])
        self.As, self.V, self.W = np.stack(As), np.stack(V), np.stack(W)
        self.lay = S.VecLay(self.d, self.B, SLOTS)
        self.lam = [(0.95 / 2 ** j, 1.05 / 2 ** j) for j in range(self.k)]
        self.v0 = np.ones(self.d) / math.sqrt(self.d)

    def encode(self):
        if self.method == "block":
            return [la.encode(self.As[0], la.O(la.gram_cube(self.d, SLOTS), (self.d, self.d)))]
        return [self.lay.encode_mat(self.As)]

    def run(self, ops, xs):
        from perseus.impl import matfun as MF
        if self.method == "block":
            v0 = np.linalg.qr(np.random.default_rng(99).standard_normal((self.d, self.k)))[0]
            V = la.block_power(ops, xs[0], self.d, self.k, self.iters, v0=v0, bound=1.05,
                               gap_kappa=2 ** self.k * 1.1, slots=SLOTS, orth="ns")
            return [V]
        vs, ls = MF.power_iteration(ops, xs[0], self.lay, self.k, self.iters, lam=self.lam, v0=self.v0)
        return vs + ls

    def outputs(self, vals):
        if self.method == "block":
            V = la.decode(np.real(vals[0]), la.O(la.full_cube(self.d, self.k, SLOTS), (self.d, self.k)))
            return {"V": V[None], "lam": None}
        V = np.stack([self.lay.decode_vec(np.real(v)).reshape(self.B, self.d) for v in vals[:self.k]], -1)
        L = np.stack([self.lay.decode_vec(np.real(v)).reshape(self.B, self.d)[:, 0] for v in vals[self.k:]], -1)
        return {"V": V, "lam": L}

    def metrics(self, out):
        V = out["V"]
        ang = []
        for b in range(V.shape[0]):
            for j in range(self.k):
                c = abs(float(V[b, :, j] @ self.V[b, :, j])) / max(np.linalg.norm(V[b, :, j]), 1e-300)
                ang.append(math.acos(min(1.0, c)))
        m = {"angle_max": float(np.max(ang))}
        if self.method == "block":
            Q = np.linalg.qr(V[0])[0]
            s = np.linalg.svd(Q.T @ self.V[0], compute_uv=False)
            m = {"angle_max": float(math.acos(min(1.0, s.min())))}
        if out.get("lam") is not None:
            m["lam_rel_err"] = float(np.max(np.abs(out["lam"] - self.W) / self.W))
        return m


# ── 3. distances and clustering ──────────────────────────────────────────────────────

def _lloyd(X, Mu, iters):
    """(centroids after `iters` steps, the last step's assignment)."""
    a = None
    for _ in range(iters):
        D = ((X[None] - Mu[:, None]) ** 2).sum(-1)
        a = D.argmin(0)
        Mu = np.stack([X[a == c].mean(0) if (a == c).any() else Mu[c] for c in range(len(Mu))])
    return Mu, a


@dataclasses.dataclass
class KMeans(Task):
    """Lloyd's k-means of N points (d features, `data` "blobs": K Gaussian clusters, or "iris": the 4 features
    scaled to [-1, 1], N = 128 of 150, K = 4) from public-index initial centroids (K points the client picks,
    encrypted), `iters` iterations; `method` of the nearest-centroid comparison "tfhe" / "ckks". Metrics: centroid
    error vs float64 Lloyd from the same start, assignment agreement."""
    N: int = 1024
    K: int = 4
    d: int = 2
    iters: int = 3
    data: str = "blobs"
    spread: float = 0.08
    method: str = "tfhe"
    res: float = 2e-3
    seed: int = 0

    name = "kmeans"

    def __post_init__(self):
        from perseus.impl import cluster as CL
        from . import data as DS
        rng = np.random.default_rng(self.seed)
        if self.data == "iris":
            X, y = DS.load("iris")
            idx = rng.permutation(len(X))[:self.N]
            self.X, self.d = DS.scaled(X[idx]), X.shape[1]
        else:
            self.X, _y, _C = DS.blobs(rng, self.N, self.K, self.d, self.spread)
        pick = rng.choice(self.N, self.K, replace=False)
        self.Mu0 = self.X[pick] + 0.05 * rng.standard_normal((self.K, self.d))
        self.lay = CL.Points(self.N, self.d, self.K, SLOTS)
        self.nmin = max(1, self.N // (8 * self.K))
        self.tfhe = self.method == "tfhe"

    def encode(self):
        return [self.lay.encode_points(self.X), self.lay.encode_refs(self.Mu0)]

    def run(self, ops, xs):
        from perseus.impl import cluster as CL
        mu, a = CL.kmeans(ops, xs[0], xs[1], self.lay, self.iters, nmin=self.nmin, method=self.method, res=self.res)
        return [mu, a]

    def outputs(self, vals):
        a = self.lay.decode_dist(np.real(vals[1]))
        return {"mu": self.lay.decode_refs(vals[0]), "assign": a}

    def metrics(self, out):
        Mu, ap = _lloyd(self.X, self.Mu0, self.iters)
        got = out["assign"].argmax(0)
        Mp, _ = _lloyd(self.X, self.Mu0, self.iters - 1)
        D = np.sort(((self.X[None] - Mp[:, None]) ** 2).sum(-1) / (4.0 * self.d), 0)
        clear = D[1] - D[0] > 2 * self.res                               # outside the comparison's resolution
        return {"mu_err": float(np.abs(out["mu"] - Mu).max()), "agree": float((got == ap).mean()),
                "agree_clear": float((got == ap)[clear].mean()), "clear_frac": float(clear.mean()),
                "onehot_err": float(np.abs(out["assign"] - np.eye(self.K)[ap].T).max())}


@dataclasses.dataclass
class KNN(Task):
    """k-NN classification of one query (`seed` picks it from the held-out split) against N training points of
    `data` (breast_cancer: 30 features; the binary label y in {0, 1}); features scaled to [-1, 1] on the training
    split. The stage returns the vote (fraction of the k nearest labelled 1); metrics: the vote against float64 k-NN
    and the predicted class's agreement."""
    N: int = 128
    k: int = 5
    data: str = "breast_cancer"
    method: str = "ckks"
    seed: int = 0

    name = "knn"

    def __post_init__(self):
        from perseus.impl import cluster as CL
        from . import data as DS
        X, y = DS.load(self.data)
        rng = np.random.default_rng(1234)
        perm = rng.permutation(len(X))
        tr, te = perm[:self.N], perm[self.N:]
        self.Xt, self.yt = DS.scaled(X[tr]), (y[tr] > 0).astype(np.float64)
        self.q = DS.scaled(X[te[self.seed % len(te)]][None], ref=X[tr])[0].clip(-1.5, 1.5)
        self.yq = float(y[te[self.seed % len(te)]] > 0)
        self.d = X.shape[1]
        self.lay = CL.Points(self.N, self.d, 1, SLOTS)

    def encode(self):
        return [self.lay.encode_points(self.Xt, one=False), self.lay.encode_refs(self.q),
                self.lay.encode_labels(self.yt)]

    def run(self, ops, xs):
        from perseus.impl import cluster as CL
        v, _m = CL.knn(ops, xs[0], xs[1], xs[2], self.lay, self.k, xbound=1.5, method=self.method)
        return [v]

    def outputs(self, vals):
        return {"vote": float(np.real(vals[0])[0])}

    def metrics(self, out):
        D = ((self.Xt - self.q) ** 2).sum(1)
        nn = np.argsort(D)[:self.k]
        ref = float(self.yt[nn].mean())
        return {"vote_err": abs(out["vote"] - ref), "agree": float((out["vote"] > 0.5) == (ref > 0.5)),
                "correct": float((out["vote"] > 0.5) == (self.yq > 0.5))}


@dataclasses.dataclass
class Kernel(Task):
    """The N x N kernel matrix of N points (d features in [-1, 1]): `kind` "sqdist", "rbf" (gamma), "poly"
    (degree, c)."""
    N: int = 32
    d: int = 8
    kind: str = "rbf"
    gamma: float = 0.5
    degree: int = 3
    seed: int = 0

    name = "kernel"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.X = rng.uniform(-1, 1, (self.N, self.d))
        self.cube = la.full_cube(self.N, self.N, SLOTS)

    def encode(self):
        return [la.encode(self.X, la.L(self.cube, (self.N, self.d))), la.encode(self.X.T, la.R(self.cube, (self.d, self.N)))]

    def run(self, ops, xs):
        from perseus.impl import cluster as CL
        dmax = 4.0 * self.d
        if self.kind == "sqdist":
            return [CL.pairwise_sqdist(ops, *xs, self.N, self.d, SLOTS, scale=1.0 / dmax)]
        if self.kind == "rbf":
            return [CL.rbf_kernel(ops, *xs, self.N, self.d, SLOTS, gamma=self.gamma / dmax * 4, dmax=dmax)]
        return [CL.poly_kernel(ops, *xs, self.N, self.d, SLOTS, degree=self.degree, scale=1.0 / self.d)]

    def outputs(self, vals):
        return {"K": la.decode(np.real(vals[0]), la.O(self.cube, (self.N, self.N)))}

    def ref(self):
        X = self.X
        D = ((X[:, None] - X[None]) ** 2).sum(-1)
        dmax = 4.0 * self.d
        if self.kind == "sqdist":
            return D / dmax
        if self.kind == "rbf":
            return np.exp(-self.gamma / dmax * 4 * D)
        return (X @ X.T / self.d + 1.0) ** self.degree

    def metrics(self, out):
        R = self.ref()
        return {"max_err": float(np.abs(out["K"] - R).max()), "rel_err": _rel(out["K"], R)}


# ── 4. encrypted training ────────────────────────────────────────────────────────────

def _split(name, N, seed=0, binary_class=0):
    """(Xtr scaled to [-1, 1] / sqrt(d), ytr in {0, 1}, Xte, yte): N training rows (a power of two), the rest test;
    multi-class sets: class `binary_class` against the rest."""
    from . import data as DS
    X, y = DS.load(name)
    y = (y == binary_class).astype(np.float64) if name != "breast_cancer" else (y > 0).astype(np.float64)
    perm = np.random.default_rng(1000 + seed).permutation(len(X))
    tr, te = perm[:N], perm[N:]
    s = 1.0 / math.sqrt(X.shape[1] + 1)
    return DS.scaled(X[tr]) * s, y[tr], np.clip(DS.scaled(X[te], ref=X[tr]), -1.5, 1.5) * s, y[te]


def _acc(w, X, y):
    return float((((np.c_[X, np.ones(len(X))] @ w) > 0) == (y > 0.5)).mean())


@dataclasses.dataclass
class Train(Task):
    """GD training of a linear model (impl.learn.train) on `data` (breast_cancer: 512 of 569 rows train; wine / iris:
    class 0 against the rest, 128 rows): `loss` "logistic" / "svm", `epochs`, `lr`, Nesterov. Metrics: train / test
    accuracy, and the same schedule in float64 (with the exact sigmoid and with the encrypted run's approximation):
    the weights' distance to it and its accuracies."""
    data: str = "breast_cancer"
    N: int = 512
    loss: str = "logistic"
    epochs: int = 6
    lr: float = 8.0
    lam: float = 0.0
    nesterov: bool = True
    zb: float = 8.0
    err: float = 1e-3
    fbs: bool = False             # the link as a functional bootstrap of the margin (impl.learn Trainer.fbs)
    seed: int = 0

    name = "train"

    def __post_init__(self):
        from perseus.impl import learn as LN
        self.Xtr, self.ytr, self.Xte, self.yte = _split(self.data, self.N, self.seed)
        self.d = self.Xtr.shape[1]
        self.lay = LN.Design(self.N, self.d, 1, SLOTS)
        self.tr = LN.Trainer(self.loss, self.lr, self.lam, self.epochs, self.nesterov, self.zb, err=self.err,
                             fbs=self.fbs)

    def encode(self):
        return [self.lay.encode_points(self.Xtr), self.lay.encode_labels(self.ytr)]

    def run(self, ops, xs):
        from perseus.impl import learn as LN
        return [LN.train(ops, xs[0], xs[1], self.lay, self.tr)]

    def outputs(self, vals):
        return {"w": self.lay.decode_w(vals[0])}

    def metrics(self, out):
        from perseus.impl import learn as LN
        w = out["w"]
        wf = LN.train_numpy(self.Xtr, self.ytr, self.tr)
        ap = self.tr.link()
        wa = LN.train_numpy(self.Xtr, self.ytr, self.tr, link=ap.numpy)
        return {"acc_train": _acc(w, self.Xtr, self.ytr), "acc_test": _acc(w, self.Xte, self.yte),
                "float_acc_train": _acc(wf, self.Xtr, self.ytr), "float_acc_test": _acc(wf, self.Xte, self.yte),
                "w_err": float(np.abs(w - wa).max() / np.abs(wa).max()),
                "w_err_exact": float(np.abs(w - wf).max() / np.abs(wf).max()),
                "zmax": float(np.abs(np.c_[self.Xtr, np.ones(self.N)] @ wf).max())}


@dataclasses.dataclass
class Ridge(Task):
    """Ridge regression / classifier (labels {0, 1}) on `data` in closed form (impl.learn.ridge_normal): `method`
    "ns" (linalg.lstsq with the ridge), "cg", "cheb" (impl.solvers on X^T X / m + lam I, cond <= (1 + lam) / lam).
    m = N rows, d + 1 features (bias) padded to a power of two (N <= slots / n^2)."""
    data: str = "wine"
    N: int = 128
    lam: float = 1e-2
    method: str = "cheb"
    eps: float = 1e-4
    seed: int = 0

    name = "ridge"

    def __post_init__(self):
        Xtr, self.ytr, Xte, self.yte = _split(self.data, self.N, self.seed)
        self.d = Xtr.shape[1]
        self.Xtr, self.Xte = np.c_[Xtr, np.ones(self.N)], np.c_[Xte, np.ones(len(Xte))]
        self.n = la.p2(self.d + 1)
        self.g = la.gram_cube(self.n, SLOTS)
        self.bound = float(np.linalg.norm(self.Xtr, 2) ** 2 / self.N) * 1.05
        self.lay = S.VecLay(self.n, 1, SLOTS)

    def encode(self):
        Xp = np.zeros((self.N, self.n))
        Xp[:, :self.d + 1] = self.Xtr
        return [la.encode(Xp, la.R(self.g, (self.N, self.n))), la.encode(self.ytr.reshape(1, -1), la.L(self.g, (1, self.N)))]

    def run(self, ops, xs):
        from perseus.impl import learn as LN
        return [LN.ridge_normal(ops, xs[0], xs[1], self.N, self.n, self.lam, bound=self.bound, slots=SLOTS,
                                method=self.method, eps=self.eps, lay=self.lay)]

    def outputs(self, vals):
        v = np.real(vals[0])
        w = la.decode(v, la.O(self.g, (self.n, 1)))[:, 0] if self.method == "ns" else self.lay.decode_vec(v)
        return {"w": w[:self.d + 1]}

    def metrics(self, out):
        X, y = self.Xtr, self.ytr
        wf = np.linalg.solve(X.T @ X / self.N + self.lam * np.eye(self.d + 1), X.T @ y / self.N)
        acc = lambda w, X, y: float((((X @ w) > 0.5) == (y > 0.5)).mean())
        return {"w_rel_err": _rel(out["w"], wf), "acc_train": acc(out["w"], X, y), "acc_test": acc(out["w"], self.Xte, self.yte),
                "float_acc_test": acc(wf, self.Xte, self.yte)}


# ── 5. signal processing ─────────────────────────────────────────────────────────────

def _smooth(rng, B, n, modes=6):
    """B smooth periodic signals of length n (a few low Fourier modes)."""
    t = np.arange(n) / n
    x = np.zeros((B, n))
    for k in range(1, modes + 1):
        a, b = rng.standard_normal((2, B, 1)) / k
        x += a * np.cos(2 * np.pi * k * t) + b * np.sin(2 * np.pi * k * t)
    return x


@dataclasses.dataclass
class Signal(Task):
    """One impl.signal operation as a stage.
    kind "cconv": a length-65536 negacyclic convolution of a coefficient-encoded signal with a public filter (`taps`
      nonzero taps; method "coeff": one plaintext product; "dft": the cyclic convolution of the same data split in
      `n`-length blocks through fourier.dft / idft (the slot-domain alternative)).
    kind "xcorr": cross-correlation of two encrypted signals: method "coeff" (length 65536 negacyclic, x conj(y)),
      "dft" (batched length-n cyclic, DFT / conj product / IDFT).
    kind "diff": the m-th derivative of smooth periodic signals (length n, batched): method "fd" (central stencil,
      accuracy `order`) or "spectral".
    kind "conv2d": a `ksize` x `ksize` Gaussian (or "sobel") filter on r x r images (r^2 = n): method "direct" /
      "fft"."""
    kind: str = "cconv"
    method: str = "coeff"
    n: int = 1024
    taps: int = 16
    m: int = 1
    order: int = 4
    ksize: int = 3
    filt: str = "gauss"
    levels: int = 0
    seed: int = 0

    name = "signal"
    complex_out = True

    def __post_init__(self):
        from perseus.impl import fourier as F
        rng = np.random.default_rng(self.seed)
        N = 2 * SLOTS
        lv = self.levels or None
        if self.kind in ("cconv", "xcorr") and self.method == "coeff":
            self.x = rng.standard_normal(N) / math.sqrt(N)
            if self.kind == "cconv":                       # the public filter is part of the program: seed-free
                self.h = np.zeros(N)
                self.h[:self.taps] = np.random.default_rng(77).standard_normal(self.taps) / math.sqrt(self.taps)
            else:
                self.y = np.roll(self.x, 37) + 0.1 * rng.standard_normal(N) / math.sqrt(N)
            return
        if self.kind == "conv2d":
            r = int(round(math.sqrt(self.n)))
            self.r = r
            self.lay = F.Layout(self.n, SLOTS, SLOTS // self.n, "stride")
            B = self.lay.batch
            self.x = rng.uniform(0, 1, (B, r, r))
            if self.filt == "sobel":
                self.ker = np.array([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4
            else:
                g = np.exp(-0.5 * ((np.arange(self.ksize) - self.ksize // 2) / max(self.ksize / 4, 0.5)) ** 2)
                self.ker = np.outer(g, g) / np.outer(g, g).sum()
            if self.method == "fft":
                self.fwd = F.dft_2d(r, lv, layout=self.lay)
                self.inv = F.dft_2d(r, lv, inverse=True, layout=self.lay)
            return
        self.lay = F.Layout(self.n, SLOTS, SLOTS // self.n, "stride")
        B = self.lay.batch
        if self.kind == "cconv":                               # the dft alternative: blocks of n, cyclic
            self.x = rng.standard_normal((B, self.n)) / math.sqrt(self.n)
            self.h = np.zeros(self.n)
            self.h[:self.taps] = np.random.default_rng(77).standard_normal(self.taps) / math.sqrt(self.taps)
        elif self.kind == "xcorr":
            self.x = (rng.standard_normal((B, self.n)) + 1j * rng.standard_normal((B, self.n))) / math.sqrt(self.n)
            self.y = np.roll(self.x, 5, axis=1) + 0.1 * rng.standard_normal((B, self.n)) / math.sqrt(self.n)
        else:
            self.x = _smooth(rng, B, self.n)
        if self.kind in ("cconv", "xcorr") or self.method == "spectral":
            self.fwd = F.dft(self.n, lv, layout=self.lay)
            self.inv = F.idft(self.n, lv, layout=self.lay)

    def encode(self):
        from perseus.impl import signal as SG
        if self.method == "coeff":
            if self.kind == "cconv":
                return [SG.coeffs_to_slots(self.x)]
            return [SG.coeffs_to_slots(self.x), SG.coeffs_to_slots(self.y)]
        if self.kind == "conv2d":
            return [self.lay.pack(self.x.reshape(len(self.x), -1))]
        if self.kind == "xcorr":
            return [self.lay.pack(self.x), self.lay.pack(self.y)]
        return [self.lay.pack(self.x)]

    def run(self, ops, xs):
        from perseus.impl import signal as SG
        k, m = self.kind, self.method
        if m == "coeff":
            return [SG.cconv(ops, xs[0], self.h) if k == "cconv" else SG.ccorr(ops, xs[0], xs[1])]
        if k == "cconv":
            H = np.repeat(np.fft.fft(self.h), self.lay.batch)
            X = self.fwd.apply(ops, xs[0])
            return [self.inv.apply(ops, ops.mult_pt(X, H, key=f"signal.cconvh.{self.n}", kind="weight"))]
        if k == "xcorr":
            return [SG.xcorr_dft(ops, xs[0], xs[1], self.fwd, self.inv)]
        if k == "diff":
            if m == "fd":
                return [SG.fdiff(ops, xs[0], self.lay, self.m, self.order, h=1.0 / self.n)]
            return [SG.spectral_diff(ops, xs[0], self.fwd, self.inv, self.m, L=1.0)]
        if m == "direct":
            return [SG.conv2d_direct(ops, xs[0], self.ker, self.r, self.lay)]
        return [SG.conv2d_fft(ops, xs[0], self.ker, self.r, self.fwd, self.inv)]

    def outputs(self, vals):
        from perseus.impl import signal as SG
        v = np.asarray(vals[0])
        if self.method == "coeff":
            return {"y": SG.slots_to_coeffs(v)}
        y = self.lay.unpack(v)
        return {"y": y if self.kind == "xcorr" else np.real(y)}

    def ref(self):
        from perseus.impl import fourier as F
        from perseus.impl import signal as SG
        k = self.kind
        if self.method == "coeff":
            return F.negacyclic_conv(self.x, self.h) if k == "cconv" else SG.negacyclic_corr(self.x, self.y)
        if k == "cconv":
            return np.real(np.fft.ifft(np.fft.fft(self.x, axis=1) * np.fft.fft(self.h), axis=1))
        if k == "xcorr":
            return np.fft.ifft(np.conj(np.fft.fft(self.x, axis=1)) * np.fft.fft(self.y, axis=1), axis=1)
        if k == "conv2d":
            return SG.conv2d_ref(self.x, self.ker).reshape(len(self.x), -1)
        # exact derivative of the low-mode signals: spectral on the float64 data
        kk = np.fft.fftfreq(self.n, d=1.0 / self.n)
        return np.real(np.fft.ifft((2j * np.pi * kk) ** self.m * np.fft.fft(self.x, axis=1), axis=1))

    def metrics(self, out):
        R = self.ref()
        e = np.abs(np.asarray(out["y"]) - R)
        rms = float(np.sqrt(np.mean(np.abs(R) ** 2)))
        return {"err_max": float(e.max() / rms), "err_rms": float(np.sqrt(np.mean(e ** 2)) / rms)}

    def describe(self):
        d = {k: v for k, v in dataclasses.asdict(self).items()}
        if hasattr(self, "fwd"):
            d["levels_dft"] = self.fwd.levels + self.inv.levels
        return d


TASKS = {"solve": Solve, "lsq": Lsq, "invsqrt": InvSqrt, "whiten": Whiten, "sign": Sign, "expm": Expm,
         "power": Power, "kmeans": KMeans, "knn": KNN, "kernel": Kernel,
         "train": Train, "ridge": Ridge, "signal": Signal}


def make(name, **kw):
    return TASKS[name](**kw)
