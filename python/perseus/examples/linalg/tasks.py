"""Linear-algebra problems for perseus.impl.linalg: test matrices with a set condition number, the encrypted
algorithm as one stage, the float64 reference and the error metrics. A `Task` is what the runner (run.py) drives on
the plaintext mirror, the CPU twin (capture for the planner), the eager device session and the planned one."""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from perseus.impl import linalg as la
from perseus.impl.ops import NumpyOps

SLOTS = 1 << 15


def test_matrix(rng, m, n, kappa, spd=False, decay="geom"):
    """m x n with singular values from 1 down to 1/kappa (geometric, or `decay`="cluster": one small value), random
    orthogonal factors (SPD: U = V)."""
    k = min(m, n)
    s = np.logspace(0, -math.log10(kappa), k) if decay == "geom" else np.r_[np.ones(k - 1), 1.0 / kappa]
    U, _ = np.linalg.qr(rng.standard_normal((m, k)))
    if spd:
        return (U * s) @ U.T
    V, _ = np.linalg.qr(rng.standard_normal((n, k)))
    return (U * s) @ V.T


class RotRec(NumpyOps):
    """The mirror recording every rotation step (the session's key list)."""

    def __init__(self):
        self.steps = set()

    def rotate(self, a, k):
        self.steps.add(int(k))
        return super().rotate(a, k)

    def rotate_many(self, a, steps):
        self.steps.update(int(k) for k in steps)
        return super().rotate_many(a, steps)

    def rotate_and_sum(self, a, start, stop):
        g = abs(int(start))
        while g < stop:
            self.steps.add(g if start > 0 else -g)
            g *= 2
        return super().rotate_and_sum(a, start, stop)


class _D:
    """A value carrying its multiplicative depth (levels since its last refresh / since the input)."""
    __slots__ = ("d",)

    def __init__(self, d):
        self.d = d


class CountOps:
    """Counts the circuit: multiplicative depth (deepest value; refreshes reset it, see `deepest_window`), ciphertext
    products, plaintext products, rotations (a rotate-and-sum ladder counts its rungs)."""

    def __init__(self):
        self.n = dict(ct_mult=0, pt_mult=0, rot=0, bts=0)
        self.depth = self.window = 0

    def _o(self, d):
        self.depth = max(self.depth, d)
        return _D(d)

    @staticmethod
    def _d(x):
        return x.d if isinstance(x, _D) else 0

    def add(self, a, b): return _D(max(self._d(a), self._d(b)))
    sub = add

    def inplace_add(self, a, b): a.d = max(a.d, self._d(b))

    def mult(self, a, b):
        if isinstance(b, _D):
            self.n["ct_mult"] += 1
        return self._o(max(self._d(a), self._d(b)) + 1)

    def square(self, a): return self.mult(a, a)
    def negate(self, a): return _D(a.d)
    def copy(self, a): return _D(a.d)

    def mult_pt(self, a, v, **kw):
        self.n["pt_mult"] += 1
        return self._o(a.d + 1)

    def add_pt(self, a, v, **kw): return _D(a.d)

    def rotate(self, a, k):
        self.n["rot"] += 1
        return _D(a.d)

    def rotate_many(self, a, steps):
        self.n["rot"] += len(steps)
        return [_D(a.d) for _ in steps]

    def rotate_and_sum(self, a, start, stop):
        g = abs(int(start))
        while g < stop:
            self.n["rot"] += 1
            g *= 2
        return _D(a.d)

    def bootstrap(self, a):
        self.n["bts"] += 1
        self.window = max(self.window, a.d)
        a.d = 0
        return a


# device ms per call, one RTX PRO 6000, n32 chain (README "Runtime" table: relinearized product, plaintext product,
# rotation, rescale, dense / 512-slot sparse refresh)
UNIT_MS = {"ct_mult": 0.356, "pt_mult": 0.015, "rot": 0.284, "rescale": 0.047, "bts": 12.2, "bts512": 10.7}


class BreakdownOps(CountOps):
    """CountOps with the rotations split by where they come from (product ladders, face moves, consolidation)."""

    def __init__(self):
        super().__init__()
        self.rot_by = {"mm": 0, "relayout": 0, "consolidate": 0, "other": 0}

    def _site(self):
        import inspect
        for fr in inspect.stack()[2:8]:
            if fr.function in ("mm", "mm_pt"):
                return "mm"
            if fr.function == "apply":
                return "relayout"
            if fr.function == "consolidate":
                return "consolidate"
        return "other"

    def rotate(self, a, k):
        self.rot_by[self._site()] += 1
        return super().rotate(a, k)

    def rotate_many(self, a, steps):
        self.rot_by[self._site()] += len(steps)
        return super().rotate_many(a, steps)

    def rotate_and_sum(self, a, start, stop):
        g, c = abs(int(start)), 0
        while g < stop:
            c, g = c + 1, g * 2
        self.rot_by[self._site()] += c
        return super().rotate_and_sum(a, start, stop)


def breakdown(task, bts_ms=None):
    """{counts, modeled ms per category} of the task's circuit: UNIT_MS x counts (a product or plaintext product
    also pays a rescale). An upper-bound model: hoisted rotations and the replay's fusions are cheaper."""
    ops = BreakdownOps()
    task.run(ops, [_D(0) for _ in task.inputs()])
    n = dict(ops.n)
    u = UNIT_MS
    bts = n["bts"] * (bts_ms if bts_ms is not None else u["bts"])
    ms = {"bootstraps": bts, "ct_products": n["ct_mult"] * (u["ct_mult"] + u["rescale"]),
          "pt_products": n["pt_mult"] * (u["pt_mult"] + u["rescale"]),
          **{f"rot_{k}": v * u["rot"] for k, v in ops.rot_by.items()}}
    return {"counts": {**n, **{f"rot_{k}": v for k, v in ops.rot_by.items()}, "depth": ops.depth,
                       "window": ops.window}, "ms": ms, "total_ms": sum(ms.values())}


def circuit(task):
    """{depth, ct_mult, pt_mult, rot, bts (deliberate), window (deepest between deliberate refreshes)} of the task."""
    ops = CountOps()
    task.run(ops, [_D(0) for _ in task.inputs()])
    return {"depth": ops.depth, **ops.n, "window": max(ops.window, 0)}


@dataclasses.dataclass
class Task:
    """One problem instance: `inputs()` -> [(matrix, Lay)] the client encrypts, `run(ops, cts)` -> output cts,
    `outputs(vals)` -> {name: matrix} from the decrypted output slot vectors, `metrics(out)` -> {name: float}."""
    seed: int = 0

    name = "task"

    def encode(self):
        return [la.encode(M, lay) for M, lay in self.inputs()]

    def rotations(self):
        rec = RotRec()
        self.run(rec, self.encode())
        return sorted(s % SLOTS for s in rec.steps if s % SLOTS)

    def mirror(self):
        ops = NumpyOps()
        return self.outputs([np.asarray(v) for v in self.run(ops, self.encode())])

    def describe(self):
        return {k: v for k, v in dataclasses.asdict(self).items()}


def rel(X, Y):
    return float(np.linalg.norm(X - Y) / max(np.linalg.norm(Y), 1e-300))


@dataclasses.dataclass
class Inverse(Task):
    """A^{-1} by Newton-Schulz. `scaled` = d > 0: A = D^1/2 B D^1/2 (SPD) with B unit-diagonal of cond `kappa` and D
    log-spaced over d decades (a badly scaled matrix whose cond is ~10^d kappa). `precond`: "none" (public bounds of
    A itself), "jacobi" (diagonal equilibration, bounds of B), "frob" (start A^T / ||A||_F^2 encrypted, the public
    norm bound `loose` x too large). `refresh`: one consolidated refresh per step (else the planner's placement)."""
    n: int = 16
    kappa: float = 100.0
    spd: bool = False
    method: str = "ns"
    iters: int = 0           # 0: from the bounds
    polish: int = 1
    kappa_bound: float = 0.0  # the public cond bound the schedule uses (0: the true cond x 1.05)
    scaled: float = 0.0
    precond: str = "none"
    loose: float = 1.0
    refresh: bool = True
    fast: bool = False       # impl.linalg.inverse_fast (batched cube, accelerated schedule, wrap faces, period tag)
    batch: int = 1           # fast: problems per ciphertext (0: as many as fit, slots / n^3)
    accel: bool = True
    wrap: bool = True
    sparse: bool = False     # fast: tag the refreshed iterate's slot period (for a sparse-route plan)

    name = "inverse"

    def __post_init__(self):
        if self.fast:
            if not self.batch:
                self.batch = SLOTS // self.n ** 3
            rng = np.random.default_rng(self.seed)
            self.As = np.stack([test_matrix(rng, self.n, self.n, self.kappa, self.spd) for _ in range(self.batch)])
            self.A = self.As[0]
            self.cube = la.batched_cube(self.n, self.batch, SLOTS)
            self.lay = la.O(self.cube, (self.n, self.n), self.batch)
            self.kb = self.kappa_bound or 1.05 * self.kappa
            return
        rng = np.random.default_rng(self.seed)
        if self.scaled:
            B = test_matrix(rng, self.n, self.n, self.kappa, spd=True)
            d = np.sqrt(np.diag(B))
            self.B = B / np.outer(d, d)
            dd = 10.0 ** -rng.permutation(np.linspace(0.0, self.scaled, self.n))
            self.A = np.sqrt(np.outer(dd, dd)) * self.B
            self.A /= np.linalg.norm(self.A, 2) * 1.0001
            self.spd = True
        else:
            self.A = test_matrix(rng, self.n, self.n, self.kappa, self.spd)
        self.lay = la.flat(self.n, self.n, SLOTS)
        self.kb = self.kappa_bound or 1.05 * float(np.linalg.cond(self.A))

    def inputs(self):
        return [(self.As if self.fast else self.A, self.lay)]

    def run(self, ops, xs):
        if self.fast:
            z, self.s = la.inverse_fast(ops, xs[0], self.n, B=self.batch, kappa=self.kb, spd=self.spd,
                                        accel=self.accel, wrap=self.wrap, slots=SLOTS, refresh_iterate=self.refresh,
                                        tag_period=self.sparse)
            return [z]
        kw = dict(method=self.method, iters=self.iters or None, polish=self.polish, slots=SLOTS, src=self.lay,
                  refresh_iterate=self.refresh)
        if self.precond == "jacobi":
            dg = np.diag(self.A)
            Bh = self.A / np.sqrt(np.outer(dg, dg))
            w = np.linalg.eigvalsh(Bh)
            z, self.s = la.inverse_jacobi(ops, xs[0], self.n, dlo=dg.min() / 1.05, dhi=dg.max() * 1.05,
                                          bound=1.05 * w.max(), kappa=1.1 * w.max() / w.min(), **kw)
        elif self.precond == "frob":
            f = float(np.sum(self.A ** 2))
            z, self.s = la.inverse(ops, xs[0], self.n, kappa=self.kb, frob=(f / 4.0, 4.0 * f), **kw)
        else:
            z, self.s = la.inverse(ops, xs[0], self.n, bound=self.loose, kappa=self.kb * self.loose, spd=self.spd,
                                   **kw)
        return [z]

    def outputs(self, vals):
        if self.fast:
            return {"X": self.s * la.decode(vals[0], self.lay)}
        return {"X": self.s * la.decode(vals[0], la.O(la.full_cube(self.n, self.n, SLOTS), (self.n, self.n)))}

    def metrics(self, out):
        if self.fast:
            Xs = out["X"].reshape(self.batch, self.n, self.n)
            errs = [rel(X, np.linalg.inv(A)) for X, A in zip(Xs, self.As)]
            res = [float(np.linalg.norm(A @ X - np.eye(self.n), 2)) for X, A in zip(Xs, self.As)]
            return {"rel_err": float(np.max(errs)), "rel_err_med": float(np.median(errs)), "resid": float(np.max(res)),
                    "cond": float(np.linalg.cond(self.A))}
        X, A = out["X"], self.A
        Ainv = np.linalg.inv(A)
        I = np.eye(self.n)
        return {"rel_err": rel(X, Ainv), "resid": float(np.linalg.norm(A @ X - I, 2)),
                "cond": float(np.linalg.cond(A))}


def _lsq_problem(rng, m, n, kappa, noise=1e-3):
    A = test_matrix(rng, m, n, kappa)
    x = rng.standard_normal(n)
    x /= np.linalg.norm(x)
    b = A @ x + noise * rng.standard_normal(m) / math.sqrt(m)
    return A, b


def gaussian_sketch(rng, s, m):
    return rng.standard_normal((s, m)) / math.sqrt(s)


@dataclasses.dataclass
class Lstsq(Task):
    """min ||A x - b|| (+ ridge): normal equations + Newton-Schulz ("ns"), Newton-Schulz on the pseudoinverse itself
    ("pinv", no A^T A), or the Chebyshev semi-iteration ("cheb").
    `sketch` > 0: the CLIENT applies a Gaussian sketch S (sketch x m) to A and b before encrypting (sketch-and-solve:
    the server sees only S A, S b; residual within (1 + eps) of optimal for sketch ~ n / eps^2)."""
    m: int = 128
    n: int = 16
    kappa: float = 100.0
    ridge: float = 0.0
    method: str = "ns"
    iters: int = 0
    sketch: int = 0
    noise: float = 1e-3

    name = "lstsq"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.A, self.b = _lsq_problem(rng, self.m, self.n, self.kappa, self.noise)
        if self.sketch:
            Sk = gaussian_sketch(rng, self.sketch, self.m)
            self.As, self.bs = Sk @ self.A, Sk @ self.b
            nrm = np.linalg.norm(self.As, 2) * 1.05          # the client rescales to ||S A|| <= 1 (public)
            self.As, self.bs, self.scale = self.As / nrm, self.bs / nrm, nrm
        else:
            self.As, self.bs, self.scale = self.A, self.b, 1.0
        self.mm_ = self.As.shape[0]
        self.cube = la.gram_cube(self.n, SLOTS)

    def _precond(self):
        """The client's preconditioner: R of the QR of a Gaussian sketch of its A (4 n rows)."""
        rng = np.random.default_rng(7000 + self.seed)
        Sk = gaussian_sketch(rng, 4 * self.n, self.mm_)
        Rk = np.linalg.qr(Sk @ self.As)[1]
        return np.linalg.inv(Rk)

    def inputs(self):
        if self.method == "precond":
            Ri = self._precond()
            pc = la.full_cube(la.p2(self.mm_), self.n, SLOTS)
            return [(self.As, la.O(pc, (self.mm_, self.n))), (Ri, la.R(pc, (self.n, self.n))),
                    (Ri.T, la.R(self.cube, (self.n, self.n))), (self.bs.reshape(1, -1), la.L(self.cube, (1, self.mm_)))]
        if self.method == "pinv":
            return [(self.As, la.R(self.cube, (self.mm_, self.n))), (self.bs.reshape(1, -1), la.L(self.cube, (1, self.mm_)))]
        return [(self.As, la.R(self.cube, (self.mm_, self.n))), (self.bs.reshape(-1, 1), la.R(self.cube, (self.mm_, 1)))]

    def run(self, ops, xs):
        kb = self.kappa * (2.0 if self.sketch else 1.0)     # the sketch moves the singular values by (1 +- eps)
        if self.method == "cheb":
            lmax = 1.0
            lmin = 1.0 / kb ** 2 + self.ridge
            it = self.iters or int(math.ceil(math.sqrt(lmax / lmin) * math.log(2.0 / 1e-4) / 2.0))
            self.s = 1.0
            return [la.chebyshev_solve(ops, xs[0], xs[1], self.mm_, self.n, lmin, lmax, it, slots=SLOTS)]
        if self.method == "pinv":
            x, self.s = la.pinv_lstsq(ops, xs[0], xs[1], self.mm_, self.n, kappa=kb, slots=SLOTS)
            return [x]
        if self.method == "precond":
            x, self.s = la.precond_lstsq(ops, *xs, self.mm_, self.n, slots=SLOTS)
            return [x]
        x, self.s = la.lstsq(ops, xs[0], xs[1], self.mm_, self.n, kappa=kb, ridge=self.ridge, slots=SLOTS)
        return [x]

    def outputs(self, vals):
        if self.method in ("pinv", "precond"):
            return {"x": self.s * la.decode(vals[0], la.O(self.cube, (1, self.n)))[0]}
        x = self.s * la.decode(vals[0], la.O(self.cube, (self.n, 1)))[:, 0]
        return {"x": x}

    def metrics(self, out):
        A, b, x = self.A, self.b, out["x"]
        if self.ridge:
            xs = np.linalg.solve(A.T @ A + self.ridge * np.eye(self.n), A.T @ b)
        else:
            xs = np.linalg.lstsq(A, b, rcond=None)[0]
        res, res_opt = np.linalg.norm(A @ x - b), np.linalg.norm(A @ xs - b)
        return {"rel_err": rel(x, xs), "resid_ratio": float(res / res_opt), "cond": float(np.linalg.cond(A))}


@dataclasses.dataclass
class QR(Task):
    """Tall-skinny QR: "mgs" (modified Gram-Schmidt), "cqr" / "cqr2" (CholeskyQR, CholeskyQR2), "scqr3" (shifted
    CholeskyQR3), "polar" (the polar factor: orthonormal Q, symmetric "R" = (A^T A)^{1/2}; no triangular factor),
    "mgs2" / "polar2" (that, then one CholeskyQR pass on Q, whose Gram is ~I so its pivots sit in a narrow band; R =
    Q^T A from a second face of A)."""
    m: int = 128
    n: int = 16
    kappa: float = 100.0
    method: str = "cqr2"
    shift: float = 0.0
    tol: float = 1e-5

    name = "qr"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.A = test_matrix(rng, self.m, self.n, self.kappa)
        self.gc = la.gram_cube(self.n, SLOTS)
        self.pc = la.full_cube(la.p2(self.m), self.n, SLOTS)

    def inputs(self):
        if self.method in ("mgs", "polar", "polarns"):
            return [(self.A, la.O(self.pc, (self.m, self.n)))]
        if self.method in ("mgs2", "polar2"):
            return [(self.A, la.O(self.pc, (self.m, self.n))), (self.A, la.R(self.gc, (self.m, self.n)))]
        return [(self.A, la.R(self.gc, (self.m, self.n))), (self.A, la.L(self.pc, (self.m, self.n)))]

    def run(self, ops, xs):
        if self.method in ("mgs2", "polar2"):     # a first orthonormalization, one CholeskyQR pass on it, R = Q^T A
            if self.method == "mgs2":
                q = la.mgs_qr(ops, xs[0], self.m, self.n, kappa=self.kappa, slots=SLOTS, tol=self.tol)[0]
            else:
                q = la.orthonormalize(ops, xs[0], self.m, self.n, bound=1.0, kappa=self.kappa, slots=SLOTS)
            qr_, ql = la.qr_faces(ops, q, self.m, self.n, SLOTS)
            q, _, _ = la.cholesky_qr(ops, qr_, ql, self.m, self.n, lo=0.5, hi=1.5, slots=SLOTS, tol=self.tol)
            qr_, _ = la.qr_faces(ops, q, self.m, self.n, SLOTS)
            qt = la.relayout(ops, qr_, la.R(self.gc, (self.m, self.n)), la.L(self.gc, (self.n, self.m)), transpose=True)
            return [q, la.mm(ops, qt, xs[1], self.gc)]
        if self.method == "mgs":
            return list(la.mgs_qr(ops, xs[0], self.m, self.n, kappa=self.kappa, slots=SLOTS, tol=self.tol))
        if self.method in ("polar", "polarns"):   # Q = A (A^T A)^{-1/2}, "R" = Q^T A = (A^T A)^{1/2} (symmetric)
            q = la.orthonormalize(ops, xs[0], self.m, self.n, bound=1.0, kappa=self.kappa, slots=SLOTS,
                                  method="ns" if self.method == "polarns" else "gram")
            qr_, _ = la.qr_faces(ops, q, self.m, self.n, SLOTS)
            gc = self.gc
            qt = la.relayout(ops, qr_, la.R(gc, (self.m, self.n)), la.L(gc, (self.n, self.m)), transpose=True)
            ar = la.relayout(ops, xs[0], la.O(self.pc, (self.m, self.n)), la.R(gc, (self.m, self.n)))
            return [q, la.mm(ops, qt, ar, gc)]
        passes = 3 if self.method == "scqr3" else (1 if self.method == "cqr" else 2)
        return list(la.cholesky_qr2(ops, xs[0], xs[1], self.m, self.n, kappa=self.kappa, shift=self.shift,
                                    passes=passes, slots=SLOTS, tol=self.tol))

    def outputs(self, vals):
        Q = la.decode(vals[0], la.O(self.pc, (self.m, self.n)))
        if self.method == "mgs":
            Rm = la.decode(vals[1], la.O(self.pc, (self.m, self.n)))[:self.n]
        elif self.method in ("polar", "polarns", "mgs2", "polar2"):
            Rm = la.decode(vals[1], la.O(self.gc, (self.n, self.n)))
        else:
            Rm = la.decode(vals[1], la.O(self.gc, (self.n, self.n)))
        return {"Q": Q, "R": Rm}

    def metrics(self, out):
        Q, Rm, A = out["Q"], out["R"], self.A
        Rr = np.linalg.qr(A)[1]
        Rr = Rr * np.sign(np.diag(Rr))[:, None]
        if self.method in ("polar", "polarns"):
            U, sv, Vt = np.linalg.svd(A, full_matrices=False)
            Rr = (Vt.T * sv) @ Vt                      # (A^T A)^{1/2}
            r_err = rel(Rm, Rr)
        else:
            r_err = rel(np.triu(Rm), Rr)
        return {"orth": float(np.linalg.norm(Q.T @ Q - np.eye(self.n), 2)),
                "resid": float(np.linalg.norm(A - Q @ Rm, 2) / np.linalg.norm(A, 2)),
                "r_err": r_err, "cond": float(np.linalg.cond(A))}


def _angle(U, V):
    """The largest principal angle (radians) between span U and span V (orthonormal columns)."""
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return float(np.arccos(np.clip(s.min(), -1.0, 1.0)))


@dataclasses.dataclass
class PCA(Task):
    """Top-k principal subspace of N samples in d dimensions (centered covariance, subspace iteration with the polar
    factor). Data: a spiked model, k leading variances 1 / (1 + j gap) above an isotropic floor."""
    N: int = 512
    d: int = 16
    k: int = 4
    iters: int = 4
    floor: float = 0.05
    gap_kappa: float = 8.0       # a public bound on cond(C V) (~lambda_1 / lambda_k, margin for the start)
    tr_lo: float = 2e-3          # a public band of ||C V||_F^2 (the normalization's inverse square root range)
    tr_hi: float = 0.2
    orth_method: str = "ns"      # "ns": polar Newton-Schulz on Y; "gram": Y (Y^T Y)^{-1/2}
    bound_c: float = 0.15        # a public bound on ||C|| (polar_ns starts at Y / bound)

    name = "pca"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        U, _ = np.linalg.qr(rng.standard_normal((self.d, self.d)))
        var = np.r_[np.linspace(1.0, 0.4, self.k), np.full(self.d - self.k, self.floor)]
        X = rng.standard_normal((self.N, self.d)) * np.sqrt(var) @ U.T + rng.standard_normal(self.d)
        sc = np.abs(X).max() * 1.01
        self.X = X / sc / math.sqrt(1.0)                 # client normalization (public): entries <= 1
        self.gc = la.gram_cube(self.d, SLOTS)
        self.rows = self.gc.d[0]
        self.v0 = np.linalg.qr(np.random.default_rng(1000 + self.seed).standard_normal((self.d, self.k)))[0]
        C = np.cov(self.X.T, bias=True)
        self.C = C
        self.bound = float(np.trace(C)) * 0 + 1.0       # ||C|| <= max|x|^2 d / 4 ... checked below
        assert np.linalg.norm(C, 2) <= self.bound

    def inputs(self):
        out = []
        for b0 in range(0, self.N, self.rows):
            blk = np.zeros((self.rows, self.d))
            part = self.X[b0:b0 + self.rows]
            blk[:len(part)] = part
            out.append((blk, la.R(self.gc, (self.rows, self.d))))
        return out

    def run(self, ops, xs):
        c = la.covariance(ops, xs, self.N, self.d, slots=SLOTS)
        v = la.block_power(ops, c, self.d, self.k, self.iters, v0=self.v0,
                           bound=self.bound if self.orth_method == "gram" else self.bound_c,
                           gap_kappa=self.gap_kappa, slots=SLOTS, trace_band=(self.tr_lo, self.tr_hi),
                           orth=self.orth_method)
        return [v, c]

    def outputs(self, vals):
        pc = la.full_cube(self.d, self.k, SLOTS)
        return {"V": la.decode(vals[0], la.O(pc, (self.d, self.k))),
                "C": la.decode(vals[1], la.O(self.gc, (self.d, self.d)))}

    def metrics(self, out):
        V, C = out["V"], self.C
        w, U = np.linalg.eigh(C)
        Uk = U[:, ::-1][:, :self.k]
        Vo = np.linalg.qr(V)[0]
        ev = float(np.trace(Vo.T @ C @ Vo) / np.trace(Uk.T @ C @ Uk))
        return {"angle": _angle(Uk, Vo), "expl_var_ratio": ev, "orth": float(np.linalg.norm(V.T @ V - np.eye(self.k), 2)),
                "cov_err": rel(out["C"], C), "gap": float(w[::-1][self.k - 1] / w[::-1][self.k])}


@dataclasses.dataclass
class RSVD(Task):
    """Randomized SVD of A (m x n, rank-k signal + a floor): range finder with q power steps on the server, the
    small SVD of B = Q^T A on the client."""
    m: int = 64
    n: int = 32
    k: int = 4
    p: int = 4
    q: int = 1
    floor: float = 0.05
    kappa: float = 50.0
    orth_iters: int = 8
    orth_method: str = "ns"

    name = "rsvd"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        U, _ = np.linalg.qr(rng.standard_normal((self.m, self.m)))
        V, _ = np.linalg.qr(rng.standard_normal((self.n, self.n)))
        r = min(self.m, self.n)
        s = np.r_[np.linspace(1.0, 0.5, self.k), self.floor * np.linspace(1.0, 0.2, r - self.k)]
        self.A = (U[:, :r] * s) @ V[:, :r].T
        self.l = self.k + self.p
        self.p1 = la.full_cube(la.p2(self.m), self.l, SLOTS)
        self.p2 = la.full_cube(la.p2(self.n), self.l, SLOTS)
        self.omega = np.linalg.qr(np.random.default_rng(2000 + self.seed).standard_normal((self.n, self.l)))[0]

    def inputs(self):
        return [(self.A, la.L(self.p1, (self.m, self.n))), (self.A.T, la.L(self.p2, (self.n, self.m)))]

    def run(self, ops, xs):
        return list(la.rsvd(ops, xs[0], xs[1], self.m, self.n, self.l, self.q, omega=self.omega, bound=1.0,
                            kappa=self.kappa, slots=SLOTS, trace_band=(0.01 * self.l, float(self.l)),
                            orth_iters=self.orth_iters or None, orth=self.orth_method))

    def outputs(self, vals):
        Q = la.decode(vals[0], la.O(self.p1, (self.m, self.l)))
        Bt = la.decode(vals[1], la.O(self.p2, (self.n, self.l)))
        Qo, Rq = np.linalg.qr(Q)                                       # the client's finish: Q = Qo Rq,
        Ub, sb, Vbt = np.linalg.svd(np.linalg.solve(Rq.T, Bt.T), full_matrices=False)   # B' = Qo^T A = Rq^-T B
        return {"Q": Q, "U": Qo @ Ub[:, :self.k], "s": sb[:self.k], "V": Vbt[:self.k].T}

    def metrics(self, out):
        U, s, Vt = np.linalg.svd(self.A, full_matrices=False)
        Qo = np.linalg.qr(out["Q"])[0]
        return {"angle_U": _angle(U[:, :self.k], np.linalg.qr(out["U"])[0]),
                "angle_V": _angle(Vt[:self.k].T, np.linalg.qr(out["V"])[0]),
                "sv_rel_err": float(np.max(np.abs(out["s"] - s[:self.k]) / s[:self.k])),
                "range_resid": float(np.linalg.norm(self.A - Qo @ (Qo.T @ self.A), 2) / s[self.k]),
                "orth": float(np.linalg.norm(out["Q"].T @ out["Q"] - np.eye(self.l), 2))}


@dataclasses.dataclass
class Refresh(Task):
    """The chain's refresh error: a d x d matrix (uniform in [-amp, amp]) through `levels` plaintext products and one
    deliberate refresh (the error a refresh adds to a unit-size iterate)."""
    d: int = 16
    amp: float = 1.0
    levels: int = 3

    name = "refresh"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.M = self.amp * rng.uniform(-1, 1, (self.d, self.d))
        self.lay = la.flat(self.d, self.d, SLOTS)

    def inputs(self):
        return [(self.M, self.lay)]

    def run(self, ops, xs):
        x = xs[0]
        for _ in range(self.levels):
            x = ops.mult_pt(x, np.ones(SLOTS), kind="mask")
        ops.bootstrap(x)
        return [ops.mult_pt(x, np.ones(SLOTS), kind="mask")]

    def outputs(self, vals):
        return {"M": la.decode(vals[0], self.lay), "slots": np.asarray(vals[0])}

    def metrics(self, out):
        e = out["slots"] - la.encode(self.M, self.lay)
        return {"max_err": float(np.abs(e).max() / self.amp), "rms_err": float(np.sqrt(np.mean(e ** 2)) / self.amp)}


TASKS = {"refresh": Refresh, "inverse": Inverse, "lstsq": Lstsq, "qr": QR, "pca": PCA, "rsvd": RSVD}


def make(name, **kw):
    return TASKS[name](**kw)
