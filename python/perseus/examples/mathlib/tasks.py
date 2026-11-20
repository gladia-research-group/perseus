"""Problems for impl.math / impl.stats / impl.sort: the inputs, the encrypted computation as one stage, the float64
reference and the error metrics. A `Task` is what run.py drives on the plaintext mirror, the CPU twin (capture for the
planner) and the planned device session."""
from __future__ import annotations

import dataclasses
import math as _m

import numpy as np

from perseus.impl import math as M
from perseus.impl import sort as T
from perseus.impl import stats as S
from perseus.impl.ops import NumpyOps

SLOTS = 1 << 15


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


class Task:
    """`encode()` -> slot vectors the client encrypts, `run(ops, cts)` -> output cts, `outputs(vals)` -> {name: array}
    from the decrypted slot vectors, `metrics(out)` -> {name: float}. `tfhe`: the stage switches to TFHE."""
    seed: int = 0
    name = "task"
    tfhe = False

    def rotations(self):
        rec = RotRec()
        self.run(rec, self.encode())
        return sorted({s % SLOTS for s in rec.steps if s % SLOTS})

    def mirror(self):
        return self.outputs([np.asarray(v) for v in self.run(NumpyOps(), self.encode())])

    def describe(self):
        return {k: v for k, v in dataclasses.asdict(self).items()}


def _err(y, t, rel=False):
    e = np.abs(np.asarray(y) - np.asarray(t))
    if rel:
        e = e / np.maximum(np.abs(t), 1e-300)
    return float(e.max()), float(np.sqrt(np.mean(e * e)))


_REF = {"exp": np.exp, "log": np.log, "sqrt": np.sqrt, "rsqrt": lambda x: 1.0 / np.sqrt(x),
        "reciprocal": lambda x: 1.0 / x, "tanh": np.tanh, "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
        "erf": None, "abs": np.abs, "relu": lambda x: np.maximum(x, 0.0), "sign": np.sign,
        "clip": None, "pow": None}


@dataclasses.dataclass
class Fn(Task):
    """One impl.math function on every slot: x uniform on [lo, hi] (a few slots pinned to the ends)."""
    fn: str = "exp"
    lo: float = -8.0
    hi: float = 0.0
    err: float = 1e-4
    p: float = 1.5
    method: str = "ckks"          # sign family: ckks / tfhe
    seed: int = 0
    name = "fn"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.x = rng.uniform(self.lo, self.hi, SLOTS)
        self.x[:2] = (self.lo, self.hi)
        self.tfhe = self.method == "tfhe" and self.fn in ("abs", "relu", "sign", "clip")

    def encode(self):
        return [self.x]

    def ref(self, x):
        if self.fn == "erf":
            from scipy.special import erf
            return erf(x)
        if self.fn == "pow":
            return x ** self.p
        if self.fn == "clip":
            return np.clip(x, 0.5 * self.lo, 0.5 * self.hi)
        return _REF[self.fn](x)

    def approx(self):
        kw = {"p": self.p} if self.fn == "pow" else {}
        return M.approx(self.fn, self.lo, self.hi, self.err, **kw)

    def run(self, ops, xs):
        x = xs[0]
        B = max(_m.fabs(self.lo), _m.fabs(self.hi))
        if self.fn == "abs":
            return [M.abs(ops, x, B, self.err, self.method)]
        if self.fn == "relu":
            return [M.relu(ops, x, B, self.err, self.method)]
        if self.fn == "sign":
            return [M.sign(ops, x, B, self.err, self.method)]
        if self.fn == "clip":
            return [M.clip(ops, x, 0.5 * self.lo, 0.5 * self.hi, B, self.err, self.method)]
        return [self.approx()(ops, x)]

    def outputs(self, vals):
        return {"y": np.real(vals[0])}

    def metrics(self, out):
        y, t = out["y"], self.ref(self.x)
        if self.fn == "sign":            # outside the gap only
            keep = np.abs(self.x) >= self.err * max(_m.fabs(self.lo), _m.fabs(self.hi))
            y, t = y[keep], t[keep]
        e, r = _err(y, t)
        rel = self.fn in ("exp", "rsqrt", "reciprocal", "pow")
        er, rr = _err(y, t, True) if rel and np.all(np.abs(t) > 0) else (e, r)
        return {"max_err": e, "rms_err": r, "max_rel": er, "rms_rel": rr}

    def describe(self):
        d = super().describe()
        if self.fn not in ("abs", "relu", "sign", "clip"):
            a = self.approx()
            d.update(method_chosen=a.method, depth=a.depth, ct_mults=a.ct_mults, params=dict(a.params),
                     err_float=a.err)
        return d


@dataclasses.dataclass
class Stats(Task):
    """impl.stats on B vectors of n values (Batch layout) or m x d samples (Features)."""
    kind: str = "zscore"          # mean / var / zscore / softmax / logsumexp / prefix / cov / corr
    n: int = 256
    B: int = 128
    m: int = 256
    d: int = 16
    err: float = 1e-3
    shift: str = "max"
    method: str = "tfhe"
    seed: int = 0
    name = "stats"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        if self.kind in ("cov", "corr"):
            self.B = min(self.B, SLOTS // (2 * self.m * self.d))
            self.lay = S.Features(self.m, self.d, self.B)
            mix = rng.standard_normal((self.d, self.d)) / _m.sqrt(self.d)
            scale = rng.uniform(0.5, 1.5, self.d)
            self.X = (rng.standard_normal((self.B, self.m, self.d)) @ mix) * scale + rng.uniform(-1, 1, self.d)
            self.X *= 0.5
        else:
            self.lay = S.Batch(self.n, self.B, pad=self.kind == "prefix")
            lo, hi = (-4.0, 4.0) if self.kind in ("softmax", "logsumexp") else (-2.0, 2.0)
            self.X = rng.uniform(lo, hi, (self.B, self.n)) * rng.uniform(0.3, 1.0, (self.B, 1))
            if self.kind == "prefix":
                self.X /= _m.sqrt(self.n)
        self.tfhe = self.kind in ("softmax", "logsumexp") and self.shift == "max" and self.method == "tfhe"

    def encode(self):
        return [self.lay.encode(self.X)]

    def _var_band(self, X, axis):
        v = X.var(axis=axis)
        return 0.5 * float(v.min()), 2.0 * float(v.max())

    def run(self, ops, xs):
        x, lay = xs[0], self.lay
        k = self.kind
        if k == "mean":
            return [S.mean(ops, x, lay)]
        if k == "var":
            return [S.variance(ops, x, lay)]
        if k == "zscore":
            lo, hi = self._var_band(self.X, 1)
            return [S.zscore(ops, x, lay, lo, hi, self.err)]
        if k == "softmax":
            return [S.softmax(ops, x, lay, -4.0, 4.0, self.err, shift=self.shift, method=self.method)]
        if k == "logsumexp":
            return [S.logsumexp(ops, x, lay, -4.0, 4.0, self.err, shift=self.shift, method=self.method)]
        if k == "prefix":
            return [S.prefix_sum(ops, x, lay)]
        if k == "cov":
            return S.covariance(ops, x, lay)
        if k == "corr":
            lo, hi = self._var_band(self.X, 1)
            return S.correlation(ops, x, lay, lo * (self.m - 1) / self.m, hi, self.err)
        raise KeyError(k)

    def outputs(self, vals):
        if self.kind in ("cov", "corr"):
            return {"C": self.lay.assemble([self.lay.decode_feature_row(v) for v in vals])}
        return {"y": self.lay.decode(vals[0])}

    def reference(self):
        X, k = self.X, self.kind
        if k == "mean":
            return X.mean(1, keepdims=True) * np.ones_like(X)
        if k == "var":
            return X.var(1, keepdims=True) * np.ones_like(X)
        if k == "zscore":
            return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
        if k == "softmax":
            e = np.exp(X - X.max(1, keepdims=True))
            return e / e.sum(1, keepdims=True)
        if k == "logsumexp":
            return np.log(np.exp(X).sum(1, keepdims=True)) * np.ones_like(X)
        if k == "prefix":
            return np.cumsum(X, 1)
        if k == "cov":
            return np.stack([np.cov(Xb.T) for Xb in X])
        if k == "corr":
            return np.stack([np.corrcoef(Xb.T) for Xb in X])
        raise KeyError(k)

    def metrics(self, out):
        y = out["C"] if self.kind in ("cov", "corr") else out["y"]
        e, r = _err(y, self.reference())
        return {"max_err": e, "rms_err": r}


@dataclasses.dataclass
class Sort(Task):
    """impl.sort on B vectors of n distinct values in [-1, 1] (a shuffled grid plus jitter: the gap between
    neighbours is >= `gap` x (2 / n), so every comparison is outside the back ends' bands unless gap ~ 0)."""
    kind: str = "rank"            # max / argmax / rank / topk / kth / median / quantile / sort / bitonic / threshold
    n: int = 16
    B: int = 1
    method: str = "ckks"
    k: int = 4
    q: float = 0.3
    probes: int = 10
    gap: float = 0.5
    seed: int = 0
    name = "sort"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        g = np.stack([rng.permutation(self.n) for _ in range(self.B)]).astype(np.float64)
        jit = rng.uniform(-0.5, 0.5, (self.B, self.n)) * (1.0 - self.gap)
        self.X = (g + 0.5 + jit) * (2.0 / self.n) - 1.0
        self.lay = S.Batch(self.n, self.B)
        self.tfhe = self.method == "tfhe"
        self.res = 0.25 * self.gap * 2.0 / self.n

    def encode(self):
        return [self.lay.encode(self.X)]

    def run(self, ops, xs):
        x, lay, k = xs[0], self.lay, self.kind
        bound, meth, res = 2.0, self.method, self.res
        if k == "max":
            return [T.vmax(ops, x, lay, bound, method=meth, err=res)]
        if k == "argmax":
            return [T.argmax(ops, x, lay, bound, method=meth, res=res, err=0.25 * res)]
        if k == "bitonic":
            return list(T.bitonic(ops, x, lay, bound, method=meth, res=res,
                                  payload=ops.add_pt(ops.mult_pt(x, np.zeros(SLOTS), kind="mask"),
                                                     lay.encode(np.tile(np.arange(self.n) / self.n, (self.B, 1))),
                                                     kind="mask")))
        if k == "threshold":
            t, mask = T.threshold(ops, x, lay, self.k, -1.0, 1.0, probes=self.probes, method=meth)
            return [t, mask]
        r = T.rank(ops, x, lay, bound, method=meth, res=res)
        if k == "rank":
            return [r]
        if k == "topk":
            return [T.topk_mask(ops, r, lay, self.k, method=meth)]
        if k == "kth":
            return [T.kth(ops, x, r, lay, self.k, method=meth)]
        if k == "median":
            return [T.median(ops, x, r, lay, method=meth)]
        if k == "quantile":
            return [T.quantile(ops, x, r, lay, self.q, method=meth)]
        if k == "sort":
            return [T.sort_by_rank(ops, x, r, lay, method=meth)]
        raise KeyError(k)

    def outputs(self, vals):
        return {f"y{i}": self.lay.decode(v) for i, v in enumerate(vals)}

    def reference(self):
        X, k, n = self.X, self.kind, self.n
        one = np.ones((1, n))
        if k == "max":
            return [X.max(1, keepdims=True) * one]
        if k == "argmax":
            return [(X == X.max(1, keepdims=True)).astype(float)]
        if k == "rank":
            return [np.argsort(np.argsort(X, 1), 1).astype(float)]
        if k == "topk":
            return [(np.argsort(np.argsort(X, 1), 1) >= n - self.k).astype(float)]
        if k == "kth":
            return [np.sort(X, 1)[:, self.k:self.k + 1] * one]
        if k == "median":
            return [np.median(X, 1, keepdims=True) * one]
        if k == "quantile":
            return [np.quantile(X, self.q, 1, keepdims=True) * one]
        if k == "sort":
            return [np.sort(X, 1)]
        if k == "bitonic":
            return [np.sort(X, 1), np.argsort(X, 1) / n]
        if k == "threshold":
            kth = np.sort(X, 1)[:, n - self.k:n - self.k + 1]
            return [kth * one, (np.argsort(np.argsort(X, 1), 1) >= n - self.k).astype(float)]
        raise KeyError(k)

    def metrics(self, out):
        ref = self.reference()
        ys = [out[f"y{i}"] for i in range(len(ref))]
        e, r = _err(ys[0], ref[0])
        res = {"max_err": e, "rms_err": r}
        k = self.kind
        if k in ("rank", "argmax", "topk"):
            res["exact"] = float(np.mean(np.rint(ys[0]) == ref[0]))
        if k in ("sort", "bitonic"):
            res["exact"] = float(np.mean(np.abs(ys[0] - ref[0]) < 0.5 * self.res))
        if k == "bitonic":
            res["argsort_exact"] = float(np.mean(np.rint(ys[1] * self.n) == np.rint(ref[1] * self.n)))
        if k == "threshold":
            w = 2.0 / 2 ** self.probes
            res["t_in_bracket"] = float(np.mean((ys[0][:, 0] <= ref[0][:, 0] + 1e-9) & (ref[0][:, 0] < ys[0][:, 0] + w)))
            res["mask_exact"] = float(np.mean(np.rint(ys[1]) == ref[1]))
        return res


@dataclasses.dataclass
class Pipeline(Task):
    """One stage calling all three modules (the planner-ready example): B vectors of n scores -> z-score (stats:
    rsqrt), a squashing tanh (math), the softmax of the squashed scores (stats: exp, max, reciprocal) and the top-k
    mask of the scores (sort: ranks). Outputs: z, tanh(z), softmax(2 tanh(z)), top-k mask."""
    n: int = 16
    B: int = 4
    k: int = 3
    err: float = 1e-3
    method: str = "tfhe"          # the comparisons (max, ranks)
    seed: int = 0
    name = "pipeline"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        g = np.stack([rng.permutation(self.n) for _ in range(self.B)]).astype(np.float64)
        self.X = ((g + 0.5 + rng.uniform(-0.25, 0.25, g.shape)) / self.n - 0.5) * rng.uniform(1.0, 3.0, (self.B, 1))
        self.lay = S.Batch(self.n, self.B)
        self.tfhe = self.method == "tfhe"
        v = self.X.var(1)
        self.band = (0.5 * float(v.min()), 2.0 * float(v.max()))

    def encode(self):
        return [self.lay.encode(self.X)]

    def run(self, ops, xs):
        x, lay = xs[0], self.lay
        z = S.zscore(ops, x, lay, *self.band, err=self.err)
        t = M.tanh(ops, z, -2.0, 2.0, self.err)
        sm = S.softmax(ops, ops.add(t, t), lay, -2.0, 2.0, self.err, shift="max", method=self.method)
        # neighbours are >= 0.5 / n apart (the construction): a public resolution, the same program for every seed
        r = T.rank(ops, x, lay, 3.0, method=self.method, res=0.25 / self.n)
        top = T.topk_mask(ops, r, lay, self.k, method=self.method)
        return [z, t, sm, top]

    def outputs(self, vals):
        return {nm: self.lay.decode(v) for nm, v in zip(("z", "tanh", "softmax", "topk"), vals)}

    def reference(self):
        X = self.X
        z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
        t = np.tanh(z)
        e = np.exp(2 * t - (2 * t).max(1, keepdims=True))
        top = (np.argsort(np.argsort(X, 1), 1) >= self.n - self.k).astype(float)
        return {"z": z, "tanh": t, "softmax": e / e.sum(1, keepdims=True), "topk": top}

    def metrics(self, out):
        ref = self.reference()
        res = {f"{k}_err": _err(out[k], ref[k])[0] for k in ("z", "tanh", "softmax", "topk")}
        res["topk_exact"] = float(np.mean(np.rint(out["topk"]) == ref["topk"]))
        return res


@dataclasses.dataclass
class AdHoc(Task):
    """An approximation the model code carries (examples/mathlib/compare.py's rows: `kind` ln / smexp / smrecip /
    relu) against impl.math's choice for the same domain and error (`variant` adhoc / ours), on every slot."""
    kind: str = "ln"
    variant: str = "adhoc"
    seed: int = 0
    name = "adhoc"

    def __post_init__(self):
        from . import compare as C
        rows = {"ln": C.ln_rows, "smexp": C.softmax_rows, "smrecip": C.softmax_rows, "relu": C.relu_rows}[self.kind]()
        pick = {"ln": "LN rsqrt transformer.h.0.ln_1", "smexp": "softmax exp transformer.h.0.attn",
                "smrecip": "softmax recip transformer.h.0.attn", "relu": "ReLU composite (7, 7, 27)"}[self.kind]
        self.row = next(r for r in rows if r["site"].startswith(pick))
        self.run_adhoc = C.RUNS[self.row["site"]]
        lo, hi = self.row["domain"]
        rng = np.random.default_rng(self.seed)
        self.x = rng.uniform(lo, hi, SLOTS)
        self.x[:2] = lo, hi
        self.ref = C.REFS[self.row["site"]]

    def encode(self):
        return [self.x]

    def run(self, ops, xs):
        if self.variant == "adhoc":
            return [self.run_adhoc(ops, xs[0])]
        lo, hi = self.row["domain"]
        e = self.row["adhoc_err"]
        if self.kind == "relu":
            return [M.relu(ops, xs[0], 1.0, e)]
        fn = {"ln": "rsqrt", "smexp": "exp", "smrecip": "reciprocal"}[self.kind]
        a = M.approx(fn, lo, hi, e)
        if self.kind == "ln":                       # the seed's output scale rides the coefficients' target
            k = self.ref(np.array([1.0]))[0]
            return [ops.mult(a(ops, xs[0]), float(k))] if k != 1.0 else [a(ops, xs[0])]
        return [a(ops, xs[0])]

    def outputs(self, vals):
        return {"y": np.real(vals[0])}

    def metrics(self, out):
        t = self.ref(self.x)
        e, r = _err(out["y"], t, rel=self.row["rel"])
        return {"max_err": e, "rms_err": r}


TASKS = {"fn": Fn, "stats": Stats, "sort": Sort, "pipeline": Pipeline, "adhoc": AdHoc}


def make(name, **kw):
    return TASKS[name](**kw)
