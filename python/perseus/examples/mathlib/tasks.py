"""Problems for impl.math / impl.stats / impl.sort: the inputs, the encrypted computation as one stage, the float64
reference and the error metrics. A `Task` is what run.py drives on the plaintext mirror, the CPU twin (capture for the
planner) and the planned device session."""
from __future__ import annotations

import dataclasses
import math as _m

import numpy as np

from perseus.impl import keyset
from perseus.impl import math as M
from perseus.impl import sort as T
from perseus.impl import stats as S
from perseus.impl.ops import NumpyOps

SLOTS = 1 << 15


class Task:
    """`encode()` -> slot vectors the client encrypts, `run(ops, cts)` -> output cts, `outputs(vals)` -> {name: array}
    from the decrypted slot vectors, `metrics(out)` -> {name: float}. `tfhe`: the stage switches to TFHE."""
    seed: int = 0
    name = "task"
    tfhe = False

    def rotations(self):
        """The rotation keys the stage needs (keyset.stage_rotations: the code run on the recording mirror)."""
        return keyset.stage_rotations(self.run, self.encode(), SLOTS)

    def mirror(self):
        return self.outputs([np.asarray(v) for v in self.run(NumpyOps(), self.encode())])

    def describe(self):
        return {k: v for k, v in dataclasses.asdict(self).items()}


def _err(y, t, rel=False):
    """(max, RMS) error; relative: against |t| floored at 1e-4 max |t| (CKKS noise is absolute: values below the
    floor, e.g. exp(-30), carry ~1e-9 of noise, not a relative error)."""
    e = np.abs(np.asarray(y) - np.asarray(t))
    if rel:
        e = e / np.maximum(np.abs(t), 1e-4 * float(np.abs(t).max()))
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
    fbs: bool = False             # the function as a functional bootstrap (impl.math `approx_fbs`)
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
        if self.fbs:
            a = M.approx_fbs(self.fn, self.lo, self.hi, self.err, self.fn in ("exp", "rsqrt", "reciprocal"))
            if a is None:
                raise M.TargetError(f"fbs {self.fn} on [{self.lo}, {self.hi}]: no form meets {self.err}")
            return a
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
        # an FBS renames its operand (its level hints): the stage input stays the harness's
        return [self.approx()(ops, ops.copy(x) if self.fbs else x)]

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
        out = {"max_err": e, "rms_err": r, "max_rel": er, "rms_rel": rr}
        if self.fn in ("abs", "relu", "sign", "clip"):
            # the decisions the comparison got wrong (an error past 10x the target), and the largest |x| among them
            bad = np.abs(y - t) > 10.0 * self.err
            xs = self.x if self.fn != "sign" else self.x[np.abs(self.x) >= self.err * max(_m.fabs(self.lo),
                                                                                          _m.fabs(self.hi))]
            out["wrong_frac"] = float(bad.mean())
            out["wrong_max_absx"] = float(np.abs(xs[bad]).max()) if bad.any() else 0.0
        return out

    def describe(self):
        d = super().describe()
        if self.fn in ("abs", "relu", "sign", "clip") and self.method == "ckks":
            B = max(_m.fabs(self.lo), _m.fabs(self.hi))
            c = M.count(lambda o, v: self.run(o, [v])[0])
            lad = (M.sign_ladder(*M._resolution(1.0, self.err)) if self.fn == "sign" else
                   M.relu_ladder(B if self.fn != "clip" else B + 0.5 * B, 0.5 * self.err if self.fn != "relu"
                                 else self.err))
            d.update(method_chosen=f"composite {lad.degs}", depth=c.depth, ct_mults=c.ct_mults,
                     params={"eps": lad.eps})
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
    fbs: bool | str = False       # softmax / logsumexp / zscore: the refreshed function may be an FBS ("force")
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
            return [S.zscore(ops, x, lay, lo, hi, self.err, fbs=self.fbs)]
        if k == "softmax":
            return [S.softmax(ops, x, lay, -4.0, 4.0, self.err, shift=self.shift, method=self.method,
                              fbs=self.fbs)]
        if k == "logsumexp":
            return [S.logsumexp(ops, x, lay, -4.0, 4.0, self.err, shift=self.shift, method=self.method,
                                fbs=self.fbs)]
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
    levels: int = 1               # TFHE comparisons' scales (sort.step)
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
            return [T.vmax(ops, x, lay, bound, method=meth, err=res, levels=self.levels)]
        if k == "argmax":
            return [T.argmax(ops, x, lay, bound, method=meth, res=res, err=0.25 * res)]
        if k == "bitonic":
            return list(T.bitonic(ops, x, lay, bound, method=meth, res=res, levels=self.levels,
                                  payload=ops.add_pt(ops.mult_pt(x, np.zeros(SLOTS), kind="mask"),
                                                     lay.encode(np.tile(np.arange(self.n) / self.n, (self.B, 1))),
                                                     kind="mask")))
        if k == "threshold":
            t, mask = T.threshold(ops, x, lay, self.k, -1.0, 1.0, probes=self.probes, method=meth, levels=self.levels)
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
    fbs: bool | str = False       # zscore's rsqrt and softmax's reciprocal may be FBS ("force")
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
        z = S.zscore(ops, x, lay, *self.band, err=self.err, fbs=self.fbs)
        t = M.tanh(ops, z, -2.0, 2.0, self.err)
        sm = S.softmax(ops, ops.add(t, t), lay, -2.0, 2.0, self.err, shift="max", method=self.method, fbs=self.fbs)
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


@dataclasses.dataclass
class Lut(Task):
    """A CKKS functional bootstrap (impl/fbs.py) on every slot: `kind` table (inputs the integers |j| <= t/2, `table`
    sign / step / rand: a fixed pseudo-random table on a 1/8 grid in [-1, 1]) or sign (the 0/1 step on |x| < 2, inputs
    outside the gap). Planned runs only (the FBS is a refresh the plan places)."""
    kind: str = "table"
    table: str = "sign"
    t: int = 16
    gap: float = 0.1
    seed: int = 0
    name = "lut"

    def __post_init__(self):
        from perseus.impl import fbs
        rng = np.random.default_rng(self.seed)
        self.t, self.gap = int(self.t), float(self.gap)
        h = self.t // 2
        if self.kind == "table":
            j = np.arange(-h, h + 1)
            if self.table == "sign":
                self.values = np.sign(j).astype(float)
            elif self.table == "step":
                self.values = (j >= 0).astype(float)
            else:
                self.values = np.random.default_rng(1234).integers(0, 17, len(j)) / 8.0 - 1.0
            self.fn = fbs.register_table(f"lut.{self.table}.{self.t}", self.values)
            self.x = rng.integers(-h, h + 1, SLOTS).astype(float)
            self.x[:2] = (-h, h)
            self.spacing = float(np.diff(np.unique(self.values)).min())
        else:
            self.fn = fbs.register_sign(f"lut.sign01.g{self.gap:g}", 0.0, 1.0, gap=float(self.gap))
            a = rng.uniform(self.gap, 1.99, SLOTS)
            self.x = np.where(rng.integers(0, 2, SLOTS) == 1, a, -a)
            self.spacing = 1.0

    def encode(self):
        return [self.x]

    def ref(self, x):
        if self.kind == "table":
            return self.values[np.round(x).astype(int) + self.t // 2]
        return (x >= 0).astype(float)

    def run(self, ops, xs):
        from perseus.impl import fbs
        # a copy: the FBS's level hints rename its operand in place, and the stage's input keeps its own name
        return [fbs.fbs(ops, ops.copy(xs[0]), self.fn)]

    def outputs(self, vals):
        return {"y": np.real(vals[0])}

    def metrics(self, out):
        from perseus.impl import fbs
        e, r = _err(out["y"], self.ref(self.x))
        wrong = int(np.sum(np.abs(out["y"] - self.ref(self.x)) >= self.spacing / 2.0))
        return {"max_err": e, "rms_err": r, "wrong": wrong, "log2_pfail_ct": fbs.p_fail(self.fn)["log2_ct"]}


@dataclasses.dataclass
class IntOp(Task):
    """Exact integer arithmetic by chained FBS lookups (impl/intops.py) on every slot: `op` add / cmp (a < b) / mux
    (sel ? a : b) / and / or / xor / addmux (a + b, then sel ? sum : a) on `bits`-bit words in radix 2^`r` digits
    (one ciphertext per digit), EvalMod `table` (impl/fbs.py TABLES). Every output must round to the exact result."""
    op: str = "add"
    bits: int = 16
    r: int = 2
    table: str = "k32"
    layout: str = "cts"           # cts: a digit per ciphertext (ripple); slots: a word's digits in one ciphertext
    seed: int = 0
    name = "intop"

    def __post_init__(self):
        from perseus.impl import intops
        self.bits, self.r = int(self.bits), int(self.r)
        rng = np.random.default_rng(self.seed)
        top = 1 << self.bits
        self.a = rng.integers(0, top, SLOTS, dtype=np.int64)
        self.b = rng.integers(0, top, SLOTS, dtype=np.int64)
        self.b[:64] = self.a[:64]                     # equal words: the compare's "=" path end to end
        self.sel = rng.integers(0, 2, SLOTS).astype(np.int64)
        if self.layout == "slots":
            self.io = intops.WordOps(None, self.bits, self.r, self.table, SLOTS)
            n = self.bits // self.r
            self.a, self.b, self.sel = self.a[:SLOTS // n], self.b[:SLOTS // n], self.sel[:SLOTS // n]
            if self.op in ("and", "or", "xor"):
                self.io.bitwise(self.op, None, None)
        else:
            self.io = intops.IntOps(None, self.r, self.table)      # registers the tables in this process
            if self.op in ("and", "or", "xor"):
                self.io.bitwise(self.op, [], [])

    def encode(self):
        from perseus.impl import intops
        if self.layout == "slots":
            v = [self.io.encode(self.a), self.io.encode(self.b)]
            return v + [np.repeat(self.sel, self.bits // self.r).astype(float)] if self.op in ("mux", "addmux") else v
        v = intops.digits(self.a, self.bits, self.r) + intops.digits(self.b, self.bits, self.r)
        return v + [self.sel.astype(float)] if self.op in ("mux", "addmux") else v

    def ref(self):
        m = (1 << self.bits) - 1
        a, b, s = self.a, self.b, self.sel
        return {"add": (a + b) & m, "cmp": (a < b).astype(np.int64), "mux": np.where(s == 1, a, b),
                "and": a & b, "or": a | b, "xor": a ^ b, "addmux": np.where(s == 1, (a + b) & m, a), "id": a}[self.op]

    def run(self, ops, xs):
        from perseus.impl import intops
        if self.layout == "slots":
            w = intops.WordOps(ops, self.bits, self.r, self.table, SLOTS)
            a, b = xs[0], xs[1]
            if self.op == "id":
                return [ops.add(a, 0.0)]
            if self.op == "add":
                return [w.add(a, b)]
            if self.op == "cmp":
                return [w.cmp(a, b)]
            if self.op == "mux":
                return [w.mux(xs[2], a, b)]
            if self.op == "addmux":
                return [w.mux(xs[2], w.add(a, b), a)]
            return [w.bitwise(self.op, a, b)]
        io = intops.IntOps(ops, self.r, self.table)
        n = self.bits // self.r
        a, b = list(xs[:n]), list(xs[n:2 * n])
        if self.op == "id":                          # the I/O baseline: inputs in, a out
            return [ops.add(ai, 0.0) for ai in a]
        if self.op == "add":
            return io.add(a, b)
        if self.op == "cmp":
            return [io.cmp(a, b)]
        if self.op == "mux":
            return io.mux(xs[2 * n], a, b)
        if self.op == "addmux":
            return io.mux(xs[2 * n], io.add(a, b), a)
        return io.bitwise(self.op, a, b)

    def outputs(self, vals):
        return {"d": [np.real(v) for v in vals]}

    def metrics(self, out):
        from perseus.impl import intops, fbs
        d = out["d"]
        if self.layout == "slots":
            n = self.bits // self.r
            got = (np.round(np.real(d[0])).astype(np.int64).reshape(-1, n)[:, -1] if self.op == "cmp"
                   else self.io.decode(d[0]))
        else:
            got = intops.value(d, self.r) if self.op != "cmp" else np.round(d[0]).astype(np.int64)
        dev = max(float(np.abs(np.asarray(v) - np.round(v)).max()) for v in d)
        wrong = int(np.sum(got != self.ref()))
        lp = max(fbs.p_fail(n)["log2_ct"] for n in self.io.lookups())
        return {"wrong": wrong, "max_dev": dev, "log2_pfail_lookup_ct": lp}


@dataclasses.dataclass
class Plut(Task):
    """An encrypted lookup table (TFHE, impl/tfhe.py Prog.pack / Prog.plut): `t` encrypted entries (multiples of 1/8 in
    [-1, 1)) and `b` encrypted indices in one ciphertext, the table packed into an accumulator (the packing key switch)
    and blindly rotated by every index, the entries read back. Planned runs only (a native program in the session's
    switch; sets PC_TFHE_NATIVE / PC_TFHE_PACK when unset). Every output must round to its entry."""
    t: int = 16
    b: int = 256
    seed: int = 0
    name = "plut"
    tfhe = True

    def __post_init__(self):
        import os
        from perseus.impl import tfhe
        self.t, self.b = int(self.t), int(self.b)
        os.environ.setdefault("PC_TFHE_NATIVE", f"4,{self.t + self.b}")
        os.environ.setdefault("PC_TFHE_PACK", "1")
        rng = np.random.default_rng(self.seed)
        self.table = rng.integers(-8, 8, self.t) / 8.0
        self.idx = rng.integers(0, self.t, self.b)
        self.idx[:self.t] = np.arange(self.t)[:self.b]
        tfhe.register_slots("plut.in", np.arange(self.t + self.b))
        tfhe.register_slots("plut.out", np.arange(self.b))
        pr = tfhe.Prog(f"plut.t{self.t}.b{self.b}", 3)
        pr.lut("id", lambda m: m)
        pr.lut("out", lambda m: m, warp=True)
        pr.load("id", 0)
        place, w = tfhe.table_place(self.t)
        pr.pack(0, self.t, place, acc=0, w=w)
        pr.lin(1, 0, 0, [(self.t + r, -1) for r in range(self.b)])
        pr.plut(1, self.b, 0, [2])
        pr.store(2, self.b, "out", "plut.out")
        self.prog = tfhe.register_prog(pr)

    def encode(self):
        from perseus.impl import tfhe
        x = np.zeros(SLOTS)
        x[:self.t] = self.table
        x[self.t:self.t + self.b] = tfhe.table_index(self.idx, self.t)
        return [x]

    def run(self, ops, xs):
        from perseus.impl import tfhe
        y = tfhe.tfhe_prog(ops, ops.copy(xs[0]), self.prog, "plut.in")
        return [y]

    def outputs(self, vals):
        return {"y": np.real(vals[0])[:self.b]}

    def metrics(self, out):
        want = self.table[self.idx]
        e = np.abs(out["y"] - want)
        return {"wrong": int(np.sum(np.round(out["y"] * 8) != np.round(want * 8))), "max_err": float(e.max()),
                "rms_err": float(np.sqrt(np.mean(e * e)))}


TASKS = {"fn": Fn, "stats": Stats, "sort": Sort, "pipeline": Pipeline, "adhoc": AdHoc, "lut": Lut, "intop": IntOp, "plut": Plut}


def make(name, **kw):
    return TASKS[name](**kw)
