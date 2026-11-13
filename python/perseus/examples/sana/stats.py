"""Per-site statistics over prompts: the input ranges every approximation is fitted on (fit.py). The SANA transformer's
sites (ref.py's) on the ref_float dumps (their Gemma rows and every step's input), or with --gemma Gemma-2's
(gemma.py's) on the dumps' prompt rows.

    .venv/bin/python -m perseus.examples.sana.stats --dumps ref --out stats_ref.npz
    .venv/bin/python -m perseus.examples.sana.stats --gemma --dumps ref --out stats_gemma.npz
"""
from __future__ import annotations

import argparse
import collections
import glob
import os

import numpy as np

from . import hf
from . import ref as R


class Stats(R.Approx):
    """Exact nonlinearities; per site the min / max of the input, its max |x| and sorted samples of the values."""

    def __init__(self, keep=4096, seed=0):
        self.lo = collections.defaultdict(lambda: np.inf)
        self.hi = collections.defaultdict(lambda: -np.inf)
        self.samples = collections.defaultdict(list)
        self.zeros = collections.Counter()
        self.keep = keep
        self.rng = np.random.default_rng(seed)

    def see(self, site, x):
        if site is None:
            return
        v = x.detach().reshape(-1).double().numpy()
        self.lo[site] = min(self.lo[site], float(v.min()))
        self.hi[site] = max(self.hi[site], float(v.max()))
        self.zeros[site] += int((v == 0).sum())
        idx = self.rng.choice(len(v), min(self.keep, len(v)), replace=False)
        self.samples[site].append(v[idx])

    def see_range(self, site, x):
        """The site's min / max only: no samples (the other sites' draws stay as without it)."""
        v = x.detach().reshape(-1).double().numpy()
        self.lo[site] = min(self.lo[site], float(v.min()))
        self.hi[site] = max(self.hi[site], float(v.max()))

    def summary(self):
        out = {}
        for s in self.lo:
            q = np.quantile(np.concatenate(self.samples[s]), [1e-4, 1e-3, 0.01, 0.5, 0.99, 0.999, 0.9999]) \
                if self.samples[s] else None
            out[s] = dict(lo=self.lo[s], hi=self.hi[s], absmax=max(abs(self.lo[s]), abs(self.hi[s])), q=q,
                          zeros=self.zeros[s])
        return out


def run_dumps(W, dumps, st, on_block=None):
    """Every step a dump has (its x_in2: the second, at hf.STEPS[1]): the steps share the sites, a site's range is the
    union."""
    k = hf.scm_constants()
    for f in dumps:
        z = np.load(f)
        sel = np.array([0] + list(range(len(z["ids"]) - hf.MAX_SEQ + 1, len(z["ids"]))))
        steps = [(W, z["noise"].astype(np.float64) * k["c_in"])]
        if "x_in2" in z.files:
            steps.append((W.step(1), z["x_in2"].astype(np.float64)))
        for Wi, x_in in steps:
            R.forward(Wi, x_in, z["gemma_out_sel"].astype(np.float64), z["mask"][sel], ap=st, on_block=on_block)
        print(f"[stats] {os.path.basename(f)} ({len(steps)} steps)", flush=True)


def run_gemma(dumps, st):
    """Gemma-2's sites on the dumps' prompt rows (the public instruction prefix, the same for every prompt, once).
    Returns the residual stream's per-channel max |x| (the embedding rows and every layer's output: fit's text
    residual scale)."""
    import math
    import torch
    from . import gemma as GM
    W = GM.GemmaWeights()
    kvs = None
    m = np.zeros(GM.D)
    for f in dumps:
        z = np.load(f)
        n_pub, n_tok = int(z["n_pub"]), int(z["n_tok"])
        ids = z["ids"][:n_pub + n_tok]
        kvs = kvs or GM.public_prefix(W, ids[:n_pub])
        x0 = (W["embed_tokens.weight"][torch.as_tensor(ids[n_pub:])] * math.sqrt(GM.D)).abs().max(0).values.numpy()
        np.maximum(m, x0, out=m)
        GM.prompt_states(W, ids, n_pub, kvs, ap=st,
                         on_layer=lambda b, x: np.maximum(m, x.abs().max(0).values.numpy(), out=m))
        print(f"[stats] gemma {os.path.basename(f)}", flush=True)
    return m


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="ref")
    ap.add_argument("--out", default="stats_ref.npz")
    ap.add_argument("--gemma", action="store_true", help="Gemma-2's sites (the text side) instead of SANA's")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)
    st = Stats()
    dumps = sorted(glob.glob(hf.scratch(a.dumps, "p*.npz")))
    if a.gemma:
        resid = run_gemma(dumps, st)
        summ = st.summary()
        np.savez(hf.scratch(a.out), summary=np.array(summ, dtype=object), resid=resid)
        for s, v in summ.items():
            print(f"{s:20s} [{v['lo']:10.4g}, {v['hi']:10.4g}] q.001 {v['q'][1]:9.3g} q.999 {v['q'][5]:9.3g}")
        return
    W = R.SanaWeights()
    resid = collections.defaultdict(lambda: np.zeros(R.D))

    def on_block(b, x):
        resid[b] = np.maximum(resid[b], x.abs().max(0).values.numpy())
    run_dumps(W, dumps, st, on_block)
    summ = st.summary()
    np.savez(hf.scratch(a.out), summary=np.array(summ, dtype=object),
             resid=np.stack([resid[b] for b in range(W.n_layers)]))
    for s, v in summ.items():
        q = v["q"]
        print(f"{s:16s} [{v['lo']:10.4g}, {v['hi']:10.4g}]" + ("" if q is None else
              f" q1e-3 {q[1]:9.3g} med {q[3]:9.3g} q.999 {q[5]:9.3g} zeros {v['zeros']}"))
    r = np.stack([resid[b] for b in range(W.n_layers)])
    for b in range(W.n_layers):
        top = np.argsort(-r[b])[:3]
        print(f"resid b{b:2d} max {r[b].max():9.3g} median-ch {np.median(r[b]):7.3g} top ch {top.tolist()}")


if __name__ == "__main__":
    main()
