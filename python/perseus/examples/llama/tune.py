"""The decode tuning of a calibrated decoder config (arch.MODELS): the steps Llama-3.1-8B's shipped decode config
(llama31_8b_n32_ctx512_silus) took after perseus-calibrate, for any Arch, so every model of the port gets the same
levers (configs/model/approximation/README.md):

- `act`: the MLP activation as one Chebyshev series of act(X t) / X on [-1, 1] per block, X the site's calibrated xmax
  (the gate carries 1 / xmax under GELU_FOLD: the weight cache keys on it), at the lowest eval_chebyshev depth whose
  largest degree meets max(the THOR composite's error on [-X, X], ACT_FLOOR), that degree; the lane mask folded into
  the series term by term where a leaf has no slack (`mask_fold` 2: no unit for the mask). Llama's SiLU: degree 28,
  depth 5.
- `design`: the SM_CHEB_GS reciprocal counts (impl.smcheb.design) over the cache lengths up to --kc-max.
- `centre`: the softmax centred on key 0 (the BOS sink) up to --kc-max keys (scripts/utils/fit_llama_decode_sink.py
  for any Arch): on OpenWebText shard 0042's document starts through the fp32 model on the visible GPU, a layer's U* is
  the largest u = f(s_j) - f(s_0) over its heads and queries, f the Arch's logit soft-cap (the identity without one:
  Llama's u = s_j - s_0); fit_gpt2_decode_sink.fit_sink fits the rest. A layer whose held-out u passes U, or whose
  centred softmax is not shallower, keeps its softmax. Under a soft-cap the runtime centres after the cap
  (attention._centre_mapped), else on the K side.
- `spru`: every block RMSNorm's mean square from one 1-slot refresh instead of its all-slot ladder (`spru`, the norm
  statistics GPT-2's decode takes this way; the final norm keeps its ladder, as GPT-2's ln_f).
- `capfold`: under a logit soft-cap (Gemma-2), each uncentred layer's exp series refitted on the raw scores as the
  series of exp(sf (cap tanh(s / cap) - c)), c the capped clip's centre, so the soft-cap's own series (degree 7-9,
  3-4 levels on every score ciphertext) goes: the raw interval is the clip's preimage within the soft-cap's fitted
  range, the degree the lowest at the exp series' depth whose error in the softmax's output (the series' relative
  error times 2^(log2delta1 + log2delta2)) meets the soft-cap fit's own (at least CAPFOLD_FLOOR). The exp's output
  range is unchanged, so the reciprocal seeds and counts stand.
- `flags`: the softmax's active mask folded into the exp series (`exp_mask_fold` 2, as the act's) and every reciprocal round's scores
  folded into its Goldschmidt recursion, each round's numerator refreshed before its last squaring, the first round
  folded too (`gs_num`, `gs_num_refresh`, `gs_num_first`).

    python -m perseus.examples.llama.tune --model gemma2_2b --base BASE.json --out OUT.json \\
        [--steps act,norms,design,centre,capfold,spru,flags]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

ACT_FLOOR = 4.7e-4          # the THOR calibrator's error level (scripts/utils/fit_gelu_series.py FLOOR)
ACT_EPS = 1e-9              # the series' coefficient trim (cheb_eps)
# the masks ride the series term by term where a leaf has no slack (mode 2): the fold then never costs a unit
# (mode 1 does on 22 of Llama's 32 SiLU sites and 16 of its 32 exp series)
MASK_FOLD = 2
CAPFOLD_FLOOR = 1e-6       # the folded series' error floor in the softmax's output (relative)
CAPFOLD_MAX_DEG = 31
SOFTMAX_FLAGS = {"exp_mask_fold": MASK_FOLD, "gs_num": 1, "gs_num_refresh": 1, "gs_num_first": 1}


def act_fn(act: str):
    """The Arch's activation (numpy, float64)."""
    if act == "silu":
        return lambda x: x / (1.0 + np.exp(-x))
    if act == "gelu_tanh":
        return lambda x: 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    raise ValueError(f"no activation {act!r}")


def thor_err(g, f, t):
    """The THOR composite's max error over [-xmax, xmax] (t the unit grid)."""
    from numpy.polynomial import chebyshev as C
    S = g["xmax"]
    u = (2 * t - (g["thor_p1_a"] + g["thor_p1_b"])) / (g["thor_p1_b"] - g["thor_p1_a"])
    p1 = C.chebval(u, g["thor_p1_cheb"])
    v = (2 * p1 - (g["thor_p2_a"] + g["thor_p2_b"])) / (g["thor_p2_b"] - g["thor_p2_a"])
    return float(np.abs(S * t * (C.chebval(v, g["thor_p2_cheb"]) + 0.5) - f(S * t)).max())


def series_depth(coeffs, eps=ACT_EPS) -> tuple[int, int]:
    """(degree, depth) of eval_chebyshev_bsgs on [-1, 1] (no affine level) after its coefficient trim."""
    from perseus.impl.poly import _cheb_dep, _cheb_geometry
    c, n, m, dt = _cheb_geometry(list(coeffs), eps)
    return n, _cheb_dep(c, m, dt)


def act_series(g: dict, act: str, max_deg: int = 255) -> dict | None:
    """The single-series entry replacing the THOR site `g` (None: keep it; the site is already a series)."""
    from numpy.polynomial import chebyshev as C
    if g.get("method") != "thor_composite":
        return None
    f = act_fn(act)
    X = float(g["xmax"])
    t = np.linspace(-1.0, 1.0, 200001)
    target = max(thor_err(g, f, t), ACT_FLOOR)
    best = {}                                   # depth -> (degree, coeffs, err): the largest degree at each depth
    for deg in range(4, max_deg + 1):
        c = C.chebinterpolate(lambda u: f(X * u) / X, deg)
        c[3::2] = 0.0                           # act(x) - x / 2 is even (SiLU, GELU): no odd terms past T_1
        n, dep = series_depth(c)
        e = float(np.abs(C.chebval(t, c) * X - f(X * t)).max())
        best[dep] = (deg, c, e, n)
        if dep in best and dep - 1 in best and best[dep - 1][2] <= target:
            break
    for dep in sorted(best):
        deg, c, e, n = best[dep]
        if e <= target:
            name = {"silu": "SiLU", "gelu_tanh": "GELU (tanh)"}[act]
            return {"method": "chebyshev", "xmax": X, "cheb_coeffs": [float(x) for x in c], "cheb_a": -1.0,
                    "cheb_b": 1.0, "cheb_eps": ACT_EPS, "mask_fold": MASK_FOLD,
                    "note": f"degree-{deg} Chebyshev interpolation of {name}(X t)/X on [-1,1], X = {X:.4f}; depth {dep}; "
                            f"max abs err {e:.1e} on [-X, X]; target {target:.2g} (max(THOR error, {ACT_FLOOR:g})); "
                            f"the lowest depth meeting it (examples/llama/tune.py)"}
    return None


def tune_act(cfg: dict, act: str, log=print):
    sec = {"silu": "silu", "gelu_tanh": "gelu"}[act]
    for name in sorted(cfg[sec], key=lambda n: [int(p) for p in n.split(".") if p.isdigit()]):
        new = act_series(cfg[sec][name], act)
        if new is None:
            log(f"[act] {name}: {cfg[sec][name].get('method')} kept")
            continue
        cfg[sec][name] = new
        log(f"[act] {name}: {new['note'].split(';')[0]}; {new['note'].split(';')[1].strip()}")


def norm_depth(coeffs, nr: int) -> int:
    """Levels of rms_norm's inverse square root: the seed (eval_chebyshev_bsgs: the affine, its own depth) and two per
    Newton iteration (poly.inv_sqrt_newton_d2)."""
    return 1 + series_depth(coeffs, eps=None)[1] + 2 * nr


def tune_norms(cfg: dict, target: float = 1e-5, max_deg: int = 63, log=print):
    """Every RMSNorm site's inverse-square-root seed refitted on its own band at the shallowest (seed, Newton) chain
    meeting `target` relative error (scripts/utils/refit_norm_cheb.fit_site's search over degrees up to max_deg), kept
    when it is not shallower than the calibrated one. Llama's shipped decode config refitted 15 sites this way
    (degree ~19 + one Newton step -> one seed of degree 33-37)."""
    for name in sorted(cfg["norm"], key=lambda n: [int(p) for p in n.split(".") if p.isdigit()] or [1 << 30]):
        j = cfg["norm"][name]
        if not j.get("cheb_coeffs"):
            continue
        old = norm_depth(j["cheb_coeffs"], int(j.get("cheb_nr_iters", 0)))
        lo, hi, s = float(j["cheb_lo"]), float(j["cheb_hi"]), float(j["inv_out_scale"])
        best = None
        for nr in (0, 1, 2):
            deg, nr_, c, err = _fit_fixed_nr(lo, hi, s, target, max_deg, nr)
            if c is None:
                continue
            dep = norm_depth(c, nr_)
            if best is None or dep < best[0]:
                best = (dep, deg, nr_, c, err)
        if best is None or best[0] >= old:
            log(f"[norms] {name}: depth {old} kept (band ratio {hi / lo:.0f})")
            continue
        dep, deg, nr, c, err = best
        log(f"[norms] {name}: band ratio {hi / lo:.0f}: depth {old} -> {dep} (degree {deg} + {nr} Newton, err {err:.1e})")
        j.update(cheb_coeffs=[float(v) for v in c], cheb_nr_iters=int(nr), depth=int(dep), fit_err=float(err))


def _fit_fixed_nr(lo, hi, scale, target, max_deg, nr):
    """refit_norm_cheb.fit_site's search with the Newton count pinned to `nr`: (deg, nr, coeffs, err), coeffs None when
    no degree up to max_deg meets the target."""
    from perseus.impl.ops import NumpyOps
    from perseus.impl.poly import eval_chebyshev, inv_sqrt_newton_d2
    ops = NumpyOps()
    t = np.cos(np.pi * (np.arange(400) + 0.5) / 400)
    f = scale / np.sqrt((t + 1) / 2 * (hi - lo) + lo)
    grid = np.geomspace(lo, hi, 4001)
    exact = scale / np.sqrt(grid)
    for deg in range(2, max_deg + 1):
        w = 1.0 / f
        c = np.polynomial.chebyshev.chebfit(t, f, deg, w=w)
        for _ in range(30):     # reweight toward minimax relative error (fit_site)
            r = np.abs(np.polynomial.chebyshev.chebval(t, c) / f - 1)
            c = np.polynomial.chebyshev.chebfit(t, f, deg, w=w * (r / r.max() + 0.05))
        y = inv_sqrt_newton_d2(ops, grid, eval_chebyshev(ops, grid, c, lo, hi), nr, 1.0 / (scale * scale))
        err = float(np.max(np.abs(y / exact - 1)))
        if err <= target:
            return deg, nr, c, err
    return None, nr, None, None


def tune_design(cfg: dict, kc_max: int, log=print):
    from perseus.impl import smcheb
    from perseus.impl.config import parse_softmax
    kcs = tuple(sorted({k for k in smcheb.KC_DESIGN if k <= kc_max} | {kc_max}))
    for name, j in cfg["softmax"].items():
        s = parse_softmax(j)
        ks, tg = smcheb.design(s.init_alpha, s.init_beta, s.log2delta2, smcheb.TGT_LAST, kcs)
        j["recip_iters_cheb"], j["recip_targets"] = ks, tg
        log(f"[design] {name}: SM_CHEB_GS {ks} targets {tg}")


def _sink_fit():
    """scripts/utils/fit_gpt2_decode_sink (OWT, fit_sink, depth): the offline fit the shipped centred configs used."""
    from perseus._paths import PKG
    sys.path.insert(0, str(PKG.parent.parent / "scripts" / "utils"))
    import fit_gpt2_decode_sink as F
    return F


def sink_docs(kmax: int, n: int, owt: str | None = None):
    """`n` document starts of `kmax` tokens (BOS first) from OpenWebText shard 0042, in fit_llama_decode_sink's order."""
    import pyarrow.parquet as pq
    from . import hf
    F = _sink_fit()
    tok = hf.tokenizer()
    if owt is None:
        from huggingface_hub import hf_hub_download
        owt = hf_hub_download(F.OWT[0], F.OWT[1], repo_type="dataset", revision=F.OWT[2])
    texts = pq.read_table(owt, columns=["text"]).column("text").to_pylist()
    rng = np.random.default_rng(1234)
    docs = []
    for i in rng.permutation(len(texts)):
        ids = tok(texts[i][:8 * kmax])["input_ids"]
        if len(ids) >= kmax:
            docs.append(ids[:kmax])
        if len(docs) == n:
            break
    if not all(d[0] == tok.bos_token_id for d in docs):
        raise ValueError("sink_docs: the tokenizer does not put BOS first")
    return docs


def sink_ranges(model, docs):
    """(hi, lo) per layer, per query head: max / min u = f(s_j) - f(s_0) over every causal row (rows past 0) of every
    document, f the Arch's soft-cap (identity without one)."""
    import importlib
    import inspect
    import torch
    from . import hf
    ar = hf.arch()
    M = importlib.import_module(ar.hf_module)
    orig = M.eager_attention_forward
    sig = inspect.signature(orig)
    stats = {}

    def eager(*args, **kw):
        ba = sig.bind(*args, **kw)
        ba.apply_defaults()
        mod, q, k = ba.arguments["module"], ba.arguments["query"], ba.arguments["key"]
        sc = ba.arguments.get("scaling")
        sc = mod.scaling if sc is None else sc
        cap = ba.arguments.get("softcap")
        s = (q @ M.repeat_kv(k, mod.num_key_value_groups).transpose(2, 3) * sc)[0].double()
        if cap:
            s = cap * torch.tanh(s / cap)
        T = s.shape[-1]
        tri = torch.tril(torch.ones(T, T, dtype=torch.bool, device=s.device))[1:]
        u = (s - s[..., :1])[:, 1:]
        hi = torch.where(tri, u, -torch.inf).amax((-1, -2)).cpu().numpy()
        lo = torch.where(tri, u, torch.inf).amin((-1, -2)).cpu().numpy()
        h0, l0 = stats.get(mod.layer_idx, (hi, lo))
        stats[mod.layer_idx] = (np.maximum(h0, hi), np.minimum(l0, lo))
        return orig(*args, **kw)

    M.eager_attention_forward = eager
    try:
        with torch.no_grad():
            for d in docs:
                model(torch.tensor([d], device=model.device))
    finally:
        M.eager_attention_forward = orig
    L = len(stats)
    return np.stack([stats[i][0] for i in range(L)]), np.stack([stats[i][1] for i in range(L)])


def tune_centre(cfg: dict, kc_max: int, n_docs: int = 48, tau: float = 4.6, margin: float = 1.25, owt=None,
                log=print):
    from . import hf
    F = _sink_fit()
    docs = sink_docs(kc_max, 2 * n_docs, owt)
    model = hf.load_model()
    hi, lo = sink_ranges(model, docs[:n_docs])
    ho, _ = sink_ranges(model, docs[n_docs:])
    del model
    cap = hf.arch().attn_softcap
    log(f"[centre] {n_docs} documents fitted, {len(docs) - n_docs} held out ({kc_max} tokens); u = "
        f"{'f(s_j) - f(s_0), f = ' + format(cap, 'g') + ' tanh(s / ' + format(cap, 'g') + ')' if cap else 's_j - s_0'}")
    name0 = hf.NAME.split("/")[-1]
    for i in range(hi.shape[0]):
        name = f"layers.{i}.attn"
        base = cfg["softmax"][name]
        c = F.fit_sink(hi[i], lo[i], base, kc_max, tau, margin)
        over = float(ho[i].max()) > c["clip_hi"]
        keep = over or F.depth(c) >= F.depth(base)
        c["note"] = c["note"].replace("OpenWebText shard 0042 document starts",
                                      f"OpenWebText shard 0042 document starts ({name0}, fp32"
                                      f"{', after the ' + format(cap, 'g') + ' soft-cap' if cap else ''})")
        if cap:
            c["note"] = c["note"].replace("u = s - s_0", "u = f(s) - f(s_0)")
        log(f"[centre] layer {i}: U {c['clip_hi']:.2f} (held-out max u {ho[i].max():.2f}) delta2 {c['delta2']} "
            f"counts {c['recip_iters_cheb']} depth {F.depth(c)} vs {F.depth(base)}"
            f"{': keeps its softmax' if keep else ''}")
        if not keep:
            cfg["softmax"][name] = c


def capfold_series(sm: dict, sc: dict, max_deg: int = CAPFOLD_MAX_DEG):
    """The folded exp series of softmax entry `sm` under soft-cap entry `sc`: (fields to update, degree, depth, error in
    the softmax's output), or None when no degree at the exp series' depth meets the target."""
    from numpy.polynomial import chebyshev as C
    cap = float(sc["cap"])
    sf = 2.0 ** (-int(sm["n_squarings"]) - int(sm["refinement_iters"]))
    c = (float(sm["clip_hi"]) + float(sm["clip_lo"])) / 2.0
    lim = 1.0 - 1e-12
    lo = max(float(sc["lo"]), cap * math.atanh(max(float(sm["clip_lo"]) / cap, -lim)))
    hi = min(float(sc["hi"]), cap * math.atanh(min(float(sm["clip_hi"]) / cap, lim)))
    mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    f = lambda x: np.exp(sf * (cap * np.tanh((mid + half * x) / cap) - c))
    t = np.cos(np.pi * (np.arange(8192) + 0.5) / 8192)
    want = f(t)
    gain = 2.0 ** (int(sm["n_squarings"]) + int(sm["refinement_iters"]))
    target = max(float(sc.get("fit_err", 0.0)), CAPFOLD_FLOOR)
    _, depth = series_depth(sm["cheb_coeffs"])
    best = None
    for deg in range(4, max_deg + 1):
        co = C.chebinterpolate(f, deg)
        n, dep = series_depth(co)
        if dep > depth:
            break
        err = gain * float(np.abs(C.chebval(t, co) / want - 1.0).max())
        if best is None or err < best[3]:
            best = (co, n, dep, err)
        if err <= target:
            break
    if best is None or best[3] > target:
        return None
    co, n, dep, err = best
    return ({"cheb_coeffs": [float(v) for v in co], "cheb_a": -half * sf, "cheb_b": half * sf,
             "clip_lo": lo, "clip_hi": hi, "cap_fold": 1}, n, dep, err)


def tune_capfold(cfg: dict, log=print):
    """`capfold` on every layer with a soft-cap and an uncentred softmax (module docstring)."""
    caps = cfg.get("softcap") or {}
    n = 0
    for name, sm in cfg["softmax"].items():
        sc = caps.get(name)
        if sc is None or sm.get("ref_keys") or sm.get("cap_fold"):
            continue
        r = capfold_series(sm, sc)
        if r is None:
            log(f"[capfold] {name}: no series at the exp's depth meets {sc.get('fit_err')}; kept")
            continue
        upd, deg, dep, err = r
        log(f"[capfold] {name}: raw [{upd['clip_lo']:.2f}, {upd['clip_hi']:.2f}] degree {deg} depth {dep} "
            f"err {err:.2e} (soft-cap degree {sc.get('degree')}, fit {sc.get('fit_err', 0):.2e})")
        sm.update(upd)
        n += 1
    log(f"[capfold] {n} of {len(cfg['softmax'])} softmax sites fold the soft-cap")


def tune_spru(cfg: dict, log=print):
    n = 0
    for name, j in cfg["norm"].items():
        if name.startswith("layers.") and j.get("cheb_coeffs"):
            j["spru"] = 1
            n += 1
    log(f"[spru] {n} block norm sites from one 1-slot refresh")


def tune_flags(cfg: dict, log=print):
    for j in cfg["softmax"].values():
        j.update(SOFTMAX_FLAGS)
    log(f"[flags] softmax {SOFTMAX_FLAGS} on {len(cfg['softmax'])} layers")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="arch.MODELS alias or HF name")
    ap.add_argument("--base", required=True, help="the calibrated config (perseus.examples.llama.calibrate)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="act,norms,design,centre,capfold,spru,flags")
    ap.add_argument("--kc-max", type=int, default=512, help="the longest cache the reciprocal counts cover")
    ap.add_argument("--docs", type=int, default=48, help="centre: documents fitted (as many again held out)")
    ap.add_argument("--owt", default=None, help="centre: OpenWebText shard 0042 (default: the HF cache's)")
    a = ap.parse_args(argv)
    from . import hf
    hf.select(a.model)
    ar = hf.arch()
    cfg = json.load(open(a.base))
    steps = [s for s in a.steps.split(",") if s]
    for s in steps:
        if s == "act":
            tune_act(cfg, ar.act)
        elif s == "design":
            tune_design(cfg, a.kc_max)
        elif s == "norms":
            tune_norms(cfg)
        elif s == "centre":
            tune_centre(cfg, a.kc_max, a.docs, owt=a.owt)
        elif s == "capfold":
            tune_capfold(cfg)
        elif s == "spru":
            tune_spru(cfg)
        elif s == "flags":
            tune_flags(cfg)
        else:
            sys.exit(f"tune: no step {s!r} (act, norms, design, centre, capfold, spru, flags)")
    cfg.setdefault("meta", {})["tune"] = {"steps": steps, "kc_max": a.kc_max, "base": os.path.basename(a.base)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"[tune] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
