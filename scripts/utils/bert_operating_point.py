"""Where does the ENCRYPTED run actually land relative to the fitted bands?

Every FHE approximation in configs.json is a fit over a DOMAIN that calibration
observed. If the deployed input drives a site outside that domain the run does
not warn — the Goldschmidt inverse-sqrt simply leaves its convergence basin and
the block detonates. This script runs the plaintext model twice, once on exactly
what `compose_get_batch` feeds the calibrator (random `block_size` windows over
the flat token pool) and once on exactly what `run_bert_forward` encrypts
(a `[CLS] ... [SEP]`-framed sentence), and prints both operating points against
the fitted band for every LayerNorm and GELU site.

Two numbers decide a norm site:
  * `cx`  = max |x - mean(x)| over the batch. Calibration sets
    center_scale = center_target / cx_pool, so a runtime cx ABOVE the pool's
    pushes the centered payload past center_target — silently.
  * `var_scaled` = center_scale_sq[pos] * var(x). This is the inverse-sqrt's
    literal input and it must sit inside [fit_lo, fit_hi]. NOTE
    center_scale_sq is indexed BY POSITION, fitted over `block_size` positions;
    a deployed sequence shorter than block_size uses the leading slice, so
    position 0 gets a rescale fitted on whatever token the pool happened to
    place there — never [CLS], since the pool is built with
    add_special_tokens=False.

A GELU site is decided by max |input| vs the fitted `xmax` (thor_composite fits
tanh at FIXED degree over [-S, S], so an input past xmax is unbounded garbage).

Usage (login node, CPU):
  HF_HOME=$SCRATCH/.cache python scripts/utils/bert_operating_point.py \
      --config configs/model/approximation/bert_base/configs.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def site_name(module_name):
    """HF module path -> the configs.json site key (mirrors export._bert_section)."""
    parts = module_name.split(".")
    if "layer" not in parts or parts.index("layer") + 1 >= len(parts):
        return None
    b = parts[parts.index("layer") + 1]
    if not b.isdigit():
        return None
    if module_name.endswith("attention.output.LayerNorm"):
        return f"transformer.h.{b}.ln_1"
    if module_name.endswith("output.LayerNorm"):
        return f"transformer.h.{b}.ln_2"
    if module_name.endswith("intermediate.intermediate_act_fn"):
        return f"transformer.h.{b}.mlp.act"
    return None


class Probe:
    """Collects per-site stats over however many batches are pushed through."""

    def __init__(self):
        self.norm = {}   # site -> dict(cx, var_by_pos {pos: [vals]}, out_abs)
        self.gelu = {}   # site -> max |input|
        self.hidden = {}  # block -> max |hidden|

    def hook_norm(self, site):
        def fn(mod, inp, out):
            x = inp[0].detach().float()
            xc = x - x.mean(-1, keepdim=True)
            var = xc.pow(2).mean(-1)                      # [B, T] biased, as LN does
            d = self.norm.setdefault(site, {"cx": 0.0, "var": {}, "out": 0.0})
            d["cx"] = max(d["cx"], float(xc.abs().amax()))
            d["out"] = max(d["out"], float(out.detach().abs().amax()))
            for t in range(var.shape[1]):
                d["var"].setdefault(t, []).extend(var[:, t].tolist())
        return fn

    def hook_gelu(self, site):
        def fn(mod, inp, out):
            v = float(inp[0].detach().abs().amax())
            self.gelu[site] = max(self.gelu.get(site, 0.0), v)
        return fn


def attach(model, probe):
    handles = []
    for name, mod in model.named_modules():
        s = site_name(name)
        if s is None:
            continue
        if s.endswith(".mlp.act"):
            handles.append(mod.register_forward_hook(probe.hook_gelu(s)))
        else:
            handles.append(mod.register_forward_hook(probe.hook_norm(s)))
    return handles


def run_pool(model, probe, pool, block_size, batches, micro_batch, device="cpu"):
    """EXACTLY compose_get_batch: uniform random `block_size` windows, no framing."""
    n = len(pool)
    g = torch.Generator().manual_seed(0)
    for _ in range(batches):
        ix = torch.randint(n - block_size, (micro_batch,), generator=g)
        x = torch.stack([torch.from_numpy(pool[i:i + block_size].astype(np.int64)) for i in ix])
        with torch.no_grad():
            model(input_ids=x.to(device), token_type_ids=torch.zeros_like(x).to(device))


def run_framed(model, probe, tok, sentences, device="cpu"):
    """EXACTLY run_bert_forward: one framed sentence at a time, natural length."""
    for s in sentences:
        enc = tok(s, return_tensors="pt")
        with torch.no_grad():
            model(input_ids=enc["input_ids"].to(device),
                  token_type_ids=enc.get("token_type_ids",
                                         torch.zeros_like(enc["input_ids"])).to(device))


def report(cfg, pool_probe, run_probe, n_pos_show=8):
    norm, gelu = cfg["norm"], cfg["softgelu"]
    blocks = sorted({int(k.split(".")[2]) for k in norm if k.startswith("transformer.h.")})

    print("\n" + "=" * 108)
    print("NORM SITES — cx (max|x-mean|) and the inverse-sqrt input var_scaled = css[pos]*var")
    print("=" * 108)
    hdr = (f"{'site':<22}{'c_scale':>9}{'cx_pool':>9}{'cx_run':>9}{'cx_r/p':>8}"
           f"{'band_lo':>10}{'band_hi':>9}{'vs_pool':>11}{'vs_run':>11}{'vs_r/hi':>9}  verdict")
    print(hdr)
    worst = []
    for b in blocks:
        for ln in ("ln_1", "ln_2"):
            site = f"transformer.h.{b}.{ln}"
            d = norm.get(site)
            if d is None or site not in run_probe.norm:
                continue
            css = d.get("center_scale_sq") or [d["center_scale"] ** 2]
            fl, fh = d["fit_lo"], d["fit_hi"]

            def vs_max(p):
                out = 0.0
                for t, vals in p.norm[site]["var"].items():
                    c = css[t] if t < len(css) else css[-1]
                    out = max(out, c * max(vals))
                return out

            vsp = vs_max(pool_probe) if site in pool_probe.norm else float("nan")
            vsr = vs_max(run_probe)
            cxp = pool_probe.norm[site]["cx"] if site in pool_probe.norm else float("nan")
            cxr = run_probe.norm[site]["cx"]
            ratio = vsr / fh
            flag = "OUT-OF-BAND" if ratio > 1.0 else ("edge" if ratio > 0.5 else "")
            if ratio > 1.0:
                worst.append((ratio, site))
            print(f"{site:<22}{d['center_scale']:>9.4g}{cxp:>9.4g}{cxr:>9.4g}{cxr/cxp:>8.2f}"
                  f"{fl:>10.4g}{fh:>9.4g}{vsp:>11.4g}{vsr:>11.4g}{ratio:>9.2f}  {flag}")

    print("\n" + "=" * 108)
    print("GELU SITES — fitted xmax vs the max |input| each regime actually produces")
    print("=" * 108)
    print(f"{'site':<22}{'xmax_fit':>11}{'max_pool':>11}{'max_run':>11}{'run/xmax':>10}  verdict")
    for b in blocks:
        site = f"transformer.h.{b}.mlp.act"
        d = gelu.get(site)
        if d is None or site not in run_probe.gelu:
            continue
        mp, mr = pool_probe.gelu.get(site, float("nan")), run_probe.gelu[site]
        r = mr / d["xmax"]
        flag = "OUT-OF-BAND" if r > 1.0 else ("edge" if r > 0.8 else "")
        print(f"{site:<22}{d['xmax']:>11.4g}{mp:>11.4g}{mr:>11.4g}{r:>10.2f}  {flag}")

    print("\n" + "=" * 108)
    print("PER-POSITION var (runtime) vs the css[pos] fitted for that position — block 0..3 ln_2")
    print("=" * 108)
    for b in blocks[:4]:
        site = f"transformer.h.{b}.ln_2"
        if site not in run_probe.norm:
            continue
        d = norm[site]
        css = d.get("center_scale_sq") or [d["center_scale"] ** 2]
        vr = run_probe.norm[site]["var"]
        vp = pool_probe.norm[site]["var"] if site in pool_probe.norm else {}
        print(f"\n{site}   fit=[{d['fit_lo']:.4g}, {d['fit_hi']:.4g}]")
        print(f"  {'pos':>4}{'css':>11}{'var_pool':>12}{'var_run':>12}{'run/pool':>10}"
              f"{'vs_run':>11}")
        for t in range(min(n_pos_show, max(vr) + 1)):
            c = css[t] if t < len(css) else css[-1]
            vrt = max(vr[t]) if t in vr else float("nan")
            vpt = max(vp[t]) if t in vp else float("nan")
            print(f"  {t:>4}{c:>11.4g}{vpt:>12.4g}{vrt:>12.4g}{vrt/vpt:>10.2f}{c*vrt:>11.4g}")

    if worst:
        worst.sort(reverse=True)
        print("\nFIRST OUT-OF-BAND SITE (by depth): " +
              min(worst, key=lambda w: int(w[1].split(".")[2]))[1])
        print("WORST OVERSHOOT: " + f"{worst[0][1]} at {worst[0][0]:.1f}x fit_hi")
    else:
        print("\nno norm site outside [fit_lo, fit_hi] in either regime")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model/approximation/bert_base/configs.json")
    ap.add_argument("--model", default="textattack/bert-base-uncased-SST-2")
    ap.add_argument("--pool", default=None, help="token pool .npy (calibration regime)")
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=5)
    ap.add_argument("--n-sentences", type=int, default=32, help="framed SST-2 sentences")
    ap.add_argument("--text", default="a masterpiece of modern cinema .",
                    help="the exact BERT_TEXT the gate encrypts")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from perseus.hub import cache_dir, load_model

    cfg = json.load(open(args.config))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, device="cpu", encoder_only=True)

    pool_path = args.pool or os.path.join(
        cache_dir("pools"), f"sst2_{args.model.split('/')[-1]}_800000.npy")
    pool = np.load(pool_path, mmap_mode="r")
    print(f"[probe] pool  {pool_path}  ({len(pool)} tokens)")

    # regime A — what calibration saw
    pool_probe = Probe()
    h = attach(model, pool_probe)
    run_pool(model, pool_probe, pool, args.block_size, args.batches, args.micro_batch)
    for x in h:
        x.remove()

    # regime B — what the encrypted gate runs: framed sentences, natural length
    from datasets import load_dataset
    sents = [args.text]
    try:
        ds = load_dataset("stanfordnlp/sst2", split="validation")
        sents += [ds[i]["sentence"] for i in range(args.n_sentences)]
    except Exception as e:                                  # offline / no dataset
        print(f"[probe] SST-2 validation unavailable ({e}); using --text only")
    run_probe = Probe()
    h = attach(model, run_probe)
    run_framed(model, run_probe, tok, sents)
    for x in h:
        x.remove()

    ids = tok(args.text)["input_ids"]
    print(f"[probe] gate text T={len(ids)} tokens={tok.convert_ids_to_tokens(ids)}")
    print(f"[probe] framed regime: {len(sents)} sentences, "
          f"lengths {min(len(tok(s)['input_ids']) for s in sents)}.."
          f"{max(len(tok(s)['input_ids']) for s in sents)}")
    report(cfg, pool_probe, run_probe)


if __name__ == "__main__":
    main()
