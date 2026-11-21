"""The runtime configs of Whisper's two halves from the calibrator's (`python -m perseus.calibrate
model=whisper_base_en dataset=librispeech_dev approximation=whisper_base_en`, or `model=whisper_base_en_dec
approximation=whisper_base_en_dec` for the decoder), refitted as the default plans' of their routes:

    .venv/bin/python -m perseus.examples.whisper.calibrate BASE.json OUT.json --audio DIR [--part decoder]

The encoder as ViT's (vit_base_224):
- norm: the LN_CHEB seeds on each LayerNorm's inverse-sqrt inputs over the clips (examples.vit.refit_norm.fit_sites),
  one centring scale per site, both statistics on one complex ladder (`zladder`);
- softmax: the active mask in the exp series (`exp_mask_fold`), the SM_CHEB_GS reciprocal counts designed up to
  --kc-max keys (scripts/utils/sm_cheb_design.py; the 1500 real keys of the 30-s window);
- softgelu: the MLP sites wider than --split-xmax range-split (calibrate/approximations/gelu.fit_split_gelu: hidden
  units sorted by range into the PCMM's output blocks, one approximation each: block 4 builds the massive channel
  from units reaching +-500); the THOR composites in Chebyshev bases (calibrate/thor_to_cheb.py).
The decoder as GPT-2's decode (gpt2_base_n32_cm):
- norm: the LN_CHEB seeds as above, keeping the per-position centring (the decode's positions are fixed: position 0,
  the start token, is its sink);
- softmax: SM_CHEB_GS counts up to --max-new + 2 keys for the self-attention, at the 1500 encoder keys for the
  cross-attention;
- softgelu: one Chebyshev series per site (scripts/utils/fit_gelu_series.py, the decode's GELU);
- cutmax: GPT-2's decode schedule (CUTMAX_REF's entry scale, powers and centring) with the scaled normalisers fitted
  on the decoder's logit rows over the emitted tokens (calibrate.approximations.cutmax.fit_scaled), every held-out row
  checked through the plaintext CutMax (examples.gpt2.ref.cutmax_ref);
- `lnf_fold`: ln_f's gain folded into the LM head, one product fewer at the end of the tail (CutMax's first masked
  product would cross the refresh envelope otherwise).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

from . import weights as W

_NORM = {"ln_1": "self_attn_layer_norm", "ln_2": "final_layer_norm", "ln_x": "encoder_attn_layer_norm"}
CUTMAX_REF = "configs/model/approximation/gpt2_base_n32_cm/configs.json"


def module_of(site, part="encoder"):
    """Runtime norm site -> the module of examples.whisper.hf.WhisperEncoderTaps / WhisperDecoderTaps it calibrates."""
    pre = "" if part == "encoder" else "decoder."
    if site == "transformer.ln_f":
        return pre + "layer_norm"
    p = site.split(".")
    return f"{pre}layers.{p[2]}.{_NORM[p[3]]}"


def taps_model(part):
    from . import hf
    return (hf.encoder if part == "encoder" else hf.decoder)(W.snapshot()).double().eval()


def collect_norm(cfgs, features, part="encoder"):
    """{site: the inverse-sqrt inputs z = c^2 (var + eps)} over every position of every clip (float64; c the site's
    centring scale at the position)."""
    import torch
    from perseus.impl.config import parse_norm
    model = taps_model(part)
    mods = dict(model.named_modules())
    out, hooks = {}, []
    for site, j in cfgs.items():
        c = parse_norm(j)

        def hook(mod, inp, _, site=site, c=c, eps=float(j["eps"])):
            x = inp[0].double()
            c2 = torch.tensor([c.c_eff_sq(p) for p in range(x.shape[-2])], dtype=torch.float64)
            cen = x - x.mean(-1, keepdim=True)
            out.setdefault(site, []).append((c2 * ((cen * cen).mean(-1) + eps)).reshape(-1).numpy())
        hooks.append(mods[module_of(site, part)].register_forward_hook(hook))
    with torch.no_grad():
        for f in features:
            model(torch.tensor(np.asarray(f)[None], dtype=torch.float64))
    for h in hooks:
        h.remove()
    return {k: np.concatenate(v) for k, v in out.items()}


def collect_units(sites, features):
    """{site: each hidden unit's max |GELU input|} of the encoder blocks' MLP sites over the clips (float64)."""
    import torch
    model = taps_model("encoder")
    mods = dict(model.named_modules())
    out, hooks = {}, []
    for site in sites:
        def hook(mod, inp, _, site=site):
            u = inp[0].double().abs().amax(dim=tuple(range(inp[0].dim() - 1))).numpy()
            out[site] = np.maximum(out.get(site, u), u)
        hooks.append(mods[f"layers.{site.split('.')[2]}.activation_fn"].register_forward_hook(hook))
    with torch.no_grad():
        for f in features:
            model(torch.tensor(np.asarray(f)[None], dtype=torch.float64))
    for h in hooks:
        h.remove()
    return out


def collect_logits(features):
    """The decoder's logit rows over the emitted tokens (hf.WhisperDecoderTaps: each greedy step's), float64."""
    import torch
    model = taps_model("decoder")
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(np.asarray(f)[None], dtype=torch.float64)).logits.numpy()
                               for f in features])


def fit_cutmax(rows, check):
    """CUTMAX_REF's schedule with its scaled normalisers fitted on `rows`; on the `check` rows the plaintext CutMax's
    argmax and its one-hot mass (the miss count, the least mass) printed."""
    from perseus._paths import resolve
    from perseus.calibrate.approximations.cutmax import fit_scaled
    from perseus.examples.gpt2.ref import cutmax_ref
    from perseus.impl.config import parse_cutmax
    ref = json.load(open(resolve(CUTMAX_REF)))["cutmax"]
    sec = fit_scaled([rows], {k: ref[k] for k in ("entry_scale", "p", "c", "m")})
    cfg = parse_cutmax(sec)
    z = [cutmax_ref(r, cfg) for r in check]
    miss = sum(int(np.argmax(v) != np.argmax(r)) for v, r in zip(z, check))
    print(f"cutmax: {len(rows)} rows fitted; held-out {len(check)} rows: {miss} argmax misses, least mass "
          f"{min(float(v.max()) for v in z):.4f}, most off-mass {max(float(np.abs(np.sort(v)[:-1]).sum()) for v in z):.3g}",
          flush=True)
    return sec


def thor_to_cheb(entry, site):
    """The THOR composites of a softgelu entry (a range split's blocks too) in Chebyshev bases, in place."""
    from perseus.calibrate.thor_to_cheb import convert_site
    for k, v in enumerate(entry.get("blocks", [])):
        thor_to_cheb(v, f"{site}.blocks.{k}")
    if entry.get("method") == "thor_composite":
        convert_site(entry, site)


def _script(name):
    from perseus._paths import ASSETS
    return str(ASSETS / "scripts" / "utils" / name)


def sm_cheb_design(path, *args):
    spec = importlib.util.spec_from_file_location("sm_cheb_design", _script("sm_cheb_design.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.main([path, "--write", *map(str, args)])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="the calibrator's configs.json")
    ap.add_argument("out")
    ap.add_argument("--audio", required=True, help="a directory of 16 kHz WAV clips (dev-clean)")
    ap.add_argument("--part", choices=("encoder", "decoder"), default="encoder")
    ap.add_argument("--n", type=int, default=48, help="clips the LayerNorm seeds are fitted on")
    ap.add_argument("--holdout", type=int, default=16, help="clips they are checked on")
    ap.add_argument("--kc-max", type=int, default=W.T, help="encoder: the softmax design's key count")
    ap.add_argument("--max-new", type=int, default=128, help="decoder: the tokens a transcription may generate")
    ap.add_argument("--safety", type=float, default=2.0)
    ap.add_argument("--target-err", type=float, default=1e-5)
    ap.add_argument("--max-deg", type=int, default=63)
    ap.add_argument("--max-nr", type=int, default=8, help="Newton steps after the seed (the sink tokens' wide bands)")
    ap.add_argument("--split-xmax", type=float, default=60.0,
                    help="encoder MLP GELU sites calibrated wider than this are range-split (THOR(31, 27) meets 4.7e-4 "
                         "to ~60)")
    ap.add_argument("--nch", type=int, default=512, help="hidden units per output block (the PCMM's)")
    ap.add_argument("--gelu-safety", type=float, default=1.5, help="a split block's xmax over its widest unit")
    a = ap.parse_args(argv)
    from perseus.examples.vit.refit_norm import fit_sites
    enc = a.part == "encoder"
    j = json.load(open(a.base))
    if enc:
        for v in j["norm"].values():     # one centring scale per site: the high-variance (sink) tokens move between
            v.pop("center_scale_sq", None)   # positions, a per-position scale widens every band (refit_norm --flat)
    files = sorted(glob.glob(os.path.join(a.audio, "*.wav")))[:a.n + a.holdout]
    path = W.snapshot()
    feats = [W.log_mel(W.load_audio(f), path) for f in files]
    z = collect_norm(j["norm"], feats[:a.n], a.part)
    zh = collect_norm(j["norm"], feats[a.n:], a.part) if len(feats) > a.n else z
    fit_sites(j["norm"], z, zh, a.safety, a.target_err, a.max_deg, a.max_nr)
    if enc:
        for v in j["norm"].values():
            v["zladder"] = 1
        for v in j["softmax"].values():
            v["exp_mask_fold"] = 2
        from perseus.calibrate.approximations.gelu import fit_split_gelu
        wide = [s for s, v in j["softgelu"].items() if s.endswith(".mlp.act") and v["xmax"] > a.split_xmax]
        units = collect_units(wide, feats[:a.n + a.holdout])
        for site in wide:
            j["softgelu"][site] = fit_split_gelu(units[site], a.nch, a.gelu_safety)
            print(f"{site:24s} range split: block xmax {[round(b['xmax'], 1) for b in j['softgelu'][site]['blocks']]} "
                  f"methods {[b['method'] for b in j['softgelu'][site]['blocks']]}", flush=True)
    for site, v in j["softgelu"].items():
        thor_to_cheb(v, site)
    j.setdefault("meta", {})["refit"] = {"tool": "examples.whisper.calibrate", "part": a.part, "clips": len(files),
                                         "n": a.n, "safety": a.safety, "target_err": a.target_err,
                                         **({"kc_max": a.kc_max} if enc else {"max_new": a.max_new})}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(j, f, indent=1)
    if enc:
        sm_cheb_design(a.out, "--kc-max", a.kc_max)
        return
    j["cutmax"] = fit_cutmax(collect_logits(feats[:a.n]), collect_logits(feats[a.n:]) if len(feats) > a.n else [])
    j["lnf_fold"] = 1
    with open(a.out, "w") as f:
        json.dump(j, f, indent=1)
    subprocess.run([sys.executable, _script("fit_gelu_series.py"), a.out, "--write"], check=True)
    sm_cheb_design(a.out, "--kc-max", a.max_new + len(W.Generation(path).start), "--sites", r"\.attn$")
    sm_cheb_design(a.out, "--kcs", W.T, "--sites", r"\.xattn$")


if __name__ == "__main__":
    main()
