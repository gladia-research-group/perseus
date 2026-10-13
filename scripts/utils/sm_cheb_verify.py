"""SM_CHEB_GS implementation check (CPU): (1) the packed goldschmidt_recip with impl smcheb constants
against the direct Chebyshev recursion and theory on each round's band; (2) ref.softmax_ref with SM_CHEB_GS=1 vs =0
against the exact softmax on all collected rows (32 windows x 12 heads x q < 160), errors propagating."""
import json, math, os, sys
import numpy as np
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, W)
from perseus.impl import smcheb
from perseus.impl.config import parse_softmax
from perseus.impl.poly import NumpyOps, goldschmidt_recip
D = os.environ.get("SM_SCORES_DIR", "")   # scores_l<layer>.npy + meta.npz (GPT-2 attention score rows)
CFG = json.load(open(f"{W}/configs/model/approximation/gpt2_base_n32/configs.json"))["softmax"]
ops = NumpyOps()
# (1) packed vs theory on the design bands
worst = 0.0
for l in range(12):
    cfg = parse_softmax(CFG[f"transformer.h.{l}.attn"])
    for kc in (1, 7, 32, 33, 100, 160):
        for j, (a, b, k, sc, ch) in enumerate(smcheb.rounds(cfg, kc)):
            lo, hi = (smcheb.band_from_seed(cfg.init_alpha, cfg.init_beta) if j == 0
                      else smcheb.refine_band(kc, float(cfg.recip_targets[j - 1])))
            x = np.geomspace(lo, hi, 4001)            # tot = scale * D
            Dv = x / sc
            r = np.real(goldschmidt_recip(ops, Dv, a, b, k, sc, Dh=0.5 * b * sc * sc * Dv, cheb=ch))
            got = np.abs(1 - Dv * r).max()
            th = smcheb.resid(lo, hi, k)
            if th > 1e-9: worst = max(worst, got / th)
            assert got <= th * (1 + 1e-6) + 1e-12, (l, kc, j, got, th)
print(f"[packed] max |1 - D R| / theory (theory > 1e-9) over 12 layers x 6 kc x rounds: {worst:.6f} (<= 1)"); import sys as _s; _s.exit(0) if os.environ.get("PACKED_ONLY") else None
# (2) end to end
os.environ["SM_CHEB_GS"] = "0"
sys.path.insert(0, W)
from examples.gpt2_from_primitives import ref
q = np.load(f"{D}/meta.npz")["q"]
tot = {"0": dict(le=0, gt=0, err=0, kl=[]), "1": dict(le=0, gt=0, err=0, kl=[])}
for l in range(12):
    cfg = parse_softmax(CFG[f"transformer.h.{l}.attn"])
    S = np.load(f"{D}/scores_l{l}.npy", mmap_mode="r")
    row = {}
    for flag in ("0", "1"):
        os.environ["SM_CHEB_GS"] = flag
        le = gt = err = 0.0
        kls = []
        for kc in range(1, 161):
            rows = np.asarray(S[q == kc - 1][:, :kc], dtype=np.float64)
            p = ref.softmax_ref(rows, cfg, kc)
            ex = np.exp(rows - rows.max(-1, keepdims=True)); ex /= ex.sum(-1, keepdims=True)
            e = np.abs(1 - p.sum(-1)).max()
            if kc <= 32: le = max(le, e)
            else: gt = max(gt, e)
            err = max(err, float(np.abs(p - ex).max()))
            pc = np.clip(p, 1e-300, None)
            kls.append(np.sum(ex * (np.log(np.clip(ex, 1e-300, None)) - np.log(pc)), -1))
        kls = np.concatenate(kls)
        row[flag] = (le, gt, err, float(np.median(kls)), float(kls.max()), bool(np.isfinite(p).all()))
        t = tot[flag]; t["le"] = max(t["le"], le); t["gt"] = max(t["gt"], gt); t["err"] = max(t["err"], err); t["kl"].append(kls)
    print(f"layer {l:2d}: " + " | ".join(f"SM_CHEB_GS={f}: |1-sum p| kc<=32 {v[0]:.1e} kc>32 {v[1]:.1e} max|p-exact| {v[2]:.1e} "
                                     f"KL med {v[3]:.1e} max {v[4]:.1e}" for f, v in row.items()), flush=True)
for f, t in tot.items():
    k = np.concatenate(t["kl"])
    print(f"ALL SM_CHEB_GS={f}: |1-sum p| kc<=32 {t['le']:.1e} kc>32 {t['gt']:.1e} max|p-exact| {t['err']:.1e} "
          f"KL median {np.median(k):.2e} max {k.max():.2e} (rows {len(k)})")
