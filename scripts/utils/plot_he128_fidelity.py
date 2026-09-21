#!/usr/bin/env python3
"""HE128 fidelity figure: teacher-forced FHE decode vs plaintext GPT-2 over 127 prompts.

ONE merged panel (single linear axis — agreement fractions and KL in nats both live in
[0,1], so no dual-scale): per-position top-1 agreement, mean top-5 overlap, and median
KL(GT||FHE) with IQR band. The merge shows the relation directly: KL grows smoothly with
decode depth while top-1 falls and top-5 overlap stays above it.

Data: logs/he128/results/he128_per_token.csv (collapsed sample(s) excluded — the run's 1
acknowledged failure, seed 1028). Style: repo notebooks/plots.ipynb palette + real LaTeX
text (usetex; /usr/bin/latex + dvipng present on Leonardo login). Cambridge blue snapped
#81b29a -> #729d88 for >=3:1 print contrast.

Usage: python scripts/plot_he128_fidelity.py [--csv ...] [--out logs/he128/results/he128_fidelity]
"""
import argparse
import csv
import statistics as st

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "lines.linewidth": 3,
})

PAL = {
    "slate":   "#335c67",   # Dark slate gray  -> top-1 agreement
    "camb":    "#729d88",   # Cambridge blue (contrast-snapped from #81b29a) -> top-5 overlap
    "auburn":  "#9e2a2b",   # Auburn -> KL
    "grid":    "#d9d9d3",
    "ink":     "#2b2b28",
    "muted":   "#6b6b64",
}


def load(csv_path):
    by_pos = {}
    excluded = set()
    rows = list(csv.DictReader(open(csv_path)))
    per_tid = {}
    for r in rows:
        if r["ref"] == "":
            continue
        per_tid.setdefault(r["tid"], []).append(r)
    for tid, ts in per_tid.items():
        deg = sum(1 for t in ts if t["degenerate"] == "True")
        if deg / len(ts) > 0.5:
            excluded.add(tid)
    for r in rows:
        if r["ref"] == "" or r["tid"] in excluded:
            continue
        d = by_pos.setdefault(int(r["pos"]), dict(match=[], ov=[], kl=[]))
        d["match"].append(1 if r["match"] == "True" else 0)
        d["ov"].append(int(r["top5_overlap"]))
        if r["degenerate"] != "True" and r["kl"]:
            d["kl"].append(float(r["kl"]))
    return by_pos, excluded


def rolling(y, w=8):
    y = np.asarray(y, float)
    out = np.empty_like(y)
    for i in range(len(y)):
        lo, hi = max(0, i - w // 2), min(len(y), i + w // 2 + 1)
        out[i] = y[lo:hi].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="logs/he128/results/he128_per_token.csv")
    ap.add_argument("--out", default="logs/he128/results/he128_fidelity")
    args = ap.parse_args()

    by_pos, excluded = load(args.csv)
    pos = np.array(sorted(by_pos))
    n = len(by_pos[pos[0]]["match"])
    top1 = np.array([np.mean(by_pos[p]["match"]) for p in pos])
    top5 = np.array([np.mean(by_pos[p]["ov"]) / 5.0 for p in pos])
    klm = np.array([st.median(by_pos[p]["kl"]) for p in pos])
    klq1 = np.array([np.percentile(by_pos[p]["kl"], 25) for p in pos])
    klq3 = np.array([np.percentile(by_pos[p]["kl"], 75) for p in pos])

    pooled1 = np.mean([m for p in pos for m in by_pos[p]["match"]])
    pooled5 = np.mean([o for p in pos for o in by_pos[p]["ov"]]) / 5.0
    kl_run = st.median([k for p in pos for k in by_pos[p]["kl"]])

    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.grid(axis="y", color=PAL["grid"], lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=PAL["muted"])
    for sp in ax.spines.values():
        sp.set_color(PAL["grid"])

    # KL: band + median (nats; same [0,1] numeric range as the agreement fractions).
    # Band edges rolling-smoothed like the lines — raw per-position quantiles are spiky.
    ax.fill_between(pos, rolling(klq1), rolling(klq3), color=PAL["auburn"], alpha=0.13,
                    lw=0, zorder=1)
    ax.plot(pos, rolling(klm), color=PAL["auburn"], zorder=3)
    ax.scatter(pos, klm, s=11, color=PAL["auburn"], alpha=0.28, lw=0, zorder=2)

    # agreement: raw per-position points faint + rolling-mean lines
    ax.scatter(pos, top5, s=13, color=PAL["camb"], alpha=0.28, lw=0, zorder=2)
    ax.scatter(pos, top1, s=13, color=PAL["slate"], alpha=0.28, lw=0, zorder=2)
    r5, r1 = rolling(top5), rolling(top1)
    ax.plot(pos, r5, color=PAL["camb"], zorder=4)
    ax.plot(pos, r1, color=PAL["slate"], zorder=4)

    ax.set_xlim(-1, 128)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 16, 32, 48, 64, 80, 96, 112, 127])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([r"0\%", r"25\%", r"50\%", r"75\%", r"100\%"])
    ax.set_xlabel(r"decode position $t$ (teacher-forced)")
    ax.set_ylabel(r"agreement percentage \,/\, "
                  r"$\mathrm{KL}\!\left(\ell_{\mathrm{enc}},\, \ell_{\mathrm{pt}}\right)$")

    ax.legend(handles=[
        plt.Line2D([], [], color=PAL["camb"], lw=3,
                   label=rf"mean top-5 overlap \ (overall {pooled5*100:.0f}\%)"),
        plt.Line2D([], [], color=PAL["slate"], lw=3,
                   label=rf"top-1 agreement \ (overall {pooled1*100:.0f}\%)"),
        plt.Line2D([], [], color=PAL["auburn"], lw=3,
                   label=rf"median $\mathrm{{KL}}\!\left(\ell_{{\mathrm{{enc}}}},\,"
                         rf" \ell_{{\mathrm{{pt}}}}\right)$ in nats \ (overall {kl_run:.2f})"),
        plt.Rectangle((0, 0), 1, 1, fc=PAL["auburn"], alpha=0.15, lw=0,
                      label=r"$\mathrm{KL}$ IQR across prompts"),
    ], loc="center left", frameon=False, bbox_to_anchor=(0.005, 0.24))

    ax.set_title(r"Decode fidelity over 128-token sequences (teacher-forced)",
                 color=PAL["ink"], pad=12)

    # methodology note (prompts count, excluded failure, rolling mean) goes in the paper caption
    fig.savefig(args.out + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(args.out + ".pdf", bbox_inches="tight", facecolor="white")
    print(f"saved {args.out}.png / .pdf  (n={n} prompts, excluded={sorted(excluded)})")


if __name__ == "__main__":
    main()
