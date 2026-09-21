#!/usr/bin/env python3
"""Aggregate the 128-sample dense-bts decode sweep (62_decode_he128_chunked.slurm).

Robust metrics. IMPORTANT: the C++ per-token KL (kl_div in cli.cu) UNDERFLOWS to ~0 when the FHE
logits blow up (softmax -> one-hot on a garbage token, every other prob underflows to 0, and the
`q[i]>0` guard skips them) — so a blown/degenerate token reports KL~=0, NOT a large value. We
therefore detect collapse from top5_overlap, NOT KL:
  - a token is DEGENERATE (blown logits) if top5_overlap==0 AND |KL|<1e-3.
  - a sample is COLLAPSED if >50% of its GT positions are degenerate.
KL statistics are reported over CLEAN (non-degenerate) tokens only; top1-agreement is over all GT
tokens (it is immune to the KL artifact). Degenerate/collapse rates are reported separately.

Usage: python scripts/aggregate_he128.py [--logdir logs/he128] [--outdir logs/he128/results]
"""
import argparse
import glob
import json
import os
import re
import statistics as st

TOK_RE = re.compile(
    r"\[cuda_cachemir\] tok(\d+) top1=(\d+)"
    r"(?: ref=(\d+) KL=([\d.eE+-]+) top5_overlap=(\d+)/5)?"
)
SUM_RE = re.compile(
    r"SUMMARY cmd=\w+ completed=(\d+)/(\d+) bootstraps=(\d+) unplanned_bts=(\d+) "
    r"weight_relevels=(\d+) s/tok=([\d.eE+-]+)(?: argmax_s/tok=([\d.eE+-]+))?"
    r"(?: e2e_s/tok=([\d.eE+-]+))?"
)


def is_degenerate(t):
    return t["ref"] is not None and t["overlap"] == 0 and abs(t["kl"]) < 1e-3


def parse_log(path):
    toks, summ = [], None
    with open(path, errors="replace") as f:
        for line in f:
            m = TOK_RE.search(line)
            if m:
                pos, top1 = int(m.group(1)), int(m.group(2))
                if m.group(3) is None:
                    toks.append(dict(pos=pos, top1=top1, ref=None, kl=None, overlap=None))
                else:
                    toks.append(dict(pos=pos, top1=top1, ref=int(m.group(3)),
                                     kl=float(m.group(4)), overlap=int(m.group(5))))
                continue
            s = SUM_RE.search(line)
            if s:
                summ = dict(completed=int(s.group(1)), requested=int(s.group(2)),
                            bootstraps=int(s.group(3)), unplanned_bts=int(s.group(4)),
                            weight_relevels=int(s.group(5)), s_per_tok=float(s.group(6)),
                            argmax_s_per_tok=float(s.group(7)) if s.group(7) else None,
                            e2e_s_per_tok=float(s.group(8)) if s.group(8) else None)
    return toks, summ


def sample_row(tid, toks, summ):
    gt = [t for t in toks if t["ref"] is not None]
    hits = sum(1 for t in gt if t["top1"] == t["ref"])
    deg = [t for t in gt if is_degenerate(t)]
    clean = [t for t in gt if not is_degenerate(t)]
    kls = [t["kl"] for t in clean]
    row = dict(
        tid=tid, seed=1000 + tid, n_tokens=len(toks), n_gt=len(gt),
        top1_hits=hits,
        top1_acc=hits / len(gt) if gt else None,
        n_degenerate=len(deg),
        degenerate_frac=len(deg) / len(gt) if gt else None,
        collapsed=(len(deg) / len(gt) > 0.5) if gt else False,
        first_degen_pos=(min(t["pos"] for t in deg) if deg else None),
        kl_mean_clean=sum(kls) / len(kls) if kls else None,
        kl_median_clean=st.median(kls) if kls else None,
        top5_overlap_mean=(sum(t["overlap"] for t in gt) / len(gt)) if gt else None,
    )
    if summ:
        row.update(completed=summ["completed"], requested=summ["requested"],
                   weight_relevels=summ["weight_relevels"], bootstraps=summ["bootstraps"],
                   s_per_tok=summ["s_per_tok"], argmax_s_per_tok=summ["argmax_s_per_tok"],
                   e2e_s_per_tok=summ["e2e_s_per_tok"])
    else:
        row.update(completed=None, requested=None, weight_relevels=None, bootstraps=None,
                   s_per_tok=None, argmax_s_per_tok=None, e2e_s_per_tok=None)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="logs/he128")
    ap.add_argument("--outdir", default="logs/he128/results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    by_tid = {}
    for path in glob.glob(os.path.join(args.logdir, "decode_*.out")):
        m = re.search(r"decode_(\d+)_(\d+)\.out$", os.path.basename(path))
        if not m:
            continue
        tid = int(m.group(1))
        if tid not in by_tid or os.path.getmtime(path) > os.path.getmtime(by_tid[tid]):
            by_tid[tid] = path

    rows, per_token = [], []
    for tid in sorted(by_tid):
        toks, summ = parse_log(by_tid[tid])
        rows.append(sample_row(tid, toks, summ))
        for t in toks:
            per_token.append(dict(tid=tid, pos=t["pos"], top1=t["top1"], ref=t["ref"],
                                  match=(t["ref"] is not None and t["top1"] == t["ref"]),
                                  kl=t["kl"], top5_overlap=t["overlap"],
                                  degenerate=is_degenerate(t)))

    ps_cols = ["tid", "seed", "n_tokens", "n_gt", "top1_hits", "top1_acc", "n_degenerate",
               "degenerate_frac", "collapsed", "first_degen_pos", "kl_mean_clean",
               "kl_median_clean", "top5_overlap_mean", "s_per_tok", "argmax_s_per_tok",
               "e2e_s_per_tok", "completed", "requested", "weight_relevels", "bootstraps"]
    with open(os.path.join(args.outdir, "he128_per_sample.csv"), "w") as f:
        f.write(",".join(ps_cols) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in ps_cols) + "\n")

    pt_cols = ["tid", "pos", "top1", "ref", "match", "kl", "top5_overlap", "degenerate"]
    with open(os.path.join(args.outdir, "he128_per_token.csv"), "w") as f:
        f.write(",".join(pt_cols) + "\n")
        for r in per_token:
            f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in pt_cols) + "\n")

    def agg(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return dict(n=len(vals), mean=sum(vals) / len(vals), median=st.median(vals),
                    stdev=st.pstdev(vals) if len(vals) > 1 else 0.0,
                    min=min(vals), max=max(vals))

    clean_samples = [r for r in rows if not r["collapsed"]]
    collapsed = [r for r in rows if r["collapsed"]]
    collapsed_tids = {r["tid"] for r in collapsed}
    gt_tokens = [t for t in per_token if t["ref"] is not None]
    n_gt = len(gt_tokens)
    n_deg = sum(1 for t in gt_tokens if t["degenerate"])
    pooled_top1 = sum(1 for t in gt_tokens if t["match"])
    # DELIVERABLE: pool over non-collapsed samples only (failures acknowledged, not averaged in)
    gt_clean = [t for t in gt_tokens if t["tid"] not in collapsed_tids]
    pooled_top1_nc = sum(1 for t in gt_clean if t["match"])

    # per-position over NON-COLLAPSED samples: top1 acc, clean-KL median (failures excluded)
    pos_top1, pos_deg, pos_kl = {}, {}, {}
    for t in gt_clean:
        pos_top1.setdefault(t["pos"], []).append(1 if t["match"] else 0)
        pos_deg.setdefault(t["pos"], []).append(1 if t["degenerate"] else 0)
        if not t["degenerate"] and t["kl"] is not None:
            pos_kl.setdefault(t["pos"], []).append(t["kl"])
    per_pos = []
    for pos in sorted(pos_top1):
        a, d = pos_top1[pos], pos_deg[pos]
        kls = sorted(pos_kl.get(pos, []))
        per_pos.append(dict(pos=pos, n=len(a),
                            top1_acc=sum(a) / len(a),
                            degenerate_frac=sum(d) / len(d),
                            kl_median_clean=(st.median(kls) if kls else None)))

    n_gt_nc = len(gt_clean)
    summary = dict(
        n_samples=len(rows),
        # --- acknowledged failures (EXCLUDED from the deliverable metrics below) ---
        n_collapsed=len(collapsed),
        collapsed_samples=[dict(tid=r["tid"], seed=r["seed"], degenerate_frac=round(r["degenerate_frac"], 3),
                                first_degen_pos=r["first_degen_pos"], top1_acc=round(r["top1_acc"], 4))
                           for r in collapsed],
        # --- DELIVERABLE metrics: over non-collapsed samples only ---
        n_reported=len(clean_samples),
        pooled_top1_acc=pooled_top1_nc / n_gt_nc if n_gt_nc else None,
        top1_acc_per_sample=agg([r["top1_acc"] for r in clean_samples]),
        kl_median_clean_per_sample=agg([r["kl_median_clean"] for r in clean_samples]),
        top5_overlap_mean_per_sample=agg([r["top5_overlap_mean"] for r in clean_samples]),
        s_per_tok=agg([r["s_per_tok"] for r in clean_samples]),
        argmax_s_per_tok=agg([r["argmax_s_per_tok"] for r in clean_samples]),
        e2e_s_per_tok=agg([r["e2e_s_per_tok"] for r in clean_samples]),
        # context (all samples incl. failures)
        pooled_top1_acc_incl_failures=pooled_top1 / n_gt if n_gt else None,
        per_position=per_pos,
    )
    with open(os.path.join(args.outdir, "he128_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    def fmt(a):
        return "n/a" if not a else f"mean={a['mean']:.4g} med={a['median']:.4g} sd={a['stdev']:.3g} [{a['min']:.4g},{a['max']:.4g}]"
    print(f"samples decoded: {len(rows)}   reported: {len(clean_samples)}   "
          f"acknowledged failures (excluded): {len(collapsed)}")
    for r in collapsed:
        print(f"  [failure] sample {r['tid']} (seed {r['seed']}): collapsed ~pos {r['first_degen_pos']}, "
              f"{r['degenerate_frac']*100:.0f}% degenerate, top1 {r['top1_hits']}/{r['n_gt']}")
    print(f"--- DELIVERABLE (over {len(clean_samples)} non-collapsed samples) ---")
    print(f"pooled top1 agreement : {pooled_top1_nc}/{n_gt_nc} = {100*pooled_top1_nc/n_gt_nc:.2f}%")
    print(f"top1_acc / sample     : {fmt(summary['top1_acc_per_sample'])}")
    print(f"KL median / sample    : {fmt(summary['kl_median_clean_per_sample'])}")
    print(f"top5_overlap / sample : {fmt(summary['top5_overlap_mean_per_sample'])}")
    print(f"s/tok (decode)        : {fmt(summary['s_per_tok'])}")
    print(f"argmax s/tok          : {fmt(summary['argmax_s_per_tok'])}")
    print(f"e2e s/tok             : {fmt(summary['e2e_s_per_tok'])}")


if __name__ == "__main__":
    main()
