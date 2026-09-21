"""Aggregate the he128x encrypted arrays into the F7 figure contract.

Parses logs/core/he128x_<arm>_<tid>_<aid>.out per-token lines
  [cuda_cachemir] tokN top1=X ref=Y KL=Z top5_overlap=A/5
into  logs/he128x/<run>/results/he128_per_token.csv  (tid,pos,ref,top1,kl,
top5_overlap,degenerate) + he128_summary.json {e2e_s_per_tok:{median,mean,n}} —
the exact format figures/gpt2/he128_health/plot_he128_health.py loads.
`degenerate` (the old campaign's collapse marker) is derived: unparseable, KL<0,
|KL|>50, or top5_overlap==0 with |KL|<1e-3. A blown token reports KL~=0, NOT a large
value (e33-scale logits saturate the softmax), so the sign/overlap tests are the ones
that fire — aggregate_he128.py documents the same and always got this right; the
`|KL|>50` rule alone matched ZERO rows while whole chains sat at 0.8% top-1. The
figure's >50%-degenerate collapsed-tid exclusion then applies unchanged.
Usage: aggregate_he128x.py <arm> <array_id> <run_name>
e.g.  aggregate_he128x.py heat 50613471 he128_heat
"""
import csv
import glob
import json
import re
import statistics
import sys
from pathlib import Path

arm, aid, run = sys.argv[1], sys.argv[2], sys.argv[3]
REPO = Path(__file__).resolve().parents[2]
out = REPO / "logs" / "he128x" / run / "results"
out.mkdir(parents=True, exist_ok=True)

TOK = re.compile(r"\[cuda_cachemir\] tok(\d+) top1=(\d+) ref=(\d+) KL=([-+\d.eE]+) "
                 r"top5_overlap=(\d)/5")
SUM = re.compile(r"SUMMARY cmd=decode completed=(\d+)/128 .* e2e_s/tok=([\d.]+)")

rows, e2es, done = [], [], 0
for f in sorted(glob.glob(str(REPO / "logs/core" / f"he128x_{arm}_*_{aid}.out"))):
    tid = int(Path(f).stem.split("_")[2])
    txt = open(f, errors="replace").read()
    m = SUM.search(txt)
    if not m or m.group(1) != "128":
        continue
    done += 1
    e2es.append(float(m.group(2)))
    for t in TOK.finditer(txt):
        pos, top1, ref, kl, ov = t.groups()
        try:
            klv = float(kl)
            # A blown token does NOT report a large KL. e33-scale logits saturate the
            # softmax, so the divergence collapses toward 0 -- often to an impossible
            # NEGATIVE value -- and the old `abs(klv) > 50` test fired on ZERO rows while
            # whole chains sat at 0.8% top-1. Detect it the way aggregate_he128.py already
            # documents (top5_overlap, not KL), plus a guard on the impossible sign.
            # Measured on the 2026-08-15 A100 arrays: 774 rows caught across base+squeeze,
            # 0 of which had a correct top-1; heat caught 0.
            degen = klv < 0 or abs(klv) > 50 or (int(ov) == 0 and abs(klv) < 1e-3)
        except ValueError:
            klv, degen = "", True
        rows.append(dict(tid=tid, pos=int(pos), ref=ref, top1=top1, kl=klv,
                         top5_overlap=ov, degenerate=degen))

with open(out / "he128_per_token.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["tid", "pos", "ref", "top1", "kl",
                                       "top5_overlap", "degenerate"])
    w.writeheader()
    w.writerows(rows)
json.dump({"e2e_s_per_tok": {"median": statistics.median(e2es),
                             "mean": statistics.fmean(e2es), "n": done}},
          open(out / "he128_summary.json", "w"), indent=1)

kl0 = [r["kl"] for r in rows if r["pos"] < 32 and not r["degenerate"] and r["kl"] != ""]
kl1 = [r["kl"] for r in rows if r["pos"] >= 96 and not r["degenerate"] and r["kl"] != ""]
print(f"[agg] {run}: {done} complete chains, {len(rows)} rows -> {out}")
if kl0 and kl1:
    print(f"[agg] median KL t<32 = {statistics.median(kl0):.4f} | "
          f"t>=96 = {statistics.median(kl1):.4f} | e2e median {statistics.median(e2es):.2f}")
