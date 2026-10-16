"""Summarize a run_bert log: agreement with float64, label accuracy (encrypted / float64), KL, wall."""
import re
import statistics as st
import sys


def main(path):
    rows = [ln for ln in open(path) if "SUMMARY" in ln]
    agree = enc = ref = 0
    kls, walls, bad = [], [], []
    for ln in rows:
        g = [float(x) for x in re.search(r"logits=\[([^\]]*)\]", ln).group(1).split(",")]
        r = [float(x) for x in re.search(r"ref=\[([^\]]*)\]", ln).group(1).split(",")]
        y = int(re.search(r"label=(\d)", ln).group(1))
        kls.append(float(re.search(r"KL=([\d.e+-]+)", ln).group(1)))
        a = int(re.search(r"agree=(\d)", ln).group(1))
        agree += a
        enc += int((g[1] > g[0]) == bool(y))
        ref += int((r[1] > r[0]) == bool(y))
        walls.append(float(re.search(r"wall=([\d.]+)", ln).group(1)))
        if not a:
            bad.append(ln[:140])
    print(f"n={len(rows)} agree={agree} enc_acc={enc} ref_acc={ref} medianKL={st.median(kls):.3g} "
          f"maxKL={max(kls):.3g} wall median={st.median(walls):.1f}s min={min(walls):.1f} max={max(walls):.1f}")
    for b in bad:
        print("disagree:", b)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
