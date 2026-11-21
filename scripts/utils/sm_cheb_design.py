"""SM_CHEB_GS design: per-layer Chebyshev-scaled Goldschmidt counts (recip_iters_cheb) and mid-round residual targets
(recip_targets) for the softmax reciprocal rounds, written into an approximation config (impl smcheb).

    python scripts/utils/sm_cheb_design.py configs/model/approximation/gpt2_base_n32/configs.json [--write]
        [--tgt-last 1e-3] [--kc-max 160] [--kcs 1500] [--sites REGEX]

Analytic (no data): round 0 on the band its calibrated seed (init_alpha, init_beta) is the minimax seed of, refine
rounds on the exact per-kc bands; the count of each round is the max over the design cache lengths up to --kc-max
(--kcs: those lengths only, e.g. a cross-attention's fixed key count). --sites: only the softmax sites matching the
regex. Prints today's counts (SM_GS_FIRST) next to the new ones."""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from perseus.impl import smcheb  # noqa: E402
from perseus.impl.config import parse_softmax  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--tgt-last", type=float, default=smcheb.TGT_LAST)
    ap.add_argument("--kc-max", type=int, default=160)
    ap.add_argument("--kcs", default=None, help="comma list of the design cache lengths (instead of up to --kc-max)")
    ap.add_argument("--sites", default=None, help="regex: only these softmax sites")
    a = ap.parse_args(argv)
    doc = json.load(open(a.config))
    kcs = tuple(sorted(int(k) for k in a.kcs.split(","))) if a.kcs else \
        tuple(sorted({k for k in smcheb.KC_DESIGN if k <= a.kc_max} | {a.kc_max}))
    tot_today = tot_new = 0
    for name in sorted(doc["softmax"], key=lambda n: [int(p) for p in n.split(".") if p.isdigit()]):
        if a.sites and not re.search(a.sites, name):
            continue
        j = doc["softmax"][name]
        cfg = parse_softmax(j)
        ks, tg = smcheb.design(cfg.init_alpha, cfg.init_beta, cfg.log2delta2, a.tgt_last, kcs)
        today = [cfg.gs_iters_first or cfg.gs_iters_scaled] + [int(x) for x in cfg.per_step_refine_iters]
        tot_today += sum(today)
        tot_new += sum(ks)
        print(f"{name:24s} today {today} = {sum(today):2d} | SM_CHEB_GS {ks} = {sum(ks):2d} targets {tg}")
        j["recip_iters_cheb"] = ks
        j["recip_targets"] = tg
    print(f"Goldschmidt levels per token: today {tot_today}, SM_CHEB_GS {tot_new} (kc {list(kcs)}, last {a.tgt_last:g})")
    if a.write:
        with open(a.config, "w") as f:
            json.dump(doc, f, indent=1)      # the config's own layout (no trailing newline)
        print(f"wrote {a.config}")


if __name__ == "__main__":
    main()
