#!/usr/bin/env python3
"""Bucket a [proftok] per-op wall table into op families (LN/SOFTMAX/GELU/LINEAR/ATTN/KV/
LMHEAD/OTHER) — the taxonomy of docs/fhe_per_op_timed.txt (2026-07-13 three-tier verdict).
Sums SELF ms per family (self excludes children, so families sum to the token total).

Usage: python scripts/parse_proftok.py <log1> [<log2> ...]   # one [proftok] table per log
"""
import re
import sys

ROW = re.compile(r"^  (\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")

FAMILY_RULES = [  # first match wins; order matters (softmax before attn, ln before linear)
    ("LMHEAD",  re.compile(r"lm_head")),
    ("KV",      re.compile(r"kv_offload|cache_kv_push|kv_pack|kv_push|cache_read")),
    ("GELU",    re.compile(r"gelu")),
    ("ATTN",    re.compile(r"qkt|softmax_v|\bpv\b|scores|head_reduce")),
    ("SOFTMAX", re.compile(r"softmax|exp_poly|attn.*(goldschmidt|recip|refine)|sm_")),
    ("LN",      re.compile(r"ln_1|ln_2|ln_f|ln1|ln2|lnf|layer_norm|inv_sqrt|newton|\.norm\b|\.norm:")),
    ("LINEAR",  re.compile(r"qkv|up_linear|down_linear|out_proj|linear|unpack_ri")),
]


def parse(path):
    fams, in_table = {}, False
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith(("[proftok]", "[profpre]")):
                in_table = True
                continue
            if not in_table:
                continue
            m = ROW.match(line.rstrip("\n"))
            if not m:
                if line.startswith("  step") or "[profile]" in line or not line.strip():
                    continue
                if in_table and line.strip() and not line.startswith("  "):
                    break  # table ended
                continue
            step, self_ms = m.group(1), float(m.group(3))
            fam = "OTHER"
            for name, rx in FAMILY_RULES:
                if rx.search(step):
                    fam = name
                    break
            fams[fam] = fams.get(fam, 0.0) + self_ms
    return fams


def main():
    results = {p: parse(p) for p in sys.argv[1:]}
    fam_order = ["SOFTMAX", "LN", "LINEAR", "GELU", "OTHER", "ATTN", "LMHEAD", "KV"]
    for p, fams in results.items():
        total = sum(fams.values())
        print(f"\n== {p}  (token total {total:.0f} ms)")
        for fam in fam_order:
            v = fams.get(fam, 0.0)
            print(f"  {fam:8s} {v:9.0f} ms  ({100*v/total:4.1f}%)")
    if len(results) == 2:
        (pa, fa), (pb, fb) = results.items()
        ta, tb = sum(fa.values()), sum(fb.values())
        print(f"\n== delta {pb} vs {pa}: total {tb:.0f} vs {ta:.0f} ms ({100*(tb-ta)/ta:+.1f}%)")
        for fam in fam_order:
            a, b = fa.get(fam, 0.0), fb.get(fam, 0.0)
            if a > 0:
                share = 100 * (a - b) / (ta - tb) if ta != tb else 0.0
                print(f"  {fam:8s} {a:9.0f} -> {b:9.0f} ms  ({100*(b-a)/a:+5.1f}%)  [{share:4.0f}% of saving]")


if __name__ == "__main__":
    main()
