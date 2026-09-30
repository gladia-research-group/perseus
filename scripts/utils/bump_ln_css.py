#!/usr/bin/env python
"""bump_ln_css.py — raise per-position `center_scale_sq` so LN's inv_sqrt_newton bootstrap
input lands back under the EvalMod wall (n32 round-N css retune).

Why: at n32 the GPT-2 eager decode detonates because `inv_sqrt_newton`'s bootstrap input
    m = y^2 = (s / sqrt(z))^2 = s^2 / z          (s = the site's `inv_out_scale`)
exceeds the |m| ~ 10 wall at a handful of LN sites. The cause is the OTHER direction from
what the name suggests: the scaled variance z UNDERSHOOTS the calibrated band, so y (and
hence y^2) blows up. The fix is to push z back UP into the band by scaling the per-token
centering, which is output-invariant: LN(c*x) == LN(x), and `weight_loader.h:203` folds
1/inv_out_scale into gamma, so nothing downstream sees c_eff.

Where the knob lives: `norm.cu:136-141` selects
    c_eff_sq = cfg.center_scale_sq[clamp(inf.output.capture_t, 0, len-1)]
i.e. ONE entry of the array per token position, and decode sets capture_t = t
(`gpt2_decode.cu:36,76`). So a per-(site, position) probe maps to exactly one array slot,
and bumping that slot moves z for that position only:
    z = c_eff_sq * (Var + eps)   =>   css[pos] *= k   scales z by k.

The inversion, per surviving probe:
    z       = inv_out_scale^2 / m            (invert m = s^2/z)
    z_target= target_frac * fit_lo           (land just inside the low edge of the band)
    k       = z_target / z
    css_new = css_old * k                    (resulting y^2 = s^2 / z_target)

This is NOT the `rescale_ln_center.py` transform, whose global-gamma premise does not
hold: no Remez coefficient touches, no band move, no refit.
Only `center_scale_sq[pos]` entries change; every other byte of the config is copied
through unchanged (verified by per-section md5).

Guards:
  * COLLAPSE GUARD (--max-m): once a token's probe stream shows a detonated value, every
    LATER row for that token is post-collapse garbage — inverting it would "correct" noise.
    Those rows are dropped and reported.
  * BASIN GUARD (--basin-safety, mirrors `rescale_basin_safety` in
    perseus/configs/approximation/gpt2.yaml): refuse any bump whose z_target would sit
    above basin_safety * fit_hi, i.e. past the top of the fitted band.

Usage:
  python scripts/utils/bump_ln_css.py \
      --log logs/core/n32_eager2_base.out \
      --src configs/model/approximation/gpt2_base_n32/configs.json \
      --dst configs/model/approximation/gpt2_base_n32_cssr1/configs.json
"""
import argparse
import hashlib
import json
import os
import re
import sys

# [bts_input] step=tok0.block.blk2.transformer_block.ln_1.norm:ln_1.inv_sqrt_newton.bootstrap
#             var=v_4029 level=46 deg=2 min=11.8 max=11.9 avg=11.9 |abs|max=11.9
PROBE_RE = re.compile(
    r"\[bts_input\]\s+step=tok(?P<tok>\d+)\.(?P<step>\S*inv_sqrt_newton\.bootstrap)"
    r".*?\|abs\|max=(?P<m>[-+0-9.eE]+)"
)
BLK_RE = re.compile(r"\bblk(?P<blk>\d+)\b")
SITE_RE = re.compile(r"norm:(?P<site>ln_1|ln_2|ln_f)\.inv_sqrt_newton\.bootstrap$")
# ln_f runs after every block, so it sorts last under the collapse guard.
LN_F_ORDER = 10 ** 6


def step_to_key(step):
    """Map a runtime step path to its `norm` config key. Returns (key, block_order)."""
    m = SITE_RE.search(step)
    if not m:
        raise ValueError(f"unrecognised inv_sqrt_newton step path: {step!r}")
    site = m.group("site")
    if site == "ln_f":
        return "transformer.ln_f", LN_F_ORDER
    b = BLK_RE.search(step)
    if not b:
        raise ValueError(f"no blkN in step path: {step!r}")
    blk = int(b.group("blk"))
    return f"transformer.h.{blk}.{site}", blk


def parse_log(path, tokens, max_m):
    """Return (rows, dropped). rows = [(tok, key, blk, m)], max-aggregated per (tok, key).

    Rows are consumed in LOG ORDER, which is execution order; the collapse guard trips on
    the first probe past --max-m and discards it plus everything after it for that token.
    """
    per_token = {}          # tok -> {key: (blk, m)}
    collapsed = {}          # tok -> (key, blk, m) that tripped the guard
    dropped = []            # [(tok, key, blk, m)] discarded post-collapse
    bad = []
    with open(path, errors="replace") as fh:
        for ln, line in enumerate(fh, 1):
            g = PROBE_RE.search(line)
            if not g:
                continue
            tok = int(g.group("tok"))
            if tokens is not None and tok not in tokens:
                continue
            try:
                key, blk = step_to_key(g.group("step"))
            except ValueError as e:
                bad.append(f"{path}:{ln}: {e}")
                continue
            m = abs(float(g.group("m")))
            if tok in collapsed:
                dropped.append((tok, key, blk, m))
                continue
            if m > max_m:
                collapsed[tok] = (key, blk, m)
                dropped.append((tok, key, blk, m))
                continue
            d = per_token.setdefault(tok, {})
            if key not in d or m > d[key][1]:
                d[key] = (blk, m)
    if bad:
        sys.exit("[css_bump] UNPARSEABLE probe step paths:\n  " + "\n  ".join(bad))
    rows = [(tok, key, blk, m)
            for tok, d in sorted(per_token.items())
            for key, (blk, m) in sorted(d.items(), key=lambda kv: (kv[1][0], kv[0]))]
    return rows, collapsed, dropped


def section_hashes(cfg):
    h = lambda x: hashlib.md5(json.dumps(x, sort_keys=True).encode()).hexdigest()
    return {k: h(v) for k, v in cfg.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="DEBUG=1 log carrying [bts_input] probes")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--target-frac", type=float, default=1.3,
                    help="land z at target_frac * fit_lo (default 1.3)")
    ap.add_argument("--tokens", default="0,1",
                    help="comma-separated token positions to consider, or 'all'")
    ap.add_argument("--min-m", type=float, default=10.0,
                    help="only bump sites whose probe exceeds this (the EvalMod wall)")
    ap.add_argument("--max-m", type=float, default=1e3,
                    help="collapse guard: a probe above this detonates the token's stream")
    ap.add_argument("--basin-safety", type=float, default=0.85,
                    help="refuse a bump whose z_target > basin_safety * fit_hi")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tokens = None if a.tokens.strip() == "all" else \
        {int(x) for x in a.tokens.split(",") if x.strip()}

    cfg = json.load(open(a.src))
    if "norm" not in cfg:
        sys.exit(f"[css_bump] {a.src}: no `norm` section")
    norm = cfg["norm"]
    src_hashes = section_hashes(cfg)

    rows, collapsed, dropped = parse_log(a.log, tokens, a.max_m)
    if not rows and not dropped:
        sys.exit(f"[css_bump] {a.log}: no inv_sqrt_newton [bts_input] probes found")

    # Every parsed site must resolve to a real config key — a silent miss would leave the
    # offending site un-bumped and the run would fail identically next round.
    unknown = sorted({k for _, k, _, _ in rows + [(t, k, b, m) for t, k, b, m in dropped]
                      if k not in norm})
    if unknown:
        sys.exit("[css_bump] step paths resolved to keys ABSENT from `norm`:\n  "
                 + "\n  ".join(unknown)
                 + "\n  config has: " + ", ".join(sorted(norm)))

    for tok, (key, blk, m) in sorted(collapsed.items()):
        n = sum(1 for t, _, _, _ in dropped if t == tok)
        print(f"[css_bump] COLLAPSE tok{tok}: first |m|={m:.3g} > max_m at {key} "
              f"(blk{blk}) — dropped {n} row(s) from there on (post-collapse garbage)")

    print(f"\n{'site':<22} {'pos':>3} {'m':>10} {'z':>11} {'z/fit_lo':>9} "
          f"{'k':>8} {'css_old':>12} {'css_new':>12} {'y2_new':>8}")
    print("-" * 104)

    bumped = skipped_low = refused = 0
    for tok, key, _blk, m in rows:
        if m <= a.min_m:
            skipped_low += 1
            continue
        site = norm[key]
        css = site.get("center_scale_sq")
        if not css:
            sys.exit(f"[css_bump] {key}: no center_scale_sq array to bump")
        pos = min(tok, len(css) - 1)
        s = site["inv_out_scale"]
        fit_lo, fit_hi = site["fit_lo"], site["fit_hi"]
        z = s * s / m
        z_target = a.target_frac * fit_lo
        if z_target > a.basin_safety * fit_hi:
            print(f"[css_bump] REFUSED {key} pos{tok}: z_target={z_target:.4g} > "
                  f"{a.basin_safety}*fit_hi={a.basin_safety * fit_hi:.4g} — the target is "
                  f"outside the fitted basin; retune the band, not the centering")
            refused += 1
            continue
        k = z_target / z
        old = css[pos]
        new = old * k
        css[pos] = new
        y2_new = s * s / z_target
        print(f"{key:<22} {tok:>3} {m:>10.4g} {z:>11.4g} {z / fit_lo:>9.4g} "
              f"{k:>8.4g} {old:>12.6g} {new:>12.6g} {y2_new:>8.4g}")
        bumped += 1

    print("-" * 104)
    print(f"[css_bump] {bumped} css entr{'y' if bumped == 1 else 'ies'} bumped "
          f"(target {a.target_frac}x fit_lo), {skipped_low} probe(s) already <= "
          f"min_m={a.min_m}, {refused} refused by the basin guard, "
          f"{len(dropped)} dropped post-collapse")

    if a.dry_run:
        print("[css_bump] --dry-run: nothing written")
        return
    if not bumped:
        sys.exit("[css_bump] nothing to bump — refusing to write a pointless copy")

    # Faithful full copy: only `norm` may differ, and inside it only css entries. indent=1
    # is the source's own formatting (verified: json.dumps(cfg, indent=1) round-trips it).
    dst_hashes = section_hashes(cfg)
    changed = [k for k in src_hashes if src_hashes[k] != dst_hashes.get(k)]
    if changed != ["norm"]:
        sys.exit(f"[css_bump] REFUSING to write: sections changed = {changed}, expected "
                 f"['norm'] only")
    print("[css_bump] section md5 check: " + ", ".join(
        f"{k}={'CHANGED' if k in changed else 'identical'}" for k in src_hashes))

    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    with open(a.dst, "w") as fh:
        json.dump(cfg, fh, indent=1)
    print(f"[css_bump] wrote {a.dst}")


if __name__ == "__main__":
    main()
