#!/usr/bin/env python3
"""Render a self-contained interactive flame CHART from a macro_run.sh profile log.

  python scripts/utils/make_flamechart.py --log logs/macro/<TAG>.out [--out <file.html>]

With `FHE_PROFILE_TRACE=1` the profiler emits `[proftrace]` — one record per scope INSTANCE
with real start/end timestamps — and this renders a true flame CHART: every frame sits at the
moment it ran, and a gap is the parent genuinely doing its own work. Without it, only the
aggregate `[proftok]` totals exist, which can support a flameGRAPH (children packed from the
parent's left edge, x carrying no meaning) but not a chart. Prefer the trace.

Output is one HTML file with no external references, openable straight from disk.

Why this exists rather than perf+FlameGraph.pl: the scope tree is EXACT, not sampled, and it
carries the FHE structure — bootstrap split by sparse route, residency plumbing, per-block.
A sampled host profile of this workload mostly shows the main thread waiting on the GPU.

 Read shares, not milliseconds. `wall` mode syncs the device at every scope boundary and
inflates the token ~1.39x. Relative width is meaningful; absolute width is not.
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import re
import sys
from pathlib import Path

ROW = re.compile(r"^  (\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")

# ── execution order ───────────────────────────────────────────────────────────────────────
# Frames run left-to-right in EXECUTION order, not sorted by size: with 12 sequential blocks a
# size sort destroys the one thing the x-axis could carry. The profiler does not record
# emission order, so it is recovered from two places:
#   1. SEQ below — the Op labels as they appear in the source, which ARE emitted in order.
#   2. natural sort — the numbered sequences (blk0..blk11, i0..i4, p0..p2, iter_N); plain
#      alphabetical would put blk10 straight after blk1.
SEQ = {n: i for i, n in enumerate([
    "decode", "argmax",
    "mask_evict", "mask_prime", "kv_prefetch_first", "block", "res_acquire", "res_install",
    "res_arena_drain", "res_extract_wait", "res_acquire_wait", "res_release", "res_pipeline_sync",
    "enc_cache_evict", "kv_finalize_last", "lnf_install", "lnf_sync", "layer_norm:ln_f", "lm_head",
    "kv_reload", "transformer_block",
    "block_in", "ln_1", "attn_residual", "ln_2", "mlp_residual",   # gpt2_block.cu:58-87
    "qkv", "attn_core", "out_proj",                                 # mha.cu:28-38
    "up_linear", "gelu", "down_linear",                             # mlp.cu
    "block_sync", "kv_offload",                                     # gpt2_residency.cu:344-346
    "entry", "cascade", "sum",                                      # cutmax
])}

def natkey(s: str):
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s))

# ── op family (drives colour) ─────────────────────────────────────────────────────────────
# Three categorical slots + an ordinal ramp for the bootstrap routes. A flamegraph is an
# ALL-PAIRS colour case (any frame can neighbour any other) and no 5-family subset of the
# reference palette clears the all-pairs floors — measured, not assumed — so families are cut
# to what validates and finer identity rides on the label + tooltip instead.
def family(name: str) -> str:
    if name.startswith("bootstrap_s0") or name in ("bootstrap", "bootstrap_precise"): return "bd"
    if name.startswith("bootstrap_s1"): return "b1"
    if name.startswith("bootstrap_s"):  return "b5"
    if re.match(r"^(res_|mask_|kv_|enc_cache_evict|block_sync)", name): return "res"
    if re.fullmatch(r"tok\d+|blk\d+|block|transformer_block|root|decode|argmax|cutmax", name): return "par"
    return "math"

TRACE_PATHS = re.compile(r"^\[proftrace_paths\] n=(\d+)")
TRACE_HDR  = re.compile(r"^\[proftrace\] n=(\d+)")

def parse_trace(path: Path):
    """-> [(phase_index, [(scope_path, depth, t0_us, t1_us), ...]), ...], one entry per phase.

    Each phase has its own epoch (the profiler resets between the decode and argmax tables), so
    the phases are separate timelines and the caller lays them end to end.
    """
    lines = path.read_text(errors="replace").splitlines()
    out, i = [], 0
    while i < len(lines):
        m = TRACE_PATHS.match(lines[i])
        if not m: i += 1; continue
        n = int(m.group(1)); names = [None]*n; i += 1
        for _ in range(n):
            k, _sp, nm = lines[i].strip().partition(" ")
            names[int(k)] = nm; i += 1
        h = TRACE_HDR.match(lines[i]) if i < len(lines) else None
        if not h: continue
        cnt = int(h.group(1)); i += 1
        recs = []
        for _ in range(cnt):
            a, d, t0, t1 = lines[i].split(); i += 1
            recs.append((names[int(a)], int(d), int(t0), int(t1)))
        out.append(recs)
    return out

def build_from_trace(phases):
    """True flame chart: x is elapsed time, y is call depth. No ordering heuristic needed —
    position comes from the timestamps, which is the whole point of recording them."""
    frames, offset, meta_phases = [], 0.0, {}
    fam_tot = collections.Counter()
    for recs in phases:
        if not recs: continue
        t_min = min(r[2] for r in recs); t_max = max(r[3] for r in recs)
        # the phase name is the first component of any path's root frame
        pname = recs[0][0].split(".")[0] if recs else "phase"
        span = (t_max - t_min)/1000.0
        # child intervals nest inside parents, so self = own span minus direct children's spans
        by_depth = collections.defaultdict(list)
        for nm, d, t0, t1 in recs: by_depth[d].append((t0, t1, nm))
        for nm, d, t0, t1 in recs:
            child = sum(min(c1, t1) - max(c0, t0)
                        for c0, c1, _ in by_depth.get(d+1, [])
                        if c0 < t1 and c1 > t0)
            leaf = nm.rsplit(".", 1)[-1]
            self_ms = max(0.0, (t1 - t0 - child)/1000.0)
            frames.append([leaf, d, round(offset + (t0 - t_min)/1000.0, 3),
                           round((t1 - t0)/1000.0, 3), round(self_ms, 3), family(leaf)])
            fam_tot[family(leaf)] += self_ms
        meta_phases[pname] = round(span, 1)
        offset += span
    return frames, {"total": round(offset, 1), "phases": meta_phases,
                    "fam": {k: round(v, 1) for k, v in fam_tot.items()},
                    "nframes": len(frames), "chart": True}

def parse_log(path: Path):
    """-> {phase: [(path, calls, self_ms)]}. Phases keyed by the table header."""
    phase, rows = None, collections.defaultdict(list)
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("[proftok]"): phase = "decode"; continue
        if line.startswith("[profarg]"): phase = "argmax"; continue
        if line.startswith("[profpre]"): phase = "prefill"; continue
        if phase is None: continue
        m = ROW.match(line)
        if m: rows[phase].append((m[1], int(m[2]), float(m[3])))
        elif line.strip() and not line.startswith("  ") and rows[phase]: phase = None
    return rows

def build(rows):
    root = {"n": "root", "s": 0.0, "c": {}}
    for phase, rs in rows.items():
        for p, _calls, self_ms in rs:
            node = root
            for fr in (phase + "." + p).split("."):
                node = node["c"].setdefault(fr, {"n": fr, "s": 0.0, "c": {}})
            node["s"] += self_ms
    def total(nd):
        nd["t"] = nd["s"] + sum(total(k) for k in nd["c"].values()); return nd["t"]
    total(root)
    frames = []
    def flatten(nd, depth, x):
        if nd["n"] != "root":
            frames.append([nd["n"], depth, round(x, 4), round(nd["t"], 4), round(nd["s"], 4),
                           family(nd["n"])])
        cx = x
        for k in sorted(nd["c"].values(), key=lambda z: (SEQ.get(z["n"], 10_000), natkey(z["n"]))):
            flatten(k, depth + 1 if nd["n"] != "root" else 0, cx); cx += k["t"]
    flatten(root, -1, 0.0)
    fam = collections.Counter()
    def walk(nd):
        if nd["n"] != "root": fam[family(nd["n"])] += nd["s"]
        for k in nd["c"].values(): walk(k)
    walk(root)
    return frames, {"total": round(root["t"], 1),
                    "phases": {k: round(v["t"], 1) for k, v in root["c"].items()},
                    "fam": {k: round(v, 1) for k, v in fam.items()},
                    "nframes": len(frames)}

def provenance(stem: Path):
    """Pull the run's identity out of macro_run.sh's sidecars so the page is self-describing."""
    p = {}
    meta = stem.with_suffix(".meta")
    if meta.exists():
        for line in meta.read_text(errors="replace").splitlines():
            m = re.match(r"^(\w+)\s+(.*)$", line.strip())
            if m and m[1] in ("plan_md5", "core_so_md5", "core_so_link", "repo_sha", "marker",
                              "wall_total_s", "date_utc"):
                p[m[1]] = m[2].strip()
            if m and m[1] == "plan_dir": p["plan"] = Path(m[2].strip()).name
        env = re.findall(r"^\s{2}(\w+)=(.*)$", meta.read_text(errors="replace"), re.M)
        p["env"] = {k: v for k, v in env}
    gate = []
    for suf in (".out", ".err"):
        f = stem.with_suffix(suf)
        if not f.exists(): continue
        for line in f.read_text(errors="replace").splitlines():
            if re.match(r"^\[\w+\] (top1|pos=)", line): gate.append(line.strip())
    p["gate"] = gate
    return p

TEMPLATE = Path(__file__).with_name("flamechart_template.html")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="macro_run.sh .out log with a [proftok] table")
    ap.add_argument("--out", help="output HTML (default: alongside the log, .flamechart.html)")
    ap.add_argument("--title", help="page title (default: derived from the log name)")
    a = ap.parse_args()

    log = Path(a.log)
    if not log.exists(): sys.exit(f"no such log: {log}")
    trace = parse_trace(log)
    if trace:
        frames, meta = build_from_trace(trace)
    else:
        rows = parse_log(log)
        if not rows:
            sys.exit(f"{log} has no [proftok]/[profarg] table — was FHE_PROFILE=wall set?\n"
                     f"  try: bash scripts/flamechart.sh <TAG>")
        print("  note: no [proftrace] in this log — falling back to the aggregate tree, which is\n"
              "        a flameGRAPH (x carries no time meaning). Re-run with FHE_PROFILE_TRACE=1.",
              file=sys.stderr)
        frames, meta = build(rows)
        meta["chart"] = False
    stem = log.with_suffix("")
    prov = provenance(stem)
    title = a.title or f"{stem.name} — CKKS decode flame chart"
    out = Path(a.out) if a.out else stem.with_suffix(".flamechart.html")

    # ── PRE-RENDER everything server-side ────────────────────────────────────────────────
    # The page used to paint every number from JS, so one scripting failure produced a page
    # with headings, a legend and a chart area all empty — "no data" with no clue why. The
    # markup now ships complete and JS only adds zoom/search/tooltip on top of it.
    FAMN = {"bd": ("Bootstrap — dense", "var(--bd)", 0),
            "b5": ("Bootstrap — sparse s=512", "var(--b5)", 0),
            "b1": ("Bootstrap — sparse s=1", "var(--b1)", 1),
            "math": ("Model math", "var(--math)", 0),
            "res": ("Residency / plumbing", "var(--res)", 0),
            "par": ("Parent / unattributed", "var(--par)", 1)}
    T = meta["total"] or 1.0
    fm = meta["fam"]; boot = fm.get("bd",0)+fm.get("b5",0)+fm.get("b1",0)
    sparse = fm.get("b5",0)+fm.get("b1",0)
    def pct(v): return f"{100*v/T:.1f}%"
    tiles = [("Profiled total", f"{meta['total']:.0f} ms",
              " + ".join(f"{k} {v:.0f}" for k, v in meta["phases"].items())),
             ("Bootstrap", pct(boot), f"{boot:.0f} ms across all routes"),
             ("…already sparse", f"{100*sparse/boot:.0f}%" if boot else "—",
              "of bootstrap time is s=1 or s=512"),
             ("Model math", pct(fm.get("math",0)), "attention, LN, softmax, GELU, linear"),
             ("Residency", pct(fm.get("res",0)), "weight H2D, KV offload, install"),
             ("Frames", str(meta["nframes"]), "exact scopes, not samples")]
    tiles_html = "".join(
        f'<div class="tile"><span class="k">{html.escape(k)}</span>'
        f'<span class="v">{html.escape(v)}</span><span class="d">{html.escape(d)}</span></div>'
        for k, v, d in tiles)
    legend_html = "".join(
        f'<i><span class="swatch" style="background:{c}"></span>{html.escape(n)}</i>'
        for n, c, _lo in FAMN.values())
    ROWH = 18
    rects = []
    for n, d, x, t, sf, f in frames:
        w = 100.0*t/T
        if w < 0.02: continue           # under ~0.2px at any sane width
        lo = " lo" if FAMN[f][2] else ""
        lbl = html.escape(n) if w > 1.2 else ""
        rects.append(f'<div class="fr{lo}" style="left:{100.0*x/T:.4f}%;width:{w:.4f}%;'
                     f'top:{d*ROWH+1}px;background:{FAMN[f][1]}" data-i="{len(rects)}">{lbl}</div>')
    frames_html = "".join(rects)
    depth = max((f[1] for f in frames), default=0)
    rows = sorted(frames, key=lambda f: -f[4])[:40]
    table_html = "".join(
        f'<tr><td class="mono">{html.escape(r[0])}</td><td><span class="swatch" '
        f'style="background:{FAMN[r[5]][1]}"></span> {html.escape(FAMN[r[5]][0])}</td>'
        f'<td class="n">{r[4]:.2f}</td><td class="n">{100*r[4]/T:.2f}%</td>'
        f'<td class="n">{r[3]:.2f}</td></tr>' for r in rows)
    E = prov.get("env", {})
    cfgdir = [p for p in (E.get("CONFIGS_PATH","").split("/")) if p][-2:-1]
    prov_html = "".join(f"<span>{html.escape(x)}</span>" for x in [
        f"plan {prov['plan']}" if prov.get("plan") else "",
        f"plan md5 {prov.get('plan_md5','')[:8]}" if prov.get("plan_md5") else "",
        f"so {prov.get('core_so_md5','')[:8]}" if prov.get("core_so_md5") else "",
        prov.get("core_so_link",""), f"config {cfgdir[0]}" if cfgdir else "",
        f"MULTI_T={E['MULTI_T']}" if E.get("MULTI_T") else "",
        f"token {E['FHE_PROFILE_TOKEN']}" if E.get("FHE_PROFILE_TOKEN") else "",
        prov.get("marker","")] if x)
    eyebrow = " · ".join(x for x in ["perseus", E.get("CHAIN",""), E.get("TASK",""),
                                     prov.get("date_utc","")[:10]] if x)

    page = TEMPLATE.read_text()
    for k, v in (("__PAYLOAD__", json.dumps({"frames": frames, "meta": meta, "prov": prov,
                                             "title": title}, separators=(",", ":"))),
                 ("__TITLE__", html.escape(title)), ("__EYEBROW__", html.escape(eyebrow)),
                 ("__H1__", html.escape(title)), ("__PROV__", prov_html),
                 ("__SUB__", html.escape("Every frame is an exact profiler scope, not a sample. "
                                         + " + ".join(meta["phases"]) + ".")),
                 ("__TILES__", tiles_html), ("__LEGEND__", legend_html),
                 ("__FRAMES__", frames_html), ("__TABLE__", table_html),
                 ("__HEIGHT__", str((depth+1)*ROWH + 2)),
                 ("__GATE__", html.escape("\n".join(prov.get("gate", []))))):
        page = page.replace(k, v)
    out.write_text(page)
    print(f"{out}  ({len(page)/1024:.0f} KB, {meta['nframes']} frames, "
          f"{meta['total']:.0f} ms profiled)")
    for k, v in sorted(meta["fam"].items(), key=lambda kv: -kv[1]):
        print(f"   {k:5} {v:9.1f} ms  {100*v/meta['total']:5.1f}%")

if __name__ == "__main__":
    main()
