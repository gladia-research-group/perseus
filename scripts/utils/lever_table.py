"""The lever table of the paper finals, read from the plans: each final's PLAN_CMD.txt recipe= / runtime= / configs=
lines and the flags of the command that runs it, so the table and the lever matrix follow the shipped plans.

    python3 scripts/utils/lever_table.py [--spec FINALS.json] [--format md|tsv]

FINALS.json: [{"name": ..., "kind": "prefill"|"decode", "plans": ["bootstrap_placements/X", ...], "flags": "..."}]
(several plans: a window-plan set, every plan must agree or the cell lists each value). Without --spec, the shipped
defaults with the flags of scripts/repro/*.sh.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FINALS = [
    {"name": "BERT", "kind": "prefill", "plans": ["bootstrap_placements/bert_sst2_128"], "flags": "--text 128 --graphs"},
    {"name": "ViT", "kind": "prefill", "plans": ["bootstrap_placements/vit_b16_224"], "flags": "--stc-cache 32 --graphs"},
    {"name": "GPT-2 prefill", "kind": "prefill", "plans": ["bootstrap_placements/gpt2_prefill128"],
     "flags": "--tokens 128 --tail --graphs"},
    {"name": "Llama prefill", "kind": "prefill", "plans": ["bootstrap_placements/llama31_8b_prefill128"],
     "flags": "--tokens 128 --graphs --stc-cache 4"},
    {"name": "GPT-2 decode", "kind": "decode",
     "plans": [f"bootstrap_placements/gpt2_decode_ctx512_w{b}" for b in (128, 256, 384, 512)], "flags": "--argmax"},
    {"name": "Llama decode", "kind": "decode",
     "plans": [f"bootstrap_placements/llama31_8b_decode_ctx512_w{b}" for b in (128, 256, 384, 512)], "flags": ""},
]


def plan_cmd(plan: str) -> dict:
    """recipe / runtime as {key: value} (the last value of a key wins, as the environment applies them), configs."""
    out = {"recipe": {}, "runtime": {}, "configs": None, "graph": "", "note": ""}
    for line in (ROOT / plan / "PLAN_CMD.txt").read_text().splitlines():
        key, _, body = line.partition("=")
        if key in ("recipe", "runtime"):
            for kv in shlex.split(body):
                k, _, v = kv.partition("=")
                out[key][k] = v
        elif key in ("configs", "graph", "note"):
            out[key] = body.strip()
    return out


def activation_series(configs: str | None) -> str:
    """'n/m blocks' of the activation entries (softgelu / silu) that are one Chebyshev series."""
    if not configs:
        return "?"
    p = ROOT / "python/perseus" / configs
    if not p.exists():
        p = ROOT / configs
    if not p.exists():
        return "?"
    j = json.loads(p.read_text())
    sec = j.get("softgelu") or j.get("silu") or {}
    n = sum(1 for v in sec.values() if isinstance(v, dict) and v.get("method") == "chebyshev")
    return f"{n}/{len(sec)}" if sec else "-"


def levers(f: dict) -> dict[str, str]:
    """One column: lever -> cell text, from the plan lines and the run flags."""
    cmds = [plan_cmd(p) for p in f["plans"]]

    def get(sec, k):
        vals = {c[sec].get(k) for c in cmds}
        vals.discard(None)
        return vals.pop() if len(vals) == 1 else (" / ".join(sorted(vals)) if vals else None)

    rt = lambda k: get("runtime", k)
    rc = lambda k: get("recipe", k)
    flags = shlex.split(f["flags"])
    on = lambda b, why="": ("on" + (f" ({why})" if why else "")) if b else "off"
    pre = f["kind"] == "prefill"
    # the prefill drivers default the staged-only mode to PF_PCMM_STAGED_ONLY or "1" (PCMM/CCMM finishes only)
    staged_only = rt("FIDESLIB_BTS_STC_FIRST_STAGED_ONLY") or rt("PF_PCMM_STAGED_ONLY") or ("1" if pre else "0")
    if rt("FIDESLIB_BTS_STC_FIRST") not in (None, "0"):
        stcf = on(True, "PCMM/CCMM finishes" if staged_only == "1" else "all dense refreshes")
    elif pre and (rt("PF_LINEAR") == "pcmm" or rt("PF_ATTN") == "ccmm"):
        stcf = on(True, "PCMM/CCMM finishes (driver default)")
    else:
        stcf = "off"
    att = "CCMM" if rt("PF_ATTN") == "ccmm" else ("rotation + BSGS" if rt("PF_ATTN_BSGS") == "1" else
                                                ("rotation" if pre else "-"))
    # a window-plan set: --plan-windows, or the default windows (several plans)
    windows = next((flags[i + 1] for i, x in enumerate(flags[:-1]) if x == "--plan-windows"),
                   f"{len(cmds)} windows" if len(cmds) > 1 else None)
    prefix = next((flags[i + 1] for i, x in enumerate(flags[:-1]) if x == "--plain-prefix"), None)
    merged = bool(rc("PLAN_MAG_SAFETY_STEPS")) or any("merged" in c["note"] for c in cmds)
    return {
        # pairs form only under the siblings schedule: the prefill default, a decode's --schedule siblings
        "Refresh pairs (PC_BTS_PAIR)": (on(True, "siblings") if rt("PC_BTS_PAIR") == "1" and (pre or "siblings" in flags)
                                        else "off" + (" (program schedule)" if rt("PC_BTS_PAIR") == "1" else "")),
        "StC-first refreshes": stcf,
        "Raise-drop search": on(rc("PLAN_RAISE_SEARCH") == "1", f"routes {rc('PLAN_RAISE_ROUTES')}"),
        "Raise-drop set": rt("FIDESLIB_BTS_RAISE_DROPS") or "-",
        "SparseB landings": rc("PLAN_SPARSE_BTS_OUT") or "-",
        "SparseB EvalMod K": rt("FIDESLIB_BTS_SPARSE_B_K") or "default",
        "CtS alternative budget": rt("FIDESLIB_CTS_ALT_BUDGET") or "off",
        "Dense refresh cap past the StC-first entry": rc("PLAN_REFRESH_ENV_CAP_ROUTES") or "off",
        "Re-route past the StC-first entry": on(rc("PLAN_STC_FIRST_REROUTE") == "1"),
        "Strict envelope": on(rc("PLAN_ENVELOPE_STRICT") == "1"),
        "StC-first error model (slim cubic)": (on(True, "per site: staged finishes"
                                                  if rc("PLAN_STC_FIRST_STAGED_ONLY") == "1" else "")
                                               if rc("PLAN_SLIM_CUBIC") == "1" else "off"),
        "PCMM finish raise drops": (rt("PF_PCMM_RAISE") or "off") if pre else "-",
        "Pack CF": on(rc("PLAN_PACK_CF") == "1"),
        "Fallback CF": on(rc("PLAN_FALLBACK_CF") == "1"),
        "Attention": att,
        "One softmax per chunk pair (HPACK)": on(rt("PF_SM_HPACK") == "1") if pre else "-",
        "K/V of the last head group packed": on(rt("PF_QKV_PACK") == "1") if pre else "-",
        "Last block on one position": on(rt("PF_LAST_QUERY") == "1" or "--tail" in flags) if pre else "-",
        "Softmax: Chebyshev-scaled Goldschmidt": on(rt("SM_CHEB_GS") == "1"),
        "Softmax head sum cleanse": on(rt("SM_HRS_CLEANSE") == "1"),
        "Activation: one Chebyshev series": (on(True, "SiLU, driver") if rt("PF_SILU_SINGLE") == "1"
                                             else activation_series(cmds[0]["configs"])),
        "Magnitudes merged over several inputs": on(merged),
        "CUDA graph replay": on("--graphs" in flags),
        "Plan per position window": on(bool(windows), windows or ""),
        "Plaintext prefix": on(bool(prefix), f"{prefix} positions" if prefix else ""),
        "Placer": rc("PLAN_PLACER") or "min_cut",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--format", choices=("md", "tsv"), default="md")
    a = ap.parse_args()
    finals = json.loads(Path(a.spec).read_text()) if a.spec else FINALS
    cols = []
    for f in finals:
        try:
            cols.append((f["name"], levers(f)))
        except FileNotFoundError as e:
            print(f"{f['name']}: {e}", file=sys.stderr)
    rows = list(cols[0][1]) if cols else []
    sep = " | " if a.format == "md" else "\t"
    head = ["Lever"] + [n for n, _ in cols]
    print(("| " if a.format == "md" else "") + sep.join(head) + (" |" if a.format == "md" else ""))
    if a.format == "md":
        print("|" + "---|" * len(head))
    for r in rows:
        cells = [r] + [c.get(r, "") for _, c in cols]
        print(("| " if a.format == "md" else "") + sep.join(cells) + (" |" if a.format == "md" else ""))


if __name__ == "__main__":
    main()
