#!/bin/bash
# step0_check.sh — which GELU does each GPT-2/ViT/BERT approximation config carry?
# Run from anywhere:  bash <repo>/src/perseus/scripts/step0_check.sh
#
# The broken artifact is the UNIQUE config whose softgelu method is softsign_inv_sqrt
# (the retired Remez+Goldschmidt+Newton GELU). Every other config in either tree is
# thor_composite, so identify by METHOD, not by domain: the h.7 xmax value is shared by
# a dozen configs and cannot discriminate.
#
# Prints one line per config found, then an explicit VERDICT. Never exits your shell.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # <repo> from src/perseus/scripts
PY="$(command -v "$ROOT/.venv/bin/python" || command -v python3 || command -v python)"
if [ -z "$PY" ]; then
    echo "step0: no python found (try: source $ROOT/.venv/bin/activate)"
    return 2 2>/dev/null || exit 2
fi

"$PY" - "$ROOT" <<'EOF'
import json, os, sys
root = sys.argv[1]
SITE = "transformer.h.7.mlp.act"
roots = [("train ", os.path.join(root, "configs/model/approximation")),
         ("deploy", os.path.join(root, "src/perseus/configs/model/approximation"))]
names = ["gpt2_base", "gpt2_baseline", "gpt2_squeeze", "gpt2_squeeze_tg",
         "gpt2_squeeze_recount1e4", "gpt2_heat",
         "vit_base", "vit_squeeze", "vit_heat", "bert_squeeze", "bert_heat"]

rows, softsign, seen = [], [], 0
for tag, base in roots:
    for n in names:
        p = os.path.join(base, n, "configs.json")
        if not os.path.exists(p):
            continue
        seen += 1
        try:
            sg = json.load(open(p))["softgelu"]
            v = sg[SITE] if SITE in sg else list(sg.values())[0]
            meth, xmax = v["method"], v["xmax"]
            extra = " ".join(f"{k}={v[k]}" for k in ("gs_iters", "newton_iters") if k in v)
        except Exception as e:
            rows.append(f"  {tag} {n:<26} UNREADABLE ({type(e).__name__}: {e})")
            continue
        rows.append(f"  {tag} {n:<26} {meth:<20} xmax={xmax!r} {extra}")
        if meth == "softsign_inv_sqrt":
            softsign.append(f"{tag.strip()}/{n}  ({p})")

print("\n".join(rows) if rows else "  (no configs found)")
print()
if seen == 0:
    print("VERDICT: NO CONFIG READ — wrong root? expected under", root)
    sys.exit(2)
if len(softsign) == 1:
    print("VERDICT: as expected on behemoth/git — the broken config is")
    print("        ", softsign[0])
    print("         proceed with handoff section 3-4 (rebuild it on the composite GELU)")
elif not softsign:
    # Since the 2026-08-14 recount burst this is the EXPECTED state: deploy/gpt2_squeeze was
    # rebuilt from gpt2_squeeze_tg (thor_composite) and the whole tree is now thor_composite.
    # Distinguish "rebuilt" from "diverged" by the recount provenance stamp, so this script
    # stops telling people to halt on a correct tree.
    import json as _json, os as _os
    _sq = _os.path.join(root, "src/perseus/configs/model/approximation/gpt2_squeeze/configs.json")
    _prov = None
    try:
        _prov = _json.load(open(_sq)).get("_provenance")
    except Exception:
        pass
    if _prov and _prov.get("target") == 1e-4:
        print("VERDICT: OK — no softsign config, and deploy/gpt2_squeeze carries the 1e-4 recount")
        print("         provenance (source %s)." % _prov.get("source_config", "?"))
        print("         This is the EXPECTED post-rebuild state. The GELU defect is fixed; the")
        print("         squeeze arm is thor_composite like base and heat. Nothing to do here.")
    else:
        print("VERDICT: NO softsign_inv_sqrt config, and NO recount provenance on deploy/gpt2_squeeze.")
        print("         The arm may have been rebuilt by other means, or this tree diverged from")
        print("         git. STOP and report: check whether the captured graph")
        print("         .cache/graph_gpt2_squeeze/block_0/graph.json still contains")
        print("         gelu_softsign_inv_sqrt nodes (if it does, the config was edited without")
        print("         recapturing and the plan is stale).")
else:
    print("VERDICT: MULTIPLE softsign configs — unexpected, report all of them:")
    for s in softsign:
        print("        ", s)
EOF
rc=$?
echo "step0 rc=$rc"
