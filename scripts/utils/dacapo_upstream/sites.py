"""Run the released DaCapo placer (hecate-opt, corelab-src/dacapo at 4616402 with
dacapo_4616402.patch) on each block of a captured graph and map its bootstrap targets back
to the graph's variables.

The patch only adds a `--dacapo-plan` pipeline (hecate's own RemoveBootstrap,
BypassDetection, CandidateSelection, DaCapoPlanner, BootstrapPlacement and ProactiveRescaling
passes, without lowering) and prints the planner's cut points and bootstrap targets.

    HECATE_OPT=<hecate-opt> GRAPH_DIR=<graph> OUT_JSON=<sites.json> [ML=48] [JOBS=4] \\
        python scripts/utils/dacapo_upstream/sites.py

The CKKS config is hecate's default (config.json) with the bootstrap level bounds set to the
deployed chain: ckks_ml48.json (5 multiplies between refreshes, as the planner's frame at a
refresh landing of 36 under a top of 48) or ckks_ml50.json (6).
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from translate import translate                                       # noqa: E402

HECATE = os.environ["HECATE_OPT"]
GRAPHS = Path(os.environ.get("GRAPH_DIR", "graphs/gpt2_decode_python_n32"))
OUT_JSON = Path(os.environ["OUT_JSON"])
CFG = HERE / f"ckks_ml{os.environ.get('ML', '48')}.json"
JOBS = int(os.environ.get("JOBS", "4"))


def run_block(b, tmp):
    base = str(Path(tmp) / f"block_{b}")
    translate(str(GRAPHS / f"block_{b}" / "graph.json"), base, f"block_{b}")
    p = subprocess.run([HECATE, "--dacapo-plan", f"--ckks-config={CFG}", "--waterline=51",
                        "--output-val=10", base + ".mlir", "-o", "/dev/null"],
                       capture_output=True, text=True,
                       env=dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=""))
    m = re.search(r"DACAPO_BTP_TARGETS:(.*)", p.stdout)
    if p.returncode != 0 or m is None:
        return b, dict(ok=False, error=(p.stderr or p.stdout)[-400:])
    targets = [int(x) for x in m.group(1).split()]
    o2v = {o: v for o, v, t, s in json.load(open(base + ".opmap.json"))["opmap"]}
    unmapped = [o for o in targets if o not in o2v or o2v[o].endswith((".pre", ".neg"))]
    return b, dict(ok=not unmapped, sites=sorted({o2v[o] for o in targets if o in o2v}),
                   n_targets=len(targets), unmapped=unmapped,
                   latency=re.search(r"Estimated Latency: ([0-9.]+)", p.stdout).group(1))


def main():
    blocks = sorted(int(d.name.split("_")[1]) for d in GRAPHS.glob("block_*")
                    if (d / "graph.json").exists())
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(JOBS) as ex:
        res = dict(ex.map(lambda b: run_block(b, tmp), blocks))
    for b in blocks:
        r = res[b]
        print(f"[block_{b}] " + (f"targets={r['n_targets']} sites={len(r['sites'])}" if r["ok"]
                                 else f"FAILED {r.get('error') or r['unmapped']}"), flush=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({f"block_{b}": res[b] for b in blocks}, indent=1))
    return 0 if all(r["ok"] for r in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
