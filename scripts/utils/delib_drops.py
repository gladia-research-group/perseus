"""delib_drops.py LOG GRAPH: the code's own refreshes that can land lower, read from a planner log run with
PLAN_RAISE_DELIB=1 (the deliberate-drop probe, plan/placer/planner.py), as the runtime switches that land them there:
PF_PCMM_RAISE_OUT for the PCMM finishes (impl/pcmm.py: "linear:out:drop[:lo-hi]", the drop counted from the chain top,
the probe's slack added to the drop the capture already has) and KV_PUSH_RAISE for the K/V push refreshes
(impl/attention.py: "block:drop"). Other deliberate refreshes the probe names are listed, not switched; with
CCMM attention (PF_ATTN=ccmm in the environment) the qkv finishes, CCMM operands, keep their drops.
    PLAN_RAISE_DELIB=1 bash scripts/utils/rd_pass.sh PLAN > LOG 2>&1
    python scripts/utils/delib_drops.py LOG graphs/<plan's graph>      -> PF_PCMM_RAISE_OUT=... KV_PUSH_RAISE=...
"""
import json
import os
import re
import sys


def probe(log: str) -> dict[int, dict[str, int]]:
    """{block: {value: slack}} from the log's `[block_<b>]` headers and probe lines."""
    out: dict[int, dict[str, int]] = {}
    blk = None
    for line in open(log):
        m = re.search(r"\[block_(\d+)\]", line)
        if m:
            blk = int(m.group(1))
            continue
        if "delib drop probe" in line and blk is not None:
            out[blk] = {v: int(k) for v, k in re.findall(r"(\S+):(\d+)(?=\s|$)", line.split("):", 1)[-1])}
    return out


def main(log: str, graph: str) -> None:
    """The probe's slack is counted from the drops the probed run had: PF_PCMM_RAISE_OUT / KV_PUSH_RAISE in this
    process's environment (a pass on a pass, while the probe still finds slack), else the capture's own."""
    import types
    from perseus.impl import pcmm
    found = probe(log)
    ccmm = os.environ.get("PF_ATTN") == "ccmm"
    kv_now = dict(tuple(int(x) for x in e.split(":")) for e in os.environ.get("KV_PUSH_RAISE", "").split(",") if e)
    pcmm_d: dict[tuple[str, int, int], list[int]] = {}
    kv: dict[int, int] = dict(kv_now)
    other: list[str] = []
    for name in sorted(os.listdir(graph)):
        if not (name.startswith("block_") and name[6:].isdigit()):
            continue
        b = int(name[6:])
        g = json.load(open(os.path.join(graph, name, "graph.json")))
        for n in g["nodes"]:
            if n["op_type"] != "deliberate_bootstrap":
                continue
            v, k = n.get("output"), found.get(b, {}).get(n.get("output"), 0)
            ext = (n.get("args") or {}).get("external") or {}
            if ext.get("op") == "pcmm" and ccmm and ext["linear"].split(".")[-1] == "qkv":
                # with CCMM attention (PF_ATTN=ccmm) the qkv finishes are CCMM operands, whose entry level the CCMM
                # service fixes (ccmm::operand_level) and the probe does not model: their drops stay the plan's
                continue
            if ext.get("op") == "pcmm":
                base = int(ext.get("raise_drop", 0))
                cur = pcmm._drop_of(types.SimpleNamespace(name=ext["linear"], n_out=ext["n_out"]), ext["out"]) or base
                if cur + k != base:
                    pcmm_d.setdefault((ext["linear"].split(".")[-1], int(ext["out"]), cur + k), []).append(b)
            elif n["step"].endswith("kv_push"):
                if k:
                    kv[b] = kv.get(b, 0) + k
            elif k:
                other.append(f"{b}:{v}:{k}:{n.get('step')}")
    spec = []
    for (lin, m, d), blocks in sorted(pcmm_d.items()):
        blocks = sorted(set(blocks))
        lo = prev = blocks[0]
        for b in blocks[1:] + [None]:
            if b is not None and b == prev + 1:
                prev = b
                continue
            spec.append(f"{lin}:{m}:{d}:{lo}-{prev}")
            if b is not None:
                lo = prev = b
    if spec:
        print("PF_PCMM_RAISE_OUT=" + ",".join(spec))
    if kv:
        print("KV_PUSH_RAISE=" + ",".join(f"{b}:{d}" for b, d in sorted(kv.items())))
    if other:
        print("# not switched: " + " ".join(other), file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:3])
