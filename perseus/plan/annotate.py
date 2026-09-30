#!/usr/bin/env python3
import json
import logging
import re
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def annotate(path: Path) -> int:
    g = json.loads(path.read_text())
    nodes = g["nodes"]
    produced = {n.get("output") for n in nodes}
    rename = {}
    for n in nodes:
        ins = n.get("inputs") or []
        lvls = n.get("input_levels") or []
        for i, v in enumerate(ins):
            m = re.fullmatch(r"(anon_\d+)(?:-lvl=\d+)?", v)
            if v in produced or v in rename or not m:
                continue
            if i < len(lvls) and lvls[i] is not None:
                rename[v] = f"cf.cache.{m.group(1)}-lvl={int(lvls[i])}"
    if not rename:
        return 0
    for n in nodes:
        ins = n.get("inputs")
        if ins:
            n["inputs"] = [rename.get(v, v) for v in ins]
    path.write_text(json.dumps(g))
    return len(rename)

def annotate_graph_dir(root: Path) -> int:
    total = 0
    for f in sorted(root.glob("block_*/graph.json")):
        k = annotate(f)
        total += k
        log.info(f"{f.parent.name}: annotated {k} anon inputs")
    log.info(f"total: {total}")
    return total

def main():
    for root in sys.argv[1:]:
        annotate_graph_dir(Path(root))

if __name__ == "__main__":
    main()
