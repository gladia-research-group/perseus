"""Block 12 magnitudes for the ln_f-folded tail capture: every node whose (op, name, inputs) the record's
merged capture (positions 0-15 x 4 pool rows) shares takes that capture's statistics; the folded suffix (one op fewer)
takes, per output_max_* key, the larger of its own (position 0) and the record capture's node one later.
    python scripts/utils/merge_lnf_tail_mags.py NEW_GRAPH_DIR REF_GRAPH_DIR
(graphs/gpt2_decode_python_n32_kc16r_cmfl from graphs/gpt2_decode_python_n32_kc16rm)"""
import json
import sys

new, ref = sys.argv[1:]
pa, pb = f"{new}/block_12/graph.json", f"{ref}/block_12/graph.json"
with open(pa) as f, open(pb) as g:
    A, B = json.load(f), json.load(g)
a, b = A["nodes"], B["nodes"]
k = lambda x: (x["op_type"], x["output"], tuple(x["inputs"]))
i = 0
while i < min(len(a), len(b)) and k(a[i]) == k(b[i]):
    i += 1
first = i
STAT = [s for s in b[0] if s.startswith("output_") and s not in ("output_level",)]
for j, x in enumerate(a):
    if j < first:
        for s in STAT:
            if s in b[j]:
                x[s] = b[j][s]
    else:
        y = b[j + 1] if j + 1 < len(b) else None
        if y is not None and y["op_type"] == x["op_type"]:
            for s in ("output_max_abs", "output_max_coeff", "output_max_coeff_ac", "output_max_dev"):
                if s in y and s in x:
                    x[s] = max(x[s], y[s])
with open(pa, "w") as f:
    json.dump(A, f)
print(f"[merge_lnf_tail_mags] {first} prefix nodes from {ref}, {len(a) - first} suffix nodes max-merged; stats {STAT}")
