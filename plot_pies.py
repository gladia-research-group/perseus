import re
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# ── Parse ──
rows = []
with open("/home/mithrillo/Downloads/aaaaaaa/times.csv") as f:
    f.readline()
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s{2,}", line)
        if len(parts) >= 3:
            step = parts[0].strip()
            try:
                self_ms = float(parts[2].strip())
            except ValueError:
                continue
            rows.append({"step": step, "self_ms": self_ms})

# ── Bootstrap GROUPING (fold goldschmidt_nice.iter_N into goldschmidt_nice) ──
def bootstrap_group(step):
    segs = step.split(".")
    fn = segs[-2]
    if fn.startswith("iter_"):
        for s in reversed(segs[:-2]):
            if s == "goldschmidt_nice":
                return "goldschmidt_nice"
            if s.startswith("iter_"):
                continue
            if s in ("gs_iters", "inv_sqrt_init", "remez_init", "refine_iter"):
                continue
            return s + "." + fn
    return fn

# ── Pie 1: bootstrap calls from tok0 only ──
boot_counts = Counter()
for r in rows:
    step = r["step"]
    if step.startswith("tok0.") and step.endswith(".bootstrap"):
        boot_counts[bootstrap_group(step)] += 1

# ── Pie 2: per-component latency, averaged across tokens ──
def component(step):
    s = step
    if "down_linear" in s:
        return "linear"
    if "qkv" in s:
        return "qkv"
    if "softmax" in s or "lane_mult" in s or "lane_mask" in s:
        return "attention"
    if "layer_norm" in s or "ln_f" in s or "ln_1" in s or "ln_2" in s:
        return "layer_norm"
    if "gelu" in s:
        return "gelu"
    if "kv_offload" in s or "kv_pack" in s or "cache_kv" in s:
        return "kv_cache"
    return "other"

# Collect latency per token per component
tok_comp = defaultdict(lambda: defaultdict(float))
for r in rows:
    step = r["step"]
    if step.startswith("tok") and step[3].isdigit():
        tok = step[:4]  # e.g. "tok0", "tok1"
        comp = component(step)
        tok_comp[tok][comp] += r["self_ms"]

# Average across tokens 0-7
token_ids = [f"tok{i}" for i in range(8)]
comp_avg = defaultdict(float)
for comp in set().union(*tok_comp.values()):
    vals = [tok_comp[t][comp] for t in token_ids]
    comp_avg[comp] = sum(vals) / len(vals)

# Global ops (not per-token) – noted as annotation
lm_head_ms = sum(r["self_ms"] for r in rows if r["step"].startswith("lm_head"))

# ── Merge tiny slices ──
def merge_small(items, threshold_pct=3.0):
    total = sum(v for _, v in items)
    main = {}
    other = 0.0
    for k, v in sorted(items, key=lambda x: -x[1]):
        if v / total * 100 >= threshold_pct:
            main[k] = v
        else:
            other += v
    if other > 0:
        main["other"] = other
    return main, total

# ── Plot ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
fig.suptitle("Autoregressive Inference over 8 Tokens (FHE)", fontsize=18, fontweight="bold")

# Pie 1: tok0 bootstrap calls
boot_main, boot_total = merge_small(list(boot_counts.items()))
labels1 = list(boot_main.keys())
sizes1 = list(boot_main.values())
colors1 = [plt.cm.tab20c(i / max(len(labels1), 1)) for i in range(len(labels1))]
wedges1, _, autotexts1 = ax1.pie(
    sizes1, labels=None, autopct="%1.1f%%", startangle=90,
    colors=colors1, pctdistance=0.80
)
ax1.set_title("Bootstrap Call Distribution (tok0 only)", fontsize=14, fontweight="bold")
legend1 = [f"{k}  ({v})" for k, v in zip(labels1, sizes1)]
ax1.legend(wedges1, legend1, title="Function (count)", loc="center left",
           bbox_to_anchor=(1.05, 0, 0.5, 1), fontsize=9)

# Pie 2: average latency per component across tokens
lat_main, lat_total = merge_small(list(comp_avg.items()))
labels2 = list(lat_main.keys())
sizes2 = list(lat_main.values())
colors2 = [plt.cm.tab20b(i / max(len(labels2), 1)) for i in range(len(labels2))]
wedges2, _, autotexts2 = ax2.pie(
    sizes2, labels=None, autopct="%1.1f%%", startangle=90,
    colors=colors2, pctdistance=0.80
)
ax2.set_title("Avg Latency per Token by Component", fontsize=14, fontweight="bold")
legend2 = [f"{k}  ({v:.0f} ms, {v/lat_total*100:.1f}%)" for k, v in zip(labels2, sizes2)]
ax2.legend(wedges2, legend2, title="Component (latency)", loc="center left",
           bbox_to_anchor=(1.05, 0, 0.5, 1), fontsize=9)
ax2.annotate(f"Global: lm_head = {lm_head_ms:.0f} ms (not per-token)",
             xy=(0.5, -0.08), xycoords="axes fraction", ha="center", fontsize=11, fontstyle="italic")

plt.tight_layout()
plt.savefig("/home/mithrillo/Downloads/aaaaaaa/pie_charts.png", dpi=150, bbox_inches="tight")
print("Saved pie_charts.png")
