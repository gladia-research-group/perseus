#!/usr/bin/env python3
"""map_plan_decisions.py OLD_GRAPHS OLD_PLAN NEW_GRAPHS NEW_PLAN: carry a Decode-8 plan's decisions (placements, fired hints, realize
anchors, routes, CFs, drops) onto another capture of the same blocks whose steps differ only inside (each step's op
sequence aligned by difflib on the op signatures, the capture's reactive refreshes left out). Unmapped realize anchors
are dropped (reported); an unmapped decision stops. Blocks 0-11 (and 0_feedback when the plan has it); stages 12-13 copied. Then
RD_FREE_CF=1 speedrun/make_n32a4_decode_rd.sh on NEW_PLAN keeps those decisions and searches the raise drops (and CFs)
on the new capture, e.g. the GELU / exp mask folds' capture with _lh's decisions:
    python scripts/utils/map_plan_decisions.py graphs/gpt2_decode_python_n32_kc16r_lhm \
        bootstrap_placements/gpt2_decode_python_n32a4_kc16r_lh graphs/gpt2_decode_python_n32_kc16r_mfm \
        bootstrap_placements/gpt2_decode_python_n32a4_kc16r_mfk"""
import json, os, shutil, sys
OG, OP, NG, NP = sys.argv[1:5]

def _strip(v):
    return v[: -len("_auto_bootstrapped")] if isinstance(v, str) and v.endswith("_auto_bootstrapped") else v


def nodes(gdir, b):
    """The block's nodes less the capture's reactive refreshes (the planner erases them), inputs read through them."""
    out = []
    for n in json.load(open(f"{gdir}/block_{b}/graph.json"))["nodes"]:
        if n["op_type"] == "auto_bootstrap":
            continue
        out.append(dict(n, inputs=[_strip(v) for v in n["inputs"]]))
    return out

def steps(ns):
    out, cur = [], None
    for n in ns:
        s = n.get("step", "?")
        if cur is None or cur[0] != s:
            cur = (s, []); out.append(cur)
        cur[1].append(n)
    return out

def sig(n):
    return (n["op_type"], len(n["inputs"]))

_NEW_IDX = {}


def vmap(on, nn):
    """Per step a difflib alignment of the op signatures; a matched pair is kept only when every input it reads that
    is itself mapped maps onto the new node's input in the same position (else the match is a coincidence)."""
    import difflib
    _NEW_IDX.clear()
    _NEW_IDX.update({n["output"]: n for n in nn})
    so, sn = steps(on), steps(nn)
    if [s for s, _ in so] != [s for s, _ in sn]:
        raise SystemExit("step sequences differ")
    m, changed = {}, []
    for (s, a), (_, b) in zip(so, sn):
        sm = difflib.SequenceMatcher(None, [sig(x) for x in a], [sig(x) for x in b], autojunk=False)
        hit = 0
        for i, j, k in sm.get_matching_blocks():
            for t in range(k):
                x, y = a[i + t], b[j + t]
                ok = all(m.get(u, w) == w for u, w in zip(x["inputs"], y["inputs"]) if u in m or u.startswith("v_"))
                if ok:
                    m[x["output"]] = y["output"]; hit += 1
        if s.endswith("softmax") or s.endswith(".attn"):
            # exp_mask_fold: the active mask's product is gone; its value is its input's (already masked) image
            for x in a:
                ptn = str(((x.get("args") or {}).get("pt") or {}).get("name", ""))
                if x["output"] not in m and x["op_type"] == "mult" and len(x["inputs"]) == 2 \
                        and x["inputs"][1].startswith("pt_") and x["inputs"][0] in m \
                        and (s.endswith("softmax") or "active" in ptn):
                    y = m[x["inputs"][0]]
                    if not any(m.get(z["output"]) == y for z in a if z is not x and z["output"] != x["inputs"][0]):
                        m[x["output"]] = y
                        hit += 1
        if s.endswith("softmax"):
            # gs_num: a reciprocal round's product y = z R is gone (the recursion returns z R): its value is the
            # recursion's result, the image of R (an old ct x ct product whose one operand is mapped to a new node
            # that no other old node maps to and whose other operand is a ciphertext)
            for x in a:
                if x["output"] in m or x["op_type"] != "mult" or len(x["inputs"]) != 2:
                    continue
                if not all(u.startswith("v_") for u in x["inputs"]):
                    continue
                for u in x["inputs"][::-1]:
                    y = m.get(u)
                    if y is None:
                        continue
                    if not any(m.get(z["output"]) == y for z in a if z is not x and z["output"] != u):
                        m[x["output"]] = y
                        hit += 1
                        break
        if hit < max(len(a), len(b)):
            changed.append((s, len(a), len(b), hit))
    # a matched consumer whose producer is unmatched on both sides (the same value computed another way, e.g. a
    # Chebyshev sum that now carries the folded mask): the old producer's name onto the new one
    byname = {x["output"]: x for x in on}
    grow = True
    while grow:
        grow = False
        used = set(m.values())
        for x in on:
            y = m.get(x["output"])
            if y is None:
                continue
            yn = next((n for n in nn if n["output"] == y), None) if y not in _NEW_IDX else _NEW_IDX[y]
            for u, w in zip(x["inputs"], yn["inputs"]):
                if u in byname and u not in m and w.startswith("v_") and w not in used:
                    m[u] = w
                    used.add(w)
                    grow = True
    return m, changed


def tr(m, v, what, b, strict=True):
    if v in m:
        return m[v]
    if strict:
        raise SystemExit(f"block {b}: {what} {v} is in a changed region")
    return None

os.makedirs(NP, exist_ok=True)
BLOCKS = [str(i) for i in range(12)] + (["0_feedback"] if os.path.exists(f"{OP}/block_0_feedback_placement.json") else [])
for b in BLOCKS:
    gb = b.split("_")[0]
    m, changed = vmap(nodes(OG, gb), nodes(NG, gb))
    d = json.load(open(f"{OP}/block_{b}_placement.json"))
    op_of = {n["output"]: n["op_type"] for n in nodes(NG, gb)}
    d["placements"] = [dict(p, target_var=tr(m, p["target_var"], "placement", b)) for p in d["placements"]]
    for p in d["placements"]:
        p["node_op"] = op_of[p["target_var"]]
    d["hint_fire"] = [tr(m, v, "fired hint", b) for v in d.get("hint_fire") or []]
    for k in ("sparse_slots", "correction_factor", "raise_drop"):
        if d.get(k) is not None:
            d[k] = {tr(m, v, k, b): x for v, x in d[k].items()}
    if d.get("real_route") is not None:
        d["real_route"] = [tr(m, v, "real_route", b) for v in d["real_route"]]
    ra = d.get("rescale_after") or []
    kept = list(dict.fromkeys(m[v] for v in ra if v in m))
    lost = [v for v in ra if v not in m]
    d["rescale_after"] = kept
    s = d["summary"]
    s["exit_var"] = tr(m, s["exit_var"], "exit", b)
    s["final_named_levels"] = {m[v]: x for v, x in s.get("final_named_levels", {}).items() if v in m}
    s["final_named_degs"] = {m[v]: x for v, x in s.get("final_named_degs", {}).items() if v in m}
    json.dump(d, open(f"{NP}/block_{b}_placement.json", "w"), indent=1)
    print(f"block {b}: {len(d['placements'])} placements, {len(d['hint_fire'])} hints mapped; anchors {len(kept)} kept, "
          f"{len(lost)} dropped; changed steps {[(c[0], c[1], c[2]) for c in changed]}")
for k in (12, 13):
    shutil.copy(f"{OP}/block_{k}_placement.json", f"{NP}/block_{k}_placement.json")
