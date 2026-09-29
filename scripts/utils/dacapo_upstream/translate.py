"""Translate a captured block graph (graph.json) to hecate's earth-dialect MLIR.

- bootstrap nodes (auto / fold / deliberate) are erased and their consumers rewired to the
  bootstrap's input; clone / hint / level_hint are aliases (no earth op). hecate's own
  pipeline starts by removing bootstraps anyway (RemoveBootstrap).
- mult / square -> earth.mul (ct x ct, or ct x pt through an earth.constant)
- add -> earth.add; sub -> earth.negate + earth.add
- negate and mult_i (a level-free monomial multiply) -> earth.negate
- rotate -> earth.rotate(offset); conjugate -> earth.rotate(offset = 0)
- a non-multiply op the planner charges a forced rescale (its captured output level drops,
  ir.Node.cost > 0) is followed by an earth.mul with a constant, so hecate sees the same level
  structure as the planner; the variable maps to that multiply.
- every value is tensor<1x!earth.ci<0*0>>: hecate's passes assign the scales.

Writes <out>.mlir and <out>.opmap.json: opid -> [our var, op type, step], where opid is the
1-based index of the earth op in program order (CandidateAnalysis numbering).

    python scripts/utils/dacapo_upstream/translate.py <graph.json> <out base>
"""
import json
import re
import sys
from pathlib import Path

CT = "tensor<1x!earth.ci<0*0>>"
PL = "tensor<1x!earth.pl<0*0>>"

BOOT = {"auto_bootstrap", "fold_bootstrap", "deliberate_bootstrap"}
ALIAS = {"clone", "hint", "level_hint"}
MUL = {"mult", "mult_inplace"}
SQ = {"square", "square_inplace"}
ADD = {"add", "add_inplace"}
SUB = {"sub_ct", "sub_inplace_ct"}


def is_pt(name):
    return name.startswith(("pt_", "const(", "mask"))


def is_meta(name):
    return name.startswith(("rot(", "hint_lev(", "lvl_cap("))


def forced_vars(graph_path):
    """Vars whose producer carries a forced (non-multiply) level cost in the planner's IR."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from perseus.plan.placer.ir import Graph, MULT_FAMILY
    g = Graph.load(graph_path, level_unit=2)
    return {n.output for n in g.nodes
            if n.output and n.cost > 0 and n.op not in MULT_FAMILY}


def translate(graph_path, out_base, func_name="block"):
    forced = forced_vars(graph_path)
    nodes = json.load(open(graph_path))["nodes"]

    alias = {}

    def res(v):
        while v in alias:
            v = alias[v]
        return v

    produced = {n["output"] for n in nodes}
    for n in nodes:
        if n["op_type"] in BOOT or n["op_type"] in ALIAS:
            ct_ins = [v for v in n["inputs"] if not is_meta(v) and not is_pt(v)]
            assert ct_ins, (n["op_type"], n["inputs"])
            alias[n["output"]] = ct_ins[0]

    ext, seen = [], set()
    for n in nodes:
        for v in n["inputs"]:
            if v in produced or is_meta(v) or is_pt(v) or v in seen:
                continue
            seen.add(v)
            ext.append(v)

    ssa = {v: f"%arg{i}" for i, v in enumerate(ext)}
    lines, opmap, opid = [], [], [0]

    def emit(rhs, our_var, op_type, step):
        opid[0] += 1
        name = f"%v{opid[0]}"
        lines.append(f"    {name} = {rhs}")
        opmap.append((opid[0], our_var, op_type, step))
        return name

    def const_ssa(pt_name, step):
        return emit(f'"earth.constant"() <{{value = dense<5.000000e-01> : tensor<1xf64>, '
                    f'rms_var = 5.000000e-01 : f64}}> : () -> {PL}', pt_name, "constant", step)

    def ct(v, step):
        v = res(v)
        if is_pt(v):
            return const_ssa(v, step), PL
        return ssa[v], CT

    consumed = set()
    for n in nodes:
        if n["op_type"] in BOOT or n["op_type"] in ALIAS:
            continue
        consumed |= {res(v) for v in n["inputs"] if not is_meta(v)}

    for n in nodes:
        t, out, step = n["op_type"], n["output"], n.get("step", "")
        ins = [v for v in n["inputs"] if not is_meta(v)]
        if t in BOOT or t in ALIAS:
            continue
        if t in MUL or t in SQ or t in ADD:
            opn = "mul" if (t in MUL or t in SQ) else "add"
            a, ta = ct(ins[0], step)
            b, tb = (a, ta) if (t in SQ or len(ins) == 1) else ct(ins[1], step)
            if ta == PL:                      # keep the ciphertext operand first
                a, b, ta, tb = b, a, tb, ta
            ssa[out] = emit(f'"earth.{opn}"({a}, {b}) : ({ta}, {tb}) -> {CT}', out, t, step)
        elif t in SUB:
            a, ta = ct(ins[0], step)
            b, tb = ct(ins[1], step)
            nb = emit(f'"earth.negate"({b}) : ({tb}) -> {CT if tb == CT else PL}',
                      out + ".neg", t, step)
            ssa[out] = emit(f'"earth.add"({a}, {nb}) : ({ta}, {tb}) -> {CT}', out, t, step)
        elif t in ("negate", "mult_i"):
            a, ta = ct(ins[0], step)
            ssa[out] = emit(f'"earth.negate"({a}) : ({ta}) -> {CT}', out, t, step)
        elif t in ("rotate", "conjugate"):
            off = 0
            for v in n["inputs"]:
                m = re.match(r"rot\((-?\d+)\)", v)
                if m and t == "rotate":
                    off = int(m.group(1))
            a, ta = ct(ins[0], step)
            ssa[out] = emit(f'"earth.rotate"({a}) <{{offset = array<i64: {off}>}}> : '
                            f'({ta}) -> {CT}', out, t, step)
        else:
            raise SystemExit(f"unhandled op_type {t}")
        if out in forced and out in ssa:
            o_, v_, t_, s_ = opmap[-1]
            opmap[-1] = (o_, v_ + ".pre", t_, s_)
            c = const_ssa(out + ".forced", step)
            ssa[out] = emit(f'"earth.mul"({ssa[out]}, {c}) : ({CT}, {PL}) -> {CT}',
                            out, "forced_rescale", step)

    rets = [n["output"] for n in nodes
            if n["op_type"] not in BOOT and n["op_type"] not in ALIAS
            and n["output"] not in consumed and res(n["output"]) in ssa]
    ret_ssa = [ssa[res(v)] for v in rets]
    ret_tys = ", ".join([CT] * len(ret_ssa))
    args = ", ".join(f"%arg{i}: {CT}" for i in range(len(ext)))
    body = "\n".join(lines)
    Path(out_base + ".mlir").write_text(
        f"module {{\n  func.func @{func_name}({args}) -> ({ret_tys}) {{\n{body}\n"
        f"    return {', '.join(ret_ssa)} : {ret_tys}\n  }}\n}}\n")
    json.dump({"opmap": opmap, "ext": ext, "rets": rets}, open(out_base + ".opmap.json", "w"))
    return opid[0]


if __name__ == "__main__":
    n = translate(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "block")
    print(f"{sys.argv[2]}: {n} earth ops")
