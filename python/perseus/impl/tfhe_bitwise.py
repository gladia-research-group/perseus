"""Native TFHE integer / bitwise ops on radix-4 words (impl/tfhe.py native programs: one switch in, PBS chains on LWEs,
one finish out). A word of `nd` digits d (base 4, least significant first; 8 to 64 bits) is nd native LWEs at value
d / 8. Noise (`analyze`): a lookup's input sd is the model pc-check tfhe-noise validates on the device (the mean-
compensated mod switch to 2 nu n_br / k and the key switch: 4.45e-3 at nu 1, k = 1), plus the lins' propagated PBS
output noise (`PBS_OUT_SD`); every table keeps p_fail per lookup < 2^-128:
- 8 bins on [-1, 1) (half bin 1/8: 2^-300 and more), or a negacyclic function of the whole torus at that spacing;
- 16 bins (half bin 1/16: 2^-137 with inputs a + 4 b, sum w^2 17): bivariate digit tables and group sums <= 15;
- many-LUT k = 2 on 8-bin inputs (the mod switch twice coarser: 2^-160): two functions of one input per rotation.
Every op computes its result in the program (digits at d / 4, linear ones included; predicates / flags at 1 / 4): the
store only reformats. The rounds of an op share one launch each (`Prog.pbsm`: a table per row).

- add / sub / neg: statuses (kill / propagate / generate) of a_i + b_i (sub: ~a + b, neg: a + 11..1, the output
  complemented), a Kogge-Stone prefix of the fewest levels: combines of fan-in 4 (statuses at 1/2 and 1/8, 2 S2 +
  S2' + 2 S8 + S8' on the whole torus; the first level's from one many-LUT lookup, a later level's as two tables per
  row of the level before) or 3 (4 S + 2 S' + S'', the negacyclic half carries the sign); the digits linear (a_i + b_i
  + c_(i-1) - 4 c_i): 2 / 3 / 3 / 4 rounds at 8 / 16 / 32 / 64 bits. `flags`: + the carry / borrow (free) and the
  signed overflow (one lookup).
- compare (lt / le / gt / ge, signed: the top group's first combine turned by the sign bits' half turns): statuses
  of a_i - b_i, the same combines (the fewest levels) as a tree from the top digit, the predicate's table last.
- eq / ne: [a_i != b_i], then one zero test of their sum on the whole torus (17 points at 1/8: only (0, 16) is
  negacyclic and their outputs differ); groups of 16 digits combined the same way.
- bitwise and / or / xor / andn / orn / xnor: one 16-bin lookup per digit on a + 4 b.
- mux: o = f(2 (a - b) / 8 + 2 sel) = -+(a - b) / 16, out = a + b + 2 o.
- public shifts / rotates / sext: whole digits are a renumbering, one bit one round (a digit's two bits from one
  many-LUT lookup). Encrypted amount: shifts a barrel shifter of AND-mux stages (amounts >= the width by an OR tree
  folded into the last stage); rotates one gated lookup per digit and stage, x + (sel ? x_s - x : 0) (16-bin on
  (x_s - x + 3) / 8 + 7 sel / 8, the digits carried linearly).
- mul: digit products (a b) mod 4 and div 4 by 16-bin lookups on a_i + 4 b_j (one launch, or, past one batch-path
  wave when the cost model prefers it, the excess riding the next launch), a carry-save tree of group lookups (sums
  <= 15 -> s mod 4, s div 4; the top column s mod 4 only; the rounds planned by a beam search with a DP closing
  round), the prefix adder. mulh / mulhsu / mulhu: the upper word, the signed terms as
  s ? ~b_i : 0 items. mulc: canonical signed digits of the constant, the same tree; divc / remc: q = (a m) >> S with
  the magic m, r = a - q d.
- div / rem (unsigned; signed: |a|, |b| by conditional negation, signs fixed after): radix-2 restoring division without
  shifting the remainder, the no-borrow of R - (b << k) over the active digits and a virtual overflow position from
  b's clz thermometer bits, the selection round folding the next step's statuses (gated lookups); b = 0: q = all ones,
  r = a.
- clz / ctz / log2: thermometer bits [n >= j] as zero tests, every output digit linear in them: 2 rounds.
- popcount: digit-pair counts, groups of 4 (sums <= 16 on the whole torus: mod 4, div 4 mod 4, [= 16]), the tree.
- min / max: the compare's selector and the AND mux; abs: conditional negation.
- with a public constant: andi / ori / xori (1 round), addi / lti / slti (pair statuses: a function of (a_i, a_(i-1))
  once c is known, span 2 in round 1), eqi (pair items). Single bits (bset / bclr / binv / bext, public index: 1
  round; encrypted index: the per-digit mask m_i = [(idx >> 1) == i] (1 + idx & 1), then a + 4 m tables: 3 rounds),
  orcb, rev8, zext16, shadd (k = 1, 3: (a << 1)'s digits by one bivariate lookup), shadduw / adduw, czeqz / cznez
  (the zero test as the AND mux's selector), clmul / clmulh / clmulr (bit products per digit pair, parity groups),
  the 32-bit ops on 64-bit words sign-extended (w32*).
`WordProg` lays words out on a slot set (a's digits, b's, then one sel per word) and appends the steps; `analyze`
gives every program's rounds, PBS count and p_fail per lookup; `case` emits a program with encrypted test inputs for
pc-check tfhe-prog, which runs it on the device's native ops (pc-check tfhe-bitwise runs its own add / cmp / mux / xor).
"""
from __future__ import annotations

import numpy as np

from . import tfhe


def _r4(m):
    return np.round(4.0 * np.asarray(m, dtype=np.float64))


def _sgn(m):
    return np.sign(_r4(m)) / 4.0


def _carry(m):
    return np.where(_r4(m) > 0, 0.125, 0.0)


def _carry_pm(m):              # the whole-torus carry: c / 8 - 1/16 (antisymmetric)
    return np.where(_r4(m) > 0, 0.0625, -0.0625)


def _digit_out(m):            # (s + c) mod 4 on bins -7/8 + k/4
    return np.mod(np.floor((np.asarray(m) + 1.0) * 4.0), 4.0) / 8.0


def _lt(m):
    return np.where(_r4(m) < 0, 0.125, 0.0)


def _mux(m):
    return -_r4(m) / 16.0


def _bitwise(op):
    f = {"xor": np.bitwise_xor, "and": np.bitwise_and, "or": np.bitwise_or}[op]

    def g(m):
        k = np.clip(np.floor((np.asarray(m) + 1.0) * 8.0), 0, 15).astype(np.int64)
        return f(k % 4, k // 4) / 8.0
    return g


def _load(m):                  # slot value (2 d - 3) / 4 -> native d / 8 (snapped: half a bin is 1/4)
    return np.clip(np.round((4.0 * np.asarray(m) + 3.0) / 2.0), 0, 3) / 8.0


def encode_digits(x, nd):
    """Words -> the slot values a load reads: digit d at (2 d - 3) / 4."""
    x = [int(v) for v in np.asarray(x).reshape(-1)]
    d = np.array([[(v >> (2 * i)) & 3 for i in range(nd)] for v in x], dtype=np.float64).reshape(len(x), nd)
    return ((2.0 * d - 3.0) / 4.0).reshape(-1)


import os as _os
MULTI = _os.environ.get("PC_TFHE_MULTI", "1") not in ("0", "")
FAN4 = _os.environ.get("PC_TFHE_FAN4", "1") not in ("0", "")
DIV_GFORM = _os.environ.get("PC_TFHE_DIV_GFORM", "1") not in ("0", "")
SPLIT = _os.environ.get("PC_TFHE_SPLIT", "auto")    # mul's products past one wave in the next round: auto / 1 / 0
BATCH_WAVE = 188               # LWEs of one batch-path wave (one per block, one block per SM of the RTX PRO 6000)


FAN4ALL = _os.environ.get("PC_TFHE_FAN4ALL", "1") not in ("0", "")


def _prefix_fans(nd, first4=True, later4=True):
    """The prefix combines' fan-ins over nd digits: the fewest levels (fan-in 4 first when `first4`, at any level
    when `later4`), then the fewest later fan-in 4 levels (each costs its previous level a second table)."""
    if nd <= 1:
        return []
    old, span = [], 1                          # the fan-in 3 plan (fan-in 4 first where it saves a level)
    while span < nd:
        f = 4 if (not old and first4 and _fan4_helps(nd)) else min(3, -(-nd // span))
        old.append(f)
        span *= f
    if not later4:
        return old
    lv, span = 0, 1
    while span < nd:
        span *= 4
        lv += 1
    if lv >= len(old):
        return old
    fans = [4] * lv                            # then fan-in 3 / 2 from the top while the span still covers nd
    for i in range(lv - 1, 0, -1):
        for f in (2, 3):
            t = fans[:i] + [f] + fans[i + 1:]
            if int(np.prod(t)) >= nd:
                fans = t
                break
    return fans


def _fan4_helps(n):
    """Whether a fan-in 4 first combine (then fan-in 3) takes fewer prefix rounds than fan-in 3 throughout."""
    def rounds(n, first):
        r, span = 0, 1
        while span < n:
            span *= first if r == 0 else 3
            r += 1
        return r
    return rounds(n, 4) < rounds(n, 3)


TREE_DP = _os.environ.get("PC_TFHE_TREE_DP", "1") not in ("0", "")


def _partitions(ms):
    """Every partition of items with maxima `ms` into groups (index lists) summing to <= 15."""
    n = len(ms)
    out = []

    def rec(i, groups):
        if i == n:
            out.append([list(g) for g in groups])
            return
        for g in groups:
            if sum(ms[k] for k in g) + ms[i] <= 15:
                g.append(i)
                rec(i + 1, groups)
                g.pop()
        groups.append([i])
        rec(i + 1, groups)
        groups.pop()
    rec(0, [])
    return out


_PART_MEMO = {}


def _col_options(ms, top, multi):
    """One column's options for a carry-save round, (mods, divs, PBS, groups), Pareto-reduced: a group's mod item
    min(3, S) stays, its div S // 4 goes up a column (none from the top column, whose groups take one lookup); a
    lone item <= 6 passes unchanged."""
    key = (tuple(ms), top, multi)
    if key in _PART_MEMO:
        return _PART_MEMO[key]
    opts = []
    for P in _partitions(ms):
        mods = divs = pbs = 0
        ok = True
        for g in P:
            sm = sum(ms[k] for k in g)
            if len(g) == 1 and sm <= 6:
                mods += sm                       # passes unchanged (a lone item needs no lookup)
                continue
            if sm <= 3:
                ok = False                       # a group that reduces nothing: covered by the all-pass partition
                break
            mods += min(3, sm)
            if top:
                pbs += 1
            else:
                divs += sm // 4
                pbs += 2 if (multi or sm > 7) else 1
        if ok:
            opts.append((mods, divs, pbs, P))
    best = []
    for o in sorted(opts, key=lambda t: (t[2], t[0], t[1])):
        if not any(b[0] <= o[0] and b[1] <= o[1] and b[2] <= o[2] for b in best):
            best.append(o)
    _PART_MEMO[key] = best
    return best


def _final_round(col, ncols, target, multi):
    """Whether one more carry-save round can leave every column of a word (`col[c]` = [(row, max)]) at <= target:
    the cheapest such partition per column (DP along the columns on the incoming carry bound), or None. Columns of
    more than 9 items are not tried (they need more than one round)."""
    if any(len(x) > 9 for x in col):
        return None
    INF = 1 << 30
    states = {0: (0, [])}                        # incoming carry bound -> (pbs, partitions so far)
    for c, x in enumerate(col):
        top = c == ncols - 1
        ms = [m for _, m in x]
        order = sorted(range(len(ms)), key=lambda k: -ms[k])
        opts = _col_options([ms[k] for k in order], top, multi)
        nxt = {}
        for inc, (cost, parts) in states.items():
            for mods, divs, pbs, P in opts:
                if mods + inc > target:
                    continue
                d = 0 if top else divs
                v = cost + pbs
                if v < nxt.get(d, (INF,))[0]:
                    nxt[d] = (v, parts + [[[order[k] for k in g] for g in P]])
        if not nxt:
            return None
        states = nxt
    return min(states.values(), key=lambda t: t[0])[1]


def _row_ms(rows):
    """est_ms's cost of one launch of `rows` PBS."""
    return next((t for b, t in COST_MS if rows <= b), COST_MS[-1][1] * rows / COST_MS[-1][0])


def _ffd(ms, cap, balanced=False):
    """Groups (index lists) of items with maxima `ms`, each summing to <= cap: first fit decreasing, or (balanced)
    the fewest groups of FFD filled least-loaded first (smaller carries)."""
    order = sorted(range(len(ms)), key=lambda k: -ms[k])
    gs = []
    for k in order:
        for g in gs:
            if sum(ms[i] for i in g) + ms[k] <= cap:
                g.append(k)
                break
        else:
            gs.append([k])
    if balanced and len(gs) > 1:
        n = len(gs)
        bs = [[] for _ in range(n)]
        for k in order:
            b = min((b for b in bs if sum(ms[i] for i in b) + ms[k] <= 15), key=lambda b: sum(ms[i] for i in b), default=None)
            if b is None:
                return gs
            b.append(k)
        return bs
    return gs


def _tree_step(cm, P, ncols, multi):
    """One carry-save round on the column maxima `cm` with the partition P (per column: groups; a lone item <= 6
    passes unchanged when its group says so, as `_emit_groups`): (new column maxima, PBS)."""
    new = [[] for _ in range(ncols)]
    pbs = 0
    for c in range(ncols):
        top = c == ncols - 1
        own = []
        for g in P[c]:
            sm = sum(cm[c][k] for k in g)
            if len(g) == 1 and sm <= 6:
                own.append(sm)
                continue
            own.append(min(3, sm))
            if top:
                pbs += 1
            else:
                pbs += 2 if (multi or sm > 7) else 1
                if sm >= 4 and c + 1 < ncols:
                    new[c + 1].append(1 if sm <= 7 else sm // 4)
        new[c] += own
    return new, pbs


def _plan_tree(cm, ncols, target, multi, words, pend, beam=10):
    """The carry-save rounds for one word's columns (maxima `cm`, the late items `pend` = [(column, max)] arriving
    after the first round): a beam search over per-round strategies (first fit decreasing into groups <= 15 / 13 /
    11 / 9, balanced, with the DP of `_final_round` closing as soon as it can), by est_ms. Returns the partitions per
    round (per column: groups, a lone item <= 6 passing unchanged)."""
    def strategies(state, first):
        outs = []
        for cap, bal, lazy in ((15, False, 0), (15, True, 0), (13, False, 0), (11, False, 0), (11, True, 0), (9, False, 0),
                               (15, False, 9), (15, False, 12), (15, True, 12)):
            P, inc = [], 0
            for c in range(ncols):
                ms = state[c]
                if sum(ms) + inc <= max(target, lazy) and all(m <= 6 for m in ms):
                    P.append([[k] for k in range(len(ms))])     # passes (lazy: left for a later round)
                    inc = 0
                    continue
                gs = _ffd(ms, cap, bal)
                gs = [g if (len(g) > 1 or ms[g[0]] > 3) else g for g in gs]
                P.append(gs)
                inc = sum((1 if sum(ms[k] for k in g) <= 7 else sum(ms[k] for k in g) // 4)
                          for g in gs if (len(g) > 1 or ms[g[0]] > 3) and sum(ms[k] for k in g) >= 4)
            outs.append(P)
        return outs

    def done(state):
        return all(sum(x) <= target for x in state)

    def greedy_rest(state):
        tot = 0.0
        for _ in range(12):
            if done(state):
                return tot
            fin = _final_round([[(None, m) for m in x] for x in state], ncols, target, multi)
            if fin is not None:
                return tot + _row_ms(_tree_step(state, fin, ncols, multi)[1] * words)
            P = strategies(state, False)[0]
            state, pbs = _tree_step(state, P, ncols, multi)
            tot += _row_ms(pbs * words)
        return tot + 100.0

    beams = [(0.0, [list(x) for x in cm], [], list(pend))]
    best = None
    for depth in range(12):
        nxt = []
        for cost, state, plan, pd in beams:
            prow = len(pd) * words
            if not pd:
                fin = _final_round([[(None, m) for m in x] for x in state], ncols, target, multi)
                if fin is not None:
                    new, pbs = _tree_step(state, fin, ncols, multi)
                    tot = cost + (_row_ms(pbs * words) if pbs else 0.0)
                    if pbs == 0:
                        tot = cost
                    if best is None or tot < best[0]:
                        best = (tot, plan + ([fin] if pbs else []))
                    continue
            seen = set()
            for P in strategies(state, not plan):
                new, pbs = _tree_step(state, P, ncols, multi)
                for c, m in pd:
                    new[c].append(m)
                key = tuple(tuple(x) for x in new)
                if key in seen:
                    continue
                seen.add(key)
                c2 = cost + _row_ms(pbs * words + prow)
                if done(new):
                    if best is None or c2 < best[0]:
                        best = (c2, plan + [P])
                    continue
                nxt.append((c2, new, plan + [P], []))
        if not nxt:
            break
        # rank by the cost so far plus a greedy completion (first fit decreasing <= 15, the DP closing): lookahead
        nxt.sort(key=lambda t: t[0] + greedy_rest(t[1]))
        beams = nxt[:beam]
        if best is not None and all(b[0] >= best[0] for b in beams):
            break
    return None if best is None else best[1]


class WordProg:
    """A native program over `words` words of `nd` radix-4 digits (`bits` = 2 nd): the loaded slot set holds a's
    digits ([word][digit]), then b's, then (mux) one sel per word at (4 sel - 3) / 4, read as a digit 0 / 2. Outputs:
    digits at value d / 4 (cmp: one value per word, 1 when a < b)."""

    def __init__(self, name, words, nd, fan=3):
        self.w, self.nd, self.fan = int(words), int(nd), int(fan)
        self.multi = MULTI             # one launch per round (pbsm: a table per row)
        self.fan4 = FAN4               # fan-in 4 first prefix combines where they save a round
        self.div_gform = DIV_GFORM     # division: every other selection round as one gated lookup per digit
        self.split_products = SPLIT    # mul: products beyond one batch wave in the next round (auto: by est_ms)
        self.bits_k2 = True            # a digit's two bits from one many-LUT lookup
        self.tree_dp = TREE_DP         # the carry-save tree's rounds planned (`_plan_tree`)
        if self.fan not in (2, 3):
            raise ValueError("tfhe_bitwise: prefix fan-in 2 or 3")
        if self.nd & (self.nd - 1):
            raise ValueError("tfhe_bitwise: digits per word must be a power of two")
        self.R = self.w * self.nd
        self.pr = tfhe.Prog(name, 5)
        p = self.pr
        p.lut("load", _load)
        for k, f in (("sgn", _sgn), ("carry", _carry), ("carry_pm", _carry_pm), ("lt", _lt), ("mux", _mux)):
            p.lut(k, f)
        p.lut("out_w", lambda m: 2.0 * _digit_out(m), warp=True)      # d / 4 for the finish
        p.lut("lt_w", lambda m: 8.0 * _lt(m), warp=True)               # 1 when a < b
        p.lut("lt_pm", lambda m: np.where(_r4(m) < 0, 0.0625, -0.0625))
        p.lut("lt_id_w", lambda m: np.where(np.asarray(m) > 0.0625, 1.0, 0.0), warp=True)   # lt / 8 -> lt
        p.lut("dig_w", lambda m: np.clip(np.round(8.0 * np.asarray(m)), 0, 3) / 4.0, warp=True)    # d / 8 in
        p.lut("dig4_w", lambda m: np.clip(np.round(4.0 * np.asarray(m)), 0, 3) / 4.0, warp=True)   # d / 4 in
        p.load("load", 0)

    def rows_in(self, sel=False):
        return 2 * self.R + (self.w if sel else 0)

    def add(self, out_slots):
        if self.fan == 3:
            return self._add3(out_slots)
        p, R, nd = self.pr, self.R, self.nd
        p.lin(1, 0, 0, [(r, R + r) for r in range(R)], ca=2, cb=2, c=-0.75)
        p.pbs(1, R, "sgn", [2])
        cur, nxt, d = 2, 3, 1
        while d < nd:
            last = 2 * d >= nd
            p.lin(4, cur, cur, [(r, r - d if r % nd >= d else -1) for r in range(R)], ca=2, cb=1)
            p.pbs(4, R, "carry" if last else "sgn", [nxt])
            cur, nxt, d = nxt, cur, 2 * d
        p.lin(4, 1, cur, [(r, r - 1 if r % nd else -1) for r in range(R)], ca=1, cb=2, c=-0.125)
        p.store(4, R, "out_w", out_slots)

    def _add3(self, out_slots):
        p, R, nd = self.pr, self.R, self.nd
        p.lin(1, 0, 0, [(r, R + r) for r in range(R)], ca=2, cb=2, c=-0.75)
        p.pbs(1, R, "sgn", [2])
        cur, nxt, d = 2, 3, 1
        while d < nd:
            f = min(3, -(-nd // d))
            last = d * f >= nd
            p.lin(4, cur, cur, [(r, r - d if r % nd >= d else -1) for r in range(R)], ca=1 << (f - 1), cb=1 << (f - 2))
            if f == 3:          # in place: row r of 4 reads only row r of 4
                p.lin(4, 4, cur, [(r, r - 2 * d if r % nd >= 2 * d else -1) for r in range(R)], ca=1, cb=1)
            p.pbs(4, R, "carry_pm" if last else "sgn", [nxt])
            cur, nxt, d = nxt, cur, d * f
        # c = g + 1/16 in {0, 1/8} (digit 0: no carry in), then (s + c) mod 4 in the store's table
        p.lin(nxt, cur, cur, [(r, -1) for r in range(R)], ca=1, cb=0, c=0.0625)
        p.lin(4, 1, nxt, [(r, r - 1 if r % nd else -1) for r in range(R)], ca=1, cb=2, c=-0.125)
        p.store(4, R, "out_w", out_slots)

    def _cmp3(self, out_slots):
        p, R, w = self.pr, self.R, self.w
        p.lin(1, 0, 0, [(r, R + r) for r in range(R)], ca=2, cb=-2)
        p.pbs(1, R, "sgn", [2])
        cur, nxt, k = 2, 3, self.nd
        while True:
            f = min(3, k)
            h = -(-k // f)
            src = [[(q // h) * k + (q % h) * f + f - 1 - x if (q % h) * f + f - 1 - x < k else -1 for x in range(f)]
                   for q in range(w * h)]
            p.lin(4, cur, cur, [(v[0], v[1] if f > 1 else -1) for v in src], ca=1 << (f - 1), cb=1 << max(f - 2, 0))
            if f == 3:
                p.lin(4, 4, cur, [(q, v[2]) for q, v in enumerate(src)], ca=1, cb=1)
            if h == 1 and f <= 2:          # the last combine (inside [-1, 1)) rides the store
                p.store(4, w, "lt_w", out_slots)
                return
            if h == 1:                     # a fan-in 3 last combine: the whole-torus lt, then the store
                p.lut("lt_pm8", TABLES["lt_pm8"][0])
                p.lut("id8_w", TABLES["id8_w"][0], warp=True)
                p.pbs(4, w, "lt_pm8", [nxt])
                p.lin(4, nxt, nxt, [(q, -1) for q in range(w)], ca=1, cb=0, c=0.125)
                p.store(4, w, "id8_w", out_slots)
                return
            p.pbs(4, w * h, "sgn", [nxt])
            cur, nxt, k = nxt, cur, h

    def cmp(self, out_slots):
        if self.fan == 3:
            return self._cmp3(out_slots)
        p, R, nd, w = self.pr, self.R, self.nd, self.w
        p.lin(1, 0, 0, [(r, R + r) for r in range(R)], ca=2, cb=-2)
        p.pbs(1, R, "sgn", [2])
        cur, nxt, k = 2, 3, nd
        while k > 1:
            h = k // 2
            p.lin(4, cur, cur, [(q // h * k + 2 * (q % h) + 1, q // h * k + 2 * (q % h)) for q in range(w * h)], ca=2, cb=1)
            if h == 1:
                p.store(4, w, "lt_w", out_slots)
                return
            p.pbs(4, w * h, "sgn", [nxt])
            cur, nxt, k = nxt, cur, h

    def mux(self, out_slots):
        p, R, nd = self.pr, self.R, self.nd
        p.lin(1, 0, 0, [(r, R + r) for r in range(R)], ca=2, cb=-2)
        # sel loaded as a digit 0 / 2 (d / 8): 8 x (2 / 8) = 2: the half turn; 0: none (x 8, not the x 16 a digit 3
        # needs: the loaded sel's noise reaches the lookup x 8, 2^-215 per lookup instead of 2^-115)
        p.lin(4, 1, 0, [(r, 2 * R + r // nd) for r in range(R)], ca=1, cb=8)
        p.pbs(4, R, "mux", [2])
        p.lin(3, 0, 0, [(r, R + r) for r in range(R)], ca=1, cb=1)
        p.lin(4, 3, 2, [(r, r) for r in range(R)], ca=1, cb=2)
        p.store(4, R, "dig4_w", out_slots)

    def bitwise(self, op, out_slots, rounds=1):
        """and / or / xor / andn (a & ~b) / orn (a | ~b) / xnor: one 16-bin lookup per digit on a + 4 b (rounds=2:
        the bits, then 8-bin tables on bit sums)."""
        if tfhe.GRID == 2048 and rounds == 2:
            return self._bitwise2(op, out_slots)
        if tfhe.GRID == 2048:
            R = self.R
            IN = self._lin(0, 0, [(r, R + r) for r in range(R)], ca=1, cb=4, c=-0.9375)
            O = self._pbs(IN, R, op + "16")
            return self._store(self._lin(O, O, [(r, -1) for r in range(R)], ca=2, cb=0), R, "dig4_w", out_slots)
        p, R = self.pr, self.R
        p.lut(op, lambda m: 2.0 * _bitwise(op)(m), warp=True)
        p.lin(1, 0, 0, [(r, R + r) for r in range(R)], ca=1, cb=4, c=-15.0 / 16.0)
        p.store(1, R, op, out_slots)

    def _bitwise2(self, op, out_slots):
        """and / or / xor at nu 1: the bits of a and b (one two-region batch), then one table on every bit pair's sum
        (lo and hi planes in one batch): 2 rounds, every lookup 8-bin."""
        R = self.R
        LH = self._lh(list(range(2 * R)))                 # lo of (a | b), then hi of (a | b): 4 R rows
        IN = self._lin(LH, LH, [(r, R + r) for r in range(R)] + [(2 * R + r, 3 * R + r) for r in range(R)], ca=2, cb=2)
        Y = self._pbs(IN, 2 * R, "bw_" + op)
        self._dig((Y, Y), [(r, R + r) for r in range(R)], ca=2, cb=4, out_slots=out_slots)

    def ident(self, out_slots):
        """The I/O baseline: load a, store it (read at d / 4: the 8-bin margin)."""
        self.pr.lin(1, 0, 0, [(r, -1) for r in range(self.R)], ca=2, cb=0)
        self.pr.store(1, self.R, "dig4_w", out_slots)

    # ── the other integer / bitwise ops (module doc) ─────────────────────────────────────────────────────────────

    def _buf(self):
        """A fresh buffer (buffer 0 holds the load; the program's buffer count follows)."""
        if not hasattr(self, "_nb"):
            self._nb = 5
        b = self._nb
        self._nb += 1
        self.pr.bufs = max(self.pr.bufs, self._nb)
        return b

    def _tab(self, name):
        if name not in self.pr.luts:
            if name in TABLESK:
                fs, warp = TABLESK[name]
                self.pr.lut(name, *fs, warp=warp)
            else:
                f, warp = TABLES[name]
                self.pr.lut(name, f, warp=warp)
        return name

    def _multi(self, jobs):
        """One launch for several lookups of one round: jobs = [(buffer, rows, lut)] (one-table luts); the inputs
        gathered into one buffer, a per-row table (`Prog.pbsm`) unless every job shares one. Returns (output
        buffer, the first row of every job)."""
        srcs, luts, offs = [], [], []
        for b, rows, lut in jobs:
            offs.append(len(srcs))
            srcs += [(b, r) for r in rows]
            luts += [self._tab(lut)] * len(rows)
        IN = jobs[0][0] if len(jobs) == 1 and list(jobs[0][1]) == list(range(len(jobs[0][1]))) else self._gather(srcs)
        if len(set(luts)) == 1:
            return self._pbs(IN, len(luts), luts[0]), offs
        dst = self._buf()
        self.pr.pbsm(IN, luts, dst)
        return dst, offs

    def _pbsk(self, src, rows, lut):
        """A many-LUT lookup (TABLESK: k tables from one rotation): one fresh buffer per table."""
        dst = [self._buf() for _ in TABLESK[lut][0]]
        self.pr.pbs(src, rows, self._tab(lut), dst)
        return dst

    def _pbs(self, src, rows, lut, dst=None):
        dst = self._buf() if dst is None else dst
        self.pr.pbs(src, rows, self._tab(lut), [dst])
        return dst

    def _lin(self, a, b, pairs, ca=1, cb=1, c=0.0, dst=None):
        dst = self._buf() if dst is None else dst
        self.pr.lin(dst, a, b, pairs, ca=ca, cb=cb, c=c)
        return dst

    def _store(self, src, rows, lut, out_slots):
        if getattr(self, "_capture", False):          # an inner program (`word32`): hand the result back
            self._captured = (src, rows, lut)
            return
        self.pr.store(src, rows, self._tab(lut) if lut else None, out_slots)
        self.out_rows = rows

    def _row(self, op, w, i):
        """Row of digit i of word w of operand op (0: a, 1: b) in the loaded buffer."""
        return op * self.R + w * self.nd + i

    def _top(self, op):
        return [self._row(op, w, self.nd - 1) for w in range(self.w)]

    def _carries(self, X):
        """The prefix on X = (s_i - 3) / 4 (R rows, s_i = x_i + y_i in 0..6): carries out of every digit, c_i in {0,
        1/8}. Combines of fan-in 4 (statuses at 1/2 and 1/8, 2 A + A' + 2 B + B'' on the whole torus at spacing 1/8)
        or 3 (statuses at 1/4); `_prefix_fans` picks the fewest rounds: 1 + ceil(log4 nd) with the later fan-in 4
        combines (`fan4all`: the level before outputs both scales, two tables per row in one launch)."""
        p, R, nd = self.pr, self.R, self.nd
        fans = _prefix_fans(nd, self.fan4, getattr(self, "fan4all", FAN4ALL) and self.multi)
        prev = lambda d: [(r, r - d if r % nd >= d else -1) for r in range(R)]
        U = self._buf()
        # level 0's statuses: (A, B) the 1/2 and 1/8 scales (buffer, row offset) when the next combine has fan-in 4,
        # else S at 1/4
        ident = list(range(R))
        if fans and fans[0] == 4:          # statuses at 1/8 and at 1/2 (many-LUT: the 8-bin input)
            S, S1 = self._buf(), self._buf()
            p.pbs(X, R, self._tab("sgn2s"), [S, S1])
            st = ("ab", (S1, 0, ident), (S, 0, ident))
        else:
            st = ("s", self._pbs(X, R, "sgn"))
        d = 1
        for li, f in enumerate(fans):
            last = li == len(fans) - 1
            U = self._buf()
            if f == 4:                     # (buffer, offset, row of position r)
                (A, oa, ma), (B, ob, mb) = st[1], st[2]
                p.lin(U, A, A, [(oa + ma[r], oa + ma[q] if q >= 0 else -1) for r, q in prev(d)], ca=2, cb=1)
                p.lin(U, U, B, [(r, ob + mb[q] if q >= 0 else -1) for r, q in prev(2 * d)], ca=1, cb=2)
                p.lin(U, U, B, [(r, ob + mb[q] if q >= 0 else -1) for r, q in prev(3 * d)], ca=1, cb=1)
                sp8 = True                 # U at spacing 1/8
            else:
                cur = st[1]
                p.lin(U, cur, cur, prev(d), ca=1 << (f - 1), cb=1 << (f - 2))
                if f == 3:
                    p.lin(U, U, cur, prev(2 * d), ca=1, cb=1)
                sp8 = False
            if last:
                C = self._pbs(U, R, "carry_pm8x" if sp8 else "carry_pm")
            elif fans[li + 1] == 4:         # both scales for the next fan-in 4 combine (two tables per row; the 1/8
                # scale only where that combine reads it: digits below nd - 2 d')
                dn = d * f
                rb = [r for r in range(R) if r % nd < nd - 2 * dn]
                OUT, offs = self._multi([(U, ident, "sgn8_2" if sp8 else "sgn1x"),
                                         (U, rb, "sgn8_8" if sp8 else "sgn8x")])
                st = ("ab", (OUT, offs[0], ident), (OUT, offs[1], {r: k for k, r in enumerate(rb)}))
            else:
                st = ("s", self._pbs(U, R, "sgn4x" if sp8 else "sgn"))
            d *= f
        self._lin(C, C, [(r, -1) for r in range(R)], ca=1, cb=0, c=0.0625, dst=C)
        return C

    def _signs(self, extra=None):
        """The half turns 2 sa + 2 sb of every word (w rows; the sign bits as one lookup on the top digits, w + w
        rows); `extra` adds 2 (a half turn) to all."""
        w = self.w
        HI = self._lin(0, 0, [(r, -1) for r in self._top(0) + self._top(1)], ca=2, cb=0)
        H = self._pbs(HI, 2 * w, "hs")
        return self._lin(H, H, [(q, w + q) for q in range(w)], ca=1, cb=1, c=2.0 if extra else 0.0)

    def _adder(self, kind, out_slots, flags=False, X=None, xy=None):
        """add / sub / neg (module doc). `X`, `xy`: an adder input built elsewhere ((s - 3) / 4 and the two operand
        sources (buffer, pairs) of its linear output; mul's final add)."""
        p, R, nd, w = self.pr, self.R, self.nd, self.w
        if X is None:
            ab = [(r, R + r) for r in range(R)]
            if kind == "add":
                X = self._lin(0, 0, ab, ca=2, cb=2, c=-0.75)
            elif kind == "sub":                 # ~a + b
                X = self._lin(0, 0, ab, ca=-2, cb=2)
            else:                               # a + 11..1
                X = self._lin(0, 0, [(r, -1) for r in range(R)], ca=2, cb=0)
        H = self._signs(extra=kind == "sub") if flags else None
        C = self._carries(X)
        prev = [(r, r - 1 if r % nd else -1) for r in range(R)]
        if not flags:                           # linear output digits (s + c_(i - 1) - 4 c_i), complemented for sub / neg
            O = self._digits_lin(X, C)
            if kind in ("sub", "neg"):
                self._lin(O, O, [(r, -1) for r in range(R)], ca=-1, cb=0, c=0.75, dst=O)
            self._store(O, R, "dig4_w", out_slots)
            return
        # linear digits at d / 4: 2 s / 8 + 2 c_(i - 1) - 8 c_i (X + 3/4 = 2 s / 8), complemented for sub / neg
        O = self._lin(X, C, prev, ca=1, cb=2, c=0.75)
        self._lin(O, C, [(r, r) for r in range(R)], ca=1, cb=-8, dst=O)
        if kind in ("sub", "neg"):
            self._lin(O, O, [(r, -1) for r in range(R)], ca=-1, cb=0, c=0.75, dst=O)
        tops = [r for r in range(R) if r % nd == nd - 1]
        V = self._lin(X, C, [(r, r - 1) for r in tops], ca=1, cb=2, c=-0.125)
        self._lin(V, H, [(q, q) for q in range(w)], ca=1, cb=1, dst=V)
        F = self._pbs(V, w, "ovf_pm")
        # rows: digits | carry (borrow for sub) | overflow, all at d / 4; the overflow rows 2 F + 1/8 ({0, 1/4}):
        # a per-row constant from a noiseless row (a lin of no terms: trivial LWEs of c)
        K = self._lin(0, 0, [(-1, -1)], ca=0, cb=0, c=0.125)
        G = self._lin(C, F, [(t, -1) for t in tops] + [(-1, q) for q in range(w)], ca=2, cb=2)
        self._lin(G, K, [(q, -1) for q in range(w)] + [(w + q, 0) for q in range(w)], ca=1, cb=1, dst=G)
        OUT = self._lin(O, G, [(r, -1) for r in range(R)] + [(-1, q) for q in range(2 * w)], ca=1, cb=1)
        self._store(OUT, R + 2 * w, "dig4_w", out_slots)

    def sub(self, out_slots, flags=False):
        """(a - b) mod 2^bits (flags: + borrow, overflow rows)."""
        self._adder("sub", out_slots, flags)

    def neg(self, out_slots):
        """(-a) mod 2^bits."""
        self._adder("neg", out_slots)

    def add_flags(self, out_slots):
        """a + b with the carry and the signed overflow rows."""
        self._adder("add", out_slots, True)

    def _tree(self, pred=None, signed=False, sel=False):
        """The compare tree (statuses sign(a_i - b_i), combines from the top digit: fan-in 4 (statuses at 1/2 and
        1/8, 2 A + A' + 2 B + B'' on the whole torus at spacing 1/8) or 3 (statuses at 1/4), the fewest levels as the
        adder's prefix (`_prefix_fans`)). Returns the buffer of w rows: the predicate (lt / gt / le / ge / eq / ne)
        at d / 8 x 8 = 1 when `sel` (unit 1, an AND mux selector), else stores nothing and returns (buffer, table)
        for the store."""
        p, R, w, nd = self.pr, self.R, self.w, self.nd
        X = self._lin(0, 0, [(r, R + r) for r in range(R)], ca=2, cb=-2)
        # a fan-in 4 first combine on statuses at two scales (1/2 and 1/8); signed words of > 16 digits keep fan-in 3
        # there (their round-1 launch would pass 64 rows with both scales and the sign bits)
        f4 = self.fan4 and not (signed and nd > 16)
        fans = _prefix_fans(nd, f4, getattr(self, "fan4all", FAN4ALL) and self.multi)
        if signed and self.multi:              # the statuses and the sign bits in one launch
            HI = self._lin(0, 0, [(r, -1) for r in self._top(0) + self._top(1)], ca=2, cb=0)
            jobs = [(X, list(range(R)), "sgn8x" if fans[0] == 4 else "sgn")] + \
                ([(X, list(range(R)), "sgn1x")] if fans[0] == 4 else []) + [(HI, list(range(2 * w)), "hs")]
            S, offs = self._multi(jobs)
            oh = offs[-1]
            st = ("ab", (S, offs[1], list(range(R))), (S, offs[0], list(range(R)))) if fans[0] == 4 else ("s", S)
            HS = self._lin(S, S, [(oh + q, oh + w + q) for q in range(w)], ca=1, cb=1)
        else:
            if fans[0] == 4:
                S, S1b = self._pbsk(X, R, "sgn2s")
                st = ("ab", (S1b, 0, list(range(R))), (S, 0, list(range(R))))
            else:
                st = ("s", self._pbs(X, R, "sgn"))
            HS = self._signs() if signed else None
        k = nd
        for li, f in enumerate(fans):
            h = -(-k // f)
            src = [[(q // h) * k + (q % h) * f + f - 1 - x if (q % h) * f + f - 1 - x < k else -1 for x in range(f)]
                   for q in range(w * h)]
            U = self._buf()
            if f == 4:                         # 2 A[top] + A[.] + 2 B[.] + B[.] (whole torus, spacing 1/8; sum w^2 10)
                (A, oa, ma), (B, ob, mb) = st[1], st[2]
                p.lin(U, A, A, [(oa + ma[v[0]], oa + ma[v[1]] if v[1] >= 0 else -1) for v in src], ca=2, cb=1)
                p.lin(U, U, B, [(q, ob + mb[v[2]] if v[2] >= 0 else -1) for q, v in enumerate(src)], ca=1, cb=2)
                p.lin(U, U, B, [(q, ob + mb[v[3]] if v[3] >= 0 else -1) for q, v in enumerate(src)], ca=1, cb=1)
            else:
                cur = st[1]
                p.lin(U, cur, cur, [(v[0], v[1] if f > 1 else -1) for v in src], ca=1 << (f - 1), cb=1 << max(f - 2, 0))
                if f == 3:
                    p.lin(U, U, cur, [(q, v[2]) for q, v in enumerate(src)], ca=1, cb=1)
            if li == 0 and signed:
                p.lin(U, U, HS, [(q, q // h if q % h == h - 1 else -1) for q in range(w * h)], ca=1, cb=1)
            if h == 1:
                break
            if fans[li + 1] == 4:              # both scales for the next fan-in 4 combine (two tables per row; the 1/8
                # scale only for the groups it reads at that scale)
                h2 = -(-h // 4)
                rb = sorted({(q // h2) * h + (q % h2) * 4 + 3 - x for q in range(w * h2) for x in (2, 3)
                             if (q % h2) * 4 + 3 - x < h})
                OUT, offs = self._multi([(U, list(range(w * h)), "sgn8_2" if f == 4 else "sgn1x"),
                                         (U, rb, "sgn8_8" if f == 4 else "sgn8x")])
                st = ("ab", (OUT, offs[0], list(range(w * h))), (OUT, offs[1], {r: k for k, r in enumerate(rb)}))
            else:
                st = ("s", self._pbs(U, w * h, "sgn" if f < 4 else "sgn4x"))
            k = h
        if f == 4:                             # the last combine at spacing 1/8: its whole-torus tables
            if sel:
                V = self._pbs(U, w, pred + "_pm1x")
                return self._lin(V, V, [(q, -1) for q in range(w)], ca=1, cb=0, c=0.5)
            V = self._pbs(U, w, pred + "_pm8x")
            return self._lin(V, V, [(q, -1) for q in range(w)], ca=1, cb=0, c=0.125), "id8_w"
        # the last combine's input U (w rows): inside [-1, 1) for fan-in <= 2, the whole torus for 3
        if f == 3 and pred in ("eq", "ne"):
            U = self._pbs(U, w, "sgn")          # the status, then the eq / ne table inside [-1, 1)
            f = 2
        if f <= 2:
            if sel:
                return self._pbs(U, w, pred + "1")
            return self._pbs(U, w, pred + "_4"), "id8_w"       # {0, 1/4}: the result in the program, not the store
        V = self._pbs(U, w, pred + ("_pm1" if sel else "_pm8"))
        if sel:
            return self._lin(V, V, [(q, -1) for q in range(w)], ca=1, cb=0, c=0.5)
        return self._lin(V, V, [(q, -1) for q in range(w)], ca=1, cb=0, c=0.125), "id8_w"

    def cmp_pred(self, out_slots, pred="lt", signed=False):
        """[a pred b] per word (pred lt / gt / le / ge / eq / ne; signed: two's complement), 1 / 0."""
        if pred not in ("lt", "gt", "le", "ge", "eq", "ne"):
            raise ValueError(f"tfhe_bitwise cmp: predicate {pred}?")
        buf, lut = self._tree(pred, signed)
        self._store(buf, self.w, lut, out_slots)

    def eq_ne(self, out_slots, ne=False):
        """[a == b] / [a != b]: u_i = [a_i != b_i] (one 8-bin lookup on a_i - b_i), then [sum of up to 16 u_i == 0] on
        the whole torus (17 points at 1/8: only the pair (0, 16) is negacyclic, and their outputs differ), groups of
        16 digits combined by another such lookup: 2 rounds up to 32 bits, 3 at 64."""
        w, nd, R = self.w, self.nd, self.R
        X = self._lin(0, 0, [(r, R + r) for r in range(R)], ca=2, cb=-2)
        U = self._pbs(X, R, "neq8")                        # [a_i != b_i] / 8
        rows = [[q * nd + i for i in range(nd)] for q in range(w)]
        while True:
            gs = [r[k:k + 16] for r in rows for k in range(0, len(r), 16)]
            last = all(len(r) <= 16 for r in rows)
            S = self._gsum(U, gs, ca=1, c=-0.9375)
            V = self._pbs(S, len(gs), ("ne" if ne else "eq") + "z_4" if last else "nez_pm")
            # +-v -> {0, 2 v}: the whole-torus outputs are antisymmetric
            V = self._lin(V, V, [(g, -1) for g in range(len(gs))], ca=1, cb=0, c=0.125 if last else 0.0625)
            if last:
                return self._store(V, w, "id8_w", out_slots)
            per = [(len(r) + 15) // 16 for r in rows]
            off = np.concatenate([[0], np.cumsum(per)])
            rows = [list(range(off[q], off[q + 1])) for q in range(w)]
            U = V

    def _dig(self, src, pairs, ca=1, cb=1, out_slots=None, rows=None):
        """Output digits: a lin into d / 4 (`ca`, `cb` already doubled), then the store's identity table."""
        O = self._lin(src[0], src[1], pairs, ca=ca, cb=cb)
        self._store(O, len(pairs) if rows is None else rows, "dig4_w", out_slots)
        return O

    def _lh(self, rows, src=0):
        """The two-region bits table on rows `rows` of `src` (digits at d / 8): buffer [lo (d & 1) / 8 | hi (d >> 1)
        / 8], len(rows) each."""
        if self.bits_k2:              # lo and hi from one many-LUT lookup per digit (half the rows), then [lo | hi]
            n = len(rows)
            IN = self._lin(src, src, [(r, -1) for r in rows], ca=2, cb=0, c=-0.375)
            LO, HI = self._pbsk(IN, n, "bitsk")
            return self._lin(LO, HI, [(r, -1) for r in range(n)] + [(-1, r) for r in range(n)], ca=1, cb=1)
        IN = self._lin(src, src, [(r, -1) for r in rows] + [(-1, r) for r in rows], ca=2, cb=-2, c=0.0625)
        return self._pbs(IN, 2 * len(rows), "bits")

    def sext(self, out_slots, from_bits):
        """Sign extension of a's low `from_bits` bits (a multiple of 2) to the word."""
        m, nd, w = int(from_bits) // 2, self.nd, self.w
        if from_bits % 2 or not 0 < m <= nd:
            raise ValueError("tfhe_bitwise sext: from_bits a multiple of 2, at most the width")
        LH = self._lh([self._row(0, q, m - 1) for q in range(w)])
        # out: a_i (i < m), 3 hi(a_(m - 1)) above: hi at d / 8 x 3 x 2
        pairs = [(self._row(0, q, i), -1) if i < m else (-1, w + q) for q in range(w) for i in range(nd)]
        self._dig((0, LH), pairs, ca=2, cb=6, out_slots=out_slots)

    def _src(self, kind, i, s):
        """The source digit of output digit i for a shift / rotate by s digits (None: the fill)."""
        nd = self.nd
        if kind in ("rotl", "rotr"):
            return (i - s) % nd if kind == "rotl" else (i + s) % nd
        j = i - s if kind == "shl" else i + s
        return j if 0 <= j < nd else None

    def shift(self, out_slots, kind, k):
        """a shifted / rotated by the public amount k (kind shl / shr / sar / rotl / rotr): whole digits renumbered,
        an odd bit through one round of the bits table; shifts by >= the width give 0 (sar: the sign)."""
        nd, w, R, bits = self.nd, self.w, self.R, 2 * self.nd
        if kind not in ("shl", "shr", "sar", "rotl", "rotr"):
            raise ValueError(f"tfhe_bitwise shift: {kind}?")
        k = int(k) % bits if kind.startswith("rot") else min(int(k), bits)
        q, t = divmod(k, 2)
        need_sign = kind == "sar" and (t or q)
        if not t:
            if need_sign:
                LH = self._lh(self._top(0))
                fill = lambda qq: (-1, w + qq)          # 3 hi(top) at d / 8: cb 6 at d / 4
            else:
                LH, fill = 0, lambda qq: (-1, -1)
            pairs = []
            for qq in range(w):
                for i in range(nd):
                    j = self._src(kind if kind != "sar" else "shr", i, q)
                    pairs.append((self._row(0, qq, j), -1) if j is not None else fill(qq))
            self._dig((0, LH), pairs, ca=2, cb=6, out_slots=out_slots)
            return
        # one bit (+ q digits): bit-shifted digits b_i, then renumbered: lo / hi of every digit (row r: lo, R + r: hi)
        LH = self._lh(list(range(R)))
        left = kind in ("shl", "rotl")
        rows = []                                # the bit-shifted word (before the digit renumbering): (lo, hi) rows
        for qq in range(w):
            for i in range(nd):
                r = qq * nd + i
                if left:                         # 2 lo_i + hi_(i - 1)
                    jj = i - 1 if i > 0 else (nd - 1 if kind == "rotl" else None)
                    rows.append((r, R + qq * nd + jj if jj is not None else -1))
                else:                            # hi_i + 2 lo_(i + 1)
                    jj = i + 1 if i < nd - 1 else (0 if kind == "rotr" else None)
                    if jj is not None:
                        rows.append((qq * nd + jj, R + r))
                    elif kind == "sar":          # 2 lo(fill) + hi_top = 3 hi_top
                        rows.append((-2, R + r))
                    else:
                        rows.append((-1, R + r))
        # 2 lo + hi at d / 8 (the sar top row: 2 hi + hi, hi through the lo slot too)
        BS = self._lin(LH, LH, [(b if a == -2 else a, b) for a, b in rows], ca=2, cb=1)
        if q == 0:
            self._store(self._lin(BS, BS, [(r, -1) for r in range(R)], ca=2, cb=0), R, "dig4_w", out_slots)
            return
        pairs = []
        for qq in range(w):
            for i in range(nd):
                j = self._src(kind if kind != "sar" else "shr", i, q)
                if j is not None:
                    pairs.append((qq * nd + j, -1))
                elif kind == "sar":
                    pairs.append((-1, R + qq * nd + nd - 1))     # 3 hi(top) at d / 8
                else:
                    pairs.append((-1, -1))
        self._dig((BS, LH), pairs, ca=2, cb=6, out_slots=out_slots)

    def _andmux(self, groups, dst_rows):
        """sum_g and(sel_g, x_g) per output row in one batch: `groups` = [(X buffer, x rows, SEL buffer, sel rows)],
        every group dst_rows long (sel at unit 1, x at d / 8). Returns the buffer of the sums (d / 8)."""
        n = dst_rows
        XX = self._buf()
        SS = self._buf()
        for g, (xb, xr, sb, sr) in enumerate(groups):
            if g == 0:
                self.pr.lin(XX, xb, xb, [(r, -1) for r in xr], ca=1, cb=0)
                self.pr.lin(SS, sb, sb, [(r, -1) for r in sr], ca=1, cb=0)
            else:                                # append: rows g n .. (g + 1) n (a fresh buffer each time)
                X2, S2 = self._buf(), self._buf()
                self.pr.lin(X2, XX, xb, [(r, -1) for r in range(g * n)] + [(-1, r) for r in xr], ca=1, cb=1)
                self.pr.lin(S2, SS, sb, [(r, -1) for r in range(g * n)] + [(-1, r) for r in sr], ca=1, cb=1)
                XX, SS = X2, S2
        G = len(groups)
        IN = self._lin(XX, SS, [(r, r) for r in range(G * n)], ca=2, cb=1, c=-0.875)
        A = self._pbs(IN, G * n, "and")
        O = self._lin(A, A, [(r, n + r if G > 1 else -1) for r in range(n)], ca=1, cb=1)
        for g in range(2, G):
            self._lin(O, A, [(r, g * n + r) for r in range(n)], ca=1, cb=1, dst=O)
        return O

    def _or_tree(self, buf, items):
        """OR of every word's items (rows of `buf`, (row, max) with max <= 3): groups summing to <= 7 through the
        threshold table, until one bit per word. Returns (buffer, rows) of the w bits at d / 8."""
        cur, its = buf, items                     # its[q] = [(row, max)]
        while any(len(x) > 1 or x[0][1] > 1 for x in its):
            groups, nxt = [], []
            for x in its:
                gs, g, s = [], [], 0
                for row, mx in sorted(x, key=lambda v: -v[1]):
                    if g and s + mx > 7:
                        gs.append(g)
                        g, s = [], 0
                    g.append(row)
                    s += mx
                gs.append(g)
                nxt.append(gs)
                groups.extend(gs)
            sums = self._gsum(cur, groups)
            B = self._pbs(sums, len(groups), "nz")
            k, its = 0, []
            for gs in nxt:
                its.append([(k + j, 1) for j in range(len(gs))])
                k += len(gs)
            cur = B
        return cur, [x[0][0] for x in its]

    def _gsum(self, buf, groups, ca=2, c=-0.875):
        """Row g = ca x sum of rows groups[g] of `buf` + c (the threshold tables' input 2 s / 8 - 7 / 8)."""
        m = max(len(g) for g in groups)
        S = self._lin(buf, buf, [(g[0], g[1] if len(g) > 1 else -1) for g in groups], ca=ca, cb=ca, c=c)
        # (a group [-1]: no term, the constant alone)
        for t in range(2, m):
            self._lin(S, buf, [(gi, g[t] if len(g) > t else -1) for gi, g in enumerate(groups)], ca=1, cb=ca, dst=S)
        return S

    def shift_enc(self, out_slots, kind):
        """a shifted / rotated by the encrypted amount b (a word; kind shl / shr / sar / rotl / rotr): a barrel
        shifter of log2(bits) AND-mux stages; shifts by >= the width give 0 (sar: the sign), rotates take b mod
        the width."""
        nd, w, R, bits = self.nd, self.w, self.R, 2 * self.nd
        if kind not in ("shl", "shr", "sar", "rotl", "rotr"):
            raise ValueError(f"tfhe_bitwise shift_enc: {kind}?")
        L = bits.bit_length() - 1
        na = (L + 1) // 2                          # amount digits holding bits 0 .. L - 1
        # round 1, one batch: lo / hi of a's digits (stage 0, the sign) and of the amount's low digits
        arows = [self._row(1, q, i) for q in range(w) for i in range(na)]
        LH = self._lh(list(range(R)) + arows)
        n1 = R + len(arows)                        # LH rows: lo of (a | amount), then hi of the same
        lo = lambda r: r
        hi = lambda r: n1 + r
        bit = lambda q, k: (lo if k % 2 == 0 else hi)(R + q * na + k // 2)      # bit k of word q's amount (/ 8)
        # selectors at unit 1: sel = 8 bit, ~sel = 1 - 8 bit (rows k w + q)
        SEL = self._lin(LH, LH, [(bit(q, k), -1) for k in range(L) for q in range(w)], ca=8, cb=0)
        NSEL = self._lin(LH, LH, [(bit(q, k), -1) for k in range(L) for q in range(w)], ca=-8, cb=0, c=1.0)
        shifting = not kind.startswith("rot")
        FILL = None
        if kind == "sar":                          # 3 hi(top) at d / 8, per word
            FILL = self._lin(LH, LH, [(hi(q * nd + nd - 1), -1) for q in range(w)], ca=3, cb=0)
        if shifting:
            # amount >= 2^L: any bit >= L: the digits above bit L (and hi of digit L // 2 when L is odd)
            q0 = L // 2
            items_rows = []
            ITEMS_pairs = []
            for q in range(w):
                its = []
                if L % 2:
                    ITEMS_pairs.append((-1, hi(R + q * na + q0) if q0 < na else -1))
                    its.append((len(ITEMS_pairs) - 1, 1))
                for i in range(q0 + (L % 2), nd):
                    ITEMS_pairs.append((self._row(1, q, i), -1))
                    its.append((len(ITEMS_pairs) - 1, 3))
                items_rows.append(its)
            IT = self._lin(0, LH, ITEMS_pairs, ca=1, cb=1)
            BIG, brow = self._or_tree(IT, items_rows)
            # last stage's selectors: s1 = [bit + ~big >= 2], s0 = [~bit + ~big >= 2] (unit 1), big (unit 1)
            kl = L - 1
            # (2 s / 8 - 7 / 8: s1's s = bit + 1 - big, s0's s = 2 - bit - big)
            IN = self._lin(LH, BIG, [(bit(q, kl), brow[q]) for q in range(w)], ca=2, cb=-2, c=-0.625)
            IN2 = self._lin(LH, BIG, [(bit(q, kl), brow[q]) for q in range(w)], ca=-2, cb=-2, c=-0.375)
            INS = self._lin(IN, IN2, [(q, -1) for q in range(w)] + [(-1, q) for q in range(w)], ca=1, cb=1)
            LS = self._pbs(INS, 2 * w, "ge2u")
            BIGU = self._lin(BIG, BIG, [(brow[q], -1) for q in range(w)], ca=8, cb=0)
        # stage 0: one bit; x_s from LH (as the public shift by 1), then the stages of 1, 2, 4 .. digits
        rows = []
        for q in range(w):
            for i in range(nd):
                r = q * nd + i
                if kind in ("shl", "rotl"):
                    jj = i - 1 if i > 0 else (nd - 1 if kind == "rotl" else None)
                    rows.append((lo(r), hi(q * nd + jj) if jj is not None else -1))
                else:
                    jj = i + 1 if i < nd - 1 else (0 if kind == "rotr" else None)
                    if jj is not None:
                        rows.append((lo(q * nd + jj), hi(r)))
                    elif kind == "sar":
                        rows.append((hi(r), hi(r)))
                    else:
                        rows.append((-1, hi(r)))
        XS = self._lin(LH, LH, rows, ca=2, cb=1)
        X = 0
        xr = list(range(R))
        last_k = L - 1
        for k in range(L):
            if k > 0:                              # shift by 2^(k - 1) digits
                s = 1 << (k - 1)
                pairs = []
                for q in range(w):
                    for i in range(nd):
                        j = self._src(kind if kind != "sar" else "shr", i, s)
                        if j is not None:
                            pairs.append((q * nd + j, -1))
                        else:
                            pairs.append((-1, q) if kind == "sar" else (-1, -1))
                XS = self._lin(X, FILL if FILL is not None else X, pairs, ca=1, cb=1)
                xr = list(range(R))
            sel_rows = [k * w + q for q in range(w) for _ in range(nd)]
            if shifting and k == last_k:
                g = [(XS, list(range(R)), LS, [q for q in range(w) for _ in range(nd)]),
                     (X, xr, LS, [w + q for q in range(w) for _ in range(nd)])]
                if kind == "sar":
                    g.append((FILL, [q for q in range(w) for _ in range(nd)], BIGU, [q for q in range(w) for _ in range(nd)]))
                X = self._andmux(g, R)
            else:
                X = self._andmux([(XS, list(range(R)), SEL, sel_rows), (X, xr, NSEL, sel_rows)], R)
            xr = list(range(R))
        self._store(self._lin(X, X, [(r, -1) for r in range(R)], ca=2, cb=0), R, "dig4_w", out_slots)

    def _reduce(self, ITEMS, cols, ncols):
        """Carry-save tree: `cols[q][c]` = [(row of ITEMS, max)] the items of word q's column c (digit position), until
        every column sums to <= 6 (the prefix adder's s). A round, low columns first: a column whose items plus the
        carries arriving this round could exceed 6 is cut into groups summing to <= 7, each one lookup pair (the
        digit s mod 4 and the carry s >= 4, both cleaned: two batches) -- every carry lands in a column that this
        same round compresses if it must, so carries never ripple. Returns (ITEMS, cols)."""
        while any(sum(m for _, m in x) > 6 for col in cols for x in col):
            groups, plan = [], []
            for q, col in enumerate(cols):
                inc = 0
                for c, x in enumerate(col):
                    tot = sum(m for _, m in x) + inc
                    ents, gs = [], []
                    if tot <= 6:
                        ents = [("pass", r, m) for r, m in x]
                    else:                        # first fit decreasing into groups of sum <= 7
                        for r, m in sorted(x, key=lambda v: -v[1]):
                            for g in gs:
                                if sum(v for _, v in g) + m <= 7:
                                    g.append((r, m))
                                    break
                            else:
                                gs.append([(r, m)])
                        for g in gs:
                            if len(g) == 1:
                                ents.append(("pass", g[0][0], g[0][1]))
                            else:
                                ents.append(("grp", len(groups), g))
                                groups.append(g)
                    inc = sum(1 for e in ents if e[0] == "grp" and sum(m for _, m in e[2]) >= 4)
                    plan.append((q, c, ents))
            S = self._gsum(ITEMS, [[r for r, _ in g] for g in groups], ca=2, c=-0.875)      # 2 s / 8 - 7 / 8
            CY = self._pbs(S, len(groups), "cy7")                                          # [s >= 4] / 8
            DG = self._pbs(S, len(groups), "md7")                                          # (s mod 4) / 8
            pass_pairs, dig_pairs = [], []
            ncol = [[[] for _ in range(ncols)] for _ in cols]
            for q, c, ents in plan:
                for e in ents:
                    if e[0] == "pass":
                        ncol[q][c].append((len(pass_pairs), e[2]))
                        pass_pairs.append((e[1], -1))
                        dig_pairs.append(-1)
                    else:
                        gi, g = e[1], e[2]
                        sm = sum(m for _, m in g)
                        ncol[q][c].append((len(pass_pairs), min(3, sm)))
                        pass_pairs.append((-1, -1))
                        dig_pairs.append(gi)
                        if sm >= 4 and c + 1 < ncols:
                            ncol[q][c + 1].append((len(pass_pairs), 1))
                            pass_pairs.append((-1, gi))
                            dig_pairs.append(-1)
            N = self._lin(ITEMS, CY, pass_pairs, ca=1, cb=1)
            self._lin(N, DG, [(r, d) for r, d in enumerate(dig_pairs)], ca=1, cb=1, dst=N)
            ITEMS, cols = N, ncol
        return ITEMS, cols

    def _final_add(self, ITEMS, cols, nout, out_slots):
        """Columns summing to <= 6 added by the prefix adder (nout digits per word, a word = nout rows)."""
        w = self.w
        if all(sum(m for _, m in x) <= 3 for col in cols for x in col):
            # no carries: every column is its digit (s <= 3)
            g = [[r for r, _ in col[c]] or [-1] for col in cols for c in range(nout)]
            O = self._gsum(ITEMS, g, ca=2, c=0.0)
            self._store(O, w * nout, "dig4_w", out_slots)
            return
        save = (self.nd, self.R)
        self.nd, self.R = nout, w * nout
        g = [[r for r, _ in col[c]] or [-1] for col in cols for c in range(nout)]
        X = self._gsum(ITEMS, g, ca=2, c=-0.75)                                            # (s - 3) / 4
        C = self._carries(X)
        prev = [(r, r - 1 if r % nout else -1) for r in range(self.R)]
        Z = self._lin(X, C, prev, ca=1, cb=2, c=-0.125)
        self._store(Z, self.R, "out_w", out_slots)
        self.nd, self.R = save

    def _reduce16(self, ITEMS, cols, ncols, target=6, pending=None):
        """Carry-save tree with 16-bin group lookups: `cols[q][c]` = [(row of ITEMS, max)]; a round cuts every
        column whose items plus this round's incoming carries could pass `target` into groups summing to <= 15
        (first fit decreasing): a group <= 7 is one many-LUT lookup (s mod 4, s >= 4: `mdcy7`), a larger one two
        (s mod 4, s div 4 <= 3: `md15` / `dv15`); the top column keeps s mod 4 only. Returns (ITEMS, cols)."""
        pend = list(pending or [])                   # [(word, column, (buffer, row) input, table, max)]
        plan_r, ri = None, 0
        if self.tree_dp:                             # every word's columns alike: one planned partition per round
            cm0 = [[m for _, m in x] for x in cols[0]]
            if all([[m for _, m in x] for x in col] == cm0 for col in cols) and all(x[0] < len(cols) for x in pend):
                pw = [(x[1], x[4]) for t in sorted({x[3] for x in pend}) for x in pend if x[3] == t and x[0] == 0]
                plan_r = _plan_tree(cm0, ncols, target, self.multi, len(cols), pw)
        while (ri < len(plan_r)) if plan_r is not None else (pend or any(sum(m for _, m in x) > target for col in cols for x in col)):
            g8, g16, gtop7, gtop15, plan = [], [], [], [], []
            for q, col in enumerate(cols):
                inc = 0
                for c, x in enumerate(col):
                    tot = sum(m for _, m in x) + inc
                    top = c == ncols - 1
                    ents = []
                    if plan_r is not None:         # the planned groups (a lone item <= 6 passes unchanged)
                        for g in plan_r[ri][c]:
                            gi = [x[k] for k in g]
                            sm = sum(m for _, m in gi)
                            if len(gi) == 1 and sm <= 6:
                                ents.append(("pass", gi[0][0], gi[0][1]))
                            elif top:
                                lst = gtop7 if sm <= 7 else gtop15
                                ents.append(("top7" if sm <= 7 else "top15", len(lst), gi))
                                lst.append(gi)
                            elif sm <= 7:
                                ents.append(("g8", len(g8), gi))
                                g8.append(gi)
                            else:
                                ents.append(("g16", len(g16), gi))
                                g16.append(gi)
                    elif tot <= target:
                        ents = [("pass", r, m) for r, m in x]
                    else:
                        gs = []
                        for r, m in sorted(x, key=lambda v: -v[1]):
                            for g in gs:
                                if sum(v for _, v in g) + m <= 15:
                                    g.append((r, m))
                                    break
                            else:
                                gs.append([(r, m)])
                        for g in gs:
                            sm = sum(m for _, m in g)
                            if len(g) == 1 and sm <= 3:
                                ents.append(("pass", g[0][0], g[0][1]))
                            elif top:
                                lst = gtop7 if sm <= 7 else gtop15
                                ents.append(("top7" if sm <= 7 else "top15", len(lst), g))
                                lst.append(g)
                            elif sm <= 7:
                                ents.append(("g8", len(g8), g))
                                g8.append(g)
                            else:
                                ents.append(("g16", len(g16), g))
                                g16.append(g)
                    inc = sum((1 if e[0] == "g8" and sum(m for _, m in e[2]) >= 4 else
                               sum(m for _, m in e[2]) // 4 if e[0] == "g16" else 0) for e in ents)
                    plan.append((q, c, ents))
            outs = {}
            if self.multi:
                # one launch: every group's sum at 1/8 (the 16-bin input), each table a row (pbsm)
                allg = g8 + g16 + gtop7 + gtop15
                S = self._gsum(ITEMS, [[r for r, _ in g] for g in allg], ca=1, c=-0.9375) if allg else None
                n8, n16, n7 = len(g8), len(g16), len(gtop7)
                jobs, keys = [], []
                for kind, lo, n, tabs in (("g8", 0, n8, ("md15", "dv15")), ("g16", n8, n16, ("md15", "dv15")),
                                          ("top7", n8 + n16, n7, ("md15",)), ("top15", n8 + n16 + n7, len(gtop15), ("md15",))):
                    for t in tabs:
                        if n:
                            jobs.append((S, list(range(lo, lo + n)), t))
                            keys.append(kind)
                if pend:                             # the late products ride this launch
                    PIN = self._rows([x[2] for x in pend])
                    for t in sorted({x[3] for x in pend}):
                        jobs.append((PIN, [k for k, x in enumerate(pend) if x[3] == t], t))
                        keys.append("pend:" + t)
                if jobs:
                    OUT, offs = self._multi(jobs)
                    for kind, off in zip(keys, offs):
                        outs.setdefault(kind, []).append(("rows", OUT, off))
            else:
                if g8:
                    S = self._gsum(ITEMS, [[r for r, _ in g] for g in g8], ca=2, c=-0.875)
                    outs["g8"] = self._pbsk(S, len(g8), "mdcy7")                 # [mod, carry]
                if g16:
                    S = self._gsum(ITEMS, [[r for r, _ in g] for g in g16], ca=1, c=-0.9375)
                    outs["g16"] = [self._pbs(S, len(g16), "md15"), self._pbs(S, len(g16), "dv15")]
                if gtop7:
                    S = self._gsum(ITEMS, [[r for r, _ in g] for g in gtop7], ca=2, c=-0.875)
                    outs["top7"] = [self._pbs(S, len(gtop7), "md7")]
                if gtop15:
                    S = self._gsum(ITEMS, [[r for r, _ in g] for g in gtop15], ca=1, c=-0.9375)
                    outs["top15"] = [self._pbs(S, len(gtop15), "md15")]
            ref = lambda o, gi: (o[1], o[2] + gi) if isinstance(o, tuple) else (o, gi)
            # the next ITEMS: one source per row (buffer, row)
            srcs = []
            ncol = [[[] for _ in range(ncols)] for _ in cols]
            for q, c, ents in plan:
                for e in ents:
                    if e[0] == "pass":
                        ncol[q][c].append((len(srcs), e[2]))
                        srcs.append((ITEMS, e[1]))
                        continue
                    kind, gi, g = e
                    sm = sum(m for _, m in g)
                    ncol[q][c].append((len(srcs), min(3, sm)))
                    srcs.append(ref(outs[kind][0], gi))
                    if kind in ("g8", "g16") and sm >= 4 and c + 1 < ncols:
                        ncol[q][c + 1].append((len(srcs), 1 if kind == "g8" else sm // 4))
                        srcs.append(ref(outs[kind][1], gi))
            if pend:
                for t in sorted({x[3] for x in pend}):
                    o = outs["pend:" + t][0]
                    for j, k in enumerate([k for k, x in enumerate(pend) if x[3] == t]):
                        q, c, _, _, mx = pend[k]
                        ncol[q][c].append((len(srcs), mx))
                        srcs.append((o[1], o[2] + j))
                pend = []
            ITEMS, cols = self._gather(srcs), ncol
            ri += 1
        return ITEMS, cols

    def _gather(self, srcs):
        """A buffer whose row r is row srcs[r][1] of buffer srcs[r][0] (lins of two sources at a time)."""
        bufs = list(dict.fromkeys(b for b, _ in srcs))
        N = None
        for i in range(0, len(bufs), 2):
            ba = bufs[i]
            bb = bufs[i + 1] if i + 1 < len(bufs) else ba
            pairs = [(r if b == ba else -1, r if (b == bb and bb != ba) else -1) for b, r in srcs]
            if N is None:
                N = self._lin(ba, bb, pairs, ca=1, cb=1)
            else:
                self._lin(N, ba, [(k, p[0]) for k, p in enumerate(pairs)], ca=1, cb=1, dst=N)
                if bb != ba:
                    self._lin(N, bb, [(k, p[1]) for k, p in enumerate(pairs)], ca=1, cb=1, dst=N)
        return N

    def _final_lin(self, ITEMS, cols, nout, out_slots):
        """Columns summing to <= 6 added by the prefix adder; the output digits linear (s + c_(i - 1) - 4 c_i)."""
        w = self.w
        save = (self.nd, self.R)
        self.nd, self.R = nout, w * nout
        R = self.R
        g = [[r for r, _ in col[c]] or [-1] for col in cols for c in range(nout)]
        if all(sum(m for _, m in x) <= 3 for col in cols for x in col):
            O = self._gsum(ITEMS, g, ca=2, c=0.0)
            self._store(O, R, "dig4_w", out_slots)
            self.nd, self.R = save
            return O
        X = self._gsum(ITEMS, g, ca=2, c=-0.75)                                            # (s - 3) / 4
        C = self._carries(X)
        O = self._digits_lin(X, C)
        self._store(O, R, "dig4_w", out_slots)
        self.nd, self.R = save
        return O

    def _digits_lin(self, X, C):
        """Digits at d / 4: (s + c_(i - 1) - 4 c_i) from X = (s - 3) / 4 and the carries C ({0, 1/8})."""
        R, nd = self.R, self.nd
        prev = [(r, r - 1 if r % nd else -1) for r in range(R)]
        O = self._lin(X, C, prev, ca=1, cb=2, c=0.75)
        self._lin(O, C, [(r, r) for r in range(R)], ca=1, cb=-8, dst=O)
        return O

    def _split_products(self, wide):
        """Whether mul2 puts the products past one batch wave into the tree's first launch: forced by
        `split_products` "1" / "0", else (auto) the form `est_ms` finds faster for this word count and width."""
        sp = self.split_products
        if sp in (True, False) or str(sp) in ("0", "1"):
            return sp is True or str(sp) == "1"
        t = {}
        for f in ("1", "0"):
            x = WordProg("t", self.w, self.nd)
            x.multi, x.fan4, x.tree_dp, x.split_products = self.multi, self.fan4, self.tree_dp, f
            x.mul2("t.o", wide=wide)
            t[f] = est_ms(x.pr)
        return t["1"] < t["0"]

    def mul2(self, out_slots, wide=False, high=None):
        """a b (mod 2^bits; wide: all 2 bits bits): digit products (a b) mod 4 / div 4 by one 16-bin lookup each on
        a_i + 4 b_j, a 16-bin carry-save tree (`_reduce16`), the prefix adder with linear output digits. high:
        the upper word of the double product, "u" unsigned, "su" signed a x unsigned b, "ss" signed x signed: the
        signed terms - s_a b 2^bits (- s_b a 2^bits) enter the tree as s_a ? ~b_i : 0 items (one 16-bin lookup on
        b_i + 4 a_top, in the products' launch) and the +1 bits s_a (s_b) at column bits / 2."""
        nd, w, R = self.nd, self.w, self.R
        if high:
            wide = True
        nc = 2 * nd if wide else nd
        prs = [(q, i, j) for q in range(w) for i in range(nd) for j in range(nd) if i + j < nc]
        prs.sort(key=lambda t: t[1] + t[2] + 1 >= nc)                     # pairs with a high part first
        nhi = sum(1 for q, i, j in prs if i + j + 1 < nc)
        IN = self._lin(0, 0, [(self._row(0, q, i), self._row(1, q, j)) for q, i, j in prs], ca=1, cb=4, c=-0.9375)
        corr = []                                   # (word, column, x row source pair, table)
        if high in ("su", "ss"):
            corr += [(q, nd + i, (self._row(1, q, i), self._row(0, q, nd - 1)), "sgnot") for q in range(w) for i in range(nd)]
            corr += [(q, nd, (-1, self._row(0, q, nd - 1)), "sbit") for q in range(w)]
        if high == "ss":
            corr += [(q, nd + i, (self._row(0, q, i), self._row(1, q, nd - 1)), "sgnot") for q in range(w) for i in range(nd)]
            corr += [(q, nd, (-1, self._row(1, q, nd - 1)), "sbit") for q in range(w)]
        if corr:
            CI = self._lin(0, 0, [c[2] for c in corr], ca=1, cb=4, c=-0.9375)
        # one batch-path wave holds BATCH_WAVE LWEs (one per block): the products beyond it may ride the tree's first
        # launch instead (round robin over the columns, high parts first), when the cost model says that is faster
        # than a two-LWE-per-block wave of all of them
        late = []
        tot = len(prs) + nhi + len(corr)
        if self.multi and nhi and not corr and BATCH_WAVE < tot <= 2 * BATCH_WAVE and self._split_products(wide):
            allp = [(k, "plo") for k in range(len(prs))] + [(k, "phi") for k in range(nhi)]
            col = lambda x: prs[x[0]][1] + prs[x[0]][2] + (x[1] == "phi")
            rank, cnt = {}, {}
            for x in sorted(allp, key=lambda x: (x[1] != "phi", col(x))):
                cnt[col(x)] = cnt.get(col(x), 0) + 1
                rank[x] = cnt[col(x)]
            allp.sort(key=lambda x: (rank[x], x[1] != "phi", col(x)))
            late = allp[:tot - BATCH_WAVE]
        lateset = set(late)
        if self.multi and nhi:
            jobs = [(IN, [k for k in range(len(prs)) if (k, "plo") not in lateset], "plo"),
                    (IN, [k for k in range(nhi) if (k, "phi") not in lateset], "phi")]
            if corr:
                jobs += [(CI, [k for k, c in enumerate(corr) if c[3] == t], t) for t in ("sgnot", "sbit")]
            P, offs = self._multi(jobs)
            pos = {}
            for (b_, rows_, t_), o_ in zip(jobs, offs):
                if t_ in ("plo", "phi"):
                    for jj, k in enumerate(rows_):
                        pos[(k, t_)] = o_ + jj
            LO, HI = (P, None), (P, None)
        else:
            LO = (self._pbs(IN, len(prs), "plo"), 0)
            HI = (self._pbs(IN, nhi, "phi"), 0) if nhi else None
            if corr:
                raise ValueError("mul2: signed high words need the one-launch rounds (PC_TFHE_MULTI)")
        cols = [[[] for _ in range(nc)] for _ in range(w)]
        srcs = []
        if corr:
            for t, o in zip(("sgnot", "sbit"), offs[2:]):
                ks = [k for k, c in enumerate(corr) if c[3] == t]
                for j, k in enumerate(ks):
                    q, col = corr[k][0], corr[k][1]
                    cols[q][col].append((len(srcs), 3 if t == "sgnot" else 1))
                    srcs.append((P, o + j))
        for k, (q, i, j) in enumerate(prs):
            if (k, "plo") in lateset:
                continue
            cols[q][i + j].append((len(srcs), 3))
            srcs.append((LO[0], pos[(k, "plo")] if LO[1] is None else LO[1] + k))
        for k, (q, i, j) in enumerate(prs[:nhi]):
            if (k, "phi") in lateset:
                continue
            cols[q][i + j + 1].append((len(srcs), 2))
            srcs.append((HI[0], pos[(k, "phi")] if HI[1] is None else HI[1] + k))
        pend = [(prs[k][0], prs[k][1] + prs[k][2] + (t == "phi"), (IN, k), t, 3 if t == "plo" else 2) for k, t in late]
        ITEMS = self._gather(srcs)
        ITEMS, cols = self._reduce16(ITEMS, cols, nc, pending=pend)
        if not high:
            return self._final_lin(ITEMS, cols, nc, out_slots)
        O = self._final_lin(ITEMS, cols, nc, None)
        self.pr.steps.pop()
        self._store(self._lin(O, O, [(q * nc + nd + i, -1) for q in range(w) for i in range(nd)], ca=1, cb=0), w * nd,
                    "dig4_w", out_slots)

    def mul(self, out_slots, wide=False):
        """a b mod 2^bits (wide: the 2 bits-bit product, 2 nd digits per word)."""
        nd, w, R = self.nd, self.w, self.R
        nc = 2 * nd if wide else nd
        # round 1: lo / hi of a's and b's digits (a << 1 and b's bits)
        LH = self._lh(list(range(2 * R)))
        lo = lambda r: r
        hi = lambda r: 2 * R + r
        # a << 1 (nd + 1 digits when wide): 2 lo(a_i) + hi(a_(i - 1)); AA = [a | a << 1], a's rows q nd + i,
        # a << 1's rows R + q (nd + 1) + i
        n1 = nd + 1
        sh = [(lo(q * nd + i) if i < nd else -1, hi(q * nd + i - 1) if i > 0 else -1) for q in range(w) for i in range(n1)]
        A1 = self._lin(LH, LH, sh, ca=2, cb=1)
        AA = self._lin(0, A1, [(r, -1) for r in range(R)] + [(-1, r) for r in range(w * n1)], ca=1, cb=1)
        xr, br, cols = [], [], [[[] for _ in range(nc)] for _ in range(w)]
        for q in range(w):
            for j in range(4 * nd // 2):         # bit j of b
                bj = (lo if j % 2 == 0 else hi)(R + q * nd + j // 2)
                for i in range(nd + (j % 2)):
                    c = i + j // 2
                    if c >= nc or (j % 2 == 0 and i >= nd):
                        continue
                    cols[q][c].append((len(xr), 3))
                    xr.append(q * nd + i if j % 2 == 0 else R + q * n1 + i)
                    br.append(bj)
        IN = self._lin(AA, LH, list(zip(xr, br)), ca=2, cb=8, c=-0.875)
        PP = self._pbs(IN, len(xr), "and")
        ITEMS, cols = self._reduce(PP, cols, nc)
        self._final_add(ITEMS, cols, nc, out_slots)

    def mulc(self, out_slots, c):
        """a c mod 2^bits for a public constant c (shifted copies of a and a << 1, the same tree)."""
        nd, w, R = self.nd, self.w, self.R
        c = int(c) % (1 << (2 * nd))
        n1 = nd + 1
        odd = any((c >> j) & 1 for j in range(1, 2 * nd, 2))
        if odd:
            LH = self._lh(list(range(R)))
            sh = [(q * nd + i if i < nd else -1, R + q * nd + i - 1 if i > 0 else -1) for q in range(w) for i in range(n1)]
            A1 = self._lin(LH, LH, sh, ca=2, cb=1)
            AA = self._lin(0, A1, [(r, -1) for r in range(R)] + [(-1, r) for r in range(w * n1)], ca=1, cb=1)
        else:
            AA = 0
        cols = [[[] for _ in range(nd)] for _ in range(w)]
        for q in range(w):
            for j in range(2 * nd):
                if not (c >> j) & 1:
                    continue
                for i in range(nd):
                    cc = i + j // 2
                    if cc < nd:
                        cols[q][cc].append((q * nd + i if j % 2 == 0 else R + q * n1 + i, 3))
        if not any(cols[q][x] for q in range(w) for x in range(nd)):
            self._dig((0, 0), [(-1, -1)] * R, out_slots=out_slots)
            return
        ITEMS, cols = self._reduce(AA, cols, nd)
        self._final_add(ITEMS, cols, nd, out_slots)

    def popcount(self, out_slots):
        """The number of set bits of a, as a word."""
        nd, w, R = self.nd, self.w, self.R
        IN = self._lin(0, 0, [(r, -1) for r in range(R)], ca=2, cb=0)
        PC = self._pbs(IN, R, "pc")
        cols = [[[(q * nd + i, 2) for i in range(nd)]] + [[] for _ in range(nd - 1)] for q in range(w)]
        ITEMS, cols = self._reduce(PC, cols, nd)
        self._final_add(ITEMS, cols, nd, out_slots)

    def popcount2(self, out_slots):
        """The number of set bits of a: popcount(d_i) + popcount(d_(i+1)) of digit pairs in one 16-bin lookup
        (items <= 4), then the 16-bin carry-save tree and the prefix adder (`mul2`'s)."""
        nd, w, R = self.nd, self.w, self.R
        nout = min(nd, ((2 * nd).bit_length() + 1) // 2)
        prs = [(q, i) for q in range(w) for i in range(0, nd, 2)]
        IN = self._lin(0, 0, [(self._row(0, q, i), self._row(0, q, i + 1) if i + 1 < nd else -1) for q, i in prs],
                       ca=1, cb=4, c=-0.9375)
        PC = self._pbs(IN, len(prs), "pc2")
        # groups of up to 4 pair counts (sums <= 16) -> s mod 4, (s div 4) mod 4, [s == 16] in one round (the whole
        # torus: 16 is the only point past [-1, 1) and all three tables are negacyclic-consistent there)
        npw = len(prs) // w
        gs = [[q * npw + k for k in range(g, min(g + 4, npw))] for q in range(w) for g in range(0, npw, 4)]
        S = self._gsum(PC, gs, ca=1, c=-0.9375)
        big = any(4 * len(g) >= 16 for g in gs)
        ng = len(gs)
        OUT, offs = self._multi([(S, list(range(ng)), t) for t in (("md16", "dv16", "eq16") if big else ("md16", "dv16"))])
        M = self._lin(OUT, OUT, [(offs[0] + g, -1) for g in range(ng)], ca=1, cb=0)
        Dv = self._lin(OUT, OUT, [(offs[1] + g, -1) for g in range(ng)], ca=1, cb=0)
        if big:
            E = self._lin(OUT, OUT, [(offs[2] + g, -1) for g in range(ng)], ca=1, cb=0, c=0.0625)
        gpw = len(gs) // w
        cols = [[[] for _ in range(nout)] for _ in range(w)]
        srcs = []
        for q in range(w):
            for k in range(gpw):
                g = q * gpw + k
                mx = 4 * len(gs[g])
                cols[q][0].append((len(srcs), min(3, mx)))
                srcs.append((M, g))
                if mx >= 4 and nout > 1:
                    cols[q][1].append((len(srcs), min(3, mx // 4)))
                    srcs.append((Dv, g))
                if mx >= 16 and nout > 2:
                    cols[q][2].append((len(srcs), 1))
                    srcs.append((E, g))
        ITEMS, cols = self._reduce16(self._gather(srcs), cols, nout)
        O = self._final_lin(ITEMS, cols, nout, None)
        # the upper digits (zero) as a lin of no terms
        self.pr.steps.pop()                                           # _final_lin's store
        OUT = self._lin(O, O, [(q * nout + i if i < nout else -1, -1) for q in range(w) for i in range(nd)], ca=1, cb=0)
        self._store(OUT, R, "dig4_w", out_slots)

    def _thermo(self, kind, op=0, src=None):
        """The thermometer bits T_j = [n >= j], j = 1 .. bits, of n = clz(a) (kind "clz") or ctz(a) ("ctz"), at
        value 1/4: zero tests ([sum == 0] on the whole torus, sums of <= 16 indicator items) of the digits above /
        below each bit boundary, the boundary digit's top / low bit as one more item. Two rounds. Returns the
        buffer, rows [word][j - 1]."""
        nd, w, R, bits = self.nd, self.w, self.R, 2 * self.nd
        sb, base = (0, op * R) if src is None else (src, 0)          # the digits: buffer sb, rows base + [word][digit]
        rowd = lambda q, i: base + q * nd + i
        X = self._lin(sb, sb, [(base + r, -1) for r in range(R)], ca=2, cb=0, c=-0.375)
        U, H = self._pbsk(X, R, "uh" if kind == "clz" else "ul")       # [d != 0], [d >= 2] (clz) / [d & 1] (ctz)
        srcs, idx = [], {}

        def item(key, buf, row):
            if key not in idx:
                idx[key] = len(srcs)
                srcs.append((buf, row))
            return idx[key]
        pair_ok = nd > 16
        if pair_ok:                                                      # [d_2p | d_(2p+1) != 0]
            PIN = self._lin(sb, sb, [(rowd(q, 2 * p), rowd(q, 2 * p + 1)) for q in range(w) for p in range(nd // 2)],
                            ca=1, cb=4, c=-0.9375)
            P = self._pbs(PIN, w * nd // 2, "nzp")
            # the boundary items of 17-item thresholds: [d_1 != 0 or d_0 >= 2] (clz), [d_(nd-2) != 0 or d_(nd-1) & 1] (ctz)
            if kind == "clz":
                BIN = self._lin(sb, sb, [(rowd(q, 0), rowd(q, 1)) for q in range(w)], ca=1, cb=4, c=-0.9375)
            else:
                BIN = self._lin(sb, sb, [(rowd(q, nd - 1), rowd(q, nd - 2)) for q in range(w)], ca=1, cb=4, c=-0.9375)
            B = self._pbs(BIN, w, "nzb" if kind == "clz" else "nzc")
        groups = []
        for q in range(w):
            for j in range(1, bits + 1):
                if kind == "clz":                     # a < 2^e, e = bits - j: digits > e // 2 zero (and >= e / 2 for e even)
                    e = bits - j
                    lo_d = e // 2 if e % 2 == 0 else e // 2 + 1
                    digs = list(range(lo_d, nd))
                    extra = [("h", e // 2)] if e % 2 else []
                else:                                 # a mod 2^j == 0: digits < j // 2 zero (+ the low bit of j // 2)
                    digs = list(range(0, j // 2))
                    extra = [("h", j // 2)] if j % 2 else []
                its = []
                if pair_ok and len(digs) + len(extra) > 16:
                    special = (kind == "clz" and extra and extra[0][1] == 0) or (kind == "ctz" and extra and extra[0][1] == nd - 1)
                    if special:                       # the 17-item cases: the bivariate boundary item
                        its.append(item(("b", q), B, q))
                        digs = digs[1:] if kind == "clz" else digs[:-1]
                        extra = []
                    k = 0
                    while k < len(digs):
                        d0 = digs[k]
                        if d0 % 2 == 0 and k + 1 < len(digs) and digs[k + 1] == d0 + 1:
                            its.append(item(("p", q, d0 // 2), P, q * (nd // 2) + d0 // 2))
                            k += 2
                        else:
                            its.append(item(("u", q, d0), U, q * nd + d0))
                            k += 1
                else:
                    its += [item(("u", q, d), U, q * nd + d) for d in digs]
                its += [item(("h", q, x[1]), H, q * nd + x[1]) for x in extra]
                assert len(its) <= 16, (kind, j, len(its))
                groups.append(its or [-1])
        ITEMS = self._gather(srcs)
        S = self._gsum(ITEMS, groups, ca=1, c=-0.9375)
        Z = self._pbs(S, len(groups), "eqz_4")
        return self._lin(Z, Z, [(g, -1) for g in range(len(groups))], ca=1, cb=0, c=0.125)

    def count_op(self, out_slots, kind):
        """clz / ctz / log2 (= bits - 1 - clz; all ones at 0): the thermometer bits, then every output digit is
        linear in them (digit_m(g(n)) = digit_m(g(0)) + sum_j T_j (digit_m(g(j)) - digit_m(g(j - 1)))): 2 rounds."""
        nd, w, bits = self.nd, self.w, 2 * self.nd
        T = self._thermo("ctz" if kind == "ctz" else "clz")
        g = (lambda n: n) if kind in ("clz", "ctz") else (lambda n: (bits - 1 - n) % (1 << bits))
        dig = lambda v, m: (v >> (2 * m)) & 3
        # build directly: for every output row, the list of (T row, coefficient)
        terms = [[] for _ in range(w * nd)]
        consts = [0.0] * (w * nd)
        for q in range(w):
            for m in range(nd):
                consts[q * nd + m] = dig(g(0), m) / 4.0
                for j in range(1, bits + 1):
                    c = dig(g(j), m) - dig(g(j - 1), m)
                    if c:
                        terms[q * nd + m].append((q * bits + j - 1, c))
        O = self._lin_terms(T, terms, consts)
        self._store(O, w * nd, "dig4_w", out_slots)

    def _lin_terms(self, src, terms, consts):
        """Rows r = consts[r] + sum c src[row] over terms[r] = [(row, c)] (integer c): one lin per (term layer,
        coefficient value), each adding one term to every row that has it."""
        n = len(terms)
        O = self._lin(src, src, [(-1, -1)] * n, ca=0, cb=0)
        depth = max((len(t) for t in terms), default=0)
        for k in range(depth):
            lay = [t[k] if k < len(t) else None for t in terms]
            for cval in sorted(set(x[1] for x in lay if x is not None)):
                self._lin(O, src, [(r, x[0] if (x is not None and x[1] == cval) else -1) for r, x in enumerate(lay)],
                          ca=1, cb=cval, dst=O)
        for cv in sorted(set(consts)):
            if cv:
                K = self._lin(O, O, [(-1, -1)], ca=0, cb=0, c=cv)
                self._lin(O, K, [(r, 0 if consts[r] == cv else -1) for r in range(n)], ca=1, cb=1, dst=O)
        return O

    def _prefix_rows(self, S, words, W, last_tabs):
        """Kogge-Stone fan-in 3 over W positions per word on statuses S (rows [word][position], -1 / 0 / 1 at 1/4);
        the last round reads position p's combine through every table of last_tabs(p) (one launch). Returns (the
        last round's buffer, row index (word, position, table number) -> row)."""
        cur, d = S, 1
        while True:
            f = min(3, -(-W // d))
            last = d * f >= W
            U = self._lin(cur, cur, [(q * W + p, q * W + p - d if p >= d else -1) for q in range(words) for p in range(W)],
                          ca=1 << (f - 1), cb=1 << max(f - 2, 0))
            if f == 3:
                self._lin(U, cur, [(q * W + p, q * W + p - 2 * d if p >= 2 * d else -1) for q in range(words) for p in range(W)],
                          ca=1, cb=1, dst=U)
            if last:
                jobs, where = [], {}
                for t in sorted({t for p in range(W) for t in last_tabs(p)}):
                    rows = [q * W + p for q in range(words) for p in range(W) if t in last_tabs(p)]
                    jobs.append((U, rows, t))
                OUT, offs = self._multi(jobs)
                for (_, rows, t), o in zip(jobs, offs):
                    for k, r in enumerate(rows):
                        where[(r // W, r % W, t)] = o + k
                return OUT, where
            cur = self._pbs(U, words * W, "sgn")
            d *= f

    def _rows(self, srcs):
        """A buffer of rows: srcs[r] = (buffer, row), None (0), ("c", value) (a noiseless constant) or ("sum",
        (buffer, row), (buffer, row))."""
        if any(x is not None and x[0] == "c" for x in srcs):
            consts = [x[1] if x is not None and x[0] == "c" else 0.0 for x in srcs]
            rest = [None if x is not None and x[0] == "c" else x for x in srcs]
            if all(x is None for x in rest):
                return self._crow(consts)
            g = self._rows(rest)
            K = self._crow(consts)
            self._lin(g, K, [(r, r) for r in range(len(srcs))], ca=1, cb=1, dst=g)
            return g
        first = [x[1] if x is not None and x[0] == "sum" else x for x in srcs]
        g = self._gather([x if x is not None else (first[0] if first[0] is not None else (0, -1)) for x in first])
        # rows that are None: zero them; "sum" rows: add the second term
        if any(x is None for x in srcs):
            self._lin(g, g, [(r if srcs[r] is not None else -1, -1) for r in range(len(srcs))], ca=1, cb=0, dst=g)
        sums = [(r, x[2]) for r, x in enumerate(srcs) if x is not None and x[0] == "sum"]
        for b in dict.fromkeys(y[0] for _, y in sums):
            self._lin(g, b, [(r, next((y[1] for rr, y in sums if rr == r and y[0] == b), -1)) for r in range(len(srcs))],
                      ca=1, cb=1, dst=g)
        return g

    def divrem(self, out_slots, want="div", A=None, B=None, fin=None, cdiv=None):
        """Unsigned a / b and a mod b (want div / rem / divrem: q's digits then r's), radix-2 restoring division
        without shifting the remainder: for k = bits - 1 .. 0, [R >= b 2^k] is the no-borrow of ~R + (b << k) on
        the digits m = k // 2 .. top (the lower digits of b << k are 0, so no carry comes from below) plus a virtual
        top position whose status is 'generate' when b 2^k overflows the word (from b's clz thermometer bits); the
        fan-in 3 prefix runs on those w + 1 positions, its top position read as the selector (three forms in one
        launch); then one round selects R' = sel ? D : R (AND muxes; D = R - (b << k), linear in the carries) and,
        in the same launch, the next step's statuses for both candidates gated by the selector (16-bin tables on
        b' - D + 7 sel): 1 + ceil(log3 (w + 1)) rounds per quotient bit. b = 0: every step subtracts 0, q = all
        ones and r = a (the common convention)."""
        nd, w, bits = self.nd, self.w, 2 * self.nd
        # operands: buffers with rows [word][digit] (default: the loaded a, b)
        ar = (lambda q, i: (0, self._row(0, q, i))) if A is None else (lambda q, i: (A, q * nd + i))
        br = (lambda q, i: (0, self._row(1, q, i))) if B is None else (lambda q, i: (B, q * nd + i))
        if cdiv is None:
            Bb0 = br(0, 0)[0]
            sh_in = self._lin(Bb0, Bb0, [(br(q, j)[1], br(q, j - 1)[1] if j else -1) for q in range(w) for j in range(nd)],
                              ca=1, cb=4, c=-0.9375)
            B1 = self._pbs(sh_in, w * nd, "shl1d")                         # (b << 1)_j / 8
            T = self._thermo("clz", op=1) if B is None else self._thermo("clz", src=B)   # [b < 2^(bits - j)] / 4
            kmax = bits - 1
        else:                                       # a public divisor: b << k's digits are constants, steps with
            cdiv = int(cdiv)                        # b 2^k >= 2^bits skipped (q_k = 0)
            kmax = bits - cdiv.bit_length()

        def bsrc(q, k, i):                                                  # digit i of b << k (None: 0)
            if cdiv is not None:
                v = ((cdiv << k) >> (2 * i)) & 3
                return ("c", v / 8.0) if v else None
            j = i - k // 2
            if j < 0:
                return None
            return br(q, j) if k % 2 == 0 else (B1, q * nd + j)
        Rs = [[ar(q, i) for i in range(nd)] for q in range(w)]
        clean_r = True
        # the quotient bits (+-1/16), [word][bit]; skipped steps' bits -1/16 (0)
        QB = self._crow([-0.0625 if (r % bits) > kmax else 0.0 for r in range(w * bits)])
        qrow = [[(QB, q * bits + kk) for kk in range(bits)] for q in range(w)]
        S = None
        for k in range(kmax, -1, -1):
            m = k // 2
            nw, W = nd - m, nd - m + 1
            n = w * nw
            if S is None:
                X = self._lin(self._rows([bsrc(q, k, i) for q in range(w) for i in range(m, nd)]),
                              self._rows([Rs[q][i] for q in range(w) for i in range(m, nd)]),
                              [(r, r) for r in range(n)], ca=2, cb=-2)
                S0 = self._pbs(X, n, "sgn")
                S = [(S0, r) for r in range(n)]
            # statuses [word][position]: digits, then the virtual top 1/4 - T_k (k = 0: never big)
            STV = self._rows([S[q * nw + p] if p < nw else ((T, q * bits + k - 1) if k and cdiv is None else None)
                              for q in range(w) for p in range(W)])
            if k and cdiv is None:
                self._lin(STV, STV, [(q * W + p, -1) for q in range(w) for p in range(W)], ca=1, cb=0, dst=STV)
                neg = self._lin(STV, STV, [(q * W + p if p < nw else -1, q * W + p if p == nw else -1) for q in range(w) for p in range(W)],
                                ca=1, cb=-1, c=0.0)
                K4 = self._lin(neg, neg, [(-1, -1)], ca=0, cb=0, c=0.25)
                self._lin(neg, K4, [(q * W + p, 0 if p == nw else -1) for q in range(w) for p in range(W)], ca=1, cb=1, dst=neg)
                STV = neg
            else:
                K4 = self._lin(STV, STV, [(-1, -1)], ca=0, cb=0, c=0.25)
                self._lin(STV, K4, [(q * W + p, -1) for q in range(w) for p in range(W)], ca=1, cb=0, dst=STV)
            tops = ("nc_pm1", "nc_pm7", "nc_pm8") if k else ("nc_pm1", "nc_pm8")
            C, where = self._prefix_rows(STV, w, W, lambda p: ("carry_pm",) if p < nw else tops)
            cr = lambda q, i: (C, where[(q, i - m, "carry_pm")]) if i >= m else None
            SEL = self._lin(C, C, [(where[(q, nw, "nc_pm1")], -1) for q in range(w)], ca=1, cb=0, c=0.5)       # unit 1
            self._lin(QB, C, [(r, where[(r // bits, nw, "nc_pm8")] if r % bits == k else -1) for r in range(w * bits)],
                      ca=1, cb=1, dst=QB)                                                                     # +-1/16
            # D_i = R_i - B_i - c_(i-1) + 4 c_i (c = carry_pm + 1/16): constant (4 - [i > m]) / 16
            Rb = self._rows([Rs[q][i] for q in range(w) for i in range(m, nd)])
            Bb = self._rows([bsrc(q, k, i) for q in range(w) for i in range(m, nd)])
            Cm = self._rows([cr(q, i - 1) if i > m else None for q in range(w) for i in range(m, nd)])
            Cc = self._rows([cr(q, i) for q in range(w) for i in range(m, nd)])
            D = self._lin(Rb, Bb, [(r, r) for r in range(n)], ca=1, cb=-1, c=0.25)
            self._lin(D, Cm, [(r, r) for r in range(n)], ca=1, cb=-1, dst=D)
            self._lin(D, Cc, [(r, r) for r in range(n)], ca=1, cb=4, dst=D)
            K = self._lin(D, D, [(-1, -1)], ca=0, cb=0, c=-0.0625)
            self._lin(D, K, [(r, 0 if r % nw else -1) for r in range(n)], ca=1, cb=1, dst=D)
            # the selection round: R' (AND muxes; or, every other step, R + (sel ? D - R : 0) from one gated lookup
            # per digit: the next step then muxes clean digits again), the next step's gated statuses
            skip_r = k == 0 and want == "div" and fin is None
            gform = k > 0 and clean_r and self.div_gform
            jobs, names = [], []
            if k or gform:
                S7 = self._lin(C, C, [(where[(q, nw, "nc_pm7")], -1) for q in range(w)], ca=1, cb=0, c=0.4375)  # 7 sel / 8
            if not skip_r:
                if gform:                                   # (D - R + 3) / 8 - 15 / 16 + 7 sel / 8
                    GI = self._lin(D, Rb, [(r, r) for r in range(n)], ca=1, cb=-1, c=0.375 - 0.9375)
                    self._lin(GI, S7, [(r, r // nw) for r in range(n)], ca=1, cb=1, dst=GI)
                    jobs.append((GI, list(range(n)), "gdel"))
                    names.append("g")
                else:
                    NSEL = self._lin(SEL, SEL, [(q, -1) for q in range(w)], ca=-1, cb=0, c=1.0)
                    IN1 = self._lin(D, SEL, [(r, r // nw) for r in range(n)], ca=2, cb=1, c=-0.875)
                    IN0 = self._lin(Rb, NSEL, [(r, r // nw) for r in range(n)], ca=2, cb=1, c=-0.875)
                    jobs += [(IN1, list(range(n)), "and"), (IN0, list(range(n)), "and")]
                    names += ["a1", "a0"]
            if k:
                k2 = k - 1
                m2 = k2 // 2
                N7 = self._lin(S7, S7, [(q, -1) for q in range(w)], ca=-1, cb=0, c=0.875)
                Bn = self._rows([bsrc(q, k2, i) for q in range(w) for i in range(m, nd)])
                # (b' - x + 3) / 8 - 15 / 16 + 7 sel / 8
                GD = self._lin(Bn, D, [(r, r) for r in range(n)], ca=1, cb=-1, c=0.375 - 0.9375)
                self._lin(GD, S7, [(r, r // nw) for r in range(n)], ca=1, cb=1, dst=GD)
                GR = self._lin(Bn, Rb, [(r, r) for r in range(n)], ca=1, cb=-1, c=0.375 - 0.9375)
                self._lin(GR, N7, [(r, r // nw) for r in range(n)], ca=1, cb=1, dst=GR)
                jobs += [(GD, list(range(n)), "gsel"), (GR, list(range(n)), "gsel")]
                names += ["gD", "gR"]
                if m2 < m:                                  # digit m - 1: untouched, a plain status
                    XS = self._lin(self._rows([bsrc(q, k2, m2) for q in range(w)]), self._rows([Rs[q][m2] for q in range(w)]),
                                   [(q, q) for q in range(w)], ca=2, cb=-2)
                    jobs.append((XS, list(range(w)), "sgn"))
                    names.append("pl")
            if jobs:
                OUT, offs = self._multi(jobs)
                off = dict(zip(names, offs))
            if not skip_r:
                for q in range(w):
                    for i in range(m, nd):
                        r = q * nw + i - m
                        Rs[q][i] = ("sum", Rs[q][i], (OUT, off["g"] + r)) if gform else ("sum", (OUT, off["a1"] + r), (OUT, off["a0"] + r))
                act = [(q, i) for q in range(w) for i in range(m, nd)]
                RB = self._rows([Rs[q][i] for q, i in act])                  # the active digits in one buffer
                for j, (q, i) in enumerate(act):
                    Rs[q][i] = (RB, j)
                clean_r = not gform
            if k:
                S = []
                for q in range(w):
                    if m2 < m:
                        S.append((OUT, off["pl"] + q))
                    S += [("sum", (OUT, off["gD"] + q * nw + i - m), (OUT, off["gR"] + q * nw + i - m)) for i in range(m, nd)]
                Sb = self._rows(S)
                S = [(Sb, r) for r in range(len(S))]
        # outputs at d / 4: q digit j = 2 (q_2j + 2 q_2j+1) / 8 from the +-1/16 rows (+ 1/16 each)
        if fin is not None:                                                 # q, r at d / 8 to the caller
            Q = self._rows([qrow[q][2 * j] for q in range(w) for j in range(nd)])
            Q1 = self._rows([qrow[q][2 * j + 1] for q in range(w) for j in range(nd)])
            QQ = self._lin(Q, Q1, [(r, r) for r in range(w * nd)], ca=1, cb=2, c=0.1875)
            return QQ, self._rows([Rs[q][i] for q in range(w) for i in range(nd)])
        outs = []
        if want in ("div", "divrem"):
            Q = self._rows([qrow[q][2 * j] for q in range(w) for j in range(nd)])
            Q1 = self._rows([qrow[q][2 * j + 1] for q in range(w) for j in range(nd)])
            QQ = self._lin(Q, Q1, [(r, r) for r in range(w * nd)], ca=2, cb=4, c=0.375)
            outs.append((QQ, w * nd))
        if want in ("rem", "divrem"):
            RR = self._rows([Rs[q][i] for q in range(w) for i in range(nd)])
            outs.append((self._lin(RR, RR, [(r, -1) for r in range(w * nd)], ca=2, cb=0), w * nd))
        if len(outs) == 1:
            return self._store(outs[0][0], outs[0][1], "dig4_w", out_slots)
        # per word: q's digits, then r's (the word q + r 2^bits)
        O = self._lin(outs[0][0], outs[1][0], [(q * nd + i, -1) if h == 0 else (-1, q * nd + i)
                                               for q in range(w) for h in range(2) for i in range(nd)], ca=1, cb=1)
        self._store(O, 2 * w * nd, "dig4_w", out_slots)

    def _cneg(self, digs, sel, sst, words):
        """Conditional negation of `words` words: digs[q][i] = (buffer, row) digits at d / 8, sel[q] the condition at
        unit 1, sst[q] its +-1/4 form (the carry-in's status). y = s ? ~x : x (and both statuses of y + 0: P when y
        = 3, else K) by one many-LUT lookup per digit, then (y + s) by the fan-in 3 prefix over the carry-in and the
        digits: 1 + ceil(log3 (nd + 1)) rounds. Returns the linear digits at d / 8 (rows [word][digit])."""
        nd = self.nd
        n = words * nd
        Xd = self._rows([digs[q][i] for q in range(words) for i in range(nd)])
        Sb = self._rows([sel[q] for q in range(words)])
        IN = self._lin(Xd, Sb, [(r, r // nd) for r in range(n)], ca=2, cb=1, c=-0.875)
        Y, YS = self._pbsk(IN, n, "cneg")                                   # y / 8, status(y) at 1/4
        W = nd + 1
        ST = self._rows([sst[q] if p == 0 else (YS, q * nd + p - 1) for q in range(words) for p in range(W)])
        C, where = self._prefix_rows(ST, words, W, lambda p: ("carry_pm",))
        # o_i = y_i + c_(p = i) - 4 c_(p = i + 1) (c = carry_pm + 1/16): constant 1/16 - 4/16
        Cl = self._rows([(C, where[(q, i, "carry_pm")]) for q in range(words) for i in range(nd)])
        Ch = self._rows([(C, where[(q, i + 1, "carry_pm")]) for q in range(words) for i in range(nd)])
        O = self._lin(Y, Cl, [(r, r) for r in range(n)], ca=1, cb=1, c=-0.1875)
        self._lin(O, Ch, [(r, r) for r in range(n)], ca=1, cb=-4, dst=O)
        return O

    def sdivrem(self, out_slots, want="sdiv"):
        """Signed (truncating) a / b, a rem b: the signs (one launch: s_a, s_b and s_a xor s_b, each as a unit-1
        selector and a +-1/4 status), |a| and |b| by conditional negation, a cleaning round, the unsigned division,
        then q negated when the signs differ and r when a < 0. MIN / -1 = MIN, r 0; b = 0: q = -1 (a >= 0) / 1
        (a < 0), r = a (the common convention)."""
        nd, w = self.nd, self.w
        top = lambda op: [self._row(op, q, nd - 1) for q in range(w)]
        DI = self._lin(0, 0, [(r, -1) for r in top(0) + top(1)], ca=2, cb=0, c=-0.375)
        XI = self._lin(0, 0, list(zip(top(0), top(1))), ca=1, cb=4, c=-0.9375)
        G = self._gather([(DI, r) for r in range(2 * w)] + [(XI, q) for q in range(w)])
        jobs = [(G, list(range(2 * w)), "sg_u1"), (G, list(range(2 * w)), "sg_st"),
                (G, list(range(2 * w, 3 * w)), "xsg_u1"), (G, list(range(2 * w, 3 * w)), "xsg_st")]
        SG, offs = self._multi(jobs)
        su = lambda k: (SG, offs[0] + k)            # k: a's word q -> q, b's -> w + q
        ss = lambda k: (SG, offs[1] + k)
        AB = self._cneg([[(0, self._row(op, q, i)) for i in range(nd)] for op in (0, 1) for q in range(w)],
                        [su(k) for k in range(2 * w)], [ss(k) for k in range(2 * w)], 2 * w)
        CL = self._lin(AB, AB, [(r, -1) for r in range(2 * w * nd)], ca=2, cb=0)
        ABc = self._pbs(CL, 2 * w * nd, "dig8")                              # clean |a|, |b| at d / 8
        Bview = self._lin(ABc, ABc, [(w * nd + r, -1) for r in range(w * nd)], ca=1, cb=0)
        Q, Rm = self.divrem(None, "divrem", A=ABc, B=Bview, fin=True)
        # q: negate when the signs differ; r: when a < 0
        QR = self._lin(Q, Rm, [(r, -1) for r in range(w * nd)] + [(-1, r) for r in range(w * nd)], ca=1, cb=1)
        O = self._cneg([[(QR, h * w * nd + q * nd + i) for i in range(nd)] for h in (0, 1) for q in range(w)],
                       [(SG, offs[2] + q) for q in range(w)] + [su(q) for q in range(w)],
                       [(SG, offs[3] + q) for q in range(w)] + [ss(q) for q in range(w)], 2 * w)
        if want == "sdiv":
            return self._store(self._lin(O, O, [(r, -1) for r in range(w * nd)], ca=2, cb=0), w * nd, "dig4_w", out_slots)
        if want == "srem":
            return self._store(self._lin(O, O, [(w * nd + r, -1) for r in range(w * nd)], ca=2, cb=0), w * nd, "dig4_w", out_slots)
        OUT = self._lin(O, O, [(h * w * nd + q * nd + i, -1) for q in range(w) for h in range(2) for i in range(nd)], ca=2, cb=0)
        self._store(OUT, 2 * w * nd, "dig4_w", out_slots)

    # ── word ops with a public constant, single bits, byte ops, conditional zero, shift-add, carry-less products ──

    def _crow(self, vals):
        """A buffer whose row r holds the noiseless constant vals[r] (lins of no terms, one per distinct value)."""
        n = len(vals)
        O = self._lin(0, 0, [(-1, -1)] * n, ca=0, cb=0)
        for v in sorted(set(vals)):
            if v:
                K = self._lin(0, 0, [(-1, -1)], ca=0, cb=0, c=float(v))
                self._lin(O, K, [(r, 0 if vals[r] == v else -1) for r in range(n)], ca=1, cb=1, dst=O)
        return O

    def _cdig(self, c):
        return [(int(c) >> (2 * i)) & 3 for i in range(self.nd)]

    def bitwise_imm(self, out_slots, op, c):
        """a op c (and / or / xor with a public c): the 16-bin table on a_i + 4 c_i (c_i a constant): one round."""
        nd, w, R = self.nd, self.w, self.R
        cd = self._cdig(int(c) % (1 << (2 * nd)))
        K = self._crow([cd[r % nd] / 2.0 - 0.9375 for r in range(R)])
        IN = self._lin(0, K, [(r, r) for r in range(R)], ca=1, cb=1)
        O = self._pbs(IN, R, op + "16")
        self._store(self._lin(O, O, [(r, -1) for r in range(R)], ca=2, cb=0), R, "dig4_w", out_slots)

    def _pair_status(self, kind, c):
        """Statuses of the digit pairs (i, i - 1) of a against the public constant c (kind "add": a + c's
        generate / propagate / kill; "cmp": c vs a, + when a < c, from the top), one 16-bin lookup per position on
        a_i + 4 a_(i-1) (position 0: a_0 alone), its table chosen by (c_i, c_(i-1)) (one launch). Rows [word]
        [position] at 1/4."""
        nd, w, R = self.nd, self.w, self.R
        cd = self._cdig(c)
        IN = self._lin(0, 0, [(self._row(0, q, i), self._row(0, q, i - 1) if i else -1) for q in range(w) for i in range(nd)],
                       ca=1, cb=4, c=-0.9375)
        luts = []
        for q in range(w):
            for i in range(nd):
                kd = "cmpt" if (kind == "cmps" and i == nd - 1) else "cmp" if kind == "cmps" else kind
                name = f"ps_{kd}_{cd[i]}_{cd[i - 1] if i else 'x'}"
                if name not in TABLES:
                    TABLES[name] = (_pair_table(kd, cd[i], cd[i - 1] if i else None), False)
                luts.append(self._tab(name))
        if len(set(luts)) == 1:
            return self._pbs(IN, R, luts[0])
        dst = self._buf()
        self.pr.pbsm(IN, luts, dst)
        return dst

    def add_imm(self, out_slots, c):
        """a + c for a public c: pair statuses (span 2) in round 1, then fan-in 3 combines at distances 2, 6, 18:
        1 + ceil(log3 (nd / 2)) rounds; the digits linear (a_i + c_i + carry_(i-1) - 4 carry_i)."""
        nd, w, R = self.nd, self.w, self.R
        c = int(c) % (1 << (2 * nd))
        cd = self._cdig(c)
        cur = self._pair_status("add", c)
        d = 2
        if nd <= 2:                                   # the pair statuses are the carries
            C = self._pbs(self._lin(cur, cur, [(r, -1) for r in range(R)], ca=1, cb=0), R, "carry_pm")
        else:
            while True:
                f = min(3, -(-nd // d))
                last = d * f >= nd
                U = self._lin(cur, cur, [(r, r - d if r % nd >= d else -1) for r in range(R)], ca=1 << (f - 1), cb=1 << max(f - 2, 0))
                if f == 3:
                    self._lin(U, cur, [(r, r - 2 * d if r % nd >= 2 * d else -1) for r in range(R)], ca=1, cb=1, dst=U)
                if last:
                    C = self._pbs(U, R, "carry_pm")
                    break
                cur = self._pbs(U, R, "sgn")
                d *= f
        C = self._lin(C, C, [(r, -1) for r in range(R)], ca=1, cb=0, c=0.0625)          # {0, 1/8}
        Kc = self._crow([2 * cd[r % nd] / 8.0 for r in range(R)])
        S = self._lin(0, Kc, [(r, r) for r in range(R)], ca=2, cb=1)                      # 2 (a + c) / 8 = s / 4
        O = self._lin(S, C, [(r, r - 1 if r % nd else -1) for r in range(R)], ca=1, cb=2)
        self._lin(O, C, [(r, r) for r in range(R)], ca=1, cb=-8, dst=O)
        self._store(O, R, "dig4_w", out_slots)

    def cmp_imm(self, out_slots, pred, c):
        """[a < c] (lt; slt: signed) / [a == c] (eq) for a public c. lt: pair statuses against c, then the fan-in 3
        tree over the nd / 2 pairs; eq: [pair != c's pair] items (16-bin), the whole-torus zero test (<= 16
        items per group)."""
        nd, w, R = self.nd, self.w, self.R
        c = int(c) % (1 << (2 * nd))
        if pred == "eq":
            cd = self._cdig(c)
            npair = (nd + 1) // 2
            IN = self._lin(0, 0, [(self._row(0, q, 2 * t), self._row(0, q, 2 * t + 1) if 2 * t + 1 < nd else -1)
                                  for q in range(w) for t in range(npair)], ca=1, cb=4, c=-0.9375)
            luts = []
            for q in range(w):
                for t in range(npair):
                    hi_c = cd[2 * t + 1] if 2 * t + 1 < nd else 0
                    name = f"nepair_{cd[2 * t]}_{hi_c}"
                    if name not in TABLES:
                        TABLES[name] = ((lambda m, x=cd[2 * t], y=hi_c: (lambda a, b: ((a != x) | (b != y)).astype(float))(*_ab16(m)) / 8.0), False)
                    luts.append(self._tab(name))
            if len(set(luts)) == 1:
                U = self._pbs(IN, len(luts), luts[0])
            else:
                U = self._buf()
                self.pr.pbsm(IN, luts, U)
            rows = [[q * npair + t for t in range(npair)] for q in range(w)]
            while True:
                gs = [r[k:k + 16] for r in rows for k in range(0, len(r), 16)]
                last = all(len(r) <= 16 for r in rows)
                S = self._gsum(U, gs, ca=1, c=-0.9375)
                V = self._pbs(S, len(gs), "eqz_4" if last else "nez_pm")
                V = self._lin(V, V, [(g, -1) for g in range(len(gs))], ca=1, cb=0, c=0.125 if last else 0.0625)
                if last:
                    return self._store(V, w, "id8_w", out_slots)
                per = [(len(r) + 15) // 16 for r in rows]
                off = np.concatenate([[0], np.cumsum(per)])
                rows = [list(range(off[q], off[q + 1])) for q in range(w)]
                U = V
        signed = pred == "slt"
        S = self._pair_status("cmps" if signed else "cmp", c)
        # the pairs (2t + 1, 2t) (the top pair's statuses compare the sign-flipped digits for slt); tree from the top
        k = (nd + 1) // 2
        cur = self._lin(S, S, [(q * nd + min(2 * t + 1, nd - 1), -1) for q in range(w) for t in range(k)], ca=1, cb=0)
        while True:
            f = min(3, k)
            h = -(-k // f)
            src = [[(qq // h) * k + (qq % h) * f + f - 1 - x if (qq % h) * f + f - 1 - x < k else -1 for x in range(f)]
                   for qq in range(w * h)]
            U = self._lin(cur, cur, [(v[0], v[1] if f > 1 else -1) for v in src], ca=1 << (f - 1), cb=1 << max(f - 2, 0))
            if f == 3:
                self._lin(U, cur, [(qq, v[2]) for qq, v in enumerate(src)], ca=1, cb=1, dst=U)
            if h == 1:
                break
            cur = self._pbs(U, w * h, "sgn")
            k = h
        # + : a < c
        if f <= 2:
            V = self._pbs(U, w, "gt_4")
            return self._store(V, w, "id8_w", out_slots)
        V = self._pbs(U, w, "gt_pm8")
        self._store(self._lin(V, V, [(q, -1) for q in range(w)], ca=1, cb=0, c=0.125), w, "id8_w", out_slots)

    def bit_op(self, out_slots, kind, j=None):
        """Single-bit ops: bset / bclr / binv (a with bit j set / cleared / flipped) and bext (bit j of a, as a word)
        with a public j (one 8-bin lookup on digit j // 2: one round) or the encrypted index b mod bits (j None):
        round 1 b's low bits, round 2 per digit i the mask m_i = [(idx >> 1) == i] (1 + idx & 1) (a zero test of
        sum_k bit_k xor i_k, idx & 1 at weight 6), round 3 one 16-bin lookup per digit on a_i + 4 m_i."""
        nd, w, R, bits = self.nd, self.w, self.R, 2 * self.nd
        if j is not None:
            j = int(j) % bits
            di, t = j // 2, j % 2
            rows = [self._row(0, q, di) for q in range(w)]
            IN = self._lin(0, 0, [(r, -1) for r in rows], ca=2, cb=0, c=-0.375)
            V = self._pbs(IN, w, f"b{kind}{t}")
            if kind == "ext":
                O = self._lin(V, V, [(q, -1) if i == 0 else (-1, -1) for q in range(w) for i in range(nd)], ca=2, cb=0)
            else:
                O = self._lin(0, V, [(self._row(0, q, i), -1) if i != di else (-1, q) for q in range(w) for i in range(nd)], ca=2, cb=2)
            return self._store(O, R, "dig4_w", out_slots)
        L = bits.bit_length() - 1                     # index bits 0 .. L - 1
        na = (L + 1) // 2
        X = self._lin(0, 0, [(self._row(1, q, i), -1) for q in range(w) for i in range(na)], ca=2, cb=0, c=-0.375)
        # b's bits / 8 (one launch: lo, hi of its low digits, and bit 0 at 6 / 8: the mask table's input weight)
        n0 = w * na
        BB, (olo, ohi, o6) = self._multi([(X, list(range(n0)), "blo"), (X, list(range(n0)), "bhi"),
                                          (X, [q * na for q in range(w)], "blo6")])
        bit = lambda q, k: (BB, (olo if k % 2 == 0 else ohi) + q * na + k // 2) if k else (BB, o6 + q)
        terms, cst = [], []
        for q in range(w):
            for i in range(nd):
                tl, c0 = [], -0.9375
                for k in range(1, L):
                    ib = (i >> (k - 1)) & 1
                    tl.append((bit(q, k), -1 if ib else 1))
                    c0 += 0.125 if ib else 0.0
                tl.append((bit(q, 0), 1))
                terms.append(tl)
                cst.append(c0)
        srcl = list(dict.fromkeys(x[0] for tl in terms for x in tl))
        G = self._gather(srcl)
        idx = {x: r for r, x in enumerate(srcl)}
        M_in = self._lin_terms(G, [[(idx[x], cf) for x, cf in tl] for tl in terms], cst)
        M = self._pbs(M_in, R, "bmask")                # m_i / 8
        IN = self._lin(0, M, [(r, r) for r in range(R)], ca=1, cb=4, c=-0.9375)
        V = self._pbs(IN, R, f"bv{kind}")
        if kind == "ext":
            O = self._gsum(V, [[q * nd + i for i in range(nd)] if i0 == 0 else [-1] for q in range(w) for i0 in range(nd)], ca=2, c=0.0)
        else:
            O = self._lin(V, V, [(r, -1) for r in range(R)], ca=2, cb=0)
        self._store(O, R, "dig4_w", out_slots)

    def shadd(self, out_slots, k, uw=False):
        """(a << k) + b, k = 0 .. 3 (uw: a's low 32 bits zero-extended first): k even is a digit renumbering, odd k
        takes (a << 1)'s digits from one 16-bin lookup on a_j + 4 a_(j-1) (one round), then the adder."""
        nd, w, R = self.nd, self.w, self.R
        keep = (lambda i: i < 16) if uw else (lambda i: True)
        if k % 2:
            IN = self._lin(0, 0, [(self._row(0, q, j) if keep(j) else -1, self._row(0, q, j - 1) if j and keep(j - 1) else -1)
                                  for q in range(w) for j in range(nd)], ca=1, cb=4, c=-0.9375)
            A1 = self._pbs(IN, R, "shl1d")
            asrc = lambda q, i: (A1, q * nd + i)
        else:
            asrc = lambda q, i: (0, self._row(0, q, i)) if keep(i) else None
        sd = k // 2
        A = self._rows([asrc(q, i - sd) if i - sd >= 0 else None for q in range(w) for i in range(nd)])
        X = self._lin(A, 0, [(r, R + r) for r in range(R)], ca=2, cb=2, c=-0.75)
        self._adder("add", out_slots, X=X)

    def byte_op(self, out_slots, kind):
        """orcb (every byte -> 0xFF if nonzero else 0: a whole-byte zero test on the sum of its 4 digits), rev8 (byte
        reversal: a renumbering), zext16 (zero-extend the low 16 bits)."""
        nd, w, R = self.nd, self.w, self.R
        nb = nd // 4
        if kind == "orcb":
            S = self._gsum(0, [[self._row(0, q, 4 * t + j) for j in range(4)] for q in range(w) for t in range(nb)], ca=1, c=-0.9375)
            V = self._pbs(S, w * nb, "nzb3")                                   # 3 / 8 when the byte is nonzero
            O = self._lin(V, V, [(q * nb + i // 4, -1) for q in range(w) for i in range(nd)], ca=2, cb=0)
        elif kind == "rev8":
            O = self._lin(0, 0, [(self._row(0, q, 4 * (nb - 1 - i // 4) + i % 4), -1) for q in range(w) for i in range(nd)], ca=2, cb=0)
        else:
            O = self._lin(0, 0, [(self._row(0, q, i) if i < 8 else -1, -1) for q in range(w) for i in range(nd)], ca=2, cb=0)
        self._store(O, R, "dig4_w", out_slots)

    def czero(self, out_slots, nez=False):
        """czeqz: c == 0 ? 0 : a; cznez: c != 0 ? 0 : a (c = b, a whole word): [c's digit (digit pair above 16 digits)
        != 0] items, the whole-torus zero test of their sum read as the AND mux's selector, the AND mux: 3 rounds."""
        nd, w, R = self.nd, self.w, self.R
        if nd > 16:
            npair = nd // 2
            IN = self._lin(0, 0, [(self._row(1, q, 2 * t), self._row(1, q, 2 * t + 1)) for q in range(w) for t in range(npair)],
                           ca=1, cb=4, c=-0.9375)
            U = self._pbs(IN, w * npair, "nzp")
            gs = [[q * npair + t for t in range(npair)] for q in range(w)]
        else:
            X = self._lin(0, 0, [(R + r, -1) for r in range(R)], ca=2, cb=0)
            U = self._pbs(X, R, "neq8")
            gs = [[q * nd + i for i in range(nd)] for q in range(w)]
        S = self._gsum(U, gs, ca=1, c=-0.9375)
        V = self._pbs(S, w, "selz_pm")                    # +1/2 when c == 0
        SEL = self._lin(V, V, [(q, -1) for q in range(w)], ca=1 if nez else -1, cb=0, c=0.5)   # keep a: unit 1
        IN = self._lin(0, SEL, [(r, r // nd) for r in range(R)], ca=2, cb=1, c=-0.875)
        A = self._pbs(IN, R, "and")
        self._store(self._lin(A, A, [(r, -1) for r in range(R)], ca=2, cb=0), R, "dig4_w", out_slots)

    def clmul(self, out_slots, part="lo"):
        """Carry-less products (XOR of the shifted partial products): bits 0 .. bits - 1 (lo), bits .. 2 bits - 1
        (hi), bits - 1 .. 2 bits - 2 (rev). Per digit pair one 16-bin lookup per result bit it touches (c0 = a0 b0,
        c1 = a0 b1 ^ a1 b0, c2 = a1 b1), then per result bit the parity of its items (groups of <= 15, one parity
        lookup per group, until one item), the result digits linear in the bits."""
        nd, w, bits = self.nd, self.w, 2 * self.nd
        lo_bit = {"lo": 0, "hi": bits, "rev": bits - 1}[part]
        items = [[[] for _ in range(bits)] for _ in range(w)]
        pairs = []
        for q in range(w):
            for i in range(nd):
                for j in range(nd):
                    for t in range(3):
                        pos = 2 * (i + j) + t - lo_bit
                        if 0 <= pos < bits:
                            pairs.append((q, i, j, t, pos))
        IN = self._lin(0, 0, [(self._row(0, q, i), self._row(1, q, j)) for q, i, j, t, pos in pairs], ca=1, cb=4, c=-0.9375)
        jobs = [(IN, [k for k, x in enumerate(pairs) if x[3] == t], f"clm{t}") for t in range(3)]
        jobs = [jb for jb in jobs if jb[1]]
        P, offs = self._multi(jobs)
        where = {}
        for (bf, rows, t), o in zip(jobs, offs):
            for jj, k in enumerate(rows):
                where[k] = o + jj
        for k, (q, i, j, t, pos) in enumerate(pairs):
            items[q][pos].append(where[k])
        buf = P
        while any(len(x) > 1 for its in items for x in its):
            groups, plan = [], []
            for q in range(w):
                for pos in range(bits):
                    x = items[q][pos]
                    if len(x) > 1:
                        gs = [x[k:k + 15] for k in range(0, len(x), 15)]
                        plan.append((q, pos, list(range(len(groups), len(groups) + len(gs)))))
                        groups += gs
                    else:
                        plan.append((q, pos, None))
            S = self._gsum(buf, groups, ca=1, c=-0.9375)
            V = self._pbs(S, len(groups), "par15")
            srcs, nitems = [], [[[] for _ in range(bits)] for _ in range(w)]
            for q, pos, gl in plan:
                if gl is None:
                    for r in items[q][pos]:
                        nitems[q][pos].append(len(srcs))
                        srcs.append((buf, r))
                else:
                    for g in gl:
                        nitems[q][pos].append(len(srcs))
                        srcs.append((V, g))
            buf, items = self._gather(srcs), nitems
        bsrc = lambda q, pos: (buf, items[q][pos][0]) if items[q][pos] else None
        B0 = self._rows([bsrc(q, 2 * i) for q in range(w) for i in range(nd)])
        B1 = self._rows([bsrc(q, 2 * i + 1) for q in range(w) for i in range(nd)])
        O = self._lin(B0, B1, [(r, r) for r in range(w * nd)], ca=2, cb=4)
        self._store(O, w * nd, "dig4_w", out_slots)

    # ── multiplication / division by a public constant ─────────────────────────────────────────────────────────

    def _csd(self, c):
        """c >= 0 in canonical signed digits: [s_j] with c = sum s_j 2^j, s_j in {-1, 0, 1}, no two adjacent nonzero."""
        out, k = [], int(c)
        while k:
            t = (2 - (k & 3)) if k & 1 else 0
            out.append(t)
            k = (k - t) >> 1
        return out

    def _const_tree(self, xdig, c, nc, extra=None):
        """x c mod 4^nc (+ the items of `extra`: [(word, column, (buffer, row), max)]) for the public c >= 0 and the
        words x (xdig(q, i): (buffer, row) of digit i < nd at d / 8, clean): canonical signed digits, + x 2^j as x's
        (x << 1's for odd j, one 16-bin lookup per digit on x_i + 4 x_(i-1), one round) digits from column j // 2,
        - x 2^j as their complements (3 - d) plus the two's complement's constant digits; then `_reduce16`.
        Returns (ITEMS, cols) for `_final_lin`."""
        nd, w = self.nd, self.w
        csd = self._csd(c)
        A1 = None
        if any(t and j % 2 for j, t in enumerate(csd)):
            b0 = xdig(0, 0)[0]
            IN = self._lin(b0, b0, [(xdig(q, i)[1] if i < nd else -1, xdig(q, i - 1)[1] if i else -1)
                                    for q in range(w) for i in range(nd + 1)], ca=1, cb=4, c=-0.9375)
            A1 = self._pbs(IN, w * (nd + 1), "shl1d")                       # (x << 1)'s nd + 1 digits
        pos, neg = [], []
        cols = [[[] for _ in range(nc)] for _ in range(w)]
        K = 0
        for j, t in enumerate(csd):
            if not t:
                continue
            s, m = j // 2, nd + (j % 2)
            for q in range(w):
                for i in range(m):
                    col = s + i
                    if col >= nc:
                        break
                    sv = xdig(q, i) if j % 2 == 0 else (A1, q * (nd + 1) + i)
                    (pos if t > 0 else neg).append((q, col, sv))
            if t < 0:                                   # - X 4^s = sum (3 - X_i) 4^(s+i) + sum_(col >= s+m) 3 4^col + 4^s
                K += (1 << (2 * s)) + sum(3 << (2 * col) for col in range(s + m, nc))
        K %= 1 << (2 * nc)
        P = self._rows([sv for _, _, sv in pos]) if pos else None
        N = self._rows([sv for _, _, sv in neg]) if neg else None
        if N is not None:
            N = self._lin(N, N, [(r, -1) for r in range(len(neg))], ca=-1, cb=0, c=0.375)       # 3 - d
        kd = [(K >> (2 * col)) & 3 for col in range(nc)]
        consts = [(q, col, kd[col]) for q in range(w) for col in range(nc) if kd[col]]
        C = self._crow([v / 8.0 for _, _, v in consts]) if consts else None
        srcs = []
        for (q, col, _), r in zip(pos, range(len(pos))):
            cols[q][col].append((len(srcs), 3))
            srcs.append((P, r))
        for (q, col, _), r in zip(neg, range(len(neg))):
            cols[q][col].append((len(srcs), 3))
            srcs.append((N, r))
        for (q, col, v), r in zip(consts, range(len(consts))):
            cols[q][col].append((len(srcs), v))
            srcs.append((C, r))
        for q, col, sv, mx in (extra or []):
            cols[q][col].append((len(srcs), mx))
            srcs.append(sv)
        if not srcs:
            return None, cols
        ITEMS = self._gather(srcs)
        return self._reduce16(ITEMS, cols, nc)

    def mulc2(self, out_slots, c):
        """a c mod 2^bits for a public c: the canonical-signed-digit partial products (`_const_tree`), the prefix
        adder."""
        nd, w, R = self.nd, self.w, self.R
        c = int(c) % (1 << (2 * nd))
        ITEMS, cols = self._const_tree(lambda q, i: (0, self._row(0, q, i)), c, nd)
        if ITEMS is None:
            return self._store(self._lin(0, 0, [(-1, -1)] * R, ca=0, cb=0), R, "dig4_w", out_slots)
        self._final_lin(ITEMS, cols, nd, out_slots)

    def divc(self, out_slots, d, want="div"):
        """a / d, a mod d (want div / rem / divrem) for a public d: q = (a m) >> S with the magic m = ceil(2^S / d),
        S = bits + l, l >= ceil(log2 d) chosen so S is even (exact for every a < 2^bits: m d - 2^S < 2^l); the
        product's columns by `_const_tree`, q the digits from S / 2. r = a - q d: q's digits cleaned (one round),
        then a + q (-d) by the same tree. d = 0: q = all ones, r = a; d a power of two: digit renumbering."""
        nd, w, R, bits = self.nd, self.w, self.R, 2 * self.nd
        d = int(d) % (1 << bits)
        if d == 0:
            q_dig = [3] * nd
            Q = self._crow([0.75] * R)
            Rm = self._lin(0, 0, [(r, -1) for r in range(R)], ca=1, cb=0)
        elif d & (d - 1) == 0 and want != "divrem":        # a power of two: a shift / a mask
            return self.shift(out_slots, "shr", d.bit_length() - 1) if want == "div" else self.bitwise_imm(out_slots, "and", d - 1)
        elif getattr(self, "_divc_form", None) == "long" or (getattr(self, "_divc_form", None) is None and _divc_long_wins(self, d, want)):
            return self.divrem(out_slots, want, cdiv=d)
        else:
            l = (d - 1).bit_length()
            S = bits + l + ((bits + l) % 2)
            m = -(-(1 << S) // d)
            nc = (S + bits) // 2
            ITEMS, cols = self._const_tree(lambda q, i: (0, self._row(0, q, i)), m, nc)
            O = self._final_lin(ITEMS, cols, nc, None)
            self.pr.steps.pop()
            Q = self._lin(O, O, [(q * nc + S // 2 + i, -1) for q in range(w) for i in range(nd)], ca=1, cb=0)   # d / 4
            Rm = None
        if want == "div":
            return self._store(Q, R, "dig4_w", out_slots)
        if Rm is None:
            Qc = self._pbs(Q, R, "dig8")                                     # q's digits clean at d / 8
            negd = (1 << bits) - d
            ITEMS, cols = self._const_tree(lambda q, i: (Qc, q * nd + i), negd, nd,
                                           extra=[(q, i, (0, self._row(0, q, i)), 3) for q in range(w) for i in range(nd)])
            O = self._final_lin(ITEMS, cols, nd, None)
            self.pr.steps.pop()
            Rm = O
        else:
            Rm = self._lin(Rm, Rm, [(r, -1) for r in range(R)], ca=2, cb=0)
        if want == "rem":
            return self._store(Rm, R, "dig4_w", out_slots)
        OUT = self._lin(Q, Rm, [(q * nd + i, -1) if h == 0 else (-1, q * nd + i) for q in range(w) for h in range(2) for i in range(nd)], ca=1, cb=1)
        self._store(OUT, 2 * R, "dig4_w", out_slots)

    def word32(self, out_slots, op):
        """The 32-bit op `op` on the low halves of 64-bit words, its result sign-extended to 64 bits: one lin
        gathers the low 16 digits of a and b into a 32-bit layout, the 32-bit program (built on its own and
        spliced in, its load buffer renamed to the gathered one) runs on w words, then one lookup per word on the
        result's digit 15 gives the fill digit 3 / 0."""
        if self.nd != 32:
            raise ValueError("word32: 64-bit words")
        w, R = self.w, self.R
        Ri = w * 16
        RM = self._lin(0, 0, [(op_ * R + q * 32 + i, -1) for op_ in (0, 1) for q in range(w) for i in range(16)]
                       + [(2 * R + q, -1) for q in range(w)], ca=1, cb=0)
        inner = WordProg("w32", w, 16)
        inner.multi, inner.fan4 = self.multi, self.fan4
        inner._capture = True
        build_into(inner, op)
        ren = {0: RM}
        for st in inner.pr.steps:
            for key in ("dst", "src", "a", "b"):
                if key in st:
                    vs = st[key] if isinstance(st[key], list) else [st[key]]
                    for v in vs:
                        if v not in ren:
                            ren[v] = self._buf()
        for name, (fs, warp) in inner.pr.luts.items():
            if name not in self.pr.luts:
                self.pr.luts[name] = (fs, warp)
        for st in inner.pr.steps:
            if st["op"] == "load":
                continue
            st = dict(st)
            for key in ("dst", "src", "a", "b"):
                if key in st:
                    st[key] = [ren[v] for v in st[key]] if isinstance(st[key], list) else ren[st[key]]
            self.pr.steps.append(st)
        src, rows, lut = inner._captured
        src = ren[src]
        if lut != "dig4_w":
            raise ValueError(f"word32 {op}: a word result expected")
        X = self._lin(src, src, [(q * 16 + 15, -1) for q in range(w)], ca=1, cb=0, c=-0.375)
        F = self._pbs(X, w, "sext3")                                          # 3 / 4 when bit 31 is set
        O = self._lin(src, F, [(q * 16 + i, -1) if i < 16 else (-1, q) for q in range(w) for i in range(32)], ca=1, cb=1)
        self._store(O, R, "dig4_w", out_slots)

    def rotate_enc(self, out_slots, kind):
        """a rotated by the encrypted amount b mod bits (kind rotl / rotr): round 1 the bits of a's digits (lo, hi:
        one many-LUT lookup) and of b's low digits at 7 / 8 (the gates' selector weight); then one stage per amount bit
        (1 bit, then 1, 2, 4 .. digits), each ONE gated lookup per digit: x' = x + (sel ? x_s - x : 0) on the 16-bin
        input (x_s - x + 3) / 8 + 7 sel / 8, the digits carried linearly (their noise grows by one PBS output per
        stage: 2^-133 at 64 bits)."""
        nd, w, R, bits = self.nd, self.w, self.R, 2 * self.nd
        L = bits.bit_length() - 1
        na = (L + 1) // 2
        X = self._lin(0, 0, [(r, -1) for r in range(R)], ca=2, cb=0, c=-0.375)
        LO, HI = self._pbsk(X, R, "bitsk")
        XA = self._lin(0, 0, [(self._row(1, q, i), -1) for q in range(w) for i in range(na)], ca=2, cb=0, c=-0.375)
        A7L, A7H = self._pbsk(XA, w * na, "bits7")                         # amount bits at 7 / 8
        sel7 = lambda q, k: ((A7L if k % 2 == 0 else A7H), q * na + k // 2)
        left = kind == "rotl"
        # stage 0 (one bit): x_s digit i = (left) 2 lo_i + hi_(i-1) | (right) hi_i + 2 lo_(i+1) (cyclic)
        rowsA, rowsB = [], []
        for q in range(w):
            for i in range(nd):
                if left:
                    rowsA.append((q * nd + i, q * nd + (i - 1) % nd))
                else:
                    rowsB.append((q * nd + (i + 1) % nd, q * nd + i))
        XS = self._lin(LO, HI, rowsA if left else rowsB, ca=2, cb=1)       # x_s at d / 8
        cur = [(0, self._row(0, q, i)) for q in range(w) for i in range(nd)]     # (buffer, row) of x's digits
        xs_src = [(XS, r) for r in range(R)]
        for k in range(L):
            if k > 0:
                s = 1 << (k - 1)
                xs_src = [cur[q * nd + ((i - s) % nd if left else (i + s) % nd)] for q in range(w) for i in range(nd)]
            Xc = self._rows(cur)
            Xs = self._rows(xs_src)
            S7 = self._rows([sel7(q, k) for q in range(w)])
            G = self._lin(Xs, Xc, [(r, r) for r in range(R)], ca=1, cb=-1, c=0.375 - 0.9375)
            self._lin(G, S7, [(r, r // nd) for r in range(R)], ca=1, cb=1, dst=G)
            Gd = self._pbs(G, R, "gdel")                                   # sel ? (x_s - x) / 8 : 0
            N = self._lin(Xc, Gd, [(r, r) for r in range(R)], ca=1, cb=1)
            cur = [(N, r) for r in range(R)]
        O = self._rows(cur)
        self._store(self._lin(O, O, [(r, -1) for r in range(R)], ca=2, cb=0), R, "dig4_w", out_slots)

    def minmax(self, out_slots, kind="min", signed=False):
        """min / max of a, b per word."""
        nd, w, R = self.nd, self.w, self.R
        S = self._tree("lt", signed, sel=True)                     # [a < b] at unit 1, w rows
        NS = self._lin(S, S, [(q, -1) for q in range(w)], ca=-1, cb=0, c=1.0)
        first, second = (0, R) if kind == "min" else (R, 0)
        sel = [q for q in range(w) for _ in range(nd)]
        O = self._andmux([(0, [first + r for r in range(R)], S, sel), (0, [second + r for r in range(R)], NS, sel)], R)
        self._store(self._lin(O, O, [(r, -1) for r in range(R)], ca=2, cb=0), R, "dig4_w", out_slots)

    def abs(self, out_slots):
        """|a| (two's complement; abs(MIN) = MIN)."""
        nd, w, R = self.nd, self.w, self.R
        X = self._lin(0, 0, [(r, -1) for r in range(R)], ca=2, cb=0)      # -a = ~(a + 11..1)
        LH = self._lh(self._top(0))
        C = self._carries(X)
        prev = [(r, r - 1 if r % nd else -1) for r in range(R)]
        # -a at d / 8: 3/8 - (a + 3/8 + c_(i - 1) - 4 c_i) = -a - c_(i - 1) + 4 c_i
        N = self._lin(0, C, prev, ca=-1, cb=-1)
        self._lin(N, C, [(r, r) for r in range(R)], ca=1, cb=4, dst=N)
        S = self._lin(LH, LH, [(w + q, -1) for q in range(w)], ca=8, cb=0)        # sign (unit 1)
        NS = self._lin(LH, LH, [(w + q, -1) for q in range(w)], ca=-8, cb=0, c=1.0)
        sel = [q for q in range(w) for _ in range(nd)]
        O = self._andmux([(N, list(range(R)), S, sel), (0, list(range(R)), NS, sel)], R)
        self._store(self._lin(O, O, [(r, -1) for r in range(R)], ca=2, cb=0), R, "dig4_w", out_slots)


# ── tables of the other ops (native values on [-1, 1); module doc) ───────────────────────────────────────────────

def _k8(m):                  # the threshold tables' input 2 s / 8 - 7 / 8 -> s in 0..7
    return np.clip(np.round(4.0 * (np.asarray(m, dtype=np.float64) + 0.875)), 0, 7)


def _bits2(m):               # lo on 2 d / 8 + 1/16 (upper region), hi on -2 d / 8 + 1/16 (lower region)
    k = np.round((np.asarray(m, dtype=np.float64) - 0.0625) * 4.0).astype(np.int64)
    return np.where(k >= 0, np.clip(k, 0, 3) & 1, np.clip(-k, 0, 3) >> 1) / 8.0


def _and(m):                 # 2 x / 8 + sel - 7 / 8 -> sel x / 8
    k = _k8(m)
    return np.where(k >= 4, k - 4, 0) / 8.0


def _k16(m):                 # a 16-bin input s / 8 - 15 / 16 -> s in 0..15
    return np.clip(np.round(8.0 * (np.asarray(m, dtype=np.float64) + 0.9375)), 0, 15)


def _ab16(m):                # a + 4 b -> (a, b)
    k = _k16(m).astype(np.int64)
    return k & 3, k >> 2


_PRED = {"lt": lambda r: r < 0, "gt": lambda r: r > 0, "le": lambda r: r <= 0, "ge": lambda r: r >= 0,
         "eq": lambda r: r == 0, "ne": lambda r: r != 0}

TABLES = {
    "sgn": (_sgn, False), "carry_pm": (_carry_pm, False),
    "out_w": (lambda m: 2.0 * _digit_out(m), True),
    "outn_w": (lambda m: 2.0 * (0.375 - _digit_out(m)), True),
    "dig4_w": (lambda m: np.clip(np.round(4.0 * np.asarray(m)), 0, 3) / 4.0, True),
    "id8_w": (lambda m: np.where(np.asarray(m) > 0.125, 1.0, 0.0), True),      # {0, 1/4} -> {0, 1}
    "hs": (lambda m: np.where(_r4(m) >= 2, 2.0, 0.0), False),          # a sign bit as a half turn (2 d / 8 in)
    "bits": (_bits2, False),
    "and": (_and, False),
    "cy7": (lambda m: np.where(_k8(m) >= 4, 0.125, 0.0), False),
    "md7": (lambda m: np.mod(_k8(m), 4) / 8.0, False),
    "nz": (lambda m: np.where(_k8(m) >= 1, 0.125, 0.0), False),
    "ge2u": (lambda m: np.where(_k8(m) >= 2, 1.0, 0.0), False),
    "ovf_pm": (lambda m: np.where((_k8(m) >= 2) & (_k8(m) <= 5), 0.0625, -0.0625), False),
    "bw_and": (lambda m: np.where(_r4(m) == 2, 0.125, 0.0), False),        # bit sums 0 / 1 / 2 at 1/4
    "bw_or": (lambda m: np.where(_r4(m) >= 1, 0.125, 0.0), False),
    "bw_xor": (lambda m: np.where(_r4(m) == 1, 0.125, 0.0), False),
    "pc": (lambda m: np.array([0, 1, 1, 2])[np.clip(_r4(m), 0, 3).astype(np.int64)] / 8.0, False),
}
def _dc(m):                  # a digit on 2 d / 8 - 3 / 8 -> d
    return np.clip(_r4(np.asarray(m, dtype=np.float64) + 0.375), 0, 3)


def _r8(m):
    return np.round(8.0 * np.asarray(m, dtype=np.float64))


TABLES.update({
    "sgn8x": (lambda m: np.sign(_r4(m)) / 8.0, False),                     # statuses for a fan-in 4 combine
    "sgn1x": (lambda m: np.sign(_r4(m)) / 2.0, False),
    "sgn4x": (lambda m: np.sign(_r8(m)) / 4.0, False),                      # combines read at spacing 1/8
    "carry_pm8x": (lambda m: np.where(_r8(m) > 0, 0.0625, -0.0625), False),
    "sgn8_2": (lambda m: np.sign(_r8(m)) / 2.0, False),                     # a fan-in 4 combine's status at 1/2
    "sgn8_8": (lambda m: np.sign(_r8(m)) / 8.0, False),                     # ... and at 1/8
    "sg_u1": (lambda m: (_dc(m) >= 2).astype(float), False),                # the sign bit as a unit-1 selector
    "sg_st": (lambda m: np.where(_dc(m) >= 2, 0.25, -0.25), False),         # ... as a carry-in status (G / K)
    "xsg_u1": (lambda m: (lambda a, b: ((a >= 2) != (b >= 2)).astype(float))(*_ab16(m)), False),
    "xsg_st": (lambda m: (lambda a, b: np.where((a >= 2) != (b >= 2), 0.25, -0.25))(*_ab16(m)), False),
    "dig8": (lambda m: np.clip(_r4(m), 0, 3) / 8.0, False),                 # 2 d / 8 (linear, noisy) -> d / 8 clean
    "shl1d": (lambda m: (lambda a, b: 2 * (a & 1) + (b >> 1))(*_ab16(m)) / 8.0, False),     # (b << 1)_j from b_j + 4 b_(j-1)
    "nc_pm1": (lambda m: np.where(_r4(m) > 0, -0.5, 0.5), False),           # no carry out: +1/2 (whole torus)
    "nc_pm7": (lambda m: np.where(_r4(m) > 0, -0.4375, 0.4375), False),
    "nc_pm8": (lambda m: np.where(_r4(m) > 0, -0.0625, 0.0625), False),
    "gsel": (lambda m: (lambda k: np.where(k >= 7, np.sign(k - 10) / 4.0, 0.0))(_k16(m)), False),   # sel ? sign(x) : 0
    "gdel": (lambda m: (lambda k: np.where(k >= 7, (k - 10) / 8.0, 0.0))(_k16(m)), False),           # sel ? x / 8 : 0
    "md16": (lambda m: np.mod(_k16(m), 4) / 8.0, False),                    # sums 0..16 (whole torus)
    "dv16": (lambda m: np.mod(np.floor(_k16(m) / 4.0), 4) / 8.0, False),
    "eq16": (lambda m: np.where(_k16(m) >= 16, 0.0625, -0.0625), False),    # -1/16 below 16: +1/16 at 16 (negacyclic)
    "pc2": (lambda m: (lambda a, b: np.array([0, 1, 1, 2])[a] + np.array([0, 1, 1, 2])[b])(*_ab16(m)) / 8.0, False),
    "nzp": (lambda m: (lambda a, b: (a + b > 0).astype(float))(*_ab16(m)) / 8.0, False),          # [a | b != 0]
    "nzb": (lambda m: (lambda a, b: ((b > 0) | (a >= 2)).astype(float))(*_ab16(m)) / 8.0, False),  # [b != 0 or a >= 2]
    "nzc": (lambda m: (lambda a, b: ((b > 0) | (a & 1 > 0)).astype(float))(*_ab16(m)) / 8.0, False),
    "neq8": (lambda m: np.where(_r4(m) != 0, 0.125, 0.0), False),            # [t != 0] on 2 t / 8, t in -3..3
    "nez_pm": (lambda m: np.where(_k16(m) >= 1, 0.0625, -0.0625), False),  # [sum != 0] / 8 - 1/16 (sums 0..16, whole torus)
    "eqz_4": (lambda m: np.where(_k16(m) == 0, 0.125, -0.125), False),      # +-1/8 (then + 1/8: {0, 1/4})
    "nez_4": (lambda m: np.where(_k16(m) != 0, 0.125, -0.125), False),
    "plo": (lambda m: (np.prod(_ab16(m), axis=0) & 3) / 8.0, False),        # (a b) mod 4 on a + 4 b
    "phi": (lambda m: (np.prod(_ab16(m), axis=0) >> 2) / 8.0, False),       # (a b) div 4
    "md15": (lambda m: np.mod(_k16(m), 4) / 8.0, False),                    # group sums <= 15: s mod 4, s div 4
    "dv15": (lambda m: np.floor(_k16(m) / 4.0) / 8.0, False),
    "xor16": (lambda m: np.bitwise_xor(*_ab16(m)) / 8.0, False),            # bitwise ops on a + 4 b, d / 8 out
    "and16": (lambda m: np.bitwise_and(*_ab16(m)) / 8.0, False),
    "or16": (lambda m: np.bitwise_or(*_ab16(m)) / 8.0, False),
    "andn16": (lambda m: (lambda a, b: a & (3 - b))(*_ab16(m)) / 8.0, False),
    "orn16": (lambda m: (lambda a, b: a | (3 - b))(*_ab16(m)) / 8.0, False),
    "xnor16": (lambda m: (lambda a, b: 3 - (a ^ b))(*_ab16(m)) / 8.0, False),
})
# many-LUT tables (k outputs from one rotation: the input read k times coarser; module doc)
TABLESK = {
    "sgn2s": ([lambda m: np.sign(_r4(m)) / 8.0, lambda m: np.sign(_r4(m)) / 2.0], False),   # a status at 1/8 and 1/2
    # conditional complement on 2 x / 8 + s - 7 / 8: y = s ? 3 - x : x, and y's status as an addend of 0 (P at 3)
    "cneg": ([lambda m: (lambda k: np.where(k >= 4, 7 - k, k))(_k8(m)) / 8.0,
              lambda m: (lambda k: np.where(np.where(k >= 4, 7 - k, k) == 3, 0.0, -0.25))(_k8(m))], False),
    "mdcy7": ([lambda m: np.mod(_k8(m), 4) / 8.0, lambda m: np.where(_k8(m) >= 4, 0.125, 0.0)], False),
    "uh": ([lambda m: (np.clip(_r4(np.asarray(m) + 0.375), 0, 3) > 0) / 8.0,
            lambda m: (np.clip(_r4(np.asarray(m) + 0.375), 0, 3) >= 2) / 8.0], False),
    "ul": ([lambda m: (np.clip(_r4(np.asarray(m) + 0.375), 0, 3) > 0) / 8.0,
            lambda m: (np.clip(_r4(np.asarray(m) + 0.375), 0, 3).astype(np.int64) & 1) / 8.0], False),
    "bits7": ([lambda m: (np.clip(_r4(np.asarray(m) + 0.375), 0, 3).astype(np.int64) & 1) * 0.875,
               lambda m: (np.clip(_r4(np.asarray(m) + 0.375), 0, 3).astype(np.int64) >> 1) * 0.875], False),
    "bitsk": ([lambda m: (np.clip(_r4(np.asarray(m) + 0.375), 0, 3).astype(np.int64) & 1) / 8.0,
               lambda m: (np.clip(_r4(np.asarray(m) + 0.375), 0, 3).astype(np.int64) >> 1) / 8.0], False),
}

def _pair_table(kind, ch, cl):
    """The status of the digit pair (a_i, a_(i-1)) (input a_i + 4 a_(i-1)) against constant digits (ch, cl); cl None:
    position 0 (a_i alone). add: of a + c (G +1/4, P 0, K -1/4); cmp: sign(c - a), the high digit first; cmpt: the
    same with both high digits' top bits flipped (the signed order on the word's top digit)."""
    def st(k, x, cdig):
        if k == "add":
            s = x + cdig
            return 1 if s >= 4 else 0 if s == 3 else -1
        return int(np.sign(cdig - x))

    def f(m):
        a, b = _ab16(m)
        out = np.zeros(len(a))
        for r in range(len(a)):
            x, y = int(a[r]), int(b[r])
            if kind == "cmpt":
                hs = st("cmp", x ^ 2, ch ^ 2)
            else:
                hs = st(kind, x, ch)
            if cl is None:
                v = hs
            else:
                ls = st("add" if kind == "add" else "cmp", y, cl)
                v = hs if hs != 0 else ls
            out[r] = v / 4.0
        return out
    return f


for _t in (0, 1):
    TABLES[f"bset{_t}"] = ((lambda m, t=_t: (_dc(m).astype(np.int64) | (1 << t)) / 8.0), False)
    TABLES[f"bclr{_t}"] = ((lambda m, t=_t: (_dc(m).astype(np.int64) & (3 - (1 << t))) / 8.0), False)
    TABLES[f"binv{_t}"] = ((lambda m, t=_t: (_dc(m).astype(np.int64) ^ (1 << t)) / 8.0), False)
    TABLES[f"bext{_t}"] = ((lambda m, t=_t: ((_dc(m).astype(np.int64) >> t) & 1) / 8.0), False)

TABLES.update({
    "sgnot": (lambda m: (lambda b, t: np.where(t >= 2, 3 - b, 0))(*_ab16(m)) / 8.0, False),   # s_a ? ~b_i : 0
    "sbit": (lambda m: (lambda x, t: (t >= 2).astype(float))(*_ab16(m)) / 8.0, False),
    "bmask": (lambda m: (lambda k: np.where((k % 6 == 0) & (k < 12), 1 + k // 6, 0))(_k16(m)) / 8.0, False),   # S + 6 b0
    "bvset": (lambda m: (lambda a, mm: a | mm)(*_ab16(m)) / 8.0, False),
    "bvclr": (lambda m: (lambda a, mm: a & (3 - mm))(*_ab16(m)) / 8.0, False),
    "bvinv": (lambda m: (lambda a, mm: a ^ mm)(*_ab16(m)) / 8.0, False),
    "bvext": (lambda m: (lambda a, mm: ((a & mm) > 0).astype(float))(*_ab16(m)) / 8.0, False),
    "nzb3": (lambda m: np.where(_k16(m) >= 1, 0.375, 0.0), False),           # a byte (4 digits, sum <= 12) != 0
    "selz_pm": (lambda m: np.where(_k16(m) == 0, 0.5, -0.5), False),         # +-1/2: the sum is 0 (whole torus)
    "clm0": (lambda m: (lambda a, b: (a & 1) * (b & 1))(*_ab16(m)) / 8.0, False),
    "clm1": (lambda m: (lambda a, b: ((a & 1) * (b >> 1)) ^ ((a >> 1) * (b & 1)))(*_ab16(m)) / 8.0, False),
    "clm2": (lambda m: (lambda a, b: (a >> 1) * (b >> 1))(*_ab16(m)) / 8.0, False),
    "par15": (lambda m: np.mod(_k16(m), 2) / 8.0, False),                    # parity of a sum <= 15
    "blo": (lambda m: (_dc(m).astype(np.int64) & 1) / 8.0, False),
    "bhi": (lambda m: (_dc(m).astype(np.int64) >> 1) / 8.0, False),
    "blo6": (lambda m: (_dc(m).astype(np.int64) & 1) * 0.75, False),
    "sext3": (lambda m: np.where(_dc(m) >= 2, 0.75, 0.0), False),            # the fill digit 3 (at d / 4) when d >= 2
})

for _p, _f in _PRED.items():
    TABLES[_p + "_w"] = (lambda m, f=_f: np.where(f(_r4(m)), 1.0, 0.0), True)
    TABLES[_p + "1"] = (lambda m, f=_f: np.where(f(_r4(m)), 1.0, 0.0), False)
    TABLES[_p + "_4"] = (lambda m, f=_f: np.where(f(_r4(m)), 0.25, 0.0), False)
    TABLES[_p + "_pm8x"] = (lambda m, f=_f: np.where(f(_r8(m)), 0.125, -0.125), False)
    TABLES[_p + "_pm1x"] = (lambda m, f=_f: np.where(f(_r8(m)), 0.5, -0.5), False)
    TABLES[_p + "_pm8"] = (lambda m, f=_f: np.where(f(_r4(m)), 0.125, -0.125), False)
    TABLES[_p + "_pm1"] = (lambda m, f=_f: np.where(f(_r4(m)), 0.5, -0.5), False)


# ── the noise model and the device cases ──────────────────────────────────────────────────────────────────────────

# a lookup's input noise (native units, period 4) at the shipped parameters (n_lwe 840, n_br 2048, sigma 2^-18 p0), the
# model pc-check tfhe-noise validates on the device (within 3 %): the mean-compensated mod switch to 2 nu n_br / k (k
# interleaved tables: k times coarser; (n_lwe / 48 + 1 / 12) in 2 nu n_br / k units) and the key switch s_br -> s_lwe
# (key noise n_br sum E d^2 sigma^2, NAF digits; its rounding hw / 12 2^-32, hw = 1365 the ternary s_br's weight)
MS_SD = {nu: (840 / 48.0 + 1.0 / 12.0) ** 0.5 * 4.0 / (2 * nu * 2048) for nu in (1, 2)}
KS_SD = ((2048 * 5.44) ** 0.5 * 2.0 ** -18 * 4.0) ** 2 + ((1365 / 12.0) ** 0.5 * 2.0 ** -16 * 4.0) ** 2
KS_SD = KS_SD ** 0.5
LOOKUP_SD = {nu: (MS_SD[nu] ** 2 + KS_SD ** 2) ** 0.5 for nu in (1, 2)}      # 4.45e-3 / 2.69e-3


def lookup_sd(nu=1, k=1):
    """A lookup's input noise sd with k interleaved tables (many-LUT)."""
    return ((k * MS_SD[int(nu)]) ** 2 + KS_SD ** 2) ** 0.5


# a PBS output's noise sd bound: measured rms 2.96e-4 at the latency key's groups of 5 (model 2.78e-4; the batch
# key's groups of 3: model 1.97e-4); groups of 6 / 7 (4.5e-4 / 4.8e-4) need `analyze(.., sp=5e-4)`
PBS_OUT_SD = 3.2e-4


def _log2_tail2(z):
    """log2 P(|N(0, 1)| > z)."""
    import math
    if z < 8.0:
        return math.log2(math.erfc(z / math.sqrt(2.0)))
    ln = -z * z / 2.0 - math.log(z) - 0.5 * math.log(2.0 * math.pi) + math.log1p(-1.0 / z**2 + 3.0 / z**4)
    return 1.0 + ln / math.log(2.0)


def _log2_tail2_v(z):
    """`_log2_tail2` on an array (scipy's log_ndtr when available)."""
    try:
        from scipy.special import log_ndtr
        return 1.0 + log_ndtr(-np.asarray(z, dtype=np.float64)) / np.log(2.0)
    except ImportError:
        return np.array([_log2_tail2(float(v)) for v in np.asarray(z).reshape(-1)])


def _boundaries(f, pts):
    """The decision boundaries of a native table on the whole torus [-2, 2) (negacyclic beyond [-1, 1)): midpoints
    between neighbouring grid points of different output."""
    step = 2.0 / pts
    g = -2.0 + step * np.arange(2 * pts)
    v = tfhe._nat(f, g, pts)
    d = np.abs(np.roll(v, -1) - v) > 1e-9
    return (g[d] + step / 2.0)


def analyze(prog, m_in, nu=None, sp=PBS_OUT_SD):
    """Rounds, PBS count and p_fail per lookup of a native program on the noiseless loaded values `m_in` (every row of
    every lookup: the margin from its noiseless input to the table's nearest decision boundary on the torus, against
    sqrt(LOOKUP_SD^2 + the lins' propagated variance)). `rounds`: the PBS depth up to the store (not counting it);
    `launches`: pbs steps (+ the store's table)."""
    import math
    nu = tfhe.GRID // 2048 if nu is None else int(nu)
    sd = LOOKUP_SD[int(nu)]
    pts = tfhe.GRID if int(nu) == int(tfhe.GRID // 2048) else 2048 * int(nu)
    B, V, D = {}, {}, {}
    worst, tot, pbs, launches, store_rounds = -math.inf, -math.inf, 0, 0, 0
    bnd = {}

    def look(lut, x, var):
        nonlocal worst, tot
        fs, _ = prog.luts[lut]
        k = len(fs)
        if lut not in bnd:
            bnd[lut] = np.unique(np.concatenate([_boundaries(f, pts // k) for f in fs]))
        b = bnd[lut]
        sdk = lookup_sd(nu, k) if pts == 2048 * int(nu) else sd
        xm = ((x + 2.0) % 4.0) - 2.0
        dist = np.abs(xm[:, None] - b[None, :])
        dist = np.minimum(dist, 4.0 - dist).min(1) if b.size else np.full(len(x), 4.0)
        z = dist / np.sqrt(sdk * sdk + var)
        lp = _log2_tail2_v(z)
        m = float(lp.max())
        worst = max(worst, m)
        here = m + math.log2(float(np.sum(2.0 ** (lp - m))))
        tot = here if tot == -math.inf else max(tot, here) + math.log2(1.0 + 2.0 ** (-abs(tot - here)))
        return [tfhe._nat(f, x, pts // k) for f in fs]

    for st in prog.steps:
        op = st["op"]
        if op == "load":
            fs = prog.luts[st["lut"]][0]
            for d, f in zip(st["dst"] if isinstance(st["dst"], list) else [st["dst"]], fs):
                B[d] = tfhe._nat(f, np.asarray(m_in, dtype=np.float64), pts // len(fs))
                V[d] = np.full(len(B[d]), sp * sp)
                D[d] = 0
        elif op == "lin":
            idx = np.asarray(st["idx"], dtype=np.int64).reshape(-1, 2)
            a, b = B.get(st["a"], np.zeros(0)), B.get(st["b"], np.zeros(0))
            va, vb = V.get(st["a"], np.zeros(0)), V.get(st["b"], np.zeros(0))
            ia, ib = idx[:, 0] >= 0, idx[:, 1] >= 0
            ta = np.where(ia, a[np.maximum(idx[:, 0], 0)] if a.size else 0.0, 0.0)
            tb = np.where(ib, b[np.maximum(idx[:, 1], 0)] if b.size else 0.0, 0.0)
            wa = np.where(ia, va[np.maximum(idx[:, 0], 0)] if va.size else 0.0, 0.0)
            wb = np.where(ib, vb[np.maximum(idx[:, 1], 0)] if vb.size else 0.0, 0.0)
            da = D.get(st["a"], 0) if ia.any() else 0
            db = D.get(st["b"], 0) if ib.any() else 0
            B[st["dst"]] = st["ca"] * ta + st["cb"] * tb + st["c"]
            V[st["dst"]] = st["ca"] ** 2 * wa + st["cb"] ** 2 * wb
            D[st["dst"]] = max(da, db)
        elif op == "pbsm":
            x = B[st["src"]][:st["rows"]]
            var = V[st["src"]][:st["rows"]]
            y = np.zeros(len(x))
            for ln in set(st["luts"]):
                sel = np.array([l == ln for l in st["luts"]])
                y[sel] = look(ln, x[sel], var[sel])[0]
            pbs += st["rows"]
            launches += 1
            B[st["dst"]], V[st["dst"]], D[st["dst"]] = y, np.full(len(y), sp * sp), D[st["src"]] + 1
        elif op in ("pbs", "store"):
            x = B[st["src"]][:st["rows"]]
            var = V[st["src"]][:st["rows"]]
            if op == "store":
                store_rounds = D[st["src"]]
                if st["lut"] is not None:
                    look(st["lut"], x, var)
                    launches += 1
                continue
            ys = look(st["lut"], x, var)
            pbs += st["rows"]
            launches += 1
            for d, y in zip(st["dst"], ys):
                B[d] = y
                V[d] = np.full(len(y), sp * sp)
                D[d] = D[st["src"]] + 1
        else:
            raise ValueError(f"analyze: step {op}")
    return {"rounds": int(store_rounds), "pbs": int(pbs), "launches": int(launches),
            "log2_pfail_lookup": worst, "log2_pfail_op": tot, "lookup_sd": sd, "pbs_out_sd": sp}


def load_values(slot_values):
    """The native values a load leaves (the load table on the slot values)."""
    return _load(np.asarray(slot_values, dtype=np.float64))


# ── the op registry (pc-check cases, the mathlib task, the mirror tests) ──────────────────────────────────────────
#
# op names: add sub neg addf subf (+ carry / borrow and overflow rows) | lt le gt ge eq ne, s-prefixed signed (slt ..)
# | sext<k> | shl<k> shr<k> sar<k> rotl<k> rotr<k> (public k) | shlv shrv sarv rotlv rotrv (amount = b) | mul mulw
# mulc<c> | min max smin smax abs popcount | mux xor and or (the round-2 ops) | id

from .intops import PREDS as _PREDS, inputs, parse, reference  # noqa: E402,F401  (the shared op registry)


# measured cost of one bootstrap launch by its rows (RTX PRO 6000, latency keys of 5 / 4 key bits; pc-check tfhe-prog,
# 4 chained rounds of B rows, 2026-11-22; up to 40 rows the key switch regenerates its seeded masks in the kernel):
# the cost model `est_ms` uses to pick between two exact forms of an op
COST_MS = ((1, 1.12), (4, 1.13), (8, 1.20), (12, 1.24), (16, 1.50), (24, 1.64), (32, 1.69), (40, 1.89), (46, 2.04),
           (48, 2.29), (56, 2.31), (64, 2.33), (72, 2.33), (80, 2.54), (88, 2.65), (92, 2.74), (94, 3.17), (128, 4.54),
           (188, 4.64), (256, 6.54), (376, 6.54), (512, 11.0), (752, 13.1), (1024, 19.4))


def est_ms(prog):
    """The latency model of a program: the launches of every round (pbs / pbsm steps by their PBS depth) at COST_MS."""
    D, tot = {}, 0.0
    for st in prog.steps:
        op = st["op"]
        if op == "load":
            for d in (st["dst"] if isinstance(st["dst"], list) else [st["dst"]]):
                D[d] = 0
        elif op == "lin":
            D[st["dst"]] = max(D.get(st["a"], 0), D.get(st["b"], 0), D.get(st["dst"], 0) if st["dst"] in (st["a"], st["b"]) else 0)
        elif op in ("pbs", "pbsm"):
            r = st["rows"]
            tot += next((t for b, t in COST_MS if r <= b), COST_MS[-1][1] * r / COST_MS[-1][0])
            for d in (st["dst"] if isinstance(st["dst"], list) else [st["dst"]]):
                D[d] = D.get(st["src"], 0) + 1
    return tot


def _divc_long_wins(wp, d, want):
    """Whether the public-divisor long division (few quotient bits for large d) beats the magic multiply."""
    t = {}
    for form in ("long", "magic"):
        x = WordProg("t", wp.w, wp.nd)
        x.multi, x.fan4 = wp.multi, wp.fan4
        x._divc_form = form
        x.divc("t.o", d, want)
        t[form] = est_ms(x.pr)
    return t["long"] < t["magic"]


def build(op, name, words, nd):
    """A WordProg computing `op` on `words` words of nd digits (inputs: a, b, sel as `WordProg.rows_in(True)`)."""
    wp = WordProg(name, words, nd)
    build_into(wp, op, name + ".out")
    if not hasattr(wp, "out_rows"):
        wp.out_rows = wp.pr.steps[-1]["rows"]
    compact(wp.pr)
    return wp


def build_into(wp, op, out=None):
    """Append `op`'s steps to the WordProg `wp` (its store into slot set `out`)."""
    base, k = parse(op)
    if base.startswith("w32"):
        return wp.word32(out, base[3:] + ("" if k is None else str(k)))
    if base == "add":
        wp._adder("add", out)
    elif base in ("sub", "neg"):
        getattr(wp, base)(out)
    elif base == "addf":
        wp.add_flags(out)
    elif base == "subf":
        wp.sub(out, flags=True)
    elif base in ("eq", "ne", "seq", "sne"):
        wp.eq_ne(out, ne=base.endswith("ne"))
    elif base in _PREDS or (base[:1] == "s" and base[1:] in _PREDS):
        wp.cmp_pred(out, base.lstrip("s") if base not in _PREDS else base, signed=base not in _PREDS)
    elif base == "cmp":
        wp.cmp_pred(out, "lt")
    elif base == "sext":
        wp.sext(out, k)
    elif base in ("shl", "shr", "sar", "rotl", "rotr"):
        wp.shift(out, base, k)
    elif base in ("rotlv", "rotrv"):
        wp.rotate_enc(out, base[:-1])
    elif base in ("rotlvb", "rotrvb"):               # the AND-mux barrel shifter
        wp.shift_enc(out, base[:-2])
    elif base.endswith("v") and base[:-1] in ("shl", "shr", "sar", "rotl", "rotr"):
        wp.shift_enc(out, base[:-1])
    elif base in ("mul", "mulw"):
        wp.mul2(out, wide=base == "mulw")
    elif base in ("mulb", "mulwb"):                 # the bit-product form (and(bit_j(b), a))
        wp.mul(out, wide=base == "mulwb")
    elif base == "mulc":
        wp.mulc2(out, k)
    elif base == "mulcb":
        wp.mulc(out, k)
    elif base in ("divc", "remc", "divremc"):
        wp.divc(out, k, {"divc": "div", "remc": "rem", "divremc": "divrem"}[base])
    elif base in ("min", "max", "smin", "smax"):
        wp.minmax(out, base.lstrip("s"), signed=base.startswith("s"))
    elif base == "abs":
        wp.abs(out)
    elif base == "popcount":
        wp.popcount2(out)
    elif base == "popcountb":
        wp.popcount(out)
    elif base in ("clz", "ctz", "log2"):
        wp.count_op(out, base)
    elif base in ("div", "rem", "divrem"):
        wp.divrem(out, base)
    elif base in ("sdiv", "srem", "sdivrem"):
        wp.sdivrem(out, base)
    elif base in ("andi", "ori", "xori"):
        wp.bitwise_imm(out, base[:-1], k)
    elif base == "addi":
        wp.add_imm(out, k)
    elif base in ("lti", "slti", "eqi"):
        wp.cmp_imm(out, base[:-1], k)
    elif base in ("bset", "bclr", "binv", "bext"):
        wp.bit_op(out, base[1:], k)
    elif base in ("bsetv", "bclrv", "binvv", "bextv"):
        wp.bit_op(out, base[1:-1])
    elif base in ("shadd", "shadduw"):
        wp.shadd(out, k, uw=base == "shadduw")
    elif base == "adduw":
        wp.shadd(out, 0, uw=True)
    elif base in ("orcb", "rev8", "zext16"):
        wp.byte_op(out, base)
    elif base in ("czeqz", "cznez"):
        wp.czero(out, nez=base == "cznez")
    elif base in ("mulh", "mulhsu", "mulhu"):
        wp.mul2(out, high={"mulh": "ss", "mulhsu": "su", "mulhu": "u"}[base])
    elif base in ("clmul", "clmulh", "clmulr"):
        wp.clmul(out, {"clmul": "lo", "clmulh": "hi", "clmulr": "rev"}[base])
    elif base == "mux":
        wp.mux(out)
    elif base in ("xor", "and", "or", "andn", "orn", "xnor"):
        wp.bitwise(base, out)
    elif base == "id":
        wp.ident(out)
    else:
        raise ValueError(f"tfhe_bitwise: op {op}?")


def compact(prog):
    """Map a load / lin / pbs / store program's buffers onto as few as their live ranges allow (linear scan; a step's
    new buffer never takes one the step reads, so only a buffer's own in-place lins share rows)."""
    def uses(st):
        op = st["op"]
        if op == "lin":
            return [st["a"], st["b"]], [st["dst"]]
        if op == "pbs":
            return [st["src"]], list(st["dst"])
        if op == "pbsm":
            return [st["src"]], [st["dst"]]
        if op == "store":
            return [st["src"]], []
        if op == "load":
            return [], st["dst"] if isinstance(st["dst"], list) else [st["dst"]]
        raise ValueError(f"compact: step {op}")
    last = {}
    for i, st in enumerate(prog.steps):
        rd, wr = uses(st)
        for v in rd + wr:
            last[v] = i
    phys, free, nphys = {}, [], 0
    for i, st in enumerate(prog.steps):
        rd, wr = uses(st)
        busy = {phys[v] for v in rd if v in phys}
        for v in wr:
            if v not in phys:
                pick = next((f for f in sorted(free) if f not in busy), None)
                if pick is None:
                    pick, nphys = nphys, nphys + 1
                else:
                    free.remove(pick)
                phys[v] = pick
        for v in set(rd + wr):
            if last[v] == i and v in phys and phys[v] not in free:
                free.append(phys[v])
        op = st["op"]
        if op == "lin":
            # a read of a buffer never written (a lin's unused operand): any physical buffer
            st["a"], st["b"] = (phys.get(st["a"], 0), phys.get(st["b"], 0))
            st["dst"] = phys[st["dst"]]
        elif op == "pbs":
            st["src"], st["dst"] = phys[st["src"]], [phys[d] for d in st["dst"]]
        elif op == "pbsm":
            st["src"], st["dst"] = phys[st["src"]], phys[st["dst"]]
        elif op == "store":
            st["src"] = phys[st["src"]]
        else:
            st["dst"] = [phys[d] for d in st["dst"]] if isinstance(st["dst"], list) else phys[st["dst"]]
    prog.bufs = max(1, nphys)
    return prog


def out_layout(op, nd):
    """(digits per word of the word result (0: none), number of 0 / 1 rows per word after them, value scale): how a
    program's store rows read (digits at d / 4 then flags at 1 / 4, or one predicate per word at 1)."""
    base, _ = parse(op)
    if base in ("addf", "subf"):
        return nd, 2, 4.0
    if base == "cmp" or base in _PREDS or (base[:1] == "s" and base[1:] in _PREDS) or base in ("lti", "slti", "eqi"):
        return 0, 1, 1.0
    if base in ("mulw", "mulwb", "divrem", "sdivrem", "divremc"):
        return 2 * nd, 0, 4.0
    if base in ("xor", "and", "or"):
        return nd, 0, 4.0
    return nd, 0, 4.0


def want_rows(op, a, b, sel, nd):
    """The exact store rows (integers in the store's units: digits, then 0 / 1 flags)."""
    ref = reference(op, a, b, sel, 2 * nd)
    ndig, nfl, _ = out_layout(op, nd)
    w = len(np.asarray(a))
    rows = []
    if ndig:
        v = ref[0]
        v = [int(x) for x in np.asarray(v).reshape(-1)]
        rows.append(np.array([[(x >> (2 * i)) & 3 for i in range(ndig)] for x in v], dtype=np.int64).reshape(-1))
        rows += [np.asarray(f).astype(np.int64) for f in ref[1:1 + nfl]]
    else:
        rows.append(np.asarray(ref[0]).astype(np.int64))
    return np.concatenate([r.reshape(-1) for r in rows]).astype(np.int64), ndig, nfl, w


def decode_rows(op, vals, nd, words):
    """Store rows (values) -> the integer rows `want_rows` gives (digits mod 4)."""
    ndig, nfl, sc = out_layout(op, nd)
    x = np.rint(np.asarray(vals, dtype=np.float64)[:words * ndig + words * nfl] * sc).astype(np.int64)
    if ndig:
        x[:words * ndig] = np.mod(x[:words * ndig], 4)
    return x


def slot_values(a, b, sel, nd):
    """The loaded slot set's values (a's digits, b's, sel) as the CKKS input encodes them."""
    return np.concatenate([encode_digits(a, nd), encode_digits(b, nd), (4.0 * np.asarray(sel) - 3.0) / 4.0])


def case(op, bits, words, seed=0, nu=None):
    """A pc-check tfhe-prog case: the program, its loaded native values, the exact store rows, the analysis."""
    import json
    nd = bits // 2
    a, b, sel = inputs(words, bits, seed, op)
    wp = build(op, f"c.{op}{bits}.w{words}", words, nd)
    m = load_values(slot_values(a, b, sel, nd))
    want, ndig, nfl, w = want_rows(op, a, b, sel, nd)
    an = analyze(wp.pr, m, nu=nu)
    prog = json.loads(wp.pr.json())
    return {"name": f"{op}{bits} x {words}", "op": op, "bits": bits, "words": words, "prog": prog,
            "bufs": wp.pr.bufs, "rows": max(rows_needed(wp.pr, len(m)), len(m)), "m_in": [float(v) for v in m],
            "want": [int(v) for v in want], "scale": out_layout(op, nd)[2], "ndig": ndig, "nflag": nfl,
            "analysis": an}


def rows_needed(prog, rows_in):
    """The largest buffer a program writes (rows)."""
    r = rows_in
    for st in prog.steps:
        if st["op"] == "lin":
            r = max(r, len(st["idx"]) // 2)
        elif st["op"] in ("pbs", "store", "pbsm"):
            r = max(r, st["rows"])
    return r


def mirror(op, bits, words, seed=0, noise=None):
    """Run `op` on the numpy mirror (with the native noise model unless noise=0): (got rows, want rows, analysis)."""
    nd = bits // 2
    a, b, sel = inputs(words, bits, seed, op)
    wp = build(op, f"m.{op}{bits}", words, nd)
    m = slot_values(a, b, sel, nd)
    out = wp.pr.run(m, noise=LOOKUP_SD[1] if noise is None else noise)
    want = want_rows(op, a, b, sel, nd)[0]
    return decode_rows(op, out, nd, words), want, analyze(wp.pr, load_values(m))


def main(argv=None):
    """python -m perseus.impl.tfhe_bitwise --cases FILE --ops add,sub --bits 8,16,32 --words 1,16,256: write the
    pc-check tfhe-prog cases (JSON lines)."""
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--ops", required=True)
    ap.add_argument("--bits", default="8,16,32")
    ap.add_argument("--words", default="1,16,256")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    with open(a.cases, "w") as fh:
        for op in a.ops.split(","):
            for bits in map(int, a.bits.split(",")):
                for words in map(int, a.words.split(",")):
                    c = case(op, bits, words, a.seed)
                    fh.write(json.dumps(c) + "\n")
                    an = c["analysis"]
                    print(f"{c['name']}: {an['rounds']} rounds, {an['pbs']} PBS, {an['launches']} launches, p_fail "
                          f"2^{an['log2_pfail_lookup']:.1f} / lookup, 2^{an['log2_pfail_op']:.1f} / op")


if __name__ == "__main__":
    main()
