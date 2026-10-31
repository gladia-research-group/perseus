//! Seam `ModEval`, implementation: the EvalMod program executor. The program is `pc_shadow::evalmod` (FIDESlib
//! btsEvalMod / approxModReduction / evalChebyshevSeries with lazy Paterson-Stockmeyer / double angle replayed on
//! metadata, device ops recorded with their integer constants), peephole-fused (`EmProgram::fused`); every op runs
//! once for a batch of `kk` ciphertexts. Complex payload (dense route): Re and Im chains as ONE batch of 2K when it
//! fits a launch; real payload (`Route::real`): one chain. Relinearizations use `KskSource::Regen` or `Cached`;
//! rescales are composite one-pass kernels with a preceding MulConst folded in (exact); a Regen relinearization the
//! program rescales next has its ModDown fused with that rescale (`RelinRescale`). Details: docs/ARCHITECTURE.md
//! "pc-gpu EvalMod".
use super::cts::{md_plan, md_plan_bytes, MdPlan};
use super::BtsCx;
use crate::context::{Context, DevStrategy, DevWord};
use crate::error::{shape, Result};
use crate::ks::{self, DotIo, DotRows, KeyGroup, KskSource, View};
use crate::ntt::{self, Dir, RowList};
use crate::poly::{CtBuf, Layout};
use pc_core::modmul::{invmod, mulmod, shoup};
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, Lane, Stream};
use pc_kernels::{AddConstArgs, EwArgs, MdEpiArgs, MulTableArgs, PolyRef, RescaleArgs, TensorArgs, WsumArgs, WsumKArgs, MdEpiToArgs};
use pc_shadow::evalmod::{build_evalmod, build_evalmod_opts, EmOp, EmProgram, Epi, PsPlan};
use pc_shadow::fides::FMeta;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::ffi::c_void;

/// Which EvalMod.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmMode {
    /// Re / Im chains (FIDESlib approxModReduction).
    Complex,
    /// One chain on x + conj(x) (approxModReductionReal).
    Real,
}

/// Composite rescale tables of one top (composite one-pass rescale plan) and row lists for kk_max cts.
struct RescalePlan<'a, W: DevWord> {
    top: usize,
    /// primes dropped (the chain's composite degree, 1 or 2)
    d: usize,
    a: ScalarPair<'a, W>,
    b: ScalarPair<'a, W>,
    alpha: ScalarPair<'a, W>,
    qam: ConstBuf<'a, W>,
    qbm: ConstBuf<'a, W>,
    qinv_ab: u64,
    c_ab: u64,
    /// alpha_i by row (host, for folds)
    alpha_host: Vec<u64>,
    intt: RowList<'a>,
    ntt: RowList<'a>,
}

struct ScalarPair<'a, W: DevWord> {
    v: ConstBuf<'a, W>,
    s: ConstBuf<'a, W>,
}

impl<'a, W: DevWord> ScalarPair<'a, W> {
    fn new<S: Lane>(arena: &'a Arena, s: &Stream<S>, primes: &[u64], vals: &[u64]) -> Result<Self> {
        let (v, sh): (Vec<W>, Vec<W>) = vals
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let p = primes[i % primes.len()];
                let c = shoup(W::from_u64(x % p), W::from_u64(p));
                (c.w, c.w_prime)
            })
            .unzip();
        Ok(ScalarPair { v: ConstBuf::new(arena, "bts.em", s, &v)?, s: ConstBuf::new(arena, "bts.em", s, &sh)? })
    }
}

/// The relinearization key's tables for the fused Q-row regen dot with the program's x2 / + constant / + buffer
/// epilogue (`ks_dot_regen` with `EpiArgs::dbl/k` and `add`; `ks::dot_regen` itself fixes them to 0): the seed and
/// either dense prime-id `b` rows or the key's in-place table (packed / banded `b`).
struct RegenEpi<'a> {
    kb: ConstBuf<'a, u64>,
    autok: ConstBuf<'a, u32>,
    seed: *const u32,
    digit_of: ConstBuf<'a, i32>,
    /// Q rows 0..=top by top
    rows: BTreeMap<usize, ConstBuf<'a, u32>>,
    key_digit_stride: u64,
    /// packed / banded relin key: its `b` read in place (ABI 7); default (null) = the dense rows of `kb`
    key: pc_kernels::KeyTab,
}

/// The ModDown of a Regen relinearization at `top` fused with the rescale the program runs right after it (`Mult;
/// Rescale` of the product, with the program's x2 and + k): the extended product is `V = P c + dot + P k` (c the
/// tensor and dot the key inner product, both doubled by the tensor on x2; k the raw constant), and the result is
/// round(V / (P q_a q_b)) = (V - conv) (P q_a q_b)^-1 with conv the exact centred conversion of V from P and the two
/// tops (the CtS stages' fused end), i.e. `(c + k) (q_a q_b)^-1 + (dot - conv) (P q_a q_b)^-1` on rows 0..top-2: one
/// rounding (<= 1/2), as the record's relinearize, x2, + k, rescale up to its ModDown's approximation. One NTT of the
/// remaining rows instead of the ModDown's and the rescale's.
struct RelinRescale<'a, W: DevWord> {
    /// dot rows: top - 1, top, then the special rows
    rows: ConstBuf<'a, u32>,
    /// INTT of those rows of `ks.ext`, the exact conversion into rows 0..top-2 of `ks.conv`, their NTT
    md: MdPlan<'a>,
    /// (q_a q_b)^-1 and (P q_a q_b)^-1 by prime id
    alpha: ScalarPair<'a, W>,
    beta: ScalarPair<'a, W>,
    /// wsum weights of `P c + dot` on the two tops: P mod q_(top-1), P mod q_top, 1, 1
    wtop: ScalarPair<'a, W>,
}

/// A constant k of the fused relinearization-rescale at one top: k (q_a q_b)^-1 by prime id (rows 0..top-2, the
/// epilogue's part-0 term) and P k on the two tops (the extended product's).
struct RrConst<'a, W: DevWord> {
    q: ConstBuf<'a, W>,
    tops: ConstBuf<'a, W>,
}

/// The program's `Mult; Rescale` pairs `RelinRescale` serves: (top, the epilogue's constant index) per pair. Its
/// conversion drops two primes (q_a q_b): chains with a level unit `d` other than 2 get none.
fn relin_rescale_pairs(ops: &[EmOp], d: usize) -> Vec<(usize, Option<usize>)> {
    if d != 2 {
        return Vec::new();
    }
    (0..ops.len()).filter_map(|i| fused_rescale_at(ops, i).map(|(n, k, _)| (n - 1, k))).collect()
}

/// The `Mult; Rescale` (or `Mult; Add dst += / -= y; Rescale`) at op `i` that `RelinRescale` serves: (rows, the
/// epilogue's constant index, the addend (y, sub)).
fn fused_rescale_at(ops: &[EmOp], i: usize) -> Option<(usize, Option<usize>, Option<(usize, bool)>)> {
    let (dst, n, epi) = match ops.get(i)? {
        EmOp::Mult { dst, n, epi, .. } | EmOp::MultAcc { dst, n, epi, .. } => (*dst, *n, *epi),
        _ => return None,
    };
    if n < 3 {
        return None;
    }
    match (ops.get(i + 1), ops.get(i + 2)) {
        (Some(EmOp::Rescale { x, n: n2, k: None }), _) if *x == dst && *n2 == n => Some((n, epi.k, None)),
        (Some(EmOp::Add { dst: d, a, b, n: n2, sub }), Some(EmOp::Rescale { x, n: n3, k: None })) if *d == dst && *a == dst && *b != dst && *n2 == n && *x == dst && *n3 == n => {
            Some((n, epi.k, Some((*b, *sub))))
        }
        _ => None,
    }
}

/// Program values an op reads and writes.
fn reads_writes(op: &EmOp) -> (Vec<usize>, Vec<usize>) {
    match op {
        EmOp::Copy { dst, src, .. } => (vec![*src], vec![*dst]),
        EmOp::Add { dst, a, b, .. } => (vec![*a, *b], vec![*dst]),
        EmOp::MulConst { x, .. } | EmOp::AddConst { x, .. } | EmOp::Rescale { x, .. } => (vec![*x], vec![*x]),
        EmOp::Mult { dst, a, b, .. } => (vec![*a, *b], vec![*dst]),
        EmOp::MultAcc { dst, pairs, .. } => (std::iter::once(*dst).chain(pairs.iter().flat_map(|p| [p.0, p.1])).collect(), vec![*dst]),
        EmOp::WSum { dst, srcs, .. } => (srcs.clone(), vec![*dst]),
    }
}

/// Two ops whose order does not matter (neither writes what the other reads or writes).
fn commute(a: &EmOp, b: &EmOp) -> bool {
    let ((ra, wa), (rb, wb)) = (reads_writes(a), reads_writes(b));
    !wa.iter().any(|v| rb.contains(v) || wb.contains(v)) && !wb.iter().any(|v| ra.contains(v))
}

/// Value `x` is not read after op `i` before it is written again.
fn dead_after(ops: &[EmOp], i: usize, x: usize) -> bool {
    for op in &ops[i + 1..] {
        let (r, w) = reads_writes(op);
        if r.contains(&x) {
            return false;
        }
        if w.contains(&x) {
            return true;
        }
    }
    x != 0
}

/// Moves the ops between a relinearization and the rescale of its result out of the way (before the Mult, or
/// after the rescale, where they commute), keeping one `dst +/-= y` between them, so the pair runs as one fused
/// relinearization-rescale (`RelinRescale`); a `Copy z <- dst; Rescale z` with dst dead after becomes
/// `Rescale dst; Copy z <- dst`. The same values in the same buffers: without the fusion the program computes
/// what the unscheduled one does, word for word.
pub fn schedule_relin_rescales(mut ops: Vec<EmOp>, d: usize) -> Vec<EmOp> {
    let mut i = 0;
    // a transform may move other relinearizations ahead of this one: they are visited again (bounded)
    let mut budget = 4 * ops.len();
    while i < ops.len() {
        let (dst, n) = match &ops[i] {
            EmOp::Mult { dst, n, .. } | EmOp::MultAcc { dst, n, .. } if *n >= 3 => (*dst, *n),
            _ => {
                i += 1;
                continue;
            }
        };
        if fused_rescale_at(&ops, i).is_some() {
            i += 1;
            continue;
        }
        // the rescale of dst (or of its copy z, dst dead after) with only commuting ops and at most one add between
        let mut r = None;
        for j in i + 1..ops.len() {
            match &ops[j] {
                EmOp::Rescale { x, n: n2, k: None } if *x == dst && *n2 == n => {
                    r = Some((j, None));
                    break;
                }
                EmOp::Copy { dst: z, src, n: n2 } if *src == dst && *n2 == n && *z != dst => {
                    if let Some(EmOp::Rescale { x, n: n3, k: None }) = ops.get(j + 1) {
                        if x == z && *n3 == n && dead_after(&ops, j + 1, dst) {
                            r = Some((j, Some(*z)));
                        }
                    }
                    break;
                }
                op => {
                    let (rd, wr) = reads_writes(op);
                    if rd.contains(&dst) || wr.contains(&dst) {
                        let ok = matches!(op, EmOp::Add { dst: dd, a, b, n: n2, .. } if *dd == dst && *a == dst && *b != dst && *n2 == n);
                        if !ok {
                            break;
                        }
                    }
                }
            }
        }
        let Some((j, copy_to)) = r else {
            i += 1;
            continue;
        };
        let mid: Vec<usize> = (i + 1..j).collect();
        let adds: Vec<usize> = mid.iter().copied().filter(|&m| reads_writes(&ops[m]).1.contains(&dst)).collect();
        if adds.len() > 1 {
            i += 1;
            continue;
        }
        let add = adds.first().copied();
        let (mut before, mut after) = (Vec::new(), Vec::new());
        let mut ok = true;
        for &m in &mid {
            if Some(m) == add {
                continue;
            }
            let o = &ops[m];
            let past_add = add.is_some_and(|a| m > a);
            let jumps_after = after.iter().all(|&a: &usize| commute(&ops[a], o));
            if commute(&ops[i], o) && jumps_after && (!past_add || commute(&ops[add.unwrap()], o)) {
                before.push(m);
            } else if add.map_or(true, |a| m > a || commute(&ops[a], o)) && copy_to.map_or(true, |z| !reads_writes(o).0.contains(&z) && !reads_writes(o).1.contains(&z)) {
                after.push(m);
            } else {
                ok = false;
                break;
            }
        }
        if !ok {
            i += 1;
            continue;
        }
        let mut sched: Vec<EmOp> = before.iter().map(|&m| ops[m].clone()).collect();
        sched.push(ops[i].clone());
        if let Some(a) = add {
            sched.push(ops[a].clone());
        }
        match copy_to {
            None => sched.push(ops[j].clone()),
            Some(z) => {
                sched.push(EmOp::Rescale { x: dst, n, k: None });
                sched.push(EmOp::Copy { dst: z, src: dst, n: n - d });
            }
        }
        sched.extend(after.iter().map(|&m| ops[m].clone()));
        let end = if copy_to.is_some() { j + 2 } else { j + 1 };
        ops.splice(i..end, sched);
        budget = budget.saturating_sub(1);
        if budget == 0 {
            i += before.len() + 1;
        }
    }
    ops
}

/// An EvalMod program with its device constants, rescale plans and keys.
pub struct EvalModPlan<'a, W: DevWord> {
    pub mode: EmMode,
    pub ops: Vec<EmOp>,
    pub nslots: usize,
    pub in_meta: FMeta,
    pub out_meta: FMeta,
    /// constant rows by index: value + Shoup (by prime id; WSum rows are nsrc blocks of |Q|), raw (add_const)
    consts: Vec<ScalarPair<'a, W>>,
    raw: Vec<ConstBuf<'a, W>>,
    consts_host: Vec<Vec<u64>>,
    /// alpha_i c_i of the MulConst folded into a Rescale, by constant index
    fused_alpha: BTreeMap<usize, ScalarPair<'a, W>>,
    rescale: BTreeMap<usize, RescalePlan<'a, W>>,
    mono_3n2: ConstBuf<'a, W>,
    mono_n2: ConstBuf<'a, W>,
    conj: KeyGroup<'a>,
    relin: KeyGroup<'a>,
    /// Ciphertexts per op launch the plans cover.
    pub kk_max: usize,
    pub relin_source: KskSource,
    regen_epi: Option<RegenEpi<'a>>,
    /// Regen relinearizations fused with the rescale after them, by top (`RelinRescale`), and the epilogue constants
    /// rescaled for them, by (constant index, top); off: relinearize then rescale.
    rr: BTreeMap<usize, RelinRescale<'a, W>>,
    rr_k: BTreeMap<(usize, usize), RrConst<'a, W>>,
    pub fuse_rescale: bool,
    bytes: usize,
    nq: usize,
}

/// NTT of X^power on every Q prime (rows = prime id).
fn monomial_rows<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, power: usize) -> Vec<W> {
    let n = ctx.lay.n;
    let logn = n.trailing_zeros();
    let roots = ctx.chain.roots();
    ctx.chain
        .q
        .par_iter()
        .zip(roots.par_iter())
        .flat_map_iter(|(&p, &psi)| {
            let step = pc_core::modmul::powmod(psi, power as u64, p);
            let s2 = mulmod(step, step, p);
            let mut pw = vec![0u64; n];
            let mut v = step;
            for x in pw.iter_mut() {
                *x = v;
                v = mulmod(v, s2, p);
            }
            (0..n).map(move |i| W::from_u64(pw[(i as u32).reverse_bits() as usize >> (32 - logn)]))
        })
        .collect()
}

/// Program buffers referenced by `ops`.
fn slots_of(ops: &[EmOp]) -> usize {
    let mut m = 0;
    for op in ops {
        let ids: Vec<usize> = match op {
            EmOp::Copy { dst, src, .. } => vec![*dst, *src],
            EmOp::Add { dst, a, b, .. } => vec![*dst, *a, *b],
            EmOp::MulConst { x, .. } | EmOp::AddConst { x, .. } | EmOp::Rescale { x, .. } => vec![*x],
            EmOp::Mult { dst, a, b, .. } => vec![*dst, *a, *b],
            EmOp::MultAcc { dst, pairs, .. } => std::iter::once(*dst).chain(pairs.iter().flat_map(|p| [p.0, p.1])).collect(),
            EmOp::WSum { dst, srcs, .. } => std::iter::once(*dst).chain(srcs.iter().copied()).collect(),
        };
        m = ids.into_iter().fold(m, |m, i| m.max(i + 1));
    }
    m
}

fn ids_mut(op: &mut EmOp) -> Vec<&mut usize> {
    match op {
        EmOp::Copy { dst, src, .. } => vec![dst, src],
        EmOp::Add { dst, a, b, .. } => vec![dst, a, b],
        EmOp::MulConst { x, .. } | EmOp::AddConst { x, .. } | EmOp::Rescale { x, .. } => vec![x],
        EmOp::Mult { dst, a, b, .. } => vec![dst, a, b],
        EmOp::MultAcc { dst, pairs, .. } => std::iter::once(dst).chain(pairs.iter_mut().flat_map(|p| [&mut p.0, &mut p.1])).collect(),
        EmOp::WSum { dst, srcs, .. } => std::iter::once(dst).chain(srcs.iter_mut()).collect(),
    }
}

/// Buffer assignment by liveness: the builder gives every program value its own buffer; two values whose
/// live ranges [first appearance, last appearance] (op indices, inclusive) are disjoint share one. Buffer 0 is the
/// input and the output (live throughout). Ranges include the op that ends one and starts the next, so no op ever
/// reads and writes one buffer it did not before. Run on the fused ops (fusion merges neighbouring ops' operands).
pub fn assign_buffers(mut ops: Vec<EmOp>) -> (Vec<EmOp>, usize) {
    let nv = slots_of(&ops).max(1);
    let (mut first, mut last) = (vec![usize::MAX; nv], vec![0usize; nv]);
    for (i, op) in ops.iter_mut().enumerate() {
        for v in ids_mut(op) {
            first[*v] = first[*v].min(i);
            last[*v] = last[*v].max(i);
        }
    }
    first[0] = 0;
    last[0] = usize::MAX;
    let mut order: Vec<usize> = (0..nv).filter(|&v| first[v] != usize::MAX).collect();
    order.sort_by_key(|&v| (first[v], v));
    let mut map = vec![usize::MAX; nv];
    let mut busy_until: Vec<usize> = Vec::new();
    for v in order {
        let p = match (v != 0).then(|| busy_until.iter().position(|&b| b < first[v])).flatten() {
            Some(p) => p,
            None => {
                busy_until.push(0);
                busy_until.len() - 1
            }
        };
        busy_until[p] = last[v];
        map[v] = p;
    }
    for op in ops.iter_mut() {
        for v in ids_mut(op) {
            *v = map[*v];
        }
    }
    (ops, busy_until.len().max(1))
}

/// Exact executor rewrites of the fused program (same residues, fewer launches), on program value ids:
/// * `Copy d <- s` followed only by `d = d +- s` steps (n rows each): one `WSum d = m s` with m = 1 +- 1 ... as a new
///   constant row (m mod q_i; FIDESlib's double-angle `x - 3y` prologue);
/// * `Copy d <- s` where s is never referenced again (and d is not the output value 0): renamed (later ops use s).
/// Returns the ops and the constant rows extended by the new weights.
pub fn rewrite(ops: Vec<EmOp>, consts: &[Vec<u64>], q: &[u64]) -> (Vec<EmOp>, Vec<Vec<u64>>) {
    let nq = consts.first().map(|r| r.len()).unwrap_or(q.len());
    let mut consts = consts.to_vec();
    let mut out: Vec<EmOp> = Vec::with_capacity(ops.len());
    let mut i = 0;
    while i < ops.len() {
        if let EmOp::Copy { dst, src, n } = ops[i] {
            let mut m: i64 = 1;
            let mut j = i + 1;
            while let Some(EmOp::Add { dst: d, a, b, n: n2, sub }) = ops.get(j) {
                if *d != dst || *a != dst || *b != src || *n2 != n || src == dst {
                    break;
                }
                m += if *sub { -1 } else { 1 };
                j += 1;
            }
            if j > i + 1 {
                let row: Vec<u64> = (0..nq).map(|t| m.rem_euclid(q[t % q.len()] as i64) as u64).collect();
                consts.push(row);
                out.push(EmOp::WSum { dst, srcs: vec![src], n, k: consts.len() - 1 });
                i = j;
                continue;
            }
        }
        out.push(ops[i].clone());
        i += 1;
    }
    // copy renaming to a fixpoint (each pass removes one copy)
    loop {
        let mut hit = None;
        for (i, op) in out.iter().enumerate() {
            if let EmOp::Copy { dst, src, .. } = *op {
                if dst != 0 && dst != src && !out[i + 1..].iter().any(|o| ids(o).contains(&src)) {
                    hit = Some((i, dst, src));
                    break;
                }
            }
        }
        let Some((i, d, s)) = hit else { break };
        out.remove(i);
        for op in out[i..].iter_mut() {
            for v in ids_mut(op) {
                if *v == d {
                    *v = s;
                }
            }
        }
    }
    sink_copies(&mut out);
    (out, consts)
}

/// (read, written) value ids of an op (MultAcc accumulates onto dst's tensor: it reads dst).
fn rw(op: &EmOp) -> (Vec<usize>, Vec<usize>) {
    match op {
        EmOp::Copy { dst, src, .. } => (vec![*src], vec![*dst]),
        EmOp::Add { dst, a, b, .. } | EmOp::Mult { dst, a, b, .. } => (vec![*a, *b], vec![*dst]),
        EmOp::MulConst { x, .. } | EmOp::AddConst { x, .. } | EmOp::Rescale { x, .. } => (vec![*x], vec![*x]),
        EmOp::MultAcc { dst, pairs, .. } => (std::iter::once(*dst).chain(pairs.iter().flat_map(|p| [p.0, p.1])).collect(), vec![*dst]),
        EmOp::WSum { dst, srcs, .. } => (srcs.clone(), vec![*dst]),
    }
}

/// Sink every `Copy d <- s` down to the first op that writes d or s; the ops it passes read s for d. When it lands on
/// an in-place `Rescale d` / `MulConst d` / `Add d = d +- b`, the executor runs that op out of place from s (no copy).
fn sink_copies(ops: &mut Vec<EmOp>) {
    let mut i = 0;
    while i < ops.len() {
        let EmOp::Copy { dst: d, src: s, .. } = ops[i] else {
            i += 1;
            continue;
        };
        let mut j = i + 1;
        while j < ops.len() {
            let (r, w) = rw(&ops[j]);
            let _ = r;
            if w.contains(&d) || w.contains(&s) {
                break;
            }
            j += 1;
        }
        if j == i + 1 {
            i += 1;
            continue;
        }
        let copy = ops.remove(i);
        for op in ops[i..j - 1].iter_mut() {
            // these ops only read d (none writes d or s): read s instead
            for v in ids_mut(op) {
                if *v == d {
                    *v = s;
                }
            }
        }
        ops.insert(j - 1, copy);
        i = j;
    }
}

fn ids(op: &EmOp) -> Vec<usize> {
    let mut o = op.clone();
    ids_mut(&mut o).into_iter().map(|x| *x).collect()
}

impl<'a, W: DevWord> EvalModPlan<'a, W> {
    /// The program of an input with metadata `input` (FIDESlib coordinates) and its buffer count (after the
    /// liveness assignment `new` runs). The returned program's ops are already fused and rewritten (`rewrite`).
    pub fn program(st: &pc_core::scale::ScaleTables, input: FMeta, ps: &PsPlan, da_its: u32) -> Result<(EmProgram, FMeta, usize)> {
        let (mut prog, out) = build_evalmod_opts(st, input, ps, da_its as i32, true)?;
        let (ops, consts) = rewrite(prog.fused(), &prog.consts, &st.q);
        let ops = schedule_relin_rescales(ops, st.d);
        let (_, n) = assign_buffers(ops.clone());
        prog.ops = ops;
        prog.consts = consts;
        Ok((prog, out, n))
    }

    /// Device bytes `new` carves for this program (constants, rescale plans, monomials, keys; the fused regen-relin
    /// epilogue's tables: a dense seeded relinearization key, which the LT stages require anyway).
    pub fn table_bytes(st: &pc_core::scale::ScaleTables, lay: &Layout, input: FMeta, ps: &PsPlan, da_its: u32, kk_max: usize) -> Result<usize> {
        let (prog, _, _) = Self::program(st, input, ps, da_its)?;
        let np = lay.rows();
        let nq = lay.nq;
        let rows_b = |k: usize| Arena::bytes_for::<W>(k);
        let mut b = 0;
        for row in &prog.consts {
            b += 3 * rows_b(row.len());
        }
        let fused = prog.ops.clone();
        let mut tops = std::collections::BTreeSet::new();
        let mut fused_k = 0;
        for op in &fused {
            if let EmOp::Rescale { n, k, .. } = op {
                tops.insert(*n);
                fused_k += k.is_some() as usize;
            }
        }
        let rl = |k: usize| Arena::bytes_for::<u32>(k) + Arena::bytes_for::<i32>(k);
        for &t in &tops {
            let d = st.d;
            b += 6 * rows_b(np) + 2 * rows_b(np) + rl(kk_max * 2 * d) + rl(kk_max * 2 * (t - d));
        }
        b += fused_k * 2 * rows_b(nq);
        // the fused regen-relin epilogue's tables (dense seeded relin key, which the LT stages require anyway): one
        // row list per Mult top, the key pointer, its automorphism (1) and the digit map
        let mut mtops = std::collections::BTreeSet::new();
        for op in &fused {
            if let EmOp::Mult { n, .. } | EmOp::MultAcc { n, .. } = op {
                mtops.insert(*n);
            }
        }
        b += mtops.iter().map(|&t| Arena::bytes_for::<u32>(t)).sum::<usize>()
            + Arena::bytes_for::<u64>(1)
            + Arena::bytes_for::<u32>(1)
            + Arena::bytes_for::<i32>(np);
        b += 2 * rows_b(nq * lay.n) + 2 * KeyGroup::bytes(1, true);
        // the fused relinearization-rescale pairs: per top the dot rows, the conversion plan, alpha / beta / the top
        // weights, the row table of rows 0..top-2; per (constant, top) the rescaled constant
        let pairs = relin_rescale_pairs(&fused, st.d);
        let primes = st.q.iter().chain(st.p.iter()).copied().collect::<Vec<u64>>();
        let rr_tops: std::collections::BTreeSet<usize> = pairs.iter().map(|p| p.0).collect();
        let rr_k: std::collections::BTreeSet<(usize, usize)> = pairs.iter().filter_map(|&(t, k)| k.map(|k| (k, t))).collect();
        for &top in &rr_tops {
            b += Arena::bytes_for::<u32>(2 + lay.kp) + md_plan_bytes::<W>(&primes, kk_max, 2, lay.kp + 2, top - 1, true)?;
            b += 2 * 2 * rows_b(nq) + 2 * rows_b(4);
        }
        // row tables 0..=top-2 of the Q-row dots (where no Mult already has one: those are keyed by n - 1)
        let mkeys: std::collections::BTreeSet<usize> = mtops.iter().map(|&n| n - 1).collect();
        b += rr_tops.iter().map(|&t| t - 2).filter(|k| !mkeys.contains(k)).collect::<std::collections::BTreeSet<_>>().iter().map(|&k| Arena::bytes_for::<u32>(k + 1)).sum::<usize>();
        b += rr_k.len() * (rows_b(nq) + rows_b(2));
        Ok(b)
    }

    /// Build the plan. `conj`/`relin`: the conjugation and relinearization keys; `kk_max`: ciphertexts per launch.
    #[allow(clippy::too_many_arguments)]
    pub fn new<M: DevStrategy<W>>(
        arena: &'a Arena,
        s: &Stream<Compute>,
        ctx: &Context<'_, W, M>,
        mode: EmMode,
        input: FMeta,
        ps: &PsPlan,
        da_its: u32,
        conj: &dyn crate::ks::KsKeyRef,
        relin: &dyn crate::ks::KsKeyRef,
        stage_ptrs: Option<(*const c_void, *const c_void)>,
        kk_max: usize,
        relin_source: KskSource,
    ) -> Result<Self> {
        let used0 = arena.used_bytes();
        let st = &ctx.scale;
        let lay = ctx.lay;
        let primes = ctx.chain.primes();
        let nq = lay.nq;
        let (prog, out_meta, nslots) = Self::program(st, input, ps, da_its)?;
        let (ops, _) = assign_buffers(prog.ops.clone());
        let mut consts = Vec::with_capacity(prog.consts.len());
        let mut raw = Vec::with_capacity(prog.consts.len());
        for row in &prog.consts {
            consts.push(ScalarPair::new(arena, s, &primes[..nq], row)?);
            let r: Vec<W> = row.iter().map(|&x| W::from_u64(x)).collect();
            raw.push(ConstBuf::new(arena, "bts.em", s, &r)?);
        }
        let mut rescale = BTreeMap::new();
        let mut fused_alpha = BTreeMap::new();
        for op in &ops {
            if let EmOp::Rescale { n, k, .. } = op {
                let top = n - 1;
                if !rescale.contains_key(&top) {
                    rescale.insert(top, Self::rescale_plan(arena, s, ctx, top, kk_max)?);
                }
                if let Some(ki) = k {
                    let rp = &rescale[&top];
                    let c = &prog.consts[*ki];
                    let al: Vec<u64> = (0..nq).map(|i| if i + rp.d <= top { mulmod(rp.alpha_host[i], c[i] % primes[i], primes[i]) } else { 0 }).collect();
                    fused_alpha.insert(*ki, ScalarPair::new(arena, s, &primes[..nq], &al)?);
                }
            }
        }
        let n = lay.n;
        let relin_group = KeyGroup::new(arena, s, n, &[relin], stage_ptrs)?;
        let in_place = relin_group.regen_tab(0)?;
        let pairs = relin_rescale_pairs(&ops, ctx.chain.level_unit);
        let regen_epi = match (relin.staging(), relin.seed_ptr()) {
            (st, Some(seed)) if (st.is_none() && relin.digit_stride() == lay.poly_words()) || in_place.is_some() => {
                let mut rows = BTreeMap::new();
                for op in &ops {
                    if let EmOp::Mult { n, .. } | EmOp::MultAcc { n, .. } = op {
                        let top = n - 1;
                        if let std::collections::btree_map::Entry::Vacant(e) = rows.entry(top) {
                            let r: Vec<u32> = (0..=top as u32).collect();
                            e.insert(ConstBuf::new(arena, "bts.em", s, &r)?);
                        }
                    }
                }
                // the fused relinearization-rescale's Q-row dots (rows 0..=top-2)
                for &(top, _) in &pairs {
                    if let std::collections::btree_map::Entry::Vacant(e) = rows.entry(top - 2) {
                        let r: Vec<u32> = (0..=(top - 2) as u32).collect();
                        e.insert(ConstBuf::new(arena, "bts.em", s, &r)?);
                    }
                }
                let digit_of: Vec<i32> = (0..lay.rows()).map(|t| ctx.chain.digit_of(t).map(|d| d as i32).unwrap_or(-1)).collect();
                Some(RegenEpi {
                    kb: ConstBuf::new(arena, "bts.em", s, &[relin.b_ptr() as u64])?,
                    autok: ConstBuf::new(arena, "bts.em", s, &[1u32])?,
                    seed,
                    digit_of: ConstBuf::new(arena, "bts.em", s, &digit_of)?,
                    rows,
                    key_digit_stride: relin.digit_stride() as u64,
                    key: in_place.unwrap_or_default(),
                })
            }
            _ => None,
        };
        let (mut rr, mut rr_k) = (BTreeMap::new(), BTreeMap::new());
        let pmod: Vec<u64> = (0..nq).map(|i| ctx.chain.p.iter().fold(1u64, |a, &x| mulmod(a, x % primes[i], primes[i]))).collect();
        for &(top, k) in &pairs {
            if let std::collections::btree_map::Entry::Vacant(e) = rr.entry(top) {
                let specials: Vec<usize> = (nq..lay.rows()).collect();
                let mut src = specials.clone();
                src.extend([top - 1, top]);
                let r: Vec<u32> = [top - 1, top].into_iter().chain(specials).map(|t| t as u32).collect();
                let qab = |i: usize| mulmod(primes[top] % primes[i], primes[top - 1] % primes[i], primes[i]);
                let al: Vec<u64> = (0..nq).map(|i| if i + 2 <= top { invmod(qab(i), primes[i]) } else { 0 }).collect();
                let be: Vec<u64> = (0..nq).map(|i| if i + 2 <= top { invmod(mulmod(qab(i), pmod[i], primes[i]), primes[i]) } else { 0 }).collect();
                e.insert(RelinRescale {
                    rows: ConstBuf::new(arena, "bts.em", s, &r)?,
                    md: md_plan::<_, W>(arena, s, &lay, &primes, &(0..kk_max).collect::<Vec<_>>(), &[0, 1], &src, top - 1, true)?,
                    alpha: ScalarPair::new(arena, s, &primes[..nq], &al)?,
                    beta: ScalarPair::new(arena, s, &primes[..nq], &be)?,
                    wtop: ScalarPair::new(arena, s, &[primes[top - 1], primes[top]], &[pmod[top - 1], pmod[top], 1, 1])?,
                });
            }
            if let Some(ki) = k {
                if let std::collections::btree_map::Entry::Vacant(e) = rr_k.entry((ki, top)) {
                    let c = &prog.consts[ki];
                    let qab = |i: usize| mulmod(primes[top] % primes[i], primes[top - 1] % primes[i], primes[i]);
                    let q: Vec<W> = (0..nq).map(|i| if i + 2 <= top { mulmod(c[i] % primes[i], invmod(qab(i), primes[i]), primes[i]) } else { 0 }).map(W::from_u64).collect();
                    let tops: Vec<W> = [top - 1, top].iter().map(|&t| W::from_u64(mulmod(c[t] % primes[t], pmod[t], primes[t]))).collect();
                    e.insert(RrConst { q: ConstBuf::new(arena, "bts.em", s, &q)?, tops: ConstBuf::new(arena, "bts.em", s, &tops)? });
                }
            }
        }
        let plan = EvalModPlan {
            mode,
            ops,
            nslots,
            in_meta: input,
            out_meta,
            consts,
            raw,
            consts_host: prog.consts.clone(),
            fused_alpha,
            rescale,
            mono_3n2: ConstBuf::new(arena, "bts.em", s, &monomial_rows(ctx, 3 * n / 2))?,
            mono_n2: ConstBuf::new(arena, "bts.em", s, &monomial_rows(ctx, n / 2))?,
            conj: KeyGroup::new(arena, s, n, &[conj], stage_ptrs)?,
            relin: relin_group,
            kk_max,
            relin_source,
            regen_epi,
            rr,
            rr_k,
            fuse_rescale: false,
            bytes: 0,
            nq,
        };
        Ok(EvalModPlan { bytes: arena.used_bytes() - used0, ..plan })
    }

    fn rescale_plan<M: DevStrategy<W>>(arena: &'a Arena, s: &Stream<Compute>, ctx: &Context<'_, W, M>, top: usize, kk_max: usize) -> Result<RescalePlan<'a, W>> {
        let st = &ctx.scale;
        let lay = ctx.lay;
        let primes = ctx.chain.primes();
        let np = primes.len();
        let q = &ctx.chain.q;
        let d = ctx.chain.level_unit;
        if !(1..=2).contains(&d) || top < d {
            return Err(shape("evalmod rescale", format!("composite degree {d} at top {top} (the one-pass kernel drops 1 or 2)")));
        }
        let (ta, tb) = (top, top - 1);
        let (mut a, mut b, mut al) = (vec![0u64; np], vec![0u64; np], vec![0u64; np]);
        for i in 0..top + 1 - d {
            if d == 2 {
                a[i] = mulmod(st.q_inv[tb][i], st.qlql[ta][i], q[i]);
                b[i] = st.qlql[tb][i];
                al[i] = mulmod(st.q_inv[tb][i], st.q_inv[ta][i], q[i]);
            } else {
                a[i] = st.qlql[ta][i];
                al[i] = st.q_inv[ta][i];
            }
        }
        let qam: Vec<W> = primes.iter().map(|&p| W::from_u64(q[ta] % p)).collect();
        let qbm: Vec<W> = primes.iter().map(|&p| W::from_u64(q[tb] % p)).collect();
        let rows = lay.rows();
        let (mut intt, mut nt) = (Vec::new(), Vec::new());
        for c in 0..kk_max {
            for p in 0..2 {
                let base = (2 * c + p) * rows;
                if d == 2 {
                    intt.push(((base + tb) as u32, tb as i32));
                }
                intt.push(((base + ta) as u32, ta as i32));
                for i in 0..top + 1 - d {
                    nt.push(((base + i) as u32, i as i32));
                }
            }
        }
        Ok(RescalePlan {
            top,
            d,
            a: ScalarPair::new(arena, s, &primes, &a)?,
            b: ScalarPair::new(arena, s, &primes, &b)?,
            alpha: ScalarPair::new(arena, s, &primes, &al)?,
            qam: ConstBuf::new(arena, "bts.em", s, &qam)?,
            qbm: ConstBuf::new(arena, "bts.em", s, &qbm)?,
            qinv_ab: if d == 2 { st.q_inv[ta][tb] } else { 0 },
            c_ab: if d == 2 { st.qlql[ta][tb] } else { 0 },
            alpha_host: al,
            intt: RowList::new(arena, s, &intt)?,
            ntt: RowList::new(arena, s, &nt)?,
        })
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Ciphertexts per program launch for a call of k: (kk, joint).
    pub fn batch(&self, k: usize) -> (usize, bool) {
        match self.mode {
            EmMode::Complex if 2 * k <= self.kk_max => (2 * k, true),
            _ => (k, false),
        }
    }

    /// EvalMod of the first k cts of `x` (CtS output, `in_meta`): the result (`out_meta`, before EvalMod's final
    /// rescale, which the caller decides) replaces them. `em`: the program buffers (BtsWork::em; joint runs use
    /// buffers 0..nslots, split runs 0..2 nslots); `tmp`: kk_max-ct rescale workspace.
    pub fn run<M: DevStrategy<W>>(&self, cx: &mut BtsCx<'_, '_, W, M>, x: &mut CtBuf<'_, Compute, W>, k: usize) -> Result<()> {
        self.run_into(cx, x, None, k)
    }

    /// Complex EvalMod whose Re and Im halves are two real payloads (a refresh pair, `StcFirstRoute::run_pair`): the
    /// Re chain's result replaces the first k cts of `x`, the Im chain's those of `y`, both at `out_meta` (`run` would
    /// return Re + i Im).
    pub fn run_split<M: DevStrategy<W>>(&self, cx: &mut BtsCx<'_, '_, W, M>, x: &mut CtBuf<'_, Compute, W>, y: &mut CtBuf<'_, Compute, W>, k: usize) -> Result<()> {
        if self.mode != EmMode::Complex {
            return Err(shape("evalmod", "a split EvalMod needs the Re / Im chains (EmMode::Complex)"));
        }
        ks::check_cts(y, 0, k, "evalmod split")?;
        self.run_into(cx, x, Some(y), k)
    }

    fn run_into<M: DevStrategy<W>>(&self, cx: &mut BtsCx<'_, '_, W, M>, x: &mut CtBuf<'_, Compute, W>, split: Option<&mut CtBuf<'_, Compute, W>>, k: usize) -> Result<()> {
        let ctx = cx.ctx;
        let lay = ctx.lay;
        let top_in = self.in_meta.top as usize;
        let (kk, joint) = self.batch(k);
        let sets = if self.mode == EmMode::Complex && !joint { 2 } else { 1 };
        if cx.work.em.len() < sets * self.nslots || cx.work.em.iter().any(|b| b.k < kk) || kk > self.kk_max {
            return Err(shape("evalmod", format!("bts.work holds {} EvalMod buffers of {} cts, the call needs {} of {kk} (BtsWorkShape::em_*)", cx.work.em.len(), cx.work.shape.em_kk, sets * self.nslots)));
        }
        let cw = lay.ct_words();
        let pw = lay.poly_words() as u64;
        // aux = conj(x) in ks.ext (key switch with the conjugation key; its ModDown ends in md_epilogue_to)
        {
            let ex = &mut *cx.ex;
            ks::modup(ctx, &ex.compute, cx.ks, &mut ex.ks, &mut ex.bconv_work, x, 0, k, 1, top_in)?;
            let v = View::of(x, 0);
            let io = DotIo {
                c1: v.part::<W>(1).base,
                c0: v.base,
                ct_stride: v.ct_stride,
                out: ex.ks.ext.buf.vptr_mut(),
                part_stride: pw,
                out_ct_stride: cw as u64,
                out_rot_stride: 0,
                c0_mode: 1,
            };
            // the conjugation follows the relin key source (Regen: in place, `ks_dot_key`)
            let staged = self.conj.reads_in_place(self.relin_source) || self.conj.needs_stage() || !self.conj.has_a();
            if staged {
                let ndig = cx.ks.ndig(top_in)?;
                ks::stage_key(ctx, &ex.compute, cx.ks, &mut ex.ks, &self.conj, 0, ndig, true)?;
            }
            ks::dot(ctx, &ex.compute, cx.ks, &ex.ks, &self.conj, 0, 1, k, top_in, DotRows::All, io, staged)?;
            // the ModDown's last step is fused into x +- conj(x) below (md_epilogue_to)
            ks::moddown_conv(ctx, &ex.compute, cx.ks, &mut ex.ks.ext, &mut ex.ks.conv, k, top_in)?;
        }
        let s = &cx.ex.compute;
        let xv = View::of(x, 0);
        let av = View::of(&cx.ex.ks.ext, 0);
        let re = View::of(&cx.work.em[0], 0);
        let im = if self.mode == EmMode::Real {
            None
        } else if joint {
            Some(View::of(&cx.work.em[0], k))
        } else {
            Some(View::of(&cx.work.em[self.nslots], 0))
        };
        let cv = View::of(&cx.ex.ks.conv, 0);
        let conj_to = |out: View, sub: i32| -> Result<()> {
            let m = MdEpiArgs {
                dst: out.r(),
                x: av.r(),
                t: cv.r(),
                binv: ctx.ks.p_inv_mod_q.v.vptr(),
                binvs: ctx.ks.p_inv_mod_q.s.vptr(),
                k: std::ptr::null(),
                pr: ctx.pr(),
                dbl: 0,
                n: (top_in + 1) as i32,
                kk: k as i32,
                pad: 0,
            };
            let a = MdEpiToArgs { m, a: xv.r(), sub, pad: 0 };
            // SAFETY: k-ct views of live context-layout buffers; out (a program buffer) aliases none of x, ext, conv
            launch_result("md_epilogue_to", unsafe { (ctx.table.md_epilogue_to)(&a, s.raw()) })?;
            Ok(())
        };
        conj_to(re, 0)?;
        if let Some(im) = im {
            conj_to(im, 1)?;
            Self::mul_table(ctx, s, im, im, &self.mono_3n2, top_in + 1, k)?;
        }
        if joint || self.mode == EmMode::Real {
            self.exec(cx, 0, kk)?;
        } else {
            self.exec(cx, 0, k)?;
            self.exec(cx, self.nslots, k)?;
        }
        let s = &cx.ex.compute;
        let n_out = self.out_meta.top as usize + 1;
        let qo = lay.rs_q(n_out - 1);
        let re = View::of(&cx.work.em[0], 0);
        match (self.mode, split) {
            (EmMode::Real, _) => ks::ew(ctx, s, ks::Ew::Copy, View::of(x, 0), re, None, qo, k, 2)?,
            (EmMode::Complex, None) => {
                let im = if joint { View::of(&cx.work.em[0], k) } else { View::of(&cx.work.em[self.nslots], 0) };
                Self::mul_table(ctx, s, im, im, &self.mono_n2, n_out, k)?;
                ks::ew(ctx, s, ks::Ew::Add, View::of(x, 0), re, Some(im), qo, k, 2)?;
            }
            (EmMode::Complex, Some(y)) => {
                let im = if joint { View::of(&cx.work.em[0], k) } else { View::of(&cx.work.em[self.nslots], 0) };
                ks::ew(ctx, s, ks::Ew::Copy, View::of(x, 0), re, None, qo, k, 2)?;
                ks::ew(ctx, s, ks::Ew::Copy, View::of(y, 0), im, None, qo, k, 2)?;
            }
        }
        Ok(())
    }

    fn mul_table<M: DevStrategy<W>>(ctx: &Context<'_, W, M>, s: &Stream<Compute>, out: View, inp: View, t: &ConstBuf<'_, W>, n: usize, k: usize) -> Result<()> {
        let a = MulTableArgs { out: out.r(), inp: inp.r(), t: t.vptr(), pr: ctx.pr(), n: n as i32, k: k as i32 };
        // SAFETY: k-ct views; the table covers every Q row; element-wise (out may alias inp)
        launch_result("mul_table", unsafe { (ctx.table.mul_table)(&a, s.raw()) })?;
        Ok(())
    }

    /// Run the program on buffers `em[base..base + nslots]`, kk cts each.
    fn exec<M: DevStrategy<W>>(&self, cx: &mut BtsCx<'_, '_, W, M>, base: usize, kk: usize) -> Result<()> {
        let ctx = cx.ctx;
        let lay = ctx.lay;
        let pr = ctx.pr();
        let slot = |cx: &BtsCx<'_, '_, W, M>, i: usize| View::of(&cx.work.em[base + i], 0);
        let mut i = 0;
        // `Copy d <- s` absorbed by the op after it (which writes d over the same rows): that op reads s for d, and
        // its in-place operand d comes from s (`from`)
        let mut from: Option<(usize, usize)> = None;
        let mut subst: EmOp;
        while i < self.ops.len() {
            let mut op = &self.ops[i];
            if let Some((d, s0)) = from.take() {
                subst = op.clone();
                for v in ids_mut(&mut subst) {
                    if *v == d {
                        *v = s0;
                    }
                }
                // the written id stays d
                match &mut subst {
                    EmOp::Add { dst, .. } | EmOp::Mult { dst, .. } | EmOp::WSum { dst, .. } => *dst = d,
                    EmOp::MulConst { .. } | EmOp::Rescale { .. } => {}
                    _ => return Err(shape("evalmod", "absorbed copy before an unsupported op")),
                }
                match subst {
                    EmOp::MulConst { n, k, .. } => {
                        let c = &self.consts[k];
                        let args = EwArgs { out: slot(cx, d).r(), a: slot(cx, s0).r(), b: slot(cx, s0).r(), rs: lay.rs_q(n - 1), sc: c.v.vptr(), scs: c.s.vptr(), pr, k: kk as i32, nparts: 2, sub: 0, zero_p: 0 };
                        // SAFETY: kk-ct views of two distinct program buffers; element-wise
                        launch_result("ew_scale", unsafe { (ctx.table.ew_scale)(&args, cx.ex.compute.raw()) })?;
                        i += 1;
                        continue;
                    }
                    EmOp::Rescale { n, k, .. } => {
                        self.rescale_from(cx, base + d, Some(base + s0), n, k, kk)?;
                        i += 1;
                        continue;
                    }
                    _ => op = &subst,
                }
            }
            let next = self.ops.get(i + 1);
            i += 1;
            let s = &cx.ex.compute;
            match op {
                EmOp::Copy { dst, src, n } => {
                    let absorb = match next {
                        Some(EmOp::Rescale { x, n: n2, .. }) | Some(EmOp::MulConst { x, n: n2, .. }) => x == dst && n2 == n,
                        Some(EmOp::Add { dst: d, n: n2, .. }) | Some(EmOp::Mult { dst: d, n: n2, .. }) | Some(EmOp::WSum { dst: d, n: n2, .. }) => d == dst && n2 == n,
                        _ => false,
                    };
                    if absorb && dst != src {
                        from = Some((*dst, *src));
                    } else {
                        ks::ew(ctx, s, ks::Ew::Copy, slot(cx, *dst), slot(cx, *src), None, lay.rs_q(n - 1), kk, 2)?
                    }
                }
                EmOp::Add { dst, a, b, n, sub } => {
                    let args = EwArgs {
                        out: slot(cx, *dst).r(),
                        a: slot(cx, *a).r(),
                        b: slot(cx, *b).r(),
                        rs: lay.rs_q(n - 1),
                        sc: std::ptr::null(),
                        scs: std::ptr::null(),
                        pr,
                        k: kk as i32,
                        nparts: 2,
                        sub: *sub as i32,
                        zero_p: 0,
                    };
                    // SAFETY: kk-ct views of the program buffers; element-wise
                    launch_result("ew_add", unsafe { (ctx.table.ew_add)(&args, s.raw()) })?;
                }
                EmOp::MulConst { x, n, k } => {
                    let c = &self.consts[*k];
                    let v = slot(cx, *x);
                    let args = EwArgs { out: v.r(), a: v.r(), b: v.r(), rs: lay.rs_q(n - 1), sc: c.v.vptr(), scs: c.s.vptr(), pr, k: kk as i32, nparts: 2, sub: 0, zero_p: 0 };
                    // SAFETY: as above; the scalar table covers every Q row
                    launch_result("ew_scale", unsafe { (ctx.table.ew_scale)(&args, s.raw()) })?;
                }
                EmOp::AddConst { x, n, k } => {
                    let a = AddConstArgs { x: slot(cx, *x).r(), k: self.raw[*k].vptr(), pr, n: *n as i32, kk: kk as i32 };
                    // SAFETY: as above
                    launch_result("add_const", unsafe { (ctx.table.add_const)(&a, s.raw()) })?;
                }
                EmOp::Rescale { x, n, k } => self.rescale_from(cx, base + *x, None, *n, *k, kk)?,
                EmOp::Mult { dst, n, .. } | EmOp::MultAcc { dst, n, .. } => {
                    // the rescale right after the relin rides its ModDown (`RelinRescale`)
                    let regen = self.regen_epi.is_some() && self.relin_source == KskSource::Regen && self.relin.seeded();
                    // (ops[i - 1] is this op, or the copy-absorbed original at the same index)
                    if let Some((_, _, add)) = fused_rescale_at(&self.ops, i - 1) {
                        if self.fuse_rescale && regen && self.rr.contains_key(&(n - 1)) {
                            i += if add.is_some() { 2 } else { 1 };
                            match op {
                                EmOp::Mult { a, b, epi, .. } => self.relin_rescale(cx, base, *dst, &[(*a, *b)], *n, false, *epi, add, kk)?,
                                EmOp::MultAcc { pairs, epi, .. } => self.relin_rescale(cx, base, *dst, pairs, *n, true, *epi, add, kk)?,
                                _ => unreachable!(),
                            }
                            continue;
                        }
                    }
                    // `dst += b` right after the relin rides its fused epilogue (regen path; exact: one more add mod p)
                    let add = match next {
                        Some(EmOp::Add { dst: d, a, b, n: n2, sub: false }) if self.regen_epi.is_some() && self.relin_source == KskSource::Regen && d == dst && a == dst && b != dst && n2 == n => {
                            i += 1;
                            Some(*b)
                        }
                        _ => None,
                    };
                    match op {
                        EmOp::Mult { a, b, epi, .. } => self.relin(cx, base, *dst, &[(*a, *b)], *n, false, *epi, add, kk)?,
                        EmOp::MultAcc { pairs, epi, .. } => self.relin(cx, base, *dst, pairs, *n, true, *epi, add, kk)?,
                        _ => unreachable!(),
                    }
                }
                EmOp::WSum { dst, srcs, n, k } => {
                    if srcs.len() > 8 {
                        return Err(shape("evalmod wsum", format!("{} sources (kernel bound 8)", srcs.len())));
                    }
                    let mut src = [PolyRef::default(); 8];
                    for (j, &x) in srcs.iter().enumerate() {
                        src[j] = slot(cx, x).r();
                    }
                    let c = &self.consts[*k];
                    let a = WsumArgs { out: slot(cx, *dst).r(), src, w: c.v.vptr(), ws: c.s.vptr(), pr, nsrc: srcs.len() as i32, rowstride: self.nq as i32, n: *n as i32, k: kk as i32 };
                    match next {
                        // a following `dst += const` (part 0) in the same launch (wsum_k = wsum + add_const)
                        Some(EmOp::AddConst { x, n: n2, k: k2 }) if x == dst && n2 == n => {
                            i += 1;
                            let ak = WsumKArgs { w: a, k: self.raw[*k2].vptr() };
                            // SAFETY: as wsum; the constant row covers every Q row
                            launch_result("wsum_k", unsafe { (ctx.table.wsum_k)(&ak, s.raw()) })?;
                        }
                        // SAFETY: as above; the weight table holds nsrc rows of |Q|
                        _ => launch_result("wsum", unsafe { (ctx.table.wsum)(&a, s.raw()) })?,
                    }
                }
            }
        }
        Ok(())
    }

    /// Rescale buffer `buf` in place, or (`src = Some(s)`) buf = rescale(s) with s unchanged: s's top rows reach buf
    /// through their INTT (the conversion's input; copied first on a 1-prime unit), the combine reads s and writes buf
    /// (the program's `Copy; Rescale`). On a 2-prime unit the conversion and the NTT of its rows run as `rescale_fwd`.
    fn rescale_from<M: DevStrategy<W>>(&self, cx: &mut BtsCx<'_, '_, W, M>, buf: usize, src: Option<usize>, n: usize, k: Option<usize>, kk: usize) -> Result<()> {
        let ctx = cx.ctx;
        let top = n - 1;
        let rp = self.rescale.get(&top).ok_or_else(|| shape("evalmod rescale", format!("no plan for top {top}")))?;
        let s = &cx.ex.compute;
        let d = rp.d;
        let keep = top + 1 - d; // rows left
        let from = src.map(|si| cx.work.em[si].buf.vptr());
        match from {
            Some(f) if d == 2 => {
                // the source's tops, inverse transformed into buf (programs buffers share their row offsets)
                let xb = &mut cx.work.em[buf];
                let sv = ntt::Src { base: f, soff: rp.intt.roff.dptr(), n: 1, stride: 0 };
                ntt::transform_rows_prefix_ex(ctx, s, Dir::Inverse, &mut xb.buf, &rp.intt, kk * 2 * d, Some(sv), None)?;
            }
            _ => {
                if let Some(si) = src {
                    // the d tops of the source into buf (the conversion reads them there, in coefficient form)
                    let off = |v: View| View { base: (v.base as *mut W).wrapping_add(keep * ctx.lay.n) as *mut c_void, ..v };
                    ks::ew(ctx, s, ks::Ew::Copy, off(View::of(&cx.work.em[buf], 0)), off(View::of(&cx.work.em[si], 0)), None, ctx.lay.rs_q(d - 1), kk, 2)?;
                }
                ntt::transform_rows_prefix(ctx, s, Dir::Inverse, &mut cx.work.em[buf].buf, &rp.intt, kk * 2 * d)?;
            }
        }
        let xb = &mut cx.work.em[buf];
        let (ca, cb, al, als) = match k {
            Some(ki) => {
                let c = &self.consts_host[*&ki];
                let fa = &self.fused_alpha[&ki];
                (c[top], if d == 2 { c[top - 1] } else { 1 }, fa.v.vptr(), fa.s.vptr())
            }
            None => (1, 1, rp.alpha.v.vptr(), rp.alpha.s.vptr()),
        };
        let tmp = &mut cx.work.lt_tmp;
        let ra = RescaleArgs {
            x: xb.w(),
            tmp: tmp.w(),
            a: rp.a.v.vptr(),
            b: rp.b.v.vptr(),
            a_s: rp.a.s.vptr(),
            b_s: rp.b.s.vptr(),
            qam: rp.qam.vptr(),
            qbm: rp.qbm.vptr(),
            pr: ctx.pr(),
            qinv_ab: rp.qinv_ab,
            c_ab: rp.c_ab,
            ca,
            cb,
            top: rp.top as i32,
            nrows: keep as i32,
            k: kk as i32,
            ndrop: d as i32,
        };
        // buf rows 0..top-2 = alpha * (buf or the source, left as it was) + NTT(tmp): the combine rides the
        // transform's epilogue (program buffers share tmp's row offsets)
        let from = from.unwrap_or(xb.buf.vptr());
        let epi = ntt::Epi::MulAdd { out: cx.work.em[buf].buf.vptr_mut(), x: from, s: al, ss: als };
        if d == 2 {
            let ntt = ntt::forward_args(ctx, &mut cx.work.lt_tmp.buf, &rp.ntt, kk * 2 * keep, Some(epi))?;
            // SAFETY: as rescale_coef and the transform below; the prep writes the LT workspace's rows top-1, top
            // (scratch, past the kk * 2 * keep transform rows)
            launch_result("rescale_fwd", unsafe { (ctx.table.rescale_fwd)(&pc_kernels::RescaleNttArgs { rs: ra, ntt }, s.raw()) })?;
            return Ok(());
        }
        // SAFETY: kk cts of a program buffer and of the LT workspace (>= kk_max cts), context-layout tables
        launch_result("rescale_coef", unsafe { (ctx.table.rescale_coef)(&ra, s.raw()) })?;
        ntt::transform_rows_prefix_ex(ctx, s, Dir::Forward, &mut cx.work.lt_tmp.buf, &rp.ntt, kk * 2 * keep, None, Some(epi))
    }

    /// One `ks_dot_regen` launch per <= REGEN_KMAX cts of the relinearization key over `nrows` rows of `rows`: raw
    /// into `ks.ext` (`epi` None) or through the epilogue.
    #[allow(clippy::too_many_arguments)]
    fn regen_dot<M: DevStrategy<W>>(&self, cx: &mut BtsCx<'_, '_, W, M>, rows: *const u32, nrows: usize, top: usize, kk: usize, epi: Option<pc_kernels::EpiArgs>) -> Result<()> {
        let ctx = cx.ctx;
        let lay = ctx.lay;
        let re = self.regen_epi.as_ref().ok_or_else(|| shape("evalmod relin", "no in-place relinearization key tables"))?;
        let ex = &*cx.ex;
        let (pw, cw) = (lay.poly_words() as u64, lay.ct_words() as u64);
        let mu_ct = (cx.ks.dnum * lay.poly_words()) as u64;
        let ndig = cx.ks.ndig(top)?;
        let ext = View::of(&ex.ks.ext, 0);
        let mut c = 0;
        while c < kk {
            let kc = (kk - c).min(ks::REGEN_KMAX);
            let at = |p: *const c_void, stride: u64| (p as *const W).wrapping_add(c * stride as usize) as *const c_void;
            let out = at(ext.base, cw);
            let ka = pc_kernels::KsArgs {
                c1: at(ext.part::<W>(1).base, cw),
                modup: at(ex.ks.digits.vptr(), mu_ct),
                c0: std::ptr::null(),
                out0: out as *mut c_void,
                out1: (out as *const W).wrapping_add(pw as usize) as *mut c_void,
                kb: re.kb.dptr() as *const *const c_void,
                ka: re.kb.dptr() as *const *const c_void,
                autok: re.autok.dptr(),
                rows,
                digit_of: re.digit_of.dptr(),
                pmod: ctx.ks.p_mod_q.v.vptr(),
                pmods: ctx.ks.p_mod_q.s.vptr(),
                pr: ctx.pr(),
                ct_stride: cw,
                mu_ct_stride: mu_ct,
                mu_digit_stride: pw,
                out_ct_stride: cw,
                out_rot_stride: 0,
                key_digit_stride: re.key_digit_stride,
                k: kc as i32,
                r: 1,
                ndig: ndig as i32,
                c0_mode: 0,
                nq: lay.nq as i32,
                nrows: nrows as i32,
            };
            let (use_epi, e) = match epi {
                Some(e) => (1, pc_kernels::EpiArgs { dst: at(e.dst, cw) as *mut c_void, conv: at(e.conv, cw), ..e }),
                None => (0, pc_kernels::EpiArgs::default()),
            };
            let a = pc_kernels::KsRegenArgs { ks: ka, seed: re.seed, epi: e, use_epi, pad: 0, key: re.key, add: std::ptr::null() };
            // SAFETY: kc-ct views of the KS workspace and the program buffers (prefix of kk <= k_max), the key's seed
            // and stored `b`, row tables built at setup over rows of the context layout
            launch_result("ks_dot_regen", unsafe { (ctx.table.ks_dot_regen)(&a, ex.compute.raw()) })?;
            c += kc;
        }
        Ok(())
    }

    /// `dst = rescale(relin(sum of pairs' tensors) [x2] [+ const])` with the ModDown fused into the rescale
    /// (`RelinRescale`; Regen relinearization key): rows 0..top-2 of dst, degree 1.
    #[allow(clippy::too_many_arguments)]
    fn relin_rescale<M: DevStrategy<W>>(&self, cx: &mut BtsCx<'_, '_, W, M>, base: usize, dst: usize, pairs: &[(usize, usize)], n: usize, acc: bool, epi: Epi, add: Option<(usize, bool)>, kk: usize) -> Result<()> {
        let ctx = cx.ctx;
        let lay = ctx.lay;
        let top = n - 1;
        let rr = self.rr.get(&top).ok_or_else(|| shape("evalmod relin-rescale", format!("no plan for top {top}")))?;
        let re = self.regen_epi.as_ref().ok_or_else(|| shape("evalmod relin-rescale", "no in-place relinearization key tables"))?;
        let qrows = re.rows.get(&(top - 2)).ok_or_else(|| shape("evalmod relin-rescale", format!("no row table for top {}", top - 2)))?;
        let (pw, nn) = (lay.poly_words() as u64, lay.n);
        let dstv = View::of(&cx.work.em[base + dst], 0);
        {
            let ex = &mut *cx.ex;
            let v = |i: usize| View::of(&cx.work.em[base + i], 0).r();
            let (a0, b0) = (v(pairs[0].0), v(pairs[0].1));
            let (a1, b1) = if pairs.len() > 1 { (v(pairs[1].0), v(pairs[1].1)) } else { (a0, b0) };
            let ext = View::of(&ex.ks.ext, 0);
            let t = TensorArgs { out: dstv.r(), d2: ext.r(), a0, b0, a1, b1, coef: std::ptr::null_mut(), coef_ct_stride: pw, pr: ctx.pr(), npairs: pairs.len() as i32, acc: acc as i32, n: n as i32, k: kk as i32, dbl: epi.dbl as i32, pad: 0 };
            // SAFETY: as in `relin`
            launch_result("tensor", unsafe { (ctx.table.tensor)(&t, ex.compute.raw()) })?;
            // the program's `dst +/-= y` between the relinearization and the rescale: onto the tensor's (d0, d1),
            // which the conversion below rescales with the key-switched d2
            if let Some((y, sub)) = add {
                let args = EwArgs { out: dstv.r(), a: dstv.r(), b: v(y), rs: lay.rs_q(top), sc: std::ptr::null(), scs: std::ptr::null(), pr: ctx.pr(), k: kk as i32, nparts: 2, sub: sub as i32, zero_p: 0 };
                // SAFETY: kk-ct views of two distinct program buffers (y != dst, checked by `fused_rescale_at`)
                launch_result("ew_add", unsafe { (ctx.table.ew_add)(&args, ex.compute.raw()) })?;
            }
            ks::modup_from(ctx, &ex.compute, cx.ks, &mut ex.ks, &mut ex.bconv_work, ext.part::<W>(1), kk, top)?;
        }
        // 1. the dot over the two tops and the special rows, raw into ks.ext
        self.regen_dot(cx, rr.rows.dptr(), 2 + lay.kp, top, kk, None)?;
        let ex = &mut *cx.ex;
        let s = &ex.compute;
        // 2. the tops of the extended product: P c + dot [+ P k]
        let kc = match epi.k {
            Some(ki) => Some(self.rr_k.get(&(ki, top)).ok_or_else(|| shape("evalmod relin-rescale", format!("no constant {ki} at top {top}")))?),
            None => None,
        };
        let row = |v: View, r: usize| PolyRef { base: (v.base as *mut W).wrapping_add(r * nn) as *mut c_void, ct_stride: v.ct_stride, part_stride: v.part_stride };
        let ext = View::of(&ex.ks.ext, 0);
        let mut src = [PolyRef::default(); 8];
        src[0] = row(dstv, top - 1);
        src[1] = row(ext, top - 1);
        let pr = ctx.pr().wrapping_add(top - 1);
        let w = WsumArgs { out: row(ext, top - 1), src, w: rr.wtop.v.vptr(), ws: rr.wtop.s.vptr(), pr, nsrc: 2, rowstride: 2, n: 2, k: kk as i32 };
        // SAFETY: rows top-1, top of kk cts of the program buffer and of ks.ext (the prime table offset to those
        // rows, the constant's two entries with them); element-wise, out aliasing src[1] index by index
        match kc {
            Some(c) => launch_result("wsum_k", unsafe { (ctx.table.wsum_k)(&WsumKArgs { w, k: c.tops.vptr() }, s.raw()) })?,
            None => launch_result("wsum", unsafe { (ctx.table.wsum)(&w, s.raw()) })?,
        }
        // 3. coefficient form of the 11 source rows, the exact conversion to rows 0..top-2, their NTT
        rr.md.run(ctx, s, &mut ex.ks.ext.buf, &mut ex.ks.conv.buf, kk)?;
        // 4. the Q-row dot into dst = alpha c + beta (dot - conv) [+ alpha k]
        let conv = View::of(&ex.ks.conv, 0);
        let k = kc.map(|c| c.q.vptr()).unwrap_or(std::ptr::null());
        let e = pc_kernels::EpiArgs { dst: dstv.base, conv: conv.base, binv: rr.beta.v.vptr(), binvs: rr.beta.s.vptr(), k, part_stride: pw, dbl: 0, pad: 0, alpha: rr.alpha.v.vptr(), alphas: rr.alpha.s.vptr() };
        self.regen_dot(cx, qrows.dptr(), top - 1, top, kk, Some(e))
    }

    /// dst = relin(sum of pairs' tensors) [x2] [+ const] [+ buffer `add`] (Mult: one pair; MultAcc: accumulate onto
    /// dst's tensor). `add` only on the fused regen epilogue path (the caller checks).
    #[allow(clippy::too_many_arguments)]
    fn relin<M: DevStrategy<W>>(&self, cx: &mut BtsCx<'_, '_, W, M>, base: usize, dst: usize, pairs: &[(usize, usize)], n: usize, acc: bool, epi: Epi, add: Option<usize>, kk: usize) -> Result<()> {
        let ctx = cx.ctx;
        let lay = ctx.lay;
        let top = n - 1;
        let pw = lay.poly_words() as u64;
        let cw = lay.ct_words() as u64;
        let ex = &mut *cx.ex;
        let v = |i: usize| View::of(&cx.work.em[base + i], 0).r();
        let (a0, b0) = (v(pairs[0].0), v(pairs[0].1));
        let (a1, b1) = if pairs.len() > 1 { (v(pairs[1].0), v(pairs[1].1)) } else { (a0, b0) };
        let out = v(dst);
        let ext = View::of(&ex.ks.ext, 0);
        let t = TensorArgs {
            out,
            d2: ext.r(),
            a0,
            b0,
            a1,
            b1,
            coef: std::ptr::null_mut(),
            coef_ct_stride: pw,
            pr: ctx.pr(),
            npairs: pairs.len() as i32,
            acc: acc as i32,
            n: n as i32,
            k: kk as i32,
            dbl: 0,
            pad: 0,
        };
        // SAFETY: kk-ct views of the program buffers (out may be a source: element-wise, each thread reads its
        // operands before writing), d2 into the key-switching workspace
        launch_result("tensor", unsafe { (ctx.table.tensor)(&t, ex.compute.raw()) })?;
        ks::modup_from(ctx, &ex.compute, cx.ks, &mut ex.ks, &mut ex.bconv_work, ext.part::<W>(1), kk, top)?;
        let io = DotIo { c1: ext.part::<W>(1).base, c0: std::ptr::null(), ct_stride: cw, out: ext.base, part_stride: pw, out_ct_stride: cw, out_rot_stride: 0, c0_mode: 0 };
        let dstv = View::of(&cx.work.em[base + dst], 0);
        let kptr = epi.k.map(|ki| self.raw[ki].vptr()).unwrap_or(std::ptr::null());
        match self.relin_source {
            KskSource::Regen if self.relin.seeded() => {
                let ndig = cx.ks.ndig(top)?;
                ks::stage_key(ctx, &ex.compute, cx.ks, &mut ex.ks, &self.relin, 0, ndig, false)?;
                ks::dot_regen(ctx, &ex.compute, cx.ks, &ex.ks, &self.relin, kk, top, DotRows::P, io, None)?;
                ks::moddown_conv(ctx, &ex.compute, cx.ks, &mut ex.ks.ext, &mut ex.ks.conv, kk, top)?;
                if let Some(re) = &self.regen_epi {
                    // Q rows straight into dst = f(dst + (dot - conv) P^-1) with the program's x2 / + constant
                    let rows = re.rows.get(&top).ok_or_else(|| shape("evalmod relin", format!("no row table for top {top}")))?;
                    let ndig = cx.ks.ndig(top)?;
                    let mu_ct = (cx.ks.dnum * lay.poly_words()) as u64;
                    let conv = View::of(&ex.ks.conv, 0);
                    let mut c = 0;
                    while c < kk {
                        let kc = (kk - c).min(ks::REGEN_KMAX);
                        let at = |p: *const c_void, stride: u64| (p as *const W).wrapping_add(c * stride as usize) as *const c_void;
                        let out = at(io.out, io.out_ct_stride);
                        let ka = pc_kernels::KsArgs {
                            c1: at(io.c1, io.ct_stride),
                            modup: at(ex.ks.digits.vptr(), mu_ct),
                            c0: std::ptr::null(),
                            out0: out as *mut c_void,
                            out1: (out as *const W).wrapping_add(pw as usize) as *mut c_void,
                            kb: re.kb.dptr() as *const *const c_void,
                            ka: re.kb.dptr() as *const *const c_void,
                            autok: re.autok.dptr(),
                            rows: rows.dptr(),
                            digit_of: re.digit_of.dptr(),
                            pmod: ctx.ks.p_mod_q.v.vptr(),
                            pmods: ctx.ks.p_mod_q.s.vptr(),
                            pr: ctx.pr(),
                            ct_stride: io.ct_stride,
                            mu_ct_stride: mu_ct,
                            mu_digit_stride: pw,
                            out_ct_stride: io.out_ct_stride,
                            out_rot_stride: 0,
                            key_digit_stride: re.key_digit_stride,
                            k: kc as i32,
                            r: 1,
                            ndig: ndig as i32,
                            c0_mode: 0,
                            nq: lay.nq as i32,
                            nrows: (top + 1) as i32,
                        };
                        let e = pc_kernels::EpiArgs {
                            dst: at(dstv.base, dstv.ct_stride) as *mut c_void,
                            conv: at(conv.base, conv.ct_stride),
                            binv: ctx.ks.p_inv_mod_q.v.vptr(),
                            binvs: ctx.ks.p_inv_mod_q.s.vptr(),
                            k: kptr,
                            part_stride: pw,
                            dbl: epi.dbl as i32,
                            pad: 0,
                            alpha: std::ptr::null(),
                            alphas: std::ptr::null(),
                        };
                        let addp = add.map(|b| at(View::of(&cx.work.em[base + b], 0).base, dstv.ct_stride)).unwrap_or(std::ptr::null());
                        let a = pc_kernels::KsRegenArgs { ks: ka, seed: re.seed, epi: e, use_epi: 1, pad: 0, key: re.key, add: addp };
                        // SAFETY: kc-ct views of the program buffers / KS workspace (prefix of kk <= k_max), the key's
                        // dense rows and device seed, row tables built at setup
                        launch_result("ks_dot_regen", unsafe { (ctx.table.ks_dot_regen)(&a, ex.compute.raw()) })?;
                        c += kc;
                    }
                    return Ok(());
                }
                ks::dot_regen(ctx, &ex.compute, cx.ks, &ex.ks, &self.relin, kk, top, DotRows::Q, io, Some((dstv, View::of(&ex.ks.conv, 0))))?;
                // the program's epilogue (x2, + constant) after the fused ModDown epilogue: the same residues
                if epi.dbl {
                    ks::ew(ctx, &ex.compute, ks::Ew::Add, dstv, dstv, Some(dstv), lay.rs_q(top), kk, 2)?;
                }
                if let Some(ki) = epi.k {
                    let a = AddConstArgs { x: dstv.r(), k: self.raw[ki].vptr(), pr: ctx.pr(), n: n as i32, kk: kk as i32 };
                    // SAFETY: kk cts of a program buffer
                    launch_result("add_const", unsafe { (ctx.table.add_const)(&a, ex.compute.raw()) })?;
                }
            }
            _ => {
                let staged = self.relin.needs_stage() || !self.relin.has_a();
                if staged {
                    let ndig = cx.ks.ndig(top)?;
                    ks::stage_key(ctx, &ex.compute, cx.ks, &mut ex.ks, &self.relin, 0, ndig, true)?;
                }
                ks::dot(ctx, &ex.compute, cx.ks, &ex.ks, &self.relin, 0, 1, kk, top, DotRows::All, io, staged)?;
                ks::moddown_conv(ctx, &ex.compute, cx.ks, &mut ex.ks.ext, &mut ex.ks.conv, kk, top)?;
                let a = MdEpiArgs {
                    dst: dstv.r(),
                    x: ext.r(),
                    t: View::of(&ex.ks.conv, 0).r(),
                    binv: ctx.ks.p_inv_mod_q.v.vptr(),
                    binvs: ctx.ks.p_inv_mod_q.s.vptr(),
                    k: kptr,
                    pr: ctx.pr(),
                    dbl: epi.dbl as i32,
                    n: n as i32,
                    kk: kk as i32,
                    pad: 0,
                };
                // SAFETY: kk-ct views of live buffers; dst is a program buffer distinct from ext / conv
                launch_result("md_epilogue", unsafe { (ctx.table.md_epilogue)(&a, ex.compute.raw()) })?;
            }
        }
        Ok(())
    }
}

/// The program buffers a plan needs per call of up to k_max cts (BtsWork::em: count, cts each).
pub fn em_work(nslots: usize, k_max: usize, mode: EmMode) -> (usize, usize) {
    match mode {
        EmMode::Complex => (2 * nslots, k_max),
        EmMode::Real => (nslots, k_max),
    }
}

/// Layout check helper (tests).
pub fn layout_rows(lay: &Layout) -> usize {
    lay.rows()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Liveness assignment on the shipped programs: fewer buffers, and every op's operands stay distinct where they
    /// were distinct before (no new aliasing inside an op).
    #[test]
    fn liveness_shipped_programs() {
        let chain = pc_core::params::Chain::generate(&pc_core::params::ChainSpec::n32()).expect("n32");
        let setup = crate::bts::BtsSetup::n32_shipped();
        let plan = crate::bts::dense::DensePlan::new(&chain, &setup).expect("dense plan");
        let st = pc_core::scale::ScaleTables::precompute(&chain).expect("tables");
        let v = &plan.variants[0];
        let ps = PsPlan::from_coeffs(&setup.evalmod_coeffs).expect("ps");
        let (prog, _) = build_evalmod_opts(&st, v.em_in, &ps, setup.evalmod_da_its as i32, true).expect("program");
        let fused = prog.fused();
        let before = slots_of(&fused);
        let (ops, after) = assign_buffers(fused.clone());
        println!("dense EvalMod: {} ops, buffers {before} -> {after}", ops.len());
        assert!(after < before);
        let (rw, _) = rewrite(fused.clone(), &prog.consts, &st.q);
        let rw = schedule_relin_rescales(rw, st.d);
        let after_rw = assign_buffers(rw.clone()).1;
        println!("dense EvalMod rewritten: {} -> {} ops, buffers {after_rw}", fused.len(), rw.len());
        assert!(after_rw <= after && after_rw == v.em_slots);
        for (a, b) in fused.iter().zip(&ops) {
            let (mut a, mut b) = (a.clone(), b.clone());
            let (ia, ib): (Vec<usize>, Vec<usize>) = (ids_mut(&mut a).into_iter().map(|x| *x).collect(), ids_mut(&mut b).into_iter().map(|x| *x).collect());
            for i in 0..ia.len() {
                for j in 0..ia.len() {
                    assert_eq!(ia[i] == ia[j], ib[i] == ib[j], "{a:?} -> {b:?}");
                }
            }
        }
        let k24 = crate::bts::sparse_b::ps_k24().expect("k24");
        let (p2, _) = build_evalmod(&st, v.em_in, &k24, 5).expect("k24 program");
        let f2 = p2.fused();
        let (r2, c2) = rewrite(f2.clone(), &p2.consts, &st.q);
        println!("K24 EvalMod: buffers {} -> {}; rewritten {} -> {} ops, buffers {}", slots_of(&f2), assign_buffers(f2.clone()).1, f2.len(), r2.len(), assign_buffers(r2.clone()).1);
        let copies = |o: &[EmOp]| o.iter().filter(|x| matches!(x, EmOp::Copy { .. })).count();
        println!("K24 copies {} -> {}, constant rows {} -> {}", copies(&f2), copies(&r2), p2.consts.len(), c2.len());
        for (i, o) in r2.iter().enumerate() {
            println!("  {i:3} {o:?}");
        }
    }
}
