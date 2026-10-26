//! `pc-bench replay`: a compiled block program replayed on the g0 oracle (GPT-2 decode token 0).
//!
//!   pc-bench replay [--plan \<dir>] [--graph \<dir>] [--scalars \<v2 capture>] [--block block_0] [--g0 \<dump>]
//!                   [--inputs anon_1] [--reps 10] [--debug] [--oracle-bts] [--ring] [--kv]
//!
//! Setup: the program compiled from plan + graph, the dump's keys / plaintexts / block inputs, the three decode routes
//! built on those keys (SPRU's block key fresh: the dump has none). Then: eager replay vs the oracle's block output;
//! graph capture / launch == eager bit-exact; device ms and host us per block, eager vs graph, 0 allocations after
//! setup; levels vs the dump's op stream. `--debug` decrypts every dumped var; `--ring` stages the plaintexts through the
//! weight ring; `--kv` round-trips the KV offload.
use crate::check_bts::OracleKeys;
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::keys::SecretKey;
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::{Compute, Device, Event, EventPool, Stream};
use pc_gpu::bts::sparse::{chain_deg, SparseBRoute, SpruRoute};
use pc_gpu::bts::sparse_b::{SparseBCfg, SparseBPrecomp};
use pc_gpu::bts::spru::{Spru, SpruCfg, SpruKey};
use pc_gpu::bts::stc_first::{StcFirstPlan, StcFirstSetup};
use pc_gpu::bts::{BtsKeyRing, BtsSetup, BtsWork, BtsWorkShape};
use pc_gpu::decrypt::{Decryptor, Which};
use pc_gpu::encrypt::Encryptor;
use pc_gpu::keygen::{test_prng, KeyGen};
use pc_gpu::keys::{DevSecret, KskShape};
use pc_gpu::ks::{DevKey, KeyA, KsPlans, KsWork, KskSource};
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::pt_store::{PtKey, PtStore, PtStoreCfg, PtStorePolicy, StageItem, StageValues};
use pc_gpu::{Context, Ct, CtPool, Executor, MemCfg, MemPlan, PoolClass};
use pc_program::bind::{BindEnv, Bound, ScalarTable, StepOp};
use pc_program::compile::{compile_dir, runtime_shadow, CompileOpts};
use pc_program::plan::{BlockPlan, PlanDir};
use pc_program::kv::KvStore;
use pc_program::replay::{CaptureLock, ReplayCfg, Replayer, Routes, Rt, Served};
use pc_program::residency::{RingCfg, WeightRing, WeightSource};
use pc_program::Program;
use crate::dump::{Dump, HostCt};
use crate::dump_g0::{npy_header, G0};
use pc_shadow::{CtMeta, PtMeta, Shadow};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

type W = u32;
type M = Barrett64;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const ORACLE: &str = "/data02/users/mithrillo/projects/perseus-oracle/fx/dumps/g0";
/// `fx/dumps/m7b/v2cap.py` output: the decode graph re-captured by the fake (graph v2: `args.scalar` exact).
const V2_SCALARS: &str = "/data02/users/mithrillo/projects/perseus-oracle/fx/dumps/m7b/gpt2_decode_v2";

struct Opts {
    plan: PathBuf,
    graph: PathBuf,
    /// A v2 capture of the same graph: the exact scalars of the v1 graph's 6-digit literals, by planner node.
    scalars: Option<PathBuf>,
    g0: PathBuf,
    block: String,
    inputs: Vec<String>,
    reps: usize,
    debug: bool,
    oracle_bts: bool,
    ring: bool,
    ring_bufs: usize,
    kv: bool,
}

fn parse(a: &[String]) -> Res<Opts> {
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut o = Opts {
        plan: assets.join("bootstrap_placements/gpt2_decode"),
        // the shipped decode plan's graph (v1) and a v2 fake capture of it (exact scalars, CPU only)
        graph: assets.join("graphs/gpt2_decode"),
        scalars: Some(PathBuf::from(V2_SCALARS)).filter(|p| p.is_dir()),
        g0: PathBuf::from(ORACLE),
        block: "block_0".into(),
        inputs: vec!["anon_1".into()],
        reps: 10,
        debug: false,
        oracle_bts: false,
        ring: false,
        ring_bufs: 16,
        kv: false,
    };
    let mut i = 0;
    while i < a.len() {
        let v = || a.get(i + 1).cloned().ok_or_else(|| format!("{} needs a value", a[i]));
        match a[i].as_str() {
            "--plan" => o.plan = PathBuf::from(v()?),
            "--graph" => o.graph = PathBuf::from(v()?),
            "--scalars" => o.scalars = Some(PathBuf::from(v()?)),
            "--g0" => o.g0 = PathBuf::from(v()?),
            "--block" => o.block = v()?,
            "--inputs" => o.inputs = v()?.split(',').map(str::to_string).collect(),
            "--reps" => o.reps = v()?.parse()?,
            "--ring-bufs" => o.ring_bufs = v()?.parse()?,
            "--debug" => {
                o.debug = true;
                i -= 1;
            }
            "--oracle-bts" => {
                o.oracle_bts = true;
                o.debug = true;
                i -= 1;
            }
            "--ring" => {
                o.ring = true;
                i -= 1;
            }
            "--kv" => {
                o.kv = true;
                i -= 1;
            }
            x => return Err(format!("unknown argument {x}").into()),
        }
        i += 2;
    }
    Ok(o)
}

pub fn cmd(args: &[String]) -> ExitCode {
    match parse(args).and_then(|o| run(&o)) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => {
            println!("[replay] GATE FAILED");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("pc-bench replay: {e}");
            ExitCode::FAILURE
        }
    }
}

fn read_sk(p: &Path) -> Res<Vec<i8>> {
    let (d, _shape, off) = npy_header(p)?;
    let b = std::fs::read(p)?;
    match d.trim_start_matches(['<', '|', '=']) {
        "i1" => Ok(b[off..].iter().map(|&x| x as i8).collect()),
        "i8" => Ok(b[off..].chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])) as i8).collect()),
        o => Err(format!("{}: dtype {o}", p.display()).into()),
    }
}

fn read_c16(p: &Path) -> Res<Vec<Complex>> {
    let (d, _shape, off) = npy_header(p)?;
    let b = std::fs::read(p)?;
    let f = |c: &[u8]| f64::from_le_bytes(c.try_into().unwrap_or([0; 8]));
    match d.trim_start_matches(['<', '|', '=']) {
        "c16" => Ok(b[off..].chunks_exact(16).map(|c| Complex::new(f(&c[..8]), f(&c[8..]))).collect()),
        "f8" => Ok(b[off..].chunks_exact(8).map(|c| Complex::new(f(c), 0.0)).collect()),
        o => Err(format!("{}: dtype {o}", p.display()).into()),
    }
}

fn meta_of(h: &HostCt, nq: usize) -> CtMeta {
    CtMeta { level: (nq - 1 - h.level) as u32, deg: h.noise_level as u8, nf: h.noise_factor, slots: h.slots as u32, tag: Default::default() }
}

/// A dumped key (CPU map `b` + KSKA seed), `a` regenerated in the dots (no resident `a`: Regen everywhere).
fn load_key<'a>(arena: &'a pc_cuda::Arena, s: &Stream<Compute>, ctx: &Context<'_, W, M>, plans: &KsPlans<'_>, d: &Dump, pj: &Value, name: &str, g: u64) -> Res<DevKey<'a, W>> {
    let file = if name == "mult" { "keys/mult.b.bin".to_string() } else { format!("keys/auto{name}.b.bin") };
    if !d.exists(&file) {
        return Err(format!("key {name} (galois {g}) not in the dump").into());
    }
    let b = d.u32s(&file);
    let v: Vec<u32> = pj["keys"][name]["a_seed"].as_array().ok_or(format!("keys.{name}.a_seed"))?.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect();
    let seed: [u32; 8] = v.try_into().map_err(|_| "seed must be 8 words")?;
    Ok(DevKey::new(arena, s, ctx, plans, g, ctx.chain.dnum(), &b, Some(seed), KeyA::None)?)
}

/// The rotation steps of linear-transform stages (baby steps, giant stride, offset), as the StC-first harness.
fn lt_steps(geoms: &[pc_gpu::bts::cts::LtGeom], n: usize) -> BTreeSet<i64> {
    let mut steps = BTreeSet::new();
    for g in geoms {
        for i in 1..g.b {
            steps.insert(((i * g.stride) % (n / 2)) as i64);
        }
        if g.g > 1 {
            steps.insert(((g.b * g.stride) % (n / 2)) as i64);
        }
        steps.insert(g.offset.rem_euclid((n / 2) as i64));
    }
    steps.remove(&0);
    steps
}

/// Dumped plaintext `name@level` (device rows, compact width expanded) -> PtStore rows + meta.
fn dumped_pt(pts: &Path, name: &str, level: u32, nq: usize, n: usize) -> Res<(Vec<u32>, PtMeta)> {
    let base = pts.join(format!("{name}@{level}"));
    let jp = format!("{}.json", base.display());
    let m: Value = serde_json::from_str(&std::fs::read_to_string(&jp).map_err(|e| format!("{jp}: {e}"))?)?;
    let ids: Vec<usize> = m["c0"]["ids"].as_array().ok_or("pt ids")?.iter().filter_map(|x| x.as_u64().map(|v| v as usize)).collect();
    let sizes: Vec<usize> = m["c0"]["q_sizes"].as_array().ok_or("pt q_sizes")?.iter().filter_map(|x| x.as_u64().map(|v| v as usize)).collect();
    let raw = crate::dump::read_u32(Path::new(&format!("{}.q.bin", base.display())));
    let enc = m["compact_mask"].as_u64().unwrap_or(0xFFFF_FFFF) as u32;
    let mut rows = vec![0u32; ids.len() * n];
    let mut off = 0;
    for (j, &w) in sizes.iter().enumerate() {
        for i in 0..n {
            let src = if w >= n {
                i
            } else if enc >> 24 == 0x7F {
                i >> (enc & 0xFF)
            } else {
                i & enc as usize
            };
            rows[j * n + i] = raw[off + src.min(w - 1)];
        }
        off += w;
    }
    if ids != (0..ids.len()).collect::<Vec<_>>() {
        return Err(format!("{name}@{level}: ids are not 0..{}", ids.len()).into());
    }
    let top = m["level"].as_u64().ok_or("pt level")? as usize;
    let meta = PtMeta { level: (nq - 1 - top) as u32, deg: m["noise_level"].as_u64().unwrap_or(1) as u8, nf: m["noise_factor"].as_f64().unwrap_or(0.0), tag: Default::default() };
    if meta.level != level {
        return Err(format!("{name}@{level}: json top {top} is level {}", meta.level).into());
    }
    Ok((rows, meta))
}

/// Host values of the dumped plaintexts (`pts/<pt>@<level>.values.npy`): the weight ring's source on g0.
/// Preloaded in memory (the driver's weights are host-resident; the ring times encode + upload, not disk).
struct G0Source {
    items: BTreeMap<(String, u32), (Vec<Complex>, u8)>,
}

impl G0Source {
    fn load(dir: &Path, keys: &[PtKey]) -> Res<Self> {
        let mut items = BTreeMap::new();
        for k in keys {
            let nm = dump_pt_name(&k.name).to_string();
            if items.contains_key(&(nm.clone(), k.level)) {
                continue;
            }
            let base = dir.join(format!("{nm}@{}", k.level));
            let values = read_c16(&PathBuf::from(format!("{}.values.npy", base.display())))?;
            let m: Value = serde_json::from_str(&std::fs::read_to_string(format!("{}.json", base.display()))?)?;
            items.insert((nm, k.level), (values, m["noise_level"].as_u64().unwrap_or(1) as u8));
        }
        Ok(G0Source { items })
    }
}

impl WeightSource for G0Source {
    fn item(&self, key: &PtKey) -> pc_program::Result<StageItem> {
        let (values, deg) = self.items.get(&(dump_pt_name(&key.name).to_string(), key.level)).ok_or_else(|| pc_program::Error::Replay { block: "ring source".into(), msg: format!("no values for {}@{}", key.name, key.level) })?;
        let slots = values.len();
        Ok(StageItem { key: key.clone(), values: StageValues::Slots { values: values.clone(), slots, payload: Payload::Complex }, deg: *deg, scale: None })
    }
}

/// The KV choreography over `layers()` layers for two tokens: token 1 fills each layer's page (a copy of `seed`
/// plus the layer index), stores it, finalizes the previous one; token 2 joins each layer (its reload prefetched
/// while the layer before it was joined), compares every word with token 1's, and stores it again. Host time per
/// choreography call is measured around the calls only (no device wait inside them).
fn kv_check<'a, K: KvStore<'a, W>>(kv: &mut K, ex: &mut Executor<'a, W>, ctx: &Context<'a, W, M>, sh: &Shadow, ops: &mut Ops<'a, W>, seed: &Ct<'a, W>, what: &str) -> Res<bool> {
    let l = kv.layers();
    let mut want: Vec<Vec<u32>> = vec![];
    let mut call_us: Vec<f64> = vec![];
    let words = |c: &Ct<'a, W>, ex: &Executor<'a, W>| -> Res<Vec<u32>> {
        let top = c.top(ctx.shape)?;
        let ids: Vec<usize> = (0..=top).collect();
        let mut v = c.buf.download_rows(&ex.compute, 0, 0, &ids)?;
        v.extend(c.buf.download_rows(&ex.compute, 0, 1, &ids)?);
        Ok(v)
    };
    // token 1
    for i in 0..l {
        let t = Instant::now();
        kv.join(ex, i)?;
        call_us.push(t.elapsed().as_secs_f64() * 1e6);
        {
            let pg = kv.page(i)?;
            pc_program::replay::copy_ct(ctx, &ex.compute, pg, seed)?;
            ops.add_scalar_inplace(ctx, ex, sh, std::slice::from_mut(pg), &[i as f64])?;
        }
        want.push(words(kv.page(i)?, ex)?);
        let t = Instant::now();
        kv.store(ex, i)?;
        if i > 0 {
            kv.finalize(ex, i - 1)?;
        }
        call_us.push(t.elapsed().as_secs_f64() * 1e6);
    }
    kv.finalize(ex, l - 1)?;
    // token 2
    let mut bad = 0usize;
    let mut tot = 0usize;
    let t0 = Instant::now();
    kv.prefetch(ex, 0)?;
    call_us.push(t0.elapsed().as_secs_f64() * 1e6);
    for i in 0..l {
        let t = Instant::now();
        kv.join(ex, i)?;
        kv.prefetch(ex, i + 1)?;
        call_us.push(t.elapsed().as_secs_f64() * 1e6);
        let got = words(kv.page(i)?, ex)?;
        tot += want[i].len();
        bad += got.iter().zip(&want[i]).filter(|(a, b)| a != b).count() + got.len().abs_diff(want[i].len());
        let t = Instant::now();
        kv.store(ex, i)?;
        if i > 0 {
            kv.finalize(ex, i - 1)?;
        }
        call_us.push(t.elapsed().as_secs_f64() * 1e6);
    }
    kv.finalize(ex, l - 1)?;
    ex.copy.sync()?;
    ex.compute.sync()?;
    let mx = call_us.iter().cloned().fold(0f64, f64::max);
    println!("[replay] kv ({what}, {l} layers, 2 tokens): reloaded pages vs stored {bad} of {tot} words differ; host per choreography call median {:.1} us, max {mx:.1} us", median(call_us.clone()));
    Ok(bad == 0)
}

fn dump_pt_name(key: &str) -> &str {
    key.rsplit('/').next().unwrap_or(key)
}

/// Centered CRT over 4 rows + the slot decode (block outputs exceed what the bottom pair holds).
fn decode4(enc: &Encoder, q: &[u64], rows: &[Vec<u64>], slots: usize, scale: f64, n: usize) -> Vec<Complex> {
    let big: u128 = q.iter().fold(1u128, |a, &x| a * x as u128);
    let hats: Vec<(u128, u64)> = q
        .iter()
        .map(|&qi| {
            let hat = big / qi as u128;
            (hat, pc_core::modmul::invmod((hat % qi as u128) as u64, qi))
        })
        .collect();
    let coef = |j: usize| -> f64 {
        let mut x: u128 = 0;
        for (i, &(hat, hinv)) in hats.iter().enumerate() {
            let t = pc_core::modmul::mulmod(rows[i][j] % q[i], hinv, q[i]) as u128;
            x = (x + t * hat) % big;
        }
        if x > big / 2 {
            -((big - x) as f64)
        } else {
            x as f64
        }
    };
    let gap = n / (2 * slots);
    let mut v: Vec<Complex> = (0..slots).map(|i| Complex::new(coef(gap * i) / scale, coef(gap * i + n / 2) / scale)).collect();
    enc.fft_special(&mut v);
    v
}

fn rel(a: &[Complex], b: &[Complex]) -> f64 {
    let m = b.iter().fold(0f64, |x, z| x.max(z.abs()));
    a.iter().zip(b).fold(0f64, |x, (p, q)| x.max(Complex::new(p.re - q.re, p.im - q.im).abs())) / m.max(1e-300)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    if v.is_empty() {
        0.0
    } else {
        v[v.len() / 2]
    }
}

/// The dumped name of a program var's state: `X~pre` = the op's own output (dump `X`), `X` = its final state
/// (after a planned refresh: dump `X_planned_bootstrapped`), `~raw` states have no dump.
fn dump_name(var: &str, has: &dyn Fn(&str) -> bool) -> Option<String> {
    if let Some(x) = var.strip_suffix("~pre") {
        return has(x).then(|| x.to_string());
    }
    if var.contains('~') {
        return None;
    }
    let pb = format!("{var}_planned_bootstrapped");
    if has(&pb) {
        return Some(pb);
    }
    has(var).then(|| var.to_string())
}

/// Exact scalars of the program's inexact literals from a v2 capture of the same graph (same planner order: node
/// index = `Origin::Node`), each checked against the literal's 6 significant digits.
fn scalar_table(o: &Opts, prog: &Program, plan_cfg: &pc_config::PlanConfig) -> Res<ScalarTable> {
    let mut t = ScalarTable::new();
    let Some(dir) = &o.scalars else { return Ok(t) };
    let lo = pc_program::compile::load_opts(plan_cfg)?;
    let b = o.block.trim_start_matches("block_").split('_').next().unwrap_or("0");
    let bg = pc_program::graph::BlockGraph::load(dir.join(format!("block_{b}")).join("graph.json"), &o.block, &lo)?;
    for nd in &bg.graph.nodes {
        if let Some(v) = bg.args(&nd.output).and_then(|a| a.get("scalar")).and_then(Value::as_f64) {
            t.insert(nd.idx as u32, v);
        }
    }
    let (mut used, mut bad) = (0, 0);
    for op in &prog.ops {
        for m in op.members() {
            let c = match &m.kind {
                pc_program::Kind::AddConst(c) | pc_program::Kind::MultConst(c) => c,
                _ => continue,
            };
            if c.exact {
                continue;
            }
            if let pc_program::program::Origin::Node(i) = m.origin {
                match t.get(&i) {
                    Some(&v) => {
                        used += 1;
                        // the literal is the value printed with 6 significant digits (sub of c records add(-c))
                        let close = |x: f64| (x - c.value).abs() <= 1e-5 * c.value.abs().max(1e-300);
                        if !(close(v) || close(-v)) {
                            bad += 1;
                            println!("[replay]   scalar node {i}: v2 {v:e} vs literal {:e}", c.value);
                        }
                    }
                    None => println!("[replay]   scalar node {i}: no v2 value"),
                }
            }
        }
    }
    println!("[replay] exact scalars from {}: {} in the table, {used} inexact literals bound, {bad} disagree with their 6 digits", dir.display(), t.len());
    if bad > 0 {
        return Err("the v2 capture's scalars do not match the program's literals (different graph?)".into());
    }
    Ok(t)
}

/// Compiled levels vs the oracle op stream (`ops.jsonl` out_level / deg of each graph name): a record is the op's
/// own output, i.e. the program's `X~pre` state when the plan realizes or refreshes X after it, else `X`; a
/// planned refresh's record (`X_planned_bootstrapped`) is the program's final `X`. Prints the disagreements by
/// (old op, our op kind) with examples.
fn levels_vs_ops(prog: &Program, g0: &G0) -> Res<()> {
    let ops_j = g0.ops()?;
    let mut rec: BTreeMap<String, (i64, i64, String, Vec<String>, Vec<i64>)> = BTreeMap::new();
    for r in ops_j.iter().filter(|r| r["step"].as_str().unwrap_or("").starts_with("blk0")) {
        if let Some(out) = r["out"].as_str() {
            let ins = r["ins"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
            let inl = r["in_levels"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
            rec.insert(out.to_string(), (r["out_level"].as_i64().unwrap_or(-1), r["deg"].as_i64().unwrap_or(-1), r["op"].as_str().unwrap_or("").to_string(), ins, inl));
        }
    }
    let names: BTreeSet<&str> = prog.vars.iter().map(|v| v.name.as_str()).collect();
    let (mut cmpd, mut bad) = (0, 0);
    let mut by: BTreeMap<String, (usize, Vec<String>)> = BTreeMap::new();
    for op in &prog.ops {
        for m in op.members() {
            let v = &prog.vars[m.outs[0] as usize];
            let has = |suf: &str, x: &str| names.contains(format!("{x}{suf}").as_str());
            // X~pre: the op before a planned refresh; X~raw: a refresh's raw output (when X~pre exists) or the op
            // before a realize anchor; X: the final state (a planned refresh's record, or no record when realized)
            let key = if let Some(x) = v.name.strip_suffix("~pre") {
                x.to_string()
            } else if let Some(x) = v.name.strip_suffix("~raw") {
                if has("~pre", x) {
                    continue;
                }
                x.to_string()
            } else if v.name.contains('~') {
                continue;
            } else if has("~pre", &v.name) {
                format!("{}_planned_bootstrapped", v.name)
            } else if has("~raw", &v.name) {
                continue;
            } else {
                v.name.clone()
            };
            let Some((l, d, opn, ins, inl)) = rec.get(&key) else { continue };
            if *l < 0 {
                // a fused accumulator's later terms (mult_slot_pt_sum loop): no level recorded
                continue;
            }
            cmpd += 1;
            if *l != v.meta.level as i64 || *d != v.meta.deg as i64 {
                bad += 1;
                let e = by.entry(format!("{opn} / {}", m.kind.name())).or_default();
                e.0 += 1;
                if e.1.len() < 3 {
                    let ours: Vec<String> = m.ins.iter().map(|&i| format!("{}@{}d{}", prog.vars[i as usize].name, prog.vars[i as usize].meta.level, prog.vars[i as usize].meta.deg)).collect();
                    e.1.push(format!("{} <- {opn}{ins:?}@{inl:?}: old level {l} deg {d}; ours level {} deg {} <- {ours:?} pre {:?}", v.name, v.meta.level, v.meta.deg, m.pre));
                }
            }
        }
    }
    println!("[replay] levels vs ops.jsonl: {cmpd} op outputs compared, {bad} differ in level or degree");
    for (k, (n, ex)) in &by {
        println!("[replay]   {k}: {n}");
        for e in ex {
            println!("[replay]     {e}");
        }
    }
    Ok(())
}

fn run(o: &Opts) -> Res<bool> {
    let t0 = Instant::now();
    let mut ok = true;
    // ── compile ───────────────────────────────────────────────────────────────────────────────────────────────────
    let dc = compile_dir(&o.plan, &o.graph, &CompileOpts::default())?;
    let prog: Program = dc.programs.into_iter().find(|(l, _)| *l == o.block).ok_or(format!("no {} in {}", o.block, o.plan.display()))?.1?;
    let pd = PlanDir::open(&o.plan)?;
    let plan_cfg = pd.plan_config()?;
    let bplan = BlockPlan::load(o.plan.join(format!("{}_placement.json", o.block)))?;
    let cfg = pd.config(&bplan)?;
    let sh0 = runtime_shadow(&cfg, &plan_cfg, bplan.file.rules.bootstrap_level)?;
    let a = &prog.accounting;
    println!(
        "[replay] {} of {} ({}): {} ops ({} launches), {} bootstraps, {} pt refs, {} slots ({:.1} MB compact), inexact scalars {}, check errors {}",
        prog.block,
        o.plan.display(),
        o.graph.display(),
        a.ops,
        prog.batch_report.launches,
        prog.bts_sites.len(),
        prog.pt_refs.len(),
        prog.mem.slot_bytes.len(),
        prog.mem.bytes as f64 / 1e6,
        prog.check.inexact_scalars,
        prog.check.errors.len()
    );
    // ── the dump ──────────────────────────────────────────────────────────────────────────────────────────────────
    let g0 = G0::open(&o.g0)?;
    levels_vs_ops(&prog, &g0)?;
    let pj = g0.params.clone();
    let keysd = Dump::open(&g0.dir);
    let cts = g0.cts();
    let chain = Chain::generate(&ChainSpec::n32())?;
    if crate::dump::as_u64v(&pj["q"]) != chain.q {
        return Err("the dump's primes differ from the n32 preset".into());
    }
    let n = chain.n;
    let n2 = 2 * n as u64;
    let nq = chain.q.len();
    let sk = SecretKey { s: read_sk(&g0.dir.join("sk.npy"))?, s_sparse: vec![] };
    // routes of the program
    let mut dense_drops: BTreeSet<u32> = BTreeSet::new();
    let mut sparse: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    let mut spru_h: Option<u32> = None;
    let mut spru_drops = 0u32;
    let mut real = false;
    let mut n_dense = 0usize;
    for s in &prog.bts_sites {
        let r = s.route;
        match pc_gpu::bts::RouteKind::of(&r, n) {
            pc_gpu::bts::RouteKind::Spru { h, .. } => {
                spru_h = Some(h);
                spru_drops |= 1 << r.raise_drop.min(31);
            }
            pc_gpu::bts::RouteKind::Sparse(sl) => {
                sparse.entry(sl).or_default().insert(r.raise_drop);
            }
            _ => {
                dense_drops.insert(r.raise_drop);
                real |= r.real;
                n_dense += 1;
            }
        }
    }
    println!("[replay] routes: dense/StC-first drops {dense_drops:?} (real {real}, {n_dense} sites), sparse {sparse:?}, SPRU h {spru_h:?}");
    // decode refreshes run one ciphertext per call (route work buffers at K = 1); the context's row tables and key
    // switching plans cover 8 (SPRU's key ciphertexts are one strided batch)
    let k_max = 1;
    let ctx_k = 8;
    let mut setup = BtsSetup::n32_shipped();
    setup.raise_drops = if dense_drops.is_empty() { vec![0] } else { dense_drops.iter().copied().collect() };
    setup.real = real;
    setup.k_max = k_max;
    let mut sfs = StcFirstSetup::n32_shipped();
    // every key the block's dense calls use must stay cached: a graph reads the cached tables
    sfs.cache_slots = n_dense.max(1);
    let splan = StcFirstPlan::new(&chain, &setup, &sfs)?;
    let st_tables = pc_core::scale::ScaleTables::precompute(&chain)?;
    let sb_cfgs: Vec<SparseBCfg> = sparse
        .iter()
        .flat_map(|(&sl, drops)| {
            drops.iter().map(move |&d| {
                let mut c = SparseBCfg::n32(sl, d);
                c.k_max = 1;
                c
            })
        })
        .collect();
    let spru_cfg = spru_h.map(|h| SpruCfg { drops: spru_drops, ..SpruCfg::new(h as usize) });
    // every rotation key: the program's, the routes'
    let mut steps: BTreeSet<i64> = BTreeSet::new();
    let half = (n / 2) as i64;
    for op in &prog.ops {
        for m in op.members() {
            match &m.kind {
                pc_program::Kind::Rotate(k) => {
                    steps.insert(k.rem_euclid(half));
                }
                pc_program::Kind::RotateSum(s) => steps.extend(s.iter().map(|k| k.rem_euclid(half))),
                _ => {}
            }
        }
    }
    {
        let mut geoms: Vec<_> = splan.dense.variants.iter().flat_map(|v| v.cts.iter().chain(&v.stc)).chain(&splan.input).map(|h| h.geom).collect();
        geoms.extend(splan.sources.iter().flatten().map(|s| s.geom));
        steps.extend(lt_steps(&geoms, n));
    }
    for c in &sb_cfgs {
        steps.extend(pc_gpu::bts::sparse_b::rot_steps(n, c)?.into_iter().map(|k| k.rem_euclid(half)));
    }
    if let Some(c) = &spru_cfg {
        steps.extend(c.rot_steps(n).into_iter().map(|k| k.rem_euclid(half)));
    }
    steps.remove(&0);
    // ── MemPlan ───────────────────────────────────────────────────────────────────────────────────────────────────
    let dev = Device::open()?;
    let mcfg = MemCfg { k_max: ctx_k, pool: vec![PoolClass { k: 1, count: 4 }], events: 64, scratch_cts: 2 };
    let mut mp = MemPlan::build::<W, M>(&chain, mcfg)?;
    let lay = mp.lay;
    let dnum = chain.dnum();
    mp.reserve("ks.plans", KsPlans::bytes::<W>(&chain, ctx_k)?);
    mp.reserve("ks.regen_a", KsWork::<W>::regen_bytes(&lay, dnum));
    mp.reserve("keys", (steps.len() + 4) * DevKey::<W>::bytes(&lay, dnum, false, true));
    let ocfg = OpsCfg { const_slots: 2048, ..OpsCfg::default() };
    Ops::<W>::reserve(&mut mp, &ocfg);
    let n_pt_keys = prog.pt_refs.iter().map(|r| (r.name.clone(), r.level)).collect::<BTreeSet<_>>().len();
    // --ring: three more bindings of the block run through a 2-frame weight ring (stage-ahead slots on top of the
    // main binding's resident ones)
    const RING_BLOCKS: usize = 3;
    let ring_slots = if o.ring { 2 * n_pt_keys + 4 } else { 0 };
    // pinned staging buffers: the encode pool works on at most this many plaintexts at once
    let pcfg = PtStoreCfg { slots: n_pt_keys + 2 + ring_slots, ring: o.ring_bufs, encode_threads: 8, policy: if o.ring { PtStorePolicy::StageAhead } else { PtStorePolicy::ResidentAll }, eager: pc_gpu::encode_dev::EncoderKind::DeviceFft, rows: 0, classes: Vec::new() };
    PtStore::<W>::reserve(&mut mp, &pcfg);
    let (req, mut wshape) = splan.mem::<W>()?;
    req.reserve(&mut mp);
    for c in &sb_cfgs {
        let (b, w) = SparseBPrecomp::<W>::bytes(&chain, &st_tables, &lay, c)?;
        mp.reserve("bts.sparse", b);
        mp.reserve("bts.route", pc_gpu::bts::raise::EncapsKeys::bytes() + pc_gpu::bts::finish::Pow2::<W>::bytes(lay.rows()));
        wshape = wshape.max(w);
    }
    let mut kg_shapes: Vec<KskShape> = vec![];
    if let Some(c) = &spru_cfg {
        mp.reserve("bts.spru", Spru::<W>::bytes(&lay, c, &chain)? + Spru::<W>::bind_bytes(c, n));
        kg_shapes.push(KskShape::bottom(&chain)?);
        kg_shapes.push(KskShape::full(&chain));
        wshape = wshape.max(BtsWorkShape { k_max: 1, g_max: 1, em_slots: 0, em_kk: 1, spare: 0 });
    }
    if !kg_shapes.is_empty() {
        mp.reserve("keygen", KeyGen::<W>::bytes(&lay, &kg_shapes));
    }
    mp.reserve("bts.work", BtsWork::<W>::bytes(&lay, wshape));
    mp.reserve("keys.secret", 2 * DevSecret::<W>::bytes(&lay));
    mp.reserve("encrypt", Encryptor::<W>::bytes(&lay, false));
    mp.reserve("decrypt", Decryptor::<W>::bytes(&lay, 4, 1) + Decryptor::<W>::bytes(&lay, 2, 1));
    mp.reserve("replay", Bound::<W>::bytes(&prog, &lay));
    if o.ring {
        mp.reserve("replay.ring", RING_BLOCKS * Bound::<W>::bytes(&prog, &lay));
    }
    const KV_LAYERS: usize = 4;
    if o.kv {
        mp.reserve("kv", pc_program::kv::PinnedKv::<W>::bytes(&lay, 2) + pc_program::kv::DeviceKv::<W>::bytes(&lay, KV_LAYERS));
    }
    mp.reserve("slack", 256 << 20);
    println!("[replay] {} rotation keys + conj/relin/encaps; MemPlan {}", steps.len(), mp.report());
    let arena = mp.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &mp)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let pool = CtPool::<W>::new(&arena, &mp)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, ctx_k)?;
    let mut ops = Ops::<W>::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    // one event per staged plaintext in flight (a Ready per slot until adopted): two frames of a block
    let pt_events = EventPool::new(&dev, 2 * n_pt_keys + 64)?;
    let mut store = PtStore::<W>::new(&arena, &dev, &ex.compute, &ctx, &pt_events, pcfg)?;
    // keys
    let tk = Instant::now();
    let mut keys = OracleKeys { rot: BTreeMap::new(), conj: None, relin: None, atob: None, btoa: None, n };
    {
        let s = &ex.compute;
        for &st in &steps {
            let g = pc_core::galois::rotation_galois(st, n)?;
            keys.rot.insert(st, load_key(&arena, s, &ctx, &kplans, &keysd, &pj, &g.to_string(), g)?);
        }
        keys.conj = Some(load_key(&arena, s, &ctx, &kplans, &keysd, &pj, &(n2 - 1).to_string(), n2 - 1)?);
        keys.relin = Some(load_key(&arena, s, &ctx, &kplans, &keysd, &pj, "mult", 1)?);
        keys.atob = Some(load_key(&arena, s, &ctx, &kplans, &keysd, &pj, &(n2 - 2).to_string(), 1)?);
        keys.btoa = Some(load_key(&arena, s, &ctx, &kplans, &keysd, &pj, &(n2 - 4).to_string(), 1)?);
    }
    println!("[replay] {} oracle keys loaded in {:.1} s", steps.len() + 4, tk.elapsed().as_secs_f64());
    let stage = ex.ks.stage_ptrs();
    // routes
    let dense = splan.build(&arena, &dev, &ex.compute, &ctx, &keys, stage)?;
    let mut sbr: Vec<SparseBRoute<'_, W>> = vec![];
    for c in &sb_cfgs {
        let s = &ex.compute;
        let encaps = pc_gpu::bts::raise::EncapsKeys::new(&arena, s, n, keys.dense_to_sparse()?, keys.sparse_to_dense()?, stage)?;
        let pow2 = pc_gpu::bts::finish::Pow2::new(&arena, s, &ctx)?;
        let pre = SparseBPrecomp::new(&arena, s, &ctx, c.clone(), &keys, stage)?;
        sbr.push(SparseBRoute { slots: c.slots, ring_n: n, variants: vec![pre], encaps, pow2, cf_default: 6, cf_window: (2, 20), deg: chain_deg(&ctx)?, ks_source: c.ks_source });
    }
    let mut prng = test_prng(0x7b);
    let mut secret = DevSecret::<W>::carve(&arena, lay)?;
    secret.load(&ctx, &ex.compute, &sk)?;
    let mut enc = Encryptor::<W>::new(&arena, &ex.compute, lay, false, &mut prng)?;
    let sh = Shadow::new(sh0.cfg.clone(), Some(Arc::new(ctx.scale.clone())));
    let spru = match &spru_cfg {
        Some(c) => {
            let mut kg = KeyGen::<W>::new(&arena, &ex.compute, &chain, &kg_shapes)?;
            let mut spru_secret = DevSecret::<W>::carve(&arena, lay)?;
            let skey = SpruKey::sample(&mut prng, n, c)?;
            let mut s = Spru::<W>::new(&arena, &ctx, &mut ex, &mut kg, &mut enc, &secret, &mut spru_secret, &sk, &skey, c.clone(), &sh, &mut prng)?;
            s.bind(&arena, &ex.compute, &keys, stage)?;
            Some(SpruRoute { spru: std::cell::RefCell::new(s), ring_n: n, cf_default: 6, deg: chain_deg(&ctx)?, carved: 0 })
        }
        None => None,
    };
    let mut work = BtsWork::<W>::new(&arena, lay, wshape)?;
    let mut dec4 = Decryptor::<W>::new(&arena, &ex.compute, lay, 4, 1)?;
    dec4.load_secret(&ex.compute, &secret)?;
    let mut dec2 = Decryptor::<W>::new(&arena, &ex.compute, lay, 2, 1)?;
    dec2.load_secret(&ex.compute, &secret)?;
    let mut routes: Routes<'_, '_, W, M> = Routes::new();
    routes.add(&dense);
    for r in &sbr {
        routes.add(r);
    }
    if let Some(r) = &spru {
        routes.add(r);
    }
    // bind
    let scalars = scalar_table(o, &prog, &plan_cfg)?;
    let env = BindEnv { arena: &arena, stream: &ex.compute, lay, keys: &keys, stage, scalars: &scalars, streams: None };
    // the oracle runtime refreshed the KV push with `bootstrap_real` (the real-payload chain under FIDESLIB_BTS_REAL=1);
    // the graph does not record it (deliberate_bootstrap args carry slots / stc_first only), so the driver says it
    let fix_routes = |b: &mut Bound<'_, W>, say: bool| -> Res<()> {
        if cfg.bts.real.unwrap_or(false) {
            for (i, s) in prog.bts_sites.iter().enumerate() {
                if s.kind == pc_program::program::SiteKind::Deliberate && !s.route.real && pc_gpu::bts::RouteKind::of(&s.route, n) == pc_gpu::bts::RouteKind::StcFirst {
                    b.set_site_route(i as u32, pc_shadow::Route { real: true, ..s.route })?;
                    if say {
                        println!("[replay] site {i} ({}, deliberate): real-payload route (bootstrap_real)", s.var);
                    }
                }
            }
        }
        Ok(())
    };
    let mut bound = Bound::<W>::new(&prog, &env, &routes)?;
    fix_routes(&mut bound, true)?;
    let mut ring_bounds: Vec<Bound<'_, W>> = vec![];
    if o.ring {
        for r in 0..RING_BLOCKS {
            let mut p2 = prog.clone();
            p2.block = format!("{}r{r}", prog.block);
            let mut b = Bound::<W>::new(&p2, &env, &routes)?;
            fix_routes(&mut b, false)?;
            ring_bounds.push(b);
        }
    }
    let kv_events = EventPool::new(&dev, 64)?;
    let mut kvs = if o.kv {
        Some((pc_program::kv::PinnedKv::<W>::new(&arena, &dev, lay, KV_LAYERS, nq, 2, &kv_events)?, pc_program::kv::DeviceKv::<W>::new(&arena, lay, KV_LAYERS)?))
    } else {
        None
    };
    // the dump's plaintexts
    let tp = Instant::now();
    let ptdir = g0.dir.join("pts");
    let mut missing = vec![];
    for k in bound.pt_keys() {
        // the store key of a block's own plaintext is `<block>/<pt>`; the dump names it `<pt>@<level>`
        match dumped_pt(&ptdir, dump_pt_name(&k.name), k.level, nq, n) {
            Ok((rows, meta)) => store.insert_rows(&ex, k.clone(), meta, &rows)?,
            Err(e) => missing.push(format!("{}@{} ({e})", k.name, k.level)),
        }
    }
    if !missing.is_empty() {
        return Err(format!("{} plaintext(s) not in the dump: {}", missing.len(), missing.iter().take(8).cloned().collect::<Vec<_>>().join("; ")).into());
    }
    bound.resolve_pts(&store)?;
    println!("[replay] {} plaintexts from the dump in {:.1} s", bound.pt_keys().len(), tp.elapsed().as_secs_f64());
    let mut xcopy = pool.acquire(1, CtMeta::new(0, 1))?;
    ex.compute.sync()?;
    dev.synchronize()?;
    arena.freeze();
    let used0 = arena.used_bytes();
    let (free0, _) = dev.mem_info()?;
    println!("[replay] setup {:.1} s; arena {:.2} of {:.2} GB (frozen); device free {:.2} GB", t0.elapsed().as_secs_f64(), used0 as f64 / 1e9, arena.capacity() as f64 / 1e9, free0 as f64 / 1e9);

    // ── helpers ───────────────────────────────────────────────────────────────────────────────────────────────────
    let encoder = Encoder::new(&chain);
    let hin: Vec<HostCt> = o.inputs.iter().map(|nm| cts.ct(nm)).collect();
    if hin.len() != bound.inputs.len() {
        return Err(format!("{} input name(s) for {} program inputs ({:?})", hin.len(), bound.inputs.len(), bound.inputs.iter().map(|x| &x.0).collect::<Vec<_>>()).into());
    }
    for (i, h) in hin.iter().enumerate() {
        let m = meta_of(h, nq);
        let want = bound.inputs[i].2;
        println!("[replay] input {} <- dump {}: level {} deg {} (compiled level {} deg {})", bound.inputs[i].0, o.inputs[i], m.level, m.deg, want.level, want.deg);
        if (m.level, m.deg) != (want.level, want.deg) {
            ok = false;
            println!("[replay]   FAIL: input metadata differs from the compiled input");
        }
    }
    let upload_inputs = |bound: &mut Bound<'_, W>, ex: &Executor<'_, W>| -> Res<()> {
        for (i, h) in hin.iter().enumerate() {
            let c = bound.input_mut(i)?;
            for part in 0..2 {
                c.buf.upload_rows(&ex.compute, 0, part, &h.ids, &h.c[part])?;
            }
            c.meta = meta_of(h, nq);
        }
        Ok(())
    };
    let words_of = |c: &Ct<'_, W>, ex: &Executor<'_, W>| -> Res<Vec<u32>> {
        let top = c.top(ctx.shape)?;
        let ids: Vec<usize> = (0..=top).collect();
        let mut v = c.buf.download_rows(&ex.compute, 0, 0, &ids)?;
        v.extend(c.buf.download_rows(&ex.compute, 0, 1, &ids)?);
        Ok(v)
    };
    let outputs_words = |bound: &Bound<'_, W>, ex: &Executor<'_, W>| -> Res<Vec<Vec<u32>>> { (0..bound.outputs.len()).map(|i| words_of(bound.output(i)?, ex)).collect() };
    // decrypt a (copied, realized) ciphertext
    let mut decrypt = |c: &Ct<'_, W>, ex: &mut Executor<'_, W>, ops: &mut Ops<'_, W>| -> Res<Vec<Complex>> {
        let x = xcopy.get_mut().ok_or("shared")?;
        let top = c.top(ctx.shape)?;
        let ids: Vec<usize> = (0..=top).collect();
        for part in 0..2 {
            let v = c.buf.download_rows(&ex.compute, 0, part, &ids)?;
            x.buf.upload_rows(&ex.compute, 0, part, &ids, &v)?;
        }
        x.meta = c.meta;
        ops.realize(&ctx, ex, &sh, std::slice::from_mut(x))?;
        let slots = if x.meta.slots == 0 { n / 2 } else { x.meta.slots as usize };
        let t = x.top(ctx.shape)?;
        if t + 1 >= 4 {
            let rows = dec4.decrypt_rows(&ctx, &ex.compute, x, Which::Dense)?;
            Ok(decode4(&encoder, &chain.q[..4], &rows[0], slots, x.meta.nf, n))
        } else {
            let rows = dec2.decrypt_rows(&ctx, &ex.compute, x, Which::Dense)?;
            Ok(encoder.decode_slots(&rows[0], slots, Payload::Complex, x.meta.nf)?)
        }
    };
    let has = |nm: &str| cts.exists(&format!("{nm}.slots.npy"));
    let locks: Vec<&dyn CaptureLock> = dense.last.iter().map(|c| c as &dyn CaptureLock).collect();
    macro_rules! rt {
        () => {
            Rt { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work, routes: &routes, source: KskSource::Regen, locks: &locks, finish: None, side: None, coeff: None }
        };
    }

    // ── 0. optional: bootstraps in isolation; every dumped var after its step (optionally with the oracle's
    //       refresh outputs injected, which isolates the non-refresh replay) ─────────────────────────────────────────
    let bd = g0.bts();
    let bts_steps: Vec<usize> = bound.steps.iter().enumerate().filter(|(_, s)| matches!(s.op, StepOp::Bootstrap { .. })).map(|(i, _)| i).collect();
    if o.debug {
        println!("[replay] refreshes in isolation (our route on the dumped input vs the dumped output; raw, realized on a copy):");
        println!("  k | step | var | route (slots, cf) | dump (slots, cf) | in level deg | ours vs old rel (log2) | ours meta == dump");
        let mut y = pool.acquire(1, CtMeta::new(0, 1))?;
        let mut z = pool.acquire(1, CtMeta::new(0, 1))?;
        let deg = chain_deg(&ctx)?;
        for (k, &i) in bts_steps.iter().enumerate() {
            let k = k + 1;
            let StepOp::Bootstrap { route, kind, .. } = bound.steps[i].op.clone() else { continue };
            if !bd.exists(&format!("{k}.json")) {
                continue;
            }
            let j = bd.json(&format!("{k}.json"));
            let dcf = j["state"]["correction"].as_u64().unwrap_or(0) as u32 + deg;
            let hin = bd.ct(&format!("{k}.in"));
            let hout = bd.ct(&format!("{k}.out"));
            let ym = y.get_mut().ok_or("shared")?;
            for part in 0..2 {
                ym.buf.upload_rows(&ex.compute, 0, part, &hin.ids, &hin.c[part])?;
            }
            ym.meta = meta_of(&hin, nq);
            ym.meta.slots = bound.slots[bound.steps[i].a as usize].meta.slots;
            let r = routes.get(kind).ok_or("route")?;
            {
                let mut cx = pc_gpu::bts::BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                r.run(&mut cx, std::slice::from_mut(ym), &route)?;
            }
            let zm = z.get_mut().ok_or("shared")?;
            for part in 0..2 {
                zm.buf.upload_rows(&ex.compute, 0, part, &hout.ids, &hout.c[part])?;
            }
            zm.meta = meta_of(&hout, nq);
            zm.meta.slots = ym.meta.slots;
            let ours = decrypt(ym, &mut ex, &mut ops)?;
            let theirs = decrypt(zm, &mut ex, &mut ops)?;
            let rr = rel(&ours, &theirs);
            let mo = meta_of(&hout, nq);
            println!(
                "  {k} | {} | {} | ({}, {:?}) | ({}, {dcf}) | {} {} | {rr:.3e} ({:.1}) | {}",
                j["step"].as_str().unwrap_or(""),
                bound.names[bound.steps[i].var as usize],
                route.slots,
                route.cf,
                j["slots"],
                hin.level,
                hin.noise_level,
                rr.log2(),
                ym.meta.level == mo.level && ym.meta.deg == mo.deg
            );
        }
        // which refresh outputs are taken from the oracle: none; SPRU (fresh key material); dense StC-first (own
        // input-side plaintexts); both (= every route we cannot reproduce word for word: SparseB stays ours); all
        let modes: Vec<&str> = if o.oracle_bts { vec!["none", "spru", "dense", "spru+dense", "all"] } else { vec!["none"] };
        for mode in modes {
            upload_inputs(&mut bound, &ex)?;
            let mut rep = Replayer::new(ReplayCfg { use_graphs: false, check_meta: true, graphs_per_block: 2 });
            let mut rows: Vec<(usize, String, String, f64, u32, u8)> = vec![];
            let mut dbg_dec = |c: &Ct<'_, W>, ex: &mut Executor<'_, W>, ops: &mut Ops<'_, W>| decrypt(c, ex, ops);
            let bts_steps = &bts_steps;
            let mode = mode;
            let bd = &bd;
            let mut hook = |i: usize, b: &mut Bound<'_, W>, rt: &mut Rt<'_, '_, W, M>| -> pc_program::Result<()> {
                let st = b.steps[i].clone();
                let e = |x: Box<dyn std::error::Error>| pc_program::Error::Replay { block: "debug".into(), msg: x.to_string() };
                use pc_gpu::bts::RouteKind as RK;
                let inj = match (&st.op, mode) {
                    (StepOp::Bootstrap { .. }, "all") => true,
                    (StepOp::Bootstrap { kind: RK::Spru { .. }, .. }, "spru" | "spru+dense") => true,
                    (StepOp::Bootstrap { kind: RK::StcFirst | RK::Dense, .. }, "dense" | "spru+dense") => true,
                    _ => false,
                };
                if inj {
                    if let Some(k) = bts_steps.iter().position(|&x| x == i) {
                        let h = bd.ct(&format!("{}.out", k + 1));
                        let c = &mut b.slots[st.out as usize];
                        let slots = c.meta.slots;
                        for part in 0..2 {
                            c.buf.upload_rows(&rt.ex.compute, 0, part, &h.ids, &h.c[part]).map_err(|x| e(x.into()))?;
                        }
                        c.meta = meta_of(&h, nq);
                        c.meta.slots = slots;
                    }
                }
                let var = &b.names[st.var as usize];
                if let Some(dn) = dump_name(var, &has) {
                    let c = &b.slots[st.out as usize];
                    let ours = dbg_dec(c, rt.ex, rt.ops).map_err(e)?;
                    let theirs = read_c16(&cts.dir.join(format!("{dn}.slots.npy"))).map_err(e)?;
                    rows.push((i, var.clone(), dn, rel(&ours, &theirs), c.meta.level, c.meta.deg));
                }
                Ok(())
            };
            let info = rep.eager_with(&mut bound, &mut rt!(), &mut hook)?;
            let worst = rows.iter().filter(|r| r.1 != "v_102").fold(0f64, |x, r| x.max(r.3));
            let pre = rows.iter().find(|r| r.1 == format!("{}~pre", bound.outputs[0].0)).map(|r| r.3).unwrap_or(f64::NAN);
            let last = rows.last().map(|r| r.3).unwrap_or(f64::NAN);
            println!(
                "[replay] debug, oracle refresh outputs injected: {mode}: {} dumped vars compared (eager, meta mismatches {}), worst (v_102 aside) 2^{:.1}; exit before its refresh 2^{:.1}, exit 2^{:.1}",
                rows.len(),
                info.meta_mismatches,
                worst.log2(),
                pre.log2(),
                last.log2()
            );
            println!("  step | var | dump | rel err (log2) | level deg");
            for (i, v, d, r, l, dg) in &rows {
                println!("  {i} | {v} | {d} | {r:.3e} ({:.1}) | {l} {dg}", r.log2());
            }
        }
    }

    // ── 1. eager vs the oracle ────────────────────────────────────────────────────────────────────────────────────
    let exit_name = bound.outputs[0].0.clone();
    let out_dump = dump_name(&exit_name, &has).ok_or(format!("the exit var {exit_name} has no dumped slots"))?;
    let oracle = read_c16(&cts.dir.join(format!("{out_dump}.slots.npy")))?;
    let mut eager = Replayer::new(ReplayCfg { use_graphs: false, check_meta: true, graphs_per_block: 2 });
    upload_inputs(&mut bound, &ex)?;
    let info = eager.run(&mut bound, &mut rt!())?;
    ex.compute.sync()?;
    let e1 = outputs_words(&bound, &ex)?;
    let exit_meta = bound.output(0)?.meta;
    let ours = decrypt(bound.output(0)?, &mut ex, &mut ops)?;
    let r = rel(&ours, &oracle);
    let amax = oracle.iter().fold(0f64, |x, z| x.max(z.abs()));
    println!("[replay] eager run 1: {} steps, host {:.1} ms, metadata mismatches vs compiled {}", bound.steps.len(), info.host_us / 1e3, info.meta_mismatches);
    for m in &info.first_mismatches {
        println!("[replay]   {m}");
    }
    let pass = r <= 2f64.powi(-20);
    ok &= pass && info.meta_mismatches == 0;
    println!(
        "[replay] GATE block output {exit_name} (level {} deg {}) vs oracle {out_dump} (max |slot| {amax:.2}): max|ours - oracle| / max|oracle| = {r:.3e} = 2^{:.2} (<= 2^-20: {})",
        exit_meta.level,
        exit_meta.deg,
        r.log2(),
        if pass { "PASS" } else { "FAIL" }
    );
    for (i, (nm, _, _)) in bound.outputs.iter().enumerate().skip(1) {
        if let Some(dn) = dump_name(nm, &has) {
            let ours = decrypt(bound.output(i)?, &mut ex, &mut ops)?;
            let theirs = read_c16(&cts.dir.join(format!("{dn}.slots.npy")))?;
            println!("[replay]   other output {nm} vs {dn}: rel {:.3e}", rel(&ours, &theirs));
        }
    }

    // ── 2. graph == eager ─────────────────────────────────────────────────────────────────────────────────────────
    let cmp = |a: &[Vec<u32>], b: &[Vec<u32>]| -> (usize, usize) {
        let mut bad = 0;
        let mut tot = 0;
        for (x, y) in a.iter().zip(b) {
            tot += x.len().max(y.len());
            bad += x.iter().zip(y).filter(|(p, q)| p != q).count() + x.len().abs_diff(y.len());
        }
        (bad, tot)
    };
    upload_inputs(&mut bound, &ex)?;
    eager.run(&mut bound, &mut rt!())?;
    ex.compute.sync()?;
    let (b2, t2) = cmp(&outputs_words(&bound, &ex)?, &e1);
    println!("[replay] eager run 2 == eager run 1: {b2} of {t2} output words differ");
    ok &= b2 == 0;
    let mut graph = Replayer::new(ReplayCfg { use_graphs: true, check_meta: false, graphs_per_block: 2 });
    upload_inputs(&mut bound, &ex)?;
    let i1 = graph.run(&mut bound, &mut rt!())?;
    ex.compute.sync()?;
    let (b3, _) = cmp(&outputs_words(&bound, &ex)?, &e1);
    upload_inputs(&mut bound, &ex)?;
    let i2 = graph.run(&mut bound, &mut rt!())?;
    ex.compute.sync()?;
    let (b4, t4) = cmp(&outputs_words(&bound, &ex)?, &e1);
    let gm = bound.output(0)?.meta;
    println!(
        "[replay] graph run 1 ({:?}, capture {:.1} ms, {} constant slots leased): {b3} words differ; graph run 2 ({:?}): {b4} of {t4} output words differ from eager; exit meta level {} deg {} (eager {} {})",
        i1.served,
        graph.stats.capture_ms,
        ops.ring.leased(),
        i2.served,
        gm.level,
        gm.deg,
        exit_meta.level,
        exit_meta.deg
    );
    let gpass = b3 == 0 && b4 == 0 && i1.served == Served::Captured && i2.served == Served::Replayed && gm == exit_meta;
    println!("[replay] GATE graph == eager bit-exact: {}", if gpass { "PASS" } else { "FAIL" });
    ok &= gpass;

    // ── 3. timing, host time, allocations ─────────────────────────────────────────────────────────────────────────
    // allocations: every run after the warm-up (eager runs, the capture + instantiation above) must leave the
    // device's free memory and the arena untouched
    ex.compute.sync()?;
    let (free_w, _) = dev.mem_info()?;
    println!("[replay] device free after setup {free0} B, after the warm-up runs + capture {free_w} B (graph instantiation: {:.1} MB, driver-side)", (free0 as f64 - free_w as f64) / 1e6);
    let free0 = free_w;
    let ev0 = Event::timing(&dev)?;
    let ev1 = Event::timing(&dev)?;
    macro_rules! time {
        ($rep:expr) => {{
            let (mut dms, mut hus, mut wall, mut lus) = (vec![], vec![], vec![], vec![]);
            for _ in 0..o.reps {
                upload_inputs(&mut bound, &ex)?;
                ex.compute.sync()?;
                let tw = Instant::now();
                ev0.record(&ex.compute)?;
                let info = $rep.run(&mut bound, &mut rt!())?;
                ev1.record(&ex.compute)?;
                ev1.sync()?;
                wall.push(tw.elapsed().as_secs_f64() * 1e3);
                dms.push(ev0.elapsed_ms(&ev1)? as f64);
                hus.push(info.host_us);
                lus.push(info.launch_us);
            }
            (median(dms), median(hus), median(wall), median(lus))
        }};
    }
    let (ed, eh, ew, _) = time!(eager);
    let (gd, gh, gw, gl) = time!(graph);
    let (ed2, eh2, ew2, _) = time!(eager);
    let (gd2, gh2, gw2, gl2) = time!(graph);
    let launches: usize = bound.steps.len();
    println!("[replay] timing ({} reps each, medians; two rounds):", o.reps);
    println!("  mode  | device ms/block | host ms issuing (run call) | wall ms (run + device) | steps");
    println!("  eager | {ed:.2} / {ed2:.2} | {:.3} / {:.3} | {ew:.2} / {ew2:.2} | {launches}", eh / 1e3, eh2 / 1e3);
    println!("  graph | {gd:.2} / {gd2:.2} | {:.3} / {:.3} (cudaGraphLaunch {:.3} / {:.3}) | {gw:.2} / {gw2:.2} | 1 launch", gh / 1e3, gh2 / 1e3, gl / 1e3, gl2 / 1e3);
    let hpass = gh < 1000.0 && gh2 < 1000.0;
    println!("[replay] GATE host time inside a block (graph) < 1 ms: {:.3} ms: {}", gh.max(gh2) / 1e3, if hpass { "PASS" } else { "FAIL" });
    ok &= hpass;
    let (free1, _) = dev.mem_info()?;
    let used1 = arena.used_bytes();
    let apass = free1 == free0 && used1 == used0 && arena.is_frozen();
    println!(
        "[replay] GATE 0 device allocations after setup ({} eager + {} graph runs): device free {} -> {} B (delta {}), arena used {} -> {} (frozen {}): {}",
        eager.stats.eager,
        graph.stats.replays + graph.stats.captures,
        free0,
        free1,
        free0 as i64 - free1 as i64,
        used0,
        used1,
        arena.is_frozen(),
        if apass { "PASS" } else { "FAIL" }
    );
    ok &= apass;
    // ── optional: weight ring ─────────────────────────────────────────────────────────────────────────────────
    if o.ring {
        let src = G0Source::load(&g0.dir.join("pts"), &ring_bounds[0].pt_keys())?;
        let mut ring = WeightRing::new(RingCfg { n_in_flight: 2 })?;
        let refs: Vec<&Bound<'_, W>> = ring_bounds.iter().collect();
        println!("[replay] ring: {} bindings of {} cycled through {} frames; stage-ahead slots needed {} (store {} slots)", ring_bounds.len(), prog.block, ring.cfg.n_in_flight, WeightRing::slots_needed(&refs, 2), store.cfg.slots);
        ring.prewarm(&mut store, &ctx, &ex, &sh)?;
        let mut rrep = Replayer::new(ReplayCfg { use_graphs: true, check_meta: false, graphs_per_block: 2 });
        let nb = ring_bounds.len();
        let runs = 4 * nb;
        // prologue: block 0's plaintexts
        ring.stage(&mut store, &ex, &sh, 0, &ring_bounds[0], &src)?;
        ring.adopt(&mut store, &ctx, &ex, 0, &mut ring_bounds[0])?;
        ex.compute.sync()?;
        let tw = Instant::now();
        let mut hosts = vec![];
        let mut stage_ms = vec![];
        let mut adopt_ms = vec![];
        let mut run_ms = vec![];
        let mut served = vec![];
        let mut t_replay: Option<Instant> = None;
        for seq in 0..runs {
            let th = Instant::now();
            let nx = (seq + 1) % nb;
            // the next block's plaintexts: host encode while the device still runs the previous launch
            if seq == 2 * nb {
                // every binding has a graph for both frames: steady state from here
                ex.compute.sync()?;
                t_replay = Some(Instant::now());
            }
            let ts = Instant::now();
            if seq + 1 < runs {
                ring.stage(&mut store, &ex, &sh, seq + 1, &ring_bounds[nx], &src)?;
            }
            stage_ms.push(ts.elapsed().as_secs_f64() * 1e3);
            let cur = &mut ring_bounds[seq % nb];
            upload_inputs(cur, &ex)?;
            let tr = Instant::now();
            let info = rrep.run(cur, &mut rt!())?;
            run_ms.push(tr.elapsed().as_secs_f64() * 1e3);
            served.push(info.served);
            let ta = Instant::now();
            if seq + 1 < runs {
                ring.adopt(&mut store, &ctx, &ex, seq + 1, &mut ring_bounds[nx])?;
            }
            adopt_ms.push(ta.elapsed().as_secs_f64() * 1e3);
            hosts.push(th.elapsed().as_secs_f64() * 1e3);
        }
        ex.compute.sync()?;
        let wall = tw.elapsed().as_secs_f64() * 1e3;
        let steady = t_replay.map(|t| t.elapsed().as_secs_f64() * 1e3 / (runs - 2 * nb) as f64).unwrap_or(f64::NAN);
        let tail = |v: &[f64]| median(v[2 * nb..].to_vec());
        println!(
            "[replay] ring steady state (graph replays, {} runs): {steady:.1} ms/block wall; host per block: stage next {:.1} ms (encode pool), run {:.3} ms, adopt {:.2} ms; resident (no ring) graph {gd:.1} ms/block",
            runs - 2 * nb,
            tail(&stage_ms),
            tail(&run_ms),
            tail(&adopt_ms)
        );
        let mut worst_or: f64 = 0.0;
        let mut worst_e: f64 = 0.0;
        let eager_exit = {
            upload_inputs(&mut bound, &ex)?;
            eager.run(&mut bound, &mut rt!())?;
            ex.compute.sync()?;
            decrypt(bound.output(0)?, &mut ex, &mut ops)?
        };
        // the ring's graph replays vs an eager run on the same staged plaintexts (the last frame's binding)
        let ring_words: Vec<Vec<Vec<u32>>> = ring_bounds.iter().map(|b| outputs_words(b, &ex)).collect::<Res<_>>()?;
        let last = (runs - 1) % nb;
        upload_inputs(&mut ring_bounds[last], &ex)?;
        eager.run(&mut ring_bounds[last], &mut rt!())?;
        ex.compute.sync()?;
        let ref_words = outputs_words(&ring_bounds[last], &ex)?;
        let ring_bad: usize = ring_words.iter().map(|w| cmp(w, &ref_words).0).sum();
        for b in &ring_bounds {
            let v = decrypt(b.output(0)?, &mut ex, &mut ops)?;
            worst_or = worst_or.max(rel(&v, &oracle));
            worst_e = worst_e.max(rel(&v, &eager_exit));
        }
        println!("[replay] ring: {} pinned staging buffers, {} encode threads", store.cfg.ring, store.cfg.encode_threads);
        println!(
            "[replay] ring: {runs} block runs in {wall:.1} ms ({:.1} ms/block; resident graph {gd:.1}); host per block (stage next + run + adopt) median {:.2} ms; stage {:.0} ms total for {} items ({} resident hits, {} evicted); runs served {:?}; replays {} captures {} re-captures {}",
            wall / runs as f64,
            median(hosts.clone()),
            ring.stats.stage_ms,
            ring.stats.staged,
            ring.stats.hits,
            ring.stats.evicted,
            served.iter().map(|s| match s {
                Served::Captured => 'C',
                Served::Replayed => 'R',
                Served::Eager => 'E',
            }).collect::<String>(),
            rrep.stats.replays,
            rrep.stats.captures,
            rrep.stats.invalidations
        );
        println!(
            "[replay] ring: every binding's graph-replayed outputs vs an eager run on the same staged plaintexts: {ring_bad} words differ; exits (host-FFT plaintexts) vs the oracle 2^{:.2}, vs the dumped-row eager run 2^{:.2}",
            worst_or.log2(),
            worst_e.log2()
        );
        ok &= ring_bad == 0 && rrep.stats.invalidations == 0;
    }
    // ── optional: KV offload choreography ─────────────────────────────────────────────────────────────────────────
    if let Some((pkv, dkv)) = kvs.as_mut() {
        ok &= kv_check(pkv, &mut ex, &ctx, &sh, &mut ops, bound.output(0)?, "pinned")?;
        ok &= kv_check(dkv, &mut ex, &ctx, &sh, &mut ops, bound.output(0)?, "device")?;
    }
    let _ = (&mut xcopy,);
    println!("[replay] total {:.1} s; {}", t0.elapsed().as_secs_f64(), if ok { "ALL GATES PASS" } else { "SOME GATE FAILED" });
    Ok(ok)
}

#[allow(dead_code)]
fn _key(k: &PtKey) -> String {
    format!("{}@{}", k.name, k.level)
}
