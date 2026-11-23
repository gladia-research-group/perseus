//! Two GPUs, independent bootstraps: the dense route on K ciphertexts, one GPU running them as one K-batch against
//! K/2 on each GPU at the same time.
//!
//!   pc-bench bts --devices 0,1 [--k 2,4] [--reps 10] [--chain n32a4] [--level 44] [--cf 6] [--real] [--graph]
//!
//! One host thread drives both devices (`Device::make_current` before every device's work). Each device gets the
//! complete setup from the same seeds, so secret, keys, tables and encryption randomness are identical. The K inputs
//! are encrypted once on the first device. Per K (even), device ms median of `reps` after 2 warm-ups:
//!
//! * `one GPU, batch of K` and `batch of K/2`: the single-GPU baseline (the first device).
//! * `two GPUs`: K/2 per device, inputs already resident (the second device got its half by a peer copy outside the
//!   timing). The span runs on the first device's clock: it waits for the second device's last event, then records
//!   the end event.
//! * `offload`: all K inputs on the first device; the second device pulls its half, refreshes it and pushes the
//!   outputs back while the first device refreshes its own half (the copies of a remote refresh in a session).
//!
//! `issue` is host time spent issuing (an eager refresh issues every launch from this thread; `--graph` replays one
//! captured graph per device and batch instead). Correctness: every output of every mode, on either device and
//! after the copy back, must equal the batch-of-K output of the same input word for word, with the same metadata.
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::keys::{Prng, SecretKey};
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::{Arena, Compute, Device, Event, GraphExec, Stream};
use pc_gpu::bts::cts::LT_REGEN_DIGITS;
use pc_gpu::bts::dense::{shadow_of, DensePlan, DenseRoute};
use pc_gpu::bts::{BtsCx, BtsRoute, BtsSetup, BtsWork, BtsWorkShape};
use pc_gpu::decrypt::{Decryptor, Which};
use pc_gpu::encrypt::Encryptor;
use pc_gpu::keygen::{test_prng, KeyGen};
use pc_gpu::keys::{DevSecret, KeySet, KeySetSpec, KskLayout, KskShape, RotSpec};
use pc_gpu::ks::{KsPlans, KskSource};
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::poly::CtBuf;
use pc_gpu::{Context, Ct, Executor, MemCfg, MemPlan, PoolClass};
use pc_shadow::{CtMeta, Route, Shadow};
use std::collections::BTreeSet;
use std::process::ExitCode;
use std::time::Instant;

type W = u32;
type M = Barrett64;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct Opts {
    devices: [i32; 2],
    ks: Vec<usize>,
    reps: usize,
    chain: String,
    level: u32,
    cf: u32,
    real: bool,
    graph: bool,
}

fn parse(a: &[String]) -> Res<Opts> {
    let mut o = Opts { devices: [0, 1], ks: vec![2, 4], reps: 10, chain: "n32a4".into(), level: 44, cf: 6, real: false, graph: false };
    let mut i = 0;
    while i < a.len() {
        let v = a.get(i + 1).cloned().unwrap_or_default();
        match a[i].as_str() {
            "--devices" => {
                let d: Vec<i32> = v.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?;
                if d.len() != 2 || d[0] == d[1] {
                    return Err(format!("--devices {v}: two distinct visible ordinals (e.g. 0,1)").into());
                }
                o.devices = [d[0], d[1]];
            }
            "--k" => o.ks = v.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?,
            "--reps" => o.reps = v.parse()?,
            "--chain" => o.chain = v,
            "--level" => o.level = v.parse()?,
            "--cf" => o.cf = v.parse()?,
            "--real" | "--graph" => {
                if a[i] == "--real" {
                    o.real = true;
                } else {
                    o.graph = true;
                }
                i += 1;
                continue;
            }
            x => return Err(format!("unknown argument {x}").into()),
        }
        i += 2;
    }
    if o.ks.is_empty() || o.ks.iter().any(|&k| k == 0 || k % 2 != 0) {
        return Err(format!("--k {:?}: even batch sizes (K/2 per device)", o.ks).into());
    }
    if o.reps == 0 {
        return Err("--reps: at least 1".into());
    }
    Ok(o)
}

pub fn cmd(args: &[String]) -> ExitCode {
    match parse(args).and_then(|o| run(&o)) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => {
            eprintln!("[mgpu] FAIL: an output differs from the single-GPU batch");
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("[mgpu] error: {e}");
            ExitCode::from(2)
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn ms(t: std::time::Duration) -> f64 {
    t.as_secs_f64() * 1e3
}

/// Uniform real payload in [-amp, amp], max anchored to amp (pc-bench's `rand_real`).
fn rand_real(seed: u64, n: usize, amp: f64) -> Vec<Complex> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut v: Vec<f64> = (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * amp
        })
        .collect();
    let m = v.iter().fold(0f64, |a, &b| a.max(b.abs()));
    v.iter_mut().for_each(|y| *y *= amp / m);
    v.into_iter().map(|y| Complex::new(y, 0.0)).collect()
}

/// What every device's setup shares (host side, built once).
struct Shared {
    chain: Chain,
    plan: DensePlan,
    spec: KeySetSpec,
    shapes: Vec<KskShape>,
    wshape: BtsWorkShape,
    kmax: usize,
}

/// One device's complete bootstrap setup.
struct Rig<'a> {
    dev: &'a Device,
    ex: Executor<'a, W>,
    ctx: Context<'a, W, M>,
    kplans: KsPlans<'a>,
    ops: Ops<'a, W>,
    work: BtsWork<'a, W>,
    _keys: KeySet<'a, W>,
    sec: DevSecret<'a, W>,
    encr: Encryptor<'a, W>,
    dec: Decryptor<'a, W>,
    route: DenseRoute<'a, W, M>,
    sh: Shadow,
    prng: Prng,
    /// the batch a refresh runs on (`x.buf.k` ciphertexts)
    x: Ct<'a, W>,
    /// pristine inputs (reloaded into `x` before every timed refresh)
    inp: Ct<'a, W>,
    /// ciphertexts of this device refreshed by the other one (offload)
    stage: Ct<'a, W>,
    /// timing events of this device
    ev: Vec<Event>,
    /// ordering event of this device (the fork / join of a split capture)
    order: Event,
}

fn rig<'a>(dev: &'a Device, arena: &'a Arena, mp: &MemPlan, s: &Shared) -> Res<Rig<'a>> {
    dev.make_current()?;
    let t0 = Instant::now();
    let (lay, chain) = (mp.lay, &s.chain);
    let mut ex = Executor::<W>::new(dev, arena, mp)?;
    ex.ks.carve_regen(arena, &lay)?;
    let ctx = Context::<W, M>::new(arena, &ex.copy, chain.clone(), mp)?;
    let kplans = KsPlans::new(arena, &ex.compute, &ctx, s.kmax)?;
    let ops = Ops::new(arena, dev, &ex.compute, &ctx, OpsCfg::default())?;
    let work = BtsWork::<W>::new(arena, lay, s.wshape)?;
    let mut keys = KeySet::<W>::carve(arena, chain, &s.spec, KskLayout::Dense)?;
    // the same seed on every device: identical secret, keys and encryption randomness
    let mut prng = test_prng(0x6a);
    let sk = SecretKey::sample(&mut prng, lay.n)?;
    let mut sec = DevSecret::<W>::carve(arena, lay)?;
    sec.load(&ctx, &ex.compute, &sk)?;
    let mut kg = KeyGen::<W>::new(arena, &ex.compute, chain, &s.shapes)?;
    let tk = kg.generate_set(&ctx, &mut ex, &sec, &mut keys, &mut prng)?;
    let encr = Encryptor::<W>::new(arena, &ex.compute, lay, false, &mut prng)?;
    let mut dec = Decryptor::<W>::new(arena, &ex.compute, lay, chain.level_unit, s.kmax)?;
    dec.load_secret(&ex.compute, &sec)?;
    let mut route = s.plan.build(arena, &ex.compute, &ctx, &keys, ex.ks.stage_ptrs())?;
    route.set_lt_baby(KskSource::Regen, LT_REGEN_DIGITS)?;
    let ct = |what: &'static str| -> Res<Ct<'a, W>> { Ok(Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(arena, what, lay, s.kmax)?)) };
    let (x, inp, stage) = (ct("mgpu.x")?, ct("mgpu.inp")?, ct("mgpu.stage")?);
    let ev = (0..6).map(|_| Event::timing(dev)).collect::<Result<Vec<_>, _>>()?;
    let order = Event::new(dev)?;
    ex.compute.sync()?;
    arena.freeze();
    let sh = shadow_of(&ctx.scale);
    println!(
        "[mgpu] device {}: {} keys in {:.1} s, setup {:.1} s, arena {:.1} of {:.1} GB",
        dev.ordinal(),
        tk.keys,
        tk.total,
        t0.elapsed().as_secs_f64(),
        arena.used_bytes() as f64 / 1e9,
        arena.capacity() as f64 / 1e9
    );
    Ok(Rig { dev, ex, ctx, kplans, ops, work, _keys: keys, sec, encr, dec, route, sh, prng, x, inp, stage, ev, order })
}

/// Copy ciphertexts `src_c..src_c + count` of `src` (memory of `sd`) into slots `dst_c..` of `dst` (memory of `dd`):
/// rows `0..rows` of both polynomials, ordered on `s`. Returns the bytes moved.
///
/// # Safety
/// As `DBuf::copy_from_peer`: `s` is ordered after the writes of `src`, nothing writes `src` before the copy
/// completed, and `dst` and `src` are distinct buffers.
#[allow(clippy::too_many_arguments)]
unsafe fn copy_cts(s: &Stream<Compute>, dst: &mut Ct<'_, W>, dd: &Device, dst_c: usize, src: &Ct<'_, W>, sd: &Device, src_c: usize, count: usize, rows: usize) -> Res<usize> {
    let n = src.buf.lay.n;
    for c in 0..count {
        for part in 0..2 {
            let (od, os) = (dst.buf.offset(dst_c + c, part, 0), src.buf.offset(src_c + c, part, 0));
            dst.buf.buf.copy_from_peer(s, dd, od, &src.buf.buf, sd, os, rows * n)?;
        }
    }
    Ok(2 * count * rows * n * std::mem::size_of::<W>())
}

/// Rows `0..rows` of both polynomials of ciphertext `c` (synchronous read on `x`'s device).
fn words(dev: &Device, s: &Stream<Compute>, x: &Ct<'_, W>, c: usize, rows: usize) -> Res<Vec<W>> {
    dev.make_current()?;
    let ids: Vec<usize> = (0..rows).collect();
    let mut v = x.buf.download_rows(s, c, 0, &ids)?;
    v.extend(x.buf.download_rows(s, c, 1, &ids)?);
    Ok(v)
}

impl<'a> Rig<'a> {
    /// `x` <- inputs `from..from + k` of `inp` (same device), metadata `meta`; synchronized.
    fn load(&mut self, from: usize, k: usize, meta: CtMeta, rows: usize) -> Res<()> {
        self.dev.make_current()?;
        let Rig { dev, ex, x, inp, .. } = self;
        x.buf.k = k;
        x.meta = meta;
        // SAFETY: inp was written and synchronized at setup and is never written after; x is a distinct buffer
        unsafe { copy_cts(&ex.compute, x, dev, 0, inp, dev, from, k, rows)? };
        ex.compute.sync()?;
        Ok(())
    }

    /// Issue one refresh of `x` on this device's compute stream: eager (`g` None) or a captured graph replay (then
    /// `x.meta` <- `out`, the metadata the capture's host side produced).
    fn go(&mut self, r: &Route, g: Option<&GraphExec>, out: CtMeta) -> Res<()> {
        let cur = pc_cuda::device::current_ordinal()?;
        if cur != self.dev.ordinal() {
            return Err(format!("issuing on device {} while device {cur} is current", self.dev.ordinal()).into());
        }
        if let Some(g) = g {
            g.launch(&self.ex.compute)?;
            self.x.meta = out;
            return Ok(());
        }
        let Rig { ctx, ex, sh, kplans, ops, work, route, x, .. } = self;
        let mut cx = BtsCx { ctx: &*ctx, ex, sh: &*sh, ks: &*kplans, ops, work };
        route.run(&mut cx, std::slice::from_mut(x), r)?;
        cx.ex.join_aux()?;
        Ok(())
    }

    /// Capture one refresh of the current `x` batch (k, metadata) into a graph; returns it with the output metadata.
    fn capture(&mut self, r: &Route) -> Res<(GraphExec, CtMeta)> {
        self.dev.make_current()?;
        let sp: *const Stream<Compute> = &self.ex.compute;
        let Rig { ctx, ex, sh, kplans, ops, work, route, x, .. } = self;
        let mut cx = BtsCx { ctx: &*ctx, ex, sh: &*sh, ks: &*kplans, ops, work };
        // SAFETY: as pc-bench --graph: the capture reads the stream handle only; route.run issues on that stream and
        // on aux, forked from it and joined back before the capture ends
        let (g, ()) = unsafe { &*sp }.capture(|_| {
            route.run(&mut cx, std::slice::from_mut(&mut *x), r).and_then(|_| cx.ex.join_aux()).map_err(|_| pc_cuda::Error::Unavailable("bootstrap under capture"))
        })?;
        Ok((g, x.meta))
    }
}

#[derive(Clone, Copy, Default)]
struct T {
    /// device ms on the first device's clock (span for two-device modes)
    dev: f64,
    issue: f64,
    wall: f64,
    /// per device / per phase device ms (mode-specific)
    a: f64,
    b: f64,
    c: f64,
    d: f64,
}

fn med(v: &[T]) -> T {
    let f = |g: fn(&T) -> f64| median(v.iter().map(g).collect());
    T { dev: f(|t| t.dev), issue: f(|t| t.issue), wall: f(|t| t.wall), a: f(|t| t.a), b: f(|t| t.b), c: f(|t| t.c), d: f(|t| t.d) }
}

/// One device, one batch of `k` (inputs `0..k`).
fn one_gpu(r: &mut Rig<'_>, k: usize, m_in: CtMeta, rows_in: usize, route: &Route, g: Option<&(GraphExec, CtMeta)>, reps: usize) -> Res<T> {
    let mut v = Vec::new();
    for rep in 0..reps + 2 {
        r.load(0, k, m_in, rows_in)?;
        let t = Instant::now();
        r.ev[0].record(&r.ex.compute)?;
        r.go(route, g.map(|g| &g.0), g.map_or(m_in, |g| g.1))?;
        r.ev[1].record(&r.ex.compute)?;
        let issue = ms(t.elapsed());
        r.ev[1].sync()?;
        let wall = ms(t.elapsed());
        if rep >= 2 {
            v.push(T { dev: r.ev[0].elapsed_ms(&r.ev[1])? as f64, issue, wall, ..T::default() });
        }
    }
    Ok(med(&v))
}

/// K/2 per device, inputs resident (`r0.inp` 0..h, `r1.inp` 0..h = inputs h..k). a, b = each device's own ms.
#[allow(clippy::too_many_arguments)]
fn two_gpu(r0: &mut Rig<'_>, r1: &mut Rig<'_>, h: usize, m_in: CtMeta, rows_in: usize, route: &Route, g: Option<(&(GraphExec, CtMeta), &(GraphExec, CtMeta))>, reps: usize) -> Res<T> {
    let mut v = Vec::new();
    for rep in 0..reps + 2 {
        r0.load(0, h, m_in, rows_in)?;
        r1.load(0, h, m_in, rows_in)?;
        let t = Instant::now();
        r0.dev.make_current()?;
        r0.ev[0].record(&r0.ex.compute)?;
        r0.go(route, g.map(|g| &g.0 .0), g.map_or(m_in, |g| g.0 .1))?;
        r0.ev[1].record(&r0.ex.compute)?;
        r1.dev.make_current()?;
        r1.ev[0].record(&r1.ex.compute)?;
        r1.go(route, g.map(|g| &g.1 .0), g.map_or(m_in, |g| g.1 .1))?;
        r1.ev[1].record(&r1.ex.compute)?;
        r0.dev.make_current()?;
        r0.ex.compute.wait(&r1.ev[1])?;
        r0.ev[2].record(&r0.ex.compute)?;
        let issue = ms(t.elapsed());
        r0.ev[2].sync()?;
        r1.ev[1].sync()?;
        let wall = ms(t.elapsed());
        if rep >= 2 {
            v.push(T { dev: r0.ev[0].elapsed_ms(&r0.ev[2])? as f64, issue, wall, a: r0.ev[0].elapsed_ms(&r0.ev[1])? as f64, b: r1.ev[0].elapsed_ms(&r1.ev[1])? as f64, ..T::default() });
        }
    }
    Ok(med(&v))
}

/// All K inputs on device 0: device 1 pulls inputs h..k (from `r0.stage`), refreshes them and pushes the outputs back
/// into `r0.stage`, while device 0 refreshes 0..h in `r0.x`. a = device 0's own ms; b, c, d = device 1's copy in,
/// refresh, copy back ms. Returns the timing and the bytes moved (in, out).
#[allow(clippy::too_many_arguments)]
fn offload(r0: &mut Rig<'_>, r1: &mut Rig<'_>, h: usize, m_in: CtMeta, rows_in: usize, m_out: CtMeta, rows_out: usize, route: &Route, g: Option<(&(GraphExec, CtMeta), &(GraphExec, CtMeta))>, reps: usize) -> Res<(T, usize, usize)> {
    let (d0, d1) = (r0.dev, r1.dev);
    let mut v = Vec::new();
    let (mut b_in, mut b_out) = (0, 0);
    for rep in 0..reps + 2 {
        r0.load(0, h, m_in, rows_in)?;
        {
            let Rig { ex, stage, inp, .. } = &mut *r0;
            stage.buf.k = h;
            stage.meta = m_in;
            // SAFETY: inp is synchronized setup data; stage is a distinct buffer, synchronized below
            unsafe { copy_cts(&ex.compute, stage, d0, 0, inp, d0, h, h, rows_in)? };
            ex.compute.sync()?;
        }
        r1.x.buf.k = h;
        r1.x.meta = m_in;
        let t = Instant::now();
        r0.ev[0].record(&r0.ex.compute)?;
        d1.make_current()?;
        r1.ex.compute.wait(&r0.ev[0])?;
        r1.ev[0].record(&r1.ex.compute)?;
        // SAFETY: r0.stage is published by r0.ev[0], which device 1's stream waits on; device 0 touches stage again
        // only after waiting for r1.ev[3] (after the copy back)
        b_in = unsafe { copy_cts(&r1.ex.compute, &mut r1.x, d1, 0, &r0.stage, d0, 0, h, rows_in)? };
        r1.ev[1].record(&r1.ex.compute)?;
        r1.go(route, g.map(|g| &g.1 .0), g.map_or(m_in, |g| g.1 .1))?;
        r1.ev[2].record(&r1.ex.compute)?;
        // SAFETY: r1.x was written by the refresh just issued on the same stream; stage is not read by device 0
        // before r1.ev[3]
        b_out = unsafe { copy_cts(&r1.ex.compute, &mut r0.stage, d0, 0, &r1.x, d1, 0, h, rows_out)? };
        r1.ev[3].record(&r1.ex.compute)?;
        d0.make_current()?;
        r0.go(route, g.map(|g| &g.0 .0), g.map_or(m_in, |g| g.0 .1))?;
        r0.ev[1].record(&r0.ex.compute)?;
        r0.ex.compute.wait(&r1.ev[3])?;
        r0.ev[2].record(&r0.ex.compute)?;
        let issue = ms(t.elapsed());
        r0.ev[2].sync()?;
        r1.ev[3].sync()?;
        let wall = ms(t.elapsed());
        if rep >= 2 {
            let e = |x: &Event, y: &Event| x.elapsed_ms(y).map(|v| v as f64);
            v.push(T { dev: e(&r0.ev[0], &r0.ev[2])?, issue, wall, a: e(&r0.ev[0], &r0.ev[1])?, b: e(&r1.ev[0], &r1.ev[1])?, c: e(&r1.ev[1], &r1.ev[2])?, d: e(&r1.ev[2], &r1.ev[3])? });
        }
    }
    r0.stage.meta = m_out;
    Ok((med(&v), b_in, b_out))
}

/// One graph captured on device 0 holding both halves of a batch: device 0's stream forks onto device 1's (an event
/// recorded on device 0, waited on by device 1), device 1 pulls inputs h..k from `r0.stage`, refreshes them and pushes
/// the outputs back, device 0 refreshes 0..h in `r0.x`, and device 0 joins device 1 before the capture ends. Whether
/// CUDA captures this cross-device fork / join into one replayable graph is what this measures. Returns the graph or
/// the error that ended the capture.
#[allow(clippy::too_many_arguments)]
fn capture_split(r0: &mut Rig<'_>, r1: &mut Rig<'_>, h: usize, m_in: CtMeta, rows_in: usize, rows_out: usize, route: &Route) -> Res<GraphExec> {
    let (d0, d1) = (r0.dev, r1.dev);
    r0.x.buf.k = h;
    r0.x.meta = m_in;
    r1.x.buf.k = h;
    r1.x.meta = m_in;
    r0.stage.buf.k = h;
    d0.make_current()?;
    let sp: *const Stream<Compute> = &r0.ex.compute;
    let mut inner: Option<Box<dyn std::error::Error>> = None;
    let mut body = || -> Res<()> {
        r0.order.record(&r0.ex.compute)?;
        d1.make_current()?;
        r1.ex.compute.wait(&r0.order)?;
        // SAFETY: inside the graph, device 1's stream is ordered after the fork event (after whatever filled stage
        // before the launch) and device 0 reads stage only after the join
        unsafe { copy_cts(&r1.ex.compute, &mut r1.x, d1, 0, &r0.stage, d0, 0, h, rows_in)? };
        r1.go(route, None, m_in)?;
        // SAFETY: r1.x was written by the refresh just issued on the same stream
        unsafe { copy_cts(&r1.ex.compute, &mut r0.stage, d0, 0, &r1.x, d1, 0, h, rows_out)? };
        r1.order.record(&r1.ex.compute)?;
        d0.make_current()?;
        r0.go(route, None, m_in)?;
        r0.ex.compute.wait(&r1.order)?;
        Ok(())
    };
    // SAFETY: the capture reads the stream handle only; every launch in `body` targets device 0's compute stream or
    // device 1's, forked from it and joined back before the capture ends
    let cap = unsafe { &*sp }.capture(|_| {
        body().map_err(|e| {
            inner = Some(e);
            pc_cuda::Error::Unavailable("split capture")
        })
    });
    d0.make_current()?;
    match (cap, inner) {
        (Ok((g, ())), _) => Ok(g),
        (Err(_), Some(e)) => Err(format!("issuing under capture: {e}").into()),
        (Err(e), None) => Err(format!("ending the capture: {e}").into()),
    }
}

/// Replay a split graph `reps` times on device 0 (inputs 0..h into `r0.x`, h..k into `r0.stage` before each launch).
fn run_split(r0: &mut Rig<'_>, g: &GraphExec, h: usize, m_in: CtMeta, rows_in: usize, m_out: CtMeta, reps: usize) -> Res<T> {
    let d0 = r0.dev;
    let mut v = Vec::new();
    for rep in 0..reps + 2 {
        r0.load(0, h, m_in, rows_in)?;
        {
            let Rig { ex, stage, inp, .. } = &mut *r0;
            stage.buf.k = h;
            // SAFETY: inp is synchronized setup data; stage is a distinct buffer, synchronized below
            unsafe { copy_cts(&ex.compute, stage, d0, 0, inp, d0, h, h, rows_in)? };
            ex.compute.sync()?;
        }
        let t = Instant::now();
        r0.ev[0].record(&r0.ex.compute)?;
        g.launch(&r0.ex.compute)?;
        r0.ev[1].record(&r0.ex.compute)?;
        let issue = ms(t.elapsed());
        r0.ev[1].sync()?;
        let wall = ms(t.elapsed());
        if rep >= 2 {
            v.push(T { dev: r0.ev[0].elapsed_ms(&r0.ev[1])? as f64, issue, wall, ..T::default() });
        }
    }
    r0.x.meta = m_out;
    r0.stage.meta = m_out;
    Ok(med(&v))
}

/// Count the ciphertexts of `x` (slots `0..count`) equal to `refw[off..off + count]` word for word, metadata `m`.
fn matches(dev: &Device, s: &Stream<Compute>, x: &Ct<'_, W>, count: usize, refw: &[Vec<W>], off: usize, m: CtMeta, rows: usize) -> Res<usize> {
    if x.meta != m {
        println!("[mgpu]   metadata differs: {:?} vs {:?}", x.meta, m);
        return Ok(0);
    }
    let mut ok = 0;
    for c in 0..count {
        let w = words(dev, s, x, c, rows)?;
        if w == refw[off + c] {
            ok += 1;
        } else {
            let first = w.iter().zip(&refw[off + c]).position(|(a, b)| a != b).unwrap_or(0);
            println!("[mgpu]   ciphertext {} (slot {c}) differs from word {first} on", off + c);
        }
    }
    Ok(ok)
}

fn run(o: &Opts) -> Res<bool> {
    let t0 = Instant::now();
    let chain = Chain::generate(&ChainSpec::preset(&o.chain)?)?;
    let (n, nq) = (chain.n, chain.q.len());
    let slots = n / 2;
    let kmax = *o.ks.iter().max().ok_or("--k")?;
    let mut setup = BtsSetup::n32_shipped();
    setup.raise_drops = vec![0];
    setup.k_max = kmax;
    let plan = DensePlan::new(&chain, &setup)?;
    let mut steps = BTreeSet::new();
    for v in &plan.variants {
        for h in v.cts.iter().chain(&v.stc) {
            let g = h.geom;
            for i in 1..g.b {
                steps.insert(((i * g.stride) % (n / 2)) as i64);
            }
            if g.g > 1 {
                steps.insert(((g.b * g.stride) % (n / 2)) as i64);
            }
            steps.insert(g.offset.rem_euclid((n / 2) as i64));
        }
    }
    steps.remove(&0);
    let spec = KeySetSpec { relin: true, conj: true, rots: steps.iter().map(|&s| RotSpec { step: s, q_top: None }).collect(), encaps: true, lean: None };
    let shapes: Vec<KskShape> = spec.keys(&chain)?.into_iter().map(|(_, s)| s).collect();
    let cfg = MemCfg { k_max: kmax, pool: vec![PoolClass { k: 1, count: 1 }], events: 64, scratch_cts: 2 };
    let mut mp = MemPlan::build::<W, M>(&chain, cfg)?;
    let lay = mp.lay;
    mp.reserve("keys", spec.bytes::<W>(&chain, KskLayout::Dense)?);
    mp.reserve("keygen", KeyGen::<W>::bytes(&lay, &shapes));
    mp.reserve("secret", DevSecret::<W>::bytes(&lay));
    mp.reserve("encrypt", Encryptor::<W>::bytes(&lay, false));
    mp.reserve("decrypt", Decryptor::<W>::bytes(&lay, chain.level_unit, kmax));
    mp.reserve("ks.plans", KsPlans::bytes::<W>(&chain, kmax)?);
    mp.reserve("ks.regen_a", pc_gpu::ks::KsWork::<W>::regen_bytes(&lay, chain.dnum()));
    Ops::<W>::reserve(&mut mp, &OpsCfg::default());
    let (req, wshape) = plan.mem::<W>()?;
    req.reserve(&mut mp);
    mp.reserve("bts.work", BtsWork::<W>::bytes(&lay, wshape));
    for what in ["mgpu.x", "mgpu.inp", "mgpu.stage"] {
        mp.reserve(what, Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, kmax)));
    }
    println!("[mgpu] chain {} (N {n}, {nq} + {} primes), {} rotation keys, host plan {:.1} s", o.chain, chain.p.len(), steps.len(), t0.elapsed().as_secs_f64());
    println!("[mgpu] MemPlan per device {}", mp.report());

    let d0 = Device::open_ordinal(o.devices[0])?;
    let d1 = Device::open_ordinal(o.devices[1])?;
    let (p01, p10) = (d0.enable_peer(&d1)?, d1.enable_peer(&d0)?);
    println!(
        "[mgpu] devices {} and {} of CUDA_VISIBLE_DEVICES={:?}, {} and {} SMs, peer access {p01} / {p10}",
        d0.ordinal(),
        d1.ordinal(),
        std::env::var("CUDA_VISIBLE_DEVICES").ok(),
        d0.sm_count()?,
        d1.sm_count()?
    );
    d0.make_current()?;
    let a0 = mp.allocate(&d0)?;
    d1.make_current()?;
    let a1 = mp.allocate(&d1)?;
    let shared = Shared { chain, plan, spec, shapes, wshape, kmax };
    let mut r0 = rig(&d0, &a0, &mp, &shared)?;
    let mut r1 = rig(&d1, &a1, &mp, &shared)?;

    // K inputs, encrypted once on device 0
    let top = nq - 1 - o.level as usize;
    let sf = r0.ctx.scale.sf(top as i32)?;
    let mut m_in = CtMeta::new(o.level, 1).with_nf(sf);
    m_in.slots = slots as u32;
    let enc = Encoder::new(&shared.chain);
    let vals: Vec<Vec<Complex>> = (0..kmax).map(|i| rand_real(17 + i as u64, slots, 1.0)).collect();
    let msgs: Vec<Vec<Vec<u64>>> = vals.iter().map(|v| enc.encode_slots(v, slots, top, Payload::Complex, sf)).collect::<Result<_, _>>()?;
    d0.make_current()?;
    {
        let Rig { ctx, ex, sec, encr, inp, prng, .. } = &mut r0;
        inp.buf.k = kmax;
        encr.encrypt_sk(ctx, &ex.compute, sec, &msgs, m_in, inp, prng)?;
        ex.compute.sync()?;
    }
    let m_in = r0.inp.meta;
    let rows_in = nq - m_in.level as usize;
    let route = Route { cf: Some(o.cf), real: o.real, raise_drop: 0, ..Route::dense() };
    let mut all_ok = true;

    for &k in &o.ks {
        let h = k / 2;
        // single-GPU baseline: the batch of K is the reference
        let one_k = one_gpu(&mut r0, k, m_in, rows_in, &route, None, o.reps)?;
        let m_out = r0.x.meta;
        let rows_out = nq - m_out.level as usize;
        let refw: Vec<Vec<W>> = (0..k).map(|c| words(&d0, &r0.ex.compute, &r0.x, c, rows_out)).collect::<Res<_>>()?;
        let err = {
            let Rig { ctx, ex, sh, ops, dec, x, .. } = &mut r0;
            ops.realize(ctx, ex, sh, std::slice::from_mut(x))?;
            let got = dec.decrypt_slots(ctx, &ex.compute, &enc, x, Payload::Complex, Which::Dense)?;
            got.iter().zip(&vals).map(|(g, v)| g.iter().zip(v).map(|(a, b)| (a.re - b.re).abs()).fold(0f64, f64::max)).fold(0f64, f64::max)
        };
        let one_h = one_gpu(&mut r0, h, m_in, rows_in, &route, None, o.reps)?;
        // device 1's resident half: inputs h..k by a peer copy (untimed)
        d1.make_current()?;
        r1.inp.buf.k = h;
        r1.inp.meta = m_in;
        // SAFETY: r0.inp was synchronized after the encryption and is never written again
        unsafe { copy_cts(&r1.ex.compute, &mut r1.inp, &d1, 0, &r0.inp, &d0, h, h, rows_in)? };
        r1.ex.compute.sync()?;

        let two = two_gpu(&mut r0, &mut r1, h, m_in, rows_in, &route, None, o.reps)?;
        let ok_two = matches(&d0, &r0.ex.compute, &r0.x, h, &refw, 0, m_out, rows_out)? + matches(&d1, &r1.ex.compute, &r1.x, h, &refw, h, m_out, rows_out)?;
        let (off, b_in, b_out) = offload(&mut r0, &mut r1, h, m_in, rows_in, m_out, rows_out, &route, None, o.reps)?;
        let ok_off = matches(&d0, &r0.ex.compute, &r0.stage, h, &refw, h, m_out, rows_out)? + matches(&d0, &r0.ex.compute, &r0.x, h, &refw, 0, m_out, rows_out)?;
        all_ok &= ok_two == k && ok_off == k;

        let x = |t: &T| one_k.dev / t.dev;
        println!("[mgpu] K={k} one GPU, batch of {k}: device {:.2} ms ({:.2} ms/ct), issue {:.2} ms, wall {:.2} ms; max err {err:.2e}; out level {} deg {}", one_k.dev, one_k.dev / k as f64, one_k.issue, one_k.wall, m_out.level, m_out.deg);
        println!("[mgpu] K={k} one GPU, batch of {h}: device {:.2} ms ({:.2} ms/ct), issue {:.2} ms, wall {:.2} ms", one_h.dev, one_h.dev / h as f64, one_h.issue, one_h.wall);
        println!(
            "[mgpu] K={k} two GPUs, {h} + {h}: span {:.2} ms (devices {:.2} / {:.2}), issue {:.2} ms, wall {:.2} ms: x{:.2} over the batch of {k} (ideal x{:.2})",
            two.dev,
            two.a,
            two.b,
            two.issue,
            two.wall,
            x(&two),
            one_k.dev / one_h.dev
        );
        println!(
            "[mgpu] K={k} offload {h} of {k}: span {:.2} ms (device 0 {:.2}; device 1 copy in {:.3} ms for {:.1} MB, refresh {:.2}, copy back {:.3} ms for {:.1} MB), issue {:.2} ms, wall {:.2} ms: x{:.2}",
            off.dev,
            off.a,
            off.b,
            b_in as f64 / 1e6,
            off.c,
            off.d,
            b_out as f64 / 1e6,
            off.issue,
            off.wall,
            x(&off)
        );
        println!("[mgpu] K={k} bit-exact against the batch of {k}: two GPUs {ok_two}/{k}, offload {ok_off}/{k}");

        if o.graph {
            r0.load(0, k, m_in, rows_in)?;
            let g0k = r0.capture(&route)?;
            r0.load(0, h, m_in, rows_in)?;
            let g0h = r0.capture(&route)?;
            r1.load(0, h, m_in, rows_in)?;
            let g1h = r1.capture(&route)?;
            let one_kg = one_gpu(&mut r0, k, m_in, rows_in, &route, Some(&g0k), o.reps)?;
            let ok_one = matches(&d0, &r0.ex.compute, &r0.x, k, &refw, 0, m_out, rows_out)?;
            let one_hg = one_gpu(&mut r0, h, m_in, rows_in, &route, Some(&g0h), o.reps)?;
            let two_g = two_gpu(&mut r0, &mut r1, h, m_in, rows_in, &route, Some((&g0h, &g1h)), o.reps)?;
            let ok_two_g = matches(&d0, &r0.ex.compute, &r0.x, h, &refw, 0, m_out, rows_out)? + matches(&d1, &r1.ex.compute, &r1.x, h, &refw, h, m_out, rows_out)?;
            let (off_g, _, _) = offload(&mut r0, &mut r1, h, m_in, rows_in, m_out, rows_out, &route, Some((&g0h, &g1h)), o.reps)?;
            let ok_off_g = matches(&d0, &r0.ex.compute, &r0.stage, h, &refw, h, m_out, rows_out)? + matches(&d0, &r0.ex.compute, &r0.x, h, &refw, 0, m_out, rows_out)?;
            all_ok &= ok_one == k && ok_two_g == k && ok_off_g == k;
            let xg = |t: &T| one_kg.dev / t.dev;
            println!(
                "[mgpu] K={k} graphs: one GPU batch of {k} {:.2} ms (issue {:.2}), batch of {h} {:.2} ms; two GPUs span {:.2} ms (devices {:.2} / {:.2}, issue {:.2}): x{:.2}; offload span {:.2} ms (copy in {:.3}, copy back {:.3}): x{:.2}",
                one_kg.dev,
                one_kg.issue,
                one_hg.dev,
                two_g.dev,
                two_g.a,
                two_g.b,
                two_g.issue,
                xg(&two_g),
                off_g.dev,
                off_g.b,
                off_g.d,
                xg(&off_g)
            );
            println!("[mgpu] K={k} graphs bit-exact against the eager batch of {k}: one GPU {ok_one}/{k}, two GPUs {ok_two_g}/{k}, offload {ok_off_g}/{k}");
            // one graph on device 0 spanning both devices
            match capture_split(&mut r0, &mut r1, h, m_in, rows_in, rows_out, &route) {
                Ok(gs) => {
                    let sp = run_split(&mut r0, &gs, h, m_in, rows_in, m_out, o.reps)?;
                    let ok_split = matches(&d0, &r0.ex.compute, &r0.x, h, &refw, 0, m_out, rows_out)? + matches(&d0, &r0.ex.compute, &r0.stage, h, &refw, h, m_out, rows_out)?;
                    all_ok &= ok_split == k;
                    println!("[mgpu] K={k} one cross-device graph: span {:.2} ms (issue {:.3} ms): x{:.2} over the graph batch of {k}; bit-exact {ok_split}/{k}", sp.dev, sp.issue, one_kg.dev / sp.dev);
                }
                Err(e) => println!("[mgpu] K={k} one cross-device graph: NOT capturable ({e}); fall back to one graph per device joined by events"),
            }
        }
    }
    println!("[mgpu] {}", if all_ok { "PASS: every output equals the single-GPU batch word for word" } else { "FAIL" });
    Ok(all_ok)
}
