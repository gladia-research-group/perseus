//! PCMM: y = x W at the StC-first split (ePrint 2024/1284 Alg. 2; Sylph, arXiv 2601.18511), a port of FIDESlib
//! `Pcmm.cu` (`pcmmUploadKeys`, `pcmmMultiPlanes`) with every buffer carved from the MemPlan (`Pcmm::bytes`), no
//! device syncs, and the int8 GEMMs as cuBLASLt plans made at setup.
//!
//! Per input: coefficient form, exact rescale by q3 then q2, key switch to s'(X^16), MLWE rows; Y = W X mod q_i as
//! balanced base-256 digit planes (4 x nW int8 GEMMs, exact under `exact_k_limit`); per output: pack back into ring
//! 2^12 ciphertexts, compose into ring 2^16, switch back to s, lift into the split's limbs (0..3; 0..2 on a lean chain,
//! whose split rescales by q2 alone), eval form. Inputs are consumed
//! before any output is written (`product_inplace`). Scale: `nf_out = nf_split / w_scale`; the driver (`linear`)
//! handles the 2^shift. Details: docs/ARCHITECTURE.md "pc-gpu PCMM".
use super::gemm::{GemmBackend, GemmPlans};
use crate::bts::finish::Pow2;
use crate::bts::staged::{record_stc_first_inputs, BtsCall, BtsRecorder, CallOpts, Phase, StagedRef};
use crate::bts::stc_first::StcFirstRoute;
use crate::bts::BtsCx;
use crate::context::{Context, DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Error, Result};
use crate::ks::View;
use crate::ntt::{self, Dir, RowList};
use pc_core::modmul::invmod;
use pc_core::pcmm_keys::{NttTab, PcmmKeys, Shape, B, L, N, N1, S_SWITCH};
use pc_cuda::blas::lt;
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, DBuf, Device, Event, Stream};
use pc_kernels::{pcmm_op as op, PcmmArgs, PolyRef};
use pc_shadow::{CtMeta, Route};
use std::collections::BTreeMap;
use std::ffi::c_void;

/// Weight digit planes per linear the kernels accept (int8 planes of a <= 32-bit integer weight).
pub const MAX_PLANES: usize = 4;
/// Largest finish batch (perseus PF_PCMM_BATCH <= 4).
pub const MAX_BATCH: usize = 4;

/// One linear's shape: `t` token lanes, `n_in -> n_out` channel blocks of `nch(t)`, `nw` weight digit planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PcmmShape {
    pub t: usize,
    pub n_in: usize,
    pub n_out: usize,
    pub nw: usize,
}

impl PcmmShape {
    pub fn lay(&self) -> Result<Shape> {
        Ok(Shape::new(self.t)?)
    }
    pub fn nch(&self) -> Result<usize> {
        Ok(self.lay()?.nch())
    }
    pub fn rows_in(&self) -> Result<usize> {
        Ok(self.nch()? * self.n_in)
    }
    pub fn rows_out(&self) -> Result<usize> {
        Ok(self.nch()? * self.n_out)
    }
    /// int8 elements of the weight planes `[nw][rows_out][rows_in]`.
    pub fn planes_len(&self) -> Result<usize> {
        Ok(self.nw * self.rows_out()? * self.rows_in()?)
    }
    /// Setup checks: 1..=4 planes, at least one block each way, nch a multiple of 256 (GEMM operand alignment),
    /// and the i32 exactness bound of the digit GEMM (pairs k 2^14 < 2^31, pairs = min(4, nw)).
    pub fn check(&self) -> Result<()> {
        let nch = self.nch()?;
        let bad = |m: String| Err(Error::Mem { knob: "PcmmCfg::shapes", value: self.t, msg: m });
        if self.nw == 0 || self.nw > MAX_PLANES || self.n_in == 0 || self.n_out == 0 {
            return bad(format!("{self:?}: need 1..={MAX_PLANES} planes and n_in, n_out >= 1"));
        }
        if nch % 256 != 0 {
            return bad(format!("{self:?}: nch {nch} is not a multiple of 256 (t <= 128)"));
        }
        let lim = lt::exact_k_limit(self.nw.min(4) as u64) as usize;
        if self.rows_in()? > lim {
            return bad(format!("{self:?}: rows_in {} above the exact int32 bound {lim} for {} planes", self.rows_in()?, self.nw));
        }
        Ok(())
    }
}

/// The PCMM engine's knobs.
#[derive(Clone, Debug, PartialEq)]
pub struct PcmmCfg {
    /// Every linear shape the engine will run (buffers and GEMM plans are sized from these at setup).
    pub shapes: Vec<PcmmShape>,
    /// Bytes of the int32 exponent accumulators per GEMM row chunk (perseus modGemm budget, 1 GiB).
    pub g_budget: usize,
    /// Time the heuristic candidates of every GEMM shape at setup and keep the fastest.
    pub autotune: bool,
    /// Heuristic candidates asked per shape.
    pub max_algos: usize,
    /// cuBLASLt workspace bytes (split-K candidates use it).
    pub gemm_workspace: usize,
    pub backend: GemmBackend,
}

impl Default for PcmmCfg {
    fn default() -> Self {
        PcmmCfg { shapes: Vec::new(), g_budget: 1 << 30, autotune: true, max_algos: 8, gemm_workspace: 32 << 20, backend: GemmBackend::CublasLt }
    }
}

/// Exponent accumulator rows per GEMM chunk: perseus `rc` (budget / (E 4 2cols), at least 512), rounded down to a
/// multiple of 256 so every chunk's output pointer stays 256-byte aligned.
pub fn chunk_rows(rows_out: usize, cols: usize, nw: usize, budget: usize) -> usize {
    let e = 3 + nw;
    let rc = rows_out.min((budget / (e * 4 * 2 * cols)).max(512));
    (rc / 256 * 256).max(256).min(rows_out)
}

/// Element counts of the engine's work buffers for a set of shapes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Sizes {
    scratch: usize,
    a: usize,
    bm: usize,
    ap: usize,
    bp: usize,
    c: usize,
    xd: usize,
    g: usize,
}

impl Sizes {
    fn of(cfg: &PcmmCfg) -> Result<Sizes> {
        let mut z = Sizes { scratch: 2 * L * B * N, ..Sizes::default() };
        for s in &cfg.shapes {
            s.check()?;
            let lay = s.lay()?;
            let (np, r, nch) = (lay.np(), lay.r(), lay.nch());
            let (ri, ro) = (s.rows_in()?, s.rows_out()?);
            z.scratch = z.scratch.max(L * B * S_SWITCH * r * N1 + 2 * B * S_SWITCH * N1);
            z.a = z.a.max(L * ri * N1);
            z.bm = z.bm.max(L * ri * np);
            z.ap = z.ap.max(L * ro * N1);
            z.bp = z.bp.max(L * ro * np);
            z.c = z.c.max(L * nch * N1);
            z.xd = z.xd.max(4 * ri * 2 * N1);
            for cols in [N1, np] {
                let rc = chunk_rows(ro, cols, s.nw, cfg.g_budget);
                z.g = z.g.max((3 + s.nw) * rc * 2 * cols);
            }
        }
        Ok(z)
    }
}

// ── keys on the device ───────────────────────────────────────────────────────────────────────────────────────────

/// One shape's PCMM keys in the device's NTT form, with the {q0, q1, P} twiddle tables.
pub struct DevPcmmKeys<'a> {
    pub t: usize,
    pub shape: Shape,
    pub primes: [u32; 3],
    pub mu: [u64; 3],
    pub split: [u64; 4],
    tw16: ConstBuf<'a, u32>,
    itw16: ConstBuf<'a, u32>,
    tw12: ConstBuf<'a, u32>,
    itw12: ConstBuf<'a, u32>,
    ninv16: [u32; 3],
    ninv12: [u32; 3],
    fwd: ConstBuf<'a, u32>,
    back: ConstBuf<'a, u32>,
    pack: ConstBuf<'a, u32>,
}

fn mu64(p: u64) -> u64 {
    ((1u128 << 64) / p as u128) as u64
}

impl<'a> DevPcmmKeys<'a> {
    pub fn bytes(t: usize) -> Result<usize> {
        let r = Shape::new(t)?.r();
        let u = |n: usize| Arena::bytes_for::<u32>(n);
        Ok(2 * u(B * N) + 2 * u(B * N1) + 2 * u(L * B * 2 * N) + u(r * L * B * 2 * N1))
    }

    /// Upload `k` (NTT forms computed on the host; setup, synchronous on `s`).
    pub fn upload(arena: &'a Arena, s: &Stream<Compute>, k: &PcmmKeys) -> Result<Self> {
        let t16 = k.tables(N)?;
        let t12 = k.tables(N1)?;
        let cat = |tabs: &[NttTab; B], inv: bool| -> Vec<u32> { tabs.iter().flat_map(|t| if inv { t.ipsi.clone() } else { t.psi.clone() }).collect() };
        let primes = [k.primes[0] as u32, k.primes[1] as u32, k.primes[2] as u32];
        Ok(DevPcmmKeys {
            t: k.shape.t,
            shape: k.shape,
            primes,
            mu: [mu64(k.primes[0]), mu64(k.primes[1]), mu64(k.primes[2])],
            split: k.split,
            tw16: ConstBuf::new(arena, "pcmm.keys", s, &cat(&t16, false))?,
            itw16: ConstBuf::new(arena, "pcmm.keys", s, &cat(&t16, true))?,
            tw12: ConstBuf::new(arena, "pcmm.keys", s, &cat(&t12, false))?,
            itw12: ConstBuf::new(arena, "pcmm.keys", s, &cat(&t12, true))?,
            ninv16: [t16[0].ninv, t16[1].ninv, t16[2].ninv],
            ninv12: [t12[0].ninv, t12[1].ninv, t12[2].ninv],
            fwd: ConstBuf::new(arena, "pcmm.keys", s, &PcmmKeys::eval_form(&k.fwd, &t16))?,
            back: ConstBuf::new(arena, "pcmm.keys", s, &PcmmKeys::eval_form(&k.back, &t16))?,
            pack: ConstBuf::new(arena, "pcmm.keys", s, &PcmmKeys::eval_form(&k.pack, &t12))?,
        })
    }

    fn tw(&self, n: usize, inverse: bool) -> (*const c_void, [u32; 3]) {
        match (n == N, inverse) {
            (true, false) => (self.tw16.vptr(), [0; 3]),
            (true, true) => (self.itw16.vptr(), self.ninv16),
            (false, false) => (self.tw12.vptr(), [0; 3]),
            (false, true) => (self.itw12.vptr(), self.ninv12),
        }
    }
}

// ── weights ──────────────────────────────────────────────────────────────────────────────────────────────────────

/// A linear as the driver needs it (the Python `PcmmLinear` minus the planes): weights' integer scale (a power of
/// two), the band shift, the site name (calibration key / descriptor).
#[derive(Clone, Debug, PartialEq)]
pub struct PcmmLinear {
    pub name: String,
    pub shape: PcmmShape,
    pub w_scale: f64,
    pub shift: u32,
}

/// Device weight digit planes `[nw][rows_out][rows_in]` int8 (stored as bytes), 256-byte aligned (an arena carve).
pub struct WeightBuf<'a> {
    pub buf: DBuf<'a, Compute, u8>,
}

impl<'a> WeightBuf<'a> {
    pub fn bytes(shape: &PcmmShape) -> Result<usize> {
        Ok(Arena::bytes_for::<u8>(shape.planes_len()?))
    }
    pub fn carve(arena: &'a Arena, shape: &PcmmShape) -> Result<Self> {
        Ok(WeightBuf { buf: arena.alloc("pcmm.weights", shape.planes_len()?)? })
    }
    /// Synchronous upload (setup / tests); the model path stages planes from pinned memory on its own lane.
    pub fn upload(&mut self, s: &Stream<Compute>, planes: &[i8]) -> Result<()> {
        // SAFETY: i8 and u8 have the same size and every bit pattern is valid for both
        let b = unsafe { std::slice::from_raw_parts(planes.as_ptr() as *const u8, planes.len()) };
        Ok(self.buf.upload(s, b)?)
    }
}

// ── the engine ───────────────────────────────────────────────────────────────────────────────────────────────────

struct Work<'a, W: DevWord> {
    /// [part][4][N] W: an input's limbs (INTT'd), later an output's (lifted, NTT'd)
    stage: DBuf<'a, Compute, W>,
    /// [part][2][N] the rescaled input
    xr: DBuf<'a, Compute, u32>,
    /// [ab][2][N] fwd / back key-switch results
    res: DBuf<'a, Compute, u32>,
    scratch: DBuf<'a, Compute, u32>,
    a: DBuf<'a, Compute, u32>,
    bm: DBuf<'a, Compute, u32>,
    ap: DBuf<'a, Compute, u32>,
    bp: DBuf<'a, Compute, u32>,
    c: DBuf<'a, Compute, u32>,
    pk: DBuf<'a, Compute, u32>,
    ra: DBuf<'a, Compute, u32>,
    rb: DBuf<'a, Compute, u32>,
    xd: DBuf<'a, Compute, u8>,
    g: DBuf<'a, Compute, i32>,
    gws: DBuf<'a, Compute, u8>,
}

/// Device timings of the product's phases (ms) from the 4 events a call recorded (`product(.., Some(ev))`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PcmmTimes {
    pub inputs: f64,
    pub gemm: f64,
    pub outputs: f64,
}

impl PcmmTimes {
    /// Host-waits for the last event.
    pub fn read(ev: [&Event; 4]) -> Result<PcmmTimes> {
        ev[3].sync()?;
        Ok(PcmmTimes { inputs: ev[0].elapsed_ms(ev[1])? as f64, gemm: ev[1].elapsed_ms(ev[2])? as f64, outputs: ev[2].elapsed_ms(ev[3])? as f64 })
    }
}

/// Timing events the linear driver records (benches): after the inputs' StC, around the product's phases, after
/// the finishes. Recording only (no host waits).
#[derive(Clone, Copy, Default)]
pub struct LinearHooks<'e> {
    pub after_stc: Option<&'e Event>,
    pub product: Option<[&'e Event; 4]>,
    pub after_finish: Option<&'e Event>,
}

/// The PCMM engine of a session: keys per token-lane count, GEMM plans per shape, one set of work buffers.
pub struct Pcmm<'a, 'lt, W: DevWord> {
    pub cfg: PcmmCfg,
    pub keys: Vec<DevPcmmKeys<'a>>,
    pub plans: GemmPlans<'lt>,
    work: Work<'a, W>,
    rows8: RowList<'a>,
    pow2: Pow2<'a, W>,
    /// The split's limbs: 4 (q0..q3), or 3 on a lean chain (q0..q2: its StC-first split, `stc_first::split_top`).
    nsplit: usize,
    /// chain q0..q3, Barrett constants, rescale inverses (q3^-1 mod q0..q2, q2^-1 mod q0, q1), lift (the product of the
    /// split's primes above q1) mod q0, q1
    chain4: [u32; 4],
    chain_mu: [u64; 4],
    inv_c: [u32; 8],
    lift: [u32; 2],
}

impl<'a, 'lt, W: DevWord> Pcmm<'a, 'lt, W> {
    /// Arena bytes of the engine for `cfg` (keys of every distinct `t`, work buffers, staging row list, 2^e table);
    /// `MemPlan::reserve("pcmm", ..)`.
    pub fn bytes(cfg: &PcmmCfg, prime_ids: usize) -> Result<usize> {
        let z = Sizes::of(cfg)?;
        let mut ts: Vec<usize> = cfg.shapes.iter().map(|s| s.t).collect();
        ts.sort();
        ts.dedup();
        let mut b = 0;
        for t in ts {
            b += DevPcmmKeys::bytes(t)?;
        }
        let u = |n: usize| Arena::bytes_for::<u32>(n);
        b += Arena::bytes_for::<W>(8 * N) + 3 * u(4 * N) + u(z.scratch) + u(z.a) + u(z.bm) + u(z.ap) + u(z.bp) + u(z.c) + u(2 * L * S_SWITCH * N1) + u(L * N);
        b += Arena::bytes_for::<u8>(z.xd) + Arena::bytes_for::<i32>(z.g) + Arena::bytes_for::<u8>(cfg.gemm_workspace);
        b += Arena::bytes_for::<u32>(8) + Arena::bytes_for::<i32>(8) + Pow2::<W>::bytes(prime_ids);
        Ok(b)
    }

    /// Setup: upload the keys (one `PcmmKeys` per distinct `t` of the shapes), carve the work buffers, plan every
    /// GEMM shape (and time the candidates when `cfg.autotune`). Synchronous on `s`.
    pub fn new<M: DevStrategy<W>>(arena: &'a Arena, dev: &Device, s: &Stream<Compute>, ctx: &Context<'_, W, M>, lt_: &'lt lt::Lt, cfg: PcmmCfg, keys: &[PcmmKeys]) -> Result<Self> {
        let z = Sizes::of(&cfg)?;
        let split: Vec<u64> = (0..4).map(|i| ctx.prime(i)).collect();
        let mut dk = Vec::new();
        for s_ in &cfg.shapes {
            if dk.iter().any(|k: &DevPcmmKeys| k.t == s_.t) {
                continue;
            }
            let k = keys.iter().find(|k| k.shape.t == s_.t).ok_or_else(|| shape("pcmm keys", format!("no PcmmKeys for t = {} (PcmmCfg::shapes)", s_.t)))?;
            if k.split.to_vec() != split {
                return Err(shape("pcmm keys", format!("keys made for split moduli {:?}, the chain's are {split:?}", k.split)));
            }
            dk.push(DevPcmmKeys::upload(arena, s, k)?);
        }
        let u = |what, n| arena.alloc::<Compute, u32>(what, n);
        let work = Work {
            stage: arena.alloc("pcmm.work", 8 * N)?,
            xr: u("pcmm.work", 4 * N)?,
            res: u("pcmm.work", 4 * N)?,
            scratch: u("pcmm.work", z.scratch)?,
            a: u("pcmm.work", z.a.max(1))?,
            bm: u("pcmm.work", z.bm.max(1))?,
            ap: u("pcmm.work", z.ap.max(1))?,
            bp: u("pcmm.work", z.bp.max(1))?,
            c: u("pcmm.work", z.c.max(1))?,
            pk: u("pcmm.work", 2 * L * S_SWITCH * N1)?,
            ra: u("pcmm.work", 4 * N)?,
            rb: u("pcmm.work", L * N)?,
            xd: arena.alloc("pcmm.work", z.xd.max(1))?,
            g: arena.alloc("pcmm.work", z.g.max(1))?,
            gws: arena.alloc("pcmm.gemm_ws", cfg.gemm_workspace)?,
        };
        // staging rows: (part, limb) -> limb offset part*4 + i, prime id i (the split's limbs)
        let nsplit = crate::bts::stc_first::split_top(&ctx.chain) + 1;
        if !(3..=4).contains(&nsplit) {
            return Err(shape("pcmm", format!("a split of {nsplit} limbs (3 or 4)")));
        }
        let rows: Vec<(u32, i32)> = (0..2).flat_map(|p| (0..nsplit).map(move |i| ((p * 4 + i) as u32, i as i32))).collect();
        let rows8 = RowList::new(arena, s, &rows)?;
        let pow2 = Pow2::new(arena, s, ctx)?;
        let q: Vec<u64> = split.clone();
        let chain4 = [q[0] as u32, q[1] as u32, q[2] as u32, q[3] as u32];
        let chain_mu = [mu64(q[0]), mu64(q[1]), mu64(q[2]), mu64(q[3])];
        let mut inv_c = [0u32; 8];
        for i in 0..3 {
            inv_c[i] = invmod(q[3] % q[i], q[i]) as u32;
        }
        for i in 0..2 {
            inv_c[3 + i] = invmod(q[2] % q[i], q[i]) as u32;
        }
        let q23 = q[2..nsplit].iter().fold(1u128, |a, &x| a * x as u128);
        let lift = [(q23 % q[0] as u128) as u32, (q23 % q[1] as u128) as u32];
        let mut plans = GemmPlans::default();
        for s_ in &cfg.shapes {
            let lay = s_.lay()?;
            let (ri, ro) = (s_.rows_in()?, s_.rows_out()?);
            for cols in [N1, lay.np()] {
                let rc = chunk_rows(ro, cols, s_.nw, cfg.g_budget);
                let mut ns = vec![rc];
                if ro % rc != 0 {
                    ns.push(ro % rc);
                }
                for n in ns {
                    plans.ensure(lt_, 2 * cols as u64, n as u64, ri as u64, cfg.max_algos, cfg.gemm_workspace)?;
                }
            }
        }
        let mut me = Pcmm { cfg, keys: dk, plans, work, rows8, pow2, nsplit, chain4, chain_mu, inv_c, lift };
        if me.cfg.autotune {
            me.autotune(dev, s)?;
        }
        s.sync()?;
        Ok(me)
    }

    /// Time every GEMM plan's candidates on the work buffers (contents irrelevant) and keep the fastest.
    pub fn autotune(&mut self, dev: &Device, s: &Stream<Compute>) -> Result<()> {
        let ev = (Event::timing(dev)?, Event::timing(dev)?);
        let w = &mut self.work;
        w.xd.zero(s)?;
        let a = w.xd.dptr() as *const i8;
        let c = w.g.dptr_mut();
        for p in self.plans.plans.iter_mut() {
            let (la, lb, lc) = p.lens();
            if la > w.xd.len() || lb > w.xd.len() || lc > 4 * w.g.len() {
                continue; // the B operand is the weights; xd stands in for it when large enough
            }
            // SAFETY: operands inside live work buffers of the sizes checked above, all on the compute lane
            unsafe { p.tune(s, (&ev.0, &ev.1), a, a, c, &mut w.gws, 3)? };
        }
        Ok(())
    }

    pub fn keys_for(&self, t: usize) -> Result<&DevPcmmKeys<'a>> {
        self.keys.iter().find(|k| k.t == t).ok_or_else(|| shape("pcmm", format!("no keys for t = {t} (PcmmCfg::shapes)")))
    }

    fn launch<M: DevStrategy<W>>(ctx: &Context<'_, W, M>, s: &Stream<Compute>, a: PcmmArgs) -> Result<()> {
        // SAFETY: every pointer in `a` addresses a live buffer of the engine / the caller's ciphertexts sized for the
        // op (checked by the callers against the configured shapes); all on the compute lane
        launch_result("pcmm", unsafe { (ctx.table.pcmm)(&a, s.raw()) })?;
        Ok(())
    }

    fn kargs(k: &DevPcmmKeys<'_>, opc: i32, n: usize) -> PcmmArgs {
        PcmmArgs { op: opc, n: n as i32, p: [k.primes[0], k.primes[1], k.primes[2], 0], mu: [k.mu[0], k.mu[1], k.mu[2], 0], ..PcmmArgs::default() }
    }

    /// res [ab][2][outs][n] = sum over the nl pieces of each output of the key switch of x [2][outs nl][n].
    #[allow(clippy::too_many_arguments)]
    fn key_switch<M: DevStrategy<W>>(ctx: &Context<'_, W, M>, s: &Stream<Compute>, k: &DevPcmmKeys<'_>, x: *const u32, res: *mut u32, n: usize, outs: usize, nl: usize, key: *const u32, scratch: *mut u32) -> Result<()> {
        let polys = outs * nl;
        let d = scratch;
        let acc = scratch.wrapping_add(L * B * polys * n);
        let (q0, q1, pp) = (k.primes[0] as u64, k.primes[1] as u64, k.primes[2] as u64);
        let mut a = Self::kargs(k, op::KS_DIGITS, n);
        a.count = polys as i32;
        a.y = x as *const c_void;
        a.x = d as *mut c_void;
        a.c[0] = invmod(q1 % q0, q0) as u32;
        a.c[1] = invmod(q0 % q1, q1) as u32;
        Self::launch(ctx, s, a)?;
        let (tw, _) = k.tw(n, false);
        let mut a = Self::kargs(k, op::NTT, n);
        a.count = (L * B * polys) as i32;
        a.i0 = polys as i32;
        a.x = d as *mut c_void;
        a.tw = tw;
        Self::launch(ctx, s, a)?;
        let mut a = Self::kargs(k, op::KS_MAC, n);
        a.i0 = outs as i32;
        a.i1 = nl as i32;
        a.y = d as *const c_void;
        a.z = key as *const c_void;
        a.x = acc as *mut c_void;
        Self::launch(ctx, s, a)?;
        let (itw, ninv) = k.tw(n, true);
        let mut a = Self::kargs(k, op::INTT, n);
        a.count = (2 * B * outs) as i32;
        a.i0 = outs as i32;
        a.x = acc as *mut c_void;
        a.tw = itw;
        a.c[..3].copy_from_slice(&ninv);
        Self::launch(ctx, s, a)?;
        let mut a = Self::kargs(k, op::MOD_DOWN, n);
        a.i0 = outs as i32;
        a.y = acc as *const c_void;
        a.x = res as *mut c_void;
        a.c[0] = invmod(pp % q0, q0) as u32;
        a.c[1] = invmod(pp % q1, q1) as u32;
        Self::launch(ctx, s, a)
    }

    /// Y [2][rows_out][cols] = W X mod q_i, X [2][rows_in][cols], W the digit planes.
    #[allow(clippy::too_many_arguments)]
    fn mod_gemm<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, k: &DevPcmmKeys<'_>, wd: *const i8, nw: usize, x: *const u32, y: *mut u32, rows_in: usize, rows_out: usize, cols: usize) -> Result<()> {
        let e_n = 3 + nw;
        let c2 = 2 * cols;
        let cin = rows_in * c2;
        let cw = rows_out * rows_in;
        let w = &mut self.work;
        if 4 * cin > w.xd.len() {
            return Err(shape("pcmm gemm", format!("digit buffer {} < {} (PcmmCfg::shapes)", w.xd.len(), 4 * cin)));
        }
        let mut a = Self::kargs(k, op::DIGITS, 0);
        a.i0 = rows_in as i32;
        a.i1 = cols as i32;
        a.y = x as *const c_void;
        a.x = w.xd.vptr_mut();
        Self::launch(ctx, s, a)?;
        let rc = chunk_rows(rows_out, cols, nw, self.cfg.g_budget);
        if e_n * rc * c2 > w.g.len() {
            return Err(shape("pcmm gemm", format!("accumulator buffer {} < {} (PcmmCfg::shapes)", w.g.len(), e_n * rc * c2)));
        }
        let xd = w.xd.dptr() as *const i8;
        let g = w.g.dptr_mut();
        let mut r0 = 0;
        while r0 < rows_out {
            let rcur = rc.min(rows_out - r0);
            let plan = self.plans.get(c2 as u64, rcur as u64, rows_in as u64)?;
            let mut seen = [false; 8];
            for t in 0..4 {
                for u in 0..nw {
                    let e = t + u;
                    // SAFETY: A = digit plane t (cin bytes), B = weight plane u rows r0.. (rcur rows of rows_in),
                    // C = exponent e's accumulator (rcur c2 int32), all inside buffers checked above / by the caller
                    unsafe {
                        plan.run(s, xd.wrapping_add(t * cin), wd.wrapping_add(u * cw + r0 * rows_in), g.wrapping_add(e * rcur * c2), seen[e], &mut w.gws)?;
                    }
                    seen[e] = true;
                }
            }
            let mut a = Self::kargs(k, op::RECOMBINE, 0);
            a.i0 = rcur as i32;
            a.i1 = r0 as i32;
            a.i2 = rows_out as i32;
            a.i3 = cols as i32;
            a.i4 = e_n as i32;
            a.y = g as *const c_void;
            a.x = y as *mut c_void;
            Self::launch(ctx, s, a)?;
            r0 += rcur;
        }
        Ok(())
    }

    /// The digit-plane GEMM alone (the `Gemm` seam's contract check, `pc-check pcmm`): y \[2\]\[rows_out\]\[cols\] =
    /// W x mod q_i for x \[2\]\[rows_in\]\[cols\] residues, `cols` = N1 (the A rows) or 2t (the B rows) of `sh`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, sh: &PcmmShape, planes: &DBuf<'_, Compute, u8>, x: &DBuf<'_, Compute, u32>, y: &mut DBuf<'_, Compute, u32>, cols: usize) -> Result<()> {
        let (ri, ro) = (sh.rows_in()?, sh.rows_out()?);
        if !self.cfg.shapes.contains(sh) || (cols != N1 && cols != sh.lay()?.np()) {
            return Err(shape("pcmm gemm", format!("{sh:?} cols {cols} not configured")));
        }
        if x.len() < 2 * ri * cols || y.len() < 2 * ro * cols || planes.len() < sh.planes_len()? {
            return Err(shape("pcmm gemm", format!("buffers x {} y {} planes {} for {sh:?} cols {cols}", x.len(), y.len(), planes.len())));
        }
        let ki = self.keys.iter().position(|k| k.t == sh.t).ok_or_else(|| shape("pcmm", format!("no keys for t = {}", sh.t)))?;
        let k = self.keys.swap_remove(ki);
        let r = self.mod_gemm(ctx, s, &k, planes.dptr() as *const i8, sh.nw, x.dptr(), y.dptr_mut(), ri, ro, cols);
        self.keys.push(k);
        r
    }

    /// The product on raw ciphertext members: `ins[i]` = (ciphertext view, member) of input block i, `outs[m]` of
    /// output block m. All inputs are read before any output is written.
    #[allow(clippy::too_many_arguments)]
    fn run_members<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, sh: &PcmmShape, planes: &DBuf<'_, Compute, u8>, ins: &[(PolyRef, usize)], outs: &[(PolyRef, usize)], ev: Option<[&Event; 4]>) -> Result<()> {
        if !self.cfg.shapes.contains(sh) {
            return Err(shape("pcmm", format!("{sh:?} not in PcmmCfg::shapes")));
        }
        if ins.len() != sh.n_in || outs.len() != sh.n_out {
            return Err(shape("pcmm", format!("{} inputs / {} outputs for {sh:?}", ins.len(), outs.len())));
        }
        if planes.len() < sh.planes_len()? || planes.dptr() as usize % 256 != 0 {
            return Err(shape("pcmm weights", format!("{} bytes at {:#x}, need {} 256-aligned", planes.len(), planes.dptr() as usize, sh.planes_len()?)));
        }
        let ki = self.keys.iter().position(|k| k.t == sh.t).ok_or_else(|| shape("pcmm", format!("no keys for t = {}", sh.t)))?;
        // the keys move out of self for the call (the GEMM borrows self mutably); restored below
        let k = self.keys.swap_remove(ki);
        let r = self.run_with(ctx, s, &k, sh, planes, ins, outs, ev);
        self.keys.push(k);
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn run_with<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, k: &DevPcmmKeys<'_>, sh: &PcmmShape, planes: &DBuf<'_, Compute, u8>, ins: &[(PolyRef, usize)], outs: &[(PolyRef, usize)], ev: Option<[&Event; 4]>) -> Result<()> {
        let lay = k.shape;
        let (np, r, nch) = (lay.np(), lay.r(), lay.nch());
        let (ri, ro) = (sh.rows_in()?, sh.rows_out()?);
        if let Some(e) = ev {
            e[0].record(s)?;
        }
        let w = &mut self.work;
        let (stage, xr, res, scratch) = (w.stage.dptr_mut() as *mut c_void, w.xr.dptr_mut(), w.res.dptr_mut(), w.scratch.dptr_mut());
        let (pa, pbm, pap, pbp, pc, ppk, pra, prb) = (w.a.dptr_mut(), w.bm.dptr_mut(), w.ap.dptr_mut(), w.bp.dptr_mut(), w.c.dptr_mut(), w.pk.dptr_mut(), w.ra.dptr_mut(), w.rb.dptr_mut());
        let (c4, cmu, inv_c, lift, ns) = (self.chain4, self.chain_mu, self.inv_c, self.lift, self.nsplit as i32);
        let chain = |opc: i32| PcmmArgs { op: opc, n: N as i32, p: c4, mu: cmu, i4: ns, ..PcmmArgs::default() };
        for (c, &(ct, m)) in ins.iter().enumerate() {
            let mut a = chain(op::GATHER);
            a.ct = ct;
            a.i0 = m as i32;
            a.x = stage;
            Self::launch(ctx, s, a)?;
            ntt::transform_rows(ctx, s, Dir::Inverse, &mut self.work.stage, &self.rows8)?;
            let mut a = chain(op::IN_RESCALE);
            a.y = stage;
            a.x = xr as *mut c_void;
            a.c = inv_c;
            Self::launch(ctx, s, a)?;
            Self::key_switch(ctx, s, k, xr, res, N, 1, 1, k.fwd.dptr(), scratch)?;
            let mut a = Self::kargs(k, op::ADD, N);
            a.count = 2;
            a.i0 = N as i32;
            a.x = xr.wrapping_add(2 * N) as *mut c_void;
            a.y = res.wrapping_add(2 * N) as *const c_void;
            Self::launch(ctx, s, a)?;
            for (opc, src) in [(op::MLWE_A, res as *const u32), (op::MLWE_B, xr.wrapping_add(2 * N) as *const u32)] {
                let mut a = Self::kargs(k, opc, N);
                a.count = nch as i32;
                a.i0 = ri as i32;
                a.i1 = (c * nch) as i32;
                a.i2 = r as i32;
                a.i3 = np as i32;
                a.y = src as *const c_void;
                a.x = (if opc == op::MLWE_A { pa } else { pbm }) as *mut c_void;
                Self::launch(ctx, s, a)?;
            }
        }
        if let Some(e) = ev {
            e[1].record(s)?;
        }
        let wd = planes.dptr() as *const i8;
        self.mod_gemm(ctx, s, k, wd, sh.nw, pa, pap, ri, ro, N1)?;
        self.mod_gemm(ctx, s, k, wd, sh.nw, pbm, pbp, ri, ro, np)?;
        if let Some(e) = ev {
            e[2].record(s)?;
        }
        let pkb = ppk.wrapping_add(L * S_SWITCH * N1);
        for (o, &(ct, m)) in outs.iter().enumerate() {
            let mut a = Self::kargs(k, op::PACK_C, N1);
            a.count = nch as i32;
            a.i0 = ro as i32;
            a.i1 = (o * nch) as i32;
            a.i2 = r as i32;
            a.i3 = np as i32;
            a.y = pap as *const c_void;
            a.x = pc as *mut c_void;
            Self::launch(ctx, s, a)?;
            Self::key_switch(ctx, s, k, pc, ppk, N1, S_SWITCH, r, k.pack.dptr(), scratch)?;
            let mut a = Self::kargs(k, op::ADD_BPK, N1);
            a.i0 = ro as i32;
            a.i1 = (o * nch) as i32;
            a.i2 = r as i32;
            a.i3 = np as i32;
            a.y = pbp as *const c_void;
            a.x = pkb as *mut c_void;
            Self::launch(ctx, s, a)?;
            for (src, dst) in [(ppk, pra), (pkb, prb)] {
                let mut a = Self::kargs(k, op::COMPOSE, N);
                a.y = src as *const c_void;
                a.x = dst as *mut c_void;
                Self::launch(ctx, s, a)?;
            }
            Self::key_switch(ctx, s, k, pra, res, N, 1, 1, k.back.dptr(), scratch)?;
            let mut a = Self::kargs(k, op::ADD, N);
            a.count = 2;
            a.i0 = N as i32;
            a.x = res.wrapping_add(2 * N) as *mut c_void;
            a.y = prb as *const c_void;
            Self::launch(ctx, s, a)?;
            for part in 0..2 {
                let mut a = chain(op::LIFT);
                a.y = res.wrapping_add(part * 2 * N) as *const c_void;
                a.x = (stage as *mut W).wrapping_add(part * 4 * N) as *mut c_void;
                a.c[0] = lift[0];
                a.c[1] = lift[1];
                Self::launch(ctx, s, a)?;
            }
            ntt::transform_rows(ctx, s, Dir::Forward, &mut self.work.stage, &self.rows8)?;
            let mut a = chain(op::SCATTER);
            a.ct = ct;
            a.i0 = m as i32;
            a.y = stage;
            Self::launch(ctx, s, a)?;
        }
        if let Some(e) = ev {
            e[3].record(s)?;
        }
        Ok(())
    }

    fn check_split<M: DevStrategy<W>>(ctx: &Context<'_, W, M>, x: &Ct<'_, W>) -> Result<()> {
        let top = crate::bts::stc_first::split_top(&ctx.chain);
        if ctx.lay != x.buf.lay || ctx.top_of(x.meta.level)? != top {
            return Err(shape("pcmm", format!("ciphertext at level {} (top {:?}): the split holds limbs 0..{top}", x.meta.level, ctx.top_of(x.meta.level).ok())));
        }
        Ok(())
    }

    fn members(cts: &[Ct<'_, W>]) -> Vec<(usize, usize)> {
        cts.iter().enumerate().flat_map(|(i, c)| (0..c.k()).map(move |m| (i, m))).collect()
    }

    /// y = x W: the `n_in` input blocks are the members of `ins` in order (each at the split), output block m goes
    /// to member `out_map[m] = (ciphertext, member)` of `outs`; every ciphertext of `outs` named by `out_map` gets the
    /// split metadata of `ins[0]` with scale `nf_out`. `ev`: 4 timing events recorded around the phases
    /// (`PcmmTimes::read`).
    #[allow(clippy::too_many_arguments)]
    pub fn product<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, sh: &PcmmShape, planes: &DBuf<'_, Compute, u8>, ins: &[Ct<'_, W>], outs: &mut [Ct<'_, W>], out_map: &[(usize, usize)], nf_out: f64, ev: Option<[&Event; 4]>) -> Result<()> {
        for x in ins {
            Self::check_split(ctx, x)?;
        }
        let meta = ins.first().ok_or_else(|| shape("pcmm", "no inputs"))?.meta.clone();
        let ir: Vec<(PolyRef, usize)> = Self::members(ins).into_iter().map(|(i, m)| (ins[i].buf.r(), m)).collect();
        for &(c, m) in out_map {
            if c >= outs.len() || m >= outs[c].k() || outs[c].buf.lay != ctx.lay {
                return Err(shape("pcmm", format!("output member ({c}, {m}) outside the {} output ciphertexts", outs.len())));
            }
        }
        let or: Vec<(PolyRef, usize)> = out_map.iter().map(|&(c, m)| (outs[c].buf.w(), m)).collect();
        self.run_members(ctx, s, sh, planes, &ir, &or, ev)?;
        for &(c, _) in out_map {
            outs[c].meta = CtMeta { nf: nf_out, ..meta.clone() };
        }
        Ok(())
    }

    /// `product` with the outputs written over the inputs' storage: inputs = the first `n_in` members of `io`,
    /// output m to member `out_map[m]` of `io` (perseus pcmmMulti's outs == ins contract).
    #[allow(clippy::too_many_arguments)]
    pub fn product_inplace<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, sh: &PcmmShape, planes: &DBuf<'_, Compute, u8>, io: &mut [Ct<'_, W>], out_map: &[(usize, usize)], nf_out: f64, ev: Option<[&Event; 4]>) -> Result<()> {
        let all = Self::members(io);
        if all.len() < sh.n_in {
            return Err(shape("pcmm", format!("{} members for {} inputs", all.len(), sh.n_in)));
        }
        for &(c, _) in &all[..sh.n_in] {
            Self::check_split(ctx, &io[c])?;
        }
        let meta = io[all[0].0].meta.clone();
        let ir: Vec<(PolyRef, usize)> = all[..sh.n_in].iter().map(|&(c, m)| (io[c].buf.r(), m)).collect();
        for &(c, m) in out_map {
            if c >= io.len() || m >= io[c].k() {
                return Err(shape("pcmm", format!("output member ({c}, {m}) outside io")));
            }
        }
        let or: Vec<(PolyRef, usize)> = out_map.iter().map(|&(c, m)| (io[c].buf.w(), m)).collect();
        self.run_members(ctx, s, sh, planes, &ir, &or, ev)?;
        for &(c, _) in out_map {
            io[c].meta = CtMeta { nf: nf_out, ..meta.clone() };
        }
        Ok(())
    }

    /// x *= 2^e on the live Q rows of every member (the band shift's undo after the refresh; metadata unchanged).
    pub fn mult_pow2<M: DevStrategy<W>>(&self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, x: &mut Ct<'_, W>, e: u32) -> Result<()> {
        let top = ctx.top_of(x.meta.level)?;
        self.pow2.mult(ctx, s, View::of(&x.buf, 0), x.k(), top, e)
    }
}

// ── the linear: inputs -> StC -> product -> finishes ─────────────────────────────────────────────────────────────

/// How a linear's refreshes run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearOpts {
    /// The refresh's own correction factor (perseus PF_PCMM_CF; 6 on n32: the outputs are band-scaled).
    pub cf: u32,
    /// Real-payload EvalMod chain (the PCMM's token blocks are complex: false).
    pub real: bool,
    /// Issue the output finishes stage-major across the groups (perseus PF_PCMM_INTERLEAVE) instead of one after another.
    pub interleave: bool,
}

impl Default for LinearOpts {
    fn default() -> Self {
        LinearOpts { cf: 6, real: false, interleave: false }
    }
}

/// The output finishes' grouping (perseus `_finish_all`): uniform drops -> chunks of `batch`; per-output drops -> one
/// output at a time, or (`batch_drops`, perseus PF_PCMM_BATCH_DROPS) the outputs of each drop value together (first
/// appearance order) in chunks of `batch`. Each group becomes one K-batched finish.
pub fn finish_groups(drops: &[u32], batch: usize, batch_drops: bool) -> Result<Vec<Vec<usize>>> {
    if batch > MAX_BATCH {
        return Err(Error::Mem { knob: "pcmm batch (PF_PCMM_BATCH)", value: batch, msg: format!("at most {MAX_BATCH}") });
    }
    let b = batch.max(1);
    let uniform = drops.windows(2).all(|w| w[0] == w[1]);
    let idx: Vec<usize> = (0..drops.len()).collect();
    if uniform {
        return Ok(idx.chunks(b).map(|c| c.to_vec()).collect());
    }
    if !(b > 1 && batch_drops) {
        return Ok(idx.into_iter().map(|m| vec![m]).collect());
    }
    let mut seen: Vec<u32> = Vec::new();
    for &d in drops {
        if !seen.contains(&d) {
            seen.push(d);
        }
    }
    let mut out = Vec::new();
    for d in seen {
        let ms: Vec<usize> = idx.iter().copied().filter(|&m| drops[m] == d).collect();
        out.extend(ms.chunks(b).map(|c| c.to_vec()));
    }
    Ok(out)
}

/// The capture descriptor of output m (pcmm.py `Pcmm.desc`; graph v2 `args.external`).
pub fn desc_json(lin: &PcmmLinear, m: usize, drop: u32) -> String {
    format!(
        "{{\"op\": \"pcmm\", \"linear\": {:?}, \"out\": {m}, \"n_in\": {}, \"n_out\": {}, \"t\": {}, \"shift\": {}, \"raise_drop\": {drop}}}",
        lin.name, lin.shape.n_in, lin.shape.n_out, lin.shape.t, lin.shift
    )
}

/// One PCMM linear end to end (perseus `Pcmm._device_cf` + `_finish_all`): stamp `w_scale 2^shift` into the inputs'
/// scale, StC-first input stages at the call's own CF (`ins` end at the split), the product into `outs` (one
/// ciphertext per finish group: `groups[g]` lists its output indices in member order, `drops[g]` its raise drop),
/// each group finished as one K-batched staged refresh (`finish_call(drop)` -> `at_split` -> Done), then x 2^shift.
/// `rec`: capture hooks (input level hints, one external bootstrap per output with its descriptor).
#[allow(clippy::too_many_arguments)]
pub fn linear<'r, 'a, 'lt, W: DevWord, M: DevStrategy<W>>(
    cx: &mut BtsCx<'_, 'a, W, M>,
    route: &'r StcFirstRoute<'a, W, M>,
    pcmm: &mut Pcmm<'a, 'lt, W>,
    lin: &PcmmLinear,
    planes: &DBuf<'_, Compute, u8>,
    ins: &mut [Ct<'a, W>],
    outs: &mut [Ct<'a, W>],
    groups: &[Vec<usize>],
    drops: &[u32],
    opts: LinearOpts,
    mut rec: Option<&mut dyn BtsRecorder>,
    hooks: Option<LinearHooks<'_>>,
) -> Result<()> {
    let hooks = hooks.unwrap_or_default();
    let sh = lin.shape;
    if groups.len() != outs.len() || drops.len() != outs.len() {
        return Err(shape("pcmm linear", format!("{} groups, {} drops for {} output ciphertexts", groups.len(), drops.len(), outs.len())));
    }
    let mut out_map = vec![(usize::MAX, 0); sh.n_out];
    for (g, ms) in groups.iter().enumerate() {
        if ms.len() != outs[g].k() || ms.len() > MAX_BATCH {
            return Err(shape("pcmm linear", format!("group {g} has {} outputs, its ciphertext {} members (<= {MAX_BATCH})", ms.len(), outs[g].k())));
        }
        for (j, &m) in ms.iter().enumerate() {
            if m >= sh.n_out || out_map[m].0 != usize::MAX {
                return Err(shape("pcmm linear", format!("output {m} listed twice or outside n_out {}", sh.n_out)));
            }
            out_map[m] = (g, j);
        }
    }
    if out_map.iter().any(|x| x.0 == usize::MAX) {
        return Err(shape("pcmm linear", "every output must be in one group"));
    }
    let names: Vec<(String, u32)> = (0..ins.len()).map(|i| (format!("{}.in{i}", lin.name), ins[i].meta.level)).collect();
    let stamp = lin.w_scale * (lin.shift as f64).exp2();
    for x in ins.iter_mut() {
        x.meta.nf *= stamp;
    }
    let r = Route { cf: Some(opts.cf), stc_first: true, real: opts.real, ..Route::dense() };
    let mut inp = BtsCall::begin(StagedRef::StcFirst(route), cx, ins, &r, CallOpts { prescaled: false, own_cf: true })?;
    if inp.stc_first_input(cx, ins)?.is_none() {
        return Err(shape("pcmm linear", format!("{}: inputs at level {} are below the StC-first entry (cap {})", lin.name, ins[0].meta.level, route.input_cap(cx.ctx.shape.n_q, cx.ctx.shape.unit))));
    }
    if let Some(rc) = rec.as_deref_mut() {
        let ni: Vec<(&str, u32)> = names.iter().map(|(n, l)| (n.as_str(), *l)).collect();
        record_stc_first_inputs(rc, route, cx, &ni);
    }
    if let Some(e) = hooks.after_stc {
        e.record(&cx.ex.compute)?;
    }
    let nf_out = ins[0].meta.nf / lin.w_scale;
    pcmm.product(cx.ctx, &cx.ex.compute, &sh, planes, ins, outs, &out_map, nf_out, hooks.product)?;
    let mut calls = Vec::with_capacity(outs.len());
    for (g, &d) in drops.iter().enumerate() {
        let mut fin = inp.finish_call(d)?;
        fin.at_split(cx, &outs[g..g + 1])?;
        calls.push(fin);
    }
    if opts.interleave {
        let mut sl: Vec<&mut [Ct<'a, W>]> = outs.chunks_mut(1).collect();
        crate::bts::staged::run_interleaved(cx, &mut calls, &mut sl, Phase::Done)?;
    } else {
        for (g, fin) in calls.iter_mut().enumerate() {
            fin.run_to(cx, &mut outs[g..g + 1], Phase::Done)?;
        }
    }
    if let Some(e) = hooks.after_finish {
        e.record(&cx.ex.compute)?;
    }
    for (g, x) in outs.iter_mut().enumerate() {
        if lin.shift > 0 {
            pcmm.mult_pow2(cx.ctx, &cx.ex.compute, x, lin.shift)?;
        }
        if let Some(rc) = rec.as_deref_mut() {
            let ni: Vec<(&str, u32)> = names.iter().map(|(n, l)| (n.as_str(), *l)).collect();
            for &m in &groups[g] {
                rc.external_bootstrap(&calls[g].external(&ni, &x.meta, &desc_json(lin, m, drops[g]), None));
            }
        }
    }
    Ok(())
}

// ── calibration (pcmm_calib.json) ────────────────────────────────────────────────────────────────────────────────

/// Calibrated max |y| per linear (a fake run's `PF_PCMM_CALIB` output, read back as `PF_PCMM_CALIB_IN`), and the
/// band shift rule of pcmm.py `band_shift`: shift = max(0, ceil(log2(ymax margin / band))).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PcmmCalib {
    pub ymax: BTreeMap<String, f64>,
}

impl PcmmCalib {
    pub fn parse(text: &str) -> Result<Self> {
        let mut ymax = BTreeMap::new();
        let body = text.trim().trim_start_matches('{').trim_end_matches('}');
        for item in body.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (k, v) = item.split_once(':').ok_or_else(|| shape("pcmm_calib.json", format!("entry {item:?}")))?;
            let k = k.trim().trim_matches('"').to_string();
            let v: f64 = v.trim().parse().map_err(|_| shape("pcmm_calib.json", format!("value of {k}: {v:?}")))?;
            ymax.insert(k, v);
        }
        Ok(PcmmCalib { ymax })
    }
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let s = std::fs::read_to_string(path).map_err(|e| shape("pcmm_calib.json", format!("{}: {e}", path.display())))?;
        Self::parse(&s)
    }
    /// The band shift of `name` (margin 1.5, band 0.25 are perseus' defaults PF_PCMM_MARGIN / PF_PCMM_BAND).
    pub fn shift(&self, name: &str, margin: f64, band: f64) -> Result<u32> {
        let y = self.ymax.get(name).ok_or_else(|| shape("pcmm calib", format!("no calibrated max |y| for {name} (pcmm_calib.json)")))?;
        Ok(((y * margin / band).log2().ceil()).max(0.0) as u32)
    }
    /// Keep the running max of a calibration run.
    pub fn record(&mut self, name: &str, ymax: f64) {
        let e = self.ymax.entry(name.to_string()).or_insert(0.0);
        *e = e.max(ymax);
    }
    pub fn to_json(&self) -> String {
        let items: Vec<String> = self.ymax.iter().map(|(k, v)| format!(" {k:?}: {v:?}")).collect();
        format!("{{\n{}\n}}", items.join(",\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups() {
        assert_eq!(finish_groups(&[0; 6], 4, false).unwrap(), vec![vec![0, 1, 2, 3], vec![4, 5]]);
        assert_eq!(finish_groups(&[0, 1, 0, 1], 2, false).unwrap(), vec![vec![0], vec![1], vec![2], vec![3]]);
        assert_eq!(finish_groups(&[0, 1, 0, 1, 0], 2, true).unwrap(), vec![vec![0, 2], vec![4], vec![1, 3]]);
        assert!(finish_groups(&[0; 3], 5, false).is_err());
    }

    #[test]
    fn calib() {
        let c = PcmmCalib::parse("{\n \"b0.qkv\": 4.5,\n \"b0.down\": 486.0\n}").unwrap();
        assert_eq!(c.shift("b0.qkv", 1.5, 0.25).unwrap(), 5); // 27 -> 2^5
        assert_eq!(c.shift("b0.down", 1.5, 0.25).unwrap(), 12);
        assert!(c.shift("b1.qkv", 1.5, 0.25).is_err());
        assert_eq!(PcmmCalib::parse(&c.to_json()).unwrap(), c);
    }

    #[test]
    fn shapes() {
        let s = PcmmShape { t: 8, n_in: 4, n_out: 1, nw: 3 };
        assert_eq!((s.rows_in().unwrap(), s.rows_out().unwrap()), (16384, 4096));
        assert!(s.check().is_ok());
        assert!(PcmmShape { t: 8, n_in: 11, n_out: 1, nw: 3 }.check().is_err()); // 45056 > 43690
        assert_eq!(chunk_rows(3072, N1, 2, 1 << 30), 3072);
        assert_eq!(chunk_rows(12288, N1, 3, 1 << 30), 5376);
    }
}
