//! CCMM: batch ciphertext x ciphertext matrix products on SinC operands (Cheon-Kang-Lee, ePrint 2025/1957, Alg. 2-4),
//! a port of FIDESlib `Ccmm.cu` (`ccmmUploadKeys`, `ccmmTranspose`, `ccmmMul`) with every buffer carved from the
//! MemPlan (`Ccmm::bytes`), no device syncs, and the PPMM's int8 GEMM one batched cuBLASLt plan autotuned at setup.
//!
//! A matrix is 16 ciphertexts at the CCMM split (limbs 0..3, eval form, SinC_{1024, 2^16}); per matrix: key switch
//! to s'(X^8), stride-8 ring switch into 128 column ciphertexts of R_{2^13}, CMT of the right operand, the four
//! PPMMs as one batched int8 GEMM per prime (exact), recombination, CMTs, relinearization, ring switch back, NTT.
//! Inputs are read before any output is written (`mul_inplace`). Leaf services (`sinc_boot`, `leaf`) serve
//! perseus `impl/ccmm.py`. Details: docs/ARCHITECTURE.md "pc-gpu CCMM".
use super::gemm::{tn_spec_batched, I8Gemm};
use crate::bts::staged::{BtsCall, CallOpts, Phase, StagedRef};
use crate::bts::stc_first::StcFirstRoute;
use crate::bts::{common_meta, BtsCx, BtsRoute, BtsState, StcHigh, Staged};
use crate::context::{Context, DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Result};
use crate::ntt::{self, Dir, RowList};
use pc_core::ccmm_keys::{key_words, tstar, CcmmKeys, D, KY, NB, NBIG, NP, NQ, NS};
use pc_core::modmul::invmod;
use pc_core::pcmm_keys::NttTab;
use pc_cuda::blas::lt;
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, DBuf, Device, Event, Stream};
use pc_kernels::{ccmm_op as op, CcmmArgs, PolyRef};
use pc_shadow::{CtMeta, Route, Shadow};
use std::ffi::c_void;

/// Words of a column set X = \[part\]\[prime\][col][NS].
pub const COLS: usize = 2 * NQ * D * NS;
/// Digit-plane bytes of one GEMM operand (one plane): [u][2D][D].
const PER: usize = 2 * D * D * KY;
/// int32 words of one exponent accumulator: [u][2D][2D].
const GPER: usize = KY * 4 * D * D;
/// Exponents of the 4 x 4 digit pairs.
const EXPS: usize = 7;

/// The CCMM engine's knobs.
#[derive(Clone, Debug, PartialEq)]
pub struct CcmmCfg {
    /// Time the heuristic candidates of the PPMM GEMM at setup and keep the fastest.
    pub autotune: bool,
    pub max_algos: usize,
    /// cuBLASLt workspace bytes.
    pub gemm_workspace: usize,
}

impl Default for CcmmCfg {
    fn default() -> Self {
        CcmmCfg { autotune: true, max_algos: 8, gemm_workspace: 32 << 20 }
    }
}

fn mu64(p: u64) -> u64 {
    ((1u128 << 64) / p as u128) as u64
}

// ── keys on the device ───────────────────────────────────────────────────────────────────────────────────────────

/// The CCMM keys in the device's NTT form, with the twiddle tables at 2^16, 2^13, 2^6 and the per-prime constants.
pub struct DevCcmmKeys<'a> {
    pub primes: [u32; NP],
    pub mu: [u64; NP],
    /// [16, 13, 6] x [prime][n]
    tw: Vec<ConstBuf<'a, u32>>,
    itw: Vec<ConstBuf<'a, u32>>,
    ninv: [[u32; NP]; 3],
    fwd: ConstBuf<'a, u32>,
    back: ConstBuf<'a, u32>,
    autos: ConstBuf<'a, u32>,
    rlk: ConstBuf<'a, u32>,
    tstar: ConstBuf<'a, i32>,
    /// (Q / q_j)^-1 mod q_j, P^-1 mod q_i, D^-1 mod q_i
    hat: [u32; NQ],
    pinv: [u32; NQ],
    dinv: [u32; NQ],
}

const SIZES: [usize; 3] = [NB, NS, KY];

impl<'a> DevCcmmKeys<'a> {
    pub fn bytes() -> usize {
        let u = |n: usize| Arena::bytes_for::<u32>(n);
        SIZES.iter().map(|&n| 2 * u(NP * n)).sum::<usize>() + 2 * u(key_words(NB)) + u(D * key_words(NS)) + u(key_words(NS)) + Arena::bytes_for::<i32>(D)
    }

    /// Upload `k` (NTT forms computed on the host; setup, synchronous on `s`).
    pub fn upload(arena: &'a Arena, s: &Stream<Compute>, k: &CcmmKeys) -> Result<Self> {
        let mut tw = Vec::new();
        let mut itw = Vec::new();
        let mut ninv = [[0u32; NP]; 3];
        let mut tabs16 = None;
        let mut tabs13 = None;
        for (si, &n) in SIZES.iter().enumerate() {
            let tabs = k.tables(n)?;
            let cat = |inv: bool| -> Vec<u32> { tabs.iter().flat_map(|t: &NttTab| if inv { t.ipsi.clone() } else { t.psi.clone() }).collect() };
            tw.push(ConstBuf::new(arena, "ccmm.keys", s, &cat(false))?);
            itw.push(ConstBuf::new(arena, "ccmm.keys", s, &cat(true))?);
            for i in 0..NP {
                ninv[si][i] = tabs[i].ninv;
            }
            match si {
                0 => tabs16 = Some(tabs),
                1 => tabs13 = Some(tabs),
                _ => {}
            }
        }
        let (t16, t13) = (tabs16.ok_or_else(|| shape("ccmm keys", "tables"))?, tabs13.ok_or_else(|| shape("ccmm keys", "tables"))?);
        let p = k.primes;
        let mut hat = [0u32; NQ];
        let mut pinv = [0u32; NQ];
        let mut dinv = [0u32; NQ];
        for j in 0..NQ {
            let mut prod = 1u64;
            for i in 0..NQ {
                if i != j {
                    prod = (prod as u128 * (p[i] % p[j]) as u128 % p[j] as u128) as u64;
                }
            }
            hat[j] = invmod(prod, p[j]) as u32;
            pinv[j] = invmod(p[NQ] % p[j], p[j]) as u32;
            dinv[j] = invmod(D as u64 % p[j], p[j]) as u32;
        }
        let tstar: Vec<i32> = (0..D).map(|t| tstar(t) as i32).collect();
        Ok(DevCcmmKeys {
            primes: [p[0] as u32, p[1] as u32, p[2] as u32, p[3] as u32, p[4] as u32],
            mu: [mu64(p[0]), mu64(p[1]), mu64(p[2]), mu64(p[3]), mu64(p[4])],
            tw,
            itw,
            ninv,
            fwd: ConstBuf::new(arena, "ccmm.keys", s, &CcmmKeys::eval_form(&k.fwd, &t16))?,
            back: ConstBuf::new(arena, "ccmm.keys", s, &CcmmKeys::eval_form(&k.back, &t16))?,
            autos: ConstBuf::new(arena, "ccmm.keys", s, &CcmmKeys::eval_form(&k.autos, &t13))?,
            rlk: ConstBuf::new(arena, "ccmm.keys", s, &CcmmKeys::eval_form(&k.rlk, &t13))?,
            tstar: ConstBuf::new(arena, "ccmm.keys", s, &tstar)?,
            hat,
            pinv,
            dinv,
        })
    }

    fn table(&self, n: usize) -> Result<usize> {
        SIZES.iter().position(|&m| m == n).ok_or_else(|| shape("ccmm ntt", format!("no tables at degree {n}")))
    }
}

// ── the engine ───────────────────────────────────────────────────────────────────────────────────────────────────

struct Work<'a, W: DevWord> {
    /// [slot < 16][part][4][NB] W: a matrix's ciphertexts (INTT'd), later the outputs' (NTT'd)
    stage: DBuf<'a, Compute, W>,
    /// [ab][4][polys][n] key-switch results (16 x 2^16 or 128 x 2^13 words per prime, the same size)
    res: DBuf<'a, Compute, u32>,
    /// key-switch digits Dg [4][5][polys][n] + accumulators [2][5][polys][n]
    scratch: DBuf<'a, Compute, u32>,
    xm: DBuf<'a, Compute, u32>,
    xmp: DBuf<'a, Compute, u32>,
    t1: DBuf<'a, Compute, u32>,
    t2: DBuf<'a, Compute, u32>,
    x1: DBuf<'a, Compute, u32>,
    /// [2 matrices][COLS] the PPMM operands, NTT along Y
    e: DBuf<'a, Compute, u32>,
    xd: DBuf<'a, Compute, u8>,
    yd: DBuf<'a, Compute, u8>,
    g: DBuf<'a, Compute, i32>,
    cn: DBuf<'a, Compute, u32>,
    /// [4][16][NB] the composed `a` part of the outputs
    biga: DBuf<'a, Compute, u32>,
    gws: DBuf<'a, Compute, u8>,
}

/// Device timings of a call's phases (ms; perseus `times[5]`: ring in, CMT, PPMM, CMT x2 + relin, ring out).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CcmmTimes {
    pub ring_in: f64,
    pub cmt: f64,
    pub ppmm: f64,
    pub cmt2_relin: f64,
    pub ring_out: f64,
}

impl CcmmTimes {
    /// Host-waits for the last event.
    pub fn read(ev: [&Event; 6]) -> Result<CcmmTimes> {
        ev[5].sync()?;
        let d = |a: usize| -> Result<f64> { Ok(ev[a].elapsed_ms(ev[a + 1])? as f64) };
        Ok(CcmmTimes { ring_in: d(0)?, cmt: d(1)?, ppmm: d(2)?, cmt2_relin: d(3)?, ring_out: d(4)? })
    }
    pub fn total(&self) -> f64 {
        self.ring_in + self.cmt + self.ppmm + self.cmt2_relin + self.ring_out
    }
    pub fn as_array(&self) -> [f64; 5] {
        [self.ring_in, self.cmt, self.ppmm, self.cmt2_relin, self.ring_out]
    }
}

/// A matrix operand: the 16 ciphertexts as (buffer view, member).
type Members = Vec<(PolyRef, usize)>;

/// The CCMM engine of a session: keys, the PPMM GEMM plan, one set of work buffers.
pub struct Ccmm<'a, 'lt, W: DevWord> {
    pub cfg: CcmmCfg,
    pub keys: DevCcmmKeys<'a>,
    pub gemm: I8Gemm<'lt>,
    /// The PPMM GEMM over one half of the left operand's rows (a two-device product, `enable_split`).
    pub gemm_half: Option<I8Gemm<'lt>>,
    work: Work<'a, W>,
    rows: RowList<'a>,
}

impl<'a, 'lt, W: DevWord> Ccmm<'a, 'lt, W> {
    /// Arena bytes of the engine (keys, work buffers, staging row list); `MemPlan::reserve("ccmm", ..)`.
    pub fn bytes(cfg: &CcmmCfg) -> usize {
        let u = |n: usize| Arena::bytes_for::<u32>(n);
        DevCcmmKeys::bytes()
            + Arena::bytes_for::<W>(NBIG * 8 * NB)
            + u(2 * NQ * NBIG * NB)
            + u((NQ * NP + 2 * NP) * NBIG * NB)
            + 5 * u(COLS)
            + u(2 * COLS)
            + 2 * Arena::bytes_for::<u8>(4 * PER)
            + Arena::bytes_for::<i32>(EXPS * GPER)
            + u(NQ * GPER)
            + u(NQ * NBIG * NB)
            + Arena::bytes_for::<u8>(cfg.gemm_workspace)
            + Arena::bytes_for::<u32>(NBIG * 8)
            + Arena::bytes_for::<i32>(NBIG * 8)
    }

    /// Setup: upload the keys (made for the chain's four bottom moduli), carve the work buffers, plan the PPMM GEMM
    /// (and time its candidates when `cfg.autotune`). Synchronous on `s`.
    pub fn new<M: DevStrategy<W>>(arena: &'a Arena, dev: &Device, s: &Stream<Compute>, ctx: &Context<'_, W, M>, lt_: &'lt lt::Lt, cfg: CcmmCfg, keys: &CcmmKeys) -> Result<Self> {
        if ctx.lay.n != NB {
            return Err(shape("ccmm", format!("ring degree {} (the CCMM needs 2^16)", ctx.lay.n)));
        }
        let chain: Vec<u64> = (0..NQ).map(|i| ctx.prime(i)).collect();
        if keys.primes[..NQ] != chain[..] {
            return Err(shape("ccmm keys", format!("keys made for moduli {:?}, the chain's four bottom ones are {chain:?}", &keys.primes[..NQ])));
        }
        let dk = DevCcmmKeys::upload(arena, s, keys)?;
        let u = |n| arena.alloc::<Compute, u32>("ccmm.work", n);
        let work = Work {
            stage: arena.alloc("ccmm.work", NBIG * 8 * NB)?,
            res: u(2 * NQ * NBIG * NB)?,
            scratch: u((NQ * NP + 2 * NP) * NBIG * NB)?,
            xm: u(COLS)?,
            xmp: u(COLS)?,
            t1: u(COLS)?,
            t2: u(COLS)?,
            x1: u(COLS)?,
            e: u(2 * COLS)?,
            xd: arena.alloc("ccmm.work", 4 * PER)?,
            yd: arena.alloc("ccmm.work", 4 * PER)?,
            g: arena.alloc("ccmm.work", EXPS * GPER)?,
            cn: u(NQ * GPER)?,
            biga: u(NQ * NBIG * NB)?,
            gws: arena.alloc("ccmm.gemm_ws", cfg.gemm_workspace)?,
        };
        // staging rows: (slot, part, limb) -> limb offset slot*8 + part*4 + l, prime id l
        let rows_v: Vec<(u32, i32)> = (0..NBIG * 8).map(|r| (r as u32, (r % 4) as i32)).collect();
        let rows = RowList::new(arena, s, &rows_v)?;
        let gemm = I8Gemm::with_spec(lt_, tn_spec_batched(2 * D as u64, 2 * D as u64, D as u64, KY as i32), cfg.max_algos, cfg.gemm_workspace)?;
        let mut me = Ccmm { cfg, keys: dk, gemm, gemm_half: None, work, rows };
        if me.cfg.autotune {
            me.autotune(dev, s)?;
        }
        s.sync()?;
        Ok(me)
    }

    /// Plan (and with `cfg.autotune` time) the half-rows PPMM GEMM a two-device product runs (`mul_split`).
    /// Synchronous on `s`.
    pub fn enable_split(&mut self, lt_: &'lt lt::Lt, dev: &Device, s: &Stream<Compute>) -> Result<()> {
        if self.gemm_half.is_some() {
            return Ok(());
        }
        // the full plan's strides (a left half is every u's rows h D .. (h + 1) D of the 2D-row plane)
        let mut spec = tn_spec_batched(2 * D as u64, 2 * D as u64, D as u64, KY as i32);
        spec.n = D as u64;
        let mut g = I8Gemm::with_spec(lt_, spec, self.cfg.max_algos, self.cfg.gemm_workspace)?;
        if self.cfg.autotune {
            let ev = (Event::timing(dev)?, Event::timing(dev)?);
            let w = &mut self.work;
            w.xd.zero(s)?;
            w.yd.zero(s)?;
            let (a, b, c) = (w.yd.dptr() as *const i8, w.xd.dptr() as *const i8, w.g.dptr_mut());
            // SAFETY: as `autotune`: the half plan reads and writes within one digit plane / one exponent
            unsafe { g.tune(s, (&ev.0, &ev.1), a, b, c, &mut w.gws, 5)? };
        }
        s.sync()?;
        self.gemm_half = Some(g);
        Ok(())
    }

    /// Time the PPMM GEMM's candidates on the work buffers (contents irrelevant) and keep the fastest.
    pub fn autotune(&mut self, dev: &Device, s: &Stream<Compute>) -> Result<()> {
        let ev = (Event::timing(dev)?, Event::timing(dev)?);
        let w = &mut self.work;
        w.xd.zero(s)?;
        w.yd.zero(s)?;
        let (a, b, c) = (w.yd.dptr() as *const i8, w.xd.dptr() as *const i8, w.g.dptr_mut());
        // SAFETY: A, B = one digit plane (PER bytes = the batched operand), C = one exponent (GPER int32) of live work
        // buffers, on the compute lane
        unsafe { self.gemm.tune(s, (&ev.0, &ev.1), a, b, c, &mut w.gws, 5) }
    }

    fn launch<M: DevStrategy<W>>(ctx: &Context<'_, W, M>, s: &Stream<Compute>, a: CcmmArgs) -> Result<()> {
        // SAFETY: every pointer in `a` addresses a live engine buffer / a caller's ciphertext of the size the op
        // reads or writes (fixed CCMM shapes, checked at setup and by `members`); all on the compute lane
        launch_result("ccmm", unsafe { (ctx.table.ccmm)(&a, s.raw()) })?;
        Ok(())
    }

    fn kargs(&self, opc: i32, n: usize) -> CcmmArgs {
        let k = &self.keys;
        let mut a = CcmmArgs { op: opc, n: n as i32, ..CcmmArgs::default() };
        a.p[..NP].copy_from_slice(&k.primes);
        a.mu[..NP].copy_from_slice(&k.mu);
        a
    }

    /// res [ab][4][polys][n] = KS of the `polys` polynomials whose digit j sits at `src + j s0 + q s1` (W words when
    /// `w_src`), key `key + q kstride` (NTT form), tables of degree n.
    #[allow(clippy::too_many_arguments)]
    fn key_switch<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, n: usize, polys: usize, src: *const c_void, s0: usize, s1: usize, w_src: bool, key: *const u32, kstride: usize) -> Result<()> {
        let ti = self.keys.table(n)?;
        let d = self.work.scratch.dptr_mut();
        let acc = d.wrapping_add(NQ * NP * polys * n);
        let res = self.work.res.dptr_mut();
        let mut a = self.kargs(op::KS_DIGITS, n);
        a.count = polys as i32;
        a.i4 = w_src as i32;
        a.y = src;
        a.x = d as *mut c_void;
        a.s[0] = s0 as i64;
        a.s[1] = s1 as i64;
        a.c[..NQ].copy_from_slice(&self.keys.hat);
        Self::launch(ctx, s, a)?;
        let mut a = self.kargs(op::NTT, n);
        a.count = (NQ * NP * polys) as i32;
        a.i0 = polys as i32;
        a.i1 = NP as i32;
        a.x = d as *mut c_void;
        a.tw = self.keys.tw[ti].vptr();
        Self::launch(ctx, s, a)?;
        let mut a = self.kargs(op::KS_MAC, n);
        a.count = polys as i32;
        a.y = d as *const c_void;
        a.z = key as *const c_void;
        a.x = acc as *mut c_void;
        a.s[0] = kstride as i64;
        Self::launch(ctx, s, a)?;
        let mut a = self.kargs(op::INTT, n);
        a.count = (2 * NP * polys) as i32;
        a.i0 = polys as i32;
        a.i1 = NP as i32;
        a.x = acc as *mut c_void;
        a.tw = self.keys.itw[ti].vptr();
        a.c[..NP].copy_from_slice(&self.keys.ninv[ti]);
        Self::launch(ctx, s, a)?;
        let mut a = self.kargs(op::MOD_DOWN, n);
        a.count = polys as i32;
        a.y = acc as *const c_void;
        a.x = res as *mut c_void;
        a.c[..NQ].copy_from_slice(&self.keys.pinv);
        Self::launch(ctx, s, a)
    }

    /// Ciphertexts `c0 .. c0 + ins.len()` of a matrix -> their columns 8 c0 .. of `x` (FIDESlib ringIn; all 16 with
    /// `c0` = 0).
    fn ring_in<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, ins: &[(PolyRef, usize)], c0: usize, x: *mut u32) -> Result<()> {
        let cnt = ins.len();
        if cnt == 0 || c0 + cnt > NBIG {
            return Err(shape("ccmm ring in", format!("ciphertexts {c0} .. {} of a {NBIG}-ciphertext matrix", c0 + cnt)));
        }
        let stage = self.work.stage.dptr_mut() as *mut c_void;
        for (slot, &(ct, m)) in ins.iter().enumerate() {
            let mut a = self.kargs(op::GATHER, NB);
            a.ct = ct;
            a.i0 = m as i32;
            a.i1 = slot as i32;
            a.x = stage;
            Self::launch(ctx, s, a)?;
        }
        ntt::transform_rows_prefix(ctx, s, Dir::Inverse, &mut self.work.stage, &self.rows, 8 * cnt)?;
        // a = part 1 of each slot: digit j at + j NB, slot q at + 8 q NB
        let src = (self.work.stage.dptr() as *const W).wrapping_add(4 * NB) as *const c_void;
        let fwd = self.keys.fwd.dptr();
        self.key_switch(ctx, s, NB, cnt, src, NB, 8 * NB, true, fwd, 0)?;
        let mut a = self.kargs(op::RING_IN, NB);
        a.y = self.work.stage.vptr();
        a.z = self.work.res.vptr();
        a.x = x as *mut c_void;
        a.i0 = c0 as i32;
        a.i1 = cnt as i32;
        Self::launch(ctx, s, a)
    }

    /// Columns 8 c0 .. of `x` -> output ciphertexts `c0 .. c0 + outs.len()` (FIDESlib ringOut; all 16 with `c0` = 0).
    fn ring_out<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, x: *const u32, outs: &[(PolyRef, usize)], c0: usize) -> Result<()> {
        let cnt = outs.len();
        if cnt == 0 || c0 + cnt > NBIG {
            return Err(shape("ccmm ring out", format!("ciphertexts {c0} .. {} of a {NBIG}-ciphertext matrix", c0 + cnt)));
        }
        let mut a = self.kargs(op::COMPOSE_A, NB);
        a.y = x as *const c_void;
        a.x = self.work.biga.vptr_mut();
        a.i0 = c0 as i32;
        a.i1 = cnt as i32;
        Self::launch(ctx, s, a)?;
        let src = self.work.biga.vptr();
        let back = self.keys.back.dptr();
        self.key_switch(ctx, s, NB, cnt, src, cnt * NB, NB, false, back, 0)?;
        let mut a = self.kargs(op::RING_OUT, NB);
        a.y = x as *const c_void;
        a.z = self.work.res.vptr();
        a.x = self.work.stage.vptr_mut();
        a.i0 = c0 as i32;
        a.i1 = cnt as i32;
        Self::launch(ctx, s, a)?;
        ntt::transform_rows_prefix(ctx, s, Dir::Forward, &mut self.work.stage, &self.rows, 8 * cnt)?;
        for (slot, &(ct, m)) in outs.iter().enumerate() {
            let mut a = self.kargs(op::SCATTER, NB);
            a.ct = ct;
            a.i0 = m as i32;
            a.i1 = slot as i32;
            a.y = self.work.stage.vptr();
            Self::launch(ctx, s, a)?;
        }
        Ok(())
    }

    fn cols_op<M: DevStrategy<W>>(&self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, opc: i32, y: *const u32, x: *mut u32, i0: i32, i1: i32, i2: i32) -> Result<()> {
        let mut a = self.kargs(opc, NS);
        a.y = y as *const c_void;
        a.x = x as *mut c_void;
        a.i0 = i0;
        a.i1 = i1;
        a.i2 = i2;
        Self::launch(ctx, s, a)
    }

    /// TWEAK (CKL Alg. 2) of `inp` using `tmp`; returns the buffer holding the result (one of the two).
    fn tweak<M: DevStrategy<W>>(&self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, inp: *mut u32, tmp: *mut u32, sgn: i32) -> Result<*mut u32> {
        self.cols_op(ctx, s, op::BITREV, inp, tmp, 0, 0, 0)?;
        let (mut a, mut b) = (tmp, inp);
        let mut h = 1;
        while h < D {
            self.cols_op(ctx, s, op::TWEAK, a, b, h as i32, (KY * D / (2 * h)) as i32, sgn)?;
            std::mem::swap(&mut a, &mut b);
            h *= 2;
        }
        Ok(a)
    }

    /// CMT (CKL Alg. 3) in place on the columns `x`: column-wise M -> column-wise M^T.
    fn cmt<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, x: *mut u32) -> Result<()> {
        let (t1, t2) = (self.work.t1.dptr_mut(), self.work.t2.dptr_mut());
        self.cols_op(ctx, s, op::MONO, x, t1, 1, 0, 0)?;
        let r = self.tweak(ctx, s, t1, t2, 1)?;
        let o = if r == t1 { t2 } else { t1 };
        let mut a = self.kargs(op::SCRAMBLE, NS);
        a.y = r as *const c_void;
        a.x = o as *mut c_void;
        a.z = self.keys.tstar.vptr();
        a.c[..NQ].copy_from_slice(&self.keys.dinv);
        Self::launch(ctx, s, a)?;
        let per = D * NS;
        let autos = self.keys.autos.dptr();
        self.key_switch(ctx, s, NS, D, o.wrapping_add(NQ * per) as *const c_void, per, NS, false, autos, key_words(NS))?;
        let mut a = self.kargs(op::KS_APPLY, NS);
        a.x = o as *mut c_void;
        a.y = self.work.res.vptr();
        a.s[0] = per as i64;
        Self::launch(ctx, s, a)?;
        let r2 = self.tweak(ctx, s, o, r, -1)?;
        self.cols_op(ctx, s, op::MONO, r2, x, -1, 0, 0)
    }

    /// The four PPMMs of M (columns `xm`) and M'^T (columns `xmp`) -> Cn [4][2D][2D][KY] (FIDESlib ppmm). `half` =
    /// Some(h): only the rows r of Cn in h D .. (h + 1) D (the left operand's part h: 0 = b, 1 = a), the rest unset.
    fn ppmm<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, xm: *const u32, xmp: *const u32, half: Option<usize>) -> Result<()> {
        let gh = match half {
            Some(h) if h < 2 => Some((h, self.gemm_half.as_ref().ok_or_else(|| shape("ccmm ppmm", "half rows without the half plan (enable_split)"))?)),
            Some(h) => return Err(shape("ccmm ppmm", format!("half {h}"))),
            None => None,
        };
        let (boff, coff) = gh.map_or((0, 0), |(h, _)| (h * D * D, h * D * 2 * D));
        let e = self.work.e.dptr_mut();
        for (src, off) in [(xm, 0), (xmp, COLS)] {
            let mut a = self.kargs(op::GATHER_U, 0);
            a.y = src as *const c_void;
            a.x = e.wrapping_add(off) as *mut c_void;
            Self::launch(ctx, s, a)?;
        }
        let ti = self.keys.table(KY)?;
        let mut a = self.kargs(op::NTT, KY);
        a.count = (2 * COLS / KY) as i32;
        a.i0 = (D * D) as i32;
        a.i1 = NQ as i32;
        a.x = e as *mut c_void;
        a.tw = self.keys.tw[ti].vptr();
        Self::launch(ctx, s, a)?;
        let (xd, yd) = (self.work.xd.dptr_mut() as *mut i8, self.work.yd.dptr_mut() as *mut i8);
        let g = self.work.g.dptr_mut();
        for pi in 0..NQ {
            for (src, dst) in [(e as *const u32, xd), (e.wrapping_add(COLS) as *const u32, yd)] {
                let mut a = self.kargs(op::DIGITS, 0);
                a.i0 = pi as i32;
                a.y = src as *const c_void;
                a.x = dst as *mut c_void;
                Self::launch(ctx, s, a)?;
            }
            let mut seen = [false; EXPS];
            for da in 0..4 {
                for db in 0..4 {
                    let ex = da + db;
                    // SAFETY: A = right digit plane db, B = left digit plane da (PER bytes each, the batched operands),
                    // C = exponent ex's accumulator (GPER int32), inside the work buffers
                    // (half: B = the left plane's rows h D .., C = the accumulator's rows h D ..)
                    unsafe {
                        let gm = gh.map_or(&self.gemm, |(_, x)| x);
                        gm.run(s, yd.wrapping_add(db * PER) as *const i8, xd.wrapping_add(da * PER + boff) as *const i8, g.wrapping_add(ex * GPER + coff), seen[ex], &mut self.work.gws)?;
                    }
                    seen[ex] = true;
                }
            }
            let mut a = self.kargs(op::RECOMBINE, 0);
            a.i0 = pi as i32;
            if let Some((h, _)) = gh {
                a.i1 = (h * D) as i32;
                a.i2 = D as i32;
            }
            a.y = g as *const c_void;
            a.x = self.work.cn.vptr_mut();
            Self::launch(ctx, s, a)?;
        }
        if let Some((h, _)) = gh {
            // per prime: its rows h D .. (h + 1) D of Cn, one block of D 2D polys of KY
            for pi in 0..NQ {
                let mut a = self.kargs(op::INTT, KY);
                a.p[0] = self.keys.primes[pi];
                a.mu[0] = self.keys.mu[pi];
                a.c[0] = self.keys.ninv[ti][pi];
                a.count = (D * 2 * D) as i32;
                a.i0 = (D * 2 * D) as i32;
                a.i1 = 1;
                a.x = self.work.cn.dptr_mut().wrapping_add((pi * 2 * D + h * D) * 2 * D * KY) as *mut c_void;
                a.tw = (self.keys.itw[ti].dptr() as *const u32).wrapping_add(pi * KY) as *const c_void;
                Self::launch(ctx, s, a)?;
            }
            return Ok(());
        }
        let mut a = self.kargs(op::INTT, KY);
        a.count = (NQ * GPER / KY) as i32;
        a.i0 = (4 * D * D) as i32;
        a.i1 = NQ as i32;
        a.x = self.work.cn.vptr_mut();
        a.tw = self.keys.itw[ti].vptr();
        a.c[..NP].copy_from_slice(&self.keys.ninv[ti]);
        Self::launch(ctx, s, a)
    }

    /// Rows `off .. off + D` of Cn -> the column set `dst` (FIDESlib ppmm's scatter: 0 = the M b-part rows, D = a).
    fn scatter_rows<M: DevStrategy<W>>(&self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, off: usize, dst: *mut u32) -> Result<()> {
        let mut ar = self.kargs(op::SCATTER_ROWS, 0);
        ar.i0 = off as i32;
        ar.y = self.work.cn.vptr();
        ar.x = dst as *mut c_void;
        Self::launch(ctx, s, ar)
    }

    /// Relinearize the CMT'd x1 (its a part times s'^2): res = KS of x1's a part under `rlk`.
    fn relin<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, x1: *const u32) -> Result<()> {
        let per = D * NS;
        let rlk = self.keys.rlk.dptr();
        self.key_switch(ctx, s, NS, D, x1.wrapping_add(NQ * per) as *const c_void, per, NS, false, rlk, 0)
    }

    /// `relin` of x1's columns c0 .. c0 + n folded into x1 (a split product): x1[0] += res[0], x1[1] = res[1] over those
    /// columns (res [2][4][n][NS] of the n-poly key switch), so that xm + x1 is the relinearized sum there.
    fn relin_fold<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, c0: usize, n: usize) -> Result<()> {
        let per = D * NS;
        let rlk = self.keys.rlk.dptr();
        let x1 = self.work.x1.dptr();
        self.key_switch(ctx, s, NS, n, x1.wrapping_add(NQ * per + c0 * NS) as *const c_void, per, NS, false, rlk, 0)?;
        for i in 0..NQ {
            let (xo, ro) = ((i * D + c0) * NS, i * n * NS);
            // one prime's n NS words: the ADD over 4 "primes" of n NS / 4 words, all p_i
            let mut a = self.kargs(op::ADD, 0);
            a.p[..NQ].fill(self.keys.primes[i]);
            a.x = self.work.x1.dptr_mut().wrapping_add(xo) as *mut c_void;
            a.y = self.work.res.dptr().wrapping_add(ro) as *const c_void;
            a.s[0] = (n * NS / NQ) as i64;
            Self::launch(ctx, s, a)?;
            self.work.x1.copy_from_at(s, NQ * per + xo, &self.work.res, (NQ + i) * n * NS, n * NS)?;
        }
        Ok(())
    }

    fn add<M: DevStrategy<W>>(&self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, x: *mut u32, y: *const u32) -> Result<()> {
        let mut a = self.kargs(op::ADD, 0);
        a.x = x as *mut c_void;
        a.y = y as *const c_void;
        a.s[0] = (D * NS) as i64;
        Self::launch(ctx, s, a)
    }

    fn mul_members<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, a: &[(PolyRef, usize)], b: &[(PolyRef, usize)], outs: &[(PolyRef, usize)], b_transposed: bool, ev: Option<[&Event; 6]>) -> Result<()> {
        let rec = |i: usize| -> Result<()> {
            if let Some(e) = ev {
                e[i].record(s)?;
            }
            Ok(())
        };
        rec(0)?;
        let (xm, xmp, x1) = (self.work.xm.dptr_mut(), self.work.xmp.dptr_mut(), self.work.x1.dptr_mut());
        self.ring_in(ctx, s, a, 0, xm)?;
        self.ring_in(ctx, s, b, 0, xmp)?;
        rec(1)?;
        if !b_transposed {
            self.cmt(ctx, s, xmp)?;
        }
        rec(2)?;
        self.ppmm(ctx, s, xm, xmp, None)?;
        rec(3)?;
        self.scatter_rows(ctx, s, 0, xm)?;
        self.scatter_rows(ctx, s, D, x1)?;
        self.cmt(ctx, s, xm)?;
        self.cmt(ctx, s, x1)?;
        self.relin(ctx, s, x1)?;
        let per = D * NS;
        let res = self.work.res.dptr();
        self.add(ctx, s, xm, res.wrapping_add(NQ * per))?;
        self.add(ctx, s, xm.wrapping_add(NQ * per), x1)?;
        self.add(ctx, s, xm.wrapping_add(NQ * per), res)?;
        rec(4)?;
        self.ring_out(ctx, s, xm, outs, 0)?;
        rec(5)
    }

    fn transpose_members<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, ins: &[(PolyRef, usize)], outs: &[(PolyRef, usize)], ev: Option<[&Event; 6]>) -> Result<()> {
        let rec = |i: usize| -> Result<()> {
            if let Some(e) = ev {
                e[i].record(s)?;
            }
            Ok(())
        };
        rec(0)?;
        let xm = self.work.xm.dptr_mut();
        self.ring_in(ctx, s, ins, 0, xm)?;
        rec(1)?;
        self.cmt(ctx, s, xm)?;
        rec(2)?;
        rec(3)?;
        rec(4)?;
        self.ring_out(ctx, s, xm, outs, 0)?;
        rec(5)
    }

    /// The 16 member views of a matrix operand (every ciphertext at the CCMM split: limbs 0..3, level checked).
    fn members<M: DevStrategy<W>>(ctx: &Context<'_, W, M>, what: &str, cts: &[Ct<'_, W>]) -> Result<Members> {
        let mut v = Vec::new();
        for x in cts {
            if ctx.lay != x.buf.lay || ctx.top_of(x.meta.level)? != NQ - 1 {
                return Err(shape("ccmm", format!("{what}: ciphertext at level {} (top {:?}); the CCMM holds limbs 0..3", x.meta.level, ctx.top_of(x.meta.level).ok())));
            }
            let r = x.buf.r();
            for m in 0..x.k() {
                v.push((r, m));
            }
        }
        if v.len() != NBIG {
            return Err(shape("ccmm", format!("{what}: {} ciphertexts, a matrix is {NBIG}", v.len())));
        }
        Ok(v)
    }

    fn write_members(cts: &mut [Ct<'_, W>]) -> Members {
        let mut v = Vec::new();
        for x in cts.iter_mut() {
            let r = x.buf.w();
            for m in 0..x.k() {
                v.push((r, m));
            }
        }
        v
    }

    fn out_meta(outs: &mut [Ct<'_, W>], meta: &CtMeta) {
        for x in outs.iter_mut() {
            x.meta = meta.clone();
        }
    }

    /// Batch CCMM (CKL Alg. 4): `outs` = M M' for column-wise SinC encryptions M (`a`) and M' (`b`; M'^T when
    /// `b_transposed`), 16 ciphertexts each (any batching), every one at the CCMM split (limbs 0..3). Outputs get
    /// `a`'s metadata (the caller restamps them: `leaf::set_scale`). bts.ccmm_mul.
    #[allow(clippy::too_many_arguments)]
    pub fn mul<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, a: &[Ct<'_, W>], b: &[Ct<'_, W>], outs: &mut [Ct<'_, W>], b_transposed: bool, ev: Option<[&Event; 6]>) -> Result<()> {
        let ma = Self::members(ctx, "a", a)?;
        let mb = Self::members(ctx, "b", b)?;
        let n_out: usize = outs.iter().map(|x| x.k()).sum();
        if n_out != NBIG || outs.iter().any(|x| x.buf.lay != ctx.lay) {
            return Err(shape("ccmm", format!("outs: {n_out} ciphertexts, a matrix is {NBIG}")));
        }
        let meta = common_meta(a)?;
        let mo = Self::write_members(outs);
        self.mul_members(ctx, s, &ma, &mb, &mo, b_transposed, ev)?;
        Self::out_meta(outs, &meta);
        Ok(())
    }

    /// `mul` writing the product over `a` (perseus `ccmm_mul(da, db, da, ..)`): `a` is read before it is written.
    pub fn mul_inplace<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, a: &mut [Ct<'_, W>], b: &[Ct<'_, W>], b_transposed: bool, ev: Option<[&Event; 6]>) -> Result<()> {
        let ma = Self::members(ctx, "a", a)?;
        let mb = Self::members(ctx, "b", b)?;
        let meta = common_meta(a)?;
        let mo = Self::write_members(a);
        self.mul_members(ctx, s, &ma, &mb, &mo, b_transposed, ev)?;
        Self::out_meta(a, &meta);
        Ok(())
    }

    /// CMT of a whole matrix (column-wise M -> M^T), bts.ccmm_transpose. Outputs get `ins`' metadata.
    pub fn transpose<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, ins: &[Ct<'_, W>], outs: &mut [Ct<'_, W>], ev: Option<[&Event; 6]>) -> Result<()> {
        let mi = Self::members(ctx, "ins", ins)?;
        let n_out: usize = outs.iter().map(|x| x.k()).sum();
        if n_out != NBIG || outs.iter().any(|x| x.buf.lay != ctx.lay) {
            return Err(shape("ccmm", format!("outs: {n_out} ciphertexts, a matrix is {NBIG}")));
        }
        let meta = common_meta(ins)?;
        let mo = Self::write_members(outs);
        self.transpose_members(ctx, s, &mi, &mo, ev)?;
        Self::out_meta(outs, &meta);
        Ok(())
    }

    /// `transpose` in place.
    pub fn transpose_inplace<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, x: &mut [Ct<'_, W>], ev: Option<[&Event; 6]>) -> Result<()> {
        let mi = Self::members(ctx, "x", x)?;
        let mo = Self::write_members(x);
        self.transpose_members(ctx, s, &mi, &mo, ev)
    }
}

// ── the product split across two devices (PC_CCMM_SPLIT) ─────────────────────────────────────────────────────────
//
// Every output ciphertext depends on every input ciphertext (the PPMM sums over all 128 columns of both operands, the
// two CMTs mix all columns), so no device can compute output ciphertexts from a share of the inputs. What splits:
// the ring switch in (per input ciphertext), the PPMM by the left operand's part (rows 0 .. D of Cn = b, D .. 2D = a:
// the two halves feed the independent CMT(xm) and CMT(x1)), the relinearization (on x1's device) and the ring switch
// out (per output ciphertext). Here = ciphertexts 0 .. 8 of each operand and of the output, Cn rows 0 .. D, CMT(xm);
// there = ciphertexts 8 .. 16, rows D .. 2D, CMT(x1) + relinearization. Both run the right operand's CMT. Two
// exchanges of columns (each device pulls on its copy stream, chunk by chunk as they are made): after the ring switch
// in (the right operand's other 64 columns, the left part the device's rows need), and before the ring switch out
// (here: x1 with its relinearization folded in, columns 0 .. 64, made first; there: xm of columns 64 .. 128).
// Bit-exact: the same integer ops, the additions regrouped mod p.

/// Events one device records in a split product (`SplitSide::ev`, on that device).
pub const SPLIT_EVENTS: usize = 10;
const EV_START: usize = 0;
/// the ring switch in of chunk c of the right (`EV_XR + c`) / left (`EV_XL + c`) operand
const EV_XR: usize = 1;
const EV_XL: usize = 3;
const EV_RXR: usize = 5;
const EV_RXL: usize = 6;
const EV_PP: usize = 7;
const EV_POST: usize = 8;
const EV_R2: usize = 9;
/// Ring-switch chunks of a device's 8 ciphertexts per operand (their columns are pulled as each one is done).
const SPLIT_CHUNKS: usize = 2;

/// One device's share of a split product: its engine (`enable_split`), context, compute stream, its copy stream
/// (pulls the other device's columns into its engine) and `SPLIT_EVENTS` events of its own.
pub struct SplitSide<'s, 'a, 'lt, 'c, W: DevWord, M: DevStrategy<W>> {
    pub eng: &'s mut Ccmm<'a, 'lt, W>,
    pub ctx: &'s Context<'c, W, M>,
    pub dev: &'s Device,
    pub s: &'s Stream<Compute>,
    pub cp: &'s Stream<Compute>,
    pub ev: &'s [Event],
    /// PC_CCMM_PHASES: 6 timing events of this device (start, ring in, CMT, PPMM, CMT(s) + relinearization, ring out)
    pub ph: Option<&'s [Event]>,
}

/// One device's operands and outputs of a split product: its 8 ciphertexts of `a`, of `b` and of the output.
pub struct SplitOps<'m> {
    pub a: &'m [(PolyRef, usize)],
    pub b: &'m [(PolyRef, usize)],
    pub out: &'m [(PolyRef, usize)],
}

/// The member views of `cts` (`n` ciphertexts in all, each at the CCMM split).
pub fn split_members<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, what: &str, cts: &[Ct<'_, W>], n: usize) -> Result<Members> {
    let mut v = Vec::new();
    for x in cts {
        if ctx.lay != x.buf.lay || ctx.top_of(x.meta.level)? != NQ - 1 {
            return Err(shape("ccmm split", format!("{what}: ciphertext at level {} (top {:?}); the CCMM holds limbs 0..3", x.meta.level, ctx.top_of(x.meta.level).ok())));
        }
        let r = x.buf.r();
        for m in 0..x.k() {
            v.push((r, m));
        }
    }
    if v.len() != n {
        return Err(shape("ccmm split", format!("{what}: {} ciphertexts, a share is {n}", v.len())));
    }
    Ok(v)
}

/// The writable member views of `cts` (a share's outputs).
pub fn split_out_members<W: DevWord>(cts: &mut [Ct<'_, W>]) -> Members {
    let mut v = Vec::new();
    for x in cts.iter_mut() {
        let r = x.buf.w();
        for m in 0..x.k() {
            v.push((r, m));
        }
    }
    v
}

/// Parts `parts` x primes of columns `cols` of the column set `src` (memory of `sd`) into the same words of `dst`
/// (memory of `dd`), on `s`.
///
/// # Safety
/// As `DBuf::copy_from_peer`: `s` waits for the writes of `src` and nothing writes `src` / reads `dst` before an event
/// recorded on `s` after the copy.
unsafe fn pull_cols(s: &Stream<Compute>, dst: &mut DBuf<'_, Compute, u32>, dd: &Device, src: &DBuf<'_, Compute, u32>, sd: &Device, parts: &[usize], cols: std::ops::Range<usize>) -> Result<()> {
    for &part in parts {
        for i in 0..NQ {
            let off = ((part * NQ + i) * D + cols.start) * NS;
            if dd.is_host() {
                // the host device (both shares on it: the host twin's check)
                dst.copy_from_at(s, off, src, off, cols.len() * NS)?;
            } else {
                dst.copy_from_peer(s, dd, off, src, sd, off, cols.len() * NS)?;
            }
        }
    }
    Ok(())
}

fn rec_ph(ph: Option<&[Event]>, s: &Stream<Compute>, i: usize) -> Result<()> {
    if let Some(e) = ph {
        e[i].record(s)?;
    }
    Ok(())
}

/// A split product's last step on one device (current): after both column pulls of the second exchange, xm += x1 (the
/// relinearization folded in, `relin_fold`) and the ring switch out of its output ciphertexts `c0 ..`.
fn finish_side<W: DevWord, M: DevStrategy<W>>(x: &mut SplitSide<'_, '_, '_, '_, W, M>, o: &SplitOps<'_>, c0: usize, r2: &[&Event]) -> Result<()> {
    for e in r2 {
        x.s.wait(e)?;
    }
    let per = D * NS;
    let e = &mut *x.eng;
    let (xm, x1) = (e.work.xm.dptr_mut(), e.work.x1.dptr());
    e.add(x.ctx, x.s, xm, x1.wrapping_add(NQ * per))?;
    e.add(x.ctx, x.s, xm.wrapping_add(NQ * per), x1)?;
    e.ring_out(x.ctx, x.s, xm, o.out, c0)?;
    rec_ph(x.ph, x.s, 5)
}

/// The product M M' (`b_transposed`: M'^T given) split across two devices `h` (here) and `t` (there), in place or not
/// as the member lists say: `h` takes ciphertexts 0 .. 8 of `a`, `b` and the output (`oh`), `t` 8 .. 16 (`ot`).
/// Everything is issued asynchronously; `h`'s device is current on return. The outputs are ready on each device's
/// compute stream; `t`'s inputs are read after everything issued on `t.s` so far, `h`'s after `h.s`'s.
pub fn mul_split<W: DevWord, M: DevStrategy<W>>(h: &mut SplitSide<'_, '_, '_, '_, W, M>, t: &mut SplitSide<'_, '_, '_, '_, W, M>, oh: &SplitOps<'_>, ot: &SplitOps<'_>, b_transposed: bool) -> Result<()> {
    const HB: usize = NBIG / 2;
    const HC: usize = D / 2;
    for (o, who) in [(oh, "here"), (ot, "there")] {
        if o.a.len() != HB || o.b.len() != HB || o.out.len() != HB {
            return Err(shape("ccmm split", format!("{who}: {} / {} / {} ciphertexts, a share is {HB}", o.a.len(), o.b.len(), o.out.len())));
        }
    }
    if h.ev.len() < SPLIT_EVENTS || t.ev.len() < SPLIT_EVENTS || h.eng.gemm_half.is_none() || t.eng.gemm_half.is_none() {
        return Err(shape("ccmm split", "both engines need enable_split and SPLIT_EVENTS events"));
    }
    let (hd, td) = (h.dev, t.dev);
    let (hev, tev): (&[Event], &[Event]) = (h.ev, t.ev);
    // a device's ciphertexts per chunk, and the columns of chunk c of the device whose first ciphertext is c0
    const CC: usize = HB / SPLIT_CHUNKS;
    let chunk_cols = |c0: usize, c: usize| (c0 + c * CC) * (D / NBIG)..(c0 + (c + 1) * CC) * (D / NBIG);
    let res = (|| -> Result<()> {
        // 1., 2. both: the ring switch in of the device's ciphertexts in chunks (the right operand first: both CMTs
        // wait for it), an event after each
        td.make_current()?;
        {
            let e = &mut *t.eng;
            rec_ph(t.ph, t.s, 0)?;
            t.ev[EV_START].record(t.s)?;
            let (xm, xmp) = (e.work.xm.dptr_mut(), e.work.xmp.dptr_mut());
            for (ops, x, ev) in [(ot.b, xmp, EV_XR), (ot.a, xm, EV_XL)] {
                for c in 0..SPLIT_CHUNKS {
                    e.ring_in(t.ctx, t.s, &ops[c * CC..(c + 1) * CC], HB + c * CC, x)?;
                    t.ev[ev + c].record(t.s)?;
                }
            }
            rec_ph(t.ph, t.s, 1)?;
        }
        hd.make_current()?;
        {
            let e = &mut *h.eng;
            rec_ph(h.ph, h.s, 0)?;
            h.ev[EV_START].record(h.s)?;
            let (xm, xmp) = (e.work.xm.dptr_mut(), e.work.xmp.dptr_mut());
            for (ops, x, ev) in [(oh.b, xmp, EV_XR), (oh.a, xm, EV_XL)] {
                for c in 0..SPLIT_CHUNKS {
                    e.ring_in(h.ctx, h.s, &ops[c * CC..(c + 1) * CC], c * CC, x)?;
                    h.ev[ev + c].record(h.s)?;
                }
            }
            rec_ph(h.ph, h.s, 1)?;
        }
        // 3. here pulls there's right columns (both parts) and its left columns' b part, chunk by chunk (after there's
        // last pull of here's columns, in the previous product)
        h.cp.wait(&h.ev[EV_START])?;
        h.cp.wait(&t.ev[EV_R2])?;
        for c in 0..SPLIT_CHUNKS {
            h.cp.wait(&t.ev[EV_XR + c])?;
            // SAFETY: the copy stream waits for the events after the source's writes and after this device's last reads
            // of the destination (START); the sources are not written before the waits below (RXR / RXL) there
            unsafe { pull_cols(h.cp, &mut h.eng.work.xmp, hd, &t.eng.work.xmp, td, &[0, 1], chunk_cols(HB, c))? };
        }
        h.ev[EV_RXR].record(h.cp)?;
        for c in 0..SPLIT_CHUNKS {
            h.cp.wait(&t.ev[EV_XL + c])?;
            // SAFETY: as above
            unsafe { pull_cols(h.cp, &mut h.eng.work.xm, hd, &t.eng.work.xm, td, &[0], chunk_cols(HB, c))? };
        }
        h.ev[EV_RXL].record(h.cp)?;
        // 4. there pulls here's right columns and its left columns' a part
        td.make_current()?;
        t.cp.wait(&t.ev[EV_START])?;
        t.cp.wait(&h.ev[EV_R2])?;
        for c in 0..SPLIT_CHUNKS {
            t.cp.wait(&h.ev[EV_XR + c])?;
            // SAFETY: as above
            unsafe { pull_cols(t.cp, &mut t.eng.work.xmp, td, &h.eng.work.xmp, hd, &[0, 1], chunk_cols(0, c))? };
        }
        t.ev[EV_RXR].record(t.cp)?;
        for c in 0..SPLIT_CHUNKS {
            t.cp.wait(&h.ev[EV_XL + c])?;
            // SAFETY: as above
            unsafe { pull_cols(t.cp, &mut t.eng.work.xm, td, &h.eng.work.xm, hd, &[1], chunk_cols(0, c))? };
        }
        t.ev[EV_RXL].record(t.cp)?;
        // 5. here: CMT of the right operand (after both pulls of it), PPMM rows 0 .. D, CMT(xm)
        hd.make_current()?;
        {
            let e = &mut *h.eng;
            let (xm, xmp) = (e.work.xm.dptr_mut(), e.work.xmp.dptr_mut());
            h.s.wait(&h.ev[EV_RXR])?;
            h.s.wait(&t.ev[EV_RXR])?;
            if !b_transposed {
                e.cmt(h.ctx, h.s, xmp)?;
            }
            rec_ph(h.ph, h.s, 2)?;
            h.s.wait(&h.ev[EV_RXL])?;
            e.ppmm(h.ctx, h.s, xm, xmp, Some(0))?;
            h.ev[EV_PP].record(h.s)?;
            rec_ph(h.ph, h.s, 3)?;
            // there has pulled xm's a part
            h.s.wait(&t.ev[EV_RXL])?;
            e.scatter_rows(h.ctx, h.s, 0, xm)?;
            e.cmt(h.ctx, h.s, xm)?;
            h.ev[EV_POST].record(h.s)?;
            rec_ph(h.ph, h.s, 4)?;
        }
        // 6. there: CMT, PPMM rows D .. 2D, CMT(x1), its relinearization folded into x1 (x1 = (x1.b + res[0], res[1])):
        // here's columns first (the pull of them overlaps the rest)
        td.make_current()?;
        {
            let e = &mut *t.eng;
            let (xm, xmp, x1) = (e.work.xm.dptr_mut(), e.work.xmp.dptr_mut(), e.work.x1.dptr_mut());
            t.s.wait(&t.ev[EV_RXR])?;
            t.s.wait(&h.ev[EV_RXR])?;
            if !b_transposed {
                e.cmt(t.ctx, t.s, xmp)?;
            }
            rec_ph(t.ph, t.s, 2)?;
            t.s.wait(&t.ev[EV_RXL])?;
            e.ppmm(t.ctx, t.s, xm, xmp, Some(1))?;
            t.ev[EV_PP].record(t.s)?;
            rec_ph(t.ph, t.s, 3)?;
            e.scatter_rows(t.ctx, t.s, D, x1)?;
            e.cmt(t.ctx, t.s, x1)?;
            e.relin_fold(t.ctx, t.s, 0, HC)?;
            t.ev[EV_POST].record(t.s)?;
            e.relin_fold(t.ctx, t.s, HC, HC)?;
            rec_ph(t.ph, t.s, 4)?;
        }
        // 7. here pulls x1 of columns 0 .. 64
        hd.make_current()?;
        h.cp.wait(&h.ev[EV_POST])?;
        h.cp.wait(&t.ev[EV_POST])?;
        // SAFETY: as above (here's x1 is free after its CMT(xm); there writes its x1 columns 0 .. 64 again only in its
        // next product, after R2 of here)
        unsafe { pull_cols(h.cp, &mut h.eng.work.x1, hd, &t.eng.work.x1, td, &[0, 1], 0..HC)? };
        h.ev[EV_R2].record(h.cp)?;
        // 8. there pulls xm of columns 64 .. 128 (its xm is free after its PPMM and here's pull of it)
        td.make_current()?;
        t.cp.wait(&t.ev[EV_PP])?;
        t.cp.wait(&h.ev[EV_RXL])?;
        t.cp.wait(&h.ev[EV_POST])?;
        // SAFETY: as above (here's adds write xm after R2 of there)
        unsafe { pull_cols(t.cp, &mut t.eng.work.xm, td, &h.eng.work.xm, hd, &[0, 1], HC..D)? };
        t.ev[EV_R2].record(t.cp)?;
        // 9., 10. both: xm += x1, the ring switch out of the device's output ciphertexts, after its own pull (here also
        // after there's pull of its xm, which its additions overwrite; there's x1 columns here pulls stay untouched)
        hd.make_current()?;
        finish_side(h, oh, 0, &[&hev[EV_R2], &tev[EV_R2]])?;
        td.make_current()?;
        finish_side(t, ot, HB, &[&tev[EV_R2]])
    })();
    let back = hd.make_current();
    res?;
    back?;
    Ok(())
}

// ── SinCBoot (CKL Sec. 3.2 on the StC-first route) ───────────────────────────────────────────────────────────────

/// The refresh state of a CCMM's outputs: `btsBegin` on the left operand's metadata BEFORE its stc_high (ccmm.py
/// `_device`: `bts.begin(d, S, False, True)`), on the StC-first route.
pub fn sinc_state<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, route: &StcFirstRoute<'_, W, M>, a_meta: &CtMeta, r: &Route) -> Result<BtsState> {
    let st = route.begin_meta(ctx, a_meta, 1, r, false)?;
    if !st.stc_first {
        return Err(shape("ccmm", format!("the operand at level {} does not run StC-first on {r:?} (the dense route must be StC-first)", a_meta.level)));
    }
    Ok(st)
}

/// The split's scale (ccmm.py `prepare`: `noise_factor` after `stc_first_input` of a token at the operands'
/// level): runs the input stages on `scratch` (its contents are overwritten).
pub fn split_nf<'a, W: DevWord, M: DevStrategy<W>>(cx: &mut BtsCx<'_, 'a, W, M>, route: &StcFirstRoute<'a, W, M>, st: &BtsState, scratch: &mut Ct<'a, W>) -> Result<f64> {
    route.stc_first_input_cf(cx, st, std::slice::from_mut(scratch), false)?;
    Ok(scratch.meta.nf)
}

/// The CCMM operands' level, limbs 0..NQ-1 (`stc_high`'s output): n32 50, a lean chain 47.
pub fn operand_level(n_q: u32) -> u32 {
    n_q - NQ as u32
}

/// CCMM outputs (limbs 0..NQ-1) restamped as the StC-first split (degree 2, scale `nf_split`: perseus `set_scale`).
/// A lean chain's split is limbs 0..2 (q2 its single prime): q3 is divided out first, exactly (one prime's rescale),
/// which the calibrated `c_unit` takes in (its probes run this path).
pub fn enter_split<'a, W: DevWord, M: DevStrategy<W>>(cx: &mut BtsCx<'_, 'a, W, M>, route: &StcFirstRoute<'a, W, M>, cts: &mut [Ct<'a, W>], nf_split: f64) -> Result<()> {
    let ctx = cx.ctx;
    let split = route.split_top();
    for x in cts.iter_mut() {
        let top = x.top(ctx.shape)?;
        if top == split + 1 && route.thin() {
            cx.ops.rescaler.rescale_one(ctx, &cx.ex.compute, &mut x.buf, top)?;
            x.meta.level += 1;
        } else if top != split {
            return Err(shape("ccmm split", format!("output at limb {top}, the split is limb {split}")));
        }
        leaf::set_scale(x, nf_split, 2);
    }
    Ok(())
}

/// Output level of SinCBoot's stage-0 transform for a HalfBTS landing `m` (ccmm.py: `stc_high(d, S, d.level - 2 -
/// (2 if deg 2 else 0), 1.0, 6)`, limb indices with unit 2): one unit below, two when a rescale is pending.
pub fn stage0_level(m: &CtMeta, unit: u32) -> u32 {
    m.level + unit + if m.deg == 2 { unit } else { 0 }
}

/// The stc_high shape SinCBoot uses on a route for operands with metadata `a_meta` (StcHighSetup::shapes needs it
/// at setup): (output level, skip mask 6).
pub fn sinc_boot_shape<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, sh: &Shadow, route: &StcFirstRoute<'_, W, M>, a_meta: &CtMeta, r: &Route) -> Result<(u32, u32)> {
    let land = BtsRoute::landing(route, sh, a_meta, r)?;
    Ok((stage0_level(&land, ctx.chain.level_unit as u32), 6))
}

/// SinCBoot of CCMM outputs (ccmm.py `_device` after the product): each batch re-entered as the StC-first split
/// (degree 2, scale `nf_split`: perseus `set_scale`), HalfBTS (mod_raise .. stc_first_output) under `st`, then StC
/// stage 0 alone in the slot domain (stc_high skip mask 6, factor 1). Slots come back in the SinC layout.
pub fn sinc_boot<'a, W: DevWord, M: DevStrategy<W>>(cx: &mut BtsCx<'_, 'a, W, M>, route: &StcFirstRoute<'a, W, M>, st: &BtsState, cts: &mut [Ct<'a, W>], nf_split: f64) -> Result<()> {
    enter_split(cx, route, cts, nf_split)?;
    let mut call = BtsCall::resume(StagedRef::StcFirst(route), st.clone(), Phase::Split, CallOpts::default());
    call.at_split(cx, cts)?;
    call.run_to(cx, cts, Phase::Done)?;
    let m = common_meta(cts)?;
    let spec = StcHigh { slots: (cx.ctx.lay.n / 2) as u32, out_level: stage0_level(&m, cx.ctx.chain.level_unit as u32), factor: 1.0, skip_mask: 6, nf_ref: 0.0 };
    Staged::stc_high(route, cx, cts, &spec)
}

/// Perseus `bts.*` metadata / small leaf functions the CCMM callers use.
pub mod leaf {
    use super::*;
    use crate::bts::finish::Pow2;
    use crate::executor::Executor;
    use crate::ks::{ew, Ew, View};
    use crate::ops::Ops;

    /// Restamp scale and degree (no data change): bts.set_scale.
    pub fn set_scale<W: DevWord>(x: &mut Ct<'_, W>, nf: f64, deg: u8) {
        x.meta.nf = nf;
        x.meta.deg = deg;
    }
    /// bts.noise_factor.
    pub fn noise_factor<W: DevWord>(x: &Ct<'_, W>) -> f64 {
        x.meta.nf
    }
    /// bts.scale_noise_factor: NoiseFactor *= f.
    pub fn scale_noise_factor<W: DevWord>(x: &mut Ct<'_, W>, f: f64) {
        x.meta.nf *= f;
    }
    /// bts.set_slots.
    pub fn set_slots<W: DevWord>(x: &mut Ct<'_, W>, slots: u32) {
        x.meta.slots = slots;
    }
    /// bts.clone: `dst` = a copy of `src` (live rows of both parts, metadata).
    pub fn clone_into<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, s: &Stream<Compute>, src: &Ct<'_, W>, dst: &mut Ct<'_, W>) -> Result<()> {
        if dst.k() != src.k() || dst.buf.lay != src.buf.lay {
            return Err(shape("clone", format!("{} ciphertexts into {}", src.k(), dst.k())));
        }
        let top = src.top(ctx.shape)?;
        ew(ctx, s, Ew::Copy, View::of(&dst.buf, 0), View::of(&src.buf, 0), None, ctx.lay.rs_q(top), src.k(), 2)?;
        dst.meta = src.meta.clone();
        Ok(())
    }
    /// bts.mult_int: exact product by a positive integer, no level (the callers pass powers of two: `1 << shift`).
    pub fn mult_int<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, s: &Stream<Compute>, pow2: &Pow2<'_, W>, x: &mut Ct<'_, W>, k: u64) -> Result<()> {
        let e = crate::bts::finish::pow2_exp(k)?;
        let top = x.top(ctx.shape)?;
        pow2.mult(ctx, s, View::of(&x.buf, 0), x.k(), top, e)
    }
    /// bts.wsum_raw: out = sum_i ws\[i\] ts\[i\] in one kernel (degree-2 inputs are rescaled in place, every input taken
    /// to the shallowest level among them; out's previous value is overwritten). Unrecorded raw path.
    #[allow(clippy::too_many_arguments)]
    pub fn wsum_raw<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, ex: &mut Executor<'_, W>, ops: &mut Ops<'_, W>, sh: &Shadow, out: &mut Ct<'_, W>, ts: &mut [&mut Ct<'_, W>], ws: &[f64]) -> Result<()> {
        if ts.is_empty() || ts.len() != ws.len() {
            return Err(shape("wsum_raw", "ts and ws must be non-empty and of equal length"));
        }
        for t in ts.iter_mut() {
            if t.meta.deg == 2 {
                ops.rescale(ctx, ex, sh, std::slice::from_mut(&mut **t))?;
            }
        }
        // the shallowest input first: eval_linear_wsum works at its first input's top (richer inputs scale onto it)
        let first = (0..ts.len()).max_by_key(|&i| ts[i].meta.level).unwrap_or(0);
        let mut order: Vec<usize> = vec![first];
        order.extend((0..ts.len()).filter(|&i| i != first));
        let a: Vec<&Ct<'_, W>> = order.iter().map(|&i| &*ts[i]).collect();
        let w: Vec<f64> = order.iter().map(|&i| ws[i]).collect();
        ops.eval_linear_wsum(ctx, ex, out, &a, &w)
    }
}
