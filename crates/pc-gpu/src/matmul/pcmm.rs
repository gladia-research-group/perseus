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

/// A second device for a linear's finish groups (`linear`'s `off`): the last half of the groups is refreshed there
/// while this device refreshes the first half.
pub trait GroupOffload<'a, W: DevWord, M: DevStrategy<W>> {
    /// Refresh group ciphertext `x` (this device's memory, at the split of `call`) to `Done` on the remote device,
    /// written back into `x`, ordered after everything issued on `here` so far.
    fn issue(&mut self, here: &Stream<Compute>, sh: &pc_shadow::Shadow, call: &BtsCall<'_, 'a, W, M>, x: &mut Ct<'a, W>) -> Result<()>;
    /// Order `here` after every issued group.
    fn join(&mut self, here: &Stream<Compute>) -> Result<()>;
    /// Every remote group refreshed here again from its saved input and compared word for word (host waits, eager
    /// only: `PC_REMOTE_CHECK`).
    fn check(&self) -> bool {
        false
    }
    /// How many of `total` finish groups (the last ones) go to the other device: by default the last half.
    fn far(&self, total: usize) -> usize {
        total / 2
    }
    /// How many of the linear's `n` inputs (the last ones) take their StC-first input stage on the other device
    /// (0: none, the default).
    fn stc_far(&self, _n: usize) -> usize {
        0
    }
    /// The input stage of `call` (at `Begun`, the linear's input call) for `xs` on the other device, the results back
    /// in `xs` at the split, ordered after everything issued on `here` so far (`join` orders `here` after it).
    fn stc_issue(&mut self, _here: &Stream<Compute>, _sh: &pc_shadow::Shadow, _call: &BtsCall<'_, 'a, W, M>, _xs: &mut [Ct<'a, W>]) -> Result<()> {
        Err(shape("pcmm offload", "this offload runs no input stages"))
    }
    /// Joint linear (`linear_joint`): whether the other device holds its own planes of linear `lin`'s far outputs.
    fn joint(&self, _lin: &str) -> bool {
        false
    }
    /// Joint: the input stage of `call` for inputs `first..first + xs.len()` (`xs`) on the other device, the results
    /// kept there for its product and copied back into `xs` (`join` orders `here` after it).
    fn joint_stc(&mut self, _here: &Stream<Compute>, _sh: &pc_shadow::Shadow, _call: &BtsCall<'_, 'a, W, M>, _first: usize, _xs: &mut [Ct<'a, W>]) -> Result<()> {
        Err(shape("pcmm offload", "this offload runs no joint linears"))
    }
    /// Joint: this device's inputs `0..xs.len()` at the split copied to the other device (after everything issued on
    /// `here` so far).
    fn joint_inputs(&mut self, _here: &Stream<Compute>, _xs: &[Ct<'a, W>]) -> Result<()> {
        Err(shape("pcmm offload", "this offload runs no joint linears"))
    }
    /// Joint: linear `lin`'s far outputs (sub-shape `sub`, numbered `0..sub.n_out` as the other device's planes) at
    /// scale `nf_out` on the other device's inputs, each group of `groups` (sub numbering) finished there at raise drop
    /// `drops[g]` from the input call `call` (at the split), the results copied back into `outs[g]` (`join` orders
    /// `here` after it; the caller applies 2^shift).
    #[allow(clippy::too_many_arguments)]
    fn joint_far(&mut self, _here: &Stream<Compute>, _sh: &pc_shadow::Shadow, _lin: &PcmmLinear, _sub: &PcmmShape, _call: &BtsCall<'_, 'a, W, M>, _nf_out: f64, _groups: &[Vec<usize>], _drops: &[u32], _outs: &mut [Ct<'a, W>]) -> Result<()> {
        Err(shape("pcmm offload", "this offload runs no joint linears"))
    }
    /// Row split (`linear_rows`, Sylph's reduction split): whether the other device holds its planes of linear `lin`'s
    /// last inputs (every output).
    fn rows(&self, _lin: &str) -> bool {
        false
    }
    /// Rows: the input stage of `call` for inputs `first..first + xs.len()` (`xs`) on the other device, kept there for
    /// its partial product (copied back into `xs` too when `back`: the remote check), ordered after everything issued
    /// on `here` so far.
    fn rows_stc(&mut self, _here: &Stream<Compute>, _sh: &pc_shadow::Shadow, _call: &BtsCall<'_, 'a, W, M>, _first: usize, _xs: &mut [Ct<'a, W>], _back: bool) -> Result<()> {
        Err(shape("pcmm offload", "this offload runs no row splits"))
    }
    /// Rows: the other device's partial product of its staged inputs (sub-shape `sub`: those inputs, every output) on
    /// its planes of `lin`, the partial's buffers returned (`Pcmm::partial_ptrs`, peer memory); `rows_wait` orders a
    /// stream of this device after it.
    fn rows_product(&mut self, _lin: &PcmmLinear, _sub: &PcmmShape) -> Result<(*const u32, *const u32)> {
        Err(shape("pcmm offload", "this offload runs no row splits"))
    }
    /// Rows: order `here` after the other device's partial product.
    fn rows_wait(&mut self, _here: &Stream<Compute>) -> Result<()> {
        Err(shape("pcmm offload", "this offload runs no row splits"))
    }
    /// Rows: on the other device, after everything issued on `here` so far, this device's partial (`here_partial`,
    /// peer memory) added into its own for its outputs `far_outs` (every-output numbering), their output stage at
    /// `meta` / `nf_out`, each group of `groups` (positions in `far_outs`) finished there at `drops[g]` from `call`, the
    /// results copied back into `outs[g]` (`join` orders `here` after it).
    #[allow(clippy::too_many_arguments)]
    fn rows_far(&mut self, _here: &Stream<Compute>, _sh: &pc_shadow::Shadow, _sub: &PcmmShape, _call: &BtsCall<'_, 'a, W, M>, _here_partial: (*const u32, *const u32), _far_outs: &[usize], _meta: &CtMeta, _nf_out: f64, _groups: &[Vec<usize>], _drops: &[u32], _outs: &mut [Ct<'a, W>]) -> Result<()> {
        Err(shape("pcmm offload", "this offload runs no row splits"))
    }
    /// One remote group checked.
    fn checked(&mut self) {}
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
        self.inputs_gemm(ctx, s, k, sh, planes, ins, ev)?;
        let os: Vec<(usize, PolyRef, usize)> = outs.iter().enumerate().map(|(o, &(ct, m))| (o, ct, m)).collect();
        self.outputs_stage(ctx, s, k, sh, &os)?;
        if let Some(e) = ev {
            e[3].record(s)?;
        }
        Ok(())
    }

    /// The per-input stage and both digit GEMMs of `ins` (the inputs of `sh`): Y = W X mod q_i over every output row
    /// of `sh`, left in the work buffers (`ap` [2][rows_out][N1], `bp` [2][rows_out][np]).
    #[allow(clippy::too_many_arguments)]
    fn inputs_gemm<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, k: &DevPcmmKeys<'_>, sh: &PcmmShape, planes: &DBuf<'_, Compute, u8>, ins: &[(PolyRef, usize)], ev: Option<[&Event; 4]>) -> Result<()> {
        let lay = k.shape;
        let (np, r, nch) = (lay.np(), lay.r(), lay.nch());
        let (ri, ro) = (sh.rows_in()?, sh.rows_out()?);
        if let Some(e) = ev {
            e[0].record(s)?;
        }
        let w = &mut self.work;
        let (stage, xr, res, scratch) = (w.stage.dptr_mut() as *mut c_void, w.xr.dptr_mut(), w.res.dptr_mut(), w.scratch.dptr_mut());
        let (pa, pbm, pap, pbp) = (w.a.dptr_mut(), w.bm.dptr_mut(), w.ap.dptr_mut(), w.bp.dptr_mut());
        let (c4, cmu, inv_c, ns) = (self.chain4, self.chain_mu, self.inv_c, self.nsplit as i32);
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
        Ok(())
    }

    /// The per-output stage of output blocks `outs` = (block index o of `sh`, ciphertext view, member) from the work
    /// buffers' GEMM rows: pack back, compose, switch back to s, lift, eval form.
    fn outputs_stage<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, k: &DevPcmmKeys<'_>, sh: &PcmmShape, outs: &[(usize, PolyRef, usize)]) -> Result<()> {
        let lay = k.shape;
        let (np, r, nch) = (lay.np(), lay.r(), lay.nch());
        let ro = sh.rows_out()?;
        let w = &mut self.work;
        let (stage, res, scratch) = (w.stage.dptr_mut() as *mut c_void, w.res.dptr_mut(), w.scratch.dptr_mut());
        let (pap, pbp, pc, ppk, pra, prb) = (w.ap.dptr_mut(), w.bp.dptr_mut(), w.c.dptr_mut(), w.pk.dptr_mut(), w.ra.dptr_mut(), w.rb.dptr_mut());
        let (c4, cmu, lift, ns) = (self.chain4, self.chain_mu, self.lift, self.nsplit as i32);
        let chain = |opc: i32| PcmmArgs { op: opc, n: N as i32, p: c4, mu: cmu, i4: ns, ..PcmmArgs::default() };
        let pkb = ppk.wrapping_add(L * S_SWITCH * N1);
        for &(o, ct, m) in outs {
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
        Ok(())
    }

    /// Row split (Sylph's reduction split), first half: the per-input stage and the GEMMs of the inputs `ins` (each
    /// at the split) of the sub-shape `sh` (these inputs, every output of the linear) on `planes` (`gather_planes_rows`
    /// of these inputs): a partial Y over every output row, left in the work buffers (`partial_ptrs`).
    pub fn product_rows<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, sh: &PcmmShape, planes: &DBuf<'_, Compute, u8>, ins: &[Ct<'_, W>]) -> Result<()> {
        for x in ins {
            Self::check_split(ctx, x)?;
        }
        let ir: Vec<(PolyRef, usize)> = Self::members(ins).into_iter().map(|(i, m)| (ins[i].buf.r(), m)).collect();
        if !self.cfg.shapes.contains(sh) || ir.len() != sh.n_in {
            return Err(shape("pcmm rows", format!("{} inputs for {sh:?} (configured: {})", ir.len(), self.cfg.shapes.contains(sh))));
        }
        if planes.len() < sh.planes_len()? || planes.dptr() as usize % 256 != 0 {
            return Err(shape("pcmm weights", format!("{} bytes at {:#x}, need {} 256-aligned", planes.len(), planes.dptr() as usize, sh.planes_len()?)));
        }
        let ki = self.keys.iter().position(|k| k.t == sh.t).ok_or_else(|| shape("pcmm", format!("no keys for t = {}", sh.t)))?;
        let k = self.keys.swap_remove(ki);
        let r = self.inputs_gemm(ctx, s, &k, sh, planes, &ir, None);
        self.keys.push(k);
        r
    }

    /// The work buffers holding the partial Y of `product_rows` (`ap`, `bp`): another device's `add_partial` reads
    /// them (peer access).
    pub fn partial_ptrs(&self) -> (*const u32, *const u32) {
        (self.work.ap.dptr(), self.work.bp.dptr())
    }

    /// Row split: another engine's partial Y (`other` = its `partial_ptrs`, same sub-shape `sh` rows, peer memory) added
    /// into this engine's for output blocks `os`, mod q_i (the sum over all inputs: exact).
    pub fn add_partial<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, sh: &PcmmShape, other: (*const u32, *const u32), os: &[usize]) -> Result<()> {
        let k = self.keys.iter().find(|k| k.t == sh.t).ok_or_else(|| shape("pcmm", format!("no keys for t = {}", sh.t)))?;
        let (np, nch, ro) = (k.shape.np(), k.shape.nch(), sh.rows_out()?);
        let (pap, pbp) = (self.work.ap.dptr_mut(), self.work.bp.dptr_mut());
        for &o in os {
            if o >= sh.n_out {
                return Err(shape("pcmm rows", format!("output {o} of {sh:?}")));
            }
            for (mine, theirs, cols) in [(pap, other.0, N1), (pbp, other.1, np)] {
                for prime in 0..2 {
                    let off = prime * ro * cols + o * nch * cols;
                    let mut a = Self::kargs(k, op::ADD, 0);
                    a.count = 1;
                    a.i0 = (nch * cols) as i32;
                    a.p[0] = k.primes[prime];
                    a.x = mine.wrapping_add(off) as *mut c_void;
                    a.y = theirs.wrapping_add(off) as *const c_void;
                    Self::launch(ctx, s, a)?;
                }
            }
        }
        Ok(())
    }

    /// Row split, second half: the per-output stage of output blocks `os` of `sh` from the (summed) work buffers into
    /// `outs` (output `os[j]` to member `out_map[j]`), each named ciphertext at `meta` with scale `nf_out`.
    #[allow(clippy::too_many_arguments)]
    pub fn product_outputs<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, sh: &PcmmShape, os: &[usize], outs: &mut [Ct<'_, W>], out_map: &[(usize, usize)], meta: &CtMeta, nf_out: f64) -> Result<()> {
        if os.len() != out_map.len() || os.iter().any(|&o| o >= sh.n_out) {
            return Err(shape("pcmm rows", format!("outputs {os:?} / map {} for {sh:?}", out_map.len())));
        }
        for &(c, m) in out_map {
            if c >= outs.len() || m >= outs[c].k() || outs[c].buf.lay != ctx.lay {
                return Err(shape("pcmm", format!("output member ({c}, {m}) outside the {} output ciphertexts", outs.len())));
            }
        }
        let ov: Vec<(usize, PolyRef, usize)> = os.iter().zip(out_map).map(|(&o, &(c, m))| (o, outs[c].buf.w(), m)).collect();
        let ki = self.keys.iter().position(|k| k.t == sh.t).ok_or_else(|| shape("pcmm", format!("no keys for t = {}", sh.t)))?;
        let k = self.keys.swap_remove(ki);
        let r = self.outputs_stage(ctx, s, &k, sh, &ov);
        self.keys.push(k);
        r?;
        for &(c, _) in out_map {
            outs[c].meta = CtMeta { nf: nf_out, ..meta.clone() };
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

/// On a lean chain the StC-first input's factors sit in 27-bit single-prime plaintexts, set for an input stamp
/// w_scale 2^shift of 2^PREDIV_STAMP_LOG2 (`stc_first::THIN_SPLIT_PCMM`: 16-bit weights at band shift 1). A linear of
/// band shift above PREDIV_SHIFT (whose inputs the plan keeps one level above the StC-first entry) first lowers its
/// inputs' scale by its stamp over that, exactly, on that level (`prediv`): every PCMM input then reaches the stages at
/// the same stamp (one factor, one cached last stage per input level).
pub const PREDIV_SHIFT: u32 = 1;
pub const PREDIV_STAMP_LOG2: i32 = 20;

/// The scale drop 2^s that the pre-division gives `lin`'s inputs on `route` (0: none): its stamp w_scale 2^shift over
/// 2^PREDIV_STAMP_LOG2, on a lean chain for a band shift above PREDIV_SHIFT.
pub fn prediv_shift<W: DevWord, M: DevStrategy<W>>(route: &StcFirstRoute<'_, W, M>, lin: &PcmmLinear) -> u32 {
    prediv_log2(route.thin(), lin)
}

/// `prediv_shift` on a chain that is lean (`thin`) or not.
pub fn prediv_log2(thin: bool, lin: &PcmmLinear) -> u32 {
    if !thin || lin.shift <= PREDIV_SHIFT {
        return 0;
    }
    let stamp = lin.w_scale * (lin.shift as f64).exp2();
    (stamp.log2().round() as i32 - PREDIV_STAMP_LOG2).max(0) as u32
}

/// The pre-division of each of `lin`'s inputs (`prediv` by `prediv_shift`), before their stamp: `linear` and the
/// per-input path of a linear whose inputs carry their own scales both run it.
pub fn prediv_inputs<'a, W: DevWord, M: DevStrategy<W>>(cx: &mut BtsCx<'_, 'a, W, M>, route: &StcFirstRoute<'a, W, M>, pcmm: &Pcmm<'a, '_, W>, lin: &PcmmLinear, ins: &mut [Ct<'a, W>]) -> Result<()> {
    let s = prediv_shift(route, lin);
    if s > 0 {
        for x in ins.iter_mut() {
            prediv(cx, route, pcmm, x, s, &lin.name)?;
        }
    }
    Ok(())
}

/// Lower `x`'s scale by 2^s at the cost of one level: x 2^k (an exact integer product) then the level's rescale, which
/// divides by its primes (~2^(k + s)). The value is unchanged; its scale, which the StC-first input's factors divide,
/// drops by the primes over 2^k. `x` must sit one level above the StC-first entry (after its pending rescale).
fn prediv<'a, W: DevWord, M: DevStrategy<W>>(cx: &mut BtsCx<'_, 'a, W, M>, route: &StcFirstRoute<'a, W, M>, pcmm: &Pcmm<'a, '_, W>, x: &mut Ct<'a, W>, s: u32, name: &str) -> Result<()> {
    let ctx = cx.ctx;
    if x.meta.deg == 2 {
        cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
    }
    let entry = route.entry_level(ctx.shape)?;
    if x.meta.level + ctx.shape.unit > entry {
        return Err(shape("pcmm linear", format!("{name}: input at level {} has no level above the StC-first entry {entry} to divide its band shift on", x.meta.level)));
    }
    let top = ctx.top_of(x.meta.level)?;
    let g = ctx.chain.drop_at(top).ok_or_else(|| shape("pcmm linear", format!("{name}: input top {top} off the level grid")))?;
    let bits: f64 = (0..g).map(|j| (ctx.chain.q[top - j] as f64).log2()).sum();
    let k = (bits.floor() as u32).checked_sub(s).ok_or_else(|| shape("pcmm linear", format!("{name}: band shift {s} past the level's {bits:.1} bits")))?;
    pcmm.mult_pow2(ctx, &cx.ex.compute, x, k)?;
    let nf = x.meta.nf * (k as f64).exp2();
    x.meta.nf = nf;
    x.meta.deg = 2;
    cx.ops.rescale(ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
    x.meta.nf = nf / ctx.scale.mod_reduce_product(top as i32);
    Ok(())
}

/// `linear` for inputs whose metadata differ (the session's blocked layout: channel blocks carry their own scales): each
/// input gets the pre-division (`prediv_inputs`, as `linear`) and its own StC-first input call (perseus pcmm.py
/// `_device_cf`: `bts.begin` + `stc_first_input` per input, the mode-2 last stage built for that input's scale), the
/// finishes run from input 0's call, as perseus does.
#[allow(clippy::too_many_arguments)]
pub fn linear_per_input<'a, W: DevWord, M: DevStrategy<W>>(cx: &mut BtsCx<'_, 'a, W, M>, route: &StcFirstRoute<'a, W, M>, pcmm: &mut Pcmm<'a, '_, W>, lin: &PcmmLinear, planes: &DBuf<'_, Compute, u8>, ins: &mut [Ct<'a, W>], outs: &mut [Ct<'a, W>], groups: &[Vec<usize>], drops: &[u32], opts: LinearOpts) -> Result<()> {
    let all: Vec<usize> = (0..lin.shape.n_out).collect();
    linear_per_input_parts(cx, route, pcmm, lin, &[(planes, &all)], ins, outs, groups, drops, opts)
}

/// `linear_per_input` with the product in parts: each part's planes (`gather_planes` of its outputs, in that order)
/// times the inputs into its outputs; the parts cover the outputs once (a joint linear's two halves, both here).
#[allow(clippy::too_many_arguments)]
pub fn linear_per_input_parts<'a, W: DevWord, M: DevStrategy<W>>(cx: &mut BtsCx<'_, 'a, W, M>, route: &StcFirstRoute<'a, W, M>, pcmm: &mut Pcmm<'a, '_, W>, lin: &PcmmLinear, parts: &[(&DBuf<'_, Compute, u8>, &[usize])], ins: &mut [Ct<'a, W>], outs: &mut [Ct<'a, W>], groups: &[Vec<usize>], drops: &[u32], opts: LinearOpts) -> Result<()> {
    linear_per_input_off(cx, route, pcmm, lin, parts, ins, outs, groups, drops, opts, None)
}

/// `linear_per_input_parts` with the finish groups' offload (`linear`'s: the last `off.far` groups refreshed on the
/// other device, issued before this device's groups, joined after them). The inputs' StC-first stages stay here
/// (each input its own call).
#[allow(clippy::too_many_arguments)]
pub fn linear_per_input_off<'a, W: DevWord, M: DevStrategy<W>>(cx: &mut BtsCx<'_, 'a, W, M>, route: &StcFirstRoute<'a, W, M>, pcmm: &mut Pcmm<'a, '_, W>, lin: &PcmmLinear, parts: &[(&DBuf<'_, Compute, u8>, &[usize])], ins: &mut [Ct<'a, W>], outs: &mut [Ct<'a, W>], groups: &[Vec<usize>], drops: &[u32], opts: LinearOpts, off: Option<&mut dyn GroupOffload<'a, W, M>>) -> Result<()> {
    let sh = lin.shape;
    let mut out_map = vec![(usize::MAX, 0); sh.n_out];
    for (g, ms) in groups.iter().enumerate() {
        for (j, &m) in ms.iter().enumerate() {
            out_map[m] = (g, j);
        }
    }
    let mut cover: Vec<usize> = parts.iter().flat_map(|(_, o)| o.iter().copied()).collect();
    cover.sort();
    if cover != (0..sh.n_out).collect::<Vec<_>>() || out_map.iter().any(|x| x.0 == usize::MAX) {
        return Err(shape("pcmm linear", format!("{}: the parts / groups do not cover the {} outputs once", lin.name, sh.n_out)));
    }
    let stamp = lin.w_scale * (lin.shift as f64).exp2();
    let r = Route { cf: Some(opts.cf), stc_first: true, real: opts.real, ..Route::dense() };
    let co = CallOpts { prescaled: false, own_cf: true };
    prediv_inputs(cx, route, pcmm, lin, ins)?;
    let mut first: Option<BtsCall<'_, 'a, W, M>> = None;
    for i in 0..ins.len() {
        ins[i].meta.nf *= stamp;
        let x = &mut ins[i..i + 1];
        let mut call = BtsCall::begin(StagedRef::StcFirst(route), cx, x, &r, co)?;
        if call.stc_first_input(cx, x)?.is_none() {
            return Err(shape("pcmm linear", format!("{}: input {i} below the StC-first entry", lin.name)));
        }
        if first.is_none() {
            first = Some(call);
        }
    }
    let inp = first.ok_or_else(|| shape("pcmm linear", "no inputs"))?;
    let nf_out = ins[0].meta.nf / lin.w_scale;
    for (planes, po) in parts {
        let sub = PcmmShape { n_out: po.len(), ..sh };
        let map: Vec<(usize, usize)> = po.iter().map(|&m| out_map[m]).collect();
        pcmm.product(cx.ctx, &cx.ex.compute, &sub, planes, ins, outs, &map, nf_out, None)?;
    }
    let mut off = off.filter(|_| drops.len() >= 2);
    let Some(o) = off.as_deref_mut() else {
        for (g, &dr) in drops.iter().enumerate() {
            let mut fin = inp.finish_call(dr)?;
            fin.at_split(cx, &outs[g..g + 1])?;
            fin.run_to(cx, &mut outs[g..g + 1], Phase::Done)?;
            if lin.shift > 0 {
                pcmm.mult_pow2(cx.ctx, &cx.ex.compute, &mut outs[g], lin.shift)?;
            }
        }
        return Ok(());
    };
    // the last groups on the offload device, issued first (they start as soon as the product is done)
    let mut calls = Vec::with_capacity(drops.len());
    for (g, &dr) in drops.iter().enumerate() {
        let mut fin = inp.finish_call(dr)?;
        fin.at_split(cx, &outs[g..g + 1])?;
        calls.push(fin);
    }
    let n_far = o.far(calls.len()).min(calls.len());
    let n_here = calls.len() - n_far;
    let check = o.check();
    let mut stash: Vec<(CtMeta, Vec<[Vec<W>; 2]>)> = Vec::new();
    for g in n_here..calls.len() {
        if check {
            stash.push((outs[g].meta, download_members(cx, &outs[g])?));
        }
        o.issue(&cx.ex.compute, cx.sh, &calls[g], &mut outs[g])?;
    }
    for (g, fin) in calls[..n_here].iter_mut().enumerate() {
        fin.run_to(cx, &mut outs[g..g + 1], Phase::Done)?;
    }
    if n_far > 0 {
        o.join(&cx.ex.compute)?;
    }
    // PC_REMOTE_CHECK: each remote group again here from its saved input, against what came back
    for (j, (m_in, data)) in stash.into_iter().enumerate() {
        let g = n_here + j;
        let far = (outs[g].meta, download_members(cx, &outs[g])?);
        let rows: Vec<usize> = (0..=pc_shadow::meta::top_limb(cx.ctx.shape, m_in.level)? as usize).collect();
        for (c, parts) in data.iter().enumerate() {
            for (part, d) in parts.iter().enumerate() {
                outs[g].buf.upload_rows(&cx.ex.compute, c, part, &rows, d)?;
            }
        }
        outs[g].meta = m_in;
        calls[g].run_to(cx, &mut outs[g..g + 1], Phase::Done)?;
        let here = (outs[g].meta, download_members(cx, &outs[g])?);
        if here != far {
            return Err(shape("pcmm linear", format!("{}: remote check: per-input finish group {g} of {} refreshed remotely differs from the same group refreshed here", lin.name, calls.len())));
        }
        o.checked();
    }
    if lin.shift > 0 {
        for x in outs[..calls.len()].iter_mut() {
            pcmm.mult_pow2(cx.ctx, &cx.ex.compute, x, lin.shift)?;
        }
    }
    Ok(())
}

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

/// `finish_groups` for a finish split across two devices: each drop class's outputs halved (the first ceil half here,
/// the rest on the other device), each half chunked into groups of `batch`, every group of this device before every
/// group of the other. Returns the groups and how many of them (the last ones) go to the other device. Halving every
/// class balances the two devices by cost whatever the classes' per-output costs (a class's outputs refresh alike).
pub fn finish_groups_split(drops: &[u32], batch: usize, batch_drops: bool) -> Result<(Vec<Vec<usize>>, usize)> {
    let whole = finish_groups(drops, batch, batch_drops)?;
    let b = batch.max(1);
    // the classes `finish_groups` batches together (one class when the drops are uniform or batching is per drop)
    let uniform = drops.windows(2).all(|w| w[0] == w[1]);
    if !(uniform || (b > 1 && batch_drops)) {
        // one output per group: the last half of the groups
        let n = whole.len();
        return Ok((whole, n / 2));
    }
    let mut classes: Vec<Vec<usize>> = Vec::new();
    let mut seen: Vec<u32> = Vec::new();
    for (m, &d) in drops.iter().enumerate() {
        match seen.iter().position(|&x| x == d) {
            Some(c) if !uniform => classes[c].push(m),
            _ if uniform && !classes.is_empty() => classes[0].push(m),
            _ => {
                seen.push(d);
                classes.push(vec![m]);
            }
        }
    }
    let (mut here, mut far) = (Vec::new(), Vec::new());
    for ms in &classes {
        let h = ms.len() - ms.len() / 2;
        here.extend(ms[..h].chunks(b).map(|c| c.to_vec()));
        far.extend(ms[h..].chunks(b).map(|c| c.to_vec()));
    }
    let n_far = far.len();
    here.extend(far);
    Ok((here, n_far))
}

/// The digit planes `[nw][rows_out][rows_in]` of a linear restricted to the outputs `outs` (in that order): the
/// planes of the sub-shape (`n_out = outs.len()`) a device computing only those outputs multiplies by (each output
/// owns `nch` consecutive rows of every plane). Returns the planes and the sub-shape.
pub fn gather_planes(planes: &[u8], sh: &PcmmShape, outs: &[usize]) -> Result<(Vec<u8>, PcmmShape)> {
    let (nch, ri, ro) = (sh.nch()?, sh.rows_in()?, sh.rows_out()?);
    if planes.len() != sh.planes_len()? || outs.is_empty() || outs.iter().any(|&m| m >= sh.n_out) {
        return Err(shape("pcmm gather", format!("{} plane bytes for {sh:?}, outputs {outs:?}", planes.len())));
    }
    let sub = PcmmShape { n_out: outs.len(), ..*sh };
    let mut v = Vec::with_capacity(sub.planes_len()?);
    for p in 0..sh.nw {
        for &m in outs {
            let r0 = p * ro * ri + m * nch * ri;
            v.extend_from_slice(&planes[r0..r0 + nch * ri]);
        }
    }
    Ok((v, sub))
}

/// The digit planes `[nw][rows_out][rows_in]` of a linear restricted to the inputs `ins` (in that order): the planes
/// of the sub-shape (`n_in = ins.len()`, every output) a device multiplies its inputs by in a row split (each input
/// owns `nch` consecutive columns of every row). Returns the planes and the sub-shape.
pub fn gather_planes_rows(planes: &[u8], sh: &PcmmShape, ins: &[usize]) -> Result<(Vec<u8>, PcmmShape)> {
    let (nch, ri, ro) = (sh.nch()?, sh.rows_in()?, sh.rows_out()?);
    if planes.len() != sh.planes_len()? || ins.is_empty() || ins.iter().any(|&i| i >= sh.n_in) {
        return Err(shape("pcmm gather rows", format!("{} plane bytes for {sh:?}, inputs {ins:?}", planes.len())));
    }
    let sub = PcmmShape { n_in: ins.len(), ..*sh };
    let mut v = Vec::with_capacity(sub.planes_len()?);
    for p in 0..sh.nw {
        for row in 0..ro {
            let base = p * ro * ri + row * ri;
            for &i in ins {
                v.extend_from_slice(&planes[base + i * nch..base + (i + 1) * nch]);
            }
        }
    }
    Ok((v, sub))
}

/// The capture descriptor of output m (pcmm.py `Pcmm.desc`; graph v2 `args.external`).
pub fn desc_json(lin: &PcmmLinear, m: usize, drop: u32) -> String {
    format!(
        "{{\"op\": \"pcmm\", \"linear\": {:?}, \"out\": {m}, \"n_in\": {}, \"n_out\": {}, \"t\": {}, \"shift\": {}, \"raise_drop\": {drop}}}",
        lin.name, lin.shape.n_in, lin.shape.n_out, lin.shape.t, lin.shift
    )
}

/// One PCMM linear split across two devices by input blocks (Sylph's reduction split, the 2-D block of a column of
/// pc = 2): each device stages and multiplies its half of the inputs into a partial product of every output on its
/// rows of the planes (`gather_planes_rows`), the partials are exchanged and summed mod q for each device's own outputs
/// (exact: `Pcmm::add_partial`), and each device runs the output stage of its outputs and finishes them. `planes_here`:
/// this device's rows (sub-shape: the first `n_in - n_in / 2` inputs, every output); `planes_full`: the whole planes
/// when this device holds them (PC_REMOTE_CHECK recomputes the whole product here). Groups as `linear_joint`.
#[allow(clippy::too_many_arguments)]
pub fn linear_rows<'r, 'a, 'lt, W: DevWord, M: DevStrategy<W>>(
    cx: &mut BtsCx<'_, 'a, W, M>,
    route: &'r StcFirstRoute<'a, W, M>,
    pcmm: &mut Pcmm<'a, 'lt, W>,
    lin: &PcmmLinear,
    planes_here: &DBuf<'_, Compute, u8>,
    planes_full: Option<&DBuf<'_, Compute, u8>>,
    ins: &mut [Ct<'a, W>],
    outs: &mut [Ct<'a, W>],
    groups: &[Vec<usize>],
    drops: &[u32],
    n_far: usize,
    opts: LinearOpts,
    off: &mut dyn GroupOffload<'a, W, M>,
) -> Result<()> {
    let sh = lin.shape;
    if groups.len() != outs.len() || drops.len() != outs.len() || n_far == 0 || n_far >= groups.len() || ins.len() < 2 {
        return Err(shape("pcmm rows", format!("{} groups, {} drops for {} outputs, {n_far} far, {} inputs", groups.len(), drops.len(), outs.len(), ins.len())));
    }
    let n_here = groups.len() - n_far;
    let (here_outs, far_outs) = joint_outputs(groups, n_far);
    let mut all: Vec<usize> = here_outs.iter().chain(&far_outs).copied().collect();
    all.sort();
    if all != (0..sh.n_out).collect::<Vec<_>>() {
        return Err(shape("pcmm rows", format!("{}: the groups do not cover the {} outputs once", lin.name, sh.n_out)));
    }
    let pos = |list: &[usize], m: usize| list.iter().position(|&x| x == m).unwrap_or(usize::MAX);
    let mut map_here = vec![(usize::MAX, 0); here_outs.len()];
    for (g, ms) in groups[..n_here].iter().enumerate() {
        if ms.len() != outs[g].k() {
            return Err(shape("pcmm rows", format!("group {g}: {} outputs, {} members", ms.len(), outs[g].k())));
        }
        for (j, &m) in ms.iter().enumerate() {
            map_here[pos(&here_outs, m)] = (g, j);
        }
    }
    let far_groups: Vec<Vec<usize>> = groups[n_here..].iter().map(|ms| ms.iter().map(|&m| pos(&far_outs, m)).collect()).collect();
    let n_in = ins.len();
    let s_far = n_in / 2;
    let s_here = n_in - s_far;
    let sub_here = PcmmShape { n_in: s_here, ..sh };
    let sub_far = PcmmShape { n_in: s_far, ..sh };
    prediv_inputs(cx, route, pcmm, lin, ins)?;
    let stamp = lin.w_scale * (lin.shift as f64).exp2();
    for x in ins.iter_mut() {
        x.meta.nf *= stamp;
    }
    let r = Route { cf: Some(opts.cf), stc_first: true, real: opts.real, ..Route::dense() };
    let mut inp = BtsCall::begin(StagedRef::StcFirst(route), cx, ins, &r, CallOpts { prescaled: false, own_cf: true })?;
    let check = off.check();
    // the other device stages its inputs (kept there) and multiplies them while this one does the same with its own
    off.rows_stc(&cx.ex.compute, cx.sh, &inp, s_here, &mut ins[s_here..], check)?;
    let far_partial = off.rows_product(lin, &sub_far)?;
    if inp.stc_first_input(cx, &mut ins[..s_here])?.is_none() {
        return Err(shape("pcmm rows", format!("{}: inputs at level {} are below the StC-first entry", lin.name, ins[0].meta.level)));
    }
    pcmm.product_rows(cx.ctx, &cx.ex.compute, &sub_here, planes_here, &ins[..s_here])?;
    let meta = ins[0].meta.clone();
    let nf_out = meta.nf / lin.w_scale;
    // the reduce-scatter: each device adds the other's partial rows of its own outputs (disjoint rows of both
    // partials); this one first (after the other's product), then the other (after everything issued here so far)
    off.rows_wait(&cx.ex.compute)?;
    pcmm.add_partial(cx.ctx, &cx.ex.compute, &sub_here, far_partial, &here_outs)?;
    off.rows_far(&cx.ex.compute, cx.sh, &sub_far, &inp, pcmm.partial_ptrs(), &far_outs, &meta, nf_out, &far_groups, &drops[n_here..], &mut outs[n_here..])?;
    pcmm.product_outputs(cx.ctx, &cx.ex.compute, &sub_here, &here_outs, &mut outs[..n_here], &map_here, &meta, nf_out)?;
    for g in 0..n_here {
        let mut fin = inp.finish_call(drops[g])?;
        fin.at_split(cx, &outs[g..g + 1])?;
        fin.run_to(cx, &mut outs[g..g + 1], Phase::Done)?;
    }
    off.join(&cx.ex.compute)?;
    if check {
        // PC_REMOTE_CHECK: the whole product here on the whole planes (the other device's staged inputs came back),
        // every group finished here, against what the split produced
        let planes_full = planes_full.ok_or_else(|| shape("pcmm rows", "PC_REMOTE_CHECK needs this device's copy of the whole planes"))?;
        let got: Vec<(CtMeta, Vec<[Vec<W>; 2]>)> = outs.iter().map(|x| download_members(cx, x).map(|d| (x.meta, d))).collect::<Result<_>>()?;
        let mut map = vec![(usize::MAX, 0); sh.n_out];
        for (g, ms) in groups.iter().enumerate() {
            for (j, &m) in ms.iter().enumerate() {
                map[m] = (g, j);
            }
        }
        pcmm.product(cx.ctx, &cx.ex.compute, &sh, planes_full, ins, outs, &map, nf_out, None)?;
        for g in 0..groups.len() {
            let x = &mut outs[g..g + 1];
            let mut fin = inp.finish_call(drops[g])?;
            fin.at_split(cx, x)?;
            fin.run_to(cx, x, Phase::Done)?;
            if (x[0].meta, download_members(cx, &x[0])?) != got[g] {
                return Err(shape("pcmm rows", format!("{}: remote check: group {g} ({} side) of the row split differs from the whole product here", lin.name, if g < n_here { "this" } else { "the other" })));
            }
            off.checked();
        }
    }
    for x in outs.iter_mut() {
        if lin.shift > 0 {
            pcmm.mult_pow2(cx.ctx, &cx.ex.compute, x, lin.shift)?;
        }
    }
    Ok(())
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
    mut off: Option<&mut dyn GroupOffload<'a, W, M>>,
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
    prediv_inputs(cx, route, pcmm, lin, ins)?;
    for x in ins.iter_mut() {
        x.meta.nf *= stamp;
    }
    let r = Route { cf: Some(opts.cf), stc_first: true, real: opts.real, ..Route::dense() };
    let mut inp = BtsCall::begin(StagedRef::StcFirst(route), cx, ins, &r, CallOpts { prescaled: false, own_cf: true })?;
    // the last inputs' StC-first stage on the offload device (PC_STC_SPLIT), issued first, joined before the product
    let n_in = ins.len();
    let s_far = off.as_deref().map_or(0, |o| o.stc_far(n_in).min(n_in.saturating_sub(1)));
    let s_here = n_in - s_far;
    let stash_in: Option<Vec<(CtMeta, Vec<[Vec<W>; 2]>)>> = match off.as_deref() {
        Some(o) if s_far > 0 && o.check() => Some(ins[s_here..].iter().map(|x| download_members(cx, x).map(|d| (x.meta, d))).collect::<Result<_>>()?),
        _ => None,
    };
    if s_far > 0 {
        if let Some(o) = off.as_deref_mut() {
            o.stc_issue(&cx.ex.compute, cx.sh, &inp, &mut ins[s_here..])?;
        }
    }
    if inp.stc_first_input(cx, &mut ins[..s_here])?.is_none() {
        return Err(shape("pcmm linear", format!("{}: inputs at level {} are below the StC-first entry (cap {})", lin.name, ins[0].meta.level, route.input_cap(cx.ctx.shape)?)));
    }
    if s_far > 0 {
        if let Some(o) = off.as_deref_mut() {
            o.join(&cx.ex.compute)?;
            // PC_REMOTE_CHECK: the remote inputs' stage again here from their saved inputs, word for word
            if let Some(stash) = stash_in {
                for (j, (m_in, data)) in stash.into_iter().enumerate() {
                    let x = &mut ins[s_here + j];
                    let far = (x.meta, download_members(cx, x)?);
                    let rows: Vec<usize> = (0..=pc_shadow::meta::top_limb(cx.ctx.shape, m_in.level)? as usize).collect();
                    for (c, parts) in data.iter().enumerate() {
                        for (part, d) in parts.iter().enumerate() {
                            x.buf.upload_rows(&cx.ex.compute, c, part, &rows, d)?;
                        }
                    }
                    x.meta = m_in;
                    let mut again = BtsCall::resume(StagedRef::StcFirst(route), inp.st.clone(), Phase::Begun, inp.opts);
                    again.stc_first_input(cx, std::slice::from_mut(x))?;
                    let here = (x.meta, download_members(cx, x)?);
                    if here != far {
                        return Err(shape("pcmm linear", format!("{}: remote check: input {} of {n_in}'s StC-first stage on the other device differs from the same stage here", lin.name, s_here + j)));
                    }
                    o.checked();
                }
            }
        }
    }
    if let Some(rc) = rec.as_deref_mut() {
        let ni: Vec<(&str, u32)> = names.iter().map(|(n, l)| (n.as_str(), *l)).collect();
        record_stc_first_inputs(rc, route, cx, &ni)?;
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
    // the last half of the groups on the offload device, issued first (it starts as soon as the product is done)
    let mut off = off.filter(|_| calls.len() >= 2);
    // (the input stages above used it whatever the group count)
    let n_far = off.as_deref().map_or(0, |o| o.far(calls.len()).min(calls.len()));
    let n_here = calls.len() - n_far;
    let mut stash: Vec<(CtMeta, Vec<[Vec<W>; 2]>)> = Vec::new();
    if let Some(o) = off.as_deref_mut() {
        let check = o.check();
        for g in n_here..calls.len() {
            if check {
                stash.push((outs[g].meta, download_members(cx, &outs[g])?));
            }
            o.issue(&cx.ex.compute, cx.sh, &calls[g], &mut outs[g])?;
        }
    }
    if opts.interleave {
        let mut sl: Vec<&mut [Ct<'a, W>]> = outs[..n_here].chunks_mut(1).collect();
        crate::bts::staged::run_interleaved(cx, &mut calls[..n_here], &mut sl, Phase::Done)?;
    } else {
        for (g, fin) in calls[..n_here].iter_mut().enumerate() {
            fin.run_to(cx, &mut outs[g..g + 1], Phase::Done)?;
        }
    }
    if let Some(o) = off.as_deref_mut() {
        if n_far > 0 {
            o.join(&cx.ex.compute)?;
        }
        // PC_REMOTE_CHECK: each remote group again here from its saved input, against what came back
        for (j, (m_in, data)) in stash.into_iter().enumerate() {
            let g = n_here + j;
            let far = (outs[g].meta, download_members(cx, &outs[g])?);
            let rows: Vec<usize> = (0..=pc_shadow::meta::top_limb(cx.ctx.shape, m_in.level)? as usize).collect();
            for (c, parts) in data.iter().enumerate() {
                for (part, d) in parts.iter().enumerate() {
                    outs[g].buf.upload_rows(&cx.ex.compute, c, part, &rows, d)?;
                }
            }
            outs[g].meta = m_in;
            calls[g].run_to(cx, &mut outs[g..g + 1], Phase::Done)?;
            let here = (outs[g].meta, download_members(cx, &outs[g])?);
            if here != far {
                return Err(shape("pcmm linear", format!("{}: remote check: finish group {g} of {} refreshed remotely differs from the same group refreshed here", lin.name, calls.len())));
            }
            o.checked();
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

/// The output order of a joint linear's two devices from its finish groups (`finish_groups_split`): this device's
/// outputs (groups `..n_here`, in order) then the other's; the planes of each side are gathered in that order.
pub fn joint_outputs(groups: &[Vec<usize>], n_far: usize) -> (Vec<usize>, Vec<usize>) {
    let n_here = groups.len() - n_far.min(groups.len());
    (groups[..n_here].concat(), groups[n_here..].concat())
}

/// One PCMM linear split across two devices by output blocks (the Sylph column split): each device computes the
/// product of its own outputs on its own planes and finishes them. `groups` / `drops` as `linear` (this device's
/// groups first, the last `n_far` on the other device: `finish_groups_split`); `planes_here` the planes of this
/// device's outputs (`gather_planes` in `joint_outputs` order), `planes_far` the same of the other's when this device
/// holds them (PC_REMOTE_CHECK recomputes the far groups here). The inputs' StC-first stage is split in two halves,
/// each device keeping all the inputs at the split for its product.
#[allow(clippy::too_many_arguments)]
pub fn linear_joint<'r, 'a, 'lt, W: DevWord, M: DevStrategy<W>>(
    cx: &mut BtsCx<'_, 'a, W, M>,
    route: &'r StcFirstRoute<'a, W, M>,
    pcmm: &mut Pcmm<'a, 'lt, W>,
    lin: &PcmmLinear,
    planes_here: &DBuf<'_, Compute, u8>,
    planes_far: Option<&DBuf<'_, Compute, u8>>,
    ins: &mut [Ct<'a, W>],
    outs: &mut [Ct<'a, W>],
    groups: &[Vec<usize>],
    drops: &[u32],
    n_far: usize,
    opts: LinearOpts,
    off: &mut dyn GroupOffload<'a, W, M>,
) -> Result<()> {
    let sh = lin.shape;
    if groups.len() != outs.len() || drops.len() != outs.len() || n_far == 0 || n_far >= groups.len() {
        return Err(shape("pcmm joint", format!("{} groups, {} drops for {} outputs, {n_far} far", groups.len(), drops.len(), outs.len())));
    }
    let n_here = groups.len() - n_far;
    let (here_outs, far_outs) = joint_outputs(groups, n_far);
    let mut all: Vec<usize> = here_outs.iter().chain(&far_outs).copied().collect();
    all.sort();
    if all != (0..sh.n_out).collect::<Vec<_>>() {
        return Err(shape("pcmm joint", format!("{}: the groups do not cover the {} outputs once", lin.name, sh.n_out)));
    }
    let pos = |list: &[usize], m: usize| list.iter().position(|&x| x == m).unwrap_or(usize::MAX);
    let sub_here = PcmmShape { n_out: here_outs.len(), ..sh };
    let sub_far = PcmmShape { n_out: far_outs.len(), ..sh };
    let mut map_here = vec![(usize::MAX, 0); sub_here.n_out];
    for (g, ms) in groups[..n_here].iter().enumerate() {
        if ms.len() != outs[g].k() {
            return Err(shape("pcmm joint", format!("group {g}: {} outputs, {} members", ms.len(), outs[g].k())));
        }
        for (j, &m) in ms.iter().enumerate() {
            map_here[pos(&here_outs, m)] = (g, j);
        }
    }
    let far_groups: Vec<Vec<usize>> = groups[n_here..].iter().map(|ms| ms.iter().map(|&m| pos(&far_outs, m)).collect()).collect();
    prediv_inputs(cx, route, pcmm, lin, ins)?;
    let stamp = lin.w_scale * (lin.shift as f64).exp2();
    for x in ins.iter_mut() {
        x.meta.nf *= stamp;
    }
    let r = Route { cf: Some(opts.cf), stc_first: true, real: opts.real, ..Route::dense() };
    let mut inp = BtsCall::begin(StagedRef::StcFirst(route), cx, ins, &r, CallOpts { prescaled: false, own_cf: true })?;
    // inputs: the other device stages the last half (kept there, copied back), this one the first, then the first half
    // goes over: both hold every input at the split
    let n_in = ins.len();
    let s_far = n_in / 2;
    let s_here = n_in - s_far;
    let check = off.check();
    let stash_far_in: Option<Vec<(CtMeta, Vec<[Vec<W>; 2]>)>> = if check && s_far > 0 {
        Some(ins[s_here..].iter().map(|x| download_members(cx, x).map(|d| (x.meta, d))).collect::<Result<_>>()?)
    } else {
        None
    };
    if s_far > 0 {
        off.joint_stc(&cx.ex.compute, cx.sh, &inp, s_here, &mut ins[s_here..])?;
    }
    if inp.stc_first_input(cx, &mut ins[..s_here])?.is_none() {
        return Err(shape("pcmm joint", format!("{}: inputs at level {} are below the StC-first entry", lin.name, ins[0].meta.level)));
    }
    off.joint_inputs(&cx.ex.compute, &ins[..s_here])?;
    // the far half's stage written back here before this device's product reads it; with no far half (a one-input
    // linear) nothing comes back, and `joint_stc`'s join would be a stale record (a capture error under graphs). The
    // other device's reads of these inputs are covered by the join after the far finishes.
    if s_far > 0 {
        off.join(&cx.ex.compute)?;
    }
    let nf_out = ins[0].meta.nf / lin.w_scale;
    // the far outputs on the other device (product, finishes) while this one does its own
    off.joint_far(&cx.ex.compute, cx.sh, lin, &sub_far, &inp, nf_out, &far_groups, &drops[n_here..], &mut outs[n_here..])?;
    pcmm.product(cx.ctx, &cx.ex.compute, &sub_here, planes_here, ins, &mut outs[..n_here], &map_here, nf_out, None)?;
    for g in 0..n_here {
        let mut fin = inp.finish_call(drops[g])?;
        fin.at_split(cx, &outs[g..g + 1])?;
        fin.run_to(cx, &mut outs[g..g + 1], Phase::Done)?;
    }
    off.join(&cx.ex.compute)?;
    if check {
        // PC_REMOTE_CHECK: the far half of the inputs' stage, the far product and its finishes again here, word for word
        let planes_far = planes_far.ok_or_else(|| shape("pcmm joint", "PC_REMOTE_CHECK needs this device's copy of the far planes"))?;
        if let Some(stash) = stash_far_in {
            for (j, (m_in, data)) in stash.into_iter().enumerate() {
                let x = &mut ins[s_here + j];
                let far = (x.meta, download_members(cx, x)?);
                let rows: Vec<usize> = (0..=pc_shadow::meta::top_limb(cx.ctx.shape, m_in.level)? as usize).collect();
                for (c, parts) in data.iter().enumerate() {
                    for (part, d) in parts.iter().enumerate() {
                        x.buf.upload_rows(&cx.ex.compute, c, part, &rows, d)?;
                    }
                }
                x.meta = m_in;
                let mut again = BtsCall::resume(StagedRef::StcFirst(route), inp.st.clone(), Phase::Begun, inp.opts);
                again.stc_first_input(cx, std::slice::from_mut(x))?;
                if (x.meta, download_members(cx, x)?) != far {
                    return Err(shape("pcmm joint", format!("{}: remote check: input {}'s stage on the other device differs", lin.name, s_here + j)));
                }
                off.checked();
            }
        }
        let far_res: Vec<(CtMeta, Vec<[Vec<W>; 2]>)> = outs[n_here..].iter().map(|x| download_members(cx, x).map(|d| (x.meta, d))).collect::<Result<_>>()?;
        let mut map_far = vec![(usize::MAX, 0); sub_far.n_out];
        for (g, ms) in far_groups.iter().enumerate() {
            for (j, &k) in ms.iter().enumerate() {
                map_far[k] = (g, j);
            }
        }
        pcmm.product(cx.ctx, &cx.ex.compute, &sub_far, planes_far, ins, &mut outs[n_here..], &map_far, nf_out, None)?;
        for g in 0..n_far {
            let x = &mut outs[n_here + g..n_here + g + 1];
            let mut fin = inp.finish_call(drops[n_here + g])?;
            fin.at_split(cx, x)?;
            fin.run_to(cx, x, Phase::Done)?;
            if (x[0].meta, download_members(cx, &x[0])?) != far_res[g] {
                return Err(shape("pcmm joint", format!("{}: remote check: far group {g} (product + finish on the other device) differs from the same here", lin.name)));
            }
            off.checked();
        }
    }
    for x in outs.iter_mut() {
        if lin.shift > 0 {
            pcmm.mult_pow2(cx.ctx, &cx.ex.compute, x, lin.shift)?;
        }
    }
    Ok(())
}

/// Every member's live rows of both polynomials (host; waits for `cx`'s compute stream).
fn download_members<W: DevWord, M: DevStrategy<W>>(cx: &BtsCx<'_, '_, W, M>, x: &Ct<'_, W>) -> Result<Vec<[Vec<W>; 2]>> {
    let rows: Vec<usize> = (0..=x.top(cx.ctx.shape)?).collect();
    (0..x.k()).map(|c| Ok([x.buf.download_rows(&cx.ex.compute, c, 0, &rows)?, x.buf.download_rows(&cx.ex.compute, c, 1, &rows)?])).collect()
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

    /// The balanced split: each drop class halved (first ceil half here), each half batched, here's groups first.
    #[test]
    fn groups_split() {
        // Llama qkv: 12 uniform outputs, batch 4: 4 + 2 here, 4 + 2 there (was 3 groups, 1 there)
        assert_eq!(finish_groups_split(&[0; 12], 4, true).unwrap(), (vec![vec![0, 1, 2, 3], vec![4, 5], vec![6, 7, 8, 9], vec![10, 11]], 2));
        assert_eq!(finish_groups_split(&[0; 8], 4, true).unwrap(), (vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]], 1));
        // per-drop classes: {0, 1, 2} -> [0, 1] | [2]; {3, 4} -> [3] | [4]; {5} -> [5] | -
        assert_eq!(finish_groups_split(&[0, 0, 0, 1, 1, 2], 4, true).unwrap(), (vec![vec![0, 1], vec![3], vec![5], vec![2], vec![4]], 2));
        // one output per group (mixed drops, no per-drop batching): the last half of the groups
        assert_eq!(finish_groups_split(&[0, 1, 0, 1], 2, false).unwrap(), (vec![vec![0], vec![1], vec![2], vec![3]], 2));
        // every output is in exactly one group
        for (d, b, bd) in [(vec![0u32; 7], 3usize, true), (vec![0, 1, 2, 1, 0, 0, 2], 4, true), (vec![3; 1], 4, true)] {
            let (g, n) = finish_groups_split(&d, b, bd).unwrap();
            let mut all: Vec<usize> = g.iter().flatten().copied().collect();
            all.sort();
            assert_eq!(all, (0..d.len()).collect::<Vec<_>>());
            assert!(n <= g.len() && g.iter().all(|x| !x.is_empty() && x.len() <= b && x.iter().all(|&m| d[m] == d[x[0]])));
        }
    }

    #[test]
    fn calib() {
        let c = PcmmCalib::parse("{\n \"b0.qkv\": 4.5,\n \"b0.down\": 486.0\n}").unwrap();
        assert_eq!(c.shift("b0.qkv", 1.5, 0.25).unwrap(), 5); // 27 -> 2^5
        assert_eq!(c.shift("b0.down", 1.5, 0.25).unwrap(), 12);
        assert!(c.shift("b1.qkv", 1.5, 0.25).is_err());
        assert_eq!(PcmmCalib::parse(&c.to_json()).unwrap(), c);
    }

    /// The pre-division's scale drop: the stamp w_scale 2^shift over 2^20, on a lean chain only, for band shifts above
    /// 1 (the served rule), never negative.
    #[test]
    fn prediv_scale() {
        let lin = |w_log2: i32, shift: u32| PcmmLinear { name: "b.out".into(), shape: PcmmShape { t: 64, n_in: 6, n_out: 2, nw: 2 }, w_scale: 2f64.powi(w_log2), shift };
        assert_eq!(prediv_log2(true, &lin(19, 5)), 4); // stamp 2^24
        assert_eq!(prediv_log2(true, &lin(19, 2)), 1); // 2^21 (b5-b8, b10 .out)
        assert_eq!(prediv_log2(true, &lin(19, 3)), 2); // 2^22 (b9.out)
        assert_eq!(prediv_log2(true, &lin(14, 4)), 0); // 2^18: below the design stamp
        assert_eq!(prediv_log2(true, &lin(21, 1)), 0); // band shift 1 (b1.up, 2^22): outside the rule
        assert_eq!(prediv_log2(false, &lin(19, 5)), 0); // a composite chain
        assert_eq!((PREDIV_SHIFT, PREDIV_STAMP_LOG2), (1, 20));
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
