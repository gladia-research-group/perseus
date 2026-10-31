//! Seam `LinearTransform`, implementation BSGS collapsed-FFT stage: one CtS or StC stage (FIDESlib
//! `LinearTransform` with hoisted baby-step rotations, `ltStep`) for K ciphertexts, bit-exact with FIDESlib's
//! single-GPU path, any giant-step count and a final offset rotation: ModUp(c1) once; `cts_fused` (baby rotations
//! times the stage's plaintexts, summed per giant step in Q u P); Horner over the giant steps; the stage end (offset
//! rotation, ModDown fused with the rescale, or plain ModDown) into the caller's buffer.
//!
//! Plaintexts: Q rows 0..=top then the special rows, one width `ptw` per stage (FIDESLIB_LT_COMPACT block-constant
//! rows). Step details: docs/ARCHITECTURE.md "pc-gpu CtS / StC stage".
use super::{BtsCx, BtsKeyRing};
use crate::bconv::{BconvPlan, JobSpec};
use crate::context::{Context, DevStrategy, DevWord, ScalarTable};
use crate::error::{shape, Result};
use crate::ks::{self, DotIo, DotRows, KeyGroup, KeyTabs, KsKeyRef, KskSource, Staging, View};
use crate::ntt::{self, Dir, RowList};
use crate::poly::{CtBuf, Layout};
use pc_core::bts_setup::C64;
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::modmul::{invmod, mulmod};
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, DBuf, Lane, Stream};
use pc_kernels::{CtsArgs, CtsRegenArgs, EwArgs, KskExpandArgs, KS_ACC};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::ffi::c_void;

/// Digits of a baby key whose `a` `cts_fused_regen` regenerates by default when the rest can be read (packed keys:
/// `KeyTails` from this digit on). Measured on GB202 and H100, every route at K = 1: 2 beats 3 and 4 on both; on
/// GB202 2 also beats 1 and 0 (reading `a` then costs more than regenerating it), on H100 each digit read gains.
pub const LT_REGEN_DIGITS: usize = 2;
/// Giant steps per `cts_fused` launch (kernel bound GMAX).
pub const LT_GMAX_LAUNCH: usize = 2;
/// Giant steps per K=1 launch of the queued u32 regen kernel (two pairs).
pub const LT_GMAX_LAUNCH_K1: usize = 4;
const _: () = assert!(LT_GMAX_LAUNCH <= ks::SRC_STRIDES, "the k > 1 ModUp reads the LT results in place");
/// Ciphertexts per `cts_fused` thread group by default (kernel KMAX 1/2/4/8; larger K runs several groups in one
/// launch): 0 = 1 for a single ciphertext, else 2 (tuned).
pub const LT_GROUP_AUTO: i32 = 0;

/// Geometry of one stage (FIDESlib `LTstep`: `bStep` baby steps, `gStep` giant steps, rotIn spacing, rotOut\[0\]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LtGeom {
    pub b: usize,
    pub g: usize,
    pub stride: usize,
    pub npts: usize,
    /// rotOut\[0\]: the rotation applied to the stage's sum (0 except where the accumulated offset is paid).
    pub offset: i64,
}

impl LtGeom {
    pub fn from_fides(st: &pc_core::bts_setup::FidesStage) -> LtGeom {
        let b = st.b_step as usize;
        let stride = if b > 1 { st.rot_in[1] - st.rot_in[0] } else { st.rot_out.get(1).copied().unwrap_or(0) - st.rot_out[0] };
        LtGeom { b, g: st.g_step as usize, stride: stride as usize, npts: st.num_diags, offset: st.rot_out[0] }
    }
    /// From a dumped stage's metadata (`boot.CtS[k]`: bStep, gStep, rotIn, rotOut, slots = #diagonals).
    pub fn from_dump(b: usize, g: usize, rot_in: &[i64], rot_out: &[i64], npts: usize) -> LtGeom {
        let stride = if b > 1 { rot_in[1] - rot_in[0] } else { rot_out.get(1).copied().unwrap_or(0) - rot_out[0] };
        LtGeom { b, g, stride: stride as usize, npts, offset: rot_out[0] }
    }
}

/// How a stage's sum ends (step 4 of the module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LtEnd {
    /// ModDown fused with the rescale: degree 1, top - d.
    FusedRescale,
    /// Plain ModDown: degree 2 at the input top.
    ModDown,
}

/// Where a stage's plaintexts come from.
pub enum PtSource<'v> {
    /// Device-layout rows per diagonal (Q rows 0..=top, `ptw` words each, then the special rows), compaction `enc`.
    Rows { rows: Vec<Vec<u32>>, ptw: usize, enc: u32 },
    /// Slot values per diagonal (length N/2 or 2 slots), encoded at `scale`.
    Slots { values: &'v [Vec<C64>], scale: f64 },
    /// Integer coefficients per diagonal (`diag_coeffs`).
    Coeffs(&'v [Vec<i128>]),
}

/// Exact integer coefficients of the diagonals' encodings at `scale` (OpenFHE FFTSpecialInv, round to nearest).
pub fn diag_coeffs(chain: &pc_core::params::Chain, values: &[Vec<C64>], scale: f64) -> Result<Vec<Vec<i128>>> {
    let enc = Encoder::new(chain);
    values
        .par_iter()
        .map(|v| {
            let c: Vec<Complex> = v.iter().map(|z| Complex::new(z.re, z.im)).collect();
            enc.slot_coeffs(&c, v.len(), Payload::Complex, scale).map_err(crate::error::Error::from)
        })
        .collect()
}

/// A stage whose giant rotation R (b stride slots) has an order `ord` below its giant-step count (R^ord = identity
/// mod N/2) with the giant steps folded mod `ord`: the diagonals of giant steps j and j + ord summed per baby step
/// (integer coefficients, so every prime's encoding is the sum of the unfolded encodings mod p), `ord` giant steps
/// left. `ord` = 1 (b stride = N/2: CtS stage 0): one product per baby step instead of `g` and no add of the extended
/// results, the same residues (bit-identical). `ord` > 1 (the StC stage of stride N/32 at b = 16: R^2 = 1): g - ord
/// fewer giant key switches, the Horner sums' extra rotations by R^ord being exact identities in the slots (not
/// bit-identical: fewer key-switch roundings).
pub fn fold_periodic_giants(n: usize, geom: LtGeom, coeffs: Vec<Vec<i128>>) -> (LtGeom, Vec<Vec<i128>>) {
    let nh = n / 2;
    let r = (geom.b * geom.stride) % nh;
    let ord = nh / gcd(r, nh);
    if geom.g <= ord {
        return (geom, coeffs);
    }
    let np = (ord * geom.b).min(geom.npts);
    let mut out: Vec<Vec<i128>> = Vec::with_capacity(np);
    for (dg, c) in coeffs.into_iter().enumerate() {
        match out.get_mut(dg % (ord * geom.b)) {
            Some(acc) => acc.iter_mut().zip(&c).for_each(|(a, &x)| *a += x),
            None => out.push(c),
        }
    }
    (LtGeom { g: ord, npts: np, ..geom }, out)
}

fn gcd(a: usize, b: usize) -> usize {
    if a == 0 {
        b
    } else {
        gcd(b % a, a)
    }
}

/// The stored width of a stage's plaintexts (every diagonal shares the stage's compaction).
pub fn stage_width(n: usize, coeffs: &[Vec<i128>]) -> usize {
    n >> coeffs.iter().map(|c| compact_shift(c)).min().unwrap_or(0)
}

/// A stage's plaintexts on the device.
pub struct LtPts<'a, W: DevWord> {
    buf: DBuf<'a, Compute, W>,
    ptrs: ConstBuf<'a, u64>,
    pub ptw: usize,
    pub enc: u32,
    /// Q rows stored per plaintext (= top + 1).
    pub prow: usize,
    pub npts: usize,
}

/// One linear-transform stage with its plaintexts, key tables and conversion plans (built for k_max cts).
pub struct LtStage<'a, W: DevWord> {
    pub geom: LtGeom,
    /// Input top limb (the plaintexts' level).
    pub top: usize,
    pub end: LtEnd,
    /// The plaintexts' scale (the product's NoiseFactor gains it).
    pub pt_nf: f64,
    pub pts: LtPts<'a, W>,
    /// `cts_fused` ciphertexts per thread group (LT_GROUP_AUTO or 1/2/4/8).
    pub group: i32,
    /// Where the baby keys' `a` comes from (`set_baby`): Regen = `cts_fused_regen` (seeded `a` in-kernel, `b` read in
    /// its stored form, dense / packed / banded), Cached = `cts_fused` on dense full-layout keys with resident `a`.
    baby: KskSource,
    /// Regen: digits below regenerate `a`, the rest read the resident `a` (needs Cached-capable keys; usize::MAX = all).
    regen_digits: usize,
    k_max: usize,
    /// baby keys: kb pointers (b - 1), ka pointers (b - 1); kinv = 5^(i stride) mod 2N
    kb: ConstBuf<'a, u64>,
    ka: ConstBuf<'a, u64>,
    kinv: ConstBuf<'a, u32>,
    /// baby keys read in place (all seeded, one form), and whether the Cached kernel can read them
    tabs: Option<KeyTabs<'a>>,
    cached_ok: bool,
    /// Every baby key has a resident tail from this digit on (`KeyTails`; `ka` points at it, shifted by `first` digits).
    tail: Option<usize>,
    rows: ConstBuf<'a, u32>,
    digit_of: ConstBuf<'a, i32>,
    giant: Option<KeyGroup<'a>>,
    offset_key: Option<KeyGroup<'a>>,
    /// ModDown of c1 of res_j (j = 1..g-1, and j = 0 when there is an offset rotation): per j
    md_c1: Vec<Option<MdPlan<'a>>>,
    /// stage end on res_0 (FusedRescale / ModDown without offset) or on lt_rot (offset)
    md_end: MdPlan<'a>,
    /// B^-1 of the stage end's conversion by prime id (B = P or P q_top-1 q_top)
    binv: ScalarTable<'a, W>,
    bytes: usize,
}

/// One ModDown-like conversion over k_max cts: INTT of the source rows (in the source buffer), bconv into the
/// target buffer, NTT of the target rows there.
pub(crate) struct MdPlan<'a> {
    intt: RowList<'a>,
    /// `N^-1 pre_t` of every `intt` row (the conversion's pre factors folded into the INTT), then Shoup, W words each.
    iscale: ConstBuf<'a, u8>,
    conv: BconvPlan<'a>,
    ntt: RowList<'a>,
    /// entries per ciphertext of intt / ntt (prefix unit), jobs per ciphertext
    intt_per: usize,
    ntt_per: usize,
    jobs_per: usize,
}

impl MdPlan<'_> {
    pub(crate) fn run<W: DevWord, M: DevStrategy<W>>(&self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, src: &mut DBuf<'_, Compute, W>, dst: &mut DBuf<'_, Compute, W>, k: usize) -> Result<()> {
        self.run_epi(ctx, s, src, dst, k, None)
    }
    /// `run` with the target NTT's epilogue (`ntt::Epi`, its buffers at the target rows' offsets).
    fn run_epi<W: DevWord, M: DevStrategy<W>>(&self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, src: &mut DBuf<'_, Compute, W>, dst: &mut DBuf<'_, Compute, W>, k: usize, epi: Option<ntt::Epi>) -> Result<()> {
        let v = self.iscale.vptr();
        let sc = (v, (v as *const W).wrapping_add(self.intt.len()) as *const c_void);
        ntt::inverse_rows_prefix_scaled(ctx, s, src, &self.intt, k * self.intt_per, None, sc)?;
        self.conv.run_prefix(ctx, s, src, 0, dst, 0, k * self.jobs_per)?;
        ntt::transform_rows_prefix_ex(ctx, s, Dir::Forward, dst, &self.ntt, k * self.ntt_per, None, epi)
    }
}

/// Position of result (c, j) in the `lt_res` buffer, in ciphertexts: giant steps in launch pairs, each pair a
/// region of `k_max * 2` cts ordered (c, j % 2).
fn res_ct(k_max: usize, g: usize, c: usize, j: usize) -> usize {
    let l = j / LT_GMAX_LAUNCH;
    let gl = (g - l * LT_GMAX_LAUNCH).min(LT_GMAX_LAUNCH);
    l * k_max * LT_GMAX_LAUNCH + c * gl + j % LT_GMAX_LAUNCH
}

fn res_stride(g: usize, j: usize) -> usize {
    let l = j / LT_GMAX_LAUNCH;
    (g - l * LT_GMAX_LAUNCH).min(LT_GMAX_LAUNCH)
}

/// The ModDown plan of `parts` of the cts at `cts[c]` (ciphertext indices in the source buffer, written at the same
/// indices of the target buffer): sources `src_rows` -> targets `0..ntgt`.
pub(crate) fn md_plan<'a, S: Lane, W: DevWord>(arena: &'a Arena, s: &Stream<S>, lay: &Layout, primes: &[u64], cts: &[usize], parts: &[usize], src_rows: &[usize], ntgt: usize, exact: bool) -> Result<MdPlan<'a>> {
    let rows = lay.rows();
    let (mut intt, mut specs, mut nttr) = (Vec::new(), Vec::new(), Vec::new());
    for &c in cts {
        for &p in parts {
            let base = (2 * c + p) * rows;
            for &t in src_rows {
                intt.push(((base + t) as u32, t as i32));
            }
            for t in 0..ntgt {
                nttr.push(((base + t) as u32, t as i32));
            }
            specs.push(JobSpec {
                src_off: base as u32,
                dst_off: base as u32,
                src: src_rows.iter().map(|&t| (t as u32, t as u32)).collect(),
                tgt: (0..ntgt).map(|t| (t as u32, t as u32)).collect(),
                exact,
                prescaled: true,
            });
        }
    }
    let np = parts.len();
    let pre = crate::bconv::pre_factors(&src_rows.iter().map(|&t| primes[t]).collect::<Vec<_>>());
    let ipre: Vec<u64> = (0..intt.len()).map(|e| pre[e % src_rows.len()]).collect();
    Ok(MdPlan {
        iscale: ConstBuf::new(arena, "lt.md", s, &ks::scale_bytes::<W>(primes, lay.n, &intt, &ipre))?,
        intt: RowList::new(arena, s, &intt)?,
        conv: BconvPlan::new::<S, W>(arena, s, primes, &specs)?,
        ntt: RowList::new(arena, s, &nttr)?,
        intt_per: np * src_rows.len(),
        ntt_per: np * ntgt,
        jobs_per: np,
    })
}

pub(crate) fn md_plan_bytes<W: DevWord>(primes: &[u64], ncts: usize, nparts: usize, nsrc: usize, ntgt: usize, exact: bool) -> Result<usize> {
    let spec = JobSpec { src_off: 0, dst_off: 0, src: (0..nsrc).map(|t| (t as u32, t as u32)).collect(), tgt: (0..ntgt).map(|t| (t as u32, t as u32)).collect(), exact, prescaled: false };
    let specs = vec![spec; ncts * nparts];
    let rl = |n: usize| Arena::bytes_for::<u32>(n) + Arena::bytes_for::<i32>(n);
    let sc = Arena::bytes_for::<u8>(2 * ncts * nparts * nsrc * std::mem::size_of::<W>());
    Ok(rl(ncts * nparts * nsrc) + rl(ncts * nparts * ntgt) + sc + BconvPlan::bytes::<W>(primes, &specs)?)
}

/// Compaction shift of a diagonal's integer coefficients: the largest s <= 12 with every nonzero coefficient at an
/// index divisible by 2^s (then the NTT row is constant on blocks of 2^s and stored at width N / 2^s).
fn compact_shift(c: &[i128]) -> u32 {
    let mut s = 12u32;
    for (i, &x) in c.iter().enumerate() {
        if x != 0 && i != 0 {
            s = s.min(i.trailing_zeros());
            if s == 0 {
                break;
            }
        }
    }
    s
}

/// Negacyclic NTT in OpenFHE order (bit-reversed output) of size `n` with root `psi` (host, compact rows).
fn host_ntt(a: &mut [u64], p: u64, psi: u64) {
    let n = a.len();
    let logn = n.trailing_zeros();
    let mut pw = vec![1u64; n];
    for i in 1..n {
        pw[i] = mulmod(pw[i - 1], psi, p);
    }
    let br = |i: usize| if logn == 0 { 0 } else { (i as u32).reverse_bits() as usize >> (32 - logn) };
    let (mut t, mut m) = (n, 1);
    while m < n {
        t /= 2;
        for i in 0..m {
            let w = pw[br(m + i)];
            for j in 2 * i * t..2 * i * t + t {
                let u = a[j];
                let v = mulmod(a[j + t], w, p);
                a[j] = if u + v >= p { u + v - p } else { u + v };
                a[j + t] = if u >= v { u - v } else { u + p - v };
            }
        }
        m *= 2;
    }
}

impl<'a, W: DevWord> LtPts<'a, W> {
    /// Device bytes of `npts` plaintexts of `rows` rows at width `ptw`.
    pub fn bytes(npts: usize, rows: usize, ptw: usize) -> usize {
        Arena::bytes_for::<W>(npts * rows * ptw) + Arena::bytes_for::<u64>(npts.max(1))
    }

    /// `bytes` of a `PtSource::Coeffs` upload of width `ptw` in a ring of dimension `n`: full-width rows are NTT'd on
    /// the device through a row list carved with them.
    pub fn coeffs_bytes(npts: usize, rows: usize, ptw: usize, n: usize) -> usize {
        let rl = if ptw == n { Arena::bytes_for::<u32>(npts * rows) + Arena::bytes_for::<i32>(npts * rows) } else { 0 };
        Self::bytes(npts, rows, ptw) + rl
    }

    /// Upload / encode the plaintexts of a stage at input top `top` (rows: Q 0..=top, then every special prime).
    pub fn new<M: DevStrategy<W>>(arena: &'a Arena, s: &Stream<Compute>, ctx: &Context<'_, W, M>, top: usize, src: PtSource<'_>) -> Result<Self> {
        let lay = ctx.lay;
        let n = lay.n;
        let prow = top + 1;
        let nrows = prow + lay.kp;
        let ids: Vec<usize> = (0..=top).chain(lay.nq..lay.rows()).collect();
        let primes: Vec<u64> = ids.iter().map(|&i| ctx.prime(i)).collect();
        match src {
            PtSource::Rows { rows, ptw, enc } => {
                let npts = rows.len();
                let mut buf = arena.alloc::<Compute, W>("bts.pts", npts * nrows * ptw)?;
                for (j, r) in rows.iter().enumerate() {
                    if r.len() != nrows * ptw {
                        return Err(shape("LtPts rows", format!("plaintext {j}: {} words, want {nrows} rows x {ptw}", r.len())));
                    }
                    let w: Vec<W> = r.iter().map(|&x| W::from_u64(x as u64)).collect();
                    buf.upload_at(s, j * nrows * ptw, &w)?;
                }
                Self::finish(arena, s, buf, npts, nrows, ptw, enc, prow)
            }
            PtSource::Slots { values, scale } => {
                let coeffs = diag_coeffs(&ctx.chain, values, scale)?;
                Self::new(arena, s, ctx, top, PtSource::Coeffs(&coeffs))
            }
            PtSource::Coeffs(coeffs) => {
                let npts = coeffs.len();
                let sh = coeffs.iter().map(|c| compact_shift(c)).min().unwrap_or(0);
                let ptw = n >> sh;
                let encm = if sh == 0 { 0xFFFF_FFFFu32 } else { 0x7F00_0000 | sh };
                let mut buf = arena.alloc::<Compute, W>("bts.pts", npts * nrows * ptw)?;
                if sh == 0 {
                    // full width: coefficient rows uploaded in place, NTT'd on the device
                    for (j, c) in coeffs.iter().enumerate() {
                        let w: Vec<W> = primes
                            .par_iter()
                            .flat_map_iter(|&p| c.iter().map(move |&x| W::from_u64(x.rem_euclid(p as i128) as u64)))
                            .collect();
                        buf.upload_at(s, j * nrows * n, &w)?;
                    }
                    let list: Vec<(u32, i32)> = (0..npts).flat_map(|j| ids.iter().enumerate().map(move |(r, &id)| ((j * nrows + r) as u32, id as i32))).collect();
                    let rl = RowList::new(arena, s, &list)?;
                    ntt::transform_rows(ctx, s, Dir::Forward, &mut buf, &rl)?;
                } else {
                    let roots = ctx.chain.roots();
                    let psis: Vec<u64> = ids.iter().map(|&i| (0..sh).fold(roots[i], |r, _| mulmod(r, r, ctx.prime(i)))).collect();
                    for (j, c) in coeffs.iter().enumerate() {
                        let small: Vec<i128> = (0..ptw).map(|i| c[i << sh]).collect();
                        let w: Vec<W> = primes
                            .par_iter()
                            .zip(psis.par_iter())
                            .flat_map_iter(|(&p, &psi)| {
                                let mut row: Vec<u64> = small.iter().map(|&x| x.rem_euclid(p as i128) as u64).collect();
                                host_ntt(&mut row, p, psi);
                                row.into_iter().map(W::from_u64)
                            })
                            .collect();
                        buf.upload_at(s, j * nrows * ptw, &w)?;
                    }
                }
                Self::finish(arena, s, buf, npts, nrows, ptw, encm, prow)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finish<S: Lane>(arena: &'a Arena, s: &Stream<S>, buf: DBuf<'a, Compute, W>, npts: usize, nrows: usize, ptw: usize, enc: u32, prow: usize) -> Result<Self> {
        let base = buf.dptr() as u64;
        let ws = std::mem::size_of::<W>() as u64;
        let p: Vec<u64> = (0..npts.max(1)).map(|j| base + (j * nrows * ptw) as u64 * ws).collect();
        Ok(LtPts { ptrs: ConstBuf::new(arena, "bts.pts", s, &p)?, buf, ptw, enc, prow, npts })
    }

    /// The plaintext rows (`[pt][Q rows 0..=top, then the special rows][ptw]`), for stages rebuilt in place (StC-first's
    /// re-levelled StC-first / stc_high caches).
    pub fn rows_mut(&mut self) -> &mut DBuf<'a, Compute, W> {
        &mut self.buf
    }

    /// Host copy of plaintext j's rows (checks).
    pub fn download(&self, s: &Stream<Compute>, j: usize) -> Result<Vec<W>> {
        let words = self.buf.len() / self.npts.max(1);
        Ok(self.buf.download_range(s, j * words, words)?)
    }
}

/// Resident `a` of the last digits of LT baby keys stored without one (packed, seeded): `cts_fused_regen` regenerates
/// digits `0..first` and reads `first..` (`LT_REGEN_DIGITS`: the measured best mix). Expanded at setup from
/// each key's seed, one buffer per rotation step (`[digit - first][prime id][N]`, full keys only).
pub struct KeyTails<'a, W: DevWord> {
    first: usize,
    tails: BTreeMap<i64, DBuf<'a, Compute, W>>,
}

/// One key with its resident tail (`KeyTails::ring`).
struct TailKey<'r> {
    key: &'r dyn KsKeyRef,
    a: *const c_void,
    first: usize,
}

impl KsKeyRef for TailKey<'_> {
    fn b_ptr(&self) -> *const c_void {
        self.key.b_ptr()
    }
    fn a_ptr(&self) -> Option<*const c_void> {
        self.key.a_ptr()
    }
    fn seed_ptr(&self) -> Option<*const u32> {
        self.key.seed_ptr()
    }
    fn galois(&self) -> u64 {
        self.key.galois()
    }
    fn digit_stride(&self) -> usize {
        self.key.digit_stride()
    }
    fn dnum(&self) -> usize {
        self.key.dnum()
    }
    fn q_band(&self) -> usize {
        self.key.q_band()
    }
    fn staging(&self) -> Option<Staging> {
        self.key.staging()
    }
    fn a_tail(&self) -> Option<(*const c_void, usize)> {
        Some((self.a, self.first))
    }
}

/// A key ring whose rotation keys carry their tails (what `LtStage::new` reads); every other key as in `inner`.
pub struct TailRing<'r> {
    inner: &'r dyn BtsKeyRing,
    keys: BTreeMap<i64, TailKey<'r>>,
    half: i64,
}

impl BtsKeyRing for TailRing<'_> {
    fn rot(&self, step: i64) -> Result<&dyn KsKeyRef> {
        match self.keys.get(&step.rem_euclid(self.half)) {
            Some(k) => Ok(k),
            None => self.inner.rot(step),
        }
    }
    fn conj(&self) -> Result<&dyn KsKeyRef> {
        self.inner.conj()
    }
    fn relin(&self) -> Result<&dyn KsKeyRef> {
        self.inner.relin()
    }
    fn dense_to_sparse(&self) -> Result<&dyn KsKeyRef> {
        self.inner.dense_to_sparse()
    }
    fn sparse_to_dense(&self) -> Result<&dyn KsKeyRef> {
        self.inner.sparse_to_dense()
    }
}

/// The baby steps of `stages` (geometry, input top) whose keys a tail from digit `first` serves: stages that regenerate
/// more than `first` digits (`LtStage::regen_now`: K = 1 at >= 5 digits).
pub fn tail_steps(chain: &pc_core::params::Chain, stages: &[(LtGeom, usize)], first: usize) -> Vec<i64> {
    let nh = (chain.n / 2) as i64;
    let ndig = |top: usize| chain.digits.iter().filter(|d| d.first().is_some_and(|&f| f <= top)).count();
    let mut v: Vec<i64> = stages
        .iter()
        .filter(|(_, top)| ndig(*top) >= 5 && ndig(*top) > first)
        .flat_map(|(g, _)| (1..g.b).map(move |i| ((i * g.stride) as i64) % nh))
        .filter(|&r| r != 0)
        .collect();
    v.sort();
    v.dedup();
    v
}

impl<'a, W: DevWord> KeyTails<'a, W> {
    /// Arena bytes of the tails of `nkeys` full keys of `dnum` digits.
    pub fn bytes(lay: &Layout, dnum: usize, first: usize, nkeys: usize) -> usize {
        if nkeys == 0 || first >= dnum {
            return 0;
        }
        nkeys * Arena::bytes_for::<W>((dnum - first) * lay.poly_words()) + Arena::bytes_for::<i32>(lay.rows())
    }
    /// Expand digits `first..` of the rotation keys of `steps` (full seeded keys: what `bytes` sizes; a stage whose keys
    /// have a resident dense `a` reads that one instead).
    pub fn new<M: DevStrategy<W>>(arena: &'a Arena, s: &Stream<Compute>, ctx: &Context<'_, W, M>, keys: &dyn BtsKeyRing, steps: &[i64], first: usize) -> Result<Self> {
        let lay = ctx.lay;
        let nh = (lay.n / 2) as i64;
        let mut tails = BTreeMap::new();
        if steps.is_empty() {
            return Ok(KeyTails { first, tails });
        }
        let ids: Vec<i32> = (0..lay.rows() as i32).collect();
        let ids = ConstBuf::new(arena, "bts.tails", s, &ids)?;
        for &st in steps {
            let st = st.rem_euclid(nh);
            if st == 0 || tails.contains_key(&st) {
                continue;
            }
            let key = keys.rot(st)?;
            let ndig = key.dnum();
            let Some(seed) = key.seed_ptr() else { continue };
            if key.q_band() + 1 < lay.nq || first >= ndig {
                continue;
            }
            let mut buf = arena.alloc::<Compute, W>("bts.tails", (ndig - first) * lay.poly_words())?;
            let a = KskExpandArgs { out: buf.vptr_mut(), seed, prime_ids: ids.dptr(), pr: ctx.pr(), nmod: lay.rows() as i32, dnum: ndig as i32, d0: first as i32, pad: 0 };
            // SAFETY: buf holds (ndig - first) full-layout polys; the seed is the key's 8-word device buffer
            launch_result("ksk_expand", unsafe { (ctx.table.ksk_expand)(&a, s.raw()) })?;
            tails.insert(st, buf);
        }
        Ok(KeyTails { first, tails })
    }
    /// `keys` with these tails attached.
    pub fn ring<'r>(&'r self, keys: &'r dyn BtsKeyRing, n: usize) -> Result<TailRing<'r>> {
        let steps: Vec<i64> = self.tails.keys().copied().collect();
        self.ring_of(keys, n, &steps)
    }
    /// `keys` with the tails of `steps` attached (tails shared by several stages: each attaches its own steps).
    pub fn ring_of<'r>(&'r self, keys: &'r dyn BtsKeyRing, n: usize, steps: &[i64]) -> Result<TailRing<'r>> {
        let nh = (n / 2) as i64;
        let mut m = BTreeMap::new();
        for st in steps {
            if let Some(buf) = self.tails.get(&st.rem_euclid(nh)) {
                m.insert(st.rem_euclid(nh), TailKey { key: keys.rot(*st)?, a: buf.vptr(), first: self.first });
            }
        }
        Ok(TailRing { inner: keys, keys: m, half: nh })
    }
}

impl<'a, W: DevWord> LtStage<'a, W> {
    /// Device bytes of a stage's tables besides its plaintexts (plans for k_max cts).
    pub fn plan_bytes(lay: &Layout, primes: &[u64], geom: &LtGeom, top: usize, end: LtEnd, k_max: usize) -> Result<usize> {
        let kp = lay.kp;
        let mut b = 2 * Arena::bytes_for::<u64>(geom.b.max(2)) + Arena::bytes_for::<u32>(geom.b.max(2)) + KeyTabs::bytes(geom.b.max(2));
        b += Arena::bytes_for::<u32>(top + 1 + kp) + Arena::bytes_for::<i32>(lay.rows());
        // giant key only when the giant step is not the identity (b stride = N/2: CtS stage 0), offset key only when
        // there is an offset; one ModDown plan per non-identity giant step and one for the offset (as `new` builds)
        let nh = (lay.n / 2) as i64;
        let giant = geom.g > 1 && (geom.b * geom.stride) as i64 % nh != 0;
        let off = geom.offset.rem_euclid(nh) != 0;
        b += (giant as usize + off as usize) * KeyGroup::bytes(1, true);
        let nmd = if giant { geom.g - 1 } else { 0 } + off as usize;
        b += nmd * md_plan_bytes::<W>(primes, k_max, 1, kp, top + 1, false)?;
        b += match (geom.offset != 0, end) {
            (true, _) | (false, LtEnd::ModDown) => md_plan_bytes::<W>(primes, k_max, 2, kp, top + 1, false)?,
            (false, LtEnd::FusedRescale) => md_plan_bytes::<W>(primes, k_max, 2, kp + 2, top - 1, true)?,
        };
        b += 2 * Arena::bytes_for::<W>(lay.rows());
        Ok(b)
    }

    /// Build a stage at input top `top`. Baby / giant / offset rotation keys come from `keys`. The babies run Regen
    /// (`cts_fused_regen`) when every baby key is seeded and shares one stored form, else Cached (dense full-layout
    /// keys with a resident `a`); `set_baby` overrides.
    #[allow(clippy::too_many_arguments)]
    pub fn new<M: DevStrategy<W>>(
        arena: &'a Arena,
        s: &Stream<Compute>,
        ctx: &Context<'_, W, M>,
        geom: LtGeom,
        top: usize,
        end: LtEnd,
        pt_nf: f64,
        pts: LtPts<'a, W>,
        keys: &dyn BtsKeyRing,
        stage_ptrs: Option<(*const c_void, *const c_void)>,
        k_max: usize,
    ) -> Result<Self> {
        let lay = ctx.lay;
        let n = lay.n;
        let nh = (n / 2) as i64;
        let primes = ctx.chain.primes();
        let d = ctx.chain.level_unit;
        if pts.prow != top + 1 || pts.npts != geom.npts {
            return Err(shape("LtStage", format!("plaintexts hold {} Q rows / {} diagonals, stage wants {} / {}", pts.prow, pts.npts, top + 1, geom.npts)));
        }
        if geom.npts > geom.b * geom.g || geom.b == 0 || geom.g == 0 {
            return Err(shape("LtStage", format!("geometry {geom:?}")));
        }
        if end == LtEnd::FusedRescale && (d != 2 || top < 2) {
            return Err(shape("LtStage", "the fused ModDown-rescale is a composite-degree-2 path"));
        }
        let mut kb = Vec::new();
        let mut ka = Vec::new();
        let mut kinv = Vec::new();
        let mut babies: Vec<&dyn crate::ks::KsKeyRef> = Vec::new();
        let mut cached_ok = true;
        let mut tails: Vec<Option<(*const c_void, usize)>> = Vec::new();
        for i in 1..geom.b {
            let r = (i * geom.stride) as i64 % nh;
            let key = keys.rot(r)?;
            let a = key.a_ptr();
            cached_ok &= key.staging().is_none() && a.is_some() && key.digit_stride() == lay.poly_words();
            kb.push(key.b_ptr() as u64);
            ka.push(a.map(|p| p as u64).unwrap_or(0));
            kinv.push(pc_core::galois::rotation_galois(r, n)? as u32);
            tails.push(key.a_tail());
            babies.push(key);
        }
        // every baby with a tail from the same digit: the kernel reads digit d >= first at ka + d * poly_words, so ka is
        // the tail moved back by `first` digits (never dereferenced below the tail)
        let tail = match tails.first().copied().flatten() {
            Some((_, f)) if !cached_ok && tails.iter().all(|t| t.is_some_and(|t| t.1 == f)) => {
                let back = (f * lay.poly_words() * std::mem::size_of::<W>()) as u64;
                for (j, t) in tails.iter().enumerate() {
                    ka[j] = (t.map(|t| t.0).unwrap_or(std::ptr::null()) as u64).wrapping_sub(back);
                }
                Some(f)
            }
            _ => None,
        };
        let tabs = if babies.is_empty() { None } else { KeyTabs::new(arena, s, n, &babies)? };
        let tabs = tabs.filter(|t| t.run_len(0, babies.len()) == babies.len() && t.q_top() >= top);
        if kb.is_empty() {
            kb.push(0);
            ka.push(0);
            kinv.push(1);
        }
        let baby = if tabs.is_some() || geom.b == 1 {
            KskSource::Regen
        } else if cached_ok {
            KskSource::Cached
        } else {
            return Err(shape("LtStage", "baby keys: neither seeded in one stored form covering the stage (Regen) nor dense full-layout with a resident `a` (Cached)"));
        };
        // the shifted tail pointers are read only above the regenerated digits (cts_fused_regen; tails need !cached_ok)
        debug_assert!(tail.is_none() || baby == KskSource::Regen, "baby-key tails need the Regen kernel");
        let giant_r = (geom.b * geom.stride) as i64 % nh;
        let giant = if geom.g > 1 && giant_r != 0 { Some(KeyGroup::new(arena, s, n, &[keys.rot(giant_r)?], stage_ptrs)?) } else { None };
        let off = geom.offset.rem_euclid(nh);
        let offset_key = if off != 0 { Some(KeyGroup::new(arena, s, n, &[keys.rot(off)?], stage_ptrs)?) } else { None };
        let rows: Vec<u32> = (0..=top).chain(lay.nq..lay.rows()).map(|t| t as u32).collect();
        let digit_of: Vec<i32> = (0..lay.rows()).map(|t| ctx.chain.digit_of(t).map(|d| d as i32).unwrap_or(-1)).collect();
        let specials: Vec<usize> = (lay.nq..lay.rows()).collect();
        let mut md_c1 = Vec::with_capacity(geom.g);
        for j in 0..geom.g {
            let need = (j > 0 && giant_r != 0) || (j == 0 && off != 0);
            md_c1.push(if need {
                let cts: Vec<usize> = (0..k_max).map(|c| res_ct(k_max, geom.g, c, j)).collect();
                Some(md_plan::<_, W>(arena, s, &lay, &primes, &cts, &[1], &specials, top + 1, false)?)
            } else {
                None
            });
        }
        let (md_end, bvals) = if off != 0 {
            let cts: Vec<usize> = (0..k_max).collect();
            (md_plan::<_, W>(arena, s, &lay, &primes, &cts, &[0, 1], &specials, top + 1, false)?, None)
        } else {
            let cts: Vec<usize> = (0..k_max).map(|c| res_ct(k_max, geom.g, c, 0)).collect();
            match end {
                LtEnd::ModDown => (md_plan::<_, W>(arena, s, &lay, &primes, &cts, &[0, 1], &specials, top + 1, false)?, None),
                LtEnd::FusedRescale => {
                    let mut src = specials.clone();
                    src.push(top - 1);
                    src.push(top);
                    let mut bv = vec![0u64; lay.rows()];
                    for t in 0..top - 1 {
                        let q = primes[t];
                        let b = src.iter().fold(1u64, |a, &x| mulmod(a, primes[x] % q, q));
                        bv[t] = invmod(b, q);
                    }
                    (md_plan::<_, W>(arena, s, &lay, &primes, &cts, &[0, 1], &src, top - 1, true)?, Some(bv))
                }
            }
        };
        let binv = match bvals {
            Some(bv) => ScalarTable::new(arena, "bts.lt", s, &primes, &bv)?,
            None => {
                let mut bv = vec![0u64; lay.rows()];
                for t in 0..=top {
                    let q = primes[t];
                    bv[t] = invmod(specials.iter().fold(1u64, |a, &x| mulmod(a, primes[x] % q, q)), q);
                }
                ScalarTable::new(arena, "bts.lt", s, &primes, &bv)?
            }
        };
        let bytes = Self::plan_bytes(&lay, &primes, &geom, top, end, k_max)? + LtPts::<W>::bytes(pts.npts, pts.prow + lay.kp, pts.ptw);
        Ok(LtStage {
            geom,
            top,
            end,
            pt_nf,
            pts,
            group: LT_GROUP_AUTO,
            baby,
            regen_digits: LT_REGEN_DIGITS,
            k_max,
            kb: ConstBuf::new(arena, "bts.lt", s, &kb)?,
            ka: ConstBuf::new(arena, "bts.lt", s, &ka)?,
            kinv: ConstBuf::new(arena, "bts.lt", s, &kinv)?,
            tabs,
            cached_ok,
            tail,
            rows: ConstBuf::new(arena, "bts.lt", s, &rows)?,
            digit_of: ConstBuf::new(arena, "bts.lt", s, &digit_of)?,
            giant,
            offset_key,
            md_c1,
            md_end,
            binv,
            bytes,
        })
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Choose the baby-key path (see the field `baby`); refuses a source the stage's keys cannot serve.
    pub fn set_baby(&mut self, src: KskSource) -> Result<()> {
        let ok = self.geom.b == 1 || match src {
            KskSource::Regen => self.tabs.is_some(),
            KskSource::Cached => self.cached_ok,
        };
        if !ok {
            return Err(shape("LtStage::set_baby", format!("baby keys cannot run {src:?} (Regen: seeded, one stored form covering top {}; Cached: dense full layout with resident `a`)", self.top)));
        }
        self.baby = src;
        Ok(())
    }
    pub fn baby(&self) -> KskSource {
        self.baby
    }
    /// Whether a launch of `k` cts at `ndig` digits regenerates `a` (`cts_fused_regen`). Keys without a resident dense
    /// `a` always do. Otherwise Regen is a policy: the gather-form regeneration is ChaCha-bound (measured: all six
    /// digits regenerated is +0.4 ms on a K=1 CtS stage, +6 ms at K=8 where the cached kernel's groups share key reads
    /// through L2), so it runs only for K = 1 at >= 5 digits with `regen_digits` (default `LT_REGEN_DIGITS`) regenerated
    /// and the rest read.
    fn regen_now(&self, k: usize, ndig: usize) -> bool {
        match self.baby {
            _ if self.tabs.is_some() && !self.cached_ok => true,
            KskSource::Regen => self.tabs.is_some() && k == 1 && ndig >= 5,
            KskSource::Cached => false,
        }
    }

    /// Regen with only the first `n` digits regenerated, the rest read from the resident `a` (keys without one
    /// regenerate every digit); `usize::MAX` = every digit.
    pub fn set_regen_digits(&mut self, n: usize) -> Result<()> {
        self.regen_digits = n;
        Ok(())
    }

    /// Output top limb and degree of the stage (FIDESlib coordinates).
    pub fn out_top_deg(&self, d: usize) -> (usize, u8) {
        if self.geom.offset == 0 && self.end == LtEnd::FusedRescale {
            (self.top - d, 1)
        } else {
            (self.top, 2)
        }
    }

    /// Run the stage on the first `k` cts of `x` (degree 1 at `self.top`, evaluation form); the result replaces them.
    pub fn run<M: DevStrategy<W>>(&self, cx: &mut BtsCx<'_, '_, W, M>, x: &mut CtBuf<'_, Compute, W>, k: usize, source: KskSource) -> Result<()> {
        let ctx = cx.ctx;
        let lay = ctx.lay;
        let g = self.geom.g;
        let top = self.top;
        if k == 0 || k > self.k_max || k > x.k {
            return Err(shape("LtStage::run", format!("{k} cts (k_max {}, buffer {})", self.k_max, x.k)));
        }
        let need_res = res_ct(self.k_max, g, self.k_max - 1, g - 1) + 1;
        if cx.work.lt_res.k < need_res || cx.work.lt_tmp.k < need_res || cx.work.lt_rot.k < k {
            return Err(shape("LtStage::run", format!("bts.work holds {} LT cts, the stage needs {need_res} (BtsWorkShape::g_max)", cx.work.lt_res.k)));
        }
        let cw = lay.ct_words();
        let pw = lay.poly_words() as u64;
        // 1. ModUp(c1)
        {
            let ex = &mut *cx.ex;
            ks::modup(ctx, &ex.compute, cx.ks, &mut ex.ks, &mut ex.bconv_work, x, 0, k, 1, top)?;
        }
        // 2. fused baby rotations x plaintexts, two giant steps per launch
        let ndig = cx.ks.ndig(top)?;
        {
            let ex = &*cx.ex;
            let res_base = cx.work.lt_res.buf.vptr_mut() as *mut W;
            // a u32 K=1 regen launch carries up to 4 giant steps (StC stages, g = 4) so the baby keys are regenerated
            // once; its second pair lands in its own region (`pair_cts`)
            let regen = self.regen_now(k, ndig) && self.tabs.is_some();
            let wide = regen && k == 1 && self.group <= 1 && std::mem::size_of::<W>() == 4;
            let per_launch = if wide { LT_GMAX_LAUNCH_K1 } else { LT_GMAX_LAUNCH };
            let mut j0 = 0;
            while j0 < g {
                let gl = (g - j0).min(per_launch);
                let region = res_ct(self.k_max, g, 0, j0);
                let pts_off = j0 * self.geom.b;
                let a = CtsArgs {
                    x: x.buf.vptr(),
                    modup: ex.ks.digits.vptr(),
                    res: res_base.wrapping_add(region * cw) as *mut c_void,
                    kb: self.kb.dptr() as *const *const c_void,
                    ka: self.ka.dptr() as *const *const c_void,
                    kinv: self.kinv.dptr(),
                    pts: (self.pts.ptrs.dptr() as *const u8).wrapping_add(pts_off * 8) as *const *const c_void,
                    rows: self.rows.dptr(),
                    digit_of: self.digit_of.dptr(),
                    pmod: ctx.ks.p_mod_q.v.vptr(),
                    pmods: ctx.ks.p_mod_q.s.vptr(),
                    pr: ctx.pr(),
                    ct_stride: cw as u64,
                    part_stride: pw,
                    mu_ct_stride: (cx.ks.dnum * lay.poly_words()) as u64,
                    mu_digit_stride: pw,
                    key_digit_stride: pw,
                    k: k as i32,
                    b: self.geom.b as i32,
                    g: gl as i32,
                    npts: (self.geom.npts - pts_off) as i32,
                    ndig: ndig as i32,
                    nq: lay.nq as i32,
                    nlive: (top + 1) as i32,
                    ptw: self.pts.ptw as i32,
                    pt_prow: self.pts.prow as i32,
                    enc: self.pts.enc,
                    ngrp: 1,
                    nrows: (top + 1 + lay.kp) as i32,
                    kmax: if self.group > 0 { self.group } else if k == 1 { 1 } else { 2 },
                    pair_cts: if gl > LT_GMAX_LAUNCH { (self.k_max * LT_GMAX_LAUNCH) as i32 } else { 0 },
                };
                match (self.regen_now(k, ndig), &self.tabs) {
                    (true, Some(t)) => {
                        let mut key = t.tab(0)?;
                        key.nregen = match self.tail {
                            _ if self.cached_ok => self.regen_digits.min(i32::MAX as usize) as i32,
                            // the digits below the tail regenerate (`regen_digits` above it: more regenerate)
                            Some(f) => self.regen_digits.max(f).min(i32::MAX as usize) as i32,
                            None => i32::MAX,
                        };
                        let ra = CtsRegenArgs { cts: a, key };
                        // SAFETY: as below; the key tables (seeds, stored `b`) were built at setup from live keys
                        launch_result("cts_fused_regen", unsafe { (ctx.table.cts_fused_regen)(&ra, ex.compute.raw()) })?;
                    }
                    _ => {
                        // SAFETY: x (k cts), the ModUp digits (k cts), the result region (k * gl cts inside lt_res,
                        // checked above), key and plaintext tables built at setup for this stage; outputs never alias
                        // the inputs
                        launch_result("cts_fused", unsafe { (ctx.table.cts_fused)(&a, ex.compute.raw()) })?;
                    }
                }
                j0 += gl;
            }
        }
        // 3. giant steps (Horner from the last)
        let res_view = |j: usize, buf: &CtBuf<'_, Compute, W>| View {
            base: (buf.buf.vptr() as *mut W).wrapping_add(res_ct(self.k_max, g, 0, j) * cw) as *mut c_void,
            ct_stride: (res_stride(g, j) * cw) as u64,
            part_stride: pw,
        };
        for j in (0..g.saturating_sub(1)).rev() {
            let rv = res_view(j, &cx.work.lt_res);
            if self.giant.is_some() {
                // res_j += res_{j+1} rotated by b * stride (c1 down to Q, up again, dot with c0 extended, added onto
                // res_j)
                self.rotate_ext(cx, j + 1, k, source, self.giant.as_ref(), Some(rv))?;
            } else {
                // b * stride = N/2: the giant rotation is the identity, the sums stay extended
                let ex = &*cx.ex;
                let nv = res_view(j + 1, &cx.work.lt_res);
                ks::ew(ctx, &ex.compute, ks::Ew::Add, rv, rv, Some(nv), lay.rs_qp(top), k, 2)?;
            }
        }
        // 4. stage end into x
        let d = ctx.chain.level_unit;
        // x = (sum - conv) B^-1 in the conversion's NTT when x's rows sit at the plan's offsets (the offset rotation's
        // lt_rot: always; the results in lt_res: one ciphertext)
        let into_x = |src: *const c_void, x: &mut CtBuf<'_, Compute, W>| ntt::Epi::SubMul { out: x.buf.vptr_mut(), x: src, s: self.binv.v.vptr(), ss: self.binv.s.vptr() };
        if self.geom.offset != 0 {
            self.rotate_ext(cx, 0, k, source, self.offset_key.as_ref(), None)?;
            let ex = &*cx.ex;
            let epi = into_x(cx.work.lt_rot.buf.vptr(), x);
            self.md_end.run_epi(ctx, &ex.compute, &mut cx.work.lt_rot.buf, &mut cx.work.lt_tmp.buf, k, Some(epi))?;
        } else if k == 1 {
            let ex = &*cx.ex;
            let epi = into_x(cx.work.lt_res.buf.vptr(), x);
            self.md_end.run_epi(ctx, &ex.compute, &mut cx.work.lt_res.buf, &mut cx.work.lt_tmp.buf, k, Some(epi))?;
        } else {
            let ex = &*cx.ex;
            self.md_end.run(ctx, &ex.compute, &mut cx.work.lt_res.buf, &mut cx.work.lt_tmp.buf, k)?;
            let rv = res_view(0, &cx.work.lt_res);
            let tv = res_view(0, &cx.work.lt_tmp);
            let out_top = if self.end == LtEnd::FusedRescale { top - d } else { top };
            self.sub_mul(ctx, &ex.compute, x, rv, tv, out_top, k)?;
        }
        Ok(())
    }

    /// x[0..k] = (a - b) B^-1 over Q rows 0..=out_top, both parts.
    #[allow(clippy::too_many_arguments)]
    fn sub_mul<M: DevStrategy<W>>(&self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, x: &mut CtBuf<'_, Compute, W>, a: View, b: View, out_top: usize, k: usize) -> Result<()> {
        let args = EwArgs {
            out: x.w(),
            a: a.r(),
            b: b.r(),
            rs: ctx.lay.rs_q(out_top),
            sc: self.binv.v.vptr(),
            scs: self.binv.s.vptr(),
            pr: ctx.pr(),
            k: k as i32,
            nparts: 2,
            sub: 0,
            zero_p: 0,
        };
        // SAFETY: k-ct views inside live buffers of the context layout; x is distinct from a and b
        launch_result("ew_sub_mul", unsafe { (ctx.table.ew_sub_mul)(&args, s.raw()) })?;
        Ok(())
    }

    /// `lt_rot` = rotation (key 0 of `key`) of result j in the extended basis: ModDown(c1 of res_j), ModUp(c1), dot
    /// with c0 as stored (extended); `acc`: added onto those ciphertexts instead.
    #[allow(clippy::too_many_arguments)]
    fn rotate_ext<M: DevStrategy<W>>(&self, cx: &mut BtsCx<'_, '_, W, M>, j: usize, k: usize, source: KskSource, key: Option<&KeyGroup<'_>>, acc: Option<View>) -> Result<()> {
        let key = key.ok_or_else(|| shape("LtStage", "rotation key missing"))?;
        let ctx = cx.ctx;
        let lay = ctx.lay;
        let top = self.top;
        let g = self.geom.g;
        let cw = lay.ct_words();
        let pw = lay.poly_words() as u64;
        let md = self.md_c1[j].as_ref().ok_or_else(|| shape("LtStage", format!("no ModDown plan for result {j}")))?;
        let ex = &mut *cx.ex;
        // c1 = (c1 - conv) P^-1 on the Q rows, in the conversion's NTT (the plan's rows sit at the same offsets in
        // lt_res and lt_tmp)
        let res = cx.work.lt_res.buf.vptr_mut();
        let epi = ntt::Epi::SubMul { out: res, x: res, s: ctx.ks.p_inv_mod_q.v.vptr(), ss: ctx.ks.p_inv_mod_q.s.vptr() };
        md.run_epi(ctx, &ex.compute, &mut cx.work.lt_res.buf, &mut cx.work.lt_tmp.buf, k, Some(epi))?;
        let rv = View {
            base: (cx.work.lt_res.buf.vptr() as *mut W).wrapping_add(res_ct(self.k_max, g, 0, j) * cw) as *mut c_void,
            ct_stride: (res_stride(g, j) * cw) as u64,
            part_stride: pw,
        };
        let c1 = rv.part::<W>(1);
        // ModUp of c1 and the dot with c0 extended (the INTT reads c1 in place, the results 1..=SRC_STRIDES ciphertext
        // layouts apart; another stride: from a copy in the coefficient workspace)
        if k == 1 || rv.ct_stride as usize <= ks::SRC_STRIDES * cw {
            ks::modup_from(ctx, &ex.compute, cx.ks, &mut ex.ks, &mut ex.bconv_work, c1, k, top)?;
        } else {
            let dst = View { base: ex.bconv_work.vptr_mut(), ct_stride: pw, part_stride: 0 };
            ks::ew(ctx, &ex.compute, ks::Ew::Copy, dst, View { part_stride: 0, ..c1 }, None, lay.rs_q(top), k, 1)?;
            ks::modup_coef_ready(ctx, &ex.compute, cx.ks, &mut ex.ks, &mut ex.bconv_work, k, top)?;
        }
        let staged = source == KskSource::Regen || key.needs_stage() || !key.has_a();
        if staged {
            let ndig = cx.ks.ndig(top)?;
            ks::stage_key(ctx, &ex.compute, cx.ks, &mut ex.ks, key, 0, ndig, true)?;
        }
        let (out, out_ct_stride, c0_mode) = match acc {
            Some(v) => (v.base, v.ct_stride, 2 | KS_ACC),
            None => (cx.work.lt_rot.buf.vptr_mut(), cw as u64, 2),
        };
        let io = DotIo { c1: c1.base, c0: rv.base, ct_stride: rv.ct_stride, out, part_stride: pw, out_ct_stride, out_rot_stride: 0, c0_mode };
        ks::dot(ctx, &ex.compute, cx.ks, &ex.ks, key, 0, 1, k, top, DotRows::All, io, staged)
    }
}
