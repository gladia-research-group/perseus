//! Base-conversion job tables (seam `BasisConv`: exact HPS with the FMA double estimate, `bconv` kernel), built at
//! setup and launched as one kernel for all jobs. Word-generic `BconvPlan`; the pool layout
//! is the one documented at `BconvJob` in `kernels/args.cuh`.
use crate::context::{Context, DevStrategy, DevWord};
use crate::error::{shape, Result};
use pc_core::modmul::{invmod, mulmod};
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, DBuf, Lane, Stream};
use pc_kernels::{BconvArgs, BconvJob};

/// Most sources one job of the register kernel may have; wider jobs (u32 words) run on `bconv_wide`.
pub const MAX_SRC: usize = 16;
/// Most sources of a wide job (u32 words; `bconv_wide`).
pub const MAX_SRC_WIDE: usize = 48;

/// One conversion: source rows `(limb offset, prime id)` -> target rows, limb offsets relative to `src_off` /
/// `dst_off` (units of N) in the src / dst buffers.
#[derive(Clone, Debug)]
pub struct JobSpec {
    pub src_off: u32,
    pub dst_off: u32,
    pub src: Vec<(u32, u32)>,
    pub tgt: Vec<(u32, u32)>,
    /// HPS-corrected (exact) conversion.
    pub exact: bool,
    /// The source rows hold `x_i pre_i` already (`pre_scales`: folded into the INTT's last pass).
    pub prescaled: bool,
}

/// `pre_i = (B / b_i)^-1 mod b_i` of the sources `bp` (`B` their product).
pub fn pre_factors(bp: &[u64]) -> Vec<u64> {
    let ns = bp.len();
    (0..ns).map(|i| invmod((0..ns).filter(|&k| k != i).fold(1u64, |h, k| mulmod(h, bp[k] % bp[i], bp[i])), bp[i])).collect()
}

/// Device job tables of a set of conversions.
pub struct BconvPlan<'a> {
    jobs: ConstBuf<'a, [u32; 8]>,
    pool: ConstBuf<'a, u8>,
    pub njobs: usize,
    pub max_tgt: usize,
    /// Most sources of a job (> MAX_SRC: the wide kernel).
    pub max_src: usize,
    /// Largest limb read in src / written in dst + 1 (bounds checks at launch).
    pub src_limbs: usize,
    pub dst_limbs: usize,
    /// `src_limbs` / `dst_limbs` of the job prefix `[0, j]` (bounds checks of `run_prefix`).
    prefix_limbs: Vec<(usize, usize)>,
}

/// The pool words of `specs` at word size `W` and the job records.
pub fn build_tables<W: DevWord>(primes: &[u64], specs: &[JobSpec]) -> Result<(Vec<W>, Vec<[u32; 8]>, usize)> {
    let wb = std::mem::size_of::<W>();
    let mut pool: Vec<W> = Vec::new();
    let mut jobs = Vec::with_capacity(specs.len());
    let mut max_tgt = 0;
    let push = |pool: &mut Vec<W>, v: u64| pool.push(W::from_u64(v));
    for j in specs {
        let max = if wb == 4 { MAX_SRC_WIDE } else { MAX_SRC };
        if j.src.is_empty() || j.src.len() > max {
            return Err(shape("bconv job", format!("{} sources (1..={max})", j.src.len())));
        }
        let ns = j.src.len();
        if ns > MAX_SRC {
            // lazy double-word sums: nsrc (max source - 1) (max target - 1) < 2^(2 bits)
            let ms = j.src.iter().map(|&(_, p)| primes[p as usize]).max().unwrap_or(0) as u128;
            let mt = j.tgt.iter().map(|&(_, p)| primes[p as usize]).max().unwrap_or(0) as u128;
            if ns as u128 * ms.saturating_sub(1) * mt.saturating_sub(1) >= 1u128 << (16 * wb) {
                return Err(shape("bconv job", format!("{ns} sources overflow the lazy double-word sum")));
            }
        }
        let off = pool.len();
        let bp: Vec<u64> = j.src.iter().map(|&(_, p)| primes[p as usize]).collect();
        for &(r, _) in &j.src {
            push(&mut pool, r as u64);
        }
        for &(_, p) in &j.src {
            push(&mut pool, p as u64);
        }
        // pre_i = (B / b_i)^-1 mod b_i
        for v in pre_factors(&bp) {
            push(&mut pool, v);
        }
        for &(r, _) in &j.tgt {
            push(&mut pool, r as u64);
        }
        for &(_, p) in &j.tgt {
            push(&mut pool, p as u64);
        }
        // M[i][t] = (B / b_i) mod t
        for i in 0..ns {
            for &(_, tp) in &j.tgt {
                let t = primes[tp as usize];
                push(&mut pool, (0..ns).filter(|&k| k != i).fold(1u64, |v, k| mulmod(v, bp[k] % t, t)));
            }
        }
        if j.exact {
            // rb_i = 1/b_i as f64, 8/sizeof(W) little-endian words each, the first 8-byte aligned in the pool
            while (pool.len() * wb) % 8 != 0 {
                pool.push(W::ZERO);
            }
            for &b in &bp {
                let bits = (1.0f64 / b as f64).to_bits();
                for w in 0..8 / wb {
                    let part = if wb == 8 { bits } else { (bits >> (8 * wb * w)) & ((1u64 << (8 * wb)) - 1) };
                    push(&mut pool, part);
                }
            }
            for &(_, tp) in &j.tgt {
                let t = primes[tp as usize];
                push(&mut pool, bp.iter().fold(1u64, |m, &b| mulmod(m, b % t, t)));
            }
        }
        max_tgt = max_tgt.max(j.tgt.len());
                // u32 words, every target below 2^30: the sums reduced as hi 2^32 + lo in 32-bit products (`BconvJob::split`)
        let split = wb == 4 && j.tgt.iter().all(|&(_, tp)| primes[tp as usize] < 1 << 30);
        jobs.push([j.src_off, j.dst_off, ns as u32, j.tgt.len() as u32, off as u32, j.exact as u32, j.prescaled as u32, split as u32]);
    }
    Ok((pool, jobs, max_tgt))
}

impl<'a> BconvPlan<'a> {
    pub fn new<S: Lane, W: DevWord>(arena: &'a Arena, s: &Stream<S>, primes: &[u64], specs: &[JobSpec]) -> Result<Self> {
        let (pool, jobs, max_tgt) = build_tables::<W>(primes, specs)?;
        // the pool travels as bytes so the ConstBuf element type does not depend on W
        let bytes: Vec<u8> = pool.iter().flat_map(|w| w.to_u64().to_le_bytes()[..std::mem::size_of::<W>()].to_vec()).collect();
        let src_limbs = specs.iter().flat_map(|j| j.src.iter().map(move |r| (j.src_off + r.0) as usize + 1)).max().unwrap_or(0);
        let dst_limbs = specs.iter().flat_map(|j| j.tgt.iter().map(move |r| (j.dst_off + r.0) as usize + 1)).max().unwrap_or(0);
        let mut prefix_limbs = Vec::with_capacity(specs.len());
        let (mut ms, mut md) = (0, 0);
        for j in specs {
            ms = j.src.iter().map(|r| (j.src_off + r.0) as usize + 1).fold(ms, usize::max);
            md = j.tgt.iter().map(|r| (j.dst_off + r.0) as usize + 1).fold(md, usize::max);
            prefix_limbs.push((ms, md));
        }
        Ok(BconvPlan {
            jobs: ConstBuf::new(arena, "bconv.jobs", s, &jobs)?,
            pool: ConstBuf::new(arena, "bconv.pool", s, &bytes)?,
            njobs: specs.len(),
            max_tgt,
            max_src: specs.iter().map(|j| j.src.len()).max().unwrap_or(0),
            src_limbs,
            dst_limbs,
            prefix_limbs,
        })
    }

    /// Device bytes of a plan (MemPlan sections for owners that size ahead).
    pub fn bytes<W: DevWord>(primes: &[u64], specs: &[JobSpec]) -> Result<usize> {
        let (pool, jobs, _) = build_tables::<W>(primes, specs)?;
        Ok(Arena::bytes_for::<[u32; 8]>(jobs.len()) + Arena::bytes_for::<W>(pool.len()))
    }

    /// Run every job: coefficient-form rows of `src` -> rows of `dst`.
    pub fn run<W: DevWord, M: DevStrategy<W>>(
        &self,
        ctx: &Context<'_, W, M>,
        s: &Stream<Compute>,
        src: &DBuf<'_, Compute, W>,
        dst: &mut DBuf<'_, Compute, W>,
    ) -> Result<()> {
        self.run_prefix(ctx, s, src, 0, dst, 0, self.njobs)
    }

    /// Run jobs `[0, njobs)` with the job offsets taken relative to element offsets `src_at` / `dst_at` of the
    /// buffers (a plan built for `k_max` ciphertexts ordered by ciphertext runs the first `k` with a prefix, at any
    /// ciphertext of a larger buffer).
    #[allow(clippy::too_many_arguments)]
    pub fn run_prefix<W: DevWord, M: DevStrategy<W>>(
        &self,
        ctx: &Context<'_, W, M>,
        s: &Stream<Compute>,
        src: &DBuf<'_, Compute, W>,
        src_at: usize,
        dst: &mut DBuf<'_, Compute, W>,
        dst_at: usize,
        njobs: usize,
    ) -> Result<()> {
        if njobs == 0 {
            return Ok(());
        }
        let n = ctx.lay.n;
        let (sl, dl) = *self.prefix_limbs.get(njobs - 1).ok_or_else(|| shape("bconv run", format!("{njobs} jobs of a {}-job plan", self.njobs)))?;
        if src_at + sl * n > src.len() || dst_at + dl * n > dst.len() {
            return Err(shape("bconv run", format!("plan spans {sl}/{dl} limbs at {src_at}/{dst_at}, buffers hold {}/{} words", src.len(), dst.len())));
        }
        const _: () = assert!(std::mem::size_of::<BconvJob>() == std::mem::size_of::<[u32; 8]>());
        let a = BconvArgs {
            // SAFETY (pointer arithmetic): offsets checked against the buffer lengths above
            src: unsafe { src.dptr().add(src_at) } as *const std::ffi::c_void,
            dst: unsafe { dst.dptr_mut().add(dst_at) } as *mut std::ffi::c_void,
            jobs: self.jobs.dptr() as *const BconvJob,
            pool: self.pool.vptr(),
            pr: ctx.pr(),
            njobs: njobs as i32,
            max_tgt: self.max_tgt as i32,
        };
        // SAFETY: job records are BconvJob-sized u32 octets; limb ranges checked above; src/dst are distinct borrows
        let f = if self.max_src > MAX_SRC { ctx.table.bconv_wide } else { ctx.table.bconv };
        launch_result("bconv", unsafe { f(&a, s.raw()) })?;
        Ok(())
    }
}
