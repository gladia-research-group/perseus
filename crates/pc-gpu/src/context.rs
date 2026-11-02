//! The immutable device context: the chain, its scale tables, the device constant tables in the kernels' layouts and the
//! kernel table of the (word, ModMul) instantiation, resolved once.
use crate::error::{shape, Result};
use crate::mem::MemPlan;
use crate::ntt::RowTables;
use crate::poly::Layout;
use pc_core::modmul::{invmod, mulmod, shoup, Barrett32, Barrett64, DevPrimeConst, Montgomery};
use pc_core::params::Chain;
use pc_core::scale::ScaleTables;
use pc_core::word::Word;
use pc_cuda::{Arena, ConstBuf, Lane, Pod, Stream};
use pc_kernels::{Instance, KernelTable};
use pc_shadow::ChainShape;
use rayon::prelude::*;
use std::marker::PhantomData;

/// A storage word the device runtime can hold: pc-core's `Word` that is also device-uploadable.
pub trait DevWord: Word + Pod {}
impl<T: Word + Pod> DevWord for T {}

/// The kernel instantiation implementing ModMul strategy `Self` over words `W` (seam "general modular product" on
/// the device). Implemented for the instantiations pc-kernels can build; whether one is compiled in is a Cargo
/// feature of pc-kernels (`Context::new` reports `NotBuilt` otherwise). Not bounded by `pc_core::ModMul<W>`: the
/// device u64 x Barrett64 instance has no host strategy in pc-core (no host Barrett64 for 64-bit words), and the device consts are the single
/// `DevPrimeConst` layout for every strategy anyway.
pub trait DevStrategy<W: DevWord>: Send + Sync + 'static {
    const INSTANCE: Instance;
}
impl DevStrategy<u32> for Barrett64 {
    const INSTANCE: Instance = Instance::U32Barrett64;
}
impl DevStrategy<u32> for Barrett32 {
    const INSTANCE: Instance = Instance::U32Barrett32;
}
impl DevStrategy<u32> for Montgomery {
    const INSTANCE: Instance = Instance::U32Mont;
}
impl DevStrategy<u64> for Barrett64 {
    const INSTANCE: Instance = Instance::U64Barrett64;
}
impl DevStrategy<u16> for Barrett32 {
    const INSTANCE: Instance = Instance::U16Barrett32;
}

/// NTT tables by prime id (`NttArgs` layout): `fw[id*N + i] = psi^brev(i)`, `iw` the inverse powers, `*s` the
/// Shoup companions at W, `ninv[id] = N^-1 mod p`; `lazy`: u32 words and every prime below 2^30 (`NttArgs::lazy`).
pub struct NttTables<'a, W: DevWord> {
    pub fw: ConstBuf<'a, W>,
    pub fws: ConstBuf<'a, W>,
    pub iw: ConstBuf<'a, W>,
    pub iws: ConstBuf<'a, W>,
    pub ninv: ConstBuf<'a, W>,
    pub ninvs: ConstBuf<'a, W>,
    pub lazy: bool,
}

/// Per-prime-id scalar table with its Shoup companion (`v[id]`, `s[id]`), the form `EwArgs::sc/scs`,
/// `MdEpiArgs::binv/binvs` etc. take. Entries outside the table's domain are 0.
pub struct ScalarTable<'a, W: DevWord> {
    pub v: ConstBuf<'a, W>,
    pub s: ConstBuf<'a, W>,
}

impl<'a, W: DevWord> ScalarTable<'a, W> {
    /// Upload `vals` (by prime id, standard form, `< p`) with their Shoup constants.
    pub fn new<S: Lane>(arena: &'a Arena, what: &'static str, s: &Stream<S>, primes: &[u64], vals: &[u64]) -> Result<Self> {
        let (v, sh) = shoup_pairs::<W>(primes, vals);
        Ok(ScalarTable { v: ConstBuf::new(arena, what, s, &v)?, s: ConstBuf::new(arena, what, s, &sh)? })
    }
    /// Elements per table (MemPlan): one entry per prime id, twice.
    pub fn elems(np: usize) -> [usize; 2] {
        [np, np]
    }
}

fn shoup_pairs<W: DevWord>(primes: &[u64], vals: &[u64]) -> (Vec<W>, Vec<W>) {
    vals.iter()
        .zip(primes)
        .map(|(&x, &p)| {
            let c = shoup(W::from_u64(x % p), W::from_u64(p));
            (c.w, c.w_prime)
        })
        .unzip()
}

/// ModDown / ModUp scalars by prime id (Q ids; special ids hold 0): `P mod q_i`, `P^-1 mod q_i`, and the same for P'
/// = the first `Chain::p_sparse` special primes (the bottom-shape keys' modulus; P itself but on a lean chain). The
/// bconv job tables themselves depend on buffer offsets and are built by their users (`bconv::BconvPlan`).
pub struct KsScalars<'a, W: DevWord> {
    pub p_mod_q: ScalarTable<'a, W>,
    pub p_inv_mod_q: ScalarTable<'a, W>,
    pub ps_mod_q: ScalarTable<'a, W>,
    pub ps_inv_mod_q: ScalarTable<'a, W>,
}

/// Device context of a chain for words `W` and ModMul strategy `M`.
pub struct Context<'a, W: DevWord, M: DevStrategy<W>> {
    pub chain: Chain,
    pub scale: ScaleTables,
    pub shape: ChainShape,
    pub lay: Layout,
    /// The kernel table of `M::INSTANCE` (checked against the ABI, word size, ModMul and log N).
    pub table: &'static KernelTable,
    /// `DevPrimeConst` by prime id (Q ids then P ids), the `pr` argument of every kernel.
    pub primes: ConstBuf<'a, DevPrimeConst>,
    pub ntt: NttTables<'a, W>,
    /// NTT row tables for every live-row span of up to `k_max` ciphertexts of `lay`.
    pub rows: RowTables<'a>,
    pub ks: KsScalars<'a, W>,
    /// Ciphertexts per launch this context's row tables cover (`MemCfg::k_max`).
    pub k_max: usize,
    _m: PhantomData<M>,
}

/// Element counts of the context's device tables, shared by `MemPlan` (sizing) and `Context::new` (carving).
pub(crate) struct ConstCounts {
    pub primes: usize,
    pub w_tables: Vec<usize>,
    pub row_entries: usize,
}

impl ConstCounts {
    pub fn of(lay: &Layout, k_max: usize) -> Self {
        let np = lay.rows();
        let mut w_tables = vec![np * lay.n; 4]; // fw fws iw iws
        w_tables.extend([np, np]); // ninv ninvs
        w_tables.extend([np; 8]); // ks scalars (P and P': 2 tables x value/shoup each)
        ConstCounts { primes: np, w_tables, row_entries: RowTables::entries(lay, k_max) }
    }
    pub fn bytes<W>(&self) -> usize {
        Arena::bytes_for::<DevPrimeConst>(self.primes)
            + self.w_tables.iter().map(|&n| Arena::bytes_for::<W>(n)).sum::<usize>()
            + Arena::bytes_for::<u32>(self.row_entries)
            + Arena::bytes_for::<i32>(self.row_entries)
    }
}

/// Checks that `chain` can be held in words `W` by the kernels (`N == 2^LOG_N`, every prime fits `DevPrimeConst`).
pub(crate) fn check_chain<W: DevWord>(chain: &Chain) -> Result<()> {
    if chain.n != pc_kernels::N {
        return Err(shape("chain.spec.logn", format!("N = {} but the kernels are built for N = {}", chain.n, pc_kernels::N)));
    }
    if chain.word_bits != W::BITS {
        return Err(shape("chain.spec.word", format!("chain words are {} bits, the context's W is {}", chain.word_bits, W::BITS)));
    }
    for &p in chain.q.iter().chain(&chain.p) {
        DevPrimeConst::try_new(p, W::BITS)?;
    }
    Ok(())
}

impl<'a, W: DevWord, M: DevStrategy<W>> Context<'a, W, M> {
    /// Build every device table of `chain` in `arena` (uploads ordered on `s`, which is synced before returning).
    /// `plan` must have been built for this chain and `(W, M)` (`MemPlan::build`).
    pub fn new<S: Lane>(arena: &'a Arena, s: &Stream<S>, chain: Chain, plan: &MemPlan) -> Result<Self> {
        check_chain::<W>(&chain)?;
        let lay = Layout::from_chain(&chain);
        if plan.lay != lay || plan.word_bits != W::BITS {
            return Err(shape("Context::new", format!("MemPlan built for {:?} at {} bits, chain is {lay:?} at {}", plan.lay, plan.word_bits, W::BITS)));
        }
        let table = pc_kernels::table(M::INSTANCE)?;
        let scale = ScaleTables::precompute(&chain)?;
        let shape = ChainShape::from_tables(&scale);
        let primes_v = chain.primes();
        let roots = chain.roots();
        let n = chain.n;
        let logn = n.trailing_zeros();

        let pc: Vec<DevPrimeConst> = primes_v.iter().map(|&p| DevPrimeConst::new(p, W::BITS)).collect();
        let primes = ConstBuf::new(arena, "ctx.primes", s, &pc)?;

        // twiddles: powers of psi in natural order scattered to bit-reversed positions 
        let per_prime: Vec<(Vec<W>, Vec<W>, Vec<W>, Vec<W>)> = primes_v
            .par_iter()
            .zip(roots.par_iter())
            .map(|(&p, &psi)| {
                let ipsi = invmod(psi, p);
                let (mut pw, mut ipw) = (vec![1u64; n], vec![1u64; n]);
                for i in 1..n {
                    pw[i] = mulmod(pw[i - 1], psi, p);
                    ipw[i] = mulmod(ipw[i - 1], ipsi, p);
                }
                let wp = W::from_u64(p);
                let mut t = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
                for i in 0..n {
                    let e = (i as u32).reverse_bits() as usize >> (32 - logn);
                    let f = shoup(W::from_u64(pw[e]), wp);
                    let g = shoup(W::from_u64(ipw[e]), wp);
                    t.0.push(f.w);
                    t.1.push(f.w_prime);
                    t.2.push(g.w);
                    t.3.push(g.w_prime);
                }
                t
            })
            .collect();
        let cat = |f: fn(&(Vec<W>, Vec<W>, Vec<W>, Vec<W>)) -> &Vec<W>| per_prime.iter().flat_map(|t| f(t).iter().copied()).collect::<Vec<W>>();
        let ninv_v: Vec<u64> = primes_v.iter().map(|&p| invmod(n as u64 % p, p)).collect();
        let (ninv, ninvs) = shoup_pairs::<W>(&primes_v, &ninv_v);
        let ntt = NttTables {
            fw: ConstBuf::new(arena, "ctx.ntt.fw", s, &cat(|t| &t.0))?,
            fws: ConstBuf::new(arena, "ctx.ntt.fws", s, &cat(|t| &t.1))?,
            iw: ConstBuf::new(arena, "ctx.ntt.iw", s, &cat(|t| &t.2))?,
            iws: ConstBuf::new(arena, "ctx.ntt.iws", s, &cat(|t| &t.3))?,
            ninv: ConstBuf::new(arena, "ctx.ntt.ninv", s, &ninv)?,
            ninvs: ConstBuf::new(arena, "ctx.ntt.ninvs", s, &ninvs)?,
            lazy: W::BITS == 32 && primes_v.iter().all(|&p| p < 1 << 30),
        };
        drop(per_prime);

        let rows = RowTables::new(arena, s, lay, plan.cfg.k_max)?;

        let nq = chain.q.len();
        let mut pmq = scale.p_mod_q.clone();
        let mut pimq = scale.p_inv_mod_q.clone();
        pmq.resize(lay.rows(), 0);
        pimq.resize(lay.rows(), 0);
        debug_assert!(nq <= pmq.len());
        let ps = &chain.p[..chain.p_sparse];
        let mut psmq: Vec<u64> = chain.q.iter().map(|&qi| ps.iter().fold(1u64 % qi, |a, &x| pc_core::modmul::mulmod(a, x % qi, qi))).collect();
        let mut psimq: Vec<u64> = psmq.iter().zip(&chain.q).map(|(&x, &qi)| invmod(x, qi)).collect();
        psmq.resize(lay.rows(), 0);
        psimq.resize(lay.rows(), 0);
        let ks = KsScalars {
            p_mod_q: ScalarTable::new(arena, "ctx.ks.p_mod_q", s, &primes_v, &pmq)?,
            p_inv_mod_q: ScalarTable::new(arena, "ctx.ks.p_inv_mod_q", s, &primes_v, &pimq)?,
            ps_mod_q: ScalarTable::new(arena, "ctx.ks.ps_mod_q", s, &primes_v, &psmq)?,
            ps_inv_mod_q: ScalarTable::new(arena, "ctx.ks.ps_inv_mod_q", s, &primes_v, &psimq)?,
        };
        s.sync()?;
        Ok(Context { chain, scale, shape, lay, table, primes, ntt, rows, ks, k_max: plan.cfg.k_max, _m: PhantomData })
    }

    /// The `pr` argument of every kernel.
    pub fn pr(&self) -> *const DevPrimeConst {
        self.primes.dptr()
    }
    /// Prime value of id `id` (Q ids then P ids).
    pub fn prime(&self, id: usize) -> u64 {
        if id < self.chain.q.len() {
            self.chain.q[id]
        } else {
            self.chain.p[id - self.chain.q.len()]
        }
    }
    /// Top limb index of a ciphertext at `level` (pc-shadow convention).
    pub fn top_of(&self, level: u32) -> Result<usize> {
        Ok(pc_shadow::meta::top_limb(self.shape, level)? as usize)
    }
}
