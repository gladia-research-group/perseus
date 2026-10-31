//! Device key generation: every switching key of a `KeySet` from the device secret, one `ksk_gen` launch
//! per key (all digits and slots; `a` from the frozen KSKA expander, `b` written straight into the key's layout,
//! packed or dense). Host side: the KSKA seed and the error polynomials (Gaussian, sigma 3.19) come from the caller's
//! `pc_core::keys::Prng` (ChaCha20 DRBG over OS entropy); errors are lifted to residues with rayon, uploaded into the
//! executor's KS workspace and NTT'd there (the workspace is free at setup, when keys are generated).
//!
//! Relation and roles: `keys` module doc. Gates: the switch relation of every generated key (device check in the
//! tests and `pc-check keys`), a rotate / relin / encapsulation round trip through a reference key switch.
use crate::context::{Context, DevStrategy, DevWord};
use crate::error::{shape, Result};
use crate::executor::Executor;
use crate::keys::{DevKsk, DevSecret, KeyRole, KeySet, KskShape, KskStore, LevelRelinSpec};
use crate::ntt::{self, Dir, RowList};
use crate::poly::{CtBuf, Layout};
use pc_core::galois;
use pc_core::keys::{DiscreteGaussian, KskSeed, Prng};
use pc_core::params::Chain;
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, Stream};
use pc_kernels::{AutomorphArgs, KskGenArgs};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::ffi::c_void;

/// Setup-time tables of the generator for a fixed set of key shapes.
pub struct KeyGen<'a, W: DevWord> {
    /// Prime id of every slot, per shape.
    ids: BTreeMap<KskShape, ConstBuf<'a, i32>>,
    /// NTT rows of the error workspace `[digit][slot]`, per shape.
    e_rows: BTreeMap<KskShape, RowList<'a>>,
    /// prime id -> digit (Q ids), -1 for special ids.
    digit_of: ConstBuf<'a, i32>,
    /// sigma_{g^-1}(s) of the rotation being generated (part 0).
    perm: CtBuf<'a, Compute, W>,
    gauss: DiscreteGaussian,
    /// Level relinearization keys (prototype): digit_of and (P' mod q, Shoup) by prime id, per limb top.
    levels: BTreeMap<usize, (ConstBuf<'a, i32>, ConstBuf<'a, W>, ConstBuf<'a, W>)>,
}

/// Seconds spent per phase over a `generate_set` call.
#[derive(Clone, Copy, Debug, Default)]
pub struct KeyGenTimes {
    pub keys: usize,
    pub host_sample: f64,
    pub upload: f64,
    pub device: f64,
    pub total: f64,
}

impl<'a, W: DevWord> KeyGen<'a, W> {
    /// Arena bytes of `KeyGen::new` for `shapes` (reserve as "keygen").
    pub fn bytes(lay: &Layout, shapes: &[KskShape]) -> usize {
        let mut sh: Vec<KskShape> = shapes.to_vec();
        sh.sort();
        sh.dedup();
        let tabs: usize = sh
            .iter()
            .map(|s| Arena::bytes_for::<i32>(s.rows()) + Arena::bytes_for::<u32>(s.ndig * s.rows()) + Arena::bytes_for::<i32>(s.ndig * s.rows()))
            .sum();
        tabs + Arena::bytes_for::<i32>(lay.rows()) + Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(lay, 1))
    }

    /// Build the tables for every shape in `shapes` (setup only, uploads synced on `s`).
    pub fn new(arena: &'a Arena, s: &Stream<Compute>, chain: &Chain, shapes: &[KskShape]) -> Result<Self> {
        let lay = Layout::from_chain(chain);
        let mut ids = BTreeMap::new();
        let mut e_rows = BTreeMap::new();
        for sh in shapes {
            if ids.contains_key(sh) {
                continue;
            }
            let rid: Vec<i32> = sh.row_ids().into_iter().map(|x| x as i32).collect();
            let rl: Vec<(u32, i32)> = (0..sh.ndig * sh.rows()).map(|r| (r as u32, rid[r % sh.rows()])).collect();
            ids.insert(*sh, ConstBuf::new(arena, "keygen.ids", s, &rid)?);
            e_rows.insert(*sh, RowList::new(arena, s, &rl)?);
        }
        let dof: Vec<i32> = (0..lay.rows()).map(|id| chain.digit_of(id).map_or(-1, |d| d as i32)).collect();
        Ok(KeyGen {
            ids,
            e_rows,
            digit_of: ConstBuf::new(arena, "keygen.digit_of", s, &dof)?,
            perm: CtBuf::new(arena, "keygen.perm", lay, 1)?,
            gauss: DiscreteGaussian::new(DiscreteGaussian::SIGMA)?,
            levels: BTreeMap::new(),
        })
    }

    /// Tables of the level relinearization keys `specs` (their shapes must be among `new`'s); setup only.
    pub fn add_levels(&mut self, arena: &'a Arena, s: &Stream<Compute>, chain: &Chain, specs: &[LevelRelinSpec]) -> Result<()> {
        let lay = Layout::from_chain(chain);
        let primes = chain.primes();
        for sp in specs {
            let dof = sp.digit_of(lay.rows());
            let pm = sp.pmod(&primes, lay.rows());
            let (v, w): (Vec<W>, Vec<W>) = pm
                .iter()
                .enumerate()
                .map(|(t, &x)| {
                    let c = pc_core::modmul::shoup(W::from_u64(x % primes[t]), W::from_u64(primes[t]));
                    (c.w, c.w_prime)
                })
                .unzip();
            self.levels.insert(sp.top, (ConstBuf::new(arena, "keygen.levels", s, &dof)?, ConstBuf::new(arena, "keygen.levels", s, &v)?, ConstBuf::new(arena, "keygen.levels", s, &w)?));
        }
        Ok(())
    }
    /// Arena bytes of `add_levels`.
    pub fn levels_bytes(lay: &Layout, nspecs: usize) -> usize {
        nspecs * (Arena::bytes_for::<i32>(lay.rows()) + 2 * Arena::bytes_for::<W>(lay.rows()))
    }

    /// Generate the level relinearization key of limb top `top` (`add_levels` tables) into `key`.
    pub fn generate_level<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        ex: &mut Executor<'_, W>,
        secret: &DevSecret<'_, W>,
        key: &mut DevKsk<'_, W>,
        top: usize,
        prng: &mut Prng,
    ) -> Result<(f64, f64)> {
        let (d, v, w) = self.levels.get(&top).ok_or_else(|| shape("KeyGen::generate_level", format!("no level tables for top {top}")))?;
        let ov = (d.dptr(), v.vptr(), w.vptr());
        self.generate_with(ctx, ex, secret, key, prng, Some(ov))
    }

    /// Generate one key into `key` (its role, shape and layout) from `secret`; the seed and errors are drawn from
    /// `prng`. Issued on `ex.compute`; uses `ex.ks.digits` for the errors. Returns (host sample s, upload s).
    pub fn generate<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        ex: &mut Executor<'_, W>,
        secret: &DevSecret<'_, W>,
        key: &mut DevKsk<'_, W>,
        prng: &mut Prng,
    ) -> Result<(f64, f64)> {
        self.generate_with(ctx, ex, secret, key, prng, None)
    }

    /// `generate` with the digit map and the gadget's (P mod q, Shoup) tables overridden (`ov`, level keys).
    #[allow(clippy::too_many_arguments)]
    fn generate_with<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        ex: &mut Executor<'_, W>,
        secret: &DevSecret<'_, W>,
        key: &mut DevKsk<'_, W>,
        prng: &mut Prng,
        ov: Option<(*const i32, *const c_void, *const c_void)>,
    ) -> Result<(f64, f64)> {
        let n = ctx.lay.n;
        let sh = key.shape;
        let ids = self.ids.get(&sh).ok_or_else(|| shape("KeyGen::generate", format!("shape {sh:?} was not given to KeyGen::new")))?;
        let erows = &self.e_rows[&sh];
        let nr = sh.ndig * sh.rows();
        if ex.ks.digits.len() < nr * n {
            return Err(shape("KeyGen::generate", format!("ks_work holds {} words, the key's errors need {}", ex.ks.digits.len(), nr * n)));
        }
        if matches!(key.role, KeyRole::EncDenseToSparse | KeyRole::EncSparseToDense) && !secret.has_sparse {
            return Err(shape("KeyGen::generate", "encapsulation key requested but the device secret has no sparse part"));
        }
        let t0 = std::time::Instant::now();
        let seed = prng.ksk_seed();
        // per-digit error streams keyed from the caller's DRBG (parallel sampling, no shared state)
        let sub: [u32; 8] = prng.ksk_seed().0;
        let primes = ctx.chain.primes();
        let row_ids = sh.row_ids();
        let rows = sh.rows();
        let errs: Vec<Vec<i64>> = (0..sh.ndig)
            .into_par_iter()
            .map(|d| {
                let mut p = Prng::from_seed(sub, d as u32);
                self.gauss.sample_vec(&mut p, n)
            })
            .collect();
        let mut host = vec![W::default(); nr * n];
        host.par_chunks_mut(n).enumerate().for_each(|(r, o)| {
            let p = primes[row_ids[r % rows]] as i64;
            for (x, &v) in o.iter_mut().zip(&errs[r / rows]) {
                *x = W::from_u64(v.rem_euclid(p) as u64);
            }
        });
        let t_host = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let s = &ex.compute;
        ex.ks.digits.upload_at(s, 0, &host)?;
        key.seed = seed;
        key.seed_dev.upload(s, &seed.0)?;
        let t_up = t1.elapsed().as_secs_f64();
        ntt::transform_rows(ctx, s, Dir::Forward, &mut ex.ks.digits, erows)?;

        let sp = secret.buf.r();
        let part = |p: usize| (sp.base as *const W).wrapping_add(p * ctx.lay.poly_words()) as *const c_void;
        let (s_new, s_old, sq) = match key.role {
            KeyRole::Relin => (part(0), part(0), 1),
            KeyRole::EncDenseToSparse => (part(1), part(0), 0),
            KeyRole::EncSparseToDense => (part(0), part(1), 0),
            KeyRole::Rot { .. } | KeyRole::Conj => {
                let g = key.role.galois(n).ok_or_else(|| shape("KeyGen::generate", "rotation without a Galois element"))?;
                let ginv = galois::galois_inverse(g, n)?;
                let a = AutomorphArgs {
                    out: self.perm.w(),
                    inp: sp,
                    rs: ctx.lay.rs_qp(ctx.lay.nq - 1),
                    pr: ctx.pr(),
                    k: ginv as u32,
                    kk: 1,
                    nparts: 1,
                    has_add: 0,
                    ..Default::default()
                };
                // SAFETY: perm and the secret are distinct context-layout buffers; rows of rs exist in both
                launch_result("automorph", unsafe { (ctx.table.automorph)(&a, s.raw()) })?;
                (self.perm.buf.vptr(), part(0), 0)
            }
        };
        let (b, a, bits, row_words) = match &mut key.store {
            KskStore::PackedSeeded { b, bits, row_words } => (b.vptr_mut(), std::ptr::null_mut(), *bits as i32, *row_words as u64),
            KskStore::Dense { b, a } => (b.vptr_mut(), a.vptr_mut(), 0, 0),
        };
        let g = KskGenArgs {
            b,
            a,
            e: ex.ks.digits.vptr(),
            s_new,
            s_old,
            seed: key.seed_dev.dptr(),
            ids: ids.dptr(),
            digit_of: ov.map(|o| o.0).unwrap_or(self.digit_of.dptr()),
            // the key's own special set: P, or P' for a bottom-shape key of a lean chain (or a level key's)
            pmod: ov.map(|o| o.1).unwrap_or(if sh.kp < ctx.lay.kp { ctx.ks.ps_mod_q.v.vptr() } else { ctx.ks.p_mod_q.v.vptr() }),
            pmods: ov.map(|o| o.2).unwrap_or(if sh.kp < ctx.lay.kp { ctx.ks.ps_mod_q.s.vptr() } else { ctx.ks.p_mod_q.s.vptr() }),
            pr: ctx.pr(),
            row_words,
            nrows: rows as i32,
            ndig: sh.ndig as i32,
            nq: ctx.lay.nq as i32,
            bits,
            sq,
            pad: 0,
        };
        // SAFETY: b/a hold ndig*rows rows of the key's layout; e holds ndig*rows NTT'd rows; secret/perm rows by
        // prime id; all on the compute lane in issue order
        launch_result("ksk_gen", unsafe { (ctx.table.ksk_gen)(&g, s.raw()) })?;
        Ok((t_host, t_up))
    }

    /// Generate every key of `set` (then syncs the compute lane).
    pub fn generate_set<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        ex: &mut Executor<'_, W>,
        secret: &DevSecret<'_, W>,
        set: &mut KeySet<'_, W>,
        prng: &mut Prng,
    ) -> Result<KeyGenTimes> {
        let t0 = std::time::Instant::now();
        let mut t = KeyGenTimes::default();
        for k in set.iter_mut() {
            let (h, u) = self.generate(ctx, ex, secret, k, prng)?;
            t.host_sample += h;
            t.upload += u;
            t.keys += 1;
        }
        ex.sync_compute()?;
        t.total = t0.elapsed().as_secs_f64();
        t.device = t.total - t.host_sample - t.upload;
        Ok(t)
    }
}

/// Every shape of `set` (for `KeyGen::new` / `KeyGen::bytes`).
pub fn shapes_of<W: DevWord>(set: &KeySet<'_, W>) -> Vec<KskShape> {
    let mut v: Vec<KskShape> = set.iter().map(|k| k.shape).collect();
    v.sort();
    v.dedup();
    v
}

/// A fixed seed for reproducible tests (`Prng::from_seed`).
pub fn test_prng(tag: u32) -> Prng {
    Prng::from_seed(KskSeed([0x5eed_0001, tag, 3, 4, 5, 6, 7, 8]).0, 0)
}
