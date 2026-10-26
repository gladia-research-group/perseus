//! The s = 1 route by SPRU (FIDESlib `Spru.cu`: `spruSetup` / `spruBootstrap`): decryption in the exponent over a
//! block-binary key s' (h blocks of B = N/h coefficients, one 1 per block, s'_0 = 1) instead of a mod raise:
//! adjust + drop to the bottom, dense -> block key switch, extraction of the 2n bootstrapping plaintexts, the
//! key-ciphertext sum, trace, product over the h blocks, mask recombination (degree 2 output).
//!
//! The output is the slot mean (a 1-slot fold) in every slot, 2 * levels(product) + 2 levels below the top (a
//! variant of drop d starts the key-ciphertext product d levels lower and lands that much deeper).
//! All device state is carved at setup (`Spru::bytes` -> MemPlan). Steps: docs/ARCHITECTURE.md "pc-gpu SPRU".
use crate::context::{Context, DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Error, Result};
use crate::executor::Executor;
use crate::keys::{DevKsk, DevSecret, KeyRole, KskLayout, KskShape};
use crate::ks::{self, DotIo, DotRows, KeyGroup, View};
use crate::ntt::{self, Dir, RowList};
use crate::poly::{CtBuf, Layout};
use crate::rotate;
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::keys::{Prng, SecretKey};
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, DBuf, Stream};
use pc_kernels::SpruExtractArgs;
use pc_shadow::{CtMeta, PtMeta, Shadow};

use super::{BtsCx, BtsKeyRing};
use crate::ks::KskSource;
use std::collections::BTreeMap;

/// SPRU parameters (Route.spru_h; n and the trace baby step are fixed by the construction / FIDESlib defaults).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpruCfg {
    /// Blocks of the block-binary key (FIDESLIB_SPRU = h).
    pub h: usize,
    /// Components per slot (2: X^0 and X^(N/2)).
    pub n: usize,
    /// Accumulate baby step (rotations per trace round + 1).
    pub acc_bstep: usize,
    /// Ciphertexts per launch the route's buffers are carved for (K-batched SPRU).
    pub k_max: usize,
    /// Raise drops the route holds a variant for, as a bit set: bit d = the key ciphertexts are read with their top d
    /// CKKS levels dropped, so the extraction, product and recombination run d levels lower and the output lands d
    /// levels deeper (`Route::raise_drop` of an s = 1 site). Bit 0 is the shipped route.
    pub drops: u32,
}

impl SpruCfg {
    pub fn new(h: usize) -> Self {
        SpruCfg { h, n: 2, acc_bstep: 4, k_max: 1, drops: 1 }
    }

    /// The drops of `drops`, ascending.
    pub fn drop_list(&self) -> Vec<u32> {
        (0..u32::BITS).filter(|d| self.drops >> d & 1 == 1).collect()
    }

    pub fn check(&self, ring_n: usize) -> Result<()> {
        let ok = self.h >= 2
            && self.h.is_power_of_two()
            && self.n == 2
            && ring_n % self.h == 0
            && (ring_n / self.h) % (2 * self.n) == 0
            && self.k_max >= 1
            && self.acc_bstep >= 2
            && self.acc_bstep.is_power_of_two()
            && self.drops != 0;
        if !ok {
            return Err(Error::Mem { knob: "bts.spru", value: self.h, msg: format!("SPRU h must be a power of two dividing N/{} (got {self:?})", 2 * self.n) });
        }
        Ok(())
    }

    /// Block size B = N / h.
    pub fn block(&self, ring_n: usize) -> usize {
        ring_n / self.h
    }

    /// The trace rounds (FIDESlib `Accumulate(acc_bstep, h n, N/2 / (h n))`): each round one hoisted rotate_sum.
    pub fn trace_rounds(&self, ring_n: usize) -> Vec<Vec<i64>> {
        accumulate_rounds(self.acc_bstep, self.h * self.n, ring_n / 2 / (self.h * self.n))
    }

    /// The product rotations n, 2n, ..., < h n.
    pub fn product_steps(&self) -> Vec<i64> {
        let mut v = vec![];
        let mut s = self.n;
        while s < self.h * self.n {
            v.push(s as i64);
            s <<= 1;
        }
        v
    }

    /// Every rotation step the route needs (trace, product, the recombination's 1); conjugation and the relin key too.
    pub fn rot_steps(&self, ring_n: usize) -> Vec<i64> {
        let mut v: Vec<i64> = self.trace_rounds(ring_n).into_iter().flatten().collect();
        v.extend(self.product_steps());
        v.push(1);
        v.sort();
        v.dedup();
        v
    }

    /// Levels (primes dropped) the route consumes from the top: the external product + the product tree.
    pub fn depth_primes(&self, unit: u32) -> u32 {
        unit * (1 + self.product_steps().len() as u32)
    }

    /// (top limb of the key-ciphertext product, top limb of the recombination) of the variant of drop `drop`, refused
    /// when the recombination's degree-2 output would have no composite level left to realize into.
    pub fn tops(&self, nq: usize, unit: u32, drop: u32) -> Result<(usize, usize)> {
        let top = (nq - 1).checked_sub((unit * drop) as usize);
        let mask_top = top.and_then(|t| t.checked_sub(self.depth_primes(unit) as usize)).filter(|&m| m + 1 >= 2 * unit as usize);
        match (top, mask_top) {
            (Some(t), Some(m)) => Ok((t, m)),
            _ => Err(Error::Mem { knob: "bts.spru.drops", value: drop as usize, msg: format!("SPRU drop {drop}: the route needs {} primes below the top", self.depth_primes(unit) + 2 * unit) }),
        }
    }
}

/// FIDESlib `Accumulate(bStep, stride, size)` as rotate_sum rounds: round r sums the rotations by
/// j * stride * bStep^r, 0 < j < bStep (cut at stride * size).
pub fn accumulate_rounds(bstep: usize, stride: usize, size: usize) -> Vec<Vec<i64>> {
    let mut out = vec![];
    let mut s = 1usize;
    while s < size {
        let mut r = vec![];
        let mut idx = stride * s;
        while idx < stride * size && idx < bstep * stride * s {
            r.push(idx as i64);
            idx += stride * s;
        }
        out.push(r);
        s *= bstep;
    }
    out
}

/// The block-binary key: block b holds its 1 at b B + pos\[b\], pos\[0\] = 0 (s'_0 = 1).
#[derive(Clone, Debug)]
pub struct SpruKey {
    pub pos: Vec<usize>,
    pub s_block: Vec<i8>,
}

impl SpruKey {
    /// Positions from the caller's DRBG (FIDESlib takes OpenFHE's CSPRNG; a narrow seed would shrink the key space).
    pub fn sample(prng: &mut Prng, ring_n: usize, cfg: &SpruCfg) -> Result<Self> {
        cfg.check(ring_n)?;
        let b = cfg.block(ring_n);
        let mut s = vec![0i8; ring_n];
        let pos: Vec<usize> = (0..cfg.h).map(|i| if i == 0 { 0 } else { prng.uniform_below(b as u64) as usize }).collect();
        for (i, &p) in pos.iter().enumerate() {
            s[i * b + p] = 1;
        }
        Ok(SpruKey { pos, s_block: s })
    }

    /// The slot vector S_u of bootstrapping key ciphertext u: slot k h n + b n + a = s'_(b B + u B/(2n) + k).
    pub fn key_slots(&self, cfg: &SpruCfg, ring_n: usize, u: usize) -> Vec<Complex> {
        let (b_sz, h, n) = (cfg.block(ring_n), cfg.h, cfg.n);
        let sub = b_sz / (2 * n);
        let mut v = vec![Complex::default(); ring_n / 2];
        for k in 0..sub {
            for b in 0..h {
                for a in 0..n {
                    v[k * h * n + b * n + a] = Complex::new(self.s_block[b * b_sz + u * sub + k] as f64, 0.0);
                }
            }
        }
        v
    }
}

/// Profile phases of `Spru::run_profiled`.
pub const PHASES: [&str; 7] = ["adjust", "switch", "extract+encode", "extmult", "trace", "product", "finish"];

/// Device material and workspaces of the route (setup-carved).
pub struct Spru<'a, W: DevWord> {
    pub cfg: SpruCfg,
    lay: Layout,
    /// Key switch dense -> block key at the bottom.
    atob: DevKsk<'a, W>,
    atob_group: Option<KeyGroup<'a>>,
    /// Rotation groups by exact step list (trace rounds, product steps), relinearization (`bind`).
    groups: BTreeMap<Vec<i64>, KeyGroup<'a>>,
    relin: Option<KeyGroup<'a>>,
    /// Key sources: rotations (the fused dots need resident `a`: Cached) and relinearization (Regen: in-kernel `a`).
    pub rot_source: KskSource,
    pub relin_source: KskSource,
    /// Enc(S_u) at the top, u < 2n.
    cs: Ct<'a, W>,
    /// Device pointers to the E_u rows (the extmult's plaintext dot table).
    ept_ptrs: ConstBuf<'a, u64>,
    /// The E_u FFT inputs `[u][N]` complex (2 doubles each) and the ping-pong buffer of the passes.
    ev: DBuf<'a, Compute, f64>,
    fb: DBuf<'a, Compute, f64>,
    /// exp(-i pi m / (N/2)), m < N/2, and exp(-i pi k / N), k < N (twiddles and the coefficient rotation).
    tw: ConstBuf<'a, f64>,
    zt: ConstBuf<'a, f64>,
    /// E_u rows `[u][0..=L][N]` (a variant fills rows 0..=its top).
    ept: DBuf<'a, Compute, W>,
    /// INTT rows 0, 1 of both parts of a context-layout ciphertext.
    bottom_rows: RowList<'a>,
    /// One per drop of `cfg.drops`, ascending.
    variants: Vec<SpruVariant<'a, W>>,
    /// conj o rot(1) (Galois 2N - 5), route-owned: not a rotation step of the key set.
    conjrot: DevKsk<'a, W>,
    fin_group: Option<KeyGroup<'a>>,
    /// Working ciphertexts (k = 1).
    work: [Ct<'a, W>; 3],
}

/// The tables of one drop: the E_u row list at the variant's top and the recombination masks at its output top.
struct SpruVariant<'a, W: DevWord> {
    drop: u32,
    /// Top limb of the key-ciphertext product (the E_u encodings) and of the recombination.
    top: usize,
    mask_top: usize,
    /// NTT rows `[u][0..=top]` of `Spru::ept`.
    ept_rows: RowList<'a>,
    /// The recombination as one linear transform of the product's output a (step 7): masks for a, conj a, rot(a, 1),
    /// conj rot(a, 1): -i me, i me, -i mo, i mo (me: even slots 1, odd i; mo swapped), at scale sf(mask_top); each
    /// mask holds Q rows 0..=mask_top, then the special rows.
    masks: DBuf<'a, Compute, W>,
}

impl<W: DevWord> SpruVariant<'_, W> {
    /// Rows of one mask.
    fn mask_rows(mask_top: usize, kp: usize) -> usize {
        mask_top + 1 + kp
    }

    /// Mask `i` at Q row 0 and at its first special row.
    fn mask(&self, i: usize, n: usize, kp: usize) -> (*const W, *const W) {
        let base = (self.masks.vptr() as *const W).wrapping_add(i * Self::mask_rows(self.mask_top, kp) * n);
        (base, base.wrapping_add((self.mask_top + 1) * n))
    }
}

impl<'a, W: DevWord> Spru<'a, W> {
    /// Arena bytes of the route (MemPlan section "bts.spru").
    pub fn bytes(lay: &Layout, cfg: &SpruCfg, chain: &pc_core::params::Chain) -> Result<usize> {
        let n = lay.n;
        let nu = 2 * cfg.n;
        let f = |k: usize| Arena::bytes_for::<f64>(k);
        let km = cfg.k_max;
        let ct = Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(lay, km));
        let ept_rows = km * nu * lay.nq;
        let rl = |k: usize| Arena::bytes_for::<u32>(k) + Arena::bytes_for::<i32>(k);
        let mut variants = 0;
        for d in cfg.drop_list() {
            let (top, mask_top) = cfg.tops(lay.nq, chain.level_unit as u32, d)?;
            let mr = SpruVariant::<W>::mask_rows(mask_top, lay.kp);
            variants += rl(km * nu * (top + 1)) + Arena::bytes_for::<W>(4 * mr * n) + rl(4 * mr);
        }
        Ok(KskShape::bottom(chain)?.bytes::<W>(n, KskLayout::Dense)
            + KeyGroup::bytes(1, true)
            + Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(lay, nu))
            + Arena::bytes_for::<u64>(km * nu)
            + 2 * f(2 * km * nu * n)
            + f(n)
            + f(2 * n)
            + Arena::bytes_for::<W>(ept_rows * n)
            + rl(4 * km)
            + variants
            + KskShape::full(chain).bytes::<W>(n, KskLayout::Dense)
            + 3 * ct)
    }

    /// Carve the route's buffers and build its key material from `secret` (dense) and `key` (the block key):
    /// the bottom-only `atob` switching key and the 2n key ciphertexts, secret-key encrypted at the top.
    #[allow(clippy::too_many_arguments)]
    pub fn new<M: DevStrategy<W>>(
        arena: &'a Arena,
        ctx: &Context<'_, W, M>,
        ex: &mut Executor<'_, W>,
        keygen: &mut crate::keygen::KeyGen<'_, W>,
        enc: &mut crate::encrypt::Encryptor<'_, W>,
        secret: &DevSecret<'_, W>,
        spru_secret: &mut DevSecret<'_, W>,
        sk: &SecretKey,
        key: &SpruKey,
        cfg: SpruCfg,
        sh: &Shadow,
        prng: &mut Prng,
    ) -> Result<Self> {
        let n = ctx.lay.n;
        cfg.check(n)?;
        let s = &ex.compute;
        let nu = 2 * cfg.n;
        let top = ctx.lay.nq - 1;
        let (mut atob, mut cs, mut conjrot) = Self::carve_material(arena, ctx, &cfg)?;
        // atob: dense s -> block s' (the encapsulation-key generator with s' as the "sparse" secret)
        spru_secret.load(ctx, s, &SecretKey { s: sk.s.clone(), s_sparse: key.s_block.clone() })?;
        keygen.generate(ctx, ex, spru_secret, &mut atob, prng)?;
        // key ciphertexts: S_u at the top, scale sf(L)
        let enc_host = Encoder::new(&ctx.chain);
        let cs_meta = cs.meta;
        let msgs: Vec<Vec<Vec<u64>>> =
            (0..nu).map(|u| enc_host.encode_slots(&key.key_slots(&cfg, n, u), n / 2, top, Payload::Real, cs_meta.nf)).collect::<std::result::Result<_, _>>()?;
        enc.encrypt_sk(ctx, &ex.compute, secret, &msgs, cs_meta, &mut cs, prng)?;
        keygen.generate(ctx, ex, secret, &mut conjrot, prng)?;
        let _ = sh;
        Self::from_material(arena, ctx, &ex.compute, cfg, atob, cs, conjrot)
    }

    /// The route's key material, carved and unfilled: the bottom-only `atob` key, the 2n key ciphertexts (one batch:
    /// the extmult reads them as one strided operand) with their metadata (level 0, scale sf(top)), and the
    /// conj o rot(1) key (Galois 2N - 5). Filled by `new` from the secret, or from a client's bundle.
    pub fn carve_material<M: DevStrategy<W>>(arena: &'a Arena, ctx: &Context<'_, W, M>, cfg: &SpruCfg) -> Result<(DevKsk<'a, W>, Ct<'a, W>, DevKsk<'a, W>)> {
        let lay = ctx.lay;
        let n = lay.n;
        let atob = DevKsk::carve(arena, KeyRole::EncDenseToSparse, KskShape::bottom(&ctx.chain)?, KskLayout::Dense, n)?;
        let sf_top = ctx.scale.sf((lay.nq - 1) as i32)?;
        let cs_meta = CtMeta { level: 0, deg: 1, nf: sf_top, slots: (n / 2) as u32, tag: Default::default() };
        let cs = Ct::new(cs_meta, CtBuf::new(arena, "bts.spru.cs", lay, 2 * cfg.n)?);
        let conjrot = DevKsk::carve(arena, KeyRole::Rot { step: 0, galois: 2 * n as u64 - 5 }, KskShape::full(&ctx.chain), KskLayout::Dense, n)?;
        Ok((atob, cs, conjrot))
    }

    /// The route from filled key material (`carve_material`): tables, masks and workspaces; no secret involved.
    pub fn from_material<M: DevStrategy<W>>(
        arena: &'a Arena,
        ctx: &Context<'_, W, M>,
        s: &Stream<Compute>,
        cfg: SpruCfg,
        atob: DevKsk<'a, W>,
        cs: Ct<'a, W>,
        conjrot: DevKsk<'a, W>,
    ) -> Result<Self> {
        let lay = ctx.lay;
        let n = lay.n;
        cfg.check(n)?;
        let nu = 2 * cfg.n;
        if cs.k() != nu {
            return Err(shape("Spru::from_material", format!("{} key ciphertexts, the route needs {nu}", cs.k())));
        }
        let enc_host = Encoder::new(&ctx.chain);
        let km = cfg.k_max;
        let ev = arena.alloc("bts.spru.ev", 2 * km * nu * n)?;
        let fb = arena.alloc("bts.spru.fft", 2 * km * nu * n)?;
        let pi = std::f64::consts::PI;
        let twv: Vec<f64> = (0..n / 2).flat_map(|m| { let a = -pi * m as f64 / (n / 2) as f64; [a.cos(), a.sin()] }).collect();
        let ztv: Vec<f64> = (0..n).flat_map(|k| { let a = -pi * k as f64 / n as f64; [a.cos(), a.sin()] }).collect();
        let tw = ConstBuf::new(arena, "bts.spru.tw", s, &twv)?;
        let zt = ConstBuf::new(arena, "bts.spru.tw", s, &ztv)?;
        // E rows `[c < k_max][u < 2n][0..=L][N]`; a variant's row list is ordered by ciphertext (prefixes serve every k)
        let ept = arena.alloc("bts.spru.ept", km * nu * lay.nq * n)?;
        let ws = std::mem::size_of::<W>() as u64;
        let ept_ptrs = ConstBuf::new(arena, "bts.spru.ept", s, &(0..km * nu).map(|u| ept.dptr() as u64 + (u * lay.nq * n) as u64 * ws).collect::<Vec<u64>>())?;
        let rows = lay.rows();
        let bl: Vec<(u32, i32)> = (0..km as u32).flat_map(|c| { let b = c * 2 * rows as u32; [(b, 0), (b + 1, 1), (b + rows as u32, 0), (b + rows as u32 + 1, 1)] }).collect();
        let bottom_rows = RowList::new(arena, s, &bl)?;
        let even: Vec<Complex> = (0..n / 2).map(|j| if j % 2 == 0 { Complex::new(1.0, 0.0) } else { Complex::new(0.0, 1.0) }).collect();
        let odd: Vec<Complex> = (0..n / 2).map(|j| if j % 2 == 1 { Complex::new(1.0, 0.0) } else { Complex::new(0.0, 1.0) }).collect();
        let times = |v: &[Complex], c: Complex| -> Vec<Complex> { v.iter().map(|z| Complex::new(z.re * c.re - z.im * c.im, z.re * c.im + z.im * c.re)).collect() };
        let (pi, mi) = (Complex::new(0.0, 1.0), Complex::new(0.0, -1.0));
        let mvals = [times(&even, mi), times(&even, pi), times(&odd, mi), times(&odd, pi)];
        let mut variants = vec![];
        for d in cfg.drop_list() {
            // trace and the product tree leave the ciphertext depth_primes below the variant's top
            let (vtop, mask_top) = cfg.tops(lay.nq, ctx.shape.unit, d)?;
            let er: Vec<(u32, i32)> = (0..km * nu).flat_map(|u| (0..=vtop).map(move |l| ((u * lay.nq + l) as u32, l as i32))).collect();
            let ept_rows = RowList::new(arena, s, &er)?;
            let sf_m = ctx.scale.sf(mask_top as i32)?;
            let ids: Vec<usize> = (0..=mask_top).chain(lay.nq..rows).collect();
            let mr = ids.len();
            let mut masks = arena.alloc("bts.spru.masks", 4 * mr * n)?;
            let mut host = vec![W::default(); 4 * mr * n];
            for (i, m) in mvals.iter().enumerate() {
                let c = enc_host.slot_coeffs(m, n / 2, Payload::Complex, sf_m)?;
                for (j, &id) in ids.iter().enumerate() {
                    let p = ctx.prime(id) as i128;
                    let row = &mut host[(i * mr + j) * n..(i * mr + j + 1) * n];
                    for (o, &x) in row.iter_mut().zip(&c) {
                        *o = W::from_u64(x.rem_euclid(p) as u64);
                    }
                }
            }
            masks.upload(s, &host)?;
            let ml: Vec<(u32, i32)> = (0..4).flat_map(|i| ids.iter().enumerate().map(move |(j, &id)| ((i * mr + j) as u32, id as i32))).collect();
            let mrows = RowList::new(arena, s, &ml)?;
            ntt::transform_rows(ctx, s, Dir::Forward, &mut masks, &mrows)?;
            variants.push(SpruVariant { drop: d, top: vtop, mask_top, ept_rows, masks });
        }
        let wk = || -> Result<Ct<'a, W>> { Ok(Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "bts.spru.work", lay, km)?)) };
        let work = [wk()?, wk()?, wk()?];
        s.sync()?;
        Ok(Spru {
            cfg,
            lay,
            atob,
            atob_group: None,
            groups: BTreeMap::new(),
            relin: None,
            rot_source: KskSource::Regen,
            relin_source: KskSource::Regen,
            cs,
            ept_ptrs,
            ev,
            fb,
            tw,
            zt,
            ept,
            bottom_rows,
            variants,
            conjrot,
            fin_group: None,
            work,
        })
    }

    /// Arena bytes of `bind` (the key tables).
    pub fn bind_bytes(cfg: &SpruCfg, ring_n: usize) -> usize {
        let mut b = KeyGroup::bytes(1, true) + KeyGroup::bytes(3, true);
        for r in cfg.trace_rounds(ring_n) {
            b += KeyGroup::bytes(r.len(), true);
        }
        b + cfg.product_steps().len() * KeyGroup::bytes(1, true)
    }

    /// Build the key tables (after `new`; they need the executor's staging buffers: the bottom-shape atob key is
    /// staged) from the session's key ring.
    pub fn bind(&mut self, arena: &'a Arena, s: &Stream<Compute>, keys: &dyn BtsKeyRing, stage: Option<(*const std::ffi::c_void, *const std::ffi::c_void)>) -> Result<()> {
        let n = self.lay.n;
        self.atob_group = Some(KeyGroup::new(arena, s, n, &[&self.atob as &dyn ks::KsKeyRef], stage)?);
        let mut lists = self.cfg.trace_rounds(n);
        lists.extend(self.cfg.product_steps().into_iter().map(|x| vec![x]));
        for l in lists {
            let refs: Vec<&dyn ks::KsKeyRef> = l.iter().map(|&st| keys.rot(st)).collect::<Result<_>>()?;
            self.groups.insert(l, KeyGroup::new(arena, s, n, &refs, stage)?);
        }
        self.fin_group = Some(KeyGroup::new(arena, s, n, &[keys.conj()?, keys.rot(1)?, &self.conjrot as &dyn ks::KsKeyRef], stage)?);
        self.relin = Some(KeyGroup::new(arena, s, n, &[keys.relin()?], stage)?);
        Ok(())
    }

    /// The output metadata the route produces for an input of any level (a function of the chain only).
    pub fn out_level(&self, unit: u32, drop: u32) -> u32 {
        self.cfg.depth_primes(unit) + unit * drop
    }

    /// Bootstrap the k <= `cfg.k_max` ciphertexts of `x` in place on the s = 1 route with correction `correction`
    /// (= CF - deg), on the variant of drop `drop`: one set of launches for the batch (the extmult and the
    /// recombination loop over its members).
    #[allow(clippy::too_many_arguments)]
    pub fn run<M: DevStrategy<W>>(&mut self, cx: &mut BtsCx<'_, '_, W, M>, x: &mut Ct<'_, W>, correction: u32, drop: u32) -> Result<()> {
        self.run_profiled(cx, x, correction, drop, None)
    }

    /// `run` with timing events recorded on the compute lane after each phase (`PHASES`: adjust, switch, encode,
    /// extmult, trace, product, finish), when `prof` holds `PHASES.len() + 1` events (the first at entry).
    #[allow(clippy::too_many_arguments)]
    pub fn run_profiled<M: DevStrategy<W>>(
        &mut self,
        cx: &mut BtsCx<'_, '_, W, M>,
        x: &mut Ct<'_, W>,
        correction: u32,
        drop: u32,
        prof: Option<&[pc_cuda::Event]>,
    ) -> Result<()> {
        let (ctx, sh) = (cx.ctx, cx.sh);
        let v = self.variants.iter().find(|v| v.drop == drop).ok_or_else(|| shape("spru", format!("no variant for drop {drop} (built {:?})", self.cfg.drop_list())))?;
        let ex = &mut *cx.ex;
        let ops = &mut *cx.ops;
        let kc = rotate::KsCfg { plans: cx.ks, source: self.rot_source };
        let kc_relin = rotate::KsCfg { plans: cx.ks, source: self.relin_source };
        let nogroup = || shape("spru", "key tables not bound (Spru::bind)");
        let (fin_g, relin_g) = (self.fin_group.as_ref().ok_or_else(nogroup)?, self.relin.as_ref().ok_or_else(nogroup)?);
        let mark = |ex: &Executor<'_, W>, i: usize| -> Result<()> {
            if let Some(p) = prof {
                if let Some(e) = p.get(i) {
                    e.record(&ex.compute)?;
                }
            }
            Ok(())
        };
        mark(ex, 0)?;
        let kb = x.k();
        if kb > self.cfg.k_max {
            return Err(Error::Mem { knob: "bts.spru.k_max", value: self.cfg.k_max, msg: format!("batch of {kb} ciphertexts") });
        }
        if ex.ks.ext.k < 3 {
            return Err(Error::Mem { knob: "MemCfg::k_max", value: ex.ks.ext.k, msg: "the SPRU recombination needs 3 extended-basis slots".into() });
        }
        for w in self.work.iter_mut() {
            w.buf.k = kb;
        }
        let lay = self.lay;
        let n = lay.n;
        let unit = ctx.shape.unit;
        let top_l = lay.nq - 1;
        let nu = 2 * self.cfg.n;
        // 1. integers m = v sf(L) 2^-correction at the bottom
        if x.meta.deg == 2 {
            ops.rescale(ctx, ex, sh, std::slice::from_mut(x))?;
        }
        let target = ctx.scale.sf(top_l as i32)? * 2f64.powi(-(correction as i32));
        let probe = sh.rescale(&sh.mult_scalar(&x.meta, 1.0)?.out)?.out;
        let adj = target / probe.nf;
        ops.mult_scalar_inplace(ctx, ex, sh, std::slice::from_mut(x), &[Complex::new(adj, 0.0)])?;
        ops.rescale(ctx, ex, sh, std::slice::from_mut(x))?;
        let bottom = ctx.shape.top_max() + 1 - unit; // level of the bottom composite limb set
        ops.drop_to_level(sh, std::slice::from_mut(x), bottom);
        mark(ex, 1)?;
        // 2. dense -> block key at the bottom (into work[2]; the input slot is free until the output)
        let grp = self.atob_group.as_ref().ok_or_else(|| shape("spru", "atob key group not bound (Spru::bind)"))?;
        let btop = (unit - 1) as usize;
        let [w0, w1, w2] = &mut self.work;
        key_switch(ctx, ex, kc, grp, x, w2, btop)?;
        mark(ex, 2)?;
        // 3. coefficient form, extraction, encode at the top
        ntt::transform_rows_prefix(ctx, &ex.compute, Dir::Inverse, &mut w2.buf.buf, &self.bottom_rows, 4 * kb)?;
        let (p0, p1) = (ctx.chain.q[0], ctx.chain.q[1]);
        let inv = pc_core::modmul::invmod(p0 % p1, p1);
        let q0 = p0 as f64 * p1 as f64;
        let delta = (q0 / (4.0 * std::f64::consts::PI * target)).powf(1.0 / self.cfg.h as f64);
        let xr = w2.buf.r();
        let ea = SpruExtractArgs {
            c0: xr.base,
            c1: (xr.base as *const W).wrapping_add(lay.poly_words()) as *const _,
            out: self.ev.vptr_mut(),
            p0,
            p1,
            inv,
            delta,
            ct_stride: lay.ct_words() as u64,
            h: self.cfg.h as i32,
            n: self.cfg.n as i32,
            b: self.cfg.block(n) as i32,
            k: kb as i32,
        };
        // SAFETY: rows 0, 1 of both parts of w2 (coefficient form) are live; ev holds 2n x N/2 complex
        launch_result("spru_extract", unsafe { (ctx.table.spru_extract)(&ea, ex.compute.raw()) })?;
        // E_u at sf(top)^2 / cs.nf (= sf(top) on the shipped route): the key ciphertexts keep the chain top's scale
        // with their top limbs dropped, so the product lands on the nominal scale of the level below the variant's top
        let sf_v = ctx.scale.sf(v.top as i32)?;
        let sf_e = sf_v * (sf_v / self.cs.meta.nf);
        // the 2n encodings at once: batched Stockham passes, then rounding into every limb of the variant's top
        let (mut src, mut dst) = (self.ev.vptr_mut(), self.fb.vptr_mut());
        let mut ns = 1;
        while ns < n {
            let fa = pc_kernels::SpruFftArgs { inp: src as *const _, out: dst, tw: self.tw.vptr(), ns: ns as i32, batch: (kb * nu) as i32 };
            // SAFETY: both buffers hold 2n N-point complex transforms; the table N/2 twiddles; in != out
            launch_result("spru_fft_pass", unsafe { (ctx.table.spru_fft_pass)(&fa, ex.compute.raw()) })?;
            std::mem::swap(&mut src, &mut dst);
            ns <<= 1;
        }
        let sa = pc_kernels::SpruScatterArgs {
            y: src as *const _,
            out: self.ept.vptr_mut(),
            zt: self.zt.vptr(),
            pr: ctx.pr(),
            scale: sf_e,
            nl: (v.top + 1) as i32,
            batch: (kb * nu) as i32,
            out_stride: (lay.nq * n) as u64,
        };
        // SAFETY: 2n FFT outputs of N complex; ept holds 2n x nq rows (v.top < nq); prime table by id
        launch_result("spru_scatter", unsafe { (ctx.table.spru_scatter)(&sa, ex.compute.raw()) })?;
        ntt::transform_rows_prefix(ctx, &ex.compute, Dir::Forward, &mut self.ept, &v.ept_rows, kb * nu * (v.top + 1))?;
        mark(ex, 3)?;
        // 4. acc = sum_u cs_u E_u (one plaintext-dot launch over the 2n key ciphertexts, rows 0..=v.top), rescale
        let level = unit * drop;
        let pm = PtMeta { level, deg: 1, nf: sf_e, tag: Default::default() };
        let (acc, tmp, acc2) = (w0, w1, w2);
        let cw = lay.ct_words() as u64;
        // the key ciphertexts are shared and the plaintexts are per member: one dot per member
        for c in 0..kb {
            let pd = pc_kernels::PtDotArgs {
                rot: View::of(&self.cs.buf, 0).r(),
                res: View::of(&acc.buf, c).r(),
                rot_stride: cw,
                res_stride: 0,
                pts: self.ept_ptrs.dptr().wrapping_add(c * nu) as *const *const std::ffi::c_void,
                pr: ctx.pr(),
                rs: lay.rs_q(v.top),
                npts: nu as i32,
                bstep: nu as i32,
                gstep: 1,
                k: 1,
                ptw: n as i32,
                pt_prow: 0,
                enc: 0xFFFF_FFFF,
                pad: 0,
            };
            // SAFETY: the 2n key ciphertexts (one strided batch), member c's E_u rows (pointer table entries
            // c 2n ..) and acc's ciphertext c are live context-layout buffers of rows 0..=L; acc is not an input
            launch_result("pt_dot", unsafe { (ctx.table.pt_dot)(&pd, ex.compute.raw()) })?;
        }
        acc.meta = sh.mult_pt(&CtMeta { level, ..self.cs.meta }, &pm)?.out;
        ops.rescale(ctx, ex, sh, std::slice::from_mut(acc))?;
        mark(ex, 4)?;
        // 5. trace over the h n period
        for steps in self.cfg.trace_rounds(n) {
            let g = self.groups.get(&steps).ok_or_else(|| shape("spru", format!("no key group for the trace round {steps:?}")))?;
            super::sparse::acc_round(ctx, ex, kc, sh, g, acc, &steps)?;
        }
        mark(ex, 5)?;
        // 6. product over the blocks (acc and acc2 alternate)
        let (mut a, mut b) = (acc, acc2);
        for st in self.cfg.product_steps() {
            let g = self.groups.get(&vec![st]).ok_or_else(|| shape("spru", format!("no key group for rotation {st}")))?;
            rotate::rotate(ctx, ex, kc, sh, g, a, st, tmp)?;
            crate::relin::mult(ctx, ex, kc_relin, sh, relin_g, a, tmp, b)?;
            ops.rescale(ctx, ex, sh, std::slice::from_mut(b))?;
            std::mem::swap(&mut a, &mut b);
        }
        mark(ex, 6)?;
        // 7. out = i [(conj a - a) me + (rot(conj a, 1) - rot(a, 1)) mo] (= 2 Im then the m0 + i m1 recombination)
        //    as one linear transform of a: one ModUp, the dots of conj / rot 1 / conj o rot 1 into the extended
        //    basis, their masks multiplied there, one ModDown; plus a's own term. Degree 2 (no rescale), like every
        //    route's raw exit.
        let a = &*a;
        let top = a.top(ctx.shape)?;
        if top != v.mask_top {
            return Err(shape("spru", format!("recombination at top {top}, masks built at {}", v.mask_top)));
        }
        let cw = lay.ct_words();
        let pw = lay.poly_words() as u64;
        let mask = |i: usize| v.mask(i, n, lay.kp);
        // per member: three rotations of one ciphertext fill three extended slots (k_max >= 3 is not assumed)
        for c in 0..kb {
            ks::modup(ctx, &ex.compute, kc.plans, &mut ex.ks, &mut ex.bconv_work, &a.buf, c, 1, 1, top)?;
            let va = View::of(&a.buf, c);
            let ext0 = View::of(&ex.ks.ext, 0);
            let io = DotIo { c1: va.part::<W>(1).base, c0: va.base, ct_stride: va.ct_stride, out: ext0.base, part_stride: pw, out_ct_stride: cw as u64, out_rot_stride: cw as u64, c0_mode: 1 };
            ks::dot(ctx, &ex.compute, kc.plans, &ex.ks, fin_g, 0, 3, 1, top, DotRows::All, io, fin_g.reads_in_place(kc.source))?;
            for j in 0..3 {
                let e = View::of(&ex.ks.ext, j);
                let (mq, mp) = mask(j + 1);
                mul_rows(ctx, &ex.compute, e, e, mq, top + 1, 0)?;
                mul_rows(ctx, &ex.compute, e, e, mp, lay.kp, lay.nq)?;
            }
            for j in 1..3 {
                ks::ew(ctx, &ex.compute, ks::Ew::Add, ext0, ext0, Some(View::of(&ex.ks.ext, j)), lay.rs_qp(top), 1, 2)?;
            }
            ks::moddown_inplace(ctx, &ex.compute, kc.plans, &mut ex.ks.ext, &mut ex.ks.conv, 1, top)?;
            let vx = View::of(&x.buf, c);
            mul_rows(ctx, &ex.compute, vx, va, mask(0).0, top + 1, 0)?;
            ks::ew(ctx, &ex.compute, ks::Ew::Add, vx, vx, Some(View::of(&ex.ks.ext, 0)), lay.rs_q(top), 1, 2)?;
        }
        let (slots, tag_in) = (x.meta.slots, x.meta.tag);
        let mm = PtMeta { level: a.meta.level, deg: 1, nf: ctx.scale.sf(v.mask_top as i32)?, tag: Default::default() };
        x.meta = sh.mult_pt(&a.meta, &mm)?.out;
        x.meta.slots = slots;
        let total = if tag_in.slots > 0 { tag_in.slots } else if slots > 0 { slots as i64 } else { (n / 2) as i64 };
        x.meta.tag = pc_shadow::tag::t_bootstrap(&tag_in, 1, total);
        mark(ex, 7)?;
        Ok(())
    }
}

/// out = keyswitch(x) with key 0 of `group` (no automorphism) at limb top `top`.
pub(crate) fn key_switch<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: rotate::KsCfg<'_, '_>,
    group: &KeyGroup<'_>,
    x: &Ct<'_, W>,
    out: &mut Ct<'_, W>,
    top: usize,
) -> Result<()> {
    let k = x.k();
    if out.k() != k {
        return Err(shape("key_switch", format!("input has {k} cts, output {}", out.k())));
    }
    ks::modup(ctx, &ex.compute, kc.plans, &mut ex.ks, &mut ex.bconv_work, &x.buf, 0, k, 1, top)?;
    let lay = ctx.lay;
    let v = View::of(&x.buf, 0);
    let io = DotIo {
        c1: v.part::<W>(1).base,
        c0: v.base,
        ct_stride: v.ct_stride,
        out: View::of(&out.buf, 0).base,
        part_stride: lay.poly_words() as u64,
        out_ct_stride: lay.ct_words() as u64,
        out_rot_stride: 0,
        c0_mode: 1,
    };
    let ndig = kc.plans.ndig(top)?;
    ks::stage_key(ctx, &ex.compute, kc.plans, &mut ex.ks, group, 0, ndig, true)?;
    ks::dot(ctx, &ex.compute, kc.plans, &ex.ks, group, 0, 1, k, top, DotRows::All, io, true)?;
    ks::moddown_inplace(ctx, &ex.compute, kc.plans, &mut out.buf, &mut ex.ks.conv, k, top)?;
    out.meta = x.meta;
    Ok(())
}

/// `out = inp * t` (prime-wise) on `nrows` rows starting at prime id `row0` of one ciphertext, both parts; `t` points
/// at the table's row of prime id `row0`, the next rows following it.
fn mul_rows<W: DevWord, M: DevStrategy<W>>(ctx: &Context<'_, W, M>, s: &Stream<Compute>, out: View, inp: View, t: *const W, nrows: usize, row0: usize) -> Result<()> {
    let n = ctx.lay.n;
    let off = |v: View| View { base: (v.base as *mut W).wrapping_add(row0 * n) as *mut _, ..v };
    let a = pc_kernels::MulTableArgs {
        out: off(out).r(),
        inp: off(inp).r(),
        t: t as *const _,
        pr: ctx.pr().wrapping_add(row0),
        n: nrows as i32,
        k: 1,
    };
    // SAFETY: rows [row0, row0 + nrows) of live context-layout buffers, nrows rows of the mask table at t;
    // element-wise (out may alias inp)
    launch_result("mul_table", unsafe { (ctx.table.mul_table)(&a, s.raw()) })?;
    Ok(())
}
