//! CKKS -> TFHE -> CKKS on the device: the switch of pc-tfhe (the bit-exact host reference) on a ciphertext at the
//! StC-first split. `TfheSwitch::apply` takes the split (limbs 0..3, eval form), evaluates a LUT on the coefficients it
//! is given (every other coefficient comes back an exact zero) and leaves the result in the same ciphertext in the
//! split format, ready for the route's finish:
//!
//!   INTT, gather + exact rescale by p3, p2 (the PCMM's ops) -> RESCALE_P1 (mod p0) -> ring switch s -> s'(X^S)
//!   (RS_DIGITS, NTT, MAC, INTT, + c0) -> RS_EXTRACT (LWEs of dimension N / S) -> LWE key switch (KS_DIGITS, four
//!   int8 GEMMs, KS_RECOMBINE) -> bootstrap (PBS_MS, PBS_INIT, PBS_ROTATE, PBS_EXTRACT: mod p0) -> repack
//!   (per key chunk REPACK_DIGITS, NTT, REPACK_MAC; INTT, REPACK_B) -> lift x p2 p3 into limbs 0..3, scatter, NTT.
//!
//! Ring-2^16 NTTs run on the PCMM launcher's NTT op with pc-tfhe's tables (the keys live in that NTT domain).
use std::collections::HashMap;
use std::ffi::c_void;
use std::time::Instant;

use pc_cuda::blas::lt;
use pc_cuda::{launch_result, Arena, Compute, DBuf, Stream};
use pc_kernels::args::{pcmm_op, tfhe_op, PcmmArgs, TfheArgs};
use pc_tfhe::modq::invmod;
use pc_tfhe::switch::Switcher;

use crate::context::{Context, DevStrategy};
use crate::ct::Ct;
use crate::error::{shape, Result};
use crate::matmul::gemm::I8Gemm;
use crate::ntt::{self, Parts, Span};

type W = u32;
const PLANES: usize = 4;
const CHUNK: usize = 64;
const WS: usize = 64 << 20;

fn mu(p: u64) -> u64 {
    ((1u128 << 64) / p as u128) as u64
}

/// The device image of a host bootstrapping key (RNS, u32): the key [pair][m][e][h][thread][4] in Montgomery form
/// (NTT form, times n^-1 2^32 per prime: the rotation's inverse NTT skips its scaling, its MAC reduces once per sum),
/// where PBS_ROTATE's thread th of 1024 handles word k = th + 1024 e of [prime][n] and loads as uint4 h the values
/// (c t = 2 h, part 0), (2 h, 1), (2 h + 1, 0), (2 h + 1, 1); the twiddles [prime][forward | inverse][n], then NTT(X^e - 1)'s
/// tables (`Rns::monomial_tables`) pm1 [prime][2 n] (Montgomery form) and E [prime][n], and the twiddles' companions; and the PBS ops' constants (args.cuh:
/// c[2..8) = n^-1 per prime with companions, q_a | q_b << 32, q_a^-1 mod q_b | its companion << 32; `m` the gadget's
/// multiplier).
pub struct PbsImage {
    pub bsk: Vec<u32>,
    pub tw: Vec<u32>,
    pub tws: Vec<u32>,
    pub c: [u64; 8],
    pub m: u64,
    pub q: u64,
}

fn shoup32(w: u64, q: u64) -> u32 {
    ((w << 32) / q) as u32
}

pub fn pbs_image(k: &pc_tfhe::pbs::BootstrapKey) -> PbsImage {
    let rns = &k.rns;
    let (qa, qb) = (rns.ntt[0].q, rns.ntt[1].q);
    let lv = k.gadget.levels;
    let ninv = [rns.ntt[0].tables().2, rns.ntt[1].tables().2];
    let mut bsk = Vec::new();
    let mont = |v: u64, q: u64| ((v << 32) % q) as u32;
    let n = rns.n();
    assert_eq!(lv, 2, "PBS_ROTATE's key layout is for two levels");
    for g in k.ggsw.iter().flatten() {
        for e in 0..4 {
            for h in 0..2 {
                for th in 0..n / 2 {
                    let (pr, j) = ((th + e * n / 2) / n, (th + e * n / 2) % n);
                    let q = [qa, qb][pr];
                    for ct in [2 * h, 2 * h + 1] {
                        let r = &g.rows()[ct / 2][ct % 2][pr];
                        for x in [r.0[j], r.1[j]] {
                            bsk.push(mont(pc_tfhe::modq::mulmod(x, ninv[pr], q), q));
                        }
                    }
                }
            }
        }
    }
    let mut tw = Vec::new();
    let mut tws = Vec::new();
    for (pr, &q) in [qa, qb].iter().enumerate() {
        let (psi, ipsi, _) = rns.ntt[pr].tables();
        for v in psi.iter().chain(ipsi) {
            tw.push(*v as u32);
            tws.push(shoup32(*v, q));
        }
    }
    let mono = [0, 1].map(|p| rns.monomial_tables(p));
    for (pr, &q) in [qa, qb].iter().enumerate() {
        tw.extend(mono[pr].0.iter().map(|&v| mont(v, q)));
    }
    for m in &mono {
        tw.extend(&m.1);
    }
    let c = [k.gadget.base_log as u64, lv as u64, ninv[0], shoup32(ninv[0], qa) as u64, ninv[1], shoup32(ninv[1], qb) as u64, qa | qb << 32, rns.inv | (shoup32(rns.inv, qb) as u64) << 32];
    PbsImage { bsk, tw, tws, c, m: k.gadget.m, q: rns.q }
}

/// A test vector (mod Q) as its residues [prime][n] (PBS_INIT's input).
pub fn tv_residues(k: &pc_tfhe::pbs::BootstrapKey, tv: &[u64]) -> Vec<u32> {
    k.rns.residues(tv).iter().flat_map(|r| r.iter().map(|&x| x as u32)).collect()
}

/// Device milliseconds per stage of the last `apply` (host walls around synced stages: diagnostic).
#[derive(Clone, Debug, Default)]
pub struct Timings {
    pub to_lwe: f64,
    pub key_switch: f64,
    pub bootstrap: f64,
    pub repack: f64,
}

pub struct TfheSwitch<'a> {
    n: usize,
    nb: usize,
    n_sub: usize,
    nl: usize,
    sfac: usize,
    max: usize,
    p: pc_tfhe::params::Params,
    p0: u64,
    p1: u64,
    rs_m: u64,
    ks_m: u64,
    pbs_c: [u64; 8],
    pbs_m: u64,
    tw_f: DBuf<'a, Compute, u32>,
    tw_i: DBuf<'a, Compute, u32>,
    ninv: [u32; 2],
    rkey: DBuf<'a, Compute, u32>,
    ksk: Vec<DBuf<'a, Compute, u8>>,
    bsk: DBuf<'a, Compute, u32>,
    btw: DBuf<'a, Compute, u32>,
    btw_s: DBuf<'a, Compute, u32>,
    rk: DBuf<'a, Compute, u32>,
    stage: DBuf<'a, Compute, u32>,
    low: DBuf<'a, Compute, u32>,
    dig: DBuf<'a, Compute, u32>,
    rs: DBuf<'a, Compute, u32>,
    idx: DBuf<'a, Compute, u32>,
    inv: DBuf<'a, Compute, u32>,
    ext: DBuf<'a, Compute, u32>,
    kdig: DBuf<'a, Compute, u8>,
    g: DBuf<'a, Compute, i32>,
    lwe: DBuf<'a, Compute, u32>,
    ms: DBuf<'a, Compute, u32>,
    acc: DBuf<'a, Compute, u32>,
    outs: DBuf<'a, Compute, u32>,
    tv: DBuf<'a, Compute, u32>,
    rdig: DBuf<'a, Compute, u32>,
    racc: DBuf<'a, Compute, u32>,
    ws: DBuf<'a, Compute, u8>,
    gemms: HashMap<usize, I8Gemm<'a>>,
}

impl<'a> TfheSwitch<'a> {
    /// Arena bytes for `max` switched coefficients per call (keys + work) on a ring of degree `n`.
    pub fn bytes(n: usize, p: &pc_tfhe::params::Params, max: usize) -> usize {
        let (nb, nl) = (p.n_br, p.n_lwe);
        let n_sub = n / p.rs_factor;
        let k = n_sub * p.ks.1;
        let words32 = 2 * 3 * n + p.rs.1 * 2 * n + nb * p.rp.1 * 4 * n + 8 * n + 2 * n + p.rs.1 * n + 2 * n + max + n
            + max * (n_sub + 1) + PLANES * (nl + 1) * max + 2 * max * (nl + 1) + max * (nb + 1) + 2 * CHUNK * p.rp.1 * n + 4 * n
            + (nl / 2 * 3) * 2 * p.pbs.1 * 4 * nb + 14 * nb + max * 4 * nb + 2 * nb;
        let bytes8 = PLANES * (nl + 1) * k + max * k + WS;
        4 * words32 + bytes8 + 64 * 256
    }

    /// Upload `sw`'s keys (host-generated: pc-tfhe `Switcher::generate`) and carve the work buffers.
    pub fn new(arena: &'a Arena, s: &Stream<Compute>, sw: &Switcher, max: usize) -> Result<Self> {
        let (n, p) = (sw.n, sw.params.clone());
        let (nb, nl, sfac) = (p.n_br, p.n_lwe, p.rs_factor);
        let n_sub = n / sfac;
        let up32 = |what: &'static str, v: &[u32]| -> Result<DBuf<'a, Compute, u32>> {
            let mut b = arena.alloc::<Compute, u32>(what, v.len())?;
            b.upload(s, v)?;
            Ok(b)
        };
        let (t0f, t0i, ni0) = sw.ntt_p[0].tables();
        let (t1f, t1i, ni1) = sw.ntt_p[1].tables();
        let tw_f: Vec<u32> = [t0f, t1f, t0f].iter().flat_map(|t| t.iter().map(|&x| x as u32)).collect();
        let tw_i: Vec<u32> = [t0i, t1i, t0i].iter().flat_map(|t| t.iter().map(|&x| x as u32)).collect();
        let rkey: Vec<u32> = sw.rs.keys().iter().flat_map(|(a, b)| a.iter().chain(b).map(|&x| x as u32)).collect();
        // the key-switching key as K-contiguous columns W[o][i levels + t] in four balanced base-256 planes
        let lv = p.ks.1;
        let k = n_sub * lv;
        let cols = nl + 1;
        let mut w = vec![0i64; cols * k];
        for i in 0..n_sub {
            for t in 0..lv {
                let key = &sw.ksk.keys[i][t];
                for o in 0..cols {
                    w[o * k + i * lv + t] = pc_tfhe::modq::centered(if o < nl { key.a[o] } else { key.b }, sw.p0);
                }
            }
        }
        let planes = pc_core::pcmm_keys::digit_planes(&w, PLANES)?;
        drop(w);
        let mut ksk = Vec::with_capacity(PLANES);
        for u in 0..PLANES {
            let mut b = arena.alloc::<Compute, u8>("tfhe.ksk", cols * k)?;
            let bytes: Vec<u8> = planes[u * cols * k..(u + 1) * cols * k].iter().map(|&x| x as u8).collect();
            b.upload(s, &bytes)?;
            ksk.push(b);
        }
        drop(planes);
        let img = pbs_image(&sw.bsk);
        let mut rk = Vec::with_capacity(nb * p.rp.1 * 4 * n);
        for kk in 0..nb {
            for t in 0..p.rp.1 {
                for l in 0..2 {
                    let (a, b) = &sw.repack.keys()[kk][t][l];
                    rk.extend(a.iter().chain(b).map(|&x| x as u32));
                }
            }
        }
        let me = TfheSwitch {
            n,
            nb,
            n_sub,
            nl,
            sfac,
            max,
            p0: sw.p0,
            p1: sw.p1,
            rs_m: sw.rs.gadget.m,
            ks_m: sw.ksk.gadget.m,
            pbs_c: img.c,
            pbs_m: img.m,
            tw_f: up32("tfhe.tw", &tw_f)?,
            tw_i: up32("tfhe.tw", &tw_i)?,
            ninv: [ni0 as u32, ni1 as u32],
            rkey: up32("tfhe.rkey", &rkey)?,
            ksk,
            bsk: up32("tfhe.bsk", &img.bsk)?,
            btw: up32("tfhe.btw", &img.tw)?,
            btw_s: up32("tfhe.btw", &img.tws)?,
            rk: up32("tfhe.rk", &rk)?,
            stage: arena.alloc("tfhe.work", 8 * n)?,
            low: arena.alloc("tfhe.work", 2 * 2 * n)?,
            dig: arena.alloc("tfhe.work", p.rs.1 * n)?,
            rs: arena.alloc("tfhe.work", 2 * n)?,
            idx: arena.alloc("tfhe.work", max)?,
            inv: arena.alloc("tfhe.work", n)?,
            ext: arena.alloc("tfhe.work", max * (n_sub + 1))?,
            kdig: arena.alloc("tfhe.work", max * k)?,
            g: arena.alloc("tfhe.work", PLANES * cols * max)?,
            lwe: arena.alloc("tfhe.work", max * cols)?,
            ms: arena.alloc("tfhe.work", max * cols)?,
            acc: arena.alloc("tfhe.work", max * 4 * nb)?,
            outs: arena.alloc("tfhe.work", max * (nb + 1))?,
            tv: arena.alloc("tfhe.work", 2 * nb)?,
            rdig: arena.alloc("tfhe.work", 2 * CHUNK * p.rp.1 * n)?,
            racc: arena.alloc("tfhe.work", 4 * n)?,
            ws: arena.alloc("tfhe.ws", WS)?,
            gemms: HashMap::new(),
            p,
        };
        Ok(me)
    }

    fn go<M: DevStrategy<W>>(ctx: &Context<'_, W, M>, s: &Stream<Compute>, a: &TfheArgs) -> Result<()> {
        // SAFETY: the pointers in `a` address this switch's buffers / the caller's ciphertext, sized for the op
        launch_result("tfhe", unsafe { (ctx.table.tfhe)(a, s.raw()) })?;
        Ok(())
    }

    fn gp<M: DevStrategy<W>>(ctx: &Context<'_, W, M>, s: &Stream<Compute>, a: &PcmmArgs) -> Result<()> {
        // SAFETY: as `go` (the PCMM launcher's chain / NTT / add ops on this switch's buffers)
        launch_result("pcmm", unsafe { (ctx.table.pcmm)(a, s.raw()) })?;
        Ok(())
    }

    /// The LUT (`tv` + `offset`, mod q_br; pc-tfhe pbs `test_vector` / `sign01`) on the split's coefficients
    /// `positions` (slot i of a real payload is coefficient brev((i - 1) mod N/2): pc-check tfhe). `x` is overwritten
    /// with the result in the split format (level, degree and scale unchanged). `lt_`: the cuBLASLt handle of the
    /// key switch's GEMMs (a plan per batch size, built on first use).
    pub fn apply<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, lt_: &'a lt::Lt, x: &mut Ct<'_, W>, positions: &[u32], tv: &[u32], offset: u64) -> Result<Timings> {
        let (n, nb, nl, n_sub) = (self.n, self.nb, self.nl, self.n_sub);
        let b = positions.len();
        if b == 0 || b > self.max {
            return Err(shape("tfhe switch", format!("{b} coefficients, the switch holds 1..={}", self.max)));
        }
        if ctx.lay != x.buf.lay || ctx.top_of(x.meta.level)? != 3 || tv.len() != 2 * nb {
            return Err(shape("tfhe switch", format!("ciphertext at level {} (the split holds limbs 0..3) / test vector of {}", x.meta.level, tv.len())));
        }
        let (p0, p1) = (self.p0, self.p1);
        let q = &ctx.chain.q;
        let c4 = [q[0] as u32, q[1] as u32, q[2] as u32, q[3] as u32];
        let cmu = [mu(q[0]), mu(q[1]), mu(q[2]), mu(q[3])];
        let mut inv_c = [0u32; 8];
        for i in 0..3 {
            inv_c[i] = invmod(q[3] % q[i], q[i]) as u32;
        }
        for i in 0..2 {
            inv_c[3 + i] = invmod(q[2] % q[i], q[i]) as u32;
        }
        let q23 = q[2] as u128 * q[3] as u128;
        let lift = [(q23 % q[0] as u128) as u32, (q23 % q[1] as u128) as u32];
        let chain = |opc: i32| PcmmArgs { op: opc, n: n as i32, p: c4, mu: cmu, ..PcmmArgs::default() };
        let pa = PcmmArgs { n: n as i32, p: [p0 as u32, p1 as u32, p0 as u32, 0], mu: [mu(p0), mu(p1), mu(p0), 0], c: [self.ninv[0], self.ninv[1], self.ninv[0], 0, 0, 0, 0, 0], ..Default::default() };
        let mut tm = Timings::default();
        let mut inv = vec![u32::MAX; n];
        for (r, &j) in positions.iter().enumerate() {
            inv[j as usize] = r as u32;
        }
        self.idx.upload_at(s, 0, positions)?;
        self.inv.upload(s, &inv)?;
        self.tv.upload(s, tv)?;
        let t0 = Instant::now();

        // the split to LWEs: coefficient form, mod q0 (staging part 0 = c1, part 1 = c0)
        ntt::inverse(ctx, s, &mut x.buf, Span::Q { top: 3 }, Parts::Both)?;
        let stage = self.stage.vptr_mut();
        Self::gp(ctx, s, &PcmmArgs { ct: x.buf.r(), i0: 0, x: stage, ..chain(pcmm_op::GATHER) })?;
        Self::gp(ctx, s, &PcmmArgs { y: stage, x: self.low.vptr_mut(), c: inv_c, ..chain(pcmm_op::IN_RESCALE) })?;
        // mod p0: rs = [c1, c0] (the staging's part order)
        let tb = TfheArgs { n: n as i32, q: p0, ..Default::default() };
        Self::go(ctx, s, &TfheArgs { op: tfhe_op::RESCALE_P1, count: 1, x: self.rs.vptr_mut(), y: self.low.vptr(), c: [invmod(p1 % p0, p0), p1, 0, 0, 0, 0, 0, mu(p0)], ..tb })?;
        // ring switch: digits of c1, NTT, MAC against the key -> stage [c0', c1'] (NTT form), INTT, + c0
        let (rs_lv, rs_bl) = (self.p.rs.1, self.p.rs.0);
        Self::go(ctx, s, &TfheArgs { op: tfhe_op::RS_DIGITS, count: 1, x: self.dig.vptr_mut(), y: self.rs.vptr(), m: self.rs_m, c: [rs_bl as u64, rs_lv as u64, 0, 0, 0, 0, 0, 0], ..tb })?;
        Self::gp(ctx, s, &PcmmArgs { op: pcmm_op::NTT, count: rs_lv as i32, i0: rs_lv as i32, x: self.dig.vptr_mut(), tw: self.tw_f.vptr(), ..pa })?;
        let st = self.stage.vptr_mut();
        Self::go(ctx, s, &TfheArgs { op: tfhe_op::MAC, count: rs_lv as i32, i1: 2, x: st, y: self.dig.vptr(), z: self.rkey.vptr(), c: [0, 0, 0, 0, 0, 0, 0, mu(p0)], ..tb })?;
        Self::gp(ctx, s, &PcmmArgs { op: pcmm_op::INTT, count: 2, i0: 2, x: st, tw: self.tw_i.vptr(), ..pa })?;
        let c0 = (self.rs.vptr() as *const u32).wrapping_add(n) as *const c_void;
        Self::gp(ctx, s, &PcmmArgs { op: pcmm_op::ADD, count: 1, i0: n as i32, x: st, y: c0, ..pa })?;
        Self::go(ctx, s, &TfheArgs { op: tfhe_op::RS_EXTRACT, count: b as i32, i1: n_sub as i32, i2: self.sfac as i32, x: self.ext.vptr_mut(), y: self.stage.vptr(), z: self.idx.vptr(), ..tb })?;
        s.sync()?;
        tm.to_lwe = t0.elapsed().as_secs_f64() * 1e3;

        // LWE key switch
        let t1 = Instant::now();
        let (ks_lv, cols) = (self.p.ks.1, nl + 1);
        let k = n_sub * ks_lv;
        Self::go(ctx, s, &TfheArgs { op: tfhe_op::KS_DIGITS, count: b as i32, i1: n_sub as i32, x: self.kdig.vptr_mut(), y: self.ext.vptr(), q: p0, m: self.ks_m, c: [self.p.ks.0 as u64, ks_lv as u64, 0, 0, 0, 0, 0, 0], ..Default::default() })?;
        if !self.gemms.contains_key(&b) {
            self.gemms.insert(b, I8Gemm::new(lt_, b as u64, cols as u64, k as u64, 8, WS)?);
        }
        let gemm = &self.gemms[&b];
        let gp_ = self.g.dptr_mut();
        for u in 0..PLANES {
            // SAFETY: operands are this switch's buffers of the plan's lengths, on `s`
            unsafe { gemm.run(s, self.kdig.dptr() as *const i8, self.ksk[u].dptr() as *const i8, gp_.add(u * cols * b), false, &mut self.ws)? };
        }
        Self::go(ctx, s, &TfheArgs { op: tfhe_op::KS_RECOMBINE, count: b as i32, i1: n_sub as i32, i2: nl as i32, x: self.lwe.vptr_mut(), y: self.g.vptr(), z: self.ext.vptr(), q: p0, c: [0, 0, PLANES as u64, 0, 0, 0, 0, 0], ..Default::default() })?;
        s.sync()?;
        tm.key_switch = t1.elapsed().as_secs_f64() * 1e3;

        // bootstrap
        let t2 = Instant::now();
        let qb = self.p.q_br;
        let pb = TfheArgs { n: nb as i32, count: b as i32, i1: nl as i32, q: qb, ..Default::default() };
        let mut cx = [0u64; 8];
        cx[6] = self.pbs_c[6];
        cx[7] = self.pbs_c[7];
        Self::go(ctx, s, &TfheArgs { op: tfhe_op::PBS_MS, x: self.ms.vptr_mut(), y: self.lwe.vptr(), m: pc_tfhe::gadget::fixed_ratio(2 * nb as u64, p0), c: [p0, 0, 0, 0, 0, 0, 0, 0], ..pb })?;
        Self::go(ctx, s, &TfheArgs { op: tfhe_op::PBS_INIT, x: self.acc.vptr_mut(), y: self.tv.vptr(), z: self.ms.vptr(), c: cx, ..pb })?;
        Self::go(ctx, s, &TfheArgs {
            op: tfhe_op::PBS_ROTATE,
            x: self.acc.vptr_mut(),
            y: self.bsk.vptr(),
            z: self.ms.vptr(),
            tw: self.btw.vptr(),
            tws: self.btw_s.vptr(),
            m: self.pbs_m,
            c: self.pbs_c,
            ..pb
        })?;
        Self::go(ctx, s, &TfheArgs { op: tfhe_op::PBS_EXTRACT, x: self.outs.vptr_mut(), y: self.acc.vptr(), m: pc_tfhe::gadget::fixed_ratio(p0, qb), c: [offset, p0, 0, 0, 0, 0, cx[6], cx[7]], ..pb })?;
        s.sync()?;
        tm.bootstrap = t2.elapsed().as_secs_f64() * 1e3;

        // repack into [part][limb][n] mod p0 p1, lift x p2 p3 into the staging, scatter, NTT
        let t3 = Instant::now();
        let rp_lv = self.p.rp.1;
        let zero = vec![0u32; 4 * n];
        self.racc.upload(s, &zero)?;
        let rb = TfheArgs { n: n as i32, count: b as i32, i1: nb as i32, i3: CHUNK as i32, q: p0, c: [self.p.rp.0 as u64, rp_lv as u64, p1 % p0, 0, p1, 0, mu(p1), mu(p0)], ..Default::default() };
        for k0 in (0..nb).step_by(CHUNK) {
            Self::go(ctx, s, &TfheArgs { op: tfhe_op::REPACK_DIGITS, i0: k0 as i32, x: self.rdig.vptr_mut(), y: self.outs.vptr(), w: self.inv.vptr(), ..rb })?;
            Self::gp(ctx, s, &PcmmArgs { op: pcmm_op::NTT, count: (2 * CHUNK * rp_lv) as i32, i0: (CHUNK * rp_lv) as i32, x: self.rdig.vptr_mut(), tw: self.tw_f.vptr(), ..pa })?;
            let kz = (self.rk.vptr() as *const u32).wrapping_add(k0 * rp_lv * 4 * n) as *const c_void;
            Self::go(ctx, s, &TfheArgs { op: tfhe_op::REPACK_MAC, x: self.racc.vptr_mut(), y: self.rdig.vptr(), z: kz, ..rb })?;
        }
        for part in 0..2 {
            let xp = (self.racc.vptr_mut() as *mut u32).wrapping_add(part * 2 * n) as *mut c_void;
            Self::gp(ctx, s, &PcmmArgs { op: pcmm_op::INTT, count: 2, i0: 1, x: xp, tw: self.tw_i.vptr(), ..pa })?;
        }
        Self::go(ctx, s, &TfheArgs { op: tfhe_op::REPACK_B, x: self.racc.vptr_mut(), y: self.outs.vptr(), z: self.idx.vptr(), ..rb })?;
        for sp in 0..2 {
            // staging part 0 = c1 (racc part 1), staging part 1 = c0 (racc part 0)
            let f = (self.racc.vptr() as *const u32).wrapping_add((1 - sp) * 2 * n) as *const c_void;
            let dst = (self.stage.vptr_mut() as *mut W).wrapping_add(sp * 4 * n) as *mut c_void;
            Self::gp(ctx, s, &PcmmArgs { y: f, x: dst, c: [lift[0], lift[1], 0, 0, 0, 0, 0, 0], ..chain(pcmm_op::LIFT) })?;
        }
        Self::gp(ctx, s, &PcmmArgs { ct: x.buf.r(), i0: 0, y: self.stage.vptr(), ..chain(pcmm_op::SCATTER) })?;
        ntt::forward(ctx, s, &mut x.buf, Span::Q { top: 3 }, Parts::Both)?;
        s.sync()?;
        tm.repack = t3.elapsed().as_secs_f64() * 1e3;
        Ok(tm)
    }
}
