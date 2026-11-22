//! CKKS -> TFHE -> CKKS on the device: the switch of pc-tfhe (the bit-exact host reference) on a ciphertext at the
//! StC-first split. `TfheSwitch::apply` takes the split (limbs 0..3, 0..2 on a lean chain; eval form), evaluates a
//! LUT on the coefficients it is given (every other coefficient comes back an exact zero) and leaves the result in
//! the same ciphertext in the split format, ready for the route's finish:
//!
//!   INTT, gather + exact rescale by p3, p2 (the PCMM's ops) -> RESCALE_P1 (mod p0) -> ring switch s -> s'(X^S)
//!   (RS_DIGITS, NTT, MAC, INTT, + c0) -> RS_EXTRACT (LWEs of dimension N / S) -> LWE key switch (KS_DIGITS, four
//!   int8 GEMMs, KS_RECOMBINE) -> bootstrap (PBS_MS, PBS_INIT, PBS_ROTATE, PBS_EXTRACT: mod p0) -> repack
//!   (`DevRepack`) -> lift x p2 p3 into limbs 0..3 (x p2 into 0..2 on a lean chain), scatter, NTT.
//!
//! Ring-2^16 NTTs run on the PCMM launcher's NTT op with pc-tfhe's tables (the keys live in that NTT domain).
use std::collections::HashMap;
use std::ffi::c_void;
use std::time::Instant;

use pc_cuda::blas::lt;
use pc_cuda::{launch_result, Arena, Compute, DBuf, Device, Event, Stream};
use pc_kernels::args::{pcmm_op, tfhe_op, KernelTable, PcmmArgs, TfheArgs};
use pc_tfhe::modq::invmod;
use pc_tfhe::switch::Switcher;

use crate::context::{Context, DevStrategy};
use crate::ct::Ct;
use crate::error::{shape, Result};
use crate::matmul::gemm::I8Gemm;
use crate::ntt::{self, Parts, Span};

type W = u32;
const PLANES: usize = 4;
/// Batches up to this many LWEs take the bootstrap's latency path (per window of KB_WINDOW groups: PBS_KEYBUNDLE,
/// then PBS_ROTATE on a cluster of two blocks per LWE: 64 KB of bundles per LWE and group): up to half the SMs.
pub const KB_MAX: usize = 94;
/// The extended rotation's latency path (four-block clusters only, args.cuh CL4_MAX): up to this many LWEs.
pub const KB_MAX_EXT: usize = 46;
/// Chunks up to this many values take the latency key (Params::lat_group) when the switch holds one: up to the
/// four-block clusters' CL4_MAX now that the key bundles are shared across LWEs (32 LWEs 1.93 -> 1.83 ms, 46 2.17 ->
/// 2.04 against the groups of 4).
pub const LAT_MAX: usize = 46;

/// LAT_MAX, overridable by PC_TFHE_LAT_MAX (read once).
/// Chunks above LAT_MAX and up to this many values take the mid-batch latency key (Params::lat_group2): four-block
/// clusters up to CL4_MAX LWEs, two-block ones up to KB_MAX (the key bundles of several LWEs per block,
/// keybundle_nb_k: 48 LWEs 2.71 -> 2.27 ms, 94 3.50 -> 3.18 against the batch key's groups of 3).
pub const LAT2_MAX: usize = 94;

pub fn lat_cap() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("PC_TFHE_LAT_MAX").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(LAT_MAX))
}

/// LAT2_MAX, overridable by PC_TFHE_LAT2_MAX (read once).
pub fn lat2_cap() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("PC_TFHE_LAT2_MAX").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(LAT2_MAX))
}

/// Groups per latency-path window: 32 keeps the bundles at 2 MB per LWE.
pub const KB_WINDOW: usize = 32;
/// LWEs per repack GEMM pass (the digits and packed buffers are sized for it) for a switch of `max` values per pass:
/// 1024 from 1024 values (one int8 GEMM of 1024 rows per key plane: -16 % repack time at 16384 values against 256),
/// 256 below; PC_TFHE_REPACK_RC overrides (a multiple of 256 up to 4096).
fn repack_rc(max: usize) -> usize {
    let env = std::env::var("PC_TFHE_REPACK_RC").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|&r| r % 256 == 0 && (256..=4096).contains(&r));
    env.unwrap_or(if max >= 1024 { 1024 } else { 256 })
}
/// The repack columns' int8 planes (centred residues < 2^27) and the repack GEMMs' workspace.
const KPL: usize = 4;
const WSR: usize = 32 << 20;
const WS: usize = 64 << 20;

fn mu(p: u64) -> u64 {
    ((1u128 << 64) / p as u128) as u64
}

/// The device image of a host bootstrapping key (RNS, u32): the key [group][m][e][h][thread][4] in Montgomery form
/// (NTT form, times n^-1 2^32 per prime: the rotation's inverse NTT skips its scaling, its MAC reduces once per sum),
/// where PBS_ROTATE's thread th of 1024 handles word k = th + 1024 e of [prime][n] and loads as uint4 h < levels the
/// values (row r = 2 h, part 0), (2 h, 1), (2 h + 1, 0), (2 h + 1, 1), row r = c levels + t; the twiddles [prime][forward | inverse][n], then NTT(X^e - 1)'s
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
    assert!(lv == 1 || lv == 2, "PBS_ROTATE takes one or two levels");
    // per GGSW (in parallel): its rows' c0 and c1 (a seeded key's masks expanded here, not kept)
    let ggsws: Vec<&pc_tfhe::pbs::Ggsw> = k.ggsw.iter().flatten().collect();
    let parts: Vec<Vec<u32>> = {
        use rayon::prelude::*;
        ggsws
            .par_iter()
            .map(|g| {
                let rows: Vec<[(&[u64], std::borrow::Cow<'_, [u64]>); 2]> = (0..2 * lv).map(|ct| [0, 1].map(|pr| (g.c0(ct / lv, ct % lv, pr), g.c1(ct / lv, ct % lv, pr)))).collect();
                let mut out = Vec::with_capacity(4 * lv * 4 * n / 2);
                for e in 0..4 {
                    for h in 0..lv {
                        for th in 0..n / 2 {
                            let (pr, j) = ((th + e * n / 2) / n, (th + e * n / 2) % n);
                            let q = [qa, qb][pr];
                            for ct in [2 * h, 2 * h + 1] {
                                let r = &rows[ct][pr];
                                for x in [r.0[j], r.1[j]] {
                                    out.push(mont(pc_tfhe::modq::mulmod(x, ninv[pr], q), q));
                                }
                            }
                        }
                    }
                }
                out
            })
            .collect()
    };
    bsk.reserve(parts.iter().map(|v| v.len()).sum());
    for v in parts {
        bsk.extend(v);
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
    // c[5] = nu stays 0 (one component): the extended rotation sets it per launch
    let c = [k.gadget.base_log as u64, lv as u64, 0, 0, 0, 0, qa | qb << 32, rns.inv | (shoup32(rns.inv, qb) as u64) << 32];
    PbsImage { bsk, tw, tws, c, m: k.gadget.m, q: rns.q }
}

/// A test vector (mod Q) as its residues [prime][n] (PBS_INIT's input).
pub fn tv_residues(k: &pc_tfhe::pbs::BootstrapKey, tv: &[u64]) -> Vec<u32> {
    k.rns.residues(tv).iter().flat_map(|r| r.iter().map(|&x| x as u32)).collect()
}

/// The device repack (pc-tfhe `RepackKey`, bit for bit): per `repack_rc` LWEs REPACK_DIGITS -> REPACK_MAC (the GEMM in
/// R_{n_br} mod p0 P; split K into int64 sums) -> REPACK_FIN (rescaled by P) -> REPACK_SCATTER (X^j P_j(X^S) into ring N), then REPACK_RING (c0, digits of
/// c1), NTT, REPACK_RMAC (the ring key), INTT, REPACK_C0: `acc` [part][limb][n] mod p0 p1 under s, coefficient form.
/// The packing key switch LWE (s_lwe, q_in) -> GLWE (s_br, q_br) on the device (pc-tfhe `pack::PackingKey::pack`,
/// bit-exact): PACK_DIGITS (lifted digits per placement into D [n_lwe levels][n], the bodies into Bacc), PACK_MAC (the
/// NTT of every D row times its key row, summed), PACK_FIN (inverse NTT, + Bacc) into PBS_INIT's encrypted accumulator
/// layout [2 part][2 prime][n]. The work is n_lwe x levels NTTs and products per prime whatever the count; the digits
/// accumulate across `digits` calls until `finish`.
pub struct DevPack<'a> {
    n: usize,
    nl: usize,
    lv: usize,
    bl: u32,
    m: u64,
    big_q: u64,
    qab: u64,
    ninv: [u64; 4],
    key: DBuf<'a, Compute, u32>,
    tw_f: DBuf<'a, Compute, u64>,
    tw_i: DBuf<'a, Compute, u64>,
    d: DBuf<'a, Compute, i32>,
    acc: DBuf<'a, Compute, u64>,
    /// rows per PACK_MAC block
    rpb: usize,
    /// the packing by rows (PACK_RDIG / RMAC / RFIN): up to `rmax` rows per pass, their digits, sums, bodies
    rmax: usize,
    rdig: Option<DBuf<'a, Compute, i32>>,
    rv: Option<DBuf<'a, Compute, u64>>,
    rb: Option<DBuf<'a, Compute, u64>>,
}

/// Rows per pass of the switch's packing by rows (`TfheSwitch::native_pack_rows`).
const PACK_RMAX: usize = 256;
/// Key rows per PACK_RMAC block.
const RMAC_ROWS: usize = 160;

fn shoup64(w: u64, q: u64) -> u64 {
    (((w as u128) << 64) / q as u128) as u64
}

impl<'a> DevPack<'a> {
    /// Arena bytes (key + tables + work) for a key of `nl` x `lv` rows at degree `n`.
    pub fn bytes(n: usize, nl: usize, lv: usize) -> usize {
        4 * nl * lv * 4 * n + 8 * 8 * n + 4 * (nl * lv * n + 4 * n) + 8 * 4 * n + (1 << 16)
    }

    /// `bytes` with the packing by rows for `rmax` rows per pass.
    pub fn bytes_rows(n: usize, nl: usize, lv: usize, rmax: usize) -> usize {
        Self::bytes(n, nl, lv) + if rmax > 0 { 4 * rmax * nl * lv + 8 * rmax * 4 * n + 8 * rmax + (3 << 8) } else { 0 }
    }

    pub fn new(arena: &'a Arena, s: &Stream<Compute>, pk: &pc_tfhe::pack::PackingKey) -> Result<Self> {
        Self::with_rows(arena, s, pk, 0)
    }

    /// `new` with the packing by rows (`pack_rows`) for up to `rmax` rows per pass.
    pub fn with_rows(arena: &'a Arena, s: &Stream<Compute>, pk: &pc_tfhe::pack::PackingKey, rmax: usize) -> Result<Self> {
        let rns = &pk.rns;
        let n = rns.n();
        let (nl, lv) = (pk.n_lwe(), pk.gadget.levels);
        let mut kw = Vec::with_capacity(nl * lv * 4 * n);
        for i in 0..nl {
            for t in 0..lv {
                for p in 0..2 {
                    let (c0, c1) = pk.row(i, t, p);
                    kw.extend(c0.iter().chain(c1.iter()).map(|&x| x as u32));
                }
            }
        }
        let mut key = arena.alloc::<Compute, u32>("tfhe.pack", kw.len())?;
        key.upload(s, &kw)?;
        let (mut tf, mut ti) = (Vec::with_capacity(4 * n), Vec::with_capacity(4 * n));
        let mut ninv = [0u64; 4];
        for p in 0..2 {
            let (psi, ipsi, ni) = rns.ntt[p].tables();
            let q = rns.ntt[p].q;
            tf.extend_from_slice(psi);
            tf.extend(psi.iter().map(|&w| shoup64(w, q)));
            ti.extend_from_slice(ipsi);
            ti.extend(ipsi.iter().map(|&w| shoup64(w, q)));
            ninv[2 * p] = ni;
            ninv[2 * p + 1] = shoup64(ni, q);
        }
        let mut tw_f = arena.alloc::<Compute, u64>("tfhe.pack", tf.len())?;
        tw_f.upload(s, &tf)?;
        let mut tw_i = arena.alloc::<Compute, u64>("tfhe.pack", ti.len())?;
        tw_i.upload(s, &ti)?;
        let rows = nl * lv;
        Ok(DevPack {
            n,
            nl,
            lv,
            bl: pk.gadget.base_log,
            m: pk.gadget.m,
            big_q: rns.q,
            qab: rns.ntt[0].q | rns.ntt[1].q << 32,
            ninv,
            key,
            tw_f,
            tw_i,
            d: arena.alloc("tfhe.pack", rows * n + 4 * n)?,
            acc: arena.alloc("tfhe.pack", 4 * n)?,
            // ~2 blocks per SM over the two primes
            rpb: rows.div_ceil(160).max(1),
            rmax,
            rdig: if rmax > 0 { Some(arena.alloc("tfhe.pack", rmax * rows)?) } else { None },
            rv: if rmax > 0 { Some(arena.alloc("tfhe.pack", rmax * 4 * n)?) } else { None },
            rb: if rmax > 0 { Some(arena.alloc("tfhe.pack", rmax)?) } else { None },
        })
    }

    /// The NTT of `e` placement polynomials `p` int32 [e][n] into `phat` u64 [e][2 prime][n] (`pack_rows`' basis).
    pub fn basis(&self, table: &KernelTable, s: &Stream<Compute>, p: *const c_void, e: usize, phat: *mut c_void) -> Result<()> {
        let a = TfheArgs { op: tfhe_op::PACK_RBASIS, n: self.n as i32, count: e as i32, x: phat, y: p, tw: self.tw_f.vptr(), c: [0, 0, 0, 0, 0, 0, self.qab, 0], ..Default::default() };
        // SAFETY: the caller's device arrays of the stated lengths, this packing's tables
        launch_result("tfhe", unsafe { (table.tfhe)(&a, s.raw()) })?;
        Ok(())
    }

    /// The packing by rows: `count` accumulators, accumulator a = sum_e LWE(rowmap[a e_n + e]) P_e(X) (P = `p` int32
    /// [e_n][n], `phat` its `basis`) into `out` u32 [count][2 part][2 prime][n]; `lwes` u32 [..][n_lwe + 1] mod
    /// `q_in` under s_lwe, `rowmap` int32 [count e_n]. Bit-exact with `pack` on the placements of P (the key rows'
    /// NTTs replaced by e_n NTT products: few rows, any number of accumulators per pass of `rmax` rows).
    #[allow(clippy::too_many_arguments)]
    pub fn pack_rows(&mut self, table: &KernelTable, s: &Stream<Compute>, lwes: *const c_void, rowmap: *const c_void, count: usize, e_n: usize, p: *const c_void, phat: *const c_void, q_in: u64, out: *mut c_void) -> Result<()> {
        if self.rmax < e_n || e_n == 0 {
            return Err(shape("tfhe pack", format!("pack_rows: {e_n} rows per accumulator, room for {}", self.rmax)));
        }
        let (n, rows) = (self.n, self.nl * self.lv);
        let per = self.rmax / e_n;
        let (rdig, rv, rb) = (self.rdig.as_mut().unwrap(), self.rv.as_mut().unwrap(), self.rb.as_mut().unwrap());
        for a0 in (0..count).step_by(per) {
            let ac = per.min(count - a0);
            let m = ac * e_n;
            let rm = (rowmap as *const i32).wrapping_add(a0 * e_n) as *const c_void;
            let dig = TfheArgs { op: tfhe_op::PACK_RDIG, n: n as i32, count: m as i32, i1: self.nl as i32, x: rdig.vptr_mut(), y: lwes, z: rm, w: rb.vptr(), m: self.m, c: [q_in, self.big_q, self.bl as u64, self.lv as u64, 0, 0, self.qab, 0], ..Default::default() };
            rv.zero_range(s, 0, m * 4 * n)?;
            let mac = TfheArgs { op: tfhe_op::PACK_RMAC, n: n as i32, count: m as i32, i0: RMAC_ROWS as i32, x: rv.vptr_mut(), y: rdig.vptr(), z: self.key.vptr(), c: [0, 0, 0, 0, rows as u64, 0, self.qab, 0], ..Default::default() };
            let o = (out as *mut u32).wrapping_add(a0 * 4 * n) as *mut c_void;
            let fin = TfheArgs { op: tfhe_op::PACK_RFIN, n: n as i32, count: ac as i32, i0: e_n as i32, x: o, y: rv.vptr(), z: phat, w: rb.vptr(), tw: self.tw_i.vptr(), tws: p, c: [0, 0, self.ninv[0], self.ninv[1], self.ninv[2], self.ninv[3], self.qab, 0], ..Default::default() };
            for a in [&dig, &mac, &fin] {
                // SAFETY: this packing's buffers (rmax rows) and the caller's device arrays of the stated lengths
                launch_result("tfhe", unsafe { (table.tfhe)(a, s.raw()) })?;
            }
        }
        Ok(())
    }

    /// Start a packing (zero the digits and sums).
    pub fn begin(&mut self, s: &Stream<Compute>) -> Result<()> {
        self.d.zero(s)?;
        Ok(self.acc.zero(s)?)
    }

    /// Add `count` placements: `place` int32 [count][2] = (row of `lwes`, coefficient < n), `w` int32 [count] weights
    /// (null: 1); `lwes` u32 [rows][n_lwe + 1] mod `q_in` under s_lwe.
    pub fn digits(&mut self, table: &KernelTable, s: &Stream<Compute>, lwes: *const c_void, place: *const c_void, w: *const c_void, count: usize, q_in: u64) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let a = TfheArgs {
            op: tfhe_op::PACK_DIGITS,
            n: self.n as i32,
            count: count as i32,
            i1: self.nl as i32,
            x: self.d.vptr_mut(),
            y: lwes,
            z: place,
            w,
            m: self.m,
            c: [q_in, self.big_q, self.bl as u64, self.lv as u64, 0, 0, self.qab, 0],
            ..Default::default()
        };
        // SAFETY: the pointers address this packing's buffers / the caller's device arrays of the stated lengths
        launch_result("tfhe", unsafe { (table.tfhe)(&a, s.raw()) })?;
        Ok(())
    }

    /// The packed GLWE into `out` u32 [2 part][2 prime][n].
    pub fn finish(&mut self, table: &KernelTable, s: &Stream<Compute>, out: *mut c_void) -> Result<()> {
        let rows = self.nl * self.lv;
        let mac = TfheArgs { op: tfhe_op::PACK_MAC, n: self.n as i32, count: rows as i32, i0: self.rpb as i32, x: self.acc.vptr_mut(), y: self.d.vptr(), z: self.key.vptr(), tw: self.tw_f.vptr(), c: [0, 0, 0, 0, 0, 0, self.qab, 0], ..Default::default() };
        let bacc = (self.d.vptr() as *const i32).wrapping_add(rows * self.n) as *const c_void;
        let fin = TfheArgs { op: tfhe_op::PACK_FIN, n: self.n as i32, count: 1, x: out, y: self.acc.vptr(), z: bacc, tw: self.tw_i.vptr(), c: [0, 0, self.ninv[0], self.ninv[1], self.ninv[2], self.ninv[3], self.qab, 0], ..Default::default() };
        // SAFETY: as `digits`
        launch_result("tfhe", unsafe { (table.tfhe)(&mac, s.raw()) })?;
        // SAFETY: as `digits`
        launch_result("tfhe", unsafe { (table.tfhe)(&fin, s.raw()) })?;
        Ok(())
    }

    /// begin + digits + finish.
    #[allow(clippy::too_many_arguments)]
    pub fn pack(&mut self, table: &KernelTable, s: &Stream<Compute>, lwes: *const c_void, place: *const c_void, w: *const c_void, count: usize, q_in: u64, out: *mut c_void) -> Result<()> {
        self.begin(s)?;
        self.digits(table, s, lwes, place, w, count, q_in)?;
        self.finish(table, s, out)
    }
}

pub struct DevRepack<'a> {
    n: usize,
    nb: usize,
    sfac: usize,
    rp: (u32, usize),
    p0: u64,
    p1: u64,
    aux: u64,
    /// the columns as four balanced int8 planes [plane][4 n_br cols][n_br levels] (K-contiguous: the GEMM's B)
    kpl: DBuf<'a, Compute, u8>,
    ring: DBuf<'a, Compute, u32>,
    /// the digits' two planes [2][rc][n_br levels], one int8 GEMM output [4 n_br][rc] and its workspace
    dpl: DBuf<'a, Compute, u8>,
    gout: DBuf<'a, Compute, i32>,
    ws: DBuf<'a, Compute, u8>,
    gemms: HashMap<usize, I8Gemm<'a>>,
    pk: DBuf<'a, Compute, u32>,
    sums: DBuf<'a, Compute, u64>,
    z: DBuf<'a, Compute, u64>,
    c0: DBuf<'a, Compute, u32>,
    rdig: DBuf<'a, Compute, u32>,
    pub acc: DBuf<'a, Compute, u32>,
    /// LWEs per GEMM pass (`repack_rc`)
    rc: usize,
}

impl<'a> DevRepack<'a> {
    pub fn bytes(n: usize, nb: usize, lv: usize, max: usize) -> usize {
        let rc = repack_rc(max);
        KPL * 4 * nb * nb * lv + 2 * rc * nb * lv + 4 * (lv * 4 * n + 4 * nb * rc + rc * 2 * nb + n + 2 * lv * n + 4 * n) + 8 * (rc * 4 * nb + 2 * n) + WSR + 11 * 256
    }

    pub fn new(arena: &'a Arena, s: &Stream<Compute>, sw: &Switcher, max: usize) -> Result<Self> {
        let (n, nb, lv) = (sw.n, sw.params.n_br, sw.params.rp.1);
        let rc = repack_rc(max);
        let up = |what: &'static str, v: &[u32]| -> Result<DBuf<'a, Compute, u32>> {
            let mut b = arena.alloc::<Compute, u32>(what, v.len())?;
            b.upload(s, v)?;
            Ok(b)
        };
        // B of the int8 GEMMs: column (limb, part, i), K-contiguous over (k, t), centred residues in KPL planes
        let kk = nb * lv;
        let mut w = vec![0i64; 4 * nb * kk];
        let primes = [sw.p0, sw.repack.aux];
        for (k, kt) in sw.repack.cols().iter().enumerate() {
            for (t, limbs) in kt.iter().enumerate() {
                for (li, (a, b)) in limbs.iter().enumerate() {
                    for (part, poly) in [a, b].into_iter().enumerate() {
                        for (i, &x) in poly.iter().enumerate() {
                            let col = (li * 2 + part) * nb + i;
                            w[col * kk + k * lv + t] = pc_tfhe::modq::centered(x, primes[li]);
                        }
                    }
                }
            }
        }
        let planes = pc_core::pcmm_keys::digit_planes(&w, KPL)?;
        drop(w);
        let mut kpl = arena.alloc::<Compute, u8>("tfhe.rcols", planes.len())?;
        let bytes: Vec<u8> = planes.iter().map(|&x| x as u8).collect();
        kpl.upload(s, &bytes)?;
        drop(planes);
        let ring: Vec<u32> = sw.repack.ring().iter().flatten().flat_map(|(a, b)| a.iter().chain(b).map(|&x| x as u32)).collect();
        Ok(DevRepack {
            n,
            nb,
            sfac: sw.params.rs_factor,
            rp: sw.params.rp,
            p0: sw.p0,
            p1: sw.p1,
            aux: sw.repack.aux,
            kpl,
            ring: up("tfhe.rring", &ring)?,
            dpl: arena.alloc("tfhe.work", 2 * rc * nb * lv)?,
            gout: arena.alloc("tfhe.work", 4 * nb * rc)?,
            ws: arena.alloc("tfhe.ws", WSR)?,
            gemms: HashMap::new(),
            pk: arena.alloc("tfhe.work", rc * 2 * nb)?,
            sums: arena.alloc("tfhe.work", rc * 4 * nb)?,
            rc,
            z: arena.alloc("tfhe.work", 2 * n)?,
            c0: arena.alloc("tfhe.work", n)?,
            rdig: arena.alloc("tfhe.work", 2 * lv * n)?,
            acc: arena.alloc("tfhe.work", 4 * n)?,
        })
    }

    fn args(&self) -> TfheArgs {
        let (p0, aux) = (self.p0, self.aux);
        TfheArgs {
            n: self.n as i32,
            i1: self.nb as i32,
            i2: self.sfac as i32,
            q: p0,
            c: [self.rp.0 as u64, self.rp.1 as u64, aux, invmod(aux % p0, p0), aux % p0, 0, 0, mu(p0)],
            ..Default::default()
        }
    }

    /// Prepare the int8 GEMM plans for passes of `b` values (setup: the replay builds none).
    pub fn prepare(&mut self, lt_: &'a lt::Lt, b: usize) -> Result<()> {
        let (nb, lv) = (self.nb, self.rp.1);
        let rc = self.rc;
        for m in [b.min(rc), b % rc] {
            let mp = m.next_multiple_of(16);
            if m > 0 && !self.gemms.contains_key(&mp) {
                self.gemms.insert(mp, I8Gemm::new(lt_, mp as u64, (4 * nb) as u64, (nb * lv) as u64, 8, WSR)?);
            }
        }
        Ok(())
    }

    /// Add the `b` LWEs `outs` ([b][n_br + 1] mod p0) at the coefficients `idx` ([b] u32) into the packed sum
    /// (`first`: start a new one); `reps` > 1: each also at `idx + r stride` (r < reps: copies of the value at other
    /// coefficients, one packing each, `reps` scatters). The GEMM digits x columns runs on int8 tensor cores: 2 x KPL
    /// plane products grouped by their power of 256 (int32 sums < 2^27), each group folded into the exact int64 sums.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate(&mut self, table: &KernelTable, s: &Stream<Compute>, lt_: &'a lt::Lt, outs: *const c_void, idx: *const c_void, b: usize, first: bool, reps: usize, stride: usize) -> Result<()> {
        let (nb, kk) = (self.nb, self.nb * self.rp.1);
        self.prepare(lt_, b)?;
        let go = |a: &TfheArgs| -> Result<()> {
            // SAFETY: the pointers address this repack's buffers / the caller's LWEs and positions, sized for the op
            launch_result("tfhe", unsafe { (table.tfhe)(a, s.raw()) })?;
            Ok(())
        };
        let rb = self.args();
        let rcp = self.rc;
        for r0 in (0..b).step_by(rcp) {
            let rc = rcp.min(b - r0);
            let o = (outs as *const u32).wrapping_add(r0 * (nb + 1)) as *const c_void;
            let ix = (idx as *const u32).wrapping_add(r0) as *const c_void;
            let sums = self.sums.vptr_mut();
            let mp = rc.next_multiple_of(16);
            let mut rd = rb;
            rd.c[5] = mp as u64;
            go(&TfheArgs { op: tfhe_op::REPACK_DIGITS, count: rc as i32, i0: (first && r0 == 0) as i32, x: self.dpl.vptr_mut(), y: o, z: self.z.vptr_mut() as *const c_void, w: sums as *const c_void, ..rd })?;
            let gemm = &self.gemms[&mp];
            for sh in 0..(KPL + 1) {
                let mut acc = false;
                for a in 0..2usize {
                    if sh < a || sh - a >= KPL {
                        continue;
                    }
                    let bk = sh - a;
                    // SAFETY: the planes and the output are this repack's buffers of the plan's lengths, on `s`
                    unsafe {
                        gemm.run(s, (self.dpl.dptr() as *const i8).add(a * mp * kk), (self.kpl.dptr() as *const i8).add(bk * 4 * nb * kk), self.gout.dptr_mut(), acc, &mut self.ws)?;
                    }
                    acc = true;
                }
                let mut rcmb = rb;
                rcmb.c[5] = sh as u64;
                go(&TfheArgs { op: tfhe_op::REPACK_COMB, count: rc as i32, i3: mp as i32, x: sums, y: self.gout.vptr(), ..rcmb })?;
            }
            go(&TfheArgs { op: tfhe_op::REPACK_FIN, count: rc as i32, x: self.pk.vptr_mut(), y: self.sums.vptr(), w: o, ..rb })?;
            for r in 0..reps.max(1) {
                let ixr = (ix as *const u32).wrapping_add(r * stride) as *const c_void;
                go(&TfheArgs { op: tfhe_op::REPACK_SCATTER, count: rc as i32, y: self.pk.vptr(), z: self.z.vptr_mut() as *const c_void, w: ixr, ..rb })?;
            }
        }
        Ok(())
    }

    /// The packed sum to `acc` [part][limb][n] mod p0 p1 under s (the ring key switch). `pa`: the ring-N NTT args
    /// (primes p0, p1 with their n^-1), `tw_f` / `tw_i` their twiddles.
    pub fn finish(&mut self, table: &KernelTable, s: &Stream<Compute>, pa: &PcmmArgs, tw_f: *const c_void, tw_i: *const c_void) -> Result<()> {
        let (n, lv, p0, p1) = (self.n, self.rp.1, self.p0, self.p1);
        let go = |a: &TfheArgs| -> Result<()> {
            // SAFETY: the pointers address this repack's buffers
            launch_result("tfhe", unsafe { (table.tfhe)(a, s.raw()) })?;
            Ok(())
        };
        let gp = |a: &PcmmArgs| -> Result<()> {
            // SAFETY: as `go` (the PCMM launcher's NTT ops on this repack's buffers)
            launch_result("pcmm", unsafe { (table.pcmm)(a, s.raw()) })?;
            Ok(())
        };
        let rb = self.args();
        let rr = TfheArgs { count: 1, c: [self.rp.0 as u64, lv as u64, p1 % p0, 0, p1, 0, mu(p1), mu(p0)], ..rb };
        go(&TfheArgs { op: tfhe_op::REPACK_RING, x: self.rdig.vptr_mut(), y: self.z.vptr(), w: self.c0.vptr_mut() as *const c_void, ..rr })?;
        gp(&PcmmArgs { op: pcmm_op::NTT, count: (2 * lv) as i32, i0: lv as i32, x: self.rdig.vptr_mut(), tw: tw_f, ..*pa })?;
        self.acc.zero(s)?;
        go(&TfheArgs { op: tfhe_op::REPACK_RMAC, x: self.acc.vptr_mut(), y: self.rdig.vptr(), z: self.ring.vptr(), ..rr })?;
        for part in 0..2 {
            let xp = (self.acc.vptr_mut() as *mut u32).wrapping_add(part * 2 * n) as *mut c_void;
            gp(&PcmmArgs { op: pcmm_op::INTT, count: 2, i0: 1, x: xp, tw: tw_i, ..*pa })?;
        }
        go(&TfheArgs { op: tfhe_op::REPACK_C0, x: self.acc.vptr_mut(), y: self.c0.vptr(), ..rr })?;
        Ok(())
    }

    /// `accumulate` then `finish`: the `b` LWEs `outs` at the coefficients `idx`.
    #[allow(clippy::too_many_arguments)]
    pub fn run(&mut self, table: &KernelTable, s: &Stream<Compute>, lt_: &'a lt::Lt, outs: *const c_void, idx: *const c_void, b: usize, pa: &PcmmArgs, tw_f: *const c_void, tw_i: *const c_void) -> Result<()> {
        self.accumulate(table, s, lt_, outs, idx, b, true, 1, 0)?;
        self.finish(table, s, pa, tw_f, tw_i)
    }
}

/// The device forms of an LWE key-switching key: for a key with seeded masks its seed and bodies (KS_REGEN
/// regenerates the masks in the kernel: 4 K bytes instead of 4 K (n_lwe + 1)), and / or the stored four balanced
/// base-256 int8 planes of the K-contiguous columns W[o][i levels + t] (the GEMMs' B; a seeded key's masks expanded
/// on the host). The key switch of `rows` LWEs takes the regeneration up to `ks_regen_max()` rows when it has it.
pub struct DevKsk<'a> {
    pub seeded: Option<(pc_core::keys::MaskSeed, DBuf<'a, Compute, u32>)>,
    pub planes: Option<Vec<DBuf<'a, Compute, u8>>>,
}

impl<'a> DevKsk<'a> {
    /// `regen`: a seeded key keeps its seed (K a multiple of 64); `planes`: the stored planes (always for a key that
    /// cannot regenerate).
    pub fn new(arena: &'a Arena, s: &Stream<Compute>, key: &pc_tfhe::lwe::LweKsk, p0: u64, regen: bool, planes: bool) -> Result<Self> {
        let k = key.rows();
        let seeded = match (regen && k % 64 == 0, key.seed()) {
            (true, Some(seed)) => {
                let b: Vec<u32> = key.b.iter().map(|&v| v as u32).collect();
                let mut body = arena.alloc::<Compute, u32>("tfhe.ksk", k)?;
                body.upload(s, &b)?;
                Some((*seed, body))
            }
            _ => None,
        };
        let planes = if planes || seeded.is_none() {
            let cols = key.n_out + 1;
            let m = key.masks();
            let mut w = vec![0i64; cols * k];
            for o in 0..cols {
                for r in 0..k {
                    let v = if o < key.n_out { m[o * k + r] as u64 } else { key.b[r] };
                    w[o * k + r] = pc_tfhe::modq::centered(v, p0);
                }
            }
            let planes = pc_core::pcmm_keys::digit_planes(&w, PLANES)?;
            drop(w);
            let mut out = Vec::with_capacity(PLANES);
            for u in 0..PLANES {
                let mut b = arena.alloc::<Compute, u8>("tfhe.ksk", cols * k)?;
                let bytes: Vec<u8> = planes[u * cols * k..(u + 1) * cols * k].iter().map(|&x| x as u8).collect();
                b.upload(s, &bytes)?;
                out.push(b);
            }
            Some(out)
        } else {
            None
        };
        Ok(DevKsk { seeded, planes })
    }

    /// Device bytes of the key.
    pub fn bytes(&self) -> usize {
        self.planes.iter().flatten().map(|b| b.len()).sum::<usize>() + self.seeded.as_ref().map_or(0, |(_, b)| 4 * b.len())
    }
}

/// Keys with seeded masks run the key switch's in-kernel regeneration (KS_REGEN) unless PC_TFHE_KS_REGEN=0 (then the
/// planes, expanded on the host).
pub fn ks_regen(p: &pc_tfhe::params::Params) -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    p.seeded && *V.get_or_init(|| std::env::var("PC_TFHE_KS_REGEN").map(|v| v.trim() != "0").unwrap_or(true))
}

/// Batches up to this many LWEs take the regeneration (KS_REGEN_MAX; PC_TFHE_KS_REGEN_MAX overrides), larger ones
/// the stored planes' GEMMs. Per round (pc-check tfhe-prog, 4 chained rounds, median of 3): 1 / 8 / 16 / 40 rows
/// 1.115 / 1.198 / 1.495 / 1.88 ms against 1.153 / 1.245 / 1.534 / 1.93 on the planes; 46 rows 2.06 vs 2.04, 64
/// 2.35 vs 2.31, 188 5.17 vs 4.65, 1024 22.4 vs 19.4 (the expansion repeats per 64 rows, and the GEMMs' tensor
/// throughput is the regeneration's best case).
pub fn ks_regen_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("PC_TFHE_KS_REGEN_MAX").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(KS_REGEN_MAX))
}
pub const KS_REGEN_MAX: usize = 40;

/// Whether a switch of `max` values per pass keeps the planes on the device (some batch exceeds `ks_regen_max()`, or
/// no regeneration).
pub fn ks_planes(p: &pc_tfhe::params::Params, max: usize) -> bool {
    !ks_regen(p) || max > ks_regen_max()
}

/// KS_REGEN's grid for `rows` LWEs (k slices, row tiles per warp): the blocks (64 columns x slices x row groups)
/// fill one wave of two per SM (`sms`); the slices overridable by PC_TFHE_KS_SLICES, the row tiles by
/// PC_TFHE_KS_MT.
pub fn ks_regen_shape(rows: usize, cols: usize, k: usize, sms: usize) -> (usize, usize) {
    let env = |n: &str| std::env::var(n).ok().and_then(|v| v.trim().parse::<usize>().ok());
    let mt = env("PC_TFHE_KS_MT").filter(|m| [1, 2, 4].contains(m)).unwrap_or(if rows <= 16 { 1 } else if rows <= 32 { 2 } else { 4 });
    let blocks = cols.div_ceil(64) * rows.div_ceil(16 * mt);
    let slices = env("PC_TFHE_KS_SLICES").unwrap_or(2 * sms / blocks).clamp(1, k / 64);
    (slices, mt)
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
    ksk: DevKsk<'a>,
    /// Params::chain: the native chains' key switch
    cksk: Option<DevKsk<'a>>,
    /// native_load: the bootstrap outputs (one buffer per LUT, [b][n_br + 1]) go here instead of the repack
    sinks: Vec<*mut c_void>,
    /// `native_extract`: the key-switched LWEs (s_lwe) into here instead of a bootstrap (null: off)
    extract_dst: *mut c_void,
    /// Params::pack: the packing key switch (encrypted lookup tables: `native_pack`, `native_plut`)
    pack: Option<DevPack<'a>>,
    bsk: DBuf<'a, Compute, u32>,
    bsk_lat: Option<DBuf<'a, Compute, u32>>,
    bsk_lat2: Option<DBuf<'a, Compute, u32>>,
    btw: DBuf<'a, Compute, u32>,
    btw_s: DBuf<'a, Compute, u32>,
    repack: DevRepack<'a>,
    stage: DBuf<'a, Compute, u32>,
    low: DBuf<'a, Compute, u32>,
    dig: DBuf<'a, Compute, u32>,
    rs: DBuf<'a, Compute, u32>,
    idx: DBuf<'a, Compute, u32>,
    ext: DBuf<'a, Compute, u32>,
    kdig: DBuf<'a, Compute, u8>,
    g: DBuf<'a, Compute, i32>,
    lwe: DBuf<'a, Compute, u32>,
    ms: DBuf<'a, Compute, u32>,
    acc: DBuf<'a, Compute, u32>,
    kb: DBuf<'a, Compute, u32>,
    outs: DBuf<'a, Compute, u32>,
    tv: DBuf<'a, Compute, u32>,
    ws: DBuf<'a, Compute, u8>,
    gemms: HashMap<usize, I8Gemm<'a>>,
    /// the latency path's key bundles run on `side`, a window ahead of the rotation (two halves of `kb`), the rotation
    /// on `rot`, of the highest priority (its blocks are not queued behind the next window's bundles)
    side: Stream<Compute>,
    rot: Stream<Compute>,
    /// LWEs per batch-path wave (two per block, one block per SM)
    wave: usize,
    /// thread-block clusters on the device (sm_90+): without them the latency path is one level, nu 1, groups <= 4
    clusters: bool,
    ev: [Event; 6],
}

impl<'a> TfheSwitch<'a> {
    /// Arena bytes for `max` switched coefficients per call (keys + work) on a ring of degree `n`.
    pub fn bytes(n: usize, p: &pc_tfhe::params::Params, max: usize) -> usize {
        let (nb, nl) = (p.n_br, p.n_lwe);
        let n_sub = n / p.rs_factor;
        let k = n_sub * p.ks.1;
        let words32 = 2 * 3 * n + p.rs.1 * 2 * n + 8 * n + 2 * n + p.rs.1 * n + 2 * n + max + n
            + max * (n_sub + 1) + PLANES * ((nl + 1) * (max + 16) + 64) + 2 * max * (nl + 1) + max * (nb + 1)
            + (nl / p.pbs_group * ((1 << p.pbs_group) - 1)) * 2 * p.pbs.1 * 4 * nb + 14 * nb + max * 4 * nb * p.nu + 2 * nb * p.nu
            + if p.lat_group > 0 { nl / p.lat_group * ((1 << p.lat_group) - 1) * 2 * p.pbs.1 * 4 * nb } else { 0 }
            + if p.lat_group2 > 0 { nl / p.lat_group2 * ((1 << p.lat_group2) - 1) * 2 * p.pbs.1 * 4 * nb } else { 0 };
        let nkeys = if p.chain { 2 } else { 1 };
        let key_bytes = if ks_regen(p) && k % 64 == 0 { 4 * k } else { 0 } + if ks_planes(p, max) { PLANES * (nl + 1) * k } else { 0 };
        let bytes8 = key_bytes * nkeys + (max + 16) * k + WS;
        let pack = if p.pack { DevPack::bytes_rows(nb, nl, pc_tfhe::pack::PACK_GADGET.1, PACK_RMAX) } else { 0 };
        4 * (words32 + max.min(KB_MAX) * KB_WINDOW * 8 * nb * p.nu * p.pbs.1) + bytes8 + DevRepack::bytes(n, nb, p.rp.1, max) + pack + (1 << 20)
    }

    /// Upload `sw`'s keys (host-generated: pc-tfhe `Switcher::generate`) and carve the work buffers.
    pub fn new(arena: &'a Arena, dev: &Device, s: &Stream<Compute>, sw: &Switcher, max: usize) -> Result<Self> {
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
        let (k, cols) = (n_sub * p.ks.1, nl + 1);
        let ksk = DevKsk::new(arena, s, &sw.ksk, sw.p0, ks_regen(&p), ks_planes(&p, max))?;
        // the native chains' key switch s_br -> s_lwe: the same shapes (n_br = N / S inputs)
        let cksk = match &sw.chain {
            Some(c) if nb == n_sub => Some(DevKsk::new(arena, s, &c.ksk, sw.p0, ks_regen(&p), ks_planes(&p, max))?),
            Some(_) => return Err(shape("tfhe switch", format!("native chains need n_br = N / S ({nb} vs {n_sub})"))),
            None => None,
        };
        let img = pbs_image(&sw.bsk);
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
            cksk,
            sinks: Vec::new(),
            extract_dst: std::ptr::null_mut(),
            pack: match &sw.pack {
                Some(pk) => Some(DevPack::with_rows(arena, s, pk, PACK_RMAX)?),
                None => None,
            },
            bsk: up32("tfhe.bsk", &img.bsk)?,
            bsk_lat: match &sw.bsk_lat {
                Some(k) => Some(up32("tfhe.bsk", &pbs_image(k).bsk)?),
                None => None,
            },
            bsk_lat2: match &sw.bsk_lat2 {
                Some(k) => Some(up32("tfhe.bsk", &pbs_image(k).bsk)?),
                None => None,
            },
            btw: up32("tfhe.btw", &img.tw)?,
            btw_s: up32("tfhe.btw", &img.tws)?,
            repack: DevRepack::new(arena, s, sw, max)?,
            stage: arena.alloc("tfhe.work", 8 * n)?,
            low: arena.alloc("tfhe.work", 2 * 2 * n)?,
            dig: arena.alloc("tfhe.work", p.rs.1 * n)?,
            rs: arena.alloc("tfhe.work", 2 * n)?,
            idx: arena.alloc("tfhe.work", max)?,
            ext: arena.alloc("tfhe.work", max * (n_sub + 1))?,
            kdig: arena.alloc("tfhe.work", max.next_multiple_of(16) * k)?,
            g: arena.alloc("tfhe.work", PLANES * (cols * max.next_multiple_of(16)).next_multiple_of(64))?,
            lwe: arena.alloc("tfhe.work", max * cols)?,
            ms: arena.alloc("tfhe.work", max * cols)?,
            acc: arena.alloc("tfhe.work", max * 4 * nb * p.nu)?,
            kb: arena.alloc("tfhe.kb", max.min(KB_MAX) * KB_WINDOW * 8 * nb * p.nu * p.pbs.1)?,
            outs: arena.alloc("tfhe.work", max * (nb + 1))?,
            tv: arena.alloc("tfhe.work", 2 * nb * p.nu)?,
            ws: arena.alloc("tfhe.ws", WS)?,
            gemms: HashMap::new(),
            side: Stream::new(dev)?,
            rot: Stream::high_priority(dev)?,
            wave: 2 * dev.sm_count()?,
            clusters: dev.clusters()?,
            ev: [Event::new(dev)?, Event::new(dev)?, Event::new(dev)?, Event::new(dev)?, Event::new(dev)?, Event::new(dev)?],
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
    /// key switch's GEMMs (a plan per batch size, built on first use). Uploads its arguments and syncs per stage
    /// (the timings): the host API of the checks; the replay calls `apply_dev`.
    pub fn apply<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, lt_: &'a lt::Lt, x: &mut Ct<'_, W>, positions: &[u32], tv: &[u32], offset: u64) -> Result<Timings> {
        if positions.is_empty() || positions.len() > self.max || tv.len() != 2 * self.p.tv_len() {
            return Err(shape("tfhe switch", format!("{} coefficients (the switch holds 1..={}) / test vector of {}", positions.len(), self.max, tv.len())));
        }
        self.idx.upload_at(s, 0, positions)?;
        self.tv.upload(s, tv)?;
        let (pos, tvp) = (self.idx.vptr(), self.tv.vptr());
        self.apply_dev(ctx, s, lt_, x, pos, positions.len(), 1, tvp, offset, true)
    }

    /// Values per pass (`new`'s `max`).
    pub fn max(&self) -> usize {
        self.max
    }

    /// Prepare the key switch's GEMM plan for chunks of `b` values (setup: the replay builds none). The GEMM runs on
    /// `b` rounded up to 16 rows (cuBLASLt's int8 kernels want aligned leading dimensions; the extra rows are unread).
    pub fn prepare(&mut self, lt_: &'a lt::Lt, b: usize) -> Result<()> {
        self.repack.prepare(lt_, b.min(self.max))?;
        let b = b.min(self.max).next_multiple_of(16);
        if b > 0 && !self.gemms.contains_key(&b) {
            let k = self.n_sub * self.p.ks.1;
            self.gemms.insert(b, I8Gemm::new(lt_, b as u64, (self.nl + 1) as u64, k as u64, 8, WS)?);
        }
        Ok(())
    }

    /// `apply` on device arguments: `pos` [reps][b] u32 coefficients (any b: chunks of the switch's capacity, the
    /// repack accumulating across them; row 0 is read, rows 1.. are where the outputs are copied too: the repack
    /// scatters each packed value `reps` times), `tv` [2 prime][n_br nu] residues. `timed`: sync per stage for the
    /// timings (else no host sync: the replay).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_dev<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, lt_: &'a lt::Lt, x: &mut Ct<'_, W>, pos: *const c_void, b: usize, reps: usize, tv: *const c_void, offset: u64, timed: bool) -> Result<Timings> {
        self.apply_dev_k(ctx, s, lt_, x, &mut [], pos, b, reps, tv, &[offset], timed)
    }

    /// `apply_dev` of k = `offsets.len()` interleaved LUTs (pc-tfhe `bootstrap_mlut`) from one blind rotation: LUT 0
    /// into `x`, LUT j into `extra[j - 1]`; k > 1 takes one pass (b <= the switch's capacity).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_dev_k<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, lt_: &'a lt::Lt, x: &mut Ct<'_, W>, extra: &mut [&mut Ct<'_, W>], pos: *const c_void, b: usize, reps: usize, tv: *const c_void, offsets: &[u64], timed: bool) -> Result<Timings> {
        let (n, nb, nl, n_sub) = (self.n, self.nb, self.nl, self.n_sub);
        let kl = offsets.len();
        let outs_ok = if self.sinks.is_empty() { extra.len() + 1 == kl } else { self.sinks.len() == kl };
        if b == 0 || kl == 0 || !outs_ok || (kl > 1 && b > self.max) || (2 * nb * self.p.nu) % kl != 0 {
            return Err(shape("tfhe switch", format!("{b} coefficients, {kl} LUTs ({} outputs; several LUTs: one pass of <= {})", extra.len() + 1, self.max)));
        }
        let top = crate::bts::stc_first::split_top(&ctx.chain);
        if ctx.lay != x.buf.lay || ctx.top_of(x.meta.level)? != top {
            return Err(shape("tfhe switch", format!("ciphertext at level {} (the split holds limbs 0..{top})", x.meta.level)));
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
        let chain = |opc: i32| PcmmArgs { op: opc, n: n as i32, p: c4, mu: cmu, i4: top as i32 + 1, ..PcmmArgs::default() };
        let pa = PcmmArgs { n: n as i32, p: [p0 as u32, p1 as u32, p0 as u32, 0], mu: [mu(p0), mu(p1), mu(p0), 0], c: [self.ninv[0], self.ninv[1], self.ninv[0], 0, 0, 0, 0, 0], ..Default::default() };
        let mut tm = Timings::default();
        let lap = |s: &Stream<Compute>, t: &mut Instant, acc: &mut f64| -> Result<()> {
            if timed {
                s.sync()?;
                *acc += t.elapsed().as_secs_f64() * 1e3;
                *t = Instant::now();
            }
            Ok(())
        };
        let mut t = Instant::now();

        // the split to LWEs: coefficient form, mod q0 (staging part 0 = c1, part 1 = c0)
        ntt::inverse(ctx, s, &mut x.buf, Span::Q { top }, Parts::Both)?;
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
        lap(s, &mut t, &mut tm.to_lwe)?;

        let cols = nl + 1;
        let qb = self.p.q_br;
        let nu = self.p.nu as u64;
        let mut cx = [0u64; 8];
        cx[5] = nu;
        cx[6] = self.pbs_c[6];
        cx[7] = self.pbs_c[7];
        let mut pbs_c = self.pbs_c;
        pbs_c[5] = nu;
        for c0 in (0..b).step_by(self.max) {
            let cb = self.max.min(b - c0);
            let pc = (pos as *const u32).wrapping_add(c0) as *const c_void;
            Self::go(ctx, s, &TfheArgs { op: tfhe_op::RS_EXTRACT, count: cb as i32, i1: n_sub as i32, i2: self.sfac as i32, x: self.ext.vptr_mut(), y: self.stage.vptr(), z: pc, ..tb })?;
            lap(s, &mut t, &mut tm.to_lwe)?;

            // LWE key switch
            let ext = self.ext.vptr();
            self.key_switch(ctx, s, lt_, ext, n_sub, cb, false)?;
            lap(s, &mut t, &mut tm.key_switch)?;
            if !self.extract_dst.is_null() {
                let d = (self.extract_dst as *mut u32).wrapping_add(c0 * cols) as *mut c_void;
                // SAFETY: `native_extract`'s caller passes b rows of n_lwe + 1 words on this device
                unsafe { self.lwe.copy_to_raw(s, 0, d, cb * cols)? };
                continue;
            }

            // bootstrap
            let pb = TfheArgs { n: nb as i32, count: cb as i32, i1: nl as i32, q: qb, ..Default::default() };
            self.rotate(ctx, s, cb, tv, kl, &cx, pbs_c)?;
            if !self.sinks.is_empty() {
                for (j, &dst) in self.sinks.iter().enumerate() {
                    let x = (dst as *mut u32).wrapping_add(c0 * (nb + 1)) as *mut c_void;
                    Self::go(ctx, s, &TfheArgs { op: tfhe_op::PBS_EXTRACT, x, y: self.acc.vptr(), m: pc_tfhe::gadget::fixed_ratio(p0, qb), c: [offsets[j], p0, 0, 0, j as u64, nu, cx[6], cx[7]], ..pb })?;
                }
                lap(s, &mut t, &mut tm.bootstrap)?;
                continue;
            }
            if kl == 1 {
                Self::go(ctx, s, &TfheArgs { op: tfhe_op::PBS_EXTRACT, x: self.outs.vptr_mut(), y: self.acc.vptr(), m: pc_tfhe::gadget::fixed_ratio(p0, qb), c: [offsets[0], p0, 0, 0, 0, nu, cx[6], cx[7]], ..pb })?;
                lap(s, &mut t, &mut tm.bootstrap)?;
                self.repack.accumulate(ctx.table, s, lt_, self.outs.vptr(), pc, cb, c0 == 0, reps, b)?;
                lap(s, &mut t, &mut tm.repack)?;
                continue;
            }
            lap(s, &mut t, &mut tm.bootstrap)?;
            // several LUTs (one pass): LUT j from coefficient j, repacked into its own ciphertext; LUT 0 last (into x)
            for j in (0..kl).rev() {
                Self::go(ctx, s, &TfheArgs { op: tfhe_op::PBS_EXTRACT, x: self.outs.vptr_mut(), y: self.acc.vptr(), m: pc_tfhe::gadget::fixed_ratio(p0, qb), c: [offsets[j], p0, 0, 0, j as u64, nu, cx[6], cx[7]], ..pb })?;
                self.repack.accumulate(ctx.table, s, lt_, self.outs.vptr(), pc, cb, true, reps, b)?;
                if j > 0 {
                    let meta = x.meta;
                    let out = &mut *extra[j - 1];
                    self.finish_into(ctx, s, &pa, out)?;
                    out.meta = meta;
                }
            }
            lap(s, &mut t, &mut tm.repack)?;
        }
        if self.sinks.is_empty() && self.extract_dst.is_null() {
            self.finish_into(ctx, s, &pa, x)?;
        }
        lap(s, &mut t, &mut tm.repack)?;
        Ok(tm)
    }

    // ── native chains (Params::chain): LWE buffers [b][n_br + 1] mod p0 under s_br, native encoding (v at phase
    // v p0 / 4); one split in (`native_load`), one finish out (`native_store`) ──────────────────────────────────────

    /// The split's coefficients `pos` (as `apply_dev`) through the LUT `tv` + `offset` into `dst` [b][n_br + 1].
    #[allow(clippy::too_many_arguments)]
    pub fn native_load<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, lt_: &'a lt::Lt, x: &mut Ct<'_, W>, pos: *const c_void, b: usize, tv: *const c_void, offset: u64, dst: *mut c_void, timed: bool) -> Result<Timings> {
        self.native_load_k(ctx, s, lt_, x, pos, b, tv, &[offset], &[dst], timed)
    }

    /// `native_load` through k interleaved LUTs (one rotation; b <= one pass): LUT j into `dsts[j]`.
    #[allow(clippy::too_many_arguments)]
    pub fn native_load_k<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, lt_: &'a lt::Lt, x: &mut Ct<'_, W>, pos: *const c_void, b: usize, tv: *const c_void, offsets: &[u64], dsts: &[*mut c_void], timed: bool) -> Result<Timings> {
        self.sinks = dsts.to_vec();
        let r = self.apply_dev_k(ctx, s, lt_, x, &mut [], pos, b, 1, tv, offsets, timed);
        self.sinks.clear();
        r
    }

    /// The split's coefficients `pos` [b] of `x` to LWEs under s_lwe mod p0 (ring switch, extraction, key switch: a
    /// bootstrap's input, what `native_pbs` reaches after its own key switch) into `dst` [b][n_lwe + 1]; no bootstrap.
    #[allow(clippy::too_many_arguments)]
    pub fn native_extract<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, lt_: &'a lt::Lt, x: &mut Ct<'_, W>, pos: *const c_void, b: usize, dst: *mut c_void, timed: bool) -> Result<Timings> {
        self.extract_dst = dst;
        let r = self.apply_dev_k(ctx, s, lt_, x, &mut [], pos, b, 1, std::ptr::null(), &[0], timed);
        self.extract_dst = std::ptr::null_mut();
        r
    }

    /// `dst[i] = ca src_a[ia[i]] + cb src_b[ib[i]]` mod p0 (`idx`: int32 pairs [b][2] on the device, -1: no term;
    /// ca, cb < p0): the chains' differences and running sums.
    #[allow(clippy::too_many_arguments)]
    pub fn native_lin<M: DevStrategy<W>>(&self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, dst: *mut c_void, src_a: *const c_void, src_b: *const c_void, idx: *const c_void, b: usize, ca: u64, cb: u64) -> Result<()> {
        self.native_lin_c(ctx, s, dst, src_a, src_b, idx, b, ca, cb, 0)
    }

    /// `native_lin` + the constant `c` (< p0) on every output's b (a native value v is round(v p0 / 4)). Launches of
    /// at most 65535 rows (the kernel's grid y): a longer lin runs in chunks (outputs and index pairs advanced).
    #[allow(clippy::too_many_arguments)]
    pub fn native_lin_c<M: DevStrategy<W>>(&self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, dst: *mut c_void, src_a: *const c_void, src_b: *const c_void, idx: *const c_void, b: usize, ca: u64, cb: u64, c: u64) -> Result<()> {
        const MAX_ROWS: usize = 65535;
        let row = self.nb + 1;
        let mut r0 = 0;
        while r0 < b {
            let cnt = (b - r0).min(MAX_ROWS);
            let x = (dst as *mut u32).wrapping_add(r0 * row) as *mut c_void;
            let w = (idx as *const i32).wrapping_add(2 * r0) as *const c_void;
            Self::go(ctx, s, &TfheArgs { op: tfhe_op::LWE_LIN, count: cnt as i32, i1: self.nb as i32, x, y: src_a, z: src_b, w, q: self.p0, c: [ca, cb, c, 0, 0, 0, 0, 0], ..Default::default() })?;
            r0 += cnt;
        }
        Ok(())
    }

    /// p0 (the native LWEs' modulus).
    pub fn p0(&self) -> u64 {
        self.p0
    }

    /// n_br (native LWE dimension).
    pub fn n_br(&self) -> usize {
        self.nb
    }

    /// The bootstrap of native LWEs `src` [b][n_br + 1] (key switch s_br -> s_lwe, then as `apply_dev_k`): LUT j of
    /// the `offsets.len()` interleaved ones into `dsts[j]` [b][n_br + 1].
    #[allow(clippy::too_many_arguments)]
    pub fn native_pbs<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, lt_: &'a lt::Lt, src: *const c_void, b: usize, tv: *const c_void, offsets: &[u64], dsts: &[*mut c_void]) -> Result<()> {
        self.native_rot(ctx, s, lt_, src, b, tv, 0, offsets, dsts)
    }

    /// The private lookup: native index LWEs `src` [b][n_br + 1] rotate the encrypted accumulator `acc` ([2 part][2
    /// prime][n_br], `native_pack`; pc-tfhe `BootstrapKey::bootstrap_acc`), the extraction of coefficient j (+ offsets[j],
    /// mod switched to p0: native LWEs under s_br, what `native_pbs` reads) into `dsts[j]`. nu 1.
    #[allow(clippy::too_many_arguments)]
    pub fn native_plut<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, lt_: &'a lt::Lt, src: *const c_void, b: usize, acc: *const c_void, offsets: &[u64], dsts: &[*mut c_void]) -> Result<()> {
        if self.p.nu != 1 {
            return Err(shape("tfhe native", "the private lookup runs at nu 1".to_string()));
        }
        self.native_rot(ctx, s, lt_, src, b, acc, 1, offsets, dsts)
    }

    /// `native_plut` with one accumulator per index LWE: `accs` [b][2 part][2 prime][n_br] (LWE r rotates accs[r]),
    /// the step of a selection tree (several tables rotated by one index digit in one batch). nu 1.
    #[allow(clippy::too_many_arguments)]
    pub fn native_plut_each<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, lt_: &'a lt::Lt, src: *const c_void, b: usize, accs: *const c_void, offsets: &[u64], dsts: &[*mut c_void]) -> Result<()> {
        if self.p.nu != 1 {
            return Err(shape("tfhe native", "the private lookup runs at nu 1".to_string()));
        }
        self.native_rot(ctx, s, lt_, src, b, accs, 3, offsets, dsts)
    }

    /// The key switch s_br -> s_lwe of `cb` native LWEs `src` [cb][n_br + 1] (cb <= the switch's capacity) into `lwe`.
    fn native_ks<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, lt_: &'a lt::Lt, src: *const c_void, cb: usize) -> Result<()> {
        self.key_switch(ctx, s, lt_, src, self.nb, cb, true)
    }

    /// The LWE key switch of `cb` LWEs `src` [cb][n_in + 1] into `lwe` [cb][n_lwe + 1]: KS_DIGITS, then the products
    /// against the key (four int8 GEMMs on the stored planes, or KS_REGEN on a key with seeded masks), KS_RECOMBINE.
    /// `chain`: the native chains' key (s_br -> s_lwe), else the switch's (s' -> s_lwe).
    #[allow(clippy::too_many_arguments)]
    fn key_switch<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, lt_: &'a lt::Lt, src: *const c_void, n_in: usize, cb: usize, chain: bool) -> Result<()> {
        let (nl, p0) = (self.nl, self.p0);
        let (ks_lv, cols) = (self.p.ks.1, nl + 1);
        Self::go(ctx, s, &TfheArgs { op: tfhe_op::KS_DIGITS, count: cb as i32, i1: n_in as i32, x: self.kdig.vptr_mut(), y: src, q: p0, m: self.ks_m, c: [self.p.ks.0 as u64, ks_lv as u64, 0, 0, 0, 0, 0, 0], ..Default::default() })?;
        let mp = cb.next_multiple_of(16);
        // each GEMM's output plane starts 256-byte aligned (cuBLASLt's plan): the stride padded to 64 words
        let ps = (cols * mp).next_multiple_of(64);
        let key = if chain { self.cksk.as_ref() } else { Some(&self.ksk) };
        let key = key.ok_or_else(|| shape("tfhe native", "no chain key (needs Params::chain)".to_string()))?;
        let regen = key.seeded.as_ref().filter(|_| cb <= ks_regen_max() || key.planes.is_none()).map(|(seed, body)| (*seed, body.vptr()));
        let planes: Vec<*const i8> = key.planes.iter().flatten().map(|b| b.dptr() as *const i8).collect();
        if let Some((seed, body)) = regen {
            let k = n_in * ks_lv;
            let (slices, mt) = ks_regen_shape(cb, cols, k, self.wave / 2);
            let mut c = [mp as u64, ps as u64, seed.tag as u64, k as u64, 0, 0, 0, 0];
            for i in 0..4 {
                c[4 + i] = seed.seed.0[2 * i] as u64 | (seed.seed.0[2 * i + 1] as u64) << 32;
            }
            Self::go(ctx, s, &TfheArgs { op: tfhe_op::KS_REGEN, count: cb as i32, i0: slices as i32, i2: nl as i32, i3: mt as i32, x: self.g.vptr_mut(), y: self.kdig.vptr(), z: body, q: p0, c, ..Default::default() })?;
        } else {
            self.prepare(lt_, cb)?;
            let gemm = &self.gemms[&mp];
            let gp_ = self.g.dptr_mut();
            for (u, &pl) in planes.iter().enumerate() {
                // SAFETY: operands are this switch's buffers of the plan's lengths, on `s`
                unsafe { gemm.run(s, self.kdig.dptr() as *const i8, pl, gp_.add(u * ps), false, &mut self.ws)? };
            }
        }
        Self::go(ctx, s, &TfheArgs { op: tfhe_op::KS_RECOMBINE, count: cb as i32, i1: n_in as i32, i2: nl as i32, x: self.lwe.vptr_mut(), y: self.g.vptr(), z: src, q: p0, c: [0, 0, PLANES as u64, ps as u64, mp as u64, 0, 0, 0], ..Default::default() })
    }

    #[allow(clippy::too_many_arguments)]
    fn native_rot<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, lt_: &'a lt::Lt, src: *const c_void, b: usize, tv: *const c_void, init: i32, offsets: &[u64], dsts: &[*mut c_void]) -> Result<()> {
        let (nb, nl, p0, qb) = (self.nb, self.nl, self.p0, self.p.q_br);
        let kl = offsets.len();
        if self.cksk.is_none() || b == 0 || kl == 0 || dsts.len() != kl || (2 * nb * self.p.nu) % kl != 0 {
            return Err(shape("tfhe native", format!("{b} LWEs, {kl} LUTs / {} outputs (needs Params::chain)", dsts.len())));
        }
        let nu = self.p.nu as u64;
        let mut cx = [0u64; 8];
        cx[5] = nu;
        cx[6] = self.pbs_c[6];
        cx[7] = self.pbs_c[7];
        let mut pbs_c = self.pbs_c;
        pbs_c[5] = nu;
        for c0 in (0..b).step_by(self.max) {
            let cb = self.max.min(b - c0);
            let sc = (src as *const u32).wrapping_add(c0 * (nb + 1)) as *const c_void;
            self.native_ks(ctx, s, lt_, sc, cb)?;
            // init bit 1: one accumulator per LWE, [b][2 part][2 prime][nu n]
            let tvc = if init & 2 != 0 { (tv as *const u32).wrapping_add(c0 * 4 * nb * self.p.nu) as *const c_void } else { tv };
            self.rotate_init(ctx, s, cb, tvc, init, kl, &cx, pbs_c)?;
            let pb = TfheArgs { n: nb as i32, count: cb as i32, i1: nl as i32, q: qb, ..Default::default() };
            for (j, (&off, &d)) in offsets.iter().zip(dsts).enumerate() {
                let x = (d as *mut u32).wrapping_add(c0 * (nb + 1)) as *mut c_void;
                Self::go(ctx, s, &TfheArgs { op: tfhe_op::PBS_EXTRACT, x, y: self.acc.vptr(), m: pc_tfhe::gadget::fixed_ratio(p0, qb), c: [off, p0, 0, 0, j as u64, nu, cx[6], cx[7]], ..pb })?;
            }
        }
        Ok(())
    }

    /// The packing key switch of native LWEs `src` [rows][n_br + 1] (key switched s_br -> s_lwe, then
    /// `DevPack::pack` at q_in = p0) into the encrypted accumulator `out` [2 part][2 prime][n_br]: `place` int32
    /// [count][2] = (row < rows, coefficient < n_br), `w` int32 [count] weights (null: 1). A native value v (phase
    /// v p0 / 4) lands at phase v q_br / 4: what `native_plut` extracts back to v. rows <= the switch's capacity.
    #[allow(clippy::too_many_arguments)]
    pub fn native_pack<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, lt_: &'a lt::Lt, src: *const c_void, rows: usize, place: *const c_void, w: *const c_void, count: usize, out: *mut c_void) -> Result<()> {
        if self.pack.is_none() || self.cksk.is_none() || rows == 0 || rows > self.max || count == 0 {
            return Err(shape("tfhe native", format!("pack of {rows} LWEs (needs Params::chain and Params::pack; <= {} rows)", self.max)));
        }
        self.native_ks(ctx, s, lt_, src, rows)?;
        let p0 = self.p0;
        let lwe = self.lwe.vptr();
        self.pack.as_mut().unwrap().pack(ctx.table, s, lwe, place, w, count, p0, out)
    }

    /// The packing by rows (`DevPack::pack_rows`) of native LWEs `src` [rows][n_br + 1] (key switched s_br -> s_lwe,
    /// packed at q_in = p0): `count` accumulators of `e_n` rows each (`rowmap` int32 [count e_n] into `src`, P / `phat`
    /// the rows' placement polynomials and their `native_pack_basis`) into `out` [count][2 part][2 prime][n_br].
    #[allow(clippy::too_many_arguments)]
    pub fn native_pack_rows<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, lt_: &'a lt::Lt, src: *const c_void, rows: usize, rowmap: *const c_void, count: usize, e_n: usize, p: *const c_void, phat: *const c_void, out: *mut c_void) -> Result<()> {
        if self.pack.is_none() || self.cksk.is_none() || rows == 0 || rows > self.max || count == 0 {
            return Err(shape("tfhe native", format!("pack of {rows} LWEs (needs Params::chain and Params::pack; <= {} rows)", self.max)));
        }
        self.native_ks(ctx, s, lt_, src, rows)?;
        let p0 = self.p0;
        let lwe = self.lwe.vptr();
        self.pack.as_mut().unwrap().pack_rows(ctx.table, s, lwe, rowmap, count, e_n, p, phat, p0, out)
    }

    /// `DevPack::basis` of the switch's packing key: NTT(P) of `e_n` placement polynomials P int32 [e_n][n_br] into
    /// `phat` u64 [e_n][2 prime][n_br].
    pub fn native_pack_basis<M: DevStrategy<W>>(&self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, p: *const c_void, e_n: usize, phat: *mut c_void) -> Result<()> {
        match &self.pack {
            Some(pk) => pk.basis(ctx.table, s, p, e_n, phat),
            None => Err(shape("tfhe native", "no packing key (Params::pack)".to_string())),
        }
    }

    /// The keys' device bytes, per key kind.
    pub fn key_bytes(&self) -> Vec<(String, usize)> {
        let mut v = vec![("ring switch".to_string(), 4 * self.rkey.len()), ("LWE key switch".to_string(), self.ksk.bytes())];
        if let Some(c) = &self.cksk {
            v.push(("chain key switch".to_string(), c.bytes()));
        }
        v.push((format!("bootstrapping (groups of {})", self.p.pbs_group), 4 * self.bsk.len()));
        if let Some(b) = &self.bsk_lat {
            v.push((format!("bootstrapping (groups of {})", self.p.lat_group), 4 * b.len()));
        }
        if let Some(b) = &self.bsk_lat2 {
            v.push((format!("bootstrapping (groups of {})", self.p.lat_group2), 4 * b.len()));
        }
        v.push(("repack columns".to_string(), self.repack.kpl.len()));
        v.push(("repack ring".to_string(), 4 * self.repack.ring.len()));
        if let Some(p) = &self.pack {
            v.push(("packing".to_string(), 4 * p.key.len()));
        }
        v
    }

    /// Whether the switch holds the packing key (`native_pack`).
    pub fn has_pack(&self) -> bool {
        self.pack.is_some()
    }

    /// Native LWEs `src` [b][n_br + 1] (their phase read as the LUT outputs the finish expects: `pbs::warp`ed) into
    /// the split's coefficients `pos` [reps][b] of `x` (zeros elsewhere), as `apply_dev` leaves it.
    #[allow(clippy::too_many_arguments)]
    pub fn native_store<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, lt_: &'a lt::Lt, src: *const c_void, pos: *const c_void, b: usize, reps: usize, x: &mut Ct<'_, W>) -> Result<()> {
        let (n, nb, p0, p1) = (self.n, self.nb, self.p0, self.p1);
        let pa = PcmmArgs { n: n as i32, p: [p0 as u32, p1 as u32, p0 as u32, 0], mu: [mu(p0), mu(p1), mu(p0), 0], c: [self.ninv[0], self.ninv[1], self.ninv[0], 0, 0, 0, 0, 0], ..Default::default() };
        for c0 in (0..b).step_by(self.max) {
            let cb = self.max.min(b - c0);
            self.repack.prepare(lt_, cb)?;
            let sc = (src as *const u32).wrapping_add(c0 * (nb + 1)) as *const c_void;
            let pc = (pos as *const u32).wrapping_add(c0) as *const c_void;
            self.repack.accumulate(ctx.table, s, lt_, sc, pc, cb, c0 == 0, reps, b)?;
        }
        self.finish_into(ctx, s, &pa, x)
    }

    /// The bootstrap of `cb` LWEs in `lwe` (dimension n_lwe, mod p0): mod switch, init on `tv`, the blind rotation
    /// (batch path, latency path or both) into `acc`.
    #[allow(clippy::too_many_arguments)]
    fn rotate<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, cb: usize, tv: *const c_void, kl: usize, cx: &[u64; 8], pbs_c: [u64; 8]) -> Result<()> {
        self.rotate_init(ctx, s, cb, tv, 0, kl, cx, pbs_c)
    }

    /// `rotate` with PBS_INIT's flags `init` (bit 0: `tv` is an encrypted accumulator [2 part][2 prime][n]).
    #[allow(clippy::too_many_arguments)]
    fn rotate_init<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, cb: usize, tv: *const c_void, init: i32, kl: usize, cx: &[u64; 8], pbs_c: [u64; 8]) -> Result<()> {
        let (nb, nl, p0, qb, nu) = (self.nb, self.nl, self.p0, self.p.q_br, self.p.nu as u64);
        let cols = nl + 1;
        let cx = *cx;
        let pb = TfheArgs { n: nb as i32, count: cb as i32, i1: nl as i32, q: qb, ..Default::default() };
        Self::go(ctx, s, &TfheArgs { op: tfhe_op::PBS_MS, x: self.ms.vptr_mut(), y: self.lwe.vptr(), m: pc_tfhe::gadget::fixed_ratio(2 * nu * nb as u64 / kl as u64, p0), c: [p0, 0, 0, 0, kl as u64, nu, 0, 0], ..pb })?;
        Self::go(ctx, s, &TfheArgs { op: tfhe_op::PBS_INIT, i0: init, x: self.acc.vptr_mut(), y: tv, z: self.ms.vptr(), c: cx, ..pb })?;
        let rot = TfheArgs {
            op: tfhe_op::PBS_ROTATE,
            i2: self.p.pbs_group as i32,
            x: self.acc.vptr_mut(),
            y: self.bsk.vptr(),
            z: self.ms.vptr(),
            tw: self.btw.vptr(),
            tws: self.btw_s.vptr(),
            m: self.pbs_m,
            c: pbs_c,
            ..pb
        };
        // the latency path (nu = 2 or two gadget levels: four-block clusters, up to two sub-batches of KB_MAX_EXT LWEs)
        let one = nu == 1 && self.p.pbs.1 == 1;
        let sub = if one { KB_MAX } else { KB_MAX_EXT };
        let lat_max = if one { KB_MAX } else { 2 * KB_MAX_EXT };
        // a batch beyond whole waves: the whole waves on the batch path, the remainder (up to KB_MAX) concurrently
        // on the latency path, not a near-empty last wave
        let (bulk, base) = match cb % self.wave {
            r if one && cb > self.wave && r > 0 && r <= KB_MAX => (cb - r, cb - r),
            _ if cb <= lat_max && nu * self.p.pbs.1 as u64 <= 2 && (one || self.clusters) => (0, 0),
            _ => (cb, cb),
        };
        let cl = cb - base;   // the latency path's LWEs, from `base`
        if cl > 0 {
            // per window the key bundles on `side`, the rotation on `s` once they are in, the next window's bundles
            // filling the other half of `kb` meanwhile
            // the latency key for the small chunks (one level, nu 1): fewer, wider groups
            let lat = self.bsk_lat.as_ref().filter(|_| one && cl <= lat_cap() && (self.clusters || self.p.lat_group <= 4)).map(|k| (k, self.p.lat_group));
            let lat = lat.or_else(|| self.bsk_lat2.as_ref().filter(|_| one && cl > lat_cap() && cl <= lat2_cap() && (self.clusters || self.p.lat_group2 <= 4)).map(|k| (k, self.p.lat_group2)));
            let rot0 = rot;
            let rot = match lat {
                Some((k, g)) => TfheArgs { i2: g as i32, y: k.vptr(), ..rot },
                None => rot,
            };
            let groups = nl / match lat { Some((_, g)) => g, None => self.p.pbs_group };
            let nsub = cl.div_ceil(sub);
            let ext = self.p.nu * self.p.pbs.1;   // uint4 per bundle word
            let half = self.max.min(KB_MAX) * KB_WINDOW * 4 * nb * ext;
            let [e_in, kb0, kb1, r0, r1, e_out] = &self.ev;
            let (ekb, erot) = ([kb0, kb1], [r0, r1]);
            s.record(e_in)?;
            self.side.wait(e_in)?;
            self.rot.wait(e_in)?;
            if bulk > 0 {
                Self::go(ctx, s, &TfheArgs { count: bulk as i32, ..rot0 })?;
            }
            let mut k = 0;
            for j in 0..nsub {
                let (l0, lc) = (base + j * cl / nsub, (j + 1) * cl / nsub - j * cl / nsub);
                let win = if lc * KB_WINDOW * 8 * nb * ext <= half { KB_WINDOW } else { KB_WINDOW / 2 };
                let rk = TfheArgs {
                    count: lc as i32,
                    x: (self.acc.vptr_mut() as *mut u32).wrapping_add(l0 * self.p.nu * 4 * nb) as *mut c_void,
                    z: (self.ms.vptr() as *const u32).wrapping_add(l0 * cols) as *const c_void,
                    ..rot
                };
                for g0 in (0..groups).step_by(win) {
                    let (gw, b) = (win.min(groups - g0), k % 2);
                    let kb = (self.kb.vptr_mut() as *mut u32).wrapping_add(b * half) as *mut c_void;
                    if k >= 2 {
                        self.side.wait(erot[b])?;
                    }
                    Self::go(ctx, &self.side, &TfheArgs { op: tfhe_op::PBS_KEYBUNDLE, i0: g0 as i32, i4: gw as i32, x: kb, ..rk })?;
                    self.side.record(ekb[b])?;
                    self.rot.wait(ekb[b])?;
                    Self::go(ctx, &self.rot, &TfheArgs { w: kb as *const c_void, i0: g0 as i32, i4: gw as i32, ..rk })?;
                    self.rot.record(erot[b])?;
                    k += 1;
                }
            }
            self.rot.record(e_out)?;
            s.wait(e_out)?;
        } else {
            match cb % self.wave {
                // a remainder of up to half a wave: one LWE per block, one wave (two per block would take a whole one)
                r if one && cb > self.wave && r > KB_MAX && r <= self.wave / 2 => {
                    Self::go(ctx, s, &TfheArgs { count: (cb - r) as i32, ..rot })?;
                    let x = (self.acc.vptr_mut() as *mut u32).wrapping_add((cb - r) * 4 * nb) as *mut c_void;
                    let z = (self.ms.vptr() as *const u32).wrapping_add((cb - r) * cols) as *const c_void;
                    Self::go(ctx, s, &TfheArgs { count: r as i32, x, z, ..rot })?;
                }
                _ => Self::go(ctx, s, &rot)?,
            }
        }
        Ok(())
    }

    /// The repack's packed sum into [part][limb][n] mod p0 p1, lifted x p2 p3 (x p2 on a lean chain) into the staging,
    /// scattered into `x` (the split's limbs), NTT.
    fn finish_into<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, pa: &PcmmArgs, x: &mut Ct<'_, W>) -> Result<()> {
        let n = self.n;
        let q = &ctx.chain.q;
        let c4 = [q[0] as u32, q[1] as u32, q[2] as u32, q[3] as u32];
        let cmu = [mu(q[0]), mu(q[1]), mu(q[2]), mu(q[3])];
        let top = crate::bts::stc_first::split_top(&ctx.chain);
        let chain = |opc: i32| PcmmArgs { op: opc, n: n as i32, p: c4, mu: cmu, i4: top as i32 + 1, ..PcmmArgs::default() };
        let q23 = q[2..=top].iter().fold(1u128, |a, &x| a * x as u128);
        let lift = [(q23 % q[0] as u128) as u32, (q23 % q[1] as u128) as u32];
        self.repack.finish(ctx.table, s, pa, self.tw_f.vptr(), self.tw_i.vptr())?;
        for sp in 0..2 {
            // staging part 0 = c1 (racc part 1), staging part 1 = c0 (racc part 0)
            let f = (self.repack.acc.vptr() as *const u32).wrapping_add((1 - sp) * 2 * n) as *const c_void;
            let dst = (self.stage.vptr_mut() as *mut W).wrapping_add(sp * 4 * n) as *mut c_void;
            Self::gp(ctx, s, &PcmmArgs { y: f, x: dst, c: [lift[0], lift[1], 0, 0, 0, 0, 0, 0], ..chain(pcmm_op::LIFT) })?;
        }
        Self::gp(ctx, s, &PcmmArgs { ct: x.buf.r(), i0: 0, y: self.stage.vptr(), ..chain(pcmm_op::SCATTER) })?;
        ntt::forward(ctx, s, &mut x.buf, Span::Q { top }, Parts::Both)?;
        Ok(())
    }
}
