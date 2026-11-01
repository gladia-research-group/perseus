//! CCMM: gates and benches of `pc_gpu::matmul::ccmm`.
//!
//!   pc-check ccmm [--seed s] [--dump \<dir>] [--host-only]   `transpose` / `mul` / `mul_inplace` on random split
//!       ciphertexts with fresh keys vs a host twin of the numpy CKL reference (ccmm_ref.py over ckl_numpy.py), every
//!       output word per prime; `--dump` writes keys, inputs and outputs as .npy for the numpy cross-check
//!   pc-bench ccmm --prec [--cases QK,PV,..]   a real Q K^T (and PV) in the SinC layout through stc_high -> ccmm_mul ->
//!       SinCBoot vs the exact product (fitted scale, bits = log2(max |ref| / max err))
//!   pc-bench ccmm [--reps 10] [--k 1,8]       device ms per CCMM by phase, per SinCBoot ciphertext at K = 1 and 8
use crate::check_bts::Tally;
use pc_core::ccmm_keys::{auto_h, key_words, tstar, CcmmKeys, D, KY, NB, NBIG, NP, NQ, NS, SW};
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::keys::{Prng, SecretKey};
use pc_core::modmul::{invmod, mulmod, Barrett64};
use pc_core::params::{Chain, ChainSpec};
use pc_core::pcmm_keys::{brev, NttTab};
use pc_cuda::{Compute, Device, Event};
use pc_gpu::bts::cts::LtGeom;
use pc_gpu::bts::stc_first::{StcFirstPlan, StcFirstSetup};
use pc_gpu::bts::stc_high::{StcHighSetup, StcHighShape};
use pc_gpu::bts::{BtsCx, BtsSetup, BtsWork, StcHigh};
use pc_gpu::decrypt::{Decryptor, Which};
use pc_gpu::encrypt::Encryptor;
use pc_gpu::keygen::{test_prng, KeyGen};
use pc_gpu::keys::hostref::HostNtt;
use pc_gpu::keys::{DevSecret, KeySet, KeySetSpec, KskLayout, RotSpec};
use pc_gpu::ks::KsPlans;
use pc_gpu::matmul::ccmm::{self, leaf, Ccmm, CcmmCfg, CcmmTimes};
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::poly::CtBuf;
use pc_gpu::{Context, Ct, Executor, MemCfg, MemPlan};
use pc_shadow::{BtsConfig, CtMeta, Quirks, Route, Shadow, ShadowConfig};
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

type W = u32;
type M = Barrett64;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub enum Mode {
    Check,
    Bench,
}

fn arg(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned()
}

fn centered(v: u64, p: u64) -> i64 {
    if v > p / 2 {
        v as i64 - p as i64
    } else {
        v as i64
    }
}

fn smod(c: i64, p: u64) -> u32 {
    c.rem_euclid(p as i64) as u32
}

// ── host twin of the numpy CKL reference (ccmm_ref.py Ref) ──────────────────────────────────────────────────────

/// A polynomial over q0..q3: \[prime\]\[n\] residues.
type P4 = Vec<u32>;
/// A column set (128 columns of R_{2^13}).
type Cols = Vec<P4>;

struct Twin {
    p: [u64; NP],
    t16: [NttTab; NP],
    t13: [NttTab; NP],
    t6: [NttTab; NP],
    fwd: Vec<u32>,
    back: Vec<u32>,
    autos: Vec<u32>,
    rlk: Vec<u32>,
    hat: [u64; NQ],
    pinv: [u64; NQ],
    dinv: [u64; NQ],
}

impl Twin {
    fn new(k: &CcmmKeys) -> Res<Twin> {
        let (t16, t13, t6) = (k.tables(NB)?, k.tables(NS)?, k.tables(KY)?);
        let p = k.primes;
        let mut hat = [0u64; NQ];
        let mut pinv = [0u64; NQ];
        let mut dinv = [0u64; NQ];
        for j in 0..NQ {
            let qq: u64 = (0..NQ).filter(|&i| i != j).fold(1u64, |a, i| mulmod(a, p[i] % p[j], p[j]));
            hat[j] = invmod(qq, p[j]);
            pinv[j] = invmod(p[NQ] % p[j], p[j]);
            dinv[j] = invmod(D as u64, p[j]);
        }
        Ok(Twin {
            fwd: CcmmKeys::eval_form(&k.fwd, &t16),
            back: CcmmKeys::eval_form(&k.back, &t16),
            autos: CcmmKeys::eval_form(&k.autos, &t13),
            rlk: CcmmKeys::eval_form(&k.rlk, &t13),
            p,
            t16,
            t13,
            t6,
            hat,
            pinv,
            dinv,
        })
    }

    /// pcmm_host.key_switch_sum(\[x\], \[key\]) -> (a', b') mod q0..q3 (x coefficient form \[4\]\[n\]; key NTT form).
    fn ks(&self, x: &[u32], n: usize, key: &[u32]) -> (P4, P4) {
        let tabs = if n == NB { &self.t16 } else { &self.t13 };
        let p = self.p;
        let mut acc = vec![0u32; 2 * NP * n];
        for pi in 0..NP {
            let pp = p[pi];
            let (mut sa, mut sb) = (vec![0u64; n], vec![0u64; n]);
            for j in 0..NQ {
                let mut dn: Vec<u32> = (0..n).map(|k| smod(centered(mulmod(x[j * n + k] as u64, self.hat[j], p[j]), p[j]), pp)).collect();
                tabs[pi].ntt(&mut dn);
                let kb = ((j * NP + pi) * 2) * n;
                for k in 0..n {
                    sa[k] = (sa[k] + dn[k] as u64 * key[kb + k] as u64) % pp;
                    sb[k] = (sb[k] + dn[k] as u64 * key[kb + n + k] as u64) % pp;
                }
            }
            for (ab, v) in [sa, sb].into_iter().enumerate() {
                let row = &mut acc[(ab * NP + pi) * n..(ab * NP + pi + 1) * n];
                for (o, x) in row.iter_mut().zip(&v) {
                    *o = *x as u32;
                }
                tabs[pi].intt(row);
            }
        }
        let md = |ab: usize| -> P4 {
            let mut out = vec![0u32; NQ * n];
            for k in 0..n {
                let xp = centered(acc[(ab * NP + NQ) * n + k] as u64, p[NQ]);
                for i in 0..NQ {
                    let v = acc[(ab * NP + i) * n + k] as u64;
                    let t = (v + p[i] - smod(xp, p[i]) as u64) % p[i];
                    out[i * n + k] = mulmod(t, self.pinv[i], p[i]) as u32;
                }
            }
            out
        };
        (md(0), md(1))
    }

    fn add(&self, x: &P4, y: &P4, n: usize) -> P4 {
        x.iter().enumerate().zip(y).map(|((k, &a), &b)| ((a as u64 + b as u64) % self.p[k / n]) as u32).collect()
    }
    fn sub(&self, x: &P4, y: &P4, n: usize) -> P4 {
        x.iter().enumerate().zip(y).map(|((k, &a), &b)| ((a as u64 + self.p[k / n] - b as u64) % self.p[k / n]) as u32).collect()
    }
    fn scale(&self, x: &P4, c: &[u64; NQ], n: usize) -> P4 {
        x.iter().enumerate().map(|(k, &a)| mulmod(a as u64, c[k / n], self.p[k / n]) as u32).collect()
    }

    /// X^e x in R_{NS} (ckl_numpy `monomial`).
    fn monomial(&self, x: &P4, e: i64) -> P4 {
        let n = NS as i64;
        let mut e = e.rem_euclid(2 * n);
        let mut sgn = false;
        if e >= n {
            e -= n;
            sgn = true;
        }
        let mut y = vec![0u32; x.len()];
        for i in 0..NQ {
            let p = self.p[i];
            for k in 0..NS {
                let (src, neg) = if (k as i64) >= e { (k - e as usize, sgn) } else { (k + NS - e as usize, !sgn) };
                let v = x[i * NS + src];
                y[i * NS + k] = if neg && v != 0 { (p - v as u64) as u32 } else { v };
            }
        }
        y
    }

    /// sigma_h in R_{NS} (ckl_numpy `automorph`).
    fn automorph(&self, x: &P4, h: usize) -> P4 {
        let mut y = vec![0u32; x.len()];
        for i in 0..NQ {
            let p = self.p[i];
            for u in 0..NS {
                let t = u * h % (2 * NS);
                let (pos, neg) = if t < NS { (t, false) } else { (t - NS, true) };
                let v = x[i * NS + u];
                y[i * NS + pos] = if neg && v != 0 { (p - v as u64) as u32 } else { v };
            }
        }
        y
    }

    /// out_i = sum_j X^{2 kk i j sgn} cts_j (ckl_numpy `tweak`, recursive).
    fn tweak(&self, cts: &[P4], sgn: i64, kk: i64) -> Cols {
        let n = cts.len();
        if n == 1 {
            return cts.to_vec();
        }
        let ev: Cols = cts.iter().step_by(2).cloned().collect();
        let od: Cols = cts.iter().skip(1).step_by(2).cloned().collect();
        let (ev, od) = rayon::join(|| self.tweak(&ev, sgn, 2 * kk), || self.tweak(&od, sgn, 2 * kk));
        let mut out = vec![Vec::new(); n];
        let pairs: Vec<(P4, P4)> = (0..n / 2)
            .into_par_iter()
            .map(|j| {
                let sh = self.monomial(&od[j], 2 * kk * j as i64 * sgn);
                (self.add(&ev[j], &sh, NS), self.sub(&ev[j], &sh, NS))
            })
            .collect();
        for (j, (a, b)) in pairs.into_iter().enumerate() {
            out[j] = a;
            out[j + n / 2] = b;
        }
        out
    }

    /// CKL Alg. 3 (ccmm_ref.py `Ref.cmt`).
    fn cmt(&self, b: &Cols, a: &Cols) -> (Cols, Cols) {
        let xb: Cols = (0..D).into_par_iter().map(|i| self.monomial(&b[i], i as i64)).collect();
        let xa: Cols = (0..D).into_par_iter().map(|i| self.monomial(&a[i], i as i64)).collect();
        let (tb, ta) = (self.tweak(&xb, 1, KY as i64), self.tweak(&xa, 1, KY as i64));
        let o: Vec<(P4, P4)> = (0..D)
            .into_par_iter()
            .map(|t| {
                let h = auto_h(t);
                let ts = tstar(t);
                let cb = self.automorph(&self.scale(&tb[ts], &self.dinv, NS), h);
                let ca = self.automorph(&self.scale(&ta[ts], &self.dinv, NS), h);
                let kw = key_words(NS);
                let (ka, kb) = self.ks(&ca, NS, &self.autos[t * kw..(t + 1) * kw]);
                (self.add(&cb, &kb, NS), ka)
            })
            .collect();
        let (ob, oa): (Cols, Cols) = o.into_iter().unzip();
        let (rb, ra) = (self.tweak(&ob, -1, KY as i64), self.tweak(&oa, -1, KY as i64));
        let fb: Cols = (0..D).into_par_iter().map(|i| self.monomial(&rb[i], -(i as i64))).collect();
        let fa: Cols = (0..D).into_par_iter().map(|i| self.monomial(&ra[i], -(i as i64))).collect();
        (fb, fa)
    }

    /// 16 ciphertexts (c0, c1) [4][NB] coefficient form -> column sets (b, a) (`Ref.ring_in`).
    fn ring_in(&self, cts: &[(P4, P4)]) -> (Cols, Cols) {
        let parts: Vec<(P4, P4)> = cts
            .par_iter()
            .map(|(c0, c1)| {
                let (ka, kb) = self.ks(c1, NB, &self.fwd);
                (self.add(c0, &kb, NB), ka)
            })
            .collect();
        let mut b = vec![vec![0u32; NQ * NS]; D];
        let mut a = vec![vec![0u32; NQ * NS]; D];
        for (c, (bb, ka)) in parts.iter().enumerate() {
            for j0 in 0..SW {
                for i in 0..NQ {
                    for z in 0..NS {
                        b[SW * c + j0][i * NS + z] = bb[i * NB + j0 + SW * z];
                        a[SW * c + j0][i * NS + z] = ka[i * NB + j0 + SW * z];
                    }
                }
            }
        }
        (b, a)
    }

    /// Column sets -> 16 ciphertexts (c0, c1) coefficient form (`Ref.ring_out`).
    fn ring_out(&self, b: &Cols, a: &Cols) -> Vec<(P4, P4)> {
        (0..NBIG)
            .into_par_iter()
            .map(|c| {
                let mut rb = vec![0u32; NQ * NB];
                let mut ra = vec![0u32; NQ * NB];
                for j0 in 0..SW {
                    for i in 0..NQ {
                        for z in 0..NS {
                            rb[i * NB + j0 + SW * z] = b[SW * c + j0][i * NS + z];
                            ra[i * NB + j0 + SW * z] = a[SW * c + j0][i * NS + z];
                        }
                    }
                }
                let (ka, kb) = self.ks(&ra, NB, &self.back);
                (self.add(&rb, &kb, NB), ka)
            })
            .collect()
    }

    /// C[i][r + D u] = sum_c L[c][i + D .] (*) R[c][r + D .] over R_64 (ckl_numpy `ppmm` on column sets).
    fn ppmm(&self, l: &Cols, r: &Cols) -> Cols {
        let mut out = vec![vec![0u32; NQ * NS]; D];
        for pi in 0..NQ {
            let p = self.p[pi];
            let tab = &self.t6[pi];
            // transformed operands [pt][row][c] (c contiguous)
            let tr = |x: &Cols| -> Vec<u32> {
                let mut h = vec![0u32; KY * D * D];
                let rows: Vec<Vec<u32>> = (0..D * D)
                    .into_par_iter()
                    .map(|idx| {
                        let (c, i) = (idx / D, idx % D);
                        let mut v: Vec<u32> = (0..KY).map(|u| x[c][pi * NS + i + D * u]).collect();
                        tab.ntt(&mut v);
                        v
                    })
                    .collect();
                for (idx, v) in rows.iter().enumerate() {
                    let (c, i) = (idx / D, idx % D);
                    for pt in 0..KY {
                        h[(pt * D + i) * D + c] = v[pt];
                    }
                }
                h
            };
            let (lh, rh) = (tr(l), tr(r));
            let res: Vec<Vec<u32>> = (0..D * D)
                .into_par_iter()
                .map(|idx| {
                    let (i, rr) = (idx / D, idx % D);
                    let mut v = vec![0u32; KY];
                    for pt in 0..KY {
                        let la = &lh[(pt * D + i) * D..(pt * D + i + 1) * D];
                        let ra = &rh[(pt * D + rr) * D..(pt * D + rr + 1) * D];
                        let s: u128 = la.iter().zip(ra).map(|(&x, &y)| x as u128 * y as u128).sum();
                        v[pt] = (s % p as u128) as u32;
                    }
                    tab.intt(&mut v);
                    v
                })
                .collect();
            for (idx, v) in res.iter().enumerate() {
                let (i, rr) = (idx / D, idx % D);
                for u in 0..KY {
                    out[i][pi * NS + rr + D * u] = v[u];
                }
            }
        }
        out
    }

    /// `Ref.mul` (b_transposed: `cts_b` already holds M'^T, its CMT is skipped).
    fn mul(&self, cts_a: &[(P4, P4)], cts_b: &[(P4, P4)], b_transposed: bool) -> Vec<(P4, P4)> {
        let (b, a) = self.ring_in(cts_a);
        let (mut bt, mut at) = self.ring_in(cts_b);
        if !b_transposed {
            (bt, at) = self.cmt(&bt, &at);
        }
        let c00 = self.ppmm(&b, &bt);
        let c01 = self.ppmm(&b, &at);
        let c10 = self.ppmm(&a, &bt);
        let c11 = self.ppmm(&a, &at);
        let (d0, d1) = self.cmt(&c00, &c01);
        let (d2, d3) = self.cmt(&c10, &c11);
        let rl: Vec<(P4, P4)> = (0..D).into_par_iter().map(|j| self.ks(&d3[j], NS, &self.rlk)).collect();
        let ob: Cols = (0..D).map(|j| self.add(&d0[j], &rl[j].1, NS)).collect();
        let oa: Cols = (0..D).map(|j| self.add(&self.add(&d1[j], &d2[j], NS), &rl[j].0, NS)).collect();
        self.ring_out(&ob, &oa)
    }

    fn transpose(&self, cts: &[(P4, P4)]) -> Vec<(P4, P4)> {
        let (b, a) = self.ring_in(cts);
        let (tb, ta) = self.cmt(&b, &a);
        self.ring_out(&tb, &ta)
    }
}

// ── .npy output (the numpy cross-check reads these) ───────────────────────────────────────────────────────────────

fn write_npy(path: &std::path::Path, shape: &[usize], data: &[u32]) -> Res<()> {
    use std::io::Write;
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let sh = if dims.len() == 1 { format!("({},)", dims[0]) } else { format!("({})", dims.join(", ")) };
    let mut header = format!("{{'descr': '<u4', 'fortran_order': False, 'shape': {sh}, }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn flat_cts(cts: &[(P4, P4)]) -> Vec<u32> {
    cts.iter().flat_map(|(c0, c1)| c0.iter().chain(c1.iter()).copied()).collect()
}

// ── pc-check ccmm ────────────────────────────────────────────────────────────────────────────────────────────────

fn check(args: &[String]) -> Res<bool> {
    let seed: u32 = arg(args, "--seed").map(|s| s.parse()).transpose()?.unwrap_or(11);
    let dump = arg(args, "--dump").map(std::path::PathBuf::from);
    let chain = Chain::generate(&ChainSpec::n32())?;
    let mut g = test_prng(0x9b00 + seed);
    let sk = SecretKey::sample(&mut g, chain.n)?;
    let t0 = Instant::now();
    let (keys, _s1) = CcmmKeys::generate(&sk.s, &chain.q[..4], &mut g)?;
    println!("[pc-check] ccmm keys {:.1} s ({:.1} MB), primes {:?}", t0.elapsed().as_secs_f64(), keys.bytes() as f64 / 1e6, keys.primes);
    let roots = chain.roots();
    let hn: Vec<HostNtt> = (0..NQ).map(|i| HostNtt::new(chain.q[i], roots[i], NB)).collect();
    // random inputs: eval rows of limbs 0..3 (both parts); the twin reads their coefficient form
    let mut mats: Vec<Vec<[Vec<u32>; 2]>> = Vec::new();
    for _ in 0..2 {
        let mut m = Vec::new();
        for _ in 0..NBIG {
            let part = |g: &mut Prng| -> Vec<u32> { (0..NQ).flat_map(|i| (0..NB).map(|_| g.uniform_below(chain.q[i]) as u32).collect::<Vec<_>>()).collect() };
            m.push([part(&mut g), part(&mut g)]);
        }
        mats.push(m);
    }
    let coef = |rows: &[u32]| -> P4 {
        let v: Vec<Vec<u32>> = (0..NQ)
            .into_par_iter()
            .map(|i| {
                let mut r: Vec<u64> = rows[i * NB..(i + 1) * NB].iter().map(|&x| x as u64).collect();
                hn[i].inv(&mut r);
                r.into_iter().map(|x| x as u32).collect()
            })
            .collect();
        v.concat()
    };
    let to_eval = |c: &P4| -> Vec<u32> {
        let v: Vec<Vec<u32>> = (0..NQ)
            .into_par_iter()
            .map(|i| {
                let mut r: Vec<u64> = c[i * NB..(i + 1) * NB].iter().map(|&x| x as u64).collect();
                hn[i].fwd(&mut r);
                r.into_iter().map(|x| x as u32).collect()
            })
            .collect();
        v.concat()
    };
    let host_in: Vec<Vec<(P4, P4)>> = mats.iter().map(|m| m.iter().map(|[c0, c1]| (coef(c0), coef(c1))).collect()).collect();

    let th = Instant::now();
    let want_t = Twin::new(&keys)?.transpose(&host_in[0]);
    let want_m = Twin::new(&keys)?.mul(&host_in[0], &host_in[1], false);
    let want_mt = Twin::new(&keys)?.mul(&host_in[0], &host_in[1], true);
    println!("[pc-check] host twin (transpose + 2 products) {:.1} s", th.elapsed().as_secs_f64());
    if let Some(dir) = dump {
        std::fs::create_dir_all(&dir)?;
        write_npy(&dir.join("primes.npy"), &[NP], &keys.primes.map(|x| x as u32))?;
        write_npy(&dir.join("fwd.npy"), &[NQ, NP, 2, NB], &keys.fwd)?;
        write_npy(&dir.join("back.npy"), &[NQ, NP, 2, NB], &keys.back)?;
        write_npy(&dir.join("autos.npy"), &[D, NQ, NP, 2, NS], &keys.autos)?;
        write_npy(&dir.join("rlk.npy"), &[NQ, NP, 2, NS], &keys.rlk)?;
        for (name, v) in [("in_a", &host_in[0]), ("in_b", &host_in[1]), ("out_t", &want_t), ("out_m", &want_m), ("out_mt", &want_mt)] {
            write_npy(&dir.join(format!("{name}.npy")), &[NBIG, 2, NQ, NB], &flat_cts(v))?;
        }
        if args.iter().any(|a| a == "--stages") {
            let tw = Twin::new(&keys)?;
            let (b, a) = tw.ring_in(&host_in[0]);
            let flat = |c: &Cols| -> Vec<u32> { c.iter().flatten().copied().collect() };
            write_npy(&dir.join("ri_b.npy"), &[D, NQ, NS], &flat(&b))?;
            write_npy(&dir.join("ri_a.npy"), &[D, NQ, NS], &flat(&a))?;
            let (tb, ta) = tw.cmt(&b, &a);
            write_npy(&dir.join("cmt_b.npy"), &[D, NQ, NS], &flat(&tb))?;
            write_npy(&dir.join("cmt_a.npy"), &[D, NQ, NS], &flat(&ta))?;
        }
        println!("[pc-check] wrote the numpy cross-check inputs to {}", dir.display());
    }
    if args.iter().any(|a| a == "--host-only") {
        println!("[pc-check] ccmm --host-only: twin outputs written, no device run");
        return Ok(true);
    }
    let dev = Device::open()?;
    let lay = pc_gpu::Layout::from_chain(&chain);
    let mut mp = MemPlan::build::<W, M>(&chain, MemCfg { k_max: 1, pool: vec![], events: 8, scratch_cts: 1 })?;
    let ccfg = CcmmCfg::default();
    mp.reserve("ccmm", Ccmm::<W>::bytes(&ccfg));
    let ctb = pc_cuda::Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, 8));
    mp.reserve("check.cts", 8 * ctb);
    let arena = mp.allocate(&dev)?;
    let ex = Executor::<W>::new(&dev, &arena, &mp)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let lt = pc_cuda::blas::Lt::new(&dev)?;
    let s = &ex.compute;
    let tp = Instant::now();
    let mut cc = Ccmm::<W>::new(&arena, &dev, s, &ctx, &lt.inner, ccfg, &keys)?;
    println!("[pc-check] ccmm setup {:.2} s; PPMM GEMM {} of {} candidates, tuned ms {:?}", tp.elapsed().as_secs_f64(), cc.gemm.label(), cc.gemm.algos(), cc.gemm.tuned_ms);
    // operands as 2 batches of 8 (members of batched ciphertexts), outputs as 16 x k = 1 views of 2 x 8
    let mut cts: Vec<Ct<'_, W>> = (0..8).map(|_| Ok(Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(&arena, "check.cts", lay, 8)?))).collect::<Res<_>>()?;
    arena.freeze();
    let split = CtMeta::new(ctx.shape.n_q - 1 - 3, 1).with_nf(1.0);
    let load = |cts: &mut [Ct<'_, W>], m: &[[Vec<u32>; 2]]| -> Res<()> {
        for (idx, [c0, c1]) in m.iter().enumerate() {
            let x = &mut cts[idx / 8];
            for (part, rows) in [c0, c1].iter().enumerate() {
                x.buf.upload_rows(s, idx % 8, part, &[0, 1, 2, 3], rows)?;
            }
            x.meta = split.clone();
        }
        Ok(())
    };
    let read = |cts: &[Ct<'_, W>]| -> Res<Vec<[Vec<u32>; 2]>> {
        let mut v = Vec::new();
        for idx in 0..NBIG {
            let x = &cts[idx / 8];
            v.push([x.buf.download_rows(s, idx % 8, 0, &[0, 1, 2, 3])?, x.buf.download_rows(s, idx % 8, 1, &[0, 1, 2, 3])?]);
        }
        Ok(v)
    };
    let mut t = Tally::default();
    let cmp = |t: &mut Tally, what: &str, got: &[[Vec<u32>; 2]], want: &[(P4, P4)]| {
        let mut bad = [0usize; NQ];
        let mut tot = 0;
        for (gv, (w0, w1)) in got.iter().zip(want) {
            for (part, w) in [w0, w1].iter().enumerate() {
                let we = to_eval(w);
                for (k, (a, b)) in gv[part].iter().zip(&we).enumerate() {
                    if a != b {
                        bad[k / NB] += 1;
                    }
                }
                tot += we.len();
            }
        }
        t.add(&format!("{what} vs the host twin (per prime {bad:?})"), bad.iter().sum(), tot);
    };
    let ev: Vec<Event> = (0..6).map(|_| Event::timing(&dev)).collect::<Result<_, _>>()?;
    let evr = [&ev[0], &ev[1], &ev[2], &ev[3], &ev[4], &ev[5]];
    let (ca, rest) = cts.split_at_mut(2);
    let (cb, rest) = rest.split_at_mut(2);
    let (co, _) = rest.split_at_mut(2);
    // (a) transpose
    load(ca, &mats[0])?;
    cc.transpose(&ctx, s, ca, co, Some(evr))?;
    let tm = CcmmTimes::read(evr)?;
    cmp(&mut t, &format!("ccmm_transpose (device ms {:.2})", tm.total()), &read(co)?, &want_t);
    // (b) mul, (c) mul with b transposed, (d) in place
    for (bt, want) in [(false, &want_m), (true, &want_mt)] {
        load(ca, &mats[0])?;
        load(cb, &mats[1])?;
        cc.mul(&ctx, s, ca, cb, co, bt, Some(evr))?;
        let tm = CcmmTimes::read(evr)?;
        cmp(&mut t, &format!("ccmm_mul b_transposed={bt} (device ms {:?})", tm.as_array().map(|x| (x * 100.0).round() / 100.0)), &read(co)?, want);
        let meta_ok = co.iter().all(|x| x.meta == split);
        t.add(&format!("ccmm_mul b_transposed={bt}: output metadata = a's"), (!meta_ok) as usize, 1);
    }
    load(ca, &mats[0])?;
    load(cb, &mats[1])?;
    cc.mul_inplace(&ctx, s, ca, cb, false, None)?;
    cmp(&mut t, "ccmm_mul in place (outs = a)", &read(ca)?, &want_m);
    // refusals
    let mut bad_meta = ca[0].meta.clone();
    bad_meta.level -= 2;
    ca[0].meta = bad_meta;
    t.add("ccmm_mul refuses an operand above the split (limbs 0..5)", cc.mul(&ctx, s, ca, cb, co, false, None).is_ok() as usize, 1);
    println!("[pc-check] ccmm: {} gates, {} mismatching", t.gates, t.bad);
    Ok(t.bad == 0)
}

pub fn cmd(mode: Mode, args: &[String]) -> ExitCode {
    let r = match mode {
        Mode::Check => check(args),
        Mode::Bench => bench(args),
    };
    match r {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("ccmm: {e}");
            ExitCode::FAILURE
        }
    }
}

// ── pc-bench ccmm ────────────────────────────────────────────────────────────────────────────────────────────────

fn rot_steps(geoms: &[LtGeom], n: usize) -> BTreeSet<i64> {
    let mut steps = BTreeSet::new();
    for g in geoms {
        for i in 1..g.b {
            steps.insert(((i * g.stride) % (n / 2)) as i64);
        }
        if g.g > 1 {
            steps.insert(((g.b * g.stride) % (n / 2)) as i64);
        }
        steps.insert(g.offset.rem_euclid((n / 2) as i64));
    }
    steps.remove(&0);
    steps
}

const SLOTS: usize = 1 << 15;
const BATCH: usize = 32;
const OFFSET: i64 = -30;

/// sinc.py `sinc_slot`: (row, col, batch l) -> (ciphertext, slot).
fn sinc_slot(row: usize, col: usize, l: usize) -> (usize, usize) {
    (col / SW, ((32 * brev(col % SW + SW * row, 10)) as i64 + l as i64 + OFFSET).rem_euclid(SLOTS as i64) as usize)
}

/// (128, 128, 32) complex, \[row\]\[col\]\[l\] -> 16 SinC slot vectors.
fn pack(z: &[Complex]) -> Vec<Vec<Complex>> {
    let mut out = vec![vec![Complex::new(0.0, 0.0); SLOTS]; NBIG];
    for r in 0..D {
        for c in 0..D {
            for l in 0..BATCH {
                let (ct, sl) = sinc_slot(r, c, l);
                out[ct][sl] = z[(r * D + c) * BATCH + l];
            }
        }
    }
    out
}

fn unpack(v: &[Vec<Complex>]) -> Vec<Complex> {
    let mut z = vec![Complex::new(0.0, 0.0); D * D * BATCH];
    for r in 0..D {
        for c in 0..D {
            for l in 0..BATCH {
                let (ct, sl) = sinc_slot(r, c, l);
                z[(r * D + c) * BATCH + l] = v[ct][sl];
            }
        }
    }
    z
}

/// ccmm_ref.py `decode_cols`: decrypted coefficients (16 x [NB] f64) -> (row, col, batch) values (unscaled).
fn decode_cols(m: &[Vec<f64>]) -> Vec<Complex> {
    let mut z = vec![Complex::new(0.0, 0.0); D * D * BATCH];
    let zeta: Vec<Complex> = (0..2 * KY).map(|e| Complex::new((std::f64::consts::PI * e as f64 / KY as f64).cos(), (std::f64::consts::PI * e as f64 / KY as f64).sin())).collect();
    for c in 0..NBIG {
        for j0 in 0..SW {
            let col = SW * c + j0;
            for i in 0..D {
                for l in 0..BATCH {
                    let e5 = pc_core::modmul::powmod(5, l as u64, 2 * KY as u64) as usize;
                    let mut acc = Complex::new(0.0, 0.0);
                    for u in 0..KY {
                        acc = acc + zeta[(e5 * u) % (2 * KY)] * m[c][j0 + SW * (u * D + i)];
                    }
                    z[(i * D + col) * BATCH + l] = acc;
                }
            }
        }
    }
    z
}

/// Old gate `bits`: (max bits, rms bits) of `got` vs `want` after a fitted real scale, on batch slots < 16.
fn bits(got: &[Complex], want: &[Complex], im: bool) -> (f64, f64, f64) {
    let sel = |z: &Complex| if im { z.im } else { z.re };
    let idx: Vec<usize> = (0..got.len()).filter(|k| k % BATCH < 16).collect();
    let (mut gw, mut ww) = (0.0, 0.0);
    for &k in &idx {
        gw += sel(&got[k]) * sel(&want[k]);
        ww += sel(&want[k]) * sel(&want[k]);
    }
    let c = gw / ww;
    let (mut emax, mut rmax, mut e2) = (0f64, 0f64, 0f64);
    for &k in &idx {
        let e = sel(&got[k]) / c - sel(&want[k]);
        emax = emax.max(e.abs());
        rmax = rmax.max(sel(&want[k]).abs());
        e2 += e * e;
    }
    let std = (e2 / idx.len() as f64).sqrt();
    ((rmax / emax).log2(), (rmax / std).log2(), c)
}

fn gauss(g: &mut Prng) -> f64 {
    let (a, b) = (1.0 - g.uniform_f64(), g.uniform_f64());
    (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

#[allow(clippy::too_many_lines)]
fn bench(args: &[String]) -> Res<bool> {
    let prec = args.iter().any(|a| a == "--prec");
    let reps: usize = arg(args, "--reps").map(|s| s.parse()).transpose()?.unwrap_or(10);
    let cf: Option<u32> = arg(args, "--cf").map(|s| s.parse()).transpose()?;
    let ks_b: Vec<usize> = arg(args, "--k").unwrap_or_else(|| "1,8".into()).split(',').map(|x| x.parse()).collect::<Result<_, _>>()?;
    // the product's |message| (4-limb integer) each case is put at: the reference gate's operating points (sincboot_gate:
    // QK 2^102.01, PV 2^98.75 with real Llama layer-15 data); --msg overrides for every case
    let msg_override: Option<f64> = arg(args, "--msg").map(|s| s.parse()).transpose()?;
    let diag = args.iter().any(|a| a == "--diag");
    let level: u32 = arg(args, "--level").map(|s| s.parse()).transpose()?.unwrap_or(44);
    let kmax = 8usize;
    let t0 = Instant::now();
    let chain = Chain::generate(&ChainSpec::n32())?;
    let n = chain.n;
    let slots = n / 2;
    let mut setup = BtsSetup::n32_shipped();
    setup.raise_drops = vec![0];
    setup.k_max = kmax;
    setup.real = false;
    let mut sf = StcFirstSetup::n32_shipped();
    sf.cache_slots = 4;
    // SinCBoot's stage-0 level: the StC-first landing of the operands' refresh (pc-shadow), one or two units deeper
    let r = Route { stc_first: true, cf, ..Route::dense() };
    let cshape = pc_shadow::ChainShape::of(&chain);
    let land = BtsConfig::n32_shipped().landing(cshape, &Quirks::none(), &Route { cf: Some(cf.or(sf.cf_default).unwrap_or(13)), ..r }, level, 1)?;
    let s0 = ccmm::stage0_level(&CtMeta::new(land.level, land.deg), chain.level_unit as u32);
    let high_shapes = vec![StcHighShape { out_level: 50, skip_mask: 1 }, StcHighShape { out_level: s0, skip_mask: 6 }];
    sf.high = Some(StcHighSetup { shapes: high_shapes.clone(), slots: 4 });
    let plan = StcFirstPlan::new(&chain, &setup, &sf)?;
    let mut geoms: Vec<LtGeom> = Vec::new();
    for v in &plan.dense.variants {
        geoms.extend(v.cts.iter().chain(&v.stc).map(|h| h.geom));
    }
    geoms.extend(plan.input.iter().map(|h| h.geom));
    geoms.extend(plan.sources.iter().flatten().map(|s| s.geom));
    let steps = rot_steps(&geoms, n);
    let spec = KeySetSpec { relin: true, conj: true, rots: steps.iter().map(|&s| RotSpec { step: s, q_top: None }).collect(), encaps: true };
    let dev = Device::open()?;
    let cfg = MemCfg { k_max: kmax, pool: vec![], events: 64, scratch_cts: 2 };
    let mut mp = MemPlan::build::<W, M>(&chain, cfg)?;
    let lay = mp.lay;
    let shapes_k: Vec<_> = spec.keys(&chain)?.into_iter().map(|(_, s)| s).collect();
    mp.reserve("keys", spec.bytes::<W>(&chain, KskLayout::Dense)?);
    mp.reserve("keygen", KeyGen::<W>::bytes(&lay, &shapes_k));
    mp.reserve("secret", DevSecret::<W>::bytes(&lay));
    mp.reserve("encrypt", Encryptor::<W>::bytes(&lay, false));
    mp.reserve("decrypt", Decryptor::<W>::bytes(&lay, chain.level_unit, kmax));
    mp.reserve("ks.plans", KsPlans::bytes::<W>(&chain, kmax)?);
    mp.reserve("ks.regen_a", pc_gpu::ks::KsWork::<W>::regen_bytes(&lay, chain.dnum()));
    let ocfg = OpsCfg::default();
    Ops::<W>::reserve(&mut mp, &ocfg);
    let (req, wshape) = plan.mem::<W>()?;
    req.reserve(&mut mp);
    mp.reserve("bts.work", BtsWork::<W>::bytes(&lay, wshape));
    let ccfg = CcmmCfg::default();
    mp.reserve("ccmm", Ccmm::<W>::bytes(&ccfg));
    let ct_bytes = |k: usize| pc_cuda::Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, k));
    mp.reserve("bench.cts", 4 * ct_bytes(kmax) + 17 * ct_bytes(1));
    mp.reserve("limbs.monomial", pc_gpu::limbs::MonomialTable::<W>::bytes(chain.q.len(), n));
    println!("[pc-bench] MemPlan {}", mp.report());
    let arena = mp.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &mp)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, kmax)?;
    let mut ops = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    let mut work = BtsWork::<W>::new(&arena, lay, wshape)?;
    let mut keys = KeySet::<W>::carve(&arena, &chain, &spec, KskLayout::Dense)?;
    let mut prng = test_prng(0x9b);
    let sk = SecretKey::sample(&mut prng, n)?;
    let mut sec = DevSecret::<W>::carve(&arena, lay)?;
    sec.load(&ctx, &ex.compute, &sk)?;
    let mut kg = KeyGen::<W>::new(&arena, &ex.compute, &chain, &shapes_k)?;
    let tk = kg.generate_set(&ctx, &mut ex, &sec, &mut keys, &mut prng)?;
    println!("[pc-bench] keygen {} keys in {:.1} s", tk.keys, tk.total);
    let tck = Instant::now();
    let (ckeys, _) = CcmmKeys::generate(&sk.s, &chain.q[..4], &mut prng)?;
    println!("[pc-bench] ccmm keys (host) {:.2} s", tck.elapsed().as_secs_f64());
    let mut encr = Encryptor::<W>::new(&arena, &ex.compute, lay, false, &mut prng)?;
    let mut dec = Decryptor::<W>::new(&arena, &ex.compute, lay, chain.level_unit, kmax)?;
    dec.load_secret(&ex.compute, &sec)?;
    let route = plan.build(&arena, &dev, &ex.compute, &ctx, &keys, ex.ks.stage_ptrs())?;
    let lt = pc_cuda::blas::Lt::new(&dev)?;
    let tp = Instant::now();
    let mut cc = Ccmm::<W>::new(&arena, &dev, &ex.compute, &ctx, &lt.inner, ccfg, &ckeys)?;
    println!("[pc-bench] ccmm setup {:.2} s; PPMM GEMM {} ({} candidates, tuned ms {:?})", tp.elapsed().as_secs_f64(), cc.gemm.label(), cc.gemm.algos(), cc.gemm.tuned_ms);
    let mut qa: Vec<Ct<'_, W>> = (0..2).map(|_| Ok(Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(&arena, "bench.cts", lay, kmax)?))).collect::<Res<_>>()?;
    let mut kb: Vec<Ct<'_, W>> = (0..2).map(|_| Ok(Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(&arena, "bench.cts", lay, kmax)?))).collect::<Res<_>>()?;
    let mut scratch = Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(&arena, "bench.cts", lay, 1)?);
    let mut outs16: Vec<Ct<'_, W>> = (0..NBIG).map(|_| Ok(Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(&arena, "bench.cts", lay, 1)?))).collect::<Res<_>>()?;
    // PF_CCMM_MERGE: output pairs merged at the split as e + X^(NB / KY) o (ccmm.py `_device`)
    let mono = pc_gpu::limbs::MonomialTable::<W>::new(&arena, &ex.compute, &ctx, NB / KY)?;
    arena.freeze();
    println!("[pc-bench] setup {:.1} s; arena used {:.1} of {:.1} GB", t0.elapsed().as_secs_f64(), arena.used_bytes() as f64 / 1e9, arena.capacity() as f64 / 1e9);
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let enc = Encoder::new(&chain);
    let nq = ctx.shape.n_q;
    let top = (nq - 1 - level) as usize;
    let sf_ = ctx.scale.sf(top as i32)?;
    let mut fresh = |x: &mut Ct<'_, W>, ex: &mut Executor<'_, W>, vals: &[Vec<Complex>], prng: &mut Prng| -> Res<()> {
        let mut meta = CtMeta::new(level, 1).with_nf(sf_);
        meta.slots = slots as u32;
        x.buf.k = vals.len();
        let msgs: Vec<Vec<Vec<u64>>> = vals.iter().map(|v| enc.encode_slots(v, slots, top, Payload::Complex, sf_)).collect::<std::result::Result<_, _>>()?;
        encr.encrypt_sk(&ctx, &ex.compute, &sec, &msgs, meta, x, prng)?;
        Ok(())
    };
    let a_meta = {
        let mut m = CtMeta::new(level, 1).with_nf(sf_);
        m.slots = slots as u32;
        m
    };
    let st = ccmm::sinc_state(&ctx, &route, &a_meta, &r)?;
    fresh(&mut scratch, &mut ex, &[vec![Complex::new(0.0, 0.0); slots]], &mut prng)?;
    let nf_split = {
        let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
        ccmm::split_nf(&mut cx, &route, &st, &mut scratch)?
    };
    println!("[pc-bench] SinCBoot state: cf {} correction {} n_cts {}; split nf 2^{:.3}; landing (pc-shadow) level {} deg {} -> stage 0 at level {s0}", st.cf, st.correction, st.n_cts, nf_split.log2(), land.level, land.deg);
    let mut t = Tally::default();
    // host decryption of a 4-limb coefficient-form ciphertext (Garner CRT, centered)
    let roots = chain.roots();
    let hn: Vec<HostNtt> = (0..NQ).map(|i| HostNtt::new(chain.q[i], roots[i], NB)).collect();
    let s_eval: Vec<Vec<u64>> = (0..NQ)
        .map(|i| {
            let mut v: Vec<u64> = sk.s.iter().map(|&c| (c as i64).rem_euclid(chain.q[i] as i64) as u64).collect();
            hn[i].fwd(&mut v);
            v
        })
        .collect();
    let q4 = [chain.q[0], chain.q[1], chain.q[2], chain.q[3]];
    let qprod: u128 = q4.iter().map(|&q| q as u128).product();
    let decrypt_coef = |x: &Ct<'_, W>, m: usize, s: &pc_cuda::Stream<Compute>| -> Res<Vec<f64>> {
        let c0 = x.buf.download_rows(s, m, 0, &[0, 1, 2, 3])?;
        let c1 = x.buf.download_rows(s, m, 1, &[0, 1, 2, 3])?;
        let rows: Vec<Vec<u64>> = (0..NQ)
            .into_par_iter()
            .map(|i| {
                let q = q4[i];
                let mut v: Vec<u64> = (0..NB).map(|k| (c0[i * NB + k] as u64 + mulmod(c1[i * NB + k] as u64, s_eval[i][k], q)) % q).collect();
                hn[i].inv(&mut v);
                v
            })
            .collect();
        Ok((0..NB)
            .into_par_iter()
            .map(|k| {
                // Garner: x = r0 + q0 (v1 + q1 (v2 + q2 v3))
                let r: Vec<u64> = (0..NQ).map(|i| rows[i][k]).collect();
                let mut v = [0u64; NQ];
                for i in 0..NQ {
                    let mut x = r[i];
                    let mut prod = 1u64;
                    let mut acc = 0u64;
                    for j in 0..i {
                        acc = (acc + mulmod(v[j], prod, q4[i])) % q4[i];
                        prod = mulmod(prod, q4[j] % q4[i], q4[i]);
                    }
                    x = (x + q4[i] - acc) % q4[i];
                    v[i] = mulmod(x, invmod(prod, q4[i]), q4[i]);
                }
                let mut val: u128 = 0;
                for i in (0..NQ).rev() {
                    val = val * q4[i] as u128 + v[i] as u128;
                }
                if val > qprod / 2 {
                    -((qprod - val) as f64)
                } else {
                    val as f64
                }
            })
            .collect())
    };
    let mut g = test_prng(0x9b01);
    // operands: QK = Q (complex: two heads per slot) x K (real, rows = keys); PV = P (softmax-like rows) x V (real)
    let mk = |g: &mut Prng, complex: bool, positive: bool| -> Vec<Complex> {
        let mut z = vec![Complex::new(0.0, 0.0); D * D * BATCH];
        for r in 0..D {
            for c in 0..D {
                for l in 0..16 {
                    let v = if positive {
                        Complex::new((-3.0 * g.uniform_f64()).exp(), if complex { (-3.0 * g.uniform_f64()).exp() } else { 0.0 })
                    } else {
                        Complex::new(gauss(g), if complex { gauss(g) } else { 0.0 })
                    };
                    z[(r * D + c) * BATCH + l] = v;
                }
            }
        }
        let amax = z.iter().fold(0f64, |m, v| m.max(v.re.abs()).max(v.im.abs()));
        z.iter().map(|v| *v * (0.4 / amax)).collect()
    };
    let exact = |a: &[Complex], b: &[Complex], bt: bool| -> Vec<Complex> {
        let mut y = vec![Complex::new(0.0, 0.0); D * D * BATCH];
        y.par_chunks_mut(D * BATCH).enumerate().for_each(|(i, row)| {
            for k in 0..D {
                for l in 0..BATCH {
                    let mut acc = Complex::new(0.0, 0.0);
                    for j in 0..D {
                        let bv = if bt { b[(k * D + j) * BATCH + l] } else { b[(j * D + k) * BATCH + l] };
                        acc = acc + a[(i * D + j) * BATCH + l] * bv;
                    }
                    row[k * BATCH + l] = acc;
                }
            }
        });
        y
    };
    let ev: Vec<Event> = (0..8).map(|_| Event::timing(&dev)).collect::<Result<_, _>>()?;
    let evr = [&ev[0], &ev[1], &ev[2], &ev[3], &ev[4], &ev[5]];
    // QK: Q K^T (K given as is = M'^T); PV: P V (positive softmax-like P, V through the CMT); QKN / PVT the other
    // operand forms (Gaussian A with the CMT, positive A with b transposed)
    let all: [(&str, bool, bool); 4] = [("QK", true, false), ("PV", false, true), ("QKN", false, false), ("PVT", true, true)];
    let list = arg(args, "--cases").unwrap_or_else(|| if prec { "QK,PV".into() } else { "QK".into() });
    let cases: Vec<(&str, bool, bool)> = list.split(',').filter(|c| *c != "QKM").map(|c| all.iter().find(|x| x.0 == c).copied().ok_or_else(|| format!("unknown case {c}"))).collect::<Result<_, _>>()?;
    let mut logf = -2.0f64;
    for (name, bt, pos) in cases {
        let target_msg = msg_override.unwrap_or(if pos { 98.75 } else { 102.0 });
        let (za, zb) = (mk(&mut g, true, pos), mk(&mut g, false, false));
        let want = exact(&za, &zb, bt);
        let (va, vb) = (pack(&za), pack(&zb));
        // stc_high(skip 1) factor 2^logf: tuned once so |product| ~ 2^102 (the reference gate's level)
        macro_rules! run {
            ($logf:expr) => {{
                for h in 0..2 {
                    fresh(&mut qa[h], &mut ex, &va[8 * h..8 * h + 8], &mut prng)?;
                    fresh(&mut kb[h], &mut ex, &vb[8 * h..8 * h + 8], &mut prng)?;
                }
                let spec = StcHigh { slots: slots as u32, out_level: 50, factor: ($logf as f64).exp2(), skip_mask: 1, nf_ref: 0.0 };
                let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                pc_gpu::bts::Staged::stc_high(&route, &mut cx, &mut qa, &spec)?;
                pc_gpu::bts::Staged::stc_high(&route, &mut cx, &mut kb, &spec)?;
                cc.mul_inplace(&ctx, &cx.ex.compute, &mut qa, &kb, bt, Some(evr))?;
                CcmmTimes::read(evr)?
            }};
        }
        let _ = run!(logf);
        let lo = (0..2).map(|m| decrypt_coef(&qa[0], m, &ex.compute).map(|v| v.iter().fold(0f64, |a, x| a.max(x.abs())).log2())).collect::<Res<Vec<f64>>>()?;
        let lo = lo.iter().fold(f64::MIN, |a, &b| a.max(b));
        logf += (target_msg - lo) / 2.0;
        let tm = run!(logf);
        let mut coefs = Vec::new();
        for idx in 0..NBIG {
            coefs.push(decrypt_coef(&qa[idx / 8], idx % 8, &ex.compute)?);
        }
        let msg = coefs.iter().flatten().fold(0f64, |a, x| a.max(x.abs())).log2();
        let got = decode_cols(&coefs);
        let (bre, bim) = (bits(&got, &want, false), bits(&got, &want, true));
        let used = got.iter().enumerate().filter(|(k, _)| k % BATCH < 16).fold(0f64, |a, (_, v)| a.max(v.abs()));
        let unused = got.iter().enumerate().filter(|(k, _)| k % BATCH >= 16).fold(0f64, |a, (_, v)| a.max(v.abs()));
        println!(
            "[pc-bench] ccmm {name} (b_transposed={bt}) product: Re {:.2} bits max {:.2} rms, Im {:.2} / {:.2}; |msg| 2^{msg:.2} of Q/2 2^{:.2} (stc_high factor 2^{logf:.2}); unused slots {:.1e} rel; device ms {:?} (old QK: 31.97 / 31.67 bits max)",
            bre.0,
            bre.1,
            bim.0,
            bim.1,
            (qprod as f64).log2() - 1.0,
            unused / used,
            tm.as_array().map(|x| (x * 100.0).round() / 100.0)
        );
        if bt {
            t.add(&format!("ccmm {name} product >= 30 bits (Re {:.2}, Im {:.2})", bre.0, bim.0), (!(bre.0.min(bim.0) >= 30.0)) as usize, 1);
        }
        // SinCBoot: the scale the split asks for, set from the fitted product scale (ccmm.py c_unit's role)
        let nf_out = nf_split;
        let b0 = Instant::now();
        if diag {
            // HalfBTS alone, its coefficient-domain output (4-limb CRT) vs the product, then stage 0
            {
                let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                for x in qa.iter_mut() {
                    leaf::set_scale(x, nf_out, 2);
                }
                let mut call = pc_gpu::bts::staged::BtsCall::resume(pc_gpu::bts::staged::StagedRef::StcFirst(&route), st.clone(), pc_gpu::bts::staged::Phase::Split, Default::default());
                call.at_split(&cx, &qa)?;
                call.run_to(&mut cx, &mut qa, pc_gpu::bts::staged::Phase::Done)?;
            }
            let mut coefs = Vec::new();
            for idx in 0..NBIG {
                coefs.push(decrypt_coef(&qa[idx / 8], idx % 8, &ex.compute)?);
            }
            let msg = coefs.iter().flatten().fold(0f64, |a, x| a.max(x.abs())).log2();
            let got = decode_cols(&coefs);
            let (bre, bim) = (bits(&got, &want, false), bits(&got, &want, true));
            println!("[pc-bench] ccmm {name} HalfBTS output (level {} deg {} nf 2^{:.2}): coefficients Re {:.2} bits max {:.2} rms, Im {:.2} / {:.2}, |msg| 2^{msg:.2} (4-limb CRT)", qa[0].meta.level, qa[0].meta.deg, qa[0].meta.nf.log2(), bre.0, bre.1, bim.0, bim.1);
            let m = qa[0].meta.clone();
            let spec = StcHigh { slots: slots as u32, out_level: ccmm::stage0_level(&m, chain.level_unit as u32), factor: 1.0, skip_mask: 6, nf_ref: 0.0 };
            let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
            pc_gpu::bts::Staged::stc_high(&route, &mut cx, &mut qa, &spec)?;
        } else {
            let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
            ccmm::sinc_boot(&mut cx, &route, &st, &mut qa, nf_out)?;
        }
        ex.compute.sync()?;
        let sb_ms = b0.elapsed().as_secs_f64() * 1e3;
        let mut vs = Vec::new();
        for h in 0..2 {
            vs.extend(dec.decrypt_slots(&ctx, &ex.compute, &enc, &qa[h], Payload::Complex, Which::Dense)?);
        }
        let got = unpack(&vs);
        let (sre, sim) = (bits(&got, &want, false), bits(&got, &want, true));
        let used = got.iter().enumerate().filter(|(k, _)| k % BATCH < 16).fold(0f64, |a, (_, v)| a.max(v.abs()));
        let unused = got.iter().enumerate().filter(|(k, _)| k % BATCH >= 16).fold(0f64, |a, (_, v)| a.max(v.abs()));
        println!(
            "[pc-bench] ccmm {name} SinCBoot (16 cts as 2 x K=8, host {sb_ms:.1} ms): Re {:.2} bits max {:.2} rms, Im {:.2} / {:.2}, scale {:.4e}; out level {} deg {}; unused slots {:.1e} rel (old QK 14.55 / 14.26, PV 16.82 / 16.69)",
            sre.0,
            sre.1,
            sim.0,
            sim.1,
            sre.2,
            qa[0].meta.level,
            qa[0].meta.deg,
            unused / used
        );
        t.add(&format!("ccmm {name} SinCBoot output level == stage-0 shape {s0}"), (qa[0].meta.level != s0) as usize, 1);
        if bt {
            t.add(&format!("ccmm {name} SinCBoot >= 14 bits (Re {:.2}, Im {:.2})", sre.0, sim.0), (!(sre.0.min(sim.0) >= 14.0)) as usize, 1);
        }
        if !prec {
            // (3) timing: CCMM per call by phase; SinCBoot per ciphertext at K in ks_b
            let mut per: Vec<[f64; 5]> = Vec::new();
            for _ in 0..reps {
                let tm = run!(logf);
                per.push(tm.as_array());
            }
            let med: Vec<f64> = (0..5).map(|i| median(per.iter().map(|p| p[i]).collect())).collect();
            let tot = median(per.iter().map(|p| p.iter().sum()).collect());
            println!("[pc-bench] ccmm {name} per CCMM (16 x 16 cts, 32 heads): {tot:.2} ms = ring in {:.2} + CMT {:.2} + PPMM {:.2} + CMT x2 + relin {:.2} + ring out {:.2}; median of {reps} (old 39 ms: 19.3 / 1.7 / 2.0 / 4.9 / 11.0)", med[0], med[1], med[2], med[3], med[4]);
            for &k in &ks_b {
                let mut ms = Vec::new();
                for rep in 0..reps + 1 {
                    run!(logf);
                    // a batch of k members of the product (k <= 8: the first Ct's)
                    let full = qa[0].buf.k;
                    qa[0].buf.k = k;
                    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
                    ev[6].record(&cx.ex.compute)?;
                    ccmm::sinc_boot(&mut cx, &route, &st, std::slice::from_mut(&mut qa[0]), nf_out)?;
                    ev[7].record(&cx.ex.compute)?;
                    ev[7].sync()?;
                    qa[0].buf.k = full;
                    if rep > 0 {
                        ms.push(ev[6].elapsed_ms(&ev[7])? as f64 / k as f64);
                    }
                }
                println!("[pc-bench] ccmm SinCBoot K={k}: {:.2} ms per ciphertext (HalfBTS + stage 0), median of {reps}; per CCMM with merge (8 SinCBoots) {:.1} ms (old: 28.5 s / 512 = 55.7 ms host per SinCBoot; HalfBTS 25.0 + stage 0 2.5 device)", median(ms.clone()), 8.0 * median(ms));
            }
            if let Some(c) = route.high.as_ref() {
                for (shp, cache) in &c.caches {
                    println!("[pc-bench] stc_high cache {shp:?}: {:?}", cache.stats());
                }
            }
        }
        let _ = leaf::noise_factor(&qa[0]);
    }
    if list.split(',').any(|c| c == "QKM") || (prec && arg(args, "--cases").is_none()) {
        // merged QK: operands duplicated on l + 16 (sinc.dup), the 16 outputs merged pairwise (2i, 2i + 1) at the
        // split, 8 SinCBoots; reference (e + w o) / 2 per slot (ccmm.py `merged`, w = w_theory)
        let dup = |mut z: Vec<Complex>| -> Vec<Complex> {
            for k in 0..D * D {
                for l in 0..16 {
                    z[k * BATCH + l + 16] = z[k * BATCH + l];
                }
            }
            z
        };
        let (za, zb) = (dup(mk(&mut g, true, false)), dup(mk(&mut g, false, false)));
        let want = pack(&exact(&za, &zb, true));
        let wt: Vec<Complex> = (0..BATCH).map(|l| {
            let e = pc_core::modmul::powmod(5, l as u64, 2 * KY as u64) as f64 * std::f64::consts::PI / KY as f64;
            Complex::new(e.cos(), e.sin())
        }).collect();
        let (va, vb) = (pack(&za), pack(&zb));
        let target = msg_override.unwrap_or(102.0) - 0.5;
        let mut logf = -2.0f64;
        for pass in 0..2 {
            for h in 0..2 {
                fresh(&mut qa[h], &mut ex, &va[8 * h..8 * h + 8], &mut prng)?;
                fresh(&mut kb[h], &mut ex, &vb[8 * h..8 * h + 8], &mut prng)?;
            }
            let spec = StcHigh { slots: slots as u32, out_level: 50, factor: logf.exp2(), skip_mask: 1, nf_ref: 0.0 };
            let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
            pc_gpu::bts::Staged::stc_high(&route, &mut cx, &mut qa, &spec)?;
            pc_gpu::bts::Staged::stc_high(&route, &mut cx, &mut kb, &spec)?;
            cc.mul(&ctx, &cx.ex.compute, &qa, &kb, &mut outs16, true, None)?;
            let lo = decrypt_coef(&outs16[0], 0, &ex.compute)?.iter().fold(0f64, |a, x| a.max(x.abs())).log2();
            if pass == 0 {
                logf += (target - lo) / 2.0;
            }
        }
        for i in 0..NBIG / 2 {
            let (e, o) = outs16.split_at_mut(2 * i + 1);
            pc_gpu::limbs::merge_monomial(&ctx, &mut ex, &mono, &mut e[2 * i], &o[0])?;
        }
        let msg = (0..NBIG / 2).map(|i| decrypt_coef(&outs16[2 * i], 0, &ex.compute).map(|v| v.iter().fold(0f64, |a, x| a.max(x.abs())).log2())).collect::<Res<Vec<f64>>>()?;
        let msg = msg.iter().fold(f64::MIN, |a, &b| a.max(b));
        for i in 0..NBIG / 2 {
            outs16.swap(i, 2 * i);
        }
        let ev2 = (Event::timing(&dev)?, Event::timing(&dev)?);
        {
            let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh, ks: &kplans, ops: &mut ops, work: &mut work };
            ev2.0.record(&cx.ex.compute)?;
            ccmm::sinc_boot(&mut cx, &route, &st, &mut outs16[..NBIG / 2], nf_split)?;
            ev2.1.record(&cx.ex.compute)?;
        }
        ev2.1.sync()?;
        let mut got = Vec::new();
        for x in &outs16[..NBIG / 2] {
            got.extend(dec.decrypt_slots(&ctx, &ex.compute, &enc, x, Payload::Complex, Which::Dense)?.remove(0));
        }
        for (wname, conj) in [("w_theory", false), ("conj(w_theory)", true)] {
            let mut r = Vec::with_capacity(got.len());
            for i in 0..NBIG / 2 {
                for sl in 0..SLOTS {
                    let w = wt[(sl as i64 - OFFSET).rem_euclid(BATCH as i64) as usize];
                    let w = if conj { Complex::new(w.re, -w.im) } else { w };
                    r.push((want[2 * i][sl] + w * want[2 * i + 1][sl]) * 0.5);
                }
            }
            let fit = |im: bool| -> (f64, f64) {
                let sel = |z: &Complex| if im { z.im } else { z.re };
                let gw: f64 = got.iter().zip(&r).map(|(a, b)| sel(a) * sel(b)).sum();
                let ww: f64 = r.iter().map(|b| sel(b) * sel(b)).sum();
                let c = gw / ww;
                let emax = got.iter().zip(&r).fold(0f64, |m, (a, b)| m.max((sel(a) / c - sel(b)).abs()));
                let rmax = r.iter().fold(0f64, |m, b| m.max(sel(b).abs()));
                ((rmax / emax).log2(), c)
            };
            let (re, im) = (fit(false), fit(true));
            println!("[pc-bench] ccmm QKM merged (8 SinCBoots of 16 outputs, merged |msg| 2^{msg:.2}, product factor 2^{logf:.2}) vs (e + w o) / 2 with {wname}: Re {:.2} bits max, Im {:.2} (scales {:.4e} / {:.4e}); SinCBoot device {:.2} ms per merged ct (8 x K=1)", re.0, im.0, re.1, im.1, ev2.0.elapsed_ms(&ev2.1)? as f64 / 8.0);
            if !conj {
                t.add(&format!("ccmm QKM merged SinCBoot >= 14 bits (Re {:.2}, Im {:.2})", re.0, im.0), (!(re.0.min(im.0) >= 14.0)) as usize, 1);
            }
        }
    }
    println!("[pc-bench] ccmm: {} gates, {} failing", t.gates, t.bad);
    Ok(t.bad == 0)
}

