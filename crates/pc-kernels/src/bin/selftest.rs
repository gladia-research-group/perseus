//! pc-kernels selftest (no dumps): every kernel of every built (word, ModMul) instantiation against host references
//! (exact u128 arithmetic), and the NTT throughput sweep.
//!
//! usage (GPU only through tools/gpu.sh): pc-kernels-selftest [--all | \<instance>...] [--no-bench] [--ring | --ring-only]
//! (`--ring`: also the ring-size tables of every built word, log n 10..17; needs feature `ring`)
#[path = "selftest/dev.rs"]
mod dev;
#[path = "selftest/host.rs"]
mod host;
#[cfg(feature = "ring")]
#[path = "selftest/ring.rs"]
mod ring;

use dev::{Ctx, Dev};
use host::*;
use pc_kernels::*;
use pc_cuda::{Arena, Compute, Device, Event, Stream};
use std::ffi::c_void;

struct TestResult {
    name: String,
    bad: u64,
    total: u64,
    note: String,
}

fn cmp(name: &str, got: &[u64], want: &[u64]) -> TestResult {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let bad = got.iter().zip(want).filter(|(a, b)| a != b).count() as u64;
    TestResult { name: name.into(), bad, total: got.len() as u64, note: String::new() }
}

#[track_caller]
fn ck(rc: i32) {
    assert!(rc == 0, "launch returned {rc}");
    if let Err(e) = pc_cuda::error::last_error("selftest launch") {
        panic!("{e}");
    }
}

fn prime_table(primes: &[u64], word_bits: u32) -> Vec<u64> {
    primes
        .iter()
        .flat_map(|&p| {
            let c = PrimeConst::new(p, word_bits);
            [c.p, c.mu_lo, c.mu_hi, c.neal_mu, c.bits as u64 | ((c.word_bits as u64) << 32), c.mont_pinv, c.mont_r2, 0]
        })
        .collect()
}

struct Ntt<'a> {
    fw: Dev<'a>,
    fws: Dev<'a>,
    iw: Dev<'a>,
    iws: Dev<'a>,
    ninv: Dev<'a>,
    ninvs: Dev<'a>,
    w: Vec<Vec<u64>>,
}

struct Setup<'a> {
    ctx: Ctx<'a>,
    t: &'static KernelTable,
    inst: Instance,
    wbits: u32,
    primes: Vec<u64>,
    pr: Dev<'a>,
    ntt: Option<Ntt<'a>>,
    rng: Rng,
}

impl<'a> Setup<'a> {
    fn s(&self) -> CudaStream {
        self.ctx.s.raw()
    }
    fn pr(&self) -> *const PrimeConst {
        self.pr.c() as *const PrimeConst
    }
    fn p(&self, id: usize) -> u64 {
        self.primes[id]
    }
    fn ntt_args(&self, base: *mut c_void, roff: &Dev, rprime: &Dev, nrows: usize) -> NttArgs {
        let t = self.ntt.as_ref().expect("NTT tables");
        NttArgs {
            base,
            roff: roff.c() as *const u32,
            rprime: rprime.c() as *const i32,
            fw: t.fw.c(),
            fws: t.fws.c(),
            iw: t.iw.c(),
            iws: t.iws.c(),
            ninv: t.ninv.c(),
            ninvs: t.ninvs.c(),
            pr: self.pr(),
            nrows: nrows as i32,
            chunk: 128,
            ..Default::default()
        }
    }
    /// rows of `nrows` limbs of prime ids pid(row) per slot of a buffer of 8-row slots
    fn rows(&self, list: &[(u32, i32)]) -> (Dev<'a>, Dev<'a>) {
        let off: Vec<u32> = list.iter().map(|r| r.0).collect();
        let pid: Vec<i32> = list.iter().map(|r| r.1).collect();
        (self.ctx.up32(&off), self.ctx.up_i32(&pid))
    }
    /// random residues for `nslots` 8-row slots (row r of a slot < prime r)
    fn rand_slots(&mut self, nslots: usize) -> Vec<u64> {
        let mut v = vec![0u64; nslots * 8 * N];
        for (i, x) in v.iter_mut().enumerate() {
            let p = self.primes[(i / N) % 8];
            *x = self.rng.below(p);
        }
        v
    }
    fn per_row(&mut self, bound2: bool) -> Vec<u64> {
        (0..8).map(|r| self.rng.below(if bound2 { 2 * self.primes[r] } else { self.primes[r] })).collect()
    }
    fn shoup_row(&self, v: &[u64]) -> Vec<u64> {
        v.iter().enumerate().map(|(r, &x)| shoup(x, self.primes[r % 8], self.wbits)).collect()
    }
}

// ================================================================================================ NTT
fn t_ntt(st: &mut Setup) -> Vec<TestResult> {
    let mut out = Vec::new();
    let host_in = st.rand_slots(1);
    let buf = st.ctx.up(&host_in);
    let list: Vec<(u32, i32)> = (0..8).map(|i| (i, i as i32)).collect();
    let (ro, rp) = st.rows(&list);
    let a = st.ntt_args(buf.p(), &ro, &rp, 8);
    ck(unsafe { (st.t.ntt_fwd)(&a, st.s()) });
    let got = buf.read(&st.ctx);
    let w = &st.ntt.as_ref().unwrap().w;
    let mut want = host_in.clone();
    for i in 0..8 {
        ntt_ref(&mut want[i * N..(i + 1) * N], &w[i], st.primes[i]);
    }
    out.push(cmp("ntt.fwd vs host", &got, &want));
    ck(unsafe { (st.t.ntt_inv)(&a, st.s()) });
    out.push(cmp("ntt.inv(fwd(x)) == x", &buf.read(&st.ctx), &host_in));
    out
}

// ================================================================================================ BConv
fn t_bconv(st: &mut Setup) -> Vec<TestResult> {
    let wb = st.ctx.wb;
    let src_h = st.rand_slots(1);
    let src = st.ctx.up(&src_h);
    // job 0: src rows 0..3 -> 5 targets (primes 3..8) at dst rows 0..5, plain
    // job 1: src rows 4..8 (src_off 4) -> 4 targets (primes 0..4) at dst rows 5..9 (dst_off 5), exact
    struct J {
        src_off: usize,
        dst_off: usize,
        sp: Vec<usize>,
        tp: Vec<usize>,
        exact: bool,
    }
    let jobs_h = [
        J { src_off: 0, dst_off: 0, sp: vec![0, 1, 2], tp: vec![3, 4, 5, 6, 7], exact: false },
        J { src_off: 4, dst_off: 5, sp: vec![4, 5, 6, 7], tp: vec![0, 1, 2, 3], exact: true },
    ];
    let mut pool: Vec<u64> = Vec::new();
    let mut jobs = Vec::new();
    let mut want = vec![0u64; 9 * N];
    for j in &jobs_h {
        let (ns, nt) = (j.sp.len(), j.tp.len());
        let off = pool.len();
        let pre: Vec<u64> = j.sp.iter().map(|&s| st.rng.below(st.p(s))).collect();
        let m: Vec<Vec<u64>> = j.sp.iter().map(|_| j.tp.iter().map(|&t| st.rng.below(st.p(t))).collect()).collect();
        let rb: Vec<f64> = j.sp.iter().map(|&s| 1.0 / st.p(s) as f64).collect();
        let bmod: Vec<u64> = j.tp.iter().map(|&t| st.rng.below(st.p(t))).collect();
        pool.extend((0..ns).map(|i| i as u64));
        pool.extend(j.sp.iter().map(|&s| s as u64));
        pool.extend(&pre);
        pool.extend((0..nt).map(|i| i as u64));
        pool.extend(j.tp.iter().map(|&t| t as u64));
        for row in &m {
            pool.extend(row);
        }
        if j.exact {
            while (pool.len() * wb) % 8 != 0 {
                pool.push(0);
            }
            for &x in &rb {
                let bits = x.to_bits();
                match wb {
                    8 => pool.push(bits),
                    4 => pool.extend([bits & 0xFFFF_FFFF, bits >> 32]),
                    _ => pool.extend((0..4).map(|q| (bits >> (16 * q)) & 0xFFFF)),
                }
            }
            pool.extend(&bmod);
        }
        jobs.push(BconvJob {
            src_off: j.src_off as u32,
            dst_off: j.dst_off as u32,
            nsrc: ns as u32,
            ntgt: nt as u32,
            pool_off: off as u32,
            exact: j.exact as u32,
            pad: [0; 2],
        });
        for idx in 0..N {
            let y: Vec<u64> = (0..ns)
                .map(|i| mulmod(src_h[(j.src_off + i) * N + idx], pre[i], st.p(j.sp[i])))
                .collect();
            let mut v = 0.0f64;
            if j.exact {
                for i in 0..ns {
                    v = (y[i] as f64).mul_add(rb[i], v);
                }
            }
            let u = (v + 0.5).floor() as u64;
            for (k, &t) in j.tp.iter().enumerate() {
                let p = st.p(t);
                let acc: u128 = (0..ns).map(|i| y[i] as u128 * m[i][k] as u128).sum();
                let mut r = (acc % p as u128) as u64;
                if j.exact {
                    r = submod(r, mulmod(u % p, bmod[k], p), p);
                }
                want[(j.dst_off + k) * N + idx] = r;
            }
        }
    }
    let pool_d = st.ctx.up(&pool);
    let job_words: Vec<u32> = jobs
        .iter()
        .flat_map(|j| [j.src_off, j.dst_off, j.nsrc, j.ntgt, j.pool_off, j.exact, 0, 0])
        .collect();
    let jobs_d = st.ctx.up32(&job_words);
    let dst = st.ctx.zeros(9 * N);
    let a = BconvArgs {
        src: src.c(),
        dst: dst.p(),
        jobs: jobs_d.c() as *const BconvJob,
        pool: pool_d.c(),
        pr: st.pr(),
        njobs: 2,
        max_tgt: 5,
    };
    ck(unsafe { (st.t.bconv)(&a, st.s()) });
    let got = dst.read(&st.ctx);
    vec![cmp("bconv (plain + exact HPS) vs host", &got, &want)]
}

// ================================================================================================ element-wise
const RS: RowSet = RowSet { nq: 3, np: 2, qmax: 5, pad: 0 };
fn rs_rows(rs: RowSet) -> Vec<usize> {
    (0..rs.nq as usize).chain((rs.qmax as usize)..(rs.qmax + rs.np) as usize).collect()
}
/// K cts x 2 parts of 8-row slots
fn ctref(d: &Dev) -> PolyRef {
    PolyRef { base: d.p(), ct_stride: (16 * N) as u64, part_stride: (8 * N) as u64 }
}
fn idx(c: usize, part: usize, row: usize, i: usize) -> usize {
    ((c * 2 + part) * 8 + row) * N + i
}

fn t_ew(st: &mut Setup) -> Vec<TestResult> {
    let k = 2;
    let (ah, bh) = (st.rand_slots(2 * k), st.rand_slots(2 * k));
    let (a, b) = (st.ctx.up(&ah), st.ctx.up(&bh));
    let sc = st.per_row(false);
    let scs = st.shoup_row(&sc);
    let (scd, scsd) = (st.ctx.up(&sc), st.ctx.up(&scs));
    let mut out = Vec::new();
    let rows = rs_rows(RS);
    for (name, kind) in [("ew_add", 0), ("ew_sub", 1), ("ew_scale (zero P)", 2), ("ew_sub_mul", 3), ("ew_copy", 4)] {
        let o = st.ctx.zeros(2 * k * 8 * N);
        let args = EwArgs {
            out: ctref(&o),
            a: ctref(&a),
            b: ctref(&b),
            rs: RS,
            sc: scd.c(),
            scs: scsd.c(),
            pr: st.pr(),
            k: k as i32,
            nparts: 2,
            sub: (kind == 1) as i32,
            zero_p: (kind == 2) as i32,
        };
        let f = match kind {
            0 | 1 => st.t.ew_add,
            2 => st.t.ew_scale,
            3 => st.t.ew_sub_mul,
            _ => st.t.ew_copy,
        };
        ck(unsafe { f(&args, st.s()) });
        let mut want = vec![0u64; 2 * k * 8 * N];
        for c in 0..k {
            for part in 0..2 {
                for (y, &r) in rows.iter().enumerate() {
                    let p = st.p(r);
                    for i in 0..N {
                        let j = idx(c, part, r, i);
                        want[j] = match kind {
                            0 => addmod(ah[j], bh[j], p),
                            1 => submod(ah[j], bh[j], p),
                            2 => if y >= RS.nq as usize { 0 } else { mulmod(ah[j], sc[r], p) },
                            3 => mulmod(submod(ah[j], bh[j], p), sc[r], p),
                            _ => ah[j],
                        };
                    }
                }
            }
        }
        out.push(cmp(&format!("{name} vs host"), &o.read(&st.ctx), &want));
    }
    out
}

fn t_pt_dot(st: &mut Setup) -> Vec<TestResult> {
    let mut out = Vec::new();
    let (k, bstep, gstep, npts, pt_prow) = (3usize, 3usize, 2usize, 5usize, 3usize);
    let rows = rs_rows(RS);
    let slot = 16 * N; // one ciphertext (2 parts x 8 rows)
    let roth = st.rand_slots(2 * k * bstep);
    let rot = st.ctx.up(&roth);
    for (enc, ptw, label) in [(0xFFFF_FFFFu32, N, "full"), (0x7F00_0001u32, N / 2, "compact")] {
        // diagonal d: 5 rows (Q rows 0..3 at their index, specials at pt_prow + (y - nq)), ptw words each
        let pth: Vec<Vec<u64>> = (0..npts)
            .map(|_| {
                (0..5 * ptw)
                    .map(|x| {
                        let prow = x / ptw;
                        let r = if prow < 3 { prow } else { RS.qmax as usize + prow - pt_prow };
                        st.rng.below(st.p(r))
                    })
                    .collect()
            })
            .collect();
        let ptd: Vec<Dev> = pth.iter().map(|v| st.ctx.up(v)).collect();
        let ptrs: Vec<*const c_void> = ptd.iter().map(|d| d.c()).collect();
        let ptsd = st.ctx.up_ptrs(&ptrs);
        let res = st.ctx.zeros(k * gstep * slot);
        let a = PtDotArgs {
            rot: PolyRef { base: rot.p(), ct_stride: (bstep * slot) as u64, part_stride: (8 * N) as u64 },
            res: PolyRef { base: res.p(), ct_stride: (gstep * slot) as u64, part_stride: (8 * N) as u64 },
            rot_stride: slot as u64,
            res_stride: slot as u64,
            pts: ptsd.c() as *const *const c_void,
            pr: st.pr(),
            rs: RS,
            npts: npts as i32,
            bstep: bstep as i32,
            gstep: gstep as i32,
            k: k as i32,
            ptw: ptw as i32,
            pt_prow: pt_prow as i32,
            enc,
            pad: 0,
        };
        ck(unsafe { (st.t.pt_dot)(&a, st.s()) });
        let mut want = vec![0u64; k * gstep * slot];
        for c in 0..k {
            for j in 0..gstep {
                for (y, &r) in rows.iter().enumerate() {
                    let p = st.p(r) as u128;
                    let prow = if y < 3 { y } else { pt_prow + y - 3 };
                    for part in 0..2 {
                        for i in 0..N {
                            let pi = if enc == 0xFFFF_FFFF { i } else { i >> 1 };
                            let mut s = 0u128;
                            for b in 0..bstep {
                                let d = bstep * j + b;
                                if d >= npts {
                                    break;
                                }
                                let x = roth[(c * bstep + b) * slot + part * 8 * N + r * N + i];
                                s += x as u128 * pth[d][prow * ptw + pi] as u128;
                            }
                            want[(c * gstep + j) * slot + part * 8 * N + r * N + i] = (s % p) as u64;
                        }
                    }
                }
            }
        }
        let got = res.read(&st.ctx);
        out.push(cmp(&format!("pt_dot ({label}) vs host"), &got, &want));
    }
    out
}

// ================================================================================================ EvalMod kernels
fn t_em(st: &mut Setup) -> Vec<TestResult> {
    let mut out = Vec::new();
    let (k, n) = (2usize, 4usize);
    let tot = 2 * k * 8 * N;
    let h: Vec<Vec<u64>> = (0..6).map(|_| st.rand_slots(2 * k)).collect();
    let d: Vec<Dev> = h.iter().map(|v| st.ctx.up(v)).collect();
    let pr = st.pr();
    // tensor (2 pairs, accumulate into out = d[4]), d2 and coef written
    {
        let o = st.ctx.up(&h[4]);
        let d2 = st.ctx.zeros(tot);
        let coef = st.ctx.zeros(k * 8 * N);
        let a = TensorArgs {
            out: ctref(&o),
            d2: ctref(&d2),
            a0: ctref(&d[0]),
            b0: ctref(&d[1]),
            a1: ctref(&d[2]),
            b1: ctref(&d[3]),
            coef: coef.p(),
            coef_ct_stride: (8 * N) as u64,
            pr,
            npairs: 2,
            acc: 1,
            n: n as i32,
            k: k as i32,
        };
        ck(unsafe { (st.t.tensor)(&a, st.s()) });
        let mut wo = h[4].clone();
        let mut wd2 = vec![0u64; tot];
        let mut wc = vec![0u64; k * 8 * N];
        for c in 0..k {
            for r in 0..n {
                let p = st.p(r) as u128;
                for i in 0..N {
                    let g = |m: usize, part: usize| h[m][idx(c, part, r, i)] as u128;
                    let s0 = g(0, 0) * g(1, 0) + g(2, 0) * g(3, 0);
                    let s1 = g(0, 0) * g(1, 1) + g(0, 1) * g(1, 0) + g(2, 0) * g(3, 1) + g(2, 1) * g(3, 0);
                    let s2 = g(0, 1) * g(1, 1) + g(2, 1) * g(3, 1);
                    wo[idx(c, 0, r, i)] = ((s0 + g(4, 0)) % p) as u64;
                    wo[idx(c, 1, r, i)] = ((s1 + g(4, 1)) % p) as u64;
                    wd2[idx(c, 1, r, i)] = (s2 % p) as u64;
                    wc[(c * 8 + r) * N + i] = (s2 % p) as u64;
                }
            }
        }
        let go = o.read(&st.ctx);
        out.push(cmp("tensor (2 pairs, acc) vs host", &go, &wo));
        out.push(cmp("tensor d2 vs host", &d2.read(&st.ctx), &wd2));
        out.push(cmp("tensor coef vs host", &coef.read(&st.ctx), &wc));
    }
    // md_epilogue (dbl, k), wsum (3 sources), add_const, mul_table
    {
        let binv = st.per_row(false);
        let binvs = st.shoup_row(&binv);
        let kc = st.per_row(true);
        let (bd, bsd, kd) = (st.ctx.up(&binv), st.ctx.up(&binvs), st.ctx.up(&kc));
        let o = st.ctx.up(&h[4]);
        let a = MdEpiArgs {
            dst: ctref(&o),
            x: ctref(&d[0]),
            t: ctref(&d[1]),
            binv: bd.c(),
            binvs: bsd.c(),
            k: kd.c(),
            pr,
            dbl: 1,
            n: n as i32,
            kk: k as i32,
            pad: 0,
        };
        ck(unsafe { (st.t.md_epilogue)(&a, st.s()) });
        let mut w = h[4].clone();
        for c in 0..k {
            for part in 0..2 {
                for r in 0..n {
                    let p = st.p(r);
                    for i in 0..N {
                        let j = idx(c, part, r, i);
                        let v = mulmod(submod(h[0][j], h[1][j], p), binv[r], p);
                        let mut x = addmod(h[4][j], v, p);
                        x = addmod(x, x, p);
                        if part == 0 {
                            x = addmod(x, kc[r] % p, p);
                        }
                        w[j] = x;
                    }
                }
            }
        }
        out.push(cmp("md_epilogue (dbl, k) vs host", &o.read(&st.ctx), &w));
        // md_epilogue_to (ABI 7): out = h5 +- (h0 - h1) binv
        for sub in [0, 1] {
            let o5 = st.ctx.zeros(tot);
            let at = MdEpiToArgs { m: MdEpiArgs { dst: ctref(&o5), ..a }, a: ctref(&d[5]), sub, pad: 0 };
            ck(unsafe { (st.t.md_epilogue_to)(&at, st.s()) });
            let mut w5 = vec![0u64; tot];
            for c in 0..k {
                for part in 0..2 {
                    for r in 0..n {
                        let p = st.p(r);
                        for i in 0..N {
                            let j = idx(c, part, r, i);
                            let v = mulmod(submod(h[0][j], h[1][j], p), binv[r], p);
                            w5[j] = if sub == 1 { submod(h[5][j], v, p) } else { addmod(h[5][j], v, p) };
                        }
                    }
                }
            }
            out.push(cmp(&format!("md_epilogue_to (sub {sub}) vs host"), &o5.read(&st.ctx), &w5));
        }

        let wv: Vec<u64> = (0..3 * 8).map(|x| st.rng.below(st.p(x % 8))).collect();
        let wsv = st.shoup_row(&wv);
        let (wd, wsd) = (st.ctx.up(&wv), st.ctx.up(&wsv));
        let o2 = st.ctx.zeros(tot);
        let mut src = [PolyRef::default(); 8];
        for j in 0..3 {
            src[j] = ctref(&d[j]);
        }
        let a = WsumArgs { out: ctref(&o2), src, w: wd.c(), ws: wsd.c(), pr, nsrc: 3, rowstride: 8, n: n as i32, k: k as i32 };
        ck(unsafe { (st.t.wsum)(&a, st.s()) });
        let mut w = vec![0u64; tot];
        for c in 0..k {
            for part in 0..2 {
                for r in 0..n {
                    let p = st.p(r);
                    for i in 0..N {
                        let j = idx(c, part, r, i);
                        w[j] = (0..3).fold(0, |s, q| addmod(s, mulmod(h[q][j], wv[q * 8 + r], p), p));
                    }
                }
            }
        }
        out.push(cmp("wsum (3 sources) vs host", &o2.read(&st.ctx), &w));
        // wsum_k (ABI 7) == wsum + add_const on part 0
        let o4 = st.ctx.zeros(tot);
        let ak = WsumKArgs { w: WsumArgs { out: ctref(&o4), ..a }, k: kd.c() };
        ck(unsafe { (st.t.wsum_k)(&ak, st.s()) });
        for c in 0..k {
            for r in 0..n {
                let p = st.p(r);
                for i in 0..N {
                    let j = idx(c, 0, r, i);
                    w[j] = addmod(w[j], kc[r] % p, p);
                }
            }
        }
        out.push(cmp("wsum_k (wsum + add_const) vs host", &o4.read(&st.ctx), &w));

        let o3 = st.ctx.up(&h[5]);
        let a = AddConstArgs { x: ctref(&o3), k: kd.c(), pr, n: n as i32, kk: k as i32 };
        ck(unsafe { (st.t.add_const)(&a, st.s()) });
        let mut w = h[5].clone();
        for c in 0..k {
            for r in 0..n {
                let p = st.p(r);
                for i in 0..N {
                    let j = idx(c, 0, r, i);
                    w[j] = addmod(h[5][j], kc[r] % p, p);
                }
            }
        }
        out.push(cmp("add_const vs host", &o3.read(&st.ctx), &w));

        let tb: Vec<u64> = (0..8 * N).map(|x| st.rng.below(st.p(x / N))).collect();
        let td = st.ctx.up(&tb);
        let o4 = st.ctx.zeros(tot);
        let a = MulTableArgs { out: ctref(&o4), inp: ctref(&d[0]), t: td.c(), pr, n: n as i32, k: k as i32 };
        ck(unsafe { (st.t.mul_table)(&a, st.s()) });
        let mut w = vec![0u64; tot];
        for c in 0..k {
            for part in 0..2 {
                for r in 0..n {
                    for i in 0..N {
                        let j = idx(c, part, r, i);
                        w[j] = mulmod(h[0][j], tb[r * N + i], st.p(r));
                    }
                }
            }
        }
        let g4 = o4.read(&st.ctx);
        out.push(cmp("mul_table vs host", &g4, &w));
    }
    out
}

/// General ModMul product over the strategy's whole prime range (mul_table on a dedicated prime table).
fn t_mul_range(st: &mut Setup) -> Vec<TestResult> {
    let bits: Vec<u32> = match (st.wbits, st.inst.mm()) {
        (16, _) => vec![10, 11, 12, 13, 14, 15, 15, 14],
        (32, 1) => vec![24, 26, 27, 28, 29, 30, 30, 29],
        (32, _) => vec![24, 27, 28, 29, 30, 31, 31, 30],
        _ => vec![30, 40, 50, 55, 59, 60, 61, 62],
    };
    let primes: Vec<u64> = bits.iter().enumerate().map(|(i, &b)| plain_primes(b, 1 + i % 2)[i % 2]).collect();
    let prt = st.ctx.up_wb(&prime_table(&primes, st.wbits), 8);
    let mut inp = vec![0u64; 8 * N];
    let mut tb = vec![0u64; 8 * N];
    for r in 0..8 {
        let p = primes[r];
        for i in 0..N {
            // edge values first, then uniform
            let e = [p - 1, p - 2, 1, 0, p / 2, p / 2 + 1];
            inp[r * N + i] = if i < 36 { e[i % 6] } else { st.rng.below(p) };
            tb[r * N + i] = if i < 36 { e[i / 6] } else { st.rng.below(p) };
        }
    }
    let (ind, td) = (st.ctx.up(&inp), st.ctx.up(&tb));
    let o = st.ctx.zeros(8 * N);
    let pref = |d: &Dev| PolyRef { base: d.p(), ct_stride: 0, part_stride: 0 };
    // n = 8 rows, K = 1 (both "parts" alias part 0: part_stride 0)
    let a = MulTableArgs { out: pref(&o), inp: pref(&ind), t: td.c(), pr: prt.c() as *const PrimeConst, n: 8, k: 1 };
    ck(unsafe { (st.t.mul_table)(&a, st.s()) });
    let want: Vec<u64> = (0..8 * N).map(|j| mulmod(inp[j], tb[j], primes[j / N])).collect();
    let mut r = cmp("ModMul over the prime range vs host", &o.read(&st.ctx), &want);
    r.note = format!("bits {bits:?}");
    vec![r]
}

// ================================================================================================ key switching
struct KsGeo {
    nq: usize,
    rows: Vec<u32>,
    digit_of: Vec<i32>,
    ndig: usize,
}
fn ks_geo() -> KsGeo {
    KsGeo { nq: 4, rows: vec![0, 1, 2, 3, 4, 5], digit_of: vec![0, 0, 1, 1, -1, -1, -1, -1], ndig: 2 }
}

#[allow(clippy::too_many_arguments)]
fn ks_host(st: &Setup, g: &KsGeo, c1: &[u64], mu: &[u64], c0: &[u64], kb: &[Vec<u64>], ka: &[Vec<u64>], autok: &[u32],
           pmod: &[u64], c0_mode: i32, k: usize, kds: usize) -> (Vec<u64>, Vec<u64>) {
    let r = kb.len();
    let (mut o0, mut o1) = (vec![0u64; k * r * 8 * N], vec![0u64; k * r * 8 * N]);
    for c in 0..k {
        for (ri, kr) in autok.iter().enumerate() {
            for &t in &g.rows {
                let t = t as usize;
                let p = st.p(t);
                let dt = if t < g.nq { g.digit_of[t] } else { -1 };
                for i in 0..N {
                    let (mut s0, mut s1) = (0u128, 0u128);
                    for d in 0..g.ndig {
                        let x = if d as i32 == dt { c1[c * 8 * N + t * N + i] } else { mu[(c * g.ndig + d) * 8 * N + t * N + i] };
                        s0 += x as u128 * kb[ri][d * kds + t * N + i] as u128;
                        s1 += x as u128 * ka[ri][d * kds + t * N + i] as u128;
                    }
                    let mut r0 = (s0 % p as u128) as u64;
                    let r1 = (s1 % p as u128) as u64;
                    let x0 = c0[c * 8 * N + t * N + i];
                    if c0_mode == 1 && t < g.nq {
                        r0 = addmod(r0, mulmod(x0, pmod[t], p), p);
                    } else if c0_mode == 2 {
                        r0 = addmod(r0, x0, p);
                    }
                    let o = if *kr == 1 { i } else { automorph_slot(*kr, i as u32) as usize };
                    let ob = (c * r + ri) * 8 * N + t * N + o;
                    o0[ob] = r0;
                    o1[ob] = r1;
                }
            }
        }
    }
    (o0, o1)
}

fn t_ks(st: &mut Setup) -> Vec<TestResult> {
    let mut out = Vec::new();
    let g = ks_geo();
    let (k, r) = (2usize, 2usize);
    let kds = 8 * N;
    let c1h = st.rand_slots(k);
    let c0h = st.rand_slots(k);
    let muh = st.rand_slots(k * g.ndig);
    let kbh: Vec<Vec<u64>> = (0..r).map(|_| st.rand_slots(g.ndig)).collect();
    let kah: Vec<Vec<u64>> = (0..r).map(|_| st.rand_slots(g.ndig)).collect();
    let autok = [1u32, 125];
    let pmod = st.per_row(false);
    let pmods = st.shoup_row(&pmod);
    let (c1, c0, mu) = (st.ctx.up(&c1h), st.ctx.up(&c0h), st.ctx.up(&muh));
    let kbd: Vec<Dev> = kbh.iter().map(|v| st.ctx.up(v)).collect();
    let kad: Vec<Dev> = kah.iter().map(|v| st.ctx.up(v)).collect();
    let kbp = st.ctx.up_ptrs(&kbd.iter().map(|d| d.c()).collect::<Vec<_>>());
    let kap = st.ctx.up_ptrs(&kad.iter().map(|d| d.c()).collect::<Vec<_>>());
    let (autod, rowsd, dod) = (st.ctx.up32(&autok), st.ctx.up32(&g.rows), st.ctx.up_i32(&g.digit_of));
    let (pmd, pmsd) = (st.ctx.up(&pmod), st.ctx.up(&pmods));
    for mode in [0, 1, 2] {
        let (o0, o1) = (st.ctx.zeros(k * r * 8 * N), st.ctx.zeros(k * r * 8 * N));
        let a = KsArgs {
            c1: c1.c(),
            modup: mu.c(),
            c0: c0.c(),
            out0: o0.p(),
            out1: o1.p(),
            kb: kbp.c() as *const *const c_void,
            ka: kap.c() as *const *const c_void,
            autok: autod.c() as *const u32,
            rows: rowsd.c() as *const u32,
            digit_of: dod.c() as *const i32,
            pmod: pmd.c(),
            pmods: pmsd.c(),
            pr: st.pr(),
            ct_stride: (8 * N) as u64,
            mu_ct_stride: (g.ndig * 8 * N) as u64,
            mu_digit_stride: (8 * N) as u64,
            out_ct_stride: (r * 8 * N) as u64,
            out_rot_stride: (8 * N) as u64,
            key_digit_stride: kds as u64,
            k: k as i32,
            r: r as i32,
            ndig: g.ndig as i32,
            c0_mode: mode,
            nq: g.nq as i32,
            nrows: g.rows.len() as i32,
        };
        ck(unsafe { (st.t.ks_dot)(&a, st.s()) });
        let (w0, w1) = ks_host(st, &g, &c1h, &muh, &c0h, &kbh, &kah, &autok, &pmod, mode, k, kds);
        let (g0, g1) = (o0.read(&st.ctx), o1.read(&st.ctx));
        out.push(cmp(&format!("ks_dot (K=2, R=2, c0 mode {mode}) out0 vs host"), &g0, &w0));
        out.push(cmp(&format!("ks_dot (K=2, R=2, c0 mode {mode}) out1 vs host"), &g1, &w1));
    }
    out
}

/// ksk_expand vs the host expander; ks_dot_regen vs ks_dot on the expanded `a`; the fused epilogue vs host.
fn t_regen(st: &mut Setup) -> Vec<TestResult> {
    let mut out = Vec::new();
    let wb = st.ctx.wb;
    let g = ks_geo();
    let k = 2usize;
    let nmod = 6usize;
    let kds = nmod * N; // ksk_expand layout [digit][m][N]
    let seed: [u32; 8] = std::array::from_fn(|_| st.rng.next() as u32);
    let seedd = st.ctx.up32(&seed);
    let ids: Vec<i32> = (0..nmod as i32).collect();
    let idsd = st.ctx.up_i32(&ids);
    let ka = st.ctx.zeros(g.ndig * kds);
    let ea = KskExpandArgs { out: ka.p(), seed: seedd.c() as *const u32, prime_ids: idsd.c() as *const i32, pr: st.pr(), nmod: nmod as i32, dnum: g.ndig as i32, d0: 0, pad: 0 };
    ck(unsafe { (st.t.ksk_expand)(&ea, st.s()) });
    let kah = ka.read(&st.ctx);
    let mut want = vec![0u64; g.ndig * kds];
    for d in 0..g.ndig {
        for m in 0..nmod {
            for s in 0..N {
                want[(d * nmod + m) * N + s] = expand_slot(&seed, s, d as u32, st.p(m), TAG_KSKA, wb);
            }
        }
    }
    out.push(cmp("ksk_expand vs host ChaCha12", &kah, &want));
    // ks_dot on the expanded key (already host-checked) is the reference of the regen kernel
    let c1h = st.rand_slots(k);
    let muh = st.rand_slots(k * g.ndig);
    let mut kbh = vec![0u64; g.ndig * kds];
    for (x, v) in kbh.iter_mut().enumerate() {
        *v = st.rng.below(st.p((x / N) % nmod));
    }
    let (c1, mu, kb) = (st.ctx.up(&c1h), st.ctx.up(&muh), st.ctx.up(&kbh));
    let kbp = st.ctx.up_ptrs(&[kb.c()]);
    let kap = st.ctx.up_ptrs(&[ka.c()]);
    let (autod, dod) = (st.ctx.up32(&[1]), st.ctx.up_i32(&g.digit_of));
    let rowsd = st.ctx.up32(&g.rows);
    let (o0, o1) = (st.ctx.zeros(k * 16 * N), st.ctx.zeros(k * 16 * N));
    let base = KsArgs {
        c1: c1.c(),
        modup: mu.c(),
        c0: std::ptr::null(),
        out0: o0.p(),
        out1: o1.p(),
        kb: kbp.c() as *const *const c_void,
        ka: kap.c() as *const *const c_void,
        autok: autod.c() as *const u32,
        rows: rowsd.c() as *const u32,
        digit_of: dod.c() as *const i32,
        pmod: std::ptr::null(),
        pmods: std::ptr::null(),
        pr: st.pr(),
        ct_stride: (8 * N) as u64,
        mu_ct_stride: (g.ndig * 8 * N) as u64,
        mu_digit_stride: (8 * N) as u64,
        out_ct_stride: (16 * N) as u64,
        out_rot_stride: 0,
        key_digit_stride: kds as u64,
        k: k as i32,
        r: 1,
        ndig: g.ndig as i32,
        c0_mode: 0,
        nq: g.nq as i32,
        nrows: g.rows.len() as i32,
    };
    ck(unsafe { (st.t.ks_dot)(&base, st.s()) });
    let (r0, r1) = (o0.read(&st.ctx), o1.read(&st.ctx));
    let (q0, q1) = (st.ctx.zeros(k * 16 * N), st.ctx.zeros(k * 16 * N));
    let mut ra = KsRegenArgs { ks: base, seed: seedd.c() as *const u32, epi: EpiArgs::default(), use_epi: 0, pad: 0, key: KeyTab::default(), add: std::ptr::null() };
    ra.ks.out0 = q0.p();
    ra.ks.out1 = q1.p();
    ra.ks.ka = std::ptr::null();
    ck(unsafe { (st.t.ks_dot_regen)(&ra, st.s()) });
    let (g0, g1) = (q0.read(&st.ctx), q1.read(&st.ctx));
    out.push(cmp("ks_dot_regen out0 == ks_dot(expanded a)", &g0, &r0));
    out.push(cmp("ks_dot_regen out1 == ks_dot(expanded a)", &g1, &r1));
    // fused epilogue on the Q rows: dst = 2 (dst + (dot - conv) binv) + k on part 0
    let dsth = st.rand_slots(2 * k);
    let convh = st.rand_slots(2 * k);
    let binv = st.per_row(false);
    let binvs = st.shoup_row(&binv);
    let kc = st.per_row(true);
    let (dst, conv, bd, bsd, kd) = (st.ctx.up(&dsth), st.ctx.up(&convh), st.ctx.up(&binv), st.ctx.up(&binvs), st.ctx.up(&kc));
    let qrows = st.ctx.up32(&[0, 1, 2, 3]);
    let mut re = ra;
    re.ks.rows = qrows.c() as *const u32;
    re.ks.nrows = 4;
    re.use_epi = 1;
    re.epi = EpiArgs { dst: dst.p(), conv: conv.c(), binv: bd.c(), binvs: bsd.c(), k: kd.c(), part_stride: (8 * N) as u64, dbl: 1, pad: 0 };
    ck(unsafe { (st.t.ks_dot_regen)(&re, st.s()) });
    let mut w = dsth.clone();
    for c in 0..k {
        for part in 0..2 {
            for t in 0..4 {
                let p = st.p(t);
                for i in 0..N {
                    let j = idx(c, part, t, i);
                    let x = if part == 0 { r0[c * 16 * N + t * N + i] } else { r1[c * 16 * N + t * N + i] };
                    let mut v = addmod(dsth[j], mulmod(submod(x, convh[j], p), binv[t], p), p);
                    v = addmod(v, v, p);
                    if part == 0 {
                        v = addmod(v, kc[t] % p, p);
                    }
                    w[j] = v;
                }
            }
        }
    }
    let ge = dst.read(&st.ctx);
    out.push(cmp("ks_dot_regen + relin epilogue vs host", &ge, &w));
    // ABI 7: + add buffer in the epilogue (both parts, after dbl / k)
    let addh = st.rand_slots(2 * k);
    let (dst3, addd) = (st.ctx.up(&dsth), st.ctx.up(&addh));
    let mut ra3 = re;
    ra3.epi.dst = dst3.p();
    ra3.add = addd.c();
    ck(unsafe { (st.t.ks_dot_regen)(&ra3, st.s()) });
    let mut w3 = w.clone();
    for c in 0..k {
        for part in 0..2 {
            for t in 0..4 {
                for i in 0..N {
                    let j = idx(c, part, t, i);
                    w3[j] = addmod(w[j], addh[j], st.p(t));
                }
            }
        }
    }
    out.push(cmp("ks_dot_regen + relin epilogue + add vs host", &dst3.read(&st.ctx), &w3));
    out
}

// ================================================================================================ CtS fused stage
fn t_cts(st: &mut Setup) -> Vec<TestResult> {
    let mut out = Vec::new();
    let g = ks_geo();
    let (k, b, gg, npts, pt_prow, kmax) = (3usize, 3usize, 2usize, 5usize, 4usize, 2i32);
    let slot = 16 * N;
    let xh = st.rand_slots(2 * k);
    let muh = st.rand_slots(k * g.ndig);
    let kbh: Vec<Vec<u64>> = (0..b - 1).map(|_| st.rand_slots(g.ndig)).collect();
    let kah: Vec<Vec<u64>> = (0..b - 1).map(|_| st.rand_slots(g.ndig)).collect();
    let kinv = [125u32, 2 * N as u32 - 5];
    let pth: Vec<Vec<u64>> = (0..npts)
        .map(|_| {
            (0..6 * N)
                .map(|x| {
                    let prow = x / N;
                    st.rng.below(st.p(if prow < 4 { prow } else { g.nq + prow - pt_prow }))
                })
                .collect()
        })
        .collect();
    let pmod = st.per_row(false);
    let pmods = st.shoup_row(&pmod);
    let (x, mu) = (st.ctx.up(&xh), st.ctx.up(&muh));
    let kbd: Vec<Dev> = kbh.iter().map(|v| st.ctx.up(v)).collect();
    let kad: Vec<Dev> = kah.iter().map(|v| st.ctx.up(v)).collect();
    let ptd: Vec<Dev> = pth.iter().map(|v| st.ctx.up(v)).collect();
    let kbp = st.ctx.up_ptrs(&kbd.iter().map(|d| d.c()).collect::<Vec<_>>());
    let kap = st.ctx.up_ptrs(&kad.iter().map(|d| d.c()).collect::<Vec<_>>());
    let ptp = st.ctx.up_ptrs(&ptd.iter().map(|d| d.c()).collect::<Vec<_>>());
    let (kinvd, rowsd, dod) = (st.ctx.up32(&kinv), st.ctx.up32(&g.rows), st.ctx.up_i32(&g.digit_of));
    let (pmd, pmsd) = (st.ctx.up(&pmod), st.ctx.up(&pmods));
    let res = st.ctx.zeros(k * gg * slot);
    let a = CtsArgs {
        x: x.c(),
        modup: mu.c(),
        res: res.p(),
        kb: kbp.c() as *const *const c_void,
        ka: kap.c() as *const *const c_void,
        kinv: kinvd.c() as *const u32,
        pts: ptp.c() as *const *const c_void,
        rows: rowsd.c() as *const u32,
        digit_of: dod.c() as *const i32,
        pmod: pmd.c(),
        pmods: pmsd.c(),
        pr: st.pr(),
        ct_stride: slot as u64,
        part_stride: (8 * N) as u64,
        mu_ct_stride: (g.ndig * 8 * N) as u64,
        mu_digit_stride: (8 * N) as u64,
        key_digit_stride: (8 * N) as u64,
        k: k as i32,
        b: b as i32,
        g: gg as i32,
        npts: npts as i32,
        ndig: g.ndig as i32,
        nq: g.nq as i32,
        nlive: 0,
        ptw: N as i32,
        pt_prow: pt_prow as i32,
        enc: 0xFFFF_FFFF,
        ngrp: 0,
        nrows: g.rows.len() as i32,
        kmax,
        pair_cts: 0,
    };
    ck(unsafe { (st.t.cts_fused)(&a, st.s()) });
    let got = res.read(&st.ctx);
    let mut want = vec![0u64; k * gg * slot];
    for c in 0..k {
        for &t in &g.rows {
            let t = t as usize;
            let p = st.p(t);
            let isq = t < g.nq;
            let prow = if isq { t } else { pt_prow + t - g.nq };
            let dt = if isq { g.digit_of[t] } else { -1 };
            let xa = |part: usize, s: usize| xh[c * slot + part * 8 * N + t * N + s];
            for o in 0..N {
                let mut acc = [[0u128; 2]; 2];
                for i in 0..b {
                    let (v0, v1) = if i == 0 {
                        if isq {
                            (mulmod(xa(0, o), pmod[t], p), mulmod(xa(1, o), pmod[t], p))
                        } else {
                            (0, 0)
                        }
                    } else {
                        let s = automorph_slot(kinv[i - 1], o as u32) as usize;
                        let (mut s0, mut s1) = (0u128, 0u128);
                        for d in 0..g.ndig {
                            let xv = if d as i32 == dt { xa(1, s) } else { muh[(c * g.ndig + d) * 8 * N + t * N + s] };
                            s0 += xv as u128 * kbh[i - 1][d * 8 * N + t * N + s] as u128;
                            s1 += xv as u128 * kah[i - 1][d * 8 * N + t * N + s] as u128;
                        }
                        let mut r0 = (s0 % p as u128) as u64;
                        if isq {
                            r0 = addmod(r0, mulmod(xa(0, s), pmod[t], p), p);
                        }
                        (r0, (s1 % p as u128) as u64)
                    };
                    for (j, accj) in acc.iter_mut().enumerate() {
                        let dg = b * j + i;
                        if dg < npts {
                            let w = pth[dg][prow * N + o] as u128;
                            accj[0] += v0 as u128 * w;
                            accj[1] += v1 as u128 * w;
                        }
                    }
                }
                for (j, accj) in acc.iter().enumerate() {
                    for part in 0..2 {
                        want[(c * gg + j) * slot + part * 8 * N + t * N + o] = (accj[part] % p as u128) as u64;
                    }
                }
            }
        }
    }
    out.push(cmp("cts_fused (K=3 in groups of 2, B=3, G=2) vs host", &got, &want));
    out
}

// ================================================================================================ seeded keys
/// Host packing of key rows (KskGenArgs bit order: value i at bits [i bits, (i+1) bits) of a LE u32 stream, a zero
/// guard word, rows rounded up to 4 words).
fn pack_rows(vals: &[u64], bits: usize) -> (Vec<u32>, usize) {
    let rw = (N * bits / 32 + 1).div_ceil(4) * 4;
    let nrows = vals.len() / N;
    let mut w = vec![0u32; nrows * rw];
    for r in 0..nrows {
        for i in 0..N {
            let v = vals[r * N + i];
            let bit = i * bits;
            let (wi, o) = (bit / 32, bit % 32);
            let x = (v as u128) << o;
            w[r * rw + wi] |= x as u32;
            w[r * rw + wi + 1] |= (x >> 32) as u32;
            if o + bits > 64 {
                w[r * rw + wi + 2] |= (x >> 64) as u32;
            }
        }
    }
    (w, rw)
}

/// ks_dot_key and cts_fused_regen (keys read through KeyTab: `a` from the seed, `b` dense / packed / banded) against
/// ks_dot / cts_fused on the expanded keys (both host-checked above): bit-identical.
fn t_keytab(st: &mut Setup) -> Vec<TestResult> {
    let mut out = Vec::new();
    let g = ks_geo();
    let nmod = 6usize; // full key: slots = prime ids 0..5 (Q 0..3, P 4..5)
    let kds = nmod * N;
    let r = 2usize;
    let seeds: Vec<[u32; 8]> = (0..r).map(|_| std::array::from_fn(|_| st.rng.next() as u32)).collect();
    let seedd: Vec<Dev> = seeds.iter().map(|sd| st.ctx.up32(sd)).collect();
    let ids: Vec<i32> = (0..nmod as i32).collect();
    let idsd = st.ctx.up_i32(&ids);
    let kad: Vec<Dev> = seedd
        .iter()
        .map(|sd| {
            let ka = st.ctx.zeros(g.ndig * kds);
            let ea = KskExpandArgs { out: ka.p(), seed: sd.c() as *const u32, prime_ids: idsd.c() as *const i32, pr: st.pr(), nmod: nmod as i32, dnum: g.ndig as i32, d0: 0, pad: 0 };
            ck(unsafe { (st.t.ksk_expand)(&ea, st.s()) });
            ka
        })
        .collect();
    let kbh: Vec<Vec<u64>> = (0..r)
        .map(|_| (0..g.ndig * kds).map(|x| st.rng.below(st.p((x / N) % nmod))).collect())
        .collect();
    let kbd: Vec<Dev> = kbh.iter().map(|v| st.ctx.up(v)).collect();
    let ptrs = |v: &[Dev]| v.iter().map(|d| d.c()).collect::<Vec<_>>();
    let (kbp, kap) = (st.ctx.up_ptrs(&ptrs(&kbd)), st.ctx.up_ptrs(&ptrs(&kad)));
    let seedp = st.ctx.up_ptrs(&ptrs(&seedd));
    // banded copy (q_top 2: slots 0..2 = Q ids 0..2, slots 3..4 = P ids 4..5), packed at u32
    let band_ids = [0usize, 1, 2, 4, 5];
    let packed = st.ctx.wb == 4;
    let bits = (0..nmod).map(|m| 64 - st.p(m).leading_zeros() as usize).max().unwrap_or(32);
    let band: Vec<(Dev, usize)> = kbh
        .iter()
        .map(|kb| {
            let rows: Vec<u64> = (0..g.ndig).flat_map(|d| band_ids.iter().flat_map(move |&t| kb[d * kds + t * N..d * kds + (t + 1) * N].iter().copied())).collect();
            if packed {
                let (w, rw) = pack_rows(&rows, bits);
                (st.ctx.up_raw32(&w), rw)
            } else {
                (st.ctx.up(&rows), N)
            }
        })
        .collect();
    let bandp = st.ctx.up_ptrs(&band.iter().map(|(d, _)| d.c()).collect::<Vec<_>>());
    let key_full = KeyTab { kb: kbp.c() as *const *const c_void, seed: seedp.c() as *const *const u32, row_words: N as u64, bits: 0, q_top: 3, srows: 6, nregen: 6 };
    let key_band = KeyTab {
        kb: bandp.c() as *const *const c_void,
        seed: seedp.c() as *const *const u32,
        row_words: band[0].1 as u64,
        bits: if packed { bits as i32 } else { 0 },
        q_top: 2,
        srows: 5,
        nregen: 6,
    };
    let band_name = if packed { format!("packed {bits}-bit + banded") } else { "banded".to_string() };
    let rows_full = st.ctx.up32(&g.rows);
    let rows_band = st.ctx.up32(&[0, 1, 2, 4, 5]);
    // ---- rotation dot: K=3 x R=2 (automorphisms 1 and 125), c0 modes 0..2
    let k = 3usize;
    let c1h = st.rand_slots(k);
    let c0h = st.rand_slots(k);
    let muh = st.rand_slots(k * g.ndig);
    let (c1, c0, mu) = (st.ctx.up(&c1h), st.ctx.up(&c0h), st.ctx.up(&muh));
    let pmod = st.per_row(false);
    let pmods = st.shoup_row(&pmod);
    let (pmd, pmsd) = (st.ctx.up(&pmod), st.ctx.up(&pmods));
    let (autod, dod) = (st.ctx.up32(&[1, 125]), st.ctx.up_i32(&g.digit_of));
    for (key, rowsd, nrows, name) in [(key_full, &rows_full, 6, "dense full".to_string()), (key_band, &rows_band, 5, band_name.clone())] {
        for mode in [0, 1, 2] {
            let (o0, o1) = (st.ctx.zeros(k * r * 8 * N), st.ctx.zeros(k * r * 8 * N));
            let mut a = KsArgs {
                c1: c1.c(),
                modup: mu.c(),
                c0: c0.c(),
                out0: o0.p(),
                out1: o1.p(),
                kb: kbp.c() as *const *const c_void,
                ka: kap.c() as *const *const c_void,
                autok: autod.c() as *const u32,
                rows: rowsd.c() as *const u32,
                digit_of: dod.c() as *const i32,
                pmod: pmd.c(),
                pmods: pmsd.c(),
                pr: st.pr(),
                ct_stride: (8 * N) as u64,
                mu_ct_stride: (g.ndig * 8 * N) as u64,
                mu_digit_stride: (8 * N) as u64,
                out_ct_stride: (r * 8 * N) as u64,
                out_rot_stride: (8 * N) as u64,
                key_digit_stride: kds as u64,
                k: k as i32,
                r: r as i32,
                ndig: g.ndig as i32,
                c0_mode: mode,
                nq: g.nq as i32,
                nrows,
            };
            ck(unsafe { (st.t.ks_dot)(&a, st.s()) });
            let mut want = o0.read(&st.ctx);
            want.extend(o1.read(&st.ctx));
            let (q0, q1) = (st.ctx.zeros(k * r * 8 * N), st.ctx.zeros(k * r * 8 * N));
            a.out0 = q0.p();
            a.out1 = q1.p();
            a.kb = std::ptr::null();
            a.ka = std::ptr::null();
            let ka = KsKeyArgs { ks: a, key };
            ck(unsafe { (st.t.ks_dot_key)(&ka, st.s()) });
            let mut got = q0.read(&st.ctx);
            got.extend(q1.read(&st.ctx));
            out.push(cmp(&format!("ks_dot_key ({name}, K=3, R=2, c0 mode {mode}) == ks_dot(expanded a)"), &got, &want));
            if key.bits == 0 && key.q_top == 3 {
                // hybrid: digit 0 regenerated, digit 1 read from the resident `a`
                let (h0, h1) = (st.ctx.zeros(k * r * 8 * N), st.ctx.zeros(k * r * 8 * N));
                let mut kh = ka;
                kh.ks.out0 = h0.p();
                kh.ks.out1 = h1.p();
                kh.ks.ka = kap.c() as *const *const c_void;
                kh.ks.key_digit_stride = kds as u64;
                kh.key.nregen = 1;
                ck(unsafe { (st.t.ks_dot_key)(&kh, st.s()) });
                let mut gh = h0.read(&st.ctx);
                gh.extend(h1.read(&st.ctx));
                out.push(cmp(&format!("ks_dot_key (hybrid: 1 of 2 digits regenerated, c0 mode {mode}) == ks_dot"), &gh, &want));
            }
        }
    }
    // ---- ABI 7: ks_dot_regen with key 0's `b` read through KeyTab (dense full / packed + banded) == ks_dot (R=1)
    {
        let k = 3usize;
        let one = st.ctx.up32(&[1]);
        for (key, rowsd, nrows, name) in [(key_full, &rows_full, 6, "dense full".to_string()), (key_band, &rows_band, 5, band_name.clone())] {
            let (o0, o1) = (st.ctx.zeros(k * 8 * N), st.ctx.zeros(k * 8 * N));
            let a = KsArgs {
                c1: c1.c(),
                modup: mu.c(),
                c0: std::ptr::null(),
                out0: o0.p(),
                out1: o1.p(),
                kb: kbp.c() as *const *const c_void,
                ka: kap.c() as *const *const c_void,
                autok: one.c() as *const u32,
                rows: rowsd.c() as *const u32,
                digit_of: dod.c() as *const i32,
                pmod: std::ptr::null(),
                pmods: std::ptr::null(),
                pr: st.pr(),
                ct_stride: (8 * N) as u64,
                mu_ct_stride: (g.ndig * 8 * N) as u64,
                mu_digit_stride: (8 * N) as u64,
                out_ct_stride: (8 * N) as u64,
                out_rot_stride: 0,
                key_digit_stride: kds as u64,
                k: k as i32,
                r: 1,
                ndig: g.ndig as i32,
                c0_mode: 0,
                nq: g.nq as i32,
                nrows,
            };
            ck(unsafe { (st.t.ks_dot)(&a, st.s()) });
            let mut want = o0.read(&st.ctx);
            want.extend(o1.read(&st.ctx));
            let (q0, q1) = (st.ctx.zeros(k * 8 * N), st.ctx.zeros(k * 8 * N));
            let mut ra = KsRegenArgs { ks: a, seed: seedd[0].c() as *const u32, epi: EpiArgs::default(), use_epi: 0, pad: 0, key, add: std::ptr::null() };
            ra.ks.out0 = q0.p();
            ra.ks.out1 = q1.p();
            ra.ks.kb = std::ptr::null();
            ra.ks.ka = std::ptr::null();
            ck(unsafe { (st.t.ks_dot_regen)(&ra, st.s()) });
            let mut got = q0.read(&st.ctx);
            got.extend(q1.read(&st.ctx));
            out.push(cmp(&format!("ks_dot_regen (b through KeyTab: {name}, K=3) == ks_dot(expanded a)"), &got, &want));
        }
    }
    // ---- fused CtS stage: K=3 in groups of 1 / 2 / 4, B=3, G=2
    let (k, b, gg, npts, pt_prow) = (3usize, 3usize, 2usize, 5usize, 4usize);
    let slot = 16 * N;
    let xh = st.rand_slots(2 * k);
    let muh = st.rand_slots(k * g.ndig);
    let kinv = [125u32, 2 * N as u32 - 5];
    let pth: Vec<Vec<u64>> = (0..npts)
        .map(|_| {
            (0..6 * N)
                .map(|x| {
                    let prow = x / N;
                    st.rng.below(st.p(if prow < 4 { prow } else { g.nq + prow - pt_prow }))
                })
                .collect()
        })
        .collect();
    let (x, mu) = (st.ctx.up(&xh), st.ctx.up(&muh));
    let ptd: Vec<Dev> = pth.iter().map(|v| st.ctx.up(v)).collect();
    let ptp = st.ctx.up_ptrs(&ptrs(&ptd));
    let kinvd = st.ctx.up32(&kinv);
    for (key, rowsd, nrows, name) in [(key_full, &rows_full, 6, "dense full".to_string()), (key_band, &rows_band, 5, band_name.clone())] {
        for kmax in [1, 2, 4] {
            let res = st.ctx.zeros(k * gg * slot);
            let mut a = CtsArgs {
                x: x.c(),
                modup: mu.c(),
                res: res.p(),
                kb: kbp.c() as *const *const c_void,
                ka: kap.c() as *const *const c_void,
                kinv: kinvd.c() as *const u32,
                pts: ptp.c() as *const *const c_void,
                rows: rowsd.c() as *const u32,
                digit_of: dod.c() as *const i32,
                pmod: pmd.c(),
                pmods: pmsd.c(),
                pr: st.pr(),
                ct_stride: slot as u64,
                part_stride: (8 * N) as u64,
                mu_ct_stride: (g.ndig * 8 * N) as u64,
                mu_digit_stride: (8 * N) as u64,
                key_digit_stride: kds as u64,
                k: k as i32,
                b: b as i32,
                g: gg as i32,
                npts: npts as i32,
                ndig: g.ndig as i32,
                nq: g.nq as i32,
                nlive: 0,
                ptw: N as i32,
                pt_prow: pt_prow as i32,
                enc: 0xFFFF_FFFF,
                ngrp: 0,
                nrows,
                kmax,
                pair_cts: 0,
            };
            ck(unsafe { (st.t.cts_fused)(&a, st.s()) });
            let want = res.read(&st.ctx);
            let res2 = st.ctx.zeros(k * gg * slot);
            a.res = res2.p();
            a.kb = std::ptr::null();
            a.ka = std::ptr::null();
            let ra = CtsRegenArgs { cts: a, key };
            ck(unsafe { (st.t.cts_fused_regen)(&ra, st.s()) });
            out.push(cmp(&format!("cts_fused_regen ({name}, K=3 groups of {kmax}, B=3, G=2) == cts_fused(expanded a)"), &res2.read(&st.ctx), &want));
            if key.bits == 0 && key.q_top == 3 {
                // hybrid: digit 0 regenerated, digit 1 read from the resident `a`
                let res3 = st.ctx.zeros(k * gg * slot);
                let mut rh = ra;
                rh.cts.res = res3.p();
                rh.cts.ka = kap.c() as *const *const c_void;
                rh.key.nregen = 1;
                ck(unsafe { (st.t.cts_fused_regen)(&rh, st.s()) });
                out.push(cmp(&format!("cts_fused_regen (hybrid: 1 of 2 digits regenerated, groups of {kmax}) == cts_fused"), &res3.read(&st.ctx), &want));
            }
        }
    }
    // ---- K=1, B=3, G=4 in one regen launch (pair layout) == two cached launches of 2 giant steps; the queued
    // pair-layout kernel exists at 32-bit words only
    if st.wbits == 32 {
        let (b4, g4, np4, pair) = (3usize, 4usize, 11usize, 5usize);
        let pth4: Vec<Vec<u64>> = (0..np4)
            .map(|_| {
                (0..6 * N)
                    .map(|x| {
                        let prow = x / N;
                        st.rng.below(st.p(if prow < 4 { prow } else { g.nq + prow - pt_prow }))
                    })
                    .collect()
            })
            .collect();
        let ptd4: Vec<Dev> = pth4.iter().map(|v| st.ctx.up(v)).collect();
        let ptp4 = st.ctx.up_ptrs(&ptrs(&ptd4));
        let res = st.ctx.zeros(2 * pair * slot);
        let base = CtsArgs {
            x: x.c(),
            modup: mu.c(),
            res: res.p(),
            kb: kbp.c() as *const *const c_void,
            ka: kap.c() as *const *const c_void,
            kinv: kinvd.c() as *const u32,
            pts: ptp4.c() as *const *const c_void,
            rows: rows_full.c() as *const u32,
            digit_of: dod.c() as *const i32,
            pmod: pmd.c(),
            pmods: pmsd.c(),
            pr: st.pr(),
            ct_stride: slot as u64,
            part_stride: (8 * N) as u64,
            mu_ct_stride: (g.ndig * 8 * N) as u64,
            mu_digit_stride: (8 * N) as u64,
            key_digit_stride: kds as u64,
            k: 1,
            b: b4 as i32,
            g: 2,
            npts: np4 as i32,
            ndig: g.ndig as i32,
            nq: g.nq as i32,
            nlive: 0,
            ptw: N as i32,
            pt_prow: pt_prow as i32,
            enc: 0xFFFF_FFFF,
            ngrp: 0,
            nrows: 6,
            kmax: 1,
            pair_cts: 0,
        };
        // cached reference: giant steps 0, 1 at region 0, giant steps 2, 3 at region `pair`
        ck(unsafe { (st.t.cts_fused)(&base, st.s()) });
        let mut a2 = base;
        a2.res = (res.p() as *mut u32).wrapping_add(pair * slot) as *mut c_void;
        a2.pts = (ptp4.c() as *const u8).wrapping_add(2 * b4 * 8) as *const *const c_void;
        a2.npts = (np4 - 2 * b4) as i32;
        ck(unsafe { (st.t.cts_fused)(&a2, st.s()) });
        let want = res.read(&st.ctx);
        let res2 = st.ctx.zeros(2 * pair * slot);
        let mut r4 = CtsRegenArgs { cts: base, key: key_full };
        r4.cts.res = res2.p();
        r4.cts.kb = std::ptr::null();
        r4.cts.ka = std::ptr::null();
        r4.cts.g = g4 as i32;
        r4.cts.pair_cts = pair as i32;
        ck(unsafe { (st.t.cts_fused_regen)(&r4, st.s()) });
        out.push(cmp("cts_fused_regen (K=1, B=3, G=4 in one launch, pair layout) == 2 cts_fused launches", &res2.read(&st.ctx), &want));
    }
    out
}

// ================================================================================================ rescale
fn centered(y: u64, q: u64) -> i128 {
    if y > q / 2 {
        y as i128 - q as i128
    } else {
        y as i128
    }
}
fn modi(x: i128, p: u64) -> u64 {
    x.rem_euclid(p as i128) as u64
}

fn t_rescale(st: &mut Setup) -> Vec<TestResult> {
    let mut out = Vec::new();
    let (k, l) = (2usize, 6usize);
    let top = l - 1;
    let has_ntt = st.ntt.is_some();
    for (ndrop, fold) in [(2usize, false), (2, true), (1, false), (1, true)] {
        let xh = st.rand_slots(2 * k);
        let kc: Vec<u64> = (0..8).map(|r| if fold { 1 + st.rng.below(st.p(r) - 1) } else { 1 }).collect();
        let nrows = l - ndrop;
        let (a, b) = (top, top - 1);
        let (qa, qb) = (st.p(a), st.p(b));
        // host: x * k, then sequential exact rescales by q_a (and q_b)
        let mut want = vec![0u64; 2 * k * 8 * N];
        for c in 0..k {
            for part in 0..2 {
                for i in 0..N {
                    let v: Vec<u64> = (0..l).map(|r| mulmod(xh[idx(c, part, r, i)], kc[r], st.p(r))).collect();
                    let ya = centered(v[a], qa);
                    let v1: Vec<u64> = (0..a).map(|r| mulmod(modi(v[r] as i128 - ya, st.p(r)), invmod(qa % st.p(r), st.p(r)), st.p(r))).collect();
                    let v2: Vec<u64> = if ndrop == 2 {
                        let yb = centered(v1[b], qb);
                        (0..b).map(|r| mulmod(modi(v1[r] as i128 - yb, st.p(r)), invmod(qb % st.p(r), st.p(r)), st.p(r))).collect()
                    } else {
                        v1
                    };
                    for r in 0..nrows {
                        want[idx(c, part, r, i)] = v2[r];
                    }
                }
            }
        }
        // device constants (OpenFHE form)
        let mut av = vec![0u64; 8];
        let mut bv = vec![0u64; 8];
        let mut al = vec![0u64; 8];
        for r in 0..nrows {
            let p = st.p(r);
            let qinv_a = invmod(qa % p, p);
            let qlql_a = (p - qinv_a) % p;
            if ndrop == 2 {
                let qinv_b = invmod(qb % p, p);
                av[r] = mulmod(qinv_b, qlql_a, p);
                bv[r] = (p - qinv_b) % p;
                al[r] = mulmod(mulmod(qinv_b, qinv_a, p), kc[r], p);
            } else {
                av[r] = qlql_a;
                al[r] = mulmod(qinv_a, kc[r], p);
            }
        }
        let qam: Vec<u64> = (0..8).map(|r| qa % st.p(r)).collect();
        let qbm: Vec<u64> = (0..8).map(|r| qb % st.p(r)).collect();
        let qinv_ab = invmod(qa % qb, qb);
        let c_ab = (qb - qinv_ab) % qb;
        let (avs, bvs, als) = (st.shoup_row(&av), st.shoup_row(&bv), st.shoup_row(&al));
        let ds: Vec<Dev> = [&av, &bv, &avs, &bvs, &qam, &qbm, &al, &als].iter().map(|v| st.ctx.up(v)).collect();
        let x = st.ctx.up(&xh);
        let tmp = st.ctx.zeros(2 * k * 8 * N);
        let ra = RescaleArgs {
            x: ctref(&x),
            tmp: ctref(&tmp),
            a: ds[0].c(),
            b: ds[1].c(),
            a_s: ds[2].c(),
            b_s: ds[3].c(),
            qam: ds[4].c(),
            qbm: ds[5].c(),
            pr: st.pr(),
            qinv_ab,
            c_ab,
            ca: kc[a],
            cb: kc[b],
            top: top as i32,
            nrows: nrows as i32,
            k: k as i32,
            ndrop: ndrop as i32,
        };
        let ca = RescaleCombineArgs { x: ctref(&x), tmp: ctref(&tmp), alpha: ds[6].c(), alphas: ds[7].c(), pr: st.pr(), n: nrows as i32, k: k as i32 };
        let mut all = Vec::new();
        let mut tops = Vec::new();
        let mut lows = Vec::new();
        for c in 0..k {
            for part in 0..2 {
                let s = ((c * 2 + part) * 8) as u32;
                for r in 0..l {
                    all.push((s + r as u32, r as i32));
                }
                tops.push((s + a as u32, a as i32));
                if ndrop == 2 {
                    tops.push((s + b as u32, b as i32));
                }
                for r in 0..nrows {
                    lows.push((s + r as u32, r as i32));
                }
            }
        }
        let label = format!("rescale (drop {ndrop}{}) vs host exact", if fold { ", folded scalar" } else { "" });
        if has_ntt {
            // NTT-domain pipeline as used: x in NTT form; INTT tops -> coef -> NTT tmp -> combine; INTT to compare
            let (ar, ap) = st.rows(&all);
            let (tr, tp) = st.rows(&tops);
            let (lr, lp) = st.rows(&lows);
            ck(unsafe { (st.t.ntt_fwd)(&st.ntt_args(x.p(), &ar, &ap, all.len()), st.s()) });
            ck(unsafe { (st.t.ntt_inv)(&st.ntt_args(x.p(), &tr, &tp, tops.len()), st.s()) });
            ck(unsafe { (st.t.rescale_coef)(&ra, st.s()) });
            ck(unsafe { (st.t.ntt_fwd)(&st.ntt_args(tmp.p(), &lr, &lp, lows.len()), st.s()) });
            // ABI 7: the out-of-place combine first (x unchanged), then the in-place one; equal in the NTT domain
            let o3 = st.ctx.zeros(2 * k * 8 * N);
            ck(unsafe { (st.t.rescale_combine_to)(&RescaleCombineToArgs { a: ca, out: ctref(&o3) }, st.s()) });
            ck(unsafe { (st.t.rescale_combine)(&ca, st.s()) });
            let mask = |v: &[u64]| -> Vec<u64> { lows.iter().flat_map(|&(o, _)| v[o as usize * N..(o as usize + 1) * N].to_vec()).collect() };
            out.push(cmp(&format!("rescale_combine_to == rescale_combine (drop {ndrop})"), &mask(&o3.read(&st.ctx)), &mask(&x.read(&st.ctx))));
            ck(unsafe { (st.t.ntt_inv)(&st.ntt_args(x.p(), &lr, &lp, lows.len()), st.s()) });
            let got = x.read(&st.ctx);
            out.push(cmp(&label, &mask(&got), &mask(&want)));
        } else {
            // no NTT primes at this word size: the same two kernels in the coefficient domain (both are linear maps)
            ck(unsafe { (st.t.rescale_coef)(&ra, st.s()) });
            ck(unsafe { (st.t.rescale_combine)(&ca, st.s()) });
            let got = x.read(&st.ctx);
            let mask = |v: &[u64]| -> Vec<u64> { lows.iter().flat_map(|&(o, _)| v[o as usize * N..(o as usize + 1) * N].to_vec()).collect() };
            out.push(cmp(&format!("{label} (coefficient domain)"), &mask(&got), &mask(&want)));
        }
    }
    out
}

// ================================================================================================ automorphism
fn t_automorph(st: &mut Setup) -> Vec<TestResult> {
    let mut out = Vec::new();
    let k = 2usize;
    let rows = rs_rows(RS);
    let ih = st.rand_slots(2 * k);
    let ah = st.rand_slots(2 * k);
    let (ind, add) = (st.ctx.up(&ih), st.ctx.up(&ah));
    for (kk, has_add) in [(powmod(5, 7, 2 * N as u64) as u32, false), (powmod(5, 7, 2 * N as u64) as u32, true), (2 * N as u32 - 1, false)] {
        let o = st.ctx.zeros(2 * k * 8 * N);
        let a = AutomorphArgs { out: ctref(&o), inp: ctref(&ind), add: ctref(&add), rs: RS, pr: st.pr(), k: kk, kk: k as i32, nparts: 2, has_add: has_add as i32 };
        ck(unsafe { (st.t.automorph)(&a, st.s()) });
        let mut want = vec![0u64; 2 * k * 8 * N];
        for c in 0..k {
            for part in 0..2 {
                for &r in &rows {
                    for i in 0..N {
                        let s = automorph_slot(kk, i as u32) as usize;
                        let v = ih[idx(c, part, r, s)];
                        want[idx(c, part, r, i)] = if has_add { addmod(v, ah[idx(c, part, r, i)], st.p(r)) } else { v };
                    }
                }
            }
        }
        out.push(cmp(&format!("automorph k={kk}{} vs host permutation", if has_add { " + add" } else { "" }), &o.read(&st.ctx), &want));
    }
    // the permutation is sigma_k: NTT(a(X^k))[o] == NTT(a)[automorph_slot(k, o)] (host, one row)
    if let Some(t) = &st.ntt {
        let p = st.p(0);
        let kk = powmod(5, 7, 2 * N as u64);
        let a = st.rng.vec(N, p);
        let mut b = poly_automorph(&a, kk, p);
        let mut an = a.clone();
        ntt_ref(&mut an, &t.w[0], p);
        ntt_ref(&mut b, &t.w[0], p);
        let g: Vec<u64> = (0..N).map(|o| an[automorph_slot(kk as u32, o as u32) as usize]).collect();
        out.push(cmp("automorph slot rule == sigma_k (host NTT of a(X^k))", &g, &b));
    }
    out
}

// ================================================================================================ encode
fn t_encode(st: &mut Setup) -> Vec<TestResult> {
    let mut out = Vec::new();
    let wb = st.ctx.wb;
    let nl = if wb == 8 { 2 } else { 3 };
    let scale = (1u64 << 40) as f64;
    let v: Vec<C64> = (0..N / 2).map(|_| C64 { re: 2.0 * st.rng.f64() - 1.0, im: 2.0 * st.rng.f64() - 1.0 }).collect();
    let vflat: Vec<f64> = v.iter().flat_map(|c| [c.re, c.im]).collect();
    let vd = st.ctx.up_f64(&vflat);
    let (sa, sb) = (st.ctx.zeros_wb(2 * N, 8), st.ctx.zeros_wb(2 * N, 8));
    let sc = st.ctx.zeros_wb(N, 8);
    let cm = st.ctx.zeros_wb(1, 8);
    let primes: Vec<u64> = st.primes[..nl].to_vec();
    for (label, df) in [("deg 1", vec![1u64; nl]), ("random deg factor", primes.iter().map(|&p| st.rng.below(p)).collect::<Vec<u64>>())] {
        let dfd = st.ctx.up(&df);
        let o = st.ctx.zeros(nl * N);
        let a = EncodeArgs {
            v: vd.c(),
            a: sa.p(),
            b: sb.p(),
            c: sc.p(),
            cmax: cm.p() as *mut u64,
            out: o.p(),
            deg_factor: dfd.c(),
            pr: st.pr(),
            scale,
            nl: nl as i32,
            pad: 0,
        };
        ck(unsafe { (st.t.encode)(&a, st.s()) });
        let dc = sc.read_f64(&st.ctx);
        let limbs = o.read(&st.ctx);
        let want: Vec<u64> = encode_limbs(&dc, &primes, &df).concat();
        out.push(cmp(&format!("encode scatter ({label}) vs host exact residues"), &limbs, &want));
        if label == "deg 1" {
            // the device FFT: its unrounded coefficients decode back to the slots within 1e-14
            let back = decode(&dc, scale);
            let errs: Vec<f64> = back.iter().zip(&v).map(|(a, b)| (a.re - b.re).abs().max((a.im - b.im).abs())).collect();
            let maxe = errs.iter().cloned().fold(0.0, f64::max);
            let bad = errs.iter().filter(|&&e| e > 1e-14).count() as u64;
            let hc = encode_coeffs(&v, scale);
            let dmax = dc.iter().zip(&hc).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            // the integer encoding (CRT of the first two limbs, or limb 0 for u64) decoded: CKKS precision, info only
            let ints: Vec<f64> = (0..N)
                .map(|kq| {
                    let x0 = limbs[kq];
                    if wb == 8 {
                        centered(x0, primes[0]) as f64
                    } else {
                        let (p0, p1) = (primes[0], primes[1]);
                        let x1 = limbs[N + kq];
                        let t = mulmod(submod(x1, x0 % p1, p1), invmod(p0 % p1, p1), p1);
                        let x = x0 as u128 + p0 as u128 * t as u128;
                        let m = p0 as u128 * p1 as u128;
                        (if x > m / 2 { x as i128 - m as i128 } else { x as i128 }) as f64
                    }
                })
                .collect();
            let bi = decode(&ints, scale);
            let ie = bi.iter().zip(&v).map(|(a, b)| (a.re - b.re).abs().max((a.im - b.im).abs())).fold(0.0, f64::max);
            out.push(TestResult {
                name: "encode FFT: decode(device coefficients) vs slots (<= 1e-14)".into(),
                bad,
                total: errs.len() as u64,
                note: format!("max err {maxe:.2e}; |dev c - host FFT c| max {dmax:.2e}; integer encode decode err {ie:.2e}"),
            });
        }
    }
    out
}

// ================================================================================================ encrypt
fn t_encrypt(st: &mut Setup) -> Vec<TestResult> {
    let mut out = Vec::new();
    let wb = st.ctx.wb;
    let nl = 3usize;
    let seed: [u32; 8] = std::array::from_fn(|_| st.rng.next() as u32);
    let seedd = st.ctx.up32(&seed);
    let lim = |st: &mut Setup| -> Vec<u64> { (0..nl * N).map(|x| st.rng.below(st.primes[x / N])).collect() };
    let (sh, eh, mh) = (lim(st), lim(st), lim(st));
    let (sd, ed, md) = (st.ctx.up(&sh), st.ctx.up(&eh), st.ctx.up(&mh));
    let nonce = 7u32;
    for with_m in [true, false] {
        let (c0, c1) = (st.ctx.zeros(nl * N), st.ctx.zeros(nl * N));
        let a = EncryptArgs {
            c0: c0.p(),
            c1: c1.p(),
            s: sd.c(),
            e: ed.c(),
            m: if with_m { md.c() } else { std::ptr::null() },
            seed: seedd.c() as *const u32,
            pr: st.pr(),
            nonce,
            nl: nl as i32,
        };
        ck(unsafe { (st.t.encrypt)(&a, st.s()) });
        let mut w0 = vec![0u64; nl * N];
        let mut w1 = vec![0u64; nl * N];
        for l in 0..nl {
            let p = st.p(l);
            for i in 0..N {
                let j = l * N + i;
                let av = expand_slot(&seed, i, nonce, p, TAG_ENCA, wb);
                w1[j] = av;
                let m = if with_m { mh[j] } else { 0 };
                w0[j] = submod(addmod(eh[j], m, p), mulmod(av, sh[j], p), p);
            }
        }
        let tag = if with_m { "" } else { ", m = 0" };
        out.push(cmp(&format!("encrypt c0 = -a s + e + m{tag} vs host"), &c0.read(&st.ctx), &w0));
        out.push(cmp(&format!("encrypt c1 = a (ENCA expander){tag} vs host"), &c1.read(&st.ctx), &w1));
    }
    out
}

// ================================================================================================ throughput
fn bench(st: &mut Setup) -> Vec<(usize, f32)> {
    let maxrows = 640;
    let big = st.ctx.zeros(maxrows * N);
    let mut res = Vec::new();
    for &nr in &[9usize, 54, 63, 108, 324, 640] {
        let list: Vec<(u32, i32)> = (0..nr).map(|i| (i as u32, (i % 8) as i32)).collect();
        let (ro, rp) = st.rows(&list);
        let a = st.ntt_args(big.p(), &ro, &rp, nr);
        let (e0, e1) = (Event::timing(st.ctx.dev).expect("event"), Event::timing(st.ctx.dev).expect("event"));
        let reps = 20;
        let run_new = |st: &Setup| -> f32 {
            for _ in 0..3 {
                ck(unsafe { (st.t.ntt_fwd)(&a, st.s()) });
            }
            e0.record(&st.ctx.s).expect("record");
            for _ in 0..reps {
                ck(unsafe { (st.t.ntt_fwd)(&a, st.s()) });
                ck(unsafe { (st.t.ntt_inv)(&a, st.s()) });
            }
            e1.record(&st.ctx.s).expect("record");
            e1.sync().expect("sync");
            e0.elapsed_ms(&e1).expect("elapsed") / (2 * reps) as f32
        };
        // best of 3 (shared box)
        let mut tn = f32::MAX;
        for _ in 0..3 {
            tn = tn.min(run_new(st));
        }
        res.push((nr, tn * 1e3));
    }
    res
}

// ================================================================================================ driver
fn run_instance(dev: &Device, inst: Instance, do_bench: bool) -> (u64, Vec<String>) {
    let t = match table(inst) {
        Ok(t) => t,
        Err(e) => panic!("{e}"),
    };
    let wb = inst.word_bytes() as usize;
    let wbits = 8 * wb as u32;
    let arena = Arena::new(dev, 3usize << 30).expect("arena");
    let ctx = Ctx { dev, arena: &arena, s: Stream::<Compute>::new(dev).expect("stream"), wb };
    let mut rng = Rng(0x1234_5678_9abc_def1 ^ (wb as u64 * 0x9E37_79B9) ^ inst.mm() as u64);
    let (primes, ntt_ok) = match wb {
        4 => (ntt_primes(28, 8), true),
        8 => (ntt_primes(60, 8), true),
        _ => (plain_primes(15, 8), false), // no p = 1 mod 2^17 below 2^16: no NTT at u16
    };
    let pr = ctx.up_wb(&prime_table(&primes, wbits), 8);
    let ntt = ntt_ok.then(|| {
        let mut fw = Vec::new();
        let mut fws = Vec::new();
        let mut iw = Vec::new();
        let mut iws = Vec::new();
        let mut ws = Vec::new();
        let mut ninv = Vec::new();
        for &p in &primes {
            let psi = psi_of(p, &mut rng);
            let (w, i) = twiddles(p, psi);
            fws.extend(w.iter().map(|&x| shoup(x, p, wbits)));
            iws.extend(i.iter().map(|&x| shoup(x, p, wbits)));
            fw.extend(&w);
            iw.extend(&i);
            ws.push(w);
            ninv.push(invmod(N as u64 % p, p));
        }
        let ninvs: Vec<u64> = ninv.iter().zip(&primes).map(|(&x, &p)| shoup(x, p, wbits)).collect();
        Ntt { fw: ctx.up(&fw), fws: ctx.up(&fws), iw: ctx.up(&iw), iws: ctx.up(&iws), ninv: ctx.up(&ninv), ninvs: ctx.up(&ninvs), w: ws }
    });
    let mut st = Setup { ctx, t, inst, wbits, primes, pr, ntt, rng };
    let name = inst.name();
    let has_ntt = st.ntt.is_some();
    type Test = fn(&mut Setup) -> Vec<TestResult>;
    let mut tests: Vec<Test> = Vec::new();
    if has_ntt {
        tests.push(t_ntt);
    }
    tests.extend([t_bconv as Test, t_ew, t_pt_dot, t_em, t_mul_range, t_ks, t_regen, t_cts, t_keytab, t_rescale, t_automorph]);
    if has_ntt {
        tests.push(t_encode);
    }
    tests.push(t_encrypt);
    let (mut total, mut nchecks) = (0, 0);
    for t in tests {
        for r in t(&mut st) {
            total += r.bad;
            nchecks += 1;
            println!(
                "[selftest] {name} | {:<62} | {:>9} mismatches of {:>9}{}",
                r.name,
                r.bad,
                r.total,
                if r.note.is_empty() { String::new() } else { format!(" | {}", r.note) }
            );
        }
    }
    println!("[selftest] {name}: {nchecks} checks, {total} mismatches");
    let mut lines = Vec::new();
    if do_bench && st.ntt.is_some() {
        for (nr, us) in bench(&mut st) {
            println!("[selftest] NTT {nr:4} rows: {us:.1} us per transform ({:.2} us/row)", us / nr as f32);
            lines.push(format!("| {name} | {nr} | {us:.1} | {:.2} |", us / nr as f32));
        }
    }
    (total, lines)
}

#[cfg(feature = "ring")]
fn run_ring(dev: &Device, do_bench: bool) -> u64 {
    let mut words = vec![RingWord::U32];
    if ring_table(RingWord::U64, 10).is_ok() {
        words.push(RingWord::U64);
    }
    let (mut bad, mut lines) = (0, Vec::new());
    for w in words {
        let (b, l) = ring::run(dev, w, do_bench);
        bad += b;
        lines.extend(l);
    }
    if !lines.is_empty() {
        println!("[selftest] ring NTT throughput (us per transform of the batch, fwd/inv average; 2^16 x 324 words per batch)");
        println!("| word | log n | rows | best us | best us per 2^16 words | generic us | generic us per 2^16 words |");
        println!("|---|---|---|---|---|---|---|");
        for l in &lines {
            println!("{l}");
        }
    }
    println!("[selftest] ring tables: total mismatches {bad}");
    bad
}
#[cfg(not(feature = "ring"))]
fn run_ring(_: &Device, _: bool) -> u64 {
    panic!("--ring needs pc-kernels built with feature `ring`");
}

/// Host-only: how often FIDESlib's Neal_mult_32 with its single conditional subtraction leaves a result >= p.
fn neal_single_guard_census() {
    let mut rng = Rng(0xfeed_beef);
    for bits in [27u32, 28, 29, 30] {
        let p = plain_primes(bits, 1)[0];
        let c = PrimeConst::new(p, 32);
        let mut bad = 0;
        let n = 1_000_000;
        for it in 0..n {
            // the largest products first, then uniform
            let (a, b) = if it < 1000 { ((p - 1 - it as u64 % 32) as u32, (p - 1 - it as u64 / 32) as u32) } else { (rng.below(p) as u32, rng.below(p) as u32) };
            let prod = a as u64 * b as u64;
            let rx = (prod >> (c.bits - 2)) as u32;
            let mut rb = (((rx << (30 - c.bits)) as u64 * (c.neal_mu as u32) as u64) >> 32) as u32 >> 1;
            rb = rb.wrapping_mul(p as u32);
            let mut lo = (prod as u32).wrapping_sub(rb);
            if lo >= p as u32 {
                lo -= p as u32;
            }
            if lo as u64 != prod % p {
                bad += 1;
            }
        }
        println!("[selftest] host census: Neal_mult_32 single guard, {bits}-bit prime: {bad} of {n} products unreduced");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let do_bench = !args.iter().any(|a| a == "--no-bench");
    let ring_only = args.iter().any(|a| a == "--ring-only");
    let do_ring = ring_only || args.iter().any(|a| a == "--ring");
    let insts: Vec<Instance> = if ring_only {
        Vec::new()
    } else if args.iter().any(|a| a == "--all") || args.iter().all(|a| a.starts_with("--")) {
        Instance::built()
    } else {
        args.iter()
            .filter(|a| !a.starts_with("--"))
            .map(|a| Instance::from_name(a).unwrap_or_else(|| panic!("unknown instance {a}")))
            .collect()
    };
    let dev = Device::open().expect("Device::open");
    neal_single_guard_census();
    let mut grand = 0;
    let mut table_lines = Vec::new();
    for inst in &insts {
        let (bad, lines) = run_instance(&dev, *inst, do_bench);
        grand += bad;
        table_lines.extend(lines);
    }
    if do_ring {
        grand += run_ring(&dev, do_bench);
    }
    if !table_lines.is_empty() {
        println!("[selftest] NTT throughput (us per transform, fwd/inv average)");
        println!("| instance | rows | us | us/row |");
        println!("|---|---|---|---|");
        for l in &table_lines {
            println!("{l}");
        }
    }
    println!("[selftest] instances {:?}: total mismatches {grand}", insts.iter().map(|i| i.name()).collect::<Vec<_>>());
    if grand != 0 {
        std::process::exit(1);
    }
}
