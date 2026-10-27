//! Ring-size tables: NTT forward vs the host reference, inverse round trip, generic == tuned at 2^16, row kernels at
//! every built (word, log n), and the throughput sweep (generic vs tuned at 2^16).
use crate::dev::Ctx;
use crate::host::{brev, invmod, is_prime, mulmod, powmod, shoup, Rng};
use crate::{ck, cmp, prime_table, TestResult};
use pc_kernels::ring::RING_LOG_N;
use pc_kernels::*;
use pc_cuda::{Arena, Compute, Device, Event, Stream};

/// `count` primes p = 1 mod 2n just below 2^bits, largest first.
fn primes(bits: u32, n: usize, count: usize) -> Vec<u64> {
    let step = 2 * n as u64;
    let mut c = (1u64 << bits) + 1;
    let mut v = Vec::new();
    while v.len() < count {
        c -= step;
        if is_prime(c) {
            v.push(c);
        }
    }
    v
}

fn psi_of(p: u64, n: usize, rng: &mut Rng) -> u64 {
    loop {
        let g = 2 + rng.below(p - 3);
        let x = powmod(g, (p - 1) / (2 * n as u64), p);
        if powmod(x, n as u64, p) == p - 1 {
            return x;
        }
    }
}

fn twiddles(p: u64, psi: u64, logn: u32) -> (Vec<u64>, Vec<u64>) {
    let n = 1usize << logn;
    let ipsi = invmod(psi, p);
    let (mut pw, mut ipw) = (vec![1u64; n], vec![1u64; n]);
    for i in 1..n {
        pw[i] = mulmod(pw[i - 1], psi, p);
        ipw[i] = mulmod(ipw[i - 1], ipsi, p);
    }
    ((0..n).map(|i| pw[brev(i, logn)]).collect(), (0..n).map(|i| ipw[brev(i, logn)]).collect())
}

struct Tabs<'a> {
    fw: crate::dev::Dev<'a>,
    fws: crate::dev::Dev<'a>,
    iw: crate::dev::Dev<'a>,
    iws: crate::dev::Dev<'a>,
    ninv: crate::dev::Dev<'a>,
    ninvs: crate::dev::Dev<'a>,
    pr: crate::dev::Dev<'a>,
    w: Vec<Vec<u64>>,
}

impl Tabs<'_> {
    fn args(&self, base: *mut std::ffi::c_void, ro: &crate::dev::Dev, rp: &crate::dev::Dev, nrows: usize, chunk: i32) -> NttArgs {
        NttArgs {
            base,
            roff: ro.c() as *const u32,
            rprime: rp.c() as *const i32,
            fw: self.fw.c(),
            fws: self.fws.c(),
            iw: self.iw.c(),
            iws: self.iws.c(),
            ninv: self.ninv.c(),
            ninvs: self.ninvs.c(),
            pr: self.pr.c() as *const PrimeConst,
            nrows: nrows as i32,
            chunk,
            ..Default::default()
        }
    }
}

fn tabs<'a>(ctx: &Ctx<'a>, primes: &[u64], logn: u32, wbits: u32, rng: &mut Rng) -> Tabs<'a> {
    let n = 1usize << logn;
    let (mut fw, mut fws, mut iw, mut iws, mut ws, mut ninv) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    for &p in primes {
        let (w, i) = twiddles(p, psi_of(p, n, rng), logn);
        fws.extend(w.iter().map(|&x| shoup(x, p, wbits)));
        iws.extend(i.iter().map(|&x| shoup(x, p, wbits)));
        fw.extend(&w);
        iw.extend(&i);
        ws.push(w);
        ninv.push(invmod(n as u64 % p, p));
    }
    let ninvs: Vec<u64> = ninv.iter().zip(primes).map(|(&x, &p)| shoup(x, p, wbits)).collect();
    Tabs {
        fw: ctx.up(&fw),
        fws: ctx.up(&fws),
        iw: ctx.up(&iw),
        iws: ctx.up(&iws),
        ninv: ctx.up(&ninv),
        ninvs: ctx.up(&ninvs),
        pr: ctx.up_wb(&prime_table(primes, wbits), 8),
        w: ws,
    }
}

/// OpenFHE ForwardTransformToBitReverse of any power-of-two length
fn ntt_ref(a: &mut [u64], w: &[u64], p: u64) {
    let n = a.len();
    let (mut t, mut m) = (n, 1);
    while m < n {
        t /= 2;
        for i in 0..m {
            let s = w[m + i];
            for j in 2 * i * t..2 * i * t + t {
                let (u, v) = (a[j], mulmod(a[j + t], s, p));
                a[j] = if u + v >= p { u + v - p } else { u + v };
                a[j + t] = if u >= v { u - v } else { u + p - v };
            }
        }
        m *= 2;
    }
}

pub fn run(dev: &Device, word: RingWord, do_bench: bool) -> (u64, Vec<String>) {
    let wb = word.bytes() as usize;
    let wbits = 8 * wb as u32;
    let pbits = if wb == 4 { 31 } else { 62 };
    let arena = Arena::new(dev, 3usize << 30).expect("arena");
    let ctx = Ctx { dev, arena: &arena, s: Stream::<Compute>::new(dev).expect("stream"), wb };
    let mut rng = Rng(0x5eed_0f_a11_5172e ^ wb as u64);
    let (mut grand, mut lines) = (0u64, Vec::new());
    for logn in RING_LOG_N {
        let t = match ring_table(word, logn) {
            Ok(t) => t,
            Err(e) => panic!("{e}"),
        };
        let n = 1usize << logn;
        let ps = primes(pbits, n, 8);
        let tb = tabs(&ctx, &ps, logn, wbits, &mut rng);
        let name = format!("ring {} 2^{logn}", word.name());
        let mut res: Vec<TestResult> = Vec::new();
        // 12 rows in a 16-row buffer, permuted offsets, prime id = row % 8
        let nr = 12;
        let offs: Vec<u32> = (0..nr as u32).map(|i| (i * 5 + 3) % 16).collect();
        let pids: Vec<i32> = (0..nr as i32).map(|i| i % 8).collect();
        let ro = ctx.up32(&offs);
        let rp = ctx.up_i32(&pids);
        let mut host = vec![0u64; 16 * n];
        for (r, &o) in offs.iter().enumerate() {
            let p = ps[pids[r] as usize];
            for x in &mut host[o as usize * n..(o as usize + 1) * n] {
                *x = rng.below(p);
            }
        }
        // a few extreme rows: all p-1, a single 1
        {
            let p = ps[pids[0] as usize];
            host[offs[0] as usize * n..(offs[0] as usize + 1) * n].fill(p - 1);
            let o1 = offs[1] as usize * n;
            host[o1..o1 + n].fill(0);
            host[o1 + n / 2 + 1] = 1;
        }
        let mut want = host.clone();
        for (r, &o) in offs.iter().enumerate() {
            let pid = pids[r] as usize;
            ntt_ref(&mut want[o as usize * n..(o as usize + 1) * n], &tb.w[pid], ps[pid]);
        }
        for (label, fwd, inv) in [("best", t.ntt_fwd, t.ntt_inv), ("generic", t.ntt_fwd_g, t.ntt_inv_g)] {
            for chunk in [0, 5] {
                let mut buf = ctx.up(&host);
                let a = tb.args(buf.p(), &ro, &rp, nr, chunk);
                ck(unsafe { fwd(&a, ctx.s.raw()) });
                let got = buf.read(&ctx);
                res.push(cmp(&format!("ntt.fwd {label} chunk {chunk} vs host"), &got, &want));
                ck(unsafe { inv(&a, ctx.s.raw()) });
                res.push(cmp(&format!("ntt.inv {label}(fwd(x)) == x"), &buf.read(&ctx), &host));
                buf.write(&ctx, &host);
                drop(buf);
            }
        }
        // ew_add over 4 rows of the ring size (row length = 2^log n)
        {
            let k = 4;
            let x: Vec<u64> = (0..k * n).map(|i| rng.below(ps[i / n])).collect();
            let y: Vec<u64> = (0..k * n).map(|i| rng.below(ps[i / n])).collect();
            let (dx, dy, out) = (ctx.up(&x), ctx.up(&y), ctx.zeros(k * n));
            let a = EwArgs {
                out: PolyRef { base: out.p(), ct_stride: 0, part_stride: 0 },
                a: PolyRef { base: dx.p(), ct_stride: 0, part_stride: 0 },
                b: PolyRef { base: dy.p(), ct_stride: 0, part_stride: 0 },
                rs: RowSet { nq: k as i32, np: 0, qmax: 0, pad: 0 },
                pr: tb.pr.c() as *const PrimeConst,
                k: 1,
                nparts: 1,
                ..Default::default()
            };
            ck(unsafe { (t.ew_add)(&a, ctx.s.raw()) });
            let want: Vec<u64> = (0..k * n).map(|i| (x[i] + y[i]) % ps[i / n]).collect();
            res.push(cmp("ew_add rows of 2^log n", &out.read(&ctx), &want));
        }
        let bad: u64 = res.iter().map(|r| r.bad).sum();
        for r in &res {
            println!("[selftest] {name} | {:<44} | {:>9} mismatches of {:>9}", r.name, r.bad, r.total);
        }
        println!("[selftest] {name}: {} checks, {bad} mismatches", res.len());
        grand += bad;
        if do_bench {
            // the same words per transform batch at every size: 2^16 x 324 (a 108-row chain, 3 ciphertext parts)
            let rows = 324usize << (16 - logn.min(16)) >> logn.saturating_sub(16);
            let big = ctx.zeros(rows * n);
            let list: Vec<u32> = (0..rows as u32).collect();
            let pl: Vec<i32> = (0..rows as i32).map(|i| i % 8).collect();
            let (bro, brp) = (ctx.up32(&list), ctx.up_i32(&pl));
            let a = tb.args(big.p(), &bro, &brp, rows, if logn >= 16 { 128 } else { 0 });
            let (e0, e1) = (Event::timing(dev).expect("event"), Event::timing(dev).expect("event"));
            let reps = 20;
            let time = |f: Launch<NttArgs>, g: Launch<NttArgs>| -> f32 {
                let mut best = f32::MAX;
                for _ in 0..3 {
                    ck(unsafe { f(&a, ctx.s.raw()) });
                    e0.record(&ctx.s).expect("record");
                    for _ in 0..reps {
                        ck(unsafe { f(&a, ctx.s.raw()) });
                        ck(unsafe { g(&a, ctx.s.raw()) });
                    }
                    e1.record(&ctx.s).expect("record");
                    e1.sync().expect("sync");
                    best = best.min(e0.elapsed_ms(&e1).expect("elapsed") / (2 * reps) as f32 * 1e3);
                }
                best
            };
            let tbest = time(t.ntt_fwd, t.ntt_inv);
            let tgen = time(t.ntt_fwd_g, t.ntt_inv_g);
            let per16 = |us: f32| us / (rows * n) as f32 * 65536.0;
            println!(
                "[selftest] {name} bench {rows} rows: best {tbest:.1} us ({:.3} us per 2^16 words), generic {tgen:.1} us ({:.3})",
                per16(tbest),
                per16(tgen)
            );
            lines.push(format!("| {} | {logn} | {rows} | {tbest:.1} | {:.3} | {tgen:.1} | {:.3} |", word.name(), per16(tbest), per16(tgen)));
        }
    }
    (grand, lines)
}
