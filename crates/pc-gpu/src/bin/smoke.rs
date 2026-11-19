//! pc-gpu-smoke: the runtime skeleton gate. Builds the runtime for the n32 preset (MemPlan -> one arena -> Executor + Context +
//! CtPool -> freeze), then runs forward/inverse NTTs of an 8-ciphertext batch through the executor and compares with
//! a host reference NTT (OpenFHE ForwardTransformToBitReverse order); also checks partial spans (row tables), one
//! base conversion against the host formula, and that the frozen arena refuses to allocate.
//!
//! usage (GPU only through tools/gpu.sh): pc-gpu-smoke
use pc_core::modmul::{addmod, invmod, mulmod, submod, Barrett64};
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::{Compute, Device, Event};
use pc_gpu::bconv::{BconvPlan, JobSpec};
use pc_gpu::ntt::{self, Parts, Span};
use pc_gpu::{Context, CtPool, Executor, MemCfg, MemPlan, PoolClass};
use pc_shadow::CtMeta;
use rayon::prelude::*;
use std::time::Instant;

const K: usize = 8;

fn brev(i: usize, logn: u32) -> usize {
    (i as u32).reverse_bits() as usize >> (32 - logn)
}

/// OpenFHE ForwardTransformToBitReverse with root psi (w\[i\] = psi^brev(i)).
fn ntt_ref(a: &mut [u32], w: &[u64], p: u64) {
    let n = a.len();
    let (mut t, mut m) = (n, 1);
    while m < n {
        t /= 2;
        for i in 0..m {
            let s = w[m + i];
            for j in 2 * i * t..2 * i * t + t {
                let u = a[j] as u64;
                let v = mulmod(a[j + t] as u64, s, p);
                a[j] = addmod(u, v, p) as u32;
                a[j + t] = submod(u, v, p) as u32;
            }
        }
        m *= 2;
    }
}

fn host_twiddles(p: u64, psi: u64, n: usize) -> Vec<u64> {
    let logn = n.trailing_zeros();
    let mut pw = vec![1u64; n];
    for i in 1..n {
        pw[i] = mulmod(pw[i - 1], psi, p);
    }
    (0..n).map(|i| pw[brev(i, logn)]).collect()
}

fn xorshift(mut x: u64) -> impl FnMut() -> u64 {
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}

struct Check {
    name: String,
    bad: u64,
    total: u64,
}

fn compare(name: &str, got: &[u32], want: &[u32]) -> Check {
    let bad = got.par_iter().zip(want.par_iter()).filter(|(a, b)| a != b).count() as u64;
    let c = Check { name: name.into(), bad, total: want.len() as u64 };
    println!("[smoke] {:<58} {:>10} mismatches / {} words", c.name, c.bad, c.total);
    c
}

/// Host NTT of the listed (c, part, row) rows of a \[c\]\[part\]\[row\]\[N\] buffer, in place.
fn host_ntt_rows(buf: &mut [u32], rows_per_poly: usize, n: usize, sel: &(dyn Fn(usize, usize, usize) -> bool + Sync), tw: &[Vec<u64>], primes: &[u64]) {
    buf.par_chunks_mut(n).enumerate().for_each(|(limb, a)| {
        let row = limb % rows_per_poly;
        let part = (limb / rows_per_poly) % 2;
        let c = limb / (2 * rows_per_poly);
        if sel(c, part, row) {
            ntt_ref(a, &tw[row], primes[row]);
        }
    });
}

fn run() -> Result<u64, Box<dyn std::error::Error>> {
    let dev = Device::open()?;
    let (free, total) = dev.mem_info()?;
    println!("[smoke] cudart {} driver {}; device memory free {} / {} MB", dev.runtime_version(), dev.driver_version(), free >> 20, total >> 20);

    let t0 = Instant::now();
    let chain = Chain::generate(&ChainSpec::n32())?;
    let primes = chain.primes();
    let roots = chain.roots();
    let (nq, n) = (chain.q.len(), chain.n);
    let cfg = MemCfg { k_max: K, pool: vec![PoolClass { k: K, count: 1 }, PoolClass { k: 1, count: 2 }], events: 16, scratch_cts: 1 };
    let mut plan = MemPlan::build::<u32, Barrett64>(&chain, cfg)?;
    // the smoke's own bconv plan is a "later owner" section
    let specs = vec![
        JobSpec { src_off: 0, dst_off: 0, src: vec![(0, 0), (1, 1), (2, 2)], tgt: vec![(0, nq as u32), (1, nq as u32 + 1), (2, 7)], exact: false, prescaled: false },
        JobSpec { src_off: 0, dst_off: 3, src: vec![(3, 3), (4, 4), (5, 5), (6, 6)], tgt: vec![(0, nq as u32 + 2), (1, 0)], exact: true, prescaled: false },
    ];
    plan.reserve("smoke.bconv", BconvPlan::bytes::<u32>(&primes, &specs)?);
    println!("[smoke] MemPlan: {}", plan.report());
    let arena = plan.allocate(&dev)?;
    let mut ex = Executor::<u32>::new(&dev, &arena, &plan)?;
    let ctx = Context::<u32, Barrett64>::new(&arena, &ex.copy, chain.clone(), &plan)?;
    let pool = CtPool::<u32>::new(&arena, &plan)?;
    let bplan = BconvPlan::new::<_, u32>(&arena, &ex.compute, &primes, &specs)?;
    println!("[smoke] setup {:.2} s; arena used {} of {} B (plan exact: {})", t0.elapsed().as_secs_f64(), arena.used_bytes(), arena.capacity(), arena.used_bytes() <= arena.capacity());
    arena.freeze();

    let mut checks = Vec::new();
    // frozen arena: an allocation must panic
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| arena.alloc::<Compute, u32>("after freeze", 1).map(|_| ())));
    std::panic::set_hook(hook);
    let froze = r.is_err();
    println!("[smoke] alloc after Arena::freeze panics: {froze}");
    checks.push(Check { name: "freeze".into(), bad: (!froze) as u64, total: 1 });

    // 8 ciphertexts, every row random (< its prime)
    let mut ct = pool.acquire(K, CtMeta::new(0, 1))?;
    let ctm = ct.get_mut().ok_or("fresh pool handle is shared")?;
    let rows = ctx.lay.rows();
    let words = ctm.buf.buf.len();
    let mut x0 = vec![0u32; words];
    x0.par_chunks_mut(n).enumerate().for_each(|(limb, a)| {
        let p = primes[limb % rows];
        let mut r = xorshift(0x9E37_79B9_7F4A_7C15 ^ (limb as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
        for v in a.iter_mut() {
            *v = ((r() as u128 * p as u128) >> 64) as u32;
        }
    });
    let s = &ex.compute;
    ctm.buf.buf.upload(s, &x0)?;
    let tw: Vec<Vec<u64>> = primes.par_iter().zip(roots.par_iter()).map(|(&p, &r)| host_twiddles(p, r, n)).collect();

    // (1) full Q span, both parts, 8 x 2 x 54 rows (one launch)
    let top = nq - 1;
    ntt::forward(&ctx, s, &mut ctm.buf, Span::Q { top }, Parts::Both)?;
    let got = ctm.buf.buf.download(s)?;
    let mut want = x0.clone();
    host_ntt_rows(&mut want, rows, n, &|_, _, r| r <= top, &tw, &primes);
    checks.push(compare(&format!("ntt fwd Q[0..={top}] x2 parts x{K} cts vs host"), &got, &want));
    ntt::inverse(&ctx, s, &mut ctm.buf, Span::Q { top }, Parts::Both)?;
    let got = ctm.buf.buf.download(s)?;
    checks.push(compare(&format!("ntt inv(fwd(x)) == x, {K} cts x 2 x {nq} rows"), &got, &x0));

    // (2) partial span through the prefix tables: c1 only, Q[0..=40] u P
    let tp = 40;
    ntt::forward(&ctx, s, &mut ctm.buf, Span::QP { top: tp }, Parts::C1)?;
    let got = ctm.buf.buf.download(s)?;
    let mut want = x0.clone();
    host_ntt_rows(&mut want, rows, n, &|_, part, r| part == 1 && (r <= tp || r >= nq), &tw, &primes);
    checks.push(compare(&format!("ntt fwd c1 QP[0..={tp}]+P vs host"), &got, &want));
    ntt::inverse(&ctx, s, &mut ctm.buf, Span::QP { top: tp }, Parts::C1)?;
    checks.push(compare("ntt inv(fwd) c1 QP roundtrip", &ctm.buf.buf.download(s)?, &x0));

    // (3) one base conversion (plain + exact job) from ciphertext 0, part 0 rows into the bconv workspace
    let work = &mut ex.bconv_work;
    bplan.run(&ctx, s, &ctm.buf.buf, work)?;
    let got = work.download_range(s, 0, 5 * n)?;
    let mut want = vec![0u32; 5 * n];
    for j in &specs {
        let bp: Vec<u64> = j.src.iter().map(|&(_, id)| primes[id as usize]).collect();
        let ns = bp.len();
        let pre: Vec<u64> = (0..ns).map(|i| invmod((0..ns).filter(|&k| k != i).fold(1, |h, k| mulmod(h, bp[k] % bp[i], bp[i])), bp[i])).collect();
        for idx in 0..n {
            let y: Vec<u64> = (0..ns).map(|i| mulmod(x0[(j.src_off + j.src[i].0) as usize * n + idx] as u64, pre[i], bp[i])).collect();
            let u = if j.exact { ((0..ns).fold(0.0f64, |v, i| (y[i] as f64).mul_add(1.0 / bp[i] as f64, v)) + 0.5).floor() as u64 } else { 0 };
            for &(tr, tid) in &j.tgt {
                let t = primes[tid as usize];
                let mut acc = 0u64;
                for i in 0..ns {
                    let m = (0..ns).filter(|&k| k != i).fold(1u64, |v, k| mulmod(v, bp[k] % t, t));
                    acc = addmod(acc, mulmod(y[i], m, t), t);
                }
                if j.exact {
                    let bm = bp.iter().fold(1u64, |m, &b| mulmod(m, b % t, t));
                    acc = submod(acc, mulmod(u % t, bm, t), t);
                }
                want[(j.dst_off + tr) as usize * n + idx] = acc as u32;
            }
        }
    }
    checks.push(compare("bconv plain (3 -> 3) + exact (4 -> 2) vs host", &got, &want));

    // throughput (information): fwd+inv of the full batch
    let (e0, e1) = (Event::timing(&dev)?, Event::timing(&dev)?);
    let reps = 20;
    e0.record(s)?;
    for _ in 0..reps {
        ntt::forward(&ctx, s, &mut ctm.buf, Span::Q { top }, Parts::Both)?;
        ntt::inverse(&ctx, s, &mut ctm.buf, Span::Q { top }, Parts::Both)?;
    }
    e1.record(s)?;
    e1.sync()?;
    let us = e0.elapsed_ms(&e1)? as f64 * 1e3 / (2 * reps) as f64;
    let nrows = K * 2 * nq;
    println!("[smoke] NTT {nrows} rows: {us:.1} us per transform ({:.3} us/row)", us / nrows as f64);
    checks.push(compare("ntt after 20 fwd/inv pairs == x", &ctm.buf.buf.download(s)?, &x0));

    let bad: u64 = checks.iter().map(|c| c.bad).sum();
    println!("[smoke] {} checks, {} with mismatches -> {}", checks.len(), checks.iter().filter(|c| c.bad > 0).count(), if bad == 0 { "PASS" } else { "FAIL" });
    drop(ct);
    Ok(bad)
}

fn main() {
    match run() {
        Ok(0) => {}
        Ok(_) => std::process::exit(1),
        Err(e) => {
            eprintln!("[smoke] error: {e}");
            std::process::exit(2);
        }
    }
}
