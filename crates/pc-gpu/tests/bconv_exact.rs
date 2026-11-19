//! Exact (HPS) base conversions on u32 words, GPU (run through tools/gpu.sh, `--ignored --test-threads=1`): every
//! exact conversion a key switch makes on the `LowP` tiers (ModUp of the live digit, ModDown from P') and in an
//! alternating user region (ModDown from P), 8 ciphertexts at each such limb top, must equal the FMA double estimate's
//! conversion word for word (the host formula below, source order). Inputs: uniform rows, and coefficients whose
//! value is within a few units of 0 or of B / 2 (B the sources' product), where the integer estimate hands the
//! rounding to the double one. Prints the device time per conversion (uniform rows). With PERSEUS_DEVICE=cpu the same
//! check runs the host kernels (the timings then from a few runs).
use pc_core::modmul::{mulmod, submod, Barrett64};
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::{Compute, Device, Event};
use pc_gpu::bconv::{pre_factors, BconvPlan, JobSpec};
use pc_gpu::{Context, Executor, MemCfg, MemPlan, PoolClass};
use rayon::prelude::*;

type W = u32;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const K: usize = 8;

/// The top limb of an alternating user region's top level (as `ks.rs`), None without one.
fn sp_user_top(chain: &Chain) -> Option<usize> {
    chain.spec.sp?;
    let mut top = chain.q.len();
    for &g in chain.groups.iter().rev() {
        if g == 1 {
            return Some(top - 1 + chain.level_unit);
        }
        top -= g;
    }
    None
}

/// A set of jobs over one source buffer and one target buffer (`ks.rs` level layout, K ciphertexts).
struct Case {
    name: String,
    specs: Vec<JobSpec>,
    src_words: usize,
    dst_words: usize,
}

/// The exact conversions of the key switches of `chain` at limb top `top` (`ks.rs` `level_specs_with`).
fn cases_at(chain: &Chain, preset: &str, top: usize) -> Vec<Case> {
    let n = chain.n;
    let rows = chain.q.len() + chain.p.len();
    let mut specials = chain.special_ids();
    specials.truncate(chain.specials_at(top));
    let low = chain.band_specials(top).is_some();
    let dnum = chain.dnum();
    let mut out = Vec::new();
    if low {
        let digs: Vec<Vec<usize>> = chain.digits.iter().map(|d| d.iter().copied().filter(|&t| t <= top).collect::<Vec<_>>()).filter(|d| !d.is_empty()).collect();
        let mut specs = Vec::new();
        for c in 0..K {
            for (d, dl) in digs.iter().enumerate() {
                let tg: Vec<usize> = (0..=top).filter(|t| !dl.contains(t)).chain(specials.iter().copied()).collect();
                specs.push(JobSpec {
                    src_off: (c * rows) as u32,
                    dst_off: ((c * dnum + d) * rows) as u32,
                    src: dl.iter().map(|&t| (t as u32, t as u32)).collect(),
                    tgt: tg.iter().map(|&t| (t as u32, t as u32)).collect(),
                    exact: true,
                    prescaled: true,
                });
            }
        }
        out.push(Case { name: format!("{preset} top {top} ModUp {:?} -> {}", digs.iter().map(|d| d.len()).collect::<Vec<_>>(), top + 1 + specials.len() - digs[0].len()), specs, src_words: K * rows * n, dst_words: K * dnum * rows * n });
    }
    if low || sp_user_top(chain).is_some_and(|u| top <= u) {
        let mut specs = Vec::new();
        for c in 0..K {
            for part in 0..2 {
                let base = ((2 * c + part) * rows) as u32;
                specs.push(JobSpec {
                    src_off: base,
                    dst_off: base,
                    src: specials.iter().map(|&t| (t as u32, t as u32)).collect(),
                    tgt: (0..=top).map(|t| (t as u32, t as u32)).collect(),
                    exact: true,
                    prescaled: true,
                });
            }
        }
        out.push(Case { name: format!("{preset} top {top} ModDown {} -> {}", specials.len(), top + 1), specs, src_words: 2 * K * rows * n, dst_words: 2 * K * rows * n });
    }
    out
}

fn xorshift(mut x: u64) -> impl FnMut() -> u64 {
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}

/// Source rows of every job (prescaled: the y_i): coefficient idx % 4 == 0 holds x = (B - 1) / 2 + d, == 1 holds
/// x = d (d in -32..32, y_i = x pre_i mod b_i), the others uniform (all of them with `uniform`).
fn fill(case: &Case, primes: &[u64], n: usize, seed: u64, uniform: bool) -> Vec<W> {
    let mut src = vec![0 as W; case.src_words];
    for (jn, j) in case.specs.iter().enumerate() {
        let bp: Vec<u64> = j.src.iter().map(|&(_, p)| primes[p as usize]).collect();
        let pre = pre_factors(&bp);
        for (i, &(r, _)) in j.src.iter().enumerate() {
            let b = bp[i];
            let row = &mut src[(j.src_off + r) as usize * n..][..n];
            let mut g = xorshift(seed ^ ((jn * 64 + i) as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            for (idx, w) in row.iter_mut().enumerate() {
                let d = (idx / 4 % 64) as i64 - 32;
                let x = match if uniform { 2 } else { idx % 4 } {
                    0 => ((b - 1) / 2) as i64 + d,
                    1 => d,
                    _ => {
                        *w = ((g() as u128 * b as u128) >> 64) as W;
                        continue;
                    }
                };
                *w = mulmod(x.rem_euclid(b as i64) as u64, pre[i], b) as W;
            }
        }
    }
    src
}

/// The FMA double estimate's conversion of every job (source order), into a copy of `dst`; also the count of
/// coefficients whose integer estimate falls back to the double one (`bconv.cuh` `int_round`).
fn reference(case: &Case, primes: &[u64], n: usize, src: &[W], dst: &[W]) -> (Vec<W>, usize) {
    let mut want = dst.to_vec();
    let mut amb = 0;
    for j in &case.specs {
        let bp: Vec<u64> = j.src.iter().map(|&(_, p)| primes[p as usize]).collect();
        let ns = bp.len();
        let tp: Vec<u64> = j.tgt.iter().map(|&(_, p)| primes[p as usize]).collect();
        let m: Vec<Vec<u64>> = (0..ns).map(|i| tp.iter().map(|&t| (0..ns).filter(|&k| k != i).fold(1u64, |v, k| mulmod(v, bp[k] % t, t))).collect()).collect();
        let bm: Vec<u64> = tp.iter().map(|&t| bp.iter().fold(1u64, |v, &b| mulmod(v, b % t, t))).collect();
        let rc: Vec<u64> = bp.iter().map(|&b| (1u64 << 59) / b).collect();
        let rows: Vec<Vec<u64>> = (0..j.tgt.len())
            .into_par_iter()
            .map(|k| {
                let t = tp[k];
                (0..n)
                    .map(|idx| {
                        let y: Vec<u64> = j.src.iter().map(|&(r, _)| src[(j.src_off + r) as usize * n + idx] as u64).collect();
                        let v = (0..ns).fold(0.0f64, |v, i| (y[i] as f64).mul_add(1.0 / bp[i] as f64, v));
                        let u = (v + 0.5).floor() as u64;
                        let acc = (0..ns).map(|i| y[i] as u128 * m[i][k] as u128).sum::<u128>() % t as u128;
                        submod(acc as u64, mulmod(u % t, bm[k], t), t)
                    })
                    .collect()
            })
            .collect();
        for (k, &(r, _)) in j.tgt.iter().enumerate() {
            for (idx, &v) in rows[k].iter().enumerate() {
                want[(j.dst_off + r) as usize * n + idx] = v as W;
            }
        }
        for idx in 0..n {
            let (mut f, mut e) = (1u64 << 58, 1u64);
            for (i, &(r, _)) in j.src.iter().enumerate() {
                let y = src[(j.src_off + r) as usize * n + idx] as u64;
                f += y * rc[i];
                e += y;
            }
            let r = f & ((1 << 59) - 1);
            amb += (r < 1 << 19 || r + e + (1 << 19) >= 1 << 59) as usize;
        }
    }
    (want, amb)
}

#[test]
#[ignore = "GPU: tools/gpu.sh <log> <bconv_exact test binary> --ignored --test-threads=1"]
fn exact_conversions_match_the_double_estimate() -> Res<()> {
    let dev = Device::open()?;
    let reps = if std::env::var("PERSEUS_DEVICE").as_deref() == Ok("cpu") { 2 } else { 200 };
    let mut bad_total = 0;
    for preset in ["n32/lp4b3/lp2b1", "n32/lp4b3", "n32a4/lp4b3/lp2b1", "n32sp", "n32sph"] {
        let chain = Chain::generate(&ChainSpec::preset(preset)?)?;
        let (n, primes) = (chain.n, chain.primes());
        let cases: Vec<Case> = (0..chain.q.len()).flat_map(|t| cases_at(&chain, preset, t)).collect();
        assert!(!cases.is_empty(), "{preset}: no exact conversions");
        let mut plan = MemPlan::build::<W, Barrett64>(&chain, MemCfg { k_max: 1, pool: vec![PoolClass { k: 1, count: 1 }], events: 4, scratch_cts: 1 })?;
        let (ms, md) = cases.iter().fold((0, 0), |(a, b), c| (a.max(c.src_words), b.max(c.dst_words)));
        plan.reserve("t.src", ms * 4 + 4096);
        plan.reserve("t.dst", md * 4 + 4096);
        // the same jobs without the correction (plain), for the timing's reference
        let plain = |c: &Case| c.specs.iter().map(|j| JobSpec { exact: false, ..j.clone() }).collect::<Vec<_>>();
        let tables: usize = cases.iter().map(|c| Ok(BconvPlan::bytes::<W>(&primes, &c.specs)? + BconvPlan::bytes::<W>(&primes, &plain(c))?)).sum::<pc_gpu::Result<usize>>()?;
        plan.reserve("t.plans", tables + 64 * 1024);
        let arena = plan.allocate(&dev)?;
        let ex = Executor::<W>::new(&dev, &arena, &plan)?;
        let ctx = Context::<W, Barrett64>::new(&arena, &ex.copy, chain.clone(), &plan)?;
        let mut src = arena.alloc::<Compute, W>("t.src", ms)?;
        let mut dst = arena.alloc::<Compute, W>("t.dst", md)?;
        let plans = cases.iter().map(|c| BconvPlan::new::<_, W>(&arena, &ex.compute, &primes, &c.specs)).collect::<pc_gpu::Result<Vec<_>>>()?;
        let plans0 = cases.iter().map(|c| BconvPlan::new::<_, W>(&arena, &ex.compute, &primes, &plain(c))).collect::<pc_gpu::Result<Vec<_>>>()?;
        let s = &ex.compute;
        for (ci, ((case, bplan), bplan0)) in cases.iter().zip(&plans).zip(&plans0).enumerate() {
            let x = fill(case, &primes, n, 0x5EED + ci as u64, false);
            let mut d0 = vec![0 as W; case.dst_words];
            let mut g = xorshift(0xD57 + ci as u64);
            d0.iter_mut().for_each(|w| *w = g() as W);
            src.upload_at(s, 0, &x)?;
            dst.upload_at(s, 0, &d0)?;
            bplan.run(&ctx, s, &src, &mut dst)?;
            let got = dst.download_range(s, 0, case.dst_words)?;
            let (want, amb) = reference(case, &primes, n, &x, &d0);
            let bad = got.par_iter().zip(want.par_iter()).filter(|(a, b)| a != b).count();
            // device time per conversion of the 8 ciphertexts on uniform rows (median of 5 batches of `reps`), exact and plain
            src.upload_at(s, 0, &fill(case, &primes, n, 0xA11 + ci as u64, true))?;
            let mut us = [0.0; 2];
            for (u, p) in us.iter_mut().zip([bplan, bplan0]) {
                let mut ts = Vec::new();
                for _ in 0..5 {
                    let (a, b) = (Event::timing(&dev)?, Event::timing(&dev)?);
                    a.record(s)?;
                    for _ in 0..reps {
                        p.run(&ctx, s, &src, &mut dst)?;
                    }
                    b.record(s)?;
                    b.sync()?;
                    ts.push(a.elapsed_ms(&b)? as f64 * 1e3 / reps as f64);
                }
                ts.sort_by(|a, b| a.total_cmp(b));
                *u = ts[2];
            }
            println!("[bconv_exact] {:<40} {:>2} jobs: {bad} mismatches / {} words, {amb} on the double estimate; {:.2} us (plain {:.2})", case.name, case.specs.len(), want.len(), us[0], us[1]);
            bad_total += bad;
        }
    }
    assert_eq!(bad_total, 0, "exact conversions differ from the double estimate");
    Ok(())
}
