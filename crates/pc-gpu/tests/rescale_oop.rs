//! The composite rescale's out-of-place path, GPU (run through tools/gpu.sh, `--ignored --test-threads=1`): the
//! out-of-place realize (`Ops::realize_to`: the tops' INTT reads the source, `rescale_fwd` combines from it) equals a
//! copy and the in-place realize word for word, at three levels, K = 1 and 3.
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::Device;
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::{Context, Ct, CtPool, Executor, MemCfg, MemPlan, PoolClass};
use pc_shadow::{BtsConfig, CtMeta, Quirks, Shadow, ShadowConfig};
use std::sync::Arc;

type W = u32;
type M = Barrett64;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Uniform rows `[unit][prime id][N]` below each prime.
fn uniform(units: usize, primes: &[u64], n: usize, seed: u64) -> Vec<W> {
    let rows = primes.len();
    let mut x = seed | 1;
    let mut v = vec![0 as W; units * rows * n];
    for (i, w) in v.iter_mut().enumerate() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *w = ((x as u128 * primes[(i / n) % rows] as u128) >> 64) as W;
    }
    v
}

/// Q rows 0..=top of both parts of every ciphertext of `x`.
fn rows(ex: &Executor<'_, W>, x: &Ct<'_, W>, top: usize) -> Res<Vec<W>> {
    let lay = x.buf.lay;
    let mut out = Vec::new();
    for c in 0..x.k() {
        for part in 0..2 {
            out.extend(x.buf.buf.download_range(&ex.compute, c * lay.ct_words() + part * lay.poly_words(), (top + 1) * lay.n)?);
        }
    }
    Ok(out)
}

#[test]
#[ignore = "GPU: tools/gpu.sh <log> <rescale_oop test binary> --ignored --test-threads=1"]
fn realize_to_equals_copy_and_realize() -> Res<()> {
    let dev = Device::open()?;
    let chain = Chain::generate(&ChainSpec::n32())?;
    let (n, nq, primes) = (chain.n, chain.q.len(), chain.primes());
    let cfg = MemCfg { k_max: 4, pool: vec![PoolClass { k: 1, count: 3 }, PoolClass { k: 3, count: 3 }], events: 16, scratch_cts: 2 };
    let mut plan = MemPlan::build::<W, M>(&chain, cfg)?;
    let ocfg = OpsCfg::default();
    assert!(ocfg.out_of_place);
    Ops::<W>::reserve(&mut plan, &ocfg);
    let arena = plan.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &plan)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &plan)?;
    let pool = CtPool::<W>::new(&arena, &plan)?;
    let mut ops = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    ex.compute.sync()?;
    arena.freeze();
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let mut bad = 0;
    for &level in &[20u32, 34, 44] {
        let top = nq - 1 - level as usize;
        for &k in &[1usize, 3] {
            let meta = CtMeta { level, deg: 2, nf: ctx.scale.sf(top as i32)?.powi(2), slots: (n / 2) as u32, tag: Default::default() };
            let mut x = pool.acquire(k, meta)?;
            let mut y = pool.acquire(k, meta)?;
            let mut out = pool.acquire(k, meta)?;
            let v = uniform(2 * k, &primes, n, 0x5ca1e + 31 * level as u64 + k as u64);
            {
                let (xm, ym) = (x.get_mut().ok_or("shared")?, y.get_mut().ok_or("shared")?);
                xm.buf.buf.upload_at(&ex.compute, 0, &v)?;
                ym.buf.buf.upload_at(&ex.compute, 0, &v)?;
            }
            let ym = y.get_mut().ok_or("shared")?;
            ops.realize(&ctx, &mut ex, &sh, std::slice::from_mut(ym))?;
            let om = out.get_mut().ok_or("shared")?;
            ops.realize_to(&ctx, &mut ex, &sh, om, &x)?;
            ex.compute.sync()?;
            let new_top = top - 2;
            let (want, got) = (rows(&ex, &y, new_top)?, rows(&ex, &out, new_top)?);
            let d = want.iter().zip(&got).filter(|(a, b)| a != b).count() + (y.meta != out.meta) as usize;
            // the source is left as it was
            let src_ok = rows(&ex, &x, top)? == v.chunks((nq + ctx.lay.kp) * n).flat_map(|p| p[..(top + 1) * n].to_vec()).collect::<Vec<_>>();
            println!("level {level} K={k}: realize_to vs copy + realize: {d} mismatches of {} words; source unchanged {src_ok}", want.len());
            bad += d + (!src_ok) as usize;
        }
    }
    assert_eq!(bad, 0);
    Ok(())
}
