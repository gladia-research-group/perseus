//! Rotation grids (`rotate::rotate_grid`) and K-batched rotate_sums (`rotate::rotate_sum_batch`), GPU (run through
//! tools/gpu.sh, `--ignored --test-threads=1`):
//! - every output of a grid is bit-identical to the plain rotation of its input (1, 2 and 4 inputs, duplicate
//!   (input, step) pairs, levels 28 and 44, packed seeded keys generated on the device), every output of a K-batched
//!   rotate_sum to the rotate_sum of its input (K = 2, 3, 4), every output of a chain link (a grid reusing an
//!   earlier grid's gather and ModUp, kept in place or in a held workspace across other key switches) likewise;
//! - device ms of a 4 x 63 grid vs its 252 plain rotations at level 28 (the prefill attention's own-chunk bursts), of
//!   a 1 x 7 grid vs 7 plain rotations at level 44 (the decode's baby steps), and of K-batched rotate_sum / relin
//!   against K single ones (what batching those siblings would save).
use pc_core::keys::SecretKey;
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::{Device, Event};
use pc_gpu::keygen::{test_prng, KeyGen};
use pc_gpu::keys::{DevSecret, KeySet, KeySetSpec, KskLayout, RotSpec};
use pc_gpu::ks::{KeyGroup, KsKeyRef, KsPlans, KsWork, KskSource};
use pc_gpu::relin;
use pc_gpu::rotate::{self, KsCfg};
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

/// Median of 5 batches of `reps` runs of `f`, ms per run (device time on the compute lane).
fn time<'a>(dev: &Device, ex: &mut Executor<'a, W>, reps: usize, mut f: impl FnMut(&mut Executor<'a, W>) -> Res<()>) -> Res<f64> {
    f(ex)?;
    ex.compute.sync()?;
    let mut v = Vec::new();
    for _ in 0..5 {
        let (a, b) = (Event::timing(dev)?, Event::timing(dev)?);
        a.record(&ex.compute)?;
        for _ in 0..reps {
            f(ex)?;
        }
        b.record(&ex.compute)?;
        b.sync()?;
        v.push(a.elapsed_ms(&b)? as f64 / reps as f64);
    }
    v.sort_by(|x, y| x.total_cmp(y));
    Ok(v[2])
}

/// Q rows 0..=top of both parts of ciphertext `c` of `x`.
fn rows(ex: &Executor<'_, W>, x: &Ct<'_, W>, c: usize, top: usize) -> Res<Vec<W>> {
    let lay = x.buf.lay;
    let mut out = Vec::with_capacity(2 * (top + 1) * lay.n);
    for part in 0..2 {
        out.extend(x.buf.buf.download_range(&ex.compute, c * lay.ct_words() + part * lay.poly_words(), (top + 1) * lay.n)?);
    }
    Ok(out)
}

#[test]
#[ignore = "GPU: tools/gpu.sh <log> <rotate_grid test binary> --ignored --test-threads=1"]
fn grid() -> Res<()> {
    let dev = Device::open()?;
    let chain = Chain::generate(&ChainSpec::n32())?;
    let (n, nq, primes) = (chain.n, chain.q.len(), chain.primes());
    let kmax = 8;
    let own: Vec<i64> = (1..=63).map(|e| -e).collect();
    let sum7: Vec<i64> = (1..=7).map(|j| 512 * j).collect();
    let steps: Vec<i64> = own.iter().chain(&sum7).copied().collect();
    let spec = KeySetSpec { relin: true, conj: false, rots: steps.iter().map(|&s| RotSpec { step: s, q_top: None }).collect(), encaps: false };
    let packed = KskLayout::packed_for::<W>(&chain)?;
    let shapes: Vec<_> = spec.keys(&chain)?.into_iter().map(|(_, s)| s).collect();
    let n_cts = 2 * 4 * 63 + 16;
    let cfg = MemCfg { k_max: kmax, pool: vec![PoolClass { k: 1, count: n_cts }, PoolClass { k: 4, count: 8 }], events: 16, scratch_cts: 2 };
    let mut plan = MemPlan::build::<W, M>(&chain, cfg)?;
    let lay = plan.lay;
    plan.reserve("ks.plans", KsPlans::bytes::<W>(&chain, kmax)?);
    plan.reserve("keys", spec.bytes::<W>(&chain, packed)?);
    plan.reserve("keygen", KeyGen::<W>::bytes(&lay, &shapes));
    plan.reserve("secret", DevSecret::<W>::bytes(&lay));
    plan.reserve("ks.regen_a", KsWork::<W>::regen_bytes(&lay, chain.dnum()));
    plan.reserve("ks.groups", (steps.len() + 8) * KeyGroup::bytes(64, true));
    let held_words = KsWork::<W>::words(&lay, kmax, chain.dnum())[0];
    plan.reserve("held", pc_cuda::Arena::bytes_for::<W>(held_words));
    let arena = plan.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &plan)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &plan)?;
    let pool = CtPool::<W>::new(&arena, &plan)?;
    let kp = KsPlans::new(&arena, &ex.compute, &ctx, kmax)?;
    let mut keys = KeySet::<W>::carve(&arena, &chain, &spec, packed)?;
    {
        let mut prng = test_prng(0x6d);
        let sk = SecretKey::sample(&mut prng, n)?;
        let mut sec = DevSecret::<W>::carve(&arena, lay)?;
        sec.load(&ctx, &ex.compute, &sk)?;
        let mut kg = KeyGen::<W>::new(&arena, &ex.compute, &chain, &shapes)?;
        kg.generate_set(&ctx, &mut ex, &sec, &mut keys, &mut prng)?;
    }
    let stage = ex.ks.stage_ptrs();
    let group = |st: &[i64]| -> Res<KeyGroup<'_>> {
        let refs: Vec<&dyn KsKeyRef> = st.iter().map(|&s| keys.rot(s).map(|k| k as &dyn KsKeyRef)).collect::<Result<_, _>>()?;
        Ok(KeyGroup::new(&arena, &ex.compute, n, &refs, stage)?)
    };
    let singles: Vec<KeyGroup<'_>> = own.iter().map(|&s| group(&[s])).collect::<Res<_>>()?;
    let g_own = group(&own)?;
    let g_sum = group(&sum7)?;
    let g_relin = KeyGroup::new(&arena, &ex.compute, n, &[keys.relin()? as &dyn KsKeyRef], stage)?;
    let mut held: pc_cuda::DBuf<'_, pc_cuda::Compute, W> = arena.alloc("held", held_words)?;
    ex.compute.sync()?;
    arena.freeze();
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let kc = KsCfg { plans: &kp, source: KskSource::Regen };
    let meta = |level: u32| CtMeta { level, deg: 1, nf: 1.0, slots: (n / 2) as u32, tag: Default::default() };

    // bit identity: (inputs, steps used) at two levels; duplicates of the first column
    for &level in &[28u32, 44] {
        let top = nq - 1 - level as usize;
        for &(k, ns) in &[(1usize, 7usize), (2, 5), (4, 63)] {
            let mut xs: Vec<_> = (0..k).map(|_| pool.acquire(1, meta(level))).collect::<Result<_, _>>()?;
            for (c, x) in xs.iter_mut().enumerate() {
                let v = uniform(2, &primes, n, 0xC7 + 97 * level as u64 + 13 * c as u64 + k as u64);
                x.get_mut().ok_or("shared")?.buf.buf.upload_at(&ex.compute, 0, &v)?;
            }
            let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|s| (0..k).map(move |c| (c, s))).chain((0..k).map(|c| (c, 0))).collect();
            let mut outs: Vec<_> = pairs.iter().map(|_| pool.acquire(1, meta(level))).collect::<Result<_, _>>()?;
            let mut gather = pool.acquire(4, meta(level))?;
            {
                let xr: Vec<&Ct<'_, W>> = xs.iter().map(|x| &**x).collect();
                let mut refs: Vec<(usize, usize, &mut Ct<'_, W>)> = pairs.iter().zip(outs.iter_mut()).map(|(&(c, s), o)| Ok((c, s, o.get_mut().ok_or("shared")?))).collect::<Res<_>>()?;
                let g = gather.get_mut().ok_or("shared")?;
                rotate::rotate_grid(&ctx, &mut ex, kc, &sh, &g_own, &xr, if k > 1 { Some(g) } else { None }, true, &own[..ns], &mut refs)?;
            }
            let mut want = pool.acquire(1, meta(level))?;
            let mut bad = 0;
            for (j, &(c, s)) in pairs.iter().enumerate() {
                let w = want.get_mut().ok_or("shared")?;
                rotate::rotate(&ctx, &mut ex, kc, &sh, &singles[s], &xs[c], own[s], w)?;
                if rows(&ex, &outs[j], 0, top)? != rows(&ex, w, 0, top)? || outs[j].meta != w.meta {
                    bad += 1;
                }
            }
            println!("[grid] level {level}: {k} inputs x {ns} steps (+{k} duplicates): {bad} of {} outputs differ from plain rotations", pairs.len());
            assert_eq!(bad, 0);
        }
    }

    // chains: a fresh grid over steps 0..3, then links over steps 3..9 reusing its gather and ModUp, either in place
    // (nothing between) or held (the ModUp workspace swapped out while another rotation runs)
    for &level in &[28u32, 44] {
        let top = nq - 1 - level as usize;
        for &(k, hold) in &[(1usize, false), (2, false), (2, true), (1, true)] {
            let mut xs: Vec<_> = (0..k).map(|_| pool.acquire(1, meta(level))).collect::<Result<_, _>>()?;
            for (c, x) in xs.iter_mut().enumerate() {
                x.get_mut().ok_or("shared")?.buf.buf.upload_at(&ex.compute, 0, &uniform(2, &primes, n, 0xB1 + 31 * level as u64 + 7 * c as u64 + k as u64))?;
            }
            let pairs: Vec<(usize, usize)> = (0..9).flat_map(|s| (0..k).map(move |c| (c, s))).collect();
            let mut outs: Vec<_> = pairs.iter().map(|_| pool.acquire(1, meta(level))).collect::<Result<_, _>>()?;
            let mut gather = pool.acquire(4, meta(level))?;
            let mut other = pool.acquire(1, meta(level))?;
            for (lo, hi) in [(0usize, 3usize), (3, 5), (5, 9)] {
                let xr: Vec<&Ct<'_, W>> = xs.iter().map(|x| &**x).collect();
                let mut refs: Vec<(usize, usize, &mut Ct<'_, W>)> =
                    pairs.iter().zip(outs.iter_mut()).filter(|(&(_, s), _)| s >= lo && s < hi).map(|(&(c, s), o)| Ok((c, s, o.get_mut().ok_or("shared")?))).collect::<Res<_>>()?;
                let g = gather.get_mut().ok_or("shared")?;
                if hold {
                    std::mem::swap(&mut ex.ks.digits, &mut held);
                }
                rotate::rotate_grid(&ctx, &mut ex, kc, &sh, &g_own, &xr, if k > 1 { Some(g) } else { None }, lo == 0, &own, &mut refs)?;
                if hold {
                    std::mem::swap(&mut ex.ks.digits, &mut held);
                    // another key switch between the links (the workspace it leaves behind is not the chain's)
                    rotate::rotate(&ctx, &mut ex, kc, &sh, &singles[20], &xs[0], own[20], other.get_mut().ok_or("shared")?)?;
                }
            }
            let mut want = pool.acquire(1, meta(level))?;
            let mut bad = 0;
            for (j, &(c, s)) in pairs.iter().enumerate() {
                let w = want.get_mut().ok_or("shared")?;
                rotate::rotate(&ctx, &mut ex, kc, &sh, &singles[s], &xs[c], own[s], w)?;
                if rows(&ex, &outs[j], 0, top)? != rows(&ex, w, 0, top)? || outs[j].meta != w.meta {
                    bad += 1;
                }
            }
            println!("[grid] level {level}: chain of 3 links over {k} input(s), {}: {bad} of {} outputs differ from plain rotations", if hold { "held" } else { "in place" }, pairs.len());
            assert_eq!(bad, 0);
        }
    }

    for &level in &[28u32, 44] {
        let top = nq - 1 - level as usize;
        for k in 2..=4usize {
            let mut xs: Vec<_> = (0..k).map(|_| pool.acquire(1, meta(level))).collect::<Result<_, _>>()?;
            for (c, x) in xs.iter_mut().enumerate() {
                x.get_mut().ok_or("shared")?.buf.buf.upload_at(&ex.compute, 0, &uniform(2, &primes, n, 0xA1 + 31 * level as u64 + 7 * c as u64 + k as u64))?;
            }
            let mut outs: Vec<_> = (0..k).map(|_| pool.acquire(1, meta(level))).collect::<Result<_, _>>()?;
            let mut gather = pool.acquire(4, meta(level))?;
            {
                let xr: Vec<&Ct<'_, W>> = xs.iter().map(|x| &**x).collect();
                let mut refs: Vec<&mut Ct<'_, W>> = outs.iter_mut().map(|o| o.get_mut().ok_or("shared")).collect::<Result<_, _>>()?;
                rotate::rotate_sum_batch(&ctx, &mut ex, kc, &sh, &g_sum, &xr, gather.get_mut().ok_or("shared")?, &sum7, &mut refs)?;
            }
            let mut want = pool.acquire(1, meta(level))?;
            let mut bad = 0;
            for c in 0..k {
                let w = want.get_mut().ok_or("shared")?;
                rotate::rotate_sum(&ctx, &mut ex, kc, &sh, &g_sum, &xs[c], &sum7, w)?;
                if rows(&ex, &outs[c], 0, top)? != rows(&ex, w, 0, top)? || outs[c].meta != w.meta {
                    bad += 1;
                }
            }
            println!("[grid] level {level}: rotate_sum K-batch of {k}: {bad} of {k} outputs differ from single rotate_sums");
            assert_eq!(bad, 0);
        }
    }

    // device ms
    let level = 28u32;
    let top = nq - 1 - level as usize;
    let mut xs: Vec<_> = (0..4).map(|_| pool.acquire(1, meta(level))).collect::<Result<_, _>>()?;
    for (c, x) in xs.iter_mut().enumerate() {
        x.get_mut().ok_or("shared")?.buf.buf.upload_at(&ex.compute, 0, &uniform(2, &primes, n, 0xD00 + c as u64))?;
    }
    let pairs: Vec<(usize, usize)> = (0..63).flat_map(|s| (0..4).map(move |c| (c, s))).collect();
    let mut outs: Vec<_> = pairs.iter().map(|_| pool.acquire(1, meta(level))).collect::<Result<_, _>>()?;
    let mut gather = pool.acquire(4, meta(level))?;
    let ms_grid = {
        let xr: Vec<&Ct<'_, W>> = xs.iter().map(|x| &**x).collect();
        let mut refs: Vec<(usize, usize, &mut Ct<'_, W>)> = pairs.iter().zip(outs.iter_mut()).map(|(&(c, s), o)| Ok((c, s, o.get_mut().ok_or("shared")?))).collect::<Res<_>>()?;
        let g = gather.get_mut().ok_or("shared")?;
        time(&dev, &mut ex, 3, |ex| Ok(rotate::rotate_grid(&ctx, ex, kc, &sh, &g_own, &xr, Some(&mut *g), true, &own, &mut refs)?))?
    };
    let ms_plain = {
        let mut refs: Vec<&mut Ct<'_, W>> = outs.iter_mut().map(|o| o.get_mut().ok_or("shared")).collect::<Result<_, _>>()?;
        time(&dev, &mut ex, 3, |ex| {
            for (j, &(c, s)) in pairs.iter().enumerate() {
                rotate::rotate(&ctx, ex, kc, &sh, &singles[s], &xs[c], own[s], refs[j])?;
            }
            Ok(())
        })?
    };
    println!("[grid] level {level} (top {top}): 4 x 63 grid {ms_grid:.3} ms vs 252 plain rotations {ms_plain:.3} ms ({:.4} vs {:.4} ms/rotation)", ms_grid / 252.0, ms_plain / 252.0);
    let lv44 = 44u32;
    let mut x1 = pool.acquire(1, meta(lv44))?;
    x1.get_mut().ok_or("shared")?.buf.buf.upload_at(&ex.compute, 0, &uniform(2, &primes, n, 0xE00))?;
    let ms7 = {
        let xr = [&*x1];
        let mut refs: Vec<(usize, usize, &mut Ct<'_, W>)> = outs.iter_mut().take(7).enumerate().map(|(s, o)| Ok((0, s, o.get_mut().ok_or("shared")?))).collect::<Res<_>>()?;
        time(&dev, &mut ex, 10, |ex| Ok(rotate::rotate_grid(&ctx, ex, kc, &sh, &g_own, &xr, None, true, &own[..7], &mut refs)?))?
    };
    let ms7p = {
        let mut refs: Vec<&mut Ct<'_, W>> = outs.iter_mut().take(7).map(|o| o.get_mut().ok_or("shared")).collect::<Result<_, _>>()?;
        time(&dev, &mut ex, 10, |ex| {
            for (s, o) in refs.iter_mut().enumerate() {
                rotate::rotate(&ctx, ex, kc, &sh, &singles[s], &x1, own[s], o)?;
            }
            Ok(())
        })?
    };
    println!("[grid] level {lv44}: 1 x 7 grid {ms7:.3} ms vs 7 plain rotations {ms7p:.3} ms");
    // a held chain of 63 links over 2 inputs (one step each, another rotation between links) vs the plain rotations
    let ms_chain = {
        let xr: Vec<&Ct<'_, W>> = xs.iter().take(2).map(|x| &**x).collect();
        let mut refs: Vec<&mut Ct<'_, W>> = outs.iter_mut().take(127).map(|o| o.get_mut().ok_or("shared")).collect::<Result<_, _>>()?;
        let (other, links) = refs.split_first_mut().ok_or("outs")?;
        let g = gather.get_mut().ok_or("shared")?;
        time(&dev, &mut ex, 2, |ex| {
            for s in 0..63 {
                let (a, b) = links[2 * s..2 * s + 2].split_at_mut(1);
                let mut o: Vec<(usize, usize, &mut Ct<'_, W>)> = vec![(0, s, &mut *a[0]), (1, s, &mut *b[0])];
                std::mem::swap(&mut ex.ks.digits, &mut held);
                rotate::rotate_grid(&ctx, ex, kc, &sh, &g_own, &xr, Some(&mut *g), s == 0, &own, &mut o)?;
                std::mem::swap(&mut ex.ks.digits, &mut held);
                rotate::rotate(&ctx, ex, kc, &sh, &singles[0], &xs[3], own[0], other)?;
            }
            Ok(())
        })?
    };
    let ms_chain_plain = {
        let mut refs: Vec<&mut Ct<'_, W>> = outs.iter_mut().take(127).map(|o| o.get_mut().ok_or("shared")).collect::<Result<_, _>>()?;
        let (other, links) = refs.split_first_mut().ok_or("outs")?;
        time(&dev, &mut ex, 2, |ex| {
            for s in 0..63 {
                for c in 0..2 {
                    rotate::rotate(&ctx, ex, kc, &sh, &singles[s], &xs[c], own[s], links[2 * s + c])?;
                }
                rotate::rotate(&ctx, ex, kc, &sh, &singles[0], &xs[3], own[0], other)?;
            }
            Ok(())
        })?
    };
    println!("[grid] level {level}: held chain of 63 links x 2 inputs (+63 other rotations) {ms_chain:.3} ms vs plain {ms_chain_plain:.3} ms");
    // which part is slow: (k, held, fresh every link, other rotation between links)
    for &(k, hold, every, other_rot) in &[(2usize, false, false, false), (2, true, false, false), (2, true, true, false), (1, true, false, true), (1, false, false, false), (2, false, true, false)] {
        let xr: Vec<&Ct<'_, W>> = xs.iter().take(k).map(|x| &**x).collect();
        let mut refs: Vec<&mut Ct<'_, W>> = outs.iter_mut().take(127).map(|o| o.get_mut().ok_or("shared")).collect::<Result<_, _>>()?;
        let (other, links) = refs.split_first_mut().ok_or("outs")?;
        let g = gather.get_mut().ok_or("shared")?;
        let ms = time(&dev, &mut ex, 2, |ex| {
            for s in 0..63 {
                let (a, b) = links[2 * s..2 * s + 2].split_at_mut(1);
                let mut o: Vec<(usize, usize, &mut Ct<'_, W>)> = vec![(0, s, &mut *a[0])];
                if k == 2 {
                    o.push((1, s, &mut *b[0]));
                }
                if hold {
                    std::mem::swap(&mut ex.ks.digits, &mut held);
                }
                rotate::rotate_grid(&ctx, ex, kc, &sh, &g_own, &xr, if k > 1 { Some(&mut *g) } else { None }, s == 0 || every, &own, &mut o)?;
                if hold {
                    std::mem::swap(&mut ex.ks.digits, &mut held);
                }
                if other_rot {
                    rotate::rotate(&ctx, ex, kc, &sh, &singles[0], &xs[3], own[0], other)?;
                }
            }
            Ok(())
        })?;
        println!("[grid] level {level}: chain of 63 links, {k} input(s), held {hold}, fresh every link {every}, other rotation {other_rot}: {ms:.3} ms");
    }
    // K-batched rotate_sum (7 keys) and relin at level 28: K = 4 in one buffer vs 4 single ciphertexts
    let mut x4 = pool.acquire(4, meta(level))?;
    x4.get_mut().ok_or("shared")?.buf.buf.upload_at(&ex.compute, 0, &uniform(8, &primes, n, 0xF00))?;
    let mut o4 = pool.acquire(4, meta(level))?;
    let ms_rs4 = {
        let o = o4.get_mut().ok_or("shared")?;
        time(&dev, &mut ex, 10, |ex| Ok(rotate::rotate_sum(&ctx, ex, kc, &sh, &g_sum, &x4, &sum7, o)?))?
    };
    let ms_rs1 = {
        let mut refs: Vec<&mut Ct<'_, W>> = outs.iter_mut().take(4).map(|o| o.get_mut().ok_or("shared")).collect::<Result<_, _>>()?;
        time(&dev, &mut ex, 10, |ex| {
            for (c, o) in refs.iter_mut().enumerate() {
                rotate::rotate_sum(&ctx, ex, kc, &sh, &g_sum, &xs[c], &sum7, o)?;
            }
            Ok(())
        })?
    };
    let ms_rsb = {
        let xr: Vec<&Ct<'_, W>> = xs.iter().map(|x| &**x).collect();
        let mut refs: Vec<&mut Ct<'_, W>> = outs.iter_mut().take(4).map(|o| o.get_mut().ok_or("shared")).collect::<Result<_, _>>()?;
        let g = gather.get_mut().ok_or("shared")?;
        time(&dev, &mut ex, 10, |ex| Ok(rotate::rotate_sum_batch(&ctx, ex, kc, &sh, &g_sum, &xr, &mut *g, &sum7, &mut refs)?))?
    };
    println!("[grid] level {level}: rotate_sum of 7 keys, K = 4 in one buffer {ms_rs4:.3} ms, K-batch of 4 gathered {ms_rsb:.3} ms vs 4 x K = 1 {ms_rs1:.3} ms");
    let ms_m4 = {
        let o = o4.get_mut().ok_or("shared")?;
        time(&dev, &mut ex, 10, |ex| Ok(relin::mult(&ctx, ex, kc, &sh, &g_relin, &x4, &x4, o)?))?
    };
    let ms_m1 = {
        let mut refs: Vec<&mut Ct<'_, W>> = outs.iter_mut().take(4).map(|o| o.get_mut().ok_or("shared")).collect::<Result<_, _>>()?;
        time(&dev, &mut ex, 10, |ex| {
            for (c, o) in refs.iter_mut().enumerate() {
                relin::mult(&ctx, ex, kc, &sh, &g_relin, &xs[c], &xs[c], o)?;
            }
            Ok(())
        })?
    };
    println!("[grid] level {level}: mult (tensor + relin), K = 4 {ms_m4:.3} ms vs 4 x K = 1 {ms_m1:.3} ms");
    // a K = 4 product is the 4 single products, ciphertext by ciphertext
    {
        let mut x1 = pool.acquire(1, meta(level))?;
        let mut w = pool.acquire(1, meta(level))?;
        let o = o4.get_mut().ok_or("shared")?;
        relin::mult(&ctx, &mut ex, kc, &sh, &g_relin, &x4, &x4, o)?;
        let mut bad = 0;
        for c in 0..4 {
            let xm = x1.get_mut().ok_or("shared")?;
            let v = x4.buf.buf.download_range(&ex.compute, c * lay.ct_words(), lay.ct_words())?;
            xm.buf.buf.upload_at(&ex.compute, 0, &v)?;
            xm.meta = x4.meta;
            let wm = w.get_mut().ok_or("shared")?;
            relin::mult(&ctx, &mut ex, kc, &sh, &g_relin, &x1, &x1, wm)?;
            if rows(&ex, &o4, c, top)? != rows(&ex, &w, 0, top)? {
                bad += 1;
            }
        }
        println!("[grid] level {level}: mult K = 4: {bad} of 4 products differ from single products");
        assert_eq!(bad, 0);
    }
    // the replay's product unit: distinct operands gathered into K buffers (`bts::batch::put`), one K-batched
    // product, the products scattered (`take`): each equal to its single product
    {
        let mut ga = pool.acquire(4, meta(level))?;
        let mut gb = pool.acquire(4, meta(level))?;
        let mut w = pool.acquire(1, meta(level))?;
        for (c, (x, y)) in xs.iter().zip(xs.iter().rev()).enumerate() {
            pc_gpu::bts::batch::put(&ctx, &ex.compute, ga.get_mut().ok_or("shared")?, c, x)?;
            pc_gpu::bts::batch::put(&ctx, &ex.compute, gb.get_mut().ok_or("shared")?, c, y)?;
        }
        for g in [&mut ga, &mut gb] {
            let g = g.get_mut().ok_or("shared")?;
            g.buf.k = 4;
            g.meta = meta(level);
        }
        let o = o4.get_mut().ok_or("shared")?;
        relin::mult(&ctx, &mut ex, kc, &sh, &g_relin, &ga, &gb, o)?;
        let mut bad = 0;
        for (c, (x, y)) in xs.iter().zip(xs.iter().rev()).enumerate() {
            let m = o4.meta;
            let wm = w.get_mut().ok_or("shared")?;
            relin::mult(&ctx, &mut ex, kc, &sh, &g_relin, x, y, wm)?;
            let single = rows(&ex, wm, 0, top)?;
            let wm = w.get_mut().ok_or("shared")?;
            pc_gpu::bts::batch::take(&ctx, &ex.compute, wm, &o4, c, m)?;
            if rows(&ex, wm, 0, top)? != single {
                bad += 1;
            }
        }
        println!("[grid] level {level}: gathered K = 4 product of distinct operands: {bad} of 4 differ from single products");
        assert_eq!(bad, 0);
    }
    Ok(())
}
