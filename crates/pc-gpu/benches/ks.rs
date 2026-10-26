//! Key-switching bench: device time per op (CUDA events on the compute lane, median of 5 batches of 20
//! after a warm-up) on the n32 chain with synthetic keys (uniform `b`, random KSKA seeds; timing does not depend on
//! the values).
//!   rotate K = 1 / 8 at levels 0, 20, 40 (6 / 4 / 2 live digits), Cached and Regen key sources on dense keys
//!   (local `DevKey`), and Regen on PackedSeeded keys (28-bit packed `b`, `a` never stored);
//!   24 hoisted rotations vs 24 plain rotations; the rotate-and-sum ladder 1 -> 32768 (radix 3 and rung by rung);
//!   relin (tensor + relinearization) K = 1 / 8; rotate_sum (7 and 3 packed keys, K = 1 / 4, levels 0 / 20 / 34) with
//!   the rotations added chunk by chunk vs inside the ModDown, outputs compared word for word; mult_pt + add vs the
//!   fused mult_pt_add (K = 1, levels 20 / 34), compared word for word.
//! usage (GPU only through tools/gpu.sh): the `ks` bench binary (`cargo bench -p pc-gpu --bench ks --no-run`).
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_core::keys::SecretKey;
use pc_cuda::{Device, Event};
use pc_gpu::keygen::{test_prng, KeyGen};
use pc_gpu::keys::{DevSecret, KeySet, KeySetSpec, KskLayout, RotSpec};
use pc_gpu::ks::{DevKey, KeyA, KeyGroup, KsKeyRef, KsPlans, KsWork, KskSource};
use pc_gpu::relin;
use pc_gpu::rotate::{self, KsCfg, RotKeys};
use pc_gpu::ops::{Ops, OpsCfg, Pt};
use pc_gpu::{Context, Ct, CtPool, Executor, MemCfg, MemPlan, PoolClass};
use pc_shadow::{BtsConfig, CtMeta, PtMeta, Quirks, Shadow, ShadowConfig};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

type W = u32;
type M = Barrett64;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn rnd(seed: u64) -> impl FnMut() -> u64 {
    let mut x = seed | 1;
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}

/// Uniform rows `[unit][prime id][N]` below each prime.
fn uniform(units: usize, primes: &[u64], n: usize, seed: u64) -> Vec<W> {
    let rows = primes.len();
    let mut v = vec![0 as W; units * rows * n];
    v.par_chunks_mut(n).enumerate().for_each(|(i, r)| {
        let p = primes[i % rows];
        let mut g = rnd(seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        for x in r.iter_mut() {
            *x = ((g() as u128 * p as u128) >> 64) as W;
        }
    });
    v
}

struct Groups<'a>(BTreeMap<Vec<i64>, KeyGroup<'a>>);
impl RotKeys for Groups<'_> {
    fn group(&self, steps: &[i64]) -> Option<&KeyGroup<'_>> {
        self.0.get(steps)
    }
}

fn ladder_groups(stop: i64, radix: usize) -> Vec<Vec<i64>> {
    let mut gaps = vec![];
    let mut g = 1;
    while g < stop {
        gaps.push(g);
        g *= 2;
    }
    let mut out = vec![];
    let mut i = 0;
    while i < gaps.len() {
        let n = radix.min(gaps.len() - i);
        out.push(if n > 1 { (1..(1i64 << n)).map(|j| j * gaps[i]).collect() } else { vec![gaps[i]] });
        i += n;
    }
    out
}

/// Median over 5 batches of `reps` runs of `f`, in ms per run (device time on the compute lane).
fn time<'a>(dev: &Device, ex: &mut Executor<'a, W>, reps: usize, mut f: impl FnMut(&mut Executor<'a, W>) -> Res<()>) -> Res<f64> {
    f(ex)?;
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

fn main() -> Res<()> {
    let dev = Device::open()?;
    let chain = Chain::generate(&ChainSpec::n32())?;
    let n = chain.n;
    let nq = chain.q.len();
    let dnum = chain.dnum();
    let primes = chain.primes();
    let kmax = 8;
    // keys: rotations 1..=24 (hoisted / plain), the ladder's radix groups and rungs, relin
    let lg3 = ladder_groups(32768, 3);
    let lg0 = ladder_groups(32768, 1);
    let mut steps: Vec<i64> = (1..=24).collect();
    for g in lg3.iter().chain(&lg0) {
        steps.extend(g);
    }
    steps.sort();
    steps.dedup();
    let cfg = MemCfg { k_max: kmax, pool: vec![PoolClass { k: 1, count: 30 }, PoolClass { k: 4, count: 2 }, PoolClass { k: kmax, count: 4 }], events: 16, scratch_cts: 2 };
    let mut plan = MemPlan::build::<W, M>(&chain, cfg)?;
    let lay = plan.lay;
    plan.reserve("ks.plans", KsPlans::bytes::<W>(&chain, kmax)?);
    plan.reserve("keys", (steps.len() + 1) * DevKey::<W>::bytes(&lay, dnum, true, true));
    plan.reserve("ks.groups", (steps.len() + lg3.len() + 4) * KeyGroup::bytes(24, true));
    plan.reserve("ks.regen_a", KsWork::<W>::regen_bytes(&lay, dnum));
    // the same rotations + relin as PackedSeeded keys (generated on the device)
    let spec = KeySetSpec { relin: true, conj: false, rots: steps.iter().map(|&s| RotSpec { step: s, q_top: None }).collect(), encaps: false };
    let packed_layout = KskLayout::packed_for::<W>(&chain)?;
    let shapes: Vec<_> = spec.keys(&chain)?.into_iter().map(|(_, s)| s).collect();
    plan.reserve("keys.packed", spec.bytes::<W>(&chain, packed_layout)?);
    plan.reserve("keygen", KeyGen::<W>::bytes(&lay, &shapes));
    plan.reserve("secret", DevSecret::<W>::bytes(&lay));
    plan.reserve("ks.groups.packed", (steps.len() + lg3.len() + 4) * KeyGroup::bytes(24, true));
    plan.reserve("ks.groups.hybrid", 2 * (steps.len() + lg3.len() + 4) * KeyGroup::bytes(24, true));
    Ops::<W>::reserve(&mut plan, &OpsCfg::default());
    plan.reserve("bench.pt", (chain.q.len() + chain.p.len()) * n * 4);
    println!("[bench] {} keys; MemPlan {}", steps.len() + 1, plan.report());
    let arena = plan.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &plan)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &plan)?;
    let pool = CtPool::<W>::new(&arena, &plan)?;
    let kp = KsPlans::new(&arena, &ex.compute, &ctx, kmax)?;
    let mut keys: BTreeMap<u64, DevKey<'_, W>> = BTreeMap::new();
    let mut seed_rng = rnd(0x5EED);
    let mut gs: Vec<u64> = steps.iter().map(|&s| pc_core::galois::rotation_galois(s, n)).collect::<Result<_, _>>()?;
    gs.push(1);
    for g in gs {
        let b = uniform(dnum, &primes, n, 0xB0B0 + g);
        let sd: [u32; 8] = std::array::from_fn(|_| seed_rng() as u32);
        keys.insert(g, DevKey::new(&arena, &ex.compute, &ctx, &kp, g, dnum, &b, Some(sd), KeyA::Expand)?);
    }
    let mut pkeys = KeySet::<W>::carve(&arena, &chain, &spec, packed_layout)?;
    {
        let mut prng = test_prng(0x5e);
        let sk = SecretKey::sample(&mut prng, n)?;
        let mut sec = DevSecret::<W>::carve(&arena, lay)?;
        sec.load(&ctx, &ex.compute, &sk)?;
        let mut kg = KeyGen::<W>::new(&arena, &ex.compute, &chain, &shapes)?;
        kg.generate_set(&ctx, &mut ex, &sec, &mut pkeys, &mut prng)?;
    }
    let regen = ex.ks.stage_ptrs();
    let mk = |st: &[i64]| -> Res<KeyGroup<'_>> {
        let mut refs: Vec<&dyn KsKeyRef> = Vec::new();
        for &s in st {
            refs.push(&keys[&pc_core::galois::rotation_galois(s, n)?]);
        }
        Ok(KeyGroup::new(&arena, &ex.compute, n, &refs, regen)?)
    };
    let mut groups = Groups(BTreeMap::new());
    for &s in &steps {
        groups.0.insert(vec![s], mk(&[s])?);
    }
    for g in &lg3 {
        if !groups.0.contains_key(g) {
            groups.0.insert(g.clone(), mk(g)?);
        }
    }
    let h24: Vec<i64> = (1..=24).collect();
    groups.0.insert(h24.clone(), mk(&h24)?);
    // dense keys read in place with only the first 4 / 2 digits of `a` regenerated (the rest read)
    let mut hyb = Vec::new();
    for nd in [4usize, 2] {
        let mut gm = Groups(BTreeMap::new());
        for key in groups.0.keys() {
            let mut g = mk(key)?;
            g.set_regen_digits(nd);
            gm.0.insert(key.clone(), g);
        }
        hyb.push(gm);
    }
    let relin_key = KeyGroup::new(&arena, &ex.compute, n, &[&keys[&1] as &dyn KsKeyRef], None)?;
    let mkp = |st: &[i64]| -> Res<KeyGroup<'_>> {
        let refs: Vec<&dyn KsKeyRef> = st.iter().map(|&s| pkeys.rot(s).map(|k| k as &dyn KsKeyRef)).collect::<Result<_, _>>()?;
        Ok(KeyGroup::new(&arena, &ex.compute, n, &refs, regen)?)
    };
    let mut pgroups = Groups(BTreeMap::new());
    for &s in &steps {
        pgroups.0.insert(vec![s], mkp(&[s])?);
    }
    for g in &lg3 {
        if !pgroups.0.contains_key(g) {
            pgroups.0.insert(g.clone(), mkp(g)?);
        }
    }
    pgroups.0.insert(h24.clone(), mkp(&h24)?);
    let g3: Vec<i64> = vec![1, 2, 3];
    pgroups.0.insert(g3.clone(), mkp(&g3)?);
    let prelin = KeyGroup::new(&arena, &ex.compute, n, &[pkeys.relin()? as &dyn KsKeyRef], regen)?;
    let mut ops = Ops::<W>::new(&arena, &dev, &ex.compute, &ctx, OpsCfg::default())?;
    let mut ptbuf = arena.alloc::<pc_cuda::Compute, W>("bench.pt", (chain.q.len() + chain.p.len()) * n)?;
    ptbuf.upload(&ex.compute, &uniform(1, &primes, n, 0x9E7))?;
    ex.compute.sync()?;
    println!("[bench] setup done, arena used {} of {} B", arena.used_bytes(), arena.capacity());
    arena.freeze();

    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let fill = |ct: &mut Ct<'_, W>, level: u32, ex: &Executor<'_, W>| -> Res<()> {
        let v = uniform(2 * ct.k(), &primes, n, 0xC7 + level as u64);
        ct.buf.buf.upload(&ex.compute, &v)?;
        ct.meta = CtMeta { level, deg: 1, nf: 1.0, slots: (n / 2) as u32, tag: Default::default() };
        Ok(())
    };
    let srcs = [(KskSource::Cached, "cached", &groups, &relin_key), (KskSource::Regen, "regen", &groups, &relin_key), (KskSource::Regen, "regen+packed", &pgroups, &prelin), (KskSource::Regen, "regen 4 digits", &hyb[0], &relin_key), (KskSource::Regen, "regen 2 digits", &hyb[1], &relin_key)];
    println!("\n| op | K | level (digits) | source | ms/op | ms/ct |");
    println!("|---|---|---|---|---|---|");
    for &k in &[1usize, 8] {
        for &level in &[0u32, 20, 40] {
            let top = nq - 1 - level as usize;
            let mut x = pool.acquire(k, CtMeta::new(level, 1))?;
            let mut o = pool.acquire(k, CtMeta::new(level, 1))?;
            fill(x.get_mut().ok_or("shared")?, level, &ex)?;
            for &(src, sn, gr, _) in &srcs {
                let kc = KsCfg { plans: &kp, source: src };
                let om = o.get_mut().ok_or("shared")?;
                let g1 = &gr.0[&vec![1i64]];
                let ms = time(&dev, &mut ex, 20, |ex| Ok(rotate::rotate(&ctx, ex, kc, &sh, g1, &x, 1, om)?))?;
                println!("| rotate | {k} | {level} ({}) | {sn} | {ms:.3} | {:.3} |", kp.ndig(top)?, ms / k as f64);
            }
            for &(src, sn, _, rk) in &srcs {
                let kc = KsCfg { plans: &kp, source: src };
                let om = o.get_mut().ok_or("shared")?;
                let ms = time(&dev, &mut ex, 20, |ex| Ok(relin::mult(&ctx, ex, kc, &sh, rk, &x, &x, om)?))?;
                println!("| mult (tensor + relin) | {k} | {level} ({}) | {sn} | {ms:.3} | {:.3} |", kp.ndig(top)?, ms / k as f64);
            }
        }
    }
    // 24 hoisted vs 24 plain, and the ladder, K = 1
    println!("\n| op | level (digits) | source | ms/op |");
    println!("|---|---|---|---|");
    for &level in &[0u32, 20] {
        let top = nq - 1 - level as usize;
        let mut x = pool.acquire(1, CtMeta::new(level, 1))?;
        fill(x.get_mut().ok_or("shared")?, level, &ex)?;
        let mut outs: Vec<_> = (0..24).map(|_| pool.acquire(1, CtMeta::new(level, 1))).collect::<Result<_, _>>()?;
        let mut tmp = pool.acquire(1, CtMeta::new(level, 1))?;
        for &(src, sn, gr, _) in &srcs {
            let kc = KsCfg { plans: &kp, source: src };
            let gh = &gr.0[&h24];
            let ms_h = {
                let mut refs: Vec<&mut Ct<'_, W>> = outs.iter_mut().map(|o| o.get_mut().ok_or("shared")).collect::<Result<_, _>>()?;
                time(&dev, &mut ex, 5, |ex| Ok(rotate::rotate_hoisted(&ctx, ex, kc, &sh, gh, &x, &h24, &mut refs)?))?
            };
            let ms_p = {
                let mut refs: Vec<&mut Ct<'_, W>> = outs.iter_mut().map(|o| o.get_mut().ok_or("shared")).collect::<Result<_, _>>()?;
                time(&dev, &mut ex, 5, |ex| {
                    for (i, o) in refs.iter_mut().enumerate() {
                        let st = i as i64 + 1;
                        rotate::rotate(&ctx, ex, kc, &sh, &gr.0[&vec![st]], &x, st, o)?;
                    }
                    Ok(())
                })?
            };
            println!("| 24 rotations hoisted | {level} ({}) | {sn} | {ms_h:.3} |", kp.ndig(top)?);
            println!("| 24 rotations plain | {level} ({}) | {sn} | {ms_p:.3} |", kp.ndig(top)?);
            for (rb, name) in [(3u32, "ladder 1..32768 radix 3 (5 rotate_sum)"), (0u32, "ladder 1..32768 rung by rung (15 rotations)")] {
                let o = &mut outs[0];
                let om = o.get_mut().ok_or("shared")?;
                let tm = tmp.get_mut().ok_or("shared")?;
                let ms = time(&dev, &mut ex, 5, |ex| Ok(rotate::rotate_and_sum(&ctx, ex, kc, &sh, gr, &x, 1, 32768, rb, om, tm)?))?;
                println!("| {name} | {level} ({}) | {sn} | {ms:.3} |", kp.ndig(top)?);
            }
        }
    }
    // rotate_sum: the dot outputs added chunk by chunk vs inside the ModDown, packed keys, Regen
    println!("\n| rotate_sum | K | level (digits) | keys | chunked ms | in ModDown ms | equal |");
    println!("|---|---|---|---|---|---|---|");
    let kc = KsCfg { plans: &kp, source: KskSource::Regen };
    for &k in &[1usize, 4] {
        for &level in &[0u32, 20, 34] {
            let top = nq - 1 - level as usize;
            let mut a = pool.acquire(k, CtMeta::new(level, 1))?;
            let mut b = pool.acquire(k, CtMeta::new(level, 1))?;
            for st in [&lg3[0], &g3] {
                let gr = &pgroups.0[st];
                let (am, bm) = (a.get_mut().ok_or("shared")?, b.get_mut().ok_or("shared")?);
                fill(am, level, &ex)?;
                fill(bm, level, &ex)?;
                rotate::rotate_sum_inplace_as(&ctx, &mut ex, kc, &sh, gr, am, st, rotate::SumForm::Chunked)?;
                rotate::rotate_sum_inplace_as(&ctx, &mut ex, kc, &sh, gr, bm, st, rotate::SumForm::InModDown)?;
                ex.compute.sync()?;
                let (va, vb) = (am.buf.buf.download(&ex.compute)?, bm.buf.buf.download(&ex.compute)?);
                let rows = lay.rows();
                // Q rows 0..=top of both parts of every ciphertext
                let live = |v: &[W]| -> Vec<W> { (0..2 * k).flat_map(|p| v[(p * rows) * n..(p * rows + top + 1) * n].to_vec()).collect() };
                let eq = live(&va) == live(&vb);
                let ms_s = time(&dev, &mut ex, 20, |ex| Ok(rotate::rotate_sum_inplace_as(&ctx, ex, kc, &sh, gr, am, st, rotate::SumForm::Chunked)?))?;
                let ms_g = time(&dev, &mut ex, 20, |ex| Ok(rotate::rotate_sum_inplace_as(&ctx, ex, kc, &sh, gr, bm, st, rotate::SumForm::InModDown)?))?;
                println!("| rotate_sum | {k} | {level} ({}) | {} | {ms_s:.3} | {ms_g:.3} | {eq} |", kp.ndig(top)?, st.len());
                if !eq {
                    return Err(format!("rotate_sum in-ModDown sum != chunked (K {k}, level {level}, {} keys)", st.len()).into());
                }
            }
        }
    }
    // mult_pt + add / sub vs the fused mult_pt_add
    println!("\n| mult_pt + add | K | level | sub | product first | two ops ms | fused ms | equal |");
    println!("|---|---|---|---|---|---|---|---|");
    for &k in &[1usize] {
        for &level in &[20u32, 34] {
            let top = nq - 1 - level as usize;
            let sf = ctx.scale.sf(top as i32)?;
            let pm = PtMeta { level, deg: 1, nf: sf, tag: Default::default() };
            let pt = Pt::of(pm, &ptbuf);
            let mut x = pool.acquire(k, CtMeta::new(level, 1))?;
            let mut y = pool.acquire(k, CtMeta::new(level, 1))?;
            let mut t = pool.acquire(k, CtMeta::new(level, 1))?;
            let mut o1 = pool.acquire(k, CtMeta::new(level, 1))?;
            let mut o2 = pool.acquire(k, CtMeta::new(level, 1))?;
            fill(x.get_mut().ok_or("shared")?, level, &ex)?;
            fill(y.get_mut().ok_or("shared")?, level + 1, &ex)?;
            let (xm, ym) = (x.get_mut().ok_or("shared")?, y.get_mut().ok_or("shared")?);
            xm.meta = CtMeta { nf: sf, ..xm.meta };
            ym.meta = sh.mult_pt(&xm.meta, &pm)?.out;
            let (xr, yr) = (&*xm, &*ym);
            for (sub, first) in [(false, false), (true, false), (true, true)] {
                if !Ops::<W>::mult_pt_add_fusable(&sh, &xr.meta, &pm, &yr.meta, sub, first)? {
                    return Err(format!("mult_pt_add not fusable at level {level}").into());
                }
                let (tm, a1, a2) = (t.get_mut().ok_or("shared")?, o1.get_mut().ok_or("shared")?, o2.get_mut().ok_or("shared")?);
                let two = |ex: &mut Executor<'_, W>, ops: &mut Ops<'_, W>, tm: &mut Ct<'_, W>, a1: &mut Ct<'_, W>| -> Res<()> {
                    ops.mult_pt(&ctx, ex, &sh, std::slice::from_mut(tm), &[xr], &[pt])?;
                    let (l, r): (&Ct<'_, W>, &Ct<'_, W>) = if first { (&*tm, yr) } else { (yr, &*tm) };
                    if sub {
                        ops.sub(&ctx, ex, &sh, std::slice::from_mut(a1), &[l], &[r])?;
                    } else {
                        ops.add(&ctx, ex, &sh, std::slice::from_mut(a1), &[l], &[r])?;
                    }
                    Ok(())
                };
                two(&mut ex, &mut ops, tm, a1)?;
                ops.mult_pt_add(&ctx, &mut ex, &sh, a2, xr, &pt, yr, sub, first)?;
                ex.compute.sync()?;
                let rows = lay.rows();
                let live = |v: &[W]| -> Vec<W> { (0..2 * k).flat_map(|p| v[(p * rows) * n..(p * rows + top + 1) * n].to_vec()).collect() };
                let eq = live(&a1.buf.buf.download(&ex.compute)?) == live(&a2.buf.buf.download(&ex.compute)?) && a1.meta == a2.meta;
                let ms2 = time(&dev, &mut ex, 20, |ex| two(ex, &mut ops, tm, a1))?;
                let ms1 = time(&dev, &mut ex, 20, |ex| Ok(ops.mult_pt_add(&ctx, ex, &sh, a2, xr, &pt, yr, sub, first)?))?;
                println!("| mult_pt + add | {k} | {level} | {sub} | {first} | {ms2:.4} | {ms1:.4} | {eq} |");
                if !eq {
                    return Err(format!("mult_pt_add != mult_pt + add (K {k}, level {level}, sub {sub}, first {first})").into());
                }
            }
        }
    }
    Ok(())
}
