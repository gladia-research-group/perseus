//! The word / modmul seams end to end: one rig generic over `(W, M)` that sets up a chain on the device
//! (keygen, encryptor, decryptor, ops, key-switch plans) and either
//! - checks it (`pc-check n64` on `(u64, Barrett64)`; also runs on any u32 strategy): encrypt/decrypt roundtrips (sk
//!   and pk) per level, the leaf ops against host references (exact words where the host can compute them, decrypted
//!   slots against numpy-style complex arithmetic otherwise), self-consistency (hoisted == plain rotations, rescale
//!   down the whole chain), and the precision of each; no oracle exists for the u64 chain, or
//! - benches it (`pc-bench modmul`): device ms per op of the ModMul-carrying kernels (elementwise products, NTT,
//!   rescale, key switching) at K = 1 / 8, plus a hash of every output so the strategies can be compared word for
//!   word (the seam contract: same inputs -> same residues under every ModMul).
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::keys::{Prng, SecretKey};
use pc_core::params::Chain;
use pc_cuda::{Compute, Device, Event};
use pc_gpu::context::{DevStrategy, DevWord};
use pc_gpu::decrypt::{Decryptor, Which};
use pc_gpu::encrypt::Encryptor;
use pc_gpu::keygen::{test_prng, KeyGen};
use pc_gpu::keys::{DevSecret, KeySet, KeySetSpec, KskLayout, RotSpec};
use pc_gpu::ks::{KeyGroup, KsKeyRef, KsPlans, KskSource};
use pc_gpu::ntt::{self, Parts, Span};
use pc_gpu::ops::{Ops, OpsCfg, Pt, Triple};
use pc_gpu::poly::CtBuf;
use pc_gpu::rotate::{self, KsCfg};
use pc_gpu::{Context, Ct, Executor, MemCfg, MemPlan, PoolClass};
use pc_shadow::{CtMeta, PtMeta};
use std::time::Instant;

pub type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub const STEPS: [i64; 4] = [1, -3, 1024, 7];
/// Rotations / conjugation decrypt at ~2^-28.5 on n32 AND n64: the ModDown is FIDESlib's plain unsigned conversion
/// (bit parity with the oracle), whose overshoot bias concentrates after the automorphism (measured: 2^-29.8
/// with unsigned vs 2^-36 with centered residues). Gate = no worse than that floor.
const ROT_BITS: f64 = 28.0;

#[derive(Clone, Debug)]
pub struct RigOpts {
    pub check: bool,
    pub bench: bool,
    pub reps: usize,
    pub ks: Vec<usize>,
    /// input levels of the leaf checks (primes dropped)
    pub levels: Vec<u32>,
    pub seed: u32,
}

#[derive(Clone, Debug, Default)]
pub struct RigOut {
    /// (what, exact mismatches or None, max |err| relative to max |want| as bits, threshold bits, pass)
    pub checks: Vec<(String, Option<usize>, Option<f64>, f64, bool)>,
    /// (op, K, device ms per call)
    pub bench: Vec<(String, usize, f64)>,
    /// (output name, FNV-1a of its words): equal across strategies of one word size
    pub hashes: Vec<(String, u64)>,
    pub setup: String,
}

fn fnv<W: DevWord>(v: &[W]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x.to_u64()).wrapping_mul(0x0000_0100_0000_01b3))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// Random complex slots with |z| <= amp (xorshift; deterministic per seed).
pub fn cvec(seed: u64, n: usize, amp: f64) -> Vec<Complex> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut u = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    (0..n).map(|_| Complex::new(u() * amp * 0.7, u() * amp * 0.7)).collect()
}

fn rel_bits(got: &[Complex], want: &[Complex]) -> f64 {
    let e = got.iter().zip(want).map(|(g, w)| (*g - *w).abs()).fold(0f64, f64::max);
    let m = want.iter().map(|w| w.abs()).fold(1e-300f64, f64::max);
    -(e / m).max(1e-300).log2()
}

/// Every word of rows 0..=top of both parts of ciphertext c.
fn rows_of<W: DevWord>(x: &Ct<'_, W>, s: &pc_cuda::Stream<Compute>, c: usize, part: usize, top: usize) -> Res<Vec<W>> {
    Ok(x.buf.download_rows(s, c, part, &(0..=top).collect::<Vec<_>>())?)
}

/// Run the rig on `chain` with words `W` and modmul strategy `M`.
pub fn run<W: DevWord, M: DevStrategy<W>>(chain: Chain, o: &RigOpts) -> Res<RigOut> {
    let mut out = RigOut::default();
    let t0 = Instant::now();
    let n = chain.n;
    let slots = n / 2;
    let nq = chain.q.len();
    let kmax = o.ks.iter().copied().max().unwrap_or(1).max(8);
    let dev = Device::open()?;
    let cfg = MemCfg { k_max: kmax.max(2), pool: vec![PoolClass { k: 1, count: 1 }], events: 64, scratch_cts: 2 };
    let mut mp = MemPlan::build::<W, M>(&chain, cfg)?;
    let lay = mp.lay;
    let spec = KeySetSpec { relin: true, conj: true, rots: STEPS.iter().map(|&s| RotSpec { step: s.rem_euclid(slots as i64), q_top: None }).collect(), encaps: false, lean: None };
    let shapes: Vec<_> = spec.keys(&chain)?.into_iter().map(|(_, s)| s).collect();
    mp.reserve("keys", spec.bytes::<W>(&chain, KskLayout::Dense)?);
    mp.reserve("keygen", KeyGen::<W>::bytes(&lay, &shapes));
    mp.reserve("secret", DevSecret::<W>::bytes(&lay));
    mp.reserve("encrypt", Encryptor::<W>::bytes(&lay, true));
    // two bottom rows: a 60-bit q0 alone holds |m| < 2^6 at scale 2^53 (n64); n32 decrypts the composite bottom
    let drows = 2 * chain.level_unit;
    mp.reserve("decrypt", Decryptor::<W>::bytes(&lay, drows, kmax.max(2)));
    mp.reserve("ks.plans", KsPlans::bytes::<W>(&chain, kmax.max(2))?);
    mp.reserve("ks.regen_a", pc_gpu::ks::KsWork::<W>::regen_bytes(&lay, chain.dnum()));
    mp.reserve("ks.groups", KeyGroup::bytes(STEPS.len() + 1, true) + 3 * KeyGroup::bytes(1, true) + STEPS.len() * KeyGroup::bytes(1, true));
    let ocfg = OpsCfg::default();
    Ops::<W>::reserve(&mut mp, &ocfg);
    let ncts = 8;
    mp.reserve("rig.cts", pc_cuda::Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, kmax.max(2))) * ncts);
    mp.reserve("rig.d2", pc_cuda::Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, kmax.max(2))));
    let arena = mp.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &mp)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let plans = KsPlans::new(&arena, &ex.compute, &ctx, kmax.max(2))?;
    let mut ops = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    let mut keys = KeySet::<W>::carve(&arena, &chain, &spec, KskLayout::Dense)?;
    let mut prng: Prng = test_prng(o.seed);
    let sk = SecretKey::sample(&mut prng, n)?;
    let mut sec = DevSecret::<W>::carve(&arena, lay)?;
    sec.load(&ctx, &ex.compute, &sk)?;
    let mut kg = KeyGen::<W>::new(&arena, &ex.compute, &chain, &shapes)?;
    let tk = kg.generate_set(&ctx, &mut ex, &sec, &mut keys, &mut prng)?;
    let mut encr = Encryptor::<W>::new(&arena, &ex.compute, lay, true, &mut prng)?;
    encr.gen_public_key(&ctx, &ex.compute, &sec, &mut prng)?;
    let mut dec = Decryptor::<W>::new(&arena, &ex.compute, lay, drows, kmax.max(2))?;
    dec.load_secret(&ex.compute, &sec)?;
    let stage = ex.ks.stage_ptrs();
    let relin = KeyGroup::new(&arena, &ex.compute, n, &[keys.relin()? as &dyn KsKeyRef], stage)?;
    let conj = KeyGroup::new(&arena, &ex.compute, n, &[keys.conj()? as &dyn KsKeyRef], stage)?;
    let one: Vec<KeyGroup<'_>> = STEPS.iter().map(|&s| keys.rot(s).and_then(|k| KeyGroup::new(&arena, &ex.compute, n, &[k as &dyn KsKeyRef], stage))).collect::<pc_gpu::Result<_>>()?;
    let hoist_refs: Vec<&dyn KsKeyRef> = STEPS.iter().map(|&s| keys.rot(s).map(|k| k as &dyn KsKeyRef)).collect::<pc_gpu::Result<_>>()?;
    let hoist = KeyGroup::new(&arena, &ex.compute, n, &hoist_refs, stage)?;
    let mut cts: Vec<Ct<'_, W>> = (0..ncts).map(|_| CtBuf::<Compute, W>::new(&arena, "rig.cts", lay, kmax.max(2)).map(|b| Ct::new(CtMeta::new(0, 1), b))).collect::<pc_gpu::Result<_>>()?;
    let mut d2 = CtBuf::<Compute, W>::new(&arena, "rig.d2", lay, kmax.max(2))?;
    ex.compute.sync()?;
    arena.freeze();
    let sh = pc_gpu::bts::dense::shadow_of(&ctx.scale);
    let enc = Encoder::new(&chain);
    out.setup = format!(
        "N=2^{} |Q|={} |P|={} dnum={} word={} {}: keygen {} keys {:.1} s, setup {:.1} s, arena {:.2} GB",
        chain.spec.logn,
        nq,
        chain.p.len(),
        chain.dnum(),
        W::BITS,
        std::any::type_name::<M>().rsplit("::").next().unwrap_or("?"),
        tk.keys,
        tk.total,
        t0.elapsed().as_secs_f64(),
        arena.capacity() as f64 / 1e9
    );
    let top_of = |level: u32| nq - 1 - level as usize;
    // fresh sk (or pk) encryptions of `vals` (one message per ct) at `level`
    let fresh = |x: &mut Ct<'_, W>, ex: &mut Executor<'_, W>, encr: &mut Encryptor<'_, W>, prng: &mut Prng, vals: &[Vec<Complex>], level: u32, pk: bool| -> Res<()> {
        let top = top_of(level);
        let sf = ctx.scale.sf(top as i32)?;
        let mut meta = CtMeta::new(level, 1).with_nf(sf);
        meta.slots = slots as u32;
        x.buf.k = vals.len();
        let msgs: Vec<Vec<Vec<u64>>> = vals.iter().map(|v| enc.encode_slots(v, slots, top, Payload::Complex, sf)).collect::<std::result::Result<_, _>>()?;
        if pk {
            encr.encrypt_pk(&ctx, ex, &msgs, meta, x, prng)?;
        } else {
            encr.encrypt_sk(&ctx, &ex.compute, &sec, &msgs, meta, x, prng)?;
        }
        Ok(())
    };
    // a plaintext of `vals` at `level` (eval form) in part 0 of `buf`
    let plain = |buf: &mut CtBuf<'_, Compute, W>, ex: &Executor<'_, W>, vals: &[Complex], level: u32| -> Res<PtMeta> {
        let top = top_of(level);
        let sf = ctx.scale.sf(top as i32)?;
        let rows = enc.encode_slots(vals, slots, top, Payload::Complex, sf)?;
        let flat: Vec<W> = rows.iter().flat_map(|r| r.iter().map(|&x| W::from_u64(x))).collect();
        buf.k = 1;
        buf.upload_rows(&ex.compute, 0, 0, &(0..=top).collect::<Vec<_>>(), &flat)?;
        ntt::forward(&ctx, &ex.compute, buf, Span::Q { top }, Parts::C0)?;
        Ok(PtMeta { level, deg: 1, nf: sf, tag: Default::default() })
    };
    let decrypt = |dec: &mut Decryptor<'_, W>, ex: &Executor<'_, W>, x: &Ct<'_, W>| -> Res<Vec<Vec<Complex>>> { Ok(dec.decrypt_slots(&ctx, &ex.compute, &enc, x, Payload::Complex, Which::Dense)?) };
    let kc = || KsCfg { plans: &plans, source: KskSource::Cached };

    if o.check {
        let mut add = |what: String, bad: Option<usize>, bits: Option<f64>, thr: f64| {
            let pass = bad.is_none_or(|b| b == 0) && bits.is_none_or(|b| b >= thr);
            out.checks.push((what, bad, bits, thr, pass));
        };
        let x = cvec(1, slots, 1.0);
        let y = cvec(2, slots, 1.0);
        // roundtrips per level: sk, pk
        let u = chain.level_unit as u32;
        let mut lv: Vec<u32> = vec![0, (nq as u32 / 2) / u * u, nq as u32 - 3 * u];
        lv.dedup();
        for &l in &lv {
            for pk in [false, true] {
                let [a, ..] = &mut cts[..] else { unreachable!() };
                fresh(a, &mut ex, &mut encr, &mut prng, &[x.clone(), y.clone()], l, pk)?;
                let got = decrypt(&mut dec, &ex, a)?;
                let b = rel_bits(&got[0], &x).min(rel_bits(&got[1], &y));
                add(format!("encrypt_{} -> decrypt, level {l} (top {}), K=2", if pk { "pk" } else { "sk" }, top_of(l)), None, Some(b), if pk { 28.0 } else { 36.0 });
            }
        }
        for &l in &o.levels {
            let top = top_of(l);
            let rows: Vec<usize> = (0..=top).collect();
            let primes = chain.primes();
            let (a, rest) = cts.split_first_mut().ok_or("cts")?;
            let (b, rest) = rest.split_first_mut().ok_or("cts")?;
            let (c, rest) = rest.split_first_mut().ok_or("cts")?;
            let (pbuf, rest) = rest.split_first_mut().ok_or("cts")?;
            let (e, rest) = rest.split_first_mut().ok_or("cts")?;
            let hz: Vec<&mut Ct<'_, W>> = rest.iter_mut().collect();
            fresh(a, &mut ex, &mut encr, &mut prng, std::slice::from_ref(&x), l, false)?;
            fresh(b, &mut ex, &mut encr, &mut prng, std::slice::from_ref(&y), l, false)?;
            c.buf.k = 1;
            e.buf.k = 1;
            let words = |x: &Ct<'_, W>, part| rows_of(x, &ex.compute, 0, part, top);
            let (a0, a1, b0, b1) = (words(a, 0)?, words(a, 1)?, words(b, 0)?, words(b, 1)?);
            let modp = |i: usize| primes[i / n];
            let cmp = |got: Vec<W>, want: &dyn Fn(usize) -> u64| got.iter().enumerate().filter(|(i, g)| g.to_u64() != want(*i)).count();
            // add / sub / negate: exact
            ops.add(&ctx, &mut ex, &sh, std::slice::from_mut(c), &[&*a], &[&*b])?;
            let bad = cmp(rows_of(c, &ex.compute, 0, 0, top)?, &|i| (a0[i].to_u64() + b0[i].to_u64()) % modp(i)) + cmp(rows_of(c, &ex.compute, 0, 1, top)?, &|i| (a1[i].to_u64() + b1[i].to_u64()) % modp(i));
            let dd = decrypt(&mut dec, &ex, c)?;
            let want: Vec<Complex> = x.iter().zip(&y).map(|(p, q)| *p + *q).collect();
            add(format!("add level {l}: words vs host, decrypted"), Some(bad), Some(rel_bits(&dd[0], &want)), 34.0);
            ops.sub(&ctx, &mut ex, &sh, std::slice::from_mut(c), &[&*a], &[&*b])?;
            let bad = cmp(rows_of(c, &ex.compute, 0, 0, top)?, &|i| (a0[i].to_u64() + modp(i) - b0[i].to_u64()) % modp(i));
            add(format!("sub level {l}: words vs host"), Some(bad), None, 0.0);
            // negate = multScalar(-1) on the GPU (degree 2, scale sf^2): decrypted after the realize
            ops.negate(&ctx, &mut ex, &sh, std::slice::from_mut(c), &[&*a])?;
            ops.realize(&ctx, &mut ex, &sh, std::slice::from_mut(c))?;
            let dd = decrypt(&mut dec, &ex, c)?;
            let want: Vec<Complex> = x.iter().map(|p| *p * -1.0).collect();
            add(format!("negate level {l}: realized, decrypted"), None, Some(rel_bits(&dd[0], &want)), 34.0);
            // mult_pt: exact pointwise product, then the decrypted product after the rescale
            let pm = plain(&mut pbuf.buf, &ex, &y, l)?;
            let pw = pbuf.buf.download_rows(&ex.compute, 0, 0, &rows)?;
            let pt = Pt::of(pm, &pbuf.buf.buf);
            ops.mult_pt(&ctx, &mut ex, &sh, std::slice::from_mut(c), &[&*a], &[pt])?;
            let mm = |u: W, v: W, p: u64| ((u.to_u64() as u128 * v.to_u64() as u128) % p as u128) as u64;
            let bad = cmp(rows_of(c, &ex.compute, 0, 0, top)?, &|i| mm(a0[i], pw[i], modp(i))) + cmp(rows_of(c, &ex.compute, 0, 1, top)?, &|i| mm(a1[i], pw[i], modp(i)));
            ops.rescale(&ctx, &mut ex, &sh, std::slice::from_mut(c))?;
            let dd = decrypt(&mut dec, &ex, c)?;
            let xy: Vec<Complex> = x.iter().zip(&y).map(|(p, q)| *p * *q).collect();
            add(format!("mult_pt level {l}: words vs host; rescaled, decrypted"), Some(bad), Some(rel_bits(&dd[0], &xy)), 30.0);
            // tensor: exact (d0 = a0 b0, d1 = a0 b1 + a1 b0, d2 = a1 b1)
            {
                d2.k = 1;
                let mut tr = Triple { c: &mut *c, d2: &mut d2 };
                ops.tensor(&ctx, &mut ex, &sh, &mut tr, &*a, &*b)?;
            }
            let t0w = rows_of(c, &ex.compute, 0, 0, top)?;
            let t1w = rows_of(c, &ex.compute, 0, 1, top)?;
            let t2w = d2.download_rows(&ex.compute, 0, 1, &rows)?;
            let bad = cmp(t0w, &|i| mm(a0[i], b0[i], modp(i)))
                + cmp(t1w, &|i| (mm(a0[i], b1[i], modp(i)) + mm(a1[i], b0[i], modp(i))) % modp(i))
                + cmp(t2w, &|i| mm(a1[i], b1[i], modp(i)));
            add(format!("tensor level {l}: (d0, d1, d2) words vs host"), Some(bad), None, 0.0);
            if top >= 2 * chain.level_unit + 1 {
                // ct x ct with relinearization (Regen and Cached agree), square
                ops.mult(&ctx, &mut ex, KsCfg { plans: &plans, source: KskSource::Regen }, &sh, &relin, c, &*a, &*b)?;
                let regen = rows_of(c, &ex.compute, 0, 0, top)?;
                ops.mult(&ctx, &mut ex, kc(), &sh, &relin, c, &*a, &*b)?;
                let bad = cmp(rows_of(c, &ex.compute, 0, 0, top)?, &|i| regen[i].to_u64());
                ops.rescale(&ctx, &mut ex, &sh, std::slice::from_mut(c))?;
                let dd = decrypt(&mut dec, &ex, c)?;
                add(format!("mult (relin) level {l}: Cached == Regen words; rescaled, decrypted"), Some(bad), Some(rel_bits(&dd[0], &xy)), 28.0);
                ops.square(&ctx, &mut ex, kc(), &sh, &relin, c, &*a)?;
                ops.rescale(&ctx, &mut ex, &sh, std::slice::from_mut(c))?;
                let dd = decrypt(&mut dec, &ex, c)?;
                let xx: Vec<Complex> = x.iter().map(|p| *p * *p).collect();
                add(format!("square level {l}: rescaled, decrypted"), None, Some(rel_bits(&dd[0], &xx)), 28.0);
                // scalars
                let z = Complex::new(0.75, -1.25);
                ops.mult_scalar(&ctx, &mut ex, &sh, std::slice::from_mut(c), &[&*a], &[z])?;
                ops.realize(&ctx, &mut ex, &sh, std::slice::from_mut(c))?;
                let dd = decrypt(&mut dec, &ex, c)?;
                let want: Vec<Complex> = x.iter().map(|p| *p * z).collect();
                add(format!("mult_scalar (0.75 - 1.25i) level {l}: decrypted"), None, Some(rel_bits(&dd[0], &want)), 30.0);
                ops.add_scalar(&ctx, &mut ex, &sh, std::slice::from_mut(c), &[&*a], &[0.375])?;
                let dd = decrypt(&mut dec, &ex, c)?;
                let want: Vec<Complex> = x.iter().map(|p| *p + Complex::new(0.375, 0.0)).collect();
                add(format!("add_scalar 0.375 level {l}: decrypted"), None, Some(rel_bits(&dd[0], &want)), 34.0);
            }
            ops.mult_i(&ctx, &mut ex, &sh, std::slice::from_mut(c), &[&*a])?;
            let dd = decrypt(&mut dec, &ex, c)?;
            let want: Vec<Complex> = x.iter().map(|p| *p * Complex::new(0.0, 1.0)).collect();
            add(format!("mult_i level {l}: decrypted"), None, Some(rel_bits(&dd[0], &want)), 34.0);
            // rotations: plain per step, decrypted vs the rolled vector; hoisted == plain word for word
            let mut plain_words = Vec::new();
            for (i, &st) in STEPS.iter().enumerate() {
                rotate::rotate(&ctx, &mut ex, kc(), &sh, &one[i], &*a, st, c)?;
                plain_words.push(rows_of(c, &ex.compute, 0, 0, top)?);
                let dd = decrypt(&mut dec, &ex, c)?;
                let r = st.rem_euclid(slots as i64) as usize;
                let want: Vec<Complex> = (0..slots).map(|j| x[(j + r) % slots]).collect();
                add(format!("rotate {st} level {l}: decrypted vs rolled"), None, Some(rel_bits(&dd[0], &want)), ROT_BITS);
            }
            {
                let mut outs: Vec<&mut Ct<'_, W>> = hz.into_iter().take(STEPS.len()).collect();
                if outs.len() == STEPS.len() {
                    outs.iter_mut().for_each(|x| x.buf.k = 1);
                    rotate::rotate_hoisted(&ctx, &mut ex, kc(), &sh, &hoist, &*a, &STEPS, &mut outs)?;
                    let mut bad = 0;
                    for (i, h) in outs.iter().enumerate() {
                        bad += cmp(rows_of(h, &ex.compute, 0, 0, top)?, &|j| plain_words[i][j].to_u64());
                    }
                    add(format!("rotate_hoisted {STEPS:?} level {l}: == plain rotations"), Some(bad), None, 0.0);
                }
            }
            rotate::conjugate(&ctx, &mut ex, kc(), &sh, &conj, &*a, c)?;
            let dd = decrypt(&mut dec, &ex, c)?;
            let want: Vec<Complex> = x.iter().map(|p| Complex::new(p.re, -p.im)).collect();
            add(format!("conjugate level {l}: decrypted"), None, Some(rel_bits(&dd[0], &want)), ROT_BITS);
            // drop
            ops.add(&ctx, &mut ex, &sh, std::slice::from_mut(c), &[&*a], &[&*a])?;
            let to = (nq - 3 * chain.level_unit) as u32;
            if to > l {
                ops.drop_to_level(&sh, std::slice::from_mut(c), to);
                let dd = decrypt(&mut dec, &ex, c)?;
                let want: Vec<Complex> = x.iter().map(|p| *p + *p).collect();
                add(format!("drop_to_level {l} -> {to}: decrypted"), None, Some(rel_bits(&dd[0], &want)), 34.0);
            }
            let _ = &e;
        }
        // depth: x <- x * 0.9 (scalar product + rescale) from the top to the bottom; every level decrypted
        {
            let (a, rest) = cts.split_first_mut().ok_or("cts")?;
            let c = &mut rest[0];
            fresh(a, &mut ex, &mut encr, &mut prng, std::slice::from_ref(&x), 0, false)?;
            c.buf.k = 1;
            let mut worst = f64::MAX;
            let mut lvls = 0;
            let mut want = x.clone();
            let mut cur_is_a = true;
            loop {
                let (src, dst): (&Ct<'_, W>, &mut Ct<'_, W>) = if cur_is_a { (&*a, &mut *c) } else { (&*c, &mut *a) };
                if top_of(src.meta.level) < 2 * chain.level_unit {
                    break;
                }
                ops.mult_scalar(&ctx, &mut ex, &sh, std::slice::from_mut(dst), &[src], &[Complex::new(0.9, 0.0)])?;
                ops.realize(&ctx, &mut ex, &sh, std::slice::from_mut(dst))?;
                want.iter_mut().for_each(|w| *w = *w * Complex::new(0.9, 0.0));
                let dd = decrypt(&mut dec, &ex, dst)?;
                worst = worst.min(rel_bits(&dd[0], &want));
                lvls += 1;
                cur_is_a = !cur_is_a;
            }
            add(format!("x *= 0.9 + rescale, top to bottom ({lvls} rescales), worst level decrypted"), None, Some(worst), 28.0);
        }
    }

    if o.bench {
        let ev0 = Event::timing(&dev)?;
        let ev1 = Event::timing(&dev)?;
        let level = 0u32;
        let top = top_of(level);
        for &k in &o.ks {
            let vals: Vec<Vec<Complex>> = (0..k).map(|i| cvec(100 + i as u64, slots, 1.0)).collect();
            let (a, rest) = cts.split_first_mut().ok_or("cts")?;
            let (b, rest) = rest.split_first_mut().ok_or("cts")?;
            let (c, rest) = rest.split_first_mut().ok_or("cts")?;
            let pbuf = &mut rest[0];
            // fixed inputs: the prng restarts per rig, so every strategy sees the same ciphertexts
            let mut ip = test_prng(o.seed ^ 0xbe);
            fresh(a, &mut ex, &mut encr, &mut ip, &vals, level, false)?;
            fresh(b, &mut ex, &mut encr, &mut ip, &vals, level, false)?;
            c.buf.k = k;
            d2.k = k;
            let pm = plain(&mut pbuf.buf, &ex, &vals[0], level)?;
            let pt = Pt::of(pm, &pbuf.buf.buf);
            let reps = o.reps;
            macro_rules! time {
                ($name:expr, $body:expr, $hash:expr) => {{
                    let mut ms = Vec::new();
                    for r in 0..reps + 2 {
                        ev0.record(&ex.compute)?;
                        $body;
                        ev1.record(&ex.compute)?;
                        ev1.sync()?;
                        if r >= 2 {
                            ms.push(ev0.elapsed_ms(&ev1)? as f64);
                        }
                    }
                    out.bench.push(($name.to_string(), k, median(ms)));
                    let h: &Ct<'_, W> = $hash;
                    let mut words = Vec::new();
                    for cc in 0..k {
                        for part in 0..2 {
                            words.extend(rows_of(h, &ex.compute, cc, part, top_of(h.meta.level))?);
                        }
                    }
                    out.hashes.push((format!("{} K={k}", $name), fnv(&words)));
                }};
            }
            time!("mult_pt", ops.mult_pt(&ctx, &mut ex, &sh, std::slice::from_mut(c), &[&*a], &[pt])?, &*c);
            time!(
                "tensor",
                {
                    let mut tr = Triple { c: &mut *c, d2: &mut d2 };
                    ops.tensor(&ctx, &mut ex, &sh, &mut tr, &*a, &*b)?
                },
                &*c
            );
            time!("mult_scalar", ops.mult_scalar(&ctx, &mut ex, &sh, std::slice::from_mut(c), &[&*a], &[Complex::new(0.3, 0.0)])?, &*c);
            time!("add", ops.add(&ctx, &mut ex, &sh, std::slice::from_mut(c), &[&*a], &[&*b])?, &*c);
            // NTT: inverse + forward of both parts over Q (the input is restored, so the hash checks the roundtrip)
            time!(
                "ntt inv+fwd",
                {
                    ntt::inverse(&ctx, &ex.compute, &mut b.buf, Span::Q { top }, Parts::Both)?;
                    ntt::forward(&ctx, &ex.compute, &mut b.buf, Span::Q { top }, Parts::Both)?
                },
                &*b
            );
            time!(
                "rescale",
                {
                    ops.mult_pt(&ctx, &mut ex, &sh, std::slice::from_mut(c), &[&*a], &[pt])?;
                    ops.rescale(&ctx, &mut ex, &sh, std::slice::from_mut(c))?
                },
                &*c
            );
            time!("rotate (KS, Cached)", rotate::rotate(&ctx, &mut ex, kc(), &sh, &one[0], &*a, STEPS[0], c)?, &*c);
            time!("rotate (KS, Regen)", rotate::rotate(&ctx, &mut ex, KsCfg { plans: &plans, source: KskSource::Regen }, &sh, &one[0], &*a, STEPS[0], c)?, &*c);
            time!("mult+relin (Regen)", ops.mult(&ctx, &mut ex, KsCfg { plans: &plans, source: KskSource::Regen }, &sh, &relin, c, &*a, &*b)?, &*c);
        }
        // mult_pt alone includes no rescale; subtract it from "rescale" for the rescale-only figure
        let get = |nm: &str, k: usize, b: &[(String, usize, f64)]| b.iter().find(|r| r.0 == nm && r.1 == k).map(|r| r.2);
        let extra: Vec<(String, usize, f64)> = o.ks.iter().filter_map(|&k| Some(("rescale only".to_string(), k, get("rescale", k, &out.bench)? - get("mult_pt", k, &out.bench)?))).collect();
        out.bench.extend(extra);
    }
    ex.compute.sync()?;
    Ok(out)
}

pub fn print(tag: &str, r: &RigOut) -> bool {
    println!("[{tag}] {}", r.setup);
    let mut ok = true;
    for (what, bad, bits, thr, pass) in &r.checks {
        ok &= pass;
        let b = bad.map(|b| format!("{b} mismatching words")).unwrap_or_default();
        let p = bits.map(|x| format!("{x:.1} bits (>= {thr})")).unwrap_or_default();
        println!("[{tag}] {:<4} {what}: {b}{}{p}", if *pass { "ok" } else { "FAIL" }, if bad.is_some() && bits.is_some() { ", " } else { "" });
    }
    for (op, k, ms) in &r.bench {
        println!("[{tag}] bench {op:<22} K={k}: {ms:.4} ms");
    }
    ok
}
