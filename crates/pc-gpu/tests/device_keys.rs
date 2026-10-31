//! Key / encrypt / decrypt GPU gates (run through tools/gpu.sh, one test at a time: `--test-threads=1`):
//! - encrypt -> decrypt round trip <= 2^-40 relative at every level, real + complex, secret- and public-key, K = 2;
//! - keygen: every generated key satisfies the switch relation (packed and dense layouts agree bit for bit);
//! - keygen rotate test: rotations (full and banded keys) and conjugation through the host reference key switch,
//!   decrypt after rotate == rotated plaintext; relin key: u0 + u1 s == x s^2 + small;
//! - encapsulation keys: dense -> sparse at the bottom level, sparse -> dense at the bottom and at the top level.
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::keys::{Prng, SecretKey};
use pc_core::modmul::{addmod, mulmod, submod, Barrett64};
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::{Arena, Device};
use pc_gpu::decrypt::{Decryptor, Which};
use pc_gpu::encrypt::Encryptor;
use pc_gpu::keygen::{shapes_of, test_prng, KeyGen};
use pc_gpu::keys::hostref::{centered_crt2, HostChain};
use pc_gpu::keys::{DevKsk, DevSecret, KeyRole, KeySet, KeySetSpec, KskLayout, KskShape, RotSpec};
use pc_gpu::{Context, Ct, CtPool, Executor, MemCfg, MemPlan, PoolClass, PoolCt};
use pc_shadow::CtMeta;
use rayon::prelude::*;

type W = u32;
type M = Barrett64;

struct Rt<'a> {
    chain: Chain,
    ctx: Context<'a, W, M>,
    ex: Executor<'a, W>,
    pool: CtPool<'a, W>,
    arena: &'a Arena,
    enc: Encoder,
}

/// Build the runtime for n32 with `extra` bytes reserved for the test's own carves, run `f`.
fn with_rt(extra: usize, f: impl FnOnce(&mut Rt<'_>)) {
    let dev = Device::open().expect("Device::open (run through tools/gpu.sh)");
    let chain = Chain::generate(&ChainSpec::n32()).unwrap();
    let cfg = MemCfg { k_max: 2, pool: vec![PoolClass { k: 1, count: 6 }, PoolClass { k: 2, count: 2 }], events: 8, scratch_cts: 1 };
    let mut plan = MemPlan::build::<W, M>(&chain, cfg).unwrap();
    plan.reserve("device_keys.test", extra);
    let arena = plan.allocate(&dev).unwrap();
    let ex = Executor::<W>::new(&dev, &arena, &plan).unwrap();
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &plan).unwrap();
    let pool = CtPool::<W>::new(&arena, &plan).unwrap();
    let enc = Encoder::new(&chain);
    let mut rt = Rt { chain, ctx, ex, pool, arena: &arena, enc };
    f(&mut rt);
}

fn rand_vals(seed: u64, n: usize, complex: bool) -> Vec<Complex> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut r = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    (0..n).map(|_| Complex::new(r(), if complex { r() } else { 0.0 })).collect()
}

fn max_rel(got: &[Complex], want: &[Complex]) -> f64 {
    let mx = want.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-300);
    got.iter().zip(want).map(|(a, b)| (*a - *b).abs()).fold(0.0, f64::max) / mx
}

fn sk_sample(tag: u32, n: usize) -> SecretKey {
    let mut p = Prng::from_seed([tag, 0xabc, 1, 2, 3, 4, 5, 6], 7);
    SecretKey::sample(&mut p, n).unwrap()
}

fn meta_at(rt: &Rt<'_>, level: u32, slots: usize) -> (CtMeta, usize, f64) {
    let top = rt.ctx.top_of(level).unwrap();
    let sf = rt.ctx.scale.sf(top as i32).unwrap();
    let mut m = CtMeta::new(level, 1).with_nf(sf);
    m.slots = slots as u32;
    (m, top, sf)
}

#[test]
#[ignore = "GPU: tools/gpu.sh <log> <device_keys test binary> --ignored --test-threads=1"]
fn roundtrip_every_level() {
    let lay = pc_gpu::Layout::from_chain(&Chain::generate(&ChainSpec::n32()).unwrap());
    let extra = DevSecret::<W>::bytes(&lay) + Encryptor::<W>::bytes(&lay, true) + Decryptor::<W>::bytes(&lay, 2, 2) + (1 << 20);
    with_rt(extra, |rt| {
        let s = &rt.ex.compute;
        let n = rt.chain.n;
        let mut sec = DevSecret::<W>::carve(rt.arena, rt.ctx.lay).unwrap();
        sec.load(&rt.ctx, s, &sk_sample(1, n)).unwrap();
        let mut prng = test_prng(11);
        let mut encr = Encryptor::<W>::new(rt.arena, s, rt.ctx.lay, true, &mut prng).unwrap();
        let mut dec = Decryptor::<W>::new(rt.arena, s, rt.ctx.lay, rt.chain.level_unit, 2).unwrap();
        dec.load_secret(s, &sec).unwrap();
        encr.gen_public_key(&rt.ctx, s, &sec, &mut prng).unwrap();
        // CtMeta::level counts primes dropped: the grid is every level_unit
        let unit = rt.chain.level_unit as u32;
        let levels = (rt.chain.q.len() / rt.chain.level_unit) as u32;
        let slots = n / 2;
        let mut worst = [0.0f64; 3];
        for level in (0..levels).map(|i| i * unit) {
            let (meta, top, sf) = meta_at(rt, level, slots);
            for (pi, payload) in [Payload::Complex, Payload::Real].into_iter().enumerate() {
                let vals: Vec<Vec<Complex>> = (0..2).map(|c| rand_vals(1000 * level as u64 + 10 * pi as u64 + c, slots, payload == Payload::Complex)).collect();
                let msgs: Vec<Vec<Vec<u64>>> = vals.iter().map(|v| rt.enc.encode_slots(v, slots, top, payload, sf).unwrap()).collect();
                let mut ctp = rt.pool.acquire(2, meta).unwrap();
                let ct = ctp.get_mut().unwrap();
                encr.encrypt_sk(&rt.ctx, &rt.ex.compute, &sec, &msgs, meta, ct, &mut prng).unwrap();
                let got = dec.decrypt_slots(&rt.ctx, &rt.ex.compute, &rt.enc, ct, payload, Which::Dense).unwrap();
                let e = (0..2).map(|c| max_rel(&got[c], &vals[c])).fold(0.0, f64::max);
                worst[pi] = worst[pi].max(e);
                assert!(e <= 2f64.powi(-40), "sk level {level} {payload:?}: rel err {e:e} = 2^{:.1}", e.log2());
                if payload == Payload::Real {
                    encr.encrypt_pk(&rt.ctx, &mut rt.ex, &msgs, meta, ct, &mut prng).unwrap();
                    let got = dec.decrypt_slots(&rt.ctx, &rt.ex.compute, &rt.enc, ct, payload, Which::Dense).unwrap();
                    let e = (0..2).map(|c| max_rel(&got[c], &vals[c])).fold(0.0, f64::max);
                    worst[2] = worst[2].max(e);
                    // public-key noise v e + e0 + e1 s is ~sqrt(N) above the secret-key one: 2^-30 bound
                    assert!(e <= 2f64.powi(-30), "pk level {level}: rel err {e:e} = 2^{:.1}", e.log2());
                }
            }
        }
        println!(
            "[m5a] roundtrip {levels} levels x K=2: sk complex max rel 2^{:.1}, sk real 2^{:.1}, pk real 2^{:.1}",
            worst[0].log2(),
            worst[1].log2(),
            worst[2].log2()
        );
    });
}

/// Host eval rows of s, s^2-free: s and s_sparse by prime id.
fn host_secret(hc: &HostChain, sk: &SecretKey) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
    let ids: Vec<usize> = (0..hc.primes.len()).collect();
    let s: Vec<i64> = sk.s.iter().map(|&x| x as i64).collect();
    let sp: Vec<i64> = sk.s_sparse.iter().map(|&x| x as i64).collect();
    (hc.eval_rows(&s, &ids), hc.eval_rows(&sp, &ids))
}

/// (s_new, s_old, sq) host rows of a key role.
fn role_secrets(role: KeyRole, n: usize, s: &[Vec<u64>], sp: &[Vec<u64>]) -> (Vec<Vec<u64>>, Vec<Vec<u64>>, bool) {
    match role {
        KeyRole::Relin => (s.to_vec(), s.to_vec(), true),
        KeyRole::EncDenseToSparse => (sp.to_vec(), s.to_vec(), false),
        KeyRole::EncSparseToDense => (s.to_vec(), sp.to_vec(), false),
        KeyRole::Rot { .. } | KeyRole::Conj | KeyRole::SparseRot { .. } => {
            let g = role.galois(n).unwrap();
            let gi = pc_core::galois::galois_inverse(g, n).unwrap();
            let old = if matches!(role, KeyRole::SparseRot { .. }) { sp.to_vec() } else { s.to_vec() };
            (s.par_iter().map(|r| pc_core::galois::apply_eval(r, gi).unwrap()).collect(), old, false)
        }
    }
}

fn test_spec() -> KeySetSpec {
    KeySetSpec {
        relin: true,
        conj: true,
        rots: vec![RotSpec { step: 1, q_top: None }, RotSpec { step: -3, q_top: None }, RotSpec { step: 5, q_top: Some(21) }],
        encaps: true,
    }
}

#[test]
#[ignore = "GPU: tools/gpu.sh <log> <device_keys test binary> --ignored --test-threads=1"]
fn keygen_relation_and_layouts() {
    let chain = Chain::generate(&ChainSpec::n32()).unwrap();
    let lay = pc_gpu::Layout::from_chain(&chain);
    let spec = test_spec();
    let packed = KskLayout::packed_for::<W>(&chain).unwrap();
    let shapes: Vec<KskShape> = spec.keys(&chain).unwrap().into_iter().map(|k| k.1).collect();
    let extra = spec.bytes::<W>(&chain, packed).unwrap() + spec.bytes::<W>(&chain, KskLayout::Dense).unwrap() + DevSecret::<W>::bytes(&lay) + KeyGen::<W>::bytes(&lay, &shapes) + (1 << 20);
    with_rt(extra, |rt| {
        let n = rt.chain.n;
        let sk = sk_sample(2, n);
        let mut sec = DevSecret::<W>::carve(rt.arena, rt.ctx.lay).unwrap();
        sec.load(&rt.ctx, &rt.ex.compute, &sk).unwrap();
        let mut kp = KeySet::<W>::carve(rt.arena, &rt.chain, &spec, packed).unwrap();
        let mut kd = KeySet::<W>::carve(rt.arena, &rt.chain, &spec, KskLayout::Dense).unwrap();
        let mut kg = KeyGen::<W>::new(rt.arena, &rt.ex.compute, &rt.chain, &shapes_of(&kp)).unwrap();
        let tp = kg.generate_set(&rt.ctx, &mut rt.ex, &sec, &mut kp, &mut test_prng(21)).unwrap();
        let td = kg.generate_set(&rt.ctx, &mut rt.ex, &sec, &mut kd, &mut test_prng(21)).unwrap();
        println!("[m5a] keygen packed: {} keys {:.3} s (host {:.3}, upload {:.3}, device {:.3}); {} B", tp.keys, tp.total, tp.host_sample, tp.upload, tp.device, kp.bytes());
        println!("[m5a] keygen dense:  {} keys {:.3} s; {} B", td.keys, td.total, kd.bytes());
        let hc = HostChain::new(&rt.chain);
        let (s, sp) = host_secret(&hc, &sk);
        let mut worst = 0;
        for (a, b) in kp.iter().zip(kd.iter()) {
            assert_eq!(a.role, b.role);
            let (pb, pa) = a.download_dense(&rt.ex.compute, &rt.chain).unwrap();
            let (db, da) = b.download_dense(&rt.ex.compute, &rt.chain).unwrap();
            let mism = pb.iter().zip(&db).filter(|(x, y)| x != y).count() + pa.iter().zip(&da).filter(|(x, y)| x != y).count();
            let (sn, so, sq) = role_secrets(a.role, n, &s, &sp);
            let (e, rows) = hc.relation_max_err(&a.shape, &pb, &pa, &sn, &so, sq);
            println!("[m5a] key {:?} shape q_top {} ndig {}: {rows} rows, max |e| {e}, packed vs dense mismatches {mism}", a.role, a.shape.q_top, a.shape.ndig);
            assert_eq!(mism, 0);
            assert!(e <= 64, "{:?}: max |e| {e}", a.role);
            worst = worst.max(e);
        }
        println!("[m5a] keygen relation: {} keys, max |e| {worst}", kp.len());
    });
}

fn ct_rows(rt: &Rt<'_>, ct: &Ct<'_, W>, c: usize, part: usize, top: usize) -> Vec<Vec<u64>> {
    let rows: Vec<usize> = (0..=top).collect();
    let v = ct.buf.download_rows(&rt.ex.compute, c, part, &rows).unwrap();
    v.chunks(rt.chain.n).map(|r| r.iter().map(|&x| x as u64).collect()).collect()
}

fn put_rows(rt: &Rt<'_>, ct: &mut Ct<'_, W>, c: usize, part: usize, rows_v: &[Vec<u64>]) {
    let rows: Vec<usize> = (0..rows_v.len()).collect();
    let flat: Vec<W> = rows_v.iter().flat_map(|r| r.iter().map(|&x| x as W)).collect();
    ct.buf.upload_rows(&rt.ex.compute, c, part, &rows, &flat).unwrap();
}

/// Key switch of ct (eval rows Q 0..=top) with `key`, then sigma_g when the role has one: the rotated / conjugated /
/// switched ciphertext, written back into `ct`.
fn switch(rt: &Rt<'_>, hc: &HostChain, key: &DevKsk<'_, W>, ct: &mut Ct<'_, W>, top: usize) {
    let (b, a) = key.download_dense(&rt.ex.compute, &rt.chain).unwrap();
    let c0 = ct_rows(rt, ct, 0, 0, top);
    let c1 = ct_rows(rt, ct, 0, 1, top);
    let (u0, u1) = hc.key_switch(&c1, top, &key.shape, &b, &a).unwrap();
    let mut r0: Vec<Vec<u64>> = (0..=top).map(|t| (0..rt.chain.n).map(|i| addmod(c0[t][i], u0[t][i], hc.primes[t])).collect()).collect();
    let mut r1 = u1;
    if let Some(g) = key.role.galois(rt.chain.n) {
        r0 = r0.par_iter().map(|r| pc_core::galois::apply_eval(r, g).unwrap()).collect();
        r1 = r1.par_iter().map(|r| pc_core::galois::apply_eval(r, g).unwrap()).collect();
    }
    put_rows(rt, ct, 0, 0, &r0);
    put_rows(rt, ct, 0, 1, &r1);
}

/// A fresh encryption of `vals` (complex, all slots) at `level` under `sec`.
fn fresh<'r, 'a>(rt: &'r Rt<'a>, encr: &mut Encryptor<'_, W>, prng: &mut Prng, vals: &[Complex], level: u32, sec: &DevSecret<'_, W>) -> (PoolCt<'r, 'a, W>, usize) {
    let slots = rt.chain.n / 2;
    let (meta, top, sf) = meta_at(rt, level, slots);
    let m = rt.enc.encode_slots(vals, slots, top, Payload::Complex, sf).unwrap();
    let mut ctp = rt.pool.acquire(1, meta).unwrap();
    encr.encrypt_sk(&rt.ctx, &rt.ex.compute, sec, &[m], meta, ctp.get_mut().unwrap(), prng).unwrap();
    (ctp, top)
}

#[test]
#[ignore = "GPU: tools/gpu.sh <log> <device_keys test binary> --ignored --test-threads=1"]
fn keygen_rotate_relin_encaps() {
    let chain = Chain::generate(&ChainSpec::n32()).unwrap();
    let lay = pc_gpu::Layout::from_chain(&chain);
    let spec = test_spec();
    let packed = KskLayout::packed_for::<W>(&chain).unwrap();
    let shapes: Vec<KskShape> = spec.keys(&chain).unwrap().into_iter().map(|k| k.1).collect();
    let extra = spec.bytes::<W>(&chain, packed).unwrap()
        + 2 * DevSecret::<W>::bytes(&lay)
        + KeyGen::<W>::bytes(&lay, &shapes)
        + Encryptor::<W>::bytes(&lay, false)
        + 2 * Decryptor::<W>::bytes(&lay, 2, 2)
        + (1 << 20);
    with_rt(extra, |rt| {
        let n = rt.chain.n;
        let slots = n / 2;
        let sk = sk_sample(3, n);
        let mut sec = DevSecret::<W>::carve(rt.arena, rt.ctx.lay).unwrap();
        sec.load(&rt.ctx, &rt.ex.compute, &sk).unwrap();
        // the same pair with the roles of s and s_sparse swapped: encrypts directly under the sparse secret
        let mut sec_sw = DevSecret::<W>::carve(rt.arena, rt.ctx.lay).unwrap();
        sec_sw.load(&rt.ctx, &rt.ex.compute, &SecretKey { s: sk.s_sparse.clone(), s_sparse: sk.s.clone() }).unwrap();
        let mut ks = KeySet::<W>::carve(rt.arena, &rt.chain, &spec, packed).unwrap();
        let mut kg = KeyGen::<W>::new(rt.arena, &rt.ex.compute, &rt.chain, &shapes_of(&ks)).unwrap();
        let mut prng = test_prng(31);
        kg.generate_set(&rt.ctx, &mut rt.ex, &sec, &mut ks, &mut prng).unwrap();
        let rt: &Rt<'_> = rt;
        let mut encr = Encryptor::<W>::new(rt.arena, &rt.ex.compute, rt.ctx.lay, false, &mut prng).unwrap();
        let mut dec = Decryptor::<W>::new(rt.arena, &rt.ex.compute, rt.ctx.lay, rt.chain.level_unit, 2).unwrap();
        dec.load_secret(&rt.ex.compute, &sec).unwrap();
        let mut dec_sw = Decryptor::<W>::new(rt.arena, &rt.ex.compute, rt.ctx.lay, rt.chain.level_unit, 2).unwrap();
        dec_sw.load_secret(&rt.ex.compute, &sec_sw).unwrap();
        let hc = HostChain::new(&rt.chain);
        let vals = rand_vals(77, slots, true);
        let rotl = |v: &[Complex], r: i64| -> Vec<Complex> { (0..v.len()).map(|j| v[(j as i64 + r).rem_euclid(v.len() as i64) as usize]).collect() };

        let mut report = Vec::new();
        // rotations: full keys at the top level, the banded key (q_top 21) at the highest level it covers
        // levels in primes dropped (top = 53 - level); 32 -> top 21, the banded key's q_top
        for (step, level) in [(1i64, 0u32), (-3, 0), (5, 32), (1, 20)] {
            let (mut ctp, top) = fresh(rt, &mut encr, &mut prng, &vals, level, &sec);
            let key = ks.rot(step).unwrap();
            let ct = ctp.get_mut().unwrap();
            switch(rt, &hc, key, ct, top);
            let got = dec.decrypt_slots(&rt.ctx, &rt.ex.compute, &rt.enc, ct, Payload::Complex, Which::Dense).unwrap();
            let e = max_rel(&got[0], &rotl(&vals, step));
            report.push(format!("rot {step:+} @ level {level} (top {top}, key q_top {}): 2^{:.1}", key.shape.q_top, e.log2()));
            assert!(e <= 2f64.powi(-25), "rotate {step} at level {level}: rel err 2^{:.1}", e.log2());
        }
        {
            let (mut ctp, top) = fresh(rt, &mut encr, &mut prng, &vals, 0, &sec);
            let ct = ctp.get_mut().unwrap();
            switch(rt, &hc, ks.conj().unwrap(), ct, top);
            let got = dec.decrypt_slots(&rt.ctx, &rt.ex.compute, &rt.enc, ct, Payload::Complex, Which::Dense).unwrap();
            let want: Vec<Complex> = vals.iter().map(|v| Complex::new(v.re, -v.im)).collect();
            let e = max_rel(&got[0], &want);
            report.push(format!("conj @ level 0: 2^{:.1}", e.log2()));
            assert!(e <= 2f64.powi(-25), "conj: rel err 2^{:.1}", e.log2());
        }
        // key-switch noise of every key in coefficient form: x = c1 of a fresh ciphertext at the top the key
        // covers; |u0 + u1 s_new - x s_old| over the bottom pair, centered (relin: s_old = s^2)
        {
            let (s, sp) = host_secret(&hc, &sk);
            for key in ks.iter() {
                let lvl = (53 - key.shape.q_top) as u32;
                let (ctp, top) = fresh(rt, &mut encr, &mut prng, &vals, lvl, &sec);
                let x = ct_rows(rt, &ctp, 0, 1, top);
                let (b, a) = key.download_dense(&rt.ex.compute, &rt.chain).unwrap();
                let (u0, u1) = hc.key_switch(&x, top, &key.shape, &b, &a).unwrap();
                let (sn, so, sq) = role_secrets(key.role, n, &s, &sp);
                let r: Vec<Vec<u64>> = (0..2)
                    .map(|t| {
                        let p = hc.primes[t];
                        let mut v: Vec<u64> = (0..n)
                            .map(|i| {
                                let o = if sq { mulmod(so[t][i], so[t][i], p) } else { so[t][i] };
                                submod(addmod(u0[t][i], mulmod(u1[t][i], sn[t][i], p), p), mulmod(x[t][i], o, p), p)
                            })
                            .collect();
                        hc.ntt[t].inv(&mut v);
                        v
                    })
                    .collect();
                let c = centered_crt2(&r[0], &r[1], hc.primes[0], hc.primes[1]);
                let e = c.iter().map(|v| v.unsigned_abs()).max().unwrap();
                let rms = (c.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64).sqrt();
                report.push(format!("ks noise {:?} @ top {top}: max |u0 + u1 s_new - x s_old| = 2^{:.1}, rms 2^{:.1}", key.role, (e as f64).log2(), rms.log2()));
                assert!(e < 1 << 30, "{:?} key-switch noise {e}", key.role);
            }
        }
        // encapsulation: dense -> sparse at the bottom level, then sparse -> dense
        let bot = (rt.chain.q.len() - rt.chain.level_unit) as u32;
        {
            let (mut ctp, top) = fresh(rt, &mut encr, &mut prng, &vals, bot, &sec);
            let bk = ks.bts().unwrap();
            let ct = ctp.get_mut().unwrap();
            switch(rt, &hc, &bk.enc_dense_to_sparse, ct, top);
            let got = dec.decrypt_slots(&rt.ctx, &rt.ex.compute, &rt.enc, ct, Payload::Complex, Which::Sparse).unwrap();
            let e1 = max_rel(&got[0], &vals);
            switch(rt, &hc, &bk.enc_sparse_to_dense, ct, top);
            let got = dec.decrypt_slots(&rt.ctx, &rt.ex.compute, &rt.enc, ct, Payload::Complex, Which::Dense).unwrap();
            let e2 = max_rel(&got[0], &vals);
            report.push(format!("encaps @ level {bot} (top {top}): dense->sparse 2^{:.1}, ->dense again 2^{:.1}", e1.log2(), e2.log2()));
            assert!(e1 <= 2f64.powi(-25) && e2 <= 2f64.powi(-25), "encaps round trip: 2^{:.1}, 2^{:.1}", e1.log2(), e2.log2());
        }
        {
            // sparse -> dense at the top level (the raised level of the bootstrap)
            let (mut ctp, top) = fresh(rt, &mut encr, &mut prng, &vals, 0, &sec_sw);
            let ct = ctp.get_mut().unwrap();
            let got = dec_sw.decrypt_slots(&rt.ctx, &rt.ex.compute, &rt.enc, ct, Payload::Complex, Which::Dense).unwrap();
            let e0 = max_rel(&got[0], &vals);
            switch(rt, &hc, &ks.bts().unwrap().enc_sparse_to_dense, ct, top);
            let got = dec.decrypt_slots(&rt.ctx, &rt.ex.compute, &rt.enc, ct, Payload::Complex, Which::Dense).unwrap();
            let e = max_rel(&got[0], &vals);
            report.push(format!("sparse->dense @ level 0 (top {top}): 2^{:.1} (fresh under s_sparse 2^{:.1})", e.log2(), e0.log2()));
            assert!(e <= 2f64.powi(-25), "sparse->dense top: 2^{:.1}", e.log2());
        }
        for r in report {
            println!("[m5a] {r}");
        }
    });
}
