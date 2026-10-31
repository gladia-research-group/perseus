//! `pc-check keys <g0 dump>`: load every key of the dump and its secret into pc-gpu's device layout and check them.
//!
//! Gates (all 0 mismatches / within bound): the chain's primes and digits; every key's `b` through the packed layout
//! (`ksk_unpack`) == the dumped `b`; `a` regenerated from the seed (`ksk_expand`) == the dumped dense `a`; the switch
//! relation of every key (max |e| <= 64); a rotation / conjugation / dense<->sparse switch with the imported key and with a
//! key generated here from the same secret decrypt to the expected slots; the whole set regenerated on the device (timed).
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::galois;
use pc_core::keys::{KskMaterial, KskSeed, Prng, SecretKey};
use pc_core::modmul::{addmod, invmod, mulmod, Barrett64};
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, DBuf, Device};
use pc_gpu::decrypt::{Decryptor, Which};
use pc_gpu::encrypt::Encryptor;
use pc_gpu::keygen::{shapes_of, KeyGen};
use pc_gpu::keys::hostref::HostChain;
use pc_gpu::keys::{import_material, DevKsk, DevSecret, KeyRole, KeySet, KeySetSpec, KskLayout, KskShape, KskStore, RotSpec};
use pc_gpu::{Context, Ct, CtPool, Executor, Layout, MemCfg, MemPlan, PoolClass};
use pc_kernels::{KskExpandArgs, KskUnpackArgs};
use crate::dump_g0::{npy_header, G0};
use pc_shadow::CtMeta;
use rayon::prelude::*;
use serde_json::Value;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

type W = u32;
type M = Barrett64;
type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn u64v(v: &Value) -> Vec<u64> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).collect()).unwrap_or_default()
}

fn read_u32(p: &Path) -> Res<Vec<u32>> {
    let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn read_sk(dir: &Path, n: usize) -> Res<Vec<i8>> {
    let p = dir.join("sk.npy");
    let (descr, shape, off) = npy_header(&p)?;
    if !descr.ends_with("i1") || shape != [n] {
        return Err(format!("sk.npy: dtype {descr} shape {shape:?}, expected int8 [{n}]").into());
    }
    let b = std::fs::read(&p)?;
    Ok(b[off..off + n].iter().map(|&x| x as i8).collect())
}

/// Role of a dump key-map name.
fn role_of(name: &str, n: usize) -> Res<KeyRole> {
    if name == "mult" {
        return Ok(KeyRole::Relin);
    }
    let g: u64 = name.parse()?;
    let m = 2 * n as u64;
    Ok(if g == m - 1 {
        KeyRole::Conj
    } else if g == m - 2 {
        KeyRole::EncDenseToSparse
    } else if g == m - 4 {
        KeyRole::EncSparseToDense
    } else {
        let r = galois::galois_to_rotation(g, n)?.ok_or_else(|| format!("key {name}: not a power of 5 mod 2N"))?;
        KeyRole::Rot { step: r as i64, galois: g }
    })
}

fn material(g0: &G0, name: &str) -> Res<KskMaterial> {
    let k = &g0.params["keys"][name];
    let shape = u64v(&k["shape"]);
    let seed: Vec<u32> = u64v(&k["a_seed"]).into_iter().map(|x| x as u32).collect();
    let seed: [u32; 8] = seed.try_into().map_err(|_| format!("key {name}: seed is not 8 words"))?;
    let moduli: Vec<u32> = u64v(&k["b_moduli"][0]).into_iter().map(|x| x as u32).collect();
    let file = if name == "mult" { "keys/mult.b.bin".to_string() } else { format!("keys/auto{name}.b.bin") };
    let b = read_u32(&g0.dir.join(file))?;
    Ok(KskMaterial::new(KskSeed(seed), shape[0] as usize, shape[2] as usize, moduli, b)?)
}

/// Host eval rows (by prime id) of s_new / s_old for a role.
fn role_secrets(role: KeyRole, n: usize, s: &[Vec<u64>], sp: &[Vec<u64>]) -> Res<(Vec<Vec<u64>>, Vec<Vec<u64>>, bool)> {
    Ok(match role {
        KeyRole::Relin => (s.to_vec(), s.to_vec(), true),
        KeyRole::EncDenseToSparse => (sp.to_vec(), s.to_vec(), false),
        KeyRole::EncSparseToDense => (s.to_vec(), sp.to_vec(), false),
        KeyRole::Rot { .. } | KeyRole::Conj | KeyRole::SparseRot { .. } => {
            let g = role.galois(n).ok_or("no galois")?;
            let gi = galois::galois_inverse(g, n)?;
            let old = if matches!(role, KeyRole::SparseRot { .. }) { sp.to_vec() } else { s.to_vec() };
            (s.par_iter().map(|r| galois::apply_eval(r, gi)).collect::<Result<Vec<_>, _>>()?, old, false)
        }
    })
}

/// Recover s_sparse from the sparse->dense key: X = INTT(b + a s) of digit 0 is e (+ (P mod q_t) s_sparse on digit-0
/// rows); e from a row outside digit 0, then s_sparse = (X_0 - e) / (P mod q0) centered.
fn recover_sparse(hc: &HostChain, m: &KskMaterial, s: &[Vec<u64>], digit1_row: usize) -> Res<Vec<i8>> {
    let n = hc.n;
    let x = |t: usize| -> Vec<u64> {
        let p = hc.primes[t];
        let a = m.a_row(0, t);
        let b = m.b_row(0, t);
        let mut v: Vec<u64> = (0..n).map(|i| addmod(b[i] as u64, mulmod(a[i] as u64, s[t][i], p), p)).collect();
        hc.ntt[t].inv(&mut v);
        v
    };
    let (x0, xe) = (x(0), x(digit1_row));
    let (q0, qe) = (hc.primes[0], hc.primes[digit1_row]);
    let pinv = invmod(hc.p_mod_q[0], q0);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let e = if xe[i] > qe / 2 { xe[i] as i64 - qe as i64 } else { xe[i] as i64 };
        if e.abs() > 64 {
            return Err(format!("sparse recovery: |e| = {} at coefficient {i} (row {digit1_row} is not a bare error)", e.abs()).into());
        }
        let v = mulmod((x0[i] + q0 - e.rem_euclid(q0 as i64) as u64) % q0, pinv, q0);
        let c = if v > q0 / 2 { v as i64 - q0 as i64 } else { v as i64 };
        if c.abs() > 1 {
            return Err(format!("sparse recovery: coefficient {i} = {c}, not ternary").into());
        }
        out.push(c as i8);
    }
    Ok(out)
}

struct Tally {
    gates: usize,
    bad: usize,
}
impl Tally {
    fn add(&mut self, what: &str, bad: usize, of: usize) {
        self.gates += 1;
        self.bad += bad;
        println!("[pc-check] {what}: {bad} mismatches of {of}");
    }
}

/// Dense rows of a key on the device: `b` (packed -> `ksk_unpack`, dense -> copy) and `a` regenerated by
/// `ksk_expand` (every digit of the full chain, slot ids of the key's shape), downloaded `[d][slot][N]`.
fn device_dense(
    ctx: &Context<'_, W, M>,
    s: &pc_cuda::Stream<Compute>,
    key: &DevKsk<'_, W>,
    ids: &ConstBuf<'_, i32>,
    wb: &mut DBuf<'_, Compute, W>,
    wa: &mut DBuf<'_, Compute, W>,
) -> Res<(Vec<u32>, Vec<u32>)> {
    let n = ctx.lay.n;
    let sh = key.shape;
    let nr = sh.ndig * sh.rows();
    match &key.store {
        KskStore::PackedSeeded { b, bits, row_words } => {
            let u = KskUnpackArgs { out: wb.vptr_mut(), inp: b.dptr(), row_words: *row_words as u64, nrows: nr as i32, bits: *bits as i32 };
            // SAFETY: wb holds >= nr rows of N; b holds nr packed rows
            launch_result("ksk_unpack", unsafe { (ctx.table.ksk_unpack)(&u, s.raw()) })?;
        }
        KskStore::Dense { b, .. } => wb.copy_from(s, b, nr * n)?,
    }
    let e = KskExpandArgs { out: wa.vptr_mut(), seed: key.seed_dev.dptr(), prime_ids: ids.dptr(), pr: ctx.pr(), nmod: sh.rows() as i32, dnum: sh.ndig as i32, d0: 0, pad: 0 };
    // SAFETY: wa holds ndig * rows rows; ids holds rows prime ids
    launch_result("ksk_expand", unsafe { (ctx.table.ksk_expand)(&e, s.raw()) })?;
    Ok((wb.download_range(s, 0, nr * n)?, wa.download_range(s, 0, nr * n)?))
}

fn max_rel(got: &[Complex], want: &[Complex]) -> f64 {
    let mx = want.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-300);
    got.iter().zip(want).map(|(a, b)| (*a - *b).abs()).fold(0.0, f64::max) / mx
}

/// Host-reference switch of ct 0 of `ct` (eval rows 0..=top) with `key`, then sigma_g; written back.
fn switch(s: &pc_cuda::Stream<Compute>, hc: &HostChain, chain: &Chain, key: &DevKsk<'_, W>, ct: &mut Ct<'_, W>, top: usize) -> Res<()> {
    let n = chain.n;
    let rows: Vec<usize> = (0..=top).collect();
    let get = |ct: &Ct<'_, W>, part| -> Res<Vec<Vec<u64>>> {
        Ok(ct.buf.download_rows(s, 0, part, &rows)?.chunks(n).map(|r| r.iter().map(|&x| x as u64).collect()).collect())
    };
    let (c0, c1) = (get(ct, 0)?, get(ct, 1)?);
    let (b, a) = key.download_dense(s, chain)?;
    let (u0, u1) = hc.key_switch(&c1, top, &key.shape, &b, &a)?;
    let mut r0: Vec<Vec<u64>> = (0..=top).map(|t| (0..n).map(|i| addmod(c0[t][i], u0[t][i], hc.primes[t])).collect()).collect();
    let mut r1 = u1;
    if let Some(g) = key.role.galois(n) {
        r0 = r0.par_iter().map(|r| galois::apply_eval(r, g)).collect::<Result<_, _>>()?;
        r1 = r1.par_iter().map(|r| galois::apply_eval(r, g)).collect::<Result<_, _>>()?;
    }
    for (part, r) in [(0, r0), (1, r1)] {
        let flat: Vec<W> = r.iter().flat_map(|x| x.iter().map(|&v| v as W)).collect();
        ct.buf.upload_rows(s, 0, part, &rows, &flat)?;
    }
    Ok(())
}

fn run(dir: &str) -> Res<Tally> {
    let mut t = Tally { gates: 0, bad: 0 };
    let t_all = Instant::now();
    let g0 = G0::open(dir)?;
    let pj = &g0.params;
    let n = g0.n;
    let chain = Chain::generate(&ChainSpec::n32())?;
    // 1. chain
    let dq: Vec<u64> = u64v(&pj["q"]).into_iter().chain(u64v(&pj["p"])).collect();
    let digits: Vec<Vec<usize>> = pj["digits"].as_array().ok_or("params.digits")?.iter().map(|d| u64v(d).into_iter().map(|x| x as usize).collect()).collect();
    let bad = dq.iter().zip(chain.primes()).filter(|(a, b)| **a != *b).count() + (dq.len() != chain.primes().len()) as usize + (digits != chain.digits) as usize;
    t.add("chain n32 primes + digits == dump", bad, dq.len() + 1);
    if bad != 0 || n != chain.n {
        return Ok(t);
    }
    let hc = HostChain::new(&chain);
    let sk_s = read_sk(&g0.dir, n)?;
    let all_ids: Vec<usize> = (0..chain.primes().len()).collect();
    let s_eval = hc.eval_rows(&sk_s.iter().map(|&x| x as i64).collect::<Vec<_>>(), &all_ids);

    // key inventory -> roles
    let keys = pj["keys"].as_object().ok_or("params.keys")?;
    let mut names: Vec<(String, KeyRole)> = keys.keys().map(|k| Ok((k.clone(), role_of(k, n)?))).collect::<Res<_>>()?;
    names.sort_by_key(|(k, _)| k.parse::<u64>().unwrap_or(u64::MAX));
    let nrot = names.iter().filter(|(_, r)| matches!(r, KeyRole::Rot { .. })).count();
    let spec = KeySetSpec {
        relin: names.iter().any(|(_, r)| *r == KeyRole::Relin),
        conj: names.iter().any(|(_, r)| *r == KeyRole::Conj),
        rots: names.iter().filter_map(|(_, r)| if let KeyRole::Rot { step, .. } = r { Some(RotSpec { step: *step, q_top: None }) } else { None }).collect(),
        encaps: names.iter().any(|(_, r)| *r == KeyRole::EncDenseToSparse) && names.iter().any(|(_, r)| *r == KeyRole::EncSparseToDense),
    };
    let packed = KskLayout::packed_for::<W>(&chain)?;
    println!("[pc-check] keys: {} in the dump ({nrot} rotations); layout {packed:?}", names.len());

    // sparse secret from the sparse->dense key (host)
    let btoa = names.iter().find(|(_, r)| *r == KeyRole::EncSparseToDense).ok_or("no sparse->dense key")?.0.clone();
    let m_btoa = material(&g0, &btoa)?;
    let row_out = chain.digits[1][0];
    let s_sp = recover_sparse(&hc, &m_btoa, &s_eval, row_out)?;
    let hw = s_sp.iter().filter(|&&x| x != 0).count();
    println!("[pc-check] sparse secret recovered from key {btoa}: Hamming weight {hw}");
    drop(m_btoa);
    let sk = SecretKey { s: sk_s.clone(), s_sparse: s_sp.clone() };
    let sp_eval = hc.eval_rows(&s_sp.iter().map(|&x| x as i64).collect::<Vec<_>>(), &all_ids);

    // memory per configuration (host arithmetic)
    let lay = Layout::from_chain(&chain);
    let dense_b = spec.bytes::<W>(&chain, KskLayout::Dense)?;
    let packed_b = spec.bytes::<W>(&chain, packed)?;
    let band = KskShape::banded(&chain, 32)?;
    println!(
        "[pc-check] key memory, g0 set ({} keys): packed {:.2} GB, dense (a + b) {:.2} GB; one full key packed {:.1} MB, dense b only {:.1} MB, band 32 packed {:.1} MB, bottom-only {:.1} MB",
        names.len(),
        packed_b as f64 / 1e9,
        dense_b as f64 / 1e9,
        KskShape::full(&chain).bytes::<W>(n, packed) as f64 / 1e6,
        Arena::bytes_for::<W>(KskShape::full(&chain).b_words(n, KskLayout::Dense)) as f64 / 1e6,
        band.bytes::<W>(n, packed) as f64 / 1e6,
        KskShape::bottom(&chain)?.bytes::<W>(n, packed) as f64 / 1e6
    );

    // runtime: imported set + generated set (both packed), secret, enc/dec, keygen tables, unpack/expand work
    let dev = Device::open()?;
    let cfg = MemCfg { k_max: 1, pool: vec![PoolClass { k: 1, count: 4 }], events: 8, scratch_cts: 1 };
    let mut plan = MemPlan::build::<W, M>(&chain, cfg)?;
    let shapes: Vec<KskShape> = spec.keys(&chain)?.into_iter().map(|k| k.1).collect();
    let full = KskShape::full(&chain);
    plan.reserve("keys", 2 * packed_b);
    plan.reserve("keys.secret", DevSecret::<W>::bytes(&lay));
    plan.reserve("keygen", KeyGen::<W>::bytes(&lay, &shapes));
    plan.reserve("encrypt", Encryptor::<W>::bytes(&lay, false));
    plan.reserve("decrypt", Decryptor::<W>::bytes(&lay, chain.level_unit, 1));
    plan.reserve("check.work", 2 * Arena::bytes_for::<W>(full.b_words(n, KskLayout::Dense)) + 4 * Arena::bytes_for::<i32>(full.rows()) + 4096);
    println!("[pc-check] MemPlan: {}", plan.report());
    let arena = plan.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &plan)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &plan)?;
    let pool = CtPool::<W>::new(&arena, &plan)?;
    let mut sec = DevSecret::<W>::carve(&arena, lay)?;
    let mut imp = KeySet::<W>::carve(&arena, &chain, &spec, packed)?;
    let mut gen = KeySet::<W>::carve(&arena, &chain, &spec, packed)?;
    let mut kg = KeyGen::<W>::new(&arena, &ex.compute, &chain, &shapes_of(&imp))?;
    let mut prng = Prng::from_os()?;
    let mut encr = Encryptor::<W>::new(&arena, &ex.compute, lay, false, &mut prng)?;
    let mut dec = Decryptor::<W>::new(&arena, &ex.compute, lay, chain.level_unit, 1)?;
    let mut wb = arena.alloc::<Compute, W>("check.wb", full.b_words(n, KskLayout::Dense))?;
    let mut wa = arena.alloc::<Compute, W>("check.wa", full.b_words(n, KskLayout::Dense))?;
    let mut id_tabs = std::collections::BTreeMap::new();
    for sh in shapes_of(&imp) {
        let ids: Vec<i32> = sh.row_ids().into_iter().map(|x| x as i32).collect();
        id_tabs.insert(sh, ConstBuf::new(&arena, "check.ids", &ex.compute, &ids)?);
    }
    arena.freeze();
    sec.load(&ctx, &ex.compute, &sk)?;
    dec.load_secret(&ex.compute, &sec)?;
    println!("[pc-check] setup {:.1} s; arena {} of {} B", t_all.elapsed().as_secs_f64(), arena.used_bytes(), arena.capacity());

    // 2-4. import every key; device unpack + regen; compare and relation
    let t_imp = Instant::now();
    let (mut bmis, mut bwords, mut amis, mut awords, mut akeys, mut rel_rows, mut rel_max) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0i64);
    let mut rel_bad = 0usize;
    let mut io = 0.0;
    for (name, role) in &names {
        let ti = Instant::now();
        let m = material(&g0, name)?;
        io += ti.elapsed().as_secs_f64();
        let key = imp.iter_mut().find(|k| k.role == *role).ok_or_else(|| format!("key {name}: no slot for {role:?}"))?;
        import_material(key, &ex.compute, &chain, &m, *role == KeyRole::EncDenseToSparse)?;
        let sh = key.shape;
        let (db, da) = device_dense(&ctx, &ex.compute, key, &id_tabs[&sh], &mut wb, &mut wa)?;
        let ids = sh.row_ids();
        let rows = sh.rows();
        let bm: usize = (0..sh.ndig * rows).into_par_iter().map(|r| db[r * n..(r + 1) * n].iter().zip(m.b_row(r / rows, ids[r % rows])).filter(|(x, y)| x != y).count()).sum();
        bmis += bm;
        bwords += db.len();
        if keys[name.as_str()]["has_a"].as_bool() == Some(true) {
            let file = if name == "mult" { "keys/mult.a.bin".to_string() } else { format!("keys/auto{name}.a.bin") };
            let a = read_u32(&g0.dir.join(file))?;
            let am: usize = (0..sh.ndig * rows).into_par_iter().map(|r| da[r * n..(r + 1) * n].iter().zip(&a[((r / rows) * m.moduli.len() + ids[r % rows]) * n..][..n]).filter(|(x, y)| x != y).count()).sum();
            println!("[pc-check]   key {name} ({role:?}): device-regenerated a vs dumped a: {am} mismatches of {}", da.len());
            amis += am;
            awords += da.len();
            akeys += 1;
        }
        let (sn, so, sq) = role_secrets(*role, n, &s_eval, &sp_eval)?;
        let b64: Vec<u64> = db.iter().map(|&x| x as u64).collect();
        let a64: Vec<u64> = da.iter().map(|&x| x as u64).collect();
        let (e, r) = hc.relation_max_err(&sh, &b64, &a64, &sn, &so, sq);
        rel_rows += r;
        rel_max = rel_max.max(e);
        if e > 64 {
            rel_bad += 1;
            println!("[pc-check]   key {name} ({role:?}): relation max |e| = {e} > 64");
        }
    }
    println!("[pc-check] import: {} keys in {:.1} s ({io:.1} s reading the dump); device key memory {:.2} GB", names.len(), t_imp.elapsed().as_secs_f64(), imp.bytes() as f64 / 1e9);
    t.add(&format!("packed b (import -> device unpack) vs dumped b, {} keys every digit + limb", names.len()), bmis, bwords);
    t.add(&format!("device-regenerated a (seed, KSKA) vs dumped dense a, {akeys} keys"), amis, awords);
    t.add(&format!("switch relation, {} keys x {rel_rows} rows with device-regenerated a, max |e| {rel_max} (bound 64)", names.len()), rel_bad, names.len());

    // 6. keygen at scale (same spec, from the dump's secret)
    let tg = kg.generate_set(&ctx, &mut ex, &sec, &mut gen, &mut prng)?;
    println!(
        "[pc-check] keygen: {} keys in {:.2} s (host sampling {:.2} s, upload {:.2} s, device {:.2} s), {:.2} GB",
        tg.keys,
        tg.total,
        tg.host_sample,
        tg.upload,
        tg.device,
        gen.bytes() as f64 / 1e9
    );
    let (mut grel_max, mut grel_bad) = (0i64, 0usize);
    for key in gen.iter() {
        let sh = key.shape;
        let (db, da) = device_dense(&ctx, &ex.compute, key, &id_tabs[&sh], &mut wb, &mut wa)?;
        let (sn, so, sq) = role_secrets(key.role, n, &s_eval, &sp_eval)?;
        let (e, _) = hc.relation_max_err(&sh, &db.iter().map(|&x| x as u64).collect::<Vec<_>>(), &da.iter().map(|&x| x as u64).collect::<Vec<_>>(), &sn, &so, sq);
        grel_max = grel_max.max(e);
        grel_bad += (e > 64) as usize;
    }
    t.add(&format!("generated keys: switch relation, {} keys, max |e| {grel_max} (bound 64)", gen.len()), grel_bad, gen.len());

    // 5. agreement: imported vs generated key results
    let enc = Encoder::new(&chain);
    let slots = n / 2;
    let vals: Vec<Complex> = {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        (0..slots)
            .map(|_| {
                let mut r = || {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    (x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
                };
                Complex::new(r(), r())
            })
            .collect()
    };
    let mut agree_bad = 0usize;
    let mut cases = 0usize;
    let rotl = |r: i64| -> Vec<Complex> { (0..slots).map(|j| vals[(j as i64 + r).rem_euclid(slots as i64) as usize]).collect() };
    let conj: Vec<Complex> = vals.iter().map(|v| Complex::new(v.re, -v.im)).collect();
    let bot = (chain.q.len() - chain.level_unit) as u32;
    type Case<'k> = (&'static str, u32, Vec<&'k DevKsk<'k, W>>, Vec<&'k DevKsk<'k, W>>, Vec<Which>, Vec<Complex>);
    let cases_v: Vec<Case> = vec![
        ("rotate +1 @ level 0", 0, vec![imp.rot(1)?], vec![gen.rot(1)?], vec![Which::Dense], rotl(1)),
        ("conjugate @ level 0", 0, vec![imp.conj()?], vec![gen.conj()?], vec![Which::Dense], conj.clone()),
        (
            "dense->sparse->dense @ bottom",
            bot,
            vec![&imp.bts()?.enc_dense_to_sparse, &imp.bts()?.enc_sparse_to_dense],
            vec![&gen.bts()?.enc_dense_to_sparse, &gen.bts()?.enc_sparse_to_dense],
            vec![Which::Sparse, Which::Dense],
            vals.clone(),
        ),
    ];
    for (what, level, ki, kgv, which, want) in cases_v {
        let top = ctx.top_of(level)?;
        let sf = ctx.scale.sf(top as i32)?;
        let mut meta = CtMeta::new(level, 1).with_nf(sf);
        meta.slots = slots as u32;
        let msg = enc.encode_slots(&vals, slots, top, Payload::Complex, sf)?;
        let mut outs = Vec::new();
        for keys in [&ki, &kgv] {
            let mut ctp = pool.acquire(1, meta)?;
            let ct = ctp.get_mut().ok_or("pool handle shared")?;
            encr.encrypt_sk(&ctx, &ex.compute, &sec, std::slice::from_ref(&msg), meta, ct, &mut prng)?;
            let mut errs = Vec::new();
            let mut last = Vec::new();
            for (k, w) in keys.iter().zip(&which) {
                switch(&ex.compute, &hc, &chain, k, ct, top)?;
                last = dec.decrypt_slots(&ctx, &ex.compute, &enc, ct, Payload::Complex, *w)?.remove(0);
                let mid_want = if *w == Which::Sparse { &vals } else { &want };
                errs.push(max_rel(&last, mid_want));
            }
            outs.push((errs, last));
        }
        let diff = max_rel(&outs[0].1, &outs[1].1);
        let worst = outs.iter().flat_map(|o| o.0.iter()).fold(0.0f64, |a, &b| a.max(b));
        let fmt = |e: &[f64]| e.iter().map(|x| format!("2^{:.1}", x.log2())).collect::<Vec<_>>().join(" -> ");
        println!("[pc-check]   {what}: imported key {}, generated key {}, |imported - generated| 2^{:.1}", fmt(&outs[0].0), fmt(&outs[1].0), diff.log2());
        cases += 1;
        if worst > 2f64.powi(-25) {
            agree_bad += 1;
        }
    }
    t.add("agreement: imported vs generated key results (decrypt == expected <= 2^-25 rel)", agree_bad, cases);
    println!("[pc-check] keys total {:.1} s", t_all.elapsed().as_secs_f64());
    Ok(t)
}

pub fn cmd_keys(args: &[String]) -> ExitCode {
    let Some(dir) = args.first() else {
        eprintln!("usage: pc-check keys <g0 dump>");
        return ExitCode::from(2);
    };
    match run(dir) {
        Ok(t) => {
            println!("[pc-check] keys: {} gates, {} mismatches", t.gates, t.bad);
            if t.bad == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            println!("[pc-check] keys {dir}: error: {e}");
            ExitCode::FAILURE
        }
    }
}
