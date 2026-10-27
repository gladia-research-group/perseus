//! PCMM at the StC-first split: gates and benches of `pc_gpu::matmul::pcmm`.
//!
//!   pc-check pcmm [--shapes gpt2.qkv,..] [--twin gpt2.qkv,..] [--seed s]   the digit-plane GEMM vs the exact integer
//!       product mod q, every element; the whole `Pcmm::product` on a random split ciphertext vs a host twin of the same
//!       formulas (FIDESlib Pcmm.cu semantics), every output word
//!   pc-bench pcmm --prec [--cf 6] [--wbits 24] [--sigma 0.1] [--wamp 1]   a real input through StC-first stages ->
//!       product -> K-batched finishes vs `_exact` (sylph methodology: complex Gaussian sigma 0.1, bits =
//!       log2(max |Y| / max err); --wamp scales the weights, for large band shifts at a bounded input)
//!   pc-bench pcmm [--shapes ..] [--k 1,2,4]      device ms per linear: product phases and output finishes per K
use crate::check_bts::Tally;
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::keys::{Prng, SecretKey};
use pc_core::modmul::{invmod, mulmod, Barrett64};
use pc_core::params::{Chain, ChainSpec};
use pc_core::pcmm_keys::{digit_planes, from_planes, weight_top, NttTab, PcmmKeys, Shape, B, L, N, N1, S_SWITCH};
use pc_cuda::{Compute, DBuf, Device, Event};
use pc_gpu::bts::cts::LtGeom;
use pc_gpu::bts::staged::BtsLog;
use pc_gpu::bts::stc_first::{StcFirstPlan, StcFirstSetup};
use pc_gpu::bts::{BtsCx, BtsSetup, BtsWork};
use pc_gpu::decrypt::{Decryptor, Which};
use pc_gpu::encrypt::Encryptor;
use pc_gpu::keygen::{test_prng, KeyGen};
use pc_gpu::keys::hostref::HostNtt;
use pc_gpu::keys::{DevSecret, KeySet, KeySetSpec, KskLayout, RotSpec};
use pc_gpu::ks::KsPlans;
use pc_gpu::matmul::pcmm::{finish_groups, linear, LinearHooks, LinearOpts, Pcmm, PcmmCfg, PcmmLinear, PcmmShape, PcmmTimes, WeightBuf};
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::poly::CtBuf;
use pc_gpu::{Context, Ct, Executor, MemCfg, MemPlan};
use pc_shadow::{BtsConfig, CtMeta, Quirks, Shadow, ShadowConfig};
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

type W = u32;
type M = Barrett64;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub enum Mode {
    Check,
    Bench,
}

/// The linears of the models: GPT-2 prefill T=64 at t = 64 lanes with 16-bit weights; Llama-3.1-8B
/// prefill at t = 8 (dense, 24-bit) and the blk64 layout (t = 64, 24-bit).
pub fn named_shapes() -> Vec<(&'static str, PcmmShape)> {
    let s = |t, n_in, n_out, nw| PcmmShape { t, n_in, n_out, nw };
    vec![
        ("gpt2.qkv", s(64, 2, 6, 2)),
        ("gpt2.out", s(64, 2, 2, 2)),
        ("gpt2.fc", s(64, 2, 6, 2)),
        ("gpt2.down", s(64, 6, 2, 2)),
        ("llama.qkv", s(8, 1, 3, 3)),
        ("llama.out", s(8, 1, 1, 3)),
        ("llama.gate_up", s(8, 1, 8, 3)),
        ("llama.down", s(8, 4, 1, 3)),
        ("blk64.gate_up", s(64, 8, 56, 3)),
        ("blk64.down", s(64, 28, 8, 3)),
    ]
}

fn pick(list: &str) -> Res<Vec<(&'static str, PcmmShape)>> {
    let all = named_shapes();
    list.split(',')
        .filter(|x| !x.is_empty())
        .map(|n| all.iter().find(|(m, _)| *m == n).copied().ok_or_else(|| format!("unknown shape {n} (one of {:?})", all.iter().map(|x| x.0).collect::<Vec<_>>()).into()))
        .collect()
}

fn centered(v: u64, p: u64) -> i64 {
    if v > p / 2 {
        v as i64 - p as i64
    } else {
        v as i64
    }
}

fn smod(c: i64, p: u64) -> u64 {
    c.rem_euclid(p as i64) as u64
}

/// Exact y \[2\]\[ro\]\[cols\] = W x mod q_pi over centered x \[2\]\[ri\]\[cols\] (W from the digit planes).
fn gemm_ref(planes: &[i8], nw: usize, ro: usize, ri: usize, x: &[u32], cols: usize, q: [u64; 2]) -> Vec<u32> {
    let w = from_planes(planes, nw);
    let mut y = vec![0u32; 2 * ro * cols];
    for (pi, &p) in q.iter().enumerate() {
        let xc: Vec<i32> = x[pi * ri * cols..(pi + 1) * ri * cols].iter().map(|&v| centered(v as u64, p) as i32).collect();
        y[pi * ro * cols..(pi + 1) * ro * cols].par_chunks_mut(cols).enumerate().for_each(|(r, out)| {
            let mut acc = vec![0i64; cols];
            let wr = &w[r * ri..(r + 1) * ri];
            for (k, &wk) in wr.iter().enumerate() {
                if wk == 0 {
                    continue;
                }
                let xr = &xc[k * cols..(k + 1) * cols];
                for (a, &xv) in acc.iter_mut().zip(xr) {
                    *a += wk * xv as i64;
                }
            }
            for (o, a) in out.iter_mut().zip(&acc) {
                *o = a.rem_euclid(p as i64) as u32;
            }
        });
    }
    y
}

// ── host twin of the whole product ───────────────────────────────────────────────────────────────────────────────

struct Twin {
    k: PcmmKeys,
    fwd: Vec<u32>,
    back: Vec<u32>,
    pack: Vec<u32>,
    t16: [NttTab; B],
    t12: [NttTab; B],
    /// the chain's q0..q3 NTTs (OpenFHE order, the device's)
    chain: Vec<HostNtt>,
    q: [u64; 4],
}

impl Twin {
    fn new(k: &PcmmKeys, chain: &Chain) -> Res<Twin> {
        let t16 = k.tables(N)?;
        let t12 = k.tables(N1)?;
        let roots = chain.roots();
        let q = [chain.q[0], chain.q[1], chain.q[2], chain.q[3]];
        Ok(Twin {
            fwd: PcmmKeys::eval_form(&k.fwd, &t16),
            back: PcmmKeys::eval_form(&k.back, &t16),
            pack: PcmmKeys::eval_form(&k.pack, &t12),
            k: k.clone(),
            t16,
            t12,
            chain: (0..4).map(|i| HostNtt::new(q[i], roots[i], N)).collect(),
            q,
        })
    }

    /// \[ab\]\[2\]\[outs\]\[n\] = sum over each output's nl pieces of the key switch of x \[2\][outs nl]\[n\] (NTT-form keys).
    fn ks(&self, x: &[u32], n: usize, outs: usize, nl: usize, key: &[u32]) -> Vec<u32> {
        let tabs = if n == N { &self.t16 } else { &self.t12 };
        let p = self.k.primes;
        let polys = outs * nl;
        let hat = [invmod(p[1] % p[0], p[0]), invmod(p[0] % p[1], p[1])];
        // D[j][pi][poly][n], NTT form
        let mut d = vec![0u32; L * B * polys * n];
        d.par_chunks_mut(n).enumerate().for_each(|(idx, row)| {
            let (j, pi, poly) = (idx / (B * polys), (idx / polys) % B, idx % polys);
            let src = &x[(j * polys + poly) * n..(j * polys + poly + 1) * n];
            for (o, &v) in row.iter_mut().zip(src) {
                let c = centered(mulmod(v as u64, hat[j], p[j]), p[j]);
                *o = smod(c, p[pi]) as u32;
            }
            tabs[pi].ntt(row);
        });
        // acc[ab][pi][o][n]
        let mut acc = vec![0u32; 2 * B * outs * n];
        acc.par_chunks_mut(n).enumerate().for_each(|(idx, row)| {
            let (ab, pi, o) = (idx / (B * outs), (idx / outs) % B, idx % outs);
            let pp = p[pi];
            for kk in 0..n {
                let mut s = 0u64;
                for j in 0..L {
                    for l in 0..nl {
                        let dv = d[((j * B + pi) * polys + o * nl + l) * n + kk] as u64;
                        let kv = key[((((l * L + j) * B + pi) * 2) + ab) * n + kk] as u64;
                        s = (s + dv * kv % pp) % pp;
                    }
                }
                row[kk] = s as u32;
            }
            tabs[pi].intt(row);
        });
        let pinv = [invmod(p[2] % p[0], p[0]), invmod(p[2] % p[1], p[1])];
        let mut out = vec![0u32; 2 * L * outs * n];
        for ab in 0..2 {
            for o in 0..outs {
                for kk in 0..n {
                    let c = centered(acc[((ab * B + 2) * outs + o) * n + kk] as u64, p[2]);
                    for i in 0..L {
                        let v = acc[((ab * B + i) * outs + o) * n + kk] as u64;
                        let t = (v + p[i] - smod(c, p[i])) % p[i];
                        out[((ab * L + i) * outs + o) * n + kk] = mulmod(t, pinv[i], p[i]) as u32;
                    }
                }
            }
        }
        out
    }

    /// The outputs' eval rows [ct part]\[limb < 4\][N] of y = x W, inputs given the same way.
    fn product(&self, sh: &PcmmShape, planes: &[i8], ins: &[Vec<u32>]) -> Res<Vec<Vec<u32>>> {
        let lay = Shape::new(sh.t)?;
        let (np, r, nch) = (lay.np(), lay.r(), lay.nch());
        let (ri, ro) = (nch * sh.n_in, nch * sh.n_out);
        let p = self.k.primes;
        let q = self.q;
        let mut a = vec![0u32; L * ri * N1];
        let mut bm = vec![0u32; L * ri * np];
        for (c, ct) in ins.iter().enumerate() {
            // staging part sp = ct part 1 - sp, coefficient form
            let mut st = vec![0u64; 8 * N];
            st.par_chunks_mut(N).enumerate().for_each(|(row, dst)| {
                let (sp, l) = (row / 4, row % 4);
                let src = &ct[((1 - sp) * 4 + l) * N..((1 - sp) * 4 + l + 1) * N];
                for (o, &v) in dst.iter_mut().zip(src) {
                    *o = v as u64;
                }
                self.chain[l].inv(dst);
            });
            let inv3: Vec<u64> = (0..3).map(|i| invmod(q[3] % q[i], q[i])).collect();
            let inv2: Vec<u64> = (0..2).map(|i| invmod(q[2] % q[i], q[i])).collect();
            let mut xr = vec![0u32; 4 * N];
            for sp in 0..2 {
                for kk in 0..N {
                    let mut v = [0u64; 4];
                    for (l, vl) in v.iter_mut().enumerate() {
                        *vl = st[(sp * 4 + l) * N + kk];
                    }
                    let c3 = centered(v[3], q[3]);
                    for i in 0..3 {
                        v[i] = mulmod((v[i] + q[i] - smod(c3, q[i])) % q[i], inv3[i], q[i]);
                    }
                    let c2 = centered(v[2], q[2]);
                    for i in 0..2 {
                        xr[(sp * 2 + i) * N + kk] = mulmod((v[i] + q[i] - smod(c2, q[i])) % q[i], inv2[i], q[i]) as u32;
                    }
                }
            }
            let res = self.ks(&xr[..2 * N], N, 1, 1, &self.fwd);
            let mut b = xr[2 * N..].to_vec();
            for i in 0..L {
                for kk in 0..N {
                    let x = &mut b[i * N + kk];
                    *x = ((*x as u64 + res[2 * N + i * N + kk] as u64) % p[i]) as u32;
                }
            }
            for pi in 0..L {
                let ka = &res[pi * N..(pi + 1) * N];
                for row in 0..nch {
                    let (sub, i) = (row / r, row % r);
                    for col in 0..N1 {
                        let (l, w) = (col / np, col % np);
                        let v = if l <= i {
                            ka[(w * r + (i - l)) * 16 + sub]
                        } else {
                            let cc = i + r - l;
                            if w == 0 {
                                let x = ka[((np - 1) * r + cc) * 16 + sub];
                                if x == 0 {
                                    0
                                } else {
                                    p[pi] as u32 - x
                                }
                            } else {
                                ka[((w - 1) * r + cc) * 16 + sub]
                            }
                        };
                        a[(pi * ri + c * nch + row) * N1 + col] = v;
                    }
                    for w in 0..np {
                        bm[(pi * ri + c * nch + row) * np + w] = b[pi * N + (w * r + i) * 16 + sub];
                    }
                }
            }
        }
        let qq = [p[0], p[1]];
        let ap = gemm_ref(planes, sh.nw, ro, ri, &a, N1, qq);
        let bp = gemm_ref(planes, sh.nw, ro, ri, &bm, np, qq);
        let q23 = q[2] as u128 * q[3] as u128;
        let lift = [(q23 % q[0] as u128) as u64, (q23 % q[1] as u128) as u64];
        let mut outs = Vec::with_capacity(sh.n_out);
        for o in 0..sh.n_out {
            let mut cbuf = vec![0u32; L * nch * N1];
            for pi in 0..L {
                for sl in 0..nch {
                    let (sub, l) = (sl / r, sl % r);
                    for idx in 0..N1 {
                        let (w, i) = (idx / r, idx % r);
                        cbuf[(pi * nch + sl) * N1 + idx] = ap[(pi * ro + o * nch + sub * r + i) * N1 + l * np + w];
                    }
                }
            }
            let mut pk = self.ks(&cbuf, N1, S_SWITCH, r, &self.pack);
            for pi in 0..L {
                for sub in 0..S_SWITCH {
                    for idx in 0..N1 {
                        let (w, i) = (idx / r, idx % r);
                        let x = &mut pk[L * S_SWITCH * N1 + (pi * S_SWITCH + sub) * N1 + idx];
                        *x = ((*x as u64 + bp[(pi * ro + o * nch + sub * r + i) * np + w] as u64) % p[pi]) as u32;
                    }
                }
            }
            let mut ra = vec![0u32; L * N];
            let mut rb = vec![0u32; L * N];
            for pi in 0..L {
                for kk in 0..N {
                    let (z, sub) = (kk / 16, kk % 16);
                    ra[pi * N + kk] = pk[(pi * S_SWITCH + sub) * N1 + z];
                    rb[pi * N + kk] = pk[L * S_SWITCH * N1 + (pi * S_SWITCH + sub) * N1 + z];
                }
            }
            let mut fb = self.ks(&ra, N, 1, 1, &self.back);
            for i in 0..L {
                for kk in 0..N {
                    let x = &mut fb[2 * N + i * N + kk];
                    *x = ((*x as u64 + rb[i * N + kk] as u64) % p[i]) as u32;
                }
            }
            // output ct part cp = staging part 1 - cp; limbs 2, 3 zero
            let mut out = vec![0u32; 8 * N];
            out.par_chunks_mut(N).enumerate().for_each(|(row, dst)| {
                let (cp, l) = (row / 4, row % 4);
                let sp = 1 - cp;
                let mut v = vec![0u64; N];
                if l < 2 {
                    for kk in 0..N {
                        v[kk] = mulmod(fb[(sp * 2 + l) * N + kk] as u64, lift[l], q[l]);
                    }
                }
                self.chain[l].fwd(&mut v);
                for (o, x) in dst.iter_mut().zip(&v) {
                    *o = *x as u32;
                }
            });
            outs.push(out);
        }
        Ok(outs)
    }
}

// ── pc-check pcmm ────────────────────────────────────────────────────────────────────────────────────────────────

fn rand_planes(g: &mut Prng, len: usize) -> Vec<i8> {
    (0..len).map(|_| g.next_u32() as u8 as i8).collect()
}

fn rand_res(g: &mut Prng, n: usize, p: u64) -> Vec<u32> {
    (0..n).map(|_| g.uniform_below(p) as u32).collect()
}

fn arg(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned()
}

fn check(args: &[String]) -> Res<bool> {
    let shapes = pick(&arg(args, "--shapes").unwrap_or_else(|| "gpt2.qkv,gpt2.down,llama.qkv,llama.gate_up,llama.down".into()))?;
    let twin = pick(&arg(args, "--twin").unwrap_or_else(|| "gpt2.qkv,gpt2.down,llama.qkv,llama.down".into()))?;
    let seed: u32 = arg(args, "--seed").map(|s| s.parse()).transpose()?.unwrap_or(9);
    let mut uniq: Vec<PcmmShape> = Vec::new();
    for (_, s) in shapes.iter().chain(&twin) {
        if !uniq.contains(s) {
            uniq.push(*s);
        }
    }
    let chain = Chain::generate(&ChainSpec::n32())?;
    let mut g = test_prng(0x9a00 + seed);
    let sk = SecretKey::sample(&mut g, chain.n)?;
    let t0 = Instant::now();
    let ts: BTreeSet<usize> = uniq.iter().map(|s| s.t).collect();
    let keys: Vec<PcmmKeys> = ts.iter().map(|&t| PcmmKeys::generate(&sk.s, &chain.q[..4], t, &mut g)).collect::<Result<_, _>>()?;
    println!("[pc-check] pcmm keys for t in {ts:?}: {:.1} s ({:.1} MB), P = {}", t0.elapsed().as_secs_f64(), keys.iter().map(|k| k.bytes()).sum::<usize>() as f64 / 1e6, keys[0].primes[2]);
    let dev = Device::open()?;
    let lay = pc_gpu::Layout::from_chain(&chain);
    let mut mp = MemPlan::build::<W, M>(&chain, MemCfg { k_max: 1, pool: vec![], events: 8, scratch_cts: 1 })?;
    let pcfg = PcmmCfg { shapes: uniq.clone(), ..PcmmCfg::default() };
    mp.reserve("pcmm", Pcmm::<W>::bytes(&pcfg, lay.rows())?);
    let wmax = uniq.iter().map(|s| WeightBuf::bytes(s)).collect::<Result<Vec<_>, _>>()?.into_iter().max().unwrap_or(0);
    mp.reserve("check.weights", wmax);
    let xmax = uniq.iter().map(|s| 2 * s.rows_in().unwrap_or(0) * N1).max().unwrap_or(0);
    let ymax = uniq.iter().map(|s| 2 * s.rows_out().unwrap_or(0) * N1).max().unwrap_or(0);
    mp.reserve("check.xy", pc_cuda::Arena::bytes_for::<u32>(xmax) + pc_cuda::Arena::bytes_for::<u32>(ymax));
    let (nin, nout) = (uniq.iter().map(|s| s.n_in).max().unwrap_or(1), uniq.iter().map(|s| s.n_out).max().unwrap_or(1));
    mp.reserve("check.cts", pc_cuda::Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, nin)) + pc_cuda::Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, nout)));
    let arena = mp.allocate(&dev)?;
    let ex = Executor::<W>::new(&dev, &arena, &mp)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let lt = pc_cuda::blas::Lt::new(&dev)?;
    let s = &ex.compute;
    let tp = Instant::now();
    let mut pc = Pcmm::<W>::new(&arena, &dev, s, &ctx, &lt.inner, pcfg, &keys)?;
    println!("[pc-check] pcmm setup {:.2} s; GEMM plans (autotuned):", tp.elapsed().as_secs_f64());
    for p in &pc.plans.plans {
        println!("[pc-check]   m {} n {} k {}: {} of {} candidates, ms {:?}", p.m, p.n, p.k, p.label(), p.algos(), p.tuned_ms.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    }
    let mut wb = WeightBuf::carve(&arena, uniq.iter().max_by_key(|s| s.planes_len().unwrap_or(0)).ok_or("no shapes")?)?;
    let mut xb: DBuf<'_, Compute, u32> = arena.alloc("check.xy", xmax)?;
    let mut yb: DBuf<'_, Compute, u32> = arena.alloc("check.xy", ymax)?;
    let mut cin = Ct::new(CtMeta::new(0, 2), CtBuf::<Compute, W>::new(&arena, "check.cts", lay, nin)?);
    let mut cout = Ct::new(CtMeta::new(0, 2), CtBuf::<Compute, W>::new(&arena, "check.cts", lay, nout)?);
    arena.freeze();
    let q = [chain.q[0], chain.q[1]];
    let mut t = Tally::default();
    // (1a) the digit-plane GEMM
    for (name, sh) in &shapes {
        let (ri, ro) = (sh.rows_in()?, sh.rows_out()?);
        let planes = rand_planes(&mut g, sh.planes_len()?);
        wb.upload(s, &planes)?;
        for cols in [N1, 2 * sh.t] {
            let x: Vec<u32> = (0..2).flat_map(|pi| rand_res(&mut g, ri * cols, q[pi])).collect();
            xb.upload(s, &x)?;
            let td = Instant::now();
            pc.gemm(&ctx, s, sh, &wb.buf, &xb, &mut yb, cols)?;
            let got = yb.download_range(s, 0, 2 * ro * cols)?;
            let dt = td.elapsed().as_secs_f64();
            let th = Instant::now();
            let want = gemm_ref(&planes, sh.nw, ro, ri, &x, cols, q);
            let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            t.add(&format!("pcmm gemm {name} {sh:?} {} rows (m {} n {ro} k {ri}): device {dt:.2} s, host exact {:.1} s", if cols == N1 { "A" } else { "B" }, 2 * cols, th.elapsed().as_secs_f64()), bad, want.len());
        }
    }
    // (1b) the whole product vs the host twin
    for (name, sh) in &twin {
        let key = keys.iter().find(|k| k.shape.t == sh.t).ok_or("keys")?;
        let tw = Twin::new(key, &chain)?;
        let planes = rand_planes(&mut g, sh.planes_len()?);
        wb.upload(s, &planes)?;
        let mut ins = Vec::new();
        cin.buf.k = sh.n_in;
        for c in 0..sh.n_in {
            let mut rows = Vec::with_capacity(8 * N);
            for part in 0..2 {
                let mut pr = Vec::with_capacity(4 * N);
                for l in 0..4 {
                    pr.extend(rand_res(&mut g, N, chain.q[l]));
                }
                cin.buf.upload_rows(s, c, part, &[0, 1, 2, 3], &pr)?;
                rows.extend(pr);
            }
            ins.push(rows);
        }
        cin.meta = CtMeta::new(ctx.shape.n_q - 1 - 3, 2).with_nf(1.0);
        cout.buf.k = sh.n_out;
        let map: Vec<(usize, usize)> = (0..sh.n_out).map(|m| (0, m)).collect();
        let ev = [Event::timing(&dev)?, Event::timing(&dev)?, Event::timing(&dev)?, Event::timing(&dev)?];
        pc.product(&ctx, s, sh, &wb.buf, std::slice::from_ref(&cin), std::slice::from_mut(&mut cout), &map, 0.5, Some([&ev[0], &ev[1], &ev[2], &ev[3]]))?;
        let tm = PcmmTimes::read([&ev[0], &ev[1], &ev[2], &ev[3]])?;
        let th = Instant::now();
        let want = tw.product(sh, &planes, &ins)?;
        let ht = th.elapsed().as_secs_f64();
        let mut bad = 0;
        let mut tot = 0;
        for (m, w) in want.iter().enumerate() {
            for part in 0..2 {
                let got = cout.buf.download_rows(s, m, part, &[0, 1, 2, 3])?;
                let wp = &w[part * 4 * N..(part + 1) * 4 * N];
                bad += got.iter().zip(wp).filter(|(a, b)| a != b).count();
                tot += got.len();
            }
        }
        t.add(&format!("pcmm product {name} {sh:?} vs host twin (device inputs {:.2} gemm {:.2} outputs {:.2} ms; host {ht:.1} s)", tm.inputs, tm.gemm, tm.outputs), bad, tot);
        let meta_ok = cout.meta.level == cin.meta.level && cout.meta.deg == 2 && cout.meta.nf == 0.5;
        t.add(&format!("pcmm product {name}: output metadata (split level, deg 2, nf_out)"), (!meta_ok) as usize, 1);
    }
    println!("[pc-check] pcmm: {} gates, {} mismatching", t.gates, t.bad);
    Ok(t.bad == 0)
}

pub fn cmd(mode: Mode, args: &[String]) -> ExitCode {
    let r = match mode {
        Mode::Check => check(args),
        Mode::Bench => bench(args),
    };
    match r {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("pcmm: {e}");
            ExitCode::FAILURE
        }
    }
}

// ── pc-bench pcmm ────────────────────────────────────────────────────────────────────────────────────────────────

fn rot_steps(geoms: &[LtGeom], n: usize) -> BTreeSet<i64> {
    let mut steps = BTreeSet::new();
    for g in geoms {
        for i in 1..g.b {
            steps.insert(((i * g.stride) % (n / 2)) as i64);
        }
        if g.g > 1 {
            steps.insert(((g.b * g.stride) % (n / 2)) as i64);
        }
        steps.insert(g.offset.rem_euclid((n / 2) as i64));
    }
    steps.remove(&0);
    steps
}

/// Complex payload with Re, Im ~ N(0, sigma^2) (the sylph PCMM gate's X).
fn rand_gauss(g: &mut Prng, n: usize, sigma: f64) -> Vec<Complex> {
    (0..n)
        .map(|_| {
            let (a, b) = (1.0 - g.uniform_f64(), g.uniform_f64());
            let r = (-2.0 * a.ln()).sqrt() * sigma;
            let th = 2.0 * std::f64::consts::PI * b;
            Complex::new(r * th.cos(), r * th.sin())
        })
        .collect()
}

/// pcmm.py `PcmmLinear.encode`: W (n_in nch x n_out nch, \[in\]\[out\]) -> integer planes \[nw\]\[out\]\[in\] in channel order
/// (ch_of_row folded in) and the power-of-two weight scale; also returns the unpermuted integer W^T \[out\]\[in\].
fn encode_weights(w: &[f64], sh: &PcmmShape, wbits: u32) -> Res<(Vec<i8>, f64, Vec<i64>)> {
    let lay = Shape::new(sh.t)?;
    let nch = lay.nch();
    let (ri, ro) = (nch * sh.n_in, nch * sh.n_out);
    let top = weight_top(wbits) as f64;
    let amax = w.iter().fold(0f64, |m, x| m.max(x.abs()));
    let ws = (top / amax).log2().floor().exp2();
    let wt: Vec<i64> = (0..ro * ri).map(|k| (w[(k % ri) * ro + k / ri] * ws).round() as i64).collect();
    let perm = lay.ch_of_row();
    let mut wi = vec![0i64; ro * ri];
    for o in 0..sh.n_out {
        for c in 0..sh.n_in {
            for a in 0..nch {
                for b in 0..nch {
                    wi[(o * nch + a) * ri + c * nch + b] = wt[(o * nch + perm[a]) * ri + c * nch + perm[b]];
                }
            }
        }
    }
    Ok((digit_planes(&wi, sh.nw)?, ws, wt))
}

/// pcmm.py `_exact`: output block m of the integer linear on the input slot vectors (the split's right-shifted
/// layout: roll -1, (nch, t), product, / w_scale, roll +1).
fn exact(wt: &[i64], sh: &PcmmShape, ws: f64, xs: &[Vec<Complex>], m: usize) -> Vec<Complex> {
    let nch = 16 * (N1 / (2 * sh.t));
    let ri = nch * sh.n_in;
    let t = sh.t;
    let n = nch * t;
    let mut y = vec![Complex::new(0.0, 0.0); n];
    for (c, x) in xs.iter().enumerate() {
        for a in 0..nch {
            let row = &wt[(m * nch + a) * ri + c * nch..(m * nch + a) * ri + (c + 1) * nch];
            for tok in 0..t {
                let mut acc = Complex::new(0.0, 0.0);
                for (b, &wv) in row.iter().enumerate() {
                    acc = acc + x[(b * t + tok + 1) % n] * (wv as f64);
                }
                let o = (a * t + tok + 1) % n;
                y[o] = y[o] + acc;
            }
        }
    }
    y.iter().map(|v| *v * (1.0 / ws)).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

#[allow(clippy::too_many_lines)]
fn bench(args: &[String]) -> Res<bool> {
    let prec = args.iter().any(|a| a == "--prec");
    let shapes = pick(&arg(args, "--shapes").unwrap_or_else(|| {
        if prec {
            "gpt2.out".into()
        } else {
            "gpt2.qkv,gpt2.out,gpt2.down,llama.qkv,llama.out,llama.gate_up,llama.down".into()
        }
    }))?;
    let ks: Vec<usize> = arg(args, "--k").unwrap_or_else(|| "1,2,4".into()).split(',').map(|x| x.parse()).collect::<Result<_, _>>()?;
    let reps: usize = arg(args, "--reps").map(|s| s.parse()).transpose()?.unwrap_or(5);
    let cf: u32 = arg(args, "--cf").map(|s| s.parse()).transpose()?.unwrap_or(6);
    let wbits: u32 = arg(args, "--wbits").map(|s| s.parse()).transpose()?.unwrap_or(24);
    let sigma: f64 = arg(args, "--sigma").map(|s| s.parse()).transpose()?.unwrap_or(0.1);
    // --wamp: the weights' amplitude (x 1/sqrt(rows_in)); larger outputs (band shifts) at a bounded input
    let wamp: f64 = arg(args, "--wamp").map(|s| s.parse()).transpose()?.unwrap_or(1.0);
    let mut uniq: Vec<PcmmShape> = Vec::new();
    for (_, s) in &shapes {
        let s = if prec { PcmmShape { nw: wbits.div_ceil(8) as usize, ..*s } } else { *s };
        if !uniq.contains(&s) {
            uniq.push(s);
        }
    }
    let kmax = 4usize;
    let t0 = Instant::now();
    // PC_BENCH_CHAIN: another chain preset (the lean n32a4's split has 3 limbs)
    let chain = Chain::generate(&ChainSpec::preset(&std::env::var("PC_BENCH_CHAIN").unwrap_or_else(|_| "n32".into()))?)?;
    let n = chain.n;
    let slots = n / 2;
    let mut setup = BtsSetup::n32_shipped();
    setup.raise_drops = vec![0, 1];
    setup.k_max = kmax;
    let mut sf = StcFirstSetup::n32_shipped();
    sf.cache_slots = 4;
    // a lean chain's PCMM inputs: their own fixed StC stages (PC_BENCH_OWN_SPLIT "a,b" overrides THIN_SPLIT_PCMM)
    sf.own_split = Some(match std::env::var("PC_BENCH_OWN_SPLIT") {
        Ok(v) => {
            let x: Vec<f64> = v.split(',').map(|t| t.trim().parse()).collect::<std::result::Result<_, _>>()?;
            [*x.first().ok_or("PC_BENCH_OWN_SPLIT")?, *x.get(1).ok_or("PC_BENCH_OWN_SPLIT")?]
        }
        Err(_) => pc_gpu::bts::stc_first::THIN_SPLIT_PCMM,
    });
    let plan = StcFirstPlan::new(&chain, &setup, &sf)?;
    let mut geoms: Vec<LtGeom> = Vec::new();
    for v in &plan.dense.variants {
        geoms.extend(v.cts.iter().chain(&v.stc).map(|h| h.geom));
    }
    geoms.extend(plan.input.iter().map(|h| h.geom));
    geoms.extend(plan.sources.iter().flatten().map(|s| s.geom));
    let steps = rot_steps(&geoms, n);
    let spec = KeySetSpec { relin: true, conj: true, rots: steps.iter().map(|&s| RotSpec { step: s, q_top: None }).collect(), encaps: true };
    let dev = Device::open()?;
    let cfg = MemCfg { k_max: kmax, pool: vec![], events: 64, scratch_cts: 2 };
    let mut mp = MemPlan::build::<W, M>(&chain, cfg)?;
    let lay = mp.lay;
    let shapes_k: Vec<_> = spec.keys(&chain)?.into_iter().map(|(_, s)| s).collect();
    mp.reserve("keys", spec.bytes::<W>(&chain, KskLayout::Dense)?);
    mp.reserve("keygen", KeyGen::<W>::bytes(&lay, &shapes_k));
    mp.reserve("secret", DevSecret::<W>::bytes(&lay));
    mp.reserve("encrypt", Encryptor::<W>::bytes(&lay, false));
    mp.reserve("decrypt", Decryptor::<W>::bytes(&lay, chain.level_unit, kmax));
    mp.reserve("ks.plans", KsPlans::bytes::<W>(&chain, kmax)?);
    mp.reserve("ks.regen_a", pc_gpu::ks::KsWork::<W>::regen_bytes(&lay, chain.dnum()));
    let ocfg = OpsCfg::default();
    Ops::<W>::reserve(&mut mp, &ocfg);
    let (req, wshape) = plan.mem::<W>()?;
    req.reserve(&mut mp);
    mp.reserve("bts.work", BtsWork::<W>::bytes(&lay, wshape));
    let pcfg = PcmmCfg { shapes: uniq.clone(), ..PcmmCfg::default() };
    mp.reserve("pcmm", Pcmm::<W>::bytes(&pcfg, lay.rows())?);
    let wmax = uniq.iter().map(|s| WeightBuf::bytes(s)).collect::<Result<Vec<_>, _>>()?.into_iter().max().unwrap_or(0);
    mp.reserve("bench.weights", wmax);
    let nin = uniq.iter().map(|s| s.n_in).max().unwrap_or(1);
    let ngroups = uniq.iter().map(|s| s.n_out).max().unwrap_or(1);
    let ct_bytes = |k: usize| pc_cuda::Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(&lay, k));
    mp.reserve("bench.cts", nin * ct_bytes(1) + ngroups * ct_bytes(kmax));
    println!("[pc-bench] MemPlan {}", mp.report());
    let arena = mp.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &mp)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &mp)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, kmax)?;
    let mut ops = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    let mut work = BtsWork::<W>::new(&arena, lay, wshape)?;
    let mut keys = KeySet::<W>::carve(&arena, &chain, &spec, KskLayout::Dense)?;
    let mut prng = test_prng(0x9a);
    let sk = SecretKey::sample(&mut prng, n)?;
    let mut sec = DevSecret::<W>::carve(&arena, lay)?;
    sec.load(&ctx, &ex.compute, &sk)?;
    let mut kg = KeyGen::<W>::new(&arena, &ex.compute, &chain, &shapes_k)?;
    let tk = kg.generate_set(&ctx, &mut ex, &sec, &mut keys, &mut prng)?;
    println!("[pc-bench] keygen {} keys in {:.1} s", tk.keys, tk.total);
    let tpk = Instant::now();
    let ts: BTreeSet<usize> = uniq.iter().map(|s| s.t).collect();
    let pkeys: Vec<PcmmKeys> = ts.iter().map(|&t| PcmmKeys::generate(&sk.s, &chain.q[..4], t, &mut prng)).collect::<Result<_, _>>()?;
    println!("[pc-bench] pcmm keys (host) {:.2} s for t in {ts:?}", tpk.elapsed().as_secs_f64());
    let mut encr = Encryptor::<W>::new(&arena, &ex.compute, lay, false, &mut prng)?;
    let mut dec = Decryptor::<W>::new(&arena, &ex.compute, lay, chain.level_unit, kmax)?;
    dec.load_secret(&ex.compute, &sec)?;
    let route = plan.build(&arena, &dev, &ex.compute, &ctx, &keys, ex.ks.stage_ptrs())?;
    let lt = pc_cuda::blas::Lt::new(&dev)?;
    let tp = Instant::now();
    let mut pc = Pcmm::<W>::new(&arena, &dev, &ex.compute, &ctx, &lt.inner, pcfg, &pkeys)?;
    println!("[pc-bench] pcmm setup (upload + plans + autotune) {:.2} s", tp.elapsed().as_secs_f64());
    for p in &pc.plans.plans {
        println!("[pc-bench]   gemm m {} n {} k {}: {} ({} candidates, tuned ms {:?})", p.m, p.n, p.k, p.label(), p.algos(), p.tuned_ms.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    }
    let mut wb = WeightBuf::carve(&arena, uniq.iter().max_by_key(|s| s.planes_len().unwrap_or(0)).ok_or("no shapes")?)?;
    let mut ins: Vec<Ct<'_, W>> = (0..nin).map(|_| Ok(Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(&arena, "bench.cts", lay, 1)?))).collect::<Res<_>>()?;
    let mut outs: Vec<Ct<'_, W>> = (0..ngroups).map(|_| Ok(Ct::new(CtMeta::new(0, 1), CtBuf::<Compute, W>::new(&arena, "bench.cts", lay, kmax)?))).collect::<Res<_>>()?;
    arena.freeze();
    println!("[pc-bench] setup {:.1} s; arena used {:.1} of {:.1} GB", t0.elapsed().as_secs_f64(), arena.used_bytes() as f64 / 1e9, arena.capacity() as f64 / 1e9);
    let sh_ = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    let enc = Encoder::new(&chain);
    let nq = ctx.shape.n_q;
    let level = 44u32; // the PCMM input cap (old _N32_CAP)
    let mut t = Tally::default();
    let mut fresh = |ins: &mut [Ct<'_, W>], ex: &mut Executor<'_, W>, vals: &[Vec<Complex>]| -> Res<()> {
        let top = (nq - 1 - level) as usize;
        let sf_ = ctx.scale.sf(top as i32)?;
        for (x, v) in ins.iter_mut().zip(vals) {
            let mut meta = CtMeta::new(level, 1).with_nf(sf_);
            meta.slots = slots as u32;
            x.buf.k = 1;
            let msgs = vec![enc.encode_slots(v, slots, top, Payload::Complex, sf_)?];
            encr.encrypt_sk(&ctx, &ex.compute, &sec, &msgs, meta, x, &mut prng)?;
        }
        Ok(())
    };
    let mut g = test_prng(0x9a01);
    for (name, sh0) in &shapes {
        let sh = if prec { PcmmShape { nw: wbits.div_ceil(8) as usize, ..*sh0 } } else { *sh0 };
        let lay_s = Shape::new(sh.t)?;
        let nch = lay_s.nch();
        let (ri, ro) = (nch * sh.n_in, nch * sh.n_out);
        // weights ~ U(-1, 1) / sqrt(rows_in): |y| ~ |x|
        let amp = wamp / (ri as f64).sqrt();
        let w: Vec<f64> = (0..ri * ro).map(|_| (2.0 * g.uniform_f64() - 1.0) * amp).collect();
        let (planes, ws, wt) = encode_weights(&w, &sh, if prec { wbits } else { 8 * sh.nw as u32 })?;
        wb.upload(&ex.compute, &planes)?;
        let xs: Vec<Vec<Complex>> = (0..sh.n_in).map(|_| rand_gauss(&mut g, slots, sigma)).collect();
        let want: Vec<Vec<Complex>> = (0..sh.n_out).map(|m| exact(&wt, &sh, ws, &xs, m)).collect();
        let ymax = want.iter().flatten().fold(0f64, |m, z| m.max(z.re.abs()).max(z.im.abs()));
        let shift = ((ymax * 1.5 / 0.25).log2().ceil()).max(0.0) as u32;
        let lin = PcmmLinear { name: name.to_string(), shape: sh, w_scale: ws, shift };
        let case_ks: Vec<usize> = if prec { vec![1, 2.min(sh.n_out), 4.min(sh.n_out)] } else { ks.clone() };
        let mut drops_cases: Vec<Vec<u32>> = vec![vec![0; sh.n_out]];
        if prec {
            drops_cases.push((0..sh.n_out).map(|m| (m % 2) as u32).collect());
            drops_cases.push(vec![1; sh.n_out]);
        }
        let mut done_k = BTreeSet::new();
        for drops in &drops_cases {
            for &k in &case_ks {
                if k > kmax || !done_k.insert((drops.clone(), k)) {
                    continue;
                }
                let batch_drops = drops.iter().any(|&d| d != drops[0]);
                let groups = finish_groups(drops, k, batch_drops)?;
                let gd: Vec<u32> = groups.iter().map(|g| drops[g[0]]).collect();
                let reps_k = if prec { 1 } else { reps + 1 };
                let mut ms_prod = Vec::new();
                let mut ms_fin = Vec::new();
                let mut ph = Vec::new();
                let mut errs = 0f64;
                let mut log = BtsLog::default();
                for rep in 0..reps_k {
                    fresh(&mut ins[..sh.n_in], &mut ex, &xs)?;
                    for (gi, gr) in groups.iter().enumerate() {
                        outs[gi].buf.k = gr.len();
                    }
                    let mut cx = BtsCx { ctx: &ctx, ex: &mut ex, sh: &sh_, ks: &kplans, ops: &mut ops, work: &mut work };
                    let e: Vec<Event> = (0..6).map(|_| Event::timing(&dev)).collect::<Result<_, _>>()?;
                    e[0].record(&cx.ex.compute)?;
                    // the linear end to end, with events around its two halves (product / finishes)
                    linear_timed(&mut cx, &route, &mut pc, &lin, &wb.buf, &mut ins[..sh.n_in], &mut outs[..groups.len()], &groups, &gd, LinearOpts { cf, ..LinearOpts::default() }, &e, if rep == 0 { Some(&mut log) } else { None })?;
                    e[5].sync()?;
                    if rep > 0 || prec {
                        ms_prod.push(e[0].elapsed_ms(&e[4])? as f64);
                        ms_fin.push(e[4].elapsed_ms(&e[5])? as f64);
                        ph.push([e[0].elapsed_ms(&e[1])? as f64, e[1].elapsed_ms(&e[2])? as f64, e[2].elapsed_ms(&e[3])? as f64, e[3].elapsed_ms(&e[4])? as f64]);
                    }
                    if prec {
                        for gi in 0..groups.len() {
                            cx.ops.realize(&ctx, cx.ex, &sh_, std::slice::from_mut(&mut outs[gi]))?;
                        }
                        drop(cx);
                        for (gi, gr) in groups.iter().enumerate() {
                            let got = dec.decrypt_slots(&ctx, &ex.compute, &enc, &outs[gi], Payload::Complex, Which::Dense)?;
                            for (j, &m) in gr.iter().enumerate() {
                                let e = got[j].iter().zip(&want[m]).fold(0f64, |a, (g, w)| a.max((g.re - w.re).abs()).max((g.im - w.im).abs()));
                                errs = errs.max(e);
                            }
                            let lv = outs[gi].meta.level;
                            let want_l = BtsConfig::n32_shipped().landing(ctx.shape, &Quirks::none(), &pc_shadow::Route { raise_drop: gd[gi], cf: Some(cf), stc_first: true, ..pc_shadow::Route::dense() }, level, 1)?;
                            t.add(&format!("pcmm {name} drops {drops:?} K={k} group {gi}: realized landing == pc-shadow (drop {})", gd[gi]), (lv != want_l.level) as usize, 1);
                        }
                    }
                }
                if prec {
                    let bits = (ymax / errs).log2();
                    println!("[pc-bench] pcmm-prec {name} {sh:?} wbits {wbits} cf {cf} shift {shift} w_scale 2^{} drops {drops:?} K={k}: max |Y| {ymax:.3}, max err {:.3e} -> {bits:.2} bits (old: PCMM finish 11.38, 1024->1536 11.49; plain StC-first 11.65)", ws.log2(), errs);
                    t.add(&format!("pcmm-prec {name} drops {drops:?} K={k}: >= 11 bits"), (!(bits >= 11.0)) as usize, 1);
                    t.add(&format!("pcmm-prec {name} drops {drops:?} K={k}: {} external refreshes recorded with descriptors", sh.n_out), (log.refreshes.len() != sh.n_out || !log.refreshes.iter().all(|r| r.desc.contains("\"op\": \"pcmm\""))) as usize, 1);
                } else {
                    let pm = median(ms_prod.clone());
                    let fm = median(ms_fin.clone());
                    let phm: Vec<f64> = (0..4).map(|i| median(ph.iter().map(|p| p[i]).collect())).collect();
                    println!(
                        "[pc-bench] pcmm {name} {sh:?} K={k}: linear {:.2} ms = stc_in {:.2} + product {:.2} (inputs {:.2}, gemm {:.2}, outputs {:.2}) + finish {:.2} ({} groups, {:.2} ms/output); median of {reps}",
                        pm + fm,
                        phm[0],
                        pm - phm[0],
                        phm[1],
                        phm[2],
                        phm[3],
                        fm,
                        groups.len(),
                        fm / sh.n_out as f64
                    );
                }
            }
        }
    }
    println!("[pc-bench] pcmm: {} gates, {} failing", t.gates, t.bad);
    Ok(t.bad == 0)
}

/// `linear` with timing events: e\[0\] before the call (recorded by the caller), e\[1\] after the inputs' StC, e\[2\]
/// after the product's input phase, e\[3\] after its GEMMs, e\[4\] after its outputs, e\[5\] after the finishes.
#[allow(clippy::too_many_arguments)]
fn linear_timed<'a>(
    cx: &mut BtsCx<'_, 'a, W, M>,
    route: &pc_gpu::bts::stc_first::StcFirstRoute<'a, W, M>,
    pc: &mut Pcmm<'a, '_, W>,
    lin: &PcmmLinear,
    planes: &DBuf<'_, Compute, u8>,
    ins: &mut [Ct<'a, W>],
    outs: &mut [Ct<'a, W>],
    groups: &[Vec<usize>],
    drops: &[u32],
    opts: LinearOpts,
    e: &[Event],
    rec: Option<&mut BtsLog>,
) -> Res<()> {
    let hooks = LinearHooks { after_stc: Some(&e[1]), product: Some([&e[1], &e[2], &e[3], &e[4]]), after_finish: Some(&e[5]) };
    linear(cx, route, pc, lin, planes, ins, outs, groups, drops, opts, rec.map(|r| r as &mut dyn pc_gpu::bts::staged::BtsRecorder), Some(hooks))?;
    Ok(())
}
