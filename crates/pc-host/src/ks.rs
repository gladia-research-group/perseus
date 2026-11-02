// Hybrid key-switch inner products (ks.cuh): ks_dot (stored keys), ks_dot_regen (relin, `a` from the seed, optional
// fused ModDown epilogue), ks_dot_key (KeyTab keys, `a` regenerated for the first nregen digits), ksk_expand. One
// task per row (prime id); the digit loop accumulates lazily in the double word and reduces once, as the device.
use crate::arith::{automorph_slot, Hw};
use crate::expand::{expand_row, seed_at, TAG_KSKA};
use crate::launch;
use crate::raw::{buf, el, par_for, Buf};
use crate::simd::w as v;
use pc_kernels::args::*;

type S = CudaStream;

#[inline(always)]
fn prime(pr: *const PrimeConst, id: usize) -> &'static PrimeConst {
    // SAFETY: prime table by id (device constant data)
    unsafe { &*pr.add(id) }
}

#[inline(always)]
fn row<W>(base: *const std::ffi::c_void, off: usize) -> &'static [W] {
    // SAFETY: a row of N words inside the launch's buffers
    unsafe { std::slice::from_raw_parts((base as *const W).add(off), N) }
}

/// Slot of prime id t in a KeyTab's digit (q_top < 0: full layout).
pub(crate) fn slot_of(k: &KeyTab, t: usize, nq: usize) -> usize {
    if t < nq || k.q_top < 0 {
        t
    } else {
        k.q_top as usize + 1 + (t - nq)
    }
}

/// The operand of digit d for ciphertext c: c1 itself in its own digit, else the ModUp output.
fn x_row<W: Hw>(a: &KsArgs, c: usize, d: usize, dt: i64, roff: usize) -> &'static [W] {
    if d as i64 == dt {
        row::<W>(a.c1, c * a.ct_stride as usize + roff)
    } else {
        row::<W>(a.modup, c * a.mu_ct_stride as usize + d * a.mu_digit_stride as usize + roff)
    }
}

/// Chunk of slots per task (u32: 16 per row; other words take whole rows: their `a` expands per row).
fn chunk<W: Hw>() -> usize {
    if W::BITS == 32 {
        4096
    } else {
        N
    }
}

/// (s0, s1)[c] = sum_d x_{c,d} (b_d, a_d) over slots [lo, lo + n) of row t for the K ciphertexts; `ab(d, b, a)` fills
/// the key words of those slots.
fn dot_chunk<W: Hw>(a: &KsArgs, t: usize, lo: usize, n: usize, ab: &mut impl FnMut(usize, &mut [W], &mut [W])) -> (Buf<W::Acc>, Buf<W::Acc>) {
    let k = a.k as usize;
    let dt = if (t as i32) < a.nq { el::<i32>(a.digit_of as _, t) as i64 } else { -1 };
    let (mut s0, mut s1) = (buf::<W::Acc>(k * n), buf::<W::Acc>(k * n));
    let (mut b, mut ar) = (buf::<W>(n), buf::<W>(n));
    for d in 0..a.ndig as usize {
        ab(d, &mut b, &mut ar);
        for c in 0..k {
            let x = &x_row::<W>(a, c, d, dt, t * N)[lo..lo + n];
            v::mac2::<W>(&mut s0[c * n..(c + 1) * n], &mut s1[c * n..(c + 1) * n], x, &b, &ar);
        }
    }
    (s0, s1)
}

/// The key words of slots [lo, lo + out.len()) of a stored `b` row: dense, or unpacked from the packed row's own
/// words (lo a multiple of 32: 32 values of `bits` bits are `bits` whole words).
fn b_chunk<W: Hw>(k: &KeyTab, kb: *const std::ffi::c_void, r: usize, lo: usize, out: &mut [W]) {
    if k.bits == 0 {
        out.copy_from_slice(&row::<W>(kb, r * k.row_words as usize)[lo..lo + out.len()]);
        return;
    }
    // SAFETY: packed row of row_words u32 words (with its guard word)
    let w = unsafe { std::slice::from_raw_parts((kb as *const u32).add(r * k.row_words as usize), k.row_words as usize) };
    let w0 = lo / 32 * k.bits as usize;
    // SAFETY: packed keys exist at W = u32 only (launch check)
    let o = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u32, out.len()) };
    crate::simd::unpack(&w[w0..(w0 + out.len() / 32 * k.bits as usize + 2).min(w.len())], k.bits as u32, o);
}

/// The expanded `a` (KSKA lane) of (seed, digit, prime) at slots [lo, lo + out.len()).
fn a_chunk<W: Hw>(seed: &[u32; 8], d: usize, p: u64, lo: usize, out: &mut [W]) {
    if W::BITS == 32 {
        let r = crate::expand::cache::get(seed, d as u32, p as u32, N);
        // SAFETY: W is u32
        let o = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u32, out.len()) };
        o.copy_from_slice(&r[lo..lo + out.len()]);
    } else {
        // whole-row chunks at other words
        expand_row::<W>(seed, d as u32, p, TAG_KSKA, out);
    }
}

/// ks_dot's outputs of key r for slots [lo, lo + n) of row t: c0 modes, automorphism, accumulation.
fn store_rot<W: Hw>(a: &KsArgs, t: usize, r: usize, lo: usize, n: usize, s0: &[W::Acc], s1: &[W::Acc]) {
    let q = prime(a.pr, t);
    let p = W::from_u64(q.p);
    let kexp = el::<u32>(a.autok as _, r);
    let mode = a.c0_mode & 3;
    let (mut r0, mut r1) = (buf::<W>(n), buf::<W>(n));
    for c in 0..a.k as usize {
        v::reduce::<W>(&s0[c * n..(c + 1) * n], q, &mut r0);
        v::reduce::<W>(&s1[c * n..(c + 1) * n], q, &mut r1);
        let c0 = || &row::<W>(a.c0, c * a.ct_stride as usize + t * N)[lo..lo + n];
        match mode {
            1 if (t as i32) < a.nq => v::shoup_row::<W>(&mut r0, c0(), el::<W>(a.pmod, t), el::<W>(a.pmods, t), p, false),
            2 => v::add_rows::<W>(&mut r0, c0(), p),
            _ => {}
        }
        let ob = c * a.out_ct_stride as usize + r * a.out_rot_stride as usize + t * N;
        let (o0, o1) = (a.out0 as *mut W, a.out1 as *mut W);
        let acc = a.c0_mode & KS_ACC != 0;
        if kexp == 1 && !acc {
            // SAFETY: the chunk of the output rows
            unsafe {
                std::slice::from_raw_parts_mut(o0.add(ob + lo), n).copy_from_slice(&r0);
                std::slice::from_raw_parts_mut(o1.add(ob + lo), n).copy_from_slice(&r1);
            }
            continue;
        }
        for i in 0..n {
            let o = if kexp == 1 { lo + i } else { automorph_slot(kexp, (lo + i) as u32, LOG_N) as usize };
            // SAFETY: output word inside the launch's buffers (the automorphism permutes the row's slots)
            unsafe {
                let (mut x0, mut x1) = (r0[i], r1[i]);
                if acc {
                    x0 = W::add(x0, *o0.add(ob + o), p);
                    x1 = W::add(x1, *o1.add(ob + o), p);
                }
                *o0.add(ob + o) = x0;
                *o1.add(ob + o) = x1;
            }
        }
    }
}

fn row_id(a: &KsArgs, y: usize) -> usize {
    el::<u32>(a.rows as _, y) as usize
}

pub extern "C" fn ks_dot<W: Hw>(a: *const KsArgs, s: S) -> i32 {
    launch(a, s, |a| if a.k > 8 { -1 } else { 0 }, |a| {
        let n = chunk::<W>();
        par_for(a.nrows as usize * (N / n), a, |a, task| {
            let (t, lo) = (row_id(a, task / (N / n)), task % (N / n) * n);
            for r in 0..a.r as usize {
                let (kb, ka) = (el::<*const std::ffi::c_void>(a.kb as _, r), el::<*const std::ffi::c_void>(a.ka as _, r));
                let off = |d: usize| d * a.key_digit_stride as usize + t * N + lo;
                let (s0, s1) = dot_chunk::<W>(a, t, lo, n, &mut |d, b, ar| {
                    b.copy_from_slice(&row::<W>(kb, off(d))[..n]);
                    ar.copy_from_slice(&row::<W>(ka, off(d))[..n]);
                });
                store_rot::<W>(a, t, r, lo, n, &s0, &s1);
            }
        })
    })
}

pub extern "C" fn ks_dot_key<W: Hw>(a: *const KsKeyArgs, s: S) -> i32 {
    let bad = |g: &KsKeyArgs| {
        if g.key.bits < 0 || g.key.bits > 32 || (g.key.bits > 0 && W::BITS != 32) {
            -1
        } else if g.ks.k > 4 {
            -2
        } else {
            0
        }
    };
    launch(a, s, bad, |g| {
        let n = chunk::<W>();
        par_for(g.ks.nrows as usize * (N / n), g, |g, task| {
            let (a, kt) = (&g.ks, &g.key);
            let (t, lo) = (row_id(a, task / (N / n)), task % (N / n) * n);
            let p = prime(a.pr, t).p;
            let slot = slot_of(kt, t, a.nq as usize);
            for r in 0..a.r as usize {
                let kb = el::<*const std::ffi::c_void>(kt.kb as _, r);
                let seed = seed_at(el::<*const u32>(kt.seed as _, r));
                let (s0, s1) = dot_chunk::<W>(a, t, lo, n, &mut |d, b, ar| {
                    if (d as i32) < kt.nregen {
                        a_chunk::<W>(&seed, d, p, lo, ar);
                    } else {
                        let ka = el::<*const std::ffi::c_void>(a.ka as _, r);
                        ar.copy_from_slice(&row::<W>(ka, d * a.key_digit_stride as usize + t * N + lo)[..n]);
                    }
                    b_chunk::<W>(kt, kb, d * kt.srows as usize + slot, lo, b);
                });
                store_rot::<W>(a, t, r, lo, n, &s0, &s1);
            }
        })
    })
}

pub extern "C" fn ks_dot_regen<W: Hw>(a: *const KsRegenArgs, s: S) -> i32 {
    let bad = |g: &KsRegenArgs| {
        if !g.key.kb.is_null() && (g.key.bits < 0 || g.key.bits > 32 || (g.key.bits > 0 && W::BITS != 32)) {
            -1
        } else if !g.add.is_null() && g.use_epi == 0 {
            -1
        } else if g.ks.k > 4 {
            -2
        } else {
            0
        }
    };
    launch(a, s, bad, |g| {
        let n = chunk::<W>();
        par_for(g.ks.nrows as usize * (N / n), g, |g, task| {
            let (a, kt, e) = (&g.ks, &g.key, &g.epi);
            let (t, lo) = (row_id(a, task / (N / n)), task % (N / n) * n);
            let q = prime(a.pr, t);
            let p = W::from_u64(q.p);
            let seed = seed_at(g.seed);
            let tab = !kt.kb.is_null();
            let kb = if tab { el::<*const std::ffi::c_void>(kt.kb as _, 0) } else { el::<*const std::ffi::c_void>(a.kb as _, 0) };
            let slot = if tab { slot_of(kt, t, a.nq as usize) } else { 0 };
            let (s0, s1) = dot_chunk::<W>(a, t, lo, n, &mut |d, b, ar| {
                a_chunk::<W>(&seed, d, q.p, lo, ar);
                if tab {
                    b_chunk::<W>(kt, kb, d * kt.srows as usize + slot, lo, b);
                } else {
                    b.copy_from_slice(&row::<W>(kb, d * a.key_digit_stride as usize + t * N + lo)[..n]);
                }
            });
            let mut x = buf::<W>(n);
            for c in 0..a.k as usize {
                let ob = c * a.out_ct_stride as usize + t * N + lo;
                if g.use_epi == 0 {
                    // SAFETY: the chunk of the output rows
                    let (o0, o1) = unsafe { (std::slice::from_raw_parts_mut((a.out0 as *mut W).add(ob), n), std::slice::from_raw_parts_mut((a.out1 as *mut W).add(ob), n)) };
                    v::reduce::<W>(&s0[c * n..(c + 1) * n], q, o0);
                    v::reduce::<W>(&s1[c * n..(c + 1) * n], q, o1);
                    continue;
                }
                let (bi, bis) = (el::<W>(e.binv, t), el::<W>(e.binvs, t));
                let kk = if e.k.is_null() { W::default() } else { W::red1(el::<W>(e.k, t), p) };
                let al = (!e.alpha.is_null()).then(|| (el::<W>(e.alpha, t), el::<W>(e.alphas, t)));
                for part in 0..2 {
                    let off = ob + part * e.part_stride as usize;
                    let cv = &row::<W>(e.conv, off)[..n];
                    let add = (!g.add.is_null()).then(|| &row::<W>(g.add, off)[..n]);
                    let sx = if part == 0 { &s0 } else { &s1 };
                    v::reduce::<W>(&sx[c * n..(c + 1) * n], q, &mut x);
                    // SAFETY: the chunk of the epilogue destination (read, then written)
                    let dp = unsafe { std::slice::from_raw_parts_mut((e.dst as *mut W).add(off), n) };
                    for i in 0..n {
                        let dv = match al {
                            Some((ai, ais)) => W::mul_shoup(dp[i], ai, ais, p),
                            None => dp[i],
                        };
                        let mut r = W::add(dv, W::mul_shoup(W::sub(x[i], cv[i], p), bi, bis, p), p);
                        if e.dbl != 0 {
                            r = W::add(r, r, p);
                        }
                        if part == 0 {
                            r = W::add(r, kk, p);
                        }
                        if let Some(ad) = add {
                            r = W::add(r, ad[i], p);
                        }
                        dp[i] = r;
                    }
                }
            }
        })
    })
}

pub extern "C" fn ksk_expand<W: Hw>(a: *const KskExpandArgs, s: S) -> i32 {
    launch(a, s, |a| if a.d0 < 0 || a.d0 >= a.dnum { -1 } else { 0 }, |a| {
        let nm = a.nmod as usize;
        par_for((a.dnum - a.d0) as usize * nm, a, |a, t| {
            let (z, m) = (t / nm, t % nm);
            let id = el::<i32>(a.prime_ids as _, m) as usize;
            let p = prime(a.pr, id).p;
            // SAFETY: output row (z, m)
            let out = unsafe { std::slice::from_raw_parts_mut((a.out as *mut W).add((z * nm + m) * N), N) };
            expand_row::<W>(&seed_at(a.seed), (a.d0 as usize + z) as u32, p, TAG_KSKA, out);
        })
    })
}
