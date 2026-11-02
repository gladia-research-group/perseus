// Hybrid key-switch inner products (ks.cuh): ks_dot (stored keys), ks_dot_regen (relin, `a` from the seed, optional
// fused ModDown epilogue), ks_dot_key (KeyTab keys, `a` regenerated for the first nregen digits), ksk_expand. One
// task per row (prime id); the digit loop accumulates lazily in the double word and reduces once, as the device.
use crate::arith::{automorph_slot, Hw};
use crate::expand::{expand_row, seed_at, TAG_KSKA};
use crate::launch;
use crate::raw::{el, par_for};
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

/// Stored `b` row `r` of a KeyTab key: dense W words, or `bits`-wide values packed in a little-endian u32 stream.
pub(crate) fn b_row<W: Hw>(k: &KeyTab, kb: *const std::ffi::c_void, r: usize, out: &mut [W]) {
    if k.bits == 0 {
        out.copy_from_slice(row::<W>(kb, r * k.row_words as usize));
        return;
    }
    // SAFETY: packed row of row_words u32 words (with its guard word)
    let w = unsafe { std::slice::from_raw_parts((kb as *const u32).add(r * k.row_words as usize), k.row_words as usize) };
    if W::BITS == 32 {
        // SAFETY: W is u32
        return crate::simd::unpack(w, k.bits as u32, unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u32, out.len()) });
    }
    let bits = k.bits as u64;
    let mask = (1u64 << bits) - 1;
    for (s, o) in out.iter_mut().enumerate() {
        let bit = s as u64 * bits;
        let (i, sh) = ((bit >> 5) as usize, bit & 31);
        let v = (w[i] as u64 | (w.get(i + 1).copied().unwrap_or(0) as u64) << 32) >> sh;
        *o = W::from_u64(v & mask);
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

/// (s0, s1)[c] = sum_d x_{c,d} (b_d, a_d) for the K ciphertexts of row t; `ab(d, b, a)` fills the key rows.
fn dot_rows<W: Hw>(a: &KsArgs, t: usize, ab: &mut impl FnMut(usize, &mut [W], &mut [W])) -> (crate::raw::Buf<W::Acc>, crate::raw::Buf<W::Acc>) {
    let k = a.k as usize;
    let dt = if (t as i32) < a.nq { el::<i32>(a.digit_of as _, t) as i64 } else { -1 };
    let (mut s0, mut s1) = (crate::raw::buf::<W::Acc>(k * N), crate::raw::buf::<W::Acc>(k * N));
    let (mut b, mut ar) = (crate::raw::buf::<W>(N), crate::raw::buf::<W>(N));
    for d in 0..a.ndig as usize {
        ab(d, &mut b, &mut ar);
        for c in 0..k {
            let x = x_row::<W>(a, c, d, dt, t * N);
            crate::simd::w::mac2::<W>(&mut s0[c * N..(c + 1) * N], &mut s1[c * N..(c + 1) * N], x, &b, &ar);
        }
    }
    (s0, s1)
}

/// ks_dot's outputs of key r for row t: c0 modes, automorphism, accumulation.
fn store_rot<W: Hw>(a: &KsArgs, t: usize, r: usize, s0: &[W::Acc], s1: &[W::Acc]) {
    let q = prime(a.pr, t);
    let p = W::from_u64(q.p);
    let kexp = el::<u32>(a.autok as _, r);
    let mode = a.c0_mode & 3;
    let (pm, pms) = if mode == 1 && (t as i32) < a.nq { (el::<W>(a.pmod, t), el::<W>(a.pmods, t)) } else { (W::default(), W::default()) };
    for c in 0..a.k as usize {
        let c0 = if mode != 0 { Some(row::<W>(a.c0, c * a.ct_stride as usize + t * N)) } else { None };
        let ob = c * a.out_ct_stride as usize + r * a.out_rot_stride as usize + t * N;
        let (o0, o1) = (a.out0 as *mut W, a.out1 as *mut W);
        for i in 0..N {
            let mut r0 = W::reduce(s0[c * N + i], q);
            let mut r1 = W::reduce(s1[c * N + i], q);
            match mode {
                1 if (t as i32) < a.nq => r0 = W::add(r0, W::mul_shoup(c0.unwrap()[i], pm, pms, p), p),
                2 => r0 = W::add(r0, c0.unwrap()[i], p),
                _ => {}
            }
            let o = if kexp == 1 { i } else { automorph_slot(kexp, i as u32, LOG_N) as usize };
            // SAFETY: output word inside the launch's buffers
            unsafe {
                if a.c0_mode & KS_ACC != 0 {
                    r0 = W::add(r0, *o0.add(ob + o), p);
                    r1 = W::add(r1, *o1.add(ob + o), p);
                }
                *o0.add(ob + o) = r0;
                *o1.add(ob + o) = r1;
            }
        }
    }
}

fn row_id(a: &KsArgs, y: usize) -> usize {
    el::<u32>(a.rows as _, y) as usize
}

pub extern "C" fn ks_dot<W: Hw>(a: *const KsArgs, s: S) -> i32 {
    launch(a, s, |a| if a.k > 8 { -1 } else { 0 }, |a| {
        par_for(a.nrows as usize, a, |a, y| {
            let t = row_id(a, y);
            for r in 0..a.r as usize {
                let (kb, ka) = (el::<*const std::ffi::c_void>(a.kb as _, r), el::<*const std::ffi::c_void>(a.ka as _, r));
                let off = |d: usize| d * a.key_digit_stride as usize + t * N;
                let (s0, s1) = dot_rows::<W>(a, t, &mut |d, b, ar| {
                    b.copy_from_slice(row::<W>(kb, off(d)));
                    ar.copy_from_slice(row::<W>(ka, off(d)));
                });
                store_rot::<W>(a, t, r, &s0, &s1);
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
        par_for(g.ks.nrows as usize, g, |g, y| {
            let (a, kt) = (&g.ks, &g.key);
            let t = row_id(a, y);
            let p = prime(a.pr, t).p;
            let slot = slot_of(kt, t, a.nq as usize);
            for r in 0..a.r as usize {
                let kb = el::<*const std::ffi::c_void>(kt.kb as _, r);
                let seed = seed_at(el::<*const u32>(kt.seed as _, r));
                let (s0, s1) = dot_rows::<W>(a, t, &mut |d, b, ar| {
                    if (d as i32) < kt.nregen {
                        expand_row::<W>(&seed, d as u32, p, TAG_KSKA, ar);
                    } else {
                        let ka = el::<*const std::ffi::c_void>(a.ka as _, r);
                        ar.copy_from_slice(row::<W>(ka, d * a.key_digit_stride as usize + t * N));
                    }
                    b_row::<W>(kt, kb, d * kt.srows as usize + slot, b);
                });
                store_rot::<W>(a, t, r, &s0, &s1);
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
        par_for(g.ks.nrows as usize, g, |g, y| {
            let (a, kt, e) = (&g.ks, &g.key, &g.epi);
            let t = row_id(a, y);
            let q = prime(a.pr, t);
            let p = W::from_u64(q.p);
            let seed = seed_at(g.seed);
            let tab = !kt.kb.is_null();
            let kb = if tab { el::<*const std::ffi::c_void>(kt.kb as _, 0) } else { el::<*const std::ffi::c_void>(a.kb as _, 0) };
            let slot = if tab { slot_of(kt, t, a.nq as usize) } else { 0 };
            let (s0, s1) = dot_rows::<W>(a, t, &mut |d, b, ar| {
                expand_row::<W>(&seed, d as u32, q.p, TAG_KSKA, ar);
                if tab {
                    b_row::<W>(kt, kb, d * kt.srows as usize + slot, b);
                } else {
                    b.copy_from_slice(row::<W>(kb, d * a.key_digit_stride as usize + t * N));
                }
            });
            for c in 0..a.k as usize {
                let ob = c * a.out_ct_stride as usize + t * N;
                if g.use_epi == 0 {
                    let (o0, o1) = (a.out0 as *mut W, a.out1 as *mut W);
                    for i in 0..N {
                        // SAFETY: output words of row t
                        unsafe {
                            *o0.add(ob + i) = W::reduce(s0[c * N + i], q);
                            *o1.add(ob + i) = W::reduce(s1[c * N + i], q);
                        }
                    }
                    continue;
                }
                let (bi, bis) = (el::<W>(e.binv, t), el::<W>(e.binvs, t));
                let kk = if e.k.is_null() { W::default() } else { W::red1(el::<W>(e.k, t), p) };
                let al = (!e.alpha.is_null()).then(|| (el::<W>(e.alpha, t), el::<W>(e.alphas, t)));
                for part in 0..2 {
                    let off = ob + part * e.part_stride as usize;
                    let (dp, cv) = ((e.dst as *mut W), row::<W>(e.conv, off));
                    let add = (!g.add.is_null()).then(|| row::<W>(g.add, off));
                    let sx = if part == 0 { &s0 } else { &s1 };
                    for i in 0..N {
                        let x = W::reduce(sx[c * N + i], q);
                        // SAFETY: epilogue destination word of row t
                        let dv = unsafe { *dp.add(off + i) };
                        let dv = match al {
                            Some((ai, ais)) => W::mul_shoup(dv, ai, ais, p),
                            None => dv,
                        };
                        let mut r = W::add(dv, W::mul_shoup(W::sub(x, cv[i], p), bi, bis, p), p);
                        if e.dbl != 0 {
                            r = W::add(r, r, p);
                        }
                        if part == 0 {
                            r = W::add(r, kk, p);
                        }
                        if let Some(ad) = add {
                            r = W::add(r, ad[i], p);
                        }
                        // SAFETY: as above
                        unsafe { *dp.add(off + i) = r };
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
