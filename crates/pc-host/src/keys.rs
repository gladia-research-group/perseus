// Secret-key encryption and switching-key generation (encrypt.cuh, keygen.cuh): `a` from the seeded expander
// (ENCA / KSKA lanes), b written dense or as a little-endian bit stream of `bits`-wide values with zeroed guard
// words; ksk_unpack reads such rows back. One task per output row.
use crate::arith::Hw;
use crate::expand::{expand_row, seed_at, TAG_ENCA, TAG_KSKA};
use crate::launch;
use crate::raw::{el, par_for};
use pc_kernels::args::*;

type S = CudaStream;

fn rowp<W>(p: *const std::ffi::c_void, off: usize) -> &'static [W] {
    // SAFETY: a row of N words inside the launch's buffers
    unsafe { std::slice::from_raw_parts((p as *const W).add(off), N) }
}
fn rowm<W>(p: *mut std::ffi::c_void, off: usize) -> &'static mut [W] {
    // SAFETY: an output row of N words (one task per row)
    unsafe { std::slice::from_raw_parts_mut((p as *mut W).add(off), N) }
}

pub extern "C" fn encrypt<W: Hw>(a: *const EncryptArgs, s: S) -> i32 {
    launch(a, s, |_| 0, |a| {
        par_for(a.nl as usize, a, |a, l| {
            // SAFETY: prime table entry l
            let q = unsafe { &*a.pr.add(l) };
            let p = W::from_u64(q.p);
            let mut ar = crate::raw::buf::<W>(N);
            expand_row::<W>(&seed_at(a.seed), a.nonce, q.p, TAG_ENCA, &mut ar);
            let (sk, e) = (rowp::<W>(a.s, l * N), rowp::<W>(a.e, l * N));
            let m = (!a.m.is_null()).then(|| rowp::<W>(a.m, l * N));
            let c0 = rowm::<W>(a.c0, l * N);
            for i in 0..N {
                let mi = m.map_or(W::default(), |m| m[i]);
                c0[i] = W::sub(W::add(e[i], mi, p), W::mul(ar[i], sk[i], q), p);
            }
            if !a.c1.is_null() {
                rowm::<W>(a.c1, l * N).copy_from_slice(&ar);
            }
        })
    })
}

/// Pack N values of `bits` bits into a little-endian u32 stream, zeroing the row's words past them (the guard).
fn pack(vals: &[u32], bits: u32, out: &mut [u32]) {
    out.fill(0);
    let mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
    for (s, &v) in vals.iter().enumerate() {
        let bit = s as u64 * bits as u64;
        let (w, o) = ((bit >> 5) as usize, (bit & 31) as u32);
        let x = ((v & mask) as u64) << o;
        out[w] |= x as u32;
        if o + bits > 32 {
            out[w + 1] |= (x >> 32) as u32;
        }
    }
}

pub extern "C" fn ksk_gen<W: Hw>(a: *const KskGenArgs, s: S) -> i32 {
    let bad = |a: &KskGenArgs| {
        if a.bits < 0 || a.bits > 32 || (a.bits > 0 && W::BITS != 32) {
            -1
        } else if a.bits > 0 && a.row_words < (N as u64 * a.bits as u64) / 32 + 1 {
            -2
        } else {
            0
        }
    };
    launch(a, s, bad, |a| {
        let nr = a.nrows as usize;
        par_for(a.ndig as usize * nr, a, |a, t| {
            let (d, m) = (t / nr, t % nr);
            let id = el::<i32>(a.ids as _, m) as usize;
            // SAFETY: prime table entry id
            let q = unsafe { &*a.pr.add(id) };
            let p = W::from_u64(q.p);
            let mut ar = crate::raw::buf::<W>(N);
            expand_row::<W>(&seed_at(a.seed), d as u32, q.p, TAG_KSKA, &mut ar);
            let row = d * nr + m;
            let (e, sn) = (rowp::<W>(a.e, row * N), rowp::<W>(a.s_new, id * N));
            let in_digit = (id as i32) < a.nq && el::<i32>(a.digit_of as _, id) == d as i32;
            let (so, pm, pms) = if in_digit {
                (Some(rowp::<W>(a.s_old, id * N)), el::<W>(a.pmod, id), el::<W>(a.pmods, id))
            } else {
                (None, W::default(), W::default())
            };
            let r: Vec<W> = (0..N)
                .map(|i| {
                    let mut x = e[i];
                    if let Some(so) = so {
                        let mut t = so[i];
                        if a.sq != 0 {
                            t = W::mul(t, t, q);
                        }
                        x = W::add(x, W::mul_shoup(t, pm, pms, p), p);
                    }
                    W::sub(x, W::mul(ar[i], sn[i], q), p)
                })
                .collect();
            if !a.a.is_null() {
                rowm::<W>(a.a, row * N).copy_from_slice(&ar);
            }
            if a.bits == 0 {
                rowm::<W>(a.b, row * N).copy_from_slice(&r);
                return;
            }
            let vals: Vec<u32> = r.iter().map(|v| v.u64() as u32).collect();
            // SAFETY: packed row of row_words u32 words
            let out = unsafe { std::slice::from_raw_parts_mut((a.b as *mut u32).add(row * a.row_words as usize), a.row_words as usize) };
            pack(&vals, a.bits as u32, out);
        })
    })
}

pub extern "C" fn ksk_unpack<W: Hw>(a: *const KskUnpackArgs, s: S) -> i32 {
    let bad = |a: &KskUnpackArgs| if a.bits <= 0 || a.bits > 32 || a.row_words < (N as u64 * a.bits as u64) / 32 + 1 { -1 } else { 0 };
    launch(a, s, bad, |a| {
        par_for(a.nrows as usize, a, |a, r| {
            // SAFETY: packed row r
            let w = unsafe { std::slice::from_raw_parts(a.inp.add(r * a.row_words as usize), a.row_words as usize) };
            let bits = a.bits as u64;
            let mask = if bits == 32 { u32::MAX as u64 } else { (1u64 << bits) - 1 };
            let out = rowm::<W>(a.out, r * N);
            for (i, o) in out.iter_mut().enumerate() {
                let bit = i as u64 * bits;
                let (j, sh) = ((bit >> 5) as usize, bit & 31);
                let v = (w[j] as u64 | (w[j + 1] as u64) << 32) >> sh;
                *o = W::from_u64(v & mask);
            }
        })
    })
}
