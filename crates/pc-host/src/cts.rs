// Fused CtS / StC stage core (cts.cuh, CtsArgs): hoisted baby rotations of the ModUp'd input and the BSGS plaintext
// dot, the rotations never stored. For row t, baby i >= 1 is computed in source order (key products at slot s) and
// gathered through s = sigma_i^-1(o); baby 0 is P x on Q rows (0 on special rows). The regen form reads `b` through
// the KeyTab and expands `a` (the KSKA lane) for its first nregen digits. Canonical outputs, so the device's partial
// reductions every 8 baby steps leave no trace.
use crate::arith::{automorph_slot, ptidx, Hw};
use crate::expand::{expand_row, seed_at, TAG_KSKA};
use crate::ks::slot_of;
use crate::launch;
use crate::raw::{buf, el, par_for};
use crate::simd::w as v;
use pc_kernels::args::*;

type S = CudaStream;

fn rowp<W>(p: *const std::ffi::c_void, off: usize) -> &'static [W] {
    // SAFETY: a row of N words inside the launch's buffers
    unsafe { std::slice::from_raw_parts((p as *const W).add(off), N) }
}

/// Outputs per task: 64 aligned 32-slot blocks of one row (the automorphisms map aligned blocks to aligned blocks).
const CH: usize = 2048;
const BL: usize = 32;

fn run<W: Hw>(r: &CtsRegenArgs, regen: bool) {
    let a = &r.cts;
    let t0 = std::time::Instant::now();
    let _g = crate::raw::Defer(move || {
        if std::env::var_os("PC_HOST_CTS").is_some() {
            eprintln!("[cts] rows {} K {} B {} G {} npts {} ndig {} nregen {} bits {} enc {:#x}: {:.1} ms", a.nrows, a.k, a.b, a.g, a.npts, a.ndig, r.key.nregen, r.key.bits, a.enc, t0.elapsed().as_secs_f64() * 1e3);
        }
    });
    let nch = N / CH;
    // the launch's expanded `a` rows (u32 regen: baby i >= 1, digit d < nregen, row y), fetched once in parallel
    let nreg = if regen && W::BITS == 32 { (a.ndig.min(r.key.nregen)).max(0) as usize } else { 0 };
    let (nb, nr) = (a.b.max(1) as usize - 1, a.nrows as usize);
    let rows: Vec<std::sync::OnceLock<std::sync::Arc<[u32]>>> = (0..nb * nreg * nr).map(|_| std::sync::OnceLock::new()).collect();
    if nreg > 0 {
        par_for(rows.len(), r, |r, z| {
            let (i, d, y) = (z / (nreg * nr) + 1, (z / nr) % nreg, z % nr);
            let t = el::<u32>(r.cts.rows as _, y) as usize;
            // SAFETY: prime table entry t
            let p = unsafe { (*r.cts.pr.add(t)).p } as u32;
            let seed = seed_at(el::<*const u32>(r.key.seed as _, i - 1));
            let _ = rows[z].set(crate::expand::cache::get(&seed, d as u32, p, N));
        });
    }
    let rows = crate::raw::Shared(&rows);
    let rows = &rows;
    par_for(a.nrows as usize * nch, r, |r, task| {
        let (y, o0) = (task / nch, (task % nch) * CH);
        let (a, kt) = (&r.cts, &r.key);
        let t = el::<u32>(a.rows as _, y) as usize;
        // SAFETY: prime table entry t
        let q = unsafe { &*a.pr.add(t) };
        let p = W::from_u64(q.p);
        let isq = (t as i32) < a.nq;
        let prow = if isq { t } else { a.pt_prow as usize + (t - a.nq as usize) };
        let dt = if isq { el::<i32>(a.digit_of as _, t) as i64 } else { -1 };
        let (pm, pms) = (el::<W>(a.pmod, t), el::<W>(a.pmods, t));
        let (k, g, b) = (a.k as usize, a.g as usize, a.b as usize);
        let rowoff = t * N;
        let x = |c: usize, part: usize| rowp::<W>(a.x, c * a.ct_stride as usize + part * a.part_stride as usize + rowoff);
        let mut acc0 = buf::<W::Acc>(k * g * CH);
        let mut acc1 = buf::<W::Acc>(k * g * CH);
        let (mut s0, mut s1) = (buf::<W::Acc>(CH), buf::<W::Acc>(CH));
        let (mut v0, mut v1) = (buf::<W>(k * CH), buf::<W>(k * CH));
        let (mut xs, mut kb, mut ka) = (buf::<W>(CH), buf::<W>(CH), buf::<W>(CH));
        let (mut r0, mut r1, mut wrow) = (buf::<W>(CH), buf::<W>(CH), buf::<W>(CH));
        let mut gidx = buf::<u32>(CH);
        let mut sblk = [0usize; CH / BL];
        let ident = a.enc == u32::MAX;
        let slot_t = if regen { slot_of(kt, t, a.nq as usize) } else { 0 };
        // block copy of the chunk's source blocks of a row into `dst`
        let gather_blocks = |dst: &mut [W], src: &[W], sblk: &[usize]| {
            for (bi, &sb) in sblk.iter().enumerate() {
                dst[bi * BL..(bi + 1) * BL].copy_from_slice(&src[sb..sb + BL]);
            }
        };
        for i in 0..b {
            if i == 0 {
                for c in 0..k {
                    let (d0, d1) = (&mut v0[c * CH..(c + 1) * CH], &mut v1[c * CH..(c + 1) * CH]);
                    if isq {
                        v::shoup_row::<W>(d0, &x(c, 0)[o0..o0 + CH], pm, pms, p, true);
                        v::shoup_row::<W>(d1, &x(c, 1)[o0..o0 + CH], pm, pms, p, true);
                    } else {
                        d0.fill(W::default());
                        d1.fill(W::default());
                    }
                }
            } else {
                // the source block of every output block, and each output's position among the gathered sources
                let kinv = el::<u32>(a.kinv as _, i - 1);
                for (bi, sb) in sblk.iter_mut().enumerate() {
                    let o = o0 + bi * BL;
                    *sb = (automorph_slot(kinv, o as u32, LOG_N) as usize) & !(BL - 1);
                    for l in 0..BL {
                        gidx[bi * BL + l] = (bi * BL + (automorph_slot(kinv, (o + l) as u32, LOG_N) as usize - *sb)) as u32;
                    }
                }
                for c in 0..k {
                    s0.fill(W::Acc::default());
                    s1.fill(W::Acc::default());
                    for d in 0..a.ndig as usize {
                        // key rows of digit d at the chunk's source blocks: `b` stored (dense or packed), `a` expanded
                        // (cached rows) or resident
                        if c == 0 || k > 1 {
                            let pre = (d < nreg).then(|| &**rows.get()[((i - 1) * nreg + d) * nr + y].get().unwrap());
                            fetch_keys::<W>(r, regen, i, d, t, slot_t, q, &sblk, &mut kb, &mut ka, pre);
                        }
                        let xv = if d as i64 == dt {
                            x(c, 1)
                        } else {
                            rowp::<W>(a.modup, c * a.mu_ct_stride as usize + d * a.mu_digit_stride as usize + rowoff)
                        };
                        gather_blocks(&mut xs, xv, &sblk);
                        v::mac2::<W>(&mut s0, &mut s1, &xs, &kb, &ka);
                    }
                    v::reduce::<W>(&s0, q, &mut r0);
                    v::reduce::<W>(&s1, q, &mut r1);
                    if isq {
                        gather_blocks(&mut xs, x(c, 0), &sblk);
                        v::shoup_row::<W>(&mut r0, &xs, pm, pms, p, false);
                    }
                    v::gather::<W>(&mut v0[c * CH..(c + 1) * CH], &r0, &gidx);
                    v::gather::<W>(&mut v1[c * CH..(c + 1) * CH], &r1, &gidx);
                }
            }
            for j in 0..g {
                let dg = b * j + i;
                if dg >= a.npts as usize {
                    continue;
                }
                // SAFETY: diagonal dg's row prow (ptw words)
                let pt = unsafe { (*(a.pts as *const *const W).add(dg)).add(prow * a.ptw as usize) };
                let wr: &[W] = if ident {
                    // SAFETY: a full diagonal row of N words
                    unsafe { std::slice::from_raw_parts(pt.add(o0), CH) }
                } else {
                    for (l, wo) in wrow.iter_mut().enumerate() {
                        // SAFETY: compact index inside the diagonal's row
                        *wo = unsafe { *pt.add(ptidx(o0 + l, a.enc)) };
                    }
                    &wrow
                };
                for c in 0..k {
                    let base = (c * g + j) * CH;
                    v::mac::<W>(&mut acc0[base..base + CH], &v0[c * CH..(c + 1) * CH], wr);
                    v::mac::<W>(&mut acc1[base..base + CH], &v1[c * CH..(c + 1) * CH], wr);
                }
            }
            if i & 7 == 7 {
                // the device's headroom reductions (canonical either way; keeps the host accumulators bounded too)
                for acc in [&mut acc0, &mut acc1] {
                    for chunk in acc.chunks_mut(CH) {
                        v::reduce::<W>(chunk, q, &mut r0);
                        for (v, &r) in chunk.iter_mut().zip(r0.iter()) {
                            *v = W::wmul(r, W::from_u64(1));
                        }
                    }
                }
            }
        }
        for c in 0..k {
            for j in 0..g {
                let at = if a.pair_cts > 0 { (j >> 1) * a.pair_cts as usize + (j & 1) } else { c * g + j };
                let res = (a.res as *mut W).wrapping_add(at * a.ct_stride as usize + rowoff + o0);
                let base = (c * g + j) * CH;
                // SAFETY: the chunk of the result rows of (c, j), both parts
                let (d0, d1) = unsafe { (std::slice::from_raw_parts_mut(res, CH), std::slice::from_raw_parts_mut(res.add(a.part_stride as usize), CH)) };
                v::reduce::<W>(&acc0[base..base + CH], q, d0);
                v::reduce::<W>(&acc1[base..base + CH], q, d1);
            }
        }
    })
}

/// Baby i's key words of digit d at the source blocks `sblk` of row t, block after block.
#[allow(clippy::too_many_arguments)]
fn fetch_keys<W: Hw>(r: &CtsRegenArgs, regen: bool, i: usize, d: usize, t: usize, slot_t: usize, q: &PrimeConst, sblk: &[usize], kb: &mut [W], ka: &mut [W], pre: Option<&[u32]>) {
    let (a, kt) = (&r.cts, &r.key);
    let rowoff = t * N;
    if !regen {
        let (kbp, kap) = (el::<*const std::ffi::c_void>(a.kb as _, i - 1), el::<*const std::ffi::c_void>(a.ka as _, i - 1));
        let (rb, ra) = (rowp::<W>(kbp, rowoff + d * a.key_digit_stride as usize), rowp::<W>(kap, rowoff + d * a.key_digit_stride as usize));
        for (bi, &sb) in sblk.iter().enumerate() {
            kb[bi * BL..(bi + 1) * BL].copy_from_slice(&rb[sb..sb + BL]);
            ka[bi * BL..(bi + 1) * BL].copy_from_slice(&ra[sb..sb + BL]);
        }
        return;
    }
    let kbp = el::<*const std::ffi::c_void>(kt.kb as _, i - 1);
    let brow = d * kt.srows as usize + slot_t;
    if kt.bits == 0 {
        let rb = rowp::<W>(kbp, brow * kt.row_words as usize);
        for (bi, &sb) in sblk.iter().enumerate() {
            kb[bi * BL..(bi + 1) * BL].copy_from_slice(&rb[sb..sb + BL]);
        }
    } else {
        // 32 values of `bits` bits are `bits` whole words: a source block unpacks from its own words
        let bits = kt.bits as usize;
        // SAFETY: packed row of row_words u32 words
        let words = unsafe { std::slice::from_raw_parts((kbp as *const u32).add(brow * kt.row_words as usize), kt.row_words as usize) };
        for (bi, &sb) in sblk.iter().enumerate() {
            let w0 = sb / BL * bits;
            // SAFETY: W is u32 for packed keys (launch check)
            let out = unsafe { std::slice::from_raw_parts_mut(kb[bi * BL..].as_mut_ptr() as *mut u32, BL) };
            crate::simd::unpack(&words[w0..(w0 + bits + 2).min(words.len())], bits as u32, out);
        }
    }
    if (d as i32) < kt.nregen {
        let seed = seed_at(el::<*const u32>(kt.seed as _, i - 1));
        if let Some(row) = pre {
            // SAFETY: W is u32 (the prefetch exists only then)
            let row = unsafe { std::slice::from_raw_parts(row.as_ptr() as *const W, N) };
            for (bi, &sb) in sblk.iter().enumerate() {
                ka[bi * BL..(bi + 1) * BL].copy_from_slice(&row[sb..sb + BL]);
            }
        } else if W::BITS == 32 {
            let row = crate::expand::cache::get(&seed, d as u32, q.p as u32, N);
            // SAFETY: W is u32
            let row = unsafe { std::slice::from_raw_parts(row.as_ptr() as *const W, N) };
            for (bi, &sb) in sblk.iter().enumerate() {
                ka[bi * BL..(bi + 1) * BL].copy_from_slice(&row[sb..sb + BL]);
            }
        } else {
            let mut full = buf::<W>(N);
            expand_row::<W>(&seed, d as u32, q.p, TAG_KSKA, &mut full);
            for (bi, &sb) in sblk.iter().enumerate() {
                ka[bi * BL..(bi + 1) * BL].copy_from_slice(&full[sb..sb + BL]);
            }
        }
    } else {
        let kap = el::<*const std::ffi::c_void>(a.ka as _, i - 1);
        let ra = rowp::<W>(kap, rowoff + d * a.key_digit_stride as usize);
        for (bi, &sb) in sblk.iter().enumerate() {
            ka[bi * BL..(bi + 1) * BL].copy_from_slice(&ra[sb..sb + BL]);
        }
    }
}

/// The launcher's checks (cts.cuh launch_any): G <= 2, or <= 4 for the K = 1 regen launch with a pair layout.
fn check<W: Hw>(r: &CtsRegenArgs, regen: bool) -> i32 {
    let a = &r.cts;
    let dmax = if W::BITS == 32 { 6 } else { 8 };
    let km = if a.kmax <= 1 { 1 } else if a.kmax <= 2 { 2 } else if a.kmax <= 4 { 4 } else { 8 };
    let g4 = regen && W::BITS == 32 && km == 1 && a.k == 1 && a.pair_cts > 0;
    if a.g > if g4 { 4 } else { 2 } || a.ndig > dmax || (a.pair_cts > 0 && !g4) {
        return -1;
    }
    if regen && W::BITS == 32 && a.b > 64 {
        return -1;
    }
    0
}

pub extern "C" fn cts_fused<W: Hw>(a: *const CtsArgs, s: S) -> i32 {
    // SAFETY: the launcher ABI hands a valid argument struct
    let mut r: CtsRegenArgs = unsafe { std::mem::zeroed() };
    // SAFETY: as above
    r.cts = unsafe { *a };
    launch(&r, s, |r| check::<W>(r, false), |r| run::<W>(r, false))
}

pub extern "C" fn cts_fused_regen<W: Hw>(a: *const CtsRegenArgs, s: S) -> i32 {
    let bad = |r: &CtsRegenArgs| {
        if r.key.bits < 0 || r.key.bits > 32 || (r.key.bits > 0 && W::BITS != 32) || r.cts.ndig > if W::BITS == 32 { 6 } else { 8 } {
            -1
        } else {
            check::<W>(r, true)
        }
    };
    launch(a, s, bad, |r| run::<W>(r, true))
}
