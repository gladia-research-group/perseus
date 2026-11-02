// Negacyclic NTT rows (ntt.cuh semantics, NttArgs): OpenFHE order, CT forward / GS inverse with N^-1, out-of-place
// sources summed mod p, the forward epilogues. One task per row. At u32 with `lazy` the butterflies are the device's
// lazy (Harvey) ones, so the words an epilogue run leaves in the row (the state after the column pass, stages
// m = 1..128) match the device's too; every transform returns canonical words.
use crate::arith::Hw;
use crate::launch;
use crate::raw::{el, par_for};
use pc_kernels::args::*;

type S = CudaStream;

/// Stage count of the device's column pass (m = 1..128): the forward epilogue leaves this state in the row.
const COL_STAGES: u32 = 8;

#[inline(always)]
fn red2p(a: u32, p: u32) -> u32 {
    if a >= 2 * p {
        a - 2 * p
    } else {
        a
    }
}
#[inline(always)]
fn shoup_lz(a: u32, w: u32, ws: u32, p: u32) -> u32 {
    let q = ((a as u64 * ws as u64) >> 32) as u32;
    a.wrapping_mul(w).wrapping_sub(q.wrapping_mul(p))
}

/// Lazy forward (u32, p < 2^30): words in [0, 4p) between stages; `mid` receives the column-pass state.
#[inline(always)]
fn fwd_lazy(x: &mut [u32], w: &[u32], ws: &[u32], p: u32, mid: Option<&mut [u32]>) {
    let mut mid = mid;
    let mut m = 1;
    let mut stage = 0;
    while m < N {
        let t = N / (2 * m);
        for i in 0..m {
            let (tw, tws) = (w[m + i], ws[m + i]);
            let (lo, hi) = x[2 * i * t..2 * i * t + 2 * t].split_at_mut(t);
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                let u = red2p(*a, p);
                let v = shoup_lz(*b, tw, tws, p);
                *a = u + v;
                *b = u.wrapping_sub(v).wrapping_add(2 * p);
            }
        }
        stage += 1;
        if stage == COL_STAGES {
            if let Some(o) = mid.take() {
                o.copy_from_slice(x);
            }
        }
        m *= 2;
    }
    for a in x.iter_mut() {
        let r = red2p(*a, p);
        *a = if r >= p { r - p } else { r };
    }
}

/// Lazy inverse (u32, p < 2^30): words in [0, 2p) between stages, N^-1 last.
#[inline(always)]
fn inv_lazy(x: &mut [u32], w: &[u32], ws: &[u32], p: u32, ni: u32, nis: u32) {
    let mut m = N / 2;
    while m >= 1 {
        let t = N / (2 * m);
        for i in 0..m {
            let (tw, tws) = (w[m + i], ws[m + i]);
            let (lo, hi) = x[2 * i * t..2 * i * t + 2 * t].split_at_mut(t);
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                let (u, v) = (*a, *b);
                *a = red2p(u + v, p);
                *b = shoup_lz(u.wrapping_sub(v).wrapping_add(2 * p), tw, tws, p);
            }
        }
        m /= 2;
    }
    for a in x.iter_mut() {
        *a = <u32 as Hw>::mul_shoup(*a, ni, nis, p);
    }
}

/// Exact forward (any word): canonical words throughout.
#[inline(always)]
pub(crate) fn fwd_exact<W: Hw>(x: &mut [W], w: &[W], ws: &[W], p: W, mid: Option<&mut [W]>) {
    let mut mid = mid;
    let mut m = 1;
    let mut stage = 0;
    while m < N {
        let t = N / (2 * m);
        for i in 0..m {
            let (tw, tws) = (w[m + i], ws[m + i]);
            let (lo, hi) = x[2 * i * t..2 * i * t + 2 * t].split_at_mut(t);
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                let v = W::mul_shoup(*b, tw, tws, p);
                let u = *a;
                *a = W::add(u, v, p);
                *b = W::sub(u, v, p);
            }
        }
        stage += 1;
        if stage == COL_STAGES {
            if let Some(o) = mid.take() {
                o.copy_from_slice(x);
            }
        }
        m *= 2;
    }
}

#[inline(always)]
fn inv_exact<W: Hw>(x: &mut [W], w: &[W], ws: &[W], p: W, ni: W, nis: W) {
    let mut m = N / 2;
    while m >= 1 {
        let t = N / (2 * m);
        for i in 0..m {
            let (tw, tws) = (w[m + i], ws[m + i]);
            let (lo, hi) = x[2 * i * t..2 * i * t + 2 * t].split_at_mut(t);
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                let (u, v) = (*a, *b);
                *a = W::add(u, v, p);
                *b = W::mul_shoup(W::sub(u, v, p), tw, tws, p);
            }
        }
        m /= 2;
    }
    for a in x.iter_mut() {
        *a = W::mul_shoup(*a, ni, nis, p);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl,avx2,bmi2")]
unsafe fn fwd_lazy_v(x: &mut [u32], w: &[u32], ws: &[u32], p: u32, mid: Option<&mut [u32]>) {
    crate::ntt_avx::fwd(x, w, ws, p, mid)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl,avx2,bmi2")]
unsafe fn inv_lazy_v(x: &mut [u32], w: &[u32], ws: &[u32], p: u32, ni: u32, nis: u32) {
    crate::ntt_avx::inv(x, w, ws, p, ni, nis)
}

fn wide() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *W.get_or_init(|| std::env::var_os("PC_HOST_SCALAR").is_none_or(|v| !v.to_string_lossy().contains("NTT")) && std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512dq") && std::is_x86_feature_detected!("avx512vl"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// As u32 when W is u32 (the lazy path's word).
fn as_u32<W: Hw>(x: &mut [W]) -> Option<&mut [u32]> {
    // SAFETY: same type, checked by size and BITS
    (W::BITS == 32 && std::mem::size_of::<W>() == 4).then(|| unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut u32, x.len()) })
}
fn as_u32c<W: Hw>(x: &[W]) -> &[u32] {
    // SAFETY: only called when W is u32
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u32, x.len()) }
}

/// The input row i: its own words, or the sum mod p of the snum source rows.
fn load<W: Hw>(a: &NttArgs, i: usize, p: W, buf: &mut [W]) {
    let dst = a.base as *const W;
    // SAFETY: row offsets / sources inside the launch's buffers
    unsafe {
        if a.soff.is_null() {
            buf.copy_from_slice(std::slice::from_raw_parts(dst.add(*a.roff.add(i) as usize * N), N));
            return;
        }
        let src = (a.src as *const W).add(*a.soff.add(i) as usize * N);
        buf.copy_from_slice(std::slice::from_raw_parts(src, N));
        for m in 1..a.snum.max(1) as usize {
            let s = std::slice::from_raw_parts(src.add(m * a.sstride as usize), N);
            buf.iter_mut().zip(s).for_each(|(b, &v)| *b = W::add(*b, v, p));
        }
    }
}

fn table<W: Hw>(t: *const std::ffi::c_void, pid: usize) -> &'static [W] {
    // SAFETY: twiddle table [prime id][N] (device constant data)
    unsafe { std::slice::from_raw_parts((t as *const W).add(pid * N), N) }
}

pub extern "C" fn fwd<W: Hw>(a: *const NttArgs, s: S) -> i32 {
    let bad = |a: &NttArgs| {
        if a.epi < 0 || a.epi > 3 || (a.epi != 0 && (a.eout.is_null() || a.ex.is_null() || a.es.is_null() || a.ess.is_null())) {
            -1
        } else {
            0
        }
    };
    launch(a, s, bad, |a| fwd_with::<W>(a, &|a, i, p, x| load(a, i, p, x)))
}

/// The forward transform of the launch's rows (epilogues included), row i's input from `src(a, i, p, x)`.
pub(crate) fn fwd_with<W: Hw>(a: &NttArgs, src: &(dyn Fn(&NttArgs, usize, W, &mut [W]) + Sync)) {
    let sh = crate::raw::Shared(src);
    let sh = &sh;
    {
        par_for(a.nrows as usize, a, |a, i| {
            // SAFETY: rows / prime ids of nrows entries
            let (roff, pid) = unsafe { (*a.roff.add(i) as usize, *a.rprime.add(i) as usize) };
            let p = W::from_u64(el::<PrimeConst>(a.pr as *const std::ffi::c_void, pid).p);
            let (w, ws) = (table::<W>(a.fw, pid), table::<W>(a.fws, pid));
            let mut x = crate::raw::buf::<W>(N);
            (sh.get())(a, i, p, &mut x);
            // SAFETY: row roff of the base buffer
            let row = unsafe { std::slice::from_raw_parts_mut((a.base as *mut W).add(roff * N), N) };
            let mid = if a.epi != 0 { Some(&mut row[..]) } else { None };
            let lazy = a.lazy != 0;
            match as_u32(&mut x) {
                Some(x32) if lazy => {
                    let mid = mid.map(|m| as_u32(m).unwrap());
                    #[cfg(target_arch = "x86_64")]
                    if wide() {
                        // SAFETY: CPU features checked
                        unsafe { fwd_lazy_v(x32, as_u32c(w), as_u32c(ws), p.u64() as u32, mid) };
                    } else {
                        fwd_lazy(x32, as_u32c(w), as_u32c(ws), p.u64() as u32, mid);
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    fwd_lazy(x32, as_u32c(w), as_u32c(ws), p.u64() as u32, mid);
                }
                _ => fwd_exact(&mut x, w, ws, p, mid),
            }
            if a.epi == 0 {
                row.copy_from_slice(&x);
                return;
            }
            let (sc, ss) = (el::<W>(a.es, pid), el::<W>(a.ess, pid));
            let r = roff * N;
            for (e, &v) in x.iter().enumerate() {
                let mut xs = el::<W>(a.ex, r + e);
                for j in 1..a.xnum.max(1) as usize {
                    xs = W::add(xs, el::<W>(a.ex, r + e + j * a.xstride as usize), p);
                }
                // SAFETY: word r + e of the epilogue output
                let o = unsafe { (a.eout as *mut W).add(r + e) };
                let y = match a.epi {
                    3 => W::add(W::mul_shoup(xs, sc, ss, p), v, p),
                    1 => W::mul_shoup(W::sub(xs, v, p), sc, ss, p),
                    _ => {
                        // SAFETY: the accumulated term, read before the store
                        let d = if a.ed.is_null() { unsafe { *o } } else { el::<W>(a.ed, r + e) };
                        W::add(d, W::mul_shoup(W::sub(xs, v, p), sc, ss, p), p)
                    }
                };
                // SAFETY: as above
                unsafe { *o = y };
            }
        })
    }
}

pub extern "C" fn inv<W: Hw>(a: *const NttArgs, s: S) -> i32 {
    launch(a, s, |a| if a.epi != 0 { -1 } else { 0 }, |a| {
        par_for(a.nrows as usize, a, |a, i| {
            // SAFETY: rows / prime ids of nrows entries
            let (roff, pid) = unsafe { (*a.roff.add(i) as usize, *a.rprime.add(i) as usize) };
            let p = W::from_u64(el::<PrimeConst>(a.pr as *const std::ffi::c_void, pid).p);
            let (w, ws) = (table::<W>(a.iw, pid), table::<W>(a.iws, pid));
            let (ni, nis) = (el::<W>(a.ninv, pid), el::<W>(a.ninvs, pid));
            let mut x = crate::raw::buf::<W>(N);
            load(a, i, p, &mut x);
            match as_u32(&mut x) {
                Some(x32) if a.lazy != 0 => {
                    let (pp, n1, n2) = (p.u64() as u32, ni.u64() as u32, nis.u64() as u32);
                    #[cfg(target_arch = "x86_64")]
                    if wide() {
                        // SAFETY: CPU features checked
                        unsafe { inv_lazy_v(x32, as_u32c(w), as_u32c(ws), pp, n1, n2) };
                    } else {
                        inv_lazy(x32, as_u32c(w), as_u32c(ws), pp, n1, n2);
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    inv_lazy(x32, as_u32c(w), as_u32c(ws), pp, n1, n2);
                }
                _ => inv_exact(&mut x, w, ws, p, ni, nis),
            }
            // SAFETY: row roff of the base buffer
            unsafe { std::slice::from_raw_parts_mut((a.base as *mut W).add(roff * N), N) }.copy_from_slice(&x);
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_rows_are_the_scalar_rows() {
        if !wide() {
            return;
        }
        let p = 268042241u32;
        let mut st = 0x1234_5678_9abc_def0u64;
        let mut rnd = |m: u32| {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 33) as u32) % m
        };
        let w: Vec<u32> = (0..N).map(|_| rnd(p)).collect();
        let ws: Vec<u32> = w.iter().map(|&v| (((v as u64) << 32) / p as u64) as u32).collect();
        let x: Vec<u32> = (0..N).map(|_| rnd(p)).collect();
        let (mut a, mut b) = (x.clone(), x.clone());
        let (mut ma, mut mb) = (vec![0u32; N], vec![0u32; N]);
        fwd_lazy(&mut a, &w, &ws, p, Some(&mut ma));
        // SAFETY: features checked
        unsafe { fwd_lazy_v(&mut b, &w, &ws, p, Some(&mut mb)) };
        assert_eq!(ma, mb);
        assert_eq!(a, b);
        let ni = rnd(p);
        let nis = (((ni as u64) << 32) / p as u64) as u32;
        inv_lazy(&mut a, &w, &ws, p, ni, nis);
        // SAFETY: as above
        unsafe { inv_lazy_v(&mut b, &w, &ws, p, ni, nis) };
        assert_eq!(a, b);
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    #[test]
    #[ignore]
    fn row_times() {
        let p = 268042241u32;
        let w: Vec<u32> = (0..N as u32).map(|i| i.wrapping_mul(2654435761) % p).collect();
        let ws: Vec<u32> = w.iter().map(|&v| (((v as u64) << 32) / p as u64) as u32).collect();
        let mut x: Vec<u32> = (0..N as u32).map(|i| i % p).collect();
        let reps = 200;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            fwd_lazy(&mut x, &w, &ws, p, None);
        }
        let s = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            // SAFETY: bench on an AVX-512 machine
            unsafe { fwd_lazy_v(&mut x, &w, &ws, p, None) };
        }
        let v = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            // SAFETY: as above
            unsafe { inv_lazy_v(&mut x, &w, &ws, p, 5, 80) };
        }
        let vi = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
        eprintln!("ntt row: scalar fwd {s:.1} us, avx512 fwd {v:.1} us, avx512 inv {vi:.1} us");
    }
}
