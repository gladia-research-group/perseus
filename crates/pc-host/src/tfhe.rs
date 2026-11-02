// CKKS <-> TFHE scheme switching (tfhe.cuh, TfheArgs; semantics per op in args.cuh and the kernels): the launcher's
// argument checks and return codes, every op word for word. The blind rotation (PBS_ROTATE, PBS_KEYBUNDLE) is exact
// RNS arithmetic mod Q = q_a q_b, so one host form stands for all the device's paths (batch, two-LWE blocks, latency
// key bundles, clusters, nu = 2): per group the digits of the accumulator, their NTTs, the key bundle (from the key's
// Montgomery words, or PBS_KEYBUNDLE's), the product, the inverse NTTs, the sum. LWEs run in batches that share each
// group's key words in cache; the per-word MACs run on AVX-512 in pbs_v.rs.
use crate::launch;
use crate::mm::mulm;
use crate::ntt_g;
use crate::raw::{buf, par_for};
use pc_kernels::args::*;

const RT_E: usize = 4;
const TH: usize = 1024;
const CL4_MAX: i32 = 46;

#[inline(always)]
fn rd<T: Copy>(p: *const std::ffi::c_void, i: usize) -> T {
    // SAFETY: word i of a launch buffer (the launch contract bounds it)
    unsafe { *(p as *const T).add(i) }
}
#[inline(always)]
fn wr<T>(p: *mut std::ffi::c_void, i: usize, v: T) {
    // SAFETY: as above; tasks write disjoint words
    unsafe { *(p as *mut T).add(i) = v }
}
fn rows<'a, T>(p: *const std::ffi::c_void, off: usize, n: usize) -> &'a [T] {
    // SAFETY: n words of a launch buffer
    unsafe { std::slice::from_raw_parts((p as *const T).add(off), n) }
}
fn rows_mut<'a, T>(p: *mut std::ffi::c_void, off: usize, n: usize) -> &'a mut [T] {
    // SAFETY: n words of a launch buffer, disjoint per task
    unsafe { std::slice::from_raw_parts_mut((p as *mut T).add(off), n) }
}

#[inline(always)]
fn addq(a: u64, b: u64, q: u64) -> u64 {
    let s = a.wrapping_add(b);
    if s >= q {
        s.wrapping_sub(q)
    } else {
        s
    }
}
#[inline(always)]
fn subq(a: u64, b: u64, q: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a.wrapping_add(q).wrapping_sub(b)
    }
}
#[inline(always)]
pub fn scale_round(x: u64, m: u64) -> u64 {
    ((x as u128 * m as u128 + (1u128 << 63)) >> 64) as u64
}
/// Every digit of the approximate balanced decomposition (tfhe.cuh `digit` for t = 0 .. lv - 1, which repeats the
/// same sequence up to t): out[t].
fn digits_of(x: u64, bl: u32, lv: usize, m: u64, out: &mut [i64]) {
    let bits = bl as usize * lv;
    let mut v = scale_round(x, m) as i64;
    if v >= 1i64 << (bits - 1) {
        v -= 1i64 << bits;
    }
    let b = 1i64 << bl;
    for (t, o) in out.iter_mut().enumerate().take(lv) {
        let mut r = if bl == 1 {
            if v & 1 != 0 {
                2 - (v & 3)
            } else {
                0
            }
        } else {
            let r = v & (b - 1);
            if r >= b / 2 {
                r - b
            } else {
                r
            }
        };
        v = (v - r) >> bl;
        if t == lv - 1 {
            r += v << bl;
        }
        *o = r;
    }
}
/// `digits_of` for up to 8 words at once, as int8 (out[l * lv + t]): branch-free lanes the compiler vectorizes.
/// scale_round(x, m) for x < 2^32 is (x m_hi + (x m_lo >> 32) + 2^31) >> 32 (the low word only adds a fraction).
#[inline(always)]
fn digits8(x: &[u32], bl: u32, lv: usize, m: u64, out: &mut [i8]) {
    let bits = bl as usize * lv;
    let (mh, ml) = (m >> 32, m & 0xffff_ffff);
    let mut w = [0u64; 8];
    for (l, &e) in x.iter().enumerate() {
        w[l] = e as u64;
    }
    let mut v = [0i64; 8];
    for l in 0..8 {
        let s = (w[l] * mh + ((w[l] * ml) >> 32) + (1 << 31)) >> 32;
        let s = s as i64;
        v[l] = if s >= 1i64 << (bits - 1) { s - (1i64 << bits) } else { s };
    }
    let (b, half) = (1i64 << bl, (1i64 << bl) / 2);
    let mut d = [[0i8; 8]; 64];
    for dt in d.iter_mut().take(lv - 1) {
        if bl == 1 {
            for l in 0..8 {
                let r = (v[l] & 1) * (2 - (v[l] & 3));
                v[l] = (v[l] - r) >> 1;
                dt[l] = r as i8;
            }
        } else {
            for l in 0..8 {
                let r0 = v[l] & (b - 1);
                let r = if r0 >= half { r0 - b } else { r0 };
                v[l] = (v[l] - r) >> bl;
                dt[l] = r as i8;
            }
        }
    }
    for l in 0..8 {
        let r = if bl == 1 {
            (v[l] & 1) * (2 - (v[l] & 3))
        } else {
            let r0 = v[l] & (b - 1);
            if r0 >= half {
                r0 - b
            } else {
                r0
            }
        };
        let rest = (v[l] - r) >> bl;
        d[lv - 1][l] = (r + (rest << bl)) as i8;
    }
    for l in 0..x.len() {
        for t in 0..lv {
            out[l * lv + t] = d[t][l];
        }
    }
}
fn digit_exact(mut v: i64, t: usize, bl: u32, lv: usize) -> i64 {
    let b = 1i64 << bl;
    let mut r = 0;
    for _ in 0..=t {
        r = v & (b - 1);
        if r >= b / 2 {
            r -= b;
        }
        v = (v - r) >> bl;
    }
    if t == lv - 1 {
        r += v << bl;
    }
    r
}
#[inline(always)]
fn shoup32(a: u32, w: u32, ws: u32, q: u32) -> u32 {
    let r = a.wrapping_mul(w).wrapping_sub((((a as u64) * ws as u64) >> 32) as u32 * q);
    if r >= q {
        r - q
    } else {
        r
    }
}
/// x mod Q from (r_a, r_b) (tfhe.cuh `crt`).
#[inline(always)]
fn crt(ra: u32, rb: u32, qa: u32, qb: u32, inv: u32, invs: u32) -> u64 {
    let ram = if ra >= qb { ra - qb } else { ra };
    let d = if rb >= ram { rb - ram } else { rb.wrapping_add(qb).wrapping_sub(ram) };
    ra as u64 + qa as u64 * shoup32(d, inv, invs, qb) as u64
}
#[inline(always)]
fn cent(v: u64, p: u64) -> i64 {
    if v > p / 2 {
        v as i64 - p as i64
    } else {
        v as i64
    }
}
#[inline(always)]
fn smod64(v: i64, p: u64) -> u64 {
    let p = p as i64;
    if (0..p).contains(&v) {
        v as u64
    } else if v < 0 && v > -p {
        (v + p) as u64
    } else {
        let r = v % p;
        (if r < 0 { r + p } else { r }) as u64
    }
}
#[inline(always)]
fn mulm32(a: u32, b: u32, p: u64, mu: u64) -> u32 {
    mulm(a, b, p as u32, mu)
}
/// -q^-1 mod 2^32.
pub(crate) fn mont_neg_inv(q: u32) -> u32 {
    let mut v = q;
    for _ in 0..4 {
        v = v.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(v)));
    }
    0u32.wrapping_sub(v)
}
/// x R^-1 mod q in [0, 2 q) (x < q 2^32).
#[inline(always)]
fn redc(x: u64, q: u32, mi: u32) -> u32 {
    let m = (x as u32).wrapping_mul(mi);
    ((x + m as u64 * q as u64) >> 32) as u32
}
#[inline(always)]
fn fold(x: u32, b: u32) -> u32 {
    x.min(x.wrapping_sub(b))
}

fn check(a: &TfheArgs) -> i32 {
    use tfhe_op::*;
    if a.count < 1 {
        return -1;
    }
    let n = a.n;
    if (a.op <= PBS_EXTRACT || a.op == PBS_ROTATE) && (!(64..=2048).contains(&n) || n & (n - 1) != 0) {
        return -1;
    }
    match a.op {
        PBS_MS | KS_DIGITS | KS_RECOMBINE | RESCALE_P1 | RS_DIGITS | MAC | REPACK_DIGITS | REPACK_COMB | REPACK_FIN | REPACK_SCATTER | REPACK_RING | REPACK_RMAC | REPACK_C0 => 0,
        PBS_INIT | PBS_EXTRACT | RS_EXTRACT | LWE_LIN if a.count > 65535 => -1,
        PBS_INIT | PBS_EXTRACT | RS_EXTRACT | LWE_LIN => 0,
        PBS_ROTATE => match rot_path(a) {
            Some(_) => 0,
            None => -1,
        },
        PBS_KEYBUNDLE => {
            let grp = if a.i2 != 0 { a.i2 } else { 2 };
            let (lv, nu) = (a.c[1], a.c[5]);
            if n != 2048 || (lv != 1 && lv != 2) || !(2..=5).contains(&grp) || a.i1 % grp != 0 || a.count > 65535 {
                return -1;
            }
            if grp == 5 && (lv != 1 || nu > 1) {
                return -1;
            }
            let gw = if a.i4 != 0 { a.i4 } else { a.i1 / grp - a.i0 };
            if a.i0 < 0 || gw < 1 || a.i0 + gw > a.i1 / grp {
                return -1;
            }
            if (lv == 2 && nu > 1) || (lv == 1 && nu > 2) {
                return -1;
            }
            0
        }
        REPACK_MAC if (2 * a.i1) % 256 != 0 || (a.i1 as u64 * a.c[1]) % 256 != 0 => -1,
        REPACK_MAC => 0,
        _ => -2,
    }
}

/// PBS_ROTATE's launcher checks; the group width the selected kernel runs with (the two-block cluster kernel's
/// widest is 4).
fn rot_path(a: &TfheArgs) -> Option<usize> {
    let lv = a.c[1];
    let (qa, qb) = (a.c[6] & 0xffff_ffff, a.c[6] >> 32);
    if a.n != 2048 || (lv != 1 && lv != 2) || 46 * qa.max(qb) >= 1 << 32 {
        return None;
    }
    let grp = if a.i2 != 0 { a.i2 } else { 2 };
    let has_w = !a.w.is_null();
    if !(2..=5).contains(&grp) || a.i1 % grp != 0 || (grp == 5 && (!has_w || lv != 1 || a.c[5] > 1 || a.count > CL4_MAX || a.i3 == 1 || a.i3 == 2)) {
        return None;
    }
    let g = grp as usize;
    if a.c[5] > 1 {
        if a.c[5] != 2 || lv != 1 || (has_w && a.count > CL4_MAX) {
            return None;
        }
        return Some(g.min(4));
    }
    if has_w {
        if lv == 2 {
            return if a.count > CL4_MAX { None } else { Some(g.min(4)) };
        }
        if a.i3 == 4 || (a.i3 == 0 && a.count <= CL4_MAX) {
            return Some(g);
        }
        if a.i3 != 1 {
            return Some(g.min(4));
        }
        return Some(g.min(4));
    }
    Some(g.min(4))
}

pub extern "C" fn tfhe(a: *const TfheArgs, s: CudaStream) -> i32 {
    launch(a, s, check, run)
}

const OPS: [&str; 21] = ["tfhe.pbs_ms", "tfhe.pbs_init", "tfhe.pbs_step", "tfhe.pbs_extract", "tfhe.ks_digits", "tfhe.ks_recombine", "tfhe.rescale_p1", "tfhe.rs_digits", "tfhe.mac", "tfhe.rs_extract", "tfhe.repack_digits", "tfhe.repack_mac", "tfhe.repack_scatter", "tfhe.pbs_rotate", "tfhe.repack_ring", "tfhe.repack_rmac", "tfhe.repack_c0", "tfhe.repack_fin", "tfhe.pbs_keybundle", "tfhe.repack_comb", "tfhe.lwe_lin"];

fn run(a: &TfheArgs) {
    use tfhe_op::*;
    let t0 = crate::mm::op_timer();
    match a.op {
        PBS_MS => par_for(a.count as usize, a, |a, b| ms(a, b)),
        PBS_INIT => par_for(a.count as usize, a, |a, b| init(a, b)),
        PBS_ROTATE => rotate(a),
        PBS_KEYBUNDLE => keybundle(a),
        PBS_EXTRACT => par_for(a.count as usize, a, |a, b| extract(a, b)),
        KS_DIGITS => {
            let (len, lv) = (a.i1 as usize, a.c[1] as usize);
            par_for(a.count as usize, a, |a, b| {
                let src = rows::<u32>(a.y, b * (len + 1), len);
                let dst = rows_mut::<i8>(a.x, b * len * lv, len * lv);
                if crate::simd::tfhe_digits(src, a.c[0] as u32, lv, a.m, dst) {
                    return;
                }
                for (x, d) in src.chunks(8).zip(dst.chunks_mut(8 * lv)) {
                    digits8(x, a.c[0] as u32, lv, a.m, d);
                }
            });
        }
        KS_RECOMBINE => {
            // blocks of 64 LWEs: the planes' rows are read contiguously (b inner), the 64 output rows stay in L1
            const BB: usize = 64;
            par_for((a.count as usize).div_ceil(BB), a, |a, blk| {
                let cols = a.i2 as usize + 1;
                let planes = a.c[2] as usize;
                let nrows = if a.c[4] != 0 { a.c[4] as usize } else { a.count as usize };
                let plane = if a.c[3] != 0 { a.c[3] as usize } else { cols * nrows };
                let q = a.q as i64;
                let bs = blk * BB..(a.count as usize).min((blk + 1) * BB);
                for o in 0..cols {
                    for b in bs.clone() {
                        let mut v = 0i64;
                        for p in (0..planes).rev() {
                            v = v.wrapping_mul(256).wrapping_add(rd::<i32>(a.y, p * plane + o * nrows + b) as i64);
                        }
                        v %= q;
                        if v < 0 {
                            v += q;
                        }
                        let mut r = v as u64;
                        if o == a.i2 as usize {
                            r = addq(r, rd::<u32>(a.z, b * (a.i1 as usize + 1) + a.i1 as usize) as u64, a.q);
                        }
                        wr(a.x, b * cols + o, r as u32);
                    }
                }
            });
        }
        RESCALE_P1 => par_for(2, a, |a, part| {
            let n = a.n as usize;
            let (q, p1) = (a.q, a.c[1]);
            let y = rows::<u32>(a.y, part * 2 * n, 2 * n);
            let x = rows_mut::<u32>(a.x, part * n, n);
            for k in 0..n {
                let d = smod64(cent(y[n + k] as u64, p1), q);
                x[k] = mulm32(subq(y[k] as u64, d, q) as u32, a.c[0] as u32, q, a.c[7]);
            }
        }),
        RS_DIGITS => {
            let n = a.n as usize;
            par_for(n.div_ceil(4096), a, |a, c| {
                let lv = a.c[1] as usize;
                let mut d = [0i64; 64];
                for k in c * 4096..((c + 1) * 4096).min(a.n as usize) {
                    digits_of(rd::<u32>(a.y, k) as u64, a.c[0] as u32, lv, a.m, &mut d);
                    for (t, &v) in d.iter().enumerate().take(lv) {
                        wr(a.x, t * a.n as usize + k, smod64(v, a.q) as u32);
                    }
                }
            });
        }
        MAC => {
            let n = a.n as usize;
            par_for(a.i1 as usize, a, |a, o| {
                let x = rows_mut::<u32>(a.x, o * n, n);
                x.fill(0);
                for t in 0..a.count as usize {
                    let y = rows::<u32>(a.y, t * n, n);
                    let z = rows::<u32>(a.z, (t * a.i1 as usize + o) * n, n);
                    for k in 0..n {
                        x[k] = addq(x[k] as u64, mulm32(y[k], z[k], a.q, a.c[7]) as u64, a.q) as u32;
                    }
                }
            });
        }
        RS_EXTRACT => par_for(a.count as usize, a, |a, b| {
            let (n, dim, s) = (a.n as usize, a.i1 as usize, a.i2 as usize);
            let j = rd::<i32>(a.z, b) as i64;
            let c0 = rows::<u32>(a.y, 0, n);
            let c1 = rows::<u32>(a.y, n, n);
            let out = rows_mut::<u32>(a.x, b * (dim + 1), dim + 1);
            for (kk, o) in out.iter_mut().enumerate() {
                if kk == dim {
                    *o = c0[j as usize];
                    continue;
                }
                let sk = (s * kk) as i64;
                *o = if sk <= j {
                    c1[(j - sk) as usize]
                } else {
                    let v = c1[(n as i64 + j - sk) as usize];
                    if v != 0 {
                        (a.q - v as u64) as u32
                    } else {
                        0
                    }
                };
            }
        }),
        REPACK_DIGITS => repack_digits(a),
        REPACK_MAC => repack_mac(a),
        REPACK_COMB => {
            let cols = 4 * a.i1 as usize;
            par_for(a.count as usize, a, |a, j| {
                let mp = a.i3 as usize;
                let x = rows_mut::<i64>(a.x, j * cols, cols);
                for (col, o) in x.iter_mut().enumerate() {
                    *o = o.wrapping_add((rd::<i32>(a.y, col * mp + j) as i64).wrapping_shl(8 * a.c[5] as u32));
                }
            });
        }
        REPACK_FIN => {
            let nb = a.i1 as usize;
            par_for(a.count as usize, a, |a, j| {
                let (p0, pp) = (a.q, a.c[2]);
                for col in 0..2 * nb {
                    let s = rows::<i64>(a.y, j * 4 * nb + col, 2 * nb + 1);
                    let mut x0 = smod64(s[0], p0);
                    if col == 0 {
                        let b = smod64(cent(rd::<u32>(a.w, j * (nb + 1) + nb) as u64, p0), p0);
                        x0 = addq(x0, mulm32(b as u32, a.c[4] as u32, p0, a.c[7]) as u64, p0);
                    }
                    let x1 = smod64(cent(smod64(s[2 * nb], pp), pp), p0);
                    wr(a.x, j * 2 * nb + col, mulm32(subq(x0, x1, p0) as u32, a.c[3] as u32, p0, a.c[7]));
                }
            });
        }
        REPACK_SCATTER => {
            // X^j PK_j(X^S) summed into z: exact u64 sums, so any order (the device's atomics)
            let (n, nb, s) = (a.n as usize, a.i1 as usize, a.i2 as usize);
            par_for(2, a, |a, part| {
                let z = rows_mut::<u64>(a.z as *mut std::ffi::c_void, part * n, n);
                for b in 0..a.count as usize {
                    let j = rd::<i32>(a.w, b) as usize;
                    let pk = rows::<u32>(a.y, (b * 2 + part) * nb, nb);
                    for (i, &v) in pk.iter().enumerate() {
                        let mut y = j + s * i;
                        let mut v = v as u64;
                        if y >= n {
                            y -= n;
                            v = if v != 0 { a.q - v } else { 0 };
                        }
                        z[y] = z[y].wrapping_add(v);
                    }
                }
            });
        }
        REPACK_RING => {
            let n = a.n as usize;
            par_for(n.div_ceil(4096), a, |a, c| {
                let (n, lv, bl) = (a.n as usize, a.c[1] as usize, a.c[0] as u32);
                for y in c * 4096..((c + 1) * 4096).min(n) {
                    wr(a.w as *mut std::ffi::c_void, y, (rd::<u64>(a.y, y) % a.q) as u32);
                    let v = cent(rd::<u64>(a.y, n + y) % a.q, a.q);
                    for t in 0..lv {
                        let d = digit_exact(v, t, bl, lv);
                        wr(a.x, t * n + y, smod64(d, a.q) as u32);
                        wr(a.x, (lv + t) * n + y, smod64(d, a.c[4]) as u32);
                    }
                }
            });
        }
        REPACK_RMAC => par_for(2, a, |a, limb| {
            let (n, lv) = (a.n as usize, a.c[1] as usize);
            let (p, mu) = if limb != 0 { (a.c[4], a.c[6]) } else { (a.q, a.c[7]) };
            let d = rows::<u32>(a.y, limb * lv * n, lv * n);
            let acc0 = rows_mut::<u32>(a.x, limb * n, n);
            let acc1 = rows_mut::<u32>(a.x, (2 + limb) * n, n);
            for t in 0..lv {
                let k0 = rows::<u32>(a.z, ((t * 2 + limb) * 2) * n, 2 * n);
                for j in 0..n {
                    let dv = d[t * n + j];
                    acc0[j] = addq(acc0[j] as u64, mulm32(dv, k0[j], p, mu) as u64, p) as u32;
                    acc1[j] = addq(acc1[j] as u64, mulm32(dv, k0[n + j], p, mu) as u64, p) as u32;
                }
            }
        }),
        REPACK_C0 => {
            let n = a.n as usize;
            par_for(2, a, |a, limb| {
                let (p, mu) = if limb != 0 { (a.c[4], a.c[6]) } else { (a.q, a.c[7]) };
                let p1 = if limb != 0 { 0 } else { a.c[2] as u32 };
                let c0 = rows_mut::<u32>(a.x, limb * n, n);
                for (j, o) in c0.iter_mut().enumerate() {
                    let v = cent(rd::<u32>(a.y, j) as u64, a.q);
                    *o = addq(*o as u64, mulm32(smod64(v, p) as u32, p1, p, mu) as u64, p) as u32;
                }
            });
        }
        LWE_LIN => par_for(a.count as usize, a, |a, r| {
            let len = a.i1 as usize + 1;
            let (ia, ib) = (rd::<i32>(a.w, 2 * r), rd::<i32>(a.w, 2 * r + 1));
            let q = a.q;
            let out = rows_mut::<u32>(a.x, r * len, len);
            for (k, o) in out.iter_mut().enumerate() {
                let u = if ia >= 0 { rd::<u32>(a.y, ia as usize * len + k) as u64 * a.c[0] % q } else { 0 };
                let v = if ib >= 0 { rd::<u32>(a.z, ib as usize * len + k) as u64 * a.c[1] % q } else { 0 };
                let s = addq(u, v, q);
                *o = (if k == len - 1 { addq(s, a.c[2], q) } else { s }) as u32;
            }
        }),
        _ => {}
    }
    crate::mm::op_prof(OPS.get(a.op as usize).copied().unwrap_or("?"), t0);
}

/// The mod switch to 2 nu n / k of one LWE, the body mean-compensated (ms_k).
fn ms(a: &TfheArgs, b: usize) {
    let n = a.i1 as usize;
    let y = rows::<u32>(a.y, b * (n + 1), n + 1);
    let x = rows_mut::<u32>(a.x, b * (n + 1), n + 1);
    let kl = if a.c[4] != 0 { a.c[4] } else { 1 };
    let two_n = 2 * a.n as u64 * if a.c[5] != 0 { a.c[5] } else { 1 } / kl;
    let m = a.m;
    let mut s = 0i64;
    for i in 0..n {
        let v = y[i] as u64;
        let lo = v.wrapping_mul(m);
        let mut hi = ((v as u128 * m as u128) >> 64) as u64;
        if lo >= 1 << 63 {
            hi += 1;
            s += (0u64.wrapping_sub(lo) >> 32) as i64;
        } else {
            s -= ((lo + 0xffff_ffff) >> 32) as i64;
        }
        x[i] = kl.wrapping_mul(hi % two_n) as u32;
    }
    let v = (y[n] as u128 * m as u128) as i128 - ((s as i128) << 31) + (1i128 << 63);
    let r = ((v >> 64) as i64) % two_n as i64;
    x[n] = kl.wrapping_mul((if r < 0 { r + two_n as i64 } else { r }) as u64) as u32;
}

fn rns(a: &TfheArgs) -> (u32, u32, u32, u32) {
    (a.c[6] as u32, (a.c[6] >> 32) as u32, a.c[7] as u32, (a.c[7] >> 32) as u32)
}
fn nu_of(a: &TfheArgs) -> usize {
    if a.c[5] != 0 {
        a.c[5] as usize
    } else {
        1
    }
}

/// acc = X^-b~ tv in R_(nu n) (component i: coefficients nu j + i), part 1 = 0 (init_k).
fn init(a: &TfheArgs, b: usize) {
    let (n, nu) = (a.n as usize, nu_of(a));
    let big = nu * n;
    let (qa, qb, _, _) = rns(a);
    let bt = rd::<u32>(a.z, b * (a.i1 as usize + 1) + a.i1 as usize);
    let kp = ((2 * big as u32).wrapping_sub(bt) % (2 * big as u32)) as i64;
    let acc = rows_mut::<u32>(a.x, b * nu * 4 * n, nu * 4 * n);
    for x in 0..big {
        let (i, o) = (x / n, x % n);
        for p in 0..2 {
            let q = if p != 0 { qb } else { qa };
            let tv = rows::<u32>(a.y, p * big, big);
            let oo = (nu * o + i) as i64;
            let mut s = oo - kp;
            acc[(i * 4 + p) * n + o] = if s >= 0 {
                tv[s as usize]
            } else {
                s += big as i64;
                if s >= 0 {
                    let v = tv[s as usize];
                    if v != 0 {
                        q - v
                    } else {
                        0
                    }
                } else {
                    tv[(s + big as i64) as usize]
                }
            };
            acc[(i * 4 + 2 + p) * n + o] = 0;
        }
    }
}

/// Sample extraction of coefficient c[4] of component 0, + c[0] mod Q, then round(v c[1] / Q) mod c[1] (extract_k).
fn extract(a: &TfheArgs, b: usize) {
    let n = a.n as usize;
    let (qa, qb, inv, invs) = rns(a);
    let (big_q, to) = (a.q, a.c[1]);
    let acc = rows::<u32>(a.y, b * nu_of(a) * 4 * n, 4 * n);
    let out = rows_mut::<u32>(a.x, b * (n + 1), n + 1);
    let j = a.c[4] as usize;
    for (k, o) in out.iter_mut().enumerate() {
        let v = if k == n {
            addq(crt(acc[j], acc[n + j], qa, qb, inv, invs), a.c[0], big_q)
        } else if k <= j {
            crt(acc[2 * n + j - k], acc[3 * n + j - k], qa, qb, inv, invs)
        } else {
            let c1 = crt(acc[3 * n + j - k], acc[4 * n + j - k], qa, qb, inv, invs);
            if c1 != 0 {
                big_q - c1
            } else {
                0
            }
        };
        *o = (scale_round(v, a.m) % to) as u32;
    }
}

fn repack_digits(a: &TfheArgs) {
    let (nb, lv, bl) = (a.i1 as usize, a.c[1] as usize, a.c[0] as u32);
    let (count, k_all, mp) = (a.count as usize, nb * lv, a.c[5] as usize);
    if a.i0 != 0 {
        rows_mut::<u64>(a.z as *mut std::ffi::c_void, 0, 2 * a.n as usize).fill(0);
    }
    rows_mut::<u64>(a.w as *mut std::ffi::c_void, 0, count * nb * 4).fill(0);
    par_for(count, a, |a, b| {
        for k in 0..nb {
            let v = cent(rd::<u32>(a.y, b * (nb + 1) + k) as u64, a.q);
            for t in 0..lv {
                let d = digit_exact(v, t, bl, lv);
                let lo = ((d + 128) % 256 + 256) % 256 - 128;
                wr(a.x, b * k_all + k * lv + t, lo as i8);
                wr(a.x, mp * k_all + b * k_all + k * lv + t, ((d - lo) / 256) as i8);
            }
        }
    });
}

/// x [count][4 n_br] (i64) += D [count][K] (i32) times the columns [K][4 n_br] (u32): exact wrapping sums.
fn repack_mac(a: &TfheArgs) {
    let nb = a.i1 as usize;
    let kk = nb * a.c[1] as usize;
    par_for(a.count as usize, a, |a, j| {
        let d = rows::<i32>(a.y, j * kk, kk);
        let mut acc = buf::<i64>(4 * nb);
        for (k, &dv) in d.iter().enumerate() {
            if dv == 0 {
                continue;
            }
            let key = rows::<u32>(a.z, k * 4 * nb, 4 * nb);
            for c in 0..2 * nb {
                acc[c] = acc[c].wrapping_add(dv as i64 * key[c] as i64);
                acc[2 * nb + c] = acc[2 * nb + c].wrapping_add(dv as i64 * key[2 * nb + c] as i64);
            }
        }
        let x = rows_mut::<i64>(a.x, j * 4 * nb, 4 * nb);
        for (o, &v) in x[..2 * nb].iter_mut().zip(&acc[..2 * nb]) {
            *o = o.wrapping_add(v);
        }
        for (o, &v) in x[2 * nb..].iter_mut().zip(&acc[2 * nb..]) {
            *o = o.wrapping_add(v);
        }
    });
}

// ── the blind rotation ──────────────────────────────────────────────────────────────────────────────────────────

/// One launch's rotation constants: ring n = 2048, per prime q, -q^-1 mod 2^32 and 2^32 mod q; the tables of
/// NTT(X^e - 1) (pm1 [prime][2 n], Montgomery; E [prime][n]); y = NTT(X) at each word (nu = 2, Montgomery).
pub struct Rot<'a> {
    pub n: usize,
    pub nu: usize,
    pub lv: usize,
    pub km: usize,
    pub q: [u32; 2],
    pub mi: [u32; 2],
    pub r1: [u32; 2],
    pub pm1: &'a [u32],
    pub ex: &'a [u32],
    pub yk: Vec<u32>,
}

impl<'a> Rot<'a> {
    fn new(a: &TfheArgs, km: usize) -> Rot<'a> {
        let n = a.n as usize;
        let (qa, qb, _, _) = rns(a);
        let q = [qa, qb];
        let pm1 = rows::<u32>(a.tw, 4 * n, 4 * n);
        let ex = rows::<u32>(a.tw, 8 * n, 2 * n);
        let r1 = q.map(|q| ((1u64 << 32) % q as u64) as u32);
        let nu = nu_of(a);
        let yk = if nu == 2 { (0..2 * n).map(|k| crate::mm::addm(pm1[(k / n) * 2 * n + (ex[k] as usize & (2 * n - 1))], r1[k / n], q[k / n])).collect() } else { vec![] };
        Rot { n, nu, lv: a.c[1] as usize, km, q, mi: q.map(mont_neg_inv), r1, pm1, ex, yk }
    }
    /// Bundle words per word: max(nu, lv) uint4.
    pub fn nk(&self) -> usize {
        self.nu.max(self.lv)
    }
}

/// Where a step's key bundle comes from: the group's key words [m][e][h][thread][4], or PBS_KEYBUNDLE's [e][nk][thread][4].
#[derive(Clone, Copy)]
pub enum Src<'a> {
    Key(&'a [u32]),
    Bundle(&'a [u32]),
}

/// The bundle words of word k (prime k / n): [slot][x, y, z, w] = G[row 0][part 0], G[0][1], G[1][0], G[1][1]
/// (slot: the level pair h, or A / B at nu = 2), Montgomery form, canonical.
pub fn bundle_s(r: &Rot, key: &[u32], em: &[u32], k: usize) -> [[u32; 4]; 2] {
    let (n, p) = (r.n, k / r.n);
    let (q, mi) = (r.q[p], r.mi[p]);
    let (e, th) = (k / TH, k % TH);
    let ej = r.ex[k] as usize;
    let f = |x: usize| r.pm1[p * 2 * n + ((ej * x) & (2 * n - 1))] as u64;
    let mut g = [[0u64; 4]; 2];
    for m in 0..r.km {
        let x = em[m] as usize;
        if x == 0 {
            continue;
        }
        if r.nu == 1 {
            for h in 0..r.lv {
                let w = &key[(((m * RT_E + e) * r.lv + h) * TH + th) * 4..][..4];
                for c in 0..4 {
                    g[h][c] += w[c] as u64 * f(x);
                }
            }
        } else {
            let w = &key[((m * RT_E + e) * TH + th) * 4..][..4];
            let pj = f(x >> 1) as u32;
            let (fa, fb) = if x & 1 != 0 { ((q - r.r1[p]) as u64, crate::mm::addm(pj, r.r1[p], q) as u64) } else { (pj as u64, 0) };
            for c in 0..4 {
                g[0][c] += w[c] as u64 * fa;
                g[1][c] += w[c] as u64 * fb;
            }
        }
    }
    g.map(|s| s.map(|v| fold(redc(v, q, mi), q)))
}

/// The step's product at word k: o [comp][part][prime][n] from the NTT'd digits dig [comp][row][prime][n] (and d1y
/// [ct][prime][n] = y d_1 at nu = 2).
pub fn word_s(r: &Rot, g: &[[u32; 4]; 2], k: usize, dig: &[u32], d1y: &[u32], o: &mut [u32]) {
    let (n, p, j) = (r.n, k / r.n, k % r.n);
    let (q, mi) = (r.q[p], r.mi[p]);
    let rows = 2 * r.lv;
    let d = |comp: usize, row: usize| dig[((comp * rows + row) * 2 + p) * n + j] as u64;
    let red = |s: u64| fold(redc(s, q, mi), q);
    for pt in 0..2 {
        if r.nu == 1 {
            let mut s = 0u64;
            for h in 0..r.lv {
                s += d(0, 2 * h) * g[h][pt] as u64 + d(0, 2 * h + 1) * g[h][2 + pt] as u64;
            }
            o[(pt * 2 + p) * n + j] = red(s);
        } else {
            let (ga, gb) = (&g[0], &g[1]);
            let y = |ct: usize| d1y[(ct * 2 + p) * n + j] as u64;
            let s0 = d(0, 0) * ga[pt] as u64 + d(0, 1) * ga[2 + pt] as u64 + y(0) * gb[pt] as u64 + y(1) * gb[2 + pt] as u64;
            let s1 = d(1, 0) * ga[pt] as u64 + d(1, 1) * ga[2 + pt] as u64 + d(0, 0) * gb[pt] as u64 + d(0, 1) * gb[2 + pt] as u64;
            o[(pt * 2 + p) * n + j] = red(s0);
            o[((4 + pt * 2) + p) * n + j] = red(s1);
        }
    }
}

/// Scratch of one LWE step: digits [comp][row][prime][n], y d_1 [ct][prime][n], o [comp][part][prime][n].
struct Scratch {
    dig: crate::raw::Buf<u32>,
    d1y: crate::raw::Buf<u32>,
    o: crate::raw::Buf<u32>,
}

/// acc += the external product of one group (the launch's tables tw / tws: [prime][fwd | inv][n]).
/// The accumulator's gadget digits dig [comp][row][prime][n] (row = part levels + level), canonical residues.
fn digits(a: &TfheArgs, r: &Rot, acc: &[u32], dig: &mut [u32]) {
    let (n, lv) = (r.n, r.lv);
    let (qa, qb, inv, invs) = rns(a);
    let g = crate::simd::RnsGadget { qa, qb, inv, invs, m: a.m, bl: a.c[0] as u32, lv };
    for i in 0..r.nu {
        for c in 0..2 {
            let (ra, rb) = (&acc[((i * 2 + c) * 2) * n..][..n], &acc[((i * 2 + c) * 2 + 1) * n..][..n]);
            crate::simd::rns_digits(ra, rb, &g, &mut dig[((i * 2 * lv + c * lv) * 2) * n..][..2 * lv * n]);
        }
    }
}

fn step(a: &TfheArgs, r: &Rot, src: Src, em: &[u32], acc: &mut [u32], sc: &mut Scratch) {
    let (n, nu) = (r.n, r.nu);
    let nrow = 2 * r.lv;
    digits(a, r, acc, &mut sc.dig);
    let tw = |p: usize, inv: bool| -> (&[u32], &[u32]) {
        let off = p * 2 * n + if inv { n } else { 0 };
        (rows::<u32>(a.tw, off, n), rows::<u32>(a.tws, off, n))
    };
    for (k, row) in sc.dig[..nu * nrow * 2 * n].chunks_mut(n).enumerate() {
        let (w, ws) = tw(k & 1, false);
        ntt_g::fwd(row, w, ws, r.q[k & 1]);
    }
    if nu == 2 {
        for k in 0..2 * n {
            let (p, j) = (k / n, k % n);
            for ct in 0..2 {
                let d = sc.dig[((nrow + ct) * 2 + p) * n + j] as u64;
                sc.d1y[(ct * 2 + p) * n + j] = fold(redc(d * r.yk[k] as u64, r.q[p], r.mi[p]), r.q[p]);
            }
        }
    }
    crate::pbs_v::words(r, src, em, &sc.dig, &sc.d1y, &mut sc.o);
    for (k, row) in sc.o[..nu * 4 * n].chunks_mut(n).enumerate() {
        let (w, ws) = tw(k & 1, true);
        ntt_g::inv(row, w, ws, r.q[k & 1]);
        crate::simd::add_rows(&mut acc[k * n..(k + 1) * n], row, r.q[k & 1]);
    }
}

/// The exponents of group pr for LWE at (the mod-switched words), mod 2 nu n: em[m - 1] for the bit masks m.
fn exponents(at: &[u32], g: usize, pr: usize, n2e: u32, em: &mut [u32]) -> bool {
    let mut nz = false;
    for (mi, e) in em.iter_mut().enumerate().take((1 << g) - 1) {
        let m = mi + 1;
        let mut s = 0u32;
        for x in 0..g {
            if m >> x & 1 != 0 {
                s = s.wrapping_add(at[g * pr + x]);
            }
            nz |= at[g * pr + x] != 0;
        }
        *e = (g as u32 * n2e).wrapping_sub(s) % n2e;
    }
    nz
}

/// LWEs per task: the batch shares each group's key words in cache; at least one batch per pool thread.
fn batch(count: usize) -> usize {
    count.div_ceil(pc_hostrt::threads().max(1)).clamp(1, 8)
}

fn rotate(a: &TfheArgs) {
    let g = rot_path(a).expect("checked");
    let km = (1 << g) - 1;
    let r = Rot::new(a, km);
    let (n, nu, nk) = (r.n, r.nu, r.nk());
    let l = a.i1 as usize;
    let g0 = a.i0 as usize;
    let gw = if a.i4 != 0 { a.i4 as usize } else { l / g - g0 };
    let kstride = 8 * r.lv * n;
    let count = a.count as usize;
    let bs = batch(count);
    let n2e = (2 * nu * n) as u32;
    let rr = &r;
    par_for(count.div_ceil(bs), a, |a, t| {
        let lws = t * bs..((t + 1) * bs).min(count);
        let mut sc = Scratch { dig: buf(nu * 4 * r.lv * n), d1y: buf(4 * n), o: buf(nu * 4 * n) };
        let mut em = [0u32; 31];
        for pr in g0..g0 + gw {
            for lw in lws.clone() {
                let at = rows::<u32>(a.z, lw * (l + 1), l + 1);
                if !exponents(at, g, pr, n2e, &mut em) {
                    continue;
                }
                let src = if a.w.is_null() {
                    Src::Key(rows::<u32>(a.y, pr * km * kstride, km * kstride))
                } else {
                    Src::Bundle(rows::<u32>(a.w, ((lw * gw + pr - g0) * RT_E * nk * TH) * 4, RT_E * nk * TH * 4))
                };
                let acc = rows_mut::<u32>(a.x, lw * nu * 4 * n, nu * 4 * n);
                step(a, rr, src, &em[..km], acc, &mut sc);
            }
        }
    });
}

fn keybundle(a: &TfheArgs) {
    let g = if a.i2 != 0 { a.i2 as usize } else { 2 };
    let km = (1 << g) - 1;
    let r = Rot::new(a, km);
    let (n, nk) = (r.n, r.nk());
    let l = a.i1 as usize;
    let g0 = a.i0 as usize;
    let gw = if a.i4 != 0 { a.i4 as usize } else { l / g - g0 };
    let kstride = 8 * r.lv * n;
    let count = a.count as usize;
    let bs = batch(count * gw);
    let n2e = (2 * r.nu * n) as u32;
    let rr = &r;
    // tasks: (group, batch of LWEs), the group's key words shared by the batch
    let nb = count.div_ceil(bs);
    par_for(gw * nb, a, |a, t| {
        let (gi, bi) = (t / nb, t % nb);
        let pr = g0 + gi;
        let key = rows::<u32>(a.y, pr * km * kstride, km * kstride);
        let mut em = [0u32; 31];
        for lw in bi * bs..((bi + 1) * bs).min(count) {
            let at = rows::<u32>(a.z, lw * (l + 1), l + 1);
            exponents(at, g, pr, n2e, &mut em);
            let out = rows_mut::<u32>(a.x, ((lw * gw + gi) * RT_E * nk * TH) * 4, RT_E * nk * TH * 4);
            crate::pbs_v::bundles(rr, key, &em[..km], out);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The device's digit t (the sequence rerun up to t).
    fn digit(x: u64, t: usize, bl: u32, lv: usize, m: u64) -> i64 {
        let bits = bl as usize * lv;
        let mut v = scale_round(x, m) as i64;
        if v >= 1i64 << (bits - 1) {
            v -= 1i64 << bits;
        }
        let b = 1i64 << bl;
        let mut r = 0;
        for _ in 0..=t {
            if bl == 1 {
                r = if v & 1 != 0 { 2 - (v & 3) } else { 0 };
            } else {
                r = v & (b - 1);
                if r >= b / 2 {
                    r -= b;
                }
            }
            v = (v - r) >> bl;
        }
        if t == lv - 1 {
            r += v << bl;
        }
        r
    }

    #[test]
    fn one_pass_digits_are_the_devices() {
        let q = 268042241u64;
        for (bl, lv) in [(1u32, 16usize), (4, 6), (23, 1), (12, 2)] {
            let m = (((1u128 << (bl as usize * lv)) << 64) / q as u128) as u64;
            let mut d = [0i64; 64];
            for x in [0u64, 1, q / 2, q / 2 + 1, q - 1, 123456789, 99999] {
                digits_of(x, bl, lv, m, &mut d);
                for t in 0..lv {
                    assert_eq!(d[t], digit(x, t, bl, lv, m), "bl {bl} lv {lv} x {x} t {t}");
                }
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    #[test]
    #[ignore]
    fn step_parts() {
        let n = 2048;
        let (qa, qb) = (67104769u32, 67084289u32);
        let tw: Vec<u32> = (0..10 * n).map(|i| (i as u32 * 2654435761) % qa.min(qb)).collect();
        let tws: Vec<u32> = (0..10 * n).map(|i| ntt_g::companion(tw[i], if (i / (2 * n)) & 1 == 0 { qa } else { qb })).collect();
        let tw2: Vec<u32> = tw.iter().enumerate().map(|(i, &v)| if i >= 8 * n { v % (2 * n as u32) } else { v }).collect();
        let a = TfheArgs { n: n as i32, count: 1, i1: 795, tw: tw2.as_ptr() as *const _, tws: tws.as_ptr() as *const _, m: ((1u128 << 87) / (qa as u128 * qb as u128)) as u64, c: [23, 1, 0, 0, 0, 0, qa as u64 | (qb as u64) << 32, 5], ..Default::default() };
        let km = 7;
        let r = Rot::new(&a, km);
        let key: Vec<u32> = (0..km * 8 * n).map(|i| (i as u32).wrapping_mul(0x9E3779B9) % qb).collect();
        let em: Vec<u32> = (0..km as u32).map(|m| m * 333 + 1).collect();
        let mut acc: Vec<u32> = (0..4 * n).map(|i| (i as u32 * 7) % qb).collect();
        let mut sc = Scratch { dig: buf(4 * n), d1y: buf(4 * n), o: buf(4 * n) };
        let reps = 2000;
        for _ in 0..reps {
            step(&a, &r, Src::Key(&key), &em, &mut acc, &mut sc);
        }
        let t = std::time::Instant::now();
        for _ in 0..reps {
            step(&a, &r, Src::Key(&key), &em, &mut acc, &mut sc);
        }
        let st = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            crate::pbs_v::words(&r, Src::Key(&key), &em, &sc.dig, &sc.d1y, &mut sc.o);
        }
        let wd = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            for row in sc.dig.chunks_mut(n) {
                ntt_g::fwd(row, &tw[..n], &tws[..n], qa);
            }
        }
        let nt = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            digits(&a, &r, &acc, &mut sc.dig);
        }
        let dg = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            for row in sc.o.chunks_mut(n) {
                ntt_g::inv(row, &tw[n..2 * n], &tws[n..2 * n], qa);
            }
        }
        let ni = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
        eprintln!("4 inverse NTTs {ni:.1} us");
        eprintln!("step {st:.1} us: words {wd:.1} us, 4 forward NTTs {nt:.1} us, digits {dg:.1} us");
    }
}

#[cfg(test)]
mod digits_tests {
    #[test]
    #[ignore]
    fn digits8_rate() {
        let (len, lv) = (1usize << 20, 16usize);
        let x: Vec<u32> = (0..len as u32).map(|i| i.wrapping_mul(2654435761)).collect();
        let mut out = vec![0i8; len * lv];
        for bl in [1u32, 2] {
            let lv = if bl == 1 { lv } else { 8 };
            let t = std::time::Instant::now();
            for (x, d) in x.chunks(8).zip(out.chunks_mut(8 * lv)) {
                super::digits8(x, bl, lv, 1 << 32, d);
            }
            eprintln!("bl {bl} lv {lv}: {:.2} ns/word", t.elapsed().as_secs_f64() * 1e9 / len as f64);
            let t = std::time::Instant::now();
            assert!(crate::simd::tfhe_digits(&x, bl, lv, 1 << 32, &mut out));
            eprintln!("vector bl {bl} lv {lv}: {:.2} ns/word", t.elapsed().as_secs_f64() * 1e9 / len as f64);
        }
        std::hint::black_box(&out);
    }

    #[test]
    fn digits8_matches_digits_of() {
        let mut rng = 0x9e3779b97f4a7c15u64;
        for (bl, lv) in [(1u32, 16usize), (2, 8), (3, 5), (4, 4), (8, 2), (1, 1), (5, 1)] {
            for m in [1u64 << 32, 0x1234_5678_9abc_def1, (1u64 << 63) + 12345, u64::MAX] {
                let x: Vec<u32> = (0..67)
                    .map(|_| {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        rng as u32
                    })
                    .collect();
                let mut out = vec![0i8; 67 * lv];
                for (x, d) in x.chunks(8).zip(out.chunks_mut(8 * lv)) {
                    super::digits8(x, bl, lv, m, d);
                }
                let mut vec_out = vec![0i8; 67 * lv];
                if crate::simd::tfhe_digits(&x, bl, lv, m, &mut vec_out) {
                    assert_eq!(vec_out, out, "vector digits bl {bl} lv {lv} m {m:#x}");
                }
                let mut want = [0i64; 64];
                for (i, &w) in x.iter().enumerate() {
                    super::digits_of(w as u64, bl, lv, m, &mut want);
                    for t in 0..lv {
                        assert_eq!(out[i * lv + t], want[t] as i8, "bl {bl} lv {lv} m {m:#x} x {w:#x} t {t}");
                    }
                }
            }
        }
    }
}
