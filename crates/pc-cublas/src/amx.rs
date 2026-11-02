// The col-TN host GEMM on AMX-INT8 tiles: TDPBSSD sums signed int8 products into int32 lanes without saturation,
// so the result is the same exact (wrapping) integer as the device's MMA. Both operands are K-contiguous rows (A's
// rows i, B's columns j). The larger one is the tiles' row operand, loaded in place (16 rows x 64 bytes, stride
// ld); the smaller one is packed into zero-padded VNNI tiles ([k/4][row][k%4]), so bytes past K in the row operand
// meet zeros. A task accumulates a block of C tiles in a thread-local buffer over K blocks that keep both operand
// blocks in L2, then writes C once (transposed when the rows of the tiles are A's).
use crate::GemmSpec;
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::OnceLock;

const TK: usize = 64; // K bytes per tile
const TILE: usize = 1024;

const RB: usize = 256; // row-operand rows per task
const VB: usize = 256; // VNNI-operand rows per task
const KC: usize = 4096 / TK; // K tiles per block: both operand blocks (2 MB) and the C block (256 KB) in L2

/// Whether this CPU runs AMX-INT8 and the kernel granted this process the tile state.
pub(crate) fn available() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        // CPUID.(7,0):EDX bit 24 AMX-TILE, bit 25 AMX-INT8
        let edx = std::arch::x86_64::__cpuid_count(7, 0).edx;
        if std::env::var_os("PC_HOST_NO_AMX").is_some() || edx & (3 << 24) != 3 << 24 {
            return false;
        }
        const ARCH_REQ_XCOMP_PERM: i64 = 0x1023;
        const XFEATURE_XTILEDATA: i64 = 18;
        // SAFETY: arch_prctl permission request, no memory passed
        unsafe { libc::syscall(libc::SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0 }
    })
}

/// Cache-line aligned scratch (tile loads off a 64-byte boundary run at a fraction of the speed).
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([u8; 64]);

/// A thread's growable scratch of at least `bytes`, 64-byte aligned.
fn scratch(v: &mut Vec<Line>, bytes: usize) -> *mut u8 {
    if v.len() * 64 < bytes {
        *v = vec![Line([0; 64]); bytes.div_ceil(64)];
    }
    v.as_mut_ptr() as *mut u8
}

/// dst[0..n] (+)= src[0..n].
#[inline(always)]
unsafe fn put(dst: *mut i32, src: *const i32, n: usize, acc: bool) {
    let (d, s) = (std::slice::from_raw_parts_mut(dst, n), std::slice::from_raw_parts(src, n));
    if acc {
        d.iter_mut().zip(s).for_each(|(d, s)| *d = d.wrapping_add(*s));
    } else {
        d.copy_from_slice(s);
    }
}

/// In-place transpose of a 16 x 16 i32 tile.
#[inline(always)]
unsafe fn transpose16(t: *mut i32) {
    for x in 0..16 {
        for l in x + 1..16 {
            std::ptr::swap(t.add(x * 16 + l), t.add(l * 16 + x));
        }
    }
}

thread_local! {
    static PACK: RefCell<Vec<Line>> = const { RefCell::new(Vec::new()) };
}

/// Pack 16 K-contiguous rows (row r at `src + r * ld`, rows >= `rows` and bytes >= `k` read as zero) into
/// `kt` row-operand tiles: tile t holds bytes [64t, 64t + 64) of every row.
unsafe fn pack_rows(src: *const i8, ld: usize, rows: usize, k: usize, out: &mut [u8]) {
    out.fill(0);
    for r in 0..rows {
        let s = std::slice::from_raw_parts(src.add(r * ld) as *const u8, k);
        for (t, ch) in s.chunks(TK).enumerate() {
            out[t * TILE + r * TK..t * TILE + r * TK + ch.len()].copy_from_slice(ch);
        }
    }
}

/// Pack 16 K-contiguous rows into VNNI tiles: tile t, row q, lane r holds bytes [64t + 4q, 64t + 4q + 4) of row r.
unsafe fn pack_vnni(src: *const i8, ld: usize, rows: usize, k: usize, out: &mut [u8]) {
    let kt = out.len() / TILE;
    let full = k / 4;
    for r in 0..16 {
        if r >= rows {
            for t in 0..kt {
                for q in 0..16 {
                    out[t * TILE + q * 64 + r * 4..t * TILE + q * 64 + r * 4 + 4].fill(0);
                }
            }
            continue;
        }
        let s = src.add(r * ld) as *const u8;
        for d in 0..kt * 16 {
            let w = if d < full {
                (s.add(4 * d) as *const u32).read_unaligned()
            } else {
                let mut b = [0u8; 4];
                for (e, x) in b.iter_mut().enumerate() {
                    if 4 * d + e < k {
                        *x = *s.add(4 * d + e);
                    }
                }
                u32::from_le_bytes(b)
            };
            let (t, q) = (d / 16, d % 16);
            (out.as_mut_ptr().add(t * TILE + q * 64 + r * 4) as *mut u32).write_unaligned(w);
        }
    }
}

#[repr(C, align(64))]
struct Cfg([u8; 64]);

fn cfg() -> Cfg {
    let mut c = [0u8; 64];
    c[0] = 1;
    for t in 0..8 {
        c[16 + 2 * t] = 64;
        c[48 + t] = 16;
    }
    Cfg(c)
}

thread_local! {
    static ACC: RefCell<Vec<Line>> = const { RefCell::new(Vec::new()) };
}

/// A panel of 16 row-operand rows: tile t at `ptr + t * step`, rows `rs` bytes apart.
#[derive(Clone, Copy)]
struct Panel {
    ptr: usize,
    step: usize,
    rs: usize,
}

struct Job {
    rp: Vec<Panel>,    // row-operand panels
    vp: usize,         // packed VNNI panels (kt tiles each)
    kt: usize,
    nr: usize,         // row-operand rows
    nv: usize,         // VNNI-operand rows
    rows_are_a: bool,  // tile rows are i (A) and lanes j
    c: usize,
    ldc: usize,
    acc: bool,
}

/// Rows [r0, r0 + RB) x VNNI rows [v0, v0 + VB) of one batch.
unsafe fn task(j: &Job, r0: usize, rb: usize, v0: usize, vb: usize, kc: usize) {
    let cf = cfg();
    std::arch::asm!("ldtilecfg [{0}]", in(reg) cf.0.as_ptr(), options(nostack));
    let (nrp, nvp) = ((j.nr.min(r0 + rb) - r0).div_ceil(16), (j.nv.min(v0 + vb) - v0).div_ceil(16));
    let (erp, evp) = (nrp.next_multiple_of(2), nvp.next_multiple_of(2));
    ACC.with(|ab| {
        let cb = scratch(&mut ab.borrow_mut(), erp * evp * TILE);
        let (pr0, pv0) = (r0 / 16, v0 / 16);
        for kb in (0..j.kt).step_by(kc) {
            let ke = j.kt.min(kb + kc);
            for rp in (0..nrp).step_by(2) {
                let ra = j.rp[pr0 + rp];
                let rb_ = j.rp[pr0 + (rp + 1).min(nrp - 1)];
                for vp in (0..nvp).step_by(2) {
                    let (va, vb_) = (pv0 + vp, pv0 + (vp + 1).min(nvp - 1));
                    // C tile (row panel rp + dr, VNNI panel vp + dv) -> tmm(2 dr + dv)
                    let ct = |dr: usize, dv: usize| cb.add(((rp + dr) * evp + vp + dv) * TILE);
                    if kb == 0 {
                        std::arch::asm!("tilezero tmm0", "tilezero tmm1", "tilezero tmm2", "tilezero tmm3", options(nostack, nomem));
                    } else {
                        std::arch::asm!(
                            "tileloadd tmm0, [{c0} + {s}]", "tileloadd tmm1, [{c1} + {s}]",
                            "tileloadd tmm2, [{c2} + {s}]", "tileloadd tmm3, [{c3} + {s}]",
                            c0 = in(reg) ct(0, 0), c1 = in(reg) ct(0, 1), c2 = in(reg) ct(1, 0), c3 = in(reg) ct(1, 1),
                            s = in(reg) 64usize, options(nostack, readonly),
                        );
                    }
                    for t in kb..ke {
                        std::arch::asm!(
                            "tileloadd tmm4, [{r0} + {rs0}]", "tileloadd tmm5, [{r1} + {rs1}]",
                            "tileloadd tmm6, [{v0} + {s}]", "tileloadd tmm7, [{v1} + {s}]",
                            "tdpbssd tmm0, tmm4, tmm6", "tdpbssd tmm1, tmm4, tmm7",
                            "tdpbssd tmm2, tmm5, tmm6", "tdpbssd tmm3, tmm5, tmm7",
                            r0 = in(reg) ra.ptr + t * ra.step, rs0 = in(reg) ra.rs,
                            r1 = in(reg) rb_.ptr + t * rb_.step, rs1 = in(reg) rb_.rs,
                            v0 = in(reg) j.vp + (va * j.kt + t) * TILE, v1 = in(reg) j.vp + (vb_ * j.kt + t) * TILE,
                            s = in(reg) 64usize, options(nostack, readonly),
                        );
                    }
                    std::arch::asm!(
                        "tilestored [{c0} + {s}], tmm0", "tilestored [{c1} + {s}], tmm1",
                        "tilestored [{c2} + {s}], tmm2", "tilestored [{c3} + {s}], tmm3",
                        c0 = in(reg) ct(0, 0), c1 = in(reg) ct(0, 1), c2 = in(reg) ct(1, 0), c3 = in(reg) ct(1, 1),
                        s = in(reg) 64usize, options(nostack),
                    );
                }
            }
        }
        std::arch::asm!("tilerelease", options(nostack, nomem));
        // write C: tile (rp, vp) row x lane l is (row r0 + 16 rp + x, VNNI row v0 + 16 vp + l)
        let c = j.c as *mut i32;
        for rp in 0..nrp {
            for vp in 0..nvp {
                let t = cb.add((rp * evp + vp) * TILE) as *mut i32;
                let (gr, gv) = (r0 + 16 * rp, v0 + 16 * vp);
                let (xr, lv) = ((j.nr - gr).min(16), (j.nv - gv).min(16));
                if j.rows_are_a {
                    transpose16(t);
                    for l in 0..lv {
                        put(c.add((gv + l) * j.ldc + gr), t.add(l * 16), xr, j.acc);
                    }
                } else {
                    for x in 0..xr {
                        put(c.add((gr + x) * j.ldc + gv), t.add(x * 16), lv, j.acc);
                    }
                }
            }
        }
    });
}

/// The col-TN GEMM: C[:, j] (+)= op(A)^T B[:, j] for every batch.
///
/// # Safety
/// As `host::run`, with `spec` col-TN and `available()` true.
pub(crate) unsafe fn run(spec: &GemmSpec, a: *const i8, b: *const i8, c: *mut i32, acc: bool) {
    let (m, n, k) = (spec.m as usize, spec.n as usize, spec.k as usize);
    let kt = k.div_ceil(TK);
    let batch = spec.batch as usize;
    // the row operand is the larger side
    let rows_are_a = m >= n;
    let (rbase, rld, rstr, nr, vbase, vld, vstr, nv) = if rows_are_a {
        (a, spec.lda as usize, spec.stride_a as usize, m, b, spec.ldb as usize, spec.stride_b as usize, n)
    } else {
        (b, spec.ldb as usize, spec.stride_b as usize, n, a, spec.lda as usize, spec.stride_a as usize, m)
    };
    let (nrp, nvp) = (nr.div_ceil(16), nv.div_ceil(16));
    // row panels read in place unless their 16 x 64 kt byte window leaves the operand's rows
    let aligned = rld % 64 == 0 && rstr % 64 == 0 && rbase as usize % 64 == 0;
    let direct = |p: usize| aligned && (16 * p + 15) * rld + kt * TK <= (nr - 1) * rld + k;
    let tails: Vec<usize> = (0..nrp).filter(|&p| !direct(p)).collect();
    let vper = nvp * kt * TILE;
    let tper = tails.len() * kt * TILE;
    PACK.with(|pk| {
        let base = scratch(&mut pk.borrow_mut(), batch * (vper + tper)) as usize;
        let (rbase, vbase, c) = (rbase as usize, vbase as usize, c as usize);
        pc_hostrt::par(|| {
            let nt = tails.len();
            (0..batch * (nvp + nt)).into_par_iter().for_each(|x| {
                let (bt, p) = (x / (nvp + nt), x % (nvp + nt));
                if p < nvp {
                    let out = std::slice::from_raw_parts_mut((base + bt * vper + p * kt * TILE) as *mut u8, kt * TILE);
                    let src = (vbase as *const i8).add(bt * vstr + p * 16 * vld);
                    pack_vnni(src, vld, (nv - p * 16).min(16), k, out);
                } else {
                    let q = tails[p - nvp];
                    let out = std::slice::from_raw_parts_mut((base + batch * vper + bt * tper + (p - nvp) * kt * TILE) as *mut u8, kt * TILE);
                    let src = (rbase as *const i8).add(bt * rstr + q * 16 * rld);
                    pack_rows(src, rld, (nr - q * 16).min(16), k, out);
                }
            });
            let jobs: Vec<Job> = (0..batch)
                .map(|bt| Job {
                    rp: (0..nrp)
                        .map(|p| match tails.iter().position(|&q| q == p) {
                            Some(i) => Panel { ptr: base + batch * vper + bt * tper + i * kt * TILE, step: TILE, rs: TK },
                            None => Panel { ptr: rbase + bt * rstr + p * 16 * rld, step: TK, rs: rld },
                        })
                        .collect(),
                    vp: base + bt * vper,
                    kt,
                    nr,
                    nv,
                    rows_are_a,
                    c: c + bt * spec.stride_c as usize * 4,
                    ldc: spec.ldc as usize,
                    acc,
                })
                .collect();
            let (nrb, nvb) = (nr.div_ceil(RB), nv.div_ceil(VB));
            (0..batch * nrb * nvb).into_par_iter().for_each(|x| {
                let (bt, r) = (x / (nrb * nvb), x % (nrb * nvb));
                let (br, bv) = (r % nrb, r / nrb);
                task(&jobs[bt], br * RB, RB, bv * VB, VB, KC);
            });
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Op, Order};

    #[test]
    fn amx_matches_the_definition() {
        if !available() {
            return;
        }
        for (m, n, k, batch) in [(7, 5, 67, 3), (16, 16, 64, 1), (33, 47, 129, 2), (600, 140, 2100, 1), (512, 128, 4096, 2), (96, 640, 8256, 1), (100, 37, 4096, 2)] {
            for (acc, off) in [(false, 0), (true, 0), (false, 1), (true, 1)] {
                let s = GemmSpec::packed(m, n, k, Op::T, Op::N, Order::Col, batch);
                // operands on a 64-byte boundary (read in place when the leading dimension allows) or one byte off
                let (mut la, mut lb) = (Vec::new(), Vec::new());
                let a = unsafe { std::slice::from_raw_parts_mut(scratch(&mut la, s.len_a() + 1).add(off) as *mut i8, s.len_a()) };
                let b = unsafe { std::slice::from_raw_parts_mut(scratch(&mut lb, s.len_b() + 1).add(off) as *mut i8, s.len_b()) };
                a.iter_mut().enumerate().for_each(|(i, x)| *x = ((i * 37 + 11) % 256) as u8 as i8);
                b.iter_mut().enumerate().for_each(|(i, x)| *x = ((i * 91 + 5) % 253) as u8 as i8);
                let mut c: Vec<i32> = (0..s.len_c()).map(|i| (i % 1000) as i32 - 500).collect();
                let c0 = c.clone();
                unsafe { run(&s, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), acc) };
                let (m, n, k) = (m as usize, n as usize, k as usize);
                for bt in 0..batch as usize {
                    for j in 0..n {
                        for i in 0..m {
                            let mut want = 0i32;
                            for t in 0..k {
                                want += a[bt * s.stride_a as usize + i * s.lda as usize + t] as i32 * b[bt * s.stride_b as usize + j * s.ldb as usize + t] as i32;
                            }
                            let o = bt * s.stride_c as usize + j * s.ldc as usize + i;
                            if acc {
                                want = want.wrapping_add(c0[o]);
                            }
                            assert_eq!(c[o], want, "m {m} n {n} k {k} acc {acc} bt {bt} i {i} j {j}");
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::{Op, Order};

    #[test]
    #[ignore]
    fn amx_bench() {
        assert!(available());
        let shapes: Vec<(u64, u64, u64)> = match std::env::var("PC_AMX_SHAPE") {
            Ok(v) => {
                let x: Vec<u64> = v.split(',').map(|t| t.parse().unwrap()).collect();
                vec![(x[0], x[1], x[2])]
            }
            Err(_) => vec![(4096, 796, 32768), (8192, 4096, 16384), (8192, 5376, 4096)],
        };
        for (m, n, k) in shapes {
            let s = GemmSpec::packed(m, n, k, Op::T, Op::N, Order::Col, 1);
            let (mut va, mut vb) = (Vec::new(), Vec::new());
            let a = scratch(&mut va, s.len_a()) as *const i8;
            let b = scratch(&mut vb, s.len_b()) as *const i8;
            let mut c = vec![0i32; s.len_c()];
            for _ in 0..2 {
                unsafe { run(&s, a, b, c.as_mut_ptr(), false) };
            }
            let t = std::time::Instant::now();
            for _ in 0..3 {
                unsafe { run(&s, a, b, c.as_mut_ptr(), false) };
            }
            let dt = t.elapsed().as_secs_f64() / 3.0;
            eprintln!("m {m} n {n} k {k}: {:.2} ms, {:.2} TMAC/s", dt * 1e3, (m * n * k) as f64 / dt / 1e12);
        }
    }
}

