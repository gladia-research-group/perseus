// The host GEMM behind a host `Lt`: D (+)= op(A) op(B) over every GemmSpec layout, exact in i32 (int8 products summed
// in i32, as the device's integer MMA). The col-TN form the matmuls use (both operands K-contiguous) runs as
// contiguous int8 dots, vectorized per CPU; every other layout takes the strided reference loop.
use crate::{GemmSpec, Op, Order};
use rayon::prelude::*;

/// Dot of two int8 vectors in i32 (the products widen to i16 pairs: the compiler's pmaddwd form).
#[inline(always)]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    let mut acc = [0i32; 16];
    let (ca, cb) = (a.chunks_exact(32), b.chunks_exact(32));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for l in 0..16 {
            acc[l] += x[2 * l] as i16 as i32 * y[2 * l] as i16 as i32 + x[2 * l + 1] as i16 as i32 * y[2 * l + 1] as i16 as i32;
        }
    }
    let mut s: i32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += *x as i32 * *y as i32;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2")]
unsafe fn tn_cols_avx512(spec: &GemmSpec, a: *const i8, b: *const i8, c: &mut [i32], j: usize, acc: bool) {
    tn_cols(spec, a, b, c, j, acc)
}

/// Column j of one batch, col-TN: c[i] (+)= <A[:, i], B[:, j]> (both K-contiguous).
#[inline(always)]
unsafe fn tn_cols(spec: &GemmSpec, a: *const i8, b: *const i8, c: &mut [i32], j: usize, acc: bool) {
    let k = spec.k as usize;
    let bj = std::slice::from_raw_parts(b.add(j * spec.ldb as usize), k);
    for (i, ci) in c.iter_mut().enumerate() {
        let ai = std::slice::from_raw_parts(a.add(i * spec.lda as usize), k);
        let d = dot_i8(ai, bj);
        *ci = if acc { ci.wrapping_add(d) } else { d };
    }
}

fn is_col_tn(s: &GemmSpec) -> bool {
    s.op_a == Op::T && s.op_b == Op::N && s.order_a == Order::Col && s.order_b == Order::Col && s.order_c == Order::Col
}

/// Run the GEMM on the calling thread's kernel pool.
///
/// # Safety
/// The pointers hold `len_a` / `len_b` int8 and `len_c` int32 elements; `c` aliases no input.
pub(crate) unsafe fn run(spec: &GemmSpec, a: *const i8, b: *const i8, c: *mut i32, accumulate: bool) {
    let (m, n, k) = (spec.m as usize, spec.n as usize, spec.k as usize);
    let (a, b, c) = (a as usize, b as usize, c as usize);
    #[cfg(target_arch = "x86_64")]
    let wide = std::is_x86_feature_detected!("avx512bw");
    pc_hostrt::par(|| {
        (0..spec.batch as usize * n).into_par_iter().for_each(|bj| {
            let (bt, j) = (bj / n, bj % n);
            let (a, b) = ((a as *const i8).add(bt * spec.stride_a as usize), (b as *const i8).add(bt * spec.stride_b as usize));
            let c = (c as *mut i32).add(bt * spec.stride_c as usize);
            if is_col_tn(spec) {
                let col = std::slice::from_raw_parts_mut(c.add(j * spec.ldc as usize), m);
                #[cfg(target_arch = "x86_64")]
                if wide {
                    return tn_cols_avx512(spec, a, b, col, j, accumulate);
                }
                return tn_cols(spec, a, b, col, j, accumulate);
            }
            for i in 0..m {
                let mut d = 0i32;
                for t in 0..k {
                    let (ar, ac) = if spec.op_a == Op::N { (i, t) } else { (t, i) };
                    let (br, bc) = if spec.op_b == Op::N { (t, j) } else { (j, t) };
                    let x = *a.add(GemmSpec::offset(spec.order_a, spec.lda, ar as u64, ac as u64)) as i32;
                    let y = *b.add(GemmSpec::offset(spec.order_b, spec.ldb, br as u64, bc as u64)) as i32;
                    d = d.wrapping_add(x * y);
                }
                let o = c.add(GemmSpec::offset(spec.order_c, spec.ldc, i as u64, j as u64));
                *o = if accumulate { (*o).wrapping_add(d) } else { d };
            }
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_gemm_matches_the_definition() {
        for (op_a, op_b, order) in [(Op::T, Op::N, Order::Col), (Op::N, Op::N, Order::Col), (Op::N, Op::T, Order::Row)] {
            let s = GemmSpec::packed(7, 5, 67, op_a, op_b, order, 3);
            let a: Vec<i8> = (0..s.len_a()).map(|i| ((i * 37 + 11) % 256) as u8 as i8).collect();
            let b: Vec<i8> = (0..s.len_b()).map(|i| ((i * 91 + 5) % 256) as u8 as i8).collect();
            let mut c = vec![3i32; s.len_c()];
            unsafe { run(&s, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), true) };
            for bt in 0..3usize {
                for i in 0..7u64 {
                    for j in 0..5u64 {
                        let mut want = 3i32;
                        for t in 0..67u64 {
                            let (ar, ac) = if op_a == Op::N { (i, t) } else { (t, i) };
                            let (br, bc) = if op_b == Op::N { (t, j) } else { (j, t) };
                            want += a[bt * s.stride_a as usize + GemmSpec::offset(order, s.lda, ar, ac)] as i32
                                * b[bt * s.stride_b as usize + GemmSpec::offset(order, s.ldb, br, bc)] as i32;
                        }
                        assert_eq!(c[bt * s.stride_c as usize + GemmSpec::offset(order, s.ldc, i, j)], want);
                    }
                }
            }
        }
    }
}
