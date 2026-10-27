// Element-wise kernels over polynomial row sets, batched over ciphertexts x parts (blockIdx.z) and
// rows (blockIdx.y), 4 consecutive words per thread (vector loads; rows are N-word aligned). Semantics in args.cuh
// (EwArgs, PtDotArgs).
#pragma once
#include "arith.cuh"

namespace pc {
namespace elem {
constexpr int BX = 256;

template <class W>
__global__ void ew_add(EwArgs A) {
    const int row = rs_row(A.rs, blockIdx.y);
    const int c = blockIdx.z / A.nparts, part = blockIdx.z % A.nparts;
    const W p = (W)A.pr[row].p;
    const int i = (blockIdx.x * BX + threadIdx.x) * 4;
    const V4<W> x = load4<W>(at<W>(A.a, c, part, row) + i), y = load4<W>(at<W>(A.b, c, part, row) + i);
    W r[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) r[w] = A.sub ? sub<W>(x.v[w], y.v[w], p) : add<W>(x.v[w], y.v[w], p);
    store4<W>(at<W>(A.out, c, part, row) + i, r[0], r[1], r[2], r[3]);
}

template <class W>
__global__ void ew_scale(EwArgs A) {
    const int row = rs_row(A.rs, blockIdx.y);
    const int c = blockIdx.z / A.nparts, part = blockIdx.z % A.nparts;
    const W p = (W)A.pr[row].p;
    const int i = (blockIdx.x * BX + threadIdx.x) * 4;
    W* o = at<W>(A.out, c, part, row) + i;
    if (A.zero_p && (int)blockIdx.y >= A.rs.nq) {
        store4<W>(o, 0, 0, 0, 0);
        return;
    }
    const W s = ((const W*)A.sc)[row], ss = ((const W*)A.scs)[row];
    const V4<W> x = load4<W>(at<W>(A.a, c, part, row) + i);
    store4<W>(o, mul_shoup<W>(x.v[0], s, ss, p), mul_shoup<W>(x.v[1], s, ss, p), mul_shoup<W>(x.v[2], s, ss, p), mul_shoup<W>(x.v[3], s, ss, p));
}

template <class W>
__global__ void ew_sub_mul(EwArgs A) {
    const int row = rs_row(A.rs, blockIdx.y);
    const int c = blockIdx.z / A.nparts, part = blockIdx.z % A.nparts;
    const W p = (W)A.pr[row].p;
    const int i = (blockIdx.x * BX + threadIdx.x) * 4;
    const W s = ((const W*)A.sc)[row], ss = ((const W*)A.scs)[row];
    const V4<W> x = load4<W>(at<W>(A.a, c, part, row) + i), y = load4<W>(at<W>(A.b, c, part, row) + i);
    W r[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) r[w] = mul_shoup<W>(sub<W>(x.v[w], y.v[w], p), s, ss, p);
    store4<W>(at<W>(A.out, c, part, row) + i, r[0], r[1], r[2], r[3]);
}

template <class W>
__global__ void ew_copy(EwArgs A) {
    const int row = rs_row(A.rs, blockIdx.y);
    const int c = blockIdx.z / A.nparts, part = blockIdx.z % A.nparts;
    const int i = (blockIdx.x * BX + threadIdx.x) * 4;
    const V4<W> x = load4<W>(at<W>(A.a, c, part, row) + i);
    store4<W>(at<W>(A.out, c, part, row) + i, x.v[0], x.v[1], x.v[2], x.v[3]);
}

template <class W, class MM, int KMAX>
__global__ void pt_dot(PtDotArgs A) {
    using Wd = WideOf<W>;
    const int y = blockIdx.y;
    const int row = rs_row(A.rs, y);
    const int prow = y < A.rs.nq ? y : A.pt_prow + (y - A.rs.nq);
    const PrimeConst& q = A.pr[row];
    const int i = blockIdx.x * BX + threadIdx.x;
    const int pi = ptidx(i, A.enc);
    for (int j = 0; j < A.gstep; ++j) {
        Wd s0[KMAX], s1[KMAX];
#pragma unroll
        for (int c = 0; c < KMAX; ++c) s0[c] = s1[c] = 0;
        for (int b = 0; b < A.bstep; ++b) {
            const int d = A.bstep * j + b;
            if (d >= A.npts) break;
            const W w = __ldg((const W*)A.pts[d] + (size_t)prow * A.ptw + pi);
#pragma unroll
            for (int c = 0; c < KMAX; ++c) {
                if (c < A.K) {
                    const W* r0 = (const W*)A.rot.base + c * A.rot.ct_stride + b * A.rot_stride + (size_t)row * N;
                    s0[c] += (Wd)r0[i] * w;
                    s1[c] += (Wd)r0[A.rot.part_stride + i] * w;
                }
            }
        }
#pragma unroll
        for (int c = 0; c < KMAX; ++c) {
            if (c < A.K) {
                W* o = (W*)A.res.base + c * A.res.ct_stride + j * A.res_stride + (size_t)row * N;
                o[i] = MM::reduce(s0[c], q);
                o[A.res.part_stride + i] = MM::reduce(s1[c], q);
            }
        }
    }
}

inline dim3 ew_grid(const EwArgs* a) { return dim3(N / (4 * BX), a->rs.nq + a->rs.np, a->K * a->nparts); }

template <class W>
int add_l(const EwArgs* a, cudaStream_t s) {
    ew_add<W><<<ew_grid(a), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W>
int scale_l(const EwArgs* a, cudaStream_t s) {
    ew_scale<W><<<ew_grid(a), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W>
int sub_mul_l(const EwArgs* a, cudaStream_t s) {
    ew_sub_mul<W><<<ew_grid(a), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W>
int copy_l(const EwArgs* a, cudaStream_t s) {
    ew_copy<W><<<ew_grid(a), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W, class MM>
int pt_dot_l(const PtDotArgs* a, cudaStream_t s) {
    dim3 grid(N / BX, a->rs.nq + a->rs.np);
    if (a->K <= 1) pt_dot<W, MM, 1><<<grid, BX, 0, s>>>(*a);
    else if (a->K <= 2) pt_dot<W, MM, 2><<<grid, BX, 0, s>>>(*a);
    else if (a->K <= 4) pt_dot<W, MM, 4><<<grid, BX, 0, s>>>(*a);
    else if (a->K <= 8) pt_dot<W, MM, 8><<<grid, BX, 0, s>>>(*a);
    else return -1;
    return (int)cudaGetLastError();
}
}  // namespace elem
}  // namespace pc
