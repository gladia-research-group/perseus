// Composite one-pass rescale (rescale_coef / rescale_combine) as a standalone pair of kernels, generic
// over (W, MM) and over the number of dropped primes (1 or 2). Formula and host constants in args.cuh RescaleArgs;
// the host runs INTT(tops) -> rescale_coef -> NTT(tmp rows) -> rescale_combine. Dropping both tops in one pass is
// bit-identical to two sequential OpenFHE-style rescales.
#pragma once
#include "arith.cuh"

namespace pc {
namespace rescale {
constexpr int BX = 256;

// centered representative of y (mod qo) reduced mod the prime of c
template <class W, class MM>
__device__ __forceinline__ W lift(W y, W qo, const PrimeConst& c) {
    const W qn = (W)c.p;
    const W yr = MM::reduce((WideOf<W>)y, c);
    if (y > (W)(qo >> 1)) {
        const W qor = MM::reduce((WideOf<W>)qo, c);
        return yr >= qor ? (W)(yr - qor) : (W)(yr + qn - qor);
    }
    return yr;
}

template <class W, class MM>
__global__ void rescale_coef(RescaleArgs A) {
    // one thread per coefficient: the tops read once, y' computed once, all target rows written
    const int c = blockIdx.z >> 1, part = blockIdx.z & 1;
    const int i = blockIdx.x * BX + threadIdx.x;
    const int a = A.top, b = A.top - 1;
    const PrimeConst& pa = A.pr[a];
    const W qa = (W)pa.p;
    W ya = at<W>(A.x, c, part, a)[i];
    if (A.ca != 1) ya = MM::mul(ya, (W)A.ca, pa);
    const bool na = ya > (W)(qa >> 1);
    W yp = 0, qb = 0;
    bool nb = false;
    if (A.ndrop == 2) {
        const PrimeConst& pb = A.pr[b];
        qb = (W)pb.p;
        W yb = at<W>(A.x, c, part, b)[i];
        if (A.ca != 1) yb = MM::mul(yb, (W)A.cb, pb);
        yp = add<W>(MM::mul((W)A.qinv_ab, yb, pb), MM::mul((W)A.C_ab, lift<W, MM>(ya, qa, pb), pb), qb);
        nb = yp > (W)(qb >> 1);
    }
    const W* Av = (const W*)A.A;
    const W* As = (const W*)A.As;
    const W* Bv = (const W*)A.B;
    const W* Bs = (const W*)A.Bs;
    const W* qam = (const W*)A.qam;
    const W* qbm = (const W*)A.qbm;
    for (int row = blockIdx.y; row < A.nrows; row += gridDim.y) {
        const PrimeConst& pq = A.pr[row];
        const W q = (W)pq.p;
        W la = reduce_small<W, MM>(ya, pq);
        if (na) la = sub<W>(la, qam[row], q);
        W w = mul_shoup<W>(la, Av[row], As[row], q);
        if (A.ndrop == 2) {
            W lb = reduce_small<W, MM>(yp, pq);
            if (nb) lb = sub<W>(lb, qbm[row], q);
            w = add<W>(w, mul_shoup<W>(lb, Bv[row], Bs[row], q), q);
        }
        at<W>(A.tmp, c, part, row)[i] = w;
    }
}

// x_i = alpha_i x_i + w_i  (NTT domain), 4 consecutive words per thread
template <class W>
__global__ void rescale_combine(RescaleCombineArgs A) {
    const int row = blockIdx.y, c = blockIdx.z >> 1, part = blockIdx.z & 1;
    const W p = (W)A.pr[row].p;
    const int i = (blockIdx.x * BX + threadIdx.x) * 4;
    W* d = at<W>(A.x, c, part, row) + i;
    const W al = ((const W*)A.alpha)[row], als = ((const W*)A.alphas)[row];
    const V4<W> x = load4<W>(d), t = load4<W>(at<W>(A.tmp, c, part, row) + i);
    W r[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) r[w] = add<W>(mul_shoup<W>(x.v[w], al, als, p), t.v[w], p);
    store4<W>(d, r[0], r[1], r[2], r[3]);
}

template <class W, class MM>
int coef_l(const RescaleArgs* a, cudaStream_t s) {
    if (a->ndrop != 1 && a->ndrop != 2) return -1;
    rescale_coef<W, MM><<<dim3(N / BX, 4, 2 * a->K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W>
int combine_l(const RescaleCombineArgs* a, cudaStream_t s) {
    rescale_combine<W><<<dim3(N / (4 * BX), a->n, 2 * a->K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
}  // namespace rescale
}  // namespace pc
