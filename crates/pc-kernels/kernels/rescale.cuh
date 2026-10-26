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

// rescale_coef's two sources of every coefficient (ndrop 2), into tmp rows top (y_a, scaled by ca) and top - 1 (y')
template <class W, class MM>
__global__ void rescale_prep(RescaleArgs A) {
    const int c = blockIdx.z >> 1, part = blockIdx.z & 1;
    const int i = blockIdx.x * BX + threadIdx.x;
    const int a = A.top, b = A.top - 1;
    const PrimeConst& pa = A.pr[a];
    const PrimeConst& pb = A.pr[b];
    const W qb = (W)pb.p;
    W ya = at<W>(A.x, c, part, a)[i];
    W yb = at<W>(A.x, c, part, b)[i];
    if (A.ca != 1) {
        ya = MM::mul(ya, (W)A.ca, pa);
        yb = MM::mul(yb, (W)A.cb, pb);
    }
    at<W>(A.tmp, c, part, a)[i] = ya;
    at<W>(A.tmp, c, part, b)[i] = add<W>(MM::mul((W)A.qinv_ab, yb, pb), MM::mul((W)A.C_ab, lift<W, MM>(ya, (W)pa.p, pb), pb), qb);
}

// The forward NTT's column pass of rescale_coef's rows (ndrop 2), each row generated from rescale_prep's sources
// (the same words rescale_coef stores, never stored here); row i of T is tmp row (c, part, r = rprime[i]).
template <class W, class MM, bool L>
__global__ void __launch_bounds__(256) cols_rs(NttArgs T, RescaleArgs A) {
    __shared__ W s[256 * 16];
    const int row = blockIdx.y;
    const int r = T.rprime[row];
    W* const base = (W*)T.base + (size_t)T.roff[row] * N + blockIdx.x * 16;
    const W* ya = base + (size_t)(A.top - r) * N;
    const W* yp = base + (size_t)(A.top - 1 - r) * N;
    const PrimeConst& pq = A.pr[r];
    const W q = (W)pq.p;
    const W ha = (W)A.pr[A.top].p >> 1, hb = (W)A.pr[A.top - 1].p >> 1;
    const W av = ((const W*)A.A)[r], as = ((const W*)A.As)[r], bv = ((const W*)A.B)[r], bs = ((const W*)A.Bs)[r];
    const W qam = ((const W*)A.qam)[r], qbm = ((const W*)A.qbm)[r];
    const int c = threadIdx.x & 15, g = threadIdx.x >> 4;
    W x[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int o = (g + 16 * j) * 256 + c;
        const W u = ya[o], v = yp[o];
        W la = reduce_small<W, MM>(u, pq);
        if (u > ha) la = sub<W>(la, qam, q);
        W lb = reduce_small<W, MM>(v, pq);
        if (v > hb) lb = sub<W>(lb, qbm, q);
        x[j] = add<W>(mul_shoup<W>(la, av, as, q), mul_shoup<W>(lb, bv, bs, q), q);
    }
    ntt::fwd_cols_core<W, L>(x, base, s, (const W*)T.fw + (size_t)r * N, (const W*)T.fws + (size_t)r * N, q, c, g);
}

template <class W, class MM>
int fwd_l(const RescaleNttArgs* a, cudaStream_t s) {
    const RescaleArgs& R = a->rs;
    const NttArgs& T = a->ntt;
    if (R.ndrop != 2 || T.soff || T.epi < 0 || T.epi > 3 || (T.epi && (!T.eout || !T.ex || !T.es || !T.ess))) return -1;
    rescale_prep<W, MM><<<dim3(N / BX, 1, 2 * R.K), BX, 0, s>>>(R);
    NttArgs b = T;
    b.snum = 1;
    b.xnum = max(T.xnum, 1);
    const bool lz = ntt::LAZY<W, true> && T.lazy;
    const int ch = T.chunk > 0 ? T.chunk : T.nrows;
    for (int i = 0; i < T.nrows; i += ch) {
        const int m = min(ch, T.nrows - i);
        b.roff = T.roff + i;
        b.rprime = T.rprime + i;
        if (lz) {
            cols_rs<W, MM, true><<<dim3(16, m), 256, 0, s>>>(b, R);
            ntt::ntt_fwd_rows<W, true><<<dim3(32, m), 256, 0, s>>>(b);
        } else {
            cols_rs<W, MM, false><<<dim3(16, m), 256, 0, s>>>(b, R);
            ntt::ntt_fwd_rows<W, false><<<dim3(32, m), 256, 0, s>>>(b);
        }
    }
    return (int)cudaGetLastError();
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
