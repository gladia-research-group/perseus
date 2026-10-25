// EvalMod element-wise kernels generic over (W, MM): Q rows [0, n) of K ciphertexts, row index ==
// prime id, 4 consecutive words per thread. Semantics in args.cuh (TensorArgs, MdEpiArgs, WsumArgs, AddConstArgs,
// MulTableArgs, MulAddArgs). The composite rescale lives in rescale.cuh.
#pragma once
#include "arith.cuh"

namespace pc {
namespace em {
constexpr int BX = 256;

template <class W, class MM>
__global__ void tensor(TensorArgs A) {
    using Wd = WideOf<W>;
    const int row = blockIdx.y, c = blockIdx.z;
    const PrimeConst& q = A.pr[row];
    const W p = (W)q.p;
    const int i = (blockIdx.x * BX + threadIdx.x) * 4;
    Wd s0[4] = {0, 0, 0, 0}, s1[4] = {0, 0, 0, 0}, s2[4] = {0, 0, 0, 0};
    for (int j = 0; j < A.npairs && j < 2; ++j) {
        const PolyRef& a = j ? A.a1 : A.a0;
        const PolyRef& b = j ? A.b1 : A.b0;
        const V4<W> x0 = load4<W>(at<W>(a, c, 0, row) + i), x1 = load4<W>(at<W>(a, c, 1, row) + i);
        const V4<W> y0 = load4<W>(at<W>(b, c, 0, row) + i), y1 = load4<W>(at<W>(b, c, 1, row) + i);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            s0[w] += (Wd)x0.v[w] * y0.v[w];
            s1[w] += (Wd)x0.v[w] * y1.v[w] + (Wd)x1.v[w] * y0.v[w];
            s2[w] += (Wd)x1.v[w] * y1.v[w];
        }
    }
    W r0[4], r1[4], r2[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        r0[w] = MM::reduce(s0[w], q);
        r1[w] = MM::reduce(s1[w], q);
        r2[w] = MM::reduce(s2[w], q);
    }
    W* o0 = at<W>(A.out, c, 0, row) + i;
    W* o1 = at<W>(A.out, c, 1, row) + i;
    if (A.acc) {
        const V4<W> d0 = load4<W>(o0), d1 = load4<W>(o1);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            r0[w] = add<W>(r0[w], d0.v[w], p);
            r1[w] = add<W>(r1[w], d1.v[w], p);
        }
    }
    if (A.dbl) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            r0[w] = add<W>(r0[w], r0[w], p);
            r1[w] = add<W>(r1[w], r1[w], p);
            r2[w] = add<W>(r2[w], r2[w], p);
        }
    }
    store4<W>(o0, r0[0], r0[1], r0[2], r0[3]);
    store4<W>(o1, r1[0], r1[1], r1[2], r1[3]);
    store4<W>(at<W>(A.d2, c, 1, row) + i, r2[0], r2[1], r2[2], r2[3]);
    if (A.coef) store4<W>((W*)A.coef + c * A.coef_ct_stride + (size_t)row * N + i, r2[0], r2[1], r2[2], r2[3]);
}

template <class W>
__global__ void md_epilogue(MdEpiArgs A) {
    const int row = blockIdx.y, c = blockIdx.z >> 1, part = blockIdx.z & 1;
    const W p = (W)A.pr[row].p;
    const int i = (blockIdx.x * BX + threadIdx.x) * 4;
    W* d = at<W>(A.dst, c, part, row) + i;
    const W bi = ((const W*)A.binv)[row], bis = ((const W*)A.binvs)[row];
    const W* k = (const W*)A.k;
    const W kk = k && part == 0 ? (k[row] >= p ? (W)(k[row] - p) : k[row]) : (W)0;
    const V4<W> x = load4<W>(at<W>(A.x, c, part, row) + i), t = load4<W>(at<W>(A.t, c, part, row) + i), dv = load4<W>(d);
    W r[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        r[w] = add<W>(dv.v[w], mul_shoup<W>(sub<W>(x.v[w], t.v[w], p), bi, bis, p), p);
        if (A.dbl) r[w] = add<W>(r[w], r[w], p);
        if (k && part == 0) r[w] = add<W>(r[w], kk, p);
    }
    store4<W>(d, r[0], r[1], r[2], r[3]);
}

template <class W>
__global__ void md_epilogue_to(MdEpiToArgs B) {
    const MdEpiArgs& A = B.m;
    const int row = blockIdx.y, c = blockIdx.z >> 1, part = blockIdx.z & 1;
    const W p = (W)A.pr[row].p;
    const int i = (blockIdx.x * BX + threadIdx.x) * 4;
    const W bi = ((const W*)A.binv)[row], bis = ((const W*)A.binvs)[row];
    const V4<W> x = load4<W>(at<W>(A.x, c, part, row) + i), t = load4<W>(at<W>(A.t, c, part, row) + i), a = load4<W>(at<W>(B.a, c, part, row) + i);
    W r[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const W v = mul_shoup<W>(sub<W>(x.v[w], t.v[w], p), bi, bis, p);
        r[w] = B.sub ? sub<W>(a.v[w], v, p) : add<W>(a.v[w], v, p);
    }
    store4<W>(at<W>(A.dst, c, part, row) + i, r[0], r[1], r[2], r[3]);
}

template <class W, bool KADD>
__global__ void wsum(WsumArgs A, const void* kc) {
    const int row = blockIdx.y, c = blockIdx.z;
    const W p = (W)A.pr[row].p;
    const int i = (blockIdx.x * BX + threadIdx.x) * 4;
    W r0[4] = {0, 0, 0, 0}, r1[4] = {0, 0, 0, 0};
    for (int j = 0; j < A.nsrc; ++j) {
        const W w = ((const W*)A.w)[j * A.rowstride + row], ws = ((const W*)A.ws)[j * A.rowstride + row];
        const V4<W> x0 = load4<W>(at<W>(A.src[j], c, 0, row) + i), x1 = load4<W>(at<W>(A.src[j], c, 1, row) + i);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            r0[u] = add<W>(r0[u], mul_shoup<W>(x0.v[u], w, ws, p), p);
            r1[u] = add<W>(r1[u], mul_shoup<W>(x1.v[u], w, ws, p), p);
        }
    }
    if constexpr (KADD) {
        const W k = ((const W*)kc)[row];
        const W kr = k >= p ? (W)(k - p) : k;
#pragma unroll
        for (int u = 0; u < 4; ++u) r0[u] = add<W>(r0[u], kr, p);
    }
    store4<W>(at<W>(A.out, c, 0, row) + i, r0[0], r0[1], r0[2], r0[3]);
    store4<W>(at<W>(A.out, c, 1, row) + i, r1[0], r1[1], r1[2], r1[3]);
}

// out = alpha x + tmp (rescale_combine with a separate output; x unchanged)
template <class W>
__global__ void rescale_combine_to(RescaleCombineToArgs R) {
    const RescaleCombineArgs& A = R.a;
    const int row = blockIdx.y, c = blockIdx.z >> 1, part = blockIdx.z & 1;
    const W p = (W)A.pr[row].p;
    const int i = (blockIdx.x * BX + threadIdx.x) * 4;
    const W al = ((const W*)A.alpha)[row], als = ((const W*)A.alphas)[row];
    const V4<W> x = load4<W>(at<W>(A.x, c, part, row) + i), t = load4<W>(at<W>(A.tmp, c, part, row) + i);
    W r[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) r[w] = add<W>(mul_shoup<W>(x.v[w], al, als, p), t.v[w], p);
    store4<W>(at<W>(R.out, c, part, row) + i, r[0], r[1], r[2], r[3]);
}

template <class W>
__global__ void add_const(AddConstArgs A) {
    const int row = blockIdx.y, c = blockIdx.z;
    const W p = (W)A.pr[row].p;
    const int i = (blockIdx.x * BX + threadIdx.x) * 4;
    W* d = at<W>(A.x, c, 0, row) + i;
    const W k = ((const W*)A.k)[row];
    const W kr = k >= p ? (W)(k - p) : k;
    const V4<W> x = load4<W>(d);
    store4<W>(d, add<W>(x.v[0], kr, p), add<W>(x.v[1], kr, p), add<W>(x.v[2], kr, p), add<W>(x.v[3], kr, p));
}

template <class W, class MM>
__global__ void mul_table(MulTableArgs A) {
    const int row = blockIdx.y, part = blockIdx.z & 1, cc = blockIdx.z >> 1;
    const PrimeConst& q = A.pr[row];
    const int i = (blockIdx.x * BX + threadIdx.x) * 4;
    const V4<W> x = load4<W>(at<W>(A.in, cc, part, row) + i), t = load4<W>((const W*)A.t + (size_t)row * N + i);
    store4<W>(at<W>(A.out, cc, part, row) + i, MM::mul(x.v[0], t.v[0], q), MM::mul(x.v[1], t.v[1], q), MM::mul(x.v[2], t.v[2], q), MM::mul(x.v[3], t.v[3], q));
}

template <class W, class MM>
__global__ void mul_add(MulAddArgs B) {
    const MulTableArgs& A = B.m;
    const int row = blockIdx.y, part = blockIdx.z & 1, cc = blockIdx.z >> 1;
    const PrimeConst& q = A.pr[row];
    const W p = (W)q.p;
    const int i = (blockIdx.x * BX + threadIdx.x) * 4;
    const V4<W> x = load4<W>(at<W>(A.in, cc, part, row) + i), t = load4<W>((const W*)A.t + (size_t)row * N + i);
    const V4<W> y = load4<W>(at<W>(B.y, cc, part, row) + i);
    W r[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const W m = MM::mul(x.v[w], t.v[w], q);
        r[w] = B.sub == 0 ? add<W>(y.v[w], m, p) : B.sub == 1 ? sub<W>(y.v[w], m, p) : sub<W>(m, y.v[w], p);
    }
    store4<W>(at<W>(A.out, cc, part, row) + i, r[0], r[1], r[2], r[3]);
}

template <class W, class MM>
int tensor_l(const TensorArgs* a, cudaStream_t s) {
    tensor<W, MM><<<dim3(N / (4 * BX), a->n, a->K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W>
int md_epilogue_l(const MdEpiArgs* a, cudaStream_t s) {
    md_epilogue<W><<<dim3(N / (4 * BX), a->n, 2 * a->K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W>
int md_epilogue_to_l(const MdEpiToArgs* a, cudaStream_t s) {
    md_epilogue_to<W><<<dim3(N / (4 * BX), a->m.n, 2 * a->m.K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W>
int wsum_l(const WsumArgs* a, cudaStream_t s) {
    if (a->nsrc > 8) return -1;
    wsum<W, false><<<dim3(N / (4 * BX), a->n, a->K), BX, 0, s>>>(*a, nullptr);
    return (int)cudaGetLastError();
}
template <class W>
int wsum_k_l(const WsumKArgs* a, cudaStream_t s) {
    if (a->w.nsrc > 8 || !a->k) return -1;
    wsum<W, true><<<dim3(N / (4 * BX), a->w.n, a->w.K), BX, 0, s>>>(a->w, a->k);
    return (int)cudaGetLastError();
}
template <class W>
int combine_to_l(const RescaleCombineToArgs* a, cudaStream_t s) {
    rescale_combine_to<W><<<dim3(N / (4 * BX), a->a.n, 2 * a->a.K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W>
int add_const_l(const AddConstArgs* a, cudaStream_t s) {
    add_const<W><<<dim3(N / (4 * BX), a->n, a->K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W, class MM>
int mul_add_l(const MulAddArgs* a, cudaStream_t s) {
    if (a->sub < 0 || a->sub > 2) return -1;
    mul_add<W, MM><<<dim3(N / (4 * BX), a->m.n, 2 * a->m.K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W, class MM>
int mul_table_l(const MulTableArgs* a, cudaStream_t s) {
    mul_table<W, MM><<<dim3(N / (4 * BX), a->n, 2 * a->K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
}  // namespace em
}  // namespace pc
