// EvalMod element-wise kernels generic over (W, MM): Q rows [0, n) of K ciphertexts, row index ==
// prime id, 4 consecutive words per thread. Semantics in args.cuh (TensorArgs, MdEpiArgs, WsumArgs, AddConstArgs,
// MulTableArgs, MulAddArgs). The composite rescale lives in rescale.cuh.
#pragma once
#include "arith.cuh"
#include "chacha.cuh"

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
    W* o2 = at<W>(A.d2, c, 1, row) + i;
    if (A.acc & 2) {
        const V4<W> d = load4<W>(o2);
#pragma unroll
        for (int w = 0; w < 4; ++w) r2[w] = add<W>(r2[w], d.v[w], p);
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
    store4<W>(o2, r2[0], r2[1], r2[2], r2[3]);
    if (A.coef) store4<W>((W*)A.coef + c * A.coef_ct_stride + (size_t)row * N + i, r2[0], r2[1], r2[2], r2[3]);
}

template <class W, class MM>
__global__ void tensor_sum(TensorSumArgs A) {
    using Wd = WideOf<W>;
    const int row = blockIdx.y, c = blockIdx.z;
    const PrimeConst& q = A.pr[row];
    const W p = (W)q.p;
    const int i = (blockIdx.x * BX + threadIdx.x) * 4;
    Wd s0[4] = {0, 0, 0, 0}, s1[4] = {0, 0, 0, 0}, s2[4] = {0, 0, 0, 0};
    for (int j = 0; j < A.npairs; ++j) {
        const V4<W> x0 = load4<W>(at<W>(A.a[j], c, 0, row) + i), x1 = load4<W>(at<W>(A.a[j], c, 1, row) + i);
        const V4<W> y0 = load4<W>(at<W>(A.b[j], c, 0, row) + i), y1 = load4<W>(at<W>(A.b[j], c, 1, row) + i);
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
    W* o2 = at<W>(A.d2, c, 1, row) + i;
    if (A.acc) {
        const V4<W> d0 = load4<W>(o0), d1 = load4<W>(o1);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            r0[w] = add<W>(r0[w], d0.v[w], p);
            r1[w] = add<W>(r1[w], d1.v[w], p);
        }
    }
    if (A.acc & 2) {
        const V4<W> d = load4<W>(o2);
#pragma unroll
        for (int w = 0; w < 4; ++w) r2[w] = add<W>(r2[w], d.v[w], p);
    }
    store4<W>(o0, r0[0], r0[1], r0[2], r0[3]);
    store4<W>(o1, r1[0], r1[1], r1[2], r1[3]);
    store4<W>(o2, r2[0], r2[1], r2[2], r2[3]);
}

__device__ __forceinline__ bool same_ref(const PolyRef& x, const PolyRef& y) {
    return x.base == y.base && x.ct_stride == y.ct_stride && x.part_stride == y.part_stride;
}

// Coefficients i..i + 3 (i % 4 == 0) of a packed row (args.cuh, seeded | 4): their 112 bits start at bit 112 i / 4, in
// words 7 i / 8 .. + 3 from bit 0 or 16.
static_assert(PC_PACK_BITS == 28, "unpack4 reads 4 x 28 bits");
__device__ __forceinline__ V4<uint32_t> unpack4(const uint32_t* row, int i) {
    const int t = i >> 2;
    const uint32_t* w = row + ((7 * t) >> 1);
    const uint32_t s = (t & 1) * 16, w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    const uint32_t v0 = __funnelshift_r(w0, w1, s), v1 = __funnelshift_r(w1, w2, s), v2 = __funnelshift_r(w2, w3, s), v3 = w3 >> s;
    constexpr uint32_t M = (1u << PC_PACK_BITS) - 1;
    return V4<uint32_t>{{v0 & M, __funnelshift_r(v0, v1, 28) & M, __funnelshift_r(v1, v2, 24) & M, __funnelshift_r(v2, v3, 20) & M}};
}

// PACKED: seeded | 4 (its own instantiation: the other modes keep their code). At u32, 2 blocks per SM (128 registers,
// a few spilled): the kernel is bound by the loads in flight, ~20% faster than one block of 168 registers.
template <class W, class MM, bool PACKED>
__global__ void __launch_bounds__(BX, sizeof(W) == 4 ? 2 : 1) tensor_multi(TensorMultiArgs A) {
    using Wd = WideOf<W>;
    const int row = blockIdx.y, c = blockIdx.z;
    const PrimeConst& q = A.pr[row];
    const W p = (W)q.p;
    const int i = (blockIdx.x * BX + threadIdx.x) * 4;
    Wd s[PC_TMULTI_OUT][3][4];
#pragma unroll
    for (int o = 0; o < PC_TMULTI_OUT; ++o)
#pragma unroll
        for (int t = 0; t < 3; ++t)
#pragma unroll
            for (int w = 0; w < 4; ++w) s[o][t][w] = 0;
    for (int j = 0; j < A.npairs; ++j) {
        V4<W> x0, x1, y0, y1;
#pragma unroll
        for (int o = 0; o < PC_TMULTI_OUT; ++o) {
            if (o >= A.nout) break;
            if (o == 0 || !same_ref(A.a[o][j], A.a[o - 1][j])) {
                if constexpr (sizeof(W) == 4) {
                    if constexpr (PACKED) {
                        const V4<uint32_t> u = unpack4((const uint32_t*)at<W>(A.a[o][j], c, 0, 0) + (size_t)row * PC_PACK_ROW_WORDS, i);
#pragma unroll
                        for (int w = 0; w < 4; ++w) x0.v[w] = u.v[w];
                    } else {
                        x0 = load4<W>(at<W>(A.a[o][j], c, 0, row) + i);
                    }
                    if ((A.seeded & 3) == 2) {
                        x1 = load4<W>(at<W>(A.x1[o][j], 0, 0, row) + i);  // drawn by cts_expand
                    } else if (A.seeded & 3) {
                        // c1 drawn from the row's seed (part 1: 8 words per row); the host admits seeded at u32 only
                        const uint32_t* sd = (const uint32_t*)at<W>(A.a[o][j], c, 1, 0) + 8 * row;
                        uint32_t key[8], v[4];
#pragma unroll
                        for (int w = 0; w < 8; ++w) key[w] = sd[w];
                        chacha::cts_expand4(key, threadIdx.x & 31u, (uint32_t)p, (uint32_t)i, v);
#pragma unroll
                        for (int w = 0; w < 4; ++w) x1.v[w] = v[w];
                    } else {
                        x1 = load4<W>(at<W>(A.a[o][j], c, 1, row) + i);
                    }
                } else {
                    x0 = load4<W>(at<W>(A.a[o][j], c, 0, row) + i);
                    x1 = load4<W>(at<W>(A.a[o][j], c, 1, row) + i);
                }
            }
            if (o == 0 || !same_ref(A.b[o][j], A.b[o - 1][j])) {
                y0 = load4<W>(at<W>(A.b[o][j], c, 0, row) + i);
                y1 = load4<W>(at<W>(A.b[o][j], c, 1, row) + i);
            }
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                s[o][0][w] += (Wd)x0.v[w] * y0.v[w];
                s[o][1][w] += (Wd)x0.v[w] * y1.v[w] + (Wd)x1.v[w] * y0.v[w];
                s[o][2][w] += (Wd)x1.v[w] * y1.v[w];
            }
        }
    }
#pragma unroll
    for (int o = 0; o < PC_TMULTI_OUT; ++o) {
        if (o >= A.nout) break;
        W r0[4], r1[4], r2[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            r0[w] = MM::reduce(s[o][0][w], q);
            r1[w] = MM::reduce(s[o][1][w], q);
            r2[w] = MM::reduce(s[o][2][w], q);
        }
        W* o0 = at<W>(A.out[o], c, 0, row) + i;
        W* o1 = at<W>(A.out[o], c, 1, row) + i;
        W* o2 = at<W>(A.d2[o], c, 1, row) + i;
        if (A.acc) {
            const V4<W> d0 = load4<W>(o0), d1 = load4<W>(o1);
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                r0[w] = add<W>(r0[w], d0.v[w], p);
                r1[w] = add<W>(r1[w], d1.v[w], p);
            }
        }
        if (A.acc & 2) {
            const V4<W> d = load4<W>(o2);
#pragma unroll
            for (int w = 0; w < 4; ++w) r2[w] = add<W>(r2[w], d.v[w], p);
        }
        store4<W>(o0, r0[0], r0[1], r0[2], r0[3]);
        store4<W>(o1, r1[0], r1[1], r1[2], r1[3]);
        store4<W>(o2, r2[0], r2[1], r2[2], r2[3]);
    }
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

// One thread per 16-word block of a c1 row: grid (N / (16 BXE), n, nops).
constexpr int BXE = 128;
__global__ void cts_expand(CtsExpandArgs A) {
    const int row = blockIdx.y, u = blockIdx.z;
    const uint32_t b = blockIdx.x * BXE + threadIdx.x;
    const uint32_t* sd = (const uint32_t*)at<uint32_t>(A.src[u], 0, 1, 0) + 8 * row;
    uint32_t key[8], v[16];
#pragma unroll
    for (int w = 0; w < 8; ++w) key[w] = sd[w];
    chacha::regen_block16(key, b, 0u, (uint32_t)A.pr[row].p, (uint32_t)N >> 4, chacha::TAG_CTSA, v, chacha::CTS_T_MAX);
    uint4* d = (uint4*)((uint32_t*)A.dst + ((size_t)u * A.n + row) * N + 16 * (size_t)b);
#pragma unroll
    for (int w = 0; w < 4; ++w) d[w] = make_uint4(v[4 * w], v[4 * w + 1], v[4 * w + 2], v[4 * w + 3]);
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
template <class W, class MM>
int tensor_sum_l(const TensorSumArgs* a, cudaStream_t s) {
    if (a->npairs < 1 || a->npairs > PC_TSUM_MAX) return -1;
    tensor_sum<W, MM><<<dim3(N / (4 * BX), a->n, a->K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W, class MM>
int tensor_multi_l(const TensorMultiArgs* a, cudaStream_t s) {
    if (a->nout < 1 || a->nout > PC_TMULTI_OUT || a->npairs < 1 || a->npairs > PC_TMULTI_PAIRS) return -1;
    if (a->seeded & 4) {
        if (sizeof(W) != 4) return -1;
        tensor_multi<W, MM, true><<<dim3(N / (4 * BX), a->n, a->K), BX, 0, s>>>(*a);
    } else {
        tensor_multi<W, MM, false><<<dim3(N / (4 * BX), a->n, a->K), BX, 0, s>>>(*a);
    }
    return (int)cudaGetLastError();
}
template <class W>
int cts_expand_l(const CtsExpandArgs* a, cudaStream_t s) {
    if (sizeof(W) != 4 || a->nops < 1 || a->nops > PC_CTS_EXPAND_MAX) return -1;
    cts_expand<<<dim3(N / (16 * BXE), a->n, a->nops), BXE, 0, s>>>(*a);
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
