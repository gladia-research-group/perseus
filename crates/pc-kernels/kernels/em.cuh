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

template <class W, class MM>
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
                x0 = load4<W>(at<W>(A.a[o][j], c, 0, row) + i);
                x1 = load4<W>(at<W>(A.a[o][j], c, 1, row) + i);
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

// Coefficients i..i + 3 (i % 4 == 0) of a packed row (args.cuh, TensorSeededArgs): their 112 bits start at bit 112 i / 4,
// in words 7 i / 8 .. + 3 from bit 0 or 16.
static_assert(PC_PACK_BITS == 28, "unpack4 reads 4 x 28 bits");
__device__ __forceinline__ V4<uint32_t> unpack4(const uint32_t* row, int i) {
    const int t = i >> 2;
    const uint32_t* w = row + ((7 * t) >> 1);
    const uint32_t s = (t & 1) * 16, w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    const uint32_t v0 = __funnelshift_r(w0, w1, s), v1 = __funnelshift_r(w1, w2, s), v2 = __funnelshift_r(w2, w3, s), v3 = w3 >> s;
    constexpr uint32_t M = (1u << PC_PACK_BITS) - 1;
    return V4<uint32_t>{{v0 & M, __funnelshift_r(v0, v1, 28) & M, __funnelshift_r(v1, v2, 24) & M, __funnelshift_r(v2, v3, 20) & M}};
}

// tensor_seeded at u32: a block is a TW-word tile of one row for nout outputs (TW / 4 threads per output, 4 words
// each), in stages of SEED_JS pairs. A stage's c0 and b words are copied into shared memory asynchronously (cp.async)
// while its c1 words are drawn there, one ChaCha block (16 words) per thread: nout TW / 16 blocks per pair =
// blockDim.x per stage. Every output's products then read their operands from shared memory (b once per block).
// 3 blocks per SM: 80 registers, no spills (4 blocks spill and run 1.6x slower)
constexpr int SEED_JS = 4, SEED_MINB = 3;
// uint4 u of an output's c1 tile: each group of 4 rotated by u / 8, so the 8 uint4 of a quarter warp's accesses (a
// drawing thread's w-th of its block, or 8 consecutive words of the products) hit distinct banks
__device__ __forceinline__ int seed_swz(int u) { return (u & ~3) | ((u + (u >> 3)) & 3); }
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory"); }
// uint4 of a stage's shared memory: c1 (nout TW / 4 per pair), c0 (nout c0_q(TW) per pair), b (2 TW / 4 per pair)
template <bool PACKED>
__host__ __device__ constexpr int c0_q(int tw) { return PACKED ? tw * PC_PACK_BITS / 128 : tw / 4; }
template <bool PACKED, int TW>
__host__ __device__ constexpr int seed_smem_q(int nout) { return SEED_JS * (nout * TW / 4 + nout * c0_q<PACKED>(TW) + 2 * TW / 4); }

// The c1 words of drawing thread t's ChaCha block (pair j0 + t / (nout TW / 16), output t / (TW / 16) % nout, block t %
// (TW / 16) of the tile) into the stage's c1 tile: need == 0, the first draw (returns the mask of rejected words); else
// the redraws of the words in need (chacha::regen_block16's words: each attempt's word stored over the last).
template <int TW>
__device__ __forceinline__ uint32_t seed_draw(const TensorSeededArgs& A, int row, int j0, int nout, int t, uint32_t p, uint4* xs, uint32_t need) {
    constexpr int TPO = TW / 4, NB = TW / 16;
    const int gj = t / (nout * NB), go = t / NB % nout, gb = t % NB;
    const uint32_t* sd = (const uint32_t*)A.a[go][j0 + gj] + A.seed_off + 8 * row;
    uint32_t key[8], v[16];
#pragma unroll
    for (int w = 0; w < 8; ++w) key[w] = sd[w];
    const uint32_t b = (uint32_t)(blockIdx.x * NB + gb), m_p = (0xFFFFFFFFu / p) * p, rp = 0xFFFFFFFFu / p;
    const auto red = [&](uint32_t x) {
        x -= __umulhi(x, rp) * p;
        return x >= p ? x - p : x;
    };
    uint4* d = xs + (gj * nout + go) * TPO;
    if (!need) {
        chacha::chacha12(key, b, 0u, p, chacha::TAG_CTSA, v);
#pragma unroll
        for (int k = 0; k < 16; ++k) need |= (v[k] >= m_p ? 1u : 0u) << k;
#pragma unroll
        for (int w = 0; w < 4; ++w) d[seed_swz(4 * gb + w)] = make_uint4(red(v[4 * w]), red(v[4 * w + 1]), red(v[4 * w + 2]), red(v[4 * w + 3]));
        return need;
    }
    for (uint32_t u = 1; need && u < chacha::CTS_T_MAX; ++u) {
        chacha::chacha12(key, b + u * ((uint32_t)N >> 4), 0u, p, chacha::TAG_CTSA, v);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            if (need >> k & 1u) {
                ((uint32_t*)&d[seed_swz(4 * gb + (k >> 2))])[k & 3] = red(v[k]);
                if (v[k] < m_p) need &= ~(1u << k);
            }
        }
    }
    return 0;
}

template <class MM, bool PACKED, int TW>
__global__ void __launch_bounds__(256, SEED_MINB) tensor_seeded(TensorSeededArgs A) {
    constexpr int TPO = TW / 4, NB = TW / 16, C0Q = c0_q<PACKED>(TW);
    extern __shared__ uint4 sm[];
    const int nout = A.nout, row = blockIdx.y, t = threadIdx.x, o = t / TPO, l = t % TPO;
    uint4* xs = sm;                                  // [SEED_JS][nout TPO], seed_swz within an output
    uint4* cs = xs + SEED_JS * nout * TPO;           // [SEED_JS][nout][C0Q]
    uint4* ys = cs + SEED_JS * nout * C0Q;           // [SEED_JS][2][TPO]
    const PrimeConst& q = A.pr[row];
    const uint32_t p = (uint32_t)q.p;
    const int w0 = blockIdx.x * TW;
    const int gj = t / (nout * NB);
    const size_t c0off = (size_t)row * (PACKED ? PC_PACK_ROW_WORDS : N) + (size_t)w0 * (PACKED ? PC_PACK_BITS : 32) / 32;
    const int ncs = SEED_JS * nout * C0Q, ncp = ncs + SEED_JS * 2 * TPO;
    uint64_t s[3][4];
#pragma unroll
    for (int u = 0; u < 3; ++u)
#pragma unroll
        for (int w = 0; w < 4; ++w) s[u][w] = 0;
    for (int j0 = 0; j0 < A.npairs; j0 += SEED_JS) {
        if (j0) __syncthreads();
        for (int k = t; k < ncp; k += blockDim.x) {
            if (k < ncs) {
                const int jj = k / (nout * C0Q), r = k % (nout * C0Q);
                if (j0 + jj < A.npairs) cp_async16(cs + k, (const uint4*)((const uint32_t*)A.a[r / C0Q][j0 + jj] + c0off) + r % C0Q);
            } else {
                const int u = k - ncs, jj = u / (2 * TPO), part = u / TPO % 2;
                if (j0 + jj < A.npairs) cp_async16(ys + u, (const uint4*)(at<uint32_t>(A.b[j0 + jj], 0, part, row) + w0) + u % TPO);
            }
        }
        if (j0 + gj < A.npairs) {
            const uint32_t need = seed_draw<TW>(A, row, j0, nout, t, p, xs, 0u);
            if (need) seed_draw<TW>(A, row, j0, nout, t, p, xs, need);
        }
        cp_async_wait();
        __syncthreads();
#pragma unroll
        for (int jj = 0; jj < SEED_JS; ++jj) {
            if (j0 + jj >= A.npairs) break;
            const uint32_t* c0 = (const uint32_t*)(cs + (jj * nout + o) * C0Q);
            const V4<uint32_t> x0 = PACKED ? unpack4(c0, 4 * l) : load4<uint32_t>(c0 + 4 * l);
            const uint4 x1 = xs[(jj * nout + o) * TPO + seed_swz(l)];
            const uint32_t x1v[4] = {x1.x, x1.y, x1.z, x1.w};
            const uint4 y0 = ys[(jj * 2) * TPO + l], y1 = ys[(jj * 2 + 1) * TPO + l];
            const uint32_t y0v[4] = {y0.x, y0.y, y0.z, y0.w}, y1v[4] = {y1.x, y1.y, y1.z, y1.w};
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                s[0][w] += (uint64_t)x0.v[w] * y0v[w];
                s[1][w] += (uint64_t)x0.v[w] * y1v[w] + (uint64_t)x1v[w] * y0v[w];
                s[2][w] += (uint64_t)x1v[w] * y1v[w];
            }
        }
    }
    const int i = w0 + 4 * l;
    uint32_t r0[4], r1[4], r2[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        r0[w] = MM::reduce(s[0][w], q);
        r1[w] = MM::reduce(s[1][w], q);
        r2[w] = MM::reduce(s[2][w], q);
    }
    uint32_t* o0 = at<uint32_t>(A.out[o], 0, 0, row) + i;
    uint32_t* o1 = at<uint32_t>(A.out[o], 0, 1, row) + i;
    uint32_t* o2 = at<uint32_t>(A.d2[o], 0, 1, row) + i;
    if (A.acc) {
        const V4<uint32_t> d0 = load4<uint32_t>(o0), d1 = load4<uint32_t>(o1);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            r0[w] = add<uint32_t>(r0[w], d0.v[w], p);
            r1[w] = add<uint32_t>(r1[w], d1.v[w], p);
        }
    }
    if (A.acc & 2) {
        const V4<uint32_t> d = load4<uint32_t>(o2);
#pragma unroll
        for (int w = 0; w < 4; ++w) r2[w] = add<uint32_t>(r2[w], d.v[w], p);
    }
    store4<uint32_t>(o0, r0[0], r0[1], r0[2], r0[3]);
    store4<uint32_t>(o1, r1[0], r1[1], r1[2], r1[3]);
    store4<uint32_t>(o2, r2[0], r2[1], r2[2], r2[3]);
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
template <class W, class MM>
int tensor_sum_l(const TensorSumArgs* a, cudaStream_t s) {
    if (a->npairs < 1 || a->npairs > PC_TSUM_MAX) return -1;
    tensor_sum<W, MM><<<dim3(N / (4 * BX), a->n, a->K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W, class MM>
int tensor_multi_l(const TensorMultiArgs* a, cudaStream_t s) {
    if (a->nout < 1 || a->nout > PC_TMULTI_OUT || a->npairs < 1 || a->npairs > PC_TMULTI_PAIRS) return -1;
    tensor_multi<W, MM><<<dim3(N / (4 * BX), a->n, a->K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class MM, bool PACKED, int TW>
int tensor_seeded_tw(const TensorSeededArgs* a, cudaStream_t s) {
    tensor_seeded<MM, PACKED, TW><<<dim3(N / TW, a->n), a->nout * TW / 4, seed_smem_q<PACKED, TW>(a->nout) * 16, s>>>(*a);
    return (int)cudaGetLastError();
}
template <class W, class MM>
int tensor_seeded_l(const TensorSeededArgs* a, cudaStream_t s) {
    if constexpr (sizeof(W) != 4) {
        return -1;
    } else {
        if (a->nout < 1 || a->nout > PC_TSEED_OUT || a->npairs < 1 || a->npairs > PC_TSEED_PAIRS) return -1;
        // 64-word tiles (a warp from two outputs on; wider tiles measured no faster)
        if (a->nout == 1) return a->packed ? tensor_seeded_tw<MM, true, 128>(a, s) : tensor_seeded_tw<MM, false, 128>(a, s);
        return a->packed ? tensor_seeded_tw<MM, true, 64>(a, s) : tensor_seeded_tw<MM, false, 64>(a, s);
    }
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
