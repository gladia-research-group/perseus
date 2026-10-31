// Batched fast base conversion, generic over (W, MM); layout and formula in args.cuh BconvArgs.
// The exact (HPS-corrected) jobs keep FIDESlib's FMA double estimate in source order.
#pragma once
#include "arith.cuh"

namespace pc {
namespace bconv {
constexpr int BX = 128;  // coefficients per block
constexpr int TG = 32;   // targets per block (blockIdx.y)

template <class W, class MM>
__global__ void __launch_bounds__(BX) bconv_kernel(const W* __restrict__ src, W* __restrict__ dst,
                                                   const BconvJob* __restrict__ jobs, const W* __restrict__ pool,
                                                   const PrimeConst* __restrict__ pr) {
    using Wd = WideOf<W>;
    const BconvJob J = jobs[blockIdx.z];
    const int t0 = blockIdx.y * TG;
    if (t0 >= (int)J.ntgt) return;
    const int nt = min(TG, (int)J.ntgt - t0);
    const int ns = J.nsrc;
    const W* P = pool + J.pool_off;
    const W* srow = P;
    const W* sprime = P + ns;
    const W* pre = P + 2 * ns;
    const W* trow = P + 3 * ns;
    const W* tprime = trow + J.ntgt;
    const W* M = tprime + J.ntgt;
    __shared__ __align__(32) W sm[16 * TG];
    __shared__ uint32_t stp[TG], str[TG];
    // J.split (u32 words, every target below 2^30): the sums are reduced as hi 2^32 + lo with 32-bit Shoup products
    // (R = 2^32 mod p, Rs = floor(R 2^32 / p), m = floor(2^32 / p)) from shared memory: the same canonical residue
    __shared__ uint32_t sp_[TG], sR[TG], sRs[TG], sm32[TG];
    const bool fast = sizeof(W) == 4 && J.split && !J.exact;
    for (int x = threadIdx.x; x < ns * TG; x += BX) {
        const int i = x / TG, k = x % TG;
        sm[x] = k < nt ? M[i * J.ntgt + t0 + k] : (W)0;
    }
    if (threadIdx.x < TG) {
        const int k = threadIdx.x < nt ? threadIdx.x : 0;
        stp[threadIdx.x] = (uint32_t)tprime[t0 + k];
        str[threadIdx.x] = (uint32_t)trow[t0 + k];
        if (fast) {
            const uint32_t p = (uint32_t)pr[stp[threadIdx.x]].p;
            const uint32_t m = 0xFFFFFFFFu / p;  // floor(2^32 / p): p odd
            const uint32_t R = 0u - m * p;       // 2^32 mod p
            sp_[threadIdx.x] = p;
            sm32[threadIdx.x] = m;
            sR[threadIdx.x] = R;
            sRs[threadIdx.x] = (uint32_t)(((uint64_t)R << 32) / p);
        }
    }
    __syncthreads();
    const int idx = blockIdx.x * BX + threadIdx.x;
    W y[16];
    double v = 0.0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i < ns) {
            const PrimeConst& q = pr[sprime[i]];
            const W x = src[(size_t)(J.src_off + srow[i]) * N + idx];
            y[i] = J.prescaled ? x : MM::mul(x, pre[i], q);
        } else
            y[i] = 0;
    }
    uint64_t u = 0;
    const W* bmod = nullptr;
    if (J.exact) {
        const W* rbw = M + ns * J.ntgt;
        while ((uintptr_t)rbw & 7) ++rbw;  // 8-byte aligned doubles
        const double* rb = (const double*)rbw;
#pragma unroll
        for (int i = 0; i < 16; ++i)
            if (i < ns) v = __fma_rn((double)y[i], rb[i], v);
        u = (uint64_t)floor(v + 0.5);
        bmod = (const W*)(rb + ns);
    }
    // 4 targets per step: one vector shared load per source (matrix rows are TG wide, zero past nt)
    for (int k = 0; k < nt; k += 4) {
        Wd a0 = 0, a1 = 0, a2 = 0, a3 = 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (i < ns) {
                const V4<W> m = load4<W>(sm + i * TG + k);
                const Wd yi = y[i];
                a0 += yi * m.v[0];
                a1 += yi * m.v[1];
                a2 += yi * m.v[2];
                a3 += yi * m.v[3];
            }
        }
        const Wd acc[4] = {a0, a1, a2, a3};
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
            if (k + q4 < nt) {
                if constexpr (sizeof(W) == 4) {
                    if (fast) {
                        const uint32_t p = sp_[k + q4];
                        const uint32_t hi = (uint32_t)(acc[q4] >> 32), lo = (uint32_t)acc[q4];
                        uint32_t a = hi * sR[k + q4] - __umulhi(hi, sRs[k + q4]) * p;  // hi R mod p, in [0, 2p)
                        a = a >= p ? a - p : a;
                        uint32_t b = lo - __umulhi(lo, sm32[k + q4]) * p;  // lo mod p, in [0, 2p)
                        b = b >= p ? b - p : b;
                        a += b;
                        dst[(size_t)(J.dst_off + str[k + q4]) * N + idx] = (W)(a >= p ? a - p : a);
                        continue;
                    }
                }
                const PrimeConst& q = pr[stp[k + q4]];
                const W p = (W)q.p;
                W r = MM::reduce(acc[q4], q);
                if (J.exact) {
                    const W sv = MM::reduce((Wd)u * bmod[t0 + k + q4], q);
                    r = sub<W>(r, sv, p);
                }
                dst[(size_t)(J.dst_off + str[k + q4]) * N + idx] = r;
            }
        }
    }
}

template <class W, class MM>
int launch(const BconvArgs* a, cudaStream_t s) {
    if (a->njobs == 0) return 0;
    dim3 grid(N / BX, (a->max_tgt + TG - 1) / TG, a->njobs);
    bconv_kernel<W, MM><<<grid, BX, 0, s>>>((const W*)a->src, (W*)a->dst, a->jobs, (const W*)a->pool, a->pr);
    return (int)cudaGetLastError();
}
}  // namespace bconv
}  // namespace pc
