// Batched fast base conversion, generic over (W, MM); layout and formula in args.cuh BconvArgs.
// The exact (HPS-corrected) jobs keep FIDESlib's FMA double estimate in source order; on u32 words its rounding u is
// taken from an integer estimate where that estimate is clear of the rounding boundary, else from the double one (the
// same u either way, `int_round`). A launch whose jobs are all exact with 1 to 4 or 13 to 15 sources (BconvArgs::kind,
// u32 words) runs a kernel of that body alone.
#pragma once
#include "arith.cuh"

namespace pc {
namespace bconv {
constexpr int BX = 128;  // coefficients per block
constexpr int TG = 32;   // targets per block (blockIdx.y)

// The block's shared tables, one set for every body: M (TG wide, zero past the block's targets), the targets' prime
// ids and rows, and (p, R, Rs, m) of each target (the split reduction's constants)
__shared__ __align__(32) unsigned char sh_m[16 * TG * 8];
__shared__ uint32_t stp[TG], str[TG];
__shared__ uint4 sk[TG];
// the exact jobs (u32 words): c_i = floor(2^59 / b_i) of each source (the job's pool)
__shared__ unsigned long long sh_rc[16];

// u = floor(v + 0.5), v = the source-order FMA estimate sum_i y_i rb_i in doubles, from 64-bit integers where that is
// safe (u32 words, y_i < b_i): F = 2^58 + sum_i y_i c_i <= 2^59 (sum_i y_i / b_i + 1/2) < F + E, E = 1 + sum_i y_i,
// and v + 0.5 (rounded) is within 17 2^-50 < 2^-45 of that real number (each FMA rounds once, v < 16; rb_i once; the
// + 0.5 once more), so floor(F 2^-59) is the estimate's u whenever F mod 2^59 is at least 2^19 and below
// 2^59 - E - 2^19; otherwise (a chance of about ns 2^-31 per coefficient) the estimate itself runs
template <class W, int NY>
__device__ __forceinline__ uint64_t int_round(const W (&y)[NY], int ns, const double* __restrict__ rb) {
    uint64_t f = 1ull << 58, e = 1;
#pragma unroll
    for (int i = 0; i < NY; ++i) {
        if (i < ns) {
            f += (uint64_t)y[i] * sh_rc[i];
            e += (uint64_t)y[i];
        }
    }
    const uint64_t r = f & ((1ull << 59) - 1);
    if (r >= (1ull << 19) && r + e + (1ull << 19) < (1ull << 59)) return f >> 59;
    double v = 0.0;
#pragma unroll
    for (int i = 0; i < NY; ++i)
        if (i < ns) v = __fma_rn((double)y[i], rb[i], v);
    return (uint64_t)floor(v + 0.5);
}

// A centred job (J.exact == 2, JobSpec::centre): the fast lift x + u B has u = round(sum_i y_i / b_i), whose mean over
// uniform y_i is ns / 2; taking that mean for u (odd ns: (ns - 1) / 2 or (ns + 1) / 2 by a multiplicative hash of the
// coefficient index) leaves no bias, only the lift's zero-mean spread. The same u on every target and part of a
// coefficient (a function of its index alone).
__host__ __device__ __forceinline__ uint64_t centre_u(int ns, int idx) {
    return (uint64_t)(ns >> 1) + (uint64_t)((ns & 1) & (((uint32_t)idx * 0x9E3779B1u) >> 31));
}

// NS > 0: every job of the launch path has exactly NS sources (a compile-time count: no predicated source steps)
// EXM: 0 plain jobs, 1 exact jobs, 2 either (J.exact at run time)
template <class W, class MM, int NS, int EXM>
__device__ __forceinline__ void bconv_body(const BconvJob& J, const W* __restrict__ src, W* __restrict__ dst,
                                           const W* __restrict__ pool, const PrimeConst* __restrict__ pr) {
    W* const sm = (W*)sh_m;
    using Wd = WideOf<W>;
    constexpr int NY = NS > 0 ? NS : 16;
    // the job's targets split evenly over the gridDim.y blocks (a multiple of 4 each, at most TG): no light last block
    const int per = (((int)J.ntgt + (int)gridDim.y - 1) / (int)gridDim.y + 3) & ~3;
    const int t0 = blockIdx.y * per;
    if (t0 >= (int)J.ntgt) return;
    const int nt = min(per, (int)J.ntgt - t0);
    const int ns = NS > 0 ? NS : (int)J.nsrc;
    const W* P = pool + J.pool_off;
    const W* srow = P;
    const W* sprime = P + ns;
    const W* pre = P + 2 * ns;
    const W* trow = P + 3 * ns;
    const W* tprime = trow + J.ntgt;
    const W* M = tprime + J.ntgt;
    // exact jobs on u32 words: u by `int_round`
    const bool ir = sizeof(W) == 4 && (EXM == 1 || (EXM == 2 && J.exact));
    // J.split (u32 words, every target below 2^30): the sums are reduced as hi 2^32 + lo with 32-bit Shoup products
    // (R = 2^32 mod p, Rs = floor(R 2^32 / p), m = floor(2^32 / p)) from shared memory: the same canonical residue
    // (not the dispatch kernels' exact jobs: measured slower there)
    const bool fast = sizeof(W) == 4 && J.split && (EXM != 2 || !J.exact);
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
            sk[threadIdx.x] = make_uint4(p, R, (uint32_t)(((uint64_t)R << 32) / p), m);
        }
    }
    if (ir && threadIdx.x < ns) {
        // c_i after bmod (args.cuh), 8-byte aligned
        const W* rcw = M + ns * J.ntgt;
        while ((uintptr_t)rcw & 7) ++rcw;
        rcw += (8 / sizeof(W)) * ns + J.ntgt;
        while ((uintptr_t)rcw & 7) ++rcw;
        sh_rc[threadIdx.x] = ((const unsigned long long*)rcw)[threadIdx.x];
    }
    __syncthreads();
    // the predecessor's rows (the INTT's) are read from here on; the job tables above are setup constants
    pdl_wait();
    const int idx = blockIdx.x * BX + threadIdx.x;
    W y[NY];
    double v = 0.0;
#pragma unroll
    for (int i = 0; i < NY; ++i) {
        if (NS > 0 || i < ns) {
            const PrimeConst& q = pr[sprime[i]];
            const W x = src[(size_t)(J.src_off + srow[i]) * N + idx];
            y[i] = J.prescaled ? x : MM::mul(x, pre[i], q);
        } else
            y[i] = 0;
    }
    uint64_t u = 0;
    const W* bmod = nullptr;
    const bool ex = EXM == 1 || (EXM == 2 && J.exact);
    if (EXM != 0 && ex) {
        const W* rbw = M + ns * J.ntgt;
        while ((uintptr_t)rbw & 7) ++rbw;  // 8-byte aligned doubles
        const double* rb = (const double*)rbw;
        if (J.exact == 2) {
            u = centre_u(ns, idx);
        } else if (ir) {
            u = int_round<W, NY>(y, ns, rb);
        } else {
#pragma unroll
            for (int i = 0; i < NY; ++i)
                if (NS > 0 || i < ns) v = __fma_rn((double)y[i], rb[i], v);
            u = (uint64_t)floor(v + 0.5);
        }
        bmod = (const W*)(rb + ns);
    }
    // 4 targets per step: one vector shared load per source (matrix rows are TG wide, zero past nt)
    for (int k = 0; k < nt; k += 4) {
        Wd a0 = 0, a1 = 0, a2 = 0, a3 = 0;
#pragma unroll
        for (int i = 0; i < NY; ++i) {
            if (NS > 0 || i < ns) {
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
                        const uint4 c = sk[k + q4];  // p, R, Rs, m
                        const uint32_t p = c.x;
                        // x mod p for a 64-bit x as hi 2^32 + lo: [0, 4p) before the two min-subtracts (p < 2^30)
                        auto red = [&](uint64_t x) {
                            const uint32_t hi = (uint32_t)(x >> 32), lo = (uint32_t)x;
                            const uint32_t a = hi * c.y - __umulhi(hi, c.z) * p;  // hi R mod p, in [0, 2p)
                            const uint32_t b = lo - __umulhi(lo, c.w) * p;         // lo mod p, in [0, 2p)
                            uint32_t r = a + b;
                            r = min(r, r - 2 * p);
                            return min(r, r - p);
                        };
                        // exact: + u (p - B mod p) in the lazy sum (J.split bounds it, bconv.rs)
                        const uint32_t r = red(EXM != 0 && ex ? acc[q4] + u * (uint64_t)(p - (uint32_t)bmod[t0 + k + q4]) : acc[q4]);
                        dst[(size_t)(J.dst_off + str[k + q4]) * N + idx] = (W)r;
                        continue;
                    }
                }
                const PrimeConst& q = pr[stp[k + q4]];
                const W p = (W)q.p;
                W r = MM::reduce(acc[q4], q);
                if (EXM != 0 && ex) {
                    const W sv = MM::reduce((Wd)u * bmod[t0 + k + q4], q);
                    r = sub<W>(r, sv, p);
                }
                dst[(size_t)(J.dst_off + str[k + q4]) * N + idx] = r;
            }
        }
    }
}

// the body by job kind: exact (HPS) jobs, plain jobs of 13 sources (a full digit of the lean chain, the special set P,
// unrolled at compile time), other plain jobs
template <class W, class MM>
__device__ __forceinline__ void bconv_dispatch(const W* __restrict__ src, W* __restrict__ dst, const BconvJob* __restrict__ jobs,
                                               const W* __restrict__ pool, const PrimeConst* __restrict__ pr) {
    pdl_trigger();
    const BconvJob J = jobs[blockIdx.z];
    if (J.exact && J.nsrc == 15) bconv_body<W, MM, 15, 1>(J, src, dst, pool, pr);  // P and the two tops of the lean chain
    else if (!J.exact && J.nsrc == 13) bconv_body<W, MM, 13, 0>(J, src, dst, pool, pr);
    else bconv_body<W, MM, 0, 2>(J, src, dst, pool, pr);
}

// Two occupancies of one body: up to 48 registers (10 blocks per SM, the shorter per-thread chains) for a launch that
// fits in one wave there, 40 registers (12 blocks per SM) for a larger one, which then runs in fewer waves.
template <class W, class MM>
__global__ void __launch_bounds__(BX, 10) bconv_kernel(const W* __restrict__ src, W* __restrict__ dst, const BconvJob* __restrict__ jobs,
                                                   const W* __restrict__ pool, const PrimeConst* __restrict__ pr) {
    bconv_dispatch<W, MM>(src, dst, jobs, pool, pr);
}
template <class W, class MM>
__global__ void __launch_bounds__(BX, 12) bconv_kernel12(const W* __restrict__ src, W* __restrict__ dst, const BconvJob* __restrict__ jobs,
                                                         const W* __restrict__ pool, const PrimeConst* __restrict__ pr) {
    bconv_dispatch<W, MM>(src, dst, jobs, pool, pr);
}

// A launch of exact jobs of NS sources only (BconvArgs::kind = NS; u32 words): that body alone (a low-P key switch's
// ModUp of its live digit or ModDown from P': 1 to 4; an alternating user region's ModDown from P: 13; a fused
// relinearization-rescale's from P and its one or two dropped tops: 14, 15), at the two occupancies of the dispatch
// kernels
template <class W, class MM, int NS>
__global__ void __launch_bounds__(BX) bconv_exact_kernel(const W* __restrict__ src, W* __restrict__ dst, const BconvJob* __restrict__ jobs,
                                                         const W* __restrict__ pool, const PrimeConst* __restrict__ pr) {
    pdl_trigger();
    bconv_body<W, MM, NS, 1>(jobs[blockIdx.z], src, dst, pool, pr);
}
template <class W, class MM, int NS>
__global__ void __launch_bounds__(BX, 12) bconv_exact_kernel12(const W* __restrict__ src, W* __restrict__ dst, const BconvJob* __restrict__ jobs,
                                                               const W* __restrict__ pool, const PrimeConst* __restrict__ pr) {
    pdl_trigger();
    bconv_body<W, MM, NS, 1>(jobs[blockIdx.z], src, dst, pool, pr);
}

// Wide jobs (ABI 20): up to MW sources (u32 words: 48; u64: 16): each thread's first 16 y_i in registers, the rest
// in shared memory; otherwise the body above (same sums, same reductions, the exact estimate in source order).
template <class W>
__host__ __device__ constexpr int wide_max() { return sizeof(W) == 4 ? 48 : 16; }

template <class W, class MM>
__global__ void __launch_bounds__(BX) bconv_wide_kernel(const W* __restrict__ src, W* __restrict__ dst, const BconvJob* __restrict__ jobs,
                                                        const W* __restrict__ pool, const PrimeConst* __restrict__ pr) {
    using Wd = WideOf<W>;
    constexpr int MW = wide_max<W>();
    const BconvJob J = jobs[blockIdx.z];
    const int per = (((int)J.ntgt + (int)gridDim.y - 1) / (int)gridDim.y + 3) & ~3;
    const int t0 = blockIdx.y * per;
    if (t0 >= (int)J.ntgt) return;
    const int nt = min(per, (int)J.ntgt - t0);
    const int ns = min((int)J.nsrc, MW);
    const W* P = pool + J.pool_off;
    const W* srow = P;
    const W* sprime = P + J.nsrc;
    const W* pre = P + 2 * J.nsrc;
    const W* trow = P + 3 * J.nsrc;
    const W* tprime = trow + J.ntgt;
    const W* M = tprime + J.ntgt;
    __shared__ __align__(32) W sm[MW * TG];
    __shared__ W ys[(MW > 16 ? MW - 16 : 1) * BX];
    __shared__ uint32_t stp[TG], str[TG];
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
            const uint32_t m = 0xFFFFFFFFu / p;
            const uint32_t R = 0u - m * p;
            sp_[threadIdx.x] = p;
            sm32[threadIdx.x] = m;
            sR[threadIdx.x] = R;
            sRs[threadIdx.x] = (uint32_t)(((uint64_t)R << 32) / p);
        }
    }
    const int idx = blockIdx.x * BX + threadIdx.x;
    const double* rb = nullptr;
    const W* bmod = nullptr;
    if (J.exact) {
        const W* rbw = M + J.nsrc * J.ntgt;
        while ((uintptr_t)rbw & 7) ++rbw;
        rb = (const double*)rbw;
        bmod = (const W*)(rb + J.nsrc);
    }
    double v = 0.0;
    W yr[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i < ns) {
            const PrimeConst& q = pr[sprime[i]];
            const W x = src[(size_t)(J.src_off + srow[i]) * N + idx];
            yr[i] = J.prescaled ? x : MM::mul(x, pre[i], q);
            if (J.exact) v = __fma_rn((double)yr[i], rb[i], v);
        } else
            yr[i] = 0;
    }
    for (int i = 16; i < ns; ++i) {
        const PrimeConst& q = pr[sprime[i]];
        const W x = src[(size_t)(J.src_off + srow[i]) * N + idx];
        const W y = J.prescaled ? x : MM::mul(x, pre[i], q);
        ys[(i - 16) * BX + threadIdx.x] = y;
        if (J.exact) v = __fma_rn((double)y, rb[i], v);
    }
    const uint64_t u = J.exact == 2 ? centre_u(ns, idx) : J.exact ? (uint64_t)floor(v + 0.5) : 0;
    __syncthreads();
    for (int k = 0; k < nt; k += 4) {
        Wd a0 = 0, a1 = 0, a2 = 0, a3 = 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (i < ns) {
                const V4<W> m = load4<W>(sm + i * TG + k);
                const Wd yi = yr[i];
                a0 += yi * m.v[0];
                a1 += yi * m.v[1];
                a2 += yi * m.v[2];
                a3 += yi * m.v[3];
            }
        }
#pragma unroll 4
        for (int i = 16; i < ns; ++i) {
            const V4<W> m = load4<W>(sm + i * TG + k);
            const Wd yi = ys[(i - 16) * BX + threadIdx.x];
            a0 += yi * m.v[0];
            a1 += yi * m.v[1];
            a2 += yi * m.v[2];
            a3 += yi * m.v[3];
        }
        const Wd acc[4] = {a0, a1, a2, a3};
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
            if (k + q4 < nt) {
                if constexpr (sizeof(W) == 4) {
                    if (fast) {
                        const uint32_t p = sp_[k + q4];
                        const uint32_t hi = (uint32_t)(acc[q4] >> 32), lo = (uint32_t)acc[q4];
                        uint32_t a = hi * sR[k + q4] - __umulhi(hi, sRs[k + q4]) * p;
                        a = a >= p ? a - p : a;
                        uint32_t b = lo - __umulhi(lo, sm32[k + q4]) * p;
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
int launch_wide(const BconvArgs* a, cudaStream_t s) {
    if (a->njobs == 0) return 0;
    dim3 grid(N / BX, (a->max_tgt + TG - 1) / TG, a->njobs);
    bconv_wide_kernel<W, MM><<<grid, BX, 0, s>>>((const W*)a->src, (W*)a->dst, a->jobs, (const W*)a->pool, a->pr);
    return (int)cudaGetLastError();
}

// SMs of the current device (cached; a failed query selects the one-wave kernel everywhere)
inline int num_sms() {
    static int n = 0;
    if (!n) {
        int d = 0, v = 0;
        n = cudaGetDevice(&d) == cudaSuccess && cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, d) == cudaSuccess && v > 0 ? v : 1 << 20;
    }
    return n;
}

template <class W, class MM>
int launch(const BconvArgs* a, cudaStream_t s) {
    if (a->njobs == 0) return 0;
    dim3 grid(N / BX, (a->max_tgt + TG - 1) / TG, a->njobs);
    // a launch short of one wave at 12 blocks per SM: each job's targets over more blocks (down to 8 each), so the
    // wave fills and every thread's chain is shorter (the sources are read once more per extra block, from L2)
    const long wave = 12L * num_sms(), per_y = (long)grid.x * grid.z;
    const int most = (a->max_tgt + 7) / 8;
    if ((long)grid.y * per_y < wave && (int)grid.y < most) grid.y = (unsigned)max((long)grid.y, min((long)most, wave / per_y));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(BX);
    cfg.stream = s;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    const bool big = (long)grid.x * grid.y * grid.z > 10L * num_sms();
    const W* sp = (const W*)a->src;
    W* dp = (W*)a->dst;
    const W* pp = (const W*)a->pool;
    if constexpr (sizeof(W) == 4) {
        switch (a->kind) {
            case 13:
                if (big) cudaLaunchKernelEx(&cfg, bconv_exact_kernel12<W, MM, 13>, sp, dp, a->jobs, pp, a->pr);
                else cudaLaunchKernelEx(&cfg, bconv_exact_kernel<W, MM, 13>, sp, dp, a->jobs, pp, a->pr);
                return (int)cudaGetLastError();
            case 14:
                if (big) cudaLaunchKernelEx(&cfg, bconv_exact_kernel12<W, MM, 14>, sp, dp, a->jobs, pp, a->pr);
                else cudaLaunchKernelEx(&cfg, bconv_exact_kernel<W, MM, 14>, sp, dp, a->jobs, pp, a->pr);
                return (int)cudaGetLastError();
            case 15:
                if (big) cudaLaunchKernelEx(&cfg, bconv_exact_kernel12<W, MM, 15>, sp, dp, a->jobs, pp, a->pr);
                else cudaLaunchKernelEx(&cfg, bconv_exact_kernel<W, MM, 15>, sp, dp, a->jobs, pp, a->pr);
                return (int)cudaGetLastError();
            case 1:
                if (big) cudaLaunchKernelEx(&cfg, bconv_exact_kernel12<W, MM, 1>, sp, dp, a->jobs, pp, a->pr);
                else cudaLaunchKernelEx(&cfg, bconv_exact_kernel<W, MM, 1>, sp, dp, a->jobs, pp, a->pr);
                return (int)cudaGetLastError();
            case 2:
                if (big) cudaLaunchKernelEx(&cfg, bconv_exact_kernel12<W, MM, 2>, sp, dp, a->jobs, pp, a->pr);
                else cudaLaunchKernelEx(&cfg, bconv_exact_kernel<W, MM, 2>, sp, dp, a->jobs, pp, a->pr);
                return (int)cudaGetLastError();
            case 3:
                if (big) cudaLaunchKernelEx(&cfg, bconv_exact_kernel12<W, MM, 3>, sp, dp, a->jobs, pp, a->pr);
                else cudaLaunchKernelEx(&cfg, bconv_exact_kernel<W, MM, 3>, sp, dp, a->jobs, pp, a->pr);
                return (int)cudaGetLastError();
            case 4:
                if (big) cudaLaunchKernelEx(&cfg, bconv_exact_kernel12<W, MM, 4>, sp, dp, a->jobs, pp, a->pr);
                else cudaLaunchKernelEx(&cfg, bconv_exact_kernel<W, MM, 4>, sp, dp, a->jobs, pp, a->pr);
                return (int)cudaGetLastError();
            default:
                break;
        }
    }
    if (big)
        cudaLaunchKernelEx(&cfg, bconv_kernel12<W, MM>, sp, dp, a->jobs, pp, a->pr);
    else
        cudaLaunchKernelEx(&cfg, bconv_kernel<W, MM>, sp, dp, a->jobs, pp, a->pr);
    return (int)cudaGetLastError();
}
}  // namespace bconv
}  // namespace pc
