// CKKS <-> TFHE scheme switching: the programmable bootstrap's kernels (semantics per op in args.cuh TfheArgs). Every
// step mirrors pc-tfhe (the host reference) operation for operation, so outputs are bit-exact: the negacyclic NTT is
// pc-core's order (psi powers bit-reversed, Cooley-Tukey forward / Gentleman-Sande inverse), the gadget digits and the
// mod switches are the fixed-point roundings of pc-tfhe gadget.rs (`scale_round`). Products mod the bootstrapping
// prime q < 2^62 are Shoup products against precomputed companions (twiddles, n^-1 and the bootstrapping key).
#pragma once

namespace pc {
namespace tfhe {
constexpr int STEP_TH = 1024;   // n / 2 butterflies per stage for n = 2048

__device__ __forceinline__ uint64_t shoup(uint64_t a, uint64_t w, uint64_t ws, uint64_t q) {
    const uint64_t r = a * w - __umul64hi(a, ws) * q;
    return r >= q ? r - q : r;
}
__device__ __forceinline__ uint64_t addq(uint64_t a, uint64_t b, uint64_t q) {
    const uint64_t s = a + b;
    return s >= q ? s - q : s;
}
__device__ __forceinline__ uint64_t subq(uint64_t a, uint64_t b, uint64_t q) {
    return a >= b ? a - b : a + q - b;
}
// round(x num / den) = (x m + 2^63) >> 64, m = floor(2^64 num / den)
__device__ __forceinline__ uint64_t scale_round(uint64_t x, uint64_t m) {
    const uint64_t lo = x * m;
    const uint64_t lo2 = lo + (1ull << 63);
    return __umul64hi(x, m) + (lo2 < lo ? 1 : 0);
}
// digit t of the approximate balanced decomposition (pc-tfhe Gadget::decompose)
__device__ __forceinline__ int64_t digit(uint64_t x, int t, int bl, int lv, uint64_t m) {
    const int bits = bl * lv;
    int64_t v = (int64_t)scale_round(x, m);
    if (v >= (1ll << (bits - 1))) v -= 1ll << bits;
    const int64_t B = 1ll << bl;
    int64_t r = 0;
    for (int s = 0; s <= t; s++) {
        r = v & (B - 1);
        if (r >= B / 2) r -= B;
        v = (v - r) >> bl;
    }
    if (t == lv - 1) r += v << bl;   // the carry of the edge value folds into the top digit
    return r;
}
// out[o] = (a X^kp)[o] (pc-tfhe rlwe::monomial_mul, kp in [0, 2n)), read as a gather
__device__ __forceinline__ uint64_t mono(const uint64_t* a, int n, int kp, int o, uint64_t q) {
    int i = o - kp;
    if (i >= 0) return a[i];
    i += n;
    if (i >= 0) return a[i] ? q - a[i] : 0;
    return a[i + n];
}

__device__ void ntt_fwd(uint64_t* a, int n, const uint64_t* psi, const uint64_t* psis, uint64_t q) {
    for (int m = 1, t = n / 2; m < n; m *= 2, t /= 2) {
        for (int b = threadIdx.x; b < n / 2; b += blockDim.x) {
            const int i = b / t, j = 2 * i * t + b % t;
            const uint64_t u = a[j], v = shoup(a[j + t], psi[m + i], psis[m + i], q);
            a[j] = addq(u, v, q);
            a[j + t] = subq(u, v, q);
        }
        __syncthreads();
    }
}
__device__ void ntt_inv(uint64_t* a, int n, const uint64_t* ipsi, const uint64_t* ipsis, uint64_t ninv, uint64_t ninvs, uint64_t q) {
    for (int m = n / 2, t = 1; m >= 1; m /= 2, t *= 2) {
        for (int b = threadIdx.x; b < n / 2; b += blockDim.x) {
            const int i = b / t, j = 2 * i * t + b % t;
            const uint64_t u = a[j], v = a[j + t];
            a[j] = addq(u, v, q);
            a[j + t] = shoup(subq(u, v, q), ipsi[m + i], ipsis[m + i], q);
        }
        __syncthreads();
    }
    for (int o = threadIdx.x; o < n; o += blockDim.x) a[o] = shoup(a[o], ninv, ninvs, q);
    __syncthreads();
}

__global__ void ms_k(TfheArgs A) {
    const int len = A.i1 + 1;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)A.count * len) return;
    const uint64_t two_n = 2ull * A.n;
    ((uint32_t*)A.x)[i] = (uint32_t)(scale_round(((const uint32_t*)A.y)[i], A.m) % two_n);
}

__global__ void init_k(TfheArgs A) {
    const int n = A.n;
    const uint32_t bt = ((const uint32_t*)A.z)[(size_t)blockIdx.y * (A.i1 + 1) + A.i1];
    const int kp = (int)((2 * n - bt) % (2 * n));
    uint64_t* acc = (uint64_t*)A.x + (size_t)blockIdx.y * 2 * n;
    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < n; o += gridDim.x * blockDim.x) {
        acc[o] = mono((const uint64_t*)A.y, n, kp, o, A.q);
        acc[n + o] = 0;
    }
}

// one CMux of every LWE of the batch (a block each): acc += GGSW_key [x] (acc X^-a~ - acc)
__global__ void __launch_bounds__(STEP_TH) step_k(TfheArgs A) {
    extern __shared__ uint64_t sm[];
    const int n = A.n, L = A.i1, key = A.i0, bl = (int)A.c[0], lv = (int)A.c[1];
    const uint64_t q = A.q;
    const uint32_t ak = ((const uint32_t*)A.z)[(size_t)blockIdx.x * (L + 1) + key];
    if (ak == 0) return;
    const int kp = (int)((2 * n - ak) % (2 * n));
    uint64_t *acc0 = sm, *acc1 = sm + n, *dif = sm + 2 * n, *tb = sm + 3 * n, *o0 = sm + 4 * n, *o1 = sm + 5 * n;
    uint64_t* g = (uint64_t*)A.x + (size_t)blockIdx.x * 2 * n;
    for (int o = threadIdx.x; o < n; o += blockDim.x) {
        acc0[o] = g[o];
        acc1[o] = g[n + o];
        o0[o] = 0;
        o1[o] = 0;
    }
    __syncthreads();
    const uint64_t* psi = (const uint64_t*)A.tw;
    const uint64_t* psis = (const uint64_t*)A.tws;
    const size_t kstride = (size_t)2 * lv * 2 * n;
    const uint64_t* bsk = (const uint64_t*)A.y + key * kstride;
    const uint64_t* bsks = (const uint64_t*)A.w + key * kstride;
    for (int c = 0; c < 2; c++) {
        const uint64_t* src = c ? acc1 : acc0;
        for (int o = threadIdx.x; o < n; o += blockDim.x) dif[o] = subq(mono(src, n, kp, o, q), src[o], q);
        __syncthreads();
        for (int t = 0; t < lv; t++) {
            for (int o = threadIdx.x; o < n; o += blockDim.x) {
                const int64_t d = digit(dif[o], t, bl, lv, A.m);
                tb[o] = d < 0 ? (uint64_t)((int64_t)q + d) : (uint64_t)d;
            }
            __syncthreads();
            ntt_fwd(tb, n, psi, psis, q);
            const uint64_t* r0 = bsk + ((size_t)(c * lv + t) * 2) * n;
            const uint64_t* r0s = bsks + ((size_t)(c * lv + t) * 2) * n;
            for (int o = threadIdx.x; o < n; o += blockDim.x) {
                o0[o] = addq(o0[o], shoup(tb[o], r0[o], r0s[o], q), q);
                o1[o] = addq(o1[o], shoup(tb[o], r0[n + o], r0s[n + o], q), q);
            }
            __syncthreads();
        }
    }
    ntt_inv(o0, n, psi + n, psis + n, A.c[2], A.c[3], q);
    ntt_inv(o1, n, psi + n, psis + n, A.c[2], A.c[3], q);
    for (int o = threadIdx.x; o < n; o += blockDim.x) {
        g[o] = addq(acc0[o], o0[o], q);
        g[n + o] = addq(acc1[o], o1[o], q);
    }
}

__global__ void extract_k(TfheArgs A) {
    const int n = A.n;
    const uint64_t q = A.q, to = A.c[1];
    const uint64_t* acc = (const uint64_t*)A.y + (size_t)blockIdx.y * 2 * n;
    uint32_t* out = (uint32_t*)A.x + (size_t)blockIdx.y * (n + 1);
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k <= n; k += gridDim.x * blockDim.x) {
        uint64_t v;
        if (k == n) {
            v = addq(acc[0], A.c[0], q);
        } else if (k == 0) {
            v = acc[n];
        } else {
            const uint64_t c1 = acc[n + n - k];
            v = c1 ? q - c1 : 0;
        }
        out[k] = (uint32_t)(scale_round(v, A.m) % to);
    }
}

int launch(const TfheArgs* a, cudaStream_t s) {
    const TfheArgs A = *a;
    if (A.n < 64 || A.n > 2048 || (A.n & (A.n - 1)) || A.count < 1) return -1;
    switch (A.op) {
        case PC_TFHE_PBS_MS: {
            const size_t tot = (size_t)A.count * (A.i1 + 1);
            ms_k<<<(unsigned)((tot + 255) / 256), 256, 0, s>>>(A);
            break;
        }
        case PC_TFHE_PBS_INIT:
            if (A.count > 65535) return -1;
            init_k<<<dim3((A.n + 255) / 256, A.count), 256, 0, s>>>(A);
            break;
        case PC_TFHE_PBS_STEP: {
            const int smem = 6 * A.n * (int)sizeof(uint64_t);
            static bool attr = false;
            if (!attr) {   // the opt-in for the largest ring (n = 2048: 96 KB), once
                cudaFuncSetAttribute(step_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 6 * 2048 * (int)sizeof(uint64_t));
                attr = true;
            }
            step_k<<<A.count, A.n / 2 < STEP_TH ? A.n / 2 : STEP_TH, smem, s>>>(A);
            break;
        }
        case PC_TFHE_PBS_EXTRACT:
            if (A.count > 65535) return -1;
            extract_k<<<dim3((A.n + 256) / 256, A.count), 256, 0, s>>>(A);
            break;
        default:
            return -2;
    }
    return (int)cudaGetLastError();
}
}  // namespace tfhe
}  // namespace pc
