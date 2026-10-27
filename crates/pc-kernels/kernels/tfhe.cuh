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

// acc (shared, both parts) += GGSW_key [x] (acc X^-ak - acc); dif, tb, o0, o1: shared scratch of n words each
__device__ void cmux(const TfheArgs& A, int key, uint32_t ak, uint64_t* acc0, uint64_t* acc1, uint64_t* dif, uint64_t* tb, uint64_t* o0, uint64_t* o1) {
    const int n = A.n, bl = (int)A.c[0], lv = (int)A.c[1];
    const uint64_t q = A.q;
    const int kp = (int)((2 * n - ak) % (2 * n));
    for (int o = threadIdx.x; o < n; o += blockDim.x) {
        o0[o] = 0;
        o1[o] = 0;
    }
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
        acc0[o] = addq(acc0[o], o0[o], q);
        acc1[o] = addq(acc1[o], o1[o], q);
    }
    __syncthreads();
}

// one CMux of every LWE of the batch (a block each): acc += GGSW_key [x] (acc X^-a~ - acc)
__global__ void __launch_bounds__(STEP_TH) step_k(TfheArgs A) {
    extern __shared__ uint64_t sm[];
    const int n = A.n, L = A.i1, key = A.i0;
    const uint32_t ak = ((const uint32_t*)A.z)[(size_t)blockIdx.x * (L + 1) + key];
    if (ak == 0) return;
    uint64_t* g = (uint64_t*)A.x + (size_t)blockIdx.x * 2 * n;
    for (int o = threadIdx.x; o < n; o += blockDim.x) {
        sm[o] = g[o];
        sm[n + o] = g[n + o];
    }
    __syncthreads();
    cmux(A, key, ak, sm, sm + n, sm + 2 * n, sm + 3 * n, sm + 4 * n, sm + 5 * n);
    for (int o = threadIdx.x; o < n; o += blockDim.x) {
        g[o] = sm[o];
        g[n + o] = sm[n + o];
    }
}

// the whole blind rotation of every LWE (a block each): the accumulator stays in shared memory across the i1 CMuxes
__global__ void __launch_bounds__(STEP_TH) rotate_k(TfheArgs A) {
    extern __shared__ uint64_t sm[];
    const int n = A.n, L = A.i1;
    const uint32_t* at = (const uint32_t*)A.z + (size_t)blockIdx.x * (L + 1);
    uint64_t* g = (uint64_t*)A.x + (size_t)blockIdx.x * 2 * n;
    for (int o = threadIdx.x; o < n; o += blockDim.x) {
        sm[o] = g[o];
        sm[n + o] = g[n + o];
    }
    __syncthreads();
    for (int key = 0; key < L; key++) {
        const uint32_t ak = at[key];
        if (ak) cmux(A, key, ak, sm, sm + n, sm + 2 * n, sm + 3 * n, sm + 4 * n, sm + 5 * n);
    }
    for (int o = threadIdx.x; o < n; o += blockDim.x) {
        g[o] = sm[o];
        g[n + o] = sm[n + o];
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

__global__ void ks_digits_k(TfheArgs A) {
    const int len = A.i1, lv = (int)A.c[1];
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)A.count * len) return;
    const size_t b = idx / len, i = idx % len;
    const uint64_t x = ((const uint32_t*)A.y)[b * (len + 1) + i];
    int8_t* out = (int8_t*)A.x + (b * len + i) * lv;
    for (int t = 0; t < lv; t++) out[t] = (int8_t)digit(x, t, (int)A.c[0], lv, A.m);
}

__global__ void ks_recombine_k(TfheArgs A) {
    const int cols = A.i2 + 1, planes = (int)A.c[2];
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)A.count * cols) return;
    const size_t b = idx / cols, o = idx % cols;
    const int32_t* g = (const int32_t*)A.y;
    const size_t plane = (size_t)cols * A.count;
    int64_t v = 0;
    for (int p = planes - 1; p >= 0; p--) v = v * 256 + g[p * plane + o * A.count + b];
    const int64_t q = (int64_t)A.q;
    v %= q;
    if (v < 0) v += q;
    uint64_t r = (uint64_t)v;
    if ((int)o == A.i2) r = addq(r, ((const uint32_t*)A.z)[b * (A.i1 + 1) + A.i1], A.q);
    ((uint32_t*)A.x)[b * cols + o] = (uint32_t)r;
}

// ── the bridge (u32, ring 2^16) ──────────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ uint32_t mulm32(uint32_t a, uint32_t b, uint64_t p, uint64_t mu) {
    const uint64_t x = (uint64_t)a * b;
    uint64_t r = x - __umul64hi(x, mu) * p;
    if (r >= p) r -= p;
    if (r >= p) r -= p;
    return (uint32_t)r;
}
__device__ __forceinline__ int64_t cent(uint64_t v, uint64_t p) {
    return v > p / 2 ? (int64_t)v - (int64_t)p : (int64_t)v;
}
__device__ __forceinline__ uint32_t smod32(int64_t v, uint64_t p) {
    const int64_t r = v % (int64_t)p;
    return (uint32_t)(r < 0 ? r + (int64_t)p : r);
}
// exact balanced digit t of the centered x (pc-tfhe Gadget::decompose, exact)
__device__ __forceinline__ int64_t digit_exact(int64_t v, int t, int bl, int lv) {
    const int64_t B = 1ll << bl;
    int64_t r = 0;
    for (int s = 0; s <= t; s++) {
        r = v & (B - 1);
        if (r >= B / 2) r -= B;
        v = (v - r) >> bl;
    }
    if (t == lv - 1) r += v << bl;
    return r;
}

__global__ void rescale_p1_k(TfheArgs A) {
    const int n = A.n;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int part = blockIdx.y;
    const uint32_t* y = (const uint32_t*)A.y + (size_t)part * 2 * n;
    const uint64_t q = A.q, p1 = A.c[1];
    const uint32_t d = smod32(cent(y[n + k], p1), q);
    ((uint32_t*)A.x)[(size_t)part * n + k] = mulm32((uint32_t)subq(y[k], d, q), (uint32_t)A.c[0], q, A.c[7]);
}

__global__ void rs_digits_k(TfheArgs A) {
    const int n = A.n, lv = (int)A.c[1];
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const uint64_t x = ((const uint32_t*)A.y)[k];
    for (int t = 0; t < lv; t++) ((uint32_t*)A.x)[(size_t)t * n + k] = smod32(digit(x, t, (int)A.c[0], lv, A.m), A.q);
}

__global__ void mac_k(TfheArgs A) {
    const int n = A.n;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int o = blockIdx.y;
    const uint32_t* y = (const uint32_t*)A.y;
    const uint32_t* z = (const uint32_t*)A.z;
    uint64_t acc = 0;
    for (int t = 0; t < A.count; t++) acc = addq(acc, mulm32(y[(size_t)t * n + k], z[((size_t)t * A.i1 + o) * n + k], A.q, A.c[7]), A.q);
    ((uint32_t*)A.x)[(size_t)o * n + k] = (uint32_t)acc;
}

__global__ void rs_extract_k(TfheArgs A) {
    const int n = A.n, dim = A.i1, S = A.i2;
    const int b = blockIdx.y;
    const int j = ((const int32_t*)A.z)[b];
    const uint32_t* c0 = (const uint32_t*)A.y;
    const uint32_t* c1 = c0 + n;
    uint32_t* out = (uint32_t*)A.x + (size_t)b * (dim + 1);
    for (int kk = blockIdx.x * blockDim.x + threadIdx.x; kk <= dim; kk += gridDim.x * blockDim.x) {
        if (kk == dim) {
            out[kk] = c0[j];
            continue;
        }
        const int sk = S * kk;
        if (sk <= j) {
            out[kk] = c1[j - sk];
        } else {
            const uint32_t v = c1[n + j - sk];
            out[kk] = v ? (uint32_t)(A.q - v) : 0;
        }
    }
}

__global__ void repack_digits_k(TfheArgs A) {
    const int n = A.n, dim = A.i1, lv = (int)A.c[1], bl = (int)A.c[0], kc = A.i3;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int kk = blockIdx.y, limb = blockIdx.z;
    const uint64_t p = limb ? A.c[2 + 2] : A.q;   // c[4] = p1
    const int row = ((const int32_t*)A.w)[j];
    int64_t v = 0;
    if (row >= 0) v = cent(((const uint32_t*)A.y)[(size_t)row * (dim + 1) + A.i0 + kk], A.q);
    uint32_t* out = (uint32_t*)A.x + ((size_t)(limb * kc + kk) * lv) * n;
    for (int t = 0; t < lv; t++) out[(size_t)t * n + j] = row >= 0 ? smod32(digit_exact(v, t, bl, lv), p) : 0;
}

__global__ void repack_mac_k(TfheArgs A) {
    const int n = A.n, lv = (int)A.c[1], kc = A.i3;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int limb = blockIdx.y;
    const uint64_t p = limb ? A.c[4] : A.q, mu = limb ? A.c[6] : A.c[7];
    const uint32_t* d = (const uint32_t*)A.y + (size_t)limb * kc * lv * n;
    const uint32_t* key = (const uint32_t*)A.z;
    uint32_t* acc0 = (uint32_t*)A.x + (size_t)limb * n;            // [part][limb][n]: part 0
    uint32_t* acc1 = (uint32_t*)A.x + (size_t)(2 + limb) * n;      // part 1
    uint64_t a0 = acc0[j], a1 = acc1[j];
    for (int kk = 0; kk < kc; kk++) {
        for (int t = 0; t < lv; t++) {
            const uint32_t dv = d[((size_t)kk * lv + t) * n + j];
            const uint32_t* k0 = key + ((((size_t)kk * lv + t) * 2 + limb) * 2) * n;
            a0 = addq(a0, mulm32(dv, k0[j], p, mu), p);
            a1 = addq(a1, mulm32(dv, k0[n + j], p, mu), p);
        }
    }
    acc0[j] = (uint32_t)a0;
    acc1[j] = (uint32_t)a1;
}

__global__ void repack_b_k(TfheArgs A) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= A.count) return;
    const int n = A.n, dim = A.i1;
    const int j = ((const int32_t*)A.z)[b];
    const int64_t v = cent(((const uint32_t*)A.y)[(size_t)b * (dim + 1) + dim], A.q);
    for (int limb = 0; limb < 2; limb++) {
        const uint64_t p = limb ? A.c[4] : A.q, mu = limb ? A.c[6] : A.c[7];
        const uint32_t p1 = limb ? 0 : (uint32_t)A.c[2];   // p1 mod p1 = 0
        uint32_t* c0 = (uint32_t*)A.x + (size_t)limb * n;   // part 0 of [part][limb][n]
        c0[j] = (uint32_t)addq(c0[j], mulm32(smod32(v, p), p1, p, mu), p);
    }
}

int launch(const TfheArgs* a, cudaStream_t s) {
    const TfheArgs A = *a;
    if (A.count < 1) return -1;
    if ((A.op <= PC_TFHE_PBS_EXTRACT || A.op == PC_TFHE_PBS_ROTATE) && (A.n < 64 || A.n > 2048 || (A.n & (A.n - 1)))) return -1;
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
        case PC_TFHE_PBS_ROTATE: {
            const int smem = 6 * A.n * (int)sizeof(uint64_t);
            static bool attr_r = false;
            if (!attr_r) {
                cudaFuncSetAttribute(rotate_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 6 * 2048 * (int)sizeof(uint64_t));
                attr_r = true;
            }
            rotate_k<<<A.count, A.n / 2 < STEP_TH ? A.n / 2 : STEP_TH, smem, s>>>(A);
            break;
        }
        case PC_TFHE_PBS_EXTRACT:
            if (A.count > 65535) return -1;
            extract_k<<<dim3((A.n + 256) / 256, A.count), 256, 0, s>>>(A);
            break;
        case PC_TFHE_KS_DIGITS: {
            const size_t tot = (size_t)A.count * A.i1;
            ks_digits_k<<<(unsigned)((tot + 255) / 256), 256, 0, s>>>(A);
            break;
        }
        case PC_TFHE_KS_RECOMBINE: {
            const size_t tot = (size_t)A.count * (A.i2 + 1);
            ks_recombine_k<<<(unsigned)((tot + 255) / 256), 256, 0, s>>>(A);
            break;
        }
        case PC_TFHE_RESCALE_P1:
            rescale_p1_k<<<dim3((A.n + 255) / 256, 2), 256, 0, s>>>(A);
            break;
        case PC_TFHE_RS_DIGITS:
            rs_digits_k<<<(A.n + 255) / 256, 256, 0, s>>>(A);
            break;
        case PC_TFHE_MAC:
            mac_k<<<dim3((A.n + 255) / 256, A.i1), 256, 0, s>>>(A);
            break;
        case PC_TFHE_RS_EXTRACT:
            if (A.count > 65535) return -1;
            rs_extract_k<<<dim3((A.i1 + 256) / 256, A.count), 256, 0, s>>>(A);
            break;
        case PC_TFHE_REPACK_DIGITS:
            repack_digits_k<<<dim3((A.n + 255) / 256, A.i3, 2), 256, 0, s>>>(A);
            break;
        case PC_TFHE_REPACK_MAC:
            repack_mac_k<<<dim3((A.n + 255) / 256, 2), 256, 0, s>>>(A);
            break;
        case PC_TFHE_REPACK_B:
            repack_b_k<<<(A.count + 255) / 256, 256, 0, s>>>(A);
            break;
        default:
            return -2;
    }
    return (int)cudaGetLastError();
}
}  // namespace tfhe
}  // namespace pc
