// CCMM: FIDESlib Ccmm.cu's kernels (CKL ePrint 2025/1957: ring switch 2^16 -> 2^13 into 128 column
// ciphertexts, TWEAK / CMT, the PPMM over R_64 as balanced int8 digit planes for the exact int8 x int8 -> int32
// batched GEMM, recombination, relinearization, ring switch back) behind one launcher `ccmm` dispatching on
// CcmmArgs::op (semantics per op in args.cuh). Everything but the ciphertext-facing ops (gather / scatter / the
// staging reads of ring_in, ks_digits and ring_out, which use the instantiation's word W) is u32 over the CCMM's
// primes {q0..q3, P} (all < 2^31): products are reduced with a 64-bit Barrett step (mu = floor(2^64 / p)).
#pragma once
#include "arith.cuh"

namespace pc {
namespace ccmm {
constexpr int BX = 256;
constexpr int NB = N;
constexpr int NS = 1 << 13, D = 128, KY = NS / D, SW = NB / NS, NBIG = D / SW, NQ = 4, NP = 5;
constexpr size_t COLS = (size_t)2 * NQ * D * NS;
constexpr int SMEM = 8192;   // words of one shared-memory NTT block (a whole ring-2^13 polynomial)
constexpr int SMEM_TH = 512;
static_assert(NB == 65536, "the CCMM is written for N = 2^16");

__device__ __forceinline__ uint32_t mulm(uint32_t a, uint32_t b, uint32_t p, uint64_t mu) {
    const uint64_t x = (uint64_t)a * b;
    const uint64_t q = __umul64hi(x, mu);
    uint64_t r = x - q * p;
    if (r >= p) r -= p;
    if (r >= p) r -= p;
    return (uint32_t)r;
}
__device__ __forceinline__ uint32_t addm(uint32_t a, uint32_t b, uint32_t p) {
    const uint32_t s = a + b;
    return s >= p ? s - p : s;
}
__device__ __forceinline__ uint32_t subm(uint32_t a, uint32_t b, uint32_t p) {
    return a >= b ? a - b : a + p - b;
}
__device__ __forceinline__ int64_t centered(uint64_t v, uint32_t p) {
    return v > p / 2 ? (int64_t)v - (int64_t)p : (int64_t)v;
}
__device__ __forceinline__ uint32_t smod(int64_t c, uint32_t p) {
    const int64_t r = c % (int64_t)p;
    return (uint32_t)(r < 0 ? r + p : r);
}

// ── ciphertext-facing ops (word W) ──────────────────────────────────────────────────────────────────────────────

template <class W>
__global__ void gather_k(CcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int row = blockIdx.y;  // part * 4 + limb
    if (k >= NB) return;
    const int part = row / 4, l = row % 4;
    const W* src = (const W*)A.ct.base + (size_t)A.i0 * A.ct.ct_stride + (size_t)part * A.ct.part_stride + (size_t)l * NB;
    ((W*)A.x)[((size_t)A.i1 * 8 + row) * NB + k] = src[k];
}
template <class W>
__global__ void scatter_k(CcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int row = blockIdx.y;
    if (k >= NB) return;
    const int part = row / 4, l = row % 4;
    W* dst = (W*)A.ct.base + (size_t)A.i0 * A.ct.ct_stride + (size_t)part * A.ct.part_stride + (size_t)l * NB;
    dst[k] = ((const W*)A.y)[((size_t)A.i1 * 8 + row) * NB + k];
}
// The ring-switch ops run on a batch of cnt = i1 (0: NBIG) ciphertexts, ciphertexts i0 .. i0 + cnt of the matrix
// (columns 8 i0 .. 8 (i0 + cnt)): staging slot / key-switch poly lc <-> ciphertext i0 + lc.
__device__ __forceinline__ int batch_cnt(const CcmmArgs& A) {
    return A.i1 > 0 ? A.i1 : NBIG;
}
// X[0][i][8c + j][z] = c0 + fwd-switched b, X[1] = fwd-switched a (FIDESlib ringIn's ksApply + splitBig)
template <class W>
__global__ void ring_in_k(CcmmArgs A) {
    const int z = blockIdx.x * BX + threadIdx.x;
    const int lcol = blockIdx.y, i = blockIdx.z;
    if (z >= NS) return;
    const int cnt = batch_cnt(A);
    const int c = lcol / SW, idx = lcol % SW + SW * z, col = A.i0 * SW + lcol;
    const W* st = (const W*)A.y;
    const uint32_t* res = (const uint32_t*)A.z;
    uint32_t* X = (uint32_t*)A.x;
    const uint32_t b0 = (uint32_t)st[(((size_t)c * 2 + 0) * 4 + i) * NB + idx];
    X[(((size_t)0 * NQ + i) * D + col) * NS + z] = addm(b0, res[(((size_t)1 * NQ + i) * cnt + c) * NB + idx], A.p[i]);
    X[(((size_t)1 * NQ + i) * D + col) * NS + z] = res[(((size_t)0 * NQ + i) * cnt + c) * NB + idx];
}
// bigA[i][lc][j + 8z] = X[1][i][8 (i0 + lc) + j][z]
__global__ void compose_a_k(CcmmArgs A) {
    const int idx = blockIdx.x * BX + threadIdx.x;
    const int lc = blockIdx.y, i = blockIdx.z;
    if (idx >= NB) return;
    const int cnt = batch_cnt(A), c = A.i0 + lc;
    const uint32_t* X = (const uint32_t*)A.y;
    ((uint32_t*)A.x)[((size_t)i * cnt + lc) * NB + idx] = X[(((size_t)1 * NQ + i) * D + c * SW + idx % SW) * NS + idx / SW];
}
// staging (coefficient form): part 0 = composed X[0] + back-switched b, part 1 = back-switched a
template <class W>
__global__ void ring_out_k(CcmmArgs A) {
    const int idx = blockIdx.x * BX + threadIdx.x;
    const int lc = blockIdx.y, i = blockIdx.z;
    if (idx >= NB) return;
    const int cnt = batch_cnt(A), c = A.i0 + lc;
    const uint32_t* X = (const uint32_t*)A.y;
    const uint32_t* res = (const uint32_t*)A.z;
    W* st = (W*)A.x;
    const uint32_t xb = X[(((size_t)0 * NQ + i) * D + c * SW + idx % SW) * NS + idx / SW];
    st[(((size_t)lc * 2 + 0) * 4 + i) * NB + idx] = (W)addm(xb, res[(((size_t)1 * NQ + i) * cnt + lc) * NB + idx], A.p[i]);
    st[(((size_t)lc * 2 + 1) * 4 + i) * NB + idx] = (W)res[(((size_t)0 * NQ + i) * cnt + lc) * NB + idx];
}

// ── key switching over {q0..q3, P} ──────────────────────────────────────────────────────────────────────────────

template <class T>
__global__ void ks_digits_k(CcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int q = blockIdx.y, polys = A.count, n = A.n;
    if (k >= n) return;
    const T* x = (const T*)A.y;
    uint32_t* Dg = (uint32_t*)A.x;
    for (int j = 0; j < NQ; ++j) {
        const uint32_t qj = A.p[j];
        const uint32_t v = (uint32_t)x[(size_t)j * A.s[0] + (size_t)q * A.s[1] + k];
        const int64_t c = centered(mulm(v, A.c[j], qj, A.mu[j]), qj);
        for (int pi = 0; pi < NP; ++pi) Dg[(((size_t)j * NP + pi) * polys + q) * n + k] = smod(c, A.p[pi]);
    }
}
__global__ void ks_mac_k(CcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int q = blockIdx.y, pi = blockIdx.z;
    const int n = A.n, polys = A.count;
    if (k >= n) return;
    const uint32_t p = A.p[pi];
    const uint64_t mu = A.mu[pi];
    const uint32_t* Dg = (const uint32_t*)A.y;
    const uint32_t* key = (const uint32_t*)A.z + (size_t)q * A.s[0];
    uint64_t a = 0, b = 0;   // < 4 p < 2^33
    for (int j = 0; j < NQ; ++j) {
        const uint32_t d = Dg[(((size_t)j * NP + pi) * polys + q) * n + k];
        const size_t kb = (((size_t)j * NP + pi) * 2) * n + k;
        a += mulm(d, key[kb], p, mu);
        b += mulm(d, key[kb + n], p, mu);
    }
    uint32_t* acc = (uint32_t*)A.x;
    acc[(((size_t)0 * NP + pi) * polys + q) * n + k] = (uint32_t)(a % p);
    acc[(((size_t)1 * NP + pi) * polys + q) * n + k] = (uint32_t)(b % p);
}
__global__ void mod_down_k(CcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int q = blockIdx.y, ab = blockIdx.z;
    const int n = A.n, polys = A.count;
    if (k >= n) return;
    const uint32_t* acc = (const uint32_t*)A.y;
    const int64_t c = centered(acc[(((size_t)ab * NP + NQ) * polys + q) * n + k], A.p[NQ]);
    for (int i = 0; i < NQ; ++i) {
        const uint32_t qi = A.p[i];
        const uint32_t v = acc[(((size_t)ab * NP + i) * polys + q) * n + k];
        ((uint32_t*)A.x)[(((size_t)ab * NQ + i) * polys + q) * n + k] = mulm(subm(v, smod(c, qi), qi), A.c[i], qi, A.mu[i]);
    }
}

// ── negacyclic NTT over the CCMM primes: poly k uses prime (k / i0) % i1 ───────────────────────────────────────
// Tables (the host NttTab order): tw[prime * n + m + i] = psi^brev(m + i) (forward) / psi^-brev(m + i) (inverse).

__global__ void ntt_stage_k(CcmmArgs A, int m, int t) {
    const int b = blockIdx.x * BX + threadIdx.x;
    const int n = A.n;
    if (b >= n / 2) return;
    const int poly = blockIdx.y;
    const int pi = (poly / A.i0) % A.i1;
    const uint32_t p = A.p[pi];
    const int i = b / t, j = 2 * i * t + (b % t);
    uint32_t* a = (uint32_t*)A.x + (size_t)poly * n;
    const uint32_t w = ((const uint32_t*)A.tw)[(size_t)pi * n + m + i];
    const uint32_t u = a[j], v = mulm(a[j + t], w, p, A.mu[pi]);
    a[j] = addm(u, v, p);
    a[j + t] = subm(u, v, p);
}
__global__ void intt_stage_k(CcmmArgs A, int m, int t) {
    const int b = blockIdx.x * BX + threadIdx.x;
    const int n = A.n;
    if (b >= n / 2) return;
    const int poly = blockIdx.y;
    const int pi = (poly / A.i0) % A.i1;
    const uint32_t p = A.p[pi];
    const int i = b / t, j = 2 * i * t + (b % t);
    uint32_t* a = (uint32_t*)A.x + (size_t)poly * n;
    const uint32_t w = ((const uint32_t*)A.tw)[(size_t)pi * n + m + i];
    const uint32_t u = a[j], v = a[j + t];
    a[j] = addm(u, v, p);
    a[j + t] = mulm(subm(u, v, p), w, p, A.mu[pi]);
}
__global__ void scale_k(CcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int n = A.n;
    if (k >= n) return;
    const int poly = blockIdx.y;
    const int pi = (poly / A.i0) % A.i1;
    uint32_t* a = (uint32_t*)A.x + (size_t)poly * n;
    a[k] = mulm(a[k], A.c[pi], A.p[pi], A.mu[pi]);
}
// The stages whose butterfly groups fit one SMEM-word block, in shared memory. L = min(n, SMEM) words per poly
// piece, S = n / L sub-blocks per poly (n > SMEM), PP = SMEM / L polys per block (n < SMEM; they share a prime:
// the launcher checks i0 % PP == 0). Forward: m = S .. n/2; inverse: m = n/2 .. S, then (S == 1) the n^-1 scaling.
__global__ void ntt_smem_k(CcmmArgs A, int inverse) {
    __shared__ uint32_t s[SMEM];
    const int n = A.n;
    const int L = n < SMEM ? n : SMEM, S = n / L, PP = SMEM / L;
    const size_t blk = blockIdx.x;
    size_t poly0;
    int sb;
    if (S > 1) {
        poly0 = blk / S;
        sb = (int)(blk % S);
    } else {
        poly0 = blk * PP;
        sb = 0;
    }
    const int pi = (int)((poly0 / A.i0) % A.i1);
    const uint32_t p = A.p[pi];
    const uint64_t mu = A.mu[pi];
    uint32_t* g = (uint32_t*)A.x + poly0 * n + (size_t)sb * L;
    const uint32_t* w = (const uint32_t*)A.tw + (size_t)pi * n;
    for (int k = threadIdx.x; k < SMEM; k += blockDim.x) s[k] = g[k];
    __syncthreads();
    const int half = L / 2;
    if (!inverse) {
        for (int m = S, t = half; t >= 1; m *= 2, t /= 2) {
            const int gpb = m / S;
            for (int b = threadIdx.x; b < SMEM / 2; b += blockDim.x) {
                const int lp = b / half, bl = b % half;
                const int il = bl / t, j = lp * L + 2 * il * t + (bl % t);
                const uint32_t u = s[j], v = mulm(s[j + t], w[m + sb * gpb + il], p, mu);
                s[j] = addm(u, v, p);
                s[j + t] = subm(u, v, p);
            }
            __syncthreads();
        }
    } else {
        for (int m = n / 2, t = 1; t <= half; m /= 2, t *= 2) {
            const int gpb = m / S;
            for (int b = threadIdx.x; b < SMEM / 2; b += blockDim.x) {
                const int lp = b / half, bl = b % half;
                const int il = bl / t, j = lp * L + 2 * il * t + (bl % t);
                const uint32_t u = s[j], v = s[j + t];
                s[j] = addm(u, v, p);
                s[j + t] = mulm(subm(u, v, p), w[m + sb * gpb + il], p, mu);
            }
            __syncthreads();
        }
    }
    const bool sc = inverse && S == 1;
    for (int k = threadIdx.x; k < SMEM; k += blockDim.x) g[k] = sc ? mulm(s[k], A.c[pi], p, mu) : s[k];
}

// ── CMT pieces (CKL Alg. 2, 3) over the columns X [2][4][D][NS] ─────────────────────────────────────────────────

__device__ __forceinline__ uint32_t mono_at(const uint32_t* v, int k, int64_t e, uint32_t p) {
    e %= 2 * NS;
    if (e < 0) e += 2 * NS;
    bool neg = false;
    if (e >= NS) {
        e -= NS;
        neg = true;
    }
    int src = k - (int)e;
    if (src < 0) {
        src += NS;
        neg = !neg;
    }
    const uint32_t x = v[src];
    return neg ? subm(0, x, p) : x;
}
__global__ void mono_k(CcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int col = blockIdx.y, pp = blockIdx.z;
    if (k >= NS) return;
    const size_t base = ((size_t)pp * D + col) * NS;
    ((uint32_t*)A.x)[base + k] = mono_at((const uint32_t*)A.y + base, k, (int64_t)A.i0 * col, A.p[pp % NQ]);
}
__global__ void bitrev_k(CcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int col = blockIdx.y, pp = blockIdx.z;
    if (k >= NS) return;
    const int r = __brev(col) >> (32 - 7);
    ((uint32_t*)A.x)[((size_t)pp * D + col) * NS + k] = ((const uint32_t*)A.y)[((size_t)pp * D + r) * NS + k];
}
__global__ void tweak_k(CcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int pair = blockIdx.y, pp = blockIdx.z;
    if (k >= NS) return;
    const uint32_t p = A.p[pp % NQ];
    const int h = A.i0, kk = A.i1, sgn = A.i2;
    const int g = pair / h, j = pair % h;
    const int i0 = g * 2 * h + j, i1 = i0 + h;
    const uint32_t* in = (const uint32_t*)A.y;
    uint32_t* out = (uint32_t*)A.x;
    const size_t b0 = ((size_t)pp * D + i0) * NS, b1 = ((size_t)pp * D + i1) * NS;
    const uint32_t u = in[b0 + k];
    const uint32_t v = mono_at(in + b1, k, (int64_t)2 * kk * j * sgn, p);
    out[b0 + k] = addm(u, v, p);
    out[b1 + k] = subm(u, v, p);
}
__global__ void scramble_k(CcmmArgs A) {
    const int u = blockIdx.x * BX + threadIdx.x;
    const int t = blockIdx.y, pp = blockIdx.z;
    if (u >= NS) return;
    const int qi = pp % NQ;
    const uint32_t p = A.p[qi];
    const int h = 2 * KY * t + 1;
    const int ts = ((const int32_t*)A.z)[t];
    const uint32_t x = mulm(((const uint32_t*)A.y)[((size_t)pp * D + ts) * NS + u], A.c[qi], p, A.mu[qi]);
    int64_t pos = ((int64_t)u * h) % (2 * NS);
    bool neg = false;
    if (pos >= NS) {
        pos -= NS;
        neg = true;
    }
    ((uint32_t*)A.x)[((size_t)pp * D + t) * NS + pos] = neg ? subm(0, x, p) : x;
}
__global__ void ks_apply_k(CcmmArgs A) {
    const size_t k = blockIdx.x * (size_t)BX + threadIdx.x;
    const size_t per = (size_t)A.s[0];
    if (k >= NQ * per) return;
    const uint32_t* res = (const uint32_t*)A.y;
    uint32_t* x = (uint32_t*)A.x;
    x[k] = addm(x[k], res[NQ * per + k], A.p[k / per]);
    x[NQ * per + k] = res[k];
}
__global__ void add_k(CcmmArgs A) {
    const size_t k = blockIdx.x * (size_t)BX + threadIdx.x;
    const size_t per = (size_t)A.s[0];
    if (k >= NQ * per) return;
    uint32_t* x = (uint32_t*)A.x;
    x[k] = addm(x[k], ((const uint32_t*)A.y)[k], A.p[k / per]);
}

// ── PPMM over R_64 ──────────────────────────────────────────────────────────────────────────────────────────────

// E[pc][i][u] = X[pc][i + D u]  (pc = (part, prime, col))
__global__ void gather_u_k(CcmmArgs A) {
    const size_t k = blockIdx.x * (size_t)BX + threadIdx.x;
    if (k >= COLS) return;
    const int u = (int)(k % KY);
    const size_t rest = k / KY;
    const int i = (int)(rest % D);
    const size_t pc = rest / D;
    ((uint32_t*)A.x)[k] = ((const uint32_t*)A.y)[pc * NS + i + (size_t)D * u];
}
// d[dg][u][r][t] = digit dg of centered E[r / D][prime][t][r % D][u]: both GEMM operands K-contiguous (col-TN)
__global__ void digits_k(CcmmArgs A) {
    const size_t k = blockIdx.x * (size_t)BX + threadIdx.x;
    const size_t per = (size_t)2 * D * D * KY;
    if (k >= per) return;
    const int prime = A.i0;
    const uint32_t p = A.p[prime];
    const int t = (int)(k % D);
    const int r = (int)((k / D) % (2 * D));
    const int u = (int)(k / ((size_t)D * 2 * D));
    const int part = r / D, i = r % D;
    int64_t v = centered(((const uint32_t*)A.y)[((((size_t)part * NQ + prime) * D + t) * D + i) * KY + u], p);
    int8_t* dst = (int8_t*)A.x;
    for (int dg = 0; dg < 4; ++dg) {
        int64_t q = ((v % 256) + 256) % 256;
        if (q > 127) q -= 256;
        dst[(size_t)dg * per + k] = (int8_t)q;
        v = (v - q) / 256;
    }
}
// Cn[prime][r][c][u] = sum_e G_e[u][r][c] 256^e mod p, rows r = i1 .. i1 + nr (nr = i2, 0: all 2D)
__global__ void recombine_k(CcmmArgs A) {
    const size_t k = blockIdx.x * (size_t)BX + threadIdx.x;
    const size_t gper = (size_t)KY * 2 * D * 2 * D;
    const int nr = A.i2 > 0 ? A.i2 : 2 * D;
    if (k >= (size_t)KY * nr * 2 * D) return;
    const int prime = A.i0;
    const int64_t p = A.p[prime];
    const int c = (int)(k % (2 * D));
    const int r = A.i1 + (int)((k / (2 * D)) % nr);
    const int u = (int)(k / ((size_t)2 * D * nr));
    const size_t g = ((size_t)u * 2 * D + r) * 2 * D + c;
    const int32_t* G = (const int32_t*)A.y;
    int64_t v = 0;
    for (int e = 6; e >= 0; --e) v = (v * 256 + G[(size_t)e * gper + g]) % p;
    if (v < 0) v += p;
    ((uint32_t*)A.x)[(((size_t)prime * 2 * D + r) * 2 * D + c) * KY + u] = (uint32_t)v;
}
// X[part][prime][j][i + D u] = Cn[prime][rowOff + j][part D + i][u]
__global__ void scatter_rows_k(CcmmArgs A) {
    const size_t k = blockIdx.x * (size_t)BX + threadIdx.x;
    if (k >= COLS) return;
    const int coef = (int)(k % NS);
    const size_t rest = k / NS;
    const int j = (int)(rest % D);
    const int pp = (int)(rest / D);
    const int part = pp / NQ, prime = pp % NQ;
    const int i = coef % D, u = coef / D;
    ((uint32_t*)A.x)[k] = ((const uint32_t*)A.y)[(((size_t)prime * 2 * D + A.i0 + j) * 2 * D + part * D + i) * KY + u];
}

inline unsigned blocks(size_t n) {
    return (unsigned)((n + BX - 1) / BX);
}

template <class W>
int launch(const CcmmArgs* a, cudaStream_t s) {
    const CcmmArgs A = *a;
    const int n = A.n;
    const dim3 cols(NS / BX, D, 2 * NQ);
    switch (A.op) {
        case CCMM_GATHER:
            gather_k<W><<<dim3(blocks(NB), 8), BX, 0, s>>>(A);
            break;
        case CCMM_SCATTER:
            scatter_k<W><<<dim3(blocks(NB), 8), BX, 0, s>>>(A);
            break;
        case CCMM_NTT:
        case CCMM_INTT: {
            if (n < 64 || n > NB || (n & (n - 1)) || A.count < 1 || A.i0 < 1 || A.i1 < 1 || A.i1 > NP) return -1;
            const int L = n < SMEM ? n : SMEM, S = n / L, PP = SMEM / L;
            if ((size_t)A.count * n % SMEM || A.i0 % PP) return -1;
            if (S > 1 && A.count > 65535) return -1;
            const unsigned nblk = (unsigned)((size_t)A.count * n / SMEM);
            const dim3 g(blocks(n / 2), A.count);
            if (A.op == CCMM_NTT) {
                for (int m = 1, t = n / 2; m < S; m *= 2, t /= 2) ntt_stage_k<<<g, BX, 0, s>>>(A, m, t);
                ntt_smem_k<<<nblk, SMEM_TH, 0, s>>>(A, 0);
            } else {
                ntt_smem_k<<<nblk, SMEM_TH, 0, s>>>(A, 1);
                if (S > 1) {
                    for (int m = S / 2, t = n / (2 * m); m >= 1; m /= 2, t *= 2) intt_stage_k<<<g, BX, 0, s>>>(A, m, t);
                    scale_k<<<dim3(blocks(n), A.count), BX, 0, s>>>(A);
                }
            }
            break;
        }
        case CCMM_KS_DIGITS:
            if (A.count < 1 || A.count > 65535) return -1;
            if (A.i4) ks_digits_k<W><<<dim3(blocks(n), A.count), BX, 0, s>>>(A);
            else ks_digits_k<uint32_t><<<dim3(blocks(n), A.count), BX, 0, s>>>(A);
            break;
        case CCMM_KS_MAC:
            if (A.count < 1 || A.count > 65535) return -1;
            ks_mac_k<<<dim3(blocks(n), A.count, NP), BX, 0, s>>>(A);
            break;
        case CCMM_MOD_DOWN:
            if (A.count < 1 || A.count > 65535) return -1;
            mod_down_k<<<dim3(blocks(n), A.count, 2), BX, 0, s>>>(A);
            break;
        case CCMM_RING_IN:
        case CCMM_COMPOSE_A:
        case CCMM_RING_OUT: {
            const int cnt = A.i1 > 0 ? A.i1 : NBIG;
            if (A.i0 < 0 || A.i1 < 0 || A.i0 + cnt > NBIG) return -1;
            if (A.op == CCMM_RING_IN) ring_in_k<W><<<dim3(NS / BX, SW * cnt, NQ), BX, 0, s>>>(A);
            else if (A.op == CCMM_COMPOSE_A) compose_a_k<<<dim3(blocks(NB), cnt, NQ), BX, 0, s>>>(A);
            else ring_out_k<W><<<dim3(blocks(NB), cnt, NQ), BX, 0, s>>>(A);
            break;
        }
        case CCMM_MONO:
            mono_k<<<cols, BX, 0, s>>>(A);
            break;
        case CCMM_BITREV:
            bitrev_k<<<cols, BX, 0, s>>>(A);
            break;
        case CCMM_TWEAK:
            if (A.i0 < 1 || A.i0 >= D) return -1;
            tweak_k<<<dim3(NS / BX, D / 2, 2 * NQ), BX, 0, s>>>(A);
            break;
        case CCMM_SCRAMBLE:
            scramble_k<<<cols, BX, 0, s>>>(A);
            break;
        case CCMM_KS_APPLY:
            ks_apply_k<<<blocks(NQ * (size_t)A.s[0]), BX, 0, s>>>(A);
            break;
        case CCMM_ADD:
            add_k<<<blocks(NQ * (size_t)A.s[0]), BX, 0, s>>>(A);
            break;
        case CCMM_GATHER_U:
            gather_u_k<<<blocks(COLS), BX, 0, s>>>(A);
            break;
        case CCMM_DIGITS:
            if (A.i0 < 0 || A.i0 >= NQ) return -1;
            digits_k<<<blocks((size_t)2 * D * D * KY), BX, 0, s>>>(A);
            break;
        case CCMM_RECOMBINE: {
            const int nr = A.i2 > 0 ? A.i2 : 2 * D;
            if (A.i0 < 0 || A.i0 >= NQ || A.i1 < 0 || A.i2 < 0 || A.i1 + nr > 2 * D) return -1;
            recombine_k<<<blocks((size_t)KY * nr * 2 * D), BX, 0, s>>>(A);
            break;
        }
        case CCMM_SCATTER_ROWS:
            scatter_rows_k<<<blocks(COLS), BX, 0, s>>>(A);
            break;
        default:
            return -2;
    }
    return (int)cudaGetLastError();
}
}  // namespace ccmm
}  // namespace pc
