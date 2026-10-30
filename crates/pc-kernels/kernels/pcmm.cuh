// PCMM at the StC-first split: FIDESlib Pcmm.cu's kernels (ring switch 2^16 -> 2^12, MLWE rows,
// balanced int8 digit planes for the exact int8 x int8 -> int32 GEMM, recombination, pack back, compose, lift)
// behind one launcher `pcmm` dispatching on PcmmArgs::op (semantics per op in args.cuh). Everything but the
// ciphertext-facing ops (gather / scatter / in_rescale / lift, which read or write the instantiation's word W) is
// u32 over the PCMM's own primes {q0, q1, P} (all < 2^31): products are reduced with a 64-bit Barrett step
// (mu = floor(2^64 / p)), exact for every product of two residues.
#pragma once
#include "arith.cuh"

namespace pc {
namespace pcmm {
constexpr int BX = 256;
constexpr int SMEM_N = 4096;   // a whole ring-2^12 polynomial (and one 4096-block of a ring-2^16 one) in shared memory
constexpr int SMEM_TH = 512;

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
// centered residue of v mod p
__device__ __forceinline__ int64_t centered(uint64_t v, uint32_t p) {
    return v > p / 2 ? (int64_t)v - (int64_t)p : (int64_t)v;
}
// c mod p for a signed c
__device__ __forceinline__ uint32_t smod(int64_t c, uint32_t p) {
    const int64_t r = c % (int64_t)p;
    return (uint32_t)(r < 0 ? r + p : r);
}

// ── ciphertext-facing ops (word W) ──────────────────────────────────────────────────────────────────────────────

// The split's limbs (i4): 4 (q0..q3: the StC-first split of a composite chain) or 3 (q0..q2: a lean chain's); the
// staging keeps 4 rows per part either way.
__device__ __forceinline__ int split_limbs(const PcmmArgs& A) {
    return A.i4 ? A.i4 : 4;
}
// staging [part][4][N] <- ct member c (staging part 0 = c1 = "a", part 1 = c0 = "b", FIDESlib's x order)
template <class W>
__global__ void gather_k(PcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int ns = split_limbs(A);
    const int row = blockIdx.y;  // part * ns + limb
    if (k >= N) return;
    const int sp = row / ns, l = row % ns;
    const W* src = (const W*)A.ct.base + (size_t)A.i0 * A.ct.ct_stride + (size_t)(1 - sp) * A.ct.part_stride + (size_t)l * N;
    ((W*)A.x)[(size_t)(sp * 4 + l) * N + k] = src[k];
}
template <class W>
__global__ void scatter_k(PcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int ns = split_limbs(A);
    const int row = blockIdx.y;
    if (k >= N) return;
    const int sp = row / ns, l = row % ns;
    W* dst = (W*)A.ct.base + (size_t)A.i0 * A.ct.ct_stride + (size_t)(1 - sp) * A.ct.part_stride + (size_t)l * N;
    dst[k] = ((const W*)A.y)[(size_t)(sp * 4 + l) * N + k];
}
// exact rescale of the coefficient-form staging down to q0, q1 (FIDESlib Pcmm.cu rescaleTop: by q3 then q2 on 4 limbs,
// by q2 on 3): x[part][2][N] u32
template <class W>
__global__ void in_rescale_k(PcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int part = blockIdx.y;
    if (k >= N) return;
    const W* y = (const W*)A.y + (size_t)part * 4 * N;
    const int ns = split_limbs(A);
    uint32_t v[4];
    for (int i = 0; i < ns; ++i) v[i] = (uint32_t)y[(size_t)i * N + k];
    if (ns == 4) {
        const int64_t c3 = centered(v[3], A.p[3]);
        for (int i = 0; i < 3; ++i) v[i] = mulm(subm(v[i], smod(c3, A.p[i]), A.p[i]), A.c[i], A.p[i], A.mu[i]);
    }
    const int64_t c2 = centered(v[2], A.p[2]);
    uint32_t* x = (uint32_t*)A.x + (size_t)part * 2 * N;
    for (int i = 0; i < 2; ++i) x[(size_t)i * N + k] = mulm(subm(v[i], smod(c2, A.p[i]), A.p[i]), A.c[3 + i], A.p[i], A.mu[i]);
}
// one part of the output: limb i < 2 = f[i] * lift[i] mod q_i, the split's other limbs 0 (staging [4][N], coefficient
// form; grid rows = the split's limbs)
template <class W>
__global__ void lift_k(PcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int i = blockIdx.y;
    if (k >= N) return;
    const uint32_t* f = (const uint32_t*)A.y;
    ((W*)A.x)[(size_t)i * N + k] = i < 2 ? (W)mulm(f[(size_t)i * N + k], A.c[i], A.p[i], A.mu[i]) : (W)0;
}

// ── key switching over {q0, q1, P} ──────────────────────────────────────────────────────────────────────────────

// D[j][prime][poly][n] = centered(x[j][poly] * hat_j mod q_j) mod prime
__global__ void ks_digits_k(PcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int poly = blockIdx.y, polys = A.count, n = A.n;
    if (k >= n) return;
    const uint32_t* x = (const uint32_t*)A.y;
    uint32_t* D = (uint32_t*)A.x;
    for (int j = 0; j < 2; ++j) {
        const uint32_t q = A.p[j];
        const int64_t c = centered(mulm(x[((size_t)j * polys + poly) * n + k], A.c[j], q, A.mu[j]), q);
        for (int pi = 0; pi < 3; ++pi) D[(((size_t)j * 3 + pi) * polys + poly) * n + k] = smod(c, A.p[pi]);
    }
}

// acc[ab][prime][out][n] = sum_j sum_l D[j][prime][out nl + l][n] key[l][j][prime][ab][n]   (NTT form)
__global__ void ks_mac_k(PcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int o = blockIdx.y, pi = blockIdx.z;
    const int n = A.n, outs = A.i0, nl = A.i1;
    if (k >= n) return;
    const uint32_t p = A.p[pi];
    const uint64_t mu = A.mu[pi];
    const uint32_t* D = (const uint32_t*)A.y;
    const uint32_t* key = (const uint32_t*)A.z;
    const size_t polys = (size_t)outs * nl;
    uint64_t a = 0, b = 0;   // < 2 nl p < 2^40
    for (int j = 0; j < 2; ++j)
        for (int l = 0; l < nl; ++l) {
            const uint32_t d = D[(((size_t)j * 3 + pi) * polys + (size_t)o * nl + l) * n + k];
            const size_t kb = ((((size_t)l * 2 + j) * 3 + pi) * 2) * n + k;
            a += mulm(d, key[kb], p, mu);
            b += mulm(d, key[kb + n], p, mu);
        }
    uint32_t* acc = (uint32_t*)A.x;
    acc[(((size_t)0 * 3 + pi) * outs + o) * n + k] = (uint32_t)(a % p);
    acc[(((size_t)1 * 3 + pi) * outs + o) * n + k] = (uint32_t)(b % p);
}

// out[ab][i < 2][out][n] = (acc_i - centered(acc_P) mod q_i) P^-1 mod q_i
__global__ void mod_down_k(PcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int o = blockIdx.y, ab = blockIdx.z;
    const int n = A.n, outs = A.i0;
    if (k >= n) return;
    const uint32_t* acc = (const uint32_t*)A.y;
    const int64_t c = centered(acc[(((size_t)ab * 3 + 2) * outs + o) * n + k], A.p[2]);
    for (int i = 0; i < 2; ++i) {
        const uint32_t q = A.p[i];
        const uint32_t v = acc[(((size_t)ab * 3 + i) * outs + o) * n + k];
        ((uint32_t*)A.x)[(((size_t)ab * 2 + i) * outs + o) * n + k] = mulm(subm(v, smod(c, q), q), A.c[i], q, A.mu[i]);
    }
}

// x[k] += y[k], prime index k / per
__global__ void add_k(PcmmArgs A) {
    const size_t k = blockIdx.x * (size_t)BX + threadIdx.x;
    if (k >= (size_t)A.count * A.i0) return;
    const uint32_t p = A.p[k / A.i0];
    uint32_t* x = (uint32_t*)A.x;
    x[k] = addm(x[k], ((const uint32_t*)A.y)[k], p);
}

// ── negacyclic NTT over {q0, q1, P}: poly k uses prime (k / inner) % 3 ─────────────────────────────────────────
// Tables (the host NttTab order): tw[prime * n + m + i] = psi^brev(m + i) (forward) / psi^-brev(m + i) (inverse).

// one Cooley-Tukey stage (m groups of 2t) over every poly
__global__ void ntt_stage_k(PcmmArgs A, int m, int t) {
    const int b = blockIdx.x * BX + threadIdx.x;
    const int n = A.n;
    if (b >= n / 2) return;
    const int poly = blockIdx.y;
    const int pi = (poly / A.i0) % 3;
    const uint32_t p = A.p[pi];
    const int i = b / t, j = 2 * i * t + (b % t);
    uint32_t* a = (uint32_t*)A.x + (size_t)poly * n;
    const uint32_t w = ((const uint32_t*)A.tw)[(size_t)pi * n + m + i];
    const uint32_t u = a[j], v = mulm(a[j + t], w, p, A.mu[pi]);
    a[j] = addm(u, v, p);
    a[j + t] = subm(u, v, p);
}
// one Gentleman-Sande stage
__global__ void intt_stage_k(PcmmArgs A, int m, int t) {
    const int b = blockIdx.x * BX + threadIdx.x;
    const int n = A.n;
    if (b >= n / 2) return;
    const int poly = blockIdx.y;
    const int pi = (poly / A.i0) % 3;
    const uint32_t p = A.p[pi];
    const int i = b / t, j = 2 * i * t + (b % t);
    uint32_t* a = (uint32_t*)A.x + (size_t)poly * n;
    const uint32_t w = ((const uint32_t*)A.tw)[(size_t)pi * n + m + i];
    const uint32_t u = a[j], v = a[j + t];
    a[j] = addm(u, v, p);
    a[j + t] = mulm(subm(u, v, p), w, p, A.mu[pi]);
}
__global__ void scale_k(PcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int n = A.n;
    if (k >= n) return;
    const int poly = blockIdx.y;
    const int pi = (poly / A.i0) % 3;
    uint32_t* a = (uint32_t*)A.x + (size_t)poly * n;
    a[k] = mulm(a[k], A.c[pi], A.p[pi], A.mu[pi]);
}
// The stages whose butterfly groups fit one 4096-block, in shared memory: block = (poly, sub-block sb) of S = n/4096
// sub-blocks. Forward: m = S .. n/2 (t = 2048 .. 1); inverse: m = n/2 .. S, then (n == 4096) the n^-1 scaling.
__global__ void ntt_smem_k(PcmmArgs A, int inverse) {
    __shared__ uint32_t s[SMEM_N];
    const int n = A.n, S = n / SMEM_N;
    const int poly = blockIdx.x / S, sb = blockIdx.x % S;
    const int pi = (poly / A.i0) % 3;
    const uint32_t p = A.p[pi];
    const uint64_t mu = A.mu[pi];
    uint32_t* g = (uint32_t*)A.x + (size_t)poly * n + (size_t)sb * SMEM_N;
    const uint32_t* w = (const uint32_t*)A.tw + (size_t)pi * n;
    for (int k = threadIdx.x; k < SMEM_N; k += blockDim.x) s[k] = g[k];
    __syncthreads();
    if (!inverse) {
        for (int m = S, t = SMEM_N / 2; t >= 1; m *= 2, t /= 2) {
            const int gpb = m / S;  // groups per sub-block
            for (int b = threadIdx.x; b < SMEM_N / 2; b += blockDim.x) {
                const int il = b / t, j = 2 * il * t + (b % t);
                const uint32_t u = s[j], v = mulm(s[j + t], w[m + sb * gpb + il], p, mu);
                s[j] = addm(u, v, p);
                s[j + t] = subm(u, v, p);
            }
            __syncthreads();
        }
    } else {
        for (int m = n / 2, t = 1; t <= SMEM_N / 2; m /= 2, t *= 2) {
            const int gpb = m / S;
            for (int b = threadIdx.x; b < SMEM_N / 2; b += blockDim.x) {
                const int il = b / t, j = 2 * il * t + (b % t);
                const uint32_t u = s[j], v = s[j + t];
                s[j] = addm(u, v, p);
                s[j + t] = mulm(subm(u, v, p), w[m + sb * gpb + il], p, mu);
            }
            __syncthreads();
        }
    }
    const bool sc = inverse && S == 1;
    for (int k = threadIdx.x; k < SMEM_N; k += blockDim.x) g[k] = sc ? mulm(s[k], A.c[pi], p, mu) : s[k];
}

// ── MLWE rows, digit planes, recombination ──────────────────────────────────────────────────────────────────────

// A[pi][rowOff + row][col], row = sub R + i < NCH, col = l NP + w < N1:
//   l <= i: ka[(w R + (i - l)) 16 + sub];  l > i: w == 0 ? -ka[((NP-1) R + c) 16 + sub] : ka[((w-1) R + c) 16 + sub],
//   c = i - l + R
__global__ void mlwe_a_k(PcmmArgs A) {
    const int col = blockIdx.x * BX + threadIdx.x;
    const int row = blockIdx.y, pi = blockIdx.z;
    const int R = A.i2, NP = A.i3, rowsTotal = A.i0, rowOff = A.i1;
    const int N1 = N / 16;
    if (col >= N1) return;
    const uint32_t p = A.p[pi];
    const int sub = row / R, i = row % R, l = col / NP, w = col % NP;
    const uint32_t* a = (const uint32_t*)A.y + (size_t)pi * N;
    uint32_t v;
    if (l <= i) {
        v = a[(w * R + (i - l)) * 16 + sub];
    } else {
        const int c = i - l + R;
        v = w == 0 ? subm(0, a[((NP - 1) * R + c) * 16 + sub], p) : a[((w - 1) * R + c) * 16 + sub];
    }
    ((uint32_t*)A.x)[((size_t)pi * rowsTotal + rowOff + row) * N1 + col] = v;
}
// Bm[pi][rowOff + row][w] = b[pi][(w R + i) 16 + sub]
__global__ void mlwe_b_k(PcmmArgs A) {
    const int w = blockIdx.x * BX + threadIdx.x;
    const int row = blockIdx.y, pi = blockIdx.z;
    const int R = A.i2, NP = A.i3, rowsTotal = A.i0, rowOff = A.i1;
    if (w >= NP) return;
    const int sub = row / R, i = row % R;
    ((uint32_t*)A.x)[((size_t)pi * rowsTotal + rowOff + row) * NP + w] = ((const uint32_t*)A.y)[(size_t)pi * N + (w * R + i) * 16 + sub];
}
// d[t][col2][row] (int8) = balanced base-256 digit t of centered X[pi][row][col], col2 = pi cols + col: the GEMM's A
// operand stored k x m column-major (K-contiguous, cuBLASLt's col-TN layout). 32 x 32 tiles through shared memory.
__global__ void digits_k(PcmmArgs A) {
    __shared__ int8_t tile[4][32][33];
    const int rows = A.i0, cols = A.i1;
    const int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32, pi = blockIdx.z;
    const int tx = threadIdx.x, ty = threadIdx.y;  // 32 x 8
    const uint32_t p = A.p[pi];
    const uint32_t* X = (const uint32_t*)A.y + (size_t)pi * rows * cols;
    for (int dy = ty; dy < 32; dy += 8) {
        const int r = r0 + dy, c = c0 + tx;
        if (r < rows && c < cols) {
            int64_t v = centered(X[(size_t)r * cols + c], p);
            for (int t = 0; t < 4; ++t) {
                int64_t q = ((v % 256) + 256) % 256;
                if (q > 127) q -= 256;
                tile[t][dy][tx] = (int8_t)q;
                v = (v - q) / 256;
            }
        }
    }
    __syncthreads();
    const size_t c2 = 2 * (size_t)cols, count = c2 * rows;
    int8_t* d = (int8_t*)A.x;
    for (int dy = ty; dy < 32; dy += 8) {
        const int c = c0 + dy, r = r0 + tx;
        if (r < rows && c < cols) {
            const size_t o = ((size_t)pi * cols + c) * rows + r;
            for (int t = 0; t < 4; ++t) d[(size_t)t * count + o] = tile[t][tx][dy];
        }
    }
}
// Y[pi][rowOff + r][col] = sum_e G_e 256^e mod p (Horner), G[e][r][col2] int32 (the GEMM's m = 2 cols x n = rc output)
__global__ void recombine_k(PcmmArgs A) {
    const size_t k = blockIdx.x * (size_t)BX + threadIdx.x;
    const int rc = A.i0, rowOff = A.i1, rowsOut = A.i2, cols = A.i3, E = A.i4;
    const size_t c2 = 2 * (size_t)cols, count = (size_t)rc * c2;
    if (k >= count) return;
    const size_t r = k / c2, cc = k % c2;
    const int pi = (int)(cc / cols);
    const int64_t p = A.p[pi];
    const int32_t* G = (const int32_t*)A.y;
    int64_t v = 0;
    for (int e = E - 1; e >= 0; --e) v = (v * 256 + G[(size_t)e * count + k]) % p;
    if (v < 0) v += p;
    ((uint32_t*)A.x)[((size_t)pi * rowsOut + rowOff + r) * cols + cc % cols] = (uint32_t)v;
}

// ── pack back / compose ─────────────────────────────────────────────────────────────────────────────────────────

// C[pi][sub R + l][w R + i] = Ap[pi][rowOff + sub R + i][l NP + w]
__global__ void pack_c_k(PcmmArgs A) {
    const int idx = blockIdx.x * BX + threadIdx.x;
    const int sl = blockIdx.y, pi = blockIdx.z;
    const int R = A.i2, NP = A.i3, rowsTotal = A.i0, rowOff = A.i1, NCH = A.count;
    const int N1 = N / 16;
    if (idx >= N1) return;
    const int sub = sl / R, l = sl % R, w = idx / R, i = idx % R;
    ((uint32_t*)A.x)[((size_t)pi * NCH + sl) * N1 + idx] = ((const uint32_t*)A.y)[((size_t)pi * rowsTotal + rowOff + sub * R + i) * N1 + l * NP + w];
}
// pb[pi][sub][w R + i] += Bp[pi][rowOff + sub R + i][w]
__global__ void add_bpk_k(PcmmArgs A) {
    const int idx = blockIdx.x * BX + threadIdx.x;
    const int sub = blockIdx.y, pi = blockIdx.z;
    const int R = A.i2, NP = A.i3, rowsTotal = A.i0, rowOff = A.i1;
    const int N1 = N / 16;
    if (idx >= N1) return;
    const int w = idx / R, i = idx % R;
    uint32_t* x = (uint32_t*)A.x + ((size_t)pi * 16 + sub) * N1 + idx;
    *x = addm(*x, ((const uint32_t*)A.y)[((size_t)pi * rowsTotal + rowOff + sub * R + i) * NP + w], A.p[pi]);
}
// RX[pi][z 16 + sub] = px[pi][sub][z]
__global__ void compose_k(PcmmArgs A) {
    const int k = blockIdx.x * BX + threadIdx.x;
    const int pi = blockIdx.y;
    if (k >= N) return;
    const int N1 = N / 16;
    const int z = k / 16, sub = k % 16;
    ((uint32_t*)A.x)[(size_t)pi * N + k] = ((const uint32_t*)A.y)[((size_t)pi * 16 + sub) * N1 + z];
}

inline unsigned blocks(size_t n) {
    return (unsigned)((n + BX - 1) / BX);
}

template <class W>
int launch(const PcmmArgs* a, cudaStream_t s) {
    const PcmmArgs A = *a;
    const int n = A.n;
    switch (A.op) {
        case PCMM_GATHER:
            if (A.i4 != 0 && A.i4 != 3 && A.i4 != 4) return -1;
            gather_k<W><<<dim3(blocks(N), 2 * (A.i4 ? A.i4 : 4)), BX, 0, s>>>(A);
            break;
        case PCMM_SCATTER:
            if (A.i4 != 0 && A.i4 != 3 && A.i4 != 4) return -1;
            scatter_k<W><<<dim3(blocks(N), 2 * (A.i4 ? A.i4 : 4)), BX, 0, s>>>(A);
            break;
        case PCMM_IN_RESCALE:
            if (A.i4 != 0 && A.i4 != 3 && A.i4 != 4) return -1;
            in_rescale_k<W><<<dim3(blocks(N), 2), BX, 0, s>>>(A);
            break;
        case PCMM_LIFT:
            if (A.i4 != 0 && A.i4 != 3 && A.i4 != 4) return -1;
            lift_k<W><<<dim3(blocks(N), A.i4 ? A.i4 : 4), BX, 0, s>>>(A);
            break;
        case PCMM_KS_DIGITS:
            if (A.count < 1 || A.count > 65535) return -1;
            ks_digits_k<<<dim3(blocks(n), A.count), BX, 0, s>>>(A);
            break;
        case PCMM_KS_MAC:
            if (A.i0 < 1 || A.i0 > 65535) return -1;
            ks_mac_k<<<dim3(blocks(n), A.i0, 3), BX, 0, s>>>(A);
            break;
        case PCMM_MOD_DOWN:
            mod_down_k<<<dim3(blocks(n), A.i0, 2), BX, 0, s>>>(A);
            break;
        case PCMM_ADD:
            add_k<<<blocks((size_t)A.count * A.i0), BX, 0, s>>>(A);
            break;
        case PCMM_NTT:
        case PCMM_INTT: {
            if (n < SMEM_N || n % SMEM_N || A.count < 1 || A.count > 65535) return -1;
            const int S = n / SMEM_N;
            const dim3 g(blocks(n / 2), A.count);
            if (A.op == PCMM_NTT) {
                for (int m = 1, t = n / 2; m < S; m *= 2, t /= 2) ntt_stage_k<<<g, BX, 0, s>>>(A, m, t);
                ntt_smem_k<<<A.count * S, SMEM_TH, 0, s>>>(A, 0);
            } else {
                ntt_smem_k<<<A.count * S, SMEM_TH, 0, s>>>(A, 1);
                if (S > 1) {
                    for (int m = S / 2, t = n / (2 * m); m >= 1; m /= 2, t *= 2) intt_stage_k<<<g, BX, 0, s>>>(A, m, t);
                    scale_k<<<dim3(blocks(n), A.count), BX, 0, s>>>(A);
                }
            }
            break;
        }
        case PCMM_MLWE_A:
            mlwe_a_k<<<dim3(blocks(N / 16), A.count, 2), BX, 0, s>>>(A);
            break;
        case PCMM_MLWE_B:
            mlwe_b_k<<<dim3(blocks(A.i3), A.count, 2), BX, 0, s>>>(A);
            break;
        case PCMM_DIGITS:
            if ((A.i0 + 31) / 32 > 65535) return -1;
            digits_k<<<dim3((A.i1 + 31) / 32, (A.i0 + 31) / 32, 2), dim3(32, 8), 0, s>>>(A);
            break;
        case PCMM_RECOMBINE:
            recombine_k<<<blocks((size_t)A.i0 * 2 * A.i3), BX, 0, s>>>(A);
            break;
        case PCMM_PACK_C:
            pack_c_k<<<dim3(blocks(N / 16), A.count, 2), BX, 0, s>>>(A);
            break;
        case PCMM_ADD_BPK:
            add_bpk_k<<<dim3(blocks(N / 16), 16, 2), BX, 0, s>>>(A);
            break;
        case PCMM_COMPOSE:
            compose_k<<<dim3(blocks(N), 2), BX, 0, s>>>(A);
            break;
        default:
            return -2;
    }
    return (int)cudaGetLastError();
}
}  // namespace pcmm
}  // namespace pc
