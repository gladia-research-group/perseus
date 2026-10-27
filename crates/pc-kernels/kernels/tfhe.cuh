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

// ── the blind rotation in RNS: Q = q_a q_b (two ~25-bit primes, u32 residues, 32-bit Shoup products) ─────────────
// c[0] base log, c[1] levels, c[2..6) n^-1 mod q_a and its companion, n^-1 mod q_b and its companion,
// c[6] = q_a | q_b << 32, c[7] = q_a^-1 mod q_b | its Shoup companion mod q_b << 32; q = Q;
// m = floor(2^(64 + levels base) / Q)
__device__ __forceinline__ uint32_t shoup32(uint32_t a, uint32_t w, uint32_t ws, uint32_t q) {
    const uint32_t r = a * w - __umulhi(a, ws) * q;
    return r >= q ? r - q : r;
}
// lazy Shoup product: any a < 2^32, result in [0, 2 q)
__device__ __forceinline__ uint32_t shoup32l(uint32_t a, uint32_t w, uint32_t ws, uint32_t q) {
    return a * w - __umulhi(a, ws) * q;
}
// x in [0, 2 b) -> [0, b)
__device__ __forceinline__ uint32_t fold32(uint32_t x, uint32_t b) { return min(x, x - b); }
__device__ __forceinline__ uint32_t add32(uint32_t a, uint32_t b, uint32_t q) {
    const uint32_t s = a + b;
    return s >= q ? s - q : s;
}
__device__ __forceinline__ uint32_t sub32(uint32_t a, uint32_t b, uint32_t q) {
    return a >= b ? a - b : a + q - b;
}
__device__ __forceinline__ uint32_t mono32(const uint32_t* a, int n, int kp, int o, uint32_t q) {
    int i = o - kp;
    if (i >= 0) return a[i];
    i += n;
    if (i >= 0) return a[i] ? q - a[i] : 0;
    return a[i + n];
}
// x mod Q from (r_a, r_b): r_a + q_a ((r_b - r_a) q_a^-1 mod q_b); invs = the Shoup companion of inv mod q_b.
// q_a < 2 q_b (two 25-bit primes): r_a mod q_b is one conditional subtraction
__device__ __forceinline__ uint64_t crt(uint32_t ra, uint32_t rb, uint32_t qa, uint32_t qb, uint32_t inv, uint32_t invs) {
    const uint32_t ram = ra >= qb ? ra - qb : ra;
    const uint32_t k = shoup32(sub32(rb, ram, qb), inv, invs, qb);
    return ra + (uint64_t)qa * k;
}
// both primes' polynomials [2][n] in place: forward (natural -> bit-reversed) / inverse (with n^-1)
__device__ void ntt2_fwd(uint32_t* a, int n, const uint32_t* tw, const uint32_t* tws, uint32_t qa, uint32_t qb) {
    for (int m = 1, t = n / 2; m < n; m *= 2, t /= 2) {
        for (int bb = threadIdx.x; bb < n; bb += blockDim.x) {
            const int p = bb >= n / 2, b = bb - p * (n / 2);
            const uint32_t q = p ? qb : qa;
            uint32_t* x = a + p * n;
            const int i = b / t, j = 2 * i * t + b % t;
            const int wi = p * 2 * n + m + i;
            const uint32_t u = x[j], v = shoup32(x[j + t], tw[wi], tws[wi], q);
            x[j] = add32(u, v, q);
            x[j + t] = sub32(u, v, q);
        }
        __syncthreads();
    }
}
// P polynomials [P][n] in place, poly k at prime k & 1 (forward, natural -> bit-reversed)
__device__ void nttp_fwd(uint32_t* a, int P, int n, const uint32_t* tw, const uint32_t* tws, uint32_t qa, uint32_t qb) {
    // every divisor is a power of two: shifts and masks (a variable integer division costs more than the product)
    const int half = n / 2, lh = __ffs(half) - 1;
    for (int m = 1, t = half, lt = lh; m < n; m *= 2, t /= 2, lt--) {
        for (int bb = threadIdx.x; bb < P * half; bb += blockDim.x) {
            const int k = bb >> lh, b = bb & (half - 1), p = k & 1;
            const uint32_t q = p ? qb : qa;
            uint32_t* x = a + k * n;
            const int i = b >> lt, j = (i << (lt + 1)) + (b & (t - 1));
            const int wi = p * 2 * n + m + i;
            const uint32_t u = x[j], v = shoup32(x[j + t], tw[wi], tws[wi], q);
            x[j] = add32(u, v, q);
            x[j + t] = sub32(u, v, q);
        }
        __syncthreads();
    }
}
// P polynomials [P][n] in place, poly k at prime k & 1 (inverse, bit-reversed -> natural, WITHOUT the n^-1: the
// rotation's key carries it)
__device__ void nttp_inv(uint32_t* a, int P, int n, const uint32_t* tw, const uint32_t* tws, uint32_t qa, uint32_t qb) {
    const int half = n / 2, lh = __ffs(half) - 1;
    for (int m = half, t = 1, lt = 0; m >= 1; m /= 2, t *= 2, lt++) {
        for (int bb = threadIdx.x; bb < P * half; bb += blockDim.x) {
            const int k = bb >> lh, b = bb & (half - 1), p = k & 1;
            const uint32_t q = p ? qb : qa;
            uint32_t* x = a + k * n;
            const int i = b >> lt, j = (i << (lt + 1)) + (b & (t - 1));
            const int wi = p * 2 * n + n + m + i;
            const uint32_t u = x[j], v = x[j + t];
            x[j] = add32(u, v, q);
            x[j + t] = shoup32(sub32(u, v, q), tw[wi], tws[wi], q);
        }
        __syncthreads();
    }
}
// Register-blocked NTTs of P polynomials [P][2048] (poly k at prime k & 1): every phase loads groups of 2^S
// coefficients into registers, runs S butterfly stages on them and writes back once (11 stages = 2 + 3 + 3 + 3:
// four shared-memory round trips and barriers instead of eleven). Same butterflies and twiddles as nttp_fwd / nttp_inv,
// reduced lazily (q < 2^25 leaves 7 bits of headroom): the forward takes [0, q) and adds < 2 q per stage (output
// < 23 q); the inverse keeps [0, 2 q). The rotation reduces exactly once, when it adds the product into acc.
constexpr int RN = 2048, RLN = 11;
// The shared polynomials (digits, products) are stored swizzled: word j at j ^ (bits 5-7 into 0-2) ^ (bits 6-7 into
// 3-4). The phases with groups of stride 1 and 8 then hit 32 distinct banks per warp (plain layout: 8- and 4-way
// conflicts); stride >= 32 stays a permutation of one row.
__device__ __forceinline__ int swz(int j) { return j ^ ((j >> 5) & 7) ^ (((j >> 6) & 3) << 3); }
// c (1, 2 or 4) consecutive twiddles and companions from wi (a multiple of c) in one vector load each
__device__ __forceinline__ void twl(const uint32_t* tw, const uint32_t* tws, int wi, int c, uint32_t* w, uint32_t* ws) {
    if (c == 4) {
        const uint4 a = __ldg((const uint4*)(tw + wi)), b = __ldg((const uint4*)(tws + wi));
        w[0] = a.x, w[1] = a.y, w[2] = a.z, w[3] = a.w, ws[0] = b.x, ws[1] = b.y, ws[2] = b.z, ws[3] = b.w;
    } else if (c == 2) {
        const uint2 a = __ldg((const uint2*)(tw + wi)), b = __ldg((const uint2*)(tws + wi));
        w[0] = a.x, w[1] = a.y, ws[0] = b.x, ws[1] = b.y;
    } else {
        w[0] = __ldg(tw + wi), ws[0] = __ldg(tws + wi);
    }
}
// forward (Cooley-Tukey) phase: stages t = tt, tt / 2, .., tt >> (S - 1); a group = j0 + k tl (tl = tt >> (S - 1))
template <int S>
__device__ __forceinline__ void fwd_phase(uint32_t* a, int P, int ltt, const uint32_t* tw, const uint32_t* tws, uint32_t qa, uint32_t qb) {
    constexpr int G = 1 << S;
    const int ltl = ltt - (S - 1), tl = 1 << ltl;
    const int lgpp = RLN - S;   // groups per polynomial = RN / G
    for (int gg = threadIdx.x; gg < (P << lgpp); gg += blockDim.x) {
        const int k = gg >> lgpp, g = gg & ((1 << lgpp) - 1), p = k & 1;
        const uint32_t q = p ? qb : qa;
        uint32_t* x = a + k * RN;
        const int gi = g >> ltl;                         // the group's 2 tt block
        const int j0 = (gi << (ltt + 1)) + (g & (tl - 1));
        uint32_t v[G];
#pragma unroll
        for (int e = 0; e < G; e++) v[e] = x[swz(j0 + e * tl)];
#pragma unroll
        for (int st = 0; st < S; st++) {
            const int h = G >> (st + 1);                 // partner distance in the group
            const int lt = ltt - st;                     // log2 t of this stage
            const int m = RN >> (lt + 1);
            uint32_t wt[4], wts[4];   // this stage's twiddles: i = (gi << st) + r, r < 2^st
            twl(tw, tws, p * 2 * RN + m + (gi << st), 1 << st, wt, wts);
#pragma unroll
            for (int e = 0; e < G; e++) {
                if (e & h) continue;
                const int r = e >> (S - st);
                const uint32_t u = v[e], w = shoup32l(v[e + h], wt[r], wts[r], q);
                v[e] = u + w;
                v[e + h] = u - w + 2 * q;
            }
        }
#pragma unroll
        for (int e = 0; e < G; e++) x[swz(j0 + e * tl)] = v[e];
    }
    __syncthreads();
}
__device__ void nttr_fwd(uint32_t* a, int P, const uint32_t* tw, const uint32_t* tws, uint32_t qa, uint32_t qb) {
    fwd_phase<2>(a, P, 10, tw, tws, qa, qb);   // t = 1024, 512
    fwd_phase<3>(a, P, 8, tw, tws, qa, qb);    // t = 256 .. 64
    fwd_phase<3>(a, P, 5, tw, tws, qa, qb);    // t = 32 .. 8
    fwd_phase<3>(a, P, 2, tw, tws, qa, qb);    // t = 4 .. 1
}
// inverse (Gentleman-Sande) phase: stages t = tb, 2 tb, .., tb << (S - 1); a group = j0 + k tb, spanning G tb
template <int S>
__device__ __forceinline__ void inv_phase(uint32_t* a, int P, int ltb, const uint32_t* tw, const uint32_t* tws, uint32_t qa, uint32_t qb) {
    constexpr int G = 1 << S;
    const int tb = 1 << ltb;
    const int lgpp = RLN - S;
    for (int gg = threadIdx.x; gg < (P << lgpp); gg += blockDim.x) {
        const int k = gg >> lgpp, g = gg & ((1 << lgpp) - 1), p = k & 1;
        const uint32_t q = p ? qb : qa;
        uint32_t* x = a + k * RN;
        const int gi = g >> ltb;                         // the group's G tb block
        const int j0 = (gi << (ltb + S)) + (g & (tb - 1));
        uint32_t v[G];
#pragma unroll
        for (int e = 0; e < G; e++) v[e] = x[swz(j0 + e * tb)];
#pragma unroll
        for (int st = 0; st < S; st++) {
            const int h = 1 << st;                       // partner distance
            const int lt = ltb + st;                     // log2 t of this stage
            const int m = RN >> (lt + 1);
            uint32_t wt[4], wts[4];   // i = (gi << (S - 1 - st)) + r, r < 2^(S - 1 - st)
            twl(tw, tws, p * 2 * RN + RN + m + (gi << (S - 1 - st)), 1 << (S - 1 - st), wt, wts);
#pragma unroll
            for (int e = 0; e < G; e++) {
                if (e & h) continue;
                const int r = e >> (st + 1);
                const uint32_t u = v[e], w = v[e + h];
                v[e] = fold32(u + w, 2 * q);
                v[e + h] = shoup32l(u - w + 2 * q, wt[r], wts[r], q);
            }
        }
#pragma unroll
        for (int e = 0; e < G; e++) x[swz(j0 + e * tb)] = v[e];
    }
    __syncthreads();
}
__device__ void nttr_inv(uint32_t* a, int P, const uint32_t* tw, const uint32_t* tws, uint32_t qa, uint32_t qb) {
    inv_phase<3>(a, P, 0, tw, tws, qa, qb);    // t = 1 .. 4
    inv_phase<3>(a, P, 3, tw, tws, qa, qb);    // t = 8 .. 32
    inv_phase<3>(a, P, 6, tw, tws, qa, qb);    // t = 64 .. 256
    inv_phase<2>(a, P, 9, tw, tws, qa, qb);    // t = 512, 1024
}
__device__ void ntt2_inv(uint32_t* a, int n, const uint32_t* tw, const uint32_t* tws, const TfheArgs& A, uint32_t qa, uint32_t qb) {
    for (int m = n / 2, t = 1, lt = 0; m >= 1; m /= 2, t *= 2, lt++) {
        for (int bb = threadIdx.x; bb < n; bb += blockDim.x) {
            const int p = bb >= n / 2, b = bb - p * (n / 2);
            const uint32_t q = p ? qb : qa;
            uint32_t* x = a + p * n;
            const int i = b >> lt, j = (i << (lt + 1)) + (b & (t - 1));
            const int wi = p * 2 * n + n + m + i;
            const uint32_t u = x[j], v = x[j + t];
            x[j] = add32(u, v, q);
            x[j + t] = shoup32(sub32(u, v, q), tw[wi], tws[wi], q);
        }
        __syncthreads();
    }
    for (int k = threadIdx.x; k < 2 * n; k += blockDim.x) {
        const int p = k >= n;
        a[k] = shoup32(a[k], (uint32_t)A.c[p ? 4 : 2], (uint32_t)A.c[p ? 5 : 3], p ? qb : qa);
    }
    __syncthreads();
}

// acc [count][2 part][2 prime][n]: part 0 = X^-b~ tv (y = tv residues [2 prime][n]), part 1 = 0
__global__ void init_k(TfheArgs A) {
    const int n = A.n;
    const uint32_t qa = (uint32_t)A.c[6], qb = (uint32_t)(A.c[6] >> 32);
    const uint32_t bt = ((const uint32_t*)A.z)[(size_t)blockIdx.y * (A.i1 + 1) + A.i1];
    const int kp = (int)((2 * n - bt) % (2 * n));
    uint32_t* acc = (uint32_t*)A.x + (size_t)blockIdx.y * 4 * n;
    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < n; o += gridDim.x * blockDim.x) {
        for (int p = 0; p < 2; p++) {
            acc[p * n + o] = mono32((const uint32_t*)A.y + p * n, n, kp, o, p ? qb : qa);
            acc[(2 + p) * n + o] = 0;
        }
    }
}

// the whole blind rotation of every LWE (a block each, n / 2 = 1024 threads: two coefficients each); shared:
// acc [2 part][2 prime][n], tb [level][2 prime][n], o [2 part][2 prime][n] (12 n words at two levels). Per key bit:
// acc += GGSW [x] (acc X^-a~ - acc); bsk [key][c][t][part][prime][n], pre-multiplied by n^-1 (the inverse NTT then
// needs no scaling pass: exact arithmetic, so the result is the host's bit for bit).
constexpr int RT_E = 4;   // coefficients of [prime][n] (2 n) per thread at n = 2048, 1024 threads
__global__ void __launch_bounds__(STEP_TH) rotate_k(TfheArgs A) {
    extern __shared__ uint32_t sm32[];
    const int n = A.n, L = A.i1, bl = (int)A.c[0], lv = (int)A.c[1];
    const uint32_t qa = (uint32_t)A.c[6], qb = (uint32_t)(A.c[6] >> 32), inv = (uint32_t)A.c[7], invs = (uint32_t)(A.c[7] >> 32);
    uint32_t *acc = sm32, *tb = sm32 + 4 * n, *o = sm32 + 8 * n;
    const uint32_t* at = (const uint32_t*)A.z + (size_t)blockIdx.x * (L + 1);
    uint32_t* g = (uint32_t*)A.x + (size_t)blockIdx.x * 4 * n;
    for (int k = threadIdx.x; k < 4 * n; k += blockDim.x) acc[k] = g[k];
    __syncthreads();
    const uint32_t* tw = (const uint32_t*)A.tw;
    const uint32_t* tws = (const uint32_t*)A.tws;
    const size_t kstride = (size_t)2 * lv * 2 * 2 * n;
    const int bits = bl * lv;
    const int64_t B = 1ll << bl;
    for (int key = 0; key < L; key++) {
        const uint32_t ak = at[key];
        if (!ak) continue;
        const int kp = (int)((2 * n - ak) % (2 * n));
        const uint32_t* bsk = (const uint32_t*)A.y + key * kstride;
        const uint32_t* bsks = (const uint32_t*)A.w + key * kstride;
        uint32_t a0[RT_E], a1[RT_E];
#pragma unroll
        for (int e = 0; e < RT_E; e++) a0[e] = a1[e] = 0;
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const uint32_t* sa = acc + (2 * c) * n;       // part c, prime a
            const uint32_t* sb = acc + (2 * c + 1) * n;   // part c, prime b
            for (int j = threadIdx.x; j < n; j += blockDim.x) {
                const uint32_t da = sub32(mono32(sa, n, kp, j, qa), sa[j], qa);
                const uint32_t db = sub32(mono32(sb, n, kp, j, qb), sb[j], qb);
                int64_t v = (int64_t)scale_round(crt(da, db, qa, qb, inv, invs), A.m);
                if (v >= (1ll << (bits - 1))) v -= 1ll << bits;
                for (int t = 0; t < lv; t++) {
                    int64_t r = v & (B - 1);
                    if (r >= B / 2) r -= B;
                    v = (v - r) >> bl;
                    if (t == lv - 1) r += v << bl;
                    tb[(2 * t) * n + swz(j)] = r < 0 ? (uint32_t)((int64_t)qa + r) : (uint32_t)r;
                    tb[(2 * t + 1) * n + swz(j)] = r < 0 ? (uint32_t)((int64_t)qb + r) : (uint32_t)r;
                }
            }
            __syncthreads();
            nttr_fwd(tb, 2 * lv, tw, tws, qa, qb);
#pragma unroll
            for (int e = 0; e < RT_E; e++) {
                const int k = threadIdx.x + e * blockDim.x;
                const uint32_t q = k >= n ? qb : qa;
#pragma unroll
                for (int t = 0; t < 2; t++) {
                    const uint32_t d = tb[t * 2 * n + swz(k)];
                    const size_t off = (size_t)(c * lv + t) * 4 * n + k;
                    a0[e] += shoup32l(d, __ldg(bsk + off), __ldg(bsks + off), q);   // 4 terms < 2 q
                    a1[e] += shoup32l(d, __ldg(bsk + off + 2 * n), __ldg(bsks + off + 2 * n), q);
                }
            }
            __syncthreads();   // tb is rewritten by the next part's digits
        }
#pragma unroll
        for (int e = 0; e < RT_E; e++) {
            const int k = threadIdx.x + e * blockDim.x;
            const uint32_t q = k >= n ? qb : qa;
            o[swz(k)] = fold32(fold32(a0[e], 4 * q), 2 * q);
            o[2 * n + swz(k)] = fold32(fold32(a1[e], 4 * q), 2 * q);
        }
        __syncthreads();
        nttr_inv(o, 4, tw, tws, qa, qb);   // o [part][prime][n]: poly k at prime k & 1
        for (int k = threadIdx.x; k < 4 * n; k += blockDim.x) {
            const uint32_t q = (k >> (__ffs(n) - 1)) & 1 ? qb : qa;
            acc[k] = add32(acc[k], fold32(o[swz(k)], q), q);
        }
        __syncthreads();
    }
    for (int k = threadIdx.x; k < 4 * n; k += blockDim.x) g[k] = acc[k];
}

// sample extraction of coefficient 0 (CRT to mod Q), + offset c[0] (mod Q), then round(v c[1] / Q) mod c[1]
__global__ void extract_k(TfheArgs A) {
    const int n = A.n;
    const uint64_t Q = A.q, to = A.c[1];
    const uint32_t qa = (uint32_t)A.c[6], qb = (uint32_t)(A.c[6] >> 32), inv = (uint32_t)A.c[7], invs = (uint32_t)(A.c[7] >> 32);
    const uint32_t* acc = (const uint32_t*)A.y + (size_t)blockIdx.y * 4 * n;
    uint32_t* out = (uint32_t*)A.x + (size_t)blockIdx.y * (n + 1);
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k <= n; k += gridDim.x * blockDim.x) {
        uint64_t v;
        if (k == n) {
            v = addq(crt(acc[0], acc[n], qa, qb, inv, invs), A.c[0], Q);
        } else if (k == 0) {
            v = crt(acc[2 * n], acc[3 * n], qa, qb, inv, invs);
        } else {
            const uint64_t c1 = crt(acc[2 * n + n - k], acc[3 * n + n - k], qa, qb, inv, invs);
            v = c1 ? Q - c1 : 0;
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
        case PC_TFHE_PBS_ROTATE: {
            // the kernel is specialised to n = 2048 at two levels: 1024 threads, RT_E = 4 coefficients each
            const int th = STEP_TH;
            if (A.n != 2048 || A.c[1] != 2) return -1;
            const int smem = 12 * A.n * (int)sizeof(uint32_t);
            static bool attr_r = false;
            if (!attr_r) {   // the opt-in for the largest ring (n = 2048: 96 KB), once
                cudaFuncSetAttribute(rotate_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 12 * 2048 * (int)sizeof(uint32_t));
                attr_r = true;
            }
            rotate_k<<<A.count, th, smem, s>>>(A);
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
