// CKKS <-> TFHE scheme switching: the programmable bootstrap's kernels (semantics per op in args.cuh TfheArgs). Every
// step mirrors pc-tfhe (the host reference) operation for operation, so outputs are bit-exact: the negacyclic NTT is
// pc-core's order (psi powers bit-reversed, Cooley-Tukey forward / Gentleman-Sande inverse), the gadget digits and the
// mod switches are the fixed-point roundings of pc-tfhe gadget.rs (`scale_round`). Products mod the bootstrapping
// prime q < 2^62 are Shoup products against precomputed companions (twiddles, n^-1 and the bootstrapping key).
#pragma once   // needs <cooperative_groups.h> (instantiate.inc includes it at file scope)

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

// ── the blind rotation in RNS: Q = q_a q_b (two ~26-bit primes, u32 residues, 32-bit Shoup products) ─────────────
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
// Montgomery (R = 2^32): -q^-1 mod 2^32 (Newton), and x R^-1 mod q in [0, 2 q) for x < q 2^32
__device__ __forceinline__ uint32_t mont_neg_inv(uint32_t q) {
    uint32_t v = q;   // q v = 1 mod 2^3 for odd q
    for (int i = 0; i < 4; i++) v *= 2 - q * v;
    return 0u - v;
}
__device__ __forceinline__ uint32_t redc(uint64_t x, uint32_t q, uint32_t mi) {
    const uint32_t m = (uint32_t)x * mi;
    return (uint32_t)((x + (uint64_t)m * q) >> 32);
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
// q_a < 2 q_b (two primes of one size): r_a mod q_b is one conditional subtraction
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
// Register-blocked NTTs of P polynomials [P][2048] (poly k at prime k & pm: pm = 0 puts every poly at the first,
// with tw / tws / qa then that prime's): every phase loads groups of 2^S
// coefficients into registers, runs S butterfly stages on them and writes back once (11 stages = 2 + 3 + 3 + 3:
// four shared-memory round trips and barriers instead of eleven). Same butterflies and twiddles as nttp_fwd / nttp_inv,
// reduced lazily (q < 2^26 leaves 6 bits of headroom): the forward takes [0, q) and adds < 2 q per stage (output
// < 23 q); the inverse keeps [0, 2 q). The rotation reduces exactly once, when it adds the product into acc.
constexpr int RN = 2048, RLN = 11;
// The shared polynomials (digits, products) are stored swizzled: word j at j ^ (bits 5-7 into 0-2) ^ (bits 6-7 into
// 3-4). The phases with groups of stride 1 and 8 then hit 32 distinct banks per warp (plain layout: 8- and 4-way
// conflicts); stride >= 32 stays a permutation of one row.
__device__ __forceinline__ int swz(int j) { return j ^ ((j >> 5) & 7) ^ (((j >> 6) & 3) << 3); }
// c (1, 2 or 4) consecutive twiddles and companions from wi (a multiple of c) in one vector load each
template <bool TS = false>   // TS: the tables sit in shared memory (plain loads)
__device__ __forceinline__ void twl(const uint32_t* tw, const uint32_t* tws, int wi, int c, uint32_t* w, uint32_t* ws) {
    if (TS) {
        for (int i = 0; i < c; i++) w[i] = tw[wi + i], ws[i] = tws[wi + i];
    } else if (c == 4) {
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
template <int S, bool TS = false>
__device__ __forceinline__ void fwd_phase(uint32_t* a, int P, int ltt, const uint32_t* tw, const uint32_t* tws, uint32_t qa, uint32_t qb, int pm) {
    constexpr int G = 1 << S;
    const int ltl = ltt - (S - 1), tl = 1 << ltl;
    const int lgpp = RLN - S;   // groups per polynomial = RN / G
    for (int gg = threadIdx.x; gg < (P << lgpp); gg += blockDim.x) {
        const int k = gg >> lgpp, g = gg & ((1 << lgpp) - 1), p = k & pm;
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
            twl<TS>(tw, tws, p * 2 * RN + m + (gi << st), 1 << st, wt, wts);
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
template <bool TS = false>
__device__ void nttr_fwd(uint32_t* a, int P, const uint32_t* tw, const uint32_t* tws, uint32_t qa, uint32_t qb, int pm = 1) {
    fwd_phase<2, TS>(a, P, 10, tw, tws, qa, qb, pm);   // t = 1024, 512
    fwd_phase<3, TS>(a, P, 8, tw, tws, qa, qb, pm);    // t = 256 .. 64
    fwd_phase<3, TS>(a, P, 5, tw, tws, qa, qb, pm);    // t = 32 .. 8
    fwd_phase<3, TS>(a, P, 2, tw, tws, qa, qb, pm);    // t = 4 .. 1
}
// inverse (Gentleman-Sande) phase: stages t = tb, 2 tb, .., tb << (S - 1); a group = j0 + k tb, spanning G tb
template <int S, bool TS = false>
__device__ __forceinline__ void inv_phase(uint32_t* a, int P, int ltb, const uint32_t* tw, const uint32_t* tws, uint32_t qa, uint32_t qb, int pm) {
    constexpr int G = 1 << S;
    const int tb = 1 << ltb;
    const int lgpp = RLN - S;
    for (int gg = threadIdx.x; gg < (P << lgpp); gg += blockDim.x) {
        const int k = gg >> lgpp, g = gg & ((1 << lgpp) - 1), p = k & pm;
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
            twl<TS>(tw, tws, p * 2 * RN + RN + m + (gi << (S - 1 - st)), 1 << (S - 1 - st), wt, wts);
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
template <bool TS = false>
__device__ void nttr_inv(uint32_t* a, int P, const uint32_t* tw, const uint32_t* tws, uint32_t qa, uint32_t qb, int pm = 1) {
    inv_phase<3, TS>(a, P, 0, tw, tws, qa, qb, pm);    // t = 1 .. 4
    inv_phase<3, TS>(a, P, 3, tw, tws, qa, qb, pm);    // t = 8 .. 32
    inv_phase<3, TS>(a, P, 6, tw, tws, qa, qb, pm);    // t = 64 .. 256
    inv_phase<2, TS>(a, P, 9, tw, tws, qa, qb, pm);    // t = 512, 1024
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

// the whole blind rotation of NB LWEs per block (n / 2 = 1024 threads), unrolled by groups of G key bits (pc-tfhe
// pbs.rs; i2 = G): per group acc += sum_m (X^-e_m - 1) (K_m [x] acc), m = 1 .. 2^G - 1, e_m = the sum of the a's in
// the bit mask m. acc lives in registers (thread th owns
// coefficients th and th + 1024 of [part][prime]); shared: tb [NB][c][level][2 prime][n] (the digits of both parts;
// each LWE's product o [2 part][2 prime][n] reuses the first 4 n of its tb): 4 LV NB n words, then pm1 (4 n). The NB LWEs read each
// key word once. bsk [group][m][e][h < LV][thread][4] (pbs_image), Montgomery form and pre-multiplied by n^-1 (the
// inverse NTT then needs no scaling pass: exact arithmetic, so the result is the host's bit for bit). tw carries after
// the twiddles [prime][fwd | inv][n] the tables of NTT(X^e - 1): pm1 [prime][2 n] = g^x - 1 (Montgomery form) and
// E [prime][n], NTT(X^e - 1)[j] = pm1[E[j] e mod 2 n]. Both products sum in 64 bits and reduce once (redc).
#ifndef RT_SKIP
#define RT_SKIP 0   // EXPERIMENT timing: 1 skip MAC, 2 skip NTTs
#endif
constexpr int RT_E = 4;   // coefficients of [prime][n] (2 n) per thread at n = 2048, 1024 threads
template <int LV, int NB, int G, bool KB = false>
__global__ void __launch_bounds__(STEP_TH) rotate_k(TfheArgs A) {
    constexpr int KM = (1 << G) - 1;   // keys per group
    extern __shared__ uint32_t sm32[];
    const int n = A.n, L = A.i1, bl = (int)A.c[0];
    const int nbv = min(NB, A.count - (int)blockIdx.x * NB);   // LWEs of this block (the last may hold fewer)
    const uint32_t qa = (uint32_t)A.c[6], qb = (uint32_t)(A.c[6] >> 32), inv = (uint32_t)A.c[7], invs = (uint32_t)(A.c[7] >> 32);
    const size_t tbs = (size_t)4 * LV * n;
    const uint32_t* at[NB];
    uint32_t ar[NB][2][4];   // [lwe][jj][c prime]: coefficient th + 1024 jj
#pragma unroll
    for (int b = 0; b < NB; b++) {
        const int lw = (int)blockIdx.x * NB + min(b, nbv - 1);
        at[b] = (const uint32_t*)A.z + (size_t)lw * (L + 1);
        const uint32_t* g = (const uint32_t*)A.x + (size_t)lw * 4 * n;
#pragma unroll
        for (int jj = 0; jj < 2; jj++)
#pragma unroll
            for (int cp = 0; cp < 4; cp++) ar[b][jj][cp] = g[cp * n + threadIdx.x + jj * blockDim.x];
    }
    const uint32_t* tw = (const uint32_t*)A.tw;
    const uint32_t* tws = (const uint32_t*)A.tws;
    const uint32_t* ex = tw + 8 * n;
    const uint32_t mia = mont_neg_inv(qa), mib = mont_neg_inv(qb);
    const size_t kstride = (size_t)2 * LV * 2 * 2 * n;   // one GGSW
    const int bits = bl * LV, n2 = 2 * n;
    // pm1 [prime][2 n] in shared memory after the digits: its gathers (data-dependent, 2^G - 1 per word) sat on the
    // MAC's critical path from L2
    uint32_t* pm1 = sm32 + NB * tbs;
    for (int x = threadIdx.x; x < 2 * n2; x += blockDim.x) pm1[x] = __ldg(tw + 4 * n + x);
    const int64_t B = 1ll << bl;
    for (int pr = 0; pr < L / G; pr++) {
        int em[NB][KM];
        bool any = false;
#pragma unroll
        for (int b = 0; b < NB; b++) {
            uint32_t kx[G];
            bool nz = false;
#pragma unroll
            for (int x = 0; x < G; x++) {
                kx[x] = at[b][G * pr + x];
                nz |= kx[x] != 0;
            }
            const bool on = b < nbv;
#pragma unroll
            for (int m = 1; m <= KM; m++) {
                uint32_t e = 0;
#pragma unroll
                for (int x = 0; x < G; x++)
                    if (m >> x & 1) e += kx[x];
                em[b][m - 1] = on ? (int)((G * n2 - e) % n2) : 0;
            }
            any |= on && nz;
        }
        if (!any) continue;
#pragma unroll
        for (int b = 0; b < NB; b++) {
            uint32_t* tbb = sm32 + b * tbs;
#pragma unroll
            for (int jj = 0; jj < 2; jj++) {
                const int j = threadIdx.x + jj * blockDim.x;
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    int64_t v = (int64_t)scale_round(crt(ar[b][jj][2 * c], ar[b][jj][2 * c + 1], qa, qb, inv, invs), A.m);
                    if (v >= (1ll << (bits - 1))) v -= 1ll << bits;
#pragma unroll
                    for (int t = 0; t < LV; t++) {
                        int64_t r = v & (B - 1);
                        if (r >= B / 2) r -= B;
                        v = (v - r) >> bl;
                        if (t == LV - 1) r += v << bl;
                        uint32_t* d = tbb + (size_t)(c * LV + t) * 2 * n;
                        d[swz(j)] = r < 0 ? (uint32_t)((int64_t)qa + r) : (uint32_t)r;
                        d[n + swz(j)] = r < 0 ? (uint32_t)((int64_t)qb + r) : (uint32_t)r;
                    }
                }
            }
        }
        __syncthreads();
        if (RT_SKIP != 2) nttr_fwd(sm32, 4 * LV * NB, tw, tws, qa, qb);
        const uint4* bsk = (const uint4*)((const uint32_t*)A.y + (size_t)pr * KM * kstride) + threadIdx.x;
#pragma unroll
        for (int e = 0; e < RT_E; e++) {
            const int k = threadIdx.x + e * blockDim.x, p = k >= n;
            const uint32_t q = p ? qb : qa, mi = p ? mib : mia;
            const int ej = (int)ex[k];   // E [prime][n] at k = prime n + j
            uint32_t d[NB][2 * LV];   // [lwe][c][t]
#pragma unroll
            for (int b = 0; b < NB; b++)
#pragma unroll
                for (int ct = 0; ct < 2 * LV; ct++) d[b][ct] = sm32[b * tbs + ct * 2 * n + swz(k)];
            if constexpr (KB) {
                // the key bundle precomputed by PBS_KEYBUNDLE (w: [lwe][group][e][thread] uint4, NB = LV = 1)
                const uint4 g = __ldg((const uint4*)A.w + (((size_t)blockIdx.x * (L / G) + pr) * RT_E + e) * blockDim.x + threadIdx.x);
                const uint64_t s0 = (uint64_t)d[0][0] * g.x + (uint64_t)d[0][1] * g.z;
                const uint64_t s1 = (uint64_t)d[0][0] * g.y + (uint64_t)d[0][1] * g.w;
                sm32[swz(k)] = redc(s0, q, mi);   // < 2 q
                sm32[2 * n + swz(k)] = redc(s1, q, mi);
            } else {
                // the group's key first (TFHE-rs's key bundle, in NTT form): G[ct][part] = sum_m f_m K_m[ct][part], f_m =
                // NTT(X^-e_m - 1) (KM <= 7 terms < q q: < q 2^32), then one product with the digits: 2 KM LV + 2 LV wide
                // products and 2 LV + 2 reductions per word instead of KM (2 LV + 2) and 2 KM + 2
                uint64_t gk[NB][2 * LV][2];   // KM <= 15 terms < q q: < q 2^32 at q < 2^28
    #pragma unroll
                for (int b = 0; b < NB; b++)
    #pragma unroll
                    for (int ct = 0; ct < 2 * LV; ct++) gk[b][ct][0] = gk[b][ct][1] = 0;
    #pragma unroll
                for (int m = 0; m < (RT_SKIP == 1 ? 0 : KM); m++) {
                    uint4 w[LV];
    #pragma unroll
                    for (int h = 0; h < LV; h++) w[h] = __ldg(bsk + ((m * RT_E + e) * LV + h) * blockDim.x);
    #pragma unroll
                    for (int b = 0; b < NB; b++) {
                        if (!em[b][m]) continue;
                        const uint64_t f = pm1[p * n2 + ((ej * em[b][m]) & (n2 - 1))];
    #pragma unroll
                        for (int h = 0; h < LV; h++) {
                            gk[b][2 * h][0] += w[h].x * f;
                            gk[b][2 * h][1] += w[h].y * f;
                            gk[b][2 * h + 1][0] += w[h].z * f;
                            gk[b][2 * h + 1][1] += w[h].w * f;
                        }
                    }
                }
                // o aliases the digits: this thread alone reads and writes word k
    #pragma unroll
                for (int b = 0; b < NB; b++) {
                    uint64_t s0 = 0, s1 = 0;   // 2 LV terms < 23 q q: < q 2^32 (the launcher checks 46 LV q < 2^32)
    #pragma unroll
                    for (int ct = 0; ct < 2 * LV; ct++) {
                        s0 += (uint64_t)d[b][ct] * fold32(redc(gk[b][ct][0], q, mi), q);
                        s1 += (uint64_t)d[b][ct] * fold32(redc(gk[b][ct][1], q, mi), q);
                    }
                    sm32[b * tbs + swz(k)] = redc(s0, q, mi);   // < 2 q
                    sm32[b * tbs + 2 * n + swz(k)] = redc(s1, q, mi);
    
            }
        }
        }
        __syncthreads();
        // o [lwe][part][prime][n]: poly k at prime k & 1 (the stride between LWEs is a whole number of polys)
        if (LV == 1) {
            if (RT_SKIP != 2) nttr_inv(sm32, 4 * NB, tw, tws, qa, qb);
        } else {
#pragma unroll
            for (int b = 0; b < NB; b++) nttr_inv(sm32 + b * tbs, 4, tw, tws, qa, qb);
        }
#pragma unroll
        for (int b = 0; b < NB; b++)
#pragma unroll
            for (int jj = 0; jj < 2; jj++)
#pragma unroll
                for (int cp = 0; cp < 4; cp++) {
                    const uint32_t q = cp & 1 ? qb : qa;
                    const int k = cp * n + threadIdx.x + jj * blockDim.x;
                    ar[b][jj][cp] = add32(ar[b][jj][cp], fold32(sm32[b * tbs + swz(k)], q), q);
                }
        __syncthreads();
    }
#pragma unroll
    for (int b = 0; b < NB; b++) {
        if (b >= nbv) break;
        uint32_t* g = (uint32_t*)A.x + ((size_t)blockIdx.x * NB + b) * 4 * n;
#pragma unroll
        for (int jj = 0; jj < 2; jj++)
#pragma unroll
            for (int cp = 0; cp < 4; cp++) g[cp * n + threadIdx.x + jj * blockDim.x] = ar[b][jj][cp];
    }
}

// PBS_ROTATE's latency path on a cluster of two blocks per LWE, one per RNS prime (r = the block's rank, on the key
// bundles of PBS_KEYBUNDLE, one level): each block keeps its prime's acc [2 part][n] in shared memory, double-buffered
// by step parity, and reads the partner's through distributed shared memory for the digits (the CRT of both
// residues). Per group: cluster sync (the partner's acc is final), digits of both parts at its prime, forward NTT of
// the two digit polys, the MAC on its half of the bundle words, inverse NTT, acc' = acc + o into the other parity
// (the partner may still read this one: no second sync). Shared: acc 2 x 2 n, tb 2 n, this prime's twiddles and
// companions 4 n words.
template <int G>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(STEP_TH) rotate_cl_k(TfheArgs A) {
    extern __shared__ uint32_t sm32[];
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    const int r = (int)cl.block_rank();
    const int n = A.n, L = A.i1, bl = (int)A.c[0], lw = (int)blockIdx.x / 2;
    const uint32_t qa = (uint32_t)A.c[6], qb = (uint32_t)(A.c[6] >> 32), inv = (uint32_t)A.c[7], invs = (uint32_t)(A.c[7] >> 32);
    const uint32_t q = r ? qb : qa, mi = mont_neg_inv(q);
    uint32_t *acc = sm32, *tb = sm32 + 4 * n, *twsm = sm32 + 6 * n;   // acc [parity][2 part][n]; twsm: [fwd | inv][n] x 2
    const uint32_t* pacc = cl.map_shared_rank(acc, r ^ 1);
    int st = 0;   // the parity of the current acc
    uint32_t* g = (uint32_t*)A.x + (size_t)lw * 4 * n;   // [part][prime][n]
    for (int x = threadIdx.x; x < 2 * n; x += blockDim.x) acc[x] = g[(2 * (x / n) + r) * n + x % n];
    const uint32_t* at = (const uint32_t*)A.z + (size_t)lw * (L + 1);
    for (int x = threadIdx.x; x < 2 * n; x += blockDim.x) {
        twsm[x] = __ldg((const uint32_t*)A.tw + (size_t)r * 2 * n + x);
        twsm[2 * n + x] = __ldg((const uint32_t*)A.tws + (size_t)r * 2 * n + x);
    }
    const uint32_t *tw = twsm, *tws = twsm + 2 * n;
    const int64_t B = 1ll << bl;
    for (int pr = 0; pr < L / G; pr++) {
        bool nz = false;
#pragma unroll
        for (int x = 0; x < G; x++) nz |= at[G * pr + x] != 0;
        if (!nz) continue;   // the same in both blocks
        // the bundle words first: their L2 latency hides behind the digits and the forward NTT
        uint4 w[2];
#pragma unroll
        for (int ee = 0; ee < 2; ee++)
            w[ee] = __ldg((const uint4*)A.w + (((size_t)lw * (L / G) + pr) * RT_E + 2 * r + ee) * blockDim.x + threadIdx.x);
        cl.sync();
#pragma unroll
        for (int jj = 0; jj < 2; jj++) {
            const int j = threadIdx.x + jj * blockDim.x;
#pragma unroll
            for (int c = 0; c < 2; c++) {
                const uint32_t own = acc[st * 2 * n + c * n + j], oth = pacc[st * 2 * n + c * n + j];
                int64_t v = (int64_t)scale_round(crt(r ? oth : own, r ? own : oth, qa, qb, inv, invs), A.m);
                if (v >= B / 2) v -= B;   // one level: the digit is the centred value
                tb[c * n + swz(j)] = v < 0 ? (uint32_t)((int64_t)q + v) : (uint32_t)v;
            }
        }
        __syncthreads();
        nttr_fwd<true>(tb, 2, tw, tws, q, q, 0);
#pragma unroll
        for (int ee = 0; ee < 2; ee++) {
            const int j = threadIdx.x + ee * blockDim.x;
            const uint64_t d0 = tb[swz(j)], d1 = tb[n + swz(j)];
            // o aliases the digits: this thread alone reads and writes word j
            tb[swz(j)] = redc(d0 * w[ee].x + d1 * w[ee].z, q, mi);   // < 2 q
            tb[n + swz(j)] = redc(d0 * w[ee].y + d1 * w[ee].w, q, mi);
        }
        __syncthreads();
        nttr_inv<true>(tb, 2, tw, tws, q, q, 0);
#pragma unroll
        for (int jj = 0; jj < 2; jj++)
#pragma unroll
            for (int c = 0; c < 2; c++) {
                const int x = c * n + threadIdx.x + jj * blockDim.x;
                acc[(st ^ 1) * 2 * n + x] = add32(acc[st * 2 * n + x], fold32(tb[c * n + swz(threadIdx.x + jj * blockDim.x)], q), q);
            }
        st ^= 1;
    }
    cl.sync();   // the partner may still read this block's acc
    for (int x = threadIdx.x; x < 2 * n; x += blockDim.x) g[(2 * (x / n) + r) * n + x % n] = acc[st * 2 * n + x];
}

// PBS_KEYBUNDLE: the key bundle of every (LWE, group) for the latency path (TFHE-rs's multi-bit keybundle): x
// [count][group][e][thread] uint4 = (G[0][0], G[0][1], G[1][0], G[1][1]), G[c][part] = sum_m f_m K_m[c][part] (Montgomery
// form, < q; one level), the MAC of rotate_k computed for all groups at once, off the rotation's sequential path. y =
// bsk, z = the mod-switched LWEs, tw as PBS_ROTATE; grid (groups, count), 1024 threads.
template <int G>
__global__ void __launch_bounds__(STEP_TH) keybundle_k(TfheArgs A) {
    extern __shared__ uint32_t sm32[];
    constexpr int KM = (1 << G) - 1;
    const int n = A.n, L = A.i1, n2 = 2 * n, pr = blockIdx.x;
    const uint32_t qa = (uint32_t)A.c[6], qb = (uint32_t)(A.c[6] >> 32);
    const uint32_t mia = mont_neg_inv(qa), mib = mont_neg_inv(qb);
    const uint32_t* tw = (const uint32_t*)A.tw;
    const uint32_t* ex = tw + 8 * n;
    uint32_t* pm1 = sm32;
    for (int x = threadIdx.x; x < 2 * n2; x += blockDim.x) pm1[x] = __ldg(tw + 4 * n + x);
    const uint32_t* at = (const uint32_t*)A.z + (size_t)blockIdx.y * (L + 1);
    int em[KM];
#pragma unroll
    for (int m = 1; m <= KM; m++) {
        uint32_t e = 0;
#pragma unroll
        for (int x = 0; x < G; x++)
            if (m >> x & 1) e += at[G * pr + x];
        em[m - 1] = (int)((G * n2 - e) % n2);
    }
    __syncthreads();
    const size_t kstride = (size_t)2 * 2 * 2 * n;   // one GGSW at one level
    const uint4* bsk = (const uint4*)((const uint32_t*)A.y + (size_t)pr * KM * kstride) + threadIdx.x;
    uint4* out = (uint4*)A.x + ((size_t)blockIdx.y * (L / G) + pr) * RT_E * blockDim.x + threadIdx.x;
#pragma unroll
    for (int e = 0; e < RT_E; e++) {
        const int k = threadIdx.x + e * blockDim.x, p = k >= n;
        const uint32_t q = p ? qb : qa, mi = p ? mib : mia;
        const int ej = (int)ex[k];
        uint64_t g00 = 0, g01 = 0, g10 = 0, g11 = 0;   // KM <= 15 terms < q q
#pragma unroll
        for (int m = 0; m < KM; m++) {
            if (!em[m]) continue;
            const uint4 w = __ldg(bsk + (m * RT_E + e) * blockDim.x);
            const uint64_t f = pm1[p * n2 + ((ej * em[m]) & (n2 - 1))];
            g00 += w.x * f;
            g01 += w.y * f;
            g10 += w.z * f;
            g11 += w.w * f;
        }
        out[e * blockDim.x] = make_uint4(fold32(redc(g00, q, mi), q), fold32(redc(g01, q, mi), q), fold32(redc(g10, q, mi), q), fold32(redc(g11, q, mi), q));
    }
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

// ── the repack (pc-tfhe switch.rs RepackKey): q = p0, n = N, i1 = n_br, c[0] base log, c[1] levels ─────────────
// REPACK_DIGITS: D [count][n_br levels] (i32, k t) = exact digits of the LWEs' a (y = [count][n_br + 1] mod p0); also
// zeroes the GEMM's accumulator w ([count][2 limb][2 n_br] i64) and, when i0 = 1, the scatter's z ([2 part][n] u64)
__global__ void repack_digits_k(TfheArgs A) {
    const int nb = A.i1, lv = (int)A.c[1], bl = (int)A.c[0];
    const size_t tot = (size_t)A.count * nb, gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    if (A.i0)
        for (size_t z = gid; z < 2 * (size_t)A.n; z += stride) ((uint64_t*)A.z)[z] = 0;
    for (size_t z = gid; z < tot * 4; z += stride) ((uint64_t*)A.w)[z] = 0;
    if (gid >= tot) return;
    const size_t b = gid / nb, k = gid % nb;
    const int64_t v = cent(((const uint32_t*)A.y)[b * (nb + 1) + k], A.q);
    int32_t* out = (int32_t*)A.x + gid * lv;
    for (int t = 0; t < lv; t++) out[t] = (int32_t)digit_exact(v, t, bl, lv);
}

// REPACK_MAC: x [count][2 limb][2 n_br] (i64) += sum_kt D[.][kt] cols[kt][limb][part][i], columns (part, i)
// (limbs p0, P; sums < 2^55 exact in int64). z = cols [n_br levels][2 limb][2 part][n_br], y = D. A block: RG
// columns x RJ LWEs x RKC of the k t range (split K: exact int64 atomics), the digits staged in shared memory.
constexpr int RJ = 16, RKC = 256, RG = 256;
__global__ void __launch_bounds__(RG) repack_gemm_k(TfheArgs A) {
    __shared__ int32_t ds[RJ][RKC];
    const int nb = A.i1, K = nb * (int)A.c[1], b = A.count;
    const int col = blockIdx.x * RG + threadIdx.x, j0 = blockIdx.y * RJ;
    const uint32_t* key = (const uint32_t*)A.z + col;
    const int32_t* D = (const int32_t*)A.y;
    int64_t a0[RJ], a1[RJ];
#pragma unroll
    for (int jj = 0; jj < RJ; jj++) a0[jj] = a1[jj] = 0;
    {
        const int kc = blockIdx.z * RKC;
        for (int e = threadIdx.x; e < RJ * RKC; e += RG) {
            const int jj = e / RKC, kk = e % RKC;
            ds[jj][kk] = j0 + jj < b ? D[(size_t)(j0 + jj) * K + kc + kk] : 0;
        }
        __syncthreads();
        for (int kk = 0; kk < RKC; kk++) {
            const uint32_t* kp = key + (size_t)(kc + kk) * 4 * nb;
            const int64_t k0 = __ldg(kp), k1 = __ldg(kp + 2 * nb);
#pragma unroll
            for (int jj = 0; jj < RJ; jj++) {
                a0[jj] += ds[jj][kk] * k0;
                a1[jj] += ds[jj][kk] * k1;
            }
        }
    }
#pragma unroll
    for (int jj = 0; jj < RJ; jj++) {
        const int j = j0 + jj;
        if (j >= b) break;
        unsigned long long* o = (unsigned long long*)A.x + (size_t)j * 4 * nb + col;
        atomicAdd(o, (unsigned long long)a0[jj]);
        atomicAdd(o + 2 * nb, (unsigned long long)a1[jj]);
    }
}

// REPACK_FIN: PK [count][2 part][n_br] mod p0 = the rescale by P of (P b, 0) + the GEMM's sums y; w = the LWEs (b);
// c[2] = P, c[3] = P^-1 mod p0, c[4] = P mod p0, c[7] = mu(p0)
__global__ void repack_fin_k(TfheArgs A) {
    const int nb = A.i1;
    const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (size_t)A.count * 2 * nb) return;
    const size_t j = gid / (2 * nb);
    const int col = (int)(gid % (2 * nb));
    const uint64_t p0 = A.q, P = A.c[2];
    const int64_t* a = (const int64_t*)A.y + j * 4 * nb + col;
    uint64_t x0 = smod32(a[0], p0);
    if (col == 0) x0 = addq(x0, mulm32(smod32(cent(((const uint32_t*)A.w)[j * (nb + 1) + nb], p0), p0), (uint32_t)A.c[4], p0, A.c[7]), p0);
    const uint32_t x1 = smod32(cent(smod32(a[2 * nb], P), P), p0);
    ((uint32_t*)A.x)[gid] = mulm32((uint32_t)subq(x0, x1, p0), (uint32_t)A.c[3], p0, A.c[7]);
}

// REPACK_SCATTER: X^j PK_j(X^S) summed into z [2 part][n] (u64 atomics, reduced by REPACK_RING); y = PK, w = the
// positions j, i2 = S
__global__ void repack_scatter_k(TfheArgs A) {
    const int n = A.n, nb = A.i1, S = A.i2;
    const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (size_t)A.count * 2 * nb) return;
    const size_t b = gid / (2 * nb);
    const int part = (int)(gid / nb) & 1, i = (int)(gid % nb);
    int y = ((const int32_t*)A.w)[b] + S * i;
    uint64_t v = ((const uint32_t*)A.y)[gid];
    if (y >= n) {
        y -= n;
        v = v ? A.q - v : 0;
    }
    atomicAdd((unsigned long long*)A.z + (size_t)part * n + y, (unsigned long long)v);
}

// REPACK_RING: (c0, c1) = z mod p0; c0 into w [n]; the exact digits of c1 into x [2 limb][levels][n] mod (p0, p1 = c[4])
__global__ void repack_ring_k(TfheArgs A) {
    const int n = A.n, lv = (int)A.c[1], bl = (int)A.c[0];
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= n) return;
    const uint64_t* z = (const uint64_t*)A.y;
    ((uint32_t*)A.w)[y] = (uint32_t)(z[y] % A.q);
    const int64_t v = cent(z[n + y] % A.q, A.q);
    for (int t = 0; t < lv; t++) {
        const int64_t d = digit_exact(v, t, bl, lv);
        ((uint32_t*)A.x)[(size_t)t * n + y] = smod32(d, A.q);
        ((uint32_t*)A.x)[(size_t)(lv + t) * n + y] = smod32(d, A.c[4]);
    }
}

// REPACK_RMAC: acc [2 part][2 limb][n] += sum_t D[limb][t] R[t][limb][part] (NTT form; the ring key); c[4] = p1,
// c[6] = mu(p1), c[7] = mu(p0)
__global__ void repack_rmac_k(TfheArgs A) {
    const int n = A.n, lv = (int)A.c[1];
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int limb = blockIdx.y;
    const uint64_t p = limb ? A.c[4] : A.q, mu = limb ? A.c[6] : A.c[7];
    const uint32_t* d = (const uint32_t*)A.y + (size_t)limb * lv * n;
    const uint32_t* key = (const uint32_t*)A.z;
    uint32_t* acc0 = (uint32_t*)A.x + (size_t)limb * n;            // [part][limb][n]: part 0
    uint32_t* acc1 = (uint32_t*)A.x + (size_t)(2 + limb) * n;      // part 1
    uint64_t a0 = acc0[j], a1 = acc1[j];
    for (int t = 0; t < lv; t++) {
        const uint32_t dv = d[(size_t)t * n + j];
        const uint32_t* k0 = key + (((size_t)t * 2 + limb) * 2) * n;
        a0 = addq(a0, mulm32(dv, k0[j], p, mu), p);
        a1 = addq(a1, mulm32(dv, k0[n + j], p, mu), p);
    }
    acc0[j] = (uint32_t)a0;
    acc1[j] = (uint32_t)a1;
}

// REPACK_C0: acc part 0 [limb][n] += p1 c0 (y = c0 mod p0, centred; c[2] = p1 mod p0)
__global__ void repack_c0_k(TfheArgs A) {
    const int n = A.n;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int64_t v = cent(((const uint32_t*)A.y)[j], A.q);
    for (int limb = 0; limb < 2; limb++) {
        const uint64_t p = limb ? A.c[4] : A.q, mu = limb ? A.c[6] : A.c[7];
        const uint32_t p1 = limb ? 0 : (uint32_t)A.c[2];   // p1 mod p1 = 0
        uint32_t* c0 = (uint32_t*)A.x + (size_t)limb * n;
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
            // the kernel is specialised to n = 2048 at one or two levels: 1024 threads, RT_E = 4 coefficients each; the
            // lazy MAC needs 46 levels q < 2^32. Two LWEs per block (sharing the key reads) once the batch fills two
            // blocks per SM (one level only: the digits of two LWEs then fit in 64 KB)
            const int th = STEP_TH, lv = (int)A.c[1];
            const uint64_t qa = A.c[6] & 0xffffffffull, qb = A.c[6] >> 32, qmax = qa > qb ? qa : qb;
            if (A.n != 2048 || (lv != 1 && lv != 2) || 46ull * lv * qmax >= (1ull << 32)) return -1;
            const int grp = A.i2 ? A.i2 : 2;
            if (grp < 2 || grp > 4 || A.i1 % grp) return -1;
            static int sms = 0;
            static bool attr_r = false;
            if (!attr_r) {   // the shared-memory opt-ins (64 / 96 KB), once
                int dev = 0;
                cudaGetDevice(&dev);
                cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
                const int s4 = 8 * 2048 * (int)sizeof(uint32_t), s8 = 12 * 2048 * (int)sizeof(uint32_t);
                cudaFuncSetAttribute(rotate_k<1, 1, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, s4);
                cudaFuncSetAttribute(rotate_k<1, 2, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, s8);
                cudaFuncSetAttribute(rotate_k<2, 1, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, s8);
                cudaFuncSetAttribute(rotate_k<1, 1, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, s4);
                cudaFuncSetAttribute(rotate_k<1, 2, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, s8);
                cudaFuncSetAttribute(rotate_k<2, 1, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, s8);
                cudaFuncSetAttribute(rotate_k<1, 1, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, s4);
                cudaFuncSetAttribute(rotate_k<1, 2, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, s8);
                cudaFuncSetAttribute(rotate_k<2, 1, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, s8);
                attr_r = true;
            }
            // digits (4 / 8 n words) + pm1 (4 n)
            const int s4 = 8 * A.n * (int)sizeof(uint32_t), s8 = 12 * A.n * (int)sizeof(uint32_t);
            if (A.w) {   // the latency path: the key bundles precomputed (PBS_KEYBUNDLE), one LWE per block, one level
                static bool attr_k = false;
                if (!attr_k) {
                    cudaFuncSetAttribute(rotate_k<1, 1, 2, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, s4);
                    cudaFuncSetAttribute(rotate_k<1, 1, 3, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, s4);
                    cudaFuncSetAttribute(rotate_k<1, 1, 4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, s4);
                    attr_k = true;
                }
                if (lv != 1) return -1;
                if (A.i3 != 1) {   // a cluster of two blocks per LWE (i3 = 1: one block, for comparison)
                    const int sc = 10 * A.n * (int)sizeof(uint32_t);
                    static bool attr_c = false;
                    if (!attr_c) {
                        cudaFuncSetAttribute(rotate_cl_k<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 10 * 2048 * (int)sizeof(uint32_t));
                        cudaFuncSetAttribute(rotate_cl_k<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 10 * 2048 * (int)sizeof(uint32_t));
                        cudaFuncSetAttribute(rotate_cl_k<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 10 * 2048 * (int)sizeof(uint32_t));
                        attr_c = true;
                    }
                    if (grp == 2) rotate_cl_k<2><<<2 * A.count, th, sc, s>>>(A);
                    else if (grp == 3) rotate_cl_k<3><<<2 * A.count, th, sc, s>>>(A);
                    else rotate_cl_k<4><<<2 * A.count, th, sc, s>>>(A);
                    break;
                }
                if (grp == 2) rotate_k<1, 1, 2, true><<<A.count, th, s4, s>>>(A);
                else if (grp == 3) rotate_k<1, 1, 3, true><<<A.count, th, s4, s>>>(A);
                else rotate_k<1, 1, 4, true><<<A.count, th, s4, s>>>(A);
                break;
            }
            const bool two = lv == 1 && A.count >= 2 * sms;
            const unsigned blocks = two ? (A.count + 1) / 2 : A.count;
            if (grp == 2) {
                if (lv == 2) rotate_k<2, 1, 2><<<blocks, th, s8, s>>>(A);
                else if (two) rotate_k<1, 2, 2><<<blocks, th, s8, s>>>(A);
                else rotate_k<1, 1, 2><<<blocks, th, s4, s>>>(A);
            } else if (grp == 3) {
                if (lv == 2) rotate_k<2, 1, 3><<<blocks, th, s8, s>>>(A);
                else if (two) rotate_k<1, 2, 3><<<blocks, th, s8, s>>>(A);
                else rotate_k<1, 1, 3><<<blocks, th, s4, s>>>(A);
            } else {
                if (lv == 2) rotate_k<2, 1, 4><<<blocks, th, s8, s>>>(A);
                else if (two) rotate_k<1, 2, 4><<<blocks, th, s8, s>>>(A);
                else rotate_k<1, 1, 4><<<blocks, th, s4, s>>>(A);
            }
            break;
        }
        case PC_TFHE_PBS_KEYBUNDLE: {
            const int grp = A.i2 ? A.i2 : 2;
            if (A.n != 2048 || A.c[1] != 1 || grp < 2 || grp > 4 || A.i1 % grp || A.count > 65535) return -1;
            const int sm = 4 * A.n * (int)sizeof(uint32_t);   // pm1
            const dim3 grid(A.i1 / grp, A.count);
            if (grp == 2) keybundle_k<2><<<grid, STEP_TH, sm, s>>>(A);
            else if (grp == 3) keybundle_k<3><<<grid, STEP_TH, sm, s>>>(A);
            else keybundle_k<4><<<grid, STEP_TH, sm, s>>>(A);
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
        case PC_TFHE_REPACK_DIGITS: {
            const size_t tot = (size_t)A.count * A.i1;
            repack_digits_k<<<(unsigned)((tot + 255) / 256), 256, 0, s>>>(A);
            break;
        }
        case PC_TFHE_REPACK_MAC:
            if ((2 * A.i1) % RG || (A.i1 * A.c[1]) % RKC) return -1;
            repack_gemm_k<<<dim3(2 * A.i1 / RG, (A.count + RJ - 1) / RJ, (unsigned)(A.i1 * A.c[1] / RKC)), RG, 0, s>>>(A);
            break;
        case PC_TFHE_REPACK_FIN: {
            const size_t tot = (size_t)A.count * 2 * A.i1;
            repack_fin_k<<<(unsigned)((tot + 255) / 256), 256, 0, s>>>(A);
            break;
        }
        case PC_TFHE_REPACK_SCATTER: {
            const size_t tot = (size_t)A.count * 2 * A.i1;
            repack_scatter_k<<<(unsigned)((tot + 255) / 256), 256, 0, s>>>(A);
            break;
        }
        case PC_TFHE_REPACK_RING:
            repack_ring_k<<<(A.n + 255) / 256, 256, 0, s>>>(A);
            break;
        case PC_TFHE_REPACK_RMAC:
            repack_rmac_k<<<dim3((A.n + 255) / 256, 2), 256, 0, s>>>(A);
            break;
        case PC_TFHE_REPACK_C0:
            repack_c0_k<<<(A.n + 255) / 256, 256, 0, s>>>(A);
            break;
        default:
            return -2;
    }
    return (int)cudaGetLastError();
}
}  // namespace tfhe
}  // namespace pc
