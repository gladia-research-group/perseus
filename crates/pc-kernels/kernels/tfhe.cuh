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
// digit t of the approximate balanced decomposition (pc-tfhe Gadget::decompose; base 2: the non-adjacent form)
__device__ __forceinline__ int64_t digit(uint64_t x, int t, int bl, int lv, uint64_t m) {
    const int bits = bl * lv;
    int64_t v = (int64_t)scale_round(x, m);
    if (v >= (1ll << (bits - 1))) v -= 1ll << bits;
    const int64_t B = 1ll << bl;
    int64_t r = 0;
    for (int s = 0; s <= t; s++) {
        if (bl == 1) {
            r = v & 1 ? 2 - (v & 3) : 0;
        } else {
            r = v & (B - 1);
            if (r >= B / 2) r -= B;
        }
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

// the mod switch to 2 n of one LWE per block (MS_TH threads): masks rounded, the body mean-compensated (pc-tfhe
// gadget::ms_body_mean: round(b m / 2^64 - sum of the masks' rounding residuals / 2), the residuals in 2^-32 units)
constexpr int MS_TH = 256;
__global__ void __launch_bounds__(MS_TH) ms_k(TfheArgs A) {
    const int n = A.i1;
    const size_t base = (size_t)blockIdx.x * (n + 1);
    const uint32_t* y = (const uint32_t*)A.y + base;
    uint32_t* x = (uint32_t*)A.x + base;
    // c[5] = nu: the extended rotation's 2 nu n; c[4] = k: k interleaved LUTs, the switch to 2 nu n / k, times k
    const uint64_t kl = A.c[4] ? A.c[4] : 1, two_n = 2ull * A.n * (A.c[5] ? A.c[5] : 1) / kl, m = A.m;
    long long s = 0;
    for (int i = threadIdx.x; i < n; i += MS_TH) {
        const uint64_t v = y[i], lo = v * m;
        uint64_t hi = __umul64hi(v, m);
        if (lo >= (1ull << 63)) {
            hi += 1;
            s += (long long)((0ull - lo) >> 32);
        } else {
            s -= (long long)((lo + 0xffffffffull) >> 32);
        }
        x[i] = (uint32_t)(kl * (hi % two_n));
    }
    __shared__ long long red[MS_TH];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int k = MS_TH / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const __int128 v = (__int128)((unsigned __int128)y[n] * m) - ((__int128)red[0] << 31) + ((__int128)1 << 63);
        long long r = (long long)(v >> 64) % (long long)two_n;
        x[n] = (uint32_t)(kl * (r < 0 ? r + (long long)two_n : r));
    }
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
// pm1 [prime][2 n] in shared memory, word x at x ^ (bits 7-11 into 0-4): a warp's gathers pm1[E[k] e mod 2 n] over 32
// consecutive k differ only in bits >= 7 (E[k] = 2 brev(k) + 1), all in one bank unswizzled (32-way conflicts)
__device__ __forceinline__ int pswz(int x) { return x ^ ((x >> 7) & 31); }
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

// acc [count][nu][2 part][2 prime][n] (c[5] = nu, 0: 1): part 0 = X^-b~ tv in R_(nu n), component i holding the
// coefficients nu j + i (y = tv residues [2 prime][nu n]), part 1 = 0
__global__ void init_k(TfheArgs A) {
    const int n = A.n, nu = A.c[5] ? (int)A.c[5] : 1, big = nu * n;
    const uint32_t qa = (uint32_t)A.c[6], qb = (uint32_t)(A.c[6] >> 32);
    const uint32_t bt = ((const uint32_t*)A.z)[(size_t)blockIdx.y * (A.i1 + 1) + A.i1];
    const int kp = (int)((2 * big - bt) % (2 * big));
    uint32_t* acc = (uint32_t*)A.x + (size_t)blockIdx.y * nu * 4 * n;
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < big; x += gridDim.x * blockDim.x) {
        const int i = x / n, o = x % n;
        for (int p = 0; p < 2; p++) {
            acc[(i * 4 + p) * n + o] = mono32((const uint32_t*)A.y + p * big, big, kp, nu * o + i, p ? qb : qa);
            acc[(i * 4 + 2 + p) * n + o] = 0;
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
constexpr int CL4_MAX = 46;   // PBS_ROTATE latency path: four-block clusters up to this many LWEs (47 x 4 blocks no longer fit)
// NU = 2 (NB = 2, one LWE per block): the extended rotation (pc-tfhe blind_rotate_ext) at nu = 2, the two slots its
// components; per word the group's bundles A = sum_(e even) (y^(e/2) - 1) K_m - sum_(e odd) K_m and B = sum_(e odd)
// y^((e - 1) / 2) K_m (y = NTT(Y), Y = X^2) give o_0 = A d_0 + B (y d_1), o_1 = A d_1 + B d_0.
template <int LV, int NB, int G, bool KB = false, int NU = 1>
__global__ void __launch_bounds__(STEP_TH) rotate_k(TfheArgs A) {
    static_assert(NU == 1 || (NU == 2 && NB == 2 && LV == 1 && !KB), "the extended rotation: nu = 2 on the two slots");
    constexpr int KM = (1 << G) - 1;   // keys per group
    constexpr int NL = NB / NU;        // LWEs per block
    extern __shared__ uint32_t sm32[];
    const int n = A.n, L = A.i1, bl = (int)A.c[0];
    const int nbv = NU == 1 ? min(NB, A.count - (int)blockIdx.x * NB) : NB;   // slots of this block (the last may hold fewer)
    const uint32_t qa = (uint32_t)A.c[6], qb = (uint32_t)(A.c[6] >> 32), inv = (uint32_t)A.c[7], invs = (uint32_t)(A.c[7] >> 32);
    const size_t tbs = (size_t)4 * LV * n;
    const uint32_t* at[NB];
    uint32_t ar[NB][2][4];   // [lwe][jj][c prime]: coefficient th + 1024 jj
#pragma unroll
    for (int b = 0; b < NB; b++) {
        const int lw = (int)blockIdx.x * NL + min(b, nbv - 1) / NU;
        at[b] = (const uint32_t*)A.z + (size_t)lw * (L + 1);
        const uint32_t* g = (const uint32_t*)A.x + ((size_t)lw * NU + (NU == 1 ? 0 : b)) * 4 * n;
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
    const int bits = bl * LV, n2 = 2 * n, n2e = NU * n2;   // the exponents live mod 2 nu n
    // the groups [g0, g0 + gw) (i0, i4; i4 = 0: all): the latency path runs its windows of key bundles one at a time
    const int g0 = A.i0, gw = A.i4 ? A.i4 : L / G - A.i0;
    // pm1 [prime][2 n] in shared memory after the digits: its gathers (data-dependent, 2^G - 1 per word) sat on the
    // MAC's critical path from L2
    uint32_t* pm1 = sm32 + NB * tbs;
    for (int x = threadIdx.x; x < 2 * n2; x += blockDim.x) pm1[(x & ~(n2 - 1)) + pswz(x & (n2 - 1))] = __ldg(tw + 4 * n + x);
    const int64_t B = 1ll << bl;
    for (int pr = g0; pr < g0 + gw; pr++) {
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
                em[b][m - 1] = on ? (int)((G * n2e - e) % n2e) : 0;
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
                // the key bundle precomputed by PBS_KEYBUNDLE (w: [lwe][window group][e][thread] uint4, NB = LV = 1)
                const uint4 g = __ldg((const uint4*)A.w + (((size_t)blockIdx.x * gw + pr - g0) * RT_E + e) * blockDim.x + threadIdx.x);
                const uint64_t s0 = (uint64_t)d[0][0] * g.x + (uint64_t)d[0][1] * g.z;
                const uint64_t s1 = (uint64_t)d[0][0] * g.y + (uint64_t)d[0][1] * g.w;
                sm32[swz(k)] = redc(s0, q, mi);   // < 2 q
                sm32[2 * n + swz(k)] = redc(s1, q, mi);
            } else if constexpr (NU == 2) {
                const uint32_t r1 = (uint32_t)((1ull << 32) % q), mone = q - r1;   // 1, -1 in Montgomery form
                uint64_t ga[2][2] = {}, gb[2][2] = {};   // the bundles A, B at this word: [ct][part], KM terms < q q
                uint4 wn = __ldg(bsk + e * blockDim.x);   // one m ahead (as the generic MAC)
    #pragma unroll
                for (int m = 0; m < (RT_SKIP == 1 ? 0 : KM); m++) {
                    const uint4 w = wn;
                    if (m + 1 < KM) wn = __ldg(bsk + ((m + 1) * RT_E + e) * blockDim.x);
                    const int x = em[0][m];   // the exponent, mod 2 nu n
                    if (!x) continue;
                    const uint32_t pj = pm1[p * n2 + pswz((ej * (x >> 1)) & (n2 - 1))];   // y^j - 1, j = x div 2
                    const uint64_t fa = x & 1 ? mone : pj, fb = x & 1 ? add32(pj, r1, q) : 0;
                    ga[0][0] += w.x * fa, ga[0][1] += w.y * fa, ga[1][0] += w.z * fa, ga[1][1] += w.w * fa;
                    gb[0][0] += w.x * fb, gb[0][1] += w.y * fb, gb[1][0] += w.z * fb, gb[1][1] += w.w * fb;
                }
                const uint32_t yk = add32(pm1[p * n2 + pswz(ej & (n2 - 1))], r1, q);   // y at this word
                uint32_t ra[2][2], rb[2][2], d1y[2];
    #pragma unroll
                for (int ct = 0; ct < 2; ct++) {
                    d1y[ct] = fold32(redc((uint64_t)d[1][ct] * yk, q, mi), q);
    #pragma unroll
                    for (int pt = 0; pt < 2; pt++) ra[ct][pt] = fold32(redc(ga[ct][pt], q, mi), q), rb[ct][pt] = fold32(redc(gb[ct][pt], q, mi), q);
                }
                // o_0 = A d_0 + B (y d_1), o_1 = A d_1 + B d_0; two terms per reduction (< 46 q q < q 2^32)
    #pragma unroll
                for (int pt = 0; pt < 2; pt++) {
                    const uint32_t a0 = redc((uint64_t)d[0][0] * ra[0][pt] + (uint64_t)d[0][1] * ra[1][pt], q, mi);
                    const uint32_t b0 = redc((uint64_t)d1y[0] * rb[0][pt] + (uint64_t)d1y[1] * rb[1][pt], q, mi);
                    const uint32_t a1 = redc((uint64_t)d[1][0] * ra[0][pt] + (uint64_t)d[1][1] * ra[1][pt], q, mi);
                    const uint32_t b1 = redc((uint64_t)d[0][0] * rb[0][pt] + (uint64_t)d[0][1] * rb[1][pt], q, mi);
                    sm32[swz(k) + pt * 2 * n] = fold32(a0 + b0, 2 * q);   // < 2 q
                    sm32[tbs + swz(k) + pt * 2 * n] = fold32(a1 + b1, 2 * q);
                }
            } else {
                // the group's key first (TFHE-rs's key bundle, in NTT form): G[ct][part] = sum_m f_m K_m[ct][part], f_m =
                // NTT(X^-e_m - 1) (KM <= 7 terms < q q: < q 2^32), then one product with the digits: 2 KM LV + 2 LV wide
                // products and 2 LV + 2 reductions per word instead of KM (2 LV + 2) and 2 KM + 2
                uint64_t gk[NB][2 * LV][2];   // KM <= 15 terms < q q: < q 2^32 at q < 2^28
    #pragma unroll
                for (int b = 0; b < NB; b++)
    #pragma unroll
                    for (int ct = 0; ct < 2 * LV; ct++) gk[b][ct][0] = gk[b][ct][1] = 0;
                // the key words one m ahead: a load's L2 latency hides behind the MACs of the one before
                uint4 wn[LV];
    #pragma unroll
                for (int h = 0; h < LV; h++) wn[h] = __ldg(bsk + (e * LV + h) * blockDim.x);
    #pragma unroll
                for (int m = 0; m < (RT_SKIP == 1 ? 0 : KM); m++) {
                    uint4 w[LV];
    #pragma unroll
                    for (int h = 0; h < LV; h++) {
                        w[h] = wn[h];
                        if (m + 1 < KM) wn[h] = __ldg(bsk + (((m + 1) * RT_E + e) * LV + h) * blockDim.x);
                    }
    #pragma unroll
                    for (int b = 0; b < NB; b++) {
                        if (!em[b][m]) continue;
                        const uint64_t f = pm1[p * n2 + pswz((ej * em[b][m]) & (n2 - 1))];
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
                    // per level, 2 terms < 23 q q: < q 2^32 (the launcher checks 46 q < 2^32); levels add reduced
                    uint32_t o0 = 0, o1 = 0;
    #pragma unroll
                    for (int h = 0; h < LV; h++) {
                        uint64_t s0 = 0, s1 = 0;
    #pragma unroll
                        for (int ct = 2 * h; ct < 2 * h + 2; ct++) {
                            s0 += (uint64_t)d[b][ct] * fold32(redc(gk[b][ct][0], q, mi), q);
                            s1 += (uint64_t)d[b][ct] * fold32(redc(gk[b][ct][1], q, mi), q);
                        }
                        o0 = h ? fold32(o0 + redc(s0, q, mi), 2 * q) : redc(s0, q, mi);   // < 2 q
                        o1 = h ? fold32(o1 + redc(s1, q, mi), 2 * q) : redc(s1, q, mi);
                    }
                    sm32[b * tbs + swz(k)] = o0;
                    sm32[b * tbs + 2 * n + swz(k)] = o1;
    
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

// Thread-block clusters need sm_90+: below it (the image's sm_89 pass) the cluster kernels compile to a trap and
// the launcher takes the one-block path (tfhe_clusters).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define PC_CLUSTER_DIMS(x)
#else
#define PC_CLUSTER_DIMS(x) __cluster_dims__(x, 1, 1)
#endif

// PBS_ROTATE's latency path on a cluster of two blocks per LWE, one per RNS prime (r = the block's rank, on the key
// bundles of PBS_KEYBUNDLE, one level): each block keeps its prime's acc [2 part][n] in shared memory, double-buffered
// by step parity, and reads the partner's through distributed shared memory for the digits (the CRT of both
// residues). Per group: cluster sync (the partner's acc is final), digits of both parts at its prime, forward NTT of
// the two digit polys, the MAC on its half of the bundle words, inverse NTT, acc' = acc + o into the other parity
// (the partner may still read this one: no second sync). Shared: acc 2 x 2 n, tb 2 n, this prime's twiddles and
// companions 4 n words.
template <int G>
__global__ void PC_CLUSTER_DIMS(2) __launch_bounds__(STEP_TH) rotate_cl_k(TfheArgs A) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    __trap();   // no clusters below sm_90: the launcher never picks this kernel there
#else
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
    const int g0 = A.i0, gw = A.i4 ? A.i4 : L / G - A.i0;   // the window of groups (as rotate_k)
    for (int pr = g0; pr < g0 + gw; pr++) {
        bool nz = false;
#pragma unroll
        for (int x = 0; x < G; x++) nz |= at[G * pr + x] != 0;
        if (!nz) continue;   // the same in both blocks
        // the bundle words first: their L2 latency hides behind the digits and the forward NTT
        uint4 w[2];
#pragma unroll
        for (int ee = 0; ee < 2; ee++)
            w[ee] = __ldg((const uint4*)A.w + (((size_t)lw * gw + pr - g0) * RT_E + 2 * r + ee) * blockDim.x + threadIdx.x);
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
#endif
}

// PBS_ROTATE's latency path on a cluster of four blocks per LWE, block rank 2 c + r: RNS prime r, acc part c (n = 2048,
// 1024 threads, two positions each). Layouts P1: j = x 2^10 + l 2^5 + w, P2: j = (l >> 4) 2^10 + w 2^5 + y 2^4 + (l & 15),
// one shared-memory transpose between them; the partners' copies are read through distributed shared memory.
__device__ __forceinline__ int sw(int j) { return j ^ ((j >> 5) & 31) ^ ((j >> 10) << 4); }
// CT butterfly over lane bit lb (partner lane l ^ 2^lb): the lane with the bit set holds x[j + t]
__device__ __forceinline__ uint32_t ct_shfl(uint32_t me, bool hi, int lb, uint32_t w, uint32_t ws, uint32_t q) {
    const uint32_t t = hi ? shoup32l(me, w, ws, q) : me;
    const uint32_t o = __shfl_xor_sync(0xffffffffu, t, 1 << lb);
    return hi ? o - t + 2 * q : t + o;
}
// GS butterfly over lane bit lb
__device__ __forceinline__ uint32_t gs_shfl(uint32_t me, bool hi, int lb, uint32_t w, uint32_t ws, uint32_t q) {
    const uint32_t o = __shfl_xor_sync(0xffffffffu, me, 1 << lb);
    return hi ? shoup32l(o - me + 2 * q, w, ws, q) : fold32(me + o, 2 * q);
}
// NU = 2 (c[5]): the extended rotation, both components per block (bundles A, B as rotate_k); LV = 2: two gadget
// levels, the bundle's two uint4 per word the decomposed parts p.
template <int G, int NU = 1, int LV = 1>
__global__ void PC_CLUSTER_DIMS(4) __launch_bounds__(STEP_TH) rotate_cl4_k(TfheArgs A) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    __trap();   // no clusters below sm_90: the launcher never picks this kernel there
#else
    extern __shared__ uint32_t sm32[];
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    constexpr int n = RN, NF = NU * LV, NK = NU > LV ? NU : LV;   // forward polys; uint4 bundle words per position
    static_assert(NU == 1 || LV == 1, "the extended rotation at one level");
    const int rk = (int)cl.block_rank(), r = rk & 1, c = rk >> 1;
    const int L = A.i1, bl = (int)A.c[0], lw = (int)blockIdx.x / 4;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const uint32_t qa = (uint32_t)A.c[6], qb = (uint32_t)(A.c[6] >> 32), inv = (uint32_t)A.c[7], invs = (uint32_t)(A.c[7] >> 32);
    const uint32_t q = r ? qb : qa, mi = mont_neg_inv(q);
    // acc copy [NU][n], digit copy [NF][n], transpose buffers [NF][n] and [NU][n]; then twiddles and companions 2 n each
    uint32_t *acp = sm32, *dgp = sm32 + NU * n, *ta = dgp + NF * n, *tb = ta + NF * n, *tw = tb + NU * n, *tws = tw + 2 * n;
    const uint32_t* pacc = cl.map_shared_rank(acp, rk ^ 1);   // the other prime's acc part c
    const uint32_t* pdg = cl.map_shared_rank(dgp, rk ^ 2);    // the other part's digits at prime r
    uint32_t* g = (uint32_t*)A.x + (size_t)lw * NU * 4 * n + (size_t)(2 * c + r) * n;   // [component][part][prime][n]
    int j1[2], j2[2];
#pragma unroll
    for (int x = 0; x < 2; x++) {
        j1[x] = (x << 10) | (l << 5) | w;
        j2[x] = ((l >> 4) << 10) | (w << 5) | (x << 4) | (l & 15);
    }
    uint32_t ac[NU][2], v[NF][2], yk[2] = {};
#pragma unroll
    for (int i = 0; i < NU; i++)
#pragma unroll
        for (int x = 0; x < 2; x++) {
            ac[i][x] = g[i * 4 * n + j1[x]];
            acp[i * n + x * 1024 + threadIdx.x] = ac[i][x];
        }
    for (int x = threadIdx.x; x < 2 * n; x += blockDim.x) {
        tw[x] = __ldg((const uint32_t*)A.tw + (size_t)r * 2 * n + x);
        tws[x] = __ldg((const uint32_t*)A.tws + (size_t)r * 2 * n + x);
    }
    if constexpr (NU == 2) {   // y = NTT(Y) at the P2 positions: pm1 [prime][2 n] + 1, E [prime][n] (PBS_ROTATE's tables)
        const uint32_t* t = (const uint32_t*)A.tw;
#pragma unroll
        for (int y = 0; y < 2; y++) yk[y] = add32(__ldg(t + 4 * n + r * 2 * n + __ldg(t + 8 * n + r * n + j2[y])), (uint32_t)((1ull << 32) % q), q);
    }
    const uint32_t* at = (const uint32_t*)A.z + (size_t)lw * (L + 1);
    const int64_t B = 1ll << bl;
    const int g0 = A.i0, gw = A.i4 ? A.i4 : L / G - A.i0;   // the window of groups (as rotate_k)
    for (int pr = g0; pr < g0 + gw; pr++) {
        bool nz = false;
#pragma unroll
        for (int x = 0; x < G; x++) nz |= at[G * pr + x] != 0;
        if (!nz) continue;   // the same in every block
        // the bundle words (G[0][c], G[1][c]; NU = 2: of A and of B) of the P2 positions first: their L2 latency hides
        // behind the forward NTT
        uint32_t k0[NK][2], k1[NK][2];
        const uint4* bw = (const uint4*)A.w + ((size_t)lw * gw + pr - g0) * RT_E * NK * blockDim.x;
#pragma unroll
        for (int ab = 0; ab < NK; ab++)
#pragma unroll
            for (int y = 0; y < 2; y++) {
                const uint4 kk = __ldg(bw + ((2 * r + (j2[y] >> 10)) * NK + ab) * blockDim.x + (j2[y] & 1023));
                k0[ab][y] = c ? kk.y : kk.x;
                k1[ab][y] = c ? kk.w : kk.z;
            }
        cl.sync();
#pragma unroll
        for (int i = 0; i < NF; i++) {
            if (i % LV == 0)   // the digits of component i / LV, both levels (interleaved with the components' NTTs)
#pragma unroll
                for (int x = 0; x < 2; x++) {
                    const uint32_t oth = pacc[(i / LV) * n + x * 1024 + threadIdx.x];
                    const uint32_t own = ac[i / LV][x];
                    int64_t d = (int64_t)scale_round(crt(r ? oth : own, r ? own : oth, qa, qb, inv, invs), A.m);
                    if constexpr (LV == 1) {
                        if (d >= B / 2) d -= B;   // one level: the digit is the centred value
                        v[i][x] = d < 0 ? (uint32_t)((int64_t)q + d) : (uint32_t)d;
                    } else {
                        if (d >= (1ll << (LV * bl - 1))) d -= 1ll << (LV * bl);   // centred in 2^(LV b)
#pragma unroll
                        for (int t = 0; t < LV; t++) {   // balanced digits, the edge carry into the top one (rotate_k)
                            int64_t rr = d & (B - 1);
                            if (rr >= B / 2) rr -= B;
                            d = (d - rr) >> bl;
                            if (t == LV - 1) rr += d << bl;
                            v[i + t][x] = rr < 0 ? (uint32_t)((int64_t)q + rr) : (uint32_t)rr;
                        }
                    }
                }
            {   // forward: bit 10 (registers), bits 9..5 (lanes) in P1
                const uint32_t u = v[i][0], t = shoup32l(v[i][1], tw[1], tws[1], q);
                v[i][0] = u + t;
                v[i][1] = u - t + 2 * q;
            }
#pragma unroll
            for (int b = 9; b >= 5; b--) {
                const bool hi = (l >> (b - 5)) & 1;
#pragma unroll
                for (int x = 0; x < 2; x++) {
                    const int wi = (n >> (b + 1)) + (j1[x] >> (b + 1));
                    v[i][x] = ct_shfl(v[i][x], hi, b - 5, tw[wi], tws[wi], q);
                }
            }
#pragma unroll
            for (int x = 0; x < 2; x++) ta[i * n + sw(j1[x])] = v[i][x];
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < NF; i++) {
#pragma unroll
            for (int y = 0; y < 2; y++) v[i][y] = ta[i * n + sw(j2[y])];
            {   // bit 4 (registers), bits 3..0 (lanes) in P2
                const int wi = (n >> 5) + (j2[0] >> 5);
                const uint32_t u = v[i][0], t = shoup32l(v[i][1], tw[wi], tws[wi], q);
                v[i][0] = u + t;
                v[i][1] = u - t + 2 * q;
            }
#pragma unroll
            for (int b = 3; b >= 0; b--) {
                const bool hi = (l >> b) & 1;
#pragma unroll
                for (int y = 0; y < 2; y++) {
                    const int wi = (n >> (b + 1)) + (j2[y] >> (b + 1));
                    v[i][y] = ct_shfl(v[i][y], hi, b, tw[wi], tws[wi], q);
                }
            }
#pragma unroll
            for (int y = 0; y < 2; y++) dgp[i * n + y * 1024 + threadIdx.x] = v[i][y];
        }
        cl.sync();
        // the MAC: o[c] = d0 K[0][c] + d1 K[1][c] (NU = 2: per component, A and B as above)
        if constexpr (NU == 1 && LV == 1) {
#pragma unroll
            for (int y = 0; y < 2; y++) {
                const uint64_t od = pdg[y * 1024 + threadIdx.x];
                const uint64_t d0 = c ? od : v[0][y], d1 = c ? v[0][y] : od;
                v[0][y] = redc(d0 * k0[0][y] + d1 * k1[0][y], q, mi);   // < 2 q
            }
        } else if constexpr (LV == 2) {
            // o[c] = sum_(p, t) d[p][t] G[p][t][c]: bundle word p holds both levels of the decomposed part p
#pragma unroll
            for (int y = 0; y < 2; y++) {
                uint64_t dp[2][2];   // [part p][level]
#pragma unroll
                for (int t = 0; t < 2; t++) {
                    const uint64_t od = pdg[t * n + y * 1024 + threadIdx.x];
                    dp[c][t] = v[t][y];
                    dp[1 - c][t] = od;
                }
                const uint32_t a0 = redc(dp[0][0] * k0[0][y] + dp[0][1] * k1[0][y], q, mi);
                const uint32_t a1 = redc(dp[1][0] * k0[1][y] + dp[1][1] * k1[1][y], q, mi);
                v[0][y] = fold32(a0 + a1, 2 * q);   // < 2 q
            }
        } else {
#pragma unroll
            for (int y = 0; y < 2; y++) {
                uint64_t d[2][2];   // [component][ct]
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const uint64_t od = pdg[i * n + y * 1024 + threadIdx.x];
                    d[i][0] = c ? od : v[i][y];
                    d[i][1] = c ? v[i][y] : od;
                }
                const uint64_t y0 = fold32(redc(d[1][0] * yk[y], q, mi), q), y1 = fold32(redc(d[1][1] * yk[y], q, mi), q);
                const uint32_t a0 = redc(d[0][0] * k0[0][y] + d[0][1] * k1[0][y], q, mi), b0 = redc(y0 * k0[1][y] + y1 * k1[1][y], q, mi);
                const uint32_t a1 = redc(d[1][0] * k0[0][y] + d[1][1] * k1[0][y], q, mi), b1 = redc(d[0][0] * k0[1][y] + d[0][1] * k1[1][y], q, mi);
                v[0][y] = fold32(a0 + b0, 2 * q);   // < 2 q
                v[1][y] = fold32(a1 + b1, 2 * q);
            }
        }
#pragma unroll
        for (int i = 0; i < NU; i++) {
            // inverse: bits 0..3 (lanes), bit 4 (registers) in P2
#pragma unroll
            for (int b = 0; b <= 3; b++) {
                const bool hi = (l >> b) & 1;
#pragma unroll
                for (int y = 0; y < 2; y++) {
                    const int wi = n + (n >> (b + 1)) + (j2[y] >> (b + 1));
                    v[i][y] = gs_shfl(v[i][y], hi, b, tw[wi], tws[wi], q);
                }
            }
            {
                const int wi = n + (n >> 5) + (j2[0] >> 5);
                const uint32_t u = v[i][0], t = v[i][1];
                v[i][0] = fold32(u + t, 2 * q);
                v[i][1] = shoup32l(u - t + 2 * q, tw[wi], tws[wi], q);
            }
#pragma unroll
            for (int y = 0; y < 2; y++) tb[i * n + sw(j2[y])] = v[i][y];
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < NU; i++) {
#pragma unroll
            for (int x = 0; x < 2; x++) v[i][x] = tb[i * n + sw(j1[x])];
            // bits 5..9 (lanes), bit 10 (registers) in P1
#pragma unroll
            for (int b = 5; b <= 9; b++) {
                const bool hi = (l >> (b - 5)) & 1;
#pragma unroll
                for (int x = 0; x < 2; x++) {
                    const int wi = n + (n >> (b + 1)) + (j1[x] >> (b + 1));
                    v[i][x] = gs_shfl(v[i][x], hi, b - 5, tw[wi], tws[wi], q);
                }
            }
            {
                const uint32_t u = v[i][0], t = v[i][1];
                v[i][0] = fold32(u + t, 2 * q);
                v[i][1] = shoup32l(u - t + 2 * q, tw[n + 1], tws[n + 1], q);
            }
#pragma unroll
            for (int x = 0; x < 2; x++) {
                ac[i][x] = add32(ac[i][x], fold32(v[i][x], q), q);
                acp[i * n + x * 1024 + threadIdx.x] = ac[i][x];
            }
        }
    }
    cl.sync();   // the partners may still read this block's copies
#pragma unroll
    for (int i = 0; i < NU; i++)
#pragma unroll
        for (int x = 0; x < 2; x++) g[i * 4 * n + j1[x]] = ac[i][x];
#endif
}

// PBS_KEYBUNDLE: the key bundle of every (LWE, group) for the latency path (TFHE-rs's multi-bit keybundle): x
// [count][group][e][thread] uint4 = (G[0][0], G[0][1], G[1][0], G[1][1]), G[c][part] = sum_m f_m K_m[c][part] (Montgomery
// form, < q; one level), the MAC of rotate_k computed for a window of groups at once (i0 = the first, i4 = how many),
// off the rotation's sequential path. y = bsk, z = the mod-switched LWEs, tw as PBS_ROTATE; grid (window, count),
// 1024 threads; x [count][window group][e][thread]. NU = 2 (c[5]) / LV = 2 (c[1]): two uint4 per word,
// x [..][e][A | B][thread] / x [..][e][p][thread].
template <int G, int NU = 1, int LV = 1>
__global__ void __launch_bounds__(STEP_TH) keybundle_k(TfheArgs A) {
    extern __shared__ uint32_t sm32[];
    constexpr int KM = (1 << G) - 1;
    const int n = A.n, L = A.i1, n2 = 2 * n, n2e = NU * n2, pr = A.i0 + blockIdx.x;
    const uint32_t qa = (uint32_t)A.c[6], qb = (uint32_t)(A.c[6] >> 32);
    const uint32_t mia = mont_neg_inv(qa), mib = mont_neg_inv(qb);
    const uint32_t* tw = (const uint32_t*)A.tw;
    const uint32_t* ex = tw + 8 * n;
    uint32_t* pm1 = sm32;
    for (int x = threadIdx.x; x < 2 * n2; x += blockDim.x) pm1[(x & ~(n2 - 1)) + pswz(x & (n2 - 1))] = __ldg(tw + 4 * n + x);
    const uint32_t* at = (const uint32_t*)A.z + (size_t)blockIdx.y * (L + 1);
    int em[KM];
#pragma unroll
    for (int m = 1; m <= KM; m++) {
        uint32_t e = 0;
#pragma unroll
        for (int x = 0; x < G; x++)
            if (m >> x & 1) e += at[G * pr + x];
        em[m - 1] = (int)((G * n2e - e) % n2e);
    }
    __syncthreads();
    constexpr int NK = NU > LV ? NU : LV;   // uint4 per word
    const size_t kstride = (size_t)2 * LV * 2 * 2 * n;   // one GGSW
    const uint4* bsk = (const uint4*)((const uint32_t*)A.y + (size_t)pr * KM * kstride) + threadIdx.x;
    uint4* out = (uint4*)A.x + ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * RT_E * NK * blockDim.x + threadIdx.x;
#pragma unroll
    for (int e = 0; e < RT_E; e++) {
        const int k = threadIdx.x + e * blockDim.x, p = k >= n;
        const uint32_t q = p ? qb : qa, mi = p ? mib : mia;
        const int ej = (int)ex[k];
        if constexpr (LV == 2) {
            uint64_t gh[2][4] = {};
#pragma unroll
            for (int m = 0; m < KM; m++) {
                if (!em[m]) continue;
                const uint64_t f = pm1[p * n2 + pswz((ej * em[m]) & (n2 - 1))];
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const uint4 w = __ldg(bsk + ((m * RT_E + e) * 2 + h) * blockDim.x);
                    gh[h][0] += w.x * f, gh[h][1] += w.y * f, gh[h][2] += w.z * f, gh[h][3] += w.w * f;
                }
            }
            auto r = [&](uint64_t v) { return fold32(redc(v, q, mi), q); };
#pragma unroll
            for (int h = 0; h < 2; h++) out[(2 * e + h) * blockDim.x] = make_uint4(r(gh[h][0]), r(gh[h][1]), r(gh[h][2]), r(gh[h][3]));
            continue;
        }
        if constexpr (NU == 2) {
            const uint32_t r1 = (uint32_t)((1ull << 32) % q), mone = q - r1;   // 1, -1 in Montgomery form
            uint64_t ga[4] = {}, gb[4] = {};
#pragma unroll
            for (int m = 0; m < KM; m++) {
                const int x = em[m];
                if (!x) continue;
                const uint4 w = __ldg(bsk + (m * RT_E + e) * blockDim.x);
                const uint32_t pj = pm1[p * n2 + pswz((ej * (x >> 1)) & (n2 - 1))];
                const uint64_t fa = x & 1 ? mone : pj, fb = x & 1 ? add32(pj, r1, q) : 0;
                ga[0] += w.x * fa, ga[1] += w.y * fa, ga[2] += w.z * fa, ga[3] += w.w * fa;
                gb[0] += w.x * fb, gb[1] += w.y * fb, gb[2] += w.z * fb, gb[3] += w.w * fb;
            }
            auto r = [&](uint64_t v) { return fold32(redc(v, q, mi), q); };
            out[2 * e * blockDim.x] = make_uint4(r(ga[0]), r(ga[1]), r(ga[2]), r(ga[3]));
            out[(2 * e + 1) * blockDim.x] = make_uint4(r(gb[0]), r(gb[1]), r(gb[2]), r(gb[3]));
            continue;
        }
        uint64_t g00 = 0, g01 = 0, g10 = 0, g11 = 0;   // KM <= 15 terms < q q
#pragma unroll
        for (int m = 0; m < KM; m++) {
            if (!em[m]) continue;
            const uint4 w = __ldg(bsk + (m * RT_E + e) * blockDim.x);
            const uint64_t f = pm1[p * n2 + pswz((ej * em[m]) & (n2 - 1))];
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
    const uint32_t* acc = (const uint32_t*)A.y + (size_t)blockIdx.y * (A.c[5] ? A.c[5] : 1) * 4 * n;   // component 0
    uint32_t* out = (uint32_t*)A.x + (size_t)blockIdx.y * (n + 1);
    const int j = (int)A.c[4];   // the coefficient (many-LUT: LUT j)
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k <= n; k += gridDim.x * blockDim.x) {
        uint64_t v;
        if (k == n) {
            v = addq(crt(acc[j], acc[n + j], qa, qb, inv, invs), A.c[0], Q);
        } else if (k <= j) {
            v = crt(acc[2 * n + j - k], acc[3 * n + j - k], qa, qb, inv, invs);
        } else {
            const uint64_t c1 = crt(acc[2 * n + n + j - k], acc[3 * n + n + j - k], qa, qb, inv, invs);
            v = c1 ? Q - c1 : 0;
        }
        out[k] = (uint32_t)(scale_round(v, A.m) % to);
    }
}

__global__ void lwe_lin_k(TfheArgs A) {
    const int len = A.i1 + 1;
    const int2 ix = ((const int2*)A.w)[blockIdx.y];
    const uint64_t q = A.q;
    uint32_t* out = (uint32_t*)A.x + (size_t)blockIdx.y * len;
    const uint32_t* a = ix.x >= 0 ? (const uint32_t*)A.y + (size_t)ix.x * len : nullptr;
    const uint32_t* b = ix.y >= 0 ? (const uint32_t*)A.z + (size_t)ix.y * len : nullptr;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < len; k += gridDim.x * blockDim.x) {
        const uint64_t u = a ? (uint64_t)a[k] * A.c[0] % q : 0, v = b ? (uint64_t)b[k] * A.c[1] % q : 0;
        const uint64_t r = addq(u, v, q);
        out[k] = (uint32_t)(k == len - 1 ? addq(r, A.c[2], q) : r);
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
    // c[4]: the GEMM's row count (>= count: padded rows), c[3]: a padded plane stride (words)
    const size_t rows = A.c[4] ? (size_t)A.c[4] : (size_t)A.count;
    const size_t plane = A.c[3] ? (size_t)A.c[3] : (size_t)cols * rows;
    int64_t v = 0;
    for (int p = planes - 1; p >= 0; p--) v = v * 256 + g[p * plane + o * rows + b];
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
// REPACK_DIGITS: the exact digits of the LWEs' a (y = [count][n_br + 1] mod p0) as two balanced int8 planes x [2][mp]
// [n_br levels] (k t; mp = c[5] >= count rows, the GEMM's; digits |d| < 2^14: base 2^15); also zeroes the sums w
// ([count][2 limb][2 n_br] i64) and, when i0 = 1, the scatter's z ([2 part][n] u64)
__global__ void repack_digits_k(TfheArgs A) {
    const int nb = A.i1, lv = (int)A.c[1], bl = (int)A.c[0];
    const size_t tot = (size_t)A.count * nb, gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    if (A.i0)
        for (size_t z = gid; z < 2 * (size_t)A.n; z += stride) ((uint64_t*)A.z)[z] = 0;
    for (size_t z = gid; z < tot * 4; z += stride) ((uint64_t*)A.w)[z] = 0;
    if (gid >= tot) return;
    const size_t b = gid / nb, k = gid % nb, K = (size_t)nb * lv, mp = A.c[5];
    const int64_t v = cent(((const uint32_t*)A.y)[b * (nb + 1) + k], A.q);
    int8_t* out = (int8_t*)A.x;
    for (int t = 0; t < lv; t++) {
        const int64_t d = digit_exact(v, t, bl, lv);
        const int64_t lo = ((d + 128) % 256 + 256) % 256 - 128;
        out[b * K + k * lv + t] = (int8_t)lo;
        out[mp * K + b * K + k * lv + t] = (int8_t)((d - lo) / 256);
    }
}

// REPACK_COMB: sums x [count][4 n_br] (i64) += G (y, [4 n_br][mp] i32: an int8 GEMM's output, i3 = mp) 2^(8 c[5])
__global__ void repack_comb_k(TfheArgs A) {
    const size_t cols = 4 * (size_t)A.i1, mp = A.i3;
    const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (size_t)A.count * cols) return;
    const size_t j = gid / cols, col = gid % cols;
    ((int64_t*)A.x)[gid] += (int64_t)((const int32_t*)A.y)[col * mp + j] << (8 * A.c[5]);
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

// Thread-block clusters (sm_90+) on the current device, queried once: without them PBS_ROTATE's latency path runs one
// block per LWE, and the cluster-only variants (two gadget levels, the extended rotation, groups of 5) are refused.
static bool tfhe_clusters() {
    static const bool c = [] {
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        return major >= 9;
    }();
    return c;
}

int launch(const TfheArgs* a, cudaStream_t s) {
    const TfheArgs A = *a;
    if (A.count < 1) return -1;
    if ((A.op <= PC_TFHE_PBS_EXTRACT || A.op == PC_TFHE_PBS_ROTATE) && (A.n < 64 || A.n > 2048 || (A.n & (A.n - 1)))) return -1;
    switch (A.op) {
        case PC_TFHE_PBS_MS: {
            ms_k<<<(unsigned)A.count, MS_TH, 0, s>>>(A);
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
            if (A.n != 2048 || (lv != 1 && lv != 2) || 46ull * qmax >= (1ull << 32)) return -1;
            const int grp = A.i2 ? A.i2 : 2;
            const bool cl = tfhe_clusters();
            // groups of 5 (31 keys per bundle): the one-level four-block latency path only
            if (grp < 2 || grp > 5 || A.i1 % grp || (grp == 5 && (!A.w || lv != 1 || A.c[5] > 1 || A.count > CL4_MAX || A.i3 == 1 || A.i3 == 2 || !cl))) return -1;
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
            if (A.c[5] > 1 && A.w) {   // the extended rotation's latency path: four-block clusters, both components each
                if (A.c[5] != 2 || lv != 1 || A.count > CL4_MAX || !cl) return -1;
                const int sc = 12 * A.n * (int)sizeof(uint32_t);
                static bool attr_e4 = false;
                if (!attr_e4) {
                    cudaFuncSetAttribute(rotate_cl4_k<2, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, sc);
                    cudaFuncSetAttribute(rotate_cl4_k<3, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, sc);
                    cudaFuncSetAttribute(rotate_cl4_k<4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, sc);
                    attr_e4 = true;
                }
                if (grp == 2) rotate_cl4_k<2, 2><<<4 * A.count, th, sc, s>>>(A);
                else if (grp == 3) rotate_cl4_k<3, 2><<<4 * A.count, th, sc, s>>>(A);
                else rotate_cl4_k<4, 2><<<4 * A.count, th, sc, s>>>(A);
                break;
            }
            if (A.c[5] > 1) {   // the extended rotation (c[5] = nu = 2): one LWE per block, its components on the two slots
                if (A.c[5] != 2 || lv != 1) return -1;
                static bool attr_e = false;
                if (!attr_e) {
                    cudaFuncSetAttribute(rotate_k<1, 2, 2, false, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, s8);
                    cudaFuncSetAttribute(rotate_k<1, 2, 3, false, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, s8);
                    cudaFuncSetAttribute(rotate_k<1, 2, 4, false, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, s8);
                    attr_e = true;
                }
                if (grp == 2) rotate_k<1, 2, 2, false, 2><<<A.count, th, s8, s>>>(A);
                else if (grp == 3) rotate_k<1, 2, 3, false, 2><<<A.count, th, s8, s>>>(A);
                else rotate_k<1, 2, 4, false, 2><<<A.count, th, s8, s>>>(A);
                break;
            }
            if (A.w) {   // the latency path: the key bundles precomputed (PBS_KEYBUNDLE), one LWE per block, one level
                static bool attr_k = false;
                if (!attr_k) {
                    cudaFuncSetAttribute(rotate_k<1, 1, 2, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, s4);
                    cudaFuncSetAttribute(rotate_k<1, 1, 3, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, s4);
                    cudaFuncSetAttribute(rotate_k<1, 1, 4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, s4);
                    attr_k = true;
                }
                if (lv == 2) {   // two gadget levels: four-block clusters, both levels of the block's part
                    if (A.count > CL4_MAX || !cl) return -1;
                    const int sc = 12 * A.n * (int)sizeof(uint32_t);
                    static bool attr_l2 = false;
                    if (!attr_l2) {
                        cudaFuncSetAttribute(rotate_cl4_k<2, 1, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, sc);
                        cudaFuncSetAttribute(rotate_cl4_k<3, 1, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, sc);
                        cudaFuncSetAttribute(rotate_cl4_k<4, 1, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, sc);
                        attr_l2 = true;
                    }
                    if (grp == 2) rotate_cl4_k<2, 1, 2><<<4 * A.count, th, sc, s>>>(A);
                    else if (grp == 3) rotate_cl4_k<3, 1, 2><<<4 * A.count, th, sc, s>>>(A);
                    else rotate_cl4_k<4, 1, 2><<<4 * A.count, th, sc, s>>>(A);
                    break;
                }
                if (lv != 1) return -1;
                // i3: 0 = by batch (four blocks per LWE up to CL4_MAX LWEs, else two), 1 = one block, 2 / 4 = that cluster
                const int i3 = cl ? A.i3 : 1;
                if (i3 == 4 || (i3 == 0 && A.count <= CL4_MAX)) {
                    const int sc = 8 * A.n * (int)sizeof(uint32_t);
                    static bool attr_c4 = false;
                    if (!attr_c4) {
                        cudaFuncSetAttribute(rotate_cl4_k<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 2048 * (int)sizeof(uint32_t));
                        cudaFuncSetAttribute(rotate_cl4_k<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 2048 * (int)sizeof(uint32_t));
                        cudaFuncSetAttribute(rotate_cl4_k<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 2048 * (int)sizeof(uint32_t));
                        cudaFuncSetAttribute(rotate_cl4_k<5>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 2048 * (int)sizeof(uint32_t));
                        attr_c4 = true;
                    }
                    if (grp == 2) rotate_cl4_k<2><<<4 * A.count, th, sc, s>>>(A);
                    else if (grp == 3) rotate_cl4_k<3><<<4 * A.count, th, sc, s>>>(A);
                    else if (grp == 4) rotate_cl4_k<4><<<4 * A.count, th, sc, s>>>(A);
                    else rotate_cl4_k<5><<<4 * A.count, th, sc, s>>>(A);
                    break;
                }
                if (i3 != 1) {   // a cluster of two blocks per LWE
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
            // two LWEs per block as soon as one per block takes a second round of blocks
            const bool two = lv == 1 && A.count > sms;
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
            if (A.n != 2048 || (A.c[1] != 1 && A.c[1] != 2) || grp < 2 || grp > 5 || A.i1 % grp || A.count > 65535) return -1;
            if (grp == 5 && (A.c[1] != 1 || A.c[5] > 1)) return -1;
            const int sm = 4 * A.n * (int)sizeof(uint32_t);   // pm1
            const int gw = A.i4 ? A.i4 : A.i1 / grp - A.i0;
            if (A.i0 < 0 || gw < 1 || A.i0 + gw > A.i1 / grp) return -1;
            const dim3 grid(gw, A.count);
            if (A.c[1] == 2) {   // two gadget levels
                if (A.c[5] > 1) return -1;
                if (grp == 2) keybundle_k<2, 1, 2><<<grid, STEP_TH, sm, s>>>(A);
                else if (grp == 3) keybundle_k<3, 1, 2><<<grid, STEP_TH, sm, s>>>(A);
                else keybundle_k<4, 1, 2><<<grid, STEP_TH, sm, s>>>(A);
                break;
            }
            if (A.c[5] == 2) {   // the extended rotation's A, B
                if (grp == 2) keybundle_k<2, 2><<<grid, STEP_TH, sm, s>>>(A);
                else if (grp == 3) keybundle_k<3, 2><<<grid, STEP_TH, sm, s>>>(A);
                else keybundle_k<4, 2><<<grid, STEP_TH, sm, s>>>(A);
                break;
            }
            if (A.c[5] > 1) return -1;
            if (grp == 2) keybundle_k<2><<<grid, STEP_TH, sm, s>>>(A);
            else if (grp == 3) keybundle_k<3><<<grid, STEP_TH, sm, s>>>(A);
            else if (grp == 4) keybundle_k<4><<<grid, STEP_TH, sm, s>>>(A);
            else keybundle_k<5><<<grid, STEP_TH, sm, s>>>(A);
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
        case PC_TFHE_REPACK_COMB: {
            const size_t tot = (size_t)A.count * 4 * A.i1;
            repack_comb_k<<<(unsigned)((tot + 255) / 256), 256, 0, s>>>(A);
            break;
        }
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
        case PC_TFHE_LWE_LIN:
            if (A.count > 65535) return -1;
            lwe_lin_k<<<dim3((A.i1 + 256) / 256, A.count), 256, 0, s>>>(A);
            break;
        default:
            return -2;
    }
    return (int)cudaGetLastError();
}
}  // namespace tfhe
}  // namespace pc
