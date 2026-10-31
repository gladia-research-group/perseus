// Device modular arithmetic generic over the word W (uint16_t / uint32_t / uint64_t) and the ModMul strategy MM.
//  - add/sub/Shoup are strategy-free: constant operands (twiddles, per-row scalars) always use Shoup with a
//    precomputed companion w' = floor(w 2^BITS / p).
//  - MM::mul(a, b, q): product of two runtime residues.  MM::reduce(x, q): exact x mod p for ANY double word x (the
//    lazy form: accumulators of several products are reduced once).
// Strategies: Barrett64 (double-word Barrett), Barrett32 (FIDESlib Neal_mult_32 at W = u32,
// p < 2^30), Montgomery (REDC twice, operands stay in normal form; reduce is the double-word Barrett).
#pragma once
#include "args.cuh"

namespace pc {
constexpr int LOGN = PC_LOGN;
constexpr int N = PC_N;

template <class W>
struct Word;
template <>
struct Word<uint16_t> {
    using Wide = uint32_t;
    static constexpr int BITS = 16;
    __device__ __forceinline__ static uint16_t mulhi(uint16_t a, uint16_t b) { return (uint16_t)(((uint32_t)a * b) >> 16); }
};
template <>
struct Word<uint32_t> {
    using Wide = uint64_t;
    static constexpr int BITS = 32;
    __device__ __forceinline__ static uint32_t mulhi(uint32_t a, uint32_t b) { return __umulhi(a, b); }
};
template <>
struct Word<uint64_t> {
    using Wide = unsigned __int128;
    static constexpr int BITS = 64;
    __device__ __forceinline__ static uint64_t mulhi(uint64_t a, uint64_t b) { return __umul64hi(a, b); }
};
template <class W>
using WideOf = typename Word<W>::Wide;

// a * b in W arithmetic (mod 2^BITS); u16 goes through u32 so the int promotion cannot overflow
template <class W>
__device__ __forceinline__ W wlo(W a, W b) {
    if constexpr (sizeof(W) == 2) return (W)((uint32_t)a * (uint32_t)b);
    else return a * b;
}
template <class W>
__device__ __forceinline__ WideOf<W> wmul(W a, W b) {
    return (WideOf<W>)a * b;
}

template <class W>
__device__ __forceinline__ W add(W a, W b, W p) {
    const W s = a + b;
    return s >= p ? (W)(s - p) : s;
}
template <class W>
__device__ __forceinline__ W sub(W a, W b, W p) {
    return a >= b ? (W)(a - b) : (W)(a + p - b);
}
// Shoup: w' = floor(w 2^BITS / p), a < 2^BITS (any), w < p < 2^(BITS-1)
template <class W>
__device__ __forceinline__ W mul_shoup(W a, W w, W ws, W p) {
    const W q = Word<W>::mulhi(a, ws);
    const W r = (W)(wlo(a, w) - wlo(q, p));
    return r >= p ? (W)(r - p) : r;
}

// Double-word Barrett with mu = floor(2^(2 BITS) / p): exact for every x < 2^(2 BITS).
template <class W>
__device__ __forceinline__ W reduce_dw(WideOf<W> x, const PrimeConst& c);
template <>
__device__ __forceinline__ uint16_t reduce_dw<uint16_t>(uint32_t x, const PrimeConst& c) {
    const uint32_t p = (uint32_t)c.p;
    const uint32_t q = __umulhi(x, (uint32_t)c.mu_lo);
    uint32_t r = x - q * p;
    if (r >= p) r -= p;
    if (r >= p) r -= p;
    return (uint16_t)r;
}
template <>
__device__ __forceinline__ uint32_t reduce_dw<uint32_t>(uint64_t x, const PrimeConst& c) {
    if (c.split) {
        // x = hi 2^32 + lo: hi (2^32 mod p) and lo (times 1) as Shoup products in 32-bit words, each in [0, 2p) before
        // its guard (p < 2^31); m = floor(2^32 / p) is the high word of mu_lo
        const uint32_t p = (uint32_t)c.p, m = (uint32_t)(c.mu_lo >> 32);
        const uint32_t hi = (uint32_t)(x >> 32), lo = (uint32_t)x;
        uint32_t a = hi * (uint32_t)c.split - __umulhi(hi, (uint32_t)(c.split >> 32)) * p;
        a = a >= p ? a - p : a;
        uint32_t b = lo - __umulhi(lo, m) * p;
        b = b >= p ? b - p : b;
        a += b;
        return a >= p ? a - p : a;
    }
    // q underestimates by at most 1, two guards
    const uint64_t p = c.p;
    const uint64_t q = __umul64hi(x, c.mu_lo);
    uint64_t r = x - q * p;
    if (r >= p) r -= p;
    if (r >= p) r -= p;
    return (uint32_t)r;
}
template <>
__device__ __forceinline__ uint64_t reduce_dw<uint64_t>(unsigned __int128 x, const PrimeConst& c) {
    // q = floor(x mu / 2^128) estimated from the partial products; only q mod 2^64 is needed because the true
    // remainder is < 4p < 2^64 (p < 2^62), so carries past bit 64 of q vanish.
    const uint64_t x0 = (uint64_t)x, x1 = (uint64_t)(x >> 64);
    const uint64_t m0 = c.mu_lo, m1 = c.mu_hi;
    const unsigned __int128 mid = (unsigned __int128)x1 * m0 + (unsigned __int128)x0 * m1 + __umul64hi(x0, m0);
    const uint64_t q = x1 * m1 + (uint64_t)(mid >> 64);
    const uint64_t p = c.p;
    uint64_t r = x0 - q * p;
    if (r >= p) r -= p;
    if (r >= p) r -= p;
    if (r >= p) r -= p;
    return r;
}

template <class W>
struct Barrett64 {
    static constexpr uint32_t KIND = 0;
    __device__ __forceinline__ static W reduce(WideOf<W> x, const PrimeConst& c) { return reduce_dw<W>(x, c); }
    __device__ __forceinline__ static W mul(W a, W b, const PrimeConst& c) { return reduce_dw<W>(wmul(a, b), c); }
};

// FIDESlib Neal_mult_32 (improved Barrett, mu = floor(2^(2 bits + 1) / p)) plus a second conditional subtraction as
// insurance: no bound <= 1 on the quotient error is proven here (the selftest's host census finds no product that needs
// it at 27-30 bits, largest products included). Same canonical output as FIDESlib wherever its single guard suffices.
// Other words fall back to the double-word Barrett.
template <class W>
struct Barrett32 {
    static constexpr uint32_t KIND = 1;
    __device__ __forceinline__ static W reduce(WideOf<W> x, const PrimeConst& c) { return reduce_dw<W>(x, c); }
    __device__ __forceinline__ static W mul(W a, W b, const PrimeConst& c) {
        if constexpr (sizeof(W) == 4) {
            const uint32_t p = (uint32_t)c.p, qbit = c.bits;
            const uint64_t prod = (uint64_t)a * b;
            const uint32_t rx = (uint32_t)(prod >> (qbit - 2));
            uint32_t rb = __umulhi(rx << (30 - qbit), (uint32_t)c.neal_mu) >> 1;
            rb *= p;
            uint32_t lo = (uint32_t)prod - rb;
            lo -= p * (lo >= p);
            lo -= p * (lo >= p);
            return lo;
        } else {
            return reduce_dw<W>(wmul(a, b), c);
        }
    }
};

// Montgomery with R = 2^BITS. Residues stay in normal form: mul = REDC(REDC(a b) R^2) = a b. Exists for the seam's
// contract test; it is not expected to beat Barrett64 here (two REDCs per product).
template <class W>
struct Montgomery {
    static constexpr uint32_t KIND = 2;
    __device__ __forceinline__ static W redc(WideOf<W> t, const PrimeConst& c) {
        using Wd = WideOf<W>;
        const W p = (W)c.p;
        const W m = wlo((W)t, (W)c.mont_pinv);
        const Wd s = t + (Wd)m * p;  // < 2 p 2^BITS: no overflow for p < 2^(BITS-1)
        W r = (W)(s >> Word<W>::BITS);
        return r >= p ? (W)(r - p) : r;
    }
    __device__ __forceinline__ static W reduce(WideOf<W> x, const PrimeConst& c) { return reduce_dw<W>(x, c); }
    __device__ __forceinline__ static W mul(W a, W b, const PrimeConst& c) {
        return redc(wmul(redc(wmul(a, b), c), (W)c.mont_r2), c);
    }
};

// y mod q for a residue y of another prime: exact; the subtract path covers the chain's 27-28 bit primes (y < 3q)
template <class W, class MM>
__device__ __forceinline__ W reduce_small(W y, const PrimeConst& c) {
    const W q = (W)c.p;
    if (y < 3 * (WideOf<W>)q) {
        if (y >= q) y -= q;
        if (y >= q) y -= q;
        return y;
    }
    return MM::reduce((WideOf<W>)y, c);
}

// ------------------------------------------------------------------------------------------------ vector access
// 4 consecutive words (16-byte aligned for u32, 32 for u64, 8 for u16)
template <class W>
struct V4 {
    W v[4];
};
template <class W>
__device__ __forceinline__ V4<W> load4(const W* p) {
    V4<W> r;
    if constexpr (sizeof(W) == 4) {
        const uint4 x = *(const uint4*)p;
        r.v[0] = x.x; r.v[1] = x.y; r.v[2] = x.z; r.v[3] = x.w;
    } else if constexpr (sizeof(W) == 8) {
        const ulonglong2 x = ((const ulonglong2*)p)[0], y = ((const ulonglong2*)p)[1];
        r.v[0] = x.x; r.v[1] = x.y; r.v[2] = y.x; r.v[3] = y.y;
    } else {
        const ushort4 x = *(const ushort4*)p;
        r.v[0] = x.x; r.v[1] = x.y; r.v[2] = x.z; r.v[3] = x.w;
    }
    return r;
}
template <class W>
__device__ __forceinline__ V4<W> ldg4(const W* p) {
    V4<W> r;
    if constexpr (sizeof(W) == 4) {
        const uint4 x = __ldg((const uint4*)p);
        r.v[0] = x.x; r.v[1] = x.y; r.v[2] = x.z; r.v[3] = x.w;
    } else if constexpr (sizeof(W) == 8) {
        const ulonglong2 x = __ldg((const ulonglong2*)p), y = __ldg((const ulonglong2*)p + 1);
        r.v[0] = x.x; r.v[1] = x.y; r.v[2] = y.x; r.v[3] = y.y;
    } else {
        const ushort4 x = __ldg((const ushort4*)p);
        r.v[0] = x.x; r.v[1] = x.y; r.v[2] = x.z; r.v[3] = x.w;
    }
    return r;
}
template <class W>
__device__ __forceinline__ void store4(W* p, W a, W b, W c, W d) {
    if constexpr (sizeof(W) == 4) {
        *(uint4*)p = make_uint4(a, b, c, d);
    } else if constexpr (sizeof(W) == 8) {
        ((ulonglong2*)p)[0] = make_ulonglong2(a, b);
        ((ulonglong2*)p)[1] = make_ulonglong2(c, d);
    } else {
        *(ushort4*)p = make_ushort4(a, b, c, d);
    }
}

template <class W>
__device__ __forceinline__ W shfl_xor(unsigned mask, W v, int lanemask) {
    if constexpr (sizeof(W) == 2) return (W)__shfl_xor_sync(mask, (uint32_t)v, lanemask);
    else return __shfl_xor_sync(mask, v, lanemask);
}

// bit-reversed evaluation order automorphism (FIDESlib automorph_slot): slot of sigma_k's source for output `slot`
__device__ __forceinline__ uint32_t automorph_slot(uint32_t k, uint32_t slot) {
    const uint32_t j = __brev(slot) >> (32 - LOGN);
    const uint32_t jt = (j << 1) + 1;
    const uint32_t r = ((jt * k) & ((1u << (LOGN + 1)) - 1)) >> 1;
    return __brev(r) >> (32 - LOGN);
}

// compact plaintext index (see PtDotArgs)
__device__ __forceinline__ int ptidx(int idx, uint32_t enc) {
    return ((enc >> 24) == 0x7Fu) ? (idx >> (enc & 0xFFu)) : (int)((uint32_t)idx & enc);
}

template <class W>
__device__ __forceinline__ W* at(const PolyRef& r, int c, int part, int row) {
    return (W*)r.base + c * r.ct_stride + part * r.part_stride + (size_t)row * N;
}
__device__ __forceinline__ int rs_row(const RowSet& rs, int y) { return y < rs.nq ? y : rs.qmax + (y - rs.nq); }
}  // namespace pc
