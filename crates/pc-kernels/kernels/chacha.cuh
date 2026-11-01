// Seeded uniform expander, ChaCha12 counter mode, state words 12..15 = (block counter, w13, w14, tag), rejection with a
// block stride. FROZEN at two words (every seeded key depends on them; host twins in pc_core::keys):
// - u32 = FIDESlib KSKA lane (KskSeedExpand.cuh::expand_coeff): w13 = digit, w14 = p, tag "KSKA"; 16 values per block.
// - u64 = FIDESlib KSKB lane (SPEC v2, expand_coeff64 / OpenFHE patch kska-seed): w13 = (digit & 0xF) | hi32(p) << 4,
//   w14 = lo32(p), tag "KSKB" = "KSKA" + 1; 8 values per block (little-endian word pairs), stride N/8. Needs digit < 16
//   and p < 2^60 (hi32(p) < 2^28); the host refuses other chains (pc_core::keys::check_kskb).
// The encryption stream ("ENCA", w13 = nonce) takes the same per-word rule ("ENCB" at u64); it is never serialized.
// u16 (not frozen, compile-only word): 32 values per block, w14 = p.
#pragma once
#include "arith.cuh"

namespace pc {
namespace chacha {
constexpr uint32_t TAG_KSKA = 0x4b534b41u;  // "KSKA"
constexpr uint32_t TAG_ENCA = 0x454e4341u;  // "ENCA"

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
__device__ __forceinline__ void qr(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = rotl32(d, 16);
    c += d; b ^= c; b = rotl32(b, 12);
    a += b; d ^= a; d = rotl32(d, 8);
    c += d; b ^= c; b = rotl32(b, 7);
}
__device__ __forceinline__ void chacha12(const uint32_t* key, uint32_t ctr, uint32_t w13, uint32_t w14, uint32_t w15,
                                         uint32_t out[16]) {
    const uint32_t s[16] = {0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u, key[0], key[1], key[2], key[3],
                            key[4],      key[5],      key[6],      key[7],      ctr,    w13,    w14,    w15};
    uint32_t x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = s[i];
#pragma unroll
    for (int r = 0; r < 12; r += 2) {
        qr(x[0], x[4], x[8], x[12]);
        qr(x[1], x[5], x[9], x[13]);
        qr(x[2], x[6], x[10], x[14]);
        qr(x[3], x[7], x[11], x[15]);
        qr(x[0], x[5], x[10], x[15]);
        qr(x[1], x[6], x[11], x[12]);
        qr(x[2], x[7], x[8], x[13]);
        qr(x[3], x[4], x[9], x[14]);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = x[i] + s[i];
}

template <class W>
constexpr int PER_BLOCK = 64 / (int)sizeof(W);  // values per 64-byte ChaCha block
template <class W>
__device__ __forceinline__ uint32_t pword(W p) {
    return (uint32_t)p;  // u64: low word (KSKB s[14]); the high word goes into w13
}
template <class W>
__device__ __forceinline__ uint32_t lane_w13(uint32_t w13, W p) {
    // xor, not or: a bijection in w13 for fixed p (distinct encryption nonces never share a stream); == the KSKB
    // packing (digit & 0xF) | hi32(p) << 4 for digit < 16
    if constexpr (sizeof(W) == 8) return w13 ^ ((uint32_t)((uint64_t)p >> 32) << 4);
    else return w13;
}
template <class W>
__device__ __forceinline__ uint32_t lane_tag(uint32_t tag) {
    if constexpr (sizeof(W) == 8) return tag + 1u;  // KSKA -> KSKB, ENCA -> ENCB
    else return tag;
}
template <class W>
__device__ __forceinline__ W value_at(const uint32_t blk[16], int j) {
    if constexpr (sizeof(W) == 4) return blk[j];
    else if constexpr (sizeof(W) == 8) return (uint64_t)blk[2 * j] | ((uint64_t)blk[2 * j + 1] << 32);
    else return (uint16_t)(blk[j >> 1] >> (16 * (j & 1)));
}

// Value of `slot` (one thread, full block per try): bit-identical to the cooperative path at W = u32.
template <class W, class MM>
__device__ __forceinline__ W expand_slot(const uint32_t* seed, uint32_t slot, uint32_t w13, const PrimeConst& c,
                                         uint32_t tag) {
    const W p = (W)c.p;
    const uint32_t nblk = (uint32_t)N / PER_BLOCK<W>;
    const uint32_t b0 = slot / PER_BLOCK<W>;
    const int j = (int)(slot % PER_BLOCK<W>);
    const W mp = (W)((W)(~(W)0) / p) * p;
    const uint32_t w13l = lane_w13<W>(w13, p), tagl = lane_tag<W>(tag);
    uint32_t blk[16];
    W v = 0;
    for (uint32_t t = 0; t < 16; ++t) {
        chacha12(seed, b0 + t * nblk, w13l, pword<W>(p), tagl, blk);
        v = value_at<W>(blk, j);
        if (v < mp) break;
    }
    return MM::reduce((WideOf<W>)v, c);
}

// ---------------------------------------------------------------- cooperative u32 path (FIDESlib fusedDotKSKRegen4_)
// A 4-lane group computes one 16-word block column-wise (lane k holds words k, k+4, k+8, k+12); each lane owns 4
// consecutive slots.
__device__ __forceinline__ uint32_t sel4(uint32_t k, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    uint32_t r = v0;
    r = (k == 1) ? v1 : r;
    r = (k == 2) ? v2 : r;
    r = (k == 3) ? v3 : r;
    return r;
}
__device__ __forceinline__ void chacha_coop4(uint32_t gmask, uint32_t k, uint32_t key0, uint32_t key4, uint32_t ctr,
                                             uint32_t w13, uint32_t w14, uint32_t tag, uint32_t& oa, uint32_t& ob,
                                             uint32_t& oc, uint32_t& od) {
    const uint32_t A = sel4(k, 0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u);
    const uint32_t D = sel4(k, ctr, w13, w14, tag);
    uint32_t a = A, b = key0, c = key4, d = D;
#pragma unroll
    for (int r = 0; r < 12; r += 2) {
        a += b; d ^= a; d = __funnelshift_l(d, d, 16);
        c += d; b ^= c; b = __funnelshift_l(b, b, 12);
        a += b; d ^= a; d = __funnelshift_l(d, d, 8);
        c += d; b ^= c; b = __funnelshift_l(b, b, 7);
        b = __shfl_sync(gmask, b, (k + 1) & 3u, 4);
        c = __shfl_sync(gmask, c, (k + 2) & 3u, 4);
        d = __shfl_sync(gmask, d, (k + 3) & 3u, 4);
        a += b; d ^= a; d = __funnelshift_l(d, d, 16);
        c += d; b ^= c; b = __funnelshift_l(b, b, 12);
        a += b; d ^= a; d = __funnelshift_l(d, d, 8);
        c += d; b ^= c; b = __funnelshift_l(b, b, 7);
        b = __shfl_sync(gmask, b, (k + 3) & 3u, 4);
        c = __shfl_sync(gmask, c, (k + 2) & 3u, 4);
        d = __shfl_sync(gmask, d, (k + 1) & 3u, 4);
    }
    oa = a + A; ob = b + key0; oc = c + key4; od = d + D;
}

// 4 regenerated words (slots 4k'..4k'+3 of the group's block = this lane's slots), rejection-escalated; u32 only
__device__ __forceinline__ void regen4(const uint32_t* seed, uint32_t lane, uint32_t b0, uint32_t w13, uint32_t p,
                                       uint32_t n16, uint32_t tag, uint32_t out[4]) {
    const uint32_t k = lane & 3u;
    const uint32_t gmask = 0xFu << (lane & ~3u);
    const uint32_t key0 = seed[k], key4 = seed[4 + k];
    const uint32_t m_p = (0xFFFFFFFFu / p) * p;
    uint32_t ca, cb, cc, cd;
    chacha_coop4(gmask, k, key0, key4, b0, w13, p, tag, ca, cb, cc, cd);
    uint32_t need = ((ca >= m_p) ? 1u : 0u) | ((cb >= m_p) ? 2u : 0u) | ((cc >= m_p) ? 4u : 0u) | ((cd >= m_p) ? 8u : 0u);
    uint32_t gneed = need;
    gneed |= __shfl_xor_sync(gmask, gneed, 1, 4);
    gneed |= __shfl_xor_sync(gmask, gneed, 2, 4);
    if (gneed) {
        for (uint32_t t = 1; t < 16u; ++t) {
            uint32_t ea, eb, ec, ed;
            chacha_coop4(gmask, k, key0, key4, b0 + t * n16, w13, p, tag, ea, eb, ec, ed);
            if (need & 1u) { ca = ea; if (ea < m_p) need &= ~1u; }
            if (need & 2u) { cb = eb; if (eb < m_p) need &= ~2u; }
            if (need & 4u) { cc = ec; if (ec < m_p) need &= ~4u; }
            if (need & 8u) { cd = ed; if (ed < m_p) need &= ~8u; }
            gneed = need;
            gneed |= __shfl_xor_sync(gmask, gneed, 1, 4);
            gneed |= __shfl_xor_sync(gmask, gneed, 2, 4);
            if (!gneed) break;
        }
    }
    // transpose: lane k wants words 4k..4k+3 = column word k of lanes 0..3
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const uint32_t pub = sel4((k - (uint32_t)s) & 3u, ca, cb, cc, cd);
        const uint32_t rcv = __shfl_sync(gmask, pub, (k + (uint32_t)s) & 3u, 4);
        const uint32_t r = (k + (uint32_t)s) & 3u;
        out[r] = rcv;
    }
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const uint32_t v = out[w] - __umulhi(out[w], 0xFFFFFFFFu / p) * p;
        out[w] = v >= p ? v - p : v;
    }
}

// regen4 where lane m of the 4-lane group takes the block's quarter (jsel >> 2m) & 3 (words 4j..4j+3) instead of
// quarter m: jsel names a permutation of the quarters, the same value on the group's 4 lanes. The same words.
__device__ __forceinline__ void regen4q(const uint32_t* seed, uint32_t lane, uint32_t b0, uint32_t w13, uint32_t p,
                                        uint32_t n16, uint32_t tag, uint32_t jsel, uint32_t out[4]) {
    const uint32_t k = lane & 3u;
    const uint32_t gmask = 0xFu << (lane & ~3u);
    const uint32_t key0 = seed[k], key4 = seed[4 + k];
    const uint32_t m_p = (0xFFFFFFFFu / p) * p;
    uint32_t ca, cb, cc, cd;
    chacha_coop4(gmask, k, key0, key4, b0, w13, p, tag, ca, cb, cc, cd);
    uint32_t need = ((ca >= m_p) ? 1u : 0u) | ((cb >= m_p) ? 2u : 0u) | ((cc >= m_p) ? 4u : 0u) | ((cd >= m_p) ? 8u : 0u);
    uint32_t gneed = need;
    gneed |= __shfl_xor_sync(gmask, gneed, 1, 4);
    gneed |= __shfl_xor_sync(gmask, gneed, 2, 4);
    if (gneed) {
        for (uint32_t t = 1; t < 16u; ++t) {
            uint32_t ea, eb, ec, ed;
            chacha_coop4(gmask, k, key0, key4, b0 + t * n16, w13, p, tag, ea, eb, ec, ed);
            if (need & 1u) { ca = ea; if (ea < m_p) need &= ~1u; }
            if (need & 2u) { cb = eb; if (eb < m_p) need &= ~2u; }
            if (need & 4u) { cc = ec; if (ec < m_p) need &= ~4u; }
            if (need & 8u) { cd = ed; if (ed < m_p) need &= ~8u; }
            gneed = need;
            gneed |= __shfl_xor_sync(gmask, gneed, 1, 4);
            gneed |= __shfl_xor_sync(gmask, gneed, 2, 4);
            if (!gneed) break;
        }
    }
    // transpose: lane m wants quarter j_m = column word j_m of lanes 0..3; at step s lane k reads lane (k + s) & 3,
    // which publishes the column its reader (k) wants
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const uint32_t rd = (k - (uint32_t)s) & 3u;
        const uint32_t pub = sel4((jsel >> (2u * rd)) & 3u, ca, cb, cc, cd);
        const uint32_t rcv = __shfl_sync(gmask, pub, (k + (uint32_t)s) & 3u, 4);
        out[(k + (uint32_t)s) & 3u] = rcv;
    }
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const uint32_t v = out[w] - __umulhi(out[w], 0xFFFFFFFFu / p) * p;
        out[w] = v >= p ? v - p : v;
    }
}

// ---------------------------------------------------------------- per-thread u32 blocks
// The 16 words of block b0 of digit d expanded by one thread: four independent quarter-rounds per step, no shuffles,
// rejection with the block stride n16 and the final reduction of regen4 (the same words).
__device__ __forceinline__ void regen_block16(const uint32_t (&key)[8], uint32_t b0, uint32_t d, uint32_t p, uint32_t n16,
                                              uint32_t tag, uint32_t (&v)[16]) {
    chacha12(key, b0, d, p, tag, v);
    const uint32_t m_p = (0xFFFFFFFFu / p) * p;
    uint32_t need = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) need |= (v[i] >= m_p ? 1u : 0u) << i;
    for (uint32_t t = 1; need && t < 16u; ++t) {
        uint32_t e[16];
        chacha12(key, b0 + t * n16, d, p, tag, e);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (need >> i & 1u) {
                v[i] = e[i];
                if (e[i] < m_p) need &= ~(1u << i);
            }
        }
    }
    const uint32_t r = 0xFFFFFFFFu / p;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint32_t x = v[i] - __umulhi(v[i], r) * p;
        v[i] = x >= p ? x - p : x;
    }
}

// 4 consecutive slots base4..base4+3 of this thread (lane-group cooperative at u32, per-thread otherwise).
// All 32 lanes of the warp must call it together at u32.
template <class W, class MM>
__device__ __forceinline__ void regen_slots4(const uint32_t* seed, uint32_t lane, uint32_t base4, uint32_t w13,
                                             const PrimeConst& c, uint32_t tag, W out[4]) {
    if constexpr (sizeof(W) == 4) {
        uint32_t o[4];
        regen4(seed, lane, base4 >> 4, w13, (uint32_t)c.p, N >> 4, tag, o);
#pragma unroll
        for (int w = 0; w < 4; ++w) out[w] = o[w];
    } else {
#pragma unroll
        for (int w = 0; w < 4; ++w) out[w] = expand_slot<W, MM>(seed, base4 + w, w13, c, tag);
    }
}
}  // namespace chacha
}  // namespace pc
