// Hybrid key-switch inner products and the seeded key-`a` expansion, generic over
// (W, MM). Semantics in args.cuh (KsArgs, EpiArgs, KsRegenArgs, KskExpandArgs).
#pragma once
#include "arith.cuh"
#include "chacha.cuh"

namespace pc {
// Seeded key sets read in place (KeyTab, ABI 4): b of digit d at prime id t, slot s; packed rows unpacked in registers.
namespace kskey {
// q_top < 0: full layout (slot = prime id)
__device__ __forceinline__ int slot_of(const KeyTab& T, int t, int nq) { return (t < nq || T.q_top < 0) ? t : T.q_top + 1 + (t - nq); }
template <class W>
__device__ __forceinline__ W b_at(const KeyTab& T, const void* kb, size_t row, uint32_t s) {
    if constexpr (sizeof(W) == 4) {
        if (T.bits) {
            const uint32_t* r = (const uint32_t*)kb + row * T.row_words;
            const uint64_t bit = (uint64_t)s * (uint32_t)T.bits;
            const uint32_t w = (uint32_t)(bit >> 5), o = (uint32_t)(bit & 31);
            const uint64_t v = ((uint64_t)__ldg(r + w) | ((uint64_t)__ldg(r + w + 1) << 32)) >> o;
            return (W)(v & ((1ull << T.bits) - 1));
        }
    }
    return __ldg((const W*)kb + row * T.row_words + s);
}
// 4 consecutive words read once (evict-first in L2: the key stream does not push the ModUp digits and the dot's
// outputs out of L2)
template <class W>
__device__ __forceinline__ V4<W> ldcs4(const W* p) {
    V4<W> r;
    if constexpr (sizeof(W) == 4) {
        const uint4 x = __ldcs((const uint4*)p);
        r.v[0] = x.x; r.v[1] = x.y; r.v[2] = x.z; r.v[3] = x.w;
    } else {
        r = ldg4<W>(p);
    }
    return r;
}
// PACKED: the table's rows are packed (T.bits > 0; u32 only), a compile-time split so dense reads carry no packed code
template <class W, bool PACKED>
__device__ __forceinline__ V4<W> b_at4(const KeyTab& T, const void* kb, size_t row, uint32_t s4) {
    if constexpr (PACKED && sizeof(W) == 4) {
        // 4 consecutive values span at most 5 words from the first one's word (bits <= 32); load only the words
        // they touch (the row's guard word covers the funnel's look-ahead) and funnel-shift each value out
        const uint32_t* r = (const uint32_t*)kb + row * T.row_words;
        const uint32_t bits = (uint32_t)T.bits;
        const uint64_t bit = (uint64_t)s4 * bits;
        const uint32_t w0 = (uint32_t)(bit >> 5), o = (uint32_t)(bit & 31);
        const uint32_t nw = (o + 4 * bits + 31) >> 5;
        uint32_t wd[5];
#pragma unroll
        for (int t = 0; t < 5; ++t) wd[t] = (uint32_t)t < nw ? __ldcs(r + w0 + t) : 0u;
        const uint32_t mask = (uint32_t)((1ull << bits) - 1);
        V4<W> v;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t lo = o + (uint32_t)w * bits, i = lo >> 5;  // i <= 3
            const uint32_t a = i == 0 ? wd[0] : i == 1 ? wd[1] : i == 2 ? wd[2] : wd[3];
            const uint32_t b = i == 0 ? wd[1] : i == 1 ? wd[2] : i == 2 ? wd[3] : wd[4];
            v.v[w] = (W)(__funnelshift_r(a, b, lo & 31) & mask);
        }
        return v;
    }
    return ldcs4<W>((const W*)kb + row * T.row_words + s4);
}
}  // namespace kskey

namespace ks {
constexpr int BX = 128;

template <class W, class MM, int KMAX>
__global__ void __launch_bounds__(BX) ks_dot(KsArgs A) {
    using Wd = WideOf<W>;
    const uint32_t t = A.rows[blockIdx.y];
    const PrimeConst& q = A.pr[t];
    const W p = (W)q.p;
    const int idx = blockIdx.x * BX + threadIdx.x;
    const int dt = (int)t < A.nq ? A.digit_of[t] : -1;
    const size_t roff = (size_t)t * N + idx;
    const W* c1 = (const W*)A.c1;
    const W* mu = (const W*)A.modup;
    const W* c0 = (const W*)A.c0;
    for (int r = 0; r < A.R; ++r) {
        Wd s0[KMAX], s1[KMAX];
#pragma unroll
        for (int c = 0; c < KMAX; ++c) {
            s0[c] = 0;
            s1[c] = 0;
        }
        const W* kb = (const W*)A.kb[r];
        const W* ka = (const W*)A.ka[r];
        for (int d = 0; d < A.ndig; ++d) {
            const W b = __ldg(kb + d * A.key_digit_stride + roff);
            const W a = __ldg(ka + d * A.key_digit_stride + roff);
#pragma unroll
            for (int c = 0; c < KMAX; ++c) {
                if (c < A.K) {
                    const W x = (d == dt) ? c1[c * A.ct_stride + roff] : mu[c * A.mu_ct_stride + d * A.mu_digit_stride + roff];
                    s0[c] += (Wd)x * b;
                    s1[c] += (Wd)x * a;
                }
            }
        }
        const uint32_t k = A.autok[r];
        const uint32_t o = k == 1 ? (uint32_t)idx : automorph_slot(k, idx);
#pragma unroll
        for (int c = 0; c < KMAX; ++c) {
            if (c < A.K) {
                W r0 = MM::reduce(s0[c], q);
                W r1 = MM::reduce(s1[c], q);
                if ((A.c0_mode & 3) == 1) {
                    if ((int)t < A.nq)
                        r0 = add<W>(r0, mul_shoup<W>(c0[c * A.ct_stride + roff], ((const W*)A.pmod)[t], ((const W*)A.pmods)[t], p), p);
                } else if ((A.c0_mode & 3) == 2) {
                    r0 = add<W>(r0, c0[c * A.ct_stride + roff], p);
                }
                const size_t ob = c * A.out_ct_stride + r * A.out_rot_stride + (size_t)t * N + o;
                if (A.c0_mode & KS_ACC) {
                    r0 = add<W>(r0, ((const W*)A.out0)[ob], p);
                    r1 = add<W>(r1, ((const W*)A.out1)[ob], p);
                }
                ((W*)A.out0)[ob] = r0;
                ((W*)A.out1)[ob] = r1;
            }
        }
    }
}

// TAB: 0 = dense prime-id rows of ks.kb[0]; 1 = key 0 of G.key, dense / banded rows; 2 = G.key, packed rows
template <class W, class MM, int KMAX, bool EPI, int TAB>
__global__ void __launch_bounds__(BX) ks_dot_regen(KsRegenArgs G) {
    using Wd = WideOf<W>;
    const KsArgs& A = G.ks;
    const EpiArgs& E = G.epi;
    const uint32_t t = A.rows[blockIdx.y];
    const PrimeConst& q = A.pr[t];
    const W p = (W)q.p;
    const uint32_t gtid = blockIdx.x * BX + threadIdx.x;
    const int base = gtid * 4;
    const uint32_t lane = threadIdx.x & 31u;
    const int dt = (int)t < A.nq ? A.digit_of[t] : -1;
    const size_t roff = (size_t)t * N + base;
    Wd s0[KMAX][4], s1[KMAX][4];
#pragma unroll
    for (int c = 0; c < KMAX; ++c)
#pragma unroll
        for (int w = 0; w < 4; ++w) s0[c][w] = s1[c][w] = 0;
    // `b`: key 0 of G.key read in place (dense / packed / banded slots) when the table is set, else the dense
    // prime-id rows of ks.kb[0]
    const KeyTab& T = G.key;
    const void* kb = TAB ? T.kb[0] : A.kb[0];
    const int slot = TAB ? kskey::slot_of(T, (int)t, A.nq) : 0;
    for (int d = 0; d < A.ndig; ++d) {
        W a4[4];
        chacha::regen_slots4<W, MM>(G.seed, lane, (uint32_t)base, (uint32_t)d, q, chacha::TAG_KSKA, a4);
        V4<W> b4;
        if constexpr (TAB != 0) b4 = kskey::b_at4<W, TAB == 2>(T, kb, (size_t)d * T.srows + slot, (uint32_t)base);
        else b4 = kskey::ldcs4<W>((const W*)kb + d * A.key_digit_stride + roff);
#pragma unroll
        for (int c = 0; c < KMAX; ++c) {
            if (c < A.K) {
                const V4<W> x4 = (d == dt) ? load4<W>((const W*)A.c1 + c * A.ct_stride + roff)
                                           : load4<W>((const W*)A.modup + c * A.mu_ct_stride + d * A.mu_digit_stride + roff);
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    s0[c][w] += (Wd)x4.v[w] * b4.v[w];
                    s1[c][w] += (Wd)x4.v[w] * a4[w];
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < KMAX; ++c) {
        if (c < A.K) {
            W r0[4], r1[4];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                r0[w] = MM::reduce(s0[c][w], q);
                r1[w] = MM::reduce(s1[c][w], q);
            }
            const size_t ob = c * A.out_ct_stride + (size_t)t * N + base;
            if constexpr (EPI) {
                const W bi = ((const W*)E.binv)[t], bis = ((const W*)E.binvs)[t];
                const W* kp = (const W*)E.k;
                const W kk = kp ? (kp[t] >= p ? (W)(kp[t] - p) : kp[t]) : (W)0;
                const W* al = (const W*)E.alpha;
                const W ai = al ? al[t] : (W)0, ais = al ? ((const W*)E.alphas)[t] : (W)0;
#pragma unroll
                for (int part = 0; part < 2; ++part) {
                    W* dp = (W*)E.dst + ob + part * E.part_stride;
                    const V4<W> cv = load4<W>((const W*)E.conv + ob + part * E.part_stride);
                    V4<W> dv = load4<W>(dp);
                    V4<W> av{};
                    if (G.add) av = load4<W>((const W*)G.add + ob + part * E.part_stride);
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const W x = part ? r1[w] : r0[w];
                        const W d = al ? mul_shoup<W>(dv.v[w], ai, ais, p) : dv.v[w];
                        W r = add<W>(d, mul_shoup<W>(sub<W>(x, cv.v[w], p), bi, bis, p), p);
                        if (E.dbl) r = add<W>(r, r, p);
                        if (part == 0) r = add<W>(r, kk, p);
                        if (G.add) r = add<W>(r, av.v[w], p);
                        dv.v[w] = r;
                    }
                    store4<W>(dp, dv.v[0], dv.v[1], dv.v[2], dv.v[3]);
                }
            } else {
                store4<W>((W*)A.out0 + ob, r0[0], r0[1], r0[2], r0[3]);
                store4<W>((W*)A.out1 + ob, r1[0], r1[1], r1[2], r1[3]);
            }
        }
    }
}

// ks_dot with the keys read through KeyTab: `a` regenerated (cooperative ChaCha over 4-lane groups at u32, each lane
// 4 consecutive slots), `b` dense or packed in the key's slot order; outputs as ks_dot (automorphism, c0 modes). K = 1
// at u32 runs ks_dot_seg instead (below).
template <class W, class MM, int KMAX, bool PACKED>
__global__ void __launch_bounds__(BX) ks_dot_key(KsKeyArgs G) {
    using Wd = WideOf<W>;
    const KsArgs& A = G.ks;
    const KeyTab& T = G.key;
    const uint32_t t = A.rows[blockIdx.y];
    const PrimeConst& q = A.pr[t];
    const W p = (W)q.p;
    const uint32_t gtid = blockIdx.x * BX + threadIdx.x;
    const int base = gtid * 4;
    const uint32_t lane = threadIdx.x & 31u;
    const int dt = (int)t < A.nq ? A.digit_of[t] : -1;
    const size_t roff = (size_t)t * N + base;
    const int slot = kskey::slot_of(T, (int)t, A.nq);
    for (int r = 0; r < A.R; ++r) {
        Wd s0[KMAX][4], s1[KMAX][4];
#pragma unroll
        for (int c = 0; c < KMAX; ++c)
#pragma unroll
            for (int w = 0; w < 4; ++w) s0[c][w] = s1[c][w] = 0;
        const void* kb = T.kb[r];
        const uint32_t* seed = T.seed[r];
        for (int d = 0; d < A.ndig; ++d) {
            W a4[4];
            if (d < T.nregen) {
                chacha::regen_slots4<W, MM>(seed, lane, (uint32_t)base, (uint32_t)d, q, chacha::TAG_KSKA, a4);
            } else {  // resident dense `a` (full layout, prime-id rows)
                const V4<W> av = ldg4<W>((const W*)A.ka[r] + d * A.key_digit_stride + roff);
#pragma unroll
                for (int w = 0; w < 4; ++w) a4[w] = av.v[w];
            }
            const V4<W> b4 = kskey::b_at4<W, PACKED>(T, kb, (size_t)d * T.srows + slot, (uint32_t)base);
#pragma unroll
            for (int c = 0; c < KMAX; ++c) {
                if (c < A.K) {
                    const V4<W> x4 = (d == dt) ? load4<W>((const W*)A.c1 + c * A.ct_stride + roff)
                                               : load4<W>((const W*)A.modup + c * A.mu_ct_stride + d * A.mu_digit_stride + roff);
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        s0[c][w] += (Wd)x4.v[w] * b4.v[w];
                        s1[c][w] += (Wd)x4.v[w] * a4[w];
                    }
                }
            }
        }
        const uint32_t k = A.autok[r];
#pragma unroll
        for (int c = 0; c < KMAX; ++c) {
            if (c < A.K) {
                V4<W> c04{};
                if (A.c0_mode & 3) c04 = load4<W>((const W*)A.c0 + c * A.ct_stride + roff);
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    W r0 = MM::reduce(s0[c][w], q);
                    W r1 = MM::reduce(s1[c][w], q);
                    if ((A.c0_mode & 3) == 1) {
                        if ((int)t < A.nq) r0 = add<W>(r0, mul_shoup<W>(c04.v[w], ((const W*)A.pmod)[t], ((const W*)A.pmods)[t], p), p);
                    } else if ((A.c0_mode & 3) == 2) {
                        r0 = add<W>(r0, c04.v[w], p);
                    }
                    const uint32_t i = (uint32_t)base + w;
                    const uint32_t o = k == 1 ? i : automorph_slot(k, i);
                    const size_t ob = c * A.out_ct_stride + r * A.out_rot_stride + (size_t)t * N + o;
                    if (A.c0_mode & KS_ACC) {
                        r0 = add<W>(r0, ((const W*)A.out0)[ob], p);
                        r1 = add<W>(r1, ((const W*)A.out1)[ob], p);
                    }
                    ((W*)A.out0)[ob] = r0;
                    ((W*)A.out1)[ob] = r1;
                }
            }
        }
    }
}

// ks_dot_key at K = 1 (u32) on warp-sized row segments: a warp owns slots R * 256.. of row t, 8 consecutive per lane.
// Its lanes expand `a` one 16-slot ChaCha block per thread (two digits per pass, through shared memory), and the
// automorphism maps the segment onto one segment (in the bit-reversed order a slot's segment decides its image's), so
// the outputs are permuted inside the warp.
// Each digit first asks L2 for the next digit's lines of the warp's segment (`b`, the ModUp digit or c1, a resident `a`),
// so its loads hit L2 instead of DRAM. At most 80 registers: 3 blocks per SM (shared memory holds 3).
template <class MM, bool PACKED>
__global__ void __launch_bounds__(256, 3) ks_dot_seg(KsKeyArgs G) {
    using W = uint32_t;
    using Wd = WideOf<W>;
    __shared__ W sa[8][2][264];
    __shared__ W so[8][264];
    const KsArgs& A = G.ks;
    const KeyTab& T = G.key;
    const uint32_t t = A.rows[blockIdx.y];
    const PrimeConst& q = A.pr[t];
    const W p = (W)q.p;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const uint32_t R = blockIdx.x * 8 + wp;
    const uint32_t base = R * 256 + 8 * lane;
    const int dt = (int)t < A.nq ? A.digit_of[t] : -1;
    const size_t roff = (size_t)t * N + base;
    const int slot = kskey::slot_of(T, (int)t, A.nq);
    const int nreg = min(A.ndig, T.nregen);
    auto pad = [](int e) { return e + (e >> 5); };
    for (int r = 0; r < A.R; ++r) {
        Wd s0[8], s1[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) s0[u] = s1[u] = 0;
        const void* kb = T.kb[r];
        uint32_t key[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) key[i] = __ldg(T.seed[r] + i);
        for (int d = 0; d < A.ndig; ++d) {
            if (d + 1 < A.ndig) {
                // digit e's lines of the segment, 8 per source: lanes 0-7 `b`, 8-15 the ModUp digit (c1), 16-23 a resident `a`
                const int e = d + 1;
                const size_t seg = (size_t)t * N + R * 256;
                const char* p = nullptr;
                if (lane < 8) {
                    const size_t row = ((size_t)e * T.srows + slot) * T.row_words;
                    p = PACKED ? (const char*)((const uint32_t*)kb + row + (((uint64_t)R * 256 * (uint32_t)T.bits) >> 5))
                               : (const char*)((const W*)kb + row + R * 256);
                } else if (lane < 16) {
                    p = (const char*)((e == dt) ? (const W*)A.c1 + seg : (const W*)A.modup + e * A.mu_digit_stride + seg);
                } else if (lane < 24 && e >= nreg) {
                    p = (const char*)((const W*)A.ka[r] + e * A.key_digit_stride + seg);
                }
                if (lane < 8) {
                    // the key's lines as evict-first (a 16-byte aligned bulk prefetch with an evict-first policy), the
                    // segment's bytes only (32 bits per 128-byte line of a packed row)
                    if (PACKED && lane * 4 >= (int)T.bits) p = nullptr;
                    uint64_t pol;
                    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
                    const char* q16 = (const char*)((uintptr_t)(p + 128 * lane) & ~(uintptr_t)15);
                    if (p) asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], 128, %1;" ::"l"(q16), "l"(pol) : "memory");
                } else if (p)
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(p + 128 * (lane & 7)));
            }
            if ((d & 1) == 0 && d < nreg) {
                // digits d, d + 1: lanes 0-15 / 16-31 expand the segment's 16 blocks of one digit each
                const int dd = d + (lane >> 4);
                __syncwarp();
                if (dd < nreg) {
                    uint32_t v[16];
                    chacha::regen_block16(key, R * 16 + (lane & 15), (uint32_t)dd, p, N >> 4, chacha::TAG_KSKA, v);
#pragma unroll
                    for (int k = 0; k < 16; ++k) sa[wp][lane >> 4][pad(16 * (lane & 15) + k)] = v[k];
                }
                __syncwarp();
            }
            W a[8];
            if (d < nreg) {
#pragma unroll
                for (int u = 0; u < 8; ++u) a[u] = sa[wp][d & 1][pad(8 * lane + u)];
            } else {  // resident dense `a` (full layout, prime-id rows)
                const W* ka = (const W*)A.ka[r] + d * A.key_digit_stride + roff;
                const V4<W> a0 = ldg4<W>(ka), a1 = ldg4<W>(ka + 4);
#pragma unroll
                for (int u = 0; u < 4; ++u) a[u] = a0.v[u], a[4 + u] = a1.v[u];
            }
            const size_t brow = (size_t)d * T.srows + slot;
            const V4<W> b0 = kskey::b_at4<W, PACKED>(T, kb, brow, base), b1 = kskey::b_at4<W, PACKED>(T, kb, brow, base + 4);
            const W* xp = (d == dt) ? (const W*)A.c1 + roff : (const W*)A.modup + d * A.mu_digit_stride + roff;
            const V4<W> x0 = load4<W>(xp), x1 = load4<W>(xp + 4);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                s0[u] += (Wd)x0.v[u] * b0.v[u];
                s0[4 + u] += (Wd)x1.v[u] * b1.v[u];
                s1[u] += (Wd)x0.v[u] * a[u];
                s1[4 + u] += (Wd)x1.v[u] * a[4 + u];
            }
        }
        W r0[8], r1[8];
        V4<W> c0[2] = {};
        if (A.c0_mode & 3) c0[0] = load4<W>((const W*)A.c0 + roff), c0[1] = load4<W>((const W*)A.c0 + roff + 4);
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            r0[u] = MM::reduce(s0[u], q);
            r1[u] = MM::reduce(s1[u], q);
            const W cv = c0[u >> 2].v[u & 3];
            if ((A.c0_mode & 3) == 1) {
                if ((int)t < A.nq) r0[u] = add<W>(r0[u], mul_shoup<W>(cv, ((const W*)A.pmod)[t], ((const W*)A.pmods)[t], p), p);
            } else if ((A.c0_mode & 3) == 2) {
                r0[u] = add<W>(r0[u], cv, p);
            }
        }
        const uint32_t k = A.autok[r];
        uint32_t o[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) o[u] = k == 1 ? base + u : automorph_slot(k, base + u);
        const uint32_t R2 = o[0] >> 8;
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            __syncwarp();
#pragma unroll
            for (int u = 0; u < 8; ++u) so[wp][pad(o[u] & 255)] = part ? r1[u] : r0[u];
            __syncwarp();
            W x[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) x[u] = so[wp][pad(8 * lane + u)];
            W* g = (W*)(part ? A.out1 : A.out0) + r * A.out_rot_stride + (size_t)t * N + R2 * 256 + 8 * lane;
            if (A.c0_mode & KS_ACC) {
                const V4<W> y0 = load4<W>(g), y1 = load4<W>(g + 4);
#pragma unroll
                for (int u = 0; u < 4; ++u) x[u] = add<W>(x[u], y0.v[u], p), x[4 + u] = add<W>(x[4 + u], y1.v[u], p);
            }
            store4<W>(g, x[0], x[1], x[2], x[3]);
            store4<W>(g + 4, x[4], x[5], x[6], x[7]);
        }
    }
}

template <class W, class MM>
__global__ void ksk_expand(KskExpandArgs A) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y, d = A.d0 + (int)blockIdx.z;
    const PrimeConst& c = A.pr[A.prime_ids[m]];
    ((W*)A.out)[((size_t)blockIdx.z * A.nmod + m) * N + slot] = chacha::expand_slot<W, MM>(A.seed, slot, (uint32_t)d, c, chacha::TAG_KSKA);
}

template <class W, class MM>
int dot_l(const KsArgs* a, cudaStream_t s) {
    dim3 grid(N / BX, a->nrows);
    if (a->K <= 1) ks_dot<W, MM, 1><<<grid, BX, 0, s>>>(*a);
    else if (a->K <= 2) ks_dot<W, MM, 2><<<grid, BX, 0, s>>>(*a);
    else if (a->K <= 4) ks_dot<W, MM, 4><<<grid, BX, 0, s>>>(*a);
    else if (a->K <= 8) ks_dot<W, MM, 8><<<grid, BX, 0, s>>>(*a);
    else return -1;
    return (int)cudaGetLastError();
}
template <class W, class MM, bool EPI, int TAB>
int regen_k(const KsRegenArgs* a, cudaStream_t s) {
    dim3 grid(N / (BX * 4), a->ks.nrows);
    const int K = a->ks.K;
    if (K <= 1) ks_dot_regen<W, MM, 1, EPI, TAB><<<grid, BX, 0, s>>>(*a);
    else if (K <= 2) ks_dot_regen<W, MM, 2, EPI, TAB><<<grid, BX, 0, s>>>(*a);
    else if (K <= 4) ks_dot_regen<W, MM, 4, EPI, TAB><<<grid, BX, 0, s>>>(*a);
    else return -2;
    return (int)cudaGetLastError();
}
template <class W, class MM>
int regen_l(const KsRegenArgs* a, cudaStream_t s) {
    if (a->key.kb && (a->key.bits < 0 || a->key.bits > 32 || (a->key.bits > 0 && sizeof(W) != 4))) return -1;
    if (a->add && !a->use_epi) return -1;
    const int tab = !a->key.kb ? 0 : a->key.bits ? 2 : 1;
    if (a->use_epi) {
        if (tab == 0) return regen_k<W, MM, true, 0>(a, s);
        if (tab == 1) return regen_k<W, MM, true, 1>(a, s);
        return regen_k<W, MM, true, 2>(a, s);
    }
    if (tab == 0) return regen_k<W, MM, false, 0>(a, s);
    if (tab == 1) return regen_k<W, MM, false, 1>(a, s);
    return regen_k<W, MM, false, 2>(a, s);
}
template <class W, class MM, bool PACKED>
int key_k(const KsKeyArgs* a, cudaStream_t s) {
    dim3 grid(N / (BX * 4), a->ks.nrows);
    const int K = a->ks.K;
    if constexpr (sizeof(W) == 4) {
        if (K == 1) {
            ks_dot_seg<MM, PACKED><<<dim3(N / 2048, a->ks.nrows), 256, 0, s>>>(*a);
            return (int)cudaGetLastError();
        }
    }
    if (K <= 1) ks_dot_key<W, MM, 1, PACKED><<<grid, BX, 0, s>>>(*a);
    else if (K <= 2) ks_dot_key<W, MM, 2, PACKED><<<grid, BX, 0, s>>>(*a);
    else if (K <= 4) ks_dot_key<W, MM, 4, PACKED><<<grid, BX, 0, s>>>(*a);
    else return -2;
    return (int)cudaGetLastError();
}
template <class W, class MM>
int key_l(const KsKeyArgs* a, cudaStream_t s) {
    if (a->key.bits < 0 || a->key.bits > 32 || (a->key.bits > 0 && sizeof(W) != 4)) return -1;
    return a->key.bits ? key_k<W, MM, true>(a, s) : key_k<W, MM, false>(a, s);
}
template <class W, class MM>
int expand_l(const KskExpandArgs* a, cudaStream_t s) {
    if (a->d0 < 0 || a->d0 >= a->dnum) return -1;
    ksk_expand<W, MM><<<dim3(N / 128, a->nmod, a->dnum - a->d0), 128, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
}  // namespace ks
}  // namespace pc
