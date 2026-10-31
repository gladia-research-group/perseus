// Fused CtS stage core generic over (W, MM): hoisted baby-step rotations + BSGS plaintext dot in
// the extended basis, K ciphertexts per launch, rotations never materialized. For output position o of row t (gather
// form; the automorphism maps aligned 32-slot blocks to aligned 32-slot blocks in the bit-reversed evaluation order):
//   rot_0[o] = (P c0[o], P c1[o])  (0 on special rows)
//   rot_i[o] = (P c0[s] + sum_d D_d[s] b_{i,d}[s],  sum_d D_d[s] a_{i,d}[s]),  s = sigma_i^{-1}(o)
//   res_j[o] = sum_i rot_i[o] * pt_{B j + i}[o]          (j < G; diagonals past npts skipped)
// Accumulators are reduced every 8 baby steps (headroom of the double word).
// REGEN (cts_fused_regen): the baby keys are read through KeyTab, `a` regenerated in-kernel. A warp's 32
// outputs gather from one aligned 32-slot source block per baby step, so the warp regenerates that block (2 ChaCha
// blocks per digit; 4 digits per round, 8 lanes each, cooperative over 4-lane groups at u32) into shared memory and
// every lane reads its slot from there: the key traffic of a stage drops to `b` alone.
#pragma once
#include "arith.cuh"
#include "ks.cuh"

namespace pc {
namespace cts {
constexpr int BX = 128;
// Hybrid digits a stage reads (registers / shared `a` blocks): 6 at u32 (n32: dnum 6), 8 at other words (n64:
// dnum 7). The launchers refuse more.
template <class W>
constexpr int DMAX = sizeof(W) == 4 ? 6 : 8;

template <class W, class MM, int KMAX, int GMAX, bool REGEN>
__global__ void __launch_bounds__(BX) cts_fused(CtsRegenArgs R) {
    using Wd = WideOf<W>;
    const CtsArgs& A = R.cts;
    __shared__ W sha[REGEN ? BX / 32 : 1][DMAX<W>][32];
    const int cg = blockIdx.x % A.ngrp;
    const int c0 = cg * KMAX;
    const int Kg = min(KMAX, A.K - c0);
    const W* X = (const W*)A.x + c0 * A.ct_stride;
    const W* MU = (const W*)A.modup + c0 * A.mu_ct_stride;
    W* RES = (W*)A.res + (size_t)c0 * A.G * A.ct_stride;
    const uint32_t t = A.rows[blockIdx.y];
    const PrimeConst& q = A.pr[t];
    const W p = (W)q.p;
    const bool isq = (int)t < A.nq;
    const int prow = isq ? (int)t : A.pt_prow + ((int)t - A.nq);
    const int o = (blockIdx.x / A.ngrp) * BX + threadIdx.x;
    const int po = ptidx(o, A.enc);
    const int dt = isq ? A.digit_of[t] : -1;
    const size_t rowoff = (size_t)t * N;
    const W pm = ((const W*)A.pmod)[t], pms = ((const W*)A.pmods)[t];
    Wd acc0[KMAX][GMAX], acc1[KMAX][GMAX];
#pragma unroll
    for (int c = 0; c < KMAX; ++c)
#pragma unroll
        for (int j = 0; j < GMAX; ++j) acc0[c][j] = acc1[c][j] = 0;

    for (int i = 0; i < A.B; ++i) {
        W v0[KMAX], v1[KMAX];
        if (i == 0) {
#pragma unroll
            for (int c = 0; c < KMAX; ++c) {
                if (c < Kg && isq) {
                    const W* xc = X + c * A.ct_stride + rowoff;
                    v0[c] = mul_shoup<W>(xc[o], pm, pms, p);
                    v1[c] = mul_shoup<W>(xc[A.part_stride + o], pm, pms, p);
                } else {
                    v0[c] = v1[c] = 0;
                }
            }
        } else {
            const uint32_t s = automorph_slot(A.kinv[i - 1], o);
            Wd s0[KMAX], s1[KMAX];
#pragma unroll
            for (int c = 0; c < KMAX; ++c) s0[c] = s1[c] = 0;
            const W* kb = nullptr;
            const W* ka = nullptr;
            if constexpr (!REGEN) {
                kb = (const W*)A.kb[i - 1] + rowoff + s;
                ka = (const W*)A.ka[i - 1] + rowoff + s;
            }
            const uint32_t* seed = nullptr;
            size_t krow = 0;
            const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
            if constexpr (REGEN) {
                seed = R.key.seed[i - 1];
                krow = (size_t)kskey::slot_of(R.key, (int)t, A.nq);
                if constexpr (sizeof(W) == 4) {
                    const uint32_t sb = s & ~31u;  // the warp's source block (uniform)
                    const uint32_t g = lane >> 3, j = lane & 7u;
                    const int nreg = min(A.ndig, R.key.nregen);
                    for (int d0 = 0; d0 < nreg; d0 += 4) {
                        const int d = d0 + (int)g;
                        if (d < nreg) {  // whole 4-lane groups agree
                            uint32_t a4[4];
                            chacha::regen4(seed, lane, (sb + 4 * j) >> 4, (uint32_t)d, (uint32_t)p, N >> 4, chacha::TAG_KSKA, a4);
#pragma unroll
                            for (int w = 0; w < 4; ++w) sha[warp][d][4 * j + w] = (W)a4[w];
                        }
                    }
                    __syncwarp();
                }
            }
#pragma unroll
            for (int d = 0; d < DMAX<W>; ++d) {
                if (d >= A.ndig) break;
                W b, a;
                if constexpr (REGEN) {
                    b = kskey::b_at<W>(R.key, R.key.kb[i - 1], (size_t)d * R.key.srows + krow, s);
                    if (d >= R.key.nregen) a = __ldg((const W*)A.ka[i - 1] + rowoff + s + d * A.key_digit_stride);
                    else if constexpr (sizeof(W) == 4) a = sha[warp][d][s & 31u];
                    else a = chacha::expand_slot<W, MM>(seed, s, (uint32_t)d, q, chacha::TAG_KSKA);
                } else {
                    b = __ldg(kb + d * A.key_digit_stride);
                    a = __ldg(ka + d * A.key_digit_stride);
                }
#pragma unroll
                for (int c = 0; c < KMAX; ++c) {
                    if (c < Kg) {
                        const W xv = (d == dt) ? X[c * A.ct_stride + A.part_stride + rowoff + s]
                                               : MU[c * A.mu_ct_stride + d * A.mu_digit_stride + rowoff + s];
                        s0[c] += (Wd)xv * b;
                        s1[c] += (Wd)xv * a;
                    }
                }
            }
#pragma unroll
            if constexpr (REGEN && sizeof(W) == 4) __syncwarp();  // the shared `a` block is rewritten next baby step
            for (int c = 0; c < KMAX; ++c) {
                if (c < Kg) {
                    W r0 = MM::reduce(s0[c], q);
                    if (isq) r0 = add<W>(r0, mul_shoup<W>(X[c * A.ct_stride + rowoff + s], pm, pms, p), p);
                    v0[c] = r0;
                    v1[c] = MM::reduce(s1[c], q);
                } else {
                    v0[c] = v1[c] = 0;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < GMAX; ++j) {
            const int dg = A.B * j + i;
            if (j < A.G && dg < A.npts) {
                const W w = __ldg((const W*)A.pts[dg] + (size_t)prow * A.ptw + po);
#pragma unroll
                for (int c = 0; c < KMAX; ++c) {
                    acc0[c][j] += (Wd)v0[c] * w;
                    acc1[c][j] += (Wd)v1[c] * w;
                }
            }
        }
        if ((i & 7) == 7) {  // 8 products of < 2^56 -> reduce to keep headroom for the next 8
#pragma unroll
            for (int c = 0; c < KMAX; ++c)
#pragma unroll
                for (int j = 0; j < GMAX; ++j) {
                    acc0[c][j] = MM::reduce(acc0[c][j], q);
                    acc1[c][j] = MM::reduce(acc1[c][j], q);
                }
        }
    }
#pragma unroll
    for (int c = 0; c < KMAX; ++c) {
        if (c >= Kg) break;
#pragma unroll
        for (int j = 0; j < GMAX; ++j) {
            if (j < A.G) {
                W* r = RES + (c * A.G + j) * A.ct_stride + rowoff;
                r[o] = MM::reduce(acc0[c][j], q);
                r[A.part_stride + o] = MM::reduce(acc1[c][j], q);
            }
        }
    }
}

// cts_fused_regen at u32: the products of cts_fused, with the warp's `a` blocks regenerated as a stream of 16-word
// ChaCha items. A baby step needs 2 items (block halves h) per regenerated digit; a rejected word is retried at block
// counter + t N/16 (the KSKA rule: t < 16, the 16th try kept as is). Each pass gives every 2-lane group one item,
// pending retries first, then fresh items of the next baby steps that fit the warp's 12 digit buffers, so no group
// idles on a partial round or on another group's retry. A baby step's products run once all its items are done;
// accepted words go from the ChaCha column layout straight to shared memory. Same results, same summation order.
namespace rq {
// ChaCha12 block over a 2-lane group (lane k holds columns 2k, 2k+1: words 2k, 2k+1, 4+2k, 5+2k, 8+2k, 9+2k,
// 12+2k, 13+2k in that order in o[]); the diagonal rounds rotate rows b, c, d across the pair (4 shuffles each way)
__device__ __forceinline__ void qr(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = __funnelshift_l(d, d, 16);
    c += d; b ^= c; b = __funnelshift_l(b, b, 12);
    a += b; d ^= a; d = __funnelshift_l(d, d, 8);
    c += d; b ^= c; b = __funnelshift_l(b, b, 7);
}
__device__ __forceinline__ void chacha_coop2(uint32_t gmask, uint32_t k, const uint32_t* key, uint32_t ctr, uint32_t w13,
                                             uint32_t w14, uint32_t tag, uint32_t o[8]) {
    const uint32_t a0i = k ? 0x79622d32u : 0x61707865u, a1i = k ? 0x6b206574u : 0x3320646eu;
    const uint32_t b0i = key[2 * k], b1i = key[2 * k + 1], c0i = key[4 + 2 * k], c1i = key[5 + 2 * k];
    const uint32_t d0i = k ? w14 : ctr, d1i = k ? tag : w13;
    uint32_t a0 = a0i, a1 = a1i, b0 = b0i, b1 = b1i, c0 = c0i, c1 = c1i, d0 = d0i, d1 = d1i;
#pragma unroll
    for (int r = 0; r < 12; r += 2) {
        qr(a0, b0, c0, d0);
        qr(a1, b1, c1, d1);
        // diagonal: column i takes b_{i+1}, c_{i+2}, d_{i+3}
        {
            const uint32_t ob0 = __shfl_xor_sync(gmask, b0, 1, 2);
            const uint32_t oc0 = __shfl_xor_sync(gmask, c0, 1, 2), oc1 = __shfl_xor_sync(gmask, c1, 1, 2);
            const uint32_t od1 = __shfl_xor_sync(gmask, d1, 1, 2);
            b0 = b1; b1 = ob0;
            c0 = oc0; c1 = oc1;
            d1 = d0; d0 = od1;
        }
        qr(a0, b0, c0, d0);
        qr(a1, b1, c1, d1);
        {
            const uint32_t ob1 = __shfl_xor_sync(gmask, b1, 1, 2);
            const uint32_t oc0 = __shfl_xor_sync(gmask, c0, 1, 2), oc1 = __shfl_xor_sync(gmask, c1, 1, 2);
            const uint32_t od0 = __shfl_xor_sync(gmask, d0, 1, 2);
            b1 = b0; b0 = ob1;
            c0 = oc0; c1 = oc1;
            d0 = d1; d1 = od0;
        }
    }
    o[0] = a0 + a0i; o[1] = a1 + a1i; o[2] = b0 + b0i; o[3] = b1 + b1i;
    o[4] = c0 + c0i; o[5] = c1 + c1i; o[6] = d0 + d0i; o[7] = d1 + d1i;
}
// queue entry: baby i (bits 0-5), digit (6-8), half (9), try t (10-13), words still needed (16-31)
__device__ __forceinline__ uint32_t mk(uint32_t i, uint32_t d, uint32_t h, uint32_t t, uint32_t need) {
    return i | d << 6 | h << 9 | t << 10 | need << 16;
}
}  // namespace rq

// Blocks per SM the launch bound asks for: the K = 1 four-giant-step variant (the StC stages) at 7 (70 registers;
// unbounded it takes 96, 5 blocks: its loads wait with too few warps), the others unbounded (0: no bound).
template <int KMAX, int GMAX>
constexpr int rq_blocks() {
    return KMAX == 1 && GMAX == 4 ? 7 : 0;
}

template <class MM, int KMAX, int GMAX>
__global__ void __launch_bounds__(BX, rq_blocks<KMAX, GMAX>()) cts_fused_rq(CtsRegenArgs R) {
    using W = uint32_t;
    using Wd = WideOf<W>;
    const CtsArgs& A = R.cts;
    constexpr int NW = BX / 32;
    constexpr int NBUF = 12;  // digit buffers per warp
    __shared__ W sha[NW][NBUF][32];
    __shared__ uint32_t rqs[NW][16];
    const int cg = blockIdx.x % A.ngrp;
    const int c0 = cg * KMAX;
    const int Kg = min(KMAX, A.K - c0);
    const W* X = (const W*)A.x + c0 * A.ct_stride;
    const W* MU = (const W*)A.modup + c0 * A.mu_ct_stride;
    W* RES = (W*)A.res + (size_t)c0 * A.G * A.ct_stride;
    const uint32_t t = A.rows[blockIdx.y];
    const PrimeConst& q = A.pr[t];
    const W p = (W)q.p;
    const bool isq = (int)t < A.nq;
    const int prow = isq ? (int)t : A.pt_prow + ((int)t - A.nq);
    const int o = (blockIdx.x / A.ngrp) * BX + threadIdx.x;
    const int po = ptidx(o, A.enc);
    const int dt = isq ? A.digit_of[t] : -1;
    const size_t rowoff = (size_t)t * N;
    const W pm = ((const W*)A.pmod)[t], pms = ((const W*)A.pmods)[t];
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t grp = lane >> 1, kq = lane & 1u;
    const uint32_t gmask = 3u << (lane & ~1u);
    const uint32_t m_p = (0xFFFFFFFFu / p) * p, mu32 = 0xFFFFFFFFu / p;
    const size_t krow = (size_t)kskey::slot_of(R.key, (int)t, A.nq);
    const int nreg = min(A.ndig, R.key.nregen);
    const int nbaby = nreg > 0 ? NBUF / nreg : NBUF;  // baby steps in flight
    Wd acc0[KMAX][GMAX], acc1[KMAX][GMAX];
#pragma unroll
    for (int c = 0; c < KMAX; ++c)
#pragma unroll
        for (int j = 0; j < GMAX; ++j) acc0[c][j] = acc1[c][j] = 0;

    auto l2 = [](const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); };
    // K = 1: a baby step's products first ask L2 for the next baby step's lines, so its loads (serialized digit by
    // digit) hit L2 instead of DRAM. Baby step i's lines (1 <= i < B), one per lane: lanes 2d, 2d+1 the packed `b`
    // words of digit d (two lines at most), 16 + d the ModUp digit d (c1 at d = dt), 24 + d the resident `a` of
    // d >= nreg, 30 + j the diagonal of giant step j (j < 2)
    auto pf = [&](int i) {
        if (i >= A.B) return;
        const uint32_t sb = automorph_slot(A.kinv[i - 1], o) & ~31u;
        const int nd = A.ndig;
        if ((int)lane < 2 * nd) {
            const int d = lane >> 1;
            const uint32_t* r = (const uint32_t*)R.key.kb[i - 1] + ((size_t)d * R.key.srows + krow) * R.key.row_words;
            if (R.key.bits) {
                const uint64_t b0 = (uint64_t)sb * (uint32_t)R.key.bits;
                l2(r + ((lane & 1) ? (b0 + 32u * (uint32_t)R.key.bits) >> 5 : b0 >> 5));
            } else if ((lane & 1) == 0) {
                l2((const W*)r + sb);
            }
        } else if (lane >= 16 && (int)lane < 16 + nd) {
            const int d = lane - 16;
            for (int c = 0; c < Kg; ++c)
                l2(d == dt ? X + c * A.ct_stride + A.part_stride + rowoff + sb : MU + c * A.mu_ct_stride + d * A.mu_digit_stride + rowoff + sb);
        } else if (lane >= 24 && (int)lane < 24 + nd) {
            const int d = lane - 24;
            if (d >= nreg) l2((const W*)A.ka[i - 1] + rowoff + sb + d * A.key_digit_stride);
        } else if (lane >= 30) {
            const int dg = A.B * (lane - 30) + i;
            if ((int)(lane - 30) < A.G && dg < A.npts) l2((const W*)A.pts[dg] + (size_t)prow * A.ptw + po);
        }
    };
    // products of baby step i (its regenerated `a` digits in buffers (i % nbaby) * nreg ..) into the accumulators
    auto mac = [&](int i) {
        if constexpr (KMAX == 1) pf(i + 1);
        W v0[KMAX], v1[KMAX];
        if (i == 0) {
#pragma unroll
            for (int c = 0; c < KMAX; ++c) {
                if (c < Kg && isq) {
                    const W* xc = X + c * A.ct_stride + rowoff;
                    v0[c] = mul_shoup<W>(xc[o], pm, pms, p);
                    v1[c] = mul_shoup<W>(xc[A.part_stride + o], pm, pms, p);
                } else {
                    v0[c] = v1[c] = 0;
                }
            }
        } else {
            const uint32_t s = automorph_slot(A.kinv[i - 1], o);
            Wd s0[KMAX], s1[KMAX];
#pragma unroll
            for (int c = 0; c < KMAX; ++c) s0[c] = s1[c] = 0;
            const W* sa = &sha[warp][(i % nbaby) * nreg][0];
#pragma unroll
            for (int d = 0; d < DMAX<W>; ++d) {
                if (d >= A.ndig) break;
                const W b = kskey::b_at<W>(R.key, R.key.kb[i - 1], (size_t)d * R.key.srows + krow, s);
                const W a = d < nreg ? sa[d * 32 + (s & 31u)] : __ldg((const W*)A.ka[i - 1] + rowoff + s + d * A.key_digit_stride);
#pragma unroll
                for (int c = 0; c < KMAX; ++c) {
                    if (c < Kg) {
                        const W xv = (d == dt) ? X[c * A.ct_stride + A.part_stride + rowoff + s]
                                               : MU[c * A.mu_ct_stride + d * A.mu_digit_stride + rowoff + s];
                        s0[c] += (Wd)xv * b;
                        s1[c] += (Wd)xv * a;
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < KMAX; ++c) {
                if (c < Kg) {
                    W r0 = MM::reduce(s0[c], q);
                    if (isq) r0 = add<W>(r0, mul_shoup<W>(X[c * A.ct_stride + rowoff + s], pm, pms, p), p);
                    v0[c] = r0;
                    v1[c] = MM::reduce(s1[c], q);
                } else {
                    v0[c] = v1[c] = 0;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < GMAX; ++j) {
            const int dg = A.B * j + i;
            if (j < A.G && dg < A.npts) {
                const W w = __ldg((const W*)A.pts[dg] + (size_t)prow * A.ptw + po);
#pragma unroll
                for (int c = 0; c < KMAX; ++c) {
                    acc0[c][j] += (Wd)v0[c] * w;
                    acc1[c][j] += (Wd)v1[c] * w;
                }
            }
        }
        if ((i & 7) == 7) {  // 8 products of < 2^56 -> reduce to keep headroom for the next 8
#pragma unroll
            for (int c = 0; c < KMAX; ++c)
#pragma unroll
                for (int j = 0; j < GMAX; ++j) {
                    acc0[c][j] = MM::reduce(acc0[c][j], q);
                    acc1[c][j] = MM::reduce(acc1[c][j], q);
                }
        }
    };

    mac(0);
    const int nit = 2 * nreg;  // items per baby step
    const int nb = A.B - 1;    // baby steps with a key
    const int total = nb * nit;
    // warp-uniform: next fresh item, queue length, next baby step to multiply
    int fi = 0, nq = 0, mb = 1;
    const uint32_t n16 = N >> 4;
    while (mb <= nb) {
        const bool mine = kq == 0 && (int)grp < nq;
        const uint32_t qe = mine ? rqs[warp][grp] : 0u;
        if (fi >= mb * nit && __ballot_sync(0xFFFFFFFFu, mine && (qe & 63u) == (uint32_t)mb) == 0) {
            __syncwarp();  // baby mb's words are in sha
            mac(mb);
            __syncwarp();  // its buffers are free for baby mb + nbaby
            ++mb;
            continue;
        }
        const int lim = min(total, (mb + nbaby - 1) * nit);
        const int nf = max(0, min(16 - nq, lim - fi));
        uint32_t ent = 0;
        bool act = false;
        if ((int)grp < nq) {
            ent = rqs[warp][grp];
            act = true;
        } else if ((int)grp < nq + nf) {
            const int f = fi + (int)grp - nq;
            const int r = f % nit;
            ent = rq::mk((uint32_t)(f / nit + 1), (uint32_t)(r >> 1), (uint32_t)(r & 1), 0, 0xFFFFu);
            act = true;
        }
        __syncwarp();  // the queue is read before it is rewritten
        uint32_t nn = 0;
        const uint32_t bi = ent & 63u, d = (ent >> 6) & 7u, h = (ent >> 9) & 1u, tt = (ent >> 10) & 15u, need = ent >> 16;
        if (act) {  // uniform over the 2-lane group
            const uint32_t* seed = R.key.seed[bi - 1];
            const uint32_t sb = automorph_slot(A.kinv[bi - 1], (uint32_t)o) & ~31u;
            uint32_t cw[8];
            rq::chacha_coop2(gmask, kq, seed, ((sb + 16u * h) >> 4) + tt * n16, d, (uint32_t)p, chacha::TAG_KSKA, cw);
            W* dst = &sha[warp][(bi % (uint32_t)nbaby) * nreg + d][16 * h];
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                const uint32_t w = 2u * kq + (m & 1) + 4u * (m >> 1), v = cw[m];
                if ((need >> w) & 1u) {
                    if (v < m_p || tt == 15u) {
                        const uint32_t r = v - __umulhi(v, mu32) * p;
                        dst[w] = r >= p ? r - p : r;
                    } else {
                        nn |= 1u << w;
                    }
                }
            }
            nn |= __shfl_xor_sync(gmask, nn, 1, 2);
        }
        const bool pub = act && kq == 0 && nn != 0;
        const uint32_t ball = __ballot_sync(0xFFFFFFFFu, pub);
        if (pub) rqs[warp][__popc(ball & ((1u << lane) - 1u))] = rq::mk(bi, d, h, tt + 1, nn);
        nq = __popc(ball);
        fi += nf;
        __syncwarp();
    }
#pragma unroll
    for (int c = 0; c < KMAX; ++c) {
        if (c >= Kg) break;
#pragma unroll
        for (int j = 0; j < GMAX; ++j) {
            if (j < A.G) {
                // pair_cts > 0 (K = 1 launches only): giant pair j / 2 at its own region, (c = 0, j % 2) inside it
                const size_t at = A.pair_cts > 0 ? (size_t)(j >> 1) * A.pair_cts + (j & 1) : (size_t)(c * A.G + j);
                W* r = RES + at * A.ct_stride + rowoff;
                r[o] = MM::reduce(acc0[c][j], q);
                r[A.part_stride + o] = MM::reduce(acc1[c][j], q);
            }
        }
    }
}

template <class W, class MM, bool REGEN>
int launch_any(const CtsRegenArgs* a, cudaStream_t s) {
    const int km = a->cts.kmax <= 1 ? 1 : a->cts.kmax <= 2 ? 2 : a->cts.kmax <= 4 ? 4 : 8;
    // G 3..4: the queued K=1 kernel only (pair_cts names the second pair's region)
    const bool g4 = REGEN && sizeof(W) == 4 && km == 1 && a->cts.K == 1 && a->cts.pair_cts > 0;
    if (a->cts.G > (g4 ? 4 : 2) || a->cts.ndig > DMAX<W>) return -1;
    if (a->cts.pair_cts > 0 && !g4) return -1;
    CtsRegenArgs b = *a;
    b.cts.ngrp = (a->cts.K + km - 1) / km;
    dim3 grid((N / BX) * b.cts.ngrp, a->cts.nrows);
    if constexpr (REGEN && sizeof(W) == 4) {
        if (a->cts.B > 64) return -1;  // queue entries hold the baby index in 6 bits
        if (g4) cts_fused_rq<MM, 1, 4><<<grid, BX, 0, s>>>(b);
        else if (km == 1) cts_fused_rq<MM, 1, 2><<<grid, BX, 0, s>>>(b);
        else if (km == 2) cts_fused_rq<MM, 2, 2><<<grid, BX, 0, s>>>(b);
        else if (km == 4) cts_fused_rq<MM, 4, 2><<<grid, BX, 0, s>>>(b);
        else cts_fused_rq<MM, 8, 2><<<grid, BX, 0, s>>>(b);
    } else {
        if (km == 1) cts_fused<W, MM, 1, 2, REGEN><<<grid, BX, 0, s>>>(b);
        else if (km == 2) cts_fused<W, MM, 2, 2, REGEN><<<grid, BX, 0, s>>>(b);
        else if (km == 4) cts_fused<W, MM, 4, 2, REGEN><<<grid, BX, 0, s>>>(b);
        else cts_fused<W, MM, 8, 2, REGEN><<<grid, BX, 0, s>>>(b);
    }
    return (int)cudaGetLastError();
}
template <class W, class MM>
int launch(const CtsArgs* a, cudaStream_t s) {
    CtsRegenArgs r{};
    r.cts = *a;
    return launch_any<W, MM, false>(&r, s);
}
template <class W, class MM>
int launch_regen(const CtsRegenArgs* a, cudaStream_t s) {
    if (a->key.bits < 0 || a->key.bits > 32 || (a->key.bits > 0 && sizeof(W) != 4) || a->cts.ndig > DMAX<W>) return -1;
    return launch_any<W, MM, true>(a, s);
}
}  // namespace cts
}  // namespace pc
