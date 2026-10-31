// Bootstrap raise kernels, generic over (W, MM); semantics in args.cuh (ModRaiseArgs, SmallIntArgs).
//  - mod_raise: FIDESlib compositeModRaise_ (each Garner term centred on its own prime, then re-reduced; the sum may
//    leave (-Q0/2, Q0/2], which is why the small-integer scaling that follows reconstructs from 3 limbs).
//  - small_int: FIDESlib smallIntDivRound_ (Garner, centre, fixed-point multiply by R and >> 192 with round half up,
//    reduce mod every output prime). Only the integer q must match FIDESlib; the residues are canonical, so the
//    reduction here uses the double-word Barrett instead of `%`. u32 words only (the launcher refuses others).
#pragma once
#include "arith.cuh"

namespace pc {
namespace bts {
constexpr int BX = 128;

template <class W, class MM>
__global__ void __launch_bounds__(BX) mod_raise(ModRaiseArgs A) {
    using Wd = WideOf<W>;
    const int t = A.r0 + blockIdx.y;
    const int c = blockIdx.z / A.nparts, part = blockIdx.z % A.nparts;
    const int i = blockIdx.x * BX + threadIdx.x;
    W* x = (W*)A.x + c * A.ct_stride + part * A.part_stride;
    const PrimeConst& qt = A.pr[t];
    const W pt = (W)qt.p;
    W acc = 0;
    for (int k = 0; k < A.d; ++k) {
        const PrimeConst& qk = A.pr[k];
        const W pk = (W)qk.p;
        W y = MM::mul(x[(size_t)k * N + i], (W)A.qhatinv[k], qk);
        // SwitchModulus(k -> t): centred lift of y, reduced mod q_t
        Wd v = y;
        if (y > (pk >> 1)) v += (Wd)pt - (Wd)(pk % pt);
        const W r = (W)(v % pt);
        acc = add<W>(acc, MM::mul(r, (W)A.qhat[k][t], qt), pt);
    }
    x[(size_t)t * N + i] = acc;
}

// x mod q for any 32-bit x (m = floor(2^32 / q), q odd): x - floor(x m / 2^32) q is in [0, 2q)
__device__ __forceinline__ uint32_t mod32(uint32_t x, uint32_t q, uint32_t m) {
    const uint32_t r = x - __umulhi(x, m) * q;
    return r >= q ? r - q : r;
}

// The same integers as FIDESlib smallIntDivRound_ with every per-coefficient array in registers (fixed trip counts,
// zero words past the live ones) and the Garner steps on 32-bit words: x mod q by mod32, the products by q_j^-1 as
// Shoup products (their companions in shared memory, once per block).
template <class W, class MM>
__global__ void __launch_bounds__(BX) small_int(SmallIntArgs A) {
    constexpr int XW = PC_SI_MAXW + PC_SI_MAXR;
    const SmallIntTab* tab = A.tab;
    const int part = blockIdx.y, c = blockIdx.z;
    const int idx = blockIdx.x * BX + threadIdx.x;
    const int k = tab->k, nw = tab->nw;
    __shared__ uint32_t sq[PC_SI_MAXK], sm[PC_SI_MAXK], sqi[PC_SI_MAXK][PC_SI_MAXK], sqis[PC_SI_MAXK][PC_SI_MAXK];
    if (threadIdx.x < PC_SI_MAXK * PC_SI_MAXK) {
        const int i = threadIdx.x / PC_SI_MAXK, j = threadIdx.x % PC_SI_MAXK;
        if (i < k) {
            const uint32_t qi = tab->q[i];
            if (j == 0) {
                sq[i] = qi;
                sm[i] = 0xFFFFFFFFu / qi;  // floor(2^32 / q): q odd
            }
            if (j < i) {
                const uint32_t w = tab->qinv[i][j];
                sqi[i][j] = w;
                sqis[i][j] = (uint32_t)(((uint64_t)w << 32) / qi);
            }
        }
    }
    __syncthreads();
    const W* src = (const W*)A.src + c * A.src_ct_stride + part * A.src_part_stride;
    uint32_t v[PC_SI_MAXK];
#pragma unroll
    for (int i = 0; i < PC_SI_MAXK; ++i) {
        v[i] = 0;
        if (i < k) {
            const uint32_t qi = sq[i], mi = sm[i];
            uint32_t t = mod32((uint32_t)src[(size_t)i * N + idx], qi, mi);
#pragma unroll
            for (int j = 0; j < i; ++j) {
                const uint32_t vj = mod32(v[j], qi, mi);
                t = t + qi - vj;
                t = t >= qi ? t - qi : t;
                t = mul_shoup<uint32_t>(t, sqi[i][j], sqis[i][j], qi);
            }
            v[i] = t;
        }
    }
    // x = v[k-1] q[k-2] ... + v[0] (mixed radix, at most k words)
    uint32_t x[XW];
#pragma unroll
    for (int w = 0; w < XW; ++w) x[w] = 0;
#pragma unroll
    for (int i = PC_SI_MAXK - 1; i >= 0; --i) {
        if (i == k - 1) x[0] = v[i];
        else if (i < k - 1) {
            const uint32_t qi = sq[i];
            uint64_t carry = v[i];
#pragma unroll
            for (int w = 0; w < PC_SI_MAXK; ++w) {
                const uint64_t t = (uint64_t)x[w] * qi + carry;
                x[w] = (uint32_t)t;
                carry = t >> 32;
            }
        }
    }
    bool neg = false;
    {
        int st = 0;  // compare x with Qk / 2 from the top word: 1 above, -1 below
#pragma unroll
        for (int w = PC_SI_MAXW - 1; w >= 0; --w)
            if (w < nw && st == 0 && x[w] != tab->Qkh[w]) st = x[w] > tab->Qkh[w] ? 1 : -1;
        if (st > 0) {
            neg = true;
            uint32_t borrow = 0;
#pragma unroll
            for (int w = 0; w < PC_SI_MAXW; ++w) {
                if (w < nw) {
                    const uint64_t t = (uint64_t)tab->Qk[w] - (uint64_t)x[w] - borrow;
                    borrow = (uint32_t)(t >> 63);
                    x[w] = (uint32_t)t;
                }
            }
        }
    }
    int nq = nw;
    if (tab->divide) {
        // x R (R's words past nr are zero in the table), + 2^191 (round half up), drop 6 words
        uint32_t prod[XW];
#pragma unroll
        for (int i = 0; i < XW; ++i) prod[i] = 0;
#pragma unroll
        for (int i = 0; i < PC_SI_MAXW; ++i) {
            if (i < nw) {
                uint64_t carry = 0;
#pragma unroll
                for (int j = 0; j < PC_SI_MAXR; ++j) {
                    const uint64_t t = (uint64_t)x[i] * tab->R[j] + prod[i + j] + carry;
                    prod[i + j] = (uint32_t)t;
                    carry = t >> 32;
                }
#pragma unroll
                for (int p_ = i + PC_SI_MAXR; p_ < XW; ++p_) {
                    const uint64_t t = (uint64_t)prod[p_] + carry;
                    prod[p_] = (uint32_t)t;
                    carry = t >> 32;
                }
            }
        }
        {
            uint64_t carry = 1ull << 31;
#pragma unroll
            for (int i = 5; i < XW; ++i) {
                const uint64_t t = (uint64_t)prod[i] + carry;
                prod[i] = (uint32_t)t;
                carry = t >> 32;
            }
        }
        constexpr int DW = PC_SI_SHIFT / 32;
        nq = nw + tab->nr - DW;
        if (nq < 1) nq = 1;
#pragma unroll
        for (int i = 0; i < XW; ++i) x[i] = i < nq && i + DW < XW ? prod[i + DW] : 0;
    }
    W* out = (W*)A.out + c * A.ct_stride + part * A.part_stride;
    for (int o = 0; o < tab->nout; ++o) {
        const int row = tab->orow[o];
        const PrimeConst& q = A.pr[row];
        // sum_w x_w (2^32)^w: terms < 2^32 * 2^28, at most 16 of them -> < 2^64
        uint64_t acc = 0;
#pragma unroll
        for (int w = 0; w < XW; ++w)
            if (w < nq) acc += (uint64_t)x[w] * tab->pow32[o][w];
        W r = MM::reduce((WideOf<W>)acc, q);
        if (neg && r) r = (W)q.p - r;
        out[(size_t)row * N + idx] = r;
    }
}

// See args.cuh SmallInt1Args.
template <class W, class MM>
__global__ void __launch_bounds__(BX) small_int1(SmallInt1Args A) {
    const int part = blockIdx.y, c = blockIdx.z;
    const int idx = blockIdx.x * BX + threadIdx.x;
    W* x = (W*)A.x + c * A.ct_stride + part * A.part_stride;
    const uint64_t q0 = A.pr[0].p;
    const uint64_t v = (uint64_t)x[idx];
    const bool neg = v > (q0 >> 1);
    const uint64_t mag = neg ? q0 - v : v;
    unsigned __int128 prod = (unsigned __int128)mag * A.mc;
    prod += (unsigned __int128)1 << (A.sh - 1);
    const uint64_t q = (uint64_t)(prod >> A.sh);
    for (int t = 0; t < A.nrows; ++t) {
        const uint64_t p = A.pr[t].p;
        const uint64_t r = q % p;
        x[(size_t)t * N + idx] = (W)((neg && r) ? p - r : r);
    }
}

template <class W, class MM>
int small_int1_l(const SmallInt1Args* a, cudaStream_t s) {
    if (a->sh < 1 || a->sh > 127 || a->nrows < 1 || a->nrows > 64 || a->K < 1 || a->nparts < 1 || a->nparts > 2) return -1;
    small_int1<W, MM><<<dim3(N / BX, a->nparts, a->K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}

template <class W, class MM>
int mod_raise_l(const ModRaiseArgs* a, cudaStream_t s) {
    if (a->d < 1 || a->d > 4 || a->r0 < a->d || a->r1 > 64 || a->r1 <= a->r0 || a->nparts < 1 || a->nparts > 2) return -1;
    mod_raise<W, MM><<<dim3(N / BX, a->r1 - a->r0, a->K * a->nparts), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}

template <class W, class MM>
int small_int_l(const SmallIntArgs* a, cudaStream_t s) {
    if (sizeof(W) != 4) return -2;
    if (a->nparts < 1 || a->nparts > 2 || a->K < 1) return -1;
    small_int<W, MM><<<dim3(N / BX, a->nparts, a->K), BX, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
}  // namespace bts
}  // namespace pc
