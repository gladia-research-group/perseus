// NTT (register passes) generic over the word W: N = 2^16 as 256 x 256.
//  column pass: 256 threads = 16 columns x 16 threads; a thread holds 16 rows of its column, 4 butterfly stages in
//               registers, one shared-memory transpose, 4 more stages in registers. Chunks of at most COLS8_MAX rows,
//               whose passes are latency-bound, run on 8-column blocks: 32 threads per column, 8 rows each, 3 + 3 + 2
//               stages around two transposes (twice the blocks at half the work per thread).
//  row pass:    one warp per 256-element row, 8 elements per lane: 3 stages in registers, a warp-local transpose,
//               2 stages across lane pairs (shuffles), 3 stages in registers (no block barrier).
// Only Shoup products (twiddles are constants), so the kernels do not depend on the ModMul strategy. NttArgs.lazy (u32,
// every prime < 2^30): lazy (Harvey) butterflies, forward words in [0, 4p) and inverse words in [0, 2p) between stages;
// either way a transform returns canonical words. Out-of-place first pass (`soff`) and the forward epilogues (`epi`): NttArgs.
#pragma once
#include "arith.cuh"

namespace pc {
namespace ntt {
template <class W>
constexpr int PH = sizeof(W) == 8 ? 2 : 1;  // transpose phases of a 32-column pass (coeffrows.cuh; shared memory <= 32 KB)

template <class W>
__device__ __forceinline__ void ct_bf(W& a, W& b, W w, W ws, W p) {
    const W v = mul_shoup<W>(b, w, ws, p);
    const W u = a;
    a = add<W>(u, v, p);
    b = sub<W>(u, v, p);
}
template <class W>
__device__ __forceinline__ void gs_bf(W& a, W& b, W w, W ws, W p) {
    const W u = a, v = b;
    a = add<W>(u, v, p);
    b = mul_shoup<W>(sub<W>(u, v, p), w, ws, p);
}

// Butterflies of the 2^16 transforms, lazy when L at u32 (other words keep the reduced ones): the Shoup product without
// its last correction is in [0, 2p) for any operand; ct_lz takes and gives [0, 4p), gs_lz [0, 2p).
template <class W, bool L>
constexpr bool LAZY = L && sizeof(W) == 4;
template <class W>
__device__ __forceinline__ W shoup_lz(W a, W w, W ws, W p) {
    return (W)(wlo(a, w) - wlo(Word<W>::mulhi(a, ws), p));
}
template <class W>
__device__ __forceinline__ W red2p(W a, W p) {
    return a >= (W)(2 * p) ? (W)(a - 2 * p) : a;
}
// a word of [0, 4p) (a forward lazy one) as its residue in [0, p)
template <class W, bool L>
__device__ __forceinline__ W canon(W a, W p) {
    if constexpr (LAZY<W, L>) {
        a = red2p<W>(a, p);
        return a >= p ? (W)(a - p) : a;
    }
    return a;
}
// CT with the second operand's product t already taken (t in [0, 2p) when lazy, [0, p) otherwise)
template <class W, bool L>
__device__ __forceinline__ void ct_t(W& a, W& b, W t, W p) {
    if constexpr (LAZY<W, L>) {
        const W u = red2p<W>(a, p);
        a = u + t;
        b = (W)(u - t + 2 * p);
    } else {
        const W u = a;
        a = add<W>(u, t, p);
        b = sub<W>(u, t, p);
    }
}
template <class W, bool L>
__device__ __forceinline__ W mul_lz(W b, W w, W ws, W p) {
    if constexpr (LAZY<W, L>) return shoup_lz<W>(b, w, ws, p);
    return mul_shoup<W>(b, w, ws, p);
}
template <class W, bool L>
__device__ __forceinline__ void ct_lz(W& a, W& b, W w, W ws, W p) {
    ct_t<W, L>(a, b, mul_lz<W, L>(b, w, ws, p), p);
}
// GS: (a + b, (a - b) w)
template <class W, bool L>
__device__ __forceinline__ W gs_sum(W a, W b, W p) {
    if constexpr (LAZY<W, L>) return red2p<W>(a + b, p);
    return add<W>(a, b, p);
}
template <class W, bool L>
__device__ __forceinline__ W gs_dif(W a, W b, W p) {
    if constexpr (LAZY<W, L>) return (W)(a - b + 2 * p);
    return sub<W>(a, b, p);
}
template <class W, bool L>
__device__ __forceinline__ void gs_lz(W& a, W& b, W w, W ws, W p) {
    const W u = a, v = b;
    a = gs_sum<W, L>(u, v, p);
    b = mul_lz<W, L>(gs_dif<W, L>(u, v, p), w, ws, p);
}

// Row-pass twiddles of the last three stages (m = 2^s, s = 13..15) from two factors already in the table: for
// i = R 2^(s-8) + c, psi^brev16(m + i) = U * V with U = w[m + (R << (s-8))] (uniform per warp) and V = psi^(2^(24-s)
// brev_(s-8)(c)), the table entry vidx(s, c) < 128. The two Shoup products give the same residue as one product by
// the full twiddle, while the row pass reads ~1/8 of the twiddle bytes.
__device__ __forceinline__ int vidx(int s, uint32_t c) {
    const uint32_t r = __brev(c) >> (32 - (s - 8));  // brev_(s-8)(c)
    if (r == 0) return 0;                             // psi^0 = 1 (w[0])
    const int j = __ffs(r) - 1;
    const uint32_t o = r >> j;                        // odd: exponent 2^(24-s+j) o = 2^(15-s') (1 + 2 brev_s'(i'))
    const int sp = s - 9 - j;
    return (1 << sp) + (sp ? (int)(__brev((o - 1) >> 1) >> (32 - sp)) : 0);
}
template <class W, bool L>
__device__ __forceinline__ W mul_uv(W x, const W* w, const W* ws, int ui, int vi, W p) {
    return mul_lz<W, L>(mul_lz<W, L>(x, __ldg(w + ui), __ldg(ws + ui), p), __ldg(w + vi), __ldg(ws + vi), p);
}

// 4 consecutive words of the sum mod p of n rows `stride` words apart (n >= 1)
template <class W>
__device__ __forceinline__ V4<W> load4_sum(const W* a, int n, uint64_t stride, W p) {
    V4<W> x = load4<W>(a);
    for (int j = 1; j < n; ++j) {
        const V4<W> y = load4<W>(a + j * stride);
#pragma unroll
        for (int u = 0; u < 4; ++u) x.v[u] = add<W>(x.v[u], y.v[u], p);
    }
    return x;
}

// NttArgs.epi on 4 consecutive transform values v of word r (prime p, scalar s with Shoup companion ss)
template <class W>
__device__ __forceinline__ void epi4(const NttArgs& T, size_t r, const W* v, W s, W ss, W p) {
    W* o = (W*)T.eout + r;
    const V4<W> x = load4_sum<W>((const W*)T.ex + r, T.xnum, T.xstride, p);
    V4<W> d{};
    if (T.epi == 2) d = load4<W>(T.ed ? (const W*)T.ed + r : o);
    W y[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        if (T.epi == 3) {
            y[u] = add<W>(mul_shoup<W>(x.v[u], s, ss, p), v[u], p);
        } else {
            y[u] = mul_shoup<W>(sub<W>(x.v[u], v[u], p), s, ss, p);
            if (T.epi == 2) y[u] = add<W>(d.v[u], y[u], p);
        }
    }
    store4<W>(o, y[0], y[1], y[2], y[3]);
}

// ---------------------------------------------------------------- forward column passes
// The column pass of one 16-column block of a row (prime p, twiddles w / ws) on x = the block's words held as
// x[j] = word (q + 16 j) * 256 + c (canonical), stored at a (the block's first word); s: the transpose buffer.
template <class W, bool L>
__device__ __forceinline__ void fwd_cols_core(W (&x)[16], W* a, W* s, const W* w, const W* ws, W p, int c, int q) {
    // stages tr = 128..16 (m = 1..8): pairs (j, j + tr/16), group i = j / (2 tr/16)
#pragma unroll
    for (int st = 0; st < 4; ++st) {
        const int h = 8 >> st;  // tr / 16
        const int m = 1 << st;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if ((j & h) == 0) {
                const int i = j / (2 * h);
                ct_lz<W, L>(x[j], x[j + h], __ldg(w + m + i), __ldg(ws + m + i), p);
            }
        }
    }
    // now the 16 consecutive rows of group q: x[u] = row 16 q + u
#pragma unroll
    for (int j = 0; j < 16; ++j) s[(q + 16 * j) * 16 + c] = x[j];
    __syncthreads();
#pragma unroll
    for (int u = 0; u < 16; ++u) x[u] = s[(16 * q + u) * 16 + c];
    // stages tr = 8..1 (m = 16..128): pairs (u, u + tr), group i = (16 q + u) / (2 tr)
#pragma unroll
    for (int st = 0; st < 4; ++st) {
        const int h = 8 >> st;  // tr
        const int m = 16 << st;
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            if ((u & h) == 0) {
                const int i = (m >> 4) * q + u / (2 * h);
                ct_lz<W, L>(x[u], x[u + h], __ldg(w + m + i), __ldg(ws + m + i), p);
            }
        }
    }
#pragma unroll
    for (int u = 0; u < 16; ++u) a[(16 * q + u) * 256 + c] = x[u];
}

template <class W, bool L>
__global__ void __launch_bounds__(256) ntt_fwd_cols(NttArgs T) {
    __shared__ W s[256 * 16];
    const int row = blockIdx.y;
    W* a = (W*)T.base + (size_t)T.roff[row] * N + blockIdx.x * 16;
    const W* in = T.soff ? (const W*)T.src + (size_t)T.soff[row] * N + blockIdx.x * 16 : a;
    const int pid = T.rprime[row];
    const W p = (W)T.pr[pid].p;
    const int c = threadIdx.x & 15, q = threadIdx.x >> 4;
    W x[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) x[j] = in[(q + 16 * j) * 256 + c];
    for (int m = 1; m < T.snum; ++m) {
#pragma unroll
        for (int j = 0; j < 16; ++j) x[j] = add<W>(x[j], in[m * T.sstride + (q + 16 * j) * 256 + c], p);
    }
    fwd_cols_core<W, L>(x, a, s, (const W*)T.fw + (size_t)pid * N, (const W*)T.fws + (size_t)pid * N, p, c, q);
}

// The column pass of a short chunk on 8-column blocks, x[j] in three layouts: rows q + 32 j (tr = 128..32), rows
// lo + 4 j + 32 hi with q = 4 hi + lo (tr = 16..4), rows 8 q + u (tr = 2, 1).
template <class W, bool L>
__global__ void __launch_bounds__(256) ntt_fwd_cols8(NttArgs T) {
    __shared__ W s[256 * 8];
    const int row = blockIdx.y;
    W* a = (W*)T.base + (size_t)T.roff[row] * N + blockIdx.x * 8;
    const W* in = T.soff ? (const W*)T.src + (size_t)T.soff[row] * N + blockIdx.x * 8 : a;
    const int pid = T.rprime[row];
    const W p = (W)T.pr[pid].p;
    const W* w = (const W*)T.fw + (size_t)pid * N;
    const W* ws = (const W*)T.fws + (size_t)pid * N;
    const int c = threadIdx.x & 7, q = threadIdx.x >> 3, lo = q & 3, hi = q >> 2;
    W x[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = in[(q + 32 * j) * 256 + c];
    for (int m = 1; m < T.snum; ++m) {
#pragma unroll
        for (int j = 0; j < 8; ++j) x[j] = add<W>(x[j], in[m * T.sstride + (q + 32 * j) * 256 + c], p);
    }
#pragma unroll
    for (int st = 0; st < 3; ++st) {  // m = 1, 2, 4
        const int h = 4 >> st, m = 1 << st;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if ((j & h) == 0) ct_lz<W, L>(x[j], x[j + h], __ldg(w + m + j / (2 * h)), __ldg(ws + m + j / (2 * h)), p);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) s[(q + 32 * j) * 8 + c] = x[j];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = s[(lo + 4 * j + 32 * hi) * 8 + c];
#pragma unroll
    for (int st = 0; st < 3; ++st) {  // m = 8, 16, 32: group (lo + 4 j + 32 hi) / (2 tr)
        const int h = 4 >> st, m = 8 << st;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if ((j & h) == 0) {
                const int i = (m >> 3) * hi + j / (2 * h);
                ct_lz<W, L>(x[j], x[j + h], __ldg(w + m + i), __ldg(ws + m + i), p);
            }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) s[(lo + 4 * j + 32 * hi) * 8 + c] = x[j];
    __syncthreads();
#pragma unroll
    for (int u = 0; u < 8; ++u) x[u] = s[(8 * q + u) * 8 + c];
#pragma unroll
    for (int st = 0; st < 2; ++st) {  // m = 64, 128: group (8 q + u) / (2 tr)
        const int h = 2 >> st, m = 64 << st;
#pragma unroll
        for (int u = 0; u < 8; ++u)
            if ((u & h) == 0) {
                const int i = (m >> 5) * q + u / (2 * h);
                ct_lz<W, L>(x[u], x[u + h], __ldg(w + m + i), __ldg(ws + m + i), p);
            }
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) a[(8 * q + u) * 256 + c] = x[u];
}

// ---------------------------------------------------------------- forward row pass (warp per row, 8 rows per block)
template <class W, bool L>
__global__ void __launch_bounds__(256) ntt_fwd_rows(NttArgs T) {
    __shared__ W s[8][256 + 8];
    const int row = blockIdx.y;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int R = blockIdx.x * 8 + wp;  // 256-element row inside the limb
    W* a = (W*)T.base + (size_t)T.roff[row] * N + R * 256;
    const int pid = T.rprime[row];
    const W p = (W)T.pr[pid].p;
    const W* w = (const W*)T.fw + (size_t)pid * N;
    const W* ws = (const W*)T.fws + (size_t)pid * N;
    W x[8];
    // layout A: x[j] = e = lane + 32 j
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = a[lane + 32 * j];
    {  // t = 128, m = 256: i = 0 -> index 256 + R
        const W tw = __ldg(w + 256 + R), tws = __ldg(ws + 256 + R);
#pragma unroll
        for (int j = 0; j < 4; ++j) ct_lz<W, L>(x[j], x[j + 4], tw, tws, p);
    }
    {  // t = 64, m = 512: i = j / 4 (j<4 -> 0); pairs (j, j+2) for j in {0,1,4,5}
#pragma unroll
        for (int hb = 0; hb < 2; ++hb) {
            const int ti = 512 + 2 * R + hb;
            const W tw = __ldg(w + ti), tws = __ldg(ws + ti);
            ct_lz<W, L>(x[4 * hb], x[4 * hb + 2], tw, tws, p);
            ct_lz<W, L>(x[4 * hb + 1], x[4 * hb + 3], tw, tws, p);
        }
    }
    {  // t = 32, m = 1024: i = j / 2; pairs (j, j+1) for even j
#pragma unroll
        for (int qb = 0; qb < 4; ++qb) {
            const int ti = 1024 + 4 * R + qb;
            const W tw = __ldg(w + ti), tws = __ldg(ws + ti);
            ct_lz<W, L>(x[2 * qb], x[2 * qb + 1], tw, tws, p);
        }
    }
    // warp-local transpose to layout B: x[u] = e = 8 lane + u
#pragma unroll
    for (int j = 0; j < 8; ++j) s[wp][lane + 32 * j + ((lane + 32 * j) >> 5)] = x[j];
    __syncwarp();
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int e = 8 * lane + u;
        x[u] = s[wp][e + (e >> 5)];
    }
    // t = 16, m = 2048: pair (e, e+16) = lanes (l, l^2); i_local = e / 32 = lane / 4
    {
        const int ti = 2048 + 8 * R + (lane >> 2);
        const W tw = __ldg(w + ti), tws = __ldg(ws + ti);
        const bool lo = (lane & 2) == 0;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const W o = shfl_xor<W>(0xffffffffu, x[u], 2);
            // lo lane holds U (e), hi lane holds V (e+16)
            W U = lo ? x[u] : o, V = lo ? o : x[u];
            ct_t<W, L>(U, V, mul_lz<W, L>(V, tw, tws, p), p);
            x[u] = lo ? U : V;
        }
    }
    // t = 8, m = 4096: pair (e, e+8) = lanes (l, l^1); i_local = e / 16 = lane / 2
    {
        const int ti = 4096 + 16 * R + (lane >> 1);
        const W tw = __ldg(w + ti), tws = __ldg(ws + ti);
        const bool lo = (lane & 1) == 0;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const W o = shfl_xor<W>(0xffffffffu, x[u], 1);
            W U = lo ? x[u] : o, V = lo ? o : x[u];
            ct_t<W, L>(U, V, mul_lz<W, L>(V, tw, tws, p), p);
            x[u] = lo ? U : V;
        }
    }
    // CT butterfly with the factored twiddle U V
    auto ct_uv = [&](W& a, W& b, int ui, int vi) { ct_t<W, L>(a, b, mul_uv<W, L>(b, w, ws, ui, vi, p), p); };
    {  // t = 4, m = 8192: i_local = e / 8 = lane
        const int ui = 8192 + 32 * R, vi = vidx(13, lane);
#pragma unroll
        for (int u = 0; u < 4; ++u) ct_uv(x[u], x[u + 4], ui, vi);
    }
    {  // t = 2, m = 16384: i_local = e / 4 = 2 lane + u/4
#pragma unroll
        for (int hb = 0; hb < 2; ++hb) {
            const int ui = 16384 + 64 * R, vi = vidx(14, 2 * lane + hb);
            ct_uv(x[4 * hb], x[4 * hb + 2], ui, vi);
            ct_uv(x[4 * hb + 1], x[4 * hb + 3], ui, vi);
        }
    }
    {  // t = 1, m = 32768: i_local = e / 2 = 4 lane + u/2
#pragma unroll
        for (int qb = 0; qb < 4; ++qb) ct_uv(x[2 * qb], x[2 * qb + 1], 32768 + 128 * R, vidx(15, 4 * lane + qb));
    }
    // store: layout B is 8 consecutive elements per lane -> vectorized, canonical words
#pragma unroll
    for (int u = 0; u < 8; ++u) x[u] = canon<W, L>(x[u], p);
    if (T.epi) {
        const size_t r = (size_t)T.roff[row] * N + R * 256 + 8 * lane;
        const W s = ((const W*)T.es)[pid], ss = ((const W*)T.ess)[pid];
        epi4<W>(T, r, x, s, ss, p);
        epi4<W>(T, r + 4, x + 4, s, ss, p);
        return;
    }
    store4<W>(a + 8 * lane, x[0], x[1], x[2], x[3]);
    store4<W>(a + 8 * lane + 4, x[4], x[5], x[6], x[7]);
}

// ---------------------------------------------------------------- inverse row pass (GS, t = 1 .. 128)
template <class W, bool L>
__global__ void __launch_bounds__(256) ntt_inv_rows(NttArgs T) {
    __shared__ W s[8][256 + 8];
    const int row = blockIdx.y;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int R = blockIdx.x * 8 + wp;
    W* a = (W*)T.base + (size_t)T.roff[row] * N + R * 256;
    const W* in = T.soff ? (const W*)T.src + (size_t)T.soff[row] * N + R * 256 : a;
    const int pid = T.rprime[row];
    const W p = (W)T.pr[pid].p;
    const W* w = (const W*)T.iw + (size_t)pid * N;
    const W* ws = (const W*)T.iws + (size_t)pid * N;
    W x[8];
    {
        const V4<W> v0 = load4_sum<W>(in + 8 * lane, T.snum, T.sstride, p), v1 = load4_sum<W>(in + 8 * lane + 4, T.snum, T.sstride, p);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            x[u] = v0.v[u];
            x[4 + u] = v1.v[u];
        }
    }
    // GS butterfly with the factored twiddle U V (inverse table)
    auto gs_uv = [&](W& a, W& b, int ui, int vi) {
        const W u = a, v = b;
        a = gs_sum<W, L>(u, v, p);
        b = mul_uv<W, L>(gs_dif<W, L>(u, v, p), w, ws, ui, vi, p);
    };
    {  // t = 1, m = 32768
#pragma unroll
        for (int qb = 0; qb < 4; ++qb) gs_uv(x[2 * qb], x[2 * qb + 1], 32768 + 128 * R, vidx(15, 4 * lane + qb));
    }
    {  // t = 2, m = 16384
#pragma unroll
        for (int hb = 0; hb < 2; ++hb) {
            const int ui = 16384 + 64 * R, vi = vidx(14, 2 * lane + hb);
            gs_uv(x[4 * hb], x[4 * hb + 2], ui, vi);
            gs_uv(x[4 * hb + 1], x[4 * hb + 3], ui, vi);
        }
    }
    {  // t = 4, m = 8192
        const int ui = 8192 + 32 * R, vi = vidx(13, lane);
#pragma unroll
        for (int u = 0; u < 4; ++u) gs_uv(x[u], x[u + 4], ui, vi);
    }
    {  // t = 8, m = 4096: lanes (l, l^1)
        const int ti = 4096 + 16 * R + (lane >> 1);
        const W tw = __ldg(w + ti), tws = __ldg(ws + ti);
        const bool lo = (lane & 1) == 0;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const W o = shfl_xor<W>(0xffffffffu, x[u], 1);
            const W U = lo ? x[u] : o, V = lo ? o : x[u];
            x[u] = lo ? gs_sum<W, L>(U, V, p) : mul_lz<W, L>(gs_dif<W, L>(U, V, p), tw, tws, p);
        }
    }
    {  // t = 16, m = 2048: lanes (l, l^2)
        const int ti = 2048 + 8 * R + (lane >> 2);
        const W tw = __ldg(w + ti), tws = __ldg(ws + ti);
        const bool lo = (lane & 2) == 0;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const W o = shfl_xor<W>(0xffffffffu, x[u], 2);
            const W U = lo ? x[u] : o, V = lo ? o : x[u];
            x[u] = lo ? gs_sum<W, L>(U, V, p) : mul_lz<W, L>(gs_dif<W, L>(U, V, p), tw, tws, p);
        }
    }
    // transpose B -> A
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int e = 8 * lane + u;
        s[wp][e + (e >> 5)] = x[u];
    }
    __syncwarp();
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = s[wp][lane + 32 * j + ((lane + 32 * j) >> 5)];
    {  // t = 32, m = 1024
#pragma unroll
        for (int qb = 0; qb < 4; ++qb) {
            const int ti = 1024 + 4 * R + qb;
            gs_lz<W, L>(x[2 * qb], x[2 * qb + 1], __ldg(w + ti), __ldg(ws + ti), p);
        }
    }
    {  // t = 64, m = 512
#pragma unroll
        for (int hb = 0; hb < 2; ++hb) {
            const int ti = 512 + 2 * R + hb;
            const W tw = __ldg(w + ti), tws = __ldg(ws + ti);
            gs_lz<W, L>(x[4 * hb], x[4 * hb + 2], tw, tws, p);
            gs_lz<W, L>(x[4 * hb + 1], x[4 * hb + 3], tw, tws, p);
        }
    }
    {  // t = 128, m = 256
        const W tw = __ldg(w + 256 + R), tws = __ldg(ws + 256 + R);
#pragma unroll
        for (int j = 0; j < 4; ++j) gs_lz<W, L>(x[j], x[j + 4], tw, tws, p);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) a[lane + 32 * j] = x[j];
}

// ---------------------------------------------------------------- inverse column passes (GS tr = 1..128), N^-1
// ntt_fwd_cols' layouts in reverse order: rows 16 q + u (tr = 1..8), one transpose, rows q + 16 j (tr = 16..128)
template <class W, bool L>
__global__ void __launch_bounds__(256) ntt_inv_cols(NttArgs T) {
    __shared__ W s[256 * 16];
    const int row = blockIdx.y;
    W* a = (W*)T.base + (size_t)T.roff[row] * N + blockIdx.x * 16;
    const int pid = T.rprime[row];
    const W p = (W)T.pr[pid].p;
    const W* w = (const W*)T.iw + (size_t)pid * N;
    const W* ws = (const W*)T.iws + (size_t)pid * N;
    const int c = threadIdx.x & 15, q = threadIdx.x >> 4;
    W x[16];
#pragma unroll
    for (int u = 0; u < 16; ++u) x[u] = a[(16 * q + u) * 256 + c];
#pragma unroll
    for (int st = 3; st >= 0; --st) {  // m = 128..16
        const int h = 8 >> st, m = 16 << st;
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            if ((u & h) == 0) {
                const int i = (m >> 4) * q + u / (2 * h);
                gs_lz<W, L>(x[u], x[u + h], __ldg(w + m + i), __ldg(ws + m + i), p);
            }
        }
    }
#pragma unroll
    for (int u = 0; u < 16; ++u) s[(16 * q + u) * 16 + c] = x[u];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 16; ++j) x[j] = s[(q + 16 * j) * 16 + c];
#pragma unroll
    for (int st = 3; st >= 0; --st) {  // m = 8..1
        const int h = 8 >> st, m = 1 << st;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if ((j & h) == 0) {
                const int i = j / (2 * h);
                gs_lz<W, L>(x[j], x[j + h], __ldg(w + m + i), __ldg(ws + m + i), p);
            }
        }
    }
    const W ni = ((const W*)T.ninv)[pid], nis = ((const W*)T.ninvs)[pid];
#pragma unroll
    for (int j = 0; j < 16; ++j) a[(q + 16 * j) * 256 + c] = mul_shoup<W>(x[j], ni, nis, p);
}

// ntt_fwd_cols8's layouts in reverse order
template <class W, bool L>
__global__ void __launch_bounds__(256) ntt_inv_cols8(NttArgs T) {
    __shared__ W s[256 * 8];
    const int row = blockIdx.y;
    W* a = (W*)T.base + (size_t)T.roff[row] * N + blockIdx.x * 8;
    const int pid = T.rprime[row];
    const W p = (W)T.pr[pid].p;
    const W* w = (const W*)T.iw + (size_t)pid * N;
    const W* ws = (const W*)T.iws + (size_t)pid * N;
    const int c = threadIdx.x & 7, q = threadIdx.x >> 3, lo = q & 3, hi = q >> 2;
    W x[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) x[u] = a[(8 * q + u) * 256 + c];
#pragma unroll
    for (int st = 1; st >= 0; --st) {  // m = 128, 64
        const int h = 2 >> st, m = 64 << st;
#pragma unroll
        for (int u = 0; u < 8; ++u)
            if ((u & h) == 0) {
                const int i = (m >> 5) * q + u / (2 * h);
                gs_lz<W, L>(x[u], x[u + h], __ldg(w + m + i), __ldg(ws + m + i), p);
            }
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) s[(8 * q + u) * 8 + c] = x[u];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = s[(lo + 4 * j + 32 * hi) * 8 + c];
#pragma unroll
    for (int st = 2; st >= 0; --st) {  // m = 32, 16, 8
        const int h = 4 >> st, m = 8 << st;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if ((j & h) == 0) {
                const int i = (m >> 3) * hi + j / (2 * h);
                gs_lz<W, L>(x[j], x[j + h], __ldg(w + m + i), __ldg(ws + m + i), p);
            }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) s[(lo + 4 * j + 32 * hi) * 8 + c] = x[j];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = s[(q + 32 * j) * 8 + c];
#pragma unroll
    for (int st = 2; st >= 0; --st) {  // m = 4, 2, 1
        const int h = 4 >> st, m = 1 << st;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if ((j & h) == 0) gs_lz<W, L>(x[j], x[j + h], __ldg(w + m + j / (2 * h)), __ldg(ws + m + j / (2 * h)), p);
    }
    const W ni = ((const W*)T.ninv)[pid], nis = ((const W*)T.ninvs)[pid];
#pragma unroll
    for (int j = 0; j < 8; ++j) a[(q + 32 * j) * 256 + c] = mul_shoup<W>(x[j], ni, nis, p);
}

// Chunks of at most this many rows run their column passes on 8-column blocks.
constexpr int COLS8_MAX = 9;

template <class W, bool L>
void fwd_chunk(const NttArgs& b, int m, cudaStream_t s) {
    if (m <= COLS8_MAX) ntt_fwd_cols8<W, L><<<dim3(32, m), 256, 0, s>>>(b);
    else ntt_fwd_cols<W, L><<<dim3(16, m), 256, 0, s>>>(b);
    ntt_fwd_rows<W, L><<<dim3(32, m), 256, 0, s>>>(b);
}
template <class W, bool L>
void inv_chunk(const NttArgs& b, int m, cudaStream_t s) {
    ntt_inv_rows<W, L><<<dim3(32, m), 256, 0, s>>>(b);
    if (m <= COLS8_MAX) ntt_inv_cols8<W, L><<<dim3(32, m), 256, 0, s>>>(b);
    else ntt_inv_cols<W, L><<<dim3(16, m), 256, 0, s>>>(b);
}

template <class W>
int fwd(const NttArgs* a, cudaStream_t s) {
    if (a->epi < 0 || a->epi > 3 || (a->epi && (!a->eout || !a->ex || !a->es || !a->ess))) return -1;
    NttArgs b = *a;
    b.snum = a->soff ? max(a->snum, 1) : 1;
    b.xnum = max(a->xnum, 1);
    const bool lz = LAZY<W, true> && a->lazy;
    const int ch = a->chunk > 0 ? a->chunk : a->nrows;
    for (int i = 0; i < a->nrows; i += ch) {
        const int m = min(ch, a->nrows - i);
        b.roff = a->roff + i;
        b.rprime = a->rprime + i;
        b.soff = a->soff ? a->soff + i : nullptr;
        if (lz) fwd_chunk<W, true>(b, m, s);
        else fwd_chunk<W, false>(b, m, s);
    }
    return (int)cudaGetLastError();
}
template <class W>
int inv(const NttArgs* a, cudaStream_t s) {
    if (a->epi) return -1;  // the epilogues are forward-only
    NttArgs b = *a;
    b.snum = a->soff ? max(a->snum, 1) : 1;
    const bool lz = LAZY<W, true> && a->lazy;
    const int ch = a->chunk > 0 ? a->chunk : a->nrows;
    for (int i = 0; i < a->nrows; i += ch) {
        const int m = min(ch, a->nrows - i);
        b.roff = a->roff + i;
        b.rprime = a->rprime + i;
        b.soff = a->soff ? a->soff + i : nullptr;
        if (lz) inv_chunk<W, true>(b, m, s);
        else inv_chunk<W, false>(b, m, s);
    }
    return (int)cudaGetLastError();
}
}  // namespace ntt
}  // namespace pc
