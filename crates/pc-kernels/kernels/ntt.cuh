// NTT (register passes) generic over the word W: N = 2^16 as 256 x 256.
//  column pass: 256 threads = 8 warps x 32 columns; a thread holds 32 rows of its column, 5 butterfly stages in
//               registers, one shared-memory transpose (two halves for u64), 3 more stages in registers.
//  row pass:    one warp per 256-element row, 8 elements per lane: 3 stages in registers, a warp-local transpose,
//               2 stages across lane pairs (shuffles), 3 stages in registers (no block barrier).
// Only Shoup products (twiddles are constants), so the kernels do not depend on the ModMul strategy. Out-of-place first
// pass (`soff`) and the forward epilogues (`epi`): args.cuh NttArgs.
#pragma once
#include "arith.cuh"

namespace pc {
namespace ntt {
template <class W>
constexpr int PH = sizeof(W) == 8 ? 2 : 1;  // column-pass transpose phases (shared memory <= 32 KB)

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
template <class W>
__device__ __forceinline__ W mul_uv(W x, const W* w, const W* ws, int ui, int vi, W p) {
    return mul_shoup<W>(mul_shoup<W>(x, __ldg(w + ui), __ldg(ws + ui), p), __ldg(w + vi), __ldg(ws + vi), p);
}

// NttArgs.epi on 4 consecutive transform values v of word r (prime p, scalar s with Shoup companion ss)
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

template <class W>
__device__ __forceinline__ void epi4(const NttArgs& T, size_t r, const W* v, W s, W ss, W p) {
    W* o = (W*)T.eout + r;
    const V4<W> x = load4_sum<W>((const W*)T.ex + r, T.xnum, T.xstride, p);
    V4<W> d{};
    if (T.epi == 2) d = load4<W>(o);
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

// ---------------------------------------------------------------- forward column pass
template <class W>
__global__ void __launch_bounds__(256) ntt_fwd_cols(NttArgs T) {
    __shared__ W s[256 * 32 / PH<W>];
    const int row = blockIdx.y;
    W* a = (W*)T.base + (size_t)T.roff[row] * N + blockIdx.x * 32;
    const W* in = T.soff ? (const W*)T.src + (size_t)T.soff[row] * N + blockIdx.x * 32 : a;
    const int pid = T.rprime[row];
    const W p = (W)T.pr[pid].p;
    const W* w = (const W*)T.fw + (size_t)pid * N;
    const W* ws = (const W*)T.fws + (size_t)pid * N;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    W x[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) x[j] = in[(wp + 8 * j) * 256 + lane];
    for (int m = 1; m < T.snum; ++m) {
#pragma unroll
        for (int j = 0; j < 32; ++j) x[j] = add<W>(x[j], in[m * T.sstride + (wp + 8 * j) * 256 + lane], p);
    }
    // stages tr = 128..8 (m = 1..16): pairs (j, j + tr/8), group i = j / (2 tr/8)
#pragma unroll
    for (int st = 0; st < 5; ++st) {
        const int h = 16 >> st;  // tr / 8
        const int m = 1 << st;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            if ((j & h) == 0) {
                const int i = j / (2 * h);
                ct_bf<W>(x[j], x[j + h], __ldg(w + m + i), __ldg(ws + m + i), p);
            }
        }
    }
    // now groups g = wp + 8 q (q < 4) of 8 consecutive rows: x[8 q + u] = row 8 g + u. With PH = 2 (u64) the
    // exchange runs in two halves of 128 rows (32 KB of shared memory each).
#pragma unroll
    for (int ph = 0; ph < PH<W>; ++ph) {
        if (ph) __syncthreads();
#pragma unroll
        for (int j = ph * 32 / PH<W>; j < (ph + 1) * 32 / PH<W>; ++j) s[(wp + 8 * j - ph * 256 / PH<W>) * 32 + lane] = x[j];
        __syncthreads();
#pragma unroll
        for (int q = ph * 4 / PH<W>; q < (ph + 1) * 4 / PH<W>; ++q)
#pragma unroll
            for (int u = 0; u < 8; ++u) x[8 * q + u] = s[(8 * (wp + 8 * q) + u - ph * 256 / PH<W>) * 32 + lane];
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int g = wp + 8 * q;
        W* y = x + 8 * q;
        {  // tr = 4, m = 32, i = g
            const W tw = __ldg(w + 32 + g), tws = __ldg(ws + 32 + g);
#pragma unroll
            for (int u = 0; u < 4; ++u) ct_bf<W>(y[u], y[u + 4], tw, tws, p);
        }
        {  // tr = 2, m = 64, i = 2g + u/4
#pragma unroll
            for (int hb = 0; hb < 2; ++hb) {
                const W tw = __ldg(w + 64 + 2 * g + hb), tws = __ldg(ws + 64 + 2 * g + hb);
                ct_bf<W>(y[4 * hb], y[4 * hb + 2], tw, tws, p);
                ct_bf<W>(y[4 * hb + 1], y[4 * hb + 3], tw, tws, p);
            }
        }
        {  // tr = 1, m = 128, i = 4g + u/2
#pragma unroll
            for (int qb = 0; qb < 4; ++qb) {
                const W tw = __ldg(w + 128 + 4 * g + qb), tws = __ldg(ws + 128 + 4 * g + qb);
                ct_bf<W>(y[2 * qb], y[2 * qb + 1], tw, tws, p);
            }
        }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int u = 0; u < 8; ++u) a[(8 * (wp + 8 * q) + u) * 256 + lane] = x[8 * q + u];
}

// ---------------------------------------------------------------- forward row pass (warp per row, 8 rows per block)
template <class W>
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
        for (int j = 0; j < 4; ++j) ct_bf<W>(x[j], x[j + 4], tw, tws, p);
    }
    {  // t = 64, m = 512: i = j / 4 (j<4 -> 0); pairs (j, j+2) for j in {0,1,4,5}
#pragma unroll
        for (int hb = 0; hb < 2; ++hb) {
            const int ti = 512 + 2 * R + hb;
            const W tw = __ldg(w + ti), tws = __ldg(ws + ti);
            ct_bf<W>(x[4 * hb], x[4 * hb + 2], tw, tws, p);
            ct_bf<W>(x[4 * hb + 1], x[4 * hb + 3], tw, tws, p);
        }
    }
    {  // t = 32, m = 1024: i = j / 2; pairs (j, j+1) for even j
#pragma unroll
        for (int qb = 0; qb < 4; ++qb) {
            const int ti = 1024 + 4 * R + qb;
            const W tw = __ldg(w + ti), tws = __ldg(ws + ti);
            ct_bf<W>(x[2 * qb], x[2 * qb + 1], tw, tws, p);
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
            const W U = lo ? x[u] : o, V = lo ? o : x[u];
            const W v = mul_shoup<W>(V, tw, tws, p);
            x[u] = lo ? add<W>(U, v, p) : sub<W>(U, v, p);
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
            const W U = lo ? x[u] : o, V = lo ? o : x[u];
            const W v = mul_shoup<W>(V, tw, tws, p);
            x[u] = lo ? add<W>(U, v, p) : sub<W>(U, v, p);
        }
    }
    // CT butterfly with the factored twiddle U V
    auto ct_uv = [&](W& a, W& b, int ui, int vi) {
        const W v = mul_uv<W>(b, w, ws, ui, vi, p);
        const W u = a;
        a = add<W>(u, v, p);
        b = sub<W>(u, v, p);
    };
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
    // store: layout B is 8 consecutive elements per lane -> vectorized
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
template <class W>
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
        a = add<W>(u, v, p);
        b = mul_uv<W>(sub<W>(u, v, p), w, ws, ui, vi, p);
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
            x[u] = lo ? add<W>(U, V, p) : mul_shoup<W>(sub<W>(U, V, p), tw, tws, p);
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
            x[u] = lo ? add<W>(U, V, p) : mul_shoup<W>(sub<W>(U, V, p), tw, tws, p);
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
            gs_bf<W>(x[2 * qb], x[2 * qb + 1], __ldg(w + ti), __ldg(ws + ti), p);
        }
    }
    {  // t = 64, m = 512
#pragma unroll
        for (int hb = 0; hb < 2; ++hb) {
            const int ti = 512 + 2 * R + hb;
            const W tw = __ldg(w + ti), tws = __ldg(ws + ti);
            gs_bf<W>(x[4 * hb], x[4 * hb + 2], tw, tws, p);
            gs_bf<W>(x[4 * hb + 1], x[4 * hb + 3], tw, tws, p);
        }
    }
    {  // t = 128, m = 256
        const W tw = __ldg(w + 256 + R), tws = __ldg(ws + 256 + R);
#pragma unroll
        for (int j = 0; j < 4; ++j) gs_bf<W>(x[j], x[j + 4], tw, tws, p);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) a[lane + 32 * j] = x[j];
}

// ---------------------------------------------------------------- inverse column pass (GS tr = 1..128), N^-1
template <class W>
__global__ void __launch_bounds__(256) ntt_inv_cols(NttArgs T) {
    __shared__ W s[256 * 32 / PH<W>];
    const int row = blockIdx.y;
    W* a = (W*)T.base + (size_t)T.roff[row] * N + blockIdx.x * 32;
    const int pid = T.rprime[row];
    const W p = (W)T.pr[pid].p;
    const W* w = (const W*)T.iw + (size_t)pid * N;
    const W* ws = (const W*)T.iws + (size_t)pid * N;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    W x[32];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int u = 0; u < 8; ++u) x[8 * q + u] = a[(8 * (wp + 8 * q) + u) * 256 + lane];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int g = wp + 8 * q;
        W* y = x + 8 * q;
        {  // tr = 1, m = 128
#pragma unroll
            for (int qb = 0; qb < 4; ++qb)
                gs_bf<W>(y[2 * qb], y[2 * qb + 1], __ldg(w + 128 + 4 * g + qb), __ldg(ws + 128 + 4 * g + qb), p);
        }
        {  // tr = 2, m = 64
#pragma unroll
            for (int hb = 0; hb < 2; ++hb) {
                const W tw = __ldg(w + 64 + 2 * g + hb), tws = __ldg(ws + 64 + 2 * g + hb);
                gs_bf<W>(y[4 * hb], y[4 * hb + 2], tw, tws, p);
                gs_bf<W>(y[4 * hb + 1], y[4 * hb + 3], tw, tws, p);
            }
        }
        {  // tr = 4, m = 32
            const W tw = __ldg(w + 32 + g), tws = __ldg(ws + 32 + g);
#pragma unroll
            for (int u = 0; u < 4; ++u) gs_bf<W>(y[u], y[u + 4], tw, tws, p);
        }
    }
#pragma unroll
    for (int ph = 0; ph < PH<W>; ++ph) {
        if (ph) __syncthreads();
#pragma unroll
        for (int q = ph * 4 / PH<W>; q < (ph + 1) * 4 / PH<W>; ++q)
#pragma unroll
            for (int u = 0; u < 8; ++u) s[(8 * (wp + 8 * q) + u - ph * 256 / PH<W>) * 32 + lane] = x[8 * q + u];
        __syncthreads();
#pragma unroll
        for (int j = ph * 32 / PH<W>; j < (ph + 1) * 32 / PH<W>; ++j) x[j] = s[(wp + 8 * j - ph * 256 / PH<W>) * 32 + lane];
    }
    // stages tr = 8..128 (m = 16..1): pairs (j, j + tr/8)
#pragma unroll
    for (int st = 4; st >= 0; --st) {
        const int h = 16 >> st;
        const int m = 1 << st;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            if ((j & h) == 0) {
                const int i = j / (2 * h);
                gs_bf<W>(x[j], x[j + h], __ldg(w + m + i), __ldg(ws + m + i), p);
            }
        }
    }
    const W ni = ((const W*)T.ninv)[pid], nis = ((const W*)T.ninvs)[pid];
#pragma unroll
    for (int j = 0; j < 32; ++j) a[(wp + 8 * j) * 256 + lane] = mul_shoup<W>(x[j], ni, nis, p);
}

template <class W>
int fwd(const NttArgs* a, cudaStream_t s) {
    if (a->epi < 0 || a->epi > 3 || (a->epi && (!a->eout || !a->ex || !a->es || !a->ess))) return -1;
    NttArgs b = *a;
    b.snum = a->soff ? max(a->snum, 1) : 1;
    b.xnum = max(a->xnum, 1);
    const int ch = a->chunk > 0 ? a->chunk : a->nrows;
    for (int i = 0; i < a->nrows; i += ch) {
        const int m = min(ch, a->nrows - i);
        b.roff = a->roff + i;
        b.rprime = a->rprime + i;
        b.soff = a->soff ? a->soff + i : nullptr;
        ntt_fwd_cols<W><<<dim3(8, m), 256, 0, s>>>(b);
        ntt_fwd_rows<W><<<dim3(32, m), 256, 0, s>>>(b);
    }
    return (int)cudaGetLastError();
}
template <class W>
int inv(const NttArgs* a, cudaStream_t s) {
    if (a->epi) return -1;  // the epilogues are forward-only
    NttArgs b = *a;
    b.snum = a->soff ? max(a->snum, 1) : 1;
    const int ch = a->chunk > 0 ? a->chunk : a->nrows;
    for (int i = 0; i < a->nrows; i += ch) {
        const int m = min(ch, a->nrows - i);
        b.roff = a->roff + i;
        b.rprime = a->rprime + i;
        b.soff = a->soff ? a->soff + i : nullptr;
        ntt_inv_rows<W><<<dim3(32, m), 256, 0, s>>>(b);
        ntt_inv_cols<W><<<dim3(8, m), 256, 0, s>>>(b);
    }
    return (int)cudaGetLastError();
}
}  // namespace ntt
}  // namespace pc
