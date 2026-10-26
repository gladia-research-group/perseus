// Streamed-weight inner sums from coefficient rows (FIDESlib CoeffRows.cu multCoeffRowsSum):
//   out (+)= sum_r ct_r * pt_r,  pt_r[k] = round(rows_r[k] * scale) mod p_l, in the NTT domain,
// for n <= PC_COEFF_MAXT terms whose ciphertexts share one level (limbs 0..L-1). A plaintext never exists as an
// object: its float32 coefficient row (the weight as streamed from the host) is expanded on the fly.
//   cols_f32: per (term, limb) the NTT column pass with the expansion on its load (scale, round half to even like
//             FIDESlib's llrint, |m| reduced exactly, sign applied) -> scratch [term][limb][N];
//   rows_mac: per limb the NTT row pass of every term's scratch row, multiplied by the term's ciphertext parts and
//             summed in registers; one write of the output (or an add onto it: `accumulate`, chunks of a long sum).
// The column pass (on 32-column blocks) and the row pass follow ntt.cuh's with another load / store; they are repeated
// here rather than templated there so the NTT kernels stay untouched. Semantics in args.cuh CoeffArgs.
#pragma once
#include "arith.cuh"
#include "ntt.cuh"

namespace pc {
namespace coeffrows {

// round(f * scale) mod p for a float32 coefficient (clamped to +-2^62 past the int64 range; counted)
template <class W>
__device__ __forceinline__ W residue(float f, double scale, const PrimeConst& c, unsigned long long* ovf) {
    const double x = (double)f * scale;
    long long m;
    if (fabs(x) < 9.2e18) {
        m = __double2ll_rn(x);
    } else {
        m = x < 0 ? -(1ll << 62) : (1ll << 62);
        if (ovf) atomicAdd(ovf, 1ull);
    }
    const unsigned long long a = (unsigned long long)(m < 0 ? -m : m);
    W v;
    if constexpr (sizeof(W) == 2) v = (W)(a % c.p);
    else v = reduce_dw<W>((WideOf<W>)a, c);
    return (m < 0 && v != 0) ? (W)((W)c.p - v) : v;
}

template <class W>
__global__ void __launch_bounds__(256) cols_f32(CoeffArgs A) {
    __shared__ W s[256 * 32 / ntt::PH<W>];
    const int r = blockIdx.y / A.L, l = blockIdx.y - r * A.L;
    const float* src = A.rows[r] + blockIdx.x * 32;
    W* a = (W*)A.scratch + ((size_t)r * A.L + l) * N + blockIdx.x * 32;
    const PrimeConst& pk = A.pr[l];
    const W p = (W)pk.p;
    const W* w = (const W*)A.fw + (size_t)l * N;
    const W* ws = (const W*)A.fws + (size_t)l * N;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    W x[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) x[j] = residue<W>(src[(wp + 8 * j) * 256 + lane], A.scale, pk, A.overflow);
    // from here: the column pass on 32-column blocks
#pragma unroll
    for (int st = 0; st < 5; ++st) {
        const int h = 16 >> st;
        const int m = 1 << st;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            if ((j & h) == 0) {
                const int i = j / (2 * h);
                ntt::ct_bf<W>(x[j], x[j + h], __ldg(w + m + i), __ldg(ws + m + i), p);
            }
        }
    }
#pragma unroll
    for (int ph = 0; ph < ntt::PH<W>; ++ph) {
        if (ph) __syncthreads();
#pragma unroll
        for (int j = ph * 32 / ntt::PH<W>; j < (ph + 1) * 32 / ntt::PH<W>; ++j) s[(wp + 8 * j - ph * 256 / ntt::PH<W>) * 32 + lane] = x[j];
        __syncthreads();
#pragma unroll
        for (int q = ph * 4 / ntt::PH<W>; q < (ph + 1) * 4 / ntt::PH<W>; ++q)
#pragma unroll
            for (int u = 0; u < 8; ++u) x[8 * q + u] = s[(8 * (wp + 8 * q) + u - ph * 256 / ntt::PH<W>) * 32 + lane];
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int g = wp + 8 * q;
        W* y = x + 8 * q;
        {
            const W tw = __ldg(w + 32 + g), tws = __ldg(ws + 32 + g);
#pragma unroll
            for (int u = 0; u < 4; ++u) ntt::ct_bf<W>(y[u], y[u + 4], tw, tws, p);
        }
        {
#pragma unroll
            for (int hb = 0; hb < 2; ++hb) {
                const W tw = __ldg(w + 64 + 2 * g + hb), tws = __ldg(ws + 64 + 2 * g + hb);
                ntt::ct_bf<W>(y[4 * hb], y[4 * hb + 2], tw, tws, p);
                ntt::ct_bf<W>(y[4 * hb + 1], y[4 * hb + 3], tw, tws, p);
            }
        }
        {
#pragma unroll
            for (int qb = 0; qb < 4; ++qb) {
                const W tw = __ldg(w + 128 + 4 * g + qb), tws = __ldg(ws + 128 + 4 * g + qb);
                ntt::ct_bf<W>(y[2 * qb], y[2 * qb + 1], tw, tws, p);
            }
        }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int u = 0; u < 8; ++u) a[(8 * (wp + 8 * q) + u) * 256 + lane] = x[8 * q + u];
}

// ntt_fwd_rows on one 256-element segment R held by a warp: x in layout A (x[j] = element lane + 32 j) on entry,
// layout B (x[u] = element 8 lane + u) on exit
template <class W>
__device__ __forceinline__ void row_pass(W (&x)[8], W* sw, const W* w, const W* ws, W p, int R, int lane) {
    {
        const W tw = __ldg(w + 256 + R), tws = __ldg(ws + 256 + R);
#pragma unroll
        for (int j = 0; j < 4; ++j) ntt::ct_bf<W>(x[j], x[j + 4], tw, tws, p);
    }
    {
#pragma unroll
        for (int hb = 0; hb < 2; ++hb) {
            const int ti = 512 + 2 * R + hb;
            const W tw = __ldg(w + ti), tws = __ldg(ws + ti);
            ntt::ct_bf<W>(x[4 * hb], x[4 * hb + 2], tw, tws, p);
            ntt::ct_bf<W>(x[4 * hb + 1], x[4 * hb + 3], tw, tws, p);
        }
    }
    {
#pragma unroll
        for (int qb = 0; qb < 4; ++qb) {
            const int ti = 1024 + 4 * R + qb;
            const W tw = __ldg(w + ti), tws = __ldg(ws + ti);
            ntt::ct_bf<W>(x[2 * qb], x[2 * qb + 1], tw, tws, p);
        }
    }
    __syncwarp();
#pragma unroll
    for (int j = 0; j < 8; ++j) sw[lane + 32 * j + ((lane + 32 * j) >> 5)] = x[j];
    __syncwarp();
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int e = 8 * lane + u;
        x[u] = sw[e + (e >> 5)];
    }
    {
        const int ti = 2048 + 8 * R + (lane >> 2);
        const W tw = __ldg(w + ti), tws = __ldg(ws + ti);
        const bool lo = (lane & 2) == 0;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const W o = shfl_xor<W>(0xffffffffu, x[u], 2);
            const W U = lo ? x[u] : o, V = lo ? o : x[u];
            const W v = mul_shoup<W>(V, tw, tws, p);
            x[u] = lo ? add<W>(U, v, p) : sub<W>(U, v, p);
        }
    }
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
    {
        const int ti = 8192 + 32 * R + lane;
        const W tw = __ldg(w + ti), tws = __ldg(ws + ti);
#pragma unroll
        for (int u = 0; u < 4; ++u) ntt::ct_bf<W>(x[u], x[u + 4], tw, tws, p);
    }
    {
#pragma unroll
        for (int hb = 0; hb < 2; ++hb) {
            const int ti = 16384 + 64 * R + 2 * lane + hb;
            const W tw = __ldg(w + ti), tws = __ldg(ws + ti);
            ntt::ct_bf<W>(x[4 * hb], x[4 * hb + 2], tw, tws, p);
            ntt::ct_bf<W>(x[4 * hb + 1], x[4 * hb + 3], tw, tws, p);
        }
    }
    {
#pragma unroll
        for (int qb = 0; qb < 4; ++qb) {
            const int ti = 32768 + 128 * R + 4 * lane + qb;
            const W tw = __ldg(w + ti), tws = __ldg(ws + ti);
            ntt::ct_bf<W>(x[2 * qb], x[2 * qb + 1], tw, tws, p);
        }
    }
}

template <class W, class MM>
__global__ void __launch_bounds__(256) rows_mac(CoeffArgs A) {
    __shared__ W s[8][256 + 8];
    const int l = blockIdx.y;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int R = blockIdx.x * 8 + wp;
    const PrimeConst& pk = A.pr[l];
    const W p = (W)pk.p;
    const W* w = (const W*)A.fw + (size_t)l * N;
    const W* ws = (const W*)A.fws + (size_t)l * N;
    const size_t seg = (size_t)l * N + (size_t)R * 256 + 8 * lane;   // this lane's 8 outputs in a row-l polynomial
    W acc0[8], acc1[8];
    if (A.accumulate) {
        const V4<W> a0 = load4<W>((const W*)A.out + seg), a1 = load4<W>((const W*)A.out + seg + 4);
        const V4<W> b0 = load4<W>((const W*)A.out + A.part_stride + seg), b1 = load4<W>((const W*)A.out + A.part_stride + seg + 4);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            acc0[u] = a0.v[u]; acc0[4 + u] = a1.v[u];
            acc1[u] = b0.v[u]; acc1[4 + u] = b1.v[u];
        }
    } else {
#pragma unroll
        for (int u = 0; u < 8; ++u) acc0[u] = acc1[u] = 0;
    }
    for (int r = 0; r < A.n; ++r) {
        const W* sc = (const W*)A.scratch + ((size_t)r * A.L + l) * N + (size_t)R * 256;
        W x[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) x[j] = sc[lane + 32 * j];
        row_pass<W>(x, s[wp], w, ws, p, R, lane);
        const W* c0 = (const W*)A.ct[r] + seg;
        const W* c1 = c0 + A.part_stride;
        const V4<W> u0 = ldg4<W>(c0), u1 = ldg4<W>(c0 + 4), v0 = ldg4<W>(c1), v1 = ldg4<W>(c1 + 4);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            acc0[u] = MM::reduce(wmul<W>(x[u], u0.v[u]) + acc0[u], pk);
            acc0[4 + u] = MM::reduce(wmul<W>(x[4 + u], u1.v[u]) + acc0[4 + u], pk);
            acc1[u] = MM::reduce(wmul<W>(x[u], v0.v[u]) + acc1[u], pk);
            acc1[4 + u] = MM::reduce(wmul<W>(x[4 + u], v1.v[u]) + acc1[4 + u], pk);
        }
    }
    W* o0 = (W*)A.out + seg;
    W* o1 = o0 + A.part_stride;
    store4<W>(o0, acc0[0], acc0[1], acc0[2], acc0[3]);
    store4<W>(o0 + 4, acc0[4], acc0[5], acc0[6], acc0[7]);
    store4<W>(o1, acc1[0], acc1[1], acc1[2], acc1[3]);
    store4<W>(o1 + 4, acc1[4], acc1[5], acc1[6], acc1[7]);
}

template <class W, class MM>
int launch(const CoeffArgs* a, cudaStream_t s) {
    if (a->n < 1 || a->n > PC_COEFF_MAXT || a->L < 1 || a->L > PC_COEFF_MAXL) return -1;
    if (!a->scratch || !a->out || !a->fw || !a->fws || !a->pr) return -2;
    cols_f32<W><<<dim3(8, a->n * a->L), 256, 0, s>>>(*a);
    rows_mac<W, MM><<<dim3(32, a->L), 256, 0, s>>>(*a);
    return (int)cudaGetLastError();
}
}  // namespace coeffrows
}  // namespace pc
