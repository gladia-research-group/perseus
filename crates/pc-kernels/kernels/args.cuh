// The host <-> kernel ABI of pc-kernels: per-prime constants, launch argument structs and the KernelTable.
// Every struct here is mirrored field for field by `#[repr(C)]` structs in `src/args.rs`, with size asserts on both
// sides. Change a layout = bump PC_KERNELS_ABI and change the mirror in the same commit.
//
// Word-generic data: every polynomial / table pointer is `void*` and is read as the instantiation's word W
// (uint16_t, uint32_t or uint64_t). Strides and offsets are in units of W words, never bytes. A row is N = 2^16 words.
// Pointers to index tables (rows, prime ids, digit maps, automorphism exponents, seeds) are always 32-bit.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#define PC_KERNELS_ABI 11u
#define PC_RING_ABI 5u
// the ring tables (RingTable) are compiled once per ring size with -DPC_LOGN; everything else is N = 2^16
#ifndef PC_LOGN
#define PC_LOGN 16
#endif
#define PC_N (1 << PC_LOGN)

// ------------------------------------------------------------------------------------------------ PrimeConst
// One entry per prime id, in device global memory (table indexed by prime id). Built for ONE word size (the
// Barrett constant depends on it); every ModMul strategy reads the same table, so a table works for any MM of
// its word: the kernels' contract is the same normal-form residues in and out under every strategy, and every
// strategy also needs the double-word Barrett constant for its lazy accumulators. 64 bytes, 8-aligned.
// pc-core's per-strategy host consts (modmul.rs) differ; at W = u32 the first 16 bytes here equal its Barrett64Const
// (p, pad = 0, mu = floor(2^64/p)). Host side builds this table with pc_kernels::PrimeConst::new(p, word_bits).
//   off  field      meaning
//    0   p          the prime (< 2^(BITS-1); Barrett32 at W=u32 additionally needs p < 2^30)
//    8   mu_lo      low 64 bits of floor(2^(2*BITS) / p)   (u16: floor(2^32/p), u32: floor(2^64/p),
//                   u64: low word of floor(2^128/p), computed as floor((2^128-1)/p))
//   16   mu_hi      high 64 bits of floor(2^(2*BITS) / p)  (u64 only; 0 for u16/u32)
//   24   neal_mu    floor(2^(2*bits + 1) / p)               (Barrett32 = FIDESlib Neal_mult_32, mu_new with NEIL)
//   32   bits       bit_width(p)
//   36   word_bits  BITS of the word this entry was built for (16, 32 or 64); kernels do not read it, checks do
//   40   mont_pinv  -p^-1 mod 2^BITS                        (Montgomery)
//   48   mont_r2    2^(2*BITS) mod p                         (Montgomery: R^2, converts a REDC result back)
//   56   reserved   0
struct PrimeConst {
    uint64_t p;
    uint64_t mu_lo;
    uint64_t mu_hi;
    uint64_t neal_mu;
    uint32_t bits;
    uint32_t word_bits;
    uint64_t mont_pinv;
    uint64_t mont_r2;
    uint64_t reserved;
};
static_assert(sizeof(PrimeConst) == 64, "PrimeConst layout");

// ------------------------------------------------------------------------------------------------ common views
// Row set of the element-wise kernels: the Q rows [0, nq) and, when np > 0, the special rows [qmax, qmax + np);
// grid row y -> row index (== prime id) y < nq ? y : qmax + (y - nq).
struct RowSet {
    int32_t nq, np, qmax, pad;
};
static_assert(sizeof(RowSet) == 16, "RowSet layout");

// Polynomial part `part` of ciphertext c, row r at base + c*ct_stride + part*part_stride + r*N (words).
struct PolyRef {
    void* base;
    uint64_t ct_stride, part_stride;
};
static_assert(sizeof(PolyRef) == 24, "PolyRef layout");

// ------------------------------------------------------------------------------------------------ NTT
// In-place negacyclic NTT of `nrows` rows: row i at base + roff[i]*N, prime id rprime[i]. OpenFHE order (CT,
// bit-reversed output; GS inverse with N^-1 folded in). Twiddle tables: [prime id][N] words in bit-reversed index
// order (fw[m + i] = psi^brev(m + i)), with Shoup companions; ninv/ninvs: [prime id]. `chunk` rows per launch pair
// (both passes of a chunk run back to back so the intermediate stays in L2; 0 = all rows at once).
struct NttArgs {
    void* base;
    const uint32_t* roff;
    const int32_t* rprime;
    const void* fw;
    const void* fws;
    const void* iw;
    const void* iws;
    const void* ninv;
    const void* ninvs;
    const PrimeConst* pr;
    int32_t nrows, chunk;
};
static_assert(sizeof(NttArgs) == 88, "NttArgs layout");

// ------------------------------------------------------------------------------------------------ BConv
// Batched fast base conversion (coefficient domain), one launch for many jobs (digits x ciphertexts):
//   y_i = x_i * pre_i mod s_i (i < nsrc <= 16);  out_t = sum_i y_i * M[i][t] mod t;
//   exact jobs: v = sum_i y_i * rb_i (double FMA, source order), u = floor(v + 0.5), out_t -= u * (B mod t).
// The pool is an array of W words. Pool layout at pool_off (all counts in W words):
//   src_row[nsrc], src_prime[nsrc], pre[nsrc], tgt_row[ntgt], tgt_prime[ntgt], M[nsrc][ntgt],
//   (exact) rb[nsrc] as double (8/sizeof(W) words each, first one 8-byte aligned in the pool), bmod[ntgt].
// Bound: nsrc * (max p - 1)^2 < 2^(2*BITS) (lazy accumulation in the double word).
struct BconvJob {
    uint32_t src_off, dst_off;  // limb offsets (units of N) into src / dst
    uint32_t nsrc, ntgt;
    uint32_t pool_off;          // W words
    uint32_t exact;
    uint32_t pad[2];
};
static_assert(sizeof(BconvJob) == 32, "BconvJob layout");

struct BconvArgs {
    const void* src;
    void* dst;
    const BconvJob* jobs;
    const void* pool;
    const PrimeConst* pr;
    int32_t njobs, max_tgt;
};
static_assert(sizeof(BconvArgs) == 48, "BconvArgs layout");

// ------------------------------------------------------------------------------------------------ element-wise
// One struct for the four row-set kernels; grid = (N/256, rows of rs, K*nparts).
//   ew_add:     out = a + b (sub: a - b)
//   ew_scale:   out = a * sc[row] (Shoup companion scs); special rows written as 0 when zero_p
//   ew_sub_mul: out = (a - b) * sc[row]  (ModDown / ModDown+rescale epilogue in the NTT domain)
//   ew_copy:    out = a
struct EwArgs {
    PolyRef out, a, b;
    RowSet rs;
    const void* sc;
    const void* scs;
    const PrimeConst* pr;
    int32_t K, nparts, sub, zero_p;
};
static_assert(sizeof(EwArgs) == 128, "EwArgs layout");

// BSGS plaintext dot of an LT stage in the extended basis, K <= 8 ciphertexts sharing the plaintext reads:
//   res[c][j][part] = sum_{i < bstep} rot[c][i][part] * pt[bstep*j + i]   (j < gstep; diagonals >= npts skipped)
// rot (c, i) at rot.base + c*rot.ct_stride + i*rot_stride; res (c, j) at res.base + c*res.ct_stride + j*res_stride.
// pts[d]: diagonal d's rows (ptw words per row; Q rows then special rows from pt_prow), read at the compact index
// (enc = 0xFFFFFFFF full; (enc >> 24) == 0x7F: idx >> (enc & 0xFF); else idx & enc).
struct PtDotArgs {
    PolyRef rot, res;
    uint64_t rot_stride, res_stride;
    const void* const* pts;
    const PrimeConst* pr;
    RowSet rs;
    int32_t npts, bstep, gstep, K;
    int32_t ptw, pt_prow;
    uint32_t enc;
    int32_t pad;
};
static_assert(sizeof(PtDotArgs) == 128, "PtDotArgs layout");

// ------------------------------------------------------------------------------------------------ key switching
// Batched hybrid key-switch inner product, K ciphertexts x R keys per launch (each key element read once):
//   s1 = sum_d D_d[t][i] a_{r,d}[t][i],  s0 = C0[t][i] + sum_d D_d[t][i] b_{r,d}[t][i]
// D_d[t] = c1 itself when t is in digit d, else the NTT'd ModUp output; C0: c0_mode 0 none, 1 P*c0 on Q rows,
// 2 c0 extended. Outputs written at automorph_slot(autok[r], i) (k = 1: identity). Grid row y -> prime id rows[y].
struct KsArgs {
    const void* c1;
    const void* modup;
    const void* c0;
    void* out0;
    void* out1;
    const void* const* kb;  // per key r: [digit][prime id][N]
    const void* const* ka;
    const uint32_t* autok;
    const uint32_t* rows;
    const int32_t* digit_of;  // prime id -> digit (Q rows), -1 special
    const void* pmod;         // prime id -> P mod t (mode 1), Shoup in pmods
    const void* pmods;
    const PrimeConst* pr;
    uint64_t ct_stride, mu_ct_stride, mu_digit_stride, out_ct_stride, out_rot_stride, key_digit_stride;
    int32_t K, R, ndig, c0_mode, nq, nrows;
};
static_assert(sizeof(KsArgs) == 176, "KsArgs layout");

// Fused relin epilogue on Q rows: dst = f (dst + (dot - conv) binv) [+ k on part 0], f = 2 when dbl.
struct EpiArgs {
    void* dst;
    const void* conv;
    const void* binv;
    const void* binvs;
    const void* k;  // per-row constant (nullptr: none)
    uint64_t part_stride;
    int32_t dbl, pad;
};
static_assert(sizeof(EpiArgs) == 56, "EpiArgs layout");

// A set of switching keys read straight from their stored form (KeyTab, ABI 4; see the seeded keys section).
struct KeyTab {
    const void* const* kb;         // per key: b storage
    const uint32_t* const* seed;   // per key: 8-word KSKA seed of `a`
    uint64_t row_words;            // words per stored row: W words (bits == 0) or u32 words (packed)
    int32_t bits;                  // 0 = dense W rows, else packed value width (u32 instantiations only)
    int32_t q_top;                 // highest Q prime id the keys hold; -1 = full layout (slot = prime id)
    int32_t srows;                 // slots per digit = q_top + 1 + |P|
    int32_t nregen;                // digits [0, nregen) regenerate `a`, the rest read the resident dense `a` (full
                                   // layout) through cts.ka / ks.ka per key and the args' key_digit_stride
};
static_assert(sizeof(KeyTab) == 40, "KeyTab layout");

// Relin inner product with the key's `a` regenerated from its seed (KSKA expander, see KskExpandArgs); K <= 4; no
// automorphism, no c0 term. `b`: key 0 of `key` read in place when key.kb is set (ABI 7: dense / packed / banded as
// stored; key.seed / nregen unused, the seed is `seed`), else ks.kb[0] (dense prime-id rows). use_epi: write through
// EpiArgs instead of out0/out1; `add` (ABI 7, epilogue only): + add on both parts (laid out as dst), after dbl / k.
struct KsRegenArgs {
    KsArgs ks;
    const uint32_t* seed;  // 8 words
    EpiArgs epi;
    int32_t use_epi, pad;
    KeyTab key;
    const void* add;
};
static_assert(sizeof(KsRegenArgs) == 296, "KsRegenArgs layout");

// Expander, ChaCha12 counter mode, state words 12..15 = (block counter, w13, w14, tag); FROZEN at u32 and u64.
// u32 (FIDESlib KSKA lane): w13 = d, w14 = p, tag "KSKA"; slot s -> block s/16 word s%16; a value v is accepted when
// v < floor((2^32-1)/p)*p, else retried at block counter + t*N/16 (t = 1..15); output v mod p.
// u64 (FIDESlib KSKB lane): w13 = d ^ hi32(p) << 4 (= (d & 0xF) | hi32(p) << 4 for d < 16), w14 = lo32(p),
// tag "KSKB" (= "KSKA" + 1); slot s -> words 2j, 2j+1 (little-endian) of block s/8, j = s%8; retry stride N/8;
// threshold floor((2^64-1)/p)*p. Needs dnum <= 16 and p < 2^60. Host twins: pc_core::keys::{kska_expand, kskb_expand}.
// u16 (not frozen): 32 values per block (halves), w14 = p.
// ksk_expand: out[d][m][N] for d < dnum, m < nmod (prime ids prime_ids[m]), w13 = d, tag "KSKA" (0x4b534b41).
struct KskExpandArgs {
    void* out;
    const uint32_t* seed;
    const int32_t* prime_ids;
    const PrimeConst* pr;
    int32_t nmod, dnum;
};
static_assert(sizeof(KskExpandArgs) == 40, "KskExpandArgs layout");

// Fused CtS stage core: hoisted baby rotations + BSGS plaintext dot in the extended basis, gather form, K cts (groups
// of kmax in {1, 2, 4, 8}), G <= 2 giant steps. See cts.cuh for the formula.
struct CtsArgs {
    const void* x;      // ct c at x + c*ct_stride, c1 at + part_stride
    const void* modup;  // ct c at + c*mu_ct_stride, digit d at + d*mu_digit_stride
    void* res;          // ct c, giant j at res + (c*G + j)*ct_stride
    const void* const* kb;  // baby i >= 1: key i-1, [digit][prime id][N]
    const void* const* ka;
    const uint32_t* kinv;   // baby i >= 1: inverse automorphism exponent
    const void* const* pts;
    const uint32_t* rows;
    const int32_t* digit_of;
    const void* pmod;
    const void* pmods;
    const PrimeConst* pr;
    uint64_t ct_stride, part_stride, mu_ct_stride, mu_digit_stride, key_digit_stride;
    int32_t K, B, G, npts, ndig, nq, nlive, ptw, pt_prow;
    uint32_t enc;
    int32_t ngrp;  // set by the launcher
    int32_t nrows, kmax;
    // > 0: giant steps j = 2l, 2l+1 of a launch sit at res + l * pair_cts cts (the LtStage result layout), so one
    // K=1 launch carries up to 4 giant steps and regenerates the baby keys once; 0: all at res + (c G + j) cts (G <= 2)
    int32_t pair_cts;
};
static_assert(sizeof(CtsArgs) == 192, "CtsArgs layout");

// ------------------------------------------------------------------------------------------------ EvalMod kernels
// Q rows [0, n) of K ciphertexts (row index == prime id), PolyRef layout.
// tensor: (c0, c1, d2) (= | +=) sum_{j < npairs} (a0 b0, a0 b1 + a1 b0, a1 b1); d2 written to part 1 of `d2`, and to
// coef (when non-null) at coef + c*coef_ct_stride + row*N.
struct TensorArgs {
    PolyRef out, d2, a0, b0, a1, b1;
    void* coef;
    uint64_t coef_ct_stride;
    const PrimeConst* pr;
    int32_t npairs, acc, n, K;
};
static_assert(sizeof(TensorArgs) == 184, "TensorArgs layout");

// md_epilogue: for parts 0, 1: dst = f (dst + (x - t) binv) [+ k on part 0]
struct MdEpiArgs {
    PolyRef dst, x, t;
    const void* binv;
    const void* binvs;
    const void* k;
    const PrimeConst* pr;
    int32_t dbl, n, K, pad;
};
static_assert(sizeof(MdEpiArgs) == 120, "MdEpiArgs layout");

// md_epilogue_to (ABI 7): dst = a +- (x - t) binv over Q rows [0, n), both parts (sub: minus); m.k / m.dbl unused.
// EvalMod's x +- conj(x) straight from the conjugation's ModDown.
struct MdEpiToArgs {
    MdEpiArgs m;
    PolyRef a;
    int32_t sub, pad;
};
static_assert(sizeof(MdEpiToArgs) == 152, "MdEpiToArgs layout");

// wsum: out = sum_{j < nsrc <= 8} w[j][row] * src_j (both parts), Shoup companions ws
struct WsumArgs {
    PolyRef out;
    PolyRef src[8];
    const void* w;
    const void* ws;
    const PrimeConst* pr;
    int32_t nsrc, rowstride, n, K;
};
static_assert(sizeof(WsumArgs) == 256, "WsumArgs layout");

// wsum_k (ABI 7): wsum, then part 0 += k[row] (raw per-row constant, reduced once if >= p; = wsum + add_const)
struct WsumKArgs {
    WsumArgs w;
    const void* k;
};
static_assert(sizeof(WsumKArgs) == 264, "WsumKArgs layout");

// add_const: part 0 += k[row];  mul_table: out = in * t[row][i] (both parts; general ModMul product)
struct AddConstArgs {
    PolyRef x;
    const void* k;
    const PrimeConst* pr;
    int32_t n, K;
};
static_assert(sizeof(AddConstArgs) == 48, "AddConstArgs layout");

struct MulTableArgs {
    PolyRef out, in;
    const void* t;
    const PrimeConst* pr;
    int32_t n, K;
};
static_assert(sizeof(MulTableArgs) == 72, "MulTableArgs layout");

// ------------------------------------------------------------------------------------------------ rescale
// Composite one-pass rescale by the top ndrop (1 or 2) primes:
//   host: INTT the dropped tops of x; rescale_coef -> tmp (coefficient form); NTT the nrows rows of tmp;
//   rescale_combine: x_i = alpha_i x_i + tmp_i (NTT domain).
// With a = top, b = top - 1, lift(y, q) = centered y (y > q/2 -> y - q) mod q_i:
//   ndrop 2: y' = qinv_ab y_b + C_ab lift(y_a) mod q_b;  tmp_i = A_i lift(y_a) + B_i lift(y')
//   ndrop 1: tmp_i = A_i lift(y_a)
// ca, cb: a scalar folded in front (MulConst before the rescale) on the tops (1 = none).
// Host constants (OpenFHE): A_i = qinv_b qlql_a (ndrop 2) or qlql_a (ndrop 1), B_i = qlql_b, alpha_i = qinv_a qinv_b
// (ndrop 2) or qinv_a, qlql_x(i) = -q_x^-1 mod q_i, qinv_ab = q_a^-1 mod q_b, C_ab = qlql_a(b); qam/qbm = q_a, q_b mod q_i.
struct RescaleArgs {
    PolyRef x, tmp;
    const void* A;
    const void* B;
    const void* As;
    const void* Bs;
    const void* qam;
    const void* qbm;
    const PrimeConst* pr;
    uint64_t qinv_ab, C_ab, ca, cb;
    int32_t top, nrows, K, ndrop;
};
static_assert(sizeof(RescaleArgs) == 152, "RescaleArgs layout");

struct RescaleCombineArgs {
    PolyRef x, tmp;
    const void* alpha;
    const void* alphas;
    const PrimeConst* pr;
    int32_t n, K;
};
static_assert(sizeof(RescaleCombineArgs) == 80, "RescaleCombineArgs layout");

// rescale_combine_to (ABI 7, EvalMod's out-of-place rescale): out_i = alpha_i x_i + tmp_i (x read only)
struct RescaleCombineToArgs {
    RescaleCombineArgs a;
    PolyRef out;
};
static_assert(sizeof(RescaleCombineToArgs) == 104, "RescaleCombineToArgs layout");

// ------------------------------------------------------------------------------------------------ automorphism
// out = sigma_k(in) in the bit-reversed evaluation order (gather): out[o] = in[automorph_slot(k, o)], where
// automorph_slot(k, o) = brev(((2 brev(o) + 1) k mod 2N) >> 1). With add: out[o] = in[slot] + add[o].
// Row set rs, K cts x nparts parts. out must not alias in.
struct AutomorphArgs {
    PolyRef out, in, add;
    RowSet rs;
    const PrimeConst* pr;
    uint32_t k;
    int32_t K, nparts, has_add;
};
static_assert(sizeof(AutomorphArgs) == 112, "AutomorphArgs layout");

// ------------------------------------------------------------------------------------------------ encode
// CKKS packed encode on the device (FIDESlib encodeOnDevice): slot j at zeta^(5^j) and its conjugate; radix-2 Stockham
// FFT (exp(-2 pi i ...)); c_k = Re(Y_k zeta^-k / N) * scale; split = max(0, ceil(log2 max|c|) - 61);
// m_k = llround(c_k / 2^split); out limb l (prime id l) = m_k * deg_factor[l] * 2^split mod p_l, COEFFICIENT form
// (NTT it with ntt_fwd). v: N/2 double2 (re, im) slots; scratch a, b: N double2; c: N double; cmax: one u64.
struct EncodeArgs {
    const void* v;
    void* a;
    void* b;
    void* c;
    unsigned long long* cmax;
    void* out;               // limb l at out + l*N
    const void* deg_factor;  // [limb] round(scale)^(deg-1) mod p_l (W words)
    const PrimeConst* pr;
    double scale;
    int32_t nl, pad;
};
static_assert(sizeof(EncodeArgs) == 80, "EncodeArgs layout");

// ------------------------------------------------------------------------------------------------ encrypt
// Secret-key encryption in the NTT domain, limb l (prime id l) at + l*N:
//   a = expander(seed, w13 = nonce, tag "ENCA" = 0x454e4341) per limb (the KskExpandArgs rule),
//   c0 = -a s + e + m (m optional), c1 = a (written when c1 != nullptr; otherwise regenerable from the seed).
struct EncryptArgs {
    void* c0;
    void* c1;
    const void* s;
    const void* e;
    const void* m;
    const uint32_t* seed;
    const PrimeConst* pr;
    uint32_t nonce;
    int32_t nl;
};
static_assert(sizeof(EncryptArgs) == 64, "EncryptArgs layout");

// ------------------------------------------------------------------------------------------------ keygen (ABI 2)
// One hybrid switching key (OpenFHE KeySwitchGenInternal relation), every stored digit d < ndig and slot m < nrows
// (prime id ids[m]) in one launch, eval form:
//   a = expander(seed, w13 = d, tag "KSKA") at prime ids[m] (the KskExpandArgs rule),
//   b = e[d][m] - a s_new[id] + (id < nq && digit_of[id] == d ? pmod[id] * s_old'[id] : 0), s_old' = s_old^2 when sq.
// s_new / s_old: [prime id][N]; e: [d][m][N]; pmod: by prime id with Shoup companions pmods.
// b written dense ([d][m][N] W words, bits = 0) or bit-packed (bits > 0, W = u32 only): slot m of digit d at
// b + (d*nrows + m)*row_words u32 words, value i at bits [i*bits, (i+1)*bits) of a little-endian u32 stream, the
// words from N*bits/32 to row_words zeroed (guard). a written dense when a != nullptr.
struct KskGenArgs {
    void* b;
    void* a;
    const void* e;
    const void* s_new;
    const void* s_old;
    const uint32_t* seed;
    const int32_t* ids;
    const int32_t* digit_of;  // prime id -> digit (Q ids)
    const void* pmod;
    const void* pmods;
    const PrimeConst* pr;
    uint64_t row_words;
    int32_t nrows, ndig, nq, bits, sq, pad;
};
static_assert(sizeof(KskGenArgs) == 120, "KskGenArgs layout");

// Packed key rows -> dense W rows: out[r][i] = value i of packed row r (in + r*row_words), r < nrows.
struct KskUnpackArgs {
    void* out;
    const uint32_t* in;
    uint64_t row_words;
    int32_t nrows, bits;
};
static_assert(sizeof(KskUnpackArgs) == 32, "KskUnpackArgs layout");

// ------------------------------------------------------------------------------------------------ bootstrap raise
// Composite ModRaise (FIDESlib compositeModRaise_), coefficient form: for rows t in [r0, r1) of K ciphertexts
// (nparts polynomials from part 0), x[t] = sum_{k < d} centred([x_k * qhatinv_k]_{q_k}) * qhat[k][t] mod q_t, the
// sources x_k = rows 0..d-1 of the same polynomial (each Garner term centred on its own prime; not written).
struct ModRaiseArgs {
    void* x;
    const PrimeConst* pr;
    uint64_t ct_stride, part_stride;
    uint64_t qhatinv[4];
    uint64_t qhat[4][64];
    int32_t K, d, r0, r1, nparts, pad;
};
static_assert(sizeof(ModRaiseArgs) == 2136, "ModRaiseArgs layout");

// Exact small-integer scaling (FIDESlib smallIntDivRound_, u32 chains): per coefficient, Garner from the k source
// rows, centre against Q_k/2, optionally q = round(x * R / 2^192) (R = round(2^192 / D)), reduce modulo every output
// row orow[o] (prime ids, (2^32)^w mod p in pow32[o]), restore the sign. Tables built on the host (bts/raise.rs).
#define PC_SI_MAXK 4
#define PC_SI_MAXW 8
#define PC_SI_MAXR 8
#define PC_SI_SHIFT 192
#define PC_SI_MAXOUT 64
struct SmallIntTab {
    int32_t k, nout, nw, nr, divide, pad;
    uint32_t q[PC_SI_MAXK];
    uint32_t qinv[PC_SI_MAXK][PC_SI_MAXK];  // qinv[i][j] = q_j^-1 mod q_i, j < i
    uint32_t Qk[PC_SI_MAXW], Qkh[PC_SI_MAXW];
    uint32_t R[PC_SI_MAXR];
    int32_t orow[PC_SI_MAXOUT];
    uint32_t pow32[PC_SI_MAXOUT][PC_SI_MAXW + PC_SI_MAXR];
};
static_assert(sizeof(SmallIntTab) == 4552, "SmallIntTab layout");
// source row i of (c, part) at src + c*src_ct_stride + part*src_part_stride + i*N; output row orow[o] of (c, part)
// at out + c*ct_stride + part*part_stride + orow[o]*N.
struct SmallIntArgs {
    const void* src;
    void* out;
    const SmallIntTab* tab;
    const PrimeConst* pr;
    uint64_t src_ct_stride, src_part_stride, ct_stride, part_stride;
    int32_t K, nparts;
};
static_assert(sizeof(SmallIntArgs) == 72, "SmallIntArgs layout");

// ------------------------------------------------------------------------------------------------ spru (ABI 3)
// SPRU extraction (FIDESlib Spru.cu): for each of k ciphertexts (rows of ct c at + c*ct_stride), the coefficient rows
// of prime ids 0 and 1 (c0 / c1 at + l*N, coefficient form) give, per key index i < N and component a < n, the value
// v = [c1 * X^-i]_(a N/n) (+ c0_(a N/n) for i = 0) mod q0 = p0 p1 (CRT with inv = p0^-1 mod p1) and
//   value delta * exp(i pi 2 v / q0) of slot k' h n + b n + a of encoding u = (i mod B) / sub (b = i / B,
//   k' = (i mod B) mod sub, sub = B / (2n), B = N / h), written as the N-point FFT input of that encoding:
//   out[c][u][(t-1)/2] = value, out[c][u][(2N-t-1)/2] = conj value, t = 5^slot mod 2N. out holds k * 2n * N double2.
struct SpruExtractArgs {
    const void* c0;
    const void* c1;
    void* out;
    uint64_t p0, p1, inv;
    double delta;
    uint64_t ct_stride;
    int32_t h, n, B, k;
};
static_assert(sizeof(SpruExtractArgs) == 80, "SpruExtractArgs layout");
// Batched forward Stockham pass of size-N complex transforms (encoding e at in/out + e*N double2), half-size Ns, twiddle
// table tw[m] = exp(-i pi m / (N/2)).
struct SpruFftArgs {
    const void* in;
    void* out;
    const void* tw;
    int32_t ns, batch;
};
static_assert(sizeof(SpruFftArgs) == 32, "SpruFftArgs layout");
// Coefficients of `batch` FFT outputs y (e at + e*N double2): m_k = round(Re(y_k zt_k / N) scale), zt[k] = exp(-i pi k/N),
// reduced into limbs l < nl (prime ids 0..nl) at out + e*out_stride + l*N (coefficient form).
struct SpruScatterArgs {
    const void* y;
    void* out;
    const void* zt;
    const PrimeConst* pr;
    double scale;
    int32_t nl, batch;
    uint64_t out_stride;
};
static_assert(sizeof(SpruScatterArgs) == 56, "SpruScatterArgs layout");
// ------------------------------------------------------------------------------------------------ seeded keys (ABI 4)
// A set of switching keys read straight from their stored form: `a` regenerated in-kernel from each key's
// KSKA seed (the KskExpandArgs rule: w13 = digit, prime of the row's id), `b` read from its storage, dense W rows or
// 28-bit-style packed u32 rows (KskGenArgs bit order, guard word), in the key's slot order: digit d, slot m at row
// d*srows + m; Q prime id t <= q_top at slot t, special prime id nq + j at slot q_top + 1 + j (KskShape::slot_of).

// cts_fused with the baby keys read through KeyTab (`cts.kb` unused; `cts.ka` only for digits >= key.nregen).
struct CtsRegenArgs {
    CtsArgs cts;
    KeyTab key;
};
static_assert(sizeof(CtsRegenArgs) == 232, "CtsRegenArgs layout");

// ks_dot (K <= 4 ciphertexts x R keys, automorphism, c0 modes) with the keys read through KeyTab (`ks.kb` unused;
// `ks.ka` / `ks.key_digit_stride` only for digits >= key.nregen); 4 consecutive slots per thread (cooperative ChaCha
// at u32).
struct KsKeyArgs {
    KsArgs ks;
    KeyTab key;
};
static_assert(sizeof(KsKeyArgs) == 216, "KsKeyArgs layout");

// ------------------------------------------------------------------------------------------------ pcmm (ABI 5)
// PCMM at the StC-first split (FIDESlib Pcmm.cu): one launcher, op-coded. Buffers are u32 residues
// unless stated; "staging" = W words [part][4][N] (part 0 = c1 "a", part 1 = c0 "b"); primes p[] are q0, q1, P (key
// ops) or the chain's q0..q3 (in_rescale), mu[i] = floor(2^64 / p[i]); n = ring degree of the op (N or N1 = N/16).
//   GATHER     x = staging (W) <- ct member i0 (PolyRef ct), rows 0..3 of both parts
//   SCATTER    ct member i0 rows 0..3 <- y = staging (W)
//   IN_RESCALE y = staging (W, coefficient form) -> x [part][2][N]: exact rescale by p[3] then p[2]
//              (c[0..3) = p[3]^-1 mod p[0..3), c[3..5) = p[2]^-1 mod p[0], p[1])
//   LIFT       y = f [2][N] -> x = staging part [4][N] (W): limb i < 2 = f[i] c[i] mod p[i], limbs 2, 3 = 0
//   KS_DIGITS  y = x [2][count][n] -> x = D [2][3][count][n] = centered(x[j] c[j] mod p[j]) mod p[pi]
//   KS_MAC     y = D [2][3][i0 i1][n], z = keys [i1][2][3][2][n] -> x = acc [2][3][i0][n] (outs i0, nl i1)
//   MOD_DOWN   y = acc -> x = [2][2][i0][n]: (acc_i - centered(acc_P)) c[i] mod p[i]   (c = P^-1 mod q_i)
//   ADD        x += y over count primes x i0 words (prime of word k = k / i0)
//   NTT, INTT  x = count polys of n in place, prime of poly k = (k / i0) % 3; tw = [3][n] twiddles (psi^brev /
//              psi^-brev); INTT scales by c[prime] = n^-1. n a multiple of 4096.
//   MLWE_A     y = ka [3][N] -> x = A [2][i0 rows][N1] at row offset i1 (count = NCH rows, R = i2, NP = i3)
//   MLWE_B     y = b [3][N] -> x = Bm [2][i0][NP]
//   DIGITS     y = X [2][i0 rows][i1 cols] -> x = int8 d[4][2 i1][i0] (balanced base-256 digits of centered X;
//              K-contiguous: the GEMM's A stored k x m column-major)
//   RECOMBINE  y = G int32 [i4][i0][2 i3] -> x = Y [2][i2 rows][i3 cols] at row offset i1 (Horner in 256)
//   PACK_C     y = Ap [2][i0][N1] -> x = C [2][count = NCH][N1] at row offset i1 (R = i2, NP = i3)
//   ADD_BPK    x = pb [2][16][N1] += y = Bp [2][i0][NP] rows at offset i1 (R = i2, NP = i3)
//   COMPOSE    y = px [2][16][N1] -> x = RX [2][N]
enum PcmmOp : int32_t {
    PCMM_GATHER = 0,
    PCMM_SCATTER = 1,
    PCMM_IN_RESCALE = 2,
    PCMM_LIFT = 3,
    PCMM_KS_DIGITS = 4,
    PCMM_KS_MAC = 5,
    PCMM_MOD_DOWN = 6,
    PCMM_ADD = 7,
    PCMM_NTT = 8,
    PCMM_INTT = 9,
    PCMM_MLWE_A = 10,
    PCMM_MLWE_B = 11,
    PCMM_DIGITS = 12,
    PCMM_RECOMBINE = 13,
    PCMM_PACK_C = 14,
    PCMM_ADD_BPK = 15,
    PCMM_COMPOSE = 16,
};
struct PcmmArgs {
    int32_t op, n, count, i0, i1, i2, i3, i4;
    void* x;
    const void* y;
    const void* z;
    const void* tw;
    PolyRef ct;
    uint32_t p[4];
    uint64_t mu[4];
    uint32_t c[8];
};
static_assert(sizeof(PcmmArgs) == 168, "PcmmArgs layout");

// ------------------------------------------------------------------------------------------------ ccmm (ABI 6)
// CCMM (FIDESlib Ccmm.cu, CKL ePrint 2025/1957): one launcher, op-coded. Buffers are u32 residues unless
// stated; primes p[0..4) = q0..q3 (the chain's limbs 0..3), p[4] = P, mu[i] = floor(2^64 / p[i]). Shapes (constants
// of the kernels): NB = N = 2^16, NS = 2^13, D = 128, KY = 64, NBIG = 16, NQ = 4, NP = 5. "Staging" = W words
// [slot][part][4][NB] (part 0 = c0 = b, part 1 = c1 = a); "columns" X = [part][prime < 4][col < 128][NS].
//   GATHER       x = staging slot i1 <- ct member i0 (PolyRef ct), limbs 0..3 of both parts
//   SCATTER      ct member i0 limbs 0..3 <- y = staging slot i1
//   NTT, INTT    x = count polys of n in place (64 <= n <= 2^16, powers of two), prime of poly k = (k / i0) % i1;
//                tw = [prime][n] (psi^brev / psi^-brev, the NttTab order); INTT scales by c[prime] = n^-1
//   KS_DIGITS    source y (i4 = 1: W words, else u32), digit j of poly q at y + j s[0] + q s[1] -> x = Dg
//                [4][5][count][n] = centered(src c[j] mod p[j]) mod p[pi]   (c[0..4) = (Q / q_j)^-1 mod q_j)
//   KS_MAC       y = Dg, z = keys [4][5][2][n] (+ q s[0] for poly q; s[0] = 0: one key) -> x = acc [2][5][count][n]
//   MOD_DOWN     y = acc -> x = res [2][4][count][n]: (acc_i - centered(acc_P)) c[i] mod p[i]   (c = P^-1 mod q_i)
//   RING_IN      y = staging (W, coefficient form, 16 slots), z = res of the fwd switch [2][4][16][NB] -> x = X:
//                X[0][i][8c + j][z] = c0[c][i][j + 8z] + res[1][i][c][j + 8z], X[1][i][8c + j][z] = res[0][i][c][j + 8z]
//   COMPOSE_A    y = X -> x = bigA [4][16][NB]: bigA[i][c][j + 8z] = X[1][i][8c + j][z]
//   RING_OUT     y = X, z = res of the back switch -> x = staging (W, 16 slots, coefficient form):
//                part 0 = X[0] composed + res[1], part 1 = res[0]
//   MONO         y -> x: column col times X^(i0 col) (negacyclic, NS), over [2][4][D][NS]
//   BITREV       y -> x: column col <- column brev7(col)
//   TWEAK        y -> x: one radix-2 level of CKL Alg. 2 (h = i0, kk = i1, sgn = i2)
//   SCRAMBLE     y -> x: x[t][pos(u h_t)] = +-y[tstar[t]][u] c[prime]  (z = int32 tstar[128], c = D^-1 mod q_i)
//   KS_APPLY     x = X [2][4][s[0] words per prime]: b += res[1], a = res[0] (y = res [2][4][s[0]])
//   ADD          x += y over 4 primes x s[0] words
//   GATHER_U     y = X -> x = E [2][4][D col][D i][KY u] = X[..][col][i + D u]
//   DIGITS       y = E (one matrix), prime i0 -> x = int8 d[4][KY u][2D r][D t] = balanced base-256 digit of
//                centered E[r / D][i0][t][r % D][u] (K-contiguous: both GEMM operands of the col-TN layout)
//   RECOMBINE    y = G int32 [7][KY][2D][2D] -> x = Cn [4][2D r][2D c][KY] at prime i0 (Horner in 256)
//   SCATTER_ROWS y = Cn -> x = X: X[part][i][j][ii + D u] = Cn[i][i0 + j][part D + ii][u]
enum CcmmOp : int32_t {
    CCMM_GATHER = 0,
    CCMM_SCATTER = 1,
    CCMM_NTT = 2,
    CCMM_INTT = 3,
    CCMM_KS_DIGITS = 4,
    CCMM_KS_MAC = 5,
    CCMM_MOD_DOWN = 6,
    CCMM_RING_IN = 7,
    CCMM_COMPOSE_A = 8,
    CCMM_RING_OUT = 9,
    CCMM_MONO = 10,
    CCMM_BITREV = 11,
    CCMM_TWEAK = 12,
    CCMM_SCRAMBLE = 13,
    CCMM_KS_APPLY = 14,
    CCMM_ADD = 15,
    CCMM_GATHER_U = 16,
    CCMM_DIGITS = 17,
    CCMM_RECOMBINE = 18,
    CCMM_SCATTER_ROWS = 19,
};
struct CcmmArgs {
    int32_t op, n, count, i0, i1, i2, i3, i4;
    void* x;
    const void* y;
    const void* z;
    const void* tw;
    PolyRef ct;
    uint32_t p[8];
    uint64_t mu[8];
    uint32_t c[16];
    int64_t s[4];
};
static_assert(sizeof(CcmmArgs) == 280, "CcmmArgs layout");

// ------------------------------------------------------------------------------------------------ small_int1 (ABI 8)
// The exact post-raise scaling from ONE source prime (composite degree 1: the n64 chain; any word). Per
// coefficient of rows [0, nrows) of K ciphertexts (nparts polynomials from part 0, coefficient form): x = row 0
// centred against q0 (|x| <= q0/2), q = round(|x| mc / 2^sh) (half up; c = mc 2^-sh is the double's mantissa and
// exponent, so the product is exact), row t := sign(x) q mod q_t. In place: row 0 is read before any row is written.
// The host guarantees 1 <= sh <= 127 and q < 2^64 (bts/raise.rs RaisePlan::new).
struct SmallInt1Args {
    void* x;
    const PrimeConst* pr;
    uint64_t ct_stride, part_stride;
    uint64_t mc;
    int32_t sh, nrows, K, nparts;
};
static_assert(sizeof(SmallInt1Args) == 56, "SmallInt1Args layout");

// ------------------------------------------------------------------------------------------------ coeff_sum (ABI 9)
// Streamed-weight inner sum (coeffrows.cuh): out (+)= sum_{r < n} ct[r] * pt_r over Q limbs 0..L-1 (prime
// id = limb), pt_r[k] = round(rows[r][k] * scale) (half to even, |.| clamped at 2^62 past the int64 range and counted
// in *overflow when non-null) mod p, NTT'd. `ct[r]`, `out`: part 0 row 0 of a ciphertext buffer, part 1 at
// +part_stride words, row l at +l*N. `rows[r]`: N float32 coefficients (device memory). `scratch`: n*L*N words.
// accumulate != 0 adds onto out (a sum longer than PC_COEFF_MAXT runs in chunks). n in [1, MAXT], L in [1, MAXL].
#define PC_COEFF_MAXT 32
#define PC_COEFF_MAXL 64
struct CoeffArgs {
    void* scratch;
    void* out;
    const void* fw;
    const void* fws;
    const PrimeConst* pr;
    unsigned long long* overflow;
    uint64_t part_stride;
    double scale;
    int32_t n, L, accumulate, pad;
    const float* rows[PC_COEFF_MAXT];
    const void* ct[PC_COEFF_MAXT];
};
static_assert(sizeof(CoeffArgs) == 80 + 16 * PC_COEFF_MAXT, "CoeffArgs layout");

// ------------------------------------------------------------------------------------------------ tfhe (ABI 11)
// CKKS <-> TFHE scheme switching (pc-tfhe is the bit-exact host reference). One launcher, op-coded. Blind rotation
// runs over the GLWE ring of degree n (power of two, <= 2048) at the bootstrapping prime q < 2^62 (u64, Shoup products):
//   PBS_MS       y = u32 LWEs [count][i1 + 1] mod c[0] (a, then b) -> x = u32 [count][i1 + 1] in [0, 2n):
//                round(v 2n / c[0]) = (v m + 2^63) >> 64 mod 2n  (m = floor(2^64 2n / c[0]))
//   PBS_INIT     y = test vector u64 [n], z = PBS_MS output -> x = acc u64 [count][2][n]: X^-b~ tv, 0
//   PBS_STEP     key i0 of i1: acc += GGSW_i0 [x] (acc X^-a~_i0 - acc); y = bsk u64 [i1][2 c][l][2 part][n] (NTT form),
//                w = its Shoup companions, z = PBS_MS output, tw = [psi n | ipsi n], tws = their Shoup companions;
//                gadget: c[0] = base log, c[1] = levels, m = floor(2^(64 + levels base) / q); c[2] = n^-1, c[3] its Shoup
//   PBS_EXTRACT  y = acc -> x = u32 LWEs [count][n + 1] mod c[1]: sample extract of coefficient 0, b + c[0] (offset),
//                then round(v c[1] / q) = (v m + 2^63) >> 64 mod c[1]  (m = floor(2^64 c[1] / q))
//   KS_DIGITS    y = u32 LWEs [count][i1 + 1] mod q -> x = int8 [count][i1 levels]: the balanced digits (base 2^c[0],
//                c[1] levels, approximate, m = floor(2^(64 + levels base) / q)) of a_i at i levels + t
//   KS_RECOMBINE y = int32 GEMM planes [c[2]][i2 + 1][count] (plane p, output column o, LWE row), z = the input LWEs
//                (i1 + 1 words each) -> x = u32 LWEs [count][i2 + 1]: sum_p 256^p G_p (+ the input b on column i2) mod q
#define PC_TFHE_PBS_MS 0
#define PC_TFHE_PBS_INIT 1
#define PC_TFHE_PBS_STEP 2
#define PC_TFHE_PBS_EXTRACT 3
#define PC_TFHE_KS_DIGITS 4
#define PC_TFHE_KS_RECOMBINE 5
struct TfheArgs {
    int32_t op, n, count, i0, i1, i2, i3, i4;
    void* x;
    const void* y;
    const void* z;
    const void* w;
    const void* tw;
    const void* tws;
    uint64_t q, m;
    uint64_t c[8];
};
static_assert(sizeof(TfheArgs) == 32 + 6 * 8 + 16 + 64, "TfheArgs layout");

// ------------------------------------------------------------------------------------------------ KernelTable
// One table per (W, MM) instantiation, exported as `extern "C" const KernelTable* pc_table_<w>_<mm>()`.
// Every launcher: (args, stream) -> cudaError_t of the launch (0 = ok; -1/-2 = argument out of the kernel's range).
struct KernelTable {
    uint32_t abi;         // PC_KERNELS_ABI
    uint32_t word_bytes;  // sizeof(W)
    uint32_t mm;          // 0 Barrett64, 1 Barrett32, 2 Montgomery
    uint32_t log_n;       // PC_LOGN
    int (*ntt_fwd)(const NttArgs*, cudaStream_t);
    int (*ntt_inv)(const NttArgs*, cudaStream_t);
    int (*bconv)(const BconvArgs*, cudaStream_t);
    int (*ew_add)(const EwArgs*, cudaStream_t);
    int (*ew_scale)(const EwArgs*, cudaStream_t);
    int (*ew_sub_mul)(const EwArgs*, cudaStream_t);
    int (*ew_copy)(const EwArgs*, cudaStream_t);
    int (*pt_dot)(const PtDotArgs*, cudaStream_t);
    int (*ks_dot)(const KsArgs*, cudaStream_t);
    int (*ks_dot_regen)(const KsRegenArgs*, cudaStream_t);
    int (*ksk_expand)(const KskExpandArgs*, cudaStream_t);
    int (*cts_fused)(const CtsArgs*, cudaStream_t);
    int (*tensor)(const TensorArgs*, cudaStream_t);
    int (*md_epilogue)(const MdEpiArgs*, cudaStream_t);
    int (*wsum)(const WsumArgs*, cudaStream_t);
    int (*add_const)(const AddConstArgs*, cudaStream_t);
    int (*mul_table)(const MulTableArgs*, cudaStream_t);
    int (*rescale_coef)(const RescaleArgs*, cudaStream_t);
    int (*rescale_combine)(const RescaleCombineArgs*, cudaStream_t);
    int (*automorph)(const AutomorphArgs*, cudaStream_t);
    int (*encode)(const EncodeArgs*, cudaStream_t);
    int (*encrypt)(const EncryptArgs*, cudaStream_t);
    int (*ksk_gen)(const KskGenArgs*, cudaStream_t);
    int (*ksk_unpack)(const KskUnpackArgs*, cudaStream_t);
    int (*mod_raise)(const ModRaiseArgs*, cudaStream_t);
    int (*small_int)(const SmallIntArgs*, cudaStream_t);
    int (*spru_extract)(const SpruExtractArgs*, cudaStream_t);
    int (*spru_fft_pass)(const SpruFftArgs*, cudaStream_t);
    int (*spru_scatter)(const SpruScatterArgs*, cudaStream_t);
    int (*cts_fused_regen)(const CtsRegenArgs*, cudaStream_t);
    int (*ks_dot_key)(const KsKeyArgs*, cudaStream_t);
    int (*pcmm)(const PcmmArgs*, cudaStream_t);
    int (*ccmm)(const CcmmArgs*, cudaStream_t);
    int (*wsum_k)(const WsumKArgs*, cudaStream_t);
    int (*rescale_combine_to)(const RescaleCombineToArgs*, cudaStream_t);
    int (*md_epilogue_to)(const MdEpiToArgs*, cudaStream_t);
    int (*small_int1)(const SmallInt1Args*, cudaStream_t);
    int (*coeff_sum)(const CoeffArgs*, cudaStream_t);
    int (*tfhe)(const TfheArgs*, cudaStream_t);
};
static_assert(sizeof(KernelTable) == 16 + 39 * 8, "KernelTable layout");

// ------------------------------------------------------------------------------------------------ ring tables
// One table per (W, MM, log_n), exported as `extern "C" const RingTable* pc_ring_<w>_<log_n>()`: the transforms
// and row kernels that take the ring size from the instantiation (NttArgs rows at roff[i] * 2^log_n, twiddle tables
// [prime id][2^log_n]; EwArgs/BconvArgs rows of 2^log_n words).
//   ntt_fwd / ntt_inv: the fastest transform of the size; ntt_fwd_g / ntt_inv_g: always the generic shared-memory one.
struct RingTable {
    uint32_t abi;  // PC_RING_ABI
    uint32_t word_bytes;
    uint32_t mm;
    uint32_t log_n;
    int (*ntt_fwd)(const NttArgs*, cudaStream_t);
    int (*ntt_inv)(const NttArgs*, cudaStream_t);
    int (*ntt_fwd_g)(const NttArgs*, cudaStream_t);
    int (*ntt_inv_g)(const NttArgs*, cudaStream_t);
    int (*bconv)(const BconvArgs*, cudaStream_t);
    int (*ew_add)(const EwArgs*, cudaStream_t);
    int (*ew_scale)(const EwArgs*, cudaStream_t);
    int (*ew_sub_mul)(const EwArgs*, cudaStream_t);
    int (*ew_copy)(const EwArgs*, cudaStream_t);
};
static_assert(sizeof(RingTable) == 16 + 9 * 8, "RingTable layout");
