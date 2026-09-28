"""Complex-lane pairing: two real payloads in one ciphertext (a + i b), so one product,
rotation or reduction serves both. Every function here is exact: its result equals the
real computation it replaces (tests/test_pairing_cpu.py, tests/gpu/test_pairing.py).

The decode's own uses (the K buckets, the V pair masks, the paired P.V, the output-packed
up- and down-projections, the fused K + iV linear) live in attention.py and linear.py; this
module holds the constructions in their general form, including the ones the decode does not
use: input pairing (slower than the real linear it replaces) and the extraction folded into a
contraction (the decode has no linear fed by two packed payloads):

  output pairing   two outputs that share an input: one complex plaintext W_a + i W_b, one
                   product, then one conjugate splits Re / Im (unpack_ri);
  input pairing    two inputs that share an output: x_a + i x_b against the CONJUGATED
                   weight W_a - i W_b, whose real part is x_a W_a + x_b W_b;
  paired products  Re((v_a + i v_b)(s_a - i s_b)) = v_a s_a + v_b s_b.

Weights are slot vectors (numpy); `key` names them for the runtime's plaintext cache
(ops.mult_pt), None encodes at every use."""
from __future__ import annotations

import numpy as np

from .poly import rotsum


# ── packing and extraction ───────────────────────────────────────────────────────────────

def pack_ri(ops, a, b):
    """a + i b through the cached complex constant i (one level, like a mask)."""
    return ops.add(a, ops.mult_const(b, 0.0, 1.0))


def pair_pack(ops, a, b):
    """a + i b through the monomial multiply (level-free)."""
    return ops.add(a, ops.mult_i(b))


def real_part(ops, P):
    """Re P = (P + conj P) / 2."""
    return ops.mult(ops.add(P, ops.conjugate(P)), 0.5)


def unpack_ri(ops, P, halved=False):
    """(Re P, Im P) from one conjugate: (P + cj) / 2 and i (cj - P) / 2. `halved`: the 1/2
    was folded into the weights that made P (pair_weights(..., halved=True)), so the unpack
    is (P + cj, i (cj - P)): no constant products, no level."""
    cj = ops.conjugate(P)
    re, im = ops.add(P, cj), ops.mult_i(ops.sub(cj, P))
    if halved:
        return re, im
    return ops.mult(re, 0.5), ops.mult(im, 0.5)


def _key(key, *idx):
    return None if key is None else ".".join([key] + [str(i) for i in idx])


def _dot(ops, xs, ws, key, *prefix):
    """sum_j xs[j] * ws[j] (plaintext products)."""
    acc = None
    for j, (x, w) in enumerate(zip(xs, ws)):
        t = ops.mult_pt(x, w, key=_key(key, *prefix, j))
        if acc is None:
            acc = t
        else:
            ops.inplace_add(acc, t)
    return acc


# ── diagonal (BSGS) linears ──────────────────────────────────────────────────────────────
# W has shape (G, B, N): G output blocks, B baby steps. rx[0] = x, rx[j] = rot(x, baby[j-1]);
# block g is sum_j rx[j] W[g, j], placed by rot(., giant[g-1]) for g > 0.

def baby_rotations(ops, x, baby):
    return [x] + ops.rotate_many(x, list(baby))


def _place(ops, y, giant, g):
    return y if g == 0 else ops.rotate(y, giant[g - 1])


def diag_linear(ops, x, W, baby, giant, key=None):
    """The real BSGS diagonal linear."""
    rx = baby_rotations(ops, x, baby)
    out = None
    for g in range(W.shape[0]):
        y = _place(ops, _dot(ops, rx, W[g], key, g), giant, g)
        if out is None:
            out = y
        else:
            ops.inplace_add(out, y)
    return out


def outpack_weights(W, halved=False):
    """Output blocks 2g' and 2g'+1 in one complex plaintext: (G/2, B, N); `halved` folds the
    unpack's 1/2 into them."""
    if W.shape[0] % 2:
        raise ValueError(f"output pairing needs an even block count, got {W.shape[0]}")
    return pair_weights(W[0::2], W[1::2], halved)


def diag_linear_outpack(ops, x, Wc, baby, giant, key=None, halved=False):
    """Output pairing: G/2 complex blocks, each unpacked by one conjugate into its two real
    blocks, which are then placed like the real linear's. Half the plaintext products; the
    placement rotations are NOT halved (the two halves land at different offsets)."""
    rx = baby_rotations(ops, x, baby)
    out = None
    for gp in range(Wc.shape[0]):
        re, im = unpack_ri(ops, _dot(ops, rx, Wc[gp], key, gp), halved)
        for g, y in ((2 * gp, re), (2 * gp + 1, im)):
            y = _place(ops, y, giant, g)
            if out is None:
                out = y
            else:
                ops.inplace_add(out, y)
    return out


def inpack_weights(W):
    """Baby steps 2j' and 2j'+1 as the conjugated plaintext W_a - i W_b: (G, B/2, N)."""
    if W.shape[1] % 2:
        raise ValueError(f"input pairing needs an even baby-step count, got {W.shape[1]}")
    return W[:, 0::2] - 1j * W[:, 1::2]


def diag_linear_inpack(ops, x, Wc, baby, giant, key=None):
    """Input pairing: the baby-step copies paired as rx[2j'] + i rx[2j'+1] (monomial, no
    level), contracted with the conjugated plaintexts, one real part at the end. Half the
    plaintext products; every baby-step rotation is still needed."""
    rx = baby_rotations(ops, x, baby)
    X = [pair_pack(ops, rx[2 * j], rx[2 * j + 1]) for j in range(len(rx) // 2)]
    out = None
    for g in range(Wc.shape[0]):
        y = _place(ops, _dot(ops, X, Wc[g], key, g), giant, g)
        if out is None:
            out = y
        else:
            ops.inplace_add(out, y)
    return real_part(ops, out)


# ── two outputs from one input (fused projections) ───────────────────────────────────────

def pair_weights(Wa, Wb, halved=False):
    """Two weights that share an input as one complex plaintext Wa + i Wb (times 1/2 when
    `halved`: the unpack's 1/2 folded in, see unpack_ri)."""
    return (0.5 if halved else 1.0) * (np.asarray(Wa) + 1j * np.asarray(Wb))


def fused_pair(ops, rs, Wc, key=None, halved=False):
    """(sum_j rs[j] Wa[j], sum_j rs[j] Wb[j]) for Wc = pair_weights(Wa, Wb, halved): one
    complex contraction and one unpack."""
    return unpack_ri(ops, _dot(ops, rs, Wc, key), halved)


# ── two inputs into one output (extraction folded into a contraction) ────────────────────

def conj_pair_weights(Wa, Wb):
    """Two weights that share an output as the conjugated plaintext Wa - i Wb."""
    return np.asarray(Wa) - 1j * np.asarray(Wb)


def contract_pairs(ops, xs, Wc, key=None):
    """sum_j (a_j Wa[j] + b_j Wb[j]) for packed xs[j] = a_j + i b_j and
    Wc = conj_pair_weights(Wa, Wb): the conjugated contraction and one real part, instead of
    extracting a_j, b_j first. Pack the inputs with pair_pack: a pack_ri input carries a
    pending rescale, which every product then pays."""
    return real_part(ops, _dot(ops, xs, Wc, key))


# ── paired products (P.V) ────────────────────────────────────────────────────────────────

def paired_products(ops, C, sa, sb):
    """sum_k (va_k sa_k + vb_k sb_k) from C_k = va_k + i vb_k: one product per pair
    against sa_k - i sb_k (monomial), one real part."""
    acc = None
    for c, a, b in zip(C, sa, sb):
        t = ops.mult(c, ops.sub(a, ops.mult_i(b)))
        if acc is None:
            acc = t
        else:
            ops.inplace_add(acc, t)
    return real_part(ops, acc)


# ── lane masks and reductions ────────────────────────────────────────────────────────────

def pair_masks(m_re, m_im):
    """One complex mask selecting lane m_re into the real axis and m_im into the imaginary."""
    return np.asarray(m_re) + 1j * np.asarray(m_im)


def paired_reduction(ops, q, kpack, stop):
    """(sum of q*ka, sum of q*kb) over the rotate-and-sum ladder 1..stop, for a cached
    kpack = ka + i kb (pack_ri / pair_pack): one product, one ladder, one unpack."""
    return unpack_ri(ops, rotsum(ops, ops.mult(q, kpack), 1, stop))
