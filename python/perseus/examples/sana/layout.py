"""The SANA transformer's ciphertext layout: perseus.impl.blocked's channel-blocked prompt layout on the latent tokens.
Ciphertext [g][k] holds channels [k NCH, (k + 1) NCH) of token slots [128 g, 128 g + 128): channel r of block k at
slot r LANES + lane, token slot 128 g + lane in the real part, 128 g + 64 + lane in the imaginary part. The latent image
(IMG x IMG: 24 at 768 px) is row-major on rows of GRID = 32 slots (the last GRID - IMG empty), so a ciphertext's real
part holds image rows 4 g and 4 g + 1 (lanes 0-31 and 32-63) and its imaginary part rows 4 g + 2 and 4 g + 3: every
lane has one image column (col_of_lane), a vertical neighbour is 32 lanes away.

SANA_TIGHT_ROWS=1 (read here, at import: set it before any sana import; a plan is cut for one of the two): the image is
row-major with no gap between its rows (GRID = IMG), token slot p on row p // IMG, column p % IMG, and the slots past
the image (TOK - IMG^2: 64 at 768 px) hold copies of the first tokens. At 768 px that is G = 5 chunk pairs for the 576
tokens instead of 6 (12% fewer refreshes, the same values); the two halves of a ciphertext then sit at different image
columns (64 mod 24 = 16), which the depthwise convolution's masks carry (kernels.depthwise3x3)."""
from __future__ import annotations

import os

import numpy as np

from .hf import RES

N, NCH, LANES = 1 << 15, 512, 64
IMG = RES // 32                           # the latent image's side (DC-AE's 32x: 24 at 768 px)
TIGHT = os.environ.get("SANA_TIGHT_ROWS", "0") not in ("", "0")
GRID = IMG if TIGHT else 32               # token slots per image row
G = -(-IMG * GRID // (2 * LANES))         # chunk pairs (ciphertexts per channel block)
TOK = G * 2 * LANES                       # token slots (IMG x IMG tokens; the others hold copies)
VALID = (np.arange(TOK) % GRID < IMG) & (np.arange(TOK) < IMG * GRID)     # the slots holding a token
D = 1152


def blocks_of(C: int) -> int:
    return -(-C // NCH)


def geometry():
    """(Dims, Blocks, Chunks) of the residual stream for perseus.impl.blocked (norm_blocked, shift_blocked):
    hid 1536 = 3 blocks over the 1152 channels. The head fields only satisfy Blocks' check (the SANA attention
    kernels are this package's)."""
    from perseus.impl.blocked import Blocks
    from perseus.impl.layout import Dims
    from perseus.impl.prefill import Chunks
    d = Dims(N, 3 * NCH, D, 24, 18, 12 * NCH, 5760)
    B = Blocks(d, NCH)
    return d, B, Chunks(TOK, B.t, bidir=True)


def encode(rows, nblk: int | None = None):
    """(T, C) rows -> slot vectors [g][k] (complex128, N); T a multiple of 2 LANES (TOK for the latent)."""
    rows = np.asarray(rows, dtype=np.float64)
    T, C = rows.shape
    g_n = T // (2 * LANES)
    assert T == g_n * 2 * LANES
    nblk = nblk or blocks_of(C)
    full = np.zeros((T, nblk * NCH))
    full[:, :C] = rows
    x = full.reshape(g_n, 2, LANES, nblk, NCH)                       # g, half, lane, k, r
    z = x[:, 0] + 1j * x[:, 1]                                        # g, lane, k, r
    return [[np.ascontiguousarray(z[g, :, k, :].T.reshape(N)) for k in range(nblk)] for g in range(g_n)]


def decode(cts, C: int | None = None):
    """Slot vectors [g][k] (complex) -> (2 LANES len(cts), C) rows."""
    nblk = len(cts[0])
    z = np.stack([np.stack([np.asarray(c).reshape(NCH, LANES).T for c in row], 1) for row in cts])  # g, lane, k, r
    x = np.stack([z.real, z.imag], 1).reshape(len(cts) * 2 * LANES, nblk * NCH)
    return x if C is None else x[:, :C]


def lane_vec(per_lane, per_channel=None):
    """A slot vector from a (LANES,) lane pattern times an optional (NCH,) channel pattern."""
    ch = np.ones(NCH) if per_channel is None else np.asarray(per_channel)
    return np.outer(ch, np.asarray(per_lane)).reshape(N)


def col_of_lane():
    return np.arange(LANES) % GRID


def slot_of(g, half):
    """The token slots of chunk pair g's half (0: real, 1: imaginary), per lane."""
    return 2 * LANES * g + LANES * half + np.arange(LANES)


def to_slots(img):
    """(C, IMG, IMG) -> (TOK, C) slot rows. The empty slots of a row repeat its first GRID - IMG columns: tokens like
    the image's (zeros would sit outside every approximation's range); no real token reads them (the convolution's
    column masks, the attention keys' rk)."""
    img = np.asarray(img, dtype=np.float64)
    C = img.shape[0]
    if TIGHT:                                                         # the slots past the image: its first tokens
        flat = img.reshape(C, IMG * IMG)
        return np.concatenate([flat, flat[:, :TOK - IMG * IMG]], axis=1).T
    full = np.concatenate([img, img[:, :, :GRID - IMG]], axis=2)      # (C, IMG, GRID)
    return full.reshape(C, TOK).T


def tokens(rows):
    """(TOK, C) slot rows -> (IMG * IMG, C) token rows (row-major)."""
    return np.asarray(rows)[VALID]


def from_slots(rows):
    """(TOK, C) slot rows -> (C, IMG, IMG)."""
    t = tokens(rows)
    return t.T.reshape(t.shape[1], IMG, IMG)
