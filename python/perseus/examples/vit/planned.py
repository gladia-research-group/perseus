"""ViT-B/16 PLANNED on the new runtime: `examples.bert.planned.EncoderPlannedPrefill` on `VitPrefill` (patch embedding
as a PCMM in block 0, two-page bidirectional CCMM attention). One recorded stream serves every image: the op stream
depends on the token count only, the image enters as block 0's encrypted patch rows."""
from __future__ import annotations

import os

import numpy as np

from perseus.examples.bert.planned import EncoderPlannedPrefill


class VitPlannedPrefill(EncoderPlannedPrefill):
    def _rows_in(self, rows):
        # VitPrefill.prefill: rows padded to whole 128-token pages with copies of the last ones (VIT_PAD)
        rows = np.asarray(rows, dtype=np.float64)
        n = len(rows)
        full = -(-n // 128) * 128
        if os.environ.get("VIT_PAD", "1") not in ("", "0") and full > n:
            rows = np.vstack([rows, rows[n - (full - n):]])
        return rows

    def linears(self):
        out = super().linears()
        for lin in self.model.extra_linears():
            out[lin.name] = lin
        return out

    def hidden(self, b=None):
        """(n_keys, dim) rows of block b's output (default: the last), residual scale undone by the caller."""
        return super().hidden(b)[:self.ch.n_keys]
