"""Where BERT's GELU / LayerNorm extremes sit at T=128: real tokens vs padding, per position class (diagnostic)."""
import sys

import numpy as np

from . import data as D


def main():
    st, cl = D.store(), D.client()
    texts = [s for s, _ in D.sentences(24)] + [D.TEXT128.read_text().strip()]
    for pad_mode in ("pad", "sep"):
        g_real, g_pad, ln_real, ln_pad = {}, {}, {}, {}
        for text in texts:
            ids, n = D.encode(text, 128)
            if pad_mode == "sep" and n < 128:
                ids[n:] = ids[n - 1]          # pad with the [SEP] token instead of [PAD]
            stats = {}
            D.exact_blocks(st, D.embed(ids, cl), n, stats=stats)
            for b in range(12):
                u = stats[("gelu", b)][0]
                g_real[b] = max(g_real.get(b, 0), float(np.abs(u[:n]).max()))
                if n < 128:
                    g_pad[b] = max(g_pad.get(b, 0), float(np.abs(u[n:]).max()))
                for k in ("ln1", "ln2"):
                    y = stats[(k, b)][0]
                    v = ((y - y.mean(-1, keepdims=True)) ** 2).mean(-1)
                    r = ln_real.setdefault((k, b), [np.inf, 0])
                    r[0], r[1] = min(r[0], v[:n].min()), max(r[1], v[:n].max())
                    if n < 128:
                        p = ln_pad.setdefault((k, b), [np.inf, 0])
                        p[0], p[1] = min(p[0], v[n:].min()), max(p[1], v[n:].max())
        print(f"== padding with {pad_mode}")
        for b in range(12):
            print(f"blk {b:2d} gelu max|u| real {g_real[b]:6.2f} pad {g_pad.get(b, 0):6.2f} | "
                  + " ".join(f"{k} var real [{ln_real[(k, b)][0]:.3g},{ln_real[(k, b)][1]:.3g}] "
                             f"pad [{ln_pad.get((k, b), [0, 0])[0]:.3g},{ln_pad.get((k, b), [0, 0])[1]:.3g}]"
                             for k in ("ln1", "ln2")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
