"""Per-layer max |q|, |k|, |v|, |score|, |attention output| of BERT at T=128 (the CCMM operand / shift bounds; diagnostic)."""
import sys

import numpy as np

from . import data as D


def main():
    from examples.gpt2_from_primitives import weights as W
    st, cl = D.store(), D.client()
    texts = [s for s, _ in D.sentences(32)] + [D.TEXT128.read_text().strip()]
    mx = np.zeros((12, 5))
    for text in texts:
        ids, n = D.encode(text, 128)
        x = D.embed(ids, cl)
        t = st.tensor
        for b in range(12):
            nm = W.block_names(b)
            qkv = x @ t(nm["qkv_w"]).T + t(nm["qkv_b"])
            q, k, v = np.split(qkv, 3, axis=1)
            o = np.zeros_like(q)
            for hh in range(12):
                sl = slice(hh * 64, hh * 64 + 64)
                s = q[:, sl] @ k[:n, sl].T / 8.0
                p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                o[:, sl] = p @ v[:n, sl]
                mx[b, 3] = max(mx[b, 3], np.abs(s).max())
            mx[b, :3] = np.maximum(mx[b, :3], [np.abs(q).max(), np.abs(k).max(), np.abs(v).max()])
            mx[b, 4] = max(mx[b, 4], np.abs(o).max())
            x = D.exact_blocks(st, x, n, 1)[0] if False else None
            break
        hs = D.exact_blocks(st, D.embed(ids, cl), n)
        for b in range(1, 12):
            x = hs[b - 1]
            nm = W.block_names(b)
            q, k, v = np.split(x @ t(nm["qkv_w"]).T + t(nm["qkv_b"]), 3, axis=1)
            o = np.zeros_like(q)
            for hh in range(12):
                sl = slice(hh * 64, hh * 64 + 64)
                s = q[:, sl] @ k[:n, sl].T / 8.0
                p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                o[:, sl] = p @ v[:n, sl]
                mx[b, 3] = max(mx[b, 3], np.abs(s).max())
            mx[b, :3] = np.maximum(mx[b, :3], [np.abs(q).max(), np.abs(k).max(), np.abs(v).max()])
            mx[b, 4] = max(mx[b, 4], np.abs(o).max())
    print("layer max|q| max|k| max|v| max|score| max|o|")
    for b in range(12):
        print(b, " ".join(f"{v:7.2f}" for v in mx[b]), flush=True)
    print("all ", " ".join(f"{v:7.2f}" for v in mx.max(0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
