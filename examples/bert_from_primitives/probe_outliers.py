"""Which (token, channel) entries carry BERT's GELU / LN2 outliers at T=128 (diagnostic)."""
import collections
import sys

import numpy as np

from . import data as D


def main():
    st, cl = D.store(), D.client()
    tok = D.tokenizer()
    texts = [s for s, _ in D.sentences(24)] + [D.TEXT128.read_text().strip()]
    big_ch = collections.Counter()
    big_tok = collections.Counter()
    tot = 0
    nbig = 0
    ln_tok = collections.Counter()
    for text in texts:
        ids, n = D.encode(text, 128)
        stats = {}
        D.exact_blocks(st, D.embed(ids, cl), n, stats=stats)
        words = tok.convert_ids_to_tokens(ids[:n])
        for b in range(12):
            u = stats[("gelu", b)][0][:n]
            tot += u.size
            r, c = np.nonzero(np.abs(u) > 24)
            nbig += len(r)
            for ri, ci in zip(r, c):
                big_ch[(b, int(ci))] += 1
                big_tok[words[ri]] += 1
            y = stats[("ln2", b)][0][:n]
            v = ((y - y.mean(-1, keepdims=True)) ** 2).mean(-1)
            for ri in np.nonzero(v > 20)[0]:
                ln_tok[(b, words[ri], int(ri == n - 1), int(ri == 0))] += 1
    print(f"gelu |u|>24: {nbig} of {tot} entries ({nbig / tot:.2e})")
    print("channels:", big_ch.most_common(20))
    print("tokens:", big_tok.most_common(20))
    print("ln2 var>20 (blk, token, is_last, is_first):", ln_tok.most_common(30))
    return 0


if __name__ == "__main__":
    sys.exit(main())
