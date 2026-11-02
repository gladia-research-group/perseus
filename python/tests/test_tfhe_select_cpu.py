"""TFHE-sign selection (impl/tfhe_select.py) on the plaintext mirror: the GPT-2 head's argmax and top-k sampling
(examples/gpt2/tfhe_head.py) on synthetic logits in the LM head's tile layout."""
import numpy as np
import pytest

from perseus.examples.gpt2.tfhe_head import VOCAB, Head, _check_onehot
from perseus.impl import tfhe, tfhe_select as sel
from perseus.impl.ops import NumpyOps


def _logits(seed):
    """A GPT-2-like spread (~30 logits) with the top two >= 0.5 apart (no near-tie)."""
    rng = np.random.default_rng(seed)
    lg = rng.standard_normal(VOCAB) * 3.0 - 100.0
    i = np.argsort(-lg)[:2]
    lg[i[0]] = lg[i[1]] + 0.5 + rng.uniform()
    return lg


def test_msign_levels():
    """[d >= 0] at every scale level: wrap-free on |d| < R, the fine levels deciding inside the coarse band."""
    n, R = 64, 128.0
    d = np.zeros(n)
    heads = np.arange(0, n, 8)
    d[heads] = [-100.0, -3.0, -0.2, -0.01, 0.01, 0.2, 3.0, 100.0]
    for levels in (1, 2, 3):
        s = np.real(sel.msign(NumpyOps(), d, n, heads, R, levels, f"t{levels}"))
        want = (d[heads] >= 0).astype(float)
        if levels == 1:     # one level reads |d| s_0 on a grid of ~0.07 here: the +-0.01 pair is below it
            keep = np.abs(d[heads]) > 0.1
            np.testing.assert_allclose(s[heads][keep], want[keep], atol=1e-12)
        else:
            np.testing.assert_allclose(s[heads], want, atol=1e-12)
        assert not np.any(np.delete(s, heads))


@pytest.mark.parametrize("seed", range(3))
def test_argmax_onehot(seed):
    lg = _logits(seed)
    h = Head("argmax")
    v, _ = h.decode(h.stage(NumpyOps(), h.encode(lg)))
    assert _check_onehot(v) and int(np.argmax(v)) == int(np.argmax(lg))


@pytest.mark.parametrize("native", ["0", "plain", "exact", "draw", "arity3", "arity5"])
def test_topk_sample(native, monkeypatch):
    """The member set is the top k of the bucket maxima; the sample is the Gumbel-max over it (also with the
    maximum, threshold search and mask as native programs: both count kinds, the draw, k-ary searches)."""
    if native != "0":
        monkeypatch.setenv("PF_SELECT_NATIVE", "1")
        monkeypatch.setenv("PF_SELECT_NAT_COUNT", "plain" if native == "plain" else "exact")
        monkeypatch.setenv("PF_SELECT_NAT_DRAW", "1" if native == "draw" else "0")
        monkeypatch.setenv("PF_SELECT_NAT_ARITY", native[5:] if native.startswith("arity") else "2")
    lg = _logits(7)
    h = Head("topk", k=40)
    g = np.random.default_rng(3).gumbel(size=h.L)
    v, mask = h.decode(h.stage(NumpyOps(), h.encode(lg, g)))
    bt = h.buckets(lg)
    members = bt[np.rint(mask) > 0.5]
    top = bt[np.argsort(-lg[bt])[:40]]
    # the k-ary searches' mask sits a margin (0.1 / 0.2 logits) below their threshold: a few more members
    assert 40 <= len(members) <= (52 if native.startswith("arity") else 44) and set(top) <= set(members)
    gb = np.zeros(VOCAB)
    gb[bt] = g
    assert _check_onehot(v) and int(np.argmax(v)) == int(members[np.argmax(lg[members] + gb[members])])


def test_minp_sample():
    """min-p: the members are the bucket maxima within ln(p) of the largest."""
    lg = _logits(5)
    h = Head("minp", minp=0.05)
    g = np.random.default_rng(4).gumbel(size=h.L)
    v, mask = h.decode(h.stage(NumpyOps(), h.encode(lg, g)))
    bt = h.buckets(lg)
    members = set(bt[np.rint(mask) > 0.5].tolist())
    want = set(bt[lg[bt] >= lg.max() + np.log(0.05)].tolist())
    assert members == want and _check_onehot(v) and int(np.argmax(v)) in members


@pytest.mark.parametrize("k", ["1", "4"])
def test_topp_sample(k, monkeypatch):
    """top-p: the nucleus of the bucket maxima is inside the members (their margin adds a few within it below)."""
    monkeypatch.setenv("PF_SELECT_NATIVE", "1")
    monkeypatch.setenv("PF_SELECT_TOPP_K", k)
    lg = _logits(9)
    h = Head("topp", topp=0.9)
    g = np.random.default_rng(2).gumbel(size=h.L)
    v, mask = h.decode(h.stage(NumpyOps(), h.encode(lg, g)))
    bt = h.buckets(lg)
    members = set(bt[np.rint(mask) > 0.5].tolist())
    o = np.argsort(-lg[bt])
    c = np.cumsum(np.exp(lg[bt][o] - lg[bt][o[0]]))
    nuc = set(bt[o[:int(np.searchsorted(c / c[-1], 0.9)) + 1]].tolist())
    below = lg[bt][o][len(nuc) - 1] - 0.35 * int(k) - 0.2
    assert nuc <= members and all(lg[t] >= below for t in members)
    assert _check_onehot(v) and int(np.argmax(v)) in members


def test_ckks_sign_baseline(monkeypatch):
    """PF_SELECT_SIGN=ckks: the same tournament on the composite polynomial sign."""
    monkeypatch.setenv("PF_SELECT_SIGN", "ckks:0.03")
    lg = _logits(11)
    h = Head("argmax")
    v, _ = h.decode(h.stage(NumpyOps(), h.encode(lg)))
    assert _check_onehot(v) and int(np.argmax(v)) == int(np.argmax(lg))


def test_slot_budget():
    """The slot sets a head registers fit the session's 2^16 positions (equal sets share one registration)."""
    for mode in ("argmax", "topk"):
        tfhe.SLOTS.clear()
        Head(mode).rotations()
        assert sum(len(s) for s, _c in tfhe.SLOTS.values()) <= 1 << 16
