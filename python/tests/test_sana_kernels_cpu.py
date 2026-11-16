"""SANA kernels (examples/sana/kernels.py) on the plaintext mirror against float64 torch: the layout round trip, the
3 x 3 depthwise convolution."""
import numpy as np
import pytest

from perseus.examples.sana import kernels as KN
from perseus.examples.sana import layout as L

torch = pytest.importorskip("torch")


def _rt():
    from perseus.examples.sana.mirror_run import mirror_rt
    return mirror_rt()


def test_layout_round_trip():
    x = np.random.default_rng(0).standard_normal((L.TOK, 1300))
    np.testing.assert_array_equal(L.decode(L.encode(x), 1300), x)
    img = np.random.default_rng(0).standard_normal((32, L.IMG, L.IMG))
    np.testing.assert_array_equal(L.from_slots(L.to_slots(img)), img)


def test_depthwise3x3():
    rng = np.random.default_rng(1)
    C = 700
    img = rng.standard_normal((C, L.IMG, L.IMG))
    w = rng.standard_normal((C, 3, 3))
    b = rng.standard_normal(C)
    got = L.decode(KN.depthwise3x3(_rt(), L.encode(L.to_slots(img)), w, b, "t"), C)
    ref = torch.nn.functional.conv2d(torch.as_tensor(img[None]), torch.as_tensor(w[:, None]), torch.as_tensor(b),
                                     padding=1, groups=C)
    np.testing.assert_allclose(L.tokens(got), ref.reshape(C, L.IMG * L.IMG).T.numpy(), atol=1e-10)


class ExactNL:
    """The kernels' nonlinear hooks (nl.PolyNL's), exact: the unit ReLU forms read their input through 1 / 2 (the
    real halves' 2 Re), the paired softmax's doubled sums through recip's dscale."""
    rounds = 3

    def relu_u(self, rt, u, site, room=None):
        return np.maximum(0.5 * u.real, 0.0).astype(complex)

    def relu_u2(self, rt, ua, ub, site, room=None, mul=None, tail=0):
        ra, rb = self.relu_u(rt, ua, site), self.relu_u(rt, ub, site)
        return (ra, rb) if mul is None else (ra * mul[0], rb * mul[1])

    def inv_sqrt(self, rt, z, site):
        return (1.0 / np.sqrt(z.real)).astype(complex)

    def recip(self, rt, d, site, dscale=1.0):
        return (1.0 / d.real).astype(complex)

    def recip_cost(self, site):
        return None

    def exp(self, rt, x, site, folded=False, prescaled=False):
        assert not (folded or prescaled), "ExactNL has no exp domain to fold (PF_SM_FOLD off)"
        return np.exp(0.5 * x.real / 2 ** self.rounds).astype(complex)       # x = 2 s

    def exp_need(self, rt, x, site):
        return x

    def sm_rounds(self, site):
        return self.rounds

    def stat_scale(self, site):
        return 1.0

    def la_sigma(self, site):
        return 1.0


def _la_ref(q, k, v, wk, H=36, dh=32):
    """ref.linear_attention between the projections and to_out (q already times norm_q.weight); the keys and values
    the image's tokens (layout.VALID), every slot a query."""
    rk = 1.0 / np.sqrt((k * k).mean(-1, keepdims=True) + 1e-5)
    fq = np.maximum(q, 0).reshape(-1, H, dh)
    fk = np.maximum(k * rk * wk, 0).reshape(-1, H, dh)
    vv = v.reshape(-1, H, dh)
    S = np.einsum("tha,thb->hab", fk[L.VALID], vv[L.VALID])
    z = fk[L.VALID].sum(0)
    num = np.einsum("tha,hab->thb", fq, S)
    den = np.einsum("tha,ha->th", fq, z)[..., None]
    return (num / (den + 1e-15)).reshape(len(q), H * dh)


def test_linear_attention():
    rng = np.random.default_rng(2)
    T, D = L.TOK, 1152
    q, k, v = (rng.standard_normal((T, D)) for _ in range(3))
    wk = rng.uniform(0.5, 2.0, D)
    got = KN.linear_attention(_rt(), L.encode(q), L.encode(k * wk), L.encode(v), wk, ExactNL(), "la")
    np.testing.assert_allclose(L.decode(got, D), _la_ref(q, k, v, wk), atol=1e-9)


def test_cross_attention():
    nl = ExactNL()
    rng = np.random.default_rng(3)
    T, H, dh, n_key = L.TOK, 16, 72, 7
    q = rng.standard_normal((T, H * dh)) * 3
    wq = rng.uniform(0.5, 2.0, H * dh)
    kh = rng.standard_normal((n_key, H * dh))
    v = rng.standard_normal((n_key, H * dh))
    perm = KN.xhead_perm()
    xh = lambda r: np.bincount(perm, weights=r, minlength=4 * L.NCH)      # noqa: E731
    Kb, Vb = [], []
    for j in range(n_key):
        kr = KN.bcast_rows(xh(kh[j] / np.sqrt(dh)), 4)
        vr = KN.bcast_rows(xh(v[j]), 4)
        if j == 0:                                       # the public BOS row
            Kb.append([KN.Pt(x, ("bos", k)) for k, x in enumerate(kr)])
            Vb.append([KN.Pt(x, ("bos", k)) for k, x in enumerate(vr)])
        else:
            Kb.append([x.astype(complex) for x in kr])
            Vb.append([x.astype(complex) for x in vr])
    Q = L.encode(np.stack([xh(r) for r in q]), 4)
    got = KN.cross_attention(_rt(), Q, Kb, Vb, wq, nl, "xa")
    o = L.decode(got)[:, perm]
    qn = (q / wq) * (1.0 / np.sqrt(((q / wq) ** 2).mean(-1, keepdims=True) + 1e-5)) * wq
    s = np.einsum("thd,jhd->htj", qn.reshape(T, H, dh), kh.reshape(n_key, H, dh)) / np.sqrt(dh)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    ref = np.einsum("htj,jhd->thd", p, v.reshape(n_key, H, dh)).reshape(T, H * dh)
    np.testing.assert_allclose(o, ref, atol=1e-9)


def test_stage_levels():
    """The store's slot classes fit every stage's plaintexts (planned.stage_levels; a key takes the smallest free
    slot it fits)."""
    from perseus.examples.sana.planned import stage_levels
    sets = [{("a", 10), ("b", 20)}, {("c", 10), ("d", 10), ("e", 30)}]
    lv = stage_levels(sets, 40)
    assert sorted(lv) == [10, 10, 30]
    for s in sets:
        free = sorted(40 - x for x in lv)
        for _k, l in sorted(s, key=lambda kl: -(40 - kl[1])):
            i = next(i for i, r in enumerate(free) if r >= 40 - l)
            free.pop(i)
